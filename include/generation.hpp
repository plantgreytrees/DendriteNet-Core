#pragma once
/// generation.hpp — Next-token code generation for DendriteNet.
///
/// Adds autoregressive text/code generation by composing existing framework
/// primitives:
///
///   TokenVocab    — word-level vocabulary loaded from data/code_vocab.txt
///   EmbedTable    — learned token embedding table (wraps DenseLayer)
///   GenerativeLoop — teacher-forcing training + autoregressive inference
///
/// Architecture alignment:
///   - TaskContext (working memory) accumulates each generated token embedding,
///     functioning as the context window.  No new state machinery needed.
///   - DendriticLayer inside each branch takes (token_embed, context) naturally;
///     the context retrieval via task_context.get_context() IS the key-value cache.
///   - EarlyExitClassifier provides entropy-based confidence for sampling control.
///   - Morality layer already runs on every infer() call — generation is safe.
///
/// See examples/code_generator.cpp for usage.

#include "layer.hpp"
#include "tensor.hpp"
#include "checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dendrite {

// ---------------------------------------------------------------------------
// TokenVocab — word-level vocabulary
// ---------------------------------------------------------------------------

/// Lightweight word-level tokenizer and vocabulary.
/// Vocabulary file format: one token per line, line number = token ID.
/// Special tokens <PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3 must occupy lines 0-3.
struct TokenVocab {
    static constexpr int PAD_ID = 0;
    static constexpr int BOS_ID = 1;
    static constexpr int EOS_ID = 2;
    static constexpr int UNK_ID = 3;

    std::vector<std::string>              id2tok;
    std::unordered_map<std::string, int>  tok2id;

    /// Load vocabulary from a text file (one token per line).
    explicit TokenVocab(const std::string& vocab_path) {
        std::ifstream f(vocab_path);
        if (!f.is_open())
            throw std::runtime_error("TokenVocab: cannot open '" + vocab_path +
                                     "'. Run: python tools/prepare_code_dataset.py");
        std::string line;
        int idx = 0;
        while (std::getline(f, line)) {
            // Strip trailing \r on Windows
            if (!line.empty() && line.back() == '\r') line.pop_back();
            id2tok.push_back(line);
            tok2id[line] = idx++;
        }
    }

    size_t size() const { return id2tok.size(); }

    /// Convert text to token ID list using the same rules as the Python script.
    /// Splits on non-alphanumeric boundaries, lowercases.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::string cur;
        auto flush = [&]() {
            if (cur.empty()) return;
            auto it = tok2id.find(cur);
            ids.push_back(it != tok2id.end() ? it->second : UNK_ID);
            cur.clear();
        };
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else {
                flush();
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    // Single punctuation token
                    std::string punct(1, c);
                    auto it = tok2id.find(punct);
                    ids.push_back(it != tok2id.end() ? it->second : UNK_ID);
                }
            }
        }
        flush();
        return ids;
    }

    /// Convert token IDs back to a space-joined string.
    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (size_t i = 0; i < ids.size(); i++) {
            if (ids[i] == BOS_ID || ids[i] == PAD_ID) continue;
            if (ids[i] == EOS_ID) break;
            if (!out.empty()) out += ' ';
            int safe_id = (ids[i] >= 0 && ids[i] < static_cast<int>(id2tok.size()))
                          ? ids[i] : UNK_ID;
            out += id2tok[static_cast<size_t>(safe_id)];
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// EmbedTable — learned token embedding table
// ---------------------------------------------------------------------------

/// Wraps a DenseLayer as a token embedding lookup table.
/// The DenseLayer weights matrix W[embed_dim × vocab_size] stores embeddings
/// as columns.  lookup(id) extracts the id-th column.
struct EmbedTable {
    size_t     vocab_size_;
    size_t     embed_dim_;
    DenseLayer layer_;   // shape: embed_dim × vocab_size (one embedding per column)
    size_t     adam_step_ = 0;

    EmbedTable(size_t vocab_size, size_t embed_dim, std::mt19937& rng)
        : vocab_size_(vocab_size), embed_dim_(embed_dim),
          layer_(vocab_size, embed_dim, Activation::NONE, rng) {}

    /// Load initial embeddings from the binary file produced by
    /// tools/prepare_code_dataset.py.
    /// File format: int32 vocab_size, int32 embed_dim, then
    ///              vocab_size × embed_dim float32 (row-major, one row per token).
    void load_init(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f)
            throw std::runtime_error("EmbedTable: cannot open '" + path + "'");
        int vs = 0, ed = 0;
        std::fread(&vs, sizeof(int), 1, f);
        std::fread(&ed, sizeof(int), 1, f);
        if (static_cast<size_t>(vs) != vocab_size_ ||
            static_cast<size_t>(ed) != embed_dim_) {
            std::fclose(f);
            throw std::runtime_error("EmbedTable: dimension mismatch in '" + path + "'");
        }
        // File is row-major [vocab_size × embed_dim].
        // layer_.weights is [embed_dim × vocab_size] (out × in), so fill column by column.
        std::vector<float> row(embed_dim_);
        for (size_t v = 0; v < vocab_size_; v++) {
            std::fread(row.data(), sizeof(float), embed_dim_, f);
            for (size_t d = 0; d < embed_dim_; d++) {
                // weights[d * vocab_size + v] = weights at row d, col v
                layer_.weights[d * vocab_size_ + v] = row[d];
            }
        }
        std::fclose(f);
    }

    /// Return the embed_dim-dimensional embedding for token_id.
    /// NaN guard included.
    [[nodiscard]] Tensor lookup(int token_id) const {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size_)
            token_id = TokenVocab::UNK_ID;
        Tensor emb({embed_dim_});
        for (size_t d = 0; d < embed_dim_; d++) {
            float v = layer_.weights[d * vocab_size_ + static_cast<size_t>(token_id)];
            emb[d] = std::isfinite(v) ? v : 0.0f;
        }
        return emb;
    }

    /// Accumulate gradient for a single token's embedding.
    void backward_token(int token_id, const Tensor& grad) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size_) return;
        for (size_t d = 0; d < embed_dim_; d++) {
            float g = (d < grad.size()) ? grad[d] : 0.0f;
            if (!std::isfinite(g)) g = 0.0f;
            g = std::max(-1.0f, std::min(1.0f, g));  // gradient clip
            layer_.grad_w[d * vocab_size_ + static_cast<size_t>(token_id)] += g;
        }
        // Note: grad_w accumulation is flushed by apply_adam() → layer_.apply_adam()
    }

    /// Apply Adam update to embedding weights.
    void apply_adam(float lr) {
        adam_step_++;
        layer_.apply_adam(lr);
    }

    void serialize(CheckpointWriter& cw, const std::string& prefix) const {
        layer_.serialize(cw, prefix + "embed_");
    }

    void deserialize(const CheckpointReader& cr, const std::string& prefix) {
        layer_.deserialize(cr, prefix + "embed_");
    }

    size_t param_count() const { return layer_.param_count(); }
};

// ---------------------------------------------------------------------------
// GenerativeLoop — teacher-forcing training + autoregressive inference
// ---------------------------------------------------------------------------

/// Sequences binary file format (produced by tools/prepare_code_dataset.py):
///   int32  num_sequences
///   then for each sequence:
///     int32  length
///     length × int32  token_ids

/// Manages autoregressive training and generation.
/// Works with any DendriteNet3D configured as:
///   DendriteNet3D net(embed_dim, vocab_size)
/// where embed_dim = EmbedTable::embed_dim_ and vocab_size = EmbedTable::vocab_size_.
struct GenerativeLoop {
    const TokenVocab& vocab_;
    float temperature_ = 0.8f;   ///< Sampling temperature for generation

    explicit GenerativeLoop(const TokenVocab& vocab, float temperature = 0.8f)
        : vocab_(vocab), temperature_(temperature) {}

    // ------------------------------------------------------------------
    // Training (teacher forcing)
    // ------------------------------------------------------------------

    /// Train on a single token sequence using teacher forcing.
    /// Resets working memory before the sequence, then for each position i
    /// feeds embed(token[i]) and trains to predict token[i+1].
    ///
    /// @param token_ids  Sequence including BOS and EOS markers.
    /// @param net        DendriteNet3D(embed_dim, vocab_size).
    /// @param emb        EmbedTable with matching dimensions.
    /// @return Mean cross-entropy loss over the sequence.
    template <typename NetT>
    float train_sequence(const std::vector<int>& token_ids,
                         NetT& net,
                         EmbedTable& emb) {
        if (token_ids.size() < 2) return 0.0f;

        net.task_context.reset();
        float total_loss = 0.0f;
        size_t n_steps   = 0;

        for (size_t i = 0; i + 1 < token_ids.size(); i++) {
            int current_id = token_ids[i];
            int next_id    = token_ids[i + 1];

            // Skip PAD tokens
            if (current_id == TokenVocab::PAD_ID) continue;

            Tensor input_emb = emb.lookup(current_id);

            // One-hot target over vocab
            Tensor target({vocab_.size()});
            if (next_id >= 0 && static_cast<size_t>(next_id) < vocab_.size())
                target[static_cast<size_t>(next_id)] = 1.0f;

            float loss = net.train_sample(input_emb, target);
            if (std::isfinite(loss)) {
                total_loss += loss;
                n_steps++;
            }
        }

        // Update embedding table with the same LR as the network
        emb.apply_adam(net.learning_rate);

        return n_steps > 0 ? total_loss / static_cast<float>(n_steps) : 0.0f;
    }

    /// Load all sequences from the binary file and train for one epoch.
    /// Returns mean loss over all sequences.
    template <typename NetT>
    float train_epoch(const std::string& seq_path,
                      NetT& net,
                      EmbedTable& emb,
                      std::mt19937& rng) {
        // Load sequences into memory once
        std::vector<std::vector<int>> sequences = load_sequences(seq_path);

        // Shuffle sequence order each epoch
        std::shuffle(sequences.begin(), sequences.end(), rng);

        float total_loss = 0.0f;
        size_t n = 0;
        for (const auto& seq : sequences) {
            float loss = train_sequence(seq, net, emb);
            if (std::isfinite(loss) && loss > 0.0f) {
                total_loss += loss;
                n++;
            }
        }
        return n > 0 ? total_loss / static_cast<float>(n) : 0.0f;
    }

    // ------------------------------------------------------------------
    // Inference (autoregressive generation)
    // ------------------------------------------------------------------

    /// Generate a token sequence given a text prompt.
    ///
    /// @param prompt     Seed text (will be tokenised).
    /// @param net        Trained DendriteNet3D(embed_dim, vocab_size).
    /// @param emb        Trained EmbedTable.
    /// @param max_tokens Maximum number of new tokens to generate.
    /// @return Decoded output string.
    template <typename NetT>
    std::string generate(const std::string& prompt,
                         NetT& net,
                         EmbedTable& emb,
                         int max_tokens = 128) {
        net.task_context.reset();
        std::mt19937 sample_rng(std::random_device{}());

        // Encode prompt — warm up working memory with context
        std::vector<int> prompt_ids = vocab_.encode(prompt);
        int last_id = TokenVocab::BOS_ID;
        for (int id : prompt_ids) {
            Tensor emb_t = emb.lookup(id);
            net.infer(emb_t, "", /*update_memory=*/true);
            last_id = id;
        }

        // Autoregressive generation loop
        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(max_tokens));

        for (int step = 0; step < max_tokens; step++) {
            Tensor input_emb = emb.lookup(last_id);
            auto result = net.infer(input_emb, "", /*update_memory=*/true);

            // Sample next token from output distribution with temperature
            int next_id = sample_token(result.output, temperature_, sample_rng);

            if (next_id == TokenVocab::EOS_ID) break;
            if (next_id == TokenVocab::PAD_ID) continue;

            generated.push_back(next_id);
            last_id = next_id;
        }

        return vocab_.decode(generated);
    }

    // ------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------

    /// Load sequences from the binary file produced by prepare_code_dataset.py.
    static std::vector<std::vector<int>> load_sequences(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f)
            throw std::runtime_error(
                "GenerativeLoop: cannot open '" + path +
                "'. Run: python tools/prepare_code_dataset.py");

        int num_seqs = 0;
        std::fread(&num_seqs, sizeof(int), 1, f);

        std::vector<std::vector<int>> seqs;
        seqs.reserve(static_cast<size_t>(num_seqs));

        for (int s = 0; s < num_seqs; s++) {
            int len = 0;
            if (std::fread(&len, sizeof(int), 1, f) != 1) break;
            std::vector<int> seq(static_cast<size_t>(len));
            if (std::fread(seq.data(), sizeof(int),
                           static_cast<size_t>(len), f) !=
                static_cast<size_t>(len)) break;
            seqs.push_back(std::move(seq));
        }
        std::fclose(f);
        return seqs;
    }

private:
    /// Sample a token index from a probability distribution with temperature.
    static int sample_token(const Tensor& probs, float temperature, std::mt19937& rng) {
        size_t n = probs.size();
        if (n == 0) return TokenVocab::EOS_ID;

        if (temperature <= 0.0f) {
            // Greedy (argmax)
            return static_cast<int>(probs.argmax());
        }

        // Apply temperature scaling to logits (re-softmax after scaling)
        std::vector<float> logits(n);
        float max_l = -1e9f;
        for (size_t i = 0; i < n; i++) {
            float v = probs[i];
            if (!std::isfinite(v)) v = 0.0f;
            // Convert probability back to logit space via log, then scale
            logits[i] = (v > 1e-10f ? std::log(v) : -20.0f) / temperature;
            if (logits[i] > max_l) max_l = logits[i];
        }

        // Stable softmax
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            logits[i] = std::exp(logits[i] - max_l);
            sum += logits[i];
        }
        if (sum < 1e-10f) return static_cast<int>(n / 2);
        for (float& v : logits) v /= sum;

        // Multinomial sample
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        float r = uni(rng);
        float cum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            cum += logits[i];
            if (r <= cum) return static_cast<int>(i);
        }
        return static_cast<int>(n - 1);
    }
};

} // namespace dendrite
