#pragma once
#include "tensor.hpp"
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace dendrite {

struct ProcessedToken {
    std::string text;
    float weight;       // 1.0 for content words, reduced for stop-words
    bool is_stop;
    int original_pos;   // position in original sentence
};

// ============================================================
// TextPreprocessor: compresses function words like the brain does
// "the cat sat on the mat" → content words get full weight,
// stop-words get reduced weight but preserve structural info
// ============================================================
class TextPreprocessor {
public:
    std::unordered_set<std::string> stop_words;
    float compression_intensity = 0.3f;  // 0=equal, 1=fully ignore stops

    TextPreprocessor() { load_default_stops(); }

    explicit TextPreprocessor(float intensity) : compression_intensity(intensity) {
        load_default_stops();
    }

    std::vector<ProcessedToken> process(const std::string& input) const {
        std::vector<ProcessedToken> tokens;
        std::istringstream stream(input);
        std::string word;
        int pos = 0;

        while (stream >> word) {
            std::string lower = to_lower(strip_punct(word));
            if (lower.empty()) { pos++; continue; }

            ProcessedToken tok;
            tok.text = lower;
            tok.original_pos = pos;
            tok.is_stop = stop_words.count(lower) > 0;
            tok.weight = tok.is_stop ? (1.0f - compression_intensity) : 1.0f;
            tokens.push_back(tok);
            pos++;
        }
        return tokens;
    }

    // Build a weighted embedding sequence from tokens + a vocabulary
    // Stop-words share a single placeholder embedding, scaled by weight
    // Content words get their full embedding at weight 1.0
    struct EmbeddedSequence {
        std::vector<Tensor> embeddings;
        std::vector<float> weights;
        std::vector<std::string> tokens;
        size_t content_count = 0;
        size_t stop_count = 0;
    };

    EmbeddedSequence embed(const std::string& input,
                           const std::unordered_map<std::string, Tensor>& vocab,
                           const Tensor& unknown_embedding,
                           const Tensor& stop_placeholder) const {
        auto tokens = process(input);
        EmbeddedSequence seq;

        for (auto& tok : tokens) {
            seq.tokens.push_back(tok.text);
            seq.weights.push_back(tok.weight);

            if (tok.is_stop) {
                // All stop-words share the same placeholder, scaled by weight
                seq.embeddings.push_back(stop_placeholder * tok.weight);
                seq.stop_count++;
            } else {
                auto it = vocab.find(tok.text);
                if (it != vocab.end()) {
                    seq.embeddings.push_back(it->second);
                } else {
                    seq.embeddings.push_back(unknown_embedding);
                }
                seq.content_count++;
            }
        }
        return seq;
    }

    // Produce a single weighted-average embedding for the whole input
    Tensor embed_pooled(const std::string& input,
                        const std::unordered_map<std::string, Tensor>& vocab,
                        const Tensor& unknown_embedding,
                        const Tensor& stop_placeholder) const {
        auto seq = embed(input, vocab, unknown_embedding, stop_placeholder);
        if (seq.embeddings.empty()) return unknown_embedding;

        size_t dim = seq.embeddings[0].size();
        Tensor pooled({dim});
        float total_weight = 0;

        for (size_t i = 0; i < seq.embeddings.size(); i++) {
            float w = seq.weights[i];
            for (size_t d = 0; d < dim; d++)
                pooled[d] += seq.embeddings[i][d] * w;
            total_weight += w;
        }
        if (total_weight > 1e-7f)
            for (size_t d = 0; d < dim; d++) pooled[d] /= total_weight;

        return pooled;
    }

    void add_stop_word(const std::string& word) { stop_words.insert(to_lower(word)); }
    void remove_stop_word(const std::string& word) { stop_words.erase(to_lower(word)); }

    void print_stats(const std::string& input) const {
        auto tokens = process(input);
        size_t stops = 0, content = 0;
        for (auto& t : tokens) t.is_stop ? stops++ : content++;
        printf("  Input: \"%s\"\n", input.c_str());
        printf("  Tokens: %zu total, %zu content (full weight), %zu stop (%.0f%% weight)\n",
               tokens.size(), content, stops, (1.0f - compression_intensity) * 100);
        printf("  Effective reduction: %.1f%% fewer full-weight tokens\n",
               tokens.size() > 0 ? 100.0f * stops / tokens.size() : 0.0f);
    }

private:
    void load_default_stops() {
        // ~150 most common English function words
        const char* defaults[] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "because", "but", "and", "or", "if", "while", "although",
            "though", "after", "that", "which", "who", "whom", "this", "these",
            "those", "am", "it", "its", "itself", "they", "them", "their", "theirs",
            "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his",
            "she", "her", "hers", "i", "me", "my", "mine", "what", "about", "up",
            "also", "still", "even", "now", "already", "yet", "much", "many"
        };
        for (auto w : defaults) stop_words.insert(w);
    }

    static std::string to_lower(const std::string& s) {
        std::string r = s;
        std::transform(r.begin(), r.end(), r.begin(), ::tolower);
        return r;
    }

    static std::string strip_punct(const std::string& s) {
        std::string r;
        for (char c : s)
            if (std::isalnum(c) || c == '\'') r += c;
        return r;
    }
};

} // namespace dendrite
