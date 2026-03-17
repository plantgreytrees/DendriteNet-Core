#pragma once
#include "layer.hpp"
#include <unordered_map>
#include <cstdlib>   // std::getenv
#ifdef DENDRITE_ONNX
#include "onnx_encoder.hpp"
#endif

namespace dendrite {

// ============================================================
// ModalityModule: base for image and audio modules
// ============================================================
// In the POC, these use simulated embeddings. The ONNX Runtime
// integration points are marked with comments for when you add
// the real encoder backends.
//
// Selected models for future integration:
//   Image: MobileNetV2 (ONNX Zoo) → 1280-dim embeddings
//   Audio: YAMNet (exported to ONNX) → 1024-dim embeddings
// ============================================================
class ModalityModule {
public:
    std::string name;
    size_t encoder_dim;        // dimension of raw encoder output
    size_t shared_dim;         // dimension after projection to shared space
    bool loaded = false;

    MiniNetwork projection;    // encoder_dim → shared_dim

    // Association memory: concept name → embedding centroid
    std::unordered_map<std::string, Tensor> associations;

    // Stats
    size_t total_encodes = 0;
    size_t total_lookups = 0;
    size_t association_hits = 0;

    // ONNX Runtime session (null when running in stub/simulation mode)
#ifdef DENDRITE_ONNX
    std::shared_ptr<OrtEncoder> ort_encoder_;
#endif
    bool ort_warn_shown_ = false;  // suppress repeated fallback warnings

    ModalityModule() = default;

    ModalityModule(const std::string& name_, size_t encoder_dim_,
                   size_t shared_dim_, std::mt19937& rng)
        : name(name_), encoder_dim(encoder_dim_), shared_dim(shared_dim_) {
        projection = MiniNetwork(name_ + "_proj",
            {encoder_dim_, shared_dim_ * 2, shared_dim_},
            Activation::RELU, Activation::TANH, rng);
        loaded = true;
    }

    // --------------------------------------------------------
    // Load a real ONNX encoder to replace the simulation stub.
    //
    // @param model_path   Path to a .onnx model file.
    // @param input_shape  Flat shape of the model's input tensor, e.g.
    //                     {1,3,224,224} for MobileNetV2 or {1,15600} for YAMNet.
    // @param input_node   ONNX input node name  (default "input").
    // @param output_node  ONNX output node name (default "output").
    // @return true if loaded successfully; false with a stderr warning on failure.
    //
    // When -DDENDRITE_ONNX is not set this always returns false and prints a
    // one-line hint about how to recompile.
    // --------------------------------------------------------
    bool load_onnx(const std::string&        model_path,
                   const std::vector<int64_t>& input_shape,
                   const std::string&        input_node  = "input",
                   const std::string&        output_node = "output") {
#ifdef DENDRITE_ONNX
        try {
            ort_encoder_ = std::make_shared<OrtEncoder>(
                model_path, input_shape, encoder_dim, input_node, output_node);
            ort_warn_shown_ = false;
            std::printf("[DendriteNet] %s: ONNX encoder loaded — %s "
                        "(input %zu elems → %zu-dim embedding)\n",
                        name.c_str(), model_path.c_str(),
                        ort_encoder_->input_size(), encoder_dim);
            return true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr,
                "[DendriteNet] %s: failed to load ONNX model '%s': %s\n"
                "  → falling back to simulation stub\n",
                name.c_str(), model_path.c_str(), e.what());
            ort_encoder_.reset();
            return false;
        }
#else
        (void)model_path; (void)input_shape;
        (void)input_node; (void)output_node;
        std::fprintf(stderr,
            "[DendriteNet] %s: ONNX support not compiled in "
            "(rebuild with -DDENDRITE_ONNX -lonnxruntime)\n",
            name.c_str());
        return false;
#endif
    }

    // --------------------------------------------------------
    // Encode raw input → encoder_dim-dimensional embedding.
    //
    // If an ONNX encoder is loaded (via load_onnx or auto env-var), the real
    // model is run; otherwise the deterministic simulation stub is used.
    // NaN guard is applied to all ONNX outputs before returning.
    // --------------------------------------------------------
    Tensor encode(const Tensor& raw_input) {
#ifdef DENDRITE_ONNX
        if (ort_encoder_) {
            if (raw_input.size() == ort_encoder_->input_size()) {
                try {
                    auto emb = ort_encoder_->run(raw_input.data.data());
                    Tensor result({encoder_dim});
                    for (size_t i = 0; i < encoder_dim; i++)
                        result[i] = (i < emb.size() && std::isfinite(emb[i]))
                                    ? emb[i] : 0.0f;
                    total_encodes++;
                    return result;
                } catch (const Ort::Exception& e) {
                    if (!ort_warn_shown_) {
                        std::fprintf(stderr,
                            "[DendriteNet] %s: ONNX inference error (%s) "
                            "— falling back to stub\n",
                            name.c_str(), e.what());
                        ort_warn_shown_ = true;
                    }
                }
            } else if (!ort_warn_shown_) {
                std::fprintf(stderr,
                    "[DendriteNet] %s: input size %zu != expected %zu "
                    "— using simulation stub\n",
                    name.c_str(), raw_input.size(), ort_encoder_->input_size());
                ort_warn_shown_ = true;
            }
        }
#endif
        // Simulation stub — deterministic pseudo-encoding (always available as fallback)
        Tensor embedding({encoder_dim});
        for (size_t i = 0; i < encoder_dim; i++) {
            float sum = 0;
            for (size_t j = 0; j < raw_input.size(); j++)
                sum += raw_input[j] * std::sin((float)(i * raw_input.size() + j) * 0.1f);
            embedding[i] = std::tanh(sum);
        }
        total_encodes++;
        return embedding;
    }

    // --------------------------------------------------------
    // Project embedding into shared space
    // --------------------------------------------------------
    Tensor project(const Tensor& embedding) {
        return projection.forward(embedding);
    }

    // --------------------------------------------------------
    // Full pipeline: raw input → shared-space embedding
    // --------------------------------------------------------
    Tensor process(const Tensor& raw_input) {
        return project(encode(raw_input));
    }

    // --------------------------------------------------------
    // Association memory: lookup concept by name
    // Returns shared-space embedding if concept is known
    // --------------------------------------------------------
    bool has_association(const std::string& concept_name) const {
        return associations.count(concept_name) > 0;
    }

    Tensor lookup(const std::string& concept_name) {
        total_lookups++;
        auto it = associations.find(concept_name);
        if (it != associations.end()) {
            association_hits++;
            return it->second;
        }
        return Tensor({shared_dim});  // zero vector if unknown
    }

    // --------------------------------------------------------
    // Register an association: bind a concept to an embedding
    // --------------------------------------------------------
    void register_association(const std::string& concept_name, const Tensor& raw_input) {
        associations[concept_name] = process(raw_input);
    }

    void register_association_direct(const std::string& concept_name, const Tensor& shared_embedding) {
        associations[concept_name] = shared_embedding;
    }

    // --------------------------------------------------------
    // Batch register from concept-data pairs
    // --------------------------------------------------------
    void register_batch(const std::vector<std::pair<std::string, Tensor>>& pairs) {
        for (auto& [concept_name, data] : pairs)
            register_association(concept_name, data);
    }

    // --------------------------------------------------------
    // Find closest association to a given embedding
    // --------------------------------------------------------
    std::string find_closest(const Tensor& embedding, float* best_sim_out = nullptr) const {
        std::string best_concept;
        float best_sim = -1e9f;

        for (auto& [concept_name, assoc_emb] : associations) {
            float dot = 0, na = 0, nb = 0;
            size_t dim = std::min(embedding.size(), assoc_emb.size());
            for (size_t i = 0; i < dim; i++) {
                dot += embedding[i] * assoc_emb[i];
                na += embedding[i] * embedding[i];
                nb += assoc_emb[i] * assoc_emb[i];
            }
            float sim = (na > 1e-7f && nb > 1e-7f) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0;
            if (sim > best_sim) {
                best_sim = sim;
                best_concept = concept_name;
            }
        }
        if (best_sim_out) *best_sim_out = best_sim;
        return best_concept;
    }

    // --------------------------------------------------------
    // Enhancement #25: CLIP-style contrastive alignment loss.
    // Measures InfoNCE alignment between a text/branch embedding and
    // a modality association embedding.
    // Returns: scalar loss (positive = pull together, penalize divergence).
    // Both embeddings are L2-normalized before comparison.
    // --------------------------------------------------------
    static float alignment_loss(const Tensor& text_embed,
                                 const Tensor& mod_embed,
                                 float temperature = 0.07f) {
        if (text_embed.size() == 0 || mod_embed.size() == 0) return 0.0f;

        // L2-normalize text_embed
        float na = 0.0f;
        for (size_t i = 0; i < text_embed.size(); i++) na += text_embed[i] * text_embed[i];
        na = std::sqrt(na + 1e-8f);

        // L2-normalize mod_embed
        float nb = 0.0f;
        size_t dim = std::min(text_embed.size(), mod_embed.size());
        for (size_t i = 0; i < dim; i++) nb += mod_embed[i] * mod_embed[i];
        nb = std::sqrt(nb + 1e-8f);

        // Cosine similarity
        float sim = 0.0f;
        for (size_t i = 0; i < dim; i++)
            sim += (text_embed[i] / na) * (mod_embed[i] / nb);

        // InfoNCE single-pair: -log σ(sim/T) → maximise aligned similarity
        float scaled = std::clamp(sim / temperature, -20.0f, 20.0f);
        float loss = -std::log(1.0f / (1.0f + std::exp(-scaled)) + 1e-8f);
        return std::isfinite(loss) ? loss : 0.0f;
    }

    bool is_loaded()      const { return loaded; }
    /// Returns true when a real ONNX encoder is active (false = simulation stub).
    bool is_onnx_active() const {
#ifdef DENDRITE_ONNX
        return ort_encoder_ != nullptr;
#else
        return false;
#endif
    }
    void apply_adam(float lr) { projection.apply_adam(lr); }
    size_t param_count() const { return projection.param_count(); }

    void print_stats() const {
#ifdef DENDRITE_ONNX
        const char* enc_tag = ort_encoder_ ? "ONNX" : "stub";
#else
        const char* enc_tag = "stub";
#endif
        printf("  [%s/%s] %s: %zu associations, %zu encodes, %zu lookups (%zu hits)\n",
               loaded ? "ON" : "OFF", enc_tag, name.c_str(),
               associations.size(), total_encodes, total_lookups, association_hits);
    }
};

// ============================================================
// Enhancement 10: GatedCrossAttention — multimodal fusion layer
// ============================================================
// Inserts between branch specialist output and final fusion.
// Q from branch output; K/V from collected modality embeddings.
// Flamingo-style scalar gate initialised to 0 → no-op at start.
// Gate grows via schedule as Q/K/V projections converge.
// ============================================================
struct GatedCrossAttention {
    MiniNetwork q_proj;   // branch_dim → attn_dim
    MiniNetwork k_proj;   // modality_dim → attn_dim
    MiniNetwork v_proj;   // modality_dim → attn_dim
    size_t branch_dim = 0;
    size_t modality_dim = 0;
    size_t attn_dim = 16;
    float gate_alpha = 0.0f;     // current mixing weight [0, max_gate]
    float max_gate = 0.5f;       // upper bound — keeps learned routing dominant
    float gate_open_rate = 5e-5f;  // per apply_adam() step increase
    bool initialised = false;

    GatedCrossAttention() = default;

    void init(size_t branch_dim_, size_t modality_dim_, std::mt19937& rng,
              size_t attn_dim_ = 16) {
        branch_dim = branch_dim_;
        modality_dim = modality_dim_;
        attn_dim = attn_dim_;
        gate_alpha = 0.0f;  // Flamingo-style: start closed
        q_proj = MiniNetwork("gca_q", {branch_dim,   attn_dim}, Activation::NONE, Activation::NONE, rng);
        k_proj = MiniNetwork("gca_k", {modality_dim, attn_dim}, Activation::NONE, Activation::NONE, rng);
        v_proj = MiniNetwork("gca_v", {modality_dim, attn_dim}, Activation::NONE, Activation::NONE, rng);
        initialised = true;
    }

    // Apply cross-attention: branch_out → enriched via modality embeddings
    // Returns branch_out when gate is 0 (safe default)
    Tensor forward(const Tensor& branch_out,
                   const std::vector<Tensor>& mod_embeddings) {
        if (!initialised || mod_embeddings.empty() || gate_alpha < 1e-6f)
            return branch_out;

        Tensor q = q_proj.forward(branch_out);
        for (auto& v : q.data) if (!std::isfinite(v)) v = 0.0f;

        float scale = 1.0f / std::sqrt((float)attn_dim);
        size_t n = mod_embeddings.size();
        std::vector<float> attn_logits(n);
        std::vector<Tensor> vals(n);
        for (size_t i = 0; i < n; i++) {
            vals[i] = v_proj.forward(mod_embeddings[i]);
            Tensor k = k_proj.forward(mod_embeddings[i]);
            float dot = 0.0f;
            size_t d = std::min(q.size(), k.size());
            for (size_t j = 0; j < d; j++) dot += q[j] * k[j];
            attn_logits[i] = dot * scale;
            if (!std::isfinite(attn_logits[i])) attn_logits[i] = 0.0f;
        }

        // Stable softmax
        float mx = *std::max_element(attn_logits.begin(), attn_logits.end());
        float sum_exp = 0.0f;
        std::vector<float> attn(n);
        for (size_t i = 0; i < n; i++) {
            attn[i] = std::exp(attn_logits[i] - mx);
            sum_exp += attn[i];
        }
        if (sum_exp > 1e-7f) for (auto& a : attn) a /= sum_exp;

        // Attention-weighted value sum (attn_dim space)
        Tensor agg({attn_dim});
        for (size_t i = 0; i < n; i++) {
            size_t d = std::min(agg.size(), vals[i].size());
            for (size_t j = 0; j < d; j++) {
                agg[j] += attn[i] * vals[i][j];
            }
        }

        // Add gated residual: result = branch_out + gate_alpha * agg[:branch_dim]
        Tensor result = branch_out;
        size_t d = std::min(agg.size(), branch_out.size());
        for (size_t i = 0; i < d; i++) {
            result[i] += gate_alpha * agg[i];
            if (!std::isfinite(result[i])) result[i] = branch_out[i];
        }
        return result;
    }

    void apply_adam(float lr) {
        if (!initialised) return;
        q_proj.apply_adam(lr);
        k_proj.apply_adam(lr);
        v_proj.apply_adam(lr);
        // Slowly open the gate so cross-attention comes online as Q/K/V learn
        gate_alpha = std::min(max_gate, gate_alpha + gate_open_rate);
    }

    size_t param_count() const {
        if (!initialised) return 0;
        return q_proj.param_count() + k_proj.param_count() + v_proj.param_count();
    }
};

// ============================================================
// Convenience: pre-configured image and audio modules.
//
// Auto-load ONNX encoders from environment variables when
// -DDENDRITE_ONNX is active:
//   DENDRITE_IMAGE_MODEL  — path to MobileNetV2 .onnx  (input [1,3,224,224])
//   DENDRITE_AUDIO_MODEL  — path to YAMNet/audio .onnx (input [1,15600])
//
// Generate model files with:
//   python tools/export_encoders.py --out-dir models/
// ============================================================
inline ModalityModule create_image_module(size_t shared_dim, std::mt19937& rng,
                                          const std::string& model_path = "") {
    // MobileNetV2 penultimate layer = 1280-dim
    ModalityModule m("image", 1280, shared_dim, rng);
#ifdef DENDRITE_ONNX
    // Resolve model path: explicit arg → env var → skip
    std::string path = model_path;
    if (path.empty()) {
        const char* env = std::getenv("DENDRITE_IMAGE_MODEL");
        if (env) path = env;
    }
    if (!path.empty())
        m.load_onnx(path, {1, 3, 224, 224}, "input", "output");
#else
    (void)model_path;
#endif
    return m;
}

inline ModalityModule create_audio_module(size_t shared_dim, std::mt19937& rng,
                                          const std::string& model_path = "") {
    // YAMNet penultimate layer = 1024-dim
    // Input: [1, 15600] float32 (0.975 s at 16 kHz)
    ModalityModule m("audio", 1024, shared_dim, rng);
#ifdef DENDRITE_ONNX
    std::string path = model_path;
    if (path.empty()) {
        const char* env = std::getenv("DENDRITE_AUDIO_MODEL");
        if (env) path = env;
    }
    if (!path.empty())
        m.load_onnx(path, {1, 15600}, "input", "output");
#else
    (void)model_path;
#endif
    return m;
}

} // namespace dendrite
