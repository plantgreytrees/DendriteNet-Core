#pragma once
// Enhancement #24: Perceiver IO Fusion
// Modality-agnostic fusion via a fixed-size learned latent bottleneck.
// Latents (Q) cross-attend to all modality tokens (K/V), then self-attend
// to refine the representation. Decouples compute from input count.
//
// Key reference: Jaegle et al., DeepMind, ICML 2021.
//
// At DendriteNet scale: num_latents=8, latent_dim=32.
// Each modality embedding = one "token" (single-vector tokenization).

#include "layer.hpp"

namespace dendrite {

// ============================================================
// PerceiverIO: latent cross-attention + self-attention fusion.
//
// Forward: modality_tokens → latents attend → pooled output → proj.
// Compute is O(L × M) where L=num_latents, M=num_modality_tokens.
// ============================================================
struct PerceiverIO {
    size_t num_latents = 8;    // number of latent query vectors
    size_t latent_dim  = 32;   // dimension of each latent
    size_t output_dim  = 6;    // final projection target (= branch output_dim)
    bool enabled = false;

    // Learned initial latent array [num_latents × latent_dim] stored as flat vector
    Tensor latent_init;   // shape: {num_latents * latent_dim}
    Tensor latent_grad;   // gradient for latent_init
    Tensor latent_m, latent_v;
    int latent_adam_t = 0;

    // Cross-attention: latents (Q) attend to modality tokens (K, V)
    DenseLayer xattn_q;  // latent_dim → latent_dim (query projection)
    DenseLayer xattn_k;  // token_dim  → latent_dim (key projection)
    DenseLayer xattn_v;  // token_dim  → latent_dim (value projection)

    // Self-attention: latents attend to each other (2 layers)
    DenseLayer sattn_q1, sattn_k1, sattn_v1;
    DenseLayer sattn_q2, sattn_k2, sattn_v2;

    // Output projection: mean-pooled latent → output_dim
    DenseLayer output_proj;  // latent_dim → output_dim

    PerceiverIO() = default;

    /// Initialize Perceiver IO.
    /// token_dim: dimension of each modality embedding token (= modality shared_dim).
    /// out_dim: output dimension to project to (= branch output_dim).
    void init(size_t token_dim, size_t out_dim, std::mt19937& rng,
              size_t n_latents = 8, size_t lat_dim = 32) {
        num_latents = n_latents;
        latent_dim  = lat_dim;
        output_dim  = out_dim;

        // Learned latent array — Xavier init
        latent_init = Tensor({num_latents * latent_dim});
        latent_init.xavier_init(rng);
        latent_grad = Tensor({num_latents * latent_dim});
        latent_m    = Tensor({num_latents * latent_dim});
        latent_v    = Tensor({num_latents * latent_dim});

        // Cross-attention projections
        xattn_q = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);
        xattn_k = DenseLayer(token_dim,  latent_dim, Activation::NONE, rng);
        xattn_v = DenseLayer(token_dim,  latent_dim, Activation::NONE, rng);

        // Self-attention projections (2 rounds)
        sattn_q1 = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);
        sattn_k1 = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);
        sattn_v1 = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);

        sattn_q2 = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);
        sattn_k2 = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);
        sattn_v2 = DenseLayer(latent_dim, latent_dim, Activation::NONE, rng);

        // Output projection
        output_proj = DenseLayer(latent_dim, out_dim, Activation::SOFTMAX, rng);

        enabled = true;
    }

    // --------------------------------------------------------
    // Scaled dot-product attention over a set of key-value pairs.
    // query: [latent_dim], keys/values: each [latent_dim].
    // Returns attended output of shape [latent_dim].
    // --------------------------------------------------------
    static Tensor dot_attend(const Tensor& query,
                              const std::vector<Tensor>& keys,
                              const std::vector<Tensor>& values,
                              float scale) {
        if (keys.empty()) return query;
        // Compute attention weights: softmax(q·k / scale)
        std::vector<float> scores(keys.size());
        for (size_t j = 0; j < keys.size(); j++) {
            float dot = 0.0f;
            size_t d = std::min(query.size(), keys[j].size());
            for (size_t i = 0; i < d; i++) dot += query[i] * keys[j][i];
            scores[j] = dot * scale;
        }
        // Stable softmax
        float max_s = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (auto& s : scores) { s = std::exp(s - max_s); sum_exp += s; }
        if (sum_exp < 1e-8f) sum_exp = 1e-8f;
        for (auto& s : scores) s /= sum_exp;

        // Weighted sum of values
        size_t vdim = values[0].size();
        Tensor out({vdim});
        for (size_t j = 0; j < values.size(); j++)
            for (size_t i = 0; i < vdim && i < values[j].size(); i++)
                out[i] += scores[j] * values[j][i];
        for (auto& v : out.data) if (!std::isfinite(v)) v = 0.0f;
        return out;
    }

    // --------------------------------------------------------
    // Forward pass: fuse a list of modality tokens into output_dim vector.
    // Falls back to mean-pooling if tokens empty or not enabled.
    // --------------------------------------------------------
    Tensor forward(const std::vector<Tensor>& modality_tokens) {
        if (!enabled || modality_tokens.empty() || latent_init.size() == 0) {
            // Fallback: return zeros (caller blends with GCA or branch output)
            return Tensor({output_dim});
        }

        // Scale factor for attention
        float scale = 1.0f / std::sqrt(static_cast<float>(latent_dim));

        // Project all modality tokens to K and V
        std::vector<Tensor> token_keys, token_vals;
        for (const auto& tok : modality_tokens) {
            token_keys.push_back(xattn_k.forward(tok));
            token_vals.push_back(xattn_v.forward(tok));
            for (auto& v : token_keys.back().data) if (!std::isfinite(v)) v = 0.0f;
            for (auto& v : token_vals.back().data) if (!std::isfinite(v)) v = 0.0f;
        }

        // === Cross-attention pass: latents Q attend to modality tokens K/V ===
        std::vector<Tensor> latents(num_latents);
        for (size_t l = 0; l < num_latents; l++) {
            // Extract latent l from flat init tensor
            Tensor lat_l({latent_dim});
            for (size_t d = 0; d < latent_dim; d++)
                lat_l[d] = latent_init[l * latent_dim + d];

            Tensor q = xattn_q.forward(lat_l);
            for (auto& v : q.data) if (!std::isfinite(v)) v = 0.0f;

            Tensor attn_out = dot_attend(q, token_keys, token_vals, scale);
            // Residual: latent + attn_out
            for (size_t d = 0; d < latent_dim && d < attn_out.size(); d++) {
                lat_l[d] += attn_out[d];
                if (!std::isfinite(lat_l[d])) lat_l[d] = 0.0f;
            }
            latents[l] = lat_l;
        }

        // === Self-attention pass 1: latents attend to each other ===
        {
            std::vector<Tensor> k1, v1;
            for (auto& lat : latents) {
                k1.push_back(sattn_k1.forward(lat));
                v1.push_back(sattn_v1.forward(lat));
                for (auto& v : k1.back().data) if (!std::isfinite(v)) v = 0.0f;
                for (auto& v : v1.back().data) if (!std::isfinite(v)) v = 0.0f;
            }
            for (size_t l = 0; l < num_latents; l++) {
                Tensor q = sattn_q1.forward(latents[l]);
                Tensor a = dot_attend(q, k1, v1, scale);
                for (size_t d = 0; d < latent_dim && d < a.size(); d++) {
                    latents[l][d] += a[d];
                    if (!std::isfinite(latents[l][d])) latents[l][d] = 0.0f;
                }
            }
        }

        // === Self-attention pass 2 ===
        {
            std::vector<Tensor> k2, v2;
            for (auto& lat : latents) {
                k2.push_back(sattn_k2.forward(lat));
                v2.push_back(sattn_v2.forward(lat));
                for (auto& v : k2.back().data) if (!std::isfinite(v)) v = 0.0f;
                for (auto& v : v2.back().data) if (!std::isfinite(v)) v = 0.0f;
            }
            for (size_t l = 0; l < num_latents; l++) {
                Tensor q = sattn_q2.forward(latents[l]);
                Tensor a = dot_attend(q, k2, v2, scale);
                for (size_t d = 0; d < latent_dim && d < a.size(); d++) {
                    latents[l][d] += a[d];
                    if (!std::isfinite(latents[l][d])) latents[l][d] = 0.0f;
                }
            }
        }

        // === Mean-pool latents → output projection ===
        Tensor pooled({latent_dim});
        for (const auto& lat : latents)
            for (size_t d = 0; d < latent_dim && d < lat.size(); d++)
                pooled[d] += lat[d];
        float inv_l = 1.0f / static_cast<float>(num_latents);
        for (auto& v : pooled.data) {
            v *= inv_l;
            if (!std::isfinite(v)) v = 0.0f;
        }

        Tensor out = output_proj.forward(pooled);
        for (auto& v : out.data) if (!std::isfinite(v)) v = 0.0f;
        return out;
    }

    // --------------------------------------------------------
    // Minimal Adam update for projection layers (no full backprop).
    // Called once per batch step when perceiver is active.
    // --------------------------------------------------------
    void apply_adam(float lr) {
        xattn_q.apply_adam(lr); xattn_k.apply_adam(lr); xattn_v.apply_adam(lr);
        sattn_q1.apply_adam(lr); sattn_k1.apply_adam(lr); sattn_v1.apply_adam(lr);
        sattn_q2.apply_adam(lr); sattn_k2.apply_adam(lr); sattn_v2.apply_adam(lr);
        output_proj.apply_adam(lr);
        // Update latent_init via Adam
        latent_adam_t++;
        float bc1 = 1.0f - std::pow(0.9f,  static_cast<float>(latent_adam_t));
        float bc2 = 1.0f - std::pow(0.999f, static_cast<float>(latent_adam_t));
        for (size_t i = 0; i < latent_init.size(); i++) {
            latent_m[i] = 0.9f * latent_m[i] + 0.1f  * latent_grad[i];
            latent_v[i] = 0.999f * latent_v[i] + 0.001f * latent_grad[i] * latent_grad[i];
            latent_init[i] -= lr * (latent_m[i] / bc1) / (std::sqrt(latent_v[i] / bc2) + 1e-8f);
            if (!std::isfinite(latent_init[i])) latent_init[i] = 0.0f;
        }
        latent_grad.zero();
    }

    size_t param_count() const {
        return xattn_q.param_count() + xattn_k.param_count() + xattn_v.param_count()
             + sattn_q1.param_count() + sattn_k1.param_count() + sattn_v1.param_count()
             + sattn_q2.param_count() + sattn_k2.param_count() + sattn_v2.param_count()
             + output_proj.param_count() + latent_init.size();
    }
};

} // namespace dendrite
