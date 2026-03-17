#pragma once
// Enhancement #26: Hypernetwork Branch Generation
// A generator network maps learned domain embeddings to specialist weights,
// enabling faster initialization of new branches and zero-shot generation
// for known domain families.
//
// Key references: Ha et al. ICLR 2017 (HyperNetworks),
//                 Modular Hypernetworks, ICLR 2024.

#include "layer.hpp"
#include <string>
#include <unordered_map>
#include <cmath>

namespace dendrite {

// ============================================================
// DomainEmbedding: trainable vector representing a branch domain.
// Initialized from the domain name's character hash for variety;
// updated via Adam during the hypernetwork auxiliary training step.
// ============================================================
struct DomainEmbedding {
    Tensor embed;   // the learnable embedding vector
    Tensor grad;    // gradient accumulator (cleared after Adam step)
    Tensor m, v;    // Adam first/second moments
    int adam_t = 0;

    DomainEmbedding() = default;

    /// Construct from a domain name — deterministic but varied initialization.
    DomainEmbedding(size_t dim, const std::string& domain_name) {
        embed = Tensor({dim});
        grad  = Tensor({dim});
        m     = Tensor({dim});
        v     = Tensor({dim});
        // Character-hash init: gives each domain a distinct starting point
        for (size_t i = 0; i < dim; i++) {
            float val = 0.0f;
            for (size_t c = 0; c < domain_name.size(); c++) {
                val += static_cast<float>(static_cast<unsigned char>(domain_name[c]))
                     * std::sin(static_cast<float>(i + 1) * static_cast<float>(c + 1) * 0.37f);
            }
            embed[i] = std::tanh(val * 0.05f);
            if (!std::isfinite(embed[i])) embed[i] = 0.0f;
        }
    }

    void apply_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t++;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(adam_t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(adam_t));
        for (size_t i = 0; i < embed.size(); i++) {
            if (!std::isfinite(grad[i])) grad[i] = 0.0f;
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
            embed[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
            if (!std::isfinite(embed[i])) embed[i] = 0.0f;
        }
        grad.zero();
    }
};

// ============================================================
// BranchGenerator: hypernetwork that produces specialist weights.
// Architecture: domain_embed → [hidden] → tanh → flat_specialist_weights
// Trained via auxiliary MSE loss: ||generate(e) − actual_weights||²
// ============================================================
struct BranchGenerator {
    size_t domain_embed_dim = 16;
    size_t total_specialist_params = 0;
    bool enabled = false;
    MiniNetwork generator;  // domain_embed_dim → hidden → total_specialist_params

    BranchGenerator() = default;

    /// Initialize generator. Call after build() so total_params is known.
    void init(size_t embed_dim, size_t total_params, std::mt19937& rng) {
        domain_embed_dim = embed_dim;
        total_specialist_params = total_params;
        size_t hidden = std::min((size_t)128, total_params);
        generator = MiniNetwork("hypernet_gen",
            {embed_dim, hidden, total_params},
            Activation::RELU, Activation::TANH, rng);
        enabled = true;
    }

    // --------------------------------------------------------
    // Utility: count, flatten, and unflatten specialist weights
    // --------------------------------------------------------
    static size_t count_params(const MiniNetwork& net) {
        size_t total = 0;
        for (const auto& layer : net.layers)
            total += layer.weights.size() + layer.bias.size();
        return total;
    }

    /// Flatten all weights and biases into a single Tensor.
    static Tensor flatten(const MiniNetwork& specialist) {
        size_t total = count_params(specialist);
        Tensor flat({total});
        size_t off = 0;
        for (const auto& layer : specialist.layers) {
            for (size_t i = 0; i < layer.weights.size(); i++) flat[off++] = layer.weights[i];
            for (size_t i = 0; i < layer.bias.size();    i++) flat[off++] = layer.bias[i];
        }
        return flat;
    }

    /// Load a flat weight vector into a specialist's layer tensors.
    static void unflatten(MiniNetwork& specialist, const Tensor& flat) {
        size_t off = 0;
        for (auto& layer : specialist.layers) {
            for (size_t i = 0; i < layer.weights.size() && off < flat.size(); i++)
                layer.weights[i] = flat[off++];
            for (size_t i = 0; i < layer.bias.size()    && off < flat.size(); i++)
                layer.bias[i] = flat[off++];
        }
    }

    // --------------------------------------------------------
    // Generate and populate a specialist from a domain embedding.
    // Used at branch creation and split time.
    // --------------------------------------------------------
    void populate_specialist(MiniNetwork& specialist, const Tensor& domain_embed) {
        if (!enabled) return;
        Tensor generated = generator.forward(domain_embed);
        for (auto& v : generated.data) if (!std::isfinite(v)) v = 0.0f;
        // Scale to match He init magnitude (tanh output is in [-1,1])
        float scale = std::sqrt(2.0f / static_cast<float>(specialist.layers[0].in_dim));
        for (auto& v : generated.data) v *= scale;
        unflatten(specialist, generated);
    }

    // --------------------------------------------------------
    // Compute auxiliary meta-loss: ||generate(embed) − actual||²
    // Backprops through generator and accumulates grad into domain_embed_state.
    // Returns MSE loss (for monitoring).
    // --------------------------------------------------------
    float meta_step(DomainEmbedding& domain_state, const MiniNetwork& specialist) {
        if (!enabled) return 0.0f;
        const Tensor& embed = domain_state.embed;
        Tensor generated = generator.forward(embed);
        Tensor actual    = flatten(specialist);
        if (generated.size() != actual.size()) return 0.0f;

        // MSE gradient: grad_i = 2/N * (generated_i − actual_i)
        Tensor gen_grad({generated.size()});
        float loss = 0.0f;
        float inv_n = 1.0f / static_cast<float>(generated.size());
        for (size_t i = 0; i < generated.size(); i++) {
            float diff = generated[i] - actual[i];
            loss += diff * diff;
            gen_grad[i] = std::clamp(2.0f * diff * inv_n, -0.5f, 0.5f);
        }
        loss *= inv_n;
        if (!std::isfinite(loss)) return 0.0f;

        // Backprop through generator; propagated grad reaches domain_embed
        Tensor embed_grad = generator.backward(gen_grad);
        for (size_t i = 0; i < domain_state.grad.size() && i < embed_grad.size(); i++) {
            domain_state.grad[i] += std::clamp(embed_grad[i], -0.5f, 0.5f);
        }
        return loss;
    }

    void apply_adam(float lr) { generator.apply_adam(lr); }

    size_t param_count() const { return generator.param_count(); }
};

} // namespace dendrite
