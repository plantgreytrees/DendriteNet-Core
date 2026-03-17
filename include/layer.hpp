#pragma once
#include "tensor.hpp"
#include "checkpoint.hpp"
#include <array>
#include <optional>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dendrite {

enum class Activation { NONE, RELU, SIGMOID, SOFTMAX, TANH };

struct DenseLayer {
    Tensor weights, bias, grad_w, grad_b;
    Tensor last_input, last_pre_act, last_output;
    // Batch path cache (forward_batch / backward_batch)
    Tensor last_batch_input;   // [B × in_dim]
    Tensor last_batch_pre_act; // [B × out_dim]
    Tensor last_batch_output;  // [B × out_dim]
    // Adam state
    Tensor m_w, v_w, m_b, v_b;
    int adam_t = 0;

    // Synaptic Intelligence state
    Tensor omega_w, omega_b;              // per-parameter importance scores
    Tensor prev_w, prev_b;               // weights at last consolidation
    Tensor running_sum_w, running_sum_b; // accumulated path integral
    float si_lambda = 0.1f;             // SI penalty strength
    bool si_enabled = false;            // only active after parametrised construction

    // 2:4 Structured Sparsity
    std::vector<uint8_t> sparsity_mask; // 1=keep, 0=zero; empty = all kept
    bool sparsity_enabled = false;
    size_t mask_refresh_interval = 100; // steps between mask updates
    size_t sparsity_step = 0;

    Activation act;
    size_t in_dim, out_dim;

    DenseLayer() = default;
    DenseLayer(size_t in, size_t out, Activation activation, std::mt19937& rng)
        : act(activation), in_dim(in), out_dim(out) {
        weights = Tensor({out, in}); bias = Tensor({out});
        grad_w = Tensor({out, in}); grad_b = Tensor({out});
        m_w = Tensor({out, in}); v_w = Tensor({out, in});
        m_b = Tensor({out}); v_b = Tensor({out});
        (activation == Activation::RELU) ? weights.he_init(rng) : weights.xavier_init(rng);
        // Synaptic Intelligence initialisation
        omega_w = Tensor({out, in}); omega_b = Tensor({out});
        prev_w = weights;  // snapshot weights as consolidation reference
        prev_b = bias;
        running_sum_w = Tensor({out, in}); running_sum_b = Tensor({out});
        si_enabled = true;
    }

    Tensor forward(const Tensor& input) {
        last_input = input;
        last_pre_act = Tensor::matvec(weights, input, bias);
        switch (act) {
            case Activation::RELU:    last_output = last_pre_act.relu(); break;
            case Activation::SIGMOID: last_output = last_pre_act.sigmoid(); break;
            case Activation::SOFTMAX: last_output = last_pre_act.softmax(); break;
            case Activation::TANH:    last_output = last_pre_act.tanh_act(); break;
            default:                  last_output = last_pre_act; break;
        }
        return last_output;
    }

    Tensor backward(const Tensor& grad_out) {
        Tensor delta(grad_out.shape);
        switch (act) {
            case Activation::RELU: delta = grad_out * last_pre_act.relu_derivative(); break;
            case Activation::SIGMOID: {
                Tensor ones({last_output.size()}); ones.fill(1.0f);
                delta = grad_out * last_output * (ones - last_output);
                break;
            }
            case Activation::TANH: {
                Tensor sq(last_output.shape);
                for (size_t i = 0; i < sq.size(); i++) sq[i] = 1.0f - last_output[i] * last_output[i];
                delta = grad_out * sq;
                break;
            }
            default: delta = grad_out; break;
        }
        auto dw = Tensor::outer(delta, last_input);
        for (size_t i = 0; i < grad_w.size(); i++) grad_w[i] += dw[i];
        for (size_t i = 0; i < grad_b.size(); i++) grad_b[i] += delta[i];

        auto Wt = weights.T();
        Tensor grad_input({in_dim});
        for (size_t c = 0; c < in_dim; c++)
            grad_input[c] = Tensor::dot(&Wt.data[c * out_dim], delta.data.data(), out_dim);
        return grad_input;
    }

    // --------------------------------------------------------
    // Batched forward pass: input_batch [B × in_dim] → output [B × out_dim].
    // Caches input and pre-activation for backward_batch.
    // Uses Tensor::matmul_ABt (input[B×in] @ W.T[in×out]) + bias broadcast.
    // When DENDRITE_OPENCL is set and B*out_dim >= DENDRITE_GPU_MIN_ELEMS,
    // the matmul dispatches to GPU automatically.
    // --------------------------------------------------------
    Tensor forward_batch(const Tensor& input_batch) {
        const size_t B = input_batch.shape[0];
        // pre_act[B, out_dim] = input[B, in_dim] @ W.T[in_dim, out_dim]
        // W stored as [out_dim, in_dim] → W.T is [in_dim, out_dim]
        // matmul_AtB(W, input.T) gives [in_dim, B]... easier: matmul(input, W.T())
        // Use AVX2+OpenMP matmul; GPU dispatch handled inside matmul().
        Tensor W_t = weights.T();   // [in_dim, out_dim] — cheap at 64×64
        Tensor pre_act = Tensor::matmul(input_batch, W_t);  // [B, out_dim]
        // Add bias broadcast
        for (size_t b = 0; b < B; b++)
            for (size_t j = 0; j < out_dim; j++)
                pre_act.data[b * out_dim + j] += bias[j];
        last_batch_input   = input_batch;
        last_batch_pre_act = pre_act;
        Tensor output = pre_act;
        switch (act) {
            case Activation::RELU:
                for (auto& v : output.data) v = v > 0.f ? v : 0.f;
                break;
            case Activation::SIGMOID:
                for (auto& v : output.data) v = 1.f / (1.f + std::exp(-v));
                break;
            case Activation::TANH:
                for (auto& v : output.data) v = std::tanh(v);
                break;
            case Activation::SOFTMAX:
                for (size_t b = 0; b < B; b++) {
                    float* row = &output.data[b * out_dim];
                    float mx = *std::max_element(row, row + out_dim);
                    float s = 0.f;
                    for (size_t j = 0; j < out_dim; j++) { row[j] = std::exp(row[j] - mx); s += row[j]; }
                    if (s > 1e-7f) for (size_t j = 0; j < out_dim; j++) row[j] /= s;
                }
                break;
            default: break;
        }
        for (auto& v : output.data) if (!std::isfinite(v)) v = 0.f;
        last_batch_output = output;
        return output;
    }

    // --------------------------------------------------------
    // Batched backward pass: grad_batch [B × out_dim] → grad_input [B × in_dim].
    // Accumulates grad_w and grad_b averaged over the batch.
    // --------------------------------------------------------
    Tensor backward_batch(const Tensor& grad_batch) {
        const size_t B = grad_batch.shape[0];
        const float inv_B = 1.f / static_cast<float>(B);
        Tensor delta = grad_batch;
        switch (act) {
            case Activation::RELU:
                for (size_t i = 0; i < delta.data.size(); i++)
                    delta.data[i] *= last_batch_pre_act.data[i] > 0.f ? 1.f : 0.f;
                break;
            case Activation::SIGMOID:
                for (size_t i = 0; i < delta.data.size(); i++) {
                    const float s = last_batch_output.data[i];
                    delta.data[i] *= s * (1.f - s);
                }
                break;
            case Activation::TANH:
                for (size_t i = 0; i < delta.data.size(); i++) {
                    const float s = last_batch_output.data[i];
                    delta.data[i] *= 1.f - s * s;
                }
                break;
            default: break; // NONE, SOFTMAX: delta = grad (cross-entropy absorbs softmax Jacobian)
        }
        // grad_w += inv_B * delta.T @ input  →  matmul_AtB(delta[B,out], input[B,in]) = [out,in]
        // GPU dispatch inside matmul_AtB when DENDRITE_OPENCL set and out*in >= threshold.
        {
            Tensor dw = Tensor::matmul_AtB(delta, last_batch_input);  // [out_dim, in_dim]
            for (size_t i = 0; i < dw.data.size(); i++)
                if (std::isfinite(dw.data[i])) grad_w.data[i] += dw.data[i] * inv_B;
        }
        // grad_b += inv_B * col_sum(delta)
        for (size_t b = 0; b < B; b++)
            for (size_t j = 0; j < out_dim; j++) {
                const float dv = delta.data[b * out_dim + j] * inv_B;
                if (std::isfinite(dv)) grad_b[j] += dv;
            }
        // grad_input = delta[B,out] @ weights[out,in]  →  standard matmul → [B,in]
        Tensor grad_input = Tensor::matmul(delta, weights);  // [B, in_dim]
        for (auto& v : grad_input.data) if (!std::isfinite(v)) v = 0.0f;
        return grad_input;
    }

    // --------------------------------------------------------
    // Checkpoint serialization
    // --------------------------------------------------------
    void serialize(CheckpointWriter& wr, const std::string& prefix) const {
        wr.add(prefix + "weights", weights);
        wr.add(prefix + "bias",    bias);
        wr.add(prefix + "mw",      m_w);
        wr.add(prefix + "vw",      v_w);
        wr.add(prefix + "mb",      m_b);
        wr.add(prefix + "vb",      v_b);
        wr.add_scalar(prefix + "t", static_cast<float>(adam_t));
        if (si_enabled) {
            wr.add(prefix + "omega_w", omega_w);
            wr.add(prefix + "omega_b", omega_b);
        }
    }

    void deserialize(const CheckpointReader& rd, const std::string& prefix) {
        rd.restore(prefix + "weights", weights);
        rd.restore(prefix + "bias",    bias);
        rd.restore(prefix + "mw",      m_w);
        rd.restore(prefix + "vw",      v_w);
        rd.restore(prefix + "mb",      m_b);
        rd.restore(prefix + "vb",      v_b);
        float t_f = 0.f;
        if (rd.restore_scalar(prefix + "t", t_f)) adam_t = static_cast<int>(t_f);
        if (si_enabled) {
            rd.restore(prefix + "omega_w", omega_w);
            rd.restore(prefix + "omega_b", omega_b);
        }
    }

    void apply_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t++;
        // Sparsity: zero masked gradients; refresh mask periodically
        if (sparsity_enabled) {
            sparsity_step++;
            apply_sparsity_to_grad();
            if (mask_refresh_interval > 0 && sparsity_step % mask_refresh_interval == 0)
                update_sparsity_mask();
        }
        float bc1 = 1.0f - std::pow(beta1, adam_t);
        float bc2 = 1.0f - std::pow(beta2, adam_t);
        for (size_t i = 0; i < weights.size(); i++) {
            if (si_enabled) {
                // Accumulate path integral: -grad * displacement (before penalty modifies grad)
                running_sum_w[i] += -grad_w[i] * (weights[i] - prev_w[i]);
                // SI penalty: pull toward anchor for important weights
                grad_w[i] += si_lambda * omega_w[i] * (weights[i] - prev_w[i]);
                if (!std::isfinite(grad_w[i])) grad_w[i] = 0.0f;
            }
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w[i];
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * grad_w[i] * grad_w[i];
            weights[i] -= lr * (m_w[i] / bc1) / (std::sqrt(v_w[i] / bc2) + eps);
        }
        for (size_t i = 0; i < bias.size(); i++) {
            if (si_enabled) {
                running_sum_b[i] += -grad_b[i] * (bias[i] - prev_b[i]);
                grad_b[i] += si_lambda * omega_b[i] * (bias[i] - prev_b[i]);
                if (!std::isfinite(grad_b[i])) grad_b[i] = 0.0f;
            }
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b[i];
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * grad_b[i] * grad_b[i];
            bias[i] -= lr * (m_b[i] / bc1) / (std::sqrt(v_b[i] / bc2) + eps);
        }
        grad_w.zero(); grad_b.zero();
    }

    /// Consolidate accumulated path integrals into importance scores.
    /// Call once per epoch (or per task boundary) after training samples complete.
    void consolidate_importance() {
        if (!si_enabled) return;
        for (size_t i = 0; i < weights.size(); i++) {
            float delta = weights[i] - prev_w[i];
            float denom = delta * delta + 1e-8f;
            omega_w[i] += std::max(0.0f, running_sum_w[i]) / denom;
            omega_w[i] = std::min(omega_w[i], 10.0f);  // cap to prevent freezing
            if (!std::isfinite(omega_w[i])) omega_w[i] = 0.0f;
            running_sum_w[i] = 0.0f;
            prev_w[i] = weights[i];
        }
        for (size_t i = 0; i < bias.size(); i++) {
            float delta = bias[i] - prev_b[i];
            float denom = delta * delta + 1e-8f;
            omega_b[i] += std::max(0.0f, running_sum_b[i]) / denom;
            omega_b[i] = std::min(omega_b[i], 10.0f);
            if (!std::isfinite(omega_b[i])) omega_b[i] = 0.0f;
            running_sum_b[i] = 0.0f;
            prev_b[i] = bias[i];
        }
    }

    size_t param_count() const { return weights.size() + bias.size(); }

    // --------------------------------------------------------
    // 2:4 Structured Sparsity: keep 2 largest-magnitude weights per group of 4.
    // Call periodically after Adam updates (not during warmup).
    // --------------------------------------------------------
    void update_sparsity_mask() {
        if (!sparsity_enabled) return;
        size_t total = weights.size();
        sparsity_mask.assign(total, 1);
        // Process weights in groups of 4 (row-major, as flat vector)
        for (size_t i = 0; i + 3 < total; i += 4) {
            // Find top-2 by magnitude within group
            std::array<size_t, 4> idx = {i, i+1, i+2, i+3};
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                return std::abs(weights[a]) > std::abs(weights[b]);
            });
            // Zero out the 2 smallest
            sparsity_mask[idx[2]] = 0; weights[idx[2]] = 0.0f;
            sparsity_mask[idx[3]] = 0; weights[idx[3]] = 0.0f;
        }
    }

    // Apply sparsity mask to gradients: block gradient through zeroed weights
    void apply_sparsity_to_grad() {
        if (!sparsity_enabled || sparsity_mask.empty()) return;
        for (size_t i = 0; i < grad_w.size() && i < sparsity_mask.size(); i++) {
            if (!sparsity_mask[i]) grad_w[i] = 0.0f;
        }
    }

    // --------------------------------------------------------
    // Resize: add one output neuron (new row in weights, new element in bias)
    // --------------------------------------------------------
    void add_output(std::mt19937& rng, float noise_std = 0.01f) {
        std::normal_distribution<float> dist(0.0f, noise_std);
        auto grow_2d = [&](Tensor& t) {
            Tensor nt({out_dim + 1, in_dim});
            for (size_t i = 0; i < out_dim * in_dim; i++) nt[i] = t[i];
            for (size_t j = 0; j < in_dim; j++) nt[out_dim * in_dim + j] = dist(rng);
            t = nt;
        };
        auto grow_1d = [&](Tensor& t) {
            Tensor nt({out_dim + 1});
            for (size_t i = 0; i < out_dim; i++) nt[i] = t[i];
            nt[out_dim] = 0.0f;
            t = nt;
        };
        auto grow_2d_zero = [&](Tensor& t) {
            Tensor nt({out_dim + 1, in_dim});
            for (size_t i = 0; i < out_dim * in_dim; i++) nt[i] = t[i];
            t = nt;
        };
        grow_2d(weights);
        grow_1d(bias);
        grow_2d_zero(grad_w);      grow_1d(grad_b);
        grow_2d_zero(m_w);         grow_1d(m_b);
        grow_2d_zero(v_w);         grow_1d(v_b);
        grow_2d_zero(omega_w);     grow_1d(omega_b);
        grow_2d_zero(prev_w);      grow_1d(prev_b);
        grow_2d_zero(running_sum_w); grow_1d(running_sum_b);
        out_dim++;
    }

    // --------------------------------------------------------
    // Resize: remove one output neuron (row `row` from weights, element from bias)
    // --------------------------------------------------------
    void remove_output(size_t row) {
        if (out_dim <= 1 || row >= out_dim) return;
        size_t new_out = out_dim - 1;
        auto shrink_2d = [&](Tensor& t) {
            Tensor nt({new_out, in_dim});
            for (size_t i = 0, ni = 0; i < out_dim; i++) {
                if (i == row) continue;
                for (size_t j = 0; j < in_dim; j++)
                    nt[ni * in_dim + j] = t[i * in_dim + j];
                ni++;
            }
            t = nt;
        };
        auto shrink_1d = [&](Tensor& t) {
            Tensor nt({new_out});
            for (size_t i = 0, ni = 0; i < out_dim; i++) {
                if (i == row) continue;
                nt[ni++] = t[i];
            }
            t = nt;
        };
        shrink_2d(weights);        shrink_1d(bias);
        shrink_2d(grad_w);         shrink_1d(grad_b);
        shrink_2d(m_w);            shrink_1d(m_b);
        shrink_2d(v_w);            shrink_1d(v_b);
        shrink_2d(omega_w);        shrink_1d(omega_b);
        shrink_2d(prev_w);         shrink_1d(prev_b);
        shrink_2d(running_sum_w);  shrink_1d(running_sum_b);
        out_dim = new_out;
    }
};

/// Lightweight classifier attached at an intermediate layer for early exit during inference.
/// Uses Shannon entropy gating: exits only when output entropy is very low (genuinely peaked).
struct EarlyExitClassifier {
    DenseLayer classifier;    // hidden_dim → output_dim
    float entropy_ceiling = 0.4f;  // exit when entropy < this (nats)
    bool enabled = true;

    struct ExitResult {
        Tensor output;       // softmax probabilities
        float entropy;       // Shannon entropy (lower = more confident)
        float confidence;    // max probability (for reporting)
        bool should_exit;    // true if entropy < ceiling
    };

    EarlyExitClassifier() = default;
    EarlyExitClassifier(size_t hidden_dim, size_t output_dim, std::mt19937& rng,
                        float entropy_ceil = 0.4f)
        : entropy_ceiling(entropy_ceil), enabled(true) {
        classifier = DenseLayer(hidden_dim, output_dim, Activation::NONE, rng);
    }

    ExitResult evaluate(const Tensor& hidden) {
        Tensor logits = classifier.forward(hidden);
        Tensor probs = logits.softmax();
        // Shannon entropy: H = -sum(p * log(p))
        float H = 0.0f;
        float max_p = 0.0f;
        for (size_t i = 0; i < probs.size(); i++) {
            max_p = std::max(max_p, probs[i]);
            if (probs[i] > 1e-7f)
                H -= probs[i] * std::log(probs[i]);
        }
        if (!std::isfinite(H)) H = 10.0f;  // high entropy = don't exit
        return {probs, H, max_p, enabled && H < entropy_ceiling};
    }

    Tensor backward(const Tensor& grad) {
        return classifier.backward(grad);
    }
    void apply_adam(float lr) { classifier.apply_adam(lr); }
    size_t param_count() const { return classifier.param_count(); }
};

struct MiniNetwork {
    std::vector<DenseLayer> layers;
    std::string name;
    std::optional<EarlyExitClassifier> exit_classifier;
    bool last_exited_early = false;
    Tensor last_exit_hidden;  // cached hidden state at exit point for auxiliary loss

    // Enhancement #30: Adaptive branch depth (per-layer Straight-Through gates)
    bool adaptive_depth = false;
    std::vector<DenseLayer> depth_gates;  // one per intermediate layer (not last)
    size_t last_exit_layer_idx = 0;       // index of layer that triggered early exit

    MiniNetwork() = default;
    MiniNetwork(const std::string& name_, const std::vector<size_t>& sizes,
                Activation hidden_act, Activation out_act, std::mt19937& rng) : name(name_) {
        for (size_t i = 0; i + 1 < sizes.size(); i++) {
            Activation a = (i + 2 == sizes.size()) ? out_act : hidden_act;
            layers.emplace_back(sizes[i], sizes[i + 1], a, rng);
        }
    }

    /// Enhancement #30: Enable per-layer depth gates for adaptive early exit.
    /// Creates one lightweight gate (hidden→1) per intermediate layer.
    void enable_adaptive_depth(std::mt19937& rng) {
        if (layers.size() < 2) return;
        adaptive_depth = true;
        depth_gates.clear();
        for (size_t i = 0; i + 1 < layers.size(); i++) {
            size_t hidden = layers[i].out_dim;
            depth_gates.emplace_back(hidden, 1, Activation::NONE, rng);
        }
    }

    /// Initialize early exit classifier at layer index 1 (midpoint of a 4-layer network).
    void init_early_exit(size_t output_dim_, std::mt19937& rng, float threshold = 0.85f) {
        if (layers.size() < 3) return;
        size_t mid_dim = layers[1].out_dim;
        exit_classifier.emplace(mid_dim, output_dim_, rng, threshold);
    }

    /// Inference forward pass. When allow_exit=true, may skip remaining layers on confident predictions.
    Tensor forward(const Tensor& input, bool allow_exit = false) {
        Tensor x = input;
        last_exited_early = false;

        for (size_t i = 0; i < layers.size(); i++) {
            x = layers[i].forward(x);

            if (i == 1 && allow_exit && exit_classifier.has_value() && exit_classifier->enabled) {
                last_exit_hidden = x;
                auto result = exit_classifier->evaluate(x);
                if (result.should_exit) {
                    last_exited_early = true;
                    return result.output;
                }
            }
        }
        return x;
    }

    /// Enhancement #30: Adaptive-depth forward pass.
    /// Inference: exits after layer i if depth gate < 0.5, applying last layer as skip projection.
    /// Training: all layers always execute (Straight-Through Estimator).
    Tensor forward_adaptive(const Tensor& input, bool training = false) {
        if (!adaptive_depth || depth_gates.empty()) return forward(input);

        Tensor x = input;
        last_exited_early = false;
        last_exit_layer_idx = layers.size() - 1;

        for (size_t i = 0; i < layers.size(); i++) {
            x = layers[i].forward(x);

            // Gate check: intermediate layers only (not on last layer)
            if (!training && i + 1 < layers.size() && i < depth_gates.size()) {
                Tensor g = depth_gates[i].forward(x);
                float raw = g.size() == 0 ? 1.0f : g[0];
                raw = std::clamp(raw, -20.0f, 20.0f);
                float cont = 1.0f / (1.0f + std::exp(-raw));
                if (!std::isfinite(cont)) cont = 1.0f;

                if (cont < 0.5f) {
                    last_exit_layer_idx = i;
                    // Skip projection: apply final layer to map hidden→output_dim
                    Tensor out = layers.back().forward(x);
                    for (auto& v : out.data) if (!std::isfinite(v)) v = 0.0f;
                    return out;
                }
            }
        }
        return x;
    }

    /// Full forward pass — always runs all layers. Used during training.
    Tensor forward_full(const Tensor& input) {
        Tensor x = input;
        last_exited_early = false;
        for (size_t i = 0; i < layers.size(); i++) {
            x = layers[i].forward(x);
            if (i == 1 && exit_classifier.has_value()) {
                last_exit_hidden = x;
            }
        }
        return x;
    }

    Tensor backward(const Tensor& grad) {
        Tensor g = grad;
        for (int i = layers.size() - 1; i >= 0; i--) g = layers[i].backward(g);
        return g;
    }

    // --------------------------------------------------------
    // Batched forward: input_batch [B × in_dim] → output [B × out_dim].
    // Runs all layers in sequence using their forward_batch methods.
    // --------------------------------------------------------
    Tensor forward_batch(const Tensor& input_batch) {
        Tensor x = input_batch;
        for (auto& layer : layers) x = layer.forward_batch(x);
        return x;
    }

    // --------------------------------------------------------
    // Batched backward: grad_batch [B × out_dim] → grad_input [B × in_dim].
    // Accumulates gradients averaged over the batch into each layer.
    // --------------------------------------------------------
    Tensor backward_batch(const Tensor& grad_batch) {
        Tensor g = grad_batch;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
            g = layers[i].backward_batch(g);
        return g;
    }

    // --------------------------------------------------------
    // Checkpoint: serialize all layers with prefix "prefixL<i>_"
    // --------------------------------------------------------
    void serialize(CheckpointWriter& wr, const std::string& prefix) const {
        for (size_t i = 0; i < layers.size(); i++)
            layers[i].serialize(wr, prefix + "L" + std::to_string(i) + "_");
    }

    void deserialize(const CheckpointReader& rd, const std::string& prefix) {
        for (size_t i = 0; i < layers.size(); i++)
            layers[i].deserialize(rd, prefix + "L" + std::to_string(i) + "_");
    }

    void apply_adam(float lr) {
        for (auto& l : layers) l.apply_adam(lr);
        for (auto& g : depth_gates) g.apply_adam(lr);  // Enhancement #30
    }
    void consolidate_all() { for (auto& l : layers) l.consolidate_importance(); }

    // Add one output neuron to the last layer (for branch growing)
    void add_output(std::mt19937& rng, float noise_std = 0.01f) {
        if (!layers.empty()) layers.back().add_output(rng, noise_std);
    }
    // Remove output row from the last layer (for branch pruning)
    void remove_output(size_t row) {
        if (!layers.empty()) layers.back().remove_output(row);
    }
    size_t output_size() const {
        return layers.empty() ? 0 : layers.back().out_dim;
    }

    size_t param_count() const {
        size_t t = 0;
        for (auto& l : layers) t += l.param_count();
        if (exit_classifier.has_value()) t += exit_classifier->param_count();
        for (auto& g : depth_gates) t += g.param_count();
        return t;
    }
};

/// Active dendrite layer: each neuron has N dendritic segments that receive
/// context input. The max-activation segment modulates the feedforward output,
/// making each neuron a conditional function of both input and context.
/// Implements Numenta's Active Dendrites (Iyer et al. 2022).
struct DendriticLayer {
    size_t input_dim, output_dim;
    size_t num_segments;      // dendritic segments per neuron (e.g., 7)
    size_t context_dim;       // dimension of context vector
    float kwta_percent;       // fraction of neurons active (e.g., 0.2 = top 20%)

    // Feedforward weights (standard)
    Tensor weights;           // [output_dim x input_dim]
    Tensor bias;              // [output_dim]
    Tensor grad_w, grad_b;

    // Dendritic segment weights: [output_dim * num_segments * context_dim]
    Tensor dendrite_weights;
    Tensor grad_dw;

    // Forward pass cache
    Tensor last_input, last_context;
    Tensor last_ff;           // feedforward output before modulation
    Tensor last_modulation;   // tanh(max_segment_activation)
    Tensor last_output;       // final output after kWTA
    std::vector<size_t> last_active_mask;  // kWTA active neuron indices

    // Adam state for feedforward weights
    Tensor m_w, v_w, m_b, v_b;
    // Adam state for dendritic weights
    Tensor m_dw, v_dw;
    int adam_t = 0;

    // Synaptic Intelligence state (mirrors DenseLayer pattern)
    Tensor omega_w, omega_b, omega_dw;
    Tensor prev_w_si, prev_b_si, prev_dw_si;
    Tensor running_sum_w, running_sum_b, running_sum_dw;
    float si_lambda = 0.1f;
    bool si_enabled = false;

    DendriticLayer() = default;
    DendriticLayer(size_t in, size_t out, size_t ctx_dim, std::mt19937& rng,
                   size_t segments = 7, float kwta = 0.2f)
        : input_dim(in), output_dim(out), num_segments(segments),
          context_dim(ctx_dim), kwta_percent(kwta) {
        weights = Tensor({out, in}); weights.he_init(rng);
        bias = Tensor({out});
        grad_w = Tensor({out, in}); grad_b = Tensor({out});
        m_w = Tensor({out, in}); v_w = Tensor({out, in});
        m_b = Tensor({out}); v_b = Tensor({out});

        size_t dw_size = out * segments * ctx_dim;
        dendrite_weights = Tensor({dw_size});
        dendrite_weights.xavier_init(rng);
        // Full xavier init (no 0.1x scale — avoid gradient starvation in tanh gate)

        grad_dw = Tensor({dw_size});
        m_dw = Tensor({dw_size}); v_dw = Tensor({dw_size});

        omega_w = Tensor({out, in}); omega_b = Tensor({out}); omega_dw = Tensor({dw_size});
        prev_w_si = weights; prev_b_si = bias; prev_dw_si = dendrite_weights;
        running_sum_w = Tensor({out, in}); running_sum_b = Tensor({out});
        running_sum_dw = Tensor({dw_size});
        si_enabled = true;
    }

    Tensor forward(const Tensor& input, const Tensor& context) {
        last_input = input;
        last_context = context;
        last_ff = Tensor::matvec(weights, input, bias);

        last_modulation = Tensor({output_dim});
        for (size_t n = 0; n < output_dim; n++) {
            float max_act = -1e30f;
            for (size_t s = 0; s < num_segments; s++) {
                float act = 0.0f;
                size_t base = (n * num_segments + s) * context_dim;
                size_t cdim = std::min(context_dim, context.size());
                for (size_t c = 0; c < cdim; c++)
                    act += dendrite_weights[base + c] * context[c];
                max_act = std::max(max_act, act);
            }
            last_modulation[n] = std::tanh(max_act);
            if (!std::isfinite(last_modulation[n])) last_modulation[n] = 0.0f;
        }

        Tensor pre_kwta({output_dim});
        for (size_t n = 0; n < output_dim; n++) {
            pre_kwta[n] = last_ff[n] * (1.0f + last_modulation[n]);
            if (!std::isfinite(pre_kwta[n])) pre_kwta[n] = 0.0f;
        }

        last_output = pre_kwta;
        last_active_mask.clear();
        size_t k = std::max((size_t)1, (size_t)(output_dim * kwta_percent));
        std::vector<float> sorted_vals(output_dim);
        for (size_t i = 0; i < output_dim; i++) sorted_vals[i] = std::abs(pre_kwta[i]);
        std::partial_sort(sorted_vals.begin(), sorted_vals.begin() + k, sorted_vals.end(),
                          std::greater<float>());
        float threshold = sorted_vals[std::min(k - 1, output_dim - 1)];
        for (size_t i = 0; i < output_dim; i++) {
            if (std::abs(pre_kwta[i]) >= threshold && last_active_mask.size() < k)
                last_active_mask.push_back(i);
            else
                last_output[i] = 0.0f;
        }
        return last_output;
    }

    Tensor backward(const Tensor& grad_out) {
        Tensor effective_grad({output_dim});
        std::vector<bool> is_active(output_dim, false);
        for (size_t idx : last_active_mask) is_active[idx] = true;
        for (size_t i = 0; i < output_dim; i++)
            effective_grad[i] = is_active[i] ? grad_out[i] : 0.0f;

        Tensor grad_ff({output_dim});
        for (size_t i = 0; i < output_dim; i++)
            grad_ff[i] = effective_grad[i] * (1.0f + last_modulation[i]);

        auto dw = Tensor::outer(grad_ff, last_input);
        for (size_t i = 0; i < grad_w.size(); i++) grad_w[i] += dw[i];
        for (size_t i = 0; i < grad_b.size(); i++) grad_b[i] += grad_ff[i];

        for (size_t n = 0; n < output_dim; n++) {
            if (!is_active[n]) continue;
            float grad_mod = effective_grad[n] * last_ff[n];
            float dtanh = 1.0f - last_modulation[n] * last_modulation[n];
            float grad_pre_tanh = grad_mod * dtanh;
            if (!std::isfinite(grad_pre_tanh)) continue;

            float max_act = -1e30f;
            size_t best_seg = 0;
            for (size_t s = 0; s < num_segments; s++) {
                float act = 0.0f;
                size_t base = (n * num_segments + s) * context_dim;
                size_t cdim = std::min(context_dim, last_context.size());
                for (size_t c = 0; c < cdim; c++)
                    act += dendrite_weights[base + c] * last_context[c];
                if (act > max_act) { max_act = act; best_seg = s; }
            }
            size_t base = (n * num_segments + best_seg) * context_dim;
            size_t cdim = std::min(context_dim, last_context.size());
            for (size_t c = 0; c < cdim; c++) {
                grad_dw[base + c] += grad_pre_tanh * last_context[c];
                if (!std::isfinite(grad_dw[base + c])) grad_dw[base + c] = 0.0f;
            }
        }

        Tensor grad_input({input_dim});
        for (size_t j = 0; j < input_dim; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < output_dim; i++)
                sum += grad_ff[i] * weights.data[i * input_dim + j];
            grad_input[j] = std::isfinite(sum) ? sum : 0.0f;
        }
        return grad_input;
    }

    void apply_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t++;
        float bc1 = 1.0f - std::pow(beta1, adam_t);
        float bc2 = 1.0f - std::pow(beta2, adam_t);
        for (size_t i = 0; i < weights.size(); i++) {
            if (si_enabled) {
                running_sum_w[i] += -grad_w[i] * (weights[i] - prev_w_si[i]);
                grad_w[i] += si_lambda * omega_w[i] * (weights[i] - prev_w_si[i]);
                if (!std::isfinite(grad_w[i])) grad_w[i] = 0.0f;
            }
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w[i];
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * grad_w[i] * grad_w[i];
            weights[i] -= lr * (m_w[i] / bc1) / (std::sqrt(v_w[i] / bc2) + eps);
            if (!std::isfinite(weights[i])) weights[i] = 0.0f;
        }
        for (size_t i = 0; i < bias.size(); i++) {
            if (si_enabled) {
                running_sum_b[i] += -grad_b[i] * (bias[i] - prev_b_si[i]);
                grad_b[i] += si_lambda * omega_b[i] * (bias[i] - prev_b_si[i]);
                if (!std::isfinite(grad_b[i])) grad_b[i] = 0.0f;
            }
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b[i];
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * grad_b[i] * grad_b[i];
            bias[i] -= lr * (m_b[i] / bc1) / (std::sqrt(v_b[i] / bc2) + eps);
            if (!std::isfinite(bias[i])) bias[i] = 0.0f;
        }
        for (size_t i = 0; i < dendrite_weights.size(); i++) {
            if (si_enabled) {
                running_sum_dw[i] += -grad_dw[i] * (dendrite_weights[i] - prev_dw_si[i]);
                grad_dw[i] += si_lambda * omega_dw[i] * (dendrite_weights[i] - prev_dw_si[i]);
                if (!std::isfinite(grad_dw[i])) grad_dw[i] = 0.0f;
            }
            m_dw[i] = beta1 * m_dw[i] + (1 - beta1) * grad_dw[i];
            v_dw[i] = beta2 * v_dw[i] + (1 - beta2) * grad_dw[i] * grad_dw[i];
            dendrite_weights[i] -= lr * (m_dw[i] / bc1) / (std::sqrt(v_dw[i] / bc2) + eps);
            if (!std::isfinite(dendrite_weights[i])) dendrite_weights[i] = 0.0f;
        }
        grad_w.zero(); grad_b.zero(); grad_dw.zero();
    }

    void consolidate_importance() {
        if (!si_enabled) return;
        auto consolidate_vec = [](Tensor& w, Tensor& omega, Tensor& prev, Tensor& running) {
            for (size_t i = 0; i < w.size(); i++) {
                float delta = w[i] - prev[i];
                float denom = delta * delta + 1e-8f;
                omega[i] += std::max(0.0f, running[i]) / denom;
                omega[i] = std::min(omega[i], 10.0f);  // cap to prevent freezing
                if (!std::isfinite(omega[i])) omega[i] = 0.0f;
                running[i] = 0.0f;
                prev[i] = w[i];
            }
        };
        consolidate_vec(weights, omega_w, prev_w_si, running_sum_w);
        consolidate_vec(bias, omega_b, prev_b_si, running_sum_b);
        consolidate_vec(dendrite_weights, omega_dw, prev_dw_si, running_sum_dw);
    }

    size_t param_count() const {
        return weights.size() + bias.size() + dendrite_weights.size();
    }
};

} // namespace dendrite
