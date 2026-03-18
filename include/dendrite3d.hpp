#pragma once
#include "branch.hpp"
#include "conductor.hpp"
#include "task_context.hpp"
#include "morality.hpp"
#include "data_cleaner.hpp"
#include "modality.hpp"
#include "text_preprocessor.hpp"
#include "hypernetwork.hpp"
#include "perceiver_fusion.hpp"
#include "model_config.hpp"
#include <memory>
#include <chrono>
#include <unordered_map>
#include <set>

namespace dendrite {

// ============================================================
// SpecializationMetrics: branch-class activation statistics
// Records which branches activate for which classes, then computes
// Mutual Information and per-branch Gini coefficients.
// ============================================================
struct SpecializationMetrics {
    std::vector<std::vector<size_t>> branch_class_counts; // [branch][class]
    std::vector<size_t> branch_totals;
    std::vector<size_t> class_totals;
    size_t num_branches = 0, num_classes = 0;

    void init(size_t nb, size_t nc) {
        num_branches = nb; num_classes = nc;
        branch_class_counts.assign(nb, std::vector<size_t>(nc, 0));
        branch_totals.assign(nb, 0);
        class_totals.assign(nc, 0);
    }

    // Record which of the top-k branches (by heat) activate for this sample.
    // Uses rank rather than a flat threshold so MI is meaningful even when
    // NMDA pushes all heat values similarly high.
    void record(const Tensor& heat, size_t true_class, int top_k = 2) {
        if (true_class >= num_classes) return;
        class_totals[true_class]++;
        // Collect and rank branches by heat
        std::vector<std::pair<float, size_t>> ranked;
        for (size_t b = 0; b < num_branches && b < heat.size(); b++)
            ranked.push_back({heat[b], b});
        std::sort(ranked.rbegin(), ranked.rend());
        for (int i = 0; i < top_k && i < (int)ranked.size(); i++) {
            size_t b = ranked[i].second;
            branch_class_counts[b][true_class]++;
            branch_totals[b]++;
        }
    }

    // Mutual Information: I(branch_selection; class) — higher = more specialised
    [[nodiscard]] float mutual_information() const {
        float mi = 0.0f;
        size_t total = 0;
        for (auto t : class_totals) total += t;
        if (total == 0) return 0.0f;
        for (size_t b = 0; b < num_branches; b++) {
            for (size_t c = 0; c < num_classes; c++) {
                if (branch_class_counts[b][c] == 0) continue;
                float p_bc = (float)branch_class_counts[b][c] / total;
                float p_b  = (float)branch_totals[b]  / total;
                float p_c  = (float)class_totals[c]   / total;
                if (p_b > 0 && p_c > 0)
                    mi += p_bc * std::log(p_bc / (p_b * p_c));
            }
        }
        return std::isfinite(mi) ? mi : 0.0f;
    }

    // Per-branch Gini impurity (0 = maximally specialised, high = handles all classes)
    [[nodiscard]] std::vector<float> branch_gini() const {
        std::vector<float> ginis(num_branches, 0.0f);
        for (size_t b = 0; b < num_branches; b++) {
            if (branch_totals[b] == 0) continue;
            float sum_sq = 0.0f;
            for (size_t c = 0; c < num_classes; c++) {
                float p = (float)branch_class_counts[b][c] / branch_totals[b];
                sum_sq += p * p;
            }
            ginis[b] = 1.0f - sum_sq;
        }
        return ginis;
    }

    void report() const {
        printf("  Specialization MI=%.4f\n", mutual_information());
        auto ginis = branch_gini();
        for (size_t b = 0; b < num_branches; b++)
            printf("    Branch %zu: gini=%.3f  visits=%zu\n", b, ginis[b], branch_totals[b]);
    }
};

// ============================================================
// InferenceResult3D: rich trace of what happened
// ============================================================
struct InferenceResult3D {
    Tensor output;
    Tensor heat_scores;
    FusionStrategy strategy_used;
    std::vector<int> active_branch_ids;
    std::vector<float> branch_confidences;
    std::vector<bool> branch_redirects;
    int redirect_count;
    float total_confidence;
    int branches_evaluated;
    double time_us;

    void print() const {
        std::cout << "  Strategy: " << fusion_name(strategy_used) << "\n";
        std::cout << "  Active branches: ";
        for (size_t i = 0; i < active_branch_ids.size(); i++) {
            printf("%d(heat=%.2f,conf=%.2f%s) ",
                   active_branch_ids[i], heat_scores[active_branch_ids[i]],
                   branch_confidences[i],
                   branch_redirects[i] ? ",REDIR" : "");
        }
        std::cout << "\n  Branches evaluated: " << branches_evaluated
                  << "  Redirects: " << redirect_count
                  << "  Context items: " << context_items_active
                  << "  Morality: " << (morality_triggered.empty() ? "clear" : morality_triggered)
                  << "  Time: " << time_us << "us\n";
    }

    // v3 additions
    int context_items_active = 0;
    std::string morality_triggered;
    std::vector<std::string> modalities_activated;

    // Concept bottleneck: interpretable routing concepts (empty if bottleneck disabled)
    Tensor concept_scores;
};

// ============================================================
// DendriteNet3D: The full 3D interconnected neural tree
// ============================================================
class DendriteNet3D {
public:
    Conductor conductor;
    std::vector<BranchPtr> branches;
    std::mt19937 rng;

    size_t input_dim, output_dim;
    float learning_rate   = 0.003f;
    float lr_warmup_steps = 500.0f;   // steps over which LR ramps from lr_min to learning_rate
    float lr_min          = 1e-5f;    // LR floor during warmup and decay
    float lr_decay_steps  = 108000.0f;  // decay horizon: cosine first half + exp second half (~45 epochs)
    float heat_threshold  = 0.15f;
    size_t summary_dim = 16;
    size_t attn_dim = 16;
    int top_k = 3;

    // === v3: New components ===
    TaskContext task_context;
    MoralityLayer morality;
    DataCleaner data_cleaner;
    TextPreprocessor text_preprocessor;

    // Multimodal modules (optional)
    ModalityModule image_module;
    ModalityModule audio_module;
    bool image_enabled = false;
    bool audio_enabled = false;

    // Gated cross-attention: fuses modality embeddings into branch outputs
    GatedCrossAttention gated_cross_attn;
    bool gca_enabled = false;

    // Enhancement #24: Perceiver IO — modality-agnostic latent fusion
    PerceiverIO perceiver;
    bool perceiver_enabled = false;

    // Context integration dimension
    size_t context_dim = 64;

    // Active dendrite modulation layer (applies before each branch specialist)
    DendriticLayer dendrite_layer;
    bool has_dendrite_layer = false;

    // Lateral inhibition state
    std::vector<CrossTalkMessage> prev_cross_talk;   // cached from previous step
    size_t lateral_inhibition_burnin = 200;           // steps before inhibition activates

    // Shared expert branch (always-on, bypasses heat gating)
    MiniNetwork shared_expert;   // always-on expert
    DenseLayer alpha_gate;       // learned mixing weight: sigmoid → [0,1]
    bool has_shared_expert = false;
    size_t exit_enabled_after_steps = 0;  // early exit activates after this many train steps

    // Output history for dendritic context (running mean of recent predictions)
    Tensor output_history;           // EMA of recent output distributions
    float output_history_alpha = 0.15f;  // EMA decay rate

    // Branch growing/pruning
    GrowthController growth_controller;
    int specialist_hidden_cached = 64;

    // Per-task output heads for continual learning
    // When active, each branch specialist's last layer is bypassed and task_heads[task_id] is used
    std::unordered_map<int, std::vector<DenseLayer>> task_heads;  // task_id → per-branch output layers
    int current_task_id = -1;  // -1 = single-task mode (use specialist's own output layer)

    // Learned task embeddings for dendritic context (stronger than output history)
    std::vector<Tensor> task_embeddings;  // per-task learned embedding vectors

    // VICReg EMA tracking for batch-free statistics
    std::vector<Tensor> branch_ema_mean;   // per-branch exponential moving average of output means
    std::vector<Tensor> branch_ema_var;    // per-branch exponential moving average of output variances
    float vicreg_ema_alpha = 0.1f;         // EMA decay rate
    size_t vicreg_warmup = 500;            // steps before VICReg activates
    float vicreg_weight = 0.1f;            // weight of VICReg loss relative to main loss

    // Enhancement 14: cumulative topology change count
    size_t topology_changes = 0;

    // Enhancement #26: Hypernetwork branch generation
    BranchGenerator branch_generator;
    std::unordered_map<int, DomainEmbedding> domain_embeds;  // branch_id → embedding
    bool hypernetwork_enabled = false;
    float hypernetwork_meta_weight = 0.01f;  // weight of auxiliary meta-loss

    // Enhancement #25: track whether the last infer() matched any text→concept association.
    // Cleared at the start of every train_sample() — training has no text input.
    bool modality_concepts_active = false;

    // Stats
    size_t total_inferences = 0;
    size_t total_train_steps = 0;
    size_t total_redirects = 0;
    size_t total_crosstalk_msgs = 0;

    DendriteNet3D(size_t input_dim_, size_t output_dim_, unsigned seed = 42)
        : rng(seed), input_dim(input_dim_), output_dim(output_dim_) {}

    [[nodiscard]] float effective_lr() const {
        const float s = static_cast<float>(total_train_steps);
        if (s < lr_warmup_steps) {
            // linear warmup: lr_min → learning_rate
            float lr = lr_min + (learning_rate - lr_min) * (s / lr_warmup_steps);
            return std::isfinite(lr) ? lr : lr_min;
        }
        // Hybrid schedule: cosine for the first half (flat top preserves routing quality),
        // then exponential for the second half (straight line on log scale, no cliff).
        float t = std::min(1.0f, (s - lr_warmup_steps) / lr_decay_steps);
        float lr;
        if (t <= 0.5f) {
            // Cosine first half: learning_rate → midpoint (slow, flat top ~22 epochs)
            float cos_factor = 0.5f * (1.0f + std::cos(3.14159265f * t));
            lr = lr_min + (learning_rate - lr_min) * cos_factor;
        } else {
            // Exponential second half: midpoint → lr_min (straight line on log scale)
            float mid_lr = lr_min + (learning_rate - lr_min) * 0.5f;
            float t_exp = (t - 0.5f) / 0.5f;  // normalise [0.5, 1.0] → [0, 1]
            lr = mid_lr * std::pow(lr_min / mid_lr, t_exp);
        }
        return std::isfinite(lr) ? lr : lr_min;
    }

    [[nodiscard]] float current_tau() const {
        // Slow decay: keeps tau > 0.5 for ~10 epochs, preserving routing exploration
        // during the critical specialisation period (was /5000 → tau floor by epoch 2).
        return std::max(0.1f, 1.0f - static_cast<float>(total_train_steps) / 20000.0f);
    }

    // --------------------------------------------------------
    // Initialise new v3 components (call after build)
    // --------------------------------------------------------
    void init_v3(const std::string& morality_config = "") {
        // Task context: working memory
        task_context = TaskContext(output_dim, context_dim, rng);

        // Morality layer
        if (!morality_config.empty())
            morality.load_config(morality_config);
        morality.init_learned_components(output_dim, rng);

        // Text preprocessor (default 30% stop-word compression)
        text_preprocessor = TextPreprocessor(0.3f);

        std::cout << "[DendriteNet3D] v3 components initialised\n";
        std::cout << "  Task context: " << context_dim << "-dim projection, "
                  << task_context.max_items << " item capacity\n";
        std::cout << "  Morality: " << morality.rules.size() << " rules loaded\n";
        std::cout << "  Text preprocessor: " << text_preprocessor.stop_words.size()
                  << " stop-words, " << (text_preprocessor.compression_intensity * 100)
                  << "% compression\n";
    }

    // Enable image module
    void enable_image(size_t shared_dim = 0) {
        if (shared_dim == 0) shared_dim = output_dim;
        image_module = create_image_module(shared_dim, rng);
        image_enabled = true;
        std::cout << "[DendriteNet3D] Image module enabled (MobileNetV2 → "
                  << shared_dim << "-dim shared space)\n";
        init_gated_cross_attention();
    }

    // Enable audio module
    void enable_audio(size_t shared_dim = 0) {
        if (shared_dim == 0) shared_dim = output_dim;
        audio_module = create_audio_module(shared_dim, rng);
        audio_enabled = true;
        std::cout << "[DendriteNet3D] Audio module enabled (YAMNet → "
                  << shared_dim << "-dim shared space)\n";
        init_gated_cross_attention();
    }

    // Initialise Gated Cross-Attention multimodal fusion
    // Called automatically when a modality module is enabled.
    // Uses the shared_dim of the most recently enabled modality.
    void init_gated_cross_attention() {
        size_t mod_dim = 0;
        if (audio_enabled) mod_dim = audio_module.shared_dim;
        else if (image_enabled) mod_dim = image_module.shared_dim;
        if (mod_dim == 0) return;
        gated_cross_attn.init(output_dim, mod_dim, rng, /*attn_dim=*/16);
        gca_enabled = true;
        std::cout << "[DendriteNet3D] Gated cross-attention enabled (branch_dim="
                  << output_dim << ", mod_dim=" << mod_dim << ")\n";
    }

    // --------------------------------------------------------
    // Enhancement #24: Enable Perceiver IO modality fusion.
    // Replaces GCA with a latent cross-attention bottleneck when modalities are active.
    // token_dim: the shared projection dim used by modality modules.
    // --------------------------------------------------------
    void enable_perceiver(size_t token_dim = 0) {
        if (token_dim == 0) {
            // Infer from active modality modules
            if (audio_enabled) token_dim = audio_module.shared_dim;
            else if (image_enabled) token_dim = image_module.shared_dim;
            if (token_dim == 0) token_dim = output_dim;
        }
        perceiver.init(token_dim, output_dim, rng, /*n_latents=*/8, /*lat_dim=*/32);
        perceiver_enabled = true;
        printf("[PerceiverIO] Enabled: token_dim=%zu  latents=8×32  params=%zu\n",
               token_dim, perceiver.param_count());
    }

    // --------------------------------------------------------
    // Enhancement #27: Enable hierarchical (bottom-up + top-down) message passing.
    // --------------------------------------------------------
    void enable_hierarchical() {
        conductor.init_hierarchical(rng);
        printf("[Hierarchical] Enabled: guidance_proj=%zu→%zu\n", (size_t)2, summary_dim);
    }

    // --------------------------------------------------------
    // Enhancement #29: Enable multi-compartment branches.
    // Each branch gets a distal (context) MiniNetwork that modulates
    // the proximal (specialist) output via passive multiplicative coupling.
    // Call after build(). distal_input_dim = output_dim + summary_dim.
    // --------------------------------------------------------
    void enable_multi_compartment(float coupling = 0.3f) {
        size_t distal_in = output_dim + summary_dim;  // context + cross-talk summary
        size_t distal_hidden = 24;
        for (auto& b : branches) {
            b->distal = MiniNetwork(
                "distal_" + b->domain,
                {distal_in, distal_hidden, output_dim},
                Activation::RELU, Activation::TANH, rng);
            // Zero-init output layer so tanh(~0) ≈ 0 → near-identity modulation at start.
            // He init leaves large pre-tanh values (near ±1), adding ±30% random noise
            // from the first forward pass and suppressing initial accuracy below chance.
            if (!b->distal.layers.empty()) {
                auto& last = b->distal.layers.back();
                for (auto& w : last.weights.data) w *= 0.01f;
                for (auto& bi : last.bias.data)   bi = 0.0f;
            }
            b->has_distal = true;
            b->distal_coupling = coupling;
        }
        printf("[MultiCompartment] Enabled: distal_in=%zu  coupling=%.2f  +%zu params/branch\n",
               distal_in, coupling,
               branches.empty() ? 0 : branches[0]->distal.param_count());
    }

    // --------------------------------------------------------
    // Enhancement #26: Enable hypernetwork branch generation.
    // Call after build() so specialist architecture is known.
    // --------------------------------------------------------
    void enable_hypernetwork(size_t embed_dim = 16) {
        if (branches.empty()) return;
        size_t total_params = BranchGenerator::count_params(branches[0]->specialist);
        branch_generator.init(embed_dim, total_params, rng);
        // Build domain embeddings for all existing branches
        for (auto& b : branches) {
            domain_embeds.emplace(b->id, DomainEmbedding(embed_dim, b->domain));
        }
        hypernetwork_enabled = true;
        printf("[Hypernetwork] Enabled: embed_dim=%zu, specialist_params=%zu, gen_params=%zu\n",
               embed_dim, total_params, branch_generator.param_count());
    }

    // --------------------------------------------------------
    // Build the network with named branches
    // --------------------------------------------------------
    void build(const std::vector<std::string>& branch_names,
               int specialist_hidden = 64,
               const std::vector<std::vector<std::string>>& sub_branch_names = {}) {
        size_t n = branch_names.size();

        // Create conductor
        conductor = Conductor(input_dim, output_dim, n, summary_dim, attn_dim, top_k, rng);

        // Create branches
        branches = ModelBuilder::build_branches(
            branch_names, sub_branch_names,
            input_dim, output_dim, specialist_hidden, summary_dim, rng);

        // Initialise load-balancing bias.
        // Pass 1 (target_usage = 1/num_branches ≈ 0.25 for 4 branches) so each branch
        // is expected to "own" ~1 class.  Using top_k (=2) here set target_usage=0.5,
        // which actively fought specialisation by penalising any branch above 50% usage.
        conductor.init_load_balancing(1);
        growth_controller.init(branches.size());

        // Initialise concept bottleneck routing (8 interpretable concepts)
        conductor.init_concept_bottleneck(rng, 8);

        // Initialize dendritic layer: context = output_dim (output history EMA)
        dendrite_layer = DendriticLayer(input_dim, input_dim, output_dim, rng, 7, 0.75f);
        has_dendrite_layer = true;
        output_history = Tensor({output_dim});  // zero-initialized

        // Initialize VICReg EMA tracking
        branch_ema_mean.resize(branches.size());
        branch_ema_var.resize(branches.size());
        for (size_t i = 0; i < branches.size(); i++) {
            branch_ema_mean[i] = Tensor({output_dim});
            branch_ema_var[i] = Tensor({output_dim});
            branch_ema_var[i].fill(1.0f);  // initial variance estimate = 1
        }

        // Build shared expert branch (same architecture as specialist networks)
        size_t hidden_dim = static_cast<size_t>(specialist_hidden);
        shared_expert = MiniNetwork("shared_expert",
            {input_dim, hidden_dim, hidden_dim / 2, output_dim},
            Activation::RELU, Activation::SOFTMAX, rng);
        alpha_gate = DenseLayer(output_dim, 1, Activation::SIGMOID, rng);
        // Bias init -1.0: start favoring specialists over shared expert
        alpha_gate.bias[0] = -1.0f;
        has_shared_expert = true;

        std::cout << "[DendriteNet3D] Built: " << n << " branches, "
                  << param_count() << " total parameters\n";
        print_architecture();
    }

    // --------------------------------------------------------
    // Build from a ModelConfig struct — applies topology settings and activates
    // any requested enhancements. Call init_v3() / enable_image() / enable_audio()
    // separately, then run the training loop using TrainingConfig values.
    // --------------------------------------------------------
    void build_from_config(const ModelConfig& cfg) {
        heat_threshold = cfg.heat_threshold;
        top_k          = cfg.top_k;

        auto sub = cfg.sub_branches.empty()
                       ? std::vector<std::vector<std::string>>(cfg.branch_names.size())
                       : cfg.sub_branches;
        build(cfg.branch_names, cfg.hidden_size, sub);

        const auto& e = cfg.enhancements;
        if (e.hypernetwork)     enable_hypernetwork(static_cast<size_t>(e.hypernetwork_embed_dim));
        if (e.multi_compartment) enable_multi_compartment(e.multi_compartment_coupling);
        if (e.hierarchical)     enable_hierarchical();
        if (e.perceiver)        enable_perceiver(e.perceiver_token_dim);
        if (e.oscillatory) {
            conductor.oscillatory_sync = true;
            conductor.osc_coupling     = e.oscillatory_coupling;
        }
    }

    // --------------------------------------------------------
    // 3D Inference v3: morality → context → heat → branches → fuse → morality
    // --------------------------------------------------------
    InferenceResult3D infer(const Tensor& input, const std::string& text_input = "",
                            bool update_memory = true) {
        auto start = std::chrono::high_resolution_clock::now();
        InferenceResult3D result;

        // === Step 0: Morality input check ===
        auto moral_check = morality.check_input(input, text_input);
        if (!moral_check.allowed) {
            result.output = Tensor({output_dim}); // zero output
            result.morality_triggered = moral_check.triggered_rule;
            auto end = std::chrono::high_resolution_clock::now();
            result.time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            return result;
        }

        // === Step 0.5: Task context — step decay, get context vector ===
        if (update_memory) task_context.step();
        Tensor context_vec = task_context.get_context(input);
        result.context_items_active = task_context.size();

        // === Step 1: Conductor computes initial heat ===
        // Incorporate context into heat computation if context is available
        Tensor heat_input = input;
        if (!task_context.empty() && context_vec.size() > 0) {
            // Reinforce context items similar to current input
            task_context.reinforce_similar(input, 0.4f);
        }
        Tensor heat = conductor.compute_heat(heat_input, &result.concept_scores);

        // Apply morality redirect if active
        if (moral_check.redirect && moral_check.redirect_branch >= 0) {
            // Force the redirect branch to max heat
            if (moral_check.redirect_branch < (int)heat.size())
                heat[moral_check.redirect_branch] = 1.0f;
            result.morality_triggered = moral_check.triggered_rule;
        }

        // Load-balancing update: use raw compute_heat (bias+NMDA applied, pre-lateral-inhibition).
        // This is the same signal used by compute_specialization_metrics(), so the bias correction
        // directly tracks the same branch ranking shown in Phase 8 diagnostics.
        if (total_inferences > 500)
            conductor.update_load_balancing(heat, conductor.top_k);

        // === Step 1.5: Lateral inhibition (uses previous step's cross-talk summaries) ===
        if (total_inferences > lateral_inhibition_burnin && !prev_cross_talk.empty()) {
            heat = conductor.apply_lateral_inhibition(heat, prev_cross_talk);
        }
        update_phases(heat);     // Enhancement #28: oscillator phases
        hierarchical_pass(heat); // Enhancement #27: bottom-up + top-down messages

        result.heat_scores = heat;

        // === Step 2: Collect modality embeddings (before branch eval, for GCA) ===
        std::vector<Tensor> modality_embeddings;
        if (gca_enabled) {
            if (image_enabled) modality_embeddings.push_back(image_module.process(input));
            if (audio_enabled) modality_embeddings.push_back(audio_module.process(input));
        }

        // === Step 2.5: Activate hot branches and get outputs ===
        std::vector<Tensor> branch_outputs(branches.size(), Tensor({output_dim}));
        std::vector<BranchSignal> signals;
        std::vector<int> active;

        // Enhancement #24: compute Perceiver IO output once for all branches.
        // Guard: PerceiverIO has no backward() so its weights stay random — applying it
        // during inference without training adds ~20% random noise to branch outputs.
        // Only blend when concepts were actually triggered by text (genuine modality context).
        Tensor perceiver_out({output_dim});
        bool use_perceiver = perceiver_enabled && !modality_embeddings.empty()
                             && modality_concepts_active;
        if (use_perceiver) perceiver_out = perceiver.forward(modality_embeddings);

        for (size_t i = 0; i < branches.size(); i++) {
            if (heat[i] >= heat_threshold) {
                active.push_back(i);
                branch_outputs[i] = evaluate_branch(branches[i], input, /*is_inference=*/true, get_dendrite_context(), i);
                if (use_perceiver) {
                    // Blend Perceiver output into branch output (0.8 branch + 0.2 perceiver)
                    for (size_t d = 0; d < output_dim && d < perceiver_out.size(); d++)
                        branch_outputs[i][d] = 0.8f * branch_outputs[i][d] + 0.2f * perceiver_out[d];
                } else if (gca_enabled && !modality_embeddings.empty() && modality_concepts_active) {
                    // Gate GCA on modality_concepts_active: stub encoders produce random
                    // embeddings on every call; applying GCA without real modal context
                    // overwrites branch outputs with random noise.
                    branch_outputs[i] = gated_cross_attn.forward(branch_outputs[i], modality_embeddings);
                }
            }
        }
        if (active.empty()) {
            for (size_t i = 0; i < branches.size(); i++) {
                active.push_back(i);
                branch_outputs[i] = evaluate_branch(branches[i], input, /*is_inference=*/true, get_dendrite_context(), i);
                if (use_perceiver) {
                    for (size_t d = 0; d < output_dim && d < perceiver_out.size(); d++)
                        branch_outputs[i][d] = 0.8f * branch_outputs[i][d] + 0.2f * perceiver_out[d];
                } else if (gca_enabled && !modality_embeddings.empty() && modality_concepts_active) {
                    branch_outputs[i] = gated_cross_attn.forward(branch_outputs[i], modality_embeddings);
                }
            }
        }

        result.active_branch_ids = active;
        result.branches_evaluated = active.size();

        // === Step 3: Multimodal text-concept activation logging ===
        // Check if any text concepts trigger image/audio associations
        if (!text_input.empty()) {
            auto tokens = text_preprocessor.process(text_input);
            for (auto& tok : tokens) {
                if (tok.is_stop) continue;  // skip function words
                if (image_enabled && image_module.has_association(tok.text)) {
                    result.modalities_activated.push_back("image:" + tok.text);
                }
                if (audio_enabled && audio_module.has_association(tok.text)) {
                    result.modalities_activated.push_back("audio:" + tok.text);
                }
            }
        }
        modality_concepts_active = !result.modalities_activated.empty();

        // === Step 3: Cross-talk with GAT enrichment ===
        // First pass: compute raw summaries + GAT keys/values
        std::vector<CrossTalkMessage> cross_talk;
        std::vector<std::pair<Tensor,Tensor>> gat_kvs;  // (key, value) per active branch
        for (int b : active) {
            CrossTalkMessage msg;
            msg.from_branch = b;
            msg.summary = branches[b]->sub_conductor.summary_net.forward(input);
            for (auto& v : msg.summary.data) if (!std::isfinite(v)) v = 0.0f;
            cross_talk.push_back(msg);
            total_crosstalk_msgs++;
            // Compute GAT K/V for this branch's summary
            auto gat_p = branches[b]->sub_conductor.compute_gat(msg.summary);
            gat_kvs.push_back({gat_p.k, gat_p.v});
        }
        // Second pass: enrich each branch's summary with GAT attention over siblings
        for (size_t i = 0; i < active.size(); i++) {
            int b = active[i];
            // Sibling KVs = all except self; phase-gate when oscillatory_sync is on
            std::vector<std::pair<Tensor,Tensor>> sibling_kvs;
            for (size_t j = 0; j < gat_kvs.size(); j++) {
                if ((int)j == (int)i) continue;
                if (conductor.oscillatory_sync) {
                    // Enhancement #28: weight by cos(θᵢ − θⱼ); suppress anti-phase siblings
                    float gate = std::max(0.0f,
                        std::cos(branches[b]->phase - branches[active[j]]->phase));
                    if (gate < 1e-4f) continue;  // fully blocked — skip this sibling
                    // Scale the value tensor by phase gate
                    Tensor gated_v = gat_kvs[j].second;
                    for (auto& v : gated_v.data) v *= gate;
                    sibling_kvs.push_back({gat_kvs[j].first, gated_v});
                } else {
                    sibling_kvs.push_back(gat_kvs[j]);
                }
            }
            if (!sibling_kvs.empty()) {
                cross_talk[i].summary = branches[b]->sub_conductor.gat_enrich(
                    cross_talk[i].summary, sibling_kvs);
            }
        }

        // === Step 4: Sub-conductor feedback ===
        result.redirect_count = 0;
        for (int b : active) {
            auto signal = branches[b]->sub_conductor.evaluate(
                input, branch_outputs[b], cross_talk);
            signals.push_back(signal);
            result.branch_confidences.push_back(signal.confidence);
            result.branch_redirects.push_back(signal.wants_redirect);
            if (signal.wants_redirect) {
                result.redirect_count++;
                total_redirects++;
            }
        }

        // === Step 5: Integrate feedback ===
        std::vector<BranchSignal> full_signals(branches.size());
        for (auto& sig : signals) {
            if (sig.branch_id >= 0 && sig.branch_id < (int)branches.size())
                full_signals[sig.branch_id] = sig;
        }
        Tensor adjusted_heat = conductor.integrate_feedback(heat, full_signals);

        // === Step 6: Choose fusion strategy ===
        auto [strategy, ignored_weights] = conductor.choose_strategy(
            input, adjusted_heat, false, 0.1f, rng);
        result.strategy_used = strategy;

        // === Step 7: Fuse (with optional debate modulation) ===
        apply_debate_to_heat(adjusted_heat, branch_outputs, active, input);
        Tensor fused = conductor.fuse(strategy, adjusted_heat, branch_outputs, active);

        // === Step 8: Finalize ===
        result.output = conductor.finalize(fused);

        // === Step 8.5: Shared expert blending (always evaluates, regardless of heat) ===
        if (has_shared_expert) {
            Tensor shared_out = shared_expert.forward(input);
            for (auto& v : shared_out.data) if (!std::isfinite(v)) v = 0.0f;

            // Alpha gate takes the fused output and produces a mixing weight in [0,1]
            // alpha=1 → use fused specialist output; alpha=0 → use shared expert output
            Tensor alpha_out = alpha_gate.forward(result.output);
            float alpha = alpha_out[0];
            if (!std::isfinite(alpha)) alpha = 0.5f;

            // Blend: output = alpha * fused + (1 - alpha) * shared
            for (size_t i = 0; i < result.output.size(); i++) {
                result.output[i] = alpha * result.output[i] + (1.0f - alpha) * shared_out[i];
                if (!std::isfinite(result.output[i])) result.output[i] = 0.0f;
            }
        }

        // === Step 9: Morality output check (regex + critique + conf gate) ===
        // context_sim: cosine similarity between input and context vector
        float context_sim = 0.5f;
        if (context_vec.size() > 0 && input.size() > 0) {
            float dot = 0, na = 0, nb = 0;
            size_t csim_dim = std::min(input.size(), context_vec.size());
            for (size_t i = 0; i < csim_dim; i++) {
                dot += input[i] * context_vec[i];
                na  += input[i] * input[i];
                nb  += context_vec[i] * context_vec[i];
            }
            if (na > 1e-7f && nb > 1e-7f)
                context_sim = std::clamp(dot / (std::sqrt(na) * std::sqrt(nb)), 0.0f, 1.0f);
        }
        auto out_check = morality.check_output(result.output, "", heat, context_sim);
        if (!out_check.allowed) {
            result.output = Tensor({output_dim}); // zero blocked output
            result.morality_triggered = out_check.triggered_rule;
        }

        // === Step 10: Store result in working memory ===
        // Find the hottest branch and store its output
        if (!active.empty()) {
            int hottest = active[0];
            for (int b : active)
                if (adjusted_heat[b] > adjusted_heat[hottest]) hottest = b;
            task_context.store(branch_outputs[hottest], hottest, adjusted_heat[hottest],
                             "branch_" + std::to_string(hottest));
            // Only reinforce top-2 branches by adjusted_heat, weighted by heat score
            std::vector<int> top2 = active;
            std::sort(top2.begin(), top2.end(),
                [&](int a, int b){ return adjusted_heat[a] > adjusted_heat[b]; });
            if (top2.size() > 2) top2.resize(2);
            for (int b : top2) task_context.reinforce_weighted(b, adjusted_heat[b]);
        }

        // Cache cross-talk summaries for next step's lateral inhibition
        prev_cross_talk = cross_talk;

        auto end = std::chrono::high_resolution_clock::now();
        result.time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // Update output history EMA (dendritic context for mode detection)
        if (update_memory && result.output.size() == output_history.size()) {
            for (size_t i = 0; i < output_dim; i++) {
                output_history[i] = (1.0f - output_history_alpha) * output_history[i]
                                     + output_history_alpha * result.output[i];
            }
        }

        total_inferences++;
        return result;
    }

    // --------------------------------------------------------
    // Run data cleaning pipeline
    // --------------------------------------------------------
    DataCleaner::CleaningResult clean_data(const std::vector<Tensor>& inputs,
                                           const std::vector<Tensor>& targets) {
        auto heat_fn = [this](const Tensor& input) -> std::pair<Tensor, Tensor> {
            Tensor heat = conductor.compute_heat(input);
            // Quick inference for output
            std::vector<Tensor> outputs(branches.size(), Tensor({output_dim}));
            for (size_t i = 0; i < branches.size(); i++)
                outputs[i] = evaluate_branch(branches[i], input, /*is_inference=*/true);
            // Weighted average for output estimate
            Tensor combined({output_dim});
            float total_h = 0;
            for (size_t i = 0; i < branches.size(); i++) total_h += heat[i];
            if (total_h < 1e-7f) total_h = 1.0f;
            for (size_t i = 0; i < branches.size(); i++)
                for (size_t d = 0; d < output_dim; d++)
                    combined[d] += outputs[i][d] * heat[i] / total_h;
            return {heat, combined};
        };
        data_cleaner.current_training_steps = static_cast<int>(total_train_steps);
        return data_cleaner.clean(inputs, targets, heat_fn, branches.size());
    }

    // --------------------------------------------------------
    // New session: reset working memory
    // --------------------------------------------------------
    void new_session() {
        task_context.reset();
        output_history.zero();  // reset output history for new context
    }

    // --------------------------------------------------------
    // Multi-task: per-task output heads + task embeddings
    // --------------------------------------------------------

    /// Register a task with its own output heads (one per branch + shared expert).
    /// Call before training on that task.
    void register_task(int task_id) {
        if (task_heads.count(task_id)) return;  // already registered
        std::vector<DenseLayer> heads;
        for (auto& branch : branches) {
            // Output head: hidden/2 → output_dim (matches specialist's last layer)
            size_t hidden = branch->specialist.layers.back().in_dim;
            heads.emplace_back(hidden, output_dim, Activation::SOFTMAX, rng);
        }
        // Shared expert head
        if (has_shared_expert) {
            size_t hidden = shared_expert.layers.back().in_dim;
            heads.emplace_back(hidden, output_dim, Activation::SOFTMAX, rng);
        }
        task_heads[task_id] = std::move(heads);

        // Task embedding for dendritic context
        if (task_embeddings.size() <= (size_t)task_id)
            task_embeddings.resize(task_id + 1);
        task_embeddings[task_id] = Tensor({output_dim});
        // Initialize with distinct pattern per task
        for (size_t d = 0; d < output_dim; d++)
            task_embeddings[task_id][d] = (d == (size_t)(task_id % output_dim)) ? 1.0f : -0.5f;
    }

    /// Set the active task for training and inference
    void set_task(int task_id) {
        current_task_id = task_id;
        if (!task_heads.count(task_id)) register_task(task_id);
    }

    /// Set active task embedding for dendrites only (no per-task output heads)
    void set_task_context(int task_id) {
        if (task_embeddings.size() <= (size_t)task_id)
            task_embeddings.resize(task_id + 1);
        if (task_embeddings[task_id].size() == 0) {
            task_embeddings[task_id] = Tensor({output_dim});
            for (size_t d = 0; d < output_dim; d++)
                task_embeddings[task_id][d] = (d == (size_t)(task_id % output_dim)) ? 1.0f : -0.5f;
        }
        current_task_id = task_id;
    }

    /// Get the dendritic context: task embedding if available, else output history
    Tensor get_dendrite_context() const {
        if (current_task_id >= 0 && (size_t)current_task_id < task_embeddings.size()
            && task_embeddings[current_task_id].size() > 0)
            return task_embeddings[current_task_id];
        return output_history;
    }

    // --------------------------------------------------------
    // Continual Learning: replay buffer + task boundary consolidation
    // --------------------------------------------------------
    struct ReplayBuffer {
        std::vector<Tensor> inputs, targets;
        size_t capacity = 500;
        size_t total_seen = 0;   // reservoir sampling counter
        std::mt19937 rng{42};

        void store(const Tensor& input, const Tensor& target) {
            total_seen++;
            if (inputs.size() < capacity) {
                inputs.push_back(input);
                targets.push_back(target);
            } else {
                // Reservoir sampling: replace random slot with probability capacity/total_seen
                std::uniform_int_distribution<size_t> dist(0, total_seen - 1);
                size_t idx = dist(rng);
                if (idx < capacity) {
                    inputs[idx]  = input;
                    targets[idx] = target;
                }
            }
        }
        bool empty() const { return inputs.empty(); }
        size_t size() const { return inputs.size(); }
    };
    ReplayBuffer replay_buffer;
    float replay_ratio = 0.2f;  // fraction of replay samples per training batch

    /// Save current training data to replay buffer for future replay
    void save_to_replay(const std::vector<Tensor>& inputs,
                        const std::vector<Tensor>& targets, size_t n = 0) {
        if (n == 0) n = std::min(inputs.size(), replay_buffer.capacity);
        std::mt19937 sample_rng(42);
        std::vector<size_t> idx(inputs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), sample_rng);
        for (size_t i = 0; i < n && i < inputs.size(); i++)
            replay_buffer.store(inputs[idx[i]], targets[idx[i]]);
    }

    /// Consolidate SI importance at task boundary (not every epoch)
    void consolidate_at_boundary() {
        for (auto& branch : branches) {
            branch->specialist.consolidate_all();
            if (branch->has_children)
                branch->child_router.consolidate_all();
        }
        conductor.heat_network.consolidate_all();
        conductor.combiner.consolidate_all();
        if (has_dendrite_layer) dendrite_layer.consolidate_importance();
        if (has_shared_expert) shared_expert.consolidate_all();
    }

    /// Train one epoch with optional replay from previous tasks
    float train_epoch_with_replay(const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& targets) {
        std::vector<size_t> idx(inputs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        float epoch_loss = 0;
        size_t replay_every = (replay_ratio > 0 && !replay_buffer.empty())
                              ? std::max((size_t)1, (size_t)(1.0f / replay_ratio))
                              : 0;
        std::uniform_int_distribution<size_t> replay_dist(0, std::max((size_t)1, replay_buffer.size()) - 1);

        for (size_t i = 0; i < idx.size(); i++) {
            epoch_loss += train_sample(inputs[idx[i]], targets[idx[i]]);
            // Interleave replay samples
            if (replay_every > 0 && (i % replay_every == 0)) {
                size_t ri = replay_dist(rng);
                train_sample(replay_buffer.inputs[ri], replay_buffer.targets[ri]);
            }
        }
        return epoch_loss / inputs.size();
    }

    // --------------------------------------------------------
    // Training
    // --------------------------------------------------------
    float train_sample(const Tensor& input, const Tensor& target) {
        modality_concepts_active = false;  // no text context during batch training
        // Forward pass
        Tensor heat = conductor.compute_heat(input);
        // NaN guard on heat
        for (size_t i = 0; i < heat.size(); i++)
            if (std::isnan(heat[i]) || std::isinf(heat[i])) heat[i] = 0.5f;

        // Load-balancing update: same pre-lateral-inhibition signal as infer() and
        // compute_specialization_metrics() for consistent bias correction.
        if (total_train_steps > 500)
            conductor.update_load_balancing(heat, conductor.top_k);

        // Lateral inhibition: suppress similar branches (uses previous step's summaries)
        if (total_train_steps > lateral_inhibition_burnin && !prev_cross_talk.empty()) {
            heat = conductor.apply_lateral_inhibition(heat, prev_cross_talk);
        }
        update_phases(heat);     // Enhancement #28: oscillator phases
        hierarchical_pass(heat); // Enhancement #27: bottom-up + top-down messages

        std::vector<Tensor> branch_outputs(branches.size(), Tensor({output_dim}));
        std::vector<int> active;

        // Get context vector for dendritic modulation during training
        task_context.step();
        Tensor context_vec = task_context.get_context(input);
        for (auto& v : context_vec.data) if (!std::isfinite(v)) v = 0.0f;

        // During training, always evaluate all branches for stability
        for (size_t i = 0; i < branches.size(); i++) {
            active.push_back(i);
            branch_outputs[i] = evaluate_branch(branches[i], input, false, get_dendrite_context(), i);
            // NaN guard on branch outputs
            for (size_t j = 0; j < branch_outputs[i].size(); j++)
                if (std::isnan(branch_outputs[i][j]) || std::isinf(branch_outputs[i][j]))
                    branch_outputs[i][j] = 1.0f / output_dim;
        }

        // Cross-talk (forward only, no backward needed for summaries during training)
        // Enhancement #28: phase-gate summary contributions between branches
        std::vector<CrossTalkMessage> cross_talk;
        for (size_t ci = 0; ci < active.size(); ci++) {
            int b = active[ci];
            CrossTalkMessage msg;
            msg.from_branch = b;
            Tensor raw_summary = branches[b]->sub_conductor.summary_net.forward(input);
            for (auto& v : raw_summary.data) if (!std::isfinite(v)) v = 0.0f;
            // Scale summary contribution by mean phase alignment with all other active branches
            if (conductor.oscillatory_sync && active.size() > 1) {
                float mean_gate = 0.0f;
                for (size_t cj = 0; cj < active.size(); cj++) {
                    if (ci == cj) continue;
                    mean_gate += std::max(0.0f,
                        std::cos(branches[active[ci]]->phase - branches[active[cj]]->phase));
                }
                mean_gate /= static_cast<float>(active.size() - 1);
                for (auto& v : raw_summary.data) v *= mean_gate;
            }
            msg.summary = raw_summary;
            cross_talk.push_back(msg);
        }
        // Cache for next step's lateral inhibition
        prev_cross_talk = cross_talk;

        // Sub-conductor signals
        std::vector<BranchSignal> full_signals(branches.size());
        for (int b : active) {
            auto signal = branches[b]->sub_conductor.evaluate(
                input, branch_outputs[b], cross_talk);
            full_signals[b] = signal;
        }

        Tensor adjusted_heat = conductor.integrate_feedback(heat, full_signals);
        for (size_t i = 0; i < adjusted_heat.size(); i++)
            if (std::isnan(adjusted_heat[i]) || std::isinf(adjusted_heat[i]))
                adjusted_heat[i] = 1.0f / branches.size();

        auto [strategy, strategy_weights] = conductor.choose_strategy(
            input, adjusted_heat, true, current_tau(), rng);

        // Debate verification: modulate heat by branch self-consistency (only when uncertain)
        if (total_train_steps > 1000)  // only after initial training stabilises
            apply_debate_to_heat(adjusted_heat, branch_outputs, active, input);

        Tensor fused = conductor.fuse(strategy, adjusted_heat, branch_outputs, active);

        // NaN guard on fused
        for (size_t i = 0; i < fused.size(); i++)
            if (std::isnan(fused[i]) || std::isinf(fused[i]))
                fused[i] = 1.0f / output_dim;

        Tensor output = conductor.finalize(fused);
        // NaN guard on output
        bool had_nan = false;
        for (size_t i = 0; i < output.size(); i++) {
            if (std::isnan(output[i]) || std::isinf(output[i])) {
                output[i] = 1.0f / output_dim;
                had_nan = true;
            }
        }
        if (had_nan) return 2.0f; // skip backward if NaN detected

        // Shared expert forward (cached for backward; blends into output)
        Tensor shared_out_train({output_dim});
        float shared_alpha = 0.5f;
        if (has_shared_expert) {
            Tensor pre_blend_output = output; // fused specialist output before blend
            shared_out_train = shared_expert.forward(input);
            for (auto& v : shared_out_train.data) if (!std::isfinite(v)) v = 0.0f;

            Tensor alpha_out = alpha_gate.forward(pre_blend_output);
            shared_alpha = alpha_out[0];
            if (!std::isfinite(shared_alpha)) shared_alpha = 0.5f;

            // Apply blend to output (updates output in-place for loss computation)
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = shared_alpha * pre_blend_output[i]
                          + (1.0f - shared_alpha) * shared_out_train[i];
                if (!std::isfinite(output[i])) output[i] = 1.0f / output_dim;
            }
        }

        // Compute loss
        float loss = 0;
        for (size_t i = 0; i < target.size(); i++)
            if (target[i] > 0)
                loss -= target[i] * std::log(std::max(output[i], 1e-7f));

        // VICReg branch diversity loss (active after warmup)
        float vicreg_loss = 0.0f;
        if (total_train_steps > vicreg_warmup) {
            // Collect active branch indices and outputs
            std::vector<size_t> active_indices;
            std::vector<Tensor> active_outputs;
            for (size_t b = 0; b < branches.size(); b++) {
                if (heat[b] > heat_threshold) {
                    active_indices.push_back(b);
                    active_outputs.push_back(branch_outputs[b]);
                }
            }

            if (active_outputs.size() >= 2) {
                // 1. VARIANCE LOSS: Each branch output should maintain variance > 1.0
                float var_loss = 0.0f;
                for (size_t i = 0; i < active_indices.size(); i++) {
                    size_t b = active_indices[i];
                    const auto& out = active_outputs[i];

                    // Update EMA mean
                    for (size_t d = 0; d < output_dim; d++) {
                        branch_ema_mean[b][d] = (1.0f - vicreg_ema_alpha) * branch_ema_mean[b][d]
                                                 + vicreg_ema_alpha * out[d];
                    }
                    // Update EMA variance
                    for (size_t d = 0; d < output_dim; d++) {
                        float diff = out[d] - branch_ema_mean[b][d];
                        branch_ema_var[b][d] = (1.0f - vicreg_ema_alpha) * branch_ema_var[b][d]
                                                + vicreg_ema_alpha * diff * diff;
                    }
                    // Hinge loss: penalize if sqrt(var) < 1.0
                    for (size_t d = 0; d < output_dim; d++) {
                        float std_dev = std::sqrt(branch_ema_var[b][d] + 1e-4f);
                        var_loss += std::max(0.0f, 0.3f - std_dev);  // scaled for softmax outputs
                    }
                }
                var_loss /= (active_indices.size() * output_dim);

                // 2. COVARIANCE LOSS: decorrelate output dimensions within each branch
                float cov_loss = 0.0f;
                for (size_t i = 0; i < active_indices.size(); i++) {
                    const auto& out = active_outputs[i];
                    size_t b = active_indices[i];
                    for (size_t d = 0; d + 1 < output_dim; d++) {
                        float centered_d  = out[d]     - branch_ema_mean[b][d];
                        float centered_d1 = out[d + 1] - branch_ema_mean[b][d + 1];
                        cov_loss += centered_d * centered_d1;
                    }
                }
                cov_loss = std::abs(cov_loss) / std::max((size_t)1, active_indices.size() * output_dim);

                // 3. INTER-BRANCH DIVERSITY: penalize positive cosine similarity between branch outputs
                float div_loss = 0.0f;
                size_t pair_count = 0;
                for (size_t i = 0; i < active_outputs.size(); i++) {
                    for (size_t j = i + 1; j < active_outputs.size(); j++) {
                        float sim = active_outputs[i].cosine_similarity(active_outputs[j]);
                        div_loss += std::max(0.0f, sim);
                        pair_count++;
                    }
                }
                if (pair_count > 0) div_loss /= pair_count;

                vicreg_loss = vicreg_weight * (var_loss + 0.5f * cov_loss + div_loss);
                if (!std::isfinite(vicreg_loss)) vicreg_loss = 0.0f;
            }
        }

        // Contrastive branch specialisation: penalise cosine similarity between co-active branches
        // Pushes branches to learn complementary (not redundant) representations.
        float contrastive_loss = 0.0f;
        {
            std::vector<int> hot;
            for (int b : active) if (heat[b] >= heat_threshold) hot.push_back(b);
            size_t np = hot.size() * (hot.size() > 0 ? hot.size() - 1 : 0) / 2;
            for (size_t i = 0; i < hot.size(); i++)
                for (size_t j = i + 1; j < hot.size(); j++) {
                    float sim = branch_outputs[hot[i]].cosine_similarity(branch_outputs[hot[j]]);
                    contrastive_loss += std::max(0.0f, sim);
                }
            if (np > 0) contrastive_loss /= np;
            if (!std::isfinite(contrastive_loss)) contrastive_loss = 0.0f;
        }

        // Enhancement #25: CLIP contrastive alignment — align highest-heat branch output
        // with modality associations, but only when text input actually triggered a concept
        // match (modality_concepts_active). prev_cross_talk is always non-empty after
        // warmup so was firing every step against a random "cat" embedding — incorrect.
        float clip_loss = 0.0f;
        if ((image_enabled || audio_enabled) && modality_concepts_active) {
            // Use the hottest branch's output as the "text" embedding
            int hot_branch = active[0];
            for (int b : active)
                if (heat[b] > heat[hot_branch]) hot_branch = b;
            const Tensor& text_proxy = branch_outputs[hot_branch];

            // Align with any registered modality associations we have
            int align_count = 0;
            auto try_align = [&](const ModalityModule& mod) {
                if (!mod.is_loaded() || mod.associations.empty()) return;
                // Pick a random association as positive pair
                auto it = mod.associations.begin();
                if (it->second.size() > 0) {
                    clip_loss += ModalityModule::alignment_loss(text_proxy, it->second);
                    align_count++;
                }
            };
            if (image_enabled) try_align(image_module);
            if (audio_enabled) try_align(audio_module);
            if (align_count > 0) clip_loss /= align_count;
            if (!std::isfinite(clip_loss)) clip_loss = 0.0f;
        }

        float total_loss = loss + vicreg_loss + 0.05f * contrastive_loss + 0.1f * clip_loss;

        // === Backward pass with gradient clipping ===
        const float grad_clip = 1.0f;

        // 1. Combiner backward
        Tensor grad_output = output - target;
        grad_output.clip(-grad_clip, grad_clip);
        Tensor grad_fused = conductor.combiner.backward(grad_output);
        grad_fused.clip(-grad_clip, grad_clip);
        conductor.combiner.apply_adam(effective_lr());

        // 2. Branch specialists: scale gradient by heat weight
        float heat_sum = 0;
        for (int b : active) heat_sum += adjusted_heat[b];
        if (heat_sum < 1e-6f) heat_sum = 1.0f;

        Tensor dendrite_grad({input_dim});  // accumulate input grads for dendritic layer
        for (int b : active) {
            float w = adjusted_heat[b] / heat_sum;
            Tensor branch_grad = grad_fused * w;
            branch_grad.clip(-grad_clip, grad_clip);

            auto it = last_used_specialist.find(b);
            if (it != last_used_specialist.end() && it->second) {
                Tensor g_in;
                // Per-task head backward: grad flows through task head then backbone
                if (current_task_id >= 0 && task_heads.count(current_task_id) &&
                    b < (int)task_heads[current_task_id].size()) {
                    auto& head = task_heads[current_task_id][b];
                    Tensor head_g = head.backward(branch_grad);
                    head.apply_adam(effective_lr());
                    head_g.clip(-grad_clip, grad_clip);
                    // Backward through backbone (all layers except last)
                    Tensor g = head_g;
                    for (int l = (int)it->second->layers.size() - 2; l >= 0; l--)
                        g = it->second->layers[l].backward(g);
                    for (auto& layer : it->second->layers) layer.apply_adam(effective_lr());
                    g_in = g;
                } else {
                    g_in = it->second->backward(branch_grad);
                    it->second->apply_adam(effective_lr());
                }
                if (has_dendrite_layer) {
                    g_in.clip(-grad_clip, grad_clip);
                    for (size_t j = 0; j < input_dim && j < g_in.size(); j++)
                        dendrite_grad[j] += g_in[j];
                }
            }

            // Enhancement #29: Distal compartment backward
            if (branches[b]->has_distal
                && branches[b]->last_prox_out.size() == output_dim
                && branches[b]->last_dist_out.size() == output_dim) {
                const float C = branches[b]->distal_coupling;
                Tensor grad_dist({output_dim});
                for (size_t d = 0; d < output_dim; d++) {
                    float tanh_val = branches[b]->last_dist_out[d];  // already tanh-activated
                    float sech2 = 1.0f - tanh_val * tanh_val;
                    grad_dist[d] = branch_grad[d] * branches[b]->last_prox_out[d] * C * sech2;
                    grad_dist[d] = std::clamp(grad_dist[d], -grad_clip, grad_clip);
                }
                branches[b]->distal.backward(grad_dist);
                branches[b]->distal.apply_adam(effective_lr());
            }

            branches[b]->running_loss = 0.95f * branches[b]->running_loss + 0.05f * loss;
            branches[b]->visit_count++;
        }

        // 2b. Dendritic layer backward using accumulated input gradients (no second backward)
        if (has_dendrite_layer && context_vec.size() > 0) {
            dendrite_grad.clip(-grad_clip, grad_clip);
            dendrite_layer.backward(dendrite_grad);
            dendrite_layer.apply_adam(effective_lr());
        }

        // 3. Heat network: supervised gradient toward specialised routing.
        //
        //    Key design decisions:
        //    (a) Non-best heat_target = 0.05 (not 0.2).  0.2 matched heat_threshold
        //        exactly, teaching the network to keep every branch on the selection
        //        boundary → uniform routing.  0.05 is well below threshold, giving
        //        the heat network a clear signal to suppress non-best branches.
        //
        //    (b) Margin gate: only fire the gradient when the best branch is
        //        meaningfully better than the second-best (margin > 0.1 nats).
        //        On ambiguous samples where all branches perform similarly, the
        //        gradient adds noise that pushes toward the uniform attractor.
        float best_branch_loss   = 1e9f;
        float second_branch_loss = 1e9f;
        int   best_branch = 0;
        for (int b : active) {
            float bl = 0;
            for (size_t i = 0; i < target.size(); i++)
                if (target[i] > 0)
                    bl -= target[i] * std::log(std::max(branch_outputs[b][i], 1e-7f));
            if (bl < best_branch_loss) {
                second_branch_loss = best_branch_loss;
                best_branch_loss   = bl;
                best_branch        = b;
            } else if (bl < second_branch_loss) {
                second_branch_loss = bl;
            }
        }
        // Only update routing when there is a clear winner (margin > 0.1 nats)
        const float margin = second_branch_loss - best_branch_loss;
        Tensor heat_target({branches.size()});
        for (size_t i = 0; i < branches.size(); i++)
            heat_target[i] = ((int)i == best_branch) ? 1.0f : 0.05f;
        Tensor heat_grad = (heat - heat_target) * (margin > 0.1f ? 0.05f : 0.0f);
        heat_grad.clip(-0.5f, 0.5f);
        if (conductor.concept_bottleneck_enabled) {
            // Concept bottleneck path: heat_grad → concept_heat_net → concept_predictor
            Tensor concept_heat_grad = conductor.concept_heat_net.backward(heat_grad);
            concept_heat_grad.clip(-0.5f, 0.5f);
            conductor.concept_predictor.backward(concept_heat_grad);
        } else {
            conductor.heat_network.backward(heat_grad);
        }

        // 4. Strategy network: reward-based gradient (very small)
        float reward = std::max(-loss, -5.0f); // clamp reward
        Tensor strat_grad({(size_t)FusionStrategy::NUM_STRATEGIES});
        int cs = (int)strategy;
        for (int i = 0; i < (int)FusionStrategy::NUM_STRATEGIES; i++)
            strat_grad[i] = (i == cs) ? -reward * 0.01f : reward * 0.005f;
        strat_grad.clip(-0.2f, 0.2f);
        conductor.strategy_network.backward(strat_grad);

        // 4b. Child router: REINFORCE policy gradient
        for (auto& br : branches) {
            if (!br->has_children || br->last_chosen_child < 0) continue;
            size_t nc = br->children.size();
            Tensor child_grad({nc});
            for (size_t c = 0; c < nc; c++) {
                float g = ((int)c == br->last_chosen_child)
                    ? -reward * 0.01f
                    : reward * 0.01f / std::max(1, (int)nc - 1);
                child_grad[c] = std::isfinite(g) ? g : 0.0f;
            }
            child_grad.clip(-0.5f, 0.5f);
            br->child_router.backward(child_grad);
            br->child_router.apply_adam(effective_lr());
        }

        // 5. Sub-conductor relevance: supervised toward heat target
        for (int b : active) {
            Tensor rel_grad({1});
            rel_grad[0] = (full_signals[b].heat - heat_target[b]) * 0.02f;
            branches[b]->sub_conductor.relevance_net.backward(rel_grad);
            branches[b]->sub_conductor.apply_adam(effective_lr() * 0.5f);
        }

        // 6. Shared expert backward: full gradient (not heat-scaled)
        if (has_shared_expert) {
            // Shared expert receives (1 - alpha) * grad_output
            Tensor shared_grad({output_dim});
            for (size_t i = 0; i < output_dim; i++) {
                shared_grad[i] = (1.0f - shared_alpha) * grad_output[i];
                if (!std::isfinite(shared_grad[i])) shared_grad[i] = 0.0f;
            }
            shared_grad.clip(-grad_clip, grad_clip);
            shared_expert.backward(shared_grad);
            shared_expert.apply_adam(effective_lr());

            // Alpha gate gradient
            float blend_grad = 0.0f;
            for (size_t i = 0; i < output_dim; i++) {
                float alpha_safe = std::max(shared_alpha, 1e-4f);
                float fused_val = (output[i] - (1.0f - shared_alpha) * shared_out_train[i]) / alpha_safe;
                float d_blend = fused_val - shared_out_train[i];
                if (!std::isfinite(d_blend)) d_blend = 0.0f;
                blend_grad += grad_output[i] * d_blend;
            }
            if (!std::isfinite(blend_grad)) blend_grad = 0.0f;
            Tensor alpha_grad_t({1});
            alpha_grad_t[0] = blend_grad;
            alpha_grad_t.clip(-0.5f, 0.5f);
            alpha_gate.backward(alpha_grad_t);
            alpha_gate.apply_adam(effective_lr() * 0.1f); // slow learning for stability
        }

        // 7. Auxiliary early exit loss (delayed: only after exit_enabled_after_steps)
        const float exit_loss_weight = 0.3f;
        if (total_train_steps < exit_enabled_after_steps) goto skip_exit_loss;
        for (int b : active) {
            auto& branch = branches[b];
            if (!branch->specialist.exit_classifier.has_value()) continue;
            if (branch->specialist.last_exit_hidden.size() == 0) continue;

            auto exit_result = branch->specialist.exit_classifier->evaluate(
                branch->specialist.last_exit_hidden);

            Tensor exit_grad(exit_result.output.shape);
            for (size_t i = 0; i < exit_grad.size(); i++) {
                exit_grad[i] = (exit_result.output[i] - target[i]) * exit_loss_weight;
            }
            exit_grad.clip(-0.5f, 0.5f);
            branch->specialist.exit_classifier->backward(exit_grad);
            branch->specialist.exit_classifier->apply_adam(effective_lr());
        }
        skip_exit_loss:

        conductor.apply_adam(effective_lr());
        task_context.apply_adam(effective_lr() * 0.5f);
        morality.apply_adam(effective_lr() * 0.1f);  // slow — critique + conf gate
        if (gca_enabled) gated_cross_attn.apply_adam(effective_lr() * 0.5f);
        if (perceiver_enabled) perceiver.apply_adam(effective_lr() * 0.5f); // Enhancement #24
        // Update output history EMA for dendritic context
        if (output.size() == output_history.size()) {
            for (size_t i = 0; i < output_dim; i++)
                output_history[i] = (1.0f - output_history_alpha) * output_history[i]
                                     + output_history_alpha * output[i];
        }
        total_train_steps++;
        return total_loss;
    }

    // --------------------------------------------------------
    // Split a branch: clone specialist with noise, expand conductor output
    // --------------------------------------------------------
    void split_branch(size_t branch_idx) {
        if (branch_idx >= branches.size()) return;
        if (branches.size() >= growth_controller.max_branches) return;
        auto& src = branches[branch_idx];

        auto new_branch = std::make_shared<Branch>();
        new_branch->id = branches.size();
        new_branch->domain = src->domain + "_v2";
        new_branch->visit_count = 0;
        new_branch->running_loss = src->running_loss;
        if (hypernetwork_enabled && domain_embeds.count(src->id)) {
            // Enhancement #26: generate new branch weights from domain embedding
            std::string new_domain = src->domain + "_v2";
            DomainEmbedding new_embed(branch_generator.domain_embed_dim, new_domain);
            // Blend parent embedding with new domain embedding for continuity
            for (size_t i = 0; i < new_embed.embed.size(); i++) {
                new_embed.embed[i] = 0.7f * domain_embeds[src->id].embed[i]
                                   + 0.3f * new_embed.embed[i];
                if (!std::isfinite(new_embed.embed[i])) new_embed.embed[i] = 0.0f;
            }
            new_branch->specialist = src->specialist;  // copy architecture
            branch_generator.populate_specialist(new_branch->specialist, new_embed.embed);
            domain_embeds.emplace(static_cast<int>(branches.size()), std::move(new_embed));
        } else {
            new_branch->specialist = src->specialist;  // copy weights
            // Perturb specialist weights with small noise to break symmetry
            std::normal_distribution<float> nd(0.0f, 0.01f);
            for (auto& layer : new_branch->specialist.layers)
                for (auto& w : layer.weights.data) w += nd(rng);
        }

        new_branch->sub_conductor = SubConductor(
            new_branch->id, input_dim, summary_dim, branches.size() + 1, rng);

        branches.push_back(new_branch);
        conductor.add_branch(rng);
        // Extend growth_controller health vector
        while (growth_controller.health.size() < branches.size())
            growth_controller.health.push_back({});
        topology_changes++;
        printf("[Topology] Split: branch %zu ('%s') → child %zu\n",
               branch_idx, src->domain.c_str(), branches.size() - 1);
    }

    // --------------------------------------------------------
    // Prune a branch: remove from tree, shrink conductor output
    // --------------------------------------------------------
    void prune_branch(size_t branch_idx) {
        if (branches.size() <= growth_controller.min_branches) return;
        if (branch_idx >= branches.size()) return;

        printf("[Topology] Prune: branch %zu ('%s')\n",
               branch_idx, branches[branch_idx]->domain.c_str());
        branches.erase(branches.begin() + branch_idx);
        // Renumber + rebuild sub-conductors (small overhead, infrequent event)
        for (size_t i = 0; i < branches.size(); i++) {
            branches[i]->id = i;
            branches[i]->sub_conductor = SubConductor(
                i, input_dim, summary_dim, branches.size(), rng);
        }
        conductor.remove_branch(branch_idx, rng);
        // Trim health vector
        if (branch_idx < growth_controller.health.size())
            growth_controller.health.erase(growth_controller.health.begin() + branch_idx);
        topology_changes++;
    }

    float train_batch(const std::vector<Tensor>& inputs,
                      const std::vector<Tensor>& targets, int epochs = 1) {
        float last_loss = 0;
        for (int e = 0; e < epochs; e++) {
            std::vector<size_t> idx(inputs.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            float epoch_loss = 0;
            size_t replay_every = (replay_ratio > 0 && !replay_buffer.empty())
                                  ? std::max((size_t)1, (size_t)(1.0f / replay_ratio)) : 0;
            std::uniform_int_distribution<size_t> replay_dist(0,
                std::max((size_t)1, replay_buffer.size()) - 1);
            for (size_t si = 0; si < idx.size(); si++) {
                epoch_loss += train_sample(inputs[idx[si]], targets[idx[si]]);
                replay_buffer.store(inputs[idx[si]], targets[idx[si]]);
                // Interleave replay: every ~5th sample replay one old sample
                if (replay_every > 0 && si % replay_every == 0 && !replay_buffer.empty()) {
                    size_t ri = replay_dist(rng);
                    train_sample(replay_buffer.inputs[ri], replay_buffer.targets[ri]);
                }
            }
            last_loss = epoch_loss / inputs.size();

            // SI consolidation moved to consolidate_at_boundary() — call between tasks

            if (e % 5 == 0)
                std::cout << "  [warmup] effective_lr=" << effective_lr()
                          << " tau=" << current_tau() << "\n";

            // Growth controller: periodic split/prune evaluation
            auto decision = growth_controller.evaluate(branches, inputs.size());
            for (size_t idx : decision.to_split) split_branch(idx);
            for (size_t idx : decision.to_prune) prune_branch(idx);

            // Enhancement #26: hypernetwork meta-loss step (once per epoch)
            if (hypernetwork_enabled) {
                float meta_lr = effective_lr() * hypernetwork_meta_weight;
                for (auto& b : branches) {
                    if (domain_embeds.count(b->id)) {
                        branch_generator.meta_step(domain_embeds[b->id], b->specialist);
                    }
                }
                branch_generator.apply_adam(meta_lr);
                for (auto& kv : domain_embeds) kv.second.apply_adam(meta_lr);
            }
        }
        return last_loss;
    }

    // --------------------------------------------------------
    // Checkpoint: save / load all specialist and conductor weights.
    //
    // Saves: branch specialists, conductor heat/strategy/combiner,
    //        modality projections (if enabled), total_train_steps.
    // Does NOT save: morality config, replay buffer, task context
    //   (morality comes from morality.cfg; context is session-scoped).
    //
    // Usage:
    //   net.save_checkpoint("runs/epoch50.dnrt");
    //   net.load_checkpoint("runs/epoch50.dnrt");
    // --------------------------------------------------------
    [[nodiscard]] bool save_checkpoint(const std::string& path) const {
        CheckpointWriter wr;
        wr.add_scalar("total_train_steps", static_cast<float>(total_train_steps));

        // Branch specialists
        for (size_t i = 0; i < branches.size(); i++) {
            const std::string pfx = "branch" + std::to_string(i) + "_";
            branches[i]->specialist.serialize(wr, pfx + "specialist_");
            // Sub-conductor specialist networks
            branches[i]->sub_conductor.relevance_net.serialize(wr, pfx + "sub_relevance_");
            branches[i]->sub_conductor.summary_net.serialize(wr, pfx + "sub_summary_");
            branches[i]->sub_conductor.redirect_net.serialize(wr, pfx + "sub_redirect_");
            // Child branches (one level deep)
            if (branches[i]->has_children) {
                for (size_t c = 0; c < branches[i]->children.size(); c++) {
                    const std::string cpfx = pfx + "child" + std::to_string(c) + "_";
                    branches[i]->children[c]->specialist.serialize(wr, cpfx + "specialist_");
                }
            }
        }

        // Conductor routing networks
        conductor.heat_network.serialize(wr,     "cond_heat_");
        conductor.strategy_network.serialize(wr, "cond_strategy_");
        conductor.combiner.serialize(wr,         "cond_combiner_");
        if (conductor.concept_bottleneck_enabled) {
            conductor.concept_predictor.serialize(wr, "cond_concept_pred_");
            conductor.concept_heat_net.serialize(wr,  "cond_concept_heat_");
        }
        if (conductor.hierarchical_enabled)
            conductor.guidance_proj.serialize(wr, "cond_guidance_");

        // Modality projections
        if (image_enabled)
            image_module.projection.serialize(wr, "image_proj_");
        if (audio_enabled)
            audio_module.projection.serialize(wr, "audio_proj_");

        const bool ok = wr.save(path);
        if (ok)
            printf("[Checkpoint] Saved %zu entries to %s\n", wr.num_entries(), path.c_str());
        else
            fprintf(stderr, "[Checkpoint] ERROR: failed to write %s\n", path.c_str());
        return ok;
    }

    [[nodiscard]] bool load_checkpoint(const std::string& path) {
        CheckpointReader rd;
        if (!rd.load(path)) {
            fprintf(stderr, "[Checkpoint] ERROR: failed to read %s\n", path.c_str());
            return false;
        }

        float steps_f = 0.f;
        if (rd.restore_scalar("total_train_steps", steps_f))
            total_train_steps = static_cast<size_t>(steps_f);

        for (size_t i = 0; i < branches.size(); i++) {
            const std::string pfx = "branch" + std::to_string(i) + "_";
            branches[i]->specialist.deserialize(rd, pfx + "specialist_");
            branches[i]->sub_conductor.relevance_net.deserialize(rd, pfx + "sub_relevance_");
            branches[i]->sub_conductor.summary_net.deserialize(rd, pfx + "sub_summary_");
            branches[i]->sub_conductor.redirect_net.deserialize(rd, pfx + "sub_redirect_");
            if (branches[i]->has_children) {
                for (size_t c = 0; c < branches[i]->children.size(); c++) {
                    const std::string cpfx = pfx + "child" + std::to_string(c) + "_";
                    branches[i]->children[c]->specialist.deserialize(rd, cpfx + "specialist_");
                }
            }
        }

        conductor.heat_network.deserialize(rd,     "cond_heat_");
        conductor.strategy_network.deserialize(rd, "cond_strategy_");
        conductor.combiner.deserialize(rd,         "cond_combiner_");
        if (conductor.concept_bottleneck_enabled) {
            conductor.concept_predictor.deserialize(rd, "cond_concept_pred_");
            conductor.concept_heat_net.deserialize(rd,  "cond_concept_heat_");
        }
        if (conductor.hierarchical_enabled)
            conductor.guidance_proj.deserialize(rd, "cond_guidance_");
        if (image_enabled)
            image_module.projection.deserialize(rd, "image_proj_");
        if (audio_enabled)
            audio_module.projection.deserialize(rd, "audio_proj_");

        printf("[Checkpoint] Loaded %zu entries from %s  (step=%zu)\n",
               rd.num_entries(), path.c_str(), total_train_steps);
        return true;
    }

    // --------------------------------------------------------
    // Minibatch training — batched GEMM fast path for specialist networks.
    //
    // Routes all samples through the heat network, groups them by assigned
    // branch, then processes each group as a single matrix multiply instead
    // of B separate matvec calls.  A single Adam step is taken per branch
    // per call (vs B steps in train_batch), which is correct mini-batch SGD.
    //
    // Trade-offs vs train_batch:
    //   + 2–6× faster throughput at batch_size >= 32 (cache-friendly GEMM)
    //   + Single Adam update reduces per-sample overhead
    //   – Skips VICReg/contrastive/CLIP diversity losses (call train_batch
    //     every ~5 epochs to maintain branch specialisation)
    //   – Routing networks are not updated (heat/strategy/combiner)
    //
    // Returns mean cross-entropy loss over the minibatch.
    // --------------------------------------------------------
    float train_minibatch(const std::vector<Tensor>& inputs,
                          const std::vector<Tensor>& targets) {
        if (inputs.empty()) return 0.0f;
        const size_t B = inputs.size();
        const float lr = effective_lr();
        float total_loss = 0.0f;

        // --- Route all samples (sequential; routing nets are tiny) ---
        std::vector<std::vector<size_t>> per_branch(branches.size());
        for (size_t i = 0; i < B; i++) {
            Tensor heat = conductor.compute_heat(inputs[i]);
            for (auto& v : heat.data) if (!std::isfinite(v)) v = 0.f;
            int best = heat.argmax();
            if (best < 0 || best >= static_cast<int>(branches.size())) best = 0;
            per_branch[static_cast<size_t>(best)].push_back(i);
            branches[static_cast<size_t>(best)]->visit_count++;
        }

        // --- Batched specialist forward + backward per branch ---
        for (size_t b = 0; b < branches.size(); b++) {
            const auto& idx = per_branch[b];
            if (idx.empty()) continue;
            const size_t nb = idx.size();
            const size_t id = static_cast<size_t>(input_dim);
            const size_t od = static_cast<size_t>(output_dim);

            // Pack samples into batch matrices
            Tensor in_b({nb, id}), tgt_b({nb, od});
            for (size_t k = 0; k < nb; k++) {
                std::copy(inputs[idx[k]].data.begin(),  inputs[idx[k]].data.end(),
                          in_b.data.begin()  + k * id);
                std::copy(targets[idx[k]].data.begin(), targets[idx[k]].data.end(),
                          tgt_b.data.begin() + k * od);
            }

            // Batched forward through all specialist layers
            Tensor out_b = branches[b]->specialist.forward_batch(in_b);

            // Cross-entropy loss + gradient — in-place over out_b to avoid per-sample allocs.
            // Gradient clipped to [-1, 1] per nn-conventions.md.
            Tensor grad_b({nb, od});
            for (size_t k = 0; k < nb; k++) {
                float* logit_row = &out_b.data[k * od];
                float* grad_row  = &grad_b.data[k * od];
                const float* tgt_row = &tgt_b.data[k * od];
                // In-place softmax on logit_row
                float mx = logit_row[0];
                for (size_t d = 1; d < od; d++) if (logit_row[d] > mx) mx = logit_row[d];
                float sum_e = 0.0f;
                for (size_t d = 0; d < od; d++) { logit_row[d] = std::exp(logit_row[d] - mx); sum_e += logit_row[d]; }
                const float inv = sum_e > 1e-7f ? 1.0f / sum_e : 1.0f;
                for (size_t d = 0; d < od; d++) {
                    const float p = logit_row[d] * inv;
                    if (tgt_row[d] > 0.5f)
                        total_loss -= std::log(std::max(p, 1e-7f));
                    grad_row[d] = std::clamp(p - tgt_row[d], -1.0f, 1.0f);
                }
            }

            // Batched backward + single Adam step for this branch
            branches[b]->specialist.backward_batch(grad_b);
            branches[b]->specialist.apply_adam(lr);
            branches[b]->running_loss = total_loss / static_cast<float>(b + 1);
        }

        total_train_steps += B;
        return total_loss / static_cast<float>(B);
    }

    // --------------------------------------------------------
    // Print architecture
    // --------------------------------------------------------
    void print_architecture() const {
        std::cout << "\n  Architecture:\n";
        std::cout << "  ┌─ Conductor (heat + strategy + attention + feedback)\n";
        std::cout << "  │    params: " << conductor.param_count() << "\n";
        if (has_dendrite_layer) {
            std::cout << "  ├─ DendriticLayer (context modulation + kWTA)\n";
            std::cout << "  │    params: " << dendrite_layer.param_count()
                      << "  segments: " << dendrite_layer.num_segments
                      << "  kwta: " << (int)(dendrite_layer.kwta_percent * 100) << "%\n";
        }
        for (size_t i = 0; i < branches.size(); i++) {
            auto& b = branches[i];
            bool last = (i == branches.size() - 1);
            std::cout << "  " << (last ? "└" : "├") << "─ Branch [" << b->id
                      << "] '" << b->domain << "'\n";
            std::cout << "  " << (last ? " " : "│") << "    specialist: "
                      << b->specialist.param_count() << " params\n";
            std::cout << "  " << (last ? " " : "│") << "    sub-conductor: "
                      << b->sub_conductor.param_count() << " params\n";
            if (b->has_children) {
                for (size_t c = 0; c < b->children.size(); c++) {
                    auto& ch = b->children[c];
                    bool clast = (c == b->children.size() - 1);
                    std::cout << "  " << (last ? " " : "│") << "    "
                              << (clast ? "└" : "├") << "─ Sub [" << ch->id
                              << "] '" << ch->domain << "' ("
                              << ch->specialist.param_count() << " params)\n";
                }
            }
        }
        if (has_shared_expert) {
            std::cout << "  └─ SharedExpert (always-on, alpha-gated blend)\n";
            std::cout << "       specialist: " << shared_expert.param_count() << " params\n";
            std::cout << "       alpha_gate: " << alpha_gate.param_count() << " params"
                      << "  (bias_init=-1.0, lr_mult=0.1x)\n";
        }
        std::cout << "\n";
    }

    void print_stats() const {
        std::cout << "\n=== DendriteNet3D v3 Statistics ===\n";
        std::cout << "  Total parameters: " << param_count() << "\n";
        std::cout << "  Total inferences: " << total_inferences << "\n";
        std::cout << "  Total training steps: " << total_train_steps << "\n";
        std::cout << "  Total redirects: " << total_redirects << "\n";
        std::cout << "  Total cross-talk messages: " << total_crosstalk_msgs << "\n";
        conductor.print_strategy_stats();
        std::cout << "\n  Branch visit distribution:\n";
        for (auto& b : branches) {
            printf("    %-20s %6zu visits  loss=%.4f\n",
                   b->domain.c_str(), b->visit_count, b->running_loss);
            if (b->has_children) {
                for (auto& ch : b->children)
                    printf("      child %-16s %6zu visits\n",
                           ch->domain.c_str(), ch->visit_count);
            }
        }
        std::cout << "\n  v3 components:\n";
        task_context.print_state();
        morality.print_stats();
        if (image_enabled) image_module.print_stats();
        if (audio_enabled) audio_module.print_stats();
        printf("  Text preprocessor: %zu stop-words, %.0f%% compression\n",
               text_preprocessor.stop_words.size(),
               text_preprocessor.compression_intensity * 100);
    }

    // --------------------------------------------------------
    // Compute branch specialization metrics over a labelled dataset.
    // labels[i] = ground-truth class index for inputs[i].
    // --------------------------------------------------------
    // Compute branch specialisation metrics over a labelled dataset.
    // Uses top-k routing (matching actual inference) so MI is meaningful
    // regardless of NMDA steepness. Preserves nmda_steps (no training side-effects).
    [[nodiscard]] SpecializationMetrics compute_specialization_metrics(
            const std::vector<Tensor>& inputs,
            const std::vector<int>& labels,
            size_t num_classes) const {
        SpecializationMetrics m;
        m.init(branches.size(), num_classes);
        // Save and restore nmda_steps so diagnostic calls don't advance annealing
        Conductor& cond = const_cast<Conductor&>(conductor);
        size_t saved_steps = cond.nmda_steps;
        for (size_t i = 0; i < inputs.size() && i < labels.size(); i++) {
            Tensor heat = cond.compute_heat(inputs[i]);
            cond.nmda_steps = saved_steps;  // undo step increment
            m.record(heat, static_cast<size_t>(labels[i]), top_k);
        }
        return m;
    }

    size_t param_count() const {
        size_t t = conductor.param_count();
        for (auto& b : branches) t += b->param_count();
        t += task_context.param_count();
        t += morality.param_count();
        if (gca_enabled) t += gated_cross_attn.param_count();
        if (image_enabled) t += image_module.param_count();
        if (audio_enabled) t += audio_module.param_count();
        if (has_dendrite_layer) t += dendrite_layer.param_count();
        if (has_shared_expert) t += shared_expert.param_count() + alpha_gate.param_count();
        return t;
    }

    // --------------------------------------------------------
    // Shapley Values: exact branch attribution via 2^n subset enumeration.
    // Only feasible for small branch counts (≤12). Returns per-branch contribution.
    // --------------------------------------------------------
    [[nodiscard]] std::vector<float> compute_shapley_values(
            const Tensor& input, const Tensor& target) {
        size_t n = branches.size();
        std::vector<float> shapley(n, 0.0f);
        if (n == 0 || n > 20) return shapley;  // guard against huge branch counts

        // Precompute forward pass for every subset (2^n passes)
        size_t total_subsets = 1u << n;
        std::vector<float> subset_loss(total_subsets, 0.0f);

        for (size_t mask = 0; mask < total_subsets; mask++) {
            Tensor out = forward_with_branch_mask(input, mask);
            float L = 0.0f;
            for (size_t i = 0; i < target.size(); i++)
                if (target[i] > 0) L -= target[i] * std::log(std::max(out[i], 1e-7f));
            subset_loss[mask] = std::isfinite(L) ? L : 10.0f;
        }

        // Shapley weight: |S|!(n-|S|-1)! / n!  where S = subset without branch b
        // Precompute factorials
        std::vector<float> fact(n + 1, 1.0f);
        for (size_t i = 1; i <= n; i++) fact[i] = fact[i-1] * (float)i;

        for (size_t b = 0; b < n; b++) {
            float phi = 0.0f;
            for (size_t mask = 0; mask < total_subsets; mask++) {
                if (!(mask & (1u << b))) continue;  // b must be IN the set
                size_t without_b = mask & ~(1u << b);
                size_t s = (size_t)__builtin_popcountll(without_b);  // |S without b|
                float w = fact[s] * fact[n - s - 1] / fact[n];
                float marginal = subset_loss[without_b] - subset_loss[mask];
                phi += w * marginal;
            }
            shapley[b] = phi;
        }
        return shapley;
    }

private:
    // Track which specialist was actually used for each branch
    std::unordered_map<int, MiniNetwork*> last_used_specialist;

    // --------------------------------------------------------
    // Debate-Inspired Verification: when uncertain (max heat < debate_threshold),
    // each of the top-2 branches re-evaluates the other's output. The branch whose
    // output is more self-consistent under this adversarial probe gets higher heat.
    // --------------------------------------------------------
    // Enhancement #27: Hierarchical message passing.
    // Phase 1 (bottom-up): children → parent.child_context
    // Phase 2 (top-down): conductor → branch.conductor_guidance
    // Phase 3 (lateral): existing cross-talk (handled separately).
    // --------------------------------------------------------
    void hierarchical_pass(const Tensor& heat) {
        if (!conductor.hierarchical_enabled) return;

        // Phase 1: Bottom-up — aggregate child outputs into parent.child_context
        for (auto& b : branches) {
            if (!b->has_children || b->children.empty()) {
                b->child_context = Tensor({summary_dim});
                continue;
            }
            // Mean-pool children's last specialist output (via their sub_conductor summary)
            Tensor agg({summary_dim});
            for (const auto& child : b->children) {
                // Use child's cross-talk summary if available, else zeros
                for (const auto& msg : prev_cross_talk) {
                    if (msg.from_branch == child->id && msg.summary.size() > 0) {
                        size_t n = std::min(agg.size(), msg.summary.size());
                        for (size_t i = 0; i < n; i++) agg[i] += msg.summary[i];
                        break;
                    }
                }
            }
            float inv = 1.0f / static_cast<float>(b->children.size());
            for (auto& v : agg.data) {
                v *= inv;
                if (!std::isfinite(v)) v = 0.0f;
            }
            b->child_context = agg;
        }

        // Phase 2: Top-down — conductor sends heat-based guidance to each branch
        float max_heat = 0.0f;
        for (size_t i = 0; i < heat.size(); i++) max_heat = std::max(max_heat, heat[i]);
        for (size_t i = 0; i < branches.size() && i < heat.size(); i++) {
            Tensor guide_in({2});
            guide_in[0] = heat[i];
            guide_in[1] = max_heat;
            Tensor guidance = conductor.guidance_proj.forward(guide_in);
            for (auto& v : guidance.data) if (!std::isfinite(v)) v = 0.0f;
            branches[i]->conductor_guidance = guidance;
        }
    }

    // --------------------------------------------------------
    // Enhancement #28: Kuramoto oscillator phase update.
    // Each branch i: Δθᵢ = ωᵢ + (K/N) Σⱼ sin(θⱼ−θᵢ)·heatⱼ
    // Branches with aligned phases weight cross-talk more heavily.
    // --------------------------------------------------------
    void update_phases(const Tensor& heat) {
        if (!conductor.oscillatory_sync || branches.empty()) return;
        const float K  = conductor.osc_coupling;
        const float dt = conductor.osc_dt;
        const float TWO_PI = 2.0f * 3.14159265f;
        const size_t n = branches.size();
        std::vector<float> new_phases(n);
        for (size_t i = 0; i < n; i++) {
            float coupling = 0.0f;
            for (size_t j = 0; j < n; j++) {
                if (i == j) continue;
                float hj = (j < heat.size()) ? heat[j] : 0.0f;
                coupling += std::sin(branches[j]->phase - branches[i]->phase) * hj;
            }
            coupling *= K / static_cast<float>(n);
            new_phases[i] = branches[i]->phase + dt * (branches[i]->natural_freq + coupling);
            new_phases[i] = std::fmod(new_phases[i], TWO_PI);
            if (new_phases[i] < 0.0f) new_phases[i] += TWO_PI;
        }
        for (size_t i = 0; i < n; i++) branches[i]->phase = new_phases[i];
    }

    // Modulates adjusted_heat in-place.
    // --------------------------------------------------------
    void apply_debate_to_heat(Tensor& adjusted_heat,
                               const std::vector<Tensor>& branch_outputs,
                               const std::vector<int>& active,
                               const Tensor& input) {
        if (!conductor.debate_enabled || active.size() < 2) return;
        // Gate: only trigger when routing is uncertain (max heat < threshold)
        float max_heat = 0.0f;
        for (int b : active) max_heat = std::max(max_heat, adjusted_heat[b]);
        if (max_heat >= conductor.debate_threshold) return;

        // Find top-2 active branches by heat
        std::vector<std::pair<float, int>> sorted;
        for (int b : active) sorted.push_back({adjusted_heat[b], b});
        std::sort(sorted.rbegin(), sorted.rend());
        int a = sorted[0].second, b = sorted[1].second;

        // Self-consistency: each branch's specialist re-processes the original input
        // and we compare its output to the cached branch_outputs to measure stability.
        Tensor a_self = branches[a]->specialist.forward(input);
        Tensor b_self = branches[b]->specialist.forward(input);
        float a_self_sim = a_self.cosine_similarity(branch_outputs[a]);
        float b_self_sim = b_self.cosine_similarity(branch_outputs[b]);
        if (!std::isfinite(a_self_sim)) a_self_sim = 0.5f;
        if (!std::isfinite(b_self_sim)) b_self_sim = 0.5f;

        // Modulate heat by robustness (self-consistency score), clamped to [0.1, 1]
        adjusted_heat[a] *= std::max(0.1f, a_self_sim);
        adjusted_heat[b] *= std::max(0.1f, b_self_sim);
        adjusted_heat[a] = std::clamp(adjusted_heat[a], 0.0f, 1.0f);
        adjusted_heat[b] = std::clamp(adjusted_heat[b], 0.0f, 1.0f);
    }

    // Forward pass with only branches in `mask` active (bitmask, bit b = branch b).
    // Branches outside the mask contribute zero. Used by Shapley value computation.
    Tensor forward_with_branch_mask(const Tensor& input, size_t mask) {
        size_t n = branches.size();
        std::vector<Tensor> branch_outputs(n, Tensor({output_dim}));
        // Uniform heat for active branches in mask
        Tensor heat({n});
        size_t active_count = 0;
        for (size_t b = 0; b < n; b++) if (mask & (1u << b)) active_count++;
        float uniform_heat = active_count > 0 ? 1.0f / active_count : 0.0f;
        std::vector<int> active;
        for (size_t b = 0; b < n; b++) {
            if (mask & (1u << b)) {
                heat[b] = uniform_heat;
                active.push_back((int)b);
                branch_outputs[b] = const_cast<DendriteNet3D*>(this)->evaluate_branch(
                    branches[b], input, true);
                for (auto& v : branch_outputs[b].data)
                    if (!std::isfinite(v)) v = 1.0f / output_dim;
            }
        }
        if (active.empty()) {
            Tensor uniform({output_dim});
            uniform.fill(1.0f / output_dim);
            return uniform;
        }
        Tensor fused = conductor.fuse(FusionStrategy::WEIGHTED_BLEND, heat, branch_outputs, active);
        Tensor out = conductor.finalize(fused);
        for (auto& v : out.data) if (!std::isfinite(v)) v = 1.0f / output_dim;
        return out;
    }

    // is_inference=true  → forward with early exit; false → forward_full for training
    // context → dendritic modulation applied before specialist
    // branch_idx → index into branches[] for task-head lookup
    Tensor evaluate_branch(const BranchPtr& branch, const Tensor& input,
                           bool is_inference = false,
                           const Tensor& context = Tensor(),
                           int branch_idx = -1) {
        // Apply dendritic modulation to input before the specialist
        Tensor modulated_input = input;
        if (has_dendrite_layer && context.size() > 0) {
            modulated_input = dendrite_layer.forward(input, context);
            for (auto& v : modulated_input.data)
                if (!std::isfinite(v)) v = 0.0f;
        }

        if (branch->has_children) {
            Tensor child_probs = branch->child_router.forward(input);
            int best = child_probs.argmax();
            branch->last_child_probs  = child_probs;
            branch->last_chosen_child = best;
            if (best < (int)branch->children.size()) {
                branch->children[best]->visit_count++;
                last_used_specialist[branch->id] = &branch->children[best]->specialist;
                return is_inference
                    ? branch->children[best]->specialist.forward(modulated_input)
                    : branch->children[best]->specialist.forward_full(modulated_input);
            }
        }
        last_used_specialist[branch->id] = &branch->specialist;

        // Per-task output head: run backbone (all layers except last), then task head
        if (current_task_id >= 0 && task_heads.count(current_task_id) &&
            branch_idx >= 0 && branch_idx < (int)task_heads[current_task_id].size()) {
            auto& spec = branch->specialist;
            Tensor x = modulated_input;
            // Run all layers except the last (backbone)
            for (size_t l = 0; l + 1 < spec.layers.size(); l++) {
                x = spec.layers[l].forward(x);
                if (l == 1 && spec.exit_classifier.has_value())
                    spec.last_exit_hidden = x;  // cache for early exit
            }
            // Apply task-specific output head instead of specialist's last layer
            return task_heads[current_task_id][branch_idx].forward(x);
        }

        bool allow_exit = is_inference && (total_train_steps >= exit_enabled_after_steps);
        Tensor prox_out = is_inference
            ? branch->specialist.forward(modulated_input, allow_exit)
            : branch->specialist.forward_full(modulated_input);

        // Enhancement #29: Distal (context) compartment modulates proximal output
        if (branch->has_distal) {
            // Distal input: concat(context, prev_cross_talk_summary_for_this_branch)
            Tensor ctx = context;
            if (ctx.size() == 0) ctx = Tensor({output_dim});  // fallback: zeros

            // Find this branch's cross-talk summary from previous step
            Tensor ct_summary({summary_dim});
            for (const auto& msg : prev_cross_talk) {
                if (msg.from_branch == branch->id && msg.summary.size() > 0) {
                    size_t copy_n = std::min(ct_summary.size(), msg.summary.size());
                    for (size_t i = 0; i < copy_n; i++) ct_summary[i] = msg.summary[i];
                    break;
                }
            }

            // Build distal input (pad/trim to expected size)
            // Enhancement #27: blend conductor guidance into cross-talk summary
            Tensor ct_blended = ct_summary;
            if (conductor.hierarchical_enabled && branch->conductor_guidance.size() == summary_dim) {
                for (size_t i = 0; i < summary_dim; i++) {
                    ct_blended[i] = 0.7f * ct_summary[i] + 0.3f * branch->conductor_guidance[i];
                    if (!std::isfinite(ct_blended[i])) ct_blended[i] = ct_summary[i];
                }
            }
            Tensor distal_input({output_dim + summary_dim});
            size_t clen = std::min(ctx.size(), (size_t)output_dim);
            for (size_t i = 0; i < clen; i++) distal_input[i] = ctx[i];
            for (size_t i = 0; i < summary_dim; i++)
                distal_input[output_dim + i] = ct_blended[i];

            Tensor dist_out = branch->distal.forward(distal_input);  // tanh activation
            for (auto& v : dist_out.data) if (!std::isfinite(v)) v = 0.0f;

            // Cache for backward pass
            branch->last_prox_out = prox_out;
            branch->last_dist_out = dist_out;

            // Passive coupling: distal modulates proximal ±coupling
            Tensor combined({prox_out.size()});
            for (size_t d = 0; d < prox_out.size() && d < dist_out.size(); d++) {
                combined[d] = prox_out[d] * (1.0f + branch->distal_coupling * dist_out[d]);
                if (!std::isfinite(combined[d])) combined[d] = prox_out[d];
            }
            return combined;
        }

        return prox_out;
    }
};

} // namespace dendrite
