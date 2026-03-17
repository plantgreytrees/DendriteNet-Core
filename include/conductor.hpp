#pragma once
#include "layer.hpp"
#include <array>

namespace dendrite {

// ============================================================
// Fusion strategies the conductor can dynamically choose
// ============================================================
enum class FusionStrategy {
    WEIGHTED_BLEND,   // Mix all outputs by heat scores
    TOP_K_BLEND,      // Pick top-K hottest, blend those
    ATTENTION_FUSE,   // Branches attend to each other, then fuse
    VOTING,           // Each branch votes, majority wins
    NUM_STRATEGIES
};

static const char* fusion_name(FusionStrategy s) {
    switch (s) {
        case FusionStrategy::WEIGHTED_BLEND: return "weighted_blend";
        case FusionStrategy::TOP_K_BLEND:    return "top_k_blend";
        case FusionStrategy::ATTENTION_FUSE:  return "attention_fuse";
        case FusionStrategy::VOTING:          return "voting";
        default: return "unknown";
    }
}

// ============================================================
// BranchSignal: what a sub-conductor sends back to the root
// ============================================================
struct BranchSignal {
    int branch_id;
    float heat;             // how relevant this branch thinks it is (0-1)
    float confidence;       // how confident the branch is in its output
    Tensor summary;         // compressed representation of branch state
    Tensor output;          // the branch's actual prediction
    bool wants_redirect;    // sub-conductor thinks input belongs elsewhere
    int redirect_suggestion;// which branch it should go to instead
};

// ============================================================
// CrossTalkMessage: lateral communication between branches
// ============================================================
struct CrossTalkMessage {
    int from_branch;
    Tensor summary;         // compressed state vector shared with siblings
};

// ============================================================
// Conductor: the root-level orchestrator
// ============================================================
// The conductor:
//   1. Computes heat scores for all branches given input
//   2. Dynamically picks a fusion strategy based on input characteristics
//   3. Gathers outputs from hot branches
//   4. Receives feedback (signals) from sub-conductors
//   5. Makes the final call on how to combine results
// ============================================================
class Conductor {
public:
    // Heat network: input -> heat score per branch
    MiniNetwork heat_network;

    // Strategy selector: input + heat scores -> strategy choice
    MiniNetwork strategy_network;

    // Attention fusion: computes cross-branch attention weights
    MiniNetwork query_proj;   // branch output -> query
    MiniNetwork key_proj;     // branch output -> key
    MiniNetwork value_proj;   // branch output -> value

    // Final combiner: fused result -> output
    MiniNetwork combiner;

    // Feedback integrator: processes sub-conductor signals
    MiniNetwork feedback_network;

    size_t input_dim;
    size_t output_dim;
    size_t num_branches;
    size_t summary_dim;       // dimension of cross-talk summaries
    size_t attn_dim;          // attention projection dimension
    int top_k;

    // Concept Bottleneck: interpretable intermediate routing layer
    MiniNetwork concept_predictor;  // input → num_concepts (SIGMOID)
    MiniNetwork concept_heat_net;   // concepts → num_branches (SIGMOID)
    size_t num_concepts = 0;
    bool concept_bottleneck_enabled = false;
    std::vector<std::string> concept_names;  // optional human-readable labels

    // Running statistics
    std::array<size_t, (size_t)FusionStrategy::NUM_STRATEGIES> strategy_counts = {};
    size_t total_decisions = 0;

    // Load balancing bias (gradient-free)
    std::vector<float> branch_bias;
    std::vector<float> running_usage;
    float bias_update_rate = 0.005f;
    float target_usage = 0.25f;
    bool load_balancing_enabled = false;

    // NMDA spike thresholding
    bool nmda_enabled = true;
    float nmda_steepness = 10.0f;        // final steepness (anneals from 3)
    float nmda_threshold = 0.3f;         // NMDA spike threshold
    size_t nmda_steps = 0;               // steps for steepness annealing
    float nmda_anneal_steps = 30000.0f;  // anneal window; 1000 was too fast for class-conditional learning

    // Enhancement #27: Hierarchical message passing (top-down guidance from conductor)
    MiniNetwork guidance_proj;      // [heat_b, max_heat] → summary_dim guidance vector
    bool hierarchical_enabled = false;

    // Enhancement #28: Oscillatory synchronization (Kuramoto model)
    bool oscillatory_sync = false;
    float osc_coupling = 0.5f;   // coupling strength K: higher → faster synchronization
    float osc_dt = 0.1f;         // integration step size

    // NMDA-style activation: steep sigmoid shifted by threshold
    // steepness=3 (soft, early training) → 10 (crisp, converged)
    static float nmda_activation(float x, float threshold, float steepness) {
        float shifted = steepness * (x - threshold);
        shifted = std::clamp(shifted, -20.0f, 20.0f);
        float result = 1.0f / (1.0f + std::exp(-shifted));
        return std::isfinite(result) ? result : (x >= threshold ? 1.0f : 0.0f);
    }

    Conductor() = default;

    Conductor(size_t input_dim_, size_t output_dim_, size_t num_branches_,
              size_t summary_dim_, size_t attn_dim_, int top_k_,
              std::mt19937& rng)
        : input_dim(input_dim_), output_dim(output_dim_),
          num_branches(num_branches_), summary_dim(summary_dim_),
          attn_dim(attn_dim_), top_k(top_k_) {

        size_t hidden = 48;

        // Heat network: input -> [num_branches] sigmoid scores
        heat_network = MiniNetwork("heat",
            {input_dim, hidden, num_branches},
            Activation::RELU, Activation::SIGMOID, rng);

        // Strategy selector: input + heat scores -> 4 strategies (softmax)
        strategy_network = MiniNetwork("strategy",
            {input_dim + num_branches, hidden / 2,
             (size_t)FusionStrategy::NUM_STRATEGIES},
            Activation::RELU, Activation::SOFTMAX, rng);

        // Attention projections (for ATTENTION_FUSE strategy)
        attn_dim = std::max(attn_dim_, (size_t)16);
        query_proj = MiniNetwork("attn_q", {output_dim, attn_dim}, Activation::NONE, Activation::NONE, rng);
        key_proj   = MiniNetwork("attn_k", {output_dim, attn_dim}, Activation::NONE, Activation::NONE, rng);
        value_proj = MiniNetwork("attn_v", {output_dim, attn_dim}, Activation::NONE, Activation::NONE, rng);

        // Combiner: takes fused output and produces final prediction
        combiner = MiniNetwork("combiner",
            {output_dim, hidden, output_dim},
            Activation::RELU, Activation::SOFTMAX, rng);

        // Feedback integrator: processes sub-conductor signals
        // Input: concatenation of (heat_scores, confidence_scores, redirect_flags)
        feedback_network = MiniNetwork("feedback",
            {num_branches * 3, hidden / 2, num_branches},
            Activation::RELU, Activation::SIGMOID, rng);
    }

    // --------------------------------------------------------
    // Initialise concept bottleneck routing
    // Inserts a num_concepts-dim interpretable layer before heat routing.
    // After calling this, compute_heat() routes: input → concepts → heat.
    // --------------------------------------------------------
    void init_concept_bottleneck(std::mt19937& rng, size_t n_concepts = 8) {
        num_concepts = n_concepts;
        size_t hidden = 32;

        // concept_predictor: input → concepts (SIGMOID = each concept 0..1)
        concept_predictor = MiniNetwork("concept_pred",
            {input_dim, hidden, num_concepts},
            Activation::RELU, Activation::SIGMOID, rng);

        // concept_heat_net: concepts → branch heat scores (SIGMOID)
        concept_heat_net = MiniNetwork("concept_heat",
            {num_concepts, hidden, num_branches},
            Activation::RELU, Activation::SIGMOID, rng);

        // Default concept names
        concept_names = {"spatial", "temporal", "logical", "social",
                         "numeric", "linguistic", "causal", "abstract"};
        if (n_concepts != 8) {
            concept_names.clear();
            for (size_t i = 0; i < n_concepts; i++)
                concept_names.push_back("concept_" + std::to_string(i));
        }

        concept_bottleneck_enabled = true;
    }

    // --------------------------------------------------------
    // Initialise load-balancing bias vectors
    // --------------------------------------------------------
    // top_k: how many branches are selected per step — determines fair share per branch
    void init_load_balancing(size_t top_k = 1) {
        target_usage = std::min(1.0f,
            (float)top_k / (float)std::max((size_t)1, num_branches));
        branch_bias.assign(num_branches, 0.0f);
        running_usage.assign(num_branches, target_usage);
        load_balancing_enabled = true;
    }

    // --------------------------------------------------------
    // Enhancement #27: Initialise hierarchical message passing.
    // Creates the top-down guidance projector: [heat_b, max_heat] → summary_dim.
    // --------------------------------------------------------
    void init_hierarchical(std::mt19937& rng) {
        guidance_proj = MiniNetwork("hierarchical_guidance",
            {2, summary_dim},
            Activation::NONE, Activation::TANH, rng);
        hierarchical_enabled = true;
    }

    // --------------------------------------------------------
    // Compute heat scores: how "hot" is each branch for this input
    // Optionally routes through concept bottleneck when enabled.
    // last_concepts is filled with concept scores (diagnostic).
    // --------------------------------------------------------
    Tensor compute_heat(const Tensor& input, Tensor* last_concepts = nullptr) {
        Tensor heat;
        if (concept_bottleneck_enabled) {
            // input → concept predictor → concept scores (SIGMOID, 0..1)
            Tensor concepts = concept_predictor.forward(input);
            for (auto& v : concepts.data) {
                if (!std::isfinite(v)) v = 0.0f;
            }
            if (last_concepts) *last_concepts = concepts;

            // concept scores → branch heat (via concept_heat_net)
            heat = concept_heat_net.forward(concepts);
        } else {
            heat = heat_network.forward(input);
        }
        for (auto& v : heat.data) if (!std::isfinite(v)) v = 0.0f;
        if (load_balancing_enabled) {
            for (size_t b = 0; b < heat.size() && b < branch_bias.size(); b++) {
                heat[b] += branch_bias[b];
                heat[b] = std::clamp(heat[b], 0.0f, 1.0f);
            }
        }
        // NMDA spike thresholding: anneal steepness from 3 → nmda_steepness over nmda_anneal_steps
        if (nmda_enabled) {
            nmda_steps++;
            float t = std::min(1.0f, (float)nmda_steps / nmda_anneal_steps);
            float steepness = 3.0f + t * (nmda_steepness - 3.0f);
            for (size_t b = 0; b < heat.size(); b++) {
                heat[b] = nmda_activation(heat[b], nmda_threshold, steepness);
            }
        }
        return heat;
    }

    // --------------------------------------------------------
    // Update load-balancing bias using EMA of actual top-k selection.
    // Tracks rank (was this branch in the top-k?) rather than a flat
    // threshold, so branches that lose every top-k contest correctly
    // accumulate a positive bias even when their heat exceeds 0.15.
    // --------------------------------------------------------
    void update_load_balancing(const Tensor& heat, int k) {
        if (!load_balancing_enabled) return;
        // Rank branches by heat; mark the top-k as selected
        std::vector<std::pair<float, size_t>> ranked;
        for (size_t b = 0; b < heat.size() && b < running_usage.size(); b++)
            ranked.push_back({heat[b], b});
        std::sort(ranked.rbegin(), ranked.rend());
        std::vector<bool> in_topk(running_usage.size(), false);
        for (int i = 0; i < k && i < (int)ranked.size(); i++)
            in_topk[ranked[i].second] = true;

        for (size_t b = 0; b < heat.size() && b < running_usage.size(); b++) {
            float activated = in_topk[b] ? 1.0f : 0.0f;
            running_usage[b] = 0.99f * running_usage[b] + 0.01f * activated;
            float usage_error = target_usage - running_usage[b];
            branch_bias[b] += bias_update_rate * usage_error;
            branch_bias[b] = std::clamp(branch_bias[b], -0.5f, 0.5f);
        }
    }

    // --------------------------------------------------------
    // Choose fusion strategy dynamically based on input + heat
    // --------------------------------------------------------
    std::pair<FusionStrategy, Tensor> choose_strategy(
            const Tensor& input, const Tensor& heat,
            bool training, float tau, std::mt19937& rng) {
        Tensor combined = Tensor::concat(input, heat);
        Tensor strategy_probs = strategy_network.forward(combined);

        Tensor weights;
        FusionStrategy strat;
        if (training) {
            weights = strategy_probs.gumbel_softmax(tau, rng);
            strat = static_cast<FusionStrategy>(weights.argmax());
        } else {
            int best = strategy_probs.argmax();
            strat = static_cast<FusionStrategy>(best);
            weights = Tensor({(size_t)FusionStrategy::NUM_STRATEGIES});
            weights[best] = 1.0f;
        }
        strategy_counts[(size_t)strat]++;
        total_decisions++;
        return {strat, weights};
    }

    // --------------------------------------------------------
    // Process sub-conductor feedback to adjust heat scores
    // --------------------------------------------------------
    Tensor integrate_feedback(const Tensor& heat,
                              const std::vector<BranchSignal>& signals) {
        // Build feedback feature vector:
        // [heat_0, heat_1, ..., conf_0, conf_1, ..., redir_0, redir_1, ...]
        Tensor feedback_input({num_branches * 3});
        for (size_t i = 0; i < num_branches; i++) {
            feedback_input[i] = heat[i];
            if (i < signals.size()) {
                feedback_input[num_branches + i] = signals[i].confidence;
                feedback_input[num_branches * 2 + i] = signals[i].wants_redirect ? 1.0f : 0.0f;
            }
        }

        // Feedback network outputs adjusted heat scores
        Tensor adjusted = feedback_network.forward(feedback_input);

        // Blend original heat with feedback adjustment (root has final say)
        Tensor final_heat({num_branches});
        for (size_t i = 0; i < num_branches; i++) {
            final_heat[i] = 0.6f * heat[i] + 0.4f * adjusted[i];  // root weighs more
        }
        return final_heat;
    }

    // --------------------------------------------------------
    // Lateral inhibition: strongly activated branches suppress similar weaker branches.
    // Uses previous-step cached cross-talk summaries for inter-branch similarity.
    // alpha controls inhibition strength (0.3 = moderate suppression).
    // --------------------------------------------------------
    Tensor apply_lateral_inhibition(const Tensor& heat,
                                    const std::vector<CrossTalkMessage>& cross_talk,
                                    float alpha = 0.3f) const {
        Tensor inhibited = heat;
        size_t n = heat.size();

        for (size_t i = 0; i < n; i++) {
            if (heat[i] < 0.1f) continue;  // Only dominant branches inhibit others
            for (size_t j = 0; j < n; j++) {
                if (i == j) continue;
                float sim = 0.0f;
                if (i < cross_talk.size() && j < cross_talk.size()) {
                    sim = cross_talk[i].summary.cosine_similarity(cross_talk[j].summary);
                    sim = std::max(0.0f, sim);  // Only positive similarity inhibits
                }
                inhibited[j] -= alpha * heat[i] * sim;
            }
        }

        // Clamp to [0, 1] and NaN guard
        for (size_t i = 0; i < n; i++) {
            inhibited[i] = std::clamp(inhibited[i], 0.0f, 1.0f);
            if (!std::isfinite(inhibited[i])) inhibited[i] = 0.0f;
        }
        return inhibited;
    }

    // --------------------------------------------------------
    // Fuse branch outputs using the chosen strategy
    // --------------------------------------------------------
    Tensor fuse(FusionStrategy strategy, const Tensor& heat,
                const std::vector<Tensor>& branch_outputs,
                const std::vector<int>& active_branches) {

        Tensor fused({output_dim});

        switch (strategy) {
        case FusionStrategy::WEIGHTED_BLEND: {
            // All branches contribute, weighted by heat
            float total_heat = 0;
            for (int b : active_branches) total_heat += heat[b];
            if (total_heat < 1e-7f) total_heat = 1.0f;

            for (int b : active_branches) {
                float w = heat[b] / total_heat;
                for (size_t i = 0; i < output_dim; i++)
                    fused[i] += branch_outputs[b][i] * w;
            }
            break;
        }

        case FusionStrategy::TOP_K_BLEND: {
            // Only top-K hottest branches contribute
            std::vector<std::pair<float, int>> scored;
            for (int b : active_branches)
                scored.push_back({heat[b], b});
            std::sort(scored.rbegin(), scored.rend());

            float total_w = 0;
            int k = std::min(top_k, (int)scored.size());
            for (int i = 0; i < k; i++) total_w += scored[i].first;
            if (total_w < 1e-7f) total_w = 1.0f;

            for (int i = 0; i < k; i++) {
                int b = scored[i].second;
                float w = scored[i].first / total_w;
                for (size_t j = 0; j < output_dim; j++)
                    fused[j] += branch_outputs[b][j] * w;
            }
            break;
        }

        case FusionStrategy::ATTENTION_FUSE: {
            // Cross-branch attention: branches attend to each other
            size_t n = active_branches.size();
            if (n == 0) break;

            // Project to Q, K, V
            std::vector<Tensor> queries, keys, values;
            for (int b : active_branches) {
                queries.push_back(query_proj.forward(branch_outputs[b]));
                keys.push_back(key_proj.forward(branch_outputs[b]));
                values.push_back(value_proj.forward(branch_outputs[b]));
            }

            // Compute attention scores: Q_i dot K_j / sqrt(d)
            float scale = 1.0f / std::sqrt((float)attn_dim);
            std::vector<Tensor> attended(n, Tensor({attn_dim}));

            for (size_t i = 0; i < n; i++) {
                // Compute softmax attention weights for branch i over all branches
                Tensor scores({n});
                for (size_t j = 0; j < n; j++)
                    scores[j] = Tensor::dot(queries[i].data.data(),
                                            keys[j].data.data(), attn_dim) * scale;
                scores = scores.softmax();

                // Weighted sum of values
                for (size_t j = 0; j < n; j++)
                    for (size_t d = 0; d < attn_dim; d++)
                        attended[i][d] += values[j][d] * scores[j];
            }

            // Average the attended representations and project back
            Tensor avg_attended = Tensor::mean(attended);
            // Simple projection back to output_dim
            for (size_t i = 0; i < output_dim && i < attn_dim; i++)
                fused[i] = avg_attended[i];
            // Residual connection from weighted blend
            float total_h = 0;
            for (int b : active_branches) total_h += heat[b];
            if (total_h < 1e-7f) total_h = 1.0f;
            for (int b : active_branches) {
                float w = heat[b] / total_h;
                for (size_t i = 0; i < output_dim; i++)
                    fused[i] += branch_outputs[b][i] * w * 0.5f;
            }
            break;
        }

        case FusionStrategy::VOTING: {
            // Each branch "votes" for its top prediction
            Tensor votes({output_dim});
            for (int b : active_branches) {
                int vote = branch_outputs[b].argmax();
                votes[vote] += heat[b];  // weighted vote
            }
            // Convert votes to probabilities
            float total_v = votes.sum();
            if (total_v > 1e-7f)
                for (size_t i = 0; i < output_dim; i++) fused[i] = votes[i] / total_v;
            break;
        }

        default: break;
        }

        return fused;
    }

    // --------------------------------------------------------
    // Final output: combiner refines the fused result
    // --------------------------------------------------------
    Tensor finalize(const Tensor& fused) {
        return combiner.forward(fused);
    }

    // Debate parameters (logic implemented in DendriteNet3D::apply_debate_to_heat)
    bool debate_enabled = true;
    float debate_threshold = 0.6f;  // trigger debate when max heat < this

    // --------------------------------------------------------
    // --------------------------------------------------------
    // Dynamic topology: add/remove a branch output from the heat network
    // Strategy and feedback networks are rebuilt fresh (small, relearn quickly)
    // --------------------------------------------------------
    void add_branch(std::mt19937& rng) {
        if (concept_bottleneck_enabled)
            concept_heat_net.add_output(rng);
        else
            heat_network.add_output(rng);
        // Rebuild strategy + feedback with new num_branches count
        num_branches++;
        size_t hidden = 48;
        strategy_network = MiniNetwork("strategy",
            {input_dim + num_branches, hidden / 2,
             (size_t)FusionStrategy::NUM_STRATEGIES},
            Activation::RELU, Activation::SOFTMAX, rng);
        feedback_network = MiniNetwork("feedback",
            {num_branches * 3, hidden / 2, num_branches},
            Activation::RELU, Activation::SIGMOID, rng);
        if (load_balancing_enabled) {
            branch_bias.push_back(0.0f);
            running_usage.push_back(target_usage);
        }
    }

    void remove_branch(size_t branch_idx, std::mt19937& rng) {
        if (num_branches <= 2) return;  // minimum
        if (concept_bottleneck_enabled)
            concept_heat_net.remove_output(branch_idx);
        else
            heat_network.remove_output(branch_idx);
        num_branches--;
        size_t hidden = 48;
        strategy_network = MiniNetwork("strategy",
            {input_dim + num_branches, hidden / 2,
             (size_t)FusionStrategy::NUM_STRATEGIES},
            Activation::RELU, Activation::SOFTMAX, rng);
        feedback_network = MiniNetwork("feedback",
            {num_branches * 3, hidden / 2, num_branches},
            Activation::RELU, Activation::SIGMOID, rng);
        if (load_balancing_enabled && branch_idx < branch_bias.size()) {
            branch_bias.erase(branch_bias.begin() + branch_idx);
            running_usage.erase(running_usage.begin() + branch_idx);
        }
    }

    // Training
    // --------------------------------------------------------
    void apply_adam(float lr) {
        if (concept_bottleneck_enabled) {
            concept_predictor.apply_adam(lr);
            concept_heat_net.apply_adam(lr);
        } else {
            heat_network.apply_adam(lr);
        }
        strategy_network.apply_adam(lr);
        query_proj.apply_adam(lr);
        key_proj.apply_adam(lr);
        value_proj.apply_adam(lr);
        combiner.apply_adam(lr);
        feedback_network.apply_adam(lr);
    }

    size_t param_count() const {
        size_t heat_params = concept_bottleneck_enabled
            ? concept_predictor.param_count() + concept_heat_net.param_count()
            : heat_network.param_count();
        return heat_params + strategy_network.param_count()
             + query_proj.param_count() + key_proj.param_count()
             + value_proj.param_count() + combiner.param_count()
             + feedback_network.param_count();
    }

    void print_strategy_stats() const {
        if (total_decisions == 0) return;
        std::cout << "  Strategy usage:\n";
        for (int i = 0; i < (int)FusionStrategy::NUM_STRATEGIES; i++) {
            float pct = 100.0f * strategy_counts[i] / total_decisions;
            printf("    %-18s %5zu (%5.1f%%)\n",
                   fusion_name(static_cast<FusionStrategy>(i)),
                   strategy_counts[i], pct);
        }
    }
};

// ============================================================
// SubConductor: lives inside each branch, communicates upward
// ============================================================
// Enhancement 09: GAT Cross-Talk — Q/K/V projections for selective
// attention-weighted inter-branch message passing. Each branch projects
// its summary into query/key/value spaces; gat_enrich() computes
// attention over sibling messages and adds a residual to the summary.
// ============================================================
class SubConductor {
public:
    MiniNetwork relevance_net;   // assesses how relevant this branch is
    MiniNetwork summary_net;     // compresses branch state for cross-talk
    MiniNetwork redirect_net;    // decides if input should go elsewhere

    // GAT projections: summary_dim → gat_dim
    MiniNetwork q_proj;   // query: what this branch is looking for from siblings
    MiniNetwork k_proj;   // key:   what this branch offers to siblings
    MiniNetwork v_proj;   // value: content this branch shares

    size_t input_dim;
    size_t summary_dim;
    size_t gat_dim = 16;
    int branch_id;
    bool has_gat = false;

    SubConductor() = default;

    SubConductor(int branch_id_, size_t input_dim_, size_t summary_dim_,
                 size_t num_siblings, std::mt19937& rng)
        : input_dim(input_dim_), summary_dim(summary_dim_), branch_id(branch_id_) {

        // Relevance: how well does this branch match the input?
        relevance_net = MiniNetwork("sub_relevance",
            {input_dim_, 24, 1}, Activation::RELU, Activation::SIGMOID, rng);

        // Summary: compress branch state for siblings to read
        summary_net = MiniNetwork("sub_summary",
            {input_dim_, 32, summary_dim_}, Activation::RELU, Activation::TANH, rng);

        // Redirect: should this input go to a different branch?
        redirect_net = MiniNetwork("sub_redirect",
            {input_dim_ + summary_dim_, 16, num_siblings},
            Activation::RELU, Activation::SOFTMAX, rng);

        // GAT projections (Xavier init for attention)
        gat_dim = std::max((size_t)16, summary_dim_ / 2);
        q_proj = MiniNetwork("gat_q", {summary_dim_, gat_dim},
                             Activation::NONE, Activation::NONE, rng);
        k_proj = MiniNetwork("gat_k", {summary_dim_, gat_dim},
                             Activation::NONE, Activation::NONE, rng);
        v_proj = MiniNetwork("gat_v", {summary_dim_, gat_dim},
                             Activation::NONE, Activation::NONE, rng);
        has_gat = true;
    }

    // --------------------------------------------------------
    // Compute this branch's Q, K, V from its summary
    // --------------------------------------------------------
    struct GATProjections { Tensor q, k, v; };
    GATProjections compute_gat(const Tensor& summary) {
        GATProjections p;
        if (!has_gat) return p;
        p.q = q_proj.forward(summary);
        p.k = k_proj.forward(summary);
        p.v = v_proj.forward(summary);
        for (auto& x : {&p.q, &p.k, &p.v})
            for (auto& val : x->data) if (!std::isfinite(val)) val = 0.0f;
        return p;
    }

    // --------------------------------------------------------
    // Enrich a summary with attention-weighted sibling messages.
    // sibling_kvs: list of (key, value) pairs from other branches.
    // Returns original summary + attention-weighted residual (clamped).
    // --------------------------------------------------------
    Tensor gat_enrich(const Tensor& summary,
                      const std::vector<std::pair<Tensor,Tensor>>& sibling_kvs) {
        if (!has_gat || sibling_kvs.empty()) return summary;

        Tensor my_q = q_proj.forward(summary);
        for (auto& v : my_q.data) if (!std::isfinite(v)) v = 0.0f;

        float scale = 1.0f / std::sqrt((float)gat_dim);
        std::vector<float> attn_logits(sibling_kvs.size());
        for (size_t i = 0; i < sibling_kvs.size(); i++) {
            const Tensor& k = sibling_kvs[i].first;
            float dot = 0.0f;
            size_t d = std::min(my_q.size(), k.size());
            for (size_t j = 0; j < d; j++) dot += my_q[j] * k[j];
            attn_logits[i] = dot * scale;
            if (!std::isfinite(attn_logits[i])) attn_logits[i] = 0.0f;
        }

        // Numerically stable softmax
        float mx = *std::max_element(attn_logits.begin(), attn_logits.end());
        float sum_exp = 0.0f;
        std::vector<float> attn(sibling_kvs.size());
        for (size_t i = 0; i < sibling_kvs.size(); i++) {
            attn[i] = std::exp(attn_logits[i] - mx);
            sum_exp += attn[i];
        }
        if (sum_exp > 1e-7f) for (auto& a : attn) a /= sum_exp;

        // Attention-weighted sum of values (in gat_dim space)
        Tensor agg({gat_dim});
        for (size_t i = 0; i < sibling_kvs.size(); i++) {
            const Tensor& v = sibling_kvs[i].second;
            size_t d = std::min(agg.size(), v.size());
            for (size_t j = 0; j < d; j++) agg[j] += attn[i] * v[j];
        }

        // Residual: project back to summary_dim via v_proj transpose-approx
        // Simple approach: pad/truncate agg to summary_dim, add as residual
        Tensor residual({summary_dim});
        size_t d = std::min(agg.size(), summary_dim);
        for (size_t i = 0; i < d; i++) residual[i] = agg[i];

        // enriched = summary + residual (clamped to [-1, 1])
        Tensor enriched({summary_dim});
        for (size_t i = 0; i < summary_dim; i++) {
            enriched[i] = std::tanh(summary[i] + 0.1f * residual[i]);
            if (!std::isfinite(enriched[i])) enriched[i] = summary[i];
        }
        return enriched;
    }

    BranchSignal evaluate(const Tensor& input, const Tensor& branch_output,
                          const std::vector<CrossTalkMessage>& sibling_messages) {
        BranchSignal signal;
        signal.branch_id = branch_id;
        signal.output = branch_output;

        // Compute relevance / heat
        Tensor heat_raw = relevance_net.forward(input);
        signal.heat = heat_raw[0];

        // Compute summary for cross-talk
        signal.summary = summary_net.forward(input);
        for (auto& v : signal.summary.data) if (!std::isfinite(v)) v = 0.0f;

        // Compute confidence from output entropy
        float entropy = 0;
        for (size_t i = 0; i < branch_output.size(); i++) {
            if (branch_output[i] > 1e-7f)
                entropy -= branch_output[i] * std::log(branch_output[i]);
        }
        float max_entropy = std::log((float)branch_output.size());
        signal.confidence = 1.0f - (entropy / max_entropy);  // high conf = low entropy

        // Check if we should redirect
        Tensor redirect_input = Tensor::concat(input, signal.summary);
        Tensor redirect_probs = redirect_net.forward(redirect_input);
        int suggested = redirect_probs.argmax();
        signal.wants_redirect = (suggested != branch_id && redirect_probs[suggested] > 0.7f);
        signal.redirect_suggestion = suggested;

        return signal;
    }

    void apply_adam(float lr) {
        relevance_net.apply_adam(lr);
        summary_net.apply_adam(lr);
        redirect_net.apply_adam(lr);
        if (has_gat) {
            q_proj.apply_adam(lr);
            k_proj.apply_adam(lr);
            v_proj.apply_adam(lr);
        }
    }

    size_t param_count() const {
        size_t n = relevance_net.param_count() + summary_net.param_count()
             + redirect_net.param_count();
        if (has_gat)
            n += q_proj.param_count() + k_proj.param_count() + v_proj.param_count();
        return n;
    }
};

} // namespace dendrite
