#pragma once
#include "conductor.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dendrite {

// ============================================================
// Branch: a specialist sub-tree with its own sub-conductor
// ============================================================
struct Branch {
    int id;
    std::string domain;
    MiniNetwork specialist;       // the actual prediction network
    SubConductor sub_conductor;   // local conductor for feedback

    // Children branches (for nested depth)
    std::vector<std::shared_ptr<Branch>> children;
    MiniNetwork child_router;     // routes to children if present
    bool has_children = false;

    // Statistics
    size_t visit_count = 0;
    float running_loss = 0;
    float running_confidence = 0;

    // Child router forward-pass cache (for backprop)
    Tensor last_child_probs;
    int    last_chosen_child = -1;

    // Enhancement #28: Oscillatory phase (Kuramoto model)
    float phase = 0.0f;         // current phase θ ∈ [0, 2π)
    float natural_freq = 0.0f;  // intrinsic angular frequency ω

    // Enhancement #27: Hierarchical message passing — per-step context buffers
    Tensor child_context;      // bottom-up: aggregate of children outputs
    Tensor conductor_guidance; // top-down: guidance signal from conductor

    // Enhancement #29: Multi-compartment — distal (context) pathway
    MiniNetwork distal;          // context + cross-talk → modulation vector
    bool has_distal = false;
    float distal_coupling = 0.3f;  // how much distal can modulate proximal (±30%)
    // Cached intermediates for backward pass
    Tensor last_prox_out;        // proximal (specialist) output before modulation
    Tensor last_dist_out;        // distal output (tanh-modulation signal)

    size_t param_count() const {
        size_t t = specialist.param_count() + sub_conductor.param_count();
        if (has_distal) t += distal.param_count();
        if (has_children) {
            t += child_router.param_count();
            for (auto& c : children) t += c->param_count();
        }
        return t;
    }
};

using BranchPtr = std::shared_ptr<Branch>;

// ============================================================
// GrowthController: periodic split/prune decisions for self-adapting topology
// ============================================================
struct GrowthController {
    size_t eval_interval    = 10;
    float  split_threshold  = 1.5f;
    float  prune_threshold  = 0.02f;
    size_t min_visits_to_split = 100;
    size_t prune_patience   = 3;
    size_t max_branches     = 12;
    size_t min_branches     = 2;
    size_t epochs_evaluated = 0;

    struct BranchHealth {
        float  avg_loss          = 0.0f;
        float  visit_ratio       = 0.0f;
        size_t low_usage_cycles  = 0;
    };
    std::vector<BranchHealth> health;

    void init(size_t num_branches) { health.assign(num_branches, BranchHealth{}); }

    struct Decision {
        std::vector<size_t> to_split;
        std::vector<size_t> to_prune;
    };

    Decision evaluate(const std::vector<BranchPtr>& branches, size_t total_samples) {
        Decision d;
        epochs_evaluated++;
        if (eval_interval == 0 || epochs_evaluated % eval_interval != 0) return d;
        while (health.size() < branches.size()) health.push_back(BranchHealth{});
        for (size_t b = 0; b < branches.size(); b++) {
            float visit_ratio = (float)branches[b]->visit_count / std::max(total_samples, (size_t)1);
            health[b].avg_loss = branches[b]->running_loss;
            health[b].visit_ratio = visit_ratio;
            if (!branches[b]->has_children &&
                branches[b]->running_loss > split_threshold &&
                branches[b]->visit_count > min_visits_to_split &&
                branches.size() < max_branches)
                d.to_split.push_back(b);
            if (visit_ratio < prune_threshold) {
                health[b].low_usage_cycles++;
                if (health[b].low_usage_cycles >= prune_patience && branches.size() > min_branches)
                    d.to_prune.push_back(b);
            } else {
                health[b].low_usage_cycles = 0;
            }
        }
        return d;
    }
};

// ============================================================
// ModelBuilder: constructs the branch tree from named parameters.
// Called by DendriteNet3D::build() — no dependency on ModelConfig.
// ============================================================
struct ModelBuilder {
    /// Build top-level branches (and their optional children) from names + dimensions.
    /// @param branch_names      Names of top-level branches.
    /// @param sub_branch_names  Per-branch child names; empty inner vector = no children.
    /// @param input_dim         Network input dimension.
    /// @param output_dim        Network output dimension.
    /// @param specialist_hidden Hidden units for specialist MiniNetworks.
    /// @param summary_dim       SubConductor summary vector dimension.
    /// @param rng               RNG shared with the network (advanced in-place).
    [[nodiscard]] static std::vector<BranchPtr> build_branches(
        const std::vector<std::string>& branch_names,
        const std::vector<std::vector<std::string>>& sub_branch_names,
        size_t input_dim, size_t output_dim, int specialist_hidden,
        size_t summary_dim, std::mt19937& rng)
    {
        const size_t n = branch_names.size();
        std::vector<BranchPtr> branches;
        branches.reserve(n);

        for (size_t i = 0; i < n; i++) {
            auto branch = std::make_shared<Branch>();
            branch->id     = static_cast<int>(i);
            branch->domain = branch_names[i];

            // Specialist network
            branch->specialist = MiniNetwork(
                "specialist_" + branch_names[i],
                {input_dim, (size_t)specialist_hidden, (size_t)specialist_hidden / 2, output_dim},
                Activation::RELU, Activation::SOFTMAX, rng);

            // Sub-conductor
            branch->sub_conductor = SubConductor(
                static_cast<int>(i), input_dim, summary_dim, n, rng);

            // Enhancement #28: oscillator — distinct natural freq per branch, random init phase
            branch->natural_freq = 0.5f + 0.15f * static_cast<float>(i);
            {
                std::uniform_real_distribution<float> pd(0.0f, 2.0f * 3.14159265f);
                branch->phase = pd(rng);
            }

            // Optional sub-branches
            if (i < sub_branch_names.size() && !sub_branch_names[i].empty()) {
                const size_t num_children = sub_branch_names[i].size();
                branch->has_children = true;

                branch->child_router = MiniNetwork(
                    "child_router_" + branch_names[i],
                    {input_dim, 32, num_children},
                    Activation::RELU, Activation::SOFTMAX, rng);

                for (size_t c = 0; c < num_children; c++) {
                    auto child = std::make_shared<Branch>();
                    child->id     = static_cast<int>(n + i * 10 + c);
                    child->domain = sub_branch_names[i][c];
                    child->specialist = MiniNetwork(
                        "specialist_" + sub_branch_names[i][c],
                        {input_dim, (size_t)specialist_hidden / 2, output_dim},
                        Activation::RELU, Activation::SOFTMAX, rng);
                    child->sub_conductor = SubConductor(
                        child->id, input_dim, summary_dim, num_children, rng);
                    branch->children.push_back(child);
                }
            }

            // Init early-exit classifier (tight 0.15 entropy ceiling).
            // Child specialists are 2-layer and silently skip this.
            branch->specialist.init_early_exit(output_dim, rng, 0.15f);

            branches.push_back(std::move(branch));
        }
        return branches;
    }
};

} // namespace dendrite
