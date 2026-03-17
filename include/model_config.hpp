#pragma once
#include <string>
#include <vector>

namespace dendrite {

/// Configuration describing the model topology (branch structure, routing params,
/// which optional enhancements are active). Passed to DendriteNet3D::build_from_config().
struct ModelConfig {
    /// Names of top-level branches (e.g. {"analytical", "creative", "factual"}).
    std::vector<std::string> branch_names;

    /// Per-branch sub-branch names; empty inner vector = no sub-branches for that branch.
    /// Must be the same length as branch_names, or empty (no sub-branches anywhere).
    std::vector<std::vector<std::string>> sub_branches;

    /// Hidden units for the specialist MiniNetwork at each branch.
    int hidden_size = 64;

    /// Minimum heat score for a branch to be considered active during inference.
    float heat_threshold = 0.2f;

    /// How many top branches contribute to the fused output.
    int top_k = 2;

    /// Optional Tier-4 enhancements.
    struct Enhancements {
        /// Enhancement #26 — meta-learning branch init via domain embeddings.
        bool hypernetwork = false;
        int  hypernetwork_embed_dim = 16;

        /// Enhancement #29 — proximal/distal branch compartments.
        bool  multi_compartment = false;
        float multi_compartment_coupling = 0.3f;

        /// Enhancement #27 — bottom-up + top-down hierarchical message passing.
        bool hierarchical = false;

        /// Enhancement #24 — Perceiver IO latent cross-attention multimodal fusion.
        bool   perceiver = false;
        size_t perceiver_token_dim = 0;  // 0 = infer from active modality module

        /// Enhancement #28 — Kuramoto oscillatory synchronisation across branches.
        bool  oscillatory = false;
        float oscillatory_coupling = 0.4f;
    } enhancements;
};

/// Configuration controlling the training loop (epochs, LR, batch size, data split).
/// Kept separate from ModelConfig so topology and training schedule can be changed
/// independently.
struct TrainingConfig {
    float  learning_rate = 0.003f;
    size_t epochs        = 55;
    size_t batch_size    = 1;    // per-sample when 1; minibatch when >1
    float  train_split   = 0.8f; // fraction of data used for training
};

} // namespace dendrite
