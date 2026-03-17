#pragma once
#include "tensor.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace dendrite {

// ============================================================
// DataProfile: analysis of a single training sample
// ============================================================
struct DataProfile {
    size_t sample_index;
    Tensor heat_signature;              // heat from all branches
    std::vector<int> hot_branches;      // branches above threshold
    float output_entropy;               // how confused the model is
    float max_confidence;               // best branch confidence
    bool is_anomaly = false;            // true only if genuinely unclaimed
    int assigned_branch = -1;           // primary branch ownership
    std::vector<int> secondary_branches;// also relevant branches

    // Intersection/differentiation (for multi-hot samples)
    Tensor intersection_features;       // shared across hot branches
    std::vector<Tensor> difference_features; // unique per hot branch
};

// ============================================================
// CleaningReport: summary of the data cleaning pass
// ============================================================
struct CleaningReport {
    size_t total_samples;
    size_t anomaly_count;
    size_t multi_hot_count;             // samples claimed by multiple branches
    size_t single_hot_count;
    size_t no_hot_count;                // true anomalies
    std::vector<size_t> branch_assignments; // count per branch
    std::vector<size_t> anomaly_indices;

    void print() const {
        printf("  === Data Cleaning Report ===\n");
        printf("  Total samples: %zu\n", total_samples);
        printf("  Single-branch: %zu (%.1f%%)\n", single_hot_count,
               100.0f * single_hot_count / std::max(total_samples, (size_t)1));
        printf("  Multi-branch (correlated): %zu (%.1f%%)\n", multi_hot_count,
               100.0f * multi_hot_count / std::max(total_samples, (size_t)1));
        printf("  Anomalies (unclaimed): %zu (%.1f%%)\n", anomaly_count,
               100.0f * anomaly_count / std::max(total_samples, (size_t)1));
        printf("  Branch assignments: ");
        for (size_t i = 0; i < branch_assignments.size(); i++)
            printf("b%zu=%zu ", i, branch_assignments[i]);
        printf("\n");
    }
};

// ============================================================
// TrainingSchedule: ordered curriculum from cleaning
// ============================================================
struct TrainingSchedule {
    // Ordered by confidence (easy → hard)
    struct ScheduleEntry {
        size_t sample_index;
        int primary_branch;
        std::vector<int> secondary_branches;
        float difficulty;   // 0=easy, 1=hard
        bool is_multi_hot;
    };
    std::vector<ScheduleEntry> entries;

    // Phase 1: primary samples per branch (easy first)
    std::vector<std::vector<size_t>> branch_primary;
    // Phase 2: cross-training samples
    std::vector<std::vector<size_t>> branch_secondary;
};

// ============================================================
// DataCleaner: self-organising data pipeline
// ============================================================
// Three phases:
// 1. Heat profiling: forward-pass all samples, record heat signatures
// 2. Correlation & differentiation: find shared/unique features
// 3. Curriculum generation: sort by difficulty, assign to branches
// ============================================================
class DataCleaner {
public:
    float heat_threshold = 0.3f;
    float anomaly_entropy_threshold = 2.0f;  // normalised entropy; >1.0 disables gate
    float low_heat_threshold = 0.15f;        // all branches below this = unclaimed
    int min_training_steps     = 200;        // suppress entropy gate until network has trained this many steps
    int current_training_steps = 0;          // set by caller (DendriteNet3D) before each clean_data call

    // --------------------------------------------------------
    // Phase 1: Profile all samples using the network's heat scores
    // --------------------------------------------------------
    // The caller provides a function that runs inference and returns
    // heat scores + output for a given sample.
    // --------------------------------------------------------
    using HeatFn = std::function<std::pair<Tensor, Tensor>(const Tensor& input)>;

    std::vector<DataProfile> profile_dataset(
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& targets,
        HeatFn heat_fn,
        size_t num_branches) {

        std::vector<DataProfile> profiles(inputs.size());

        for (size_t i = 0; i < inputs.size(); i++) {
            auto [heat, output] = heat_fn(inputs[i]);
            auto& p = profiles[i];
            p.sample_index = i;
            p.heat_signature = heat;

            // Find hot branches
            for (size_t b = 0; b < num_branches; b++) {
                if (heat[b] >= heat_threshold)
                    p.hot_branches.push_back(b);
            }

            // Compute output entropy (normalised)
            float entropy = 0;
            float max_val = 0;
            for (size_t j = 0; j < output.size(); j++) {
                if (output[j] > 1e-7f)
                    entropy -= output[j] * std::log(output[j]);
                max_val = std::max(max_val, output[j]);
            }
            float max_entropy = std::log((float)output.size());
            p.output_entropy = (max_entropy > 0) ? entropy / max_entropy : 0;
            p.max_confidence = max_val;

            // Anomaly detection
            bool all_low = true;
            for (size_t b = 0; b < num_branches; b++) {
                if (heat[b] >= low_heat_threshold) { all_low = false; break; }
            }
            p.is_anomaly = all_low || ((current_training_steps >= min_training_steps) &&
                                        (p.output_entropy > anomaly_entropy_threshold));

            // Assign primary branch (hottest)
            if (!p.hot_branches.empty()) {
                p.assigned_branch = p.hot_branches[0];
                float best_heat = heat[p.hot_branches[0]];
                for (int b : p.hot_branches) {
                    if (heat[b] > best_heat) {
                        best_heat = heat[b];
                        p.assigned_branch = b;
                    }
                }
                // Secondary branches = all hot except primary
                for (int b : p.hot_branches) {
                    if (b != p.assigned_branch)
                        p.secondary_branches.push_back(b);
                }
            }
        }

        return profiles;
    }

    // --------------------------------------------------------
    // Phase 2: Correlation & differentiation for multi-hot samples
    // --------------------------------------------------------
    // For samples where multiple branches fire, compute what's SHARED
    // (intersection) and what's UNIQUE (difference) across the branches.
    // This stores both — shared strengthens general knowledge,
    // differences sharpen specialisation.
    // --------------------------------------------------------
    void compute_correlations(std::vector<DataProfile>& profiles,
                              const std::vector<Tensor>& inputs,
                              HeatFn heat_fn) {
        for (auto& p : profiles) {
            if (p.hot_branches.size() <= 1) continue;

            // For multi-hot samples: intersection = element-wise min of heat-weighted inputs
            // difference = deviation from intersection per branch
            size_t dim = inputs[p.sample_index].size();
            Tensor input = inputs[p.sample_index];

            // Intersection: features that ALL hot branches respond to
            // Approximation: element-wise minimum of (input * heat[b]) across hot branches
            Tensor intersection({dim});
            intersection.fill(1e9f);

            for (int b : p.hot_branches) {
                float h = p.heat_signature[b];
                for (size_t d = 0; d < dim; d++) {
                    float weighted = std::abs(input[d]) * h;
                    intersection[d] = std::min(intersection[d], weighted);
                }
            }
            // Normalise
            float int_max = intersection.max_val();
            if (int_max > 1e-7f)
                for (size_t d = 0; d < dim; d++) intersection[d] /= int_max;

            p.intersection_features = intersection;

            // Difference: per branch, what's unique = (input * heat[b]) - intersection
            p.difference_features.clear();
            for (int b : p.hot_branches) {
                Tensor diff({dim});
                float h = p.heat_signature[b];
                for (size_t d = 0; d < dim; d++) {
                    diff[d] = std::max(0.0f, std::abs(input[d]) * h - intersection[d]);
                }
                p.difference_features.push_back(diff);
            }
        }
    }

    // --------------------------------------------------------
    // Phase 3: Generate training curriculum
    // --------------------------------------------------------
    TrainingSchedule generate_curriculum(const std::vector<DataProfile>& profiles,
                                         size_t num_branches) {
        TrainingSchedule sched;
        sched.branch_primary.resize(num_branches);
        sched.branch_secondary.resize(num_branches);

        for (auto& p : profiles) {
            if (p.is_anomaly) continue;  // skip anomalies

            TrainingSchedule::ScheduleEntry entry;
            entry.sample_index = p.sample_index;
            entry.primary_branch = p.assigned_branch;
            entry.secondary_branches = p.secondary_branches;
            entry.difficulty = p.output_entropy;  // higher entropy = harder
            entry.is_multi_hot = p.hot_branches.size() > 1;
            sched.entries.push_back(entry);

            if (p.assigned_branch >= 0)
                sched.branch_primary[p.assigned_branch].push_back(p.sample_index);
            for (int sb : p.secondary_branches)
                sched.branch_secondary[sb].push_back(p.sample_index);
        }

        // Sort by difficulty (easy first)
        std::sort(sched.entries.begin(), sched.entries.end(),
            [](auto& a, auto& b) { return a.difficulty < b.difficulty; });

        return sched;
    }

    // --------------------------------------------------------
    // Full pipeline: profile → correlate → curriculum
    // --------------------------------------------------------
    struct CleaningResult {
        std::vector<DataProfile> profiles;
        TrainingSchedule schedule;
        CleaningReport report;
    };

    CleaningResult clean(const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& targets,
                         HeatFn heat_fn,
                         size_t num_branches) {
        CleaningResult result;

        // Phase 1
        result.profiles = profile_dataset(inputs, targets, heat_fn, num_branches);

        // Phase 2
        compute_correlations(result.profiles, inputs, heat_fn);

        // Phase 3
        result.schedule = generate_curriculum(result.profiles, num_branches);

        // Build report
        auto& r = result.report;
        r.total_samples = inputs.size();
        r.anomaly_count = 0;
        r.multi_hot_count = 0;
        r.single_hot_count = 0;
        r.no_hot_count = 0;
        r.branch_assignments.resize(num_branches, 0);

        for (auto& p : result.profiles) {
            if (p.is_anomaly) {
                r.anomaly_count++;
                r.anomaly_indices.push_back(p.sample_index);
            } else if (p.hot_branches.size() > 1) {
                r.multi_hot_count++;
            } else if (p.hot_branches.size() == 1) {
                r.single_hot_count++;
            } else {
                r.no_hot_count++;
            }
            if (p.assigned_branch >= 0 && p.assigned_branch < (int)num_branches)
                r.branch_assignments[p.assigned_branch]++;
        }

        return result;
    }
};

} // namespace dendrite
