#include "../include/data_cleaner.hpp"
#include "test_runner.hpp"
#include <iostream>

using namespace dendrite;

// Heat function: heat[i] = input[i]; output = softmax of input
static DataCleaner::HeatFn make_identity_heat(size_t n_branches) {
    return [n_branches](const Tensor& input) -> std::pair<Tensor, Tensor> {
        Tensor heat({n_branches});
        for (size_t i = 0; i < n_branches && i < input.size(); i++)
            heat[i] = std::max(0.0f, input[i]);
        Tensor output = input.softmax();
        return {heat, output};
    };
}

// ---- anomaly detection --------------------------------------------------

TEST(all_zero_input_is_anomaly) {
    DataCleaner dc;
    dc.heat_threshold   = 0.3f;
    dc.low_heat_threshold = 0.15f;
    size_t nb = 3;
    std::vector<Tensor> inputs  = {Tensor({3}, {0.0f, 0.0f, 0.0f})};
    std::vector<Tensor> targets = {Tensor({3}, {1.0f, 0.0f, 0.0f})};
    auto profiles = dc.profile_dataset(inputs, targets, make_identity_heat(nb), nb);
    ASSERT(profiles[0].is_anomaly);
}

TEST(strong_signal_not_anomaly) {
    DataCleaner dc;
    dc.heat_threshold   = 0.3f;
    dc.low_heat_threshold = 0.15f;
    size_t nb = 3;
    std::vector<Tensor> inputs  = {Tensor({3}, {0.9f, 0.05f, 0.05f})};
    std::vector<Tensor> targets = {Tensor({3}, {1.0f, 0.0f, 0.0f})};
    auto profiles = dc.profile_dataset(inputs, targets, make_identity_heat(nb), nb);
    ASSERT(!profiles[0].is_anomaly);
}

TEST(strong_signal_assigned_to_hottest_branch) {
    DataCleaner dc;
    dc.heat_threshold   = 0.3f;
    dc.low_heat_threshold = 0.15f;
    size_t nb = 3;
    // Branch 1 hottest
    std::vector<Tensor> inputs  = {Tensor({3}, {0.1f, 0.8f, 0.1f})};
    std::vector<Tensor> targets = {Tensor({3}, {0.0f, 1.0f, 0.0f})};
    auto profiles = dc.profile_dataset(inputs, targets, make_identity_heat(nb), nb);
    ASSERT(!profiles[0].is_anomaly);
    ASSERT_EQ(profiles[0].assigned_branch, 1);
}

// ---- curriculum sorting -------------------------------------------------

TEST(curriculum_sorted_ascending_difficulty) {
    DataCleaner dc;
    std::vector<DataProfile> profiles(3);
    profiles[0].is_anomaly = false; profiles[0].assigned_branch = 0;
    profiles[0].output_entropy = 0.8f;  // hard
    profiles[1].is_anomaly = false; profiles[1].assigned_branch = 0;
    profiles[1].output_entropy = 0.2f;  // easy
    profiles[2].is_anomaly = false; profiles[2].assigned_branch = 0;
    profiles[2].output_entropy = 0.5f;  // medium

    auto sched = dc.generate_curriculum(profiles, 1);
    ASSERT_EQ(sched.entries.size(), 3u);
    ASSERT_LT(sched.entries[0].difficulty, sched.entries[1].difficulty);
    ASSERT_LT(sched.entries[1].difficulty, sched.entries[2].difficulty);
}

TEST(curriculum_skips_anomalies) {
    DataCleaner dc;
    std::vector<DataProfile> profiles(3);
    profiles[0].is_anomaly = true;
    profiles[1].is_anomaly = false; profiles[1].assigned_branch = 0;
    profiles[1].output_entropy = 0.3f;
    profiles[2].is_anomaly = true;

    auto sched = dc.generate_curriculum(profiles, 1);
    ASSERT_EQ(sched.entries.size(), 1u);
}

// ---- full pipeline / report ---------------------------------------------

TEST(report_total_sample_count) {
    DataCleaner dc;
    dc.heat_threshold   = 0.3f;
    dc.low_heat_threshold = 0.15f;
    size_t nb = 2;
    std::vector<Tensor> inputs = {
        Tensor({2}, {0.9f, 0.1f}),
        Tensor({2}, {0.1f, 0.9f}),
        Tensor({2}, {0.0f, 0.0f}),
    };
    std::vector<Tensor> targets(3, Tensor({2}, {0.5f, 0.5f}));
    auto result = dc.clean(inputs, targets, make_identity_heat(nb), nb);
    ASSERT_EQ(result.report.total_samples, 3u);
}

TEST(report_anomaly_count) {
    DataCleaner dc;
    dc.heat_threshold   = 0.3f;
    dc.low_heat_threshold = 0.15f;
    size_t nb = 2;
    std::vector<Tensor> inputs = {
        Tensor({2}, {0.9f, 0.1f}),   // normal
        Tensor({2}, {0.0f, 0.0f}),   // anomaly
    };
    std::vector<Tensor> targets(2, Tensor({2}, {0.5f, 0.5f}));
    auto result = dc.clean(inputs, targets, make_identity_heat(nb), nb);
    ASSERT_EQ(result.report.anomaly_count, 1u);
    ASSERT_EQ(result.report.single_hot_count, 1u);
}

TEST(report_multi_hot_count) {
    DataCleaner dc;
    dc.heat_threshold   = 0.3f;
    dc.low_heat_threshold = 0.15f;
    size_t nb = 2;
    // Both branches fire above threshold
    std::vector<Tensor> inputs = {Tensor({2}, {0.8f, 0.8f})};
    std::vector<Tensor> targets(1, Tensor({2}, {0.5f, 0.5f}));
    auto result = dc.clean(inputs, targets, make_identity_heat(nb), nb);
    ASSERT_EQ(result.report.multi_hot_count, 1u);
}

TEST(profiles_count_matches_input_count) {
    DataCleaner dc;
    size_t nb = 2;
    size_t n = 5;
    std::vector<Tensor> inputs(n, Tensor({2}, {0.5f, 0.5f}));
    std::vector<Tensor> targets(n, Tensor({2}, {0.5f, 0.5f}));
    auto profiles = dc.profile_dataset(inputs, targets, make_identity_heat(nb), nb);
    ASSERT_EQ(profiles.size(), n);
}

// ---- anomaly calibration (Epic 2) ---------------------------------------

TEST(anomaly_rate_below_20pct) {
    DataCleaner dc;
    dc.current_training_steps = 300;  // above burn-in of 200
    size_t nb = 2;
    size_t n  = 50;
    // Clear heat signal: branch 0 = 0.5, branch 1 = 0.4 — both above low_heat_threshold
    auto hfn = [nb](const Tensor& /*input*/) -> std::pair<Tensor, Tensor> {
        Tensor heat({nb});
        heat[0] = 0.5f; heat[1] = 0.4f;
        Tensor out({3}, {0.7f, 0.2f, 0.1f});  // dominant class → low entropy
        return {heat, out.softmax()};
    };
    std::vector<Tensor> inputs(n,  Tensor({2}, {0.5f, 0.4f}));
    std::vector<Tensor> targets(n, Tensor({3}, {1.0f, 0.0f, 0.0f}));
    auto profiles = dc.profile_dataset(inputs, targets, hfn, nb);
    size_t anomalies = 0;
    for (auto& p : profiles) if (p.is_anomaly) anomalies++;
    ASSERT_EQ(anomalies, 0u);
}

TEST(no_anomaly_when_heat_clear) {
    DataCleaner dc;
    dc.current_training_steps = 0;   // burn-in not reached — entropy gate must be suppressed
    size_t nb = 2;
    auto hfn = [nb](const Tensor& /*input*/) -> std::pair<Tensor, Tensor> {
        Tensor heat({nb});
        heat[0] = 0.6f; heat[1] = 0.5f;
        Tensor out({3}, {0.8f, 0.1f, 0.1f});
        return {heat, out.softmax()};
    };
    std::vector<Tensor> inputs(1,  Tensor({2}, {0.6f, 0.5f}));
    std::vector<Tensor> targets(1, Tensor({3}, {1.0f, 0.0f, 0.0f}));
    auto profiles = dc.profile_dataset(inputs, targets, hfn, nb);
    ASSERT_EQ(profiles[0].is_anomaly, false);
}

int main() {
    std::cout << "=== data_cleaner ===\n";
    RUN_TEST(all_zero_input_is_anomaly);
    RUN_TEST(strong_signal_not_anomaly);
    RUN_TEST(strong_signal_assigned_to_hottest_branch);
    RUN_TEST(curriculum_sorted_ascending_difficulty);
    RUN_TEST(curriculum_skips_anomalies);
    RUN_TEST(report_total_sample_count);
    RUN_TEST(report_anomaly_count);
    RUN_TEST(report_multi_hot_count);
    RUN_TEST(profiles_count_matches_input_count);
    RUN_TEST(anomaly_rate_below_20pct);
    RUN_TEST(no_anomaly_when_heat_clear);
    return report("data_cleaner");
}
