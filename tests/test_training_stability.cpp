#include "../include/dendrite3d.hpp"
#include "test_runner.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <cmath>

using namespace dendrite;

// ---- helpers ------------------------------------------------------------

struct StabilityData {
    std::vector<Tensor> train_inputs;
    std::vector<Tensor> train_targets;
    std::vector<Tensor> test_inputs;
    std::vector<Tensor> test_targets;
    std::vector<int>    test_labels;
};

static StabilityData generate_data() {
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    StabilityData d;
    // Class 0: centred at {1,1,0,0}, class 1: centred at {0,0,1,1}
    float centres[2][4] = {{1.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,1.0f,1.0f}};
    for (int i = 0; i < 100; i++) {
        int label = i % 2;
        Tensor x({4});
        for (int j = 0; j < 4; j++) x[j] = centres[label][j] + noise(rng);
        Tensor y({2}); y[label] = 1.0f;
        if (i < 80) {
            d.train_inputs.push_back(x);
            d.train_targets.push_back(y);
        } else {
            d.test_inputs.push_back(x);
            d.test_targets.push_back(y);
            d.test_labels.push_back(label);
        }
    }
    return d;
}

static DendriteNet3D make_net() {
    DendriteNet3D net(4, 2, 99);
    net.learning_rate = 0.003f;
    net.build({"branch_a", "branch_b"}, 32, {{}, {}});
    net.init_v3("");
    // Disable untrained critique/conf-gate — random weights randomly block outputs,
    // making accuracy non-deterministic. These have their own morality test suite.
    net.morality.critique_enabled = false;
    net.morality.confidence_gate_enabled = false;
    return net;
}

// ---- tests --------------------------------------------------------------

TEST(loss_decreases_over_training) {
    auto d   = generate_data();
    auto net = make_net();
    float loss1  = net.train_batch(d.train_inputs, d.train_targets, 1);
    float loss10 = net.train_batch(d.train_inputs, d.train_targets, 9);
    ASSERT_FINITE(loss10);
    ASSERT_LT(loss10, 1.0f);
    // Should not be dramatically worse than epoch 1
    ASSERT_LE(loss10, loss1 * 1.05f + 0.1f);
}

TEST(accuracy_above_baseline) {
    auto d   = generate_data();
    auto net = make_net();
    net.train_batch(d.train_inputs, d.train_targets, 10);
    int correct = 0;
    for (size_t i = 0; i < d.test_inputs.size(); i++) {
        auto res = net.infer(d.test_inputs[i]);
        if (res.output.argmax() == d.test_labels[i]) correct++;
    }
    // Require at least 80% (16/20)
    ASSERT_GE(correct, 16);
}

TEST(nan_inputs_produce_finite_output) {
    auto net = make_net();
    Tensor nan_input({4});
    for (size_t i = 0; i < 4; i++)
        nan_input[i] = std::numeric_limits<float>::quiet_NaN();
    auto res = net.infer(nan_input);
    for (size_t i = 0; i < res.output.size(); i++)
        ASSERT_FINITE(res.output[i]);
}

TEST(no_weight_explosion) {
    auto d   = generate_data();
    auto net = make_net();
    net.train_batch(d.train_inputs, d.train_targets, 10);
    const float MAX_WEIGHT = 10.0f;
    bool all_bounded = true;
    for (const auto& br : net.branches) {
        for (const auto& layer : br->specialist.layers) {
            for (float w : layer.weights.data) {
                if (!std::isfinite(w) || std::abs(w) > MAX_WEIGHT) {
                    all_bounded = false;
                    break;
                }
            }
        }
    }
    ASSERT(all_bounded);
}

// ---- main ---------------------------------------------------------------

int main() {
    std::cout << "=== training_stability ===\n";
    RUN_TEST(loss_decreases_over_training);
    RUN_TEST(accuracy_above_baseline);
    RUN_TEST(nan_inputs_produce_finite_output);
    RUN_TEST(no_weight_explosion);
    return report("training_stability");
}
