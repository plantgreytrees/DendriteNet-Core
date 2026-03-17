#include "../include/tensor.hpp"
#include "test_runner.hpp"
#include <iostream>

using namespace dendrite;

// ---- clip ---------------------------------------------------------------

TEST(clip_basic) {
    Tensor t({3}, {-2.0f, 0.5f, 3.0f});
    t.clip(-1.0f, 1.0f);
    ASSERT_NEAR(t[0], -1.0f, 1e-6f);
    ASSERT_NEAR(t[1],  0.5f, 1e-6f);
    ASSERT_NEAR(t[2],  1.0f, 1e-6f);
}

TEST(clip_no_nan) {
    Tensor t({3}, {0.0f, 0.0f, 0.0f});
    t.clip(-1.0f, 1.0f);
    for (size_t i = 0; i < t.size(); i++) ASSERT_FINITE(t[i]);
}

TEST(clip_gradient_clipping_backprop) {
    // Backprop gradients must stay in [-1.0, 1.0] per convention
    Tensor g({4}, {-5.0f, 0.3f, 1.5f, -0.1f});
    g.clip(-1.0f, 1.0f);
    for (size_t i = 0; i < g.size(); i++) {
        ASSERT_GE(g[i], -1.0f);
        ASSERT_LE(g[i],  1.0f);
    }
}

TEST(clip_gradient_clipping_policy) {
    // Policy gradient must stay in [-0.5, 0.5]
    Tensor g({4}, {-5.0f, 0.3f, 1.5f, -0.1f});
    g.clip(-0.5f, 0.5f);
    for (size_t i = 0; i < g.size(); i++) {
        ASSERT_GE(g[i], -0.5f);
        ASSERT_LE(g[i],  0.5f);
    }
}

// ---- relu ---------------------------------------------------------------

TEST(relu_no_negative) {
    Tensor t({4}, {-1.0f, 0.0f, 0.5f, 2.0f});
    auto r = t.relu();
    for (size_t i = 0; i < r.size(); i++) ASSERT_GE(r[i], 0.0f);
}

TEST(relu_derivative_binary) {
    Tensor t({4}, {-1.0f, 0.0f, 0.5f, 2.0f});
    auto d = t.relu_derivative();
    ASSERT_NEAR(d[0], 0.0f, 1e-6f);
    ASSERT_NEAR(d[2], 1.0f, 1e-6f);
    ASSERT_NEAR(d[3], 1.0f, 1e-6f);
}

// ---- softmax ------------------------------------------------------------

TEST(softmax_sums_to_one) {
    Tensor t({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto s = t.softmax();
    ASSERT_NEAR(s.sum(), 1.0f, 1e-5f);
}

TEST(softmax_no_nan_extreme_values) {
    // Extreme values trigger overflow if not numerically stabilised
    Tensor t({4}, {100.0f, 200.0f, -100.0f, 0.0f});
    auto s = t.softmax();
    for (size_t i = 0; i < s.size(); i++) ASSERT_FINITE(s[i]);
}

TEST(softmax_all_nonnegative) {
    Tensor t({4}, {-3.0f, 1.0f, 2.0f, -1.0f});
    auto s = t.softmax();
    for (size_t i = 0; i < s.size(); i++) ASSERT_GE(s[i], 0.0f);
}

// ---- concat / slice -----------------------------------------------------

TEST(concat_size_and_values) {
    Tensor a({3}, {1.0f, 2.0f, 3.0f});
    Tensor b({2}, {4.0f, 5.0f});
    auto c = Tensor::concat(a, b);
    ASSERT_EQ(c.size(), 5u);
    ASSERT_NEAR(c[0], 1.0f, 1e-6f);
    ASSERT_NEAR(c[3], 4.0f, 1e-6f);
    ASSERT_NEAR(c[4], 5.0f, 1e-6f);
}

TEST(concat_many_size) {
    Tensor a({2}, {1.0f, 2.0f});
    Tensor b({3}, {3.0f, 4.0f, 5.0f});
    Tensor c({1}, {6.0f});
    auto out = Tensor::concat_many({a, b, c});
    ASSERT_EQ(out.size(), 6u);
    ASSERT_NEAR(out[5], 6.0f, 1e-6f);
}

TEST(slice_correct) {
    Tensor t({5}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    auto s = t.slice(1, 3);
    ASSERT_EQ(s.size(), 3u);
    ASSERT_NEAR(s[0], 1.0f, 1e-6f);
    ASSERT_NEAR(s[2], 3.0f, 1e-6f);
}

// ---- matvec -------------------------------------------------------------

TEST(matvec_shape_and_values) {
    // 2×3 identity-like matrix
    Tensor W({2, 3}, {1, 0, 0,  0, 1, 0});
    Tensor x({3},    {1.0f, 2.0f, 3.0f});
    Tensor b({2},    {0.0f, 0.0f});
    auto y = Tensor::matvec(W, x, b);
    ASSERT_EQ(y.size(), 2u);
    ASSERT_NEAR(y[0], 1.0f, 1e-6f);
    ASSERT_NEAR(y[1], 2.0f, 1e-6f);
}

TEST(matvec_bias_applied) {
    Tensor W({2, 2}, {1, 0,  0, 1});
    Tensor x({2},    {1.0f, 1.0f});
    Tensor b({2},    {10.0f, -10.0f});
    auto y = Tensor::matvec(W, x, b);
    ASSERT_NEAR(y[0], 11.0f, 1e-6f);
    ASSERT_NEAR(y[1], -9.0f, 1e-6f);
}

// ---- weighted_sum / mean ------------------------------------------------

TEST(weighted_sum_correct) {
    Tensor a({2}, {1.0f, 2.0f});
    Tensor b({2}, {3.0f, 4.0f});
    auto ws = Tensor::weighted_sum({a, b}, {0.5f, 0.5f});
    ASSERT_NEAR(ws[0], 2.0f, 1e-6f);
    ASSERT_NEAR(ws[1], 3.0f, 1e-6f);
}

TEST(mean_correct) {
    Tensor a({2}, {0.0f, 2.0f});
    Tensor b({2}, {2.0f, 4.0f});
    auto m = Tensor::mean({a, b});
    ASSERT_NEAR(m[0], 1.0f, 1e-6f);
    ASSERT_NEAR(m[1], 3.0f, 1e-6f);
}

// ---- misc ---------------------------------------------------------------

TEST(norm_non_negative) {
    Tensor t({3}, {-1.0f, 0.0f, 1.0f});
    ASSERT_GE(t.norm(), 0.0f);
    ASSERT_FINITE(t.norm());
}

TEST(argmax_correct) {
    Tensor t({4}, {0.1f, 0.5f, 0.3f, 0.2f});
    ASSERT_EQ(t.argmax(), 1);
}

TEST(zero_fills_zeros) {
    Tensor t({3}, {1.0f, 2.0f, 3.0f});
    t.zero();
    for (size_t i = 0; i < t.size(); i++) ASSERT_NEAR(t[i], 0.0f, 1e-9f);
}

TEST(transpose_correct) {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    auto mt = m.T();
    ASSERT_EQ(mt.shape[0], 3u);
    ASSERT_EQ(mt.shape[1], 2u);
    ASSERT_NEAR(mt.at(0, 1), 4.0f, 1e-6f); // was m.at(1,0)
    ASSERT_NEAR(mt.at(2, 0), 3.0f, 1e-6f); // was m.at(0,2)
}

// ---- gumbel_softmax -----------------------------------------------------

TEST(gumbel_sums_to_one) {
    std::mt19937 rng(42);
    Tensor t({4}, {1.0f, 2.0f, 0.5f, 3.0f});
    for (int trial = 0; trial < 10; trial++) {
        Tensor r = t.gumbel_softmax(1.0f, rng);
        float s = 0;
        for (size_t i = 0; i < r.size(); i++) s += r[i];
        ASSERT_NEAR(s, 1.0f, 1e-4f);
    }
}

TEST(gumbel_no_nan) {
    std::mt19937 rng(7);
    Tensor t({4}, {1.0f, 2.0f, 0.5f, 3.0f});
    Tensor r = t.gumbel_softmax(1.0f, rng);
    for (size_t i = 0; i < r.size(); i++) ASSERT_FINITE(r[i]);
}

TEST(gumbel_low_tau_argmax) {
    // At very low tau, dominant logit should win almost always
    int wins = 0;
    for (int trial = 0; trial < 100; trial++) {
        std::mt19937 rng(trial);
        Tensor t({4}, {0.0f, 0.0f, 10.0f, 0.0f});
        Tensor r = t.gumbel_softmax(0.01f, rng);
        if (r.argmax() == 2) wins++;
    }
    ASSERT_GE(wins, 95);
}

int main() {
    std::cout << "=== tensor ===\n";
    RUN_TEST(clip_basic);
    RUN_TEST(clip_no_nan);
    RUN_TEST(clip_gradient_clipping_backprop);
    RUN_TEST(clip_gradient_clipping_policy);
    RUN_TEST(relu_no_negative);
    RUN_TEST(relu_derivative_binary);
    RUN_TEST(softmax_sums_to_one);
    RUN_TEST(softmax_no_nan_extreme_values);
    RUN_TEST(softmax_all_nonnegative);
    RUN_TEST(concat_size_and_values);
    RUN_TEST(concat_many_size);
    RUN_TEST(slice_correct);
    RUN_TEST(matvec_shape_and_values);
    RUN_TEST(matvec_bias_applied);
    RUN_TEST(weighted_sum_correct);
    RUN_TEST(mean_correct);
    RUN_TEST(norm_non_negative);
    RUN_TEST(argmax_correct);
    RUN_TEST(zero_fills_zeros);
    RUN_TEST(transpose_correct);
    RUN_TEST(gumbel_sums_to_one);
    RUN_TEST(gumbel_no_nan);
    RUN_TEST(gumbel_low_tau_argmax);
    return report("tensor");
}
