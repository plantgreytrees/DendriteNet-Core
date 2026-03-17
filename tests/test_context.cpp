#include "../include/task_context.hpp"
#include "test_runner.hpp"
#include <iostream>
#include <cmath>

using namespace dendrite;

static std::mt19937 rng(42);

static TaskContext make_ctx(size_t item_dim = 4, size_t proj_dim = 8) {
    std::mt19937 local_rng(42);
    return TaskContext(item_dim, proj_dim, local_rng);
}

// ---- store / size -------------------------------------------------------

TEST(store_increases_size) {
    auto ctx = make_ctx();
    ASSERT_EQ(ctx.size(), 0u);
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 1.0f, "item0");
    ASSERT_EQ(ctx.size(), 1u);
}

TEST(store_multiple_items) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.5f, 0.2f, 0.1f});
    ctx.store(t, 0);
    ctx.store(t, 1);
    ctx.store(t, 2);
    ASSERT_EQ(ctx.size(), 3u);
}

// ---- Ebbinghaus decay ---------------------------------------------------

TEST(step_decays_relevance) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 1.0f);
    float before = ctx.episodic_items[0].relevance;
    ctx.step();
    float after = ctx.episodic_items[0].relevance;
    ASSERT_LT(after, before);
    // Should decay by decay_rate (0.85 default)
    ASSERT_NEAR(after, before * ctx.decay_rate, 1e-5f);
}

TEST(step_evicts_below_threshold) {
    auto ctx = make_ctx();
    ctx.decay_rate = 0.1f;          // aggressive decay
    ctx.eviction_threshold = 0.5f;
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 1.0f);
    // After enough steps relevance drops below 0.5
    for (int i = 0; i < 10; i++) ctx.step();
    ASSERT_EQ(ctx.size(), 0u);
}

TEST(item_survives_if_above_threshold) {
    auto ctx = make_ctx();
    ctx.decay_rate = 0.99f;         // very slow decay
    ctx.eviction_threshold = 0.1f;
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 1.0f);
    ctx.step();
    ASSERT_GT(ctx.size(), 0u);
}

// ---- reinforce ----------------------------------------------------------

TEST(reinforce_bumps_relevance) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 0.5f);
    ctx.step();
    float after_decay = ctx.episodic_items[0].relevance;
    ctx.reinforce(0);
    float after_reinforce = ctx.episodic_items[0].relevance;
    ASSERT_GT(after_reinforce, after_decay);
}

TEST(reinforce_clamps_at_one) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 1.0f);
    // Reinforce many times — should not exceed 1.0
    for (int i = 0; i < 20; i++) ctx.reinforce(0);
    ASSERT_LE(ctx.episodic_items[0].relevance, 1.0f + 1e-5f);
}

TEST(reinforce_only_affects_matching_branch) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 0.5f);
    ctx.store(t, 1, 0.5f);
    ctx.step();
    float b1_before = ctx.episodic_items[1].relevance;
    ctx.reinforce(0);  // only branch 0
    float b1_after = ctx.episodic_items[1].relevance;
    ASSERT_NEAR(b1_before, b1_after, 1e-5f);
}

// ---- max_items ----------------------------------------------------------

TEST(max_items_enforced) {
    auto ctx = make_ctx();
    ctx.max_items = 3;
    Tensor t({4}, {0.5f, 0.5f, 0.0f, 0.0f});
    for (int i = 0; i < 10; i++) ctx.store(t, i % 3, 1.0f);
    ASSERT_LE(ctx.size(), 3u);
}

// ---- get_context --------------------------------------------------------

TEST(get_context_correct_dim) {
    auto ctx = make_ctx(4, 8);
    Tensor t({4}, {1.0f, 0.5f, 0.2f, 0.1f});
    ctx.store(t, 0);
    auto c = ctx.get_context(t);
    ASSERT_EQ(c.size(), 8u);
}

TEST(get_context_no_nan) {
    auto ctx = make_ctx(4, 8);
    Tensor t({4}, {1.0f, 0.5f, 0.2f, 0.1f});
    ctx.store(t, 0);
    auto c = ctx.get_context(t);
    for (size_t i = 0; i < c.size(); i++) ASSERT_FINITE(c[i]);
}

TEST(get_context_empty_returns_zero_dim) {
    auto ctx = make_ctx(4, 8);
    Tensor query({4});  // zero query — memory is empty so value unused
    auto c = ctx.get_context(query);  // empty memory
    ASSERT_EQ(c.size(), 8u);          // still correct dim (zero vector)
}

// ---- reset --------------------------------------------------------------

TEST(reset_clears_all) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0);
    ctx.store(t, 1);
    ctx.step();
    ctx.reset();
    ASSERT_EQ(ctx.size(), 0u);
    ASSERT_EQ(ctx.current_step, 0);
    ASSERT(ctx.empty());
}

// ---- query_top_k --------------------------------------------------------

TEST(query_top_k_returns_at_most_k) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    for (int i = 0; i < 5; i++) ctx.store(t, i);
    auto top = ctx.query_top_k(t, 3);
    ASSERT_LE(top.size(), 3u);
}

// ---- reinforce calibration (Epic 3) -------------------------------------

TEST(reinforce_does_not_saturate) {
    auto ctx = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx.store(t, 0, 0.5f);          // stored at importance=0.5 → rel=0.5
    ctx.step();                      // rel → 0.5 * 0.85 = 0.425
    ctx.reinforce(0);                // rel → 0.425 + 0.1 = 0.525
    float rel = ctx.episodic_items.front().relevance;
    ASSERT_GE(rel, 0.4f);
    ASSERT_LE(rel, 0.65f);          // must NOT jump to 1.0
}

TEST(reinforce_weighted_scales_boost) {
    auto ctx1 = make_ctx();
    auto ctx2 = make_ctx();
    Tensor t({4}, {1.0f, 0.0f, 0.0f, 0.0f});
    ctx1.store(t, 0, 0.5f); ctx1.step();
    ctx2.store(t, 0, 0.5f); ctx2.step();
    float before = ctx1.episodic_items.front().relevance;
    ctx1.reinforce(0);                   // full boost
    ctx2.reinforce_weighted(0, 0.5f);    // half boost
    float full_delta = ctx1.episodic_items.front().relevance - before;
    float half_delta = ctx2.episodic_items.front().relevance - before;
    ASSERT_GT(full_delta, half_delta);   // weighted gives less boost
    ASSERT_GT(half_delta, 0.0f);         // but still some boost
}

int main() {
    std::cout << "=== task_context ===\n";
    RUN_TEST(store_increases_size);
    RUN_TEST(store_multiple_items);
    RUN_TEST(step_decays_relevance);
    RUN_TEST(step_evicts_below_threshold);
    RUN_TEST(item_survives_if_above_threshold);
    RUN_TEST(reinforce_bumps_relevance);
    RUN_TEST(reinforce_clamps_at_one);
    RUN_TEST(reinforce_only_affects_matching_branch);
    RUN_TEST(max_items_enforced);
    RUN_TEST(get_context_correct_dim);
    RUN_TEST(get_context_no_nan);
    RUN_TEST(get_context_empty_returns_zero_dim);
    RUN_TEST(reset_clears_all);
    RUN_TEST(query_top_k_returns_at_most_k);
    RUN_TEST(reinforce_does_not_saturate);
    RUN_TEST(reinforce_weighted_scales_boost);
    return report("task_context");
}
