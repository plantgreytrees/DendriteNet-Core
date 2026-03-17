#include "../include/morality.hpp"
#include "test_runner.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>

using namespace dendrite;

// Write a temporary morality config and return its path
static std::string tmp_cfg(const std::string& content) {
    std::string path = "tests/tmp_morality.cfg";
    std::ofstream f(path);
    f << content;
    return path;
}

// ---- hard_block ---------------------------------------------------------

TEST(hard_block_fires_on_match) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[block_harm]\n"
        "type = hard_block\n"
        "description = Block harmful content\n"
        "patterns = harm ; destroy\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor dummy({4}, {0.1f, 0.2f, 0.3f, 0.4f});
    auto r = m.check_input(dummy, "I want to harm someone");
    ASSERT(!r.allowed);
    ASSERT_EQ(r.triggered_rule, std::string("block_harm"));
    std::remove(path.c_str());
}

TEST(hard_block_passes_clean_input) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[block_harm]\n"
        "type = hard_block\n"
        "description = Block harmful content\n"
        "patterns = harm ; destroy\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor dummy({4}, {0.25f, 0.25f, 0.25f, 0.25f});
    auto r = m.check_input(dummy, "Hello world");
    ASSERT(r.allowed);
    std::remove(path.c_str());
}

// ---- soft_redirect ------------------------------------------------------

TEST(soft_redirect_fires_and_sets_branch) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[redir]\n"
        "type = soft_redirect\n"
        "description = Redirect uncertain queries\n"
        "patterns = uncertain ; maybe\n"
        "redirect_branch = 2\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor dummy({4}, {0.25f, 0.25f, 0.25f, 0.25f});
    auto r = m.check_input(dummy, "maybe this is uncertain");
    ASSERT(r.redirect);
    ASSERT_EQ(r.redirect_branch, 2);
    ASSERT(r.allowed);  // soft_redirect does NOT block — it redirects
    std::remove(path.c_str());
}

// ---- confidence_gate ----------------------------------------------------

TEST(confidence_gate_blocks_low_confidence) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[conf_gate]\n"
        "type = confidence_gate\n"
        "description = Block low-confidence outputs\n"
        "min_confidence = 0.8\n"
        "active = true\n"
    );
    m.load_config(path);
    // Uniform output: max = 0.25, below 0.8
    Tensor output({4}, {0.25f, 0.25f, 0.25f, 0.25f});
    auto r = m.check_output(output);
    ASSERT(!r.allowed);
    std::remove(path.c_str());
}

TEST(confidence_gate_passes_high_confidence) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[conf_gate]\n"
        "type = confidence_gate\n"
        "description = Block low-confidence outputs\n"
        "min_confidence = 0.5\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor output({4}, {0.9f, 0.05f, 0.03f, 0.02f});
    auto r = m.check_output(output);
    ASSERT(r.allowed);
    std::remove(path.c_str());
}

// ---- disabled rule ------------------------------------------------------

TEST(disabled_rule_is_ignored) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[disabled]\n"
        "type = hard_block\n"
        "description = Should not fire\n"
        "patterns = harm\n"
        "active = false\n"
    );
    m.load_config(path);
    Tensor dummy({2}, {0.5f, 0.5f});
    auto r = m.check_input(dummy, "harm something");
    ASSERT(r.allowed);  // disabled rule must not fire
    std::remove(path.c_str());
}

// ---- audit trail --------------------------------------------------------

TEST(audit_log_grows_on_block) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[block_test]\n"
        "type = hard_block\n"
        "description = Audit test\n"
        "patterns = blocked_word\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor dummy({2}, {0.5f, 0.5f});
    size_t before = m.audit_log.size();
    m.check_input(dummy, "this contains blocked_word");
    ASSERT_GT(m.audit_log.size(), before);
    std::remove(path.c_str());
}

TEST(audit_log_never_decreases) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[block_test]\n"
        "type = hard_block\n"
        "description = Audit monotonicity\n"
        "patterns = trigger\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor dummy({2}, {0.5f, 0.5f});
    m.check_input(dummy, "trigger");
    size_t after_first = m.audit_log.size();
    m.check_input(dummy, "trigger again");
    ASSERT_GE(m.audit_log.size(), after_first);
    std::remove(path.c_str());
}

// ---- statistics ---------------------------------------------------------

TEST(stats_total_blocks_counted) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[b]\n"
        "type = hard_block\n"
        "description = Count test\n"
        "patterns = evil\n"
        "active = true\n"
    );
    m.load_config(path);
    Tensor dummy({2}, {0.5f, 0.5f});
    m.check_input(dummy, "evil input");
    m.check_input(dummy, "evil again");
    m.check_input(dummy, "clean input");
    ASSERT_EQ(m.total_blocks, 2u);
    ASSERT_EQ(m.total_checks, 3u);
    std::remove(path.c_str());
}

// ---- integrity ----------------------------------------------------------

TEST(integrity_check_passes_unmodified) {
    MoralityLayer m;
    auto path = tmp_cfg(
        "[r]\ntype = hard_block\ndescription = Test\npatterns = x\nactive = true\n"
    );
    m.load_config(path);
    ASSERT(m.verify_integrity());
    std::remove(path.c_str());
}

int main() {
    std::cout << "=== morality ===\n";
    RUN_TEST(hard_block_fires_on_match);
    RUN_TEST(hard_block_passes_clean_input);
    RUN_TEST(soft_redirect_fires_and_sets_branch);
    RUN_TEST(confidence_gate_blocks_low_confidence);
    RUN_TEST(confidence_gate_passes_high_confidence);
    RUN_TEST(disabled_rule_is_ignored);
    RUN_TEST(audit_log_grows_on_block);
    RUN_TEST(audit_log_never_decreases);
    RUN_TEST(stats_total_blocks_counted);
    RUN_TEST(integrity_check_passes_unmodified);
    return report("morality");
}
