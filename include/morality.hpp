#pragma once
#include "tensor.hpp"
#include "layer.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <functional>
#include <regex>

namespace dendrite {

// ============================================================
// MoralityRule: a single guardrail rule
// ============================================================
struct MoralityRule {
    std::string id;
    std::string type;       // "hard_block", "soft_redirect", "confidence_gate"
    std::string description;
    std::vector<std::string> patterns;    // regex patterns to match
    std::vector<std::regex> compiled;     // compiled regex (built from patterns)
    int redirect_branch = -1;             // for soft_redirect
    float min_confidence = 0.0f;          // for confidence_gate
    bool active = true;
};

// ============================================================
// AuditEntry: log of a morality layer activation
// ============================================================
struct AuditEntry {
    std::string timestamp;
    std::string rule_id;
    std::string action;     // "blocked", "redirected", "low_confidence", "critique_fail"
    std::string details;
    size_t input_hash;
};

// ============================================================
// MoralityCheckResult
// ============================================================
struct MoralityCheckResult {
    bool allowed = true;
    bool redirect = false;
    int redirect_branch = -1;
    std::string triggered_rule;
    std::string reason;
    float confidence_score = 1.0f;   // from ConfidenceGateNetwork (0..1)
    float critique_score = 1.0f;     // from CritiqueNetwork (0..1, higher = safer)
};

// ============================================================
// Enhancement 12: Constitutional Self-Critique
// ============================================================
// A set of constitutional principles with frozen embeddings.
// A learned CritiqueNetwork scores outputs against each principle.
// Low scores trigger a block regardless of regex patterns.
// ============================================================
struct ConstitutionalPrinciple {
    std::string name;
    std::string description;
    Tensor embedding;    // frozen prototype vector (set at init, not trained)
    float threshold;     // critique score below this → flag
};

// Learned scorer: takes output tensor + principle embedding → safety score [0,1]
struct CritiqueNetwork {
    MiniNetwork scorer;   // [output_dim + embed_dim] → [hidden] → [1]
    size_t output_dim = 0;
    size_t embed_dim = 0;
    bool initialised = false;

    CritiqueNetwork() = default;

    void init(size_t out_dim, size_t emb_dim, std::mt19937& rng) {
        output_dim = out_dim;
        embed_dim = emb_dim;
        size_t in_dim = out_dim + emb_dim;
        scorer = MiniNetwork("critique",
            {in_dim, in_dim * 2, 16, 1},
            Activation::RELU, Activation::SIGMOID, rng);
        initialised = true;
    }

    // Score output against one principle embedding; returns [0,1] (1 = safe)
    float score(const Tensor& output, const Tensor& principle_emb) {
        if (!initialised || output.size() == 0 || principle_emb.size() == 0) return 1.0f;
        Tensor combined = Tensor::concat(output, principle_emb);
        Tensor result = scorer.forward(combined);
        float s = result.size() == 0 ? 1.0f : result[0];
        if (!std::isfinite(s)) s = 1.0f;
        return std::clamp(s, 0.0f, 1.0f);
    }

    // Score against all principles; return the minimum (weakest link)
    float score_all(const Tensor& output,
                    const std::vector<ConstitutionalPrinciple>& principles) {
        if (!initialised || principles.empty()) return 1.0f;
        float min_score = 1.0f;
        for (auto& p : principles) {
            if (p.embedding.size() == 0) continue;
            float s = score(output, p.embedding);
            min_score = std::min(min_score, s);
        }
        return min_score;
    }

    void apply_adam(float lr) { if (initialised) scorer.apply_adam(lr); }
    size_t param_count() const { return initialised ? scorer.param_count() : 0; }
};

// ============================================================
// Enhancement 15: Multi-Dimensional Confidence Gate
// ============================================================
// Replaces the single max_conf threshold with a learned 5-feature gate.
// Features: entropy, max_confidence, branch_agreement, heat_concentration,
//           context_similarity.
// A MiniNetwork maps these 5 features → a single gate score [0,1].
// Hard floor of 0.1 remains as an absolute failsafe.
// ============================================================
struct ConfidenceGateNetwork {
    MiniNetwork gate_net;   // 5 → 16 → 1 (SIGMOID output)
    bool initialised = false;
    float hard_floor = 0.1f;  // absolute minimum confidence

    ConfidenceGateNetwork() = default;

    void init(std::mt19937& rng) {
        gate_net = MiniNetwork("conf_gate",
            {5, 16, 8, 1},
            Activation::RELU, Activation::SIGMOID, rng);
        initialised = true;
    }

    // Extract 5 features and compute gate score
    // Returns gate_score in [0,1] — scores below threshold indicate low confidence
    float compute(const Tensor& output,
                  const Tensor& heat_scores,
                  float context_sim = 0.5f) {
        if (!initialised || output.size() == 0) return 1.0f;

        // Feature 1: output entropy (normalised by log(n_classes))
        float entropy = 0.0f;
        size_t n = output.size();
        for (size_t i = 0; i < n; i++) {
            float p = std::clamp(output[i], 1e-7f, 1.0f);
            entropy -= p * std::log(p);
        }
        float max_entropy = (n > 1) ? std::log((float)n) : 1.0f;
        float norm_entropy = (max_entropy > 1e-7f) ? std::clamp(entropy / max_entropy, 0.0f, 1.0f) : 0.0f;

        // Feature 2: max confidence
        float max_conf = output.max_val();
        if (!std::isfinite(max_conf)) max_conf = 0.0f;

        // Feature 3: branch agreement (std dev of heat scores, lower = more agreement)
        float heat_mean = 0.0f, heat_var = 0.0f;
        size_t nh = heat_scores.size();
        if (nh > 0) {
            for (size_t i = 0; i < nh; i++) heat_mean += heat_scores[i];
            heat_mean /= (float)nh;
            for (size_t i = 0; i < nh; i++) {
                float d = heat_scores[i] - heat_mean;
                heat_var += d * d;
            }
            heat_var /= (float)nh;
        }
        float branch_agreement = 1.0f - std::clamp(std::sqrt(heat_var), 0.0f, 1.0f);

        // Feature 4: heat concentration (max heat / sum heat)
        float heat_sum = 0.0f, heat_max = 0.0f;
        for (size_t i = 0; i < nh; i++) {
            heat_sum += heat_scores[i];
            heat_max = std::max(heat_max, heat_scores[i]);
        }
        float heat_concentration = (heat_sum > 1e-7f) ? heat_max / heat_sum : 0.0f;

        // Feature 5: context similarity (passed in)
        float ctx_sim = std::clamp(context_sim, 0.0f, 1.0f);

        // Pack 5 features into a tensor
        Tensor features({5});
        features[0] = norm_entropy;
        features[1] = max_conf;
        features[2] = branch_agreement;
        features[3] = heat_concentration;
        features[4] = ctx_sim;
        for (auto& v : features.data) if (!std::isfinite(v)) v = 0.5f;

        Tensor gate_out = gate_net.forward(features);
        float gate_score = gate_out.size() == 0 ? 1.0f : gate_out[0];
        if (!std::isfinite(gate_score)) gate_score = 1.0f;
        return std::clamp(gate_score, hard_floor, 1.0f);
    }

    void apply_adam(float lr) { if (initialised) gate_net.apply_adam(lr); }
    size_t param_count() const { return initialised ? gate_net.param_count() : 0; }
};

// ============================================================
// MoralityLayer: immutable guardrails wrapping the conductor
// ============================================================
// - Loaded from config file, NOT modifiable by training
// - Checks input BEFORE routing and output AFTER fusion
// - Layer 1: regex/pattern matching (hard_block, soft_redirect)
// - Layer 2: constitutional self-critique (CritiqueNetwork)
// - Layer 3: multi-dim confidence gate (ConfidenceGateNetwork)
// - Logs all activations to an audit trail
// - Config file is checksummed to detect tampering
// ============================================================
class MoralityLayer {
public:
    std::vector<MoralityRule> rules;
    std::vector<AuditEntry> audit_log;
    std::vector<ConstitutionalPrinciple> principles;
    CritiqueNetwork critique_net;
    ConfidenceGateNetwork confidence_gate;
    bool enabled = true;
    bool critique_enabled = false;      // enabled once critique_net is init'd
    bool confidence_gate_enabled = false;
    std::string config_path;
    size_t config_checksum = 0;

    // Critique threshold (score below this triggers a block)
    float critique_threshold = 0.3f;

    // Confidence gate threshold (gate_score below this → low_confidence)
    float confidence_gate_threshold = 0.4f;

    // Statistics
    size_t total_checks = 0;
    size_t total_blocks = 0;
    size_t total_redirects = 0;
    size_t total_confidence_gates = 0;
    size_t total_critique_blocks = 0;

    MoralityLayer() = default;

    // --------------------------------------------------------
    // Initialise learned sub-networks (call after knowing output_dim)
    // --------------------------------------------------------
    void init_learned_components(size_t output_dim, std::mt19937& rng) {
        // Constitutional self-critique: embed_dim = output_dim (same space)
        critique_net.init(output_dim, output_dim, rng);
        critique_enabled = true;

        // Initialise default principles from the output space
        init_default_principles(output_dim, rng);

        // Multi-dim confidence gate
        confidence_gate.init(rng);
        confidence_gate_enabled = true;
    }

    // --------------------------------------------------------
    // Load rules from a simple config format
    // --------------------------------------------------------
    bool load_config(const std::string& path) {
        config_path = path;
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cout << "[Morality] WARNING: config not found at " << path
                      << ", running without guardrails\n";
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        config_checksum = std::hash<std::string>{}(content);

        // Parse
        rules.clear();
        std::istringstream stream(content);
        std::string line;
        MoralityRule current;
        bool in_rule = false;

        while (std::getline(stream, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            if (line[0] == '[' && line.back() == ']') {
                if (in_rule) finalize_rule(current);
                current = MoralityRule();
                current.id = line.substr(1, line.size() - 2);
                in_rule = true;
                continue;
            }

            if (!in_rule) continue;

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));

            if (key == "type") current.type = val;
            else if (key == "description") current.description = val;
            else if (key == "patterns") {
                std::istringstream ps(val);
                std::string p;
                while (std::getline(ps, p, ';')) {
                    p = trim(p);
                    if (!p.empty()) current.patterns.push_back(p);
                }
            }
            else if (key == "redirect_branch") current.redirect_branch = std::stoi(val);
            else if (key == "min_confidence") current.min_confidence = std::stof(val);
            else if (key == "active") current.active = (val == "true" || val == "1");
        }
        if (in_rule) finalize_rule(current);

        std::cout << "[Morality] Loaded " << rules.size() << " rules from " << path << "\n";
        return true;
    }

    // --------------------------------------------------------
    // Check input: Layer 1 (regex) only — called BEFORE routing
    // --------------------------------------------------------
    MoralityCheckResult check_input(const Tensor& input,
                                    const std::string& text_input = "") {
        total_checks++;
        MoralityCheckResult result;
        if (!enabled) return result;

        for (auto& rule : rules) {
            if (!rule.active) continue;

            if (rule.type == "hard_block") {
                if (matches_patterns(text_input, rule)) {
                    result.allowed = false;
                    result.triggered_rule = rule.id;
                    result.reason = "Input blocked by rule: " + rule.description;
                    log_action(rule.id, "blocked", "input matched hard_block pattern");
                    total_blocks++;
                    return result;
                }
            }
            else if (rule.type == "soft_redirect") {
                if (matches_patterns(text_input, rule)) {
                    result.redirect = true;
                    result.redirect_branch = rule.redirect_branch;
                    result.triggered_rule = rule.id;
                    result.reason = "Input redirected by rule: " + rule.description;
                    log_action(rule.id, "redirected",
                              "input redirected to branch " + std::to_string(rule.redirect_branch));
                    total_redirects++;
                }
            }
        }
        return result;
    }

    // --------------------------------------------------------
    // Check output: Layer 1 (regex) + Layer 2 (critique) + Layer 3 (conf gate)
    // Called AFTER conductor produces output
    // heat_scores and context_sim feed the confidence gate
    // --------------------------------------------------------
    MoralityCheckResult check_output(const Tensor& output,
                                     const std::string& text_output = "",
                                     const Tensor& heat_scores = Tensor(),
                                     float context_sim = 0.5f) {
        total_checks++;
        MoralityCheckResult result;
        if (!enabled) return result;

        // ---- Layer 1: Regex / pattern-based rules ----
        // Also supports legacy single-threshold confidence_gate from config
        for (auto& rule : rules) {
            if (!rule.active) continue;

            if (rule.type == "hard_block" && !text_output.empty()) {
                if (matches_patterns(text_output, rule)) {
                    result.allowed = false;
                    result.triggered_rule = rule.id;
                    result.reason = "Output blocked by rule: " + rule.description;
                    log_action(rule.id, "blocked", "output matched hard_block pattern");
                    total_blocks++;
                    return result;
                }
            }

            // Legacy single-threshold confidence gate from config
            if (rule.type == "confidence_gate" && !confidence_gate_enabled) {
                float max_conf = output.max_val();
                if (max_conf < rule.min_confidence) {
                    result.allowed = false;
                    result.triggered_rule = rule.id;
                    result.reason = "Output confidence too low (" +
                        std::to_string(max_conf) + " < " +
                        std::to_string(rule.min_confidence) + ")";
                    log_action(rule.id, "low_confidence",
                              "confidence=" + std::to_string(max_conf));
                    total_confidence_gates++;
                    return result;
                }
            }
        }

        // ---- Layer 2: Constitutional self-critique ----
        if (critique_enabled && output.size() > 0) {
            float critique_score = critique_net.score_all(output, principles);
            result.critique_score = critique_score;
            if (critique_score < critique_threshold) {
                result.allowed = false;
                result.triggered_rule = "constitutional_critique";
                result.reason = "Output failed constitutional critique (score=" +
                    std::to_string(critique_score) + " < " +
                    std::to_string(critique_threshold) + ")";
                log_action("constitutional_critique", "critique_fail",
                          "critique_score=" + std::to_string(critique_score));
                total_critique_blocks++;
                return result;
            }
        }

        // ---- Layer 3: Multi-dimensional confidence gate ----
        if (confidence_gate_enabled && output.size() > 0) {
            float gate_score = confidence_gate.compute(output, heat_scores, context_sim);
            result.confidence_score = gate_score;
            if (gate_score < confidence_gate_threshold) {
                result.allowed = false;
                result.triggered_rule = "multi_dim_confidence_gate";
                result.reason = "Multi-dim confidence gate triggered (score=" +
                    std::to_string(gate_score) + " < " +
                    std::to_string(confidence_gate_threshold) + ")";
                log_action("multi_dim_confidence_gate", "low_confidence",
                          "gate_score=" + std::to_string(gate_score));
                total_confidence_gates++;
                return result;
            }
        }

        return result;
    }

    // --------------------------------------------------------
    // Verify config integrity
    // --------------------------------------------------------
    bool verify_integrity() const {
        if (config_path.empty()) return true;
        std::ifstream file(config_path);
        if (!file.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        return std::hash<std::string>{}(content) == config_checksum;
    }

    // Adam for learned sub-networks
    void apply_adam(float lr) {
        if (critique_enabled) critique_net.apply_adam(lr);
        if (confidence_gate_enabled) confidence_gate.apply_adam(lr);
    }

    size_t param_count() const {
        return critique_net.param_count() + confidence_gate.param_count();
    }

    // --------------------------------------------------------
    // Print
    // --------------------------------------------------------
    void print_rules() const {
        std::cout << "  Morality rules (" << rules.size() << " loaded):\n";
        for (auto& r : rules) {
            printf("    [%s] %s (%s) %s\n", r.id.c_str(), r.type.c_str(),
                   r.description.c_str(), r.active ? "ACTIVE" : "DISABLED");
        }
        if (critique_enabled)
            printf("  Constitutional critique: %zu principles, threshold=%.2f\n",
                   principles.size(), critique_threshold);
        if (confidence_gate_enabled)
            printf("  Multi-dim confidence gate: threshold=%.2f, floor=%.2f\n",
                   confidence_gate_threshold, confidence_gate.hard_floor);
    }

    void print_stats() const {
        printf("  Morality layer: %zu checks, %zu blocks, %zu redirects, "
               "%zu conf-gates, %zu critique-blocks\n",
               total_checks, total_blocks, total_redirects,
               total_confidence_gates, total_critique_blocks);
        printf("  Audit log: %zu entries\n", audit_log.size());
    }

    void print_audit(int last_n = 10) const {
        int start = std::max(0, (int)audit_log.size() - last_n);
        for (int i = start; i < (int)audit_log.size(); i++) {
            auto& e = audit_log[i];
            printf("    [%s] rule=%s action=%s detail=%s\n",
                   e.timestamp.c_str(), e.rule_id.c_str(),
                   e.action.c_str(), e.details.c_str());
        }
    }

private:
    void init_default_principles(size_t dim, std::mt19937& rng) {
        // Initialise a few prototype constitutional principles
        // Their embeddings are random unit vectors — the critique network
        // learns to associate output patterns with safety scores against them.
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto make_unit = [&](const std::string& name, const std::string& desc, float thresh) {
            ConstitutionalPrinciple p;
            p.name = name;
            p.description = desc;
            p.threshold = thresh;
            p.embedding = Tensor({dim});
            float norm = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                p.embedding[i] = dist(rng);
                norm += p.embedding[i] * p.embedding[i];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-7f)
                for (size_t i = 0; i < dim; i++) p.embedding[i] /= norm;
            principles.push_back(std::move(p));
        };
        make_unit("harmlessness",    "Output should not cause harm",           0.3f);
        make_unit("helpfulness",     "Output should be genuinely helpful",     0.2f);
        make_unit("honesty",         "Output should be accurate and truthful", 0.2f);
        make_unit("non_deception",   "Output should not deceive the user",     0.3f);
    }

    void finalize_rule(MoralityRule& rule) {
        for (auto& p : rule.patterns) {
            try {
                rule.compiled.push_back(std::regex(p, std::regex::icase));
            } catch (...) {
                std::cout << "[Morality] WARNING: invalid regex '" << p
                          << "' in rule " << rule.id << "\n";
            }
        }
        rules.push_back(rule);
    }

    bool matches_patterns(const std::string& text, const MoralityRule& rule) const {
        if (text.empty()) return false;
        for (auto& re : rule.compiled) {
            if (std::regex_search(text, re)) return true;
        }
        return false;
    }

    void log_action(const std::string& rule_id, const std::string& action,
                    const std::string& details) {
        AuditEntry entry;
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        entry.timestamp = buf;
        entry.rule_id = rule_id;
        entry.action = action;
        entry.details = details;
        audit_log.push_back(entry);
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
};

} // namespace dendrite
