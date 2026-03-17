#pragma once
#include "layer.hpp"
#include <deque>

namespace dendrite {

// ============================================================
// ContextItem: a single piece of information in working memory
// ============================================================
struct ContextItem {
    Tensor data;
    Tensor key;             // learned key projection for content-based addressing
    float relevance;        // starts at 1.0, decays each step
    int branch_source;      // which branch produced this
    int step_created;       // when this was stored
    int last_accessed;      // last step where this was reinforced
    std::string label;      // human-readable description
    bool consolidated = false;  // true once promoted to semantic store
};

// ============================================================
// TaskContext: dual working memory with Ebbinghaus-style fading
// ============================================================
// Episodic memory: fast, volatile (decay=0.85, max 32 items)
//   - Stores recent branch outputs; items decay quickly without reinforcement
// Semantic memory: slow, persistent (decay=0.99, max 64 items)
//   - High-relevance episodic items are promoted here (consolidation)
//   - Represents durable "knowledge" across many steps
// get_context() blends both stores (60% episodic, 40% semantic)
// ============================================================
class TaskContext {
public:
    // Episodic store (fast, volatile)
    std::deque<ContextItem> episodic_items;
    // Semantic store (slow, durable)
    std::deque<ContextItem> semantic_items;

    MiniNetwork context_projection;  // compress all items → fixed-dim vector
    MiniNetwork key_projector;       // projects data → key space
    size_t projection_dim = 0;       // output dimension of the projection
    size_t key_dim = 32;             // dimension of key vectors

    // Episodic parameters
    float decay_rate = 0.85f;                // per-step decay multiplier (episodic)
    float eviction_threshold = 0.1f;         // below this, item is removed
    float reinforce_boost = 0.2f;            // added when an item is accessed
    size_t max_items = 32;                   // hard cap on episodic memory

    // Semantic parameters
    float semantic_decay_rate = 0.99f;       // much slower decay
    float consolidation_threshold = 0.8f;    // episodic relevance needed for promotion
    size_t max_semantic_items = 64;          // hard cap on semantic memory

    // Blend weights for get_context()
    float episodic_blend = 0.60f;
    float semantic_blend = 0.40f;

    int current_step = 0;

    // Statistics
    size_t total_stored = 0;
    size_t total_evicted = 0;
    size_t total_reinforced = 0;
    size_t total_consolidated = 0;

    TaskContext() = default;

    TaskContext(size_t item_dim, size_t proj_dim, std::mt19937& rng)
        : projection_dim(proj_dim), key_dim(std::min((size_t)32, proj_dim)) {
        // Projection network: takes concatenated summary → compressed vector
        // Input: item_dim * 2 (mean + max pool of all items)
        context_projection = MiniNetwork("ctx_proj",
            {item_dim * 2, proj_dim * 2, proj_dim},
            Activation::RELU, Activation::TANH, rng);
        // Key projector: maps stored/query data into a shared key space for attention
        key_projector = MiniNetwork("key_proj",
            {item_dim, key_dim * 2, key_dim},
            Activation::RELU, Activation::TANH, rng);
    }

    // --------------------------------------------------------
    // Step: decay all items, consolidate high-relevance episodic
    // items into semantic store, evict dead items
    // --------------------------------------------------------
    void step() {
        current_step++;

        // Decay episodic items
        for (auto& item : episodic_items)
            item.relevance *= decay_rate;

        // Decay semantic items (much slower)
        for (auto& item : semantic_items)
            item.relevance *= semantic_decay_rate;

        // Consolidation: promote high-relevance episodic items to semantic store
        for (auto& item : episodic_items) {
            if (!item.consolidated && item.relevance >= consolidation_threshold) {
                consolidate_to_semantic(item);
                item.consolidated = true;
            }
        }

        // Evict episodic items below threshold
        size_t before = episodic_items.size();
        episodic_items.erase(
            std::remove_if(episodic_items.begin(), episodic_items.end(),
                [this](const ContextItem& item) {
                    return item.relevance < eviction_threshold;
                }),
            episodic_items.end()
        );
        total_evicted += before - episodic_items.size();

        // Evict semantic items below a lower threshold (they persist longer)
        semantic_items.erase(
            std::remove_if(semantic_items.begin(), semantic_items.end(),
                [this](const ContextItem& item) {
                    return item.relevance < eviction_threshold;
                }),
            semantic_items.end()
        );
    }

    // --------------------------------------------------------
    // Store: add a new item to episodic working memory
    // --------------------------------------------------------
    void store(const Tensor& data, int branch_id, float importance = 1.0f,
               const std::string& label = "") {
        ContextItem item;
        item.data = data;
        item.relevance = importance;
        item.branch_source = branch_id;
        item.step_created = current_step;
        item.last_accessed = current_step;
        item.label = label;
        item.consolidated = false;

        // Project data into key space for content-based addressing
        item.key = key_projector.forward(data);
        for (auto& v : item.key.data) if (!std::isfinite(v)) v = 0.0f;  // NaN guard

        // If at capacity, evict the least relevant item
        if (episodic_items.size() >= max_items) {
            auto min_it = std::min_element(episodic_items.begin(), episodic_items.end(),
                [](const ContextItem& a, const ContextItem& b) {
                    return a.relevance < b.relevance;
                });
            if (min_it != episodic_items.end()) {
                episodic_items.erase(min_it);
                total_evicted++;
            }
        }

        episodic_items.push_back(item);
        total_stored++;
    }

    // --------------------------------------------------------
    // Reinforce: bump relevance of items matching a branch
    // Call when a branch is actively being used
    // --------------------------------------------------------
    void reinforce(int branch_id) {
        for (auto& item : episodic_items) {
            if (item.branch_source == branch_id) {
                item.relevance = std::min(1.0f, item.relevance + reinforce_boost);
                item.last_accessed = current_step;
                total_reinforced++;
            }
        }
        // Also reinforce matching semantic items
        for (auto& item : semantic_items) {
            if (item.branch_source == branch_id) {
                item.relevance = std::min(1.0f, item.relevance + reinforce_boost * 0.5f);
                item.last_accessed = current_step;
            }
        }
    }

    void reinforce_weighted(int branch_id, float weight) {
        if (!std::isfinite(weight)) weight = 0.0f;
        weight = std::clamp(weight, 0.0f, 1.0f);
        for (auto& item : episodic_items) {
            if (item.branch_source == branch_id) {
                item.relevance = std::min(1.0f, item.relevance + reinforce_boost * weight);
                item.last_accessed = current_step;
                total_reinforced++;
            }
        }
        for (auto& item : semantic_items) {
            if (item.branch_source == branch_id) {
                item.relevance = std::min(1.0f, item.relevance + reinforce_boost * weight * 0.5f);
                item.last_accessed = current_step;
            }
        }
    }

    // Reinforce items whose data is similar to a query
    void reinforce_similar(const Tensor& query, float threshold = 0.5f) {
        auto reinforce_store = [&](std::deque<ContextItem>& store, float boost_scale) {
            for (auto& item : store) {
                float dot = 0, na = 0, nb = 0;
                size_t dim = std::min(query.size(), item.data.size());
                for (size_t i = 0; i < dim; i++) {
                    dot += query[i] * item.data[i];
                    na += query[i] * query[i];
                    nb += item.data[i] * item.data[i];
                }
                float sim = (na > 1e-7f && nb > 1e-7f) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0;
                if (sim > threshold) {
                    item.relevance = std::min(1.0f, item.relevance + reinforce_boost * sim * boost_scale);
                    item.last_accessed = current_step;
                    total_reinforced++;
                }
            }
        };
        reinforce_store(episodic_items, 1.0f);
        reinforce_store(semantic_items, 0.5f);
    }

    // --------------------------------------------------------
    // Get context: blended retrieval from episodic + semantic stores
    // 60% episodic (recency) + 40% semantic (durable knowledge)
    // --------------------------------------------------------
    Tensor get_context(const Tensor& query) {
        bool have_episodic = !episodic_items.empty();
        bool have_semantic = !semantic_items.empty();

        if (!have_episodic && !have_semantic) {
            return Tensor({projection_dim});  // zero vector
        }

        Tensor episodic_ctx({projection_dim});
        Tensor semantic_ctx({projection_dim});

        if (have_episodic)
            episodic_ctx = compute_store_context(query, episodic_items);
        if (have_semantic)
            semantic_ctx = compute_store_context(query, semantic_items);

        // Blend: 60% episodic + 40% semantic
        Tensor result({projection_dim});
        float e_w = have_episodic ? episodic_blend : 0.0f;
        float s_w = have_semantic ? semantic_blend : 0.0f;
        float total_w = e_w + s_w;
        if (total_w < 1e-7f) return result;
        e_w /= total_w;
        s_w /= total_w;
        for (size_t i = 0; i < projection_dim; i++) {
            result[i] = e_w * episodic_ctx[i] + s_w * semantic_ctx[i];
            if (!std::isfinite(result[i])) result[i] = 0.0f;
        }
        return result;
    }

    // --------------------------------------------------------
    // Query: find the most relevant items for a given input
    // --------------------------------------------------------
    std::vector<const ContextItem*> query_top_k(const Tensor& input, int k = 3) const {
        std::vector<std::pair<float, const ContextItem*>> scored;
        auto score_store = [&](const std::deque<ContextItem>& store) {
            for (auto& item : store) {
                float dot = 0, na = 0, nb = 0;
                size_t dim = std::min(input.size(), item.data.size());
                for (size_t i = 0; i < dim; i++) {
                    dot += input[i] * item.data[i];
                    na += input[i] * input[i];
                    nb += item.data[i] * item.data[i];
                }
                float sim = (na > 1e-7f && nb > 1e-7f) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0;
                scored.push_back({sim * item.relevance, &item});
            }
        };
        score_store(episodic_items);
        score_store(semantic_items);
        std::sort(scored.begin(), scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });

        std::vector<const ContextItem*> result;
        for (int i = 0; i < k && i < (int)scored.size(); i++)
            result.push_back(scored[i].second);
        return result;
    }

    // --------------------------------------------------------
    // Reset: clear all memory (end of task/session)
    // --------------------------------------------------------
    void reset() {
        episodic_items.clear();
        semantic_items.clear();
        current_step = 0;
    }

    bool empty() const { return episodic_items.empty() && semantic_items.empty(); }
    size_t size() const { return episodic_items.size(); }  // episodic count (primary)
    size_t semantic_size() const { return semantic_items.size(); }
    size_t dim() const { return projection_dim; }

    void apply_adam(float lr) {
        context_projection.apply_adam(lr);
        key_projector.apply_adam(lr);
    }
    size_t param_count() const {
        return context_projection.param_count() + key_projector.param_count();
    }

    void print_state() const {
        printf("  Working memory: %zu episodic + %zu semantic / %zu+%zu capacity "
               "(stored=%zu evicted=%zu reinforced=%zu consolidated=%zu)\n",
               episodic_items.size(), semantic_items.size(),
               max_items, max_semantic_items,
               total_stored, total_evicted, total_reinforced, total_consolidated);
        for (auto& item : episodic_items) {
            printf("    [E branch %d, rel=%.2f, age=%d] %s\n",
                   item.branch_source, item.relevance,
                   current_step - item.step_created,
                   item.label.c_str());
        }
        for (auto& item : semantic_items) {
            printf("    [S branch %d, rel=%.2f, age=%d] %s\n",
                   item.branch_source, item.relevance,
                   current_step - item.step_created,
                   item.label.c_str());
        }
    }

private:
    // Promote an episodic item into the semantic store
    void consolidate_to_semantic(const ContextItem& src) {
        if (semantic_items.size() >= max_semantic_items) {
            // Evict least relevant semantic item
            auto min_it = std::min_element(semantic_items.begin(), semantic_items.end(),
                [](const ContextItem& a, const ContextItem& b) {
                    return a.relevance < b.relevance;
                });
            if (min_it != semantic_items.end())
                semantic_items.erase(min_it);
        }
        ContextItem sem = src;
        sem.relevance = std::min(1.0f, src.relevance);  // carry forward current relevance
        sem.consolidated = true;
        semantic_items.push_back(sem);
        total_consolidated++;
    }

    // Attention-weighted context vector from a single memory store
    Tensor compute_store_context(const Tensor& query, const std::deque<ContextItem>& store) {
        // Project query into key space
        Tensor query_key = key_projector.forward(query);
        for (auto& v : query_key.data) if (!std::isfinite(v)) v = 0.0f;

        float scale = 1.0f / std::sqrt((float)key_dim);
        std::vector<float> attn_logits(store.size());
        for (size_t i = 0; i < store.size(); i++) {
            float dot = 0.0f;
            size_t d = std::min(query_key.size(), store[i].key.size());
            for (size_t j = 0; j < d; j++)
                dot += query_key[j] * store[i].key[j];
            attn_logits[i] = dot * scale * store[i].relevance;
            if (!std::isfinite(attn_logits[i])) attn_logits[i] = 0.0f;
        }

        // Numerically stable softmax
        float max_logit = *std::max_element(attn_logits.begin(), attn_logits.end());
        float sum_exp = 0.0f;
        std::vector<float> attn_weights(store.size());
        for (size_t i = 0; i < store.size(); i++) {
            attn_weights[i] = std::exp(attn_logits[i] - max_logit);
            sum_exp += attn_weights[i];
        }
        if (sum_exp > 1e-7f)
            for (auto& w : attn_weights) w /= sum_exp;

        size_t item_dim = store[0].data.size();
        Tensor weighted_sum({item_dim});
        Tensor elem_max({item_dim});
        elem_max.fill(-1e9f);
        for (size_t i = 0; i < store.size(); i++) {
            for (size_t d = 0; d < item_dim && d < store[i].data.size(); d++) {
                weighted_sum[d] += attn_weights[i] * store[i].data[d];
                elem_max[d] = std::max(elem_max[d], store[i].data[d] * attn_weights[i]);
            }
        }
        for (size_t d = 0; d < item_dim; d++)
            if (elem_max[d] < -1e8f) elem_max[d] = 0.0f;

        Tensor combined = Tensor::concat(weighted_sum, elem_max);
        Tensor result = context_projection.forward(combined);
        for (auto& v : result.data) if (!std::isfinite(v)) v = 0.0f;
        return result;
    }
};

} // namespace dendrite
