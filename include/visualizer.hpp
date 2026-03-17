#pragma once
// ============================================================
// Visualizer: DOT graph export + JSONL event stream
// ============================================================
// Option 2 — Graphviz DOT per epoch:
//   Renders the conductor tree with heat-driven colours and
//   edge weights. Convert a file with:
//     dot -Tsvg viz/epoch_10.dot -o viz/epoch_10.svg
//
// Option 3 — JSONL trace for live matplotlib dashboard:
//     python viz/plot.py viz/trace.jsonl
//
// Usage in main:
//   Visualizer viz("viz/epoch", "viz/trace.jsonl");
//   viz.log_meta(net);                                   // once
//   auto snap = net.infer(sample, "", false);
//   viz.write_dot(net, epoch, val_acc, snap);            // each eval
//   viz.log_epoch(epoch, loss, val_acc, lr, snap, ctx);  // each eval
// ============================================================
#include "dendrite3d.hpp"
#include <fstream>
#include <iomanip>
#include <set>

#if defined(_WIN32)
#  include <direct.h>
#  define DENDRITE_MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  define DENDRITE_MKDIR(p) ::mkdir(p, 0755)
#endif

namespace dendrite {

class Visualizer {
public:
    std::string dot_prefix;   // e.g. "viz/epoch"  → viz/epoch_1.dot …
    std::string json_path;    // e.g. "viz/trace.jsonl"

    Visualizer() = default;

    Visualizer(std::string dot_prefix_, std::string json_path_)
        : dot_prefix(std::move(dot_prefix_)), json_path(std::move(json_path_)) {
        // Best-effort: create the viz/ directory
        auto sep = json_path.find_last_of("/\\");
        if (sep != std::string::npos)
            DENDRITE_MKDIR(json_path.substr(0, sep).c_str());
        // Truncate / create the JSONL file
        std::ofstream(json_path, std::ios::trunc);
    }

    // --------------------------------------------------------
    // Write one "meta" event (branch topology) at startup
    // --------------------------------------------------------
    void log_meta(const DendriteNet3D& net) {
        std::ofstream f(json_path, std::ios::app);
        if (!f.is_open()) return;
        f << "{\"type\":\"meta\""
          << ",\"input_dim\":"  << net.input_dim
          << ",\"output_dim\":" << net.output_dim
          << ",\"branches\":[";
        for (size_t b = 0; b < net.branches.size(); b++) {
            if (b) f << ",";
            const auto& br = net.branches[b];
            f << "{\"id\":"         << b
              << ",\"domain\":\""   << escape_quoted(br->domain) << "\""
              << ",\"children\":"   << br->children.size()
              << "}";
        }
        f << "]}\n";
        f.flush();
    }

    // --------------------------------------------------------
    // Append one "epoch" event to the JSONL trace
    // --------------------------------------------------------
    // spec_mi: branch specialization Mutual Information (-1 = not computed)
    void log_epoch(int epoch, float loss, float val_acc, float lr,
                   const InferenceResult3D& snap, size_t context_items,
                   float spec_mi = -1.0f) {
        std::ofstream f(json_path, std::ios::app);
        if (!f.is_open()) return;
        f << std::fixed << std::setprecision(6);
        f << "{\"type\":\"epoch\""
          << ",\"epoch\":"         << epoch
          << ",\"loss\":"          << loss
          << ",\"val_acc\":"       << val_acc
          << ",\"lr\":"            << lr
          << ",\"strategy\":\""    << fusion_name(snap.strategy_used) << "\""
          << ",\"context_items\":" << context_items;
        if (spec_mi >= 0.0f) f << ",\"spec_mi\":" << spec_mi;
        f << ",\"heat\":[";
        for (size_t i = 0; i < snap.heat_scores.size(); i++) {
            if (i) f << ",";
            f << snap.heat_scores[i];
        }
        f << "]}\n";
        f.flush();
    }

    // --------------------------------------------------------
    // Write a Graphviz DOT file for the given epoch
    //
    // Node colour: cold #CFE2FF → hot #E53935 (heat-driven)
    // Active branches: bold red edge from conductor
    // Child branches: dashed grey edges
    // --------------------------------------------------------
    void write_dot(const DendriteNet3D& net, int epoch, float val_acc,
                   const InferenceResult3D& snap) {
        std::string path = dot_prefix + "_" + std::to_string(epoch) + ".dot";
        std::ofstream f(path);
        if (!f.is_open()) return;

        const Tensor& heat = snap.heat_scores;
        std::set<int> active(snap.active_branch_ids.begin(),
                             snap.active_branch_ids.end());

        // -- header --
        f << "digraph DendriteNet {\n";
        f << "  graph [label=\"DendriteNet — Epoch " << epoch
          << "  val_acc=" << std::fixed << std::setprecision(1) << (val_acc * 100.0f) << "%"
          << "  strategy=" << fusion_name(snap.strategy_used)
          << "\" fontsize=13 fontname=Helvetica bgcolor=white];\n";
        f << "  rankdir=TB; splines=ortho;\n";
        f << "  node [style=filled fontname=Helvetica fontsize=10];\n";
        f << "  edge [fontname=Helvetica fontsize=9];\n\n";

        // -- conductor root --
        f << "  root [label=\"Conductor\\n[" << fusion_name(snap.strategy_used) << "]"
          << "\\nsteps=" << net.total_train_steps
          << "\" shape=diamond fillcolor=\"#1565C0\" fontcolor=white penwidth=2];\n\n";

        // -- branch nodes --
        for (size_t b = 0; b < net.branches.size(); b++) {
            const auto& br = net.branches[b];
            float h = (b < heat.size()) ? heat[b] : 0.0f;
            h = std::max(0.0f, std::min(1.0f, h));
            bool is_active = active.count(static_cast<int>(b)) > 0;

            // Colour interpolation: #CFE2FF (cold) → #E53935 (hot)
            char fill[12];
            snprintf(fill, sizeof(fill), "#%02X%02X%02X",
                     lerp_ch(207, 229, h),   // R
                     lerp_ch(226,  57, h),   // G
                     lerp_ch(255,  53, h));  // B

            f << "  b" << b
              << " [label=\"" << escape_quoted(br->domain)
              << "\\nheat=" << std::fixed << std::setprecision(2) << h
              << "  conf="  << std::fixed << std::setprecision(2) << br->running_confidence
              << "\\nvisits=" << br->visit_count << "\""
              << " shape=box"
              << " penwidth=" << (is_active ? 3 : 1)
              << " fillcolor=\"" << fill << "\""
              << " fontcolor=" << (h > 0.55f ? "white" : "black")
              << "];\n";

            // edge: conductor → branch
            f << "  root -> b" << b
              << " [label=\"" << std::fixed << std::setprecision(2) << h << "\""
              << " penwidth=" << std::fixed << std::setprecision(1) << (1.0f + 4.0f * h)
              << " color=" << (is_active ? "\"#E53935\"" : "\"#90A4AE\"")
              << "];\n";

            // child branches (dashed)
            for (size_t c = 0; c < br->children.size(); c++) {
                const auto& ch = br->children[c];
                std::string cid = "b" + std::to_string(b) + "c" + std::to_string(c);
                f << "  " << cid
                  << " [label=\"" << escape_quoted(ch->domain)
                  << "\\nvisits=" << ch->visit_count
                  << "\" shape=ellipse fillcolor=\"#E8F5E9\" penwidth=1];\n";
                f << "  b" << b << " -> " << cid
                  << " [style=dashed color=\"#78909C\"];\n";
            }
        }

        f << "}\n";
    }

private:
    static int lerp_ch(int a, int b, float t) {
        return a + static_cast<int>((b - a) * t);
    }

    static std::string escape_quoted(const std::string& s) {
        std::string out; out.reserve(s.size());
        for (char c : s) {
            if      (c == '"')  { out += "\\\""; }
            else if (c == '\\') { out += "\\\\"; }
            else                { out += c; }
        }
        return out;
    }
};

} // namespace dendrite
