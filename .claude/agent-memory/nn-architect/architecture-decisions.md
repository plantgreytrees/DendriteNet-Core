---
name: architecture-decisions
description: Rationale for key DendriteNet architectural decisions; append when a design choice is made
type: project
---

# Architecture Decisions

*Append new entries at the top. Format: `## YYYY-MM-DD — [topic]`*

## 2026-03-14 — Initial architecture (v0.3.0 baseline)
**Decision:** Header-only, single-file-per-module, no build system.
**Why:** Zero setup friction; easy to embed. Tradeoff: whole-project recompile on any change.

**Decision:** Adam everywhere with β1=0.9, β2=0.999, ε=1e-8.
**Why:** Robust default; no per-module tuning needed. Tradeoff: may be slower than SGD+momentum on well-tuned problems.

**Decision:** Morality layer loads from `config/morality.cfg`, checksummed, immutable to training.
**Why:** Safety-critical rules must survive gradient updates. Config checksum detects tampering.

**Decision:** TaskContext uses Ebbinghaus fading (exponential decay) with `decay_rate=0.85`.
**Why:** Approximates human memory forgetting curves. Known issue: `reinforce_boost=0.4` is too large — all items clamp to rel=1.0 (issue #3).

**Decision:** DataCleaner uses `heat_threshold=0.3` for branch assignment.
**Why:** Empirically set. Known issue: produces 72% anomaly rate on synthetic data — threshold too low (issue #2).

## 2026-03-17 — Specialization MI collapse root causes (diagnosed, not yet fixed)
**Decision context:** MI starts at 0.5195 at epoch 1 then collapses to 0.0000 from epoch 5 onwards.
Four independent causes identified:

1. **heat_target floor = 0.2 > heat_threshold = 0.2**: In `train_sample`, non-best branches receive `heat_target[i] = 0.2f`. The `heat_threshold` in main.cpp is also set to `0.2f`. NMDA + load-balancing push every branch to converge on ~0.2, which is exactly at the threshold. MI computation via `compute_specialization_metrics` uses rank-based top-k, so this causes all 4 branches to appear equally tied → uniform routing → MI = 0.

2. **top_k=2 during training evaluates ALL branches unconditionally**: `train_sample` always adds all branches to `active` regardless of heat (line 1048-1050). Only inference respects `heat_threshold`. The MI diagnostic calls `compute_specialization_metrics` with `top_k=2` (the DendriteNet3D member), meaning 2 out of 4 branches are always "selected" — but if heat is uniform, ties are broken by sort order, so branches 0 and 1 always win in training and branches 1 and 2 always win at test (different call path produces different tie-breaking). This explains the training visit count = 157920 for all branches but test shows 0 visits for branches 0 and 3.

3. **Gumbel-softmax tau collapses to 0.1 by ~step 5000**: `current_tau = max(0.1, 1.0 - steps/5000)`. With ~2400 steps/epoch (600 samples × 4 branches evaluated), tau hits 0.1 during epoch 2. At tau=0.1 the Gumbel sample is nearly one-hot on the already-highest strategy — this doesn't directly cause uniform routing, but it freezes strategy exploration early.

4. **Load-balancing target_usage = top_k/num_branches = 2/4 = 0.5**: Every branch is being biased toward 50% usage. NMDA threshold is 0.3. With NMDA steepness annealing from 3→10, at steepness 10 any heat near 0.3 maps to ~0.5 NMDA output. Load-balancing then pushes all branches toward 0.5. Since heat_target non-best floor (0.2) is below NMDA threshold (0.3), the heat gradient is pushing away from where NMDA activates, counteracting specialization.

**Why this matters:** The MI is not a measurement bug in the formula itself — the formula is correct. It's a routing collapse: actual heat values converge to near-uniform because (a) heat_target non-best floor is at the NMDA threshold boundary, (b) load-balancing target is 50% for a 4-branch/top-2 system, and (c) training evaluates all branches unconditionally while the metric uses rank-based top-2, creating a disconnect between what the heat network learns and what MI measures.

<!-- New architecture decisions go above this line -->
