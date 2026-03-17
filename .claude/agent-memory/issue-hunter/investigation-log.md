---
name: investigation-log
description: Per-session investigation notes from issue-hunter; append-only
type: project
---

# Investigation Log

*Append new entries at the top. Format: `## YYYY-MM-DD — Issue #N: [title]`*

## 2026-03-15 — Accuracy Regression Investigation (84.3% → 58.7%)

### Summary
Investigated accuracy crash (epoch 1 = 96.8%, crash to 52% by epoch 10, plateau at ~59%)
after 7 Tier-1 enhancements. Found 5 confirmed root causes and 2 partial contributors.
Ranked by estimated impact.

### CONFIRMED: Issue A — Double kWTA signal destruction (HIGH IMPACT)
**File:** `include/dendrite3d.hpp` line 269, `include/layer.hpp` lines 342-353
- DendriticLayer created with `kwta=0.2f` and `input_dim=10, output_dim=10`
- kWTA zeroes 8 of 10 dimensions every forward pass
- Called once per branch per training step (4 branches × 2400 samples = 9600×/epoch)
- The 80% zeroing is permanent within the forward pass — specialists see a nearly empty input
- This alone can explain early high accuracy (random lucky routing) then collapse as patterns reinforce wrong routing

### CONFIRMED: Issue B — Dendritic backward second pass corrupts accumulated gradients (HIGH IMPACT)
**File:** `include/dendrite3d.hpp` lines 706-727
- Lines 717-718: `spec->backward(branch_grad_in)` is called a second time on the same specialist
  (first call at line 697, second at line 717) then `grad_w.zero(); grad_b.zero()` erases accumulators
- The second backward() call accumulates NEW gradient values in grad_w/grad_b
- Then line 718 immediately zeros them — so specialist weights for the dendritic pass are NOT updated
- But the FIRST backward (line 697) DID accumulate, and apply_adam at line 698 was already called
- Net effect: specialists run backward twice per step; the first apply_adam call fires on
  the gradient from pass #1; the second backward (with zeroed grad_w after adam) then
  accumulates a THIRD gradient that is never cleared before the next sample —
  causing gradient bleeding between samples

### CONFIRMED: Issue C — Double task_context.step() per training step (MEDIUM IMPACT)
**File:** `include/dendrite3d.hpp` lines 315 (infer) and 521 (train_sample)
- train_sample() calls `task_context.step()` at line 521
- infer() calls `task_context.step()` at line 315
- The evaluate() function in main.cpp calls `net.infer()` — but that's only at eval time
- The real double-step happens indirectly: `evaluate_branch()` is called from both
  `infer()` and `train_sample()`, but task_context.step() is called at the top of BOTH paths
- During the evaluate() pass in the training loop (main.cpp line 192), infer() is called
  2400+ times (once per test sample), each calling step() again
- With decay_rate=0.85, calling step() 2400 extra times per eval degrades memory entries
  far faster than intended, causing premature eviction and unstable context vectors

### CONFIRMED: Issue D — SI locks in bad weights at epoch 1 (MEDIUM IMPACT)
**File:** `include/dendrite3d.hpp` lines 844-854, `include/layer.hpp` lines 117-135
- `train_batch()` calls `consolidate_all()` at the END of every epoch (line 847)
- Epoch 1 has random He-initialized weights; the path integral accumulates noisy gradients
- `consolidate_importance()` computes omega += max(0, running_sum) / (delta^2 + 1e-8)
- After epoch 1, omega values are non-zero; SI penalty = `si_lambda * omega * (w - prev_w)`
  is added to grad_w before Adam updates
- With random early weights, the delta in denom (w - prev_w)^2 can be very small (weights
  barely moved on the first epoch), making denom near 1e-8 and omega astronomically large
- This creates massive SI gradients that fight against useful weight changes from epoch 2 onward

### CONFIRMED: Issue E — VICReg diversity loss fights classification (MEDIUM IMPACT)
**File:** `include/dendrite3d.hpp` lines 607-672
- VICReg activates at step 500 (< 1 epoch with 2400 samples)
- Variance loss hinge: `max(0, 1.0 - std_dev)` where target std_dev > 1.0
- Branch outputs are softmax probabilities ∈ [0,1]; their std_dev is at most ~0.4 for a
  6-class output (1/6 ≈ 0.167 mean, std ≪ 1.0)
- The hinge is ALWAYS active on softmax outputs (std can never reach 1.0)
- This means VICReg variance loss is a constant penalty throughout training, pushing
  branches toward more uniform distributions — directly opposing classification learning
- Weight = vicreg_weight * (var_loss + 0.5 * cov_loss + div_loss) = 0.1 * (large + ...)
  adds a non-trivial loss component at every step after warmup

### PARTIAL: Issue F — Alpha gate can dominate (LOW-MEDIUM IMPACT)
**File:** `include/dendrite3d.hpp` lines 286-290, 581-597
- Bias init = -1.0 → initial alpha = sigmoid(-1) ≈ 0.27
- After the blend, output = 0.27 * fused_specialist + 0.73 * shared_expert
- Shared expert is untrained initially; it produces near-uniform predictions
- This explains the very high epoch-1 accuracy (≈ 97%) — both specialist and shared are
  in their initial good-enough random state — but once training diverges it pulls output
  toward a stabilised but wrong shared expert
- alpha_gate lr multiplier is 0.1x (line 806), so alpha adapts very slowly; the initial
  0.73 shared-expert weighting persists for many epochs

### PARTIAL: Issue G — Early exit aux loss every active branch every step (LOW IMPACT)
**File:** `include/dendrite3d.hpp` lines 810-826
- exit_loss_weight = 0.3 applied to every active branch every step
- 4 branches × 0.3 = 1.2 total auxiliary loss added to every step (before normalization)
- Exit classifiers use `evaluate()` which runs another forward through the exit classifier
  using the cached `last_exit_hidden` — this is stale if the second backward (Issue B)
  modified the hidden activations
- Less critical than Issues A-E but contributes noise

### NOT CONFIRMED: Issue H — Lateral inhibition burn-in (200 steps)
- 200 steps ≈ 8% of epoch 1; cross_talk summaries are nearly random initially
- After burn-in, similarity is near-zero (random summaries → low cosine similarity)
- Impact is minimal; alpha=0.3 and sim≈0 means inhibition ≈ 0; safe

### Estimated impact ranking:
1. Issue B (double backward + grad bleed) — probably the single biggest contributor
2. Issue A (kWTA 80% signal destruction) — structural, always active
3. Issue E (VICReg var loss on softmax) — constant upward loss pressure
4. Issue D (SI locking bad epoch-1 weights) — compounds each epoch
5. Issue F (alpha gate initial weighting) — explains epoch-1 spike + slow recovery
6. Issue C (double step() during eval) — erodes context quality over training
7. Issue G (early exit aux loss noise) — minor

## 2026-03-14 17:10 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 14:39 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 14:42 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 14:44 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 16:47 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 17:47 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 19:06 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 23:25 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-15 23:48 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-16 00:14 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-17 12:57 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.

## 2026-03-17 19:47 — auto-compaction
Context window compacted. Prior findings preserved above. Resume with: `/fix-issue [N]` or read known-issues.md.
