---
name: known-issues
description: Status and investigation findings for all 7 DendriteNet known issues
type: project
---

# Known Issues — DendriteNet v0.3.0

## Issue #1 — Training instability after ~15 epochs
**Status:** Open
**Root cause:** Not yet investigated
**Relevant files:** `include/layer.hpp` (Adam step), `include/conductor.hpp` (training loop), `examples/main.cpp`
**Hypothesis:** No LR warmup — Adam steps are large early in training and may set bad momentum estimates that compound. A linear warmup over ~5 epochs before the main LR schedule should stabilise.
**Verification:** Loss curve should stop diverging; accuracy should continue improving past epoch 15.

## Issue #2 — Data cleaner anomaly rate 72%
**Status:** Open
**Root cause:** Not yet investigated
**Relevant files:** `include/data_cleaner.hpp` (`heat_threshold` parameter)
**Hypothesis:** `heat_threshold` set too low, flagging normal variance as anomalies.
**Verification:** Anomaly rate should drop to 10–20% on the synthetic dataset after tuning.

## Issue #3 — Context items all show rel=1.0
**Status:** Open
**Root cause:** Not yet investigated
**Relevant files:** `include/task_context.hpp` (reinforcement logic)
**Hypothesis:** Reinforcement increment too large or applied unconditionally; Ebbinghaus decay not balancing it. Result: all items clamp to 1.0.
**Verification:** relevance values should distribute across [0.1, 1.0] and decay for unused items.

## Issue #4 — Simulated image/audio encoders
**Status:** Open
**Root cause:** By design (stubs) — needs real ONNX Runtime calls
**Relevant files:** `include/modality.hpp`
**Next step:** Use `/onnx-integrate` skill to plan and implement.
**Verification:** Real embeddings should differ from zero/random stubs; downstream accuracy should improve.

## Issue #5 — Strategy selector: add Gumbel-softmax
**Status:** Open
**Root cause:** Not yet investigated
**Relevant files:** `include/conductor.hpp` (FusionStrategy selector)
**Hypothesis:** Current selector likely uses argmax (not differentiable). Gumbel-softmax would allow gradients to flow through the discrete strategy choice.
**Verification:** Strategy selector gradients should be non-zero after change.

## Issue #6 — child_router doesn't backprop
**Status:** Open
**Root cause:** Not yet investigated
**Relevant files:** `include/conductor.hpp` (SubConductor, child_router)
**Hypothesis:** child_router forward pass detaches gradient or uses `std::vector` indexing that breaks the chain.
**Verification:** child_router parameters should change between epochs after fix.

## Issue #7 — No GPU support
**Status:** Open — low priority until ONNX integration done
**Root cause:** By design — CPU-only implementation
**Relevant files:** `include/tensor.hpp`, `include/layer.hpp`
**Path forward:** ONNX Runtime GPU EP (CUDA/DirectML) is the cleanest path; avoids writing CUDA kernels.
**Verification:** Inference time should drop >10× on a GPU.
