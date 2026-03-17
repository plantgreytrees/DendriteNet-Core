# DendriteNet-Core — Framework Library

DendriteNet-Core is the reusable framework layer of DendriteNet 3D: biologically-inspired
neural network infrastructure with dendritic routing, constitutional morality, Ebbinghaus
working memory, multimodal ONNX encoders, and batched training. It is header-only C++17
with no mandatory external dependencies.

The companion model/demo repo that uses this framework is **DendriteNet**
(see `dendritenet-core/` submodule inside that repo).

## Build (framework tests)
```
# Standard build (AVX2 + OpenMP)
g++ -O3 -std=c++17 -Iinclude -ffast-math -funroll-loops -mavx2 -fopenmp \
    -o dendrite3d examples/test_harness.cpp

# Minimal (no SIMD)
g++ -O3 -std=c++17 -Iinclude -ffast-math -funroll-loops -o dendrite3d_min examples/test_harness.cpp
```

## Test
```
bash tests/build_and_run.sh
```
Runs 6 suites (tensor, morality, context, data_cleaner, training_stability, checkpoint) — 158 tests.

## Architecture files
- include/tensor.hpp — Tensor ops, NaN guards, AVX2 SIMD (dot/matvec_transposed/matmul)
- include/layer.hpp — DenseLayer + MiniNetwork + Adam; forward_batch/backward_batch; serialize/deserialize
- include/checkpoint.hpp — Binary checkpoint I/O (format "DNRT0001"); CheckpointWriter + CheckpointReader
- include/onnx_encoder.hpp — OrtEncoder RAII wrapper (gated: -DDENDRITE_ONNX); shared Env singleton, NaN-guarded run()
- include/gpu_backend.hpp — OpenCL GPU kernels + GPUContext singleton (gated: -DDENDRITE_OPENCL)
- include/conductor.hpp — Conductor + SubConductor + FusionStrategy
- include/task_context.hpp — Working memory with Ebbinghaus fading
- include/morality.hpp — Immutable guardrails with audit trail
- include/data_cleaner.hpp — Auto cleaning with correlation/differentiation
- include/modality.hpp — Image (MobileNetV2) + Audio (YAMNet) modules via ONNX
- include/text_preprocessor.hpp — Stop-word compression
- include/dendrite3d.hpp — Full 3D tree integrating everything; save_checkpoint/load_checkpoint; train_minibatch
- include/model_config.hpp — ModelConfig + TrainingConfig structs for external model definition
- include/dataset.hpp — DatasetProvider abstract interface for pluggable datasets
- config/morality.cfg — Human-editable morality rules (immutable to training)
- tools/export_encoders.py — Export MobileNetV2 (1280-dim) + audio CNN (1024-dim) to ONNX

## Code style
- C++17 header-only, no external deps (ONNX Runtime optional for multimodal)
- std::mt19937 for RNG, Adam everywhere, NaN guards at every handoff
- Gradient clipping: 1.0 backprop, 0.5 policy gradient

## Using the framework in your project
Add this repo as a git submodule at `dendritenet-core/` in your project, then build with:
```
g++ -O3 -std=c++17 -Idendritenet-core/include -ffast-math -funroll-loops \
    -mavx2 -fopenmp -o myapp examples/main.cpp
```

To define your own model, subclass `DatasetProvider` (include/dataset.hpp) and populate
`ModelConfig` + `TrainingConfig` (include/model_config.hpp), then call:
```cpp
DendriteNet3D net(input_dim, output_dim);
net.learning_rate = train_cfg.learning_rate;
net.build_from_config(model_cfg);
net.init_v3("path/to/morality.cfg");
```

## Known issues (resolved)
1. ✅ Training instability after ~15 epochs — fixed: LR warmup + cosine decay
2. ✅ Data cleaner anomaly rate too high (72%) — fixed: entropy gate tuned
3. ✅ Context items all show rel=1.0 — fixed: weighted reinforce + top-2 filter
4. ✅ Image/audio encoders — real ONNX integration added
5. ✅ Strategy selector — fixed: Gumbel-softmax with tau annealing
6. ✅ Sub-branch child_router doesn't backprop — fixed: REINFORCE policy gradient
7. GPU: Tier 1 (AVX2+OpenMP) implemented; Tier 2 (OpenCL) infrastructure ready

## Extensions

**Rules** (`.claude/rules/`): `cpp-style` · `nn-conventions` · `headers` · `morality`

**Skills**: `/build` · `/debug-build` · `/benchmark` · `/test` · `/fix-issue [1-7]` · `/review-header [file]` · `/onnx-integrate [image|audio|both]` · `/profile`

**Agents**: `cpp-expert` · `build-validator` · `nn-architect` · `issue-hunter`

**MCP** (`.mcp.json`): `github` · `sequential-thinking`

**GPU tiers**: Tier 1 always-on with `-mavx2 -fopenmp` · Tier 2 optional with `-DDENDRITE_OPENCL -lOpenCL`
