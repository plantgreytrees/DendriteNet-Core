---
name: onnx-integrate
description: Plan and implement real ONNX Runtime integration for the image and audio encoder modules. Use when working on issue #4 (simulated encoders) or adding ONNX-backed inference.
argument-hint: "[image|audio|both]"
user-invocable: true
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

Plan and implement ONNX Runtime integration for: **$ARGUMENTS** (default: both)

## Phase 1 — Audit current state
Read `include/modality.hpp` and identify:
- Where simulated encoder stubs are
- What interface they expose (input tensor shape, output embedding dim)
- What ONNX Runtime API calls are needed to replace them

## Phase 2 — Design the integration
Produce a concrete plan covering:
- ONNX Runtime C++ API calls needed (`Ort::Session`, `Ort::RunOptions`, etc.)
- Model file paths and how they'll be configured (env var? constructor arg? cfg file?)
- Compile-time gate: `#ifdef DENDRITE_ONNX` so the project still builds without ORT
- Fallback behaviour when no model file is present (return zero embedding, log warning)
- Memory ownership: who owns the `Ort::Env` and `Ort::Session`?

## Phase 3 — Implement
- Replace the simulated stub(s) with real ONNX Runtime calls
- Add `#ifdef DENDRITE_ONNX` guards
- Add NaN guards on all encoder outputs
- Update `examples/main.cpp` with a usage comment showing how to enable ONNX

## Phase 4 — Build check
```
g++ -O3 -std=c++17 -Iinclude -ffast-math -funroll-loops -o dendrite3d examples/main.cpp 2>&1
```
(Without ORT flags — confirm it still compiles in stub mode)

## ONNX Runtime reference
- C++ API header: `<onnxruntime/core/session/onnxruntime_cxx_api.h>`
- Link flags: `-lonnxruntime`
- MobileNetV2 input: `[1, 3, 224, 224]` float32 NCHW
- YAMNet input: `[1, 15600]` float32 (1 second at 16kHz)
