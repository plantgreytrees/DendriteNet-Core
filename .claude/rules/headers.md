---
paths:
  - "include/**/*.hpp"
---

# Header File Rules

- Must be fully self-contained and compilable in isolation (no missing includes)
- `#pragma once` at the very top, before any other content
- All template implementations must live in the same `.hpp` — no `.tpp` split files
- Avoid `using namespace std;` at file scope; qualify explicitly (`std::vector`, etc.)
- Every function that touches a tensor must include a NaN guard — no exceptions
- Public API types must have a brief doc-comment (one line minimum)
- When adding a new feature module, register it in `include/dendrite3d.hpp`
- ONNX-dependent code must be gated behind `#ifdef DENDRITE_ONNX` so the project compiles without ONNX Runtime installed
- Do not `#include` Windows or POSIX platform headers directly — wrap in a compatibility shim if needed
