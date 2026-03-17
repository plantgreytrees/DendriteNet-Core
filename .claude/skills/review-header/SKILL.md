---
name: review-header
description: Deep review of a DendriteNet header file for correctness, safety, and conventions. Invoke as /review-header [filename], e.g. /review-header tensor.hpp.
argument-hint: "[header filename, e.g. tensor.hpp]"
user-invocable: true
context: fork
agent: Explore
allowed-tools: Read, Grep, Glob
---

Review the header file: **$ARGUMENTS**

Read the file from `include/$ARGUMENTS` (or the path as given).

Evaluate and report on each of the following dimensions:

### 1. NaN Guards
- Every tensor operation that produces a float must guard against NaN/Inf
- Flag any missing guards with the exact line numbers

### 2. Gradient Clipping
- Backprop gradients clipped to [-1.0, 1.0]?
- Policy gradients clipped to [-0.5, 0.5]?

### 3. Memory Safety
- No raw `new`/`delete` (unless wrapped in RAII)
- No dangling references or pointer aliasing issues
- No signed/unsigned mismatch warnings

### 4. C++17 Correctness
- `#pragma once` present
- No `using namespace std` at file scope
- Templates fully defined in the header

### 5. API Clarity
- Public functions have doc-comments
- Return types are `[[nodiscard]]` where appropriate
- Function signatures are `const`-correct

### 6. Integration
- Header properly integrates with the rest of DendriteNet (check what it exports vs what `dendrite3d.hpp` expects)

### Output format
- **PASS** / **WARN** / **FAIL** per dimension
- Line-number references for every finding
- Prioritised fix list (P1 = must fix, P2 = should fix, P3 = nice to have)
