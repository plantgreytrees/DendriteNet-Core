---
name: debug-build
description: Compile DendriteNet with AddressSanitizer and debug symbols, then run it. Use when hunting crashes, memory errors, stack overflows, or undefined behaviour.
argument-hint: ""
user-invocable: true
allowed-tools: Bash
---

Build DendriteNet in debug mode with AddressSanitizer.

Step 1 — compile:
```
g++ -g -std=c++17 -Iinclude -fsanitize=address -fsanitize=undefined -o dendrite3d_dbg examples/main.cpp 2>&1
```

Step 2 — if build succeeded, run and capture output (first 150 lines):
```
./dendrite3d_dbg 2>&1 | head -150
```

Report:
- **Build**: Pass or Fail with all compiler errors/warnings
- **Runtime**: any ASAN or UBSAN reports (highlight with ⚠), or "clean run" if none
- **Root cause**: for each ASAN/UBSAN finding, identify the likely source location and what kind of error it is (heap-use-after-free, stack-buffer-overflow, etc.)

On Windows/MinGW, ASAN may not be available — fall back to `-fsanitize=undefined` only and note the limitation.
