---
name: build
description: Compile DendriteNet in release mode. Use to check for build errors after editing source files, or to produce the dendrite3d binary.
argument-hint: ""
user-invocable: true
allowed-tools: Bash
---

Build DendriteNet 3D in release mode from the project root.

Run:
```
g++ -O3 -std=c++17 -Iinclude -ffast-math -funroll-loops -o dendrite3d examples/main.cpp 2>&1
```

Then report:
- **Pass** or **Fail** clearly at the top
- All warnings and errors with their `file:line` references
- If passed: binary size via `ls -lh dendrite3d` (or `wc -c dendrite3d` on Windows)
- If failed: group errors by header file and suggest the most likely root cause for each group

Do not modify any source files during this skill — only build and report.
