---
name: fix-issue
description: Investigate and fix one of the 7 known issues listed in CLAUDE.md. Invoke as /fix-issue [issue-number], e.g. /fix-issue 1.
argument-hint: "[issue-number 1-7]"
user-invocable: true
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

Fix known issue **#$ARGUMENTS** from the DendriteNet known issues list:

1. Training instability after ~15 epochs — needs LR warmup
2. Data cleaner anomaly rate too high (72%) — tune heat_threshold
3. Context items all show rel=1.0 because reinforcement is too aggressive
4. Image/audio modules use simulated encoders — need real ONNX integration
5. Strategy selector could use Gumbel-softmax
6. Sub-branch child_router doesn't backprop
7. No GPU support yet

## Workflow

1. **Explore** — read all relevant header files to understand the current implementation
2. **Diagnose** — identify the exact root cause (not just the symptom)
3. **Plan** — write a brief plan (2-5 bullet points) before touching any code
4. **Implement** — make the minimal change that fixes the issue; do not refactor unrelated code
5. **Verify** — build (`g++ -O3 -std=c++17 -Iinclude -ffast-math -funroll-loops -o dendrite3d examples/main.cpp`) and run (`./dendrite3d`) to confirm the fix works
6. **Report** — summarise what changed and why

Respect all rules in `.claude/rules/`. NaN guards, gradient clipping, and morality constraints must never be weakened as a side effect of the fix.
