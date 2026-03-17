---
name: benchmark
description: Build DendriteNet in release mode, run the demo, and extract key metrics (accuracy, loss, timing). Use to measure the effect of a change or to get a baseline snapshot.
argument-hint: ""
user-invocable: true
allowed-tools: Bash
---

Build and run DendriteNet, then extract key metrics.

Step 1 — build release:
```
g++ -O3 -std=c++17 -Iinclude -ffast-math -funroll-loops -o dendrite3d examples/main.cpp 2>&1
```
Stop and report build errors if compilation fails.

Step 2 — run and capture full output:
```
./dendrite3d 2>&1
```

Step 3 — from the output, extract and present a structured report:

| Metric | Value |
|--------|-------|
| Final validation accuracy | ... |
| Per-class accuracy (all 6) | ... |
| Final training loss | ... |
| Epochs completed | ... |
| Morality blocks fired | ... |
| Anomalies flagged by data cleaner | ... |
| Context items (avg relevance) | ... |
| Wall-clock runtime | ... |

Compare against baseline if one is stored. Highlight any regression (>2% accuracy drop or >10% runtime increase).

Do not modify source files.
