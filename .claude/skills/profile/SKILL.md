---
name: profile
description: Build DendriteNet with profiling, run it, and report where time is spent. Use when investigating performance bottlenecks, or when working on issue #7 (GPU) to establish CPU baseline timing.
argument-hint: "[top-N functions, default 20]"
user-invocable: true
allowed-tools: Bash
---

Profile DendriteNet to find CPU hotspots.

## Step 1 — Build with profiling instrumentation
```
g++ -O2 -std=c++17 -Iinclude -pg -o dendrite3d_prof examples/main.cpp 2>&1
```
Stop and report build errors if compilation fails.

## Step 2 — Run to generate profiling data
```
./dendrite3d_prof 2>&1
```
This produces `gmon.out` in the working directory.

## Step 3 — Analyse with gprof
```
gprof dendrite3d_prof gmon.out 2>&1 | head -80
```

If `gprof` is not available, fall back to timing individual phases with `time`:
```
time ./dendrite3d_prof > /dev/null 2>&1
```

## Step 4 — Report

Present a table of the top $ARGUMENTS (default: 20) hottest functions:

| Rank | % Time | Cumulative % | Function | File |
|------|--------|--------------|----------|------|
| 1    | ...    | ...          | ...      | ...  |

Then answer:
- Which header/module dominates?
- Is time spent in matrix multiply, Adam update, softmax, or elsewhere?
- What is the estimated speedup from moving the top function to GPU?
- Any obvious CPU-side optimisation (SIMD, loop fusion, pre-allocation) worth trying first?

## Cleanup
```
rm -f gmon.out dendrite3d_prof
```
