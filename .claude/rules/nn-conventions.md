# Neural Network Implementation Conventions

## NaN Guards
Every tensor value handoff must check for NaN/Inf. Standard pattern:
```cpp
if (!std::isfinite(val)) val = 0.0f;
```
Or for vectors:
```cpp
for (auto& v : tensor) if (!std::isfinite(v)) v = 0.0f;
```

## Gradient Clipping
- Backpropagation gradients: clip to `[-1.0, 1.0]`
- Policy gradient updates: clip to `[-0.5, 0.5]`
- Apply clipping **before** the Adam update step, not after

## Adam Optimizer
- β1=0.9, β2=0.999, ε=1e-8 (defaults — don't change without a clear reason)
- Always bias-correct: divide m/v estimates by `(1 - β^t)`
- Learning rate warmup target: linear ramp over first N steps before decay

## Weight Initialisation
- Linear layers: He initialisation (for ReLU activations)
- Attention weights: Xavier/Glorot
- Bias terms: zero-init

## Random Numbers
- `std::mt19937` only — seed once with `std::random_device{}()`
- Never seed with a constant in production code; constants only in deterministic tests

## Accuracy Reporting
- Always report on a held-out validation split, never training data
- Log per-class accuracy for the 6-class synthetic task, not just aggregate

## Morality System
- The morality layer runs **before** any output is returned to the caller
- Hard-block decisions must not be bypassed in any training loop
- Confidence gate threshold must remain user-configurable via `config/morality.cfg`
