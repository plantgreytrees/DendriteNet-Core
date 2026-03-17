#!/bin/bash
# Fires on SessionStart (startup). Prints a brief project status briefing to Claude.

cat << 'EOF'
{
  "continue": true,
  "systemMessage": "DendriteNet v0.3.0 — session started.\n\nOpen known issues:\n  #1  Training instability >15 epochs (needs LR warmup)\n  #2  Data cleaner anomaly rate 72% (tune heat_threshold)\n  #3  Context items all rel=1.0 (reinforcement too aggressive)\n  #4  Image/audio use simulated encoders (need real ONNX)\n  #5  Strategy selector: add Gumbel-softmax\n  #6  child_router doesn't backprop\n  #7  No GPU support\n\nAvailable skills: /build  /debug-build  /benchmark  /fix-issue [1-7]  /review-header [file]  /onnx-integrate\nAvailable agents: cpp-expert  build-validator  nn-architect  issue-hunter"
}
EOF
