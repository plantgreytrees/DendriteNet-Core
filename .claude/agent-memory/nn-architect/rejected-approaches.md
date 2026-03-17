---
name: rejected-approaches
description: Approaches considered and ruled out, with reasons — prevents re-proposing bad ideas
type: project
---

# Rejected Approaches

*Append new entries at the top. Format: `## YYYY-MM-DD — [topic]: [what was rejected]`*

## 2026-03-17 — MI collapse fix: adjusting only NMDA threshold
**Rejected because:** Raising nmda_threshold alone does not fix the heat_target floor problem. If heat_target non-best = 0.2 and NMDA threshold is moved to 0.4, the supervised gradient still pushes non-best branches toward 0.2, which now maps to near-zero NMDA output — causing branches 0 and 3 to die (no visits at test time). Adjusting one constant without the others makes the problem asymmetric rather than fixing it.

<!-- Rejected approaches go here as they are discovered -->
