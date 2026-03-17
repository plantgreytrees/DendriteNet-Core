---
paths:
  - "include/morality.hpp"
  - "config/morality.cfg"
---

# Morality System Rules

**These files are safety-critical. Apply extra care.**

- Hard-block rules in `config/morality.cfg` must NEVER be weakened or removed without an explicit human decision — not by refactoring, not by "cleaning up"
- The audit trail in `morality.hpp` must always append, never truncate or overwrite
- The confidence gate threshold is human-configurable; do not hardcode a value that bypasses it
- Soft-redirect logic must always produce a valid alternative output — it may not silently drop the request
- `morality.cfg` is intentionally human-readable; keep it that way (no binary formats, no compression)
- Any change to the morality module must preserve backward compatibility with existing `.cfg` files
- When in doubt about a morality change, propose it and wait for explicit user confirmation before editing
