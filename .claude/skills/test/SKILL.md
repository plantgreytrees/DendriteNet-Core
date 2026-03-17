---
name: test
description: Build and run all DendriteNet unit tests (tensor, morality, task_context, data_cleaner). Use after any code change to verify correctness before committing.
argument-hint: "[tensor|morality|context|data_cleaner] (default: all)"
user-invocable: true
allowed-tools: Bash
---

Run the DendriteNet unit test suite.

```
bash tests/build_and_run.sh $ARGUMENTS 2>&1
```

The runner compiles each test suite independently with ASAN+UBSAN (if available) and reports pass/fail per assertion.

Report:
- Overall result: **PASS** or **FAIL**
- Any assertion failures with their `file:line` and the condition that failed
- Any compilation errors with precise diagnostics
- If all suites pass: confirm assertion count

If tests fail after a code change, identify which assertion broke and what the fix should be. Tests must all pass before any commit.
