#!/bin/bash
# DendriteNet test runner
# Usage: bash tests/build_and_run.sh [suite_name]
#   suite_name: tensor | morality | context | data_cleaner (default: all)

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure the MinGW linker has a writable temp directory.
# When running with a clean PATH (e.g. the git-DLL workaround), TEMP/TMP may be
# empty; g++ then falls back to C:\WINDOWS\ which is read-only → "Cannot create
# temporary file" error.  Use the MSYS2 /tmp mount if TEMP is unset.
if [[ -z "${TEMP:-}" ]]; then
    export TEMP="/tmp"
    export TMP="/tmp"
fi

# ---- detect sanitizer support -------------------------------------------
if g++ -fsanitize=address -x c++ /dev/null -o /dev/null 2>/dev/null; then
    SANITIZE="-fsanitize=address -fsanitize=undefined"
else
    echo "Note: ASAN/UBSAN not available on this toolchain — building without sanitizers"
    SANITIZE=""
fi

CFLAGS="-O1 -std=c++17 -Iinclude $SANITIZE -Wall -Wextra -Wno-unused-parameter"
mkdir -p tests/bin

PASS_SUITES=0
FAIL_SUITES=0
FILTER="${1:-all}"

run_suite() {
    local name="$1"
    local src="$2"
    local bin="tests/bin/${name}"

    [[ "$FILTER" != "all" && "$FILTER" != "$name" ]] && return

    echo ""
    echo "━━━ Building ${name} ━━━"
    if g++ $CFLAGS -o "$bin" "$src" 2>&1; then
        echo "━━━ Running ${name} ━━━"
        if "./$bin"; then
            PASS_SUITES=$((PASS_SUITES + 1))
        else
            FAIL_SUITES=$((FAIL_SUITES + 1))
        fi
    else
        echo "BUILD FAILED for ${name}"
        FAIL_SUITES=$((FAIL_SUITES + 1))
    fi
}

build_only() {
    local name="$1"
    local src="$2"
    local bin="tests/bin/${name}_buildcheck"

    [[ "$FILTER" != "all" && "$FILTER" != "$name" ]] && return

    echo ""
    echo "━━━ Build-check ${name} ━━━"
    if g++ $CFLAGS -o "$bin" "$src" 2>&1; then
        echo "  Build OK (not executed — benchmark takes minutes)"
        PASS_SUITES=$((PASS_SUITES + 1))
    else
        echo "BUILD FAILED for ${name}"
        FAIL_SUITES=$((FAIL_SUITES + 1))
    fi
}

run_suite tensor       tests/test_tensor.cpp
run_suite morality     tests/test_morality.cpp
run_suite context      tests/test_context.cpp
run_suite data_cleaner        tests/test_data_cleaner.cpp
run_suite training_stability  tests/test_training_stability.cpp
run_suite checkpoint          tests/test_checkpoint.cpp

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TOTAL=$((PASS_SUITES + FAIL_SUITES))
echo "Suites: ${TOTAL}  |  Passed: ${PASS_SUITES}  |  Failed: ${FAIL_SUITES}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$FAIL_SUITES" -eq 0 ]
