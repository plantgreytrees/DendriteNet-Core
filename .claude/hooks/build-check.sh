#!/bin/bash
# PostToolUse hook for Write and Edit.
# After editing a .hpp or .cpp file, attempts a release build and reports the result.

TMPFILE=$(mktemp /tmp/dn_hook_XXXXXX 2>/dev/null || mktemp)
cat > "$TMPFILE"
PROJ="${CLAUDE_PROJECT_DIR:-$(pwd)}"

python3 - "$TMPFILE" "$PROJ" << 'PYEOF'
import sys, json, subprocess, os

tmpfile, proj = sys.argv[1], sys.argv[2]

try:
    with open(tmpfile) as f:
        d = json.load(f)
    fp = d.get('tool_input', {}).get('file_path', '')
except Exception:
    fp = ''
finally:
    try:
        os.unlink(tmpfile)
    except Exception:
        pass

# Only trigger for C++ source files
if not fp.endswith(('.hpp', '.cpp')):
    print('{"continue": true}')
    sys.exit(0)

fname = os.path.basename(fp)

try:
    r = subprocess.run(
        [
            'g++', '-O3', '-std=c++17', '-Iinclude',
            '-ffast-math', '-funroll-loops',
            '-o', 'dendrite3d', 'examples/main.cpp'
        ],
        cwd=proj,
        capture_output=True,
        text=True,
        timeout=90
    )
    if r.returncode == 0:
        warnings = [l for l in (r.stderr + r.stdout).splitlines() if 'warning:' in l]
        warn_str = f' ({len(warnings)} warning(s))' if warnings else ''
        msg = f'✓ Build passed after editing {fname}{warn_str}'
    else:
        errs = (r.stderr + r.stdout).strip()
        # Limit output length to avoid flooding context
        if len(errs) > 1200:
            errs = errs[:1200] + '\n... (truncated)'
        msg = f'⚠ Build FAILED after editing {fname}:\n{errs}'
except subprocess.TimeoutExpired:
    msg = f'⏱ Build timed out after editing {fname} (90s limit)'
except FileNotFoundError:
    msg = 'Build check skipped: g++ not found in PATH'
except Exception as e:
    msg = f'Build check error: {e}'

print(json.dumps({'continue': True, 'systemMessage': msg}))
PYEOF
