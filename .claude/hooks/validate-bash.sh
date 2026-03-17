#!/bin/bash
# PreToolUse hook for Bash commands.
# Blocks demonstrably destructive patterns; allows everything else.

TMPFILE=$(mktemp /tmp/dn_hook_XXXXXX 2>/dev/null || mktemp)
cat > "$TMPFILE"

python3 - "$TMPFILE" << 'PYEOF'
import sys, json, re, os

tmpfile = sys.argv[1]
try:
    with open(tmpfile) as f:
        d = json.load(f)
    cmd = d.get('tool_input', {}).get('command', '')
except Exception:
    cmd = ''
finally:
    try:
        os.unlink(tmpfile)
    except Exception:
        pass

# Patterns that must be blocked outright
BLOCKED = [
    (r'rm\s+-[a-zA-Z]*r[a-zA-Z]*f\s+/', 'rm -rf on an absolute path'),
    (r'rm\s+-[a-zA-Z]*f[a-zA-Z]*r\s+/', 'rm -rf on an absolute path'),
    (r'rm\s+-rf\s+\.\s*$', 'rm -rf on current directory'),
    (r'rm\s+-rf\s+\*', 'rm -rf with wildcard'),
    (r':\s*\(\s*\)\s*\{', 'fork bomb pattern'),
    (r'dd\s+if=.*of=/dev/', 'dd writing to device'),
    (r'mkfs\b', 'filesystem format'),
    (r'>\s*/dev/sd[a-z]', 'writing to raw block device'),
    (r'chmod\s+-R\s+777\s+/', 'recursive 777 on root path'),
]

for pattern, reason in BLOCKED:
    if re.search(pattern, cmd, re.IGNORECASE):
        result = {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': (
                    f'Blocked by DendriteNet safety hook: {reason}. '
                    f'Command was: {cmd[:120]!r}. '
                    'If this is intentional, run the command in your terminal directly.'
                )
            }
        }
        print(json.dumps(result))
        sys.exit(0)

# All other commands: allow
print('{"continue": true}')
PYEOF
