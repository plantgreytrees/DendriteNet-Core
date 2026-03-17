#!/bin/bash
# PreCompact hook — fires before Claude auto-compacts the context window.
# Appends a compaction marker to the issue-hunter investigation log so
# that root-cause findings are not lost when context is compressed.

PROJ="${CLAUDE_PROJECT_DIR:-$(pwd)}"
LOG="$PROJ/.claude/agent-memory/issue-hunter/investigation-log.md"
DATE=$(date +%Y-%m-%d\ %H:%M)

# Only write if the log file exists (issue-hunter has been used this session)
if [ -f "$LOG" ]; then
    printf "\n## %s — auto-compaction\nContext window compacted. Prior findings preserved above. Resume with: \`/fix-issue [N]\` or read known-issues.md.\n" "$DATE" >> "$LOG"
fi

cat << 'EOF'
{
  "continue": true,
  "systemMessage": "Context compacted. Investigation findings are preserved in .claude/agent-memory/issue-hunter/. Use /fix-issue [N] to continue where you left off."
}
EOF
