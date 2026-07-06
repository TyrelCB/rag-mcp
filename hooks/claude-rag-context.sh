#!/usr/bin/env bash
# Claude Code UserPromptSubmit hook: fetch relevant prior context from rag-mcp.
# Fail-open: if the service is down or slow, print nothing and exit 0.
set -uo pipefail

INPUT=$(cat)
PAYLOAD=$(jq -c '{q: .prompt, session_id: .session_id, cwd: .cwd}' <<<"$INPUT" 2>/dev/null) || exit 0

curl -s --max-time 2 -X POST http://127.0.0.1:8004/api/context \
  -H 'content-type: application/json' -d "$PAYLOAD" 2>/dev/null || true
exit 0
