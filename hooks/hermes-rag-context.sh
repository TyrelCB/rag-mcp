#!/usr/bin/env bash
# Hermes pre_llm_call shell hook: inject relevant prior context from rag-mcp.
# Stdin: {hook_event_name, session_id, cwd, extra:{user_message, is_first_turn, model, platform, ...}}
# Stdout: {"context": "..."} to prepend to the user message; nothing = no injection.
# Fail-open: any failure prints nothing and exits 0.
set -uo pipefail

INPUT=$(cat)
PAYLOAD=$(jq -c '{q: (.extra.user_message // ""), session_id: .session_id, cwd: .cwd}' <<<"$INPUT" 2>/dev/null) || exit 0

BODY=$(curl -s --max-time 2 -X POST http://127.0.0.1:8004/api/context \
  -H 'content-type: application/json' -d "$PAYLOAD" 2>/dev/null) || exit 0

[ -n "$BODY" ] && jq -n --arg ctx "$BODY" '{context: $ctx}'
exit 0
