#!/usr/bin/env bash
# Hermes on_session_end shell hook: enqueue the finished session for RAG ingestion.
# Stdin: {hook_event_name, tool_name, tool_input, session_id, cwd, extra:{completed, interrupted, model, platform, ...}}
# Fail-open: never block hermes.
set -uo pipefail

INPUT=$(cat)
SESSION_ID=$(jq -r '.session_id // empty' <<<"$INPUT" 2>/dev/null) || exit 0
[ -n "$SESSION_ID" ] || exit 0

TRANSCRIPT="$HOME/.hermes/sessions/${SESSION_ID}.jsonl"
[ -f "$TRANSCRIPT" ] || TRANSCRIPT="$HOME/.hermes/sessions/${SESSION_ID}.json"
[ -f "$TRANSCRIPT" ] || exit 0

PAYLOAD=$(jq -c --arg tp "$TRANSCRIPT" \
  '{source: "hermes", session_id: .session_id, transcript_path: $tp, extra: (.extra // {})}' \
  <<<"$INPUT" 2>/dev/null) || exit 0

curl -s --max-time 3 -X POST http://127.0.0.1:8004/api/ingest \
  -H 'content-type: application/json' -d "$PAYLOAD" >/dev/null 2>&1 || true
exit 0
