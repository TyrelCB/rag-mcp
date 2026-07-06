#!/usr/bin/env bash
# Claude Code SessionEnd hook: enqueue this session for RAG ingestion.
# Fail-open: never block session teardown.
set -uo pipefail

INPUT=$(cat)
PAYLOAD=$(jq -c '{source: "claude", session_id: .session_id, transcript_path: .transcript_path, reason: (.reason // null)}' <<<"$INPUT" 2>/dev/null) || exit 0

curl -s --max-time 3 -X POST http://127.0.0.1:8004/api/ingest \
  -H 'content-type: application/json' -d "$PAYLOAD" >/dev/null 2>&1 || true
exit 0
