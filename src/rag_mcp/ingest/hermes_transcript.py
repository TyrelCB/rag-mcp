"""Parse Hermes session files (~/.hermes/sessions/<id>.jsonl OpenAI-style chat rows,
or single-object cron .json files) into a normalized Session."""

import json
from pathlib import Path

from .claude_transcript import _clean_tool_text
from .types import Session, Turn


def _row_to_turn(row: dict) -> Turn | None:
    role = row.get("role")
    content = row.get("content")
    if isinstance(content, list):  # multimodal rows: keep text parts only
        content = "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    content = (content or "").strip()
    ts = row.get("timestamp")

    if role == "user":
        return Turn("user", content, ts) if content else None
    if role == "assistant":
        parts = [content] if content else []
        for tc in row.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = str(fn.get("arguments", ""))[:200]
            parts.append(f"[tool: {fn.get('name')} {args}]")
        text = "\n".join(parts).strip()
        return Turn("assistant", text, ts) if text else None
    if role == "tool":
        if not content:
            return None
        is_error = content[:200].lower().startswith(("error", "traceback")) or '"error"' in content[:200]
        return Turn("tool", _clean_tool_text(content), ts, is_error=is_error)
    return None  # system rows (incl. giant cron system_prompt) dropped


def parse_hermes_session(path: str | Path, session_id: str | None = None) -> Session:
    path = Path(path)
    sess = Session(source="hermes", session_id=session_id or path.stem)

    if path.suffix == ".json":  # cron session: one object with a messages list
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        sess.session_id = data.get("session_id") or sess.session_id
        sess.model = data.get("model")
        sess.platform = data.get("platform")
        rows = data.get("messages") or []
    else:
        rows = []
        for line in path.open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for row in rows:
        if not isinstance(row, dict):
            continue
        turn = _row_to_turn(row)
        if turn:
            sess.turns.append(turn)
            if turn.ts:
                if sess.started_at is None:
                    sess.started_at = turn.ts
                sess.ended_at = turn.ts
    return sess
