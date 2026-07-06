"""Parse Claude Code NDJSON transcripts (~/.claude/projects/<slug>/<uuid>.jsonl)
into a normalized Session."""

import json
import re
from pathlib import Path

from .types import Session, Turn

TOOL_RESULT_MAX = 500
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_HARNESS_RE = re.compile(
    r"^\s*<(command-name|command-message|command-args|local-command-stdout|"
    r"local-command-stderr|user-memory-input|task-notification|system-warning|"
    r"session-(start|end)-hook)>", re.S
)


def _clean_user_text(text: str) -> str:
    """Strip injected system-reminder blocks; drop pure harness-XML rows."""
    text = _REMINDER_RE.sub("", text).strip()
    if _HARNESS_RE.match(text):
        return ""
    return text


def _clean_tool_text(text: str) -> str:
    text = _BASE64_RE.sub("[binary]", text)
    if len(text) > TOOL_RESULT_MAX:
        text = text[:TOOL_RESULT_MAX] + " …[truncated]"
    return text


def _block_text(block) -> str:
    """Extract plain text from a content block or nested tool_result content."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        if block.get("type") == "text":
            return block.get("text", "")
        if isinstance(block.get("content"), str):
            return block["content"]
        if isinstance(block.get("content"), list):
            return "\n".join(_block_text(b) for b in block["content"])
    return ""


def parse_claude_transcript(path: str | Path, session_id: str | None = None) -> Session:
    path = Path(path)
    sess = Session(source="claude", session_id=session_id or path.stem)
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rtype = row.get("type")
        if rtype not in ("user", "assistant"):
            continue
        if row.get("isSidechain"):
            continue
        msg = row.get("message") or {}
        ts = row.get("timestamp")
        sess.cwd = row.get("cwd") or sess.cwd
        sess.git_branch = row.get("gitBranch") or sess.git_branch
        if sess.started_at is None and ts:
            sess.started_at = ts
        if ts:
            sess.ended_at = ts

        if rtype == "user":
            content = msg.get("content")
            if isinstance(content, str):
                text = _clean_user_text(content)
                if text:
                    sess.turns.append(Turn("user", text, ts))
            elif isinstance(content, list):
                texts, tool_results = [], []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tool_results.append(block)
                text = _clean_user_text("\n".join(t for t in texts if t.strip()))
                if text:
                    sess.turns.append(Turn("user", text, ts))
                for tr in tool_results:
                    body = _clean_tool_text(_block_text(tr).strip())
                    if body:
                        sess.turns.append(
                            Turn("tool", body, ts, is_error=bool(tr.get("is_error")))
                        )
        else:  # assistant
            model = msg.get("model")
            if model and not str(model).startswith("<"):
                sess.model = model
            content = msg.get("content")
            if isinstance(content, list):
                texts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        texts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        args = json.dumps(block.get("input", {}))[:200]
                        texts.append(f"[tool: {block.get('name')} {args}]")
                    # thinking blocks intentionally dropped
                text = "\n".join(t for t in texts if t.strip()).strip()
                if text:
                    sess.turns.append(Turn("assistant", text, ts, model=model))
            elif isinstance(content, str) and content.strip():
                sess.turns.append(Turn("assistant", content.strip(), ts, model=model))
    return sess
