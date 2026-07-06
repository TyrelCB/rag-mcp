"""Transform session transcripts into TRL-compatible chat-format SFT JSONL.

Two variants per export:
  train_tools.jsonl  — full trajectories with OpenAI-style tool_calls / tool messages
  train_chat.jsonl   — user/assistant text turns only (tool activity stripped)
Plus val_*.jsonl splits and stats.json.
"""

import hashlib
import json
import random
from pathlib import Path

from ..config import Config
from ..ingest.claude_transcript import _clean_tool_text, _clean_user_text, _REMINDER_RE
from ..ingest.hermes_transcript import parse_hermes_session
from ..ingest.claude_transcript import parse_claude_transcript
from ..scrub import scrub
from .quality import quality_check

SYSTEM_MSG = "You are a capable software engineering assistant with access to tools."
TOOL_RESULT_MAX = 2000
MAX_EXAMPLE_TOKENS = 16000  # ~chars/4; split at user-turn boundaries past this


def _args_dict(raw) -> dict:
    """Chat templates (Qwen et al.) iterate tool-call arguments as a mapping,
    so emit dicts rather than OpenAI-wire JSON strings."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": str(raw)}


def _scrub_count(text: str, stats: dict) -> str:
    out, counts = scrub(text)
    for k, v in counts.items():
        stats["redactions"][k] = stats["redactions"].get(k, 0) + v
    return out


def _claude_messages(path: Path, stats: dict) -> list[dict]:
    """Claude NDJSON -> OpenAI-style messages with structured tool_calls."""
    msgs: list[dict] = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") not in ("user", "assistant") or row.get("isSidechain"):
            continue
        content = (row.get("message") or {}).get("content")

        if row["type"] == "user":
            if isinstance(content, str):
                text = _clean_user_text(content)
                if text:
                    msgs.append({"role": "user", "content": _scrub_count(text, stats)})
            elif isinstance(content, list):
                texts = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        texts.append(b.get("text", ""))
                    elif b.get("type") == "tool_result":
                        body = b.get("content")
                        if isinstance(body, list):
                            body = "\n".join(
                                x.get("text", "") for x in body
                                if isinstance(x, dict) and x.get("type") == "text"
                            )
                        body = _clean_tool_text((body or "").strip())[:TOOL_RESULT_MAX]
                        msgs.append({
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": _scrub_count(body, stats),
                        })
                text = _clean_user_text("\n".join(t for t in texts if t.strip()))
                if text:
                    msgs.append({"role": "user", "content": _scrub_count(text, stats)})
        else:
            if not isinstance(content, list):
                if isinstance(content, str) and content.strip():
                    msgs.append({"role": "assistant", "content": _scrub_count(content, stats)})
                continue
            texts, tool_calls = [], []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": _args_dict(
                                _scrub_count(json.dumps(b.get("input", {})), stats)
                            ),
                        },
                    })
            msg = {"role": "assistant",
                   "content": _scrub_count("\n".join(t for t in texts if t.strip()), stats)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if msg["content"] or tool_calls:
                msgs.append(msg)
    return msgs


def _hermes_messages(path: Path, stats: dict) -> list[dict]:
    """Hermes rows are already OpenAI-shaped; scrub + truncate + drop system."""
    msgs: list[dict] = []
    rows = []
    if path.suffix == ".json":
        rows = json.loads(path.read_text(errors="replace")).get("messages") or []
    else:
        for line in path.open(errors="replace"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for row in rows:
        if not isinstance(row, dict):
            continue
        role, content = row.get("role"), row.get("content")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text")
        content = (content or "").strip()
        if role == "user" and content:
            content = _REMINDER_RE.sub("", content).strip()
            if content:
                msgs.append({"role": "user", "content": _scrub_count(content, stats)})
        elif role == "assistant":
            msg = {"role": "assistant", "content": _scrub_count(content, stats)}
            if row.get("tool_calls"):
                calls = json.loads(_scrub_count(json.dumps(row["tool_calls"]), stats))
                for tc in calls:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(fn, dict):
                        fn["arguments"] = _args_dict(fn.get("arguments"))
                msg["tool_calls"] = calls
            if msg["content"] or msg.get("tool_calls"):
                msgs.append(msg)
        elif role == "tool" and content:
            msgs.append({"role": "tool", "tool_call_id": row.get("tool_call_id", ""),
                         "content": _scrub_count(_clean_tool_text(content)[:TOOL_RESULT_MAX], stats)})
    return msgs


def _split_long(messages: list[dict]) -> list[list[dict]]:
    """Split a long conversation into <=MAX_EXAMPLE_TOKENS segments at user turns."""
    budget = MAX_EXAMPLE_TOKENS * 4
    if sum(len(json.dumps(m)) for m in messages) <= budget:
        return [messages]
    segments, current, size = [], [], 0
    for m in messages:
        mlen = len(json.dumps(m))
        if current and m["role"] == "user" and size + mlen > budget:
            segments.append(current)
            current, size = [], 0
        current.append(m)
        size += mlen
    if current:
        segments.append(current)
    # Drop segments that don't start with a user turn, lack an assistant reply,
    # or still blow the budget (tool loops with no user boundary to split at —
    # they'd only get truncated mid-trajectory at train time).
    return [
        s for s in segments
        if s and s[0]["role"] == "user"
        and any(m["role"] == "assistant" for m in s)
        and sum(len(json.dumps(m)) for m in s) <= 2 * budget
    ]


def _chat_only(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append(m)
        elif m["role"] == "assistant" and m.get("content"):
            out.append({"role": "assistant", "content": m["content"]})
    # collapse consecutive same-role messages
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    while merged and merged[-1]["role"] == "user":
        merged.pop()
    return merged


def run_export(cfg: Config, conn, *, out_dir: Path, source: str = "claude",
               min_turns: int = 3, since: str | None = None,
               include_tools: bool = True, val_frac: float = 0.05) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"considered": 0, "exported": 0, "rejected": {}, "redactions": {},
             "tool_examples": 0, "chat_examples": 0}

    q = "SELECT * FROM documents WHERE source=? AND status='ok' AND transcript_path IS NOT NULL"
    params: list = [source]
    if since:
        q += " AND started_at >= ?"
        params.append(since)
    docs = conn.execute(q, params).fetchall()

    seen_first_prompt: set[str] = set()
    tool_examples: list[dict] = []
    chat_examples: list[dict] = []

    for doc in docs:
        stats["considered"] += 1
        path = Path(doc["transcript_path"])
        if not path.exists():
            stats["rejected"]["missing_file"] = stats["rejected"].get("missing_file", 0) + 1
            continue
        sess = (parse_claude_transcript(path, doc["session_id"]) if doc["source"] == "claude"
                else parse_hermes_session(path, doc["session_id"]))
        sess.model = sess.model or doc["model"]
        ok, reason = quality_check(sess, min_turns=min_turns)
        if not ok:
            key = reason.split(":")[0]
            stats["rejected"][key] = stats["rejected"].get(key, 0) + 1
            continue

        msgs = (_claude_messages(path, stats) if doc["source"] == "claude"
                else _hermes_messages(path, stats))
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        fp = hashlib.sha256(first_user[:200].encode()).hexdigest()
        if fp in seen_first_prompt:
            stats["rejected"]["duplicate"] = stats["rejected"].get("duplicate", 0) + 1
            continue
        seen_first_prompt.add(fp)

        system = [{"role": "system", "content": SYSTEM_MSG}]
        if include_tools:
            for seg in _split_long(msgs):
                tool_examples.append({"messages": system + seg})
        chat = _chat_only(msgs)
        if len(chat) >= 2:
            for seg in _split_long(chat):
                chat_examples.append({"messages": system + seg})
        stats["exported"] += 1

    rng = random.Random(42)

    def _write(examples: list[dict], name: str):
        rng.shuffle(examples)
        n_val = max(1, int(len(examples) * val_frac)) if len(examples) > 10 else 0
        splits = {f"val_{name}.jsonl": examples[:n_val], f"train_{name}.jsonl": examples[n_val:]}
        for fname, rows in splits.items():
            with (out_dir / fname).open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if include_tools:
        _write(tool_examples, "tools")
        stats["tool_examples"] = len(tool_examples)
    _write(chat_examples, "chat")
    stats["chat_examples"] = len(chat_examples)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
