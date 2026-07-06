"""Select and build chunks from a normalized session + distillation output.
All text passes through the scrubber."""

import hashlib
import re

from ..scrub import scrub_text
from .types import Session

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)
CODE_MIN_LINES = 5
CODE_SPLIT_CHARS = 1500
USER_PROMPT_MIN_CHARS = 40


def _dedupe_key(text: str) -> str:
    return hashlib.sha256(text[:64].lower().encode()).hexdigest()


def build_chunks(sess: Session, distilled: dict) -> list[dict]:
    chunks: list[dict] = []
    seen: set[str] = set()

    def add(kind: str, text: str, ts: str | None = None, meta: dict | None = None):
        text = scrub_text(text.strip())
        if not text:
            return
        key = _dedupe_key(text)
        if key in seen:
            return
        seen.add(key)
        chunks.append({"kind": kind, "text": text, "ts": ts or sess.ended_at, "meta": meta})

    if distilled.get("summary"):
        add("summary", distilled["summary"])
    for fact in distilled.get("facts", []):
        add("fact", fact)
    for fix in distilled.get("error_fixes", []):
        add("error_fix", fix)

    for turn in sess.turns:
        if turn.role == "user" and len(turn.text) >= USER_PROMPT_MIN_CHARS:
            add("user_prompt", turn.text[:2000], turn.ts)

    # Final substantive assistant answer.
    for turn in reversed(sess.turns):
        if turn.role == "assistant":
            text = re.sub(r"\[tool: [^\]]*\]", "", turn.text).strip()
            if len(text) > 80:
                add("assistant_answer", text[:4000], turn.ts)
            break

    # Fenced code blocks from assistant turns.
    for turn in sess.turns:
        if turn.role != "assistant":
            continue
        for m in _CODE_FENCE_RE.finditer(turn.text):
            code = m.group(1).strip()
            if code.count("\n") + 1 <= CODE_MIN_LINES:
                continue
            for i in range(0, len(code), CODE_SPLIT_CHARS):
                add("code", code[i : i + CODE_SPLIT_CHARS], turn.ts)

    return chunks
