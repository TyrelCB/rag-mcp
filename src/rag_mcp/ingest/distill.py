"""Distill a session into summary / facts / error-fix pairs via the local
llama.cpp OpenAI-compatible server. Fail-soft: any failure returns partial or
empty results and ingestion continues with raw chunks."""

import json
import re

import httpx

from ..config import Config
from .types import Session

_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def build_digest(sess: Session, max_chars: int) -> str:
    """Compact turn-by-turn digest, head+tail if over budget."""
    lines = []
    for t in sess.turns:
        prefix = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}[t.role]
        body = t.text if t.role != "tool" else t.text[:300]
        lines.append(f"{prefix}: {body}")
    digest = "\n".join(lines)
    if len(digest) > max_chars:
        half = max_chars // 2
        digest = digest[:half] + "\n…[middle omitted]…\n" + digest[-half:]
    return digest


def _extract_json(text: str):
    text = re.sub(r"```(?:json)?|```", "", text)
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def _chat(client: httpx.AsyncClient, cfg: Config, model: str, system: str, user: str) -> str:
    r = await client.post(
        f"{cfg.distill_url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def resolve_model(client: httpx.AsyncClient, cfg: Config) -> str:
    if cfg.distill_model:
        return cfg.distill_model
    r = await client.get(f"{cfg.distill_url}/models")
    r.raise_for_status()
    return r.json()["data"][0]["id"]


async def distill(cfg: Config, sess: Session) -> dict:
    """Returns {summary: str|None, facts: [str], error_fixes: [str]}."""
    out = {"summary": None, "facts": [], "error_fixes": []}
    digest = build_digest(sess, cfg.distill_max_chars)
    sysmsg = (
        "You analyze a transcript of a coding-agent session. Answer ONLY with what is "
        "asked, no preamble. Be specific: name files, commands, projects, versions."
    )
    try:
        async with httpx.AsyncClient(timeout=cfg.distill_timeout) as client:
            model = await resolve_model(client, cfg)

            try:
                out["summary"] = (
                    await _chat(
                        client, cfg, model, sysmsg,
                        "Summarize this session in at most 150 words: what was the goal, "
                        "what was done, and the outcome.\n\n" + digest,
                    )
                ).strip() or None
            except Exception:
                pass

            try:
                facts = _extract_json(
                    await _chat(
                        client, cfg, model, sysmsg,
                        'List durable facts or decisions a future agent should know '
                        '(configuration choices, conventions, gotchas, environment facts). '
                        'Output a JSON array of strings, [] if none.\n\n' + digest,
                    )
                )
                if isinstance(facts, list):
                    out["facts"] = [str(f).strip() for f in facts if str(f).strip()][:15]
            except Exception:
                pass

            try:
                fixes = _extract_json(
                    await _chat(
                        client, cfg, model, sysmsg,
                        'List error->fix pairs encountered (an error/failure and how it was '
                        'resolved). Output a JSON array of strings formatted as '
                        '"ERROR: ... FIX: ...", [] if none.\n\n' + digest,
                    )
                )
                if isinstance(fixes, list):
                    out["error_fixes"] = [str(f).strip() for f in fixes if str(f).strip()][:10]
            except Exception:
                pass
    except Exception:
        pass
    return out
