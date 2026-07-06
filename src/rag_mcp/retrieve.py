"""Hybrid retrieval: sqlite-vec KNN + FTS5 BM25, RRF fusion, boosts,
per-session injection dedupe, token-capped context packing."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import db as dbmod
from .config import Config
from .embed import embed_query

RRF_K = 60
POOL = 20
KIND_WEIGHT = {
    "summary": 1.10, "fact": 1.10, "decision": 1.10, "error_fix": 1.10,
    "assistant_answer": 1.0, "code": 1.0, "user_prompt": 0.9, "manual": 1.15,
}


def fts_sanitize(query: str) -> str:
    """Quote each term so raw user text can't break FTS5 syntax."""
    terms = re.findall(r"[A-Za-z0-9_./-]{2,}", query)[:12]
    return " OR ".join(f'"{t}"' for t in terms)


def _age_days(ts: str | None) -> float:
    if not ts:
        return 365.0
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    except ValueError:
        return 365.0


def hybrid_search(
    conn: sqlite3.Connection,
    query_embedding: list[float] | None,
    query_text: str,
    *,
    k: int = 8,
    source: str | None = None,
    project: str | None = None,
    cwd: str | None = None,
    exclude_session_id: str | None = None,
    skip_injected_for: str | None = None,
) -> list[dict]:
    ranks: dict[int, float] = {}

    if query_embedding is not None:
        rows = conn.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (dbmod.serialize_embedding(query_embedding), POOL),
        ).fetchall()
        for rank, r in enumerate(rows):
            ranks[r["chunk_id"]] = ranks.get(r["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)

    fts_q = fts_sanitize(query_text)
    if fts_q:
        try:
            rows = conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (fts_q, POOL),
            ).fetchall()
            for rank, r in enumerate(rows):
                ranks[r["rowid"]] = ranks.get(r["rowid"], 0.0) + 1.0 / (RRF_K + rank + 1)
        except sqlite3.OperationalError:
            pass  # sanitized query still upset FTS5; vector side carries it

    if not ranks:
        return []

    injected: set[int] = set()
    if skip_injected_for:
        injected = {
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM injected WHERE session_id=?", (skip_injected_for,)
            )
        }

    qmarks = ",".join("?" * len(ranks))
    rows = conn.execute(
        f"SELECT c.id, c.kind, c.text, c.ts, d.source, d.session_id, d.project, d.model "
        f"FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id IN ({qmarks})",
        list(ranks.keys()),
    ).fetchall()

    cwd_project = Path(cwd).name if cwd else None
    scored: list[tuple[float, dict]] = []
    seen_text: set[str] = set()
    for r in rows:
        if r["id"] in injected:
            continue
        if exclude_session_id and r["session_id"] == exclude_session_id:
            continue
        if source and r["source"] != source:
            continue
        if project and r["project"] != project:
            continue
        tkey = hashlib.sha256(r["text"][:64].lower().encode()).hexdigest()
        if tkey in seen_text:
            continue
        seen_text.add(tkey)

        score = ranks[r["id"]] * KIND_WEIGHT.get(r["kind"], 1.0)
        if cwd_project and r["project"] == cwd_project:
            score *= 1.15
        recency = 1.0 / (1.0 + _age_days(r["ts"]) / 45.0)
        score = 0.7 * score + 0.3 * score * recency
        scored.append((score, dict(r)))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:k]]


def pack_context(results: list[dict], token_budget: int) -> tuple[str, list[int]]:
    """Render provenance-tagged lines within ~token_budget (chars/4)."""
    char_budget = token_budget * 4
    lines = ["Relevant prior context (rag-mcp):"]
    used_ids: list[int] = []
    total = len(lines[0])
    for r in results:
        date = (r.get("ts") or "")[:10]
        proj = r.get("project") or "?"
        line = f"[{r['source']} · {proj} · {date} · {r['kind']}] {r['text']}"
        if total + len(line) > char_budget:
            remaining = char_budget - total
            if remaining > 200:
                line = line[: remaining - 12] + " …[truncated]"
            else:
                break
        lines.append(line)
        used_ids.append(r["id"])
        total += len(line)
    if not used_ids:
        return "", []
    return "\n".join(lines), used_ids


async def context_for_prompt(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    q: str,
    session_id: str | None,
    cwd: str | None,
    k: int = 8,
) -> str:
    q = (q or "").strip()
    if len(q) < 12 or q.startswith("/"):
        return ""
    try:
        emb = await embed_query(cfg, q)
    except Exception:
        emb = None  # embedding service down: FTS-only, still useful
    results = hybrid_search(
        conn, emb, q,
        k=k, cwd=cwd,
        exclude_session_id=session_id,
        skip_injected_for=session_id,
    )
    block, used_ids = pack_context(results, cfg.context_token_budget)
    if session_id:
        with conn:
            if used_ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO injected (session_id, chunk_id) VALUES (?,?)",
                    [(session_id, cid) for cid in used_ids],
                )
            conn.execute(
                "INSERT INTO prompt_log (session_id, cwd, prompt, injected_chunk_ids) "
                "VALUES (?,?,?,?)",
                (session_id, cwd, q[:2000], json.dumps(used_ids)),
            )
    dbmod.prune_injected(conn)
    return block
