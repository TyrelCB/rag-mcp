"""MCP tools. Lean schemas — these load into every client session."""

import hashlib

from . import db as dbmod
from .embed import embed_query, embed_texts
from .retrieve import hybrid_search
from .scrub import scrub_text


def register_tools(mcp, state):
    @mcp.tool
    async def rag_search(query: str, k: int = 6, source: str | None = None, project: str | None = None) -> str:
        """Search past Claude Code and Hermes session memory: summaries, decisions, error fixes, code.

        Args:
            query: What to look for.
            k: Max results (default 6).
            source: Filter: claude, hermes, or manual.
            project: Filter by project name (directory basename).
        """
        try:
            emb = await embed_query(state.cfg, query)
        except Exception:
            emb = None
        results = hybrid_search(state.conn, emb, query, k=k, source=source, project=project)
        if not results:
            return "No matches."
        lines = []
        for r in results:
            date = (r.get("ts") or "")[:10]
            lines.append(f"[{r['source']} · {r.get('project') or '?'} · {date} · {r['kind']}] {r['text']}")
        return "\n\n".join(lines)

    @mcp.tool
    async def rag_ingest_text(text: str, tags: str | None = None) -> str:
        """Store a note or fact directly into RAG memory.

        Args:
            text: The note to remember.
            tags: Optional comma-separated tags.
        """
        text = scrub_text(text.strip())
        if not text:
            return "Nothing to store."
        note_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        try:
            embeddings = await embed_texts(state.cfg, [text])
        except Exception:
            embeddings = []
        dbmod.upsert_document(
            state.conn,
            source="manual", session_id=f"note-{note_id}", project=None, cwd=None,
            git_branch=None, model=None, started_at=None, ended_at=None,
            summary=text[:200], content_hash=note_id, status="ok",
            chunks=[{"kind": "manual", "text": text, "ts": None,
                     "meta": {"tags": tags} if tags else None}],
            embeddings=embeddings,
        )
        return f"Stored note {note_id}."

    @mcp.tool
    async def rag_status() -> dict:
        """Report RAG store stats: document/chunk counts, queue depth, last ingest."""
        snap = dbmod.status_snapshot(state.conn)
        snap["queue_depth"] = state.worker.depth if state.worker else 0
        return snap
