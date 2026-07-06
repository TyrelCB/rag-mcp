"""Ingestion jobs: shared by the service's background queue consumer and the
CLI backfill path. Single writer — jobs are processed serially."""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from .. import db
from ..config import Config
from ..embed import embed_texts
from .chunker import build_chunks
from .claude_transcript import parse_claude_transcript
from .distill import distill
from .hermes_transcript import parse_hermes_session
from .types import Session

log = logging.getLogger("rag-mcp.ingest")


def validate_transcript_path(cfg: Config, path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    for root in cfg.allowed_transcript_roots():
        try:
            path.relative_to(root.resolve())
            return path
        except ValueError:
            continue
    raise ValueError(f"transcript path outside allowed roots: {path}")


def content_hash(sess: Session) -> str:
    h = hashlib.sha256()
    for t in sess.turns:
        h.update(t.role.encode())
        h.update(t.text.encode())
    return h.hexdigest()


def _project_of(sess: Session) -> str | None:
    if sess.source == "claude":
        return Path(sess.cwd).name if sess.cwd else None
    return sess.platform


async def process_job(cfg: Config, conn, job: dict) -> dict:
    """job: {source, session_id, transcript_path, reason?, extra?}"""
    t0 = time.monotonic()
    source = job["source"]
    session_id = job.get("session_id") or Path(job["transcript_path"]).stem
    try:
        path = validate_transcript_path(cfg, job["transcript_path"])
        if source == "claude":
            sess = parse_claude_transcript(path, session_id)
        elif source == "hermes":
            sess = parse_hermes_session(path, session_id)
            extra = job.get("extra") or {}
            sess.model = extra.get("model") or sess.model
            sess.platform = extra.get("platform") or sess.platform
        else:
            raise ValueError(f"unknown source {source!r}")

        chash = content_hash(sess)
        if db.existing_hash(conn, source, sess.session_id) == chash:
            db.log_ingest(conn, source, sess.session_id, "skipped", "unchanged")
            return {"status": "skipped", "reason": "unchanged"}

        extra = job.get("extra") or {}
        interrupted_empty = extra.get("interrupted") and not any(
            t.role == "assistant" for t in sess.turns
        )
        if (
            len(sess.user_turns) < 2
            or sess.total_chars < cfg.min_session_chars
            or interrupted_empty
        ):
            db.upsert_document(
                conn,
                source=source, session_id=sess.session_id, project=_project_of(sess),
                cwd=sess.cwd, git_branch=sess.git_branch, model=sess.model,
                started_at=sess.started_at, ended_at=sess.ended_at, summary=None,
                content_hash=chash, status="skipped_short", chunks=[], embeddings=[],
                transcript_path=str(path),
            )
            db.log_ingest(conn, source, sess.session_id, "skipped", "too_short")
            return {"status": "skipped", "reason": "too_short"}

        distilled = await distill(cfg, sess)
        chunks = build_chunks(sess, distilled)
        embeddings: list[list[float]] = []
        if chunks:
            try:
                embeddings = await embed_texts(cfg, [c["text"] for c in chunks])
            except Exception as e:  # FTS-only ingest still useful
                log.warning("embedding failed for %s/%s: %s", source, sess.session_id, e)

        _, n = db.upsert_document(
            conn,
            source=source, session_id=sess.session_id, project=_project_of(sess),
            cwd=sess.cwd, git_branch=sess.git_branch, model=sess.model,
            started_at=sess.started_at, ended_at=sess.ended_at,
            summary=distilled.get("summary"), content_hash=chash, status="ok",
            chunks=chunks, embeddings=embeddings, transcript_path=str(path),
        )
        ms = int((time.monotonic() - t0) * 1000)
        db.log_ingest(conn, source, sess.session_id, "done", None, n, ms)
        return {"status": "done", "chunks": n, "duration_ms": ms}
    except Exception as e:
        log.exception("ingest failed for %s/%s", source, session_id)
        try:
            db.log_ingest(conn, source, session_id, "error", str(e)[:500])
        except Exception:
            pass
        return {"status": "error", "error": str(e)}


class IngestWorker:
    def __init__(self, cfg: Config, conn):
        self.cfg = cfg
        self.conn = conn
        self.queue: asyncio.Queue[tuple[int | None, dict]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self):
        # Recover jobs that were queued or in-flight when the service last stopped.
        for row in self.conn.execute("SELECT id, job FROM pending_jobs ORDER BY id"):
            try:
                self.queue.put_nowait((row["id"], json.loads(row["job"])))
            except (json.JSONDecodeError, TypeError):
                with self.conn:
                    self.conn.execute("DELETE FROM pending_jobs WHERE id=?", (row["id"],))
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def enqueue(self, job: dict):
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO pending_jobs (job) VALUES (?)", (json.dumps(job),)
            )
        self.queue.put_nowait((cur.lastrowid, job))

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def _run(self):
        while True:
            pending_id, job = await self.queue.get()
            try:
                await process_job(self.cfg, self.conn, job)
            except Exception:
                log.exception("worker job crashed")
            finally:
                if pending_id is not None:
                    with self.conn:
                        self.conn.execute("DELETE FROM pending_jobs WHERE id=?", (pending_id,))
                self.queue.task_done()
