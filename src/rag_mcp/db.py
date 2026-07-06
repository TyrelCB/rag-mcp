"""SQLite schema, sqlite-vec/FTS5 setup, idempotent upsert transactions."""

import json
import sqlite3
from pathlib import Path

import sqlite_vec

from .config import Config

SCHEMA_VERSION = "1"

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL CHECK (source IN ('claude','hermes','manual')),
  session_id TEXT NOT NULL,
  project TEXT,
  cwd TEXT,
  git_branch TEXT,
  model TEXT,
  started_at TEXT,
  ended_at TEXT,
  summary TEXT,
  content_hash TEXT NOT NULL,
  transcript_path TEXT,
  status TEXT DEFAULT 'ok',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT,
  UNIQUE (source, session_id)
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  ts TEXT,
  meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TABLE IF NOT EXISTS ingest_log (
  id INTEGER PRIMARY KEY,
  source TEXT,
  session_id TEXT,
  status TEXT,
  error TEXT,
  chunk_count INTEGER,
  duration_ms INTEGER,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prompt_log (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  cwd TEXT,
  prompt TEXT,
  injected_chunk_ids TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pending_jobs (
  id INTEGER PRIMARY KEY,
  job TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS injected (
  session_id TEXT NOT NULL,
  chunk_id INTEGER NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (session_id, chunk_id)
);
"""


def connect(cfg: Config) -> sqlite3.Connection:
    conn = sqlite3.connect(cfg.db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db(cfg: Config) -> sqlite3.Connection:
    conn = connect(cfg)
    conn.executescript(DDL)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
    if "transcript_path" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN transcript_path TEXT")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
    f"chunk_id INTEGER PRIMARY KEY, embedding float[{cfg.embed_dim}])"
    )
    # Refuse to run against a store built with a different embedding model.
    row = conn.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?), "
            "('embedding_model', ?), ('embedding_dim', ?)",
            (SCHEMA_VERSION, cfg.embed_model, str(cfg.embed_dim)),
        )
        conn.commit()
    elif row["value"] != cfg.embed_model:
        raise RuntimeError(
            f"DB was embedded with {row['value']!r} but config wants {cfg.embed_model!r}. "
            "Run `rag-mcp reembed` to migrate."
        )
    return conn


def serialize_embedding(vec: list[float]) -> bytes:
    return sqlite_vec.serialize_float32(vec)


def upsert_document(
    conn: sqlite3.Connection,
    *,
    source: str,
    session_id: str,
    project: str | None,
    cwd: str | None,
    git_branch: str | None,
    model: str | None,
    started_at: str | None,
    ended_at: str | None,
    summary: str | None,
    content_hash: str,
    status: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    transcript_path: str | None = None,
) -> tuple[int, int]:
    """Replace a document and its chunks atomically. Returns (document_id, chunk_count).

    chunks: [{kind, text, ts, meta}] aligned with embeddings (embeddings may be
    shorter if some chunks failed to embed — those are stored FTS-only).
    """
    with conn:  # single transaction
        existing = conn.execute(
            "SELECT id FROM documents WHERE source=? AND session_id=?",
            (source, session_id),
        ).fetchone()
        if existing:
            doc_id = existing["id"]
            old_chunk_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE document_id=?", (doc_id,)
                )
            ]
            if old_chunk_ids:
                qmarks = ",".join("?" * len(old_chunk_ids))
                conn.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({qmarks})", old_chunk_ids)
                conn.execute(f"DELETE FROM chunks WHERE id IN ({qmarks})", old_chunk_ids)
            conn.execute(
                "UPDATE documents SET project=?, cwd=?, git_branch=?, model=?, started_at=?, "
                "ended_at=?, summary=?, content_hash=?, status=?, transcript_path=?, "
                "updated_at=datetime('now') WHERE id=?",
                (project, cwd, git_branch, model, started_at, ended_at, summary,
                 content_hash, status, transcript_path, doc_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO documents (source, session_id, project, cwd, git_branch, model, "
                "started_at, ended_at, summary, content_hash, status, transcript_path) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (source, session_id, project, cwd, git_branch, model, started_at,
                 ended_at, summary, content_hash, status, transcript_path),
            )
            doc_id = cur.lastrowid

        count = 0
        for i, ch in enumerate(chunks):
            cur = conn.execute(
                "INSERT INTO chunks (document_id, kind, text, ts, meta) VALUES (?,?,?,?,?)",
                (doc_id, ch["kind"], ch["text"], ch.get("ts"),
                 json.dumps(ch.get("meta")) if ch.get("meta") else None),
            )
            if i < len(embeddings):
                conn.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (cur.lastrowid, serialize_embedding(embeddings[i])),
                )
            count += 1
    return doc_id, count


def existing_hash(conn: sqlite3.Connection, source: str, session_id: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM documents WHERE source=? AND session_id=?",
        (source, session_id),
    ).fetchone()
    return row["content_hash"] if row else None


def log_ingest(
    conn: sqlite3.Connection,
    source: str,
    session_id: str,
    status: str,
    error: str | None = None,
    chunk_count: int = 0,
    duration_ms: int = 0,
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO ingest_log (source, session_id, status, error, chunk_count, duration_ms) "
            "VALUES (?,?,?,?,?,?)",
            (source, session_id, status, error, chunk_count, duration_ms),
        )


def prune_injected(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("DELETE FROM injected WHERE created_at < datetime('now', '-7 days')")


def status_snapshot(conn: sqlite3.Connection) -> dict:
    docs = conn.execute(
        "SELECT source, count(*) n FROM documents GROUP BY source"
    ).fetchall()
    chunks = conn.execute("SELECT count(*) n FROM chunks").fetchone()["n"]
    last = conn.execute(
        "SELECT source, session_id, status, error, chunk_count, created_at "
        "FROM ingest_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {
        "documents": {r["source"]: r["n"] for r in docs},
        "chunks": chunks,
        "last_ingest": dict(last) if last else None,
    }
