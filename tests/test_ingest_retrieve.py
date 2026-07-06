import asyncio

import pytest

from conftest import FIXTURES
from rag_mcp import db as dbmod
from rag_mcp.ingest.worker import process_job, validate_transcript_path
from rag_mcp.retrieve import fts_sanitize, hybrid_search, pack_context


def _ingest(cfg, conn, name, source, session_id):
    # distill/embed endpoints are unroutable in tests -> fail-soft, FTS-only chunks
    cfg.distill_url = "http://127.0.0.1:1/v1"
    cfg.embed_url = "http://127.0.0.1:1"
    cfg.distill_timeout = 1
    if cfg.min_session_chars == 700:  # tests default to permissive unless overridden
        cfg.min_session_chars = 50
    return asyncio.run(process_job(cfg, conn, {
        "source": source, "session_id": session_id,
        "transcript_path": str(FIXTURES / name),
    }))


def test_path_allowlist(cfg):
    with pytest.raises(ValueError):
        validate_transcript_path(cfg, "/etc/passwd")
    assert validate_transcript_path(cfg, str(FIXTURES / "claude_session.jsonl"))


def test_ingest_idempotent_and_secrets_scrubbed(cfg, conn):
    r1 = _ingest(cfg, conn, "claude_session.jsonl", "claude", "s1")
    assert r1["status"] == "done" and r1["chunks"] > 0
    # secret in the assistant answer got scrubbed at chunk time
    texts = [r["text"] for r in conn.execute("SELECT text FROM chunks")]
    assert not any("sk-abcdef1234567890" in t for t in texts)
    # unchanged re-ingest skips
    r2 = _ingest(cfg, conn, "claude_session.jsonl", "claude", "s1")
    assert r2["status"] == "skipped"
    # only one document row
    assert conn.execute("SELECT count(*) c FROM documents").fetchone()["c"] == 1


def test_junk_session_skipped(cfg, conn):
    cfg.min_session_chars = 100000  # force the too-short path
    r = _ingest(cfg, conn, "claude_session.jsonl", "claude", "s-short")
    assert r == {"status": "skipped", "reason": "too_short"}
    row = conn.execute(
        "SELECT status FROM documents WHERE session_id='s-short'"
    ).fetchone()
    assert row["status"] == "skipped_short"


def test_hybrid_search_and_injection_dedupe(cfg, conn):
    _ingest(cfg, conn, "claude_session.jsonl", "claude", "s1")
    _ingest(cfg, conn, "hermes_session.jsonl", "hermes", "h1")

    res = hybrid_search(conn, None, "qdrant docker container persist", k=5)
    assert res and any("qdrant" in r["text"].lower() for r in res)

    block, ids = pack_context(res, token_budget=400)
    assert block.startswith("Relevant prior context")
    assert ids

    # mark as injected for a session -> excluded next time
    with conn:
        conn.executemany(
            "INSERT INTO injected (session_id, chunk_id) VALUES (?,?)",
            [("sess-x", i) for i in ids],
        )
    res2 = hybrid_search(conn, None, "qdrant docker container persist", k=5,
                         skip_injected_for="sess-x")
    assert not set(ids) & {r["id"] for r in res2}

    # own-session exclusion
    res3 = hybrid_search(conn, None, "qdrant docker container persist", k=5,
                         exclude_session_id="s1")
    assert not any(r["session_id"] == "s1" for r in res3)


def test_fts_sanitize_hostile_input():
    q = fts_sanitize('drop "table" AND (x OR *) NEAR/3 -bad')
    # every term quoted, no raw operators leak
    assert '"drop"' in q and '"table"' in q
    for tok in q.split(" OR "):
        assert tok.startswith('"') and tok.endswith('"')


def test_pack_context_respects_budget(cfg, conn):
    _ingest(cfg, conn, "claude_session.jsonl", "claude", "s1")
    res = hybrid_search(conn, None, "qdrant docker", k=8)
    block, _ = pack_context(res, token_budget=60)
    assert len(block) <= 60 * 4 + 50
