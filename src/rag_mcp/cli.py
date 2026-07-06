"""rag-mcp CLI: serve, backfill, ingest, status, export, reembed."""

import asyncio
import json
from pathlib import Path

import typer

from . import db as dbmod
from .config import load_config

app = typer.Typer(help="Hybrid RAG over Claude Code and Hermes sessions.", no_args_is_help=True)


@app.command()
def serve():
    """Run the MCP + REST service (foreground)."""
    from .server import serve as _serve
    _serve()


@app.command()
def status():
    """Print store stats."""
    cfg = load_config()
    conn = dbmod.init_db(cfg)
    print(json.dumps(dbmod.status_snapshot(conn), indent=2, default=str))


@app.command()
def ingest(
    transcript_path: str,
    source: str = typer.Option(..., help="claude or hermes"),
    session_id: str = typer.Option(None),
):
    """Ingest a single transcript synchronously (bypasses the service)."""
    from .ingest.worker import process_job
    cfg = load_config()
    conn = dbmod.init_db(cfg)
    result = asyncio.run(process_job(cfg, conn, {
        "source": source, "session_id": session_id, "transcript_path": transcript_path,
    }))
    print(json.dumps(result, indent=2))


@app.command()
def backfill(
    source: str = typer.Option("all", help="claude, hermes, or all"),
    since: str = typer.Option(None, help="Only files modified after YYYY-MM-DD"),
    limit: int = typer.Option(0, help="Stop after N sessions (0 = no limit)"),
    no_distill: bool = typer.Option(False, help="Skip LLM distillation (faster, raw chunks only)"),
):
    """Seed the store from existing session history."""
    from datetime import datetime
    from .ingest.worker import process_job

    cfg = load_config()
    if no_distill:
        cfg.distill_url = "http://127.0.0.1:1"  # unroutable: distill fails soft instantly
    conn = dbmod.init_db(cfg)
    cutoff = datetime.fromisoformat(since).timestamp() if since else 0

    files: list[tuple[str, Path]] = []
    if source in ("claude", "all"):
        files += [("claude", p) for p in sorted(cfg.claude_projects_dir.glob("*/*.jsonl"))]
    if source in ("hermes", "all"):
        files += [("hermes", p) for p in sorted(cfg.hermes_sessions_dir.glob("*.jsonl"))]
        files += [("hermes", p) for p in sorted(cfg.hermes_sessions_dir.glob("*.json"))
                  if p.name != "sessions.json"]

    files = [(s, p) for s, p in files if p.stat().st_mtime >= cutoff]
    if limit:
        files = files[:limit]
    print(f"backfilling {len(files)} sessions…")

    async def run():
        done = skipped = errors = 0
        for i, (src, path) in enumerate(files, 1):
            r = await process_job(cfg, conn, {"source": src, "transcript_path": str(path)})
            st = r.get("status")
            done += st == "done"
            skipped += st == "skipped"
            errors += st == "error"
            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  done={done} skipped={skipped} errors={errors}")
        print(f"finished: done={done} skipped={skipped} errors={errors}")

    asyncio.run(run())


@app.command()
def export(
    out: Path = typer.Option(..., help="Output directory"),
    source: str = typer.Option("claude"),
    min_turns: int = typer.Option(3, help="Min user turns per session"),
    since: str = typer.Option(None, help="Only sessions started after YYYY-MM-DD"),
    include_tools: bool = typer.Option(True, help="Emit tool-call trajectory variant"),
    val_frac: float = typer.Option(0.05, help="Validation split fraction"),
):
    """Export quality-filtered sessions as SFT chat-format JSONL."""
    from .export.sft import run_export
    cfg = load_config()
    conn = dbmod.init_db(cfg)
    stats = run_export(cfg, conn, out_dir=out, source=source, min_turns=min_turns,
                       since=since, include_tools=include_tools, val_frac=val_frac)
    print(json.dumps(stats, indent=2))


@app.command()
def reembed(
    model: str = typer.Option(..., help="New embedding model (ollama tag)"),
    dim: int = typer.Option(..., help="New embedding dimension"),
):
    """Re-embed every chunk with a new model (rebuilds chunks_vec)."""
    from .embed import embed_texts

    cfg = load_config()
    conn = dbmod.connect(cfg)  # skip init_db's model check on purpose
    cfg.embed_model, cfg.embed_dim = model, dim
    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    print(f"re-embedding {len(rows)} chunks with {model}…")

    async def run():
        conn.execute("DROP TABLE IF EXISTS chunks_vec")
        conn.execute(
            f"CREATE VIRTUAL TABLE chunks_vec USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}])"
        )
        for i in range(0, len(rows), 64):
            batch = rows[i : i + 64]
            embs = await embed_texts(cfg, [r["text"] for r in batch])
            with conn:
                for r, e in zip(batch, embs):
                    conn.execute(
                        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?,?)",
                        (r["id"], dbmod.serialize_embedding(e)),
                    )
            print(f"  {min(i + 64, len(rows))}/{len(rows)}")
        with conn:
            conn.execute("UPDATE meta SET value=? WHERE key='embedding_model'", (model,))
            conn.execute("UPDATE meta SET value=? WHERE key='embedding_dim'", (str(dim),))

    asyncio.run(run())
    print("done. Restart the service with matching RAG_EMBED_MODEL/RAG_EMBED_DIM.")


if __name__ == "__main__":
    app()
