"""REST handlers used by the hook scripts. Registered in server.py."""

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from . import db as dbmod
from .ingest.worker import validate_transcript_path
from .retrieve import context_for_prompt


def make_handlers(state):
    """state: AppState with .cfg, .conn, .worker"""

    async def health(request: Request) -> Response:
        snap = dbmod.status_snapshot(state.conn)
        snap["queue_depth"] = state.worker.depth if state.worker else 0
        return JSONResponse(snap)

    async def ingest(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        source = body.get("source")
        path = body.get("transcript_path")
        if source not in ("claude", "hermes") or not path:
            return JSONResponse({"error": "need source (claude|hermes) and transcript_path"}, status_code=400)
        try:
            validate_transcript_path(state.cfg, path)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        job = {
            "source": source,
            "session_id": body.get("session_id"),
            "transcript_path": path,
            "reason": body.get("reason"),
            "extra": body.get("extra"),
        }
        dbmod.log_ingest(state.conn, source, job["session_id"] or "?", "queued")
        state.worker.enqueue(job)
        return JSONResponse({"queued": True}, status_code=202)

    async def context(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        block = await context_for_prompt(
            state.cfg, state.conn,
            q=body.get("q") or body.get("prompt") or "",
            session_id=body.get("session_id"),
            cwd=body.get("cwd"),
            k=int(body.get("k") or 8),
        )
        if not block:
            return Response(status_code=204)
        return PlainTextResponse(block)

    return health, ingest, context
