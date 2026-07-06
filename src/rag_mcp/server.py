"""FastMCP app: MCP tools + REST routes for hooks + background ingest worker."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastmcp import FastMCP

from . import db as dbmod
from .api import make_handlers
from .config import Config, load_config
from .ingest.worker import IngestWorker
from .mcp_tools import register_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@dataclass
class AppState:
    cfg: Config
    conn: object = None
    worker: IngestWorker = field(default=None)


def build_app(cfg: Config | None = None) -> FastMCP:
    cfg = cfg or load_config()
    state = AppState(cfg=cfg)
    state.conn = dbmod.init_db(cfg)
    state.worker = IngestWorker(cfg, state.conn)

    @asynccontextmanager
    async def lifespan(app):
        state.worker.start()
        try:
            yield
        finally:
            await state.worker.stop()
            state.conn.close()

    mcp = FastMCP("rag-mcp", lifespan=lifespan)
    register_tools(mcp, state)

    health, ingest, context = make_handlers(state)
    mcp.custom_route("/health", methods=["GET"])(health)
    mcp.custom_route("/api/ingest", methods=["POST"])(ingest)
    mcp.custom_route("/api/context", methods=["POST"])(context)
    return mcp


def serve():
    cfg = load_config()
    app = build_app(cfg)
    app.run(
        transport="http",
        host="0.0.0.0",
        port=cfg.port,
        path="/mcp",
        stateless_http=True,
        # Caddy terminates TLS and owns the security layer for the fleet.
        host_origin_protection=False,
    )
