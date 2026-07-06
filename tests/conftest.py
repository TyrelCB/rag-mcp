import pathlib
import sys

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path):
    from rag_mcp.config import Config
    return Config(
        db_path=tmp_path / "rag.db",
        embed_dim=4,
        claude_projects_dir=FIXTURES,
        hermes_sessions_dir=FIXTURES,
    )


@pytest.fixture
def conn(cfg):
    from rag_mcp import db
    c = db.init_db(cfg)
    yield c
    c.close()
