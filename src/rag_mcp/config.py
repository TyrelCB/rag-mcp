"""Env-driven settings. Every endpoint/model/path is overridable for testing."""

import os
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    port: int = int(_env("PORT", "8004"))
    db_path: Path = field(
        default_factory=lambda: Path(
            _env("RAG_DB", str(HOME / ".local/share/rag-mcp/rag.db"))
        ).expanduser()
    )
    embed_url: str = _env("RAG_EMBED_URL", "http://127.0.0.1:11434")
    embed_model: str = _env("RAG_EMBED_MODEL", "qwen3-embedding:0.6b")
    embed_dim: int = int(_env("RAG_EMBED_DIM", "1024"))
    distill_url: str = _env("RAG_DISTILL_URL", "http://127.0.0.1:9090/v1")
    distill_model: str = _env("RAG_DISTILL_MODEL", "")  # empty = first from /v1/models
    context_token_budget: int = int(_env("RAG_CONTEXT_TOKENS", "1500"))
    min_session_chars: int = int(_env("RAG_MIN_SESSION_CHARS", "700"))
    distill_timeout: float = float(_env("RAG_DISTILL_TIMEOUT", "120"))
    distill_max_chars: int = int(_env("RAG_DISTILL_MAX_CHARS", "24000"))
    # Roots ingestion is allowed to read transcripts from.
    claude_projects_dir: Path = field(
        default_factory=lambda: Path(
            _env("RAG_CLAUDE_PROJECTS", str(HOME / ".claude/projects"))
        ).expanduser()
    )
    hermes_sessions_dir: Path = field(
        default_factory=lambda: Path(
            _env("RAG_HERMES_SESSIONS", str(HOME / ".hermes/sessions"))
        ).expanduser()
    )

    def allowed_transcript_roots(self) -> list[Path]:
        return [self.claude_projects_dir, self.hermes_sessions_dir]


def load_config() -> Config:
    cfg = Config()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg
