"""Normalized session representation shared by both transcript parsers."""

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # user | assistant | tool
    text: str
    ts: str | None = None
    model: str | None = None
    tool_name: str | None = None
    is_error: bool = False


@dataclass
class Session:
    source: str  # claude | hermes
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    cwd: str | None = None
    git_branch: str | None = None
    model: str | None = None
    platform: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def total_chars(self) -> int:
        return sum(len(t.text) for t in self.turns if t.role in ("user", "assistant"))
