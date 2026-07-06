from rag_mcp.ingest.claude_transcript import parse_claude_transcript
from rag_mcp.ingest.hermes_transcript import parse_hermes_session

from conftest import FIXTURES


def test_claude_parse_basic():
    s = parse_claude_transcript(FIXTURES / "claude_session.jsonl")
    assert s.source == "claude"
    assert s.cwd == "/home/tyrel/projects/embeddings"
    assert s.git_branch == "main"
    assert s.model == "claude-fable-5"
    assert s.started_at == "2026-07-01T10:00:00.000Z"
    assert s.ended_at == "2026-07-01T10:03:00.000Z"


def test_claude_parse_filters_noise():
    s = parse_claude_transcript(FIXTURES / "claude_session.jsonl")
    users = [t.text for t in s.turns if t.role == "user"]
    # harness /clear command row dropped
    assert not any("<command-name>" in u for u in users)
    # system-reminder stripped from real prompt
    assert any(u == "the container crashes on start with a permissions error" for u in users)
    # thinking blocks never surface
    assert not any("secret internal reasoning" in t.text for t in s.turns)
    # tool_use recorded compactly on assistant turn
    assert any("[tool: Write" in t.text for t in s.turns if t.role == "assistant")
    # tool_result becomes a tool turn
    assert any(t.role == "tool" and "File created" in t.text for t in s.turns)


def test_harness_noise_rows_dropped():
    from rag_mcp.ingest.claude_transcript import _clean_user_text
    assert _clean_user_text("<task-notification>\n<task-id>x</task-id>\n</task-notification>") == ""
    assert _clean_user_text("<command-name>/clear</command-name>") == ""
    assert _clean_user_text("real question about ports") == "real question about ports"


def test_hermes_parse():
    s = parse_hermes_session(FIXTURES / "hermes_session.jsonl")
    roles = [t.role for t in s.turns]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    # system prompt dropped
    assert not any("giant system prompt" in t.text for t in s.turns)
    # tool_calls summarized on assistant turn
    assert "[tool: shell" in s.turns[1].text
    assert s.started_at == "2026-06-20T09:00:01Z"
