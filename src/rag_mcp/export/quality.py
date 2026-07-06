"""Session quality heuristics for SFT export. Frontier-model sessions only."""

import re

from ..ingest.types import Session

_FAILURE_ENDINGS = re.compile(
    r"(?i)(i apologize|i'm sorry|i am sorry|unfortunately,? i (was unable|could ?n[o']t)|"
    r"i (was unable|could ?n[o']t) (to )?(complete|finish|resolve))"
)

TOOL_ERROR_MAX_RATIO = 0.30


def is_frontier(model: str | None) -> bool:
    return bool(model) and model.startswith("claude")


def quality_check(sess: Session, *, min_turns: int, completed: bool = True) -> tuple[bool, str]:
    """Returns (passes, reason_if_not)."""
    if not is_frontier(sess.model):
        return False, f"non_frontier_model:{sess.model}"
    if not completed:
        return False, "interrupted"
    if len(sess.user_turns) < min_turns:
        return False, "too_few_turns"

    tool_turns = [t for t in sess.turns if t.role == "tool"]
    if tool_turns:
        ratio = sum(t.is_error for t in tool_turns) / len(tool_turns)
        if ratio >= TOOL_ERROR_MAX_RATIO:
            return False, f"tool_error_ratio:{ratio:.2f}"

    final = next((t for t in reversed(sess.turns) if t.role == "assistant"), None)
    if final is None:
        return False, "no_assistant_output"
    if _FAILURE_ENDINGS.search(final.text[:400]):
        return False, "failure_ending"
    return True, ""
