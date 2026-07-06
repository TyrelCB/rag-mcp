"""Secret/PII scrubber shared by ingestion and SFT export."""

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret", re.compile(r"(?i)\baws_secret_access_key\b\s*[:=]\s*\S+")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abp]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b")),
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S),
    ),
    ("bearer", re.compile(r"(?i)\bauthorization:\s*bearer\s+\S+")),
    (
        "generic_secret",
        re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_/+.-]{8,}['\"]?"),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]


def scrub(text: str) -> tuple[str, dict[str, int]]:
    """Return (scrubbed_text, {pattern_name: replacement_count})."""
    counts: dict[str, int] = {}
    for name, pat in _PATTERNS:
        text, n = pat.subn(f"[REDACTED:{name}]", text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def scrub_text(text: str) -> str:
    return scrub(text)[0]
