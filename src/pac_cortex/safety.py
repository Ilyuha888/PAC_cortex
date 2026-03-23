"""Injection resistance and validation for PAC agent."""

import re

# Patterns that suggest prompt injection in tool results
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("system prompt override", re.compile(r"(?i)ignore\s+(previous|above|all)\s+instructions")),
    ("role injection", re.compile(r"(?i)\b(system|assistant)\s*:\s*")),
    ("instruction override", re.compile(r"(?i)you\s+are\s+now\b")),
    ("prompt leak request", re.compile(r"(?i)reveal\s+(your|the)\s+(system|initial)\s+prompt")),
    ("encoded payload", re.compile(r"(?i)base64|\\x[0-9a-f]{2}")),
]

# Env var names that must never appear in outputs
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*\S+"),
]


def scan_for_injection(content: str) -> list[str]:
    """Flag suspicious patterns in content. Returns list of warning descriptions."""
    warnings: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(content):
            warnings.append(label)
    return warnings


def validate_tool_call(name: str, args: dict, allowed_tools: set[str] | None = None) -> bool:
    """Check tool call against allowed actions. Returns True if valid."""
    return allowed_tools is None or name in allowed_tools


def redact_secrets(content: str) -> str:
    """Remove potential secret values from content."""
    result = content
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result
