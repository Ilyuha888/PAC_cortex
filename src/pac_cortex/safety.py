"""Injection resistance and validation for PAC agent."""

import re

# Patterns that suggest prompt injection in tool results
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("system prompt override", re.compile(r"(?i)ignore\s+(previous|above|all)\s+instructions")),
    ("role injection", re.compile(r"(?i)\b(system|assistant)\s*:\s*")),
    ("instruction override", re.compile(r"(?i)you\s+are\s+now\b")),
    ("prompt leak request", re.compile(r"(?i)reveal\s+(your|the)\s+(system|initial)\s+prompt")),
    ("encoded payload", re.compile(r"(?i)base64|\\x[0-9a-f]{2}")),
    # Exec-approval social engineering: file poses as a privileged work order
    ("exec approval social engineering", re.compile(r"(?i)exec[_\-\s]?approval[_\-\s]?needed")),
    # New-task injection: file content tries to reframe the agent's mission
    ("new task injection", re.compile(
        r"(?i)(new\s+task\s*:|your\s+(next\s+)?task\s+is\s*:|task\s*:\s*(delete|remove|send|email|upload|push))"
    )),
    # Targeting critical system/config files — legitimate tasks never come from file content
    ("system file targeting", re.compile(
        r"(?i)(delete|remove|overwrite|truncate|rm\b)\s+.{0,60}"
        r"(AGENTS\.md|\.env|/etc/|/root/|/home/|config\.py|settings\.py|pyproject\.toml)"
    )),
    # Exfiltration: instructions to send content to external addresses
    ("exfiltration instruction", re.compile(
        r"(?i)(send|email|forward|upload|push|post)\s+.{0,60}(password|secret|key|token|credential|api)"
    )),
    # Priority/urgent override: file poses as an urgent ops directive
    ("priority override injection", re.compile(
        r"(?i)(urgent|priority|immediate|critical)\s*[:\-]?\s*(override|admin|ops|escalation|directive)\b"
    )),
    # Action directives in file content: "please execute/run/do X immediately"
    ("action directive injection", re.compile(
        r"(?i)\b(execute|perform|run|carry\s+out)\s+(the\s+following|immediately|now|urgently|this\s+task)"
    )),
    # Hidden log / hide action: file tries to suppress audit trail
    ("log suppression injection", re.compile(
        r"(?i)(hide|suppress|do\s+not\s+log|without\s+logging|don.t\s+log|no\s+trace)"
    )),
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
