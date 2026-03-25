"""Injection resistance and validation for PAC agent."""

import re
import unicodedata

# Patterns that suggest prompt injection in tool results
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("system prompt override", re.compile(r"(?i)ignore\s+(previous|above|all)\s+instructions")),
    ("role injection", re.compile(r"(?i)\b(system|assistant)\s*:\s*")),
    ("instruction override", re.compile(r"(?i)you\s+are\s+now\b")),
    ("prompt leak request", re.compile(r"(?i)reveal\s+(your|the)\s+(system|initial)\s+prompt")),
    ("encoded payload", re.compile(r"(?i)base64|\\x[0-9a-f]{2}")),
    # Exec-approval social engineering: file poses as a privileged work order
    # Requires a dangerous verb after ":" to avoid FP on SYSTEM_PROMPT which quotes this as example
    ("exec approval social engineering", re.compile(
        r"(?i)exec[_\-\s]?approval[_\-\s]?needed\s*:\s*"
        r"(delete|remove|run|execute|sudo|push|upload|post|send|purge|wipe|overwrite)"
    )),
    # New-task injection: file content tries to reframe the agent's mission
    ("new task injection", re.compile(
        r"(?i)(new\s+task\s*:|your\s+(next\s+)?task\s+is\s*:|task\s*:\s*(delete|remove|send|email|upload|push))"
    )),
    # Targeting critical system/config files — legitimate tasks never come from file content
    ("system file targeting", re.compile(
        r"(?i)(delete|remove|overwrite|truncate|rm\b)\s+.{0,60}"
        r"((?<!\()AGENTS\.md|(?<!, )\.env|/etc/|/root/|/home/"
        r"|config\.py|settings\.py|pyproject\.toml)"
    )),
    # Exfiltration: instructions to send content to external addresses
    ("exfiltration instruction", re.compile(
        r"(?i)(send|email|forward|upload|push|post)\s+.{3,60}(password|secret|key|token|credential|api)"
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
    # Persona jailbreaks
    ("jailbreak persona", re.compile(
        r"(?i)\b(DAN|DUDE|AIM|STAN|JAILBREAK|dev\s*mode|developer\s*mode|god\s*mode|unrestricted\s*mode)\b"
    )),
    # Synonyms for "ignore safety rules"
    ("disregard safety", re.compile(
        r"(?i)\b(disregard|forget|bypass|circumvent|override)\s+(all\s+)?(safety|restriction|constraint|guard|filter|rule)s?\b"
    )),
    # Delimiter spoofing
    ("instruction delimiter injection", re.compile(
        r"(?i)[-=*#]{5,}\s*(end\s+of\s+(system\s+)?prompt|instructions?\s+end|stop\s+here)"
    )),
    # ChatML/XML tag boundary spoofing
    ("prompt boundary spoof", re.compile(
        r"(?i)<\s*/?\s*(system|user|assistant|instruction|prompt|context)\s*>"
    )),
    # CSS hide tricks
    ("hidden text directive", re.compile(
        r"(?i)(color\s*:\s*white|font.size\s*:\s*0|display\s*:\s*none|visibility\s*:\s*hidden)"
    )),
    # Print-the-key instructions
    ("credential handoff", re.compile(
        r"(?i)(print|output|log|return|echo|show|display)\s+(the\s+)?(api[_-]?key|secret|token|password|credential)"
    )),
    # Sudo/root framing
    ("privilege escalation framing", re.compile(
        r"(?i)\b(as\s+(root|admin|superuser|system)|sudo\s+|run\s+as\s+(root|admin))\b"
    )),
    # Known exfil endpoints
    ("webhook exfiltration", re.compile(
        r"(?i)(https?://[^\s]{0,80}(webhook|exfil|callback|c2|ngrok|burp|requestbin|pipedream))"
    )),
]

# Env var names and credential formats that must never appear in outputs
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-._~+/]{8,}"),
    re.compile(r"(?i)export\s+[A-Z_]{3,}[A-Z0-9_]*\s*=\s*\S+"),
    re.compile(r'(?i)"(api[_-]?key|secret|token|password|credential|private[_-]?key)"\s*:\s*"[^"]{4,}"'),
    re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"),
    re.compile(r"\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16})\b"),
]

_PATH_TRAVERSAL_RE = re.compile(r"(?:^|/|\\)\.\.(?:/|\\|$)")
_PATH_ARG_KEYS: frozenset[str] = frozenset({"path", "from_name", "to_name", "root", "name"})


def scan_for_injection(content: str) -> list[str]:
    """Flag suspicious patterns in content. Returns list of warning descriptions."""
    nfkc = unicodedata.normalize("NFKC", content)
    # Replace zero-width / soft-hyphen / BOM chars with a space to preserve word boundaries
    pre_norm = re.sub(r'[\u00ad\u200b\u200c\u200d\u2060-\u206f\ufeff]+', ' ', nfkc)
    normalized = re.sub(r'\s+', ' ', pre_norm).strip()
    # collapsed: remove all separators → catches "b-a-s-e-6-4"
    collapsed = re.sub(r'[\s\-_.,]+', '', normalized)
    # despaced: collapse letter-by-letter obfuscation "i g n o r e" → "ignore"
    # Must run on pre_norm (multi-space word separators preserved) so word boundaries survive
    despaced = re.sub(
        r'(?<!\w)(?:\w )+\w(?!\w)',
        lambda m: m.group(0).replace(' ', ''),
        pre_norm,
    )
    despaced = re.sub(r'\s+', ' ', despaced).strip()

    seen: set[str] = set()
    warnings: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        if label in seen:
            continue
        for variant in (content, normalized, collapsed, despaced):
            if pattern.search(variant):
                warnings.append(label)
                seen.add(label)
                break
    return warnings


def validate_tool_call(name: str, args: dict, allowed_tools: set[str] | None = None) -> bool:
    """Check tool call against allowed actions. Returns True if valid."""
    if allowed_tools is not None and name not in allowed_tools:
        return False
    for key, val in args.items():
        if key in _PATH_ARG_KEYS and isinstance(val, str) and _PATH_TRAVERSAL_RE.search(val):
            return False
    return True


def redact_secrets(content: str) -> str:
    """Remove potential secret values from content."""
    result = content
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result
