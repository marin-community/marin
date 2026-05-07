# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared secret redaction helpers."""

import json
import math
import re

REDACTED_VALUE = "[REDACTED]"

_MIN_KEY_ENTROPY = 3.5
_KEY_CHARS_RE = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")
KEY_LIKE_RE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{32,}={0,2}(?![A-Za-z0-9+/_=-])")
PREFIXED_SECRET_RE = re.compile(
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----"
    r"|(?<![A-Za-z0-9_-])(?:"
    r"sk-[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|ASIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{36,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r")(?![A-Za-z0-9_-])"
)
SENSITIVE_KEY_RE = re.compile(
    r"api[_-]?key|secret|token|password|passwd|pwd|auth|credential|private[_-]?key|access[_-]?key|bearer|session",
    re.IGNORECASE,
)


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0

    frequencies: dict[str, int] = {}
    for character in value:
        frequencies[character] = frequencies.get(character, 0) + 1

    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in frequencies.values())


def looks_like_key(value: str, min_len: int = 24, min_entropy: float = _MIN_KEY_ENTROPY) -> bool:
    return len(value) >= min_len and _KEY_CHARS_RE.fullmatch(value) is not None and shannon_entropy(value) >= min_entropy


def is_sensitive_key_name(name: str) -> bool:
    return SENSITIVE_KEY_RE.search(name) is not None


def _redact_key_like_match(match: re.Match[str]) -> str:
    value = match.group(0)
    if shannon_entropy(value) >= _MIN_KEY_ENTROPY:
        return REDACTED_VALUE
    return value


def redact_string(value: str) -> str:
    redacted = PREFIXED_SECRET_RE.sub(REDACTED_VALUE, value)
    return KEY_LIKE_RE.sub(_redact_key_like_match, redacted)


def redact_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: REDACTED_VALUE if isinstance(key, str) and is_sensitive_key_name(key) else redact_value(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(redact_value(child) for child in value)
    if isinstance(value, str):
        if looks_like_key(value):
            return REDACTED_VALUE
        return redact_string(value)
    return value


def redact_json_text(rendered: str) -> str:
    """Parse *rendered* as JSON, redact the structure, and re-emit a compact JSON string.

    Falls back to :func:`redact_string` when the input is not valid JSON, so callers
    never lose protection on malformed previews. An empty string is returned as-is.
    """
    if not rendered:
        return rendered
    try:
        tree = json.loads(rendered)
    except (ValueError, TypeError):
        return redact_string(rendered)
    return json.dumps(redact_value(tree), separators=(",", ":"))
