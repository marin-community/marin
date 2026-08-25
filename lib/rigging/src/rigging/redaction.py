# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared secret redaction helpers."""

import itertools
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
# Words that mark a key as secret-bearing, matched against the key's words
# rather than as raw substrings. A substring match is wrong in both directions
# for real env var names: "token" appears inside ``TOKENIZERS_PARALLELISM`` and
# ``TOKENIZER_PATH``, and "auth" inside ``AUTHOR``, none of which are secrets.
_SENSITIVE_WORDS = frozenset(
    {
        "apikey",
        "auth",
        "authentication",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "passphrase",
        "passwd",
        "password",
        "passwords",
        "pwd",
        "secret",
        "secrets",
        "session",
        "sessions",
        "token",
        "tokens",
    }
)

# Sensitive only as an adjacent word pair. "key" on its own is too common in
# non-secret names (``cache_key``, ``sort_key``, ``key_name``) to redact.
_SENSITIVE_PHRASES = frozenset(
    {
        ("api", "key"),
        ("access", "key"),
        ("private", "key"),
    }
)

# "token" is a credential in ``HF_TOKEN`` but a sequence length in
# ``max_tokens`` / ``token_count``. When a counting word sits alongside it and
# nothing else in the key is sensitive, read it as the length sense.
_TOKEN_WORDS = frozenset({"token", "tokens"})
_COUNT_WORDS = frozenset(
    {
        "avg",
        "budget",
        "count",
        "len",
        "length",
        "limit",
        "max",
        "mean",
        "min",
        "num",
        "per",
        "size",
        "sum",
        "total",
        "window",
    }
)

# Splits an identifier into words on separator characters and camelCase humps,
# so ``WANDB_API_KEY``, ``wandbApiKey`` and ``wandb-api-key`` all yield
# ``["wandb", "api", "key"]``. An all-caps run stays one word, which is what
# keeps ``TOKENIZERS`` from reading as ``TOKEN``.
_KEY_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")

# Keys whose values are known to be identifiers (job/task/worker IDs, names,
# zones, hostnames, etc.) rather than secrets. Values under safe keys skip
# the entropy-based key-like heuristic (which has false positives on
# hyphenated job paths like ``/alice/some-job/0``); prefix-based detection
# (``sk-...``, ``ghp_...``, AKIA, JWTs, PEM blocks) still applies, so a real
# secret accidentally placed in a safe field is still caught.
#
# Sensitive-key detection runs first, so a key that is both sensitive and safe
# (e.g., ``auth_id``) is still redacted as sensitive.
SAFE_KEY_RE = re.compile(
    # Generic identifier suffixes: foo_id, foo_ids, foo_name, foo_index, foo_no, foo_num
    r".*_(id|name|index|idx|no|num)s?$"
    # Common bare identifier / categorical / location fields. Keep this list
    # tight: every entry should be unambiguously an identifier or label, not
    # a free-form payload that might carry a secret.
    r"|^(name|namespace|hostname|username|address|ip_address|zone|region|"
    r"status|state|phase|stage|shard|kind|mode|reason|"
    r"url|uri|path|file|filename|filepath|"
    r"scale_group|image_tag|git_hash|git_sha|commit|commit_sha|"
    r"target|peer|user_agent)$",
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


def key_words(name: str) -> list[str]:
    """Split *name* into its lowercased words on separators and camelCase humps."""
    return [word.lower() for word in _KEY_WORD_RE.findall(name)]


def is_sensitive_key_name(name: str) -> bool:
    """Return True if *name* names a secret-bearing field.

    Matches whole words, not substrings, so ``HF_TOKEN`` is sensitive while
    ``TOKENIZERS_PARALLELISM`` is not. "token" alongside a counting word and no
    other sensitive word reads as a length, so ``max_tokens`` is not sensitive
    either, while ``session_token_count`` still is.
    """
    words = key_words(name)
    hits = {word for word in words if word in _SENSITIVE_WORDS}
    if hits <= _TOKEN_WORDS and any(word in _COUNT_WORDS for word in words):
        hits = set()
    if hits:
        return True
    return any(pair in _SENSITIVE_PHRASES for pair in itertools.pairwise(words))


def is_safe_key_name(name: str) -> bool:
    """Return True if *name* is a known non-sensitive identifier/name field.

    Values under safe keys skip the entropy-based key-like heuristic, which
    has false positives on hyphenated job paths and structured identifiers
    such as ``/alice/some-job-name/0``. Prefix-based secret detection (for
    ``sk-...``, ``ghp_...``, AKIA, JWTs, and PEM blocks) still applies, so a
    real API token accidentally placed in a safe field is still caught.

    Sensitive-key detection takes precedence over this check, so a key like
    ``auth_id`` is still redacted as sensitive.
    """
    return SAFE_KEY_RE.match(name) is not None


def _redact_key_like_match(match: re.Match[str]) -> str:
    value = match.group(0)
    if shannon_entropy(value) >= _MIN_KEY_ENTROPY:
        return REDACTED_VALUE
    return value


def redact_string(value: str) -> str:
    redacted = PREFIXED_SECRET_RE.sub(REDACTED_VALUE, value)
    return KEY_LIKE_RE.sub(_redact_key_like_match, redacted)


def _redact_string_prefix_only(value: str) -> str:
    """Strip prefixed secrets only; skip the entropy heuristic.

    Used for string values under safe identifier keys, where the entropy
    heuristic produces false positives on legitimate identifiers (job paths,
    hostnames, etc.).
    """
    return PREFIXED_SECRET_RE.sub(REDACTED_VALUE, value)


def _redact_under_safe_key(value: object) -> object:
    """Redact *value* assuming it lives under a safe identifier key.

    Strings only get prefix-based redaction (no entropy check). Lists and
    tuples recurse with the same semantics so e.g. ``task_ids: [...]`` is
    handled. Dicts and other types fall back to default :func:`redact_value`,
    since the safe-key context only applies to the immediate identifier
    payload, not to arbitrary nested structures.
    """
    if isinstance(value, str):
        return _redact_string_prefix_only(value)
    if isinstance(value, list):
        return [_redact_under_safe_key(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_redact_under_safe_key(child) for child in value)
    return redact_value(value)


def redact_value(value: object) -> object:
    if isinstance(value, dict):
        result: dict[object, object] = {}
        for key, child in value.items():
            if isinstance(key, str):
                if is_sensitive_key_name(key):
                    result[key] = REDACTED_VALUE
                    continue
                if is_safe_key_name(key):
                    result[key] = _redact_under_safe_key(child)
                    continue
            result[key] = redact_value(child)
        return result
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
