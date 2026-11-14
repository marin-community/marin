# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-specific redaction wrappers around :mod:`rigging.redaction`.

The actual secret-pattern matching lives in ``rigging``; this module only
adapts iris-specific shapes (the ``LaunchJobRequest`` proto and Click argv
for ``iris job run``) into a form the shared redactor can consume, and finds
Iris proxy capability tokens in free text.
"""

import base64
import json
import re
from collections.abc import Iterator

from rigging.redaction import REDACTED_VALUE, is_sensitive_key_name, redact_string, redact_value

from iris.cluster.controller.auth import PROXY_PLANE_AUDIENCE
from iris.rpc import controller_pb2

# A line break inside a token: whitespace, or an escaped n/r/t behind any number of
# backslashes (JSON inside JSON doubles them, e.g. a log line in a tool output in a trace).
_BREAK = r"(?:\s|\\+[nrt])"
_BREAKS_RE = re.compile(rf"{_BREAK}+")
_CHAR = r"[A-Za-z0-9_\-]"
# Header and payload end at their dot. The signature has no terminator, so it is matched by length:
# capability tokens are EdDSA (Ed25519) signed, and a 64-byte signature is 86 base64url characters.
# An unbounded signature would run through a break into the text that follows the token.
_SIGNATURE_CHARS = 86
_SEGMENT = rf"{_CHAR}(?:{_CHAR}|{_BREAK})*"
_SIGNATURE = rf"(?:{_CHAR}{_BREAK}*){{{_SIGNATURE_CHARS - 1}}}{_CHAR}"
_BROKEN_JWT_RE = re.compile(rf"e{_BREAK}*y{_BREAK}*J{_SEGMENT}\.{_BREAK}*{_SEGMENT}\.{_BREAK}*{_SIGNATURE}")

# CLI flags on `iris job run` that take a (KEY, VALUE) pair via Click's
# type=(str, str). Keep in sync with the Click option definition in
# iris/cli/job.py; see the `-e/--env-vars` option on the `run` command.
_ENV_VAR_LONG_FLAG = "--env-vars"
_ENV_VAR_SHORT_FLAG = "-e"


def _env_var_flag_key(token: str, next_token: str | None) -> str | None:
    """Extract the KEY from an env-var flag token, or None if not such a flag.

    Handles all Click syntaxes for a `(str, str)` option:
      * bare long form: `--env-vars KEY VALUE` → KEY is *next_token*
      * attached long form: `--env-vars=KEY VALUE` → KEY is embedded
      * bare short form: `-e KEY VALUE` → KEY is *next_token*
      * attached short form: `-eKEY VALUE` → KEY is embedded
    """
    if token == _ENV_VAR_LONG_FLAG or token == _ENV_VAR_SHORT_FLAG:
        return next_token
    if token.startswith(_ENV_VAR_LONG_FLAG + "="):
        return token[len(_ENV_VAR_LONG_FLAG) + 1 :]
    if token.startswith(_ENV_VAR_SHORT_FLAG) and len(token) > 2 and not token.startswith("--"):
        return token[2:]
    return None


def redact_request_env_vars(
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Redact credential-looking env values and every explicit secret value."""
    if not request.environment.env_vars and not request.environment.secrets:
        return request

    redacted = controller_pb2.Controller.LaunchJobRequest()
    redacted.CopyFrom(request)
    env_vars = redacted.environment.env_vars
    redacted_env = redact_value(dict(env_vars))
    env_vars.clear()
    env_vars.update(redacted_env)
    redacted.environment.secrets.update(dict.fromkeys(request.environment.secrets, REDACTED_VALUE))
    return redacted


def redact_submit_argv(argv: list[str]) -> list[str]:
    """Redact secret-looking values from a captured CLI argv.

    Handles every Click syntax for the `-e`/`--env-vars` tuple option:

      * ``-e KEY VALUE`` / ``--env-vars KEY VALUE`` — bare
      * ``--env-vars=KEY VALUE`` — attached long form
      * ``-eKEY VALUE`` — attached short form

    When KEY matches the shared sensitive-key regex, VALUE is replaced with
    REDACTED_VALUE. Values under benign keys still pass through the shared
    string redactor so prefixed or high-entropy tokens do not leak.
    """
    out = list(argv)
    n = len(out)
    i = 0
    while i < n:
        tok = out[i]
        next_tok = out[i + 1] if i + 1 < n else None
        key = _env_var_flag_key(tok, next_tok)
        if key is None:
            i += 1
            continue

        # Locate the VALUE token. For bare forms KEY is the next token and
        # VALUE is the one after; for attached forms KEY is embedded and
        # VALUE is the next token.
        attached = tok != _ENV_VAR_LONG_FLAG and tok != _ENV_VAR_SHORT_FLAG
        val_idx = i + 1 if attached else i + 2
        if val_idx >= n:
            # Malformed (no VALUE token present); leave argv alone.
            i += 1
            continue

        value = out[val_idx]
        out[val_idx] = REDACTED_VALUE if is_sensitive_key_name(key) else redact_string(value)
        i = val_idx + 1
    return out


def _is_capability_token(token: str) -> bool:
    parts = _BREAKS_RE.sub("", token).split(".")
    if len(parts) != 3:
        return False
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except ValueError:  # not base64 or not JSON: some other dotted string
        return False
    audience = payload.get("aud") if isinstance(payload, dict) else None
    return audience == PROXY_PLANE_AUDIENCE or (isinstance(audience, list) and PROXY_PLANE_AUDIENCE in audience)


def _capability_matches(text: str) -> Iterator[re.Match[str]]:
    position = 0
    while match := _BROKEN_JWT_RE.search(text, position):
        if _is_capability_token(match.group(0)):
            yield match
            position = match.end()
        else:
            # A shorter, unrelated JWT can consume the next token's header as part
            # of its signature. Search inside rejected candidates for that token.
            position = match.start() + 1


def count_capability_tokens(text: str) -> int:
    """Count Iris proxy capability tokens in *text*, line breaks inside a token included."""
    return sum(1 for _ in _capability_matches(text))


def scrub_capability_tokens(text: str) -> tuple[str, int]:
    """Replace each Iris proxy capability token in *text* and return the text and the count replaced.

    A capability token is a JWT with ``aud`` ``iris-proxy``; other JWTs and the text around a token are
    kept. Agents can print a capability URL (e.g. an ``env`` listing), and terminals and JSON logs wrap
    it, so tokens split by whitespace or escaped line breaks are matched too. Callers should confirm
    that ``count_capability_tokens`` of the result is 0.
    """
    parts = []
    position = 0
    found = 0
    for match in _capability_matches(text):
        parts.extend((text[position : match.start()], REDACTED_VALUE))
        position = match.end()
        found += 1
    parts.append(text[position:])
    return "".join(parts), found
