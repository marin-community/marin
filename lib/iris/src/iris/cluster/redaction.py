# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for redacting secrets from job submissions and responses.

Used on both the client (before capturing argv for bookkeeping) and the
controller (when returning job requests via RPC).
"""

import json
import logging
import re

from iris.rpc import controller_pb2

logger = logging.getLogger(__name__)

SENSITIVE_ENV_KEY_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
REDACTED_VALUE = "**REDACTED**"

# CLI flags on `iris job run` that take a (KEY, VALUE) pair via Click's
# type=(str, str). Keep in sync with the Click option definition in
# iris/cli/job.py; see the `-e/--env-vars` option on the `run` command.
_ENV_VAR_LONG_FLAG = "--env-vars"
_ENV_VAR_SHORT_FLAG = "-e"


def is_sensitive_env_key(key: str) -> bool:
    """Return True if *key* looks like a secret env var name."""
    return bool(SENSITIVE_ENV_KEY_RE.search(key))


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
    """Return a copy of *request* with sensitive env var values replaced."""
    if not request.environment.env_vars:
        return request

    redacted = controller_pb2.Controller.LaunchJobRequest()
    redacted.CopyFrom(request)
    env_vars = redacted.environment.env_vars
    for key in list(env_vars):
        if is_sensitive_env_key(key):
            env_vars[key] = REDACTED_VALUE
    return redacted


def _redact_tree(node):
    """Walk a parsed-JSON tree, redacting values under sensitive-looking keys."""
    if isinstance(node, dict):
        return {k: REDACTED_VALUE if is_sensitive_env_key(k) else _redact_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_tree(v) for v in node]
    return node


def redact_json_preview(rendered: str) -> str:
    """Return *rendered* with values under sensitive-looking keys replaced.

    The input is arbitrary JSON (typically a protobuf rendered via
    ``MessageToJson``). Any key matching :data:`SENSITIVE_ENV_KEY_RE` at
    any depth has its value replaced with :data:`REDACTED_VALUE`; this also
    covers map<string,string> fields like ``env_vars`` where secrets tend
    to live. Unparseable input is returned unchanged so callers never lose
    the preview entirely.
    """
    if not rendered:
        return rendered
    try:
        tree = json.loads(rendered)
    except ValueError:
        logger.debug("redact_json_preview: input was not valid JSON, returning as-is")
        return rendered
    return json.dumps(_redact_tree(tree), separators=(",", ":"))


def redact_submit_argv(argv: list[str]) -> list[str]:
    """Redact secret-looking values from a captured CLI argv.

    Handles every Click syntax for the `-e`/`--env-vars` tuple option:

      * ``-e KEY VALUE`` / ``--env-vars KEY VALUE`` — bare
      * ``--env-vars=KEY VALUE`` — attached long form
      * ``-eKEY VALUE`` — attached short form

    When KEY matches SENSITIVE_ENV_KEY_RE, VALUE is replaced with
    REDACTED_VALUE. Other tokens pass through unchanged.
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

        if is_sensitive_env_key(key):
            out[val_idx] = REDACTED_VALUE
        i = val_idx + 1
    return out
