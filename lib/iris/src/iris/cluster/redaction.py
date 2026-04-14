# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for redacting secrets from job submissions and responses.

Used on both the client (before capturing argv for bookkeeping) and the
controller (when returning job requests via RPC).
"""

import re

from iris.rpc import controller_pb2

SENSITIVE_ENV_KEY_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
REDACTED_VALUE = "**REDACTED**"

# CLI flags on `iris job run` that take a (KEY, VALUE) pair via Click's
# type=(str, str). Keep in sync with the Click option definition in
# iris/cli/job.py; see the `-e/--env-vars` option on the `run` command.
_ENV_VAR_FLAGS = frozenset({"-e", "--env-vars"})


def is_sensitive_env_key(key: str) -> bool:
    """Return True if *key* looks like a secret env var name."""
    return bool(SENSITIVE_ENV_KEY_RE.search(key))


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


def redact_submit_argv(argv: list[str]) -> list[str]:
    """Redact secret-looking values from a captured CLI argv.

    Handles `-e KEY VALUE` / `--env-vars KEY VALUE` pairs (Click's
    ``type=(str, str)`` form): when KEY matches SENSITIVE_ENV_KEY_RE, VALUE
    is replaced with REDACTED_VALUE. Other tokens pass through unchanged.
    """
    out = list(argv)
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _ENV_VAR_FLAGS and i + 2 < len(out):
            key = out[i + 1]
            if is_sensitive_env_key(key):
                out[i + 2] = REDACTED_VALUE
            i += 3
            continue
        i += 1
    return out
