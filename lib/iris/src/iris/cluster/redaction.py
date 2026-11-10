# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-specific redaction wrappers around :mod:`rigging.redaction`.

The actual secret-pattern matching lives in ``rigging``; this module only
adapts iris-specific shapes (the ``LaunchJobRequest`` proto and Click argv
for ``iris job run``) into a form the shared redactor can consume.
"""

import re

from rigging.redaction import REDACTED_VALUE, is_sensitive_key_name, redact_string, redact_value

from iris.rpc import controller_pb2

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


_SECRET_FLAG = "--secret"


def _flag_name(token: str) -> str | None:
    """The name of a ``--name`` / ``--name=value`` / ``-n`` token, or None."""
    if not token.startswith("-") or token == "-" or token == "--":
        return None
    return token.lstrip("-").split("=", 1)[0]


def redact_command_argv(argv: list[str]) -> list[str]:
    """Redact a user command for long-term storage.

    A value after a flag whose name looks sensitive (``--hf-token X``,
    ``--api-key=X``) is replaced. Every other token passes through
    :func:`redact_script`, which catches sensitive ``NAME=value`` assignments
    (as in ``bash -c "export HF_TOKEN=..."``) and prefixed or high-entropy tokens. A secret
    passed as a plain positional argument or inside a URL with an ordinary
    shape is not caught.
    """
    out: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next and not token.startswith("-"):
            out.append(REDACTED_VALUE)
            redact_next = False
            continue
        redact_next = False
        name = _flag_name(token)
        if name is not None and is_sensitive_key_name(name):
            if "=" in token:
                out.append(token.split("=", 1)[0] + "=" + REDACTED_VALUE)
            else:
                out.append(token)
                redact_next = True
            continue
        out.append(redact_script(token))
    return out


def redact_stored_submit_argv(argv: list[str]) -> list[str]:
    """Redact an ``iris job run`` command line for long-term storage, failing closed.

    Every ``-e``/``--env-vars`` value and every ``--secret`` source is replaced,
    whatever its name, in the bare and attached forms. Then the whole argv goes
    through :func:`redact_command_argv`, since the command may follow the flags
    without ``--``.
    """
    out = list(argv)
    i = 0
    while i < len(out):
        token = out[i]
        next_token = out[i + 1] if i + 1 < len(out) else None
        if token == "--":
            break
        if _env_var_flag_key(token, next_token) is not None:
            attached = token not in (_ENV_VAR_LONG_FLAG, _ENV_VAR_SHORT_FLAG)
            value_index = i + 1 if attached else i + 2
            if value_index < len(out):
                out[value_index] = REDACTED_VALUE
            i = value_index + 1
            continue
        if token == _SECRET_FLAG or token.startswith(_SECRET_FLAG + "="):
            value_index = i + 1 if "=" in token else i + 2
            if value_index < len(out):
                out[value_index] = REDACTED_VALUE
            i = value_index + 1
            continue
        i += 1
    # ``iris job run`` accepts the command without ``--``, so the sensitive-flag
    # rule covers the whole argv. It may also hide a --secret NAME; that fails closed.
    return redact_command_argv(out)


_ASSIGNMENT_RE = re.compile(r"""\b([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|'[^']*'|\S+)""")


def redact_script(script: str) -> str:
    """Redact a shell script or command string for long-term storage.

    The value of a ``NAME=value`` assignment whose NAME looks sensitive
    (``export WANDB_API_KEY=...``) is replaced, and the rest goes through the
    shared string redactor.
    """

    def assignment(match: re.Match[str]) -> str:
        name = match.group(1)
        return f"{name}={REDACTED_VALUE}" if is_sensitive_key_name(name) else match.group(0)

    return redact_string(_ASSIGNMENT_RE.sub(assignment, script))
