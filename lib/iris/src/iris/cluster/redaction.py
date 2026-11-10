# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-specific redaction wrappers around :mod:`rigging.redaction`.

The actual secret-pattern matching lives in ``rigging``; this module only
adapts iris-specific shapes (the ``LaunchJobRequest`` proto and Click argv
for ``iris job run``) into a form the shared redactor can consume.
"""

import re
import shlex

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


# Names-only redaction for command lines and scripts kept in long-term storage.
# Fail-closed: only command words, flag names and assignment names survive.
NAMES_ONLY_REDACTED = "<redacted>"
NAMES_ONLY_UNPARSED = "<unparsed>"
_SHELL_SEPARATORS = frozenset({"&&", "||", ";", "|", "&", "--"})
_ASSIGNMENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def _names_only(tokens: list[str]) -> list[str]:
    """Keep each command word, flag name and assignment name; replace every other token."""
    out: list[str] = []
    command_word_next = True
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            out.append(token)
            command_word_next = True
            continue
        assignment = _ASSIGNMENT_NAME_RE.match(token)
        if assignment is not None:
            out.append(assignment.group(0))
        elif token.startswith("--"):
            out.append(token.split("=", 1)[0])
        elif token.startswith("-") and token != "-":
            # A short flag can carry its value attached (-pPASSWORD, -eKEY).
            out.append(token[:2])
        elif command_word_next:
            out.append(token)
            command_word_next = False
        else:
            out.append(NAMES_ONLY_REDACTED)
    return out


def names_only_argv(argv: list[str]) -> list[str]:
    """Reduce an already split command line to its command words, flag names and assignment names.

    ``python train.py --hf-token X --lr 3e-4`` becomes
    ``python <redacted> --hf-token <redacted> --lr <redacted>``. A ``bash -c``
    body is one argument, so it is redacted whole.
    """
    return _names_only(list(argv))


def names_only_script(script: str) -> str:
    """Reduce a shell script or command string the same way, line by line.

    Each line is tokenized with :mod:`shlex`; a line that does not parse becomes
    ``<unparsed>``.
    """
    lines = []
    for line in script.splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            lines.append(NAMES_ONLY_UNPARSED)
            continue
        lines.append(" ".join(_names_only(tokens)))
    return "\n".join(lines)
