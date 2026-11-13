# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-specific redaction wrappers around :mod:`rigging.redaction`.

The actual secret-pattern matching lives in ``rigging``; this module only
adapts iris-specific shapes (the ``LaunchJobRequest`` proto, Click argv for
``iris job run`` and the environment inputs copied to the controller's Finelog
rows) into a form the shared redactor can consume.
"""

import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import cast

from rigging.provenance import LAUNCH_PROVENANCE_ENV, Provenance
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


def _sensitive_flag_name(token: str) -> str | None:
    """The name of a ``-flag`` or ``--flag[=value]`` token, if it names a secret."""
    if not token.startswith("-"):
        return None
    name = token.lstrip("-").split("=", 1)[0]
    return name if name and is_sensitive_key_name(name) else None


def strip_url_userinfo(url: str) -> str:
    """*url* without its ``user:password@`` part, which can hold a credential.

    Only the text up to the last ``@`` of the authority is removed. Nothing else
    is parsed, so a malformed port or host is kept as is and this never raises.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url
    authority_end = min((index for index in map(rest.find, "/?#") if index >= 0), default=len(rest))
    authority = rest[:authority_end]
    if "@" not in authority:
        return url
    return f"{scheme}://{authority.rpartition('@')[2]}{rest[authority_end:]}"


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

    Any other flag whose name matches that regex (``--api-key VALUE``,
    ``--password=VALUE``) also has its value replaced with REDACTED_VALUE.
    """
    out = list(argv)
    n = len(out)
    i = 0
    while i < n:
        tok = out[i]
        next_tok = out[i + 1] if i + 1 < n else None
        key = _env_var_flag_key(tok, next_tok)
        if key is None:
            # A flag named like a secret (``--api-key VALUE``,
            # ``--password=VALUE``) has its value masked.
            if _sensitive_flag_name(tok) is not None:
                if "=" in tok:
                    out[i] = tok.split("=", 1)[0] + "=" + REDACTED_VALUE
                elif next_tok is not None and not next_tok.startswith("-"):
                    out[i + 1] = REDACTED_VALUE
                    i += 1
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


def _redacted_provenance(raw: str) -> dict[str, object] | None:
    """The launch provenance's fields, each redacted; ``None`` if *raw* does not parse."""
    try:
        provenance = Provenance.from_json(raw)
    except (ValueError, KeyError, TypeError):
        return None
    fields = dataclasses.asdict(provenance)
    if provenance.git_remote is not None:
        fields["git_remote"] = strip_url_userinfo(provenance.git_remote)
    fields["command_line"] = redact_submit_argv(list(provenance.command_line))
    return cast(dict[str, object], redact_value(fields))


# The environment variables whose values the controller copies into its Finelog
# rows (``iris.controller_db__job_config.env_inputs``), each with its redactor.
# Every other variable, and every secret, is dropped.
CONTROLLER_ROW_ENV_INPUTS: dict[str, Callable[[str], object | None]] = {
    LAUNCH_PROVENANCE_ENV: _redacted_provenance,
}


def redact_env_inputs(env_vars: Mapping[str, str]) -> str | None:
    """The :data:`CONTROLLER_ROW_ENV_INPUTS` among *env_vars*, redacted, as a JSON object.

    ``MARIN_PROVENANCE`` is stored parsed, as its ``Provenance`` fields: the
    remote URL loses its userinfo, the command line is redacted like a submit
    argv, and every value then goes through the shared value redactor. A value
    that does not parse is dropped. ``None`` when no allowed key is set.
    """
    inputs: dict[str, object] = {}
    for key, redact in CONTROLLER_ROW_ENV_INPUTS.items():
        raw = env_vars.get(key)
        value = redact(raw) if raw is not None else None
        if value is not None:
            inputs[key] = value
    return json.dumps(inputs, sort_keys=True) if inputs else None
