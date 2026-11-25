# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-specific redaction wrappers around :mod:`rigging.redaction`.

The actual secret-pattern matching lives in ``rigging``; this module only
adapts iris-specific shapes (the ``LaunchJobRequest`` proto and Click argv
for ``iris job run``) into a form the shared redactor can consume.
"""

from rigging.redaction import REDACTED_VALUE, redact_env_argv, redact_value
from rigging.redaction import redact_string as redact_string

from iris.rpc import controller_pb2


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


redact_submit_argv = redact_env_argv
