# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona provider causes expressed through the shared task error schema."""

from typing import NotRequired, TypedDict

from daytona import DaytonaConflictError, DaytonaConnectionError, DaytonaError
from rigging.task_errors import ErrorIdentity, RetryDisposition, exception_identity

from shellbox.task_errors import ERROR_CATEGORIES

DAYTONA_DELETE_OPERATION = "delete"


def daytona_error_identity(error: Exception, *, operation: str) -> ErrorIdentity:
    """Classify SDK failures and eligible deletion conflicts."""
    identity = exception_identity(error, operation=operation, categories=ERROR_CATEGORIES)
    if identity.exception_type == "SandboxCapacityError":
        return identity.model_copy(update={"retry_disposition": RetryDisposition.TRANSIENT})
    if identity.retry_disposition is RetryDisposition.PERMANENT:
        return identity
    if (isinstance(error, DaytonaConnectionError) and identity.provider_status_code is None) or (
        isinstance(error, DaytonaConflictError)
        and operation == DAYTONA_DELETE_OPERATION
        and identity.provider_status_code is None
    ):
        return identity.model_copy(update={"retry_disposition": RetryDisposition.TRANSIENT})
    return identity


DAYTONA_DIAGNOSTIC_CALLS = frozenset(
    {
        ("daytona._async.daytona", "create"),
        ("daytona._async.daytona", "_create"),
        ("daytona._async.daytona", "delete"),
        ("daytona._async.snapshot", "get"),
        ("daytona._async.snapshot", "create"),
        ("daytona._async.sandbox", "wait_for_sandbox_start"),
        ("daytona._async.process", "exec"),
        ("daytona._async.filesystem", "upload_file_stream"),
        ("daytona._async.filesystem", "download_file"),
        ("daytona_api_client_async.api.sandbox_api", "create_sandbox"),
        ("daytona_api_client_async.api.sandbox_api", "get_toolbox_proxy_url"),
        ("daytona_api_client_async.api.sandbox_api", "get_sandbox"),
    }
)
DAYTONA_RATE_LIMIT_HEADERS = frozenset(
    f"{prefix}-{throttler}"
    for prefix in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after")
    for throttler in ("anonymous", "authenticated", "sandbox-create", "sandbox-lifecycle")
) | {"retry-after"}


class DaytonaErrorDiagnostics(TypedDict):
    provider_sdk_call: str
    provider_rate_limits: NotRequired[dict[str, int]]


def daytona_error_diagnostics(error: DaytonaError) -> DaytonaErrorDiagnostics:
    """Return allowlisted SDK call and rate-limit metadata without response text."""
    diagnostics: DaytonaErrorDiagnostics = {"provider_sdk_call": "unknown"}
    rate_limits = {
        name.lower(): int(value)
        for name, value in (error.headers or {}).items()
        if name.lower() in DAYTONA_RATE_LIMIT_HEADERS
        and isinstance(value, str)
        and value.isascii()
        and value.isdecimal()
        and len(value) <= 12
    }
    if rate_limits:
        diagnostics["provider_rate_limits"] = rate_limits
    # This identifies an SDK call, not the HTTP phase of a compound operation.
    nested_error: BaseException | None = error
    seen = set()
    while nested_error is not None and id(nested_error) not in seen:
        seen.add(id(nested_error))
        traceback = nested_error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            call = (frame.f_globals.get("__name__"), frame.f_code.co_name)
            if call in DAYTONA_DIAGNOSTIC_CALLS:
                diagnostics["provider_sdk_call"] = ".".join(call)
            traceback = traceback.tb_next
        # The SDK translates HTTP errors with `raise ... from None`.
        nested_error = nested_error.__cause__ or nested_error.__context__
    return diagnostics
