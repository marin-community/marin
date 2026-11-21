# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task error types and categories shared by multiple producers.

Frameworks own their remaining names. Compose their maps with this common map;
no framework exception classes are imported here.
"""

from collections.abc import Mapping
from enum import StrEnum
from ssl import SSLCertVerificationError
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ErrorCategory(StrEnum):
    """Semantic category for a persisted task exception."""

    INFRASTRUCTURE = "infrastructure"  # Outside the agent; SkyRL excludes the sample.
    AGENT = "agent"  # Attributable to the agent; SkyRL assigns zero reward.
    PASSTHROUGH = "passthrough"  # Usable work remains; SkyRL keeps the verifier reward.
    UNKNOWN = "unknown"  # Requires the consumer's configured fallback.


API_CONNECTION_ERROR = "APIConnectionError"
API_TIMEOUT_ERROR = "APITimeoutError"
AGENT_SETUP_TIMEOUT_ERROR = "AgentSetupTimeoutError"
AGENT_TIMEOUT_ERROR = "AgentTimeoutError"
CONTEXT_LENGTH_EXCEEDED_ERROR = "ContextLengthExceededError"
ENVIRONMENT_START_TIMEOUT_ERROR = "EnvironmentStartTimeoutError"
INTERNAL_SERVER_ERROR = "InternalServerError"
REWARD_FILE_EMPTY_ERROR = "RewardFileEmptyError"
REWARD_FILE_NOT_FOUND_ERROR = "RewardFileNotFoundError"
TRIAL_TIMEOUT_ERROR = "TrialTimeoutError"
VERIFIER_OUTPUT_PARSE_ERROR = "VerifierOutputParseError"
VERIFIER_RUNTIME_ERROR = "VerifierRuntimeError"
VERIFIER_TIMEOUT_ERROR = "VerifierTimeoutError"


SHARED_ERROR_CATEGORIES = MappingProxyType(
    {
        API_CONNECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        API_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        AGENT_SETUP_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        AGENT_TIMEOUT_ERROR: ErrorCategory.AGENT,
        CONTEXT_LENGTH_EXCEEDED_ERROR: ErrorCategory.AGENT,
        ENVIRONMENT_START_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        INTERNAL_SERVER_ERROR: ErrorCategory.INFRASTRUCTURE,
        REWARD_FILE_EMPTY_ERROR: ErrorCategory.INFRASTRUCTURE,
        REWARD_FILE_NOT_FOUND_ERROR: ErrorCategory.INFRASTRUCTURE,
        TRIAL_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        VERIFIER_OUTPUT_PARSE_ERROR: ErrorCategory.INFRASTRUCTURE,
        VERIFIER_RUNTIME_ERROR: ErrorCategory.INFRASTRUCTURE,
        VERIFIER_TIMEOUT_ERROR: ErrorCategory.UNKNOWN,
    }
)


def compose_error_categories(*sources: Mapping[str, ErrorCategory]) -> Mapping[str, ErrorCategory]:
    """Compose category maps, rejecting names owned by more than one source."""
    categories: dict[str, ErrorCategory] = {}
    for source in sources:
        duplicates = categories.keys() & source.keys()
        if duplicates:
            raise ValueError(f"Task error names have multiple owners: {sorted(duplicates)}")
        categories.update(source)
    return MappingProxyType(categories)


def error_category(exception_type: str, *, categories: Mapping[str, ErrorCategory]) -> ErrorCategory:
    return categories.get(exception_type, ErrorCategory.UNKNOWN)


def errors_by_category(category: ErrorCategory, *, categories: Mapping[str, ErrorCategory]) -> frozenset[str]:
    return frozenset(name for name, value in categories.items() if value == category)


def known_error_types(*, categories: Mapping[str, ErrorCategory]) -> frozenset[str]:
    return frozenset(categories)


DEFAULT_RETRY_EXCLUSIONS = frozenset(
    {
        AGENT_TIMEOUT_ERROR,
        VERIFIER_TIMEOUT_ERROR,
        REWARD_FILE_NOT_FOUND_ERROR,
        REWARD_FILE_EMPTY_ERROR,
        VERIFIER_OUTPUT_PARSE_ERROR,
    }
)


class RetrySettings(BaseModel):
    """Retry budget and name filters shared by task execution paths."""

    max_retries: int = Field(default=0, ge=0)
    include_exceptions: set[str] | None = Field(default=None, description="No name inclusion filter when None")
    exclude_exceptions: set[str] | None = Field(
        default_factory=lambda: set(DEFAULT_RETRY_EXCLUSIONS),
        description="Excluded names take precedence over included names",
    )
    include_categories: set[ErrorCategory] | None = None
    exclude_categories: set[ErrorCategory] = Field(default_factory=set)
    wait_multiplier: float = Field(default=1.0, ge=1, allow_inf_nan=False)
    min_wait_sec: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    max_wait_sec: float = Field(default=60.0, ge=0, allow_inf_nan=False)


class RetryDisposition(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


_TRANSIENT_HTTP_STATUS = frozenset({408, 429, 502, 503, 504})
_PERMANENT_HTTP_STATUS = frozenset({400, 401, 403, 405, 422})


def http_retry_disposition(status_code: int | None) -> RetryDisposition:
    """Classify a provider status without using its response text."""
    if status_code in _TRANSIENT_HTTP_STATUS:
        return RetryDisposition.TRANSIENT
    if status_code in _PERMANENT_HTTP_STATUS:
        return RetryDisposition.PERMANENT
    return RetryDisposition.UNKNOWN


SHARED_RETRYABLE_ERROR_TYPES = frozenset(
    {
        API_CONNECTION_ERROR,
        API_TIMEOUT_ERROR,
        INTERNAL_SERVER_ERROR,
        ENVIRONMENT_START_TIMEOUT_ERROR,
        AGENT_SETUP_TIMEOUT_ERROR,
        TRIAL_TIMEOUT_ERROR,
    }
)


def record_retry_disposition(
    exception_type: str,
    cause_type: str | None,
    provider_status_code: int | None,
    *,
    retryable_error_types: frozenset[str] = SHARED_RETRYABLE_ERROR_TYPES,
) -> RetryDisposition:
    """Classify records without a disposition, honoring provider status and cause boundaries."""
    if provider_status_code is not None:
        return http_retry_disposition(provider_status_code)
    if cause_type is None and exception_type in retryable_error_types:
        return RetryDisposition.TRANSIENT
    return RetryDisposition.UNKNOWN


class ErrorIdentity(BaseModel):
    """Persisted task failure identity, independent of training treatment."""

    model_config = ConfigDict(frozen=True)

    exception_type: str
    category: ErrorCategory = Field(
        default_factory=lambda data: error_category(data["exception_type"], categories=SHARED_ERROR_CATEGORIES)
    )
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    cause_type: str | None = None
    operation: str | None = None
    provider_status_code: int | None = None
    retry_disposition: RetryDisposition = Field(
        default_factory=lambda data: record_retry_disposition(
            data["exception_type"], data["cause_type"], data["provider_status_code"]
        )
    )


@runtime_checkable
class HttpStatusError(Protocol):
    """Provider exceptions that expose a response status independently of text."""

    @property
    def status_code(self) -> int | None: ...


def exception_identity(
    error: BaseException,
    *,
    operation: str | None = None,
    categories: Mapping[str, ErrorCategory] = SHARED_ERROR_CATEGORIES,
    retryable_error_types: frozenset[str] = SHARED_RETRYABLE_ERROR_TYPES,
) -> ErrorIdentity:
    """Capture safe exception names and retryability from provider causes."""
    cause = error.__cause__ or (error.__context__ if isinstance(error, HttpStatusError) else None)
    disposition = RetryDisposition.UNKNOWN
    status_code = None
    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HttpStatusError):
            status = current.status_code
            if isinstance(status, int) and 100 <= status <= 599:
                status_code = int(status)
                disposition = http_retry_disposition(status)
                break
            # SDKs can translate transport exceptions with `raise ... from None`.
            current = current.__cause__ or current.__context__
            continue
        if isinstance(
            current,
            (
                PermissionError,
                FileNotFoundError,
                SSLCertVerificationError,
                httpx.LocalProtocolError,
                httpx.UnsupportedProtocol,
            ),
        ):
            disposition = RetryDisposition.PERMANENT
            break
        if isinstance(
            current,
            (TimeoutError, ConnectionError, httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError),
        ):
            disposition = RetryDisposition.TRANSIENT
        current = current.__cause__
    exception_type = type(error).__name__
    cause_type = None if cause is None else type(cause).__name__
    if disposition is RetryDisposition.UNKNOWN:
        disposition = record_retry_disposition(
            exception_type, cause_type, status_code, retryable_error_types=retryable_error_types
        )
    return ErrorIdentity(
        exception_type=exception_type,
        category=error_category(exception_type, categories=categories),
        cause_type=cause_type,
        operation=operation,
        retry_disposition=disposition,
        provider_status_code=status_code,
    )


def retry_delay(failure: ErrorIdentity | None, settings: RetrySettings, retries: int) -> float | None:
    """Return the shared retry delay, or None for a terminal attempt.

    Exclusions match the outer name, cause name, or category and always win.
    Permanent failures cannot be selected for retry. Unknown failures require
    an explicit name or category inclusion. Transient failures retry by default
    and must match an inclusion when an include filter is configured.
    """
    if failure is None or retries >= settings.max_retries:
        return None
    names = {failure.exception_type, failure.cause_type} - {None}
    category = failure.category
    if names.intersection(settings.exclude_exceptions or ()) or category in settings.exclude_categories:
        return None
    disposition = failure.retry_disposition
    if disposition is RetryDisposition.PERMANENT:
        return None
    selected = bool(names.intersection(settings.include_exceptions or ())) or category in (
        settings.include_categories or ()
    )
    filtered = settings.include_exceptions is not None or settings.include_categories is not None
    if (filtered and not selected) or (disposition is RetryDisposition.UNKNOWN and not selected):
        return None
    return min(settings.min_wait_sec * settings.wait_multiplier**retries, settings.max_wait_sec)
