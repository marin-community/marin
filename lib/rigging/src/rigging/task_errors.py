# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task failure categories and retry settings.

Categories describe the source of a failure. Consumers choose training treatment
separately from retry settings. Persisted exception names are stable identifiers.
"""

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ErrorCategory(StrEnum):
    """Semantic category for a persisted task exception."""

    INFRASTRUCTURE = "infrastructure"
    AGENT = "agent"
    PASSTHROUGH = "passthrough"
    UNKNOWN = "unknown"


_ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    # --- Task and context-management infrastructure ---
    "TaskMachineError": ErrorCategory.INFRASTRUCTURE,
    "TemplateError": ErrorCategory.INFRASTRUCTURE,
    "ContextManagementInfrastructureError": ErrorCategory.INFRASTRUCTURE,
    # --- Artifact persistence infrastructure ---
    "ArtifactPersistenceError": ErrorCategory.INFRASTRUCTURE,
    "ArtifactUploadTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "ArtifactWriteTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "ArtifactWriterBacklogError": ErrorCategory.INFRASTRUCTURE,
    # --- Environment / sandbox infrastructure ---
    "EnvironmentStartTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "TrialTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "PhaseCancellationError": ErrorCategory.INFRASTRUCTURE,
    "RuntimeServiceError": ErrorCategory.INFRASTRUCTURE,
    "RuntimeLogCollectionError": ErrorCategory.INFRASTRUCTURE,
    "EnvironmentStopError": ErrorCategory.INFRASTRUCTURE,
    "SandboxBuildFailedError": ErrorCategory.INFRASTRUCTURE,
    "SandboxDiskFullError": ErrorCategory.INFRASTRUCTURE,
    "ContainerRegistryThrottledError": ErrorCategory.INFRASTRUCTURE,
    "SnapshotQuotaExceeded": ErrorCategory.INFRASTRUCTURE,
    "HealthcheckError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaConnectionError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaOutputUnavailable": ErrorCategory.INFRASTRUCTURE,
    "DaytonaProcessError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaDockerSetupError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaSandboxStopError": ErrorCategory.INFRASTRUCTURE,
    "BridgeOutageError": ErrorCategory.INFRASTRUCTURE,
    "EnrootMemoryLimitExceededError": ErrorCategory.INFRASTRUCTURE,
    "MissingExtraError": ErrorCategory.INFRASTRUCTURE,
    # --- Agent lifecycle infrastructure (model never produced a real answer) ---
    "AgentKilledBySignalError": ErrorCategory.INFRASTRUCTURE,
    "AgentSetupTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "ModelAuthenticationError": ErrorCategory.INFRASTRUCTURE,
    "ContextBudgetExceededError": ErrorCategory.INFRASTRUCTURE,
    "LLMRequestTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "OpenAITransportConnectTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "InternalServerError": ErrorCategory.INFRASTRUCTURE,
    "APIConnectionError": ErrorCategory.INFRASTRUCTURE,
    "APITimeoutError": ErrorCategory.INFRASTRUCTURE,
    "TmuxBatchProtocolError": ErrorCategory.INFRASTRUCTURE,
    "TmuxCommandError": ErrorCategory.INFRASTRUCTURE,
    "TmuxExecTimeoutError": ErrorCategory.INFRASTRUCTURE,
    "TmuxSessionEndedError": ErrorCategory.INFRASTRUCTURE,
    "Tau3RuntimeInfrastructureError": ErrorCategory.INFRASTRUCTURE,
    # --- Verifier infrastructure ---
    "DownloadVerifierDirError": ErrorCategory.INFRASTRUCTURE,
    "RewardFileNotFoundError": ErrorCategory.INFRASTRUCTURE,
    "RewardFileEmptyError": ErrorCategory.INFRASTRUCTURE,
    "VerifierRuntimeError": ErrorCategory.INFRASTRUCTURE,
    "VerifierOutputParseError": ErrorCategory.INFRASTRUCTURE,
    "AddTestsDirError": ErrorCategory.INFRASTRUCTURE,
    # --- Agent-attributable failures (reward zero) ---
    "AgentTimeoutError": ErrorCategory.AGENT,
    "AgentDiskFullError": ErrorCategory.AGENT,
    "ContextLengthExceededError": ErrorCategory.AGENT,
    "NonZeroAgentExitCodeError": ErrorCategory.AGENT,
    "ModelInitiatedTmuxFailure": ErrorCategory.AGENT,
    # --- Partial / real work; a verifier result may still be usable ---
    "OutputLengthExceededError": ErrorCategory.PASSTHROUGH,
    "TurnCapExhaustedError": ErrorCategory.PASSTHROUGH,
    "MalformedModelResponseError": ErrorCategory.PASSTHROUGH,
    # --- No safe default; force an explicit consumer decision ---
    "TrialNotScoredError": ErrorCategory.UNKNOWN,
    "VerificationNotCompletedError": ErrorCategory.UNKNOWN,
    "VerifierTimeoutError": ErrorCategory.UNKNOWN,
}

# Exception types whose on-disk trial result should be RE-RUN on a fresh
# ``harbor jobs resume`` (a whole new job invocation), rather than kept as a
# terminal error result. This is the resume-level retry decision and is
# deliberately DISTINCT from the in-trial retry classification: it is keyed on
# the persisted exception name and covers only transient, job-level failures.
_RESUME_RETRYABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "CancelledError",
        "ArtifactPersistenceError",
        "ArtifactUploadTimeoutError",
        "ArtifactWriteTimeoutError",
        "ArtifactWriterBacklogError",
        "SandboxBuildFailedError",
        "ContainerRegistryThrottledError",
        "ContextBudgetExceededError",
        "ContextManagementInfrastructureError",
        "DaytonaConnectionError",
        "DaytonaOutputUnavailable",
        "DaytonaProcessError",
        "DaytonaDockerSetupError",
        "LLMRequestTimeoutError",
        "OpenAITransportConnectTimeoutError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "PhaseCancellationError",
        "RuntimeServiceError",
        "RuntimeLogCollectionError",
        "EnvironmentStopError",
        "TmuxBatchProtocolError",
        "TmuxCommandError",
        "TmuxExecTimeoutError",
        "TmuxSessionEndedError",
        "Tau3RuntimeInfrastructureError",
        "TrialTimeoutError",
    }
)


def error_category(exception_type: str) -> ErrorCategory:
    """Return the semantic category for a persisted exception type name.

    Unrecognized identifiers resolve to :attr:`ErrorCategory.UNKNOWN` so that
    contract drift is observable rather than silently defaulted.
    """
    return _ERROR_CATEGORIES.get(exception_type, ErrorCategory.UNKNOWN)


def errors_by_category(category: ErrorCategory) -> frozenset[str]:
    """Return every persisted exception name mapped to ``category``.

    Consumers derive their default ``mask`` / ``zero`` / ``passthrough`` sets
    from this rather than duplicating Harbor exception names.
    """
    return frozenset(name for name, value in _ERROR_CATEGORIES.items() if value == category)


def is_resume_retryable(exception_type: str) -> bool:
    """Whether a persisted exception should be re-run on a fresh job resume."""
    return exception_type in _RESUME_RETRYABLE_ERROR_TYPES


def resume_retryable_error_types() -> frozenset[str]:
    """The persisted exception names that trigger a fresh-resume re-run."""
    return _RESUME_RETRYABLE_ERROR_TYPES


def known_error_types() -> frozenset[str]:
    """Every persisted exception name with an explicit category decision.

    A name absent here resolves to :attr:`ErrorCategory.UNKNOWN` via
    :func:`error_category`, but the absence itself signals contract drift;
    Harbor's contract test enforces that every persisted exception is present.
    """
    return frozenset(_ERROR_CATEGORIES)


class RetrySettings(BaseModel):
    """Retry budget and name filters shared by task execution paths."""

    max_retries: int = Field(default=0, ge=0)
    include_exceptions: set[str] | None = None
    exclude_exceptions: set[str] | None = Field(
        default_factory=lambda: {
            "AgentTimeoutError",
            "VerifierTimeoutError",
            "RewardFileNotFoundError",
            "RewardFileEmptyError",
            "VerifierOutputParseError",
        }
    )
    wait_multiplier: float = 1.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 60.0


class RetryDisposition(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


class ErrorIdentity(BaseModel):
    """Persisted task failure identity, independent of training treatment."""

    model_config = ConfigDict(frozen=True)

    exception_type: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    cause_type: str | None = None
    operation: str | None = None
    retry_disposition: RetryDisposition = RetryDisposition.UNKNOWN
    provider_status_code: int | None = None

    @property
    def category(self) -> ErrorCategory:
        return error_category(self.exception_type)


_TRANSIENT_HTTP_STATUS = frozenset({408, 429, 502, 503, 504})
_PERMANENT_HTTP_STATUS = frozenset({400, 401, 403, 404, 405, 410, 422})


def http_retry_disposition(status_code: int | None) -> RetryDisposition:
    """Classify a provider status without using its response text."""
    if status_code in _TRANSIENT_HTTP_STATUS:
        return RetryDisposition.TRANSIENT
    if status_code in _PERMANENT_HTTP_STATUS:
        return RetryDisposition.PERMANENT
    return RetryDisposition.UNKNOWN


@runtime_checkable
class HttpStatusError(Protocol):
    """Provider exceptions that expose a response status independently of text."""

    @property
    def status_code(self) -> int | None: ...


def exception_identity(error: BaseException, *, operation: str | None = None) -> ErrorIdentity:
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
            if type(status) is int and 100 <= status <= 599:
                status_code = status
                disposition = http_retry_disposition(status)
                break
            # SDKs can translate transport exceptions with `raise ... from None`.
            current = current.__cause__ or current.__context__
            continue
        if isinstance(current, (TimeoutError, ConnectionError, httpx.TransportError)):
            disposition = RetryDisposition.TRANSIENT
            break
        current = current.__cause__
    return ErrorIdentity(
        exception_type=type(error).__name__,
        cause_type=None if cause is None else type(cause).__name__,
        operation=operation,
        retry_disposition=disposition,
        provider_status_code=status_code,
    )
