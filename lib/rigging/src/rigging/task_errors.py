# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task failure identities, treatment categories, and retry decisions.

Categories describe the source of a failure. Consumers choose training treatment
separately from retry settings. Persisted exception names are stable identifiers.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
    return frozenset(
        name for name, value in _ERROR_CATEGORIES.items() if value == category
    )


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


class RetryDisposition(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorIdentity:
    """Safe, serializable identity of a task failure and its provider cause."""

    exception_type: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cause_type: str | None = None
    operation: str | None = None
    retry_disposition: RetryDisposition = RetryDisposition.UNKNOWN

    @property
    def category(self) -> ErrorCategory:
        return error_category(self.exception_type)


class RetrySettings(BaseModel):
    """Retry budget and filters shared by every task execution path."""

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
    include_categories: set[ErrorCategory] | None = None
    exclude_categories: set[ErrorCategory] = Field(default_factory=set)
    wait_multiplier: float = Field(default=1.0, ge=1)
    min_wait_sec: float = Field(default=1.0, ge=0)
    max_wait_sec: float = Field(default=60.0, ge=0)


_PERMANENT_ERROR_TYPES = {
    "AgentTimeoutError",
    "AgentDiskFullError",
    "ContextLengthExceededError",
    "NonZeroAgentExitCodeError",
    "ModelInitiatedTmuxFailure",
    "OutputLengthExceededError",
    "TurnCapExhaustedError",
    "MalformedModelResponseError",
    "ModelAuthenticationError",
    "MissingExtraError",
    "SandboxDiskFullError",
    "SnapshotQuotaExceeded",
}


def retry_disposition(failure: ErrorIdentity) -> RetryDisposition:
    """Resolve retryability without inferring it from training treatment."""
    if failure.retry_disposition is not RetryDisposition.UNKNOWN:
        return failure.retry_disposition
    if failure.exception_type in _RESUME_RETRYABLE_ERROR_TYPES:
        return RetryDisposition.TRANSIENT
    if failure.exception_type in _PERMANENT_ERROR_TYPES:
        return RetryDisposition.PERMANENT
    return RetryDisposition.UNKNOWN


def retry_delay(failure: ErrorIdentity | None, settings: RetrySettings, retries: int) -> float | None:
    """Return a bounded backoff, or None when the result must remain terminal.

    Names can match the task error or its provider cause. Exclusions take
    precedence over inclusions. Unknown failures require an explicit inclusion;
    permanent failures cannot be retried.
    """
    if failure is None or retries >= settings.max_retries:
        return None
    names = {failure.exception_type, failure.cause_type} - {None}
    if names.intersection(settings.exclude_exceptions or ()) or failure.category in settings.exclude_categories:
        return None
    disposition = retry_disposition(failure)
    if disposition is RetryDisposition.PERMANENT:
        return None
    selected = bool(
        names.intersection(settings.include_exceptions or ())
        or failure.category in (settings.include_categories or ())
    )
    if settings.include_exceptions or settings.include_categories:
        if not selected:
            return None
    if disposition is RetryDisposition.UNKNOWN and not selected:
        return None
    return min(settings.min_wait_sec * settings.wait_multiplier**retries, settings.max_wait_sec)
