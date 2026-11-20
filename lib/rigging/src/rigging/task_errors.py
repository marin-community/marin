# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task failure categories and retry settings.

Categories describe the source of a failure. Consumers choose training treatment
separately from retry settings. Persisted exception names are stable identifiers.
"""

from enum import StrEnum

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
    "DaytonaConflictError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaConnectionError": ErrorCategory.INFRASTRUCTURE,
    "DaytonaOutputUnavailable": ErrorCategory.INFRASTRUCTURE,
    "MachineTerminated": ErrorCategory.INFRASTRUCTURE,
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
    "AgentCommandOutputLostError": ErrorCategory.AGENT,
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
        "DaytonaConflictError",
        "DaytonaConnectionError",
        "DaytonaOutputUnavailable",
        "MachineTerminated",
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
