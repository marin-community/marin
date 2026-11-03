"""Stable error taxonomy for Harbor trial failures.

Harbor persists every trial failure as an :class:`~harbor.models.trial.result.ExceptionInfo`
carrying the exception's type name (``type(exc).__name__``). Downstream consumers
such as the MarinSkyRL rollout adapter receive that serialized string, not the
original Python exception, so they classify by name against locally maintained
sets. That couples every consumer to Harbor's exception names and silently goes
stale when Harbor adds a new failure mode.

This module is the shared contract. It maps the stable exception identifiers
Harbor persists to semantic categories a consumer can translate into
training-signal treatment *without importing the full Harbor runtime*. Harbor
owns the taxonomy for a given release; consumers pin an immutable
``harbor-config`` revision and inherit the complete mapping.

Category semantics:

* :attr:`ErrorCategory.INFRASTRUCTURE` -- the evaluated model never produced a
  valid answer because Harbor, the sandbox, or the verifier failed. Exclude from
  the RL baseline (no gradient contribution).
* :attr:`ErrorCategory.AGENT` -- a real failure attributable to agent behavior;
  include in the baseline with reward zero so the policy learns to avoid it.
* :attr:`ErrorCategory.PASSTHROUGH` -- the agent produced partial or real work
  and a verifier result may still be usable; use the verifier reward.
* :attr:`ErrorCategory.UNKNOWN` -- no default decision; the consumer must apply
  an explicit fallback policy and surface the category rather than silently
  converting it to training signal.

Retryability is a separate axis (an infrastructure failure may be retried at
the job level while still being excluded from the baseline), so it is exposed
via :func:`is_resume_retryable` rather than folded into the category enum.
"""

from enum import StrEnum

__all__ = [
    "ErrorCategory",
    "error_category",
    "errors_by_category",
    "is_resume_retryable",
    "known_error_types",
    "resume_retryable_error_types",
]


class ErrorCategory(StrEnum):
    """Semantic treatment for a persisted Harbor exception."""

    INFRASTRUCTURE = "infrastructure"
    AGENT = "agent"
    PASSTHROUGH = "passthrough"
    UNKNOWN = "unknown"


_ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    # --- Harbor context-management infrastructure (the gap this closes) ---
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
