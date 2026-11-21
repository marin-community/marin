"""Harbor categories and fresh-job resume policy."""

from types import MappingProxyType

from rigging.task_errors import ErrorCategory, SHARED_ERROR_CATEGORIES, compose_error_categories
from shellbox.task_errors import SHELLBOX_ERROR_CATEGORIES


HARBOR_ERROR_CATEGORIES = MappingProxyType(
    {
        "AddTestsDirError": ErrorCategory.INFRASTRUCTURE,
        "AgentCommandOutputLostError": ErrorCategory.AGENT,
        "AgentDiskFullError": ErrorCategory.AGENT,
        "AgentKilledBySignalError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactPersistenceError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactUploadTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactWriteTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "ArtifactWriterBacklogError": ErrorCategory.INFRASTRUCTURE,
        "BridgeOutageError": ErrorCategory.INFRASTRUCTURE,
        "ContainerRegistryThrottledError": ErrorCategory.INFRASTRUCTURE,
        "ContextBudgetExceededError": ErrorCategory.INFRASTRUCTURE,
        "ContextManagementInfrastructureError": ErrorCategory.INFRASTRUCTURE,
        "DownloadVerifierDirError": ErrorCategory.INFRASTRUCTURE,
        "EnrootMemoryLimitExceededError": ErrorCategory.INFRASTRUCTURE,
        "EnvironmentStopError": ErrorCategory.INFRASTRUCTURE,
        "HealthcheckError": ErrorCategory.INFRASTRUCTURE,
        "LLMRequestTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "MalformedModelResponseError": ErrorCategory.PASSTHROUGH,
        "MissingExtraError": ErrorCategory.INFRASTRUCTURE,
        "ModelAuthenticationError": ErrorCategory.INFRASTRUCTURE,
        "ModelInitiatedTmuxFailure": ErrorCategory.AGENT,
        "NonZeroAgentExitCodeError": ErrorCategory.AGENT,
        "OpenAITransportConnectTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "OutputLengthExceededError": ErrorCategory.PASSTHROUGH,
        "PhaseCancellationError": ErrorCategory.INFRASTRUCTURE,
        "RuntimeLogCollectionError": ErrorCategory.INFRASTRUCTURE,
        "RuntimeServiceError": ErrorCategory.INFRASTRUCTURE,
        "SandboxBuildFailedError": ErrorCategory.INFRASTRUCTURE,
        "SandboxDiskFullError": ErrorCategory.INFRASTRUCTURE,
        "Tau3RuntimeInfrastructureError": ErrorCategory.INFRASTRUCTURE,
        "TmuxBatchProtocolError": ErrorCategory.INFRASTRUCTURE,
        "TmuxCommandError": ErrorCategory.INFRASTRUCTURE,
        "TmuxExecTimeoutError": ErrorCategory.INFRASTRUCTURE,
        "TmuxSessionEndedError": ErrorCategory.INFRASTRUCTURE,
        "TrialNotScoredError": ErrorCategory.UNKNOWN,
        "TurnCapExhaustedError": ErrorCategory.PASSTHROUGH,
        "VerificationNotCompletedError": ErrorCategory.UNKNOWN,
    }
)

ERROR_CATEGORIES = compose_error_categories(
    SHARED_ERROR_CATEGORIES, SHELLBOX_ERROR_CATEGORIES, HARBOR_ERROR_CATEGORIES
)

RESUME_RETRYABLE_ERROR_TYPES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ArtifactPersistenceError",
        "ArtifactUploadTimeoutError",
        "ArtifactWriteTimeoutError",
        "ArtifactWriterBacklogError",
        "CancelledError",
        "ContainerRegistryThrottledError",
        "ContextBudgetExceededError",
        "ContextManagementInfrastructureError",
        "DaytonaConflictError",
        "DaytonaConnectionError",
        "DaytonaDockerSetupError",
        "DaytonaOutputUnavailable",
        "DaytonaProcessError",
        "EnvironmentStopError",
        "InternalServerError",
        "LLMRequestTimeoutError",
        "MachineTerminated",
        "OpenAITransportConnectTimeoutError",
        "PhaseCancellationError",
        "RuntimeLogCollectionError",
        "RuntimeServiceError",
        "SandboxBuildFailedError",
        "Tau3RuntimeInfrastructureError",
        "TmuxBatchProtocolError",
        "TmuxCommandError",
        "TmuxExecTimeoutError",
        "TmuxSessionEndedError",
        "TrialTimeoutError",
    }
)


def is_resume_retryable(exception_type: str) -> bool:
    """Whether a persisted exception should be re-run on a fresh job resume."""
    return exception_type in RESUME_RETRYABLE_ERROR_TYPES
