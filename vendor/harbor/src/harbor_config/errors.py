"""Harbor categories and fresh-job resume policy."""

from types import MappingProxyType

from rigging.task_errors import (
    API_CONNECTION_ERROR,
    API_TIMEOUT_ERROR,
    SHARED_RETRYABLE_ERROR_TYPES,
    INTERNAL_SERVER_ERROR,
    TRIAL_TIMEOUT_ERROR,
    ErrorCategory,
    compose_error_categories,
)
from shellbox.task_errors import (
    DAYTONA_CONFLICT_ERROR,
    DAYTONA_CONNECTION_ERROR,
    DAYTONA_DOCKER_SETUP_ERROR,
    DAYTONA_OUTPUT_UNAVAILABLE,
    DAYTONA_PROCESS_ERROR,
    MACHINE_TERMINATED,
)
from shellbox.task_errors import (
    ERROR_CATEGORIES as SHELLBOX_CATEGORIES,
)

ARTIFACT_PERSISTENCE_ERROR = "ArtifactPersistenceError"
ARTIFACT_UPLOAD_TIMEOUT_ERROR = "ArtifactUploadTimeoutError"
ARTIFACT_WRITE_TIMEOUT_ERROR = "ArtifactWriteTimeoutError"
ARTIFACT_WRITER_BACKLOG_ERROR = "ArtifactWriterBacklogError"
CONTAINER_REGISTRY_THROTTLED_ERROR = "ContainerRegistryThrottledError"
CONTEXT_BUDGET_EXCEEDED_ERROR = "ContextBudgetExceededError"
CONTEXT_MANAGEMENT_INFRASTRUCTURE_ERROR = "ContextManagementInfrastructureError"
ENVIRONMENT_STOP_ERROR = "EnvironmentStopError"
LLM_REQUEST_TIMEOUT_ERROR = "LLMRequestTimeoutError"
OPEN_AI_TRANSPORT_CONNECT_TIMEOUT_ERROR = "OpenAITransportConnectTimeoutError"
PHASE_CANCELLATION_ERROR = "PhaseCancellationError"
RUNTIME_LOG_COLLECTION_ERROR = "RuntimeLogCollectionError"
RUNTIME_SERVICE_ERROR = "RuntimeServiceError"
SANDBOX_BUILD_FAILED_ERROR = "SandboxBuildFailedError"
TAU3_RUNTIME_INFRASTRUCTURE_ERROR = "Tau3RuntimeInfrastructureError"
TMUX_BATCH_PROTOCOL_ERROR = "TmuxBatchProtocolError"
TMUX_COMMAND_ERROR = "TmuxCommandError"
TMUX_EXEC_TIMEOUT_ERROR = "TmuxExecTimeoutError"
TMUX_SESSION_ENDED_ERROR = "TmuxSessionEndedError"

HARBOR_ERROR_CATEGORIES = MappingProxyType(
    {
        "AddTestsDirError": ErrorCategory.INFRASTRUCTURE,
        "AgentCommandOutputLostError": ErrorCategory.AGENT,
        "AgentDiskFullError": ErrorCategory.AGENT,
        "AgentKilledBySignalError": ErrorCategory.INFRASTRUCTURE,
        ARTIFACT_PERSISTENCE_ERROR: ErrorCategory.INFRASTRUCTURE,
        ARTIFACT_UPLOAD_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        ARTIFACT_WRITE_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        ARTIFACT_WRITER_BACKLOG_ERROR: ErrorCategory.INFRASTRUCTURE,
        "BridgeOutageError": ErrorCategory.INFRASTRUCTURE,
        CONTAINER_REGISTRY_THROTTLED_ERROR: ErrorCategory.INFRASTRUCTURE,
        CONTEXT_BUDGET_EXCEEDED_ERROR: ErrorCategory.INFRASTRUCTURE,
        CONTEXT_MANAGEMENT_INFRASTRUCTURE_ERROR: ErrorCategory.INFRASTRUCTURE,
        "DownloadVerifierDirError": ErrorCategory.INFRASTRUCTURE,
        "EnrootMemoryLimitExceededError": ErrorCategory.INFRASTRUCTURE,
        "EnvironmentSetupError": ErrorCategory.INFRASTRUCTURE,
        ENVIRONMENT_STOP_ERROR: ErrorCategory.INFRASTRUCTURE,
        "HealthcheckError": ErrorCategory.INFRASTRUCTURE,
        LLM_REQUEST_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        "MalformedModelResponseError": ErrorCategory.PASSTHROUGH,
        "MissingExtraError": ErrorCategory.INFRASTRUCTURE,
        "ModelAuthenticationError": ErrorCategory.INFRASTRUCTURE,
        "ModelInitiatedTmuxFailure": ErrorCategory.AGENT,
        "NonZeroAgentExitCodeError": ErrorCategory.AGENT,
        OPEN_AI_TRANSPORT_CONNECT_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        "OutputLengthExceededError": ErrorCategory.PASSTHROUGH,
        PHASE_CANCELLATION_ERROR: ErrorCategory.INFRASTRUCTURE,
        RUNTIME_LOG_COLLECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        RUNTIME_SERVICE_ERROR: ErrorCategory.INFRASTRUCTURE,
        SANDBOX_BUILD_FAILED_ERROR: ErrorCategory.INFRASTRUCTURE,
        "SandboxDiskFullError": ErrorCategory.INFRASTRUCTURE,
        TAU3_RUNTIME_INFRASTRUCTURE_ERROR: ErrorCategory.INFRASTRUCTURE,
        TMUX_BATCH_PROTOCOL_ERROR: ErrorCategory.INFRASTRUCTURE,
        TMUX_COMMAND_ERROR: ErrorCategory.INFRASTRUCTURE,
        TMUX_EXEC_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        TMUX_SESSION_ENDED_ERROR: ErrorCategory.INFRASTRUCTURE,
        "TrialNotScoredError": ErrorCategory.UNKNOWN,
        "TurnCapExhaustedError": ErrorCategory.PASSTHROUGH,
        "VerificationNotCompletedError": ErrorCategory.UNKNOWN,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHELLBOX_CATEGORIES, HARBOR_ERROR_CATEGORIES)

RESUME_RETRYABLE_ERROR_TYPES = frozenset(
    {
        API_CONNECTION_ERROR,
        API_TIMEOUT_ERROR,
        ARTIFACT_PERSISTENCE_ERROR,
        ARTIFACT_UPLOAD_TIMEOUT_ERROR,
        ARTIFACT_WRITE_TIMEOUT_ERROR,
        ARTIFACT_WRITER_BACKLOG_ERROR,
        "CancelledError",
        CONTAINER_REGISTRY_THROTTLED_ERROR,
        CONTEXT_BUDGET_EXCEEDED_ERROR,
        CONTEXT_MANAGEMENT_INFRASTRUCTURE_ERROR,
        DAYTONA_CONFLICT_ERROR,
        DAYTONA_CONNECTION_ERROR,
        DAYTONA_DOCKER_SETUP_ERROR,
        DAYTONA_OUTPUT_UNAVAILABLE,
        DAYTONA_PROCESS_ERROR,
        ENVIRONMENT_STOP_ERROR,
        INTERNAL_SERVER_ERROR,
        LLM_REQUEST_TIMEOUT_ERROR,
        MACHINE_TERMINATED,
        OPEN_AI_TRANSPORT_CONNECT_TIMEOUT_ERROR,
        PHASE_CANCELLATION_ERROR,
        RUNTIME_LOG_COLLECTION_ERROR,
        RUNTIME_SERVICE_ERROR,
        SANDBOX_BUILD_FAILED_ERROR,
        TAU3_RUNTIME_INFRASTRUCTURE_ERROR,
        TMUX_BATCH_PROTOCOL_ERROR,
        TMUX_COMMAND_ERROR,
        TMUX_EXEC_TIMEOUT_ERROR,
        TMUX_SESSION_ENDED_ERROR,
        TRIAL_TIMEOUT_ERROR,
    }
)


def is_resume_retryable(exception_type: str) -> bool:
    """Whether a persisted exception should be re-run on a fresh job resume."""
    return exception_type in RESUME_RETRYABLE_ERROR_TYPES


RETRYABLE_ERROR_TYPES = RESUME_RETRYABLE_ERROR_TYPES | SHARED_RETRYABLE_ERROR_TYPES
