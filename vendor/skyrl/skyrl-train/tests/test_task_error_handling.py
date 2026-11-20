"""Freeze terminal training treatment while sharing the task error schema."""

import pytest

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.task_error_handling import ErrorHandlingConfig, classify_exception_type


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ContextManagementInfrastructureError", ErrorTreatment.MASK),
        ("ArtifactUploadTimeoutError", ErrorTreatment.MASK),
        ("ArtifactWriterBacklogError", ErrorTreatment.MASK),
        ("EnvironmentStartTimeoutError", ErrorTreatment.MASK),
        ("TrialTimeoutError", ErrorTreatment.MASK),
        ("SandboxBuildFailedError", ErrorTreatment.MASK),
        ("SnapshotQuotaExceeded", ErrorTreatment.MASK),
        ("HealthcheckError", ErrorTreatment.MASK),
        ("DaytonaSandboxStopError", ErrorTreatment.MASK),
        ("BridgeOutageError", ErrorTreatment.MASK),
        ("EnrootMemoryLimitExceededError", ErrorTreatment.MASK),
        ("MissingExtraError", ErrorTreatment.MASK),
        ("AgentKilledBySignalError", ErrorTreatment.MASK),
        ("AgentSetupTimeoutError", ErrorTreatment.MASK),
        ("ModelAuthenticationError", ErrorTreatment.MASK),
        ("ContextBudgetExceededError", ErrorTreatment.MASK),
        ("LLMRequestTimeoutError", ErrorTreatment.MASK),
        ("OpenAITransportConnectTimeoutError", ErrorTreatment.MASK),
        ("TmuxBatchProtocolError", ErrorTreatment.MASK),
        ("TmuxCommandError", ErrorTreatment.MASK),
        ("TmuxSessionEndedError", ErrorTreatment.MASK),
        ("DownloadVerifierDirError", ErrorTreatment.MASK),
        ("RewardFileNotFoundError", ErrorTreatment.MASK),
        ("RewardFileEmptyError", ErrorTreatment.MASK),
        ("VerifierRuntimeError", ErrorTreatment.MASK),
        ("VerifierOutputParseError", ErrorTreatment.MASK),
        ("AddTestsDirError", ErrorTreatment.MASK),
        ("ArtifactPersistenceError", ErrorTreatment.MASK),
        ("ArtifactWriteTimeoutError", ErrorTreatment.MASK),
        ("ContainerRegistryThrottledError", ErrorTreatment.MASK),
        ("DaytonaProcessError", ErrorTreatment.MASK),
        ("PhaseCancellationError", ErrorTreatment.MASK),
        ("SandboxDiskFullError", ErrorTreatment.MASK),
        ("Tau3RuntimeInfrastructureError", ErrorTreatment.MASK),
        ("AgentTimeoutError", ErrorTreatment.ZERO),
        ("ModelInitiatedTmuxFailure", ErrorTreatment.ZERO),
        ("AgentDiskFullError", ErrorTreatment.ZERO),
        ("ContextLengthExceededError", ErrorTreatment.ZERO),
        ("NonZeroAgentExitCodeError", ErrorTreatment.ZERO),
        ("OutputLengthExceededError", ErrorTreatment.PASSTHROUGH),
        ("TurnCapExhaustedError", ErrorTreatment.PASSTHROUGH),
        ("MalformedModelResponseError", ErrorTreatment.PASSTHROUGH),
        ("TrialNotScoredError", ErrorTreatment.ZERO),
        ("VerificationNotCompletedError", ErrorTreatment.ZERO),
        ("VerifierTimeoutError", ErrorTreatment.ZERO),
        ("RuntimeServiceError", ErrorTreatment.MASK),
        ("RuntimeLogCollectionError", ErrorTreatment.MASK),
        ("EnvironmentStopError", ErrorTreatment.MASK),
        ("DaytonaError", ErrorTreatment.MASK),
        ("DaytonaDockerSetupError", ErrorTreatment.MASK),
        ("InternalServerError", ErrorTreatment.MASK),
        ("APIConnectionError", ErrorTreatment.MASK),
        ("APITimeoutError", ErrorTreatment.MASK),
        ("TaskMachineError", ErrorTreatment.MASK),
        ("TemplateError", ErrorTreatment.MASK),
        ("DaytonaConnectionError", ErrorTreatment.MASK),
        ("DaytonaConflictError", ErrorTreatment.MASK),
        ("MachineTerminated", ErrorTreatment.MASK),
        ("AgentCommandOutputLostError", ErrorTreatment.ZERO),
        ("DaytonaOutputUnavailable", ErrorTreatment.MASK),
        ("TmuxExecTimeoutError", ErrorTreatment.MASK),
    ],
)
def test_terminal_error_treatment_contract(name, expected):
    assert classify_exception_type(name, ErrorHandlingConfig()) is expected


def test_training_override_masks_an_agent_timeout():
    config = ErrorHandlingConfig(mask_exceptions=frozenset({"AgentTimeoutError"}))
    assert classify_exception_type("AgentTimeoutError", config) is ErrorTreatment.MASK
    assert classify_exception_type("AgentTimeoutError", ErrorHandlingConfig()) is ErrorTreatment.ZERO
