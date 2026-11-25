"""Freeze terminal training treatment while sharing the task error schema."""

import pytest

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.task_error_handling import ErrorHandlingConfig, classify_exception_type


@pytest.mark.parametrize(
    "name, expected",
    [
        ("EnvironmentStartTimeoutError", ErrorTreatment.MASK),
        ("TrialTimeoutError", ErrorTreatment.MASK),
        ("SnapshotQuotaExceeded", ErrorTreatment.MASK),
        ("DaytonaSandboxStopError", ErrorTreatment.MASK),
        ("AgentSetupTimeoutError", ErrorTreatment.MASK),
        ("RewardFileNotFoundError", ErrorTreatment.MASK),
        ("RewardFileEmptyError", ErrorTreatment.MASK),
        ("VerifierRuntimeError", ErrorTreatment.MASK),
        ("VerifierOutputParseError", ErrorTreatment.MASK),
        ("DaytonaProcessError", ErrorTreatment.MASK),
        ("AgentTimeoutError", ErrorTreatment.ZERO),
        ("ContextLengthExceededError", ErrorTreatment.ZERO),
        ("VerifierTimeoutError", ErrorTreatment.ZERO),
        ("DaytonaError", ErrorTreatment.MASK),
        ("DaytonaDockerSetupError", ErrorTreatment.MASK),
        ("InternalServerError", ErrorTreatment.MASK),
        ("APIConnectionError", ErrorTreatment.MASK),
        ("APITimeoutError", ErrorTreatment.MASK),
        ("TaskMachineError", ErrorTreatment.MASK),
        ("TemplateError", ErrorTreatment.MASK),
        ("PromptTokenizationError", ErrorTreatment.MASK),
        ("ModelServerError", ErrorTreatment.MASK),
        ("DaytonaConnectionError", ErrorTreatment.MASK),
        ("DaytonaConflictError", ErrorTreatment.MASK),
        ("MachineTerminated", ErrorTreatment.MASK),
        ("DaytonaOutputUnavailable", ErrorTreatment.MASK),
    ],
)
def test_terminal_error_treatment_contract(name, expected):
    assert classify_exception_type(name, ErrorHandlingConfig()) is expected


def test_training_override_masks_an_agent_timeout():
    config = ErrorHandlingConfig(mask_exceptions=frozenset({"AgentTimeoutError"}))
    assert classify_exception_type("AgentTimeoutError", config) is ErrorTreatment.MASK
    assert classify_exception_type("AgentTimeoutError", ErrorHandlingConfig()) is ErrorTreatment.ZERO
