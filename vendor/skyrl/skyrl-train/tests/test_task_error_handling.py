import pytest

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.task_error_handling import ErrorHandlingConfig, classify_exception_type


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ContextManagementInfrastructureError", ErrorTreatment.MASK),
        ("AgentTimeoutError", ErrorTreatment.ZERO),
        ("OutputLengthExceededError", ErrorTreatment.PASSTHROUGH),
        ("TrialNotScoredError", ErrorTreatment.ZERO),
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
    ],
)
def test_terminal_error_treatment_contract(name, expected):
    assert classify_exception_type(name, ErrorHandlingConfig()) is expected


def test_training_override_preserves_default_treatment():
    config = ErrorHandlingConfig(mask_exceptions=frozenset({"AgentTimeoutError"}))
    assert classify_exception_type("AgentTimeoutError", config) is ErrorTreatment.MASK
    assert classify_exception_type("AgentTimeoutError", ErrorHandlingConfig()) is ErrorTreatment.ZERO
