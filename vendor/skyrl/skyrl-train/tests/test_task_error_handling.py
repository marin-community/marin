"""Freeze terminal training treatment while sharing the task error schema."""

import subprocess
import sys

import pytest
from rigging.task_errors import ErrorIdentity, error_category
from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.task_error_handling import ErrorHandlingConfig, classify_failure
from skyrl_train.task_errors import ERROR_CATEGORIES


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
    failure = ErrorIdentity(exception_type=name, category=error_category(name, categories=ERROR_CATEGORIES))
    assert classify_failure(failure, ErrorHandlingConfig()) is expected


def test_training_override_masks_an_agent_timeout():
    config = ErrorHandlingConfig(mask_exceptions=frozenset({"AgentTimeoutError"}))
    assert classify_failure(ErrorIdentity(exception_type="AgentTimeoutError"), config) is ErrorTreatment.MASK


def test_stored_category_controls_treatment_without_harbor():
    program = r"""
import importlib.abc
import importlib.util
import sys

class NoHarbor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "harbor" or fullname == "harbor_config" or fullname.startswith(("harbor.", "harbor_config.")):
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

original_find_spec = importlib.util.find_spec

def available_spec(name, package=None):
    return None if name == "harbor_config" else original_find_spec(name, package)

sys.meta_path.insert(0, NoHarbor())
importlib.util.find_spec = available_spec
from rigging.task_errors import ErrorIdentity, RetrySettings, retry_delay
from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.task_error_handling import ErrorHandlingConfig, classify_failure

failure = ErrorIdentity.model_validate_json(
    '{"exception_type":"RuntimeServiceError","category":"infrastructure","retry_disposition":"transient"}'
)
assert classify_failure(failure, ErrorHandlingConfig()) is ErrorTreatment.MASK
assert retry_delay(failure, RetrySettings(jitter=0, max_retries=1), 0) == 1
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
