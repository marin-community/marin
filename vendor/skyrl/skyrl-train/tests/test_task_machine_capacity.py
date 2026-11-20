"""Capacity failures retain their exclusion reason across rollout classification."""

from types import SimpleNamespace

from rolloutengine.models import GradedRolloutInterrupted, Rollout, RolloutOperation
from shellbox.backends.daytona.machine import SandboxCapacityError
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import NoGrader

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.rollouts.machines import MachineOperation, TaskMachineError
from skyrl_train.rollouts.task_worker import _failed_rollout
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig, classify_exception_type


def test_capacity_start_failure_keeps_distinct_masked_exclusion():
    provider_error = SandboxCapacityError("Daytona sandbox capacity exhausted", status_code=400)
    machine_error = TaskMachineError("Task-machine creation failed", MachineOperation.CREATE)
    machine_error.__cause__ = provider_error
    interruption = GradedRolloutInterrupted(
        Rollout(run_id="trial", conversations=(), stop_reason="error"),
        RolloutOperation.START,
        GradeResult(status=Outcome.UNAVAILABLE, reward=None),
    )
    interruption.__cause__ = machine_error

    lowered = SimpleNamespace(task=SimpleNamespace(id="trial", grader=NoGrader(reason="test")), harness="fixture")
    result = _failed_rollout(interruption, lowered)

    assert result.failure is not None
    assert result.failure.exception_type == "SandboxCapacityError"
    assert classify_exception_type(result.failure.exception_type, ErrorHandlingConfig()) is ErrorTreatment.MASK
