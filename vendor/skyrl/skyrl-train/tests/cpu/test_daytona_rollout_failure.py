# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A Daytona provider failure becomes a failed rollout with safe diagnostics."""

import pytest
from daytona.v2 import ConnectionError as ProviderConnectionError
from daytona.v2 import LimitExceededError
from rolloutengine.models import GradedRolloutInterrupted, Rollout, RolloutOperation
from taskcompendium.grading_result import GradeResult, Outcome

from shellbox.backends.daytona.machine import DaytonaCapacityExhausted
from skyrl_train.rollouts.machines import MachineOperation, TaskMachineError
from skyrl_train.rollouts.task_worker import _failed_rollout
from tests.cpu.task_specs import lowered_task, text_task


@pytest.mark.parametrize(
    ("shellbox_error", "provider_error", "expected"),
    [
        (
            DaytonaCapacityExhausted("Daytona cannot create a sandbox"),
            LimitExceededError(
                "concurrent sandbox limit reached", status_code=429, code="LIMIT_EXCEEDED", retry_after_seconds=30
            ),
            {"provider_status_code": 429, "provider_error_code": "LIMIT_EXCEEDED", "provider_retry_after_seconds": 30},
        ),
        (
            ConnectionError("Lost the connection to Daytona"),
            ProviderConnectionError("Connection reset by peer"),
            {},
        ),
    ],
    ids=["limit", "connection"],
)
def test_daytona_provider_error_fails_the_rollout_with_its_diagnostics(shellbox_error, provider_error, expected) -> None:
    # Shellbox raises a provider failure under its own type, with the SDK error as its cause.
    shellbox_error.__cause__ = provider_error
    machine_error = TaskMachineError("create failed", MachineOperation.CREATE)
    machine_error.__cause__ = shellbox_error
    interruption = GradedRolloutInterrupted(
        Rollout(run_id="run", conversations=(), stop_reason="error"),
        RolloutOperation.START,
        GradeResult(Outcome.UNAVAILABLE, None),
    )
    interruption.__cause__ = machine_error

    rollout = _failed_rollout(interruption, lowered_task(text_task()))

    assert rollout.failure.exception_type == "TaskMachineError"
    diagnostics = rollout.failure.diagnostics
    assert diagnostics["cause_error_type"] == type(shellbox_error).__name__
    assert {key: diagnostics[key] for key in expected} == expected
