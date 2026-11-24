# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from finestore.rl.task_health import ATTEMPT_RECEIPTS_DIRECTORY, Phase
from rolloutengine.models import RolloutFailure
from rolloutengine.test_support import arithmetic_task, rollout_spec
from shellbox.machine import ExitReason
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import EnvironmentRequirements

from skyrl_train.rollouts.task_health import attempt_receipt, write_attempt_receipts
from skyrl_train.rollouts.training_record import TrainingRollout


@pytest.mark.parametrize(
    ("reason", "exit_code", "recorded_exit_code"),
    [(ExitReason.EXITED, 2, 2), (ExitReason.TIMED_OUT, 124, None)],
)
def test_health_receipt_preserves_setup_exit_evidence(tmp_path, reason, exit_code, recorded_exit_code):
    task = arithmetic_task(
        tags=("agentic",), environment_requirements=EnvironmentRequirements(setup_commands=("./setup",))
    )
    message = "Could not configure the task environment"
    result = TrainingRollout(
        task_id=task.id,
        messages=(),
        prompt_token_ids=(),
        response_token_ids=(),
        loss_mask=(),
        logprobs=None,
        grade=GradeResult(Outcome.INFRA_ERROR, None, message),
        stop_reason="error",
        failure=RolloutFailure(
            "MachineSetupError",
            {"setup_reason": reason.value, "setup_exit_code": exit_code, "setup_stderr": "setup output"},
        ),
    )
    receipt = attempt_receipt("attempt-1", rollout_spec(task), result)
    assert receipt is not None
    write_attempt_receipts(str(tmp_path), [receipt])

    (stored_receipts,) = (tmp_path / ATTEMPT_RECEIPTS_DIRECTORY).glob("*.jsonl")
    stored = json.loads(stored_receipts.read_text())
    assert (stored["phase"], stored["setup_completed"], stored["setup_exit_code"]) == (
        Phase.TASK_SETUP,
        False,
        recorded_exit_code,
    )
    assert (stored["error"], stored["setup_stderr"]) == (message, "setup output")
