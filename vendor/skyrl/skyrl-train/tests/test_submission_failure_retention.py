# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission-failure evidence survives SkyRL projection and trajectory retention."""

from omegaconf import OmegaConf
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.trajectory_runners.trajectory_retention import build_trajectory_records
from skyrl_train.trajectory_runners.trajectory_retention_config import TrajectoryRetentionConfig
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.task_specs import lowered_task, task_input, text_task


class _Tokenizer:
    eos_token_id = 9

    def decode(self, token_ids, skip_special_tokens):
        return "answer"


def test_submission_failure_explanation_survives_retained_trajectory():
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(text_task())),),
        trajectory_ids=(TrajectoryID("task", 0),),
        batch_metadata=BatchMetadata(global_step=5, training_phase="train"),
    )
    rollout = TrainingRollout(
        task_id="task",
        messages=(),
        prompt_token_ids=(3,),
        response_token_ids=(4,),
        loss_mask=(1,),
        logprobs=None,
        grade=GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "Empty text submission"),
        stop_reason="length",
    )
    tokenizer = _Tokenizer()
    projection = WholeTaskProjection(
        runner_config=OmegaConf.create({"apply_overlong_filtering": False, "sampling_params": {"logprobs": None}}),
        tokenizer=tokenizer,
    )

    batch = projection.project((rollout,), request)
    record = build_trajectory_records(
        request, batch, TrajectoryRetentionConfig(run_id="run"), tokenizer, runner_name="task"
    )[0].to_json()

    verification = record["verification_result"]
    assert verification["status"] == "verified"
    assert verification["score"] == 0.0
    assert verification["reason"] is None
    assert verification["diagnostics"] == {
        "grade_status": "submission_failure",
        "grade_error": "Empty text submission",
    }
    assert record["reward"]["outcome"] == 0.0
    assert batch["rewards"] == [0.0]
    assert batch["exclude_from_baseline"] == [False]
    assert record["response"]["loss_mask"] == [1]
