# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from rigging.task_errors import ErrorIdentity
from rolloutengine.models import RolloutOperation
from skyrl_gym.verification import VerificationResult
from taskcompendium.grading_result import (
    GRADING_DURATION_SECONDS,
    GRADING_FAILURE,
    GradeResult,
    Outcome,
)

from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.grader_reporting import GraderStats, write_grader_report
from skyrl_train.io import io
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
)
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.task_inputs import TrajectoryMetadata
from skyrl_train.trajectory_runners.trajectory_processing import normalize_trajectory_batches
from skyrl_train.trajectory_runners.types import BatchFields, TaskRequestBatch, TrajectoryBatch
from tests.cpu.task_specs import lowered_task, task_input, text_task


class ProjectionTokenizer:
    """Minimal tokenizer for the projection, which runs with overlong filtering off."""

    eos_token_id = 99


def projected_failure_rollout(failure: ErrorIdentity, grade: GradeResult) -> TrajectoryBatch:
    """Project one interrupted attempt the way the task worker does, as a final whole-task row."""
    rollout = TrainingRollout(
        task_id="fixture",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(),
        loss_mask=(),
        logprobs=None,
        grade=grade,
        stop_reason="error",
        failure=failure,
    )
    projection = WholeTaskProjection(
        OmegaConf.create({"sampling_params": {"logprobs": None}, "apply_overlong_filtering": False}),
        ProjectionTokenizer(),
    )
    batch = projection.project([rollout], TaskRequestBatch(rows=(task_input(lowered_task(text_task())),)))
    batch["is_last_step"] = [True]
    return batch


def test_grader_report_keeps_baseline_and_final_success_by_task_and_subset(tmp_path):
    export_path = str(tmp_path)
    math_a = TrajectoryMetadata("row-0", "taskcompendium", "blend", task_id="math-a", subset_id="math")
    math_b = TrajectoryMetadata("row-1", "taskcompendium", "other", task_id="math-b", subset_id="math")
    code = TrajectoryMetadata("row-2", "taskcompendium", "blend", task_id="code-a", subset_id="code")
    baseline = GraderStats()
    graded_rows = TrajectoryBatch(
        prompt_token_ids=[[1]] * 6,
        response_ids=[[2]] * 6,
        loss_masks=[[0]] * 6,
        rewards=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        is_last_step=[False, True, True, True, True, True],
        verification_results=[
            VerificationResult.verified(1.0, passed=True),
            VerificationResult.verified(1.0, passed=True, diagnostics={GRADING_DURATION_SECONDS: 1.0}),
            VerificationResult.verified(0.7, passed=False, diagnostics={GRADING_DURATION_SECONDS: 3.0}),
            VerificationResult.error(
                "timeout", diagnostics={GRADING_DURATION_SECONDS: 9.0, GRADING_FAILURE: "timeout"}
            ),
            VerificationResult.error(
                "worker error", diagnostics={GRADING_DURATION_SECONDS: 2.0, GRADING_FAILURE: "execution"}
            ),
            VerificationResult.unavailable("no model output"),
        ],
    )
    model_failure = ErrorIdentity(exception_type="ModelServerError", operation=RolloutOperation.MODEL.value)
    failed = projected_failure_rollout(model_failure, GradeResult(Outcome.INFRA_ERROR, None))
    batch = normalize_trajectory_batches(
        [graded_rows, failed],
        fields=BatchFields.from_groups([BatchFields.from_batch(graded_rows), BatchFields.from_batch(failed)]),
    )
    baseline.add(batch, [math_b, math_a, math_b, math_a, code, code, math_b])
    summary = baseline.summary()
    write_grader_report(export_path, "0", {"evaluation": summary["subsets"], "evaluation_tasks": summary["tasks"]})

    final = GraderStats()
    final.add(
        TrajectoryBatch(rewards=[1.0], verification_results=[VerificationResult.verified(1.0)]),
        [math_b],
    )
    summary = final.summary()
    write_grader_report(export_path, "25", {"evaluation": summary["subsets"], "evaluation_tasks": summary["tasks"]})
    write_grader_report(export_path, "25", {"evaluation_duration_seconds": 11.0})

    report = json.loads((tmp_path / "grader_report.json").read_text())
    evaluations = report["evaluation"]
    code, math = evaluations["0"]
    assert report["evaluation_duration_seconds"] == {"25": 11.0}
    assert code["subset_id"] == "code"
    assert code["success_rate"] is None
    assert code["grading_failures"] == {"execution": 1}
    assert code["verification_statuses"] == {"error": 1, "unavailable": 1}
    assert math["subset_id"] == "math"
    assert math["data_sources"] == ["blend", "other"]
    assert math["attempted"] == 4
    assert math["verified"] == 2
    assert math["success_rate"] == 0.5
    assert math["verification_coverage"] == 0.5
    assert math["grading_failures"] == {"timeout": 1}
    assert math["rollout_failures"] == {"model": 1}
    assert math["grading_latency_seconds"] == pytest.approx({"count": 3, "mean": 13 / 3, "p95": 8.4, "max": 9})
    assert evaluations["25"][0]["success_rate"] == 1.0
    tasks = {row["task_id"]: row for row in report["evaluation_tasks"]["0"]}
    assert tasks["code-a"]["success_rate"] is None
    assert tasks["math-a"]["subset_ids"] == ["math"]
    assert tasks["math-a"]["success_rate"] == 1.0
    assert tasks["math-a"]["verification_coverage"] == 0.5
    assert tasks["math-b"]["success_rate"] == 0.0
    assert tasks["math-b"]["attempted"] == 2
    assert report["evaluation_tasks"]["25"][0]["success_rate"] == 1.0


def test_grader_report_counts_a_grade_phase_failure_from_the_projected_identity():
    """An interrupted grade without a recorded grading failure counts as a grading execution failure."""
    grade_failure = ErrorIdentity(exception_type="VerifierTimeoutError", operation=RolloutOperation.GRADE.value)
    batch = projected_failure_rollout(grade_failure, GradeResult(Outcome.UNAVAILABLE, None, "verifier interrupted"))
    stats = GraderStats()
    stats.add(batch, [TrajectoryMetadata("row-0", "taskcompendium", "fixture", subset_id="set")])
    (subset,) = stats.summary()["subsets"]
    assert subset["grading_failures"] == {"execution": 1}
    assert subset["rollout_failures"] == {}
    assert subset["verification_statuses"] == {"error": 1}


def test_grader_report_recovers_from_a_failed_write(tmp_path, monkeypatch):
    export_path = str(tmp_path)
    write_grader_report(export_path, "0", {"evaluation_duration_seconds": 1.0})

    def unavailable_storage(_path, _payload):
        raise OSError("Object storage unavailable")

    with monkeypatch.context() as storage:
        storage.setattr(io, "write_bytes_atomic", unavailable_storage)
        write_grader_report(export_path, "1", {"evaluation_duration_seconds": 2.0})

    write_grader_report(export_path, "2", {"evaluation_duration_seconds": 3.0})
    report = json.loads((tmp_path / "grader_report.json").read_text())
    assert report["evaluation_duration_seconds"] == {"0": 1.0, "2": 3.0}


@pytest.mark.asyncio
async def test_training_report_counts_a_crashed_grader_in_a_rejected_group(tmp_path):
    buffer = RolloutBuffer(
        RolloutBufferConfig(
            batch_size=1,
            max_in_flight=1,
            max_staleness_steps=0,
            batch_policy=BatchPolicy.FULL_BATCH,
            dynamic_sampling=None,
            max_candidate_groups=None,
        )
    )
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=1), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(None),
        report_graders=True,
    )
    await buffer.publish(1)

    metadata = TrajectoryMetadata("row-0", "taskcompendium", "blend", task_id="task-1", subset_id="code")
    prompt = {"input": SimpleNamespace(metadata=metadata)}
    for result, loss_mask in (
        (VerificationResult.error("crash", diagnostics={GRADING_FAILURE: "execution"}), [0]),
        (VerificationResult.verified(1.0), [1]),
    ):
        lease = await buffer.acquire_lease()
        batch = TrajectoryBatch(
            prompt_token_ids=[[1]],
            response_ids=[[2]],
            loss_masks=[loss_mask],
            rewards=[result.score or 0.0],
            verification_results=[result],
        )
        verdict = policy.verdict(RolloutGroup(batch, "row-0", lease.policy_step, prompt))
        await buffer.commit(lease.lease_id, prompt, verdict, PayloadReference(None))

    admission = await buffer.admit(timeout=1)
    assert len(admission.admitted) == 1
    summary = buffer.grader_summary()
    write_grader_report(str(tmp_path), "attempt", {"training": summary["subsets"], "training_tasks": summary["tasks"]})
    report = json.loads((tmp_path / "grader_report.json").read_text())
    (row,) = report["training"]["attempt"]
    assert row["attempted"] == 2
    assert row["success_rate"] == 1.0
    assert row["verification_coverage"] == 0.5
    assert row["grading_failures"] == {"execution": 1}
    (task,) = report["training_tasks"]["attempt"]
    assert task["task_id"] == "task-1"
    assert task["attempted"] == 2
    assert task["verified"] == 1
    assert task["success_rate"] == 1.0
