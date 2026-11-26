# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from types import SimpleNamespace

import pytest
from skyrl_gym.verification import VerificationResult
from taskcompendium.grading_result import GRADING_DURATION_SECONDS, GRADING_FAILURE

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy
from skyrl_train.grader_reporting import GraderStats, write_grader_report
from skyrl_train.io import io
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    SAMPLING_SOURCE,
    BatchPolicy,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
)
from skyrl_train.rollouts.prompt_order import JudgedGroup
from skyrl_train.rollouts import context as rollout_context
from skyrl_train.task_inputs import TrajectoryMetadata
from skyrl_train.trajectory_runners.types import TrajectoryBatch


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_judged_feedback_includes_prefetched_discards_across_resume(resume, monkeypatch):
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(DynamicSamplingType.FILTER),
    )
    config = RolloutBufferConfig(
        batch_size=1,
        max_in_flight=2,
        max_staleness_steps=1,
        batch_policy=BatchPolicy.FULL_BATCH,
        dynamic_sampling=DynamicSamplingType.FILTER,
        max_candidate_groups=3,
    )
    buffer = RolloutBuffer(config)

    async def commit(lease, uid, rewards, results, *, loss_masks=((1,), (1,))):
        prompt = {SAMPLING_SOURCE: "math"}
        batch = TrajectoryBatch(
            prompt_token_ids=[[1], [1]],
            response_ids=[[2], [3]],
            loss_masks=loss_masks,
            rewards=rewards,
            verification_results=results,
        )
        verdict = policy.verdict(RolloutGroup(batch, uid, lease.policy_step, prompt))
        await buffer.commit(lease.lease_id, prompt, verdict, PayloadReference(None))

    await buffer.publish(1)
    current = await buffer.acquire_lease()
    future = await buffer.acquire_lease()
    await commit(future, "0", [0.0, 0.0], [VerificationResult.verified(1.0)] * 2)
    await commit(
        await buffer.acquire_lease(), "3", [0.0, 0.0], [VerificationResult.error("crash")] * 2, loss_masks=[[0], [0]]
    )
    await commit(current, "2", [0.0, 0.9], [VerificationResult.verified(0.5)] * 2)
    initial = await buffer.admit(timeout=1)
    if resume:
        snapshot = buffer.snapshot()
        buffer = RolloutBuffer(config)
        await buffer.restore(snapshot)
    await buffer.publish(2)
    await commit(
        await buffer.acquire_lease(),
        "1",
        [0.0, 0.9],
        [VerificationResult.error("timeout"), VerificationResult.verified(0.5)],
    )
    admission = await buffer.admit(timeout=1)
    assert admission.selection.judged == [
        JudgedGroup("0", (0.0, 0.0), (1.0, 1.0)),
        JudgedGroup("1", (0.0, 0.9), (None, 0.5)),
    ]
    assert admission.selection.metrics["async/dynamic_sampling/candidate_count"] == 2
    assert admission.selection.metrics["curriculum/math/rejected/fully_masked"] == 1
    events = []
    monkeypatch.setattr(rollout_context, "record_event", lambda name, fields: events.append((name, fields)))
    for result in (initial, admission):
        for outcome in result.dispositions:
            rollout_context.record_sampling_result(outcome, result.batch_id)
    receipts = {fields["lease_id"]: fields for name, fields in events if name == "sampling_group_result"}
    assert len(receipts) == 4
    assert receipts[future.lease_id]["dispatch_step"] == receipts[future.lease_id]["admission_batch"] == 1
    assert receipts[future.lease_id]["feedback_batch"] == 2
    assert receipts[future.lease_id]["verified_one"] == 2
    assert receipts[current.lease_id]["feedback_batch"] == 1
    assert receipts[current.lease_id]["verified_min"] == receipts[current.lease_id]["verified_max"] == 0.5
    rejected = next(row for row in receipts.values() if row["disposition"] == "fully_masked")
    assert rejected["feedback_batch"] is None and rejected["verified_missing"] == 2
    assert all(row["generated_tokens"] == row["sample_count"] == 2 for row in receipts.values())
    rollout_context.record_sampling_training(
        rollout_context.RolloutBatchMetadata(admission.batch_id, tuple(admission.admitted), {}), Counter({"1": 0})
    )
    trained = [fields for name, fields in events if name == "sampling_group_trained"]
    assert trained == [{"group_uid": "1", "training_step": 2, "nonzero_advantage_tokens": 0}]
    assert [
        receipt["disposition"]
        for receipt in receipts.values()
        if receipt["group_uid"] == trained[0]["group_uid"] and receipt["feedback_batch"] == trained[0]["training_step"]
    ] == ["consumed"]


def test_grader_report_keeps_baseline_and_final_success_by_task_and_subset(tmp_path):
    export_path = str(tmp_path)
    math_a = TrajectoryMetadata("row-0", "taskcompendium", "blend", task_id="math-a", subset_id="math")
    math_b = TrajectoryMetadata("row-1", "taskcompendium", "other", task_id="math-b", subset_id="math")
    code = TrajectoryMetadata("row-2", "taskcompendium", "blend", task_id="code-a", subset_id="code")
    baseline = GraderStats()
    batch = TrajectoryBatch(
        rewards=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        is_last_step=[False, True, True, True, True, True, True],
        verification_results=[
            VerificationResult.verified(1.0, passed=True),
            VerificationResult.verified(1.0, passed=True, diagnostics={GRADING_DURATION_SECONDS: 1.0}),
            VerificationResult.verified(0.7, passed=False, diagnostics={GRADING_DURATION_SECONDS: 3.0}),
            VerificationResult.error(
                "timeout", diagnostics={GRADING_DURATION_SECONDS: 9.0, GRADING_FAILURE: "timeout"}
            ),
            VerificationResult.error("no response", diagnostics={"operation": "model"}),
            VerificationResult.error(
                "worker error", diagnostics={GRADING_DURATION_SECONDS: 2.0, GRADING_FAILURE: "execution"}
            ),
            VerificationResult.unavailable("no model output"),
        ],
    )
    baseline.add(batch, [math_b, math_a, math_b, math_a, math_b, code, code])
    summary = baseline.summary()
    write_grader_report(export_path, "0", {"evaluation": summary["subsets"], "evaluation_tasks": summary["tasks"]})

    final = GraderStats()
    final.add(
        TrajectoryBatch(rewards=[1.0], verification_results=[VerificationResult.verified(1.0)]),
        [math_b],
    )
    summary = final.summary()
    write_grader_report(export_path, "25", {"evaluation": summary["subsets"], "evaluation_tasks": summary["tasks"]})
    write_grader_report(export_path, "25", {"evaluation_duration_seconds": 3.5})

    report = json.loads((tmp_path / "grader_report.json").read_text())
    evaluations = report["evaluation"]
    assert report["evaluation_duration_seconds"] == {"25": 3.5}
    code, math = evaluations["0"]
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
