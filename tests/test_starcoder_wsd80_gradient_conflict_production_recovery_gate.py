# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fsspec
import pytest
from fray.types import ResourceConfig
from iris.cluster.types import JobName, TaskAttempt
from levanter.distributed import DistributedConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_production_recovery_gate as gate
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_production_recovery_gate_20260813 as analyzer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_conflict_production_recovery_gate_20260813 as freeze,
)


@dataclass(frozen=True)
class _MinimalTrainConfig:
    trainer: TrainerConfig
    initial_state_evidence_path: str | None = None


def _minimal_config() -> TrainLmOnPodConfig:
    trainer = TrainerConfig(
        id="base-run",
        tracker=WandbConfig(name="base-run", replicate_path="gs://bucket/base"),
        distributed=DistributedConfig(initialize_jax_distributed=False),
    )
    return TrainLmOnPodConfig(
        train_config=_MinimalTrainConfig(trainer=trainer),
        resources=replace(ResourceConfig.with_cpu(), preemptible=True),
        output_path="gs://bucket/base",
        env_vars={},
    )


def test_gate_uses_production_load_cadence_and_all_support_modes():
    rows = stress.rows_for_stage(gate.STAGE)

    assert gate.STAGE == 64
    assert gate.CHECKPOINT_INTERVAL.total_seconds() == 5 * 60
    assert tuple(fault.task_index for fault in gate.FAULT_INJECTIONS) == (0, 45, 62)
    assert tuple(rows[fault.task_index].support_id for fault in gate.FAULT_INJECTIONS) == (
        "m100a",
        "full",
        "m100b",
    )
    assert all(fault.trigger_step == analyzer.FIRST_OPERATIONAL_WANDB_STEP for fault in gate.FAULT_INJECTIONS)
    assert all("at_first_temporary_checkpoint" in fault.phase for fault in gate.FAULT_INJECTIONS)
    assert all(0 < fault.trigger_step < stress.TERMINAL_STEP for fault in gate.FAULT_INJECTIONS)


def test_resumable_config_keeps_stable_identity_and_full_state_discovery():
    row = SimpleNamespace(trajectory_id="row-a")

    config = gate._resumable_config(_minimal_config(), row)
    trainer = config.train_config.trainer

    assert config.output_path == "gs://bucket/base/attempt-000"
    expected_run_id = gate._run_id(row)
    assert config.env_vars == {"RUN_ID": expected_run_id}
    assert trainer.id == expected_run_id
    assert trainer.load_checkpoint is None
    assert trainer.load_checkpoint_path is None
    assert trainer.distributed.initialize_jax_distributed is False
    assert trainer.checkpointer.save_interval == gate.CHECKPOINT_INTERVAL
    assert trainer.tracker.id == expected_run_id
    assert trainer.tracker.name == expected_run_id
    assert trainer.tracker.resume == "allow"
    assert trainer.tracker.replicate_path == config.output_path


def test_checkpoint_label_restores_the_next_trainer_step(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_search_paths": ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"],
                "checkpoint_step": 123,
                "initial_state_evidence_path": "state-attempt-1",
            }
        ]
    }
    monkeypatch.setattr(
        analyzer,
        "_read_json",
        lambda _: {
            "checkpoint_search_paths": ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"],
            "run_id": gate._run_id(row),
            "state_step": 124,
            "written_at": "2026-08-13T12:00:01+00:00",
        },
    )

    authorizations = {0: {1: {"authorized_at": "2026-08-13T12:00:00+00:00"}}}
    observed, failures = analyzer._state_evidence((row,), attempts, authorizations)

    assert failures == []
    assert observed[0][1]["state_step"] == 124


def test_checkpoint_label_rejects_completed_step_as_restored_state(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_search_paths": ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"],
                "checkpoint_step": 123,
                "initial_state_evidence_path": "state-attempt-1",
            }
        ]
    }
    monkeypatch.setattr(
        analyzer,
        "_read_json",
        lambda _: {
            "checkpoint_search_paths": ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"],
            "run_id": gate._run_id(row),
            "state_step": 123,
            "written_at": "2026-08-13T12:00:01+00:00",
        },
    )

    authorizations = {0: {1: {"authorized_at": "2026-08-13T12:00:00+00:00"}}}
    _, failures = analyzer._state_evidence((row,), attempts, authorizations)

    assert failures == ["task 0 attempt 1 initialized at step 123, expected 124 from checkpoint label step-123"]


def _parent_claim(step: int = 123) -> dict[str, Any]:
    return {
        "checkpoint_path": f"gs://bucket/temporary/step-{step}",
        "checkpoint_search_paths": ["gs://bucket/permanent", "gs://bucket/temporary"],
        "checkpoint_step": step,
        "checkpoint_metadata_sha256": f"sha-{step}",
        "recovery_mode": gate.RECOVERY_MODE_CHECKPOINT,
        "injected_at": "2026-08-13T12:00:00+00:00",
        "source_attempt": 0,
    }


def _worker_claim(step: int = 123) -> dict[str, Any]:
    return {
        "attempt": 1,
        "checkpoint_path": f"gs://bucket/temporary/step-{step}",
        "checkpoint_search_paths": ["gs://bucket/permanent", "gs://bucket/temporary"],
        "checkpoint_step": step,
        "checkpoint_metadata_sha256": f"sha-{step}",
        "recovery_mode": gate.RECOVERY_MODE_CHECKPOINT,
    }


def test_forced_retry_accepts_exact_parent_checkpoint_and_next_step_state():
    worker = _worker_claim()

    matched, failures = analyzer._forced_retry_evidence(
        {0: [worker]},
        {0: _parent_claim()},
        {0: {1: {"state_step": 124, "written_at": "2026-08-13T12:00:01+00:00"}}},
    )

    assert failures == []
    assert matched == {0: worker}


@pytest.mark.parametrize(
    ("worker_step", "state_step"),
    (
        (123, 123),  # Loading checkpoint N must initialize trainer state N+1, not N.
        (122, 123),  # A worker may not recover from a checkpoint older than the parent's claim.
    ),
)
def test_forced_retry_rejects_invalid_recovery_claims(worker_step: int, state_step: int):
    matched, failures = analyzer._forced_retry_evidence(
        {0: [_worker_claim(worker_step)]},
        {0: _parent_claim()},
        {0: {1: {"state_step": state_step, "written_at": "2026-08-13T12:00:01+00:00"}}},
    )

    assert matched == {}
    assert failures == [
        "task 0 has no post-fault state restored from a temporary checkpoint "
        "at or after the parent-observed checkpoint"
    ]


def test_forced_retry_rejects_newer_permanent_checkpoint():
    worker = _worker_claim(step=1_536)
    worker["checkpoint_path"] = "gs://bucket/permanent/step-1536"

    matched, failures = analyzer._forced_retry_evidence(
        {0: [worker]},
        {0: _parent_claim(step=790)},
        {0: {1: {"state_step": 1_537, "written_at": "2026-08-13T12:00:01+00:00"}}},
    )

    assert matched == {}
    assert failures == [
        "task 0 has no post-fault state restored from a temporary checkpoint "
        "at or after the parent-observed checkpoint"
    ]


def test_forced_retry_accepts_newer_temporary_checkpoint():
    worker = _worker_claim(step=900)

    matched, failures = analyzer._forced_retry_evidence(
        {0: [worker]},
        {0: _parent_claim(step=790)},
        {0: {1: {"state_step": 901, "written_at": "2026-08-13T12:00:01+00:00"}}},
    )

    assert failures == []
    assert matched == {0: worker}


def test_wandb_gate_accepts_exact_emitted_step_domain(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    history = [
        {
            "_step": step - analyzer.FIRST_OPERATIONAL_WANDB_STEP,
            "_timestamp": float(step),
            "global_step": step,
            "throughput/tokens_per_second": 1.0,
            "throughput/loading_time": 0.1,
        }
        for step in range(analyzer.FIRST_OPERATIONAL_WANDB_STEP, stress.TERMINAL_STEP + 1)
    ]

    class FakeApi:
        def __init__(self, *, timeout):
            assert timeout == 60

        @staticmethod
        def flush():
            return None

        @staticmethod
        def run(_):
            return SimpleNamespace(id="run-a", scan_history=lambda **_: iter(history))

    monkeypatch.setattr(analyzer.wandb, "Api", FakeApi)
    monkeypatch.setattr(gate, "FAULT_INJECTIONS", ())

    evidence, failures = analyzer._wandb_evidence((row,), {0: [{"attempt": 0, "checkpoint_step": 0}]}, {})

    assert failures == []
    assert evidence[row.trajectory_id]["history_rows"] == stress.TERMINAL_STEP - 1
    assert evidence[row.trajectory_id]["terminal_global_step"] == stress.TERMINAL_STEP


def test_concurrency_gate_requires_sustained_full_c64_overlap():
    passing = {
        f"row-{index}": {"active_intervals": ((0.0, analyzer.FULL_LOAD_OVERLAP_SECONDS_MIN + 1.0),)}
        for index in range(gate.STAGE)
    }

    evidence, failures = analyzer._concurrency_evidence(passing, recovery_completed_at=0.0)

    assert failures == []
    assert evidence["peak_active_workers"] == gate.STAGE
    assert evidence["full_load_overlap_seconds"] == analyzer.FULL_LOAD_OVERLAP_SECONDS_MIN + 1.0

    _, missing_worker_failures = analyzer._concurrency_evidence(
        dict(list(passing.items())[:-1]), recovery_completed_at=0.0
    )
    assert any("peak active workers 63" in failure for failure in missing_worker_failures)

    too_short = {
        row_id: {"active_intervals": ((0.0, analyzer.FULL_LOAD_OVERLAP_SECONDS_MIN - 1.0),)} for row_id in passing
    }
    _, duration_failures = analyzer._concurrency_evidence(too_short, recovery_completed_at=0.0)
    assert any("C64 active overlap" in failure for failure in duration_failures)


def test_concurrency_gate_excludes_full_load_before_recovery():
    evidence = {f"row-{index}": {"active_intervals": ((0.0, 150.0),)} for index in range(gate.STAGE)}

    observed, failures = analyzer._concurrency_evidence(evidence, recovery_completed_at=100.0)

    assert observed["full_load_overlap_seconds"] == 50.0
    assert any("C64 active overlap 50.000s" in failure for failure in failures)


def test_retry_state_requires_parent_authorization(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    search_paths = ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"]
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_search_paths": search_paths,
                "checkpoint_step": 123,
                "initial_state_evidence_path": "state-attempt-1",
            }
        ]
    }
    monkeypatch.setattr(
        analyzer,
        "_read_json",
        lambda _: {
            "checkpoint_search_paths": search_paths,
            "run_id": gate._run_id(row),
            "state_step": 124,
            "written_at": "2026-08-13T12:00:01+00:00",
        },
    )

    _, failures = analyzer._state_evidence((row,), attempts, {0: {}})

    assert "task 0 attempt 1 initialized without parent authorization" in failures


def test_retry_state_must_follow_parent_authorization(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    search_paths = ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"]
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_search_paths": search_paths,
                "checkpoint_step": 123,
                "initial_state_evidence_path": "state-attempt-1",
            }
        ]
    }
    monkeypatch.setattr(
        analyzer,
        "_read_json",
        lambda _: {
            "checkpoint_search_paths": search_paths,
            "run_id": gate._run_id(row),
            "state_step": 124,
            "written_at": "2026-08-13T12:00:00+00:00",
        },
    )

    _, failures = analyzer._state_evidence((row,), attempts, {0: {1: {"authorized_at": "2026-08-13T12:00:01+00:00"}}})

    assert "task 0 attempt 1 initialized before authorization" in failures


def test_attempt_evidence_accepts_parent_authorizable_restart_from_zero(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    urls = ("attempt-0", "attempt-1")
    search_paths = ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"]
    evidence = {
        "attempt-0": {
            "attempt": 0,
            "checkpoint_metadata_sha256": None,
            "checkpoint_path": None,
            "checkpoint_search_paths": search_paths,
            "checkpoint_step": 0,
            "recovery_mode": gate.RECOVERY_MODE_INITIAL,
            "generation": gate.GENERATION,
            "initial_state_evidence_path": gate._initial_state_evidence_path(0, 0),
            "row_id": row.trajectory_id,
            "ready_at": "2026-08-13T11:59:59+00:00",
            "task_attempt_id": "/calvin/job/0:0",
            "task_id": "/calvin/job/0",
            "task_index": 0,
            "worker_region": "us-central1",
        },
        "attempt-1": {
            "attempt": 1,
            "checkpoint_metadata_sha256": None,
            "checkpoint_path": None,
            "checkpoint_search_paths": search_paths,
            "checkpoint_step": 0,
            "recovery_mode": gate.RECOVERY_MODE_RESTART_FROM_ZERO,
            "generation": gate.GENERATION,
            "initial_state_evidence_path": gate._initial_state_evidence_path(0, 1),
            "row_id": row.trajectory_id,
            "ready_at": "2026-08-13T12:00:00+00:00",
            "task_attempt_id": "/calvin/job/0:1",
            "task_id": "/calvin/job/0",
            "task_index": 0,
            "worker_region": "us-central1",
        },
    }
    monkeypatch.setattr(analyzer, "_evidence_objects", lambda kind: urls if kind == "attempts" else ())
    monkeypatch.setattr(analyzer, "_read_json", evidence.__getitem__)
    monkeypatch.setattr(gate, "FAULT_INJECTIONS", (SimpleNamespace(task_index=0),))

    _, failures = analyzer._attempt_evidence((row,))

    assert failures == []


def test_attempt_evidence_rejects_restart_from_zero_that_claims_checkpoint_state(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    search_paths = ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"]
    evidence = {
        "attempt-0": {
            "attempt": 0,
            "checkpoint_metadata_sha256": None,
            "checkpoint_path": None,
            "checkpoint_search_paths": search_paths,
            "checkpoint_step": 0,
            "generation": gate.GENERATION,
            "initial_state_evidence_path": gate._initial_state_evidence_path(0, 0),
            "recovery_mode": gate.RECOVERY_MODE_INITIAL,
            "row_id": row.trajectory_id,
            "task_attempt_id": "/calvin/job/0:0",
            "task_id": "/calvin/job/0",
            "task_index": 0,
            "worker_region": "us-central1",
        },
        "attempt-1": {
            "attempt": 1,
            "checkpoint_metadata_sha256": "sha-123",
            "checkpoint_path": "gs://marin-us-central1/temporary/step-123",
            "checkpoint_search_paths": search_paths,
            "checkpoint_step": 123,
            "generation": gate.GENERATION,
            "initial_state_evidence_path": gate._initial_state_evidence_path(0, 1),
            "recovery_mode": gate.RECOVERY_MODE_RESTART_FROM_ZERO,
            "row_id": row.trajectory_id,
            "task_attempt_id": "/calvin/job/0:1",
            "task_id": "/calvin/job/0",
            "task_index": 0,
            "worker_region": "us-central1",
        },
    }
    monkeypatch.setattr(
        analyzer,
        "_evidence_objects",
        lambda kind: tuple(evidence) if kind == "attempts" else (),
    )
    monkeypatch.setattr(analyzer, "_read_json", evidence.__getitem__)
    monkeypatch.setattr(gate, "FAULT_INJECTIONS", ())

    _, failures = analyzer._attempt_evidence((row,))

    assert "task 0 scratch retry 1 claimed checkpoint state" in failures


def test_parent_authorization_rejects_stale_worker_checkpoint(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)
    search_paths = ("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary")
    stale_claim = {
        "attempt": 1,
        "checkpoint_metadata_sha256": "sha-122",
        "checkpoint_path": "gs://marin-us-central1/checkpoints/step-122",
        "checkpoint_search_paths": list(search_paths),
        "checkpoint_step": 122,
        "recovery_mode": gate.RECOVERY_MODE_CHECKPOINT,
        "task_attempt_id": task_attempt.to_wire(),
    }
    parent_claim = gate.CheckpointClaim(
        path="gs://marin-us-central1/checkpoints/step-123",
        step=123,
        metadata_sha256="sha-123",
        search_paths=search_paths,
    )
    monkeypatch.setattr(gate, "_checkpoint_claim_or_none", lambda _: parent_claim)
    monkeypatch.setattr(gate, "_checkpoint_search_paths", lambda _: search_paths)
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", "memory://production-recovery-test-stale")
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="parent-worker", worker_region="us-central1"),
    )
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    with pytest.raises(RuntimeError, match="did not claim the parent-observed latest checkpoint"):
        gate._authorize_retry_attempts(job, (row,), (_minimal_config(),), {0: (stale_claim,)})


def test_parent_authorization_persists_exact_checkpoint_claim(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)
    search_paths = ("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary")
    parent_claim = gate.CheckpointClaim(
        path="gs://marin-us-central1/checkpoints/step-123",
        step=123,
        metadata_sha256="sha-123",
        search_paths=search_paths,
    )
    worker_claim = {
        "attempt": 1,
        **gate._checkpoint_payload(parent_claim),
        "task_attempt_id": task_attempt.to_wire(),
    }
    evidence_root = "memory://production-recovery-test-authorized"
    monkeypatch.setattr(gate, "_checkpoint_claim_or_none", lambda _: parent_claim)
    monkeypatch.setattr(gate, "_checkpoint_search_paths", lambda _: search_paths)
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="parent-worker", worker_region="us-central1"),
    )
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    gate._authorize_retry_attempts(job, (row,), (_minimal_config(),), {0: (worker_claim,)})

    with fsspec.open(f"{evidence_root}/authorizations/task-000/attempt-001.json") as handle:
        authorization = json.load(handle)
    assert authorization["task_attempt_id"] == task_attempt.to_wire()
    assert analyzer._checkpoint_signature(authorization) == analyzer._checkpoint_signature(worker_claim)
    assert authorization["parent_worker_region"] == "us-central1"


def test_parent_authorization_accepts_restart_from_zero_only_when_no_checkpoint_exists(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)
    search_paths = ("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary")
    worker_claim = {
        "attempt": 1,
        **gate._checkpoint_payload(None, checkpoint_search_paths=search_paths),
        "task_attempt_id": task_attempt.to_wire(),
    }
    evidence_root = "memory://production-recovery-test-scratch-authorized"
    monkeypatch.setattr(gate, "_checkpoint_claim_or_none", lambda _: None)
    monkeypatch.setattr(gate, "_checkpoint_search_paths", lambda _: search_paths)
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="parent-worker", worker_region="us-central1"),
    )
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    gate._authorize_retry_attempts(job, (row,), (_minimal_config(),), {0: (worker_claim,)})

    with fsspec.open(f"{evidence_root}/authorizations/task-000/attempt-001.json") as handle:
        authorization = json.load(handle)
    assert authorization["recovery_mode"] == gate.RECOVERY_MODE_RESTART_FROM_ZERO
    assert authorization["checkpoint_step"] == 0


def test_parent_authorization_rejects_restart_from_zero_when_checkpoint_exists(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)
    search_paths = ("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary")
    worker_claim = {
        "attempt": 1,
        **gate._checkpoint_payload(None, checkpoint_search_paths=search_paths),
        "task_attempt_id": task_attempt.to_wire(),
    }
    parent_claim = gate.CheckpointClaim(
        path="gs://marin-us-central1/temporary/step-123",
        step=123,
        metadata_sha256="sha-123",
        search_paths=search_paths,
    )
    monkeypatch.setattr(gate, "_checkpoint_claim_or_none", lambda _: parent_claim)
    monkeypatch.setattr(gate, "_checkpoint_search_paths", lambda _: search_paths)
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", "memory://production-recovery-test-scratch-rejected")
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="parent-worker", worker_region="us-central1"),
    )
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    with pytest.raises(RuntimeError, match="did not claim the parent-observed latest checkpoint"):
        gate._authorize_retry_attempts(job, (row,), (_minimal_config(),), {0: (worker_claim,)})


def test_parent_authorization_rejects_first_post_fault_permanent_checkpoint(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)
    search_paths = ("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary")
    parent_claim = gate.CheckpointClaim(
        path="gs://marin-us-central1/permanent/step-1536",
        step=1_536,
        metadata_sha256="sha-1536",
        search_paths=search_paths,
    )
    worker_claim = {
        "attempt": 1,
        **gate._checkpoint_payload(parent_claim),
        "task_attempt_id": task_attempt.to_wire(),
    }
    monkeypatch.setattr(gate, "_checkpoint_claim_or_none", lambda _: parent_claim)
    monkeypatch.setattr(gate, "_checkpoint_search_paths", lambda _: search_paths)
    monkeypatch.setattr(gate, "_fault_receipt", lambda _: {"source_attempt": 0})
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", "memory://production-recovery-test-forced-permanent")
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="parent-worker", worker_region="us-central1"),
    )
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    with pytest.raises(RuntimeError, match="did not restore a temporary checkpoint"):
        gate._authorize_retry_attempts(job, (row,), (_minimal_config(),), {0: (worker_claim,)})


def test_retry_waits_for_parent_authorization_before_training(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)
    events: list[str] = []
    monkeypatch.setattr(gate.stress, "_current_task_attempt", lambda: task_attempt)
    monkeypatch.setattr(gate, "_checkpoint_search_paths", lambda _: ("permanent", "temporary"))
    monkeypatch.setattr(gate, "_checkpoint_claim_or_none", lambda _: None)
    monkeypatch.setattr(gate, "_write_attempt_evidence", lambda **_: events.append("claim"))
    monkeypatch.setattr(gate, "_wait_for_recovery_authorization", lambda **_: events.append("authorize"))
    monkeypatch.setattr(gate, "_wait_for_admission_release", lambda **_: events.append("admit"))
    monkeypatch.setattr(gate, "run_levanter_train_lm", lambda _: events.append("train"))

    gate._run_row((_minimal_config(),), (row,))

    assert events == ["claim", "authorize", "admit", "train"]


def test_state_evidence_accepts_authorized_restart_from_zero(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    search_paths = ["gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"]
    attempt = {
        "attempt": 1,
        "checkpoint_search_paths": search_paths,
        "checkpoint_step": 0,
        "initial_state_evidence_path": "state-attempt-1",
        "recovery_mode": gate.RECOVERY_MODE_RESTART_FROM_ZERO,
    }
    monkeypatch.setattr(
        analyzer,
        "_read_json",
        lambda _: {
            "checkpoint_search_paths": search_paths,
            "run_id": gate._run_id(row),
            "state_step": 0,
            "written_at": "2026-08-13T12:00:01+00:00",
        },
    )

    observed, failures = analyzer._state_evidence(
        (row,),
        {0: [attempt]},
        {0: {1: {"authorized_at": "2026-08-13T12:00:00+00:00"}}},
    )

    assert failures == []
    assert observed[0][1]["state_step"] == 0


def test_fault_evidence_rejects_permanent_checkpoint(monkeypatch):
    fault = gate.FaultInjection(task_index=0, trigger_step=100, phase="test")
    evidence = {
        "task_index": 0,
        "trigger_step": 100,
        "phase": "test",
        "checkpoint_path": "gs://marin-us-central1/permanent/step-90",
        "checkpoint_search_paths": [
            "gs://marin-us-central1/permanent",
            "gs://marin-us-central1/temporary",
        ],
        "checkpoint_step": 90,
        "checkpoint_metadata_sha256": "sha-90",
        "generation": gate.GENERATION,
        "injected_at": "2026-08-13T12:00:00+00:00",
        "observed_global_step": 100,
        "recovery_mode": gate.RECOVERY_MODE_CHECKPOINT,
        "requested_state": "preempted",
        "source_attempt": 0,
        "task_attempt_id": "/calvin/child/0:0",
    }
    monkeypatch.setattr(gate, "FAULT_INJECTIONS", (fault,))
    monkeypatch.setattr(analyzer, "_evidence_objects", lambda kind: ("fault",) if kind == "faults" else ())
    monkeypatch.setattr(analyzer, "_read_json", lambda _: evidence)

    _, failures = analyzer._fault_evidence()

    assert "task 0 forced fault did not use a temporary checkpoint" in failures


def _admission_attempt(task_index: int, attempt_index: int, row_id: str) -> dict[str, Any]:
    task = JobName.from_wire(f"/calvin/child/{task_index}")
    return {
        "attempt": attempt_index,
        "ready_at": f"2026-08-13T12:00:{task_index:02d}+00:00",
        "row_id": row_id,
        "task_attempt_id": TaskAttempt(task_id=task, attempt_id=attempt_index).to_wire(),
        "task_id": task.to_wire(),
        "task_index": task_index,
        "worker_id": f"worker-{task_index}-{attempt_index}",
        "worker_region": "us-central1",
    }


def _admission_status(task_index: int, attempt_index: int, *, state: int) -> Any:
    return SimpleNamespace(
        current_attempt_id=attempt_index,
        state=state,
        task_id=f"/calvin/child/{task_index}",
        worker_id=f"worker-{task_index}-{attempt_index}" if state == gate.job_pb2.TASK_STATE_RUNNING else "",
    )


def test_admission_bindings_require_every_current_attempt_running():
    rows = (SimpleNamespace(trajectory_id="row-a"), SimpleNamespace(trajectory_id="row-b"))
    records: dict[int, tuple[dict[str, Any], ...]] = {
        0: (_admission_attempt(0, 0, "row-a"),),
        1: (_admission_attempt(1, 0, "row-b"),),
    }
    client = SimpleNamespace(
        list_tasks=lambda _: (
            _admission_status(0, 0, state=gate.job_pb2.TASK_STATE_RUNNING),
            _admission_status(1, 0, state=gate.job_pb2.TASK_STATE_PENDING),
        )
    )
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    assert gate._current_admission_bindings(client, job, rows, records) is None


def test_admission_bindings_reject_worker_identity_drift():
    rows = (SimpleNamespace(trajectory_id="row-a"),)
    record = _admission_attempt(0, 0, "row-a")
    record["worker_id"] = "different-worker"
    client = SimpleNamespace(list_tasks=lambda _: (_admission_status(0, 0, state=gate.job_pb2.TASK_STATE_RUNNING),))
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))

    with pytest.raises(RuntimeError, match="Admission identity drift"):
        gate._current_admission_bindings(client, job, rows, {0: (record,)})


def test_admission_bindings_require_retry_authorization(monkeypatch):
    evidence_root = "memory://production-recovery-admission-authorization"
    rows = (SimpleNamespace(trajectory_id="row-a"),)
    records = {0: (_admission_attempt(0, 0, "row-a"), _admission_attempt(0, 1, "row-a"))}
    client = SimpleNamespace(list_tasks=lambda _: (_admission_status(0, 1, state=gate.job_pb2.TASK_STATE_RUNNING),))
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)

    assert gate._current_admission_bindings(client, job, rows, records) is None


def test_admission_release_is_single_and_post_release_retries_do_not_rebind(monkeypatch):
    evidence_root = "memory://production-recovery-admission-single"
    rows = (SimpleNamespace(trajectory_id="row-a"), SimpleNamespace(trajectory_id="row-b"))
    statuses = [
        _admission_status(0, 0, state=gate.job_pb2.TASK_STATE_RUNNING),
        _admission_status(1, 0, state=gate.job_pb2.TASK_STATE_RUNNING),
    ]
    client = SimpleNamespace(list_tasks=lambda _: tuple(statuses))
    job = SimpleNamespace(job_id=JobName.from_wire("/calvin/child"))
    records: dict[int, tuple[dict[str, Any], ...]] = {
        0: (_admission_attempt(0, 0, "row-a"),),
        1: (_admission_attempt(1, 0, "row-b"),),
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 2)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="parent-worker", worker_region="us-central1"),
    )

    first = gate._release_current_admission_epoch(client, job, rows, records)

    assert first is not None and first["epoch"] == 0
    statuses[1] = _admission_status(1, 1, state=gate.job_pb2.TASK_STATE_RUNNING)
    retry = _admission_attempt(1, 1, "row-b")
    records[1] = (*records[1], retry)
    authorization = {
        "attempt": 1,
        "authorized_at": "2026-08-13T12:01:00+00:00",
        "generation": gate.GENERATION,
        "row_id": "row-b",
        "task_attempt_id": retry["task_attempt_id"],
        "task_index": 1,
    }
    fs, path = fsspec.core.url_to_fs(gate._recovery_authorization_path(1, 1))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, authorization)

    second = gate._release_current_admission_epoch(client, job, rows, records)

    assert second == first
    assert second is not None and second["bindings"][1]["task_attempt_id"].endswith(":0")


def test_worker_consumes_only_an_exact_attempt_qualified_release(monkeypatch):
    evidence_root = "memory://production-recovery-admission-consume"
    row = SimpleNamespace(trajectory_id="row-a")
    attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=0)
    binding = {
        **_admission_attempt(0, 0, "row-a"),
        "controller_state": "TASK_STATE_RUNNING",
    }
    release = {
        "bindings": [binding],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="worker-0-0", worker_region="us-central1"),
    )
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)

    gate._wait_for_admission_release(
        task_index=0,
        row=row,
        attempt=attempt,
        ready_at=binding["ready_at"],
    )

    with fsspec.open(gate._admission_consumed_path(0, 0)) as handle:
        consumed = json.load(handle)
    assert consumed["task_attempt_id"] == attempt.to_wire()
    assert consumed["admission_mode"] == "bound_current_attempt"
    assert consumed["release_epoch"] == 0
    assert consumed["release_sha256"]


def test_post_release_retry_inherits_barrier_after_recovery_authorization(monkeypatch):
    evidence_root = "memory://production-recovery-admission-post-retry"
    row = SimpleNamespace(trajectory_id="row-a")
    initial = _admission_attempt(0, 0, "row-a")
    retry = _admission_attempt(0, 1, "row-a")
    retry["ready_at"] = "2026-08-13T12:02:00+00:00"
    release = {
        "bindings": [{**initial, "controller_state": "TASK_STATE_RUNNING"}],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="worker-0-1", worker_region="us-central1"),
    )
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)
    attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=1)

    gate._wait_for_admission_release(
        task_index=0,
        row=row,
        attempt=attempt,
        ready_at=retry["ready_at"],
    )

    with fsspec.open(gate._admission_consumed_path(0, 1)) as handle:
        consumed = json.load(handle)
    assert consumed["admission_mode"] == "post_release_retry"
    assert consumed["controller_state"] is None
    assert consumed["task_attempt_id"] == attempt.to_wire()


def test_initial_attempt_cannot_inherit_a_release_that_does_not_bind_it(monkeypatch):
    evidence_root = "memory://production-recovery-admission-initial-refusal"
    row = SimpleNamespace(trajectory_id="row-a")
    attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=0)
    wrong_attempt = _admission_attempt(0, 1, "row-a")
    release = {
        "bindings": [{**wrong_attempt, "controller_state": "TASK_STATE_RUNNING"}],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    monkeypatch.setattr(gate, "ADMISSION_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="worker-0-0", worker_region="us-central1"),
    )
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)

    with pytest.raises(TimeoutError, match="Timed out waiting for an admission release"):
        gate._wait_for_admission_release(
            task_index=0,
            row=row,
            attempt=attempt,
            ready_at="2026-08-13T12:02:00+00:00",
        )


def test_worker_rejects_release_with_wrong_binding_count(monkeypatch):
    evidence_root = "memory://production-recovery-admission-binding-count"
    row = SimpleNamespace(trajectory_id="row-a")
    attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=0)
    release = {
        "bindings": [
            {**_admission_attempt(0, 0, "row-a"), "controller_state": "TASK_STATE_RUNNING"},
            {**_admission_attempt(1, 0, "row-b"), "controller_state": "TASK_STATE_RUNNING"},
        ],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    monkeypatch.setattr(
        gate,
        "get_job_info",
        lambda: SimpleNamespace(worker_id="worker-0-0", worker_region="us-central1"),
    )
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)

    with pytest.raises(TypeError, match="does not bind C64"):
        gate._wait_for_admission_release(
            task_index=0,
            row=row,
            attempt=attempt,
            ready_at="2026-08-13T12:00:00+00:00",
        )


def test_admission_analyzer_accepts_exact_initial_release_and_consumption(monkeypatch):
    evidence_root = "memory://production-recovery-admission-analysis"
    row = SimpleNamespace(trajectory_id="row-a")
    attempt = _admission_attempt(0, 0, "row-a")
    release = {
        "bindings": [{**attempt, "controller_state": "TASK_STATE_RUNNING"}],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)
    release_url = gate._admission_release_objects()[0]
    with fsspec.open(release_url, "rb") as handle:
        release_sha256 = hashlib.sha256(handle.read()).hexdigest()
    consumed = {
        **attempt,
        "admission_mode": "bound_current_attempt",
        "consumed_at": "2026-08-13T12:01:01+00:00",
        "controller_state": "TASK_STATE_RUNNING",
        "generation": gate.GENERATION,
        "release_epoch": 0,
        "release_path": release_url,
        "release_sha256": release_sha256,
        "stage": 1,
    }
    consumed_fs, consumed_path = fsspec.core.url_to_fs(gate._admission_consumed_path(0, 0))
    consumed_fs.makedirs(str(Path(consumed_path).parent), exist_ok=True)
    stress._write_json_once(consumed_fs, consumed_path, consumed)

    observed, failures = analyzer._admission_evidence((row,), {0: [attempt]}, {0: {}})

    assert failures == []
    assert observed["consumed"][0][0]["release_epoch"] == 0
    assert observed["releases"][0]["release_sha256"] == release_sha256


def test_admission_analyzer_rejects_unauthorized_retry_that_predates_release(monkeypatch):
    evidence_root = "memory://production-recovery-admission-analysis-retry"
    row = SimpleNamespace(trajectory_id="row-a")
    initial = _admission_attempt(0, 0, "row-a")
    retry = _admission_attempt(0, 1, "row-a")
    retry["ready_at"] = "2026-08-13T12:00:30+00:00"
    release = {
        "bindings": [{**initial, "controller_state": "TASK_STATE_RUNNING"}],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)
    release_url = gate._admission_release_objects()[0]
    with fsspec.open(release_url, "rb") as handle:
        release_sha256 = hashlib.sha256(handle.read()).hexdigest()
    consumed = {
        **retry,
        "admission_mode": "post_release_retry",
        "consumed_at": "2026-08-13T12:01:01+00:00",
        "controller_state": None,
        "generation": gate.GENERATION,
        "release_epoch": 0,
        "release_path": release_url,
        "release_sha256": release_sha256,
        "stage": 1,
    }
    consumed_fs, consumed_path = fsspec.core.url_to_fs(gate._admission_consumed_path(0, 1))
    consumed_fs.makedirs(str(Path(consumed_path).parent), exist_ok=True)
    stress._write_json_once(consumed_fs, consumed_path, consumed)

    _, failures = analyzer._admission_evidence((row,), {0: [initial, retry]}, {0: {}})

    assert "task 0 post-release retry 1 lacks recovery authorization" in failures
    assert "task 0 attempt 1 predates its post-release admission" in failures


def test_trainer_initialization_must_follow_admission_consumption():
    state = {0: {0: {"written_at": "2026-08-13T12:01:02+00:00"}}}
    admission = {
        "consumed": {0: {0: {"consumed_at": "2026-08-13T12:01:01+00:00", "release_epoch": 0}}},
        "releases": {0: {"released_at": "2026-08-13T12:01:00+00:00"}},
    }

    assert analyzer._initialization_after_admission(state, admission) == []
    assert analyzer._initialization_after_admission(state, {"consumed": {}, "releases": {}}) == [
        "task 0 attempt 0 initialized without admission"
    ]
    admission["consumed"][0][0]["consumed_at"] = "2026-08-13T12:01:03+00:00"
    assert analyzer._initialization_after_admission(state, admission) == [
        "task 0 attempt 0 initialized before admission consumption"
    ]


def test_malformed_admission_receipt_fails_without_analyzer_exception(monkeypatch):
    evidence_root = "memory://production-recovery-admission-malformed"
    row = SimpleNamespace(trajectory_id="row-a")
    attempt = _admission_attempt(0, 0, "row-a")
    release = {
        "bindings": [{**attempt, "controller_state": "TASK_STATE_RUNNING"}],
        "epoch": 0,
        "generation": gate.GENERATION,
        "parent_worker_id": "parent-worker",
        "parent_worker_region": "us-central1",
        "released_at": "2026-08-13T12:01:00+00:00",
        "stage": 1,
    }
    malformed = {
        **attempt,
        "admission_mode": "bound_current_attempt",
        "consumed_at": "2026-08-13T12:01:01+00:00",
        "controller_state": "TASK_STATE_RUNNING",
        "generation": gate.GENERATION,
        "release_epoch": "not-an-integer",
        "stage": 1,
    }
    monkeypatch.setattr(gate, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(gate, "STAGE", 1)
    fs, path = fsspec.core.url_to_fs(gate._admission_release_path(0))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, release)
    consumed_fs, consumed_path = fsspec.core.url_to_fs(gate._admission_consumed_path(0, 0))
    consumed_fs.makedirs(str(Path(consumed_path).parent), exist_ok=True)
    stress._write_json_once(consumed_fs, consumed_path, malformed)

    observed, failures = analyzer._admission_evidence((row,), {0: [attempt]}, {0: {}})

    assert observed["consumed"] == {0: {}}
    assert failures == ["task 0 attempt 0 has malformed admission epoch"]
    assert analyzer._initialization_after_admission({0: {0: {"written_at": "2026-08-13T12:01:02+00:00"}}}, observed) == [
        "task 0 attempt 0 initialized without admission"
    ]


def test_supervisor_kicks_exact_attempt_not_logical_task(monkeypatch):
    fault = gate.FaultInjection(task_index=0, trigger_step=100, phase="test")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=0)
    checkpoint = gate.CheckpointClaim(
        path="gs://marin-us-central1/temporary/step-90",
        step=90,
        metadata_sha256="sha-90",
        search_paths=("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary"),
    )
    kicked: list[str] = []
    receipts: list[dict[str, Any]] = []

    class FakeClient:
        @staticmethod
        def kick_tasks(task_ids, *, desired_state, reason):
            assert desired_state == gate.job_pb2.TASK_STATE_PREEMPTED
            assert "generation" in reason
            kicked.extend(task_ids)
            return [SimpleNamespace(queued=True)]

    class FakeJob:
        job_id = JobName.from_wire("/calvin/child")

        def __init__(self):
            self.states = iter((gate.job_pb2.JOB_STATE_RUNNING, gate.job_pb2.JOB_STATE_SUCCEEDED))

        def state_only(self):
            return next(self.states)

        @staticmethod
        def wait(*, timeout, raise_on_failure):
            assert timeout == 60
            assert raise_on_failure is False
            return SimpleNamespace(state=gate.job_pb2.JOB_STATE_SUCCEEDED)

    monkeypatch.setattr(gate, "FAULT_INJECTIONS", (fault,))
    monkeypatch.setattr(gate, "_authorize_retry_attempts", lambda *_, **__: None)
    monkeypatch.setattr(gate, "_release_current_admission_epoch", lambda *_, **__: None)
    monkeypatch.setattr(gate, "_checkpoint_claim", lambda _: checkpoint)
    monkeypatch.setattr(
        gate,
        "_all_attempt_records",
        lambda _: {0: ({"attempt": 0, "task_attempt_id": task_attempt.to_wire()},)},
    )
    monkeypatch.setattr(gate, "_write_fault_receipt", lambda *_, **kwargs: receipts.append(kwargs))
    monkeypatch.setattr(gate, "WandbProgress", lambda: SimpleNamespace(global_step=lambda _: 100))
    monkeypatch.setattr(gate, "iris_ctx", lambda: SimpleNamespace(client=FakeClient()))
    monkeypatch.setattr(gate.time, "sleep", lambda _: None)

    gate._supervise_gate(FakeJob(), (SimpleNamespace(trajectory_id="row-a"),), (_minimal_config(),))

    assert kicked == [task_attempt.to_wire()]
    assert receipts[0]["task_attempt_id"] == task_attempt.to_wire()


def test_supervisor_waits_for_completed_temporary_checkpoint(monkeypatch):
    fault = gate.FaultInjection(task_index=0, trigger_step=100, phase="test")
    task_attempt = TaskAttempt(task_id=JobName.from_wire("/calvin/child/0"), attempt_id=0)
    search_paths = ("gs://marin-us-central1/permanent", "gs://marin-us-central1/temporary")
    checkpoints = iter(
        (
            gate.CheckpointClaim(
                path="gs://marin-us-central1/permanent/step-80",
                step=80,
                metadata_sha256="sha-80",
                search_paths=search_paths,
            ),
            gate.CheckpointClaim(
                path="gs://marin-us-central1/temporary/step-90",
                step=90,
                metadata_sha256="sha-90",
                search_paths=search_paths,
            ),
        )
    )
    kicked: list[str] = []

    class FakeClient:
        @staticmethod
        def kick_tasks(task_ids, *, desired_state, reason):
            assert desired_state == gate.job_pb2.TASK_STATE_PREEMPTED
            assert reason
            kicked.extend(task_ids)
            return [SimpleNamespace(queued=True)]

    class FakeJob:
        job_id = JobName.from_wire("/calvin/child")

        def __init__(self):
            self.states = iter(
                (
                    gate.job_pb2.JOB_STATE_RUNNING,
                    gate.job_pb2.JOB_STATE_RUNNING,
                    gate.job_pb2.JOB_STATE_SUCCEEDED,
                )
            )

        def state_only(self):
            return next(self.states)

        @staticmethod
        def wait(*, timeout, raise_on_failure):
            assert timeout == 60
            assert raise_on_failure is False
            return SimpleNamespace(state=gate.job_pb2.JOB_STATE_SUCCEEDED)

    monkeypatch.setattr(gate, "FAULT_INJECTIONS", (fault,))
    monkeypatch.setattr(gate, "_authorize_retry_attempts", lambda *_, **__: None)
    monkeypatch.setattr(gate, "_release_current_admission_epoch", lambda *_, **__: None)
    monkeypatch.setattr(gate, "_checkpoint_claim", lambda _: next(checkpoints))
    monkeypatch.setattr(
        gate,
        "_all_attempt_records",
        lambda _: {0: ({"attempt": 0, "task_attempt_id": task_attempt.to_wire()},)},
    )
    monkeypatch.setattr(gate, "_write_fault_receipt", lambda *_, **__: None)
    monkeypatch.setattr(gate, "WandbProgress", lambda: SimpleNamespace(global_step=lambda _: 100))
    monkeypatch.setattr(gate, "iris_ctx", lambda: SimpleNamespace(client=FakeClient()))
    monkeypatch.setattr(gate.time, "sleep", lambda _: None)

    gate._supervise_gate(FakeJob(), (SimpleNamespace(trajectory_id="row-a"),), (_minimal_config(),))

    assert kicked == [task_attempt.to_wire()]


def test_freeze_binds_production_recovery_and_gen19_prerequisite():
    preregistration = freeze.build_preregistration()

    assert preregistration["stage"] == 64
    assert preregistration["admission_barrier"] == {
        "all_current_attempts_required": 64,
        "controller_state_required": "TASK_STATE_RUNNING",
        "immutable_attempt_qualified_initial_release": True,
        "post_release_retries_require_recovery_authorization": True,
        "pretraining": True,
        "release_binds_controller_and_worker_identity": True,
        "timeout_seconds": 2 * 60 * 60,
        "trainer_initialization_requires_release_consumption": True,
    }
    assert preregistration["checkpoint_recovery"]["interval_seconds"] == 5 * 60
    assert preregistration["checkpoint_recovery"]["forced_fault_requires_temporary_checkpoint"] is True
    assert preregistration["checkpoint_recovery"]["forced_retry_requires_temporary_checkpoint"] is True
    assert preregistration["checkpoint_recovery"]["parent_rejects_non_temporary_forced_retry"] is True
    assert preregistration["checkpoint_recovery"]["retry_modes"] == [
        gate.RECOVERY_MODE_CHECKPOINT,
        gate.RECOVERY_MODE_RESTART_FROM_ZERO,
    ]
    assert preregistration["integrity"]["job_timeout_seconds"] == 4 * 60 * 60
    assert preregistration["integrity"]["recovery_authorization_timeout_seconds"] == 20 * 60
    assert preregistration["gates"]["post_forced_recovery_c64_active_overlap_seconds_min"] == 120.0
    assert preregistration["endpoint_metrics_read"] is False
    assert preregistration["prior_gate"] == {
        "analyzer_revision_sha256": gate.GEN19_ANALYZER_REVISION_SHA256,
        "pass_report_sha256": gate.GEN19_PASS_REPORT_SHA256,
        "preregistration_sha256": gate.GEN19_PREREGISTRATION_SHA256,
        "review_session_id": gate.GEN19_REVIEW_SESSION_ID,
        "review_verdict": gate.GEN19_REVIEW_VERDICT,
    }
    assert set(
        (
            "base_launcher",
            "dense_launcher",
            "trainer",
            "callbacks_core",
            "callbacks_metrics",
            "marin_training",
            "output_inventory",
            "runtime_gate",
            "support_audit",
            "recovery_tests",
        )
    ) <= set(preregistration["implementation_sha256"])
