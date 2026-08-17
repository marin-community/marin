# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Supervisor recovery-loop behavior, exercised with a fake CPU trainer.

Covers the full state machine (clean run, recover from sticky/crash/hang,
propagate hard failures, exhaust the restart budget, health-gate escalation)
without a GPU by spawning ``levanter.recovery.fake_trainer.fake_trainer``.
"""

from __future__ import annotations

import signal

import pytest

from levanter.recovery import supervisor as sup
from levanter.recovery.fake_trainer import ENV_BEHAVIOR, fake_trainer
from levanter.recovery.detection import DetectionConfig
from levanter.recovery.supervisor import GPUHangSupervisor
from levanter.recovery.types import FaultClass, RunOutcome, classify_returncode


def _make_supervisor(tmp_path, **overrides):
    kwargs = dict(
        detection=DetectionConfig(execution_terminate_timeout_seconds=60.0),
        deadman_timeout=3.0,
        snapshot_tmpfs_base=str(tmp_path / "shm"),
        poll_interval=0.25,
        startup_timeout=30.0,
        work_dir=str(tmp_path / "work"),
    )
    kwargs.update(overrides)
    return GPUHangSupervisor(**kwargs)


def test_clean_run_no_faults(tmp_path):
    with _make_supervisor(tmp_path) as s:
        result = s.run(fake_trainer, {"steps": 5}, label="clean", env={ENV_BEHAVIOR: "complete"})
    assert result.outcome is RunOutcome.COMPLETED
    assert result.attempts == 1
    assert result.faults == []
    assert result.final_step == 5


@pytest.mark.parametrize(
    "behavior,expected_class",
    [
        ("sticky", FaultClass.STICKY),
        ("crash", FaultClass.CRASH),
        ("hang", FaultClass.STALL),
    ],
)
def test_recovers_from_one_off_fault(tmp_path, behavior, expected_class):
    with _make_supervisor(tmp_path) as s:
        result = s.run(
            fake_trainer,
            {"steps": 6, "fault_step": 2},
            label=f"recover-{behavior}",
            env={ENV_BEHAVIOR: behavior},
        )
    assert result.outcome is RunOutcome.COMPLETED
    assert result.recovered
    assert result.attempts == 2
    assert [f.fault_class for f in result.faults] == [expected_class]
    assert result.restored_from_snapshot


def test_deadman_does_not_override_a_concurrent_child_crash():
    assert classify_returncode(-signal.SIGABRT, deadman=True) is FaultClass.CRASH
    assert classify_returncode(-signal.SIGTERM, deadman=True) is FaultClass.STALL


def test_hard_failure_is_not_restarted(tmp_path):
    with _make_supervisor(tmp_path) as s:
        result = s.run(
            fake_trainer,
            {"steps": 6, "fault_step": 1},
            label="hard",
            env={ENV_BEHAVIOR: "hard"},
        )
    assert result.outcome is RunOutcome.FAILED
    assert result.attempts == 1
    assert result.faults[-1].fault_class is FaultClass.HARD


def test_restart_budget_exhausted(tmp_path):
    with _make_supervisor(tmp_path, max_restarts_per_run=2) as s:
        result = s.run(
            fake_trainer,
            {"steps": 6, "fault_step": 1, "always": True},
            label="budget",
            env={ENV_BEHAVIOR: "sticky"},
        )
    assert result.outcome is RunOutcome.FAILED
    # 1 initial attempt + 2 restarts, all faulting.
    assert result.attempts == 3
    assert all(f.fault_class is FaultClass.STICKY for f in result.faults)


def test_unhealthy_gpu_escalates(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "_gpu_is_healthy", lambda *a, **k: False)
    with _make_supervisor(tmp_path) as s:
        result = s.run(
            fake_trainer,
            {"steps": 6, "fault_step": 1},
            label="unhealthy",
            env={ENV_BEHAVIOR: "crash"},
        )
    assert result.outcome is RunOutcome.UNHEALTHY_GPU
    assert result.attempts == 1
