# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_resume_canary as resume
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_resume_canary_revision2_20260813 as analyzer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_conflict_resume_analyzer_revision2_20260813 as freeze,
)


def test_checkpoint_label_restores_the_next_trainer_step(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_step": 123,
                "initial_state_evidence_path": "state-attempt-1",
            }
        ]
    }
    monkeypatch.setattr(
        analyzer.original,
        "_read_json",
        lambda _: {"run_id": resume._run_id(row), "state_step": 124},
    )

    observed, failures = analyzer._state_evidence((row,), attempts)

    assert failures == []
    assert observed[0][1]["state_step"] == 124


def test_checkpoint_label_rejects_the_completed_step_as_restored_state(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_step": 123,
                "initial_state_evidence_path": "state-attempt-1",
            }
        ]
    }
    monkeypatch.setattr(
        analyzer.original,
        "_read_json",
        lambda _: {"run_id": resume._run_id(row), "state_step": 123},
    )

    _, failures = analyzer._state_evidence((row,), attempts)

    assert failures == ["task 0 attempt 1 initialized at step 123, expected 124 from checkpoint label step-123"]


def test_forced_retry_accepts_parent_checkpoint_or_newer_with_next_step_state():
    parent = {
        "checkpoint_path": "gs://bucket/checkpoints/step-123",
        "checkpoint_step": 123,
        "checkpoint_metadata_sha256": "abc",
        "source_attempt": 0,
    }
    worker = {
        "attempt": 1,
        "checkpoint_path": "gs://bucket/checkpoints/step-130",
        "checkpoint_step": 130,
        "checkpoint_metadata_sha256": "def",
    }

    matched, failures = analyzer._forced_retry_evidence(
        {0: [worker]},
        {0: parent},
        {0: {1: {"state_step": 131}}},
    )

    assert failures == []
    assert matched[0] == worker


def test_wandb_gate_accepts_exact_operational_hook_domain(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    history = [
        {
            "_step": step - analyzer.FIRST_OPERATIONAL_WANDB_STEP,
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
    monkeypatch.setattr(analyzer.resume, "FAULT_INJECTIONS", ())

    evidence, failures = analyzer._wandb_evidence((row,), {0: []}, {})

    assert failures == []
    assert evidence[row.trajectory_id]["history_rows"] == stress.TERMINAL_STEP - 1
    assert evidence[row.trajectory_id]["status"] == "pass"


def test_wandb_gate_rejects_a_missing_emitted_step(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    history = [
        {"_step": step, "global_step": step}
        for step in range(analyzer.FIRST_OPERATIONAL_WANDB_STEP, stress.TERMINAL_STEP + 1)
        if step != 100
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
    monkeypatch.setattr(analyzer.resume, "FAULT_INJECTIONS", ())

    evidence, failures = analyzer._wandb_evidence((row,), {0: []}, {})

    assert evidence[row.trajectory_id]["status"] == "fail"
    assert len(failures) == 1
    assert "missing=[100]" in failures[0]


def test_revision_freeze_binds_original_failure_and_only_two_corrections():
    revision = freeze.build_revision()

    assert revision["original_preregistration_sha256"] == analyzer.ORIGINAL_PREREGISTRATION_SHA256
    assert revision["original_failed_report_sha256"] == analyzer.ORIGINAL_FAILED_REPORT_SHA256
    assert revision["semantic_corrections"] == {
        "checkpoint_metadata_step_to_restored_state_step": "N_to_N_plus_1",
        "operational_wandb_global_steps": [2, stress.TERMINAL_STEP],
    }
    assert revision["unchanged_contract"]["no_threshold_changed"] is True
    assert revision["endpoint_metrics_read"] is False
