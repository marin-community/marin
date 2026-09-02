# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Freeze the Generation-24 admission-gated production C64 checkpoint-recovery gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_production_recovery_gate as gate
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYZER = Path(__file__).with_name("analyze_starcoder_wsd80_gradient_conflict_production_recovery_gate_20260813.py")
TESTS = REPO_ROOT / "tests/test_starcoder_wsd80_gradient_conflict_canary.py"
RECOVERY_TESTS = REPO_ROOT / "tests/test_starcoder_wsd80_gradient_conflict_production_recovery_gate.py"
TRAIN_LM = REPO_ROOT / "lib/levanter/src/levanter/main/train_lm.py"
TRAINER = REPO_ROOT / "lib/levanter/src/levanter/trainer.py"
CALLBACKS_CORE = REPO_ROOT / "lib/levanter/src/levanter/callbacks/_core.py"
CALLBACKS_METRICS = REPO_ROOT / "lib/levanter/src/levanter/callbacks/_metrics.py"
LOADER = REPO_ROOT / "lib/levanter/src/levanter/data/loader.py"
DATASETS = REPO_ROOT / "lib/levanter/src/levanter/data/text/datasets.py"
CHECKPOINT = REPO_ROOT / "lib/levanter/src/levanter/checkpoint.py"
TRACKER = REPO_ROOT / "lib/levanter/src/levanter/tracker/wandb.py"
RUNTIME_GATE = Path(__file__).with_name("analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811.py")
BASE_LAUNCHER = REPO_ROOT / "experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py"
DENSE_LAUNCHER = REPO_ROOT / "experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py"
OUTPUT_INVENTORY = Path(full.output_inventory.__file__)
SUPPORT_AUDIT = Path(full.support_audit.__file__)
MARIN_TRAINING = REPO_ROOT / "lib/marin/src/marin/training/training.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preregistration() -> dict[str, Any]:
    """Build the exact endpoint-blind Generation-24 gate contract."""
    rows = stress.rows_for_stage(gate.STAGE)
    return {
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "admission_barrier": {
            "all_current_attempts_required": gate.STAGE,
            "controller_state_required": "TASK_STATE_RUNNING",
            "immutable_attempt_qualified_initial_release": True,
            "post_release_retries_require_recovery_authorization": True,
            "pretraining": True,
            "release_binds_controller_and_worker_identity": True,
            "timeout_seconds": gate.ADMISSION_TIMEOUT_SECONDS,
            "trainer_initialization_requires_release_consumption": True,
        },
        "checkpoint_recovery": {
            "automatic_run_local_discovery": True,
            "full_state_run_local_resume": True,
            "interval_seconds": int(gate.CHECKPOINT_INTERVAL.total_seconds()),
            "keep_last_temporary_checkpoints": 1,
            "restart_step_tolerance": 0,
            "stable_output_identity_across_attempts": True,
            "stable_wandb_identity_across_attempts": True,
            "every_retry_requires_parent_authorization": True,
            "forced_fault_requires_temporary_checkpoint": True,
            "forced_retry_requires_temporary_checkpoint": True,
            "parent_rejects_non_temporary_forced_retry": True,
            "parent_independently_attests_worker_recovery_claim": True,
            "retry_modes": [gate.RECOVERY_MODE_CHECKPOINT, gate.RECOVERY_MODE_RESTART_FROM_ZERO],
        },
        "decision": {
            "next_on_fail": "diagnose this single production-cadence C64 gate; do not launch training",
            "next_on_pass": "freeze and independently review the 256-trajectory training release",
            "reason": (
                "Generation 19 established exact full-state recovery under forced preemption. This final operational "
                "gate combines those semantics with 64 independent workers, the unreleased training panel's "
                "five-minute checkpoint cadence, a parent-issued admission barrier binding all 64 live attempts, "
                "forced recovery from a completed temporary checkpoint, "
                "parent-authorized recovery for natural preemptions, and sustained C64 activity after forced "
                "recovery; no additional concurrency rung is authorized."
            ),
        },
        "endpoint_metrics_read": False,
        "fault_injections": [asdict(fault) for fault in gate.FAULT_INJECTIONS],
        "generation": gate.GENERATION,
        "gates": {
            "all_rows_reach_terminal_step": True,
            "all_trainer_initializations_follow_admission_release_or_authorized_post_release_retry": True,
            "admission_release_binds_64_live_current_attempts": True,
            "child_preemptions_min": len(gate.FAULT_INJECTIONS),
            "each_forced_row_has_retry_claim": True,
            "each_forced_retry_discovers_nonzero_temporary_checkpoint": True,
            "each_natural_retry_uses_parent_authorized_checkpoint_or_restart_from_zero": True,
            "forced_retry_loads_parent_checkpoint_or_newer": True,
            "restart_from_zero_is_allowed_only_before_any_checkpoint_exists": True,
            "post_initialization_state_matches_worker_checkpoint_claim": True,
            "wandb_operational_history_covers_emitted_steps_2_through_3327": True,
            "post_forced_recovery_c64_active_overlap_seconds_min": 120.0,
            "iris_parent_and_child_succeed": True,
            "no_endpoint_metrics": True,
            "remote_preregistration_hash_matches": True,
            "fresh_checkpoint_and_wandb_namespaces": True,
        },
        "implementation_sha256": {
            "analyzer": _sha256(ANALYZER),
            "base_launcher": _sha256(BASE_LAUNCHER),
            "callbacks_core": _sha256(CALLBACKS_CORE),
            "callbacks_metrics": _sha256(CALLBACKS_METRICS),
            "checkpoint": _sha256(CHECKPOINT),
            "datasets": _sha256(DATASETS),
            "dense_launcher": _sha256(DENSE_LAUNCHER),
            "full_launcher": _sha256(Path(full.__file__)),
            "launcher": _sha256(Path(gate.__file__)),
            "loader": _sha256(LOADER),
            "marin_training": _sha256(MARIN_TRAINING),
            "output_inventory": _sha256(OUTPUT_INVENTORY),
            "recovery_tests": _sha256(RECOVERY_TESTS),
            "runtime_gate": _sha256(RUNTIME_GATE),
            "stress_launcher": _sha256(Path(stress.__file__)),
            "support_audit": _sha256(SUPPORT_AUDIT),
            "tests": _sha256(TESTS),
            "tracker": _sha256(TRACKER),
            "train_lm": _sha256(TRAIN_LM),
            "trainer": _sha256(TRAINER),
        },
        "integrity": {
            "application_failure_retries": 0,
            "child_preemptible": True,
            "cross_replica_jax": False,
            "iris_topology_coscheduling": False,
            "job_timeout_seconds": gate.JOB_TIMEOUT_SECONDS,
            "max_preemption_retries_per_task": gate.MAX_PREEMPTION_RETRIES,
            "parent_preemptible": False,
            "parent_retries": 0,
            "placement_region": "us-central1",
            "placement_zone": "us-central1-a",
            "recovery_authorization_timeout_seconds": gate.RECOVERY_AUTHORIZATION_TIMEOUT_SECONDS,
        },
        "prior_gate": {
            "analyzer_revision_sha256": gate.GEN19_ANALYZER_REVISION_SHA256,
            "pass_report_sha256": gate.GEN19_PASS_REPORT_SHA256,
            "preregistration_sha256": gate.GEN19_PREREGISTRATION_SHA256,
            "review_session_id": gate.GEN19_REVIEW_SESSION_ID,
            "review_verdict": gate.GEN19_REVIEW_VERDICT,
        },
        "rows": {
            "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
            "support_ids": [row.support_id for row in rows],
            "training_seeds": [row.training_seed for row in rows],
            "trajectory_ids": [row.trajectory_id for row in rows],
        },
        "stage": gate.STAGE,
        "version": "2026-08-13-c64-admission-production-recovery-v6",
    }


def main() -> None:
    preregistration = build_preregistration()
    payload = json.dumps(preregistration, indent=2, sort_keys=True) + "\n"
    gate.PREREGISTRATION_PATH.write_text(payload)
    print(json.dumps({"path": str(gate.PREREGISTRATION_PATH), "sha256": _sha256(gate.PREREGISTRATION_PATH)}))


if __name__ == "__main__":
    main()
