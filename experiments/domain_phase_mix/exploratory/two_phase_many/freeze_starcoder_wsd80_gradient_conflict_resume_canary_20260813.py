# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Freeze the generation-19 per-task checkpoint-resume mechanism canary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_resume_canary as resume
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYZER = Path(__file__).with_name("analyze_starcoder_wsd80_gradient_conflict_resume_canary_20260813.py")
TESTS = REPO_ROOT / "tests/test_starcoder_wsd80_gradient_conflict_canary.py"
TRAIN_LM = REPO_ROOT / "lib/levanter/src/levanter/main/train_lm.py"
LOADER = REPO_ROOT / "lib/levanter/src/levanter/data/loader.py"
DATASETS = REPO_ROOT / "lib/levanter/src/levanter/data/text/datasets.py"
CHECKPOINT = REPO_ROOT / "lib/levanter/src/levanter/checkpoint.py"
TRACKER = REPO_ROOT / "lib/levanter/src/levanter/tracker/wandb.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = stress.rows_for_stage(resume.STAGE)
    preregistration = {
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "checkpoint_recovery": {
            "automatic_run_local_discovery": True,
            "interval_seconds": int(resume.CHECKPOINT_INTERVAL.total_seconds()),
            "keep_last_temporary_checkpoints": 1,
            "restart_step_tolerance": 0,
            "stable_output_identity_across_attempts": True,
            "stable_wandb_identity_across_attempts": True,
        },
        "decision": {
            "next_on_fail": "diagnose this bounded mechanism canary; do not launch C64",
            "next_on_pass": "freeze one production-cadence C64 recovery gate; do not add concurrency rungs",
            "reason": (
                "Generation 18 admitted all 64 independent tasks but a routine preemption invalidated a gate that "
                "disabled the retry and checkpoint semantics used by production. This C6 is a mechanism test, not "
                "another load rung."
            ),
        },
        "endpoint_metrics_read": False,
        "fault_injections": [asdict(fault) for fault in resume.FAULT_INJECTIONS],
        "generation": resume.GENERATION,
        "gates": {
            "all_rows_reach_terminal_step": True,
            "child_preemptions_min": len(resume.FAULT_INJECTIONS),
            "each_forced_row_has_retry_claim": True,
            "each_forced_retry_discovers_nonzero_checkpoint": True,
            "forced_retry_loads_parent_checkpoint_or_newer": True,
            "post_initialization_state_matches_worker_checkpoint_claim": True,
            "wandb_operational_history_step_complete": True,
            "iris_parent_and_child_succeed": True,
            "no_endpoint_metrics": True,
            "remote_preregistration_hash_matches": True,
            "fresh_checkpoint_and_wandb_namespaces": True,
        },
        "implementation_sha256": {
            "analyzer": _sha256(ANALYZER),
            "checkpoint": _sha256(CHECKPOINT),
            "datasets": _sha256(DATASETS),
            "full_launcher": _sha256(Path(full.__file__)),
            "launcher": _sha256(Path(resume.__file__)),
            "loader": _sha256(LOADER),
            "stress_launcher": _sha256(Path(stress.__file__)),
            "tests": _sha256(TESTS),
            "tracker": _sha256(TRACKER),
            "train_lm": _sha256(TRAIN_LM),
        },
        "integrity": {
            "application_failure_retries": 0,
            "child_preemptible": True,
            "cross_replica_jax": False,
            "iris_topology_coscheduling": False,
            "max_preemption_retries_per_task": resume.MAX_PREEMPTION_RETRIES,
            "parent_preemptible": False,
            "parent_retries": 0,
            "placement_region": "us-central1",
            "placement_zone": "us-central1-a",
        },
        "rows": {
            "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
            "support_ids": [row.support_id for row in rows],
            "training_seeds": [row.training_seed for row in rows],
            "trajectory_ids": [row.trajectory_id for row in rows],
        },
        "stage": resume.STAGE,
        "version": "2026-08-13-c06-per-task-resume-canary-v4",
    }
    payload = json.dumps(preregistration, indent=2, sort_keys=True) + "\n"
    resume.PREREGISTRATION_PATH.write_text(payload)
    print(json.dumps({"path": str(resume.PREREGISTRATION_PATH), "sha256": _sha256(resume.PREREGISTRATION_PATH)}))


if __name__ == "__main__":
    main()
