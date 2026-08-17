# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Freeze the reviewed generation-11 C6 cohort-recovery operational gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811 as runtime_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/"
    "starcoder_wsd80_gradient_conflict_stress_gate_preregistration_generation11_20260812.json"
)
TEST_PATH = REPO_ROOT / "tests/test_starcoder_wsd80_gradient_conflict_canary.py"
HISTORY_AUDIT_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "backtest_starcoder_wsd80_gradient_conflict_stress_gate_20260811.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = stress.rows_for_stage(6)
    output = {
        "preregistration_version": "2026-08-12-c06-cohort-recovery-gate-v5",
        "generation": 11,
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "historical_operational_evidence_used": {
            "generation": 10,
            "scope": "runtime telemetry only; no endpoint outcomes",
            "use": (
                "identify independent-row invalidation after five infrastructure preemptions and freeze "
                "cohort-level recovery without changing rows or thresholds"
            ),
        },
        "implementation_sha256": {
            "stress_launcher": _sha256(Path(stress.__file__).resolve()),
            "runtime_gate": _sha256(Path(runtime_gate.__file__).resolve()),
            "tests": _sha256(TEST_PATH),
            "history_audit": _sha256(HISTORY_AUDIT_PATH),
        },
        "design": {
            "stage": 6,
            "total_steps": stress.TOTAL_STEPS,
            "terminal_step": stress.TERMINAL_STEP,
            "data_switch_step": stress.DATA_SWITCH_STEP,
            "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
            "support_ids": [row.support_id for row in rows],
            "training_seeds": [row.training_seed for row in rows],
            "rendezvous_ready_spread_seconds_max": runtime_gate.RENDEZVOUS_READY_SPREAD_SECONDS_MAX,
            "cohort": {
                "replicas": 6,
                "coscheduling_group_by": stress.COHORT_COSCHEDULING_GROUP,
                "parent_managed_whole_cohort_retries": stress.COHORT_MAX_PREEMPTION_RETRIES,
                "per_attempt_wait_timeout_seconds": stress.COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS,
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
                "max_task_failures": 0,
                "initialize_jax_distributed": False,
                "start_and_completion_barriers_required": True,
                "assignment_time_iris_redispatch_allowed": True,
                "barriers_scoped_by_parent_and_iris_attempt": True,
                "finished_child_history_preserved_on_parent_restart": True,
            },
        },
        "release_gate": {
            "unchanged_row_checks": {
                "loading_p99_seconds_max": runtime_gate.LOADING_P99_MAX,
                "loading_over_one_second_fraction_max": runtime_gate.LOADING_OVER_ONE_SECOND_FRACTION_MAX,
                "loading_seconds_max": runtime_gate.LOADING_MAX,
                "duty_fraction_min": runtime_gate.DUTY_FRACTION_MIN,
                "tokens_per_second_p50_min": runtime_gate.TOKENS_PER_SECOND_P50_MIN,
                "mfu_percent_p50_min": runtime_gate.MFU_P50_MIN,
                "rolling_window_steps": runtime_gate.ROLLING_WINDOW_STEPS,
                "rolling_tokens_per_second_p50_min": runtime_gate.ROLLING_TOKENS_PER_SECOND_P50_MIN,
                "history_coverage_fraction_min": runtime_gate.HISTORY_COVERAGE_FRACTION_MIN,
                "event_tokens_per_second_p50_min": runtime_gate.EVENT_TOKENS_PER_SECOND_P50_MIN,
                "event_loading_p99_seconds_max": runtime_gate.EVENT_LOADING_P99_MAX,
                "event_history_coverage_fraction_min": runtime_gate.EVENT_HISTORY_COVERAGE_FRACTION_MIN,
                "runtime_accounting_ratio_min": runtime_gate.RUNTIME_ACCOUNTING_RATIO_MIN,
                "runtime_accounting_ratio_max": runtime_gate.RUNTIME_ACCOUNTING_RATIO_MAX,
            },
            "short_window": {
                "window_steps": runtime_gate.STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
                "tokens_per_second_min": runtime_gate.STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN,
                "max_consecutive_positions": runtime_gate.STRESS_SHORT_WINDOW_MAX_POSITIONS,
                "max_exceeding_rows": runtime_gate.STRESS_SHORT_WINDOW_EXCEEDANCE_MAX_ROWS,
            },
            "synchronized_depression": {
                "quorum_fraction": runtime_gate.STRESS_SYNCHRONIZED_QUORUM_FRACTION,
                "depression_fraction": runtime_gate.STRESS_SYNCHRONIZED_DEPRESSION_FRACTION,
                "bin_seconds": runtime_gate.STRESS_SYNCHRONIZED_BIN_SECONDS,
                "grid_anchor": "all-row host-time overlap start",
                "checkpoint_exclusion_rule": (
                    "Exclude any fixed grid bin intersecting the union of declared checkpoint-pause intervals; "
                    "retain all rows on the shared host clock."
                ),
                "max_seconds": runtime_gate.STRESS_SYNCHRONIZED_MAX_SECONDS,
            },
            "pause_accounting": {
                "pause_detection_rule": (
                    "A consecutive host-time gap is a pause when delta exceeds the right step's recorded "
                    "duration plus loading time plus the fixed slack."
                ),
                "accounting_slack_seconds": runtime_gate.PAUSE_ACCOUNTING_SLACK_SECONDS,
                "declared_checkpoint_steps": [stress.DATA_SWITCH_STEP, stress.TERMINAL_STEP],
                "unexplained_pause_longest_seconds_max": runtime_gate.UNEXPLAINED_PAUSE_LONGEST_SECONDS_MAX,
                "unexplained_pause_total_seconds_max": runtime_gate.UNEXPLAINED_PAUSE_TOTAL_SECONDS_MAX,
                "checkpoint_pause_seconds_max": runtime_gate.CHECKPOINT_PAUSE_SECONDS_MAX,
                "checkpoint_pause_cross_row_spread_seconds_max": (
                    runtime_gate.CHECKPOINT_PAUSE_CROSS_ROW_SPREAD_SECONDS_MAX
                ),
            },
            "concurrency_and_integrity": {
                "concurrent_overlap_seconds_min": runtime_gate.CONCURRENT_OVERLAP_SECONDS_MIN,
                "concurrent_overlap_fraction_min": runtime_gate.CONCURRENT_OVERLAP_FRACTION_MIN,
                "runtime_start_skew_seconds_max": runtime_gate.RUNTIME_START_SKEW_SECONDS_MAX,
                "rendezvous_ready_spread_seconds_max": runtime_gate.RENDEZVOUS_READY_SPREAD_SECONDS_MAX,
                "finite_to_full_loading_p99_ratio_max": runtime_gate.FINITE_TO_FULL_LOADING_P99_RATIO_MAX,
                "permanent_checkpoint_bytes_max": runtime_gate.PERMANENT_CHECKPOINT_BYTES_MAX,
                "clean_iris_completion_required": True,
                "historical_infrastructure_preemptions_allowed": True,
                "application_failures_allowed": False,
                "cohort_coscheduling_required": True,
                "latest_released_attempt_required": True,
                "one_complete_attempt_membership_required": True,
                "one_shared_iris_attempt_per_barrier_required": True,
                "parent_managed_whole_cohort_replacement_required": True,
                "preempted_child_attempts_must_have_zero_application_failures": True,
                "completion_barrier_required": True,
                "fresh_checkpoint_namespace_per_attempt_required": True,
                "fresh_wandb_namespace_per_attempt_required": True,
                "cross_replica_jax_forbidden": True,
                "checkpoint_inventory_required": True,
                "all_event_windows_inside_all_row_overlap_required": True,
                "unique_rendezvous_worker_claims_required": True,
                "remote_preregistration_hash_required": True,
            },
        },
        "diagnostic_only": {
            "event_recovery": True,
            "decay_alignment": {
                "c06_classification": "underpowered",
                "primary_support_id": "m100a",
                "c12_identification_requirement": (
                    "Eight same-support rows in four replicated onset cohorts; exact onset-assignment permutation "
                    "inference at alpha 0.01 with fixed pretrend and placebo falsifications."
                ),
            },
            "decision_rule": (
                "C6 event-relative summaries never override an operational gate. Causal attribution is deferred "
                "to the separately preregistered C12 design."
            ),
        },
        "predecessor": {
            "generation": 10,
            "preregistration_sha256": "a987fd32a2867dab92f85209b210fc4dd14284a8cf479473d896650c99bcbc46",
            "runtime_gate_status": "fail",
            "failure": (
                "five of six independently scheduled rows were preempted after the phase boundary, so the "
                "frozen zero-preemption, attempt-zero gate could not certify one synchronized cohort"
            ),
            "integrity_rule": "Generation 10 remains failed and cannot release C12.",
        },
        "next_stage": {
            "on_pass": (
                "C12 only after this exact report passes and its SHA-256 is pinned; C12 requires a separately "
                "reviewed generation-12 preregistration"
            ),
            "on_fail": "Do not launch C12; diagnose without changing this generation's thresholds",
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"path": str(OUTPUT_PATH), "sha256": _sha256(OUTPUT_PATH)}))


if __name__ == "__main__":
    main()
