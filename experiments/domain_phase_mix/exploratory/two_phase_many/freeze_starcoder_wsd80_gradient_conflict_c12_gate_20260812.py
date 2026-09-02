# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Freeze the generation-12 C12 operational and randomized-onset gate."""

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
    "starcoder_wsd80_gradient_conflict_stress_gate_preregistration_generation14_20260812.json"
)
TEST_PATH = REPO_ROOT / "tests/test_starcoder_wsd80_gradient_conflict_canary.py"
PREDECESSOR_BUNDLE_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_gradient_conflict_stress_retry11_results_20260812/"
    "generation11_reproducibility_bundle.tar.gz"
)
PREDECESSOR_MANIFEST_PATH = PREDECESSOR_BUNDLE_PATH.with_name("generation11_reproducibility_manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = stress.rows_for_stage(12)
    primary_rows = tuple(row for row in rows if row.support_id == "m100a")
    output = {
        "preregistration_version": "2026-08-12-c12-randomized-onset-gate-v3",
        "generation": 14,
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "implementation_sha256": {
            "stress_launcher": _sha256(Path(stress.__file__).resolve()),
            "runtime_gate": _sha256(Path(runtime_gate.__file__).resolve()),
            "tests": _sha256(TEST_PATH),
            "gradient_conflict_full": _sha256(Path(stress.full.__file__).resolve()),
            "wsd80_surface": _sha256(Path(stress.base.__file__).resolve()),
        },
        "design": {
            "stage": 12,
            "total_steps": stress.TOTAL_STEPS,
            "terminal_step": stress.TERMINAL_STEP,
            "data_switch_step": stress.DATA_SWITCH_STEP,
            "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
            "support_ids": [row.support_id for row in rows],
            "training_seeds": [row.training_seed for row in rows],
            "primary_randomized_onset_cohort": {
                "support_id": "m100a",
                "row_count": len(primary_rows),
                "assignment_seed": stress.C12_ONSET_ASSIGNMENT_SEED,
                "assignment_algorithm": "Python random.Random(seed).shuffle over two copies of each onset",
                "assignment_initial_multiset_order": list(stress.C12_PRIMARY_ONSET_MULTISET),
                "candidate_onset_steps": sorted(set(stress.C12_PRIMARY_OPTIMIZER_DECAY_STEPS)),
                "assigned_onset_steps_in_trajectory_order": list(stress.C12_PRIMARY_OPTIMIZER_DECAY_STEPS),
                "replicas_per_onset": 2,
            },
            "control_rows": {
                "support_ids": [row.support_id for row in rows if row.support_id != "m100a"],
                "optimizer_decay_steps": [row.optimizer_decay_step for row in rows if row.support_id != "m100a"],
                "role": "loader and support controls; excluded from onset-assignment inference",
            },
            "cohort": {
                "replicas": 12,
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
                "completion_rendezvous_ready_spread_seconds_max": (
                    runtime_gate.COMPLETION_RENDEZVOUS_READY_SPREAD_SECONDS_MAX
                ),
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
                "all_pre_and_post_event_windows_inside_all_row_overlap_required": True,
                "unique_rendezvous_worker_claims_required": True,
                "remote_preregistration_hash_required": True,
                "report_upload_create_only_and_remote_hash_verified": True,
                "new_cohort_attempt_rejects_preexisting_start_or_completion_rendezvous_state": True,
                "existing_child_reattachment_may_reuse_its_immutable_rendezvous_state": True,
            },
        },
        "diagnostic_only": {
            "decay_alignment": {
                "primary_support_id": "m100a",
                "primary_row_count": 8,
                "estimand": "mean log post/pre median-throughput ratio at assigned optimizer-decay onset",
                "window_steps": runtime_gate.C12_ASSIGNMENT_WINDOW_STEPS,
                "assignment_test": "exact enumeration of all 2,520 balanced onset assignments",
                "primary_alternative": "lower",
                "alpha": runtime_gate.C12_ASSIGNMENT_ALPHA,
                "minimum_attainable_p_value": 1 / 2_520,
                "non_rejection_interpretation": (
                    "Failure to reject is not evidence of no timing effect; the eight-row design has coarse power."
                ),
                "pretrend_falsification": "two-sided exact assignment test one window before onset",
                "lead_placebo_falsification": {
                    "offset_steps": runtime_gate.C12_LEAD_PLACEBO_OFFSET_STEPS,
                    "direction": "lead; evaluate at assigned onset minus offset_steps",
                    "test": "two-sided exact assignment test",
                },
                "classification_rule": (
                    "decay_aligned only when primary p<=0.01 and both falsification p-values exceed 0.01; "
                    "falsification_failed takes precedence; otherwise no_detectable_decay_alignment"
                ),
                "host_time_control": (
                    "Every permutation preserves two rows at each candidate step, so a shared step-indexed disturbance "
                    "is present in every balanced assignment. Random assignment, not host-clock alignment, justifies "
                    "inference for host-time disturbances."
                ),
                "window_separation": (
                    "Candidate onsets are 384 steps apart, so another assigned onset's 64-step response window cannot "
                    "overlap the primary, pretrend, or 256-step lead-placebo windows."
                ),
            },
            "decision_rule": (
                "The randomized-onset classification is an infrastructure diagnostic. It never overrides the "
                "operational release gate and is not evidence about endpoint loss or gradient conflict."
            ),
            "failure_policy": (
                "Any diagnostic exception is recorded as classification=unavailable; the operational report is still "
                "written and its verdict is unchanged."
            ),
            "generation_policy": (
                "Do not repeat C12 after inspecting its classification. Only independently diagnosed infrastructure "
                "failure before a complete report may motivate a new frozen generation; generation 14 supersedes the "
                "unlaunched blocked generations 12 and 13."
            ),
        },
        "predecessor": {
            "generation": 11,
            "stage": 6,
            "runtime_report_sha256": "44e930e801681d1b809c3b6dd91124f317ba612d9d949fed44ef618aa635e6f1",
            "runtime_report_remote_generation": "1786596853124878",
            "runtime_report_status": "pass",
            "integrity_rule": "Only this exact passing C6 runtime-report object authorizes C12 generation 14.",
        },
        "predecessor_evidence_archive": {
            "preregistration_sha256": "1960c88aceaf218ffbc5b73f2a6cfa8134a413d441954aecb46bb811f0cfad7d",
            "reproducibility_bundle_sha256": _sha256(PREDECESSOR_BUNDLE_PATH),
            "reproducibility_manifest_sha256": _sha256(PREDECESSOR_MANIFEST_PATH),
            "role": "audit provenance; these hashes are not additional launch authorization checks",
        },
        "next_stage": {
            "on_pass": (
                "C24 only after this exact report passes and its SHA-256 and remote object generation are pinned; "
                "C24 requires a separately reviewed fresh-generation preregistration."
            ),
            "on_fail": "Do not launch C24; diagnose without changing this generation's thresholds.",
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"path": str(OUTPUT_PATH), "sha256": _sha256(OUTPUT_PATH)}))


if __name__ == "__main__":
    main()
