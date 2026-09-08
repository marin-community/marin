# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Freeze the forced-retry C6 gate and its direct C64 successor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import fsspec

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811 as runtime_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_PATHS = {
    "forced-retry": (
        REPO_ROOT / "experiments/domain_phase_mix/"
        "starcoder_wsd80_gradient_conflict_forced_retry_preregistration_generation15_revision2_20260813.json"
    ),
    "c64": (
        REPO_ROOT / "experiments/domain_phase_mix/"
        "starcoder_wsd80_gradient_conflict_c64_preregistration_generation16_20260813.json"
    ),
    "c64-ungrouped": (
        REPO_ROOT / "experiments/domain_phase_mix/"
        "starcoder_wsd80_gradient_conflict_c64_preregistration_generation17_20260813.json"
    ),
    "c64-independent-fail-closed": (
        REPO_ROOT / "experiments/domain_phase_mix/"
        "starcoder_wsd80_gradient_conflict_c64_preregistration_generation18_20260813.json"
    ),
}
RELEASE_GATE_TEMPLATE_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/"
    "starcoder_wsd80_gradient_conflict_stress_gate_preregistration_generation14_20260812.json"
)
TEST_PATH = REPO_ROOT / "tests/test_starcoder_wsd80_gradient_conflict_canary.py"
PRIOR_C6_REPORT_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_gradient_conflict_stress_retry11_results_20260812/stage-c06/runtime_gate.json"
)
PRIOR_C6_REPORT_SHA256 = "44e930e801681d1b809c3b6dd91124f317ba612d9d949fed44ef618aa635e6f1"
PRIOR_C6_REMOTE_GENERATION = "1786596853124878"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_generation(url: str) -> str | None:
    fs, path = fsspec.core.url_to_fs(url)
    info = fs.info(path)
    return None if info.get("generation") is None else str(info["generation"])


def _implementation_sha256() -> dict[str, str]:
    return {
        "stress_launcher": _sha256(Path(stress.__file__).resolve()),
        "runtime_gate": _sha256(Path(runtime_gate.__file__).resolve()),
        "tests": _sha256(TEST_PATH),
        "gradient_conflict_full": _sha256(Path(stress.full.__file__).resolve()),
        "wsd80_surface": _sha256(Path(stress.base.__file__).resolve()),
    }


def _release_gate() -> dict[str, Any]:
    template = json.loads(RELEASE_GATE_TEMPLATE_PATH.read_text())
    return copy.deepcopy(template["release_gate"])


def _design(
    stage: int,
    *,
    placement_mode: stress.CohortPlacementMode = stress.CohortPlacementMode.TOPOLOGY_COSCHEDULED,
    start_barrier_timeout_seconds: float = stress.RENDEZVOUS_TIMEOUT_SECONDS,
    completion_barrier_timeout_seconds: float = stress.RENDEZVOUS_TIMEOUT_SECONDS,
    include_runtime_placement_contract: bool = False,
    parent_managed_whole_cohort_retries: int = stress.COHORT_MAX_PREEMPTION_RETRIES,
) -> dict[str, Any]:
    if stress.full.REQUIRED_STRESS_CONCURRENCIES != (6, 64):
        raise ValueError("The direct operational release path must remain exactly C6 -> C64")
    rows = stress.rows_for_stage(stage)
    design = {
        "stage": stage,
        "total_steps": stress.TOTAL_STEPS,
        "terminal_step": stress.TERMINAL_STEP,
        "data_switch_step": stress.DATA_SWITCH_STEP,
        "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
        "support_ids": [row.support_id for row in rows],
        "training_seeds": [row.training_seed for row in rows],
        "cohort": {
            "replicas": stage,
            "coscheduling_group_by": (
                stress.COHORT_COSCHEDULING_GROUP
                if placement_mode is stress.CohortPlacementMode.TOPOLOGY_COSCHEDULED
                else None
            ),
            "parent_managed_whole_cohort_retries": parent_managed_whole_cohort_retries,
            "per_attempt_wait_timeout_seconds": stress.COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS,
            "max_retries_failure": 0,
            "max_retries_preemption": 0,
            "max_task_failures": 0,
            "initialize_jax_distributed": False,
            "start_and_completion_barriers_required": True,
            "barriers_scoped_by_parent_and_iris_attempt": True,
            "fresh_checkpoint_and_wandb_namespaces_per_parent_attempt": True,
        },
        "parent_submission": {
            "preemptible": False,
            "max_retries_failure": 0,
            "max_retries_preemption": 0,
            "runtime_inherited_central1_placement_asserted": True,
            "runtime_preemptible_true_rejected_if_present": True,
            "runtime_retry_attestation_asserted": True,
            "controller_readback_required_before_fault_injection": {
                "preemptible": False,
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
            },
            "required_cli_flags": ["--no-preemptible", "--max-retries 0", "--max-preemption-retries 0"],
            "required_environment": {
                "MARIN_STRESS_PARENT_PREEMPTIBLE": "false",
                "MARIN_STRESS_PARENT_MAX_RETRIES_FAILURE": "0",
                "MARIN_STRESS_PARENT_MAX_RETRIES_PREEMPTION": "0",
            },
        },
        "future_training_release": {
            "required_operational_concurrencies": list(stress.full.REQUIRED_STRESS_CONCURRENCIES),
            "expected_exact_value": [6, 64],
        },
    }
    if include_runtime_placement_contract:
        design["cohort"].update(
            {
                "placement_mode": placement_mode,
                "coscheduling_enabled": placement_mode is stress.CohortPlacementMode.TOPOLOGY_COSCHEDULED,
                "start_barrier_timeout_seconds": start_barrier_timeout_seconds,
                "completion_barrier_timeout_seconds": completion_barrier_timeout_seconds,
            }
        )
    return design


def _forced_retry_preregistration() -> dict[str, Any]:
    prior_report = json.loads(PRIOR_C6_REPORT_PATH.read_text())
    if _sha256(PRIOR_C6_REPORT_PATH) != PRIOR_C6_REPORT_SHA256:
        raise ValueError("The prior clean C6 runtime report drifted")
    if prior_report.get("status") != "pass" or prior_report.get("stage") != 6:
        raise ValueError("The prior clean C6 runtime report is not a passing C6 report")
    target_child = stress.cohort_child_name(6, 15, 0)
    return {
        "preregistration_version": "2026-08-13-c06-forced-retry-gate-v2",
        "generation": 15,
        "supersedes_failed_preregistration_sha256": "d5d9f600bca0ba8353ed32a50a47ff4346235832db3bd90088229375b870b91e",
        "supersession_reason": (
            "The v1 parent failed before child allocation because Iris intentionally propagates only region "
            "and zone, not per-job preemptibility, through IRIS_JOB_CONSTRAINTS. V2 asserts the observable "
            "placement contract and requires controller job_config readback before fault injection."
        ),
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "implementation_sha256": _implementation_sha256(),
        "design": _design(6),
        "fault_injection": {
            "forced_preemption_recovery_required": True,
            "mode": "operator_iris_kick",
            "target_parent_cohort_attempt": 0,
            "target_child_job_basename": target_child,
            "target_task_index": 0,
            "requested_state": "preempted",
            "reason": "preregistered generation-15 whole-cohort recovery fault",
            "trigger": (
                "Inject exactly once after the attempt-000 start rendezvous release exists and logical task 0 "
                "is active, before the completion rendezvous exists. Abort rather than retarget if attempt 000 "
                "preempts or completes before this trigger."
            ),
            "expected_recovery": (
                "Attempt 000 becomes a terminal preemption-only child; the non-preemptible parent submits "
                "attempt 001 for all six rows; the analyzer selects the latest complete attempt and requires "
                "at least one abandoned cohort."
            ),
            "application_failure_count_allowed": 0,
        },
        "release_gate": _release_gate(),
        "prior_clean_c6_evidence": {
            "generation": 11,
            "runtime_report_sha256": PRIOR_C6_REPORT_SHA256,
            "runtime_report_remote_generation": PRIOR_C6_REMOTE_GENERATION,
            "role": "establishes the no-fault path only; it does not authorize C64",
        },
        "next_stage": {
            "on_pass": (
                "Freeze and independently review one direct C64 generation-16 gate citing this exact report "
                "hash and remote object generation."
            ),
            "on_fail": "Do not launch C64; diagnose the replacement failure without changing this gate.",
        },
    }


def _c64_preregistration(c6_report_path: Path) -> dict[str, Any]:
    c6_report = json.loads(c6_report_path.read_text())
    c6_sha256 = _sha256(c6_report_path)
    if c6_report.get("status") != "pass" or c6_report.get("stage") != 6 or c6_report.get("generation") != 15:
        raise ValueError("The direct C64 predecessor must be a passing generation-15 C6 report")
    recovery = c6_report.get("preemption_recovery", {})
    if not recovery.get("forced_preemption_recovery_required") or recovery.get("abandoned_cohort_attempts", 0) < 1:
        raise ValueError("The generation-15 report did not certify forced whole-cohort recovery")
    remote_url = stress.report_path(6, 15)
    if stress.full._remote_sha256(remote_url) != c6_sha256:
        raise ValueError("The remote generation-15 C6 report does not match the local report")
    return {
        "preregistration_version": "2026-08-13-c64-direct-target-gate-v1",
        "generation": 16,
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "implementation_sha256": _implementation_sha256(),
        "design": _design(64),
        "fault_injection": {"forced_preemption_recovery_required": False},
        "release_gate": _release_gate(),
        "predecessor": {
            "stage": 6,
            "generation": 15,
            "runtime_report_sha256": c6_sha256,
            "runtime_report_remote_generation": _object_generation(remote_url),
            "runtime_report_status": "pass",
            "integrity_rule": "Only this forced-recovery C6 report authorizes the direct C64 gate.",
        },
        "skipped_intermediate_stages": {
            "stages": [12, 24, 48],
            "rationale": (
                "The direct path minimizes wall-clock latency: C6 certifies replacement semantics and C64 tests "
                "the exact full-panel target concurrency. Skipping intermediate load rungs deliberately accepts "
                "that C64 may fail closed without localizing the first unsafe concurrency."
            ),
        },
        "next_stage": {
            "on_pass": "Freeze the 256-trajectory, maximum-concurrency-64 training-only release.",
            "on_fail": "Do not launch the full panel; diagnose without adding intermediate canaries by default.",
        },
    }


def _ungrouped_c64_preregistration(c6_report_path: Path) -> dict[str, Any]:
    preregistration = _c64_preregistration(c6_report_path)
    preregistration["preregistration_version"] = "2026-08-13-c64-independent-placement-gate-v2"
    preregistration["generation"] = 17
    preregistration["supersedes_generation"] = 16
    preregistration["supersession_reason"] = (
        "Iris topology coscheduling represents one physical multi-VM slice. The generation-16 C64 job instead "
        "requested 64 one-VM v5p-8 slices, so autoscaler routing rejected its demand as coschedule_mismatch and "
        "reported current_demand=0. Generation 17 removes only Iris topology coscheduling; the immutable 64-row "
        "start/completion barriers remain the measured concurrency contract."
    )
    preregistration["implementation_sha256"] = _implementation_sha256()
    preregistration["design"] = _design(
        64,
        placement_mode=stress.CohortPlacementMode.INDEPENDENT,
        start_barrier_timeout_seconds=5_400.0,
        completion_barrier_timeout_seconds=stress.RENDEZVOUS_TIMEOUT_SECONDS,
        include_runtime_placement_contract=True,
    )
    preregistration["design"]["cohort"].update(
        {
            "barrier_timeout_retryable": False,
            "barrier_timeout_policy": (
                "Fail closed rather than guessing whether a partial-admission timeout was infrastructure or "
                "application failure; no task-level retry may reuse a barrier attempt."
            ),
            "incremental_admission_expected": True,
        }
    )
    integrity = preregistration["release_gate"]["concurrency_and_integrity"]
    integrity.update(
        {
            "cohort_coscheduling_required": False,
            "cohort_ungrouped_placement_expected": True,
            "iris_topology_coscheduling_forbidden": True,
            "distinct_physical_worker_per_row_required": True,
            "required_realized_worker_region": "us-central1",
            "required_realized_worker_id_regex": (
                r"^marin-tpu-v5p-preemptible-8-us-central1-[0-9]{8}-[0-9]{4}-[0-9a-f]+-worker-0$"
            ),
            "rendezvous_ready_spread_seconds_max": 5_400.0,
            "incremental_admission_wait_reporting_required": True,
        }
    )
    preregistration["generation16_failure_evidence"] = {
        "parent_job": "/calvinxu/dm-starcoder-wsd80-gradient-conflict-direct-c64-g16-20260813",
        "child_job": (
            "/calvinxu/dm-starcoder-wsd80-gradient-conflict-direct-c64-g16-20260813/" "stage-c64-cohort-g16-attempt-000"
        ),
        "allocation_count": 0,
        "output_count": 0,
        "endpoint_metrics_read": False,
        "scheduler_snapshot_sha256": "65148647a71ceb785dc4bfa700a6478ecfcbeccf63959203b216460676aa1d93",
        "autoscaler_snapshot_sha256": "76abdce9592fb0badb6a7cb3299ea23f748f80b2f6c9a9091678a06959ac3630",
        "autoscaler_current_demand": 0,
        "v5p8_ready_slices": 9,
        "v5p8_historical_peak_demand": 506,
        "routing_failure": (
            "coschedule_mismatch: job needs 64 tasks coscheduled but no matching group has num_vms=64 "
            "(tpu_v5p-preemptible_8-us-central1-a=1)"
        ),
    }
    preregistration["predecessor"]["mechanism_delta"] = (
        "Generation-15 C6 certifies whole-cohort preemption replacement. It does not certify independent "
        "incremental placement; generation 17 certifies that exact C64 path directly."
    )
    preregistration["skipped_intermediate_stages"]["rationale"] = (
        "The direct target gate avoids another low-information canary. Generation 17 must itself prove 64 distinct "
        "workers, complete barrier admission, near-full measured overlap, and bounded runtime start skew; it fails "
        "closed if independent admission cannot satisfy those exact target-concurrency conditions."
    )
    return preregistration


def _fail_closed_ungrouped_c64_preregistration(c6_report_path: Path) -> dict[str, Any]:
    preregistration = _ungrouped_c64_preregistration(c6_report_path)
    preregistration["preregistration_version"] = "2026-08-13-c64-independent-placement-gate-v3"
    preregistration["generation"] = 18
    preregistration["supersedes_generation"] = 17
    preregistration["supersession_reason"] = (
        "Generation 17 was blocked before submission by independent Opus review: Iris does not expose a clean "
        "preemption-only whole-cohort failure under independent placement, and an assigned-phase eviction may retry "
        "one task into a different immutable barrier namespace. Generation 18 removes the unreachable automatic "
        "replacement claim and treats every nonzero parent or Iris attempt as a fail-closed result requiring manual "
        "review and a fresh generation."
    )
    preregistration["implementation_sha256"] = _implementation_sha256()
    preregistration["design"] = _design(
        64,
        placement_mode=stress.CohortPlacementMode.INDEPENDENT,
        start_barrier_timeout_seconds=5_400.0,
        completion_barrier_timeout_seconds=stress.RENDEZVOUS_TIMEOUT_SECONDS,
        include_runtime_placement_contract=True,
        parent_managed_whole_cohort_retries=0,
    )
    preregistration["design"]["cohort"].update(
        {
            "barrier_timeout_retryable": False,
            "barrier_timeout_policy": (
                "Fail closed rather than guessing whether a partial-admission timeout was infrastructure or "
                "application failure; no task-level retry may reuse a barrier attempt."
            ),
            "incremental_admission_expected": True,
            "automatic_whole_cohort_replacement_enabled": False,
            "infrastructure_preemption_policy": "fail_closed_manual_fresh_generation_relaunch",
            "assigned_phase_retry_behavior": (
                "Iris may retry an assigned-phase eviction despite max_retries_preemption=0. Immutable "
                "attempt-scoped barriers prevent mixed attempts from passing; any nonzero Iris attempt invalidates "
                "this gate."
            ),
        }
    )
    integrity = preregistration["release_gate"]["concurrency_and_integrity"]
    integrity.update(
        {
            "cohort_coscheduling_required": False,
            "cohort_ungrouped_placement_expected": True,
            "iris_topology_coscheduling_forbidden": True,
            "distinct_physical_worker_per_row_required": True,
            "required_realized_worker_region": "us-central1",
            "required_realized_worker_id_regex": (
                r"^marin-tpu-v5p-preemptible-8-us-central1-[0-9]{8}-[0-9]{4}-[0-9a-f]+-worker-0$"
            ),
            "rendezvous_ready_spread_seconds_max": 5_400.0,
            "incremental_admission_wait_reporting_required": True,
            "historical_infrastructure_preemptions_allowed": False,
            "parent_managed_whole_cohort_replacement_required": False,
            "parent_managed_whole_cohort_replacement_forbidden": True,
            "zero_parent_and_iris_attempt_required": True,
        }
    )
    preregistration["predecessor"]["mechanism_delta"] = (
        "Generation-15 C6 certifies grouped whole-cohort preemption replacement only. Generation 18 does not reuse "
        "that mechanism: it certifies the independent C64 success path and fails closed on any infrastructure event."
    )
    preregistration["generation17_review_evidence"] = {
        "job_submitted": False,
        "endpoint_metrics_read": False,
        "review_model": "claude-opus-5",
        "review_session_id": "922d4868-139c-4cf5-8b96-bf48b1bcefed",
        "verdict": "BLOCKED",
        "blocking_issue": (
            "Independent placement cannot produce the preemption-only child status required by generation 17's "
            "parent-managed replacement contract; assigned-phase task retry also uses a different barrier namespace."
        ),
    }
    preregistration["skipped_intermediate_stages"]["rationale"] = (
        "The direct target gate avoids another low-information concurrency rung. Generation 18 must itself prove 64 "
        "distinct workers, complete barrier admission, near-full measured overlap, and bounded runtime start skew. "
        "Any preemption, task retry, partial admission, or application failure fails closed and requires a separately "
        "reviewed fresh generation."
    )
    preregistration["next_stage"] = {
        "on_pass": "Freeze the 256-trajectory, maximum-concurrency-64 training-only release.",
        "on_fail": (
            "Do not launch the full panel. Classify the failure; infrastructure-only failure may be relaunched only "
            "under a fresh generation after review, without adding intermediate concurrency rungs by default."
        ),
    }
    return preregistration


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(OUTPUT_PATHS))
    parser.add_argument("--c6-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "forced-retry":
        if args.c6_report is not None:
            raise ValueError("The forced-retry freeze does not accept --c6-report")
        output = _forced_retry_preregistration()
    elif args.mode == "c64":
        if args.c6_report is None:
            raise ValueError("The C64 freeze requires --c6-report")
        output = _c64_preregistration(args.c6_report)
    elif args.mode == "c64-ungrouped":
        if args.c6_report is None:
            raise ValueError("The ungrouped C64 freeze requires --c6-report")
        output = _ungrouped_c64_preregistration(args.c6_report)
    else:
        if args.c6_report is None:
            raise ValueError("The fail-closed independent C64 freeze requires --c6-report")
        output = _fail_closed_ungrouped_c64_preregistration(args.c6_report)
    output_path = OUTPUT_PATHS[args.mode]
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"path": str(output_path), "sha256": _sha256(output_path)}))


if __name__ == "__main__":
    main()
