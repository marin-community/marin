# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Build the review-v9 full-training release from immutable gate evidence.

The first invocation writes a fail-closed candidate for independent review. A
second invocation must cite that exact candidate and the review session before
the launcher-valid release can be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress

REPO_ROOT = Path(__file__).resolve().parents[4]
DECOUPLED_SWITCH_REPORT_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_gradient_conflict_forced_retry_results_20260813/stage-c06/runtime_gate.json"
)
GEN19_RECOVERY_REPORT_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_gradient_conflict_resume_canary_revision2_20260813/runtime_gate.json"
)
GEN24_PREREGISTRATION_PATH = REPO_ROOT / (
    "experiments/domain_phase_mix/"
    "starcoder_wsd80_gradient_conflict_production_recovery_gate_generation24_20260813.json"
)
LONG_GATE_REPORT_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_gradient_conflict_long_gate_results_20260811/runtime_gate.json"
)
FINAL_RELEASE_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_gradient_conflict_training_release_v9_20260811.json"
)
EXPECTED_DECOUPLED_SWITCH_REPORT_SHA256 = "df70c0d3ace52ca00372e59158948a61de8e7a17a60dc69b31decfbca10282c7"
EXPECTED_DECOUPLED_SWITCH_PREREGISTRATION_SHA256 = "01f933cf5c5fd6e21fbb0701586c5d9ee0b60e8ae9d1e574dd6ea74fabba13ec"
EXPECTED_GEN19_REPORT_SHA256 = full.GEN19_RECOVERY_REPORT_SHA256
EXPECTED_GEN19_PREREGISTRATION_SHA256 = full.GEN19_RECOVERY_PREREGISTRATION_SHA256
EXPECTED_GEN19_ANALYZER_REVISION_SHA256 = full.GEN19_RECOVERY_ANALYZER_REVISION_SHA256
EXPECTED_GEN24_REPORT_SHA256 = full.GEN24_RECOVERY_REPORT_SHA256
EXPECTED_GEN24_PREREGISTRATION_SHA256 = full.GEN24_RECOVERY_PREREGISTRATION_SHA256
EXPECTED_LONG_GATE_REPORT_SHA256 = "37561085c14934dcb8ca54ccf178b2b9720cfbdf4dbee0ea6a453ef139335395"
PENDING_REVIEW_VERDICT = "PENDING_INDEPENDENT_REVIEW"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_runtime_report(
    path: Path,
    *,
    stage: int,
    generation: int | None,
    preregistration_sha256: str | None,
    previous_stage_report_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    report = _load_json(path)
    observed = {
        "status": report.get("status"),
        "stage": report.get("stage"),
        "generation": report.get("generation"),
        "endpoint_metrics_read": report.get("endpoint_metrics_read"),
        "scientific_inference_allowed": report.get("scientific_inference_allowed"),
        "failures": report.get("failures"),
        "preregistration_sha256": report.get("preregistration_sha256"),
        "previous_stage_report_sha256": report.get("previous_stage_report_sha256"),
    }
    expected = {
        "status": "pass",
        "stage": stage,
        "generation": generation,
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "failures": [],
        "preregistration_sha256": preregistration_sha256,
        "previous_stage_report_sha256": previous_stage_report_sha256,
    }
    if observed != expected:
        raise ValueError(f"Runtime report failed the frozen release contract: {path}: {observed} != {expected}")
    return report, _sha256(path)


def _validate_decoupled_switch_report() -> tuple[dict[str, Any], str]:
    report, report_sha256 = _validate_runtime_report(
        DECOUPLED_SWITCH_REPORT_PATH,
        stage=6,
        generation=15,
        preregistration_sha256=EXPECTED_DECOUPLED_SWITCH_PREREGISTRATION_SHA256,
        previous_stage_report_sha256=None,
    )
    if report_sha256 != EXPECTED_DECOUPLED_SWITCH_REPORT_SHA256:
        raise ValueError(
            "Generation-15 decoupled-switch report drifted: "
            f"{report_sha256} != {EXPECTED_DECOUPLED_SWITCH_REPORT_SHA256}"
        )
    recovery = report.get("preemption_recovery")
    if not isinstance(recovery, dict) or recovery.get("forced_preemption_recovery_required") is not True:
        raise ValueError("Generation-15 C6 report lacks forced-preemption recovery evidence")
    if recovery.get("parent_managed_whole_cohort_replacement") is not True:
        raise ValueError("Generation-15 C6 report lacks whole-cohort replacement evidence")
    if recovery.get("abandoned_cohort_attempts", 0) < 1 or recovery.get("cohort_attempt", 0) < 1:
        raise ValueError("Generation-15 C6 report did not replace an abandoned cohort")
    return report, report_sha256


def _validate_completed_iris(report: dict[str, Any], *, expected_tasks: int) -> None:
    for field, total_tasks in (("parent_iris", 1), ("child_iris", expected_tasks)):
        iris = report.get(field)
        expected = {
            "state": "succeeded",
            "exit": "0",
            "total_tasks": total_tasks,
            "completed_tasks": total_tasks,
            "running_tasks": 0,
            "failures": 0,
        }
        if not isinstance(iris, dict) or {key: iris.get(key) for key in expected} != expected:
            raise ValueError(f"{field} does not prove {total_tasks} completed tasks")


def _validate_recovery_wandb(report: dict[str, Any], *, expected_runs: int) -> None:
    wandb = report.get("wandb")
    if not isinstance(wandb, dict) or len(wandb) != expected_runs:
        raise ValueError(f"Recovery report must contain {expected_runs} W&B histories")
    for row_id, history in wandb.items():
        expected = {"status": "pass", "terminal_global_step": 3327, "history_rows": 3326}
        if not isinstance(history, dict) or {key: history.get(key) for key in expected} != expected:
            raise ValueError(f"{row_id}: W&B operational history is incomplete")


def _attempt_maximum_sum(report: dict[str, Any], *, expected_tasks: int) -> int:
    attempts = report.get("attempts")
    expected_keys = {str(index) for index in range(expected_tasks)}
    if not isinstance(attempts, dict) or set(attempts) != expected_keys:
        raise ValueError("Recovery report attempt inventory does not cover every task")
    maxima = []
    for task_index, rows in attempts.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Task {task_index} has no attempt receipts")
        observed_attempts: list[int] = []
        for row in rows:
            attempt = row.get("attempt") if isinstance(row, dict) else None
            if not isinstance(attempt, int):
                raise ValueError(f"Task {task_index} has malformed attempt receipts")
            observed_attempts.append(attempt)
        if observed_attempts[0] != 0 or len(set(observed_attempts)) != len(observed_attempts):
            raise ValueError(f"Task {task_index} has invalid attempt numbering")
        maxima.append(max(observed_attempts))
    return sum(maxima)


def _validate_forced_retries(
    report: dict[str, Any],
    *,
    expected_task_indices: set[int],
) -> None:
    forced_retries = report.get("forced_retries")
    if not isinstance(forced_retries, dict) or set(forced_retries) != {str(index) for index in expected_task_indices}:
        raise ValueError("Recovery report forced-retry rows drifted")
    state_evidence = report.get("state_evidence")
    if not isinstance(state_evidence, dict):
        raise ValueError("Recovery report lacks state evidence")
    for task_index in expected_task_indices:
        forced = forced_retries[str(task_index)]
        if not isinstance(forced, dict):
            raise ValueError(f"Task {task_index} forced-retry evidence is malformed")
        attempt = forced.get("attempt")
        checkpoint_step = forced.get("checkpoint_step")
        checkpoint_path = forced.get("checkpoint_path")
        if attempt != 1 or not isinstance(checkpoint_step, int) or checkpoint_step <= 0:
            raise ValueError(f"Task {task_index} did not restore its first retry from a nonzero checkpoint")
        if not isinstance(checkpoint_path, str) or "/checkpoints-temp/" not in checkpoint_path:
            raise ValueError(f"Task {task_index} did not restore the intended temporary checkpoint")
        row_state = state_evidence.get(str(task_index))
        attempt_state = row_state.get(str(attempt)) if isinstance(row_state, dict) else None
        if not isinstance(attempt_state, dict) or attempt_state.get("state_step") != checkpoint_step + 1:
            raise ValueError(f"Task {task_index} loaded state does not match its parent-authorized checkpoint")


def _validate_gen19_recovery_report() -> tuple[dict[str, Any], str]:
    report = _load_json(GEN19_RECOVERY_REPORT_PATH)
    expected = {
        "status": "pass",
        "stage": 6,
        "generation": full.GEN19_RECOVERY_GENERATION,
        "endpoint_metrics_read": False,
        "failures": [],
        "preregistration_sha256": EXPECTED_GEN19_PREREGISTRATION_SHA256,
        "revision_sha256": EXPECTED_GEN19_ANALYZER_REVISION_SHA256,
    }
    if {key: report.get(key) for key in expected} != expected:
        raise ValueError("Generation-19 recovery report failed its frozen contract")
    report_sha256 = _sha256(GEN19_RECOVERY_REPORT_PATH)
    if report_sha256 != EXPECTED_GEN19_REPORT_SHA256:
        raise ValueError(f"Generation-19 recovery report drifted: {report_sha256}")
    _validate_completed_iris(report, expected_tasks=6)
    _validate_recovery_wandb(report, expected_runs=6)
    _validate_forced_retries(report, expected_task_indices={0, 1, 2})
    if _attempt_maximum_sum(report, expected_tasks=6) != report["child_iris"]["preemptions"]:
        raise ValueError("Generation-19 attempt inventory does not reconcile with Iris preemptions")
    return report, report_sha256


def _validate_gen24_recovery_report(path: Path) -> tuple[dict[str, Any], str]:
    report = _load_json(path)
    expected = {
        "status": "pass",
        "stage": 64,
        "generation": full.GEN24_RECOVERY_GENERATION,
        "endpoint_metrics_read": False,
        "failures": [],
        "preregistration_sha256": EXPECTED_GEN24_PREREGISTRATION_SHA256,
    }
    if {key: report.get(key) for key in expected} != expected:
        raise ValueError("Generation-24 recovery report failed its frozen contract")
    report_sha256 = _sha256(path)
    if report_sha256 != EXPECTED_GEN24_REPORT_SHA256:
        raise ValueError(f"Generation-24 recovery report drifted: {report_sha256}")
    _validate_completed_iris(report, expected_tasks=64)
    _validate_recovery_wandb(report, expected_runs=64)
    _validate_forced_retries(report, expected_task_indices={0, 45, 62})
    if _attempt_maximum_sum(report, expected_tasks=64) != report["child_iris"]["preemptions"]:
        raise ValueError("Generation-24 attempt inventory does not reconcile with Iris preemptions")

    concurrency = report.get("concurrency")
    if (
        not isinstance(concurrency, dict)
        or concurrency.get("required_active_workers") != 64
        or concurrency.get("peak_active_workers") != 64
        or concurrency.get("full_load_overlap_seconds_min") != 120.0
        or not isinstance(concurrency.get("full_load_overlap_seconds"), (int, float))
        or concurrency["full_load_overlap_seconds"] < 120.0
    ):
        raise ValueError("Generation-24 report does not prove the preregistered 64-way overlap")

    admission = report.get("admission")
    releases = admission.get("releases") if isinstance(admission, dict) else None
    consumed = admission.get("consumed") if isinstance(admission, dict) else None
    if not isinstance(releases, dict) or len(releases) != 1 or not isinstance(consumed, dict) or len(consumed) != 64:
        raise ValueError("Generation-24 report does not prove one 64-row initial admission")
    release = next(iter(releases.values()))
    bindings = release.get("bindings") if isinstance(release, dict) else None
    if not isinstance(bindings, list) or len(bindings) != 64:
        raise ValueError("Generation-24 initial release does not bind 64 current attempts")
    bindings_by_task = {binding.get("task_index"): binding for binding in bindings if isinstance(binding, dict)}
    if set(bindings_by_task) != set(range(64)):
        raise ValueError("Generation-24 initial release task bindings drifted")
    for task_index in range(64):
        binding = bindings_by_task[task_index]
        bound_attempt = binding.get("attempt")
        row_consumed = consumed.get(str(task_index))
        initial = row_consumed.get(str(bound_attempt)) if isinstance(row_consumed, dict) else None
        if (
            not isinstance(bound_attempt, int)
            or not isinstance(initial, dict)
            or initial.get("admission_mode") != "bound_current_attempt"
            or initial.get("task_attempt_id") != binding.get("task_attempt_id")
        ):
            raise ValueError(f"Task {task_index} lacks initial bound-attempt admission evidence")

    preregistration_sha256 = _sha256(GEN24_PREREGISTRATION_PATH)
    if preregistration_sha256 != EXPECTED_GEN24_PREREGISTRATION_SHA256:
        raise ValueError(f"Generation-24 preregistration drifted: {preregistration_sha256}")
    preregistration = _load_json(GEN24_PREREGISTRATION_PATH)
    expected_prior_gate = {
        "pass_report_sha256": EXPECTED_GEN19_REPORT_SHA256,
        "preregistration_sha256": EXPECTED_GEN19_PREREGISTRATION_SHA256,
        "analyzer_revision_sha256": EXPECTED_GEN19_ANALYZER_REVISION_SHA256,
        "review_verdict": full.GEN19_RECOVERY_REVIEW_VERDICT,
        "review_session_id": full.GEN19_RECOVERY_REVIEW_SESSION_ID,
    }
    if (
        preregistration.get("prior_gate") != expected_prior_gate
        or preregistration.get("endpoint_metrics_read") is not False
    ):
        raise ValueError("Generation-24 preregistration no longer binds the reviewed generation-19 gate")
    return report, report_sha256


def _validate_long_gate_report() -> str:
    report = _load_json(LONG_GATE_REPORT_PATH)
    expected = {
        "status": "pass",
        "mode": "long-gate",
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "failures": [],
    }
    observed = {key: report.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Full-length switch gate failed the frozen release contract: {observed} != {expected}")
    if len(report.get("runs", [])) != 3 or len(report.get("iris_children", [])) != 3:
        raise ValueError("Full-length switch gate must contain exactly three completed trajectories")
    report_sha256 = _sha256(LONG_GATE_REPORT_PATH)
    if report_sha256 != EXPECTED_LONG_GATE_REPORT_SHA256:
        raise ValueError(f"Full-length switch report drifted: {report_sha256} != {EXPECTED_LONG_GATE_REPORT_SHA256}")
    return report_sha256


def _validate_output_inventory(path: Path) -> tuple[dict[str, Any], str]:
    inventory = _load_json(path)
    expected = {
        "design_version": full.EXPECTED_DESIGN_VERSION,
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
        "expected_root_count": full.EXPECTED_TRAJECTORY_COUNT,
        "empty_root_count": full.EXPECTED_TRAJECTORY_COUNT,
        "bookkeeping_root_count": 0,
        "resumable_root_count": 0,
        "completed_root_count": 0,
        "partial_root_count": 0,
        "unexpected_root_count": 0,
    }
    observed = {key: inventory.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Review-v9 output inventory is not pristine: {observed} != {expected}")
    report_sha256 = _sha256(path)
    if report_sha256 != full.OUTPUT_INVENTORY_REPORT_SHA256:
        raise ValueError(f"Review-v9 output inventory report drifted: {report_sha256}")
    return inventory, report_sha256


def _pending_review() -> dict[str, Any]:
    return {
        "verdict": PENDING_REVIEW_VERDICT,
        "reviewer": "claude-opus-5",
        "scope": "exact review-v9 full-training release",
    }


def _load_reviewed_candidate(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Reviewed candidate file hash drifted: {observed_sha256} != {expected_sha256}")
    return _load_json(path)


def _release_payload(
    *,
    c64_report_path: Path,
    output_inventory_path: Path,
    independent_review: dict[str, Any],
) -> dict[str, Any]:
    switch_report, switch_report_sha256 = _validate_decoupled_switch_report()
    _, gen19_report_sha256 = _validate_gen19_recovery_report()
    _, gen24_report_sha256 = _validate_gen24_recovery_report(c64_report_path)
    long_gate_report_sha256 = _validate_long_gate_report()
    inventory, inventory_report_sha256 = _validate_output_inventory(output_inventory_path)
    _, trajectories, _ = full.load_design()
    optimizer_decay_steps = sorted({int(row["optimizer_decay_step"]) for row in switch_report["runs"]})
    if not optimizer_decay_steps or stress.DATA_SWITCH_STEP in optimizer_decay_steps:
        raise ValueError("Generation-15 canary does not establish a decoupled data-switch and optimizer-decay schedule")

    release: dict[str, Any] = {
        "release_version": full.EXPECTED_RELEASE_VERSION,
        "release_sha256": "",
        "design_version": full.EXPECTED_DESIGN_VERSION,
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
        "design_manifest_sha256": full.EXPECTED_DESIGN_MANIFEST_SHA256,
        "training_fanout_allowed": True,
        "probe_fanout_allowed": False,
        "maximum_trajectory_count": full.EXPECTED_TRAJECTORY_COUNT,
        "maximum_concurrent_trajectories": full.MAX_RELEASE_CONCURRENCY,
        "allowed_trajectory_ids": [row.trajectory_id for row in trajectories],
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": "gs://marin-us-central1",
        "trajectory_count": full.EXPECTED_TRAJECTORY_COUNT,
        "checkpoint_count": full.EXPECTED_CHECKPOINT_COUNT,
        "runtime_source_sha256": full._runtime_source_sha256(),
        "train_holdout_seed": full.EXPECTED_TRAIN_HOLDOUT_SEED,
        "train_holdout_partition": full.EXPECTED_TRAIN_HOLDOUT_PARTITION,
        "support_partition_audit_sha256": full.EXPECTED_SUPPORT_PARTITION_AUDIT_SHA256,
        "validated_evidence": {
            "starcoder_flat_field_token_count": full.EXPECTED_STARCODER_SOURCE_TOKENS,
            "starcoder_packed_sequence_count": full.EXPECTED_STARCODER_SOURCE_SEQUENCES,
            "starcoder_trailing_token_count": full.EXPECTED_STARCODER_TRAILING_TOKENS,
            "finite_support_required_tokens": full.EXPECTED_FINITE_SUPPORT_REQUIRED_TOKENS,
            "runtime_config_count": full.EXPECTED_TRAJECTORY_COUNT,
            "support_audit_source_sha256": full.EXPECTED_SUPPORT_AUDIT_SOURCE_SHA256,
            "long_gate": {
                "status": "pass",
                "endpoint_metrics_read": False,
                "trajectory_count": 3,
            },
            "decoupled_switch_canary": {
                "status": "pass",
                "endpoint_metrics_read": False,
                "data_switch_step": stress.DATA_SWITCH_STEP,
                "optimizer_decay_step": optimizer_decay_steps[0],
                "optimizer_decay_steps": optimizer_decay_steps,
                "trajectory_count": 6,
            },
            "recovery_gates": [
                {
                    "maximum_concurrent": full.REQUIRED_RECOVERY_CONCURRENCIES[0],
                    "generation": full.GEN19_RECOVERY_GENERATION,
                    "status": "pass",
                    "report_sha256": gen19_report_sha256,
                    "preregistration_sha256": EXPECTED_GEN19_PREREGISTRATION_SHA256,
                    "analyzer_revision_sha256": EXPECTED_GEN19_ANALYZER_REVISION_SHA256,
                    "independent_review_verdict": full.GEN19_RECOVERY_REVIEW_VERDICT,
                    "independent_review_session_id": full.GEN19_RECOVERY_REVIEW_SESSION_ID,
                    "endpoint_metrics_read": False,
                },
                {
                    "maximum_concurrent": full.REQUIRED_RECOVERY_CONCURRENCIES[1],
                    "generation": full.GEN24_RECOVERY_GENERATION,
                    "status": "pass",
                    "report_sha256": gen24_report_sha256,
                    "preregistration_sha256": EXPECTED_GEN24_PREREGISTRATION_SHA256,
                    "prior_gate_report_sha256": gen19_report_sha256,
                    "independent_review_verdict": full.GEN24_RECOVERY_REVIEW_VERDICT,
                    "independent_review_session_id": full.GEN24_RECOVERY_REVIEW_SESSION_ID,
                    "endpoint_metrics_read": False,
                },
            ],
            "orchestration_scope": {
                "c64_gate_scope": "Levanter run-local checkpoint recovery under 64-way child preemption",
                "c64_gate_exercises_step_runner_fanout": False,
                "production_fanout": "StepRunner dispatches one independent Fray/Iris child per trajectory",
                "production_fanout_live_gate": False,
                "production_fanout_source_hash_pinned": True,
                "parent_failure_policy": "fail closed; resubmit the exact command against owned resumable roots",
                "child_application_failure_retries": 0,
                "child_preemption_retries": 100,
                "child_wall_timeout_seconds": None,
            },
            "output_inventory": {
                key: inventory[key]
                for key in (
                    "expected_root_count",
                    "empty_root_count",
                    "bookkeeping_root_count",
                    "resumable_root_count",
                    "completed_root_count",
                    "partial_root_count",
                    "unexpected_root_count",
                )
            },
            "long_gate_report_sha256": long_gate_report_sha256,
            "decoupled_switch_canary_report_sha256": switch_report_sha256,
            "checkpoint_recovery_report_sha256": gen19_report_sha256,
            "operational_threshold_report_sha256": gen24_report_sha256,
            "output_inventory_report_sha256": inventory_report_sha256,
        },
        "independent_review": independent_review,
    }
    release["release_sha256"] = full._canonical_sha256({**release, "release_sha256": ""})
    return release


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c64-report", type=Path, required=True)
    parser.add_argument("--output-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-candidate", type=Path)
    parser.add_argument("--reviewed-candidate-sha256")
    parser.add_argument("--review-session-id")
    parser.add_argument("--review-completed-at")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.reviewed_candidate is None:
        if (
            args.reviewed_candidate_sha256 is not None
            or args.review_session_id is not None
            or args.review_completed_at is not None
        ):
            raise ValueError("Review metadata requires --reviewed-candidate")
        review = _pending_review()
    else:
        if not args.reviewed_candidate_sha256 or not args.review_session_id or not args.review_completed_at:
            raise ValueError("Final release requires candidate hash, review session, and completion time")
        observed_candidate_sha256 = args.reviewed_candidate_sha256
        candidate = _load_reviewed_candidate(args.reviewed_candidate, observed_candidate_sha256)
        expected_candidate = _release_payload(
            c64_report_path=args.c64_report,
            output_inventory_path=args.output_inventory,
            independent_review=_pending_review(),
        )
        if candidate != expected_candidate:
            raise ValueError("Reviewed candidate no longer matches the exact gate evidence")
        review = {
            "verdict": "PASS_FULL_TRAINING",
            "reviewer": "claude-opus-5",
            "session_id": args.review_session_id,
            "completed_at": args.review_completed_at,
            "reviewed_candidate_sha256": observed_candidate_sha256,
            "scope": "exact review-v9 full-training release",
        }

    release = _release_payload(
        c64_report_path=args.c64_report,
        output_inventory_path=args.output_inventory,
        independent_review=review,
    )
    if review["verdict"] == "PASS_FULL_TRAINING":
        full._validate_training_release(release)
        if args.output.resolve() != FINAL_RELEASE_PATH.resolve():
            raise ValueError(f"Final release must be written to {FINAL_RELEASE_PATH}")
    elif args.output.resolve() == FINAL_RELEASE_PATH.resolve():
        raise ValueError("A pending-review candidate cannot overwrite the launch-authorizing release path")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "file_sha256": _sha256(args.output),
                "release_sha256": release["release_sha256"],
                "review_verdict": review["verdict"],
            }
        )
    )


if __name__ == "__main__":
    main()
