# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec[gcs]",
#   "wandb>=0.21",
# ]
# ///

"""Reanalyze the Gen19 resume canary with corrected Levanter step semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import fsspec
import wandb

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_resume_canary as resume
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_resume_canary_20260813 as original,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811 as runtime_gate,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
REVISION_PATH = REPO_ROOT / (
    "experiments/domain_phase_mix/" "starcoder_wsd80_gradient_conflict_resume_analyzer_revision2_20260813.json"
)
ORIGINAL_PREREGISTRATION_SHA256 = "1c3a125f2521116db8a10ebb87a7e52e3bfe6128539af55da1ee9308f45b418f"
ORIGINAL_FAILED_REPORT_SHA256 = "d4271cd8c904cd695269cdf3aba539eacafe46834930cd9a347ca985a14810ea"
FIRST_OPERATIONAL_WANDB_STEP = 2
REMOTE_REPORT_NAME = "runtime_gate_revision2.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _validated_revision(expected_sha256: str) -> dict[str, Any]:
    observed_sha256 = _sha256(REVISION_PATH)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Analyzer revision drifted: {observed_sha256} != {expected_sha256}")
    revision = json.loads(REVISION_PATH.read_text())
    if revision.get("original_preregistration_sha256") != ORIGINAL_PREREGISTRATION_SHA256:
        raise ValueError("Analyzer revision is not bound to the Gen19 preregistration")
    if revision.get("original_failed_report_sha256") != ORIGINAL_FAILED_REPORT_SHA256:
        raise ValueError("Analyzer revision is not bound to the original failed report")
    expected_corrections = {
        "checkpoint_metadata_step_to_restored_state_step": "N_to_N_plus_1",
        "operational_wandb_global_steps": [FIRST_OPERATIONAL_WANDB_STEP, stress.TERMINAL_STEP],
    }
    if revision.get("semantic_corrections") != expected_corrections:
        raise ValueError("Analyzer revision contains an unrecognized semantic correction")
    if revision.get("endpoint_metrics_read") is not False:
        raise ValueError("Analyzer revision is not endpoint-blind")
    for relative_path, expected_hash in revision.get("implementation_sha256", {}).items():
        observed_hash = _sha256(REPO_ROOT / relative_path)
        if observed_hash != expected_hash:
            raise ValueError(f"Revision dependency drifted: {relative_path}: {observed_hash} != {expected_hash}")
    return revision


def _state_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, dict[int, dict[str, Any]]], list[str]]:
    observed: dict[int, dict[int, dict[str, Any]]] = {index: {} for index in range(len(rows))}
    failures: list[str] = []
    for task_index, row in enumerate(rows):
        task_attempts = attempts[task_index]
        if not task_attempts:
            continue
        latest_attempt = max(int(attempt["attempt"]) for attempt in task_attempts)
        for attempt in task_attempts:
            attempt_index = int(attempt["attempt"])
            path = str(attempt["initial_state_evidence_path"])
            try:
                evidence = original._read_json(path)
            except FileNotFoundError:
                if attempt_index == latest_attempt:
                    failures.append(f"task {task_index} final attempt {attempt_index} lacks state evidence")
                continue
            observed[task_index][attempt_index] = evidence
            if evidence.get("run_id") != resume._run_id(row):
                failures.append(f"task {task_index} attempt {attempt_index} state evidence has a run identity mismatch")
            state_step = int(evidence.get("state_step", -1))
            checkpoint_step = int(attempt.get("checkpoint_step", 0))
            expected_state_step = checkpoint_step + 1 if checkpoint_step > 0 else 0
            if state_step != expected_state_step:
                failures.append(
                    f"task {task_index} attempt {attempt_index} initialized at step {state_step}, "
                    f"expected {expected_state_step} from checkpoint label step-{checkpoint_step}"
                )
    return observed, failures


def _forced_retry_evidence(
    attempts: dict[int, list[dict[str, Any]]],
    faults: dict[int, dict[str, Any]],
    state_evidence: dict[int, dict[int, dict[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    matched: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    for task_index, fault in faults.items():
        source_attempt = int(fault.get("source_attempt", -1))
        parent_step = int(fault.get("checkpoint_step", 0))
        candidates: list[dict[str, Any]] = []
        for evidence in attempts.get(task_index, []):
            attempt_index = int(evidence["attempt"])
            worker_step = int(evidence.get("checkpoint_step", 0))
            if attempt_index <= source_attempt or worker_step < parent_step:
                continue
            if worker_step == parent_step and original._checkpoint_signature(evidence) != original._checkpoint_signature(
                fault
            ):
                continue
            state = state_evidence.get(task_index, {}).get(attempt_index)
            if state is None or int(state.get("state_step", -1)) != worker_step + 1:
                continue
            candidates.append(evidence)
        if not candidates:
            failures.append(
                f"task {task_index} has no post-fault state restored from the parent-observed checkpoint or newer"
            )
            continue
        matched[task_index] = min(candidates, key=lambda evidence: int(evidence["attempt"]))
    return matched, failures


def _wandb_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
    forced_retries: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    api = wandb.Api(timeout=60)
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    fault_tasks = {fault.task_index for fault in resume.FAULT_INJECTIONS}
    expected_steps = list(range(FIRST_OPERATIONAL_WANDB_STEP, stress.TERMINAL_STEP + 1))
    for task_index, row in enumerate(rows):
        run_id = resume._run_id(row)
        try:
            api.flush()
            run = api.run(f"marin-community/marin/{run_id}")
            history = original._history(run)
        except (ValueError, wandb.errors.CommError) as error:
            failure = f"{row.trajectory_id}: W&B operational history unavailable: {error}"
            failures.append(failure)
            evidence[row.trajectory_id] = {
                "attempt_count": len(attempts[task_index]),
                "forced_retry_attempt": None,
                "history_rows": 0,
                "run_id": run_id,
                "status": "fail",
                "terminal_global_step": None,
            }
            continue
        global_steps = [int(item["global_step"]) for item in history]
        terminal_step = max(global_steps)
        row_failures: list[str] = []
        if terminal_step != stress.TERMINAL_STEP:
            row_failures.append(f"terminal global step {terminal_step} != {stress.TERMINAL_STEP}")
        forced_retry = forced_retries.get(task_index)
        if task_index in fault_tasks and forced_retry is None:
            row_failures.append("forced-fault run lacks a matched checkpoint-recovery claim")
        if global_steps != expected_steps:
            missing = sorted(set(expected_steps) - set(global_steps))
            duplicates = len(global_steps) - len(set(global_steps))
            row_failures.append(
                f"W&B operational history is not exactly step-complete over "
                f"{FIRST_OPERATIONAL_WANDB_STEP}..{stress.TERMINAL_STEP}: "
                f"missing={missing[:20]}, duplicate_count={duplicates}"
            )
        failures.extend(f"{row.trajectory_id}: {failure}" for failure in row_failures)
        evidence[row.trajectory_id] = {
            "attempt_count": len(attempts[task_index]),
            "forced_retry_attempt": None if forced_retry is None else int(forced_retry["attempt"]),
            "history_rows": len(history),
            "run_id": run_id,
            "status": "pass" if not row_failures else "fail",
            "terminal_global_step": terminal_step,
        }
    return evidence, failures


def analyze(*, parent_job: str, preregistration_sha256: str, revision_sha256: str) -> dict[str, Any]:
    """Return the endpoint-blind Gen19 report under the frozen semantic correction."""
    if preregistration_sha256 != ORIGINAL_PREREGISTRATION_SHA256:
        raise ValueError("Revision 2 only applies to the frozen Gen19 preregistration")
    revision = _validated_revision(revision_sha256)
    preregistration = resume.validate_preregistration(preregistration_sha256)
    rows = stress.rows_for_stage(resume.STAGE)
    parent = runtime_gate._iris_summary(parent_job)
    children = runtime_gate._iris_child_summaries(parent_job)
    failures = runtime_gate._iris_failures(parent)
    if len(children) != 1:
        failures.append(f"expected one resume-canary child, found {len(children)}")
        child = None
    else:
        child = children[0]
        failures.extend(runtime_gate._iris_failures(child, allow_preemptions=True))
        if child["preemptions"] < len(resume.FAULT_INJECTIONS):
            failures.append(
                f"child recorded {child['preemptions']} preemptions; expected at least {len(resume.FAULT_INJECTIONS)}"
            )

    attempts, attempt_failures = original._attempt_evidence(rows)
    faults, fault_failures = original._fault_evidence()
    state_evidence, state_failures = _state_evidence(rows, attempts)
    forced_retries, forced_retry_failures = _forced_retry_evidence(attempts, faults, state_evidence)
    wandb_evidence, wandb_failures = _wandb_evidence(rows, attempts, forced_retries)
    failures.extend(attempt_failures)
    failures.extend(fault_failures)
    failures.extend(state_failures)
    failures.extend(forced_retry_failures)
    failures.extend(wandb_failures)
    remote_preregistration = original._remote_preregistration(preregistration_sha256)
    return {
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "analyzer_revision": revision,
        "attempts": attempts,
        "child_iris": child,
        "data_continuity_argument": {
            "checkpoint_label_is_completed_step": True,
            "checkpoint_restores_next_trainer_step": True,
            "deterministic_loader_restarts_from_restored_step": True,
            "frozen_train_lm_sha256": preregistration["implementation_sha256"]["train_lm"],
            "frozen_loader_sha256": preregistration["implementation_sha256"]["loader"],
            "frozen_dataset_sha256": preregistration["implementation_sha256"]["datasets"],
            "phase_schedule_is_global_step_indexed": True,
        },
        "endpoint_metrics_read": False,
        "failures": failures,
        "faults": faults,
        "forced_retries": forced_retries,
        "generation": resume.GENERATION,
        "original_failed_report_sha256": ORIGINAL_FAILED_REPORT_SHA256,
        "parent_iris": parent,
        "preregistration_sha256": preregistration_sha256,
        "remote_preregistration": remote_preregistration,
        "revision_sha256": revision_sha256,
        "stage": resume.STAGE,
        "state_evidence": state_evidence,
        "status": "pass" if not failures else "fail",
        "wandb": wandb_evidence,
    }


def _write_report(report: dict[str, Any], output: Path, *, upload: bool) -> str:
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    report_sha256 = _sha256_bytes(payload)
    if upload:
        fs, path = fsspec.core.url_to_fs(f"{resume.EVIDENCE_ROOT}/{REMOTE_REPORT_NAME}")
        stress._write_json_once(fs, path, report)
    return report_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--revision-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    report = analyze(
        parent_job=args.parent_job,
        preregistration_sha256=args.preregistration_sha256,
        revision_sha256=args.revision_sha256,
    )
    report_sha256 = _write_report(report, args.output, upload=args.upload and report["status"] == "pass")
    print(json.dumps({"output": str(args.output), "report_sha256": report_sha256, "status": report["status"]}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
