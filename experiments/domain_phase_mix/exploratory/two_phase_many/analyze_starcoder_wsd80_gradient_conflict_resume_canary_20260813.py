# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec[gcs]",
#   "wandb>=0.21",
# ]
# ///

"""Certify the endpoint-blind production-style WSD80 resume canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fsspec
import wandb

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_resume_canary as resume
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811 as runtime_gate,
)

logger = logging.getLogger(__name__)

HISTORY_KEYS = ("_step", "global_step", "throughput/tokens_per_second", "throughput/loading_time")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(url: str) -> dict[str, Any]:
    with fsspec.open(url) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {url}")
    return value


def _remote_preregistration(expected_sha256: str) -> dict[str, Any]:
    url = f"{resume.EVIDENCE_ROOT}/preregistration.json"
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as handle:
        payload = handle.read()
    observed_sha256 = _sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Remote resume-canary preregistration drifted: {observed_sha256} != {expected_sha256}")
    return {
        "path": url,
        "sha256": observed_sha256,
        "generation": None if fs.info(path).get("generation") is None else str(fs.info(path)["generation"]),
    }


def _evidence_objects(kind: str) -> tuple[str, ...]:
    fs, root = fsspec.core.url_to_fs(resume.EVIDENCE_ROOT)
    return tuple(sorted(fs.unstrip_protocol(path) for path in fs.glob(f"{root}/{kind}/**/*.json")))


def _attempt_evidence(rows: tuple[Any, ...]) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    by_task: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(rows))}
    failures: list[str] = []
    for url in _evidence_objects("attempts"):
        evidence = _read_json(url)
        task_index = int(evidence["task_index"])
        if task_index not in by_task:
            failures.append(f"unexpected attempt-evidence task index {task_index}")
            continue
        by_task[task_index].append(evidence)

    fault_tasks = {fault.task_index for fault in resume.FAULT_INJECTIONS}
    for task_index, row in enumerate(rows):
        evidence = sorted(by_task[task_index], key=lambda item: int(item["attempt"]))
        by_task[task_index] = evidence
        attempts = [int(item["attempt"]) for item in evidence]
        if not attempts or attempts[0] != 0 or attempts != sorted(set(attempts)):
            failures.append(f"task {task_index} attempt inventory is not unique and zero-based: {attempts}")
            continue
        if any(item.get("row_id") != row.trajectory_id for item in evidence):
            failures.append(f"task {task_index} attempt evidence has a row identity mismatch")
        if task_index in fault_tasks and len(attempts) < 2:
            failures.append(f"forced-fault task {task_index} never recorded a retry attempt")
        for item in evidence:
            attempt = int(item["attempt"])
            if item.get("initial_state_evidence_path") != resume._initial_state_evidence_path(task_index, attempt):
                failures.append(f"task {task_index} attempt {attempt} has a drifted initial-state evidence path")
    return by_task, failures


def _fault_evidence() -> tuple[dict[int, dict[str, Any]], list[str]]:
    expected = {fault.task_index: fault for fault in resume.FAULT_INJECTIONS}
    observed: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    for url in _evidence_objects("faults"):
        evidence = _read_json(url)
        task_index = int(evidence["task_index"])
        if task_index in observed:
            failures.append(f"duplicate fault receipt for task {task_index}")
            continue
        observed[task_index] = evidence
    if set(observed) != set(expected):
        failures.append(f"fault receipt tasks {sorted(observed)} != expected {sorted(expected)}")
    for task_index, fault in expected.items():
        evidence = observed.get(task_index)
        if evidence is None:
            continue
        frozen = asdict(fault)
        if any(evidence.get(key) != value for key, value in frozen.items()):
            failures.append(f"task {task_index} fault receipt drifted from the frozen plan")
        if evidence.get("requested_state") != "preempted":
            failures.append(f"task {task_index} fault receipt did not request preemption")
        if int(evidence.get("observed_global_step", -1)) < fault.trigger_step:
            failures.append(f"task {task_index} was preempted before its frozen trigger")
        if (
            int(evidence.get("checkpoint_step", 0)) <= 0
            or not evidence.get("checkpoint_path")
            or not evidence.get("checkpoint_metadata_sha256")
        ):
            failures.append(f"task {task_index} fault receipt lacks an independently observed checkpoint")
        if int(evidence.get("source_attempt", -1)) < 0:
            failures.append(f"task {task_index} fault receipt lacks its source attempt")
    return observed, failures


def _checkpoint_signature(evidence: dict[str, Any]) -> tuple[str | None, int, str | None]:
    return (
        evidence.get("checkpoint_path"),
        int(evidence.get("checkpoint_step", 0)),
        evidence.get("checkpoint_metadata_sha256"),
    )


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
                evidence = _read_json(path)
            except FileNotFoundError:
                # A preemptible worker can be reclaimed after writing its attempt receipt but before
                # trainer initialization. Only the final, successful attempt must reach this marker;
                # forced-retry candidates are checked separately below.
                if attempt_index == latest_attempt:
                    failures.append(f"task {task_index} final attempt {attempt_index} lacks state evidence")
                continue
            observed[task_index][attempt_index] = evidence
            if evidence.get("run_id") != resume._run_id(row):
                failures.append(f"task {task_index} attempt {attempt_index} state evidence has a run identity mismatch")
            state_step = int(evidence.get("state_step", -1))
            checkpoint_step = int(attempt.get("checkpoint_step", 0))
            expected_state_step = checkpoint_step if checkpoint_step > 0 else 0
            if state_step != expected_state_step:
                failures.append(
                    f"task {task_index} attempt {attempt_index} initialized at step {state_step}, "
                    f"expected {expected_state_step} from its checkpoint claim"
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
            if worker_step == parent_step and _checkpoint_signature(evidence) != _checkpoint_signature(fault):
                continue
            state = state_evidence.get(task_index, {}).get(attempt_index)
            if state is None or int(state.get("state_step", -1)) != worker_step:
                continue
            candidates.append(evidence)
        if not candidates:
            failures.append(
                f"task {task_index} has no post-fault initialized state at or after the parent-observed checkpoint"
            )
            continue
        matched[task_index] = min(candidates, key=lambda evidence: int(evidence["attempt"]))
    return matched, failures


def _history(run: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_wandb_steps: dict[int, dict[str, Any]] = {}
    for row in run.scan_history(keys=list(HISTORY_KEYS), page_size=10_000):
        if row.get("_step") is None or row.get("global_step") is None:
            continue
        wandb_step = int(row["_step"])
        selected = {key: row.get(key) for key in HISTORY_KEYS}
        existing = seen_wandb_steps.get(wandb_step)
        if existing is not None:
            if existing != selected:
                raise ValueError(f"Conflicting duplicate W&B history step {wandb_step} in {run.id}")
            continue
        seen_wandb_steps[wandb_step] = selected
        rows.append(selected)
    rows.sort(key=lambda row: int(row["_step"]))
    if not rows:
        raise ValueError(f"No operational training history for {run.id}")
    return rows


def _wandb_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
    forced_retries: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    api = wandb.Api(timeout=60)
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    fault_tasks = {fault.task_index for fault in resume.FAULT_INJECTIONS}
    for task_index, row in enumerate(rows):
        run_id = resume._run_id(row)
        try:
            api.flush()
            run = api.run(f"marin-community/marin/{run_id}")
            history = _history(run)
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
        expected_steps = list(range(stress.TERMINAL_STEP + 1))
        if global_steps != expected_steps:
            missing = sorted(set(expected_steps) - set(global_steps))
            duplicates = len(global_steps) - len(set(global_steps))
            row_failures.append(
                f"W&B operational history is not exactly step-complete: missing={missing[:20]}, "
                f"duplicate_count={duplicates}"
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


def analyze(*, parent_job: str, preregistration_sha256: str) -> dict[str, Any]:
    """Return a complete endpoint-blind recovery report."""
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

    attempts, attempt_failures = _attempt_evidence(rows)
    faults, fault_failures = _fault_evidence()
    state_evidence, state_failures = _state_evidence(rows, attempts)
    forced_retries, forced_retry_failures = _forced_retry_evidence(attempts, faults, state_evidence)
    wandb_evidence, wandb_failures = _wandb_evidence(rows, attempts, forced_retries)
    failures.extend(attempt_failures)
    failures.extend(fault_failures)
    failures.extend(state_failures)
    failures.extend(forced_retry_failures)
    failures.extend(wandb_failures)
    remote_preregistration = _remote_preregistration(preregistration_sha256)
    return {
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "attempts": attempts,
        "child_iris": child,
        "data_continuity_argument": {
            "checkpoint_restores_trainer_step": True,
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
        "parent_iris": parent,
        "preregistration_sha256": preregistration_sha256,
        "remote_preregistration": remote_preregistration,
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
        fs, path = fsspec.core.url_to_fs(f"{resume.EVIDENCE_ROOT}/runtime_gate.json")
        stress._write_json_once(fs, path, report)
    return report_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    report = analyze(parent_job=args.parent_job, preregistration_sha256=args.preregistration_sha256)
    report_sha256 = _write_report(report, args.output, upload=args.upload and report["status"] == "pass")
    print(json.dumps({"output": str(args.output), "report_sha256": report_sha256, "status": report["status"]}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
