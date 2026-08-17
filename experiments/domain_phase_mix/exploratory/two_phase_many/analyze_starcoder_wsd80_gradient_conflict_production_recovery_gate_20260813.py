# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec[gcs]",
#   "wandb>=0.21",
# ]
# ///

"""Certify endpoint-blind checkpoint recovery under the production C64 load."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import wandb
from iris.cluster.types import TaskAttempt

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_production_recovery_gate as gate
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811 as runtime_gate,
)

logger = logging.getLogger(__name__)

HISTORY_KEYS = (
    "_step",
    "_timestamp",
    "global_step",
    "throughput/tokens_per_second",
    "throughput/loading_time",
)
FIRST_OPERATIONAL_WANDB_STEP = 2
ACTIVE_GAP_SECONDS_MAX = 30.0
FULL_LOAD_OVERLAP_SECONDS_MIN = 120.0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(url: str) -> dict[str, Any]:
    with fsspec.open(url) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {url}")
    return value


def _timestamp(value: Any, *, label: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not an ISO timestamp: {value!r}")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} lacks a timezone: {value!r}")
    return parsed.astimezone(UTC).timestamp()


def _remote_preregistration(expected_sha256: str) -> dict[str, Any]:
    url = f"{gate.EVIDENCE_ROOT}/preregistration.json"
    fs, path = fsspec.core.url_to_fs(url)
    with fs.open(path, "rb") as handle:
        payload = handle.read()
    observed_sha256 = _sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Remote production-recovery preregistration drifted: {observed_sha256} != {expected_sha256}")
    return {
        "path": url,
        "sha256": observed_sha256,
        "generation": None if fs.info(path).get("generation") is None else str(fs.info(path)["generation"]),
    }


def _evidence_objects(kind: str) -> tuple[str, ...]:
    fs, root = fsspec.core.url_to_fs(gate.EVIDENCE_ROOT)
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

    fault_tasks = {fault.task_index for fault in gate.FAULT_INJECTIONS}
    for task_index, row in enumerate(rows):
        evidence = sorted(by_task[task_index], key=lambda item: int(item["attempt"]))
        by_task[task_index] = evidence
        attempts = [int(item["attempt"]) for item in evidence]
        if not attempts or attempts[0] != 0 or attempts != sorted(set(attempts)):
            failures.append(f"task {task_index} attempt inventory is not unique and zero-based: {attempts}")
            continue
        if any(item.get("row_id") != row.trajectory_id for item in evidence):
            failures.append(f"task {task_index} attempt evidence has a row identity mismatch")
        if any(int(item.get("generation", -1)) != gate.GENERATION for item in evidence):
            failures.append(f"task {task_index} attempt evidence has a generation mismatch")
        if any(item.get("worker_region") != "us-central1" for item in evidence):
            failures.append(f"task {task_index} attempt evidence escaped us-central1")
        if task_index in fault_tasks and len(attempts) < 2:
            failures.append(f"forced-fault task {task_index} never recorded a retry attempt")
        for item in evidence:
            attempt = int(item["attempt"])
            try:
                _timestamp(item.get("ready_at"), label=f"task {task_index} attempt {attempt} ready_at")
            except ValueError as error:
                failures.append(str(error))
            if item.get("initial_state_evidence_path") != gate._initial_state_evidence_path(task_index, attempt):
                failures.append(f"task {task_index} attempt {attempt} has a drifted initial-state evidence path")
            try:
                task_attempt = TaskAttempt.from_wire(str(item.get("task_attempt_id")))
                _, observed_task_index = task_attempt.task_id.require_task()
                observed_attempt = task_attempt.require_attempt()
            except ValueError as error:
                failures.append(f"task {task_index} attempt {attempt} has invalid attempt identity: {error}")
                continue
            if observed_task_index != task_index or observed_attempt != attempt:
                failures.append(f"task {task_index} attempt {attempt} identity points to another task or attempt")
            if item.get("task_id") != task_attempt.task_id.to_wire():
                failures.append(f"task {task_index} attempt {attempt} task identity is internally inconsistent")
            search_paths = item.get("checkpoint_search_paths")
            if (
                not isinstance(search_paths, list)
                or len(search_paths) != 2
                or not all(isinstance(path, str) and path.startswith("gs://marin-us-central1/") for path in search_paths)
            ):
                failures.append(f"task {task_index} attempt {attempt} has invalid checkpoint search paths")
            checkpoint_step = int(item.get("checkpoint_step", 0))
            recovery_mode = item.get("recovery_mode")
            if attempt == 0:
                if (
                    recovery_mode != gate.RECOVERY_MODE_INITIAL
                    or checkpoint_step != 0
                    or item.get("checkpoint_path") is not None
                    or item.get("checkpoint_metadata_sha256") is not None
                ):
                    failures.append(f"task {task_index} initial attempt unexpectedly claimed a checkpoint")
            elif recovery_mode == gate.RECOVERY_MODE_CHECKPOINT:
                if checkpoint_step <= 0 or not item.get("checkpoint_path") or not item.get("checkpoint_metadata_sha256"):
                    failures.append(f"task {task_index} retry attempt {attempt} lacks a nonzero full-state checkpoint")
            elif recovery_mode == gate.RECOVERY_MODE_RESTART_FROM_ZERO:
                if (
                    checkpoint_step != 0
                    or item.get("checkpoint_path") is not None
                    or item.get("checkpoint_metadata_sha256") is not None
                ):
                    failures.append(f"task {task_index} scratch retry {attempt} claimed checkpoint state")
            else:
                failures.append(f"task {task_index} retry attempt {attempt} has invalid recovery mode {recovery_mode!r}")
    return by_task, failures


def _fault_evidence() -> tuple[dict[int, dict[str, Any]], list[str]]:
    expected = {fault.task_index: fault for fault in gate.FAULT_INJECTIONS}
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
        if int(evidence.get("generation", -1)) != gate.GENERATION:
            failures.append(f"task {task_index} fault receipt has a generation mismatch")
        if int(evidence.get("observed_global_step", -1)) < fault.trigger_step:
            failures.append(f"task {task_index} was preempted before its frozen trigger")
        if (
            int(evidence.get("checkpoint_step", 0)) <= 0
            or not evidence.get("checkpoint_path")
            or not evidence.get("checkpoint_metadata_sha256")
        ):
            failures.append(f"task {task_index} fault receipt lacks an independently observed checkpoint")
        if evidence.get("recovery_mode") != gate.RECOVERY_MODE_CHECKPOINT:
            failures.append(f"task {task_index} fault receipt is not checkpoint-backed")
        raw_search_paths = evidence.get("checkpoint_search_paths")
        if isinstance(raw_search_paths, list) and len(raw_search_paths) == 2:
            temporary_root = str(raw_search_paths[1]).rstrip("/")
            checkpoint_path = str(evidence.get("checkpoint_path", ""))
            if checkpoint_path != temporary_root and not checkpoint_path.startswith(f"{temporary_root}/"):
                failures.append(f"task {task_index} forced fault did not use a temporary checkpoint")
        else:
            failures.append(f"task {task_index} fault receipt has invalid checkpoint search paths")
        if int(evidence.get("source_attempt", -1)) < 0:
            failures.append(f"task {task_index} fault receipt lacks its source attempt")
        else:
            try:
                task_attempt = TaskAttempt.from_wire(str(evidence.get("task_attempt_id")))
                _, observed_task_index = task_attempt.task_id.require_task()
                observed_attempt = task_attempt.require_attempt()
            except ValueError as error:
                failures.append(f"task {task_index} fault receipt has invalid attempt identity: {error}")
            else:
                if observed_task_index != task_index or observed_attempt != int(evidence["source_attempt"]):
                    failures.append(f"task {task_index} fault receipt targeted another task or attempt")
        try:
            _timestamp(evidence.get("injected_at"), label=f"task {task_index} fault injected_at")
        except ValueError as error:
            failures.append(str(error))
    return observed, failures


def _checkpoint_signature(evidence: dict[str, Any]) -> tuple[str | None, int, str | None, tuple[str, ...], str | None]:
    raw_search_paths = evidence.get("checkpoint_search_paths")
    search_paths = tuple(raw_search_paths) if isinstance(raw_search_paths, list) else ()
    return (
        evidence.get("checkpoint_path"),
        int(evidence.get("checkpoint_step", 0)),
        evidence.get("checkpoint_metadata_sha256"),
        search_paths,
        evidence.get("recovery_mode"),
    )


def _authorization_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, dict[int, dict[str, Any]]], list[str]]:
    observed: dict[int, dict[int, dict[str, Any]]] = {index: {} for index in range(len(rows))}
    failures: list[str] = []
    for url in _evidence_objects("authorizations"):
        evidence = _read_json(url)
        task_index = int(evidence.get("task_index", -1))
        attempt_index = int(evidence.get("attempt", -1))
        if task_index not in observed or attempt_index <= 0:
            failures.append(f"unexpected recovery authorization task/attempt {task_index}/{attempt_index}")
            continue
        if attempt_index in observed[task_index]:
            failures.append(f"duplicate recovery authorization for task {task_index} attempt {attempt_index}")
            continue
        observed[task_index][attempt_index] = evidence

    for task_index, row in enumerate(rows):
        attempts_by_index = {int(item["attempt"]): item for item in attempts[task_index]}
        for attempt_index, authorization in observed[task_index].items():
            attempt = attempts_by_index.get(attempt_index)
            if attempt is None:
                failures.append(f"task {task_index} attempt {attempt_index} authorization has no worker claim")
                continue
            if int(authorization.get("generation", -1)) != gate.GENERATION:
                failures.append(f"task {task_index} attempt {attempt_index} authorization generation drifted")
            if authorization.get("row_id") != row.trajectory_id:
                failures.append(f"task {task_index} attempt {attempt_index} authorization row identity drifted")
            if authorization.get("task_attempt_id") != attempt.get("task_attempt_id"):
                failures.append(f"task {task_index} attempt {attempt_index} authorization attempt identity drifted")
            if authorization.get("parent_worker_region") != "us-central1":
                failures.append(f"task {task_index} attempt {attempt_index} authorization escaped us-central1")
            if not authorization.get("parent_worker_id"):
                failures.append(f"task {task_index} attempt {attempt_index} authorization lacks parent identity")
            if _checkpoint_signature(authorization) != _checkpoint_signature(attempt):
                failures.append(
                    f"task {task_index} attempt {attempt_index} worker checkpoint differs from parent authorization"
                )
            try:
                _timestamp(
                    authorization.get("authorized_at"),
                    label=f"task {task_index} attempt {attempt_index} authorized_at",
                )
            except ValueError as error:
                failures.append(str(error))
    return observed, failures


def _admission_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
    authorizations: dict[int, dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Validate immutable C64 releases and each attempt's release-consumption receipt."""
    failures: list[str] = []
    releases: dict[int, dict[str, Any]] = {}
    for url in _evidence_objects("admission/releases"):
        with fsspec.open(url, "rb") as handle:
            encoded = handle.read()
        release = json.loads(encoded)
        if not isinstance(release, dict):
            failures.append(f"admission release is not a JSON object: {url}")
            continue
        epoch = int(release.get("epoch", -1))
        if epoch in releases:
            failures.append(f"duplicate admission release epoch {epoch}")
            continue
        release["release_path"] = url
        release["release_sha256"] = _sha256_bytes(encoded)
        releases[epoch] = release

    epochs = sorted(releases)
    if epochs != [0]:
        failures.append(f"admission release inventory must be exactly epoch 0: {epochs}")
    attempt_lookup = {
        (task_index, int(attempt["attempt"])): attempt
        for task_index, task_attempts in attempts.items()
        for attempt in task_attempts
    }
    for epoch, release in releases.items():
        if int(release.get("generation", -1)) != gate.GENERATION:
            failures.append(f"admission epoch {epoch} generation drifted")
        if int(release.get("stage", -1)) != gate.STAGE:
            failures.append(f"admission epoch {epoch} stage drifted")
        if release.get("parent_worker_region") != "us-central1" or not release.get("parent_worker_id"):
            failures.append(f"admission epoch {epoch} lacks a central1 parent identity")
        try:
            released_at = _timestamp(release.get("released_at"), label=f"admission epoch {epoch} released_at")
        except ValueError as error:
            failures.append(str(error))
            released_at = math.inf
        bindings = release.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != gate.STAGE:
            failures.append(
                f"admission epoch {epoch} binds {0 if not isinstance(bindings, list) else len(bindings)} rows"
            )
            continue
        task_indexes = [int(binding.get("task_index", -1)) for binding in bindings]
        attempt_ids = [binding.get("task_attempt_id") for binding in bindings]
        if sorted(task_indexes) != list(range(gate.STAGE)):
            failures.append(f"admission epoch {epoch} does not bind every task exactly once")
        if len(set(attempt_ids)) != gate.STAGE:
            failures.append(f"admission epoch {epoch} has duplicate attempt identities")
        for binding in bindings:
            task_index = int(binding.get("task_index", -1))
            attempt_index = int(binding.get("attempt", -1))
            if task_index not in range(len(rows)):
                continue
            attempt = attempt_lookup.get((task_index, attempt_index))
            if attempt is None:
                failures.append(f"admission epoch {epoch} task {task_index}/{attempt_index} has no worker receipt")
                continue
            expected = {
                "attempt": attempt_index,
                "controller_state": "TASK_STATE_RUNNING",
                "ready_at": attempt.get("ready_at"),
                "row_id": rows[task_index].trajectory_id,
                "task_attempt_id": attempt.get("task_attempt_id"),
                "task_id": attempt.get("task_id"),
                "task_index": task_index,
                "worker_id": attempt.get("worker_id"),
                "worker_region": "us-central1",
            }
            observed = {key: binding.get(key) for key in expected}
            if observed != expected:
                failures.append(
                    f"admission epoch {epoch} task {task_index}/{attempt_index} binding differs from worker receipt"
                )
            try:
                ready_at = _timestamp(binding.get("ready_at"), label=f"admission task {task_index} ready_at")
            except ValueError as error:
                failures.append(str(error))
            else:
                if released_at < ready_at:
                    failures.append(f"admission epoch {epoch} was released before task {task_index} became ready")

    consumed: dict[int, dict[int, dict[str, Any]]] = {index: {} for index in range(len(rows))}
    seen_consumption: set[tuple[int, int]] = set()
    for url in _evidence_objects("admission/consumed"):
        receipt = _read_json(url)
        try:
            task_index = int(receipt.get("task_index", -1))
            attempt_index = int(receipt.get("attempt", -1))
        except (TypeError, ValueError):
            failures.append(f"malformed admission consumption task/attempt at {url}")
            continue
        if task_index not in consumed or attempt_index < 0:
            failures.append(f"unexpected admission consumption task/attempt {task_index}/{attempt_index}")
            continue
        consumption_key = (task_index, attempt_index)
        if consumption_key in seen_consumption:
            failures.append(f"duplicate admission consumption for task {task_index} attempt {attempt_index}")
            continue
        seen_consumption.add(consumption_key)
        try:
            release_epoch = int(receipt.get("release_epoch", -1))
        except (TypeError, ValueError):
            failures.append(f"task {task_index} attempt {attempt_index} has malformed admission epoch")
            continue
        release = releases.get(release_epoch)
        if release is None:
            failures.append(
                f"task {task_index} attempt {attempt_index} consumed unknown admission epoch {release_epoch}"
            )
            continue
        consumed[task_index][attempt_index] = receipt
        if receipt.get("release_path") != release.get("release_path"):
            failures.append(f"task {task_index} attempt {attempt_index} admission path drifted")
        if receipt.get("release_sha256") != release.get("release_sha256"):
            failures.append(f"task {task_index} attempt {attempt_index} admission hash drifted")
        attempt = attempt_lookup.get((task_index, attempt_index))
        if attempt is None:
            failures.append(f"task {task_index} attempt {attempt_index} consumption has no worker receipt")
            continue
        admission_mode = receipt.get("admission_mode")
        expected_receipt = {
            "attempt": attempt_index,
            "controller_state": "TASK_STATE_RUNNING" if admission_mode == "bound_current_attempt" else None,
            "ready_at": attempt.get("ready_at"),
            "row_id": rows[task_index].trajectory_id,
            "task_attempt_id": attempt.get("task_attempt_id"),
            "task_id": attempt.get("task_id"),
            "task_index": task_index,
            "worker_id": attempt.get("worker_id"),
            "worker_region": "us-central1",
        }
        observed_receipt = {key: receipt.get(key) for key in expected_receipt}
        if observed_receipt != expected_receipt:
            failures.append(f"task {task_index} attempt {attempt_index} consumption identity drifted")
        bindings = release.get("bindings")
        if not isinstance(bindings, list):
            continue
        matching = [
            binding
            for binding in bindings
            if int(binding.get("task_index", -1)) == task_index and int(binding.get("attempt", -1)) == attempt_index
        ]
        if admission_mode == "bound_current_attempt":
            if len(matching) != 1:
                failures.append(f"task {task_index} attempt {attempt_index} consumed a release that did not bind it")
        elif admission_mode == "post_release_retry":
            if attempt_index == 0:
                failures.append(f"task {task_index} initial attempt claimed post-release retry admission")
            if attempt_index not in authorizations.get(task_index, {}):
                failures.append(f"task {task_index} post-release retry {attempt_index} lacks recovery authorization")
            try:
                ready_at = _timestamp(
                    attempt.get("ready_at"), label=f"task {task_index} attempt {attempt_index} ready_at"
                )
                released_at = _timestamp(
                    release.get("released_at"), label=f"admission epoch {release_epoch} released_at"
                )
            except ValueError as error:
                failures.append(str(error))
            else:
                if ready_at < released_at:
                    failures.append(f"task {task_index} attempt {attempt_index} predates its post-release admission")
        else:
            failures.append(f"task {task_index} attempt {attempt_index} has invalid admission mode {admission_mode!r}")
        if int(receipt.get("generation", -1)) != gate.GENERATION or int(receipt.get("stage", -1)) != gate.STAGE:
            failures.append(f"task {task_index} attempt {attempt_index} consumption scope drifted")
        try:
            consumed_at = _timestamp(
                receipt.get("consumed_at"), label=f"task {task_index} attempt {attempt_index} consumed_at"
            )
            released_at = _timestamp(release.get("released_at"), label=f"admission epoch {release_epoch} released_at")
        except ValueError as error:
            failures.append(str(error))
        else:
            if consumed_at < released_at:
                failures.append(f"task {task_index} attempt {attempt_index} consumed admission before release")
    return {"consumed": consumed, "releases": releases}, failures


def _initialization_after_admission(
    state_evidence: dict[int, dict[int, dict[str, Any]]],
    admission: dict[str, Any],
) -> list[str]:
    """Require every observed trainer initialization to follow its matching release."""
    failures: list[str] = []
    consumed = admission.get("consumed", {})
    releases = admission.get("releases", {})
    for task_index, attempts in state_evidence.items():
        for attempt_index, state in attempts.items():
            receipt = consumed.get(task_index, {}).get(attempt_index)
            if receipt is None:
                failures.append(f"task {task_index} attempt {attempt_index} initialized without admission")
                continue
            try:
                release_epoch = int(receipt.get("release_epoch", -1))
            except (TypeError, ValueError):
                failures.append(f"task {task_index} attempt {attempt_index} has malformed admission epoch")
                continue
            release = releases.get(release_epoch)
            if release is None:
                continue
            try:
                initialized_at = _timestamp(
                    state.get("written_at"), label=f"task {task_index} attempt {attempt_index} state written_at"
                )
                consumed_at = _timestamp(
                    receipt.get("consumed_at"), label=f"task {task_index} attempt {attempt_index} consumed_at"
                )
            except ValueError as error:
                failures.append(str(error))
                continue
            if initialized_at < consumed_at:
                failures.append(f"task {task_index} attempt {attempt_index} initialized before admission consumption")
    return failures


def _state_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
    authorizations: dict[int, dict[int, dict[str, Any]]],
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
            if evidence.get("run_id") != gate._run_id(row):
                failures.append(f"task {task_index} attempt {attempt_index} state evidence has a run identity mismatch")
            state_step = int(evidence.get("state_step", -1))
            checkpoint_step = int(attempt.get("checkpoint_step", 0))
            expected_state_step = checkpoint_step + 1 if checkpoint_step > 0 else 0
            if state_step != expected_state_step:
                failures.append(
                    f"task {task_index} attempt {attempt_index} initialized at step {state_step}, "
                    f"expected {expected_state_step} from checkpoint label step-{checkpoint_step}"
                )
            if evidence.get("checkpoint_search_paths") != attempt.get("checkpoint_search_paths"):
                failures.append(f"task {task_index} attempt {attempt_index} state evidence search paths drifted")
            try:
                written_at = _timestamp(
                    evidence.get("written_at"),
                    label=f"task {task_index} attempt {attempt_index} state written_at",
                )
            except ValueError as error:
                failures.append(str(error))
                continue
            if attempt_index > 0:
                authorization = authorizations[task_index].get(attempt_index)
                if authorization is None:
                    failures.append(
                        f"task {task_index} attempt {attempt_index} initialized without parent authorization"
                    )
                    continue
                try:
                    authorized_at = _timestamp(
                        authorization.get("authorized_at"),
                        label=f"task {task_index} attempt {attempt_index} authorized_at",
                    )
                except ValueError as error:
                    failures.append(str(error))
                    continue
                if written_at < authorized_at:
                    failures.append(f"task {task_index} attempt {attempt_index} initialized before authorization")
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
            if (
                attempt_index <= source_attempt
                or worker_step < parent_step
                or evidence.get("recovery_mode") != gate.RECOVERY_MODE_CHECKPOINT
            ):
                continue
            if worker_step == parent_step and _checkpoint_signature(evidence) != _checkpoint_signature(fault):
                continue
            raw_search_paths = evidence.get("checkpoint_search_paths")
            if not isinstance(raw_search_paths, list) or len(raw_search_paths) != 2:
                continue
            temporary_root = str(raw_search_paths[1]).rstrip("/")
            checkpoint_path = str(evidence.get("checkpoint_path", ""))
            if checkpoint_path != temporary_root and not checkpoint_path.startswith(f"{temporary_root}/"):
                continue
            state = state_evidence.get(task_index, {}).get(attempt_index)
            if state is None or int(state.get("state_step", -1)) != worker_step + 1:
                continue
            try:
                state_written_at = _timestamp(
                    state.get("written_at"),
                    label=f"task {task_index} attempt {attempt_index} state written_at",
                )
                fault_injected_at = _timestamp(
                    fault.get("injected_at"),
                    label=f"task {task_index} fault injected_at",
                )
            except ValueError:
                continue
            if state_written_at <= fault_injected_at:
                continue
            candidates.append(evidence)
        if not candidates:
            failures.append(
                f"task {task_index} has no post-fault state restored from a temporary checkpoint "
                "at or after the parent-observed checkpoint"
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
    if any(item.get("_timestamp") is None or not math.isfinite(float(item["_timestamp"])) for item in rows):
        raise ValueError(f"Non-finite operational timestamps for {run.id}")
    return rows


def _active_intervals(history: list[dict[str, Any]]) -> tuple[tuple[float, float], ...]:
    """Split one W&B history into active spans, excluding retry/preemption gaps."""
    intervals: list[tuple[float, float]] = []
    start = float(history[0]["_timestamp"])
    previous_time = start
    previous_step = int(history[0]["global_step"])
    for item in history[1:]:
        current_time = float(item["_timestamp"])
        current_step = int(item["global_step"])
        if current_time < previous_time:
            raise ValueError("Operational W&B timestamps are not monotone")
        if current_time - previous_time > ACTIVE_GAP_SECONDS_MAX or current_step <= previous_step:
            intervals.append((start, previous_time))
            start = current_time
        previous_time = current_time
        previous_step = current_step
    intervals.append((start, previous_time))
    return tuple(interval for interval in intervals if interval[1] > interval[0])


def _concurrency_evidence(
    wandb_evidence: dict[str, Any],
    *,
    recovery_completed_at: float,
) -> tuple[dict[str, Any], list[str]]:
    """Measure sustained C64 activity only after every forced recovery completed."""
    events: dict[float, int] = {}
    for row_id, evidence in wandb_evidence.items():
        intervals = evidence.get("active_intervals", [])
        if not intervals:
            return {}, [f"{row_id}: no active interval available for C64 overlap"]
        for start, stop in intervals:
            clipped_start = max(float(start), recovery_completed_at)
            clipped_stop = float(stop)
            if clipped_stop <= clipped_start:
                continue
            events[clipped_start] = events.get(clipped_start, 0) + 1
            events[clipped_stop] = events.get(clipped_stop, 0) - 1

    active = 0
    peak = 0
    previous_time: float | None = None
    duration_at_full_load = 0.0
    for timestamp in sorted(events):
        if previous_time is not None and active == gate.STAGE:
            duration_at_full_load += timestamp - previous_time
        active += events[timestamp]
        peak = max(peak, active)
        previous_time = timestamp

    failures: list[str] = []
    if peak != gate.STAGE:
        failures.append(f"peak active workers {peak} != required C64 load {gate.STAGE}")
    if duration_at_full_load < FULL_LOAD_OVERLAP_SECONDS_MIN:
        failures.append(f"C64 active overlap {duration_at_full_load:.3f}s < {FULL_LOAD_OVERLAP_SECONDS_MIN:.3f}s")
    return {
        "active_gap_seconds_max": ACTIVE_GAP_SECONDS_MAX,
        "full_load_overlap_seconds": duration_at_full_load,
        "full_load_overlap_seconds_min": FULL_LOAD_OVERLAP_SECONDS_MIN,
        "peak_active_workers": peak,
        "recovery_completed_at": recovery_completed_at,
        "required_active_workers": gate.STAGE,
    }, failures


def _wandb_evidence(
    rows: tuple[Any, ...],
    attempts: dict[int, list[dict[str, Any]]],
    forced_retries: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    api = wandb.Api(timeout=60)
    evidence: dict[str, Any] = {}
    failures: list[str] = []
    fault_tasks = {fault.task_index for fault in gate.FAULT_INJECTIONS}
    expected_step_set = set(range(FIRST_OPERATIONAL_WANDB_STEP, stress.TERMINAL_STEP + 1))
    for task_index, row in enumerate(rows):
        run_id = gate._run_id(row)
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
        terminal_step = global_steps[-1]
        row_failures: list[str] = []
        if terminal_step != stress.TERMINAL_STEP:
            row_failures.append(f"terminal global step {terminal_step} != {stress.TERMINAL_STEP}")
        forced_retry = forced_retries.get(task_index)
        if task_index in fault_tasks and forced_retry is None:
            row_failures.append("forced-fault run lacks a matched checkpoint-recovery claim")
        observed_step_set = set(global_steps)
        if observed_step_set != expected_step_set:
            missing = sorted(expected_step_set - observed_step_set)
            unexpected = sorted(observed_step_set - expected_step_set)
            row_failures.append(
                f"W&B operational history does not cover exactly emitted steps "
                f"{FIRST_OPERATIONAL_WANDB_STEP}..{stress.TERMINAL_STEP}: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        try:
            intervals = _active_intervals(history)
        except ValueError as error:
            row_failures.append(f"invalid W&B activity timeline: {error}")
            intervals = ()
        failures.extend(f"{row.trajectory_id}: {failure}" for failure in row_failures)
        evidence[row.trajectory_id] = {
            "attempt_count": len(attempts[task_index]),
            "forced_retry_attempt": None if forced_retry is None else int(forced_retry["attempt"]),
            "history_rows": len(history),
            "active_intervals": intervals,
            "run_id": run_id,
            "status": "pass" if not row_failures else "fail",
            "terminal_global_step": terminal_step,
        }
    return evidence, failures


def analyze(*, parent_job: str, preregistration_sha256: str) -> dict[str, Any]:
    """Return a complete endpoint-blind recovery report."""
    preregistration = gate.validate_preregistration(preregistration_sha256)
    rows = stress.rows_for_stage(gate.STAGE)
    parent = runtime_gate._iris_summary(parent_job)
    children = runtime_gate._iris_child_summaries(parent_job)
    failures = runtime_gate._iris_failures(parent)
    if len(children) != 1:
        failures.append(f"expected one production-recovery child, found {len(children)}")
        child = None
    else:
        child = children[0]
        failures.extend(runtime_gate._iris_failures(child, allow_preemptions=True))
        if child["total_tasks"] != gate.STAGE:
            failures.append(f"child task count {child['total_tasks']} != required C64 load {gate.STAGE}")
        if child["preemptions"] < len(gate.FAULT_INJECTIONS):
            failures.append(
                f"child recorded {child['preemptions']} preemptions; expected at least {len(gate.FAULT_INJECTIONS)}"
            )

    attempts, attempt_failures = _attempt_evidence(rows)
    faults, fault_failures = _fault_evidence()
    authorizations, authorization_failures = _authorization_evidence(rows, attempts)
    admission, admission_failures = _admission_evidence(rows, attempts, authorizations)
    state_evidence, state_failures = _state_evidence(rows, attempts, authorizations)
    initialization_admission_failures = _initialization_after_admission(state_evidence, admission)
    forced_retries, forced_retry_failures = _forced_retry_evidence(attempts, faults, state_evidence)
    wandb_evidence, wandb_failures = _wandb_evidence(rows, attempts, forced_retries)
    recovery_completed_times: list[float] = []
    for task_index, retry in forced_retries.items():
        attempt_index = int(retry["attempt"])
        state = state_evidence.get(task_index, {}).get(attempt_index)
        if state is not None:
            try:
                recovery_completed_times.append(
                    _timestamp(
                        state.get("written_at"),
                        label=f"task {task_index} forced recovery written_at",
                    )
                )
            except ValueError as error:
                forced_retry_failures.append(str(error))
    if len(recovery_completed_times) == len(gate.FAULT_INJECTIONS):
        concurrency, concurrency_failures = _concurrency_evidence(
            wandb_evidence,
            recovery_completed_at=max(recovery_completed_times),
        )
    else:
        concurrency = {}
        concurrency_failures = [
            f"cannot measure post-recovery C64 overlap: recovered {len(recovery_completed_times)} "
            f"of {len(gate.FAULT_INJECTIONS)} forced rows"
        ]
    failures.extend(attempt_failures)
    failures.extend(fault_failures)
    failures.extend(authorization_failures)
    failures.extend(admission_failures)
    failures.extend(state_failures)
    failures.extend(initialization_admission_failures)
    failures.extend(forced_retry_failures)
    failures.extend(wandb_failures)
    failures.extend(concurrency_failures)
    remote_preregistration = _remote_preregistration(preregistration_sha256)
    return {
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "admission": admission,
        "attempts": attempts,
        "authorizations": authorizations,
        "child_iris": child,
        "concurrency": concurrency,
        "data_continuity_evidence": {
            "all_retry_attempts_require_parent_authorized_recovery_state": True,
            "pre_checkpoint_retries_require_parent_authorized_restart_from_zero": True,
            "checkpoint_label_is_completed_step": True,
            "checkpoint_restores_next_trainer_step": True,
            "deterministic_loader_position_is_derived_from_verified_restored_step": True,
            "frozen_train_lm_sha256": preregistration["implementation_sha256"]["train_lm"],
            "frozen_loader_sha256": preregistration["implementation_sha256"]["loader"],
            "frozen_dataset_sha256": preregistration["implementation_sha256"]["datasets"],
            "phase_schedule_is_global_step_indexed": True,
        },
        "endpoint_metrics_read": False,
        "failures": failures,
        "faults": faults,
        "forced_retries": forced_retries,
        "generation": gate.GENERATION,
        "parent_iris": parent,
        "preregistration_sha256": preregistration_sha256,
        "remote_preregistration": remote_preregistration,
        "stage": gate.STAGE,
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
        fs, path = fsspec.core.url_to_fs(f"{gate.EVIDENCE_ROOT}/runtime_gate.json")
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
