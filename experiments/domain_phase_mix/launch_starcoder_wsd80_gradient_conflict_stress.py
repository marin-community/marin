# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run short, non-scientific concurrency gates for the WSD80 gradient panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
import urllib.parse
from dataclasses import asdict, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import fsspec
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.client import Job, JobAlreadyExists, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import WellKnownAttribute, preemptible_constraint, region_constraint, zone_constraint
from iris.cluster.types import CoschedulingConfig, Entrypoint, TaskAttempt
from iris.rpc import job_pb2
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.muonh import MuonHConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, apply_output_path, run_levanter_train_lm
from rigging.filesystem import prefix_join

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

DEFAULT_GENERATION = 18
GENERATION_DATES = {
    9: "20260811",
    10: "20260812",
    11: "20260812",
    12: "20260812",
    13: "20260812",
    14: "20260812",
    15: "20260813",
    16: "20260813",
    17: "20260813",
    18: "20260813",
    19: "20260813",
    20: "20260813",
    21: "20260813",
    22: "20260813",
    23: "20260813",
}
STAGE_CONCURRENCIES = (6, 12, 24, 48, 64)
STAGE_SUPPORT_COUNTS = {
    6: {"m100a": 4, "full": 1, "m100b": 1},
    12: {"m100a": 8, "full": 3, "m100b": 1},
    24: {"m100a": 17, "full": 6, "m100b": 1},
    48: {"m100a": 34, "full": 12, "m100b": 2},
    64: {"m100a": 45, "full": 17, "m100b": 2},
}
TOTAL_STEPS = 3_328
DATA_SWITCH_STEP = 1_536
OPTIMIZER_DECAY_STEP = 2_688
C6_OPTIMIZER_DECAY_STEPS = (2_048, 2_176, 2_560, 2_688, 2_304, 2_432)
C12_ONSET_ASSIGNMENT_SEED = 2_026_081_202
C12_PRIMARY_ONSET_MULTISET = (1_920, 1_920, 2_304, 2_304, 2_688, 2_688, 3_072, 3_072)
C12_PRIMARY_OPTIMIZER_DECAY_STEPS = (2_304, 1_920, 1_920, 3_072, 3_072, 2_304, 2_688, 2_688)
C12_CONTROL_OPTIMIZER_DECAY_STEPS = (2_304, 2_432, 2_304, 2_432)
C12_OPTIMIZER_DECAY_STEPS = C12_PRIMARY_OPTIMIZER_DECAY_STEPS + C12_CONTROL_OPTIMIZER_DECAY_STEPS
TERMINAL_STEP = TOTAL_STEPS - 1
PHASE_0_STARCODER = 0.02
PHASE_1_STARCODER = 0.82
SUPPORT_BATCHES = 1_068
SUPPORT_POOL_SEED = 2_026_081_101
TRAIN_HOLDOUT_SEQUENCES = 4_096
RENDEZVOUS_ROOT = (
    "gs://marin-us-central1/tmp/ttl=14d/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_stress_rendezvous_20260811"
)
RENDEZVOUS_TIMEOUT_SECONDS = 1_800.0
RENDEZVOUS_POLL_SECONDS = 1.0
RENDEZVOUS_PROTOCOL_VERSION = "2026-08-13-parent-managed-cohort-retry-v8"
COHORT_MAX_PREEMPTION_RETRIES = 20
COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS = 4 * 60 * 60
COHORT_COSCHEDULING_GROUP = WellKnownAttribute.TPU_TOPOLOGY
PREREGISTRATION_PATHS = {
    9: Path(__file__).with_name("starcoder_wsd80_gradient_conflict_stress_gate_preregistration_20260812.json"),
    10: (
        Path(__file__).with_name(
            "starcoder_wsd80_gradient_conflict_stress_gate_preregistration_generation10_20260812.json"
        )
    ),
    11: (
        Path(__file__).with_name(
            "starcoder_wsd80_gradient_conflict_stress_gate_preregistration_generation11_20260812.json"
        )
    ),
    14: (
        Path(__file__).with_name(
            "starcoder_wsd80_gradient_conflict_stress_gate_preregistration_generation14_20260812.json"
        )
    ),
    15: (
        Path(__file__).with_name(
            "starcoder_wsd80_gradient_conflict_forced_retry_preregistration_generation15_revision2_20260813.json"
        )
    ),
    16: Path(__file__).with_name("starcoder_wsd80_gradient_conflict_c64_preregistration_generation16_20260813.json"),
    17: Path(__file__).with_name("starcoder_wsd80_gradient_conflict_c64_preregistration_generation17_20260813.json"),
    18: Path(__file__).with_name("starcoder_wsd80_gradient_conflict_c64_preregistration_generation18_20260813.json"),
}

_RENDEZVOUS_ID_ENV = "MARIN_STRESS_RENDEZVOUS_ID"
_RENDEZVOUS_ROW_ENV = "MARIN_STRESS_RENDEZVOUS_ROW"
_RENDEZVOUS_ROWS_ENV = "MARIN_STRESS_RENDEZVOUS_ROWS"
_RENDEZVOUS_ROOT_ENV = "MARIN_STRESS_RENDEZVOUS_ROOT"
_STRESS_GENERATION_ENV = "MARIN_STRESS_GENERATION"
_STRESS_STAGE_ENV = "MARIN_STRESS_STAGE"
_PARENT_PREEMPTIBLE_ENV = "MARIN_STRESS_PARENT_PREEMPTIBLE"
_PARENT_MAX_RETRIES_FAILURE_ENV = "MARIN_STRESS_PARENT_MAX_RETRIES_FAILURE"
_PARENT_MAX_RETRIES_PREEMPTION_ENV = "MARIN_STRESS_PARENT_MAX_RETRIES_PREEMPTION"


class _RendezvousObjectIncomplete(RuntimeError):
    """Signal that a rendezvous object exists but is not yet readable."""


class CohortPlacementMode(StrEnum):
    """Iris placement semantics for one stress cohort."""

    TOPOLOGY_COSCHEDULED = "topology_coscheduled"
    INDEPENDENT = "independent"


def _validate_generation(generation: int) -> None:
    """Reject invalid stress-attempt generation numbers."""
    if generation < 1:
        raise ValueError("Stress generation must be positive")


def generation_date(generation: int) -> str:
    """Return the historical or current date namespace for one generation."""
    _validate_generation(generation)
    return GENERATION_DATES.get(generation, "20260811" if generation < 10 else "20260812")


def preregistration_path(generation: int) -> Path:
    """Return the immutable local preregistration for one supported generation."""
    _validate_generation(generation)
    try:
        return PREREGISTRATION_PATHS[generation]
    except KeyError as error:
        raise ValueError(f"No frozen preregistration for stress generation {generation}") from error


def _local_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_preregistration(expected_sha256: str, generation: int, stage: int) -> dict[str, Any]:
    """Load the exact reviewed gate definition and validate this launcher against it."""
    if len(expected_sha256) != 64:
        raise ValueError("The preregistration SHA-256 must contain 64 hexadecimal characters")
    path = preregistration_path(generation)
    observed_sha256 = _local_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Stress preregistration drifted: {observed_sha256} != {expected_sha256}")
    preregistration = json.loads(path.read_text())
    if preregistration.get("generation") != generation:
        raise ValueError("Stress preregistration generation does not match the launch generation")
    if preregistration.get("design", {}).get("stage") != stage:
        raise ValueError("Stress preregistration does not authorize this concurrency stage")
    if preregistration.get("analysis_scope") != "operational_only_no_endpoint_metrics":
        raise ValueError("Stress preregistration scope drifted")
    expected_launcher_sha256 = preregistration.get("implementation_sha256", {}).get("stress_launcher")
    observed_launcher_sha256 = _local_sha256(Path(__file__))
    if observed_launcher_sha256 != expected_launcher_sha256:
        raise ValueError(
            f"Stress launcher drifted from preregistration: {observed_launcher_sha256} != {expected_launcher_sha256}"
        )
    implementation = preregistration.get("implementation_sha256", {})
    imported_sources = {
        "gradient_conflict_full": Path(full.__file__).resolve(),
        "wsd80_surface": Path(base.__file__).resolve(),
        "tests": Path(__file__).resolve().parents[2] / "tests/test_starcoder_wsd80_gradient_conflict_canary.py",
    }
    for name, path in imported_sources.items():
        observed = _local_sha256(path)
        if implementation.get(name) != observed:
            raise ValueError(f"Stress dependency {name} drifted: {observed} != {implementation.get(name)}")

    rows = rows_for_stage(stage)
    design = preregistration.get("design", {})
    expected_design = {
        "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
        "support_ids": [row.support_id for row in rows],
        "training_seeds": [row.training_seed for row in rows],
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            raise ValueError(f"Stress preregistration design field {key} drifted")
    _cohort_runtime_contract(preregistration)
    return preregistration


def _cohort_runtime_contract(preregistration: dict[str, Any]) -> tuple[str | None, float, float, int]:
    """Return the frozen placement and barrier timing for one cohort."""
    cohort = preregistration.get("design", {}).get("cohort", {})
    group_by = cohort.get("coscheduling_group_by")
    raw_placement_mode = cohort.get("placement_mode")
    if raw_placement_mode is None and group_by == COHORT_COSCHEDULING_GROUP:
        raw_placement_mode = CohortPlacementMode.TOPOLOGY_COSCHEDULED
    try:
        placement_mode = CohortPlacementMode(raw_placement_mode)
    except ValueError as error:
        raise ValueError("Stress preregistration lacks a valid cohort placement mode") from error
    if placement_mode is CohortPlacementMode.TOPOLOGY_COSCHEDULED:
        if group_by != COHORT_COSCHEDULING_GROUP:
            raise ValueError("Topology-coscheduled cohort group drifted")
    elif group_by is not None:
        raise ValueError("Independent cohort placement must not specify an Iris coscheduling group")

    start_timeout = float(cohort.get("start_barrier_timeout_seconds", RENDEZVOUS_TIMEOUT_SECONDS))
    completion_timeout = float(cohort.get("completion_barrier_timeout_seconds", RENDEZVOUS_TIMEOUT_SECONDS))
    if start_timeout <= 0 or completion_timeout <= 0:
        raise ValueError("Stress barrier timeouts must be positive")
    if max(start_timeout, completion_timeout) >= COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS:
        raise ValueError("Stress barrier timeout must be shorter than the parent cohort-attempt timeout")
    parent_retries = int(cohort.get("parent_managed_whole_cohort_retries", COHORT_MAX_PREEMPTION_RETRIES))
    if not 0 <= parent_retries <= COHORT_MAX_PREEMPTION_RETRIES:
        raise ValueError("Stress parent-managed retry count is outside the supported range")
    if placement_mode is CohortPlacementMode.INDEPENDENT and parent_retries != 0:
        raise ValueError("Independent cohort placement cannot safely use parent-managed whole-cohort retries")
    return group_by, start_timeout, completion_timeout, parent_retries


def namespace_name(generation: int) -> str:
    """Return the generation-scoped checkpoint namespace."""
    _validate_generation(generation)
    return (
        "pinlin_calvin_xu/data_mixture/"
        f"starcoder_wsd80_gradient_conflict_stress_retry{generation}_{generation_date(generation)}"
    )


def namespace_version(generation: int) -> str:
    """Return the generation-scoped immutable checkpoint version."""
    _validate_generation(generation)
    date = generation_date(generation)
    return f"{date[:4]}.{date[4:6]}.{date[6:]}.{generation}"


def wandb_group(generation: int) -> str:
    """Return the generation-scoped W&B group."""
    return namespace_name(generation)


def report_root(generation: int) -> str:
    """Return the generation-scoped remote report root."""
    _validate_generation(generation)
    return (
        "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
        f"starcoder_wsd80_gradient_conflict_stress_retry{generation}_{generation_date(generation)}"
    )


def stage_rendezvous_id(stage: int, generation: int) -> str:
    """Return the generation-scoped rendezvous identity for one stage."""
    if stage not in STAGE_CONCURRENCIES:
        raise ValueError(f"Unknown stress stage: {stage}")
    _validate_generation(generation)
    return f"c{stage:02d}-retry{generation}-{generation_date(generation)}"


NAME = namespace_name(DEFAULT_GENERATION)
VERSION = namespace_version(DEFAULT_GENERATION)
WANDB_GROUP = wandb_group(DEFAULT_GENERATION)
REPORT_ROOT = report_root(DEFAULT_GENERATION)


def report_path(stage: int, generation: int = DEFAULT_GENERATION) -> str:
    """Return the immutable report path expected for one completed stage."""
    return f"{report_root(generation)}/stage-c{stage:02d}.json"


def remote_preregistration_path(generation: int = DEFAULT_GENERATION) -> str:
    """Return the generation-scoped immutable preregistration path."""
    return f"{report_root(generation)}/preregistration.json"


def wandb_run_id(
    row: full.Trajectory,
    generation: int = DEFAULT_GENERATION,
    cohort_attempt: int | None = None,
) -> str:
    """Return the stage-specific W&B identity for one stress row and cohort attempt."""
    _validate_generation(generation)
    base = f"gcfstressr{generation}_{row.trajectory_id}"
    return base if cohort_attempt is None else f"{base}_a{cohort_attempt:03d}"


def cohort_attempt_id(stage: int, generation: int, cohort_attempt: int) -> str:
    """Return the parent-managed attempt identity for a co-scheduled cohort."""
    if cohort_attempt < 0:
        raise ValueError("Cohort attempt must be nonnegative")
    return f"{stage_rendezvous_id(stage, generation)}-attempt{cohort_attempt:03d}"


def cohort_rendezvous_id(stage: int, generation: int, cohort_attempt: int, iris_attempt: int) -> str:
    """Return one Iris-dispatch-scoped start barrier identity."""
    if iris_attempt < 0:
        raise ValueError("Iris task attempt must be nonnegative")
    return f"{cohort_attempt_id(stage, generation, cohort_attempt)}-iris{iris_attempt:03d}"


def completion_rendezvous_id(stage: int, generation: int, cohort_attempt: int, iris_attempt: int) -> str:
    """Return one Iris-dispatch-scoped completion barrier identity."""
    return f"complete-{cohort_rendezvous_id(stage, generation, cohort_attempt, iris_attempt)}"


def cohort_child_name(stage: int, generation: int, cohort_attempt: int) -> str:
    """Return the immutable Iris child name for one parent-managed attempt."""
    if cohort_attempt < 0:
        raise ValueError("Cohort attempt must be nonnegative")
    return f"stage-c{stage:02d}-cohort-g{generation}-attempt-{cohort_attempt:03d}"


def attempt_output_path(output_path: str, cohort_attempt: int) -> str:
    """Return a fresh output namespace for one infrastructure attempt."""
    if cohort_attempt < 0:
        raise ValueError("Cohort attempt must be nonnegative")
    return prefix_join(output_path, f"attempt-{cohort_attempt:03d}")


def _validate_rendezvous_contract(rendezvous_id: str, row_ids: tuple[str, ...]) -> None:
    """Validate one stage's exact rendezvous identity before allocation."""
    if not rendezvous_id or "/" in rendezvous_id:
        raise ValueError("The rendezvous ID must be a nonempty path segment")
    if not row_ids or len(set(row_ids)) != len(row_ids):
        raise ValueError("Rendezvous row IDs must be nonempty and unique")


def _rendezvous_paths(root: str, rendezvous_id: str, row_ids: tuple[str, ...]) -> tuple[Any, str, list[str], str]:
    """Resolve one isolated rendezvous into its filesystem paths."""
    _validate_rendezvous_contract(rendezvous_id, row_ids)
    base_url = prefix_join(root, rendezvous_id)
    fs, base_path = fsspec.core.url_to_fs(base_url)
    ready_dir = prefix_join(base_path, "ready")
    ready_paths = [prefix_join(ready_dir, f"{row_id}.json") for row_id in row_ids]
    return fs, ready_dir, ready_paths, prefix_join(base_path, "release.json")


def _validate_empty_rendezvous(root: str, rendezvous_id: str, row_ids: tuple[str, ...]) -> None:
    """Reject a reused rendezvous before scheduling any TPU workers."""
    fs, _, ready_paths, release_path = _rendezvous_paths(root, rendezvous_id, row_ids)
    occupied_paths = [path for path in [*ready_paths, release_path] if fs.exists(path)]
    if occupied_paths:
        raise ValueError(f"Stress rendezvous {rendezvous_id!r} already contains state: {occupied_paths}")


def _validate_empty_cohort_attempt_rendezvous(
    root: str,
    stage: int,
    generation: int,
    cohort_attempt: int,
) -> None:
    """Reject barrier state for any Iris dispatch of a not-yet-submitted cohort."""
    fs, root_path = fsspec.core.url_to_fs(root)
    attempt_id = cohort_attempt_id(stage, generation, cohort_attempt)
    patterns = (
        f"{attempt_id}-iris*/ready/*.json",
        f"{attempt_id}-iris*/release.json",
        f"complete-{attempt_id}-iris*/ready/*.json",
        f"complete-{attempt_id}-iris*/release.json",
    )
    occupied = tuple(sorted(path for pattern in patterns for path in fs.glob(prefix_join(root_path, pattern))))
    if occupied:
        raise ValueError(f"Stress cohort attempt {attempt_id!r} already contains rendezvous state: {occupied}")


def _cohort_release_paths(
    root: str,
    stage: int,
    generation: int,
    *,
    completion: bool = False,
) -> tuple[Any, tuple[str, ...]]:
    """Return every start or completion release for one cohort generation."""
    base_id = stage_rendezvous_id(stage, generation)
    prefix = "complete-" if completion else ""
    fs, root_path = fsspec.core.url_to_fs(root)
    pattern = prefix_join(root_path, f"{prefix}{base_id}-attempt*-iris*/release.json")
    return fs, tuple(sorted(fs.glob(pattern)))


def released_cohort_executions(
    root: str,
    stage: int,
    generation: int,
    *,
    completion: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Return parent and Iris attempts that reached one cohort barrier."""
    base_id = stage_rendezvous_id(stage, generation)
    prefix = "complete-" if completion else ""
    _, releases = _cohort_release_paths(root, stage, generation, completion=completion)
    pattern = re.compile(rf"(?:^|/){prefix}{re.escape(base_id)}-attempt(\d+)-iris(\d+)/release\.json$")
    executions: list[tuple[int, int]] = []
    for path in releases:
        match = pattern.search(path)
        if match is None:
            raise ValueError(f"Unexpected cohort release path: {path}")
        executions.append((int(match.group(1)), int(match.group(2))))
    return tuple(sorted(executions))


def _validate_resumable_stage_outputs(
    marin_prefix: str,
    stage: int,
    generation: int = DEFAULT_GENERATION,
    parent_managed_preemption_retries: int = COHORT_MAX_PREEMPTION_RETRIES,
) -> None:
    """Allow only parent-managed attempt outputs when resuming a stage."""
    stage_root = prefix_join(
        marin_prefix,
        f"tmp/ttl=14d/checkpoints/{namespace_name(generation)}/stage-c{stage:02d}",
    )
    fs, stage_path = fsspec.core.url_to_fs(stage_root)
    occupied_outputs = tuple(fs.find(stage_path, withdirs=False)) if fs.exists(stage_path) else ()
    report_fs, report_object = fsspec.core.url_to_fs(report_path(stage, generation))
    invalid_outputs: list[str] = []
    for path in occupied_outputs:
        match = re.search(r"(?:^|/)attempt-(\d{3})(?:/|$)", path)
        if match is None or int(match.group(1)) > parent_managed_preemption_retries:
            invalid_outputs.append(path)
    if invalid_outputs or report_fs.exists(report_object):
        raise ValueError(
            f"Stress stage C={stage} generation {generation} contains non-resumable state; "
            f"invalid_outputs={tuple(invalid_outputs)}, report_exists={report_fs.exists(report_object)}"
        )


def _write_json_once(fs: Any, path: str, payload: dict[str, Any]) -> None:
    """Atomically create one immutable rendezvous object."""
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    try:
        with fs.open(path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        with fs.open(path, "rb") as handle:
            existing = handle.read()
        if existing != encoded:
            raise RuntimeError(f"Stress rendezvous object is already claimed: {path}") from error
    with fs.open(path, "rb") as handle:
        persisted = handle.read()
    if persisted != encoded:
        raise RuntimeError(f"Stress rendezvous object did not persist exactly: {path}")


def _publish_preregistration_once(expected_sha256: str, generation: int, stage: int) -> dict[str, Any]:
    """Publish the locally validated preregistration before allocating workers."""
    validate_preregistration(expected_sha256, generation, stage)
    remote_url = remote_preregistration_path(generation)
    fs, path = fsspec.core.url_to_fs(remote_url)
    encoded = preregistration_path(generation).read_bytes()
    try:
        with fs.open(path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        with fs.open(path, "rb") as handle:
            if handle.read() != encoded:
                raise RuntimeError(f"Remote preregistration is already claimed: {remote_url}") from error
    with fs.open(path, "rb") as handle:
        remote_sha256 = hashlib.sha256(handle.read()).hexdigest()
    if remote_sha256 != expected_sha256:
        raise RuntimeError(f"Remote preregistration drifted: {remote_sha256} != {expected_sha256}")
    info = fs.info(path)
    return {
        "path": remote_url,
        "sha256": remote_sha256,
        "generation": None if info.get("generation") is None else str(info["generation"]),
    }


def _object_generation(fs: Any, path: str) -> str | None:
    """Return the immutable GCS generation when the filesystem exposes one."""
    generation = fs.info(path).get("generation")
    return None if generation is None else str(generation)


def _current_task_attempt() -> TaskAttempt:
    """Return the exact Iris child task attempt."""
    raw_task_id = os.environ.get("IRIS_TASK_ID")
    if raw_task_id is None:
        raise RuntimeError("Stress rendezvous requires an Iris task identity")
    return TaskAttempt.from_wire(raw_task_id)


def _validate_parent_runtime_contract() -> None:
    """Validate the parent metadata Iris exposes inside the running task.

    Iris propagates region and zone to child-submitting parents, but deliberately
    keeps preemptibility and retry budgets as per-job controller policy. The
    latter are externally read back from ``job_config`` before fault injection.
    """
    info = get_job_info()
    if info is None:
        raise RuntimeError("Stress cohort submission requires Iris parent-job metadata")
    required_constraints = {
        region_constraint(["us-central1"]),
        zone_constraint("us-central1-a"),
    }
    if not required_constraints.issubset(set(info.constraints)):
        raise RuntimeError(f"Stress parent placement metadata drifted: {info.constraints}")
    if preemptible_constraint(True) in info.constraints:
        raise RuntimeError("Stress parent must not inherit a preemptible=true constraint")
    expected_env = {
        _PARENT_PREEMPTIBLE_ENV: "false",
        _PARENT_MAX_RETRIES_FAILURE_ENV: "0",
        _PARENT_MAX_RETRIES_PREEMPTION_ENV: "0",
    }
    observed_env = {key: info.env.get(key) for key in expected_env}
    if observed_env != expected_env:
        raise RuntimeError(f"Stress parent retry attestation drifted: {observed_env} != {expected_env}")
    if _current_task_attempt().require_attempt() != 0:
        raise RuntimeError("Stress parent must run without an Iris retry attempt")


def _ready_marker_payload(
    *,
    worker_claim_id: str,
    row_id: str,
    rendezvous_id: str,
    row_ids: tuple[str, ...],
    physical_worker_id: str | None = None,
    worker_region: str | None = None,
) -> dict[str, Any]:
    """Build one attempt-stable marker for a logical child job."""
    if not worker_claim_id:
        raise ValueError("Stress rendezvous worker claim ID must be nonempty")
    marker_nonce = hashlib.sha256(
        "\0".join(
            (
                RENDEZVOUS_PROTOCOL_VERSION,
                rendezvous_id,
                row_id,
                worker_claim_id,
                physical_worker_id or "",
                worker_region or "",
            )
        ).encode()
    ).hexdigest()
    return {
        "marker_nonce": marker_nonce,
        "protocol_version": RENDEZVOUS_PROTOCOL_VERSION,
        "rendezvous_id": rendezvous_id,
        "row_id": row_id,
        "row_ids": list(row_ids),
        "worker_claim_id": worker_claim_id,
        "physical_worker_id": physical_worker_id,
        "worker_region": worker_region,
    }


def _read_ready_marker(
    fs: Any, path: str, *, row_id: str, rendezvous_id: str, row_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Read and validate one immutable ready marker plus its object generation."""
    try:
        with fs.open(path) as handle:
            marker = json.load(handle)
    except (EOFError, json.JSONDecodeError) as error:
        raise _RendezvousObjectIncomplete(f"Stress rendezvous marker is not yet readable: {path}") from error
    worker_claim_id = marker.get("worker_claim_id")
    physical_worker_id = marker.get("physical_worker_id")
    worker_region = marker.get("worker_region")
    expected_marker = (
        _ready_marker_payload(
            worker_claim_id=worker_claim_id,
            row_id=row_id,
            rendezvous_id=rendezvous_id,
            row_ids=row_ids,
            physical_worker_id=physical_worker_id,
            worker_region=worker_region,
        )
        if isinstance(worker_claim_id, str) and worker_claim_id
        else None
    )
    if (
        marker.get("protocol_version") != RENDEZVOUS_PROTOCOL_VERSION
        or marker.get("rendezvous_id") != rendezvous_id
        or marker.get("row_id") != row_id
        or marker.get("row_ids") != list(row_ids)
        or expected_marker is None
        or marker.get("marker_nonce") != expected_marker["marker_nonce"]
    ):
        raise RuntimeError(f"Stress rendezvous ready marker metadata does not match row {row_id}")
    return {
        "marker_nonce": str(marker["marker_nonce"]),
        "generation": _object_generation(fs, path),
        "worker_claim_id": worker_claim_id,
        "physical_worker_id": physical_worker_id,
        "worker_region": worker_region,
    }


def _ready_marker_identities(
    fs: Any,
    ready_paths: list[str],
    *,
    rendezvous_id: str,
    row_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Read the exact immutable marker identity for every stage row."""
    return {
        row_id: _read_ready_marker(
            fs,
            path,
            row_id=row_id,
            rendezvous_id=rendezvous_id,
            row_ids=row_ids,
        )
        for row_id, path in zip(row_ids, ready_paths, strict=True)
    }


def _validate_attempt_scoped_claims(
    marker_identities: dict[str, dict[str, Any]],
    *,
    rendezvous_id: str,
) -> None:
    """Require one complete Iris child dispatch in an execution barrier."""
    attempt_match = re.search(r"-iris(\d+)$", rendezvous_id)
    if attempt_match is None:
        return
    expected_attempt = int(attempt_match.group(1))
    claims = [TaskAttempt.from_wire(str(marker["worker_claim_id"])) for marker in marker_identities.values()]
    job_ids = {claim.job_id for claim in claims}
    attempt_ids = {claim.require_attempt() for claim in claims}
    task_indices = {claim.task_id.require_task()[1] for claim in claims}
    expected_indices = set(range(len(marker_identities)))
    if len(job_ids) != 1 or attempt_ids != {expected_attempt} or task_indices != expected_indices:
        raise RuntimeError(
            "Stress rendezvous does not contain one complete cohort attempt: "
            f"jobs={sorted(map(str, job_ids))}, attempts={sorted(attempt_ids)}, "
            f"tasks={sorted(task_indices)}, expected_tasks={sorted(expected_indices)}"
        )


def _wait_for_stage_rendezvous(
    *,
    root: str,
    rendezvous_id: str,
    row_id: str,
    row_ids: tuple[str, ...],
    worker_claim_id: str,
    physical_worker_id: str | None = None,
    worker_region: str | None = None,
    timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    poll_seconds: float = RENDEZVOUS_POLL_SECONDS,
) -> None:
    """Hold one allocated stress worker until every stage peer is ready."""
    _validate_rendezvous_contract(rendezvous_id, row_ids)
    if row_id not in row_ids:
        raise ValueError(f"Rendezvous row {row_id!r} is not in the expected stage rows")

    fs, ready_dir, ready_paths, release_path = _rendezvous_paths(root, rendezvous_id, row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    marker_path = ready_paths[row_ids.index(row_id)]
    marker_payload = _ready_marker_payload(
        worker_claim_id=worker_claim_id,
        row_id=row_id,
        rendezvous_id=rendezvous_id,
        row_ids=row_ids,
        physical_worker_id=physical_worker_id,
        worker_region=worker_region,
    )
    _write_json_once(fs, marker_path, marker_payload)

    logger.info("Stress rendezvous %s: %s is ready (%d expected)", rendezvous_id, row_id, len(row_ids))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(fs.exists(path) for path in ready_paths):
            try:
                marker_identities = _ready_marker_identities(
                    fs,
                    ready_paths,
                    rendezvous_id=rendezvous_id,
                    row_ids=row_ids,
                )
            except _RendezvousObjectIncomplete:
                time.sleep(poll_seconds)
                continue
            _validate_attempt_scoped_claims(marker_identities, rendezvous_id=rendezvous_id)
            if row_id == row_ids[0] and not fs.exists(release_path):
                _write_json_once(
                    fs,
                    release_path,
                    {
                        "protocol_version": RENDEZVOUS_PROTOCOL_VERSION,
                        "ready_markers": marker_identities,
                        "rendezvous_id": rendezvous_id,
                        "row_ids": list(row_ids),
                    },
                )
            if fs.exists(release_path):
                try:
                    with fs.open(release_path) as handle:
                        release = json.load(handle)
                except (EOFError, json.JSONDecodeError):
                    time.sleep(poll_seconds)
                    continue
                if (
                    release.get("protocol_version") != RENDEZVOUS_PROTOCOL_VERSION
                    or release.get("rendezvous_id") != rendezvous_id
                    or release.get("row_ids") != list(row_ids)
                    or release.get("ready_markers") != marker_identities
                ):
                    raise RuntimeError("Stress rendezvous release metadata does not match this stage")
                if marker_identities[row_id]["worker_claim_id"] != worker_claim_id:
                    raise RuntimeError(f"Stress rendezvous marker ownership changed for row {row_id}")
                logger.info("Stress rendezvous %s released row %s", rendezvous_id, row_id)
                return
        time.sleep(poll_seconds)

    ready_rows = [row for row, path in zip(row_ids, ready_paths, strict=True) if fs.exists(path)]
    raise TimeoutError(
        f"Stress rendezvous {rendezvous_id} timed out with {len(ready_rows)}/{len(row_ids)} rows ready: {ready_rows}"
    )


def _attempt_scoped_config(config: TrainLmOnPodConfig, cohort_attempt: int) -> TrainLmOnPodConfig:
    """Give one cohort attempt fresh checkpoints and W&B history."""
    if config.output_path is None:
        raise ValueError("Stress cohort configuration requires an output path")
    train_config = cast(TrainLmConfig, config.train_config)
    tracker = train_config.trainer.tracker
    if not isinstance(tracker, WandbConfig) or tracker.name is None:
        raise ValueError("Stress cohort configuration requires one named W&B tracker")

    run_id = f"{tracker.name}_a{cohort_attempt:03d}"
    output_path = attempt_output_path(config.output_path, cohort_attempt)
    trainer = replace(
        train_config.trainer,
        id=run_id,
        load_checkpoint=False,
        distributed=replace(train_config.trainer.distributed, initialize_jax_distributed=False),
        tracker=replace(
            tracker,
            id=run_id,
            name=run_id,
            replicate_path=output_path,
            resume="never",
        ),
    )
    return replace(
        config,
        output_path=output_path,
        train_config=replace(train_config, trainer=trainer),
        env_vars={**(config.env_vars or {}), "RUN_ID": run_id},
    )


def _run_stress_cohort(
    configs: tuple[TrainLmOnPodConfig, ...],
    cohort_attempt: int,
    start_barrier_timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    completion_barrier_timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
) -> None:
    """Run the row assigned to this replica inside one barrier-synchronized cohort."""
    attempt = _current_task_attempt()
    _, task_index = attempt.task_id.require_task()
    if task_index >= len(configs):
        raise ValueError(f"Stress cohort task index {task_index} exceeds {len(configs)} rows")
    iris_attempt = attempt.require_attempt()
    info = get_job_info()
    if info is None or info.worker_id is None or info.worker_region is None:
        raise RuntimeError("Stress cohort requires realized Iris worker identity and region")

    config = _attempt_scoped_config(configs[task_index], cohort_attempt)
    env = config.env_vars or {}
    rendezvous_id = cohort_rendezvous_id(
        int(env[_STRESS_STAGE_ENV]),
        int(env[_STRESS_GENERATION_ENV]),
        cohort_attempt,
        iris_attempt,
    )
    _wait_for_stage_rendezvous(
        root=env[_RENDEZVOUS_ROOT_ENV],
        rendezvous_id=rendezvous_id,
        row_id=env[_RENDEZVOUS_ROW_ENV],
        row_ids=tuple(json.loads(env[_RENDEZVOUS_ROWS_ENV])),
        worker_claim_id=attempt.to_wire(),
        physical_worker_id=info.worker_id,
        worker_region=info.worker_region,
        timeout_seconds=start_barrier_timeout_seconds,
    )
    run_levanter_train_lm(config)
    _wait_for_stage_rendezvous(
        root=env[_RENDEZVOUS_ROOT_ENV],
        rendezvous_id=completion_rendezvous_id(
            int(env[_STRESS_STAGE_ENV]),
            int(env[_STRESS_GENERATION_ENV]),
            cohort_attempt,
            iris_attempt,
        ),
        row_id=env[_RENDEZVOUS_ROW_ENV],
        row_ids=tuple(json.loads(env[_RENDEZVOUS_ROWS_ENV])),
        worker_claim_id=attempt.to_wire(),
        physical_worker_id=info.worker_id,
        worker_region=info.worker_region,
        timeout_seconds=completion_barrier_timeout_seconds,
    )


def _retryable_preempted_cohort(status: job_pb2.JobStatus) -> bool:
    """Return whether a failed child is safe to replace with a fresh cohort."""
    return status.state != job_pb2.JOB_STATE_SUCCEEDED and status.failure_count == 0 and status.preemption_count > 0


def _submit_stress_cohort(
    configs: tuple[TrainLmOnPodConfig, ...],
    *,
    stage: int,
    generation: int,
    rendezvous_root: str,
    coscheduling_group_by: str | None = COHORT_COSCHEDULING_GROUP,
    start_barrier_timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    completion_barrier_timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    parent_managed_preemption_retries: int = COHORT_MAX_PREEMPTION_RETRIES,
) -> str:
    """Run one cohort, optionally replacing cleanly preempted grouped attempts."""
    if len(configs) != stage:
        raise ValueError(f"Stress cohort has {len(configs)} configs for stage C={stage}")
    resources = configs[0].resources
    if any(config.resources != resources for config in configs[1:]):
        raise ValueError("Every stress cohort replica must request identical resources")

    context = iris_ctx()
    if context.client is None or context.job_id is None:
        raise RuntimeError("Stress cohort submission requires an in-cluster Iris client")
    constraints = [*convert_constraints(resources), preemptible_constraint(True)]
    if not 0 <= parent_managed_preemption_retries <= COHORT_MAX_PREEMPTION_RETRIES:
        raise ValueError("Parent-managed preemption retry count is outside the supported range")
    if coscheduling_group_by is None and parent_managed_preemption_retries != 0:
        raise ValueError("Independent cohort placement cannot safely retry a partially preempted cohort")
    for cohort_attempt in range(parent_managed_preemption_retries + 1):
        name = cohort_child_name(stage, generation, cohort_attempt)
        rendezvous_error: ValueError | None = None
        try:
            _validate_empty_cohort_attempt_rendezvous(rendezvous_root, stage, generation, cohort_attempt)
        except ValueError as error:
            rendezvous_error = error
        try:
            job = context.client.submit(
                entrypoint=Entrypoint.from_callable(
                    _run_stress_cohort,
                    configs,
                    cohort_attempt,
                    start_barrier_timeout_seconds,
                    completion_barrier_timeout_seconds,
                ),
                name=name,
                resources=convert_resources(resources),
                constraints=constraints,
                coscheduling=(
                    None if coscheduling_group_by is None else CoschedulingConfig(group_by=coscheduling_group_by)
                ),
                replicas=stage,
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
                existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
            )
        except JobAlreadyExists:
            job = Job(context.client, context.job_id.child(name))
            logger.info("Reattached to existing stress cohort %s", job.job_id)
        else:
            if rendezvous_error is not None:
                job.terminate()
                raise RuntimeError(
                    f"Refusing newly submitted stress cohort {job.job_id} because its rendezvous namespace is stale"
                ) from rendezvous_error
        logger.info("Waiting for barrier-synchronized stress cohort %s", job.job_id)
        try:
            status = job.wait(
                timeout=COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS,
                raise_on_failure=False,
            )
        except TimeoutError:
            job.terminate()
            raise
        if status.state == job_pb2.JOB_STATE_SUCCEEDED:
            return job.job_id.to_wire()
        if not _retryable_preempted_cohort(status):
            raise RuntimeError(
                f"Stress cohort {job.job_id} failed without a retryable infrastructure preemption: "
                f"state={job_pb2.JobState.Name(status.state)}, failures={status.failure_count}, "
                f"preemptions={status.preemption_count}"
            )
        logger.warning(
            "Stress cohort %s was preempted; replacing all %d replicas with attempt %d/%d",
            job.job_id,
            stage,
            cohort_attempt + 1,
            parent_managed_preemption_retries,
        )
    raise RuntimeError(f"Stress cohort exhausted {parent_managed_preemption_retries} parent-managed preemption retries")


def _stress_resources(*, tpu_type: str, tpu_region: str, tpu_zone: str) -> ResourceConfig:
    """Return schedulable central1 spot resources used by every stress row."""
    return ResourceConfig.with_tpu(
        tpu_type,
        cpu=full.TPU_HOST_CPU,
        preemptible=True,
        ram=full.TPU_HOST_RAM,
        regions=(tpu_region,),
        zone=tpu_zone,
    )


def rows_for_stage(stage: int) -> tuple[full.Trajectory, ...]:
    """Construct the exact non-scientific rows for a concurrency stage."""
    if stage not in STAGE_CONCURRENCIES:
        raise ValueError(f"Unknown stress stage: {stage}")
    support_ids = tuple(support_id for support_id, count in STAGE_SUPPORT_COUNTS[stage].items() for _ in range(count))
    if len(support_ids) != stage:
        raise ValueError(f"Stress-stage composition does not sum to {stage}")

    if stage == 6:
        optimizer_decay_steps = C6_OPTIMIZER_DECAY_STEPS
    elif stage == 12:
        expected_onsets = list(C12_PRIMARY_ONSET_MULTISET)
        random.Random(C12_ONSET_ASSIGNMENT_SEED).shuffle(expected_onsets)
        if tuple(expected_onsets) != C12_PRIMARY_OPTIMIZER_DECAY_STEPS:
            raise ValueError("C12 randomized optimizer-decay assignment drifted")
        optimizer_decay_steps = C12_OPTIMIZER_DECAY_STEPS
    else:
        optimizer_decay_steps = (OPTIMIZER_DECAY_STEP,) * stage
    rows: list[full.Trajectory] = []
    for index, support_id in enumerate(support_ids):
        support_batches = None if support_id == "full" else SUPPORT_BATCHES
        support_start = None
        support_seed = None
        if support_id == "m100a":
            support_start = 0
            support_seed = SUPPORT_POOL_SEED
        elif support_id == "m100b":
            support_start = SUPPORT_BATCHES
            support_seed = SUPPORT_POOL_SEED
        rows.append(
            full.Trajectory(
                trajectory_id=f"c{stage:02d}_{index:03d}_{support_id}",
                arm="operational_stress",
                cell_id="stress_h0640_s03328",
                support_id=support_id,
                support_pool_seed=support_seed,
                training_seed=2_026_089_000 + stage * 100 + index,
                policy_role="decoupled_switch_s1536_staggered_decay",
                phase_0_fraction=DATA_SWITCH_STEP / TOTAL_STEPS,
                phase_1_fraction=1.0 - DATA_SWITCH_STEP / TOTAL_STEPS,
                phase_0_starcoder=PHASE_0_STARCODER,
                phase_1_starcoder=PHASE_1_STARCODER,
                aggregate_starcoder=(
                    DATA_SWITCH_STEP * PHASE_0_STARCODER + (TOTAL_STEPS - DATA_SWITCH_STEP) * PHASE_1_STARCODER
                )
                / TOTAL_STEPS,
                phase_contrast_p0_minus_p1=PHASE_0_STARCODER - PHASE_1_STARCODER,
                upstream_phase_contrast_p1_minus_p0=PHASE_1_STARCODER - PHASE_0_STARCODER,
                coordinate_selection_rule="operational gate; no scientific inference",
                total_steps=TOTAL_STEPS,
                boundary_step=DATA_SWITCH_STEP,
                optimizer_decay_step=optimizer_decay_steps[index],
                primary_inference=False,
                support_start_batches=support_start,
                support_batches=support_batches,
                train_holdout_sequences_per_component=TRAIN_HOLDOUT_SEQUENCES,
                train_holdout_seed=full.EXPECTED_TRAIN_HOLDOUT_SEED,
                train_holdout_partition=full.EXPECTED_TRAIN_HOLDOUT_PARTITION,
                starcoder_phase_0_sequences=0,
                starcoder_phase_1_sequences=0,
                starcoder_total_sequences=0,
                realized_aggregate_starcoder=0.0,
                realized_phase_0_starcoder_per_block=0,
                realized_phase_1_starcoder_per_block=0,
            )
        )
    return tuple(rows)


def _optimizer_for_row(row: full.Trajectory) -> MuonHConfig:
    """Return the historical optimizer with this operational row's decay onset."""
    optimizer = base._optimizer(TOTAL_STEPS * base.BATCH_SIZE * base.SEQ_LEN)
    return replace(optimizer, decay=TOTAL_STEPS - row.optimizer_decay_step)


def _configure_stress_training(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    row: full.Trajectory,
    phase_weights: list[tuple[int, dict[str, float]]],
    training_component_names: tuple[str, ...],
    starcoder_name: str,
    rendezvous_id: str,
    rendezvous_row_ids: tuple[str, ...],
    stage: int,
    generation: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Install the production loader and checkpoint contracts."""

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod_config = training.build_config(ctx)
        train_config = cast(TrainLmConfig, pod_config.train_config)
        support_cap = None if row.support_batches is None else {starcoder_name: row.support_batches}
        support_start = None if row.support_start_batches is None else {starcoder_name: row.support_start_batches}
        data_config = replace(
            train_config.data,
            train_weights=phase_weights,
            mixture_block_size=base.MIXTURE_BLOCK_SIZE,
            experiment_budget=None,
            target_budget=None,
            simulated_epoch_subset_seed=None,
            max_train_batches=support_cap,
            max_train_batches_subset_seed=row.support_pool_seed,
            max_train_batches_start=support_start,
            train_holdout_sequences={name: TRAIN_HOLDOUT_SEQUENCES for name in training_component_names},
            train_holdout_seed=full.EXPECTED_TRAIN_HOLDOUT_SEED,
            train_holdout_partition=full.EXPECTED_TRAIN_HOLDOUT_PARTITION,
        )
        trainer = replace(
            train_config.trainer,
            seed=row.training_seed,
            distributed=replace(train_config.trainer.distributed, initialize_jax_distributed=False),
            checkpointer=replace(
                train_config.trainer.checkpointer,
                save_interval=timedelta(hours=1),
                keep=[
                    {"every": DATA_SWITCH_STEP, "until": DATA_SWITCH_STEP},
                    {"every": TERMINAL_STEP, "until": None},
                ],
                keep_last_temporary_checkpoints=1,
            ),
        )
        return replace(
            pod_config,
            train_config=replace(
                train_config,
                data=data_config,
                data_seed=row.training_seed,
                trainer=trainer,
            ),
            env_vars={
                **(pod_config.env_vars or {}),
                _RENDEZVOUS_ID_ENV: rendezvous_id,
                _RENDEZVOUS_ROOT_ENV: RENDEZVOUS_ROOT,
                _RENDEZVOUS_ROW_ENV: row.trajectory_id,
                _RENDEZVOUS_ROWS_ENV: json.dumps(rendezvous_row_ids),
                _STRESS_GENERATION_ENV: str(generation),
                _STRESS_STAGE_ENV: str(stage),
            },
        )

    return replace(training, build_config=build_config)


def build_steps(
    *,
    stage: int,
    marin_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    rendezvous_id: str,
    generation: int = DEFAULT_GENERATION,
) -> tuple[tuple[full.Trajectory, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build the exact short rows for one concurrency stage."""
    rows = rows_for_stage(stage)
    nemotron, starcoder, _ = full._training_data()
    handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    component_names = tuple(handle.name for handle in handles)
    if component_names != full.EXPECTED_TRAINING_COMPONENT_NAMES:
        raise ValueError("Stress-stage training components drifted")
    resources = _stress_resources(tpu_type=tpu_type, tpu_region=tpu_region, tpu_zone=tpu_zone)
    name = namespace_name(generation)
    version = namespace_version(generation)
    model = CompletedAdamHHeuristic()._build_model_config(640, seq_len=base.SEQ_LEN)
    if model.total_trainable_params(llama3_tokenizer_vocab_size) != 210_052_480:
        raise ValueError("Stress-stage model shape drifted")

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    rendezvous_row_ids = tuple(row.trajectory_id for row in rows)
    for row in rows:
        phase_0_weights = base._phase_leaf_weights(PHASE_0_STARCODER, nemotron=nemotron, starcoder=starcoder)
        phase_1_weights = base._phase_leaf_weights(PHASE_1_STARCODER, nemotron=nemotron, starcoder=starcoder)
        static_weights = {handle: phase_0_weights[handle.name] for handle in handles}
        training = train_lm(
            name=f"tmp/ttl=14d/checkpoints/{name}/stage-c{stage:02d}/{row.trajectory_id}",
            version=version,
            model=model,
            optimizer=_optimizer_for_row(row),
            datasets=static_weights,
            validation=(),
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=TOTAL_STEPS,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=TOTAL_STEPS + 1,
            wandb_project="marin",
            wandb_group=wandb_group(generation),
            run_id=wandb_run_id(row, generation),
            tags=("gradient_conflict_stress", f"c{stage:02d}", row.support_id, "no_scientific_inference"),
        )
        steps.append(
            _configure_stress_training(
                training,
                row=row,
                phase_weights=[(0, phase_0_weights), (DATA_SWITCH_STEP, phase_1_weights)],
                training_component_names=component_names,
                starcoder_name=starcoder.name,
                rendezvous_id=rendezvous_id,
                rendezvous_row_ids=rendezvous_row_ids,
                stage=stage,
                generation=generation,
            )
        )
    return rows, tuple(steps)


def audit_runtime_configs(
    rows: tuple[full.Trajectory, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
    stage: int,
    rendezvous_id: str,
    generation: int = DEFAULT_GENERATION,
) -> tuple[TrainLmOnPodConfig, ...]:
    """Materialize and validate every short stress configuration."""
    loaded_artifacts: dict[int, Any] = {}
    nemotron, starcoder, _ = full._training_data()
    configs: list[TrainLmOnPodConfig] = []
    for row, step in zip(rows, steps, strict=True):
        pod_config = materialized_config(step, marin_prefix, artifact_cache=loaded_artifacts)
        configs.append(pod_config)
        if pod_config.resources.cpu != full.TPU_HOST_CPU:
            raise ValueError(f"{row.trajectory_id}: TPU host CPU request drifted")
        if pod_config.resources.ram != full.TPU_HOST_RAM:
            raise ValueError(f"{row.trajectory_id}: TPU host RAM request drifted")
        if pod_config.resources.preemptible is not True:
            raise ValueError(f"{row.trajectory_id}: stress child TPU request must be explicitly preemptible")
        expected_rendezvous_env = {
            _RENDEZVOUS_ID_ENV: rendezvous_id,
            _RENDEZVOUS_ROOT_ENV: RENDEZVOUS_ROOT,
            _RENDEZVOUS_ROW_ENV: row.trajectory_id,
            _RENDEZVOUS_ROWS_ENV: json.dumps(tuple(candidate.trajectory_id for candidate in rows)),
            _STRESS_GENERATION_ENV: str(generation),
            _STRESS_STAGE_ENV: str(stage),
        }
        if pod_config.env_vars != expected_rendezvous_env:
            raise ValueError(f"{row.trajectory_id}: stress rendezvous contract drifted")
        train_config = cast(TrainLmConfig, pod_config.train_config)
        expected_output = prefix_join(
            marin_prefix,
            f"tmp/ttl=14d/checkpoints/{namespace_name(generation)}/stage-c{stage:02d}/"
            f"{row.trajectory_id}/{namespace_version(generation)}",
        )
        if pod_config.output_path != expected_output:
            raise ValueError(f"{row.trajectory_id}: output path drifted")
        if train_config.trainer.num_train_steps != TOTAL_STEPS:
            raise ValueError(f"{row.trajectory_id}: training horizon drifted")
        if train_config.trainer.seed != row.training_seed or train_config.data_seed != row.training_seed:
            raise ValueError(f"{row.trajectory_id}: model/data seed drifted")
        if train_config.trainer.distributed.initialize_jax_distributed:
            raise ValueError(f"{row.trajectory_id}: independent cohort row would join cross-replica JAX")
        expected_optimizer = asdict(_optimizer_for_row(row))
        if asdict(train_config.optimizer) != expected_optimizer:
            raise ValueError(f"{row.trajectory_id}: optimizer schedule drifted")
        if asdict(train_config.optimizer)["decay"] != TOTAL_STEPS - row.optimizer_decay_step:
            raise ValueError(f"{row.trajectory_id}: decay onset drifted")
        phase_weights = train_config.data.train_weights
        if not isinstance(phase_weights, list) or [boundary for boundary, _ in phase_weights] != [0, DATA_SWITCH_STEP]:
            raise ValueError(f"{row.trajectory_id}: data-switch schedule drifted")
        expected_phase_weights = [
            base._phase_leaf_weights(PHASE_0_STARCODER, nemotron=nemotron, starcoder=starcoder),
            base._phase_leaf_weights(PHASE_1_STARCODER, nemotron=nemotron, starcoder=starcoder),
        ]
        for phase_index, ((_, observed_weights), expected_weights) in enumerate(
            zip(phase_weights, expected_phase_weights, strict=True)
        ):
            if observed_weights != expected_weights:
                raise ValueError(f"{row.trajectory_id}: phase-{phase_index} weights drifted")
        expected_cap = None if row.support_batches is None else {"dolma/starcoder": row.support_batches}
        expected_start = None if row.support_start_batches is None else {"dolma/starcoder": row.support_start_batches}
        if train_config.data.max_train_batches != expected_cap:
            raise ValueError(f"{row.trajectory_id}: finite-support cap drifted")
        if train_config.data.max_train_batches_start != expected_start:
            raise ValueError(f"{row.trajectory_id}: finite-support offset drifted")
        if train_config.data.max_train_batches_subset_seed != row.support_pool_seed:
            raise ValueError(f"{row.trajectory_id}: support-pool seed drifted")
        if train_config.data.train_holdout_sequences != {
            name: TRAIN_HOLDOUT_SEQUENCES for name in full.EXPECTED_TRAINING_COMPONENT_NAMES
        }:
            raise ValueError(f"{row.trajectory_id}: holdout count drifted")
        if train_config.data.train_holdout_seed != full.EXPECTED_TRAIN_HOLDOUT_SEED:
            raise ValueError(f"{row.trajectory_id}: holdout seed drifted")
        if train_config.data.train_holdout_partition != full.EXPECTED_TRAIN_HOLDOUT_PARTITION:
            raise ValueError(f"{row.trajectory_id}: holdout partition drifted")
        if train_config.data.permutation_type != "feistel":
            raise ValueError(f"{row.trajectory_id}: permutation type drifted")
        if train_config.data.mixture_block_size != base.MIXTURE_BLOCK_SIZE:
            raise ValueError(f"{row.trajectory_id}: mixture block size drifted")
        if train_config.data.experiment_budget is not None or train_config.data.target_budget is not None:
            raise ValueError(f"{row.trajectory_id}: simulated budget leaked into stress stage")
        if train_config.data.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{row.trajectory_id}: simulated subset leaked into stress stage")
        tracker = train_config.trainer.tracker
        if not isinstance(tracker, WandbConfig) or tracker.name != wandb_run_id(row, generation):
            raise ValueError(f"{row.trajectory_id}: W&B identity drifted")
        if train_config.trainer.checkpointer.save_interval != timedelta(hours=1):
            raise ValueError(f"{row.trajectory_id}: periodic checkpoint interval drifted")
        if train_config.trainer.checkpointer.keep != [
            {"every": DATA_SWITCH_STEP, "until": DATA_SWITCH_STEP},
            {"every": TERMINAL_STEP, "until": None},
        ]:
            raise ValueError(f"{row.trajectory_id}: permanent checkpoint policy drifted")
        if train_config.trainer.checkpointer.keep_last_temporary_checkpoints != 1:
            raise ValueError(f"{row.trajectory_id}: temporary checkpoint retention drifted")
        attempt_config = _attempt_scoped_config(pod_config, 0)
        expected_attempt_output = attempt_output_path(expected_output, 0)
        if attempt_config.output_path != expected_attempt_output:
            raise ValueError(f"{row.trajectory_id}: attempt-scoped output path drifted")
        attempt_train_config = cast(TrainLmConfig, attempt_config.train_config)
        if attempt_train_config.trainer.distributed.initialize_jax_distributed:
            raise ValueError(f"{row.trajectory_id}: attempt-scoped config would join cross-replica JAX")
        if attempt_train_config.trainer.id != wandb_run_id(row, generation, 0):
            raise ValueError(f"{row.trajectory_id}: attempt-scoped trainer identity drifted")
        runtime_train_config = apply_output_path(attempt_train_config, expected_attempt_output)
        runtime_checkpointer = runtime_train_config.trainer.checkpointer
        if runtime_checkpointer.base_path != prefix_join(expected_attempt_output, "checkpoints"):
            raise ValueError(f"{row.trajectory_id}: runtime checkpoint path drifted")
        parsed_output = urllib.parse.urlparse(expected_attempt_output)
        output_component = f"{parsed_output.netloc}{parsed_output.path}".strip("/")
        expected_temporary_path = prefix_join(
            marin_prefix,
            f"tmp/ttl=14d/checkpoints-temp/{output_component}/checkpoints",
        )
        if runtime_checkpointer.temporary_base_path != expected_temporary_path:
            raise ValueError(f"{row.trajectory_id}: runtime temporary checkpoint path drifted")
        if runtime_checkpointer.append_run_id_to_base_path:
            raise ValueError(f"{row.trajectory_id}: runtime checkpointer would append a second run identity")
        if runtime_checkpointer.keep_last_temporary_checkpoints != 1:
            raise ValueError(f"{row.trajectory_id}: runtime temporary retention drifted")
    return tuple(configs)


def _validate_previous_stage(
    stage: int,
    report_sha256: str | None,
    previous_stage_generation: int | None = None,
    *,
    preregistration: dict[str, Any] | None = None,
) -> None:
    stage_index = STAGE_CONCURRENCIES.index(stage)
    if stage_index == 0:
        if report_sha256 is not None or previous_stage_generation is not None:
            raise ValueError("The first stress stage must not cite a predecessor report or generation")
        return
    predecessor = preregistration.get("predecessor") if preregistration is not None else None
    preregistered_stage = predecessor.get("stage") if isinstance(predecessor, dict) else None
    if preregistered_stage is not None and not isinstance(preregistered_stage, int):
        raise ValueError(f"Stage C={stage} preregistration predecessor stage must be an integer")
    previous_stage = preregistered_stage if preregistered_stage is not None else STAGE_CONCURRENCIES[stage_index - 1]
    if previous_stage not in STAGE_CONCURRENCIES or STAGE_CONCURRENCIES.index(previous_stage) >= stage_index:
        raise ValueError(f"Stage C={stage} preregistration has an invalid predecessor stage C={previous_stage}")
    if report_sha256 is None or len(report_sha256) != 64:
        raise ValueError(f"Stage C={stage} requires the SHA-256 of the passing C={previous_stage} report")
    if previous_stage_generation is None:
        raise ValueError(f"Stage C={stage} requires the generation of the passing C={previous_stage} report")
    if preregistration is not None:
        if not isinstance(predecessor, dict):
            raise ValueError(f"Stage C={stage} preregistration lacks a predecessor contract")
        if predecessor.get("stage", previous_stage) != previous_stage:
            raise ValueError("Previous stress-stage stage does not match the preregistration")
        if predecessor.get("generation") != previous_stage_generation:
            raise ValueError("Previous stress-stage generation does not match the preregistration")
        if predecessor.get("runtime_report_sha256") != report_sha256:
            raise ValueError("Previous stress-stage report hash does not match the preregistration")
    path = report_path(previous_stage, previous_stage_generation)
    observed_sha256 = full._remote_sha256(path)
    if observed_sha256 != report_sha256:
        raise ValueError(f"Previous stress-stage report drifted: {observed_sha256} != {report_sha256}")
    with fsspec.open(path) as handle:
        report = json.load(handle)
    if report.get("stage") != previous_stage or report.get("status") != "pass":
        raise ValueError(f"Previous stress stage C={previous_stage} did not pass")
    if preregistration is not None:
        fs, remote_path = fsspec.core.url_to_fs(path)
        observed_generation = _object_generation(fs, remote_path)
        expected_generation = preregistration["predecessor"].get("runtime_report_remote_generation")
        if observed_generation != expected_generation:
            raise ValueError(
                f"Previous stress-stage object generation drifted: {observed_generation} != {expected_generation}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=STAGE_CONCURRENCIES, required=True)
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--generation", type=int, default=DEFAULT_GENERATION)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--previous-stage-report-sha256")
    parser.add_argument("--previous-stage-generation", type=int)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--audit-runtime-configs", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping WSD80 gradient-conflict stress stage in CI")
        return
    if args.max_concurrent != args.stage:
        raise ValueError("Each stress stage must run all of its rows concurrently")
    if (
        args.marin_prefix != base.DEFAULT_MARIN_PREFIX
        or args.tpu_region != base.DEFAULT_TPU_REGION
        or args.tpu_zone != base.DEFAULT_TPU_ZONE
    ):
        raise ValueError("Historical StarCoder stress gates must remain central1-local")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError("Stress-stage accelerator shape drifted")

    _validate_generation(args.generation)
    preregistration = validate_preregistration(args.preregistration_sha256, args.generation, args.stage)
    (
        coscheduling_group_by,
        start_barrier_timeout,
        completion_barrier_timeout,
        parent_managed_preemption_retries,
    ) = _cohort_runtime_contract(preregistration)
    rendezvous_id = stage_rendezvous_id(args.stage, args.generation)
    _validate_previous_stage(
        args.stage,
        args.previous_stage_report_sha256,
        args.previous_stage_generation,
        preregistration=preregistration,
    )
    rows, steps = build_steps(
        stage=args.stage,
        marin_prefix=args.marin_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        rendezvous_id=rendezvous_id,
        generation=args.generation,
    )
    configs = audit_runtime_configs(
        rows,
        steps,
        marin_prefix=args.marin_prefix,
        stage=args.stage,
        rendezvous_id=rendezvous_id,
        generation=args.generation,
    )
    if args.audit_runtime_configs:
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    full.dense._validate_runtime_scientific_environment()
    full.audit_sources(args.marin_prefix, rows)
    if args.audit_source:
        return
    _validate_resumable_stage_outputs(
        args.marin_prefix,
        args.stage,
        args.generation,
        parent_managed_preemption_retries,
    )
    _validate_parent_runtime_contract()
    _publish_preregistration_once(args.preregistration_sha256, args.generation, args.stage)
    child_job = _submit_stress_cohort(
        configs,
        stage=args.stage,
        generation=args.generation,
        rendezvous_root=RENDEZVOUS_ROOT,
        coscheduling_group_by=coscheduling_group_by,
        start_barrier_timeout_seconds=start_barrier_timeout,
        completion_barrier_timeout_seconds=completion_barrier_timeout,
        parent_managed_preemption_retries=parent_managed_preemption_retries,
    )
    logger.info("Barrier-synchronized stress cohort completed: %s", child_job)


if __name__ == "__main__":
    main()
