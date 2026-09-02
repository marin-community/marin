# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise production-cadence checkpoint recovery under the full WSD80 C64 load."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import fsspec
import wandb
from fray.iris_backend import convert_constraints, convert_resources
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import preemptible_constraint
from iris.cluster.types import Entrypoint, JobName, TaskAttempt
from iris.rpc import job_pb2
from levanter.checkpoint import latest_checkpoint_path
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from marin.training.training import TrainLmOnPodConfig, apply_output_path, run_levanter_train_lm
from rigging.filesystem import prefix_join

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base

logger = logging.getLogger(__name__)

GENERATION = 24
STAGE = 64
CHECKPOINT_INTERVAL = full.CHECKPOINT_INTERVAL
MAX_PREEMPTION_RETRIES = 100
JOB_TIMEOUT_SECONDS = 4 * 60 * 60
POLL_SECONDS = 5.0
RECOVERY_AUTHORIZATION_TIMEOUT_SECONDS = 20 * 60
ADMISSION_TIMEOUT_SECONDS = 2 * 60 * 60
RECOVERY_MODE_CHECKPOINT = "checkpoint"
RECOVERY_MODE_INITIAL = "initial"
RECOVERY_MODE_RESTART_FROM_ZERO = "restart_from_zero"
PREREGISTRATION_PATH = Path(__file__).with_name(
    "starcoder_wsd80_gradient_conflict_production_recovery_gate_generation24_20260813.json"
)
EVIDENCE_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_production_recovery_gate_generation24_20260813"
)
GEN19_PREREGISTRATION_SHA256 = "1c3a125f2521116db8a10ebb87a7e52e3bfe6128539af55da1ee9308f45b418f"
GEN19_ANALYZER_REVISION_SHA256 = "2f8b030deeb632fac55e275b996e216243de67a80e84906ff41148fe576b334b"
GEN19_PASS_REPORT_SHA256 = "ae4b6dd5a47321518b40bd558bdfc2972432364d2e316391af945d6d3e402405"
GEN19_REVIEW_SESSION_ID = "f3b38b65-58b1-4657-9ac3-22a2eb0bca99"
GEN19_REVIEW_VERDICT = "ACCEPT_GEN19_SEMANTIC_ERRATUM"


@dataclass(frozen=True)
class FaultInjection:
    task_index: int
    trigger_step: int
    phase: str


@dataclass(frozen=True)
class CheckpointClaim:
    path: str
    step: int
    metadata_sha256: str
    search_paths: tuple[str, ...]


FAULT_INJECTIONS = (
    FaultInjection(task_index=0, trigger_step=2, phase="m100a_at_first_temporary_checkpoint"),
    FaultInjection(task_index=45, trigger_step=2, phase="full_at_first_temporary_checkpoint"),
    FaultInjection(task_index=62, trigger_step=2, phase="m100b_at_first_temporary_checkpoint"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_preregistration(expected_sha256: str) -> dict[str, Any]:
    """Validate the immutable production-recovery design and implementation hashes."""
    observed_sha256 = _sha256(PREREGISTRATION_PATH)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Production-recovery preregistration drifted: {observed_sha256} != {expected_sha256}")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text())
    if preregistration.get("generation") != GENERATION or preregistration.get("stage") != STAGE:
        raise ValueError("Production-recovery generation or stage drifted")
    if preregistration.get("analysis_scope") != "operational_only_no_endpoint_metrics":
        raise ValueError("Production-recovery analysis scope drifted")
    expected_prior_gate = {
        "analyzer_revision_sha256": GEN19_ANALYZER_REVISION_SHA256,
        "pass_report_sha256": GEN19_PASS_REPORT_SHA256,
        "preregistration_sha256": GEN19_PREREGISTRATION_SHA256,
        "review_session_id": GEN19_REVIEW_SESSION_ID,
        "review_verdict": GEN19_REVIEW_VERDICT,
    }
    if preregistration.get("prior_gate") != expected_prior_gate:
        raise ValueError("Production-recovery Gen19 prerequisite drifted")
    expected_faults = [asdict(fault) for fault in FAULT_INJECTIONS]
    if preregistration.get("fault_injections") != expected_faults:
        raise ValueError("Production-recovery fault-injection plan drifted")
    expected_checkpoint_recovery = {
        "automatic_run_local_discovery": True,
        "every_retry_requires_parent_authorization": True,
        "forced_fault_requires_temporary_checkpoint": True,
        "parent_rejects_non_temporary_forced_retry": True,
        "forced_retry_requires_temporary_checkpoint": True,
        "full_state_run_local_resume": True,
        "interval_seconds": int(CHECKPOINT_INTERVAL.total_seconds()),
        "keep_last_temporary_checkpoints": 1,
        "parent_independently_attests_worker_recovery_claim": True,
        "restart_step_tolerance": 0,
        "retry_modes": [RECOVERY_MODE_CHECKPOINT, RECOVERY_MODE_RESTART_FROM_ZERO],
        "stable_output_identity_across_attempts": True,
        "stable_wandb_identity_across_attempts": True,
    }
    if preregistration.get("checkpoint_recovery") != expected_checkpoint_recovery:
        raise ValueError("Production-recovery checkpoint contract drifted")
    expected_admission_barrier = {
        "all_current_attempts_required": STAGE,
        "controller_state_required": "TASK_STATE_RUNNING",
        "immutable_attempt_qualified_initial_release": True,
        "post_release_retries_require_recovery_authorization": True,
        "pretraining": True,
        "release_binds_controller_and_worker_identity": True,
        "timeout_seconds": ADMISSION_TIMEOUT_SECONDS,
        "trainer_initialization_requires_release_consumption": True,
    }
    if preregistration.get("admission_barrier") != expected_admission_barrier:
        raise ValueError("Production-recovery admission barrier drifted")
    expected_integrity = {
        "application_failure_retries": 0,
        "child_preemptible": True,
        "cross_replica_jax": False,
        "iris_topology_coscheduling": False,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "max_preemption_retries_per_task": MAX_PREEMPTION_RETRIES,
        "parent_preemptible": False,
        "parent_retries": 0,
        "placement_region": "us-central1",
        "placement_zone": "us-central1-a",
        "recovery_authorization_timeout_seconds": RECOVERY_AUTHORIZATION_TIMEOUT_SECONDS,
    }
    if preregistration.get("integrity") != expected_integrity:
        raise ValueError("Production-recovery integrity contract drifted")
    expected_gates = {
        "all_rows_reach_terminal_step": True,
        "all_trainer_initializations_follow_admission_release_or_authorized_post_release_retry": True,
        "admission_release_binds_64_live_current_attempts": True,
        "child_preemptions_min": len(FAULT_INJECTIONS),
        "each_forced_row_has_retry_claim": True,
        "each_forced_retry_discovers_nonzero_temporary_checkpoint": True,
        "each_natural_retry_uses_parent_authorized_checkpoint_or_restart_from_zero": True,
        "forced_retry_loads_parent_checkpoint_or_newer": True,
        "fresh_checkpoint_and_wandb_namespaces": True,
        "iris_parent_and_child_succeed": True,
        "no_endpoint_metrics": True,
        "post_forced_recovery_c64_active_overlap_seconds_min": 120.0,
        "post_initialization_state_matches_worker_checkpoint_claim": True,
        "remote_preregistration_hash_matches": True,
        "restart_from_zero_is_allowed_only_before_any_checkpoint_exists": True,
        "wandb_operational_history_covers_emitted_steps_2_through_3327": True,
    }
    if preregistration.get("gates") != expected_gates:
        raise ValueError("Production-recovery acceptance gates drifted")
    frozen_rows = stress.rows_for_stage(STAGE)
    expected_rows = {
        "optimizer_decay_steps": [row.optimizer_decay_step for row in frozen_rows],
        "support_ids": [row.support_id for row in frozen_rows],
        "training_seeds": [row.training_seed for row in frozen_rows],
        "trajectory_ids": [row.trajectory_id for row in frozen_rows],
    }
    if preregistration.get("rows") != expected_rows:
        raise ValueError("Production-recovery row inventory drifted")

    implementation = preregistration.get("implementation_sha256", {})
    sources = {
        "launcher": Path(__file__),
        "stress_launcher": Path(stress.__file__),
        "full_launcher": Path(full.__file__),
        "analyzer": (
            Path(__file__).parent
            / "exploratory/two_phase_many/analyze_starcoder_wsd80_gradient_conflict_production_recovery_gate_20260813.py"
        ),
        "base_launcher": Path(base.__file__),
        "callbacks_core": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/callbacks/_core.py",
        "callbacks_metrics": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/callbacks/_metrics.py",
        "checkpoint": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/checkpoint.py",
        "datasets": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/data/text/datasets.py",
        "dense_launcher": Path(full.dense.__file__),
        "loader": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/data/loader.py",
        "marin_training": Path(__file__).resolve().parents[2] / "lib/marin/src/marin/training/training.py",
        "output_inventory": Path(full.output_inventory.__file__),
        "tests": Path(__file__).resolve().parents[2] / "tests/test_starcoder_wsd80_gradient_conflict_canary.py",
        "recovery_tests": (
            Path(__file__).resolve().parents[2]
            / "tests/test_starcoder_wsd80_gradient_conflict_production_recovery_gate.py"
        ),
        "runtime_gate": (
            Path(__file__).parent
            / "exploratory/two_phase_many/analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811.py"
        ),
        "support_audit": Path(full.support_audit.__file__),
        "tracker": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/tracker/wandb.py",
        "train_lm": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/main/train_lm.py",
        "trainer": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/trainer.py",
    }
    if set(implementation) != set(sources):
        raise ValueError(
            f"Production-recovery implementation inventory drifted: {sorted(implementation)} != {sorted(sources)}"
        )
    for name, path in sources.items():
        observed = _sha256(path)
        if implementation.get(name) != observed:
            raise ValueError(f"Production-recovery dependency {name} drifted: {observed} != {implementation.get(name)}")
    return preregistration


def _publish_preregistration_once(expected_sha256: str) -> None:
    remote_url = f"{EVIDENCE_ROOT}/preregistration.json"
    fs, path = fsspec.core.url_to_fs(remote_url)
    encoded = PREREGISTRATION_PATH.read_bytes()
    try:
        with fs.open(path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        with fs.open(path, "rb") as handle:
            if handle.read() != encoded:
                raise RuntimeError(
                    f"Remote production-recovery preregistration is already claimed: {remote_url}"
                ) from error
    with fs.open(path, "rb") as handle:
        remote_sha256 = hashlib.sha256(handle.read()).hexdigest()
    if remote_sha256 != expected_sha256:
        raise RuntimeError(f"Remote production-recovery preregistration drifted: {remote_sha256} != {expected_sha256}")


def _run_id(row: full.Trajectory) -> str:
    return f"gcfrecoveryg{GENERATION}_{row.trajectory_id}"


def _resumable_config(config: TrainLmOnPodConfig, row: full.Trajectory) -> TrainLmOnPodConfig:
    """Apply stable retry identity and the actual production checkpoint cadence."""
    if config.output_path is None:
        raise ValueError("Production recovery gate requires an output path")
    output_path = stress.attempt_output_path(config.output_path, 0)
    run_id = _run_id(row)
    train_config = cast(TrainLmConfig, config.train_config)
    tracker = train_config.trainer.tracker
    if not isinstance(tracker, WandbConfig):
        raise TypeError("Production recovery gate requires a W&B tracker")
    trainer = replace(
        train_config.trainer,
        id=run_id,
        load_checkpoint=None,
        checkpointer=replace(train_config.trainer.checkpointer, save_interval=CHECKPOINT_INTERVAL),
        tracker=replace(
            tracker,
            id=run_id,
            name=run_id,
            replicate_path=output_path,
            resume="allow",
        ),
    )
    return replace(
        config,
        output_path=output_path,
        train_config=replace(train_config, trainer=trainer),
        env_vars={"RUN_ID": run_id},
    )


def build_configs(
    *,
    marin_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[tuple[full.Trajectory, ...], tuple[TrainLmOnPodConfig, ...]]:
    """Build and audit the 64 frozen production-loader rows."""
    rendezvous_id = stress.stage_rendezvous_id(STAGE, GENERATION)
    rows, steps = stress.build_steps(
        stage=STAGE,
        marin_prefix=marin_prefix,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        rendezvous_id=rendezvous_id,
        generation=GENERATION,
    )
    base_configs = stress.audit_runtime_configs(
        rows,
        steps,
        marin_prefix=marin_prefix,
        stage=STAGE,
        rendezvous_id=rendezvous_id,
        generation=GENERATION,
    )
    configs = tuple(_resumable_config(config, row) for row, config in zip(rows, base_configs, strict=True))
    audit_runtime_configs(rows, configs, marin_prefix=marin_prefix)
    return rows, configs


def audit_runtime_configs(
    rows: tuple[full.Trajectory, ...],
    configs: tuple[TrainLmOnPodConfig, ...],
    *,
    marin_prefix: str,
) -> None:
    """Verify the retry-sensitive runtime contract for every C64 row."""
    if len(rows) != STAGE or len(configs) != STAGE:
        raise ValueError("Production recovery gate must contain exactly 64 rows")
    for row, config in zip(rows, configs, strict=True):
        if config.output_path is None or not config.output_path.endswith("/attempt-000"):
            raise ValueError(f"{row.trajectory_id}: resume output identity drifted")
        if config.resources.preemptible is not True:
            raise ValueError(f"{row.trajectory_id}: production recovery gate must use preemptible TPU resources")
        train_config = cast(TrainLmConfig, config.train_config)
        trainer = train_config.trainer
        if trainer.id != _run_id(row) or config.env_vars != {"RUN_ID": _run_id(row)}:
            raise ValueError(f"{row.trajectory_id}: stable retry run identity drifted")
        if trainer.load_checkpoint is not None or trainer.load_checkpoint_path is not None:
            raise ValueError(f"{row.trajectory_id}: automatic checkpoint discovery drifted")
        if trainer.distributed.initialize_jax_distributed:
            raise ValueError(f"{row.trajectory_id}: recovery gate would join cross-replica JAX")
        if train_config.initial_state_evidence_path is not None:
            raise ValueError(f"{row.trajectory_id}: attempt-specific state evidence leaked into the base config")
        if trainer.checkpointer.save_interval != CHECKPOINT_INTERVAL:
            raise ValueError(f"{row.trajectory_id}: production checkpoint cadence drifted")
        if trainer.checkpointer.keep_last_temporary_checkpoints != 1:
            raise ValueError(f"{row.trajectory_id}: temporary checkpoint retention drifted")
        tracker = trainer.tracker
        if not isinstance(tracker, WandbConfig):
            raise TypeError(f"{row.trajectory_id}: W&B tracker type drifted")
        if tracker.id != _run_id(row) or tracker.name != _run_id(row) or tracker.resume != "allow":
            raise ValueError(f"{row.trajectory_id}: W&B retry identity drifted")
        if tracker.replicate_path != config.output_path:
            raise ValueError(f"{row.trajectory_id}: W&B replicate path drifted")

        runtime_train_config = apply_output_path(train_config, config.output_path)
        search_paths = runtime_train_config.trainer.checkpoint_search_paths(_run_id(row))
        if len(search_paths) != 2:
            raise ValueError(f"{row.trajectory_id}: permanent/temporary checkpoint search contract drifted")
        if not all(path.startswith(marin_prefix) for path in search_paths):
            raise ValueError(f"{row.trajectory_id}: checkpoint search escaped central1 storage")


def _checkpoint_search_paths(config: TrainLmOnPodConfig) -> tuple[str, ...]:
    if config.output_path is None:
        raise ValueError("Production recovery gate requires an output path")
    train_config = apply_output_path(cast(TrainLmConfig, config.train_config), config.output_path)
    run_id = train_config.trainer.id
    if run_id is None:
        raise ValueError("Production recovery gate requires a stable trainer ID")
    return tuple(train_config.trainer.checkpoint_search_paths(run_id))


def _checkpoint_claim(config: TrainLmOnPodConfig) -> CheckpointClaim:
    search_paths = _checkpoint_search_paths(config)
    for url in search_paths:
        fs, path = fsspec.core.url_to_fs(url)
        fs.invalidate_cache(path)
    checkpoint_path = latest_checkpoint_path(*search_paths)
    with fsspec.open(prefix_join(checkpoint_path, "metadata.json"), "rb") as handle:
        payload = handle.read()
    metadata = json.loads(payload)
    step = int(metadata["step"])
    if step <= 0:
        raise RuntimeError(f"Production recovery gate discovered a nonpositive checkpoint step: {step}")
    return CheckpointClaim(
        path=checkpoint_path,
        step=step,
        metadata_sha256=hashlib.sha256(payload).hexdigest(),
        search_paths=search_paths,
    )


def _checkpoint_claim_or_none(config: TrainLmOnPodConfig) -> CheckpointClaim | None:
    """Return the latest full-state checkpoint, or prove that none exists yet."""
    try:
        return _checkpoint_claim(config)
    except FileNotFoundError:
        return None


def _is_temporary_checkpoint(checkpoint: CheckpointClaim) -> bool:
    temporary_root = checkpoint.search_paths[1].rstrip("/")
    return checkpoint.path == temporary_root or checkpoint.path.startswith(f"{temporary_root}/")


def _initial_state_evidence_path(task_index: int, attempt_index: int) -> str:
    return f"{EVIDENCE_ROOT}/state/task-{task_index:03d}/attempt-{attempt_index:03d}.json"


def _recovery_authorization_path(task_index: int, attempt_index: int) -> str:
    return f"{EVIDENCE_ROOT}/authorizations/task-{task_index:03d}/attempt-{attempt_index:03d}.json"


def _admission_release_path(epoch: int) -> str:
    return f"{EVIDENCE_ROOT}/admission/releases/epoch-{epoch:03d}.json"


def _admission_consumed_path(task_index: int, attempt_index: int) -> str:
    return f"{EVIDENCE_ROOT}/admission/consumed/task-{task_index:03d}/attempt-{attempt_index:03d}.json"


def _admission_release_objects() -> tuple[str, ...]:
    fs, root = fsspec.core.url_to_fs(f"{EVIDENCE_ROOT}/admission/releases")
    fs.invalidate_cache(root)
    return tuple(sorted(fs.unstrip_protocol(path) for path in fs.glob(f"{root}/epoch-*.json")))


def _fault_receipt(task_index: int) -> dict[str, Any] | None:
    fs, path = fsspec.core.url_to_fs(f"{EVIDENCE_ROOT}/faults/task-{task_index:03d}.json")
    if not fs.exists(path):
        return None
    with fs.open(path) as handle:
        receipt = json.load(handle)
    if not isinstance(receipt, dict):
        raise TypeError(f"Forced-fault receipt for task {task_index} is not a JSON object")
    return receipt


def _checkpoint_payload(
    checkpoint: CheckpointClaim | None,
    *,
    checkpoint_search_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if checkpoint is None:
        if checkpoint_search_paths is None:
            raise ValueError("A scratch-restart claim requires the audited checkpoint search paths")
        return {
            "checkpoint_metadata_sha256": None,
            "checkpoint_path": None,
            "checkpoint_search_paths": list(checkpoint_search_paths),
            "checkpoint_step": 0,
            "recovery_mode": RECOVERY_MODE_RESTART_FROM_ZERO,
        }
    return {
        "checkpoint_metadata_sha256": checkpoint.metadata_sha256,
        "checkpoint_path": checkpoint.path,
        "checkpoint_search_paths": list(checkpoint.search_paths),
        "checkpoint_step": checkpoint.step,
        "recovery_mode": RECOVERY_MODE_CHECKPOINT,
    }


def _write_attempt_evidence(
    *,
    task_index: int,
    row: full.Trajectory,
    attempt: TaskAttempt,
    checkpoint: CheckpointClaim | None,
    checkpoint_search_paths: tuple[str, ...],
    initial_state_evidence_path: str,
    ready_at: str,
) -> None:
    info = get_job_info()
    if info is None or info.worker_id is None or info.worker_region is None:
        raise RuntimeError("Production recovery gate requires realized Iris worker identity")
    attempt_index = attempt.require_attempt()
    recovery = _checkpoint_payload(checkpoint, checkpoint_search_paths=checkpoint_search_paths)
    if attempt_index == 0:
        recovery = {
            **recovery,
            "checkpoint_search_paths": list(checkpoint_search_paths),
            "recovery_mode": RECOVERY_MODE_INITIAL,
        }
    payload = {
        **recovery,
        "attempt": attempt_index,
        "generation": GENERATION,
        "initial_state_evidence_path": initial_state_evidence_path,
        "row_id": row.trajectory_id,
        "task_id": attempt.task_id.to_wire(),
        "task_attempt_id": attempt.to_wire(),
        "task_index": task_index,
        "worker_id": info.worker_id,
        "worker_region": info.worker_region,
        "ready_at": ready_at,
    }
    fs, path = fsspec.core.url_to_fs(f"{EVIDENCE_ROOT}/attempts/task-{task_index:03d}/attempt-{attempt_index:03d}.json")
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, payload)


def _wait_for_recovery_authorization(
    *,
    task_index: int,
    row: full.Trajectory,
    attempt: TaskAttempt,
    checkpoint: CheckpointClaim | None,
    checkpoint_search_paths: tuple[str, ...],
) -> None:
    authorization_url = _recovery_authorization_path(task_index, attempt.require_attempt())
    fs, path = fsspec.core.url_to_fs(authorization_url)
    deadline = time.monotonic() + RECOVERY_AUTHORIZATION_TIMEOUT_SECONDS
    while not fs.exists(path):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for parent recovery authorization at {authorization_url}")
        time.sleep(POLL_SECONDS)
        fs.invalidate_cache(path)
    with fs.open(path) as handle:
        authorization = json.load(handle)
    expected = {
        **_checkpoint_payload(checkpoint, checkpoint_search_paths=checkpoint_search_paths),
        "attempt": attempt.require_attempt(),
        "generation": GENERATION,
        "row_id": row.trajectory_id,
        "task_attempt_id": attempt.to_wire(),
        "task_index": task_index,
    }
    observed = {key: authorization.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"Parent recovery authorization drifted: {observed} != {expected}")


def _wait_for_admission_release(
    *,
    task_index: int,
    row: full.Trajectory,
    attempt: TaskAttempt,
    ready_at: str,
) -> None:
    """Block trainer initialization until a parent release binds this exact attempt."""
    info = get_job_info()
    if info is None or info.worker_id is None or info.worker_region is None:
        raise RuntimeError("Production recovery admission requires realized Iris worker identity")
    deadline = time.monotonic() + ADMISSION_TIMEOUT_SECONDS
    while True:
        for release_url in reversed(_admission_release_objects()):
            fs, path = fsspec.core.url_to_fs(release_url)
            with fs.open(path, "rb") as handle:
                encoded = handle.read()
            release = json.loads(encoded)
            if not isinstance(release, dict):
                raise TypeError(f"Admission release is not a JSON object: {release_url}")
            if int(release.get("generation", -1)) != GENERATION or int(release.get("stage", -1)) != STAGE:
                raise RuntimeError(f"Admission release escaped the frozen generation/stage: {release_url}")
            bindings = release.get("bindings")
            if not isinstance(bindings, list) or len(bindings) != STAGE:
                raise TypeError(f"Admission release does not bind C64: {release_url}")
            task_indexes = sorted(int(item.get("task_index", -1)) for item in bindings)
            if task_indexes != list(range(STAGE)):
                raise RuntimeError(f"Admission release task inventory drifted: {release_url}")
            binding = next(
                (
                    item
                    for item in bindings
                    if int(item.get("task_index", -1)) == task_index and item.get("task_attempt_id") == attempt.to_wire()
                ),
                None,
            )
            expected_binding = {
                "attempt": attempt.require_attempt(),
                "controller_state": "TASK_STATE_RUNNING",
                "ready_at": ready_at,
                "row_id": row.trajectory_id,
                "task_attempt_id": attempt.to_wire(),
                "task_id": attempt.task_id.to_wire(),
                "task_index": task_index,
                "worker_id": info.worker_id,
                "worker_region": info.worker_region,
            }
            admission_mode = "bound_current_attempt"
            if binding is None:
                released_at = datetime.fromisoformat(str(release["released_at"]))
                attempt_ready_at = datetime.fromisoformat(ready_at)
                if attempt.require_attempt() == 0 or attempt_ready_at < released_at:
                    continue
                admission_mode = "post_release_retry"
            else:
                observed_binding = {key: binding.get(key) for key in expected_binding}
                if observed_binding != expected_binding:
                    raise RuntimeError(f"Admission binding drifted: {observed_binding} != {expected_binding}")
            consumed_identity = {
                **expected_binding,
                "controller_state": "TASK_STATE_RUNNING" if admission_mode == "bound_current_attempt" else None,
            }
            consumed = {
                **consumed_identity,
                "admission_mode": admission_mode,
                "consumed_at": datetime.now(UTC).isoformat(),
                "generation": GENERATION,
                "release_epoch": int(release["epoch"]),
                "release_path": release_url,
                "release_sha256": hashlib.sha256(encoded).hexdigest(),
                "stage": STAGE,
            }
            consumed_fs, consumed_path = fsspec.core.url_to_fs(
                _admission_consumed_path(task_index, attempt.require_attempt())
            )
            consumed_fs.makedirs(str(Path(consumed_path).parent), exist_ok=True)
            stress._write_json_once(consumed_fs, consumed_path, consumed)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for an admission release for {attempt.to_wire()}")
        time.sleep(POLL_SECONDS)


def _run_row(configs: tuple[TrainLmOnPodConfig, ...], rows: tuple[full.Trajectory, ...]) -> None:
    attempt = stress._current_task_attempt()
    _, task_index = attempt.task_id.require_task()
    if task_index >= len(configs):
        raise ValueError(f"Production-recovery task index {task_index} exceeds {len(configs)} rows")
    attempt_index = attempt.require_attempt()
    checkpoint_search_paths = _checkpoint_search_paths(configs[task_index])
    checkpoint = _checkpoint_claim_or_none(configs[task_index]) if attempt_index > 0 else None
    ready_at = datetime.now(UTC).isoformat()
    initial_state_evidence_path = _initial_state_evidence_path(task_index, attempt_index)
    train_config = cast(TrainLmConfig, configs[task_index].train_config)
    config = replace(
        configs[task_index],
        train_config=replace(train_config, initial_state_evidence_path=initial_state_evidence_path),
    )
    _write_attempt_evidence(
        task_index=task_index,
        row=rows[task_index],
        attempt=attempt,
        checkpoint=checkpoint,
        checkpoint_search_paths=checkpoint_search_paths,
        initial_state_evidence_path=initial_state_evidence_path,
        ready_at=ready_at,
    )
    if attempt_index > 0:
        _wait_for_recovery_authorization(
            task_index=task_index,
            row=rows[task_index],
            attempt=attempt,
            checkpoint=checkpoint,
            checkpoint_search_paths=checkpoint_search_paths,
        )
    _wait_for_admission_release(
        task_index=task_index,
        row=rows[task_index],
        attempt=attempt,
        ready_at=ready_at,
    )
    run_levanter_train_lm(config)


class WandbProgress:
    """Read only the operational training step used to trigger faults."""

    def __init__(self) -> None:
        self._api = wandb.Api(timeout=60)

    def global_step(self, run_id: str) -> int | None:
        self._api.flush()
        try:
            run = self._api.run(f"marin-community/marin/{run_id}")
        except wandb.errors.CommError:
            return None
        step = run.summary.get("global_step")
        return None if step is None else int(step)


def _write_fault_receipt(
    fault: FaultInjection,
    *,
    checkpoint: CheckpointClaim,
    observed_step: int,
    source_attempt: int,
    task_attempt_id: str,
) -> None:
    payload = {
        **asdict(fault),
        **_checkpoint_payload(checkpoint),
        "generation": GENERATION,
        "injected_at": datetime.now(UTC).isoformat(),
        "observed_global_step": observed_step,
        "requested_state": "preempted",
        "source_attempt": source_attempt,
        "task_attempt_id": task_attempt_id,
    }
    fs, path = fsspec.core.url_to_fs(f"{EVIDENCE_ROOT}/faults/task-{fault.task_index:03d}.json")
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, payload)


def _all_attempt_records(task_count: int) -> dict[int, tuple[dict[str, Any], ...]]:
    fs, root = fsspec.core.url_to_fs(EVIDENCE_ROOT)
    attempts_root = f"{root}/attempts"
    fs.invalidate_cache(attempts_root)
    records: dict[int, dict[int, dict[str, Any]]] = {index: {} for index in range(task_count)}
    for path in fs.glob(f"{attempts_root}/**/*.json"):
        with fs.open(path) as handle:
            evidence = json.load(handle)
        task_index = int(evidence["task_index"])
        if task_index not in records:
            raise RuntimeError(f"Unexpected attempt evidence task index {task_index}")
        attempt_index = int(evidence["attempt"])
        if attempt_index in records[task_index]:
            raise RuntimeError(f"Duplicate attempt evidence for task {task_index} attempt {attempt_index}")
        records[task_index][attempt_index] = evidence
    return {
        task_index: tuple(by_attempt[index] for index in sorted(by_attempt))
        for task_index, by_attempt in records.items()
    }


def _current_admission_bindings(
    client: Any,
    job: Any,
    rows: tuple[full.Trajectory, ...],
    records_by_task: dict[int, tuple[dict[str, Any], ...]],
) -> tuple[dict[str, Any], ...] | None:
    """Return exact worker/controller bindings only when all C64 attempts are live."""
    statuses_by_task: dict[int, Any] = {}
    for status in client.list_tasks(job.job_id):
        task_id = JobName.from_wire(status.task_id)
        _, task_index = task_id.require_task()
        if task_index in statuses_by_task:
            raise RuntimeError(f"Duplicate Iris task status for admission task {task_index}")
        statuses_by_task[task_index] = status
    if set(statuses_by_task) != set(range(len(rows))):
        return None

    bindings: list[dict[str, Any]] = []
    for task_index, row in enumerate(rows):
        status = statuses_by_task[task_index]
        if status.state != job_pb2.TASK_STATE_RUNNING or not status.worker_id:
            return None
        attempt_index = int(status.current_attempt_id)
        attempt = TaskAttempt(task_id=job.job_id.task(task_index), attempt_id=attempt_index)
        matching = tuple(
            record for record in records_by_task[task_index] if int(record.get("attempt", -1)) == attempt_index
        )
        if len(matching) != 1:
            return None
        evidence = matching[0]
        expected_identity = {
            "attempt": attempt_index,
            "row_id": row.trajectory_id,
            "task_attempt_id": attempt.to_wire(),
            "task_id": attempt.task_id.to_wire(),
            "task_index": task_index,
            "worker_id": status.worker_id,
            "worker_region": "us-central1",
        }
        observed_identity = {key: evidence.get(key) for key in expected_identity}
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"Admission identity drift for task {task_index}: {observed_identity} != {expected_identity}"
            )
        if not evidence.get("ready_at"):
            raise RuntimeError(f"Admission attempt {attempt.to_wire()} lacks a ready timestamp")
        if attempt_index > 0:
            authorization_fs, authorization_path = fsspec.core.url_to_fs(
                _recovery_authorization_path(task_index, attempt_index)
            )
            authorization_fs.invalidate_cache(authorization_path)
            if not authorization_fs.exists(authorization_path):
                return None
        bindings.append(
            {
                **expected_identity,
                "controller_state": job_pb2.TaskState.Name(status.state),
                "ready_at": evidence["ready_at"],
            }
        )
    return tuple(bindings)


def _release_current_admission_epoch(
    client: Any,
    job: Any,
    rows: tuple[full.Trajectory, ...],
    records_by_task: dict[int, tuple[dict[str, Any], ...]],
) -> dict[str, Any] | None:
    """Release one immutable initial epoch when all current attempts are live and identified."""
    release_objects = _admission_release_objects()
    if release_objects:
        with fsspec.open(release_objects[0]) as handle:
            release = json.load(handle)
        if not isinstance(release, dict):
            raise TypeError(f"Admission release is not a JSON object: {release_objects[0]}")
        return release
    bindings = _current_admission_bindings(client, job, rows, records_by_task)
    if bindings is None:
        return None

    parent_info = get_job_info()
    if parent_info is None or parent_info.worker_id is None or parent_info.worker_region is None:
        raise RuntimeError("Production recovery admission release requires realized parent worker identity")
    epoch = 0
    payload = {
        "bindings": list(bindings),
        "epoch": epoch,
        "generation": GENERATION,
        "parent_worker_id": parent_info.worker_id,
        "parent_worker_region": parent_info.worker_region,
        "released_at": datetime.now(UTC).isoformat(),
        "stage": STAGE,
    }
    fs, path = fsspec.core.url_to_fs(_admission_release_path(epoch))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, payload)
    logger.info("Released admission epoch %d for %d live current attempts", epoch, len(bindings))
    return payload


def _authorize_retry_attempts(
    job: Any,
    rows: tuple[full.Trajectory, ...],
    configs: tuple[TrainLmOnPodConfig, ...],
    records_by_task: dict[int, tuple[dict[str, Any], ...]],
) -> None:
    """Independently attest each retry checkpoint before the worker may initialize training."""
    parent_info = get_job_info()
    if parent_info is None or parent_info.worker_id is None or parent_info.worker_region is None:
        raise RuntimeError("Production recovery authorization requires realized parent worker identity")
    for task_index, (row, config) in enumerate(zip(rows, configs, strict=True)):
        for attempt_evidence in records_by_task[task_index]:
            attempt_index = int(attempt_evidence["attempt"])
            if attempt_index == 0:
                continue
            authorization_url = _recovery_authorization_path(task_index, attempt_index)
            fs, path = fsspec.core.url_to_fs(authorization_url)
            if fs.exists(path):
                continue
            task_attempt = TaskAttempt(task_id=job.job_id.task(task_index), attempt_id=attempt_index)
            if attempt_evidence.get("task_attempt_id") != task_attempt.to_wire():
                raise RuntimeError(
                    f"Task {task_index} attempt {attempt_index} identity drifted: "
                    f"{attempt_evidence.get('task_attempt_id')} != {task_attempt.to_wire()}"
                )
            checkpoint = _checkpoint_claim_or_none(config)
            expected_checkpoint = _checkpoint_payload(
                checkpoint,
                checkpoint_search_paths=_checkpoint_search_paths(config),
            )
            observed_checkpoint = {key: attempt_evidence.get(key) for key in expected_checkpoint}
            if observed_checkpoint != expected_checkpoint:
                raise RuntimeError(
                    f"Task {task_index} attempt {attempt_index} did not claim the parent-observed latest checkpoint: "
                    f"{observed_checkpoint} != {expected_checkpoint}"
                )
            fault_receipt = _fault_receipt(task_index)
            if fault_receipt is not None:
                source_attempt = int(fault_receipt.get("source_attempt", -1))
                post_fault_attempts = tuple(
                    int(record["attempt"])
                    for record in records_by_task[task_index]
                    if int(record["attempt"]) > source_attempt
                )
                if post_fault_attempts and attempt_index == min(post_fault_attempts):
                    if checkpoint is None or not _is_temporary_checkpoint(checkpoint):
                        raise RuntimeError(
                            f"Task {task_index} first post-fault retry {attempt_index} did not restore a "
                            "temporary checkpoint"
                        )
            payload = {
                **expected_checkpoint,
                "attempt": attempt_index,
                "authorized_at": datetime.now(UTC).isoformat(),
                "generation": GENERATION,
                "parent_worker_id": parent_info.worker_id,
                "parent_worker_region": parent_info.worker_region,
                "row_id": row.trajectory_id,
                "task_attempt_id": task_attempt.to_wire(),
                "task_index": task_index,
            }
            fs.makedirs(str(Path(path).parent), exist_ok=True)
            stress._write_json_once(fs, path, payload)
            logger.info(
                "Authorized %s with recovery mode %s at checkpoint step %d",
                task_attempt.to_wire(),
                expected_checkpoint["recovery_mode"],
                expected_checkpoint["checkpoint_step"],
            )


def _supervise_gate(
    job: Any,
    rows: tuple[full.Trajectory, ...],
    configs: tuple[TrainLmOnPodConfig, ...],
) -> None:
    pending = {fault.task_index: fault for fault in FAULT_INJECTIONS}
    context = iris_ctx()
    if context.client is None:
        raise RuntimeError("Production-recovery fault injection requires an in-cluster Iris client")
    progress = WandbProgress()
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while True:
        records_by_task = _all_attempt_records(len(rows))
        _authorize_retry_attempts(job, rows, configs, records_by_task)
        _release_current_admission_epoch(context.client, job, rows, records_by_task)
        state = job.state_only()
        if state == job_pb2.JOB_STATE_SUCCEEDED:
            if pending:
                raise RuntimeError(f"Production-recovery child succeeded before faults {sorted(pending)} were injected")
            status = job.wait(timeout=60, raise_on_failure=False)
            if status.state != job_pb2.JOB_STATE_SUCCEEDED:
                raise RuntimeError(f"Production-recovery child changed state after success: {status.state}")
            return
        if state in {
            job_pb2.JOB_STATE_FAILED,
            job_pb2.JOB_STATE_KILLED,
            job_pb2.JOB_STATE_UNSCHEDULABLE,
            job_pb2.JOB_STATE_WORKER_FAILED,
        }:
            raise RuntimeError(
                f"Production-recovery child became terminal before every fault was injected: "
                f"state={job_pb2.JobState.Name(state)}, pending={sorted(pending)}"
            )
        if time.monotonic() >= deadline:
            job.terminate()
            raise TimeoutError(f"Production-recovery fault injection timed out with pending tasks {sorted(pending)}")

        for task_index, fault in tuple(pending.items()):
            observed_step = progress.global_step(_run_id(rows[task_index]))
            if observed_step is None or observed_step < fault.trigger_step:
                continue
            if observed_step >= stress.TERMINAL_STEP:
                raise RuntimeError(f"Task {task_index} reached terminal step before its fault could be injected")
            try:
                checkpoint = _checkpoint_claim(configs[task_index])
            except FileNotFoundError:
                logger.info("Task %d reached step %d but has no complete checkpoint yet", task_index, observed_step)
                continue
            if not _is_temporary_checkpoint(checkpoint):
                logger.info(
                    "Task %d reached step %d but latest checkpoint %s is not temporary yet",
                    task_index,
                    observed_step,
                    checkpoint.path,
                )
                continue
            attempt_records = records_by_task[task_index]
            if not attempt_records:
                raise RuntimeError(f"Task {task_index} reached its fault trigger without attempt evidence")
            source_attempt = int(attempt_records[-1]["attempt"])
            task_attempt_id = TaskAttempt(
                task_id=job.job_id.task(task_index),
                attempt_id=source_attempt,
            ).to_wire()
            if attempt_records[-1].get("task_attempt_id") != task_attempt_id:
                raise RuntimeError(f"Task {task_index} latest attempt identity drifted before fault injection")
            results = context.client.kick_tasks(
                [task_attempt_id],
                desired_state=job_pb2.TASK_STATE_PREEMPTED,
                reason=f"preregistered generation-{GENERATION} {fault.phase} resume fault",
            )
            if len(results) != 1 or not results[0].queued:
                logger.info("Attempt-qualified fault was not queued for %s; retrying current attempt", task_attempt_id)
                continue
            _write_fault_receipt(
                fault,
                checkpoint=checkpoint,
                observed_step=observed_step,
                source_attempt=source_attempt,
                task_attempt_id=task_attempt_id,
            )
            del pending[task_index]
            logger.info("Injected preregistered fault into %s at global step %d", task_attempt_id, observed_step)
        time.sleep(POLL_SECONDS)


def _submit_gate(configs: tuple[TrainLmOnPodConfig, ...], rows: tuple[full.Trajectory, ...]) -> str:
    resources = configs[0].resources
    if any(config.resources != resources for config in configs[1:]):
        raise ValueError("Production-recovery replicas must request identical resources")
    context = iris_ctx()
    if context.client is None or context.job_id is None:
        raise RuntimeError("Production-recovery submission requires an in-cluster Iris client")
    name = f"stage-c64-production-recovery-g{GENERATION}"
    job = context.client.submit(
        entrypoint=Entrypoint.from_callable(_run_row, configs, rows),
        name=name,
        resources=convert_resources(resources),
        constraints=[*convert_constraints(resources), preemptible_constraint(True)],
        coscheduling=None,
        replicas=STAGE,
        max_retries_failure=0,
        max_retries_preemption=MAX_PREEMPTION_RETRIES,
        max_task_failures=0,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
    )
    try:
        _supervise_gate(job, rows, configs)
    except BaseException:
        job.terminate()
        raise
    return job.job_id.to_wire()


def _validate_fresh_state(configs: tuple[TrainLmOnPodConfig, ...]) -> None:
    """Reject stale evidence, checkpoints, or W&B runs before allocating a generation."""
    evidence_fs, evidence_path = fsspec.core.url_to_fs(EVIDENCE_ROOT)
    if evidence_fs.exists(evidence_path) and evidence_fs.find(evidence_path, withdirs=False):
        raise ValueError(f"Production-recovery evidence namespace is not empty: {EVIDENCE_ROOT}")

    occupied_outputs: list[str] = []
    checked_paths: set[str] = set()
    for config in configs:
        if config.output_path is None:
            raise ValueError("Production recovery gate requires an output path")
        train_config = apply_output_path(cast(TrainLmConfig, config.train_config), config.output_path)
        run_id = train_config.trainer.id
        if run_id is None:
            raise ValueError("Production recovery gate requires a stable trainer ID")
        for url in (config.output_path, *train_config.trainer.checkpoint_search_paths(run_id)):
            if url in checked_paths:
                continue
            checked_paths.add(url)
            fs, path = fsspec.core.url_to_fs(url)
            if fs.exists(path):
                occupied_outputs.extend(fs.find(path, withdirs=False))
    if occupied_outputs:
        raise ValueError(f"Production-recovery checkpoint namespace is not empty: {tuple(occupied_outputs[:20])}")

    run_id_set: set[str] = set()
    for config in configs:
        run_id = cast(TrainLmConfig, config.train_config).trainer.id
        if run_id is None:
            raise ValueError("Production recovery gate requires a stable W&B identity")
        run_id_set.add(run_id)
    run_ids = sorted(run_id_set)
    api = wandb.Api(timeout=60)
    api.flush()
    existing_runs = list(
        api.runs(
            "marin-community/marin",
            filters={"name": {"$in": run_ids}},
            per_page=max(len(run_ids), 1),
        )
    )
    if existing_runs:
        raise ValueError(
            f"Production-recovery W&B identities already exist: {tuple(sorted(run.id for run in existing_runs))}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--audit-runtime-configs", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--audit-fresh-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping WSD80 gradient-conflict production recovery gate in CI")
        return
    if (
        args.marin_prefix != base.DEFAULT_MARIN_PREFIX
        or args.tpu_region != base.DEFAULT_TPU_REGION
        or args.tpu_zone != base.DEFAULT_TPU_ZONE
        or args.tpu_type != base.DEFAULT_TPU_TYPE
    ):
        raise ValueError("Historical StarCoder production recovery gate must retain its central1 runtime contract")

    validate_preregistration(args.preregistration_sha256)
    rows, configs = build_configs(
        marin_prefix=args.marin_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.audit_runtime_configs:
        return
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    full.dense._validate_runtime_scientific_environment()
    full.audit_sources(args.marin_prefix, rows)
    if args.audit_source:
        return
    _validate_fresh_state(configs)
    if args.audit_fresh_state:
        return
    stress._validate_parent_runtime_contract()
    _publish_preregistration_once(args.preregistration_sha256)
    child_job = _submit_gate(configs, rows)
    logger.info("Production-cadence C64 recovery gate completed: %s", child_job)


if __name__ == "__main__":
    main()
