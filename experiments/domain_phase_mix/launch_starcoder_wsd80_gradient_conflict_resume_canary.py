# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise production-style per-task checkpoint recovery for the WSD80 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import fsspec
import wandb
from fray.iris_backend import convert_constraints, convert_resources
from iris.client import Job, JobAlreadyExists
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import preemptible_constraint
from iris.cluster.types import Entrypoint, TaskAttempt
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

GENERATION = 19
STAGE = 6
CHECKPOINT_INTERVAL = timedelta(minutes=2)
MAX_PREEMPTION_RETRIES = 100
JOB_TIMEOUT_SECONDS = 2 * 60 * 60
POLL_SECONDS = 5.0
PREREGISTRATION_PATH = Path(__file__).with_name(
    "starcoder_wsd80_gradient_conflict_resume_canary_preregistration_generation19_20260813.json"
)
EVIDENCE_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_resume_canary_generation19_20260813"
)


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


FAULT_INJECTIONS = (
    FaultInjection(task_index=0, trigger_step=1_024, phase="before_data_switch"),
    FaultInjection(task_index=1, trigger_step=2_048, phase="after_data_switch"),
    FaultInjection(task_index=2, trigger_step=2_944, phase="after_optimizer_decay"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_preregistration(expected_sha256: str) -> dict[str, Any]:
    """Validate the immutable canary design and implementation hashes."""
    observed_sha256 = _sha256(PREREGISTRATION_PATH)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Resume-canary preregistration drifted: {observed_sha256} != {expected_sha256}")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text())
    if preregistration.get("generation") != GENERATION or preregistration.get("stage") != STAGE:
        raise ValueError("Resume-canary generation or stage drifted")
    if preregistration.get("analysis_scope") != "operational_only_no_endpoint_metrics":
        raise ValueError("Resume-canary analysis scope drifted")
    expected_faults = [asdict(fault) for fault in FAULT_INJECTIONS]
    if preregistration.get("fault_injections") != expected_faults:
        raise ValueError("Resume-canary fault-injection plan drifted")

    implementation = preregistration.get("implementation_sha256", {})
    sources = {
        "launcher": Path(__file__),
        "stress_launcher": Path(stress.__file__),
        "full_launcher": Path(full.__file__),
        "analyzer": (
            Path(__file__).parent
            / "exploratory/two_phase_many/analyze_starcoder_wsd80_gradient_conflict_resume_canary_20260813.py"
        ),
        "checkpoint": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/checkpoint.py",
        "datasets": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/data/text/datasets.py",
        "loader": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/data/loader.py",
        "tests": Path(__file__).resolve().parents[2] / "tests/test_starcoder_wsd80_gradient_conflict_canary.py",
        "tracker": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/tracker/wandb.py",
        "train_lm": Path(__file__).resolve().parents[2] / "lib/levanter/src/levanter/main/train_lm.py",
    }
    for name, path in sources.items():
        observed = _sha256(path)
        if implementation.get(name) != observed:
            raise ValueError(f"Resume-canary dependency {name} drifted: {observed} != {implementation.get(name)}")
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
                raise RuntimeError(f"Remote resume-canary preregistration is already claimed: {remote_url}") from error
    with fs.open(path, "rb") as handle:
        remote_sha256 = hashlib.sha256(handle.read()).hexdigest()
    if remote_sha256 != expected_sha256:
        raise RuntimeError(f"Remote resume-canary preregistration drifted: {remote_sha256} != {expected_sha256}")


def _run_id(row: full.Trajectory) -> str:
    return f"gcfresumeg{GENERATION}_{row.trajectory_id}"


def _resumable_config(config: TrainLmOnPodConfig, row: full.Trajectory) -> TrainLmOnPodConfig:
    """Apply the production retry identity and a short mechanism-test checkpoint cadence."""
    if config.output_path is None:
        raise ValueError("Resume canary requires an output path")
    output_path = stress.attempt_output_path(config.output_path, 0)
    run_id = _run_id(row)
    train_config = cast(TrainLmConfig, config.train_config)
    tracker = train_config.trainer.tracker
    if not isinstance(tracker, WandbConfig):
        raise TypeError("Resume canary requires a W&B tracker")
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
    """Build and audit the six frozen production-loader rows."""
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
    """Verify the retry-sensitive runtime contract for every canary row."""
    if len(rows) != STAGE or len(configs) != STAGE:
        raise ValueError("Resume canary must contain exactly six rows")
    for row, config in zip(rows, configs, strict=True):
        if config.output_path is None or not config.output_path.endswith("/attempt-000"):
            raise ValueError(f"{row.trajectory_id}: resume output identity drifted")
        if config.resources.preemptible is not True:
            raise ValueError(f"{row.trajectory_id}: resume canary must use preemptible TPU resources")
        train_config = cast(TrainLmConfig, config.train_config)
        trainer = train_config.trainer
        if trainer.id != _run_id(row) or config.env_vars != {"RUN_ID": _run_id(row)}:
            raise ValueError(f"{row.trajectory_id}: stable retry run identity drifted")
        if trainer.load_checkpoint is not None or trainer.load_checkpoint_path is not None:
            raise ValueError(f"{row.trajectory_id}: automatic checkpoint discovery drifted")
        if trainer.distributed.initialize_jax_distributed:
            raise ValueError(f"{row.trajectory_id}: resume canary would join cross-replica JAX")
        if train_config.initial_state_evidence_path is not None:
            raise ValueError(f"{row.trajectory_id}: attempt-specific state evidence leaked into the base config")
        if trainer.checkpointer.save_interval != CHECKPOINT_INTERVAL:
            raise ValueError(f"{row.trajectory_id}: mechanism-test checkpoint cadence drifted")
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


def _checkpoint_claim(config: TrainLmOnPodConfig) -> CheckpointClaim:
    if config.output_path is None:
        raise ValueError("Resume canary requires an output path")
    train_config = apply_output_path(cast(TrainLmConfig, config.train_config), config.output_path)
    run_id = train_config.trainer.id
    if run_id is None:
        raise ValueError("Resume canary requires a stable trainer ID")
    checkpoint_path = latest_checkpoint_path(*train_config.trainer.checkpoint_search_paths(run_id))
    with fsspec.open(prefix_join(checkpoint_path, "metadata.json"), "rb") as handle:
        payload = handle.read()
    metadata = json.loads(payload)
    step = int(metadata["step"])
    if step <= 0:
        raise RuntimeError(f"Resume canary discovered a nonpositive checkpoint step: {step}")
    return CheckpointClaim(
        path=checkpoint_path,
        step=step,
        metadata_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _optional_checkpoint_claim(config: TrainLmOnPodConfig) -> CheckpointClaim | None:
    try:
        return _checkpoint_claim(config)
    except FileNotFoundError:
        return None


def _initial_state_evidence_path(task_index: int, attempt_index: int) -> str:
    return f"{EVIDENCE_ROOT}/state/task-{task_index:03d}/attempt-{attempt_index:03d}.json"


def _write_attempt_evidence(
    *,
    task_index: int,
    row: full.Trajectory,
    attempt: TaskAttempt,
    checkpoint: CheckpointClaim | None,
    initial_state_evidence_path: str,
) -> None:
    info = get_job_info()
    if info is None or info.worker_id is None or info.worker_region is None:
        raise RuntimeError("Resume canary requires realized Iris worker identity")
    attempt_index = attempt.require_attempt()
    payload = {
        "attempt": attempt_index,
        "checkpoint_path": None if checkpoint is None else checkpoint.path,
        "checkpoint_step": 0 if checkpoint is None else checkpoint.step,
        "checkpoint_metadata_sha256": None if checkpoint is None else checkpoint.metadata_sha256,
        "generation": GENERATION,
        "initial_state_evidence_path": initial_state_evidence_path,
        "row_id": row.trajectory_id,
        "task_id": attempt.task_id.to_wire(),
        "task_index": task_index,
        "worker_id": info.worker_id,
        "worker_region": info.worker_region,
    }
    fs, path = fsspec.core.url_to_fs(f"{EVIDENCE_ROOT}/attempts/task-{task_index:03d}/attempt-{attempt_index:03d}.json")
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, payload)


def _run_row(configs: tuple[TrainLmOnPodConfig, ...], rows: tuple[full.Trajectory, ...]) -> None:
    attempt = stress._current_task_attempt()
    _, task_index = attempt.task_id.require_task()
    if task_index >= len(configs):
        raise ValueError(f"Resume-canary task index {task_index} exceeds {len(configs)} rows")
    attempt_index = attempt.require_attempt()
    checkpoint = _optional_checkpoint_claim(configs[task_index]) if attempt_index > 0 else None
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
        initial_state_evidence_path=initial_state_evidence_path,
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
    task_id: str,
) -> None:
    payload = {
        **asdict(fault),
        "checkpoint_metadata_sha256": checkpoint.metadata_sha256,
        "checkpoint_path": checkpoint.path,
        "checkpoint_step": checkpoint.step,
        "generation": GENERATION,
        "injected_at": datetime.now(UTC).isoformat(),
        "observed_global_step": observed_step,
        "requested_state": "preempted",
        "source_attempt": source_attempt,
        "task_id": task_id,
    }
    fs, path = fsspec.core.url_to_fs(f"{EVIDENCE_ROOT}/faults/task-{fault.task_index:03d}.json")
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    stress._write_json_once(fs, path, payload)


def _existing_fault_tasks() -> set[int]:
    fs, root = fsspec.core.url_to_fs(EVIDENCE_ROOT)
    return {
        fault.task_index for fault in FAULT_INJECTIONS if fs.exists(f"{root}/faults/task-{fault.task_index:03d}.json")
    }


def _attempt_indices(task_index: int) -> tuple[int, ...]:
    fs, root = fsspec.core.url_to_fs(EVIDENCE_ROOT)
    indices: list[int] = []
    for path in fs.glob(f"{root}/attempts/task-{task_index:03d}/attempt-*.json"):
        with fs.open(path) as handle:
            evidence = json.load(handle)
        indices.append(int(evidence["attempt"]))
    return tuple(sorted(set(indices)))


def _inject_faults(
    job: Any,
    rows: tuple[full.Trajectory, ...],
    configs: tuple[TrainLmOnPodConfig, ...],
) -> None:
    existing = _existing_fault_tasks()
    pending = {fault.task_index: fault for fault in FAULT_INJECTIONS if fault.task_index not in existing}
    if not pending:
        logger.info("All preregistered resume-canary faults already have immutable receipts")
        return
    context = iris_ctx()
    if context.client is None:
        raise RuntimeError("Resume-canary fault injection requires an in-cluster Iris client")
    progress = WandbProgress()
    if existing:
        logger.info("Resume canary reattached with existing fault receipts for tasks %s", sorted(existing))
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while pending:
        state = job.state_only()
        if state in {
            job_pb2.JOB_STATE_SUCCEEDED,
            job_pb2.JOB_STATE_FAILED,
            job_pb2.JOB_STATE_KILLED,
            job_pb2.JOB_STATE_UNSCHEDULABLE,
            job_pb2.JOB_STATE_WORKER_FAILED,
        }:
            raise RuntimeError(
                f"Resume-canary child became terminal before every fault was injected: "
                f"state={job_pb2.JobState.Name(state)}, pending={sorted(pending)}"
            )
        if time.monotonic() >= deadline:
            job.terminate()
            raise TimeoutError(f"Resume-canary fault injection timed out with pending tasks {sorted(pending)}")

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
            attempt_indices = _attempt_indices(task_index)
            if not attempt_indices:
                raise RuntimeError(f"Task {task_index} reached its fault trigger without attempt evidence")
            task_id = job.job_id.task(task_index).to_wire()
            results = context.client.kick_tasks(
                [task_id],
                desired_state=job_pb2.TASK_STATE_PREEMPTED,
                reason=f"preregistered generation-{GENERATION} {fault.phase} resume fault",
            )
            if len(results) != 1 or not results[0].queued:
                raise RuntimeError(f"Resume-canary fault was not queued for {task_id}: {results}")
            _write_fault_receipt(
                fault,
                checkpoint=checkpoint,
                observed_step=observed_step,
                source_attempt=max(attempt_indices),
                task_id=task_id,
            )
            del pending[task_index]
            logger.info("Injected preregistered fault into %s at global step %d", task_id, observed_step)
        time.sleep(POLL_SECONDS)


def _submit_canary(configs: tuple[TrainLmOnPodConfig, ...], rows: tuple[full.Trajectory, ...]) -> str:
    resources = configs[0].resources
    if any(config.resources != resources for config in configs[1:]):
        raise ValueError("Resume-canary replicas must request identical resources")
    context = iris_ctx()
    if context.client is None or context.job_id is None:
        raise RuntimeError("Resume-canary submission requires an in-cluster Iris client")
    name = f"stage-c06-resume-g{GENERATION}"
    try:
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
    except JobAlreadyExists:
        job = Job(context.client, context.job_id.child(name))
        logger.info("Reattached to existing resume-canary child %s", job.job_id)
    _inject_faults(job, rows, configs)
    status = job.wait(timeout=JOB_TIMEOUT_SECONDS, raise_on_failure=False)
    if status.state != job_pb2.JOB_STATE_SUCCEEDED:
        raise RuntimeError(
            f"Resume-canary child failed: state={job_pb2.JobState.Name(status.state)}, "
            f"failures={status.failure_count}, preemptions={status.preemption_count}"
        )
    return job.job_id.to_wire()


def _validate_fresh_state(configs: tuple[TrainLmOnPodConfig, ...]) -> None:
    """Reject stale evidence, checkpoints, or W&B runs before allocating a generation."""
    evidence_fs, evidence_path = fsspec.core.url_to_fs(EVIDENCE_ROOT)
    if evidence_fs.exists(evidence_path) and evidence_fs.find(evidence_path, withdirs=False):
        raise ValueError(f"Resume-canary evidence namespace is not empty: {EVIDENCE_ROOT}")

    occupied_outputs: list[str] = []
    checked_paths: set[str] = set()
    for config in configs:
        if config.output_path is None:
            raise ValueError("Resume canary requires an output path")
        train_config = apply_output_path(cast(TrainLmConfig, config.train_config), config.output_path)
        run_id = train_config.trainer.id
        if run_id is None:
            raise ValueError("Resume canary requires a stable trainer ID")
        for url in (config.output_path, *train_config.trainer.checkpoint_search_paths(run_id)):
            if url in checked_paths:
                continue
            checked_paths.add(url)
            fs, path = fsspec.core.url_to_fs(url)
            if fs.exists(path):
                occupied_outputs.extend(fs.find(path, withdirs=False))
    if occupied_outputs:
        raise ValueError(f"Resume-canary checkpoint namespace is not empty: {tuple(occupied_outputs[:20])}")

    run_id_set: set[str] = set()
    for config in configs:
        run_id = cast(TrainLmConfig, config.train_config).trainer.id
        if run_id is None:
            raise ValueError("Resume canary requires a stable W&B identity")
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
        raise ValueError(f"Resume-canary W&B identities already exist: {tuple(sorted(run.id for run in existing_runs))}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--audit-runtime-configs", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping WSD80 gradient-conflict resume canary in CI")
        return
    if (
        args.marin_prefix != base.DEFAULT_MARIN_PREFIX
        or args.tpu_region != base.DEFAULT_TPU_REGION
        or args.tpu_zone != base.DEFAULT_TPU_ZONE
        or args.tpu_type != base.DEFAULT_TPU_TYPE
    ):
        raise ValueError("Historical StarCoder resume canary must retain its central1 runtime contract")

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
    stress._validate_parent_runtime_contract()
    _validate_fresh_state(configs)
    _publish_preregistration_once(args.preregistration_sha256)
    child_job = _submit_canary(configs, rows)
    logger.info("Production-style resume canary completed: %s", child_job)


if __name__ == "__main__":
    main()
