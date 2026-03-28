# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from fray.v2 import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment, current_client, wait_all
from iris.cluster.client import get_job_info
from iris.marin_fs import marin_temp_bucket
from levanter.callbacks import StepInfo
from levanter.elastic import ElasticTrainingConfig, FileBackedPeerSyncController, PeerAveragingSyncConfig
from levanter.utils import fsspec_utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElasticTransferDemoConfig:
    """Configuration for the parent Iris demo job."""

    tpu_type: str = "v5p-8"
    output_root: str | None = None
    timeout_seconds: float = 240.0


@dataclass(frozen=True)
class ElasticTransferWorkerConfig:
    """Configuration for a single TPU worker in the demo."""

    group_id: str
    state_root: str
    timeout_seconds: float = 120.0
    worker_id: str | None = None
    source_worker_id: str = "w001"
    target_worker_id: str = "w000"


@dataclass
class _DummyState:
    step: int
    model: dict[str, Any]
    opt_state: None = None
    model_averaging: None = None


def _single_device_mesh() -> Mesh:
    return Mesh(np.array([jax.devices()[0]]), axis_names=("replica",))


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(plain_path, "w") as f:
        json.dump(payload, f, sort_keys=True)


def _read_json(path: str) -> dict[str, Any] | None:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    if not fs.exists(plain_path):
        return None
    with fs.open(plain_path) as f:
        return json.load(f)


def _wait_for_path(path: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if fsspec_utils.exists(path):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {path}")


def _wait_for_bootstrap(
    controller: FileBackedPeerSyncController,
    *,
    timeout_seconds: float,
) -> tuple[_DummyState, float]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = controller.bootstrap_state(_DummyState(step=0, model={"weight": jnp.array([0.0], dtype=jnp.float32)}))
        weight = float(jax.device_get(state.model["weight"])[0])
        if abs(weight - 2.0) < 1e-6:
            return state, weight
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for elastic bootstrap")


def _wait_for_sync(
    controller: FileBackedPeerSyncController,
    *,
    timeout_seconds: float,
) -> float:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        info = controller.maybe_update_state(
            StepInfo(
                state=_DummyState(step=1, model={"weight": jnp.array([0.0], dtype=jnp.float32)}),
                loss=0.0,
                step_duration=0.0,
            )
        )
        weight = float(jax.device_get(info.state.model["weight"])[0])
        if abs(weight - 1.0) < 1e-6:
            return weight
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for elastic peer sync")


def run_elastic_transfer_worker(config: ElasticTransferWorkerConfig) -> dict[str, Any]:
    """Run a minimal TPU worker that exercises elastic bootstrap and peer sync."""

    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("Elastic transfer worker must run inside an Iris job")

    worker_id = config.worker_id or f"w{job_info.task_index:03d}"
    worker_root = fsspec_utils.join_path(config.state_root, worker_id)
    reports_dir = fsspec_utils.join_path(config.state_root, "reports")
    markers_dir = fsspec_utils.join_path(config.state_root, "markers")
    bootstrap_marker = fsspec_utils.join_path(markers_dir, f"{config.target_worker_id}-bootstrapped.json")
    sync_marker = fsspec_utils.join_path(markers_dir, f"{config.target_worker_id}-synced.json")
    report_path = fsspec_utils.join_path(reports_dir, f"{worker_id}.json")
    traceback_after_seconds = max(30.0, min(config.timeout_seconds, 120.0))

    faulthandler.dump_traceback_later(traceback_after_seconds, repeat=True)
    print(
        json.dumps(
            {
                "event": "elastic_transfer_worker_start",
                "job_id": str(job_info.job_id),
                "task_index": job_info.task_index,
                "worker_id": worker_id,
                "state_root": config.state_root,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    controller = FileBackedPeerSyncController(
        config=ElasticTrainingConfig(
            enabled=True,
            group_id=config.group_id,
            worker_id=worker_id,
            state_path=fsspec_utils.join_path(config.state_root, "elastic"),
            sync_interval_steps=1,
            publish_interval_steps=1,
            sync=PeerAveragingSyncConfig(),
            transport="jax_transfer",
            transfer_timeout=timedelta(seconds=int(config.timeout_seconds)),
            request_poll_interval_seconds=0.05,
        ),
        checkpoint_base_path=fsspec_utils.join_path(worker_root, "checkpoints"),
        run_id=f"{config.group_id}-{worker_id}",
        axis_mapping={},
        mesh=_single_device_mesh(),
    )
    print(
        json.dumps(
            {
                "event": "elastic_transfer_controller_ready",
                "transport_kind": controller.transport_kind,
                "worker_id": worker_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    result: dict[str, Any] = {
        "worker_id": worker_id,
        "task_index": job_info.task_index,
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "transport_kind": controller.transport_kind,
    }

    try:
        if controller.transport_kind != "jax_transfer":
            raise RuntimeError(f"Expected jax_transfer, got {controller.transport_kind}")

        if worker_id == config.source_worker_id:
            print(json.dumps({"event": "publish_bootstrap", "worker_id": worker_id}, sort_keys=True), flush=True)
            controller._publish_state(
                _DummyState(step=1, model={"weight": jnp.array([2.0], dtype=jnp.float32)}),
                step=0,
            )
            _wait_for_path(bootstrap_marker, timeout_seconds=config.timeout_seconds)
            print(json.dumps({"event": "publish_sync", "worker_id": worker_id}, sort_keys=True), flush=True)
            controller._publish_state(
                _DummyState(step=2, model={"weight": jnp.array([2.0], dtype=jnp.float32)}),
                step=1,
            )
            _wait_for_path(sync_marker, timeout_seconds=config.timeout_seconds)
            result["published_steps"] = [0, 1]
        elif worker_id == config.target_worker_id:
            print(json.dumps({"event": "wait_for_bootstrap", "worker_id": worker_id}, sort_keys=True), flush=True)
            _, bootstrapped_weight = _wait_for_bootstrap(controller, timeout_seconds=config.timeout_seconds)
            _write_json(
                bootstrap_marker,
                {
                    "bootstrapped_weight": bootstrapped_weight,
                    "worker_id": worker_id,
                },
            )
            print(json.dumps({"event": "wait_for_sync", "worker_id": worker_id}, sort_keys=True), flush=True)
            synced_weight = _wait_for_sync(controller, timeout_seconds=config.timeout_seconds)
            _write_json(
                sync_marker,
                {
                    "synced_weight": synced_weight,
                    "worker_id": worker_id,
                },
            )
            result["bootstrapped_weight"] = bootstrapped_weight
            result["synced_weight"] = synced_weight
        else:
            raise ValueError(
                f"Unexpected worker_id {worker_id}; expected {config.target_worker_id} or {config.source_worker_id}"
            )

        _write_json(report_path, result)
        logger.info("Elastic transfer demo worker result: %s", result)
        print(json.dumps({"event": "elastic_transfer_worker_done", **result}, sort_keys=True), flush=True)
        return result
    finally:
        controller.close()
        faulthandler.cancel_dump_traceback_later()


def run_elastic_transfer_demo(config: ElasticTransferDemoConfig) -> dict[str, Any]:
    """Launch two single-slice TPU jobs that exercise JAX TransferServer sync."""

    client = current_client()
    group_id = f"elastic-transfer-demo-{uuid.uuid4().hex[:8]}"
    state_root = config.output_root or marin_temp_bucket(ttl_days=7, prefix=f"elastic-transfer-demo/{group_id}")
    worker_config = ElasticTransferWorkerConfig(
        group_id=group_id,
        state_root=state_root,
        timeout_seconds=config.timeout_seconds,
    )
    worker_resources = ResourceConfig.with_tpu(config.tpu_type)
    worker_env = create_environment(
        extras=["tpu"],
        env_vars={
            "WANDB_MODE": "disabled",
            "JAX_TRACEBACK_FILTERING": "off",
        },
    )

    jobs = []
    for worker_name in (worker_config.target_worker_id, worker_config.source_worker_id):
        worker_specific_config = replace(worker_config, worker_id=worker_name)
        jobs.append(
            client.submit(
                JobRequest(
                    name=f"elastic-transfer-demo-{group_id}-{worker_name}",
                    entrypoint=Entrypoint.from_callable(run_elastic_transfer_worker, args=[worker_specific_config]),
                    resources=worker_resources,
                    environment=worker_env,
                    max_retries_failure=0,
                    max_retries_preemption=2,
                )
            )
        )

    statuses = wait_all(jobs, raise_on_failure=False)
    summary: dict[str, Any] = {
        "group_id": group_id,
        "state_root": state_root,
        "statuses": [status.value for status in statuses],
        "jobs": [str(job.job_id) for job in jobs],
    }
    logger.info("Launched elastic transfer demo: %s", summary)
    print(json.dumps({"event": "elastic_transfer_demo_launched", **summary}, sort_keys=True), flush=True)

    report_paths = {
        worker_id: fsspec_utils.join_path(fsspec_utils.join_path(state_root, "reports"), f"{worker_id}.json")
        for worker_id in (worker_config.target_worker_id, worker_config.source_worker_id)
    }
    reports = {worker_id: _read_json(path) for worker_id, path in report_paths.items()}
    summary["reports"] = reports

    if any(status != JobStatus.SUCCEEDED for status in statuses):
        raise RuntimeError(f"Elastic transfer demo jobs did not all succeed: {summary}")

    for worker_id, report in reports.items():
        if report is None:
            raise RuntimeError(f"Missing demo report for {worker_id}: {summary}")
        if report.get("transport_kind") != "jax_transfer":
            raise RuntimeError(f"Worker {worker_id} did not use jax_transfer: {summary}")

    target_report = reports[worker_config.target_worker_id]
    assert target_report is not None
    if abs(float(target_report["bootstrapped_weight"]) - 2.0) > 1e-6:
        raise RuntimeError(f"Unexpected bootstrap weight: {summary}")
    if abs(float(target_report["synced_weight"]) - 1.0) > 1e-6:
        raise RuntimeError(f"Unexpected synced weight: {summary}")

    logger.info("Elastic transfer demo succeeded: %s", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parse_args() -> ElasticTransferDemoConfig:
    parser = argparse.ArgumentParser(description="Launch a two-worker elastic JAX transfer demo on Iris.")
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()
    return ElasticTransferDemoConfig(
        tpu_type=args.tpu_type,
        output_root=args.output_root,
        timeout_seconds=args.timeout_seconds,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run_elastic_transfer_demo(_parse_args())


if __name__ == "__main__":
    main()
