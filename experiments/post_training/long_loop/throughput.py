# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a pinned qualification or load workload through Marin's native vLLM serving path."""

import argparse
import asyncio
import json
import logging
import socket
import time
from dataclasses import asdict, replace

import httpx
from iris.cluster.client.job_info import get_job_info
from marin.inference.backend import ModelSpec
from marin.inference.config import ServingGeometry, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.load_test import LoadConfig, LoadRequest, measure_request, request_digest, run_load
from marin.inference.load_workloads import synthetic_requests
from marin.inference.vllm_backend import VllmBackend
from marin.inference.vllm_server import VllmEnvironment
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.context_cohort import run_cohort
from experiments.post_training.long_loop.serving import checked_serving_release

logger = logging.getLogger(__name__)
QUALIFICATION_REQUESTS = 4


def write_json(root: str, name: str, value: dict) -> None:
    (StoragePath(root) / name).write_text(json.dumps(value, indent=2))


def workload_requests(workload: dict, seed: int) -> list[LoadRequest]:
    if workload["kind"] == "synthetic":
        requests = synthetic_requests(
            seed=seed,
            context_repetitions=workload["context_repetitions"],
            output_tokens=workload["output_tokens"],
            count=256,
            prompt_group_size=workload["prompt_group_size"],
        )
        return [replace(request, body={**request.body, **workload["sampling_params"]}) for request in requests]
    if workload["kind"] != "replay":
        raise ValueError(f"Unknown workload kind: {workload['kind']}")
    payload = (StoragePath(workload["uri"])).read_text()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if request_digest({"requests": rows}) != workload["sha256"]:
        raise ValueError("Replay request inventory changed")
    return [LoadRequest(**row) for row in rows]


async def qualify(base_url: str, model: str) -> dict:
    request = synthetic_requests(seed=17, context_repetitions=1, output_tokens=32, count=1, prompt_group_size=1)[0]
    async with httpx.AsyncClient(timeout=300) as client:
        rows = [
            await measure_request(client, base_url, model, request, seed=17 + index)
            for index in range(QUALIFICATION_REQUESTS)
        ]
    if any(row.error is not None or not row.output_tokens for row in rows):
        raise ValueError(f"Qualification requests failed: {[asdict(row) for row in rows]}")
    return {"requests": [asdict(row) for row in rows]}


def graph_capture_observed(logs: str) -> bool:
    return "Graph capturing finished" in logs or "CUDA graph capturing finished" in logs


def qualify_environment(environment: VllmEnvironment, spec: dict, startup: float) -> None:
    root = spec["output_uri"]
    environment.wait_until_ready()
    ready = time.perf_counter()
    base_url = environment.server_url
    model = spec["model"]["api_model"]
    evidence = asyncio.run(qualify(base_url, model))
    logs = environment.logs()
    (StoragePath(root) / "leader-vllm.log").write_text(logs)
    captured = graph_capture_observed(logs)
    write_json(
        root,
        "qualification.json",
        {
            "source_revision": spec["source_revision"],
            "model": spec["model"],
            "geometry": spec["geometry"],
            "engine": spec["engine"],
            "startup_seconds": ready - startup,
            "cuda_graph_capture_observed": captured,
            "host": socket.gethostname(),
            **evidence,
        },
    )
    if not captured:
        raise ValueError("Loading and requests passed, but CUDA graph capture is not confirmed in the retained log")


def run_sweep(environment: VllmEnvironment, spec: dict) -> None:
    root = spec["output_uri"]
    base_url = environment.server_url
    model = spec["model"]["api_model"]
    for workload in spec["workloads"]:
        for seed in spec["seeds"]:
            requests = workload_requests(workload, seed)
            for concurrency in spec["concurrencies"]:
                config = LoadConfig(
                    concurrency=concurrency,
                    seed=seed,
                    gpu_count=ServingGeometry(**spec["geometry"]).total_gpus,
                    warmup_seconds=spec["warmup_seconds"],
                    measurement_seconds=spec["measurement_seconds"],
                    request_timeout_seconds=spec["request_timeout_seconds"],
                )
                result = asyncio.run(run_load(base_url, environment.metrics_url, model, requests, config))
                name = f"{workload['name']}-seed{seed}-c{concurrency}.json"
                write_json(root, name, {"source_revision": spec["source_revision"], "workload": workload, **result})
                logger.info("Completed %s: valid=%s server=%s", name, result["valid"], result["server"])
                if not result["valid"]:
                    raise ValueError(f"Load run {name} failed; retain the invalid result and stop the sweep")


def run(spec: dict) -> None:
    if spec["mode"] not in {"qualify", "benchmark", "cohort"}:
        raise ValueError(f"Unknown mode: {spec['mode']}")
    geometry = ServingGeometry(**spec["geometry"])
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA, source=VllmSource.MARIN_FORK, release=checked_serving_release(), **spec["engine"]
    )
    model = ModelSpec(**spec["model"])
    backend = VllmBackend(engine)
    info = get_job_info()
    if info is None:
        raise RuntimeError("Run this worker in Iris so placement and source provenance are recorded")
    started = time.perf_counter()
    with iris_vllm_launch(geometry) as launch:
        with backend.start(model, extra_args=launch.extra_cli_args, subprocess_env=launch.subprocess_env) as environment:
            if launch.is_leader:
                with iris_vllm_followers(launch):
                    qualify_environment(environment, spec, started)
                    if spec["mode"] == "benchmark":
                        run_sweep(environment, spec)
                    elif spec["mode"] == "cohort":
                        requests = workload_requests(spec["workloads"][0], 17)
                        asyncio.run(
                            run_cohort(
                                environment.server_url,
                                environment.metrics_url,
                                spec["model"]["api_model"],
                                requests,
                                spec,
                            )
                        )
            else:
                wait_for_iris_vllm_shutdown(launch, environment.check_alive)
            logs = environment.logs()
            (StoragePath(spec["output_uri"]) / f"task-{info.task_index}-vllm.log").write_text(logs)
            captured = graph_capture_observed(logs)
            write_json(
                spec["output_uri"],
                f"task-{info.task_index}-serving.json",
                {"task_id": str(info.task_id), "cuda_graph_capture_observed": captured},
            )
            if not captured:
                raise ValueError(f"CUDA graph capture not confirmed on task {info.task_index}")
        if not launch.is_leader:
            notify_iris_vllm_stopped(launch)
    write_json(
        spec["output_uri"],
        f"task-{info.task_index}-complete.json",
        {"task_id": str(info.task_id), "source_revision": spec["source_revision"], "mode": spec["mode"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    spec = json.loads(StoragePath(args.spec).read_text())
    spec["source_revision"] = args.source_revision
    run(spec)


if __name__ == "__main__":
    main()
