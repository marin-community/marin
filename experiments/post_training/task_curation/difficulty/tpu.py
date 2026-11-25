# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional TPU placement for the existing difficulty engine-pool owner."""

import argparse
import asyncio
import contextvars
import json
import logging
import os
import queue
import threading
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
from finestore.rl.difficulty import SOLVER_MODELS, Solver
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import EndpointAccess
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.config import ObjectStoreLoadMode, RemoteInferenceConfig, VllmEngineConfig, VllmLauncherType
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.inference.iris import remote_inference
from marin.inference.model_preparation import resolve_model_path
from marin.inference.types import InferenceBackendState
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.difficulty.campaign import stage_campaign
from experiments.post_training.task_curation.difficulty.ledger import digest
from experiments.post_training.task_curation.difficulty.pool_receipts import publish_pool_receipt, warm_throughput
from experiments.post_training.task_curation.difficulty.regional import immutable_json, regional_worker
from experiments.post_training.task_curation.difficulty.tpu_router import pool_router
from experiments.post_training.task_curation.difficulty_protocol import PINS
from experiments.post_training.task_curation.difficulty_sampling import request_parameters, sampling_settings
from experiments.post_training.task_curation.difficulty_serving_probe import (
    hf_prompt_tokens,
    probe_endpoint,
    probe_serving_endpoint,
)

LOGGER = logging.getLogger(__name__)
REGION = "us-central2"
ZONE = "us-central2-b"
REGIONAL_ROOT = "gs://marin-us-central2/rl-data/atlas/difficulty-tpu-v4"
CHIP_BUDGET = 256
FAILURE_RETRIES = 3
PREEMPTION_RETRIES = 10


@dataclass(frozen=True)
class TpuShape:
    slice_type: str
    chips: int
    tensor_parallel: int
    data_parallel: int
    replicas: int
    attention_data_parallel: int = 1


SHAPES = {
    Solver.SMALL: TpuShape("v4-16", 8, 8, 1, 8),
    Solver.LARGE: TpuShape("v4-32", 16, 16, 1, 8),
    Solver.SNOWBALL: TpuShape("v4-32", 16, 1, 1, 4, attention_data_parallel=16),
}


def stage_solver(solver: Solver, root: str) -> None:
    """Use Marin's streaming HF mirror from a regional CPU job, before TPU submission."""
    regional_worker()
    os.environ["HF_TOKEN"] = require_secret("HF_TOKEN")
    if solver not in SHAPES:
        raise ValueError("The hosted solver stays on Together and has no TPU weights")
    pin = PINS[solver]
    weights = resolve_model_path(SOLVER_MODELS[solver], 30, pin.revision)
    if not weights.startswith("gs://marin-us-central2/"):
        raise ValueError(f"Model cache is not regional: {weights}")
    marker = json.loads((StoragePath(weights) / ".cache_metadata.json").read_text())
    if marker["source_revision"] != pin.revision:
        raise ValueError("Regional cache does not bind the frozen checkpoint")
    immutable_json(
        StoragePath(root) / "staging" / f"{solver.value}.json",
        {
            "solver": solver.value,
            "model": SOLVER_MODELS[solver],
            "revision": pin.revision,
            "weights": weights,
            "config_sha256": file_sha256(StoragePath(weights) / "config.json"),
            "cache_metadata": marker,
        },
    )


def tpu_model(solver: Solver, weights: str) -> ModelConfig:
    shape = SHAPES[solver]
    if not weights.startswith("gs://marin-us-central2/"):
        raise ValueError("TPU weights must be staged in us-central2")
    extra = ("--enable-auto-tool-choice", "--enable-prefix-caching")
    if solver == Solver.SNOWBALL:
        # Ordinary DP replicates the expert weights. Attention DP keeps all five
        # KV heads local while the backend's EXPERT axis shards the MoE weights.
        extra += (
            "--enable-expert-parallel",
            "--api-server-count",
            "1",
            "--additional-config",
            json.dumps(
                {
                    "sharding": {
                        "sharding_strategy": {
                            "tensor_parallelism": shape.attention_data_parallel,
                            "enable_dp_attention": True,
                            "attn_dp_size": shape.attention_data_parallel,
                        }
                    }
                }
            ),
        )
    return ModelConfig(
        name=f"atlas-difficulty-v4-{solver.value}",
        location=weights,
        identity=f"{SOLVER_MODELS[solver]}@{PINS[solver].revision}",
        tokenizer=SOLVER_MODELS[solver],
        tokenizer_revision=PINS[solver].revision,
        resource_hint=ResourceHint(cpu=64, memory="320g", disk="80g"),
        serve=ServeConfig(
            tensor_parallel_size=shape.tensor_parallel,
            data_parallel_size=shape.data_parallel,
            instances=1,
            max_model_len=73728 if solver == Solver.SNOWBALL else 65536,
            max_num_batched_tokens=16384,
            max_num_seqs=128,
            gpu_memory_utilization=0.90,
            tool_call_parser=(
                "hermes" if solver == Solver.SNOWBALL else "qwen3_coder" if solver == Solver.SMALL else "qwen3_xml"
            ),
            reasoning_parser="qwen3" if solver == Solver.LARGE else None,
            auto_overrides=False,
            object_store_load_mode=ObjectStoreLoadMode.STREAM,
            vllm_extra_args=extra,
        ),
    )


def tpu_config(solver: Solver, weights: str) -> RemoteInferenceConfig:
    shape = SHAPES[solver]
    config = inference_config_for_model(
        tpu_model(solver, weights),
        AcceleratorChoice(Platform.TPU, tpu_type=shape.slice_type, region=REGION, preemptible=False),
        env_vars={
            "MARIN_TEMP_PREFIX": "gs://marin-us-central2",
            "MODEL_IMPL_TYPE": "auto",
            "TPU_MULTIPROCESS_DP": "0",
            **({"NEW_MODEL_DESIGN": "1"} if solver == Solver.SNOWBALL else {}),
        },
        priority=priority_band_value("interactive"),
        api_model=SOLVER_MODELS[solver],
    )
    if not isinstance(config.engine, VllmEngineConfig) or config.engine.launcher != VllmLauncherType.TPU:
        raise ValueError("Difficulty TPU serving requires Marin's vLLM TPU launcher")
    return replace(
        config,
        broker=None,
        iris=replace(
            config.iris,
            max_retries_failure=FAILURE_RETRIES,
            max_retries_preemption=PREEMPTION_RETRIES,
            worker_timeout=None,
            endpoint_ready_timeout_seconds=7200,
        ),
        engine=replace(config.engine, startup_timeout_seconds=7200),
    )


def long_context_request(solver: Solver) -> tuple[dict, int]:
    """Build and locally validate the frozen tokenizer's near-budget prefill."""
    settings = sampling_settings(solver, [])
    parameters = request_parameters(settings)
    request = {
        "model": SOLVER_MODELS[solver],
        "messages": [{"role": "user", "content": "red " * 49000}],
        "max_tokens": 32,
        **{key: value for key, value in parameters.items() if key != "extra_body"},
        **parameters.get("extra_body", {}),
    }
    expected = hf_prompt_tokens(
        request,
        model_identity=SOLVER_MODELS[solver],
        model_revision=PINS[solver].revision,
        chat_template_kwargs=request.get("chat_template_kwargs", {}),
    )
    if not settings.input_tokens * 0.9 <= expected <= settings.input_tokens:
        raise ValueError("Capacity probe must exercise the frozen input budget")
    return request, expected


def long_context_probe(solver: Solver, endpoint: str, root: str) -> dict:
    """Exercise a near-budget prefill; capacity evidence never counts as a scored attempt."""
    request, expected = long_context_request(solver)
    evidence = probe_serving_endpoint(endpoint, "EMPTY", request, expected_prompt_tokens=expected)
    path = StoragePath(root) / "qualification" / solver.value / (digest(evidence) + ".json")
    immutable_json(path, evidence)
    return {
        "prompt_tokens": expected,
        "evidence_uri": str(path),
        "request_sha256": digest(request),
        "capacity_probe_only": True,
    }


def run_tpu_pool(solver: Solver, root: str, replicas: int) -> None:
    """Own direct slice sessions; publish only a completely qualified generation."""
    regional_worker()
    configure_coreweave_s3()
    os.environ["HF_TOKEN"] = require_secret("HF_TOKEN")
    shape = SHAPES[solver]
    if not 1 <= replicas <= shape.replicas:
        raise ValueError("Replica count exceeds the provisional solver allocation")
    staged = json.loads((StoragePath(root) / "staging" / f"{solver.value}.json").read_text())
    if (staged["model"], staged["revision"]) != (SOLVER_MODELS[solver], PINS[solver].revision):
        raise ValueError("Staged weights differ from the frozen identity")
    config = tpu_config(solver, staged["weights"])
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def replica(index: int) -> None:
        try:
            with remote_inference(config) as session:
                ready = session.ready_direct_session()
                effective = ready.effective_serving
                if effective is None or effective.max_model_len is None or effective.max_model_len < 65536:
                    raise ValueError("TPU endpoint did not qualify the frozen context length")
                endpoint = ready.model.endpoint.base_url
                probe = probe_endpoint(solver, endpoint, SOLVER_MODELS[solver], "EMPTY", sampling_settings(solver, []))
                context_probe = long_context_probe(solver, endpoint, root)
                sweep = [
                    asyncio.run(warm_throughput(endpoint, SOLVER_MODELS[solver], shape.chips, concurrency=n))
                    for n in (8, 32, 128)
                ]
                for measurement in sweep:
                    measurement["output_tokens_per_second_per_chip"] = measurement.pop(
                        "output_tokens_per_second_per_gpu"
                    )
                measurement = max(sweep, key=lambda value: value["output_tokens_per_second"])
                events.put(
                    (
                        "ready",
                        {
                            "index": index,
                            "endpoint_name": session.endpoint_name,
                            "engine_jobs": [str(job.job_id) for job in session.jobs],
                            "effective_serving": asdict(effective),
                            "probe": probe,
                            "long_context_probe": context_probe,
                            "warm_throughput": measurement,
                            "throughput_sweep": sweep,
                        },
                    )
                )
                health_failures = 0
                while not stop.wait(30):
                    session.check_alive()
                    if session.backend_state() != InferenceBackendState.READY:
                        continue
                    try:
                        current = session.direct_session().model.endpoint.base_url
                        response = httpx.get(current.rstrip("/") + "/models", timeout=30)
                        response.raise_for_status()
                        if SOLVER_MODELS[solver] not in {model["id"] for model in response.json()["data"]}:
                            raise ValueError("Live TPU endpoint stopped advertising its frozen solver")
                    except Exception:
                        health_failures += 1
                        if health_failures >= 3:
                            raise
                    else:
                        health_failures = 0
        except BaseException as error:
            events.put(("failed", error))
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=replicas) as workers:
        futures = [workers.submit(contextvars.copy_context().run, replica, index) for index in range(replicas)]
        try:
            ready = []
            while len(ready) < replicas:
                event, value = events.get(timeout=7500)
                if event == "failed":
                    raise value
                ready.append(value)
            ready.sort(key=lambda value: value["index"])
            receipt = {
                "solver": solver,
                "model": SOLVER_MODELS[solver],
                "revision": PINS[solver].revision,
                "platform": "tpu",
                "region": REGION,
                "zone": ZONE,
                "priority": "INTERACTIVE",
                "chip_count": shape.chips * replicas,
                "gpu_count": 0,
                "shape": asdict(shape),
                "replicas": ready,
                "endpoint_names": [item["endpoint_name"] for item in ready],
                "effective_serving": ready[0]["effective_serving"],
                "serving_stack": tomllib.loads(Path("config/external/vllm/tpu.toml").read_text()),
                "compilation_cache": config.engine.compilation_cache,
                "ready_at": datetime.now(UTC).isoformat(),
            }
            # Retain expensive qualification measurements even if publication fails.
            immutable_json(StoragePath(root) / "qualification" / solver.value / (digest(receipt) + ".json"), receipt)
            context = iris_ctx()
            info = get_job_info()
            assert info is not None
            name = f"/serve/atlas-difficulty-tpu-{solver.value}-{uuid.uuid4().hex}"
            app = pool_router(receipt["endpoint_names"], context.client.resolve_endpoint)
            with bind_serving_socket(info.advertise_host, 0) as sock, serve_app_background(app, sock):
                address = f"http://{info.advertise_host}:{sock.getsockname()[1]}"
                with context.registry.registered(
                    name, address, {"model": receipt["model"]}, access=EndpointAccess.ENDPOINT_ACCESS_LINK
                ):
                    receipt["endpoint_name"] = name
                    publish_pool_receipt(root + "/pools", solver, receipt)
                    LOGGER.info("DIFFICULTY_ENGINE_READY %s", json.dumps(receipt, sort_keys=True))
                    while not stop.wait(30):
                        for future in futures:
                            if future.done():
                                future.result()
                                raise RuntimeError("TPU replica stopped unexpectedly")
                    event, value = events.get(timeout=60)
                    raise RuntimeError(f"TPU pool lost a replica: {event}") from value
        finally:
            stop.set()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage-solver", "stage-inputs"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--solver", type=Solver)
    parser.add_argument("--sealed-manifest")
    parser.add_argument("--sealed-sha256")
    parser.add_argument("--judge-qualification")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.operation == "stage-solver":
        if args.solver is None:
            parser.error("stage-solver requires --solver")
        stage_solver(args.solver, args.root)
    else:
        if not all((args.sealed_manifest, args.sealed_sha256, args.judge_qualification)):
            parser.error("stage-inputs requires the sealed manifest, hash and judge qualification")
        stage_campaign(args.root, args.sealed_manifest, args.sealed_sha256, args.judge_qualification)


if __name__ == "__main__":
    main()
