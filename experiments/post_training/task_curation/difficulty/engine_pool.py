# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent regional solver engines and their measured serving receipts."""

import argparse
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

import httpx
from finestore.rl.difficulty import SOLVER_MODELS, Solver
from iris.rpc.proto_display import priority_band_name, priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import BrokerConfig, ObjectStoreLoadMode, RemoteInferenceConfig, VllmEngineConfig
from marin.inference.iris import remote_inference
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.launch import JOB_TIMEOUT
from experiments.post_training.task_curation.difficulty.runtime import TOGETHER_BASE_URL, hosted_liveness_response
from experiments.post_training.task_curation.difficulty_config import TOGETHER_API_KEY_ENV
from experiments.post_training.task_curation.difficulty_protocol import PINS

LOGGER = logging.getLogger(__name__)
POOL_CLUSTER = "cw-us-east-08a"
POOL_CACHE_ROOT = "s3://marin-us-east-08a"
POOL_GPU_TYPE = "GB200"
HEALTH_INTERVAL = 30
WARM_CONCURRENCY = 128
WARM_OUTPUT_TOKENS = 256
POOL_FAILURE_RETRIES = 3
POOL_PREEMPTION_RETRIES = 10


@dataclass(frozen=True)
class EngineShape:
    instances: int
    tensor_parallel: int
    data_parallel: int
    pipeline_parallel: int = 1

    @property
    def gpu_count(self) -> int:
        return self.instances * self.tensor_parallel * self.data_parallel * self.pipeline_parallel


ENGINE_SHAPES = {
    Solver.SMALL: EngineShape(4, 1, 4),
    Solver.LARGE: EngineShape(8, 2, 2),
    Solver.SNOWBALL: EngineShape(4, 1, 4),
}


def pool_model(solver: Solver) -> ModelConfig:
    """Keep the protocol's checkpoint identity while sizing one GB200 pool."""
    if solver == Solver.HOSTED:
        raise ValueError("The hosted solver retains its provider's opaque weight identity")
    shape = ENGINE_SHAPES[solver]
    extra = ("--enable-auto-tool-choice", "--enable-prefix-caching")
    parser = "qwen3_coder" if solver == Solver.SMALL else "qwen3_xml"
    reasoning = "qwen3" if solver == Solver.LARGE else None
    if solver == Solver.SNOWBALL:
        parser = "hermes"
        extra += (
            "--enable-expert-parallel",
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "PIECEWISE",
                    "splitting_ops": [
                        "vllm::unified_attention_with_output",
                        "vllm::unified_kv_cache_update",
                        "vllm::grug_moe_short_conv",
                    ],
                }
            ),
        )
    return ModelConfig(
        name=f"atlas-difficulty-gb200-{solver.value}",
        location=SOLVER_MODELS[solver],
        identity=f"{SOLVER_MODELS[solver]}@{PINS[solver].revision}",
        revision=PINS[solver].revision,
        tokenizer=SOLVER_MODELS[solver],
        tokenizer_revision=PINS[solver].revision,
        resource_hint=ResourceHint(
            gpu={POOL_GPU_TYPE: shape.tensor_parallel * shape.data_parallel},
            cpu=64,
            memory="512g",
            disk="500g",
        ),
        serve=ServeConfig(
            tensor_parallel_size=shape.tensor_parallel,
            data_parallel_size=shape.data_parallel,
            pipeline_parallel_size=shape.pipeline_parallel,
            instances=shape.instances,
            max_model_len=73728 if solver == Solver.SNOWBALL else 65536,
            max_num_batched_tokens=16384,
            max_num_seqs=128,
            gpu_memory_utilization=0.90,
            tool_call_parser=parser,
            reasoning_parser=reasoning,
            auto_overrides=solver != Solver.SNOWBALL,
            object_store_load_mode=ObjectStoreLoadMode.STAGE_LOCAL,
            vllm_extra_args=extra,
        ),
    )


def pool_config(solver: Solver) -> RemoteInferenceConfig:
    model = pool_model(solver)
    shape = ENGINE_SHAPES[solver]
    config = inference_config_for_model(
        model,
        AcceleratorChoice(
            Platform.GPU,
            gpu_type=POOL_GPU_TYPE,
            gpu_count=shape.tensor_parallel * shape.data_parallel,
            target_cluster=POOL_CLUSTER,
        ),
        env_vars={"MARIN_TEMP_PREFIX": POOL_CACHE_ROOT, "NCCL_BUFFSIZE": "4194304"},
        priority=priority_band_value("interactive"),
        api_model=SOLVER_MODELS[solver],
    )
    if not isinstance(config.engine, VllmEngineConfig):
        raise ValueError("Solver pools require vLLM")
    return replace(
        config,
        engine=replace(config.engine, gpu_release=VLLM_GPU_RELEASE, startup_timeout_seconds=7200),
        broker=BrokerConfig(
            max_retries_failure=POOL_FAILURE_RETRIES,
            max_retries_preemption=POOL_PREEMPTION_RETRIES,
            priority=priority_band_value("production"),
        ),
        iris=replace(
            config.iris,
            max_retries_failure=POOL_FAILURE_RETRIES,
            max_retries_preemption=POOL_PREEMPTION_RETRIES,
            worker_timeout=JOB_TIMEOUT,
            worker_resources=replace(config.iris.worker_resources, target_cluster=None, regions=None),
            endpoint_ready_timeout_seconds=7200,
        ),
    )


async def warm_throughput(
    endpoint: str, model: str, gpu_count: int, api_key: str | None = None, concurrency: int = WARM_CONCURRENCY
) -> dict:
    """Measure completed output tokens over a concurrent warm batch, including HTTP overhead."""
    headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(
        timeout=1800, limits=httpx.Limits(max_connections=concurrency), headers=headers
    ) as client:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Continue counting integers separated by spaces, starting at 1."}],
            "max_tokens": WARM_OUTPUT_TOKENS,
            "temperature": 0.7,
        }
        if gpu_count:
            body["ignore_eos"] = True

        async def request() -> dict:
            response = await client.post(endpoint.rstrip("/") + "/chat/completions", json=body)
            response.raise_for_status()
            return response.json()["usage"]

        await request()
        started = time.monotonic()
        usages = await asyncio.gather(*(request() for _ in range(concurrency)))
        elapsed = time.monotonic() - started
    output = sum(usage["completion_tokens"] for usage in usages)
    return {
        "requests": len(usages),
        "concurrency": concurrency,
        "output_tokens": output,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": output / elapsed,
        "output_tokens_per_second_per_gpu": output / elapsed / gpu_count if gpu_count else None,
        "basis": "completed decode output tokens / concurrent-batch wall time, including prefill and HTTP overhead",
    }


def publish_pool_receipt(registry: str, solver: Solver, receipt: dict) -> None:
    """Retain ready generations before replacing the active solver endpoint."""
    target = StoragePath(registry) / f"{solver.value}.json"
    encoded = json.dumps(receipt, sort_keys=True) + "\n"
    versions = [encoded]
    if target.exists():
        previous = target.read_text()
        previous_receipt = json.loads(previous)
        for field in ("solver", "model", "revision"):
            if previous_receipt[field] != receipt[field]:
                raise ValueError(f"Cannot replace solver pool with a different {field}")
        versions.insert(0, previous)
    history = target.parent / "history" / solver.value
    history.mkdirs()
    for version in versions:
        digest = hashlib.sha256(version.encode()).hexdigest()
        (history / f"{digest}.json").write_text(version)
    # S3 publishes each complete object on close; readers retain the previous
    # ready endpoint until this replacement succeeds.
    target.write_text(encoded)


def run_pool(solver: Solver, registry: str) -> None:
    configure_coreweave_s3()
    shape = ENGINE_SHAPES[solver]
    config = pool_config(solver)
    broker = config.broker
    assert broker is not None and broker.priority is not None
    LOGGER.info("DIFFICULTY_ENGINE_START %s", json.dumps({"solver": solver, "shape": asdict(shape)}))
    # remote_inference stages uncached HF weights through a CPU child at this
    # root's regional placement before reserving any accelerator workers.
    with remote_inference(config) as session:
        ready = session.ready_direct_session()
        if ready.effective_serving is None or ready.effective_serving.max_model_len < 65536:
            raise ValueError("Solver pool did not confirm the protocol context window")
        measurement = asyncio.run(warm_throughput(ready.model.endpoint.base_url, SOLVER_MODELS[solver], shape.gpu_count))
        engine = asdict(config.engine)
        engine["extra_metric_families"] = sorted(engine["extra_metric_families"])
        receipt = {
            "solver": solver,
            "model": SOLVER_MODELS[solver],
            "revision": PINS[solver].revision,
            "endpoint_name": session.endpoint_name,
            "gpu_count": shape.gpu_count,
            "engine_jobs": [str(job.job_id) for job in session.jobs],
            "shape": asdict(shape),
            "warm_throughput": measurement,
            "effective_serving": asdict(ready.effective_serving),
            "requested_config": asdict(pool_model(solver)),
            "configured_priority_bands": {
                "gpu_workers": priority_band_name(config.iris.priority),
                "cpu_broker": priority_band_name(broker.priority),
            },
            "engine": engine,
            "vllm_gpu_release": asdict(VLLM_GPU_RELEASE),
            "ready_at": datetime.now(UTC).isoformat(),
        }
        publish_pool_receipt(registry, solver, receipt)
        LOGGER.info("DIFFICULTY_ENGINE_READY %s", json.dumps(receipt, sort_keys=True))
        while True:
            session.check_alive()
            time.sleep(HEALTH_INTERVAL)


def run_provider_warm(registry: str) -> None:
    configure_coreweave_s3()
    model = SOLVER_MODELS[Solver.HOSTED]
    key = require_secret(TOGETHER_API_KEY_ENV)
    response = hosted_liveness_response(TOGETHER_BASE_URL, key, model)
    response.raise_for_status()
    measurement = asyncio.run(warm_throughput(TOGETHER_BASE_URL, model, 0, key, concurrency=32))
    receipt = {
        "solver": Solver.HOSTED,
        "model": model,
        "gpu_count": 0,
        "route": "provider",
        "revision": f"unexposed:{response.json().get('system_fingerprint')}",
        "tokenizer_revision": PINS[Solver.HOSTED].revision,
        "warm_throughput": measurement,
        "ready_at": datetime.now(UTC).isoformat(),
    }
    target = StoragePath(registry) / "hosted.json"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdirs()
    target.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    LOGGER.info("DIFFICULTY_PROVIDER_WARM %s", json.dumps(receipt, sort_keys=True))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Solver, required=True)
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()
    if args.solver == Solver.HOSTED:
        run_provider_warm(args.registry)
    else:
        run_pool(args.solver, args.registry)


if __name__ == "__main__":
    main()
