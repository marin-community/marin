# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import requests
from fray.client import JobHandle
from fray.current_client import current_client
from fray.types import ActorConfig, Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cluster.client.job_info import get_job_info
from rigging.log_setup import configure_logging

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.broker import InferenceBroker
from marin.inference.proxy import serve_inference_proxy
from marin.inference.types import InferenceRequestProvider, InferenceResponseProvider, OpenAIEndpoint, RunningModel
from marin.inference.vllm_server import VllmEnvironment
from marin.inference.worker import (
    InferenceWorker,
    run_inference_worker,
)
from marin.training.run_environment import env_vars_for_dependency_groups

logger = logging.getLogger(__name__)

DEFAULT_BROKERED_REQUEST_TIMEOUT_MARGIN = timedelta(seconds=60)
DEFAULT_BROKERED_PROXY_REQUEST_TIMEOUT = timedelta(seconds=300)
DEFAULT_BROKERED_REQUEST_LEASE_TIMEOUT = DEFAULT_BROKERED_PROXY_REQUEST_TIMEOUT - DEFAULT_BROKERED_REQUEST_TIMEOUT_MARGIN
DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT = (
    DEFAULT_BROKERED_REQUEST_LEASE_TIMEOUT - DEFAULT_BROKERED_REQUEST_TIMEOUT_MARGIN
)
DEFAULT_BROKERED_PROXY_START_TIMEOUT = timedelta(seconds=10)

# Default eval-client concurrency and per-worker HTTP fanout.
DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER = 16


@dataclass(frozen=True)
class VllmServerConfig:
    # Local port for `vllm serve` inside each worker process.
    port: int = 8000
    # vLLM TPU startup can include model download and compile work.
    timeout_seconds: int = 1800
    # Default vLLM maximum sequence length, including prompt and generated tokens.
    max_model_len: int = 4096
    # Keep prefill modest on v5p-8 while still exercising vLLM's internal batching.
    max_num_batched_tokens: int = 1024


@dataclass(frozen=True)
class VllmProxyConfig:
    # The eval client talks to this local OpenAI-compatible URL.
    host: str = "127.0.0.1"
    # Local OpenAI-compatible proxy port; 0 chooses an available port.
    port: int = 0
    # End-to-end timeout from eval request to brokered response.
    request_timeout_seconds: float = DEFAULT_BROKERED_PROXY_REQUEST_TIMEOUT.total_seconds()
    # Startup probe timeout through the brokered proxy.
    readiness_timeout_seconds: float = DEFAULT_BROKERED_PROXY_REQUEST_TIMEOUT.total_seconds()
    # Backpressure guard for client requests waiting on broker responses.
    max_pending_requests: int = 256
    # Response poll batch size; larger than default concurrency so one poll can drain normal completions.
    response_fetch_batch_size: int = 64
    # Request fields accepted by the eval client but unsupported by the serving backend.
    ignored_request_fields: tuple[str, ...] = ()
    # Local uvicorn startup should be quick; vLLM startup has a separate timeout.
    server_start_timeout_seconds: float = DEFAULT_BROKERED_PROXY_START_TIMEOUT.total_seconds()


@dataclass(frozen=True)
class InferenceWorkerConfig:
    # Number of inference worker jobs in Iris mode. Local mode requires one worker.
    count: int = 1
    # Individual HTTP requests each worker may keep active against its vLLM server.
    max_in_flight_per_worker: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER
    # Slow vLLM calls fail once from the worker before their broker lease can expire.
    request_timeout_seconds: float = DEFAULT_BROKERED_WORKER_REQUEST_TIMEOUT.total_seconds()


@dataclass(frozen=True)
class BrokeredVllmSystemConfig:
    model: str
    tokenizer: str | None = None
    # Recovery timeout for work fetched by a worker but not answered.
    request_lease_timeout_seconds: float = DEFAULT_BROKERED_REQUEST_LEASE_TIMEOUT.total_seconds()
    server: VllmServerConfig = field(default_factory=VllmServerConfig)
    proxy: VllmProxyConfig = field(default_factory=VllmProxyConfig)
    workers: InferenceWorkerConfig = field(default_factory=InferenceWorkerConfig)
    # Required when running worker jobs through Iris; ignored by local mode.
    worker_resources: ResourceConfig | None = None
    # Broker is CPU-only; TPU work happens in worker jobs.
    broker_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig.with_cpu(cpu=2, ram="8g", disk="20g")
    )
    # Worker jobs need the generic TPU stack plus the TPU-vLLM runtime stack; the CPU parent only needs base Marin.
    worker_environment_extras: tuple[str, ...] = ("tpu", "vllm")
    # TPU/vLLM-specific env vars stay explicit at the entrypoint.
    worker_env_vars: Mapping[str, str] = field(default_factory=dict)
    # Actor startup waits on Iris endpoint registration.
    broker_ready_timeout_seconds: float = 900.0
    # Applied to broker actor and TPU worker child jobs.
    priority: int = 0

    def __post_init__(self) -> None:
        _validate_timeout_ordering(self)


@contextlib.contextmanager
def start_local_brokered_vllm(config: BrokeredVllmSystemConfig) -> Iterator[RunningModel]:
    """Start broker, local vLLM, worker, and local proxy in this process."""

    if config.workers.count != 1:
        raise ValueError("Local brokered vLLM mode supports exactly one worker; use Iris mode for multiple workers.")

    broker = InferenceBroker(request_lease_timeout_seconds=config.request_lease_timeout_seconds)
    with start_local_vllm_server(config) as upstream:
        worker = InferenceWorker(
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=config.workers.request_timeout_seconds,
        )
        with (
            _start_proxy(config, broker) as proxy_model,
            run_inference_worker(worker, max_in_flight=config.workers.max_in_flight_per_worker),
        ):
            _wait_for_brokered_vllm_ready(proxy_model, timeout_seconds=config.proxy.readiness_timeout_seconds)
            logger.info("Started local InferenceWorker")
            yield proxy_model


@contextlib.contextmanager
def start_iris_brokered_vllm(config: BrokeredVllmSystemConfig) -> Iterator[RunningModel]:
    """Start Iris child broker/workers and expose a local proxy in the parent job."""

    if config.worker_resources is None:
        raise ValueError("worker_resources must be set for Iris brokered vLLM mode.")

    client = current_client()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("start_iris_brokered_vllm must run inside an Iris job.")
    run_id = uuid.uuid4().hex[:8]
    broker_name = f"vllm-broker-{run_id}"
    worker_prefix = f"vllm-worker-{run_id}"
    broker_group = client.create_actor_group(
        InferenceBroker,
        name=broker_name,
        count=1,
        request_lease_timeout_seconds=config.request_lease_timeout_seconds,
        resources=config.broker_resources,
        actor_config=ActorConfig(max_task_retries=0, priority=config.priority),
    )
    worker_jobs: list[JobHandle] = []
    try:
        logger.info("Waiting for broker actor name=%s", broker_name)
        broker_handle = broker_group.wait_ready(count=1, timeout=config.broker_ready_timeout_seconds)[0]
        request_provider = cast(InferenceRequestProvider, broker_handle)
        response_provider = cast(InferenceResponseProvider, broker_handle)
        worker_extras = list(config.worker_environment_extras)
        worker_environment = create_environment(
            extras=worker_extras,
            env_vars=env_vars_for_dependency_groups(
                config.worker_resources,
                worker_extras,
                dict(config.worker_env_vars),
            ),
        )
        for worker_index in range(config.workers.count):
            job = client.submit(
                JobRequest(
                    name=f"{worker_prefix}-{worker_index}",
                    entrypoint=Entrypoint.from_callable(_run_iris_inference_worker, args=(config, request_provider)),
                    resources=config.worker_resources,
                    environment=worker_environment,
                    priority=config.priority,
                )
            )
            worker_jobs.append(job)
            logger.info("Submitted vLLM worker job_id=%s index=%d", job.job_id, worker_index)

        with _start_proxy(config, response_provider) as running_model:
            _wait_for_brokered_vllm_ready(running_model, timeout_seconds=config.proxy.readiness_timeout_seconds)
            yield running_model
    finally:
        _terminate_jobs(worker_jobs)
        with contextlib.suppress(Exception):
            broker_group.shutdown()


@contextlib.contextmanager
def start_local_vllm_server(config: BrokeredVllmSystemConfig) -> Iterator[RunningModel]:
    server_config = config.server
    vllm_model = ModelConfig(
        name="brokered-vllm",
        path=config.model,
        engine_kwargs={
            "max_model_len": server_config.max_model_len,
            "max_num_batched_tokens": server_config.max_num_batched_tokens,
        },
    )
    logger.info(
        "Starting local vLLM server model=%s port=%d max_model_len=%d max_num_batched_tokens=%d",
        config.model,
        server_config.port,
        server_config.max_model_len,
        server_config.max_num_batched_tokens,
    )
    with VllmEnvironment(
        model=vllm_model, port=server_config.port, timeout_seconds=server_config.timeout_seconds
    ) as env:
        if env.model_id is None:
            raise RuntimeError("Expected vLLM server to expose a model id.")
        yield RunningModel(
            endpoint=OpenAIEndpoint(base_url=env.server_url, model=env.model_id),
            tokenizer=config.tokenizer,
        )


def _run_iris_inference_worker(config: BrokeredVllmSystemConfig, broker_handle: InferenceRequestProvider) -> None:
    configure_logging()
    with start_local_vllm_server(config) as upstream:
        worker = InferenceWorker(
            broker=broker_handle,
            upstream=upstream,
            request_timeout_seconds=config.workers.request_timeout_seconds,
        )
        # Worker jobs poll until the parent job terminates them after the eval client exits.
        worker.run_forever(max_in_flight=config.workers.max_in_flight_per_worker)


@contextlib.contextmanager
def _start_proxy(config: BrokeredVllmSystemConfig, broker: InferenceResponseProvider) -> Iterator[RunningModel]:
    proxy_config = config.proxy
    with serve_inference_proxy(
        broker=broker,
        model=config.model,
        host=proxy_config.host,
        port=proxy_config.port,
        request_timeout_seconds=proxy_config.request_timeout_seconds,
        readiness_timeout_seconds=config.proxy.readiness_timeout_seconds,
        max_pending_requests=proxy_config.max_pending_requests,
        response_fetch_batch_size=proxy_config.response_fetch_batch_size,
        server_start_timeout_seconds=proxy_config.server_start_timeout_seconds,
        ignored_request_fields=proxy_config.ignored_request_fields,
    ) as running_model:
        yield RunningModel(endpoint=running_model.endpoint, tokenizer=config.tokenizer)


def _terminate_jobs(jobs: list[JobHandle]) -> None:
    for job in jobs:
        with contextlib.suppress(Exception):
            logger.info("Terminating vLLM worker job_id=%s", job.job_id)
            job.terminate()


def _validate_timeout_ordering(config: BrokeredVllmSystemConfig) -> None:
    worker_timeout = config.workers.request_timeout_seconds
    lease_timeout = config.request_lease_timeout_seconds
    proxy_timeout = config.proxy.request_timeout_seconds
    if not 0 < worker_timeout < lease_timeout < proxy_timeout:
        raise ValueError(
            "Brokered vLLM timeouts must satisfy "
            "0 < workers.request_timeout_seconds < request_lease_timeout_seconds "
            "< proxy.request_timeout_seconds; "
            f"got worker={worker_timeout:.1f}s lease={lease_timeout:.1f}s proxy={proxy_timeout:.1f}s."
        )
    if config.proxy.readiness_timeout_seconds <= 0:
        raise ValueError("proxy.readiness_timeout_seconds must be positive.")


def _wait_for_brokered_vllm_ready(running_model: RunningModel, *, timeout_seconds: float) -> None:
    models_url = running_model.endpoint.url("models")
    logger.info("Waiting for brokered vLLM readiness url=%s timeout_seconds=%.1f", models_url, timeout_seconds)
    response = requests.get(models_url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("data"):
        raise RuntimeError(f"No models returned from brokered vLLM readiness check: {str(payload)[:2000]}")
    logger.info("Brokered vLLM readiness check passed url=%s", models_url)
