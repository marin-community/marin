# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import cast

import requests
from fray import current_client
from fray.client import JobHandle
from fray.types import ActorConfig, Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from rigging.log_setup import configure_logging

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.local_workload_broker import LocalWorkloadBroker
from marin.inference.types import OpenAIEndpoint, RunningModel
from marin.inference.vllm_http_proxy import serve_vllm_http_proxy
from marin.inference.vllm_server import VllmEnvironment
from marin.inference.vllm_worker import (
    VllmWorker,
    run_vllm_worker,
)
from marin.inference.workload_broker import WorkloadBroker
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

DEFAULT_BROKERED_PROXY_TIMEOUT_SECONDS = 300.0

# Default eval-client concurrency and per-worker HTTP fanout.
DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER = 16


@dataclass(frozen=True)
class VllmServerConfig:
    # Local port for `vllm serve` inside each worker process.
    port: int = 8000
    # vLLM TPU startup can include model download and compile work.
    timeout_seconds: int = 1800
    # Compact default context length; VLLM_ALLOW_LONG_MAX_MODEL_LEN is still set by the caller's env.
    max_model_len: int = 4096
    # Keep prefill modest on v5p-8 while still exercising vLLM's internal batching.
    max_num_batched_tokens: int = 1024


@dataclass(frozen=True)
class VllmProxyConfig:
    # The eval client talks to this local OpenAI-compatible URL.
    host: str = "127.0.0.1"
    # 0 picks a free loopback port; Iris CPU workers can be reused and may already have 8001 bound.
    port: int = 0
    # End-to-end timeout from eval request to brokered response.
    request_timeout_seconds: float = DEFAULT_BROKERED_PROXY_TIMEOUT_SECONDS
    # Startup probe timeout through the brokered proxy.
    readiness_timeout_seconds: float = DEFAULT_BROKERED_PROXY_TIMEOUT_SECONDS
    # Backpressure guard for client requests waiting on broker responses.
    max_pending_requests: int = 256
    # Response poll batch size; larger than default concurrency so one poll can drain normal completions.
    response_fetch_batch_size: int = 64
    # Local uvicorn startup should be quick; vLLM startup has a separate timeout.
    server_start_timeout_seconds: float = 10.0
    # Hint for clients rejected by the proxy pending-request guard.
    retry_after_seconds: int = 1


@dataclass(frozen=True)
class VllmWorkerConfig:
    # Number of vLLM worker jobs in Iris mode. Local mode requires one worker.
    count: int = 1
    # Individual HTTP requests each worker may keep active against its vLLM server.
    max_in_flight_per_worker: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER
    # Slow vLLM calls fail once from the worker before their broker lease can expire.
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class BrokeredVllmSystemConfig:
    model: str
    tokenizer: str | None = None
    # Recovery timeout for work fetched by a worker but not answered.
    request_lease_timeout_seconds: float = 150.0
    server: VllmServerConfig = field(default_factory=VllmServerConfig)
    proxy: VllmProxyConfig = field(default_factory=VllmProxyConfig)
    workers: VllmWorkerConfig = field(default_factory=VllmWorkerConfig)

    def __post_init__(self) -> None:
        _validate_timeout_ordering(self)


@dataclass(frozen=True)
class IrisBrokeredVllmRuntimeConfig:
    # All Iris jobs in this serving topology must stay in one region to avoid cross-region serving traffic.
    region: str
    # Model- and eval-specific TPU resources for each vLLM worker.
    worker_resources: ResourceConfig
    # Broker is CPU-only: it holds queues and compressed request/response payloads.
    broker_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig.with_cpu(cpu=2, ram="8g", disk="20g")
    )
    # Worker jobs need the TPU/vLLM extras; the CPU parent only needs the base Marin environment.
    worker_environment_extras: tuple[str, ...] = ("tpu", "vllm")
    # TPU/vLLM-specific env vars stay explicit at the entrypoint.
    worker_env_vars: Mapping[str, str] = field(default_factory=dict)
    # Actor startup waits on Iris endpoint registration.
    broker_ready_timeout_seconds: float = 900.0
    # Applied to broker actor and TPU worker child jobs.
    priority_band: job_pb2.PriorityBand = job_pb2.PRIORITY_BAND_UNSPECIFIED

    def __post_init__(self) -> None:
        _validate_resource_zone(self.broker_resources, name="broker_resources", region=self.region)
        _validate_resource_zone(self.worker_resources, name="worker_resources", region=self.region)

    @property
    def broker_resources_in_region(self) -> ResourceConfig:
        return replace(self.broker_resources, regions=[self.region])

    @property
    def worker_resources_in_region(self) -> ResourceConfig:
        return replace(self.worker_resources, regions=[self.region])


@contextlib.contextmanager
def start_local_brokered_vllm(config: BrokeredVllmSystemConfig) -> Iterator[RunningModel]:
    """Start broker, local vLLM, worker, and local proxy in this process."""

    if config.workers.count != 1:
        raise ValueError("Local brokered vLLM mode supports exactly one worker; use Iris mode for multiple workers.")

    broker = LocalWorkloadBroker(request_lease_timeout_seconds=config.request_lease_timeout_seconds)
    with start_local_vllm_server(config) as upstream:
        worker = VllmWorker(
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=config.workers.request_timeout_seconds,
        )
        with (
            _start_proxy(config, broker) as proxy_model,
            run_vllm_worker(worker, max_in_flight=config.workers.max_in_flight_per_worker),
        ):
            _wait_for_brokered_vllm_ready(proxy_model, timeout_seconds=_readiness_timeout_seconds(config))
            logger.info("Started local VllmWorker")
            yield proxy_model


@contextlib.contextmanager
def start_iris_brokered_vllm(
    config: BrokeredVllmSystemConfig,
    runtime: IrisBrokeredVllmRuntimeConfig,
) -> Iterator[RunningModel]:
    """Start Iris child broker/workers and expose a local proxy in the parent job."""

    client = current_client()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("start_iris_brokered_vllm must run inside an Iris job.")
    if job_info.worker_region != runtime.region:
        raise RuntimeError(
            "Brokered vLLM parent, broker, and workers must run in one Iris region. "
            f"Parent region={job_info.worker_region!r}, runtime region={runtime.region!r}."
        )
    run_id = uuid.uuid4().hex[:8]
    broker_name = f"vllm-broker-{run_id}"
    worker_prefix = f"vllm-worker-{run_id}"
    broker_group = client.create_actor_group(
        LocalWorkloadBroker,
        name=broker_name,
        count=1,
        request_lease_timeout_seconds=config.request_lease_timeout_seconds,
        resources=runtime.broker_resources_in_region,
        actor_config=ActorConfig(max_task_retries=0, priority_band=runtime.priority_band),
    )
    worker_jobs: list[JobHandle] = []
    try:
        logger.info("Waiting for LocalWorkloadBroker actor name=%s", broker_name)
        broker_handle = cast(
            WorkloadBroker,
            broker_group.wait_ready(count=1, timeout=runtime.broker_ready_timeout_seconds)[0],
        )
        worker_environment = create_environment(
            extras=runtime.worker_environment_extras,
            env_vars=dict(runtime.worker_env_vars),
        )
        for worker_index in range(config.workers.count):
            job = client.submit(
                JobRequest(
                    name=f"{worker_prefix}-{worker_index}",
                    entrypoint=Entrypoint.from_callable(_run_iris_vllm_worker, args=(config, broker_handle)),
                    resources=runtime.worker_resources_in_region,
                    environment=worker_environment,
                    priority_band=runtime.priority_band,
                )
            )
            worker_jobs.append(job)
            logger.info("Submitted vLLM worker job_id=%s index=%d", job.job_id, worker_index)

        with _start_proxy(config, broker_handle) as running_model:
            _wait_for_brokered_vllm_ready(running_model, timeout_seconds=_readiness_timeout_seconds(config))
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


def _run_iris_vllm_worker(config: BrokeredVllmSystemConfig, broker_handle: WorkloadBroker) -> None:
    configure_logging()
    with remove_tpu_lockfile_on_exit(), start_local_vllm_server(config) as upstream:
        worker = VllmWorker(
            broker=broker_handle,
            upstream=upstream,
            request_timeout_seconds=config.workers.request_timeout_seconds,
        )
        # Worker jobs poll until the parent job terminates them after the eval client exits.
        asyncio.run(worker.run_forever(max_in_flight=config.workers.max_in_flight_per_worker))


@contextlib.contextmanager
def _start_proxy(config: BrokeredVllmSystemConfig, broker: WorkloadBroker) -> Iterator[RunningModel]:
    proxy_config = config.proxy
    with serve_vllm_http_proxy(
        broker=broker,
        model=config.model,
        host=proxy_config.host,
        port=proxy_config.port,
        request_timeout_seconds=proxy_config.request_timeout_seconds,
        readiness_timeout_seconds=_readiness_timeout_seconds(config),
        max_pending_requests=proxy_config.max_pending_requests,
        response_fetch_batch_size=proxy_config.response_fetch_batch_size,
        server_start_timeout_seconds=proxy_config.server_start_timeout_seconds,
        retry_after_seconds=proxy_config.retry_after_seconds,
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


def _validate_resource_zone(resources: ResourceConfig, *, name: str, region: str) -> None:
    if resources.zone is not None and _region_from_zone(resources.zone) != region:
        raise ValueError(f"{name}.zone={resources.zone!r} is outside required region {region!r}.")


def _region_from_zone(zone: str) -> str:
    return zone.rsplit("-", maxsplit=1)[0]


def _readiness_timeout_seconds(config: BrokeredVllmSystemConfig) -> float:
    return config.proxy.readiness_timeout_seconds


def _wait_for_brokered_vllm_ready(running_model: RunningModel, *, timeout_seconds: float) -> None:
    models_url = running_model.endpoint.url("models")
    logger.info("Waiting for brokered vLLM readiness url=%s timeout_seconds=%.1f", models_url, timeout_seconds)
    response = requests.get(models_url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("data"):
        raise RuntimeError(f"No models returned from brokered vLLM readiness check: {str(payload)[:2000]}")
    logger.info("Brokered vLLM readiness check passed url=%s", models_url)
