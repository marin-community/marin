# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start inference workers through Iris."""

import contextlib
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import cast

import requests
from fray.client import JobHandle
from fray.current_client import current_client
from fray.iris_backend import IrisJobHandle
from fray.types import ActorConfig, CpuConfig, Entrypoint, JobRequest, JobStatus
from iris.client import Job, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess
from rigging.connect import proxy_path
from rigging.log_setup import configure_logging

from marin.inference.broker import InferenceBroker
from marin.inference.config import (
    BrokerConfig,
    IrisConfig,
    LevanterEngineConfig,
    ServedModelConfig,
    VllmEngineConfig,
)
from marin.inference.dashboard_server import ServingInfo, bind_serving_socket, build_dashboard_app, serve_app_background
from marin.inference.proxy import serve_inference_proxy
from marin.inference.serve import LocalInferenceSession, local_inference
from marin.inference.types import (
    InferenceRequestProvider,
    InferenceResponseProvider,
    InferenceWorkerMetadata,
    OpenAIEndpoint,
    RunningModel,
)
from marin.inference.worker import InferenceWorker, run_inference_worker

logger = logging.getLogger(__name__)

_TIMEOUT_POLL_SECONDS = 30
_ENDPOINT_READY_POLL_SECONDS = 2.0
_METADATA_BACKEND = "backend"
_METADATA_TENSOR_PARALLEL_SIZE = "tensor_parallel_size"


class RemoteInferenceStartupError(RuntimeError):
    """Remote inference failed before yielding a usable endpoint."""

    def __init__(self, message: str, jobs: tuple[JobHandle, ...]):
        super().__init__(message)
        self.jobs = jobs


@dataclass(frozen=True)
class RemoteInferenceSession:
    model: RunningModel
    jobs: tuple[JobHandle, ...]
    endpoint_name: str | None
    streaming: bool
    tensor_parallel_size: int
    backend_name: str
    iris_job: Job | None = None

    def resolve_model(self) -> RunningModel:
        """Resolve the current address after a direct worker restart."""

        if self.endpoint_name is None:
            return self.model
        endpoints = iris_ctx().client._cluster_client.list_endpoints(self.endpoint_name, exact=True)
        if not endpoints:
            raise RuntimeError(f"Inference endpoint {self.endpoint_name!r} is not registered")
        endpoint = replace(
            self.model.endpoint,
            base_url=f"{endpoints[0].address.rstrip('/')}/v1",
        )
        return replace(self.model, endpoint=endpoint)


@dataclass(frozen=True)
class IrisServiceConfig:
    model: ServedModelConfig
    engine: VllmEngineConfig | LevanterEngineConfig
    iris: IrisConfig
    endpoint_name: str
    instances: int = 1
    broker: BrokerConfig | None = None
    access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE
    timeout_hours: float = 24.0
    controller_proxy_timeout_seconds: float = 600.0
    port_name: str | None = "http"

    def __post_init__(self) -> None:
        if self.instances <= 0:
            raise ValueError("instances must be positive")


def _broker_config(instances: int, broker: BrokerConfig | None) -> BrokerConfig | None:
    if instances <= 0:
        raise ValueError("instances must be positive")
    if broker is not None:
        return broker
    if instances > 1:
        return BrokerConfig()
    return None


def _accelerator_label(iris: IrisConfig) -> str:
    device = iris.worker_resources.device
    if isinstance(device, CpuConfig):
        raise ValueError("Inference workers require an accelerator")
    if device.kind == "gpu":
        return f"{device.variant}x{device.chip_count()}"
    return device.variant


def _resolved_model(model: ServedModelConfig, iris: IrisConfig) -> tuple[ServedModelConfig, int]:
    # Keep model-cache and Transformers imports inside accelerator workers.
    from marin.inference.model_preparation import (  # noqa: PLC0415
        read_attention_heads,
        resolve_model_path,
        select_tensor_parallel_size,
    )

    model_path = model.model_path or resolve_model_path(model.model, iris.cache_ttl_days)
    num_chips = iris.worker_resources.device.chip_count()
    tensor_parallel_size = model.tensor_parallel_size
    if tensor_parallel_size is None:
        num_attention_heads, num_key_value_heads = read_attention_heads(model_path)
        tensor_parallel_size = select_tensor_parallel_size(num_attention_heads, num_chips, num_key_value_heads)
    return replace(model, model_path=model_path, tensor_parallel_size=tensor_parallel_size), num_chips


@contextlib.contextmanager
def _prepared_local_inference(
    model: ServedModelConfig,
    engine: VllmEngineConfig | LevanterEngineConfig,
    iris: IrisConfig,
) -> Iterator[LocalInferenceSession]:
    resolved_model, num_chips = _resolved_model(model, iris)
    with local_inference(resolved_model, engine, num_chips=num_chips) as session:
        yield session


def _server_root(model: RunningModel) -> str:
    return model.endpoint.base_url.removesuffix("/v1")


def _detect_chat_support(model: RunningModel) -> bool:
    try:
        response = requests.post(
            model.endpoint.url("chat/completions"),
            json={"model": model.endpoint.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            timeout=60,
        )
    except requests.RequestException as exc:
        logger.warning("Chat-support probe failed (%s); defaulting to completion mode", exc)
        return False
    return response.status_code == 200


@contextlib.contextmanager
def _register_dashboard(
    service: IrisServiceConfig,
    model: RunningModel,
    *,
    tensor_parallel_size: int,
    backend_name: str,
    streaming: bool,
) -> Iterator[None]:
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("Iris service must run inside an Iris job")
    ctx = iris_ctx()
    port = ctx.get_port(service.port_name) if service.port_name is not None else 0
    serving_socket = bind_serving_socket(job_info.advertise_host, port)
    serving_port = serving_socket.getsockname()[1]
    has_chat_template = service.model.chat_template_content is not None or _detect_chat_support(model)
    info = ServingInfo(
        model=model.endpoint.model,
        backend=backend_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=service.model.max_model_len,
        dtype=service.model.dtype,
        has_chat_template=has_chat_template,
        tpu_type=_accelerator_label(service.iris),
        endpoint=service.endpoint_name,
        streaming=streaming,
    )
    app = build_dashboard_app(
        upstream_base_url=_server_root(model),
        model_id=model.endpoint.model,
        info=info,
        request_timeout_seconds=service.controller_proxy_timeout_seconds,
    )
    with serve_app_background(app, serving_socket):
        address = f"http://{job_info.advertise_host}:{serving_port}"
        metadata = {
            "model": model.endpoint.model,
            "kind": "marin-serve",
            _METADATA_BACKEND: backend_name,
            "accelerator": _accelerator_label(service.iris),
            _METADATA_TENSOR_PARALLEL_SIZE: str(tensor_parallel_size),
            "streaming": str(streaming).lower(),
            PROXY_TIMEOUT_METADATA_KEY: str(service.controller_proxy_timeout_seconds),
        }
        endpoint_id = ctx.registry.register(service.endpoint_name, address, metadata, access=service.access)
        logger.info(
            "Registered inference endpoint name=%s address=%s id=%s proxy_path=%s",
            service.endpoint_name,
            address,
            endpoint_id,
            proxy_path(service.endpoint_name),
        )
        try:
            yield
        finally:
            try:
                ctx.registry.unregister(endpoint_id)
            except Exception:
                logger.warning("Failed to unregister inference endpoint id=%s", endpoint_id, exc_info=True)


def _block_until_timeout(session: LocalInferenceSession, timeout_hours: float) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        session.check_alive()
        time.sleep(_TIMEOUT_POLL_SECONDS)


def _block_remote_until_timeout(session: RemoteInferenceSession, timeout_hours: float) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        for job in session.jobs:
            status = job.status()
            if JobStatus.finished(status):
                raise RuntimeError(f"Inference job {job.job_id} finished unexpectedly with status {status}")
        time.sleep(_TIMEOUT_POLL_SECONDS)


def run_iris_service(service: IrisServiceConfig) -> None:
    """Run a long-lived direct or brokered Iris endpoint."""

    configure_logging()
    broker = _broker_config(service.instances, service.broker)
    if broker is None:
        with _prepared_local_inference(service.model, service.engine, service.iris) as local_session:
            tensor_parallel_size = cast(int, local_session.tensor_parallel_size)
            with _register_dashboard(
                service,
                local_session.model,
                tensor_parallel_size=tensor_parallel_size,
                backend_name=local_session.backend_name,
                streaming=True,
            ):
                _block_until_timeout(local_session, service.timeout_hours)
        return

    with remote_inference(
        service.model,
        service.engine,
        service.iris,
        instances=service.instances,
        broker=broker,
    ) as session:
        with _register_dashboard(
            service,
            session.model,
            tensor_parallel_size=session.tensor_parallel_size,
            backend_name=session.backend_name,
            streaming=session.streaming,
        ):
            _block_remote_until_timeout(session, service.timeout_hours)


def _wait_for_endpoint(job: JobHandle, endpoint_name: str, timeout_seconds: float) -> tuple[str, dict[str, str]]:
    ctx = iris_ctx()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        endpoints = ctx.client._cluster_client.list_endpoints(endpoint_name, exact=True)
        if endpoints:
            return endpoints[0].address, dict(endpoints[0].metadata)
        if job.status().value in {"succeeded", "failed", "stopped"}:
            raise RuntimeError(f"Inference job {job.job_id} finished before registering {endpoint_name!r}")
        time.sleep(_ENDPOINT_READY_POLL_SECONDS)
    raise TimeoutError(f"Timed out waiting for inference endpoint {endpoint_name!r}")


@contextlib.contextmanager
def remote_inference(
    model: ServedModelConfig,
    engine: VllmEngineConfig | LevanterEngineConfig,
    iris: IrisConfig,
    *,
    instances: int = 1,
    broker: BrokerConfig | None = None,
) -> Iterator[RemoteInferenceSession]:
    """Start inference on Iris, adding broker transport when requested or required."""

    if get_job_info() is None:
        raise RuntimeError("remote_inference must run inside an Iris job")
    resolved_broker = _broker_config(instances, broker)
    if resolved_broker is None:
        with _start_direct_inference(model, engine, iris) as session:
            yield session
        return
    with _start_brokered_inference(model, engine, iris, instances, resolved_broker) as session:
        yield session


@contextlib.contextmanager
def _start_direct_inference(
    model: ServedModelConfig,
    engine: VllmEngineConfig | LevanterEngineConfig,
    iris: IrisConfig,
) -> Iterator[RemoteInferenceSession]:
    run_id = uuid.uuid4().hex[:8]
    endpoint_name = f"/serve/inference-{run_id}"
    service = IrisServiceConfig(
        model=model,
        engine=engine,
        iris=iris,
        endpoint_name=endpoint_name,
        timeout_hours=24 * 365,
        port_name=None,
    )
    job = current_client().submit(
        JobRequest(
            name=f"inference-{run_id}",
            entrypoint=Entrypoint.from_callable(run_iris_service, args=(service,)),
            resources=iris.worker_resources,
            environment=iris.worker_environment,
            max_retries_failure=iris.max_retries_failure,
            max_retries_preemption=iris.max_retries_preemption,
            priority=iris.priority,
        )
    )
    try:
        try:
            address, metadata = _wait_for_endpoint(
                job,
                endpoint_name,
                timeout_seconds=iris.endpoint_ready_timeout_seconds,
            )
            tensor_parallel_size = int(metadata[_METADATA_TENSOR_PARALLEL_SIZE])
            backend_name = metadata[_METADATA_BACKEND]
        except Exception as exc:
            raise RemoteInferenceStartupError(
                f"Inference job {job.job_id} failed to register a usable endpoint: {exc}",
                jobs=(job,),
            ) from exc
        yield RemoteInferenceSession(
            model=RunningModel(
                endpoint=OpenAIEndpoint(base_url=f"{address.rstrip('/')}/v1", model=model.model),
                tokenizer=model.tokenizer,
            ),
            jobs=(job,),
            endpoint_name=endpoint_name,
            streaming=True,
            tensor_parallel_size=tensor_parallel_size,
            backend_name=backend_name,
            iris_job=cast(IrisJobHandle, job).iris_job,
        )
    finally:
        _terminate_job(job)


def _run_broker_worker(
    worker_id: str,
    model: ServedModelConfig,
    engine: VllmEngineConfig | LevanterEngineConfig,
    iris: IrisConfig,
    broker_config: BrokerConfig,
    broker: InferenceRequestProvider,
) -> None:
    configure_logging()
    with _prepared_local_inference(model, engine, iris) as local_session:
        tensor_parallel_size = cast(int, local_session.tensor_parallel_size)
        broker.register_worker(
            worker_id,
            InferenceWorkerMetadata(
                tensor_parallel_size=tensor_parallel_size,
                backend_name=local_session.backend_name,
            ),
        )
        worker = InferenceWorker(
            broker=broker,
            upstream=local_session.model,
            request_timeout_seconds=broker_config.worker.request_timeout_seconds,
        )
        with run_inference_worker(worker, max_in_flight=broker_config.worker.max_in_flight):
            while True:
                local_session.check_alive()
                time.sleep(_TIMEOUT_POLL_SECONDS)


@contextlib.contextmanager
def _start_brokered_inference(
    model: ServedModelConfig,
    engine: VllmEngineConfig | LevanterEngineConfig,
    iris: IrisConfig,
    instances: int,
    broker: BrokerConfig,
) -> Iterator[RemoteInferenceSession]:
    client = current_client()
    job_info = get_job_info()
    assert job_info is not None
    run_id = uuid.uuid4().hex[:8]
    broker_group = None
    worker_jobs: list[JobHandle] = []
    started = False
    try:
        broker_group = client.create_actor_group(
            InferenceBroker,
            name=f"inference-broker-{run_id}",
            count=1,
            request_lease_timeout_seconds=broker.request_lease_timeout_seconds,
            resources=broker.broker_resources,
            actor_config=ActorConfig(max_task_retries=0, priority=iris.priority),
        )
        broker_handle = broker_group.wait_ready(count=1, timeout=broker.broker_ready_timeout_seconds)[0]
        request_provider = cast(InferenceRequestProvider, broker_handle)
        response_provider = cast(InferenceResponseProvider, broker_handle)
        for worker_index in range(instances):
            worker_id = f"inference-worker-{run_id}-{worker_index}"
            job = client.submit(
                JobRequest(
                    name=worker_id,
                    entrypoint=Entrypoint.from_callable(
                        _run_broker_worker,
                        args=(worker_id, model, engine, iris, broker, request_provider),
                    ),
                    resources=iris.worker_resources,
                    environment=iris.worker_environment,
                    max_retries_failure=broker.max_retries_failure,
                    max_retries_preemption=broker.max_retries_preemption,
                    priority=iris.priority,
                )
            )
            worker_jobs.append(job)
        proxy = broker.proxy
        with serve_inference_proxy(
            broker=response_provider,
            model=model.model,
            host=job_info.advertise_host,
            port=proxy.port,
            request_timeout_seconds=proxy.request_timeout_seconds,
            readiness_timeout_seconds=proxy.readiness_timeout_seconds,
            max_pending_requests=proxy.max_pending_requests,
            response_fetch_batch_size=proxy.response_fetch_batch_size,
            server_start_timeout_seconds=proxy.server_start_timeout_seconds,
            ignored_request_fields=proxy.ignored_request_fields,
        ) as running_model:
            response = requests.get(running_model.endpoint.url("models"), timeout=proxy.readiness_timeout_seconds)
            response.raise_for_status()
            metadata_by_worker = request_provider.worker_metadata()
            if not metadata_by_worker:
                raise RuntimeError("Brokered inference became ready before a worker registered metadata")
            worker_metadata = next(iter(metadata_by_worker.values()))
            started = True
            yield RemoteInferenceSession(
                model=replace(running_model, tokenizer=model.tokenizer),
                jobs=tuple(worker_jobs),
                endpoint_name=None,
                streaming=False,
                tensor_parallel_size=worker_metadata.tensor_parallel_size,
                backend_name=worker_metadata.backend_name,
            )
    except Exception as exc:
        if not started:
            raise RemoteInferenceStartupError(
                f"Brokered inference failed to start: {exc}",
                jobs=tuple(worker_jobs),
            ) from exc
        raise
    finally:
        for job in worker_jobs:
            _terminate_job(job)
        if broker_group is not None:
            try:
                broker_group.shutdown()
            except Exception:
                logger.warning("Failed to shut down inference broker actor", exc_info=True)


def _terminate_job(job: JobHandle) -> None:
    try:
        job.terminate()
    except Exception:
        logger.warning("Failed to terminate inference job job_id=%s", job.job_id, exc_info=True)
