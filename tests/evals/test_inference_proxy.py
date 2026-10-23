# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import cast

import httpx
import marin.inference.iris as iris_module
import pytest
from fray.client import JobHandle
from fray.types import JobStatus, ResourceConfig, create_environment
from iris.cluster.types import EndpointAccess
from iris.resources.state import JobState, TaskState
from marin.execution.lazy import lower
from marin.inference.broker import InferenceBroker
from marin.inference.config import (
    BrokerConfig,
    InferenceProxyConfig,
    InferenceWorkerConfig,
    IrisConfig,
    RemoteInferenceConfig,
    ServedModelConfig,
    ServingGeometry,
    VllmEngineConfig,
)
from marin.inference.iris import (
    InferenceBackendState,
    RemoteInferenceSession,
    RemoteInferenceStartupError,
    remote_inference,
)
from marin.inference.proxy import InferenceProxy, serve_inference_proxy
from marin.inference.types import (
    STREAM_BUFFER_BYTES,
    WORKER_LEASE_SECONDS,
    InferenceRequest,
    InferenceResponse,
    InferenceWorkerMetadata,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    OpenAIEndpoint,
    RunningModel,
    StreamChunk,
    StreamEnd,
    StreamError,
    StreamStart,
)
from marin.inference.worker import InferenceWorker, run_inference_worker
from rigging.timing import ExponentialBackoff

from experiments.evals.federated_inference_proxy_demo import _redacted_url
from experiments.evals.served_qwen3 import QWEN3_GPU_EVAL_RESULTS
from tests.evals.openai_stub import (
    DeterministicOpenAIStub,
    assert_completions_scoring_contract,
    serve_deterministic_openai_stub,
)

BROKER_LEASE_TIMEOUT_SECONDS = 300.0


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://iris.example/proxy/t/secret-token/serve.inference/v1",
            "https://iris.example/proxy/t/<redacted>/serve.inference/v1",
        ),
        (
            "https://iris.example/proxy/t/cluster=cw-us-west-04a/secret-token/serve.inference/v1",
            "https://iris.example/proxy/t/cluster=cw-us-west-04a/<redacted>/serve.inference/v1",
        ),
    ],
)
def test_capability_url_redaction(url: str, expected: str) -> None:
    assert _redacted_url(url) == expected


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def _json_payload(payload: bytes) -> dict:
    return json.loads(payload)


def test_brokered_gpu_eval_lowers_with_symbolic_worker_resources() -> None:
    step = lower(QWEN3_GPU_EVAL_RESULTS)
    fingerprint = json.loads(step.fingerprint_payload)

    # Artifact lowering replaces runtime resources with a symbolic value. Backend validation must
    # remain at the serving boundary, where the concrete ResourceConfig is available.
    assert fingerprint["inference"]["iris"]["worker_resources"] == "<worker_resources>"
    assert fingerprint["inference"]["instances"] == 1
    assert fingerprint["inference"]["engine"]["launcher"] == "cuda"


def test_remote_topology_selection() -> None:
    explicit = BrokerConfig()

    assert iris_module._broker_config(1, None) is None
    assert isinstance(iris_module._broker_config(2, None), BrokerConfig)
    assert iris_module._broker_config(1, explicit) is explicit
    with pytest.raises(ValueError, match="instances must be positive"):
        iris_module._broker_config(0, None)


@pytest.mark.parametrize("pipeline_size", [1, 2])
def test_remote_inference_uses_controller_minted_federated_capability_url(monkeypatch, pipeline_size) -> None:
    class _Job:
        job_id = "serve-job"
        iris_job = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True

    minted: list[tuple[str, object]] = []
    submitted = []
    job = _Job()
    iris_client = SimpleNamespace(
        mint_endpoint_token=lambda name, ttl: minted.append((name, ttl))
        or SimpleNamespace(
            token="secret-token",
            capability_url="https://iris.example/proxy/t/cluster=cw-us-west-04a/secret-token/serve.inference",
        ),
    )
    monkeypatch.setattr(iris_module, "get_job_info", lambda: SimpleNamespace())
    monkeypatch.setattr(
        iris_module,
        "current_client",
        lambda: SimpleNamespace(submit=lambda request: submitted.append(request) or job),
    )
    monkeypatch.setattr(iris_module, "iris_ctx", lambda: SimpleNamespace(client=iris_client))
    monkeypatch.setattr(
        iris_module,
        "_wait_for_endpoint",
        lambda *_args, **_kwargs: (
            "http://10.0.0.1:8000",
            {
                "tensor_parallel_size": "8",
                "backend": "vllm",
                "data_parallel_size": "1",
                "pipeline_parallel_size": str(pipeline_size),
                "task_count": str(pipeline_size),
                "max_model_len": "4096",
            },
        ),
    )
    iris = IrisConfig(
        worker_resources=ResourceConfig.with_gpu("H100", count=8, replicas=pipeline_size),
        worker_environment=create_environment(docker_image="test"),
        serving_geometry=ServingGeometry(8, 1, pipeline_size, 8),
    )

    with remote_inference(
        RemoteInferenceConfig(
            model=ServedModelConfig(
                weights="physical-model",
                tensor_parallel_size=8,
                api_model="public-model",
                tokenizer="Qwen/Qwen3-0.6B",
            ),
            engine=VllmEngineConfig(),
            iris=iris,
            capability_origin="https://iris.example",
        ),
    ) as session:
        model = session.model

    (request,) = submitted
    assert request.replicas == pipeline_size
    assert session.effective_serving is not None
    assert session.effective_serving.task_count == pipeline_size
    assert session.effective_serving.max_model_len == 4096
    (service,) = request.entrypoint.callable_entrypoint.args
    assert service.controller_proxy_timeout_seconds > 1800
    assert service.endpoint_name == minted[0][0]
    assert "physical-model" not in service.endpoint_name
    assert model.endpoint.base_url == (
        "https://iris.example/proxy/t/cluster=cw-us-west-04a/secret-token/serve.inference/v1"
    )
    assert "10.0.0.1" not in model.endpoint.base_url
    assert model.endpoint.model == "public-model"
    assert model.tokenizer == "Qwen/Qwen3-0.6B"
    assert "secret-token" not in repr(model.endpoint)
    assert job.terminated


def test_remote_inference_reports_direct_startup_job(monkeypatch) -> None:
    class _FailedJob:
        job_id = "failed-serve"
        terminated = False

        def status(self) -> JobStatus:
            return JobStatus.FAILED

        def terminate(self) -> None:
            self.terminated = True

    job = _FailedJob()
    requests = []
    monkeypatch.setattr(iris_module, "get_job_info", lambda: SimpleNamespace())
    monkeypatch.setattr(
        iris_module,
        "current_client",
        lambda: SimpleNamespace(submit=lambda request: requests.append(request) or job),
    )
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(client=SimpleNamespace(list_endpoints=lambda *_args, **_kwargs: [])),
    )
    iris = IrisConfig(
        worker_resources=ResourceConfig.with_tpu("v6e-4"),
        worker_environment=create_environment(extras=["tpu"]),
    )

    with pytest.raises(RemoteInferenceStartupError) as exc_info:
        with remote_inference(
            RemoteInferenceConfig(
                model=ServedModelConfig(weights="gpt2"),
                engine=VllmEngineConfig(),
                iris=iris,
            )
        ):
            pass

    assert exc_info.value.jobs == (job,)
    assert job.terminated
    assert len(requests) == 1


@dataclass
class _SessionJob:
    job_status: JobStatus
    job_id: str = "/tester/inference"

    def status(self) -> JobStatus:
        return self.job_status


def _remote_session(job_status: JobStatus = JobStatus.RUNNING) -> RemoteInferenceSession:
    return RemoteInferenceSession(
        model=RunningModel(endpoint=OpenAIEndpoint(base_url="https://iris.example/capability/v1", model="model")),
        jobs=(_SessionJob(job_status),),
        endpoint_name="/serve/inference",
        endpoint_health_timeout_seconds=1800.0,
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
    )


@pytest.mark.parametrize(
    ("task_state", "endpoint_count", "expected"),
    [
        (TaskState.PENDING, 1, InferenceBackendState.RECOVERING),
        (TaskState.RUNNING, 0, InferenceBackendState.RECOVERING),
        (TaskState.RUNNING, 1, InferenceBackendState.READY),
    ],
)
def test_direct_inference_session_reports_backend_state(
    task_state: TaskState,
    endpoint_count: int,
    expected: InferenceBackendState,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                job_status=lambda _job_id: SimpleNamespace(
                    state=JobState.RUNNING,
                    task_count=1,
                    task_state_counts={task_state: 1},
                ),
                list_endpoint_instances=lambda _endpoint_name: [SimpleNamespace()] * endpoint_count,
            )
        ),
    )
    session = _remote_session()

    assert session.backend_state() is expected


@pytest.mark.parametrize(
    ("task_state", "display_worker", "attempt_worker", "node_name", "placed"),
    [
        (TaskState.PENDING, "peer-label", "", "node-1", False),
        (TaskState.BUILDING, "", "", "", False),
        (TaskState.BUILDING, "peer-label", "", "", False),
        (TaskState.BUILDING, "peer-label", "", "node-1", True),
        (TaskState.ASSIGNED, "worker-1", "worker-1", "", True),
        (TaskState.RUNNING, "", "", "", True),
    ],
)
def test_inference_endpoint_wait_distinguishes_queued_and_placed_tasks(
    task_state: TaskState, display_worker: str, attempt_worker: str, node_name: str, placed: bool, monkeypatch
) -> None:
    class FixedDeadline:
        def __init__(self, seconds: float) -> None:
            self.seconds = seconds

        def expired(self) -> bool:
            return self.seconds == 0

    endpoint_probes = 0

    def list_endpoint_instances(_endpoint_name: str):
        nonlocal endpoint_probes
        endpoint_probes += 1
        if endpoint_probes > 1:
            return [SimpleNamespace(address="https://inference.example", metadata={})]
        return []

    monkeypatch.setattr(iris_module.Deadline, "from_seconds", FixedDeadline)
    monkeypatch.setattr(iris_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                list_endpoint_instances=list_endpoint_instances,
                list_tasks=lambda _job_id: [
                    SimpleNamespace(
                        state=task_state,
                        worker_id=display_worker,
                        current_attempt_number=0,
                        attempts=(SimpleNamespace(attempt_number=0, worker_id=attempt_worker, node_name=node_name),),
                    ),
                ],
            )
        ),
    )
    job = cast(JobHandle, _SessionJob(JobStatus.RUNNING))

    if placed:
        with pytest.raises(TimeoutError):
            iris_module._wait_for_endpoint(job, "/serve/inference", timeout_seconds=0)
    else:
        assert iris_module._wait_for_endpoint(job, "/serve/inference", timeout_seconds=0) == (
            "https://inference.example",
            {},
        )


def test_inference_endpoint_wait_preserves_startup_budget_across_gate_and_requeue(monkeypatch) -> None:
    now = 0.0
    delays = iter((6, 1, 3, 7, 1, 4))

    def sleep(_seconds: float) -> None:
        nonlocal now
        now += next(delays)

    def task(state: TaskState, current_attempt: int, *nodes: str):
        return SimpleNamespace(
            state=state,
            worker_id="",
            current_attempt_number=current_attempt,
            attempts=tuple(
                SimpleNamespace(attempt_number=i, worker_id="", node_name=node) for i, node in enumerate(nodes)
            ),
        )

    snapshots = iter(
        (
            task(TaskState.BUILDING, 0, ""),
            task(TaskState.BUILDING, 0, ""),
            task(TaskState.BUILDING, 0, "node-a"),
            task(TaskState.PENDING, 0, "node-a"),
            task(TaskState.BUILDING, 1, "node-a", ""),
            task(TaskState.BUILDING, 1, "node-a", "node-b"),
        )
    )
    probes = 0

    def list_endpoint_instances(_endpoint_name: str):
        nonlocal probes
        probes += 1
        if probes == 7:
            return [SimpleNamespace(address="https://inference.example", metadata={})]
        return []

    monkeypatch.setattr(iris_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(iris_module.time, "sleep", sleep)
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                list_endpoint_instances=list_endpoint_instances,
                list_tasks=lambda _job_id: [next(snapshots)],
            )
        ),
    )
    job = cast(JobHandle, _SessionJob(JobStatus.RUNNING))

    assert iris_module._wait_for_endpoint(job, "/serve/inference", timeout_seconds=5) == (
        "https://inference.example",
        {},
    )


def test_inference_endpoint_wait_requeue_preserves_placement_deadline(monkeypatch) -> None:
    now = 0.0
    snapshots = iter(
        (
            SimpleNamespace(state=TaskState.BUILDING, worker_id="", current_attempt_number=0, attempts=()),
            SimpleNamespace(
                state=TaskState.BUILDING,
                worker_id="",
                current_attempt_number=0,
                attempts=(SimpleNamespace(attempt_number=0, worker_id="", node_name="node-a"),),
            ),
            SimpleNamespace(state=TaskState.PENDING, worker_id="", current_attempt_number=0, attempts=()),
        )
    )

    def sleep(_seconds: float) -> None:
        nonlocal now
        now += 6

    monkeypatch.setattr(iris_module, "_ENDPOINT_PLACEMENT_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(iris_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(iris_module.time, "sleep", sleep)
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                list_endpoint_instances=lambda _endpoint_name: [],
                list_tasks=lambda _job_id: [next(snapshots)],
            )
        ),
    )
    job = cast(JobHandle, _SessionJob(JobStatus.RUNNING))

    with pytest.raises(TimeoutError, match="to be placed"):
        iris_module._wait_for_endpoint(job, "/serve/inference", timeout_seconds=5)


def test_inference_recovery_stops_when_job_becomes_terminal(monkeypatch) -> None:
    session = _remote_session(JobStatus.FAILED)
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(job_status=lambda *_args, **_kwargs: SimpleNamespace(state=JobState.FAILED, tasks=[]))
        ),
    )

    with pytest.raises(RuntimeError):
        session.wait_until_ready()


def test_inference_recovery_waits_for_tasks_and_routed_endpoint(monkeypatch) -> None:
    task_pending = True
    probe_statuses: list[int] = []

    def job_status(_job_id):
        nonlocal task_pending
        state = TaskState.PENDING if task_pending else TaskState.RUNNING
        task_pending = False
        return SimpleNamespace(state=JobState.RUNNING, task_count=1, task_state_counts={state: 1})

    def probe_endpoint(_url, timeout):
        del timeout
        status_code = 502 if not probe_statuses else 200
        probe_statuses.append(status_code)
        return SimpleNamespace(status_code=status_code)

    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                job_status=job_status,
                list_endpoint_instances=lambda _endpoint_name: [SimpleNamespace()],
                resolve_endpoint=lambda _name: "https://inference.example",
            )
        ),
    )
    monkeypatch.setattr(iris_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(iris_module.requests, "get", probe_endpoint)

    session = _remote_session()
    session.wait_until_ready()

    assert probe_statuses == [502, 200]


def test_inference_recovery_times_out_when_routed_endpoint_stays_unhealthy(monkeypatch) -> None:
    class ExpiredDeadline:
        def __init__(self) -> None:
            self.checks = 0

        def remaining_seconds(self) -> float:
            return 5.0

        def raise_if_expired(self, message: str) -> None:
            self.checks += 1
            if self.checks > 1:
                raise TimeoutError(message)

    probes: list[str] = []

    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(
                job_status=lambda *_args: SimpleNamespace(
                    state=JobState.RUNNING,
                    task_count=1,
                    task_state_counts={TaskState.RUNNING: 1},
                ),
                list_endpoint_instances=lambda _endpoint_name: [SimpleNamespace()],
                resolve_endpoint=lambda _name: "https://replacement.example",
            )
        ),
    )
    monkeypatch.setattr(
        iris_module.Deadline,
        "from_seconds",
        lambda _timeout: ExpiredDeadline(),
    )
    monkeypatch.setattr(
        iris_module.requests,
        "get",
        lambda url, **_kwargs: probes.append(url) or SimpleNamespace(status_code=502),
    )

    with pytest.raises(TimeoutError, match=r"did not become healthy within 1800\.0s"):
        _remote_session().wait_until_ready()
    assert probes == ["https://replacement.example/v1/models"]


def test_broker_config_rejects_invalid_timeout_ordering() -> None:
    with pytest.raises(ValueError, match=r"worker\.request_timeout_seconds"):
        BrokerConfig(worker=InferenceWorkerConfig(request_timeout_seconds=240))


@dataclass
class MockInferenceCluster:
    broker: InferenceBroker
    model: str
    upstream: DeterministicOpenAIStub
    proxy: RunningModel


@pytest.fixture
def mock_cluster() -> Iterator[MockInferenceCluster]:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with serve_deterministic_openai_stub() as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = InferenceWorker(worker_id="test-worker", broker=broker, upstream=upstream, request_timeout_seconds=5)
        with (
            _serve_inference_proxy(
                broker=broker,
                model=upstream_stub.model,
                request_timeout_seconds=5,
                readiness_timeout_seconds=5,
            ) as proxy,
            run_inference_worker(worker, max_in_flight=InferenceWorkerConfig().max_in_flight),
        ):
            yield MockInferenceCluster(
                broker=broker,
                model=upstream_stub.model,
                upstream=upstream_stub,
                proxy=proxy,
            )


def test_broker_stream_expiry_never_replays_a_started_response() -> None:
    now = [0.0]
    broker = InferenceBroker(request_lease_timeout_seconds=10, clock=lambda: now[0])
    request = InferenceRequest("stream", "POST", "/v1/chat/completions", b"{}", streaming=True)
    broker.submit_request(request)
    [old] = broker.fetch_requests(max_items=1)
    now[0] = 11
    [replacement] = broker.fetch_requests(max_items=1)
    assert not broker.submit_stream_event(StreamStart("stream", old.lease_id, 0, 200, ()))
    start = StreamStart("stream", replacement.lease_id, 0, 200, ())
    assert broker.submit_stream_event(start)
    assert broker.submit_stream_event(StreamChunk("stream", replacement.lease_id, 1, b"first"))
    assert not broker.submit_stream_event(StreamChunk("stream", replacement.lease_id, 1, b"duplicate"))
    now[0] = 22
    events = broker.fetch_responses(max_items=10)
    assert events[:2] == [start, StreamChunk("stream", replacement.lease_id, 1, b"first")]
    assert isinstance(events[2], StreamError)
    assert broker.fetch_requests(max_items=1) == []
    assert not broker.submit_stream_event(StreamEnd("stream", replacement.lease_id, 2))
    assert broker.pending() == []


def test_broker_cancellation_removes_queued_and_leased_requests() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=10)
    for name in ("leased", "queued"):
        broker.submit_request(InferenceRequest(name, "POST", "/v1/completions", b"{}"))
    [leased] = broker.fetch_requests(max_items=1)
    broker.cancel_request("leased")
    broker.cancel_request("queued")
    broker.submit_responses([LeasedInferenceResponse(leased.lease_id, InferenceResponse("leased", 200, b"late"))])
    assert broker.fetch_requests(max_items=1) == []
    assert broker.fetch_responses(max_items=10) == []
    assert broker.size() == 0


def test_stream_buffer_overflow_fails_without_replaying() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=10)
    broker.submit_request(InferenceRequest("overflow", "POST", "/v1/completions", b"{}", streaming=True))
    [lease] = broker.fetch_requests(max_items=1)
    assert broker.submit_stream_event(StreamStart("overflow", lease.lease_id, 0, 200, ()))
    assert not broker.submit_stream_event(StreamChunk("overflow", lease.lease_id, 1, b"x" * (STREAM_BUFFER_BYTES + 1)))
    events = broker.fetch_responses(max_items=10)
    assert isinstance(events[-1], StreamError)
    assert not any(isinstance(event, StreamChunk) for event in events)
    assert broker.fetch_requests(max_items=1) == []
    assert broker.pending() == []


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_preserves_compression_only_for_raw_streams(streaming) -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=10)
    broker.submit_request(InferenceRequest("gzip", "GET", "/metrics", b"", streaming=streaming))
    [lease] = broker.fetch_requests(max_items=1)
    compressed = gzip.compress(b"metric_total 1\n")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=httpx.ByteStream(compressed),
        )
    )
    worker = InferenceWorker(
        worker_id="test-worker",
        broker=broker,
        upstream=RunningModel(OpenAIEndpoint("http://upstream/v1", "test")),
        request_timeout_seconds=5,
    )
    with httpx.Client(transport=transport) as client:
        result = worker._forward_one(client, lease)
    if streaming:
        events = broker.fetch_responses(max_items=10)
        assert isinstance(events[0], StreamStart)
        assert dict(events[0].headers)["content-encoding"] == "gzip"
        assert b"".join(event.payload for event in events if isinstance(event, StreamChunk)) == compressed
    else:
        assert result is not None
        assert result.response.payload == b"metric_total 1\n"
        assert "content-encoding" not in dict(result.response.headers)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "send", "deadline"])
async def test_stream_response_cleans_up_when_http_delivery_stops(failure) -> None:
    release = threading.Event()
    disconnected = asyncio.Event()
    blocked = asyncio.Event()
    broker = InferenceBroker(request_lease_timeout_seconds=10)
    with _streaming_upstream(release) as upstream:
        worker = InferenceWorker(worker_id="test-worker", broker=broker, upstream=upstream, request_timeout_seconds=5)
        with (
            run_inference_worker(worker, max_in_flight=1),
            InferenceProxy(
                broker=broker,
                model="test",
                request_timeout_seconds=5,
                readiness_timeout_seconds=5,
                max_pending_requests=4,
                response_fetch_batch_size=16,
            ) as proxy,
        ):
            try:
                response = await proxy.forward_raw_request(
                    "/v1/chat/completions",
                    b"{}",
                    method="POST",
                    streaming=True,
                    timeout_seconds=0.5 if failure == "deadline" else 5,
                )

                async def receive():
                    await disconnected.wait()
                    return {"type": "http.disconnect"}

                async def send(message):
                    if message["type"] != "http.response.body":
                        return
                    if failure == "send":
                        raise OSError("client connection closed")
                    if failure == "disconnect":
                        disconnected.set()
                    await blocked.wait()

                try:
                    await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
                except (OSError, ExceptionGroup):
                    assert failure == "send"
            finally:
                release.set()
            assert broker.pending() == []
            assert broker.fetch_requests(max_items=1) == []


def test_broker_readiness_expires_without_evaluator_iris_polling() -> None:
    now = [0.0]
    broker = InferenceBroker(request_lease_timeout_seconds=100, clock=lambda: now[0])
    session = RemoteInferenceSession(
        model=RunningModel(OpenAIEndpoint("http://broker/v1", "test")),
        jobs=(),
        endpoint_name="broker",
        endpoint_health_timeout_seconds=5,
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
        broker=broker,
    )
    assert session.backend_state() is InferenceBackendState.RECOVERING
    broker.register_worker("worker", InferenceWorkerMetadata(1, "vllm"))
    assert session.backend_state() is InferenceBackendState.READY
    now[0] += WORKER_LEASE_SECONDS + 1
    assert session.backend_state() is InferenceBackendState.RECOVERING
    assert session.preemption_count() == 1
    broker.heartbeat("worker")
    assert session.backend_state() is InferenceBackendState.READY
    broker.finish()
    assert session.backend_state() is InferenceBackendState.FINISHED


@pytest.mark.asyncio
async def test_proxy_distinguishes_liveness_from_worker_readiness() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=10)
    with InferenceProxy(
        broker=broker,
        model="test",
        request_timeout_seconds=5,
        readiness_timeout_seconds=5,
        max_pending_requests=4,
        response_fetch_batch_size=16,
    ) as proxy:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://broker") as client:
            assert (await client.get("/health")).status_code == 200
            response = await client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "recovering"
            broker.register_worker("worker", InferenceWorkerMetadata(1, "vllm"))
            assert (await client.get("/ready")).status_code == 200
            broker.finish()
            response = await client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "finished"


@pytest.mark.asyncio
async def test_disconnect_before_headers_cancels_the_broker_request() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=10)
    received_body = False

    async def receive():
        nonlocal received_body
        if not received_body:
            received_body = True
            return {"type": "http.request", "body": b'{"stream":true}', "more_body": False}
        await _fetch_until_requests(broker, count=1)
        return {"type": "http.disconnect"}

    async def send(_message):
        pass

    with InferenceProxy(
        broker=broker,
        model="test",
        request_timeout_seconds=5,
        readiness_timeout_seconds=5,
        max_pending_requests=4,
        response_fetch_batch_size=16,
    ) as proxy:
        await proxy.app(
            {
                "type": "http",
                "asgi": {"spec_version": "2.3"},
                "method": "POST",
                "path": "/v1/chat/completions",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )
    assert broker.pending() == []
    assert broker.size() == 0


@pytest.mark.parametrize("interrupt", [False, True])
def test_broker_forwards_incremental_stream_and_exposes_worker_loss(interrupt) -> None:
    release = threading.Event()
    now = [0.0]
    broker = InferenceBroker(request_lease_timeout_seconds=10, clock=lambda: now[0])
    with _streaming_upstream(release) as upstream:
        worker = InferenceWorker(worker_id="test-worker", broker=broker, upstream=upstream, request_timeout_seconds=5)
        with (
            run_inference_worker(worker, max_in_flight=1),
            serve_inference_proxy(
                broker=broker,
                model="test",
                request_timeout_seconds=5,
                readiness_timeout_seconds=5,
                max_pending_requests=4,
                response_fetch_batch_size=16,
                server_start_timeout_seconds=5,
            ) as proxy,
        ):
            try:
                with httpx.stream(
                    "POST", proxy.endpoint.url("chat/completions"), json={"stream": True}, timeout=5
                ) as response:
                    assert response.status_code == 200
                    chunks = response.iter_raw()
                    assert next(chunks) == b"data: first\n\n"
                    if interrupt:
                        now[0] = 11
                        with pytest.raises(httpx.RemoteProtocolError):
                            next(chunks)
                    else:
                        release.set()
                        assert b"".join(chunks) == b"data: [DONE]\n\n"
            finally:
                release.set()
    assert broker.pending() == []
    assert broker.size() == 0


@contextmanager
def _streaming_upstream(release: threading.Event) -> Iterator[RunningModel]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for payload in (b"data: first\n\n", b"data: [DONE]\n\n"):
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                self.wfile.flush()
                if not release.wait(timeout=5):
                    self.close_connection = True
                    return
            self.wfile.write(b"0\r\n\r\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield RunningModel(OpenAIEndpoint(f"http://127.0.0.1:{server.server_port}/v1", "test"))
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_inference_broker_round_trip() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    request = InferenceRequest(request_id="req-1", method="POST", path="/v1/completions", payload=b"request")

    broker.submit_request(request)

    assert broker.pending() == ["req-1"]
    assert broker.size() == 1
    leased_requests = broker.fetch_requests(max_items=8)
    assert [leased.request for leased in leased_requests] == [request]
    assert broker.fetch_requests(max_items=8) == []

    response_a = InferenceResponse(request_id="req-1", status_code=200, payload=b"a")
    broker.submit_responses([LeasedInferenceResponse(lease_id=leased_requests[0].lease_id, response=response_a)])

    assert broker.pending() == []
    assert broker.size() == 1
    assert broker.fetch_responses(max_items=1) == [response_a]
    assert broker.fetch_responses(max_items=8) == []
    assert broker.size() == 0


def test_inference_worker_preserves_raw_transport_fields() -> None:
    request = InferenceRequest(
        request_id="raw",
        method="POST",
        path="/v1/embeddings",
        query_string="encoding_format=float",
        headers=(("content-type", "application/octet-stream"), ("x-request-id", "caller-1")),
        payload=b"\x00raw-body",
    )

    def _handle(upstream_request: httpx.Request) -> httpx.Response:
        assert upstream_request.url == "http://upstream/v1/embeddings?encoding_format=float"
        assert upstream_request.headers["x-request-id"] == "caller-1"
        assert upstream_request.content == b"\x00raw-body"
        return httpx.Response(
            201,
            content=b"\x00raw-response",
            headers={"content-type": "application/octet-stream", "x-request-id": "upstream-1"},
        )

    worker = InferenceWorker(
        worker_id="test-worker",
        broker=InferenceBroker(request_lease_timeout_seconds=5),
        upstream=RunningModel(endpoint=OpenAIEndpoint(base_url="http://upstream/v1", model="gpt2")),
        request_timeout_seconds=5,
    )
    with httpx.Client(transport=httpx.MockTransport(_handle)) as client:
        response = worker._forward_one(client, LeasedInferenceRequest(lease_id="lease", request=request))

    assert response.response.status_code == 201
    assert response.response.payload == b"\x00raw-response"
    assert dict(response.response.headers) == {
        "content-type": "application/octet-stream",
        "x-request-id": "upstream-1",
    }


def test_inference_broker_requeues_unanswered_request_after_lease_timeout() -> None:
    now = [0.0]
    broker = InferenceBroker(request_lease_timeout_seconds=10, clock=lambda: now[0])
    request = InferenceRequest(request_id="req-1", method="POST", path="/v1/completions", payload=b"request")

    broker.submit_request(request)

    leased_a = broker.fetch_requests(max_items=1)
    assert [leased.request for leased in leased_a] == [request]
    assert broker.fetch_requests(max_items=1) == []

    now[0] = 11.0

    leased_b = broker.fetch_requests(max_items=1)
    assert [leased.request for leased in leased_b] == [request]
    assert leased_b[0].lease_id != leased_a[0].lease_id


def test_inference_broker_drops_response_for_expired_lease_after_requeue() -> None:
    now = [0.0]
    broker = InferenceBroker(request_lease_timeout_seconds=10, clock=lambda: now[0])
    request = InferenceRequest(request_id="req-1", method="POST", path="/v1/completions", payload=b"request")

    broker.submit_request(request)
    [lease_a] = broker.fetch_requests(max_items=1)

    now[0] = 11.0
    [lease_b] = broker.fetch_requests(max_items=1)
    assert lease_b.request == request
    assert lease_b.lease_id != lease_a.lease_id

    stale_response = InferenceResponse(request_id="req-1", status_code=504, payload=b"stale")
    fresh_response = InferenceResponse(request_id="req-1", status_code=200, payload=b"fresh")

    broker.submit_responses([LeasedInferenceResponse(lease_id=lease_a.lease_id, response=stale_response)])

    assert broker.fetch_responses(max_items=1) == []
    assert broker.pending() == ["req-1"]

    broker.submit_responses([LeasedInferenceResponse(lease_id=lease_b.lease_id, response=fresh_response)])

    assert broker.pending() == []
    assert broker.fetch_responses(max_items=1) == [fresh_response]


@pytest.mark.parametrize("instances", [1, 2])
def test_remote_inference_brokers_single_and_multiple_instances(monkeypatch, instances) -> None:
    broker_actor = InferenceBroker(request_lease_timeout_seconds=240)
    broker_actor.register_worker("worker-0", InferenceWorkerMetadata(tensor_parallel_size=1, backend_name="vllm"))
    proxy_models: list[str] = []
    events = []
    minted = []

    class _FakeJob:
        job_id = "worker-0"

        def terminate(self) -> None:
            pass

        def status(self) -> JobStatus:
            return JobStatus.RUNNING

    class _FakeActorGroup:
        def wait_ready(self, *, count: int, timeout: float):
            assert count == 1
            assert timeout > 0
            return [broker_actor]

        def shutdown(self) -> None:
            pass

    class _FakeClient:
        def __init__(self) -> None:
            self.submissions = []

        def create_actor_group(self, *args, **kwargs):
            return _FakeActorGroup()

        def submit(self, job_request):
            self.submissions.append(job_request)
            return _FakeJob()

    @contextmanager
    def _fake_start_proxy(**kwargs):
        proxy_models.append(kwargs["model"])
        yield RunningModel(endpoint=OpenAIEndpoint(base_url="http://127.0.0.1:1/v1", model=kwargs["model"]))

    client = _FakeClient()

    @contextmanager
    def registered(*args, **kwargs):
        events.append(("register", args, kwargs))
        try:
            yield "endpoint-id"
        finally:
            events.append(("unregister", "endpoint-id"))

    registry = SimpleNamespace(registered=registered)
    iris_client = SimpleNamespace(
        mint_endpoint_token=lambda name, ttl: minted.append(name)
        or SimpleNamespace(token="broker-token", capability_url=""),
    )
    monkeypatch.setattr(iris_module, "current_client", lambda: client)
    monkeypatch.setattr(
        iris_module,
        "iris_ctx",
        lambda: SimpleNamespace(registry=registry, client=iris_client),
    )
    monkeypatch.setattr(iris_module, "get_job_info", lambda: SimpleNamespace(advertise_host="127.0.0.1"))
    monkeypatch.setattr(iris_module, "serve_inference_proxy", _fake_start_proxy)
    monkeypatch.setattr(
        iris_module.requests, "get", lambda *_args, **_kwargs: SimpleNamespace(raise_for_status=lambda: None)
    )

    iris = IrisConfig(
        worker_resources=ResourceConfig.with_tpu("v6e-4", regions=["us-east5"], zone="us-east5-a"),
        worker_environment=create_environment(
            extras=["tpu"],
            env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
        ),
    )

    with remote_inference(
        RemoteInferenceConfig(
            model=ServedModelConfig(weights="physical-model", api_model="public-model"),
            engine=VllmEngineConfig(),
            iris=iris,
            instances=instances,
            broker=BrokerConfig(),
            capability_origin="https://iris.example",
        )
    ) as session:
        assert session.model.endpoint.model == "public-model"
        assert session.model.endpoint.base_url.startswith("https://iris.example/proxy/t/broker-token/")
        assert session.effective_serving is not None
        assert session.effective_serving.task_count == instances

    assert proxy_models == ["public-model"]
    _, register_args, register_kwargs = events[0]
    endpoint_name, address, metadata = register_args
    assert endpoint_name == minted[0]
    assert address == "http://127.0.0.1:1"
    assert metadata["model"] == "public-model"
    assert metadata["task_count"] == str(instances)
    assert register_kwargs["access"] == EndpointAccess.ENDPOINT_ACCESS_LINK
    assert events[-1] == ("unregister", "endpoint-id")
    assert len(client.submissions) == instances
    for worker_request in client.submissions:
        assert worker_request.environment.extras == ["tpu"]
        assert worker_request.environment.env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        assert worker_request.resources.regions == ["us-east5"]
        assert worker_request.resources.zone == "us-east5-a"


def test_inference_proxy_forwards_completions_to_running_model(mock_cluster: MockInferenceCluster) -> None:
    assert_completions_scoring_contract(mock_cluster.proxy.endpoint.base_url, mock_cluster.proxy.endpoint.model)

    upstream_requests = mock_cluster.upstream.requests_for("/v1/completions")
    assert len(upstream_requests) == 1
    assert mock_cluster.broker.pending() == []
    assert mock_cluster.broker.size() == 0


def test_inference_proxy_routes_models_readiness_to_running_model(mock_cluster: MockInferenceCluster) -> None:
    response = httpx.get(f"{mock_cluster.proxy.endpoint.base_url}/models", timeout=5)

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == mock_cluster.model
    assert len(mock_cluster.upstream.requests_for("/v1/models")) == 1
    assert mock_cluster.broker.pending() == []
    assert mock_cluster.broker.size() == 0


def test_inference_proxy_forwards_server_root_tokenize_route(mock_cluster: MockInferenceCluster) -> None:
    server_root = mock_cluster.proxy.endpoint.base_url.removesuffix("/v1")
    response = httpx.post(
        f"{server_root}/tokenize",
        json={
            "model": mock_cluster.model,
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": True,
        },
        timeout=5,
    )

    assert response.status_code == 200
    assert response.json() == {"count": 3, "max_model_len": 32768, "tokens": [1, 2, 3]}
    assert len(mock_cluster.upstream.requests_for("/tokenize")) == 1
    assert mock_cluster.broker.pending() == []
    assert mock_cluster.broker.size() == 0


def test_inference_proxy_preserves_upstream_response_to_streaming_request(
    mock_cluster: MockInferenceCluster,
) -> None:
    response = httpx.post(
        f"{mock_cluster.proxy.endpoint.base_url}/chat/completions",
        json={"model": mock_cluster.model, "messages": [], "stream": True},
        timeout=5,
    )

    assert response.status_code == 200
    assert response.json()["choices"]
    assert len(mock_cluster.upstream.requests_for("/v1/chat/completions")) == 1
    assert mock_cluster.broker.size() == 0


@pytest.mark.asyncio
async def test_inference_worker_refills_slots_while_slow_request_is_in_flight() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    for request_id, prompt in [("slow", "slow"), ("fast-a", "fast a"), ("fast-b", "fast b")]:
        broker.submit_request(_completion_inference_request(request_id=request_id, prompt=prompt))

    release_slow = threading.Event()
    with serve_deterministic_openai_stub(
        prompt_pauses={"slow": release_slow},
    ) as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = InferenceWorker(
            worker_id="test-worker",
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=5,
        )
        with run_inference_worker(
            worker,
            max_in_flight=2,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
        ):
            try:
                responses = await _fetch_until_responses(broker, count=2)
                assert {response.request_id for response in responses} == {"fast-a", "fast-b"}
            finally:
                release_slow.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "expected_status"),
    [("GET", "/v1/models", 200), ("GET", "/metrics", 404), ("POST", "/tokenize", 200)],
)
async def test_auxiliary_requests_bypass_full_generation_queue(method, path, expected_status) -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    release = threading.Event()
    with serve_deterministic_openai_stub(prompt_pauses={"slow": release}) as upstream_stub:
        upstream = RunningModel(OpenAIEndpoint(upstream_stub.base_url, upstream_stub.model))
        worker = InferenceWorker(worker_id="test-worker", broker=broker, upstream=upstream, request_timeout_seconds=10)
        with (
            _serve_inference_proxy(
                broker=broker, model=upstream_stub.model, request_timeout_seconds=10, max_pending_requests=1
            ) as proxy,
            run_inference_worker(
                worker, max_in_flight=1, backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0)
            ),
        ):
            async with httpx.AsyncClient(base_url=proxy.endpoint.base_url.removesuffix("/v1")) as client:
                generation = asyncio.create_task(
                    client.post(
                        "/v1/completions",
                        content=_completion_inference_request(request_id="slow", prompt="slow").payload,
                    )
                )
                try:
                    async with asyncio.timeout(5):
                        while not upstream_stub.requests_for("/v1/completions"):
                            await asyncio.sleep(0.01)
                    # Also put a generation ahead of the auxiliary request in the broker's FIFO.
                    broker.submit_request(_completion_inference_request(request_id="queued", prompt="queued"))
                    response = await client.request(
                        method, path, content=_json_bytes({"model": upstream_stub.model, "prompt": "hello"})
                    )
                    assert response.status_code == expected_status
                    assert not generation.done()
                    assert len(upstream_stub.requests_for("/v1/completions")) == 1
                    if path == "/v1/models":
                        assert response.json()["data"][0]["id"] == upstream_stub.model
                    elif path == "/tokenize":
                        assert response.json()["count"] == 3
                finally:
                    release.set()
                    await generation


@pytest.mark.asyncio
async def test_inference_worker_returns_504_for_upstream_timeout() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_inference_request(request_id="slow", prompt="slow"))

    release_slow = threading.Event()
    with serve_deterministic_openai_stub(prompt_pauses={"slow": release_slow}) as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = InferenceWorker(
            worker_id="test-worker",
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=0.05,
        )
        with run_inference_worker(
            worker,
            max_in_flight=1,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
        ):
            try:
                responses = await _fetch_until_responses(broker, count=1)
            finally:
                release_slow.set()

    assert responses[0].request_id == "slow"
    assert responses[0].status_code == 504
    assert "error" in _json_payload(responses[0].payload)


@pytest.mark.asyncio
async def test_inference_worker_returns_502_for_upstream_connection_failure() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_inference_request(request_id="connect-failure", prompt="connect failure"))
    upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=f"http://127.0.0.1:{_closed_port()}/v1", model="gpt2"))
    worker = InferenceWorker(
        worker_id="test-worker",
        broker=broker,
        upstream=upstream,
        request_timeout_seconds=1,
    )
    with run_inference_worker(
        worker,
        max_in_flight=1,
        backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
    ):
        responses = await _fetch_until_responses(broker, count=1)

    assert responses[0].request_id == "connect-failure"
    assert responses[0].status_code == 502
    assert "error" in _json_payload(responses[0].payload)


@pytest.mark.asyncio
async def test_inference_worker_preserves_status_for_non_json_upstream_response() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_inference_request(request_id="non-json", prompt="non json"))
    with _serve_text_upstream(status_code=503, body="temporarily unavailable") as upstream:
        worker = InferenceWorker(
            worker_id="test-worker",
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=1,
        )
        with run_inference_worker(
            worker,
            max_in_flight=1,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
        ):
            responses = await _fetch_until_responses(broker, count=1)

    assert responses[0].request_id == "non-json"
    assert responses[0].status_code == 503
    assert responses[0].payload == b"temporarily unavailable"
    assert dict(responses[0].headers)["content-type"].startswith("text/plain")


@pytest.mark.asyncio
async def test_inference_proxy_matches_out_of_order_responses_to_inflight_requests() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with _serve_inference_proxy(
        broker=broker,
        model="gpt2",
        request_timeout_seconds=5,
    ) as proxy_model:
        async with httpx.AsyncClient() as client:
            first = asyncio.create_task(
                client.post(f"{proxy_model.endpoint.base_url}/completions", json={"model": "gpt2", "prompt": "first"})
            )
            second = asyncio.create_task(
                client.post(f"{proxy_model.endpoint.base_url}/completions", json={"model": "gpt2", "prompt": "second"})
            )

            requests = await _fetch_until_two_requests(broker)
            requests_by_prompt = {_json_payload(request.request.payload)["prompt"]: request for request in requests}
            broker.submit_responses(
                [
                    _leased_response(
                        requests_by_prompt["second"],
                        InferenceResponse(
                            request_id=requests_by_prompt["second"].request.request_id,
                            status_code=200,
                            payload=_json_bytes({"prompt": "second"}),
                            headers=(("content-type", "application/json"),),
                        ),
                    ),
                    _leased_response(
                        requests_by_prompt["first"],
                        InferenceResponse(
                            request_id=requests_by_prompt["first"].request.request_id,
                            status_code=200,
                            payload=_json_bytes({"prompt": "first"}),
                            headers=(("content-type", "application/json"),),
                        ),
                    ),
                ]
            )

            first_response, second_response = await asyncio.gather(first, second)

    assert first_response.json() == {"prompt": "first"}
    assert second_response.json() == {"prompt": "second"}


@pytest.mark.asyncio
async def test_inference_proxy_rejects_when_pending_queue_is_full() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with _serve_inference_proxy(
        broker=broker,
        model="gpt2",
        request_timeout_seconds=5,
        max_pending_requests=1,
    ) as proxy_model:
        async with httpx.AsyncClient() as client:
            first = asyncio.create_task(
                client.post(f"{proxy_model.endpoint.base_url}/completions", json={"model": "gpt2", "prompt": "first"})
            )

            requests = await _fetch_until_requests(broker, count=1)
            rejected = await client.post(
                f"{proxy_model.endpoint.base_url}/completions",
                json={"model": "gpt2", "prompt": "second"},
            )

            broker.submit_responses(
                [
                    _leased_response(
                        requests[0],
                        InferenceResponse(
                            request_id=requests[0].request.request_id,
                            status_code=200,
                            payload=_json_bytes({"prompt": "first"}),
                            headers=(("content-type", "application/json"),),
                        ),
                    )
                ]
            )
            first_response = await first

    assert rejected.status_code == 429
    assert rejected.headers["Retry-After"] == "1"
    assert "error" in rejected.json()
    assert first_response.json() == {"prompt": "first"}


@pytest.mark.asyncio
async def test_inference_proxy_times_out_inflight_request() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with _serve_inference_proxy(
        broker=broker,
        model="gpt2",
        request_timeout_seconds=0.05,
    ) as proxy_model:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{proxy_model.endpoint.base_url}/completions", json={"model": "gpt2", "prompt": "timeout"}
            )

    assert response.status_code == 504


@pytest.mark.asyncio
async def test_inference_proxy_drops_stale_responses() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    request = InferenceRequest(request_id="stale", method="POST", path="/v1/completions", payload=b"request")
    broker.submit_request(request)
    [leased_request] = broker.fetch_requests(max_items=1)
    broker.submit_responses(
        [
            _leased_response(
                leased_request,
                InferenceResponse(
                    request_id="stale",
                    status_code=200,
                    payload=_json_bytes({"prompt": "stale"}),
                ),
            )
        ]
    )
    proxy = InferenceProxy(
        broker=broker,
        model="gpt2",
        request_timeout_seconds=5,
        readiness_timeout_seconds=5,
        max_pending_requests=8,
        response_fetch_batch_size=8,
    )

    assert proxy.tick() == 1

    assert broker.fetch_responses(max_items=1) == []
    assert proxy.stats.dropped_responses == 1


async def _fetch_until_two_requests(broker: InferenceBroker) -> list[LeasedInferenceRequest]:
    return await _fetch_until_requests(broker, count=2)


async def _fetch_until_requests(broker: InferenceBroker, *, count: int) -> list[LeasedInferenceRequest]:
    requests: list[LeasedInferenceRequest] = []
    deadline = asyncio.get_running_loop().time() + 5
    while len(requests) < count and asyncio.get_running_loop().time() < deadline:
        requests.extend(broker.fetch_requests(max_items=count - len(requests)))
        if len(requests) < count:
            await asyncio.sleep(0.01)
    assert len(requests) == count
    return requests


async def _fetch_until_responses(broker: InferenceBroker, *, count: int) -> list[InferenceResponse]:
    responses: list[InferenceResponse] = []
    deadline = asyncio.get_running_loop().time() + 5
    while len(responses) < count and asyncio.get_running_loop().time() < deadline:
        responses.extend(broker.fetch_responses(max_items=count - len(responses)))
        if len(responses) < count:
            await asyncio.sleep(0.01)
    assert len(responses) == count
    return responses


def _leased_response(leased_request: LeasedInferenceRequest, response: InferenceResponse) -> LeasedInferenceResponse:
    return LeasedInferenceResponse(lease_id=leased_request.lease_id, response=response)


def _completion_inference_request(*, request_id: str, prompt: str) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        method="POST",
        path="/v1/completions",
        payload=_json_bytes(
            {
                "model": "gpt2",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            }
        ),
        headers=(("content-type", "application/json"),),
    )


@contextmanager
def _serve_inference_proxy(
    *,
    broker: InferenceBroker,
    model: str,
    request_timeout_seconds: float,
    max_pending_requests: int | None = None,
    readiness_timeout_seconds: float | None = None,
) -> Iterator[RunningModel]:
    config = InferenceProxyConfig(
        request_timeout_seconds=request_timeout_seconds,
        readiness_timeout_seconds=(
            request_timeout_seconds if readiness_timeout_seconds is None else readiness_timeout_seconds
        ),
    )
    with serve_inference_proxy(
        broker=broker,
        model=model,
        request_timeout_seconds=config.request_timeout_seconds,
        readiness_timeout_seconds=config.readiness_timeout_seconds,
        max_pending_requests=config.max_pending_requests if max_pending_requests is None else max_pending_requests,
        response_fetch_batch_size=config.response_fetch_batch_size,
        server_start_timeout_seconds=config.server_start_timeout_seconds,
    ) as running_model:
        yield running_model


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _TextResponseServer(ThreadingHTTPServer):
    status_code: int
    body: str


class _TextResponseHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        server = cast(_TextResponseServer, self.server)
        body = server.body.encode("utf-8")
        self.send_response(server.status_code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _serve_text_upstream(*, status_code: int, body: str) -> Iterator[RunningModel]:
    server = _TextResponseServer(("127.0.0.1", 0), _TextResponseHandler)
    server.status_code = status_code
    server.body = body
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield RunningModel(endpoint=OpenAIEndpoint(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="gpt2"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
