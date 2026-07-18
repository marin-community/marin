# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
import marin.inference.vllm as vllm_module
import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import lower
from marin.inference.broker import InferenceBroker
from marin.inference.proxy import InferenceProxy, serve_inference_proxy
from marin.inference.types import (
    InferenceRequest,
    InferenceResponse,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    OpenAIEndpoint,
    RunningModel,
    pack_json_payload,
    unpack_json_payload,
)
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    GpuVllmBackend,
    InferenceWorkerConfig,
    TpuVllmBackend,
    VllmProxyConfig,
    start_iris_brokered_vllm,
    start_local_brokered_vllm,
    start_local_vllm_server,
)
from marin.inference.vllm_server import DEFAULT_CUDA_VLLM_VERSION
from marin.inference.worker import InferenceWorker, run_inference_worker
from rigging.timing import ExponentialBackoff

from experiments.evals.served_qwen3 import QWEN3_GPU_EVAL_RESULTS
from tests.evals.openai_stub import (
    DeterministicOpenAIStub,
    assert_completions_scoring_contract,
    serve_deterministic_openai_stub,
)

BROKER_LEASE_TIMEOUT_SECONDS = 300.0


def test_brokered_gpu_eval_lowers_with_symbolic_worker_resources() -> None:
    step = lower(QWEN3_GPU_EVAL_RESULTS)
    fingerprint = json.loads(step.fingerprint_payload)

    # Artifact lowering replaces runtime resources with a symbolic value. Backend validation must
    # remain at the serving boundary, where the concrete ResourceConfig is available.
    assert fingerprint["inference"]["worker_resources"] == "<worker_resources>"
    assert fingerprint["inference"]["backend"] == {
        "source": "upstream",
        "tensor_parallel_size": None,
        "version": DEFAULT_CUDA_VLLM_VERSION,
    }


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
        worker = InferenceWorker(broker=broker, upstream=upstream, request_timeout_seconds=5)
        with (
            _serve_inference_proxy(
                broker=broker,
                model=upstream_stub.model,
                request_timeout_seconds=5,
                readiness_timeout_seconds=5,
            ) as proxy,
            run_inference_worker(worker, max_in_flight=DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER),
        ):
            yield MockInferenceCluster(
                broker=broker,
                model=upstream_stub.model,
                upstream=upstream_stub,
                proxy=proxy,
            )


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


def test_local_brokered_vllm_rejects_multiple_workers() -> None:
    config = BrokeredVllmSystemConfig(model="gpt2", workers=InferenceWorkerConfig(count=2))

    with pytest.raises(ValueError):
        with start_local_brokered_vllm(config):
            pass


@pytest.mark.parametrize(
    ("backend", "worker_resources", "expected_extras", "expected_target_device"),
    [
        (TpuVllmBackend(), ResourceConfig.with_tpu("v6e-4"), ["tpu", "vllm"], "tpu"),
        (GpuVllmBackend(), ResourceConfig.with_gpu("H100", count=8), [], None),
    ],
)
def test_iris_brokered_vllm_worker_environment_matches_backend(
    monkeypatch,
    backend,
    worker_resources,
    expected_extras,
    expected_target_device,
) -> None:
    class _FakeJob:
        job_id = "worker-0"

        def terminate(self) -> None:
            pass

    class _FakeActorGroup:
        def wait_ready(self, *, count: int, timeout: float):
            assert count == 1
            assert timeout > 0
            return [object()]

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
    def _fake_start_proxy(config, response_provider):
        yield RunningModel(endpoint=OpenAIEndpoint(base_url="http://127.0.0.1:1", model=config.model))

    client = _FakeClient()
    monkeypatch.setattr(vllm_module, "current_client", lambda: client)
    monkeypatch.setattr(vllm_module, "get_job_info", lambda: SimpleNamespace(job_id="parent"))
    monkeypatch.setattr(vllm_module, "_start_proxy", _fake_start_proxy)
    monkeypatch.setattr(vllm_module, "_wait_for_brokered_vllm_ready", lambda *args, **kwargs: None)

    config = BrokeredVllmSystemConfig(
        model="gpt2",
        backend=backend,
        worker_resources=worker_resources,
        worker_env_vars=(("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),),
    )

    with start_iris_brokered_vllm(config):
        pass

    [worker_request] = client.submissions
    assert worker_request.environment.extras == expected_extras
    assert worker_request.environment.env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert worker_request.environment.env_vars.get("VLLM_TARGET_DEVICE") == expected_target_device


@pytest.mark.parametrize(
    ("backend", "worker_resources"),
    [
        (TpuVllmBackend(), ResourceConfig.with_gpu("H100")),
        (GpuVllmBackend(), ResourceConfig.with_tpu("v6e-4")),
    ],
)
def test_brokered_vllm_rejects_resources_for_the_wrong_backend(backend, worker_resources) -> None:
    config = BrokeredVllmSystemConfig(model="gpt2", backend=backend, worker_resources=worker_resources)

    with pytest.raises(ValueError, match=r"Backend requires .* worker_resources"):
        with start_local_vllm_server(config):
            pass


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
async def test_inference_worker_returns_504_for_upstream_timeout() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_inference_request(request_id="slow", prompt="slow"))

    release_slow = threading.Event()
    with serve_deterministic_openai_stub(prompt_pauses={"slow": release_slow}) as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = InferenceWorker(
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
    assert "error" in unpack_json_payload(responses[0].payload)


@pytest.mark.asyncio
async def test_inference_worker_returns_502_for_upstream_connection_failure() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_inference_request(request_id="connect-failure", prompt="connect failure"))
    upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=f"http://127.0.0.1:{_closed_port()}/v1", model="gpt2"))
    worker = InferenceWorker(
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
    assert "error" in unpack_json_payload(responses[0].payload)


@pytest.mark.asyncio
async def test_inference_worker_preserves_status_for_non_json_upstream_response() -> None:
    broker = InferenceBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_inference_request(request_id="non-json", prompt="non json"))
    with _serve_text_upstream(status_code=503, body="temporarily unavailable") as upstream:
        worker = InferenceWorker(
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

    payload = unpack_json_payload(responses[0].payload)
    assert responses[0].request_id == "non-json"
    assert responses[0].status_code == 503
    assert "error" in payload
    assert payload["error"]["body"] == "temporarily unavailable"


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
            requests_by_prompt = {
                unpack_json_payload(request.request.payload)["prompt"]: request for request in requests
            }
            broker.submit_responses(
                [
                    _leased_response(
                        requests_by_prompt["second"],
                        InferenceResponse(
                            request_id=requests_by_prompt["second"].request.request_id,
                            status_code=200,
                            payload=pack_json_payload({"prompt": "second"}),
                        ),
                    ),
                    _leased_response(
                        requests_by_prompt["first"],
                        InferenceResponse(
                            request_id=requests_by_prompt["first"].request.request_id,
                            status_code=200,
                            payload=pack_json_payload({"prompt": "first"}),
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
                            payload=pack_json_payload({"prompt": "first"}),
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
                    payload=pack_json_payload({"prompt": "stale"}),
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
        payload=pack_json_payload(
            {
                "model": "gpt2",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            }
        ),
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
    config = VllmProxyConfig(
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
