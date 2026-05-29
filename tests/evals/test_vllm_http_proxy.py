# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx
import pytest
from fray.types import ResourceConfig
from marin.inference.brokered_vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    IrisBrokeredVllmRuntimeConfig,
    VllmProxyConfig,
    VllmWorkerConfig,
    start_local_brokered_vllm,
)
from marin.inference.local_workload_broker import LocalWorkloadBroker
from marin.inference.types import OpenAIEndpoint, RunningModel
from marin.inference.vllm_http_proxy import VllmHttpProxy, serve_vllm_http_proxy
from marin.inference.vllm_worker import VllmWorker, run_vllm_worker
from marin.inference.workload_broker import (
    LeasedWorkloadRequest,
    LeasedWorkloadResponse,
    WorkloadRequest,
    WorkloadResponse,
    pack_json_payload,
    unpack_json_payload,
)
from rigging.timing import ExponentialBackoff

from tests.evals.openai_stub import assert_completions_scoring_contract, serve_deterministic_openai_stub

BROKER_LEASE_TIMEOUT_SECONDS = 300.0


def test_local_workload_broker_round_trip() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    request = WorkloadRequest(request_id="req-1", method="POST", path="/v1/completions", payload=b"request")

    broker.submit_request(request)

    assert broker.pending() == ["req-1"]
    assert broker.size() == 1
    leased_requests = broker.fetch_requests(max_items=8)
    assert [leased.request for leased in leased_requests] == [request]
    assert broker.fetch_requests(max_items=8) == []

    response_a = WorkloadResponse(request_id="req-1", status_code=200, payload=b"a")
    broker.submit_responses([LeasedWorkloadResponse(lease_id=leased_requests[0].lease_id, response=response_a)])

    assert broker.pending() == []
    assert broker.size() == 1
    assert broker.fetch_responses(max_items=1) == [response_a]
    assert broker.fetch_responses(max_items=8) == []
    assert broker.size() == 0


def test_local_workload_broker_requeues_unanswered_request_after_lease_timeout() -> None:
    now = [0.0]
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=10, clock=lambda: now[0])
    request = WorkloadRequest(request_id="req-1", method="POST", path="/v1/completions", payload=b"request")

    broker.submit_request(request)

    leased_a = broker.fetch_requests(max_items=1)
    assert [leased.request for leased in leased_a] == [request]
    assert broker.fetch_requests(max_items=1) == []

    now[0] = 11.0

    leased_b = broker.fetch_requests(max_items=1)
    assert [leased.request for leased in leased_b] == [request]
    assert leased_b[0].lease_id != leased_a[0].lease_id


def test_local_workload_broker_drops_response_for_expired_lease_after_requeue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [0.0]
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=10, clock=lambda: now[0])
    request = WorkloadRequest(request_id="req-1", method="POST", path="/v1/completions", payload=b"request")

    broker.submit_request(request)
    [lease_a] = broker.fetch_requests(max_items=1)

    now[0] = 11.0
    [lease_b] = broker.fetch_requests(max_items=1)
    assert lease_b.request == request
    assert lease_b.lease_id != lease_a.lease_id

    stale_response = WorkloadResponse(request_id="req-1", status_code=504, payload=b"stale")
    fresh_response = WorkloadResponse(request_id="req-1", status_code=200, payload=b"fresh")

    with caplog.at_level(logging.WARNING):
        broker.submit_responses([LeasedWorkloadResponse(lease_id=lease_a.lease_id, response=stale_response)])

    assert broker.fetch_responses(max_items=1) == []
    assert broker.pending() == ["req-1"]
    assert "dropped responses for inactive request leases" in caplog.text
    assert "request_ids=req-1" in caplog.text

    broker.submit_responses([LeasedWorkloadResponse(lease_id=lease_b.lease_id, response=fresh_response)])

    assert broker.pending() == []
    assert broker.fetch_responses(max_items=1) == [fresh_response]


def test_local_brokered_vllm_rejects_multiple_workers() -> None:
    config = BrokeredVllmSystemConfig(model="gpt2", workers=VllmWorkerConfig(count=2))

    with pytest.raises(ValueError, match="Local brokered vLLM mode supports exactly one worker"):
        with start_local_brokered_vllm(config):
            pass


def test_brokered_vllm_rejects_timeout_ordering_without_recovery_window() -> None:
    expected = "workers.request_timeout_seconds < request_lease_timeout_seconds < proxy.request_timeout_seconds"
    with pytest.raises(ValueError, match=expected):
        BrokeredVllmSystemConfig(model="gpt2", workers=VllmWorkerConfig(request_timeout_seconds=250))

    with pytest.raises(ValueError, match=expected):
        BrokeredVllmSystemConfig(model="gpt2", proxy=VllmProxyConfig(request_timeout_seconds=130))


def test_iris_runtime_leaves_child_placement_to_parent_region_inheritance() -> None:
    runtime = IrisBrokeredVllmRuntimeConfig(
        worker_resources=ResourceConfig.with_tpu("v5p-8"),
    )

    assert runtime.broker_resources.regions is None
    assert runtime.broker_resources.zone is None
    assert runtime.worker_resources.regions is None
    assert runtime.worker_resources.zone is None


def test_vllm_http_proxy_forwards_completions_to_running_model() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with serve_deterministic_openai_stub() as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = VllmWorker(broker=broker, upstream=upstream, request_timeout_seconds=5)
        with _serve_vllm_http_proxy(
            broker=broker,
            model=upstream_stub.model,
            request_timeout_seconds=5,
        ) as proxy_model:
            with run_vllm_worker(worker, max_in_flight=DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER):
                assert_completions_scoring_contract(proxy_model.endpoint.base_url, proxy_model.endpoint.model)

    upstream_requests = upstream_stub.requests_for("/v1/completions")
    assert len(upstream_requests) == 1
    assert broker.pending() == []
    assert broker.size() == 0


def test_vllm_http_proxy_routes_models_readiness_to_running_model() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with serve_deterministic_openai_stub() as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = VllmWorker(broker=broker, upstream=upstream, request_timeout_seconds=5)
        with _serve_vllm_http_proxy(
            broker=broker,
            model=upstream_stub.model,
            request_timeout_seconds=5,
            readiness_timeout_seconds=5,
        ) as proxy_model:
            with run_vllm_worker(worker, max_in_flight=DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER):
                response = httpx.get(f"{proxy_model.endpoint.base_url}/models", timeout=5)

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == upstream_stub.model
    assert len(upstream_stub.requests_for("/v1/models")) == 1
    assert broker.pending() == []
    assert broker.size() == 0


@pytest.mark.asyncio
async def test_vllm_worker_refills_slots_while_slow_request_is_in_flight() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    for request_id, prompt in [("slow", "slow"), ("fast-a", "fast a"), ("fast-b", "fast b")]:
        broker.submit_request(_completion_workload_request(request_id=request_id, prompt=prompt))

    release_slow_request = threading.Event()
    stop_event = threading.Event()
    with serve_deterministic_openai_stub(
        blocked_prompts={"slow": release_slow_request},
    ) as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = VllmWorker(
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=5,
        )
        worker_task = asyncio.create_task(
            worker.run_forever(
                stop_event=stop_event,
                max_in_flight=2,
                backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
            )
        )
        try:
            responses = await _fetch_until_responses(broker, count=2)
            assert {response.request_id for response in responses} == {"fast-a", "fast-b"}
        finally:
            release_slow_request.set()
            stop_event.set()
            async with asyncio.timeout(5):
                await worker_task


@pytest.mark.asyncio
async def test_vllm_worker_returns_504_for_upstream_timeout() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_workload_request(request_id="slow", prompt="slow"))

    release_slow_request = threading.Event()
    stop_event = threading.Event()
    with serve_deterministic_openai_stub(blocked_prompts={"slow": release_slow_request}) as upstream_stub:
        upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=upstream_stub.base_url, model=upstream_stub.model))
        worker = VllmWorker(
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=0.05,
        )
        worker_task = asyncio.create_task(
            worker.run_forever(
                stop_event=stop_event,
                max_in_flight=1,
                backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
            )
        )
        try:
            responses = await _fetch_until_responses(broker, count=1)
        finally:
            release_slow_request.set()
            stop_event.set()
            async with asyncio.timeout(5):
                await worker_task

    assert responses[0].request_id == "slow"
    assert responses[0].status_code == 504
    assert unpack_json_payload(responses[0].payload)["error"]["message"] == "timed out forwarding request to vLLM"


@pytest.mark.asyncio
async def test_vllm_worker_returns_502_for_upstream_connection_failure() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_workload_request(request_id="connect-failure", prompt="connect failure"))
    stop_event = threading.Event()
    upstream = RunningModel(endpoint=OpenAIEndpoint(base_url=f"http://127.0.0.1:{_closed_port()}/v1", model="gpt2"))
    worker = VllmWorker(
        broker=broker,
        upstream=upstream,
        request_timeout_seconds=1,
    )
    worker_task = asyncio.create_task(
        worker.run_forever(
            stop_event=stop_event,
            max_in_flight=1,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
        )
    )
    try:
        responses = await _fetch_until_responses(broker, count=1)
    finally:
        stop_event.set()
        async with asyncio.timeout(5):
            await worker_task

    assert responses[0].request_id == "connect-failure"
    assert responses[0].status_code == 502
    assert unpack_json_payload(responses[0].payload)["error"]["message"] == "failed forwarding request to vLLM"


@pytest.mark.asyncio
async def test_vllm_worker_preserves_status_for_non_json_upstream_response() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    broker.submit_request(_completion_workload_request(request_id="non-json", prompt="non json"))
    stop_event = threading.Event()
    with _serve_text_upstream(status_code=503, body="temporarily unavailable") as upstream:
        worker = VllmWorker(
            broker=broker,
            upstream=upstream,
            request_timeout_seconds=1,
        )
        worker_task = asyncio.create_task(
            worker.run_forever(
                stop_event=stop_event,
                max_in_flight=1,
                backoff=ExponentialBackoff(initial=0.01, maximum=0.01, jitter=0),
            )
        )
        try:
            responses = await _fetch_until_responses(broker, count=1)
        finally:
            stop_event.set()
            async with asyncio.timeout(5):
                await worker_task

    payload = unpack_json_payload(responses[0].payload)
    assert responses[0].request_id == "non-json"
    assert responses[0].status_code == 503
    assert payload["error"]["message"] == "vLLM returned a non-JSON response"
    assert payload["error"]["body"] == "temporarily unavailable"


@pytest.mark.asyncio
async def test_vllm_http_proxy_matches_out_of_order_responses_to_inflight_requests() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with _serve_vllm_http_proxy(
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
            broker.submit_responses(
                [
                    _leased_response(
                        requests[1],
                        WorkloadResponse(
                            request_id=requests[1].request.request_id,
                            status_code=200,
                            payload=pack_json_payload({"prompt": "second"}),
                        ),
                    ),
                    _leased_response(
                        requests[0],
                        WorkloadResponse(
                            request_id=requests[0].request.request_id,
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
async def test_vllm_http_proxy_rejects_when_pending_queue_is_full() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with _serve_vllm_http_proxy(
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
                        WorkloadResponse(
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
    assert rejected.json() == {"error": "too many pending proxy requests; back off and retry"}
    assert first_response.json() == {"prompt": "first"}


@pytest.mark.asyncio
async def test_vllm_http_proxy_times_out_inflight_request() -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    with _serve_vllm_http_proxy(
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
async def test_vllm_http_proxy_drops_stale_responses_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    broker = LocalWorkloadBroker(request_lease_timeout_seconds=BROKER_LEASE_TIMEOUT_SECONDS)
    request = WorkloadRequest(request_id="stale", method="POST", path="/v1/completions", payload=b"request")
    broker.submit_request(request)
    [leased_request] = broker.fetch_requests(max_items=1)
    broker.submit_responses(
        [
            _leased_response(
                leased_request,
                WorkloadResponse(
                    request_id="stale",
                    status_code=200,
                    payload=pack_json_payload({"prompt": "stale"}),
                ),
            )
        ]
    )
    proxy = VllmHttpProxy(
        broker=broker,
        model="gpt2",
        request_timeout_seconds=5,
        readiness_timeout_seconds=5,
        max_pending_requests=8,
        response_fetch_batch_size=8,
    )

    with caplog.at_level(logging.WARNING):
        assert await proxy.tick() == 1

    assert broker.fetch_responses(max_items=1) == []
    assert "dropped stale or duplicate workload responses" in caplog.text
    assert "request_ids=stale" in caplog.text


async def _fetch_until_two_requests(broker: LocalWorkloadBroker) -> list[LeasedWorkloadRequest]:
    return await _fetch_until_requests(broker, count=2)


async def _fetch_until_requests(broker: LocalWorkloadBroker, *, count: int) -> list[LeasedWorkloadRequest]:
    requests: list[LeasedWorkloadRequest] = []
    deadline = asyncio.get_running_loop().time() + 5
    while len(requests) < count and asyncio.get_running_loop().time() < deadline:
        requests.extend(broker.fetch_requests(max_items=count - len(requests)))
        if len(requests) < count:
            await asyncio.sleep(0.01)
    assert len(requests) == count
    return requests


async def _fetch_until_responses(broker: LocalWorkloadBroker, *, count: int) -> list[WorkloadResponse]:
    responses: list[WorkloadResponse] = []
    deadline = asyncio.get_running_loop().time() + 5
    while len(responses) < count and asyncio.get_running_loop().time() < deadline:
        responses.extend(broker.fetch_responses(max_items=count - len(responses)))
        if len(responses) < count:
            await asyncio.sleep(0.01)
    assert len(responses) == count
    return responses


def _leased_response(leased_request: LeasedWorkloadRequest, response: WorkloadResponse) -> LeasedWorkloadResponse:
    return LeasedWorkloadResponse(lease_id=leased_request.lease_id, response=response)


def _completion_workload_request(*, request_id: str, prompt: str) -> WorkloadRequest:
    return WorkloadRequest(
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
def _serve_vllm_http_proxy(
    *,
    broker: LocalWorkloadBroker,
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
    with serve_vllm_http_proxy(
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
