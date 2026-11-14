# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests against the released router in its isolated uvx environment."""

import asyncio
import json
import os
import signal
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import requests
from fray.types import ResourceConfig, create_environment
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.inference import iris_vllm
from marin.inference.config import (
    InferenceModelConfig,
    IrisConfig,
    ServedModelConfig,
    ServingGeometry,
    VllmEngineConfig,
    VllmRouterConfig,
)
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.inference.iris import IrisServiceConfig, run_iris_service
from marin.inference.serve import local_inference
from marin.inference.vllm_router import available_port, vllm_router
from marin.inference.vllm_server import VllmEnvironment
from marin.testing.inference.fake_vllm_server import FakeVllmLauncher
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route


@dataclass(frozen=True)
class RouterWorker:
    url: str
    calls: list[dict]
    cancelled: threading.Event
    next_chunk: threading.Event


@pytest.fixture
def router_worker():
    calls = []
    cancelled = threading.Event()
    next_chunk = threading.Event()

    async def health(_request):
        return PlainTextResponse("")

    async def models(_request):
        return JSONResponse({"object": "list", "data": [{"id": "fake-model"}]})

    async def chat(request):
        payload = await request.json()
        calls.append(payload)
        result = {"rank": int(request.headers.get("x-data-parallel-rank", "0")), "received": payload}
        if payload["messages"][0]["content"] == "fail":
            return JSONResponse(result, status_code=503)
        if not payload.get("stream"):
            return JSONResponse(result)

        async def chunks():
            try:
                yield f"data: {json.dumps(result)}\n\n".encode()
                if payload["messages"][0]["content"] == "cancel":
                    await asyncio.to_thread(next_chunk.wait)
                    while True:
                        yield b'data: {"choices": []}\n\n'
                        # Yield control to Starlette's disconnect listener.
                        await asyncio.sleep(0)
                yield b"data: [DONE]\n\n"
            finally:
                cancelled.set()

        return StreamingResponse(chunks(), media_type="text/event-stream")

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ]
    )
    sock = bind_serving_socket("127.0.0.1", 0)
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    with serve_app_background(app, sock):
        yield RouterWorker(url=url, calls=calls, cancelled=cancelled, next_chunk=next_chunk)


def _chat(url, session=None, content="hello", *, stream=False):
    return requests.post(
        f"{url}/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": content}], "stream": stream},
        headers={"X-Session-ID": session} if session else {},
        stream=stream,
        timeout=10,
    )


def _assert_native_rank_affinity(url, rank_count):
    ranks = set()
    for index in range(128):
        session = f"native-conversation-{index}"
        first = _chat(url, session)
        later = _chat(url, session, "later turn")
        first.raise_for_status()
        later.raise_for_status()
        assert later.json()["rank"] == first.json()["rank"]
        ranks.add(first.json()["rank"])
        if len(ranks) == rank_count:
            break
    assert ranks == set(range(rank_count))


def test_router_later_turns_reach_the_same_dp_rank(router_worker):
    port = available_port("127.0.0.1")
    with vllm_router(
        VllmRouterConfig(), router_worker.url, data_parallel_size=4, port=port, startup_timeout_seconds=30
    ) as router:
        _assert_native_rank_affinity(router.server_url, 4)
        # Ordinary OpenAI requests remain valid without a session header.
        assert _chat(router.server_url).status_code == 200
        with _chat(router.server_url, "stream-session", stream=True) as response:
            assert response.status_code == 200
            events = [line for line in response.iter_lines() if line.startswith(b"data:")]
            assert events[-1] == b"data: [DONE]"
            assert json.loads(events[0][5:])["received"]["stream"] is True
        router.process.send_signal(signal.SIGKILL)
        router.process.wait(timeout=10)
        with pytest.raises(RuntimeError):
            router.check_alive()
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/health", timeout=1)


def test_router_does_not_retry_a_failed_generation(router_worker):
    with vllm_router(VllmRouterConfig(), router_worker.url, data_parallel_size=1, startup_timeout_seconds=30) as router:
        assert _chat(router.server_url, "failed-session", "fail").status_code == 503
        assert len(router_worker.calls) == 1


@pytest.mark.parametrize("workload_error", [False, True])
def test_router_forced_shutdown_reaps_the_process_and_preserves_caller_errors(
    router_worker, monkeypatch, workload_error
):
    monkeypatch.setattr("marin.inference.vllm_router._ROUTER_SHUTDOWN_TIMEOUT", 1)
    port = available_port("127.0.0.1")
    expected = pytest.raises(RuntimeError, match="workload failed") if workload_error else nullcontext()
    with expected:
        with vllm_router(
            VllmRouterConfig(), router_worker.url, data_parallel_size=1, port=port, startup_timeout_seconds=30
        ) as router:
            # A frozen process cannot handle SIGTERM; teardown must escalate.
            os.killpg(router.process.pid, signal.SIGSTOP)
            if workload_error:
                raise RuntimeError("workload failed")
    assert router.process.poll() is not None
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/health", timeout=1)


def test_router_client_disconnect_cancels_the_worker_stream(router_worker):
    with vllm_router(VllmRouterConfig(), router_worker.url, data_parallel_size=1, startup_timeout_seconds=30) as router:
        with _chat(router.server_url, "cancel-session", "cancel", stream=True) as response:
            assert response.status_code == 200
            assert next(response.iter_content(chunk_size=1)) == b"d"
        # Released Router notices the closed receiver on the next upstream chunk.
        router_worker.next_chunk.set()
        assert ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
            router_worker.cancelled.is_set, timeout=Duration.from_seconds(5)
        )


def test_router_never_publishes_an_endpoint_with_an_unavailable_worker():
    missing = f"http://127.0.0.1:{available_port('127.0.0.1')}"
    port = available_port("127.0.0.1")
    with pytest.raises((RuntimeError, TimeoutError)):
        with vllm_router(VllmRouterConfig(), missing, data_parallel_size=1, port=port, startup_timeout_seconds=2):
            pytest.fail("published an endpoint with an unavailable worker")
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/health", timeout=1)


@pytest.mark.parametrize(
    "native_args,expected_dp",
    [
        ((), 1),
        pytest.param(
            (
                "--data-parallel-size=2",
                "--data_parallel_size=4",
                "--data-parallel-size-local=2",
                "--nnodes=2",
                "--node-rank=0",
                "--data-parallel-backend=mp",
            ),
            4,
            id="multi-host-internal-leader",
        ),
    ],
)
def test_local_inference_owns_router_and_worker_lifetimes_with_separate_metrics(
    monkeypatch, tmp_path, native_args, expected_dp
):
    # The real router must not inherit the serving runtime's dependency constraints.
    monkeypatch.setenv("UV_CONSTRAINT", str(tmp_path / "vllm-only-constraints.txt"))
    monkeypatch.setattr("marin.inference.vllm_backend.vllm_launcher", lambda _: FakeVllmLauncher("serve"))
    port = available_port("127.0.0.1")
    with local_inference(
        ServedModelConfig(weights="fake", chat_template_content="{{ messages }}"),
        VllmEngineConfig(router=VllmRouterConfig(), extra_args=native_args),
        port=port,
    ) as session:
        assert session.model.endpoint.base_url == f"http://127.0.0.1:{port}/v1"
        assert (
            requests.get(f"{session.model.endpoint.base_url}/models", timeout=5).json()["data"][0]["id"] == "fake-model"
        )
        assert session.metrics_url is not None
        assert session.metrics_url.removesuffix("/metrics") != session.model.endpoint.base_url.removesuffix("/v1")
        assert requests.get(session.metrics_url, timeout=5).status_code == 200
        _assert_native_rank_affinity(session.model.endpoint.base_url, expected_dp)
        session.check_alive()
    for url in (session.model.endpoint.base_url, session.metrics_url):
        with pytest.raises(requests.ConnectionError):
            requests.get(url, timeout=1)


def test_environment_routes_all_native_ranks():
    with VllmEnvironment(
        InferenceModelConfig("fake", None, {}),
        launcher=FakeVllmLauncher("serve"),
        port=available_port("127.0.0.1"),
        extra_args=["--data-parallel-size=4", "--data-parallel-size-local=2"],
        router=VllmRouterConfig(),
    ) as environment:
        assert environment.server_url != environment.worker_server_url
        _assert_native_rank_affinity(environment.server_url, 4)


@pytest.mark.parametrize(
    "native_args",
    [
        ("-dpe",),
        ("--data-parallel-rank=0",),
        ("--data-parallel-size-local=2", "--data-parallel-start-rank=0"),
        ("-n", "2", "--data-parallel-start-rank=0"),
    ],
)
def test_managed_router_rejects_native_load_balancers_that_cannot_dispatch_global_ranks(native_args):
    with pytest.raises(ValueError):
        with VllmEnvironment(
            InferenceModelConfig("fake", None, {}),
            launcher=FakeVllmLauncher("serve"),
            extra_args=["--data-parallel-size", "4", *native_args],
            router=VllmRouterConfig(),
        ):
            pytest.fail("published a router without global native rank dispatch")


@pytest.mark.parametrize(
    "geometry,native_args",
    [
        (None, ("-dp", "4")),
        (ServingGeometry(1, 4, 1, 4), ()),
    ],
)
def test_iris_router_routes_all_native_ranks_and_registers_the_same_topology(
    monkeypatch, tmp_path, geometry, native_args
):
    monkeypatch.setattr("marin.inference.vllm_backend.vllm_launcher", lambda _: FakeVllmLauncher("serve"))
    registered_metadata = {}

    @contextmanager
    def registered(_name, address, metadata=None, **_kwargs):
        if metadata is not None:
            registered_metadata.update(metadata)
            _assert_native_rank_affinity(f"{address}/v1", 4)
        yield

    context = SimpleNamespace(registry=SimpleNamespace(registered=registered))
    monkeypatch.setattr("marin.inference.iris.iris_ctx", lambda: context)
    monkeypatch.setattr(iris_vllm, "iris_ctx", lambda: context)
    coordinator = iris_vllm.VllmCoordinatorActor("127.0.0.1")
    monkeypatch.setattr(iris_vllm, "_coordinator_client", lambda _: coordinator)
    service = IrisServiceConfig(
        model=ServedModelConfig(weights=str(tmp_path), tensor_parallel_size=1, chat_template_content="{{ messages }}"),
        engine=VllmEngineConfig(router=VllmRouterConfig(), extra_args=native_args),
        iris=IrisConfig(
            worker_resources=ResourceConfig.with_gpu("H100", count=4),
            worker_environment=create_environment(docker_image="test"),
            serving_geometry=geometry,
        ),
        endpoint_name="/serve/router",
        timeout_hours=0,
        port_name=None,
    )
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/router/0")))
    try:
        run_iris_service(service)
    finally:
        set_job_info(None)
    assert int(registered_metadata["data_parallel_size"]) == 4
