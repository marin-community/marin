# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests against the released router in its isolated uvx environment."""

import asyncio
import json
import signal
import threading
from contextlib import ExitStack

import pytest
import requests
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmRouterConfig
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.inference.serve import local_inference
from marin.inference.vllm_router import available_port, vllm_router
from marin.testing.inference.fake_vllm_server import FakeVllmLauncher
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

pytestmark = pytest.mark.integration


@pytest.fixture
def router_workers():
    with ExitStack() as stack:

        def start(worker: str):
            calls = []
            cancelled = threading.Event()
            next_chunk = threading.Event()

            async def health(_request):
                return PlainTextResponse("")

            async def models(_request):
                return JSONResponse({"object": "list", "data": [{"id": "fake-model"}]})

            async def chat(request):
                payload = await request.json()
                rank = request.headers.get("x-data-parallel-rank")
                calls.append(payload)
                result = {"worker": worker, "rank": rank, "received": payload}
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
            stack.enter_context(serve_app_background(app, sock))
            return url, calls, cancelled, next_chunk

        yield start


def _chat(url, session=None, content="hello", *, stream=False):
    return requests.post(
        f"{url}/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": content}], "stream": stream},
        headers={"X-Session-ID": session} if session else {},
        stream=stream,
        timeout=10,
    )


@pytest.mark.parametrize("worker_count,dp_size", [(2, 1), (1, 4)])
def test_router_later_turns_reach_the_same_worker_and_dp_rank(router_workers, worker_count, dp_size):
    workers = [router_workers(str(index))[0] for index in range(worker_count)]
    port = available_port("127.0.0.1")
    with vllm_router(VllmRouterConfig(dp_size), workers, port=port, startup_timeout_seconds=30) as router:
        destinations = set()
        for index in range(32):
            session = f"conversation-{index}"
            first = _chat(router.server_url, session)
            later = _chat(router.server_url, session, "a different turn")
            first.raise_for_status()
            later.raise_for_status()
            destination = first.json()["worker"], first.json()["rank"]
            assert (later.json()["worker"], later.json()["rank"]) == destination
            destinations.add(destination)
        assert len(destinations) == worker_count * dp_size
        # Ordinary OpenAI requests remain valid without a session header.
        assert _chat(router.server_url).status_code == 200
        with _chat(router.server_url, "stream-session", stream=True) as response:
            assert response.status_code == 200
            events = [line for line in response.iter_lines() if line.startswith(b"data:")]
            assert events[-1] == b"data: [DONE]"
            assert json.loads(events[0][5:])["received"]["stream"] is True
        router.process.send_signal(signal.SIGKILL)
        router.process.wait(timeout=10)
        with pytest.raises(RuntimeError, match="Router exited"):
            router.check_alive()
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/health", timeout=1)


def test_router_does_not_retry_a_failed_generation(router_workers):
    url, calls, _, _ = router_workers("one")
    with vllm_router(VllmRouterConfig(), (url,), startup_timeout_seconds=30) as router:
        assert _chat(router.server_url, "failed-session", "fail").status_code == 503
        assert len(calls) == 1


def test_router_client_disconnect_cancels_the_worker_stream(router_workers):
    url, _, cancelled, next_chunk = router_workers("one")
    with vllm_router(VllmRouterConfig(), (url,), startup_timeout_seconds=30) as router:
        with _chat(router.server_url, "cancel-session", "cancel", stream=True) as response:
            assert response.status_code == 200
            assert next(response.iter_content(chunk_size=1)) == b"d"
        # Released Router notices the closed receiver on the next upstream chunk.
        next_chunk.set()
        assert ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
            cancelled.is_set, timeout=Duration.from_seconds(5)
        )


def test_router_never_publishes_an_endpoint_with_an_unavailable_worker(router_workers):
    url, _, _, _ = router_workers("one")
    missing = f"http://127.0.0.1:{available_port('127.0.0.1')}"
    port = available_port("127.0.0.1")
    with pytest.raises((RuntimeError, TimeoutError)):
        with vllm_router(VllmRouterConfig(), (url, missing), port=port, startup_timeout_seconds=2):
            pytest.fail("published a partially ready deployment")
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/health", timeout=1)


def test_local_inference_owns_router_and_worker_lifetimes_with_separate_metrics(monkeypatch, tmp_path):
    # The real router must not inherit the serving runtime's dependency constraints.
    monkeypatch.setenv("UV_CONSTRAINT", str(tmp_path / "vllm-only-constraints.txt"))
    monkeypatch.setattr("marin.inference.vllm_backend.vllm_launcher", lambda _: FakeVllmLauncher("serve"))
    port = available_port("127.0.0.1")
    with local_inference(
        ServedModelConfig(weights="fake", chat_template_content="{{ messages }}"),
        VllmEngineConfig(router=VllmRouterConfig()),
        port=port,
    ) as session:
        assert session.model.endpoint.base_url == f"http://127.0.0.1:{port}/v1"
        assert (
            requests.get(f"{session.model.endpoint.base_url}/models", timeout=5).json()["data"][0]["id"] == "fake-model"
        )
        assert session.metrics_url is not None
        assert session.metrics_url.removesuffix("/metrics") != session.model.endpoint.base_url.removesuffix("/v1")
        assert requests.get(session.metrics_url, timeout=5).status_code == 200
        session.check_alive()
    for url in (session.model.endpoint.base_url, session.metrics_url):
        with pytest.raises(requests.ConnectionError):
            requests.get(url, timeout=1)
