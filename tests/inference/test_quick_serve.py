# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the quick-serve TP auto-selection and dashboard reverse proxy."""

import dataclasses
import json
import socket
import time
from unittest.mock import MagicMock

import pytest
import requests
from iris.rpc import controller_pb2
from iris.time_proto import timestamp_to_proto
from marin.inference.quick_serve import (
    QuickServeConfig,
    resolve_model_path,
    select_tensor_parallel_size,
    select_vllm_launcher,
)
from marin.inference.quick_serve_cli import (
    _checkout_free_setup_script,
    _mint_and_print_capability_url,
)
from marin.inference.quick_serve_dashboard import (
    ServingInfo,
    bind_serving_socket,
    build_dashboard_app,
    serve_app_background,
)
from marin.inference.vllm_server import IsolatedCudaVllm, IsolatedTpuVllm, WorkspaceVllm
from rigging.timing import Timestamp
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route


@pytest.mark.parametrize(
    ("heads", "chips", "kv_heads", "expected"),
    [
        # Non-power-of-two head counts on an 8-chip slice still pick a valid TP.
        (30, 8, None, 2),  # only 1 and 2 are power-of-two divisors of 30
        (11, 8, None, 1),  # odd/prime head count cannot shard
        # Power-of-two head counts use the whole slice.
        (32, 8, 8, 8),
        (16, 4, 8, 4),
        (16, 8, 8, 8),
        # KV heads must stay compatible: tp must divide or be divisible by them.
        (32, 8, 2, 8),  # 8 % 2 == 0
        (12, 8, 4, 4),  # 8 does not divide 12; 4 does and 4 % 4 == 0
        # Degenerate slices fall back to single-chip serving.
        (16, 1, 8, 1),
        (7, 8, None, 1),
    ],
)
def test_select_tensor_parallel_size(heads, chips, kv_heads, expected):
    assert select_tensor_parallel_size(heads, chips, kv_heads) == expected


@pytest.mark.parametrize(
    ("model", "ttl_days"),
    [
        ("gs://bucket/ckpt", 14),  # object-store paths are served directly, never mirrored
        ("s3://bucket/ckpt", 14),
        ("Qwen/Qwen3-0.6B", 0),  # caching disabled
    ],
)
def test_resolve_model_path_passthrough(model, ttl_days):
    # These paths must not touch the network or GCS; they return the input unchanged.
    assert resolve_model_path(model, ttl_days) == model


def test_checkout_free_setup_script_pins_marin_core_with_extras():
    # The worker install folds the requested extras and the launching CLI's exact version
    # (for cloudpickle compat) into the pip spec; vLLM stays out — it comes from uvx.
    script = _checkout_free_setup_script("0.2.44", ("tpu",))
    assert "marin-core[tpu]==0.2.44" in script
    assert "vllm" not in script


def test_select_vllm_launcher_gpu_provisions_isolated_cuda_vllm():
    config = QuickServeConfig(
        model="Qwen/Qwen3-0.6B", endpoint_name="/serve/x", gpu_type="H100", gpu_count=8, vllm_version="0.25.0"
    )
    assert select_vllm_launcher(config) == IsolatedCudaVllm(version="0.25.0")


def test_select_vllm_launcher_falls_back_to_workspace_without_version():
    # The TPU path (no version) and a --task-image GPU path (image ships its own vLLM)
    # both serve from the vLLM already on PATH.
    tpu = QuickServeConfig(model="m", endpoint_name="/serve/x", tpu_type="v6e-8")
    gpu_with_image = QuickServeConfig(
        model="m", endpoint_name="/serve/x", gpu_type="H100", gpu_count=8, vllm_version=None
    )
    assert select_vllm_launcher(tpu) == WorkspaceVllm()
    assert select_vllm_launcher(gpu_with_image) == WorkspaceVllm()


def test_select_vllm_launcher_tpu_isolated_from_refs():
    config = QuickServeConfig(
        model="Qwen/Qwen3-0.6B",
        endpoint_name="/serve/x",
        tpu_type="v6e-8",
        tpu_vllm_ref="vllm @ git+https://github.com/marin-community/vllm.git@abc",
        tpu_inference_ref="tpu-inference @ git+https://github.com/marin-community/tpu-inference.git@def",
    )
    assert select_vllm_launcher(config) == IsolatedTpuVllm(
        vllm_ref="vllm @ git+https://github.com/marin-community/vllm.git@abc",
        tpu_inference_ref="tpu-inference @ git+https://github.com/marin-community/tpu-inference.git@def",
    )


def test_select_vllm_launcher_tpu_ref_requires_tpu_inference():
    # A vLLM fork without its tpu-inference runtime would boot a broken TPU server; fail
    # at config time instead.
    config = QuickServeConfig(model="m", endpoint_name="/serve/x", tpu_type="v6e-8", tpu_vllm_ref="vllm @ git+...@abc")
    with pytest.raises(ValueError, match="tpu_inference_ref"):
        select_vllm_launcher(config)


def _mint_response(token: str, ttl_hours: float) -> controller_pb2.Controller.MintEndpointTokenResponse:
    expires = Timestamp.from_ms(int(time.time() * 1000) + int(ttl_hours * 3_600_000))
    return controller_pb2.Controller.MintEndpointTokenResponse(token=token, expires_at=timestamp_to_proto(expires))


def test_mint_and_print_capability_url_prints_off_cluster_url(capsys):
    """LINK serve prints the OpenAI base_url with the scoped token in the URL path."""
    client = MagicMock()
    client._cluster_client.mint_endpoint_token.return_value = _mint_response("ep-token-xyz", 24.0)

    _mint_and_print_capability_url(client, "/serve/foo", "https://iris.oa.dev", 24.0)

    out = capsys.readouterr().out
    # The scoped token rides in the URL path (gist-style); possession is the credential.
    assert "https://iris.oa.dev/proxy/t/ep-token-xyz/serve.foo/v1" in out


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sse(chunks: list[dict]) -> StreamingResponse:
    async def body():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


def _fake_vllm_app() -> Starlette:
    """A stand-in for the local vLLM OpenAI server the dashboard proxies to."""

    async def health(_request):
        return PlainTextResponse("", status_code=200)

    async def models(_request):
        return JSONResponse({"object": "list", "data": [{"id": "fake-model"}]})

    async def chat(_request):
        return _sse([{"choices": [{"delta": {"content": tok}}]} for tok in ("Hello", ", ", "world", "!")])

    async def completions(_request):
        return _sse([{"choices": [{"text": tok}]} for tok in ("123", "456")])

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/v1/completions", completions, methods=["POST"]),
        ]
    )


def _collect_sse_text(response: requests.Response, field: str) -> str:
    text = ""
    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: ") :].strip()
        if payload == b"[DONE]":
            break
        delta = json.loads(payload)["choices"][0]
        text += delta["delta"]["content"] if field == "delta" else delta["text"]
    return text


def test_dashboard_serves_ui_and_reverse_proxies_streaming():
    upstream_sock = bind_serving_socket("127.0.0.1", 0)
    upstream_port = upstream_sock.getsockname()[1]
    dashboard_sock = bind_serving_socket("127.0.0.1", 0)
    dashboard_port = dashboard_sock.getsockname()[1]
    info = ServingInfo(
        model="fake-model",
        tensor_parallel_size=2,
        max_model_len=4096,
        dtype="bfloat16",
        has_chat_template=True,
        tpu_type="v6e-8",
        endpoint="/serve/fake",
    )

    with serve_app_background(_fake_vllm_app(), upstream_sock):
        app = build_dashboard_app(
            upstream_base_url=f"http://127.0.0.1:{upstream_port}", model_id="fake-model", info=info
        )
        with serve_app_background(app, dashboard_sock):
            base = f"http://127.0.0.1:{dashboard_port}"

            page = requests.get(f"{base}/", timeout=10)
            assert page.status_code == 200
            assert "marin · quick serve" in page.text

            assert requests.get(f"{base}/info", timeout=10).json() == dataclasses.asdict(info)
            assert requests.get(f"{base}/health", timeout=10).json() == {"status": "ok", "model": "fake-model"}
            assert requests.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"] == "fake-model"

            chat = requests.post(
                f"{base}/v1/chat/completions",
                json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                stream=True,
                timeout=10,
            )
            assert _collect_sse_text(chat, "delta") == "Hello, world!"

            completion = requests.post(
                f"{base}/v1/completions",
                json={"model": "fake-model", "prompt": "x", "stream": True},
                stream=True,
                timeout=10,
            )
            assert _collect_sse_text(completion, "text") == "123456"


def test_dashboard_health_reports_loading_when_upstream_down():
    dashboard_sock = bind_serving_socket("127.0.0.1", 0)
    dashboard_port = dashboard_sock.getsockname()[1]
    info = ServingInfo(
        model="fake-model",
        tensor_parallel_size=1,
        max_model_len=None,
        dtype="bfloat16",
        has_chat_template=False,
        tpu_type="v6e-8",
        endpoint="/serve/fake",
    )
    # Point at a closed port so the upstream health probe fails fast.
    app = build_dashboard_app(upstream_base_url=f"http://127.0.0.1:{_free_port()}", model_id="fake-model", info=info)
    with serve_app_background(app, dashboard_sock):
        response = requests.get(f"http://127.0.0.1:{dashboard_port}/health", timeout=10)
    assert response.status_code == 503
    assert response.json()["status"] == "loading"
