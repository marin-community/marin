# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the quick-serve TP auto-selection and dashboard reverse proxy."""

import dataclasses
import json
import socket
import time
from unittest.mock import MagicMock

import click
import pytest
import requests
from click.testing import CliRunner
from iris.rpc import controller_pb2
from iris.time_proto import timestamp_to_proto
from marin.inference.quick_serve import (
    resolve_model_path,
    select_tensor_parallel_size,
)
from marin.inference.quick_serve_cli import (
    _checkout_free_setup_script,
    _mint_and_print_capability_url,
    _resolve_serving_plan,
    main,
)
from marin.inference.quick_serve_dashboard import (
    ServingInfo,
    bind_serving_socket,
    build_dashboard_app,
    serve_app_background,
)
from marin.inference.serving_backend import (
    DEFAULT_LEVANTER_MAX_SEQ_LEN,
    LevanterBackend,
    VllmBackend,
    inference_mesh,
    levanter_max_seq_len,
    validate_levanter_dtype,
)
from marin.inference.tpu_vllm_pins import vllm_fork_ref
from marin.inference.vllm_server import (
    WORKER_PYTHON_VERSION,
    IsolatedCudaVllm,
    IsolatedTpuVllm,
    VllmType,
    WorkspaceVllm,
)
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


def test_isolated_cuda_vllm_upstream_command_and_env():
    launcher = IsolatedCudaVllm(source=VllmType.UPSTREAM, version="0.25.0")
    assert launcher.command() == [
        "uvx",
        "--from",
        "vllm[runai]==0.25.0",
        "--python",
        WORKER_PYTHON_VERSION,
        "--torch-backend",
        "cu128",
        "vllm",
    ]
    assert launcher.env() == {}


def test_isolated_cuda_vllm_marin_fork_command_and_env():
    launcher = IsolatedCudaVllm(source=VllmType.MARIN_FORK)
    cmd = launcher.command()
    assert cmd[:3] == ["uvx", "--from", vllm_fork_ref()]
    assert "--torch-backend" in cmd and cmd[cmd.index("--torch-backend") + 1] == "cu130"
    assert "runai-model-streamer[s3]==0.16.0" in cmd
    env = launcher.env()
    assert env["VLLM_USE_PRECOMPILED"] == "1"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "AWS_CONFIG_FILE" in env


def test_isolated_cuda_vllm_upstream_requires_version():
    with pytest.raises(ValueError, match="requires an explicit vLLM version"):
        IsolatedCudaVllm(source=VllmType.UPSTREAM)


def test_vllm_backend_falls_back_to_workspace_without_version():
    # No launcher (the TPU path, or a --task-image GPU path whose image ships its own vLLM) serves
    # from the vLLM already on PATH.
    assert VllmBackend().select_launcher() == WorkspaceVllm()


def test_vllm_backend_returns_its_composed_launcher():
    launcher = IsolatedTpuVllm(vllm_ref="vllm @ git+...@abc", tpu_inference_ref="tpu-inference @ git+...@def")
    assert VllmBackend(launcher=launcher).select_launcher() is launcher


def test_levanter_max_seq_len_defaults_within_the_models_window():
    # A model advertising a huge window still serves a modest KV cache by default...
    assert levanter_max_seq_len(None, 131072) == DEFAULT_LEVANTER_MAX_SEQ_LEN
    # ...and a model with a smaller window than the default clamps down to it.
    assert levanter_max_seq_len(None, 2048) == 2048
    # An explicit request is honored up to the model's window, and rejected past it.
    assert levanter_max_seq_len(8192, 131072) == 8192
    with pytest.raises(ValueError, match="exceeds the model"):
        levanter_max_seq_len(8192, 4096)


def test_validate_levanter_dtype_rejects_vllm_aliases():
    assert validate_levanter_dtype("bfloat16") == "bfloat16"
    # vLLM accepts these; Levanter loads weights at a concrete dtype, so they are errors here.
    for alias in ("auto", "half", "float"):
        with pytest.raises(ValueError, match="not supported by the levanter backend"):
            validate_levanter_dtype(alias)


@pytest.mark.parametrize(
    ("num_chips", "tensor_parallel_size", "expected"),
    [
        (8, 8, {"replica": 1, "data": 1, "model": 8}),  # the slice divides the head count: shard across it
        (8, 2, {"replica": 1, "data": 4, "model": 2}),  # it does not: the leftover chips replicate
    ],
)
def test_inference_mesh_covers_every_chip(num_chips, tensor_parallel_size, expected):
    assert dict(inference_mesh(num_chips, tensor_parallel_size).axes) == expected


def test_inference_mesh_rejects_a_tp_that_does_not_divide_the_slice():
    with pytest.raises(ValueError, match="does not divide"):
        inference_mesh(8, 3)


def test_cli_rejects_vllm_flags_under_the_levanter_backend():
    result = CliRunner().invoke(main, ["Qwen/Qwen3-0.6B", "--backend", "levanter", "--vllm-arg", "--enforce-eager"])
    assert result.exit_code != 0
    assert "--vllm-arg cannot be used with --backend levanter" in result.output


def test_cli_defaulted_vllm_options_do_not_trip_the_levanter_backend(monkeypatch):
    """--vllm-version and --max-num-batched-tokens have non-None defaults.

    Rejecting a vLLM-only option by its *value* rather than by "the user typed it" would fail
    every levanter serve, so reaching the controller is the assertion.
    """
    reached_controller = RuntimeError("reached the controller")

    def _fail_at_controller(*_args, **_kwargs):
        raise reached_controller

    monkeypatch.setattr("marin.inference.quick_serve_cli.open_controller_endpoint", _fail_at_controller)
    result = CliRunner().invoke(main, ["Qwen/Qwen3-0.6B", "--backend", "levanter", "--max-seqs", "4"])
    assert result.exception is reached_controller


def test_cli_rejects_levanter_flags_under_the_vllm_backend():
    result = CliRunner().invoke(main, ["Qwen/Qwen3-0.6B", "--page-size", "64"])
    assert result.exit_code != 0
    assert "--page-size cannot be used with --backend vllm" in result.output


def _plan(**overrides):
    args = {
        "backend": "vllm",
        "tpu": "v6e-8",
        "gpu": None,
        "in_checkout": True,
        "isolated_vllm": False,
        "task_image": None,
        "cuda_vllm_version": "0.25.0",
        "vllm_source": VllmType.UPSTREAM,
        "vllm": VllmBackend(),
        "levanter": LevanterBackend(),
        "extras": (),
    }
    return _resolve_serving_plan(**{**args, **overrides})


@pytest.mark.parametrize(
    ("overrides", "backend_type", "worker_extras"),
    [
        # vLLM in a checkout builds from the workspace lock, so the venv needs both TPU extras.
        ({}, VllmBackend, ("tpu", "vllm")),
        # Outside a checkout (or with --isolated-vllm) vLLM comes from uvx: no `vllm` extra.
        ({"in_checkout": False}, VllmBackend, ("tpu",)),
        ({"isolated_vllm": True}, VllmBackend, ("tpu",)),
        # CUDA vLLM is provisioned by uvx, so the GPU worker venv needs no accelerator extra.
        ({"gpu": "H100x8"}, VllmBackend, ()),
        # Levanter computes in the worker venv, so that venv carries the accelerator's JAX itself.
        ({"backend": "levanter"}, LevanterBackend, ("tpu",)),
        ({"backend": "levanter", "gpu": "H100x8"}, LevanterBackend, ("gpu",)),
    ],
)
def test_resolve_serving_plan_picks_the_worker_extras_the_backend_needs(overrides, backend_type, worker_extras):
    plan = _plan(**overrides)
    assert isinstance(plan.backend, backend_type)
    assert plan.worker_extras == worker_extras


def test_gpu_plan_defaults_to_upstream_launcher():
    plan = _plan(gpu="H100x8")
    assert plan.backend.launcher == IsolatedCudaVllm(source=VllmType.UPSTREAM, version="0.25.0")


def test_gpu_plan_marin_fork_selects_fork_launcher():
    plan = _plan(gpu="H100x8", vllm_source=VllmType.MARIN_FORK)
    assert plan.backend.launcher == IsolatedCudaVllm(source=VllmType.MARIN_FORK)


def test_gpu_plan_task_image_serves_workspace_vllm():
    # A prebuilt --task-image ships its own vLLM on PATH, so no launcher is provisioned.
    assert _plan(gpu="H100x8", task_image="img").backend.launcher is None
    assert _plan(gpu="H100x8", task_image="img").backend.select_launcher() == WorkspaceVllm()


def test_tpu_plan_isolates_vllm_outside_a_checkout():
    # No checkout to build the TPU-vLLM fork from, so it comes from a pinned uvx env; in a checkout
    # it serves the workspace vLLM instead.
    assert isinstance(_plan(in_checkout=False).backend.select_launcher(), IsolatedTpuVllm)
    assert isinstance(_plan().backend.select_launcher(), WorkspaceVllm)


def test_marin_fork_requires_gpu():
    with pytest.raises(click.ClickException, match="requires --gpu"):
        _plan(vllm_source=VllmType.MARIN_FORK)  # default tpu path


def test_resolve_serving_plan_rejects_multihost_slices():
    with pytest.raises(click.ClickException, match="multi-host"):
        _plan(tpu="v6e-16")


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
        backend="vllm",
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
        backend="vllm",
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
