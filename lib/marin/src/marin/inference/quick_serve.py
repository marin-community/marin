# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick single-model vLLM inference server for an Iris TPU slice.

A quick-serve job boots one native vLLM server on a single-host TPU slice, fronts
it with a browser dashboard + OpenAI-compatible reverse proxy, and registers the
dashboard as an Iris endpoint so it is reachable through the controller proxy. The
job shuts itself down after a wall-clock timeout so a forgotten server does not sit
on a slice indefinitely.

This module holds the serving config and the in-job entrypoint that boots vLLM on
the slice; the ``marin-serve`` launcher CLI is a separate module.
"""

import json
import logging
import socket
import tempfile
import time
from dataclasses import dataclass, field

import requests
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.tpu_topology import get_tpu_topology
from iris.cluster.types import EndpointAccess
from levanter.model_cache import resolve_cached_model_path
from rigging.connect import proxy_path
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging
from transformers import AutoConfig

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.quick_serve_dashboard import (
    ServingInfo,
    bind_serving_socket,
    build_dashboard_app,
    serve_app_background,
)
from marin.inference.vllm_server import VllmEnvironment, _is_object_store_path

logger = logging.getLogger(__name__)

# vLLM's OpenAI server mounts its routes under /v1; the dashboard reverse-proxy
# forwards verbatim paths, so we keep the upstream root separately.
_VLLM_API_SUFFIX = "/v1"
# Cadence of the wall-clock timeout / liveness loop.
_TIMEOUT_POLL_SECONDS = 30
# GCS prefix (under the region-local TTL temp bucket) for mirrored HF snapshots.
_MODEL_CACHE_PREFIX = "quick-serve-models"


@dataclass(frozen=True)
class QuickServeConfig:
    """Everything the in-job entrypoint needs to serve one model.

    This is cloudpickled into the Iris job, so every field must be a plain value.
    """

    model: str
    """HF model id (e.g. ``Qwen/Qwen3-0.6B``) or object-store path (``gs://...``)."""
    tpu_type: str
    """Single-host TPU slice type, e.g. ``v6e-8`` or ``v5litepod-8``."""
    endpoint_name: str
    """Iris endpoint name registered for the dashboard (a leading ``/`` is verbatim)."""
    access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE
    """Proxy access mode. PRIVATE (cluster identity only) or LINK (a scoped capability
    URL, minted CLI-side, that anyone holding the link can call off-cluster)."""
    port_name: str = "http"
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    """vLLM max sequence length. ``None`` lets vLLM derive it from the model config,
    which keeps models with a clamped RoPE window (e.g. Delphi's 4k) bootable."""
    tensor_parallel_size: int | None = None
    """``None`` auto-selects the largest power-of-two TP that divides the model's
    attention-head count and fits the slice's chip count."""
    max_num_batched_tokens: int = 512
    """Prefill batch size. Kept modest because the TPU paged-attention kernel's
    on-chip (VMEM) scratch grows with this; large values overflow VMEM at compile."""
    chat_template_content: str | None = None
    """Inline Jinja chat template forwarded to vLLM; resolved from a path/URL by the CLI."""
    cache_ttl_days: int = 14
    """Mirror HF models to a region-local GCS cache with this lifecycle TTL so repeat
    serves skip the HuggingFace download. ``0`` disables caching; ignored for gs:// paths."""
    timeout_hours: float = 24.0
    vllm_startup_timeout_seconds: int = 1800
    extra_vllm_args: tuple[str, ...] = field(default_factory=tuple)


def select_tensor_parallel_size(
    num_attention_heads: int,
    num_chips: int,
    num_key_value_heads: int | None = None,
) -> int:
    """Pick the largest power-of-two tensor-parallel size valid for this model+slice.

    vLLM requires ``num_attention_heads`` to be divisible by the TP size and the
    KV-head count to be compatible (divides TP or is divisible by it). TPU slices
    expose power-of-two chip counts, so we search powers of two up to ``num_chips``.
    Models with odd or prime head counts fall back to TP 1.
    """
    if num_chips < 1:
        return 1
    best = 1
    candidate = 1
    while candidate <= num_chips:
        if num_attention_heads % candidate == 0 and _kv_heads_compatible(num_key_value_heads, candidate):
            best = candidate
        candidate *= 2
    return best


def _kv_heads_compatible(num_key_value_heads: int | None, tensor_parallel_size: int) -> bool:
    if not num_key_value_heads:
        return True
    return num_key_value_heads % tensor_parallel_size == 0 or tensor_parallel_size % num_key_value_heads == 0


def read_attention_heads(model: str) -> tuple[int, int | None]:
    """Return ``(num_attention_heads, num_key_value_heads)`` for an HF id or object-store path."""
    config_dict = _read_model_config_dict(model)
    for scope in (config_dict, config_dict.get("text_config"), config_dict.get("llm_config")):
        if not isinstance(scope, dict):
            continue
        heads = scope.get("num_attention_heads")
        if heads:
            kv_heads = scope.get("num_key_value_heads")
            return int(heads), (int(kv_heads) if kv_heads else None)
    raise ValueError(f"Could not find num_attention_heads in the model config for {model!r}.")


def _read_model_config_dict(model: str) -> dict:
    if _is_object_store_path(model):
        config_path = model.rstrip("/") + "/config.json"
        with open_url(config_path, "r") as handle:
            return json.load(handle)
    return AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()


def resolve_model_path(model: str, cache_ttl_days: int) -> str:
    """Resolve ``model`` to a path vLLM can load, mirroring HF repos to a TTL'd GCS cache.

    HuggingFace repo ids are mirrored once to a region-local GCS cache under a distributed
    lock, so a later serve of the same model reads the snapshot from same-region GCS instead
    of re-downloading from HuggingFace; object-store paths are served directly.
    """
    return resolve_cached_model_path(model, cache_ttl_days=cache_ttl_days, cache_prefix=_MODEL_CACHE_PREFIX)


def detect_chat_support(vllm_base_url: str, model_id: str) -> bool:
    """Probe whether the served model accepts ``/v1/chat/completions``.

    Base/midtrained checkpoints ship no chat template, so vLLM rejects chat
    requests; the dashboard defaults such models to completion mode.
    """
    try:
        response = requests.post(
            f"{vllm_base_url}{_VLLM_API_SUFFIX}/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            timeout=60,
        )
    except requests.RequestException as exc:
        logger.warning("Chat-support probe failed (%s); defaulting to completion mode.", exc)
        return False
    return response.status_code == 200


def _write_chat_template(content: str | None) -> str | None:
    if content is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", prefix="quick_serve_chat_", delete=False) as handle:
        handle.write(content)
        return handle.name


def _reserve_localhost_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_vllm_extra_args(
    config: QuickServeConfig, tensor_parallel_size: int, chat_template_path: str | None
) -> list[str]:
    # Pin the served model name to the requested model so the OpenAI API id stays
    # the friendly HF id regardless of whether the backing path is local or gs://.
    args = [
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--dtype",
        config.dtype,
        "--served-model-name",
        config.model,
    ]
    if chat_template_path is not None:
        args += ["--chat-template", chat_template_path]
    args += list(config.extra_vllm_args)
    return args


def _block_until_timeout(env: VllmEnvironment, timeout_hours: float) -> None:
    """Block until the timeout elapses, failing early if vLLM dies."""
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        server = env.vllm_server
        if server is not None and server.process.poll() is not None:
            raise RuntimeError(f"vLLM server exited unexpectedly with code {server.process.returncode}.")
        time.sleep(_TIMEOUT_POLL_SECONDS)
    logger.info("quick-serve reached its %.1fh timeout; shutting down.", timeout_hours)


def serve_in_job(config: QuickServeConfig) -> None:
    """Iris job entrypoint: boot vLLM, serve the dashboard, register the endpoint, block."""
    configure_logging()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("serve_in_job must run inside an Iris job.")
    ctx = iris_ctx()
    port = ctx.get_port(config.port_name)
    advertise_host = job_info.advertise_host
    # Claim the dashboard's port now, before vLLM launches: Iris' named-port range
    # overlaps the OS ephemeral range, so vLLM's internal sockets could otherwise
    # squat it. Binding here reserves it for us until uvicorn takes over. Bind only
    # the advertised interface (the address the controller proxy connects to), not
    # all interfaces.
    serving_socket = bind_serving_socket(advertise_host, port)

    model_path = resolve_model_path(config.model, config.cache_ttl_days)
    num_chips = get_tpu_topology(config.tpu_type).chips_per_vm
    if config.tensor_parallel_size is not None:
        tensor_parallel_size = config.tensor_parallel_size
        logger.info(
            "quick-serve model=%s tpu=%s chips=%d tensor_parallel_size=%d (user-specified)",
            config.model,
            config.tpu_type,
            num_chips,
            tensor_parallel_size,
        )
    else:
        num_attention_heads, num_key_value_heads = read_attention_heads(model_path)
        tensor_parallel_size = select_tensor_parallel_size(num_attention_heads, num_chips, num_key_value_heads)
        logger.info(
            "quick-serve model=%s tpu=%s chips=%d heads=%d kv_heads=%s -> tensor_parallel_size=%d",
            config.model,
            config.tpu_type,
            num_chips,
            num_attention_heads,
            num_key_value_heads,
            tensor_parallel_size,
        )

    chat_template_path = _write_chat_template(config.chat_template_content)
    engine_kwargs: dict[str, object] = {"max_num_batched_tokens": config.max_num_batched_tokens}
    if config.max_model_len is not None:
        engine_kwargs["max_model_len"] = config.max_model_len
    vllm_model = ModelConfig(name="quick-serve", path=model_path, engine_kwargs=engine_kwargs)
    extra_args = _build_vllm_extra_args(config, tensor_parallel_size, chat_template_path)
    internal_port = _reserve_localhost_port()

    with VllmEnvironment(
        vllm_model,
        host="127.0.0.1",
        port=internal_port,
        timeout_seconds=config.vllm_startup_timeout_seconds,
        extra_args=extra_args,
    ) as env:
        if env.model_id is None:
            raise RuntimeError("vLLM server did not report a model id.")
        model_id = env.model_id
        upstream_base_url = env.server_url.removesuffix(_VLLM_API_SUFFIX)
        has_chat_template = config.chat_template_content is not None or detect_chat_support(upstream_base_url, model_id)
        info = ServingInfo(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            has_chat_template=has_chat_template,
            tpu_type=config.tpu_type,
            endpoint=config.endpoint_name,
        )
        app = build_dashboard_app(upstream_base_url=upstream_base_url, model_id=model_id, info=info)
        with serve_app_background(app, serving_socket):
            address = f"http://{advertise_host}:{port}"
            metadata = {
                "model": str(model_id),
                "kind": "quick-serve",
                "tpu": config.tpu_type,
                "tensor_parallel_size": str(tensor_parallel_size),
            }
            endpoint_id = ctx.registry.register(config.endpoint_name, address, metadata, access=config.access)
            logger.info(
                "Registered quick-serve endpoint name=%s address=%s id=%s proxy_path=%s",
                config.endpoint_name,
                address,
                endpoint_id,
                proxy_path(config.endpoint_name),
            )
            try:
                _block_until_timeout(env, config.timeout_hours)
            finally:
                try:
                    ctx.registry.unregister(endpoint_id)
                except Exception:
                    # Best-effort during teardown: the controller may already have
                    # dropped the endpoint when the task began terminating.
                    logger.warning("Failed to unregister quick-serve endpoint id=%s", endpoint_id, exc_info=True)
