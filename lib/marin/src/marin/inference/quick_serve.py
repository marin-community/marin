# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick single-model inference server for an Iris TPU or GPU slice.

A quick-serve job boots one serving backend on a single-host slice, fronts it with a browser
dashboard + OpenAI-compatible reverse proxy, and registers the dashboard as an Iris endpoint so it
is reachable through the controller proxy. The job shuts itself down after a wall-clock timeout so
a forgotten server does not sit on a slice indefinitely.

Which stack answers requests is the backend's business (:mod:`marin.inference.serving_backend`):
vLLM as a subprocess, or Levanter's inference engine in-process. This module holds the serving
config and the in-job entrypoint; the ``marin-serve`` launcher CLI is a separate module.
"""

import json
import logging
import time
from dataclasses import dataclass

import requests
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.tpu_topology import get_tpu_topology
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess
from levanter.model_cache import resolve_cached_model_path
from rigging.connect import proxy_path
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from transformers import AutoConfig

from marin.inference.quick_serve_dashboard import (
    ServingInfo,
    bind_serving_socket,
    build_dashboard_app,
    serve_app_background,
)
from marin.inference.serving_backend import OPENAI_API_SUFFIX, ModelSpec, ServedModel, ServingBackend
from marin.inference.vllm_server import _is_object_store_path

logger = logging.getLogger(__name__)

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
    endpoint_name: str
    """Iris endpoint name registered for the dashboard (a leading ``/`` is verbatim)."""
    backend: ServingBackend
    """The stack that serves the model: :class:`~marin.inference.serving_backend.VllmBackend` or
    :class:`~marin.inference.serving_backend.LevanterBackend`, carrying its own knobs."""
    tpu_type: str | None = None
    """Single-host TPU slice type, e.g. ``v6e-8`` or ``v5litepod-8``. Set on the TPU path;
    ``None`` on the GPU path, where ``gpu_type``/``gpu_count`` describe the slice instead."""
    gpu_type: str | None = None
    """GPU variant (e.g. ``H100``) when serving on a GPU slice; ``None`` on the TPU path."""
    gpu_count: int | None = None
    """GPU count per node when serving on a GPU slice. Its presence selects the GPU path:
    ``num_chips`` is taken directly from it rather than a TPU topology lookup."""
    access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE
    """Proxy access mode. PRIVATE (cluster identity only) or LINK (a scoped capability
    URL, minted CLI-side, that anyone holding the link can call off-cluster)."""
    port_name: str = "http"
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    """Maximum sequence length. ``None`` lets vLLM derive it from the model config, which keeps
    models with a clamped RoPE window (e.g. Delphi's 4k) bootable; the Levanter backend, which
    must size its KV cache up front, falls back to its own default window."""
    tensor_parallel_size: int | None = None
    """``None`` auto-selects the largest power-of-two TP that divides the model's
    attention-head count and fits the slice's chip count."""
    chat_template_content: str | None = None
    """Inline Jinja chat template; resolved from a path/URL by the CLI."""
    cache_ttl_days: int = 14
    """Mirror HF models to a region-local GCS cache with this lifecycle TTL so repeat
    serves skip the HuggingFace download. ``0`` disables caching; ignored for gs:// paths."""
    timeout_hours: float = 24.0
    proxy_timeout_seconds: float = 600.0
    """Registered as endpoint metadata so the controller proxy waits this long for a
    single completion before returning 504; sized for long (e.g. reasoning) generations."""

    @property
    def accelerator_label(self) -> str:
        """Human-readable accelerator string for the dashboard and endpoint metadata.

        Reports the GPU slice (e.g. ``H100x8``) on the GPU path and the TPU type on
        the TPU path.
        """
        if self.gpu_count is not None:
            return f"{self.gpu_type}x{self.gpu_count}" if self.gpu_type else f"gpux{self.gpu_count}"
        return self.tpu_type or "unknown"

    @property
    def num_chips(self) -> int:
        """Chips on this slice: the GPU count, or the TPU type's per-VM chip count."""
        if self.gpu_count is not None:
            return self.gpu_count
        if self.tpu_type is None:
            raise ValueError("QuickServeConfig requires tpu_type on the TPU path (or gpu_count on the GPU path).")
        return get_tpu_topology(self.tpu_type).chips_per_vm


def select_tensor_parallel_size(
    num_attention_heads: int,
    num_chips: int,
    num_key_value_heads: int | None = None,
) -> int:
    """Pick the largest power-of-two tensor-parallel size valid for this model+slice.

    vLLM requires ``num_attention_heads`` to be divisible by the TP size and the
    KV-head count to be compatible (divides TP or is divisible by it); Levanter shards the
    same head axis across its model mesh axis. TPU slices expose power-of-two chip counts, so we
    search powers of two up to ``num_chips``. Models with odd or prime head counts fall back to
    TP 1.
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
        return json.loads(StoragePath(config_path).read_text())
    return AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()


def resolve_model_path(model: str, cache_ttl_days: int) -> str:
    """Resolve ``model`` to a path the backend can load, mirroring HF repos to a TTL'd GCS cache.

    HuggingFace repo ids are mirrored once to a region-local GCS cache under a distributed
    lock, so a later serve of the same model reads the snapshot from same-region GCS instead
    of re-downloading from HuggingFace; object-store paths are served directly.
    """
    return resolve_cached_model_path(model, cache_ttl_days=cache_ttl_days, cache_prefix=_MODEL_CACHE_PREFIX)


def detect_chat_support(base_url: str, model_id: str) -> bool:
    """Probe whether the served model accepts ``/v1/chat/completions``.

    Base/midtrained checkpoints ship no chat template, so the backend rejects chat
    requests; the dashboard defaults such models to completion mode.
    """
    try:
        response = requests.post(
            f"{base_url}{OPENAI_API_SUFFIX}/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            timeout=60,
        )
    except requests.RequestException as exc:
        logger.warning("Chat-support probe failed (%s); defaulting to completion mode.", exc)
        return False
    return response.status_code == 200


def build_model_spec(config: QuickServeConfig, model_path: str) -> ModelSpec:
    """Resolve the slice-and-model inputs the backend needs, inferring TP when unset."""
    num_chips = config.num_chips
    if config.tensor_parallel_size is not None:
        tensor_parallel_size = config.tensor_parallel_size
        logger.info(
            "quick-serve model=%s accelerator=%s chips=%d tensor_parallel_size=%d (user-specified)",
            config.model,
            config.accelerator_label,
            num_chips,
            tensor_parallel_size,
        )
    else:
        num_attention_heads, num_key_value_heads = read_attention_heads(model_path)
        tensor_parallel_size = select_tensor_parallel_size(num_attention_heads, num_chips, num_key_value_heads)
        logger.info(
            "quick-serve model=%s accelerator=%s chips=%d heads=%d kv_heads=%s -> tensor_parallel_size=%d",
            config.model,
            config.accelerator_label,
            num_chips,
            num_attention_heads,
            num_key_value_heads,
            tensor_parallel_size,
        )
    return ModelSpec(
        model=config.model,
        model_path=model_path,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        chat_template_content=config.chat_template_content,
    )


def _block_until_timeout(served: ServedModel, timeout_hours: float) -> None:
    """Block until the timeout elapses, failing early if the backend dies."""
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        served.check_alive()
        time.sleep(_TIMEOUT_POLL_SECONDS)
    logger.info("quick-serve reached its %.1fh timeout; shutting down.", timeout_hours)


def serve_in_job(config: QuickServeConfig) -> None:
    """Iris job entrypoint: boot the backend, serve the dashboard, register the endpoint, block."""
    configure_logging()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("serve_in_job must run inside an Iris job.")
    ctx = iris_ctx()
    port = ctx.get_port(config.port_name)
    advertise_host = job_info.advertise_host
    # Claim the dashboard's port now, before the backend launches: Iris' named-port range
    # overlaps the OS ephemeral range, so the backend's internal sockets could otherwise
    # squat it. Binding here reserves it for us until uvicorn takes over. Bind only
    # the advertised interface (the address the controller proxy connects to), not
    # all interfaces.
    serving_socket = bind_serving_socket(advertise_host, port)
    # On the k8s runtime (e.g. CoreWeave GPU pods) named ports are kernel-assigned:
    # ``get_port`` returns 0 and the real port is only known after binding. Register
    # the actual bound port so the controller proxy can reach the endpoint.
    serving_port = serving_socket.getsockname()[1]

    model_path = resolve_model_path(config.model, config.cache_ttl_days)
    spec = build_model_spec(config, model_path)
    accelerator = config.accelerator_label

    with config.backend.serve(spec) as served:
        has_chat_template = config.chat_template_content is not None or detect_chat_support(
            served.base_url, served.model_id
        )
        info = ServingInfo(
            model=served.model_id,
            backend=config.backend.name,
            tensor_parallel_size=spec.tensor_parallel_size,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            has_chat_template=has_chat_template,
            tpu_type=accelerator,
            endpoint=config.endpoint_name,
        )
        app = build_dashboard_app(upstream_base_url=served.base_url, model_id=served.model_id, info=info)
        with serve_app_background(app, serving_socket):
            address = f"http://{advertise_host}:{serving_port}"
            metadata = {
                "model": str(served.model_id),
                "kind": "quick-serve",
                "backend": config.backend.name,
                "accelerator": accelerator,
                "tensor_parallel_size": str(spec.tensor_parallel_size),
                PROXY_TIMEOUT_METADATA_KEY: str(config.proxy_timeout_seconds),
            }
            endpoint_id = ctx.registry.register(config.endpoint_name, address, metadata, access=config.access)
            logger.info(
                "Registered quick-serve endpoint name=%s backend=%s address=%s id=%s proxy_path=%s",
                config.endpoint_name,
                config.backend.name,
                address,
                endpoint_id,
                proxy_path(config.endpoint_name),
            )
            try:
                _block_until_timeout(served, config.timeout_hours)
            finally:
                try:
                    ctx.registry.unregister(endpoint_id)
                except Exception:
                    # Best-effort during teardown: the controller may already have
                    # dropped the endpoint when the task began terminating.
                    logger.warning("Failed to unregister quick-serve endpoint id=%s", endpoint_id, exc_info=True)
