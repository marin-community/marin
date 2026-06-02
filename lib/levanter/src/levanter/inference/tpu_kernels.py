# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend dispatch for paged TPU inference kernels."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray

from levanter.inference.page_table import PageBatchInfo
from levanter.layers.kv_cache import KvPageCache


class TpuPagedAttentionBackend(StrEnum):
    AUTO = "auto"
    TPU_INFERENCE = "tpu_inference"
    JAX_RPA = "jax_rpa"
    REFERENCE = "reference"


@dataclasses.dataclass(frozen=True, slots=True)
class TpuPagedAttentionConfig:
    backend: TpuPagedAttentionBackend | Sequence[TpuPagedAttentionBackend] = TpuPagedAttentionBackend.AUTO
    allow_autotune: bool = False
    fail_on_reference_fallback: bool = True
    tpu_inference_out_dtype: str | None = None
    preserve_attention_output_dtype: bool = False


class UnsupportedTpuPagedAttentionBackend(RuntimeError):
    """Raised when a selected paged-attention backend cannot run for the current platform or shape."""


class TpuPagedAttentionFallbackWarning(UserWarning):
    """Warning emitted when backend dispatch falls back from one configured backend to another."""


@dataclasses.dataclass(frozen=True, slots=True)
class TpuPagedAttentionShape:
    platform: str
    device_kind: str
    dtype: jnp.dtype
    page_size: int
    head_size: int
    num_q_heads: int
    num_kv_heads: int
    q_heads_per_group: int
    max_model_len: int
    tensor_parallel_size: int


def _current_platform() -> str:
    try:
        return jax.default_backend()
    except Exception as exc:
        raise UnsupportedTpuPagedAttentionBackend(f"Could not determine JAX backend: {exc}") from exc


def _current_device_kind() -> str:
    try:
        return str(getattr(jax.devices()[0], "device_kind", ""))
    except Exception:
        return ""


def _is_tpu() -> bool:
    return _current_platform() == "tpu"


def _normalize_backend(backend: TpuPagedAttentionBackend | str) -> TpuPagedAttentionBackend:
    if isinstance(backend, TpuPagedAttentionBackend):
        return backend
    return TpuPagedAttentionBackend(backend)


def _backend_order(config: TpuPagedAttentionConfig) -> tuple[TpuPagedAttentionBackend, ...]:
    backend = config.backend
    if isinstance(backend, TpuPagedAttentionBackend | str):
        selected = (_normalize_backend(backend),)
    else:
        selected = tuple(_normalize_backend(item) for item in backend)
        if len(selected) == 0:
            raise ValueError("tpu_paged_attention backend sequence must not be empty")

    expanded: list[TpuPagedAttentionBackend] = []
    for item in selected:
        if item == TpuPagedAttentionBackend.AUTO:
            expanded.extend(
                [TpuPagedAttentionBackend.TPU_INFERENCE] if _is_tpu() else [TpuPagedAttentionBackend.REFERENCE]
            )
        else:
            expanded.append(item)
    return tuple(expanded)


def available_tpu_paged_attention_backends() -> tuple[TpuPagedAttentionBackend, ...]:
    """Return backends importable and platform-supported in the current process, excluding `AUTO`."""

    platform = _current_platform()
    if platform != "tpu":
        return (TpuPagedAttentionBackend.REFERENCE,)

    backends: list[TpuPagedAttentionBackend] = []
    from levanter.inference import tpu_inference_adapter

    if tpu_inference_adapter.is_available():
        backends.append(TpuPagedAttentionBackend.TPU_INFERENCE)

    from levanter.layers import attention as attention_module

    if attention_module.tpu_ragged_paged_attention is not None:
        backends.append(TpuPagedAttentionBackend.JAX_RPA)

    return tuple(backends)


def tpu_paged_attention_supports_shape(
    backend: TpuPagedAttentionBackend,
    shape: TpuPagedAttentionShape,
) -> tuple[bool, str | None]:
    """Return whether `backend` supports `shape`, plus a rejection reason when unsupported."""

    backend = _normalize_backend(backend)
    if backend == TpuPagedAttentionBackend.AUTO:
        return False, "AUTO is a dispatch policy, not a concrete backend"
    if backend == TpuPagedAttentionBackend.REFERENCE:
        return True, None
    if shape.platform != "tpu":
        return False, f"{backend.value} only runs on TPU"
    if "tpu v2" in shape.device_kind.lower() or "tpu v3" in shape.device_kind.lower():
        return False, f"{backend.value} does not support {shape.device_kind}"
    if shape.dtype != jnp.bfloat16:
        return False, f"initial {backend.value} target only supports bf16 KV cache"
    if shape.page_size != 128:
        return False, "initial Qwen3 8B target requires page_size=128"
    if shape.head_size != 128:
        return False, "initial Qwen3 8B target requires head_size=128"
    if shape.num_q_heads != 32 or shape.num_kv_heads != 8:
        return False, "initial Qwen3 8B target requires 32 query heads and 8 KV heads"
    if shape.tensor_parallel_size != 4:
        return False, "initial Qwen3 8B target requires tensor_parallel_size=4"
    if shape.max_model_len != 4096:
        return False, "initial Qwen3 8B target requires max_model_len=4096"
    return True, None


def _concrete_array(value: Any) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(jax.device_get(value))
    except Exception:
        return None


def _validate_no_duplicate_token_dests(batch_info: PageBatchInfo) -> None:
    dests = _concrete_array(batch_info.new_token_dests.array)
    num_tokens = _concrete_array(batch_info.num_new_tokens)
    if dests is None or num_tokens is None:
        return
    valid_dests = dests[: int(num_tokens)].reshape(-1)
    valid_dests = valid_dests[valid_dests >= 0]
    if len(valid_dests) != len(set(valid_dests.tolist())):
        raise ValueError("batch_info.new_token_dests contains duplicate valid destinations")


def _unsupported(backend: TpuPagedAttentionBackend, reason: str) -> UnsupportedTpuPagedAttentionBackend:
    return UnsupportedTpuPagedAttentionBackend(f"{backend.value} paged attention backend is unsupported: {reason}")


def _run_reference_backend(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
) -> tuple[NamedArray, KvPageCache]:
    from levanter.layers.attention import default_ragged_paged_attention

    kv_cache = kv_cache.update(batch_info, new_k, new_v)
    attn = default_ragged_paged_attention(
        q,
        kv_cache.kv_pages,
        batch_info.seq_lens,
        batch_info.page_indices,
        batch_info.cu_q_lens.array,
        batch_info.num_seqs,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
    )
    return attn, kv_cache


def _run_jax_rpa_backend(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    num_kv_pages_per_block: int | None,
    num_queries_per_block: int | None,
    vmem_limit_bytes: int | None,
) -> tuple[NamedArray, KvPageCache]:
    if not _is_tpu():
        raise _unsupported(TpuPagedAttentionBackend.JAX_RPA, "JAX RPA only runs on TPU")

    from levanter.layers.attention import _do_tpu_ragged_paged_attention

    kv_cache = kv_cache.update(batch_info, new_k, new_v)
    attn = _do_tpu_ragged_paged_attention(
        q,
        kv_cache.kv_pages,
        batch_info.seq_lens,
        batch_info.page_indices,
        batch_info.cu_q_lens,
        batch_info.num_seqs,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return attn, kv_cache


def _run_tpu_inference_backend(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    config: TpuPagedAttentionConfig,
    vmem_limit_bytes: int | None,
) -> tuple[NamedArray, KvPageCache]:
    if not _is_tpu():
        raise _unsupported(TpuPagedAttentionBackend.TPU_INFERENCE, "tpu-inference only runs on TPU")

    from levanter.inference import tpu_inference_adapter

    if not tpu_inference_adapter.is_available():
        raise _unsupported(TpuPagedAttentionBackend.TPU_INFERENCE, "the tpu-inference package is not installed")

    out_dtype = None
    if config.tpu_inference_out_dtype is not None:
        out_dtype = jnp.dtype(config.tpu_inference_out_dtype)

    return tpu_inference_adapter.paged_attention_with_kv_update(
        q,
        new_k,
        new_v,
        kv_cache,
        batch_info,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )


def _run_backend(
    backend: TpuPagedAttentionBackend,
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    config: TpuPagedAttentionConfig,
    num_kv_pages_per_block: int | None,
    num_queries_per_block: int | None,
    vmem_limit_bytes: int | None,
) -> tuple[NamedArray, KvPageCache]:
    if backend == TpuPagedAttentionBackend.REFERENCE:
        if _is_tpu() and config.fail_on_reference_fallback:
            raise ValueError(
                "REFERENCE paged attention backend was selected on TPU with fail_on_reference_fallback=True"
            )
        return _run_reference_backend(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
        )
    if backend == TpuPagedAttentionBackend.JAX_RPA:
        return _run_jax_rpa_backend(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        )
    if backend == TpuPagedAttentionBackend.TPU_INFERENCE:
        return _run_tpu_inference_backend(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            config=config,
            vmem_limit_bytes=vmem_limit_bytes,
        )
    raise ValueError(f"Unsupported paged attention backend: {backend}")


def paged_attention_with_kv_update(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    config: TpuPagedAttentionConfig,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[NamedArray, KvPageCache]:
    """Compute paged attention for packed tokens and return the updated KV cache."""

    _validate_no_duplicate_token_dests(batch_info)
    backends = _backend_order(config)
    failures: list[Exception] = []

    for index, backend in enumerate(backends):
        try:
            return _run_backend(
                backend,
                q,
                new_k,
                new_v,
                kv_cache,
                batch_info,
                sm_scale=sm_scale,
                soft_cap=soft_cap,
                config=config,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
                vmem_limit_bytes=vmem_limit_bytes,
            )
        except (UnsupportedTpuPagedAttentionBackend, ValueError) as exc:
            failures.append(exc)
            if index == len(backends) - 1:
                raise
            warnings.warn(
                f"{backend.value} paged attention backend failed; trying {backends[index + 1].value}: {exc}",
                TpuPagedAttentionFallbackWarning,
                stacklevel=2,
            )

    raise RuntimeError(f"No configured paged attention backend could run: {failures}")
