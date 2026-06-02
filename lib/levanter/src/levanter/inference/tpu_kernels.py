# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend dispatch for paged attention inference kernels."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

from draccus import ChoiceRegistry
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray

from levanter.inference.page_table import PageBatchInfo
from levanter.layers.kv_cache import KvPageCache


class PagedAttentionConfig(ChoiceRegistry):
    """Execution policy for paged attention during decode."""

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "auto"


@PagedAttentionConfig.register_subclass("auto")
@dataclasses.dataclass(frozen=True, slots=True)
class AutoPagedAttentionConfig(PagedAttentionConfig):
    """Pick the fastest supported backend for the current platform."""

    fail_on_reference_fallback: bool = True


@PagedAttentionConfig.register_subclass("reference")
@dataclasses.dataclass(frozen=True, slots=True)
class ReferencePagedAttentionConfig(PagedAttentionConfig):
    """Use the pure JAX reference paged-attention path."""

    fail_on_reference_fallback: bool = True


@PagedAttentionConfig.register_subclass("tpu-inference")
@dataclasses.dataclass(frozen=True, slots=True)
class TpuInferencePagedAttentionConfig(PagedAttentionConfig):
    """Use the tpu-inference paged-attention kernel."""

    fail_on_reference_fallback: bool = True
    out_dtype: str | None = None
    preserve_attention_output_dtype: bool = False
    vmem_limit_bytes: int | None = None


@PagedAttentionConfig.register_subclass("jax-rpa")
@dataclasses.dataclass(frozen=True, slots=True)
class JaxRpaPagedAttentionConfig(PagedAttentionConfig):
    """Use Levanter's JAX/Pallas ragged paged-attention path."""

    fail_on_reference_fallback: bool = True
    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    vmem_limit_bytes: int | None = None


class UnsupportedPagedAttentionBackend(RuntimeError):
    """Raised when a selected paged-attention backend cannot run for the current platform or shape."""


class PagedAttentionFallbackWarning(UserWarning):
    """Warning emitted when backend dispatch falls back from one configured backend to another."""


@dataclasses.dataclass(frozen=True, slots=True)
class PagedAttentionShape:
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
        raise UnsupportedPagedAttentionBackend(f"Could not determine JAX backend: {exc}") from exc


def _current_device_kind() -> str:
    try:
        return str(getattr(jax.devices()[0], "device_kind", ""))
    except Exception:
        return ""


def _is_tpu() -> bool:
    return _current_platform() == "tpu"


def _has_nonempty_mesh() -> bool:
    return not jax.sharding.get_abstract_mesh().empty


def _config_name(config: PagedAttentionConfig) -> str:
    if isinstance(config, AutoPagedAttentionConfig):
        return "auto"
    if isinstance(config, ReferencePagedAttentionConfig):
        return "reference"
    if isinstance(config, TpuInferencePagedAttentionConfig):
        return "tpu-inference"
    if isinstance(config, JaxRpaPagedAttentionConfig):
        return "jax-rpa"
    raise TypeError(f"Unknown paged attention config type: {type(config)}")


def paged_attention_config_name(config: PagedAttentionConfig) -> str:
    """Return the draccus choice name for a paged-attention config."""

    return _config_name(config)


def _expand_config(config: PagedAttentionConfig) -> tuple[PagedAttentionConfig, ...]:
    if not isinstance(config, AutoPagedAttentionConfig):
        return (config,)

    if not _is_tpu():
        return (ReferencePagedAttentionConfig(fail_on_reference_fallback=config.fail_on_reference_fallback),)

    configs: list[PagedAttentionConfig] = [
        TpuInferencePagedAttentionConfig(fail_on_reference_fallback=config.fail_on_reference_fallback)
    ]
    if _has_nonempty_mesh():
        configs.append(JaxRpaPagedAttentionConfig(fail_on_reference_fallback=config.fail_on_reference_fallback))
    else:
        configs.append(ReferencePagedAttentionConfig(fail_on_reference_fallback=False))
    return tuple(configs)


def available_paged_attention_configs() -> tuple[PagedAttentionConfig, ...]:
    """Return paged-attention configs importable and platform-supported in the current process, excluding `auto`."""

    platform = _current_platform()
    if platform != "tpu":
        return (ReferencePagedAttentionConfig(),)

    configs: list[PagedAttentionConfig] = []
    from levanter.inference import tpu_inference_adapter  # noqa: PLC0415

    if tpu_inference_adapter.is_available():
        configs.append(TpuInferencePagedAttentionConfig())

    from levanter.layers import attention as attention_module  # noqa: PLC0415

    if _has_nonempty_mesh() and attention_module.tpu_ragged_paged_attention is not None:
        configs.append(JaxRpaPagedAttentionConfig())

    return tuple(configs)


def paged_attention_supports_shape(
    config: PagedAttentionConfig,
    shape: PagedAttentionShape,
) -> tuple[bool, str | None]:
    """Return whether `config` supports `shape`, plus a rejection reason when unsupported."""

    if isinstance(config, AutoPagedAttentionConfig):
        return False, "AUTO is a dispatch policy, not a concrete backend"
    if isinstance(config, ReferencePagedAttentionConfig):
        return True, None
    if shape.platform != "tpu":
        return False, f"{_config_name(config)} only runs on TPU"
    if "tpu v2" in shape.device_kind.lower() or "tpu v3" in shape.device_kind.lower():
        return False, f"{_config_name(config)} does not support {shape.device_kind}"
    if shape.dtype != jnp.bfloat16:
        return False, f"initial {_config_name(config)} target only supports bf16 KV cache"
    if shape.page_size != 128:
        return False, "initial Qwen3 8B target requires page_size=128"
    if shape.head_size != 128:
        return False, "initial Qwen3 8B target requires head_size=128"
    if shape.num_q_heads != 32 or shape.num_kv_heads != 8:
        return False, "initial Qwen3 8B target requires 32 query heads and 8 KV heads"
    if shape.max_model_len != 4096:
        return False, "initial Qwen3 8B target requires max_model_len=4096"
    return True, None


def paged_attention_preserves_output_dtype(config: PagedAttentionConfig) -> bool:
    """Return whether the attention output projection should see the backend output dtype."""

    return isinstance(config, TpuInferencePagedAttentionConfig) and config.preserve_attention_output_dtype


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


def _unsupported(config: PagedAttentionConfig, reason: str) -> UnsupportedPagedAttentionBackend:
    return UnsupportedPagedAttentionBackend(f"{_config_name(config)} paged attention backend is unsupported: {reason}")


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
    from levanter.layers.attention import default_ragged_paged_attention  # noqa: PLC0415

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
    config: JaxRpaPagedAttentionConfig,
) -> tuple[NamedArray, KvPageCache]:
    if not _is_tpu():
        raise _unsupported(config, "JAX RPA only runs on TPU")

    from levanter.layers.attention import _do_tpu_ragged_paged_attention  # noqa: PLC0415

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
        num_kv_pages_per_block=config.num_kv_pages_per_block,
        num_queries_per_block=config.num_queries_per_block,
        vmem_limit_bytes=config.vmem_limit_bytes,
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
    config: TpuInferencePagedAttentionConfig,
) -> tuple[NamedArray, KvPageCache]:
    if not _is_tpu():
        raise _unsupported(config, "tpu-inference only runs on TPU")

    from levanter.inference import tpu_inference_adapter  # noqa: PLC0415

    if not tpu_inference_adapter.is_available():
        raise _unsupported(
            config,
            "the tpu-inference package or one of its runtime dependencies is not importable",
        )

    out_dtype = None
    if config.out_dtype is not None:
        out_dtype = jnp.dtype(config.out_dtype)

    try:
        return tpu_inference_adapter.paged_attention_with_kv_update(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            out_dtype=out_dtype,
            vmem_limit_bytes=config.vmem_limit_bytes,
        )
    except ImportError as exc:
        raise _unsupported(
            config,
            f"the tpu-inference kernel import failed: {exc}",
        ) from exc


def _run_backend(
    backend_config: PagedAttentionConfig,
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
) -> tuple[NamedArray, KvPageCache]:
    if isinstance(backend_config, ReferencePagedAttentionConfig):
        if _is_tpu() and backend_config.fail_on_reference_fallback:
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
    if isinstance(backend_config, JaxRpaPagedAttentionConfig):
        return _run_jax_rpa_backend(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            config=backend_config,
        )
    if isinstance(backend_config, TpuInferencePagedAttentionConfig):
        return _run_tpu_inference_backend(
            q,
            new_k,
            new_v,
            kv_cache,
            batch_info,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            config=backend_config,
        )
    if isinstance(backend_config, AutoPagedAttentionConfig):
        raise ValueError("AutoPagedAttentionConfig must be expanded before backend dispatch")
    raise TypeError(f"Unknown paged attention config type: {type(backend_config)}")


def paged_attention_with_kv_update(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    config: PagedAttentionConfig,
) -> tuple[NamedArray, KvPageCache]:
    """Compute paged attention for packed tokens and return the updated KV cache."""

    _validate_no_duplicate_token_dests(batch_info)
    backend_configs = _expand_config(config)
    failures: list[Exception] = []

    for index, backend_config in enumerate(backend_configs):
        try:
            return _run_backend(
                backend_config,
                q,
                new_k,
                new_v,
                kv_cache,
                batch_info,
                sm_scale=sm_scale,
                soft_cap=soft_cap,
            )
        except (UnsupportedPagedAttentionBackend, ValueError) as exc:
            failures.append(exc)
            if index == len(backend_configs) - 1:
                raise
            next_backend_config = backend_configs[index + 1]
            warnings.warn(
                f"{_config_name(backend_config)} paged attention backend failed; "
                f"trying {_config_name(next_backend_config)}: {exc}",
                PagedAttentionFallbackWarning,
                stacklevel=2,
            )

    raise RuntimeError(f"No configured paged attention backend could run: {failures}")
