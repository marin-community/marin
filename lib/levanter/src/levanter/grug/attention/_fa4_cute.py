# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from levanter.grug.attention._core import AttentionMask
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward


@dataclass(frozen=True)
class _Flash4CuteKernelConfig:
    forward_tile: tuple[int, int]
    backward_tile: tuple[int, int]
    num_threads: int


def _batched_segment_ids(segment_ids: jax.Array, *, batch_size: int, seq_len: int) -> jax.Array:
    if segment_ids.ndim == 1:
        if segment_ids.shape[0] != seq_len:
            raise ValueError(f"1D segment_ids must match sequence length {seq_len}, got {segment_ids.shape}")
        segment_ids = jnp.broadcast_to(segment_ids[None, :], (batch_size, seq_len))
    elif segment_ids.ndim == 2:
        if segment_ids.shape[0] not in (1, batch_size) or segment_ids.shape[1] != seq_len:
            raise ValueError(f"2D segment_ids must have shape [1|{batch_size}, {seq_len}], got {segment_ids.shape}")
        if segment_ids.shape[0] == 1 and batch_size != 1:
            segment_ids = jnp.broadcast_to(segment_ids, (batch_size, seq_len))
    else:
        raise ValueError(f"segment_ids must be 1D or 2D, got ndim={segment_ids.ndim}")
    return segment_ids


def _segment_starts(segment_ids: jax.Array) -> jax.Array:
    valid = segment_ids >= 0
    previous = jnp.concatenate([segment_ids[:, :1], segment_ids[:, :-1]], axis=1)
    first = jnp.zeros_like(valid).at[:, 0].set(True)
    return valid & (first | (segment_ids != previous))


def _packed_segment_start_positions(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
) -> Int[Array, "B S"]:
    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    valid = segment_ids >= 0
    starts = _segment_starts(segment_ids)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    start_positions = jnp.where(starts, positions, 0)
    current_start = jax.lax.associative_scan(jnp.maximum, start_positions, axis=1)
    return jnp.where(valid, current_start, seq_len)


def _packed_segment_causal_lower_bounds(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int | None,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    """Return per-token inclusive key lower bounds for dynamic packed causal attention."""
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    valid = segment_ids >= 0
    lower_bounds = _packed_segment_start_positions(
        segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if sliding_window is not None:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        window_lower_bounds = positions - (sliding_window - 1)
        lower_bounds = jnp.maximum(lower_bounds, window_lower_bounds)
    return jnp.where(valid, lower_bounds, seq_len), valid


def _packed_self_attention_segment_ids(
    q: jax.Array,
    k: jax.Array,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> Int[Array, "B S"]:
    if isinstance(mask, jax.Array):
        raise NotImplementedError(f"{backend_name} does not support dense masks.")
    if not isinstance(mask, AttentionMask) or mask.segment_ids is None:
        raise NotImplementedError(f"{backend_name} currently requires packed segment_ids.")
    if not mask.is_causal:
        raise NotImplementedError(f"{backend_name} currently supports only causal packed self-attention.")
    if q.shape[0] != k.shape[0]:
        raise NotImplementedError(f"{backend_name} requires matching q/kv batch sizes.")
    if q.shape[1] != k.shape[1]:
        raise NotImplementedError(f"{backend_name} requires self-attention q_len == k_len.")
    if mask.sliding_window is not None and mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")

    q_segment_ids, kv_segment_ids = mask.segment_ids
    same_segment_ids = q_segment_ids is kv_segment_ids
    q_segment_ids = _batched_segment_ids(q_segment_ids, batch_size=q.shape[0], seq_len=q.shape[1])
    if not same_segment_ids:
        kv_segment_ids = _batched_segment_ids(kv_segment_ids, batch_size=k.shape[0], seq_len=k.shape[1])
        q_segment_ids = eqx.error_if(
            q_segment_ids,
            jnp.any(q_segment_ids != kv_segment_ids),
            f"{backend_name} requires matching q/kv segment_ids for packed self-attention.",
        )
    return q_segment_ids


def _validate_head_layout(q: jax.Array, k: jax.Array, *, backend_name: str) -> None:
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"{backend_name} requires Hq divisible by Hkv, got q={q.shape}, k={k.shape}")


def _flash4_cute_kernel_config(
    head_dim: int,
    *,
    arch: int,
) -> _Flash4CuteKernelConfig:
    arch_family = arch // 10
    if arch_family == 10:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
        )
    if arch_family == 12:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(64, 64),
            num_threads=128,
        )
    if arch_family == 8:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 64),
            backward_tile=(128, 64),
            num_threads=128,
        )
    if arch_family == 9:
        return _Flash4CuteKernelConfig(
            forward_tile=(128, 128 if head_dim <= 64 else 64),
            backward_tile=(128, 64),
            num_threads=128,
        )
    raise NotImplementedError(f"FA4/CuTe attention does not support SM{arch}.")


def _gpu_compute_arch() -> int:
    for device in jax.local_devices(backend="gpu"):
        compute_capability = getattr(device, "compute_capability", None)
        if callable(compute_capability):
            compute_capability = compute_capability()
        if isinstance(compute_capability, tuple) and len(compute_capability) >= 2:
            return int(compute_capability[0]) * 10 + int(compute_capability[1])
        if isinstance(compute_capability, str):
            major, _, minor = compute_capability.partition(".")
            if major.isdigit() and minor.isdigit():
                return int(major) * 10 + int(minor)
    raise RuntimeError("Could not determine CUDA compute capability for FA4/CuTe attention.")


def gpu_fa4_cute_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Run dynamic packed segment attention through a FlashAttention-4/CuTe JAX FFI backend."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_cute_attention requires the JAX GPU backend.")

    _validate_head_layout(q, k, backend_name="gpu_fa4_cute_attention")
    q_segment_ids = _packed_self_attention_segment_ids(q, k, mask, backend_name="gpu_fa4_cute_attention")
    assert isinstance(mask, AttentionMask)
    lower_bounds, valid = _packed_segment_causal_lower_bounds(
        q_segment_ids,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        sliding_window=mask.sliding_window,
    )
    kernel_config = _flash4_cute_kernel_config(q.shape[-1], arch=_gpu_compute_arch())

    return fa4_cute_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        kernel_config=kernel_config,
    )


__all__ = [
    "gpu_fa4_cute_attention",
]
