# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import equinox as eqx
import jax
from jax import shard_map
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Bool, Float, Int

from levanter.grug.attention._core import AttentionMask
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, flash4_cute_kernel_config

_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


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
    current_start: jax.Array = jax.lax.associative_scan(jnp.maximum, start_positions, axis=1)
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


def _simple_causal_lower_bounds(
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int | None,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    if sliding_window is None:
        lower_bounds = jnp.zeros((1, seq_len), dtype=jnp.int32)
    else:
        lower_bounds = jnp.maximum(positions - (sliding_window - 1), 0)
    lower_bounds = jnp.broadcast_to(lower_bounds, (batch_size, seq_len))
    valid = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    return lower_bounds, valid


def _packed_self_attention_segment_ids(
    q: jax.Array,
    k: jax.Array,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> Int[Array, "B S"]:
    mask = _validate_causal_self_attention(q, k, mask, backend_name=backend_name)
    if mask.segment_ids is None:
        raise NotImplementedError(f"{backend_name} currently requires packed segment_ids.")

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


def _validate_causal_self_attention(
    q: jax.Array,
    k: jax.Array,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> AttentionMask:
    if isinstance(mask, jax.Array):
        raise NotImplementedError(f"{backend_name} does not support dense masks.")
    if not isinstance(mask, AttentionMask):
        raise NotImplementedError(f"{backend_name} requires an AttentionMask.")
    if not mask.is_causal:
        raise NotImplementedError(f"{backend_name} currently supports only causal self-attention.")
    if q.shape[0] != k.shape[0]:
        raise NotImplementedError(f"{backend_name} requires matching q/kv batch sizes.")
    if q.shape[1] != k.shape[1]:
        raise NotImplementedError(f"{backend_name} requires self-attention q_len == k_len.")
    if mask.sliding_window is not None and mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
    return mask


def _self_attention_lower_bounds(
    q: jax.Array,
    k: jax.Array,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    mask = _validate_causal_self_attention(q, k, mask, backend_name=backend_name)
    if mask.segment_ids is None:
        return _simple_causal_lower_bounds(
            batch_size=q.shape[0],
            seq_len=q.shape[1],
            sliding_window=mask.sliding_window,
        )

    q_segment_ids = _packed_self_attention_segment_ids(q, k, mask, backend_name=backend_name)
    return _packed_segment_causal_lower_bounds(
        q_segment_ids,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        sliding_window=mask.sliding_window,
    )


def _validate_head_layout(q: jax.Array, k: jax.Array, *, backend_name: str) -> None:
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"{backend_name} requires Hq divisible by Hkv, got q={q.shape}, k={k.shape}")


def _active_batch_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple[str, ...]:
    return tuple(axis for axis in _BATCH_AXES if axis in mesh.shape)


def _head_axis(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> str | None:
    if "model" not in mesh.shape:
        return None
    return "model"


def _assert_sequence_axis_unsharded(name: str, x: jax.Array) -> None:
    sharding = getattr(x, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return

    spec = tuple(sharding.spec)
    if len(spec) > 1 and spec[1] is not None:
        raise ValueError(
            f"FA4/CuTe shard_map requires unsharded sequence axis for {name}, got sharding {sharding.spec}."
        )


def _fa4_cute_attention_forward_sharded(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> jax.Array:
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return fa4_cute_attention_forward(
            q,
            k,
            v,
            lower_bounds,
            valid,
            sm_scale=sm_scale,
            kernel_config=kernel_config,
        )

    batch_axes = _active_batch_axes(mesh)
    if not batch_axes:
        return fa4_cute_attention_forward(
            q,
            k,
            v,
            lower_bounds,
            valid,
            sm_scale=sm_scale,
            kernel_config=kernel_config,
        )

    qkv_spec = P(batch_axes, None, _head_axis(mesh), None)
    metadata_spec = P(batch_axes, None)
    _assert_sequence_axis_unsharded("q", q)
    _assert_sequence_axis_unsharded("k", k)
    _assert_sequence_axis_unsharded("v", v)
    _assert_sequence_axis_unsharded("lower_bounds", lower_bounds)
    _assert_sequence_axis_unsharded("valid", valid)
    lower_bounds = reshard(lower_bounds, metadata_spec)
    valid = reshard(valid, metadata_spec)

    @shard_map(
        mesh=mesh,
        out_specs=qkv_spec,
        check_vma=False,
    )
    def _local_fa4_attention(q_local, k_local, v_local, lower_bounds_local, valid_local):
        return fa4_cute_attention_forward(
            q_local,
            k_local,
            v_local,
            lower_bounds_local,
            valid_local,
            sm_scale=sm_scale,
            kernel_config=kernel_config,
        )

    return _local_fa4_attention(q, k, v, lower_bounds, valid)


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


def _segmented_kernel_config(head_dim: int):
    arch = _gpu_compute_arch()
    kernel_config = flash4_cute_kernel_config(head_dim, arch=arch)

    # Upstream flash-attn-4 4.0.0b15 dense SM100 FA4 uses 128x128 tiles in
    # flash_attn/cute/interface.py. This Grug port is not that native SM100
    # kernel: it carries dynamic lower-bound metadata through the SM80/SM120
    # segmented fork. On B200 d5120 Grug shapes, 64x64 fwd/bwd is consistently
    # faster than both the prior 128x64/64x64 config and dense-upstream 128x128.
    if arch // 10 == 10 and head_dim == 128:
        return replace(kernel_config, forward_tile=(64, 64), backward_tile=(64, 64), num_threads=128)
    return kernel_config


def gpu_fa4_cute_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
) -> Float[Array, "B Q Hq D"]:
    """Run causal self-attention through a FlashAttention-4/CuTe JAX FFI backend."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_cute_attention requires the JAX GPU backend.")

    _validate_head_layout(q, k, backend_name="gpu_fa4_cute_attention")
    lower_bounds, valid = _self_attention_lower_bounds(
        q,
        k,
        mask,
        backend_name="gpu_fa4_cute_attention",
    )
    kernel_config = _segmented_kernel_config(q.shape[-1])

    return _fa4_cute_attention_forward_sharded(
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
