# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import jax
from jax import shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Bool, Float, Int

from levanter.grug.attention._core import AttentionMask, Fa4CuteMetadata
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


def _segment_starts(segment_ids: jax.Array, valid: jax.Array) -> jax.Array:
    starts_tail = valid[:, 1:] & (segment_ids[:, 1:] != segment_ids[:, :-1])
    return jnp.concatenate([valid[:, :1], starts_tail], axis=1)


def _packed_segment_start_positions_and_valid(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    segment_ids = _batched_segment_ids(segment_ids, batch_size=batch_size, seq_len=seq_len)
    valid = segment_ids >= 0
    starts = _segment_starts(segment_ids, valid)
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    start_positions = jnp.where(starts, positions, 0)
    current_start: jax.Array = jax.lax.associative_scan(jnp.maximum, start_positions, axis=1)
    return jnp.where(valid, current_start, seq_len), valid


def _packed_segment_start_positions(
    segment_ids: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
) -> Int[Array, "B S"]:
    lower_bounds, _ = _packed_segment_start_positions_and_valid(
        segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    return lower_bounds


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

    lower_bounds, valid = _packed_segment_start_positions_and_valid(
        segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if sliding_window is not None and sliding_window < seq_len:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        window_lower_bounds = positions - (sliding_window - 1)
        lower_bounds = jnp.maximum(lower_bounds, window_lower_bounds)
    return jnp.where(valid, lower_bounds, seq_len), valid


def _causal_lower_bounds(
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int | None,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    """Return per-token lower bounds for unsegmented causal self-attention."""
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    lower_bounds = jnp.zeros((1, seq_len), dtype=jnp.int32)
    if sliding_window is not None and sliding_window < seq_len:
        positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        lower_bounds = jnp.maximum(lower_bounds, positions - (sliding_window - 1))
    lower_bounds = jnp.broadcast_to(lower_bounds, (batch_size, seq_len))
    valid = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    return lower_bounds, valid


def _metadata_reshard_if_mesh(x: jax.Array) -> jax.Array:
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return x
    return reshard(x, P(_BATCH_AXES, None))


def _active_batch_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple[str, ...]:
    return tuple(axis for axis in _BATCH_AXES if axis in mesh.shape)


def _head_axis(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> str | None:
    if "model" not in mesh.shape or int(mesh.shape["model"]) == 1:
        return None
    return "model"


def _fa4_cute_attention_forward_sharded(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
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
            sm_scale=sm_scale,
            kernel_config=kernel_config,
        )

    q_spec = P(batch_axes, None, _head_axis(mesh), None)
    kv_spec = P(batch_axes, None, None, None)
    metadata_spec = P(batch_axes, None)
    q = reshard(q, q_spec)
    k = reshard(k, kv_spec)
    v = reshard(v, kv_spec)
    lower_bounds = reshard(lower_bounds, metadata_spec)

    @shard_map(
        mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec, metadata_spec),
        out_specs=q_spec,
        check_vma=False,
    )
    def _local_fa4_attention(q_local, k_local, v_local, lower_bounds_local):
        return fa4_cute_attention_forward(
            q_local,
            k_local,
            v_local,
            lower_bounds_local,
            sm_scale=sm_scale,
            kernel_config=kernel_config,
        )

    return _local_fa4_attention(q, k, v, lower_bounds)


def _validate_causal_self_attention_mask(
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
    if mask.sliding_window is not None and mask.sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
    return mask


def _validate_packed_self_attention_mask(
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> AttentionMask:
    mask = _validate_causal_self_attention_mask(mask, backend_name=backend_name)
    if mask.segment_ids is None:
        raise NotImplementedError(f"{backend_name} currently requires packed segment_ids.")
    return mask


def _packed_self_attention_segment_ids_from_mask(
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    batch_size: int,
    seq_len: int,
    backend_name: str,
) -> Int[Array, "B S"]:
    mask = _validate_packed_self_attention_mask(mask, backend_name=backend_name)
    segment_ids = mask.segment_ids
    assert segment_ids is not None
    q_segment_ids, _ = segment_ids
    return _batched_segment_ids(q_segment_ids, batch_size=batch_size, seq_len=seq_len)


def _packed_self_attention_segment_ids(
    q: jax.Array,
    k: jax.Array,
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    backend_name: str,
) -> Int[Array, "B S"]:
    _validate_self_attention_shape(q, k, backend_name=backend_name)
    return _packed_self_attention_segment_ids_from_mask(
        mask,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        backend_name=backend_name,
    )


def _validate_head_layout(q: jax.Array, k: jax.Array, *, backend_name: str) -> None:
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"{backend_name} requires Hq divisible by Hkv, got q={q.shape}, k={k.shape}")


def _validate_self_attention_shape(q: jax.Array, k: jax.Array, *, backend_name: str) -> None:
    if q.shape[0] != k.shape[0]:
        raise NotImplementedError(f"{backend_name} requires matching q/kv batch sizes.")
    if q.shape[1] != k.shape[1]:
        raise NotImplementedError(f"{backend_name} requires self-attention q_len == k_len.")


def _fa4_metadata_matches(mask: AttentionMask, *, batch_size: int, seq_len: int) -> bool:
    metadata = mask.fa4_cute_metadata
    return (
        metadata is not None
        and metadata.lower_bounds.shape == (batch_size, seq_len)
        and metadata.sliding_window == mask.sliding_window
    )


def with_fa4_cute_metadata(
    mask: AttentionMask,
    *,
    batch_size: int,
    seq_len: int,
) -> AttentionMask:
    with jax.named_scope("fa4_cute_metadata"):
        mask = _validate_causal_self_attention_mask(mask, backend_name="gpu_fa4_cute_attention")
        if _fa4_metadata_matches(mask, batch_size=batch_size, seq_len=seq_len):
            return mask
        if mask.segment_ids is None:
            lower_bounds, _ = _causal_lower_bounds(
                batch_size=batch_size,
                seq_len=seq_len,
                sliding_window=mask.sliding_window,
            )
        else:
            q_segment_ids = _packed_self_attention_segment_ids_from_mask(
                mask,
                batch_size=batch_size,
                seq_len=seq_len,
                backend_name="gpu_fa4_cute_attention",
            )
            lower_bounds, _ = _packed_segment_causal_lower_bounds(
                q_segment_ids,
                batch_size=batch_size,
                seq_len=seq_len,
                sliding_window=mask.sliding_window,
            )
    lower_bounds = _metadata_reshard_if_mesh(lower_bounds)
    return AttentionMask(
        is_causal=mask.is_causal,
        segment_ids=mask.segment_ids,
        thd_segment_metadata=mask.thd_segment_metadata,
        fa4_cute_metadata=Fa4CuteMetadata(
            lower_bounds=lower_bounds,
            sliding_window=mask.sliding_window,
        ),
        sliding_window=mask.sliding_window,
    )


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
    """Run dynamic packed segment attention through a FlashAttention-4/CuTe JAX FFI backend."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_cute_attention requires the JAX GPU backend.")

    _validate_head_layout(q, k, backend_name="gpu_fa4_cute_attention")
    _validate_self_attention_shape(q, k, backend_name="gpu_fa4_cute_attention")
    mask = _validate_causal_self_attention_mask(mask, backend_name="gpu_fa4_cute_attention")
    with jax.named_scope("fa4_cute_prepare_metadata"):
        if mask.fa4_cute_metadata is not None:
            if mask.fa4_cute_metadata.lower_bounds.shape != q.shape[:2]:
                raise ValueError(
                    "gpu_fa4_cute_attention metadata shape must match the q batch and sequence dimensions, "
                    f"got lower_bounds={mask.fa4_cute_metadata.lower_bounds.shape}, q={q.shape}"
                )
            if mask.fa4_cute_metadata.sliding_window != mask.sliding_window:
                raise ValueError(
                    "gpu_fa4_cute_attention metadata sliding_window must match the attention mask, "
                    f"got metadata={mask.fa4_cute_metadata.sliding_window}, mask={mask.sliding_window}"
                )
            lower_bounds = mask.fa4_cute_metadata.lower_bounds
        elif mask.segment_ids is not None:
            q_segment_ids = _packed_self_attention_segment_ids(q, k, mask, backend_name="gpu_fa4_cute_attention")
            lower_bounds, _ = _packed_segment_causal_lower_bounds(
                q_segment_ids,
                batch_size=q.shape[0],
                seq_len=q.shape[1],
                sliding_window=mask.sliding_window,
            )
            lower_bounds = _metadata_reshard_if_mesh(lower_bounds)
        else:
            lower_bounds, _ = _causal_lower_bounds(
                batch_size=q.shape[0],
                seq_len=q.shape[1],
                sliding_window=mask.sliding_window,
            )
            lower_bounds = _metadata_reshard_if_mesh(lower_bounds)
    kernel_config = _segmented_kernel_config(q.shape[-1])

    with jax.named_scope("fa4_cute_kernel"):
        return _fa4_cute_attention_forward_sharded(
            q,
            k,
            v,
            lower_bounds,
            sm_scale=1.0 / math.sqrt(q.shape[-1]),
            kernel_config=kernel_config,
        )


__all__ = [
    "gpu_fa4_cute_attention",
    "with_fa4_cute_metadata",
]
