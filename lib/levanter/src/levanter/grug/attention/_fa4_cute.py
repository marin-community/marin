# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace
from typing import get_args

import equinox as eqx
import jax
import numpy as np
from jax import shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Bool, Float, Int

from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention._core import AttentionMask, ContextAttentionLayout
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import (
    SM100_GQA_RATIOS,
    SM100_HEAD_DIM,
    Flash4CuteKernelConfig,
    flash4_cute_kernel_config,
    sm100_flash4_cute_kernel_config,
)
from levanter.sharding import partition_spec_of, partitioned_dims


def _replicate_metadata(x: jax.Array) -> jax.Array:
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return x
    return reshard(x, P(None, None))


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
    return _replicate_sequence_axis(segment_ids)


def _replicate_sequence_axis(x: jax.Array) -> jax.Array:
    """Replicate a ``[B, S]`` metadata array over sequence, preserving batch sharding."""
    spec = partition_spec_of(x)
    if spec is None or len(spec) < 2 or spec[1] is None:
        return x
    return reshard(x, P(spec[0], None))


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
    positions = _replicate_metadata(jnp.arange(seq_len, dtype=jnp.int32)[None, :])
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
        positions = _replicate_metadata(jnp.arange(seq_len, dtype=jnp.int32)[None, :])
        window_lower_bounds = positions - (sliding_window - 1)
        lower_bounds = jnp.maximum(lower_bounds, window_lower_bounds)

    # The forward kernel uses the first query's lower bound to skip whole key
    # tiles. If an M tile begins with left padding, a ``seq_len`` sentinel would
    # incorrectly skip every key tile for valid queries later in that M tile.
    # Give each invalid query the next valid query's bound instead. Trailing
    # padding keeps the ``seq_len`` sentinel, so fully padded tail tiles remain
    # cheap. The per-score predicate still uses ``valid`` as the authority.
    next_valid_lower_bound = jax.lax.associative_scan(
        jnp.minimum,
        jnp.where(valid, lower_bounds, seq_len),
        axis=1,
        reverse=True,
    )
    return jnp.where(valid, lower_bounds, next_valid_lower_bound), valid


def _simple_causal_lower_bounds(
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int | None,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    positions = _replicate_metadata(jnp.arange(seq_len, dtype=jnp.int32)[None, :])
    if sliding_window is None:
        lower_bounds = _replicate_metadata(jnp.zeros((1, seq_len), dtype=jnp.int32))
    else:
        lower_bounds = jnp.maximum(positions - (sliding_window - 1), 0)
    lower_bounds = _replicate_metadata(jnp.broadcast_to(lower_bounds, (batch_size, seq_len)))
    valid = _replicate_metadata(jnp.ones((batch_size, seq_len), dtype=jnp.bool_))
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


def _fa4_cute_attention_forward_sharded(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    context_layout: ContextAttentionLayout,
) -> jax.Array:
    # Check global lengths before shard_map replaces Q with one rank's query slice.
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"FA4/CuTe self-attention requires q_len == k_len globally, got q={q.shape}, k={k.shape}")
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
            q_offset=jnp.zeros((1,), dtype=jnp.int32),
        )

    q_dims = partitioned_dims(q, mesh)
    if q_dims[3]:
        raise ValueError(f"FA4/CuTe requires an unsharded q feature dimension, got {partition_spec_of(q)}.")
    for name, x in (("k", k), ("v", v)):
        kv_dims = partitioned_dims(x, mesh)
        if kv_dims != (q_dims[0], (), q_dims[2], ()):
            raise ValueError(
                f"FA4/CuTe requires {name} to match q's batch/head sharding with unsharded sequence/feature "
                f"dimensions, got q={partition_spec_of(q)}, {name}={partition_spec_of(x)}."
            )
    if q_dims[1] and kernel_config.sm90_backward is not None:
        # Hopper's native segmented backward assumes local Q and K have equal lengths.
        # The 64x64 segmented path handles the query offset used by context parallelism.
        kernel_config = replace(kernel_config, sm90_backward=None)
    if not any(q_dims):
        return fa4_cute_attention_forward(
            q,
            k,
            v,
            lower_bounds,
            valid,
            sm_scale=sm_scale,
            kernel_config=kernel_config,
            q_offset=jnp.zeros((1,), dtype=jnp.int32),
        )

    sequence_axes = q_dims[1]
    shards = math.prod(mesh.shape[axis] for axis in sequence_axes)
    # Return Q's own spec, including length-1 axes: explicit sharding compares axis names, so an
    # output that drops them cannot be combined with the caller's activations.
    q_spec = tuple(partition_spec_of(q))
    output_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))
    # Bounds use global key positions but are sliced alongside their query rows.
    metadata_spec = P(*output_spec[:2])
    lower_bounds = reshard(lower_bounds, metadata_spec)
    valid = reshard(valid, metadata_spec)

    @shard_map(
        mesh=mesh,
        out_specs=output_spec,
        check_vma=False,
    )
    def _local_fa4_attention(q_local, k_local, v_local, lower_bounds_local, valid_local):
        def _run(q_rows, lower_bounds_rows, valid_rows, first_position):
            return fa4_cute_attention_forward(
                q_rows,
                k_local,
                v_local,
                lower_bounds_rows,
                valid_rows,
                sm_scale=sm_scale,
                kernel_config=kernel_config,
                q_offset=jnp.reshape(first_position, (1,)).astype(jnp.int32),
            )

        if not sequence_axes:
            return _run(q_local, lower_bounds_local, valid_local, 0)
        shard_index = jax.lax.axis_index(sequence_axes).astype(jnp.int32)
        local_len = q_local.shape[1]
        if context_layout == "contiguous":
            return _run(q_local, lower_bounds_local, valid_local, shard_index * local_len)
        # shard_map's transpose sums partial dK/dV for replicated K/V. Each shard runs two kernels
        # below, and each casts its fp32 accumulators to the K/V dtype before the local and
        # cross-shard sums, so gradient rounding error grows with twice the context degree.
        #
        # Causal attention costs more for later queries, so the shard holding the end of the
        # sequence would set the pace. Shard c holds positions [cL, (c+1)L); it keeps the first half
        # of its queries and swaps the second half with the mirror shard N-1-c, whose second half is
        # as late as shard c's first half is early. Every shard then attends one early and one
        # matching late block, and the outputs swap back afterwards.
        # Every shard splits at the same row, so the swapped halves match in shape even for odd L.
        split = local_len - local_len // 2
        # ppermute takes its axes, and numbers shards, in mesh order; sequence position follows Q's spec order.
        permute_axes = tuple(sorted(sequence_axes, key=mesh.axis_names.index))

        def _permute_index(shard: int) -> int:
            coords = dict(zip(sequence_axes, np.unravel_index(shard, [mesh.shape[a] for a in sequence_axes])))
            return int(np.ravel_multi_index([coords[a] for a in permute_axes], [mesh.shape[a] for a in permute_axes]))

        mirror = [(_permute_index(shard), _permute_index(shards - 1 - shard)) for shard in range(shards)]

        def _swap_late(x):
            return jax.lax.ppermute(x[:, split:], permute_axes, mirror)

        early = _run(
            q_local[:, :split], lower_bounds_local[:, :split], valid_local[:, :split], shard_index * local_len
        )
        late = _run(
            _swap_late(q_local),
            _swap_late(lower_bounds_local),
            _swap_late(valid_local),
            (shards - 1 - shard_index) * local_len + split,
        )
        return jnp.concatenate([early, jax.lax.ppermute(late, permute_axes, mirror)], axis=1)

    return _local_fa4_attention(q, k, v, lower_bounds, valid)


def _segmented_kernel_config(head_dim: int) -> Flash4CuteKernelConfig:
    arch = gpu_compute_capability()
    kernel_config = flash4_cute_kernel_config(head_dim, arch=arch)

    # Upstream flash-attn-4 4.0.0b15 dense SM100 FA4 uses 128x128 tiles in
    # flash_attn/cute/interface.py. This Grug port is not that native SM100
    # kernel: it carries dynamic lower-bound metadata through the SM80/SM120
    # segmented fork. On B200 d5120 Grug shapes, 64x64 fwd/bwd is consistently
    # faster than both the prior 128x64/64x64 config and dense-upstream 128x128.
    if arch // 10 == 10 and head_dim == 128:
        return replace(kernel_config, forward_tile=(64, 64), backward_tile=(64, 64), num_threads=128)
    return kernel_config


def _gpu_fa4_cute_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    kernel_config: Flash4CuteKernelConfig,
    context_layout: ContextAttentionLayout,
) -> Float[Array, "B Q Hq D"]:
    if context_layout not in get_args(ContextAttentionLayout):
        raise ValueError(f"context_layout must be one of {get_args(ContextAttentionLayout)}, got {context_layout!r}")
    _validate_head_layout(q, k, backend_name="gpu_fa4_cute_attention")
    if isinstance(mask, AttentionMask) and mask.fa4_bounds is not None:
        # The caller precomputed and selected the per-token metadata outside any per-layer scan/cond
        # (see fa4_cute_segment_bounds); use it directly instead of rebuilding from the static mask.
        lower_bounds, valid = mask.fa4_bounds
    else:
        lower_bounds, valid = _self_attention_lower_bounds(
            q,
            k,
            mask,
            backend_name="gpu_fa4_cute_attention",
        )

    return _fa4_cute_attention_forward_sharded(
        q,
        k,
        v,
        lower_bounds,
        valid,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        kernel_config=kernel_config,
        context_layout=context_layout,
    )


def gpu_fa4_cute_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    context_layout: ContextAttentionLayout = "zigzag",
) -> Float[Array, "B Q Hq D"]:
    """Run causal self-attention through the segmented FA4/CuTe kernel."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_cute_attention requires the JAX GPU backend.")
    return _gpu_fa4_cute_attention(
        q, k, v, mask, kernel_config=_segmented_kernel_config(q.shape[-1]), context_layout=context_layout
    )


def gpu_fa4_cute_sm100_attention(
    q: Float[Array, "B Q Hq D"],
    k: Float[Array, "B K Hkv D"],
    v: Float[Array, "B K Hkv D"],
    mask: AttentionMask | Bool[Array, "B Q K"] | Float[Array, "B Q K"] | None,
    *,
    context_layout: ContextAttentionLayout = "zigzag",
) -> Float[Array, "B Q Hq D"]:
    """Run native SM100 forward and one-block backward for BF16 D128 GQA.

    Select with ``implementation="gpu_fa4_cute_sm100"``. Supports GQA ratios
    4, 6, and 8 and the same packed causal/window masks as ``gpu_fa4_cute``.
    Use ``gpu_fa4_cute`` on Hopper or for other supported head layouts.
    """
    if jax.default_backend() != "gpu":
        raise RuntimeError("gpu_fa4_cute_sm100_attention requires the JAX GPU backend.")
    arch = gpu_compute_capability()
    _validate_head_layout(q, k, backend_name="gpu_fa4_cute_sm100")
    if arch != 100 or q.dtype != jnp.bfloat16 or q.shape[-1] != SM100_HEAD_DIM or v.shape[-1] != SM100_HEAD_DIM:
        raise ValueError(f"gpu_fa4_cute_sm100 requires SM100 with BF16 and D == Dv == {SM100_HEAD_DIM}.")
    if q.shape[2] // k.shape[2] not in SM100_GQA_RATIOS:
        raise ValueError(f"gpu_fa4_cute_sm100 requires a GQA ratio in {SM100_GQA_RATIOS}.")
    config = sm100_flash4_cute_kernel_config()
    return _gpu_fa4_cute_attention(q, k, v, mask, kernel_config=config, context_layout=context_layout)


def fa4_cute_segment_bounds(
    mask: AttentionMask,
    *,
    batch_size: int,
    seq_len: int,
    sliding_window: int | None,
) -> tuple[Int[Array, "B S"], Bool[Array, "B S"]]:
    """Compute FA4/CuTe per-token ``(lower_bounds, valid)`` for ``mask`` at a given window.

    Exposed so callers can precompute the metadata once outside a ``lax.scan``/``lax.cond`` and
    select it per layer via :meth:`AttentionMask.with_fa4_bounds`. This lets a homogeneous layer
    scan pick a per-layer sliding window (a static field the scan body cannot vary) by selecting
    between precomputed bound arrays.
    """
    if not isinstance(mask, AttentionMask):
        raise NotImplementedError("fa4_cute_segment_bounds requires an AttentionMask.")
    if not mask.is_causal:
        raise NotImplementedError("fa4_cute_segment_bounds supports only causal self-attention.")
    if mask.segment_ids is None:
        return _simple_causal_lower_bounds(batch_size=batch_size, seq_len=seq_len, sliding_window=sliding_window)
    q_segment_ids, kv_segment_ids = mask.segment_ids
    # These bounds derive from the q ids alone. Self-attention commonly carries q and kv ids as
    # distinct-but-equal arrays, so assert value equality at runtime (matching the regular FA4 path
    # in ``_packed_self_attention_segment_ids``) rather than rejecting on object identity, which
    # would break normal packed training.
    same_segment_ids = q_segment_ids is kv_segment_ids
    q_segment_ids = _batched_segment_ids(q_segment_ids, batch_size=batch_size, seq_len=seq_len)
    if not same_segment_ids:
        kv_segment_ids = _batched_segment_ids(kv_segment_ids, batch_size=batch_size, seq_len=seq_len)
        q_segment_ids = eqx.error_if(
            q_segment_ids,
            jnp.any(q_segment_ids != kv_segment_ids),
            "fa4_cute_segment_bounds requires matching q/kv segment ids.",
        )
    return _packed_segment_causal_lower_bounds(
        q_segment_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        sliding_window=sliding_window,
    )


__all__ = [
    "fa4_cute_segment_bounds",
    "gpu_fa4_cute_attention",
    "gpu_fa4_cute_sm100_attention",
]
