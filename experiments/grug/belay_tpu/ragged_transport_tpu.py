# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coalesce expert-granular transfers without changing accepted assignments."""

import jax
import jax.numpy as jnp


def gather_rows(values, indices):
    """Copy BF16 pairs as uint32 to avoid TPU subword gather expansion."""
    rows, width = values.shape
    if values.dtype == jnp.bfloat16 and width % 2 == 0:
        packed = jax.lax.bitcast_convert_type(values.reshape(rows, width // 2, 2), jnp.uint32)
        selected = jnp.take(packed, indices, axis=0, mode="clip")
        return jax.lax.bitcast_convert_type(selected, jnp.bfloat16).reshape(indices.size, width)
    return jnp.take(values, indices, axis=0, mode="clip")


def compact_segments(values, offsets, sizes):
    """Gather disjoint segments in their supplied order into a contiguous prefix."""
    ends = jnp.cumsum(sizes)
    starts = ends - sizes
    rows = jnp.arange(values.shape[0], dtype=jnp.int32)
    groups = jnp.minimum(jnp.searchsorted(ends, rows, side="right"), sizes.size - 1)
    selected = gather_rows(values, offsets[groups] + rows - starts[groups])
    return jnp.where((rows < ends[-1])[:, None], selected, 0)


def restore_segments(packed, initial, offsets, sizes):
    """Restore disjoint segments, preserving the initial buffer outside their union."""
    packed_starts = jnp.cumsum(sizes) - sizes
    keys = jnp.where(sizes > 0, offsets, jnp.iinfo(jnp.int32).max)
    order = jnp.argsort(keys)
    ends = jnp.where(sizes > 0, offsets + sizes, jnp.iinfo(jnp.int32).max)[order]
    rows = jnp.arange(initial.shape[0], dtype=jnp.int32)
    groups = order[jnp.minimum(jnp.searchsorted(ends, rows, side="right"), sizes.size - 1)]
    within = rows - offsets[groups]
    valid = (within >= 0) & (within < sizes[groups])
    selected = gather_rows(packed, packed_starts[groups] + within)
    return jnp.where(valid[:, None], selected, initial)


def coalesced_ragged_all_to_all(
    values, initial, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name: str, peers: int
):
    """Transport the same disjoint segments with one contiguous transfer per peer.

    Offsets and sizes use the native ragged-all-to-all peer-major order. Input
    and output segments must be disjoint, and their total sizes must fit their
    static buffers. One small metadata all-to-all supplies incoming destination
    offsets and the remote packed-buffer offsets. No routing capacity changes.
    """
    local_experts = send_sizes.size // peers
    peer_sends = send_sizes.reshape(peers, local_experts).sum(axis=1)
    peer_receives = recv_sizes.reshape(peers, local_experts).sum(axis=1)
    send_starts = jnp.cumsum(peer_sends) - peer_sends
    receive_starts = jnp.cumsum(peer_receives) - peer_receives
    metadata = jnp.concatenate((output_offsets.reshape(peers, local_experts), receive_starts[:, None]), axis=1)
    incoming = jax.lax.all_to_all(metadata, axis_name, split_axis=0, concat_axis=0, tiled=True)
    packed = compact_segments(values, input_offsets, send_sizes)
    received = jax.lax.ragged_all_to_all(
        packed,
        jnp.zeros_like(initial),
        send_starts,
        peer_sends,
        incoming[:, -1],
        peer_receives,
        axis_name=axis_name,
    )
    return restore_segments(received, initial, incoming[:, :-1].reshape(-1), recv_sizes)
