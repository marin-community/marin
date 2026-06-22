# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-bucket all-to-all expert-parallel Grug MoE backend.

This is an experimental alternative to `jax.lax.ragged_all_to_all`. It trades
some padding for fewer collective launches by packing each destination shard
into a fixed-size bucket, using `lax.all_to_all`, and carrying explicit bucket
indices through combine.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import ragged_dot

from levanter.grug._moe.common import split_moe_w13_output
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _expert_prefix_keep_mask,
    _permute_by_global_expert,
    _prefix_cap_counts,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug.sharding import _batch_axes


class _FixedBucketDispatch(NamedTuple):
    x_dispatch: jax.Array
    local_group_sizes: jax.Array
    local_source_offsets: jax.Array
    sender_bucket_offsets: jax.Array
    sorted_indices: jax.Array
    keep_mask: jax.Array
    sender_group_sizes: jax.Array
    dropped_local: jax.Array
    per_peer_capacity: int


def _destination_index_for_positions(
    group_sizes: jax.Array,
    *,
    local_experts: int,
    total_size: int,
) -> jax.Array:
    segment_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    positions = jnp.arange(total_size, dtype=jnp.int32)
    expert_index = jnp.searchsorted(segment_ends, positions, side="right")
    expert_index = jnp.minimum(expert_index, group_sizes.shape[0] - 1)
    return expert_index // local_experts


def _rank_within_destination(
    group_sizes: jax.Array,
    accepted_group_sizes: jax.Array,
    *,
    local_experts: int,
    total_size: int,
) -> jax.Array:
    segment_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    segment_starts = jnp.concatenate((jnp.array([0], dtype=segment_ends.dtype), segment_ends[:-1]))
    positions = jnp.arange(total_size, dtype=jnp.int32)
    expert_index = jnp.searchsorted(segment_ends, positions, side="right")
    expert_index = jnp.minimum(expert_index, group_sizes.shape[0] - 1)
    rank_in_expert = positions - segment_starts[expert_index]

    expert_starts = (expert_index // local_experts) * local_experts
    expert_ids = jnp.arange(group_sizes.shape[0], dtype=jnp.int32)
    before = expert_ids[None, :] < expert_index[:, None]
    same_destination = expert_ids[None, :] >= expert_starts[:, None]
    prior_in_destination = jnp.sum(
        jnp.where(before & same_destination, accepted_group_sizes[None, :], 0),
        axis=1,
        dtype=jnp.int32,
    )
    return prior_in_destination + rank_in_expert


def _pack_sender_buckets(
    sorted_values: jax.Array,
    group_sizes: jax.Array,
    accepted_group_sizes: jax.Array,
    *,
    local_experts: int,
    per_peer_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    total_size = sorted_values.shape[0]
    ep_size = group_sizes.shape[0] // local_experts
    keep_mask = _expert_prefix_keep_mask(
        group_sizes.astype(jnp.int32),
        accepted_group_sizes.astype(jnp.int32),
        total_size=total_size,
    )
    destination = _destination_index_for_positions(group_sizes, local_experts=local_experts, total_size=total_size)
    rank = _rank_within_destination(
        group_sizes,
        accepted_group_sizes,
        local_experts=local_experts,
        total_size=total_size,
    )
    bucket_offsets = destination * per_peer_capacity + rank
    flat_shape = (ep_size * per_peer_capacity, *sorted_values.shape[1:])
    flat = jnp.zeros(flat_shape, dtype=sorted_values.dtype)
    safe_offsets = jnp.where(keep_mask, bucket_offsets, 0)
    values = jnp.where(keep_mask.reshape((total_size,) + (1,) * (sorted_values.ndim - 1)), sorted_values, 0)
    flat = flat.at[safe_offsets].add(values)
    return flat.reshape((ep_size, per_peer_capacity, *sorted_values.shape[1:])), bucket_offsets, keep_mask


def _clip_sender_destination_group_sizes(
    clipped_group_sizes: jax.Array,
    *,
    local_experts: int,
    per_peer_capacity: int,
) -> jax.Array:
    ep_size = int(clipped_group_sizes.shape[0])
    clipped_rows: list[jax.Array] = []
    for sender_index in range(ep_size):
        row_blocks: list[jax.Array] = []
        for destination_index in range(ep_size):
            start = destination_index * local_experts
            stop = start + local_experts
            row_blocks.append(
                _prefix_cap_counts(clipped_group_sizes[sender_index, start:stop], capacity=per_peer_capacity)
            )
        clipped_rows.append(jnp.concatenate(row_blocks, axis=0))
    return jnp.stack(clipped_rows, axis=0)


def _fixed_all_to_all(values: jax.Array) -> jax.Array:
    return jax.lax.all_to_all(values, "expert", split_axis=0, concat_axis=0, tiled=True)


def _local_permute_fixed_buckets(
    received: jax.Array,
    clipped_group_sizes: jax.Array,
    *,
    local_experts: int,
    shard_index: jax.Array,
    per_peer_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    local_counts_by_sender = jax.lax.dynamic_slice_in_dim(
        clipped_group_sizes,
        start_index=shard_index * local_experts,
        slice_size=local_experts,
        axis=1,
    )
    local_group_sizes = jnp.sum(local_counts_by_sender, axis=0)
    ep_size = clipped_group_sizes.shape[0]
    positions = jnp.arange(ep_size * per_peer_capacity, dtype=jnp.int32)
    sender = positions // per_peer_capacity
    rank = positions % per_peer_capacity

    sender_counts = local_counts_by_sender[sender]
    segment_ends = jnp.cumsum(sender_counts, axis=1, dtype=jnp.int32)
    local_expert = jnp.sum(rank[:, None] >= segment_ends, axis=1, dtype=jnp.int32)
    valid = rank < jnp.sum(sender_counts, axis=1, dtype=jnp.int32)
    local_expert = jnp.where(valid, local_expert, local_experts)

    order = jnp.argsort(local_expert * (ep_size * per_peer_capacity) + positions)
    sorted_values = _sort_activations(received.reshape((ep_size * per_peer_capacity, *received.shape[2:])), order)
    total_valid = jnp.sum(local_group_sizes, dtype=jnp.int32)
    output_positions = jnp.arange(ep_size * per_peer_capacity, dtype=jnp.int32)
    sorted_values = jnp.where(
        (output_positions < total_valid).reshape((ep_size * per_peer_capacity,) + (1,) * (received.ndim - 2)),
        sorted_values,
        0,
    )
    local_source_offsets = _sort_activations(positions, order)
    group_sizes = local_group_sizes.at[-1].add(ep_size * per_peer_capacity - total_valid)
    return sorted_values, local_source_offsets, group_sizes


def _dispatch_fixed_buckets(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    *,
    num_experts: int,
    local_experts: int,
    capacity_factor: float,
) -> _FixedBucketDispatch:
    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    assignments_per_shard = x_local.shape[0] * topk
    receiver_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments_per_shard)))
    per_peer_capacity = max(local_experts, int(math.ceil(receiver_capacity / ep_size)))

    sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
        x_local,
        selected_experts_local,
        num_experts=num_experts,
    )
    all_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
    clipped_group_sizes = _clip_receiver_group_sizes(
        all_group_sizes,
        local_expert_size=local_experts,
        receiver_capacity=receiver_capacity,
    )
    clipped_group_sizes = _clip_sender_destination_group_sizes(
        clipped_group_sizes,
        local_experts=local_experts,
        per_peer_capacity=per_peer_capacity,
    )
    sender_group_sizes = clipped_group_sizes[shard_id]
    bucketed, sender_bucket_offsets, keep_mask = _pack_sender_buckets(
        sorted_x,
        group_sizes.astype(jnp.int32),
        sender_group_sizes,
        local_experts=local_experts,
        per_peer_capacity=per_peer_capacity,
    )
    received = _fixed_all_to_all(bucketed)
    x_dispatch, local_source_offsets, local_group_sizes = _local_permute_fixed_buckets(
        received,
        clipped_group_sizes,
        local_experts=local_experts,
        shard_index=shard_id,
        per_peer_capacity=per_peer_capacity,
    )
    dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
    return _FixedBucketDispatch(
        x_dispatch=x_dispatch,
        local_group_sizes=local_group_sizes,
        local_source_offsets=local_source_offsets,
        sender_bucket_offsets=sender_bucket_offsets,
        sorted_indices=sorted_indices,
        keep_mask=keep_mask,
        sender_group_sizes=sender_group_sizes,
        dropped_local=dropped_local,
        per_peer_capacity=per_peer_capacity,
    )


def _moe_mlp_ep_padded_a2a_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: str = "none",
) -> tuple[jax.Array, jax.Array]:
    del remat_mode
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    with jax.named_scope("moe_ep_padded_a2a/dispatch"):
        dispatch = _dispatch_fixed_buckets(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
            local_experts=local_experts,
            capacity_factor=capacity_factor,
        )

    with jax.named_scope("moe_expert_mlp/w13_ragged_dot"):
        w13_out = ragged_dot(dispatch.x_dispatch, moe_w13_local, dispatch.local_group_sizes)
    with jax.named_scope("moe_expert_mlp/split_gate_up"):
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
    with jax.named_scope("moe_expert_mlp/activation"):
        hidden = activation_fn(gate) * up
    with jax.named_scope("moe_expert_mlp/w2_ragged_dot"):
        out_dispatch = ragged_dot(hidden, moe_w2_local, dispatch.local_group_sizes)

    with jax.named_scope("moe_ep_padded_a2a/combine_pack"):
        ep_size = num_experts // local_experts
        flat_shape = (ep_size * dispatch.per_peer_capacity, out_dispatch.shape[1])
        send_flat = jnp.zeros(flat_shape, dtype=out_dispatch.dtype)
        send_flat = send_flat.at[dispatch.local_source_offsets].set(out_dispatch)
        send_buckets = send_flat.reshape((ep_size, dispatch.per_peer_capacity, out_dispatch.shape[1]))
    with jax.named_scope("moe_ep_padded_a2a/combine_transport"):
        returned_buckets = _fixed_all_to_all(send_buckets)
    with jax.named_scope("moe_ep_padded_a2a/combine_expand_and_weight"):
        returned_flat = returned_buckets.reshape((ep_size * dispatch.per_peer_capacity, out_dispatch.shape[1]))
        safe_offsets = jnp.where(dispatch.keep_mask, dispatch.sender_bucket_offsets, 0)
        returned_sorted = jnp.take(returned_flat, safe_offsets, axis=0)
        returned_sorted = jnp.where(dispatch.keep_mask[:, None], returned_sorted, 0)
        out_local = _unpermute_from_global_expert(
            returned_sorted,
            dispatch.sorted_indices,
            combine_weights_local,
            tokens_per_shard=x_local.shape[0],
            topk=selected_experts_local.shape[1],
        ).astype(x_local.dtype)
    with jax.named_scope("moe_ep_padded_a2a/dropped_assignments"):
        dropped_total = jax.lax.psum(dispatch.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
