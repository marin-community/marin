# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared expert-parallel routing helpers for Grug MoE."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.sonic import sonic_gather_sum_available, sonic_gather_sum_masked


def _combine_gather_sum_impl(
    send_output: Float[Array, "send H"],
    gather_indices: Int[Array, " assignments"],
    combine_weights: Float[Array, "Tlocal K"],
    output_dtype: DTypeLike,
) -> Float[Array, "Tlocal H"]:
    tokens_per_shard, topk = combine_weights.shape
    hidden_dim = send_output.shape[1]
    gather_indices = gather_indices.reshape(tokens_per_shard, topk)
    keep = (gather_indices >= 0) & (gather_indices < send_output.shape[0])

    if sonic_gather_sum_available():
        weights = jnp.where(keep, combine_weights, 0)
        return sonic_gather_sum_masked(send_output, gather_indices, weights, output_dtype=output_dtype)

    out = jnp.zeros((tokens_per_shard, hidden_dim), dtype=jnp.float32)

    def add_route(route_index, out):
        rows = send_output[jnp.minimum(gather_indices[:, route_index], send_output.shape[0] - 1)]
        rows = jnp.where(keep[:, route_index, None], rows, 0)
        weight = combine_weights[:, route_index, None].astype(jnp.float32)
        return out + rows.astype(jnp.float32) * weight

    return jax.lax.fori_loop(0, topk, add_route, out).astype(output_dtype)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _combine_gather_sum(
    send_output: Float[Array, "send H"],
    gather_indices: Int[Array, " assignments"],
    assignment_sources: Int[Array, " send"],
    combine_weights: Float[Array, "Tlocal K"],
    output_dtype: DTypeLike,
) -> Float[Array, "Tlocal H"]:
    return _combine_gather_sum_impl(send_output, gather_indices, combine_weights, output_dtype)


def _combine_gather_sum_fwd(send_output, gather_indices, assignment_sources, combine_weights, output_dtype):
    out = _combine_gather_sum_impl(send_output, gather_indices, combine_weights, output_dtype)
    return out, (send_output, gather_indices, assignment_sources, combine_weights)


def _combine_gather_sum_bwd(_output_dtype, residual, cotangent):
    send_output, gather_indices, assignment_sources, combine_weights = residual
    tokens_per_shard, topk = combine_weights.shape
    assignments_per_shard = tokens_per_shard * topk
    valid = assignment_sources < assignments_per_shard
    sources = jnp.minimum(assignment_sources, assignments_per_shard - 1)
    source_tokens = sources // topk
    source_weights = combine_weights.reshape(-1)[sources].astype(jnp.float32)
    d_send_output = cotangent[source_tokens].astype(jnp.float32) * source_weights[:, None]
    d_send_output = jnp.where(valid[:, None], d_send_output, 0).astype(send_output.dtype)

    gather_indices = gather_indices.reshape(tokens_per_shard, topk)
    keep = (gather_indices >= 0) & (gather_indices < send_output.shape[0])
    d_combine_weights = jnp.zeros_like(combine_weights)

    def set_route(route_index, d_combine_weights):
        rows = send_output[jnp.minimum(gather_indices[:, route_index], send_output.shape[0] - 1)]
        d_weight = jnp.sum(cotangent.astype(jnp.float32) * rows.astype(jnp.float32), axis=1)
        d_weight = jnp.where(keep[:, route_index], d_weight, 0).astype(combine_weights.dtype)
        return jax.lax.dynamic_update_slice_in_dim(d_combine_weights, d_weight[:, None], route_index, axis=1)

    d_combine_weights = jax.lax.fori_loop(0, topk, set_route, d_combine_weights)
    return d_send_output, None, None, d_combine_weights


_combine_gather_sum.defvjp(_combine_gather_sum_fwd, _combine_gather_sum_bwd)


def _sort_activations(inputs: Float[Array, "N *tail"], sort_indices: Int[Array, "N"]) -> Float[Array, "N *tail"]:
    if inputs.shape[0] != sort_indices.shape[0]:
        raise ValueError(f"Expected matching leading dims, got {inputs.shape[0]} and {sort_indices.shape[0]}")
    return _sort_activations_custom(inputs, sort_indices)


@jax.custom_vjp
def _sort_activations_custom(
    inputs: Float[Array, "N *tail"], sort_indices: Int[Array, "N"]
) -> Float[Array, "N *tail"]:
    return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(
    inputs: Float[Array, "N *tail"], sort_indices: Int[Array, "N"]
) -> tuple[Float[Array, "N *tail"], Int[Array, "N"]]:
    return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(
    residuals: Int[Array, "N"], grads: Float[Array, "N *tail"]
) -> tuple[Float[Array, "N *tail"], None]:
    sort_indices = residuals
    return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def _ranks_within_groups(
    group_ids: Int[Array, "N"],
    *,
    num_groups: int,
    valid: Bool[Array, "N"],
) -> Int[Array, "N"]:
    """Return each valid item's zero-based group rank; invalid ranks are unspecified."""
    sortable_groups = jnp.where(valid, group_ids, num_groups)
    safe_groups = jnp.where(valid, group_ids, 0)
    order = jnp.argsort(sortable_groups, stable=True)
    inverse_order = jnp.argsort(order)
    counts = jnp.bincount(sortable_groups, length=num_groups).astype(jnp.int32)
    starts = jnp.cumsum(counts) - counts
    sorted_ranks = jnp.arange(group_ids.shape[0], dtype=jnp.int32) - starts[safe_groups[order]]
    return sorted_ranks[inverse_order]


def _assignment_sources(
    linear_indices: Int[Array, "N"],
    *,
    send_size: int,
) -> Int[Array, "send"]:
    """Map each fixed send slot to its source assignment."""
    assignments = linear_indices.shape[0]
    return (
        jnp.full((send_size,), assignments, dtype=jnp.int32)
        .at[linear_indices]
        .set(jnp.arange(assignments, dtype=jnp.int32), mode="drop")
    )


def _token_sources(
    assignment_sources: Int[Array, "send"],
    *,
    assignments: int,
    topk: int,
    tokens: int,
) -> Int[Array, "send"]:
    """Map send slots to token rows and use the padding row for empty slots."""
    return jnp.where(
        assignment_sources < assignments,
        assignment_sources // topk,
        tokens,
    )


def _prefix_cap_counts(counts: Int[Array, "*batch E"], *, capacity: int | Int[Array, ""]) -> Int[Array, "*batch E"]:
    """Allocate capacity in order along the last axis, independently for each batch."""
    starts = jnp.cumsum(counts, axis=-1, dtype=jnp.int32) - counts
    return jnp.minimum(counts, jnp.maximum(jnp.asarray(capacity, dtype=jnp.int32) - starts, 0))


def _clip_receiver_group_sizes(
    global_group_sizes: Int[Array, "S E"],
    *,
    local_expert_size: int,
    receiver_capacity: int | Int[Array, ""],
) -> Int[Array, "S E"]:
    """Clip sender->expert group sizes so each receiver shard stays within capacity."""
    num_senders = int(global_group_sizes.shape[0])
    num_experts = int(global_group_sizes.shape[1])
    if num_experts % local_expert_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local_expert_size={local_expert_size}")
    num_receivers = num_experts // local_expert_size
    if num_receivers != num_senders:
        raise ValueError(f"sender/receiver shard mismatch: num_senders={num_senders}, num_receivers={num_receivers}")

    # Each receiver prioritizes earlier experts, then earlier senders within each expert.
    # Batch receivers so traced capacities do not produce a separate kernel chain for each one.
    receiver_counts = global_group_sizes.reshape(num_senders, num_receivers, local_expert_size)
    receiver_counts = receiver_counts.transpose(1, 2, 0).reshape(num_receivers, -1)
    accepted = _prefix_cap_counts(receiver_counts, capacity=receiver_capacity)
    return (
        accepted.reshape(num_receivers, local_expert_size, num_senders)
        .transpose(2, 0, 1)
        .reshape(global_group_sizes.shape)
    )


class ExpertA2aParams(NamedTuple):
    """Offset/size vectors for one direction of an expert-granular ``ragged_all_to_all``."""

    input_offsets: Int[Array, "U"]
    send_sizes: Int[Array, "U"]
    output_offsets: Int[Array, "U"]
    recv_sizes: Int[Array, "U"]


def _expert_granular_a2a_params(
    all_group_sizes: Int[Array, "S E"],
    clipped_group_sizes: Int[Array, "S E"],
    shard_id: Int[Array, ""],
    *,
    local_expert_size: int,
) -> tuple[ExpertA2aParams, ExpertA2aParams]:
    """Build dispatch and return ``ragged_all_to_all`` parameters at (peer, expert) granularity.

    One update per (destination shard, local expert). Sender reads each global-expert
    group at its *unclipped* offset with its *clipped* size, so accepted rows need no
    compaction: they are the prefix of each group. Receiver offsets pack arriving rows
    expert-major (sender-major within each expert), so the received buffer needs no local
    permute before the grouped MLP. The return direction is the exact mirror: it reads the
    expert-major receiver buffer and writes valid prefixes back to the sender's unclipped
    positions, leaving dropped rows at the output operand's values.
    """
    num_shards = all_group_sizes.shape[0]

    # [src, dest, e]: rows sender `src` contributes to `dest`'s local expert `e`.
    clipped = clipped_group_sizes.reshape(num_shards, num_shards, local_expert_size)

    # Sender side: unclipped group starts in this shard's expert-sorted buffer.
    unclipped_starts = jnp.cumsum(all_group_sizes, axis=1) - all_group_sizes
    my_send = clipped[shard_id]  # [dest, e]
    dispatch_input_offsets = unclipped_starts[shard_id].reshape(num_shards, local_expert_size)

    # Receiver side: expert-major segment starts on each destination, sender-major within.
    dest_totals = jnp.sum(clipped, axis=0)  # [dest, e]
    expert_starts = jnp.cumsum(dest_totals, axis=1) - dest_totals
    senders_before_me = (jnp.cumsum(clipped, axis=0) - clipped)[shard_id]  # [dest, e]
    dispatch_output_offsets = expert_starts + senders_before_me

    # What each source sends this shard, source-major -- also the return direction's sends.
    inbound = clipped[:, shard_id, :]  # [src, e]

    dispatch = ExpertA2aParams(
        dispatch_input_offsets.reshape(-1),
        my_send.reshape(-1),
        dispatch_output_offsets.reshape(-1),
        inbound.reshape(-1),
    )

    # Return: read this shard's expert-major receiver buffer, write back to each original
    # sender's unclipped sorted positions for the experts this shard owns.
    my_expert_starts = expert_starts[shard_id]  # [e]
    senders_before = jnp.cumsum(inbound, axis=0) - inbound  # [src, e]
    return_input_offsets = my_expert_starts[None, :] + senders_before
    my_global_experts = jnp.arange(local_expert_size, dtype=jnp.int32) + shard_id * local_expert_size
    return_output_offsets = unclipped_starts[:, my_global_experts]  # [src, e]

    ret = ExpertA2aParams(
        return_input_offsets.reshape(-1),
        inbound.reshape(-1),
        return_output_offsets.reshape(-1),
        my_send.reshape(-1),
    )
    return dispatch, ret
