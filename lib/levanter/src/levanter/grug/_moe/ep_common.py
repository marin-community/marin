# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared expert-parallel routing helpers for Grug MoE."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


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


def _ranks_within_groups(group_ids: Int[Array, "N"], *, num_groups: int) -> Int[Array, "N"]:
    """Return the zero-based rank of each item in its group."""
    order = jnp.argsort(group_ids, stable=True)
    inverse_order = jnp.argsort(order)
    counts = jnp.bincount(group_ids, length=num_groups).astype(jnp.int32)
    starts = jnp.cumsum(counts) - counts
    sorted_ranks = jnp.arange(group_ids.shape[0], dtype=jnp.int32) - starts[group_ids[order]]
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


def _prefix_cap_counts(counts: Int[Array, "E"], *, capacity: int) -> Int[Array, "E"]:
    accepted = []
    remaining = jnp.array(capacity, dtype=jnp.int32)
    for expert in range(int(counts.shape[0])):
        take = jnp.minimum(counts[expert], remaining)
        accepted.append(take)
        remaining = jnp.maximum(remaining - take, 0)
    return jnp.stack(accepted, axis=0)


def _clip_receiver_group_sizes(
    global_group_sizes: Int[Array, "S E"],
    *,
    local_expert_size: int,
    receiver_capacity: int,
) -> Int[Array, "S E"]:
    """Clip sender->expert group sizes so each receiver shard stays within capacity."""
    num_senders = int(global_group_sizes.shape[0])
    num_experts = int(global_group_sizes.shape[1])
    if num_experts % local_expert_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local_expert_size={local_expert_size}")
    num_receivers = num_experts // local_expert_size
    if num_receivers != num_senders:
        raise ValueError(f"sender/receiver shard mismatch: num_senders={num_senders}, num_receivers={num_receivers}")

    clipped_by_receiver: list[jax.Array] = []
    for receiver_index in range(num_receivers):
        start = receiver_index * local_expert_size
        stop = start + local_expert_size
        receiver_counts = global_group_sizes[:, start:stop]
        receiver_totals = jnp.sum(receiver_counts, axis=0, dtype=jnp.int32)
        accepted_totals = _prefix_cap_counts(receiver_totals, capacity=receiver_capacity)
        remaining = accepted_totals
        accepted_rows: list[jax.Array] = []
        for sender_index in range(num_senders):
            # Greedy first-sender-wins: earlier shards get priority when capacity is scarce.
            accepted = jnp.minimum(receiver_counts[sender_index], remaining)
            accepted_rows.append(accepted)
            remaining = remaining - accepted
        clipped_by_receiver.append(jnp.stack(accepted_rows, axis=0))

    return jnp.concatenate(clipped_by_receiver, axis=1)


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
