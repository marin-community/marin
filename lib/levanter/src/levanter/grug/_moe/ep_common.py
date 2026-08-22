# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared expert-parallel routing helpers for Grug MoE."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from levanter.grug._moe.sonic import sonic_gather_sum, sonic_gather_sum_available


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


def _use_sonic_gather_sum() -> bool:
    return sonic_gather_sum_available() and jax.default_backend() == "gpu"


def _unpermute_from_global_expert(
    intermediate: Float[Array, "TK H"],
    sorted_indices: Int[Array, "TK"],
    combine_weights_local: Float[Array, "Tlocal K"],
    *,
    tokens_per_shard: int,
    topk: int,
) -> Float[Array, "Tlocal H"]:
    """Weight each token's expert outputs by its routing weights and sum them."""
    positions = jnp.argsort(sorted_indices)
    if _use_sonic_gather_sum():
        # One kernel for the gather and the sum, materializing neither the unpermuted
        # ``[TK, H]`` buffer nor the ``[T, K, H]`` view -- at top-8 that view is eight times
        # the output. It accumulates in fp32 like the einsum below and keeps the routing
        # weight in fp32 through the multiply, where the einsum has to cast it down to avoid
        # promoting the larger operand, so the two agree to a single rounding.
        return sonic_gather_sum(intermediate, positions.reshape(tokens_per_shard, topk), combine_weights_local)
    unsorted = _sort_activations(intermediate, positions)
    reshaped = unsorted.reshape(tokens_per_shard, topk, -1)
    return jnp.einsum(
        "tkd,tk->td", reshaped, combine_weights_local.astype(reshaped.dtype), preferred_element_type=jnp.float32
    )


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


def _split_sizes(counts: Int[Array, "..."], splits: int) -> Int[Array, "... k"]:
    """Divide each count into ``splits`` near-equal parts, remainder to the earliest."""
    split_indices = jnp.arange(splits, dtype=counts.dtype)
    return counts[..., None] // splits + (split_indices < counts[..., None] % splits)


def _split_starts(split_sizes: Int[Array, "... k"]) -> Int[Array, "... k"]:
    return jnp.cumsum(split_sizes, axis=-1) - split_sizes


def _expert_granular_a2a_params(
    all_group_sizes: Int[Array, "S E"],
    clipped_group_sizes: Int[Array, "S E"],
    shard_id: Int[Array, ""],
    *,
    local_expert_size: int,
    splits_per_group: int,
) -> tuple[ExpertA2aParams, ExpertA2aParams]:
    """Build dispatch and return ``ragged_all_to_all`` parameters at (peer, expert) granularity.

    One update per (destination shard, local expert, split). Sender reads each global-expert
    group at its *unclipped* offset with its *clipped* size, so accepted rows need no
    compaction: they are the prefix of each group. Receiver offsets pack arriving rows
    expert-major (sender-major within each expert), so the received buffer needs no local
    permute before the grouped MLP. The return direction is the exact mirror: it reads the
    expert-major receiver buffer and writes valid prefixes back to the sender's unclipped
    positions, leaving dropped rows at the output operand's values.
    """
    if splits_per_group <= 0:
        raise ValueError(f"splits_per_group must be positive, got {splits_per_group}")
    num_shards = all_group_sizes.shape[0]

    # [src, dest, e]: rows sender `src` contributes to `dest`'s local expert `e`.
    clipped = clipped_group_sizes.reshape(num_shards, num_shards, local_expert_size)

    # Sender side: unclipped group starts in this shard's expert-sorted buffer.
    unclipped_starts = jnp.cumsum(all_group_sizes, axis=1) - all_group_sizes
    my_send = clipped[shard_id]  # [dest, e]
    my_send_splits = _split_sizes(my_send, splits_per_group)
    dispatch_input_offsets = unclipped_starts[shard_id].reshape(num_shards, local_expert_size)[
        ..., None
    ] + _split_starts(my_send_splits)

    # Receiver side: expert-major segment starts on each destination, sender-major within.
    dest_totals = jnp.sum(clipped, axis=0)  # [dest, e]
    expert_starts = jnp.cumsum(dest_totals, axis=1) - dest_totals
    senders_before_me = (jnp.cumsum(clipped, axis=0) - clipped)[shard_id]  # [dest, e]
    dispatch_output_offsets = (expert_starts + senders_before_me)[..., None] + _split_starts(my_send_splits)

    # What each source sends this shard, source-major -- also the return direction's sends.
    inbound = clipped[:, shard_id, :]  # [src, e]
    inbound_splits = _split_sizes(inbound, splits_per_group)

    dispatch = ExpertA2aParams(
        dispatch_input_offsets.reshape(-1),
        my_send_splits.reshape(-1),
        dispatch_output_offsets.reshape(-1),
        inbound_splits.reshape(-1),
    )

    # Return: read this shard's expert-major receiver buffer, write back to each original
    # sender's unclipped sorted positions for the experts this shard owns.
    my_expert_starts = expert_starts[shard_id]  # [e]
    senders_before = jnp.cumsum(inbound, axis=0) - inbound  # [src, e]
    return_input_offsets = (my_expert_starts[None, :] + senders_before)[..., None] + _split_starts(inbound_splits)
    my_global_experts = jnp.arange(local_expert_size, dtype=jnp.int32) + shard_id * local_expert_size
    sender_unclipped_starts = unclipped_starts[:, my_global_experts]  # [src, e]
    return_output_offsets = sender_unclipped_starts[..., None] + _split_starts(inbound_splits)

    ret = ExpertA2aParams(
        return_input_offsets.reshape(-1),
        inbound_splits.reshape(-1),
        return_output_offsets.reshape(-1),
        my_send_splits.reshape(-1),
    )
    return dispatch, ret


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _gather_dispatch_rows(x_local, sorted_indices, topk):
    """Build the expert-sorted dispatch buffer with one gather.

    Equivalent to ``jnp.repeat(x_local, topk, axis=0)[sorted_indices]`` without
    materializing the repeated buffer or running a data-sized permute. The backward pass
    is the transpose: each token sums the cotangent rows of its ``topk`` sorted slots.
    """
    return x_local[sorted_indices // topk]


def _gather_dispatch_rows_fwd(x_local, sorted_indices, topk):
    return _gather_dispatch_rows(x_local, sorted_indices, topk), sorted_indices


def _gather_dispatch_rows_bwd(topk, sorted_indices, cotangent):
    tokens_per_shard = sorted_indices.shape[0] // topk
    positions = jnp.argsort(sorted_indices).reshape(tokens_per_shard, topk)
    if _use_sonic_gather_sum():
        ones = jnp.ones((tokens_per_shard, topk), dtype=jnp.float32)
        grad_x = sonic_gather_sum(cotangent, positions, ones)
    else:
        grad_x = jnp.sum(cotangent[positions], axis=1, dtype=jnp.float32)
    return grad_x.astype(cotangent.dtype), None


_gather_dispatch_rows.defvjp(_gather_dispatch_rows_fwd, _gather_dispatch_rows_bwd)
