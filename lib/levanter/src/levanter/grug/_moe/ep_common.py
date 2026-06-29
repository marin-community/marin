# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared expert-parallel routing helpers for Grug MoE."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


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


def _prefix_cap_counts(counts: Int[Array, "E"], *, capacity: int) -> Int[Array, "E"]:
    accepted = []
    remaining = jnp.array(capacity, dtype=jnp.int32)
    for expert in range(int(counts.shape[0])):
        take = jnp.minimum(counts[expert], remaining)
        accepted.append(take)
        remaining = jnp.maximum(remaining - take, 0)
    return jnp.stack(accepted, axis=0)


def _permute_by_global_expert(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    num_experts: int,
) -> tuple[Float[Array, "TK H"], Int[Array, "TK"], Int[Array, "E"]]:
    topk = selected_experts_local.shape[1]
    flat_selected = selected_experts_local.reshape(-1)
    sorted_indices = jnp.argsort(flat_selected)
    repeated_x = jnp.repeat(x_local, topk, axis=0)
    sorted_x = _sort_activations(repeated_x, sorted_indices)
    group_sizes = jnp.bincount(flat_selected, length=num_experts).astype(jnp.int32)
    return sorted_x, sorted_indices, group_sizes


def _unpermute_from_global_expert(
    intermediate: Float[Array, "TK H"],
    sorted_indices: Int[Array, "TK"],
    combine_weights_local: Float[Array, "Tlocal K"],
    *,
    tokens_per_shard: int,
    topk: int,
) -> Float[Array, "Tlocal H"]:
    unsorted = _sort_activations(intermediate, jnp.argsort(sorted_indices))
    reshaped = unsorted.reshape(tokens_per_shard, topk, -1)
    return jnp.einsum(
        "tkd,tk->td", reshaped, combine_weights_local.astype(reshaped.dtype), preferred_element_type=jnp.float32
    )


def _shard_a2a_params(
    shard_counts: Int[Array, "S S"],
    shard_id: Int[Array, ""],
) -> tuple[Int[Array, "S"], Int[Array, "S"], Int[Array, "S"], Int[Array, "S"]]:
    row = shard_counts[shard_id]
    input_offsets = jnp.cumsum(jnp.concatenate((jnp.array([0], dtype=row.dtype), row[:-1])))
    send_sizes = row

    recv_sizes = shard_counts[:, shard_id]
    # `ragged_all_to_all` expects sender-side output offsets: for each
    # destination shard, where this sender's slice should land in the remote
    # receiver buffer. JAX computes the local receive offsets by transposing
    # these offsets with an internal all_to_all.
    sender_output_offsets = jnp.cumsum(shard_counts, axis=0, dtype=shard_counts.dtype) - shard_counts
    output_offsets = sender_output_offsets[shard_id]
    return input_offsets, send_sizes, output_offsets, recv_sizes


def _local_permute_from_counts(
    inputs: Float[Array, "C H"],
    global_group_sizes: Int[Array, "S E"],
    *,
    local_expert_size: int,
    shard_index: Int[Array, ""],
) -> tuple[Float[Array, "C H"], Int[Array, "C"], Int[Array, "Elocal"]]:
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        start_index=shard_index * local_expert_size,
        slice_size=local_expert_size,
        axis=1,
    )
    local_group_sizes = jnp.sum(all_shard_local_sizes, axis=0)
    local_sizes = all_shard_local_sizes.reshape(-1)
    total_valid = jnp.sum(local_sizes, dtype=jnp.int32)
    segment_ends = jnp.cumsum(local_sizes, dtype=jnp.int32)
    positions = jnp.arange(inputs.shape[0], dtype=jnp.int32)
    segment_index = jnp.searchsorted(segment_ends, positions, side="right")
    local_expert_ids = jnp.where(positions < total_valid, segment_index % local_expert_size, local_expert_size)
    sorted_indices = jnp.argsort(local_expert_ids)
    sorted_inputs = _sort_activations(inputs, sorted_indices)
    sorted_inputs = jnp.where((positions < total_valid)[:, None], sorted_inputs, 0)
    group_sizes = local_group_sizes.at[-1].add(inputs.shape[0] - total_valid)
    return sorted_inputs, sorted_indices, group_sizes


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


def _expert_prefix_keep_mask(
    group_sizes: Int[Array, "E"],
    accepted_group_sizes: Int[Array, "E"],
    *,
    total_size: int,
) -> Bool[Array, "TK"]:
    segment_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    segment_starts = jnp.concatenate((jnp.array([0], dtype=segment_ends.dtype), segment_ends[:-1]))
    positions = jnp.arange(total_size, dtype=jnp.int32)
    expert_index = jnp.searchsorted(segment_ends, positions, side="right")
    # Explicitly clip overflow positions to the last segment rather than
    # depending on implicit out-of-bounds `jnp.take` behavior. Those clipped
    # positions will have local_rank >= accepted, so they are masked out.
    expert_index = jnp.minimum(expert_index, group_sizes.shape[0] - 1)
    local_rank = positions - segment_starts[expert_index]
    accepted = accepted_group_sizes[expert_index]
    return local_rank < accepted


def _compact_by_keep_mask(inputs: Float[Array, "N *tail"], keep_mask: Bool[Array, "N"]) -> Float[Array, "N *tail"]:
    total_size = inputs.shape[0]
    positions = jnp.arange(total_size, dtype=jnp.int32)
    sort_key = jnp.where(keep_mask, positions, positions + total_size)
    compacted = _sort_activations(inputs, jnp.argsort(sort_key))
    valid = positions < jnp.sum(keep_mask.astype(jnp.int32), dtype=jnp.int32)
    return jnp.where(valid[:, None], compacted, 0)


def _expand_from_keep_mask(compacted: Float[Array, "N *tail"], keep_mask: Bool[Array, "N"]) -> Float[Array, "N *tail"]:
    keep_i32 = keep_mask.astype(jnp.int32)
    compact_index = jnp.cumsum(keep_i32, dtype=jnp.int32) - 1
    gathered = jnp.take(compacted, jnp.maximum(compact_index, 0), axis=0)
    return jnp.where(keep_mask[:, None], gathered, 0)
