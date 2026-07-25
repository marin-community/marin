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


def _keep_mask_compaction_indices(
    keep_mask: Bool[Array, "N"],
    *,
    output_size: int,
) -> tuple[Int[Array, "M"], Int[Array, "N"]]:
    """Build the two directions of a stable, bounded mask compaction."""
    input_size = keep_mask.shape[0]
    input_positions = jnp.arange(input_size, dtype=jnp.int32)
    compact_positions = jnp.cumsum(keep_mask.astype(jnp.int32), dtype=jnp.int32) - 1
    keep_within_output = jnp.logical_and(keep_mask, compact_positions < output_size)
    input_to_output = jnp.where(keep_within_output, compact_positions, output_size)
    output_to_input = (
        jnp.full((output_size,), input_size, dtype=jnp.int32).at[input_to_output].set(input_positions, mode="drop")
    )
    return output_to_input, input_to_output


def _one_to_one_gather_impl(
    inputs: Float[Array, "N *tail"],
    source_indices: Int[Array, "M"],
) -> Float[Array, "M *tail"]:
    padded_inputs = jnp.concatenate(
        [inputs, jnp.zeros((1, *inputs.shape[1:]), dtype=inputs.dtype)],
        axis=0,
    )
    return padded_inputs[source_indices]


@jax.custom_vjp
def _one_to_one_gather(
    inputs: Float[Array, "N *tail"],
    source_indices: Int[Array, "M"],
    reverse_indices: Int[Array, "N"],
) -> Float[Array, "M *tail"]:
    """Gather a partial bijection whose adjoint is the reverse gather."""
    return _one_to_one_gather_impl(inputs, source_indices)


def _one_to_one_gather_fwd(
    inputs: Float[Array, "N *tail"],
    source_indices: Int[Array, "M"],
    reverse_indices: Int[Array, "N"],
) -> tuple[Float[Array, "M *tail"], tuple[Int[Array, "M"], Int[Array, "N"]]]:
    return _one_to_one_gather_impl(inputs, source_indices), (source_indices, reverse_indices)


def _one_to_one_gather_bwd(
    residuals: tuple[Int[Array, "M"], Int[Array, "N"]],
    output_cotangent: Float[Array, "M *tail"],
) -> tuple[Float[Array, "N *tail"], None, None]:
    source_indices, reverse_indices = residuals
    del source_indices
    return _one_to_one_gather_impl(output_cotangent, reverse_indices), None, None


_one_to_one_gather.defvjp(_one_to_one_gather_fwd, _one_to_one_gather_bwd)


def _compact_by_keep_mask_to_size(
    inputs: Float[Array, "N *tail"],
    keep_mask: Bool[Array, "N"],
    *,
    output_size: int,
) -> Float[Array, "M *tail"]:
    output_to_input, input_to_output = _keep_mask_compaction_indices(keep_mask, output_size=output_size)
    return _one_to_one_gather(inputs, output_to_input, input_to_output)


def _compact_by_keep_mask(inputs: Float[Array, "N *tail"], keep_mask: Bool[Array, "N"]) -> Float[Array, "N *tail"]:
    return _compact_by_keep_mask_to_size(inputs, keep_mask, output_size=inputs.shape[0])


def _expand_from_keep_mask(compacted: Float[Array, "N *tail"], keep_mask: Bool[Array, "N"]) -> Float[Array, "N *tail"]:
    compact_to_output, output_to_compact = _keep_mask_compaction_indices(
        keep_mask,
        output_size=compacted.shape[0],
    )
    return _one_to_one_gather(compacted, output_to_compact, compact_to_output)
