# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Assigned-token expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    MOE_REMAT_SAVE_NAMES,
    MoERematMode,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
)


class AssignedTokenDispatch(NamedTuple):
    """Receiver-local routed assignments grouped for expert GMM."""

    x_dispatch: jax.Array
    assignment_weights: jax.Array
    local_sorted_indices: jax.Array
    local_group_sizes: jax.Array
    sender_sorted_indices: jax.Array
    sender_keep_mask: jax.Array
    all_shard_assignment_counts: jax.Array
    dropped_local: jax.Array


def _unpermute_weighted_from_global_expert(
    weighted_assignments: jax.Array,
    sorted_indices: jax.Array,
    *,
    tokens_per_shard: int,
    topk: int,
) -> jax.Array:
    unsorted = _sort_activations(weighted_assignments, jnp.argsort(sorted_indices))
    return jnp.einsum(
        "tkd->td",
        unsorted.reshape(tokens_per_shard, topk, -1),
        preferred_element_type=jnp.float32,
    )


def _assigned_token_dispatch(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    *,
    num_experts: int,
    local_experts: int,
    capacity_factor: float,
) -> AssignedTokenDispatch:
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    local_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    local_capacity = max(local_experts, local_capacity)
    recv_capacity = local_capacity

    sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
        x_local,
        selected_experts_local,
        num_experts=num_experts,
    )
    sorted_weights = _sort_activations(
        combine_weights_local.reshape(-1)[:, None].astype(x_local.dtype), sorted_indices
    )
    all_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
    clipped_group_sizes = _clip_receiver_group_sizes(
        all_group_sizes,
        local_expert_size=local_experts,
        receiver_capacity=local_capacity,
    )
    sender_group_sizes = clipped_group_sizes[shard_id]
    keep_mask = _expert_prefix_keep_mask(
        group_sizes.astype(jnp.int32),
        sender_group_sizes,
        total_size=assignments_per_shard,
    )
    sorted_x = _compact_by_keep_mask(sorted_x, keep_mask)
    sorted_weights = _compact_by_keep_mask(sorted_weights, keep_mask)

    all_shard_counts = jnp.sum(clipped_group_sizes.reshape(ep_size, ep_size, local_experts), axis=2)
    input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
    dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
    x_dispatched = jax.lax.ragged_all_to_all(
        sorted_x,
        dispatch_out_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )
    weight_out_shape = jnp.zeros((recv_capacity, 1), dtype=x_local.dtype)
    weights_dispatched = jax.lax.ragged_all_to_all(
        sorted_weights,
        weight_out_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )
    x_dispatch, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
        x_dispatched,
        clipped_group_sizes,
        local_expert_size=local_experts,
        shard_index=shard_id,
    )
    assignment_weights = _sort_activations(weights_dispatched, local_sorted_indices).reshape(-1)
    recv_valid = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.sum(recv_sizes, dtype=jnp.int32)
    assignment_weights = jnp.where(recv_valid, assignment_weights, 0)
    dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
    return AssignedTokenDispatch(
        x_dispatch=tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT),
        assignment_weights=assignment_weights,
        local_sorted_indices=local_sorted_indices,
        local_group_sizes=local_group_sizes,
        sender_sorted_indices=sorted_indices,
        sender_keep_mask=keep_mask,
        all_shard_assignment_counts=all_shard_counts,
        dropped_local=dropped_local,
    )


def _moe_mlp_ep_assigned_token_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: MoERematMode = "none",
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    shard_id = jax.lax.axis_index("expert")
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk

    with jax.named_scope("dispatch"):
        dispatch = _assigned_token_dispatch(
            x_local,
            selected_experts_local,
            combine_weights_local,
            num_experts=num_experts,
            local_experts=local_experts,
            capacity_factor=capacity_factor,
        )

    def moe_up_down(
        x_dispatch: jax.Array,
        local_group_sizes: jax.Array,
        moe_w13: jax.Array,
        moe_w2: jax.Array,
    ) -> jax.Array:
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13, local_group_sizes),
            _CHECKPOINT_EXPERT_HIDDEN,
        )
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        return tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, local_group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("moe_up_down"):
        if remat_mode == "none":
            out_dispatch = moe_up_down(
                dispatch.x_dispatch,
                dispatch.local_group_sizes,
                moe_w13_local,
                moe_w2_local,
            )
        else:
            policy = None
            if remat_mode == "save_moe":
                policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
            out_dispatch = jax.checkpoint(moe_up_down, policy=policy)(
                dispatch.x_dispatch,
                dispatch.local_group_sizes,
                moe_w13_local,
                moe_w2_local,
            )

    with jax.named_scope("combine"):
        weighted_dispatch = out_dispatch * dispatch.assignment_weights[:, None]
        local_output = _sort_activations(weighted_dispatch, jnp.argsort(dispatch.local_sorted_indices))
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=local_output.dtype)
        return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
            dispatch.all_shard_assignment_counts.T, shard_id
        )
        returned = jax.lax.ragged_all_to_all(
            local_output,
            return_out_shape,
            return_input_offsets,
            return_send_sizes,
            return_output_offsets,
            return_recv_sizes,
            axis_name="expert",
        )
        returned = _expand_from_keep_mask(returned, dispatch.sender_keep_mask)
        out_local = _unpermute_weighted_from_global_expert(
            returned,
            dispatch.sender_sorted_indices,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_total = jax.lax.psum(dispatch.dropped_local, ("data", "expert"))
    return out_local, dropped_total
