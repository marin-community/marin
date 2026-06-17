# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP intranode expert-parallel Grug MoE backend.

DeepEP source: https://github.com/deepseek-ai/DeepEP
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS,
    _CHECKPOINT_DEEPEP_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK,
    _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES,
    _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS,
    _CHECKPOINT_DEEPEP_RANK_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_SRC_IDX,
    _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES,
    _CHECKPOINT_DEEPEP_RECV_TOPK_IDX,
    _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS,
    _CHECKPOINT_DEEPEP_RECV_X,
    _CHECKPOINT_DEEPEP_SEND_HEAD,
    _CHECKPOINT_EXPERT_HIDDEN,
    MOE_REMAT_SAVE_NAMES,
    MoERematMode,
    split_moe_w13_output,
)
from levanter.kernels.deepep import deepep_combine_intranode, deepep_dispatch_intranode, deepep_get_dispatch_layout


class DeepEPLocalAssignments(NamedTuple):
    """Local expert assignment batch after DeepEP token dispatch.

    Attributes:
        x_dispatch: Received token activations repeated once per local expert assignment.
        assignment_weights: Combine weights aligned with `x_dispatch`.
        recv_token_indices: Receive-buffer token row for each local assignment.
        local_group_sizes: Assignment counts per local expert for `ragged_dot`.
    """

    x_dispatch: Float[Array, "TK D"]
    assignment_weights: Float[Array, "TK"]
    recv_token_indices: Int[Array, "TK"]
    local_group_sizes: Int[Array, "EL"]


def _deepep_moe_up_down(
    x_dispatch: Float[Array, "TK D"],
    local_group_sizes: Int[Array, "EL"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "TK D"]:
    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, local_group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        return tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )


def _deepep_moe_up_down_remat(
    x_dispatch: Float[Array, "TK D"],
    local_group_sizes: Int[Array, "EL"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    remat_mode: MoERematMode,
) -> Float[Array, "TK D"]:
    if remat_mode == "none":
        return _deepep_moe_up_down(
            x_dispatch,
            local_group_sizes,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
        )

    policy = None
    if remat_mode == "save_moe":
        policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)

    def compute(
        remat_x_dispatch: jax.Array,
        remat_local_group_sizes: jax.Array,
        remat_moe_w13_local: jax.Array,
        remat_moe_w2_local: jax.Array,
    ) -> jax.Array:
        return _deepep_moe_up_down(
            remat_x_dispatch,
            remat_local_group_sizes,
            remat_moe_w13_local,
            remat_moe_w2_local,
            activation_fn=activation_fn,
        )

    return jax.checkpoint(compute, policy=policy)(x_dispatch, local_group_sizes, moe_w13_local, moe_w2_local)


def _pack_deepep_local_assignments(
    recv_x: Float[Array, "TR D"],
    recv_topk_idx: Int[Array, "TR K"],
    recv_topk_weights: Float[Array, "TR K"],
    *,
    local_experts: int,
    num_recv_tokens: Int[Array, ""],
) -> DeepEPLocalAssignments:
    with jax.named_scope("deepep_pack_local_assignments"):
        max_recv_tokens, topk = recv_topk_idx.shape
        total_assignments = max_recv_tokens * topk

        recv_token_indices = jnp.repeat(jnp.arange(max_recv_tokens, dtype=jnp.int32), topk)
        expert_flat = recv_topk_idx.reshape(-1).astype(jnp.int32)
        recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < num_recv_tokens
        local_mask = recv_valid[:, None] & (recv_topk_idx >= 0) & (recv_topk_idx < local_experts)
        local_mask_flat = local_mask.reshape(-1)
        local_bucket = jnp.where(local_mask_flat, expert_flat, local_experts)
        local_group_sizes = jnp.bincount(local_bucket, length=local_experts + 1).astype(jnp.int32)[:-1]
        total_valid = jnp.sum(local_group_sizes, dtype=jnp.int32)

        flat_positions = jnp.arange(total_assignments, dtype=jnp.int32)
        order_key = local_bucket * total_assignments + flat_positions
        max_order_key = (local_experts + 1) * total_assignments
        selection_key = jnp.where(local_mask_flat, max_order_key - order_key, -1)
        _, sorted_assignment_indices = jax.lax.top_k(selection_key, total_assignments)

        recv_token_indices = jnp.take(recv_token_indices, sorted_assignment_indices, axis=0)
        x_dispatch = jnp.take(recv_x, recv_token_indices, axis=0)
        assignment_weights = jnp.take(recv_topk_weights.reshape(-1), sorted_assignment_indices, axis=0).astype(
            recv_x.dtype
        )
        valid_sorted = jnp.arange(total_assignments, dtype=jnp.int32) < total_valid
        x_dispatch = jnp.where(valid_sorted[:, None], x_dispatch, 0)
        assignment_weights = jnp.where(valid_sorted, assignment_weights, 0)
        assignment_weights = tree_checkpoint_name(assignment_weights, _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS)
        recv_token_indices = tree_checkpoint_name(recv_token_indices, _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES)
        local_group_sizes = tree_checkpoint_name(local_group_sizes, _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES)
        return DeepEPLocalAssignments(x_dispatch, assignment_weights, recv_token_indices, local_group_sizes)


def _collapse_deepep_local_assignments(
    out_dispatch: Float[Array, "TK D"],
    assignment_weights: Float[Array, "TK"],
    recv_token_indices: Int[Array, "TK"],
    *,
    recv_capacity: int,
    num_recv_tokens: Int[Array, ""],
) -> Float[Array, "TR D"]:
    with jax.named_scope("deepep_collapse_local_assignments"):
        recv_out = jax.ops.segment_sum(
            out_dispatch * assignment_weights[:, None],
            recv_token_indices,
            num_segments=recv_capacity,
            indices_are_sorted=False,
        )
        recv_valid = jnp.arange(recv_capacity, dtype=jnp.int32) < num_recv_tokens
        return jnp.where(recv_valid[:, None], recv_out, 0)


def _moe_mlp_ep_deepep_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: MoERematMode = "none",
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """DeepEP dispatch/combine path for an intranode expert mesh."""
    del capacity_factor
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    ep_size = num_experts // local_experts
    max_recv_tokens = x_local.shape[0] * ep_size

    with jax.named_scope("dispatch"):
        with jax.named_scope("deepep_layout"):
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
                selected_experts_local,
                num_ranks=ep_size,
                num_experts=num_experts,
            )
        with jax.named_scope("deepep_dispatch_transport"):
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                _local_expert_counts,
                num_recv_tokens,
            ) = deepep_dispatch_intranode(
                x_local,
                selected_experts_local,
                combine_weights_local,
                num_tokens_per_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
            )
        recv_x = tree_checkpoint_name(recv_x, _CHECKPOINT_DEEPEP_RECV_X)
        recv_topk_idx = tree_checkpoint_name(recv_topk_idx, _CHECKPOINT_DEEPEP_RECV_TOPK_IDX)
        recv_topk_weights = tree_checkpoint_name(recv_topk_weights, _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS)
        recv_src_idx = tree_checkpoint_name(recv_src_idx, _CHECKPOINT_DEEPEP_RECV_SRC_IDX)
        rank_prefix_matrix = tree_checkpoint_name(rank_prefix_matrix, _CHECKPOINT_DEEPEP_RANK_PREFIX_MATRIX)
        channel_prefix_matrix = tree_checkpoint_name(channel_prefix_matrix, _CHECKPOINT_DEEPEP_CHANNEL_PREFIX_MATRIX)
        recv_channel_prefix_matrix = tree_checkpoint_name(
            recv_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_RECV_CHANNEL_PREFIX_MATRIX,
        )
        send_head = tree_checkpoint_name(send_head, _CHECKPOINT_DEEPEP_SEND_HEAD)
        num_recv_tokens = tree_checkpoint_name(num_recv_tokens, _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS)
        is_token_in_rank = tree_checkpoint_name(is_token_in_rank, _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK)
        num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
        local_assignments = _pack_deepep_local_assignments(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            local_experts=local_experts,
            num_recv_tokens=num_recv_tokens_scalar,
        )
        x_dispatch = tree_checkpoint_name(local_assignments.x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    out_dispatch = _deepep_moe_up_down_remat(
        x_dispatch,
        local_assignments.local_group_sizes,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        remat_mode=remat_mode,
    )

    with jax.named_scope("combine"):
        recv_out = _collapse_deepep_local_assignments(
            out_dispatch,
            local_assignments.assignment_weights,
            local_assignments.recv_token_indices,
            recv_capacity=recv_x.shape[0],
            num_recv_tokens=num_recv_tokens_scalar,
        )
        with jax.named_scope("deepep_combine_transport"):
            out_local, _ = deepep_combine_intranode(
                recv_out,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                num_recv_tokens,
                is_token_in_rank,
            )
        dropped_total = jnp.array(0, dtype=jnp.int32)
    return out_local.astype(x_local.dtype), dropped_total
