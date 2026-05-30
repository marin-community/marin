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
    _CHECKPOINT_EXPERT_HIDDEN,
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


def _pack_deepep_local_assignments(
    recv_x: Float[Array, "TR D"],
    recv_topk_idx: Int[Array, "TR K"],
    recv_topk_weights: Float[Array, "TR K"],
    *,
    expert_start: Int[Array, ""],
    local_experts: int,
    num_recv_tokens: Int[Array, ""],
) -> DeepEPLocalAssignments:
    max_recv_tokens, topk = recv_topk_idx.shape
    total_assignments = max_recv_tokens * topk

    recv_token_indices = jnp.repeat(jnp.arange(max_recv_tokens, dtype=jnp.int32), topk)
    expert_flat = (recv_topk_idx.reshape(-1) - expert_start).astype(jnp.int32)
    recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < num_recv_tokens
    local_mask = recv_valid[:, None] & (recv_topk_idx >= expert_start) & (recv_topk_idx < expert_start + local_experts)
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
    return DeepEPLocalAssignments(x_dispatch, assignment_weights, recv_token_indices, local_group_sizes)


def _collapse_deepep_local_assignments(
    out_dispatch: Float[Array, "TK D"],
    assignment_weights: Float[Array, "TK"],
    recv_token_indices: Int[Array, "TK"],
    *,
    recv_capacity: int,
    num_recv_tokens: Int[Array, ""],
) -> Float[Array, "TR D"]:
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

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts
    max_recv_tokens = x_local.shape[0] * ep_size

    with jax.named_scope("dispatch"):
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
            selected_experts_local,
            num_ranks=ep_size,
            num_experts=num_experts,
        )
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
        num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
        local_assignments = _pack_deepep_local_assignments(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            expert_start=expert_start,
            local_experts=local_experts,
            num_recv_tokens=num_recv_tokens_scalar,
        )
        x_dispatch = tree_checkpoint_name(local_assignments.x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, local_assignments.local_group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, local_assignments.local_group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        recv_out = _collapse_deepep_local_assignments(
            out_dispatch,
            local_assignments.assignment_weights,
            local_assignments.recv_token_indices,
            recv_capacity=recv_x.shape[0],
            num_recv_tokens=num_recv_tokens_scalar,
        )
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
