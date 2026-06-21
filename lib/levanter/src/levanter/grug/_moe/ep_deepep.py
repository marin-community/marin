# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP intranode expert-parallel Grug MoE backend.

DeepEP source: https://github.com/deepseek-ai/DeepEP
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS,
    _CHECKPOINT_DEEPEP_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK,
    _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES,
    _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS,
    _CHECKPOINT_DEEPEP_RANK_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_CHANNEL_PREFIX_MATRIX,
    _CHECKPOINT_DEEPEP_RECV_SRC_IDX,
    _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES,
    _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS,
    _CHECKPOINT_DEEPEP_RECV_X,
    _CHECKPOINT_DEEPEP_SEND_HEAD,
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    MOE_REMAT_SAVE_NAMES,
    MoERematMode,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.kernels.deepep import (
    deepep_collapse_local_assignments,
    deepep_combine_intranode,
    deepep_dispatch_intranode_with_assignments,
    deepep_get_dispatch_layout,
)


def _deepep_moe_up_down(
    x_dispatch: Float[Array, "TK D"],
    local_group_sizes: Int[Array, "EL"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "TK D"]:
    with jax.named_scope("moe_expert_mlp/w13_ragged_dot"):
        w13_out = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, local_group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
    with jax.named_scope("moe_expert_mlp/split_gate_up"):
        moe_dim = moe_w2_local.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
    with jax.named_scope("moe_expert_mlp/activation"):
        hidden = activation_fn(gate) * up
    with jax.named_scope("moe_expert_mlp/w2_ragged_dot"):
        return tree_checkpoint_name(
            ragged_dot(hidden, moe_w2_local, local_group_sizes),
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
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    ep_size = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    local_capacity = int(math.ceil(capacity_factor * x_local.shape[0] * topk))
    local_capacity = max(local_experts, local_capacity)
    max_recv_tokens = x_local.shape[0] * ep_size

    with jax.named_scope("moe_ep_deepep/dispatch"):
        with jax.named_scope("deepep_layout"):
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
                selected_experts_local,
                num_ranks=ep_size,
                num_experts=num_experts,
            )
        with jax.named_scope("deepep_dispatch_transport"):
            dispatch = deepep_dispatch_intranode_with_assignments(
                x_local,
                selected_experts_local,
                combine_weights_local,
                num_tokens_per_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                assignment_capacity=local_capacity,
            )
        recv_x = tree_checkpoint_name(dispatch.recv_x, _CHECKPOINT_DEEPEP_RECV_X)
        recv_topk_weights = tree_checkpoint_name(dispatch.recv_topk_weights, _CHECKPOINT_DEEPEP_RECV_TOPK_WEIGHTS)
        recv_src_idx = tree_checkpoint_name(dispatch.recv_src_idx, _CHECKPOINT_DEEPEP_RECV_SRC_IDX)
        rank_prefix_matrix = tree_checkpoint_name(dispatch.rank_prefix_matrix, _CHECKPOINT_DEEPEP_RANK_PREFIX_MATRIX)
        channel_prefix_matrix = tree_checkpoint_name(
            dispatch.channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_CHANNEL_PREFIX_MATRIX,
        )
        recv_channel_prefix_matrix = tree_checkpoint_name(
            dispatch.recv_channel_prefix_matrix,
            _CHECKPOINT_DEEPEP_RECV_CHANNEL_PREFIX_MATRIX,
        )
        send_head = tree_checkpoint_name(dispatch.send_head, _CHECKPOINT_DEEPEP_SEND_HEAD)
        num_recv_tokens = tree_checkpoint_name(dispatch.num_recv_tokens, _CHECKPOINT_DEEPEP_NUM_RECV_TOKENS)
        is_token_in_rank = tree_checkpoint_name(is_token_in_rank, _CHECKPOINT_DEEPEP_IS_TOKEN_IN_RANK)
        local_group_sizes = _prefix_cap_counts(dispatch.local_group_sizes, capacity=local_capacity)
        accepted_total = jnp.sum(local_group_sizes, dtype=jnp.int32)
        dropped_local = jnp.sum(dispatch.local_group_sizes, dtype=jnp.int32) - accepted_total
        local_group_sizes = tree_checkpoint_name(local_group_sizes, _CHECKPOINT_DEEPEP_LOCAL_GROUP_SIZES)
        x_dispatch = tree_checkpoint_name(dispatch.x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        assignment_weights = tree_checkpoint_name(
            dispatch.assignment_weights,
            _CHECKPOINT_DEEPEP_ASSIGNMENT_WEIGHTS,
        )
        recv_token_indices = tree_checkpoint_name(
            dispatch.recv_token_indices,
            _CHECKPOINT_DEEPEP_RECV_TOKEN_INDICES,
        )
        assignment_destinations = dispatch.assignment_destinations

    out_dispatch = _deepep_moe_up_down_remat(
        x_dispatch,
        local_group_sizes,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        remat_mode=remat_mode,
    )

    with jax.named_scope("moe_ep_deepep/combine"):
        with jax.named_scope("deepep_collapse_local_assignments"):
            recv_out = deepep_collapse_local_assignments(
                out_dispatch,
                assignment_weights,
                recv_token_indices,
                assignment_destinations,
                local_group_sizes,
                num_recv_tokens,
                recv_capacity=recv_x.shape[0],
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
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local.astype(x_local.dtype), dropped_total
