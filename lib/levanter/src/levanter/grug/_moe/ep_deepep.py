# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP intranode expert-parallel Grug MoE backend.

DeepEP source: https://github.com/deepseek-ai/DeepEP
"""

import math
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
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.kernels.deepep import (
    deepep_collapse_local_assignments,
    deepep_combine_intranode,
    deepep_dispatch_intranode_with_assignments,
    deepep_get_dispatch_layout,
)


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

    with jax.named_scope("dispatch"):
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
            selected_experts_local,
            num_ranks=ep_size,
            num_experts=num_experts,
        )
        (
            recv_x,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            local_group_sizes,
            num_recv_tokens,
            x_dispatch,
            assignment_weights,
            recv_token_indices,
            assignment_destinations,
        ) = deepep_dispatch_intranode_with_assignments(
            x_local,
            selected_experts_local,
            combine_weights_local,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            max_recv_tokens=max_recv_tokens,
        )
        accepted_group_sizes = _prefix_cap_counts(local_group_sizes, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_group_sizes, dtype=jnp.int32)
        dropped_local = jnp.sum(local_group_sizes, dtype=jnp.int32) - accepted_total
        x_dispatch = x_dispatch[:local_capacity]
        assignment_weights = assignment_weights[:local_capacity]
        recv_token_indices = recv_token_indices[:local_capacity]
        local_assignments = DeepEPLocalAssignments(
            x_dispatch,
            assignment_weights,
            recv_token_indices,
            accepted_group_sizes,
        )
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

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
        recv_out = deepep_collapse_local_assignments(
            out_dispatch,
            local_assignments.assignment_weights,
            local_assignments.recv_token_indices,
            assignment_destinations,
            local_assignments.local_group_sizes,
            num_recv_tokens,
            recv_capacity=recv_x.shape[0],
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
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local.astype(x_local.dtype), dropped_total
