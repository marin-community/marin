# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter-add local Grug MoE backend."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _zero_dropped_assignments,
    split_moe_w13_output,
)


def _gather_sum_moe_output(
    dispatch_output: Float[Array, "TK H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T H"]:
    """Combine each token's expert outputs in a fixed top-k order."""

    tokens, topk = dispatch_positions.shape
    out = jnp.zeros((tokens, dispatch_output.shape[1]), dtype=dispatch_output.dtype)
    weights = combine_weights.astype(dispatch_output.dtype)
    for topk_index in range(topk):
        expert_output = jnp.take(
            dispatch_output,
            dispatch_positions[:, topk_index],
            axis=0,
            unique_indices=True,
        )
        out = out + expert_output * weights[:, topk_index, None]
    return out


def _moe_mlp_local_scatter(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """Local fallback MoE path: sorted grouped GMM then fixed-order combine."""
    token_dispatch, dispatch_positions, group_sizes, _ = _prepare_moe_dispatch_indices_with_assignment_ids(
        selected_experts,
        num_experts=num_experts,
    )
    x_dispatch = x[token_dispatch]
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        out = _gather_sum_moe_output(out_dispatch, dispatch_positions, combine_weights)
    return out, _zero_dropped_assignments()
