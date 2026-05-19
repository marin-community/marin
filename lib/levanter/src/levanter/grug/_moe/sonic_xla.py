# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Sonic-style local Grug MoE backends expressed through XLA/JAX primitives."""

from collections.abc import Callable

import jax
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.down_gather import custom_vjp_interleaved_down_gather_sum
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_MOE_OUTPUT,
    _gather_sum_reference,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _zero_dropped_assignments,
    split_moe_w13_output,
)

# Leave ragged_dot backend selection inside haliax.nn.ragged_dot so the
# custom-VJP interleaved path and reference paths use the same backend policy.


def _moe_mlp_local_sonic_xla(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Local Sonic-style path with XLA gather-sum combine and concat W13 layout."""
    token_ids_sort, dispatch_positions, group_sizes, _sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            num_experts=num_experts,
        )
    )
    x_dispatch = tree_checkpoint_name(x[token_ids_sort], _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("gather_sum"):
        out = _gather_sum_reference(out_dispatch, dispatch_positions, combine_weights)
    return out, _zero_dropped_assignments()


def _moe_mlp_local_sonic_xla_interleaved_reference(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Local Sonic-style path with XLA gather-sum combine and interleaved W13 layout."""
    token_ids_sort, dispatch_positions, group_sizes, _sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            num_experts=num_experts,
        )
    )
    x_dispatch = tree_checkpoint_name(x[token_ids_sort], _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=True)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("gather_sum"):
        out = _gather_sum_reference(out_dispatch, dispatch_positions, combine_weights)
    return out, _zero_dropped_assignments()


def _moe_mlp_local_sonic_xla_interleaved(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Local Sonic-style path with XLA W13 and a custom-VJP down/gather boundary."""
    del activation_fn
    token_ids_sort, dispatch_positions, group_sizes, sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            num_experts=num_experts,
        )
    )
    x_dispatch = tree_checkpoint_name(x[token_ids_sort], _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        out = tree_checkpoint_name(
            custom_vjp_interleaved_down_gather_sum(
                w13_out,
                combine_weights,
                moe_w2,
                token_ids_sort,
                sorted_assignment_ids,
                dispatch_positions,
                group_sizes,
            ),
            _CHECKPOINT_MOE_OUTPUT,
        )
    return out, _zero_dropped_assignments()
