# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter-add local Grug MoE backend."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from shape_extensions import Int, IntVar

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _prepare_moe_dispatch,
    _zero_dropped_assignments,
    _zero_inactive_grouped_rows,
    split_moe_w13_output,
)


def _moe_mlp_local_scatter[T: IntVar, K: IntVar, H: IntVar, E: IntVar, I: IntVar, I2: IntVar](
    x: jax.Array[[T, H]],
    selected_experts: jax.Array[[T, K]],
    combine_weights: jax.Array[[T, K]],
    token_valid: jax.Array[[T]],
    moe_w13: jax.Array[[E, H, I2]],
    moe_w2: jax.Array[[E, I, H]],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: Int[E],
) -> tuple[jax.Array[[T, H]], jax.Array[[]]]:
    """Local fallback MoE path: sorted grouped GMM then scatter-add combine."""
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        token_valid,
        num_experts=num_experts,
    )
    cumulative_group_sizes = jnp.cumsum(group_sizes).astype(jnp.int32)
    x_dispatch = _zero_inactive_grouped_rows(x_dispatch, cumulative_group_sizes)
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        # Rows past the last group are unspecified kernel output. Every consumer between the
        # two projections is row-local or group-bounded, so only the combine boundary below
        # needs zeroing (a zero weight times an unspecified row is not zero).
        # haliax's `ragged_dot` is unshaped, so pin the result.
        w13_out: jax.Array[[T * K, I2]] = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch: jax.Array[[T * K, H]] = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        weighted = _zero_inactive_grouped_rows(out_dispatch * w_dispatch[:, None], cumulative_group_sizes)
        out = jnp.zeros_like(x).at[token_dispatch].add(weighted, mode="drop")
    return out, _zero_dropped_assignments()
