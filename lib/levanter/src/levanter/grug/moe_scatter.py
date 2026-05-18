# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter-add local Grug MoE backend."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug.moe_common import _prepare_moe_dispatch, _zero_dropped_assignments, split_moe_w13_output


def _moe_mlp_local_scatter(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Local fallback MoE path: sorted grouped GMM then scatter-add combine."""
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        num_experts=num_experts,
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, "grug_moe_dispatch_input")

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), "grug_moe_expert_hidden")
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2, group_sizes),
            "grug_moe_dispatch_output",
        )

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()
