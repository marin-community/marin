# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Grouped-layer assigned-token expert-parallel Grug MoE backend.

This is a JAX reference for the DeepEP-style transport shape we want for grouped
MoE banks: keep the layer-group axis visible through the EP boundary, flatten it
into the routed token/expert space, and do one variable-size transport for the
whole grouped bank instead of vmapping a per-layer transport.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from levanter.grug._moe.common import MoERematMode
from levanter.grug._moe.ep_assigned_token import _moe_mlp_ep_assigned_token_local


def _moe_mlp_ep_grouped_assigned_token_local(
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
    valid_group_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run a grouped MoE bank through one assigned-token EP transport.

    Inputs are shard-local slices inside a `shard_map`, with shapes:
    - x/selected/combine: `[G, T_local, ...]`
    - weights: `[G, E_local, ...]`

    The implementation treats `(group, expert)` as one logical expert id:
    `expert_id + group_id * num_experts`. That lets the existing single-layer
    assigned-token backend route the entire grouped bank in one collective pair.
    """
    group_size = int(x_local.shape[0])
    if valid_group_size is None:
        valid_group_size = group_size
    if valid_group_size < 1 or valid_group_size > group_size:
        raise ValueError(f"valid_group_size must be in [1, {group_size}], got {valid_group_size}")

    tokens_per_group = int(x_local.shape[1])
    hidden_dim = int(x_local.shape[2])
    topk = int(selected_experts_local.shape[2])
    local_experts = int(moe_w13_local.shape[1])

    with jax.named_scope("moe_ep_grouped_assigned_token/flatten_group"):
        group_ids = jnp.arange(group_size, dtype=selected_experts_local.dtype)[:, None, None]
        group_valid = jnp.arange(group_size) < valid_group_size
        selected = selected_experts_local + group_ids * num_experts
        selected = jnp.where(group_valid[:, None, None], selected, jnp.zeros_like(selected))
        combine_weights = jnp.where(group_valid[:, None, None], combine_weights_local, 0)

        x_flat = x_local.reshape(group_size * tokens_per_group, hidden_dim)
        selected_flat = selected.reshape(group_size * tokens_per_group, topk)
        combine_flat = combine_weights.reshape(group_size * tokens_per_group, topk)
        w13_flat = moe_w13_local.reshape(group_size * local_experts, moe_w13_local.shape[2], moe_w13_local.shape[3])
        w2_flat = moe_w2_local.reshape(group_size * local_experts, moe_w2_local.shape[2], moe_w2_local.shape[3])

    with jax.named_scope("moe_ep_grouped_assigned_token/transport_and_mlp"):
        out_flat, dropped_total = _moe_mlp_ep_assigned_token_local(
            x_flat,
            selected_flat,
            combine_flat,
            w13_flat,
            w2_flat,
            activation_fn=activation_fn,
            num_experts=group_size * num_experts,
            capacity_factor=capacity_factor,
            remat_mode=remat_mode,
        )

    with jax.named_scope("moe_ep_grouped_assigned_token/unflatten_group"):
        out = out_flat.reshape(group_size, tokens_per_group, hidden_dim)
        out = jnp.where(group_valid[:, None, None], out, jnp.zeros_like(out))
        dropped = jnp.full((group_size,), dropped_total, dtype=dropped_total.dtype)
        dropped = jnp.where(group_valid, dropped, jnp.zeros_like(dropped))
    return out, dropped
