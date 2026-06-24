# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug.sharding import _batch_axes


class RingDispatchUp(NamedTuple):
    """Receiver-local rows and metadata for ring-style expert dispatch-up."""

    x_dispatch: jax.Array
    group_sizes: jax.Array
    token_local: jax.Array
    weight_dispatch: jax.Array
    valid: jax.Array
    dropped_local: jax.Array


def _dispatch_up_ep_ring_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> RingDispatchUp:
    """All-gather tokens and pack this rank's local expert rows for W13."""

    with jax.named_scope("moe_ep_ring/gather_inputs"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

    with jax.named_scope("moe_ep_ring/route_local_experts"):
        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        local_capacity = int(math.ceil(capacity_factor * assignments / ep_size))
        local_capacity = max(local_experts, local_capacity)

        expert_axis = jax.lax.axis_index("expert")
        expert_start = expert_axis * local_experts
        local_expert: jax.Array = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)

    with jax.named_scope("moe_ep_ring/count_local_assignments"):
        # Keep only the assignments this shard will execute, ordered by
        # (local expert id, original flat position). This avoids the global
        # argsort + fused takes over all assignments that dominated high-EP
        # shapes, while preserving the grouped layout expected by ragged_dot.
        local_expert = jnp.where(local_mask, local_expert, 0)
        # TPU lowers this small-expert count reduction better as a dense
        # compare+sum than as `bincount`.
        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        local_mask_i32 = local_mask.astype(jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
        valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    with jax.named_scope("moe_ep_ring/select_local_assignments"):
        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    with jax.named_scope("moe_ep_ring/dispatch_gather_tokens"):
        token_local = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))

    with jax.named_scope("moe_ep_ring/group_sizes"):
        group_sizes = accepted_counts

    return RingDispatchUp(
        x_dispatch=x_dispatch,
        group_sizes=group_sizes,
        token_local=token_local,
        weight_dispatch=weight_dispatch,
        valid=valid,
        dropped_local=dropped_local,
    )


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "TL D"],
    selected_experts_local: Int[Array, "TL K"],
    combine_weights_local: Float[Array, "TL K"],
    moe_w13_local: Float[Array, "EL D I2"],
    moe_w2_local: Float[Array, "EL I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    remat_mode: str = "none",
) -> tuple[Float[Array, "TL D"], Int[Array, ""]]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect."""
    del remat_mode
    local_experts = moe_w13_local.shape[0]
    dispatch = _dispatch_up_ep_ring_local(
        x_local,
        selected_experts_local,
        combine_weights_local,
        local_experts=local_experts,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    with jax.named_scope("moe_expert_mlp/w13_ragged_dot"):
        w13_out = tree_checkpoint_name(
            ragged_dot(dispatch.x_dispatch, moe_w13_local, dispatch.group_sizes),
            _CHECKPOINT_EXPERT_HIDDEN,
        )
    with jax.named_scope("moe_expert_mlp/split_gate_up"):
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    with jax.named_scope("moe_expert_mlp/activation"):
        hidden = activation_fn(gate) * up
    with jax.named_scope("moe_expert_mlp/w2_ragged_dot"):
        out_dispatch = tree_checkpoint_name(
            ragged_dot(hidden, moe_w2_local, dispatch.group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("moe_ep_ring/combine_scatter_add"):
        out_global = (
            jnp.zeros(
                (x_local.shape[0] * (num_experts // local_experts), x_local.shape[1]),
                dtype=out_dispatch.dtype,
            )
            .at[dispatch.token_local]
            .add(out_dispatch * dispatch.weight_dispatch[:, None], mode="drop")
        )
    with jax.named_scope("moe_ep_ring/psum_scatter_output"):
        # #2710 ring EP strategy: collect only this shard's token slice after
        # reducing contributions from experts across the EP mesh.
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
    with jax.named_scope("moe_ep_ring/psum_dropped_assignments"):
        dropped_total = jax.lax.psum(dispatch.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
