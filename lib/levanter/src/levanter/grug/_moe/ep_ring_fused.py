# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bulk-ring expert parallelism with an output-oriented combine."""

import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug.sharding import _batch_axes


RingCombineImplementation = Literal["xla", "reference"]


def ring_combine_reference(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    *,
    tokens: int,
    topk: int,
) -> Float[Array, "T H"]:
    """Combine compact expert outputs with the ring backend's scatter oracle."""
    token_indices = jnp.floor_divide(local_assignment_indices, topk)
    weights = jnp.take(assignment_weights, local_assignment_indices, axis=0).astype(out_dispatch.dtype)
    contributions = jnp.where(valid[:, None], out_dispatch * weights[:, None], 0)
    return (
        jnp.zeros((tokens, out_dispatch.shape[1]), dtype=out_dispatch.dtype)
        .at[token_indices]
        .add(
            contributions,
            mode="drop",
        )
    )


def ring_combine_xla(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    *,
    tokens: int,
    topk: int,
) -> Float[Array, "T H"]:
    """Combine by gathering compact rows, writing each global output row once.

    The assignment-to-row table is small compared with the global hidden-state
    tensor. XLA can fuse the final gather, mask, weight, and top-k reduction into
    the producer of the ReduceScatter operand, avoiding a full-size zero fill
    followed by sparse scatter updates.
    """
    assignments = tokens * topk
    capacity = out_dispatch.shape[0]
    sentinel = jnp.array(capacity, dtype=jnp.int32)
    dispatch_rows = jnp.full((assignments,), sentinel, dtype=jnp.int32)
    compact_rows = jnp.arange(capacity, dtype=jnp.int32)
    dispatch_rows = dispatch_rows.at[local_assignment_indices].set(
        jnp.where(valid, compact_rows, sentinel),
        mode="drop",
    )
    dispatch_rows = dispatch_rows.reshape(tokens, topk)

    # Clip the sentinel to an in-bounds row before masking it. Capacity is
    # always at least the number of local experts and therefore nonzero.
    gathered = jnp.take(out_dispatch, jnp.minimum(dispatch_rows, capacity - 1), axis=0)
    weights = assignment_weights.reshape(tokens, topk).astype(out_dispatch.dtype)
    contributions = jnp.where((dispatch_rows < capacity)[..., None], gathered * weights[..., None], 0)
    return jnp.sum(contributions, axis=1, dtype=out_dispatch.dtype)


def ring_combine(
    out_dispatch: Float[Array, "C H"],
    local_assignment_indices: Int[Array, "C"],
    valid: Bool[Array, "C"],
    assignment_weights: Float[Array, "A"],
    *,
    tokens: int,
    topk: int,
    implementation: RingCombineImplementation = "xla",
) -> Float[Array, "T H"]:
    """Combine compact ring outputs using an output-oriented or reference path."""
    if implementation == "xla":
        return ring_combine_xla(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
        )
    if implementation == "reference":
        return ring_combine_reference(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
        )
    raise ValueError(f"Unknown ring combine implementation {implementation!r}")


def _moe_mlp_ep_ring_fused_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    combine_implementation: RingCombineImplementation = "xla",
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run bulk-ring MoE with a fused output-oriented combine producer."""
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

        local_experts = moe_w13_local.shape[0]
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        local_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments / ep_size)))

        expert_axis = jax.lax.axis_index("expert")
        local_expert = expert_flat - expert_axis * local_experts
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
        local_expert = jnp.where(local_mask, local_expert, 0)

        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask[:, None].astype(jnp.int32),
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
        valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, local_capacity)

        token_local = jnp.floor_divide(local_idx, topk)
        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, 0)
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        intermediate_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [intermediate_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out_global = ring_combine(
            out_dispatch,
            local_idx,
            valid,
            weight_flat,
            tokens=tokens,
            topk=topk,
            implementation=combine_implementation,
        )
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
