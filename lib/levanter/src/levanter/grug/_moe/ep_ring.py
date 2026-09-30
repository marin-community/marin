# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring expert-parallel Grug MoE backend."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from shape_extensions import IntVar

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _assignment_validity,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _prefix_cap_counts


def _moe_mlp_ep_ring_local[Tlocal: IntVar, K: IntVar, H: IntVar, Elocal: IntVar, I: IntVar, I2: IntVar, Tg: IntVar](
    x_local: jax.Array[[Tlocal, H]],
    selected_experts_local: jax.Array[[Tlocal, K]],
    combine_weights_local: jax.Array[[Tlocal, K]],
    token_valid_local: jax.Array[[Tlocal]],
    moe_w13_local: jax.Array[[Elocal, H, I2]],
    moe_w2_local: jax.Array[[Elocal, I, H]],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
) -> tuple[jax.Array[[Tlocal, H]], CapacityDrops]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect.

    ``Tlocal``/``K``/``H``/``Elocal``/``I``/``I2`` describe this shard's per-device
    shapes, as seen inside the ``shard_map`` this function is the body of. ``Tg``
    (global token count after the all-gather) is introduced fresh here: the
    shard_map wrapper (applied outside this module, in ``grug_moe.py``) erases the
    function's generic signature the same way ``functools.partial`` does, so
    there is no static connection between this function's per-device type
    parameters and the caller's global-shape ones -- the relationship
    (``Tg == Tlocal * expert_axis_size``) is a runtime/convention invariant only.
    """
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    with jax.named_scope("gather"):
        # `all_gather` drops the shape (the gathered size depends on the mesh axis); pin a fresh Tg dim.
        x_global: jax.Array[[Tg, H]] = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global: jax.Array[[Tg, K]] = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global: jax.Array[[Tg, K]] = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        token_valid_global: jax.Array[[Tg]] = jax.lax.all_gather(token_valid_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat: jax.Array[[Tg * K]] = selected_experts_global.reshape(assignments)
        weight_flat: jax.Array[[Tg * K]] = combine_weights_global.reshape(assignments)

        local_experts = moe_w13_local.shape[0]
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        physical_capacity = int(math.ceil(capacity_factor * assignments / ep_size))
        physical_capacity = min(assignments, max(local_experts, physical_capacity))
        assignment_valid = _assignment_validity(token_valid_global, tokens=tokens, topk=topk)
        valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
        logical_capacity = _scaled_capacity(
            valid_assignments,
            capacity_factor=capacity_factor,
            divisor=ep_size,
            minimum=local_experts,
            maximum=physical_capacity,
        )

        expert_axis = jax.lax.axis_index("expert")
        expert_start = expert_axis * local_experts
        local_expert: jax.Array[[Tg * K]] = expert_flat - expert_start
        local_mask = assignment_valid & jnp.logical_and(local_expert >= 0, local_expert < local_experts)

        # Keep only the assignments this shard will execute, ordered by
        # (local expert id, original flat position). This avoids the global
        # argsort + fused takes over all assignments that dominated high-EP
        # shapes, while preserving the grouped layout expected by ragged_dot.
        local_expert = jnp.where(local_mask, local_expert, 0)
        # TPU lowers this small-expert count reduction better as a dense
        # compare+sum than as `bincount`.
        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        local_mask_i32 = local_mask.astype(jnp.int32)
        counts: jax.Array[[Elocal]] = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=logical_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
        # `physical_capacity` is a plain int derived from assignment demand, so the capacity-bounded
        # buffers below use an unknown length rather than a type parameter.
        valid: jax.Array[[int]] = jnp.arange(physical_capacity, dtype=jnp.int32) < accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key: jax.Array[[Tg * K]] = jnp.where(local_mask, max_order_key - order_key, -1)
        # pyrefly: ignore[bad-assignment]  # top_k with a non-literal `k` falls back to a rank-0 result.
        _, local_idx = jax.lax.top_k(selection_key, physical_capacity)
        local_idx: jax.Array[[int]] = local_idx

        token_local: jax.Array[[int]] = jnp.floor_divide(local_idx, topk)
        weight_local: jax.Array[[int]] = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take: jax.Array[[int, H]] = jnp.take(x_global, token_local, axis=0)
        x_dispatch: jax.Array[[int, H]] = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch: jax.Array[[int]] = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts
    # `local_idx` pads by appending invalid rows at the end; keep GMM segment
    # boundaries aligned by attributing padding to the final expert segment.
    group_sizes = group_sizes.at[-1].add(physical_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    with jax.named_scope("moe_up_down"):
        w13_out: jax.Array[[int, I2]] = tree_checkpoint_name(
            ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN
        )
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch: jax.Array[[int, H]] = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out_global: jax.Array[[Tg, H]] = (
            jnp.zeros_like(x_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        )
        # #2710 ring EP strategy: collect only this shard's token slice after
        # reducing contributions from experts across the EP mesh.
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=jnp.zeros_like(dropped_total), receiver_dropped=dropped_total)
