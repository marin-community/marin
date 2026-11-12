# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ring expert-parallel Grug MoE backend."""

import math
from collections.abc import Callable
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _assignment_validity,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _assignment_sources, _prefix_cap_counts

RingAssignmentSelection: TypeAlias = Literal["top_k", "rank_scatter"]
"""How the ring backends order a shard's accepted assignments by expert.

``top_k`` takes a ``top_k`` over (local expert id, flat position) keys. ``rank_scatter`` scatters each flat position to
its expert-major rank instead, which avoids the sort ``top_k`` lowers to. On 2x MI355X it was faster; on H100, GB200
and TPU v4 it made the ``ring`` layer 8-16% slower in forward and backward, so it is opt-in.
"""


def _expert_major_slots(
    local_expert: Int[Array, "A"], local_mask: Bool[Array, "A"], one_hot: Int[Array, "A Elocal"], *, out_of_range: int
) -> Int[Array, "A"]:
    """Each assignment's rank in (local expert id, flat position) order, as distinct scatter indices.

    ``one_hot[a, e]`` is one when assignment ``a`` is masked in and routed to local expert ``e``. Masked-in
    assignments get their rank, which lies below the assignment count: their expert's start plus their rank within
    the expert. Masked-out assignments get distinct slots from ``max(assignments, out_of_range)`` up, so a scatter
    into ``out_of_range`` slots drops them without colliding with any masked-in slot.
    """
    assignments = local_expert.shape[0]
    position = jnp.arange(assignments, dtype=jnp.int32)
    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    rank_in_expert = jnp.take_along_axis(jnp.cumsum(one_hot, axis=0, dtype=jnp.int32), local_expert[:, None], axis=1)
    slot = (jnp.cumsum(counts) - counts)[local_expert] + rank_in_expert[:, 0] - 1
    return jnp.where(local_mask, slot, max(assignments, out_of_range) + position)


def _select_local_assignments(
    local_expert: Int[Array, "A"],
    local_mask: Bool[Array, "A"],
    *,
    local_experts: int,
    logical_capacity: jax.Array,
    physical_capacity: int,
    selection: RingAssignmentSelection,
) -> tuple[Int[Array, "Elocal"], jax.Array, Int[Array, "C"], Bool[Array, "C"]]:
    """Pick this shard's assignments in (local expert id, flat position) order under ring's capacity rule.

    ``local_expert`` must be zero where ``local_mask`` is false. Returns accepted counts per expert, the dropped
    assignment count, picked flat positions, and validity for each of the ``physical_capacity`` picks. Both
    selections pick the same valid positions; the invalid picks past the accepted total differ, and every caller
    masks them.
    """
    assignments = local_expert.shape[0]
    # TPU lowers this small-expert count reduction better as a dense compare+sum than as `bincount`.
    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    one_hot = ((local_expert[:, None] == expert_ids[None, :]) & local_mask[:, None]).astype(jnp.int32)
    counts = jnp.sum(one_hot, axis=0, dtype=jnp.int32)
    accepted_counts = _prefix_cap_counts(counts, capacity=logical_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    dropped = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    valid = jnp.arange(physical_capacity, dtype=jnp.int32) < accepted_total

    position = jnp.arange(assignments, dtype=jnp.int32)
    if selection == "top_k":
        # A top_k over (local expert id, flat position) keys avoids a global argsort while preserving the
        # grouped layout expected by ragged_dot.
        order_key = local_expert * assignments + position
        selection_key = jnp.where(local_mask, local_experts * assignments - order_key, -1)
        _, picked = jax.lax.top_k(selection_key, physical_capacity)
    elif selection == "rank_scatter":
        # Scattering the flat positions to their expert-major slots keeps the grouped layout without a sort.
        slot = _expert_major_slots(local_expert, local_mask, one_hot, out_of_range=physical_capacity)
        picked = jnp.zeros((physical_capacity,), jnp.int32).at[slot].set(position, mode="drop", unique_indices=True)
    else:
        raise ValueError(f"Unknown ring assignment selection {selection!r}")
    # Picks past accepted_total are padding: invalid and outside every group, so ragged_dot skips them and leaves
    # their output unspecified; the dispatch and combine read only the valid rows.
    return accepted_counts, dropped, picked, valid


def _gather_sum_slots(rows: Float[Array, "P H"], slots: Int[Array, "T K"]) -> Float[Array, "T H"]:
    """``out[t] = sum_k rows[slots[t, k]]`` with out-of-range slots read as zero, summed in float32."""
    out = jnp.zeros((slots.shape[0], rows.shape[1]), dtype=jnp.float32)
    for k in range(slots.shape[1]):
        out += jnp.take(rows, slots[:, k], axis=0, mode="fill", fill_value=0).astype(jnp.float32)
    return out.astype(rows.dtype)


def _take_valid_rows(
    x: Float[Array, "T H"], slot_token: Int[Array, "P"], slot_valid: Bool[Array, "P"]
) -> Float[Array, "P H"]:
    rows = jnp.take(x, slot_token, axis=0)
    return jnp.where(slot_valid[:, None], rows, jnp.zeros_like(rows))


# The dispatch gather and the combine are transposes of each other. Autodiff would turn the gather's transpose,
# and the combine itself, into scatter-adds over the token buffer, which ROCm runs at about 1.2 TB/s. With the
# inverse map `slots` both directions are gathers: one reads each token's (at most K) slot rows.
@jax.custom_vjp
def _dispatch_rows(
    x: Float[Array, "T H"], slot_token: Int[Array, "P"], slot_valid: Bool[Array, "P"], slots: Int[Array, "T K"]
) -> Float[Array, "P H"]:
    return _take_valid_rows(x, slot_token, slot_valid)


def _dispatch_rows_fwd(x, slot_token, slot_valid, slots):
    return _take_valid_rows(x, slot_token, slot_valid), slots


def _dispatch_rows_bwd(slots, g):
    return _gather_sum_slots(g, slots), None, None, None


_dispatch_rows.defvjp(_dispatch_rows_fwd, _dispatch_rows_bwd)


@jax.custom_vjp
def _combine_rows(
    rows: Float[Array, "P H"], slot_token: Int[Array, "P"], slot_valid: Bool[Array, "P"], slots: Int[Array, "T K"]
) -> Float[Array, "T H"]:
    return _gather_sum_slots(rows, slots)


def _combine_rows_fwd(rows, slot_token, slot_valid, slots):
    return _gather_sum_slots(rows, slots), (slot_token, slot_valid)


def _combine_rows_bwd(residuals, g):
    slot_token, slot_valid = residuals
    return _take_valid_rows(g, slot_token, slot_valid), None, None, None


_combine_rows.defvjp(_combine_rows_fwd, _combine_rows_bwd)


class _DispatchCombine(NamedTuple):
    """Moves rows from the gathered token buffer to this shard's dispatch slots, and sums them back per token.

    ``combine(rows, weights)`` scales each dispatch row by its routing weight before summing it into its token.
    """

    dispatch: Callable[[Float[Array, "T H"]], Float[Array, "P H"]]
    combine: Callable[[Float[Array, "P H"], Float[Array, "P"]], Float[Array, "T H"]]


# Builds a `_DispatchCombine` from the picked flat assignment positions, their validity, and the gathered
# token count and top-k.
_DispatchCombineFactory: TypeAlias = Callable[[Int[Array, "P"], Bool[Array, "P"], int, int], _DispatchCombine]


def _scatter_add_dispatch_combine(
    picked: Int[Array, "P"], valid: Bool[Array, "P"], tokens: int, topk: int
) -> _DispatchCombine:
    """Dispatch with a take and combine with a scatter-add; autodiff transposes each into the other."""
    token = jnp.floor_divide(picked, topk)
    # Invalid rows scatter out of range and are dropped: their ragged_dot output is unspecified, and a zero weight
    # times a NaN row is still NaN.
    combine_token = jnp.where(valid, token, tokens)
    return _DispatchCombine(
        dispatch=lambda x: _take_valid_rows(x, token, valid),
        combine=lambda rows, weights: jnp.zeros((tokens, rows.shape[1]), rows.dtype)
        .at[combine_token]
        .add(rows * weights[:, None], mode="drop"),
    )


def _gather_dispatch_combine(
    picked: Int[Array, "P"], valid: Bool[Array, "P"], tokens: int, topk: int
) -> _DispatchCombine:
    """Dispatch and combine, forward and backward, as gathers through the inverse map from assignments to slots."""
    token = jnp.floor_divide(picked, topk)
    # The slot holding each (token, k) assignment, or `P` where no valid slot does.
    assignments = tokens * topk
    slots = _assignment_sources(jnp.where(valid, picked, assignments), send_size=assignments).reshape(tokens, topk)
    return _DispatchCombine(
        dispatch=lambda x: _dispatch_rows(x, token, valid, slots),
        combine=lambda rows, weights: _combine_rows(rows * weights[:, None], token, valid, slots),
    )


def _moe_mlp_ep_ring_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    make_dispatch_combine: _DispatchCombineFactory,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
    ring_selection: RingAssignmentSelection,
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    """Ring-style EP routed path: all-gather dispatch + psum-scatter collect.

    ``make_dispatch_combine`` chooses how rows move between the gathered token buffer and the dispatch slots.
    """
    # #2710 ring EP strategy: gather tokens and their selected-expert routing
    # assignments across expert shards, then psum-scatter back to local tokens.
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)
        token_valid_global = jax.lax.all_gather(token_valid_local, "expert", tiled=True)

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
        local_expert: jax.Array = expert_flat - expert_start
        local_mask = assignment_valid & jnp.logical_and(local_expert >= 0, local_expert < local_experts)
        local_expert = jnp.where(local_mask, local_expert, 0)
        group_sizes, dropped_local, local_idx, valid = _select_local_assignments(
            local_expert,
            local_mask,
            local_experts=local_experts,
            logical_capacity=logical_capacity,
            physical_capacity=physical_capacity,
            selection=ring_selection,
        )

        dispatch_combine = make_dispatch_combine(local_idx, valid, tokens, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_dispatch = tree_checkpoint_name(dispatch_combine.dispatch(x_global), _CHECKPOINT_DISPATCH_INPUT)
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        out_global = dispatch_combine.combine(out_dispatch, weight_dispatch)
        # #2710 ring EP strategy: collect only this shard's token slice after
        # reducing contributions from experts across the EP mesh.
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=jnp.zeros_like(dropped_total), receiver_dropped=dropped_total)
