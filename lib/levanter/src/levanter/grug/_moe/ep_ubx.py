# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""NCCL UB-X expert-parallel Grug MoE backend."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_MOE_OUTPUT,
)
from levanter.grug._moe.ep_ring import (
    _BulkRingDispatchState,
    _RingRouting,
    _bulk_ring_expert_compute,
    _ring_routing_prepass,
)
from levanter.grug._moe.ep_ubx_maps import UbXRoutingMaps, build_ubx_routing_maps
from levanter.grug.sharding import _batch_axes
from levanter.kernels.ubx import combine_push3_bf16, dispatch_topk_bf16


_UBX_RANKS = 8


def _validate_ubx_contract(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
) -> None:
    if x_local.dtype != jnp.bfloat16 or moe_w13_local.dtype != jnp.bfloat16 or moe_w2_local.dtype != jnp.bfloat16:
        raise TypeError("UB-X requires bfloat16 activations and expert weights")
    if combine_weights_local.dtype != jnp.float32:
        raise TypeError("UB-X requires float32 combine weights")
    if not jnp.issubdtype(selected_experts_local.dtype, jnp.integer):
        raise TypeError("UB-X requires integer selected experts")
    if x_local.ndim != 2 or selected_experts_local.ndim != 2 or combine_weights_local.ndim != 2:
        raise ValueError("UB-X inputs must have shapes [T,H], [T,K], and [T,K]")
    if selected_experts_local.shape != combine_weights_local.shape:
        raise ValueError("UB-X selected experts and combine weights must have the same shape")
    if selected_experts_local.shape[0] != x_local.shape[0]:
        raise ValueError("UB-X routing must have one row per activation token")
    if x_local.shape[0] == 0 or x_local.shape[1] == 0 or selected_experts_local.shape[1] == 0:
        raise ValueError("UB-X token, hidden, and top-k dimensions must be non-empty")
    if x_local.shape[1] % 32:
        raise ValueError(f"UB-X hidden size must be divisible by 32, got {x_local.shape[1]}")
    if moe_w13_local.ndim != 3 or moe_w2_local.ndim != 3:
        raise ValueError("UB-X expert weights must have shapes [Elocal,H,2I] and [Elocal,I,H]")
    if moe_w13_local.shape[0] != moe_w2_local.shape[0]:
        raise ValueError("UB-X W13 and W2 must have the same local expert count")
    if moe_w13_local.shape[1] != x_local.shape[1] or moe_w2_local.shape[2] != x_local.shape[1]:
        raise ValueError("UB-X W13/W2 hidden dimensions must match the activations")
    if moe_w13_local.shape[2] != 2 * moe_w2_local.shape[1]:
        raise ValueError("UB-X W13 output must be twice the W2 intermediate dimension")
    if moe_w13_local.shape[0] <= 0:
        raise ValueError("UB-X requires at least one local expert")
    if num_experts != _UBX_RANKS * moe_w13_local.shape[0]:
        raise ValueError(
            f"UB-X requires exactly {_UBX_RANKS} expert ranks; "
            f"got num_experts={num_experts} and local_experts={moe_w13_local.shape[0]}"
        )
    if capacity_factor <= 0:
        raise ValueError(f"UB-X capacity_factor must be positive, got {capacity_factor}")


def _dense_accepted_gates(
    selected_experts_local: Int[Array, "T K"],
    combine_weights_local: Float[Array, "T K"],
    accepted_local: Bool[Array, "T K"],
    *,
    num_experts: int,
) -> Float[Array, "T E"]:
    """Convert accepted routes to the dense FP32 gate matrix required by PUSH3."""
    tokens, topk = selected_experts_local.shape
    token_rows = jnp.repeat(jnp.arange(tokens, dtype=jnp.int32), topk)
    values = jnp.where(accepted_local, combine_weights_local, 0).astype(jnp.float32)
    return (
        jnp.zeros((tokens, num_experts), dtype=jnp.float32)
        .at[
            token_rows,
            selected_experts_local.reshape(-1),
        ]
        .add(values.reshape(-1))
    )


def _accepted_unit_gates(
    topk_idx_local: Int[Array, "T K"],
    *,
    num_experts: int,
) -> Float[Array, "T E"]:
    """Return one for each accepted source route and zero elsewhere."""
    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
    return jnp.any(topk_idx_local[:, :, None] == expert_ids[None, None, :], axis=1).astype(jnp.float32)


def _sorted_topk_values(
    values_local: Float[Array, "T K"],
    selected_experts_local: Int[Array, "T K"],
    topk_idx_local: Int[Array, "T K"],
) -> Float[Array, "T K"]:
    """Align values in router order to UB-X's accepted, expert-sorted top-k order."""
    matches = topk_idx_local[:, :, None] == selected_experts_local[:, None, :]
    return jnp.sum(jnp.where(matches, values_local[:, None, :], 0), axis=2)


def _slot_values_from_inverse_map(
    sorted_values_global: Float[Array, "T K"],
    inverse_map_local: Int[Array, "C 4"],
    *,
    tokens_per_rank: int,
) -> Float[Array, "C"]:
    """Gather source-local sorted route values into destination slot order."""
    source_token = inverse_map_local[:, 0] * tokens_per_rank + inverse_map_local[:, 1]
    source_topk = inverse_map_local[:, 2]
    values = sorted_values_global[source_token, source_topk]
    return jnp.where(inverse_map_local[:, 3].astype(jnp.bool_), values, 0)


def _scatter_slot_values_to_global_topk(
    slot_values_local: Float[Array, "C"],
    inverse_map_local: Int[Array, "C 4"],
    *,
    tokens_per_rank: int,
    expert_axis_size: int,
    topk: int,
) -> Float[Array, "Tglobal K"]:
    """Scatter destination slots into source token and sorted-top-k positions."""
    source_token = inverse_map_local[:, 0] * tokens_per_rank + inverse_map_local[:, 1]
    source_topk = inverse_map_local[:, 2]
    valid = inverse_map_local[:, 3].astype(jnp.bool_)
    safe_source_token = jnp.where(valid, source_token, tokens_per_rank * expert_axis_size)
    values = jnp.where(valid, slot_values_local, 0)
    return (
        jnp.zeros((tokens_per_rank * expert_axis_size + 1, topk), dtype=slot_values_local.dtype)
        .at[safe_source_token, source_topk]
        .add(values)[: tokens_per_rank * expert_axis_size]
    )


def _unsort_topk_values(
    sorted_values_local: Float[Array, "T K"],
    topk_idx_local: Int[Array, "T K"],
    selected_experts_local: Int[Array, "T K"],
    accepted_local: Bool[Array, "T K"],
) -> Float[Array, "T K"]:
    """Restore expert-sorted accepted route values to the router's top-k order."""
    matches = selected_experts_local[:, :, None] == topk_idx_local[:, None, :]
    values = jnp.sum(jnp.where(matches, sorted_values_local[:, None, :], 0), axis=2)
    return jnp.where(accepted_local, values, 0)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _ubx_dispatch(
    x_local: Float[Array, "T H"],
    dispatch_topk_expert: Int[Array, "T K"],
    dispatch_topk_slot: Int[Array, "T K"],
    dispatch_valid: Bool[Array, "C"],
    assignment_indices: Int[Array, "C"],
    assignment_valid: Bool[Array, "C"],
    topk: int,
    tokens_per_rank: int,
) -> Float[Array, "C H"]:
    del assignment_indices, assignment_valid, topk, tokens_per_rank
    return dispatch_topk_bf16(x_local, dispatch_topk_expert, dispatch_topk_slot, dispatch_valid)


def _ubx_dispatch_fwd(
    x_local: jax.Array,
    dispatch_topk_expert: jax.Array,
    dispatch_topk_slot: jax.Array,
    dispatch_valid: jax.Array,
    assignment_indices: jax.Array,
    assignment_valid: jax.Array,
    topk: int,
    tokens_per_rank: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    del topk, tokens_per_rank
    output = dispatch_topk_bf16(x_local, dispatch_topk_expert, dispatch_topk_slot, dispatch_valid)
    return output, (assignment_indices, assignment_valid)


def _ubx_dispatch_bwd(
    topk: int,
    tokens_per_rank: int,
    residuals: tuple[jax.Array, jax.Array],
    output_cotangent: jax.Array,
) -> tuple[jax.Array, None, None, None, None, None]:
    assignment_indices, assignment_valid = residuals
    token_global = jnp.floor_divide(assignment_indices, topk)

    def ring_dispatch(value: jax.Array) -> jax.Array:
        value_global = jax.lax.all_gather(value, "expert", tiled=True)
        value_take = jnp.take(value_global, token_global, axis=0)
        return jnp.where(assignment_valid[:, None], value_take, jnp.zeros_like(value_take))

    _, pullback = jax.vjp(
        ring_dispatch,
        jnp.zeros((tokens_per_rank, output_cotangent.shape[1]), dtype=output_cotangent.dtype),
    )
    (x_cotangent,) = pullback(output_cotangent)
    return x_cotangent, None, None, None, None, None


_ubx_dispatch.defvjp(_ubx_dispatch_fwd, _ubx_dispatch_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(9,))
def _ubx_combine(
    expert_outputs_local: Float[Array, "C H"],
    combine_weights_local: Float[Array, "T K"],
    selected_experts_local: Int[Array, "T K"],
    accepted_local: Bool[Array, "T K"],
    dispatch_topk_expert: Int[Array, "T K"],
    dispatch_topk_slot: Int[Array, "T K"],
    dispatch_valid: Bool[Array, "C"],
    inverse_map: Int[Array, "C 4"],
    topk_idx: Int[Array, "T K"],
    num_experts: int,
) -> Float[Array, "T H"]:
    dense_gates = _dense_accepted_gates(
        selected_experts_local,
        combine_weights_local,
        accepted_local,
        num_experts=num_experts,
    )
    return combine_push3_bf16(expert_outputs_local, inverse_map, topk_idx, dense_gates)


def _ubx_combine_fwd(
    expert_outputs_local: jax.Array,
    combine_weights_local: jax.Array,
    selected_experts_local: jax.Array,
    accepted_local: jax.Array,
    dispatch_topk_expert: jax.Array,
    dispatch_topk_slot: jax.Array,
    dispatch_valid: jax.Array,
    inverse_map: jax.Array,
    topk_idx: jax.Array,
    num_experts: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    dense_gates = _dense_accepted_gates(
        selected_experts_local,
        combine_weights_local,
        accepted_local,
        num_experts=num_experts,
    )
    output = combine_push3_bf16(expert_outputs_local, inverse_map, topk_idx, dense_gates)
    residuals = (
        expert_outputs_local,
        combine_weights_local,
        selected_experts_local,
        accepted_local,
        dispatch_topk_expert,
        dispatch_topk_slot,
        dispatch_valid,
        inverse_map,
        topk_idx,
    )
    return output, residuals


def _ubx_combine_bwd(
    num_experts: int,
    residuals: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None, None, None, None, None]:
    del num_experts
    (
        expert_outputs_local,
        combine_weights_local,
        selected_experts_local,
        accepted_local,
        dispatch_topk_expert,
        dispatch_topk_slot,
        dispatch_valid,
        inverse_map,
        topk_idx,
    ) = residuals
    tokens_per_rank, topk = selected_experts_local.shape
    expert_axis_size = jax.sharding.get_abstract_mesh().shape["expert"]

    output_cotangent_dispatch = dispatch_topk_bf16(
        output_cotangent.astype(jnp.bfloat16),
        dispatch_topk_expert,
        dispatch_topk_slot,
        dispatch_valid,
    )
    sorted_weights_local = _sorted_topk_values(
        combine_weights_local,
        selected_experts_local,
        topk_idx,
    )
    sorted_weights_global = jax.lax.all_gather(sorted_weights_local, "expert", tiled=True)
    slot_weights = _slot_values_from_inverse_map(
        sorted_weights_global,
        inverse_map,
        tokens_per_rank=tokens_per_rank,
    )
    expert_outputs_cotangent = (output_cotangent_dispatch.astype(jnp.float32) * slot_weights[:, None]).astype(
        expert_outputs_local.dtype
    )

    slot_weight_cotangents = jnp.sum(
        output_cotangent_dispatch.astype(jnp.float32) * expert_outputs_local.astype(jnp.float32),
        axis=1,
        dtype=jnp.float32,
    )
    sorted_weight_cotangents_global = _scatter_slot_values_to_global_topk(
        slot_weight_cotangents,
        inverse_map,
        tokens_per_rank=tokens_per_rank,
        expert_axis_size=expert_axis_size,
        topk=topk,
    )
    sorted_weight_cotangents_local = jax.lax.psum_scatter(
        sorted_weight_cotangents_global,
        "expert",
        scatter_dimension=0,
        tiled=True,
    )
    combine_weights_cotangent = _unsort_topk_values(
        sorted_weight_cotangents_local,
        topk_idx,
        selected_experts_local,
        accepted_local,
    ).astype(combine_weights_local.dtype)
    return (
        expert_outputs_cotangent,
        combine_weights_cotangent,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_ubx_combine.defvjp(_ubx_combine_fwd, _ubx_combine_bwd)


def _routing_maps(
    selected_experts_local: Int[Array, "Tlocal K"],
    *,
    local_experts: int,
    num_experts: int,
    capacity_factor: float,
) -> tuple[UbXRoutingMaps, _RingRouting]:
    routing = _ring_routing_prepass(
        selected_experts_local,
        local_experts=local_experts,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
    accepted_indices_by_rank = jax.lax.all_gather(routing.assignment_indices, "expert")
    accepted_valid_by_rank = jax.lax.all_gather(routing.valid, "expert")
    maps = build_ubx_routing_maps(
        selected_experts_global,
        accepted_indices_by_rank,
        accepted_valid_by_rank,
        rank=jax.lax.axis_index("expert"),
        local_experts=local_experts,
    )
    return maps, routing


def _moe_mlp_ep_ubx_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run exact Ring routing with NCCL UB-X dispatch and combine transport."""
    _validate_ubx_contract(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    maps, routing = _routing_maps(
        selected_experts_local,
        local_experts=moe_w13_local.shape[0],
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    x_dispatch = _ubx_dispatch(
        x_local,
        maps.dispatch_topk_expert,
        maps.dispatch_topk_slot,
        maps.dispatch_valid,
        routing.assignment_indices,
        routing.valid,
        routing.topk,
        routing.tokens_per_shard,
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    dispatch = _BulkRingDispatchState(
        x_dispatch=x_dispatch,
        weight_dispatch=jnp.zeros((x_dispatch.shape[0],), dtype=jnp.float32),
        token_global=jnp.zeros((x_dispatch.shape[0],), dtype=jnp.int32),
        group_sizes=maps.group_sizes,
        dropped_local=routing.dropped_local[None],
    )
    expert = _bulk_ring_expert_compute(
        dispatch,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
    )
    output = _ubx_combine(
        expert.out_dispatch,
        combine_weights_local,
        selected_experts_local,
        maps.accepted_local,
        maps.dispatch_topk_expert,
        maps.dispatch_topk_slot,
        maps.dispatch_valid,
        maps.inverse_map,
        maps.topk_idx,
        num_experts,
    )
    output = tree_checkpoint_name(output, _CHECKPOINT_MOE_OUTPUT)
    dropped_total = jax.lax.psum(routing.dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return output, dropped_total
