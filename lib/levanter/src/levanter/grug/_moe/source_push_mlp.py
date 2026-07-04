# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MLP-level source-push MoE boundary with an H-expert-major residual contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int

from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    SourcePushForwardHostInputs,
    source_push_forward_with_h_from_plan,
)
from levanter.grug._moe.source_push_inbox import PushInboxConfig
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushPlan,
    _source_push_out_sharding,
    build_source_push_plan,
)


SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SourcePushMlpImplementation: TypeAlias = Literal["reference", "pallas_mgpu"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushMlpRouteTable:
    """Static route table for an MLP-level source-push custom VJP.

    The table is derived from ``SourcePushPlan`` and contains only route identity
    and expert-major row placement. Differentiable route weights are read from
    the ``route_weights`` primal by ``source_rank/token_id/route_slot``.
    """

    source_rank: Int[Array, "R"]
    token_id: Int[Array, "R"]
    route_slot: Int[Array, "R"]
    destination_rank: Int[Array, "R"]
    local_expert: Int[Array, "R"]
    expert_row: Int[Array, "R"]
    ep_size: int = field(metadata={"static": True})
    experts_per_rank: int = field(metadata={"static": True})
    tokens_per_source: int = field(metadata={"static": True})
    expert_capacity: int = field(metadata={"static": True})


def build_source_push_mlp_route_table(
    route_assignments: Int[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    block_m: int,
    capacity_factor: float = 1.25,
    entries_per_dst: int | None = None,
) -> tuple[SourcePushMlpRouteTable, Int[Array, ""]]:
    """Build the source-push MLP route table and dropped-route count."""

    plan = build_source_push_plan(
        route_assignments,
        route_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
        entries_per_dst=entries_per_dst,
    )
    return source_push_mlp_route_table_from_plan(plan), plan.dropped_routes


def source_push_mlp_route_table_from_plan(
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
) -> SourcePushMlpRouteTable:
    """Convert a ``SourcePushPlan`` into differentiable MLP route indices."""

    assignment_ids = np.asarray(jax.device_get(plan.assignment_ids), dtype=np.int32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    local_experts = np.asarray(jax.device_get(plan.local_experts), dtype=np.int32)
    local_row_starts = np.asarray(jax.device_get(plan.local_row_starts), dtype=np.int32)
    if src_base_by_expert is None:
        src_base_host = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
    else:
        src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)

    source_ranks = []
    token_list = []
    slot_list = []
    destination_ranks = []
    expert_list = []
    row_list = []
    ep_size = assignment_ids.shape[0]
    experts_per_rank = src_base_host.shape[-1]
    for src in range(ep_size):
        for dst_ordinal in range(assignment_ids.shape[1]):
            dst = (src + dst_ordinal) % ep_size
            for entry in range(assignment_ids.shape[2]):
                expert = int(local_experts[src, dst_ordinal, entry])
                if expert < 0:
                    continue
                base_row = int(src_base_host[dst, src, expert]) + int(local_row_starts[src, dst_ordinal, entry])
                for row in range(assignment_ids.shape[3]):
                    if not valid_mask[src, dst_ordinal, entry, row]:
                        continue
                    source_ranks.append(src)
                    token_list.append(int(token_ids[src, dst_ordinal, entry, row]))
                    slot_list.append(int(route_slots[src, dst_ordinal, entry, row]))
                    destination_ranks.append(dst)
                    expert_list.append(expert)
                    row_list.append(base_row + row)

    expert_capacity = max(row_list) + 1 if row_list else 0
    return SourcePushMlpRouteTable(
        source_rank=jnp.asarray(source_ranks, dtype=jnp.int32),
        token_id=jnp.asarray(token_list, dtype=jnp.int32),
        route_slot=jnp.asarray(slot_list, dtype=jnp.int32),
        destination_rank=jnp.asarray(destination_ranks, dtype=jnp.int32),
        local_expert=jnp.asarray(expert_list, dtype=jnp.int32),
        expert_row=jnp.asarray(row_list, dtype=jnp.int32),
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        tokens_per_source=plan.tokens_per_source,
        expert_capacity=expert_capacity,
    )


def source_push_moe_mlp(
    x: Float[Array, "S T D"],
    route_assignments: Int[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    *,
    ep_size: int,
    experts_per_rank: int,
    block_m: int,
    capacity_factor: float = 1.25,
    entries_per_dst: int | None = None,
) -> tuple[Float[Array, "S T D"], Int[Array, ""]]:
    """Run the MLP-level source-push boundary with a custom VJP."""

    route_table, dropped_routes = build_source_push_mlp_route_table(
        route_assignments,
        route_weights,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        block_m=block_m,
        capacity_factor=capacity_factor,
        entries_per_dst=entries_per_dst,
    )
    return source_push_moe_mlp_custom_vjp(route_table, x, route_weights, w13, w2), dropped_routes


def source_push_moe_mlp_reference(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> Float[Array, "S T D"]:
    """Pure-JAX source-push MLP reference using H as the stable intermediate."""

    y, _ = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    return y


def source_push_moe_mlp_reference_with_h(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst E C twoI"]]:
    """Return ``(y, H_expert_major)`` for the source-push MLP reference."""

    h = _source_push_w13_h(route_table, x, w13)
    y = _source_push_w2_from_h_return_combine(route_table, h, route_weights, w2)
    return y, h


def source_push_moe_mlp_reference_with_h_flat(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_rows_per_rank: int,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    """Return ``(y, H_flat)`` using the production flat W13-H layout."""

    h_flat = _source_push_w13_h_flat(route_table, expert_base, h_rows_per_rank, x, w13)
    y = _source_push_w2_from_h_flat_return_combine(route_table, expert_base, h_flat, route_weights, w2)
    return y, h_flat


def source_push_moe_mlp_from_plan(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    *,
    implementation: SourcePushMlpImplementation = SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Int[Array, ""]]:
    """Run the preplanned source-push MLP with a flat-H custom VJP residual."""

    _validate_source_push_mlp_implementation(implementation)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    h_rows_per_rank = config.hidden_rows_per_rank

    @jax.custom_vjp
    def _custom_vjp(
        x_arg: Float[Array, "S T D"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "S E D twoI"],
        w2_arg: Float[Array, "S E I D"],
    ) -> Float[Array, "S T D"]:
        y, _ = _source_push_moe_mlp_from_plan_forward_value(
            config,
            host_inputs,
            route_table,
            expert_base,
            h_rows_per_rank,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=implementation,
            execution_mode=execution_mode,
            mesh=mesh,
        )
        return y

    def _fwd(
        x_arg: Float[Array, "S T D"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "S E D twoI"],
        w2_arg: Float[Array, "S E I D"],
    ):
        y, h_flat = _source_push_moe_mlp_from_plan_forward_value(
            config,
            host_inputs,
            route_table,
            expert_base,
            h_rows_per_rank,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            implementation=implementation,
            execution_mode=execution_mode,
            mesh=mesh,
        )
        return y, (x_arg, route_weights_arg, w13_arg, w2_arg, h_flat)

    def _bwd(residuals, dy):
        x_arg, route_weights_arg, w13_arg, w2_arg, h_flat = residuals
        if isinstance(dy, jax.custom_derivatives.SymbolicZero):
            return (
                jnp.zeros_like(x_arg),
                jnp.zeros_like(route_weights_arg),
                jnp.zeros_like(w13_arg),
                jnp.zeros_like(w2_arg),
            )
        dx, d_route_weights, dw13, dw2 = _source_push_moe_mlp_backward_from_h_flat(
            route_table,
            expert_base,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            h_flat,
            dy,
        )
        return dx, d_route_weights, dw13, dw2

    _custom_vjp.defvjp(_fwd, _bwd)
    return _custom_vjp(x, route_weights, w13, w2), host_inputs.plan.dropped_routes


@jax.custom_vjp
def source_push_moe_mlp_custom_vjp(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> Float[Array, "S T D"]:
    """MLP-level custom VJP whose forward residual is ``H_expert_major``."""

    return source_push_moe_mlp_reference(route_table, x, route_weights, w13, w2)


def _source_push_moe_mlp_fwd(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
):
    y, h = source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    return y, (route_table, x, route_weights, w13, w2, h)


def _source_push_moe_mlp_bwd(residuals, dy):
    route_table, x, route_weights, w13, w2, h = residuals
    if isinstance(dy, jax.custom_derivatives.SymbolicZero):
        return (
            None,
            jnp.zeros_like(x),
            jnp.zeros_like(route_weights),
            jnp.zeros_like(w13),
            jnp.zeros_like(w2),
        )
    dx, d_route_weights, dw13, dw2 = _source_push_moe_mlp_backward_from_h(
        route_table,
        x,
        route_weights,
        w13,
        w2,
        h,
        dy,
    )
    return None, dx, d_route_weights, dw13, dw2


source_push_moe_mlp_custom_vjp.defvjp(_source_push_moe_mlp_fwd, _source_push_moe_mlp_bwd)


def _validate_source_push_mlp_implementation(implementation: SourcePushMlpImplementation) -> None:
    if implementation not in (
        SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    ):
        raise ValueError(
            "source-push MLP implementation must be one of "
            f"{(SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU)}, "
            f"got {implementation!r}"
        )


def _source_push_moe_mlp_from_plan_forward_value(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_rows_per_rank: int,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    *,
    implementation: SourcePushMlpImplementation,
    execution_mode: str,
    mesh: Mesh | None,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    if implementation == SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE:
        return source_push_moe_mlp_reference_with_h_flat(
            route_table,
            expert_base,
            h_rows_per_rank,
            x,
            route_weights,
            w13,
            w2,
        )
    y, h, _ = source_push_forward_with_h_from_plan(
        config,
        host_inputs,
        x,
        route_weights,
        w13,
        w2,
        execution_mode=execution_mode,
        mesh=mesh,
    )
    return y, h


def _source_push_w13_h(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    w13: Float[Array, "S E D twoI"],
) -> Float[Array, "Dst E C twoI"]:
    x_rows = x.at[route_table.source_rank, route_table.token_id].get(
        out_sharding=_source_push_out_sharding(None, None)
    )
    w13_rows = w13.at[route_table.destination_rank, route_table.local_expert].get(
        out_sharding=_source_push_out_sharding(None, None, None)
    )
    h_rows = jnp.einsum("rd,rdo->ro", x_rows.astype(jnp.float32), w13_rows.astype(jnp.float32))
    out_shape = (route_table.ep_size, route_table.experts_per_rank, route_table.expert_capacity, w13.shape[-1])
    h = jnp.zeros(out_shape, dtype=h_rows.dtype)
    return (
        h.at[route_table.destination_rank, route_table.local_expert, route_table.expert_row]
        .set(h_rows, out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None))
        .astype(w13.dtype)
    )


def _source_push_w13_h_flat(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_rows_per_rank: int,
    x: Float[Array, "S T D"],
    w13: Float[Array, "S E D twoI"],
) -> Float[Array, "Dst rows twoI"]:
    x_rows = x.at[route_table.source_rank, route_table.token_id].get(
        out_sharding=_source_push_out_sharding(None, None)
    )
    w13_rows = w13.at[route_table.destination_rank, route_table.local_expert].get(
        out_sharding=_source_push_out_sharding(None, None, None)
    )
    h_rows = jnp.einsum("rd,rdo->ro", x_rows.astype(jnp.float32), w13_rows.astype(jnp.float32))
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    base_row = expert_base.at[route_table.destination_rank, route_table.local_expert].get(
        out_sharding=_source_push_out_sharding(None)
    )
    flat_row = base_row + route_table.expert_row
    out_shape = (route_table.ep_size, h_rows_per_rank, w13.shape[-1])
    h = jnp.zeros(out_shape, dtype=h_rows.dtype)
    return (
        h.at[route_table.destination_rank, flat_row]
        .set(h_rows, out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None))
        .astype(w13.dtype)
    )


def _source_push_w2_from_h_return_combine(
    route_table: SourcePushMlpRouteTable,
    h: Float[Array, "Dst E C twoI"],
    route_weights: Float[Array, "S T K"],
    w2: Float[Array, "S E I D"],
) -> Float[Array, "S T D"]:
    h_rows = h.at[route_table.destination_rank, route_table.local_expert, route_table.expert_row].get(
        out_sharding=_source_push_out_sharding(None, None)
    )
    h_rows = h_rows.astype(jnp.float32)
    intermediate_dim = h_rows.shape[-1] // 2
    gate = h_rows[:, :intermediate_dim]
    up = h_rows[:, intermediate_dim:]
    activation = jax.nn.silu(gate) * up
    weights = route_weights.at[route_table.source_rank, route_table.token_id, route_table.route_slot].get(
        out_sharding=_source_push_out_sharding(None)
    )
    weights = weights.astype(jnp.float32)
    w2_rows = w2.at[route_table.destination_rank, route_table.local_expert].get(
        out_sharding=_source_push_out_sharding(None, None, None)
    )
    w2_rows = w2_rows.astype(jnp.float32)
    route_y = jnp.einsum("ri,rid->rd", activation * weights[:, None], w2_rows)
    out_shape = (route_table.ep_size, route_table.tokens_per_source, w2.shape[-1])
    y = jnp.zeros(out_shape, dtype=route_y.dtype)
    return (
        y.at[route_table.source_rank, route_table.token_id]
        .add(route_y, out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None))
        .astype(w2.dtype)
    )


def _source_push_w2_from_h_flat_return_combine(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_flat: Float[Array, "Dst rows twoI"],
    route_weights: Float[Array, "S T K"],
    w2: Float[Array, "S E I D"],
) -> Float[Array, "S T D"]:
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    dst = route_table.destination_rank
    expert = route_table.local_expert
    base_row = expert_base.at[dst, expert].get(out_sharding=_source_push_out_sharding(None))
    flat_row = base_row + route_table.expert_row
    h_rows = h_flat.at[dst, flat_row].get(out_sharding=_source_push_out_sharding(None, None))
    h_rows = h_rows.astype(jnp.float32)
    intermediate_dim = h_rows.shape[-1] // 2
    gate = h_rows[:, :intermediate_dim]
    up = h_rows[:, intermediate_dim:]
    activation = jax.nn.silu(gate) * up
    weights = route_weights.at[route_table.source_rank, route_table.token_id, route_table.route_slot].get(
        out_sharding=_source_push_out_sharding(None)
    )
    weights = weights.astype(jnp.float32)
    w2_rows = w2.at[dst, expert].get(out_sharding=_source_push_out_sharding(None, None, None))
    w2_rows = w2_rows.astype(jnp.float32)
    route_y = jnp.einsum("ri,rid->rd", activation * weights[:, None], w2_rows)
    out_shape = (route_table.ep_size, route_table.tokens_per_source, w2.shape[-1])
    y = jnp.zeros(out_shape, dtype=route_y.dtype)
    return (
        y.at[route_table.source_rank, route_table.token_id]
        .add(route_y, out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None))
        .astype(w2.dtype)
    )


def _source_push_moe_mlp_backward_from_h(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h: Float[Array, "Dst E C twoI"],
    dy: Float[Array, "S T D"],
) -> tuple[
    Float[Array, "S T D"],
    Float[Array, "S T K"],
    Float[Array, "S E D twoI"],
    Float[Array, "S E I D"],
]:
    dst = route_table.destination_rank
    expert = route_table.local_expert
    row = route_table.expert_row
    h_rows = h.at[dst, expert, row].get(out_sharding=_source_push_out_sharding(None, None))
    h_rows = h_rows.astype(jnp.float32)
    return _source_push_moe_mlp_backward_from_h_rows(route_table, x, route_weights, w13, w2, h_rows, dy)


def _source_push_moe_mlp_backward_from_h_flat(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h_flat: Float[Array, "Dst rows twoI"],
    dy: Float[Array, "S T D"],
) -> tuple[
    Float[Array, "S T D"],
    Float[Array, "S T K"],
    Float[Array, "S E D twoI"],
    Float[Array, "S E I D"],
]:
    dst = route_table.destination_rank
    expert = route_table.local_expert
    base_row = expert_base.at[dst, expert].get(out_sharding=_source_push_out_sharding(None))
    flat_row = base_row + route_table.expert_row
    h_rows = h_flat.at[dst, flat_row].get(out_sharding=_source_push_out_sharding(None, None))
    h_rows = h_rows.astype(jnp.float32)
    return _source_push_moe_mlp_backward_from_h_rows(route_table, x, route_weights, w13, w2, h_rows, dy)


def _source_push_moe_mlp_backward_from_h_rows(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h_rows: Float[Array, "R twoI"],
    dy: Float[Array, "S T D"],
) -> tuple[
    Float[Array, "S T D"],
    Float[Array, "S T K"],
    Float[Array, "S E D twoI"],
    Float[Array, "S E I D"],
]:
    src = route_table.source_rank
    token = route_table.token_id
    slot = route_table.route_slot
    dst = route_table.destination_rank
    expert = route_table.local_expert
    row = route_table.expert_row

    intermediate_dim = h_rows.shape[-1] // 2
    block_shape = (route_table.ep_size, route_table.experts_per_rank, route_table.expert_capacity)
    block_sharding = _source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    scalar_block_sharding = _source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)

    h_blocks = (
        jnp.zeros((*block_shape, h_rows.shape[-1]), dtype=jnp.float32)
        .at[dst, expert, row]
        .set(h_rows.astype(jnp.float32), out_sharding=block_sharding)
    )
    gate = h_blocks[..., :intermediate_dim]
    up = h_blocks[..., intermediate_dim:]
    silu_gate = jax.nn.silu(gate)
    activation = silu_gate * up
    weights = route_weights.at[src, token, slot].get(out_sharding=_source_push_out_sharding(None))
    weights = weights.astype(jnp.float32)
    weight_blocks = (
        jnp.zeros(block_shape, dtype=jnp.float32).at[dst, expert, row].set(weights, out_sharding=scalar_block_sharding)
    )
    weighted_activation = activation * weight_blocks[..., None]

    dy_rows = dy.at[src, token].get(out_sharding=_source_push_out_sharding(None, None))
    dy_rows = dy_rows.astype(jnp.float32)
    dy_blocks = (
        jnp.zeros((*block_shape, dy.shape[-1]), dtype=jnp.float32)
        .at[dst, expert, row]
        .set(dy_rows, out_sharding=block_sharding)
    )
    w2_blocks = w2.astype(jnp.float32)
    d_weighted_activation = jnp.einsum("secd,seid->seci", dy_blocks, w2_blocks)
    d_route_blocks = jnp.sum(d_weighted_activation * activation, axis=-1)
    d_route_rows = d_route_blocks.at[dst, expert, row].get(out_sharding=_source_push_out_sharding(None))
    d_activation = d_weighted_activation * weight_blocks[..., None]

    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_gate = d_activation * up * d_silu
    d_up = d_activation * silu_gate
    d_h_blocks = jnp.concatenate([d_gate, d_up], axis=-1)

    x_rows = x.at[src, token].get(out_sharding=_source_push_out_sharding(None, None))
    x_rows = x_rows.astype(jnp.float32)
    x_blocks = (
        jnp.zeros((*block_shape, x.shape[-1]), dtype=jnp.float32)
        .at[dst, expert, row]
        .set(x_rows, out_sharding=block_sharding)
    )
    w13_blocks = w13.astype(jnp.float32)
    dx_blocks = jnp.einsum("seco,sedo->secd", d_h_blocks, w13_blocks)
    dx_rows = dx_blocks.at[dst, expert, row].get(out_sharding=_source_push_out_sharding(None, None))
    dw13 = jnp.einsum("secd,seco->sedo", x_blocks, d_h_blocks)
    dw2 = jnp.einsum("seci,secd->seid", weighted_activation, dy_blocks)

    dx = (
        jnp.zeros_like(x, dtype=jnp.float32)
        .at[src, token]
        .add(
            dx_rows,
            out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
        )
    )
    d_route_weights = (
        jnp.zeros_like(route_weights, dtype=jnp.float32)
        .at[src, token, slot]
        .add(
            d_route_rows,
            out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
        )
    )
    return (
        dx.astype(x.dtype),
        d_route_weights.astype(route_weights.dtype),
        dw13.astype(w13.dtype),
        dw2.astype(w2.dtype),
    )
