# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MLP-level source-push MoE boundary with an H-expert-major residual contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
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
SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED = "blackwell_staged"
_SOURCE_PUSH_MLP_IMPLEMENTATIONS = (
    SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED,
)
SourcePushMlpImplementation: TypeAlias = Literal["reference", "pallas_mgpu", "blackwell_staged"]


class SourcePushMlpReferenceResidual(NamedTuple):
    """Custom-VJP residual for the compact expert-major H reference path.

    The stable checkpoint is ``h``. The source-major primals are carried only
    because JAX custom VJP backward rules do not receive differentiable primals
    separately; this residual intentionally excludes packed/recv x, post-SwiGLU
    activation, per-route W2 outputs, and kernel scratch buffers.
    """

    route_table: "SourcePushMlpRouteTable"
    x: Float[Array, "S T D"]
    route_weights: Float[Array, "S T K"]
    w13: Float[Array, "S E D twoI"]
    w2: Float[Array, "S E I D"]
    h: Float[Array, "Dst E C twoI"]


class SourcePushMlpFlatResidual(NamedTuple):
    """Custom-VJP residual for the production flat-H source-push path."""

    x: Float[Array, "S T D"]
    route_weights: Float[Array, "S T K"]
    w13: Float[Array, "S E D twoI"]
    w2: Float[Array, "S E I D"]
    h_flat: Float[Array, "Dst rows twoI"]


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
    source_rank_by_expert: Int[Array, "Dst E C"]
    token_id_by_expert: Int[Array, "Dst E C"]
    route_slot_by_expert: Int[Array, "Dst E C"]
    valid_by_expert: Bool[Array, "Dst E C"]
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
    """Build the source-push MLP route table and dropped-route count.

    Route weights are differentiable MLP inputs and must not be captured while
    building static placement metadata. The plan only needs their shape.
    """

    if route_assignments.shape != route_weights.shape:
        raise ValueError(
            f"route_assignments shape {route_assignments.shape} must match route_weights {route_weights.shape}"
        )

    plan = build_source_push_plan(
        route_assignments,
        np.zeros(route_assignments.shape, dtype=np.float32),
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
    source_rank_by_expert = np.full((ep_size, experts_per_rank, expert_capacity), -1, dtype=np.int32)
    token_id_by_expert = np.full_like(source_rank_by_expert, -1)
    route_slot_by_expert = np.full_like(source_rank_by_expert, -1)
    valid_by_expert = np.zeros_like(source_rank_by_expert, dtype=np.bool_)
    for route in range(len(source_ranks)):
        dst = destination_ranks[route]
        expert = expert_list[route]
        row = row_list[route]
        source_rank_by_expert[dst, expert, row] = source_ranks[route]
        token_id_by_expert[dst, expert, row] = token_list[route]
        route_slot_by_expert[dst, expert, row] = slot_list[route]
        valid_by_expert[dst, expert, row] = True

    return SourcePushMlpRouteTable(
        source_rank=jnp.asarray(source_ranks, dtype=jnp.int32),
        token_id=jnp.asarray(token_list, dtype=jnp.int32),
        route_slot=jnp.asarray(slot_list, dtype=jnp.int32),
        destination_rank=jnp.asarray(destination_ranks, dtype=jnp.int32),
        local_expert=jnp.asarray(expert_list, dtype=jnp.int32),
        expert_row=jnp.asarray(row_list, dtype=jnp.int32),
        source_rank_by_expert=jnp.asarray(source_rank_by_expert, dtype=jnp.int32),
        token_id_by_expert=jnp.asarray(token_id_by_expert, dtype=jnp.int32),
        route_slot_by_expert=jnp.asarray(route_slot_by_expert, dtype=jnp.int32),
        valid_by_expert=jnp.asarray(valid_by_expert, dtype=jnp.bool_),
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

    with jax.ensure_compile_time_eval():
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
        return y, SourcePushMlpFlatResidual(x_arg, route_weights_arg, w13_arg, w2_arg, h_flat)

    def _bwd(residuals: SourcePushMlpFlatResidual, dy):
        if isinstance(dy, jax.custom_derivatives.SymbolicZero):
            return (
                jnp.zeros_like(residuals.x),
                jnp.zeros_like(residuals.route_weights),
                jnp.zeros_like(residuals.w13),
                jnp.zeros_like(residuals.w2),
            )
        dx, d_route_weights, dw13, dw2 = _source_push_moe_mlp_backward_from_h_flat(
            route_table,
            expert_base,
            residuals.x,
            residuals.route_weights,
            residuals.w13,
            residuals.w2,
            residuals.h_flat,
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
    return y, SourcePushMlpReferenceResidual(route_table, x, route_weights, w13, w2, h)


def _source_push_moe_mlp_bwd(residuals: SourcePushMlpReferenceResidual, dy):
    if isinstance(dy, jax.custom_derivatives.SymbolicZero):
        return (
            None,
            jnp.zeros_like(residuals.x),
            jnp.zeros_like(residuals.route_weights),
            jnp.zeros_like(residuals.w13),
            jnp.zeros_like(residuals.w2),
        )
    dx, d_route_weights, dw13, dw2 = _source_push_moe_mlp_backward_from_h(
        residuals.route_table,
        residuals.x,
        residuals.route_weights,
        residuals.w13,
        residuals.w2,
        residuals.h,
        dy,
    )
    return None, dx, d_route_weights, dw13, dw2


source_push_moe_mlp_custom_vjp.defvjp(_source_push_moe_mlp_fwd, _source_push_moe_mlp_bwd)


def _validate_source_push_mlp_implementation(implementation: SourcePushMlpImplementation) -> None:
    if implementation not in (
        SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
        SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED,
    ):
        raise ValueError(
            "source-push MLP implementation must be one of "
            f"{_SOURCE_PUSH_MLP_IMPLEMENTATIONS}, "
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
        implementation=(
            SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED
            if implementation == SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED
            else SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU
        ),
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
    return _source_push_moe_mlp_backward_by_expert(
        route_table,
        x,
        route_weights,
        w13,
        w2,
        dy,
        lambda expert: h[:, expert],
    )


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
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)

    def h_for_expert(expert: int) -> Float[Array, "Dst C twoI"]:
        return _source_push_mlp_h_flat_for_expert(route_table, expert_base, h_flat, expert)

    return _source_push_moe_mlp_backward_by_expert(route_table, x, route_weights, w13, w2, dy, h_for_expert)


def _source_push_mlp_h_flat_for_expert(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_flat: Float[Array, "Dst rows twoI"],
    expert: int,
) -> Float[Array, "Dst C twoI"]:
    dst_index = jnp.arange(route_table.ep_size, dtype=jnp.int32)[:, None]
    row_offsets = jnp.arange(route_table.expert_capacity, dtype=jnp.int32)[None, :]
    base_row = expert_base[:, expert][:, None]
    flat_row = base_row + row_offsets
    return h_flat.at[dst_index, flat_row].get(
        mode="fill",
        fill_value=0,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def _source_push_mlp_expert_route_indices(
    route_table: SourcePushMlpRouteTable,
    expert: int,
) -> tuple[
    Bool[Array, "Dst C"],
    Int[Array, "Dst C"],
    Int[Array, "Dst C"],
    Int[Array, "Dst C"],
    Float[Array, "Dst C"],
]:
    valid = route_table.valid_by_expert[:, expert]
    safe_src = jnp.maximum(route_table.source_rank_by_expert[:, expert], 0)
    safe_token = jnp.maximum(route_table.token_id_by_expert[:, expert], 0)
    safe_slot = jnp.maximum(route_table.route_slot_by_expert[:, expert], 0)
    return valid, safe_src, safe_token, safe_slot, valid.astype(jnp.float32)


def _source_push_mlp_activation_from_h(
    h_block: Float[Array, "Dst C twoI"],
) -> tuple[
    Float[Array, "Dst C I"],
    Float[Array, "Dst C I"],
    Float[Array, "Dst C I"],
    Float[Array, "Dst C I"],
]:
    intermediate_dim = h_block.shape[-1] // 2
    gate = h_block[..., :intermediate_dim]
    up = h_block[..., intermediate_dim:]
    silu_gate = jax.nn.silu(gate)
    return gate, up, silu_gate, silu_gate * up


def _source_push_mlp_route_weights_to_expert_major(
    route_weights: Float[Array, "S T K"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    safe_slot: Int[Array, "Dst C"],
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C"]:
    weights = route_weights.at[safe_src, safe_token, safe_slot].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None)
    )
    return weights.astype(jnp.float32) * valid_f


def _source_push_mlp_dy_to_expert_major(
    dy: Float[Array, "S T D"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C D"]:
    dy_block = dy.at[safe_src, safe_token].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    )
    return dy_block.astype(jnp.float32) * valid_f[..., None]


def _source_push_mlp_weight_activation(
    activation: Float[Array, "Dst C I"],
    weights: Float[Array, "Dst C"],
) -> Float[Array, "Dst C I"]:
    return activation * weights[..., None]


def _source_push_mlp_w2_backward_for_expert(
    dy_block: Float[Array, "Dst C D"],
    activation: Float[Array, "Dst C I"],
    weighted_activation: Float[Array, "Dst C I"],
    w2_block: Float[Array, "Dst I D"],
    valid_f: Float[Array, "Dst C"],
) -> tuple[
    Float[Array, "Dst C I"],
    Float[Array, "Dst C"],
    Float[Array, "Dst I D"],
]:
    d_weighted_activation = jnp.einsum("scd,sid->sci", dy_block, w2_block)
    d_route_block = jnp.sum(d_weighted_activation * activation, axis=-1) * valid_f
    dw2_block = jnp.einsum("sci,scd->sid", weighted_activation, dy_block)
    return d_weighted_activation, d_route_block, dw2_block


def _source_push_mlp_swiglu_backward_from_h(
    d_weighted_activation: Float[Array, "Dst C I"],
    weights: Float[Array, "Dst C"],
    gate: Float[Array, "Dst C I"],
    up: Float[Array, "Dst C I"],
    silu_gate: Float[Array, "Dst C I"],
) -> Float[Array, "Dst C twoI"]:
    d_activation = d_weighted_activation * weights[..., None]
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_gate = d_activation * up * d_silu
    d_up = d_activation * silu_gate
    return jnp.concatenate([d_gate, d_up], axis=-1)


def _source_push_mlp_x_to_expert_major(
    x: Float[Array, "S T D"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C D"]:
    x_block = x.at[safe_src, safe_token].get(out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None))
    return x_block.astype(jnp.float32) * valid_f[..., None]


def _source_push_mlp_w13_backward_for_expert(
    x_block: Float[Array, "Dst C D"],
    d_h_block: Float[Array, "Dst C twoI"],
    w13_block: Float[Array, "Dst D twoI"],
) -> tuple[
    Float[Array, "Dst C D"],
    Float[Array, "Dst D twoI"],
]:
    dx_block = jnp.einsum("sco,sdo->scd", d_h_block, w13_block)
    dw13_block = jnp.einsum("scd,sco->sdo", x_block, d_h_block)
    return dx_block, dw13_block


def _source_push_mlp_accumulate_expert_backward_outputs(
    dx: Float[Array, "S T D"],
    d_route_weights: Float[Array, "S T K"],
    dw13: Float[Array, "S E D twoI"],
    dw2: Float[Array, "S E I D"],
    expert: int,
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    safe_slot: Int[Array, "Dst C"],
    dx_block: Float[Array, "Dst C D"],
    d_route_block: Float[Array, "Dst C"],
    dw13_block: Float[Array, "Dst D twoI"],
    dw2_block: Float[Array, "Dst I D"],
) -> tuple[
    Float[Array, "S T D"],
    Float[Array, "S T K"],
    Float[Array, "S E D twoI"],
    Float[Array, "S E I D"],
]:
    dx = dx.at[safe_src, safe_token].add(
        dx_block,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    d_route_weights = d_route_weights.at[safe_src, safe_token, safe_slot].add(
        d_route_block,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    dw13 = dw13.at[:, expert].set(dw13_block)
    dw2 = dw2.at[:, expert].set(dw2_block)
    return dx, d_route_weights, dw13, dw2


def _source_push_moe_mlp_backward_by_expert(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    dy: Float[Array, "S T D"],
    h_for_expert: Callable[[int], Float[Array, "Dst C twoI"]],
) -> tuple[
    Float[Array, "S T D"],
    Float[Array, "S T K"],
    Float[Array, "S E D twoI"],
    Float[Array, "S E I D"],
]:
    dx = jnp.zeros_like(x, dtype=jnp.float32)
    d_route_weights = jnp.zeros_like(route_weights, dtype=jnp.float32)
    dw13 = jnp.zeros_like(w13, dtype=jnp.float32)
    dw2 = jnp.zeros_like(w2, dtype=jnp.float32)

    for expert in range(route_table.experts_per_rank):
        _, safe_src, safe_token, safe_slot, valid_f = _source_push_mlp_expert_route_indices(route_table, expert)

        h_block = h_for_expert(expert).astype(jnp.float32) * valid_f[..., None]
        gate, up, silu_gate, activation = _source_push_mlp_activation_from_h(h_block)
        weights = _source_push_mlp_route_weights_to_expert_major(
            route_weights,
            safe_src,
            safe_token,
            safe_slot,
            valid_f,
        )
        dy_block = _source_push_mlp_dy_to_expert_major(dy, safe_src, safe_token, valid_f)
        w2_block = w2[:, expert].astype(jnp.float32)
        weighted_activation = _source_push_mlp_weight_activation(activation, weights)
        d_weighted_activation, d_route_block, dw2_block = _source_push_mlp_w2_backward_for_expert(
            dy_block,
            activation,
            weighted_activation,
            w2_block,
            valid_f,
        )
        d_h_block = _source_push_mlp_swiglu_backward_from_h(
            d_weighted_activation,
            weights,
            gate,
            up,
            silu_gate,
        )

        x_block = _source_push_mlp_x_to_expert_major(x, safe_src, safe_token, valid_f)
        w13_block = w13[:, expert].astype(jnp.float32)
        dx_block, dw13_block = _source_push_mlp_w13_backward_for_expert(x_block, d_h_block, w13_block)
        dx, d_route_weights, dw13, dw2 = _source_push_mlp_accumulate_expert_backward_outputs(
            dx,
            d_route_weights,
            dw13,
            dw2,
            expert,
            safe_src,
            safe_token,
            safe_slot,
            dx_block,
            d_route_block,
            dw13_block,
            dw2_block,
        )

    return (
        dx.astype(x.dtype),
        d_route_weights.astype(route_weights.dtype),
        dw13.astype(w13.dtype),
        dw2.astype(w2.dtype),
    )
