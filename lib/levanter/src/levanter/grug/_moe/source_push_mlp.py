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

from levanter.grug._moe.source_push_backward_dy_route import (
    SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    _source_push_backward_dy_to_expert_major,
    _source_push_backward_dy_to_h_rows,
)
from levanter.grug._moe.source_push_backward_return import (
    SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX,
    SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
    SourcePushBackwardFlatRouteIndices,
    source_push_backward_return,
    source_push_backward_return_flat,
)
from levanter.grug._moe.source_push_backward_w13 import (
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED,
    source_push_w13_backward,
    source_push_w13_backward_expert_blocks_reference,
    source_push_w13_backward_expert_blocks_tiled_reference,
)
from levanter.grug._moe.source_push_backward_w2 import (
    SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
    _source_push_w2_backward_expert_blocks,
    _source_push_w2_backward_from_flat_h,
)
from levanter.grug._moe.source_push_combine import _make_route_inverse
from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_MODES,
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    SourcePushForwardHostInputs,
    make_source_push_forward_plan_inputs,
    source_push_forward_with_compact_h_from_plan,
)
from levanter.grug._moe.source_push_inbox import (
    PushInboxConfig,
    _compact_h_expert_capacity_from_metadata,
    _recv_meta_from_send_meta,
    _source_padded_plan_send_meta,
)
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SOURCE_PUSH_META_FIELDS,
    SOURCE_PUSH_META_VALID_ROWS,
    SourcePushPlan,
    _source_push_out_sharding,
    build_source_push_plan,
    source_push_source_padded_row_bases,
)


SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SourcePushMlpImplementation: TypeAlias = Literal["reference", "pallas_mgpu"]
_STATIC_ROUTE_PLAN_ERROR = (
    "source_push_moe_mlp builds SourcePushPlan on the host, so route_assignments must be concrete/static "
    "at Python plan-build time. For jitted execution, build SourcePushPlan/SourcePushMlpRouteTable outside "
    "the differentiable MLP boundary and call source_push_moe_mlp_from_plan(...)."
)


class SourcePushMlpReferenceResidual(NamedTuple):
    """Custom-VJP residual for the compact expert-major H reference path.

    The stable checkpoint is ``h``. Route weights are kept in expert-major
    order to match ``h``; the remaining primals are carried only because JAX
    custom VJP backward rules do not receive differentiable primals separately.
    This residual intentionally excludes packed/recv x, post-SwiGLU activation,
    per-route W2 outputs, and kernel scratch buffers.
    """

    route_table: "SourcePushMlpRouteTable"
    x: Float[Array, "S T D"]
    expert_route_weights: Float[Array, "Dst E C"]
    w13: Float[Array, "S E D twoI"]
    w2: Float[Array, "S E I D"]
    h: Float[Array, "Dst E C twoI"]


class SourcePushMlpPlanResidual(NamedTuple):
    """Custom-VJP residual for the preplanned source-push path.

    The staged Pallas forward writes W13 preactivation directly into compact
    expert-major H. This keeps backward organized around ``[Dst, E, C, 2I]``
    and avoids treating source-padded flat rows as the saved contract.
    """

    x: Float[Array, "S T D"]
    expert_route_weights: Float[Array, "Dst E C"]
    w13: Float[Array, "S E D twoI"]
    w2: Float[Array, "S E I D"]
    h: Float[Array, "Dst E C twoI"]


class _SourcePushMlpRouteTableHostData(NamedTuple):
    source_rank: np.ndarray
    token_id: np.ndarray
    route_slot: np.ndarray
    destination_rank: np.ndarray
    local_expert: np.ndarray
    expert_row: np.ndarray
    source_rank_by_expert: np.ndarray
    token_id_by_expert: np.ndarray
    route_slot_by_expert: np.ndarray
    valid_by_expert: np.ndarray
    ep_size: int
    experts_per_rank: int
    tokens_per_source: int
    topk: int
    expert_capacity: int


class _SourcePushMlpExpertRouteIndices(NamedTuple):
    valid: Bool[Array, "Dst C"]
    safe_src: Int[Array, "Dst C"]
    safe_token: Int[Array, "Dst C"]
    safe_slot: Int[Array, "Dst C"]
    valid_f: Float[Array, "Dst C"]


class _SourcePushMlpExpertBackwardInputs(NamedTuple):
    route_indices: _SourcePushMlpExpertRouteIndices
    h_block: Float[Array, "Dst C twoI"]
    weights: Float[Array, "Dst C"]
    dy_block: Float[Array, "Dst C D"]
    w2_block: Float[Array, "Dst I D"]
    w13_block: Float[Array, "Dst D twoI"]


class _SourcePushMlpExpertBackwardOutput(NamedTuple):
    route_indices: _SourcePushMlpExpertRouteIndices
    dx_block: Float[Array, "Dst C D"]
    d_route_block: Float[Array, "Dst C"]
    dw13_block: Float[Array, "Dst D twoI"]
    dw2_block: Float[Array, "Dst I D"]


class _SourcePushMlpW2SwiGLUBackwardOutput(NamedTuple):
    d_h_block: Float[Array, "Dst C twoI"]
    d_route_block: Float[Array, "Dst C"]
    dw2_block: Float[Array, "Dst I D"]


class _SourcePushMlpXW13BackwardOutput(NamedTuple):
    dx_block: Float[Array, "Dst C D"]
    dw13_block: Float[Array, "Dst D twoI"]


class _SourcePushMlpSourceGradientOutput(NamedTuple):
    dx: Float[Array, "S T D"]
    d_route_weights: Float[Array, "S T K"]


class _SourcePushMlpGradients(NamedTuple):
    dx: Float[Array, "S T D"]
    d_route_weights: Float[Array, "S T K"]
    dw13: Float[Array, "S E D twoI"]
    dw2: Float[Array, "S E I D"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushMlpRouteTable:
    """Static route table for an MLP-level source-push custom VJP.

    The table is derived from ``SourcePushPlan`` and contains only route identity
    and expert-major row placement. Differentiable route weights are carried as
    a separate expert-major residual aligned with these rows.
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
    topk: int = field(metadata={"static": True})
    expert_capacity: int = field(metadata={"static": True})


def build_source_push_mlp_route_table(
    route_assignments: Int[Array, "S T K"],
    *,
    ep_size: int,
    experts_per_rank: int,
    block_m: int,
    capacity_factor: float = 1.25,
    entries_per_dst: int | None = None,
) -> tuple[SourcePushMlpRouteTable, Int[Array, ""]]:
    """Build the source-push MLP route table and dropped-route count.

    Route weights are differentiable MLP inputs and must not be captured while
    building static placement metadata. The plan builder receives all-zero
    placeholders so the route table depends only on nondifferentiable
    assignments and static capacity metadata.
    """

    try:
        plan = build_source_push_plan(
            route_assignments,
            np.zeros(route_assignments.shape, dtype=np.float32),
            ep_size=ep_size,
            experts_per_rank=experts_per_rank,
            block_m=block_m,
            capacity_factor=capacity_factor,
            entries_per_dst=entries_per_dst,
        )
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError) as exc:
        raise ValueError(_STATIC_ROUTE_PLAN_ERROR) from exc
    return source_push_mlp_route_table_from_plan(plan), plan.dropped_routes


def source_push_mlp_route_table_from_plan(
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
) -> SourcePushMlpRouteTable:
    """Convert a ``SourcePushPlan`` into differentiable MLP route indices."""

    host_data = _source_push_mlp_route_table_host_data_from_plan(plan, src_base_by_expert=src_base_by_expert)
    return SourcePushMlpRouteTable(
        source_rank=jnp.asarray(host_data.source_rank, dtype=jnp.int32),
        token_id=jnp.asarray(host_data.token_id, dtype=jnp.int32),
        route_slot=jnp.asarray(host_data.route_slot, dtype=jnp.int32),
        destination_rank=jnp.asarray(host_data.destination_rank, dtype=jnp.int32),
        local_expert=jnp.asarray(host_data.local_expert, dtype=jnp.int32),
        expert_row=jnp.asarray(host_data.expert_row, dtype=jnp.int32),
        source_rank_by_expert=jnp.asarray(host_data.source_rank_by_expert, dtype=jnp.int32),
        token_id_by_expert=jnp.asarray(host_data.token_id_by_expert, dtype=jnp.int32),
        route_slot_by_expert=jnp.asarray(host_data.route_slot_by_expert, dtype=jnp.int32),
        valid_by_expert=jnp.asarray(host_data.valid_by_expert, dtype=jnp.bool_),
        ep_size=host_data.ep_size,
        experts_per_rank=host_data.experts_per_rank,
        tokens_per_source=host_data.tokens_per_source,
        topk=host_data.topk,
        expert_capacity=host_data.expert_capacity,
    )


def _source_push_mlp_route_table_host_data_from_plan(
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
) -> _SourcePushMlpRouteTableHostData:
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

    return _SourcePushMlpRouteTableHostData(
        source_rank=np.asarray(source_ranks, dtype=np.int32),
        token_id=np.asarray(token_list, dtype=np.int32),
        route_slot=np.asarray(slot_list, dtype=np.int32),
        destination_rank=np.asarray(destination_ranks, dtype=np.int32),
        local_expert=np.asarray(expert_list, dtype=np.int32),
        expert_row=np.asarray(row_list, dtype=np.int32),
        source_rank_by_expert=source_rank_by_expert,
        token_id_by_expert=token_id_by_expert,
        route_slot_by_expert=route_slot_by_expert,
        valid_by_expert=valid_by_expert,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
        tokens_per_source=plan.tokens_per_source,
        topk=plan.topk,
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
    implementation: SourcePushMlpImplementation = SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    config: PushInboxConfig | None = None,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Int[Array, ""]]:
    """Run the MLP-level source-push boundary with a custom VJP.

    This convenience wrapper builds ``SourcePushPlan`` on the host, so
    ``route_assignments`` must be concrete/static at Python plan-build time.
    Jitted callers with prebuilt metadata should call
    ``source_push_moe_mlp_from_plan``. The default implementation is the compact
    JAX reference path; staged Pallas execution requires an explicit
    ``PushInboxConfig`` and delegates to ``source_push_moe_mlp_from_plan``.
    """

    _validate_source_push_mlp_implementation(implementation)
    _validate_source_push_mlp_execution_mode(execution_mode)
    if route_assignments.shape != route_weights.shape:
        raise ValueError(
            f"route_assignments shape {route_assignments.shape} must match route_weights {route_weights.shape}"
        )

    if config is not None:
        _validate_source_push_mlp_convenience_config(
            config,
            ep_size=ep_size,
            experts_per_rank=experts_per_rank,
            block_m=block_m,
            capacity_factor=capacity_factor,
            entries_per_dst=entries_per_dst,
        )
        with jax.ensure_compile_time_eval():
            try:
                host_inputs = make_source_push_forward_plan_inputs(config, route_assignments)
            except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError) as exc:
                raise ValueError(_STATIC_ROUTE_PLAN_ERROR) from exc
            route_table = source_push_mlp_route_table_from_plan(
                host_inputs.plan,
                src_base_by_expert=host_inputs.src_base_by_expert,
            )
        return source_push_moe_mlp_from_plan(
            config,
            host_inputs,
            route_table,
            x,
            route_weights,
            w13,
            w2,
            implementation=implementation,
            execution_mode=execution_mode,
            mesh=mesh,
        )

    if implementation != SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE:
        raise ValueError(
            f"source_push_moe_mlp implementation={implementation!r} requires an explicit PushInboxConfig; "
            "use config=... or call source_push_moe_mlp_from_plan(...) with prebuilt metadata"
        )
    if execution_mode != FORWARD_EXECUTION_STAGED_HOST_SYNC:
        raise ValueError("source_push_moe_mlp execution_mode is only used with config=... staged execution")
    if mesh is not None:
        raise ValueError(
            "source_push_moe_mlp reference path does not use mesh; pass config=... for staged MGPU execution"
        )

    with jax.ensure_compile_time_eval():
        route_table, dropped_routes = build_source_push_mlp_route_table(
            route_assignments,
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

    y, _, _ = _source_push_moe_mlp_reference_with_h_and_expert_route_weights(
        route_table,
        x,
        route_weights,
        w13,
        w2,
    )
    return y


def source_push_moe_mlp_reference_with_h(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst E C twoI"]]:
    """Return ``(y, H_expert_major)`` for the source-push MLP reference."""

    y, h, _ = _source_push_moe_mlp_reference_with_h_and_expert_route_weights(
        route_table,
        x,
        route_weights,
        w13,
        w2,
    )
    return y, h


def _source_push_moe_mlp_reference_with_h_and_expert_route_weights(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> tuple[
    Float[Array, "S T D"],
    Float[Array, "Dst E C twoI"],
    Float[Array, "Dst E C"],
]:
    """Return reference output, H, and route weights aligned with H rows."""

    expert_route_weights = _source_push_mlp_route_weights_to_all_expert_major(route_table, route_weights)
    h = _source_push_w13_h(route_table, x, w13)
    y = _source_push_w2_from_h_return_combine(route_table, h, expert_route_weights, w2)
    return y, h, expert_route_weights


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

    expert_route_weights = _source_push_mlp_route_weights_to_all_expert_major(route_table, route_weights)
    h_flat = _source_push_w13_h_flat(route_table, expert_base, h_rows_per_rank, x, w13)
    y = _source_push_w2_from_h_flat_return_combine(route_table, expert_base, h_flat, expert_route_weights, w2)
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
    """Run the preplanned source-push MLP with a compact-H custom VJP residual.

    This is the production-shaped differentiable boundary: route metadata is
    prebuilt and static, while ``x``, route weights, ``w13``, and ``w2`` are the
    differentiable inputs. The forward residual is W13 preactivation H in
    compact expert-major layout plus H-aligned route weights; it does not save
    packed/recv x, flat source-padded H rows, post-SwiGLU activation, per-route
    W2 outputs, or scratch.
    """

    _validate_source_push_mlp_implementation(implementation)
    _validate_source_push_mlp_execution_mode(execution_mode)
    _validate_source_push_mlp_mesh(config, mesh, implementation)
    _validate_source_push_mlp_from_plan_shapes(config, route_table, x, route_weights, w13, w2)
    _validate_source_push_mlp_from_plan_metadata(config, host_inputs, route_table)
    expert_base = jnp.asarray(host_inputs.expert_base, dtype=jnp.int32)
    h_rows_per_rank = config.hidden_rows_per_rank

    @jax.custom_vjp
    def _custom_vjp(
        x_arg: Float[Array, "S T D"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "S E D twoI"],
        w2_arg: Float[Array, "S E I D"],
    ) -> Float[Array, "S T D"]:
        y, _ = source_push_moe_mlp_forward_with_compact_h_from_plan(
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
        y, h = source_push_moe_mlp_forward_with_compact_h_from_plan(
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
        expert_route_weights = _source_push_mlp_route_weights_to_all_expert_major(
            route_table,
            route_weights_arg,
        )
        return y, SourcePushMlpPlanResidual(x_arg, expert_route_weights, w13_arg, w2_arg, h)

    def _bwd(residuals: SourcePushMlpPlanResidual, dy):
        if isinstance(dy, jax.custom_derivatives.SymbolicZero):
            return (
                jnp.zeros_like(residuals.x),
                _source_push_mlp_zero_source_route_weights(
                    route_table,
                    residuals.expert_route_weights.dtype,
                ),
                jnp.zeros_like(residuals.w13),
                jnp.zeros_like(residuals.w2),
            )
        gradients = _source_push_moe_mlp_backward_from_plan_h_compact(
            config,
            host_inputs,
            route_table,
            residuals.x,
            residuals.expert_route_weights,
            residuals.w13,
            residuals.w2,
            residuals.h,
            dy,
            return_implementation=_source_push_mlp_backward_return_implementation(implementation),
            dy_route_implementation=_source_push_mlp_backward_dy_route_implementation(implementation),
            w2_implementation=_source_push_mlp_backward_w2_implementation(implementation),
            w2_matmul_implementation=_source_push_mlp_backward_w2_matmul_implementation(implementation),
            w2_swiglu_implementation=_source_push_mlp_backward_w2_swiglu_implementation(implementation),
            w13_implementation=_source_push_mlp_backward_w13_implementation(implementation),
            mesh=mesh,
        )
        return gradients.dx, gradients.d_route_weights, gradients.dw13, gradients.dw2

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
    y, h, expert_route_weights = _source_push_moe_mlp_reference_with_h_and_expert_route_weights(
        route_table,
        x,
        route_weights,
        w13,
        w2,
    )
    return y, SourcePushMlpReferenceResidual(route_table, x, expert_route_weights, w13, w2, h)


def _source_push_moe_mlp_bwd(residuals: SourcePushMlpReferenceResidual, dy):
    if isinstance(dy, jax.custom_derivatives.SymbolicZero):
        return (
            None,
            jnp.zeros_like(residuals.x),
            _source_push_mlp_zero_source_route_weights(
                residuals.route_table,
                residuals.expert_route_weights.dtype,
            ),
            jnp.zeros_like(residuals.w13),
            jnp.zeros_like(residuals.w2),
        )
    gradients = _source_push_moe_mlp_backward_from_h(
        residuals.route_table,
        residuals.x,
        residuals.expert_route_weights,
        residuals.w13,
        residuals.w2,
        residuals.h,
        dy,
    )
    return None, gradients.dx, gradients.d_route_weights, gradients.dw13, gradients.dw2


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


def _validate_source_push_mlp_execution_mode(execution_mode: str) -> None:
    if execution_mode not in FORWARD_EXECUTION_MODES:
        raise ValueError(f"unknown execution_mode={execution_mode!r}; expected one of {FORWARD_EXECUTION_MODES}")


def _validate_source_push_mlp_convenience_config(
    config: PushInboxConfig,
    *,
    ep_size: int,
    experts_per_rank: int,
    block_m: int,
    capacity_factor: float,
    entries_per_dst: int | None,
) -> None:
    if config.ep_size != ep_size:
        raise ValueError(f"config ep_size {config.ep_size} must match ep_size {ep_size}")
    if config.experts_per_rank != experts_per_rank:
        raise ValueError(
            f"config experts_per_rank {config.experts_per_rank} must match experts_per_rank {experts_per_rank}"
        )
    if config.block_m != block_m:
        raise ValueError(f"config block_m {config.block_m} must match block_m {block_m}")
    if config.capacity_factor != capacity_factor:
        raise ValueError(
            f"config capacity_factor {config.capacity_factor} must match capacity_factor {capacity_factor}"
        )
    if entries_per_dst is not None and config.entries_per_rank != entries_per_dst:
        raise ValueError(
            f"config entries_per_rank {config.entries_per_rank} must match entries_per_dst {entries_per_dst}"
        )


def _validate_source_push_mlp_mesh(
    config: PushInboxConfig,
    mesh: Mesh | None,
    implementation: SourcePushMlpImplementation,
) -> None:
    if implementation != SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU or mesh is None:
        return
    if SOURCE_PUSH_MESH_AXIS not in mesh.shape:
        raise ValueError(
            f"{SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU!r} requires a {SOURCE_PUSH_MESH_AXIS!r} mesh axis"
        )
    axis_size = int(mesh.shape[SOURCE_PUSH_MESH_AXIS])
    if axis_size != config.ep_size:
        raise ValueError(
            f"{SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU!r} mesh axis {SOURCE_PUSH_MESH_AXIS!r} size "
            f"{axis_size} must match config ep_size {config.ep_size}"
        )


def _validate_source_push_mlp_from_plan_shapes(
    config: PushInboxConfig,
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> None:
    expected_x_shape = (config.ep_size, config.tokens_per_rank, config.hidden_dim)
    if x.shape != expected_x_shape:
        raise ValueError(f"x shape {x.shape} must match source-push plan shape {expected_x_shape}")

    expected_route_weight_shape = (config.ep_size, config.tokens_per_rank, config.topk)
    if route_weights.shape != expected_route_weight_shape:
        raise ValueError(
            f"route_weights shape {route_weights.shape} must match source-push plan shape "
            f"{expected_route_weight_shape}"
        )

    expected_w13_shape = (
        config.ep_size,
        config.experts_per_rank,
        config.hidden_dim,
        2 * config.intermediate_dim,
    )
    if w13.shape != expected_w13_shape:
        raise ValueError(f"w13 shape {w13.shape} must match source-push plan shape {expected_w13_shape}")

    expected_w2_shape = (
        config.ep_size,
        config.experts_per_rank,
        config.intermediate_dim,
        config.hidden_dim,
    )
    if w2.shape != expected_w2_shape:
        raise ValueError(f"w2 shape {w2.shape} must match source-push plan shape {expected_w2_shape}")

    if route_table.ep_size != config.ep_size:
        raise ValueError(f"route_table ep_size {route_table.ep_size} must match config ep_size {config.ep_size}")
    if route_table.experts_per_rank != config.experts_per_rank:
        raise ValueError(
            f"route_table experts_per_rank {route_table.experts_per_rank} must match config "
            f"experts_per_rank {config.experts_per_rank}"
        )
    if route_table.tokens_per_source != config.tokens_per_rank:
        raise ValueError(
            f"route_table tokens_per_source {route_table.tokens_per_source} must match config tokens_per_rank "
            f"{config.tokens_per_rank}"
        )
    if route_table.topk != config.topk:
        raise ValueError(f"route_table topk {route_table.topk} must match config topk {config.topk}")


def _validate_source_push_mlp_from_plan_metadata(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    route_table: SourcePushMlpRouteTable,
) -> None:
    _validate_source_push_mlp_plan_config(config, host_inputs.plan)
    _validate_source_push_mlp_host_layout_metadata(config, host_inputs)
    _validate_source_push_mlp_host_inverse_metadata(config, host_inputs)
    expected = _source_push_mlp_route_table_host_data_from_plan(
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    for field_name in (
        "source_rank",
        "token_id",
        "route_slot",
        "destination_rank",
        "local_expert",
        "expert_row",
        "source_rank_by_expert",
        "token_id_by_expert",
        "route_slot_by_expert",
        "valid_by_expert",
    ):
        observed_value = np.asarray(jax.device_get(getattr(route_table, field_name)))
        expected_value = getattr(expected, field_name)
        if not np.array_equal(observed_value, expected_value):
            raise ValueError(f"route_table field {field_name!r} must match host_inputs.plan")

    for field_name in ("ep_size", "experts_per_rank", "tokens_per_source", "topk", "expert_capacity"):
        if getattr(route_table, field_name) != getattr(expected, field_name):
            raise ValueError(f"route_table field {field_name!r} must match host_inputs.plan")


def _validate_source_push_mlp_host_inverse_metadata(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
) -> None:
    expected_inverse = _make_route_inverse(config, host_inputs.plan)
    for field_name in ("queue_dst_ord", "queue_entry", "queue_row", "route_valid_mask"):
        _validate_source_push_mlp_host_array(
            field_name,
            getattr(host_inputs, field_name),
            expected_inverse[field_name],
        )

    expected_shape = expected_inverse["route_combine_weights"].shape
    observed_shape = np.asarray(jax.device_get(host_inputs.route_combine_weights)).shape
    if observed_shape != expected_shape:
        raise ValueError(f"host_inputs.route_combine_weights shape {observed_shape} must match {expected_shape}")
    # The MLP H path applies differentiable route weights before W2, so these
    # values may be stale placeholders. Only the inverse layout shape is static metadata.


def _validate_source_push_mlp_plan_config(config: PushInboxConfig, plan: SourcePushPlan) -> None:
    expected_queue_shape = (config.ep_size, config.ep_size, config.entries_per_rank, config.block_m)
    for field_name in ("assignment_ids", "token_ids", "route_slots", "combine_weights", "valid_mask"):
        observed_shape = getattr(plan, field_name).shape
        if observed_shape != expected_queue_shape:
            raise ValueError(
                f"host_inputs.plan {field_name} shape {observed_shape} must match source-push config "
                f"{expected_queue_shape}"
            )

    expected_entry_shape = (config.ep_size, config.ep_size, config.entries_per_rank)
    for field_name in ("local_experts", "local_row_starts"):
        observed_shape = getattr(plan, field_name).shape
        if observed_shape != expected_entry_shape:
            raise ValueError(
                f"host_inputs.plan {field_name} shape {observed_shape} must match source-push config "
                f"{expected_entry_shape}"
            )

    expected_meta_shape = (config.ep_size, config.ep_size, config.entries_per_rank, SOURCE_PUSH_META_FIELDS)
    for field_name in ("send_meta", "recv_meta"):
        observed_shape = getattr(plan, field_name).shape
        if observed_shape != expected_meta_shape:
            raise ValueError(
                f"host_inputs.plan {field_name} shape {observed_shape} must match source-push config "
                f"{expected_meta_shape}"
            )

    expected_counts_shape = (config.ep_size, config.ep_size, config.experts_per_rank)
    if plan.counts_by_src_dst_expert.shape != expected_counts_shape:
        raise ValueError(
            f"host_inputs.plan counts_by_src_dst_expert shape {plan.counts_by_src_dst_expert.shape} must match "
            f"source-push config {expected_counts_shape}"
        )
    expected_expert_shape = (config.ep_size, config.experts_per_rank)
    for field_name in ("rows_per_local_expert", "expert_base"):
        observed_shape = getattr(plan, field_name).shape
        if observed_shape != expected_expert_shape:
            raise ValueError(
                f"host_inputs.plan {field_name} shape {observed_shape} must match source-push config "
                f"{expected_expert_shape}"
            )
    if plan.src_base_by_expert.shape != expected_counts_shape:
        raise ValueError(
            f"host_inputs.plan src_base_by_expert shape {plan.src_base_by_expert.shape} must match "
            f"source-push config {expected_counts_shape}"
        )
    if plan.tokens_per_source != config.tokens_per_rank:
        raise ValueError(
            f"host_inputs.plan tokens_per_source {plan.tokens_per_source} must match config tokens_per_rank "
            f"{config.tokens_per_rank}"
        )
    if plan.topk != config.topk:
        raise ValueError(f"host_inputs.plan topk {plan.topk} must match config topk {config.topk}")


def _validate_source_push_mlp_host_layout_metadata(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
) -> None:
    plan = host_inputs.plan
    if host_inputs.use_exact_expert_major:
        expected_send_meta = np.asarray(jax.device_get(plan.send_meta), dtype=np.int32)
        valid_rows = expected_send_meta[..., SOURCE_PUSH_META_VALID_ROWS]
        tail_blocks = (valid_rows > 0) & (valid_rows < config.block_m)
        if np.any(tail_blocks):
            raise ValueError(
                "exact source-push MLP layout requires block_m-aligned live blocks; "
                "use the source-padded layout for tail blocks"
            )
        expected_recv_meta = np.asarray(jax.device_get(plan.recv_meta), dtype=np.int32)
        expected_expert_base = np.asarray(jax.device_get(plan.expert_base), dtype=np.int32)
        expected_src_base_by_expert = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
        expected_rows_per_local_expert = np.asarray(jax.device_get(plan.rows_per_local_expert), dtype=np.int32)
    else:
        rounded_counts, expected_expert_base, expected_src_base_by_expert = source_push_source_padded_row_bases(
            plan,
            config.block_m,
        )
        expected_send_meta = _source_padded_plan_send_meta(
            plan,
            expected_expert_base,
            expected_src_base_by_expert,
        )
        expected_recv_meta = _recv_meta_from_send_meta(config, expected_send_meta)
        expected_rows_per_local_expert = np.sum(rounded_counts, axis=0, dtype=np.int32)

    _validate_source_push_mlp_host_array("send_meta", host_inputs.send_meta, expected_send_meta)
    _validate_source_push_mlp_host_array("recv_meta", host_inputs.recv_meta, expected_recv_meta)
    _validate_source_push_mlp_host_array("expert_base", host_inputs.expert_base, expected_expert_base)
    _validate_source_push_mlp_host_array(
        "src_base_by_expert",
        host_inputs.src_base_by_expert,
        expected_src_base_by_expert,
    )

    required_hidden_rows = int(np.max(expected_expert_base + expected_rows_per_local_expert))
    if required_hidden_rows > config.hidden_rows_per_rank:
        raise ValueError(
            f"source-push MLP flat-H layout requires {required_hidden_rows} rows per rank, "
            f"but config.hidden_rows_per_rank is {config.hidden_rows_per_rank}"
        )


def _validate_source_push_mlp_host_array(
    field_name: str,
    observed: np.ndarray,
    expected: np.ndarray,
) -> None:
    observed_value = np.asarray(jax.device_get(observed), dtype=expected.dtype)
    if observed_value.shape != expected.shape:
        raise ValueError(f"host_inputs.{field_name} shape {observed_value.shape} must match {expected.shape}")
    if not np.array_equal(observed_value, expected):
        raise ValueError(f"host_inputs.{field_name} must match host_inputs.plan and source-push metadata")


def source_push_moe_mlp_forward_with_compact_h_from_plan(
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
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst E C twoI"]]:
    """Run the preplanned forward and return compact W13 preactivation H.

    The Pallas implementation writes W13 preactivation directly into compact
    expert-major H, then runs W2 from that saved boundary.
    """

    if implementation == SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE:
        return source_push_moe_mlp_reference_with_h(route_table, x, route_weights, w13, w2)
    if execution_mode != FORWARD_EXECUTION_STAGED_HOST_SYNC:
        raise ValueError(
            "compact-H source-push MLP forward currently requires "
            f"execution_mode={FORWARD_EXECUTION_STAGED_HOST_SYNC!r}; got {execution_mode!r}"
        )
    expert_route_weights = _source_push_mlp_route_weights_to_all_expert_major(route_table, route_weights)
    compact_expert_capacity = _compact_h_expert_capacity_from_metadata(
        config,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    h_route_weights = _source_push_mlp_pad_expert_route_weights(expert_route_weights, compact_expert_capacity)
    y, h, _ = source_push_forward_with_compact_h_from_plan(
        config,
        host_inputs,
        x,
        h_route_weights,
        w13,
        w2,
        compact_expert_capacity=compact_expert_capacity,
        mesh=mesh,
    )
    return y, h[:, :, : route_table.expert_capacity, :]


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


def _source_push_mlp_flat_h_to_compact(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_flat: Float[Array, "Dst rows twoI"],
) -> Float[Array, "Dst E C twoI"]:
    """Map source-padded/exact flat W13 rows into compact expert-major H."""

    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    dst = route_table.destination_rank
    expert = route_table.local_expert
    flat_row = expert_base.at[dst, expert].get(out_sharding=_source_push_out_sharding(None)) + route_table.expert_row
    h_rows = h_flat.at[dst, flat_row].get(out_sharding=_source_push_out_sharding(None, None))
    h = jnp.zeros(
        (route_table.ep_size, route_table.experts_per_rank, route_table.expert_capacity, h_flat.shape[-1]),
        dtype=h_rows.dtype,
    )
    return h.at[dst, expert, route_table.expert_row].set(
        h_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )


def _source_push_mlp_compact_h_to_flat(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    h_rows_per_rank: int,
    h: Float[Array, "Dst E C twoI"],
) -> Float[Array, "Dst rows twoI"]:
    """Map compact expert-major H into the legacy flat W13 row layout."""

    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    dst = route_table.destination_rank
    expert = route_table.local_expert
    flat_row = expert_base.at[dst, expert].get(out_sharding=_source_push_out_sharding(None)) + route_table.expert_row
    h_rows = h.at[dst, expert, route_table.expert_row].get(out_sharding=_source_push_out_sharding(None, None))
    h_flat = jnp.zeros((route_table.ep_size, h_rows_per_rank, h.shape[-1]), dtype=h_rows.dtype)
    return h_flat.at[dst, flat_row].set(
        h_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def _source_push_mlp_pad_expert_route_weights(
    expert_route_weights: Float[Array, "Dst E C"],
    compact_expert_capacity: int,
) -> Float[Array, "Dst E C"]:
    """Pad compact expert-major route weights to the W13 compact store capacity."""

    current_capacity = expert_route_weights.shape[2]
    if compact_expert_capacity < current_capacity:
        raise ValueError(
            f"compact_expert_capacity {compact_expert_capacity} must be at least route capacity {current_capacity}"
        )
    if compact_expert_capacity == current_capacity:
        return expert_route_weights
    pad_rows = compact_expert_capacity - current_capacity
    return jnp.pad(expert_route_weights, ((0, 0), (0, 0), (0, pad_rows)))


def _source_push_w2_from_h_return_combine(
    route_table: SourcePushMlpRouteTable,
    h: Float[Array, "Dst E C twoI"],
    expert_route_weights: Float[Array, "Dst E C"],
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
    weights = _source_push_mlp_route_weights_for_routes(route_table, expert_route_weights)
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
    expert_route_weights: Float[Array, "Dst E C"],
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
    weights = _source_push_mlp_route_weights_for_routes(route_table, expert_route_weights)
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
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h: Float[Array, "Dst E C twoI"],
    dy: Float[Array, "S T D"],
) -> _SourcePushMlpGradients:
    return _source_push_moe_mlp_backward_by_expert(
        route_table,
        x,
        expert_route_weights,
        w13,
        w2,
        dy,
        lambda expert: h[:, expert],
    )


def _source_push_moe_mlp_backward_from_h_flat(
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    x: Float[Array, "S T D"],
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h_flat: Float[Array, "Dst rows twoI"],
    dy: Float[Array, "S T D"],
) -> _SourcePushMlpGradients:
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)

    def h_for_expert(expert: int) -> Float[Array, "Dst C twoI"]:
        return _source_push_mlp_h_flat_for_expert(route_table, expert_base, h_flat, expert)

    return _source_push_moe_mlp_backward_by_expert(route_table, x, expert_route_weights, w13, w2, dy, h_for_expert)


def _source_push_moe_mlp_backward_from_plan_h_compact(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h: Float[Array, "Dst E C twoI"],
    dy: Float[Array, "S T D"],
    return_implementation: str = SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX,
    dy_route_implementation: str = SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    w2_implementation: str = SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    w2_matmul_implementation: str | None = None,
    w2_swiglu_implementation: str | None = None,
    w13_implementation: str = SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE,
    mesh: Mesh | None = None,
) -> _SourcePushMlpGradients:
    """From-plan backward organized around compact expert-major H.

    This is the production-shaped counterpart to the staged-block benchmark
    path: route ``dy`` to ``[Dst, E, C, D]``, run W2/SwiGLU backward from saved
    ``H_expert_major``, run W13 backward from compact ``dH``, then return
    compact expert-major token gradients to source-owned ``dx`` and route
    weights. Defaults stay on stable reference stages until individual Pallas
    kernels clear their target-shape gates.
    """

    valid = route_table.valid_by_expert
    dy_blocks = _source_push_backward_dy_to_expert_major(
        dy,
        route_table.source_rank_by_expert,
        route_table.token_id_by_expert,
        valid,
        implementation=dy_route_implementation,
        mesh=mesh,
    )
    w2_grads = _source_push_w2_backward_expert_blocks(
        h,
        expert_route_weights,
        dy_blocks,
        w2,
        valid,
        implementation=w2_implementation,
        matmul_implementation=w2_matmul_implementation,
        swiglu_implementation=w2_swiglu_implementation,
        mesh=mesh,
    )
    if w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED:
        w13_grads = source_push_w13_backward_expert_blocks_tiled_reference(
            x,
            w2_grads.d_h,
            w13,
            route_table.source_rank_by_expert,
            route_table.token_id_by_expert,
            valid,
        )
    elif w13_implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE:
        w13_grads = source_push_w13_backward_expert_blocks_reference(
            x,
            w2_grads.d_h,
            w13,
            host_inputs.plan,
            host_inputs.send_meta,
            jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
            host_inputs.src_base_by_expert,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
    else:
        raise ValueError(
            "compact-H from-plan MLP backward supports W13 implementations "
            f"{(SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED)}, "
            f"got {w13_implementation!r}"
        )
    returned = source_push_backward_return(
        w13_grads.dx_expert_major,
        w2_grads.d_route_weight,
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
        implementation=return_implementation,
        mesh=mesh,
    )
    return _SourcePushMlpGradients(
        dx=returned.dx.astype(x.dtype),
        d_route_weights=returned.d_route_weights.astype(expert_route_weights.dtype),
        dw13=w13_grads.dw13.astype(w13.dtype),
        dw2=w2_grads.dw2.astype(w2.dtype),
    )


def _source_push_moe_mlp_backward_from_plan_h_flat(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    route_table: SourcePushMlpRouteTable,
    expert_base: Int[Array, "Dst E"],
    x: Float[Array, "S T D"],
    h_route_weights: Float[Array, "Dst rows"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    h_flat: Float[Array, "Dst rows twoI"],
    dy: Float[Array, "S T D"],
    return_implementation: str = SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX,
    return_route_indices: SourcePushBackwardFlatRouteIndices | None = None,
    dy_route_implementation: str = SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    w2_implementation: str = SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    w2_matmul_implementation: str | None = None,
    w2_swiglu_implementation: str | None = None,
    w13_implementation: str = SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE,
    mesh: Mesh | None = None,
) -> _SourcePushMlpGradients:
    """Reference from-plan backward organized around replaceable stage boundaries."""

    dy_flat = _source_push_backward_dy_to_h_rows(
        config,
        host_inputs,
        dy,
        implementation=dy_route_implementation,
        mesh=mesh,
    )
    w2_grads = _source_push_w2_backward_from_flat_h(
        expert_base,
        h_flat,
        h_route_weights,
        dy_flat,
        w2,
        route_table.valid_by_expert,
        implementation=w2_implementation,
        matmul_implementation=w2_matmul_implementation,
        swiglu_implementation=w2_swiglu_implementation,
        mesh=mesh,
        contiguous_expert_gather=not host_inputs.use_exact_expert_major,
    )
    w13_grads = source_push_w13_backward(
        x,
        w2_grads.d_h,
        w13,
        host_inputs.plan,
        host_inputs.send_meta,
        expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
        implementation=w13_implementation,
        mesh=mesh,
    )
    returned = source_push_backward_return_flat(
        w13_grads.dx_expert_major,
        w2_grads.d_route_weight,
        host_inputs.plan,
        expert_base=expert_base,
        src_base_by_expert=host_inputs.src_base_by_expert,
        route_indices=return_route_indices,
        implementation=return_implementation,
        mesh=mesh,
    )
    return _SourcePushMlpGradients(
        dx=returned.dx.astype(x.dtype),
        d_route_weights=returned.d_route_weights.astype(h_route_weights.dtype),
        dw13=w13_grads.dw13.astype(w13.dtype),
        dw2=w2_grads.dw2.astype(w2.dtype),
    )


def _source_push_mlp_backward_return_implementation(
    mlp_implementation: SourcePushMlpImplementation,
) -> str:
    if mlp_implementation == SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU and jax.default_backend() == "gpu":
        return SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU
    return SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX


def _source_push_mlp_backward_w2_implementation(
    mlp_implementation: SourcePushMlpImplementation,
) -> str:
    # The staged Pallas W2/SwiGLU derivative currently exposes full
    # destination-major outputs to Mosaic and exceeds H100 shared-memory limits
    # at target shape. Keep it explicit-only until the W2 backward boundary is
    # rewritten around a tiled output contract.
    _ = mlp_implementation
    return SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE


def _source_push_mlp_backward_w2_matmul_implementation(
    mlp_implementation: SourcePushMlpImplementation,
) -> str | None:
    if mlp_implementation == SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU and jax.default_backend() == "gpu":
        return SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU
    return None


def _source_push_mlp_backward_w2_swiglu_implementation(
    mlp_implementation: SourcePushMlpImplementation,
) -> str | None:
    if mlp_implementation == SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU and jax.default_backend() == "gpu":
        return SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE
    return None


def _source_push_mlp_backward_w13_implementation(
    mlp_implementation: SourcePushMlpImplementation,
) -> str:
    # The Pallas x-rematerialization bridge currently lowers with a full-output
    # shared-memory contract at target shape. Keep W13 Pallas explicit-only
    # until the W13 backward boundary is tiled around dx/dW13 WGMMA work.
    _ = mlp_implementation
    return SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE


def _source_push_mlp_backward_dy_route_implementation(
    mlp_implementation: SourcePushMlpImplementation,
) -> str:
    # The Pallas dy-route kernels currently expose a full [rows, D] output to
    # Mosaic and exceed H100 shared-memory limits at target shape. Keep them
    # explicit-only until the dy route is rewritten around a tiled output
    # contract.
    _ = mlp_implementation
    return SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE


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
) -> _SourcePushMlpExpertRouteIndices:
    valid = route_table.valid_by_expert[:, expert]
    safe_src = jnp.maximum(route_table.source_rank_by_expert[:, expert], 0)
    safe_token = jnp.maximum(route_table.token_id_by_expert[:, expert], 0)
    safe_slot = jnp.maximum(route_table.route_slot_by_expert[:, expert], 0)
    return _SourcePushMlpExpertRouteIndices(valid, safe_src, safe_token, safe_slot, valid.astype(jnp.float32))


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
    expert_route_weights: Float[Array, "Dst E C"],
    expert: int,
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C"]:
    return expert_route_weights[:, expert].astype(jnp.float32) * valid_f


def _source_push_mlp_route_weights_for_routes(
    route_table: SourcePushMlpRouteTable,
    expert_route_weights: Float[Array, "Dst E C"],
) -> Float[Array, "R"]:
    weights = expert_route_weights.at[
        route_table.destination_rank,
        route_table.local_expert,
        route_table.expert_row,
    ].get(out_sharding=_source_push_out_sharding(None))
    return weights.astype(jnp.float32)


def _source_push_mlp_route_weights_to_all_expert_major(
    route_table: SourcePushMlpRouteTable,
    route_weights: Float[Array, "S T K"],
) -> Float[Array, "Dst E C"]:
    """Gather source-major route weights into destination expert-major order."""

    safe_src = jnp.maximum(route_table.source_rank_by_expert, 0)
    safe_token = jnp.maximum(route_table.token_id_by_expert, 0)
    safe_slot = jnp.maximum(route_table.route_slot_by_expert, 0)
    weights = route_weights.at[safe_src, safe_token, safe_slot].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    )
    return weights * route_table.valid_by_expert.astype(weights.dtype)


def _source_push_mlp_zero_source_route_weights(
    route_table: SourcePushMlpRouteTable,
    dtype: jnp.dtype,
) -> Float[Array, "S T K"]:
    return jnp.zeros(
        (route_table.ep_size, route_table.tokens_per_source, route_table.topk),
        dtype=dtype,
    )


def _source_push_mlp_zero_gradients(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> _SourcePushMlpGradients:
    """Initialize the full MLP-level backward gradient state."""

    return _SourcePushMlpGradients(
        dx=jnp.zeros_like(x),
        d_route_weights=_source_push_mlp_zero_source_route_weights(route_table, expert_route_weights.dtype),
        dw13=jnp.zeros_like(w13),
        dw2=jnp.zeros_like(w2),
    )


def _source_push_mlp_cast_gradients(
    gradients: _SourcePushMlpGradients,
    x: Float[Array, "S T D"],
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
) -> _SourcePushMlpGradients:
    return _SourcePushMlpGradients(
        dx=gradients.dx.astype(x.dtype),
        d_route_weights=gradients.d_route_weights.astype(expert_route_weights.dtype),
        dw13=gradients.dw13.astype(w13.dtype),
        dw2=gradients.dw2.astype(w2.dtype),
    )


def _source_push_mlp_source_tensor_to_expert_major(
    source_tensor: Float[Array, "S T D"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C D"]:
    """Gather a source-owned token tensor into destination expert-major rows."""

    block = source_tensor.at[safe_src, safe_token].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    )
    return block.astype(jnp.float32) * valid_f[..., None]


def _source_push_mlp_dy_to_expert_major(
    dy: Float[Array, "S T D"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C D"]:
    routed = _source_push_backward_dy_to_expert_major(
        dy,
        safe_src[:, None, :],
        safe_token[:, None, :],
        valid_f[:, None, :] > 0,
        implementation=SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    )
    return routed[:, 0]


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


def _source_push_mlp_w2_swiglu_backward_for_expert(
    h_block: Float[Array, "Dst C twoI"],
    weights: Float[Array, "Dst C"],
    dy_block: Float[Array, "Dst C D"],
    w2_block: Float[Array, "Dst I D"],
    valid_f: Float[Array, "Dst C"],
) -> _SourcePushMlpW2SwiGLUBackwardOutput:
    """Compute W2, route-weight, and SwiGLU gradients from saved H."""

    h_block = h_block.astype(jnp.float32) * valid_f[..., None]
    weights = weights.astype(jnp.float32) * valid_f
    gate, up, silu_gate, activation = _source_push_mlp_activation_from_h(h_block)
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
    return _SourcePushMlpW2SwiGLUBackwardOutput(d_h_block, d_route_block, dw2_block)


def _source_push_mlp_x_to_expert_major(
    x: Float[Array, "S T D"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    valid_f: Float[Array, "Dst C"],
) -> Float[Array, "Dst C D"]:
    return _source_push_mlp_source_tensor_to_expert_major(x, safe_src, safe_token, valid_f)


def _source_push_mlp_w13_backward_for_expert(
    x_block: Float[Array, "Dst C D"],
    d_h_block: Float[Array, "Dst C twoI"],
    w13_block: Float[Array, "Dst D twoI"],
) -> _SourcePushMlpXW13BackwardOutput:
    dx_block = jnp.einsum("sco,sdo->scd", d_h_block, w13_block)
    dw13_block = jnp.einsum("scd,sco->sdo", x_block, d_h_block)
    return _SourcePushMlpXW13BackwardOutput(dx_block, dw13_block)


def _source_push_mlp_x_w13_backward_for_expert(
    x: Float[Array, "S T D"],
    route_indices: _SourcePushMlpExpertRouteIndices,
    d_h_block: Float[Array, "Dst C twoI"],
    w13_block: Float[Array, "Dst D twoI"],
) -> _SourcePushMlpXW13BackwardOutput:
    """Rematerialize expert-major x and compute W13 gradients."""

    x_block = _source_push_mlp_x_to_expert_major(
        x,
        route_indices.safe_src,
        route_indices.safe_token,
        route_indices.valid_f,
    )
    return _source_push_mlp_w13_backward_for_expert(x_block, d_h_block, w13_block)


def _source_push_mlp_accumulate_expert_major_to_source_tokens(
    source_tokens: Float[Array, "S T D"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    expert_block: Float[Array, "Dst C D"],
) -> Float[Array, "S T D"]:
    """Return/combine expert-major token rows into source-owned token rows."""

    return source_tokens.at[safe_src, safe_token].add(
        expert_block,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def _source_push_mlp_accumulate_expert_major_to_source_routes(
    source_routes: Float[Array, "S T K"],
    safe_src: Int[Array, "Dst C"],
    safe_token: Int[Array, "Dst C"],
    safe_slot: Int[Array, "Dst C"],
    expert_block: Float[Array, "Dst C"],
) -> Float[Array, "S T K"]:
    """Scatter/add expert-major route rows into source-owned route slots."""

    return source_routes.at[safe_src, safe_token, safe_slot].add(
        expert_block,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def _source_push_mlp_accumulate_expert_weight_gradients(
    dw13: Float[Array, "S E D twoI"],
    dw2: Float[Array, "S E I D"],
    expert: int,
    dw13_block: Float[Array, "Dst D twoI"],
    dw2_block: Float[Array, "Dst I D"],
) -> tuple[Float[Array, "S E D twoI"], Float[Array, "S E I D"]]:
    """Place one local expert's per-destination W13/W2 gradient blocks."""

    dw13 = dw13.at[:, expert].set(dw13_block.astype(dw13.dtype))
    dw2 = dw2.at[:, expert].set(dw2_block.astype(dw2.dtype))
    return dw13, dw2


def _source_push_mlp_return_expert_backward_to_sources(
    dx: Float[Array, "S T D"],
    d_route_weights: Float[Array, "S T K"],
    route_indices: _SourcePushMlpExpertRouteIndices,
    dx_block: Float[Array, "Dst C D"],
    d_route_block: Float[Array, "Dst C"],
) -> _SourcePushMlpSourceGradientOutput:
    """Return/combine one expert's dx and route-weight gradients to source rows."""

    dx = _source_push_mlp_accumulate_expert_major_to_source_tokens(
        dx,
        route_indices.safe_src,
        route_indices.safe_token,
        dx_block.astype(dx.dtype),
    )
    d_route_weights = _source_push_mlp_accumulate_expert_major_to_source_routes(
        d_route_weights,
        route_indices.safe_src,
        route_indices.safe_token,
        route_indices.safe_slot,
        d_route_block.astype(d_route_weights.dtype),
    )
    return _SourcePushMlpSourceGradientOutput(dx, d_route_weights)


def _source_push_mlp_accumulate_expert_backward_outputs(
    gradients: _SourcePushMlpGradients,
    expert: int,
    backward: _SourcePushMlpExpertBackwardOutput,
) -> _SourcePushMlpGradients:
    route_indices = backward.route_indices
    source_grads = _source_push_mlp_return_expert_backward_to_sources(
        gradients.dx,
        gradients.d_route_weights,
        route_indices,
        backward.dx_block,
        backward.d_route_block,
    )
    dw13, dw2 = _source_push_mlp_accumulate_expert_weight_gradients(
        gradients.dw13,
        gradients.dw2,
        expert,
        backward.dw13_block,
        backward.dw2_block,
    )
    return _SourcePushMlpGradients(source_grads.dx, source_grads.d_route_weights, dw13, dw2)


def _source_push_mlp_expert_backward_inputs(
    route_table: SourcePushMlpRouteTable,
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    dy: Float[Array, "S T D"],
    h_block: Float[Array, "Dst C twoI"],
    expert: int,
) -> _SourcePushMlpExpertBackwardInputs:
    """Route dy and load per-expert inputs for one backward expert block."""

    route_indices = _source_push_mlp_expert_route_indices(route_table, expert)
    weights = _source_push_mlp_route_weights_to_expert_major(
        expert_route_weights,
        expert,
        route_indices.valid_f,
    )
    dy_block = _source_push_mlp_dy_to_expert_major(
        dy,
        route_indices.safe_src,
        route_indices.safe_token,
        route_indices.valid_f,
    )
    return _SourcePushMlpExpertBackwardInputs(
        route_indices=route_indices,
        h_block=h_block,
        weights=weights,
        dy_block=dy_block,
        w2_block=w2[:, expert].astype(jnp.float32),
        w13_block=w13[:, expert].astype(jnp.float32),
    )


def _source_push_mlp_backward_for_expert(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    dy: Float[Array, "S T D"],
    h_block: Float[Array, "Dst C twoI"],
    expert: int,
) -> _SourcePushMlpExpertBackwardOutput:
    backward_inputs = _source_push_mlp_expert_backward_inputs(
        route_table,
        expert_route_weights,
        w13,
        w2,
        dy,
        h_block,
        expert,
    )
    w2_swiglu = _source_push_mlp_w2_swiglu_backward_for_expert(
        backward_inputs.h_block,
        backward_inputs.weights,
        backward_inputs.dy_block,
        backward_inputs.w2_block,
        backward_inputs.route_indices.valid_f,
    )

    x_w13 = _source_push_mlp_x_w13_backward_for_expert(
        x,
        backward_inputs.route_indices,
        w2_swiglu.d_h_block,
        backward_inputs.w13_block,
    )
    return _SourcePushMlpExpertBackwardOutput(
        route_indices=backward_inputs.route_indices,
        dx_block=x_w13.dx_block,
        d_route_block=w2_swiglu.d_route_block,
        dw13_block=x_w13.dw13_block,
        dw2_block=w2_swiglu.dw2_block,
    )


def _source_push_moe_mlp_backward_by_expert(
    route_table: SourcePushMlpRouteTable,
    x: Float[Array, "S T D"],
    expert_route_weights: Float[Array, "Dst E C"],
    w13: Float[Array, "S E D twoI"],
    w2: Float[Array, "S E I D"],
    dy: Float[Array, "S T D"],
    h_for_expert: Callable[[int], Float[Array, "Dst C twoI"]],
) -> _SourcePushMlpGradients:
    gradients = _source_push_mlp_zero_gradients(route_table, x, expert_route_weights, w13, w2)

    for expert in range(route_table.experts_per_rank):
        backward = _source_push_mlp_backward_for_expert(
            route_table,
            x,
            expert_route_weights,
            w13,
            w2,
            dy,
            h_for_expert(expert),
            expert,
        )
        gradients = _source_push_mlp_accumulate_expert_backward_outputs(
            gradients,
            expert,
            backward,
        )

    return _source_push_mlp_cast_gradients(gradients, x, expert_route_weights, w13, w2)
