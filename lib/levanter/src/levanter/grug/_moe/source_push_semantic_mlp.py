# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Package-private semantic source-push MoE MLP integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushSemanticPlan,
    build_source_push_semantic_plan_jax,
)
from levanter.grug._moe.source_push_semantic_backward_pallas import (
    SourcePushSemanticBackwardPallasBlockSizes,
    source_push_semantic_backward_source_expand_from_expert_major_jax,
    source_push_semantic_backward_source_expand_from_expert_major_pallas_mgpu,
    source_push_semantic_dx_return_source_gather_jax,
    source_push_semantic_dx_return_source_gather_pallas_mgpu,
    source_push_semantic_swiglu_backward_expert_major_jax,
)
from levanter.grug._moe.source_push_semantic_backward_w13_pallas import (
    SourcePushSemanticW13BackwardPallasBlockSizes,
    source_push_semantic_w13_backward_expert_major_pallas_mgpu,
)
from levanter.grug._moe.source_push_semantic_backward_w2_pallas import (
    SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes,
    source_push_semantic_w2_backward_expert_major_pallas_mgpu,
)
from levanter.grug._moe.source_push_semantic_forward_pallas import (
    SourcePushSemanticGatherXPallasBlockSizes,
    SourcePushSemanticW13ExpertMajorPallasBlockSizes,
    source_push_semantic_w13_from_x_expert_pallas_mgpu,
    source_push_semantic_x_to_expert_major_direct_pallas_mgpu,
)
from levanter.grug._moe.source_push_semantic_fused_w13 import (
    SourcePushSemanticFusedW13Config,
    source_push_semantic_fused_w13,
)
from levanter.grug._moe.source_push_semantic_fused_w13_backward import source_push_semantic_fused_w13_backward
from levanter.grug._moe.source_push_semantic_fused_w2_backward import source_push_semantic_fused_w2_backward
from levanter.grug._moe.source_push_semantic_fused_w2_return import source_push_semantic_fused_w2_return
from levanter.grug._moe.source_push_semantic_w2_pallas import (
    SourcePushSemanticForwardReturnPallasBlockSizes,
    SourcePushSemanticW2ExpertMajorPallasBlockSizes,
    source_push_semantic_forward_return_sum_owner_sharded_pallas_mgpu,
    source_push_semantic_w2_expert_major_assume_zero_invalid_pallas_mgpu,
)


@dataclass(frozen=True, slots=True)
class SourcePushSemanticMlpCapacity:
    """Static semantic route capacities for one source-push MLP invocation."""

    rows_per_src_dst: int
    rows_per_expert: int


class _SourcePushSemanticMlpResidual(NamedTuple):
    plan: SourcePushSemanticPlan
    x_expert: Float[Array, "Dst E C H"]
    valid: Bool[Array, "Dst E C"]
    z: Float[Array, "Dst E C twoI"]
    h: Float[Array, "Dst E C I"]
    route_y: Float[Array, "Dst E C H"]
    w13: Float[Array, "Dst E H twoI"]
    w2: Float[Array, "Dst E I H"]


class _SourcePushSemanticFusedMlpResidual(NamedTuple):
    plan: SourcePushSemanticPlan
    x: Float[Array, "S T H"]
    z: Float[Array, "Dst E C twoI"]
    return_y: Float[Array, "S DstOrd Q M H"]
    w13: Float[Array, "Dst E H twoI"]
    w2: Float[Array, "Dst E I H"]


_GATHER_BLOCK_SIZES = SourcePushSemanticGatherXPallasBlockSizes(row_block=16, hidden_block=512)
_W13_FORWARD_BLOCK_SIZES = SourcePushSemanticW13ExpertMajorPallasBlockSizes(
    row_block=64,
    hidden_block=128,
    intermediate_block=128,
)
_W2_FORWARD_BLOCK_SIZES = SourcePushSemanticW2ExpertMajorPallasBlockSizes(
    row_block=64,
    intermediate_block=128,
    hidden_block=128,
)
_FORWARD_RETURN_BLOCK_SIZES = SourcePushSemanticForwardReturnPallasBlockSizes(
    row_block=64,
    hidden_block=256,
)
_SOURCE_EXPAND_BLOCK_SIZES = SourcePushSemanticBackwardPallasBlockSizes(
    row_block=128,
    hidden_block=128,
)
_W2_BACKWARD_BLOCK_SIZES = SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes(
    row_block=64,
    intermediate_block=128,
    hidden_block=128,
)
_W13_BACKWARD_BLOCK_SIZES = SourcePushSemanticW13BackwardPallasBlockSizes(
    row_block=64,
    hidden_block=256,
    output_block=64,
)
_DX_GATHER_BLOCK_SIZES = SourcePushSemanticBackwardPallasBlockSizes(
    row_block=64,
    hidden_block=256,
)
_FUSED_CONFIG = SourcePushSemanticFusedW13Config()


def source_push_moe_mlp_semantic_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    x: Float[Array, "S T H"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "Dst E H twoI"],
    w2: Float[Array, "Dst E I H"],
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    mesh: Mesh,
) -> tuple[Float[Array, "S T H"], Int[Array, ""]]:
    """Run the fixed-profile semantic source-push MLP on an explicit GPU mesh."""

    _validate_source_push_semantic_mlp_inputs(
        selected_experts,
        x,
        route_weights,
        w13,
        w2,
        capacity=capacity,
        capacity_factor=capacity_factor,
        validate_profile=True,
    )
    _validate_source_push_semantic_mlp_mesh(mesh, source_count=x.shape[0])
    return _source_push_moe_mlp_semantic_pallas_mgpu(
        selected_experts,
        x,
        route_weights,
        w13,
        w2,
        capacity=capacity,
        capacity_factor=capacity_factor,
        mesh=mesh,
        interpret=False,
    )


def _source_push_moe_mlp_semantic_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    x: Float[Array, "S T H"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "Dst E H twoI"],
    w2: Float[Array, "Dst E I H"],
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    mesh: Mesh | None,
    interpret: bool,
) -> tuple[Float[Array, "S T H"], Int[Array, ""]]:
    """Interpret-capable semantic MLP boundary used by CPU correctness tests."""

    _validate_source_push_semantic_mlp_inputs(
        selected_experts,
        x,
        route_weights,
        w13,
        w2,
        capacity=capacity,
        capacity_factor=capacity_factor,
        validate_profile=not interpret,
    )
    if not interpret and mesh is None:
        raise ValueError("non-interpreted semantic source-push MLP requires an explicit mesh")

    @jax.custom_vjp
    def _custom_vjp(
        selected_experts_arg: Int[Array, "S T K"],
        x_arg: Float[Array, "S T H"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "Dst E H twoI"],
        w2_arg: Float[Array, "Dst E I H"],
    ) -> tuple[Float[Array, "S T H"], Int[Array, ""]]:
        y, residual = _source_push_semantic_mlp_forward(
            selected_experts_arg,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=capacity_factor,
            mesh=mesh,
            interpret=interpret,
        )
        return y, residual.plan.dropped_routes

    def _fwd(
        selected_experts_arg: Int[Array, "S T K"],
        x_arg: Float[Array, "S T H"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "Dst E H twoI"],
        w2_arg: Float[Array, "Dst E I H"],
    ):
        y, residual = _source_push_semantic_mlp_forward(
            selected_experts_arg,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=capacity_factor,
            mesh=mesh,
            interpret=interpret,
        )
        return (y, residual.plan.dropped_routes), residual

    def _bwd(
        residual: _SourcePushSemanticMlpResidual,
        cotangents: tuple[Float[Array, "S T H"], Array],
    ):
        dy, _dropped_cotangent = cotangents
        if isinstance(dy, jax.custom_derivatives.SymbolicZero):
            return None, *_source_push_semantic_mlp_zero_gradients(residual)
        gradients = _source_push_semantic_mlp_backward(
            residual,
            dy,
            mesh=mesh,
            interpret=interpret,
        )
        return None, *gradients

    _custom_vjp.defvjp(_fwd, _bwd)
    return _custom_vjp(selected_experts, x, route_weights, w13, w2)


def source_push_moe_mlp_semantic_fused_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    x: Float[Array, "S T H"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "Dst E H twoI"],
    w2: Float[Array, "Dst E I H"],
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    mesh: Mesh,
) -> tuple[Float[Array, "S T H"], Int[Array, ""]]:
    """Run the fused semantic source-push MLP on an explicit GPU mesh."""

    _validate_source_push_semantic_mlp_mesh(mesh, source_count=x.shape[0])
    return _source_push_moe_mlp_semantic_fused_pallas_mgpu(
        selected_experts,
        x,
        route_weights,
        w13,
        w2,
        capacity=capacity,
        capacity_factor=capacity_factor,
        mesh=mesh,
        interpret=False,
    )


def _source_push_moe_mlp_semantic_fused_pallas_mgpu(
    selected_experts: Int[Array, "S T K"],
    x: Float[Array, "S T H"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "Dst E H twoI"],
    w2: Float[Array, "Dst E I H"],
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    mesh: Mesh | None,
    interpret: bool,
) -> tuple[Float[Array, "S T H"], Int[Array, ""]]:
    """Opt-in fused semantic MLP boundary used by Hopper and interpreted correctness tests."""

    _validate_source_push_semantic_mlp_inputs(
        selected_experts,
        x,
        route_weights,
        w13,
        w2,
        capacity=capacity,
        capacity_factor=capacity_factor,
        validate_profile=False,
    )
    if not interpret:
        if mesh is None:
            raise ValueError("non-interpreted fused semantic source-push MLP requires an explicit mesh")
        _validate_source_push_semantic_fused_mlp_profile(x, w13, w2, capacity)
        selected_experts = _constrain_replicated(selected_experts, mesh)
        route_weights = _constrain_replicated(route_weights, mesh)
        x = _constrain_destination_major(x, mesh)
        w13 = _constrain_destination_major(w13, mesh)
        w2 = _constrain_destination_major(w2, mesh)

    send_chunks_per_dst, entries_per_dst = _source_push_semantic_fused_queue_geometry(
        rows_per_src_dst=capacity.rows_per_src_dst,
        experts_per_rank=w13.shape[1],
    )

    @jax.custom_vjp
    def _custom_vjp(
        selected_experts_arg: Int[Array, "S T K"],
        x_arg: Float[Array, "S T H"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "Dst E H twoI"],
        w2_arg: Float[Array, "Dst E I H"],
    ) -> tuple[Float[Array, "S T H"], Int[Array, ""]]:
        y, dropped, _residual = _source_push_semantic_fused_mlp_forward(
            selected_experts_arg,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=capacity_factor,
            send_chunks_per_dst=send_chunks_per_dst,
            entries_per_dst=entries_per_dst,
            mesh=mesh,
            interpret=interpret,
        )
        return y, dropped

    def _fwd(
        selected_experts_arg: Int[Array, "S T K"],
        x_arg: Float[Array, "S T H"],
        route_weights_arg: Float[Array, "S T K"],
        w13_arg: Float[Array, "Dst E H twoI"],
        w2_arg: Float[Array, "Dst E I H"],
    ):
        y, dropped, residual = _source_push_semantic_fused_mlp_forward(
            selected_experts_arg,
            x_arg,
            route_weights_arg,
            w13_arg,
            w2_arg,
            capacity=capacity,
            capacity_factor=capacity_factor,
            send_chunks_per_dst=send_chunks_per_dst,
            entries_per_dst=entries_per_dst,
            mesh=mesh,
            interpret=interpret,
        )
        return (y, dropped), residual

    def _bwd(
        residual: _SourcePushSemanticFusedMlpResidual,
        cotangents: tuple[Float[Array, "S T H"], Array],
    ):
        dy, _dropped_cotangent = cotangents
        if isinstance(dy, jax.custom_derivatives.SymbolicZero):
            return None, *_source_push_semantic_fused_mlp_zero_gradients(residual)
        gradients = _source_push_semantic_fused_mlp_backward(
            residual,
            dy,
            send_chunks_per_dst=send_chunks_per_dst,
            mesh=mesh,
            interpret=interpret,
        )
        return None, *gradients

    _custom_vjp.defvjp(_fwd, _bwd)
    return _custom_vjp(selected_experts, x, route_weights, w13, w2)


def _source_push_semantic_fused_mlp_forward(
    selected_experts: Int[Array, "S T K"],
    x: Float[Array, "S T H"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "Dst E H twoI"],
    w2: Float[Array, "Dst E I H"],
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    send_chunks_per_dst: int,
    entries_per_dst: int,
    mesh: Mesh | None,
    interpret: bool,
) -> tuple[Float[Array, "S T H"], Int[Array, ""], _SourcePushSemanticFusedMlpResidual]:
    plan = _build_source_push_semantic_mlp_plan(
        selected_experts,
        route_weights,
        experts_per_rank=w13.shape[1],
        capacity=capacity,
        capacity_factor=capacity_factor,
    )
    fused_w13 = source_push_semantic_fused_w13(
        x,
        w13,
        plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=capacity.rows_per_expert,
        mesh=mesh,
        interpret=interpret,
    )
    fused_w2 = source_push_semantic_fused_w2_return(
        fused_w13.z,
        w2,
        plan,
        entries_per_dst=entries_per_dst,
        mesh=mesh,
        interpret=interpret,
    )
    queue_overflow_routes = jnp.maximum(fused_w13.queue_overflow_routes, fused_w2.queue_overflow_routes)
    layout_overflow_rows = jnp.maximum(fused_w13.layout_overflow_rows, fused_w2.layout_overflow_rows)
    dropped = plan.dropped_routes + queue_overflow_routes + layout_overflow_rows
    residual = _SourcePushSemanticFusedMlpResidual(plan, x, fused_w13.z, fused_w2.return_y, w13, w2)
    return fused_w2.y, dropped, residual


def _source_push_semantic_fused_mlp_backward(
    residual: _SourcePushSemanticFusedMlpResidual,
    dy: Float[Array, "S T H"],
    *,
    send_chunks_per_dst: int,
    mesh: Mesh | None,
    interpret: bool,
) -> tuple[
    Float[Array, "S T H"],
    Float[Array, "S T K"],
    Float[Array, "Dst E H twoI"],
    Float[Array, "Dst E I H"],
]:
    fused_w2_backward = source_push_semantic_fused_w2_backward(
        dy.astype(jnp.bfloat16),
        residual.return_y,
        residual.z,
        residual.w2,
        residual.plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=residual.z.shape[2],
        mesh=mesh,
        interpret=interpret,
    )
    fused_w13_backward = source_push_semantic_fused_w13_backward(
        residual.x,
        fused_w2_backward.d_z13.astype(residual.w13.dtype),
        residual.w13,
        residual.plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=residual.z.shape[2],
        mesh=mesh,
        interpret=interpret,
    )
    d_route_weight = fused_w2_backward.d_route_weight.astype(residual.plan.route_weights.dtype)
    d_route_weight = _constrain_replicated(d_route_weight, mesh)
    return (
        fused_w13_backward.dx.astype(residual.x.dtype),
        d_route_weight,
        fused_w13_backward.dw13.astype(residual.w13.dtype),
        fused_w2_backward.d_w2.astype(residual.w2.dtype),
    )


def _source_push_semantic_fused_queue_geometry(*, rows_per_src_dst: int, experts_per_rank: int) -> tuple[int, int]:
    if rows_per_src_dst <= 0:
        raise ValueError(f"rows_per_src_dst must be positive, got {rows_per_src_dst}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    compute_blocks = (rows_per_src_dst + _FUSED_CONFIG.compute_m - 1) // _FUSED_CONFIG.compute_m
    required_entries_per_dst = compute_blocks + experts_per_rank - 1
    send_chunks_per_dst = (
        required_entries_per_dst + _FUSED_CONFIG.compute_blocks_per_send - 1
    ) // _FUSED_CONFIG.compute_blocks_per_send
    entries_per_dst = send_chunks_per_dst * _FUSED_CONFIG.compute_blocks_per_send
    return send_chunks_per_dst, entries_per_dst


def _source_push_semantic_mlp_forward(
    selected_experts: Int[Array, "S T K"],
    x: Float[Array, "S T H"],
    route_weights: Float[Array, "S T K"],
    w13: Float[Array, "Dst E H twoI"],
    w2: Float[Array, "Dst E I H"],
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    mesh: Mesh | None,
    interpret: bool,
) -> tuple[Float[Array, "S T H"], _SourcePushSemanticMlpResidual]:
    plan = _build_source_push_semantic_mlp_plan(
        selected_experts,
        route_weights,
        experts_per_rank=w13.shape[1],
        capacity=capacity,
        capacity_factor=capacity_factor,
    )
    x_expert, valid = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
        x,
        plan,
        rows_per_expert_capacity=capacity.rows_per_expert,
        block_sizes=_GATHER_BLOCK_SIZES,
        interpret=interpret,
    )
    if mesh is not None:
        x_expert = _constrain_destination_major(x_expert, mesh)
        valid = _constrain_destination_major(valid, mesh)
        w13 = _constrain_destination_major(w13, mesh)
        w2 = _constrain_destination_major(w2, mesh)
    z, h = source_push_semantic_w13_from_x_expert_pallas_mgpu(
        x_expert,
        w13,
        valid,
        block_sizes=_W13_FORWARD_BLOCK_SIZES,
        interpret=interpret,
        mesh=mesh,
    )
    h = h.astype(w2.dtype)
    route_y = source_push_semantic_w2_expert_major_assume_zero_invalid_pallas_mgpu(
        h,
        w2,
        block_sizes=_W2_FORWARD_BLOCK_SIZES,
        interpret=interpret,
        mesh=mesh,
    )
    if interpret:
        y = source_push_semantic_forward_return_sum_owner_sharded_pallas_mgpu(
            route_y,
            plan,
            block_sizes=_FORWARD_RETURN_BLOCK_SIZES,
            interpret=True,
            mesh=mesh,
        )
    else:
        assert mesh is not None
        y = source_push_semantic_forward_return_sum_owner_sharded_pallas_mgpu(
            route_y,
            plan,
            block_sizes=_FORWARD_RETURN_BLOCK_SIZES,
            interpret=False,
            mesh=mesh,
        )
    residual = _SourcePushSemanticMlpResidual(plan, x_expert, valid, z, h, route_y, w13, w2)
    return y, residual


def _source_push_semantic_mlp_backward(
    residual: _SourcePushSemanticMlpResidual,
    dy: Float[Array, "S T H"],
    *,
    mesh: Mesh | None,
    interpret: bool,
) -> tuple[
    Float[Array, "S T H"],
    Float[Array, "S T K"],
    Float[Array, "Dst E H twoI"],
    Float[Array, "Dst E I H"],
]:
    if interpret:
        dy_route, d_route_weights = source_push_semantic_backward_source_expand_from_expert_major_jax(
            dy,
            residual.route_y,
            residual.plan,
        )
    else:
        assert mesh is not None
        dy = _constrain_replicated(dy, mesh)
        dy_route, d_route_weights = source_push_semantic_backward_source_expand_from_expert_major_pallas_mgpu(
            dy,
            residual.route_y,
            residual.plan,
            block_sizes=_SOURCE_EXPAND_BLOCK_SIZES,
            interpret=False,
            mesh=mesh,
        )
    dh, dw2 = source_push_semantic_w2_backward_expert_major_pallas_mgpu(
        residual.h,
        dy_route.astype(residual.w2.dtype),
        residual.w2,
        residual.valid,
        block_sizes=_W2_BACKWARD_BLOCK_SIZES,
        interpret=interpret,
        mesh=mesh,
    )
    dz = source_push_semantic_swiglu_backward_expert_major_jax(dh, residual.z, residual.valid).astype(
        residual.w13.dtype
    )
    dx_route, dw13 = source_push_semantic_w13_backward_expert_major_pallas_mgpu(
        residual.x_expert,
        dz,
        residual.w13,
        residual.valid,
        block_sizes=_W13_BACKWARD_BLOCK_SIZES,
        interpret=interpret,
        lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        mesh=mesh,
    )
    if interpret:
        dx = source_push_semantic_dx_return_source_gather_jax(dx_route, residual.plan)
    else:
        assert mesh is not None
        dx = source_push_semantic_dx_return_source_gather_pallas_mgpu(
            dx_route,
            residual.plan,
            block_sizes=_DX_GATHER_BLOCK_SIZES,
            interpret=False,
            mesh=mesh,
        )
    return (
        dx.astype(residual.x_expert.dtype),
        d_route_weights.astype(residual.plan.route_weights.dtype),
        dw13.astype(residual.w13.dtype),
        dw2.astype(residual.w2.dtype),
    )


def _build_source_push_semantic_mlp_plan(
    selected_experts: Int[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    *,
    experts_per_rank: int,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
) -> SourcePushSemanticPlan:
    return build_source_push_semantic_plan_jax(
        selected_experts,
        route_weights,
        ep_size=selected_experts.shape[0],
        experts_per_rank=experts_per_rank,
        rows_per_src_dst_capacity=capacity.rows_per_src_dst,
        rows_per_expert_capacity=capacity.rows_per_expert,
        capacity_factor=capacity_factor,
    )


def _source_push_semantic_mlp_zero_gradients(
    residual: _SourcePushSemanticMlpResidual,
) -> tuple[Array, Array, Array, Array]:
    plan = residual.plan
    dx = jnp.zeros(
        (plan.assignment_ids.shape[0], plan.tokens_per_source, residual.x_expert.shape[-1]),
        dtype=residual.x_expert.dtype,
    )
    d_route_weights = jnp.zeros(plan.reverse_route.route_valid.shape, dtype=plan.route_weights.dtype)
    return dx, d_route_weights, jnp.zeros_like(residual.w13), jnp.zeros_like(residual.w2)


def _source_push_semantic_fused_mlp_zero_gradients(
    residual: _SourcePushSemanticFusedMlpResidual,
) -> tuple[Array, Array, Array, Array]:
    plan = residual.plan
    d_route_weights = jnp.zeros(plan.reverse_route.route_valid.shape, dtype=plan.route_weights.dtype)
    return jnp.zeros_like(residual.x), d_route_weights, jnp.zeros_like(residual.w13), jnp.zeros_like(residual.w2)


def _validate_source_push_semantic_mlp_inputs(
    selected_experts: Array,
    x: Array,
    route_weights: Array,
    w13: Array,
    w2: Array,
    *,
    capacity: SourcePushSemanticMlpCapacity,
    capacity_factor: float,
    validate_profile: bool,
) -> None:
    if selected_experts.ndim != 3:
        raise ValueError(f"selected_experts must have shape [source, token, topk], got {selected_experts.shape}")
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if route_weights.shape != selected_experts.shape:
        raise ValueError(
            f"route_weights shape {route_weights.shape} must match selected_experts {selected_experts.shape}"
        )
    if x.shape[:2] != selected_experts.shape[:2]:
        raise ValueError(
            f"x source/token shape {x.shape[:2]} must match selected_experts {selected_experts.shape[:2]}"
        )
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [destination, expert, hidden, 2 * intermediate], got {w13.shape}")
    if w13.shape[0] != x.shape[0]:
        raise ValueError(f"w13 destination dim {w13.shape[0]} must match source count {x.shape[0]}")
    if w13.shape[2] != x.shape[2]:
        raise ValueError(f"w13 hidden dim {w13.shape[2]} must match x hidden dim {x.shape[2]}")
    if w13.shape[3] % 2:
        raise ValueError(f"w13 output dim must be even gate/up pairs, got {w13.shape[3]}")
    expected_w2_shape = (x.shape[0], w13.shape[1], w13.shape[3] // 2, x.shape[2])
    if w2.shape != expected_w2_shape:
        raise ValueError(f"w2 shape {w2.shape} must be {expected_w2_shape}")
    if capacity.rows_per_src_dst <= 0:
        raise ValueError(f"capacity.rows_per_src_dst must be positive, got {capacity.rows_per_src_dst}")
    if capacity.rows_per_expert <= 0:
        raise ValueError(f"capacity.rows_per_expert must be positive, got {capacity.rows_per_expert}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    if not jnp.issubdtype(selected_experts.dtype, jnp.integer):
        raise ValueError(f"selected_experts must have an integer dtype, got {selected_experts.dtype}")
    for name, value in (("x", x), ("route_weights", route_weights), ("w13", w13), ("w2", w2)):
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise ValueError(f"{name} must have a floating dtype, got {value.dtype}")
    if validate_profile:
        _validate_source_push_semantic_mlp_profile(x, w13, capacity)


def _validate_source_push_semantic_mlp_profile(
    x: Array,
    w13: Array,
    capacity: SourcePushSemanticMlpCapacity,
) -> None:
    if capacity.rows_per_expert % _SOURCE_EXPAND_BLOCK_SIZES.row_block:
        raise ValueError(
            f"capacity.rows_per_expert={capacity.rows_per_expert} must be divisible by fixed profile row block "
            f"{_SOURCE_EXPAND_BLOCK_SIZES.row_block}"
        )
    if x.shape[2] % _DX_GATHER_BLOCK_SIZES.hidden_block:
        raise ValueError(
            f"hidden dim {x.shape[2]} must be divisible by fixed profile hidden block "
            f"{_DX_GATHER_BLOCK_SIZES.hidden_block}"
        )
    intermediate_dim = w13.shape[3] // 2
    if intermediate_dim % _W13_FORWARD_BLOCK_SIZES.intermediate_block:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by fixed profile intermediate block "
            f"{_W13_FORWARD_BLOCK_SIZES.intermediate_block}"
        )


def _validate_source_push_semantic_fused_mlp_profile(
    x: Array,
    w13: Array,
    w2: Array,
    capacity: SourcePushSemanticMlpCapacity,
) -> None:
    for name, value in (("x", x), ("w13", w13), ("w2", w2)):
        if value.dtype != jnp.bfloat16:
            raise ValueError(
                f"non-interpreted fused semantic source-push MLP requires bfloat16 {name}, got {value.dtype}"
            )
    if capacity.rows_per_expert % _FUSED_CONFIG.compute_m:
        raise ValueError(
            f"capacity.rows_per_expert={capacity.rows_per_expert} must be divisible by fused compute row block "
            f"{_FUSED_CONFIG.compute_m}"
        )
    if x.shape[-1] % _FUSED_CONFIG.send_k:
        raise ValueError(f"hidden dim {x.shape[-1]} must be divisible by fused send block {_FUSED_CONFIG.send_k}")
    intermediate_dim = w13.shape[-1] // 2
    if intermediate_dim % _FUSED_CONFIG.block_n:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by fused output block {_FUSED_CONFIG.block_n}"
        )


def _validate_source_push_semantic_mlp_mesh(mesh: Mesh, *, source_count: int) -> None:
    if not isinstance(mesh, Mesh):
        raise ValueError(f"semantic source-push MLP requires a concrete Mesh, got {type(mesh).__name__}")
    if SOURCE_PUSH_MESH_AXIS not in mesh.shape:
        raise ValueError(f"semantic source-push MLP requires mesh axis {SOURCE_PUSH_MESH_AXIS!r}")
    if int(mesh.shape[SOURCE_PUSH_MESH_AXIS]) != source_count:
        raise ValueError(
            f"mesh axis {SOURCE_PUSH_MESH_AXIS!r} size {mesh.shape[SOURCE_PUSH_MESH_AXIS]} "
            f"must match source count {source_count}"
        )
    axis_types = dict(zip(mesh.axis_names, mesh.axis_types, strict=True))
    if axis_types[SOURCE_PUSH_MESH_AXIS] is not AxisType.Explicit:
        raise ValueError(f"mesh axis {SOURCE_PUSH_MESH_AXIS!r} must use AxisType.Explicit")
    for axis_name, axis_size in mesh.shape.items():
        if axis_name != SOURCE_PUSH_MESH_AXIS and int(axis_size) != 1:
            raise ValueError(f"semantic source-push MLP does not support nontrivial mesh axis {axis_name!r}")
    platforms = {device.platform for device in mesh.devices.flat}
    if platforms != {"gpu"}:
        raise NotImplementedError(f"semantic source-push MLP requires GPU mesh devices, got {sorted(platforms)}")


def _constrain_destination_major(value: Array, mesh: Mesh) -> Array:
    spec = P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(value.ndim - 1)))
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        return jax.sharding.reshard(value, spec)
    return jax.sharding.reshard(value, NamedSharding(mesh, spec))


def _constrain_replicated(value: Array, mesh: Mesh | None) -> Array:
    spec = P(*(None for _ in range(value.ndim)))
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        return jax.sharding.reshard(value, spec)
    if mesh is None:
        return value
    return jax.sharding.reshard(value, NamedSharding(mesh, spec))
