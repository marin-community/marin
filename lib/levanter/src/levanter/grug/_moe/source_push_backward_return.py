# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Return/combine helpers for source-push MLP backward rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushPlan,
    _source_push_out_sharding,
    _with_source_push_sharding,
)


SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX = "jax"
SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SourcePushBackwardReturnImplementation: TypeAlias = Literal["jax", "pallas_mgpu"]
SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATIONS = (
    SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX,
    SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU,
)
DEFAULT_BACKWARD_RETURN_HIDDEN_BLOCK = 128


@dataclass(frozen=True, slots=True)
class SourcePushBackwardReturnPallasBlockSizes:
    """Tile sizes for the source-local backward return Pallas kernel."""

    hidden_block: int = DEFAULT_BACKWARD_RETURN_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushBackwardReturnPallasBlockSizes":
        return cls()


class SourcePushBackwardReturnOutput(NamedTuple):
    """Source-owned token and route-weight gradients."""

    dx: Float[Array, "S T D"]
    d_route_weights: Float[Array, "S T K"]


class SourcePushBackwardReturnRouteBuffer(NamedTuple):
    """Per-route source-owned gradients before top-k token summation."""

    dx_routes: Float[Array, "S T K D"]
    d_route_weights: Float[Array, "S T K"]


class SourcePushBackwardFlatRouteIndices(NamedTuple):
    """Source-local inverse map from token route slots to destination flat rows."""

    dst: Int[Array, "S T K"]
    row: Int[Array, "S T K"]
    valid: Bool[Array, "S T K"]


class SourcePushBackwardCompactRouteIndices(NamedTuple):
    """Source-local inverse map from token route slots to destination compact rows."""

    dst: Int[Array, "S T K"]
    expert: Int[Array, "S T K"]
    row: Int[Array, "S T K"]
    valid: Bool[Array, "S T K"]


def source_push_backward_return(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
    route_indices: SourcePushBackwardCompactRouteIndices | None = None,
    implementation: SourcePushBackwardReturnImplementation = SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes | None = None,
    mesh: Mesh | None = None,
) -> SourcePushBackwardReturnOutput:
    """Return compact expert-major backward rows to source-owned gradients.

    ``dx_expert_major`` and ``d_route_block`` are aligned with the destination
    expert-major layout ``[destination, local_expert, row]``. Duplicate top-k
    token contributions are first placed in a source route buffer and then
    summed in route-slot order.
    """

    if implementation == SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU:
        _validate_compact_shapes(dx_expert_major, d_route_block, plan, src_base_by_expert)
        if mesh is not None:
            if route_indices is not None:
                _validate_compact_index_request(dx_expert_major, d_route_block, route_indices)
            else:
                route_indices = source_push_backward_return_route_indices_jax(
                    plan,
                    src_base_by_expert=src_base_by_expert,
                )
            return _source_push_backward_return_compact_pallas_mgpu(
                dx_expert_major,
                d_route_block,
                route_indices,
                block_sizes=block_sizes,
                mesh=mesh,
            )
        if route_indices is None:
            route_indices = source_push_backward_return_route_indices_jax(
                plan,
                src_base_by_expert=src_base_by_expert,
            )
        return _source_push_backward_return_compact_pallas_mgpu(
            dx_expert_major,
            d_route_block,
            route_indices,
            block_sizes=block_sizes,
        )
    if implementation != SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX:
        raise ValueError(
            "source-push backward return implementation must be one of "
            f"{SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATIONS}, "
            f"got {implementation!r}"
        )
    return source_push_backward_return_jax(
        dx_expert_major,
        d_route_block,
        plan,
        src_base_by_expert=src_base_by_expert,
        route_indices=route_indices,
    )


def source_push_backward_return_jax(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
    route_indices: SourcePushBackwardCompactRouteIndices | None = None,
) -> SourcePushBackwardReturnOutput:
    """JAX reference return/combine for compact expert-major backward rows."""

    if route_indices is not None:
        _validate_compact_shapes(dx_expert_major, d_route_block, plan, src_base_by_expert)
        return _source_push_backward_return_compact_from_indices_jax(dx_expert_major, d_route_block, route_indices)

    buffers = source_push_backward_return_route_buffer_jax(
        dx_expert_major,
        d_route_block,
        plan,
        src_base_by_expert=src_base_by_expert,
    )
    return SourcePushBackwardReturnOutput(
        dx=jnp.sum(buffers.dx_routes, axis=2),
        d_route_weights=buffers.d_route_weights,
    )


def source_push_backward_return_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> SourcePushBackwardReturnOutput:
    """Readable JAX reference for compact expert-major backward return/combine."""

    return source_push_backward_return_jax(
        dx_expert_major,
        d_route_block,
        plan,
        src_base_by_expert=src_base_by_expert,
    )


def source_push_backward_return_route_buffer_jax(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> SourcePushBackwardReturnRouteBuffer:
    """Return compact expert-major rows into a source-owned route buffer."""

    _validate_compact_shapes(dx_expert_major, d_route_block, plan, src_base_by_expert)
    route_rows = _compact_route_rows(plan, src_base_by_expert)
    queue_dx = dx_expert_major.at[route_rows.dst, route_rows.expert, route_rows.row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None, None)
    )
    queue_d_route = d_route_block.at[route_rows.dst, route_rows.expert, route_rows.row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    return _source_route_buffer_from_queue_rows(queue_dx, queue_d_route, route_rows, plan)


def source_push_backward_return_route_indices_jax(
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> SourcePushBackwardCompactRouteIndices:
    """Build the source-local inverse map consumed by compact return kernels.

    This map keeps the expert-block layout intact: token route slots point to
    ``[dst, local_expert, row]`` instead of a flattened row. It is routing
    metadata and should be cached with the plan by callers that run the backward
    path repeatedly.
    """

    route_rows = _compact_route_rows(plan, src_base_by_expert)
    route_shape = (plan.valid_mask.shape[0], plan.tokens_per_source, plan.topk)
    valid_i = jnp.zeros(route_shape, dtype=jnp.int32)
    dst = jnp.zeros(route_shape, dtype=jnp.int32)
    expert = jnp.zeros(route_shape, dtype=jnp.int32)
    row = jnp.zeros(route_shape, dtype=jnp.int32)

    valid_i = valid_i.at[route_rows.src, route_rows.token, route_rows.slot].add(route_rows.valid.astype(jnp.int32))
    dst = dst.at[route_rows.src, route_rows.token, route_rows.slot].add(
        jnp.where(route_rows.valid, route_rows.dst, 0),
    )
    expert = expert.at[route_rows.src, route_rows.token, route_rows.slot].add(
        jnp.where(route_rows.valid, route_rows.expert, 0),
    )
    row = row.at[route_rows.src, route_rows.token, route_rows.slot].add(
        jnp.where(route_rows.valid, route_rows.row, 0),
    )
    return SourcePushBackwardCompactRouteIndices(dst=dst, expert=expert, row=row, valid=valid_i > 0)


def _source_push_backward_return_compact_from_indices_jax(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_indices: SourcePushBackwardCompactRouteIndices,
) -> SourcePushBackwardReturnOutput:
    _validate_compact_index_request(dx_expert_major, d_route_block, route_indices)
    dx, d_route_weights = _source_push_backward_return_compact_from_indices_reference(
        dx_expert_major,
        d_route_block,
        route_indices.dst,
        route_indices.expert,
        route_indices.row,
        route_indices.valid.astype(jnp.int32),
    )
    return SourcePushBackwardReturnOutput(dx=dx, d_route_weights=d_route_weights)


def source_push_backward_return_flat_jax(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    plan: SourcePushPlan,
    *,
    expert_base: Int[Array, "Dst E"] | None = None,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
    route_indices: SourcePushBackwardFlatRouteIndices | None = None,
) -> SourcePushBackwardReturnOutput:
    """JAX reference return/combine for flat destination expert-major rows."""

    if route_indices is not None:
        _validate_flat_shapes(dx_expert_major, d_route_block, plan, expert_base, src_base_by_expert)
        return _source_push_backward_return_flat_from_indices_jax(dx_expert_major, d_route_block, route_indices)

    buffers = source_push_backward_return_flat_route_buffer_jax(
        dx_expert_major,
        d_route_block,
        plan,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
    )
    return SourcePushBackwardReturnOutput(
        dx=jnp.sum(buffers.dx_routes, axis=2),
        d_route_weights=buffers.d_route_weights,
    )


def source_push_backward_return_flat(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    plan: SourcePushPlan,
    *,
    expert_base: Int[Array, "Dst E"] | None = None,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
    route_indices: SourcePushBackwardFlatRouteIndices | None = None,
    implementation: SourcePushBackwardReturnImplementation = SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes | None = None,
    mesh: Mesh | None = None,
) -> SourcePushBackwardReturnOutput:
    """Return flat destination expert-major backward rows to source-owned gradients."""

    if implementation == SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_PALLAS_MGPU:
        _validate_flat_shapes(dx_expert_major, d_route_block, plan, expert_base, src_base_by_expert)
        if mesh is not None:
            if route_indices is not None:
                _validate_flat_index_request(dx_expert_major, d_route_block, route_indices)
            route_rows = _flat_route_rows(plan, expert_base, src_base_by_expert)
            return _source_push_backward_return_flat_pallas_mgpu(
                dx_expert_major,
                d_route_block,
                route_indices,
                route_rows=route_rows,
                route_shape=(plan.valid_mask.shape[0], plan.tokens_per_source, plan.topk),
                block_sizes=block_sizes,
                mesh=mesh,
            )
        if route_indices is None:
            route_indices = source_push_backward_return_flat_route_indices_jax(
                plan,
                expert_base=expert_base,
                src_base_by_expert=src_base_by_expert,
            )
        return _source_push_backward_return_flat_pallas_mgpu(
            dx_expert_major,
            d_route_block,
            route_indices,
            block_sizes=block_sizes,
        )
    if implementation != SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATION_JAX:
        raise ValueError(
            "source-push backward return implementation must be one of "
            f"{SOURCE_PUSH_BACKWARD_RETURN_IMPLEMENTATIONS}, "
            f"got {implementation!r}"
        )
    return source_push_backward_return_flat_jax(
        dx_expert_major,
        d_route_block,
        plan,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        route_indices=route_indices,
    )


def source_push_backward_return_flat_reference(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    plan: SourcePushPlan,
    *,
    expert_base: Int[Array, "Dst E"] | None = None,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
    route_indices: SourcePushBackwardFlatRouteIndices | None = None,
) -> SourcePushBackwardReturnOutput:
    """Readable JAX reference for flat expert-major backward return/combine."""

    return source_push_backward_return_flat_jax(
        dx_expert_major,
        d_route_block,
        plan,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        route_indices=route_indices,
    )


def source_push_backward_return_flat_route_buffer_jax(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    plan: SourcePushPlan,
    *,
    expert_base: Int[Array, "Dst E"] | None = None,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> SourcePushBackwardReturnRouteBuffer:
    """Return flat destination expert-major rows into a source-owned route buffer."""

    _validate_flat_shapes(dx_expert_major, d_route_block, plan, expert_base, src_base_by_expert)
    route_rows = _flat_route_rows(plan, expert_base, src_base_by_expert)
    queue_dx = dx_expert_major.at[route_rows.dst, route_rows.row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None, None)
    )
    queue_d_route = d_route_block.at[route_rows.dst, route_rows.row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    return _source_route_buffer_from_queue_rows(queue_dx, queue_d_route, route_rows, plan)


def source_push_backward_return_flat_route_indices_jax(
    plan: SourcePushPlan,
    *,
    expert_base: Int[Array, "Dst E"] | None = None,
    src_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> SourcePushBackwardFlatRouteIndices:
    """Build the source-local inverse map consumed by the flat Pallas return kernel.

    This map is a plan-derived value, not a differentiable activation. The
    production custom VJP should cache or precompute it with the other routing
    metadata instead of rebuilding it in the hot backward path.
    """

    route_rows = _flat_route_rows(plan, expert_base, src_base_by_expert)
    route_shape = (plan.valid_mask.shape[0], plan.tokens_per_source, plan.topk)
    valid_i = jnp.zeros(route_shape, dtype=jnp.int32)
    dst = jnp.zeros(route_shape, dtype=jnp.int32)
    row = jnp.zeros(route_shape, dtype=jnp.int32)

    valid_i = valid_i.at[route_rows.src, route_rows.token, route_rows.slot].add(route_rows.valid.astype(jnp.int32))
    dst = dst.at[route_rows.src, route_rows.token, route_rows.slot].add(
        jnp.where(route_rows.valid, route_rows.dst, 0),
    )
    row = row.at[route_rows.src, route_rows.token, route_rows.slot].add(
        jnp.where(route_rows.valid, route_rows.row, 0),
    )
    return SourcePushBackwardFlatRouteIndices(dst=dst, row=row, valid=valid_i > 0)


def _source_push_backward_return_flat_from_indices_jax(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    route_indices: SourcePushBackwardFlatRouteIndices,
) -> SourcePushBackwardReturnOutput:
    _validate_flat_index_request(dx_expert_major, d_route_block, route_indices)
    dx, d_route_weights = _source_push_backward_return_flat_from_indices_reference(
        dx_expert_major,
        d_route_block,
        route_indices.dst,
        route_indices.row,
        route_indices.valid.astype(jnp.int32),
    )
    return SourcePushBackwardReturnOutput(dx=dx, d_route_weights=d_route_weights)


def _source_push_backward_return_compact_pallas_mgpu(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_indices: SourcePushBackwardCompactRouteIndices | None,
    *,
    route_rows: _RouteRows | None = None,
    route_shape: tuple[int, int, int] | None = None,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> SourcePushBackwardReturnOutput:
    """Pallas source-local compact return/combine kernel.

    This entrypoint consumes the production W13 backward layout directly:
    ``dx_expert_major`` stays ``[destination, local_expert, row, hidden]``.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU backward return requires a GPU backend; use the JAX reference on CPU")
    block_sizes = SourcePushBackwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    if mesh is not None and not interpret:
        if route_indices is None:
            raise ValueError("route_indices are required for sharded compact Pallas backward return")
        _validate_compact_pallas_request(dx_expert_major, d_route_block, route_indices, block_sizes)
        return _source_push_backward_return_compact_direct_gather_mgpu(
            mesh,
            dx_expert_major,
            d_route_block,
            route_indices,
            block_sizes=block_sizes,
        )
    if route_indices is None:
        raise ValueError("route_indices are required for non-sharded Pallas backward return")
    _validate_compact_pallas_request(dx_expert_major, d_route_block, route_indices, block_sizes)
    dx, d_route_weights = _source_push_backward_return_compact_pallas_call(
        dx_expert_major,
        d_route_block,
        route_indices.dst,
        route_indices.expert,
        route_indices.row,
        route_indices.valid.astype(jnp.int32),
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )
    return SourcePushBackwardReturnOutput(dx=dx, d_route_weights=d_route_weights)


def _source_push_backward_return_compact_remote_write_route_buffer_mgpu(
    mesh: Mesh,
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_rows: _RouteRows,
    *,
    route_shape: tuple[int, int, int],
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> SourcePushBackwardReturnRouteBuffer:
    """Write destination-local compact rows directly into source-owned route buffers."""

    _validate_compact_remote_write_request(dx_expert_major, d_route_block, route_rows, block_sizes)
    source_count, _, _ = route_shape
    route_slot_valid = _source_route_slot_valid_mask(route_rows, route_shape)
    route_slot_valid = _with_source_push_sharding(route_slot_valid, SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        dx_local: Float[Array, "1 E C D"],
        d_route_local: Float[Array, "1 E C"],
        route_expert: Int[Array, "S Dst Q M"],
        route_row: Int[Array, "S Dst Q M"],
        route_token: Int[Array, "S Dst Q M"],
        route_slot: Int[Array, "S Dst Q M"],
        route_valid: Bool[Array, "S Dst Q M"],
    ) -> tuple[Float[Array, "1 T K D"], Float[Array, "1 T K"]]:
        dx_routes, d_route_weights = _source_push_backward_return_compact_remote_write_route_buffer_pallas_call(
            dx_local[0],
            d_route_local[0],
            route_expert,
            route_row,
            route_token,
            route_slot,
            route_valid.astype(jnp.int32),
            route_shape=route_shape,
            hidden_block=block_sizes.hidden_block,
        )
        return dx_routes[None, ...], d_route_weights[None, ...]

    dx_routes, d_route_weights = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(
        dx_expert_major,
        d_route_block,
        route_rows.expert,
        route_rows.row,
        route_rows.token,
        route_rows.slot,
        route_rows.valid,
    )
    dx_routes, d_route_weights = _sharded_backward_return_remote_write_completion_barrier(mesh)(
        dx_routes,
        d_route_weights,
    )
    dx_routes = jnp.where(route_slot_valid[..., None], dx_routes, jnp.zeros((), dtype=dx_routes.dtype))
    d_route_weights = jnp.where(route_slot_valid, d_route_weights, jnp.zeros((), dtype=d_route_weights.dtype))
    return SourcePushBackwardReturnRouteBuffer(dx_routes=dx_routes, d_route_weights=d_route_weights)


def _source_push_backward_return_compact_direct_gather_mgpu(
    mesh: Mesh,
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_indices: SourcePushBackwardCompactRouteIndices,
    *,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> SourcePushBackwardReturnOutput:
    """Gather compact destination rows directly into source-owned gradients."""

    _validate_compact_pallas_request(dx_expert_major, d_route_block, route_indices, block_sizes)
    route_dst = _with_source_push_sharding(route_indices.dst, SOURCE_PUSH_MESH_AXIS, None, None)
    route_expert = _with_source_push_sharding(route_indices.expert, SOURCE_PUSH_MESH_AXIS, None, None)
    route_row = _with_source_push_sharding(route_indices.row, SOURCE_PUSH_MESH_AXIS, None, None)
    route_valid = _with_source_push_sharding(route_indices.valid.astype(jnp.int32), SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        dx_local: Float[Array, "1 E C D"],
        d_route_local: Float[Array, "1 E C"],
        route_dst_local: Int[Array, "1 T K"],
        route_expert_local: Int[Array, "1 T K"],
        route_row_local: Int[Array, "1 T K"],
        route_valid_local: Int[Array, "1 T K"],
    ) -> tuple[Float[Array, "1 T D"], Float[Array, "1 T K"]]:
        dx, d_route_weights = _source_push_backward_return_compact_direct_gather_pallas_call(
            dx_local[0],
            d_route_local[0],
            route_dst_local[0],
            route_expert_local[0],
            route_row_local[0],
            route_valid_local[0],
            source_count=dx_expert_major.shape[0],
            hidden_block=block_sizes.hidden_block,
        )
        return dx[None, ...], d_route_weights[None, ...]

    dx, d_route_weights = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(
        dx_expert_major,
        d_route_block,
        route_dst,
        route_expert,
        route_row,
        route_valid,
    )
    return SourcePushBackwardReturnOutput(dx=dx, d_route_weights=d_route_weights)


def _source_push_backward_return_compact_direct_gather_pallas_call(
    dx_local: Float[Array, "E C D"],
    d_route_local: Float[Array, "E C"],
    route_dst: Int[Array, "T K"],
    route_expert: Int[Array, "T K"],
    route_row: Int[Array, "T K"],
    route_valid: Int[Array, "T K"],
    *,
    source_count: int,
    hidden_block: int,
) -> tuple[Float[Array, "T D"], Float[Array, "T K"]]:
    tokens_per_source, topk = route_valid.shape
    hidden_dim = dx_local.shape[-1]
    output_shape = (
        jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), dx_local.dtype),
        jax.ShapeDtypeStruct((tokens_per_source, topk), d_route_local.dtype),
    )
    kernel = _make_source_push_backward_return_compact_direct_gather_kernel(
        source_count=source_count,
        topk=topk,
        hidden_block=hidden_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        kernel,
        out_shape=output_shape,
        grid=(tokens_per_source, hidden_dim // hidden_block),
        grid_names=("token", "hidden_tile"),
        compiler_params=compiler_params,
    )(
        dx_local,
        d_route_local,
        route_dst,
        route_expert,
        route_row,
        route_valid,
    )


def _make_source_push_backward_return_compact_direct_gather_kernel(
    *,
    source_count: int,
    topk: int,
    hidden_block: int,
):
    dst_offsets = tuple(range(source_count))

    def kernel(
        dx_ref: Float[pl.Ref, "E C D"],
        d_route_ref: Float[pl.Ref, "E C"],
        route_dst_ref: Int[pl.Ref, "T K"],
        route_expert_ref: Int[pl.Ref, "T K"],
        route_row_ref: Int[pl.Ref, "T K"],
        route_valid_ref: Int[pl.Ref, "T K"],
        dx_out_ref: Float[pl.Ref, "T D"],
        d_route_out_ref: Float[pl.Ref, "T K"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        token = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        hidden_start = hidden_tile * hidden_block
        dx_acc = jnp.zeros((hidden_block,), dtype=jnp.float32)

        def _read_slot(slot: int):
            valid = route_valid_ref[token, slot] != 0
            dst = route_dst_ref[token, slot]
            expert = route_expert_ref[token, slot]
            row = route_row_ref[token, slot]
            dst_ordinal = (dst - rank) % source_count

            def _branch(static_dst_ordinal: int):
                def _read_branch(_) -> tuple[jax.Array, jax.Array]:
                    static_dst = (rank + static_dst_ordinal) % source_count
                    if static_dst_ordinal == 0:
                        dx_tile = dx_ref[expert, row, pl.ds(hidden_start, hidden_block)]
                        d_route = d_route_ref[expert, row]
                    else:
                        remote_dx_ref = mgpu.remote_ref(
                            dx_ref,
                            static_dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        remote_d_route_ref = mgpu.remote_ref(
                            d_route_ref,
                            static_dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        dx_tile = remote_dx_ref[expert, row, pl.ds(hidden_start, hidden_block)]
                        d_route = remote_d_route_ref[expert, row]
                    dx_tile = jnp.where(valid, dx_tile, jnp.zeros((hidden_block,), dtype=dx_ref.dtype))
                    d_route = jnp.where(valid, d_route, jnp.zeros((), dtype=d_route_ref.dtype))
                    return dx_tile, d_route

                return _read_branch

            branches = tuple(_branch(static_dst_ordinal) for static_dst_ordinal in dst_offsets)
            return lax.switch(dst_ordinal, branches, None)

        for slot in range(topk):
            dx_tile, d_route = _read_slot(slot)
            dx_acc += dx_tile.astype(jnp.float32)

            @pl.when(hidden_tile == 0)
            def _write_d_route() -> None:
                d_route_out_ref[token, slot] = d_route

        dx_out_ref[token, pl.ds(hidden_start, hidden_block)] = dx_acc.astype(dx_out_ref.dtype)

    return kernel


def _source_push_backward_return_compact_remote_write_route_buffer_pallas_call(
    dx_local: Float[Array, "E C D"],
    d_route_local: Float[Array, "E C"],
    route_expert: Int[Array, "S Dst Q M"],
    route_row: Int[Array, "S Dst Q M"],
    route_token: Int[Array, "S Dst Q M"],
    route_slot: Int[Array, "S Dst Q M"],
    route_valid: Int[Array, "S Dst Q M"],
    *,
    route_shape: tuple[int, int, int],
    hidden_block: int,
) -> tuple[Float[Array, "T K D"], Float[Array, "T K"]]:
    source_count = route_valid.shape[0]
    entries_per_dst = route_valid.shape[2]
    block_m = route_valid.shape[3]
    hidden_dim = dx_local.shape[-1]
    _, tokens_per_source, topk = route_shape
    output_shape = (
        jax.ShapeDtypeStruct((tokens_per_source, topk, hidden_dim), dx_local.dtype),
        jax.ShapeDtypeStruct((tokens_per_source, topk), d_route_local.dtype),
    )
    kernel = _make_source_push_backward_return_compact_remote_write_route_buffer_kernel(
        source_count=source_count,
        hidden_block=hidden_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        kernel,
        out_shape=output_shape,
        grid=(source_count, entries_per_dst, block_m, hidden_dim // hidden_block),
        grid_names=("src_ordinal", "entry", "row_in_block", "hidden_tile"),
        compiler_params=compiler_params,
    )(
        dx_local,
        d_route_local,
        route_expert,
        route_row,
        route_token,
        route_slot,
        route_valid,
    )


def _make_source_push_backward_return_compact_remote_write_route_buffer_kernel(
    *,
    source_count: int,
    hidden_block: int,
):
    src_offsets = tuple(range(source_count))

    def kernel(
        dx_ref: Float[pl.Ref, "E C D"],
        d_route_ref: Float[pl.Ref, "E C"],
        route_expert_ref: Int[pl.Ref, "S Dst Q M"],
        route_row_ref: Int[pl.Ref, "S Dst Q M"],
        route_token_ref: Int[pl.Ref, "S Dst Q M"],
        route_slot_ref: Int[pl.Ref, "S Dst Q M"],
        route_valid_ref: Int[pl.Ref, "S Dst Q M"],
        dx_routes_ref: Float[pl.Ref, "T K D"],
        d_route_ref_out: Float[pl.Ref, "T K"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        row_in_block = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        hidden_start = hidden_tile * hidden_block

        def _write_to_source(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % source_count
            dst_ordinal = (-static_src_ordinal) % source_count

            valid = route_valid_ref[src, dst_ordinal, entry, row_in_block] != 0

            @pl.when(valid)
            def _copy_route_slot() -> None:
                expert = route_expert_ref[src, dst_ordinal, entry, row_in_block]
                row = route_row_ref[src, dst_ordinal, entry, row_in_block]
                token = route_token_ref[src, dst_ordinal, entry, row_in_block]
                slot = route_slot_ref[src, dst_ordinal, entry, row_in_block]

                def _copy_dx_scope(dx_smem) -> None:
                    dx_smem[:] = dx_ref[expert, row, pl.ds(hidden_start, hidden_block)]
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            dx_smem,
                            dx_routes_ref.at[token, slot, pl.ds(hidden_start, hidden_block)],
                        )
                    else:
                        remote_dx_routes_ref = mgpu.remote_ref(
                            dx_routes_ref,
                            src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        mgpu.copy_smem_to_gmem(
                            dx_smem,
                            remote_dx_routes_ref.at[token, slot, pl.ds(hidden_start, hidden_block)],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    _copy_dx_scope,
                    dx_smem=mgpu.SMEM((hidden_block,), dtype=dx_routes_ref.dtype),
                )

                @pl.when(hidden_tile == 0)
                def _copy_d_route() -> None:
                    def _copy_d_route_scope(d_route_smem) -> None:
                        d_route_smem[0] = d_route_ref[expert, row]
                        mgpu.commit_smem()
                        if static_src_ordinal == 0:
                            mgpu.copy_smem_to_gmem(d_route_smem, d_route_ref_out.at[token, pl.ds(slot, 1)])
                        else:
                            remote_d_route_ref = mgpu.remote_ref(
                                d_route_ref_out,
                                src,
                                device_id_type=pl.DeviceIdType.LOGICAL,
                            )
                            mgpu.copy_smem_to_gmem(
                                d_route_smem,
                                remote_d_route_ref.at[token, pl.ds(slot, 1)],
                            )
                        mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                    pl.run_scoped(
                        _copy_d_route_scope,
                        d_route_smem=mgpu.SMEM((1,), dtype=d_route_ref_out.dtype),
                    )

        def _switch_write_to_source(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _write_branch(_) -> None:
                    _write_to_source(static_src_ordinal)

                return _write_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_write_to_source(src_ordinal)

    return kernel


def _source_push_backward_return_flat_pallas_mgpu(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    route_indices: SourcePushBackwardFlatRouteIndices | None,
    *,
    route_rows: _RouteRows | None = None,
    route_shape: tuple[int, int, int] | None = None,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> SourcePushBackwardReturnOutput:
    """Pallas source-local flat return/combine kernel.

    Without a mesh this runs the source-local interpreter/debug kernel. With a
    mesh, destination ranks remote-write each accepted row into its unique
    source/token/route slot, then sources sum those route slots locally.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU backward return requires a GPU backend; use the JAX reference on CPU")
    block_sizes = SourcePushBackwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    if mesh is not None and not interpret:
        if route_rows is None:
            raise ValueError("route_rows are required for sharded remote-write backward return")
        if route_shape is None:
            if route_indices is None:
                raise ValueError("route_shape is required when route_indices are not supplied")
            route_shape = _route_shape_from_indices(route_indices)
        if route_indices is not None:
            _validate_flat_pallas_request(dx_expert_major, d_route_block, route_indices, block_sizes)
        _validate_flat_remote_write_request(dx_expert_major, d_route_block, route_rows, block_sizes)
        buffers = _source_push_backward_return_flat_remote_write_route_buffer_mgpu(
            mesh,
            dx_expert_major,
            d_route_block,
            route_rows,
            route_shape=route_shape,
            block_sizes=block_sizes,
        )
        return SourcePushBackwardReturnOutput(
            dx=jnp.sum(buffers.dx_routes, axis=2),
            d_route_weights=buffers.d_route_weights,
        )
    if route_indices is None:
        raise ValueError("route_indices are required for non-sharded Pallas backward return")
    _validate_flat_pallas_request(dx_expert_major, d_route_block, route_indices, block_sizes)
    dx, d_route_weights = _source_push_backward_return_flat_pallas_call(
        dx_expert_major,
        d_route_block,
        route_indices.dst,
        route_indices.row,
        route_indices.valid.astype(jnp.int32),
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )
    return SourcePushBackwardReturnOutput(dx=dx, d_route_weights=d_route_weights)


def _source_push_backward_return_flat_remote_write_route_buffer_mgpu(
    mesh: Mesh,
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    route_rows: _RouteRows,
    *,
    route_shape: tuple[int, int, int],
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> SourcePushBackwardReturnRouteBuffer:
    """Write destination-local flat rows directly into source-owned route buffers."""

    _validate_flat_remote_write_request(dx_expert_major, d_route_block, route_rows, block_sizes)
    source_count, _, _ = route_shape
    route_slot_valid = _source_route_slot_valid_mask(route_rows, route_shape)
    route_slot_valid = _with_source_push_sharding(route_slot_valid, SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        dx_local: Float[Array, "1 rows D"],
        d_route_local: Float[Array, "1 rows"],
        route_row: Int[Array, "S Dst Q M"],
        route_token: Int[Array, "S Dst Q M"],
        route_slot: Int[Array, "S Dst Q M"],
        route_valid: Bool[Array, "S Dst Q M"],
    ) -> tuple[Float[Array, "1 T K D"], Float[Array, "1 T K"]]:
        dx_routes, d_route_weights = _source_push_backward_return_flat_remote_write_route_buffer_pallas_call(
            dx_local[0],
            d_route_local[0],
            route_row,
            route_token,
            route_slot,
            route_valid.astype(jnp.int32),
            route_shape=route_shape,
            hidden_block=block_sizes.hidden_block,
        )
        return dx_routes[None, ...], d_route_weights[None, ...]

    dx_routes, d_route_weights = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(
        dx_expert_major,
        d_route_block,
        route_rows.row,
        route_rows.token,
        route_rows.slot,
        route_rows.valid,
    )
    dx_routes, d_route_weights = _sharded_backward_return_remote_write_completion_barrier(mesh)(
        dx_routes,
        d_route_weights,
    )
    dx_routes = jnp.where(route_slot_valid[..., None], dx_routes, jnp.zeros((), dtype=dx_routes.dtype))
    d_route_weights = jnp.where(route_slot_valid, d_route_weights, jnp.zeros((), dtype=d_route_weights.dtype))
    return SourcePushBackwardReturnRouteBuffer(dx_routes=dx_routes, d_route_weights=d_route_weights)


def _source_push_backward_return_flat_remote_write_route_buffer_pallas_call(
    dx_local: Float[Array, "rows D"],
    d_route_local: Float[Array, "rows"],
    route_row: Int[Array, "S Dst Q M"],
    route_token: Int[Array, "S Dst Q M"],
    route_slot: Int[Array, "S Dst Q M"],
    route_valid: Int[Array, "S Dst Q M"],
    *,
    route_shape: tuple[int, int, int],
    hidden_block: int,
) -> tuple[Float[Array, "T K D"], Float[Array, "T K"]]:
    source_count = route_valid.shape[0]
    entries_per_dst = route_valid.shape[2]
    block_m = route_valid.shape[3]
    hidden_dim = dx_local.shape[-1]
    _, tokens_per_source, topk = route_shape
    output_shape = (
        jax.ShapeDtypeStruct((tokens_per_source, topk, hidden_dim), dx_local.dtype),
        jax.ShapeDtypeStruct((tokens_per_source, topk), d_route_local.dtype),
    )
    kernel = _make_source_push_backward_return_flat_remote_write_route_buffer_kernel(
        source_count=source_count,
        hidden_block=hidden_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        kernel,
        out_shape=output_shape,
        grid=(source_count, entries_per_dst, block_m, hidden_dim // hidden_block),
        grid_names=("src_ordinal", "entry", "row_in_block", "hidden_tile"),
        compiler_params=compiler_params,
    )(
        dx_local,
        d_route_local,
        route_row,
        route_token,
        route_slot,
        route_valid,
    )


def _make_source_push_backward_return_flat_remote_write_route_buffer_kernel(
    *,
    source_count: int,
    hidden_block: int,
):
    src_offsets = tuple(range(source_count))

    def kernel(
        dx_ref: Float[pl.Ref, "rows D"],
        d_route_ref: Float[pl.Ref, "rows"],
        route_row_ref: Int[pl.Ref, "S Dst Q M"],
        route_token_ref: Int[pl.Ref, "S Dst Q M"],
        route_slot_ref: Int[pl.Ref, "S Dst Q M"],
        route_valid_ref: Int[pl.Ref, "S Dst Q M"],
        dx_routes_ref: Float[pl.Ref, "T K D"],
        d_route_ref_out: Float[pl.Ref, "T K"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        row_in_block = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        hidden_start = hidden_tile * hidden_block

        def _write_to_source(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % source_count
            dst_ordinal = (-static_src_ordinal) % source_count

            valid = route_valid_ref[src, dst_ordinal, entry, row_in_block] != 0

            @pl.when(valid)
            def _copy_route_slot() -> None:
                row = route_row_ref[src, dst_ordinal, entry, row_in_block]
                token = route_token_ref[src, dst_ordinal, entry, row_in_block]
                slot = route_slot_ref[src, dst_ordinal, entry, row_in_block]

                def _copy_dx_scope(dx_smem) -> None:
                    dx_smem[:] = dx_ref[row, pl.ds(hidden_start, hidden_block)]
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            dx_smem,
                            dx_routes_ref.at[token, slot, pl.ds(hidden_start, hidden_block)],
                        )
                    else:
                        remote_dx_routes_ref = mgpu.remote_ref(
                            dx_routes_ref,
                            src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        mgpu.copy_smem_to_gmem(
                            dx_smem,
                            remote_dx_routes_ref.at[token, slot, pl.ds(hidden_start, hidden_block)],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    _copy_dx_scope,
                    dx_smem=mgpu.SMEM((hidden_block,), dtype=dx_routes_ref.dtype),
                )

                @pl.when(hidden_tile == 0)
                def _copy_d_route() -> None:
                    def _copy_d_route_scope(d_route_smem) -> None:
                        d_route_smem[0] = d_route_ref[row]
                        mgpu.commit_smem()
                        if static_src_ordinal == 0:
                            mgpu.copy_smem_to_gmem(d_route_smem, d_route_ref_out.at[token, pl.ds(slot, 1)])
                        else:
                            remote_d_route_ref = mgpu.remote_ref(
                                d_route_ref_out,
                                src,
                                device_id_type=pl.DeviceIdType.LOGICAL,
                            )
                            mgpu.copy_smem_to_gmem(
                                d_route_smem,
                                remote_d_route_ref.at[token, pl.ds(slot, 1)],
                            )
                        mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                    pl.run_scoped(
                        _copy_d_route_scope,
                        d_route_smem=mgpu.SMEM((1,), dtype=d_route_ref_out.dtype),
                    )

        def _switch_write_to_source(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _write_branch(_) -> None:
                    _write_to_source(static_src_ordinal)

                return _write_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_write_to_source(src_ordinal)

    return kernel


def _source_push_backward_return_remote_write_route_buffer_cost_estimate(
    dx_local: Array,
    d_route_local: Array,
    route_row: Array,
    route_token: Array,
    route_slot: Array,
    route_valid: Array,
    output_shape: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
) -> pl.CostEstimate:
    metadata_bytes = sum(
        int(np.prod(array.shape)) * jnp.dtype(array.dtype).itemsize
        for array in (route_row, route_token, route_slot, route_valid)
    )
    input_row_bytes = int(np.prod(dx_local.shape)) * dx_local.dtype.itemsize
    input_route_bytes = int(np.prod(d_route_local.shape)) * d_route_local.dtype.itemsize
    output_bytes = sum(int(np.prod(spec.shape)) * jnp.dtype(spec.dtype).itemsize for spec in output_shape)
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=metadata_bytes + input_row_bytes + input_route_bytes + output_bytes,
        remote_bytes_transferred=output_bytes,
    )


def _source_push_backward_return_compact_remote_write_route_buffer_cost_estimate(
    dx_local: Array,
    d_route_local: Array,
    route_expert: Array,
    route_row: Array,
    route_token: Array,
    route_slot: Array,
    route_valid: Array,
    output_shape: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
) -> pl.CostEstimate:
    metadata_bytes = sum(
        int(np.prod(array.shape)) * jnp.dtype(array.dtype).itemsize
        for array in (route_expert, route_row, route_token, route_slot, route_valid)
    )
    input_row_bytes = int(np.prod(dx_local.shape)) * dx_local.dtype.itemsize
    input_route_bytes = int(np.prod(d_route_local.shape)) * d_route_local.dtype.itemsize
    output_bytes = sum(int(np.prod(spec.shape)) * jnp.dtype(spec.dtype).itemsize for spec in output_shape)
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=metadata_bytes + input_row_bytes + input_route_bytes + output_bytes,
        remote_bytes_transferred=output_bytes,
    )


def _route_shape_from_indices(
    route_indices: SourcePushBackwardCompactRouteIndices | SourcePushBackwardFlatRouteIndices,
) -> tuple[int, int, int]:
    source_count, tokens_per_source, topk = route_indices.valid.shape
    return source_count, tokens_per_source, topk


def _sharded_backward_return_remote_write_completion_barrier(mesh: Mesh):
    """Synchronize remote route-buffer writes before source-local route summation."""

    def local_fn(
        dx_routes_local: Float[Array, "1 T K D"],
        d_route_local: Float[Array, "1 T K"],
    ) -> tuple[Float[Array, "1 T K D"], Float[Array, "1 T K"]]:
        dx_routes_local = dx_routes_local[0]
        d_route_local = d_route_local[0]
        marker = dx_routes_local[0, 0, 0].astype(jnp.float32) + d_route_local[0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = barrier - lax.optimization_barrier(barrier)
        dx_routes_local = dx_routes_local.at[0, 0, 0].add(zero.astype(dx_routes_local.dtype))
        d_route_local = d_route_local.at[0, 0].add(zero.astype(d_route_local.dtype))
        return dx_routes_local[None, ...], d_route_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )


def _source_push_backward_return_compact_pallas_call(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_row: Int[Array, "S T K"],
    route_valid: Int[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Float[Array, "S T K"]]:
    if mesh is not None and not interpret:
        raise ValueError(
            "sharded Pallas backward return uses destination remote writes; "
            "call _source_push_backward_return_compact_pallas_mgpu with route_rows"
        )

    source_count, tokens_per_source, topk = route_valid.shape
    hidden_dim = dx_expert_major.shape[-1]
    output_shape = (
        jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), dx_expert_major.dtype),
        jax.ShapeDtypeStruct((source_count, tokens_per_source, topk), d_route_block.dtype),
    )

    cost_estimate = _source_push_backward_return_compact_pallas_cost_estimate(
        dx_expert_major,
        d_route_block,
        route_dst,
        route_expert,
        route_row,
        route_valid,
        output_shape,
    )
    kernel = _make_source_push_backward_return_compact_kernel(topk=topk, hidden_block=hidden_block)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_backward_return_compact_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dx_expert_major, d_route_block, route_dst, route_expert, route_row, route_valid)


def _make_source_push_backward_return_compact_kernel(*, topk: int, hidden_block: int):
    def kernel(
        dx_ref: Float[pl.Ref, "Dst E C D"],
        d_route_ref: Float[pl.Ref, "Dst E C"],
        route_dst_ref: Int[pl.Ref, "S T K"],
        route_expert_ref: Int[pl.Ref, "S T K"],
        route_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        dx_out_ref: Float[pl.Ref, "S T D"],
        d_route_out_ref: Float[pl.Ref, "S T K"],
    ) -> None:
        src = pl.program_id(0)
        token = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        dx_acc = jnp.zeros((hidden_block,), dtype=jnp.float32)

        def _write_d_route(slot: int, valid, dst, expert, row) -> None:
            @pl.when(hidden_tile == 0)
            def _write() -> None:
                d_route_out_ref[src, token, slot] = jnp.where(
                    valid,
                    d_route_ref[dst, expert, row],
                    jnp.zeros((), dtype=d_route_ref.dtype),
                )

        for slot in range(topk):
            valid = route_valid_ref[src, token, slot] != 0
            dst = route_dst_ref[src, token, slot]
            expert = route_expert_ref[src, token, slot]
            row = route_row_ref[src, token, slot]
            dx_tile = dx_ref[dst, expert, row, pl.ds(hidden_start, hidden_block)]
            dx_acc += jnp.where(valid, dx_tile.astype(jnp.float32), jnp.zeros((hidden_block,), dtype=jnp.float32))
            _write_d_route(slot, valid, dst, expert, row)

        dx_out_ref[src, token, pl.ds(hidden_start, hidden_block)] = dx_acc.astype(dx_out_ref.dtype)

    return kernel


def _source_push_backward_return_compact_pallas_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_row: Int[Array, "S T K"],
    route_valid: Int[Array, "S T K"],
) -> tuple[Float[Array, "S T D"], Float[Array, "S T K"]]:
    valid = route_valid != 0
    safe_dst = jnp.where(valid, route_dst, 0)
    safe_expert = jnp.where(valid, route_expert, 0)
    safe_row = jnp.where(valid, route_row, 0)
    route_dx = dx_expert_major.at[safe_dst, safe_expert, safe_row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    route_dx = jnp.where(valid[..., None], route_dx, jnp.zeros((), dtype=route_dx.dtype))
    route_d = d_route_block.at[safe_dst, safe_expert, safe_row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    )
    route_d = jnp.where(valid, route_d, jnp.zeros((), dtype=route_d.dtype))
    return jnp.sum(route_dx, axis=2), route_d


def _source_push_backward_return_compact_from_indices_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    d_route_block: Float[Array, "Dst E C"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_row: Int[Array, "S T K"],
    route_valid: Int[Array, "S T K"],
) -> tuple[Float[Array, "S T D"], Float[Array, "S T K"]]:
    return _source_push_backward_return_compact_pallas_reference(
        dx_expert_major,
        d_route_block,
        route_dst,
        route_expert,
        route_row,
        route_valid,
    )


def _source_push_backward_return_compact_pallas_cost_estimate(
    dx_expert_major: Array,
    d_route_block: Array,
    route_dst: Array,
    route_expert: Array,
    route_row: Array,
    route_valid: Array,
    output_shape: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(dx_expert_major.shape, dx_expert_major.dtype),
        jax.ShapeDtypeStruct(d_route_block.shape, d_route_block.dtype),
        jax.ShapeDtypeStruct(route_dst.shape, route_dst.dtype),
        jax.ShapeDtypeStruct(route_expert.shape, route_expert.dtype),
        jax.ShapeDtypeStruct(route_row.shape, route_row.dtype),
        jax.ShapeDtypeStruct(route_valid.shape, route_valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_backward_return_compact_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_backward_return_flat_pallas_call(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    route_dst: Int[Array, "S T K"],
    route_row: Int[Array, "S T K"],
    route_valid: Int[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Float[Array, "S T K"]]:
    if mesh is not None and not interpret:
        raise ValueError(
            "sharded Pallas backward return uses destination remote writes; "
            "call _source_push_backward_return_flat_pallas_mgpu with route_rows"
        )

    source_count, tokens_per_source, topk = route_valid.shape
    hidden_dim = dx_expert_major.shape[-1]
    output_shape = (
        jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), dx_expert_major.dtype),
        jax.ShapeDtypeStruct((source_count, tokens_per_source, topk), d_route_block.dtype),
    )

    cost_estimate = _source_push_backward_return_flat_pallas_cost_estimate(
        dx_expert_major,
        d_route_block,
        route_dst,
        route_row,
        route_valid,
        output_shape,
    )
    kernel = _make_source_push_backward_return_flat_kernel(topk=topk, hidden_block=hidden_block)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_backward_return_flat_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dx_expert_major, d_route_block, route_dst, route_row, route_valid)


def _make_source_push_backward_return_flat_kernel(*, topk: int, hidden_block: int):
    def kernel(
        dx_ref: Float[pl.Ref, "Dst rows D"],
        d_route_ref: Float[pl.Ref, "Dst rows"],
        route_dst_ref: Int[pl.Ref, "S T K"],
        route_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        dx_out_ref: Float[pl.Ref, "S T D"],
        d_route_out_ref: Float[pl.Ref, "S T K"],
    ) -> None:
        src = pl.program_id(0)
        token = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        dx_acc = jnp.zeros((hidden_block,), dtype=jnp.float32)

        def _write_d_route(slot: int, valid, dst, row) -> None:
            @pl.when(hidden_tile == 0)
            def _write() -> None:
                d_route_out_ref[src, token, slot] = jnp.where(
                    valid,
                    d_route_ref[dst, row],
                    jnp.zeros((), dtype=d_route_ref.dtype),
                )

        for slot in range(topk):
            valid = route_valid_ref[src, token, slot] != 0
            dst = route_dst_ref[src, token, slot]
            row = route_row_ref[src, token, slot]
            dx_tile = dx_ref[dst, row, pl.ds(hidden_start, hidden_block)]
            dx_acc += jnp.where(valid, dx_tile.astype(jnp.float32), jnp.zeros((hidden_block,), dtype=jnp.float32))
            _write_d_route(slot, valid, dst, row)

        dx_out_ref[src, token, pl.ds(hidden_start, hidden_block)] = dx_acc.astype(dx_out_ref.dtype)

    return kernel


def _source_push_backward_return_flat_pallas_reference(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    route_dst: Int[Array, "S T K"],
    route_row: Int[Array, "S T K"],
    route_valid: Int[Array, "S T K"],
) -> tuple[Float[Array, "S T D"], Float[Array, "S T K"]]:
    valid = route_valid != 0
    safe_dst = jnp.where(valid, route_dst, 0)
    safe_row = jnp.where(valid, route_row, 0)
    route_dx = dx_expert_major.at[safe_dst, safe_row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    route_dx = jnp.where(valid[..., None], route_dx, jnp.zeros((), dtype=route_dx.dtype))
    route_d = d_route_block.at[safe_dst, safe_row].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    )
    route_d = jnp.where(valid, route_d, jnp.zeros((), dtype=route_d.dtype))
    return jnp.sum(route_dx, axis=2), route_d


def _source_push_backward_return_flat_from_indices_reference(
    dx_expert_major: Float[Array, "Dst rows D"],
    d_route_block: Float[Array, "Dst rows"],
    route_dst: Int[Array, "S T K"],
    route_row: Int[Array, "S T K"],
    route_valid: Int[Array, "S T K"],
) -> tuple[Float[Array, "S T D"], Float[Array, "S T K"]]:
    return _source_push_backward_return_flat_pallas_reference(
        dx_expert_major,
        d_route_block,
        route_dst,
        route_row,
        route_valid,
    )


def _source_push_backward_return_flat_pallas_cost_estimate(
    dx_expert_major: Array,
    d_route_block: Array,
    route_dst: Array,
    route_row: Array,
    route_valid: Array,
    output_shape: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(dx_expert_major.shape, dx_expert_major.dtype),
        jax.ShapeDtypeStruct(d_route_block.shape, d_route_block.dtype),
        jax.ShapeDtypeStruct(route_dst.shape, route_dst.dtype),
        jax.ShapeDtypeStruct(route_row.shape, route_row.dtype),
        jax.ShapeDtypeStruct(route_valid.shape, route_valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_backward_return_flat_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


class _RouteRows(NamedTuple):
    src: Int[Array, "S Dst Q M"]
    dst: Int[Array, "S Dst Q M"]
    expert: Int[Array, "S Dst Q M"]
    row: Int[Array, "S Dst Q M"]
    token: Int[Array, "S Dst Q M"]
    slot: Int[Array, "S Dst Q M"]
    valid: Bool[Array, "S Dst Q M"]


def _compact_route_rows(
    plan: SourcePushPlan,
    src_base_by_expert: Int[Array, "Dst S E"] | None,
) -> _RouteRows:
    valid = jnp.asarray(plan.valid_mask, dtype=jnp.bool_)
    ep_size, dst_ord_count, entries_per_dst, block_m = valid.shape
    src_base = plan.src_base_by_expert if src_base_by_expert is None else src_base_by_expert
    src_base = jnp.asarray(src_base, dtype=jnp.int32)

    src = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None, None]
    dst_ord = jnp.arange(dst_ord_count, dtype=jnp.int32)[None, :, None, None]
    dst = (src + dst_ord) % ep_size
    src = jnp.broadcast_to(src, valid.shape)
    dst = jnp.broadcast_to(dst, valid.shape)

    safe_expert_by_entry = jnp.maximum(jnp.asarray(plan.local_experts, dtype=jnp.int32), 0)
    row_start_by_entry = jnp.asarray(plan.local_row_starts, dtype=jnp.int32)
    src_by_entry = src[..., 0]
    dst_by_entry = dst[..., 0]
    row_base = src_base.at[dst_by_entry, src_by_entry, safe_expert_by_entry].get()
    row_offsets = jnp.arange(block_m, dtype=jnp.int32)[None, None, None, :]
    row = row_base[..., None] + row_start_by_entry[..., None] + row_offsets
    safe_expert = jnp.broadcast_to(safe_expert_by_entry[..., None], valid.shape)

    return _RouteRows(
        src=src,
        dst=dst,
        expert=jnp.where(valid, safe_expert, 0),
        row=jnp.where(valid, row, 0),
        token=jnp.maximum(jnp.asarray(plan.token_ids, dtype=jnp.int32), 0),
        slot=jnp.maximum(jnp.asarray(plan.route_slots, dtype=jnp.int32), 0),
        valid=valid,
    )


def _flat_route_rows(
    plan: SourcePushPlan,
    expert_base: Int[Array, "Dst E"] | None,
    src_base_by_expert: Int[Array, "Dst S E"] | None,
) -> _RouteRows:
    compact_rows = _compact_route_rows(plan, src_base_by_expert)
    expert_base = plan.expert_base if expert_base is None else expert_base
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    flat_row = expert_base.at[compact_rows.dst, compact_rows.expert].get() + compact_rows.row
    return _RouteRows(
        src=compact_rows.src,
        dst=compact_rows.dst,
        expert=compact_rows.expert,
        row=jnp.where(compact_rows.valid, flat_row, 0),
        token=compact_rows.token,
        slot=compact_rows.slot,
        valid=compact_rows.valid,
    )


def _source_route_buffer_from_queue_rows(
    queue_dx: Float[Array, "S Dst Q M D"],
    queue_d_route: Float[Array, "S Dst Q M"],
    route_rows: _RouteRows,
    plan: SourcePushPlan,
) -> SourcePushBackwardReturnRouteBuffer:
    valid = route_rows.valid
    queue_dx = jnp.where(valid[..., None], queue_dx, jnp.zeros((), dtype=queue_dx.dtype))
    queue_d_route = jnp.where(valid, queue_d_route, jnp.zeros((), dtype=queue_d_route.dtype))

    dx_routes = jnp.zeros(
        (valid.shape[0], plan.tokens_per_source, plan.topk, queue_dx.shape[-1]),
        dtype=queue_dx.dtype,
    )
    dx_routes = dx_routes.at[route_rows.src, route_rows.token, route_rows.slot].add(
        queue_dx,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )
    d_route_weights = jnp.zeros((valid.shape[0], plan.tokens_per_source, plan.topk), dtype=queue_d_route.dtype)
    d_route_weights = d_route_weights.at[route_rows.src, route_rows.token, route_rows.slot].add(
        queue_d_route,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    return SourcePushBackwardReturnRouteBuffer(dx_routes, d_route_weights)


def _source_route_slot_valid_mask(
    route_rows: _RouteRows,
    route_shape: tuple[int, int, int],
) -> Bool[Array, "S T K"]:
    route_slot_valid = jnp.zeros(route_shape, dtype=jnp.int32)
    route_slot_valid = route_slot_valid.at[route_rows.src, route_rows.token, route_rows.slot].add(
        route_rows.valid.astype(jnp.int32),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    return route_slot_valid > 0


def _validate_compact_shapes(
    dx_expert_major: Array,
    d_route_block: Array,
    plan: SourcePushPlan,
    src_base_by_expert: Array | None,
) -> None:
    if dx_expert_major.ndim != 4:
        raise ValueError(f"dx_expert_major must have shape [dst, expert, row, D], got {dx_expert_major.shape}")
    if d_route_block.shape != dx_expert_major.shape[:3]:
        raise ValueError(
            f"d_route_block shape {d_route_block.shape} must match dx_expert_major rows {dx_expert_major.shape[:3]}"
        )
    _validate_plan_layout_shapes(plan, dx_expert_major.shape[0], dx_expert_major.shape[1], src_base_by_expert)


def _validate_compact_dx_shape(
    dx_expert_major: Array,
    plan: SourcePushPlan,
    src_base_by_expert: Array | None,
) -> None:
    if dx_expert_major.ndim != 4:
        raise ValueError(f"dx_expert_major must have shape [dst, expert, row, D], got {dx_expert_major.shape}")
    _validate_plan_layout_shapes(plan, dx_expert_major.shape[0], dx_expert_major.shape[1], src_base_by_expert)


def _validate_flat_shapes(
    dx_expert_major: Array,
    d_route_block: Array,
    plan: SourcePushPlan,
    expert_base: Array | None,
    src_base_by_expert: Array | None,
) -> None:
    if dx_expert_major.ndim != 3:
        raise ValueError(f"dx_expert_major must have shape [dst, row, D], got {dx_expert_major.shape}")
    if d_route_block.shape != dx_expert_major.shape[:2]:
        raise ValueError(
            f"d_route_block shape {d_route_block.shape} must match dx_expert_major rows {dx_expert_major.shape[:2]}"
        )
    experts_per_rank = plan.expert_base.shape[1] if expert_base is None else expert_base.shape[1]
    _validate_plan_layout_shapes(plan, dx_expert_major.shape[0], experts_per_rank, src_base_by_expert)
    if expert_base is not None and expert_base.shape != (dx_expert_major.shape[0], experts_per_rank):
        raise ValueError(
            f"expert_base shape {expert_base.shape} must be {(dx_expert_major.shape[0], experts_per_rank)}"
        )


def _validate_flat_pallas_request(
    dx_expert_major: Array,
    d_route_block: Array,
    route_indices: SourcePushBackwardFlatRouteIndices,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> None:
    _validate_flat_index_request(dx_expert_major, d_route_block, route_indices)
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dx_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dx_expert_major hidden dimension must be divisible by hidden_block; "
            f"got D={dx_expert_major.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )


def _validate_compact_pallas_request(
    dx_expert_major: Array,
    d_route_block: Array,
    route_indices: SourcePushBackwardCompactRouteIndices,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> None:
    _validate_compact_index_request(dx_expert_major, d_route_block, route_indices)
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dx_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dx_expert_major hidden dimension must be divisible by hidden_block; "
            f"got D={dx_expert_major.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )


def _validate_compact_index_request(
    dx_expert_major: Array,
    d_route_block: Array,
    route_indices: SourcePushBackwardCompactRouteIndices,
) -> None:
    if dx_expert_major.ndim != 4:
        raise ValueError(f"dx_expert_major must have shape [dst, expert, row, D], got {dx_expert_major.shape}")
    if d_route_block.shape != dx_expert_major.shape[:3]:
        raise ValueError(
            f"d_route_block shape {d_route_block.shape} must match dx_expert_major rows {dx_expert_major.shape[:3]}"
        )
    if (
        route_indices.dst.shape != route_indices.expert.shape
        or route_indices.dst.shape != route_indices.row.shape
        or route_indices.dst.shape != route_indices.valid.shape
    ):
        raise ValueError(
            "route index shapes must match; got "
            f"dst={route_indices.dst.shape}, expert={route_indices.expert.shape}, "
            f"row={route_indices.row.shape}, valid={route_indices.valid.shape}"
        )
    if route_indices.dst.ndim != 3:
        raise ValueError(f"route indices must have shape [S, T, K], got {route_indices.dst.shape}")


def _validate_flat_index_request(
    dx_expert_major: Array,
    d_route_block: Array,
    route_indices: SourcePushBackwardFlatRouteIndices,
) -> None:
    if dx_expert_major.ndim != 3:
        raise ValueError(f"dx_expert_major must have shape [dst, row, D], got {dx_expert_major.shape}")
    if d_route_block.shape != dx_expert_major.shape[:2]:
        raise ValueError(
            f"d_route_block shape {d_route_block.shape} must match dx_expert_major rows {dx_expert_major.shape[:2]}"
        )
    if route_indices.dst.shape != route_indices.row.shape or route_indices.dst.shape != route_indices.valid.shape:
        raise ValueError(
            "route index shapes must match; got "
            f"dst={route_indices.dst.shape}, row={route_indices.row.shape}, valid={route_indices.valid.shape}"
        )
    if route_indices.dst.ndim != 3:
        raise ValueError(f"route indices must have shape [S, T, K], got {route_indices.dst.shape}")


def _validate_flat_remote_write_request(
    dx_expert_major: Array,
    d_route_block: Array,
    route_rows: _RouteRows,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> None:
    if dx_expert_major.ndim != 3:
        raise ValueError(f"dx_expert_major must have shape [dst, row, D], got {dx_expert_major.shape}")
    if d_route_block.shape != dx_expert_major.shape[:2]:
        raise ValueError(
            f"d_route_block shape {d_route_block.shape} must match dx_expert_major rows {dx_expert_major.shape[:2]}"
        )
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dx_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dx_expert_major hidden dimension must be divisible by hidden_block; "
            f"got D={dx_expert_major.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )
    row_shape = route_rows.row.shape
    if (
        route_rows.token.shape != row_shape
        or route_rows.slot.shape != row_shape
        or route_rows.valid.shape != row_shape
    ):
        raise ValueError(
            "route row shapes must match; got "
            f"row={route_rows.row.shape}, token={route_rows.token.shape}, "
            f"slot={route_rows.slot.shape}, valid={route_rows.valid.shape}"
        )
    if len(row_shape) != 4 or row_shape[0] != dx_expert_major.shape[0] or row_shape[1] != dx_expert_major.shape[0]:
        raise ValueError(
            "route rows must have shape [S, Dst, Q, M] with S == Dst == dx_expert_major.shape[0], "
            f"got route rows {row_shape} and dx_expert_major {dx_expert_major.shape}"
        )


def _validate_compact_remote_write_request(
    dx_expert_major: Array,
    d_route_block: Array,
    route_rows: _RouteRows,
    block_sizes: SourcePushBackwardReturnPallasBlockSizes,
) -> None:
    if dx_expert_major.ndim != 4:
        raise ValueError(f"dx_expert_major must have shape [dst, expert, row, D], got {dx_expert_major.shape}")
    if d_route_block.shape != dx_expert_major.shape[:3]:
        raise ValueError(
            f"d_route_block shape {d_route_block.shape} must match dx_expert_major rows {dx_expert_major.shape[:3]}"
        )
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dx_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dx_expert_major hidden dimension must be divisible by hidden_block; "
            f"got D={dx_expert_major.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )
    row_shape = route_rows.row.shape
    if (
        route_rows.expert.shape != row_shape
        or route_rows.token.shape != row_shape
        or route_rows.slot.shape != row_shape
        or route_rows.valid.shape != row_shape
    ):
        raise ValueError(
            "route row shapes must match; got "
            f"expert={route_rows.expert.shape}, row={route_rows.row.shape}, "
            f"token={route_rows.token.shape}, slot={route_rows.slot.shape}, valid={route_rows.valid.shape}"
        )
    if len(row_shape) != 4 or row_shape[0] != dx_expert_major.shape[0] or row_shape[1] != dx_expert_major.shape[0]:
        raise ValueError(
            "route rows must have shape [S, Dst, Q, M] with S == Dst == dx_expert_major.shape[0], "
            f"got route rows {row_shape} and dx_expert_major {dx_expert_major.shape}"
        )


def _validate_plan_layout_shapes(
    plan: SourcePushPlan,
    ep_size: int,
    experts_per_rank: int,
    src_base_by_expert: Array | None,
) -> None:
    if plan.valid_mask.shape[0] != ep_size or plan.valid_mask.shape[1] != ep_size:
        raise ValueError(f"plan valid_mask shape {plan.valid_mask.shape} is incompatible with ep_size={ep_size}")
    src_base_shape = plan.src_base_by_expert.shape if src_base_by_expert is None else src_base_by_expert.shape
    expected_src_base_shape = (ep_size, ep_size, experts_per_rank)
    if src_base_shape != expected_src_base_shape:
        raise ValueError(f"src_base_by_expert shape {src_base_shape} must be {expected_src_base_shape}")
