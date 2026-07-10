# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas scaffolding for slot-free source-push semantic backward transport.

Implemented stages:

* source dy expansion:
  ``dy[S,T,H] -> dy_route[Dst,E,C,H]`` and ``dcombine[S,T,K]``
* dx combine:
  ``dx_route[Dst,E,C,H] -> dx[S,T,H]``

The transport kernels use the expert-major row contract shared by the semantic
W2 and W13 matmul boundaries: rows for ``(destination, local_expert)`` are
packed by source using ``src_base_by_expert``. Pair-flat compatibility wrappers
remain available for scaffolds that have not moved to expert-major buffers yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_backward_w2 import SOURCE_PUSH_MESH_AXIS
from levanter.grug._moe.source_push_plan import (
    SourcePushSemanticQueueMetadata,
    SourcePushSemanticPlan,
    _source_push_semantic_expert_row_indices_from_metadata_jax,
    source_push_semantic_backward_source_expand_jax,
    source_push_semantic_dx_combine_jax,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_queue_metadata_jax,
    source_push_semantic_reverse_route_jax,
)


DEFAULT_SEMANTIC_BACKWARD_ROW_BLOCK = 128
DEFAULT_SEMANTIC_BACKWARD_HIDDEN_BLOCK = 128
DEFAULT_SEMANTIC_DX_RETURN_ROW_BLOCK = 64
DEFAULT_SEMANTIC_DX_RETURN_HIDDEN_BLOCK = 256
MAX_SEMANTIC_DX_QUEUE_COPY_DIM = 256


@dataclass(frozen=True, slots=True)
class SourcePushSemanticBackwardPallasBlockSizes:
    """Tile sizes for pair-flat semantic backward transport kernels."""

    row_block: int = DEFAULT_SEMANTIC_BACKWARD_ROW_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_BACKWARD_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticBackwardPallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushSemanticDxReturnPallasBlockSizes:
    """Tile sizes for queue-based expert-major dx return and source combine."""

    row_block: int = DEFAULT_SEMANTIC_DX_RETURN_ROW_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_DX_RETURN_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticDxReturnPallasBlockSizes":
        return cls()


def source_push_semantic_swiglu_backward_expert_major_jax(
    dh_expert: Float[Array, "Dst E C I"],
    z_expert: Float[Array, "Dst E C twoI"],
    valid: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C twoI"]:
    """Reference SwiGLU backward over expert-major rows using fp32 math."""

    gate, up = jnp.split(z_expert.astype(jnp.float32), 2, axis=-1)
    dh_expert = dh_expert.astype(jnp.float32)
    sigmoid_gate = jax.nn.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    dz_gate = dh_expert * up * d_silu_gate
    dz_up = dh_expert * silu_gate
    dz_expert = jnp.concatenate([dz_gate, dz_up], axis=-1)
    return jnp.where(valid[..., None], dz_expert, jnp.zeros((), dtype=dz_expert.dtype))


def source_push_semantic_backward_source_expand_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int | None = None,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Expand source-token dy to expert-major route dy and route-weight gradients."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push backward expand requires a GPU backend")
    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    rows_per_expert_capacity = _rows_per_expert_capacity(plan, rows_per_expert_capacity, block_sizes.row_block)
    _validate_backward_source_expand_request(dy, route_y, plan, rows_per_expert_capacity, block_sizes)
    dy_route = _source_push_semantic_backward_dy_route_expert_major_pallas_call(
        dy,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        dst_offset=jnp.asarray(0, dtype=jnp.int32),
        local_dst_count=plan.xcounts.shape[1],
        rows_per_expert_capacity=rows_per_expert_capacity,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    dcombine = _source_push_semantic_backward_dcombine_pallas_call(
        dy,
        route_y,
        plan.token_ids,
        plan.route_slots,
        plan.valid_mask,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    return dy_route, dcombine


def source_push_semantic_backward_source_expand_from_expert_major_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Expand dy and compute route-weight gradients from expert-major route outputs."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push backward expand requires a GPU backend")
    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_backward_source_expand_expert_major_route_request(dy, route_y_expert, plan, block_sizes)
    if mesh is not None and not interpret:
        return _source_push_semantic_backward_source_expand_expert_major_sharded_mgpu_kernel(
            mesh,
            dy,
            route_y_expert,
            plan.token_ids,
            plan.route_slots,
            plan.route_weights,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            topk=plan.topk,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    dy_route = _source_push_semantic_backward_dy_route_expert_major_pallas_call(
        dy,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        dst_offset=jnp.asarray(0, dtype=jnp.int32),
        local_dst_count=plan.xcounts.shape[1],
        rows_per_expert_capacity=route_y_expert.shape[2],
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    dcombine = _source_push_semantic_backward_dcombine_expert_major_pallas_call(
        dy,
        route_y_expert,
        plan.token_ids,
        plan.route_slots,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        dst_offset=jnp.asarray(0, dtype=jnp.int32),
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    return dy_route, dcombine


def source_push_semantic_backward_source_expand_from_expert_major_owner_sharded_dcombine_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Expand dy while returning dcombine sharded by the source-token axis."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_backward_source_expand_expert_major_route_request(dy, route_y_expert, plan, block_sizes)
    if interpret:
        return source_push_semantic_backward_source_expand_from_expert_major_jax(dy, route_y_expert, plan)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic owner-sharded source-expand requires a GPU backend")
    return _source_push_semantic_backward_source_expand_expert_major_owner_sharded_dcombine_mgpu_kernel(
        mesh,
        dy,
        route_y_expert,
        plan.token_ids,
        plan.route_slots,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_backward_dy_route_expert_major_jax(
    dy: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> Float[Array, "Dst E C H"]:
    """Reference dy routing into destination expert-major rows."""

    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    safe_tokens = jnp.maximum(plan.token_ids, 0)
    dy_pair = dy.at[source_index, safe_tokens].get()
    dy_pair = jnp.where(plan.valid_mask[..., None], dy_pair, jnp.zeros((), dtype=dy_pair.dtype))
    dy_route_pair = dy_pair * plan.route_weights[..., None].astype(dy_pair.dtype)
    dy_route = _semantic_pair_to_expert_major_jax(
        dy_route_pair,
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    return dy_route


def source_push_semantic_backward_dy_route_destination_pull_pallas_mgpu(
    dy: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int | None = None,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Route dy with one destination-owned local gather kernel per expert rank."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    rows_per_expert_capacity = _rows_per_expert_capacity(plan, rows_per_expert_capacity, block_sizes.row_block)
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    _validate_backward_dy_route_source_push_request(dy, plan, rows_per_expert_capacity, block_sizes)
    if interpret:
        return source_push_semantic_backward_dy_route_expert_major_jax(
            dy,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=source_row_base_by_expert,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic destination-pull dy-route requires a GPU backend")
    if mesh is None:
        raise ValueError("destination-pull dy-route requires an expert mesh")
    destination_count = plan.assignment_ids.shape[1]
    mesh_size = mesh.shape[SOURCE_PUSH_MESH_AXIS]
    if destination_count % mesh_size:
        raise ValueError(f"destination count {destination_count} must divide expert mesh size {mesh_size}")
    return _source_push_semantic_backward_dy_route_destination_pull_sharded_mgpu_kernel(
        mesh,
        dy,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )


def source_push_semantic_backward_dy_route_source_push_pallas_mgpu(
    dy: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int | None = None,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Route source-token dy to destination expert-major rows with source-owned remote writes."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    rows_per_expert_capacity = _rows_per_expert_capacity(plan, rows_per_expert_capacity, block_sizes.row_block)
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    _validate_backward_dy_route_source_push_request(dy, plan, rows_per_expert_capacity, block_sizes)
    if interpret:
        return source_push_semantic_backward_dy_route_expert_major_jax(
            dy,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=source_row_base_by_expert,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic dy-route source-push requires a GPU backend")
    if mesh is None:
        raise ValueError("source-push dy-route requires a mesh so remote destination writes have rank ownership")
    return _source_push_semantic_backward_dy_route_source_push_sharded_mgpu_kernel(
        mesh,
        dy,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )


def source_push_semantic_backward_source_expand_from_expert_major_source_push_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Expand dy using source-owned dy routing plus the existing expert-major dcombine path."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_backward_source_expand_expert_major_route_request(dy, route_y_expert, plan, block_sizes)
    if interpret:
        return source_push_semantic_backward_source_expand_from_expert_major_jax(dy, route_y_expert, plan)
    dy_route = source_push_semantic_backward_dy_route_source_push_pallas_mgpu(
        dy,
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    if mesh is None:
        raise ValueError("source-push source-expand requires a mesh for the dcombine shard map")
    dcombine = _source_push_semantic_backward_dcombine_expert_major_sharded_mgpu_kernel(
        mesh,
        dy,
        route_y_expert,
        plan.token_ids,
        plan.route_slots,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    return dy_route, dcombine


def source_push_semantic_backward_dcombine_from_return_queue_jax(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    plan: SourcePushSemanticPlan,
    *,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> Float[Array, "S T K"]:
    """Reference dcombine from the saved source-owned forward return queue."""

    metadata = _resolve_source_queue_metadata(
        plan,
        queue,
        row_block=return_y.shape[3],
        entries_per_dst=return_y.shape[2],
    )
    source_index = jnp.arange(dy.shape[0], dtype=jnp.int32)[:, None, None]
    route_y_get = return_y.at[
        source_index,
        metadata.route_dst_ordinal,
        metadata.route_entry,
        metadata.route_queue_row,
    ]
    route_y_sharding = _source_major_out_sharding_from_named_input(return_y, rank=4)
    if route_y_sharding is None:
        route_y = route_y_get.get(mode="clip")
    else:
        route_y = route_y_get.get(mode="clip", out_sharding=route_y_sharding)
    dcombine = jnp.sum(
        dy[:, :, None, :].astype(jnp.float32) * route_y.astype(jnp.float32),
        axis=-1,
        dtype=jnp.float32,
    )
    return jnp.where(metadata.route_valid, dcombine, jnp.zeros((), dtype=dcombine.dtype))


def source_push_semantic_backward_source_expand_from_return_queue_jax(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Reference source expansion using only the saved forward return queue."""

    dy_route = source_push_semantic_backward_dy_route_expert_major_jax(
        dy,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        source_row_base_by_expert=source_row_base_by_expert,
    )
    dcombine = source_push_semantic_backward_dcombine_from_return_queue_jax(
        dy,
        return_y,
        plan,
        queue=queue,
    )
    return dy_route, dcombine


def source_push_semantic_backward_dcombine_from_return_queue_pallas_mgpu(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T K"]:
    """Compute dcombine locally from the saved source-owned forward return queue."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    metadata = _resolve_source_queue_metadata(
        plan,
        queue,
        row_block=return_y.shape[3],
        entries_per_dst=return_y.shape[2],
    )
    _validate_backward_dcombine_return_queue_request(dy, return_y, plan, metadata, block_sizes)
    if interpret:
        return source_push_semantic_backward_dcombine_from_return_queue_jax(
            dy,
            return_y,
            plan,
            queue=metadata,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic return-queue dcombine requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_backward_dcombine_return_queue_pallas_call(
            dy,
            return_y,
            metadata.route_dst_ordinal,
            metadata.route_entry,
            metadata.route_queue_row,
            metadata.route_valid,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )

    source_sharding_3d = _semantic_source_major_sharding(mesh, rank=3)
    return_y = jax.sharding.reshard(return_y, _semantic_source_major_sharding(mesh, rank=5))
    dy = jax.sharding.reshard(dy, source_sharding_3d)
    route_dst_ordinal = jax.sharding.reshard(metadata.route_dst_ordinal, source_sharding_3d)
    route_entry = jax.sharding.reshard(metadata.route_entry, source_sharding_3d)
    route_queue_row = jax.sharding.reshard(metadata.route_queue_row, source_sharding_3d)
    route_valid = jax.sharding.reshard(metadata.route_valid, source_sharding_3d)
    return _source_push_semantic_backward_dcombine_return_queue_sharded_mgpu_kernel(
        mesh,
        dy,
        return_y,
        route_dst_ordinal,
        route_entry,
        route_queue_row,
        route_valid,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_backward_source_expand_from_return_queue_pallas_mgpu(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int | None = None,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Expand dy and compute dcombine without rebuilding expert-major forward outputs."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    rows_per_expert_capacity = _rows_per_expert_capacity(plan, rows_per_expert_capacity, block_sizes.row_block)
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    queue = _resolve_source_queue_metadata(
        plan,
        queue,
        row_block=return_y.shape[3],
        entries_per_dst=return_y.shape[2],
    )
    _validate_backward_dcombine_return_queue_request(dy, return_y, plan, queue, block_sizes)
    _validate_backward_dy_route_source_push_request(dy, plan, rows_per_expert_capacity, block_sizes)
    if interpret:
        return source_push_semantic_backward_source_expand_from_return_queue_jax(
            dy,
            return_y,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
            source_row_base_by_expert=source_row_base_by_expert,
            queue=queue,
        )
    dy_route = source_push_semantic_backward_dy_route_destination_pull_pallas_mgpu(
        dy,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        source_row_base_by_expert=source_row_base_by_expert,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    dcombine = source_push_semantic_backward_dcombine_from_return_queue_pallas_mgpu(
        dy,
        return_y,
        plan,
        block_sizes=block_sizes,
        queue=queue,
        interpret=interpret,
        mesh=mesh,
    )
    return dy_route, dcombine


def source_push_semantic_backward_dcombine_source_gather_jax(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S T K"]:
    """Reference dcombine using source-owned reverse-route gathers."""

    reverse_route = source_push_semantic_reverse_route_jax(plan)
    safe_dst = jnp.where(reverse_route.route_valid, reverse_route.route_dst, 0)
    safe_expert = jnp.where(reverse_route.route_valid, reverse_route.route_expert, 0)
    safe_row = jnp.where(reverse_route.route_valid, reverse_route.route_expert_row, 0)
    route_slot_sharding = _replicated_out_sharding_from_named_input(route_y_expert, rank=4)
    route_y_get = route_y_expert.at[safe_dst, safe_expert, safe_row]
    if route_slot_sharding is None:
        route_y_by_slot = route_y_get.get(mode="fill", fill_value=0)
    else:
        route_y_by_slot = route_y_get.get(mode="fill", fill_value=0, out_sharding=route_slot_sharding)
    dcombine = jnp.sum(dy[:, :, None, :].astype(jnp.float32) * route_y_by_slot.astype(jnp.float32), axis=-1)
    return jnp.where(reverse_route.route_valid, dcombine, jnp.zeros((), dtype=dcombine.dtype))


def source_push_semantic_backward_dcombine_source_gather_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T K"]:
    """Compute dcombine by source-owned reverse-route gather from expert-major route outputs."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_backward_dcombine_source_gather_request(dy, route_y_expert, plan, block_sizes)
    reverse_route = source_push_semantic_reverse_route_jax(plan)
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic dcombine reverse-route gather requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_backward_dcombine_source_gather_pallas_call(
            dy,
            route_y_expert,
            reverse_route.route_dst,
            reverse_route.route_expert,
            reverse_route.route_expert_row,
            reverse_route.route_valid,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_backward_dcombine_source_gather_sharded_mgpu_kernel(
        mesh,
        dy,
        route_y_expert,
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
        reverse_route.route_valid,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_backward_source_expand_from_expert_major_source_gather_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """Expand dy with source-owned dy-route writes and source-owned dcombine gathers."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_backward_source_expand_expert_major_route_request(dy, route_y_expert, plan, block_sizes)
    if interpret:
        return source_push_semantic_backward_source_expand_from_expert_major_jax(dy, route_y_expert, plan)
    dy_route = source_push_semantic_backward_dy_route_source_push_pallas_mgpu(
        dy,
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    dcombine = source_push_semantic_backward_dcombine_source_gather_pallas_mgpu(
        dy,
        route_y_expert,
        plan,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    return dy_route, dcombine


def source_push_semantic_dx_combine_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Combine expert-major route-level dx rows back to source-token dx."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    dx = source_push_semantic_dx_return_source_gather_pallas_mgpu(
        dx_route,
        plan,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    return dx


def source_push_semantic_dx_return_expert_major_jax(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "S T H"], Float[Array, "S T K H"]]:
    """JAX reference for returning expert-major route dx to source route slots."""

    dx_pair = source_push_semantic_expert_major_to_pair_jax(dx_route, plan)
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    source_index = jnp.broadcast_to(source_index, plan.assignment_ids.shape)
    token_ids = jnp.where(plan.valid_mask, plan.token_ids, plan.tokens_per_source)
    route_slots = jnp.where(plan.valid_mask, plan.route_slots, plan.topk)
    dx_pair = jnp.where(plan.valid_mask[..., None], dx_pair, jnp.zeros((), dtype=dx_pair.dtype))
    dx_by_slot = jnp.zeros(
        (
            plan.assignment_ids.shape[0],
            plan.tokens_per_source,
            plan.topk,
            dx_route.shape[-1],
        ),
        dtype=dx_route.dtype,
    )
    dx_by_slot = dx_by_slot.at[source_index, token_ids, route_slots].set(dx_pair, mode="drop")
    return jnp.sum(dx_by_slot, axis=2), dx_by_slot


def source_push_semantic_dx_return_source_gather_jax(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
) -> Float[Array, "S T H"]:
    """JAX reference for source-owned reverse-route dx return/combine."""

    reverse_route = source_push_semantic_reverse_route_jax(
        plan,
        source_row_base_by_expert=source_row_base_by_expert,
    )
    safe_dst = jnp.where(reverse_route.route_valid, reverse_route.route_dst, 0)
    safe_expert = jnp.where(reverse_route.route_valid, reverse_route.route_expert, 0)
    safe_row = jnp.where(reverse_route.route_valid, reverse_route.route_expert_row, 0)
    route_slot_sharding = _replicated_out_sharding_from_named_input(dx_route, rank=4)
    dx_get = dx_route.at[safe_dst, safe_expert, safe_row]
    if route_slot_sharding is None:
        dx_by_slot = dx_get.get(mode="fill", fill_value=0)
    else:
        dx_by_slot = dx_get.get(mode="fill", fill_value=0, out_sharding=route_slot_sharding)
    dx_by_slot = jnp.where(reverse_route.route_valid[..., None], dx_by_slot, jnp.zeros((), dtype=dx_route.dtype))
    return jnp.sum(dx_by_slot, axis=2)


def _replicated_out_sharding_from_named_input(
    value: Array,
    *,
    rank: int,
) -> jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None:
    spec = P(*(None for _ in range(rank)))
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        return spec
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return None
    return jax.sharding.NamedSharding(sharding.mesh, spec)


def _source_major_out_sharding_from_named_input(
    value: Array,
    *,
    rank: int,
) -> jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None:
    spec = P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(rank - 1)))
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        return spec
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return None
    return jax.sharding.NamedSharding(sharding.mesh, spec)


def source_push_semantic_dx_return_direct_to_source_reference_jax(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes | None = None,
    route_buffer_dtype: jnp.dtype = jnp.bfloat16,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> Float[Array, "S DstOrd Q M H"]:
    """Reference expert-major dx copy into source-owned return queues."""

    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    _validate_dx_queue_return_request(dx_route, plan, block_sizes, route_buffer_dtype)
    metadata = _source_push_semantic_dx_queue_metadata_jax(
        plan,
        row_block=block_sizes.row_block,
        queue=queue,
    )
    source_count, destination_count = plan.assignment_ids.shape[:2]
    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None]
    actual_dst = (source_index + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(metadata.local_expert, 0)
    expert_source_base = source_row_base_by_expert.at[actual_dst, source_index, safe_expert].get()
    expert_row_start = expert_source_base + metadata.local_row_start
    queue_row = jnp.arange(block_sizes.row_block, dtype=jnp.int32)[None, None, None, :]
    expert_row = expert_row_start[..., None] + queue_row
    safe_expert_row = jnp.minimum(expert_row, dx_route.shape[2] - 1)
    return_dx_get = dx_route.at[
        actual_dst[..., None],
        safe_expert[..., None],
        safe_expert_row,
    ]
    out_sharding = _source_major_out_sharding_from_named_input(dx_route, rank=5)
    if out_sharding is None:
        return_dx = return_dx_get.get()
    else:
        return_dx = return_dx_get.get(out_sharding=out_sharding)
    valid = queue_row < metadata.valid_rows[..., None]
    return jnp.where(valid[..., None], return_dx, jnp.zeros((), dtype=return_dx.dtype)).astype(route_buffer_dtype)


def source_push_semantic_dx_combine_source_queue_reference_jax(
    return_dx: Float[Array, "S DstOrd Q M H"],
    metadata: SourcePushSemanticQueueMetadata,
    *,
    output_dtype: jnp.dtype = jnp.bfloat16,
) -> Float[Array, "S T H"]:
    """Reference source-local K reduction from a direct-return dx queue."""

    source_count = metadata.route_dst_ordinal.shape[0]
    source_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    route_dx_get = return_dx.at[
        source_index,
        metadata.route_dst_ordinal,
        metadata.route_entry,
        metadata.route_queue_row,
    ]
    route_dx_sharding = _source_major_out_sharding_from_named_input(return_dx, rank=4)
    if route_dx_sharding is None:
        route_dx = route_dx_get.get(mode="clip")
    else:
        route_dx = route_dx_get.get(mode="clip", out_sharding=route_dx_sharding)
    route_dx = jnp.where(
        metadata.route_valid[..., None],
        route_dx.astype(jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
    )
    return jnp.sum(route_dx, axis=2, dtype=jnp.float32).astype(output_dtype)


def source_push_semantic_dx_return_direct_to_source_combine_reference_jax(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes | None = None,
    output_dtype: jnp.dtype = jnp.bfloat16,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> Float[Array, "S T H"]:
    """JAX reference for direct queue dX return followed by source-local reduction."""

    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    queue = _source_push_semantic_dx_queue_metadata_jax(
        plan,
        row_block=block_sizes.row_block,
        queue=queue,
    )
    return_dx = source_push_semantic_dx_return_direct_to_source_reference_jax(
        dx_route,
        plan,
        source_row_base_by_expert=source_row_base_by_expert,
        block_sizes=block_sizes,
        queue=queue,
    )
    return source_push_semantic_dx_combine_source_queue_reference_jax(
        return_dx,
        queue,
        output_dtype=output_dtype,
    )


def source_push_semantic_dx_return_direct_to_source_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes | None = None,
    route_buffer_dtype: jnp.dtype = jnp.bfloat16,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S DstOrd Q M H"]:
    """Copy destination expert-major dx blocks directly to source-owned queues."""

    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    _validate_dx_queue_return_request(dx_route, plan, block_sizes, route_buffer_dtype)
    if interpret:
        return source_push_semantic_dx_return_direct_to_source_reference_jax(
            dx_route,
            plan,
            source_row_base_by_expert=source_row_base_by_expert,
            block_sizes=block_sizes,
            route_buffer_dtype=route_buffer_dtype,
            queue=queue,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic direct dx return requires a GPU backend")
    if mesh is None:
        raise ValueError("mesh must be provided for non-interpreted direct dx return")

    metadata = _source_push_semantic_dx_queue_metadata_jax(
        plan,
        row_block=block_sizes.row_block,
        queue=queue,
    )
    recv_local_expert, recv_expert_row_start, recv_valid_rows = _source_push_semantic_dx_receive_metadata_jax(
        plan,
        metadata,
        source_row_base_by_expert,
    )
    # The direct queue is a bf16 transport. Materialize that contract before
    # the kernel so the local GMEM -> SMEM leg can use TMA rather than a large
    # register-mediated dynamic slice with an in-kernel dtype conversion.
    dx_route = dx_route.astype(route_buffer_dtype)
    dx_route = jax.sharding.reshard(dx_route, _semantic_destination_major_sharding(mesh, rank=4))
    recv_local_expert = jax.sharding.reshard(
        recv_local_expert,
        _semantic_destination_major_sharding(mesh, rank=3),
    )
    recv_expert_row_start = jax.sharding.reshard(
        recv_expert_row_start,
        _semantic_destination_major_sharding(mesh, rank=3),
    )
    recv_valid_rows = jax.sharding.reshard(
        recv_valid_rows,
        _semantic_destination_major_sharding(mesh, rank=3),
    )
    return _source_push_semantic_dx_return_direct_to_source_sharded_mgpu_kernel(
        mesh,
        dx_route,
        recv_local_expert,
        recv_expert_row_start,
        recv_valid_rows,
        source_count=plan.assignment_ids.shape[0],
        entries_per_dst=metadata.entries_per_dst,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_dtype=route_buffer_dtype,
    )


def source_push_semantic_dx_combine_source_queue_pallas_mgpu(
    return_dx: Float[Array, "S DstOrd Q M H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes | None = None,
    output_dtype: jnp.dtype = jnp.bfloat16,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Sum K returned route gradients in fp32 on each source owner."""

    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    metadata = _resolve_source_queue_metadata(
        plan,
        queue,
        row_block=block_sizes.row_block,
        entries_per_dst=return_dx.shape[2],
    )
    _validate_dx_queue_combine_request(return_dx, plan, metadata, block_sizes)
    if interpret:
        return source_push_semantic_dx_combine_source_queue_reference_jax(
            return_dx,
            metadata,
            output_dtype=output_dtype,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-local dx queue combine requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_dx_combine_source_queue_pallas_call(
            return_dx,
            metadata.route_dst_ordinal,
            metadata.route_entry,
            metadata.route_queue_row,
            metadata.route_valid,
            token_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            output_dtype=output_dtype,
            interpret=interpret,
        )

    source_sharding_5d = _semantic_source_major_sharding(mesh, rank=5)
    source_sharding_3d = _semantic_source_major_sharding(mesh, rank=3)
    return_dx = jax.sharding.reshard(return_dx, source_sharding_5d)
    route_dst_ordinal = jax.sharding.reshard(metadata.route_dst_ordinal, source_sharding_3d)
    route_entry = jax.sharding.reshard(metadata.route_entry, source_sharding_3d)
    route_queue_row = jax.sharding.reshard(metadata.route_queue_row, source_sharding_3d)
    route_valid = jax.sharding.reshard(metadata.route_valid, source_sharding_3d)
    return _source_push_semantic_dx_combine_source_queue_sharded_mgpu_kernel(
        mesh,
        return_dx,
        route_dst_ordinal,
        route_entry,
        route_queue_row,
        route_valid,
        token_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_dtype=output_dtype,
        interpret=interpret,
    )


def source_push_semantic_dx_return_direct_to_source_combine_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes | None = None,
    output_dtype: jnp.dtype = jnp.bfloat16,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Direct queue dx return followed by source-local unweighted K reduction."""

    block_sizes = SourcePushSemanticDxReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    queue = _source_push_semantic_dx_queue_metadata_jax(
        plan,
        row_block=block_sizes.row_block,
        queue=queue,
    )
    return_dx = source_push_semantic_dx_return_direct_to_source_pallas_mgpu(
        dx_route,
        plan,
        source_row_base_by_expert=source_row_base_by_expert,
        block_sizes=block_sizes,
        route_buffer_dtype=jnp.bfloat16,
        queue=queue,
        interpret=interpret,
        mesh=mesh,
    )
    return source_push_semantic_dx_combine_source_queue_pallas_mgpu(
        return_dx,
        plan,
        block_sizes=block_sizes,
        output_dtype=output_dtype,
        queue=queue,
        interpret=interpret,
        mesh=mesh,
    )


def source_push_semantic_dx_return_source_gather_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Return expert-major route-level dx by source-owned reverse-route gather."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_source_gather_request(dx_route, plan, block_sizes)
    reverse_route = source_push_semantic_reverse_route_jax(
        plan,
        source_row_base_by_expert=source_row_base_by_expert,
    )
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push dx reverse-route gather requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_dx_return_source_gather_pallas_call(
            dx_route,
            reverse_route.route_dst,
            reverse_route.route_expert,
            reverse_route.route_expert_row,
            reverse_route.route_valid,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_dx_return_source_gather_sharded_mgpu_kernel(
        mesh,
        dx_route,
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
        reverse_route.route_valid,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_dx_return_source_gather_owner_sharded_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh,
) -> Float[Array, "S T H"]:
    """Return route-level dx by reverse-route gather with the source-token axis sharded."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_source_gather_request(dx_route, plan, block_sizes)
    reverse_route = source_push_semantic_reverse_route_jax(plan)
    if interpret:
        return source_push_semantic_dx_return_source_gather_jax(dx_route, plan)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push owner-sharded dx gather requires a GPU backend")
    return _source_push_semantic_dx_return_source_gather_owner_sharded_mgpu_kernel(
        mesh,
        dx_route,
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
        reverse_route.route_valid,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_dx_return_remote_source_gather_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Return dx by having source-token owners pull remote expert-major route rows."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_source_gather_request(dx_route, plan, block_sizes)
    if plan.tokens_per_source % block_sizes.row_block:
        raise ValueError(
            f"tokens_per_source={plan.tokens_per_source} must be divisible by row_block={block_sizes.row_block}"
        )
    if interpret:
        return source_push_semantic_dx_return_source_gather_jax(dx_route, plan)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push remote dx gather requires a GPU backend")
    if mesh is None:
        raise ValueError("mesh must be provided for non-interpreted remote dx gather")
    reverse_route = source_push_semantic_reverse_route_jax(plan)
    return _source_push_semantic_dx_return_remote_source_gather_sharded_mgpu_kernel(
        mesh,
        dx_route,
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
        reverse_route.route_valid,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_dx_return_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T H"], Float[Array, "S T K H"]]:
    """Return expert-major route-level dx rows to source route slots, then sum slots."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push dx return requires a GPU backend")
    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_combine_request(dx_route, plan, block_sizes)
    if mesh is None:
        dx_by_slot = _source_push_semantic_dx_return_expert_major_pallas_call(
            dx_route,
            plan.token_ids,
            plan.route_slots,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            tokens_per_source=plan.tokens_per_source,
            topk=plan.topk,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    else:
        dx_by_slot = _source_push_semantic_dx_return_expert_major_sharded_mgpu_kernel(
            mesh,
            dx_route,
            plan.token_ids,
            plan.route_slots,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            tokens_per_source=plan.tokens_per_source,
            topk=plan.topk,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return jnp.sum(dx_by_slot, axis=2), dx_by_slot


def source_push_semantic_dx_return_slot_reduce_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Benchmark diagnostic: materialize dx route slots, then reduce slots exactly."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push dx slot-reduce diagnostic requires a GPU backend")
    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_combine_request(dx_route, plan, block_sizes)
    if mesh is None:
        dx_by_slot = _source_push_semantic_dx_return_expert_major_pallas_call(
            dx_route,
            plan.token_ids,
            plan.route_slots,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            tokens_per_source=plan.tokens_per_source,
            topk=plan.topk,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
        return jnp.sum(dx_by_slot, axis=2)
    return _source_push_semantic_dx_return_slot_reduce_expert_major_sharded_mgpu_kernel(
        mesh,
        dx_route,
        plan.token_ids,
        plan.route_slots,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_dx_return_sum_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Return expert-major route-level dx rows and sum directly into source tokens."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_combine_request(dx_route, plan, block_sizes)
    if interpret:
        dx, _dx_by_slot = source_push_semantic_dx_return_expert_major_jax(dx_route, plan)
        return dx
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push dx return requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_dx_return_sum_expert_major_pallas_call(
            dx_route,
            plan.token_ids,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            tokens_per_source=plan.tokens_per_source,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_dx_return_sum_expert_major_sharded_mgpu_kernel(
        mesh,
        dx_route,
        plan.token_ids,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_dx_return_copy_only_pallas_mgpu(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Diagnostic dx return traversal that omits the source-token atomic scatter."""

    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_combine_request(dx_route, plan, block_sizes)
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push dx return copy diagnostic requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_dx_return_copy_only_expert_major_pallas_call(
            dx_route,
            plan.token_ids,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_dx_return_copy_only_expert_major_sharded_mgpu_kernel(
        mesh,
        dx_route,
        plan.token_ids,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_backward_source_expand_expert_major_jax(
    dy: Float[Array, "S T H"],
    route_y: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """JAX reference for expert-major semantic backward source expansion."""

    dy_pair, dcombine = source_push_semantic_backward_source_expand_jax(dy, route_y, plan)
    dy_route, _valid = source_push_semantic_pair_to_expert_major_jax(
        dy_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    return dy_route, dcombine


def source_push_semantic_backward_source_expand_from_expert_major_jax(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    """JAX reference for source expansion using expert-major route outputs."""

    route_y = source_push_semantic_expert_major_to_pair_jax(route_y_expert, plan)
    return source_push_semantic_backward_source_expand_expert_major_jax(
        dy,
        route_y,
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
    )


def source_push_semantic_dx_combine_expert_major_jax(
    dx_route: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S T H"]:
    """JAX reference for combining expert-major route-level dx."""

    dx_pair = source_push_semantic_expert_major_to_pair_jax(dx_route, plan)
    return source_push_semantic_dx_combine_jax(dx_pair, plan)


def source_push_semantic_backward_source_expand_pair_pallas_mgpu(
    dy: Float[Array, "S T H"],
    route_y: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "S Dst R H"], Float[Array, "S T K"]]:
    """Compatibility wrapper for pair-flat semantic source expansion."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push backward expand requires a GPU backend")
    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_backward_source_expand_pair_request(dy, route_y, plan, block_sizes)
    dy_route = _source_push_semantic_backward_dy_route_pair_pallas_call(
        dy,
        plan.token_ids,
        plan.route_weights,
        plan.valid_mask,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    dcombine = _source_push_semantic_backward_dcombine_pallas_call(
        dy,
        route_y,
        plan.token_ids,
        plan.route_slots,
        plan.valid_mask,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )
    return dy_route, dcombine


def source_push_semantic_dx_combine_pair_pallas_mgpu(
    dx_pair: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "S T H"]:
    """Compatibility wrapper for pair-flat semantic dx combine."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push dx combine requires a GPU backend")
    block_sizes = SourcePushSemanticBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx_combine_pair_request(dx_pair, plan, block_sizes)
    return _source_push_semantic_dx_combine_pair_pallas_call(
        dx_pair,
        plan.token_ids,
        plan.valid_mask,
        tokens_per_source=plan.tokens_per_source,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def _source_push_semantic_backward_dy_route_expert_major_pallas_call(
    dy: Float[Array, "S T H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    dst_offset: Int[Array, ""],
    local_dst_count: int,
    *,
    rows_per_expert_capacity: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "Dst E C H"]:
    _source_count, _global_dst_count, experts_per_rank = xcounts.shape
    hidden_dim = dy.shape[-1]
    output_dtype = dy.dtype
    output_shape = jax.ShapeDtypeStruct(
        (local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim),
        output_dtype,
    )
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dy_route_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            local_dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        interpret=interpret,
        name="source_push_semantic_backward_dy_route_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dy,
            token_ids,
            route_weights,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
        ),
    )(dy, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert, dst_offset)


def _source_push_semantic_backward_dy_route_source_push_pallas_call(
    dy: Float[Array, "T H"],
    token_ids: Int[Array, "Dst R"],
    route_weights: Float[Array, "Dst R"],
    xcounts: Int[Array, "Dst E"],
    pair_expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    source_rank: Int[Array, ""],
    dy_route_init: Float[Array, "Dst E C H"],
    *,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    global_dst_count, experts_per_rank = xcounts.shape
    _local_dst_count, _experts_per_rank, rows_per_expert_capacity, hidden_dim = dy_route_init.shape
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dy_route_source_push_kernel(
            global_dst_count=global_dst_count,
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=dy_route_init.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=jax.ShapeDtypeStruct(dy_route_init.shape, dy_route_init.dtype),
        grid=(
            global_dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        input_output_aliases={7: 0},
        name="source_push_semantic_backward_dy_route_source_push_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dy,
            token_ids,
            route_weights,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            source_rank,
            dy_route_init,
            dy_route_init,
        ),
    )(dy, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert, source_rank, dy_route_init)


def _source_push_semantic_backward_dy_route_source_push_sharded_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    experts_per_rank = xcounts.shape[-1]
    hidden_dim = dy.shape[-1]
    local_dst_count = max(1, xcounts.shape[1] // mesh.devices.size)
    if local_dst_count != 1 or xcounts.shape[1] != mesh.devices.size:
        raise ValueError(
            "source-push dy-route currently expects one destination shard per rank, got "
            f"{xcounts.shape[1]=} and {mesh.devices.size=}"
        )
    source_sharding_3d = jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    replicated_3d = jax.sharding.NamedSharding(mesh, P(None, None, None))
    dy = jax.sharding.reshard(dy, source_sharding_3d)
    token_ids = jax.sharding.reshard(token_ids, source_sharding_3d)
    route_weights = jax.sharding.reshard(route_weights, source_sharding_3d)
    xcounts = jax.sharding.reshard(xcounts, source_sharding_3d)
    pair_expert_base = jax.sharding.reshard(pair_expert_base, source_sharding_3d)
    src_base_by_expert = jax.sharding.reshard(src_base_by_expert, replicated_3d)
    init = jnp.zeros((xcounts.shape[1], experts_per_rank, rows_per_expert_capacity, hidden_dim), dtype=dy.dtype)
    init = jax.device_put(init, jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)))

    def local_fn(
        dy_local: Float[Array, "1 T H"],
        token_ids_local: Int[Array, "1 Dst R"],
        route_weights_local: Float[Array, "1 Dst R"],
        xcounts_local: Int[Array, "1 Dst E"],
        pair_expert_base_local: Int[Array, "1 Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
        dy_route_local: Float[Array, "DstLocal E C H"],
    ) -> Float[Array, "DstLocal E C H"]:
        source_rank = jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS), dtype=jnp.int32)
        return _source_push_semantic_backward_dy_route_source_push_pallas_call(
            dy_local[0],
            token_ids_local[0],
            route_weights_local[0],
            xcounts_local[0],
            pair_expert_base_local[0],
            src_base_by_expert_global,
            source_rank,
            dy_route_local,
            row_block=row_block,
            hidden_block=hidden_block,
        )

    routed = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dy, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert, init)
    return _source_push_semantic_backward_dy_route_remote_write_barrier(mesh)(routed)


def _make_source_push_semantic_backward_dy_route_source_push_kernel(
    *,
    global_dst_count: int,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dy_ref: Float[pl.Ref, "T H"],
        token_ids_ref: Int[pl.Ref, "Dst R"],
        route_weights_ref: Float[pl.Ref, "Dst R"],
        xcounts_ref: Int[pl.Ref, "Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        source_rank_ref: Int[pl.Ref, ""],
        _dy_route_init_ref: Float[pl.Ref, "DstLocal E C H"],
        dy_route_ref: Float[pl.Ref, "DstLocal E C H"],
    ) -> None:
        source_rank = source_rank_ref[()]
        dst_program = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        local_row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)

        for static_dst in range(global_dst_count):
            is_program_dst = dst_program == static_dst
            valid_rows = xcounts_ref[static_dst, expert]

            @pl.when(is_program_dst & (valid_rows > local_row_start))
            def _write_static_dst() -> None:
                pair_base = pair_expert_base_ref[static_dst, expert]
                expert_row_base = src_base_by_expert_ref[static_dst, source_rank, expert]
                remote_dy_route_ref = mgpu.remote_ref(
                    dy_route_ref,
                    static_dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

                @pl.loop(0, row_block)
                def _row_loop(row_offset) -> None:
                    local_row = local_row_start + row_offset
                    row_valid = local_row < valid_rows
                    pair_row = pair_base + local_row
                    safe_token = jnp.maximum(token_ids_ref[static_dst, pair_row], 0)
                    dy_tile = dy_ref[safe_token, pl.ds(hidden_start, hidden_block)].astype(output_dtype)
                    weight = route_weights_ref[static_dst, pair_row].astype(output_dtype)
                    out_tile = jnp.where(row_valid, dy_tile * weight, zero_tile)
                    remote_dy_route_ref[
                        0,
                        expert,
                        expert_row_base + local_row,
                        pl.ds(hidden_start, hidden_block),
                    ] = out_tile

    return kernel


def _source_push_semantic_backward_dy_route_remote_write_barrier(mesh: Mesh):
    """Synchronize after semantic dy-route source-push remote writes."""

    def local_fn(routed_local: Float[Array, "1 E C H"]) -> Float[Array, "1 E C H"]:
        marker = routed_local[0, 0, 0, 0].astype(jnp.float32)
        barrier = jax.lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = (barrier - jax.lax.optimization_barrier(barrier)).astype(routed_local.dtype)
        return routed_local.at[0, 0, 0, 0].add(zero)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )


def _source_push_semantic_backward_dcombine_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    def local_fn(
        dy_global: Float[Array, "S T H"],
        route_y_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "S T K"]:
        partial = _source_push_semantic_backward_dcombine_expert_major_pallas_call(
            dy_global,
            route_y_local,
            token_ids_global,
            route_slots_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=_destination_offset_for_local_shard(route_y_local.shape[0]),
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(partial, SOURCE_PUSH_MESH_AXIS)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(dy, route_y_expert, token_ids, route_slots, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_backward_dcombine_source_gather_pallas_call(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    dst_offset: Int[Array, ""],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    source_count, tokens_per_source, topk = route_dst.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, topk), jnp.float32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dcombine_source_gather_kernel(hidden_block=hidden_block),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source, topk),
        interpret=interpret,
        name="source_push_semantic_backward_dcombine_source_gather_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dy,
            route_y_expert,
            route_dst,
            route_expert,
            route_expert_row,
            route_valid,
            dst_offset,
            output_shape,
        ),
    )(dy, route_y_expert, route_dst, route_expert, route_expert_row, route_valid.astype(jnp.int32), dst_offset)


def _source_push_semantic_backward_dcombine_source_gather_sharded_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    def local_fn(
        dy_global: Float[Array, "S T H"],
        route_y_local: Float[Array, "Dst E C H"],
        route_dst_global: Int[Array, "S T K"],
        route_expert_global: Int[Array, "S T K"],
        route_expert_row_global: Int[Array, "S T K"],
        route_valid_global: Bool[Array, "S T K"],
    ) -> Float[Array, "S T K"]:
        partial = _source_push_semantic_backward_dcombine_source_gather_pallas_call(
            dy_global,
            route_y_local,
            route_dst_global,
            route_expert_global,
            route_expert_row_global,
            route_valid_global,
            dst_offset=_destination_offset_for_local_shard(route_y_local.shape[0]),
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(partial, SOURCE_PUSH_MESH_AXIS)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(dy, route_y_expert, route_dst, route_expert, route_expert_row, route_valid)


def _source_push_semantic_backward_dcombine_return_queue_pallas_call(
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    route_dst_ordinal: Int[Array, "S T K"],
    route_entry: Int[Array, "S T K"],
    route_queue_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    source_count, tokens_per_source, topk = route_dst_ordinal.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, topk), jnp.float32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dcombine_return_queue_kernel(hidden_block=hidden_block),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source, topk),
        interpret=interpret,
        name="source_push_semantic_backward_dcombine_return_queue_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dy,
            return_y,
            route_dst_ordinal,
            route_entry,
            route_queue_row,
            route_valid,
            output_shape,
        ),
    )(
        dy,
        return_y,
        route_dst_ordinal,
        route_entry,
        route_queue_row,
        route_valid.astype(jnp.int32),
    )


def _source_push_semantic_backward_dcombine_return_queue_local_pallas_call(
    dy: Float[Array, "T H"],
    return_y: Float[Array, "DstOrd Q M H"],
    route_dst_ordinal: Int[Array, "T K"],
    route_entry: Int[Array, "T K"],
    route_queue_row: Int[Array, "T K"],
    route_valid: Bool[Array, "T K"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "T K"]:
    tokens_per_source, topk = route_dst_ordinal.shape
    output_shape = jax.ShapeDtypeStruct((tokens_per_source, topk), jnp.float32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dcombine_return_queue_local_kernel(hidden_block=hidden_block),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(tokens_per_source, topk),
        interpret=interpret,
        name="source_push_semantic_backward_dcombine_return_queue_local_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dy,
            return_y,
            route_dst_ordinal,
            route_entry,
            route_queue_row,
            route_valid,
            output_shape,
        ),
    )(
        dy,
        return_y,
        route_dst_ordinal,
        route_entry,
        route_queue_row,
        route_valid.astype(jnp.int32),
    )


def _source_push_semantic_backward_dcombine_return_queue_sharded_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    return_y: Float[Array, "S DstOrd Q M H"],
    route_dst_ordinal: Int[Array, "S T K"],
    route_entry: Int[Array, "S T K"],
    route_queue_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    def local_fn(
        dy_local: Float[Array, "1 T H"],
        return_y_local: Float[Array, "1 DstOrd Q M H"],
        route_dst_ordinal_local: Int[Array, "1 T K"],
        route_entry_local: Int[Array, "1 T K"],
        route_queue_row_local: Int[Array, "1 T K"],
        route_valid_local: Bool[Array, "1 T K"],
    ) -> Float[Array, "1 T K"]:
        dcombine = _source_push_semantic_backward_dcombine_return_queue_local_pallas_call(
            dy_local[0],
            return_y_local[0],
            route_dst_ordinal_local[0],
            route_entry_local[0],
            route_queue_row_local[0],
            route_valid_local[0],
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return dcombine[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(dy, return_y, route_dst_ordinal, route_entry, route_queue_row, route_valid)


def _source_push_semantic_backward_dy_route_pair_pallas_call(
    dy: Float[Array, "S T H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S Dst R H"]:
    source_count, dst_count, rows_per_pair = token_ids.shape
    hidden_dim = dy.shape[-1]
    output_dtype = dy.dtype
    output_shape = jax.ShapeDtypeStruct((*token_ids.shape, hidden_dim), output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dy_route_pair_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, dst_count, rows_per_pair // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_backward_dy_route_pair_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(dy, token_ids, route_weights, valid_mask, output_shape),
    )(dy, token_ids, route_weights, valid_mask.astype(jnp.int32))


def _source_push_semantic_backward_dcombine_pallas_call(
    dy: Float[Array, "S T H"],
    route_y: Float[Array, "S Dst R H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    source_count, dst_count, rows_per_pair = token_ids.shape
    output_dtype = jnp.dtype(jnp.float32)
    output_shape = jax.ShapeDtypeStruct((source_count, dy.shape[1], topk), output_dtype)
    dcombine_init = jnp.zeros(output_shape.shape, dtype=output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dcombine_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, dst_count, rows_per_pair // row_block),
        input_output_aliases={5: 0},
        interpret=interpret,
        name="source_push_semantic_backward_dcombine_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(dy, route_y, token_ids, route_slots, valid_mask, output_shape),
    )(dy, route_y, token_ids, route_slots, valid_mask.astype(jnp.int32), dcombine_init)


def _source_push_semantic_backward_dcombine_expert_major_pallas_call(
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    dst_offset: Int[Array, ""],
    *,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K"]:
    source_count, _global_dst_count, experts_per_rank = xcounts.shape
    local_dst_count, _experts_per_rank, rows_per_expert_capacity, _hidden_dim = route_y_expert.shape
    output_dtype = jnp.dtype(jnp.float32)
    output_shape = jax.ShapeDtypeStruct((source_count, dy.shape[1], topk), output_dtype)
    dcombine_init = jnp.zeros(output_shape.shape, dtype=output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_backward_dcombine_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(local_dst_count, experts_per_rank, rows_per_expert_capacity // row_block),
        input_output_aliases={8: 0},
        interpret=interpret,
        name="source_push_semantic_backward_dcombine_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dy,
            route_y_expert,
            token_ids,
            route_slots,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
        ),
    )(
        dy,
        route_y_expert,
        token_ids,
        route_slots,
        xcounts,
        pair_expert_base,
        src_base_by_expert,
        dst_offset,
        dcombine_init,
    )


def _source_push_semantic_backward_dy_route_destination_pull_sharded_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    source_row_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    destination_count = xcounts.shape[1]
    local_destination_count = destination_count // mesh.shape[SOURCE_PUSH_MESH_AXIS]

    def local_fn(
        dy_global: Float[Array, "S T H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        source_row_base_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "Dst E C H"]:
        return _source_push_semantic_backward_dy_route_expert_major_pallas_call(
            dy_global,
            token_ids_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            source_row_base_global,
            dst_offset=_destination_offset_for_local_shard(local_destination_count),
            local_dst_count=local_destination_count,
            rows_per_expert_capacity=rows_per_expert_capacity,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=False,
        )

    def replicated(rank: int) -> NamedSharding:
        return NamedSharding(mesh, P(*(None for _ in range(rank))))

    dy = jax.sharding.reshard(dy, replicated(3))
    token_ids = jax.sharding.reshard(token_ids, replicated(3))
    route_weights = jax.sharding.reshard(route_weights, replicated(3))
    xcounts = jax.sharding.reshard(xcounts, replicated(3))
    pair_expert_base = jax.sharding.reshard(pair_expert_base, replicated(3))
    source_row_base_by_expert = jax.sharding.reshard(source_row_base_by_expert, replicated(3))
    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(P(None, None, None),) * 6,
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dy, token_ids, route_weights, xcounts, pair_expert_base, source_row_base_by_expert)


def _source_push_semantic_backward_source_expand_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    rows_per_expert_capacity = route_y_expert.shape[2]

    def local_fn(
        dy_global: Float[Array, "S T H"],
        route_y_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
        dst_offset = _destination_offset_for_local_shard(route_y_local.shape[0])
        dy_route_local = _source_push_semantic_backward_dy_route_expert_major_pallas_call(
            dy_global,
            token_ids_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=dst_offset,
            local_dst_count=route_y_local.shape[0],
            rows_per_expert_capacity=rows_per_expert_capacity,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        dcombine_partial = _source_push_semantic_backward_dcombine_expert_major_pallas_call(
            dy_global,
            route_y_local,
            token_ids_global,
            route_slots_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=dst_offset,
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return dy_route_local, jax.lax.psum(dcombine_partial, SOURCE_PUSH_MESH_AXIS)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=(P(SOURCE_PUSH_MESH_AXIS, None, None, None), P(None, None, None)),
        check_vma=False,
    )(dy, route_y_expert, token_ids, route_slots, route_weights, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_backward_source_expand_expert_major_owner_sharded_dcombine_mgpu_kernel(
    mesh: Mesh,
    dy: Float[Array, "S T H"],
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "S T K"]]:
    rows_per_expert_capacity = route_y_expert.shape[2]

    def local_fn(
        dy_global: Float[Array, "S T H"],
        route_y_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> tuple[Float[Array, "Dst E C H"], Float[Array, "1 T K"]]:
        dst_offset = _destination_offset_for_local_shard(route_y_local.shape[0])
        dy_route_local = _source_push_semantic_backward_dy_route_expert_major_pallas_call(
            dy_global,
            token_ids_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=dst_offset,
            local_dst_count=route_y_local.shape[0],
            rows_per_expert_capacity=rows_per_expert_capacity,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        dcombine_partial = _source_push_semantic_backward_dcombine_expert_major_pallas_call(
            dy_global,
            route_y_local,
            token_ids_global,
            route_slots_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=dst_offset,
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        dcombine_shard = jax.lax.psum_scatter(
            dcombine_partial,
            SOURCE_PUSH_MESH_AXIS,
            scatter_dimension=0,
            tiled=True,
        )
        return dy_route_local, dcombine_shard

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=(P(SOURCE_PUSH_MESH_AXIS, None, None, None), P(SOURCE_PUSH_MESH_AXIS, None, None)),
        check_vma=False,
    )(dy, route_y_expert, token_ids, route_slots, route_weights, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_dx_return_source_gather_pallas_call(
    dx_route: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    dst_offset: Int[Array, ""],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    source_count, tokens_per_source, _topk = route_dst.shape
    local_dst_count, _experts_per_rank, _rows_per_expert_capacity, hidden_dim = dx_route.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), dx_route.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_return_source_gather_kernel(hidden_block=hidden_block),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_dx_return_source_gather_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dx_route,
            route_dst,
            route_expert,
            route_expert_row,
            route_valid,
            dst_offset,
            output_shape,
        ),
    )(dx_route, route_dst, route_expert, route_expert_row, route_valid.astype(jnp.int32), dst_offset)


def _source_push_semantic_dx_return_remote_source_gather_pallas_call(
    dx_route: Float[Array, "DstLocal E C H"],
    route_dst: Int[Array, "T K"],
    route_expert: Int[Array, "T K"],
    route_expert_row: Int[Array, "T K"],
    route_valid: Bool[Array, "T K"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "T H"]:
    tokens_per_source, topk = route_dst.shape
    _local_dst_count, _experts_per_rank, _rows_per_expert_capacity, hidden_dim = dx_route.shape
    output_shape = jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), dx_route.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_return_remote_source_gather_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(tokens_per_source // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_dx_return_remote_source_gather_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dx_route,
            route_dst,
            route_expert,
            route_expert_row,
            route_valid,
            output_shape,
        ),
    )(dx_route, route_dst, route_expert, route_expert_row, route_valid)


def _source_push_semantic_dx_return_source_gather_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        dx_route_local: Float[Array, "Dst E C H"],
        route_dst_global: Int[Array, "S T K"],
        route_expert_global: Int[Array, "S T K"],
        route_expert_row_global: Int[Array, "S T K"],
        route_valid_global: Bool[Array, "S T K"],
    ) -> Float[Array, "S T H"]:
        partial = _source_push_semantic_dx_return_source_gather_pallas_call(
            dx_route_local,
            route_dst_global,
            route_expert_global,
            route_expert_row_global,
            route_valid_global,
            dst_offset=_destination_offset_for_local_shard(dx_route_local.shape[0]),
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(partial, SOURCE_PUSH_MESH_AXIS)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(dx_route, route_dst, route_expert, route_expert_row, route_valid)


def _source_push_semantic_dx_return_source_gather_owner_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        dx_route_local: Float[Array, "Dst E C H"],
        route_dst_global: Int[Array, "S T K"],
        route_expert_global: Int[Array, "S T K"],
        route_expert_row_global: Int[Array, "S T K"],
        route_valid_global: Bool[Array, "S T K"],
    ) -> Float[Array, "1 T H"]:
        partial = _source_push_semantic_dx_return_source_gather_pallas_call(
            dx_route_local,
            route_dst_global,
            route_expert_global,
            route_expert_row_global,
            route_valid_global,
            dst_offset=_destination_offset_for_local_shard(dx_route_local.shape[0]),
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum_scatter(partial, SOURCE_PUSH_MESH_AXIS, scatter_dimension=0, tiled=True)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(dx_route, route_dst, route_expert, route_expert_row, route_valid)


def _source_push_semantic_dx_return_remote_source_gather_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    dx_route = jax.sharding.reshard(
        dx_route,
        jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )
    source_sharding_3d = jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    route_dst = jax.sharding.reshard(route_dst, source_sharding_3d)
    route_expert = jax.sharding.reshard(route_expert, source_sharding_3d)
    route_expert_row = jax.sharding.reshard(route_expert_row, source_sharding_3d)
    route_valid = jax.sharding.reshard(route_valid, source_sharding_3d)

    def local_fn(
        dx_route_local: Float[Array, "1 E C H"],
        route_dst_local: Int[Array, "1 T K"],
        route_expert_local: Int[Array, "1 T K"],
        route_expert_row_local: Int[Array, "1 T K"],
        route_valid_local: Bool[Array, "1 T K"],
    ) -> Float[Array, "1 T H"]:
        source_dx = _source_push_semantic_dx_return_remote_source_gather_pallas_call(
            dx_route_local,
            route_dst_local[0],
            route_expert_local[0],
            route_expert_row_local[0],
            route_valid_local[0],
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return source_dx[None, :, :]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(dx_route, route_dst, route_expert, route_expert_row, route_valid)


def _source_push_semantic_dx_return_expert_major_pallas_call(
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    dst_offset: Int[Array, ""],
    *,
    tokens_per_source: int,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K H"]:
    source_count, _global_dst_count, experts_per_rank = xcounts.shape
    local_dst_count, _experts_per_rank, rows_per_expert_capacity, hidden_dim = dx_route.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, topk, hidden_dim), dx_route.dtype)
    dx_by_slot_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_return_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(local_dst_count, experts_per_rank, rows_per_expert_capacity // row_block, hidden_dim // hidden_block),
        input_output_aliases={7: 0},
        interpret=interpret,
        name="source_push_semantic_dx_return_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dx_route,
            token_ids,
            route_slots,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
            output_shape,
        ),
    )(dx_route, token_ids, route_slots, xcounts, pair_expert_base, src_base_by_expert, dst_offset, dx_by_slot_init)


def _source_push_semantic_dx_return_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    tokens_per_source: int,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T K H"]:
    def local_fn(
        dx_route_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "1 S T K H"]:
        partial = _source_push_semantic_dx_return_expert_major_pallas_call(
            dx_route_local,
            token_ids_global,
            route_slots_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=_destination_offset_for_local_shard(dx_route_local.shape[0]),
            tokens_per_source=tokens_per_source,
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return partial[None, ...]

    partials_by_destination = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(dx_route, token_ids, route_slots, xcounts, pair_expert_base, src_base_by_expert)
    return jnp.sum(partials_by_destination, axis=0)


def _source_push_semantic_dx_return_slot_reduce_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    tokens_per_source: int,
    topk: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        dx_route_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "S T H"]:
        partial_slots = _source_push_semantic_dx_return_expert_major_pallas_call(
            dx_route_local,
            token_ids_global,
            route_slots_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=_destination_offset_for_local_shard(dx_route_local.shape[0]),
            tokens_per_source=tokens_per_source,
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(jnp.sum(partial_slots, axis=2), SOURCE_PUSH_MESH_AXIS)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(dx_route, token_ids, route_slots, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_dx_return_sum_expert_major_pallas_call(
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    dst_offset: Int[Array, ""],
    *,
    tokens_per_source: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    source_count, _global_dst_count, experts_per_rank = xcounts.shape
    local_dst_count, _experts_per_rank, rows_per_expert_capacity, hidden_dim = dx_route.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), jnp.float32)
    dx_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_return_sum_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(local_dst_count, experts_per_rank, rows_per_expert_capacity // row_block, hidden_dim // hidden_block),
        input_output_aliases={6: 0},
        interpret=interpret,
        name="source_push_semantic_dx_return_sum_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dx_route,
            token_ids,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
            output_shape,
        ),
    )(dx_route, token_ids, xcounts, pair_expert_base, src_base_by_expert, dst_offset, dx_init)


def _source_push_semantic_dx_return_sum_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    tokens_per_source: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        dx_route_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "S T H"]:
        partial = _source_push_semantic_dx_return_sum_expert_major_pallas_call(
            dx_route_local,
            token_ids_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=_destination_offset_for_local_shard(dx_route_local.shape[0]),
            tokens_per_source=tokens_per_source,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(partial, SOURCE_PUSH_MESH_AXIS)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(dx_route, token_ids, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_dx_return_copy_only_expert_major_pallas_call(
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    dst_offset: Int[Array, ""],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "Dst E C H"]:
    local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim = dx_route.shape
    output_shape = jax.ShapeDtypeStruct(
        (local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim),
        jnp.float32,
    )
    dx_copy_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_return_copy_only_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(local_dst_count, experts_per_rank, rows_per_expert_capacity // row_block, hidden_dim // hidden_block),
        input_output_aliases={6: 0},
        interpret=interpret,
        name="source_push_semantic_dx_return_copy_only_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            dx_route,
            token_ids,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
            output_shape,
        ),
    )(dx_route, token_ids, xcounts, pair_expert_base, src_base_by_expert, dst_offset, dx_copy_init)


def _source_push_semantic_dx_return_copy_only_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "Dst E C H"]:
    def local_fn(
        dx_route_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "Dst E C H"]:
        return _source_push_semantic_dx_return_copy_only_expert_major_pallas_call(
            dx_route_local,
            token_ids_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            dst_offset=_destination_offset_for_local_shard(dx_route_local.shape[0]),
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dx_route, token_ids, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_dx_return_direct_to_source_sharded_mgpu_kernel(
    mesh: Mesh,
    dx_route: Float[Array, "Dst E C H"],
    recv_local_expert: Int[Array, "Dst SrcOrd Q"],
    recv_expert_row_start: Int[Array, "Dst SrcOrd Q"],
    recv_valid_rows: Int[Array, "Dst SrcOrd Q"],
    *,
    source_count: int,
    entries_per_dst: int,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
) -> Float[Array, "S DstOrd Q M H"]:
    kernel = _make_source_push_semantic_dx_return_direct_to_source_kernel(
        source_count=source_count,
        entries_per_dst=entries_per_dst,
        row_block=row_block,
        hidden_dim=dx_route.shape[-1],
        hidden_block=hidden_block,
        output_dtype=output_dtype,
    )

    def local_fn(
        dx_route_local: Float[Array, "1 E C H"],
        recv_local_expert_local: Int[Array, "1 SrcOrd Q"],
        recv_expert_row_start_local: Int[Array, "1 SrcOrd Q"],
        recv_valid_rows_local: Int[Array, "1 SrcOrd Q"],
    ) -> Float[Array, "1 DstOrd Q M H"]:
        return_dx = kernel(
            dx_route_local,
            recv_local_expert_local,
            recv_expert_row_start_local,
            recv_valid_rows_local,
        )
        return return_dx[None, ...]

    return_dx = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(dx_route, recv_local_expert, recv_expert_row_start, recv_valid_rows)
    return _source_push_semantic_dx_return_remote_write_barrier(mesh)(return_dx)


def _source_push_semantic_dx_return_remote_write_barrier(mesh: Mesh):
    """Synchronize direct dX queue writes before source-local reduction."""

    def local_fn(return_dx_local: Float[Array, "1 DstOrd Q M H"]) -> Float[Array, "1 DstOrd Q M H"]:
        marker = return_dx_local[0, 0, 0, 0, 0].astype(jnp.float32)
        barrier = jax.lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = barrier - jax.lax.optimization_barrier(barrier)
        return return_dx_local.at[0, 0, 0, 0, 0].add(zero.astype(return_dx_local.dtype))

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )


def _source_push_semantic_dx_combine_source_queue_pallas_call(
    return_dx: Float[Array, "S DstOrd Q M H"],
    route_dst_ordinal: Int[Array, "S T K"],
    route_entry: Int[Array, "S T K"],
    route_queue_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    token_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "S T H"]:
    source_count, tokens_per_source, topk = route_dst_ordinal.shape
    hidden_dim = return_dx.shape[-1]
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_combine_source_queue_kernel(
            token_block=token_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source // token_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_dx_combine_source_queue_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            return_dx,
            route_dst_ordinal,
            route_entry,
            route_queue_row,
            route_valid,
            output_shape,
        ),
    )(
        return_dx,
        route_dst_ordinal,
        route_entry,
        route_queue_row,
        route_valid.astype(jnp.int32),
    )


def _source_push_semantic_dx_combine_source_queue_local_pallas_call(
    return_dx: Float[Array, "DstOrd Q M H"],
    route_dst_ordinal: Int[Array, "T K"],
    route_entry: Int[Array, "T K"],
    route_queue_row: Int[Array, "T K"],
    route_valid: Bool[Array, "T K"],
    *,
    token_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "T H"]:
    tokens_per_source, topk = route_dst_ordinal.shape
    hidden_dim = return_dx.shape[-1]
    output_shape = jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_combine_source_queue_local_kernel(
            token_block=token_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(tokens_per_source // token_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_dx_combine_source_queue_local_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            return_dx,
            route_dst_ordinal,
            route_entry,
            route_queue_row,
            route_valid,
            output_shape,
        ),
    )(
        return_dx,
        route_dst_ordinal,
        route_entry,
        route_queue_row,
        route_valid.astype(jnp.int32),
    )


def _source_push_semantic_dx_combine_source_queue_sharded_mgpu_kernel(
    mesh: Mesh,
    return_dx: Float[Array, "S DstOrd Q M H"],
    route_dst_ordinal: Int[Array, "S T K"],
    route_entry: Int[Array, "S T K"],
    route_queue_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    token_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        return_dx_local: Float[Array, "1 DstOrd Q M H"],
        route_dst_ordinal_local: Int[Array, "1 T K"],
        route_entry_local: Int[Array, "1 T K"],
        route_queue_row_local: Int[Array, "1 T K"],
        route_valid_local: Bool[Array, "1 T K"],
    ) -> Float[Array, "1 T H"]:
        dx = _source_push_semantic_dx_combine_source_queue_local_pallas_call(
            return_dx_local[0],
            route_dst_ordinal_local[0],
            route_entry_local[0],
            route_queue_row_local[0],
            route_valid_local[0],
            token_block=token_block,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
            interpret=interpret,
        )
        return dx[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(return_dx, route_dst_ordinal, route_entry, route_queue_row, route_valid)


def _destination_offset_for_local_shard(local_dst_count: int) -> Int[Array, ""]:
    return jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS) * local_dst_count, dtype=jnp.int32)


def _source_push_semantic_dx_combine_pair_pallas_call(
    dx_pair: Float[Array, "S Dst R H"],
    token_ids: Int[Array, "S Dst R"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    tokens_per_source: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    source_count, _dst_count, _rows_per_pair, hidden_dim = dx_pair.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), dx_pair.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_dx_combine_pair_kernel(hidden_block=hidden_block),
        in_specs=(gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_dx_combine_pair_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(dx_pair, token_ids, valid_mask, output_shape),
    )(dx_pair, token_ids, valid_mask.astype(jnp.int32))


def _make_source_push_semantic_backward_dy_route_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dy_ref: Float[pl.Ref, "S T H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_weights_ref: Float[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        dy_route_ref: Float[pl.Ref, "Dst E C H"],
    ) -> None:
        local_dst = pl.program_id(0)
        dst = local_dst + dst_offset_ref[()]
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        expert_row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)
        for row_offset in range(row_block):
            expert_row = expert_row_start + row_offset
            src = jnp.asarray(0, dtype=jnp.int32)
            local_row = jnp.asarray(0, dtype=jnp.int32)
            valid = jnp.asarray(False)
            for candidate_src in range(xcounts_ref.shape[0]):
                source_expert_base = src_base_by_expert_ref[dst, candidate_src, expert]
                count = xcounts_ref[candidate_src, dst, expert]
                row_matches = (expert_row >= source_expert_base) & (expert_row < source_expert_base + count)
                src = jnp.where(row_matches, candidate_src, src)
                local_row = jnp.where(row_matches, expert_row - source_expert_base, local_row)
                valid = valid | row_matches

            pair_base = pair_expert_base_ref[src, dst, expert]
            pair_row = jnp.minimum(pair_base + local_row, token_ids_ref.shape[-1] - 1)
            safe_token = jnp.maximum(token_ids_ref[src, dst, pair_row], 0)
            dy_tile = dy_ref[pl.ds(src, 1), pl.ds(safe_token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
            weighted = dy_tile.astype(output_dtype) * route_weights_ref[src, dst, pair_row].astype(output_dtype)
            dy_route_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(valid, weighted, zero_tile)[None, None, None, :]

    return kernel


def _make_source_push_semantic_backward_dy_route_pair_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dy_ref: Float[pl.Ref, "S T H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_weights_ref: Float[pl.Ref, "S Dst R"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        dy_route_ref: Float[pl.Ref, "S Dst R H"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)
        for row_offset in range(row_block):
            row = row_start + row_offset
            valid = valid_mask_ref[src, dst, row] != 0
            safe_token = jnp.maximum(token_ids_ref[src, dst, row], 0)
            dy_tile = dy_ref[pl.ds(src, 1), pl.ds(safe_token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
            weighted = dy_tile.astype(output_dtype) * route_weights_ref[src, dst, row].astype(output_dtype)
            dy_route_ref[
                pl.ds(src, 1),
                pl.ds(dst, 1),
                pl.ds(row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(
                valid, weighted, zero_tile
            )[None, None, None, :]

    return kernel


def _make_source_push_semantic_backward_dcombine_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dy_ref: Float[pl.Ref, "S T H"],
        route_y_ref: Float[pl.Ref, "S Dst R H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_slots_ref: Int[pl.Ref, "S Dst R"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        _dcombine_init_ref: Float[pl.Ref, "S T K"],
        dcombine_ref: Float[pl.Ref, "S T K"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        row_tile = pl.program_id(2)
        row_start = row_tile * row_block

        for row_offset in range(row_block):
            row = row_start + row_offset
            valid = valid_mask_ref[src, dst, row] != 0
            safe_token = jnp.maximum(token_ids_ref[src, dst, row], 0)
            safe_slot = jnp.maximum(route_slots_ref[src, dst, row], 0)
            acc = jnp.zeros((), dtype=output_dtype)
            for hidden_start in range(0, dy_ref.shape[-1], hidden_block):
                dy_tile = dy_ref[pl.ds(src, 1), pl.ds(safe_token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
                route_tile = route_y_ref[
                    pl.ds(src, 1),
                    pl.ds(dst, 1),
                    pl.ds(row, 1),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, 0, :]
                acc += jnp.sum(dy_tile.astype(output_dtype) * route_tile.astype(output_dtype))
            current = dcombine_ref[src, safe_token, safe_slot]
            dcombine_ref[src, safe_token, safe_slot] = jnp.where(valid, current + acc, current)

    return kernel


def _make_source_push_semantic_backward_dcombine_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dy_ref: Float[pl.Ref, "S T H"],
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_slots_ref: Int[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _dcombine_init_ref: Float[pl.Ref, "S T K"],
        dcombine_ref: Float[pl.Ref, "S T K"],
    ) -> None:
        local_dst = pl.program_id(0)
        dst = local_dst + dst_offset_ref[()]
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        expert_row_start = row_tile * row_block

        for row_offset in range(row_block):
            expert_row = expert_row_start + row_offset
            src = jnp.asarray(0, dtype=jnp.int32)
            local_row = jnp.asarray(0, dtype=jnp.int32)
            valid = jnp.asarray(False)
            for candidate_src in range(xcounts_ref.shape[0]):
                source_expert_base = src_base_by_expert_ref[dst, candidate_src, expert]
                count = xcounts_ref[candidate_src, dst, expert]
                row_matches = (expert_row >= source_expert_base) & (expert_row < source_expert_base + count)
                src = jnp.where(row_matches, candidate_src, src)
                local_row = jnp.where(row_matches, expert_row - source_expert_base, local_row)
                valid = valid | row_matches

            pair_base = pair_expert_base_ref[src, dst, expert]
            pair_row = jnp.minimum(pair_base + local_row, token_ids_ref.shape[-1] - 1)
            safe_token = jnp.maximum(token_ids_ref[src, dst, pair_row], 0)
            safe_slot = jnp.maximum(route_slots_ref[src, dst, pair_row], 0)
            acc = jnp.zeros((), dtype=output_dtype)
            for hidden_start in range(0, dy_ref.shape[-1], hidden_block):
                dy_tile = dy_ref[pl.ds(src, 1), pl.ds(safe_token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
                route_tile = route_y_expert_ref[
                    pl.ds(local_dst, 1),
                    pl.ds(expert, 1),
                    pl.ds(expert_row, 1),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, 0, :]
                acc += jnp.sum(dy_tile.astype(output_dtype) * route_tile.astype(output_dtype))
            current = dcombine_ref[src, safe_token, safe_slot]
            dcombine_ref[src, safe_token, safe_slot] = jnp.where(valid, current + acc, current)

    return kernel


def _make_source_push_semantic_backward_dcombine_source_gather_kernel(*, hidden_block: int):
    def kernel(
        dy_ref: Float[pl.Ref, "S T H"],
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        route_dst_ref: Int[pl.Ref, "S T K"],
        route_expert_ref: Int[pl.Ref, "S T K"],
        route_expert_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        dst_offset_ref: Int[pl.Ref, ""],
        dcombine_ref: Float[pl.Ref, "S T K"],
    ) -> None:
        src = pl.program_id(0)
        token = pl.program_id(1)
        route_slot = pl.program_id(2)
        dst_offset = dst_offset_ref[()]
        global_dst = route_dst_ref[src, token, route_slot]
        local_dst = global_dst - dst_offset
        local_dst_valid = (local_dst >= 0) & (local_dst < route_y_expert_ref.shape[0])
        valid = (route_valid_ref[src, token, route_slot] != 0) & local_dst_valid
        safe_dst = jnp.where(local_dst_valid, local_dst, 0)
        expert = route_expert_ref[src, token, route_slot]
        expert_row = route_expert_row_ref[src, token, route_slot]
        acc = jnp.zeros((), dtype=jnp.float32)

        for hidden_start in range(0, dy_ref.shape[-1], hidden_block):
            dy_tile = dy_ref[pl.ds(src, 1), pl.ds(token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
            route_tile = route_y_expert_ref[
                pl.ds(safe_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]
            acc += jnp.sum(dy_tile.astype(jnp.float32) * route_tile.astype(jnp.float32))

        dcombine_ref[src, token, route_slot] = jnp.where(valid, acc, jnp.zeros((), dtype=jnp.float32))

    return kernel


def _make_source_push_semantic_backward_dcombine_return_queue_kernel(*, hidden_block: int):
    def kernel(
        dy_ref: Float[pl.Ref, "S T H"],
        return_y_ref: Float[pl.Ref, "S DstOrd Q M H"],
        route_dst_ordinal_ref: Int[pl.Ref, "S T K"],
        route_entry_ref: Int[pl.Ref, "S T K"],
        route_queue_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        dcombine_ref: Float[pl.Ref, "S T K"],
    ) -> None:
        source = pl.program_id(0)
        token = pl.program_id(1)
        route_slot = pl.program_id(2)
        dst_ordinal = route_dst_ordinal_ref[source, token, route_slot]
        entry = route_entry_ref[source, token, route_slot]
        queue_row = route_queue_row_ref[source, token, route_slot]
        valid = route_valid_ref[source, token, route_slot] != 0
        acc = jnp.zeros((), dtype=jnp.float32)

        for hidden_start in range(0, dy_ref.shape[-1], hidden_block):
            dy_tile = dy_ref[
                pl.ds(source, 1),
                pl.ds(token, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, :]
            route_y_tile = return_y_ref[
                source,
                dst_ordinal,
                entry,
                queue_row,
                pl.ds(hidden_start, hidden_block),
            ]
            acc += jnp.sum(dy_tile.astype(jnp.float32) * route_y_tile.astype(jnp.float32))

        dcombine_ref[source, token, route_slot] = jnp.where(valid, acc, jnp.zeros((), dtype=jnp.float32))

    return kernel


def _make_source_push_semantic_backward_dcombine_return_queue_local_kernel(*, hidden_block: int):
    def kernel(
        dy_ref: Float[pl.Ref, "T H"],
        return_y_ref: Float[pl.Ref, "DstOrd Q M H"],
        route_dst_ordinal_ref: Int[pl.Ref, "T K"],
        route_entry_ref: Int[pl.Ref, "T K"],
        route_queue_row_ref: Int[pl.Ref, "T K"],
        route_valid_ref: Int[pl.Ref, "T K"],
        dcombine_ref: Float[pl.Ref, "T K"],
    ) -> None:
        token = pl.program_id(0)
        route_slot = pl.program_id(1)
        dst_ordinal = route_dst_ordinal_ref[token, route_slot]
        entry = route_entry_ref[token, route_slot]
        queue_row = route_queue_row_ref[token, route_slot]
        valid = route_valid_ref[token, route_slot] != 0
        acc = jnp.zeros((), dtype=jnp.float32)

        for hidden_start in range(0, dy_ref.shape[-1], hidden_block):
            dy_tile = dy_ref[pl.ds(token, 1), pl.ds(hidden_start, hidden_block)][0, :]
            route_y_tile = return_y_ref[
                dst_ordinal,
                entry,
                queue_row,
                pl.ds(hidden_start, hidden_block),
            ]
            acc += jnp.sum(dy_tile.astype(jnp.float32) * route_y_tile.astype(jnp.float32))

        dcombine_ref[token, route_slot] = jnp.where(valid, acc, jnp.zeros((), dtype=jnp.float32))

    return kernel


def _make_source_push_semantic_dx_return_source_gather_kernel(*, hidden_block: int):
    def kernel(
        dx_route_ref: Float[pl.Ref, "Dst E C H"],
        route_dst_ref: Int[pl.Ref, "S T K"],
        route_expert_ref: Int[pl.Ref, "S T K"],
        route_expert_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        dst_offset_ref: Int[pl.Ref, ""],
        dx_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        src = pl.program_id(0)
        token = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        dst_offset = dst_offset_ref[()]
        acc = jnp.zeros((hidden_block,), dtype=dx_route_ref.dtype)

        for route_slot in range(route_valid_ref.shape[2]):
            global_dst = route_dst_ref[src, token, route_slot]
            local_dst = global_dst - dst_offset
            local_dst_valid = (local_dst >= 0) & (local_dst < dx_route_ref.shape[0])
            valid = (route_valid_ref[src, token, route_slot] != 0) & local_dst_valid
            safe_dst = jnp.where(local_dst_valid, local_dst, 0)
            expert = route_expert_ref[src, token, route_slot]
            expert_row = route_expert_row_ref[src, token, route_slot]
            dx_tile = dx_route_ref[
                pl.ds(safe_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]
            acc += jnp.where(valid, dx_tile, jnp.zeros((hidden_block,), dtype=dx_route_ref.dtype))

        dx_ref[pl.ds(src, 1), pl.ds(token, 1), pl.ds(hidden_start, hidden_block)] = acc[None, None, :]

    return kernel


def _make_source_push_semantic_dx_return_remote_source_gather_kernel(
    *,
    row_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dx_route_ref: Float[pl.Ref, "DstLocal E C H"],
        route_dst_ref: Int[pl.Ref, "T K"],
        route_expert_ref: Int[pl.Ref, "T K"],
        route_expert_row_ref: Int[pl.Ref, "T K"],
        route_valid_ref: Bool[pl.Ref, "T K"],
        dx_ref: Float[pl.Ref, "T H"],
    ) -> None:
        token_tile = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        token_start = token_tile * row_block
        hidden_start = hidden_tile * hidden_block
        zero = jnp.zeros((hidden_block,), dtype=jnp.float32)

        for token_offset in range(row_block):
            token = token_start + token_offset
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for route_slot in range(topk):
                dst = route_dst_ref[token, route_slot]
                valid = route_valid_ref[token, route_slot]
                safe_dst = jnp.where(valid, dst, 0)
                remote_dx_route_ref = mgpu.remote_ref(
                    dx_route_ref,
                    safe_dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                expert = route_expert_ref[token, route_slot]
                expert_row = route_expert_row_ref[token, route_slot]
                dx_tile = remote_dx_route_ref[
                    0,
                    pl.ds(expert, 1),
                    pl.ds(expert_row, 1),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, :].astype(jnp.float32)
                acc += jnp.where(valid, dx_tile, zero)

            dx_ref[token, pl.ds(hidden_start, hidden_block)] = acc.astype(output_dtype)

    return kernel


def _make_source_push_semantic_dx_return_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
):
    def kernel(
        dx_route_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_slots_ref: Int[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _dx_by_slot_init_ref: Float[pl.Ref, "S T K H"],
        dx_by_slot_ref: Float[pl.Ref, "S T K H"],
    ) -> None:
        local_dst = pl.program_id(0)
        global_dst = dst_offset_ref[()] + local_dst
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        expert_row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        for row_offset in range(row_block):
            expert_row = expert_row_start + row_offset
            src = jnp.asarray(0, dtype=jnp.int32)
            local_row = jnp.asarray(0, dtype=jnp.int32)
            valid = jnp.asarray(False)
            for candidate_src in range(xcounts_ref.shape[0]):
                source_expert_base = src_base_by_expert_ref[global_dst, candidate_src, expert]
                count = xcounts_ref[candidate_src, global_dst, expert]
                row_matches = (expert_row >= source_expert_base) & (expert_row < source_expert_base + count)
                src = jnp.where(row_matches, candidate_src, src)
                local_row = jnp.where(row_matches, expert_row - source_expert_base, local_row)
                valid = valid | row_matches

            pair_base = pair_expert_base_ref[src, global_dst, expert]
            pair_row = jnp.minimum(pair_base + local_row, token_ids_ref.shape[-1] - 1)
            safe_token = jnp.maximum(token_ids_ref[src, global_dst, pair_row], 0)
            safe_slot = jnp.maximum(route_slots_ref[src, global_dst, pair_row], 0)
            dx_tile = dx_route_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]

            @pl.when(valid)
            def _store_valid_route() -> None:
                dx_by_slot_ref[
                    pl.ds(src, 1),
                    pl.ds(safe_token, 1),
                    pl.ds(safe_slot, 1),
                    pl.ds(hidden_start, hidden_block),
                ] = dx_tile[None, None, None, :]

    return kernel


def _make_source_push_semantic_dx_return_sum_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
):
    def kernel(
        dx_route_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _dx_init_ref: Float[pl.Ref, "S T H"],
        dx_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        local_dst = pl.program_id(0)
        global_dst = dst_offset_ref[()] + local_dst
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        expert_row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        for row_offset in range(row_block):
            expert_row = expert_row_start + row_offset
            src = jnp.asarray(0, dtype=jnp.int32)
            local_row = jnp.asarray(0, dtype=jnp.int32)
            valid = jnp.asarray(False)
            for candidate_src in range(xcounts_ref.shape[0]):
                source_expert_base = src_base_by_expert_ref[global_dst, candidate_src, expert]
                count = xcounts_ref[candidate_src, global_dst, expert]
                row_matches = (expert_row >= source_expert_base) & (expert_row < source_expert_base + count)
                src = jnp.where(row_matches, candidate_src, src)
                local_row = jnp.where(row_matches, expert_row - source_expert_base, local_row)
                valid = valid | row_matches

            pair_base = pair_expert_base_ref[src, global_dst, expert]
            pair_row = jnp.minimum(pair_base + local_row, token_ids_ref.shape[-1] - 1)
            safe_token = jnp.maximum(token_ids_ref[src, global_dst, pair_row], 0)
            dx_tile = dx_route_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :].astype(jnp.float32)

            @pl.when(valid)
            def _add_valid_route() -> None:
                mgpu.atomic_add(
                    dx_ref.at[src, safe_token, pl.ds(hidden_start, hidden_block)],
                    dx_tile,
                )

    return kernel


def _make_source_push_semantic_dx_return_copy_only_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        dx_route_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _dx_copy_init_ref: Float[pl.Ref, "Dst E C H"],
        dx_copy_ref: Float[pl.Ref, "Dst E C H"],
    ) -> None:
        local_dst = pl.program_id(0)
        global_dst = dst_offset_ref[()] + local_dst
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        expert_row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)
        for row_offset in range(row_block):
            expert_row = expert_row_start + row_offset
            src = jnp.asarray(0, dtype=jnp.int32)
            local_row = jnp.asarray(0, dtype=jnp.int32)
            valid = jnp.asarray(False)
            for candidate_src in range(xcounts_ref.shape[0]):
                source_expert_base = src_base_by_expert_ref[global_dst, candidate_src, expert]
                count = xcounts_ref[candidate_src, global_dst, expert]
                row_matches = (expert_row >= source_expert_base) & (expert_row < source_expert_base + count)
                src = jnp.where(row_matches, candidate_src, src)
                local_row = jnp.where(row_matches, expert_row - source_expert_base, local_row)
                valid = valid | row_matches

            pair_base = pair_expert_base_ref[src, global_dst, expert]
            pair_row = jnp.minimum(pair_base + local_row, token_ids_ref.shape[-1] - 1)
            token_is_valid = token_ids_ref[src, global_dst, pair_row] >= 0
            dx_tile = dx_route_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :].astype(output_dtype)
            dx_copy_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(valid & token_is_valid, dx_tile, zero_tile)[None, None, None, :]

    return kernel


def _make_source_push_semantic_dx_return_direct_to_source_kernel(
    *,
    source_count: int,
    entries_per_dst: int,
    row_block: int,
    hidden_dim: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    source_ordinals = tuple(range(source_count))

    def body(
        dx_route_ref: Float[pl.Ref, "Dst E C H"],
        recv_local_expert_ref: Int[pl.Ref, "Dst SrcOrd Q"],
        recv_expert_row_start_ref: Int[pl.Ref, "Dst SrcOrd Q"],
        recv_valid_rows_ref: Int[pl.Ref, "Dst SrcOrd Q"],
        return_dx_ref: Float[pl.Ref, "DstOrd Q M H"],
    ) -> None:
        rank = jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block

        def _copy_to_source(static_src_ordinal: int) -> None:
            valid_rows = recv_valid_rows_ref[0, static_src_ordinal, entry]

            @pl.when(valid_rows > 0)
            def _copy_return_block() -> None:
                expert = recv_local_expert_ref[0, static_src_ordinal, entry]
                expert_row_start = recv_expert_row_start_ref[0, static_src_ordinal, entry]

                def copy_scope(dx_smem, ready_barrier) -> None:
                    mgpu.copy_gmem_to_smem(
                        dx_route_ref.at[
                            0,
                            expert,
                            pl.ds(expert_row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dx_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    source = (rank + static_src_ordinal) % source_count
                    dst_ordinal = (-static_src_ordinal) % source_count
                    if static_src_ordinal == 0:
                        destination_ref = return_dx_ref
                    else:
                        destination_ref = mgpu.remote_ref(
                            return_dx_ref,
                            source,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                    mgpu.copy_smem_to_gmem(
                        dx_smem,
                        destination_ref.at[
                            dst_ordinal,
                            entry,
                            pl.ds(0, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                    )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    copy_scope,
                    dx_smem=mgpu.SMEM((row_block, hidden_block), dtype=output_dtype),
                    ready_barrier=mgpu.Barrier(num_arrivals=1),
                )

        def _branch(static_src_ordinal: int):
            def _copy_branch(_) -> None:
                _copy_to_source(static_src_ordinal)

            return _copy_branch

        jax.lax.switch(src_ordinal, tuple(_branch(src) for src in source_ordinals), None)

    out_shape = jax.ShapeDtypeStruct(
        (source_count, entries_per_dst, row_block, hidden_dim),
        output_dtype,
    )
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(source_count, entries_per_dst, hidden_dim // hidden_block),
        grid_names=("src_ordinal", "entry", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _make_source_push_semantic_dx_combine_source_queue_kernel(
    *,
    token_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        return_dx_ref: Float[pl.Ref, "S DstOrd Q M H"],
        route_dst_ordinal_ref: Int[pl.Ref, "S T K"],
        route_entry_ref: Int[pl.Ref, "S T K"],
        route_queue_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        dx_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        source = pl.program_id(0)
        token_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        token_start = token_tile * token_block
        hidden_start = hidden_tile * hidden_block
        zero = jnp.zeros((hidden_block,), dtype=jnp.float32)

        for token_offset in range(token_block):
            token = token_start + token_offset
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for route_slot in range(topk):
                valid = route_valid_ref[source, token, route_slot] != 0
                dst_ordinal = route_dst_ordinal_ref[source, token, route_slot]
                entry = route_entry_ref[source, token, route_slot]
                queue_row = route_queue_row_ref[source, token, route_slot]
                route_dx = return_dx_ref[
                    source,
                    dst_ordinal,
                    entry,
                    queue_row,
                    pl.ds(hidden_start, hidden_block),
                ].astype(jnp.float32)
                acc += jnp.where(valid, route_dx, zero)

            dx_ref[source, token, pl.ds(hidden_start, hidden_block)] = acc.astype(output_dtype)

    return kernel


def _make_source_push_semantic_dx_combine_source_queue_local_kernel(
    *,
    token_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        return_dx_ref: Float[pl.Ref, "DstOrd Q M H"],
        route_dst_ordinal_ref: Int[pl.Ref, "T K"],
        route_entry_ref: Int[pl.Ref, "T K"],
        route_queue_row_ref: Int[pl.Ref, "T K"],
        route_valid_ref: Int[pl.Ref, "T K"],
        dx_ref: Float[pl.Ref, "T H"],
    ) -> None:
        token_tile = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        token_start = token_tile * token_block
        hidden_start = hidden_tile * hidden_block
        zero = jnp.zeros((hidden_block,), dtype=jnp.float32)

        for token_offset in range(token_block):
            token = token_start + token_offset
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for route_slot in range(topk):
                valid = route_valid_ref[token, route_slot] != 0
                dst_ordinal = route_dst_ordinal_ref[token, route_slot]
                entry = route_entry_ref[token, route_slot]
                queue_row = route_queue_row_ref[token, route_slot]
                route_dx = return_dx_ref[
                    dst_ordinal,
                    entry,
                    queue_row,
                    pl.ds(hidden_start, hidden_block),
                ].astype(jnp.float32)
                acc += jnp.where(valid, route_dx, zero)

            dx_ref[token, pl.ds(hidden_start, hidden_block)] = acc.astype(output_dtype)

    return kernel


def _make_source_push_semantic_dx_combine_pair_kernel(*, hidden_block: int):
    def kernel(
        dx_pair_ref: Float[pl.Ref, "S Dst R H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        dx_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        src = pl.program_id(0)
        token = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        acc = jnp.zeros((hidden_block,), dtype=dx_pair_ref.dtype)
        for dst in range(dx_pair_ref.shape[1]):
            for row in range(dx_pair_ref.shape[2]):
                valid = (valid_mask_ref[src, dst, row] != 0) & (token_ids_ref[src, dst, row] == token)
                dx_tile = dx_pair_ref[
                    pl.ds(src, 1),
                    pl.ds(dst, 1),
                    pl.ds(row, 1),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, 0, :]
                acc += jnp.where(valid, dx_tile, jnp.zeros((hidden_block,), dtype=dx_pair_ref.dtype))
        dx_ref[pl.ds(src, 1), pl.ds(token, 1), pl.ds(hidden_start, hidden_block)] = acc[None, None, :]

    return kernel


def _simple_io_cost_estimate(*arrays_and_outputs: Array | jax.ShapeDtypeStruct) -> pl.CostEstimate:
    bytes_accessed = 0
    for value in arrays_and_outputs:
        dtype = jnp.dtype(value.dtype)
        bytes_accessed += math.prod(value.shape) * dtype.itemsize
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=bytes_accessed,
        remote_bytes_transferred=0,
    )


def _resolve_source_row_base_by_expert(
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None,
) -> Int[Array, "Dst S E"]:
    if source_row_base_by_expert is None:
        source_row_base_by_expert = plan.src_base_by_expert
    expected_shape = (plan.xcounts.shape[1], plan.xcounts.shape[0], plan.xcounts.shape[2])
    if source_row_base_by_expert.shape != expected_shape:
        raise ValueError(
            f"source_row_base_by_expert shape {source_row_base_by_expert.shape} must match {expected_shape}"
        )
    if jnp.dtype(source_row_base_by_expert.dtype) != jnp.dtype(jnp.int32):
        raise ValueError(f"source_row_base_by_expert must have dtype int32, got {source_row_base_by_expert.dtype}")
    return source_row_base_by_expert


def _semantic_expert_row_indices_jax(
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Int[Array, "Dst S E"],
) -> tuple[Int[Array, "S Dst R"], Int[Array, "S Dst R"]]:
    return _source_push_semantic_expert_row_indices_from_metadata_jax(
        xcounts=plan.xcounts,
        pair_expert_base=plan.pair_expert_base,
        src_base_by_expert=source_row_base_by_expert,
        rows_per_pair_capacity=plan.assignment_ids.shape[-1],
    )


def _semantic_pair_to_expert_major_jax(
    pair_values: Float[Array, "S Dst R F"],
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
) -> Float[Array, "Dst E C F"]:
    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    if pair_values.shape[:3] != plan.assignment_ids.shape:
        raise ValueError(
            f"pair_values shape {pair_values.shape[:3]} must match semantic rows {plan.assignment_ids.shape}"
        )
    expert_ids, expert_rows = _semantic_expert_row_indices_jax(plan, source_row_base_by_expert)
    valid = plan.valid_mask & (expert_rows < rows_per_expert_capacity)
    scatter_rows = jnp.where(valid, expert_rows, rows_per_expert_capacity)
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    out = jnp.zeros(
        (
            plan.assignment_ids.shape[1],
            plan.xcounts.shape[-1],
            rows_per_expert_capacity,
            pair_values.shape[-1],
        ),
        dtype=pair_values.dtype,
    )
    return out.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(valid[..., None], pair_values, jnp.zeros((), dtype=pair_values.dtype)),
        mode="drop",
    )


def _validate_backward_source_expand_request(
    dy: Array,
    route_y: Array,
    plan: SourcePushSemanticPlan,
    rows_per_expert_capacity: int,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    _validate_backward_source_expand_pair_request(dy, route_y, plan, block_sizes)
    _validate_expert_major_metadata(plan)
    _validate_transport_block_sizes(rows_per_expert_capacity, dy.shape[2], block_sizes)


def _validate_backward_dy_route_source_push_request(
    dy: Array,
    plan: SourcePushSemanticPlan,
    rows_per_expert_capacity: int,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, hidden], got {dy.shape}")
    if dy.shape[0] != plan.assignment_ids.shape[0]:
        raise ValueError(f"dy source dim {dy.shape[0]} must match plan source dim {plan.assignment_ids.shape[0]}")
    if dy.shape[1] != plan.tokens_per_source:
        raise ValueError(f"dy token dim {dy.shape[1]} must match plan tokens_per_source={plan.tokens_per_source}")
    _validate_expert_major_metadata(plan)
    _validate_transport_block_sizes(rows_per_expert_capacity, dy.shape[2], block_sizes)


def _validate_backward_source_expand_expert_major_route_request(
    dy: Array,
    route_y_expert: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, hidden], got {dy.shape}")
    if route_y_expert.ndim != 4:
        raise ValueError(
            f"route_y_expert must have shape [destination, expert, row, hidden], got {route_y_expert.shape}"
        )
    if route_y_expert.shape[:2] != (plan.assignment_ids.shape[1], plan.xcounts.shape[2]):
        raise ValueError(
            f"route_y_expert leading shape {route_y_expert.shape[:2]} must match "
            f"{(plan.assignment_ids.shape[1], plan.xcounts.shape[2])}"
        )
    if dy.shape[0] != plan.assignment_ids.shape[0]:
        raise ValueError(f"dy source dim {dy.shape[0]} must match plan source dim {plan.assignment_ids.shape[0]}")
    if dy.shape[1] != plan.tokens_per_source:
        raise ValueError(f"dy token dim {dy.shape[1]} must match plan tokens_per_source={plan.tokens_per_source}")
    if dy.shape[2] != route_y_expert.shape[3]:
        raise ValueError(f"dy hidden dim {dy.shape[2]} must match route_y_expert hidden dim {route_y_expert.shape[3]}")
    _validate_expert_major_metadata(plan)
    _validate_transport_block_sizes(route_y_expert.shape[2], route_y_expert.shape[3], block_sizes)


def _validate_backward_dcombine_source_gather_request(
    dy: Array,
    route_y_expert: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, hidden], got {dy.shape}")
    if route_y_expert.ndim != 4:
        raise ValueError(
            f"route_y_expert must have shape [destination, expert, row, hidden], got {route_y_expert.shape}"
        )
    if route_y_expert.shape[:2] != (plan.assignment_ids.shape[1], plan.xcounts.shape[2]):
        raise ValueError(
            f"route_y_expert leading shape {route_y_expert.shape[:2]} must match "
            f"{(plan.assignment_ids.shape[1], plan.xcounts.shape[2])}"
        )
    if dy.shape[0] != plan.assignment_ids.shape[0]:
        raise ValueError(f"dy source dim {dy.shape[0]} must match plan source dim {plan.assignment_ids.shape[0]}")
    if dy.shape[1] != plan.tokens_per_source:
        raise ValueError(f"dy token dim {dy.shape[1]} must match plan tokens_per_source={plan.tokens_per_source}")
    if dy.shape[2] != route_y_expert.shape[3]:
        raise ValueError(f"dy hidden dim {dy.shape[2]} must match route_y_expert hidden dim {route_y_expert.shape[3]}")
    _validate_expert_major_metadata(plan)
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dy.shape[2] % block_sizes.hidden_block:
        raise ValueError(f"hidden dim {dy.shape[2]} must be divisible by hidden_block={block_sizes.hidden_block}")


def _validate_backward_dcombine_return_queue_request(
    dy: Array,
    return_y: Array,
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, hidden], got {dy.shape}")
    if return_y.ndim != 5:
        raise ValueError(f"return_y must have shape [source, dst_ord, entry, row, hidden], got {return_y.shape}")
    source_count, destination_count = plan.assignment_ids.shape[:2]
    if dy.shape[:2] != (source_count, plan.tokens_per_source):
        raise ValueError(f"dy source/token shape {dy.shape[:2]} must match {(source_count, plan.tokens_per_source)}")
    if return_y.shape[:2] != (source_count, destination_count):
        raise ValueError(
            f"return_y source/destination-ordinal shape {return_y.shape[:2]} must match "
            f"{(source_count, destination_count)}"
        )
    if return_y.shape[3] <= 0:
        raise ValueError(f"return_y row dim must be positive, got {return_y.shape[3]}")
    if return_y.shape[2] != queue.entries_per_dst:
        raise ValueError(
            f"return_y entry dim {return_y.shape[2]} must match queue entries_per_dst={queue.entries_per_dst}"
        )
    if return_y.shape[3] != queue.return_row_block:
        raise ValueError(
            f"return_y row dim {return_y.shape[3]} must match queue return_row_block={queue.return_row_block}"
        )
    if dy.shape[-1] != return_y.shape[-1]:
        raise ValueError(f"dy hidden dim {dy.shape[-1]} must match return_y hidden dim {return_y.shape[-1]}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dy.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"hidden dim {dy.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")


def _validate_backward_source_expand_pair_request(
    dy: Array,
    route_y: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, hidden], got {dy.shape}")
    if route_y.ndim != 4:
        raise ValueError(f"route_y must have shape [source, destination, row, hidden], got {route_y.shape}")
    if plan.token_ids.shape != route_y.shape[:3]:
        raise ValueError(
            f"plan token_ids shape {plan.token_ids.shape} must match route_y route shape {route_y.shape[:3]}"
        )
    if plan.route_slots.shape != plan.token_ids.shape:
        raise ValueError(
            f"route_slots shape {plan.route_slots.shape} must match token_ids shape {plan.token_ids.shape}"
        )
    if plan.route_weights.shape != plan.token_ids.shape:
        raise ValueError(
            f"route_weights shape {plan.route_weights.shape} must match token_ids shape {plan.token_ids.shape}"
        )
    if plan.valid_mask.shape != plan.token_ids.shape:
        raise ValueError(f"valid_mask shape {plan.valid_mask.shape} must match token_ids shape {plan.token_ids.shape}")
    if dy.shape[0] != route_y.shape[0]:
        raise ValueError(f"dy source dim {dy.shape[0]} must match route_y source dim {route_y.shape[0]}")
    if dy.shape[1] != plan.tokens_per_source:
        raise ValueError(f"dy token dim {dy.shape[1]} must match plan tokens_per_source={plan.tokens_per_source}")
    if dy.shape[2] != route_y.shape[3]:
        raise ValueError(f"dy hidden dim {dy.shape[2]} must match route_y hidden dim {route_y.shape[3]}")
    _validate_transport_block_sizes(plan.token_ids.shape[2], dy.shape[2], block_sizes)


def _source_push_semantic_dx_queue_metadata_jax(
    plan: SourcePushSemanticPlan,
    *,
    row_block: int,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> SourcePushSemanticQueueMetadata:
    return _resolve_source_queue_metadata(
        plan,
        queue,
        row_block=row_block,
    )


def _resolve_source_queue_metadata(
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata | None,
    *,
    row_block: int,
    entries_per_dst: int | None = None,
) -> SourcePushSemanticQueueMetadata:
    if row_block <= 0:
        raise ValueError(f"row_block must be positive, got {row_block}")
    if queue is None:
        if entries_per_dst is None:
            rows_per_pair = plan.assignment_ids.shape[-1]
            experts_per_rank = plan.xcounts.shape[-1]
            entries_per_dst = (rows_per_pair + row_block - 1) // row_block + experts_per_rank - 1
        queue = source_push_semantic_queue_metadata_jax(
            plan,
            return_row_block=row_block,
            entries_per_dst=entries_per_dst,
        )
    _validate_source_queue_metadata(plan, queue, row_block=row_block)
    if entries_per_dst is not None and queue.entries_per_dst != entries_per_dst:
        raise ValueError(
            f"queue entries_per_dst {queue.entries_per_dst} must match return buffer entry dim {entries_per_dst}"
        )
    return queue


def _validate_source_queue_metadata(
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    *,
    row_block: int,
) -> None:
    source_count, destination_count = plan.assignment_ids.shape[:2]
    queue_shape = (source_count, destination_count, queue.entries_per_dst)
    for name, value in (
        ("local_expert", queue.local_expert),
        ("local_row_start", queue.local_row_start),
        ("valid_rows", queue.valid_rows),
    ):
        if value.shape != queue_shape:
            raise ValueError(f"queue {name} shape {value.shape} must match {queue_shape}")
    if queue.required_entries_per_dst.shape != (source_count, destination_count):
        raise ValueError(
            "queue required_entries_per_dst shape "
            f"{queue.required_entries_per_dst.shape} must match {(source_count, destination_count)}"
        )
    route_shape = (source_count, plan.tokens_per_source, plan.topk)
    for name, value in (
        ("route_dst_ordinal", queue.route_dst_ordinal),
        ("route_entry", queue.route_entry),
        ("route_queue_row", queue.route_queue_row),
        ("route_valid", queue.route_valid),
    ):
        if value.shape != route_shape:
            raise ValueError(f"queue {name} shape {value.shape} must match {route_shape}")
    if queue.return_row_block != row_block:
        raise ValueError(f"queue return_row_block {queue.return_row_block} must match requested {row_block}")


def _source_push_semantic_dx_receive_metadata_jax(
    plan: SourcePushSemanticPlan,
    metadata: SourcePushSemanticQueueMetadata,
    source_row_base_by_expert: Int[Array, "Dst S E"],
) -> tuple[Int[Array, "Dst SrcOrd Q"], Int[Array, "Dst SrcOrd Q"], Int[Array, "Dst SrcOrd Q"]]:
    source_count, destination_count = plan.assignment_ids.shape[:2]
    destination = jnp.arange(destination_count, dtype=jnp.int32)[:, None]
    source_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    source = (destination + source_ordinal) % source_count
    dst_ordinal = jnp.broadcast_to((-source_ordinal) % source_count, source.shape)
    local_expert = metadata.local_expert.at[source, dst_ordinal].get()
    local_row_start = metadata.local_row_start.at[source, dst_ordinal].get()
    valid_rows = metadata.valid_rows.at[source, dst_ordinal].get()
    safe_expert = jnp.maximum(local_expert, 0)
    expert_source_base = source_row_base_by_expert.at[
        destination[:, :, None],
        source[:, :, None],
        safe_expert,
    ].get()
    expert_row_start = expert_source_base + local_row_start
    expert_row_start = jnp.where(valid_rows > 0, expert_row_start, 0).astype(jnp.int32)
    return local_expert, expert_row_start, valid_rows


def _semantic_destination_major_sharding(mesh: Mesh, *, rank: int) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(rank - 1))))


def _semantic_source_major_sharding(mesh: Mesh, *, rank: int) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(rank - 1))))


def _validate_dx_queue_return_request(
    dx_route: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes,
    route_buffer_dtype: jnp.dtype,
) -> None:
    if block_sizes.row_block > MAX_SEMANTIC_DX_QUEUE_COPY_DIM:
        raise ValueError(
            f"row_block={block_sizes.row_block} exceeds the Mosaic async-copy limit "
            f"{MAX_SEMANTIC_DX_QUEUE_COPY_DIM}"
        )
    if block_sizes.hidden_block > MAX_SEMANTIC_DX_QUEUE_COPY_DIM:
        raise ValueError(
            f"hidden_block={block_sizes.hidden_block} exceeds the Mosaic async-copy limit "
            f"{MAX_SEMANTIC_DX_QUEUE_COPY_DIM}"
        )
    _validate_dx_combine_request(
        dx_route,
        plan,
        SourcePushSemanticBackwardPallasBlockSizes(
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
        ),
    )
    if jnp.dtype(route_buffer_dtype) != jnp.dtype(jnp.bfloat16):
        raise ValueError(f"route_buffer_dtype must be bfloat16, got {jnp.dtype(route_buffer_dtype)}")


def _validate_dx_queue_combine_request(
    return_dx: Array,
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    block_sizes: SourcePushSemanticDxReturnPallasBlockSizes,
) -> None:
    if return_dx.ndim != 5:
        raise ValueError(f"return_dx must have shape [source, dst_ord, entry, row, hidden], got {return_dx.shape}")
    source_count, destination_count = plan.assignment_ids.shape[:2]
    if return_dx.shape[:2] != (source_count, destination_count):
        raise ValueError(
            f"return_dx source/destination-ordinal shape {return_dx.shape[:2]} must match "
            f"{(source_count, destination_count)}"
        )
    if return_dx.shape[3] != block_sizes.row_block:
        raise ValueError(f"return_dx row dim {return_dx.shape[3]} must match row_block={block_sizes.row_block}")
    if return_dx.shape[2] != queue.entries_per_dst:
        raise ValueError(
            f"return_dx entry dim {return_dx.shape[2]} must match queue entries_per_dst={queue.entries_per_dst}"
        )
    if jnp.dtype(return_dx.dtype) != jnp.dtype(jnp.bfloat16):
        raise ValueError(f"return_dx must use bfloat16 storage, got {return_dx.dtype}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if plan.tokens_per_source % block_sizes.row_block:
        raise ValueError(
            f"tokens_per_source={plan.tokens_per_source} must be divisible by row_block={block_sizes.row_block}"
        )
    if return_dx.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            f"return_dx hidden dim {return_dx.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )


def _validate_dx_combine_request(
    dx_route: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dx_route.ndim != 4:
        raise ValueError(f"dx_route must have shape [destination, expert, row, hidden], got {dx_route.shape}")
    if dx_route.shape[:2] != (plan.assignment_ids.shape[1], plan.xcounts.shape[2]):
        raise ValueError(
            f"dx_route leading shape {dx_route.shape[:2]} must match "
            f"{(plan.assignment_ids.shape[1], plan.xcounts.shape[2])}"
        )
    _validate_expert_major_metadata(plan)
    _validate_transport_block_sizes(dx_route.shape[2], dx_route.shape[3], block_sizes)


def _validate_dx_source_gather_request(
    dx_route: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dx_route.ndim != 4:
        raise ValueError(f"dx_route must have shape [destination, expert, row, hidden], got {dx_route.shape}")
    if dx_route.shape[:2] != (plan.assignment_ids.shape[1], plan.xcounts.shape[2]):
        raise ValueError(
            f"dx_route leading shape {dx_route.shape[:2]} must match "
            f"{(plan.assignment_ids.shape[1], plan.xcounts.shape[2])}"
        )
    _validate_expert_major_metadata(plan)
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dx_route.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"hidden dim {dx_route.shape[3]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )


def _validate_dx_combine_pair_request(
    dx_pair: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if dx_pair.ndim != 4:
        raise ValueError(f"dx_pair must have shape [source, destination, row, hidden], got {dx_pair.shape}")
    if plan.token_ids.shape != dx_pair.shape[:3]:
        raise ValueError(
            f"plan token_ids shape {plan.token_ids.shape} must match dx_pair route shape {dx_pair.shape[:3]}"
        )
    if plan.valid_mask.shape != plan.token_ids.shape:
        raise ValueError(f"valid_mask shape {plan.valid_mask.shape} must match token_ids shape {plan.token_ids.shape}")
    _validate_transport_block_sizes(dx_pair.shape[2], dx_pair.shape[3], block_sizes)


def _validate_expert_major_metadata(plan: SourcePushSemanticPlan) -> None:
    if plan.xcounts.ndim != 3:
        raise ValueError(f"xcounts must have shape [source, destination, expert], got {plan.xcounts.shape}")
    if plan.pair_expert_base.shape != plan.xcounts.shape:
        raise ValueError(
            f"pair_expert_base shape {plan.pair_expert_base.shape} must match xcounts shape {plan.xcounts.shape}"
        )
    if plan.src_base_by_expert.shape != (plan.xcounts.shape[1], plan.xcounts.shape[0], plan.xcounts.shape[2]):
        raise ValueError(
            f"src_base_by_expert shape {plan.src_base_by_expert.shape} must match "
            f"{(plan.xcounts.shape[1], plan.xcounts.shape[0], plan.xcounts.shape[2])}"
        )


def _rows_per_expert_capacity(
    plan: SourcePushSemanticPlan,
    requested_capacity: int | None,
    row_block: int,
) -> int:
    if row_block <= 0:
        raise ValueError(f"row_block must be positive, got {row_block}")
    if requested_capacity is not None:
        return requested_capacity
    static_upper_bound = plan.assignment_ids.shape[0] * plan.assignment_ids.shape[2]
    return max(row_block, _ceil_div(static_upper_bound, row_block) * row_block)


def _ceil_div(lhs: int, rhs: int) -> int:
    return -(-lhs // rhs)


def _validate_transport_block_sizes(
    rows_per_pair: int,
    hidden_dim: int,
    block_sizes: SourcePushSemanticBackwardPallasBlockSizes,
) -> None:
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if rows_per_pair % block_sizes.row_block:
        raise ValueError(
            "semantic backward transport prototype currently requires row capacity divisible by row_block, "
            f"got rows_per_pair={rows_per_pair}, row_block={block_sizes.row_block}"
        )
    if hidden_dim % block_sizes.hidden_block:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by hidden_block={block_sizes.hidden_block}")
