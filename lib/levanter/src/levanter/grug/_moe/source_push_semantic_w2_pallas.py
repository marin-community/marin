# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas scaffolding for semantic source-push W2/return kernels.

The production target is a destination-local W2 that consumes pair-flat
``h_pair[S, Dst, R, I]`` rows and writes unweighted route outputs
``route_y[S, Dst, R, H]`` before the source-side combine. This scaffold keeps
the slot-free semantic metadata contract intact and derives each row's local
expert from ``pair_expert_base/xcounts`` instead of passing a dense expert-id
stream.

The combine/return step remains the JAX scatter-add reference for now. A true
Pallas return kernel needs a separate scatter/atomic design decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_backward_w2 import SOURCE_PUSH_MESH_AXIS
from levanter.grug._moe.source_push_plan import (
    SourcePushSemanticReverseRoute,
    SourcePushSemanticPlan,
    SourcePushSemanticQueueMetadata,
    source_push_semantic_combine_jax,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_pair_expert_ids_jax,
    source_push_semantic_queue_metadata_jax,
    source_push_semantic_reverse_route_jax,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


DEFAULT_SEMANTIC_W2_ROW_BLOCK = 1
DEFAULT_SEMANTIC_W2_INTERMEDIATE_BLOCK = 128
DEFAULT_SEMANTIC_W2_HIDDEN_BLOCK = 128
DEFAULT_SEMANTIC_W2_EXPERT_MAJOR_ROW_BLOCK = 128
DEFAULT_SEMANTIC_W2_EXPERT_MAJOR_INTERMEDIATE_BLOCK = 64
DEFAULT_SEMANTIC_W2_EXPERT_MAJOR_HIDDEN_BLOCK = 128
SEMANTIC_W2_WGMMA_TILE_M = 8
SEMANTIC_W2_WGMMA_SWIZZLE_BYTES = 128


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW2PallasBlockSizes:
    """Tile sizes for the pair-flat semantic W2 Pallas scaffold."""

    row_block: int = DEFAULT_SEMANTIC_W2_ROW_BLOCK
    intermediate_block: int = DEFAULT_SEMANTIC_W2_INTERMEDIATE_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W2_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW2PallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW2ExpertMajorPallasBlockSizes:
    """Tile sizes for the expert-major semantic W2 WGMMA kernel."""

    row_block: int = DEFAULT_SEMANTIC_W2_EXPERT_MAJOR_ROW_BLOCK
    intermediate_block: int = DEFAULT_SEMANTIC_W2_EXPERT_MAJOR_INTERMEDIATE_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W2_EXPERT_MAJOR_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW2ExpertMajorPallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushSemanticForwardReturnPallasBlockSizes:
    """Tile sizes for returning expert-major route rows to source slots."""

    row_block: int = 64
    hidden_block: int = 256

    @classmethod
    def get_default(cls) -> "SourcePushSemanticForwardReturnPallasBlockSizes":
        return cls()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticForwardReturnQueueMetadata:
    """Queue-oriented metadata for source-owned W2 return/combine."""

    queue_local_expert: Int[Array, "S DstOrd Q"]
    queue_local_row_start: Int[Array, "S DstOrd Q"]
    queue_valid_rows: Int[Array, "S DstOrd Q"]
    recv_local_expert: Int[Array, "Dst SrcOrd Q"]
    recv_expert_row_start: Int[Array, "Dst SrcOrd Q"]
    recv_valid_rows: Int[Array, "Dst SrcOrd Q"]
    queue_dst_ord: Int[Array, "S T K"]
    queue_entry: Int[Array, "S T K"]
    queue_row: Int[Array, "S T K"]
    route_weight: Float[Array, "S T K"]
    route_valid: Bool[Array, "S T K"]


def source_push_semantic_w2_pallas_mgpu(
    h_pair: Float[Array, "S Dst R I"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW2PallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
) -> Float[Array, "S Dst R H"]:
    """Compute unweighted semantic W2 route outputs with Pallas/Mosaic GPU.

    ``Warpgroup`` lowering is the default because this kernel has no peer-id
    GMEM refs; it only reads local destination buffers and local weights.
    ``Lane`` can still be requested by callers when debugging lowering issues.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W2 requires a GPU backend")
    block_sizes = SourcePushSemanticW2PallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w2_pallas_request(h_pair, w_down, plan, block_sizes)
    return _source_push_semantic_w2_pallas_call(
        h_pair,
        w_down,
        plan.xcounts,
        plan.pair_expert_base,
        plan.valid_mask,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
    )


def source_push_semantic_w2_and_combine_pallas_scaffold_mgpu(
    h_pair: Float[Array, "S Dst R I"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW2PallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
) -> tuple[Float[Array, "S T H"], Float[Array, "S Dst R H"]]:
    """Compute W2 with Pallas, then combine routes with the JAX reference."""

    route_y = source_push_semantic_w2_pallas_mgpu(
        h_pair,
        w_down,
        plan,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
    )
    return source_push_semantic_combine_jax(route_y, plan), route_y


def source_push_semantic_w2_expert_major_reference_jax(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    valid: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C H"]:
    """Reference W2 over destination expert-major rows."""

    _validate_semantic_w2_expert_major_request(h_expert, w_down, valid)
    route_y = jnp.einsum(
        "deci,deih->dech",
        h_expert.astype(jnp.float32),
        w_down.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    return route_y * valid.astype(jnp.float32)[..., None]


def source_push_semantic_forward_return_expert_major_reference_jax(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
) -> tuple[Float[Array, "S T H"], Float[Array, "S T K H"]]:
    """Reference return/combine from destination expert-major route rows."""

    route_y_pair = source_push_semantic_expert_major_to_pair_jax(route_y_expert, plan)
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    source_index = jnp.broadcast_to(source_index, plan.assignment_ids.shape)
    token_ids = jnp.where(plan.valid_mask, plan.token_ids, plan.tokens_per_source)
    route_slots = jnp.where(plan.valid_mask, plan.route_slots, plan.topk)
    weighted_pair = route_y_pair * plan.route_weights[..., None].astype(route_y_pair.dtype)
    weighted_pair = jnp.where(plan.valid_mask[..., None], weighted_pair, jnp.zeros((), dtype=weighted_pair.dtype))
    route_by_slot = jnp.zeros(
        (
            plan.assignment_ids.shape[0],
            plan.tokens_per_source,
            plan.topk,
            route_y_expert.shape[-1],
        ),
        dtype=route_y_expert.dtype,
    )
    route_by_slot = route_by_slot.at[source_index, token_ids, route_slots].set(weighted_pair, mode="drop")
    return jnp.sum(route_by_slot, axis=2), route_by_slot


def source_push_semantic_forward_return_source_gather_reference_jax(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
) -> Float[Array, "S T H"]:
    """Reference source-owned return/combine from reverse route metadata."""

    _validate_semantic_forward_return_source_gather_request(
        route_y_expert,
        plan,
        SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=1),
    )
    reverse_route = source_push_semantic_reverse_route_jax(plan)
    route_weights = _source_push_semantic_reverse_route_weights_jax(plan)
    return _source_push_semantic_forward_return_source_gather_reference_from_reverse_jax(
        route_y_expert,
        reverse_route,
        route_weights,
    )


def source_push_semantic_forward_return_expert_major_lookup_metadata_jax(
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
) -> tuple[Int[Array, "Dst E C"], Int[Array, "Dst E C"], Float[Array, "Dst E C"], Bool[Array, "Dst E C"]]:
    """Build expert-major row lookup metadata for source-owned return sum."""

    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    expert_ids, expert_rows = _source_push_semantic_expert_row_indices_from_plan(plan)
    expert_valid = plan.valid_mask & (expert_rows < rows_per_expert_capacity)
    src_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    source_shape = (plan.assignment_ids.shape[1], plan.xcounts.shape[-1], rows_per_expert_capacity)
    scatter_rows = jnp.where(expert_valid, expert_rows, rows_per_expert_capacity)
    source_lookup = jnp.zeros(source_shape, dtype=jnp.int32)
    token_lookup = jnp.zeros(source_shape, dtype=jnp.int32)
    weight_lookup = jnp.zeros(source_shape, dtype=plan.route_weights.dtype)
    valid_lookup = jnp.zeros(source_shape, dtype=jnp.bool_)
    source_lookup = source_lookup.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(expert_valid, src_index, 0),
        mode="drop",
    )
    token_lookup = token_lookup.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(expert_valid, jnp.maximum(plan.token_ids, 0), 0),
        mode="drop",
    )
    weight_lookup = weight_lookup.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(expert_valid, plan.route_weights, jnp.zeros((), dtype=plan.route_weights.dtype)),
        mode="drop",
    )
    valid_lookup = valid_lookup.at[dst_index, expert_ids, scatter_rows].set(expert_valid, mode="drop")
    return source_lookup, token_lookup, weight_lookup, valid_lookup


def source_push_semantic_forward_return_queue_metadata_jax(
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    return_row_block: int,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> SourcePushSemanticForwardReturnQueueMetadata:
    """Build direct-return queue metadata inside the JAX boundary."""

    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    if return_row_block <= 0:
        raise ValueError(f"return_row_block must be positive, got {return_row_block}")
    source_count = plan.assignment_ids.shape[0]
    destination_count = plan.assignment_ids.shape[1]
    expert_count = plan.xcounts.shape[-1]
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    if queue is None:
        rows_per_pair = plan.assignment_ids.shape[-1]
        entries_per_dst = (rows_per_pair + return_row_block - 1) // return_row_block + expert_count - 1
        queue = source_push_semantic_queue_metadata_jax(
            plan,
            return_row_block=return_row_block,
            entries_per_dst=entries_per_dst,
        )
    else:
        _validate_direct_return_queue(plan, queue, return_row_block)

    dst_index = jnp.arange(destination_count, dtype=jnp.int32)[:, None]
    recv_src_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    source_index = (dst_index + recv_src_ordinal) % source_count
    dst_ordinal = jnp.broadcast_to((-recv_src_ordinal) % source_count, source_index.shape)
    recv_local_expert = queue.local_expert.at[source_index, dst_ordinal].get()
    recv_local_row_start = queue.local_row_start.at[source_index, dst_ordinal].get()
    recv_valid_rows = queue.valid_rows.at[source_index, dst_ordinal].get()
    safe_expert = jnp.maximum(recv_local_expert, 0)
    expert_source_base = source_row_base_by_expert.at[
        jnp.arange(destination_count, dtype=jnp.int32)[:, None, None],
        source_index[:, :, None],
        safe_expert,
    ].get()
    recv_expert_row_start = expert_source_base + recv_local_row_start
    recv_expert_row_start = jnp.where(recv_valid_rows > 0, recv_expert_row_start, 0).astype(jnp.int32)

    route_weight = _source_push_semantic_reverse_route_weights_jax(plan)
    route_weight = jnp.where(queue.route_valid, route_weight, jnp.zeros((), dtype=route_weight.dtype))

    return SourcePushSemanticForwardReturnQueueMetadata(
        queue_local_expert=queue.local_expert,
        queue_local_row_start=queue.local_row_start,
        queue_valid_rows=queue.valid_rows,
        recv_local_expert=recv_local_expert,
        recv_expert_row_start=recv_expert_row_start,
        recv_valid_rows=recv_valid_rows,
        queue_dst_ord=queue.route_dst_ordinal,
        queue_entry=queue.route_entry,
        queue_row=queue.route_queue_row,
        route_weight=route_weight,
        route_valid=queue.route_valid,
    )


def source_push_semantic_forward_return_direct_to_source_reference_jax(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    route_buffer_dtype: jnp.dtype = jnp.bfloat16,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
) -> Float[Array, "S DstOrd Q M H"]:
    """Reference source-visible direct return queue."""

    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(
        route_y_expert,
        plan,
        SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=block_sizes.hidden_block),
    )
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    source_count, destination_count = plan.assignment_ids.shape[:2]
    metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=route_y_expert.shape[2],
        return_row_block=block_sizes.row_block,
        source_row_base_by_expert=source_row_base_by_expert,
        queue=queue,
    )
    src_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None]
    actual_dst = (src_index + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(metadata.queue_local_expert, 0)
    expert_source_base = source_row_base_by_expert.at[actual_dst, src_index, safe_expert].get()
    expert_row_start = expert_source_base + metadata.queue_local_row_start
    queue_row = jnp.arange(block_sizes.row_block, dtype=jnp.int32)[None, None, None, :]
    expert_row = expert_row_start[..., None] + queue_row
    safe_expert_row = jnp.minimum(expert_row, route_y_expert.shape[2] - 1)
    return_y_get = route_y_expert.at[
        actual_dst[..., None],
        safe_expert[..., None],
        safe_expert_row,
    ]
    source_queue_sharding = _source_major_out_sharding_from_named_input(route_y_expert, rank=5)
    if source_queue_sharding is None:
        return_y = return_y_get.get()
    else:
        return_y = return_y_get.get(out_sharding=source_queue_sharding)
    valid = queue_row < metadata.queue_valid_rows[..., None]
    return jnp.where(valid[..., None], return_y, jnp.zeros((), dtype=return_y.dtype)).astype(route_buffer_dtype)


def source_push_semantic_forward_combine_source_gather_reference_jax(
    return_y: Float[Array, "S DstOrd Q M H"],
    metadata: SourcePushSemanticForwardReturnQueueMetadata,
    *,
    output_dtype: jnp.dtype = jnp.bfloat16,
) -> Float[Array, "S T H"]:
    """Reference source-owned combine from direct-return queue metadata."""

    source_count, tokens_per_source, topk = metadata.queue_dst_ord.shape
    src_index = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    route_value_get = return_y.at[
        src_index,
        metadata.queue_dst_ord,
        metadata.queue_entry,
        metadata.queue_row,
    ]
    route_value_sharding = _source_major_out_sharding_from_named_input(return_y, rank=4)
    if route_value_sharding is None:
        route_value = route_value_get.get(mode="clip")
    else:
        route_value = route_value_get.get(mode="clip", out_sharding=route_value_sharding)
    weighted = route_value.astype(jnp.float32) * metadata.route_weight.astype(jnp.float32)[..., None]
    weighted = jnp.where(metadata.route_valid[..., None], weighted, jnp.zeros((), dtype=weighted.dtype))
    return jnp.sum(weighted, axis=2).reshape(source_count, tokens_per_source, return_y.shape[-1]).astype(output_dtype)


def source_push_semantic_forward_combine_source_gather_pallas_mgpu(
    return_y: Float[Array, "S DstOrd Q M H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    output_dtype: jnp.dtype = jnp.bfloat16,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Source-local combine from direct-return queue rows."""

    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=1,
        return_row_block=block_sizes.row_block,
        queue=queue,
    )
    _validate_semantic_forward_combine_source_gather_request(
        return_y,
        plan,
        block_sizes,
        entries_per_dst=metadata.queue_local_expert.shape[2],
    )
    if interpret:
        return source_push_semantic_forward_combine_source_gather_reference_jax(
            return_y,
            metadata,
            output_dtype=output_dtype,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source combine requires a GPU backend")
    if mesh is None:
        return _source_push_semantic_forward_combine_source_gather_pallas_call(
            return_y,
            metadata.queue_dst_ord,
            metadata.queue_entry,
            metadata.queue_row,
            metadata.route_weight,
            metadata.route_valid,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            output_dtype=output_dtype,
            interpret=interpret,
        )
    source_sharding_5d = jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    source_sharding_3d = jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    return_y = jax.sharding.reshard(return_y, source_sharding_5d)
    queue_dst_ord = jax.sharding.reshard(metadata.queue_dst_ord, source_sharding_3d)
    queue_entry = jax.sharding.reshard(metadata.queue_entry, source_sharding_3d)
    queue_row = jax.sharding.reshard(metadata.queue_row, source_sharding_3d)
    route_weight = jax.sharding.reshard(metadata.route_weight, source_sharding_3d)
    route_valid = jax.sharding.reshard(metadata.route_valid, source_sharding_3d)
    return _source_push_semantic_forward_combine_source_gather_sharded_mgpu_kernel(
        mesh,
        return_y,
        queue_dst_ord,
        queue_entry,
        queue_row,
        route_weight,
        route_valid,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_dtype=output_dtype,
        interpret=interpret,
    )


def source_push_semantic_forward_return_direct_to_source_pallas_mgpu(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    valid: Bool[Array, "Dst E C"],
    plan: SourcePushSemanticPlan,
    *,
    w2_block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes | None = None,
    return_block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    route_buffer_dtype: jnp.dtype = jnp.bfloat16,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S DstOrd Q M H"]:
    """Compute W2 and remote-write unweighted route rows into source-owned queues."""

    w2_block_sizes = (
        SourcePushSemanticW2ExpertMajorPallasBlockSizes.get_default() if w2_block_sizes is None else w2_block_sizes
    )
    return_block_sizes = (
        SourcePushSemanticForwardReturnPallasBlockSizes.get_default()
        if return_block_sizes is None
        else return_block_sizes
    )
    source_row_base_by_expert = _resolve_source_row_base_by_expert(plan, source_row_base_by_expert)
    if interpret:
        _validate_semantic_w2_expert_major_request(h_expert, w_down, valid)
        route_y_expert = source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, valid)
        return source_push_semantic_forward_return_direct_to_source_reference_jax(
            route_y_expert,
            plan,
            block_sizes=return_block_sizes,
            route_buffer_dtype=route_buffer_dtype,
            source_row_base_by_expert=source_row_base_by_expert,
            queue=queue,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 direct source return requires a GPU backend")
    if mesh is None:
        raise ValueError("mesh must be provided for non-interpreted direct source return")
    _validate_semantic_w2_direct_return_request(h_expert, w_down, valid, plan, w2_block_sizes, return_block_sizes)

    metadata = source_push_semantic_forward_return_queue_metadata_jax(
        plan,
        rows_per_expert_capacity=h_expert.shape[2],
        return_row_block=return_block_sizes.row_block,
        source_row_base_by_expert=source_row_base_by_expert,
        queue=queue,
    )
    h_expert = jax.sharding.reshard(h_expert, _destination_major_sharding(mesh, rank=4))
    w_down = jax.sharding.reshard(w_down, _destination_major_sharding(mesh, rank=4))
    recv_local_expert = _constrain_destination_major_metadata(metadata.recv_local_expert, mesh)
    recv_expert_row_start = _constrain_destination_major_metadata(metadata.recv_expert_row_start, mesh)
    recv_valid_rows = _constrain_destination_major_metadata(metadata.recv_valid_rows, mesh)
    return _source_push_semantic_w2_direct_return_to_source_sharded_mgpu_kernel(
        mesh,
        h_expert,
        w_down,
        recv_local_expert,
        recv_expert_row_start,
        recv_valid_rows,
        source_count=plan.assignment_ids.shape[0],
        entries_per_dst=metadata.queue_local_expert.shape[2],
        return_row_block=return_block_sizes.row_block,
        output_dtype=route_buffer_dtype,
        row_block=w2_block_sizes.row_block,
        intermediate_block=w2_block_sizes.intermediate_block,
        hidden_block=w2_block_sizes.hidden_block,
    )


def source_push_semantic_forward_expert_major_direct_return_combine_pallas_mgpu(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    valid: Int[Array, "Dst E C"],
    plan: SourcePushSemanticPlan,
    *,
    w2_block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes | None = None,
    return_block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    route_buffer_dtype: jnp.dtype = jnp.bfloat16,
    output_dtype: jnp.dtype = jnp.bfloat16,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    queue: SourcePushSemanticQueueMetadata | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Direct W2 source-return followed by source-owned route combine."""

    return_block_sizes = (
        SourcePushSemanticForwardReturnPallasBlockSizes.get_default()
        if return_block_sizes is None
        else return_block_sizes
    )
    return_y = source_push_semantic_forward_return_direct_to_source_pallas_mgpu(
        h_expert,
        w_down,
        valid,
        plan,
        w2_block_sizes=w2_block_sizes,
        return_block_sizes=return_block_sizes,
        route_buffer_dtype=route_buffer_dtype,
        source_row_base_by_expert=source_row_base_by_expert,
        queue=queue,
        interpret=interpret,
        mesh=mesh,
    )
    return source_push_semantic_forward_combine_source_gather_pallas_mgpu(
        return_y,
        plan,
        block_sizes=return_block_sizes,
        output_dtype=output_dtype,
        queue=queue,
        interpret=interpret,
        mesh=mesh,
    )


def source_push_semantic_forward_return_source_gather_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Gather expert-major route rows by source token and combine route slots.

    This is the source-owned non-atomic return path. The wrapper derives
    reverse route metadata inside the JAX boundary with
    ``source_push_semantic_reverse_route_jax(plan)``.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 source-gather return requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_source_gather_request(route_y_expert, plan, block_sizes)
    reverse_route = source_push_semantic_reverse_route_jax(plan)
    route_weights = _source_push_semantic_reverse_route_weights_jax(plan)
    if mesh is not None:
        return _source_push_semantic_forward_return_source_gather_sharded_mgpu_kernel(
            mesh,
            route_y_expert,
            reverse_route.route_dst,
            reverse_route.route_expert,
            reverse_route.route_expert_row,
            reverse_route.route_valid,
            route_weights,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_forward_return_source_gather_pallas_call(
        route_y_expert,
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
        reverse_route.route_valid,
        route_weights,
        dst_offset=jnp.asarray(0, dtype=jnp.int32),
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_expert_major_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T H"], Float[Array, "S T K H"]]:
    """Return weighted expert-major W2 rows to source route slots, then sum slots."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 return requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    if mesh is None:
        route_by_slot = _source_push_semantic_forward_return_expert_major_pallas_call(
            route_y_expert,
            plan.token_ids,
            plan.route_slots,
            plan.route_weights,
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
        route_by_slot = _source_push_semantic_forward_return_expert_major_sharded_mgpu_kernel(
            mesh,
            route_y_expert,
            plan.token_ids,
            plan.route_slots,
            plan.route_weights,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            tokens_per_source=plan.tokens_per_source,
            topk=plan.topk,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return jnp.sum(route_by_slot, axis=2), route_by_slot


def source_push_semantic_forward_return_slot_reduce_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Benchmark diagnostic: materialize route slots, then reduce slots exactly.

    This intentionally exposes the existing exact non-atomic route-slot write
    path as a y-only benchmark. It is not a production return API.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 slot-reduce return diagnostic requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    if mesh is None:
        route_by_slot = _source_push_semantic_forward_return_expert_major_pallas_call(
            route_y_expert,
            plan.token_ids,
            plan.route_slots,
            plan.route_weights,
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
        return jnp.sum(route_by_slot, axis=2)
    return _source_push_semantic_forward_return_slot_reduce_expert_major_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        plan.token_ids,
        plan.route_slots,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_slot_reduce_owner_sharded_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Materialize exact route slots, then reduce to source-owner shards.

    This is an atomic-free forward-return experiment that keeps the
    destination-major W2 output layout. It differs from
    ``source_push_semantic_forward_return_slot_reduce_pallas_mgpu`` only in the
    final collective shape: route slots are scattered onto source owners before
    the local route-slot sum.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError(
            "Pallas/MGPU semantic W2 owner-sharded slot-reduce return diagnostic requires a GPU backend"
        )
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    if mesh is None:
        route_by_slot = _source_push_semantic_forward_return_expert_major_pallas_call(
            route_y_expert,
            plan.token_ids,
            plan.route_slots,
            plan.route_weights,
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
        return jnp.sum(route_by_slot, axis=2)
    return _source_push_semantic_forward_return_slot_reduce_expert_major_owner_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        plan.token_ids,
        plan.route_slots,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        topk=plan.topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_remote_source_gather_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Gather remote expert-major rows directly onto the source-token owner."""

    if interpret:
        return source_push_semantic_forward_return_source_gather_reference_jax(route_y_expert, plan)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 remote source-gather return requires a GPU backend")
    if mesh is None:
        raise ValueError("mesh must be provided for non-interpreted remote source-gather return")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_source_gather_request(route_y_expert, plan, block_sizes)
    reverse_route = source_push_semantic_reverse_route_jax(plan)
    route_weights = _source_push_semantic_reverse_route_weights_jax(plan)
    return _source_push_semantic_forward_return_remote_source_gather_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
        reverse_route.route_valid,
        route_weights,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_sum_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Return weighted expert-major W2 rows and sum directly into source tokens."""

    if interpret:
        y, _route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(route_y_expert, plan)
        return y
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 direct return requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    if mesh is None:
        return _source_push_semantic_forward_return_sum_expert_major_pallas_call(
            route_y_expert,
            plan.token_ids,
            plan.route_weights,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            tokens_per_source=plan.tokens_per_source,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_forward_return_sum_expert_major_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_sum_owner_sharded_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh,
) -> Float[Array, "S T H"]:
    """Return weighted expert-major W2 rows with the source-token axis sharded."""

    if interpret:
        y, _route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(route_y_expert, plan)
        return y
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 owner-sharded return requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    return _source_push_semantic_forward_return_sum_expert_major_owner_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        tokens_per_source=plan.tokens_per_source,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_sum_lookup_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "S T H"]:
    """Return weighted expert-major rows using precomputed row lookup metadata."""

    if interpret:
        y, _route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(route_y_expert, plan)
        return y
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 lookup return requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    source_lookup, token_lookup, weight_lookup, valid_lookup = (
        source_push_semantic_forward_return_expert_major_lookup_metadata_jax(
            plan,
            rows_per_expert_capacity=route_y_expert.shape[2],
        )
    )
    if mesh is None:
        return _source_push_semantic_forward_return_sum_lookup_expert_major_pallas_call(
            route_y_expert,
            source_lookup,
            token_lookup,
            weight_lookup,
            valid_lookup,
            source_count=plan.assignment_ids.shape[0],
            tokens_per_source=plan.tokens_per_source,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    source_lookup = _constrain_destination_major_metadata(source_lookup, mesh)
    token_lookup = _constrain_destination_major_metadata(token_lookup, mesh)
    weight_lookup = _constrain_destination_major_metadata(weight_lookup, mesh)
    valid_lookup = _constrain_destination_major_metadata(valid_lookup, mesh)
    return _source_push_semantic_forward_return_sum_lookup_expert_major_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        source_lookup,
        token_lookup,
        weight_lookup,
        valid_lookup,
        source_count=plan.assignment_ids.shape[0],
        tokens_per_source=plan.tokens_per_source,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_sum_lookup_owner_sharded_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None,
) -> Float[Array, "S T H"]:
    """Return lookup-weighted expert-major rows with source-token output sharded."""

    if interpret:
        y, _route_by_slot = source_push_semantic_forward_return_expert_major_reference_jax(route_y_expert, plan)
        return y
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 lookup owner-sharded return requires a GPU backend")
    if mesh is None:
        raise ValueError("mesh must be provided for non-interpreted owner-sharded lookup return")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    source_lookup, token_lookup, weight_lookup, valid_lookup = (
        source_push_semantic_forward_return_expert_major_lookup_metadata_jax(
            plan,
            rows_per_expert_capacity=route_y_expert.shape[2],
        )
    )
    source_lookup = _constrain_destination_major_metadata(source_lookup, mesh)
    token_lookup = _constrain_destination_major_metadata(token_lookup, mesh)
    weight_lookup = _constrain_destination_major_metadata(weight_lookup, mesh)
    valid_lookup = _constrain_destination_major_metadata(valid_lookup, mesh)
    return _source_push_semantic_forward_return_sum_lookup_expert_major_owner_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        source_lookup,
        token_lookup,
        weight_lookup,
        valid_lookup,
        source_count=plan.assignment_ids.shape[0],
        tokens_per_source=plan.tokens_per_source,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_forward_return_copy_only_pallas_mgpu(
    route_y_expert: Float[Array, "Dst E C H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Diagnostic return traversal that weights rows without source-token atomics."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 return copy diagnostic requires a GPU backend")
    block_sizes = SourcePushSemanticForwardReturnPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_forward_return_expert_major_request(route_y_expert, plan, block_sizes)
    if mesh is None:
        return _source_push_semantic_forward_return_copy_only_expert_major_pallas_call(
            route_y_expert,
            plan.token_ids,
            plan.route_weights,
            plan.xcounts,
            plan.pair_expert_base,
            plan.src_base_by_expert,
            dst_offset=jnp.asarray(0, dtype=jnp.int32),
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            interpret=interpret,
        )
    return _source_push_semantic_forward_return_copy_only_expert_major_sharded_mgpu_kernel(
        mesh,
        route_y_expert,
        plan.token_ids,
        plan.route_weights,
        plan.xcounts,
        plan.pair_expert_base,
        plan.src_base_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
    )


def source_push_semantic_w2_expert_major_pallas_mgpu(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Compute W2 from prepacked destination expert-major activations.

    This is the production-shaped W2 compute boundary after source-push W13 has
    produced ``h_expert[dst, expert, row, intermediate]``. The GPU path uses
    explicit WGMMA over aligned expert-major tiles; ``interpret=True`` uses an
    independent JAX reference and remains usable on CPU.
    """

    block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w2_expert_major_request(h_expert, w_down, valid)
    if interpret:
        return source_push_semantic_w2_expert_major_reference_jax(h_expert, w_down, valid)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 from expert-major h requires a GPU backend")
    _validate_semantic_w2_expert_major_pallas_request(h_expert, w_down, valid, block_sizes)
    valid = valid.astype(jnp.float32)
    if mesh is None:
        route_y = _source_push_semantic_w2_expert_major_pallas_call(
            h_expert,
            w_down,
            valid,
            row_block=block_sizes.row_block,
            intermediate_block=block_sizes.intermediate_block,
            hidden_block=block_sizes.hidden_block,
        )
    else:
        route_y = _source_push_semantic_w2_expert_major_sharded_mgpu_kernel(
            mesh,
            h_expert,
            w_down,
            valid,
            row_block=block_sizes.row_block,
            intermediate_block=block_sizes.intermediate_block,
            hidden_block=block_sizes.hidden_block,
        )
    return route_y


def source_push_semantic_w2_expert_major_assume_zero_invalid_pallas_mgpu(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    *,
    block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Compute expert-major W2 assuming invalid rows are already zeroed.

    This diagnostic path matches the production W13->W2 contract where W13
    writes zero activations for invalid expert-major rows. It intentionally does
    not replace the masked prepacked API, because arbitrary prepacked inputs can
    carry nonzero invalid-row payloads.
    """

    block_sizes = SourcePushSemanticW2ExpertMajorPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w2_expert_major_h_w_request(h_expert, w_down)
    if interpret:
        return _source_push_semantic_w2_expert_major_unmasked_reference_jax(h_expert, w_down)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W2 from expert-major h requires a GPU backend")
    _validate_semantic_w2_expert_major_unmasked_pallas_request(h_expert, w_down, block_sizes)
    if mesh is None:
        return _source_push_semantic_w2_expert_major_unmasked_pallas_call(
            h_expert,
            w_down,
            row_block=block_sizes.row_block,
            intermediate_block=block_sizes.intermediate_block,
            hidden_block=block_sizes.hidden_block,
        )
    return _source_push_semantic_w2_expert_major_unmasked_sharded_mgpu_kernel(
        mesh,
        h_expert,
        w_down,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
    )


def _source_push_semantic_forward_return_expert_major_pallas_call(
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
    source_count, _global_dst_count, _rows_per_pair = token_ids.shape
    local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim = route_y_expert.shape
    output_shape = jax.ShapeDtypeStruct(
        (source_count, tokens_per_source, topk, hidden_dim),
        route_y_expert.dtype,
    )
    route_by_slot_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            local_dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        input_output_aliases={8: 0},
        interpret=interpret,
        name="source_push_semantic_forward_return_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            token_ids,
            route_slots,
            route_weights,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
        ),
    )(
        route_y_expert,
        token_ids,
        route_slots,
        route_weights,
        xcounts,
        pair_expert_base,
        src_base_by_expert,
        dst_offset,
        route_by_slot_init,
    )


def _source_push_semantic_forward_return_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
        route_y_local: Float[Array, "1 E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "1 S T K H"]:
        partial = _source_push_semantic_forward_return_expert_major_pallas_call(
            route_y_local,
            token_ids_global,
            route_slots_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS), dtype=jnp.int32),
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
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(
        route_y_expert,
        token_ids,
        route_slots,
        route_weights,
        xcounts,
        pair_expert_base,
        src_base_by_expert,
    )
    return jnp.sum(partials_by_destination, axis=0)


def _source_push_semantic_forward_return_slot_reduce_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
        route_y_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "S T H"]:
        route_by_slot = _source_push_semantic_forward_return_expert_major_pallas_call(
            route_y_local,
            token_ids_global,
            route_slots_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS) * route_y_local.shape[0], dtype=jnp.int32),
            tokens_per_source=tokens_per_source,
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(jnp.sum(route_by_slot, axis=2), SOURCE_PUSH_MESH_AXIS)

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
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(
        route_y_expert,
        token_ids,
        route_slots,
        route_weights,
        xcounts,
        pair_expert_base,
        src_base_by_expert,
    )


def _source_push_semantic_forward_return_slot_reduce_expert_major_owner_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_slots: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
        route_y_local: Float[Array, "1 E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_slots_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "1 T H"]:
        route_by_slot = _source_push_semantic_forward_return_expert_major_pallas_call(
            route_y_local,
            token_ids_global,
            route_slots_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS), dtype=jnp.int32),
            tokens_per_source=tokens_per_source,
            topk=topk,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        route_by_slot = jax.lax.psum_scatter(
            route_by_slot,
            SOURCE_PUSH_MESH_AXIS,
            scatter_dimension=0,
            tiled=True,
        )
        return jnp.sum(route_by_slot, axis=2)

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
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(
        route_y_expert,
        token_ids,
        route_slots,
        route_weights,
        xcounts,
        pair_expert_base,
        src_base_by_expert,
    )


def _source_push_semantic_forward_return_sum_expert_major_pallas_call(
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
    source_count, _global_dst_count, _experts_per_rank = xcounts.shape
    local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim = route_y_expert.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), jnp.float32)
    y_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_sum_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            local_dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        input_output_aliases={7: 0},
        interpret=interpret,
        name="source_push_semantic_forward_return_sum_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            token_ids,
            route_weights,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
        ),
    )(route_y_expert, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert, dst_offset, y_init)


def _source_push_semantic_forward_return_sum_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
        route_y_local: Float[Array, "1 E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "S T H"]:
        partial = _source_push_semantic_forward_return_sum_expert_major_pallas_call(
            route_y_local,
            token_ids_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS), dtype=jnp.int32),
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
            P(None, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(route_y_expert, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_forward_return_sum_expert_major_owner_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
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
        route_y_local: Float[Array, "1 E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "1 T H"]:
        partial = _source_push_semantic_forward_return_sum_expert_major_pallas_call(
            route_y_local,
            token_ids_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS), dtype=jnp.int32),
            tokens_per_source=tokens_per_source,
            row_block=row_block,
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
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(route_y_expert, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_forward_return_sum_lookup_expert_major_pallas_call(
    route_y_expert: Float[Array, "Dst E C H"],
    source_lookup: Int[Array, "Dst E C"],
    token_lookup: Int[Array, "Dst E C"],
    weight_lookup: Float[Array, "Dst E C"],
    valid_lookup: Bool[Array, "Dst E C"],
    *,
    source_count: int,
    tokens_per_source: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim = route_y_expert.shape
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), jnp.float32)
    y_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_sum_lookup_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            local_dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        input_output_aliases={5: 0},
        interpret=interpret,
        name="source_push_semantic_forward_return_sum_lookup_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            source_lookup,
            token_lookup,
            weight_lookup,
            valid_lookup,
            output_shape,
        ),
    )(route_y_expert, source_lookup, token_lookup, weight_lookup, valid_lookup.astype(jnp.int32), y_init)


def _source_push_semantic_forward_return_sum_lookup_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    source_lookup: Int[Array, "Dst E C"],
    token_lookup: Int[Array, "Dst E C"],
    weight_lookup: Float[Array, "Dst E C"],
    valid_lookup: Bool[Array, "Dst E C"],
    *,
    source_count: int,
    tokens_per_source: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        route_y_local: Float[Array, "1 E C H"],
        source_lookup_local: Int[Array, "1 E C"],
        token_lookup_local: Int[Array, "1 E C"],
        weight_lookup_local: Float[Array, "1 E C"],
        valid_lookup_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "S T H"]:
        partial = _source_push_semantic_forward_return_sum_lookup_expert_major_pallas_call(
            route_y_local,
            source_lookup_local,
            token_lookup_local,
            weight_lookup_local,
            valid_lookup_local,
            source_count=source_count,
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
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )(route_y_expert, source_lookup, token_lookup, weight_lookup, valid_lookup)


def _source_push_semantic_forward_return_sum_lookup_expert_major_owner_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    source_lookup: Int[Array, "Dst E C"],
    token_lookup: Int[Array, "Dst E C"],
    weight_lookup: Float[Array, "Dst E C"],
    valid_lookup: Bool[Array, "Dst E C"],
    *,
    source_count: int,
    tokens_per_source: int,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        route_y_local: Float[Array, "1 E C H"],
        source_lookup_local: Int[Array, "1 E C"],
        token_lookup_local: Int[Array, "1 E C"],
        weight_lookup_local: Float[Array, "1 E C"],
        valid_lookup_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 T H"]:
        partial = _source_push_semantic_forward_return_sum_lookup_expert_major_pallas_call(
            route_y_local,
            source_lookup_local,
            token_lookup_local,
            weight_lookup_local,
            valid_lookup_local,
            source_count=source_count,
            tokens_per_source=tokens_per_source,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum_scatter(partial, SOURCE_PUSH_MESH_AXIS, scatter_dimension=0, tiled=True)

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
    )(route_y_expert, source_lookup, token_lookup, weight_lookup, valid_lookup)


def _source_push_semantic_forward_return_copy_only_expert_major_pallas_call(
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    dst_offset: Int[Array, ""],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "Dst E C H"]:
    local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim = route_y_expert.shape
    output_shape = jax.ShapeDtypeStruct(
        (local_dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim),
        jnp.float32,
    )
    weighted_route_init = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_copy_only_expert_major_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            local_dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        input_output_aliases={7: 0},
        interpret=interpret,
        name="source_push_semantic_forward_return_copy_only_expert_major_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            token_ids,
            route_weights,
            xcounts,
            pair_expert_base,
            src_base_by_expert,
            dst_offset,
            output_shape,
            output_shape,
        ),
    )(
        route_y_expert,
        token_ids,
        route_weights,
        xcounts,
        pair_expert_base,
        src_base_by_expert,
        dst_offset,
        weighted_route_init,
    )


def _source_push_semantic_forward_return_copy_only_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    token_ids: Int[Array, "S Dst R"],
    route_weights: Float[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "Dst E C H"]:
    def local_fn(
        route_y_local: Float[Array, "Dst E C H"],
        token_ids_global: Int[Array, "S Dst R"],
        route_weights_global: Float[Array, "S Dst R"],
        xcounts_global: Int[Array, "S Dst E"],
        pair_expert_base_global: Int[Array, "S Dst E"],
        src_base_by_expert_global: Int[Array, "Dst S E"],
    ) -> Float[Array, "Dst E C H"]:
        return _source_push_semantic_forward_return_copy_only_expert_major_pallas_call(
            route_y_local,
            token_ids_global,
            route_weights_global,
            xcounts_global,
            pair_expert_base_global,
            src_base_by_expert_global,
            jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS) * route_y_local.shape[0], dtype=jnp.int32),
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
            P(None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(route_y_expert, token_ids, route_weights, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_forward_combine_source_gather_pallas_call(
    return_y: Float[Array, "S DstOrd Q M H"],
    queue_dst_ord: Int[Array, "S T K"],
    queue_entry: Int[Array, "S T K"],
    queue_row: Int[Array, "S T K"],
    route_weight: Float[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "S T H"]:
    source_count, tokens_per_source, topk = queue_dst_ord.shape
    hidden_dim = return_y.shape[-1]
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_combine_source_gather_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_forward_combine_source_gather_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            return_y,
            queue_dst_ord,
            queue_entry,
            queue_row,
            route_weight,
            route_valid,
            output_shape,
        ),
    )(return_y, queue_dst_ord, queue_entry, queue_row, route_weight, route_valid.astype(jnp.int32))


def _source_push_semantic_forward_combine_source_gather_local_pallas_call(
    return_y: Float[Array, "DstOrd Q M H"],
    queue_dst_ord: Int[Array, "T K"],
    queue_entry: Int[Array, "T K"],
    queue_row: Int[Array, "T K"],
    route_weight: Float[Array, "T K"],
    route_valid: Bool[Array, "T K"],
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "T H"]:
    tokens_per_source, topk = queue_dst_ord.shape
    hidden_dim = return_y.shape[-1]
    output_shape = jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), output_dtype)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_combine_source_gather_local_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(tokens_per_source // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_forward_combine_source_gather_local_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            return_y,
            queue_dst_ord,
            queue_entry,
            queue_row,
            route_weight,
            route_valid,
            output_shape,
        ),
    )(return_y, queue_dst_ord, queue_entry, queue_row, route_weight, route_valid.astype(jnp.int32))


def _source_push_semantic_forward_combine_source_gather_sharded_mgpu_kernel(
    mesh: Mesh,
    return_y: Float[Array, "S DstOrd Q M H"],
    queue_dst_ord: Int[Array, "S T K"],
    queue_entry: Int[Array, "S T K"],
    queue_row: Int[Array, "S T K"],
    route_weight: Float[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        return_y_local: Float[Array, "1 DstOrd Q M H"],
        queue_dst_ord_local: Int[Array, "1 T K"],
        queue_entry_local: Int[Array, "1 T K"],
        queue_row_local: Int[Array, "1 T K"],
        route_weight_local: Float[Array, "1 T K"],
        route_valid_local: Bool[Array, "1 T K"],
    ) -> Float[Array, "1 T H"]:
        y = _source_push_semantic_forward_combine_source_gather_local_pallas_call(
            return_y_local[0],
            queue_dst_ord_local[0],
            queue_entry_local[0],
            queue_row_local[0],
            route_weight_local[0],
            route_valid_local[0],
            row_block=row_block,
            hidden_block=hidden_block,
            output_dtype=output_dtype,
            interpret=interpret,
        )
        return y[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(return_y, queue_dst_ord, queue_entry, queue_row, route_weight, route_valid)


def _destination_offset_for_local_shard(local_dst_count: int) -> Int[Array, ""]:
    return jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS) * local_dst_count, dtype=jnp.int32)


def _sharded_forward_return_remote_write_completion_barrier(mesh: Mesh):
    """Synchronize direct-return remote writes before source-local combine."""

    def local_fn(return_y_local: Float[Array, "1 DstOrd Q M H"]) -> Float[Array, "1 DstOrd Q M H"]:
        value = return_y_local[0, 0, 0, 0, 0].astype(jnp.float32)
        barrier = jax.lax.psum(value, SOURCE_PUSH_MESH_AXIS)
        zero = barrier - jax.lax.optimization_barrier(barrier)
        return return_y_local.at[0, 0, 0, 0, 0].add(zero.astype(return_y_local.dtype))

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )


def _source_push_semantic_forward_return_source_gather_pallas_call(
    route_y_expert: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    dst_offset: Int[Array, ""],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    source_count, tokens_per_source, topk = route_dst.shape
    hidden_dim = route_y_expert.shape[-1]
    output_shape = jax.ShapeDtypeStruct((source_count, tokens_per_source, hidden_dim), jnp.float32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_source_gather_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(source_count, tokens_per_source // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_forward_return_source_gather_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            route_dst,
            route_expert,
            route_expert_row,
            route_valid,
            route_weights,
            dst_offset,
            output_shape,
        ),
    )(route_y_expert, route_dst, route_expert, route_expert_row, route_valid, route_weights, dst_offset)


def _source_push_semantic_forward_return_source_gather_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    def local_fn(
        route_y_expert_local: Float[Array, "Dst E C H"],
        route_dst_global: Int[Array, "S T K"],
        route_expert_global: Int[Array, "S T K"],
        route_expert_row_global: Int[Array, "S T K"],
        route_valid_global: Bool[Array, "S T K"],
        route_weights_global: Float[Array, "S T K"],
    ) -> Float[Array, "S T H"]:
        partial_by_destination = _source_push_semantic_forward_return_source_gather_by_destination_pallas_call(
            route_y_expert_local,
            route_dst_global,
            route_expert_global,
            route_expert_row_global,
            route_valid_global,
            route_weights_global,
            dst_offset=jnp.asarray(jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS), dtype=jnp.int32),
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return jax.lax.psum(jnp.sum(partial_by_destination, axis=0), SOURCE_PUSH_MESH_AXIS)

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
    )(route_y_expert, route_dst, route_expert, route_expert_row, route_valid, route_weights)


def _source_push_semantic_forward_return_remote_source_gather_sharded_mgpu_kernel(
    mesh: Mesh,
    route_y_expert: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "S T H"]:
    route_y_expert = jax.sharding.reshard(route_y_expert, _destination_major_sharding(mesh, rank=4))
    source_sharding_3d = jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    route_dst = jax.sharding.reshard(route_dst, source_sharding_3d)
    route_expert = jax.sharding.reshard(route_expert, source_sharding_3d)
    route_expert_row = jax.sharding.reshard(route_expert_row, source_sharding_3d)
    route_valid = jax.sharding.reshard(route_valid, source_sharding_3d)
    route_weights = jax.sharding.reshard(route_weights, source_sharding_3d)

    def local_fn(
        route_y_expert_local: Float[Array, "1 E C H"],
        route_dst_local: Int[Array, "1 T K"],
        route_expert_local: Int[Array, "1 T K"],
        route_expert_row_local: Int[Array, "1 T K"],
        route_valid_local: Bool[Array, "1 T K"],
        route_weights_local: Float[Array, "1 T K"],
    ) -> Float[Array, "1 T H"]:
        source_y = _source_push_semantic_forward_return_remote_source_gather_pallas_call(
            route_y_expert_local,
            route_dst_local[0],
            route_expert_local[0],
            route_expert_row_local[0],
            route_valid_local[0],
            route_weights_local[0],
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=interpret,
        )
        return source_y[None, :, :]

    return shard_map(
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
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(route_y_expert, route_dst, route_expert, route_expert_row, route_valid, route_weights)


def _source_push_semantic_forward_return_remote_source_gather_pallas_call(
    route_y_expert: Float[Array, "DstLocal E C H"],
    route_dst: Int[Array, "T K"],
    route_expert: Int[Array, "T K"],
    route_expert_row: Int[Array, "T K"],
    route_valid: Bool[Array, "T K"],
    route_weights: Float[Array, "T K"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "T H"]:
    tokens_per_source, topk = route_dst.shape
    _local_dst_count, _experts_per_rank, _rows_per_expert_capacity, hidden_dim = route_y_expert.shape
    output_shape = jax.ShapeDtypeStruct((tokens_per_source, hidden_dim), jnp.float32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_remote_source_gather_kernel(
            hidden_block=hidden_block,
            row_block=row_block,
            topk=topk,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(tokens_per_source // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_forward_return_remote_source_gather_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            route_dst,
            route_expert,
            route_expert_row,
            route_valid,
            route_weights,
            output_shape,
        ),
    )(route_y_expert, route_dst, route_expert, route_expert_row, route_valid, route_weights)


def _source_push_semantic_forward_return_source_gather_by_destination_pallas_call(
    route_y_expert: Float[Array, "Dst E C H"],
    route_dst: Int[Array, "S T K"],
    route_expert: Int[Array, "S T K"],
    route_expert_row: Int[Array, "S T K"],
    route_valid: Bool[Array, "S T K"],
    route_weights: Float[Array, "S T K"],
    dst_offset: Int[Array, ""],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
) -> Float[Array, "Dst S T H"]:
    source_count, tokens_per_source, topk = route_dst.shape
    local_dst_count, _experts_per_rank, _rows_per_expert_capacity, hidden_dim = route_y_expert.shape
    output_shape = jax.ShapeDtypeStruct(
        (local_dst_count, source_count, tokens_per_source, hidden_dim),
        jnp.float32,
    )
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_forward_return_source_gather_by_destination_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            topk=topk,
            output_dtype=output_shape.dtype,
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            local_dst_count,
            source_count,
            tokens_per_source // row_block,
            hidden_dim // hidden_block,
        ),
        interpret=interpret,
        name="source_push_semantic_forward_return_source_gather_by_destination_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=_simple_io_cost_estimate(
            route_y_expert,
            route_dst,
            route_expert,
            route_expert_row,
            route_valid,
            route_weights,
            dst_offset,
            output_shape,
        ),
    )(route_y_expert, route_dst, route_expert, route_expert_row, route_valid, route_weights, dst_offset)


def _source_push_semantic_forward_return_source_gather_reference_from_reverse_jax(
    route_y_expert: Float[Array, "Dst E C H"],
    reverse_route: SourcePushSemanticReverseRoute,
    route_weights: Float[Array, "S T K"],
) -> Float[Array, "S T H"]:
    route_slot_sharding = _replicated_out_sharding_from_named_input(route_y_expert, rank=4)
    route_value_get = route_y_expert.at[
        reverse_route.route_dst,
        reverse_route.route_expert,
        reverse_route.route_expert_row,
    ]
    if route_slot_sharding is None:
        route_values = route_value_get.get(mode="clip")
    else:
        route_values = route_value_get.get(mode="clip", out_sharding=route_slot_sharding)
    route_values = route_values.astype(jnp.float32) * route_weights.astype(jnp.float32)[..., None]
    route_values = jnp.where(
        reverse_route.route_valid[..., None],
        route_values,
        jnp.zeros((), dtype=route_values.dtype),
    )
    return jnp.sum(route_values, axis=2)


def _replicated_out_sharding_from_named_input(value: Array, *, rank: int) -> jax.sharding.NamedSharding | None:
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return None
    return jax.sharding.NamedSharding(sharding.mesh, P(*(None for _ in range(rank))))


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


def _destination_major_sharding(mesh: Mesh, *, rank: int) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(rank - 1))))


def _constrain_destination_major_metadata(value: Array, mesh: Mesh) -> Array:
    return jax.sharding.reshard(value, _destination_major_sharding(mesh, rank=value.ndim))


def _source_push_semantic_reverse_route_weights_jax(plan: SourcePushSemanticPlan) -> Float[Array, "S T K"]:
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    token_ids = jnp.where(plan.valid_mask, jnp.maximum(plan.token_ids, 0), plan.tokens_per_source)
    route_slots = jnp.where(plan.valid_mask, jnp.maximum(plan.route_slots, 0), plan.topk)
    route_weights = jnp.zeros(
        (plan.assignment_ids.shape[0], plan.tokens_per_source, plan.topk),
        dtype=plan.route_weights.dtype,
    )
    return route_weights.at[source_index, token_ids, route_slots].set(plan.route_weights, mode="drop")


def _source_push_semantic_expert_row_indices_from_plan(
    plan: SourcePushSemanticPlan,
) -> tuple[Int[Array, "S Dst R"], Int[Array, "S Dst R"]]:
    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    rows = jnp.arange(plan.assignment_ids.shape[-1], dtype=jnp.int32)[None, None, :]
    src_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    pair_base = plan.pair_expert_base.at[src_index, dst_index, expert_ids].get()
    src_base = plan.src_base_by_expert.at[dst_index, src_index, expert_ids].get()
    expert_rows = src_base + rows - pair_base
    return expert_ids, jnp.maximum(expert_rows, 0).astype(jnp.int32)


def _make_source_push_semantic_forward_combine_source_gather_kernel(
    *,
    row_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        return_y_ref: Float[pl.Ref, "S DstOrd Q M H"],
        queue_dst_ord_ref: Int[pl.Ref, "S T K"],
        queue_entry_ref: Int[pl.Ref, "S T K"],
        queue_row_ref: Int[pl.Ref, "S T K"],
        route_weight_ref: Float[pl.Ref, "S T K"],
        route_valid_ref: Int[pl.Ref, "S T K"],
        y_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        src = pl.program_id(0)
        token_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        token_start = token_tile * row_block
        hidden_start = hidden_tile * hidden_block
        zero = jnp.zeros((hidden_block,), dtype=jnp.float32)

        for token_offset in range(row_block):
            token = token_start + token_offset
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for route_slot in range(topk):
                valid = route_valid_ref[src, token, route_slot] != 0
                dst_ord = queue_dst_ord_ref[src, token, route_slot]
                entry = queue_entry_ref[src, token, route_slot]
                row = queue_row_ref[src, token, route_slot]
                value = return_y_ref[
                    src,
                    dst_ord,
                    entry,
                    row,
                    pl.ds(hidden_start, hidden_block),
                ].astype(jnp.float32)
                weight = route_weight_ref[src, token, route_slot].astype(jnp.float32)
                acc += jnp.where(valid, value * weight, zero)

            y_ref[src, token, pl.ds(hidden_start, hidden_block)] = acc.astype(output_dtype)

    return kernel


def _make_source_push_semantic_forward_combine_source_gather_local_kernel(
    *,
    row_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        return_y_ref: Float[pl.Ref, "DstOrd Q M H"],
        queue_dst_ord_ref: Int[pl.Ref, "T K"],
        queue_entry_ref: Int[pl.Ref, "T K"],
        queue_row_ref: Int[pl.Ref, "T K"],
        route_weight_ref: Float[pl.Ref, "T K"],
        route_valid_ref: Int[pl.Ref, "T K"],
        y_ref: Float[pl.Ref, "T H"],
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
                valid = route_valid_ref[token, route_slot] != 0
                dst_ord = queue_dst_ord_ref[token, route_slot]
                entry = queue_entry_ref[token, route_slot]
                row = queue_row_ref[token, route_slot]
                value = return_y_ref[
                    dst_ord,
                    entry,
                    row,
                    pl.ds(hidden_start, hidden_block),
                ].astype(jnp.float32)
                weight = route_weight_ref[token, route_slot].astype(jnp.float32)
                acc += jnp.where(valid, value * weight, zero)

            y_ref[token, pl.ds(hidden_start, hidden_block)] = acc.astype(output_dtype)

    return kernel


def _make_source_push_semantic_forward_return_source_gather_kernel(
    *,
    row_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        route_dst_ref: Int[pl.Ref, "S T K"],
        route_expert_ref: Int[pl.Ref, "S T K"],
        route_expert_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Bool[pl.Ref, "S T K"],
        route_weights_ref: Float[pl.Ref, "S T K"],
        dst_offset_ref: Int[pl.Ref, ""],
        y_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        src = pl.program_id(0)
        token_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        token_start = token_tile * row_block
        hidden_start = hidden_tile * hidden_block
        dst_offset = dst_offset_ref[()]

        zero = jnp.zeros((hidden_block,), dtype=output_dtype)
        for token_offset in range(row_block):
            token = token_start + token_offset
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for route_slot in range(topk):
                global_dst = route_dst_ref[src, token, route_slot]
                local_dst = global_dst - dst_offset
                local_dst_valid = (local_dst >= 0) & (local_dst < route_y_expert_ref.shape[0])
                valid = route_valid_ref[src, token, route_slot] & local_dst_valid
                expert = route_expert_ref[src, token, route_slot]
                expert_row = route_expert_row_ref[src, token, route_slot]
                if route_y_expert_ref.shape[0] == 1:
                    route_tile = route_y_expert_ref[
                        0,
                        pl.ds(expert, 1),
                        pl.ds(expert_row, 1),
                        pl.ds(hidden_start, hidden_block),
                    ][0, 0, :].astype(jnp.float32)
                else:
                    safe_dst = jnp.where(local_dst_valid, local_dst, 0)
                    route_tile = route_y_expert_ref[
                        pl.ds(safe_dst, 1),
                        pl.ds(expert, 1),
                        pl.ds(expert_row, 1),
                        pl.ds(hidden_start, hidden_block),
                    ][0, 0, 0, :].astype(jnp.float32)
                weight = route_weights_ref[src, token, route_slot].astype(jnp.float32)
                acc += jnp.where(valid, route_tile * weight, zero)

            y_ref[
                pl.ds(src, 1),
                pl.ds(token, 1),
                pl.ds(hidden_start, hidden_block),
            ] = acc.astype(
                output_dtype
            )[None, None, :]

    return kernel


def _make_source_push_semantic_forward_return_source_gather_by_destination_kernel(
    *,
    row_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        route_dst_ref: Int[pl.Ref, "S T K"],
        route_expert_ref: Int[pl.Ref, "S T K"],
        route_expert_row_ref: Int[pl.Ref, "S T K"],
        route_valid_ref: Bool[pl.Ref, "S T K"],
        route_weights_ref: Float[pl.Ref, "S T K"],
        dst_offset_ref: Int[pl.Ref, ""],
        partial_y_ref: Float[pl.Ref, "Dst S T H"],
    ) -> None:
        local_dst = pl.program_id(0)
        src = pl.program_id(1)
        token_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        global_dst = dst_offset_ref[()] + local_dst
        token_start = token_tile * row_block
        hidden_start = hidden_tile * hidden_block

        zero = jnp.zeros((hidden_block,), dtype=output_dtype)
        for token_offset in range(row_block):
            token = token_start + token_offset
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for route_slot in range(topk):
                valid = route_valid_ref[src, token, route_slot] & (route_dst_ref[src, token, route_slot] == global_dst)
                expert = route_expert_ref[src, token, route_slot]
                expert_row = route_expert_row_ref[src, token, route_slot]
                route_tile = route_y_expert_ref[
                    pl.ds(local_dst, 1),
                    pl.ds(expert, 1),
                    pl.ds(expert_row, 1),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, 0, :].astype(jnp.float32)
                weight = route_weights_ref[src, token, route_slot].astype(jnp.float32)
                acc += jnp.where(valid, route_tile * weight, zero)

            partial_y_ref[
                pl.ds(local_dst, 1),
                pl.ds(src, 1),
                pl.ds(token, 1),
                pl.ds(hidden_start, hidden_block),
            ] = acc.astype(output_dtype)[None, None, None, :]

    return kernel


def _make_source_push_semantic_forward_return_remote_source_gather_kernel(
    *,
    row_block: int,
    hidden_block: int,
    topk: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "DstLocal E C H"],
        route_dst_ref: Int[pl.Ref, "T K"],
        route_expert_ref: Int[pl.Ref, "T K"],
        route_expert_row_ref: Int[pl.Ref, "T K"],
        route_valid_ref: Bool[pl.Ref, "T K"],
        route_weights_ref: Float[pl.Ref, "T K"],
        y_ref: Float[pl.Ref, "T H"],
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
                remote_route_y_expert_ref = mgpu.remote_ref(
                    route_y_expert_ref,
                    safe_dst,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                expert = route_expert_ref[token, route_slot]
                expert_row = route_expert_row_ref[token, route_slot]
                route_tile = remote_route_y_expert_ref[
                    0,
                    pl.ds(expert, 1),
                    pl.ds(expert_row, 1),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, :].astype(jnp.float32)
                weight = route_weights_ref[token, route_slot].astype(jnp.float32)
                acc += jnp.where(valid, route_tile * weight, zero)

            y_ref[token, pl.ds(hidden_start, hidden_block)] = acc.astype(output_dtype)

    return kernel


def _make_source_push_semantic_forward_return_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_slots_ref: Int[pl.Ref, "S Dst R"],
        route_weights_ref: Float[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _route_by_slot_init_ref: Float[pl.Ref, "S T K H"],
        route_by_slot_ref: Float[pl.Ref, "S T K H"],
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
            route_tile = route_y_expert_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]
            weight = route_weights_ref[src, global_dst, pair_row].astype(output_dtype)
            weighted = route_tile.astype(output_dtype) * weight

            @pl.when(valid)
            def _store_valid_route() -> None:
                route_by_slot_ref[
                    pl.ds(src, 1),
                    pl.ds(safe_token, 1),
                    pl.ds(safe_slot, 1),
                    pl.ds(hidden_start, hidden_block),
                ] = weighted[None, None, None, :]

    return kernel


def _make_source_push_semantic_forward_return_sum_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_weights_ref: Float[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _y_init_ref: Float[pl.Ref, "S T H"],
        y_ref: Float[pl.Ref, "S T H"],
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
            route_tile = route_y_expert_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]
            weight = route_weights_ref[src, global_dst, pair_row].astype(output_dtype)
            weighted = route_tile.astype(output_dtype) * weight

            @pl.when(valid)
            def _add_valid_route() -> None:
                mgpu.atomic_add(
                    y_ref.at[src, safe_token, pl.ds(hidden_start, hidden_block)],
                    weighted,
                )

    return kernel


def _make_source_push_semantic_forward_return_sum_lookup_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        source_lookup_ref: Int[pl.Ref, "Dst E C"],
        token_lookup_ref: Int[pl.Ref, "Dst E C"],
        weight_lookup_ref: Float[pl.Ref, "Dst E C"],
        valid_lookup_ref: Int[pl.Ref, "Dst E C"],
        _y_init_ref: Float[pl.Ref, "S T H"],
        y_ref: Float[pl.Ref, "S T H"],
    ) -> None:
        local_dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        expert_row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        for row_offset in range(row_block):
            expert_row = expert_row_start + row_offset
            valid = valid_lookup_ref[local_dst, expert, expert_row] != 0
            src = source_lookup_ref[local_dst, expert, expert_row]
            safe_token = token_lookup_ref[local_dst, expert, expert_row]
            route_tile = route_y_expert_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]
            weight = weight_lookup_ref[local_dst, expert, expert_row].astype(output_dtype)
            weighted = route_tile.astype(output_dtype) * weight

            @pl.when(valid)
            def _add_valid_route() -> None:
                mgpu.atomic_add(
                    y_ref.at[src, safe_token, pl.ds(hidden_start, hidden_block)],
                    weighted,
                )

    return kernel


def _make_source_push_semantic_forward_return_copy_only_expert_major_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        route_y_expert_ref: Float[pl.Ref, "Dst E C H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        route_weights_ref: Float[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        dst_offset_ref: Int[pl.Ref, ""],
        _weighted_route_init_ref: Float[pl.Ref, "Dst E C H"],
        weighted_route_ref: Float[pl.Ref, "Dst E C H"],
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
            route_tile = route_y_expert_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, 0, :]
            weight = route_weights_ref[src, global_dst, pair_row].astype(output_dtype)
            weighted = route_tile.astype(output_dtype) * weight
            weighted_route_ref[
                pl.ds(local_dst, 1),
                pl.ds(expert, 1),
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(valid & token_is_valid, weighted, zero_tile)[None, None, None, :]

    return kernel


def _source_push_semantic_w2_pallas_call(
    h_pair: Float[Array, "S Dst R I"],
    w_down: Float[Array, "Dst E I H"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    interpret: bool,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "S Dst R H"]:
    source_count, dst_count, rows_per_pair, intermediate_dim = h_pair.shape
    hidden_dim = w_down.shape[-1]
    output_shape = jax.ShapeDtypeStruct((source_count, dst_count, rows_per_pair, hidden_dim), jnp.float32)
    valid_mask_i32 = valid_mask.astype(jnp.int32)
    input_specs, output_specs = _source_push_semantic_w2_block_specs()
    cost_estimate = _source_push_semantic_w2_cost_estimate(
        h_pair,
        w_down,
        xcounts,
        pair_expert_base,
        valid_mask_i32,
        output_shape,
    )
    return pl.pallas_call(
        _make_source_push_semantic_w2_kernel(
            row_block=row_block,
            intermediate_block=intermediate_block,
            hidden_block=hidden_block,
            intermediate_dim=intermediate_dim,
            experts_per_rank=w_down.shape[1],
        ),
        in_specs=input_specs,
        out_specs=output_specs,
        out_shape=output_shape,
        grid=(source_count, dst_count, rows_per_pair // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_w2_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
        cost_estimate=cost_estimate,
    )(h_pair, w_down, xcounts, pair_expert_base, valid_mask_i32)


def _source_push_semantic_w2_expert_major_pallas_call(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_w2_expert_major_kernel(
        dst_count=h_expert.shape[0],
        experts_per_rank=h_expert.shape[1],
        rows_per_expert_capacity=h_expert.shape[2],
        intermediate_dim=h_expert.shape[-1],
        hidden_dim=w_down.shape[-1],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )
    return kernel(h_expert, w_down, valid)


def _source_push_semantic_w2_expert_major_unmasked_pallas_call(
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_w2_expert_major_unmasked_kernel(
        dst_count=h_expert.shape[0],
        experts_per_rank=h_expert.shape[1],
        rows_per_expert_capacity=h_expert.shape[2],
        intermediate_dim=h_expert.shape[-1],
        hidden_dim=w_down.shape[-1],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )
    return kernel(h_expert, w_down)


def _source_push_semantic_w2_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_w2_expert_major_kernel(
        dst_count=1,
        experts_per_rank=h_expert.shape[1],
        rows_per_expert_capacity=h_expert.shape[2],
        intermediate_dim=h_expert.shape[-1],
        hidden_dim=w_down.shape[-1],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )

    def local_fn(
        h_local: Float[Array, "1 E C I"],
        w_local: Float[Array, "1 E I H"],
        valid_local: Float[Array, "1 E C"],
    ) -> Float[Array, "1 E C H"]:
        return kernel(h_local, w_local, valid_local)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(h_expert, w_down, valid)


def _source_push_semantic_w2_expert_major_unmasked_sharded_mgpu_kernel(
    mesh: Mesh,
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_w2_expert_major_unmasked_kernel(
        dst_count=1,
        experts_per_rank=h_expert.shape[1],
        rows_per_expert_capacity=h_expert.shape[2],
        intermediate_dim=h_expert.shape[-1],
        hidden_dim=w_down.shape[-1],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )

    def local_fn(
        h_local: Float[Array, "1 E C I"],
        w_local: Float[Array, "1 E I H"],
    ) -> Float[Array, "1 E C H"]:
        return kernel(h_local, w_local)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(h_expert, w_down)


def _source_push_semantic_w2_direct_return_to_source_sharded_mgpu_kernel(
    mesh: Mesh,
    h_expert: Float[Array, "Dst E C I"],
    w_down: Float[Array, "Dst E I H"],
    recv_local_expert: Int[Array, "Dst SrcOrd Q"],
    recv_expert_row_start: Int[Array, "Dst SrcOrd Q"],
    recv_valid_rows: Int[Array, "Dst SrcOrd Q"],
    *,
    source_count: int,
    entries_per_dst: int,
    return_row_block: int,
    output_dtype: jnp.dtype,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "S DstOrd Q M H"]:
    kernel = _make_source_push_semantic_w2_direct_return_to_source_kernel(
        source_count=source_count,
        intermediate_dim=h_expert.shape[-1],
        hidden_dim=w_down.shape[-1],
        entries_per_dst=entries_per_dst,
        return_row_block=return_row_block,
        output_dtype=output_dtype,
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )

    def local_fn(
        h_local: Float[Array, "1 E C I"],
        w_local: Float[Array, "1 E I H"],
        recv_local_expert_local: Int[Array, "1 SrcOrd Q"],
        recv_expert_row_start_local: Int[Array, "1 SrcOrd Q"],
        recv_valid_rows_local: Int[Array, "1 SrcOrd Q"],
    ) -> Float[Array, "1 DstOrd Q M H"]:
        return_y = kernel(
            h_local,
            w_local,
            recv_local_expert_local,
            recv_expert_row_start_local,
            recv_valid_rows_local,
        )
        return return_y[None, ...]

    return_y = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(h_expert, w_down, recv_local_expert, recv_expert_row_start, recv_valid_rows)
    return _sharded_forward_return_remote_write_completion_barrier(mesh)(return_y)


def _source_push_semantic_w2_block_specs() -> tuple[tuple[pl.BlockSpec, ...], pl.BlockSpec]:
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec), gmem_spec


def _make_source_push_semantic_w2_direct_return_to_source_kernel(
    *,
    source_count: int,
    intermediate_dim: int,
    hidden_dim: int,
    entries_per_dst: int,
    return_row_block: int,
    output_dtype: jnp.dtype,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
):
    if row_block != return_row_block:
        raise ValueError(
            "direct-return W2 block-store path requires w2 row_block to match return row_block; "
            f"got row_block={row_block}, return_row_block={return_row_block}"
        )
    intermediate_tiles = intermediate_dim // intermediate_block
    src_offsets = tuple(range(source_count))

    def body(
        h_expert_ref: Float[pl.Ref, "Dst E C I"],
        w_down_ref: Float[pl.Ref, "Dst E I H"],
        recv_local_expert_ref: Int[pl.Ref, "Dst SrcOrd Q"],
        recv_expert_row_start_ref: Int[pl.Ref, "Dst SrcOrd Q"],
        recv_valid_rows_ref: Int[pl.Ref, "Dst SrcOrd Q"],
        return_y_ref: Float[pl.Ref, "DstOrd Q M H"],
    ) -> None:
        rank = jax.lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block

        def _compute_to_source(static_src_ordinal: int) -> None:
            valid_rows = recv_valid_rows_ref[0, static_src_ordinal, entry]

            @pl.when(valid_rows > 0)
            def _compute_return_block() -> None:
                expert = recv_local_expert_ref[0, static_src_ordinal, entry]
                expert_row_start = recv_expert_row_start_ref[0, static_src_ordinal, entry]

                def acc_scope(acc_ref) -> jax.Array:
                    def smem_scope(h_smem, w_smem, ready_barrier) -> None:
                        @pl.loop(0, intermediate_tiles)
                        def _intermediate_loop(intermediate_tile) -> None:
                            intermediate_start = intermediate_tile * intermediate_block
                            mgpu.copy_gmem_to_smem(
                                h_expert_ref.at[
                                    0,
                                    expert,
                                    pl.ds(expert_row_start, row_block),
                                    pl.ds(intermediate_start, intermediate_block),
                                ],
                                h_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_down_ref.at[
                                    0,
                                    expert,
                                    pl.ds(intermediate_start, intermediate_block),
                                    pl.ds(hidden_start, hidden_block),
                                ],
                                w_smem,
                                ready_barrier,
                            )
                            mgpu.barrier_wait(ready_barrier)
                            mgpu.commit_smem()
                            mgpu.wgmma(acc_ref, h_smem, w_smem)
                            mgpu.wgmma_wait(0)

                    pl.run_scoped(
                        smem_scope,
                        h_smem=_semantic_w2_wgmma_smem((row_block, intermediate_block), h_expert_ref.dtype),
                        w_smem=_semantic_w2_wgmma_smem((intermediate_block, hidden_block), w_down_ref.dtype),
                        ready_barrier=mgpu.Barrier(num_arrivals=2),
                    )
                    return acc_ref[...].astype(output_dtype)

                output = pl.run_scoped(
                    acc_scope,
                    acc_ref=mgpu.ACC((row_block, hidden_block)),
                )

                def store_scope(output_smem) -> None:
                    output_smem[:, :] = output
                    mgpu.commit_smem()
                    src = (rank + static_src_ordinal) % source_count
                    dst_ordinal = (-static_src_ordinal) % source_count
                    if static_src_ordinal == 0:
                        destination_ref = return_y_ref
                    else:
                        destination_ref = mgpu.remote_ref(
                            return_y_ref,
                            src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                    mgpu.copy_smem_to_gmem(
                        output_smem,
                        destination_ref.at[
                            dst_ordinal,
                            entry,
                            pl.ds(0, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                    )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    store_scope,
                    output_smem=mgpu.SMEM((row_block, hidden_block), dtype=output_dtype),
                )

        def _branch(static_src_ordinal: int):
            def _compute_branch(_) -> None:
                _compute_to_source(static_src_ordinal)

            return _compute_branch

        jax.lax.switch(src_ordinal, tuple(_branch(src) for src in src_offsets), None)

    out_shape = jax.ShapeDtypeStruct(
        (source_count, entries_per_dst, return_row_block, hidden_dim),
        output_dtype,
    )
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(source_count, entries_per_dst, hidden_dim // hidden_block),
        grid_names=("src_ordinal", "entry", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _make_source_push_semantic_w2_expert_major_kernel(
    *,
    dst_count: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    intermediate_dim: int,
    hidden_dim: int,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
):
    intermediate_tiles = intermediate_dim // intermediate_block

    def body(
        h_expert_ref: Float[pl.Ref, "Dst E C I"],
        w_down_ref: Float[pl.Ref, "Dst E I H"],
        valid_ref: Float[pl.Ref, "Dst E C"],
        route_y_ref: Float[pl.Ref, "Dst E C H"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(h_smem, w_smem, valid_smem, ready_barrier, valid_ready_barrier) -> None:
                zero_row_slice = pl.ds(0, row_block)
                mgpu.copy_gmem_to_smem(
                    valid_ref.at[
                        dst,
                        expert,
                        pl.ds(row_start, row_block),
                    ],
                    valid_smem,
                    valid_ready_barrier,
                )
                mgpu.barrier_wait(valid_ready_barrier)
                valid_vec = mgpu.load(
                    valid_smem,
                    (zero_row_slice,),
                    layout=mgpu.Layout.WGMMA.reduce(1),
                ).astype(jnp.float32)
                valid_f = jax.lax.broadcast_in_dim(valid_vec, (row_block, intermediate_block), (0,))

                @pl.loop(0, intermediate_tiles)
                def _intermediate_loop(intermediate_tile) -> None:
                    intermediate_start = intermediate_tile * intermediate_block
                    mgpu.copy_gmem_to_smem(
                        h_expert_ref.at[
                            dst,
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(intermediate_start, intermediate_block),
                        ],
                        h_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w_down_ref.at[
                            dst,
                            expert,
                            pl.ds(intermediate_start, intermediate_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        w_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    h_smem[:, :] = (h_smem[:, :].astype(jnp.float32) * valid_f).astype(h_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, h_smem, w_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                h_smem=_semantic_w2_wgmma_smem((row_block, intermediate_block), h_expert_ref.dtype),
                w_smem=_semantic_w2_wgmma_smem((intermediate_block, hidden_block), w_down_ref.dtype),
                valid_smem=mgpu.SMEM((row_block,), dtype=valid_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
                valid_ready_barrier=mgpu.Barrier(num_arrivals=1),
            )
            return acc_ref[...]

        output = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((row_block, hidden_block)),
        )
        route_y_ref[
            dst,
            expert,
            pl.ds(row_start, row_block),
            pl.ds(hidden_start, hidden_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct(
        (dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim),
        jnp.float32,
    )
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(
            dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        grid_names=("destination", "expert", "row_tile", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _make_source_push_semantic_w2_expert_major_unmasked_kernel(
    *,
    dst_count: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    intermediate_dim: int,
    hidden_dim: int,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
):
    intermediate_tiles = intermediate_dim // intermediate_block

    def body(
        h_expert_ref: Float[pl.Ref, "Dst E C I"],
        w_down_ref: Float[pl.Ref, "Dst E I H"],
        route_y_ref: Float[pl.Ref, "Dst E C H"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(h_smem, w_smem, ready_barrier) -> None:
                @pl.loop(0, intermediate_tiles)
                def _intermediate_loop(intermediate_tile) -> None:
                    intermediate_start = intermediate_tile * intermediate_block
                    mgpu.copy_gmem_to_smem(
                        h_expert_ref.at[
                            dst,
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(intermediate_start, intermediate_block),
                        ],
                        h_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w_down_ref.at[
                            dst,
                            expert,
                            pl.ds(intermediate_start, intermediate_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        w_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, h_smem, w_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                h_smem=_semantic_w2_wgmma_smem((row_block, intermediate_block), h_expert_ref.dtype),
                w_smem=_semantic_w2_wgmma_smem((intermediate_block, hidden_block), w_down_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((row_block, hidden_block)),
        )
        route_y_ref[
            dst,
            expert,
            pl.ds(row_start, row_block),
            pl.ds(hidden_start, hidden_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct(
        (dst_count, experts_per_rank, rows_per_expert_capacity, hidden_dim),
        jnp.float32,
    )
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(
            dst_count,
            experts_per_rank,
            rows_per_expert_capacity // row_block,
            hidden_dim // hidden_block,
        ),
        grid_names=("destination", "expert", "row_tile", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _make_source_push_semantic_w2_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    intermediate_dim: int,
    experts_per_rank: int,
):
    def kernel(
        h_pair_ref: Float[pl.Ref, "S Dst R I"],
        w_down_ref: Float[pl.Ref, "Dst E I H"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        route_y_ref: Float[pl.Ref, "S Dst R H"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        for row_offset in range(row_block):
            row = row_start + row_offset
            expert = jnp.asarray(0, dtype=jnp.int32)
            in_expert_interval = jnp.asarray(False)
            for candidate_expert in range(experts_per_rank):
                expert_base = pair_expert_base_ref[src, dst, candidate_expert]
                expert_count = xcounts_ref[src, dst, candidate_expert]
                row_matches = (row >= expert_base) & (row < expert_base + expert_count)
                expert = jnp.where(row_matches, candidate_expert, expert)
                in_expert_interval = in_expert_interval | row_matches

            row_is_valid = (valid_mask_ref[src, dst, row] != 0) & in_expert_interval
            acc = jnp.zeros((1, hidden_block), dtype=jnp.float32)
            for intermediate_start in range(0, intermediate_dim, intermediate_block):
                h_tile = h_pair_ref[
                    pl.ds(src, 1),
                    pl.ds(dst, 1),
                    pl.ds(row, 1),
                    pl.ds(intermediate_start, intermediate_block),
                ][0, 0, :, :].astype(jnp.float32)
                w_tile = w_down_ref[
                    pl.ds(dst, 1),
                    pl.ds(expert, 1),
                    pl.ds(intermediate_start, intermediate_block),
                    pl.ds(hidden_start, hidden_block),
                ][0, 0, :, :].astype(jnp.float32)
                acc += pl.dot(h_tile, w_tile)

            route_y_ref[
                pl.ds(src, 1),
                pl.ds(dst, 1),
                pl.ds(row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(
                row_is_valid, acc, jnp.zeros_like(acc)
            )[None, None, :, :]

    return kernel


def _source_push_semantic_w2_reference(
    h_pair: Array,
    w_down: Array,
    xcounts: Array,
    valid_mask: Array,
) -> Array:
    rows = jnp.arange(h_pair.shape[2], dtype=jnp.int32)
    pair_ends = jnp.cumsum(xcounts, axis=2, dtype=jnp.int32)

    def pair_expert_ids(ends):
        expert = jnp.searchsorted(ends, rows, side="right").astype(jnp.int32)
        return jnp.minimum(expert, xcounts.shape[-1] - 1)

    expert_ids = jax.vmap(jax.vmap(pair_expert_ids, in_axes=0), in_axes=0)(pair_ends)
    dst_index = jnp.arange(h_pair.shape[1], dtype=jnp.int32)[None, :, None]
    w_pair = w_down.at[dst_index, expert_ids].get().astype(jnp.float32)
    route_y = jnp.einsum(
        "sdri,sdrih->sdrh",
        h_pair.astype(jnp.float32),
        w_pair,
        preferred_element_type=jnp.float32,
    )
    return jnp.where(valid_mask[..., None] != 0, route_y, jnp.zeros((), dtype=route_y.dtype))


def _source_push_semantic_w2_expert_major_unmasked_reference_jax(
    h_expert: Array,
    w_down: Array,
) -> Array:
    return jnp.einsum(
        "deci,deih->dech",
        h_expert.astype(jnp.float32),
        w_down.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )


def _source_push_semantic_w2_cost_estimate(
    h_pair: Array,
    w_down: Array,
    xcounts: Array,
    pair_expert_base: Array,
    valid_mask_i32: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(h_pair.shape, h_pair.dtype),
        jax.ShapeDtypeStruct(w_down.shape, w_down.dtype),
        jax.ShapeDtypeStruct(xcounts.shape, xcounts.dtype),
        jax.ShapeDtypeStruct(pair_expert_base.shape, pair_expert_base.dtype),
        jax.ShapeDtypeStruct(valid_mask_i32.shape, valid_mask_i32.dtype),
    )

    def reference(h_pair_spec, w_down_spec, xcounts_spec, _pair_expert_base_spec, valid_mask_spec):
        return _source_push_semantic_w2_reference(
            h_pair_spec,
            w_down_spec,
            xcounts_spec,
            valid_mask_spec,
        )

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


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
    expected_shape = (
        plan.assignment_ids.shape[1],
        plan.assignment_ids.shape[0],
        plan.xcounts.shape[2],
    )
    if source_row_base_by_expert.shape != expected_shape:
        raise ValueError(
            f"source_row_base_by_expert shape {source_row_base_by_expert.shape} must match {expected_shape}"
        )
    if source_row_base_by_expert.dtype != jnp.int32:
        raise ValueError(f"source_row_base_by_expert must have dtype int32, got {source_row_base_by_expert.dtype}")
    return source_row_base_by_expert


def _validate_direct_return_queue(
    plan: SourcePushSemanticPlan,
    queue: SourcePushSemanticQueueMetadata,
    return_row_block: int,
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
    if queue.return_row_block != return_row_block:
        raise ValueError(f"queue return_row_block {queue.return_row_block} must match requested {return_row_block}")


def _validate_semantic_forward_return_expert_major_request(
    route_y_expert: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes,
) -> None:
    if route_y_expert.ndim != 4:
        raise ValueError(
            f"route_y_expert must have shape [destination, expert, row, hidden], got {route_y_expert.shape}"
        )
    if route_y_expert.shape[0] != plan.assignment_ids.shape[1]:
        raise ValueError(
            f"route_y_expert destination dim {route_y_expert.shape[0]} must match plan destination dim "
            f"{plan.assignment_ids.shape[1]}"
        )
    if route_y_expert.shape[1] != plan.xcounts.shape[2]:
        raise ValueError(
            f"route_y_expert expert dim {route_y_expert.shape[1]} must match plan expert dim {plan.xcounts.shape[2]}"
        )
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if route_y_expert.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"route_y_expert row capacity {route_y_expert.shape[2]} must be divisible by "
            f"row_block={block_sizes.row_block}"
        )
    if route_y_expert.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"route_y_expert hidden dim {route_y_expert.shape[3]} must be divisible by "
            f"hidden_block={block_sizes.hidden_block}"
        )


def _validate_semantic_forward_return_source_gather_request(
    route_y_expert: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes,
) -> None:
    _validate_semantic_forward_return_expert_major_request(
        route_y_expert,
        plan,
        SourcePushSemanticForwardReturnPallasBlockSizes(row_block=1, hidden_block=block_sizes.hidden_block),
    )
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if plan.tokens_per_source % block_sizes.row_block:
        raise ValueError(
            f"tokens_per_source {plan.tokens_per_source} must be divisible by row_block={block_sizes.row_block}"
        )


def _validate_semantic_forward_combine_source_gather_request(
    return_y: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes,
    *,
    entries_per_dst: int,
) -> None:
    if return_y.ndim != 5:
        raise ValueError(f"return_y must have shape [source, dst_ord, entry, row, hidden], got {return_y.shape}")
    source_count, destination_count = plan.assignment_ids.shape[:2]
    if return_y.shape[0] != source_count:
        raise ValueError(f"return_y source dim {return_y.shape[0]} must match plan source dim {source_count}")
    if return_y.shape[1] != destination_count:
        raise ValueError(
            f"return_y destination ordinal dim {return_y.shape[1]} must match destination dim {destination_count}"
        )
    if return_y.shape[3] != block_sizes.row_block:
        raise ValueError(f"return_y row dim {return_y.shape[3]} must match row_block={block_sizes.row_block}")
    if return_y.shape[2] != entries_per_dst:
        raise ValueError(f"return_y entry dim {return_y.shape[2]} must match expected {entries_per_dst}")
    if plan.tokens_per_source % block_sizes.row_block:
        raise ValueError(
            f"tokens_per_source {plan.tokens_per_source} must be divisible by row_block={block_sizes.row_block}"
        )
    if return_y.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            f"return_y hidden dim {return_y.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )


def _validate_semantic_w2_pallas_request(
    h_pair: Array,
    w_down: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticW2PallasBlockSizes,
) -> None:
    if h_pair.ndim != 4:
        raise ValueError(f"h_pair must have shape [source, destination, row, intermediate], got {h_pair.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must have shape [destination, expert, intermediate, hidden], got {w_down.shape}")
    if plan.valid_mask.shape != h_pair.shape[:3]:
        raise ValueError(
            f"plan valid_mask shape {plan.valid_mask.shape} must match h_pair route shape {h_pair.shape[:3]}"
        )
    if plan.xcounts.shape[:2] != h_pair.shape[:2]:
        raise ValueError(
            f"plan xcounts source/destination shape {plan.xcounts.shape[:2]} must match {h_pair.shape[:2]}"
        )
    if plan.pair_expert_base.shape != plan.xcounts.shape:
        raise ValueError(
            f"plan pair_expert_base shape {plan.pair_expert_base.shape} must match xcounts shape {plan.xcounts.shape}"
        )
    if w_down.shape[0] != h_pair.shape[1]:
        raise ValueError(
            f"w_down destination dim {w_down.shape[0]} must match h_pair destination dim {h_pair.shape[1]}"
        )
    if w_down.shape[1] != plan.xcounts.shape[2]:
        raise ValueError(f"w_down expert dim {w_down.shape[1]} must match plan expert dim {plan.xcounts.shape[2]}")
    if w_down.shape[2] != h_pair.shape[3]:
        raise ValueError(
            f"w_down intermediate dim {w_down.shape[2]} must match h_pair intermediate dim {h_pair.shape[3]}"
        )
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if h_pair.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"h_pair row capacity {h_pair.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if h_pair.shape[3] % block_sizes.intermediate_block:
        raise ValueError(
            f"h_pair intermediate dim {h_pair.shape[3]} must be divisible by intermediate_block="
            f"{block_sizes.intermediate_block}"
        )
    if w_down.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"w_down hidden dim {w_down.shape[3]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )


def _validate_semantic_w2_expert_major_request(
    h_expert: Array,
    w_down: Array,
    valid: Array,
) -> None:
    _validate_semantic_w2_expert_major_h_w_request(h_expert, w_down)
    if valid.shape != h_expert.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match h_expert row shape {h_expert.shape[:3]}")


def _validate_semantic_w2_expert_major_h_w_request(
    h_expert: Array,
    w_down: Array,
) -> None:
    if h_expert.ndim != 4:
        raise ValueError(f"h_expert must have shape [destination, expert, row, intermediate], got {h_expert.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must have shape [destination, expert, intermediate, hidden], got {w_down.shape}")
    if w_down.shape[:3] != (h_expert.shape[0], h_expert.shape[1], h_expert.shape[3]):
        raise ValueError(
            "w_down destination/expert/intermediate dims "
            f"{w_down.shape[:3]} must match {(h_expert.shape[0], h_expert.shape[1], h_expert.shape[3])}"
        )


def _validate_semantic_w2_expert_major_pallas_request(
    h_expert: Array,
    w_down: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes,
) -> None:
    _validate_semantic_w2_expert_major_request(h_expert, w_down, valid)
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.row_block % 64:
        raise ValueError(f"row_block must be a multiple of 64 for WGMMA, got {block_sizes.row_block}")
    if h_expert.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"h_expert capacity {h_expert.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if h_expert.shape[3] % block_sizes.intermediate_block:
        raise ValueError(
            f"h_expert intermediate dim {h_expert.shape[3]} must be divisible by "
            f"intermediate_block={block_sizes.intermediate_block}"
        )
    if w_down.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"w_down hidden dim {w_down.shape[3]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )
    _semantic_w2_wgmma_transforms((block_sizes.row_block, block_sizes.intermediate_block), h_expert.dtype)
    _semantic_w2_wgmma_transforms((block_sizes.intermediate_block, block_sizes.hidden_block), w_down.dtype)


def _validate_semantic_w2_expert_major_unmasked_pallas_request(
    h_expert: Array,
    w_down: Array,
    block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes,
) -> None:
    dummy_valid = jax.ShapeDtypeStruct(h_expert.shape[:3], jnp.bool_)
    _validate_semantic_w2_expert_major_pallas_request(h_expert, w_down, dummy_valid, block_sizes)


def _validate_semantic_w2_direct_return_request(
    h_expert: Array,
    w_down: Array,
    valid: Array,
    plan: SourcePushSemanticPlan,
    w2_block_sizes: SourcePushSemanticW2ExpertMajorPallasBlockSizes,
    return_block_sizes: SourcePushSemanticForwardReturnPallasBlockSizes,
) -> None:
    _validate_semantic_w2_expert_major_pallas_request(h_expert, w_down, valid, w2_block_sizes)
    if h_expert.shape[:3] != (plan.assignment_ids.shape[1], plan.xcounts.shape[2], h_expert.shape[2]):
        raise ValueError(
            "h_expert destination/expert dims must match plan destination/expert dims; "
            f"got {h_expert.shape[:2]} vs {(plan.assignment_ids.shape[1], plan.xcounts.shape[2])}"
        )
    if return_block_sizes.row_block <= 0:
        raise ValueError(f"return row_block must be positive, got {return_block_sizes.row_block}")
    if return_block_sizes.hidden_block <= 0:
        raise ValueError(f"return hidden_block must be positive, got {return_block_sizes.hidden_block}")
    if w2_block_sizes.row_block != return_block_sizes.row_block:
        raise ValueError(
            "direct-return W2 requires matching compute and return row blocks; "
            f"got {w2_block_sizes.row_block} and {return_block_sizes.row_block}"
        )


def _semantic_w2_wgmma_smem(shape: tuple[int, int], dtype):
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=_semantic_w2_wgmma_transforms(shape, dtype),
    )


def _semantic_w2_wgmma_transforms(shape: tuple[int, int], dtype):
    swizzle_elems = SEMANTIC_W2_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % SEMANTIC_W2_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "Semantic W2 WGMMA SMEM operands must be divisible by "
            f"tile=({SEMANTIC_W2_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )
    return (
        mgpu.TilingTransform((SEMANTIC_W2_WGMMA_TILE_M, swizzle_elems)),
        mgpu.SwizzleTransform(SEMANTIC_W2_WGMMA_SWIZZLE_BYTES),
    )
