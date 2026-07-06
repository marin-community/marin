# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Route source-owned MLP output gradients into flat source-push H rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_forward import SourcePushForwardHostInputs
from levanter.grug._moe.source_push_inbox import PushInboxConfig
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SOURCE_PUSH_META_FIELDS,
    SOURCE_PUSH_META_LOCAL_EXPERT,
    SOURCE_PUSH_META_LOCAL_ROW_START,
    SOURCE_PUSH_META_VALID_ROWS,
    SourcePushPlan,
    _source_push_out_sharding,
    _with_source_push_sharding,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU = "source_push_pallas_mgpu"
SourcePushDyRouteImplementation: TypeAlias = Literal["reference", "pallas_mgpu", "source_push_pallas_mgpu"]
MIN_SOURCE_PUSH_DY_ROUTE_GPU_ROW_BLOCK = 128
DEFAULT_SOURCE_PUSH_DY_ROUTE_ROW_BLOCK = MIN_SOURCE_PUSH_DY_ROUTE_GPU_ROW_BLOCK
DEFAULT_SOURCE_PUSH_DY_ROUTE_HIDDEN_BLOCK = 128


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True, slots=True)
class SourcePushDyRoutePallasBlockSizes:
    """Tile sizes for the flat-H dy route Pallas kernel."""

    row_block: int = DEFAULT_SOURCE_PUSH_DY_ROUTE_ROW_BLOCK
    hidden_block: int = DEFAULT_SOURCE_PUSH_DY_ROUTE_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushDyRoutePallasBlockSizes":
        return cls()


class _SourcePushDyRouteInverseIndices(NamedTuple):
    """Destination flat-row inverse map to source-owned dy rows."""

    src: Int[Array, "Dst rows"]
    token: Int[Array, "Dst rows"]
    valid: Bool[Array, "Dst rows"]


def _source_push_backward_dy_to_expert_major(
    dy: Float[Array, "S T D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    implementation: SourcePushDyRouteImplementation = SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDyRoutePallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C D"]:
    """Route source-owned ``dy`` directly into compact destination expert-major rows."""

    if implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE:
        return _source_push_backward_dy_to_expert_major_reference(
            dy,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
        )
    if implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU:
        return _source_push_backward_dy_to_expert_major_pallas_mgpu(
            dy,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    if implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU:
        raise NotImplementedError("compact source-push dy route requires SourcePushPlan metadata at the call site")
    raise ValueError(
        "source-push backward compact dy route implementation must be one of "
        f"{(
            SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
            SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU,
            SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU,
        )}, "
        f"got {implementation!r}"
    )


def _source_push_backward_dy_to_h_rows(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    dy: Float[Array, "S T D"],
    *,
    implementation: SourcePushDyRouteImplementation = SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDyRoutePallasBlockSizes | None = None,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst rows D"]:
    """Route ``dy`` into the same flat destination row layout as forward H.

    The metadata contract is intentionally the same one used by
    ``source_push_forward_with_h_from_plan``: ``host_inputs.plan`` supplies
    source queue order, ``host_inputs.send_meta`` supplies the row-start mode,
    and ``host_inputs.expert_base``/``src_base_by_expert`` describe the exact
    expert-major layout when requested. Invalid queue padding and unused flat
    rows are zeroed. The result is float32 to match the current compact
    backward reference.
    """

    _validate_dy_route_request(config, host_inputs, dy)
    if implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE:
        return _source_push_backward_dy_to_h_rows_reference(
            dy,
            host_inputs.plan,
            host_inputs.send_meta,
            host_inputs.expert_base,
            host_inputs.src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
    if implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU:
        inverse_indices = _source_push_dy_route_inverse_indices(
            host_inputs.plan,
            host_inputs.send_meta,
            host_inputs.expert_base,
            host_inputs.src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
        return _source_push_backward_dy_to_h_rows_pallas_mgpu(
            dy,
            inverse_indices,
            block_sizes=block_sizes,
            mesh=mesh,
        )
    if implementation == SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU:
        if mesh is None:
            raise ValueError("source_push_pallas_mgpu dy route requires a mesh")
        if jax.default_backend() != "gpu":
            raise NotImplementedError(
                "Pallas/MGPU source-push dy route requires a GPU backend; use the reference route on CPU"
            )
        block_sizes = SourcePushDyRoutePallasBlockSizes.get_default() if block_sizes is None else block_sizes
        _validate_dy_route_source_push_pallas_request(config, host_inputs, dy, block_sizes)
        return _source_push_backward_dy_to_h_rows_source_push_pallas_mgpu(
            config,
            dy,
            host_inputs.plan.token_ids,
            host_inputs.send_meta,
            block_sizes=block_sizes,
            mesh=mesh,
        )
    raise ValueError(
        "source-push backward dy route implementation must be one of "
        f"{(
            SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_REFERENCE,
            SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_PALLAS_MGPU,
            SOURCE_PUSH_DY_ROUTE_IMPLEMENTATION_SOURCE_PUSH_PALLAS_MGPU,
        )}, "
        f"got {implementation!r}"
    )


def _source_push_backward_dy_to_h_rows_reference(
    dy: Float[Array, "S T D"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool,
) -> Float[Array, "Dst rows D"]:
    """Readable JAX reference for the source-push backward dy route."""

    queue_dy = _source_push_queue_dy_rows(dy, plan)
    flat_dst, flat_row, valid_mask = _source_push_h_flat_indices(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=use_exact_expert_major,
    )

    routed_rows = jnp.where(valid_mask[..., None], queue_dy, jnp.zeros((), dtype=queue_dy.dtype))
    out = jnp.zeros((plan.assignment_ids.shape[0], hidden_rows_per_rank, dy.shape[-1]), dtype=jnp.float32)
    routed = out.at[flat_dst, flat_row].add(
        routed_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    return _with_source_push_sharding(routed, SOURCE_PUSH_MESH_AXIS, None, None)


def _source_push_queue_dy_rows(
    dy: Float[Array, "S T D"],
    plan: SourcePushPlan,
) -> Float[Array, "S Dst Q M D"]:
    """Gather source-owned ``dy`` into the source-push queue order."""

    source_indices = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None, None]
    token_ids = jnp.maximum(plan.token_ids, 0)
    queue_dy = dy.at[source_indices, token_ids].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None, None)
    )
    queue_dy = queue_dy.astype(jnp.float32)
    queue_dy = jnp.where(plan.valid_mask[..., None], queue_dy, jnp.zeros((), dtype=queue_dy.dtype))
    return _with_source_push_sharding(queue_dy, SOURCE_PUSH_MESH_AXIS, None, None, None, None)


def _source_push_h_flat_indices(
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool,
) -> tuple[Int[Array, "S Dst Q M"], Int[Array, "S Dst Q M"], Bool[Array, "S Dst Q M"]]:
    """Return destination rank and flat H-row indices for every queue row."""

    send_meta = jnp.asarray(send_meta, dtype=jnp.int32)
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    src_base_by_expert = jnp.asarray(src_base_by_expert, dtype=jnp.int32)
    valid_mask = jnp.asarray(plan.valid_mask, dtype=jnp.bool_)

    ep_size, _, entries_per_dst, block_m = valid_mask.shape
    src = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(ep_size, dtype=jnp.int32)[None, :, None]
    src = jnp.broadcast_to(src, (ep_size, ep_size, entries_per_dst))
    dst = (src + dst_ordinal) % ep_size

    metadata_row_start = send_meta[..., SOURCE_PUSH_META_LOCAL_ROW_START]
    if use_exact_expert_major:
        expert = jnp.maximum(send_meta[..., SOURCE_PUSH_META_LOCAL_EXPERT], 0)
        base_row = expert_base.at[dst, expert].get()
        src_base = src_base_by_expert.at[dst, src, expert].get()
        row_start = base_row + src_base + metadata_row_start
    else:
        row_start = metadata_row_start

    row_offsets = jnp.arange(block_m, dtype=jnp.int32)[None, None, None, :]
    flat_row = jnp.where(valid_mask, row_start[..., None] + row_offsets, jnp.zeros((), dtype=jnp.int32))
    flat_dst = jnp.where(valid_mask, jnp.broadcast_to(dst[..., None], flat_row.shape), jnp.zeros((), dtype=jnp.int32))
    return flat_dst, flat_row, valid_mask


def _source_push_dy_route_inverse_indices(
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool,
) -> _SourcePushDyRouteInverseIndices:
    """Build the destination flat-row inverse map consumed by the dy route kernel."""

    flat_dst, flat_row, valid_mask = _source_push_h_flat_indices(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=use_exact_expert_major,
    )
    dst_count = plan.assignment_ids.shape[0]
    source = jnp.arange(dst_count, dtype=jnp.int32)[:, None, None, None]
    source = jnp.broadcast_to(source, valid_mask.shape)
    safe_source = jnp.where(valid_mask, source, jnp.zeros((), dtype=jnp.int32))
    safe_token = jnp.where(valid_mask, jnp.maximum(plan.token_ids, 0), jnp.zeros((), dtype=jnp.int32))

    row_shape = (dst_count, hidden_rows_per_rank)
    row_src = jnp.zeros(row_shape, dtype=jnp.int32)
    row_token = jnp.zeros(row_shape, dtype=jnp.int32)
    row_valid_i = jnp.zeros(row_shape, dtype=jnp.int32)
    row_src = row_src.at[flat_dst, flat_row].add(
        safe_source,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    row_token = row_token.at[flat_dst, flat_row].add(
        safe_token,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    row_valid_i = row_valid_i.at[flat_dst, flat_row].add(
        valid_mask.astype(jnp.int32),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    return _SourcePushDyRouteInverseIndices(
        src=_with_source_push_sharding(row_src, SOURCE_PUSH_MESH_AXIS, None),
        token=_with_source_push_sharding(row_token, SOURCE_PUSH_MESH_AXIS, None),
        valid=_with_source_push_sharding(row_valid_i > 0, SOURCE_PUSH_MESH_AXIS, None),
    )


def _source_push_backward_dy_to_expert_major_reference(
    dy: Float[Array, "S T D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C D"]:
    """Readable compact JAX reference for the backward dy route."""

    safe_src = jnp.where(valid_by_expert, source_rank_by_expert, jnp.zeros((), dtype=jnp.int32))
    safe_token = jnp.where(valid_by_expert, token_id_by_expert, jnp.zeros((), dtype=jnp.int32))
    routed = dy.at[safe_src, safe_token].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    return jnp.where(valid_by_expert[..., None], routed.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))


def _source_push_backward_dy_to_expert_major_pallas_mgpu(
    dy: Float[Array, "S T D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushDyRoutePallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C D"]:
    """Destination-owned Pallas/MGPU gather into compact expert-major dy rows."""

    if interpret:
        return _source_push_backward_dy_to_expert_major_reference(
            dy,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
        )
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU compact dy route requires a GPU backend; use reference on CPU")
    block_sizes = SourcePushDyRoutePallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dy_route_compact_pallas_request(
        dy,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes,
    )
    return _source_push_backward_dy_to_expert_major_pallas_call(
        dy,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )


def _source_push_backward_dy_to_expert_major_pallas_call(
    dy: Float[Array, "S T D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C D"]:
    capacity = source_rank_by_expert.shape[2]
    padded_capacity = _round_up_to_multiple(capacity, row_block)
    if padded_capacity != capacity:
        pad_width = ((0, 0), (0, 0), (0, padded_capacity - capacity))
        source_rank_by_expert = jnp.pad(source_rank_by_expert, pad_width, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, pad_width, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, pad_width, constant_values=False)

    if mesh is not None and not interpret:
        routed = _source_push_backward_dy_to_expert_major_sharded_pallas_call(
            mesh,
            dy,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            row_block=row_block,
            hidden_block=hidden_block,
        )
        return routed[:, :, :capacity, :]

    valid_i32_by_expert = valid_by_expert.astype(jnp.int32)
    output_shape = jax.ShapeDtypeStruct(source_rank_by_expert.shape + (dy.shape[-1],), jnp.float32)
    cost_estimate = _source_push_backward_dy_compact_pallas_cost_estimate(
        dy,
        source_rank_by_expert,
        token_id_by_expert,
        valid_i32_by_expert,
        output_shape,
    )
    kernel = _make_source_push_backward_dy_route_compact_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
    )
    in_specs, out_spec = _source_push_backward_dy_route_compact_block_specs(
        row_block=row_block,
        hidden_block=hidden_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    routed = pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=output_shape,
        grid=(
            source_rank_by_expert.shape[0],
            source_rank_by_expert.shape[1],
            source_rank_by_expert.shape[2] // row_block,
            dy.shape[-1] // hidden_block,
        ),
        interpret=interpret,
        name="source_push_backward_dy_route_compact_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dy, source_rank_by_expert, token_id_by_expert, valid_i32_by_expert)
    return routed[:, :, :capacity, :]


def _source_push_backward_dy_to_expert_major_sharded_pallas_call(
    mesh: Mesh,
    dy: Float[Array, "S T D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C D"]:
    """Run the compact dy-route kernel with destination-local route metadata refs."""

    def local_fn(
        dy_local: Float[Array, "1 T D"],
        source_rank_local: Int[Array, "1 E C"],
        token_id_local: Int[Array, "1 E C"],
        valid_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 E C D"]:
        dy_all = lax.all_gather(dy_local[0], SOURCE_PUSH_MESH_AXIS, axis=0)
        return _source_push_backward_dy_to_expert_major_pallas_call(
            dy_all,
            source_rank_local,
            token_id_local,
            valid_local,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=False,
            mesh=None,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dy, source_rank_by_expert, token_id_by_expert, valid_by_expert)


def _source_push_backward_dy_to_h_rows_pallas_mgpu(
    dy: Float[Array, "S T D"],
    inverse_indices: _SourcePushDyRouteInverseIndices,
    *,
    block_sizes: SourcePushDyRoutePallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst rows D"]:
    """Pallas/MGPU flat-H dy route kernel.

    The inverse map makes the kernel destination-owned: every ``[dst, row, D]``
    tile is written exactly once, and rows not present in the map are zeroed.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU dy route requires a GPU backend; use the reference route on CPU")
    block_sizes = SourcePushDyRoutePallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dy_route_pallas_request(dy, inverse_indices, block_sizes)
    return _source_push_backward_dy_to_h_rows_pallas_call(
        dy,
        inverse_indices.src,
        inverse_indices.token,
        inverse_indices.valid,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )


def _source_push_backward_dy_to_h_rows_pallas_call(
    dy: Float[Array, "S T D"],
    row_src: Int[Array, "Dst rows"],
    row_token: Int[Array, "Dst rows"],
    row_valid: Bool[Array, "Dst rows"],
    *,
    row_block: int,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst rows D"]:
    if mesh is not None and not interpret:
        return _source_push_backward_dy_to_h_rows_sharded_pallas_call(
            mesh,
            dy,
            row_src,
            row_token,
            row_valid,
            row_block=row_block,
            hidden_block=hidden_block,
        )

    dst_count, hidden_rows_per_rank = row_valid.shape
    hidden_dim = dy.shape[-1]
    output_shape = jax.ShapeDtypeStruct((dst_count, hidden_rows_per_rank, hidden_dim), jnp.float32)
    cost_estimate = _source_push_backward_dy_route_pallas_cost_estimate(
        dy,
        row_src,
        row_token,
        row_valid,
        output_shape,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    kernel = _make_source_push_backward_dy_route_kernel(row_block=row_block, hidden_block=hidden_block)
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(dst_count, hidden_rows_per_rank // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_backward_dy_route_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dy, row_src, row_token, row_valid)


def _source_push_backward_dy_to_h_rows_sharded_pallas_call(
    mesh: Mesh,
    dy: Float[Array, "S T D"],
    row_src: Int[Array, "Dst rows"],
    row_token: Int[Array, "Dst rows"],
    row_valid: Bool[Array, "Dst rows"],
    *,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst rows D"]:
    """Run the dy-route Pallas kernel with destination-local metadata refs.

    This is a correctness bridge for the destination-owned gather kernel. The
    metadata/output first axis is made local with ``shard_map`` so Pallas does
    not integer-index a sharded ref. ``dy`` is explicitly gathered across the EP
    axis; replacing that with source-push remote writes is the performance path.
    """

    def local_fn(
        dy_local: Float[Array, "1 T D"],
        row_src_local: Int[Array, "1 rows"],
        row_token_local: Int[Array, "1 rows"],
        row_valid_local: Bool[Array, "1 rows"],
    ) -> Float[Array, "1 rows D"]:
        dy_all = lax.all_gather(dy_local[0], SOURCE_PUSH_MESH_AXIS, axis=0)
        return _source_push_backward_dy_to_h_rows_pallas_call(
            dy_all,
            row_src_local,
            row_token_local,
            row_valid_local,
            row_block=row_block,
            hidden_block=hidden_block,
            interpret=False,
            mesh=None,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None),
            P(SOURCE_PUSH_MESH_AXIS, None),
            P(SOURCE_PUSH_MESH_AXIS, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(dy, row_src, row_token, row_valid)


def _source_push_backward_dy_to_h_rows_source_push_pallas_mgpu(
    config: PushInboxConfig,
    dy: Float[Array, "S T D"],
    token_ids: Int[Array, "S Dst Q M"],
    send_meta: Int[Array, "S Dst Q F"],
    *,
    block_sizes: SourcePushDyRoutePallasBlockSizes,
    mesh: Mesh,
) -> Float[Array, "Dst rows D"]:
    """Source-push dy route kernel using remote writes into destination rows."""

    return _source_push_backward_dy_to_h_rows_source_push_pallas_call(
        mesh,
        dy,
        token_ids,
        send_meta,
        ep_size=config.ep_size,
        entries_per_rank=config.entries_per_rank,
        block_m=config.block_m,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )


def _source_push_backward_dy_to_h_rows_source_push_pallas_call(
    mesh: Mesh,
    dy: Float[Array, "S T D"],
    token_ids: Int[Array, "S Dst Q M"],
    send_meta: Int[Array, "S Dst Q F"],
    *,
    ep_size: int,
    entries_per_rank: int,
    block_m: int,
    hidden_rows_per_rank: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst rows D"]:
    """Run source-owned dy remote writes under ``shard_map``.

    Each source rank owns ``dy`` and source queue metadata. It writes accepted
    route rows to the destination-owned flat H-row layout. The output is aliased
    with a zero input buffer so rows that are not written by any source remain
    zero.
    """

    routed_init = jax.device_put(
        jnp.zeros((ep_size, hidden_rows_per_rank, dy.shape[-1]), dtype=jnp.float32),
        NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None)),
    )

    def local_fn(
        dy_local: Float[Array, "1 T D"],
        token_ids_global: Int[Array, "S Dst Q M"],
        send_meta_global: Int[Array, "S Dst Q F"],
        routed_init_local: Float[Array, "1 rows D"],
    ) -> Float[Array, "1 rows D"]:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        dy_local = dy_local[0]
        token_ids_local = token_ids_global[rank]
        send_meta_local = send_meta_global[rank]
        routed = _source_push_backward_dy_to_h_rows_source_push_pallas_local_call(
            dy_local,
            token_ids_local,
            send_meta_local,
            routed_init_local[0],
            ep_size=ep_size,
            entries_per_rank=entries_per_rank,
            block_m=block_m,
            row_block=row_block,
            hidden_block=hidden_block,
        )
        return routed[None, ...]

    routed = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(dy, token_ids, send_meta, routed_init)
    return _source_push_backward_dy_remote_write_barrier(mesh)(routed)


def _source_push_backward_dy_to_expert_major_source_push_pallas_call(
    mesh: Mesh,
    dy: Float[Array, "S T D"],
    token_ids: Int[Array, "S Dst Q M"],
    send_meta: Int[Array, "S Dst Q F"],
    *,
    ep_size: int,
    entries_per_rank: int,
    block_m: int,
    experts_per_rank: int,
    expert_capacity: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C D"]:
    """Source-owned remote writes from ``dy`` into compact expert-major rows."""

    routed_init = jax.device_put(
        jnp.zeros((ep_size, experts_per_rank, expert_capacity, dy.shape[-1]), dtype=jnp.float32),
        NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None)),
    )

    def local_fn(
        dy_local: Float[Array, "1 T D"],
        token_ids_global: Int[Array, "S Dst Q M"],
        send_meta_global: Int[Array, "S Dst Q F"],
        routed_init_local: Float[Array, "1 E C D"],
    ) -> Float[Array, "1 E C D"]:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        routed = _source_push_backward_dy_to_expert_major_source_push_pallas_local_call(
            dy_local[0],
            token_ids_global[rank],
            send_meta_global[rank],
            routed_init_local[0],
            ep_size=ep_size,
            entries_per_rank=entries_per_rank,
            block_m=block_m,
            row_block=row_block,
            hidden_block=hidden_block,
        )
        return routed[None, ...]

    routed = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(None, None, None, None),
            P(None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dy, token_ids, send_meta, routed_init)
    return _source_push_backward_dy_expert_major_remote_write_barrier(mesh)(routed)


def _source_push_backward_dy_to_expert_major_source_push_pallas_local_call(
    dy: Float[Array, "T D"],
    token_ids: Int[Array, "Dst Q M"],
    send_meta: Int[Array, "Dst Q F"],
    zero: Float[Array, "E C D"],
    *,
    ep_size: int,
    entries_per_rank: int,
    block_m: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "E C D"]:
    hidden_dim = dy.shape[-1]
    output_shape = jax.ShapeDtypeStruct(zero.shape, zero.dtype)
    kernel = _make_source_push_backward_dy_route_expert_major_source_push_kernel(
        ep_size=ep_size,
        block_m=block_m,
        row_block=row_block,
        hidden_block=hidden_block,
    )
    cost_estimate = _source_push_backward_dy_source_push_pallas_cost_estimate(
        token_ids,
        send_meta,
        output_shape,
        hidden_block=hidden_block,
        row_block=row_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        kernel,
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(ep_size, entries_per_rank, block_m // row_block, hidden_dim // hidden_block),
        input_output_aliases={3: 0},
        name="source_push_backward_dy_route_expert_major_source_push_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dy, token_ids, send_meta, zero)


def _source_push_backward_dy_to_h_rows_source_push_pallas_local_call(
    dy: Float[Array, "T D"],
    token_ids: Int[Array, "Dst Q M"],
    send_meta: Int[Array, "Dst Q F"],
    zero: Float[Array, "rows D"],
    *,
    ep_size: int,
    entries_per_rank: int,
    block_m: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "rows D"]:
    hidden_dim = dy.shape[-1]
    output_shape = jax.ShapeDtypeStruct(zero.shape, zero.dtype)
    kernel = _make_source_push_backward_dy_route_source_push_kernel(
        ep_size=ep_size,
        block_m=block_m,
        row_block=row_block,
        hidden_block=hidden_block,
    )
    cost_estimate = _source_push_backward_dy_source_push_pallas_cost_estimate(
        token_ids,
        send_meta,
        output_shape,
        hidden_block=hidden_block,
        row_block=row_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(ep_size, entries_per_rank, block_m // row_block, hidden_dim // hidden_block),
        input_output_aliases={3: 0},
        name="source_push_backward_dy_route_source_push_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dy, token_ids, send_meta, zero)


def _make_source_push_backward_dy_route_source_push_kernel(
    *,
    ep_size: int,
    block_m: int,
    row_block: int,
    hidden_block: int,
):
    dst_offsets = tuple(range(ep_size))

    def kernel(
        dy_ref: Float[pl.Ref, "T D"],
        token_ids_ref: Int[pl.Ref, "Dst Q M"],
        send_meta_ref: Int[pl.Ref, "Dst Q F"],
        _zero_ref: Float[pl.Ref, "rows D"],
        out_ref: Float[pl.Ref, "rows D"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        dst_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_offset_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        def _write_to_dst(static_dst_ordinal: int) -> None:
            dst = (rank + static_dst_ordinal) % ep_size
            valid_rows = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            @pl.when(valid_rows > row_offset_start)
            def _write_live_tile() -> None:
                row_start = (
                    send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START] + row_offset_start
                )

                if static_dst_ordinal == 0:

                    @pl.loop(0, row_block)
                    def _row_loop(row) -> None:
                        @pl.when((row_offset_start + row) < valid_rows)
                        def _store_valid_row() -> None:
                            token = token_ids_ref[static_dst_ordinal, entry, row_offset_start + row]
                            out_row = dy_ref[token, pl.ds(hidden_start, hidden_block)].astype(jnp.float32)
                            out_ref[row_start + row, pl.ds(hidden_start, hidden_block)] = out_row

                else:
                    remote_out_ref = mgpu.remote_ref(out_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                    @pl.loop(0, row_block)
                    def _row_loop(row) -> None:
                        @pl.when((row_offset_start + row) < valid_rows)
                        def _store_valid_row() -> None:
                            token = token_ids_ref[static_dst_ordinal, entry, row_offset_start + row]
                            out_row = dy_ref[token, pl.ds(hidden_start, hidden_block)].astype(jnp.float32)
                            remote_out_ref[row_start + row, pl.ds(hidden_start, hidden_block)] = out_row

        def _branch(static_dst_ordinal: int):
            def _write_branch(_) -> None:
                _write_to_dst(static_dst_ordinal)

            return _write_branch

        lax.switch(dst_ordinal, tuple(_branch(dst_offset) for dst_offset in dst_offsets), None)

    return kernel


def _make_source_push_backward_dy_route_expert_major_source_push_kernel(
    *,
    ep_size: int,
    block_m: int,
    row_block: int,
    hidden_block: int,
):
    dst_offsets = tuple(range(ep_size))

    def kernel(
        dy_ref: Float[pl.Ref, "T D"],
        token_ids_ref: Int[pl.Ref, "Dst Q M"],
        send_meta_ref: Int[pl.Ref, "Dst Q F"],
        _zero_ref: Float[pl.Ref, "E C D"],
        out_ref: Float[pl.Ref, "E C D"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        dst_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_offset_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        def _write_to_dst(static_dst_ordinal: int) -> None:
            dst = (rank + static_dst_ordinal) % ep_size
            valid_rows = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            @pl.when(valid_rows > row_offset_start)
            def _write_live_tile() -> None:
                expert = send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                row_start = (
                    send_meta_ref[static_dst_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START] + row_offset_start
                )

                if static_dst_ordinal == 0:

                    @pl.loop(0, row_block)
                    def _row_loop(row) -> None:
                        @pl.when((row_offset_start + row) < valid_rows)
                        def _store_valid_row() -> None:
                            token = token_ids_ref[static_dst_ordinal, entry, row_offset_start + row]
                            out_row = dy_ref[token, pl.ds(hidden_start, hidden_block)].astype(jnp.float32)
                            out_ref[expert, row_start + row, pl.ds(hidden_start, hidden_block)] = out_row

                else:
                    remote_out_ref = mgpu.remote_ref(out_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                    @pl.loop(0, row_block)
                    def _row_loop(row) -> None:
                        @pl.when((row_offset_start + row) < valid_rows)
                        def _store_valid_row() -> None:
                            token = token_ids_ref[static_dst_ordinal, entry, row_offset_start + row]
                            out_row = dy_ref[token, pl.ds(hidden_start, hidden_block)].astype(jnp.float32)
                            remote_out_ref[
                                expert,
                                row_start + row,
                                pl.ds(hidden_start, hidden_block),
                            ] = out_row

        def _branch(static_dst_ordinal: int):
            def _write_branch(_) -> None:
                _write_to_dst(static_dst_ordinal)

            return _write_branch

        lax.switch(dst_ordinal, tuple(_branch(dst_offset) for dst_offset in dst_offsets), None)

    return kernel


def _source_push_backward_dy_remote_write_barrier(mesh: Mesh):
    """Synchronize after dy-route remote writes before reading destination rows."""

    def local_fn(routed_local: Float[Array, "1 rows D"]):
        routed_local = routed_local[0]
        marker = routed_local[0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = (barrier - lax.optimization_barrier(barrier)).astype(routed_local.dtype)
        routed_local = routed_local.at[0, 0].add(zero)
        return routed_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )


def _source_push_backward_dy_expert_major_remote_write_barrier(mesh: Mesh):
    """Synchronize after compact dy-route remote writes before reading destination blocks."""

    def local_fn(routed_local: Float[Array, "1 E C D"]):
        routed_local = routed_local[0]
        marker = routed_local[0, 0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = (barrier - lax.optimization_barrier(barrier)).astype(routed_local.dtype)
        routed_local = routed_local.at[0, 0, 0].add(zero)
        return routed_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )


def _source_push_backward_dy_source_push_pallas_cost_estimate(
    token_ids: Array,
    send_meta: Array,
    output_shape: jax.ShapeDtypeStruct,
    *,
    hidden_block: int,
    row_block: int,
) -> pl.CostEstimate:
    written_bytes = int(np.prod(output_shape.shape)) * jnp.dtype(output_shape.dtype).itemsize
    token_bytes = int(np.prod(token_ids.shape)) * token_ids.dtype.itemsize
    meta_bytes = int(np.prod(send_meta.shape)) * send_meta.dtype.itemsize
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=written_bytes + token_bytes + meta_bytes,
        remote_bytes_transferred=written_bytes,
    )


def _make_source_push_backward_dy_route_kernel(*, row_block: int, hidden_block: int):
    def kernel(
        dy_ref: Float[pl.Ref, "S T D"],
        row_src_ref: Int[pl.Ref, "Dst rows"],
        row_token_ref: Int[pl.Ref, "Dst rows"],
        row_valid_ref: Bool[pl.Ref, "Dst rows"],
        out_ref: Float[pl.Ref, "Dst rows D"],
    ) -> None:
        dst = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        hidden_offsets = hidden_start + jnp.arange(hidden_block, dtype=jnp.int32)

        src = row_src_ref[dst, pl.ds(row_start, row_block)]
        token = row_token_ref[dst, pl.ds(row_start, row_block)]
        valid = row_valid_ref[dst, pl.ds(row_start, row_block)]
        dy_tile = dy_ref[src[:, None], token[:, None], hidden_offsets[None, :]].astype(jnp.float32)
        zeros = jnp.zeros((row_block, hidden_block), dtype=jnp.float32)
        out_ref[dst, pl.ds(row_start, row_block), pl.ds(hidden_start, hidden_block)] = jnp.where(
            valid[:, None],
            dy_tile,
            zeros,
        )

    return kernel


def _source_push_backward_dy_route_compact_block_specs(
    *,
    row_block: int,
    hidden_block: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    route_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    out_spec = pl.BlockSpec(
        (1, 1, row_block, hidden_block),
        lambda dst, expert, row_tile, hidden_tile: (dst, expert, row_tile, hidden_tile),
    )
    dy_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (dy_spec, route_spec, route_spec, route_spec), out_spec


def _make_source_push_backward_dy_route_compact_kernel(
    *,
    row_block: int,
    hidden_block: int,
):
    def kernel(
        dy_ref: Float[pl.Ref, "S T D"],
        source_rank_ref: Int[pl.Ref, "Dst E C"],
        token_id_ref: Int[pl.Ref, "Dst E C"],
        valid_i32_ref: Int[pl.Ref, "Dst E C"],
        out_ref: Float[pl.Ref, "C D"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        src = source_rank_ref[dst, expert, pl.ds(row_start, row_block)]
        token = token_id_ref[dst, expert, pl.ds(row_start, row_block)]
        valid = valid_i32_ref[dst, expert, pl.ds(row_start, row_block)] != 0
        safe_src = jnp.where(valid, src, jnp.zeros((), dtype=src.dtype))
        safe_token = jnp.where(valid, token, jnp.zeros((), dtype=token.dtype))
        dy_tile = dy_ref[safe_src, safe_token, pl.ds(hidden_start, hidden_block)].astype(jnp.float32)
        zeros = jnp.zeros((row_block, hidden_block), dtype=jnp.float32)
        out_ref[0, 0, pl.ds(0, row_block), pl.ds(0, hidden_block)] = jnp.where(
            valid[:, None],
            dy_tile,
            zeros,
        )

    return kernel


def _source_push_backward_dy_route_pallas_reference(
    dy: Float[Array, "S T D"],
    row_src: Int[Array, "Dst rows"],
    row_token: Int[Array, "Dst rows"],
    row_valid: Bool[Array, "Dst rows"],
) -> Float[Array, "Dst rows D"]:
    safe_src = jnp.where(row_valid, row_src, jnp.zeros((), dtype=jnp.int32))
    safe_token = jnp.where(row_valid, row_token, jnp.zeros((), dtype=jnp.int32))
    routed = dy.at[safe_src, safe_token].get().astype(jnp.float32)
    return jnp.where(row_valid[..., None], routed, jnp.zeros((), dtype=jnp.float32))


def _source_push_backward_dy_route_pallas_cost_estimate(
    dy: Array,
    row_src: Array,
    row_token: Array,
    row_valid: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(dy.shape, dy.dtype),
        jax.ShapeDtypeStruct(row_src.shape, row_src.dtype),
        jax.ShapeDtypeStruct(row_token.shape, row_token.dtype),
        jax.ShapeDtypeStruct(row_valid.shape, row_valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_backward_dy_route_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_backward_dy_compact_pallas_cost_estimate(
    dy: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    valid_by_expert: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    metadata_bytes = sum(
        int(np.prod(array.shape)) * jnp.dtype(array.dtype).itemsize
        for array in (source_rank_by_expert, token_id_by_expert, valid_by_expert)
    )
    dy_bytes = int(np.prod(dy.shape)) * jnp.dtype(dy.dtype).itemsize
    output_bytes = int(np.prod(output_shape.shape)) * jnp.dtype(output_shape.dtype).itemsize
    return pl.CostEstimate(
        flops=0,
        transcendentals=0,
        bytes_accessed=metadata_bytes + dy_bytes + output_bytes,
        remote_bytes_transferred=0,
    )


def _validate_dy_route_request(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    dy: Float[Array, "S T D"],
) -> None:
    expected_dy = (config.ep_size, config.tokens_per_rank, config.hidden_dim)
    if dy.shape != expected_dy:
        raise ValueError(f"dy shape {dy.shape} must match {expected_dy}")

    expected_queue = (config.ep_size, config.ep_size, config.entries_per_rank, config.block_m)
    if host_inputs.plan.assignment_ids.shape != expected_queue:
        raise ValueError(f"plan queue shape {host_inputs.plan.assignment_ids.shape} must match {expected_queue}")
    if host_inputs.plan.token_ids.shape != expected_queue:
        raise ValueError(f"plan token_ids shape {host_inputs.plan.token_ids.shape} must match {expected_queue}")
    if host_inputs.plan.valid_mask.shape != expected_queue:
        raise ValueError(f"plan valid_mask shape {host_inputs.plan.valid_mask.shape} must match {expected_queue}")

    expected_meta = (*expected_queue[:3], SOURCE_PUSH_META_FIELDS)
    if host_inputs.send_meta.shape != expected_meta:
        raise ValueError(f"send_meta shape {host_inputs.send_meta.shape} must match {expected_meta}")

    expected_expert_base = (config.ep_size, config.experts_per_rank)
    if host_inputs.expert_base.shape != expected_expert_base:
        raise ValueError(f"expert_base shape {host_inputs.expert_base.shape} must match {expected_expert_base}")

    expected_src_base = (config.ep_size, config.ep_size, config.experts_per_rank)
    if host_inputs.src_base_by_expert.shape != expected_src_base:
        raise ValueError(
            f"src_base_by_expert shape {host_inputs.src_base_by_expert.shape} must match {expected_src_base}"
        )

    if host_inputs.plan.tokens_per_source != config.tokens_per_rank:
        raise ValueError(
            f"plan tokens_per_source {host_inputs.plan.tokens_per_source} must match {config.tokens_per_rank}"
        )
    if host_inputs.plan.topk != config.topk:
        raise ValueError(f"plan topk {host_inputs.plan.topk} must match {config.topk}")


def _validate_dy_route_pallas_request(
    dy: Array,
    inverse_indices: _SourcePushDyRouteInverseIndices,
    block_sizes: SourcePushDyRoutePallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, D], got {dy.shape}")
    if (
        inverse_indices.src.shape != inverse_indices.token.shape
        or inverse_indices.src.shape != inverse_indices.valid.shape
    ):
        raise ValueError(
            "inverse index shapes must match; got "
            f"src={inverse_indices.src.shape}, token={inverse_indices.token.shape}, "
            f"valid={inverse_indices.valid.shape}"
        )
    if inverse_indices.src.ndim != 2:
        raise ValueError(f"inverse indices must have shape [dst, rows], got {inverse_indices.src.shape}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if inverse_indices.src.shape[1] % block_sizes.row_block:
        raise ValueError(
            "inverse row count must be divisible by row_block; "
            f"got rows={inverse_indices.src.shape[1]} row_block={block_sizes.row_block}"
        )
    if dy.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dy hidden dimension must be divisible by hidden_block; "
            f"got D={dy.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )


def _validate_dy_route_compact_pallas_request(
    dy: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    valid_by_expert: Array,
    block_sizes: SourcePushDyRoutePallasBlockSizes,
) -> None:
    if dy.ndim != 3:
        raise ValueError(f"dy must have shape [source, token, D], got {dy.shape}")
    if source_rank_by_expert.shape != token_id_by_expert.shape or source_rank_by_expert.shape != valid_by_expert.shape:
        raise ValueError(
            "compact route metadata shapes must match; got "
            f"source_rank={source_rank_by_expert.shape}, token_id={token_id_by_expert.shape}, "
            f"valid={valid_by_expert.shape}"
        )
    if source_rank_by_expert.ndim != 3:
        raise ValueError(
            f"compact route metadata must have shape [dst, expert, capacity], got {source_rank_by_expert.shape}"
        )
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if dy.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dy hidden dimension must be divisible by hidden_block; "
            f"got D={dy.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )


def _validate_dy_route_source_push_pallas_request(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    dy: Array,
    block_sizes: SourcePushDyRoutePallasBlockSizes,
) -> None:
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if config.block_m % block_sizes.row_block:
        raise ValueError(
            "source-push dy route requires block_m divisible by row_block; "
            f"got block_m={config.block_m} row_block={block_sizes.row_block}"
        )
    if dy.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            "dy hidden dimension must be divisible by hidden_block; "
            f"got D={dy.shape[-1]} hidden_block={block_sizes.hidden_block}"
        )
    expected_token_ids = (config.ep_size, config.ep_size, config.entries_per_rank, config.block_m)
    if host_inputs.plan.token_ids.shape != expected_token_ids:
        raise ValueError(f"plan token_ids shape {host_inputs.plan.token_ids.shape} must match {expected_token_ids}")
    expected_meta = (config.ep_size, config.ep_size, config.entries_per_rank, SOURCE_PUSH_META_FIELDS)
    if host_inputs.send_meta.shape != expected_meta:
        raise ValueError(f"send_meta shape {host_inputs.send_meta.shape} must match {expected_meta}")
