# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Package-private DX13 diagnostic helpers for source-push MoE backward.

The production-relevant boundary is expert-owner compute followed by a
source-push compact contribution return.  The Pallas source-compact path
computes the W13 input-gradient contribution from local ``d_activation``, saved
W13 preactivation ``z``, and local ``w13``, then remote-stores the result into a
source-owned compact contribution queue.  It deliberately stops before final
token accumulation: source-local combine remains a separate step and does not
require remote atomics into ``x.grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushPlan,
    _source_push_out_sharding,
    source_push_route_rows_host_from_plan,
)


SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SourcePushDx13Implementation: TypeAlias = Literal["reference", "pallas_mgpu"]
DEFAULT_DX13_ROW_BLOCK = 64
DEFAULT_DX13_HIDDEN_BLOCK = 128
DEFAULT_DX13_OUTPUT_BLOCK = 64
MIN_MOSAIC_INT32_TRANSFER_ELEMENTS = 128
DX13_WGMMA_SWIZZLE_BYTES = 128
DX13_WGMMA_TILE_M = 8


@dataclass(frozen=True, slots=True)
class SourcePushDx13PallasBlockSizes:
    """Tile sizes for the local expert-owner DX13 WGMMA diagnostic."""

    row_block: int = DEFAULT_DX13_ROW_BLOCK
    hidden_block: int = DEFAULT_DX13_HIDDEN_BLOCK
    output_block: int = DEFAULT_DX13_OUTPUT_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushDx13PallasBlockSizes":
        return cls()


def source_push_dx13_pallas_resolved_block_sizes(
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    *,
    interpret: bool = False,
) -> SourcePushDx13PallasBlockSizes:
    """Return the concrete DX13 Pallas tile sizes used by the GPU wrapper."""

    block_sizes = SourcePushDx13PallasBlockSizes.get_default() if block_sizes is None else block_sizes
    if interpret or block_sizes.row_block >= MIN_MOSAIC_INT32_TRANSFER_ELEMENTS:
        return block_sizes
    return SourcePushDx13PallasBlockSizes(
        row_block=MIN_MOSAIC_INT32_TRANSFER_ELEMENTS,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )


class SourcePushDx13CompactOutput(NamedTuple):
    """Expert-major DX13 contributions plus source-return metadata."""

    dx_expert_major: Float[Array, "Dst E C D"]
    source_rank_by_expert: Int[Array, "Dst E C"]
    token_id_by_expert: Int[Array, "Dst E C"]
    route_slot_by_expert: Int[Array, "Dst E C"]
    valid_by_expert: Bool[Array, "Dst E C"]


class SourcePushDx13RouteBufferOutput(NamedTuple):
    """Source-owned DX13 route buffer plus source-local token combine."""

    dx_routes: Float[Array, "S T K D"]
    dx: Float[Array, "S T D"]


class SourcePushDx13SourceCompactSlots(NamedTuple):
    """Expert-major rows mapped back to source-owned compact assignment slots."""

    source_rank_by_expert: Int[Array, "Dst E C"]
    dst_ordinal_by_expert: Int[Array, "Dst E C"]
    entry_by_expert: Int[Array, "Dst E C"]
    row_in_entry_by_expert: Int[Array, "Dst E C"]
    valid_by_expert: Bool[Array, "Dst E C"]


class SourcePushDx13SourceCompactOutput(NamedTuple):
    """DX13 contributions in the source-local compact queue layout."""

    dx_contrib: Float[Array, "S Dst Q M D"]


class SourcePushDx13SourceGroupedOutput(NamedTuple):
    """DX13 contributions grouped by original source rank before token combine.

    This is the intended source-push return boundary.  Expert-owner programs can
    write contiguous source chunks instead of scattering each row directly into
    ``[source, token, route_slot]``.  The source-local combine can then use the
    token/slot metadata to build ``dx`` without remote atomics.
    """

    dx_by_source: Float[Array, "Dst S E Csrc D"]
    token_id_by_source: Int[Array, "Dst S E Csrc"]
    route_slot_by_source: Int[Array, "Dst S E Csrc"]
    valid_by_source: Bool[Array, "Dst S E Csrc"]


def source_push_dx13_compact_assignment_slots_from_fields(
    source_rank_by_expert: Int[Array, "Dst E C"],
    dst_ordinal_by_expert: Int[Array, "Dst E C"],
    entry_by_expert: Int[Array, "Dst E C"],
    row_in_entry_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> SourcePushDx13SourceCompactSlots:
    """Package route-table fields as source-compact DX13 contribution slots."""

    return SourcePushDx13SourceCompactSlots(
        source_rank_by_expert=source_rank_by_expert,
        dst_ordinal_by_expert=dst_ordinal_by_expert,
        entry_by_expert=entry_by_expert,
        row_in_entry_by_expert=row_in_entry_by_expert,
        valid_by_expert=valid_by_expert,
    )


def source_push_dx13_compact_assignment_slots_from_plan(
    plan: SourcePushPlan,
    *,
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray | None = None,
    expert_capacity: int | None = None,
) -> SourcePushDx13SourceCompactSlots:
    """Derive the source compact queue slot for each expert-major DX13 row.

    The source-push plan is source-queue-owned, while DX13 work is
    expert-owner-major. This helper is the inverse metadata a production
    push-contribution kernel needs: for each ``[dst, expert, row]`` contribution,
    identify the owning source rank and the source-local compact assignment slot
    ``[dst_ordinal, entry, row_in_entry]``. The resulting buffer can be combined
    later on the source rank using the plan's token/route-slot metadata.
    """

    route_rows = source_push_route_rows_host_from_plan(plan, src_base_by_expert=src_base_by_expert)
    valid_host = np.asarray(route_rows.valid, dtype=np.bool_)
    source_count, dst_count, entries_per_dst, block_m = valid_host.shape
    if expert_capacity is None:
        if np.any(valid_host):
            expert_capacity = int(np.max(route_rows.expert_row[valid_host])) + 1
        else:
            expert_capacity = 0
    if expert_capacity < 0:
        raise ValueError(f"expert_capacity must be nonnegative, got {expert_capacity}")
    experts_per_rank = int(np.asarray(jax.device_get(plan.counts_by_src_dst_expert)).shape[-1])
    out_shape = (dst_count, experts_per_rank, expert_capacity)

    source_rank = np.zeros(out_shape, dtype=np.int32)
    dst_ordinal = np.zeros(out_shape, dtype=np.int32)
    entry_by_expert = np.zeros(out_shape, dtype=np.int32)
    row_in_entry_by_expert = np.zeros(out_shape, dtype=np.int32)
    valid_by_expert = np.zeros(out_shape, dtype=np.bool_)

    for src, dst_ord, entry, row_in_entry in np.argwhere(valid_host):
        dst = int(route_rows.dst[src, dst_ord, entry, row_in_entry])
        expert = int(route_rows.local_expert[src, dst_ord, entry, row_in_entry])
        expert_row = int(route_rows.expert_row[src, dst_ord, entry, row_in_entry])
        if expert_row >= expert_capacity:
            raise ValueError(f"expert_capacity={expert_capacity} is too small for expert row {expert_row}")
        if valid_by_expert[dst, expert, expert_row]:
            raise ValueError(
                "source-push plan maps multiple source compact slots to "
                f"[dst={dst}, expert={expert}, row={expert_row}]"
            )
        source_rank[dst, expert, expert_row] = int(src)
        dst_ordinal[dst, expert, expert_row] = int(dst_ord)
        entry_by_expert[dst, expert, expert_row] = int(entry)
        row_in_entry_by_expert[dst, expert, expert_row] = int(row_in_entry)
        valid_by_expert[dst, expert, expert_row] = True

    _ = (source_count, entries_per_dst, block_m)
    return SourcePushDx13SourceCompactSlots(
        source_rank_by_expert=jnp.asarray(source_rank, dtype=jnp.int32),
        dst_ordinal_by_expert=jnp.asarray(dst_ordinal, dtype=jnp.int32),
        entry_by_expert=jnp.asarray(entry_by_expert, dtype=jnp.int32),
        row_in_entry_by_expert=jnp.asarray(row_in_entry_by_expert, dtype=jnp.int32),
        valid_by_expert=jnp.asarray(valid_by_expert, dtype=jnp.bool_),
    )


def source_push_dx13_push_contrib(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    compact_slots: SourcePushDx13SourceCompactSlots,
    *,
    queue_shape: tuple[int, int, int, int],
    implementation: SourcePushDx13Implementation = SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13SourceCompactOutput:
    """Compute DX13 and push contributions into source compact assignment slots."""

    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE:
        return source_push_dx13_push_contrib_reference(
            d_activation,
            z,
            w13,
            compact_slots,
            queue_shape=queue_shape,
        )
    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU:
        if interpret:
            return source_push_dx13_push_contrib_reference(
                d_activation,
                z,
                w13,
                compact_slots,
                queue_shape=queue_shape,
            )
        if jax.default_backend() != "gpu":
            raise NotImplementedError("Pallas/MGPU DX13 source-compact push requires a GPU backend")
        if mesh is None:
            raise ValueError("Pallas/MGPU DX13 source-compact push requires a mesh")
        source_count, dst_count, entries_per_dst, block_m = queue_shape
        if source_count != d_activation.shape[0]:
            raise ValueError(f"queue source count {source_count} must match d_activation {d_activation.shape[0]}")
        if min(dst_count, entries_per_dst, block_m) <= 0:
            raise ValueError(f"source compact queue dimensions must be positive, got {queue_shape}")
        _validate_source_compact_slot_metadata(d_activation.shape[:3], compact_slots, queue_shape)
        block_sizes = source_push_dx13_pallas_resolved_block_sizes(block_sizes, interpret=interpret)
        original_rows = d_activation.shape[2]
        padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
        if padded_rows != original_rows:
            row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
            d_activation = jnp.pad(d_activation, (*row_pad, (0, 0)))
            z = jnp.pad(z, (*row_pad, (0, 0)))
            compact_slots = SourcePushDx13SourceCompactSlots(
                source_rank_by_expert=jnp.pad(compact_slots.source_rank_by_expert, row_pad, constant_values=0),
                dst_ordinal_by_expert=jnp.pad(compact_slots.dst_ordinal_by_expert, row_pad, constant_values=0),
                entry_by_expert=jnp.pad(compact_slots.entry_by_expert, row_pad, constant_values=0),
                row_in_entry_by_expert=jnp.pad(compact_slots.row_in_entry_by_expert, row_pad, constant_values=0),
                valid_by_expert=jnp.pad(compact_slots.valid_by_expert, row_pad, constant_values=False),
            )
        _validate_dx13_source_compact_pallas_request(
            d_activation,
            z,
            w13,
            compact_slots,
            queue_shape=queue_shape,
            block_sizes=block_sizes,
        )
        dx_contrib = _source_push_dx13_source_compact_sharded_mgpu_kernel(
            mesh,
            d_activation.astype(w13.dtype),
            z.astype(w13.dtype),
            w13,
            compact_slots.source_rank_by_expert,
            compact_slots.dst_ordinal_by_expert,
            compact_slots.entry_by_expert,
            compact_slots.row_in_entry_by_expert,
            compact_slots.valid_by_expert,
            queue_shape=queue_shape,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            output_block=block_sizes.output_block,
        )
        return SourcePushDx13SourceCompactOutput(dx_contrib=dx_contrib)
    raise ValueError(
        "source-push DX13 contribution implementation must be one of "
        f"{(SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU)}, "
        f"got {implementation!r}"
    )


def source_push_dx13_push_compact_contrib(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    compact_slots: SourcePushDx13SourceCompactSlots,
    *,
    queue_shape: tuple[int, int, int, int],
    implementation: SourcePushDx13Implementation = SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13SourceCompactOutput:
    """Explicit DX13 source-push contribution boundary.

    This is the boundary described by the source-push return experiment: compute
    DX13 on the expert owner and write compact assignment contributions owned by
    the original source rank.  The returned object is not a final ``dx`` tensor
    and not a source route buffer; callers must run a source-local combine.
    """

    return source_push_dx13_push_contrib(
        d_activation,
        z,
        w13,
        compact_slots,
        queue_shape=queue_shape,
        implementation=implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )


def source_push_dx13_push_contrib_block_contiguous_pallas_mgpu(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    compact_slots: SourcePushDx13SourceCompactSlots,
    *,
    queue_shape: tuple[int, int, int, int],
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13SourceCompactOutput:
    """Diagnostic DX13 source-compact push for one full queue entry per row tile.

    This path is intentionally stricter than ``source_push_dx13_push_contrib``:
    it only accepts metadata where each valid expert-major row tile maps to one
    source-compact queue entry with ``row_in_entry == 0..row_block-1``.  That
    lets the kernel issue one contiguous SMEM->GMEM copy per tile instead of one
    remote copy and source-rank switch per row.
    """

    block_sizes = SourcePushDx13PallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_dx13_source_compact_block_contiguous_slots(
        d_activation.shape[:3],
        compact_slots,
        queue_shape,
        row_block=block_sizes.row_block,
    )
    if interpret:
        return source_push_dx13_push_contrib_reference(
            d_activation,
            z,
            w13,
            compact_slots,
            queue_shape=queue_shape,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU block-contiguous DX13 source-compact push requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU block-contiguous DX13 source-compact push requires a mesh")
    _validate_dx13_source_compact_pallas_request(
        d_activation,
        z,
        w13,
        compact_slots,
        queue_shape=queue_shape,
        block_sizes=block_sizes,
    )
    dx_contrib = _source_push_dx13_source_compact_sharded_mgpu_kernel(
        mesh,
        d_activation.astype(w13.dtype),
        z.astype(w13.dtype),
        w13,
        compact_slots.source_rank_by_expert,
        compact_slots.dst_ordinal_by_expert,
        compact_slots.entry_by_expert,
        compact_slots.row_in_entry_by_expert,
        compact_slots.valid_by_expert,
        queue_shape=queue_shape,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
        block_contiguous=True,
    )
    return SourcePushDx13SourceCompactOutput(dx_contrib=dx_contrib)


def source_push_dx13_push_contrib_reference(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    compact_slots: SourcePushDx13SourceCompactSlots,
    *,
    queue_shape: tuple[int, int, int, int],
) -> SourcePushDx13SourceCompactOutput:
    """Readable reference for DX13 push into source compact assignment slots."""

    dx_expert_major = source_push_dx13_expert_major_reference(
        d_activation,
        z,
        w13,
        compact_slots.valid_by_expert,
    )
    return source_push_dx13_contrib_buffer_from_expert_reference(
        dx_expert_major,
        compact_slots,
        queue_shape=queue_shape,
    )


def source_push_dx13_contrib_buffer_from_expert_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    compact_slots: SourcePushDx13SourceCompactSlots,
    *,
    queue_shape: tuple[int, int, int, int],
) -> SourcePushDx13SourceCompactOutput:
    """Place expert-major DX13 rows into source-local compact queue slots."""

    _validate_source_compact_slots(dx_expert_major, compact_slots, queue_shape)
    source_count, dst_count, entries_per_dst, block_m = queue_shape
    valid = compact_slots.valid_by_expert.astype(jnp.bool_)
    safe_src = jnp.where(valid, compact_slots.source_rank_by_expert, 0)
    safe_dst_ord = jnp.where(valid, compact_slots.dst_ordinal_by_expert, 0)
    safe_entry = jnp.where(valid, compact_slots.entry_by_expert, 0)
    safe_row = jnp.where(valid, compact_slots.row_in_entry_by_expert, 0)
    dx_clean = jnp.where(valid[..., None], dx_expert_major, jnp.zeros((), dtype=dx_expert_major.dtype))
    dx_contrib = jnp.zeros(
        (source_count, dst_count, entries_per_dst, block_m, dx_expert_major.shape[-1]),
        dtype=dx_expert_major.dtype,
    )
    dx_contrib = dx_contrib.at[safe_src, safe_dst_ord, safe_entry, safe_row].add(
        dx_clean,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
    )
    return SourcePushDx13SourceCompactOutput(dx_contrib=dx_contrib)


def source_push_dx13_source_compact_to_route_buffer_reference(
    compact_output: SourcePushDx13SourceCompactOutput,
    plan: SourcePushPlan,
) -> SourcePushDx13RouteBufferOutput:
    """Source-local combine from compact assignment slots into token route slots."""

    dx_contrib = compact_output.dx_contrib
    if dx_contrib.shape[:4] != plan.valid_mask.shape:
        raise ValueError(f"dx_contrib queue shape {dx_contrib.shape[:4]} must match plan {plan.valid_mask.shape}")
    safe_token = jnp.maximum(plan.token_ids, 0)
    safe_slot = jnp.maximum(plan.route_slots, 0)
    source_idx = jnp.arange(plan.valid_mask.shape[0], dtype=jnp.int32)[:, None, None, None]
    valid = plan.valid_mask.astype(jnp.bool_)
    dx_clean = jnp.where(valid[..., None], dx_contrib, jnp.zeros((), dtype=dx_contrib.dtype))
    dx_routes = jnp.zeros(
        (plan.valid_mask.shape[0], plan.tokens_per_source, plan.topk, dx_contrib.shape[-1]),
        dtype=dx_contrib.dtype,
    )
    dx_routes = dx_routes.at[source_idx, safe_token, safe_slot].add(
        dx_clean,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )
    return SourcePushDx13RouteBufferOutput(dx_routes=dx_routes, dx=jnp.sum(dx_routes, axis=2))


def source_push_dx13_source_compact_combine_reference(
    compact_output: SourcePushDx13SourceCompactOutput,
    plan: SourcePushPlan,
) -> Float[Array, "S T D"]:
    """Source-local token combine from compact assignment slots.

    This is the production-relevant half of the source-compact return boundary:
    the expert owner has pushed DX13 contributions into the source queue layout,
    and the source rank only needs the final per-token sum.  It deliberately
    skips the intermediate ``[source, token, route_slot, hidden]`` route buffer.
    """

    dx_contrib = compact_output.dx_contrib
    if dx_contrib.shape[:4] != plan.valid_mask.shape:
        raise ValueError(f"dx_contrib queue shape {dx_contrib.shape[:4]} must match plan {plan.valid_mask.shape}")
    source_count = plan.valid_mask.shape[0]
    safe_token = jnp.maximum(plan.token_ids, 0)
    source_idx = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None]
    valid = plan.valid_mask.astype(jnp.bool_)
    dx_clean = jnp.where(valid[..., None], dx_contrib, jnp.zeros((), dtype=dx_contrib.dtype))
    dx = jnp.zeros((source_count, plan.tokens_per_source, dx_contrib.shape[-1]), dtype=dx_contrib.dtype)
    return dx.at[source_idx, safe_token].add(
        dx_clean,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def source_push_dx13_push_compact(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    implementation: SourcePushDx13Implementation = SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13CompactOutput:
    """Compute compact DX13 contributions aligned with expert-major route rows.

    ``pallas_mgpu`` covers the local expert-owner diagnostic: recompute dSwiGLU
    from ``d_activation``/``z`` and multiply by ``W13.T`` into compact
    expert-major rows.  It deliberately returns compact rows plus source-return
    metadata; the remote-write route-buffer epilogue and source-local combine
    stay separate so this path does not write atomically into final ``x.grad``.
    """

    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE:
        return source_push_dx13_push_compact_reference(
            d_activation,
            z,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
        )
    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU:
        return source_push_dx13_push_compact_pallas_mgpu(
            d_activation,
            z,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    raise ValueError(
        "source-push DX13 implementation must be one of "
        f"{(SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU)}, "
        f"got {implementation!r}"
    )


def source_push_dx13_push_compact_pallas_mgpu(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13CompactOutput:
    """Pallas/MGPU local expert-owner DX13 diagnostic with compact metadata."""

    dx_expert_major = source_push_dx13_expert_major_pallas_mgpu(
        d_activation,
        z,
        w13,
        valid_by_expert,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    return SourcePushDx13CompactOutput(
        dx_expert_major=dx_expert_major,
        source_rank_by_expert=source_rank_by_expert,
        token_id_by_expert=token_id_by_expert,
        route_slot_by_expert=route_slot_by_expert,
        valid_by_expert=valid_by_expert,
    )


def source_push_dx13_push_compact_reference(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> SourcePushDx13CompactOutput:
    """Readable reference for the first compact DX13 diagnostic slice."""

    dx_expert_major = source_push_dx13_expert_major_reference(d_activation, z, w13, valid_by_expert)
    return SourcePushDx13CompactOutput(
        dx_expert_major=dx_expert_major,
        source_rank_by_expert=source_rank_by_expert,
        token_id_by_expert=token_id_by_expert,
        route_slot_by_expert=route_slot_by_expert,
        valid_by_expert=valid_by_expert,
    )


def source_push_dx13_push_route_buffer(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
    implementation: SourcePushDx13Implementation = SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13RouteBufferOutput:
    """Compute DX13 contributions into the source-owned route buffer.

    This is the desired backward boundary for DX13 source-push: expert-owner
    work recomputes dSwiGLU from ``d_activation`` and saved W13 preactivation
    ``z`` while multiplying by ``W13.T``, then each contribution lands in its
    explicit source route slot ``[source_rank, token_id, route_slot]``. The
    returned ``dx`` is the source-local top-k sum over ``dx_routes``.
    """

    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE:
        return source_push_dx13_push_route_buffer_reference(
            d_activation,
            z,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            tokens_per_source=tokens_per_source,
            topk=topk,
        )
    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU:
        return source_push_dx13_push_route_buffer_pallas_mgpu(
            d_activation,
            z,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            tokens_per_source=tokens_per_source,
            topk=topk,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    raise ValueError(
        "source-push DX13 route-buffer implementation must be one of "
        f"{(SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU)}, "
        f"got {implementation!r}"
    )


def source_push_dx13_push_route_buffer_reference(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
) -> SourcePushDx13RouteBufferOutput:
    """Readable DX13 source route-buffer reference.

    The implementation keeps ``dZ`` as an internal conceptual value only:
    callers pass ``d_activation`` and saved ``z``, and the SwiGLU derivative is
    recomputed at the DX13 boundary.
    """

    compact_output = source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
    )
    return source_push_dx13_route_buffer_epilogue_reference(
        compact_output,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )


def source_push_dx13_push_route_buffer_pallas_mgpu(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13RouteBufferOutput:
    """Target-shaped MGPU DX13 route-buffer entry point.

    The real kernel grid is ``(dst, expert, row_tile, hidden_tile)``. Each
    program computes one compact contribution tile, recomputes dSwiGLU from
    ``d_activation``/``z`` on load, multiplies by ``W13.T``, and remote-stores
    the resulting ``[row_tile, hidden_tile]`` into the source-owned
    ``dx_routes[source_rank, token_id, route_slot, hidden_tile]`` buffer.
    """

    if interpret:
        return source_push_dx13_push_route_buffer_reference(
            d_activation,
            z,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            tokens_per_source=tokens_per_source,
            topk=topk,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU DX13 route-buffer push requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU DX13 route-buffer push requires a mesh")

    block_sizes = source_push_dx13_pallas_resolved_block_sizes(block_sizes, interpret=interpret)
    _validate_dx13_route_buffer_shapes(
        d_activation,
        z,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )
    original_rows = d_activation.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        d_activation = jnp.pad(d_activation, (*row_pad, (0, 0)))
        z = jnp.pad(z, (*row_pad, (0, 0)))
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        route_slot_by_expert = jnp.pad(route_slot_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    _validate_dx13_route_buffer_pallas_request(
        d_activation,
        z,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
        tokens_per_source=tokens_per_source,
        topk=topk,
        block_sizes=block_sizes,
    )
    dx_routes = _source_push_dx13_route_buffer_sharded_mgpu_kernel(
        mesh,
        d_activation.astype(w13.dtype),
        z.astype(w13.dtype),
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
        tokens_per_source=tokens_per_source,
        topk=topk,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )
    route_slot_valid = _source_push_dx13_route_slot_valid_mask(
        source_rank_by_expert[:, :, :original_rows],
        token_id_by_expert[:, :, :original_rows],
        route_slot_by_expert[:, :, :original_rows],
        valid_by_expert[:, :, :original_rows],
        tokens_per_source=tokens_per_source,
        topk=topk,
    )
    route_slot_valid = _with_source_push_source_sharding(route_slot_valid, mesh=mesh)
    dx_routes = jnp.where(route_slot_valid[..., None], dx_routes, jnp.zeros((), dtype=dx_routes.dtype))
    return SourcePushDx13RouteBufferOutput(dx_routes=dx_routes, dx=jnp.sum(dx_routes, axis=2))


def source_push_dx13_expert_major_reference(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C D"]:
    """Compute ``dX = dZ @ W13.T`` from saved W13 preactivation rows."""

    _validate_dx13_shapes(d_activation, z, w13, valid_by_expert)
    d_z = source_push_dx13_dz_from_swiglu_reference(d_activation, z, valid_by_expert)
    dx = jnp.einsum("deco,deho->dech", d_z, w13.astype(jnp.float32))
    return dx * valid_by_expert.astype(dx.dtype)[..., None]


def source_push_dx13_expert_major_xla(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C D"]:
    """Benchmark-only XLA floor for compact expert-major DX13 math."""

    return source_push_dx13_expert_major_reference(d_activation, z, w13, valid_by_expert)


def source_push_dx13_push_compact_xla(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> SourcePushDx13CompactOutput:
    """Benchmark-only XLA compact DX13 floor with source-return metadata."""

    dx_expert_major = source_push_dx13_expert_major_xla(d_activation, z, w13, valid_by_expert)
    return SourcePushDx13CompactOutput(
        dx_expert_major=dx_expert_major,
        source_rank_by_expert=source_rank_by_expert,
        token_id_by_expert=token_id_by_expert,
        route_slot_by_expert=route_slot_by_expert,
        valid_by_expert=valid_by_expert,
    )


def source_push_dx13_expert_major_pallas_mgpu(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst E C D"]:
    """Compute compact DX13 rows with the local expert-owner WGMMA diagnostic."""

    block_sizes = source_push_dx13_pallas_resolved_block_sizes(block_sizes, interpret=interpret)
    _validate_dx13_shapes(d_activation, z, w13, valid_by_expert)
    if interpret:
        return source_push_dx13_expert_major_reference(d_activation, z, w13, valid_by_expert)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU DX13 expert-major diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU DX13 expert-major diagnostic requires a mesh")
    original_rows = d_activation.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        d_activation = jnp.pad(d_activation, (*row_pad, (0, 0)))
        z = jnp.pad(z, (*row_pad, (0, 0)))
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    _validate_dx13_expert_major_pallas_request(d_activation, z, w13, valid_by_expert, block_sizes)
    dx_expert_major = _source_push_dx13_expert_major_sharded_mgpu_kernel(
        mesh,
        d_activation.astype(w13.dtype),
        z.astype(w13.dtype),
        w13,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )
    return _with_source_push_destination_sharding(
        dx_expert_major[:, :, :original_rows, :],
        mesh=mesh,
        like=d_activation,
    )


def source_push_dx13_expert_major_store_zero_pallas_mgpu(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst E C D"]:
    """Diagnostic: write a zero DX13 expert-major output with the Pallas grid."""

    block_sizes = source_push_dx13_pallas_resolved_block_sizes(block_sizes, interpret=interpret)
    _validate_dx13_shapes(d_activation, z, w13, valid_by_expert)
    if interpret:
        return jnp.zeros((*d_activation.shape[:3], w13.shape[-2]), dtype=jnp.float32)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU DX13 store-zero diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU DX13 store-zero diagnostic requires a mesh")
    original_rows = d_activation.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        d_activation = jnp.pad(d_activation, (*row_pad, (0, 0)))

    if d_activation.shape[2] % block_sizes.row_block:
        raise ValueError("DX13 store-zero rows must be divisible by row_block after padding")
    if w13.shape[-2] % block_sizes.hidden_block:
        raise ValueError(
            f"DX13 hidden_dim {w13.shape[-2]} must be divisible by hidden_block {block_sizes.hidden_block}"
        )
    dx_expert_major = _source_push_dx13_expert_major_store_zero_sharded_mgpu_kernel(
        mesh,
        d_activation,
        hidden_dim=w13.shape[-2],
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )
    return _with_source_push_destination_sharding(
        dx_expert_major[:, :, :original_rows, :],
        mesh=mesh,
        like=d_activation,
    )


def source_push_dx13_dz_from_swiglu_reference(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C twoI"]:
    """Recompute ``dZ=[dgate, dup]`` from ``d_activation`` and saved ``z``."""

    _validate_swiglu_shapes(d_activation, z, valid_by_expert)
    intermediate_dim = d_activation.shape[-1]
    valid_f = valid_by_expert.astype(jnp.float32)
    gate = z[..., :intermediate_dim].astype(jnp.float32) * valid_f[..., None]
    up = z[..., intermediate_dim:].astype(jnp.float32) * valid_f[..., None]
    d_activation = d_activation.astype(jnp.float32) * valid_f[..., None]

    silu_gate = jax.nn.silu(gate)
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_gate = d_activation * up * d_silu_gate
    d_up = d_activation * silu_gate
    return jnp.concatenate([d_gate, d_up], axis=-1) * valid_f[..., None]


def source_push_dx13_route_buffer_epilogue_reference(
    compact_output: SourcePushDx13CompactOutput,
    *,
    tokens_per_source: int,
    topk: int,
) -> SourcePushDx13RouteBufferOutput:
    """Model the future source-push route-buffer epilogue for compact DX13 rows.

    The epilogue writes each expert-major contribution into its explicit source
    route slot ``[source_rank, token_id, route_slot]`` and then performs the
    source-local top-k reduction. This is a correctness contract, not a remote
    atomic add into final ``x.grad``.
    """

    return source_push_dx13_route_buffer_epilogue_from_fields_reference(
        compact_output.dx_expert_major,
        compact_output.source_rank_by_expert,
        compact_output.token_id_by_expert,
        compact_output.route_slot_by_expert,
        compact_output.valid_by_expert,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )


def source_push_dx13_route_buffer_epilogue_from_fields_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
) -> SourcePushDx13RouteBufferOutput:
    """Build the source-owned route buffer and source-local combined DX."""

    dx_routes = source_push_dx13_source_route_buffer_reference(
        dx_expert_major,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )
    return SourcePushDx13RouteBufferOutput(dx_routes=dx_routes, dx=jnp.sum(dx_routes, axis=2))


def source_push_dx13_push_source_grouped_reference(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
) -> SourcePushDx13SourceGroupedOutput:
    """Compute DX13 contributions and group them into contiguous source chunks.

    This keeps the DX13 producer boundary compact but avoids direct token-slot
    scatter from the expert owner.  A production Pallas source-push epilogue
    should write this grouped buffer back to each source rank, then let a
    source-local combine consume the token/slot metadata.
    """

    compact_output = source_push_dx13_push_compact_reference(
        d_activation,
        z,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
    )
    return source_push_dx13_source_grouped_from_fields_reference(
        compact_output.dx_expert_major,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
        src_base_by_expert,
    )


def source_push_dx13_source_grouped_from_fields(
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    implementation: SourcePushDx13Implementation = SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13SourceGroupedOutput:
    """Pack compact DX13 rows into the source-grouped return boundary."""

    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE:
        return source_push_dx13_source_grouped_from_fields_reference(
            dx_expert_major,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            src_base_by_expert,
        )
    if implementation == SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU:
        return source_push_dx13_source_grouped_from_fields_pallas_mgpu(
            dx_expert_major,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            src_base_by_expert,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    raise ValueError(
        "source-push DX13 source-grouped implementation must be one of "
        f"{(SOURCE_PUSH_DX13_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_DX13_IMPLEMENTATION_PALLAS_MGPU)}, "
        f"got {implementation!r}"
    )


def source_push_dx13_source_grouped_from_fields_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
) -> SourcePushDx13SourceGroupedOutput:
    """Pack expert-major DX rows into ``[dst, source, expert, source_row]``."""

    _validate_compact_metadata(dx_expert_major, source_rank_by_expert, token_id_by_expert, valid_by_expert)
    if route_slot_by_expert.shape != valid_by_expert.shape:
        raise ValueError(
            f"route_slot_by_expert shape {route_slot_by_expert.shape} must match valid shape {valid_by_expert.shape}"
        )
    src_base = jnp.asarray(src_base_by_expert, dtype=jnp.int32)
    dst_count, local_experts, expert_capacity = valid_by_expert.shape
    source_count = source_rank_by_expert.shape[0]
    if src_base.shape != (dst_count, source_count, local_experts):
        raise ValueError(
            f"src_base_by_expert shape {src_base.shape} must be {(dst_count, source_count, local_experts)}"
        )

    max_source_rows = _dx13_max_source_group_rows(
        source_rank_by_expert,
        valid_by_expert,
        src_base,
    )
    out_shape = (dst_count, source_count, local_experts, max_source_rows)
    if max_source_rows == 0:
        return SourcePushDx13SourceGroupedOutput(
            dx_by_source=jnp.zeros((*out_shape, dx_expert_major.shape[-1]), dtype=dx_expert_major.dtype),
            token_id_by_source=jnp.zeros(out_shape, dtype=jnp.int32),
            route_slot_by_source=jnp.zeros(out_shape, dtype=jnp.int32),
            valid_by_source=jnp.zeros(out_shape, dtype=jnp.bool_),
        )

    dst_idx = jnp.broadcast_to(jnp.arange(dst_count, dtype=jnp.int32)[:, None, None], valid_by_expert.shape)
    expert_idx = jnp.broadcast_to(jnp.arange(local_experts, dtype=jnp.int32)[None, :, None], valid_by_expert.shape)
    row_idx = jnp.broadcast_to(jnp.arange(expert_capacity, dtype=jnp.int32)[None, None, :], valid_by_expert.shape)
    valid = valid_by_expert.astype(jnp.bool_)
    safe_src = jnp.where(valid, source_rank_by_expert, 0)
    source_base = src_base.at[dst_idx, safe_src, expert_idx].get()
    source_row = jnp.where(valid, row_idx - source_base, 0)
    dx_clean = jnp.where(valid[..., None], dx_expert_major, jnp.zeros((), dtype=dx_expert_major.dtype))
    valid_i = valid.astype(jnp.int32)

    dx_by_source = jnp.zeros((*out_shape, dx_expert_major.shape[-1]), dtype=dx_expert_major.dtype)
    token_by_source = jnp.zeros(out_shape, dtype=jnp.int32)
    slot_by_source = jnp.zeros(out_shape, dtype=jnp.int32)
    valid_by_source = jnp.zeros(out_shape, dtype=jnp.int32)
    dx_by_source = dx_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(dx_clean)
    token_by_source = token_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(
        jnp.where(valid, token_id_by_expert, 0)
    )
    slot_by_source = slot_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(
        jnp.where(valid, route_slot_by_expert, 0)
    )
    valid_by_source = valid_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(valid_i)
    return SourcePushDx13SourceGroupedOutput(
        dx_by_source=dx_by_source,
        token_id_by_source=token_by_source,
        route_slot_by_source=slot_by_source,
        valid_by_source=valid_by_source > 0,
    )


def source_push_dx13_source_grouped_from_fields_pallas_mgpu(
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    block_sizes: SourcePushDx13PallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushDx13SourceGroupedOutput:
    """Pallas/MGPU source-grouped copy epilogue for already-computed DX13 rows."""

    if interpret:
        return source_push_dx13_source_grouped_from_fields_reference(
            dx_expert_major,
            source_rank_by_expert,
            token_id_by_expert,
            route_slot_by_expert,
            valid_by_expert,
            src_base_by_expert,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU DX13 source-grouped epilogue requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU DX13 source-grouped epilogue requires a mesh")

    block_sizes = source_push_dx13_pallas_resolved_block_sizes(block_sizes, interpret=interpret)
    src_base = jnp.asarray(src_base_by_expert, dtype=jnp.int32)
    _validate_dx13_source_grouped_fields_pallas_request(
        dx_expert_major,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid_by_expert,
        src_base,
        block_sizes,
    )
    source_rows = _dx13_max_source_group_rows(source_rank_by_expert, valid_by_expert, src_base)
    if source_rows == 0:
        dst_count, local_experts, _expert_capacity = valid_by_expert.shape
        source_count = source_rank_by_expert.shape[0]
        out_shape = (dst_count, source_count, local_experts, 0)
        return SourcePushDx13SourceGroupedOutput(
            dx_by_source=jnp.zeros((*out_shape, dx_expert_major.shape[-1]), dtype=dx_expert_major.dtype),
            token_id_by_source=jnp.zeros(out_shape, dtype=jnp.int32),
            route_slot_by_source=jnp.zeros(out_shape, dtype=jnp.int32),
            valid_by_source=jnp.zeros(out_shape, dtype=jnp.bool_),
        )

    original_rows = dx_expert_major.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        dx_expert_major = jnp.pad(dx_expert_major, (*row_pad, (0, 0)))
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        route_slot_by_expert = jnp.pad(route_slot_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    metadata = _source_grouped_metadata_from_fields_reference(
        source_rank_by_expert[:, :, :original_rows],
        token_id_by_expert[:, :, :original_rows],
        route_slot_by_expert[:, :, :original_rows],
        valid_by_expert[:, :, :original_rows],
        src_base,
        source_rows=source_rows,
    )
    dx_by_source_by_src = _source_push_dx13_source_grouped_sharded_mgpu_kernel(
        mesh,
        dx_expert_major,
        source_rank_by_expert,
        valid_by_expert,
        src_base,
        source_rows=source_rows,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )
    dx_by_source = jnp.transpose(dx_by_source_by_src, (1, 0, 2, 3, 4))
    return SourcePushDx13SourceGroupedOutput(
        dx_by_source=dx_by_source,
        token_id_by_source=metadata.token_id_by_source,
        route_slot_by_source=metadata.route_slot_by_source,
        valid_by_source=metadata.valid_by_source,
    )


def source_push_dx13_source_grouped_to_route_buffer_reference(
    grouped_output: SourcePushDx13SourceGroupedOutput,
    *,
    tokens_per_source: int,
    topk: int,
) -> SourcePushDx13RouteBufferOutput:
    """Source-local token/slot combine for grouped DX13 contributions."""

    if tokens_per_source <= 0:
        raise ValueError(f"tokens_per_source must be positive, got {tokens_per_source}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    dx_by_source = grouped_output.dx_by_source
    valid = grouped_output.valid_by_source.astype(jnp.bool_)
    if grouped_output.token_id_by_source.shape != valid.shape:
        raise ValueError("token_id_by_source must match valid_by_source shape")
    if grouped_output.route_slot_by_source.shape != valid.shape:
        raise ValueError("route_slot_by_source must match valid_by_source shape")

    dst_count, source_count, local_experts, source_rows = valid.shape
    src_idx = jnp.broadcast_to(jnp.arange(source_count, dtype=jnp.int32)[None, :, None, None], valid.shape)
    safe_src = jnp.where(valid, src_idx, 0)
    safe_token = jnp.where(valid, grouped_output.token_id_by_source, 0)
    safe_slot = jnp.where(valid, grouped_output.route_slot_by_source, 0)
    dx_clean = jnp.where(valid[..., None], dx_by_source, jnp.zeros((), dtype=dx_by_source.dtype))
    dx_routes = jnp.zeros((source_count, tokens_per_source, topk, dx_by_source.shape[-1]), dtype=dx_by_source.dtype)
    dx_routes = dx_routes.at[safe_src, safe_token, safe_slot].add(
        dx_clean,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )
    _ = (dst_count, local_experts, source_rows)
    return SourcePushDx13RouteBufferOutput(dx_routes=dx_routes, dx=jnp.sum(dx_routes, axis=2))


def source_push_dx13_source_route_buffer_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
) -> Float[Array, "S T K D"]:
    """Place compact expert-major DX13 rows in source-owned route slots."""

    _validate_compact_metadata(dx_expert_major, source_rank_by_expert, token_id_by_expert, valid_by_expert)
    if route_slot_by_expert.shape != valid_by_expert.shape:
        raise ValueError(
            f"route_slot_by_expert shape {route_slot_by_expert.shape} must match valid shape {valid_by_expert.shape}"
        )
    if tokens_per_source <= 0:
        raise ValueError(f"tokens_per_source must be positive, got {tokens_per_source}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    source_count = source_rank_by_expert.shape[0]
    dx_routes = jnp.zeros((source_count, tokens_per_source, topk, dx_expert_major.shape[-1]), dx_expert_major.dtype)
    valid = valid_by_expert.astype(jnp.bool_)
    safe_src = jnp.where(valid, source_rank_by_expert, 0)
    safe_token = jnp.where(valid, token_id_by_expert, 0)
    safe_slot = jnp.where(valid, route_slot_by_expert, 0)
    dx_clean = jnp.where(valid[..., None], dx_expert_major, jnp.zeros((), dtype=dx_expert_major.dtype))
    return dx_routes.at[safe_src, safe_token, safe_slot].add(
        dx_clean,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )


def source_push_dx13_combine_source_tokens_reference(
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
) -> Float[Array, "S T D"]:
    """Source-local combine for compact DX13 contributions.

    This models the post-push source-side reduction.  It is not a production
    remote atomic path into final ``x.grad``.
    """

    _validate_compact_metadata(dx_expert_major, source_rank_by_expert, token_id_by_expert, valid_by_expert)
    if tokens_per_source <= 0:
        raise ValueError(f"tokens_per_source must be positive, got {tokens_per_source}")

    source_count = source_rank_by_expert.shape[0]
    dx = jnp.zeros((source_count, tokens_per_source, dx_expert_major.shape[-1]), dx_expert_major.dtype)
    valid = valid_by_expert.astype(jnp.bool_)
    safe_src = jnp.where(valid, source_rank_by_expert, 0)
    safe_token = jnp.where(valid, token_id_by_expert, 0)
    dx_clean = jnp.where(valid[..., None], dx_expert_major, jnp.zeros((), dtype=dx_expert_major.dtype))
    return dx.at[safe_src, safe_token].add(
        dx_clean,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def _dx13_max_source_group_rows(
    source_rank_by_expert: Array,
    valid_by_expert: Array,
    src_base_by_expert: Array,
) -> int:
    source_rank_host = np.asarray(jax.device_get(source_rank_by_expert), dtype=np.int32)
    valid_host = np.asarray(jax.device_get(valid_by_expert), dtype=np.bool_)
    src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)
    max_rows = 0
    dst_count, local_experts, expert_capacity = valid_host.shape
    for dst in range(dst_count):
        for expert in range(local_experts):
            for row in range(expert_capacity):
                if not valid_host[dst, expert, row]:
                    continue
                src = int(source_rank_host[dst, expert, row])
                max_rows = max(max_rows, row - int(src_base_host[dst, src, expert]) + 1)
    return max_rows


def _source_push_dx13_expert_major_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst E C D"]:
    kernel = _make_source_push_dx13_expert_major_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_activation.shape[1],
        rows=d_activation.shape[2],
        hidden_dim=w13.shape[-2],
        intermediate_dim=d_activation.shape[-1],
    )
    d_activation_spec = _source_push_destination_or_replicated_spec(d_activation, 4)
    z_spec = _source_push_destination_or_replicated_spec(z, 4)
    w13_spec = _source_push_destination_or_replicated_spec(w13, 4)
    valid_spec = _source_push_destination_or_replicated_spec(valid, 3)
    destination_spec_4d = P(SOURCE_PUSH_MESH_AXIS, None, None, None)
    destination_spec_3d = P(SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        d_activation_arg: Float[Array, "Dst E C I"],
        z_arg: Float[Array, "Dst E C twoI"],
        w13_arg: Float[Array, "Dst E D twoI"],
        valid_arg: Bool[Array, "Dst E C"],
    ) -> Float[Array, "Dst E C D"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if d_activation_spec == destination_spec_4d:
            d_activation_local = d_activation_arg
        else:
            d_activation_local = lax.dynamic_slice_in_dim(d_activation_arg, dst, 1, axis=0)
        if z_spec == destination_spec_4d:
            z_local = z_arg
        else:
            z_local = lax.dynamic_slice_in_dim(z_arg, dst, 1, axis=0)
        if w13_spec == destination_spec_4d:
            w13_local = w13_arg
        else:
            w13_local = lax.dynamic_slice_in_dim(w13_arg, dst, 1, axis=0)
        if valid_spec == destination_spec_3d:
            valid_local = valid_arg
        else:
            valid_local = lax.dynamic_slice_in_dim(valid_arg, dst, 1, axis=0)
        return kernel(
            d_activation_local[0],
            z_local[0],
            w13_local[0],
            valid_local[0].astype(jnp.int32),
        )[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(d_activation_spec, z_spec, w13_spec, valid_spec),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(d_activation, z, w13, valid)


def _source_push_dx13_expert_major_store_zero_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    d_activation: Float[Array, "Dst E C I"],
    *,
    hidden_dim: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C D"]:
    kernel = _make_source_push_dx13_expert_major_store_zero_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        experts_per_rank=d_activation.shape[1],
        rows=d_activation.shape[2],
        hidden_dim=hidden_dim,
    )
    d_activation_spec = _source_push_destination_or_replicated_spec(d_activation, 4)
    destination_spec_4d = P(SOURCE_PUSH_MESH_AXIS, None, None, None)

    def local_fn(d_activation_arg: Float[Array, "Dst E C I"]) -> Float[Array, "Dst E C D"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if d_activation_spec == destination_spec_4d:
            d_activation_local = d_activation_arg
        else:
            d_activation_local = lax.dynamic_slice_in_dim(d_activation_arg, dst, 1, axis=0)
        return kernel(d_activation_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=d_activation_spec,
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(d_activation)


def _source_push_dx13_route_buffer_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "S T K D"]:
    kernel = _make_source_push_dx13_route_buffer_mgpu_kernel(
        source_count=d_activation.shape[0],
        tokens_per_source=tokens_per_source,
        topk=topk,
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_activation.shape[1],
        rows=d_activation.shape[2],
        hidden_dim=w13.shape[-2],
        intermediate_dim=d_activation.shape[-1],
    )
    d_activation_spec = _source_push_destination_or_replicated_spec(d_activation, 4)
    z_spec = _source_push_destination_or_replicated_spec(z, 4)
    w13_spec = _source_push_destination_or_replicated_spec(w13, 4)
    source_rank_spec = _source_push_destination_or_replicated_spec(source_rank_by_expert, 3)
    token_id_spec = _source_push_destination_or_replicated_spec(token_id_by_expert, 3)
    route_slot_spec = _source_push_destination_or_replicated_spec(route_slot_by_expert, 3)
    valid_spec = _source_push_destination_or_replicated_spec(valid, 3)
    destination_spec_4d = P(SOURCE_PUSH_MESH_AXIS, None, None, None)
    destination_spec_3d = P(SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        d_activation_arg: Float[Array, "Dst E C I"],
        z_arg: Float[Array, "Dst E C twoI"],
        w13_arg: Float[Array, "Dst E D twoI"],
        source_rank_arg: Int[Array, "Dst E C"],
        token_id_arg: Int[Array, "Dst E C"],
        route_slot_arg: Int[Array, "Dst E C"],
        valid_arg: Bool[Array, "Dst E C"],
    ) -> Float[Array, "1 T K D"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if d_activation_spec == destination_spec_4d:
            d_activation_local = d_activation_arg
        else:
            d_activation_local = lax.dynamic_slice_in_dim(d_activation_arg, dst, 1, axis=0)
        if z_spec == destination_spec_4d:
            z_local = z_arg
        else:
            z_local = lax.dynamic_slice_in_dim(z_arg, dst, 1, axis=0)
        if w13_spec == destination_spec_4d:
            w13_local = w13_arg
        else:
            w13_local = lax.dynamic_slice_in_dim(w13_arg, dst, 1, axis=0)
        if source_rank_spec == destination_spec_3d:
            source_rank_local = source_rank_arg
        else:
            source_rank_local = lax.dynamic_slice_in_dim(source_rank_arg, dst, 1, axis=0)
        if token_id_spec == destination_spec_3d:
            token_id_local = token_id_arg
        else:
            token_id_local = lax.dynamic_slice_in_dim(token_id_arg, dst, 1, axis=0)
        if route_slot_spec == destination_spec_3d:
            route_slot_local = route_slot_arg
        else:
            route_slot_local = lax.dynamic_slice_in_dim(route_slot_arg, dst, 1, axis=0)
        if valid_spec == destination_spec_3d:
            valid_local = valid_arg
        else:
            valid_local = lax.dynamic_slice_in_dim(valid_arg, dst, 1, axis=0)
        dx_routes = kernel(
            d_activation_local[0],
            z_local[0],
            w13_local[0],
            source_rank_local[0],
            token_id_local[0],
            route_slot_local[0],
            valid_local[0].astype(jnp.int32),
        )
        return dx_routes[None, ...]

    dx_routes = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            d_activation_spec,
            z_spec,
            w13_spec,
            source_rank_spec,
            token_id_spec,
            route_slot_spec,
            valid_spec,
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(d_activation, z, w13, source_rank_by_expert, token_id_by_expert, route_slot_by_expert, valid)
    return _sharded_dx13_route_buffer_remote_write_completion_barrier(mesh)(dx_routes)


def _source_push_dx13_source_compact_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    dst_ordinal_by_expert: Int[Array, "Dst E C"],
    entry_by_expert: Int[Array, "Dst E C"],
    row_in_entry_by_expert: Int[Array, "Dst E C"],
    valid: Bool[Array, "Dst E C"],
    *,
    queue_shape: tuple[int, int, int, int],
    row_block: int,
    hidden_block: int,
    output_block: int,
    block_contiguous: bool = False,
) -> Float[Array, "S Dst Q M D"]:
    source_count, dst_count, entries_per_dst, block_m = queue_shape
    kernel = _make_source_push_dx13_source_compact_mgpu_kernel(
        source_count=source_count,
        dst_count=dst_count,
        entries_per_dst=entries_per_dst,
        block_m=block_m,
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_activation.shape[1],
        rows=d_activation.shape[2],
        hidden_dim=w13.shape[-2],
        intermediate_dim=d_activation.shape[-1],
        block_contiguous=block_contiguous,
    )
    d_activation_spec = _source_push_destination_or_replicated_spec(d_activation, 4)
    z_spec = _source_push_destination_or_replicated_spec(z, 4)
    w13_spec = _source_push_destination_or_replicated_spec(w13, 4)
    source_rank_spec = _source_push_destination_or_replicated_spec(source_rank_by_expert, 3)
    dst_ordinal_spec = _source_push_destination_or_replicated_spec(dst_ordinal_by_expert, 3)
    entry_spec = _source_push_destination_or_replicated_spec(entry_by_expert, 3)
    row_in_entry_spec = _source_push_destination_or_replicated_spec(row_in_entry_by_expert, 3)
    valid_spec = _source_push_destination_or_replicated_spec(valid, 3)
    destination_spec_4d = P(SOURCE_PUSH_MESH_AXIS, None, None, None)
    destination_spec_3d = P(SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        d_activation_arg: Float[Array, "Dst E C I"],
        z_arg: Float[Array, "Dst E C twoI"],
        w13_arg: Float[Array, "Dst E D twoI"],
        source_rank_arg: Int[Array, "Dst E C"],
        dst_ordinal_arg: Int[Array, "Dst E C"],
        entry_arg: Int[Array, "Dst E C"],
        row_in_entry_arg: Int[Array, "Dst E C"],
        valid_arg: Bool[Array, "Dst E C"],
    ) -> Float[Array, "1 Dst Q M D"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if d_activation_spec == destination_spec_4d:
            d_activation_local = d_activation_arg
        else:
            d_activation_local = lax.dynamic_slice_in_dim(d_activation_arg, dst, 1, axis=0)
        if z_spec == destination_spec_4d:
            z_local = z_arg
        else:
            z_local = lax.dynamic_slice_in_dim(z_arg, dst, 1, axis=0)
        if w13_spec == destination_spec_4d:
            w13_local = w13_arg
        else:
            w13_local = lax.dynamic_slice_in_dim(w13_arg, dst, 1, axis=0)
        if source_rank_spec == destination_spec_3d:
            source_rank_local = source_rank_arg
        else:
            source_rank_local = lax.dynamic_slice_in_dim(source_rank_arg, dst, 1, axis=0)
        if dst_ordinal_spec == destination_spec_3d:
            dst_ordinal_local = dst_ordinal_arg
        else:
            dst_ordinal_local = lax.dynamic_slice_in_dim(dst_ordinal_arg, dst, 1, axis=0)
        if entry_spec == destination_spec_3d:
            entry_local = entry_arg
        else:
            entry_local = lax.dynamic_slice_in_dim(entry_arg, dst, 1, axis=0)
        if row_in_entry_spec == destination_spec_3d:
            row_in_entry_local = row_in_entry_arg
        else:
            row_in_entry_local = lax.dynamic_slice_in_dim(row_in_entry_arg, dst, 1, axis=0)
        if valid_spec == destination_spec_3d:
            valid_local = valid_arg
        else:
            valid_local = lax.dynamic_slice_in_dim(valid_arg, dst, 1, axis=0)
        dx_contrib = kernel(
            d_activation_local[0],
            z_local[0],
            w13_local[0],
            source_rank_local[0],
            dst_ordinal_local[0],
            entry_local[0],
            row_in_entry_local[0],
            valid_local[0].astype(jnp.int32),
        )
        return dx_contrib[None, ...]

    dx_contrib = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            d_activation_spec,
            z_spec,
            w13_spec,
            source_rank_spec,
            dst_ordinal_spec,
            entry_spec,
            row_in_entry_spec,
            valid_spec,
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(
        d_activation,
        z,
        w13,
        source_rank_by_expert,
        dst_ordinal_by_expert,
        entry_by_expert,
        row_in_entry_by_expert,
        valid,
    )
    return _sharded_dx13_source_compact_remote_write_completion_barrier(mesh)(dx_contrib)


def _source_push_dx13_source_grouped_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    dx_expert_major: Float[Array, "Dst E C D"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    valid: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    source_rows: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "S Dst E Csrc D"]:
    kernel = _make_source_push_dx13_source_grouped_mgpu_kernel(
        source_count=dx_expert_major.shape[0],
        row_block=row_block,
        hidden_block=hidden_block,
        experts_per_rank=dx_expert_major.shape[1],
        rows=dx_expert_major.shape[2],
        source_rows=source_rows,
        hidden_dim=dx_expert_major.shape[-1],
        output_dtype=dx_expert_major.dtype,
    )
    dx_spec = _source_push_destination_or_replicated_spec(dx_expert_major, 4)
    source_rank_spec = _source_push_destination_or_replicated_spec(source_rank_by_expert, 3)
    valid_spec = _source_push_destination_or_replicated_spec(valid, 3)
    src_base_spec = _source_push_destination_or_replicated_spec(src_base_by_expert, 3)
    destination_spec_4d = P(SOURCE_PUSH_MESH_AXIS, None, None, None)
    destination_spec_3d = P(SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        dx_arg: Float[Array, "Dst E C D"],
        source_rank_arg: Int[Array, "Dst E C"],
        valid_arg: Bool[Array, "Dst E C"],
        src_base_arg: Int[Array, "Dst S E"],
    ) -> Float[Array, "1 Dst E Csrc D"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if dx_spec == destination_spec_4d:
            dx_local = dx_arg
        else:
            dx_local = lax.dynamic_slice_in_dim(dx_arg, dst, 1, axis=0)
        if source_rank_spec == destination_spec_3d:
            source_rank_local = source_rank_arg
        else:
            source_rank_local = lax.dynamic_slice_in_dim(source_rank_arg, dst, 1, axis=0)
        if valid_spec == destination_spec_3d:
            valid_local = valid_arg
        else:
            valid_local = lax.dynamic_slice_in_dim(valid_arg, dst, 1, axis=0)
        if src_base_spec == destination_spec_3d:
            src_base_local = src_base_arg
        else:
            src_base_local = lax.dynamic_slice_in_dim(src_base_arg, dst, 1, axis=0)
        dx_by_source_local = kernel(
            dx_local[0],
            source_rank_local[0],
            valid_local[0].astype(jnp.int32),
            src_base_local[0],
        )
        return dx_by_source_local[None, ...]

    dx_by_source = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(dx_spec, source_rank_spec, valid_spec, src_base_spec),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )(dx_expert_major, source_rank_by_expert, valid, src_base_by_expert)
    return _sharded_dx13_source_grouped_remote_write_completion_barrier(mesh)(dx_by_source)


def _make_source_push_dx13_expert_major_mgpu_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
    intermediate_dim: int,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = intermediate_dim // output_block

    def body(
        d_activation_ref: Float[pl.Ref, "E C I"],
        z_ref: Float[pl.Ref, "E C twoI"],
        w13_ref: Float[pl.Ref, "E D twoI"],
        valid_ref: Int[pl.Ref, "E C"],
        dx_ref: Float[pl.Ref, "E C D"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        row_slice = pl.ds(row_start, row_block)
        hidden_slice = pl.ds(hidden_start, hidden_block)

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(
                gate_smem,
                up_smem,
                d_activation_smem,
                d_gate_smem,
                d_up_smem,
                w_gate_smem,
                w_up_smem,
                ready_barrier,
            ) -> None:
                @pl.loop(0, output_tiles)
                def _output_loop(output_tile) -> None:
                    output_start = output_tile * output_block
                    activation_slice = pl.ds(output_start, output_block)
                    up_slice = pl.ds(intermediate_dim + output_start, output_block)

                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, activation_slice],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, up_slice],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_activation_ref.at[expert, row_slice, activation_slice],
                        d_activation_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[expert, hidden_slice, activation_slice],
                        w_gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[expert, hidden_slice, up_slice],
                        w_up_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    gate = gate_smem[:, :].astype(jnp.float32)
                    up = up_smem[:, :].astype(jnp.float32)
                    d_activation = d_activation_smem[:, :].astype(jnp.float32)
                    silu_gate = jax.nn.silu(gate)
                    sigmoid_gate = jax.nn.sigmoid(gate)
                    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
                    d_gate_smem[:, :] = (d_activation * up * d_silu_gate).astype(d_gate_smem.dtype)
                    d_up_smem[:, :] = (d_activation * silu_gate).astype(d_up_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, d_gate_smem, mgpu.transpose_ref(w_gate_smem, (1, 0)))
                    mgpu.wgmma(acc_ref, d_up_smem, mgpu.transpose_ref(w_up_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                gate_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                up_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_activation_smem=_dx13_wgmma_smem((row_block, output_block), d_activation_ref.dtype),
                d_gate_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_up_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                w_gate_smem=_dx13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                w_up_smem=_dx13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=5),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((row_block, hidden_block)))
        dx_ref[
            expert,
            row_slice,
            hidden_slice,
        ] = output

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, rows, hidden_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, hidden_tiles),
        grid_names=("expert", "row_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_dx13_expert_major_store_zero_mgpu_kernel(
    *,
    row_block: int,
    hidden_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block

    def body(
        d_activation_ref: Float[pl.Ref, "E C I"],
        dx_ref: Float[pl.Ref, "E C D"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        zero = jnp.zeros((row_block, hidden_block), dtype=dx_ref.dtype)
        _ = d_activation_ref
        dx_ref[
            expert,
            pl.ds(row_start, row_block),
            pl.ds(hidden_start, hidden_block),
        ] = zero

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, rows, hidden_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, hidden_tiles),
        grid_names=("expert", "row_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_dx13_source_grouped_mgpu_kernel(
    *,
    source_count: int,
    row_block: int,
    hidden_block: int,
    experts_per_rank: int,
    rows: int,
    source_rows: int,
    hidden_dim: int,
    output_dtype: jnp.dtype,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    source_offsets = tuple(range(source_count))
    row_offsets = tuple(range(row_block))

    def body(
        dx_ref: Float[pl.Ref, "E C D"],
        source_rank_ref: Int[pl.Ref, "E C"],
        valid_ref: Int[pl.Ref, "E C"],
        src_base_ref: Int[pl.Ref, "S E"],
        dx_by_source_ref: Float[pl.Ref, "Dst E Csrc D"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        hidden_slice = pl.ds(hidden_start, hidden_block)

        def copy_scope(dx_tile_smem, ready_barrier) -> None:
            mgpu.copy_gmem_to_smem(
                dx_ref.at[expert, pl.ds(row_start, row_block), hidden_slice],
                dx_tile_smem,
                ready_barrier,
            )
            mgpu.barrier_wait(ready_barrier)

            def _copy_row_to_source(row_offset: int) -> None:
                row = row_start + row_offset
                valid = valid_ref[expert, row] != 0
                src = source_rank_ref[expert, row]
                source_row = row - src_base_ref[src, expert]
                src_ordinal = (src - rank) % source_count
                source_ref = dx_tile_smem.at[row_offset, pl.ds(0, hidden_block)]

                def _copy_to_static_source(static_src_ordinal: int) -> None:
                    static_src = (rank + static_src_ordinal) % source_count
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            source_ref,
                            dx_by_source_ref.at[rank, expert, source_row, hidden_slice],
                        )
                    else:
                        remote_dx_by_source_ref = mgpu.remote_ref(
                            dx_by_source_ref,
                            static_src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        mgpu.copy_smem_to_gmem(
                            source_ref,
                            remote_dx_by_source_ref.at[rank, expert, source_row, hidden_slice],
                        )

                @pl.when(valid)
                def _switch_copy() -> None:
                    def _branch(static_src_ordinal: int):
                        def _copy_branch(_) -> None:
                            _copy_to_static_source(static_src_ordinal)

                        return _copy_branch

                    branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in source_offsets)
                    lax.switch(src_ordinal, branches, None)

            for row_offset in row_offsets:
                _copy_row_to_source(row_offset)
            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

        pl.run_scoped(
            copy_scope,
            dx_tile_smem=mgpu.SMEM((row_block, hidden_block), dtype=dx_by_source_ref.dtype),
            ready_barrier=mgpu.Barrier(num_arrivals=1),
        )

    out_shape = jax.ShapeDtypeStruct((source_count, experts_per_rank, source_rows, hidden_dim), output_dtype)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, hidden_tiles),
        grid_names=("expert", "row_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_dx13_source_compact_mgpu_kernel(
    *,
    source_count: int,
    dst_count: int,
    entries_per_dst: int,
    block_m: int,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
    intermediate_dim: int,
    block_contiguous: bool = False,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = intermediate_dim // output_block
    source_offsets = tuple(range(source_count))
    row_offsets = tuple(range(row_block))

    def body(
        d_activation_ref: Float[pl.Ref, "E C I"],
        z_ref: Float[pl.Ref, "E C twoI"],
        w13_ref: Float[pl.Ref, "E D twoI"],
        source_rank_ref: Int[pl.Ref, "E C"],
        dst_ordinal_ref: Int[pl.Ref, "E C"],
        entry_ref: Int[pl.Ref, "E C"],
        row_in_entry_ref: Int[pl.Ref, "E C"],
        valid_ref: Int[pl.Ref, "E C"],
        dx_contrib_ref: Float[pl.Ref, "Dst Q M D"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        row_slice = pl.ds(row_start, row_block)
        hidden_slice = pl.ds(hidden_start, hidden_block)

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(
                gate_smem,
                up_smem,
                d_activation_smem,
                d_gate_smem,
                d_up_smem,
                w_gate_smem,
                w_up_smem,
                ready_barrier,
            ) -> None:
                @pl.loop(0, output_tiles)
                def _output_loop(output_tile) -> None:
                    output_start = output_tile * output_block
                    activation_slice = pl.ds(output_start, output_block)
                    up_slice = pl.ds(intermediate_dim + output_start, output_block)

                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, activation_slice],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, up_slice],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_activation_ref.at[expert, row_slice, activation_slice],
                        d_activation_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[expert, hidden_slice, activation_slice],
                        w_gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[expert, hidden_slice, up_slice],
                        w_up_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    gate = gate_smem[:, :].astype(jnp.float32)
                    up = up_smem[:, :].astype(jnp.float32)
                    d_activation = d_activation_smem[:, :].astype(jnp.float32)
                    silu_gate = jax.nn.silu(gate)
                    sigmoid_gate = jax.nn.sigmoid(gate)
                    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
                    d_gate_smem[:, :] = (d_activation * up * d_silu_gate).astype(d_gate_smem.dtype)
                    d_up_smem[:, :] = (d_activation * silu_gate).astype(d_up_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, d_gate_smem, mgpu.transpose_ref(w_gate_smem, (1, 0)))
                    mgpu.wgmma(acc_ref, d_up_smem, mgpu.transpose_ref(w_up_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                gate_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                up_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_activation_smem=_dx13_wgmma_smem((row_block, output_block), d_activation_ref.dtype),
                d_gate_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_up_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                w_gate_smem=_dx13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                w_up_smem=_dx13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=5),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((row_block, hidden_block)))

        def compact_store_scope(dx_tile_smem) -> None:
            dx_tile_smem[:, :] = output.astype(dx_tile_smem.dtype)
            mgpu.commit_smem()

            def _copy_tile_to_source() -> None:
                valid = valid_ref[expert, row_start] != 0
                src = source_rank_ref[expert, row_start]
                dst_ordinal = dst_ordinal_ref[expert, row_start]
                entry = entry_ref[expert, row_start]
                src_ordinal = (src - rank) % source_count
                source_ref = dx_tile_smem.at[pl.ds(0, row_block), pl.ds(0, hidden_block)]

                def _copy_to_static_source(static_src_ordinal: int) -> None:
                    static_src = (rank + static_src_ordinal) % source_count
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            source_ref,
                            dx_contrib_ref.at[dst_ordinal, entry, pl.ds(0, row_block), hidden_slice],
                        )
                    else:
                        remote_dx_contrib_ref = mgpu.remote_ref(
                            dx_contrib_ref,
                            static_src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        mgpu.copy_smem_to_gmem(
                            source_ref,
                            remote_dx_contrib_ref.at[dst_ordinal, entry, pl.ds(0, row_block), hidden_slice],
                        )

                @pl.when(valid)
                def _switch_copy() -> None:
                    def _branch(static_src_ordinal: int):
                        def _copy_branch(_) -> None:
                            _copy_to_static_source(static_src_ordinal)

                        return _copy_branch

                    branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in source_offsets)
                    lax.switch(src_ordinal, branches, None)

            def _copy_row_to_source(row_offset: int) -> None:
                row = row_start + row_offset
                valid = valid_ref[expert, row] != 0
                src = source_rank_ref[expert, row]
                dst_ordinal = dst_ordinal_ref[expert, row]
                entry = entry_ref[expert, row]
                row_in_entry = row_in_entry_ref[expert, row]
                src_ordinal = (src - rank) % source_count
                source_ref = dx_tile_smem.at[row_offset, pl.ds(0, hidden_block)]

                def _copy_to_static_source(static_src_ordinal: int) -> None:
                    static_src = (rank + static_src_ordinal) % source_count
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            source_ref,
                            dx_contrib_ref.at[dst_ordinal, entry, row_in_entry, hidden_slice],
                        )
                    else:
                        remote_dx_contrib_ref = mgpu.remote_ref(
                            dx_contrib_ref,
                            static_src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        mgpu.copy_smem_to_gmem(
                            source_ref,
                            remote_dx_contrib_ref.at[dst_ordinal, entry, row_in_entry, hidden_slice],
                        )

                @pl.when(valid)
                def _switch_copy() -> None:
                    def _branch(static_src_ordinal: int):
                        def _copy_branch(_) -> None:
                            _copy_to_static_source(static_src_ordinal)

                        return _copy_branch

                    branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in source_offsets)
                    lax.switch(src_ordinal, branches, None)

            if block_contiguous:
                _copy_tile_to_source()
            else:
                for row_offset in row_offsets:
                    _copy_row_to_source(row_offset)
            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

        pl.run_scoped(
            compact_store_scope,
            dx_tile_smem=mgpu.SMEM((row_block, hidden_block), dtype=dx_contrib_ref.dtype),
        )

    out_shape = jax.ShapeDtypeStruct((dst_count, entries_per_dst, block_m, hidden_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, hidden_tiles),
        grid_names=("expert", "row_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_dx13_route_buffer_mgpu_kernel(
    *,
    source_count: int,
    tokens_per_source: int,
    topk: int,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
    intermediate_dim: int,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = intermediate_dim // output_block
    source_offsets = tuple(range(source_count))
    row_offsets = tuple(range(row_block))

    def body(
        d_activation_ref: Float[pl.Ref, "E C I"],
        z_ref: Float[pl.Ref, "E C twoI"],
        w13_ref: Float[pl.Ref, "E D twoI"],
        source_rank_ref: Int[pl.Ref, "E C"],
        token_id_ref: Int[pl.Ref, "E C"],
        route_slot_ref: Int[pl.Ref, "E C"],
        valid_ref: Int[pl.Ref, "E C"],
        dx_routes_ref: Float[pl.Ref, "T K D"],
    ) -> None:
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        row_slice = pl.ds(row_start, row_block)
        hidden_slice = pl.ds(hidden_start, hidden_block)

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(
                gate_smem,
                up_smem,
                d_activation_smem,
                d_gate_smem,
                d_up_smem,
                w_gate_smem,
                w_up_smem,
                ready_barrier,
            ) -> None:
                @pl.loop(0, output_tiles)
                def _output_loop(output_tile) -> None:
                    output_start = output_tile * output_block
                    activation_slice = pl.ds(output_start, output_block)
                    up_slice = pl.ds(intermediate_dim + output_start, output_block)

                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, activation_slice],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, up_slice],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_activation_ref.at[expert, row_slice, activation_slice],
                        d_activation_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[expert, hidden_slice, activation_slice],
                        w_gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[expert, hidden_slice, up_slice],
                        w_up_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    gate = gate_smem[:, :].astype(jnp.float32)
                    up = up_smem[:, :].astype(jnp.float32)
                    d_activation = d_activation_smem[:, :].astype(jnp.float32)
                    silu_gate = jax.nn.silu(gate)
                    sigmoid_gate = jax.nn.sigmoid(gate)
                    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
                    d_gate_smem[:, :] = (d_activation * up * d_silu_gate).astype(d_gate_smem.dtype)
                    d_up_smem[:, :] = (d_activation * silu_gate).astype(d_up_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, d_gate_smem, mgpu.transpose_ref(w_gate_smem, (1, 0)))
                    mgpu.wgmma(acc_ref, d_up_smem, mgpu.transpose_ref(w_up_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                gate_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                up_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_activation_smem=_dx13_wgmma_smem((row_block, output_block), d_activation_ref.dtype),
                d_gate_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_up_smem=_dx13_wgmma_smem((row_block, output_block), z_ref.dtype),
                w_gate_smem=_dx13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                w_up_smem=_dx13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=5),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((row_block, hidden_block)))

        def route_store_scope(dx_tile_smem) -> None:
            dx_tile_smem[:, :] = output.astype(dx_tile_smem.dtype)
            mgpu.commit_smem()

            def _copy_row_to_source(row_offset: int) -> None:
                row = row_start + row_offset
                valid = valid_ref[expert, row] != 0
                src = source_rank_ref[expert, row]
                token = token_id_ref[expert, row]
                slot = route_slot_ref[expert, row]
                src_ordinal = (src - rank) % source_count
                source_ref = dx_tile_smem.at[row_offset, pl.ds(0, hidden_block)]

                def _copy_to_static_source(static_src_ordinal: int) -> None:
                    static_src = (rank + static_src_ordinal) % source_count
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(source_ref, dx_routes_ref.at[token, slot, hidden_slice])
                    else:
                        remote_dx_routes_ref = mgpu.remote_ref(
                            dx_routes_ref,
                            static_src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        mgpu.copy_smem_to_gmem(source_ref, remote_dx_routes_ref.at[token, slot, hidden_slice])

                @pl.when(valid)
                def _switch_copy() -> None:
                    def _branch(static_src_ordinal: int):
                        def _copy_branch(_) -> None:
                            _copy_to_static_source(static_src_ordinal)

                        return _copy_branch

                    branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in source_offsets)
                    lax.switch(src_ordinal, branches, None)

            for row_offset in row_offsets:
                _copy_row_to_source(row_offset)
            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

        pl.run_scoped(
            route_store_scope,
            dx_tile_smem=mgpu.SMEM((row_block, hidden_block), dtype=dx_routes_ref.dtype),
        )

    out_shape = jax.ShapeDtypeStruct((tokens_per_source, topk, hidden_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, hidden_tiles),
        grid_names=("expert", "row_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _validate_dx13_expert_major_pallas_request(
    d_activation: Array,
    z: Array,
    w13: Array,
    valid: Array,
    block_sizes: SourcePushDx13PallasBlockSizes,
) -> None:
    _validate_dx13_shapes(d_activation, z, w13, valid)
    if d_activation.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"d_activation capacity {d_activation.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if w13.shape[-2] % block_sizes.hidden_block:
        raise ValueError(
            f"w13 hidden dim {w13.shape[-2]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )
    if d_activation.shape[-1] % block_sizes.output_block:
        raise ValueError(
            f"d_activation dim {d_activation.shape[-1]} must be divisible by output_block={block_sizes.output_block}"
        )
    _validate_dx13_wgmma_smem_shape((block_sizes.row_block, block_sizes.output_block), z.dtype)
    _validate_dx13_wgmma_smem_shape((block_sizes.row_block, block_sizes.output_block), d_activation.dtype)
    _validate_dx13_wgmma_smem_shape((block_sizes.hidden_block, block_sizes.output_block), w13.dtype)


def _validate_dx13_route_buffer_pallas_request(
    d_activation: Array,
    z: Array,
    w13: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    route_slot_by_expert: Array,
    valid: Array,
    *,
    tokens_per_source: int,
    topk: int,
    block_sizes: SourcePushDx13PallasBlockSizes,
) -> None:
    _validate_dx13_route_buffer_shapes(
        d_activation,
        z,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        route_slot_by_expert,
        valid,
        tokens_per_source=tokens_per_source,
        topk=topk,
    )
    _validate_dx13_expert_major_pallas_request(d_activation, z, w13, valid, block_sizes)


def _validate_dx13_source_compact_pallas_request(
    d_activation: Array,
    z: Array,
    w13: Array,
    compact_slots: SourcePushDx13SourceCompactSlots,
    *,
    queue_shape: tuple[int, int, int, int],
    block_sizes: SourcePushDx13PallasBlockSizes,
) -> None:
    _validate_source_compact_slot_metadata(d_activation.shape[:3], compact_slots, queue_shape)
    if len(queue_shape) != 4:
        raise ValueError(f"queue_shape must be [source, dst_ordinal, entry, row], got {queue_shape}")
    source_count, dst_count, entries_per_dst, block_m = queue_shape
    if source_count != d_activation.shape[0]:
        raise ValueError(f"queue source count {source_count} must match d_activation {d_activation.shape[0]}")
    if min(dst_count, entries_per_dst, block_m) <= 0:
        raise ValueError(f"source compact queue dimensions must be positive, got {queue_shape}")
    _validate_dx13_expert_major_pallas_request(d_activation, z, w13, compact_slots.valid_by_expert, block_sizes)


def _validate_dx13_source_grouped_fields_pallas_request(
    dx_expert_major: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    route_slot_by_expert: Array,
    valid: Array,
    src_base_by_expert: Array,
    block_sizes: SourcePushDx13PallasBlockSizes,
) -> None:
    _validate_compact_metadata(dx_expert_major, source_rank_by_expert, token_id_by_expert, valid)
    if route_slot_by_expert.shape != valid.shape:
        raise ValueError(
            f"route_slot_by_expert shape {route_slot_by_expert.shape} must match valid shape {valid.shape}"
        )
    if src_base_by_expert.shape != (dx_expert_major.shape[0], dx_expert_major.shape[0], dx_expert_major.shape[1]):
        raise ValueError(
            f"src_base_by_expert shape {src_base_by_expert.shape} must be "
            f"{(dx_expert_major.shape[0], dx_expert_major.shape[0], dx_expert_major.shape[1])}"
        )
    if dx_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            f"dx hidden dim {dx_expert_major.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )


def _source_grouped_metadata_from_fields_reference(
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    source_rows: int,
) -> SourcePushDx13SourceGroupedOutput:
    dst_count, local_experts, expert_capacity = valid_by_expert.shape
    source_count = source_rank_by_expert.shape[0]
    src_base = jnp.asarray(src_base_by_expert, dtype=jnp.int32)
    out_shape = (dst_count, source_count, local_experts, source_rows)
    if source_rows < 0:
        raise ValueError(f"source_rows must be nonnegative, got {source_rows}")
    if src_base.shape != (dst_count, source_count, local_experts):
        raise ValueError(
            f"src_base_by_expert shape {src_base.shape} must be {(dst_count, source_count, local_experts)}"
        )
    if source_rows == 0:
        return SourcePushDx13SourceGroupedOutput(
            dx_by_source=jnp.zeros((*out_shape, 0), dtype=jnp.float32),
            token_id_by_source=jnp.zeros(out_shape, dtype=jnp.int32),
            route_slot_by_source=jnp.zeros(out_shape, dtype=jnp.int32),
            valid_by_source=jnp.zeros(out_shape, dtype=jnp.bool_),
        )

    dst_idx = jnp.broadcast_to(jnp.arange(dst_count, dtype=jnp.int32)[:, None, None], valid_by_expert.shape)
    expert_idx = jnp.broadcast_to(jnp.arange(local_experts, dtype=jnp.int32)[None, :, None], valid_by_expert.shape)
    row_idx = jnp.broadcast_to(jnp.arange(expert_capacity, dtype=jnp.int32)[None, None, :], valid_by_expert.shape)
    valid = valid_by_expert.astype(jnp.bool_)
    safe_src = jnp.where(valid, source_rank_by_expert, 0)
    source_base = src_base.at[dst_idx, safe_src, expert_idx].get()
    source_row = jnp.where(valid, row_idx - source_base, 0)
    valid_i = valid.astype(jnp.int32)

    token_by_source = jnp.zeros(out_shape, dtype=jnp.int32)
    slot_by_source = jnp.zeros(out_shape, dtype=jnp.int32)
    valid_by_source = jnp.zeros(out_shape, dtype=jnp.int32)
    token_by_source = token_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(
        jnp.where(valid, token_id_by_expert, 0)
    )
    slot_by_source = slot_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(
        jnp.where(valid, route_slot_by_expert, 0)
    )
    valid_by_source = valid_by_source.at[dst_idx, safe_src, expert_idx, source_row].add(valid_i)
    return SourcePushDx13SourceGroupedOutput(
        dx_by_source=jnp.zeros((*out_shape, 0), dtype=jnp.float32),
        token_id_by_source=token_by_source,
        route_slot_by_source=slot_by_source,
        valid_by_source=valid_by_source > 0,
    )


def _validate_dx13_route_buffer_shapes(
    d_activation: Array,
    z: Array,
    w13: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    route_slot_by_expert: Array,
    valid: Array,
    *,
    tokens_per_source: int,
    topk: int,
) -> None:
    _validate_dx13_shapes(d_activation, z, w13, valid)
    if source_rank_by_expert.shape != valid.shape:
        raise ValueError(
            f"source_rank_by_expert shape {source_rank_by_expert.shape} must match valid shape {valid.shape}"
        )
    if token_id_by_expert.shape != valid.shape:
        raise ValueError(f"token_id_by_expert shape {token_id_by_expert.shape} must match valid shape {valid.shape}")
    if route_slot_by_expert.shape != valid.shape:
        raise ValueError(
            f"route_slot_by_expert shape {route_slot_by_expert.shape} must match valid shape {valid.shape}"
        )
    if tokens_per_source <= 0:
        raise ValueError(f"tokens_per_source must be positive, got {tokens_per_source}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")


def _source_push_dx13_route_slot_valid_mask(
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    route_slot_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    tokens_per_source: int,
    topk: int,
) -> Bool[Array, "S T K"]:
    _validate_compact_metadata(
        jnp.zeros((*valid_by_expert.shape, 1), dtype=jnp.float32),
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
    )
    if route_slot_by_expert.shape != valid_by_expert.shape:
        raise ValueError(
            f"route_slot_by_expert shape {route_slot_by_expert.shape} must match valid shape {valid_by_expert.shape}"
        )
    source_count = source_rank_by_expert.shape[0]
    valid = valid_by_expert.astype(jnp.bool_)
    safe_src = jnp.where(valid, source_rank_by_expert, 0)
    safe_token = jnp.where(valid, token_id_by_expert, 0)
    safe_slot = jnp.where(valid, route_slot_by_expert, 0)
    route_slot_valid = jnp.zeros((source_count, tokens_per_source, topk), dtype=jnp.int32)
    route_slot_valid = route_slot_valid.at[safe_src, safe_token, safe_slot].add(
        valid.astype(jnp.int32),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    return route_slot_valid > 0


def _source_push_destination_named_sharding(value: Array, ndim: int) -> NamedSharding | None:
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return None
    if SOURCE_PUSH_MESH_AXIS not in sharding.mesh.axis_names:
        return None
    return NamedSharding(sharding.mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(ndim - 1))))


def _with_source_push_destination_sharding(
    value: Array,
    *,
    mesh: Mesh | AbstractMesh | None = None,
    like: Array | None = None,
) -> Array:
    if isinstance(mesh, Mesh):
        return jax.device_put(
            value,
            NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(value.ndim - 1)))),
        )
    sharding = _source_push_destination_named_sharding(value, value.ndim)
    if sharding is None and like is not None:
        sharding = _source_push_destination_named_sharding(like, value.ndim)
    if sharding is not None:
        return jax.device_put(value, sharding)
    return value


def _with_source_push_source_sharding(
    value: Array,
    *,
    mesh: Mesh | AbstractMesh | None = None,
) -> Array:
    if isinstance(mesh, Mesh):
        return jax.device_put(
            value,
            NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(value.ndim - 1)))),
        )
    return value


def _sharded_dx13_route_buffer_remote_write_completion_barrier(mesh: Mesh | AbstractMesh):
    def local_fn(dx_routes_local: Float[Array, "1 T K D"]) -> Float[Array, "1 T K D"]:
        dx_routes_local = dx_routes_local[0]
        marker = dx_routes_local[0, 0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = barrier - lax.optimization_barrier(barrier)
        dx_routes_local = dx_routes_local.at[0, 0, 0].add(zero.astype(dx_routes_local.dtype))
        return dx_routes_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )


def _sharded_dx13_source_compact_remote_write_completion_barrier(mesh: Mesh | AbstractMesh):
    def local_fn(dx_contrib_local: Float[Array, "1 Dst Q M D"]) -> Float[Array, "1 Dst Q M D"]:
        dx_contrib_local = dx_contrib_local[0]
        marker = dx_contrib_local[0, 0, 0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = barrier - lax.optimization_barrier(barrier)
        dx_contrib_local = dx_contrib_local.at[0, 0, 0, 0].add(zero.astype(dx_contrib_local.dtype))
        return dx_contrib_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )


def _sharded_dx13_source_grouped_remote_write_completion_barrier(mesh: Mesh | AbstractMesh):
    def local_fn(dx_by_source_local: Float[Array, "1 Dst E Csrc D"]) -> Float[Array, "1 Dst E Csrc D"]:
        dx_by_source_local = dx_by_source_local[0]
        marker = dx_by_source_local[0, 0, 0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, SOURCE_PUSH_MESH_AXIS)
        zero = barrier - lax.optimization_barrier(barrier)
        dx_by_source_local = dx_by_source_local.at[0, 0, 0, 0].add(zero.astype(dx_by_source_local.dtype))
        return dx_by_source_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
        check_vma=False,
    )


def _source_push_destination_or_replicated_spec(value: Array, ndim: int) -> P:
    destination_spec = P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(ndim - 1)))
    sharding = getattr(value, "sharding", None)
    if isinstance(sharding, NamedSharding) and sharding.spec == destination_spec:
        return destination_spec
    return P(*(None for _ in range(ndim)))


def _dx13_wgmma_smem(shape: tuple[int, int], dtype):
    _validate_dx13_wgmma_smem_shape(shape, dtype)
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((DX13_WGMMA_TILE_M, DX13_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize)),
            mgpu.SwizzleTransform(DX13_WGMMA_SWIZZLE_BYTES),
        ),
    )


def _validate_dx13_wgmma_smem_shape(shape: tuple[int, int], dtype) -> None:
    swizzle_elems = DX13_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % DX13_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "DX13 WGMMA SMEM operands must be divisible by "
            f"tile=({DX13_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )


def _validate_source_compact_slots(
    dx_expert_major: Array,
    compact_slots: SourcePushDx13SourceCompactSlots,
    queue_shape: tuple[int, int, int, int],
) -> None:
    if dx_expert_major.ndim != 4:
        raise ValueError(
            f"dx_expert_major must have shape [dst, expert, capacity, hidden], got {dx_expert_major.shape}"
        )
    _validate_source_compact_slot_metadata(dx_expert_major.shape[:3], compact_slots, queue_shape)


def _validate_source_compact_slot_metadata(
    expected_shape: tuple[int, int, int],
    compact_slots: SourcePushDx13SourceCompactSlots,
    queue_shape: tuple[int, int, int, int],
) -> None:
    if len(queue_shape) != 4:
        raise ValueError(f"queue_shape must be [source, dst_ordinal, entry, row], got {queue_shape}")
    if min(queue_shape) < 0:
        raise ValueError(f"queue_shape must be nonnegative, got {queue_shape}")
    for field_name in (
        "source_rank_by_expert",
        "dst_ordinal_by_expert",
        "entry_by_expert",
        "row_in_entry_by_expert",
        "valid_by_expert",
    ):
        value = getattr(compact_slots, field_name)
        if value.shape != expected_shape:
            raise ValueError(f"{field_name} shape {value.shape} must match dx rows {expected_shape}")


def source_push_dx13_source_compact_slots_are_block_contiguous(
    compact_slots: SourcePushDx13SourceCompactSlots,
    queue_shape: tuple[int, int, int, int],
    *,
    row_block: int,
) -> bool:
    """Return whether valid row tiles are exactly one contiguous source queue entry."""

    try:
        _validate_dx13_source_compact_block_contiguous_slots(
            compact_slots.valid_by_expert.shape,
            compact_slots,
            queue_shape,
            row_block=row_block,
        )
    except ValueError:
        return False
    return True


def _validate_dx13_source_compact_block_contiguous_slots(
    expected_shape: tuple[int, int, int],
    compact_slots: SourcePushDx13SourceCompactSlots,
    queue_shape: tuple[int, int, int, int],
    *,
    row_block: int,
) -> None:
    _validate_source_compact_slot_metadata(expected_shape, compact_slots, queue_shape)
    _source_count, _dst_count, _entries_per_dst, block_m = queue_shape
    if row_block != block_m:
        raise ValueError(
            f"block-contiguous DX13 source-compact requires row_block == queue block_m, got {row_block} and {block_m}"
        )
    if expected_shape[-1] % row_block:
        raise ValueError(
            f"block-contiguous DX13 source-compact rows must be divisible by row_block={row_block}, "
            f"got {expected_shape[-1]}"
        )

    valid = np.asarray(jax.device_get(compact_slots.valid_by_expert), dtype=np.bool_)
    source_rank = np.asarray(jax.device_get(compact_slots.source_rank_by_expert), dtype=np.int32)
    dst_ordinal = np.asarray(jax.device_get(compact_slots.dst_ordinal_by_expert), dtype=np.int32)
    entry = np.asarray(jax.device_get(compact_slots.entry_by_expert), dtype=np.int32)
    row_in_entry = np.asarray(jax.device_get(compact_slots.row_in_entry_by_expert), dtype=np.int32)
    expected_rows = np.arange(row_block, dtype=np.int32)

    for dst in range(expected_shape[0]):
        for expert in range(expected_shape[1]):
            for row_start in range(0, expected_shape[2], row_block):
                row_slice = slice(row_start, row_start + row_block)
                valid_tile = valid[dst, expert, row_slice]
                if not bool(valid_tile[0]):
                    if np.any(valid_tile):
                        raise ValueError(
                            "block-contiguous DX13 source-compact cannot use partially live row tiles "
                            f"at dst={dst}, expert={expert}, row_start={row_start}"
                        )
                    continue
                if not np.all(valid_tile):
                    raise ValueError(
                        "block-contiguous DX13 source-compact requires full live row tiles "
                        f"at dst={dst}, expert={expert}, row_start={row_start}"
                    )
                if not np.all(source_rank[dst, expert, row_slice] == source_rank[dst, expert, row_start]):
                    raise ValueError(
                        "block-contiguous DX13 source-compact requires one source rank per row tile "
                        f"at dst={dst}, expert={expert}, row_start={row_start}"
                    )
                if not np.all(dst_ordinal[dst, expert, row_slice] == dst_ordinal[dst, expert, row_start]):
                    raise ValueError(
                        "block-contiguous DX13 source-compact requires one destination ordinal per row tile "
                        f"at dst={dst}, expert={expert}, row_start={row_start}"
                    )
                if not np.all(entry[dst, expert, row_slice] == entry[dst, expert, row_start]):
                    raise ValueError(
                        "block-contiguous DX13 source-compact requires one queue entry per row tile "
                        f"at dst={dst}, expert={expert}, row_start={row_start}"
                    )
                if not np.array_equal(row_in_entry[dst, expert, row_slice], expected_rows):
                    raise ValueError(
                        "block-contiguous DX13 source-compact requires row_in_entry == 0..row_block-1 "
                        f"at dst={dst}, expert={expert}, row_start={row_start}"
                    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _validate_dx13_shapes(
    d_activation: Array,
    z: Array,
    w13: Array,
    valid_by_expert: Array,
) -> None:
    _validate_swiglu_shapes(d_activation, z, valid_by_expert)
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, hidden, twoI], got {w13.shape}")
    if w13.shape[:2] != d_activation.shape[:2]:
        raise ValueError(f"w13 destination/expert shape {w13.shape[:2]} must match {d_activation.shape[:2]}")
    if w13.shape[-1] != z.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match z output dim {z.shape[-1]}")


def _validate_swiglu_shapes(
    d_activation: Array,
    z: Array,
    valid_by_expert: Array,
) -> None:
    if d_activation.ndim != 4:
        raise ValueError(
            f"d_activation must have shape [dst, expert, capacity, intermediate], got {d_activation.shape}"
        )
    if z.ndim != 4:
        raise ValueError(f"z must have shape [dst, expert, capacity, twoI], got {z.shape}")
    if valid_by_expert.shape != d_activation.shape[:3]:
        raise ValueError(f"valid shape {valid_by_expert.shape} must match d_activation {d_activation.shape[:3]}")
    if z.shape[:3] != d_activation.shape[:3]:
        raise ValueError(f"z leading shape {z.shape[:3]} must match d_activation {d_activation.shape[:3]}")
    if z.shape[-1] != 2 * d_activation.shape[-1]:
        raise ValueError(f"z output dim {z.shape[-1]} must be twice d_activation dim {d_activation.shape[-1]}")


def _validate_compact_metadata(
    dx_expert_major: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    valid_by_expert: Array,
) -> None:
    if dx_expert_major.ndim != 4:
        raise ValueError(
            f"dx_expert_major must have shape [dst, expert, capacity, hidden], got {dx_expert_major.shape}"
        )
    if valid_by_expert.shape != dx_expert_major.shape[:3]:
        raise ValueError(f"valid shape {valid_by_expert.shape} must match dx rows {dx_expert_major.shape[:3]}")
    if source_rank_by_expert.shape != valid_by_expert.shape:
        raise ValueError(
            f"source_rank_by_expert shape {source_rank_by_expert.shape} must match valid shape {valid_by_expert.shape}"
        )
    if token_id_by_expert.shape != valid_by_expert.shape:
        raise ValueError(
            f"token_id_by_expert shape {token_id_by_expert.shape} must match valid shape {valid_by_expert.shape}"
        )
