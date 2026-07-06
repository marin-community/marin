# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Package-private W13 backward helpers for the source-push MoE MLP.

The functions in this module cover backward steps 6-7 from the MLP custom-VJP
plan: rematerialize source-major ``x`` into the same flat expert-major row
layout as W13/H, then compute ``dx_expert_major`` and ``dw13`` from ``dH``.
They intentionally take source-major ``x`` plus static ``SourcePushPlan``
metadata rather than saved packed/recv-x residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SOURCE_PUSH_META_LOCAL_EXPERT,
    SOURCE_PUSH_META_LOCAL_ROW_START,
    SourcePushPlan,
    _source_push_out_sharding,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED = "tiled"
SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_COMPACT = "pallas_mgpu_compact"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY = "pallas_mgpu_compact_dx_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY = "source_gather_dw13_only"
SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu_x_remat"
SourcePushW13BackwardImplementation: TypeAlias = Literal["reference", "tiled", "pallas_mgpu"]
SourcePushXToW13RowsImplementation: TypeAlias = Literal["reference", "pallas_mgpu_x_remat"]
DEFAULT_X_REMAT_HIDDEN_BLOCK = 128
DEFAULT_W13_DX_ROW_BLOCK = 64
DEFAULT_W13_DX_HIDDEN_BLOCK = 128
DEFAULT_W13_DX_OUTPUT_BLOCK = 64
W13_WGMMA_SWIZZLE_BYTES = 128
W13_WGMMA_TILE_M = 8


class SourcePushW13BackwardOutput(NamedTuple):
    """Flat-row W13 backward outputs.

    ``x_expert_major`` is a rematerialized compute temporary, not an MLP VJP
    residual. The custom VJP should save source-major ``x`` and rematerialize
    this value only during backward. Implementations that fuse rematerialization
    into tiled W13 work may return a size-zero placeholder when callers do not
    request the debug temporary.
    """

    x_expert_major: Float[Array, "Dst rows D"]
    dx_expert_major: Float[Array, "Dst rows D"]
    dw13: Float[Array, "Dst E D twoI"]


class SourcePushW13CompactBackwardOutput(NamedTuple):
    """Compact expert-block W13 backward outputs."""

    x_expert_major: Float[Array, "Dst E C D"]
    dx_expert_major: Float[Array, "Dst E C D"]
    dw13: Float[Array, "Dst E D twoI"]


class SourcePushW13FlatRowMap(NamedTuple):
    """Inverse map from destination flat H rows back to source-major tokens."""

    src: Int[Array, "Dst rows"]
    token: Int[Array, "Dst rows"]
    expert: Int[Array, "Dst rows"]
    valid: Bool[Array, "Dst rows"]


class _SourcePushW13RowIndices(NamedTuple):
    valid: Bool[Array, "S Dst Q M"]
    valid_f: Float[Array, "S Dst Q M"]
    safe_src: Int[Array, "S Dst Q M"]
    safe_token: Int[Array, "S Dst Q M"]
    safe_dst: Int[Array, "S Dst Q M"]
    safe_row: Int[Array, "S Dst Q M"]
    safe_local_row: Int[Array, "S Dst Q M"]
    safe_expert: Int[Array, "S Dst Q M"]


@dataclass(frozen=True, slots=True)
class SourcePushXToW13RowsPallasBlockSizes:
    """Tile sizes for the source-token-to-W13-row rematerialization kernel."""

    hidden_block: int = DEFAULT_X_REMAT_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushXToW13RowsPallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushW13BackwardTiledBlockSizes:
    """Tile sizes for bounded JAX W13 backward accumulation."""

    row_block: int = DEFAULT_W13_DX_ROW_BLOCK
    hidden_block: int = DEFAULT_W13_DX_HIDDEN_BLOCK
    output_block: int = DEFAULT_W13_DX_OUTPUT_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushW13BackwardTiledBlockSizes":
        return cls()


@dataclass(frozen=True)
class SourcePushW13BackwardCostEstimate:
    """Shape-derived cost estimate for W13 backward steps 6-7."""

    useful_rows_per_rank: int
    padded_rows_per_rank: int
    w13_backward_flops_per_rank: int
    x_remat_bytes_per_rank: int
    x_remat_padded_bytes_per_rank: int
    math_seconds_at_reference_tflops_per_rank: float


def source_push_x_to_w13_rows(
    x: Float[Array, "S T D"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool = False,
    implementation: SourcePushXToW13RowsImplementation = SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushXToW13RowsPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst rows D"]:
    """Rematerialize source tokens into flat W13/H rows.

    ``pallas_mgpu_x_remat`` is intentionally scoped to the x-rematerialization
    substage. Full Pallas/MGPU W13 backward remains unimplemented.
    """

    if implementation == SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_REFERENCE:
        return source_push_x_to_w13_rows_reference(
            x,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            hidden_rows_per_rank=hidden_rows_per_rank,
            use_exact_expert_major=use_exact_expert_major,
        )
    if implementation == SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU:
        flat_row_map = source_push_w13_flat_row_map_jax(
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            hidden_rows_per_rank=hidden_rows_per_rank,
            use_exact_expert_major=use_exact_expert_major,
        )
        return _source_push_x_to_w13_rows_pallas_mgpu(
            x,
            flat_row_map,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    raise ValueError(
        "source-push x-to-W13-rows implementation must be one of "
        f"{SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_REFERENCE!r}, "
        f"{SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU!r}; "
        f"got {implementation!r}"
    )


def source_push_x_to_w13_rows_reference(
    x: Float[Array, "S T D"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool = False,
) -> Float[Array, "Dst rows D"]:
    """Rematerialize source tokens into the flat expert-major W13/H row layout."""

    row_indices = _source_push_w13_row_indices(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=use_exact_expert_major,
    )
    x_rows = x.at[row_indices.safe_src, row_indices.safe_token].get(
        out_sharding=_source_push_out_sharding(None, None, None, None, None)
    )
    x_rows = x_rows.astype(jnp.float32) * row_indices.valid_f[..., None]
    out = jnp.zeros((plan.assignment_ids.shape[0], hidden_rows_per_rank, x.shape[-1]), dtype=jnp.float32)
    return out.at[row_indices.safe_dst, row_indices.safe_row].add(
        x_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )


def source_push_w13_flat_row_map_jax(
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    hidden_rows_per_rank: int,
    use_exact_expert_major: bool = False,
) -> SourcePushW13FlatRowMap:
    """Build the inverse map consumed by the x-rematerialization Pallas kernel."""

    row_indices = _source_push_w13_row_indices(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=use_exact_expert_major,
    )
    dst_count = plan.assignment_ids.shape[0]
    out_shape = (dst_count, hidden_rows_per_rank)
    zero_i = jnp.zeros(out_shape, dtype=jnp.int32)
    valid_i = zero_i.at[row_indices.safe_dst, row_indices.safe_row].add(
        row_indices.valid.astype(jnp.int32),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    src = zero_i.at[row_indices.safe_dst, row_indices.safe_row].add(
        jnp.where(row_indices.valid, row_indices.safe_src, 0),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    token = zero_i.at[row_indices.safe_dst, row_indices.safe_row].add(
        jnp.where(row_indices.valid, row_indices.safe_token, 0),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    expert = zero_i.at[row_indices.safe_dst, row_indices.safe_row].add(
        jnp.where(row_indices.valid, row_indices.safe_expert, 0),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    return SourcePushW13FlatRowMap(src=src, token=token, expert=expert, valid=valid_i > 0)


def source_push_w13_backward_reference(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool = False,
) -> SourcePushW13BackwardOutput:
    """Compute ``x`` rematerialization, flat ``dx``, and ``dw13`` for W13 backward.

    Invalid queue rows are masked before the W13 math, so nonzero garbage in
    padded ``d_h`` rows cannot affect ``dx_expert_major`` or ``dw13``.
    """

    row_indices = _source_push_w13_row_indices(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=use_exact_expert_major,
    )
    x_rows = x.at[row_indices.safe_src, row_indices.safe_token].get(
        out_sharding=_source_push_out_sharding(None, None, None, None, None)
    )
    x_rows = x_rows.astype(jnp.float32) * row_indices.valid_f[..., None]
    d_h_rows = d_h.at[row_indices.safe_dst, row_indices.safe_row].get(
        out_sharding=_source_push_out_sharding(None, None, None, None, None)
    )
    d_h_rows = d_h_rows.astype(jnp.float32) * row_indices.valid_f[..., None]
    w13_rows = w13.at[row_indices.safe_dst, row_indices.safe_expert].get(
        out_sharding=_source_push_out_sharding(None, None, None, None, None, None)
    )
    w13_rows = w13_rows.astype(jnp.float32)

    dx_rows = jnp.einsum("...o,...do->...d", d_h_rows, w13_rows)
    dw13_rows = jnp.einsum("...d,...o->...do", x_rows, d_h_rows)

    dst_count = plan.assignment_ids.shape[0]
    hidden_rows_per_rank = d_h.shape[1]
    x_expert_major = jnp.zeros((dst_count, hidden_rows_per_rank, x.shape[-1]), dtype=jnp.float32)
    dx_expert_major = jnp.zeros_like(x_expert_major)
    dw13 = jnp.zeros(w13.shape, dtype=jnp.float32)

    x_expert_major = x_expert_major.at[row_indices.safe_dst, row_indices.safe_row].add(
        x_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    dx_expert_major = dx_expert_major.at[row_indices.safe_dst, row_indices.safe_row].add(
        dx_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    dw13 = dw13.at[row_indices.safe_dst, row_indices.safe_expert].add(
        dw13_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )
    return SourcePushW13BackwardOutput(x_expert_major, dx_expert_major, dw13)


def source_push_w13_backward_expert_blocks_reference(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool = False,
) -> SourcePushW13CompactBackwardOutput:
    """Compute W13 backward directly from compact expert-block ``dH``.

    This is the reference bridge for the H-residual backward design. It avoids
    flattening ``dH`` into ``[Dst, rows, twoI]`` before W13 and produces compact
    ``dx_expert_major`` that can feed the compact return/combine helper.
    """

    row_indices = _source_push_w13_row_indices(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        use_exact_expert_major=use_exact_expert_major,
    )
    x_rows = x.at[row_indices.safe_src, row_indices.safe_token].get(
        out_sharding=_source_push_out_sharding(None, None, None, None, None)
    )
    x_rows = x_rows.astype(jnp.float32) * row_indices.valid_f[..., None]

    dst_count = plan.assignment_ids.shape[0]
    local_experts = w13.shape[1]
    expert_capacity = d_h.shape[2]
    x_expert_major = jnp.zeros((dst_count, local_experts, expert_capacity, x.shape[-1]), dtype=jnp.float32)
    valid_blocks_i = jnp.zeros((dst_count, local_experts, expert_capacity), dtype=jnp.int32)

    x_expert_major = x_expert_major.at[
        row_indices.safe_dst,
        row_indices.safe_expert,
        row_indices.safe_local_row,
    ].add(
        x_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )
    valid_blocks_i = valid_blocks_i.at[
        row_indices.safe_dst,
        row_indices.safe_expert,
        row_indices.safe_local_row,
    ].add(
        row_indices.valid.astype(jnp.int32),
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )

    valid_f = (valid_blocks_i > 0).astype(jnp.float32)
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]
    w13 = w13.astype(jnp.float32)
    dx_expert_major = jnp.einsum("deco,deho->dech", d_h_clean, w13)
    dw13 = jnp.einsum("dech,deco->deho", x_expert_major, d_h_clean)
    return SourcePushW13CompactBackwardOutput(x_expert_major, dx_expert_major, dw13)


def source_push_w13_backward_expert_blocks_tiled_reference(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    return_x_expert_major: bool = False,
) -> SourcePushW13CompactBackwardOutput:
    """Bounded compact W13 backward that does not materialize rematerialized ``x``.

    This is the compact expert-block equivalent of
    ``source_push_w13_backward_tiled_reference``. It consumes ``dH`` in
    ``[destination, local_expert, row, 2I]`` form and gathers source-major ``x``
    by compact route metadata per matmul tile, so staged-block backward can keep
    the H boundary without flattening ``dH`` through ``[destination, rows, 2I]``.
    """

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_w13_backward_expert_blocks_tiled_request(
        x,
        d_h,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes,
    )
    dx_expert_major, dw13 = _source_push_w13_backward_tiled_from_expert_blocks(
        x,
        d_h,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes=block_sizes,
    )
    if return_x_expert_major:
        safe_src = jnp.where(valid_by_expert, source_rank_by_expert, 0)
        safe_token = jnp.where(valid_by_expert, token_id_by_expert, 0)
        x_expert_major = x.at[safe_src, safe_token].get(
            out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
        )
        x_expert_major = x_expert_major.astype(jnp.float32) * valid_by_expert[..., None].astype(jnp.float32)
    else:
        x_expert_major = jnp.zeros((0,), dtype=jnp.float32)
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=x_expert_major,
        dx_expert_major=dx_expert_major,
        dw13=dw13,
    )


def source_push_w13_dx_expert_blocks_reference(
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C D"]:
    """Diagnostic W13 ``dx`` half for compact expert blocks."""

    if d_h.ndim != 4:
        raise ValueError(f"d_h must have shape [dst, expert, capacity, twoI], got {d_h.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, D, twoI], got {w13.shape}")
    if valid_by_expert.shape != d_h.shape[:3]:
        raise ValueError(f"valid shape {valid_by_expert.shape} must match d_h blocks {d_h.shape[:3]}")
    if w13.shape[:2] != d_h.shape[:2]:
        raise ValueError(f"w13 destination/expert shape {w13.shape[:2]} must match d_h {d_h.shape[:2]}")
    if w13.shape[-1] != d_h.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match d_h output dim {d_h.shape[-1]}")
    d_h_clean = d_h.astype(jnp.float32) * valid_by_expert.astype(jnp.float32)[..., None]
    return jnp.einsum("deco,deho->dech", d_h_clean, w13.astype(jnp.float32))


def source_push_w13_dw13_expert_blocks_source_gather_tiled_reference(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
) -> Float[Array, "Dst E D twoI"]:
    """Diagnostic source-gather W13 ``dw13`` without compact ``x`` GMEM staging.

    This is the bounded JAX analogue of the intended source-push/tile-gather
    WGMMA kernel: for each ``dw13`` tile it gathers only the required source
    token rows for that row tile, multiplies by local ``dH``, and accumulates
    into destination-local expert weights. It deliberately has no
    ``x_expert_major`` output.
    """

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_w13_dw13_source_gather_request(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes,
    )
    return _source_push_w13_dw13_tiled_from_expert_blocks(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes=block_sizes,
    )


def _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only compact W13 ``dx`` diagnostic using the existing WGMMA path."""

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    if interpret:
        dx_expert_major = source_push_w13_dx_expert_blocks_reference(d_h, w13, valid_by_expert)
        return SourcePushW13CompactBackwardOutput(
            x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dx_expert_major=dx_expert_major,
            dw13=jnp.zeros(w13.shape, dtype=jnp.float32),
        )
    original_rows = d_h.shape[2]
    d_h, valid_by_expert = _pad_w13_compact_dh_for_row_block(d_h, valid_by_expert, block_sizes.row_block)
    _validate_w13_expert_blocks_pallas_request(
        d_h,
        w13,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU compact W13 dx-only diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU compact W13 dx-only diagnostic requires a mesh")

    valid_f = valid_by_expert.astype(jnp.float32)
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]
    dx_expert_major = _source_push_w13_dx_expert_blocks_sharded_mgpu_kernel(
        mesh,
        d_h_clean.astype(w13.dtype),
        w13,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=dx_expert_major[:, :, :original_rows, :],
        dw13=jnp.zeros(w13.shape, dtype=jnp.float32),
    )


def _pad_w13_compact_dh_for_row_block(
    d_h: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    row_block: int,
) -> tuple[Float[Array, "Dst E Cpad twoI"], Bool[Array, "Dst E Cpad"]]:
    original_rows = d_h.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, row_block)
    if padded_rows == original_rows:
        return d_h, valid_by_expert

    row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
    return jnp.pad(d_h, (*row_pad, (0, 0))), jnp.pad(valid_by_expert, row_pad, constant_values=False)


def source_push_w13_backward_expert_blocks_source_gather_dw13_only(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only compact W13 ``dw13`` diagnostic with source-tile gathers."""

    dw13 = source_push_w13_dw13_expert_blocks_source_gather_tiled_reference(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes=block_sizes,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros(d_h.shape[:3] + (x.shape[-1],), dtype=jnp.float32),
        dw13=dw13,
    )


def _source_push_w13_backward_expert_blocks_pallas_mgpu(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Compact expert-block W13 backward using destination-local WGMMA kernels.

    This path keeps the staged-block H boundary: ``d_h`` is consumed as
    ``[destination, expert, row, 2I]`` and the resulting ``dx`` stays compact
    for the compact return/combine helper. Source-major ``x`` is rematerialized
    into compact expert blocks before the two local WGMMA matmuls.
    """

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_w13_backward_expert_blocks_tiled_request(
        x,
        d_h,
        w13,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes,
    )
    if interpret:
        return source_push_w13_backward_expert_blocks_tiled_reference(
            x,
            d_h,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            block_sizes=block_sizes,
            return_x_expert_major=True,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU compact W13 backward requires a GPU backend; use the reference on CPU")
    if mesh is None:
        raise ValueError("Pallas/MGPU compact W13 backward requires a mesh for destination-local WGMMA kernels")

    original_rows = d_h.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        d_h = jnp.pad(d_h, (*row_pad, (0, 0)))
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    safe_src = jnp.where(valid_by_expert, source_rank_by_expert, 0)
    safe_token = jnp.where(valid_by_expert, token_id_by_expert, 0)
    x_expert_major = x.at[safe_src, safe_token].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    valid_f = valid_by_expert.astype(jnp.float32)
    x_expert_major = x_expert_major.astype(jnp.float32) * valid_f[..., None]
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]

    x_for_wgmma = x_expert_major.astype(w13.dtype)
    d_h_for_wgmma = d_h_clean.astype(w13.dtype)
    dx_expert_major = _source_push_w13_dx_expert_blocks_sharded_mgpu_kernel(
        mesh,
        d_h_for_wgmma,
        w13,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )
    dw13 = _source_push_w13_dw13_expert_blocks_sharded_mgpu_kernel(
        mesh,
        x_for_wgmma,
        d_h_for_wgmma,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=x_expert_major[:, :, :original_rows, :],
        dx_expert_major=dx_expert_major[:, :, :original_rows, :],
        dw13=dw13,
    )


def _source_push_x_to_w13_rows_pallas_mgpu(
    x: Float[Array, "S T D"],
    flat_row_map: SourcePushW13FlatRowMap,
    *,
    block_sizes: SourcePushXToW13RowsPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst rows D"]:
    """Lane-lowered Mosaic GPU x-rematerialization for flat W13/H rows."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError(
            "Pallas/MGPU x rematerialization requires a GPU backend; use the JAX reference on CPU"
        )
    block_sizes = SourcePushXToW13RowsPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_x_remat_pallas_request(x, flat_row_map, block_sizes)
    return _source_push_x_to_w13_rows_pallas_call(
        x,
        flat_row_map.src,
        flat_row_map.token,
        flat_row_map.valid,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )


def _source_push_x_to_w13_rows_pallas_call(
    x: Float[Array, "S T D"],
    row_src: Int[Array, "Dst rows"],
    row_token: Int[Array, "Dst rows"],
    row_valid: Bool[Array, "Dst rows"],
    *,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | AbstractMesh | None = None,
    shard_if_needed: bool = True,
) -> Float[Array, "Dst rows D"]:
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if shard_if_needed and not interpret and mesh is not None:
        return _source_push_x_to_w13_rows_sharded_pallas_call(
            mesh,
            x,
            row_src,
            row_token,
            row_valid,
            hidden_block=hidden_block,
        )
    if shard_if_needed and not interpret and not abstract_mesh.empty:
        return _source_push_x_to_w13_rows_sharded_pallas_call(
            abstract_mesh,
            x,
            row_src,
            row_token,
            row_valid,
            hidden_block=hidden_block,
        )

    dst_count, hidden_rows_per_rank = row_valid.shape
    hidden_dim = x.shape[-1]
    output_shape = jax.ShapeDtypeStruct((dst_count, hidden_rows_per_rank, hidden_dim), jnp.float32)
    row_valid_i = row_valid.astype(jnp.int32)
    cost_estimate = _source_push_x_to_w13_rows_pallas_cost_estimate(x, row_src, row_token, row_valid, output_shape)
    kernel = _make_source_push_x_to_w13_rows_kernel(hidden_block=hidden_block)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    in_specs, out_specs = _source_push_x_to_w13_rows_block_specs(hidden_block=hidden_block)
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=output_shape,
        grid=(dst_count, hidden_rows_per_rank, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_x_to_w13_rows_pallas_mgpu_x_remat",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(x, row_src, row_token, row_valid_i)


def _source_push_x_to_w13_rows_sharded_pallas_call(
    mesh: Mesh | AbstractMesh,
    x: Float[Array, "S T D"],
    row_src: Int[Array, "Dst rows"],
    row_token: Int[Array, "Dst rows"],
    row_valid: Bool[Array, "Dst rows"],
    *,
    hidden_block: int,
) -> Float[Array, "Dst rows D"]:
    """Run x-rematerialization with destination-local metadata refs.

    Under the explicit expert mesh, the destination axis of ``row_*`` is
    sharded. Mosaic GPU refs cannot slice that sharded axis inside a raw
    Pallas kernel, so ``shard_map`` makes each destination rank see a local
    ``[1, rows]`` metadata block. ``x`` is all-gathered as a correctness bridge;
    a later producer/consumer W13 backward path should avoid that gather.
    """

    def local_fn(
        x_local: Float[Array, "1 T D"],
        row_src_local: Int[Array, "1 rows"],
        row_token_local: Int[Array, "1 rows"],
        row_valid_local: Bool[Array, "1 rows"],
    ) -> Float[Array, "1 rows D"]:
        x_all = lax.all_gather(x_local[0], SOURCE_PUSH_MESH_AXIS, axis=0)
        return _source_push_x_to_w13_rows_pallas_call(
            x_all,
            row_src_local,
            row_token_local,
            row_valid_local,
            hidden_block=hidden_block,
            interpret=False,
            shard_if_needed=False,
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
    )(x, row_src, row_token, row_valid)


def _source_push_x_to_w13_rows_block_specs(
    *,
    hidden_block: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    out_spec = pl.BlockSpec(
        (None, None, hidden_block),
        lambda dst, row, hidden_tile: (dst, row, hidden_tile),
    )
    return (pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec()), out_spec


def _make_source_push_x_to_w13_rows_kernel(*, hidden_block: int):
    def kernel(
        x_ref: Float[pl.Ref, "S T D"],
        row_src_ref: Int[pl.Ref, "Dst rows"],
        row_token_ref: Int[pl.Ref, "Dst rows"],
        row_valid_ref: Int[pl.Ref, "Dst rows"],
        x_out_ref: Float[pl.Ref, "D"],
    ) -> None:
        dst = pl.program_id(0)
        row = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        valid = row_valid_ref[pl.ds(dst, 1), pl.ds(row, 1)][0, 0]
        src = row_src_ref[pl.ds(dst, 1), pl.ds(row, 1)][0, 0]
        token = row_token_ref[pl.ds(dst, 1), pl.ds(row, 1)][0, 0]
        x_tile = x_ref[pl.ds(src, 1), pl.ds(token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
        zero_tile = jnp.zeros((hidden_block,), dtype=jnp.float32)
        out_tile = jnp.where(valid, x_tile.astype(jnp.float32), zero_tile)
        x_out_ref[pl.ds(0, hidden_block)] = out_tile

    return kernel


def _source_push_x_to_w13_rows_pallas_reference(
    x: Float[Array, "S T D"],
    row_src: Int[Array, "Dst rows"],
    row_token: Int[Array, "Dst rows"],
    row_valid: Bool[Array, "Dst rows"],
) -> Float[Array, "Dst rows D"]:
    safe_src = jnp.where(row_valid, row_src, 0)
    safe_token = jnp.where(row_valid, row_token, 0)
    x_rows = x.at[safe_src, safe_token].get()
    return jnp.where(row_valid[..., None], x_rows.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))


def _source_push_x_to_w13_rows_pallas_cost_estimate(
    x: Array,
    row_src: Array,
    row_token: Array,
    row_valid: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(row_src.shape, row_src.dtype),
        jax.ShapeDtypeStruct(row_token.shape, row_token.dtype),
        jax.ShapeDtypeStruct(row_valid.shape, row_valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_x_to_w13_rows_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _validate_x_remat_pallas_request(
    x: Array,
    flat_row_map: SourcePushW13FlatRowMap,
    block_sizes: SourcePushXToW13RowsPallasBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if flat_row_map.src.shape != flat_row_map.valid.shape:
        raise ValueError(f"flat row src shape {flat_row_map.src.shape} must match valid {flat_row_map.valid.shape}")
    if flat_row_map.token.shape != flat_row_map.valid.shape:
        raise ValueError(
            f"flat row token shape {flat_row_map.token.shape} must match valid {flat_row_map.valid.shape}"
        )
    if flat_row_map.expert.shape != flat_row_map.valid.shape:
        raise ValueError(
            f"flat row expert shape {flat_row_map.expert.shape} must match valid {flat_row_map.valid.shape}"
        )
    if flat_row_map.valid.ndim != 2:
        raise ValueError(f"flat row map must have shape [destination, rows], got {flat_row_map.valid.shape}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"x hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")


def source_push_w13_backward(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool = False,
    implementation: SourcePushW13BackwardImplementation = SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE,
    tiled_block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    return_x_expert_major: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13BackwardOutput:
    """Run source-push W13 backward steps 6-7.

    The default is the readable JAX reference. The tiled implementation is the
    smallest production-aligned bridge: it rematerializes source-major ``x`` a
    tile at a time and accumulates bounded ``dx``/``dw13`` matmul tiles without
    materializing route-wise or all-expert ``x`` intermediates. The Pallas/MGPU
    entry remains explicit-only while its W13 matmuls are still being hardened.
    """

    if implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE:
        return source_push_w13_backward_reference(
            x,
            d_h,
            w13,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            use_exact_expert_major=use_exact_expert_major,
        )
    if implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED:
        return source_push_w13_backward_tiled_reference(
            x,
            d_h,
            w13,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            use_exact_expert_major=use_exact_expert_major,
            block_sizes=tiled_block_sizes,
            return_x_expert_major=return_x_expert_major,
        )
    if implementation == SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU:
        return _source_push_w13_backward_pallas_mgpu(
            x,
            d_h,
            w13,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            use_exact_expert_major=use_exact_expert_major,
            mesh=mesh,
        )
    raise ValueError(
        "source-push W13 backward implementation must be one of "
        f"{(SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_REFERENCE, SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_TILED, SOURCE_PUSH_W13_BACKWARD_IMPLEMENTATION_PALLAS_MGPU)}, "
        f"got {implementation!r}"
    )


def source_push_w13_backward_tiled_reference(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool = False,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    return_x_expert_major: bool = False,
) -> SourcePushW13BackwardOutput:
    """Bounded JAX W13 backward that rematerializes ``x`` per matmul tile.

    This is deliberately a shape-tiled reference, not a Pallas replacement. It
    keeps the target production boundary honest by avoiding the monolithic
    ``[Dst, rows, D]`` x-remat temporary unless ``return_x_expert_major`` is set
    for diagnostics.
    """

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    flat_row_map = source_push_w13_flat_row_map_jax(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        hidden_rows_per_rank=d_h.shape[1],
        use_exact_expert_major=use_exact_expert_major,
    )
    _validate_w13_backward_tiled_request(d_h, w13, flat_row_map, block_sizes)
    dx_expert_major, dw13 = _source_push_w13_backward_tiled_from_flat_row_map(
        x,
        d_h,
        w13,
        flat_row_map,
        block_sizes=block_sizes,
    )
    if return_x_expert_major:
        x_expert_major = source_push_x_to_w13_rows_reference(
            x,
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            hidden_rows_per_rank=d_h.shape[1],
            use_exact_expert_major=use_exact_expert_major,
        )
    else:
        x_expert_major = jnp.zeros((0,), dtype=jnp.float32)
    return SourcePushW13BackwardOutput(
        x_expert_major=x_expert_major,
        dx_expert_major=dx_expert_major,
        dw13=dw13,
    )


def _source_push_w13_backward_tiled_from_flat_row_map(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    flat_row_map: SourcePushW13FlatRowMap,
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> tuple[Float[Array, "Dst rows D"], Float[Array, "Dst E D twoI"]]:
    dst_count, hidden_rows_per_rank, output_dim = d_h.shape
    hidden_dim = w13.shape[-2]
    row_tiles = hidden_rows_per_rank // block_sizes.row_block
    hidden_tiles = hidden_dim // block_sizes.hidden_block
    output_tiles = output_dim // block_sizes.output_block

    dx_zero = jnp.zeros(d_h.shape[:2] + (hidden_dim,), dtype=jnp.float32)
    dw13_zero = jnp.zeros(w13.shape, dtype=jnp.float32)
    row_offsets = jnp.arange(block_sizes.row_block, dtype=jnp.int32)
    hidden_offsets = jnp.arange(block_sizes.hidden_block, dtype=jnp.int32)
    output_offsets = jnp.arange(block_sizes.output_block, dtype=jnp.int32)

    def dst_body(dst: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return lax.fori_loop(
            0, hidden_tiles, lambda hidden_tile, hidden_carry: hidden_body(dst, hidden_tile, hidden_carry), carry
        )

    def hidden_body(dst: int, hidden_tile: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return lax.fori_loop(
            0,
            output_tiles,
            lambda output_tile, output_carry: output_body(dst, hidden_tile, output_tile, output_carry),
            carry,
        )

    def output_body(
        dst: int,
        hidden_tile: int,
        output_tile: int,
        carry: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        return lax.fori_loop(
            0,
            row_tiles,
            lambda row_tile, row_carry: row_body(dst, hidden_tile, output_tile, row_tile, row_carry),
            carry,
        )

    def row_body(
        dst: int,
        hidden_tile: int,
        output_tile: int,
        row_tile: int,
        carry: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        dx, dw13 = carry
        row_start = row_tile * block_sizes.row_block
        rows = row_start + row_offsets
        hidden_start = hidden_tile * block_sizes.hidden_block
        hidden = hidden_start + hidden_offsets
        output_start = output_tile * block_sizes.output_block
        output = output_start + output_offsets

        valid = flat_row_map.valid[dst, rows]
        valid_f = valid.astype(jnp.float32)
        src = jnp.where(valid, flat_row_map.src[dst, rows], 0)
        token = jnp.where(valid, flat_row_map.token[dst, rows], 0)
        expert = jnp.where(valid, flat_row_map.expert[dst, rows], 0)
        x_tile = x[src[:, None], token[:, None], hidden[None, :]].astype(jnp.float32) * valid_f[:, None]
        d_h_tile = d_h[dst, rows[:, None], output[None, :]].astype(jnp.float32) * valid_f[:, None]
        w13_tile = w13[
            dst,
            expert[:, None, None],
            hidden[None, :, None],
            output[None, None, :],
        ].astype(jnp.float32)

        dx_update = jnp.einsum("ro,rho->rh", d_h_tile, w13_tile)
        dx_old = lax.dynamic_slice(
            dx,
            (dst, row_start, hidden_start),
            (1, block_sizes.row_block, block_sizes.hidden_block),
        )[0]
        dx = lax.dynamic_update_slice(
            dx,
            (dx_old + dx_update)[None, :, :],
            (dst, row_start, hidden_start),
        )

        dw13_update = x_tile[:, :, None] * d_h_tile[:, None, :]
        dw13 = dw13.at[
            dst,
            expert[:, None, None],
            hidden[None, :, None],
            output[None, None, :],
        ].add(dw13_update)
        return dx, dw13

    return lax.fori_loop(0, dst_count, dst_body, (dx_zero, dw13_zero))


def _source_push_w13_backward_tiled_from_expert_blocks(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> tuple[Float[Array, "Dst E C D"], Float[Array, "Dst E D twoI"]]:
    dst_count, local_experts, expert_capacity, output_dim = d_h.shape
    hidden_dim = w13.shape[-2]
    padded_capacity = _round_up_to_multiple(expert_capacity, block_sizes.row_block)
    if padded_capacity != expert_capacity:
        row_pad = ((0, 0), (0, 0), (0, padded_capacity - expert_capacity))
        d_h = jnp.pad(d_h, (*row_pad, (0, 0)), constant_values=0)
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    row_tiles = padded_capacity // block_sizes.row_block
    hidden_tiles = hidden_dim // block_sizes.hidden_block
    output_tiles = output_dim // block_sizes.output_block

    dx_zero = jnp.zeros(d_h.shape[:3] + (hidden_dim,), dtype=jnp.float32)
    dw13_zero = jnp.zeros(w13.shape, dtype=jnp.float32)
    row_offsets = jnp.arange(block_sizes.row_block, dtype=jnp.int32)
    hidden_offsets = jnp.arange(block_sizes.hidden_block, dtype=jnp.int32)
    output_offsets = jnp.arange(block_sizes.output_block, dtype=jnp.int32)

    def dst_body(dst: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return lax.fori_loop(
            0,
            local_experts,
            lambda expert, expert_carry: expert_body(dst, expert, expert_carry),
            carry,
        )

    def expert_body(dst: int, expert: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return lax.fori_loop(
            0,
            hidden_tiles,
            lambda hidden_tile, hidden_carry: hidden_body(dst, expert, hidden_tile, hidden_carry),
            carry,
        )

    def hidden_body(dst: int, expert: int, hidden_tile: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        return lax.fori_loop(
            0,
            output_tiles,
            lambda output_tile, output_carry: output_body(dst, expert, hidden_tile, output_tile, output_carry),
            carry,
        )

    def output_body(
        dst: int,
        expert: int,
        hidden_tile: int,
        output_tile: int,
        carry: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        return lax.fori_loop(
            0,
            row_tiles,
            lambda row_tile, row_carry: row_body(dst, expert, hidden_tile, output_tile, row_tile, row_carry),
            carry,
        )

    def row_body(
        dst: int,
        expert: int,
        hidden_tile: int,
        output_tile: int,
        row_tile: int,
        carry: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        dx, dw13 = carry
        row_start = row_tile * block_sizes.row_block
        rows = row_start + row_offsets
        hidden_start = hidden_tile * block_sizes.hidden_block
        hidden = hidden_start + hidden_offsets
        output_start = output_tile * block_sizes.output_block
        output = output_start + output_offsets

        valid = valid_by_expert[dst, expert, rows]
        valid_f = valid.astype(jnp.float32)
        src = jnp.where(valid, source_rank_by_expert[dst, expert, rows], 0)
        token = jnp.where(valid, token_id_by_expert[dst, expert, rows], 0)
        x_tile = x[src[:, None], token[:, None], hidden[None, :]].astype(jnp.float32) * valid_f[:, None]
        d_h_tile = d_h[dst, expert, rows[:, None], output[None, :]].astype(jnp.float32) * valid_f[:, None]
        w13_tile = w13[dst, expert, hidden[:, None], output[None, :]].astype(jnp.float32)

        dx_update = jnp.einsum("mo,ho->mh", d_h_tile, w13_tile)
        dx_old = lax.dynamic_slice(
            dx,
            (dst, expert, row_start, hidden_start),
            (1, 1, block_sizes.row_block, block_sizes.hidden_block),
        )[0, 0]
        dx = lax.dynamic_update_slice(
            dx,
            (dx_old + dx_update)[None, None, :, :],
            (dst, expert, row_start, hidden_start),
        )

        dw13_update = jnp.einsum("mh,mo->ho", x_tile, d_h_tile)
        dw13_old = lax.dynamic_slice(
            dw13,
            (dst, expert, hidden_start, output_start),
            (1, 1, block_sizes.hidden_block, block_sizes.output_block),
        )[0, 0]
        dw13 = lax.dynamic_update_slice(
            dw13,
            (dw13_old + dw13_update)[None, None, :, :],
            (dst, expert, hidden_start, output_start),
        )
        return dx, dw13

    dx, dw13 = lax.fori_loop(0, dst_count, dst_body, (dx_zero, dw13_zero))
    return dx[:, :, :expert_capacity, :], dw13


def _source_push_w13_dw13_tiled_from_expert_blocks(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> Float[Array, "Dst E D twoI"]:
    dst_count, local_experts, expert_capacity, output_dim = d_h.shape
    hidden_dim = x.shape[-1]
    padded_capacity = _round_up_to_multiple(expert_capacity, block_sizes.row_block)
    if padded_capacity != expert_capacity:
        row_pad = ((0, 0), (0, 0), (0, padded_capacity - expert_capacity))
        d_h = jnp.pad(d_h, (*row_pad, (0, 0)), constant_values=0)
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    row_tiles = padded_capacity // block_sizes.row_block
    hidden_tiles = hidden_dim // block_sizes.hidden_block
    output_tiles = output_dim // block_sizes.output_block

    dw13_zero = jnp.zeros((dst_count, local_experts, hidden_dim, output_dim), dtype=jnp.float32)
    row_offsets = jnp.arange(block_sizes.row_block, dtype=jnp.int32)
    hidden_offsets = jnp.arange(block_sizes.hidden_block, dtype=jnp.int32)
    output_offsets = jnp.arange(block_sizes.output_block, dtype=jnp.int32)

    def dst_body(dst: int, dw13: Array) -> Array:
        return lax.fori_loop(0, local_experts, lambda expert, expert_dw13: expert_body(dst, expert, expert_dw13), dw13)

    def expert_body(dst: int, expert: int, dw13: Array) -> Array:
        return lax.fori_loop(
            0,
            hidden_tiles,
            lambda hidden_tile, hidden_dw13: hidden_body(dst, expert, hidden_tile, hidden_dw13),
            dw13,
        )

    def hidden_body(dst: int, expert: int, hidden_tile: int, dw13: Array) -> Array:
        return lax.fori_loop(
            0,
            output_tiles,
            lambda output_tile, output_dw13: output_body(dst, expert, hidden_tile, output_tile, output_dw13),
            dw13,
        )

    def output_body(dst: int, expert: int, hidden_tile: int, output_tile: int, dw13: Array) -> Array:
        return lax.fori_loop(
            0,
            row_tiles,
            lambda row_tile, row_dw13: row_body(dst, expert, hidden_tile, output_tile, row_tile, row_dw13),
            dw13,
        )

    def row_body(dst: int, expert: int, hidden_tile: int, output_tile: int, row_tile: int, dw13: Array) -> Array:
        row_start = row_tile * block_sizes.row_block
        rows = row_start + row_offsets
        hidden_start = hidden_tile * block_sizes.hidden_block
        hidden = hidden_start + hidden_offsets
        output_start = output_tile * block_sizes.output_block
        output = output_start + output_offsets

        valid = valid_by_expert[dst, expert, rows]
        valid_f = valid.astype(jnp.float32)
        src = jnp.where(valid, source_rank_by_expert[dst, expert, rows], 0)
        token = jnp.where(valid, token_id_by_expert[dst, expert, rows], 0)
        x_tile = x[src[:, None], token[:, None], hidden[None, :]].astype(jnp.float32) * valid_f[:, None]
        d_h_tile = d_h[dst, expert, rows[:, None], output[None, :]].astype(jnp.float32) * valid_f[:, None]

        dw13_update = jnp.einsum("mh,mo->ho", x_tile, d_h_tile)
        dw13_old = lax.dynamic_slice(
            dw13,
            (dst, expert, hidden_start, output_start),
            (1, 1, block_sizes.hidden_block, block_sizes.output_block),
        )[0, 0]
        return lax.dynamic_update_slice(
            dw13,
            (dw13_old + dw13_update)[None, None, :, :],
            (dst, expert, hidden_start, output_start),
        )

    return lax.fori_loop(0, dst_count, dst_body, dw13_zero)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _validate_w13_backward_tiled_request(
    d_h: Array,
    w13: Array,
    flat_row_map: SourcePushW13FlatRowMap,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> None:
    if d_h.ndim != 3:
        raise ValueError(f"d_h must have shape [dst, rows, twoI], got {d_h.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, D, twoI], got {w13.shape}")
    if flat_row_map.valid.shape != d_h.shape[:2]:
        raise ValueError(f"flat row valid shape {flat_row_map.valid.shape} must match d_h rows {d_h.shape[:2]}")
    if flat_row_map.src.shape != flat_row_map.valid.shape:
        raise ValueError(f"flat row src shape {flat_row_map.src.shape} must match valid {flat_row_map.valid.shape}")
    if flat_row_map.token.shape != flat_row_map.valid.shape:
        raise ValueError(
            f"flat row token shape {flat_row_map.token.shape} must match valid {flat_row_map.valid.shape}"
        )
    if w13.shape[0] != d_h.shape[0]:
        raise ValueError(f"w13 destination count {w13.shape[0]} must match d_h {d_h.shape[0]}")
    if w13.shape[-1] != d_h.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match d_h {d_h.shape[-1]}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if d_h.shape[1] % block_sizes.row_block:
        raise ValueError(f"d_h rows {d_h.shape[1]} must be divisible by row_block={block_sizes.row_block}")
    if w13.shape[-2] % block_sizes.hidden_block:
        raise ValueError(
            f"w13 hidden dim {w13.shape[-2]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )
    if d_h.shape[-1] % block_sizes.output_block:
        raise ValueError(
            f"d_h output dim {d_h.shape[-1]} must be divisible by output_block={block_sizes.output_block}"
        )


def _validate_w13_backward_expert_blocks_tiled_request(
    x: Array,
    d_h: Array,
    w13: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    valid_by_expert: Array,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, D], got {x.shape}")
    if d_h.ndim != 4:
        raise ValueError(f"d_h must have shape [dst, expert, capacity, twoI], got {d_h.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, D, twoI], got {w13.shape}")
    if source_rank_by_expert.shape != d_h.shape[:3]:
        raise ValueError(
            f"source_rank_by_expert shape {source_rank_by_expert.shape} must match d_h blocks {d_h.shape[:3]}"
        )
    if token_id_by_expert.shape != d_h.shape[:3]:
        raise ValueError(f"token_id_by_expert shape {token_id_by_expert.shape} must match d_h blocks {d_h.shape[:3]}")
    if valid_by_expert.shape != d_h.shape[:3]:
        raise ValueError(f"valid_by_expert shape {valid_by_expert.shape} must match d_h blocks {d_h.shape[:3]}")
    if w13.shape[:2] != d_h.shape[:2]:
        raise ValueError(f"w13 destination/expert shape {w13.shape[:2]} must match d_h {d_h.shape[:2]}")
    if w13.shape[-2] != x.shape[-1]:
        raise ValueError(f"w13 hidden dim {w13.shape[-2]} must match x hidden dim {x.shape[-1]}")
    if w13.shape[-1] != d_h.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match d_h output dim {d_h.shape[-1]}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"x hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")
    if d_h.shape[-1] % block_sizes.output_block:
        raise ValueError(
            f"d_h output dim {d_h.shape[-1]} must be divisible by output_block={block_sizes.output_block}"
        )


def _validate_w13_dw13_source_gather_request(
    x: Array,
    d_h: Array,
    source_rank_by_expert: Array,
    token_id_by_expert: Array,
    valid_by_expert: Array,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, D], got {x.shape}")
    if d_h.ndim != 4:
        raise ValueError(f"d_h must have shape [dst, expert, capacity, twoI], got {d_h.shape}")
    if source_rank_by_expert.shape != d_h.shape[:3]:
        raise ValueError(
            f"source_rank_by_expert shape {source_rank_by_expert.shape} must match d_h blocks {d_h.shape[:3]}"
        )
    if token_id_by_expert.shape != d_h.shape[:3]:
        raise ValueError(f"token_id_by_expert shape {token_id_by_expert.shape} must match d_h blocks {d_h.shape[:3]}")
    if valid_by_expert.shape != d_h.shape[:3]:
        raise ValueError(f"valid_by_expert shape {valid_by_expert.shape} must match d_h blocks {d_h.shape[:3]}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"x hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")
    if d_h.shape[-1] % block_sizes.output_block:
        raise ValueError(
            f"d_h output dim {d_h.shape[-1]} must be divisible by output_block={block_sizes.output_block}"
        )


def _source_push_w13_backward_pallas_mgpu(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool = False,
    block_sizes: SourcePushXToW13RowsPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13BackwardOutput:
    """Staged MGPU W13 backward.

    This removes the route-expanded all-source x gather from the W13 backward
    boundary by rematerializing directly into flat H rows with Pallas. The
    remaining matmul gradients are intentionally still JAX over flat rows; full
    WGMMA lowering for ``dx`` and ``dw13`` is the next performance step.
    """

    flat_row_map = source_push_w13_flat_row_map_jax(
        plan,
        send_meta,
        expert_base,
        src_base_by_expert,
        hidden_rows_per_rank=d_h.shape[1],
        use_exact_expert_major=use_exact_expert_major,
    )
    x_expert_major = _source_push_x_to_w13_rows_pallas_mgpu(
        x,
        flat_row_map,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    if use_exact_expert_major:
        dx_expert_major = _source_push_w13_dx_exact_expert_major_pallas_mgpu(
            d_h,
            w13,
            flat_row_map.valid,
            flat_row_map.expert,
            interpret=interpret,
            mesh=mesh,
        )
        dw13 = _source_push_w13_dw13_exact_expert_major_pallas_mgpu(
            x_expert_major,
            d_h,
            w13,
            flat_row_map.valid,
            flat_row_map.expert,
            interpret=interpret,
            mesh=mesh,
        )
        return SourcePushW13BackwardOutput(
            x_expert_major=x_expert_major.astype(jnp.float32) * flat_row_map.valid.astype(jnp.float32)[..., None],
            dx_expert_major=dx_expert_major,
            dw13=dw13,
        )
    return _source_push_w13_backward_from_flat_x_rows_reference(
        x_expert_major,
        d_h,
        w13,
        flat_row_map,
    )


def _source_push_w13_backward_from_flat_x_rows_reference(
    x_expert_major: Float[Array, "Dst rows D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    flat_row_map: SourcePushW13FlatRowMap,
) -> SourcePushW13BackwardOutput:
    """Reference W13 matmul gradients after flat-row x rematerialization."""

    valid_f = flat_row_map.valid.astype(jnp.float32)
    x_rows = x_expert_major.astype(jnp.float32) * valid_f[..., None]
    d_h_rows = d_h.astype(jnp.float32) * valid_f[..., None]
    safe_expert = jnp.where(flat_row_map.valid, flat_row_map.expert, 0)
    dst_index = jnp.arange(w13.shape[0], dtype=jnp.int32)[:, None]
    w13_rows = w13.at[dst_index, safe_expert].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    w13_rows = w13_rows.astype(jnp.float32)

    dx_expert_major = jnp.einsum("dro,drho->drh", d_h_rows, w13_rows) * valid_f[..., None]
    dw13_rows = jnp.einsum("drh,dro->drho", x_rows, d_h_rows) * valid_f[..., None, None]
    dw13 = jnp.zeros(w13.shape, dtype=jnp.float32)
    dw13 = dw13.at[dst_index, safe_expert].add(
        dw13_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )
    return SourcePushW13BackwardOutput(x_expert_major=x_rows, dx_expert_major=dx_expert_major, dw13=dw13)


def _source_push_w13_dw13_from_flat_x_rows_reference(
    x_expert_major: Float[Array, "Dst rows D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    flat_row_map: SourcePushW13FlatRowMap,
) -> Float[Array, "Dst E D twoI"]:
    valid_f = flat_row_map.valid.astype(jnp.float32)
    x_rows = x_expert_major.astype(jnp.float32) * valid_f[..., None]
    d_h_rows = d_h.astype(jnp.float32) * valid_f[..., None]
    safe_expert = jnp.where(flat_row_map.valid, flat_row_map.expert, 0)
    dst_index = jnp.arange(w13.shape[0], dtype=jnp.int32)[:, None]
    dw13_rows = jnp.einsum("drh,dro->drho", x_rows, d_h_rows) * valid_f[..., None, None]
    dw13 = jnp.zeros(w13.shape, dtype=jnp.float32)
    return dw13.at[dst_index, safe_expert].add(
        dw13_rows,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None),
    )


def _source_push_w13_dw13_exact_expert_major_pallas_mgpu(
    x_expert_major: Float[Array, "Dst rows D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    row_expert: Int[Array, "Dst rows"],
    *,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst E D twoI"]:
    """Compute W13 weight gradients for exact expert-major rows."""

    if interpret:
        return _source_push_w13_dw13_from_flat_x_rows_reference(
            x_expert_major,
            d_h,
            w13,
            SourcePushW13FlatRowMap(
                src=jnp.zeros_like(row_expert),
                token=jnp.zeros_like(row_expert),
                expert=row_expert,
                valid=valid,
            ),
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU W13 dw13 backward requires a GPU backend; use the reference on CPU")
    if mesh is not None:
        return _source_push_w13_dw13_exact_expert_major_sharded_pallas_call(
            mesh,
            x_expert_major,
            d_h,
            w13,
            valid,
            row_block=DEFAULT_W13_DX_ROW_BLOCK,
            hidden_block=DEFAULT_W13_DX_HIDDEN_BLOCK,
            output_block=DEFAULT_W13_DX_OUTPUT_BLOCK,
        )
    return _source_push_w13_dw13_exact_expert_major_pallas_call(
        x_expert_major,
        d_h,
        w13,
        valid,
        row_block=DEFAULT_W13_DX_ROW_BLOCK,
        hidden_block=DEFAULT_W13_DX_HIDDEN_BLOCK,
        output_block=DEFAULT_W13_DX_OUTPUT_BLOCK,
    )


def _source_push_w13_dw13_exact_expert_major_sharded_pallas_call(
    mesh: Mesh | AbstractMesh,
    x_expert_major: Float[Array, "Dst rows D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst E D twoI"]:
    def local_fn(
        x_local: Float[Array, "1 rows D"],
        d_h_local: Float[Array, "1 rows twoI"],
        w13_local: Float[Array, "1 E D twoI"],
        valid_local: Bool[Array, "1 rows"],
    ) -> Float[Array, "1 E D twoI"]:
        return _source_push_w13_dw13_exact_expert_major_pallas_call(
            x_local,
            d_h_local,
            w13_local,
            valid_local,
            row_block=row_block,
            hidden_block=hidden_block,
            output_block=output_block,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(x_expert_major, d_h, w13, valid)


def _source_push_w13_dw13_exact_expert_major_pallas_call(
    x_expert_major: Float[Array, "Dst rows D"],
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst E D twoI"]:
    _validate_w13_dx_exact_expert_major_pallas_request(
        d_h,
        w13,
        valid,
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
    )
    if x_expert_major.shape != d_h.shape[:2] + (w13.shape[-2],):
        raise ValueError(
            f"x_expert_major shape {x_expert_major.shape} must be {(d_h.shape[0], d_h.shape[1], w13.shape[-2])}"
        )
    output_shape = jax.ShapeDtypeStruct(w13.shape, jnp.float32)
    output_zero = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    valid_i = valid.astype(jnp.int32)
    rows_per_expert = d_h.shape[1] // w13.shape[1]
    expert_row_tiles = rows_per_expert // row_block
    kernel = _make_source_push_w13_dw13_exact_expert_major_kernel(row_block=row_block)
    in_specs, out_spec = _source_push_w13_dw13_exact_expert_major_block_specs(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        expert_row_tiles=expert_row_tiles,
    )
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=output_shape,
        grid=(
            x_expert_major.shape[0],
            w13.shape[1],
            w13.shape[-2] // hidden_block,
            w13.shape[-1] // output_block,
            expert_row_tiles,
        ),
        input_output_aliases={4: 0},
        name="source_push_w13_dw13_exact_expert_major_pallas_mgpu",
    )(x_expert_major, d_h, w13, valid_i, output_zero)


def _source_push_w13_dw13_exact_expert_major_block_specs(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    expert_row_tiles: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    x_spec = pl.BlockSpec(
        (None, row_block, hidden_block),
        lambda dst, expert, hidden_tile, _output_tile, row_tile: (
            dst,
            expert * expert_row_tiles + row_tile,
            hidden_tile,
        ),
    )
    d_h_spec = pl.BlockSpec(
        (None, row_block, output_block),
        lambda dst, expert, _hidden_tile, output_tile, row_tile: (
            dst,
            expert * expert_row_tiles + row_tile,
            output_tile,
        ),
    )
    w13_spec = pl.BlockSpec(
        (None, None, hidden_block, output_block),
        lambda dst, expert, hidden_tile, output_tile, _row_tile: (dst, expert, hidden_tile, output_tile),
    )
    valid_spec = pl.BlockSpec(
        (None, row_block),
        lambda dst, expert, _hidden_tile, _output_tile, row_tile: (dst, expert * expert_row_tiles + row_tile),
    )
    zero_spec = pl.BlockSpec(
        (None, None, hidden_block, output_block),
        lambda dst, expert, hidden_tile, output_tile, _row_tile: (dst, expert, hidden_tile, output_tile),
    )
    out_spec = pl.BlockSpec(
        (None, None, hidden_block, output_block),
        lambda dst, expert, hidden_tile, output_tile, _row_tile: (dst, expert, hidden_tile, output_tile),
    )
    return (x_spec, d_h_spec, w13_spec, valid_spec, zero_spec), out_spec


def _make_source_push_w13_dw13_exact_expert_major_kernel(*, row_block: int):
    def kernel(
        x_ref: Float[pl.Ref, "M D"],
        d_h_ref: Float[pl.Ref, "M O"],
        _w13_ref: Float[pl.Ref, "D O"],
        valid_ref: Int[pl.Ref, "M"],
        _zero_ref: Float[pl.Ref, "D O"],
        dw13_ref: Float[pl.Ref, "D O"],
    ) -> None:
        valid_f = valid_ref[pl.ds(0, row_block)].astype(jnp.float32)
        x_tile = x_ref[:, :].astype(jnp.float32) * valid_f[:, None]
        d_h_tile = d_h_ref[:, :].astype(jnp.float32) * valid_f[:, None]
        acc = pl.dot(x_tile, d_h_tile, trans_a=True)
        mgpu.atomic_add(dw13_ref, acc)

    return kernel


def _source_push_w13_dx_exact_expert_major_pallas_mgpu(
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    row_expert: Int[Array, "Dst rows"],
    *,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst rows D"]:
    """Compute flat-row W13 ``dx`` for exact expert-major rows."""

    if interpret:
        return _source_push_w13_dx_exact_expert_major_reference(d_h, w13, valid, row_expert)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU W13 dx backward requires a GPU backend; use the reference on CPU")
    if mesh is not None:
        return _source_push_w13_dx_exact_expert_major_sharded_pallas_call(
            mesh,
            d_h,
            w13,
            valid,
            row_block=DEFAULT_W13_DX_ROW_BLOCK,
            hidden_block=DEFAULT_W13_DX_HIDDEN_BLOCK,
            output_block=DEFAULT_W13_DX_OUTPUT_BLOCK,
        )
    return _source_push_w13_dx_exact_expert_major_pallas_call(
        d_h,
        w13,
        valid,
        row_block=DEFAULT_W13_DX_ROW_BLOCK,
        hidden_block=DEFAULT_W13_DX_HIDDEN_BLOCK,
        output_block=DEFAULT_W13_DX_OUTPUT_BLOCK,
    )


def _source_push_w13_dx_exact_expert_major_sharded_pallas_call(
    mesh: Mesh | AbstractMesh,
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst rows D"]:
    def local_fn(
        d_h_local: Float[Array, "1 rows twoI"],
        w13_local: Float[Array, "1 E D twoI"],
        valid_local: Bool[Array, "1 rows"],
    ) -> Float[Array, "1 rows D"]:
        return _source_push_w13_dx_exact_expert_major_pallas_call(
            d_h_local,
            w13_local,
            valid_local,
            row_block=row_block,
            hidden_block=hidden_block,
            output_block=output_block,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None),
        check_vma=False,
    )(d_h, w13, valid)


def _source_push_w13_dx_exact_expert_major_pallas_call(
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst rows D"]:
    _validate_w13_dx_exact_expert_major_pallas_request(
        d_h,
        w13,
        valid,
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
    )
    output_shape = jax.ShapeDtypeStruct(d_h.shape[:2] + (w13.shape[-2],), jnp.float32)
    output_zero = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    valid_i = valid.astype(jnp.int32)
    rows_per_expert = d_h.shape[1] // w13.shape[1]
    expert_row_tiles = rows_per_expert // row_block
    kernel = _make_source_push_w13_dx_exact_expert_major_kernel(row_block=row_block)
    in_specs, out_spec = _source_push_w13_dx_exact_expert_major_block_specs(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        expert_row_tiles=expert_row_tiles,
    )
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=output_shape,
        grid=(
            d_h.shape[0],
            d_h.shape[1] // row_block,
            w13.shape[-2] // hidden_block,
            d_h.shape[-1] // output_block,
        ),
        input_output_aliases={3: 0},
        name="source_push_w13_dx_exact_expert_major_pallas_mgpu",
    )(d_h, w13, valid_i, output_zero)


def _source_push_w13_dx_exact_expert_major_block_specs(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    expert_row_tiles: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    d_h_spec = pl.BlockSpec(
        (None, row_block, output_block),
        lambda dst, row_tile, _hidden_tile, output_tile: (dst, row_tile, output_tile),
    )
    w13_spec = pl.BlockSpec(
        (None, None, hidden_block, output_block),
        lambda dst, row_tile, hidden_tile, output_tile: (
            dst,
            row_tile // expert_row_tiles,
            hidden_tile,
            output_tile,
        ),
    )
    valid_spec = pl.BlockSpec(
        (None, row_block),
        lambda dst, row_tile, _hidden_tile, _output_tile: (dst, row_tile),
    )
    zero_spec = pl.BlockSpec(
        (None, row_block, hidden_block),
        lambda dst, row_tile, hidden_tile, _output_tile: (dst, row_tile, hidden_tile),
    )
    out_spec = pl.BlockSpec(
        (None, row_block, hidden_block),
        lambda dst, row_tile, hidden_tile, _output_tile: (dst, row_tile, hidden_tile),
    )
    return (d_h_spec, w13_spec, valid_spec, zero_spec), out_spec


def _make_source_push_w13_dx_exact_expert_major_kernel(*, row_block: int):
    def kernel(
        d_h_ref: Float[pl.Ref, "M O"],
        w13_ref: Float[pl.Ref, "D O"],
        valid_ref: Int[pl.Ref, "M"],
        _zero_ref: Float[pl.Ref, "M D"],
        dx_ref: Float[pl.Ref, "M D"],
    ) -> None:
        valid_f = valid_ref[pl.ds(0, row_block)].astype(jnp.float32)
        d_h_tile = d_h_ref[:, :].astype(jnp.float32) * valid_f[:, None]
        w13_tile = w13_ref[:, :].astype(jnp.float32)
        acc = pl.dot(d_h_tile, w13_tile, trans_b=True)
        mgpu.atomic_add(dx_ref, acc)

    return kernel


def _source_push_w13_dx_exact_expert_major_reference(
    d_h: Float[Array, "Dst rows twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst rows"],
    row_expert: Int[Array, "Dst rows"],
) -> Float[Array, "Dst rows D"]:
    valid_f = valid.astype(jnp.float32)
    dst = jnp.arange(d_h.shape[0], dtype=jnp.int32)[:, None]
    safe_expert = jnp.where(valid, row_expert, 0)
    w13_rows = w13.at[dst, safe_expert].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    return jnp.einsum("dro,drho->drh", d_h.astype(jnp.float32) * valid_f[..., None], w13_rows.astype(jnp.float32))


def _validate_w13_dx_exact_expert_major_pallas_request(
    d_h: Array,
    w13: Array,
    valid: Array,
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> None:
    if d_h.ndim != 3:
        raise ValueError(f"d_h must have shape [dst, rows, twoI], got {d_h.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, D, twoI], got {w13.shape}")
    if valid.shape != d_h.shape[:2]:
        raise ValueError(f"valid shape {valid.shape} must match d_h rows {d_h.shape[:2]}")
    if w13.shape[0] != d_h.shape[0]:
        raise ValueError(f"w13 destination count {w13.shape[0]} must match d_h {d_h.shape[0]}")
    if w13.shape[-1] != d_h.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match d_h {d_h.shape[-1]}")
    if d_h.shape[1] % w13.shape[1]:
        raise ValueError(f"d_h rows {d_h.shape[1]} must be divisible by local experts {w13.shape[1]}")
    rows_per_expert = d_h.shape[1] // w13.shape[1]
    if rows_per_expert % row_block:
        raise ValueError(f"rows_per_expert {rows_per_expert} must be divisible by row_block={row_block}")
    if d_h.shape[-1] % output_block:
        raise ValueError(f"d_h output dim {d_h.shape[-1]} must be divisible by output_block={output_block}")
    if w13.shape[-2] % hidden_block:
        raise ValueError(f"w13 hidden dim {w13.shape[-2]} must be divisible by hidden_block={hidden_block}")


def _source_push_w13_dx_expert_blocks_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst E C D"]:
    _validate_w13_expert_blocks_pallas_request(
        d_h,
        w13,
        valid,
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
    )
    kernel = _make_source_push_w13_dx_expert_blocks_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_h.shape[1],
        rows=d_h.shape[2],
        hidden_dim=w13.shape[-2],
        output_dim=d_h.shape[-1],
    )

    def local_fn(
        d_h_local: Float[Array, "1 E C twoI"],
        w13_local: Float[Array, "1 E D twoI"],
        valid_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 E C D"]:
        _ = valid_local
        return kernel(d_h_local[0], w13_local[0])[None, ...]

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
    )(d_h, w13, valid)


def _source_push_w13_dw13_expert_blocks_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    x_expert_major: Float[Array, "Dst E C D"],
    d_h: Float[Array, "Dst E C twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst E D twoI"]:
    if x_expert_major.ndim != 4:
        raise ValueError(f"x_expert_major must have shape [dst, expert, capacity, D], got {x_expert_major.shape}")
    output_shape = d_h.shape[:2] + (x_expert_major.shape[-1], d_h.shape[-1])
    if x_expert_major.shape[:3] != d_h.shape[:3]:
        raise ValueError(f"x_expert_major leading shape {x_expert_major.shape[:3]} must match d_h {d_h.shape[:3]}")
    _validate_w13_expert_blocks_pallas_request(
        d_h,
        jax.ShapeDtypeStruct(output_shape, d_h.dtype),
        valid,
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
    )
    kernel = _make_source_push_w13_dw13_expert_blocks_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_h.shape[1],
        rows=d_h.shape[2],
        hidden_dim=x_expert_major.shape[-1],
        output_dim=d_h.shape[-1],
    )

    def local_fn(
        x_local: Float[Array, "1 E C D"],
        d_h_local: Float[Array, "1 E C twoI"],
        valid_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 E D twoI"]:
        return kernel(x_local[0], d_h_local[0], valid_local[0])[None, ...]

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
    )(x_expert_major, d_h, valid)


def _make_source_push_w13_dx_expert_blocks_mgpu_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
    output_dim: int,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = output_dim // output_block

    def body(
        d_h_ref: Float[pl.Ref, "E C O"],
        w13_ref: Float[pl.Ref, "E D O"],
        dx_ref: Float[pl.Ref, "E C D"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(d_h_smem, w13_smem, ready_barrier) -> None:
                @pl.loop(0, output_tiles)
                def _output_loop(output_tile) -> None:
                    output_start = output_tile * output_block
                    mgpu.copy_gmem_to_smem(
                        d_h_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(output_start, output_block),
                        ],
                        d_h_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[
                            expert,
                            pl.ds(hidden_start, hidden_block),
                            pl.ds(output_start, output_block),
                        ],
                        w13_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, d_h_smem, mgpu.transpose_ref(w13_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                d_h_smem=_w13_wgmma_smem((row_block, output_block), d_h_ref.dtype),
                w13_smem=_w13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((row_block, hidden_block)))
        dx_ref[
            expert,
            pl.ds(row_start, row_block),
            pl.ds(hidden_start, hidden_block),
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


def _make_source_push_w13_dw13_expert_blocks_mgpu_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
    output_dim: int,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = output_dim // output_block

    def body(
        x_ref: Float[pl.Ref, "E C D"],
        d_h_ref: Float[pl.Ref, "E C O"],
        _valid_ref: Bool[pl.Ref, "E C"],
        dw13_ref: Float[pl.Ref, "E D O"],
    ) -> None:
        expert = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        output_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        output_start = output_tile * output_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(x_smem, d_h_smem, ready_barrier) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    mgpu.copy_gmem_to_smem(
                        x_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        x_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_h_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(output_start, output_block),
                        ],
                        d_h_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), d_h_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                x_smem=_w13_wgmma_smem((row_block, hidden_block), x_ref.dtype),
                d_h_smem=_w13_wgmma_smem((row_block, output_block), d_h_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((hidden_block, output_block)))
        dw13_ref[
            expert,
            pl.ds(hidden_start, hidden_block),
            pl.ds(output_start, output_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, hidden_dim, output_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, hidden_tiles, output_tiles),
        grid_names=("expert", "hidden_tile", "output_tile"),
        compiler_params=compiler_params,
    )


def _validate_w13_expert_blocks_pallas_request(
    d_h: Array,
    w13: Array,
    valid: Array,
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> None:
    if d_h.ndim != 4:
        raise ValueError(f"d_h must have shape [dst, expert, capacity, twoI], got {d_h.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, D, twoI], got {w13.shape}")
    if valid.shape != d_h.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match d_h blocks {d_h.shape[:3]}")
    if w13.shape[:2] != d_h.shape[:2]:
        raise ValueError(f"w13 destination/expert shape {w13.shape[:2]} must match d_h {d_h.shape[:2]}")
    if w13.shape[-1] != d_h.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match d_h output dim {d_h.shape[-1]}")
    if d_h.shape[2] % row_block:
        raise ValueError(f"d_h capacity {d_h.shape[2]} must be divisible by row_block={row_block}")
    if w13.shape[-2] % hidden_block:
        raise ValueError(f"w13 hidden dim {w13.shape[-2]} must be divisible by hidden_block={hidden_block}")
    if d_h.shape[-1] % output_block:
        raise ValueError(f"d_h output dim {d_h.shape[-1]} must be divisible by output_block={output_block}")
    _validate_w13_wgmma_smem_shape((row_block, output_block), d_h.dtype)
    _validate_w13_wgmma_smem_shape((hidden_block, output_block), w13.dtype)
    _validate_w13_wgmma_smem_shape((row_block, hidden_block), w13.dtype)


def _w13_wgmma_smem(shape: tuple[int, int], dtype):
    _validate_w13_wgmma_smem_shape(shape, dtype)
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((W13_WGMMA_TILE_M, W13_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize)),
            mgpu.SwizzleTransform(W13_WGMMA_SWIZZLE_BYTES),
        ),
    )


def _validate_w13_wgmma_smem_shape(shape: tuple[int, int], dtype) -> None:
    swizzle_elems = W13_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % W13_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "W13 WGMMA SMEM operands must be divisible by "
            f"tile=({W13_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )


def estimate_source_push_w13_backward_cost(
    *,
    useful_rows_per_rank: int,
    padded_rows_per_rank: int,
    hidden_dim: int,
    intermediate_dim: int,
    x_dtype_bytes: int = 2,
    reference_tflops_per_rank: float = 250.0,
) -> SourcePushW13BackwardCostEstimate:
    """Estimate per-rank W13 backward math and rematerialization traffic.

    ``reference_tflops_per_rank`` is only a normalization point for stage cost
    reporting; it is not a per-kernel performance requirement.
    """

    two_intermediate_dim = 2 * intermediate_dim
    w13_backward_flops = 4 * useful_rows_per_rank * hidden_dim * two_intermediate_dim
    x_remat_bytes = useful_rows_per_rank * hidden_dim * x_dtype_bytes
    x_remat_padded_bytes = padded_rows_per_rank * hidden_dim * x_dtype_bytes
    return SourcePushW13BackwardCostEstimate(
        useful_rows_per_rank=useful_rows_per_rank,
        padded_rows_per_rank=padded_rows_per_rank,
        w13_backward_flops_per_rank=w13_backward_flops,
        x_remat_bytes_per_rank=x_remat_bytes,
        x_remat_padded_bytes_per_rank=x_remat_padded_bytes,
        math_seconds_at_reference_tflops_per_rank=w13_backward_flops / (reference_tflops_per_rank * 1e12),
    )


def _source_push_w13_row_indices(
    plan: SourcePushPlan,
    send_meta: Int[Array, "S Dst Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool,
) -> _SourcePushW13RowIndices:
    valid = jnp.asarray(plan.valid_mask, dtype=jnp.bool_)
    valid_f = valid.astype(jnp.float32)
    source_count, dst_count, entries_per_dst, block_m = valid.shape

    src_entry = jnp.arange(source_count, dtype=jnp.int32)[:, None, None]
    dst_ordinal = jnp.arange(dst_count, dtype=jnp.int32)[None, :, None]
    src_entry = jnp.broadcast_to(src_entry, (source_count, dst_count, entries_per_dst))
    dst_entry = (src_entry + dst_ordinal) % dst_count

    send_meta = jnp.asarray(send_meta, dtype=jnp.int32)
    safe_expert_entry = jnp.maximum(send_meta[..., SOURCE_PUSH_META_LOCAL_EXPERT], 0)
    metadata_row_start = send_meta[..., SOURCE_PUSH_META_LOCAL_ROW_START]
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    base_row = expert_base.at[dst_entry, safe_expert_entry].get()
    if use_exact_expert_major:
        src_base_by_expert = jnp.asarray(src_base_by_expert, dtype=jnp.int32)
        src_base = src_base_by_expert.at[dst_entry, src_entry, safe_expert_entry].get()
        local_row_start = src_base + metadata_row_start
        row_start = base_row + local_row_start
    else:
        local_row_start = metadata_row_start - base_row
        row_start = metadata_row_start

    row_offsets = jnp.arange(block_m, dtype=jnp.int32)[None, None, None, :]
    flat_row = row_start[..., None] + row_offsets
    local_row = local_row_start[..., None] + row_offsets
    src = jnp.broadcast_to(src_entry[..., None], valid.shape)
    dst = jnp.broadcast_to(dst_entry[..., None], valid.shape)
    expert = jnp.broadcast_to(safe_expert_entry[..., None], valid.shape)
    safe_token = jnp.maximum(plan.token_ids, 0)

    zeros = jnp.zeros((), dtype=jnp.int32)
    return _SourcePushW13RowIndices(
        valid=valid,
        valid_f=valid_f,
        safe_src=jnp.where(valid, src, zeros),
        safe_token=jnp.where(valid, safe_token, zeros),
        safe_dst=jnp.where(valid, dst, zeros),
        safe_row=jnp.where(valid, flat_row, zeros),
        safe_local_row=jnp.where(valid, local_row, zeros),
        safe_expert=jnp.where(valid, expert, zeros),
    )
