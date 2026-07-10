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

from dataclasses import dataclass, replace
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
SOURCE_PUSH_W13_BACKWARD_EXPERIMENT_COMPACT_DX_SOURCE_GATHER_DW13 = "pallas_mgpu_compact_dx_source_gather_dw13"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY = "pallas_mgpu_compact_dx_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY = "pallas_mgpu_compact_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY = "pallas_mgpu_prefilled_x_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY = "pallas_mgpu_exact_flat_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY = "xla_compact_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY = "xla_source_padded_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY = "xla_local_swiglu_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY = (
    "pallas_mgpu_source_padded_partials_dw13_only"
)
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY = "pallas_mgpu_local_swiglu_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY = (
    "pallas_mgpu_local_swiglu_persistent_dw13_only"
)
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY = "pallas_mgpu_local_linear_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY = "pallas_mgpu_local_swiglu_gate_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY = "pallas_mgpu_local_swiglu_up_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY = (
    "pallas_mgpu_local_swiglu_split_dw13_only"
)
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY = "source_gather_dw13_only"
SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_IMPLEMENTATIONS = (
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_EXACT_FLAT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_COMPACT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_SOURCE_PADDED_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_XLA_LOCAL_SWIGLU_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_SOURCE_PADDED_PARTIALS_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_SOURCE_GATHER_DW13_ONLY,
)
SOURCE_PUSH_W13_BACKWARD_LOCAL_DW13_TILE_DIAGNOSTICS = (
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_PREFILLED_X_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_PERSISTENT_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_LINEAR_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY,
    SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_SPLIT_DW13_ONLY,
)
SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_X_TO_W13_ROWS_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu_x_remat"
SourcePushW13BackwardImplementation: TypeAlias = Literal["reference", "tiled", "pallas_mgpu"]
SourcePushXToW13RowsImplementation: TypeAlias = Literal["reference", "pallas_mgpu_x_remat"]
SourcePushW13LocalDzBranch: TypeAlias = Literal["both", "gate", "up"]
DEFAULT_X_REMAT_HIDDEN_BLOCK = 128
DEFAULT_W13_DX_ROW_BLOCK = 64
DEFAULT_W13_DX_HIDDEN_BLOCK = 128
DEFAULT_W13_DX_OUTPUT_BLOCK = 64
DEFAULT_W13_DW13_ROW_BLOCK = 64
DEFAULT_W13_DW13_HIDDEN_BLOCK = 128
DEFAULT_W13_DW13_OUTPUT_BLOCK = 64
DEFAULT_W13_DW13_LOWERING_SEMANTICS = mgpu.LoweringSemantics.Warpgroup
MIN_MOSAIC_INT32_TRANSFER_ELEMENTS = 128
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


def _source_push_destination_named_sharding(value: Array, ndim: int) -> NamedSharding | None:
    """Return destination-rank sharding using the mesh already attached to ``value``."""

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
    """Constrain eager compact W13 arrays to destination-major source-push layout."""

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

    abstract_sharding = _source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(value.ndim - 1)))
    if abstract_sharding is None:
        return value
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        return jax.sharding.reshard(value, abstract_sharding)
    return lax.with_sharding_constraint(value, abstract_sharding)


def _source_push_destination_or_replicated_spec(value: Array, ndim: int) -> P:
    """Return the shard_map spec matching an array's destination-axis layout."""

    destination_spec = P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(ndim - 1)))
    sharding = getattr(value, "sharding", None)
    if isinstance(sharding, NamedSharding) and sharding.spec == destination_spec:
        return destination_spec
    return P(*(None for _ in range(ndim)))


@dataclass(frozen=True, slots=True)
class SourcePushXToW13RowsPallasBlockSizes:
    """Tile sizes for the source-token-to-W13-row rematerialization kernel."""

    row_block: int = MIN_MOSAIC_INT32_TRANSFER_ELEMENTS
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


def source_push_w13_dw13_default_block_sizes() -> SourcePushW13BackwardTiledBlockSizes:
    """Return the tuned default tile for local compact DW13 diagnostics."""

    return SourcePushW13BackwardTiledBlockSizes(
        row_block=DEFAULT_W13_DW13_ROW_BLOCK,
        hidden_block=DEFAULT_W13_DW13_HIDDEN_BLOCK,
        output_block=DEFAULT_W13_DW13_OUTPUT_BLOCK,
    )


def source_push_w13_backward_diagnostic_component(implementation: str) -> str | None:
    """Return the partial W13 component produced by benchmark-only diagnostics."""

    if implementation == SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_COMPACT_DX_ONLY:
        return "dx"
    if implementation in (
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_GATE_DW13_ONLY,
        SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_PALLAS_MGPU_LOCAL_SWIGLU_UP_DW13_ONLY,
    ):
        return "dw13_half"
    if implementation in SOURCE_PUSH_W13_BACKWARD_DIAGNOSTIC_IMPLEMENTATIONS:
        return "dw13"
    return None


def source_push_w13_backward_is_diagnostic_only(implementation: str) -> bool:
    return source_push_w13_backward_diagnostic_component(implementation) is not None


def source_push_w13_backward_uses_local_dw13_default_block_sizes(implementation: str) -> bool:
    return implementation in SOURCE_PUSH_W13_BACKWARD_LOCAL_DW13_TILE_DIAGNOSTICS


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
    dx_expert_major = _with_source_push_destination_sharding(dx_expert_major, like=d_h)
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
        x_expert_major = _with_source_push_destination_sharding(x_expert_major, like=d_h)
    else:
        x_expert_major = jnp.zeros((0,), dtype=jnp.float32)
    dx_expert_major = _with_source_push_destination_sharding(dx_expert_major, like=d_h)
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


def source_push_w13_dw13_local_swiglu_reference(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E D twoI"]:
    """Local expert-owner ``dw13 = X.T @ dZ`` without requiring materialized ``dZ``.

    ``d_activation`` is the W2 activation gradient, and ``z`` is the saved W13
    preactivation ``[gate, up]``. The SwiGLU derivative is recomputed from those
    local tiles before accumulation.
    """

    _validate_w13_local_swiglu_dw13_request(x_expert_major, d_activation, z, valid_by_expert)
    d_z = _source_push_w13_local_dz_from_swiglu(d_activation, z, valid_by_expert)
    x_clean = jnp.where(
        valid_by_expert[..., None],
        x_expert_major.astype(jnp.float32),
        jnp.zeros(x_expert_major.shape, dtype=jnp.float32),
    )
    return jnp.einsum("dech,deco->deho", x_clean, d_z)


def source_push_w13_dw13_local_linear_reference(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E D twoI"]:
    """Diagnostic local ``dw13`` with cheap dSwiGLU replacement.

    This is not a correctness model for W13 backward. It preserves the same
    two-output DW13 schedule as ``source_push_w13_dw13_local_swiglu_reference``
    while replacing sigmoid/silu derivative work with linear pointwise math, so
    benchmarks can isolate activation-recompute cost.
    """

    _validate_w13_local_swiglu_dw13_request(x_expert_major, d_activation, z, valid_by_expert)
    d_z = _source_push_w13_local_dz_linear_diagnostic(d_activation, z, valid_by_expert)
    x_clean = jnp.where(
        valid_by_expert[..., None],
        x_expert_major.astype(jnp.float32),
        jnp.zeros(x_expert_major.shape, dtype=jnp.float32),
    )
    return jnp.einsum("dech,deco->deho", x_clean, d_z)


def source_push_w13_dw13_local_swiglu_branch_reference(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    branch: Literal["gate", "up"],
) -> Float[Array, "Dst E D I"]:
    """Diagnostic one-branch local ``dw13`` reference for accumulator-pressure probes."""

    _validate_w13_local_swiglu_dw13_request(x_expert_major, d_activation, z, valid_by_expert)
    d_z = _source_push_w13_local_dz_from_swiglu(d_activation, z, valid_by_expert)
    intermediate_dim = d_activation.shape[-1]
    if branch == "gate":
        d_branch = d_z[..., :intermediate_dim]
    elif branch == "up":
        d_branch = d_z[..., intermediate_dim:]
    else:
        raise ValueError(f"branch must be 'gate' or 'up'; got {branch!r}")
    x_clean = jnp.where(
        valid_by_expert[..., None],
        x_expert_major.astype(jnp.float32),
        jnp.zeros(x_expert_major.shape, dtype=jnp.float32),
    )
    return jnp.einsum("dech,deco->deho", x_clean, d_branch)


def _source_push_w13_local_dz_from_swiglu(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C twoI"]:
    """Recompute ``dZ=[dgate, dup]`` from local W2 activation gradients."""

    intermediate_dim = d_activation.shape[-1]
    valid_mask = valid_by_expert[..., None]
    gate = jnp.where(
        valid_mask,
        z[..., :intermediate_dim].astype(jnp.float32),
        jnp.zeros(z.shape[:-1] + (intermediate_dim,), dtype=jnp.float32),
    )
    up = jnp.where(
        valid_mask,
        z[..., intermediate_dim:].astype(jnp.float32),
        jnp.zeros(z.shape[:-1] + (intermediate_dim,), dtype=jnp.float32),
    )
    d_activation = jnp.where(
        valid_mask,
        d_activation.astype(jnp.float32),
        jnp.zeros(d_activation.shape, dtype=jnp.float32),
    )
    silu_gate = jax.nn.silu(gate)
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_gate = d_activation * up * d_silu_gate
    d_up = d_activation * silu_gate
    return jnp.concatenate([d_gate, d_up], axis=-1)


def _source_push_w13_local_dz_linear_diagnostic(
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> Float[Array, "Dst E C twoI"]:
    """Cheap diagnostic replacement for dSwiGLU used only in perf probes."""

    intermediate_dim = d_activation.shape[-1]
    valid_mask = valid_by_expert[..., None]
    gate = jnp.where(
        valid_mask,
        z[..., :intermediate_dim].astype(jnp.float32),
        jnp.zeros(z.shape[:-1] + (intermediate_dim,), dtype=jnp.float32),
    )
    up = jnp.where(
        valid_mask,
        z[..., intermediate_dim:].astype(jnp.float32),
        jnp.zeros(z.shape[:-1] + (intermediate_dim,), dtype=jnp.float32),
    )
    d_activation = jnp.where(
        valid_mask,
        d_activation.astype(jnp.float32),
        jnp.zeros(d_activation.shape, dtype=jnp.float32),
    )
    return jnp.concatenate([d_activation * up, d_activation * gate], axis=-1)


def _source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only local ``dw13`` diagnostic that recomputes dSwiGLU inside WGMMA tiles."""

    return _source_push_w13_backward_expert_blocks_local_dz_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
        use_linear_derivative=False,
        branch="both",
        preclean_inputs=True,
    )


def source_push_w13_backward_expert_blocks_local_swiglu_dw13_only_xla(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only XLA floor for local expert-owner SwiGLU ``dw13``."""

    dw13 = source_push_w13_dw13_local_swiglu_reference(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
    )


def _source_push_w13_backward_expert_blocks_local_swiglu_persistent_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Local-only persistent DW13 diagnostic with tile-local dSwiGLU.

    This variant keeps the intended local expert-owner schedule explicit: one
    program owns one ``(expert, hidden_tile, intermediate_tile)`` output tile,
    loops over all token blocks for that expert, derives ``dgate``/``dup`` from
    ``d_activation`` and saved W13 preactivation ``z`` inside the tile loop, and
    accumulates the two W13 halves in fp32. Unlike the older local diagnostic it
    does not materialize cleaned full-size compact inputs before launch; callers
    should provide finite invalid rows, with invalid ``x`` rows already zeroed.
    """

    return _source_push_w13_backward_expert_blocks_local_dz_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
        use_linear_derivative=False,
        branch="both",
        preclean_inputs=False,
    )


def _source_push_w13_backward_expert_blocks_local_linear_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only local DW13 diagnostic with cheap linear dSwiGLU replacement."""

    return _source_push_w13_backward_expert_blocks_local_dz_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
        use_linear_derivative=True,
        branch="both",
        preclean_inputs=True,
    )


def _source_push_w13_backward_expert_blocks_local_swiglu_gate_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only local DW13 diagnostic that computes only the gate half."""

    return _source_push_w13_backward_expert_blocks_local_dz_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
        use_linear_derivative=False,
        branch="gate",
        preclean_inputs=True,
    )


def _source_push_w13_backward_expert_blocks_local_swiglu_up_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only local DW13 diagnostic that computes only the up half."""

    return _source_push_w13_backward_expert_blocks_local_dz_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
        use_linear_derivative=False,
        branch="up",
        preclean_inputs=True,
    )


def _source_push_w13_backward_expert_blocks_local_swiglu_split_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only full DW13 diagnostic built from serial one-branch kernels.

    This keeps only one ``(hidden_block, output_block)`` accumulator live per
    Pallas launch. It is a pressure probe for the existing two-accumulator local
    SwiGLU kernel, not a production implementation.
    """

    gate_output = _source_push_w13_backward_expert_blocks_local_swiglu_gate_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
    )
    up_output = _source_push_w13_backward_expert_blocks_local_swiglu_up_dw13_only_pallas_mgpu(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes=block_sizes,
        lowering_semantics=lowering_semantics,
        interpret=interpret,
        mesh=mesh,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=jnp.concatenate((gate_output.dw13, up_output.dw13), axis=-1),
    )


def _source_push_w13_backward_expert_blocks_local_dz_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
    use_linear_derivative: bool,
    branch: SourcePushW13LocalDzBranch,
    preclean_inputs: bool,
) -> SourcePushW13CompactBackwardOutput:
    """Shared implementation for local DW13 diagnostics that derive dZ inside the kernel."""

    block_sizes = source_push_w13_dw13_default_block_sizes() if block_sizes is None else block_sizes
    lowering_semantics = DEFAULT_W13_DW13_LOWERING_SEMANTICS if lowering_semantics is None else lowering_semantics
    _validate_w13_local_swiglu_dw13_request(x_expert_major, d_activation, z, valid_by_expert)
    if interpret:
        if branch == "gate":
            dw13 = source_push_w13_dw13_local_swiglu_branch_reference(
                x_expert_major,
                d_activation,
                z,
                valid_by_expert,
                branch="gate",
            )
        elif branch == "up":
            dw13 = source_push_w13_dw13_local_swiglu_branch_reference(
                x_expert_major,
                d_activation,
                z,
                valid_by_expert,
                branch="up",
            )
        elif use_linear_derivative:
            dw13 = source_push_w13_dw13_local_linear_reference(x_expert_major, d_activation, z, valid_by_expert)
        else:
            dw13 = source_push_w13_dw13_local_swiglu_reference(x_expert_major, d_activation, z, valid_by_expert)
        return SourcePushW13CompactBackwardOutput(
            x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dw13=dw13,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU local SwiGLU dw13 diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU local SwiGLU dw13 diagnostic requires a mesh")

    original_rows = x_expert_major.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        x_expert_major = jnp.pad(x_expert_major, (*row_pad, (0, 0)))
        d_activation = jnp.pad(d_activation, (*row_pad, (0, 0)))
        z = jnp.pad(z, (*row_pad, (0, 0)))
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    if preclean_inputs:
        valid_mask = valid_by_expert[..., None]
        x_expert_major = jnp.where(valid_mask, x_expert_major, jnp.zeros_like(x_expert_major))
        d_activation = jnp.where(valid_mask, d_activation, jnp.zeros_like(d_activation))
        z = jnp.where(valid_mask, z, jnp.zeros_like(z))
    _validate_w13_local_swiglu_dw13_pallas_request(
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        block_sizes,
    )
    dw13 = _source_push_w13_dw13_local_swiglu_sharded_mgpu_kernel(
        mesh,
        x_expert_major,
        d_activation,
        z,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
        lowering_semantics=lowering_semantics,
        use_linear_derivative=use_linear_derivative,
        branch=branch,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
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
        dx_expert_major = _with_source_push_destination_sharding(dx_expert_major, mesh=mesh, like=d_h)
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
    dx_expert_major = dx_expert_major[:, :, :original_rows, :]
    dx_expert_major = _with_source_push_destination_sharding(
        dx_expert_major,
        mesh=mesh,
        like=d_h,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=dx_expert_major,
        dw13=jnp.zeros(w13.shape, dtype=jnp.float32),
    )


def _source_push_w13_backward_expert_blocks_dw13_only_pallas_mgpu(
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
    lowering_semantics: mgpu.LoweringSemantics | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only compact W13 ``dw13`` diagnostic using compact x staging."""

    block_sizes = source_push_w13_dw13_default_block_sizes() if block_sizes is None else block_sizes
    lowering_semantics = DEFAULT_W13_DW13_LOWERING_SEMANTICS if lowering_semantics is None else lowering_semantics
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
        dw13 = source_push_w13_backward_expert_blocks_tiled_reference(
            x,
            d_h,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            block_sizes=block_sizes,
        ).dw13
        return SourcePushW13CompactBackwardOutput(
            x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dw13=dw13,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU compact W13 dw13-only diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU compact W13 dw13-only diagnostic requires a mesh")

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
    dw13 = _source_push_w13_dw13_expert_blocks_sharded_mgpu_kernel(
        mesh,
        x_expert_major.astype(w13.dtype),
        d_h_clean.astype(w13.dtype),
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
        lowering_semantics=lowering_semantics,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
    )


def _source_push_w13_backward_expert_blocks_prefilled_x_dw13_only_pallas_mgpu(
    x_expert_major: Float[Array, "Dst E C D"],
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
    lowering_semantics: mgpu.LoweringSemantics | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only compact ``dw13`` diagnostic with prefilled expert-major ``x``.

    The regular compact DW13 diagnostic includes the ``x[source, token]`` gather
    in the timed path. This variant takes already-materialized expert-major ``x``
    so the benchmark can isolate the local ``X.T @ dZ`` kernel cost.
    """

    block_sizes = source_push_w13_dw13_default_block_sizes() if block_sizes is None else block_sizes
    lowering_semantics = DEFAULT_W13_DW13_LOWERING_SEMANTICS if lowering_semantics is None else lowering_semantics
    _validate_w13_prefilled_x_dw13_request(x_expert_major, d_h, w13, valid_by_expert, block_sizes)
    if interpret:
        valid_f = valid_by_expert.astype(jnp.float32)
        dw13 = jnp.einsum(
            "dech,deco->deho",
            x_expert_major.astype(jnp.float32) * valid_f[..., None],
            d_h.astype(jnp.float32) * valid_f[..., None],
        )
        return SourcePushW13CompactBackwardOutput(
            x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dw13=dw13,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU prefilled-X W13 dw13-only diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU prefilled-X W13 dw13-only diagnostic requires a mesh")

    original_rows = d_h.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        x_expert_major = jnp.pad(x_expert_major, (*row_pad, (0, 0)))
        d_h = jnp.pad(d_h, (*row_pad, (0, 0)))
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    valid_f = valid_by_expert.astype(jnp.float32)
    x_clean = x_expert_major.astype(jnp.float32) * valid_f[..., None]
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]
    dw13 = _source_push_w13_dw13_expert_blocks_sharded_mgpu_kernel(
        mesh,
        x_clean.astype(w13.dtype),
        d_h_clean.astype(w13.dtype),
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
        lowering_semantics=lowering_semantics,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
    )


def _source_push_w13_backward_expert_blocks_dw13_only_exact_flat_pallas_mgpu(
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
    """Benchmark-only compact DW13 diagnostic using the exact-flat Pallas schedule."""

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    if not interpret and block_sizes.row_block < MIN_MOSAIC_INT32_TRANSFER_ELEMENTS:
        block_sizes = replace(block_sizes, row_block=MIN_MOSAIC_INT32_TRANSFER_ELEMENTS)
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
        dw13 = source_push_w13_backward_expert_blocks_tiled_reference(
            x,
            d_h,
            w13,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            block_sizes=block_sizes,
        ).dw13
        return SourcePushW13CompactBackwardOutput(
            x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
            dw13=dw13,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU exact-flat W13 dw13-only diagnostic requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU exact-flat W13 dw13-only diagnostic requires a mesh")

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
    x_clean = x_expert_major.astype(jnp.float32) * valid_f[..., None]
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]
    dst_count, local_experts, rows, hidden_dim = x_clean.shape
    output_dim = d_h_clean.shape[-1]
    x_flat = x_clean.reshape((dst_count, local_experts * rows, hidden_dim)).astype(w13.dtype)
    d_h_flat = d_h_clean.reshape((dst_count, local_experts * rows, output_dim)).astype(w13.dtype)
    valid_flat = valid_by_expert.reshape((dst_count, local_experts * rows))
    row_expert = jnp.broadcast_to(
        jnp.arange(local_experts, dtype=jnp.int32)[None, :, None],
        valid_by_expert.shape,
    ).reshape((dst_count, local_experts * rows))
    dw13 = _source_push_w13_dw13_exact_expert_major_pallas_mgpu(
        x_flat,
        d_h_flat,
        w13,
        valid_flat,
        row_expert,
        block_sizes=block_sizes,
        mesh=mesh,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
    )


def source_push_w13_backward_expert_blocks_dw13_only_xla(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E D twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Benchmark-only compact W13 ``dw13`` diagnostic using XLA GEMM lowering."""

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
    safe_src = jnp.where(valid_by_expert, source_rank_by_expert, 0)
    safe_token = jnp.where(valid_by_expert, token_id_by_expert, 0)
    x_expert_major = x.at[safe_src, safe_token].get(
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None, None)
    )
    valid_f = valid_by_expert.astype(jnp.float32)
    x_clean = x_expert_major.astype(jnp.float32) * valid_f[..., None]
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]
    dw13 = jnp.einsum("dech,deco->deho", x_clean, d_h_clean)
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
    )


def source_push_w13_backward_expert_blocks_source_padded_dw13_only_xla(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
) -> SourcePushW13CompactBackwardOutput:
    """Diagnostic ``dw13`` as a reduce over source-padded transpose GMMs.

    Source-padded H rows are laid out as contiguous source chunks within each
    destination-local expert. This diagnostic preserves that structure:
    for each source rank it gathers only that static source's token rows,
    computes ``x_src.T @ dH_src`` over the dense padded source chunk, then sums
    the partials into ``dw13``. It is intentionally XLA-only and exists to test
    the shape we want a production WGMMA kernel to implement.
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
    partials = source_push_w13_dw13_source_padded_partials_reference(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        src_base_by_expert,
        block_sizes=block_sizes,
    )
    dw13 = jnp.sum(partials, axis=0)
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dw13=dw13,
    )


def source_push_w13_dw13_source_padded_partials_reference(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> Float[Array, "S Dst E D twoI"]:
    """Reference partials for source-padded ``dw13``.

    Each output slice is one dense source chunk contribution:
    ``partials[src, dst, expert] = x_src.T @ dH_src``. A production Pallas
    version should implement this contract with source-specialized WGMMA
    programs, then reduce the leading source axis outside the kernel.
    """

    src_base_host, source_lengths = _source_padded_dw13_source_spans(
        src_base_by_expert,
        source_rank_by_expert,
        valid_by_expert,
        row_block=block_sizes.row_block,
        expert_capacity=d_h.shape[2],
        ep_size=x.shape[0],
        local_experts=d_h.shape[1],
    )
    dst_count, local_experts, expert_capacity, _ = d_h.shape
    ep_size = x.shape[0]
    max_source_rows = int(np.max(source_lengths)) if source_lengths.size else 0
    if max_source_rows == 0:
        return jnp.zeros((ep_size, dst_count, local_experts, x.shape[-1], d_h.shape[-1]), dtype=jnp.float32)

    dst_idx = jnp.arange(dst_count, dtype=jnp.int32)[:, None, None]
    expert_idx = jnp.arange(local_experts, dtype=jnp.int32)[None, :, None]
    offsets = jnp.arange(max_source_rows, dtype=jnp.int32)[None, None, :]
    src_base = jnp.asarray(src_base_host, dtype=jnp.int32)
    src_lengths = jnp.asarray(source_lengths, dtype=jnp.int32)
    partials = []

    for src in range(ep_size):
        rows = src_base[:, src, :, None] + offsets
        in_source_chunk = offsets < src_lengths[:, src, :, None]
        safe_rows = jnp.minimum(rows, expert_capacity - 1)
        valid_rows = valid_by_expert.at[dst_idx, expert_idx, safe_rows].get(
            out_sharding=_source_push_out_sharding(None, None, None)
        )
        row_sources = source_rank_by_expert.at[dst_idx, expert_idx, safe_rows].get(
            out_sharding=_source_push_out_sharding(None, None, None)
        )
        valid = in_source_chunk & valid_rows & (row_sources == src)
        token_rows = token_id_by_expert.at[dst_idx, expert_idx, safe_rows].get(
            out_sharding=_source_push_out_sharding(None, None, None)
        )
        safe_token = jnp.where(valid, token_rows, 0)
        valid_f = valid.astype(jnp.float32)
        x_rows = (
            x.at[src, safe_token, :]
            .get(out_sharding=_source_push_out_sharding(None, None, None, None))
            .astype(jnp.float32)
            * valid_f[..., None]
        )
        d_h_rows = (
            d_h.at[dst_idx, expert_idx, safe_rows, :]
            .get(out_sharding=_source_push_out_sharding(None, None, None, None))
            .astype(jnp.float32)
            * valid_f[..., None]
        )
        partials.append(jnp.einsum("dech,deco->deho", x_rows, d_h_rows))

    return jnp.stack(partials, axis=0)


def _source_push_w13_dw13_source_padded_partials_pallas_mgpu(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "S Dst E D twoI"]:
    """Source-padded partial ``dw13`` Pallas diagnostic.

    The interpreter path pins the intended contract. GPU lowering should use
    source-specialized WGMMA programs that emit per-source partials, avoiding
    compact ``x_expert_major`` staging and avoiding per-row remote-ref switches.
    """

    _ = mesh
    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_w13_dw13_source_gather_request(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes,
    )
    if interpret:
        return source_push_w13_dw13_source_padded_partials_reference(
            x,
            d_h,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            src_base_by_expert,
            block_sizes=block_sizes,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU source-padded partial dw13 requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU source-padded partial dw13 requires a mesh")

    original_rows = d_h.shape[2]
    padded_rows = _round_up_to_multiple(original_rows, block_sizes.row_block)
    if padded_rows != original_rows:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - original_rows))
        d_h = jnp.pad(d_h, (*row_pad, (0, 0)), constant_values=0)
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    src_base_host, source_lengths = _source_padded_dw13_source_spans(
        src_base_by_expert,
        source_rank_by_expert,
        valid_by_expert,
        row_block=block_sizes.row_block,
        expert_capacity=d_h.shape[2],
        ep_size=x.shape[0],
        local_experts=d_h.shape[1],
    )
    src_base = jnp.asarray(src_base_host, dtype=jnp.int32)
    source_lengths = jnp.asarray(source_lengths, dtype=jnp.int32)
    return _source_push_w13_dw13_source_padded_partials_sharded_mgpu_kernel(
        mesh,
        x,
        d_h,
        token_id_by_expert,
        valid_by_expert,
        src_base,
        source_lengths,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )


def _source_padded_dw13_source_spans(
    src_base_by_expert: Int[Array, "Dst S E"] | np.ndarray,
    source_rank_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    expert_capacity: int,
    ep_size: int,
    local_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    src_base_host = np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)
    dst_count = src_base_host.shape[0]
    if src_base_host.shape != (dst_count, ep_size, local_experts):
        raise ValueError(
            f"src_base_by_expert shape {src_base_host.shape} must be {(dst_count, ep_size, local_experts)}"
        )

    source_rank_host = np.asarray(jax.device_get(source_rank_by_expert), dtype=np.int32)
    valid_host = np.asarray(jax.device_get(valid_by_expert), dtype=np.bool_)
    source_lengths = np.zeros_like(src_base_host)
    for dst in range(dst_count):
        for expert in range(local_experts):
            for src in range(ep_size):
                if src + 1 < ep_size:
                    next_base = int(src_base_host[dst, src + 1, expert])
                    source_lengths[dst, src, expert] = max(0, next_base - int(src_base_host[dst, src, expert]))
                    continue

                live_rows = int(np.sum(valid_host[dst, expert] & (source_rank_host[dst, expert] == src)))
                source_lengths[dst, src, expert] = _round_up_to_multiple(live_rows, row_block)
                if int(src_base_host[dst, src, expert]) + int(source_lengths[dst, src, expert]) > expert_capacity:
                    source_lengths[dst, src, expert] = max(0, expert_capacity - int(src_base_host[dst, src, expert]))
    return src_base_host, source_lengths


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


def _source_push_w13_dw13_expert_blocks_source_gather_pallas_mgpu(
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid_by_expert: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst E D twoI"]:
    """Experimental WGMMA ``dw13`` path that gathers source ``x`` rows by tile.

    This is the narrow production-facing half of compact W13 backward: each
    destination rank keeps compact ``dH`` local, gathers only the source-token
    ``x`` rows needed for the current WGMMA row tile, and accumulates
    ``x.T @ dH`` without staging compact ``x_expert_major`` in GMEM.
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
    if interpret:
        return source_push_w13_dw13_expert_blocks_source_gather_tiled_reference(
            x,
            d_h,
            source_rank_by_expert,
            token_id_by_expert,
            valid_by_expert,
            block_sizes=block_sizes,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU source-gather W13 dw13 requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU source-gather W13 dw13 requires a mesh")

    padded_rows = _round_up_to_multiple(d_h.shape[2], block_sizes.row_block)
    if padded_rows != d_h.shape[2]:
        row_pad = ((0, 0), (0, 0), (0, padded_rows - d_h.shape[2]))
        d_h = jnp.pad(d_h, (*row_pad, (0, 0)), constant_values=0)
        source_rank_by_expert = jnp.pad(source_rank_by_expert, row_pad, constant_values=0)
        token_id_by_expert = jnp.pad(token_id_by_expert, row_pad, constant_values=0)
        valid_by_expert = jnp.pad(valid_by_expert, row_pad, constant_values=False)

    valid_f = valid_by_expert.astype(jnp.float32)
    d_h_clean = d_h.astype(jnp.float32) * valid_f[..., None]
    d_h_for_wgmma = d_h_clean.astype(x.dtype)

    return _source_push_w13_dw13_expert_blocks_source_gather_sharded_mgpu_kernel(
        mesh,
        x,
        d_h_for_wgmma,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
    )


def _source_push_w13_backward_expert_blocks_compact_dx_source_gather_dw13(
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
    """Compact W13 backward experiment without staging compact ``x``.

    ``dx`` uses the existing destination-local compact WGMMA path. ``dw13``
    uses the source-gather MGPU path, gathering source-major ``x`` into each
    WGMMA tile instead of first materializing ``x_expert_major``. This is an
    explicit experiment, not a saved-residual or public VJP boundary.
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
    dx_output = _source_push_w13_backward_expert_blocks_dx_only_pallas_mgpu(
        d_h,
        w13,
        valid_by_expert,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    dw13 = _source_push_w13_dw13_expert_blocks_source_gather_pallas_mgpu(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid_by_expert,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=jnp.zeros((0,), dtype=jnp.float32),
        dx_expert_major=dx_output.dx_expert_major,
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
    return_x_expert_major: bool = False,
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
            return_x_expert_major=return_x_expert_major,
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
    if return_x_expert_major:
        x_expert_major = _with_source_push_destination_sharding(
            x_expert_major[:, :, :original_rows, :],
            mesh=mesh,
            like=d_h,
        )
    else:
        x_expert_major = jnp.zeros((0,), dtype=jnp.float32)
    dx_expert_major = _with_source_push_destination_sharding(
        dx_expert_major[:, :, :original_rows, :],
        mesh=mesh,
        like=d_h,
    )
    return SourcePushW13CompactBackwardOutput(
        x_expert_major=x_expert_major,
        dx_expert_major=dx_expert_major,
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
        row_block=block_sizes.row_block,
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
    row_block: int,
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
            row_block=row_block,
            hidden_block=hidden_block,
        )
    if shard_if_needed and not interpret and not abstract_mesh.empty:
        return _source_push_x_to_w13_rows_sharded_pallas_call(
            abstract_mesh,
            x,
            row_src,
            row_token,
            row_valid,
            row_block=row_block,
            hidden_block=hidden_block,
        )

    dst_count, hidden_rows_per_rank = row_valid.shape
    hidden_dim = x.shape[-1]
    output_shape = jax.ShapeDtypeStruct((dst_count, hidden_rows_per_rank, hidden_dim), jnp.float32)
    row_valid_i = row_valid.astype(jnp.int32)
    cost_estimate = _source_push_x_to_w13_rows_pallas_cost_estimate(x, row_src, row_token, row_valid, output_shape)
    kernel = _make_source_push_x_to_w13_rows_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        use_tiled_offsets=not interpret,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    in_specs, out_specs = _source_push_x_to_w13_rows_block_specs(row_block=row_block, hidden_block=hidden_block)
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=output_shape,
        grid=(dst_count, hidden_rows_per_rank // row_block, hidden_dim // hidden_block),
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
    row_block: int,
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
            row_block=row_block,
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
    row_block: int,
    hidden_block: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    out_spec = pl.BlockSpec(
        (None, row_block, hidden_block),
        lambda dst, row_tile, hidden_tile: (dst, row_tile, hidden_tile),
    )
    return (gmem_spec, gmem_spec, gmem_spec, gmem_spec), out_spec


def _make_source_push_x_to_w13_rows_kernel(*, row_block: int, hidden_block: int, use_tiled_offsets: bool):
    def kernel(
        x_ref: Float[pl.Ref, "S T D"],
        row_src_ref: Int[pl.Ref, "Dst rows"],
        row_token_ref: Int[pl.Ref, "Dst rows"],
        row_valid_ref: Int[pl.Ref, "Dst rows"],
        x_out_ref: Float[pl.Ref, "rows D"],
    ) -> None:
        dst = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block
        hidden_offsets = jnp.arange(hidden_block, dtype=jnp.int32)
        if use_tiled_offsets:
            hidden_offsets = mgpu.layout_cast(hidden_offsets, mgpu.Layout.TILED)
        hidden_offsets = hidden_start + hidden_offsets
        valid = row_valid_ref[dst, pl.ds(row_start, row_block)]
        src = row_src_ref[dst, pl.ds(row_start, row_block)]
        token = row_token_ref[dst, pl.ds(row_start, row_block)]
        x_tile = x_ref[src[:, None], token[:, None], hidden_offsets[None, :]]
        zero_tile = jnp.zeros((row_block, hidden_block), dtype=jnp.float32)
        out_tile = jnp.where(valid[:, None], x_tile.astype(jnp.float32), zero_tile)
        x_out_ref[pl.ds(0, row_block), pl.ds(0, hidden_block)] = out_tile

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
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if flat_row_map.valid.shape[-1] % block_sizes.row_block:
        raise ValueError(
            f"flat rows {flat_row_map.valid.shape[-1]} must be divisible by row_block={block_sizes.row_block}"
        )
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

        valid = valid_by_expert.at[dst, expert, rows].get(out_sharding=_source_push_out_sharding(None))
        valid_f = valid.astype(jnp.float32)
        src = jnp.where(
            valid,
            source_rank_by_expert.at[dst, expert, rows].get(out_sharding=_source_push_out_sharding(None)),
            0,
        )
        token = jnp.where(
            valid,
            token_id_by_expert.at[dst, expert, rows].get(out_sharding=_source_push_out_sharding(None)),
            0,
        )
        x_tile = (
            x.at[src[:, None], token[:, None], hidden[None, :]]
            .get(out_sharding=_source_push_out_sharding(None, None))
            .astype(jnp.float32)
            * valid_f[:, None]
        )
        d_h_tile = (
            d_h.at[dst, expert, rows[:, None], output[None, :]]
            .get(out_sharding=_source_push_out_sharding(None, None))
            .astype(jnp.float32)
            * valid_f[:, None]
        )
        w13_tile = (
            w13.at[dst, expert, hidden[:, None], output[None, :]]
            .get(out_sharding=_source_push_out_sharding(None, None))
            .astype(jnp.float32)
        )

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

        valid = valid_by_expert.at[dst, expert, rows].get(out_sharding=_source_push_out_sharding(None))
        valid_f = valid.astype(jnp.float32)
        src = jnp.where(
            valid,
            source_rank_by_expert.at[dst, expert, rows].get(out_sharding=_source_push_out_sharding(None)),
            0,
        )
        token = jnp.where(
            valid,
            token_id_by_expert.at[dst, expert, rows].get(out_sharding=_source_push_out_sharding(None)),
            0,
        )
        x_tile = (
            x.at[src[:, None], token[:, None], hidden[None, :]]
            .get(out_sharding=_source_push_out_sharding(None, None))
            .astype(jnp.float32)
            * valid_f[:, None]
        )
        d_h_tile = (
            d_h.at[dst, expert, rows[:, None], output[None, :]]
            .get(out_sharding=_source_push_out_sharding(None, None))
            .astype(jnp.float32)
            * valid_f[:, None]
        )

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


def _validate_w13_prefilled_x_dw13_request(
    x_expert_major: Array,
    d_h: Array,
    w13: Array,
    valid_by_expert: Array,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> None:
    if x_expert_major.ndim != 4:
        raise ValueError(f"x_expert_major must have shape [dst, expert, capacity, D], got {x_expert_major.shape}")
    if d_h.ndim != 4:
        raise ValueError(f"d_h must have shape [dst, expert, capacity, twoI], got {d_h.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [dst, expert, D, twoI], got {w13.shape}")
    if x_expert_major.shape[:3] != d_h.shape[:3]:
        raise ValueError(f"x_expert_major leading shape {x_expert_major.shape[:3]} must match d_h {d_h.shape[:3]}")
    if valid_by_expert.shape != d_h.shape[:3]:
        raise ValueError(f"valid_by_expert shape {valid_by_expert.shape} must match d_h blocks {d_h.shape[:3]}")
    if w13.shape[:2] != d_h.shape[:2]:
        raise ValueError(f"w13 destination/expert shape {w13.shape[:2]} must match d_h {d_h.shape[:2]}")
    if w13.shape[-2] != x_expert_major.shape[-1]:
        raise ValueError(f"w13 hidden dim {w13.shape[-2]} must match x hidden dim {x_expert_major.shape[-1]}")
    if w13.shape[-1] != d_h.shape[-1]:
        raise ValueError(f"w13 output dim {w13.shape[-1]} must match d_h output dim {d_h.shape[-1]}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if x_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            f"x_expert_major hidden dim {x_expert_major.shape[-1]} must be divisible by "
            f"hidden_block={block_sizes.hidden_block}"
        )
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
            block_sizes=block_sizes,
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
    block_sizes: SourcePushW13BackwardTiledBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | AbstractMesh | None = None,
) -> Float[Array, "Dst E D twoI"]:
    """Compute W13 weight gradients for exact expert-major rows."""

    block_sizes = SourcePushW13BackwardTiledBlockSizes.get_default() if block_sizes is None else block_sizes
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
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            output_block=block_sizes.output_block,
        )
    return _source_push_w13_dw13_exact_expert_major_pallas_call(
        x_expert_major,
        d_h,
        w13,
        valid,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
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
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Lane,
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
        lowering_semantics=lowering_semantics,
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


def _source_push_w13_dw13_local_swiglu_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    x_expert_major: Float[Array, "Dst E C D"],
    d_activation: Float[Array, "Dst E C I"],
    z: Float[Array, "Dst E C twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    lowering_semantics: mgpu.LoweringSemantics = DEFAULT_W13_DW13_LOWERING_SEMANTICS,
    use_linear_derivative: bool = False,
    branch: SourcePushW13LocalDzBranch = "both",
) -> Float[Array, "Dst E D twoI"]:
    block_sizes = SourcePushW13BackwardTiledBlockSizes(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
    )
    _validate_w13_local_swiglu_dw13_pallas_request(x_expert_major, d_activation, z, valid, block_sizes)
    kernel = _make_source_push_w13_dw13_local_swiglu_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=x_expert_major.shape[1],
        rows=x_expert_major.shape[2],
        hidden_dim=x_expert_major.shape[-1],
        intermediate_dim=d_activation.shape[-1],
        lowering_semantics=lowering_semantics,
        use_linear_derivative=use_linear_derivative,
        branch=branch,
    )

    def local_fn(
        x_local: Float[Array, "1 E C D"],
        d_activation_local: Float[Array, "1 E C I"],
        z_local: Float[Array, "1 E C twoI"],
        valid_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 E D twoI"]:
        _ = valid_local
        return kernel(x_local[0], d_activation_local[0], z_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(x_expert_major, d_activation, z, valid)


def _source_push_w13_dw13_expert_blocks_source_gather_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    source_rank_by_expert: Int[Array, "Dst E C"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "Dst E D twoI"]:
    _validate_w13_dw13_source_gather_request(
        x,
        d_h,
        source_rank_by_expert,
        token_id_by_expert,
        valid,
        SourcePushW13BackwardTiledBlockSizes(
            row_block=row_block,
            hidden_block=hidden_block,
            output_block=output_block,
        ),
    )
    if d_h.shape[2] % row_block:
        raise ValueError(f"d_h capacity {d_h.shape[2]} must be divisible by row_block={row_block}")
    _validate_w13_wgmma_smem_shape((row_block, hidden_block), x.dtype)
    _validate_w13_wgmma_smem_shape((row_block, output_block), d_h.dtype)

    kernel = _make_source_push_w13_dw13_source_gather_expert_blocks_mgpu_kernel(
        ep_size=x.shape[0],
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_h.shape[1],
        rows=d_h.shape[2],
        hidden_dim=x.shape[-1],
        output_dim=d_h.shape[-1],
    )
    source_rank_spec = _source_push_destination_or_replicated_spec(source_rank_by_expert, 3)
    token_id_spec = _source_push_destination_or_replicated_spec(token_id_by_expert, 3)
    valid_spec = _source_push_destination_or_replicated_spec(valid, 3)
    destination_spec = P(SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        x_local: Float[Array, "1 T D"],
        d_h_local: Float[Array, "1 E C twoI"],
        source_rank_arg: Int[Array, "Dst E C"],
        token_id_arg: Int[Array, "Dst E C"],
        valid_arg: Bool[Array, "Dst E C"],
    ) -> Float[Array, "1 E D twoI"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if source_rank_spec == destination_spec:
            source_rank_local = source_rank_arg
        else:
            source_rank_local = lax.dynamic_slice_in_dim(source_rank_arg, dst, 1, axis=0)
        if token_id_spec == destination_spec:
            token_id_local = token_id_arg
        else:
            token_id_local = lax.dynamic_slice_in_dim(token_id_arg, dst, 1, axis=0)
        if valid_spec == destination_spec:
            valid_local = valid_arg
        else:
            valid_local = lax.dynamic_slice_in_dim(valid_arg, dst, 1, axis=0)
        return kernel(
            x_local[0],
            d_h_local[0],
            source_rank_local[0],
            token_id_local[0],
            valid_local[0].astype(jnp.int32),
        )[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            source_rank_spec,
            token_id_spec,
            valid_spec,
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(x, d_h, source_rank_by_expert, token_id_by_expert, valid)


def _source_push_w13_dw13_source_padded_partials_sharded_mgpu_kernel(
    mesh: Mesh | AbstractMesh,
    x: Float[Array, "S T D"],
    d_h: Float[Array, "Dst E C twoI"],
    token_id_by_expert: Int[Array, "Dst E C"],
    valid: Bool[Array, "Dst E C"],
    src_base_by_expert: Int[Array, "Dst S E"],
    source_lengths: Int[Array, "Dst S E"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
) -> Float[Array, "S Dst E D twoI"]:
    _validate_w13_dw13_source_gather_request(
        x,
        d_h,
        jnp.zeros_like(token_id_by_expert),
        token_id_by_expert,
        valid,
        SourcePushW13BackwardTiledBlockSizes(
            row_block=row_block,
            hidden_block=hidden_block,
            output_block=output_block,
        ),
    )
    if d_h.shape[2] % row_block:
        raise ValueError(f"d_h capacity {d_h.shape[2]} must be divisible by row_block={row_block}")
    if src_base_by_expert.shape != (d_h.shape[0], x.shape[0], d_h.shape[1]):
        raise ValueError(
            f"src_base_by_expert shape {src_base_by_expert.shape} must be {(d_h.shape[0], x.shape[0], d_h.shape[1])}"
        )
    if source_lengths.shape != src_base_by_expert.shape:
        raise ValueError(f"source_lengths shape {source_lengths.shape} must match {src_base_by_expert.shape}")
    _validate_w13_wgmma_smem_shape((row_block, hidden_block), x.dtype)
    _validate_w13_wgmma_smem_shape((row_block, output_block), d_h.dtype)

    max_source_rows = int(np.max(np.asarray(jax.device_get(source_lengths), dtype=np.int32)))
    max_source_rows = _round_up_to_multiple(max_source_rows, row_block)
    kernel = _make_source_push_w13_dw13_source_padded_partials_mgpu_kernel(
        ep_size=x.shape[0],
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=d_h.shape[1],
        rows=d_h.shape[2],
        max_source_rows=max_source_rows,
        hidden_dim=x.shape[-1],
        output_dim=d_h.shape[-1],
    )
    token_id_spec = _source_push_destination_or_replicated_spec(token_id_by_expert, 3)
    valid_spec = _source_push_destination_or_replicated_spec(valid, 3)
    src_base_spec = _source_push_destination_or_replicated_spec(src_base_by_expert, 3)
    source_lengths_spec = _source_push_destination_or_replicated_spec(source_lengths, 3)
    destination_spec = P(SOURCE_PUSH_MESH_AXIS, None, None)

    def local_fn(
        x_local: Float[Array, "1 T D"],
        d_h_local: Float[Array, "1 E C twoI"],
        token_id_arg: Int[Array, "Dst E C"],
        valid_arg: Bool[Array, "Dst E C"],
        src_base_arg: Int[Array, "Dst S E"],
        source_lengths_arg: Int[Array, "Dst S E"],
    ) -> Float[Array, "S 1 E D twoI"]:
        dst = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        if token_id_spec == destination_spec:
            token_id_local = token_id_arg
        else:
            token_id_local = lax.dynamic_slice_in_dim(token_id_arg, dst, 1, axis=0)
        if valid_spec == destination_spec:
            valid_local = valid_arg
        else:
            valid_local = lax.dynamic_slice_in_dim(valid_arg, dst, 1, axis=0)
        if src_base_spec == P(SOURCE_PUSH_MESH_AXIS, None, None):
            src_base_local = src_base_arg
        else:
            src_base_local = lax.dynamic_slice_in_dim(src_base_arg, dst, 1, axis=0)
        if source_lengths_spec == P(SOURCE_PUSH_MESH_AXIS, None, None):
            source_lengths_local = source_lengths_arg
        else:
            source_lengths_local = lax.dynamic_slice_in_dim(source_lengths_arg, dst, 1, axis=0)
        return kernel(
            x_local[0],
            d_h_local[0],
            token_id_local[0],
            valid_local[0].astype(jnp.int32),
            src_base_local[0],
            source_lengths_local[0],
        )[:, None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            token_id_spec,
            valid_spec,
            src_base_spec,
            source_lengths_spec,
        ),
        out_specs=P(None, SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(x, d_h, token_id_by_expert, valid, src_base_by_expert, source_lengths)


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
    lowering_semantics: mgpu.LoweringSemantics,
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
    compiler_params = mgpu.CompilerParams(lowering_semantics=lowering_semantics)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, hidden_tiles, output_tiles),
        grid_names=("expert", "hidden_tile", "output_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_w13_dw13_local_swiglu_mgpu_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    hidden_dim: int,
    intermediate_dim: int,
    lowering_semantics: mgpu.LoweringSemantics,
    use_linear_derivative: bool,
    branch: SourcePushW13LocalDzBranch,
):
    row_tiles = rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = intermediate_dim // output_block

    def body(
        x_ref: Float[pl.Ref, "E C D"],
        d_activation_ref: Float[pl.Ref, "E C I"],
        z_ref: Float[pl.Ref, "E C twoI"],
        dw13_ref: Float[pl.Ref, "E D twoI"],
    ) -> None:
        expert = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        output_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        output_start = output_tile * output_block

        def both_acc_scope(gate_acc_ref, up_acc_ref) -> None:
            def smem_scope(
                x_smem,
                gate_smem,
                up_smem,
                d_activation_smem,
                d_gate_smem,
                d_up_smem,
                ready_barrier,
            ) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    row_slice = pl.ds(row_start, row_block)
                    output_slice = pl.ds(output_start, output_block)
                    up_slice = pl.ds(intermediate_dim + output_start, output_block)

                    mgpu.copy_gmem_to_smem(
                        x_ref.at[
                            expert,
                            row_slice,
                            pl.ds(hidden_start, hidden_block),
                        ],
                        x_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, output_slice],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, up_slice],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_activation_ref.at[expert, row_slice, output_slice],
                        d_activation_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    gate = gate_smem[:, :].astype(jnp.float32)
                    up = up_smem[:, :].astype(jnp.float32)
                    d_activation = d_activation_smem[:, :].astype(jnp.float32)
                    if use_linear_derivative:
                        d_gate = d_activation * up
                        d_up = d_activation * gate
                    else:
                        silu_gate = jax.nn.silu(gate)
                        sigmoid_gate = jax.nn.sigmoid(gate)
                        d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
                        d_gate = d_activation * up * d_silu_gate
                        d_up = d_activation * silu_gate
                    d_gate_smem[:, :] = d_gate.astype(d_gate_smem.dtype)
                    d_up_smem[:, :] = d_up.astype(d_up_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(gate_acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), d_gate_smem)
                    mgpu.wgmma(up_acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), d_up_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                x_smem=_w13_wgmma_smem((row_block, hidden_block), x_ref.dtype),
                gate_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                up_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_activation_smem=_w13_wgmma_smem((row_block, output_block), d_activation_ref.dtype),
                d_gate_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                d_up_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=4),
            )

            dw13_ref[
                expert,
                pl.ds(hidden_start, hidden_block),
                pl.ds(output_start, output_block),
            ] = gate_acc_ref[...]
            dw13_ref[
                expert,
                pl.ds(hidden_start, hidden_block),
                pl.ds(intermediate_dim + output_start, output_block),
            ] = up_acc_ref[...]

        pl.run_scoped(
            both_acc_scope,
            gate_acc_ref=mgpu.ACC((hidden_block, output_block)),
            up_acc_ref=mgpu.ACC((hidden_block, output_block)),
        )

    def branch_body(
        x_ref: Float[pl.Ref, "E C D"],
        d_activation_ref: Float[pl.Ref, "E C I"],
        z_ref: Float[pl.Ref, "E C twoI"],
        dw13_ref: Float[pl.Ref, "E D I"],
    ) -> None:
        expert = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        output_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        output_start = output_tile * output_block

        def acc_scope(acc_ref) -> None:
            def gate_smem_scope(
                x_smem,
                gate_smem,
                up_smem,
                d_activation_smem,
                d_branch_smem,
                ready_barrier,
            ) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    row_slice = pl.ds(row_start, row_block)
                    output_slice = pl.ds(output_start, output_block)
                    up_slice = pl.ds(intermediate_dim + output_start, output_block)

                    mgpu.copy_gmem_to_smem(
                        x_ref.at[
                            expert,
                            row_slice,
                            pl.ds(hidden_start, hidden_block),
                        ],
                        x_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, output_slice],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, up_slice],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_activation_ref.at[expert, row_slice, output_slice],
                        d_activation_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    gate = gate_smem[:, :].astype(jnp.float32)
                    up = up_smem[:, :].astype(jnp.float32)
                    d_activation = d_activation_smem[:, :].astype(jnp.float32)
                    if use_linear_derivative:
                        d_branch = d_activation * up
                    else:
                        sigmoid_gate = jax.nn.sigmoid(gate)
                        d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
                        d_branch = d_activation * up * d_silu_gate
                    d_branch_smem[:, :] = d_branch.astype(d_branch_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), d_branch_smem)
                    mgpu.wgmma_wait(0)

            def up_smem_scope(
                x_smem,
                gate_smem,
                d_activation_smem,
                d_branch_smem,
                ready_barrier,
            ) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    row_slice = pl.ds(row_start, row_block)
                    output_slice = pl.ds(output_start, output_block)

                    mgpu.copy_gmem_to_smem(
                        x_ref.at[
                            expert,
                            row_slice,
                            pl.ds(hidden_start, hidden_block),
                        ],
                        x_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        z_ref.at[expert, row_slice, output_slice],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        d_activation_ref.at[expert, row_slice, output_slice],
                        d_activation_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    gate = gate_smem[:, :].astype(jnp.float32)
                    d_activation = d_activation_smem[:, :].astype(jnp.float32)
                    if use_linear_derivative:
                        d_branch = d_activation * gate
                    else:
                        d_branch = d_activation * jax.nn.silu(gate)
                    d_branch_smem[:, :] = d_branch.astype(d_branch_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), d_branch_smem)
                    mgpu.wgmma_wait(0)

            if branch == "gate":
                pl.run_scoped(
                    gate_smem_scope,
                    x_smem=_w13_wgmma_smem((row_block, hidden_block), x_ref.dtype),
                    gate_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                    up_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                    d_activation_smem=_w13_wgmma_smem((row_block, output_block), d_activation_ref.dtype),
                    d_branch_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                    ready_barrier=mgpu.Barrier(num_arrivals=4),
                )
            else:
                pl.run_scoped(
                    up_smem_scope,
                    x_smem=_w13_wgmma_smem((row_block, hidden_block), x_ref.dtype),
                    gate_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                    d_activation_smem=_w13_wgmma_smem((row_block, output_block), d_activation_ref.dtype),
                    d_branch_smem=_w13_wgmma_smem((row_block, output_block), z_ref.dtype),
                    ready_barrier=mgpu.Barrier(num_arrivals=3),
                )

            dw13_ref[
                expert,
                pl.ds(hidden_start, hidden_block),
                pl.ds(output_start, output_block),
            ] = acc_ref[...]

        pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((hidden_block, output_block)))

    if branch == "both":
        out_shape = jax.ShapeDtypeStruct((experts_per_rank, hidden_dim, 2 * intermediate_dim), jnp.float32)
        kernel_body = body
    elif branch in ("gate", "up"):
        out_shape = jax.ShapeDtypeStruct((experts_per_rank, hidden_dim, intermediate_dim), jnp.float32)
        kernel_body = branch_body
    else:
        raise ValueError(f"branch must be one of 'both', 'gate', or 'up'; got {branch!r}")
    compiler_params = mgpu.CompilerParams(lowering_semantics=lowering_semantics)
    return mgpu.kernel(
        kernel_body,
        out_shape=out_shape,
        grid=(experts_per_rank, hidden_tiles, output_tiles),
        grid_names=("expert", "hidden_tile", "output_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_w13_dw13_source_gather_expert_blocks_mgpu_kernel(
    *,
    ep_size: int,
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
        x_ref: Float[pl.Ref, "T D"],
        d_h_ref: Float[pl.Ref, "E C O"],
        source_rank_ref: Int[pl.Ref, "E C"],
        token_id_ref: Int[pl.Ref, "E C"],
        valid_ref: Int[pl.Ref, "E C"],
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
                def _row_tile_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    row_offsets = jnp.arange(row_block, dtype=jnp.int32)
                    rows_i = row_start + row_offsets
                    valid = valid_ref[expert, rows_i] != 0
                    src = source_rank_ref[expert, rows_i]
                    token = token_id_ref[expert, rows_i]
                    safe_src = jnp.where(valid, src, jnp.zeros((row_block,), dtype=src.dtype))
                    safe_token = jnp.where(valid, token, jnp.zeros((row_block,), dtype=token.dtype))
                    x_tile = jnp.zeros((row_block, hidden_block), dtype=x_ref.dtype)

                    @pl.loop(0, row_block)
                    def _source_row_loop(row) -> None:
                        row_valid = valid[row]
                        row_src = safe_src[row]
                        row_token = safe_token[row]

                        def _source_branch(static_src: int):
                            def _load_source_row(_) -> jax.Array:
                                source_x_ref = mgpu.remote_ref(
                                    x_ref,
                                    static_src,
                                    device_id_type=pl.DeviceIdType.LOGICAL,
                                )
                                return source_x_ref[row_token, pl.ds(hidden_start, hidden_block)]

                            return _load_source_row

                        x_row = lax.switch(row_src, tuple(_source_branch(src_id) for src_id in range(ep_size)), None)
                        x_row = jnp.where(
                            row_valid,
                            x_row,
                            jnp.zeros((hidden_block,), dtype=x_ref.dtype),
                        )
                        nonlocal x_tile
                        x_tile = lax.dynamic_update_slice(x_tile, x_row[None, :], (row, 0))

                    x_smem[:, :] = x_tile

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
                ready_barrier=mgpu.Barrier(num_arrivals=1),
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


def _make_source_push_w13_dw13_source_padded_partials_mgpu_kernel(
    *,
    ep_size: int,
    row_block: int,
    hidden_block: int,
    output_block: int,
    experts_per_rank: int,
    rows: int,
    max_source_rows: int,
    hidden_dim: int,
    output_dim: int,
):
    row_tiles = max_source_rows // row_block
    hidden_tiles = hidden_dim // hidden_block
    output_tiles = output_dim // output_block

    def body(
        x_ref: Float[pl.Ref, "T D"],
        d_h_ref: Float[pl.Ref, "E C O"],
        token_id_ref: Int[pl.Ref, "E C"],
        valid_ref: Int[pl.Ref, "E C"],
        src_base_ref: Int[pl.Ref, "S E"],
        source_lengths_ref: Int[pl.Ref, "S E"],
        partial_ref: Float[pl.Ref, "S E D O"],
    ) -> None:
        src = pl.program_id(0)
        expert = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        output_tile = pl.program_id(3)
        hidden_start = hidden_tile * hidden_block
        output_start = output_tile * output_block

        def _source_branch(static_src: int):
            def _body(_) -> jax.Array:
                source_x_ref = mgpu.remote_ref(
                    x_ref,
                    static_src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                source_base = src_base_ref[static_src, expert]
                source_length = source_lengths_ref[static_src, expert]

                def acc_scope(acc_ref) -> jax.Array:
                    def smem_scope(x_smem, d_h_smem, ready_barrier) -> None:
                        @pl.loop(0, row_tiles)
                        def _row_tile_loop(row_tile) -> None:
                            source_row_start = row_tile * row_block
                            row_start = source_base + source_row_start
                            safe_row_start = jnp.minimum(
                                row_start,
                                jnp.asarray(rows - row_block, dtype=row_start.dtype),
                            )
                            row_offsets = jnp.arange(row_block, dtype=jnp.int32)
                            rows_i = safe_row_start + row_offsets
                            active = source_row_start + row_offsets < source_length
                            valid = (valid_ref[expert, rows_i] != 0) & active
                            safe_token = jnp.where(
                                valid,
                                token_id_ref[expert, rows_i],
                                jnp.zeros((row_block,), dtype=jnp.int32),
                            )
                            x_tile = jnp.zeros((row_block, hidden_block), dtype=x_ref.dtype)

                            @pl.loop(0, row_block)
                            def _source_row_loop(row) -> None:
                                row_valid = valid[row]
                                row_token = safe_token[row]
                                x_row = source_x_ref[row_token, pl.ds(hidden_start, hidden_block)]
                                x_row = jnp.where(
                                    row_valid,
                                    x_row,
                                    jnp.zeros((hidden_block,), dtype=x_ref.dtype),
                                )
                                nonlocal x_tile
                                x_tile = lax.dynamic_update_slice(x_tile, x_row[None, :], (row, 0))

                            x_smem[:, :] = x_tile

                            mgpu.copy_gmem_to_smem(
                                d_h_ref.at[
                                    expert,
                                    pl.ds(safe_row_start, row_block),
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
                        ready_barrier=mgpu.Barrier(num_arrivals=1),
                    )
                    return acc_ref[...]

                return pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((hidden_block, output_block)))

            return _body

        output = lax.switch(src, tuple(_source_branch(src_id) for src_id in range(ep_size)), None)
        partial_ref[
            src,
            expert,
            pl.ds(hidden_start, hidden_block),
            pl.ds(output_start, output_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((ep_size, experts_per_rank, hidden_dim, output_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(ep_size, experts_per_rank, hidden_tiles, output_tiles),
        grid_names=("source", "expert", "hidden_tile", "output_tile"),
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


def _validate_w13_local_swiglu_dw13_request(
    x_expert_major: Array,
    d_activation: Array,
    z: Array,
    valid: Array,
) -> None:
    if x_expert_major.ndim != 4:
        raise ValueError(f"x_expert_major must have shape [dst, expert, capacity, D], got {x_expert_major.shape}")
    if d_activation.ndim != 4:
        raise ValueError(f"d_activation must have shape [dst, expert, capacity, I], got {d_activation.shape}")
    if z.ndim != 4:
        raise ValueError(f"z must have shape [dst, expert, capacity, twoI], got {z.shape}")
    if d_activation.shape[:3] != x_expert_major.shape[:3]:
        raise ValueError(
            f"d_activation leading shape {d_activation.shape[:3]} must match x_expert_major {x_expert_major.shape[:3]}"
        )
    if z.shape[:3] != x_expert_major.shape[:3]:
        raise ValueError(f"z leading shape {z.shape[:3]} must match x_expert_major {x_expert_major.shape[:3]}")
    if z.shape[-1] != 2 * d_activation.shape[-1]:
        raise ValueError(f"z output dim {z.shape[-1]} must be 2 * d_activation dim {d_activation.shape[-1]}")
    if valid.shape != x_expert_major.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match expert rows {x_expert_major.shape[:3]}")


def _validate_w13_local_swiglu_dw13_pallas_request(
    x_expert_major: Array,
    d_activation: Array,
    z: Array,
    valid: Array,
    block_sizes: SourcePushW13BackwardTiledBlockSizes,
) -> None:
    _validate_w13_local_swiglu_dw13_request(x_expert_major, d_activation, z, valid)
    if x_expert_major.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"x_expert_major capacity {x_expert_major.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if x_expert_major.shape[-1] % block_sizes.hidden_block:
        raise ValueError(
            f"x_expert_major hidden dim {x_expert_major.shape[-1]} must be divisible by "
            f"hidden_block={block_sizes.hidden_block}"
        )
    if d_activation.shape[-1] % block_sizes.output_block:
        raise ValueError(
            f"d_activation dim {d_activation.shape[-1]} must be divisible by output_block={block_sizes.output_block}"
        )
    if z.shape[-1] % block_sizes.output_block:
        raise ValueError(f"z output dim {z.shape[-1]} must be divisible by output_block={block_sizes.output_block}")
    _validate_w13_wgmma_smem_shape((block_sizes.row_block, block_sizes.output_block), z.dtype)
    _validate_w13_wgmma_smem_shape((block_sizes.row_block, block_sizes.hidden_block), x_expert_major.dtype)


def _w13_wgmma_transforms(dtype):
    return (
        mgpu.TilingTransform((W13_WGMMA_TILE_M, W13_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize)),
        mgpu.SwizzleTransform(W13_WGMMA_SWIZZLE_BYTES),
    )


def _w13_wgmma_smem(shape: tuple[int, int], dtype):
    _validate_w13_wgmma_smem_shape(shape, dtype)
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=_w13_wgmma_transforms(dtype),
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
