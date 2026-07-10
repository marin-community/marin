# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas scaffolding for semantic source-push W13 backward kernels.

This module implements the first correctness-oriented W13 backward Pallas
boundary for the pair-flat source-push contract:

* ``dx_pair[S, Dst, R, H] = dz_pair[S, Dst, R, 2I] @ W13[Dst, expert].T``
* ``dw13[Dst, E, H, 2I] = sum_rows x_pair[row, H] * dz_pair[row, 2I]``

The pair-flat rows are grouped by local expert in each ``(source,
destination)`` pair. The kernels derive each row's local expert from
``pair_expert_base/xcounts`` instead of consuming a dense ``expert_ids`` array.

These scaffolds do not use peer-id GMEM refs. ``Warpgroup`` lowering is the
default so the matmul-like local compute path can use Mosaic GPU's WGMMA
lowering where supported; ``Lane`` remains caller-selectable for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushSemanticPlan,
    source_push_semantic_expert_major_to_pair_jax,
    source_push_semantic_gather_x_jax,
    source_push_semantic_pair_to_expert_major_jax,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


DEFAULT_SEMANTIC_W13_BACKWARD_ROW_BLOCK = 64
DEFAULT_SEMANTIC_W13_BACKWARD_HIDDEN_BLOCK = 256
DEFAULT_SEMANTIC_W13_BACKWARD_OUTPUT_BLOCK = 64
SEMANTIC_W13_WGMMA_SWIZZLE_BYTES = 128
SEMANTIC_W13_WGMMA_TILE_M = 8


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW13BackwardPallasBlockSizes:
    """Tile sizes for the pair-flat semantic W13 backward Pallas scaffold."""

    row_block: int = DEFAULT_SEMANTIC_W13_BACKWARD_ROW_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W13_BACKWARD_HIDDEN_BLOCK
    output_block: int = DEFAULT_SEMANTIC_W13_BACKWARD_OUTPUT_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW13BackwardPallasBlockSizes":
        return cls()


def source_push_semantic_w13_backward_pallas_scaffold_mgpu(
    x: Float[Array, "S T H"],
    dz_pair: Float[Array, "S Dst R twoI"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
    rows_per_expert_capacity: int | None = None,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S Dst R H"], Float[Array, "Dst E H twoI"]]:
    """Compute semantic W13 backward ``dx_pair`` and ``dw13`` with Pallas."""

    dx_pair = source_push_semantic_w13_backward_dx_pair_pallas_mgpu(
        dz_pair,
        w_gate_up,
        plan,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        rows_per_expert_capacity=rows_per_expert_capacity,
        mesh=mesh,
    )
    dw13 = source_push_semantic_w13_backward_dw13_pallas_mgpu(
        x,
        dz_pair,
        plan,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        rows_per_expert_capacity=rows_per_expert_capacity,
        mesh=mesh,
    )
    return dx_pair, dw13


def source_push_semantic_w13_backward_expert_major_reference_jax(
    x_expert: Float[Array, "Dst E C H"],
    dz13: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E H twoI"],
    valid: Bool[Array, "Dst E C"],
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "Dst E H twoI"]]:
    """Reference W13 backward on the production expert-major contract."""

    valid_f = valid.astype(jnp.float32)
    x_expert = x_expert.astype(jnp.float32) * valid_f[..., None]
    dz13 = dz13.astype(jnp.float32) * valid_f[..., None]
    w13 = w13.astype(jnp.float32)
    dx_route = jnp.einsum("deco,deho->dech", dz13, w13, preferred_element_type=jnp.float32)
    dw13 = jnp.einsum("dech,deco->deho", x_expert, dz13, preferred_element_type=jnp.float32)
    return dx_route * valid_f[..., None], dw13


def source_push_semantic_w13_backward_expert_major_pallas_mgpu(
    x_expert: Float[Array, "Dst E C H"],
    dz13: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E H twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "Dst E H twoI"]]:
    """Compute expert-major W13 backward using explicit Mosaic GPU WGMMA kernels."""

    dx_route = source_push_semantic_w13_backward_dx_route_expert_major_pallas_mgpu(
        dz13,
        w13,
        valid,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        mesh=mesh,
    )
    dw13 = source_push_semantic_w13_backward_dw13_expert_major_pallas_mgpu(
        x_expert,
        dz13,
        valid,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        mesh=mesh,
    )
    return dx_route, dw13


def source_push_semantic_w13_backward_dx_route_expert_major_pallas_mgpu(
    dz13: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E H twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C H"]:
    """Compute expert-major W13 input gradients with explicit Mosaic GPU WGMMA."""

    _validate_expert_major_w13_dx_shapes(dz13, w13, valid)
    if interpret:
        valid_f = valid.astype(jnp.float32)
        dx_route = jnp.einsum(
            "deco,deho->dech",
            dz13.astype(jnp.float32) * valid_f[..., None],
            w13.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        return dx_route * valid_f[..., None]
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W13 backward requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU semantic source-push W13 backward requires an explicit mesh")
    block_sizes = SourcePushSemanticW13BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    original_rows = dz13.shape[2]
    dz13, valid = _pad_expert_major_dz_rows_for_w13_backward(
        dz13,
        valid,
        row_multiple=block_sizes.row_block,
    )
    _validate_expert_major_w13_dx_pallas_request(dz13, w13, valid, block_sizes)
    dz13_for_wgmma = jnp.where(valid[..., None], dz13, jnp.zeros((), dtype=dz13.dtype)).astype(w13.dtype)
    dx_route = _source_push_semantic_w13_backward_dx_route_sharded_mgpu_kernel(
        mesh,
        dz13_for_wgmma,
        w13,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
        lowering_semantics=lowering_semantics,
    )
    return dx_route[:, :, :original_rows, :]


def source_push_semantic_w13_backward_dw13_expert_major_pallas_mgpu(
    x_expert: Float[Array, "Dst E C H"],
    dz13: Float[Array, "Dst E C twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E H twoI"]:
    """Compute expert-major W13 weight gradients with explicit Mosaic GPU WGMMA."""

    _validate_expert_major_w13_dw13_shapes(x_expert, dz13, valid)
    if interpret:
        valid_f = valid.astype(jnp.float32)
        return jnp.einsum(
            "dech,deco->deho",
            x_expert.astype(jnp.float32) * valid_f[..., None],
            dz13.astype(jnp.float32) * valid_f[..., None],
            preferred_element_type=jnp.float32,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W13 backward requires a GPU backend")
    if mesh is None:
        raise ValueError("Pallas/MGPU semantic source-push W13 backward requires an explicit mesh")
    block_sizes = SourcePushSemanticW13BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    x_expert, dz13, valid = _pad_expert_major_rows_for_w13_backward(
        x_expert,
        dz13,
        valid,
        row_multiple=block_sizes.row_block,
    )
    _validate_expert_major_w13_dw13_pallas_request(x_expert, dz13, valid, block_sizes)
    dz13_for_wgmma = jnp.where(valid[..., None], dz13, jnp.zeros((), dtype=dz13.dtype)).astype(x_expert.dtype)
    x_for_wgmma = jnp.where(valid[..., None], x_expert, jnp.zeros((), dtype=x_expert.dtype))
    dw13 = _source_push_semantic_w13_backward_dw13_sharded_mgpu_kernel(
        mesh,
        x_for_wgmma,
        dz13_for_wgmma,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_block=block_sizes.output_block,
        lowering_semantics=lowering_semantics,
    )
    return dw13


def source_push_semantic_w13_backward_dx_pair_pallas_mgpu(
    dz_pair: Float[Array, "S Dst R twoI"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
    rows_per_expert_capacity: int | None = None,
    mesh: Mesh | None = None,
) -> Float[Array, "S Dst R H"]:
    """Compute pair-flat route-level ``dx`` for semantic W13 backward."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W13 backward dx requires a GPU backend")
    block_sizes = SourcePushSemanticW13BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w13_backward_pallas_request(None, dz_pair, w_gate_up, plan, block_sizes)
    rows_per_expert_capacity = _semantic_w13_rows_per_expert_capacity(dz_pair, rows_per_expert_capacity)
    dz13, valid = source_push_semantic_pair_to_expert_major_jax(
        dz_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dx_route = source_push_semantic_w13_backward_dx_route_expert_major_pallas_mgpu(
        dz13,
        w_gate_up,
        valid,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        mesh=mesh,
    )
    return source_push_semantic_expert_major_to_pair_jax(dx_route, plan)


def source_push_semantic_w13_backward_dw13_pallas_mgpu(
    x: Float[Array, "S T H"],
    dz_pair: Float[Array, "S Dst R twoI"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
    rows_per_expert_capacity: int | None = None,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E H twoI"]:
    """Compute semantic W13 weight gradient from source tokens and pair rows."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W13 backward dw13 requires a GPU backend")
    block_sizes = SourcePushSemanticW13BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w13_backward_pallas_request(x, dz_pair, None, plan, block_sizes)
    rows_per_expert_capacity = _semantic_w13_rows_per_expert_capacity(dz_pair, rows_per_expert_capacity)
    x_pair = source_push_semantic_gather_x_jax(x, plan)
    x_expert, valid = source_push_semantic_pair_to_expert_major_jax(
        x_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dz13, _dz_valid = source_push_semantic_pair_to_expert_major_jax(
        dz_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    dw13 = source_push_semantic_w13_backward_dw13_expert_major_pallas_mgpu(
        x_expert,
        dz13,
        valid,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        mesh=mesh,
    )
    return dw13


def _source_push_semantic_w13_backward_dx_pair_pallas_call(
    dz_pair: Float[Array, "S Dst R twoI"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    interpret: bool,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "S Dst R H"]:
    source_count, dst_count, rows_per_pair, output_dim = dz_pair.shape
    hidden_dim = w_gate_up.shape[2]
    output_shape = jax.ShapeDtypeStruct((source_count, dst_count, rows_per_pair, hidden_dim), jnp.float32)
    valid_mask_i32 = valid_mask.astype(jnp.int32)
    input_specs, output_specs = _source_push_semantic_w13_backward_dx_pair_block_specs()
    cost_estimate = _source_push_semantic_w13_backward_dx_pair_cost_estimate(
        dz_pair,
        w_gate_up,
        xcounts,
        pair_expert_base,
        valid_mask_i32,
        output_shape,
    )
    return pl.pallas_call(
        _make_source_push_semantic_w13_backward_dx_pair_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            output_block=output_block,
            output_dim=output_dim,
            experts_per_rank=w_gate_up.shape[1],
        ),
        in_specs=input_specs,
        out_specs=output_specs,
        out_shape=output_shape,
        grid=(source_count, dst_count, rows_per_pair // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_w13_backward_dx_pair_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
        cost_estimate=cost_estimate,
    )(dz_pair, w_gate_up, xcounts, pair_expert_base, valid_mask_i32)


def _source_push_semantic_w13_backward_dw13_pallas_call(
    x: Float[Array, "S T H"],
    dz_pair: Float[Array, "S Dst R twoI"],
    token_ids: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    experts_per_rank: int,
    hidden_block: int,
    output_block: int,
    interpret: bool,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E H twoI"]:
    _source_count, dst_count, _rows_per_pair, output_dim = dz_pair.shape
    hidden_dim = x.shape[-1]
    output_shape = jax.ShapeDtypeStruct((dst_count, experts_per_rank, hidden_dim, output_dim), jnp.float32)
    valid_mask_i32 = valid_mask.astype(jnp.int32)
    input_specs, output_specs = _source_push_semantic_w13_backward_dw13_block_specs()
    cost_estimate = _source_push_semantic_w13_backward_dw13_cost_estimate(
        x,
        dz_pair,
        token_ids,
        xcounts,
        pair_expert_base,
        valid_mask_i32,
        output_shape,
    )
    return pl.pallas_call(
        _make_source_push_semantic_w13_backward_dw13_kernel(
            hidden_block=hidden_block,
            output_block=output_block,
            source_count=dz_pair.shape[0],
            rows_per_pair=dz_pair.shape[2],
        ),
        in_specs=input_specs,
        out_specs=output_specs,
        out_shape=output_shape,
        grid=(dst_count, experts_per_rank, hidden_dim // hidden_block, output_dim // output_block),
        interpret=interpret,
        name="source_push_semantic_w13_backward_dw13_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
        cost_estimate=cost_estimate,
    )(x, dz_pair, token_ids, xcounts, pair_expert_base, valid_mask_i32)


def _source_push_semantic_w13_backward_dx_pair_block_specs() -> tuple[tuple[pl.BlockSpec, ...], pl.BlockSpec]:
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec), gmem_spec


def _source_push_semantic_w13_backward_dw13_block_specs() -> tuple[tuple[pl.BlockSpec, ...], pl.BlockSpec]:
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec), gmem_spec


def _semantic_w13_rows_per_expert_capacity(
    dz_pair: Array,
    rows_per_expert_capacity: int | None,
) -> int:
    if rows_per_expert_capacity is not None:
        if rows_per_expert_capacity <= 0:
            raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
        return rows_per_expert_capacity
    return dz_pair.shape[0] * dz_pair.shape[2]


def _pad_expert_major_rows_for_w13_backward(
    x_expert: Float[Array, "Dst E C H"],
    dz13: Float[Array, "Dst E C twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_multiple: int,
) -> tuple[Float[Array, "Dst E C H"], Float[Array, "Dst E C twoI"], Bool[Array, "Dst E C"]]:
    rows = x_expert.shape[2]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    pad_rows = padded_rows - rows
    if pad_rows == 0:
        return x_expert, dz13, valid
    x_expert = jnp.pad(x_expert, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    dz13 = jnp.pad(dz13, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad_rows)))
    return x_expert, dz13, valid


def _pad_expert_major_dz_rows_for_w13_backward(
    dz13: Float[Array, "Dst E C twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_multiple: int,
) -> tuple[Float[Array, "Dst E C twoI"], Bool[Array, "Dst E C"]]:
    rows = dz13.shape[2]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    pad_rows = padded_rows - rows
    if pad_rows == 0:
        return dz13, valid
    dz13 = jnp.pad(dz13, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad_rows)))
    return dz13, valid


def _source_push_semantic_w13_backward_dx_route_sharded_mgpu_kernel(
    mesh: Mesh,
    dz13: Float[Array, "Dst E C twoI"],
    w13: Float[Array, "Dst E H twoI"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_w13_backward_dx_route_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=dz13.shape[1],
        rows=dz13.shape[2],
        hidden_dim=w13.shape[2],
        output_dim=dz13.shape[3],
        lowering_semantics=lowering_semantics,
    )

    def local_fn(
        dz13_local: Float[Array, "1 E C twoI"],
        w13_local: Float[Array, "1 E H twoI"],
    ) -> Float[Array, "1 E C H"]:
        return kernel(dz13_local[0], w13_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dz13, w13)


def _source_push_semantic_w13_backward_dw13_sharded_mgpu_kernel(
    mesh: Mesh,
    x_expert: Float[Array, "Dst E C H"],
    dz13: Float[Array, "Dst E C twoI"],
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E H twoI"]:
    kernel = _make_source_push_semantic_w13_backward_dw13_mgpu_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_block=output_block,
        experts_per_rank=x_expert.shape[1],
        rows=x_expert.shape[2],
        hidden_dim=x_expert.shape[3],
        output_dim=dz13.shape[3],
        lowering_semantics=lowering_semantics,
    )

    def local_fn(
        x_local: Float[Array, "1 E C H"],
        dz13_local: Float[Array, "1 E C twoI"],
    ) -> Float[Array, "1 E H twoI"]:
        return kernel(x_local[0], dz13_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(x_expert, dz13)


def _make_source_push_semantic_w13_backward_dx_route_mgpu_kernel(
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
        dz13_ref: Float[pl.Ref, "E C twoI"],
        w13_ref: Float[pl.Ref, "E H twoI"],
        dx_route_ref: Float[pl.Ref, "E C H"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(dz_smem, w_smem, ready_barrier) -> None:
                @pl.loop(0, output_tiles)
                def _output_loop(output_tile) -> None:
                    output_start = output_tile * output_block
                    mgpu.copy_gmem_to_smem(
                        dz13_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(output_start, output_block),
                        ],
                        dz_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w13_ref.at[
                            expert,
                            pl.ds(hidden_start, hidden_block),
                            pl.ds(output_start, output_block),
                        ],
                        w_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, dz_smem, mgpu.transpose_ref(w_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                dz_smem=_semantic_w13_wgmma_smem((row_block, output_block), dz13_ref.dtype),
                w_smem=_semantic_w13_wgmma_smem((hidden_block, output_block), w13_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((row_block, hidden_block)))
        dx_route_ref[
            expert,
            pl.ds(row_start, row_block),
            pl.ds(hidden_start, hidden_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, rows, hidden_dim), jnp.float32)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, hidden_tiles),
        grid_names=("expert", "row_tile", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def _make_source_push_semantic_w13_backward_dw13_mgpu_kernel(
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
        x_ref: Float[pl.Ref, "E C H"],
        dz13_ref: Float[pl.Ref, "E C twoI"],
        dw13_ref: Float[pl.Ref, "E H twoI"],
    ) -> None:
        expert = pl.program_id(0)
        hidden_tile = pl.program_id(1)
        output_tile = pl.program_id(2)
        hidden_start = hidden_tile * hidden_block
        output_start = output_tile * output_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(x_smem, dz_smem, ready_barrier) -> None:
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
                        dz13_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(output_start, output_block),
                        ],
                        dz_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(x_smem, (1, 0)), dz_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                x_smem=_semantic_w13_wgmma_smem((row_block, hidden_block), x_ref.dtype),
                dz_smem=_semantic_w13_wgmma_smem((row_block, output_block), dz13_ref.dtype),
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
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, hidden_tiles, output_tiles),
        grid_names=("expert", "hidden_tile", "output_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def _semantic_w13_wgmma_smem(shape: tuple[int, int], dtype):
    swizzle_elems = SEMANTIC_W13_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % SEMANTIC_W13_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "Semantic W13 WGMMA SMEM operands must be divisible by "
            f"tile=({SEMANTIC_W13_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((SEMANTIC_W13_WGMMA_TILE_M, swizzle_elems)),
            mgpu.SwizzleTransform(SEMANTIC_W13_WGMMA_SWIZZLE_BYTES),
        ),
    )


def _make_source_push_semantic_w13_backward_dx_pair_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_block: int,
    output_dim: int,
    experts_per_rank: int,
):
    def kernel(
        dz_pair_ref: Float[pl.Ref, "S Dst R twoI"],
        w_gate_up_ref: Float[pl.Ref, "Dst E H twoI"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        dx_pair_ref: Float[pl.Ref, "S Dst R H"],
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
            acc = jnp.zeros((hidden_block,), dtype=jnp.float32)
            for output_start in range(0, output_dim, output_block):
                for output_offset in range(output_block):
                    output = output_start + output_offset
                    dz_scalar = dz_pair_ref[src, dst, row, output].astype(jnp.float32)
                    w_vec = w_gate_up_ref[
                        pl.ds(dst, 1),
                        pl.ds(expert, 1),
                        pl.ds(hidden_start, hidden_block),
                        pl.ds(output, 1),
                    ][0, 0, :, 0].astype(jnp.float32)
                    acc += dz_scalar * w_vec

            dx_pair_ref[
                pl.ds(src, 1),
                pl.ds(dst, 1),
                pl.ds(row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(
                row_is_valid, acc, jnp.zeros_like(acc)
            )[None, None, None, :]

    return kernel


def _make_source_push_semantic_w13_backward_dw13_kernel(
    *,
    hidden_block: int,
    output_block: int,
    source_count: int,
    rows_per_pair: int,
):
    def kernel(
        x_ref: Float[pl.Ref, "S T H"],
        dz_pair_ref: Float[pl.Ref, "S Dst R twoI"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        dw13_ref: Float[pl.Ref, "Dst E H twoI"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        output_tile = pl.program_id(3)
        hidden_start = hidden_tile * hidden_block
        output_start = output_tile * output_block

        acc = jnp.zeros((hidden_block, output_block), dtype=jnp.float32)
        for src in range(source_count):
            expert_base = pair_expert_base_ref[src, dst, expert]
            expert_count = xcounts_ref[src, dst, expert]
            for row in range(rows_per_pair):
                in_expert_interval = (row >= expert_base) & (row < expert_base + expert_count)
                row_is_valid = (valid_mask_ref[src, dst, row] != 0) & in_expert_interval
                safe_token = jnp.maximum(token_ids_ref[src, dst, row], 0)
                x_vec = x_ref[
                    pl.ds(src, 1),
                    pl.ds(safe_token, 1),
                    pl.ds(hidden_start, hidden_block),
                ][
                    0, 0, :
                ].astype(jnp.float32)
                for output_offset in range(output_block):
                    output = output_start + output_offset
                    dz_scalar = dz_pair_ref[src, dst, row, output].astype(jnp.float32)
                    outer_col = x_vec * dz_scalar
                    acc = acc.at[:, output_offset].add(jnp.where(row_is_valid, outer_col, jnp.zeros_like(outer_col)))

        dw13_ref[
            pl.ds(dst, 1),
            pl.ds(expert, 1),
            pl.ds(hidden_start, hidden_block),
            pl.ds(output_start, output_block),
        ] = acc[None, None, :, :]

    return kernel


def _source_push_semantic_w13_backward_dx_pair_reference(
    dz_pair: Array,
    w_gate_up: Array,
    xcounts: Array,
    valid_mask: Array,
) -> Array:
    rows = jnp.arange(dz_pair.shape[2], dtype=jnp.int32)
    pair_ends = jnp.cumsum(xcounts, axis=2, dtype=jnp.int32)

    def pair_expert_ids(ends):
        expert = jnp.searchsorted(ends, rows, side="right").astype(jnp.int32)
        return jnp.minimum(expert, xcounts.shape[-1] - 1)

    expert_ids = jax.vmap(jax.vmap(pair_expert_ids, in_axes=0), in_axes=0)(pair_ends)
    dst_index = jnp.arange(dz_pair.shape[1], dtype=jnp.int32)[None, :, None]
    w_pair = w_gate_up.at[dst_index, expert_ids].get().astype(jnp.float32)
    dx_pair = jnp.einsum(
        "sdro,sdrho->sdrh",
        dz_pair.astype(jnp.float32),
        w_pair,
        preferred_element_type=jnp.float32,
    )
    return jnp.where(valid_mask[..., None] != 0, dx_pair, jnp.zeros((), dtype=dx_pair.dtype))


def _source_push_semantic_w13_backward_dw13_reference(
    x: Array,
    dz_pair: Array,
    token_ids: Array,
    xcounts: Array,
    valid_mask: Array,
) -> Array:
    source_index = jnp.arange(token_ids.shape[0], dtype=jnp.int32)[:, None, None]
    safe_tokens = jnp.maximum(token_ids, 0)
    x_pair = x.at[source_index, safe_tokens].get().astype(jnp.float32)
    rows = jnp.arange(dz_pair.shape[2], dtype=jnp.int32)
    pair_ends = jnp.cumsum(xcounts, axis=2, dtype=jnp.int32)

    def pair_expert_ids(ends):
        expert = jnp.searchsorted(ends, rows, side="right").astype(jnp.int32)
        return jnp.minimum(expert, xcounts.shape[-1] - 1)

    expert_ids = jax.vmap(jax.vmap(pair_expert_ids, in_axes=0), in_axes=0)(pair_ends)
    dw13_parts = []
    for expert in range(xcounts.shape[2]):
        mask = (expert_ids == expert) & (valid_mask != 0)
        x_expert = jnp.where(mask[..., None], x_pair, jnp.zeros((), dtype=jnp.float32))
        dz_expert = jnp.where(mask[..., None], dz_pair.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))
        dw13_parts.append(jnp.einsum("sdrh,sdro->dho", x_expert, dz_expert, preferred_element_type=jnp.float32))
    return jnp.stack(dw13_parts, axis=1)


def _source_push_semantic_w13_backward_dx_pair_cost_estimate(
    dz_pair: Array,
    w_gate_up: Array,
    xcounts: Array,
    pair_expert_base: Array,
    valid_mask_i32: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(dz_pair.shape, dz_pair.dtype),
        jax.ShapeDtypeStruct(w_gate_up.shape, w_gate_up.dtype),
        jax.ShapeDtypeStruct(xcounts.shape, xcounts.dtype),
        jax.ShapeDtypeStruct(pair_expert_base.shape, pair_expert_base.dtype),
        jax.ShapeDtypeStruct(valid_mask_i32.shape, valid_mask_i32.dtype),
    )

    def reference(dz_pair_spec, w_gate_up_spec, xcounts_spec, _pair_expert_base_spec, valid_mask_spec):
        return _source_push_semantic_w13_backward_dx_pair_reference(
            dz_pair_spec,
            w_gate_up_spec,
            xcounts_spec,
            valid_mask_spec,
        )

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_semantic_w13_backward_dw13_cost_estimate(
    x: Array,
    dz_pair: Array,
    token_ids: Array,
    xcounts: Array,
    pair_expert_base: Array,
    valid_mask_i32: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(dz_pair.shape, dz_pair.dtype),
        jax.ShapeDtypeStruct(token_ids.shape, token_ids.dtype),
        jax.ShapeDtypeStruct(xcounts.shape, xcounts.dtype),
        jax.ShapeDtypeStruct(pair_expert_base.shape, pair_expert_base.dtype),
        jax.ShapeDtypeStruct(valid_mask_i32.shape, valid_mask_i32.dtype),
    )

    def reference(x_spec, dz_pair_spec, token_ids_spec, xcounts_spec, _pair_expert_base_spec, valid_mask_spec):
        return _source_push_semantic_w13_backward_dw13_reference(
            x_spec,
            dz_pair_spec,
            token_ids_spec,
            xcounts_spec,
            valid_mask_spec,
        )

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _validate_semantic_w13_backward_pallas_request(
    x: Array | None,
    dz_pair: Array,
    w_gate_up: Array | None,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes,
) -> None:
    if dz_pair.ndim != 4:
        raise ValueError(f"dz_pair must have shape [source, destination, row, 2 * intermediate], got {dz_pair.shape}")
    if plan.valid_mask.shape != dz_pair.shape[:3]:
        raise ValueError(
            f"plan valid_mask shape {plan.valid_mask.shape} must match dz_pair route shape {dz_pair.shape[:3]}"
        )
    if plan.xcounts.shape[:2] != dz_pair.shape[:2]:
        raise ValueError(
            f"plan xcounts source/destination shape {plan.xcounts.shape[:2]} must match {dz_pair.shape[:2]}"
        )
    if plan.pair_expert_base.shape != plan.xcounts.shape:
        raise ValueError(
            f"plan pair_expert_base shape {plan.pair_expert_base.shape} must match xcounts shape {plan.xcounts.shape}"
        )
    if x is not None:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
        if x.shape[0] != dz_pair.shape[0]:
            raise ValueError(f"x source dim {x.shape[0]} must match dz_pair source dim {dz_pair.shape[0]}")
        if x.shape[1] != plan.tokens_per_source:
            raise ValueError(f"x token dim {x.shape[1]} must match plan tokens_per_source={plan.tokens_per_source}")
        if x.shape[2] % block_sizes.hidden_block:
            raise ValueError(f"x hidden dim {x.shape[2]} must be divisible by hidden_block={block_sizes.hidden_block}")
        if plan.token_ids.shape != dz_pair.shape[:3]:
            raise ValueError(
                f"plan token_ids shape {plan.token_ids.shape} must match dz_pair route shape {dz_pair.shape[:3]}"
            )
    if w_gate_up is not None:
        if w_gate_up.ndim != 4:
            raise ValueError(
                f"w_gate_up must have shape [destination, expert, hidden, 2 * intermediate], got {w_gate_up.shape}"
            )
        if w_gate_up.shape[0] != dz_pair.shape[1]:
            raise ValueError(
                f"w_gate_up destination dim {w_gate_up.shape[0]} must match dz_pair destination dim {dz_pair.shape[1]}"
            )
        if w_gate_up.shape[1] != plan.xcounts.shape[2]:
            raise ValueError(
                f"w_gate_up expert dim {w_gate_up.shape[1]} must match plan expert dim {plan.xcounts.shape[2]}"
            )
        if w_gate_up.shape[3] != dz_pair.shape[3]:
            raise ValueError(
                f"w_gate_up output dim {w_gate_up.shape[3]} must match dz_pair output dim {dz_pair.shape[3]}"
            )
        if w_gate_up.shape[2] % block_sizes.hidden_block:
            raise ValueError(
                f"w_gate_up hidden dim {w_gate_up.shape[2]} must be divisible by hidden_block="
                f"{block_sizes.hidden_block}"
            )
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if dz_pair.shape[3] % block_sizes.output_block:
        raise ValueError(
            f"dz_pair output dim {dz_pair.shape[3]} must be divisible by output_block={block_sizes.output_block}"
        )


def _validate_expert_major_w13_backward_shapes(
    x_expert: Array,
    dz13: Array,
    w13: Array,
    valid: Array,
) -> None:
    _validate_expert_major_w13_dw13_shapes(x_expert, dz13, valid)
    _validate_expert_major_w13_dx_shapes(dz13, w13, valid)


def _validate_expert_major_w13_dx_shapes(
    dz13: Array,
    w13: Array,
    valid: Array,
) -> None:
    if dz13.ndim != 4:
        raise ValueError(f"dz13 must have shape [destination, expert, row, 2 * intermediate], got {dz13.shape}")
    if w13.ndim != 4:
        raise ValueError(f"w13 must have shape [destination, expert, hidden, 2 * intermediate], got {w13.shape}")
    if valid.shape != dz13.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match dz13 row shape {dz13.shape[:3]}")
    if w13.shape[:2] != dz13.shape[:2]:
        raise ValueError(f"w13 leading shape {w13.shape[:2]} must match dz13 leading shape {dz13.shape[:2]}")
    if w13.shape[3] != dz13.shape[3]:
        raise ValueError(f"w13 output dim {w13.shape[3]} must match dz13 output dim {dz13.shape[3]}")


def _validate_expert_major_w13_dw13_shapes(
    x_expert: Array,
    dz13: Array,
    valid: Array,
) -> None:
    if x_expert.ndim != 4:
        raise ValueError(f"x_expert must have shape [destination, expert, row, hidden], got {x_expert.shape}")
    if dz13.ndim != 4:
        raise ValueError(f"dz13 must have shape [destination, expert, row, 2 * intermediate], got {dz13.shape}")
    if valid.shape != x_expert.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match x_expert row shape {x_expert.shape[:3]}")
    if dz13.shape[:3] != x_expert.shape[:3]:
        raise ValueError(f"dz13 row shape {dz13.shape[:3]} must match x_expert row shape {x_expert.shape[:3]}")


def _validate_expert_major_w13_backward_pallas_request(
    x_expert: Array,
    dz13: Array,
    w13: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes,
) -> None:
    _validate_expert_major_w13_backward_shapes(x_expert, dz13, w13, valid)
    _validate_expert_major_w13_dw13_pallas_request(x_expert, dz13, valid, block_sizes)
    _validate_expert_major_w13_dx_pallas_request(dz13, w13, valid, block_sizes)


def _validate_expert_major_w13_dx_pallas_request(
    dz13: Array,
    w13: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes,
) -> None:
    _validate_expert_major_w13_dx_shapes(dz13, w13, valid)
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if dz13.shape[2] % block_sizes.row_block:
        raise ValueError(f"dz13 row dim {dz13.shape[2]} must be divisible by row_block={block_sizes.row_block}")
    if w13.shape[2] % block_sizes.hidden_block:
        raise ValueError(f"w13 hidden dim {w13.shape[2]} must be divisible by hidden_block={block_sizes.hidden_block}")
    if dz13.shape[3] % block_sizes.output_block:
        raise ValueError(
            f"dz13 output dim {dz13.shape[3]} must be divisible by output_block={block_sizes.output_block}"
        )
    _semantic_w13_wgmma_smem((block_sizes.row_block, block_sizes.output_block), dz13.dtype)
    _semantic_w13_wgmma_smem((block_sizes.hidden_block, block_sizes.output_block), w13.dtype)


def _validate_expert_major_w13_dw13_pallas_request(
    x_expert: Array,
    dz13: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW13BackwardPallasBlockSizes,
) -> None:
    _validate_expert_major_w13_dw13_shapes(x_expert, dz13, valid)
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.output_block <= 0:
        raise ValueError(f"output_block must be positive, got {block_sizes.output_block}")
    if x_expert.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"x_expert row dim {x_expert.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if x_expert.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"x_expert hidden dim {x_expert.shape[3]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )
    if dz13.shape[3] % block_sizes.output_block:
        raise ValueError(
            f"dz13 output dim {dz13.shape[3]} must be divisible by output_block={block_sizes.output_block}"
        )
    _semantic_w13_wgmma_smem((block_sizes.row_block, block_sizes.output_block), dz13.dtype)
    _semantic_w13_wgmma_smem((block_sizes.row_block, block_sizes.hidden_block), x_expert.dtype)
