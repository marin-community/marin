# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas kernels for semantic source-push W2 backward.

This module targets the pair-flat source-push contract:

* ``h_pair[S, Dst, R, I]``
* ``dy_route[S, Dst, R, H]``
* ``w_down[Dst, E, I, H]``

Rows for a given ``(source, destination, local_expert)`` occupy the contiguous
interval ``[pair_expert_base, pair_expert_base + xcounts)``. The legacy
pair-flat scaffold keeps the lowering straightforward:

* ``dh``: one program owns a route-row/intermediate tile and scans hidden tiles.
* ``dw2``: one program owns a destination/expert weight tile and reduces all
  source rows for that expert, avoiding atomics.

The production-relevant expert-major path operates on ``[Dst, E, C, *]``
buffers and uses explicit Mosaic GPU WGMMA for both ``dh = dy @ W2.T`` and
``dw2 = H.T @ dy``.
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

from levanter.grug._moe.source_push_plan import SOURCE_PUSH_MESH_AXIS, SourcePushSemanticPlan
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


DEFAULT_SEMANTIC_W2_BACKWARD_ROW_BLOCK = 1
DEFAULT_SEMANTIC_W2_BACKWARD_INTERMEDIATE_BLOCK = 64
DEFAULT_SEMANTIC_W2_BACKWARD_HIDDEN_BLOCK = 64
DEFAULT_SEMANTIC_W2_BACKWARD_EXPERT_MAJOR_ROW_BLOCK = 128
DEFAULT_SEMANTIC_W2_BACKWARD_EXPERT_MAJOR_INTERMEDIATE_BLOCK = 64
DEFAULT_SEMANTIC_W2_BACKWARD_EXPERT_MAJOR_HIDDEN_BLOCK = 128
SEMANTIC_W2_BACKWARD_WGMMA_SWIZZLE_BYTES = 128
SEMANTIC_W2_BACKWARD_WGMMA_TILE_M = 8


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW2BackwardPallasBlockSizes:
    """Tile sizes for the pair-flat semantic W2 backward Pallas scaffold."""

    row_block: int = DEFAULT_SEMANTIC_W2_BACKWARD_ROW_BLOCK
    intermediate_block: int = DEFAULT_SEMANTIC_W2_BACKWARD_INTERMEDIATE_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W2_BACKWARD_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW2BackwardPallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes:
    """Tile sizes for the expert-major semantic W2 backward WGMMA kernels."""

    row_block: int = DEFAULT_SEMANTIC_W2_BACKWARD_EXPERT_MAJOR_ROW_BLOCK
    intermediate_block: int = DEFAULT_SEMANTIC_W2_BACKWARD_EXPERT_MAJOR_INTERMEDIATE_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W2_BACKWARD_EXPERT_MAJOR_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes":
        return cls()


def source_push_semantic_w2_backward_expert_major_reference_jax(
    h_expert: Float[Array, "Dst E C I"],
    dy_expert: Float[Array, "Dst E C H"],
    w_down: Float[Array, "Dst E I H"],
    valid: Bool[Array, "Dst E C"],
) -> tuple[Float[Array, "Dst E C I"], Float[Array, "Dst E I H"]]:
    """Reference W2 backward on destination expert-major rows."""

    _validate_semantic_w2_backward_expert_major_shapes(h_expert, dy_expert, w_down, valid)
    valid_f = valid.astype(jnp.float32)
    h_expert = h_expert.astype(jnp.float32) * valid_f[..., None]
    dy_expert = dy_expert.astype(jnp.float32) * valid_f[..., None]
    w_down = w_down.astype(jnp.float32)
    dh_expert = jnp.einsum("dech,deih->deci", dy_expert, w_down, preferred_element_type=jnp.float32)
    dw2 = jnp.einsum("deci,dech->deih", h_expert, dy_expert, preferred_element_type=jnp.float32)
    return dh_expert * valid_f[..., None], dw2


def source_push_semantic_w2_backward_expert_major_pallas_mgpu(
    h_expert: Float[Array, "Dst E C I"],
    dy_expert: Float[Array, "Dst E C H"],
    w_down: Float[Array, "Dst E I H"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Lane,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C I"], Float[Array, "Dst E I H"]]:
    """Compute expert-major W2 backward using explicit Mosaic GPU WGMMA kernels."""

    dh_expert = source_push_semantic_w2_backward_dh_expert_major_pallas_mgpu(
        dy_expert,
        w_down,
        valid,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        mesh=mesh,
    )
    dw2 = source_push_semantic_w2_backward_dw2_expert_major_pallas_mgpu(
        h_expert,
        dy_expert,
        valid,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
        mesh=mesh,
    )
    return dh_expert, dw2


def source_push_semantic_w2_backward_dh_expert_major_pallas_mgpu(
    dy_expert: Float[Array, "Dst E C H"],
    w_down: Float[Array, "Dst E I H"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Lane,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C I"]:
    """Compute expert-major ``dh = dy @ W2.T`` with explicit Mosaic GPU WGMMA."""

    _validate_semantic_w2_backward_expert_major_dh_shapes(dy_expert, w_down, valid)
    if interpret:
        dh_expert, _dw2 = source_push_semantic_w2_backward_expert_major_reference_jax(
            jnp.zeros((*dy_expert.shape[:3], w_down.shape[2]), dtype=w_down.dtype),
            dy_expert,
            w_down,
            valid,
        )
        return dh_expert
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W2 backward requires a GPU backend")
    block_sizes = (
        SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    )
    original_rows = dy_expert.shape[2]
    original_valid = valid
    dy_expert, valid = _pad_expert_major_rows_for_w2_backward(
        dy_expert,
        valid,
        row_multiple=block_sizes.row_block,
    )
    _validate_semantic_w2_backward_expert_major_dh_pallas_request(dy_expert, w_down, valid, block_sizes)
    dy_for_wgmma = jnp.where(valid[..., None], dy_expert, jnp.zeros((), dtype=dy_expert.dtype)).astype(w_down.dtype)
    if mesh is None:
        dh_expert = _source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel_call(
            dy_for_wgmma,
            w_down,
            row_block=block_sizes.row_block,
            intermediate_block=block_sizes.intermediate_block,
            hidden_block=block_sizes.hidden_block,
            lowering_semantics=lowering_semantics,
        )
    else:
        dh_expert = _source_push_semantic_w2_backward_dh_expert_major_sharded_mgpu_kernel(
            mesh,
            dy_for_wgmma,
            w_down,
            row_block=block_sizes.row_block,
            intermediate_block=block_sizes.intermediate_block,
            hidden_block=block_sizes.hidden_block,
            lowering_semantics=lowering_semantics,
        )
    dh_expert = dh_expert[:, :, :original_rows, :]
    return jnp.where(original_valid[..., None], dh_expert, jnp.zeros((), dtype=dh_expert.dtype))


def source_push_semantic_w2_backward_dw2_expert_major_pallas_mgpu(
    h_expert: Float[Array, "Dst E C I"],
    dy_expert: Float[Array, "Dst E C H"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Lane,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E I H"]:
    """Compute expert-major ``dw2 = H.T @ dy`` with explicit Mosaic GPU WGMMA."""

    _validate_semantic_w2_backward_expert_major_dw2_shapes(h_expert, dy_expert, valid)
    if interpret:
        _dh_expert, dw2 = source_push_semantic_w2_backward_expert_major_reference_jax(
            h_expert,
            dy_expert,
            jnp.zeros((*h_expert.shape[:2], h_expert.shape[3], dy_expert.shape[3]), dtype=dy_expert.dtype),
            valid,
        )
        return dw2
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W2 backward requires a GPU backend")
    block_sizes = (
        SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    )
    h_expert, dy_expert, valid = _pad_expert_major_rows_for_w2_backward(
        h_expert,
        dy_expert,
        valid,
        row_multiple=block_sizes.row_block,
    )
    _validate_semantic_w2_backward_expert_major_dw2_pallas_request(h_expert, dy_expert, valid, block_sizes)
    h_for_wgmma = jnp.where(valid[..., None], h_expert, jnp.zeros((), dtype=h_expert.dtype))
    dy_for_wgmma = jnp.where(valid[..., None], dy_expert, jnp.zeros((), dtype=dy_expert.dtype)).astype(h_expert.dtype)
    if mesh is None:
        return _source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel_call(
            h_for_wgmma,
            dy_for_wgmma,
            row_block=block_sizes.row_block,
            intermediate_block=block_sizes.intermediate_block,
            hidden_block=block_sizes.hidden_block,
            lowering_semantics=lowering_semantics,
        )
    return _source_push_semantic_w2_backward_dw2_expert_major_sharded_mgpu_kernel(
        mesh,
        h_for_wgmma,
        dy_for_wgmma,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        lowering_semantics=lowering_semantics,
    )


def source_push_semantic_w2_backward_pallas_mgpu(
    h_pair: Float[Array, "S Dst R I"],
    dy_route: Float[Array, "S Dst R H"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
) -> tuple[Float[Array, "S Dst R I"], Float[Array, "Dst E I H"]]:
    """Compute semantic W2 backward with Pallas/Mosaic GPU.

    ``Warpgroup`` is the default because this kernel reads local pair-flat
    buffers and local weights only. There are no peer-id GMEM refs in this
    scaffold. ``Lane`` remains caller-selectable for lowering diagnostics.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W2 backward requires a GPU backend")
    block_sizes = SourcePushSemanticW2BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w2_backward_pallas_request(h_pair, dy_route, w_down, plan, block_sizes)
    dh_pair = source_push_semantic_w2_backward_dh_pallas_mgpu(
        dy_route,
        w_down,
        plan,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
    )
    dw2 = source_push_semantic_w2_backward_dw2_pallas_mgpu(
        h_pair,
        dy_route,
        plan,
        w_down_shape=w_down.shape,
        block_sizes=block_sizes,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
    )
    return dh_pair, dw2


def source_push_semantic_w2_backward_dh_pallas_mgpu(
    dy_route: Float[Array, "S Dst R H"],
    w_down: Float[Array, "Dst E I H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
) -> Float[Array, "S Dst R I"]:
    """Compute ``dh_pair = dy_route @ w_down.T`` for semantic pair-flat rows."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W2 dh requires a GPU backend")
    block_sizes = SourcePushSemanticW2BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w2_backward_dh_request(dy_route, w_down, plan, block_sizes)
    return _source_push_semantic_w2_backward_dh_pallas_call(
        dy_route,
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


def source_push_semantic_w2_backward_dw2_pallas_mgpu(
    h_pair: Float[Array, "S Dst R I"],
    dy_route: Float[Array, "S Dst R H"],
    plan: SourcePushSemanticPlan,
    *,
    w_down_shape: tuple[int, int, int, int],
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
) -> Float[Array, "Dst E I H"]:
    """Compute ``dw2`` by reducing all source rows for each destination expert."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W2 dw2 requires a GPU backend")
    block_sizes = SourcePushSemanticW2BackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w2_backward_dw2_request(h_pair, dy_route, plan, w_down_shape, block_sizes)
    return _source_push_semantic_w2_backward_dw2_pallas_call(
        h_pair,
        dy_route,
        plan.xcounts,
        plan.pair_expert_base,
        plan.valid_mask,
        w_down_shape=w_down_shape,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
    )


def _source_push_semantic_w2_backward_dh_pallas_call(
    dy_route: Float[Array, "S Dst R H"],
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
) -> Float[Array, "S Dst R I"]:
    source_count, dst_count, rows_per_pair, hidden_dim = dy_route.shape
    intermediate_dim = w_down.shape[2]
    output_shape = jax.ShapeDtypeStruct((source_count, dst_count, rows_per_pair, intermediate_dim), jnp.float32)
    valid_mask_i32 = valid_mask.astype(jnp.int32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_w2_backward_dh_kernel(
            row_block=row_block,
            intermediate_block=intermediate_block,
            hidden_block=hidden_block,
            hidden_dim=hidden_dim,
            experts_per_rank=w_down.shape[1],
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            source_count,
            dst_count,
            rows_per_pair // row_block,
            intermediate_dim // intermediate_block,
        ),
        interpret=interpret,
        name="source_push_semantic_w2_backward_dh_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
        cost_estimate=_source_push_semantic_w2_backward_dh_cost_estimate(
            dy_route,
            w_down,
            xcounts,
            pair_expert_base,
            valid_mask_i32,
            output_shape,
        ),
    )(dy_route, w_down, xcounts, pair_expert_base, valid_mask_i32)


def _source_push_semantic_w2_backward_dw2_pallas_call(
    h_pair: Float[Array, "S Dst R I"],
    dy_route: Float[Array, "S Dst R H"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    w_down_shape: tuple[int, int, int, int],
    intermediate_block: int,
    hidden_block: int,
    interpret: bool,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E I H"]:
    dst_count, experts_per_rank, intermediate_dim, hidden_dim = w_down_shape
    output_shape = jax.ShapeDtypeStruct(w_down_shape, jnp.float32)
    valid_mask_i32 = valid_mask.astype(jnp.int32)
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return pl.pallas_call(
        _make_source_push_semantic_w2_backward_dw2_kernel(
            intermediate_block=intermediate_block,
            hidden_block=hidden_block,
            rows_per_pair=h_pair.shape[2],
            source_count=h_pair.shape[0],
        ),
        in_specs=(gmem_spec, gmem_spec, gmem_spec, gmem_spec, gmem_spec),
        out_specs=gmem_spec,
        out_shape=output_shape,
        grid=(
            dst_count,
            experts_per_rank,
            intermediate_dim // intermediate_block,
            hidden_dim // hidden_block,
        ),
        interpret=interpret,
        name="source_push_semantic_w2_backward_dw2_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
        cost_estimate=_source_push_semantic_w2_backward_dw2_cost_estimate(
            h_pair,
            dy_route,
            xcounts,
            pair_expert_base,
            valid_mask_i32,
            output_shape,
        ),
    )(h_pair, dy_route, xcounts, pair_expert_base, valid_mask_i32)


def _pad_expert_major_rows_for_w2_backward(*arrays: Array, row_multiple: int) -> tuple[Array, ...]:
    rows = arrays[0].shape[2]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    pad_rows = padded_rows - rows
    if pad_rows == 0:
        return arrays
    return tuple(jnp.pad(array, ((0, 0), (0, 0), (0, pad_rows), (0, 0))[: array.ndim]) for array in arrays)


def _source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel_call(
    dy_expert: Float[Array, "Dst E C H"],
    w_down: Float[Array, "Dst E I H"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E C I"]:
    kernel = _make_source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel(
        dst_count=dy_expert.shape[0],
        experts_per_rank=dy_expert.shape[1],
        rows=dy_expert.shape[2],
        intermediate_dim=w_down.shape[2],
        hidden_dim=dy_expert.shape[3],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        lowering_semantics=lowering_semantics,
    )
    return kernel(dy_expert, w_down)


def _source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel_call(
    h_expert: Float[Array, "Dst E C I"],
    dy_expert: Float[Array, "Dst E C H"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E I H"]:
    kernel = _make_source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel(
        dst_count=h_expert.shape[0],
        experts_per_rank=h_expert.shape[1],
        rows=h_expert.shape[2],
        intermediate_dim=h_expert.shape[3],
        hidden_dim=dy_expert.shape[3],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        lowering_semantics=lowering_semantics,
    )
    return kernel(h_expert, dy_expert)


def _source_push_semantic_w2_backward_dh_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    dy_expert: Float[Array, "Dst E C H"],
    w_down: Float[Array, "Dst E I H"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E C I"]:
    kernel = _make_source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel(
        dst_count=1,
        experts_per_rank=dy_expert.shape[1],
        rows=dy_expert.shape[2],
        intermediate_dim=w_down.shape[2],
        hidden_dim=dy_expert.shape[3],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        lowering_semantics=lowering_semantics,
    )

    def local_fn(
        dy_local: Float[Array, "1 E C H"],
        w_down_local: Float[Array, "1 E I H"],
    ) -> Float[Array, "1 E C I"]:
        return kernel(dy_local, w_down_local)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dy_expert, w_down)


def _source_push_semantic_w2_backward_dw2_expert_major_sharded_mgpu_kernel(
    mesh: Mesh,
    h_expert: Float[Array, "Dst E C I"],
    dy_expert: Float[Array, "Dst E C H"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
) -> Float[Array, "Dst E I H"]:
    kernel = _make_source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel(
        dst_count=1,
        experts_per_rank=h_expert.shape[1],
        rows=h_expert.shape[2],
        intermediate_dim=h_expert.shape[3],
        hidden_dim=dy_expert.shape[3],
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        lowering_semantics=lowering_semantics,
    )

    def local_fn(
        h_local: Float[Array, "1 E C I"],
        dy_local: Float[Array, "1 E C H"],
    ) -> Float[Array, "1 E I H"]:
        return kernel(h_local, dy_local)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(h_expert, dy_expert)


def _make_source_push_semantic_w2_backward_dh_expert_major_mgpu_kernel(
    *,
    dst_count: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
):
    row_tiles = rows // row_block
    intermediate_tiles = intermediate_dim // intermediate_block
    hidden_tiles = hidden_dim // hidden_block

    def body(
        dy_ref: Float[pl.Ref, "Dst E C H"],
        w_down_ref: Float[pl.Ref, "Dst E I H"],
        dh_ref: Float[pl.Ref, "Dst E C I"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        intermediate_tile = pl.program_id(3)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(dy_smem, w_smem, ready_barrier) -> None:
                @pl.loop(0, hidden_tiles)
                def _hidden_loop(hidden_tile) -> None:
                    hidden_start = hidden_tile * hidden_block
                    mgpu.copy_gmem_to_smem(
                        dy_ref.at[
                            dst,
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dy_smem,
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
                    mgpu.wgmma(acc_ref, dy_smem, mgpu.transpose_ref(w_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                dy_smem=_semantic_w2_backward_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                w_smem=_semantic_w2_backward_wgmma_smem((intermediate_block, hidden_block), w_down_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((row_block, intermediate_block)))
        dh_ref[
            dst,
            expert,
            pl.ds(row_start, row_block),
            pl.ds(intermediate_start, intermediate_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((dst_count, experts_per_rank, rows, intermediate_dim), jnp.float32)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(dst_count, experts_per_rank, row_tiles, intermediate_tiles),
        grid_names=("destination", "expert", "row_tile", "intermediate_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def _make_source_push_semantic_w2_backward_dw2_expert_major_mgpu_kernel(
    *,
    dst_count: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    lowering_semantics: mgpu.LoweringSemantics,
):
    row_tiles = rows // row_block
    intermediate_tiles = intermediate_dim // intermediate_block
    hidden_tiles = hidden_dim // hidden_block

    def body(
        h_ref: Float[pl.Ref, "Dst E C I"],
        dy_ref: Float[pl.Ref, "Dst E C H"],
        dw2_ref: Float[pl.Ref, "Dst E I H"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        intermediate_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        intermediate_start = intermediate_tile * intermediate_block
        hidden_start = hidden_tile * hidden_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(h_smem, dy_smem, ready_barrier) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    mgpu.copy_gmem_to_smem(
                        h_ref.at[
                            dst,
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(intermediate_start, intermediate_block),
                        ],
                        h_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        dy_ref.at[
                            dst,
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dy_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(h_smem, (1, 0)), dy_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                h_smem=_semantic_w2_backward_wgmma_smem((row_block, intermediate_block), h_ref.dtype),
                dy_smem=_semantic_w2_backward_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(acc_scope, acc_ref=mgpu.ACC((intermediate_block, hidden_block)))
        dw2_ref[
            dst,
            expert,
            pl.ds(intermediate_start, intermediate_block),
            pl.ds(hidden_start, hidden_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((dst_count, experts_per_rank, intermediate_dim, hidden_dim), jnp.float32)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(dst_count, experts_per_rank, intermediate_tiles, hidden_tiles),
        grid_names=("destination", "expert", "intermediate_tile", "hidden_tile"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )


def _make_source_push_semantic_w2_backward_dh_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    hidden_dim: int,
    experts_per_rank: int,
):
    def kernel(
        dy_route_ref: Float[pl.Ref, "S Dst R H"],
        w_down_ref: Float[pl.Ref, "Dst E I H"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        dh_pair_ref: Float[pl.Ref, "S Dst R I"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        row_tile = pl.program_id(2)
        intermediate_tile = pl.program_id(3)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block

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
            acc = jnp.zeros((intermediate_block,), dtype=jnp.float32)
            for hidden_start in range(0, hidden_dim, hidden_block):
                for hidden_offset in range(hidden_block):
                    hidden = hidden_start + hidden_offset
                    dy_scalar = dy_route_ref[src, dst, row, hidden].astype(jnp.float32)
                    w_vec = w_down_ref[
                        pl.ds(dst, 1),
                        pl.ds(expert, 1),
                        pl.ds(intermediate_start, intermediate_block),
                        pl.ds(hidden, 1),
                    ][0, 0, :, 0].astype(jnp.float32)
                    acc += dy_scalar * w_vec

            dh_pair_ref[
                pl.ds(src, 1),
                pl.ds(dst, 1),
                pl.ds(row, 1),
                pl.ds(intermediate_start, intermediate_block),
            ] = jnp.where(row_is_valid, acc, jnp.zeros_like(acc))[None, None, None, :]

    return kernel


def _make_source_push_semantic_w2_backward_dw2_kernel(
    *,
    intermediate_block: int,
    hidden_block: int,
    rows_per_pair: int,
    source_count: int,
):
    def kernel(
        h_pair_ref: Float[pl.Ref, "S Dst R I"],
        dy_route_ref: Float[pl.Ref, "S Dst R H"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        dw2_ref: Float[pl.Ref, "Dst E I H"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        intermediate_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        intermediate_start = intermediate_tile * intermediate_block
        hidden_start = hidden_tile * hidden_block

        acc = jnp.zeros((intermediate_block, hidden_block), dtype=jnp.float32)
        for src in range(source_count):
            row_base = pair_expert_base_ref[src, dst, expert]
            row_count = xcounts_ref[src, dst, expert]
            for expert_row in range(rows_per_pair):
                row = row_base + expert_row
                safe_row = jnp.minimum(row, rows_per_pair - 1)
                row_is_valid = (expert_row < row_count) & (valid_mask_ref[src, dst, safe_row] != 0)
                for hidden_offset in range(hidden_block):
                    hidden = hidden_start + hidden_offset
                    dy_scalar = dy_route_ref[src, dst, safe_row, hidden].astype(jnp.float32)
                    h_vec = h_pair_ref[
                        pl.ds(src, 1),
                        pl.ds(dst, 1),
                        pl.ds(safe_row, 1),
                        pl.ds(intermediate_start, intermediate_block),
                    ][0, 0, 0, :].astype(jnp.float32)
                    outer_col = h_vec * dy_scalar
                    acc = acc.at[:, hidden_offset].add(jnp.where(row_is_valid, outer_col, jnp.zeros_like(outer_col)))

        dw2_ref[
            pl.ds(dst, 1),
            pl.ds(expert, 1),
            pl.ds(intermediate_start, intermediate_block),
            pl.ds(hidden_start, hidden_block),
        ] = acc[None, None, :, :]

    return kernel


def _source_push_semantic_w2_backward_reference_from_metadata(
    h_pair: Array,
    dy_route: Array,
    w_down: Array,
    xcounts: Array,
    pair_expert_base: Array,
    valid_mask_i32: Array,
) -> tuple[Array, Array]:
    rows = jnp.arange(h_pair.shape[2], dtype=jnp.int32)
    expert_ids = jnp.zeros((*h_pair.shape[:3],), dtype=jnp.int32)
    in_expert_interval = jnp.zeros((*h_pair.shape[:3],), dtype=bool)
    for expert in range(w_down.shape[1]):
        base = pair_expert_base[:, :, expert]
        count = xcounts[:, :, expert]
        matches = (rows[None, None, :] >= base[:, :, None]) & (rows[None, None, :] < (base + count)[:, :, None])
        expert_ids = jnp.where(matches, expert, expert_ids)
        in_expert_interval = in_expert_interval | matches

    valid_mask = (valid_mask_i32 != 0) & in_expert_interval
    dh = jnp.zeros_like(h_pair, dtype=jnp.float32)
    dw2_parts = []
    for expert in range(w_down.shape[1]):
        mask = (expert_ids == expert) & valid_mask
        h_expert = jnp.where(mask[..., None], h_pair.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))
        dy_expert = jnp.where(mask[..., None], dy_route.astype(jnp.float32), jnp.zeros((), dtype=jnp.float32))
        dw2_parts.append(jnp.einsum("sdri,sdrh->dih", h_expert, dy_expert, preferred_element_type=jnp.float32))
        dh_expert = jnp.einsum(
            "sdrh,dih->sdri",
            dy_expert,
            w_down[:, expert].astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        dh = dh + jnp.where(mask[..., None], dh_expert, jnp.zeros((), dtype=dh.dtype))
    return dh, jnp.stack(dw2_parts, axis=1)


def _source_push_semantic_w2_backward_dh_cost_estimate(
    dy_route: Array,
    w_down: Array,
    xcounts: Array,
    pair_expert_base: Array,
    valid_mask_i32: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(dy_route.shape, dy_route.dtype),
        jax.ShapeDtypeStruct(w_down.shape, w_down.dtype),
        jax.ShapeDtypeStruct(xcounts.shape, xcounts.dtype),
        jax.ShapeDtypeStruct(pair_expert_base.shape, pair_expert_base.dtype),
        jax.ShapeDtypeStruct(valid_mask_i32.shape, valid_mask_i32.dtype),
    )

    def reference(dy_route_spec, w_down_spec, xcounts_spec, pair_expert_base_spec, valid_mask_spec):
        h_pair_spec = jnp.zeros((*dy_route_spec.shape[:3], w_down_spec.shape[2]), dtype=dy_route_spec.dtype)
        dh, _dw2 = _source_push_semantic_w2_backward_reference_from_metadata(
            h_pair_spec,
            dy_route_spec,
            w_down_spec,
            xcounts_spec,
            pair_expert_base_spec,
            valid_mask_spec,
        )
        return dh

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_semantic_w2_backward_dw2_cost_estimate(
    h_pair: Array,
    dy_route: Array,
    xcounts: Array,
    pair_expert_base: Array,
    valid_mask_i32: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(h_pair.shape, h_pair.dtype),
        jax.ShapeDtypeStruct(dy_route.shape, dy_route.dtype),
        jax.ShapeDtypeStruct(xcounts.shape, xcounts.dtype),
        jax.ShapeDtypeStruct(pair_expert_base.shape, pair_expert_base.dtype),
        jax.ShapeDtypeStruct(valid_mask_i32.shape, valid_mask_i32.dtype),
    )

    def reference(h_pair_spec, dy_route_spec, xcounts_spec, pair_expert_base_spec, valid_mask_spec):
        fake_w_down = jnp.zeros(output_shape.shape, dtype=dy_route_spec.dtype)
        _dh, dw2 = _source_push_semantic_w2_backward_reference_from_metadata(
            h_pair_spec,
            dy_route_spec,
            fake_w_down,
            xcounts_spec,
            pair_expert_base_spec,
            valid_mask_spec,
        )
        return dw2

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _validate_semantic_w2_backward_expert_major_shapes(
    h_expert: Array,
    dy_expert: Array,
    w_down: Array,
    valid: Array,
) -> None:
    _validate_semantic_w2_backward_expert_major_dw2_shapes(h_expert, dy_expert, valid)
    _validate_semantic_w2_backward_expert_major_dh_shapes(dy_expert, w_down, valid)
    if h_expert.shape[3] != w_down.shape[2]:
        raise ValueError(f"h_expert dim {h_expert.shape[3]} must match w_down intermediate dim {w_down.shape[2]}")


def _validate_semantic_w2_backward_expert_major_dh_shapes(
    dy_expert: Array,
    w_down: Array,
    valid: Array,
) -> None:
    if dy_expert.ndim != 4:
        raise ValueError(f"dy_expert must have shape [destination, expert, row, hidden], got {dy_expert.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must have shape [destination, expert, intermediate, hidden], got {w_down.shape}")
    if valid.shape != dy_expert.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match dy_expert row shape {dy_expert.shape[:3]}")
    if w_down.shape[:2] != dy_expert.shape[:2]:
        raise ValueError(
            f"w_down leading shape {w_down.shape[:2]} must match dy_expert leading shape {dy_expert.shape[:2]}"
        )
    if w_down.shape[3] != dy_expert.shape[3]:
        raise ValueError(f"w_down hidden dim {w_down.shape[3]} must match dy_expert hidden dim {dy_expert.shape[3]}")


def _validate_semantic_w2_backward_expert_major_dw2_shapes(
    h_expert: Array,
    dy_expert: Array,
    valid: Array,
) -> None:
    if h_expert.ndim != 4:
        raise ValueError(f"h_expert must have shape [destination, expert, row, intermediate], got {h_expert.shape}")
    if dy_expert.ndim != 4:
        raise ValueError(f"dy_expert must have shape [destination, expert, row, hidden], got {dy_expert.shape}")
    if valid.shape != h_expert.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match h_expert row shape {h_expert.shape[:3]}")
    if dy_expert.shape[:3] != h_expert.shape[:3]:
        raise ValueError(
            f"dy_expert row shape {dy_expert.shape[:3]} must match h_expert row shape {h_expert.shape[:3]}"
        )


def _validate_semantic_w2_backward_expert_major_dh_pallas_request(
    dy_expert: Array,
    w_down: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes,
) -> None:
    _validate_semantic_w2_backward_expert_major_dh_shapes(dy_expert, w_down, valid)
    _validate_semantic_w2_backward_expert_major_block_sizes(
        rows=dy_expert.shape[2],
        intermediate_dim=w_down.shape[2],
        hidden_dim=dy_expert.shape[3],
        block_sizes=block_sizes,
    )
    _semantic_w2_backward_wgmma_smem((block_sizes.row_block, block_sizes.hidden_block), dy_expert.dtype)
    _semantic_w2_backward_wgmma_smem((block_sizes.intermediate_block, block_sizes.hidden_block), w_down.dtype)


def _validate_semantic_w2_backward_expert_major_dw2_pallas_request(
    h_expert: Array,
    dy_expert: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes,
) -> None:
    _validate_semantic_w2_backward_expert_major_dw2_shapes(h_expert, dy_expert, valid)
    _validate_semantic_w2_backward_expert_major_block_sizes(
        rows=h_expert.shape[2],
        intermediate_dim=h_expert.shape[3],
        hidden_dim=dy_expert.shape[3],
        block_sizes=block_sizes,
    )
    _semantic_w2_backward_wgmma_smem((block_sizes.row_block, block_sizes.intermediate_block), h_expert.dtype)
    _semantic_w2_backward_wgmma_smem((block_sizes.row_block, block_sizes.hidden_block), dy_expert.dtype)


def _validate_semantic_w2_backward_expert_major_block_sizes(
    *,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
    block_sizes: SourcePushSemanticW2BackwardExpertMajorPallasBlockSizes,
) -> None:
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if rows % block_sizes.row_block:
        raise ValueError(f"expert-major row dim {rows} must be divisible by row_block={block_sizes.row_block}")
    if intermediate_dim % block_sizes.intermediate_block:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by "
            f"intermediate_block={block_sizes.intermediate_block}"
        )
    if hidden_dim % block_sizes.hidden_block:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by hidden_block={block_sizes.hidden_block}")


def _semantic_w2_backward_wgmma_smem(shape: tuple[int, int], dtype):
    swizzle_elems = SEMANTIC_W2_BACKWARD_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % SEMANTIC_W2_BACKWARD_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "Semantic W2 backward WGMMA SMEM operands must be divisible by "
            f"tile=({SEMANTIC_W2_BACKWARD_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((SEMANTIC_W2_BACKWARD_WGMMA_TILE_M, swizzle_elems)),
            mgpu.SwizzleTransform(SEMANTIC_W2_BACKWARD_WGMMA_SWIZZLE_BYTES),
        ),
    )


def _validate_semantic_w2_backward_pallas_request(
    h_pair: Array,
    dy_route: Array,
    w_down: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes,
) -> None:
    _validate_semantic_w2_backward_dh_request(dy_route, w_down, plan, block_sizes)
    _validate_semantic_w2_backward_dw2_request(h_pair, dy_route, plan, w_down.shape, block_sizes)
    if h_pair.shape[:3] != dy_route.shape[:3]:
        raise ValueError(f"h_pair route shape {h_pair.shape[:3]} must match dy_route route shape {dy_route.shape[:3]}")
    if h_pair.shape[3] != w_down.shape[2]:
        raise ValueError(f"h_pair intermediate dim {h_pair.shape[3]} must match w_down dim {w_down.shape[2]}")


def _validate_semantic_w2_backward_dh_request(
    dy_route: Array,
    w_down: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes,
) -> None:
    if dy_route.ndim != 4:
        raise ValueError(f"dy_route must have shape [source, destination, row, hidden], got {dy_route.shape}")
    if w_down.ndim != 4:
        raise ValueError(f"w_down must have shape [destination, expert, intermediate, hidden], got {w_down.shape}")
    if plan.valid_mask.shape != dy_route.shape[:3]:
        raise ValueError(
            f"plan valid_mask shape {plan.valid_mask.shape} must match dy_route route shape {dy_route.shape[:3]}"
        )
    if plan.xcounts.shape[:2] != dy_route.shape[:2]:
        raise ValueError(
            f"plan xcounts source/destination shape {plan.xcounts.shape[:2]} must match {dy_route.shape[:2]}"
        )
    if plan.pair_expert_base.shape != plan.xcounts.shape:
        raise ValueError(
            f"plan pair_expert_base shape {plan.pair_expert_base.shape} must match xcounts shape {plan.xcounts.shape}"
        )
    if w_down.shape[0] != dy_route.shape[1]:
        raise ValueError(
            f"w_down destination dim {w_down.shape[0]} must match dy_route destination dim {dy_route.shape[1]}"
        )
    if w_down.shape[1] != plan.xcounts.shape[2]:
        raise ValueError(f"w_down expert dim {w_down.shape[1]} must match plan expert dim {plan.xcounts.shape[2]}")
    if w_down.shape[3] != dy_route.shape[3]:
        raise ValueError(f"w_down hidden dim {w_down.shape[3]} must match dy_route hidden dim {dy_route.shape[3]}")
    _validate_semantic_w2_backward_block_sizes(
        dy_route.shape[2],
        w_down.shape[2],
        dy_route.shape[3],
        block_sizes,
    )


def _validate_semantic_w2_backward_dw2_request(
    h_pair: Array,
    dy_route: Array,
    plan: SourcePushSemanticPlan,
    w_down_shape: tuple[int, int, int, int],
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes,
) -> None:
    if h_pair.ndim != 4:
        raise ValueError(f"h_pair must have shape [source, destination, row, intermediate], got {h_pair.shape}")
    if dy_route.ndim != 4:
        raise ValueError(f"dy_route must have shape [source, destination, row, hidden], got {dy_route.shape}")
    if len(w_down_shape) != 4:
        raise ValueError(f"w_down_shape must be [destination, expert, intermediate, hidden], got {w_down_shape}")
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
    if h_pair.shape[:3] != dy_route.shape[:3]:
        raise ValueError(f"h_pair route shape {h_pair.shape[:3]} must match dy_route route shape {dy_route.shape[:3]}")
    if w_down_shape[0] != h_pair.shape[1]:
        raise ValueError(
            f"w_down destination dim {w_down_shape[0]} must match h_pair destination dim {h_pair.shape[1]}"
        )
    if w_down_shape[1] != plan.xcounts.shape[2]:
        raise ValueError(f"w_down expert dim {w_down_shape[1]} must match plan expert dim {plan.xcounts.shape[2]}")
    if w_down_shape[2] != h_pair.shape[3]:
        raise ValueError(f"w_down intermediate dim {w_down_shape[2]} must match h_pair dim {h_pair.shape[3]}")
    if w_down_shape[3] != dy_route.shape[3]:
        raise ValueError(f"w_down hidden dim {w_down_shape[3]} must match dy_route dim {dy_route.shape[3]}")
    _validate_semantic_w2_backward_block_sizes(
        h_pair.shape[2],
        h_pair.shape[3],
        dy_route.shape[3],
        block_sizes,
    )


def _validate_semantic_w2_backward_block_sizes(
    rows_per_pair: int,
    intermediate_dim: int,
    hidden_dim: int,
    block_sizes: SourcePushSemanticW2BackwardPallasBlockSizes,
) -> None:
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if rows_per_pair % block_sizes.row_block:
        raise ValueError(f"route row capacity {rows_per_pair} must be divisible by row_block={block_sizes.row_block}")
    if intermediate_dim % block_sizes.intermediate_block:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by "
            f"intermediate_block={block_sizes.intermediate_block}"
        )
    if hidden_dim % block_sizes.hidden_block:
        raise ValueError(f"hidden dim {hidden_dim} must be divisible by hidden_block={block_sizes.hidden_block}")


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
