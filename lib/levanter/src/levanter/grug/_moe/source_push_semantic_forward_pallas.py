# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pallas scaffolding for slot-free source-push semantic forward kernels."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_backward_w2 import SOURCE_PUSH_MESH_AXIS
from levanter.grug._moe.source_push_plan import (
    SourcePushSemanticPlan,
    source_push_semantic_pair_expert_ids_jax,
    source_push_semantic_pair_to_expert_major_jax,
    source_push_semantic_w13_reference_jax,
    source_push_semantic_x_to_expert_major_jax,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed


DEFAULT_SEMANTIC_GATHER_X_ROW_BLOCK = 16
DEFAULT_SEMANTIC_GATHER_X_HIDDEN_BLOCK = 512
DEFAULT_SEMANTIC_W13_ROW_BLOCK = 1
DEFAULT_SEMANTIC_W13_HIDDEN_BLOCK = 128
DEFAULT_SEMANTIC_W13_INTERMEDIATE_BLOCK = 128
DEFAULT_SEMANTIC_W13_EXPERT_MAJOR_ROW_BLOCK = 64
DEFAULT_SEMANTIC_W13_EXPERT_MAJOR_HIDDEN_BLOCK = 128
DEFAULT_SEMANTIC_W13_EXPERT_MAJOR_INTERMEDIATE_BLOCK = 128
SEMANTIC_WGMMA_SWIZZLE_BYTES = 128
SEMANTIC_WGMMA_TILE_M = 8
SEMANTIC_MGPU_MIN_INT_TRANSFER_ELEMENTS = 128


@dataclass(frozen=True, slots=True)
class SourcePushSemanticGatherXPallasBlockSizes:
    """Tile sizes for the pair-flat semantic source-token gather kernel."""

    row_block: int = DEFAULT_SEMANTIC_GATHER_X_ROW_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_GATHER_X_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticGatherXPallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW13PallasBlockSizes:
    """Tile sizes for the local pair-flat semantic W13/SwiGLU compute kernel."""

    row_block: int = DEFAULT_SEMANTIC_W13_ROW_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W13_HIDDEN_BLOCK
    intermediate_block: int = DEFAULT_SEMANTIC_W13_INTERMEDIATE_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW13PallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushSemanticW13ExpertMajorPallasBlockSizes:
    """Tile sizes for semantic W13/SwiGLU that writes destination expert-major rows."""

    row_block: int = DEFAULT_SEMANTIC_W13_EXPERT_MAJOR_ROW_BLOCK
    hidden_block: int = DEFAULT_SEMANTIC_W13_EXPERT_MAJOR_HIDDEN_BLOCK
    intermediate_block: int = DEFAULT_SEMANTIC_W13_EXPERT_MAJOR_INTERMEDIATE_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushSemanticW13ExpertMajorPallasBlockSizes":
        return cls()


def source_push_semantic_gather_x_pallas_mgpu(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes | None = None,
    output_dtype: jnp.dtype | None = None,
    interpret: bool = False,
) -> Float[Array, "S Dst R H"]:
    """Gather source tokens into pair-flat semantic route order with Pallas.

    This is the first slot-free semantic forward primitive. It intentionally
    stops at the route-ordered ``x`` buffer so W13/W2 lowering can evolve
    independently from the metadata contract.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push gather requires a GPU backend")
    block_sizes = SourcePushSemanticGatherXPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    output_dtype = x.dtype if output_dtype is None else jnp.dtype(output_dtype)
    _validate_semantic_gather_x_pallas_request(x, plan, block_sizes)
    return _source_push_semantic_gather_x_pallas_call(
        x,
        plan.token_ids,
        plan.valid_mask,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        output_dtype=output_dtype,
        interpret=interpret,
    )


def source_push_semantic_x_to_expert_major_pallas_scaffold_mgpu(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "Dst E C H"], Bool[Array, "Dst E C"]]:
    """Produce expert-major source activations with the semantic forward scaffold.

    This keeps the activation-producer contract separate from W13 compute. The
    non-interpreter path currently uses the existing pair-flat Pallas gather
    followed by the JAX expert-major scatter; it is a correctness scaffold, not
    the final fused transport kernel.
    """

    if rows_per_expert_capacity is None:
        rows_per_expert_capacity = plan.assignment_ids.shape[0] * plan.assignment_ids.shape[-1]
    if interpret:
        return source_push_semantic_x_to_expert_major_jax(
            x,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    x_pair = source_push_semantic_gather_x_pallas_mgpu(
        x,
        plan,
        block_sizes=block_sizes,
        output_dtype=x.dtype,
        interpret=False,
    )
    return source_push_semantic_pair_to_expert_major_jax(
        x_pair,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )


def source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "Dst E C H"], Bool[Array, "Dst E C"]]:
    """Gather source tokens directly into destination expert-major row order.

    ``source_row_base_by_expert`` may place each source's rows at a padded
    expert-local base. When omitted, the compact bases from ``plan`` are used.
    """

    block_sizes = SourcePushSemanticGatherXPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    supplied_source_row_bases = source_row_base_by_expert is not None
    if source_row_base_by_expert is None:
        source_row_base_by_expert = plan.src_base_by_expert
    _validate_semantic_source_row_base_contract(
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    if supplied_source_row_bases:
        source_row_base_by_expert = _validate_semantic_source_row_base_capacity(
            plan,
            source_row_base_by_expert,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    if interpret:
        if supplied_source_row_bases:
            return _source_push_semantic_x_to_expert_major_reference(
                x,
                plan,
                source_row_base_by_expert,
                rows_per_expert_capacity=rows_per_expert_capacity,
            )
        return source_push_semantic_x_to_expert_major_jax(
            x,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic direct x-to-expert-major gather requires a GPU backend")
    _validate_semantic_x_to_expert_major_direct_pallas_request(
        x,
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity,
        block_sizes,
    )
    x_expert = _source_push_semantic_x_to_expert_major_direct_pallas_call(
        x,
        plan.token_ids,
        plan.xcounts,
        plan.pair_expert_base,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )
    valid = _source_push_semantic_valid_from_source_row_bases(
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    return x_expert, valid


def _source_push_semantic_x_to_expert_major_reference(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
) -> tuple[Float[Array, "Dst E C H"], Bool[Array, "Dst E C"]]:
    """Reference gather/scatter for explicit source-padded expert rows."""

    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    source_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    pair_rows = jnp.arange(plan.assignment_ids.shape[2], dtype=jnp.int32)[None, None, :]
    pair_base = plan.pair_expert_base.at[source_index, dst_index, expert_ids].get()
    source_base = source_row_base_by_expert.at[dst_index, source_index, expert_ids].get()
    expert_rows = source_base + pair_rows - pair_base
    route_valid = plan.valid_mask & (expert_rows >= 0) & (expert_rows < rows_per_expert_capacity)
    scatter_rows = jnp.where(route_valid, expert_rows, rows_per_expert_capacity)
    token_ids = jnp.maximum(plan.token_ids, 0)
    source_rows = x.at[source_index, token_ids].get()
    x_expert = jnp.zeros(
        (
            plan.assignment_ids.shape[1],
            plan.xcounts.shape[-1],
            rows_per_expert_capacity,
            x.shape[-1],
        ),
        dtype=x.dtype,
    )
    x_expert = x_expert.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(route_valid[..., None], source_rows, jnp.zeros((), dtype=x.dtype)),
        mode="drop",
    )
    valid = _source_push_semantic_valid_from_source_row_bases(
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    return x_expert, valid


def _source_push_semantic_valid_from_source_row_bases(
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
) -> Bool[Array, "Dst E C"]:
    rows = jnp.arange(rows_per_expert_capacity, dtype=jnp.int32)[None, None, None, :]
    source_bases = jnp.transpose(source_row_base_by_expert, (0, 2, 1))[..., None]
    source_counts = jnp.transpose(plan.xcounts, (1, 2, 0))[..., None]
    return jnp.any((rows >= source_bases) & (rows < source_bases + source_counts), axis=2)


def source_push_semantic_x_to_expert_major_lookup_pallas_mgpu(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "Dst E C H"], Bool[Array, "Dst E C"]]:
    """Gather source tokens into expert-major order via explicit row lookup metadata.

    The direct gather kernel scans every source rank for each expert-major row.
    This variant first builds destination-row lookup metadata inside the JAX
    boundary, then the Pallas copy kernel performs one source-token load per
    live expert row.
    """

    block_sizes = SourcePushSemanticGatherXPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    if interpret:
        return source_push_semantic_x_to_expert_major_jax(
            x,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic lookup x-to-expert-major gather requires a GPU backend")
    _validate_semantic_x_to_expert_major_direct_pallas_request(
        x,
        plan,
        plan.src_base_by_expert,
        rows_per_expert_capacity,
        block_sizes,
    )
    source_lookup, token_lookup, valid = source_push_semantic_expert_major_source_token_lookup_jax(
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    x_expert = _source_push_semantic_x_to_expert_major_lookup_pallas_call(
        x,
        source_lookup,
        token_lookup,
        valid,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
    )
    return x_expert, valid


def source_push_semantic_expert_major_source_token_lookup_jax(
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int,
) -> tuple[Int[Array, "Dst E C"], Int[Array, "Dst E C"], Bool[Array, "Dst E C"]]:
    """Build ``expert-major row -> source token`` lookup metadata."""

    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    expert_ids, expert_rows = _source_push_semantic_expert_row_indices_from_plan(plan)
    valid = plan.valid_mask & (expert_rows < rows_per_expert_capacity)
    src_index = jnp.arange(plan.assignment_ids.shape[0], dtype=jnp.int32)[:, None, None]
    dst_index = jnp.arange(plan.assignment_ids.shape[1], dtype=jnp.int32)[None, :, None]
    source_shape = (plan.assignment_ids.shape[1], plan.xcounts.shape[-1], rows_per_expert_capacity)
    scatter_rows = jnp.where(valid, expert_rows, rows_per_expert_capacity)
    source_lookup = jnp.zeros(source_shape, dtype=jnp.int32)
    token_lookup = jnp.zeros(source_shape, dtype=jnp.int32)
    valid_lookup = jnp.zeros(source_shape, dtype=jnp.bool_)
    source_lookup = source_lookup.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(valid, src_index, 0),
        mode="drop",
    )
    token_lookup = token_lookup.at[dst_index, expert_ids, scatter_rows].set(
        jnp.where(valid, jnp.maximum(plan.token_ids, 0), 0),
        mode="drop",
    )
    valid_lookup = valid_lookup.at[dst_index, expert_ids, scatter_rows].set(valid, mode="drop")
    return source_lookup, token_lookup, valid_lookup


def source_push_semantic_w13_pallas_scaffold_mgpu(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes | None = None,
    w13_block_sizes: SourcePushSemanticW13PallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "S Dst R twoI"], Float[Array, "S Dst R I"]]:
    """Prototype semantic W13 path using Pallas gather and local Pallas compute."""

    x_pair = source_push_semantic_gather_x_pallas_mgpu(
        x,
        plan,
        block_sizes=block_sizes,
        output_dtype=x.dtype,
        interpret=interpret,
    )
    return source_push_semantic_w13_from_x_pair_pallas_mgpu(
        x_pair,
        w_gate_up,
        plan,
        block_sizes=w13_block_sizes,
        interpret=interpret,
    )


def source_push_semantic_w13_from_x_pair_pallas_mgpu(
    x_pair: Float[Array, "S Dst R H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    *,
    block_sizes: SourcePushSemanticW13PallasBlockSizes | None = None,
    interpret: bool = False,
    lowering_semantics: mgpu.LoweringSemantics = mgpu.LoweringSemantics.Warpgroup,
) -> tuple[Float[Array, "S Dst R twoI"], Float[Array, "S Dst R I"]]:
    """Compute semantic W13/SwiGLU over gathered pair-flat rows with Pallas."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push W13 requires a GPU backend")
    block_sizes = SourcePushSemanticW13PallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_semantic_w13_pallas_request(x_pair, w_gate_up, plan, block_sizes)
    expert_ids = source_push_semantic_pair_expert_ids_jax(plan)
    return _source_push_semantic_w13_from_x_pair_pallas_call(
        x_pair,
        w_gate_up,
        expert_ids,
        plan.valid_mask,
        row_block=block_sizes.row_block,
        hidden_block=block_sizes.hidden_block,
        intermediate_block=block_sizes.intermediate_block,
        interpret=interpret,
        lowering_semantics=lowering_semantics,
    )


def source_push_semantic_w13_from_x_expert_pallas_mgpu(
    x_expert: Float[Array, "Dst E C H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    valid: Bool[Array, "Dst E C"],
    *,
    block_sizes: SourcePushSemanticW13ExpertMajorPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C I"]]:
    """Compute W13/SwiGLU from prepacked destination expert-major activations.

    The GPU path assumes ``x_expert`` is already laid out as
    ``[destination, local_expert, expert_row, hidden]``. It therefore uses
    aligned GMEM-to-SMEM copies for both activations and weights before issuing
    explicit WGMMA instructions. ``interpret=True`` is an independent JAX
    reference path and remains usable without Mosaic GPU support.
    """

    block_sizes = (
        SourcePushSemanticW13ExpertMajorPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    )
    _validate_semantic_w13_from_x_expert_request(x_expert, w_gate_up, valid)
    if interpret:
        return _source_push_semantic_w13_from_x_expert_reference(x_expert, w_gate_up, valid)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic W13 from expert-major x requires a GPU backend")
    _validate_semantic_w13_from_x_expert_pallas_request(x_expert, w_gate_up, valid, block_sizes)
    if mesh is None:
        z_expert, h_expert = _source_push_semantic_w13_from_x_expert_pallas_call(
            x_expert,
            w_gate_up,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            intermediate_block=block_sizes.intermediate_block,
        )
    else:
        z_expert, h_expert = _source_push_semantic_w13_from_x_expert_sharded_mgpu_kernel(
            mesh,
            x_expert,
            w_gate_up,
            row_block=block_sizes.row_block,
            hidden_block=block_sizes.hidden_block,
            intermediate_block=block_sizes.intermediate_block,
        )
    valid_f = valid.astype(jnp.float32)
    return z_expert * valid_f[..., None], h_expert * valid_f[..., None]


def _source_push_semantic_gather_x_pallas_call(
    x: Float[Array, "S T H"],
    token_ids: Int[Array, "S Dst R"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
    interpret: bool,
) -> Float[Array, "S Dst R H"]:
    source_count, dst_count, rows_per_pair = token_ids.shape
    hidden_dim = x.shape[-1]
    if rows_per_pair % row_block:
        raise ValueError(
            "semantic gather row capacity must be divisible by row_block for this prototype, "
            f"got rows_per_pair={rows_per_pair}, row_block={row_block}"
        )
    output_shape = jax.ShapeDtypeStruct((*token_ids.shape, hidden_dim), output_dtype)
    kernel = _make_source_push_semantic_gather_x_kernel(
        row_block=row_block,
        hidden_block=hidden_block,
        output_dtype=output_dtype,
    )
    in_specs, out_specs = _source_push_semantic_gather_x_block_specs()
    cost_estimate = _source_push_semantic_gather_x_cost_estimate(x, token_ids, valid_mask, output_shape)
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=output_shape,
        grid=(source_count, dst_count, rows_per_pair // row_block, hidden_dim // hidden_block),
        interpret=interpret,
        name="source_push_semantic_gather_x_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        cost_estimate=cost_estimate,
    )(x, token_ids, valid_mask.astype(jnp.int32))


def _source_push_semantic_gather_x_block_specs() -> tuple[tuple[pl.BlockSpec, ...], pl.BlockSpec]:
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (gmem_spec, gmem_spec, gmem_spec), gmem_spec


def _source_push_semantic_w13_from_x_pair_pallas_call(
    x_pair: Float[Array, "S Dst R H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    expert_ids: Int[Array, "S Dst R"],
    valid_mask: Bool[Array, "S Dst R"],
    *,
    row_block: int,
    hidden_block: int,
    intermediate_block: int,
    interpret: bool,
    lowering_semantics: mgpu.LoweringSemantics,
) -> tuple[Float[Array, "S Dst R twoI"], Float[Array, "S Dst R I"]]:
    source_count, dst_count, rows_per_pair, hidden_dim = x_pair.shape
    intermediate_dim = w_gate_up.shape[-1] // 2
    z_shape = jax.ShapeDtypeStruct((*expert_ids.shape, intermediate_dim * 2), jnp.float32)
    h_shape = jax.ShapeDtypeStruct((*expert_ids.shape, intermediate_dim), jnp.float32)
    in_specs, out_specs = _source_push_semantic_w13_from_x_pair_block_specs()
    cost_estimate = _source_push_semantic_w13_from_x_pair_cost_estimate(
        x_pair,
        w_gate_up,
        expert_ids,
        valid_mask,
        z_shape,
        h_shape,
    )
    return pl.pallas_call(
        _make_source_push_semantic_w13_from_x_pair_kernel(
            row_block=row_block,
            hidden_block=hidden_block,
            intermediate_block=intermediate_block,
            intermediate_dim=intermediate_dim,
        ),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=(z_shape, h_shape),
        grid=(
            source_count,
            dst_count,
            rows_per_pair // row_block,
            intermediate_dim // intermediate_block,
        ),
        interpret=interpret,
        name="source_push_semantic_w13_from_x_pair_pallas_mgpu",
        compiler_params=mgpu.CompilerParams(lowering_semantics=lowering_semantics),
        cost_estimate=cost_estimate,
    )(x_pair, w_gate_up, expert_ids, valid_mask.astype(jnp.int32))


def _source_push_semantic_w13_from_x_expert_pallas_call(
    x_expert: Float[Array, "Dst E C H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    *,
    row_block: int,
    hidden_block: int,
    intermediate_block: int,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C I"]]:
    kernel = _make_source_push_semantic_w13_from_x_expert_kernel(
        hidden_dim=x_expert.shape[-1],
        intermediate_dim=w_gate_up.shape[-1] // 2,
        rows_per_expert_capacity=x_expert.shape[2],
        row_block=row_block,
        hidden_block=hidden_block,
        intermediate_block=intermediate_block,
    )
    return kernel(x_expert, w_gate_up)


def _source_push_semantic_w13_from_x_expert_sharded_mgpu_kernel(
    mesh: Mesh,
    x_expert: Float[Array, "Dst E C H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    *,
    row_block: int,
    hidden_block: int,
    intermediate_block: int,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C I"]]:
    kernel = _make_source_push_semantic_w13_from_x_expert_kernel(
        hidden_dim=x_expert.shape[-1],
        intermediate_dim=w_gate_up.shape[-1] // 2,
        rows_per_expert_capacity=x_expert.shape[2],
        row_block=row_block,
        hidden_block=hidden_block,
        intermediate_block=intermediate_block,
    )

    def local_fn(
        x_local: Float[Array, "1 E C H"],
        w_local: Float[Array, "1 E H twoI"],
    ) -> tuple[Float[Array, "1 E C twoI"], Float[Array, "1 E C I"]]:
        return kernel(x_local, w_local)

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        check_vma=False,
    )(x_expert, w_gate_up)


def source_push_semantic_w13_expert_major_pallas_mgpu(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    *,
    rows_per_expert_capacity: int | None = None,
    source_row_base_by_expert: Int[Array, "Dst S E"] | None = None,
    block_sizes: SourcePushSemanticW13ExpertMajorPallasBlockSizes | None = None,
    pack_block_sizes: SourcePushSemanticGatherXPallasBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C I"], Bool[Array, "Dst E C"]]:
    """Compute semantic W13/SwiGLU and write destination expert-major rows.

    The GPU path first packs source token rows into a GMEM expert-major buffer,
    then feeds legal 64-row tiles from that buffer to explicit WGMMA. Keeping
    the pack as a separate kernel avoids the arbitrary row assembly that Mosaic
    GPU cannot lower into a WGMMA-compatible SMEM operand.

    ``interpret=True`` uses the independent JAX reference plus the semantic
    expert-major scatter because the CPU Pallas interpreter cannot execute
    Mosaic GPU WGMMA operations.
    """

    block_sizes = (
        SourcePushSemanticW13ExpertMajorPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    )
    if rows_per_expert_capacity is None:
        rows_per_expert_capacity = plan.assignment_ids.shape[0] * plan.assignment_ids.shape[-1]
    if interpret:
        if source_row_base_by_expert is not None:
            x_expert, valid = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
                x,
                plan,
                rows_per_expert_capacity=rows_per_expert_capacity,
                source_row_base_by_expert=source_row_base_by_expert,
                block_sizes=pack_block_sizes,
                interpret=True,
            )
            z_expert, h_expert = source_push_semantic_w13_from_x_expert_pallas_mgpu(
                x_expert,
                w_gate_up,
                valid,
                block_sizes=block_sizes,
                interpret=True,
            )
            return z_expert, h_expert, valid
        z_pair, h_pair = source_push_semantic_w13_reference_jax(x, w_gate_up, plan)
        z_expert, valid = source_push_semantic_pair_to_expert_major_jax(
            z_pair,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
        h_expert, _ = source_push_semantic_pair_to_expert_major_jax(
            h_pair,
            plan,
            rows_per_expert_capacity=rows_per_expert_capacity,
        )
        return z_expert, h_expert, valid
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU semantic source-push expert-major W13 requires a GPU backend")
    _validate_semantic_w13_expert_major_pallas_request(x, w_gate_up, plan, rows_per_expert_capacity, block_sizes)
    x_expert, valid = source_push_semantic_x_to_expert_major_direct_pallas_mgpu(
        x,
        plan,
        rows_per_expert_capacity=rows_per_expert_capacity,
        source_row_base_by_expert=source_row_base_by_expert,
        block_sizes=pack_block_sizes,
        interpret=False,
    )
    z_expert, h_expert = source_push_semantic_w13_from_x_expert_pallas_mgpu(
        x_expert,
        w_gate_up,
        valid,
        block_sizes=block_sizes,
        interpret=False,
    )
    return z_expert, h_expert, valid


def _source_push_semantic_x_to_expert_major_direct_pallas_call(
    x: Float[Array, "S T H"],
    token_ids: Int[Array, "S Dst R"],
    xcounts: Int[Array, "S Dst E"],
    pair_expert_base: Int[Array, "S Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_x_to_expert_major_direct_kernel(
        source_count=x.shape[0],
        hidden_dim=x.shape[-1],
        experts_per_rank=xcounts.shape[-1],
        rows_per_expert_capacity=rows_per_expert_capacity,
        row_block=row_block,
        hidden_block=hidden_block,
        output_dtype=x.dtype,
    )
    return kernel(x, token_ids, xcounts, pair_expert_base, src_base_by_expert)


def _source_push_semantic_x_to_expert_major_lookup_pallas_call(
    x: Float[Array, "S T H"],
    source_lookup: Int[Array, "Dst E C"],
    token_lookup: Int[Array, "Dst E C"],
    valid: Bool[Array, "Dst E C"],
    *,
    row_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C H"]:
    kernel = _make_source_push_semantic_x_to_expert_major_lookup_kernel(
        hidden_dim=x.shape[-1],
        row_block=row_block,
        hidden_block=hidden_block,
        output_dtype=x.dtype,
    )
    return kernel(x, source_lookup, token_lookup, valid.astype(jnp.int32))


def _source_push_semantic_w13_from_x_pair_block_specs() -> tuple[tuple[pl.BlockSpec, ...], tuple[pl.BlockSpec, ...]]:
    gmem_spec = pl.BlockSpec(memory_space=mgpu.GMEM)
    return (gmem_spec, gmem_spec, gmem_spec, gmem_spec), (gmem_spec, gmem_spec)


def _make_source_push_semantic_w13_from_x_expert_kernel(
    *,
    hidden_dim: int,
    intermediate_dim: int,
    rows_per_expert_capacity: int,
    row_block: int,
    hidden_block: int,
    intermediate_block: int,
):
    hidden_tiles = hidden_dim // hidden_block

    def body(
        x_expert_ref: Float[pl.Ref, "Dst E C H"],
        w_gate_up_ref: Float[pl.Ref, "Dst E H twoI"],
        z_expert_ref: Float[pl.Ref, "Dst E C twoI"],
        h_expert_ref: Float[pl.Ref, "Dst E C I"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        intermediate_tile = pl.program_id(3)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block

        def acc_scope(gate_acc_ref, up_acc_ref) -> jax.Array:
            def smem_scope(x_smem, gate_smem, up_smem, ready_barrier) -> None:
                @pl.loop(0, hidden_tiles)
                def _hidden_loop(hidden_tile) -> None:
                    hidden_start = hidden_tile * hidden_block
                    mgpu.copy_gmem_to_smem(
                        x_expert_ref.at[
                            dst,
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        x_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w_gate_up_ref.at[
                            dst,
                            expert,
                            pl.ds(hidden_start, hidden_block),
                            pl.ds(intermediate_start, intermediate_block),
                        ],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w_gate_up_ref.at[
                            dst,
                            expert,
                            pl.ds(hidden_start, hidden_block),
                            pl.ds(intermediate_dim + intermediate_start, intermediate_block),
                        ],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(gate_acc_ref, x_smem, gate_smem)
                    mgpu.wgmma(up_acc_ref, x_smem, up_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                x_smem=_semantic_wgmma_smem((row_block, hidden_block), x_expert_ref.dtype),
                gate_smem=_semantic_wgmma_smem((hidden_block, intermediate_block), w_gate_up_ref.dtype),
                up_smem=_semantic_wgmma_smem((hidden_block, intermediate_block), w_gate_up_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=3),
            )
            gate = gate_acc_ref[...]
            up = up_acc_ref[...]
            h_tile = jax.nn.silu(gate) * up
            z_expert_ref[
                dst,
                expert,
                pl.ds(row_start, row_block),
                pl.ds(intermediate_start, intermediate_block),
            ] = gate
            z_expert_ref[
                dst,
                expert,
                pl.ds(row_start, row_block),
                pl.ds(intermediate_dim + intermediate_start, intermediate_block),
            ] = up
            h_expert_ref[
                dst,
                expert,
                pl.ds(row_start, row_block),
                pl.ds(intermediate_start, intermediate_block),
            ] = h_tile
            return jnp.zeros((1,), dtype=jnp.float32)

        pl.run_scoped(
            acc_scope,
            gate_acc_ref=mgpu.ACC((row_block, intermediate_block)),
            up_acc_ref=mgpu.ACC((row_block, intermediate_block)),
        )

    def wrapped(
        x_expert: Float[Array, "Dst E C H"],
        w_gate_up: Float[Array, "Dst E H twoI"],
    ) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C I"]]:
        z_shape = jax.ShapeDtypeStruct(
            (w_gate_up.shape[0], w_gate_up.shape[1], rows_per_expert_capacity, intermediate_dim * 2),
            jnp.float32,
        )
        h_shape = jax.ShapeDtypeStruct(
            (w_gate_up.shape[0], w_gate_up.shape[1], rows_per_expert_capacity, intermediate_dim),
            jnp.float32,
        )
        return mgpu.kernel(
            body,
            out_shape=(z_shape, h_shape),
            grid=(
                w_gate_up.shape[0],
                w_gate_up.shape[1],
                rows_per_expert_capacity // row_block,
                intermediate_dim // intermediate_block,
            ),
            grid_names=("destination", "expert", "row_tile", "intermediate_tile"),
            compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        )(x_expert, w_gate_up)

    return wrapped


def _make_source_push_semantic_w13_from_x_pair_kernel(
    *,
    row_block: int,
    hidden_block: int,
    intermediate_block: int,
    intermediate_dim: int,
):
    def kernel(
        x_pair_ref: Float[pl.Ref, "S Dst R H"],
        w_gate_up_ref: Float[pl.Ref, "Dst E H twoI"],
        expert_ids_ref: Int[pl.Ref, "S Dst R"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        z_pair_ref: Float[pl.Ref, "S Dst R twoI"],
        h_pair_ref: Float[pl.Ref, "S Dst R I"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        row_tile = pl.program_id(2)
        intermediate_tile = pl.program_id(3)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block

        expert = jnp.maximum(expert_ids_ref[src, dst, row_start], 0)
        valid = valid_mask_ref[pl.ds(src, 1), pl.ds(dst, 1), pl.ds(row_start, row_block)] != 0
        gate_acc = jnp.zeros((row_block, intermediate_block), dtype=jnp.float32)
        up_acc = jnp.zeros((row_block, intermediate_block), dtype=jnp.float32)

        for hidden_start in range(0, x_pair_ref.shape[-1], hidden_block):
            x_tile = x_pair_ref[
                pl.ds(src, 1),
                pl.ds(dst, 1),
                pl.ds(row_start, row_block),
                pl.ds(hidden_start, hidden_block),
            ][0, 0, :, :].astype(jnp.float32)
            gate_tile = w_gate_up_ref[
                pl.ds(dst, 1),
                pl.ds(expert, 1),
                pl.ds(hidden_start, hidden_block),
                pl.ds(intermediate_start, intermediate_block),
            ][0, 0, :, :].astype(jnp.float32)
            up_tile = w_gate_up_ref[
                pl.ds(dst, 1),
                pl.ds(expert, 1),
                pl.ds(hidden_start, hidden_block),
                pl.ds(intermediate_dim + intermediate_start, intermediate_block),
            ][0, 0, :, :].astype(jnp.float32)
            gate_acc += pl.dot(x_tile, gate_tile)
            up_acc += pl.dot(x_tile, up_tile)

        valid_f = valid[0, 0, :].astype(jnp.float32)[:, None]
        gate_acc *= valid_f
        up_acc *= valid_f
        h_tile = jax.nn.silu(gate_acc) * up_acc
        z_pair_ref[
            pl.ds(src, 1),
            pl.ds(dst, 1),
            pl.ds(row_start, row_block),
            pl.ds(intermediate_start, intermediate_block),
        ] = gate_acc[None, None, :, :]
        z_pair_ref[
            pl.ds(src, 1),
            pl.ds(dst, 1),
            pl.ds(row_start, row_block),
            pl.ds(intermediate_dim + intermediate_start, intermediate_block),
        ] = up_acc[None, None, :, :]
        h_pair_ref[
            pl.ds(src, 1),
            pl.ds(dst, 1),
            pl.ds(row_start, row_block),
            pl.ds(intermediate_start, intermediate_block),
        ] = h_tile[None, None, :, :]

    return kernel


def _make_source_push_semantic_x_to_expert_major_direct_kernel(
    *,
    source_count: int,
    hidden_dim: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        x_ref: Float[pl.Ref, "S T H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        xcounts_ref: Int[pl.Ref, "S Dst E"],
        pair_expert_base_ref: Int[pl.Ref, "S Dst E"],
        src_base_by_expert_ref: Int[pl.Ref, "Dst S E"],
        x_expert_ref: Float[pl.Ref, "Dst E C H"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        for row_group_start in range(0, row_block, SEMANTIC_WGMMA_TILE_M):
            for row_lane in range(SEMANTIC_WGMMA_TILE_M):
                expert_row = row_start + row_group_start + row_lane
                row_value = jnp.zeros((hidden_block,), dtype=output_dtype)
                for candidate_src in range(source_count):
                    src_base = src_base_by_expert_ref[dst, candidate_src, expert]
                    count = xcounts_ref[candidate_src, dst, expert]
                    matches = (expert_row >= src_base) & (expert_row < src_base + count)
                    local_row = jnp.maximum(expert_row - src_base, 0)
                    pair_base = pair_expert_base_ref[candidate_src, dst, expert]
                    pair_row = jnp.clip(pair_base + local_row, 0, token_ids_ref.shape[-1] - 1)
                    token = jnp.maximum(token_ids_ref[candidate_src, dst, pair_row], 0)
                    src_x_row = x_ref[candidate_src, token, pl.ds(hidden_start, hidden_block)].astype(output_dtype)
                    row_value = jnp.where(matches, src_x_row, row_value)
                x_expert_ref[
                    dst,
                    expert,
                    pl.ds(row_start + row_group_start + row_lane, 1),
                    pl.ds(hidden_start, hidden_block),
                ] = row_value[None, :]

    def wrapped(
        x: Float[Array, "S T H"],
        token_ids: Int[Array, "S Dst R"],
        xcounts: Int[Array, "S Dst E"],
        pair_expert_base: Int[Array, "S Dst E"],
        src_base_by_expert: Int[Array, "Dst S E"],
    ) -> Float[Array, "Dst E C H"]:
        out_shape = jax.ShapeDtypeStruct(
            (token_ids.shape[1], experts_per_rank, rows_per_expert_capacity, hidden_dim),
            output_dtype,
        )
        return mgpu.kernel(
            kernel,
            out_shape=out_shape,
            grid=(
                token_ids.shape[1],
                experts_per_rank,
                rows_per_expert_capacity // row_block,
                hidden_dim // hidden_block,
            ),
            grid_names=("destination", "expert", "row_tile", "hidden_tile"),
            compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        )(x, token_ids, xcounts, pair_expert_base, src_base_by_expert)

    return wrapped


def _make_source_push_semantic_x_to_expert_major_lookup_kernel(
    *,
    hidden_dim: int,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        x_ref: Float[pl.Ref, "S T H"],
        source_lookup_ref: Int[pl.Ref, "Dst E C"],
        token_lookup_ref: Int[pl.Ref, "Dst E C"],
        valid_ref: Int[pl.Ref, "Dst E C"],
        x_expert_ref: Float[pl.Ref, "Dst E C H"],
    ) -> None:
        dst = pl.program_id(0)
        expert = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)
        for row_offset in range(row_block):
            expert_row = row_start + row_offset
            src = jnp.maximum(source_lookup_ref[dst, expert, expert_row], 0)
            token = jnp.maximum(token_lookup_ref[dst, expert, expert_row], 0)
            valid = valid_ref[dst, expert, expert_row] != 0
            x_tile = x_ref[src, token, pl.ds(hidden_start, hidden_block)].astype(output_dtype)
            x_expert_ref[
                dst,
                expert,
                pl.ds(expert_row, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(
                valid, x_tile, zero_tile
            )[None, :]

    def wrapped(
        x: Float[Array, "S T H"],
        source_lookup: Int[Array, "Dst E C"],
        token_lookup: Int[Array, "Dst E C"],
        valid: Int[Array, "Dst E C"],
    ) -> Float[Array, "Dst E C H"]:
        out_shape = jax.ShapeDtypeStruct((*source_lookup.shape, hidden_dim), output_dtype)
        return mgpu.kernel(
            kernel,
            out_shape=out_shape,
            grid=(
                source_lookup.shape[0],
                source_lookup.shape[1],
                source_lookup.shape[2] // row_block,
                hidden_dim // hidden_block,
            ),
            grid_names=("destination", "expert", "row_tile", "hidden_tile"),
            compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
        )(x, source_lookup, token_lookup, valid)

    return wrapped


def _semantic_wgmma_smem(shape: tuple[int, int], dtype):
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=_semantic_wgmma_transforms(shape, dtype),
    )


def _semantic_wgmma_transforms(shape: tuple[int, int], dtype):
    swizzle_elems = SEMANTIC_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % SEMANTIC_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "Semantic W13 WGMMA SMEM operands must be divisible by "
            f"tile=({SEMANTIC_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )
    return (
        mgpu.TilingTransform((SEMANTIC_WGMMA_TILE_M, swizzle_elems)),
        mgpu.SwizzleTransform(SEMANTIC_WGMMA_SWIZZLE_BYTES),
    )


def _make_source_push_semantic_gather_x_kernel(
    *,
    row_block: int,
    hidden_block: int,
    output_dtype: jnp.dtype,
):
    def kernel(
        x_ref: Float[pl.Ref, "S T H"],
        token_ids_ref: Int[pl.Ref, "S Dst R"],
        valid_mask_ref: Int[pl.Ref, "S Dst R"],
        x_pair_ref: Float[pl.Ref, "S Dst R H"],
    ) -> None:
        src = pl.program_id(0)
        dst = pl.program_id(1)
        row_tile = pl.program_id(2)
        hidden_tile = pl.program_id(3)
        row_start = row_tile * row_block
        hidden_start = hidden_tile * hidden_block

        token_tile = token_ids_ref[pl.ds(src, 1), pl.ds(dst, 1), pl.ds(row_start, row_block)]
        valid_tile = valid_mask_ref[pl.ds(src, 1), pl.ds(dst, 1), pl.ds(row_start, row_block)]
        zero_tile = jnp.zeros((hidden_block,), dtype=output_dtype)
        for row_offset in range(row_block):
            valid = valid_tile[0, 0, row_offset] != 0
            safe_token = jnp.maximum(token_tile[0, 0, row_offset], 0)
            x_tile = x_ref[pl.ds(src, 1), pl.ds(safe_token, 1), pl.ds(hidden_start, hidden_block)][0, 0, :]
            x_pair_ref[
                pl.ds(src, 1),
                pl.ds(dst, 1),
                pl.ds(row_start + row_offset, 1),
                pl.ds(hidden_start, hidden_block),
            ] = jnp.where(valid, x_tile.astype(output_dtype), zero_tile)[None, None, None, :]

    return kernel


def _source_push_semantic_gather_x_reference(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
    *,
    output_dtype: jnp.dtype,
) -> Array:
    source_indices = jnp.arange(token_ids.shape[0], dtype=jnp.int32)[:, None, None]
    safe_tokens = jnp.maximum(token_ids, 0)
    gathered = x.at[source_indices, safe_tokens].get().astype(output_dtype)
    return jnp.where(valid_mask[..., None], gathered, jnp.zeros((), dtype=output_dtype))


def _source_push_semantic_gather_x_cost_estimate(
    x: Array,
    token_ids: Array,
    valid_mask: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(token_ids.shape, token_ids.dtype),
        jax.ShapeDtypeStruct(valid_mask.shape, valid_mask.dtype),
    )

    def reference(x_spec, token_ids_spec, valid_mask_spec):
        return _source_push_semantic_gather_x_reference(
            x_spec,
            token_ids_spec,
            valid_mask_spec,
            output_dtype=output_shape.dtype,
        )

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_semantic_w13_from_x_pair_reference(
    x_pair: Array,
    w_gate_up: Array,
    expert_ids: Array,
    valid_mask: Array,
) -> tuple[Array, Array]:
    dst_index = jnp.arange(x_pair.shape[1], dtype=jnp.int32)[None, :, None]
    w_pair = w_gate_up.at[dst_index, expert_ids].get().astype(jnp.float32)
    z_pair = jnp.einsum("sdrh,sdrho->sdro", x_pair.astype(jnp.float32), w_pair, preferred_element_type=jnp.float32)
    z_pair = jnp.where(valid_mask[..., None], z_pair, jnp.zeros((), dtype=z_pair.dtype))
    gate, up = jnp.split(z_pair, 2, axis=-1)
    return z_pair, jax.nn.silu(gate) * up


def _source_push_semantic_w13_from_x_expert_reference(
    x_expert: Array,
    w_gate_up: Array,
    valid: Array,
) -> tuple[Array, Array]:
    z_expert = jnp.einsum(
        "dech,deho->deco",
        x_expert.astype(jnp.float32),
        w_gate_up.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    z_expert = z_expert * valid.astype(jnp.float32)[..., None]
    gate, up = jnp.split(z_expert, 2, axis=-1)
    h_expert = jax.nn.silu(gate) * up
    return z_expert, h_expert


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


def _source_push_semantic_w13_from_x_pair_cost_estimate(
    x_pair: Array,
    w_gate_up: Array,
    expert_ids: Array,
    valid_mask: Array,
    z_shape: jax.ShapeDtypeStruct,
    h_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(x_pair.shape, x_pair.dtype),
        jax.ShapeDtypeStruct(w_gate_up.shape, w_gate_up.dtype),
        jax.ShapeDtypeStruct(expert_ids.shape, expert_ids.dtype),
        jax.ShapeDtypeStruct(valid_mask.shape, valid_mask.dtype),
    )

    def reference(x_pair_spec, w_gate_up_spec, expert_ids_spec, valid_mask_spec):
        return _source_push_semantic_w13_from_x_pair_reference(
            x_pair_spec,
            w_gate_up_spec,
            expert_ids_spec,
            valid_mask_spec,
        )

    body_cost = pl.estimate_cost(reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=(z_shape, h_shape),
    )


def _validate_semantic_gather_x_pallas_request(
    x: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if plan.token_ids.shape != plan.valid_mask.shape:
        raise ValueError(f"token_ids shape {plan.token_ids.shape} must match valid_mask {plan.valid_mask.shape}")
    if plan.token_ids.ndim != 3:
        raise ValueError(f"token_ids must have shape [source, destination, row], got {plan.token_ids.shape}")
    if plan.token_ids.shape[0] != x.shape[0]:
        raise ValueError(f"x source dim {x.shape[0]} must match plan source dim {plan.token_ids.shape[0]}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"x hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")


def _validate_semantic_w13_pallas_request(
    x_pair: Array,
    w_gate_up: Array,
    plan: SourcePushSemanticPlan,
    block_sizes: SourcePushSemanticW13PallasBlockSizes,
) -> None:
    if x_pair.ndim != 4:
        raise ValueError(f"x_pair must have shape [source, destination, row, hidden], got {x_pair.shape}")
    if w_gate_up.ndim != 4:
        raise ValueError(
            f"w_gate_up must have shape [destination, expert, hidden, 2 * intermediate], got {w_gate_up.shape}"
        )
    if plan.valid_mask.shape != x_pair.shape[:3]:
        raise ValueError(
            f"plan valid_mask shape {plan.valid_mask.shape} must match x_pair route shape {x_pair.shape[:3]}"
        )
    if plan.xcounts.shape[:2] != x_pair.shape[:2]:
        raise ValueError(
            f"plan xcounts source/destination shape {plan.xcounts.shape[:2]} must match {x_pair.shape[:2]}"
        )
    if w_gate_up.shape[0] != x_pair.shape[1]:
        raise ValueError(
            f"w_gate_up destination dim {w_gate_up.shape[0]} must match x_pair destination dim {x_pair.shape[1]}"
        )
    if w_gate_up.shape[1] != plan.xcounts.shape[2]:
        raise ValueError(
            f"w_gate_up expert dim {w_gate_up.shape[1]} must match plan expert dim {plan.xcounts.shape[2]}"
        )
    if w_gate_up.shape[2] != x_pair.shape[3]:
        raise ValueError(f"w_gate_up hidden dim {w_gate_up.shape[2]} must match x_pair hidden dim {x_pair.shape[3]}")
    if w_gate_up.shape[3] % 2:
        raise ValueError(f"w_gate_up output dim must be even gate/up pairs, got {w_gate_up.shape[3]}")
    if block_sizes.row_block != 1:
        raise ValueError(
            "semantic W13 Pallas prototype currently requires row_block=1 so arbitrary pair-flat rows may use "
            f"different experts, got row_block={block_sizes.row_block}"
        )
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if x_pair.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"x_pair row capacity {x_pair.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if x_pair.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"x_pair hidden dim {x_pair.shape[3]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )
    intermediate_dim = w_gate_up.shape[3] // 2
    if intermediate_dim % block_sizes.intermediate_block:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by intermediate_block="
            f"{block_sizes.intermediate_block}"
        )


def _validate_semantic_w13_expert_major_pallas_request(
    x: Array,
    w_gate_up: Array,
    plan: SourcePushSemanticPlan,
    rows_per_expert_capacity: int,
    block_sizes: SourcePushSemanticW13ExpertMajorPallasBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if w_gate_up.ndim != 4:
        raise ValueError(
            f"w_gate_up must have shape [destination, expert, hidden, 2 * intermediate], got {w_gate_up.shape}"
        )
    if plan.token_ids.shape != plan.valid_mask.shape:
        raise ValueError(f"token_ids shape {plan.token_ids.shape} must match valid_mask {plan.valid_mask.shape}")
    if plan.xcounts.shape != plan.pair_expert_base.shape:
        raise ValueError(
            f"plan xcounts shape {plan.xcounts.shape} must match pair_expert_base {plan.pair_expert_base.shape}"
        )
    if plan.src_base_by_expert.shape != (plan.token_ids.shape[1], plan.token_ids.shape[0], plan.xcounts.shape[-1]):
        raise ValueError(
            f"plan src_base_by_expert shape {plan.src_base_by_expert.shape} must match "
            f"{(plan.token_ids.shape[1], plan.token_ids.shape[0], plan.xcounts.shape[-1])}"
        )
    if x.shape[0] != plan.token_ids.shape[0]:
        raise ValueError(f"x source dim {x.shape[0]} must match plan source dim {plan.token_ids.shape[0]}")
    if w_gate_up.shape[0] != plan.token_ids.shape[1]:
        raise ValueError(
            f"w_gate_up destination dim {w_gate_up.shape[0]} must match plan destination dim {plan.token_ids.shape[1]}"
        )
    if w_gate_up.shape[1] != plan.xcounts.shape[-1]:
        raise ValueError(
            f"w_gate_up expert dim {w_gate_up.shape[1]} must match plan expert dim {plan.xcounts.shape[-1]}"
        )
    if w_gate_up.shape[2] != x.shape[-1]:
        raise ValueError(f"w_gate_up hidden dim {w_gate_up.shape[2]} must match x hidden dim {x.shape[-1]}")
    if w_gate_up.shape[3] % 2:
        raise ValueError(f"w_gate_up output dim must be even gate/up pairs, got {w_gate_up.shape[3]}")
    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if rows_per_expert_capacity % block_sizes.row_block:
        raise ValueError(
            f"rows_per_expert_capacity {rows_per_expert_capacity} must be divisible by "
            f"row_block={block_sizes.row_block}"
        )
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"x hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")
    intermediate_dim = w_gate_up.shape[-1] // 2
    if intermediate_dim % block_sizes.intermediate_block:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by intermediate_block="
            f"{block_sizes.intermediate_block}"
        )
    _semantic_wgmma_transforms((block_sizes.row_block, block_sizes.hidden_block), x.dtype)
    _semantic_wgmma_transforms((block_sizes.hidden_block, block_sizes.intermediate_block), w_gate_up.dtype)


def _validate_semantic_x_to_expert_major_direct_pallas_request(
    x: Array,
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Array,
    rows_per_expert_capacity: int,
    block_sizes: SourcePushSemanticGatherXPallasBlockSizes,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if plan.token_ids.shape[:2] != plan.xcounts.shape[:2]:
        raise ValueError(f"token_ids shape {plan.token_ids.shape} is incompatible with xcounts {plan.xcounts.shape}")
    if plan.token_ids.shape[:2] != (x.shape[0], plan.xcounts.shape[1]):
        raise ValueError(f"plan source/destination shape {plan.token_ids.shape[:2]} is incompatible with x {x.shape}")
    _validate_semantic_source_row_base_contract(
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    if rows_per_expert_capacity % block_sizes.row_block:
        raise ValueError(
            "rows_per_expert_capacity must be divisible by row_block, "
            f"got rows_per_expert_capacity={rows_per_expert_capacity}, row_block={block_sizes.row_block}"
        )
    if block_sizes.row_block % SEMANTIC_WGMMA_TILE_M:
        raise ValueError(f"row_block must be divisible by {SEMANTIC_WGMMA_TILE_M}, got {block_sizes.row_block}")
    if x.shape[-1] % block_sizes.hidden_block:
        raise ValueError(f"hidden dim {x.shape[-1]} must be divisible by hidden_block={block_sizes.hidden_block}")


def _validate_semantic_source_row_base_contract(
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Array,
    *,
    rows_per_expert_capacity: int,
) -> None:
    expected_shape = (plan.token_ids.shape[1], plan.token_ids.shape[0], plan.xcounts.shape[-1])
    if source_row_base_by_expert.shape != expected_shape:
        raise ValueError(f"source_row_base_by_expert shape {source_row_base_by_expert.shape} must be {expected_shape}")
    if source_row_base_by_expert.dtype != jnp.dtype(jnp.int32):
        raise ValueError(f"source_row_base_by_expert dtype must be int32, got {source_row_base_by_expert.dtype}")
    if rows_per_expert_capacity <= 0:
        raise ValueError(f"rows_per_expert_capacity must be positive, got {rows_per_expert_capacity}")


def _validate_semantic_source_row_base_capacity(
    plan: SourcePushSemanticPlan,
    source_row_base_by_expert: Int[Array, "Dst S E"],
    *,
    rows_per_expert_capacity: int,
) -> Int[Array, "Dst S E"]:
    valid = _source_push_semantic_valid_from_source_row_bases(
        plan,
        source_row_base_by_expert,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    source_counts = jnp.transpose(plan.xcounts, (1, 0, 2))
    interval_ends = source_row_base_by_expert + source_counts
    intervals_out_of_bounds = jnp.any(source_row_base_by_expert < 0) | jnp.any(
        interval_ends > rows_per_expert_capacity
    )
    intervals_overlap = jnp.sum(valid, dtype=jnp.int32) != jnp.sum(source_counts, dtype=jnp.int32)
    return eqx.error_if(
        source_row_base_by_expert,
        intervals_out_of_bounds | intervals_overlap,
        "source_row_base_by_expert must define non-overlapping source intervals within rows_per_expert_capacity",
    )


def _validate_semantic_w13_from_x_expert_pallas_request(
    x_expert: Array,
    w_gate_up: Array,
    valid: Array,
    block_sizes: SourcePushSemanticW13ExpertMajorPallasBlockSizes,
) -> None:
    _validate_semantic_w13_from_x_expert_request(x_expert, w_gate_up, valid)
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if block_sizes.row_block % 64:
        raise ValueError(f"row_block must be a multiple of 64 for WGMMA, got {block_sizes.row_block}")
    if x_expert.shape[2] % block_sizes.row_block:
        raise ValueError(
            f"x_expert capacity {x_expert.shape[2]} must be divisible by row_block={block_sizes.row_block}"
        )
    if x_expert.shape[3] % block_sizes.hidden_block:
        raise ValueError(
            f"x_expert hidden dim {x_expert.shape[3]} must be divisible by hidden_block={block_sizes.hidden_block}"
        )
    intermediate_dim = w_gate_up.shape[-1] // 2
    if intermediate_dim % block_sizes.intermediate_block:
        raise ValueError(
            f"intermediate dim {intermediate_dim} must be divisible by intermediate_block="
            f"{block_sizes.intermediate_block}"
        )
    _semantic_wgmma_transforms((block_sizes.row_block, block_sizes.hidden_block), x_expert.dtype)
    _semantic_wgmma_transforms((block_sizes.hidden_block, block_sizes.intermediate_block), w_gate_up.dtype)


def _validate_semantic_w13_from_x_expert_request(
    x_expert: Array,
    w_gate_up: Array,
    valid: Array,
) -> None:
    if x_expert.ndim != 4:
        raise ValueError(f"x_expert must have shape [destination, expert, capacity, hidden], got {x_expert.shape}")
    if w_gate_up.ndim != 4:
        raise ValueError(
            f"w_gate_up must have shape [destination, expert, hidden, 2 * intermediate], got {w_gate_up.shape}"
        )
    if valid.shape != x_expert.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match x_expert leading shape {x_expert.shape[:3]}")
    if w_gate_up.shape[:2] != x_expert.shape[:2]:
        raise ValueError(
            f"w_gate_up destination/expert shape {w_gate_up.shape[:2]} must match x_expert {x_expert.shape[:2]}"
        )
    if w_gate_up.shape[2] != x_expert.shape[3]:
        raise ValueError(
            f"w_gate_up hidden dim {w_gate_up.shape[2]} must match x_expert hidden dim {x_expert.shape[3]}"
        )
    if w_gate_up.shape[3] % 2:
        raise ValueError(f"w_gate_up output dim must be even gate/up pairs, got {w_gate_up.shape[3]}")
