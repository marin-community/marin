# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""SonicMoE-inspired local aggregation helpers for Grug MoE.

This module intentionally starts with the final combine operation. It keeps the
current Grug grouped-GEMM boundary intact and replaces the scatter-add combine
with a gather-and-sum aggregation over the expert-sorted outputs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float, Int

try:
    from jax.experimental.pallas import mosaic_gpu as plgpu
except (ImportError, ModuleNotFoundError):
    plgpu = None  # type: ignore[assignment]

try:
    from jax.experimental.pallas import triton as pltriton
except (ImportError, ModuleNotFoundError):
    pltriton = None  # type: ignore[assignment]


SonicGatherSumImplementation: TypeAlias = Literal["xla", "pallas_triton", "pallas_triton_faithful", "pallas_mgpu"]
SonicGatherRaggedDotImplementation: TypeAlias = Literal["xla", "pallas_triton"]
SonicMetadataImplementation: TypeAlias = Literal["xla", "pallas_triton"]


class PallasUnsupportedError(NotImplementedError):
    """Raised when the Pallas GPU gather-sum backend cannot be used."""


@dataclasses.dataclass(frozen=True, slots=True)
class SonicGatherSumBlockSizes:
    """Pallas tile sizes for local SonicMoE gather-and-sum aggregation."""

    token_block_size: int = 16
    hidden_block_size: int = 64
    k_block_size: int = 4
    kernel_repeat: int = 1
    num_warps: int = 4
    use_inline_fma: bool = False

    @classmethod
    def get_default(cls) -> "SonicGatherSumBlockSizes":
        return cls()


@dataclasses.dataclass(frozen=True, slots=True)
class SonicGatherRaggedDotBlockSizes:
    """Pallas tile sizes for gather-fused varlen-M grouped matmul."""

    row_block_size: int = 128
    contraction_block_size: int = 32
    num_warps: int = 8

    @classmethod
    def get_default(cls) -> "SonicGatherRaggedDotBlockSizes":
        return cls()


@dataclasses.dataclass(frozen=True, slots=True)
class SonicMetadataBlockSizes:
    """Pallas tile sizes for SonicMoE-style top-k routing metadata."""

    assignments_per_tile: int = 1024

    @classmethod
    def get_default(cls) -> "SonicMetadataBlockSizes":
        return cls()


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _validate_shapes(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> None:
    if dispatch_output.ndim != 2:
        raise ValueError(f"dispatch_output must be rank-2 [TK, D], got shape={dispatch_output.shape}")
    if dispatch_positions.ndim != 2:
        raise ValueError(f"dispatch_positions must be rank-2 [T, K], got shape={dispatch_positions.shape}")
    if combine_weights.shape != dispatch_positions.shape:
        raise ValueError(
            "combine_weights and dispatch_positions must have identical [T, K] shapes; "
            f"got {combine_weights.shape} vs {dispatch_positions.shape}"
        )


def _validate_block_sizes(block_sizes: SonicGatherSumBlockSizes) -> None:
    if block_sizes.token_block_size <= 0:
        raise ValueError(f"token_block_size must be positive, got {block_sizes.token_block_size}")
    if block_sizes.hidden_block_size <= 0:
        raise ValueError(f"hidden_block_size must be positive, got {block_sizes.hidden_block_size}")
    if block_sizes.k_block_size <= 0:
        raise ValueError(f"k_block_size must be positive, got {block_sizes.k_block_size}")
    if block_sizes.kernel_repeat <= 0:
        raise ValueError(f"kernel_repeat must be positive, got {block_sizes.kernel_repeat}")
    if block_sizes.num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {block_sizes.num_warps}")


def _validate_gather_ragged_dot_shapes(
    x: Float[Array, "T D"],
    token_ids_sort: Int[Array, "TK"],
    rhs: Float[Array, "E D N"],
    group_sizes: Int[Array, "E"],
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if token_ids_sort.ndim != 1:
        raise ValueError(f"token_ids_sort must be rank-1 [TK], got shape={token_ids_sort.shape}")
    if rhs.ndim != 3:
        raise ValueError(f"rhs must be rank-3 [E, D, N], got shape={rhs.shape}")
    if rhs.shape[1] != x.shape[1]:
        raise ValueError(f"rhs input dimension {rhs.shape[1]} must match x hidden dimension {x.shape[1]}")
    if group_sizes.shape != (rhs.shape[0],):
        raise ValueError(f"group_sizes must have shape {(rhs.shape[0],)}, got {group_sizes.shape}")


def _validate_gather_ragged_dot_block_sizes(block_sizes: SonicGatherRaggedDotBlockSizes) -> None:
    if block_sizes.row_block_size <= 0:
        raise ValueError(f"row_block_size must be positive, got {block_sizes.row_block_size}")
    if block_sizes.contraction_block_size <= 0:
        raise ValueError(f"contraction_block_size must be positive, got {block_sizes.contraction_block_size}")
    if block_sizes.num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {block_sizes.num_warps}")


def _validate_metadata_shapes(selected_experts: Int[Array, "T K"], num_experts: int) -> None:
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")


def _validate_metadata_block_sizes(block_sizes: SonicMetadataBlockSizes) -> None:
    if block_sizes.assignments_per_tile <= 0:
        raise ValueError(f"assignments_per_tile must be positive, got {block_sizes.assignments_per_tile}")


_GATHER_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)

_GATHER_RAGGED_DOT_DLHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (2,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)

_GATHER_RAGGED_DOT_DRHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=[],
)


def _ragged_dot_xla(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dim_nums: jax.lax.RaggedDotDimensionNumbers = _GATHER_RAGGED_DOT_DIM_NUMS,
) -> jax.Array:
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dim_nums,
    )


def sonic_gather_ragged_dot_reference(
    x: Float[Array, "T D"],
    token_ids_sort: Int[Array, "TK"],
    rhs: Float[Array, "E D N"],
    group_sizes: Int[Array, "E"],
) -> Float[Array, "TK N"]:
    """Reference first-GMM path that gathers rows at the ragged-dot boundary."""
    _validate_gather_ragged_dot_shapes(x, token_ids_sort, rhs, group_sizes)
    return _ragged_dot_xla(jnp.take(x, token_ids_sort, axis=0), rhs, group_sizes)


def _sonic_gather_ragged_dot_backward(
    x: jax.Array,
    token_ids_sort: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dout: jax.Array,
) -> tuple[jax.Array, None, jax.Array, None]:
    x_gathered = jnp.take(x, token_ids_sort, axis=0)
    dx_gathered = _ragged_dot_xla(dout, rhs, group_sizes, _GATHER_RAGGED_DOT_DLHS_DIM_NUMS)
    dx = jnp.zeros_like(x).at[token_ids_sort].add(dx_gathered, mode="drop")
    drhs = _ragged_dot_xla(x_gathered, dout, group_sizes, _GATHER_RAGGED_DOT_DRHS_DIM_NUMS)
    return dx, None, drhs, None


def sonic_gather_sum_reference(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T D"]:
    """Reference gather-and-sum combine for expert-sorted MoE outputs.

    Args:
        dispatch_output: Expert-sorted output rows from the down-projection.
        dispatch_positions: For each original token/top-k slot, the matching
            row in ``dispatch_output``.
        combine_weights: Original router combine weights in token/top-k order.

    Returns:
        Token-major MoE output ``[T, D]``.
    """
    _validate_shapes(dispatch_output, dispatch_positions, combine_weights)
    acc = jnp.zeros((dispatch_positions.shape[0], dispatch_output.shape[1]), dtype=dispatch_output.dtype)
    weights = combine_weights.astype(dispatch_output.dtype)
    for topk_index in range(dispatch_positions.shape[1]):
        gathered = jnp.take(dispatch_output, dispatch_positions[:, topk_index], axis=0)
        acc = (acc + (gathered * weights[:, topk_index, None]).astype(dispatch_output.dtype)).astype(
            dispatch_output.dtype
        )
    return acc


def sonic_topk_metadata_reference(
    selected_experts: Int[Array, "T K"],
    *,
    num_experts: int,
) -> tuple[Int[Array, "TK"], Int[Array, "TK"], Int[Array, "TK"], Int[Array, "E"], Int[Array, "E1"]]:
    """Reference SonicMoE top-k routing metadata.

    Returns ``x_gather_idx``, ``s_scatter_idx``, ``s_reverse_scatter_idx``,
    expert counts, and exclusive expert offsets. These match the fixed-top-k
    SonicMoE metadata contract: expert-sorted rows point back to source token
    ids, ``s_scatter_idx`` maps expert-sorted rows to flattened top-k entries,
    and ``s_reverse_scatter_idx`` maps flattened top-k entries back to
    expert-sorted row positions.
    """
    _validate_metadata_shapes(selected_experts, num_experts)
    tokens, topk = selected_experts.shape
    assignments = tokens * topk
    expert_ids = selected_experts.reshape(assignments)
    sort_idx = jnp.argsort(expert_ids, stable=True)
    x_gather_idx = (sort_idx // topk).astype(jnp.int32)
    s_scatter_idx = sort_idx.astype(jnp.int32)
    s_reverse_scatter_idx = (
        jnp.zeros((assignments,), dtype=jnp.int32).at[sort_idx].set(jnp.arange(assignments, dtype=jnp.int32))
    )
    expert_counts = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    expert_offsets = jnp.cumulative_sum(expert_counts, include_initial=True).astype(jnp.int32)
    return x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, expert_counts, expert_offsets


def _sonic_gather_sum_backward(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    dout: jax.Array,
) -> tuple[jax.Array, None, jax.Array]:
    gathered = jnp.take(dispatch_output, dispatch_positions, axis=0)
    d_dispatch = (
        jnp.zeros_like(dispatch_output)
        .at[dispatch_positions.reshape(-1)]
        .add(
            (dout[:, None, :] * combine_weights.astype(dout.dtype)[:, :, None]).reshape(-1, dout.shape[-1]),
            mode="drop",
        )
    )
    d_weights = jnp.einsum(
        "tkd,td->tk",
        gathered.astype(dout.dtype),
        dout,
        preferred_element_type=jnp.float32,
    ).astype(combine_weights.dtype)
    return d_dispatch, None, d_weights


def _metadata_counts_pallas_triton_kernel(
    selected_experts_ref,
    tile_counts_ref,
    *,
    assignments_per_tile: int,
    topk: int,
    assignments: int,
    num_experts: int,
    expert_block_size: int,
) -> None:
    tile = pl.program_id(0)
    assignment_offsets = tile * assignments_per_tile + jnp.arange(assignments_per_tile)
    safe_assignment_offsets = jnp.minimum(assignment_offsets, assignments - 1)
    token_offsets = safe_assignment_offsets // topk
    topk_offsets = safe_assignment_offsets - token_offsets * topk
    valid = assignment_offsets < assignments
    expert_ids = pltriton.load(
        selected_experts_ref.at[token_offsets, topk_offsets],
        mask=valid,
        other=num_experts,
    )

    expert_offsets = jnp.arange(expert_block_size)
    matches = (expert_ids[:, None] == expert_offsets[None, :]) & valid[:, None]
    counts = jnp.sum(matches.astype(jnp.int32), axis=0)
    pltriton.store(tile_counts_ref.at[tile, expert_offsets], counts)


def _metadata_scatter_pallas_triton_kernel(
    selected_experts_ref,
    tile_prefix_ref,
    expert_offsets_ref,
    s_reverse_scatter_idx_ref,
    *,
    assignments_per_tile: int,
    topk: int,
    assignments: int,
    num_experts: int,
) -> None:
    tile = pl.program_id(0)
    expert = pl.program_id(1)
    assignment_offsets = tile * assignments_per_tile + jnp.arange(assignments_per_tile)
    safe_assignment_offsets = jnp.minimum(assignment_offsets, assignments - 1)
    token_offsets = safe_assignment_offsets // topk
    topk_offsets = safe_assignment_offsets - token_offsets * topk
    valid = assignment_offsets < assignments
    expert_ids = pltriton.load(
        selected_experts_ref.at[token_offsets, topk_offsets],
        mask=valid,
        other=num_experts,
    )
    matches = valid & (expert_ids == expert)
    local_ranks = jnp.cumsum(matches.astype(jnp.int32), axis=0) - 1
    tile_prefix = pltriton.load(tile_prefix_ref.at[tile, expert])
    expert_offset = pltriton.load(expert_offsets_ref.at[expert])
    output_offsets = expert_offset + tile_prefix + local_ranks

    pltriton.store(
        s_reverse_scatter_idx_ref.at[assignment_offsets],
        output_offsets.astype(jnp.int32),
        mask=matches,
    )


def _metadata_cost_estimate(
    selected_experts: jax.Array,
    *,
    num_experts: int,
    num_tiles: int,
) -> pl.CostEstimate:
    assignments = selected_experts.size
    bytes_accessed = (
        assignments * jnp.dtype(selected_experts.dtype).itemsize
        + assignments * 3 * jnp.dtype(jnp.int32).itemsize
        + (num_experts + num_experts + 1 + num_tiles * num_experts) * jnp.dtype(jnp.int32).itemsize
    )
    return pl.CostEstimate(flops=assignments * num_experts, transcendentals=0, bytes_accessed=bytes_accessed)


def _sonic_topk_metadata_pallas_triton_call(
    selected_experts: jax.Array,
    *,
    num_experts: int,
    block_sizes: SonicMetadataBlockSizes,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if pltriton is None:
        raise PallasUnsupportedError("Pallas Triton backend is not available.")
    _validate_metadata_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError("Pallas Triton SonicMoE metadata requires GPU backend unless interpret=True.")

    tokens, topk = selected_experts.shape
    assignments = tokens * topk
    num_tiles = pl.cdiv(assignments, block_sizes.assignments_per_tile)
    expert_block_size = int(pl.next_power_of_2(num_experts))

    tile_counts_padded = pl.pallas_call(
        lambda selected_experts_ref, tile_counts_ref: _metadata_counts_pallas_triton_kernel(
            selected_experts_ref,
            tile_counts_ref,
            assignments_per_tile=block_sizes.assignments_per_tile,
            topk=topk,
            assignments=assignments,
            num_experts=num_experts,
            expert_block_size=expert_block_size,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tiles, expert_block_size), jnp.int32),
        grid=(num_tiles,),
        compiler_params=pltriton.CompilerParams(num_warps=8, num_stages=4),
        cost_estimate=_metadata_cost_estimate(selected_experts, num_experts=num_experts, num_tiles=num_tiles),
        interpret=interpret,
        name="sonic_topk_metadata_counts_pallas_triton",
    )(selected_experts)
    tile_counts = tile_counts_padded[:, :num_experts]
    tile_prefix = jnp.cumulative_sum(tile_counts, axis=0, include_initial=False) - tile_counts
    expert_counts = jnp.sum(tile_counts, axis=0, dtype=jnp.int32)
    expert_offsets = jnp.cumulative_sum(expert_counts, include_initial=True).astype(jnp.int32)

    s_reverse_scatter_idx = pl.pallas_call(
        lambda selected_experts_ref, tile_prefix_ref, expert_offsets_ref, reverse_ref: (
            _metadata_scatter_pallas_triton_kernel(
                selected_experts_ref,
                tile_prefix_ref,
                expert_offsets_ref,
                reverse_ref,
                assignments_per_tile=block_sizes.assignments_per_tile,
                topk=topk,
                assignments=assignments,
                num_experts=num_experts,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((assignments,), jnp.int32),
        grid=(num_tiles, num_experts),
        compiler_params=pltriton.CompilerParams(num_warps=8, num_stages=4),
        cost_estimate=_metadata_cost_estimate(selected_experts, num_experts=num_experts, num_tiles=num_tiles),
        interpret=interpret,
        name="sonic_topk_metadata_scatter_pallas_triton",
    )(selected_experts, tile_prefix, expert_offsets[:-1])
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    s_scatter_idx = jnp.zeros((assignments,), dtype=jnp.int32).at[s_reverse_scatter_idx].set(assignment_ids)
    x_gather_idx = (s_scatter_idx // topk).astype(jnp.int32)
    return x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, expert_counts, expert_offsets


def _pad_for_pallas(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    block_sizes: SonicGatherSumBlockSizes,
) -> tuple[jax.Array, jax.Array, jax.Array, int, int]:
    tokens, _ = dispatch_positions.shape
    hidden = dispatch_output.shape[1]
    padded_tokens = _ceil_to_multiple(tokens, block_sizes.token_block_size)
    padded_hidden = _ceil_to_multiple(hidden, block_sizes.hidden_block_size)

    if padded_hidden != hidden:
        dispatch_output = jnp.pad(dispatch_output, ((0, 0), (0, padded_hidden - hidden)))
    if padded_tokens != tokens:
        token_pad = padded_tokens - tokens
        dispatch_positions = jnp.pad(dispatch_positions, ((0, token_pad), (0, 0)))
        combine_weights = jnp.pad(combine_weights, ((0, token_pad), (0, 0)))

    return dispatch_output, dispatch_positions, combine_weights, tokens, hidden


def _gather_sum_pallas_kernel(
    dispatch_output_ref,
    dispatch_positions_ref,
    combine_weights_ref,
    repeat_offsets_ref,
    output_ref,
    *,
    topk: int,
    token_block_size: int,
    hidden_block_size: int,
    kernel_repeat: int,
) -> None:
    token_block = pl.program_id(0)
    hidden_block = pl.program_id(1)
    token_offsets = token_block * token_block_size + jnp.arange(token_block_size)
    hidden_offsets = hidden_block * hidden_block_size + jnp.arange(hidden_block_size)

    acc = jnp.zeros((token_block_size, hidden_block_size), dtype=output_ref.dtype)
    for repeat_index in range(kernel_repeat):
        repeat_offset = repeat_offsets_ref[repeat_index]
        for topk_index in range(topk):
            dispatch_rows = dispatch_positions_ref[token_offsets, topk_index] + repeat_offset
            weights = combine_weights_ref[token_offsets, topk_index].astype(output_ref.dtype)
            values = dispatch_output_ref[dispatch_rows[:, None], hidden_offsets[None, :]].astype(output_ref.dtype)
            acc = (acc + (values * weights[:, None]).astype(output_ref.dtype)).astype(output_ref.dtype)

    if kernel_repeat == 1:
        output = acc
    else:
        output = (acc.astype(jnp.float32) * (1.0 / kernel_repeat)).astype(output_ref.dtype)
    output_ref[token_offsets[:, None], hidden_offsets[None, :]] = output


def _gather_sum_pallas_triton_kernel(
    dispatch_output_ref,
    dispatch_positions_ref,
    combine_weights_ref,
    repeat_offsets_ref,
    output_ref,
    *,
    topk: int,
    token_block_size: int,
    hidden_block_size: int,
    kernel_repeat: int,
) -> None:
    token_block = pl.program_id(0)
    hidden_block = pl.program_id(1)
    token_offsets = token_block * token_block_size + jnp.arange(token_block_size)
    hidden_offsets = hidden_block * hidden_block_size + jnp.arange(hidden_block_size)

    acc = jnp.zeros((token_block_size, hidden_block_size), dtype=output_ref.dtype)
    for repeat_index in range(kernel_repeat):
        repeat_offset = pltriton.load(repeat_offsets_ref.at[repeat_index])
        for topk_index in range(topk):
            dispatch_rows = pltriton.load(dispatch_positions_ref.at[token_offsets, topk_index]) + repeat_offset
            weights = pltriton.load(combine_weights_ref.at[token_offsets, topk_index]).astype(output_ref.dtype)
            values = pltriton.load(dispatch_output_ref.at[dispatch_rows[:, None], hidden_offsets[None, :]]).astype(
                output_ref.dtype
            )
            acc = (acc + (values * weights[:, None]).astype(output_ref.dtype)).astype(output_ref.dtype)

    if kernel_repeat == 1:
        output = acc
    else:
        output = (acc.astype(jnp.float32) * (1.0 / kernel_repeat)).astype(output_ref.dtype)
    pltriton.store(output_ref.at[token_offsets[:, None], hidden_offsets[None, :]], output)


def _gather_sum_pallas_triton_token_loop_kernel(
    dispatch_output_ref,
    dispatch_positions_ref,
    combine_weights_ref,
    repeat_offsets_ref,
    output_ref,
    *,
    topk: int,
    hidden_block_size: int,
    hidden_tiles: int,
    kernel_repeat: int,
) -> None:
    token_index = pl.program_id(0)
    topk_offsets = jnp.arange(topk)

    for hidden_tile in range(hidden_tiles):
        hidden_offsets = hidden_tile * hidden_block_size + jnp.arange(hidden_block_size)
        acc = jnp.zeros((hidden_block_size,), dtype=output_ref.dtype)
        for repeat_index in range(kernel_repeat):
            repeat_offset = pltriton.load(repeat_offsets_ref.at[repeat_index])
            dispatch_rows = pltriton.load(dispatch_positions_ref.at[token_index, topk_offsets]) + repeat_offset
            weights = pltriton.load(combine_weights_ref.at[token_index, topk_offsets]).astype(output_ref.dtype)
            values = pltriton.load(dispatch_output_ref.at[dispatch_rows[:, None], hidden_offsets[None, :]]).astype(
                output_ref.dtype
            )
            acc = (acc + jnp.sum((values * weights[:, None]).astype(output_ref.dtype), axis=0)).astype(
                output_ref.dtype
            )

        if kernel_repeat == 1:
            output = acc
        else:
            output = (acc.astype(jnp.float32) * (1.0 / kernel_repeat)).astype(output_ref.dtype)
        pltriton.store(output_ref.at[token_index, hidden_offsets], output)


def _pallas_triton_fma_f32(
    left: jax.Array,
    right: jax.Array,
    addend: jax.Array,
) -> jax.Array:
    if pltriton is None:
        return left * right + addend
    [out] = pltriton.elementwise_inline_asm(
        "fma.rn.f32 $0, $1, $2, $3;",
        args=[left, right, addend],
        constraints="=f,f,f,f",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(left.shape, jnp.float32)],
    )
    return out


def _weighted_accumulate_fma(
    acc: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    *,
    k_block_size: int,
    hidden_block_size: int,
) -> jax.Array:
    del k_block_size, hidden_block_size
    zero_values = jnp.zeros_like(values)
    weight_values = weights[:, None] + zero_values
    weighted_values = _pallas_triton_fma_f32(values, weight_values, zero_values)
    return acc + jnp.sum(weighted_values, axis=0)


def _gather_sum_pallas_triton_faithful_kernel(
    dispatch_output_ref,
    dispatch_positions_ref,
    combine_weights_ref,
    repeat_offsets_ref,
    output_ref,
    *,
    topk: int,
    hidden: int,
    hidden_block_size: int,
    hidden_tiles: int,
    k_block_size: int,
    k_tiles: int,
    kernel_repeat: int,
    has_weights: bool,
    use_inline_fma: bool,
) -> None:
    token_index = pl.program_id(0)
    metadata_base = token_index * topk

    for hidden_tile in range(hidden_tiles):
        hidden_offsets = hidden_tile * hidden_block_size + jnp.arange(hidden_block_size)
        hidden_mask = hidden_offsets < hidden
        acc = jnp.zeros((hidden_block_size,), dtype=jnp.float32)

        for repeat_index in range(kernel_repeat):
            repeat_offset = pltriton.load(repeat_offsets_ref.at[repeat_index])
            for k_tile in range(k_tiles):
                k_offsets = k_tile * k_block_size + jnp.arange(k_block_size)
                if topk % k_block_size == 0:
                    metadata_offsets = metadata_base + k_offsets
                    dispatch_rows = pltriton.load(dispatch_positions_ref.at[metadata_offsets]) + repeat_offset
                    values = pltriton.load(
                        dispatch_output_ref.at[dispatch_rows[:, None], hidden_offsets[None, :]],
                        mask=hidden_mask[None, :],
                        other=0.0,
                    ).astype(jnp.float32)
                    if has_weights:
                        weights = pltriton.load(combine_weights_ref.at[metadata_offsets]).astype(jnp.float32)
                        if use_inline_fma:
                            acc = _weighted_accumulate_fma(
                                acc,
                                values,
                                weights,
                                k_block_size=k_block_size,
                                hidden_block_size=hidden_block_size,
                            )
                        else:
                            acc = acc + jnp.sum(values * weights[:, None], axis=0)
                    else:
                        acc = acc + jnp.sum(values, axis=0)
                else:
                    k_mask = k_offsets < topk
                    safe_k_offsets = jnp.minimum(k_offsets, topk - 1)
                    metadata_offsets = metadata_base + safe_k_offsets
                    dispatch_rows = (
                        pltriton.load(
                            dispatch_positions_ref.at[metadata_offsets],
                            mask=k_mask,
                            other=0,
                        )
                        + repeat_offset
                    )
                    values = pltriton.load(
                        dispatch_output_ref.at[dispatch_rows[:, None], hidden_offsets[None, :]],
                        mask=k_mask[:, None] & hidden_mask[None, :],
                        other=0.0,
                    ).astype(jnp.float32)
                    if has_weights:
                        weights = pltriton.load(
                            combine_weights_ref.at[metadata_offsets],
                            mask=k_mask,
                            other=0.0,
                        ).astype(jnp.float32)
                        if use_inline_fma:
                            acc = _weighted_accumulate_fma(
                                acc,
                                values,
                                weights,
                                k_block_size=k_block_size,
                                hidden_block_size=hidden_block_size,
                            )
                        else:
                            acc = acc + jnp.sum(values * weights[:, None], axis=0)
                    else:
                        acc = acc + jnp.sum(values, axis=0)

        output = (acc * (1.0 / kernel_repeat)).astype(output_ref.dtype)
        pltriton.store(output_ref.at[token_index, hidden_offsets], output, mask=hidden_mask)


def _gather_sum_pallas_triton_token_kblock_kernel(
    dispatch_output_ref,
    dispatch_positions_ref,
    combine_weights_ref,
    repeat_offsets_ref,
    output_ref,
    *,
    topk: int,
    hidden_block_size: int,
    kernel_repeat: int,
) -> None:
    token_index = pl.program_id(0)
    hidden_block = pl.program_id(1)
    hidden_offsets = hidden_block * hidden_block_size + jnp.arange(hidden_block_size)
    topk_offsets = jnp.arange(topk)

    acc = jnp.zeros((hidden_block_size,), dtype=output_ref.dtype)
    for repeat_index in range(kernel_repeat):
        repeat_offset = pltriton.load(repeat_offsets_ref.at[repeat_index])
        dispatch_rows = pltriton.load(dispatch_positions_ref.at[token_index, topk_offsets]) + repeat_offset
        weights = pltriton.load(combine_weights_ref.at[token_index, topk_offsets]).astype(output_ref.dtype)
        values = pltriton.load(dispatch_output_ref.at[dispatch_rows[:, None], hidden_offsets[None, :]]).astype(
            output_ref.dtype
        )
        acc = (acc + jnp.sum((values * weights[:, None]).astype(output_ref.dtype), axis=0)).astype(output_ref.dtype)

    if kernel_repeat == 1:
        output = acc
    else:
        output = (acc.astype(jnp.float32) * (1.0 / kernel_repeat)).astype(output_ref.dtype)
    pltriton.store(output_ref.at[token_index, hidden_offsets], output)


def _gather_ragged_dot_pallas_triton_kernel(
    x_ref,
    token_ids_sort_ref,
    rhs_ref,
    lo_ref,
    hi_ref,
    output_ref,
    *,
    row_block_size: int,
    contraction_block_size: int,
) -> None:
    expert = pl.program_id(2)
    row_block = pl.program_id(0)
    lo = pltriton.load(lo_ref.at[expert])
    hi = pltriton.load(hi_ref.at[expert])
    start_m = lo + row_block * row_block_size

    @pl.when(start_m < hi)
    def _compute():
        row_offsets = start_m + jnp.arange(row_block_size)
        local_columns = jnp.arange(output_ref.shape[1])
        token_ids = pltriton.load(
            token_ids_sort_ref.at[row_offsets],
            mask=row_offsets < hi,
            other=0,
        )
        acc = jnp.zeros((row_block_size, output_ref.shape[1]), dtype=jnp.float32)

        def body(k_block: jax.Array, acc: jax.Array) -> jax.Array:
            start_k = k_block * contraction_block_size
            k_offsets = start_k + jnp.arange(contraction_block_size)
            k_mask = k_offsets < x_ref.shape[1]
            values = pltriton.load(
                x_ref.at[token_ids[:, None], k_offsets[None, :]],
                mask=(row_offsets < hi)[:, None] & k_mask[None, :],
                other=0.0,
            )
            weights = pltriton.load(
                rhs_ref.at[k_offsets[:, None], local_columns[None, :]],
                mask=k_mask[:, None],
                other=0.0,
            )
            dtype = jnp.result_type(values, weights)
            return acc + pl.dot(values.astype(dtype), weights.astype(dtype))

        acc = jax.lax.fori_loop(0, pl.cdiv(x_ref.shape[1], contraction_block_size), body, acc)
        pltriton.store(
            output_ref.at[row_offsets[:, None], local_columns[None, :]],
            acc.astype(output_ref.dtype),
            mask=row_offsets[:, None] < hi,
        )


def _gather_sum_cost_estimate(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    hidden: int,
    kernel_repeat: int,
    output_dtype: jnp.dtype,
) -> pl.CostEstimate:
    tokens, topk = dispatch_positions.shape
    flops = tokens * hidden * topk * kernel_repeat * 2
    bytes_accessed = (
        tokens * topk * kernel_repeat * hidden * jnp.dtype(dispatch_output.dtype).itemsize
        + tokens * topk * kernel_repeat * jnp.dtype(dispatch_positions.dtype).itemsize
        + tokens * topk * kernel_repeat * jnp.dtype(combine_weights.dtype).itemsize
        + tokens * hidden * jnp.dtype(output_dtype).itemsize
    )
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)


def _gather_ragged_dot_cost_estimate(
    x: jax.Array,
    token_ids_sort: jax.Array,
    rhs: jax.Array,
    *,
    output_dtype: jnp.dtype,
) -> pl.CostEstimate:
    rows = token_ids_sort.shape[0]
    hidden = x.shape[1]
    output = rhs.shape[2]
    flops = rows * hidden * output * 2
    bytes_accessed = (
        rows * hidden * jnp.dtype(x.dtype).itemsize
        + rows * jnp.dtype(token_ids_sort.dtype).itemsize
        + rhs.size * jnp.dtype(rhs.dtype).itemsize
        + rows * output * jnp.dtype(output_dtype).itemsize
    )
    return pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed)


def _sonic_gather_ragged_dot_pallas_triton_call(
    x: jax.Array,
    token_ids_sort: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    block_sizes: SonicGatherRaggedDotBlockSizes,
    interpret: bool,
) -> jax.Array:
    if pltriton is None:
        raise PallasUnsupportedError("Pallas Triton backend is not available.")
    _validate_gather_ragged_dot_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError("Pallas Triton gather-ragged-dot requires GPU backend unless interpret=True.")

    rows = token_ids_sort.shape[0]
    hidden = x.shape[1]
    output = rhs.shape[2]
    experts = rhs.shape[0]
    output_block_size = min(128, int(pl.next_power_of_2(output)))
    contraction_block_size = min(block_sizes.contraction_block_size, int(pl.next_power_of_2(hidden)))
    row_block_size = min(block_sizes.row_block_size, int(pl.next_power_of_2(rows)))
    cum_rows = jnp.cumulative_sum(group_sizes, include_initial=True)

    return pl.pallas_call(
        lambda x_ref, token_ids_sort_ref, rhs_ref, lo_ref, hi_ref, output_ref: (
            _gather_ragged_dot_pallas_triton_kernel(
                x_ref,
                token_ids_sort_ref,
                rhs_ref,
                lo_ref,
                hi_ref,
                output_ref,
                row_block_size=row_block_size,
                contraction_block_size=contraction_block_size,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((rows, output), x.dtype),
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.BlockSpec((None, hidden, output_block_size), lambda _, j, e: (e, 0, j)),
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((rows, output_block_size), lambda _, j, __: (0, j)),
        grid=(pl.cdiv(rows, row_block_size), pl.cdiv(output, output_block_size), experts),
        compiler_params=pltriton.CompilerParams(num_warps=block_sizes.num_warps, num_stages=4),
        cost_estimate=_gather_ragged_dot_cost_estimate(
            x,
            token_ids_sort,
            rhs,
            output_dtype=x.dtype,
        ),
        interpret=interpret,
        name="sonic_gather_ragged_dot_pallas_triton",
    )(x, token_ids_sort, rhs, cum_rows[:-1], cum_rows[1:])


def _sonic_gather_sum_pallas_mgpu_call(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    block_sizes: SonicGatherSumBlockSizes,
    interpret: bool,
    repeat_offsets: jax.Array | None = None,
) -> jax.Array:
    if plgpu is None:
        raise PallasUnsupportedError("Pallas Mosaic GPU backend is not available.")
    _validate_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError("Pallas SonicMoE gather-sum requires GPU backend unless interpret=True.")

    dispatch_output, dispatch_positions, combine_weights, tokens, hidden = _pad_for_pallas(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=block_sizes,
    )
    padded_tokens = dispatch_positions.shape[0]
    padded_hidden = dispatch_output.shape[1]
    topk = dispatch_positions.shape[1]
    if repeat_offsets is None:
        repeat_offsets = jnp.zeros((block_sizes.kernel_repeat,), dtype=jnp.int32)

    out = pl.pallas_call(
        lambda dispatch_output_ref, dispatch_positions_ref, combine_weights_ref, repeat_offsets_ref, output_ref: (
            _gather_sum_pallas_kernel(
                dispatch_output_ref,
                dispatch_positions_ref,
                combine_weights_ref,
                repeat_offsets_ref,
                output_ref,
                topk=topk,
                token_block_size=block_sizes.token_block_size,
                hidden_block_size=block_sizes.hidden_block_size,
                kernel_repeat=block_sizes.kernel_repeat,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, padded_hidden), dispatch_output.dtype),
        grid=(
            pl.cdiv(padded_tokens, block_sizes.token_block_size),
            pl.cdiv(padded_hidden, block_sizes.hidden_block_size),
        ),
        compiler_params=plgpu.CompilerParams(),
        cost_estimate=_gather_sum_cost_estimate(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            hidden=padded_hidden,
            kernel_repeat=block_sizes.kernel_repeat,
            output_dtype=dispatch_output.dtype,
        ),
        interpret=interpret,
        name="sonic_gather_sum_pallas_mgpu",
    )(dispatch_output, dispatch_positions, combine_weights, repeat_offsets)
    return out[:tokens, :hidden]


def _sonic_gather_sum_pallas_triton_call(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    block_sizes: SonicGatherSumBlockSizes,
    interpret: bool,
    repeat_offsets: jax.Array | None = None,
) -> jax.Array:
    if pltriton is None:
        raise PallasUnsupportedError("Pallas Triton backend is not available.")
    _validate_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError("Pallas Triton SonicMoE gather-sum requires GPU backend unless interpret=True.")

    dispatch_output, dispatch_positions, combine_weights, tokens, hidden = _pad_for_pallas(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=block_sizes,
    )
    padded_tokens = dispatch_positions.shape[0]
    padded_hidden = dispatch_output.shape[1]
    topk = dispatch_positions.shape[1]
    if repeat_offsets is None:
        repeat_offsets = jnp.zeros((block_sizes.kernel_repeat,), dtype=jnp.int32)

    out = pl.pallas_call(
        lambda dispatch_output_ref, dispatch_positions_ref, combine_weights_ref, repeat_offsets_ref, output_ref: (
            _gather_sum_pallas_triton_kernel(
                dispatch_output_ref,
                dispatch_positions_ref,
                combine_weights_ref,
                repeat_offsets_ref,
                output_ref,
                topk=topk,
                token_block_size=block_sizes.token_block_size,
                hidden_block_size=block_sizes.hidden_block_size,
                kernel_repeat=block_sizes.kernel_repeat,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, padded_hidden), dispatch_output.dtype),
        grid=(
            pl.cdiv(padded_tokens, block_sizes.token_block_size),
            pl.cdiv(padded_hidden, block_sizes.hidden_block_size),
        ),
        compiler_params=pltriton.CompilerParams(num_warps=block_sizes.num_warps, num_stages=4),
        cost_estimate=_gather_sum_cost_estimate(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            hidden=padded_hidden,
            kernel_repeat=block_sizes.kernel_repeat,
            output_dtype=dispatch_output.dtype,
        ),
        interpret=interpret,
        name="sonic_gather_sum_pallas_triton",
    )(dispatch_output, dispatch_positions, combine_weights, repeat_offsets)
    return out[:tokens, :hidden]


def _sonic_gather_sum_pallas_triton_token_loop_call(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    block_sizes: SonicGatherSumBlockSizes,
    interpret: bool,
    repeat_offsets: jax.Array | None = None,
) -> jax.Array:
    if pltriton is None:
        raise PallasUnsupportedError("Pallas Triton backend is not available.")
    _validate_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError(
            "Pallas Triton SonicMoE token-loop gather-sum requires GPU unless interpret=True."
        )

    dispatch_output, dispatch_positions, combine_weights, tokens, hidden = _pad_for_pallas(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=dataclasses.replace(block_sizes, token_block_size=1),
    )
    padded_tokens = dispatch_positions.shape[0]
    padded_hidden = dispatch_output.shape[1]
    topk = dispatch_positions.shape[1]
    hidden_tiles = math.ceil(padded_hidden / block_sizes.hidden_block_size)
    if repeat_offsets is None:
        repeat_offsets = jnp.zeros((block_sizes.kernel_repeat,), dtype=jnp.int32)

    out = pl.pallas_call(
        lambda dispatch_output_ref, dispatch_positions_ref, combine_weights_ref, repeat_offsets_ref, output_ref: (
            _gather_sum_pallas_triton_token_loop_kernel(
                dispatch_output_ref,
                dispatch_positions_ref,
                combine_weights_ref,
                repeat_offsets_ref,
                output_ref,
                topk=topk,
                hidden_block_size=block_sizes.hidden_block_size,
                hidden_tiles=hidden_tiles,
                kernel_repeat=block_sizes.kernel_repeat,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, padded_hidden), dispatch_output.dtype),
        grid=(padded_tokens,),
        compiler_params=pltriton.CompilerParams(num_warps=block_sizes.num_warps, num_stages=4),
        cost_estimate=_gather_sum_cost_estimate(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            hidden=padded_hidden,
            kernel_repeat=block_sizes.kernel_repeat,
            output_dtype=dispatch_output.dtype,
        ),
        interpret=interpret,
        name="sonic_gather_sum_pallas_triton_token_loop",
    )(dispatch_output, dispatch_positions, combine_weights, repeat_offsets)
    return out[:tokens, :hidden]


def _sonic_gather_sum_pallas_triton_faithful_call(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array | None,
    *,
    block_sizes: SonicGatherSumBlockSizes,
    interpret: bool,
    repeat_offsets: jax.Array | None = None,
) -> jax.Array:
    if pltriton is None:
        raise PallasUnsupportedError("Pallas Triton backend is not available.")
    _validate_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError("Pallas Triton faithful SonicMoE gather-sum requires GPU unless interpret=True.")
    if dispatch_output.ndim != 2:
        raise ValueError(f"dispatch_output must be rank-2 [TK, D], got shape={dispatch_output.shape}")
    if dispatch_positions.ndim != 2:
        raise ValueError(f"dispatch_positions must be rank-2 [T, K], got shape={dispatch_positions.shape}")
    if combine_weights is not None and combine_weights.shape != dispatch_positions.shape:
        raise ValueError(
            "combine_weights and dispatch_positions must have identical [T, K] shapes; "
            f"got {combine_weights.shape} vs {dispatch_positions.shape}"
        )

    tokens, topk = dispatch_positions.shape
    hidden = dispatch_output.shape[1]
    if topk <= 0:
        raise ValueError("dispatch_positions must have at least one top-k column")
    hidden_tiles = math.ceil(hidden / block_sizes.hidden_block_size)
    k_tiles = math.ceil(topk / block_sizes.k_block_size)
    if repeat_offsets is None:
        repeat_offsets = jnp.zeros((block_sizes.kernel_repeat,), dtype=jnp.int32)
    has_weights = combine_weights is not None
    if combine_weights is None:
        combine_weights = jnp.zeros(dispatch_positions.shape, dtype=dispatch_output.dtype)
    combine_weights_2d = combine_weights
    flat_dispatch_positions = dispatch_positions.reshape((tokens * topk,))
    flat_combine_weights = combine_weights.reshape((tokens * topk,))

    out = pl.pallas_call(
        lambda dispatch_output_ref, dispatch_positions_ref, combine_weights_ref, repeat_offsets_ref, output_ref: (
            _gather_sum_pallas_triton_faithful_kernel(
                dispatch_output_ref,
                dispatch_positions_ref,
                combine_weights_ref,
                repeat_offsets_ref,
                output_ref,
                topk=topk,
                hidden=hidden,
                hidden_block_size=block_sizes.hidden_block_size,
                hidden_tiles=hidden_tiles,
                k_block_size=block_sizes.k_block_size,
                k_tiles=k_tiles,
                kernel_repeat=block_sizes.kernel_repeat,
                has_weights=has_weights,
                use_inline_fma=block_sizes.use_inline_fma,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), dispatch_output.dtype),
        grid=(tokens,),
        compiler_params=pltriton.CompilerParams(num_warps=block_sizes.num_warps, num_stages=4),
        cost_estimate=_gather_sum_cost_estimate(
            dispatch_output,
            dispatch_positions,
            combine_weights_2d,
            hidden=hidden,
            kernel_repeat=block_sizes.kernel_repeat,
            output_dtype=dispatch_output.dtype,
        ),
        interpret=interpret,
        name="sonic_gather_sum_pallas_triton_faithful",
    )(dispatch_output, flat_dispatch_positions, flat_combine_weights, repeat_offsets)
    return out


def _sonic_gather_sum_pallas_triton_token_kblock_call(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    block_sizes: SonicGatherSumBlockSizes,
    interpret: bool,
    repeat_offsets: jax.Array | None = None,
) -> jax.Array:
    if pltriton is None:
        raise PallasUnsupportedError("Pallas Triton backend is not available.")
    _validate_block_sizes(block_sizes)
    if jax.default_backend() != "gpu" and not interpret:
        raise PallasUnsupportedError(
            "Pallas Triton SonicMoE token-kblock gather-sum requires GPU unless interpret=True."
        )

    dispatch_output, dispatch_positions, combine_weights, tokens, hidden = _pad_for_pallas(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=dataclasses.replace(block_sizes, token_block_size=1),
    )
    padded_tokens = dispatch_positions.shape[0]
    padded_hidden = dispatch_output.shape[1]
    topk = dispatch_positions.shape[1]
    if repeat_offsets is None:
        repeat_offsets = jnp.zeros((block_sizes.kernel_repeat,), dtype=jnp.int32)

    out = pl.pallas_call(
        lambda dispatch_output_ref, dispatch_positions_ref, combine_weights_ref, repeat_offsets_ref, output_ref: (
            _gather_sum_pallas_triton_token_kblock_kernel(
                dispatch_output_ref,
                dispatch_positions_ref,
                combine_weights_ref,
                repeat_offsets_ref,
                output_ref,
                topk=topk,
                hidden_block_size=block_sizes.hidden_block_size,
                kernel_repeat=block_sizes.kernel_repeat,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, padded_hidden), dispatch_output.dtype),
        grid=(
            padded_tokens,
            pl.cdiv(padded_hidden, block_sizes.hidden_block_size),
        ),
        compiler_params=pltriton.CompilerParams(num_warps=block_sizes.num_warps, num_stages=4),
        cost_estimate=_gather_sum_cost_estimate(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            hidden=padded_hidden,
            kernel_repeat=block_sizes.kernel_repeat,
            output_dtype=dispatch_output.dtype,
        ),
        interpret=interpret,
        name="sonic_gather_sum_pallas_triton_token_kblock",
    )(dispatch_output, dispatch_positions, combine_weights, repeat_offsets)
    return out[:tokens, :hidden]


def _with_xla_backward(
    impl: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
) -> jax.Array:
    @jax.custom_vjp
    def _call(
        dispatch_output: jax.Array,
        dispatch_positions: jax.Array,
        combine_weights: jax.Array,
    ) -> jax.Array:
        return impl(dispatch_output, dispatch_positions, combine_weights)

    def _fwd(
        dispatch_output: jax.Array,
        dispatch_positions: jax.Array,
        combine_weights: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        out = impl(dispatch_output, dispatch_positions, combine_weights)
        return out, (dispatch_output, dispatch_positions, combine_weights)

    def _bwd(
        residuals: tuple[jax.Array, jax.Array, jax.Array],
        dout: jax.Array,
    ) -> tuple[jax.Array, None, jax.Array]:
        dispatch_output, dispatch_positions, combine_weights = residuals
        return _sonic_gather_sum_backward(dispatch_output, dispatch_positions, combine_weights, dout)

    _call.defvjp(_fwd, _bwd)
    return _call(dispatch_output, dispatch_positions, combine_weights)


def sonic_gather_sum_pallas_triton(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    block_sizes: SonicGatherSumBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "T D"]:
    """Pallas Triton gather-and-sum combine with an XLA backward rule."""
    _validate_shapes(dispatch_output, dispatch_positions, combine_weights)
    if block_sizes is None:
        block_sizes = SonicGatherSumBlockSizes.get_default()
    return _with_xla_backward(
        lambda dispatch_output, dispatch_positions, combine_weights: _sonic_gather_sum_pallas_triton_call(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            block_sizes=block_sizes,
            interpret=interpret,
        ),
        dispatch_output,
        dispatch_positions,
        combine_weights,
    )


def sonic_gather_sum_pallas_triton_faithful(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    block_sizes: SonicGatherSumBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "T D"]:
    """Source-faithful Pallas Triton port of SonicMoE token gather/sum.

    This path keeps Sonic's one-program-per-token launch, inner hidden-tile
    loop, explicit ``BLOCK_K`` loop, masks, and fp32 accumulator. It is meant
    as a comparison control, not as the default production gather-sum backend.
    """
    _validate_shapes(dispatch_output, dispatch_positions, combine_weights)
    if block_sizes is None:
        block_sizes = SonicGatherSumBlockSizes.get_default()
    return _with_xla_backward(
        lambda dispatch_output, dispatch_positions, combine_weights: _sonic_gather_sum_pallas_triton_faithful_call(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            block_sizes=block_sizes,
            interpret=interpret,
        ),
        dispatch_output,
        dispatch_positions,
        combine_weights,
    )


def sonic_gather_sum_pallas_mgpu(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    block_sizes: SonicGatherSumBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "T D"]:
    """Pallas Mosaic GPU gather-and-sum combine with an XLA backward rule."""
    _validate_shapes(dispatch_output, dispatch_positions, combine_weights)
    if block_sizes is None:
        block_sizes = SonicGatherSumBlockSizes.get_default()

    @jax.custom_vjp
    def _call(
        dispatch_output: jax.Array,
        dispatch_positions: jax.Array,
        combine_weights: jax.Array,
    ) -> jax.Array:
        return _sonic_gather_sum_pallas_mgpu_call(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            block_sizes=block_sizes,
            interpret=interpret,
        )

    def _fwd(
        dispatch_output: jax.Array,
        dispatch_positions: jax.Array,
        combine_weights: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        out = _sonic_gather_sum_pallas_mgpu_call(
            dispatch_output,
            dispatch_positions,
            combine_weights,
            block_sizes=block_sizes,
            interpret=interpret,
        )
        return out, (dispatch_output, dispatch_positions, combine_weights)

    def _bwd(
        residuals: tuple[jax.Array, jax.Array, jax.Array],
        dout: jax.Array,
    ) -> tuple[jax.Array, None, jax.Array]:
        dispatch_output, dispatch_positions, combine_weights = residuals
        return _sonic_gather_sum_backward(dispatch_output, dispatch_positions, combine_weights, dout)

    _call.defvjp(_fwd, _bwd)
    return _call(dispatch_output, dispatch_positions, combine_weights)


def sonic_gather_ragged_dot_pallas_triton(
    x: Float[Array, "T D"],
    token_ids_sort: Int[Array, "TK"],
    rhs: Float[Array, "E D N"],
    group_sizes: Int[Array, "E"],
    *,
    block_sizes: SonicGatherRaggedDotBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "TK N"]:
    """Pallas Triton gather-fused ragged dot with an XLA backward rule."""
    _validate_gather_ragged_dot_shapes(x, token_ids_sort, rhs, group_sizes)
    if block_sizes is None:
        block_sizes = SonicGatherRaggedDotBlockSizes.get_default()

    @jax.custom_vjp
    def _call(
        x: jax.Array,
        token_ids_sort: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
    ) -> jax.Array:
        return _sonic_gather_ragged_dot_pallas_triton_call(
            x,
            token_ids_sort,
            rhs,
            group_sizes,
            block_sizes=block_sizes,
            interpret=interpret,
        )

    def _fwd(
        x: jax.Array,
        token_ids_sort: jax.Array,
        rhs: jax.Array,
        group_sizes: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        out = _sonic_gather_ragged_dot_pallas_triton_call(
            x,
            token_ids_sort,
            rhs,
            group_sizes,
            block_sizes=block_sizes,
            interpret=interpret,
        )
        return out, (x, token_ids_sort, rhs, group_sizes)

    def _bwd(
        residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        dout: jax.Array,
    ) -> tuple[jax.Array, None, jax.Array, None]:
        x, token_ids_sort, rhs, group_sizes = residuals
        return _sonic_gather_ragged_dot_backward(x, token_ids_sort, rhs, group_sizes, dout)

    _call.defvjp(_fwd, _bwd)
    return _call(x, token_ids_sort, rhs, group_sizes)


def sonic_topk_metadata_pallas_triton(
    selected_experts: Int[Array, "T K"],
    *,
    num_experts: int,
    block_sizes: SonicMetadataBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Int[Array, "TK"], Int[Array, "TK"], Int[Array, "TK"], Int[Array, "E"], Int[Array, "E1"]]:
    """Pallas Triton port of SonicMoE fixed-top-k routing metadata."""
    _validate_metadata_shapes(selected_experts, num_experts)
    if block_sizes is None:
        block_sizes = SonicMetadataBlockSizes.get_default()
    return _sonic_topk_metadata_pallas_triton_call(
        selected_experts,
        num_experts=num_experts,
        block_sizes=block_sizes,
        interpret=interpret,
    )


def sonic_topk_metadata(
    selected_experts: Int[Array, "T K"],
    *,
    num_experts: int,
    implementation: SonicMetadataImplementation | Sequence[SonicMetadataImplementation] | None = None,
    block_sizes: SonicMetadataBlockSizes | None = None,
    interpret: bool = False,
) -> tuple[Int[Array, "TK"], Int[Array, "TK"], Int[Array, "TK"], Int[Array, "E"], Int[Array, "E1"]]:
    """Dispatch SonicMoE fixed-top-k routing metadata to the requested implementation."""
    _validate_metadata_shapes(selected_experts, num_experts)
    if implementation is None:
        implementations: Sequence[SonicMetadataImplementation]
        implementations = ("pallas_triton", "xla") if jax.default_backend() == "gpu" else ("xla",)
    elif isinstance(implementation, str):
        implementations = (cast(SonicMetadataImplementation, implementation),)
    else:
        implementations = implementation

    errors: list[Exception] = []
    for impl in implementations:
        if impl == "xla":
            return sonic_topk_metadata_reference(selected_experts, num_experts=num_experts)
        if impl == "pallas_triton":
            try:
                return sonic_topk_metadata_pallas_triton(
                    selected_experts,
                    num_experts=num_experts,
                    block_sizes=block_sizes,
                    interpret=interpret,
                )
            except PallasUnsupportedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
            except NotImplementedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
        raise ValueError(f"Unknown SonicMoE metadata implementation: {impl!r}")

    if errors:
        raise RuntimeError("No SonicMoE metadata implementation was selected") from errors[-1]
    raise RuntimeError("No SonicMoE metadata implementation was selected")


def sonic_gather_ragged_dot(
    x: Float[Array, "T D"],
    token_ids_sort: Int[Array, "TK"],
    rhs: Float[Array, "E D N"],
    group_sizes: Int[Array, "E"],
    *,
    implementation: SonicGatherRaggedDotImplementation | Sequence[SonicGatherRaggedDotImplementation] | None = None,
    block_sizes: SonicGatherRaggedDotBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "TK N"]:
    """Dispatch gather-fused ragged dot to the requested implementation."""
    _validate_gather_ragged_dot_shapes(x, token_ids_sort, rhs, group_sizes)
    if implementation is None:
        implementations: Sequence[SonicGatherRaggedDotImplementation]
        implementations = ("pallas_triton", "xla") if jax.default_backend() == "gpu" else ("xla",)
    elif isinstance(implementation, str):
        implementations = (cast(SonicGatherRaggedDotImplementation, implementation),)
    else:
        implementations = implementation

    errors: list[Exception] = []
    for impl in implementations:
        if impl == "xla":
            return sonic_gather_ragged_dot_reference(x, token_ids_sort, rhs, group_sizes)
        if impl == "pallas_triton":
            try:
                return sonic_gather_ragged_dot_pallas_triton(
                    x,
                    token_ids_sort,
                    rhs,
                    group_sizes,
                    block_sizes=block_sizes,
                    interpret=interpret,
                )
            except PallasUnsupportedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
            except NotImplementedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
        raise ValueError(f"Unknown SonicMoE gather-ragged-dot implementation: {impl!r}")

    if errors:
        raise RuntimeError("No SonicMoE gather-ragged-dot implementation was selected") from errors[-1]
    raise RuntimeError("No SonicMoE gather-ragged-dot implementation was selected")


def _normalize_implementation(
    implementation: SonicGatherSumImplementation | Sequence[SonicGatherSumImplementation] | None,
) -> Sequence[SonicGatherSumImplementation]:
    if implementation is None:
        if jax.default_backend() == "gpu":
            return ("pallas_triton", "xla")
        return ("xla",)
    if isinstance(implementation, str):
        return (cast(SonicGatherSumImplementation, implementation),)
    return implementation


def sonic_gather_sum(
    dispatch_output: Float[Array, "TK D"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    implementation: SonicGatherSumImplementation | Sequence[SonicGatherSumImplementation] | None = None,
    block_sizes: SonicGatherSumBlockSizes | None = None,
    interpret: bool = False,
) -> Float[Array, "T D"]:
    """Dispatch gather-and-sum combine to the requested implementation."""
    _validate_shapes(dispatch_output, dispatch_positions, combine_weights)
    errors: list[Exception] = []
    for impl in _normalize_implementation(implementation):
        if impl == "xla":
            return sonic_gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)
        if impl == "pallas_triton":
            try:
                return sonic_gather_sum_pallas_triton(
                    dispatch_output,
                    dispatch_positions,
                    combine_weights,
                    block_sizes=block_sizes,
                    interpret=interpret,
                )
            except PallasUnsupportedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
            except NotImplementedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
        if impl == "pallas_triton_faithful":
            try:
                return sonic_gather_sum_pallas_triton_faithful(
                    dispatch_output,
                    dispatch_positions,
                    combine_weights,
                    block_sizes=block_sizes,
                    interpret=interpret,
                )
            except PallasUnsupportedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
            except NotImplementedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
        if impl == "pallas_mgpu":
            try:
                return sonic_gather_sum_pallas_mgpu(
                    dispatch_output,
                    dispatch_positions,
                    combine_weights,
                    block_sizes=block_sizes,
                    interpret=interpret,
                )
            except PallasUnsupportedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
            except NotImplementedError as exc:
                errors.append(exc)
                if implementation is not None:
                    raise
                continue
        else:
            raise ValueError(f"Unknown SonicMoE gather-sum implementation: {impl!r}")

    if errors:
        raise RuntimeError("No SonicMoE gather-sum implementation was selected") from errors[-1]
    raise RuntimeError("No SonicMoE gather-sum implementation was selected")


__all__ = [
    "PallasUnsupportedError",
    "SonicGatherRaggedDotBlockSizes",
    "SonicGatherRaggedDotImplementation",
    "SonicGatherSumBlockSizes",
    "SonicGatherSumImplementation",
    "SonicMetadataBlockSizes",
    "SonicMetadataImplementation",
    "sonic_gather_ragged_dot",
    "sonic_gather_ragged_dot_pallas_triton",
    "sonic_gather_ragged_dot_reference",
    "sonic_gather_sum",
    "sonic_gather_sum_pallas_mgpu",
    "sonic_gather_sum_pallas_triton",
    "sonic_gather_sum_pallas_triton_faithful",
    "sonic_gather_sum_reference",
    "sonic_topk_metadata",
    "sonic_topk_metadata_pallas_triton",
    "sonic_topk_metadata_reference",
]
