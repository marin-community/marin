# Copyright (c) 2025, Tri Dao.
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CuTe implementation of Shuttle's mergeable normalized-exponential Fold.

The register layout and warp reductions are adapted from flash-attn-4 4.0.0b16
``flash_attn/cute/softmax.py``.  The retained code is a generic state machine:
it owns only ``(row_max, row_sum_exp)`` and rescales an arbitrary weighted
accumulator.  QK, PV, attention, and mask semantics live outside this module.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from flash_attn.cute import utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase


@dataclass
class NormalizedExpFoldState(ParamsBase):
    """Register-resident ``(max, sum_exp)`` state for a weighted Fold."""

    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    score_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        score_scale: Float32 | None = None,
    ) -> NormalizedExpFoldState:
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return NormalizedExpFoldState(
            scale_log2, num_rows, row_max, row_sum, arch, score_scale
        )

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def update(
        self,
        values: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Merge one value tile and replace it with normalized exponentials."""
        values_mn = layout_utils.reshape_acc_to_mn(values)
        # Keep the register state as explicit parent-region SSA values. CuTe's
        # register-tensor stores are value-like, so repeatedly resolving fields
        # inside the row loop can leave the updated tensor scoped to that child
        # region instead of loop-carried to finalization.
        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch
        prior_accumulator_scale = cute.make_fragment_like(row_max, Float32)

        for row in cutlass.range(cute.size(row_max), unroll_full=True):
            row_values = values_mn[row, None].load()
            new_max = utils.fmax_reduce(
                row_values,
                init_val=row_max[row] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )
            new_max = cute.arch.warp_reduction_max(new_max, threads_in_group=4)
            old_max = row_max[row]
            row_max[row] = new_max
            safe_max = new_max
            if cutlass.const_expr(check_inf):
                safe_max = 0.0 if safe_max == -Float32.inf else safe_max
            scaled_max = safe_max * scale_log2
            exponentials = cute.math.exp2(
                row_values * scale_log2 - scaled_max, fastmath=True
            )
            if cutlass.const_expr(is_first):
                prior_accumulator_scale[row] = 1.0
                new_sum = utils.fadd_reduce(exponentials, init_val=None, arch=arch)
            else:
                prior_accumulator_scale[row] = cute.math.exp2(
                    (old_max - safe_max) * scale_log2,
                    fastmath=True,
                )
                new_sum = utils.fadd_reduce(
                    exponentials,
                    init_val=row_sum[row] * prior_accumulator_scale[row],
                    arch=arch,
                )
            row_sum[row] = new_sum
            values_mn[row, None].store(exponentials)
        return prior_accumulator_scale

    @cute.jit
    def finalize(
        self,
        output_scale: Float32 = 1.0,
        extra_logit: Float32 | cute.Tensor | None = None,
    ) -> cute.Tensor:
        """Return the final weighted-accumulator scale and replace sum with LSE."""
        if cutlass.const_expr(
            extra_logit is not None and isinstance(extra_logit, cute.Tensor)
        ):
            assert cute.size(extra_logit) == cute.size(self.row_sum)
        # CuTe needs the register tensors to be bound outside the reduction's
        # child region so their definitions dominate generated layout uses.
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        accumulator_scale = cute.make_fragment_like(row_max, Float32)
        for row in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(extra_logit is not None):
                value = (
                    extra_logit
                    if not isinstance(extra_logit, cute.Tensor)
                    else extra_logit[row]
                )
                row_sum[row] += cute.math.exp2(
                    value * math.log2(math.e) - row_max[row] * scale_log2,
                    fastmath=True,
                )
            invalid_sum = row_sum[row] == 0.0 or row_sum[row] != row_sum[row]
            denominator = row_sum[row] if not invalid_sum else 1.0
            accumulator_scale[row] = cute.arch.rcp_approx(denominator) * output_scale
            sum_value = row_sum[row]
            row_sum[row] = (
                (row_max[row] * scale_log2 + cute.math.log2(sum_value, fastmath=True))
                * math.log(2.0)
                if not invalid_sum
                else -Float32.inf
            )
        return accumulator_scale

    @cute.jit
    def rescale_weighted_accumulator(
        self, accumulator: cute.Tensor, row_scale: cute.Tensor
    ) -> None:
        """Multiply each weighted-accumulator row by its Fold merge scale."""
        accumulator_mn = layout_utils.reshape_acc_to_mn(accumulator)
        assert cute.size(row_scale) == cute.size(accumulator_mn, mode=[0])
        for row in cutlass.range(cute.size(row_scale), unroll_full=True):
            accumulator_mn[row, None].store(
                accumulator_mn[row, None].load() * row_scale[row]
            )


@cute.jit
def _floor_packed_index(
    query_index, head_group_size: cutlass.Constexpr[int]
) -> cute.Tensor:
    if cutlass.const_expr(head_group_size == 1):
        return query_index
    return query_index // head_group_size


@cute.jit
def apply_score_map_inner(
    score_tensor,
    index_tensor,
    score_map: cutlass.Constexpr,
    batch_idx,
    head_idx,
    score_scale,
    vec_size: cutlass.Constexpr,
    accumulator_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info: SeqlenInfoQK,
    constant_query_idx: cutlass.Constexpr,
    head_group_size: cutlass.Constexpr[int] = 1,
) -> None:
    """Apply a generated scalar Map to score values and logical indices."""
    value_count = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vector = cute.make_rmem_tensor(vec_size, accumulator_dtype)
    key_index_vector = cute.make_rmem_tensor(vec_size, cutlass.Int32)
    query_index_vector = cute.make_rmem_tensor(vec_size, cutlass.Int32)
    batch_vector = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to(
        (vec_size,)
    )
    if cutlass.const_expr(head_group_size > 1 and constant_query_idx is None):
        head_index_vector = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for offset in cutlass.range(0, value_count, vec_size, unroll_full=True):
        for lane in cutlass.range(vec_size, unroll_full=True):
            score_vector[lane] = score_tensor[offset + lane] * score_scale
            packed_query = index_tensor[offset + lane][0]
            if cutlass.const_expr(head_group_size > 1 and constant_query_idx is None):
                logical_query = packed_query // head_group_size
                head_offset = packed_query - logical_query * head_group_size
                head_index_vector[lane] = head_idx * head_group_size + head_offset

            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_query_idx is None):
                    query_divmod, key_divmod = fastdiv_mods
                    _, query_index_vector[lane] = divmod(
                        _floor_packed_index(packed_query, head_group_size),
                        query_divmod,
                    )
                else:
                    _, key_divmod = fastdiv_mods
                _, key_index_vector[lane] = divmod(
                    index_tensor[offset + lane][1], key_divmod
                )
            else:
                if constant_query_idx is None:
                    query_index_vector[lane] = _floor_packed_index(
                        packed_query, head_group_size
                    )
                key_index_vector[lane] = index_tensor[offset + lane][1]

        if cutlass.const_expr(constant_query_idx is None):
            query_indices = query_index_vector.load()
        else:
            query_indices = utils.scalar_to_ssa(
                constant_query_idx, cutlass.Int32
            ).broadcast_to((vec_size,))
        if cutlass.const_expr(head_group_size > 1 and constant_query_idx is None):
            head_indices = head_index_vector.load()
        else:
            head_indices = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to(
                (vec_size,)
            )
        auxiliary_arguments = []
        if cutlass.const_expr(aux_tensors is not None):
            auxiliary_arguments = aux_tensors
        mapped = score_map(
            score_vector.load(),
            batch_vector,
            head_indices,
            q_idx=query_indices,
            kv_idx=key_index_vector.load(),
            seqlen_info=seqlen_info,
            aux_tensors=auxiliary_arguments,
        )
        score_vector.store(mapped)
        for lane in cutlass.range(vec_size, unroll_full=True):
            score_tensor[offset + lane] = score_vector[lane]
