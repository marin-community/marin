# Copyright (c) 2025, Tri Dao.
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CuTe lowering of Shuttle index-domain predicates for SM90 score tiles.

The fragment-coordinate and R2P mechanics are adapted from flash-attn-4
4.0.0b16 ``flash_attn/cute/mask.py``.  The interface is a generic predicate
over ``(batch, head, left_index, right_index)`` and is independent of a
normalized-exponential Fold or contraction kind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from flash_attn.cute import utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from quack import layout_utils

type MaskGenerator = Callable[[int], Uint32]
_R2P_CHUNK = 32


@cute.jit
def _bitmask_below(limit: Int32, chunk: int) -> Uint32:
    shift = max((chunk + 1) * _R2P_CHUNK - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(shift))


@cute.jit
def _apply_r2p_mask(values: cute.Tensor, generator: cutlass.Constexpr[MaskGenerator], rank1: bool = False) -> None:
    columns = const_expr(cute.size(values.shape[cute.rank(values) - 1]) if not rank1 else cute.size(values.shape))
    for chunk in cutlass.range_constexpr(cute.ceil_div(columns, _R2P_CHUNK)):
        keep = generator(chunk)
        for lane in cutlass.range_constexpr(min(_R2P_CHUNK, columns - chunk * _R2P_CHUNK)):
            valid = cutlass.Boolean(keep & (Uint32(1) << lane))
            column = chunk * _R2P_CHUNK + lane
            if const_expr(rank1):
                values[column] = values[column] if valid else -Float32.inf
            else:
                for row in cutlass.range_constexpr(cute.size(values.shape[0])):
                    values[row, column] = values[row, column] if valid else -Float32.inf


@cute.jit
def _sm90_column_to_element(column_limit: Int32) -> Int32:
    return column_limit // 8 * 2 + min(column_limit % 8, 2)


@dataclass(frozen=True)
class DomainRestriction:
    """Apply bounds and an optional generated index predicate to a tile."""

    tile_left: cutlass.Constexpr[int]
    tile_right: cutlass.Constexpr[int]
    domain_info: SeqlenInfoQK
    window_left: Int32 | None = None
    window_right: Int32 | None = None
    head_group_size: cutlass.Constexpr[int] = 1

    @property
    def left_extent(self) -> Int32:
        return self.domain_info.seqlen_q

    @property
    def right_extent(self) -> Int32:
        return self.domain_info.seqlen_k

    @cute.jit
    def apply(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Callable | None] = None,
        aux_tensors: list | None = None,
        fastdiv_mods=(None, None),
    ) -> None:
        """Set values outside the selected index domain to negative infinity."""
        values = acc_S
        left_block = m_block
        right_block = n_block
        tiled_mma = thr_mma
        restrict_bounds = mask_seqlen
        causal = mask_causal
        local = mask_local
        predicate = mask_mod
        assert not (causal and local), "causal and local restrictions are mutually exclusive"
        values_mn = layout_utils.reshape_acc_to_mn(values)
        coordinates = cute.make_identity_tensor((self.tile_left, self.tile_right))
        thread_coordinates = layout_utils.reshape_acc_to_mn(tiled_mma.partition_C(coordinates))
        reference_coordinates = layout_utils.reshape_acc_to_mn(tiled_mma.get_slice(0).partition_C(coordinates))
        thread_column_offset = thread_coordinates[0][1]
        if right_block < 0:
            right_block = 0
        right_limit = self.right_extent - right_block * self.tile_right - thread_column_offset

        if const_expr(predicate is not None):
            self._apply_predicate(
                values_mn,
                thread_coordinates,
                reference_coordinates,
                batch_idx,
                head_idx,
                left_block,
                right_block,
                thread_column_offset,
                restrict_bounds,
                causal,
                local,
                predicate,
                aux_tensors,
                fastdiv_mods,
            )
            return

        if const_expr(causal):
            threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
            packed_row = None
            if const_expr(self.head_group_size != 1):
                assert cute.arch.WARP_SIZE % threads_per_row == 0
                packed_row = (
                    left_block * self.tile_left + thread_coordinates[tiled_mma.thr_idx % threads_per_row, 0][0]
                ) // self.head_group_size
            causal_offset = (
                1 + self.right_extent - right_block * self.tile_right - self.left_extent - thread_column_offset
            )
            for row in cutlass.range(cute.size(thread_coordinates.shape[0]), unroll_full=True):
                if const_expr(self.head_group_size == 1):
                    row_index = thread_coordinates[row, 0][0] + left_block * self.tile_left
                else:
                    row_index = utils.shuffle_sync(packed_row, row % threads_per_row, width=threads_per_row)
                column_limit = row_index + causal_offset
                if const_expr(restrict_bounds):
                    column_limit = cutlass.min(column_limit, right_limit)
                element_limit = _sm90_column_to_element(column_limit)
                _apply_r2p_mask(
                    values_mn[row, None],
                    lambda chunk, limit=element_limit: _bitmask_below(limit, chunk),
                    rank1=True,
                )
            return

        if const_expr(local):
            self._apply_window(
                values_mn,
                thread_coordinates,
                reference_coordinates,
                left_block,
                right_block,
                thread_column_offset,
                restrict_bounds,
            )
            return

        if const_expr(restrict_bounds):
            element_limit = _sm90_column_to_element(right_limit)
            _apply_r2p_mask(values_mn, lambda chunk: _bitmask_below(element_limit, chunk))

    @cute.jit
    def _apply_predicate(
        self,
        values_mn,
        thread_coordinates,
        reference_coordinates,
        batch_idx,
        head_idx,
        left_block,
        right_block,
        thread_column_offset,
        restrict_bounds,
        causal,
        local,
        predicate,
        aux_tensors,
        fastdiv_mods,
    ) -> None:
        has_fastdiv = const_expr(
            fastdiv_mods is not None and fastdiv_mods[0] is not None and fastdiv_mods[1] is not None
        )
        for row in cutlass.range_constexpr(cute.size(thread_coordinates.shape[0])):
            packed_left = thread_coordinates[row, 0][0] + left_block * self.tile_left
            logical_left = packed_left
            logical_head = head_idx
            if const_expr(self.head_group_size != 1):
                logical_left = packed_left // self.head_group_size
                logical_head = head_idx * self.head_group_size + packed_left % self.head_group_size
            left_for_predicate = logical_left
            if const_expr(has_fastdiv and aux_tensors is not None):
                _, left_for_predicate = divmod(logical_left, fastdiv_mods[0])
            for column in cutlass.range_constexpr(cute.size(thread_coordinates.shape[1])):
                global_right = thread_column_offset + reference_coordinates[0, column][1] + right_block * self.tile_right
                right_for_predicate = global_right
                if const_expr(has_fastdiv and aux_tensors is not None):
                    _, right_for_predicate = divmod(global_right, fastdiv_mods[1])
                selected = predicate(
                    utils.scalar_to_ssa(batch_idx, cutlass.Int32),
                    utils.scalar_to_ssa(logical_head, cutlass.Int32),
                    utils.scalar_to_ssa(left_for_predicate, cutlass.Int32),
                    utils.scalar_to_ssa(right_for_predicate, cutlass.Int32),
                    self.domain_info,
                    aux_tensors,
                )
                keep = cutlass.Boolean(utils.ssa_to_scalar(selected))
                if const_expr(restrict_bounds):
                    keep = keep and logical_left < self.left_extent and global_right < self.right_extent
                if const_expr(causal):
                    keep = keep and global_right <= logical_left + self.right_extent - self.left_extent
                if const_expr(local):
                    if const_expr(self.window_left is not None):
                        keep = (
                            keep
                            and global_right >= logical_left + self.right_extent - self.left_extent - self.window_left
                        )
                    if const_expr(self.window_right is not None):
                        keep = (
                            keep
                            and global_right <= logical_left + self.right_extent - self.left_extent + self.window_right
                        )
                values_mn[row, column] = values_mn[row, column] if keep else -Float32.inf

    @cute.jit
    def _apply_window(
        self,
        values_mn,
        thread_coordinates,
        reference_coordinates,
        left_block,
        right_block,
        thread_column_offset,
        restrict_bounds,
    ) -> None:
        for row in cutlass.range(cute.size(thread_coordinates.shape[0]), unroll_full=True):
            left = thread_coordinates[row, 0][0] + left_block * self.tile_left
            for column in cutlass.range(cute.size(thread_coordinates.shape[1]), unroll_full=True):
                right = reference_coordinates[0, column][1] + thread_column_offset + right_block * self.tile_right
                keep = True
                if const_expr(restrict_bounds):
                    keep = left < self.left_extent and right < self.right_extent
                if const_expr(self.window_left is not None):
                    keep = keep and right >= left + self.right_extent - self.left_extent - self.window_left
                if const_expr(self.window_right is not None):
                    keep = keep and right <= left + self.right_extent - self.left_extent + self.window_right
                values_mn[row, column] = values_mn[row, column] if keep else -Float32.inf
