# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# pyrefly: ignore-errors

"""Native SM100 FA4 backward with the banded Inkling relative-position bias applied inline.

Imported only by the SM100 backward launcher, after the optional CUDA-only modules are available.

The upstream ``FlashAttentionBackwardSm100`` ``score_mod``/``score_mod_bwd`` hooks run per element through
generic callbacks; these overrides read and write the banded bias (``levanter.grug.attention._inkling_relpos``)
directly. ``aux_tensors`` is ``(lower_bounds, valid, q_offset, band, d_band)`` with both band tensors in kernel
layout ``[S, W, Hq, B]``. The backward S tile is ``K @ Q.T``, so a thread holds one key row and a warp's 32 lanes
touch 32 adjacent band columns of one query row per access. Non-None ``score_mod``/``score_mod_bwd`` sentinels
select the upstream score-mod path, whose softmax expects natural-unit logits ``S * softmax_scale + bias``.
"""

import cutlass
import cutlass.cute as cute
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.utils import AuxData

from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK


class InklingFlashAttentionBackwardSm100(FlashAttentionBackwardSm100):
    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_copy_t2r,
        thr_mma_S,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        band = aux_data.tensors[3]
        rel_extent = band.shape[1] - REL_BIAS_BLOCK
        # The band can exceed 2**31 elements: offset its (batch, head) plane in Int64.
        band_bh = band[None, None, cutlass.Int64(head_idx), cutlass.Int64(batch_idx)]
        q0 = m_block * self.tile_m // REL_BIAS_BLOCK * REL_BIAS_BLOCK
        n0 = n_block * self.tile_n
        for i in cutlass.range_constexpr(cute.size(tSrS_t2r)):
            tSrS_t2r[i] = tSrS_t2r[i] * softmax_scale
        # Key tiles in the window [q0 - rel_extent, q0 + REL_BIAS_BLOCK) of this query block read the band.
        if cute.elem_less(q0 - rel_extent - 1, n0) and cute.elem_less(n0, q0 + REL_BIAS_BLOCK):
            cS = cute.domain_offset((n0, m_block * self.tile_m), cute.make_identity_tensor((self.tile_n, self.tile_m)))
            tScS_idx = thr_copy_t2r.partition_D(thr_mma_S.partition_C(cS))
            column_offset = rel_extent - q0
            for i in cutlass.range_constexpr(cute.size(tSrS_t2r)):
                key_idx = tScS_idx[i][0]
                query_idx = tScS_idx[i][1]
                tSrS_t2r[i] = tSrS_t2r[i] + band_bh[query_idx, key_idx + column_offset].to(cutlass.Float32)

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor,
        score_tensor,
        index_tensor,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen_info,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        # The bias is additive, so d(scaled score) = dS (grad_tensor stays as is) and dBand = dS at the band
        # positions apply_score_mod read. Each (query, key) owns one band column, so the writes never race.
        # A thread's elements share one key and (tile_m == REL_BIAS_BLOCK) one query block, hence one column.
        d_band = aux_data.tensors[4]
        width = d_band.shape[1]
        column = index_tensor[0][0] - index_tensor[0][1] // REL_BIAS_BLOCK * REL_BIAS_BLOCK + width - REL_BIAS_BLOCK
        if cute.elem_less(cutlass.Int32(-1), column) and cute.elem_less(column, width):
            d_band_bh = d_band[None, None, cutlass.Int64(head_idx), cutlass.Int64(batch_idx)]
            for i in cutlass.range_constexpr(cute.size(grad_tensor)):
                d_band_bh[index_tensor[i][1], column] = grad_tensor[i].to(d_band.element_type)
