# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# pyrefly: ignore-errors

"""Native SM100 FA4 forward with the banded Inkling relative-position bias applied inline.

Imported only by the SM100 forward launcher, after the optional CUDA-only modules are available.

The upstream ``FlashAttentionForwardSm100.apply_score_mod`` hook runs a generic per-element callback; this
override adds the banded bias (``levanter.grug.attention._inkling_relpos``) directly. ``aux_tensors`` is
``(lower_bounds, valid, q_offset, band)`` with the band in kernel layout ``[S, W, Hq, B]``. The forward S tile is
``Q @ K.T`` read from TMEM with 32x32b loads, so a thread holds one query row and its ``tile_n`` consecutive key
columns: the in-band slice of that row is one contiguous run of ``tile_n`` band entries, read with 128-bit loads.
A non-None ``score_mod`` sentinel selects the upstream score-mod path, whose softmax expects natural-unit logits
``S * softmax_scale + bias``.
"""

import cutlass
import cutlass.cute as cute
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.utils import AuxData

from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK

# bf16 elements per 128-bit load.
_BAND_VECTOR = 8


class InklingFlashAttentionForwardSm100(FlashAttentionForwardSm100):
    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        band = aux_data.tensors[3]
        width = band.shape[1]
        rel_extent = width - REL_BIAS_BLOCK
        softmax_scale = softmax.softmax_scale
        # Every row of a 128-row query tile shares the window [q0 - rel_extent, q0 + REL_BIAS_BLOCK), so key
        # tile n_block maps to band column tile (n0 - q0 + rel_extent) / tile_n; the launcher checks that
        # tile_m == REL_BIAS_BLOCK and that tile_n divides both REL_BIAS_BLOCK and rel_extent.
        band_tile = n_block - m_block * (self.m_block_size // self.n_block_size) + rel_extent // self.n_block_size
        if cute.elem_less(cutlass.Int32(-1), band_tile) and cute.elem_less(band_tile, width // self.n_block_size):
            cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
            cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
            tScS_t2r = thr_tmem_load.partition_D(thr_mma_qk.partition_C(cS)[(None, None), 0, 0])
            query_idx = cutlass.min(tScS_t2r[0][0], band.shape[0] - 1)
            # The band can exceed 2**31 elements: offset its (batch, head) plane in Int64.
            band_row = band[None, None, cutlass.Int64(head_idx), cutlass.Int64(batch_idx)][query_idx, None]
            band_cols = cute.local_tile(band_row, (self.n_block_size,), (band_tile,))
            # Element i of the thread's fragment is key column n0 + i (the order apply_mask_sm100 relies on).
            for c in cutlass.range_constexpr(self.n_block_size // _BAND_VECTOR):
                rBand = cute.make_rmem_tensor((_BAND_VECTOR,), band.element_type)
                cute.autovec_copy(cute.local_tile(band_cols, (_BAND_VECTOR,), (c,)), rBand)
                for j in cutlass.range_constexpr(_BAND_VECTOR):
                    i = c * _BAND_VECTOR + j
                    tSrS_t2r[i] = tSrS_t2r[i] * softmax_scale + rBand[j].to(cutlass.Float32)
        else:
            for i in cutlass.range_constexpr(cute.size(tSrS_t2r)):
                tSrS_t2r[i] = tSrS_t2r[i] * softmax_scale
