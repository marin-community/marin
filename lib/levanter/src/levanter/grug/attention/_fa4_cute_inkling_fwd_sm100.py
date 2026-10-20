# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# pyrefly: ignore-errors

"""Native SM100 FA4 forward with the banded Inkling relative-position bias applied inline.

Imported only by the SM100 forward launcher, after the optional CUDA-only modules are available.

``aux_tensors`` is ``(lower_bounds, valid, q_offset, band)`` with the band viewed as ``[S, W, Hq, B]`` over its tiled physical
layout (``_inkling_relpos.band_to_tiles``). The forward S tile is ``Q @ K.T`` read from TMEM with 32x32b loads,
so a thread holds one query row and its ``tile_n`` consecutive key columns; in the tiled layout a warp's rows
of one 8-column run are contiguous, so the band slice is read with coalesced 128-bit loads.

``softmax_step`` is upstream flash-attn-4 4.0.0b28's with the score-mod call replaced by the band:

* The band slice is loaded into registers *before* the softmax warps wait for the S tile, so its global-memory
  latency overlaps the QK^T MMA instead of stalling the softmax critical path.
* ``score_mod`` stays ``None``, so the softmax keeps the upstream folded scale (``exp2(S * scale * log2 e)``)
  and tiles outside the band do no extra work; in-band tiles add ``band / scale`` to the raw scores.
"""

import math
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.utils import AuxData

from levanter.grug.attention._inkling_relpos import BAND_TILE_COLUMNS, REL_BIAS_BLOCK

_LOG2_E = math.log2(math.e)


class InklingFlashAttentionForwardSm100(FlashAttentionForwardSm100):
    # Diagnosis switches (temporary).
    PREFETCH_BAND = True
    LOAD_BAND = True

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o,
        pipeline_p_lastsplit,
        pipeline_sm_stats,
        sm_stats_barrier,
        pipeline_s0_s1_sequence,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Inkling band: every row of a 128-row query tile shares the key window [q0 - rel_extent, q0 + 128), so
        # key tile n_block maps to band column tile (n0 - q0 + rel_extent) / tile_n. The launcher checks that
        # tile_m == REL_BIAS_BLOCK and that tile_n divides both REL_BIAS_BLOCK and rel_extent.
        band = aux_data.tensors[3]
        width = cute.size(band, mode=[1])
        rel_extent = width - REL_BIAS_BLOCK
        band_tile = n_block - m_block * (self.m_block_size // self.n_block_size) + rel_extent // self.n_block_size
        in_band = cute.elem_less(Int32(-1), band_tile) and cute.elem_less(band_tile, width // self.n_block_size)
        rBand = cute.make_rmem_tensor((self.n_block_size,), band.element_type)
        if const_expr(self.PREFETCH_BAND):
            self._load_band(rBand, band, in_band, band_tile, thr_tmem_load, tScS, batch_idx, head_idx, m_block)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(not self.PREFETCH_BAND):
            self._load_band(rBand, band, in_band, band_tile, thr_tmem_load, tScS, batch_idx, head_idx, m_block)
        if in_band:
            # The softmax scales raw scores by scale * log2(e); adding band / scale yields S * scale + band.
            inv_scale = Float32(_LOG2_E) / softmax.scale_log2
            # Element i of the thread's fragment is key column n0 + i (the order apply_mask_sm100 relies on).
            for i in cutlass.range_constexpr(0, self.n_block_size, 2):
                tSrS_t2r[i], tSrS_t2r[i + 1] = cute.arch.fma_packed_f32x2(
                    (rBand[i].to(Float32), rBand[i + 1].to(Float32)),
                    (inv_scale, inv_scale),
                    (tSrS_t2r[i], tSrS_t2r[i + 1]),
                )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def _load_band(self, rBand, band, in_band, band_tile, thr_tmem_load, tScS, batch_idx, head_idx, m_block):
        """Load this thread's query row of the in-band key tile into ``rBand`` (left untouched off band)."""
        if const_expr(self.LOAD_BAND):
            if in_band:
                query_idx = cutlass.min(
                    m_block * self.m_block_size + thr_tmem_load.partition_D(tScS)[0][0],
                    cute.size(band, mode=[0]) - 1,
                )
                # The band can exceed 2**31 elements: offset its (batch, head) plane in Int64.
                band_bh = band[None, None, cutlass.Int64(head_idx), cutlass.Int64(batch_idx)]
                # Column modes are (8-column run, run index): one run is 16 contiguous bytes, and the 32 lanes'
                # rows of a run are adjacent (the tiled layout), so each load is one coalesced 512-byte access.
                runs_per_tile = self.n_block_size // BAND_TILE_COLUMNS
                for c in cutlass.range_constexpr(runs_per_tile):
                    cute.autovec_copy(
                        band_bh[query_idx, (None, band_tile * runs_per_tile + c)],
                        cute.local_tile(rBand, (BAND_TILE_COLUMNS,), (c,)),
                    )
        else:
            rBand.fill(0.0)
