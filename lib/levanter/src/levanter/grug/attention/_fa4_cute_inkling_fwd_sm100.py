# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# pyrefly: ignore-errors

"""Native SM100 FA4 forward with the banded Inkling relative-position bias applied inline.

Imported only by the SM100 forward launcher, after the optional CUDA-only modules are available.

``aux_tensors`` is ``(lower_bounds, valid, q_offset, band)`` with the band in kernel layout ``[S, W, Hq, B]``.
The forward S tile is ``Q @ K.T`` read from TMEM with 32x32b loads, so a thread holds one query row and its
``tile_n`` consecutive key columns; the in-band slice of that row is one contiguous run of ``tile_n`` band
entries, read with 128-bit loads.

``softmax_step`` is upstream flash-attn-4 4.0.0b28's (SM100 path; the SM103 ``ld.red`` row-max branch is dropped)
with the score-mod call replaced by the band:

* ``score_mod`` stays ``None``, so the softmax keeps the upstream folded scale (``exp2(S * scale * log2 e)``)
  and tiles outside the band do no extra work; in-band tiles add ``band / scale`` to the raw scores.
* The band row is held as packed bf16 pairs and added chunk by chunk as S comes out of TMEM, which keeps the
  softmax warps inside their 192-register budget. Loading it before the S wait (to hide its latency) spills and
  measured slower at every depth tried.
"""

import math
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.utils import AuxData

from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK

_LOG2_E = math.log2(math.e)
# bf16 band columns per 128-bit load.
_RUN = 8


class InklingFlashAttentionForwardSm100(FlashAttentionForwardSm100):
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
        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        if in_band:
            # The softmax scales raw scores by scale * log2(e); adding band / scale yields S * scale + band.
            inv_scale = Float32(_LOG2_E) / softmax.scale_log2
            rBand = self._load_band_row(band, band_tile, thr_tmem_load, tScS, batch_idx, head_idx, m_block)
            # Read S in its TMEM-load chunks and add each band chunk as it lands, so the band registers free up
            # while the S row fills (128 + 64 live registers would exceed the softmax warps' budget).
            num_chunks = cute.size(tStS_t2r.shape[2])
            chunk = self.n_block_size // num_chunks
            bits = cute.make_rmem_tensor((2,), cutlass.Uint32)
            bits_f32 = cute.make_tensor(cute.recast_ptr(bits.iterator, dtype=Float32), cute.make_layout(2))
            for j in cutlass.range_constexpr(num_chunks):
                cute.copy(thr_tmem_load, tStS_t2r[None, None, j], tSrS_t2r[None, None, j])
                # Element i of the thread's fragment is key column n0 + i (the order apply_mask_sm100 relies on).
                for p in cutlass.range_constexpr(j * chunk // 2, (j + 1) * chunk // 2):
                    # bf16 -> f32 is exact: the bf16 bits are the f32's high half (low half of the pair first).
                    bits[0] = rBand[p] << 16
                    bits[1] = rBand[p] & cutlass.Uint32(0xFFFF0000)
                    tSrS_t2r[2 * p], tSrS_t2r[2 * p + 1] = cute.arch.fma_packed_f32x2(
                        (bits_f32[0], bits_f32[1]), (inv_scale, inv_scale), (tSrS_t2r[2 * p], tSrS_t2r[2 * p + 1])
                    )
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

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
    def _load_band_row(self, band, band_tile, thr_tmem_load, tScS, batch_idx, head_idx, m_block):
        """This thread's query row of the in-band key tile, as packed bf16 pairs (``tile_n / 2`` x u32).

        A bf16 register tensor would take one 32-bit register per element and spill next to the fp32 S row.
        """
        query_idx = cutlass.min(
            m_block * self.m_block_size + thr_tmem_load.partition_D(tScS)[0][0], cute.size(band, mode=[0]) - 1
        )
        # The band can exceed 2**31 elements: offset its (batch, head) plane in Int64.
        band_row = band[None, None, cutlass.Int64(head_idx), cutlass.Int64(batch_idx)][query_idx, None]
        rBand = cute.make_rmem_tensor((self.n_block_size // 2,), cutlass.Uint32)
        runs_per_tile = self.n_block_size // _RUN
        for c in cutlass.range_constexpr(runs_per_tile):
            run = cute.local_tile(band_row, (_RUN,), (band_tile * runs_per_tile + c,))
            cute.autovec_copy(
                cute.make_tensor(cute.recast_ptr(run.iterator, dtype=cutlass.Uint32), cute.make_layout(_RUN // 2)),
                cute.local_tile(rBand, (_RUN // 2,), (c,)),
            )
        return rBand
