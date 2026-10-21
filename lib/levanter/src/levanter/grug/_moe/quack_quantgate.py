# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# The ``kernel`` method below is forked from QuACK 0.6.4 ``quack/gemm_sm100.py`` (``GemmSm100.kernel``),
# Copyright (c) 2025-2026, QuACK team, BSD-3-Clause; edits are marked ``# quantgate:``.

"""MXFP8 varlen-k weight-gradient GEMM that quantizes its own operands along the token axis (Dist-MoE style).

Today the expert weight gradient ``dW_up = x^T d_pre`` runs two standalone passes before the GEMM: ``x`` (bf16) and
``d_pre`` (e4m3 rows with per-row exponents) are quantized along the token axis per expert (``quantize_token_blocks``
/ ``requantize_row_blocks``) into HBM, then QuACK's block-scaled varlen-k GEMM TMA-loads them. Here one persistent
launch does both: each CTA gets a third warp group (four "quant" warps) that quantizes the operands ONCE, writing the
same e4m3 values and the blocked e8m0 scale layout the GEMM reads (``blocked_scales(exponents.T[None])``) to HBM, and
release-increments a per-scale-tile readiness counter. The AB-load warp acquires the counter of every k-tile before
issuing its TMA loads, as Dist-MoE's ``dispatch_quant`` producers gate their consumers. Quantization runs under the
MMA of tiles whose k-tiles are ready instead of before the GEMM.

Work order: quant items are (scale tile ``g``, 128-column chunk of A or B), ``g``-major, strided over every quant warp
of the grid, so the ready front sweeps the token axis in the order the GEMM's waves consume it. Scale tile ``g``
holds four 32-token blocks of one expert (``token_scale_blocks``: group ``e`` starts at tile ``cu[e] // 128 + e``),
which is exactly the GEMM's k-tile ``g - cu[e] // 128 - e`` of expert ``e``. Rounding matches
``mx_quantize_row_blocks`` bit for bit (integer amax on the magnitude bits, ``ceil(log2(amax / 448))`` exponent,
power-of-two f32 scaling, ``cvt.rn.satfinite.e4m3x2.f32``).
"""

from typing import Optional, Union, Callable
from functools import partial


import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.cute.experimental import iket


from quack.tile_scheduler import (
    ag_wait_m_tile,
)
from quack.varlen_utils import VarlenManager
from quack.gemm_base import NamedBarrierGemm
from quack.gemm_config import SplitKMode
import quack.copy_utils as copy_utils


from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

import cutlass.jax as cjax
from cutlass import Uint8, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.arith import bitcast as _bitcast
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.cute_dsl_utils import mlir_namedtuple
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args
from quack.rounding import RoundingMode

from levanter.cutlass_kernel_cache import cute_launcher_factory
from levanter.grug.mxfp8 import MX_BLOCK
from levanter.grug._moe.quack_moe_cute import _ACC, _QUACK_USE_PDL, _max_active_clusters
from levanter.grug._moe.quack_mxfp8 import (
    MXFP8_WGRAD_CLUSTER,
    MXFP8_WGRAD_TILE_MN,
    _INDEX_SPEC,
    _SCALE_SPEC,
    _TOKEN_MAJOR,
    _WGRAD_OUT_SPEC,
    blocked_scales,
    padded_scale_tiles,
)

# Quant warps -> (epilogue, MMA/AB-load/C-load/scheduler, quant) registers per thread; the warp groups sum to 512.
_REGS = {4: (256, 120, 128), 8: (168, 120, 112)}
_TILE_TOKENS = 128
_COL_CHUNK = 128
_SF_ATOM_BYTES = 512
_PASS2_TOKENS = 8  # tokens per load group in the scaling pass


def _f32_bits(x) -> Uint32:
    return Uint32(_bitcast(Float32(x).ir_value(), T.i32()))


def _bits_f32(x) -> Float32:
    return Float32(_bitcast(Uint32(x).ir_value(), T.f32()))


@dsl_user_op
def _cvt_e4m3x2(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Uint16:
    """``lo`` -> byte 0, ``hi`` -> byte 1, round to nearest even, saturating (as ``astype(float8_e4m3fn)``)."""
    return Uint16(
        llvm.inline_asm(
            T.i16(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def _e4m3x4_to_f32(word: Uint32, *, loc=None, ip=None):
    """The four e4m3 bytes of ``word`` (byte 0 first) as f32 (exact)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\t.reg .b32 hlo, hhi;\n\t.reg .f16 a, b, c, d;\n\t"
        "mov.b32 {lo, hi}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 hlo, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 hhi, hi;\n\t"
        "mov.b32 {a, b}, hlo;\n\t"
        "mov.b32 {c, d}, hhi;\n\t"
        "cvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\tcvt.f32.f16 $2, c;\n\tcvt.f32.f16 $3, d;\n\t}",
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@cute.jit
def _token_exponent(amax_bits: Uint32) -> Uint32:
    """e8m0 exponent ``ceil(log2(amax / 448))`` from ``amax``'s f32 bits, clipped to [1, 254], 127 for zero
    (``mxfp8_pallas._exponent``: ``amax = m 2^E``, ``m`` in [1, 2), gives ``E - 8``, plus one when ``m > 1.75``)."""
    e = Int32((amax_bits >> 23) & 0xFF) - 8
    e = e + Int32(cutlass.select_((amax_bits & 0x7FFFFF) > 0x600000, Int32(1), Int32(0)))
    e = Int32(cutlass.select_(e < 1, Int32(1), e))
    e = Int32(cutlass.select_(e > 254, Int32(254), e))
    e = Int32(cutlass.select_(amax_bits == 0, Int32(127), e))
    return Uint32(e)


@cute.jit
def _umax(a: Uint32, b: Uint32) -> Uint32:
    return Uint32(cutlass.select_(a > b, a, b))


@cute.jit
def _sf_offset(chunk: Int32, tile: Int32, tiles: Int32, col: Int32, block: int) -> Int32:
    """Byte offset of (column ``chunk * 128 + col``, token block ``4 * tile + block``) in the blocked
    ``(1, C / 128, tiles, 32, 4, 4)`` scale layout."""
    return (chunk * tiles + tile) * _SF_ATOM_BYTES + (col % 32) * 16 + (col // 32) * 4 + block


@cute.jit
def _wait_ready(count: cute.Tensor, index: Int32, need: Int32):
    """Spin (lane 0) until ``count[index] >= need``, then order the TMA (async proxy) reads after it.

    Relaxed gpu-scope loads, as QuACK's ``ag_wait_m_tile``: an acquire load lowers to LDG.STRONG + CCTL.IVALL, a
    full L1 invalidate of this SM per poll, which evicts the quant warps' L1-resident rows. The gated bytes are read
    only by TMA, which fetches at L2, where the producers' release-ordered stores have landed when the counter is
    observed."""
    if cute.arch.lane_idx() == 0:
        ptr = (count.iterator + index).llvm_ptr
        val = cute.arch.load(ptr, Int32, sem="relaxed", scope="gpu")
        while val < need:
            cute.arch.inline_ptx("nanosleep.u32 64;", read_only_args=[])
            val = cute.arch.load(ptr, Int32, sem="relaxed", scope="gpu")
    cute.arch.sync_warp()
    cute.arch.inline_ptx("fence.proxy.async.global;", read_only_args=[])


@cute.jit
def _take_ticket(ticket, claim: int) -> Int32:
    """One relaxed atomic add of ``claim`` by lane 0, broadcast to the warp."""
    value = Int32(0)
    if cute.arch.lane_idx() == 0:
        value = cute.arch.atomic_add(ticket, Int32(claim))
    return cute.arch.shuffle_sync(value, 0)


@dsl_user_op
def _prefetch_l2(ptr, size: int, *, loc=None, ip=None):
    addr = cutlass.Int64(llvm.ptrtoint(T.i64(), ptr.llvm_ptr, loc=loc, ip=ip))
    cute.arch.inline_ptx(f"cp.async.bulk.prefetch.L2.global [{{$r0}}], {size};", read_only_args=[addr], loc=loc, ip=ip)


class QuantGateGemmSm100(GemmDefaultSm100):
    """``GemmSm100`` (default epilogue) plus a quant warp group feeding its own varlen-k A/B operands; see the
    module docstring. Only the block-scaled varlen-k, non-gather configuration is supported."""

    _extra_param_fields = GemmDefaultSm100._extra_param_fields + (
        ("q_x", Optional[cute.Tensor], None),
        ("q_dv", Optional[cute.Tensor], None),
        ("q_de", Optional[cute.Tensor], None),
        ("q_a", Optional[cute.Tensor], None),
        ("q_sfa", Optional[cute.Tensor], None),
        ("q_b", Optional[cute.Tensor], None),
        ("q_sfb", Optional[cute.Tensor], None),
        ("q_count", Optional[cute.Tensor], None),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        split_k_semaphore: Optional[cute.Tensor] = None
        split_k_workspace: Optional[cute.Tensor] = None
        mSFD: Optional[cute.Tensor] = None
        sfd_norm_const: Optional[Float32 | cute.Tensor] = None
        mSFDCol: Optional[cute.Tensor] = None
        # quantgate: bf16 x as u32 pairs [T, H/2]; d_pre e4m3 as u32 quads [T, I/4] and its row exponents
        # [T, I/32]; u16/u32 views of the A/B e4m3 operands the GEMM reads, flat u8 views of their blocked
        # scales, and the per-scale-tile readiness counters [tiles] (zero on entry).
        q_x: Optional[cute.Tensor] = None
        q_dv: Optional[cute.Tensor] = None
        q_de: Optional[cute.Tensor] = None
        q_a: Optional[cute.Tensor] = None
        q_sfa: Optional[cute.Tensor] = None
        q_b: Optional[cute.Tensor] = None
        q_sfb: Optional[cute.Tensor] = None
        q_count: Optional[cute.Tensor] = None

    def __init__(self, *args, quant_warps: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.gather_A, "quantgate GEMM does not support gather_A"
        assert self.blockscaled, "quantgate GEMM is block-scaled"
        self.quant_warp_id = self.scheduler_warp_id + 1
        self.threads_per_cta += quant_warps * cute.arch.WARP_SIZE
        self.reg_realloc = True
        self.num_regs_epi, self.num_regs_other, self.num_regs_quant = _REGS[quant_warps]
        self.gate_all_tiles = False
        self.quant_parts = "ab"
        self.prefetch_next = False
        self.claim = 4
        self.static_split = False
        self.num_quant_warps = quant_warps
        self.quant_debug = 0  # timing aid bitmask: 1 no async-proxy fence, 2 no A value stores

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        params = GemmDefaultEpiMixin.epi_to_underlying_arguments(self, args, loc=loc, ip=ip)
        return replace(
            params,
            q_x=args.q_x,
            q_dv=args.q_dv,
            q_de=args.q_de,
            q_a=args.q_a,
            q_sfa=args.q_sfa,
            q_b=args.q_b,
            q_sfb=args.q_sfb,
            q_count=args.q_count,
        )

    @cute.jit
    def _load_tma_gated(self, ab_pipeline, producer_state, copy_fns, k_tile_cnt, k_tile_start, params, cu, batch_idx):
        """``load_tma`` that acquires each k-tile's readiness counter before its TMA loads."""
        need_a = params.q_x.shape[1] // (_COL_CHUNK // 2)
        need_b = params.q_dv.shape[1] // (_COL_CHUNK // 4)
        if const_expr(self.quant_parts == "a"):
            need_b = 0
        if const_expr(self.quant_parts == "b"):
            need_a = 0
        need = Int32(need_a + need_b)
        gate0 = cu[batch_idx] // _TILE_TOKENS + batch_idx + k_tile_start
        peek_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_empty_status = ab_pipeline.producer_try_acquire(producer_state)
        if const_expr(self.gate_all_tiles):  # timing aid: hold every tile until the last scale tile is ready
            _wait_ready(params.q_count, params.q_count.shape[0] - 2, need)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            _wait_ready(params.q_count, gate0 + k_tile, need)
            ab_pipeline.producer_acquire(producer_state, peek_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(producer_state)
            smem_idx = producer_state.index
            for copy_fn in copy_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(k_tile_start + k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(producer_state)
            producer_state.advance()
            peek_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_empty_status = ab_pipeline.producer_try_acquire(producer_state)
        return producer_state

    @cute.jit
    def _item_rows(self, cu, groups, tile):
        """``(start, valid)``: the first token row of scale tile ``tile`` and its row count (0..128)."""
        group = Int32(0)
        for e in cutlass.range(1, groups, unroll=1):
            if cu[e] // _TILE_TOKENS + e <= tile:
                group = e
        start = cu[group] + _TILE_TOKENS * (tile - cu[group] // _TILE_TOKENS - group)
        valid = cu[group + 1] - start
        valid = Int32(cutlass.select_(valid < 0, Int32(0), valid))
        valid = Int32(cutlass.select_(valid > _TILE_TOKENS, Int32(_TILE_TOKENS), valid))
        return start, valid

    @cute.jit
    def _prefetch_item(self, params, cu, groups, item, items, items_a, lane):
        """Bulk-prefetch an item's rows (128 or 256 bytes each, one row per lane per step) into L2."""
        tile = item // items
        chunk = item - tile * items
        start, valid = self._item_rows(cu, groups, tile)
        for step in cutlass.range_constexpr(_TILE_TOKENS // 32):
            row = start + lane + 32 * step
            if lane + 32 * step < valid:
                if chunk < items_a:
                    ptr = params.q_x.iterator + row * params.q_x.shape[1] + chunk * (_COL_CHUNK // 2)
                    _prefetch_l2(ptr, 2 * _COL_CHUNK)
                else:
                    ptr = params.q_dv.iterator + row * params.q_dv.shape[1] + (chunk - items_a) * (_COL_CHUNK // 4)
                    _prefetch_l2(ptr, _COL_CHUNK)

    @cute.jit
    def _quant_loop(self, params, cu, quant_warp: Int32):
        """The quant warp group: (scale tile, column chunk) items from a global ticket (tile-major order, so the
        ready front sweeps the token axis; tickets, not a static split, so progress never depends on a CTA that
        is not resident), each followed by a release on its tile. The next item's rows are bulk-prefetched to L2
        while the current one is processed."""
        # PDL: the inputs and the zeroed counters come from earlier kernels.
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        lane = cute.arch.lane_idx()
        items_a = params.q_x.shape[1] // (_COL_CHUNK // 2)
        items_b = params.q_dv.shape[1] // (_COL_CHUNK // 4)
        if const_expr(self.quant_parts == "a"):  # timing aid: one operand only
            items_b = 0
        if const_expr(self.quant_parts == "b"):
            items_a = 0
        items = items_a + items_b
        tiles = params.q_count.shape[0] - 1
        total = tiles * items
        groups = cu.shape[0] - 1
        if const_expr(self.static_split):
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            first = (bx + gx * (by + gy * bz)) * self.num_quant_warps + quant_warp
            for item in cutlass.range(first, total, gx * gy * gz * self.num_quant_warps, unroll=1):
                self._quant_item(params, cu, groups, item, items, items_a, tiles, lane)
        else:
            self._quant_tickets(params, cu, groups, total, items, items_a, tiles, lane)

    @cute.jit
    def _quant_item(self, params, cu, groups, item, items, items_a, tiles, lane):
        tile = item // items
        chunk = item - tile * items
        start, valid = self._item_rows(cu, groups, tile)
        if chunk < items_a:
            self._quant_bf16_chunk(params, chunk, tile, tiles, start, valid, lane)
        else:
            self._requant_e4m3_chunk(params, chunk - items_a, tile, tiles, start, valid, lane)
        if const_expr(not (self.quant_debug & 1)):
            cute.arch.inline_ptx("fence.proxy.async.global;", read_only_args=[])
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.red(params.q_count.iterator + tile, Int32(1), op="add", dtype="s32", sem="release", scope="gpu")

    @cute.jit
    def _quant_tickets(self, params, cu, groups, total, items, items_a, tiles, lane):
        ticket = params.q_count.iterator + tiles
        # Claim several consecutive items per atomic.
        base = _take_ticket(ticket, self.claim)
        while base < total:
            for j in cutlass.range(self.claim, unroll=1):
                item = base + j
                if item < total:
                    self._quant_item(params, cu, groups, item, items, items_a, tiles, lane)
            base = _take_ticket(ticket, self.claim)

    @cute.jit
    def _quant_bf16_chunk(self, params, chunk, tile, tiles, start, valid, lane):
        """128 columns of bf16 ``x`` for one scale tile: lane owns columns 2l, 2l+1, 64+2l, 64+2l+1. Per 32-row
        block the 64 words of a lane stay in registers (all loads in flight at once)."""
        rows = params.q_x.shape[0]
        c0 = chunk * (_COL_CHUNK // 2) + lane
        for block in cutlass.range_constexpr(4):
            r0 = start + MX_BLOCK * block
            count = valid - MX_BLOCK * block
            # One in-bounds 32-row window per block (one base address, immediate row offsets); rows of the window
            # outside the block are masked to zero and never stored.
            win = Int32(cutlass.select_(r0 > rows - MX_BLOCK, rows - MX_BLOCK, r0))
            lo = r0 - win
            hi = lo + count
            w0 = []
            w1 = []
            for t in cutlass.range_constexpr(MX_BLOCK):
                ok = (t >= lo) & (t < hi)
                w0.append(Uint32(cutlass.select_(ok, Uint32(params.q_x[win + t, c0]), Uint32(0))))
                w1.append(Uint32(cutlass.select_(ok, Uint32(params.q_x[win + t, c0 + 32]), Uint32(0))))
            a0 = Uint32(0)
            a1 = Uint32(0)
            a2 = Uint32(0)
            a3 = Uint32(0)
            for t in cutlass.range_constexpr(MX_BLOCK):
                a0 = _umax(a0, (w0[t] << 16) & 0x7FFFFFFF)
                a1 = _umax(a1, w0[t] & 0x7FFF0000)
                a2 = _umax(a2, (w1[t] << 16) & 0x7FFFFFFF)
                a3 = _umax(a3, w1[t] & 0x7FFF0000)
            e0, e1, e2, e3 = _token_exponent(a0), _token_exponent(a1), _token_exponent(a2), _token_exponent(a3)
            s0 = _bits_f32((Uint32(254) - e0) << 23)
            s1 = _bits_f32((Uint32(254) - e1) << 23)
            s2 = _bits_f32((Uint32(254) - e2) << 23)
            s3 = _bits_f32((Uint32(254) - e3) << 23)
            if const_expr(not (self.quant_debug & 2)):
                for t in cutlass.range_constexpr(MX_BLOCK):
                    if (t >= lo) & (t < hi):
                        params.q_a[win + t, c0] = _cvt_e4m3x2(
                            _bits_f32(w0[t] & 0xFFFF0000) * s1, _bits_f32(w0[t] << 16) * s0
                        )
                        params.q_a[win + t, c0 + 32] = _cvt_e4m3x2(
                            _bits_f32(w1[t] & 0xFFFF0000) * s3, _bits_f32(w1[t] << 16) * s2
                        )
            col = 2 * lane
            params.q_sfa[_sf_offset(chunk, tile, tiles, col, block)] = Uint8(e0)
            params.q_sfa[_sf_offset(chunk, tile, tiles, col + 1, block)] = Uint8(e1)
            params.q_sfa[_sf_offset(chunk, tile, tiles, col + 64, block)] = Uint8(e2)
            params.q_sfa[_sf_offset(chunk, tile, tiles, col + 65, block)] = Uint8(e3)

    @cute.jit
    def _requant_e4m3_chunk(self, params, chunk, tile, tiles, start, valid, lane):
        """128 columns of row-quantized e4m3 ``d_pre`` for one scale tile: lane owns columns 4l..4l+3, all in
        row-scale block ``chunk * 4 + l // 8``. Dequantize (exact in f32), token amax, rescale. The 32 words and
        the 32 row exponents (packed four per register) of a block stay in registers. (Prefetching the next
        block into registers as well spills at the quant warps' 112 registers: 1.9 -> 3.6 ms.)"""
        rows = params.q_dv.shape[0]
        c0 = chunk * (_COL_CHUNK // 4) + lane
        sc = chunk * (_COL_CHUNK // MX_BLOCK) + lane // 8
        for block in cutlass.range_constexpr(4):
            r0 = start + MX_BLOCK * block
            count = valid - MX_BLOCK * block
            win = Int32(cutlass.select_(r0 > rows - MX_BLOCK, rows - MX_BLOCK, r0))
            lo = r0 - win
            hi = lo + count
            words = []
            packed = [Uint32(0)] * (MX_BLOCK // 4)
            for t in cutlass.range_constexpr(MX_BLOCK):
                ok = (t >= lo) & (t < hi)
                words.append(Uint32(cutlass.select_(ok, Uint32(params.q_dv[win + t, c0]), Uint32(0))))
                packed[t // 4] = packed[t // 4] | (Uint32(params.q_de[win + t, sc]) << (8 * (t % 4)))
            amax = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
            for t in cutlass.range_constexpr(MX_BLOCK):
                scale = _bits_f32(((packed[t // 4] >> (8 * (t % 4))) & 0xFF) << 23)
                v = _e4m3x4_to_f32(words[t])
                for i in cutlass.range_constexpr(4):
                    amax[i] = _umax(amax[i], _f32_bits(v[i] * scale) & 0x7FFFFFFF)
            exps = [_token_exponent(amax[i]) for i in range(4)]
            inv = [_bits_f32((Uint32(254) - exps[i]) << 23) for i in range(4)]
            for t in cutlass.range_constexpr(MX_BLOCK):
                if (t >= lo) & (t < hi):
                    scale = _bits_f32(((packed[t // 4] >> (8 * (t % 4))) & 0xFF) << 23)
                    v = _e4m3x4_to_f32(words[t])
                    even = _cvt_e4m3x2(v[1] * scale * inv[1], v[0] * scale * inv[0])
                    odd = _cvt_e4m3x2(v[3] * scale * inv[3], v[2] * scale * inv[2])
                    params.q_b[win + t, c0] = Uint32(even) | (Uint32(odd) << 16)
            for i in cutlass.range_constexpr(4):
                params.q_sfb[_sf_offset(chunk, tile, tiles, 4 * lane + i, block)] = Uint8(exps[i])

    # quantgate: forked from quack 0.6.4 GemmSm100.kernel (see the module header).
    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: Optional[cute.CopyAtom],
        mSFA_mkl: Optional[cute.Tensor],
        tma_atom_sfb: Optional[cute.CopyAtom],
        mSFB_chunks: Optional[cute.Tensor],
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        a_smem_load_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: Optional[cute.Layout],
        sfb_smem_layout: Optional[cute.Layout],
        epi_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_c_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        assert not (varlen_m and varlen_k)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch Tma desc
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                tma_atom_d,
                tma_atom_c,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates
        # Coords inside cluster
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        smem = cutlass.utils.SmemAllocator()
        storage = self.shared_storage.allocate(smem)

        # Initialize pipelines and states
        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cluster_layout_vmnk,
            is_leader_cta=is_leader_cta,
        )
        epi_pipeline = None
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(tx_count=self.epi_load_bytes_per_stage)
        acc_pipeline = self.make_acc_pipeline(cluster_layout_vmnk=cluster_layout_vmnk)
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(self.cluster_shape_mnk, has_C=has_epi_load)
            sched_data = storage.sched_data.get_tensor(cute.make_layout((4, self.sched_stage)))
        a_prefetch_pipeline = None
        if const_expr(self.gather_A):
            a_prefetch_pipeline = self.make_a_prefetch_pipeline()

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # Setup smem tensor A/B/D
        # (MMA, MMA_M, MMA_K, STAGE)
        sA_mma = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sA = storage.sA.get_tensor(a_smem_load_layout.outer, swizzle=a_smem_load_layout.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sAIdx = None
        if const_expr(self.gather_A):
            a_idx_smem_dim = self.cta_tile_shape_mnk[0] if varlen_m else self.cta_tile_shape_mnk[2]
            a_idx_smem_layout = cute.make_layout((a_idx_smem_dim, self.a_prefetch_stage))
            sAIdx = storage.sAIdx.get_tensor(a_idx_smem_layout)
        sSFA, sSFB, sSFB_chunks = None, None, None
        if const_expr(self.blockscaled):
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
            # Chunk view of the same bytes for the SFB TMA load: one 512-B SF
            # atom-chunk = 256 x Int16, k-chunks inner, atoms outer (the order
            # make_smem_layout_sfb produces; asserted against sfb_smem_layout
            # at TMA setup). Compact column-major.
            sSFB_chunks = cute.make_tensor(
                cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int16),
                cute.make_layout((256, self.sfb_chunks_per_ktile, self.sfb_window_atoms, self.ab_stage)),
            )
        sD = None
        if const_expr(has_D):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage if not self.overlap_accum_sf else 2)
        )

        varlen_manager = VarlenManager.create(
            varlen_params,
            # Only used if not varlen_m
            len_m_static=Int32(
                cute.size(mA_mkl, mode=[0])
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
            len_n_static=Int32(cute.size(mB_nkl, mode=[0])),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create,
            tile_sched_params,
            sched_data,
            sched_pipeline,
            throttle_barrier=self.clc_throttle_barrier,
        )

        epi_load_barrier = None
        if const_expr(has_epi_load):
            epi_load_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierGemm.EpilogueLoad),
                num_threads=(self.num_ab_load_warps + 1) * cute.arch.WARP_SIZE,
            )

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # quantgate: the quant warp group (its own warp group: one register budget for all four warps).
        if warp_idx >= self.quant_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_quant)
            self._quant_loop(epilogue_params, varlen_params.cu_seqlens_k, warp_idx - self.quant_warp_id)

        # Specialized AB load warps
        if warp_idx >= self.ab_load_warp_id and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps:
            # PDL: wait for prior kernel before any TMA loads (matches cutlass C++ main_load)
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            if const_expr(self.gather_A or self.reg_realloc):  # quantgate: + reg_realloc
                cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ag_last_gate = Int32(-1)  # 1-entry satisfied-gate cache (see ag_wait_m_tile)
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            a_prefetch_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.a_prefetch_stage
            )
            do_epi_load_barrier_arrive = Boolean(True)
            # CLC throttle producer: only the first load warp of CTA 0 in the
            # cluster signals; commit once per work tile started, or the scheduler
            # warp starves of credits.
            is_throttle_producer = Boolean(warp_idx == self.ab_load_warp_id)
            if const_expr(cute.size(cluster_layout_vmnk) > 1):
                is_throttle_producer = is_throttle_producer & Boolean(cute.arch.block_idx_in_cluster() == 0)
            while work_tile.is_valid_tile:
                tile_scheduler.throttle_producer_commit(is_throttle_producer)
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                # AllGather+GEMM: block until this tile's M-shard of A has been
                # pulled into local HBM by the copy stream. Only the load warp
                # gates — the MMA/epilogue warps are downstream of the AB
                # pipeline. With the ring-rotated shard-major schedule the flag
                # is normally already set and this is a single L2 load.
                # getattr: the varlen/triangular scheduler Params classes have
                # no ag_* fields at all (pre-existing trace error since the AG
                # gate landed; surfaced on any cold-cache varlen compile).
                if const_expr(getattr(tile_sched_params, "ag", None) is not None):
                    iket.range_push("ag_wait")
                    ag_last_gate = ag_wait_m_tile(
                        tile_sched_params,
                        tile_coord_mnkl[0],
                        self.cluster_shape_mnk[0],
                        ag_last_gate,
                    )
                    iket.range_pop()
                # Local_tile partition global tensors
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_coord_mnkl[1],
                    tile_coord_mnkl[3],
                )
                gA_mk = None
                if const_expr(not self.gather_A):
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    # (bM, bK, RestK)
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                # (bN, bK, RestK)
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.mma_tiler, [1, 2]),
                    (mma_tile_coord_mnl[1], None),
                )
                if const_expr(self.blockscaled):
                    # (bM, bK)
                    # SFA uses the tile-aligned per-batch offset (padded SF layout), not
                    # the A-data offset — allows varlen_m seqlens that aren't
                    # multiples of 128.
                    gSFA_mkl = cute.local_tile(
                        varlen_manager.offset_batch_SFA(mSFA_mkl, batch_idx),
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                    # (chunk, chunks-per-k-tile, window-atoms, RestK)
                    # SFB is chunk-granular: place the tile's SF window at atom
                    # coordinate j*tile_n/128, as a domain_offset because it is
                    # free-form, not a tiler multiple (tile_n=64 shares an atom
                    # between adjacent tiles, tile_n=192 advances 1.5 atoms per
                    # tile — both are just this one formula).
                    sfb_first_atom = (mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]) // 128
                    gSFB_chunks = cute.local_tile(
                        cute.domain_offset(
                            (None, None, sfb_first_atom),
                            varlen_manager.offset_batch_SFB(mSFB_chunks, batch_idx),
                        ),
                        (256, self.sfb_chunks_per_ktile, self.sfb_window_atoms),
                        (0, None, 0),
                    )

                # Partition global tensor for TiledMMA_A/B/D
                # Then partition global/shared tensor for TMA load A/B
                len_k = varlen_manager.len_k(batch_idx)
                # block_copy's lowering wants the coordinate held fixed by the
                # multicast mask: A/SFA are same-M across N peers, while B/SFB
                # are same-N across M peers. Degenerate cluster dimensions are
                # left for the compiler lowering to simplify.
                a_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "M",
                }
                b_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "N",
                }
                # SFB is duplicated (not V-split like B) across the 2-CTA MMA
                # pair, so unlike B its multicast group spans every cluster-M
                # CTA including the pair peer: use_2cta_mma_inst=False makes
                # the lowering slice/multicast across all of them (halving SFB
                # gmem traffic within a pair), while the op's cta_group still
                # aggregates transaction bytes at the pair leader's barrier.
                sfb_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "N",
                    "use_2cta_mma_inst": False,
                }
                copy_A, prefetch_A = None, None
                if const_expr(not self.gather_A):
                    # (MMA, MMA_M, MMA_K, RestK)
                    tCgA = thr_mma.partition_A(gA_mk)
                    copy_A = copy_utils.tma_get_block_copy_fn(
                        tma_atom_a, src_tensor=tCgA, dst_tensor=sA, tma_multicast=a_tma_multicast
                    )
                else:
                    # For varlen_m paths (TMA or cp.async): consume indices from
                    # a_prefetch_pipeline once per work tile.
                    sAIdx_stage = sAIdx
                    if const_expr(varlen_m):
                        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
                        # pyrefly: ignore[unsupported-operation]
                        sAIdx_stage = sAIdx[None, a_prefetch_consumer_state.index]
                    copy_A, prefetch_A = self._make_gather_A_copy(
                        mA_mkl,
                        sA,
                        sAIdx_stage,
                        tma_atom_a,
                        varlen_manager,
                        tile_coord_mnkl,
                        batch_idx,
                        warp_idx,
                    )
                    if const_expr(varlen_m):
                        a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
                        a_prefetch_consumer_state.advance()
                    if const_expr(prefetch_A is not None):
                        prefetch_A = partial(prefetch_A, a_prefetch_pipeline)
                # (MMA, MMA_N, MMA_K, RestK)
                tCgB = thr_mma.partition_B(gB_nk)
                if const_expr(self.blockscaled):
                    # (MMA, MMA_M, MMA_K)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)
                # TMA load B partition_S/D
                copy_B = copy_utils.tma_get_block_copy_fn(
                    tma_atom_b, src_tensor=tCgB, dst_tensor=sB, tma_multicast=b_tma_multicast
                )
                copy_SFA, copy_SFB = None, None
                if const_expr(self.blockscaled):
                    #  TMA load SFA partition_S/D
                    copy_SFA = copy_utils.tma_get_block_copy_fn(
                        tma_atom_sfa,
                        src_tensor=tCgSFA,
                        dst_tensor=sSFA,
                        tma_multicast=a_tma_multicast,
                    )
                    # SFB multicast: same-N across all cluster-M CTAs (see
                    # sfb_tma_multicast above).
                    copy_SFB = copy_utils.tma_get_block_copy_fn(
                        tma_atom_sfb,
                        src_tensor=gSFB_chunks,
                        dst_tensor=sSFB_chunks,
                        tma_multicast=sfb_tma_multicast,
                    )
                k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                k_tile_start, k_tile_cnt = tile_scheduler.get_split_k_tile_range(k_tile_total, split_idx)
                iket.range_push("tma_load")
                if const_expr(not self.gather_A):
                    # quantgate: acquire each k-tile's readiness counter before its TMA loads.
                    ab_producer_state = self._load_tma_gated(
                        ab_pipeline,
                        ab_producer_state,
                        [copy_A, copy_B, copy_SFA, copy_SFB],
                        k_tile_cnt,
                        k_tile_start,
                        epilogue_params,
                        varlen_params.cu_seqlens_k,
                        batch_idx,
                    )
                elif const_expr(self.use_tma_gather):
                    ab_producer_state, a_prefetch_consumer_state = self.load_AB_tma_gather(
                        ab_pipeline,
                        ab_producer_state,
                        a_prefetch_consumer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                    )
                else:
                    ab_producer_state, a_prefetch_consumer_state = self.load_AB_gather_A(
                        ab_pipeline,
                        ab_producer_state,
                        a_prefetch_consumer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                    )
                iket.range_pop()
                if const_expr(epi_load_barrier is not None):
                    # In the first work tile, the epi load warp will wait for the signal
                    # from the mainloop load warp to start loading C, to avoid interfering
                    # with loading A and B.
                    if do_epi_load_barrier_arrive:
                        epi_load_barrier.arrive()
                        do_epi_load_barrier_arrive = Boolean(False)
                # Advance to next tile
                iket.range_push("sched_fetch")
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                iket.range_pop()
            # Wait A/B buffer empty
            if warp_idx == self.ab_load_warp_id:
                ab_pipeline.producer_tail(ab_producer_state)

        # Specialized scheduler warp
        if const_expr(self.is_persistent or self.gather_A):
            if warp_idx == self.scheduler_warp_id:
                # PDL: wait for prior kernel before reading CLC state (matches cutlass C++ sched)
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                if const_expr(self.gather_A or self.reg_realloc):  # quantgate: + reg_realloc
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                is_scheduler_warp = True
                if const_expr(cute.size(cluster_layout_vmnk) > 1):
                    is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    # Advance to next tile
                    iket.range_push("clc_produce")
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    iket.range_pop()
                    iket.range_push("clc_consume")
                    work_tile = tile_scheduler.get_current_work()
                    iket.range_pop()
                    # End of persistent scheduler loop
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()
                    # Drain the pending padding tail with SERIAL-OBSERVED
                    # cancels (issue -> wait -> decode, one at a time), gated
                    # on a decoded-phantom retirement. The old fire-and-forget
                    # burst spray caused the July 2026 varlen corruption; see
                    # cancel_pending_tail and
                    # AI/clc_spurious_invalid_investigation.md.
                    tile_scheduler.cancel_pending_tail()

        # Specialized A-index prefetch warp (gather_A only)
        if const_expr(self.gather_A):
            if warp_idx == self.a_prefetch_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                tile_M = self.cta_tile_shape_mnk[0]
                tile_K = self.cta_tile_shape_mnk[2]
                tiled_copy_AIdx = copy_utils.tiled_copy_1d(Int32, num_threads=32, is_async=True)
                thr_copy_AIdx = tiled_copy_AIdx.get_slice(cute.arch.lane_idx())
                tAsAIdx = thr_copy_AIdx.partition_D(sAIdx)
                tAcAIdx = thr_copy_AIdx.partition_S(cute.make_identity_tensor(tile_M if varlen_m else tile_K))
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                a_prefetch_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.a_prefetch_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
                    if const_expr(varlen_m):
                        # (tile_M,)
                        gAIdx = cute.local_tile(mAIdx_mk, (tile_M,), (tile_coord_mnkl[0],))
                        tAgAIdx = thr_copy_AIdx.partition_S(gAIdx)
                        len_m = varlen_manager.len_m(batch_idx)
                        m_limit = len_m - tile_coord_mnkl[0] * tile_M
                        tApAIdx_m = cute.make_rmem_tensor((1, tAsAIdx.shape[1]), Boolean)
                        for m in cutlass.range(tAsAIdx.shape[1], unroll_full=True):
                            tApAIdx_m[0, m] = tAcAIdx[0, m] < m_limit
                        a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                        cute.copy(
                            thr_copy_AIdx,
                            tAgAIdx,
                            tAsAIdx[None, None, a_prefetch_producer_state.index],
                            pred=tApAIdx_m,
                        )
                        a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                        a_prefetch_producer_state.advance()
                    else:
                        # (tile_K, RestK)
                        gAIdx = cute.flat_divide(mAIdx_mk, (tile_K,))
                        tAgAIdx = thr_copy_AIdx.partition_S(gAIdx)
                        len_k = varlen_manager.len_k(batch_idx)
                        k_tile_cnt = cute.ceil_div(len_k, tile_K)
                        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
                            a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                            cute.copy(
                                thr_copy_AIdx,
                                tAgAIdx[None, None, k_tile],
                                tAsAIdx[None, None, a_prefetch_producer_state.index],
                            )
                            a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                            a_prefetch_producer_state.advance()
                        if 0 < k_tile_cnt:
                            k_tile = k_tile_cnt - 1
                            k_limit = len_k - k_tile * tile_K
                            tApAIdx_k = cute.make_rmem_tensor((1, tAsAIdx.shape[1]), Boolean)
                            for m in cutlass.range(tAsAIdx.shape[1], unroll_full=True):
                                tApAIdx_k[0, m] = tAcAIdx[0, m] < k_limit
                            a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                            cute.copy(
                                tiled_copy_AIdx,
                                tAgAIdx[None, None, k_tile],
                                tAsAIdx[None, None, a_prefetch_producer_state.index],
                                pred=tApAIdx_k,
                            )
                            a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                            a_prefetch_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()

        # Specialized TMA epi load warp
        if warp_idx == self.epi_load_warp_id:
            if const_expr(self.gather_A or self.reg_realloc):  # quantgate: + reg_realloc
                cute.arch.setmaxregister_decrease(self.num_regs_other)
            # PDL: wait for prior kernel before any C TMA loads (matches cutlass C++ epi_load)
            if const_expr(self.use_pdl and has_epi_load):
                cute.arch.griddepcontrol_wait()
            if const_expr(has_epi_load):
                epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_c_stage)
                do_epi_load_barrier_wait = Boolean(True)
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    # Get tile coord from tile scheduler
                    # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                    copy_C = None
                    if const_expr(has_C):
                        copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                            tma_atom_c,
                            varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                            self.cta_tile_shape_mnk[:2],
                            epi_tile,
                            sC,
                            tile_coord_mnkl,
                        )
                        copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                    tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                        epilogue_params,
                        epi_smem_tensors,
                        tile_coord_mnkl,
                        varlen_manager,
                        epi_pipeline,
                    )
                    copy_epi_load = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))
                    if do_epi_load_barrier_wait:
                        epi_load_barrier.arrive_and_wait()
                        do_epi_load_barrier_wait = Boolean(False)
                    epi_tile_shape = cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile).shape[
                        1
                    ]
                    epi_tile_num = const_expr(cute.size(epi_tile_shape))
                    # Hier subtile coords, ordered exactly as the epilogue store
                    # loop consumes stages (gemm_base.epilogue): every copy fn
                    # receives a subscriptable (epi_m, epi_n) coordinate — flat
                    # indices only match consumption order for m-major epilogues.
                    epi_load_layout = cute.make_ordered_layout(
                        epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
                    )
                    # Split-K (serial/parallel): only the finalizing split runs the
                    # epilogue, so only its tiles consume C — skip the loads (and the
                    # pipeline slots) for non-finalizing splits, symmetric with the
                    # epilogue warps' skip in epilogue_split_k. The const_expr prefix
                    # folds at trace time (quack.dsl.mixed_constexpr_if), so
                    # split_k == 1 codegen has no dynamic if at all.
                    if (
                        const_expr(self.split_k == 1 or self.split_k_mode == SplitKMode.SEPARATE)
                        or split_idx == self.split_k - 1
                    ):
                        for epi_idx in cutlass.range(epi_tile_num, unroll=1):
                            epi_pipeline.producer_acquire(epi_producer_state)
                            copy_epi_load(
                                src_idx=epi_load_layout.get_hier_coord(epi_idx),
                                producer_state=epi_producer_state,
                            )
                            # Epi pipeline's producer commit is a NOP
                            epi_pipeline.producer_commit(epi_producer_state)
                            epi_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                epi_pipeline.producer_tail(epi_producer_state)

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            if const_expr(self.gather_A or self.reg_realloc):  # quantgate: + reg_realloc
                cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Retrieving tensor memory ptr and make accumulator tensor
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Partition shared/tensor memory tensor for TiledMMA_A/B/D
            # (MMA, MMA_M, MMA_K, STAGE)
            tCrA = tiled_mma.make_fragment_A(sA_mma)
            # (MMA, MMA_N, MMA_K, STAGE)
            tCrB = tiled_mma.make_fragment_B(sB)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            if const_expr(self.blockscaled):
                # Make SFA tmem tensor
                acc_tmem_col_offset = const_expr(
                    tcgen05.find_tmem_tensor_col_offset(
                        tCtAcc_base if const_expr(not self.overlap_accum_sf) else tCtAcc_base[None, None, None, 0]
                    )
                )
                sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + acc_tmem_col_offset, dtype=self.sf_dtype)
                # (MMA, MMA_M, MMA_K)
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                # Make SFB tmem tensor
                sfa_tmem_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                sfb_tmem_col_offset = acc_tmem_col_offset + sfa_tmem_col_offset
                sfb_tmem_base_ptr = acc_tmem_ptr + sfb_tmem_col_offset
                sfb_tmem_ptr = cute.recast_ptr(sfb_tmem_base_ptr, dtype=self.sf_dtype)
                # (MMA, MMA_N, MMA_K)
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            else:
                tCtSFA, tCtSFB = None, None

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                k_len = varlen_manager.len_k(batch_idx)
                k_tile_total = cute.ceil_div(k_len, self.mma_tiler[2])
                _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(k_tile_total, split_idx)
                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                acc_stage_idx = (
                    acc_producer_state.phase ^ 1 if const_expr(self.overlap_accum_sf) else acc_producer_state.index
                )
                tCtAcc = tCtAcc_base[None, None, None, acc_stage_idx]
                tCtSFB_mma = tCtSFB
                if const_expr(self.blockscaled and self.sfb_n_atom_misaligned):
                    # Odd N-tiles start 64 into a 128-wide SF atom: shift the SFB
                    # tmem base by 2 columns (in the atom layout (32,4):(16,4),
                    # N+64 is element offset 8 = 2 tmem columns).
                    tCtSFB_mma = cute.make_tensor(
                        cute.recast_ptr(
                            sfb_tmem_base_ptr + Int32((tile_coord_mnkl[1] % 2) * 2),
                            dtype=self.sf_dtype,
                        ),
                        tCtSFB.layout,
                    )
                copy_s2t_sfa, copy_s2t_sfb = None, None
                sf_valid_insts = None
                if const_expr(self.blockscaled):
                    copy_s2t_sfa = copy_utils.s2t_get_copy_fn(sSFA, tCtSFA, self.cta_group)
                    copy_s2t_sfb = copy_utils.s2t_get_copy_fn(sSFB, tCtSFB, self.cta_group)
                    # Exploits the fact that for mxfp8 the MMA instruction K size
                    # equals the SF vec size (== 32), so one instruction consumes
                    # exactly one SF block and the mma loop can skip the
                    # instructions for SF pad blocks on a ragged-K last tile (see
                    # the comment in self.mma). fp4 has inst_k 64 spanning
                    # multiple SF blocks, but we don't do varlen_k for
                    # mxfp4/nvfp4. Valid instructions in that tile; % maps
                    # "aligned or full tile" to 0 = nothing to skip.
                    if const_expr(self.mma_inst_shape_mnk[2] == self.sf_vec_size):
                        num_insts = self.mma_tiler[2] // self.mma_inst_shape_mnk[2]
                        sf_valid_insts = cute.ceil_div(k_len % self.mma_tiler[2], self.sf_vec_size) % num_insts
                iket.range_push("mma")
                ab_consumer_state, acc_producer_state, tiled_mma = self.mma(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc,
                    k_tile_cnt,
                    is_leader_cta,
                    cta_rank_in_cluster,
                    tCtSFA,
                    tCtSFB_mma,
                    copy_s2t_sfa,
                    copy_s2t_sfb,
                    sf_valid_insts,
                )
                if const_expr(self.overlap_accum_sf):
                    # After iter 0, 2, ..., shift tmem ptr by -256.
                    # After iter 1, 3, ..., shift tmem ptr by 256.
                    tCtSFA, tCtSFB = [
                        cute.make_tensor(
                            cute.recast_ptr(
                                # Doing tmem ptr arithmetic requires 32-bit type, wrong otherwise
                                cute.recast_ptr(mT.iterator, dtype=Float32)
                                + cute.assume(
                                    acc_tmem_col_offset * (acc_producer_state.phase * 2 - 1),
                                    divby=acc_tmem_col_offset,
                                ),
                                dtype=self.sf_dtype,
                            ),
                            mT.layout,
                        )
                        for mT in [tCtSFA, tCtSFB]
                    ]
                iket.range_pop()
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # PDL: hint the next kernel to launch early now that all MMAs are issued
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            tmem_alloc_barrier.arrive()
            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            if const_expr(self.gather_A or self.reg_realloc):  # quantgate: + reg_realloc
                cute.arch.setmaxregister_increase(self.num_regs_epi)
            # Alloc tensor memory buffer
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            is_tma_warp = Boolean(warp_idx == self.epilog_warp_id[0])

            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_copy_t2r, self.d_layout, self.d_dtype, tTR_rD, sD, epi_tidx
            )
            tRS_rC, tSR_rC, tSR_sC = None, None, None
            tiled_copy_s2r = None
            if const_expr(mC_mnl is not None):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                    tiled_copy_t2r, self.c_layout, self.c_dtype, sC, tRS_rD.layout, epi_tidx
                )

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
            while work_tile.is_valid_tile:
                # Prefetch the next work tile before the epilogue: the response is
                # already in smem (3-stage sched pipeline), and consuming it here
                # hides the ~300ns decode (swizzle + async fence) behind this tile's
                # epilogue — the pacing chain for small-K / double-store epilogues.
                # advance_to_next_work stays after the body: num_tiles_executed must
                # count completed tiles during the body (sD stage cycling).
                next_work_tile = tile_scheduler.get_current_work()
                # Get tile coord from tile scheduler
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                epi_acc_stage = (
                    acc_consumer_state.index if const_expr(not self.overlap_accum_sf) else acc_consumer_state.phase
                )
                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, epi_acc_stage]
                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                copy_D = None
                if const_expr(has_D):
                    # Staged split-K: D is the f32 partials workspace, whose batch mode is the
                    # combined (l * split_k + split) index from the scheduler.
                    d_batch_idx = batch_idx
                    if const_expr(self.split_k > 1 and self.split_k_mode == SplitKMode.SEPARATE):
                        d_batch_idx = tile_scheduler.get_combined_batch_idx(batch_idx, split_idx)
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, d_batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )

                copy_C = None  # We're using a separate warp to load C

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                k_len = varlen_manager.len_k(batch_idx)
                epi_tile_num = cute.size(
                    cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile),
                    mode=[1],
                )
                clear_acc = varlen_k and k_len == 0
                if const_expr(self.split_k > 1):
                    # An empty split (split_k > total k-tiles) still runs the epilogue with a
                    # zero contribution so the serial-semaphore turnstile advances.
                    k_tile_total = cute.ceil_div(k_len, self.cta_tile_shape_mnk[2])
                    _, k_tile_cnt_split = tile_scheduler.get_split_k_tile_range(k_tile_total, split_idx)
                    clear_acc = k_tile_cnt_split == 0
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    acc_release_idx=(
                        self.iter_acc_early_release if const_expr(self.overlap_accum_sf) else epi_tile_num - 1
                    ),
                    clear_acc=clear_acc,
                )

                # Split-K (serial/parallel): non-finalizing splits commit raw f32 partials
                # to the tile's workspace and skip the epilogue; the last split waits for
                # the tile's completion flag and runs the full epilogue on the summed
                # accumulator (CUTLASS-3.x stream-K fixup semantics).
                iket.range_push("epilogue")
                epi_fn = partial(
                    self.epilogue,
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    None,  # epi_producer_state
                    epi_tile,
                    # load_acc_subtile is the one argument left unbound
                    tRS_rD=tRS_rD,
                    tRS_rC=tRS_rC,
                    tiled_copy_t2r=tiled_copy_t2r,
                    tiled_copy_r2s=tiled_copy_r2s,
                    tRS_sD=tRS_sD,
                    tiled_copy_s2r=tiled_copy_s2r,
                    tSR_rC=tSR_rC,
                    tSR_sC=tSR_sC,
                    copy_D=copy_D,
                    copy_C=copy_C,
                    tile_coord_mnkl=tile_coord_mnkl,
                    varlen_manager=varlen_manager,
                    epilogue_barrier=self.epilogue_barrier,
                    tile_scheduler=tile_scheduler,
                    tidx=epi_tidx,
                    is_tma_warp=is_tma_warp,
                )
                epi_read_state, _ = self.epilogue_split_k(
                    epilogue_params,
                    epi_fn,
                    load_acc_subtile,
                    tRS_rD,
                    epi_tile,
                    epi_read_state,
                    None,  # epi_producer_state
                    epi_store_pipeline,
                    tile_coord_mnkl,
                    self.epilogue_barrier,
                    epi_tidx,
                    is_tma_warp,
                )
                # acc_pipeline.consumer_release was already called in self.epi_load_acc_subtile
                acc_consumer_state.advance()
                iket.range_pop()

                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = next_work_tile

            # Wait for D store complete
            if is_tma_warp:
                epi_store_pipeline.producer_tail()

            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)


def _launch(
    stream,
    mX,
    mDv,
    mDe,
    mCu,
    mCount,
    mD,
    mA,
    mSFA,
    mB,
    mSFB,
    *,
    tile_mn,
    cluster_mnk,
    max_active_clusters,
    max_swizzle,
    gate_all_tiles,
    quant_parts,
    prefetch_next,
    quant_debug,
    quant_warps,
    claim,
    static_split,
    use_pdl,
    hidden,
    width,
):
    gemm = QuantGateGemmSm100(
        _ACC,
        cutlass.Float8E4M3FN,
        tile_mn,
        cluster_mnk,
        sf_vec_size=MX_BLOCK,
        gather_A=False,
        use_clc_persistence=False,
        use_pdl=use_pdl,
        quant_warps=quant_warps,
    )
    gemm.gate_all_tiles = gate_all_tiles
    gemm.quant_parts = quant_parts
    gemm.prefetch_next = prefetch_next
    gemm.quant_debug = quant_debug
    gemm.claim = claim
    gemm.static_split = static_split
    # Static row strides: the quant warps' 32 row loads then share one base address.
    rows = mDv.shape[0]
    epi_args = QuantGateGemmSm100.EpilogueArguments(
        q_x=cute.make_tensor(
            cute.recast_ptr(mX.iterator, dtype=Uint32), cute.make_layout((rows, hidden // 2), stride=(hidden // 2, 1))
        ),
        q_dv=cute.make_tensor(
            cute.recast_ptr(mDv.iterator, dtype=Uint32), cute.make_layout((rows, width // 4), stride=(width // 4, 1))
        ),
        q_de=cute.make_tensor(
            mDe.iterator, cute.make_layout((rows, width // MX_BLOCK), stride=(width // MX_BLOCK, 1))
        ),
        q_a=cute.make_tensor(
            cute.recast_ptr(mA.iterator, dtype=Uint16), cute.make_layout((rows, hidden // 2), stride=(hidden // 2, 1))
        ),
        q_sfa=cute.make_tensor(cute.recast_ptr(mSFA.iterator, dtype=Uint8), cute.make_layout(cute.size(mSFA))),
        q_b=cute.make_tensor(
            cute.recast_ptr(mB.iterator, dtype=Uint32), cute.make_layout((rows, width // 4), stride=(width // 4, 1))
        ),
        q_sfb=cute.make_tensor(cute.recast_ptr(mSFB.iterator, dtype=Uint8), cute.make_layout(cute.size(mSFB))),
        q_count=mCount,
    )
    scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle, None)
    varlen_args = make_varlen_args(None, mCu, None)
    gemm(mA, mB, mD, None, epi_args, scheduler_args, varlen_args, stream, mSFA, mSFB)


@cute_launcher_factory
def _build_quantgate_launcher(**config):
    @cute.jit
    def launcher(stream, mX, mDv, mDe, mCu, mCount, mD, mA, mSFA, mB, mSFB):
        _launch(stream, mX, mDv, mDe, mCu, mCount, mD, mA, mSFA, mB, mSFB, **config)

    return launcher


@cute_launcher_factory
def _build_quantgate_b_launcher(**config):
    """A (``x``'s tokens) and its scales come quantized as inputs aliased to the GEMM's A buffers."""

    @cute.jit
    def launcher(stream, mX, mDv, mDe, mCu, mCount, mA, mSFA, mD, mB, mSFB):
        _launch(stream, mX, mDv, mDe, mCu, mCount, mD, mA, mSFA, mB, mSFB, **config)

    return launcher


_ROWS = cjax.TensorSpec(divisibility=(1, 16), static=False)
_PLAIN = cjax.TensorSpec(static=False)


def mxfp8_quantgate_wgrad(
    x: jax.Array,
    d_pre_rows: tuple[jax.Array, jax.Array],
    cu_seqlens: jax.Array,
    *,
    x_tokens: tuple[jax.Array, jax.Array] | None = None,
    out_dtype=jnp.bfloat16,
    tile_mn=MXFP8_WGRAD_TILE_MN,
    cluster_mnk=MXFP8_WGRAD_CLUSTER,
    max_swizzle=8,
    gate_all_tiles=False,
    quant_parts="ab",
    prefetch_next=False,
    quant_debug=0,
    quant_warps=8,
    claim=4,
    static_split=True,
    use_pdl=_QUACK_USE_PDL,
):
    """Per-expert ``x^T d_pre`` (``[E, H, I]``) on the block-scaled varlen-k GEMM with the token-axis quantization
    inside the GEMM launch: ``x`` bf16 ``[T, H]``, ``d_pre_rows`` = (e4m3 ``[T, I]``, row exponents ``[T, I / 32]``).
    Equal to ``mxfp8_token_grouped_gemm(quantize_token_blocks(x, cu), requantize_row_blocks(*d_pre_rows,
    *token_scale_blocks(cu, T)), cu)`` bit for bit. With ``x_tokens`` (``quantize_token_blocks(x, cu)``) only
    ``d_pre`` is quantized in the launch (``x`` then only gives the width). Also returns the token-quantized
    operands ``(A e4m3 [T, H], A blocked scales, B e4m3 [T, I], B blocked scales, counters)`` (rows outside every
    block unspecified)."""
    values, exponents = d_pre_rows
    rows, width = values.shape
    hidden = x.shape[1]
    if hidden % _COL_CHUNK or width % _COL_CHUNK:
        raise ValueError(f"need 128-multiple widths, got x={x.shape}, d_pre={values.shape}")
    groups = cu_seqlens.shape[0] - 1
    tiles = padded_scale_tiles(rows, groups)
    config = dict(
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
        gate_all_tiles=gate_all_tiles,
        quant_parts=quant_parts if x_tokens is None else "b",
        prefetch_next=prefetch_next,
        quant_debug=quant_debug,
        quant_warps=quant_warps,
        claim=claim,
        static_split=static_split,
        use_pdl=use_pdl,
        hidden=hidden,
        width=width,
    )
    sf_shape = lambda cols: (1, cols // _COL_CHUNK, tiles, 32, 4, 4)  # noqa: E731
    d_struct = jax.ShapeDtypeStruct((groups, hidden, width), out_dtype)
    a_struct = (
        jax.ShapeDtypeStruct((rows, hidden), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct(sf_shape(hidden), jnp.float8_e8m0fnu),
    )
    b_struct = (
        jax.ShapeDtypeStruct((rows, width), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct(sf_shape(width), jnp.float8_e8m0fnu),
    )
    # Readiness counters per scale tile plus the quant warps' work ticket.
    count_struct = jax.ShapeDtypeStruct((tiles + 1,), jnp.int32)
    counts = jnp.zeros((tiles + 1,), jnp.int32)
    cu = cu_seqlens.astype(jnp.int32)
    # cutlass.jax directly: the memoized levanter wrapper cannot hash the alias dict.
    if x_tokens is None:
        call = cjax.cutlass_call(
            _build_quantgate_launcher(**config),
            output_shape_dtype=(d_struct, *a_struct, *b_struct, count_struct),
            input_spec=(_ROWS, _ROWS, _PLAIN, _INDEX_SPEC, _PLAIN),
            output_spec=(_WGRAD_OUT_SPEC, _TOKEN_MAJOR, _SCALE_SPEC, _TOKEN_MAJOR, _SCALE_SPEC, _PLAIN),
            input_output_aliases={4: 5},
            use_static_tensors=False,
        )
        dw, a, sfa, b, sfb, counts = call(x, values, exponents, cu, counts)
        return dw, (a, sfa, b, sfb, counts)
    a_values, a_exponents = x_tokens
    # Aliased outputs go last: cutlass.jax drops them from the output list by position.
    call = cjax.cutlass_call(
        _build_quantgate_b_launcher(**config),
        output_shape_dtype=(d_struct, *b_struct, *a_struct, count_struct),
        input_spec=(_ROWS, _ROWS, _PLAIN, _INDEX_SPEC, _PLAIN, _TOKEN_MAJOR, _SCALE_SPEC),
        output_spec=(_WGRAD_OUT_SPEC, _TOKEN_MAJOR, _SCALE_SPEC, _TOKEN_MAJOR, _SCALE_SPEC, _PLAIN),
        input_output_aliases={4: 5, 5: 3, 6: 4},
        use_static_tensors=False,
    )
    dw, b, sfb, a, sfa, counts = call(x, values, exponents, cu, counts, a_values, blocked_scales(a_exponents.T[None]))
    return dw, (a, sfa, b, sfb, counts)
