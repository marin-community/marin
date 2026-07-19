# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused dual-orientation MXFP8 quantizer kernel (CuTe DSL, sm100) -- MXFP8-002c.

One HBM read of a bf16 activation ``x[M, K]`` produces all four MXFP8 operand
tensors the grouped GEMMs consume (see ``adapter.py``):

- ``q_row [M, K]`` e4m3, quantized along K (fwd/dgrad A operand), bit-exact vs
  ``quantize_mxfp8``;
- ``s_row_t [K//32, M]`` uint8 raw rowwise e8m0 scales, TRANSPOSED so the
  per-thread 8-byte scale runs are contiguous (coalesced stores; the natural
  ``(M, K//32)`` layout would scatter single bytes at stride K//32);
- ``q_col [M, K]`` e4m3, quantized along the token axis in natural storage
  (the 2Dx2D wgrad operand -- no data transpose), bit-exact vs
  ``quantize_mxfp8_tokens``;
- ``s_col [M//32, K]`` uint8 raw token-axis scales (natural layout is already
  coalesced here).

The group-dependent SF swizzles (``build_sfa_fast`` / ``build_sf_wgrad_fast``)
stay on the existing bit-exact XLA path: the scale tensors are 1/64 of the
data bytes, so the swizzle pass is noise next to the quantize.

Kernel structure: each thread owns an 8x8 bf16 sub-tile (four 16B reads); a
CTA of 256 threads in an (32, 8) thread grid covers a (256, 64) tile. Rowwise
32-block amax = in-thread reduce over the 8 owned columns + butterfly shuffle
over 4 column-adjacent lanes (xor 1, 2); token-axis amax = in-thread reduce
over the 8 owned rows + shuffle over 4 row-adjacent lanes (xor 8, 16; the
warp's 4 thread-rows are 4-aligned in the CTA, so both groups sit inside one
warp). The e8m0 round-up is an integer-only derivation matching
``cast_to_e8m0_with_rounding_up(amax / 448)`` exactly for bf16-granular amax
(see ``_e8m0_round_up``), and the rescale multiplies by the exact
power-of-two inverse ``2^(127 - e)`` (bit pattern ``(254 - e) << 23``; for
bf16 inputs e <= 247, and the zero-amax block gives e = 0 -> inverse 2^127,
so the inverse is always a normal f32 and the multiply is bit-identical to
the reference's division). The f32 -> e4m3 conversion is an inline-PTX
``cvt.rn.satfinite.e4m3x2.f32`` (see ``_cvt2_e4m3``); satfinite saturation
subsumes the reference's clip. Finite inputs only: inf/NaN blocks diverge
from the XLA reference's inf-scale semantics.
"""

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
from cutlass._mlir.dialects import llvm

from .adapter import (
    _BLACKWELL_CUTE_ARCH,
    SF_VEC_SIZE,
    _as_gmem_tensor,
    build_sf_wgrad_fast,
    build_sfa_fast,
    ensure_blackwell_arch,
)

# Per-thread sub-tile (rows x cols) and the CTA thread grid: 256 threads as
# (32 thread-rows, 8 thread-cols) -> CTA tile (256, 64).
_THR_TILE = 8
_THR_ROWS = 32
_THR_COLS = 8
_NUM_THREADS = _THR_ROWS * _THR_COLS
_TILE_M = _THR_ROWS * _THR_TILE  # 256
_TILE_N = _THR_COLS * _THR_TILE  # 64
# 128-bit gmem loads: 8 bf16 elements.
_LOAD_VEC = 8


def _group_max(v, offsets: tuple[int, ...]):
    """Butterfly max-reduce of a scalar across the lanes reached by ``offsets``."""
    for o in offsets:
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, o))
    return v


def _cvt2_e4m3(hi, lo):
    """Pack two f32 into an e4m3x2 Int16 via inline PTX (``hi`` in the upper byte).

    The DSL-native ``frag.store(ssa.to(Float8E4M3FN))`` is bit-identical (the
    full 002c suite incl. adversarial cases passes on it) but measured ~12%
    slower on the d2560 kernel and ~9% on the 4-tensor producer total in a
    same-pod interleaved A/B (job mxfp8-cvt-ab-g1, 2 rounds) — enough to
    push the producer past its XLA break-even budget, so the explicit paired
    cvt stays. ``cvt.rn.satfinite`` is RTNE with saturation to +-448,
    bit-identical to the XLA clip+convert for finite inputs. (Historical:
    this started as a workaround for NVVM compile failures that were really
    the cutlass-dsl libs-base/cu13 wheel-shadowing conflict, fixed in the
    lockfile; the intrinsic path compiles fine on a clean cu13 install.)
    """
    res = llvm.inline_asm(
        cutlass.Int16.mlir_type,
        [cutlass.Float32(hi).ir_value(), cutlass.Float32(lo).ir_value()],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int16(res)


def _store_ssa_as_e4m3(q_ssa, frag):
    """Convert a (8, 8) f32 TensorSSA to e4m3 into ``frag`` via inline-asm cvt.

    ``frag`` must be a compact row-contiguous (8, 8) e4m3 register fragment;
    pairs of column-adjacent values pack into one i16 (lo byte = even column,
    matching little-endian storage order).
    """
    tmp = cute.make_fragment_like(frag, cutlass.Float32)
    tmp.store(q_ssa)
    frag16 = cute.recast_tensor(frag, cutlass.Int16)
    # Plain range: helper is not AST-preprocessed, so unroll at trace time.
    for r in range(_THR_TILE):
        for c in range(_THR_TILE // 2):
            frag16[(r, c)] = _cvt2_e4m3(tmp[(r, 2 * c + 1)], tmp[(r, 2 * c)])


def _e8m0_round_up(amax):
    """Integer-exact e8m0 round-up cast (Int32 byte value).

    Matches ``cast_to_e8m0_with_rounding_up(amax / 448)`` bit-for-bit for
    every amax with bf16 mantissa granularity -- exhaustively validated
    against the f32-division reference on all 32640 finite ``|bf16|`` values
    (block amaxes of bf16 inputs are always such values). Integer-only: the
    division cost 16 FDIV sequences per thread in the first cut. The sign
    mask canonicalizes the -0.0 amax of an all-(+/-)0 block to byte 0.
    Finite inputs only (inf amax is out of contract).
    """
    u = amax.bitcast(cutlass.Int32) & 0x7FFFFFFF
    exp_a = u >> 23
    m23 = u & 0x7FFFFF
    # t = amax/448 = (1.m / 1.75) * 2^(exp_a - 127 - 8): the ratio is in
    # (0.571, 1.143], so t's biased exponent is exp_a - 9 (+1 when 1.m >=
    # 1.75), and the round-up adds 1 except when the ratio is exactly 1.
    base = exp_a - 9 + (m23 >= 0x600000).to(cutlass.Int32)
    e_norm = base + 1 - (m23 == 0x600000).to(cutlass.Int32)
    # base < 1: t is subnormal; it rounds to scale byte 0 or 1 with the cut
    # between amax bits 0x04600000 and 0x04610000 (bf16-granular amax).
    is_sub = (base < 1).to(cutlass.Int32)
    e_sub = (u > 0x04600000).to(cutlass.Int32)
    return is_sub * e_sub + (1 - is_sub) * e_norm


def _dual_quantize_launcher():
    class DualQuantizeMxfp8:
        @cute.jit
        def __call__(
            self,
            stream,
            mX: cute.Tensor,
            mQr: cute.Tensor,
            mSrT: cute.Tensor,
            mQc: cute.Tensor,
            mSc: cute.Tensor,
        ):
            # cu13 DSL wheel hands cutlass_call tensors over in GENERIC address
            # space; rebuild as gmem pointers (see adapter._as_gmem_tensor).
            mX = _as_gmem_tensor(mX)
            mQr = _as_gmem_tensor(mQr)
            mSrT = _as_gmem_tensor(mSrT)
            mQc = _as_gmem_tensor(mQc)
            mSc = _as_gmem_tensor(mSc)
            m = mX.shape[0]
            k = mX.shape[1]
            # blockIdx.x walks the K tiles so consecutive CTAs touch adjacent
            # 128B column bands of the same rows (L2 locality).
            grid = (k // _TILE_N, m // _TILE_M, 1)
            self.kernel(mX, mQr, mSrT, mQc, mSc).launch(grid=grid, block=[_NUM_THREADS, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mQr: cute.Tensor,
            mSrT: cute.Tensor,
            mQc: cute.Tensor,
            mSc: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bn, bm, _ = cute.arch.block_idx()
            tr = tidx // _THR_COLS
            tc = tidx % _THR_COLS
            row_blk = bm * _THR_ROWS + tr  # 8-row block index
            col_blk = bn * _THR_COLS + tc  # 8-col block index

            # ---- single gmem read of the thread's 8x8 bf16 sub-tile --------
            gX = cute.local_tile(mX, (_THR_TILE, _THR_TILE), (row_blk, col_blk))
            frag_x = cute.make_fragment_like(gX)
            cute.autovec_copy(gX, frag_x)
            xf = frag_x.load().to(cutlass.Float32)

            # ---- 32-block amaxes for both orientations ---------------------
            # amax = max(max(x), -min(x)) instead of an elementwise abs: the
            # select-based abs cost 3 PTX ops/element (cute.absf is a
            # libdevice call the pod NVVM cannot link, job 002c-g1); paired
            # max/min reduces amortize to ~2 ops/element. 0.0 init is safe
            # (amax >= 0), and a -0.0 amax from an all-(+/-)0 block is
            # canonicalized in _e8m0_round_up.
            rmax = xf.reduce(cute.ReductionOp.MAX, 0.0, (None, 1))  # (8 rows,)
            rmin = xf.reduce(cute.ReductionOp.MIN, 0.0, (None, 1))
            cmax = xf.reduce(cute.ReductionOp.MAX, 0.0, (1, None))  # (8 cols,)
            cmin = xf.reduce(cute.ReductionOp.MIN, 0.0, (1, None))
            r_hi = cute.make_fragment(_THR_TILE, cutlass.Float32)
            r_hi.store(rmax)
            r_lo = cute.make_fragment(_THR_TILE, cutlass.Float32)
            r_lo.store(rmin)
            c_hi = cute.make_fragment(_THR_TILE, cutlass.Float32)
            c_hi.store(cmax)
            c_lo = cute.make_fragment(_THR_TILE, cutlass.Float32)
            c_lo.store(cmin)

            # ---- shuffle-combine + e8m0 round-up + exact inverses ----------
            # Rowwise 32-col block spans the 4 column-adjacent lanes (lane =
            # 8*i + j with i = tr%4, j = tc; xor 1/2 flip j); the token-axis
            # 32-row block spans the 4 row-adjacent lanes (xor 8/16 flip i;
            # the warp's thread-rows are 4-aligned in the CTA).
            sr_b = cute.make_fragment(_THR_TILE, cutlass.Uint8)
            sc_b = cute.make_fragment(_THR_TILE, cutlass.Uint8)
            inv_r = cute.make_fragment(_THR_TILE, cutlass.Float32)
            inv_c = cute.make_fragment(_THR_TILE, cutlass.Float32)
            for i in cutlass.range_constexpr(_THR_TILE):
                e_r = _e8m0_round_up(_group_max(cute.arch.fmax(r_hi[i], -r_lo[i]), (1, 2)))
                sr_b[i] = e_r.to(cutlass.Uint8)
                inv_r[i] = ((254 - e_r) << 23).bitcast(cutlass.Float32)
                e_c = _e8m0_round_up(_group_max(cute.arch.fmax(c_hi[i], -c_lo[i]), (8, 16)))
                sc_b[i] = e_c.to(cutlass.Uint8)
                inv_c[i] = ((254 - e_c) << 23).bitcast(cutlass.Float32)

            # ---- rescale + fp8 casts, all from registers -------------------
            # No explicit clip: cvt.rn.satfinite saturates to +-448, which is
            # bit-identical to the reference's clip-then-convert for finite
            # values (and q <= 448 mathematically by the round-up scale).
            qr = xf * inv_r.load().reshape((_THR_TILE, 1)).broadcast_to((_THR_TILE, _THR_TILE))
            gQr = cute.local_tile(mQr, (_THR_TILE, _THR_TILE), (row_blk, col_blk))
            frag_qr = cute.make_fragment_like(gQr)
            _store_ssa_as_e4m3(qr, frag_qr)
            cute.autovec_copy(frag_qr, gQr)

            qc = xf * inv_c.load().reshape((1, _THR_TILE)).broadcast_to((_THR_TILE, _THR_TILE))
            gQc = cute.local_tile(mQc, (_THR_TILE, _THR_TILE), (row_blk, col_blk))
            frag_qc = cute.make_fragment_like(gQc)
            _store_ssa_as_e4m3(qc, frag_qc)
            cute.autovec_copy(frag_qc, gQc)

            # ---- scale-byte stores (one lane per 4-lane group) -------------
            # Rowwise scales, transposed layout (K//32, M): this thread's 8
            # bytes (its 8 rows, one 32-col block) are contiguous along M.
            if tc % 4 == 0:
                sr_row = mSrT[(bn * (_TILE_N // SF_VEC_SIZE) + tc // 4, None)]
                gSr = cute.local_tile(sr_row, (_THR_TILE,), (row_blk,))
                cute.autovec_copy(sr_b, gSr)
            # Token-axis scales, natural layout (M//32, K): 8 bytes contiguous
            # along K.
            if tr % 4 == 0:
                sc_row = mSc[(bm * (_TILE_M // SF_VEC_SIZE) + tr // 4, None)]
                gSc = cute.local_tile(sc_row, (_THR_TILE,), (col_blk,))
                cute.autovec_copy(sc_b, gSc)

    return DualQuantizeMxfp8()


_LAUNCHER = None


def _get_launcher():
    global _LAUNCHER
    if _LAUNCHER is None:
        _LAUNCHER = _dual_quantize_launcher()
    return _LAUNCHER


def quantizer_output_shapes(m: int, k: int):
    """(q_row, s_row_t, q_col, s_col) ShapeDtypeStructs for input (m, k)."""
    return (
        jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((k // SF_VEC_SIZE, m), jnp.uint8),
        jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((m // SF_VEC_SIZE, k), jnp.uint8),
    )


def quantizer_specs():
    """(input_spec, output_spec) for ``cutlass_call``/``build_function_spec``."""
    ts = cjax.TensorSpec
    x_spec = ts(mode=(0, 1), divisibility=(1, 2 * _LOAD_VEC), static=True)
    q_spec = ts(mode=(0, 1), divisibility=(1, 2 * _LOAD_VEC), static=True)
    s_spec = ts(mode=(0, 1), divisibility=(1, _THR_TILE), static=True)
    return (x_spec,), (q_spec, s_spec, q_spec, s_spec)


def dual_quantize_mxfp8_cute(x):
    """Dual-orientation MXFP8 quantize of bf16 ``x[M, K]`` in one kernel pass.

    Returns ``(q_row, s_row_t, q_col, s_col)`` (see module docstring for the
    layouts). Requires M % 256 == 0 and K % 64 == 0 (no boundary predication:
    the target activation shapes are M=262144, K in {2560, 1280}).
    """
    ensure_blackwell_arch()
    m, k = x.shape
    assert x.dtype == jnp.bfloat16, x.dtype
    assert m % _TILE_M == 0, f"M={m} must be a multiple of {_TILE_M}"
    assert k % _TILE_N == 0, f"K={k} must be a multiple of {_TILE_N}"
    input_spec, output_spec = quantizer_specs()
    call = cjax.cutlass_call(
        _get_launcher(),
        output_shape_dtype=quantizer_output_shapes(m, k),
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_BLACKWELL_CUTE_ARCH),),
    )
    return call(x)


def dual_quantize_activation_cute(t, row_idx, col_idx, block_perm):
    """Kernel-backed drop-in for ``adapter.dual_quantize_activation``.

    Same contract: returns ``(q_row, sf_row, q_col, sf_col)`` with the swizzled
    SF tensors; the heavy quantize runs in the fused CuTe kernel and the small
    group-dependent swizzles stay on the validated XLA path.
    """
    q_row, s_row_t, q_col, s_col = dual_quantize_mxfp8_cute(t)
    sf_row = build_sfa_fast(s_row_t.T, row_idx)
    sf_col = build_sf_wgrad_fast(s_col.T, col_idx, block_perm)
    return q_row, sf_row, q_col, sf_col
