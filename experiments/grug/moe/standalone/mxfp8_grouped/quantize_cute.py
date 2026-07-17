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
warp). The e8m0 round-up is the exact bit-twiddle of
``cast_to_e8m0_with_rounding_up``, and the rescale multiplies by the exact
power-of-two inverse ``2^(127 - e)`` (bit pattern ``(254 - e) << 23``; for
bf16 inputs e <= 247, and the zero-amax block gives e = 0 -> inverse 2^127,
so the inverse is always a normal f32 and the multiply is bit-identical to
the reference's division). Finite inputs only: inf/NaN blocks produce e=255
and diverge from the XLA reference's inf-scale semantics.
"""

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp

from .adapter import (
    _BLACKWELL_CUTE_ARCH,
    E4M3_MAX,
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


def _e8m0_round_up(amax):
    """Exact ``cast_to_e8m0_with_rounding_up(amax / 448)`` bit-twiddle (Int32)."""
    t = amax / cutlass.Float32(E4M3_MAX)
    # Clear the sign bit: amax can be -0.0 for all-(+/-)0 blocks (the where
    # based abs keeps -0.0), and an arithmetic >> would smear its sign into
    # the exponent byte. abs(-0.0) == +0.0 must yield scale byte 0.
    u = t.bitcast(cutlass.Int32) & 0x7FFFFFFF
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF
    is_ru = (mant > 0) & (exp != 0xFE) & ((exp != 0) | (mant > 0x400000))
    return exp + is_ru.to(cutlass.Int32)


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
            # abs via select: cute.absf lowers to a libdevice __nv_fabsf call
            # that the pod-side FFI NVVM compile cannot link (job 002c-g1).
            # -0.0 may leak through (handled in _e8m0_round_up).
            ax = cute.where(xf < cutlass.Float32(0.0), -xf, xf)

            # ---- 32-block amaxes for both orientations ---------------------
            # In-thread partials: per-row max over the 8 owned columns, and
            # per-column max over the 8 owned rows.
            rp = ax.reduce(cute.ReductionOp.MAX, 0.0, (None, 1))  # (8 rows,)
            cp = ax.reduce(cute.ReductionOp.MAX, 0.0, (1, None))  # (8 cols,)
            r_amax = cute.make_fragment(_THR_TILE, cutlass.Float32)
            r_amax.store(rp)
            c_amax = cute.make_fragment(_THR_TILE, cutlass.Float32)
            c_amax.store(cp)

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
                e_r = _e8m0_round_up(_group_max(r_amax[i], (1, 2)))
                sr_b[i] = e_r.to(cutlass.Uint8)
                inv_r[i] = ((254 - e_r) << 23).bitcast(cutlass.Float32)
                e_c = _e8m0_round_up(_group_max(c_amax[i], (8, 16)))
                sc_b[i] = e_c.to(cutlass.Uint8)
                inv_c[i] = ((254 - e_c) << 23).bitcast(cutlass.Float32)

            # ---- rescale + clip + fp8 casts, all from registers ------------
            maxv = cutlass.Float32(E4M3_MAX)
            minv = cutlass.Float32(-E4M3_MAX)

            qr = xf * inv_r.load().reshape((_THR_TILE, 1)).broadcast_to((_THR_TILE, _THR_TILE))
            qr = cute.where(qr > maxv, maxv, qr)
            qr = cute.where(qr < minv, minv, qr)
            gQr = cute.local_tile(mQr, (_THR_TILE, _THR_TILE), (row_blk, col_blk))
            frag_qr = cute.make_fragment_like(gQr)
            frag_qr.store(qr.to(frag_qr.element_type))
            cute.autovec_copy(frag_qr, gQr)

            qc = xf * inv_c.load().reshape((1, _THR_TILE)).broadcast_to((_THR_TILE, _THR_TILE))
            qc = cute.where(qc > maxv, maxv, qc)
            qc = cute.where(qc < minv, minv, qc)
            gQc = cute.local_tile(mQc, (_THR_TILE, _THR_TILE), (row_blk, col_blk))
            frag_qc = cute.make_fragment_like(gQc)
            frag_qc.store(qc.to(frag_qc.element_type))
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
