# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Hand-written CuTe DSL probe of the producer half of a quantize-on-load MXFP8 wgrad GEMM (no MMA).

One persistent CTA per SM runs ``warps`` producer warps (128 registers each, the budget a third warp group would
get inside QuACK's SM100 GEMM). The work list emulates what the producers of an on-load varlen-k GEMM must do: every
(32-token block, 128-column strip) unit of a ``[T, C]`` bf16 operand ``reuse`` times, ordered token-block-major so
concurrent CTAs share L2 lines as a GEMM wave does. A warp owns one unit: each lane loads two bf16 pairs per token
(cols ``2l, 2l+1`` and ``64+2l, 64+2l+1``, coalesced 128-byte rows), takes the per-column amax over the 32 tokens on
the integer magnitude bits, derives the e8m0 exponent, scales in f32 and converts with ``cvt.rn.satfinite.e4m3x2``,
and stores the e4m3 bytes and exponents to a shared-memory stage, then fences the stage to the async proxy, as a
producer feeding tcgen05 would. ``quantize=False`` keeps the loads and the shared stores (no math).
"""

import functools

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.arith import bitcast as _bitcast
from cutlass.cutlass_dsl import T, dsl_user_op

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call

TOKENS = 32
STRIP_U32 = 64  # 128 bf16 columns per unit, as u32 pairs


def _f32_bits(x) -> Uint32:
    return Uint32(_bitcast(Float32(x).ir_value(), T.i32()))


def _bits_f32(x) -> Float32:
    return Float32(_bitcast(Uint32(x).ir_value(), T.f32()))


@dsl_user_op
def _cvt_e4m3x2(hi: Float32, lo: Float32, *, loc=None, ip=None) -> cutlass.Uint16:
    return cutlass.Uint16(
        llvm.inline_asm(
            T.i16(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def _exponent(amax_bits: Uint32) -> Uint32:
    """e8m0 exponent of ``ceil(log2(amax / 448))`` from amax's f32 bits, clipped to [1, 254]; 127 for zero."""
    e = Int32((amax_bits >> 23) & 0xFF) - 8
    e = e + Int32(cutlass.select_((amax_bits & 0x7FFFFF) > 0x600000, Int32(1), Int32(0)))
    e = Int32(cutlass.select_(e < 1, Int32(1), e))
    e = Int32(cutlass.select_(e > 254, Int32(254), e))
    e = Int32(cutlass.select_(amax_bits == 0, Int32(127), e))
    return Uint32(e)


@cute.jit
def _umax(a: Uint32, b: Uint32) -> Uint32:
    return Uint32(cutlass.select_(a > b, a, b))


@cute_launcher_factory
def _build_probe(*, reuse, warps, quantize, ctas, pad_bytes):
    threads = 32 * warps

    @cute.kernel
    def kernel(mX: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        smem = cutlass.utils.SmemAllocator()
        # Row-major (lane fastest: conflict-free stores). Two stages of (warps, 32 tokens, 64 u16) e4m3 pairs + exponents; pad pins one CTA per SM.
        sQ = smem.allocate_tensor(
            cutlass.Uint16,
            cute.make_layout(
                (2, warps, TOKENS, STRIP_U32), stride=(warps * TOKENS * STRIP_U32, TOKENS * STRIP_U32, STRIP_U32, 1)
            ),
            byte_alignment=128,
        )
        sE = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout((2, warps, 32), stride=(warps * 32, 32, 1)), byte_alignment=128
        )
        sPad = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(pad_bytes), byte_alignment=128)
        col_strips = mX.shape[1] // STRIP_U32
        units_per_kb = col_strips * reuse
        total = (mX.shape[0] // TOKENS) * units_per_kb
        stage = Int32(0)
        for u in cutlass.range(bidx * warps + warp, total, ctas * warps):
            kb = u // units_per_kb
            cs = (u % units_per_kb) % col_strips
            row0 = kb * TOKENS
            c0 = cs * STRIP_U32 + lane
            w0 = []
            w1 = []
            for t in cutlass.range_constexpr(TOKENS):
                w0.append(Uint32(mX[row0 + t, c0]))
                w1.append(Uint32(mX[row0 + t, c0 + 32]))
            if cutlass.const_expr(quantize):
                a0 = Uint32(0)
                a1 = Uint32(0)
                a2 = Uint32(0)
                a3 = Uint32(0)
                for t in cutlass.range_constexpr(TOKENS):
                    a0 = _umax(a0, (w0[t] << 16) & 0x7FFFFFFF)
                    a1 = _umax(a1, w0[t] & 0x7FFF0000)
                    a2 = _umax(a2, (w1[t] << 16) & 0x7FFFFFFF)
                    a3 = _umax(a3, w1[t] & 0x7FFF0000)
                e0, e1, e2, e3 = _exponent(a0), _exponent(a1), _exponent(a2), _exponent(a3)
                s0 = _bits_f32((Uint32(254) - e0) << 23)
                s1 = _bits_f32((Uint32(254) - e1) << 23)
                s2 = _bits_f32((Uint32(254) - e2) << 23)
                s3 = _bits_f32((Uint32(254) - e3) << 23)
                for t in cutlass.range_constexpr(TOKENS):
                    sQ[stage, warp, t, lane] = _cvt_e4m3x2(
                        _bits_f32(w0[t] & 0xFFFF0000) * s1, _bits_f32(w0[t] << 16) * s0
                    )
                    sQ[stage, warp, t, lane + 32] = _cvt_e4m3x2(
                        _bits_f32(w1[t] & 0xFFFF0000) * s3, _bits_f32(w1[t] << 16) * s2
                    )
                sE[stage, warp, lane] = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24)
            else:
                for t in cutlass.range_constexpr(TOKENS):
                    sQ[stage, warp, t, lane] = cutlass.Uint16(w0[t] ^ (w0[t] >> 16))
                    sQ[stage, warp, t, lane + 32] = cutlass.Uint16(w1[t] ^ (w1[t] >> 16))
            cute.arch.fence_view_async_shared()
            stage = 1 - stage
        cute.arch.sync_threads()
        sPad[tidx] = cutlass.Uint8(tidx)
        cute.arch.sync_threads()
        mOut[bidx * threads + tidx] = (
            Uint32(sQ[0, warp, lane % TOKENS, lane]) + Uint32(sE[1, warp, lane]) + Uint32(sPad[(tidx + 1) % threads])
        )

    @cute.jit
    def launcher(stream, mX, mOut):
        kernel(mX, mOut).launch(grid=[ctas, 1, 1], block=[threads, 1, 1], stream=stream)

    return launcher


@functools.partial(jax.jit, static_argnames=("reuse", "warps", "quantize", "ctas", "pad_bytes"))
def producer_probe(x, *, reuse: int, warps: int = 4, quantize: bool = True, ctas: int, pad_bytes: int = 160 * 1024):
    """Run the probe over bf16 ``x[T, C]`` (``T % 32 == 0``, ``C % 128 == 0``); returns a small checksum."""
    t, c = x.shape
    xu = jax.lax.bitcast_convert_type(x.reshape(t, c // 2, 2), jnp.uint32)
    launcher = _build_probe(reuse=reuse, warps=warps, quantize=quantize, ctas=ctas, pad_bytes=pad_bytes)
    spec = cjax.TensorSpec(static=False)
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((ctas * 32 * warps,), jnp.uint32),
        input_spec=(spec,),
        output_spec=(spec,),
        use_static_tensors=False,
    )
    return call(xu)
