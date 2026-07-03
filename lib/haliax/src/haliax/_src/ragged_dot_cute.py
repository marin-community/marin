# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""CuTe (CUTLASS Python DSL) FP8 grouped (ragged) matmul for Hopper H100.

The non-Mosaic backend for the FP8 ``ragged_dot``, wrapped as a JAX custom call
via ``cutlass.jax.cutlass_call`` (no forked jaxlib -- NVIDIA's WGMMA path
supports mixed E4M3/E5M2 out of the box).

* ``cute_ragged_dot`` (forward + dgrad) and ``cute_wgrad`` (token-M-contracting
  weight gradient) both dispatch to NVIDIA's stock Hopper TMA warp-specialized
  persistent grouped-GEMM kernel in ``_tma_grouped_gemm.py`` (WGMMA + TMA +
  per-group device tensormap updates, ~1130-1230 TFLOP/s). The wgrad runs it in
  ragged-contraction mode: the token axis is the GEMM K, so A/B keep one aligned
  full-buffer base and the per-group token offset is a TMA element coordinate.
* ``cute_cast_transpose`` (fused bf16->FP8 quantize + transpose feeding the wgrad)
  still uses the simpler ``CopyUniversalOp`` (non-TMA) elementwise kernel below.
"""

import importlib

import jax
import jax.numpy as jnp

# The CuTe DSL is an optional GPU-only dependency. Import it dynamically (as the
# mainline FA4 backend does) so the static type checker treats it as absent
# rather than an unresolved import on CPU-only checkout/lint environments.
try:
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    cjax = importlib.import_module("cutlass.jax")

    _HAS_CUTE = True
except Exception:  # optional GPU-only dep
    cutlass = cute = cjax = None
    _HAS_CUTE = False


# 128-bit vectorized gmem<->smem copies (the FFI CopyUniversalOp path).
_UNIVERSAL_COPY_BITS = 128
# Hopper FP8 WGMMA tile-K granularity: the MMA instruction consumes 32 K elements
# and tile_k = mma_inst_k * 4 = 128; K must be a multiple of this.
_FP8_WGMMA_K_GRANULARITY = 128


def cute_available() -> bool:
    """True iff the CuTe DSL imports and a GPU backend is present."""
    return _HAS_CUTE and jax.default_backend() == "gpu"


# Fused cast-transpose: one CTA reads a (tile_m tokens x tile_n) bf16 sub-tile and
# emits its row-major FP8 (coalesced 128-bit read + vectorized fp8 store) AND its
# token-major FP8 transpose. The transpose is staged through shared memory so BOTH
# gmem writes are coalesced (threads write mYt contiguously along the token axis M),
# instead of an uncoalesced element scatter.
_CAST_TRANSPOSE_TILE = 64
_CAST_TRANSPOSE_VEC = _UNIVERSAL_COPY_BITS // 16  # bf16 elems per 128-bit read (== 8)


def _cast_transpose_launcher(*, tile, out_maxv, m_vec_aligned):
    """Build the ``@cute.jit`` batched fused cast-transpose launcher for ``cutlass_call``.

    Reads bf16 ``mX[E,M,N]`` once and writes ``mY[E,M,N]`` (row-major FP8,
    bit-identical to ``quantize(x, out_dtype, scale)``) and ``mYt[E,N,M]`` (its
    per-expert transpose). ``E`` is a batch (expert) axis mapped to ``blockIdx.z``:
    ``E=1`` is the plain 2-D cast-transpose (activations / output-grad); ``E>1`` is
    the per-expert weight dual-write (``rhs[E,K,N] -> q_rhs[E,K,N] + q_rhs_t[E,N,K]``)
    -- both fp8 weight layouts from one HBM read. The quantize divides by
    ``mScale[0]`` and clips to ``[-out_maxv, out_maxv]`` before the FP8 cast, matching
    ``_src/fp8.quantize`` exactly. ``out_maxv`` is the FP8 dtype's finite max (448 for
    E4M3, 57344 for E5M2). ``m_vec_aligned`` (M % vec == 0) selects the coalesced
    vectorized token-major store; otherwise a per-element store keeps arbitrary M
    correct (row bases of ``mYt`` are unaligned when M % vec != 0).
    """
    cuda = importlib.import_module("cuda.bindings.driver")

    vec = _CAST_TRANSPOSE_VEC
    # Same 2-D thread grid for both phases: ``thr_rows`` thread-rows x ``n_vecs``
    # vec-wide value chunks (== num_threads), each thread owning ``tile // thr_rows``
    # rows via the rest dimension. Phase 1 lays value along N (coalesced read /
    # row-store); phase 2 lays value along the token axis M (coalesced transpose
    # store). 256 threads/block keeps this bandwidth-bound kernel at high occupancy.
    num_threads = 256
    n_vecs = tile // vec
    thr_rows = num_threads // n_vecs

    class CastTransposeFp8:
        @cute.jit
        def __call__(
            self,
            stream: cuda.CUstream,
            mX: cute.Tensor,
            mScale: cute.Tensor,
            mY: cute.Tensor,
            mYt: cute.Tensor,
        ):
            x_dtype = mX.element_type
            c_dtype = mY.element_type

            # thr_rows thread-rows x n_vecs vec-wide value chunks; value along mode 1.
            thread_layout = cute.make_layout((thr_rows, n_vecs), stride=(n_vecs, 1))
            value_layout = cute.make_layout((1, vec))
            tiled_copy_in = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS),
                thread_layout,
                value_layout,
            )
            # FP8 stores: same TV layout, vec fp8 elems per copy (64-bit).
            fp8_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=vec * c_dtype.width),
                thread_layout,
                value_layout,
            )

            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[cute.struct.MemRange[c_dtype, tile * tile], 1024]

            e_total = mX.shape[0]
            m_total = mX.shape[1]
            n_total = mX.shape[2]
            grid = (cute.ceil_div(m_total, tile), cute.ceil_div(n_total, tile), e_total)
            self.kernel(
                mX,
                mScale,
                mY,
                mYt,
                tiled_copy_in,
                fp8_copy,
                c_dtype,
                out_maxv,
                SharedStorage,
            ).launch(grid=grid, block=[num_threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mScale: cute.Tensor,
            mY: cute.Tensor,
            mYt: cute.Tensor,
            tiled_copy_in: cute.TiledCopy,
            fp8_copy: cute.TiledCopy,
            c_dtype: cutlass.Constexpr,
            out_maxv: cutlass.Constexpr,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bm, bn, be = cute.arch.block_idx()
            # Slice this block's expert: mX/mY are [E,M,N], mYt is [E,N,M]. E=1 leaves
            # the plain 2-D cast-transpose; E>1 is the per-expert weight dual-write.
            mXe = mX[(be, None, None)]
            mYe = mY[(be, None, None)]
            mYte = mYt[(be, None, None)]
            m_total = mXe.shape[0]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            # sQ holds the quantized [tile_m, tile_n] FP8 sub-tile; sQt views the same
            # bytes transposed ([tile_n, tile_m]) for the coalesced token-major store.
            sQ = storage.sQ.get_tensor(cute.make_layout((tile, tile), stride=(tile, 1)))
            sQt = storage.sQ.get_tensor(cute.make_layout((tile, tile), stride=(1, tile)))

            thr_in = tiled_copy_in.get_slice(tidx)
            thr_fp8 = fp8_copy.get_slice(tidx)

            # --- Phase 1: coalesced bf16 read -> quantize -> row-major FP8 + smem stage.
            gX = cute.local_tile(mXe, (tile, tile), (bm, bn))
            gY = cute.local_tile(mYe, (tile, tile), (bm, bn))
            cId = cute.make_identity_tensor((tile, tile))
            tXgX = thr_in.partition_S(gX)
            tXcId = thr_in.partition_S(cId)
            tYgY = thr_fp8.partition_D(gY)
            tXsQ = thr_fp8.partition_D(sQ)
            rest_m = cute.size(tXgX, mode=[1])

            frag_x = cute.make_fragment_like(tXgX)
            for rm in cutlass.range_constexpr(rest_m):
                if bm * tile + tXcId[0, rm, 0][0] < m_total:
                    cute.copy(tiled_copy_in, tXgX[None, rm, None], frag_x[None, rm, None])
                else:
                    frag_x[None, rm, None].fill(0)

            scale = mScale[0]
            xf = frag_x.load().to(cutlass.Float32) / scale
            xf = cute.where(xf > out_maxv, cutlass.Float32(out_maxv), xf)
            xf = cute.where(xf < -out_maxv, cutlass.Float32(-out_maxv), xf)
            frag_q = cute.make_fragment_like(tXsQ)
            frag_q.store(xf.to(c_dtype))

            cute.copy(fp8_copy, frag_q, tXsQ)  # stage full tile in smem
            for rm in cutlass.range_constexpr(rest_m):
                if bm * tile + tXcId[0, rm, 0][0] < m_total:
                    cute.copy(fp8_copy, frag_q[None, rm, None], tYgY[None, rm, None])

            cute.arch.sync_threads()

            # --- Phase 2: read smem transposed -> coalesced token-major FP8 store.
            gYt = cute.local_tile(mYte, (tile, tile), (bn, bm))
            cIdT = cute.make_identity_tensor((tile, tile))
            tPsQt = thr_fp8.partition_S(sQt)
            tPgYt = thr_fp8.partition_D(gYt)
            tPcT = thr_fp8.partition_S(cIdT)
            rest_n = cute.size(tPgYt, mode=[1])
            # Compact register fragment (from the contiguous gmem destination) so the
            # token-major store can vectorize; the strided transposed smem read fills it.
            frag_t = cute.make_fragment_like(tPgYt)
            cute.autovec_copy(tPsQt, frag_t)

            for rn in cutlass.range_constexpr(rest_n):
                base_m = bm * tile + tPcT[0, rn, 0][1]
                gn = bn * tile + tPcT[0, rn, 0][0]
                if cutlass.const_expr(m_vec_aligned):
                    # M % vec == 0: every mYt row base is vec-aligned, so a full atom is
                    # either wholly in range (coalesced 64-bit store) or wholly out.
                    if base_m + vec <= m_total:
                        cute.copy(fp8_copy, frag_t[None, rn, None], tPgYt[None, rn, None])
                else:
                    # Arbitrary M: unaligned rows forbid the vectorized store; write each
                    # token-major element individually (correctness path for odd M).
                    for v in cutlass.range_constexpr(vec):
                        if base_m + v < m_total:
                            mYte[gn, base_m + v] = frag_t[v, rn, 0]

    return CastTransposeFp8()


def _cast_transpose_call(x3, scale, *, out_dtype):
    """Run the batched dual-write kernel on ``x3[E,M,N]`` -> ``([E,M,N], [E,N,M])``."""
    tile = _CAST_TRANSPOSE_TILE
    e, m, n = x3.shape
    if n % tile != 0:
        raise ValueError(f"cute_cast_transpose: N={n} is not divisible by tile={tile}")
    out_maxv = float(jnp.finfo(out_dtype).max)
    scale_1 = jnp.reshape(scale.astype(jnp.float32), (1,))
    launcher = _cast_transpose_launcher(tile=tile, out_maxv=out_maxv, m_vec_aligned=(m % _CAST_TRANSPOSE_VEC == 0))
    x_spec = cjax.TensorSpec(mode=(0, 1, 2), divisibility=(1, 1, _CAST_TRANSPOSE_VEC), static=True)
    scale_spec = cjax.TensorSpec(mode=(0,), static=True)
    y_spec = cjax.TensorSpec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    yt_spec = cjax.TensorSpec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((e, m, n), out_dtype),
            jax.ShapeDtypeStruct((e, n, m), out_dtype),
        ),
        input_spec=(x_spec, scale_spec),
        output_spec=(y_spec, yt_spec),
        use_static_tensors=True,
    )
    return call(x3, scale_1)


def cute_cast_transpose(x, scale, *, out_dtype):
    """Fused single-read cast-transpose of bf16 ``x[M,N]`` to FP8.

    Returns ``(row_major[M,N], token_major[N,M])`` where ``row_major`` is bit-identical
    to ``quantize(x, out_dtype, scale, jnp.float32)`` and ``token_major`` is its
    transpose. One HBM read of ``x`` produces both layouts (plus the FP8 quantize),
    replacing the pure-JAX ``quantize + swapaxes`` two-pass. Requires N divisible by
    the tile and a bf16 input (guarded by the caller in ``cast_transpose``)."""
    if not cute_available():
        raise RuntimeError("cute_cast_transpose requires the CuTe DSL on a GPU backend")
    m, n = x.shape
    row, col = _cast_transpose_call(x[None], scale, out_dtype=out_dtype)
    return row[0], col[0]


def cute_cast_transpose_weight(rhs, scale, *, out_dtype):
    """Fused dual-write per-expert weight quantize: bf16 ``rhs[E,K,N]`` -> both FP8
    layouts in one HBM read.

    Returns ``(q_rhs[E,K,N], q_rhs_t[E,N,K])`` where ``q_rhs`` is bit-identical to
    ``quantize(rhs, out_dtype, scale)`` (natural layout, the dgrad operand) and
    ``q_rhs_t`` is its per-expert transpose ``quantize(swapaxes(rhs,1,2), ...)`` (the
    K-contiguous forward operand). Collapses the previous ``in_q(rhs)`` +
    ``quantize(swapaxes(rhs))`` (three HBM reads of ``rhs``) to one."""
    if not cute_available():
        raise RuntimeError("cute_cast_transpose_weight requires the CuTe DSL on a GPU backend")
    return _cast_transpose_call(rhs, scale, out_dtype=out_dtype)


# CTA tile of the TMA grouped-GEMM kernel: N must be a multiple of tile_n and K a
# multiple of the FP8 WGMMA/TMA granularity (the per-group byte offsets are then
# always 16B-aligned for TMA). The wgrad output rows (hidden dim) tile by tile_m.
_TMA_TILE_N = 256
_TMA_TILE_M = 128


def cute_ragged_dot(a, b, group_sizes, *, out_dtype, out_scale):
    """Grouped GEMM ``a[M,K] . b[E,N,K] -> [M,N]`` contracting the last axis K
    (contiguous for both). Epilogue divides the f32 accumulator by ``out_scale``.

    Backed by NVIDIA's stock Hopper TMA warp-specialized persistent grouped-GEMM
    kernel (``_tma_grouped_gemm``). Serves both the forward (E4M3xE4M3) and the
    dgrad (E5M2xE4M3) -- both arrive as ``a[M,K] . b[E,N,K]``."""
    if not cute_available():
        raise RuntimeError("cute_ragged_dot requires the CuTe DSL on a GPU backend")
    e, n, k = b.shape
    if n % _TMA_TILE_N != 0:
        raise ValueError(
            f"cute_ragged_dot: N={n} is not divisible by tile_n={_TMA_TILE_N}; pad or reshape b before calling"
        )
    if k % _FP8_WGMMA_K_GRANULARITY != 0:
        raise ValueError(
            f"cute_ragged_dot: K={k} is not divisible by {_FP8_WGMMA_K_GRANULARITY} "
            "(FP8 WGMMA K granularity); pad or reshape before calling"
        )
    # Guarded optional-dep import: the vendored TMA kernel hard-imports cutlass.
    from haliax._src._tma_grouped_gemm import tma_grouped_gemm  # noqa: PLC0415

    return tma_grouped_gemm(a, b, group_sizes, out_dtype=out_dtype, out_scale=out_scale)


def cute_wgrad(a_t, b_t, group_sizes, *, out_dtype, out_scale):
    """Grouped wgrad ``a_t[K,M] . b_t[N,M] -> [E,K,N]`` contracting the token axis M.

    ``a_t``/``b_t`` are token-major FP8 (cast-transposed activations E4M3 and
    output-grad E5M2); M is the packed, non-tile-aligned per-group contraction.
    Produces one ``[K,N]`` weight-gradient slab per expert. The epilogue divides
    the f32 accumulator by ``out_scale`` (the reciprocal of the operand scale
    product, matching the dequantize convention).

    Backed by the TMA warp-specialized persistent grouped-GEMM kernel in
    ragged-contraction mode (``tma_grouped_wgrad``): the token axis is the GEMM K,
    so the guards are on the OUTPUT dims -- K (hidden rows) tiled by ``tile_m`` and
    N (cols) by ``tile_n`` -- while the ragged M needs no divisibility (TMA
    zero-fills the tail)."""
    if not cute_available():
        raise RuntimeError("cute_wgrad requires the CuTe DSL on a GPU backend")
    k = a_t.shape[0]  # output rows (hidden dim), == out.shape[1]
    n = b_t.shape[0]  # output cols, == out.shape[2]
    if k % _TMA_TILE_M != 0:
        raise ValueError(f"cute_wgrad: K={k} is not divisible by tile_m={_TMA_TILE_M}; pad or reshape before calling")
    if n % _TMA_TILE_N != 0:
        raise ValueError(f"cute_wgrad: N={n} is not divisible by tile_n={_TMA_TILE_N}; pad or reshape before calling")
    # Guarded optional-dep import: the vendored TMA kernel hard-imports cutlass.
    from haliax._src._tma_grouped_gemm import tma_grouped_wgrad  # noqa: PLC0415

    return tma_grouped_wgrad(a_t, b_t, group_sizes, out_dtype=out_dtype, out_scale=out_scale)
