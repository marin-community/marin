# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""First-party JAX adapter wrapping the vendored NVIDIA TMA grouped-GEMM kernel.

This module owns the host-side Python code that bridges JAX (``cutlass_call``)
to ``HopperGroupedGemmPersistentKernel`` (vendored in ``_tma_grouped_gemm.py``):

- Device prologue builders (``_build_tma_launcher``, ``_build_tma_wgrad_launcher``)
  that construct the stream-first ``@cute.jit`` launcher classes consumed by
  ``cutlass_call``.
- Entry-point functions ``tma_grouped_gemm`` (forward/dgrad) and
  ``tma_grouped_wgrad`` (token-M-contracting weight gradient).
- Host-side helpers: ``ensure_hopper_arch``, ``_pad_token_groups_16``,
  ``_padded_group_offsets``, ``_total_num_clusters_upper_bound``.

The vendored kernel body stays in ``_tma_grouped_gemm.py`` (NVIDIA BSD-3 licence,
pyrefly-excluded).  This module is type-checked by pyrefly; the cutlass DSL imports
are guarded so the module loads on CPU-only lint environments.
"""

import importlib
import os

import jax
import jax.numpy as jnp

# The CuTe DSL is an optional GPU-only dependency. Use importlib so the module
# can be statically type-checked without cutlass installed (matches the pattern
# in ragged_dot_cute.py).
try:
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    cjax = importlib.import_module("cutlass.jax")
    utils = importlib.import_module("cutlass.utils")
except Exception:
    cutlass = cute = cjax = utils = None

# H100 has 132 SMs at occupancy 1 for the (128,256) 3-warpgroup FP8 kernel; the
# tensormap workspace is sized by the total CTA count and the persistent grid is
# capped at 132 / cluster_size clusters. HardwareInfo() needs a live CUDA context
# on the compile thread (unavailable on the FFI path), so this is hardcoded.
_H100_SMS = 132
_TILE_SHAPE_MN = (128, 256)
# Fixed attributes from HopperGroupedGemmPersistentKernel (vendored class):
# bytes_per_tensormap = 128, num_tensormaps = 3 (A, B, C).
_TENSORMAP_BYTES = 128
_NUM_TENSORMAPS = 3
_TENSORMAP_I64_WORDS = _TENSORMAP_BYTES // 8  # 16 Int64 words per tensormap
# 128-bit contiguous alignment for the fp8 TMA operands (16 fp8 elems).
_FP8_TMA_VEC = 16
# CuTe DSL target arch for Hopper (H100) — exported via CUTE_DSL_ARCH and passed as
# compile_options to every cutlass_call. All three uses must agree.
_HOPPER_CUTE_ARCH = "sm_90a"


def ensure_hopper_arch() -> None:
    """Resolve the CuTe target arch once and fail fast on non-Hopper GPUs.

    GPU detection on the FFI compile thread can silently default to ``sm_100a``
    (Blackwell) -> a cubin that will not load on an H100. Derive the arch from
    the live JAX device and export ``CUTE_DSL_ARCH`` if the caller has not set
    it. Only called on the GPU backend (guarded by ``cute_available``).
    """
    dev = jax.devices()[0]
    cc = dev.compute_capability  # "9.0" on H100
    if not str(cc).startswith("9"):
        raise RuntimeError(
            "cute_ragged_dot TMA kernel requires a Hopper (sm_90) GPU; got device "
            f"{dev} with compute_capability={cc!r}"
        )
    os.environ.setdefault("CUTE_DSL_ARCH", _HOPPER_CUTE_ARCH)


def _cluster_tile_mn(cluster_shape_mn: tuple[int, int]) -> tuple[int, int]:
    return (_TILE_SHAPE_MN[0] * cluster_shape_mn[0], _TILE_SHAPE_MN[1] * cluster_shape_mn[1])


def _total_num_clusters_upper_bound(m_total: int, n: int, cluster_shape_mn: tuple[int, int], group_count: int) -> int:
    """Static upper bound on cluster tiles across all (ragged) groups.

    Per-group M is ragged with fixed total ``m_total``, so
    ``sum_g ceil_div(M_g, ctm) <= ceil_div(m_total, ctm) + (group_count - 1)``.
    Times the (fixed) N cluster-tile count. Surplus tiles are predicated away by
    the scheduler's ``found`` check, so an over-estimate is safe (never a hang).
    """
    ctm, ctn = _cluster_tile_mn(cluster_shape_mn)
    m_cluster_tiles = (m_total + ctm - 1) // ctm + (group_count - 1)
    n_cluster_tiles = (n + ctn - 1) // ctn
    return m_cluster_tiles * n_cluster_tiles


def _build_tma_launcher(
    *,
    cluster_shape_mn: tuple[int, int],
    group_count: int,
    n: int,
    k: int,
    total_num_clusters: int,
    max_active_clusters: int,
    a_bytes: int,
    b_bytes: int,
    c_bytes: int,
):
    """Build the stream-first ``@cute.jit`` adapter launcher for ``cutlass_call``."""
    from haliax._src._tma_grouped_gemm import HopperGroupedGemmPersistentKernel  # noqa: PLC0415

    # Guard against silent desync with the vendored class's fixed attributes.
    assert HopperGroupedGemmPersistentKernel.bytes_per_tensormap == _TENSORMAP_BYTES
    assert HopperGroupedGemmPersistentKernel.num_tensormaps == _NUM_TENSORMAPS

    # One prologue block; >= group_count threads, rounded up to a warp.
    addr_threads = max(32, ((group_count + 31) // 32) * 32)

    class TmaGroupedLauncher:
        @cute.jit
        def __call__(
            self,
            stream,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mScale: cute.Tensor,
            mInitA: cute.Tensor,
            mInitB: cute.Tensor,
            mInitC: cute.Tensor,
            mC: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddrI32: cute.Tensor,
            mTmapI32: cute.Tensor,
        ):
            # Reinterpret the Int32-doubled scratch buffers as real Int64 (the
            # library must run in a default x64-off JAX process; JAX would
            # otherwise truncate an Int64 buffer to 32 bits).
            addr_i64 = cute.recast_tensor(mAddrI32, cutlass.Int64)  # (E, 3)
            tmap_i64 = cute.recast_tensor(mTmapI32, cutlass.Int64)  # (num_sms, 3, 16)

            # 1) Device prologue: fill problem_shape / strides / addresses.
            self.fill_metadata(mA, mB, mC, mGroupSizes, mProblemShape, mStrides, addr_i64).launch(
                grid=[1, 1, 1], block=[addr_threads, 1, 1], stream=stream
            )

            # 2) Stock persistent TMA grouped GEMM (ordered after the prologue on
            #    the same stream).
            kernel = HopperGroupedGemmPersistentKernel(
                cutlass.Float32,
                _TILE_SHAPE_MN,
                cluster_shape_mn,
                tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
            )
            kernel(
                mInitA,
                mInitB,
                mInitC,
                group_count,
                mProblemShape,
                mStrides,
                addr_i64,
                mScale,
                total_num_clusters,
                tmap_i64,
                max_active_clusters,
                stream,
            )

        @cute.kernel
        def fill_metadata(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddr: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx < group_count:
                g = tidx
                # Exclusive prefix sum of token counts as branch-free dataflow
                # (group_count <= a few hundred; each thread scans all groups).
                off = cutlass.Int32(0)
                for i in cutlass.range_constexpr(group_count):
                    pred = (cutlass.Int32(i) < g).to(cutlass.Int32)
                    off = off + mGroupSizes[i] * pred
                m_g = mGroupSizes[g]

                mProblemShape[g, 0] = m_g
                mProblemShape[g, 1] = cutlass.Int32(n)
                mProblemShape[g, 2] = cutlass.Int32(k)
                mProblemShape[g, 3] = cutlass.Int32(1)

                # A[M,K] k-major -> (K,1); B[N,K] k-major -> (K,1); C[M,N] n-major -> (N,1).
                mStrides[g, 0, 0] = cutlass.Int32(k)
                mStrides[g, 0, 1] = cutlass.Int32(1)
                mStrides[g, 1, 0] = cutlass.Int32(k)
                mStrides[g, 1, 1] = cutlass.Int32(1)
                mStrides[g, 2, 0] = cutlass.Int32(n)
                mStrides[g, 2, 1] = cutlass.Int32(1)

                off64 = cutlass.Int64(off)
                g64 = cutlass.Int64(g)
                base_a = cutlass.Int64(mA.iterator.toint())
                base_b = cutlass.Int64(mB.iterator.toint())
                base_c = cutlass.Int64(mC.iterator.toint())
                mAddr[g, 0] = base_a + off64 * (k * a_bytes)
                mAddr[g, 1] = base_b + g64 * (n * k * b_bytes)
                mAddr[g, 2] = base_c + off64 * (n * c_bytes)

    return TmaGroupedLauncher()


def _dtype_bytes(jax_dtype) -> int:
    return jnp.dtype(jax_dtype).itemsize


def tma_grouped_gemm(a, b, group_sizes, *, out_dtype, out_scale, cluster_shape_mn=(2, 1)):
    """Forward grouped GEMM ``a[M,K] . b[E,N,K] -> [M,N]`` (contract K) via the
    stock Hopper TMA warp-specialized persistent kernel.

    ``a``/``b`` are k-major 8-bit (E4M3 forward, E5M2xE4M3 dgrad). The epilogue
    DIVIDES the f32 accumulator by ``out_scale[0]`` (haliax dequantize
    convention). ``group_sizes`` may be traced (dynamic per-expert token counts).
    """
    ensure_hopper_arch()
    e, n, k = b.shape
    m = a.shape[0]
    a_dtype, b_dtype = a.dtype, b.dtype

    max_active_clusters = _H100_SMS // (cluster_shape_mn[0] * cluster_shape_mn[1])
    total_num_clusters = _total_num_clusters_upper_bound(m, n, cluster_shape_mn, e)

    launcher = _build_tma_launcher(
        cluster_shape_mn=cluster_shape_mn,
        group_count=e,
        n=n,
        k=k,
        total_num_clusters=total_num_clusters,
        max_active_clusters=max_active_clusters,
        a_bytes=_dtype_bytes(a_dtype),
        b_bytes=_dtype_bytes(b_dtype),
        c_bytes=_dtype_bytes(out_dtype),
    )

    ts = cjax.TensorSpec
    a_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    b_spec = ts(mode=(0, 1, 2), divisibility=(1, 1, _FP8_TMA_VEC), static=True)
    gs_spec = ts(mode=(0,), static=True)
    scale_spec = ts(mode=(0,), static=True)
    # Initials carry dtype + majorness only; CRITICAL: static=False. Static tiny
    # extents make CuTe canonicalize the size-1 Rest modes, collapsing tile-coord
    # math for every real tile beyond the dummy extent -> fast garbage output.
    init_a_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_b_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_c_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, 1), static=False)
    c_spec = ts(mode=(0, 1), divisibility=(1, 1), static=True)
    ps_spec = ts(mode=(0, 1), static=True)
    st_spec = ts(mode=(0, 1, 2), static=True)
    addr_spec = ts(mode=(0, 1), static=True)
    tmap_spec = ts(mode=(0, 1, 2), static=True)

    out_shapes = (
        jax.ShapeDtypeStruct((m, n), out_dtype),
        jax.ShapeDtypeStruct((e, 4), jnp.int32),  # problem_shape_mnkl
        jax.ShapeDtypeStruct((e, 3, 2), jnp.int32),  # strides_abc
        jax.ShapeDtypeStruct((e, 2 * _NUM_TENSORMAPS), jnp.int32),  # tensor_address (i32-doubled)
        jax.ShapeDtypeStruct(
            (_H100_SMS, _NUM_TENSORMAPS, 2 * _TENSORMAP_I64_WORDS), jnp.int32
        ),  # tensormap workspace (i32-doubled)
    )

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=out_shapes,
        input_spec=(a_spec, b_spec, gs_spec, scale_spec, init_a_spec, init_b_spec, init_c_spec),
        output_spec=(c_spec, ps_spec, st_spec, addr_spec, tmap_spec),
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_HOPPER_CUTE_ARCH),),
    )

    init_a = jnp.zeros((1, 128, 128), a_dtype)
    init_b = jnp.zeros((1, 128, 128), b_dtype)
    init_c = jnp.zeros((1, 128, 128), out_dtype)
    gs = group_sizes.astype(jnp.int32)
    out = call(a, b, gs, out_scale, init_a, init_b, init_c)
    return out[0]


# --------------------------------------------------------------------------- #
# Wgrad (token-M-contracting weight gradient) adapter
#
# ``drhs[g] = a_t[:, g-slice] @ b_t[:, g-slice]^T`` with the ragged TOKEN axis as
# the GEMM contraction. Unlike the forward, the group's data is the contraction
# slice of a SHARED packed buffer, so a per-group base-pointer advance (offsets[g]
# elements, 1 byte each for fp8) is NOT 16B-aligned. Instead A/B keep the aligned
# full-buffer base and the token offset is an element coordinate; the per-group
# descriptor contraction extent (offset+M_g) zero-fills the ragged tail via TMA.
# --------------------------------------------------------------------------- #


def _build_tma_wgrad_launcher(
    *,
    cluster_shape_mn: tuple[int, int],
    group_count: int,
    n: int,
    k_hidden: int,
    m_total: int,
    total_num_clusters: int,
    max_active_clusters: int,
    c_bytes: int,
):
    """Build the stream-first ``@cute.jit`` wgrad adapter launcher for ``cutlass_call``.

    The GEMM per group is ``(M=k_hidden, N=n, K=M_g)`` -- output rows are the hidden
    dim, output cols are ``n``, and the contraction is the ragged token count.
    """
    from haliax._src._tma_grouped_gemm import HopperGroupedGemmPersistentKernel  # noqa: PLC0415

    # Guard against silent desync with the vendored class's fixed attributes.
    assert HopperGroupedGemmPersistentKernel.bytes_per_tensormap == _TENSORMAP_BYTES
    assert HopperGroupedGemmPersistentKernel.num_tensormaps == _NUM_TENSORMAPS

    addr_threads = max(32, ((group_count + 31) // 32) * 32)

    class TmaWgradLauncher:
        @cute.jit
        def __call__(
            self,
            stream,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mScale: cute.Tensor,
            mInitA: cute.Tensor,
            mInitB: cute.Tensor,
            mInitC: cute.Tensor,
            mC: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddrI32: cute.Tensor,
            mTmapI32: cute.Tensor,
            mOffsets: cute.Tensor,
        ):
            addr_i64 = cute.recast_tensor(mAddrI32, cutlass.Int64)  # (E, 3)
            tmap_i64 = cute.recast_tensor(mTmapI32, cutlass.Int64)  # (num_sms, 3, 16)

            # 1) Device prologue: fill problem_shape / strides / addresses / offsets.
            self.fill_metadata(mA, mB, mC, mGroupSizes, mProblemShape, mStrides, addr_i64, mOffsets).launch(
                grid=[1, 1, 1], block=[addr_threads, 1, 1], stream=stream
            )

            # 2) Stock persistent TMA grouped GEMM in ragged-contraction (wgrad) mode.
            kernel = HopperGroupedGemmPersistentKernel(
                cutlass.Float32,
                _TILE_SHAPE_MN,
                cluster_shape_mn,
                tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
                wgrad=True,
            )
            kernel(
                mInitA,
                mInitB,
                mInitC,
                group_count,
                mProblemShape,
                mStrides,
                addr_i64,
                mScale,
                total_num_clusters,
                tmap_i64,
                max_active_clusters,
                stream,
                mOffsets,
            )

        @cute.kernel
        def fill_metadata(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mGroupSizes: cute.Tensor,
            mProblemShape: cute.Tensor,
            mStrides: cute.Tensor,
            mAddr: cute.Tensor,
            mOffsets: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx < group_count:
                g = tidx
                # Exclusive prefix sum of the 16-token-ROUNDED group sizes: every
                # group starts on a 16-token boundary so its TMA element coordinate
                # (folded offset) is 16B-aligned. Matches the host repack exactly.
                v = cutlass.Int32(_FP8_TMA_VEC)
                off = cutlass.Int32(0)
                for i in cutlass.range_constexpr(group_count):
                    pred = (cutlass.Int32(i) < g).to(cutlass.Int32)
                    padded_i = ((mGroupSizes[i] + v - cutlass.Int32(1)) // v) * v
                    off = off + padded_i * pred
                m_g = mGroupSizes[g]

                # GEMM problem per group: (M=k_hidden rows, N, K=M_g tokens, L=1).
                mProblemShape[g, 0] = cutlass.Int32(k_hidden)
                mProblemShape[g, 1] = cutlass.Int32(n)
                mProblemShape[g, 2] = m_g
                mProblemShape[g, 3] = cutlass.Int32(1)

                # A=a_t[k_hidden, m_total] token(=k)-major -> row stride m_total, k stride 1.
                # B=b_t[n, m_total] token(=k)-major -> row stride m_total, k stride 1.
                # C=out[k_hidden, n] n-major -> row stride n, col stride 1.
                mStrides[g, 0, 0] = cutlass.Int32(m_total)
                mStrides[g, 0, 1] = cutlass.Int32(1)
                mStrides[g, 1, 0] = cutlass.Int32(m_total)
                mStrides[g, 1, 1] = cutlass.Int32(1)
                mStrides[g, 2, 0] = cutlass.Int32(n)
                mStrides[g, 2, 1] = cutlass.Int32(1)

                mOffsets[g] = off

                g64 = cutlass.Int64(g)
                base_a = cutlass.Int64(mA.iterator.toint())
                base_b = cutlass.Int64(mB.iterator.toint())
                base_c = cutlass.Int64(mC.iterator.toint())
                # A/B share the aligned full-buffer base (offset folded into the TMA
                # coordinate); C advances one dense [k_hidden, n] slab per expert.
                mAddr[g, 0] = base_a
                mAddr[g, 1] = base_b
                mAddr[g, 2] = base_c + g64 * (k_hidden * n * c_bytes)

    return TmaWgradLauncher()


def _padded_group_offsets(group_sizes):
    """Exclusive-prefix token offsets after rounding each group up to 16 tokens.

    Returned offsets are all multiples of 16 (the TMA innermost-coordinate
    granularity for fp8). Mirrors the device prologue's offset accumulation so the
    host repack and the kernel agree on where each group starts.
    """
    gs = group_sizes.astype(jnp.int32)
    padded_sizes = ((gs + _FP8_TMA_VEC - 1) // _FP8_TMA_VEC) * _FP8_TMA_VEC
    dst_off = jnp.cumsum(padded_sizes) - padded_sizes  # exclusive prefix sum
    src_off = jnp.cumsum(gs) - gs
    return gs, dst_off, src_off


def _pad_token_groups_16(a_t, b_t, group_sizes, e: int, m: int):
    """Repack token-major ``a_t[K,M]``/``b_t[N,M]`` so each group starts on a
    16-token boundary, zero-filling the sub-16 gap after each group.

    Returns ``(a_pad, b_pad, m_total)`` where ``m_total`` is the static padded
    width (a multiple of 16). The pad columns are exact fp8 zero every call
    (gather ``mode='fill'``) so nothing stale leaks across XLA buffer reuse.
    """
    gs, dst_off, src_off = _padded_group_offsets(group_sizes)
    # Static worst case: each of the E groups adds <16 pad tokens; round the total
    # up to 16 so the padded row stride is itself 16B-aligned (TMA row start).
    m_total = ((m + _FP8_TMA_VEC * e + _FP8_TMA_VEC - 1) // _FP8_TMA_VEC) * _FP8_TMA_VEC
    positions = jnp.arange(m_total, dtype=jnp.int32)
    grp = jnp.clip(jnp.searchsorted(dst_off, positions, side="right") - 1, 0, e - 1)
    local = positions - dst_off[grp]
    is_real = local < gs[grp]
    # Out-of-bounds source index for pad slots -> gather fills 0.
    src = jnp.where(is_real, src_off[grp] + local, m)
    a_pad = jnp.take(a_t, src, axis=1, mode="fill", fill_value=0)
    b_pad = jnp.take(b_t, src, axis=1, mode="fill", fill_value=0)
    return a_pad, b_pad, m_total


def tma_grouped_wgrad(a_t, b_t, group_sizes, *, out_dtype, out_scale, cluster_shape_mn=(1, 1)):
    """Weight-gradient grouped GEMM ``a_t[K,M] . b_t[N,M] -> [E,K,N]`` contracting the
    ragged token axis M, via the stock Hopper TMA warp-specialized persistent kernel.

    ``a_t``/``b_t`` are token-major 8-bit (activations E4M3, output-grad E5M2). M is
    the packed, non-tile-aligned per-group contraction; the epilogue DIVIDES the f32
    accumulator by ``out_scale[0]`` (haliax dequantize convention). ``group_sizes``
    may be traced (dynamic per-expert token counts).

    Cluster ``(1,1)`` (no multicast): the GEMM M dim is the (fixed, often small) hidden
    dimension, which for K < cluster_tile_M would leave a fully out-of-range CTA in a
    B-multicast cluster (malformed launch). The multicast win is marginal here and the
    ragged axis is the contraction, so a single-CTA cluster is both safe and simplest.
    """
    ensure_hopper_arch()
    k_hidden, m = a_t.shape
    n = b_t.shape[0]
    e = group_sizes.shape[0]
    a_dtype, b_dtype = a_t.dtype, b_t.dtype

    # 16-token group padding (the TMA 16B innermost-coordinate constraint). Each
    # group's token slice is the GEMM contraction, folded into the A/B TMA element
    # coordinate; that coordinate -- the group's exclusive-prefix token offset --
    # must be a multiple of 16 fp8 elements (=16B) or the load faults. Repack the
    # token axis so every group starts on a 16-token boundary, zero-filling the
    # <16-token gap after each group. Zero pads add exact +0.0 to the f32
    # accumulator (the descriptor extent = off+M_g stops the load at the ragged
    # end anyway), so the result is bit-identical to the unpadded packing.
    a_t, b_t, m_total = _pad_token_groups_16(a_t, b_t, group_sizes, e, m)

    max_active_clusters = _H100_SMS // (cluster_shape_mn[0] * cluster_shape_mn[1])
    # The ragged axis is the contraction: the M/N (k_hidden, n) tile grid is uniform
    # per group, so the cluster-tile total is EXACT (no ragged-M surplus tiles).
    ctm, ctn = _cluster_tile_mn(cluster_shape_mn)
    total_num_clusters = e * ((k_hidden + ctm - 1) // ctm) * ((n + ctn - 1) // ctn)

    launcher = _build_tma_wgrad_launcher(
        cluster_shape_mn=cluster_shape_mn,
        group_count=e,
        n=n,
        k_hidden=k_hidden,
        m_total=m_total,
        total_num_clusters=total_num_clusters,
        max_active_clusters=max_active_clusters,
        c_bytes=_dtype_bytes(out_dtype),
    )

    ts = cjax.TensorSpec
    a_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    b_spec = ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True)
    gs_spec = ts(mode=(0,), static=True)
    scale_spec = ts(mode=(0,), static=True)
    init_a_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_b_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, _FP8_TMA_VEC), static=False)
    init_c_spec = ts(mode=(1, 2, 0), divisibility=(1, 1, 1), static=False)
    c_spec = ts(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    ps_spec = ts(mode=(0, 1), static=True)
    st_spec = ts(mode=(0, 1, 2), static=True)
    addr_spec = ts(mode=(0, 1), static=True)
    tmap_spec = ts(mode=(0, 1, 2), static=True)
    off_spec = ts(mode=(0,), static=True)

    out_shapes = (
        jax.ShapeDtypeStruct((e, k_hidden, n), out_dtype),
        jax.ShapeDtypeStruct((e, 4), jnp.int32),  # problem_shape_mnkl
        jax.ShapeDtypeStruct((e, 3, 2), jnp.int32),  # strides_abc
        jax.ShapeDtypeStruct((e, 2 * _NUM_TENSORMAPS), jnp.int32),  # tensor_address (i32-doubled)
        jax.ShapeDtypeStruct((_H100_SMS, _NUM_TENSORMAPS, 2 * _TENSORMAP_I64_WORDS), jnp.int32),  # tensormap ws
        jax.ShapeDtypeStruct((e,), jnp.int32),  # per-group token offsets
    )

    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=out_shapes,
        input_spec=(a_spec, b_spec, gs_spec, scale_spec, init_a_spec, init_b_spec, init_c_spec),
        output_spec=(c_spec, ps_spec, st_spec, addr_spec, tmap_spec, off_spec),
        use_static_tensors=True,
        compile_options=(cute.GPUArch(_HOPPER_CUTE_ARCH),),
    )

    init_a = jnp.zeros((1, 128, 128), a_dtype)
    init_b = jnp.zeros((1, 128, 128), b_dtype)
    init_c = jnp.zeros((1, 128, 128), out_dtype)
    gs = group_sizes.astype(jnp.int32)
    out = call(a_t, b_t, gs, out_scale, init_a, init_b, init_c)
    return out[0]
