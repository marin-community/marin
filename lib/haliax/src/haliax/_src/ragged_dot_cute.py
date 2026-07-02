# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""CuTe (CUTLASS Python DSL) FP8 grouped (ragged) matmul for Hopper H100.

The non-Mosaic backend for the FP8 ``ragged_dot``: a Hopper WGMMA grouped GEMM
authored in ``cutlass.cute`` and wrapped as a JAX custom call via
``cutlass.jax.cutlass_call``. Needs no forked jaxlib -- NVIDIA's WGMMA path
supports mixed E4M3/E5M2 out of the box. Dynamic per-expert token counts arrive
as a device ``problem_shape_mnkl`` tensor; copy atoms are ``CopyUniversalOp``
(never TMA) because ``cutlass_call`` operands are generic memrefs on the FFI
path (FA4 backend, _fa4_cute_kernels.py:319-336).

Kernel shape: this is a simple (non-persistent, non-warp-specialized) tiled
WGMMA GEMM. Two cooperative MMA warpgroups (256 threads) load each K tile
gmem->smem with a 128-bit ``CopyUniversalOp`` and issue ``wgmma`` from smem with
an FP32 accumulator. Groups map to ``blockIdx.z``; the un-tile-aligned per-group row
offset is handled by shifting the operand/output origin with ``domain_offset``
and predicating on the device-read token count ``M_g = problem_shape[g, 0]``.
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


def _problem_shape_mnkl(group_sizes: jax.Array, n: int, k: int) -> jax.Array:
    """(E,4) int32 device metadata of per-group (M=tokens_e, N, K, L=1)."""
    e = group_sizes.shape[0]
    return jnp.stack(
        [
            group_sizes.astype(jnp.int32),
            jnp.full((e,), n, jnp.int32),
            jnp.full((e,), k, jnp.int32),
            jnp.ones((e,), jnp.int32),
        ],
        axis=1,
    )


def _tensor_specs():
    """TensorSpecs for the cutlass_call operands and output.

    A ``[M, K]`` and B ``[E, N, K]`` are K-contiguous FP8; the K axis carries the
    128-bit vectorized copy so it is declared with FP8 divisibility. The int32
    metadata/offsets and the ``[1]`` float32 scale are plain static tensors.
    """
    vec = _UNIVERSAL_COPY_BITS // 8  # fp8 elems per 128-bit copy
    tensor_spec = cjax.TensorSpec
    a_spec = tensor_spec(mode=(0, 1), divisibility=(1, vec), static=True)
    b_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, vec), static=True)
    meta_spec = tensor_spec(mode=(0, 1), static=True)
    offsets_spec = tensor_spec(mode=(0,), static=True)
    scale_spec = tensor_spec(mode=(0,), static=True)
    c_spec = tensor_spec(mode=(0, 1), divisibility=(1, 1), static=True)
    return a_spec, b_spec, meta_spec, offsets_spec, scale_spec, c_spec


def _grouped_gemm_launcher(*, tile_shape_mn, group_count):
    """Build the ``@cute.jit`` WGMMA grouped-GEMM launcher for ``cutlass_call``.

    ``tile_shape_mn`` is the CTA tile (M, N); ``group_count`` (== number of
    experts) is a compile-time constant used for the grid Z extent and the
    per-group row-offset prefix sum.
    """
    sm90_utils = importlib.import_module("cutlass.utils.hopper_helpers")
    cuda = importlib.import_module("cuda.bindings.driver")

    tile_m, tile_n = tile_shape_mn
    acc_dtype = cutlass.Float32
    # Cooperative 2-warpgroup MMA (the proven Hopper config for a 128-row tile):
    # the two warpgroups split the M dimension 128 -> 2 x 64.
    atom_layout_mnk = (2, 1, 1)
    num_mma_warp_groups = 2
    num_threads = num_mma_warp_groups * 128

    class GroupedGemmFp8:
        @cute.jit
        def __call__(
            self,
            stream: cuda.CUstream,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mPS: cute.Tensor,
            mOff: cute.Tensor,
            mScale: cute.Tensor,
            mC: cute.Tensor,
        ):
            a_dtype = mA.element_type
            b_dtype = mB.element_type
            c_dtype = mC.element_type

            a_layout = cutlass.utils.LayoutEnum.from_tensor(mA)
            b_layout = cutlass.utils.LayoutEnum.from_tensor(mB[0, None, None])
            tiled_mma = sm90_utils.make_trivial_tiled_mma(
                a_dtype,
                b_dtype,
                a_layout.sm90_mma_major_mode(),
                b_layout.sm90_mma_major_mode(),
                acc_dtype,
                atom_layout_mnk,
                tiler_mn=(64, tile_n),
            )
            mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            tile_k = mma_inst_k * 4

            # Swizzled WGMMA-compatible smem layouts (single stage each).
            a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(a_layout, a_dtype, tile_k), a_dtype
            )
            b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(b_layout, b_dtype, tile_k), b_dtype
            )
            sA_layout = cute.tile_to_shape(a_atom, (tile_m, tile_k), order=(0, 1))
            sB_layout = cute.tile_to_shape(b_atom, (tile_n, tile_k), order=(0, 1))

            # Canonical Hopper epilogue smem: the WGMMA accumulator register layout is
            # not addressable by a naive register->smem copy, so the acc is stored one
            # epilogue sub-tile (epi_tile) at a time through a swizzled smem buffer with
            # the MMA C register->smem store atom (StMatrix for 16-bit), then read back
            # coalesced and stored to gmem (see hopper/dense_gemm.py, but with a
            # CopyUniversalOp s2g in place of the forbidden TMA store).
            c_layout_enum = cutlass.utils.LayoutEnum.from_tensor(mC)
            epi_tile = sm90_utils.compute_tile_shape_or_override(
                (tile_m, tile_n, tile_k), c_dtype, is_cooperative=True
            )
            sC_layout = sm90_utils.make_smem_layout_epi(c_dtype, c_layout_enum, epi_tile, 1)

            @cute.struct
            class SharedStorage:
                sA: cute.struct.Align[cute.struct.MemRange[a_dtype, cute.cosize(sA_layout)], 1024]
                sB: cute.struct.Align[cute.struct.MemRange[b_dtype, cute.cosize(sB_layout)], 1024]
                sC: cute.struct.Align[cute.struct.MemRange[c_dtype, cute.cosize(sC_layout)], 1024]

            # 128-bit CopyUniversalOp tiled copies (NOT TMA) for gmem->smem.
            a_vec = _UNIVERSAL_COPY_BITS // a_dtype.width
            b_vec = _UNIVERSAL_COPY_BITS // b_dtype.width
            a_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), a_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS)
            b_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), b_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS)
            a_k_vecs = tile_k // a_vec
            b_k_vecs = tile_k // b_vec
            tiled_copy_a = cute.make_tiled_copy_tv(
                a_copy,
                cute.make_layout((num_threads // a_k_vecs, a_k_vecs), stride=(a_k_vecs, 1)),
                cute.make_layout((1, a_vec)),
            )
            tiled_copy_b = cute.make_tiled_copy_tv(
                b_copy,
                cute.make_layout((num_threads // b_k_vecs, b_k_vecs), stride=(b_k_vecs, 1)),
                cute.make_layout((1, b_vec)),
            )

            # Epilogue register->smem copy: the MMA C register->smem store atom
            # (StMatrix for 16-bit output) mapped onto the tiled MMA's C layout so it
            # de-interleaves the WGMMA accumulator registers into the swizzled smem tile.
            copy_atom_r2s = sm90_utils.get_smem_store_op(c_layout_enum, c_dtype, acc_dtype)
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_r2s, tiled_mma)
            )
            # Epilogue smem->gmem copy (CopyUniversalOp, no TMA): 128-bit vectorized
            # along N over one epi_tile.
            c_vec = _UNIVERSAL_COPY_BITS // c_dtype.width
            c_n_vecs = epi_tile[1] // c_vec
            tiled_copy_s2g = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS),
                cute.make_layout((num_threads // c_n_vecs, c_n_vecs), stride=(c_n_vecs, 1)),
                cute.make_layout((1, c_vec)),
            )

            m_total = mA.shape[0]
            n = mB.shape[1]
            # Grouped-GEMM tile scheduler grid: the x dimension enumerates the linear
            # row-tile id across ALL groups, i.e. sum_e ceil_div(m_e, tile_m). The old
            # grid put the group in blockIdx.z and used ceil_div(m_total, tile_m) row
            # tiles PER group, so every group rescanned all m-tiles and ~(1 - 1/E) of
            # blocks ran the full WGMMA mainloop on zero-filled A tiles. Here each block
            # maps to exactly one real (group, row-tile) pair (see kernel), so no group
            # scans another group's rows.
            #
            # sum_e ceil_div(m_e, tile_m) <= ceil_div(m_total, tile_m) + (group_count-1)
            # (ceil(a)+ceil(b) <= ceil(a+b)+1), so this static bound covers every real
            # tile; the <= group_count-1 surplus blocks map to the last group with an
            # out-of-range row-tile and are zero-filled/predicated away like empty tiles.
            num_row_tiles = cute.ceil_div(m_total, tile_m) + (group_count - 1)
            grid = (
                num_row_tiles,
                cute.ceil_div(n, tile_n),
                1,
            )
            self.kernel(
                mA,
                mB,
                mPS,
                mOff,
                mScale,
                mC,
                sA_layout,
                sB_layout,
                sC_layout,
                tiled_copy_a,
                tiled_copy_b,
                tiled_copy_r2s,
                tiled_copy_s2g,
                tiled_mma,
                tile_m,
                tile_n,
                tile_k,
                epi_tile,
                a_dtype,
                b_dtype,
                c_dtype,
                SharedStorage,
            ).launch(grid=grid, block=[num_threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mPS: cute.Tensor,
            mOff: cute.Tensor,
            mScale: cute.Tensor,
            mC: cute.Tensor,
            sA_layout: cute.ComposedLayout,
            sB_layout: cute.ComposedLayout,
            sC_layout: cute.ComposedLayout,
            tiled_copy_a: cute.TiledCopy,
            tiled_copy_b: cute.TiledCopy,
            tiled_copy_r2s: cute.TiledCopy,
            tiled_copy_s2g: cute.TiledCopy,
            tiled_mma: cute.TiledMma,
            tile_m: cutlass.Constexpr,
            tile_n: cutlass.Constexpr,
            tile_k: cutlass.Constexpr,
            epi_tile: cutlass.Constexpr,
            a_dtype: cutlass.Constexpr,
            b_dtype: cutlass.Constexpr,
            c_dtype: cutlass.Constexpr,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            lin_tile, n_block, _ = cute.arch.block_idx()

            # Tile scheduler: map the linear row-tile id (blockIdx.x) to the group it
            # belongs to and the row-tile index within that group. Group gi owns the
            # linear range [running, running + ceil_div(m_gi, tile_m)); pick the last
            # group whose range starts at or before lin_tile (running is monotonic, so
            # this is the containing group for any in-range id). Surplus padding blocks
            # (lin_tile >= sum of all row-tiles) resolve to the last group with an
            # out-of-range m_block -> the A-load predicate zero-fills every row and the
            # store predicate suppresses the write, exactly like the reference kernel's
            # empty tiles (block-uniform integer arithmetic, no data-dependent WGMMA).
            g = cutlass.Int32(0)
            tile_base = cutlass.Int32(0)
            running = cutlass.Int32(0)
            for gi in cutlass.range_constexpr(group_count):
                if running <= lin_tile:
                    g = cutlass.Int32(gi)
                    tile_base = running
                running = running + cute.ceil_div(mPS[gi, 0], tile_m)
            m_block = lin_tile - tile_base

            m_g = mPS[g, 0]
            k = mPS[g, 2]

            # Packed row offset of group g in the [M, K]/[M, N] tensors.
            m_offset = mOff[g]

            mA_g = cute.domain_offset((m_offset, 0), mA)
            mC_g = cute.domain_offset((m_offset, 0), mC)
            mB_g = mB[g, None, None]

            gA = cute.local_tile(mA_g, (tile_m, tile_k), (m_block, None))
            gB = cute.local_tile(mB_g, (tile_n, tile_k), (n_block, None))
            gC = cute.local_tile(mC_g, (tile_m, tile_n), (m_block, n_block))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            # Attach the swizzle to the smem pointer (keep the tensor layout
            # affine) so WGMMA make_fragment_A/B accepts it.
            sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
            sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

            thr_copy_a = tiled_copy_a.get_slice(tidx)
            thr_copy_b = tiled_copy_b.get_slice(tidx)
            tAgA = thr_copy_a.partition_S(gA)
            tAsA = thr_copy_a.partition_D(sA)
            tBgB = thr_copy_b.partition_S(gB)
            tBsB = thr_copy_b.partition_D(sB)

            # Row predicate for the A load: a group's token count is not tile-aligned,
            # so the last M tile would otherwise read rows past the group (OOB).
            cA = cute.make_identity_tensor((tile_m, tile_k))
            tAcA = thr_copy_a.partition_S(cA)
            a_rest_m = cute.size(tAgA, mode=[1])

            # WGMMA is warpgroup-collective: slice the tiled MMA by warpgroup
            # index (which of the 2 cooperative warpgroups), not raw thread index.
            warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
            wg_thread_layout = cute.make_layout(2, stride=128)
            thr_mma = tiled_mma.get_slice(wg_thread_layout(warp_group_idx))
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)

            gC_mma = thr_mma.partition_C(gC)
            acc = cute.make_rmem_tensor(gC_mma.shape, cutlass.Float32)

            num_k_tiles = cute.ceil_div(k, tile_k)
            num_k_blocks = cute.size(tCrA, mode=[2])

            # Prime the WGMMA accumulator on the first instruction (ACCUMULATE=False
            # => D = A*B, ignoring uninitialized registers), then accumulate.
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                for rm in cutlass.range_constexpr(a_rest_m):
                    if m_block * tile_m + tAcA[0, rm, 0][0] < m_g:
                        cute.copy(tiled_copy_a, tAgA[None, rm, None, k_tile], tAsA[None, rm, None])
                    else:
                        tAsA[None, rm, None].fill(0)
                cute.copy(tiled_copy_b, tBgB[None, None, None, k_tile], tBsB)
                cute.arch.sync_threads()
                # A/B are filled with SIMT st.shared (CopyUniversalOp) in the generic
                # proxy, but wgmma reads its smem operands through the async proxy. A
                # generic-proxy write is NOT visible to an async-proxy read without a
                # fence.proxy.async (sync_threads only orders generic-proxy accesses and
                # warpgroup.fence only orders accumulator registers). Without it wgmma
                # occasionally reads uninitialized smem -> nondeterministic garbage, only
                # exposed at scale (many k-tiles/blocks). cp.async/TMA-filled kernels do
                # not need this because they write through the async proxy already.
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                cute.arch.sync_threads()

            # Epilogue: scale the f32 accumulator, then stream it to gmem one epilogue
            # sub-tile (epi_tile, N-chunked) at a time. The WGMMA accumulator register
            # layout is de-interleaved through tiled_copy_r2s (built from the MMA C
            # layout): each chunk goes acc(rmem) -> sC(smem) -> registers -> gmem (a
            # universal SIMT copy cannot move smem->gmem directly). The final store is
            # predicated per-row on the (non-tile-aligned) group token count.
            out_scale = mScale[0]
            acc.store(acc.load() / out_scale)

            sC = storage.sC.get_tensor(sC_layout.outer, swizzle=sC_layout.inner)
            thr_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(acc)
            rD_layout = cute.make_layout(cute.shape(thr_r2s.partition_S(sC))[:3])
            chunk = cute.size(rD_layout)

            cCid = cute.make_identity_tensor((tile_m, tile_n))
            epi_num = tile_n // epi_tile[1]
            thr_s2g = tiled_copy_s2g.get_slice(tidx)

            for epi_idx in cutlass.range_constexpr(epi_num):
                rAcc = cute.make_rmem_tensor(rD_layout, cutlass.Float32)
                for v in cutlass.range_constexpr(chunk):
                    rAcc[v] = tRS_rAcc[epi_idx * chunk + v]
                rC = cute.make_rmem_tensor(rD_layout, c_dtype)
                rC.store(rAcc.load().to(c_dtype))
                cute.copy(tiled_copy_r2s, rC, tRS_sC[None, None, None, 0])
                # StMatrix writes go through the async proxy; fence + barrier so the
                # generic-proxy s2g load sees them (else uninitialized smem -> garbage).
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()

                gC_chunk = cute.local_tile(gC, (tile_m, epi_tile[1]), (0, epi_idx))
                cC_chunk = cute.local_tile(cCid, (tile_m, epi_tile[1]), (0, epi_idx))
                sC_chunk = sC[None, None, 0]
                tSG_sC = thr_s2g.partition_S(sC_chunk)
                tSG_gC = thr_s2g.partition_D(gC_chunk)
                tSG_cC = thr_s2g.partition_S(cC_chunk)
                tSG_rC = cute.make_fragment_like(tSG_sC)
                cute.autovec_copy(tSG_sC, tSG_rC)
                for rm in cutlass.range_constexpr(cute.size(tSG_gC, mode=[1])):
                    if m_block * tile_m + tSG_cC[0, rm, 0][0] < m_g:
                        cute.copy(tiled_copy_s2g, tSG_rC[None, rm, None], tSG_gC[None, rm, None])
                cute.arch.sync_threads()

    return GroupedGemmFp8()


def cute_ragged_dot(a, b, group_sizes, *, out_dtype, out_scale):
    """Grouped GEMM ``a[M,K] . b[E,N,K] -> [M,N]`` contracting the last axis K
    (contiguous for both). Epilogue divides the f32 accumulator by ``out_scale``."""
    if not cute_available():
        raise RuntimeError("cute_ragged_dot requires the CuTe DSL on a GPU backend")
    tile_shape_mn = (128, 256)
    e, n, k = b.shape
    if n % tile_shape_mn[1] != 0:
        raise ValueError(
            f"cute_ragged_dot: N={n} is not divisible by tile_n={tile_shape_mn[1]}; " "pad or reshape b before calling"
        )
    if k % _FP8_WGMMA_K_GRANULARITY != 0:
        raise ValueError(
            f"cute_ragged_dot: K={k} is not divisible by {_FP8_WGMMA_K_GRANULARITY} "
            "(FP8 WGMMA K granularity); pad or reshape before calling"
        )
    m = a.shape[0]
    meta = _problem_shape_mnkl(group_sizes, n, k)
    # Packed per-group row offsets (exclusive prefix sum of token counts).
    offsets = (jnp.cumsum(group_sizes.astype(jnp.int32)) - group_sizes.astype(jnp.int32)).astype(jnp.int32)
    launcher = _grouped_gemm_launcher(tile_shape_mn=tile_shape_mn, group_count=e)
    a_spec, b_spec, meta_spec, offsets_spec, scale_spec, c_spec = _tensor_specs()
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((m, n), out_dtype),
        input_spec=(a_spec, b_spec, meta_spec, offsets_spec, scale_spec),
        output_spec=(c_spec,),
        use_static_tensors=True,
    )
    return call(a, b, meta, offsets, out_scale)
