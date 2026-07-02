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


def _wgrad_tensor_specs():
    """TensorSpecs for the wgrad ``cutlass_call`` operands and output.

    A ``a_t[K,M]`` and B ``b_t[N,M]`` are token-major FP8 (the contraction M is
    the contiguous minor axis, padded to a multiple of the tile so the FAST
    globally-aligned 128-bit load is legal). The M axis carries the 16-element
    (128-bit) vectorized copy, so it is declared with FP8 divisibility. The int32
    metadata/offsets and the ``[1]`` float32 scale are plain static tensors; the
    ``[E,K,N]`` output is declared like the forward's.
    """
    vec = _UNIVERSAL_COPY_BITS // 8  # fp8 elems per 128-bit copy
    tensor_spec = cjax.TensorSpec
    a_spec = tensor_spec(mode=(0, 1), divisibility=(1, vec), static=True)
    b_spec = tensor_spec(mode=(0, 1), divisibility=(1, vec), static=True)
    meta_spec = tensor_spec(mode=(0, 1), static=True)
    offsets_spec = tensor_spec(mode=(0,), static=True)
    scale_spec = tensor_spec(mode=(0,), static=True)
    c_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    return a_spec, b_spec, meta_spec, offsets_spec, scale_spec, c_spec


def _wgrad_gemm_launcher(*, tile_shape_mn, group_count):
    """Build the ``@cute.jit`` WGMMA launcher for the token-M-contracting wgrad.

    Computes ``mC[g,k,n] = sum_{m in group g} mA[k,m]*mB[n,m]`` where the token
    axis M is the (variable, non-tile-aligned) contraction. Reuses the forward
    kernel's smem layouts / WGMMA config / StMatrix epilogue verbatim; only the
    tensor indexing and the mainloop load differ:

    * The output tiles per group are UNIFORM (K,N are static and tile-aligned),
      so the group maps to ``blockIdx.z`` directly (no cross-group tile
      scheduler needed) and the epilogue stores the full tile (no row predicate).
    * The contraction length ``M_g`` is dynamic and NOT tile-aligned. The kernel
      iterates GLOBALLY tile-aligned token tiles (never ``domain_offset``-ing the
      contiguous contraction, which would break the 128-bit load alignment). A
      tile fully inside a group's ``[m_offset, m_offset+M_g)`` range takes the
      FAST 128-bit vectorized load; the (at most two) boundary tiles take a
      vector-1 element-predicated load (two-sided ``m_offset <= m < m_offset+M_g``)
      so no token of an adjacent group is dropped or contaminated.

    ``tile_shape_mn`` tiles the output (M=K rows, N cols); ``group_count`` is the
    grid Z extent.
    """
    sm90_utils = importlib.import_module("cutlass.utils.hopper_helpers")
    cuda = importlib.import_module("cuda.bindings.driver")

    tile_m, tile_n = tile_shape_mn
    acc_dtype = cutlass.Float32
    atom_layout_mnk = (2, 1, 1)
    num_mma_warp_groups = 2
    num_threads = num_mma_warp_groups * 128

    class WgradGemmFp8:
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
            b_layout = cutlass.utils.LayoutEnum.from_tensor(mB)
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

            a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(a_layout, a_dtype, tile_k), a_dtype
            )
            b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(b_layout, b_dtype, tile_k), b_dtype
            )
            sA_layout = cute.tile_to_shape(a_atom, (tile_m, tile_k), order=(0, 1))
            sB_layout = cute.tile_to_shape(b_atom, (tile_n, tile_k), order=(0, 1))

            c_layout_enum = cutlass.utils.LayoutEnum.from_tensor(mC[0, None, None])
            epi_tile = sm90_utils.compute_tile_shape_or_override(
                (tile_m, tile_n, tile_k), c_dtype, is_cooperative=True
            )
            sC_layout = sm90_utils.make_smem_layout_epi(c_dtype, c_layout_enum, epi_tile, 1)

            @cute.struct
            class SharedStorage:
                sA: cute.struct.Align[cute.struct.MemRange[a_dtype, cute.cosize(sA_layout)], 1024]
                sB: cute.struct.Align[cute.struct.MemRange[b_dtype, cute.cosize(sB_layout)], 1024]
                sC: cute.struct.Align[cute.struct.MemRange[c_dtype, cute.cosize(sC_layout)], 1024]

            # Two gmem->smem load paths along the contraction M:
            #  * FAST: 128-bit vectorized copy (as in the forward kernel) for tiles
            #    fully within M_g. Real MoE token counts are typically multiples of
            #    tile_k, so this covers essentially all the contraction.
            #  * TAIL: element-granular (vector = 1) predicated copy for the single
            #    boundary tile where M_g is not tile-aligned -- a vectorized copy
            #    there would drop or contaminate boundary tokens of adjacent groups.
            a_vec = _UNIVERSAL_COPY_BITS // a_dtype.width
            b_vec = _UNIVERSAL_COPY_BITS // b_dtype.width
            a_k_vecs = tile_k // a_vec
            b_k_vecs = tile_k // b_vec
            tiled_copy_a_fast = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), a_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS),
                cute.make_layout((num_threads // a_k_vecs, a_k_vecs), stride=(a_k_vecs, 1)),
                cute.make_layout((1, a_vec)),
            )
            tiled_copy_b_fast = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), b_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS),
                cute.make_layout((num_threads // b_k_vecs, b_k_vecs), stride=(b_k_vecs, 1)),
                cute.make_layout((1, b_vec)),
            )
            # Tail copies put the (contiguous) contraction on the fast thread axis so
            # the vector-1 loads still coalesce across threads.
            tiled_copy_a_tail = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), a_dtype, num_bits_per_copy=a_dtype.width),
                cute.make_layout((num_threads // tile_k, tile_k), stride=(tile_k, 1)),
                cute.make_layout((1, 1)),
            )
            tiled_copy_b_tail = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), b_dtype, num_bits_per_copy=b_dtype.width),
                cute.make_layout((num_threads // tile_k, tile_k), stride=(tile_k, 1)),
                cute.make_layout((1, 1)),
            )

            copy_atom_r2s = sm90_utils.get_smem_store_op(c_layout_enum, c_dtype, acc_dtype)
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_r2s, tiled_mma)
            )
            c_vec = _UNIVERSAL_COPY_BITS // c_dtype.width
            c_n_vecs = epi_tile[1] // c_vec
            tiled_copy_s2g = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=_UNIVERSAL_COPY_BITS),
                cute.make_layout((num_threads // c_n_vecs, c_n_vecs), stride=(c_n_vecs, 1)),
                cute.make_layout((1, c_vec)),
            )

            k = mA.shape[0]  # output rows (hidden), == mC.shape[1]
            n = mB.shape[0]  # output cols, == mC.shape[2]
            grid = (
                cute.ceil_div(k, tile_m),
                cute.ceil_div(n, tile_n),
                group_count,
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
                tiled_copy_a_fast,
                tiled_copy_b_fast,
                tiled_copy_a_tail,
                tiled_copy_b_tail,
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
            tiled_copy_a_fast: cute.TiledCopy,
            tiled_copy_b_fast: cute.TiledCopy,
            tiled_copy_a_tail: cute.TiledCopy,
            tiled_copy_b_tail: cute.TiledCopy,
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
            k_block, n_block, g = cute.arch.block_idx()

            # Group g owns the full [K,N] output slab mC[g]; its tokens are the
            # packed contraction slice [m_offset, m_offset+M_g) of A[K,M]/B[N,M].
            m_g = mPS[g, 0]
            m_offset = mOff[g]

            mC_g = mC[g, None, None]

            # Tile A/B over the WHOLE (padded) token axis at global 128-aligned
            # boundaries -- never domain_offset the contiguous contraction, which
            # would break the FAST 128-bit load's 16-byte alignment.
            gA = cute.local_tile(mA, (tile_m, tile_k), (k_block, None))
            gB = cute.local_tile(mB, (tile_n, tile_k), (n_block, None))
            gC = cute.local_tile(mC_g, (tile_m, tile_n), (k_block, n_block))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
            sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

            # Fast (vectorized) and tail (vector-1, predicated) load partitions.
            thr_a_fast = tiled_copy_a_fast.get_slice(tidx)
            thr_b_fast = tiled_copy_b_fast.get_slice(tidx)
            tAgA_f = thr_a_fast.partition_S(gA)
            tAsA_f = thr_a_fast.partition_D(sA)
            tBgB_f = thr_b_fast.partition_S(gB)
            tBsB_f = thr_b_fast.partition_D(sB)

            thr_a_tail = tiled_copy_a_tail.get_slice(tidx)
            thr_b_tail = tiled_copy_b_tail.get_slice(tidx)
            tAgA_t = thr_a_tail.partition_S(gA)
            tAsA_t = thr_a_tail.partition_D(sA)
            tBgB_t = thr_b_tail.partition_S(gB)
            tBsB_t = thr_b_tail.partition_D(sB)
            # Token (tile_k) coordinate for the tail predicate.
            cA = cute.make_identity_tensor((tile_m, tile_k))
            tAcA = thr_a_tail.partition_S(cA)
            cB = cute.make_identity_tensor((tile_n, tile_k))
            tBcB = thr_b_tail.partition_S(cB)
            a_rest_m = cute.size(tAgA_t, mode=[1])
            a_rest_k = cute.size(tAgA_t, mode=[2])
            b_rest_n = cute.size(tBgB_t, mode=[1])
            b_rest_k = cute.size(tBgB_t, mode=[2])

            warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
            wg_thread_layout = cute.make_layout(2, stride=128)
            thr_mma = tiled_mma.get_slice(wg_thread_layout(warp_group_idx))
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)

            gC_mma = thr_mma.partition_C(gC)
            acc = cute.make_rmem_tensor(gC_mma.shape, cutlass.Float32)
            # Accumulate onto a zeroed acc (ACCUMULATE always True): an empty group
            # (M_g==0) runs no tile and correctly stores zeros.
            acc.fill(0.0)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            num_k_blocks = cute.size(tCrA, mode=[2])
            m_end = m_offset + m_g
            first_tile = m_offset // tile_k
            last_tile = (m_end - 1) // tile_k  # inclusive; empty groups yield <= first_tile-1
            n_tiles = last_tile - first_tile + 1

            for i in cutlass.range(n_tiles, unroll=1):
                t = first_tile + i
                lo = t * tile_k
                if lo >= m_offset and lo + tile_k <= m_end:
                    # Tile fully inside the group: fast 128-bit vectorized load.
                    cute.copy(tiled_copy_a_fast, tAgA_f[None, None, None, t], tAsA_f)
                    cute.copy(tiled_copy_b_fast, tBgB_f[None, None, None, t], tBsB_f)
                else:
                    # Boundary tile: vector-1 load, two-sided per-token predicate.
                    for rk in cutlass.range_constexpr(a_rest_k):
                        for rm in cutlass.range_constexpr(a_rest_m):
                            gm = lo + tAcA[0, rm, rk][1]
                            if m_offset <= gm and gm < m_end:
                                cute.copy(tiled_copy_a_tail, tAgA_t[None, rm, rk, t], tAsA_t[None, rm, rk])
                            else:
                                tAsA_t[None, rm, rk].fill(0)
                    for rk in cutlass.range_constexpr(b_rest_k):
                        for rn in cutlass.range_constexpr(b_rest_n):
                            gm = lo + tBcB[0, rn, rk][1]
                            if m_offset <= gm and gm < m_end:
                                cute.copy(tiled_copy_b_tail, tBgB_t[None, rn, rk, t], tBsB_t[None, rn, rk])
                            else:
                                tBsB_t[None, rn, rk].fill(0)
                cute.arch.sync_threads()
                # Generic-proxy (SIMT st.shared) writes are not visible to the
                # async-proxy WGMMA smem read without this fence (see the forward
                # kernel); omitting it yields nondeterministic garbage at scale.
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                cute.arch.sync_threads()

            # Epilogue: identical to the forward (scale, r2s StMatrix, s2g), but
            # the output [K,N] is fully tile-aligned so every row is stored.
            out_scale = mScale[0]
            acc.store(acc.load() / out_scale)

            sC = storage.sC.get_tensor(sC_layout.outer, swizzle=sC_layout.inner)
            thr_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(acc)
            rD_layout = cute.make_layout(cute.shape(thr_r2s.partition_S(sC))[:3])
            chunk = cute.size(rD_layout)

            epi_num = tile_n // epi_tile[1]
            thr_s2g = tiled_copy_s2g.get_slice(tidx)

            for epi_idx in cutlass.range_constexpr(epi_num):
                rAcc = cute.make_rmem_tensor(rD_layout, cutlass.Float32)
                for v in cutlass.range_constexpr(chunk):
                    rAcc[v] = tRS_rAcc[epi_idx * chunk + v]
                rC = cute.make_rmem_tensor(rD_layout, c_dtype)
                rC.store(rAcc.load().to(c_dtype))
                cute.copy(tiled_copy_r2s, rC, tRS_sC[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()

                gC_chunk = cute.local_tile(gC, (tile_m, epi_tile[1]), (0, epi_idx))
                sC_chunk = sC[None, None, 0]
                tSG_sC = thr_s2g.partition_S(sC_chunk)
                tSG_gC = thr_s2g.partition_D(gC_chunk)
                tSG_rC = cute.make_fragment_like(tSG_sC)
                cute.autovec_copy(tSG_sC, tSG_rC)
                for rm in cutlass.range_constexpr(cute.size(tSG_gC, mode=[1])):
                    cute.copy(tiled_copy_s2g, tSG_rC[None, rm, None], tSG_gC[None, rm, None])
                cute.arch.sync_threads()

    return WgradGemmFp8()


# Fused cast-transpose: one CTA reads a (tile_m tokens x tile_n) bf16 sub-tile and
# emits its row-major FP8 (coalesced 128-bit read + vectorized fp8 store) AND its
# token-major FP8 transpose. The transpose is staged through shared memory so BOTH
# gmem writes are coalesced (threads write mYt contiguously along the token axis M),
# instead of an uncoalesced element scatter.
_CAST_TRANSPOSE_TILE = 64
_CAST_TRANSPOSE_VEC = _UNIVERSAL_COPY_BITS // 16  # bf16 elems per 128-bit read (== 8)


def _cast_transpose_launcher(*, tile, out_maxv, m_vec_aligned):
    """Build the ``@cute.jit`` fused cast-transpose launcher for ``cutlass_call``.

    Reads bf16 ``mX[M,N]`` once and writes ``mY[M,N]`` (row-major FP8, bit-identical
    to ``quantize(x, out_dtype, scale)``) and ``mYt[N,M]`` (its transpose). The
    quantize divides by ``mScale[0]`` and clips to ``[-out_maxv, out_maxv]`` before
    the FP8 cast, matching ``_src/fp8.quantize`` exactly. ``out_maxv`` is the FP8
    dtype's finite max (448 for E4M3, 57344 for E5M2). ``m_vec_aligned`` (M % vec == 0)
    selects the coalesced vectorized token-major store; otherwise a per-element store
    keeps arbitrary M correct (row bases of ``mYt`` are unaligned when M % vec != 0).
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

            m_total = mX.shape[0]
            n_total = mX.shape[1]
            grid = (cute.ceil_div(m_total, tile), cute.ceil_div(n_total, tile), 1)
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
            bm, bn, _ = cute.arch.block_idx()
            m_total = mX.shape[0]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            # sQ holds the quantized [tile_m, tile_n] FP8 sub-tile; sQt views the same
            # bytes transposed ([tile_n, tile_m]) for the coalesced token-major store.
            sQ = storage.sQ.get_tensor(cute.make_layout((tile, tile), stride=(tile, 1)))
            sQt = storage.sQ.get_tensor(cute.make_layout((tile, tile), stride=(1, tile)))

            thr_in = tiled_copy_in.get_slice(tidx)
            thr_fp8 = fp8_copy.get_slice(tidx)

            # --- Phase 1: coalesced bf16 read -> quantize -> row-major FP8 + smem stage.
            gX = cute.local_tile(mX, (tile, tile), (bm, bn))
            gY = cute.local_tile(mY, (tile, tile), (bm, bn))
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
            gYt = cute.local_tile(mYt, (tile, tile), (bn, bm))
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
                            mYt[gn, base_m + v] = frag_t[v, rn, 0]

    return CastTransposeFp8()


def cute_cast_transpose(x, scale, *, out_dtype):
    """Fused single-read cast-transpose of bf16 ``x[M,N]`` to FP8.

    Returns ``(row_major[M,N], token_major[N,M])`` where ``row_major`` is bit-identical
    to ``quantize(x, out_dtype, scale, jnp.float32)`` and ``token_major`` is its
    transpose. One HBM read of ``x`` produces both layouts (plus the FP8 quantize),
    replacing the pure-JAX ``quantize + swapaxes`` two-pass. Requires N divisible by
    the tile and a bf16 input (guarded by the caller in ``cast_transpose``)."""
    if not cute_available():
        raise RuntimeError("cute_cast_transpose requires the CuTe DSL on a GPU backend")
    tile = _CAST_TRANSPOSE_TILE
    m, n = x.shape
    if n % tile != 0:
        raise ValueError(f"cute_cast_transpose: N={n} is not divisible by tile={tile}")
    out_maxv = float(jnp.finfo(out_dtype).max)
    scale_1 = jnp.reshape(scale.astype(jnp.float32), (1,))
    launcher = _cast_transpose_launcher(tile=tile, out_maxv=out_maxv, m_vec_aligned=(m % _CAST_TRANSPOSE_VEC == 0))
    x_spec = cjax.TensorSpec(mode=(0, 1), divisibility=(1, _CAST_TRANSPOSE_VEC), static=True)
    scale_spec = cjax.TensorSpec(mode=(0,), static=True)
    y_spec = cjax.TensorSpec(mode=(0, 1), divisibility=(1, 1), static=True)
    yt_spec = cjax.TensorSpec(mode=(0, 1), divisibility=(1, 1), static=True)
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=(jax.ShapeDtypeStruct((m, n), out_dtype), jax.ShapeDtypeStruct((n, m), out_dtype)),
        input_spec=(x_spec, scale_spec),
        output_spec=(y_spec, yt_spec),
        use_static_tensors=True,
    )
    return call(x, scale_1)


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


def cute_wgrad(a_t, b_t, group_sizes, *, out_dtype, out_scale):
    """Grouped wgrad ``a_t[K,M] . b_t[N,M] -> [E,K,N]`` contracting the token axis M.

    ``a_t``/``b_t`` are token-major FP8 (cast-transposed activations E4M3 and
    output-grad E5M2); M is the packed, non-tile-aligned per-group contraction.
    Produces one ``[K,N]`` weight-gradient slab per expert. The epilogue divides
    the f32 accumulator by ``out_scale`` (the reciprocal of the operand scale
    product, matching the dequantize convention)."""
    if not cute_available():
        raise RuntimeError("cute_wgrad requires the CuTe DSL on a GPU backend")
    tile_shape_mn = (128, 256)
    tile_k = _FP8_WGMMA_K_GRANULARITY
    k, m = a_t.shape
    n = b_t.shape[0]
    e = group_sizes.shape[0]
    if k % tile_shape_mn[0] != 0:
        raise ValueError(
            f"cute_wgrad: K={k} is not divisible by tile_m={tile_shape_mn[0]}; pad or reshape before calling"
        )
    if n % tile_shape_mn[1] != 0:
        raise ValueError(
            f"cute_wgrad: N={n} is not divisible by tile_n={tile_shape_mn[1]}; pad or reshape before calling"
        )
    # Pad the contraction M up to a tile_k multiple with zeros (beyond the last
    # group, so no group's contraction changes) so the base M extent is 16-aligned
    # and the globally-tile-aligned FAST 128-bit load is legal.
    m_pad = ((m + tile_k - 1) // tile_k) * tile_k
    if m_pad != m:
        a_t = jnp.pad(a_t, ((0, 0), (0, m_pad - m)))
        b_t = jnp.pad(b_t, ((0, 0), (0, m_pad - m)))
    # meta carries per-group token counts (contraction M) in mode 0; offsets are
    # the packed exclusive prefix sum of token counts along M.
    meta = _problem_shape_mnkl(group_sizes, n, k)
    offsets = (jnp.cumsum(group_sizes.astype(jnp.int32)) - group_sizes.astype(jnp.int32)).astype(jnp.int32)
    launcher = _wgrad_gemm_launcher(tile_shape_mn=tile_shape_mn, group_count=e)
    a_spec, b_spec, meta_spec, offsets_spec, scale_spec, c_spec = _wgrad_tensor_specs()
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((e, k, n), out_dtype),
        input_spec=(a_spec, b_spec, meta_spec, offsets_spec, scale_spec),
        output_spec=(c_spec,),
        use_static_tensors=True,
    )
    return call(a_t, b_t, meta, offsets, out_scale)
