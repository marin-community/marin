# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored and lightly modified from JAX (Apache-2.0):
#   jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py
#   jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py
# (jax 0.10.0).  These Pallas-Mosaic-GPU grouped matmuls drive the Hopper FP8
# tensor cores (wgmma.fp8) and handle genuinely *non-uniform, dynamic*
# group_sizes.  Modifications vs upstream:
#   * `out_dtype` parameter so an FP8 contraction can emit a BF16/FP32 result
#     (upstream hardcodes the output dtype to ``lhs.dtype``);
#   * the transposed kernel's same-dtype guard relaxed to allow the mixed
#     E5M2 x E4M3 backward (Hopper wgmma takes independent FP8 operand types);
#   * the transposed kernel's group-boundary masking rewritten with
#     ``jnp.where`` because the upstream ``bool -> fp8`` cast is unsupported.
# The ``main``/``ref_``/profiling helpers are dropped.

import dataclasses
import functools
import math

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

# On stock jaxlib, Mosaic ``wgmma`` requires both operands to share a dtype, so
# every FP8 GEMM driven through this kernel is same-dtype ``e4m3 x e4m3``. The
# ``mgpu_dwgrad`` path below is written to also accept a mixed ``e5m2 x e4m3``
# pair (Hopper ``wgmma`` takes independent FP8 operand ``.atype``/``.btype``),
# which a follow-up that relaxes the operand-dtype guard will exercise; it is
# not called on the same-dtype code paths used here.


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        """Get the group info for the current block."""

        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]

        # We usually only have very few groups, so we unroll the loop processing
        # them. Normally we'd break out of the loop early, once we'd have found our
        # boundary, but we can't do that when unrolling, so we rely on many selects
        # to mask out the epilogue of the loop.
        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, b in enumerate(group_boundaries):
            # Start/end are inclusive
            start = end
            end = start + b
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
            # How many blocks after is our block?
            this_is_group = (tid_begin <= tid) & (tid < tid_end)
            block = lax.select(this_is_group, tid - tid_begin + start_block, block)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(
            group_id=group,
            block=block,
            block_start=block_start,
            actual_start=actual_start,
            actual_end=actual_end,
            start_within_block=start_within_block,
            actual_size=actual_size,
        )


def mgpu_ragged_dot(
    lhs,  # (M, K)
    rhs,  # (G, K, N)
    *,
    group_sizes,  # (G,)
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
    transpose_rhs: bool = False,
    load_group_sizes_to_register: bool = True,
    out_dtype=None,
    out_scale=None,
) -> jax.Array:
    # lhs and rhs share a dtype, except that the e4m3/e5m2 FP8 pair may be mixed:
    # Hopper `wgmma` takes independent `.atype`/`.btype` FP8 operands (the dgrad of
    # the TE-style hybrid is e5m2-grad x e4m3-rhs). Mirrors the relaxation in
    # `jax/_src/pallas/mosaic_gpu/primitives.py:wgmma` and the Mosaic dialect verifier.
    _fp8_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    _is_mixed_fp8 = lhs.dtype in _fp8_dtypes and rhs.dtype in _fp8_dtypes
    if lhs.dtype != rhs.dtype and not _is_mixed_fp8:
        raise NotImplementedError(f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}")
    m, k = lhs.shape
    g, k2, n = rhs.shape
    _od = lhs.dtype if out_dtype is None else out_dtype
    # Per-tensor dequant scale, folded into the output store (saves a full pass).
    _scale = jnp.ones((1,), jnp.float32) if out_scale is None else jnp.asarray(out_scale, jnp.float32).reshape(1)

    if transpose_rhs:
        k2, n = n, k2

    if group_sizes.shape[0] != g:
        raise ValueError(f"Expected group_sizes to have shape {g} but got {group_sizes.shape}")

    if k != k2:
        raise ValueError(f"lhs.shape={k} must match rhs.shape={k2}")

    if k % block_k != 0:
        raise ValueError(f"k={k} must be a multiple of block_k={block_k}")

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, scale_gmem, o_gmem):
        grid_m = pl.cdiv(m, block_m) + g - 1
        grid_n = pl.cdiv(n, block_n)
        grid = (grid_m * grid_n,)

        @plgpu.nd_loop(grid, collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(
                loop_info.index[0],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )
            group_info = GroupInfo.create(rows_per_expert_gmem, block_m, mi)

            def acc_scope(acc_ref):
                plgpu.emit_pipeline(
                    lambda _, lhs_smem, rhs_smem: plgpu.wgmma(
                        acc_ref,
                        lhs_smem,
                        plgpu.transpose_ref(rhs_smem, (1, 0)) if transpose_rhs else rhs_smem,
                    ),
                    grid=(k // block_k,),
                    in_specs=[
                        plgpu.BlockSpec(
                            (block_m, block_k),
                            lambda k: (group_info.block, k),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_n, block_k) if transpose_rhs else (block_k, block_n),
                            lambda k: (ni, k) if transpose_rhs else (k, ni),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(lhs_gmem, rhs_gmem.at[group_info.group_id])
                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype))
            def store_scope(o_smem):
                o_smem[...] = (acc * scale_gmem[0]).astype(o_smem.dtype)
                plgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, m)
                # TMA descriptors need to be generated with static tile sizes along each
                # axis, but we do not know at compile time how many rows we will need to
                # store. We only know that the number of rows to store is bounded by
                # min(block_m, m).
                #
                # In order to work around that, we construct a logarithmic ladder of
                # TMA descriptors, where each descriptor can store 2**i rows for some
                # i between 0 and log2(min(block_m, m)). This allows storing any
                # number of rows we will need to store, so long as this number of rows
                # is between `1` and `min(block_m, m)`.
                #
                # E.g., imagine we have block_m = 8, m = 16. The loop below will be
                # unrolled into 4 iterations, where the first one will generate a TMA
                # descriptor that can store 8 rows, the second one will generate a TMA
                # descriptor that can store 4 rows, etc. all the way to 1 row.
                #
                # At run time, we finally know the actual number of rows we need to
                # store as we go through the unrolled loop iterations. Let's imagine
                # that we need to store 5 rows.
                #
                # The first unrolled iteration will check whether we can store 8 rows.
                # Since we only need to store 5 rows, we won't store anything then.
                #
                # The second unrolled iteration will check whether we can store 4 rows.
                # We're able to store 4 rows, and are left with a single remaining row.
                #
                # The fourth unrolled iteration will store the single remaining row, and
                # we end up with a storing scheme as follows for our 5 rows:
                #
                #     -----------------------------------------------------------
                #  0  |                                                         |
                #  1  |                                                         |
                #  2  |                       Store 4 rows                      |
                #  3  |                                                         |
                #     -----------------------------------------------------------
                #  4  |                       Store 1 row                       |
                #     -----------------------------------------------------------
                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
                        o_gref_slice = o_gmem.at[
                            pl.ds(group_info.block_start + smem_start, const_rows_len),
                            pl.ds(ni * block_n, block_n),
                        ]
                        plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

                    smem_start += group_info.actual_size & const_rows_len
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    # There are 132 SMs on a H100 SXM GPU.
    num_sms = 132
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((m, n), _od),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup,
        ),
    )
    return kernel(group_sizes, lhs, rhs, _scale)


def mgpu_transposed_ragged_dot(
    lhs,  # (K, M)
    rhs,  # (K, N)
    *,
    group_sizes,  # (G,)
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
    out_dtype=None,
) -> jax.Array:
    _fp8 = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    _mixed_fp8 = lhs.dtype in _fp8 and rhs.dtype in _fp8
    if lhs.dtype != rhs.dtype and not _mixed_fp8:
        raise NotImplementedError(f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}")
    k, m = lhs.shape
    k2, n = rhs.shape
    _od = lhs.dtype if out_dtype is None else out_dtype
    g = group_sizes.shape[0]

    if k != k2:
        raise ValueError(f"lhs.shape={k} must match rhs.shape={k2}")

    if m % block_m != 0:
        raise ValueError(f"m={m} must be a multiple of block_m={block_m}")
    if n % block_n != 0:
        raise ValueError(f"n={n} must be a multiple of block_n={block_n}")

    group_sizes = group_sizes.astype(int)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=int), jnp.cumsum(group_sizes)[:-1]]).astype(int)
    group_ends = jnp.cumsum(group_sizes)
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    group_num_blocks = (group_block_ends - group_block_starts) // block_k

    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(lhs.dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
    # The output SMEM tile uses `out_dtype`, which may differ from the FP8 input
    # dtype (e.g. FP8 in, BF16 out).  The swizzle byte-width is fixed, but the
    # tiling's trailing element count scales with the dtype width, so the output
    # tile needs transforms computed for `_od` (upstream assumes in==out dtype).
    _o_swizzle_elems = (swizzle * 8) // (jnp.dtype(_od).itemsize * 8)
    o_transforms = (
        plgpu.TilingTransform((8, _o_swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )

    def body(
        group_sizes_gmem,
        group_starts_gmem,
        group_ends_gmem,
        group_num_blocks_gmem,
        group_block_starts_gmem,
        lhs_gmem,
        rhs_gmem,
        o_gmem,
    ):

        grid_m = pl.cdiv(m, block_m)
        grid_n = pl.cdiv(n, block_n)

        @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            g_i = loop_info.index[0]
            m_i, n_i = plgpu.planar_snake(
                loop_info.index[1],
                (grid_m, grid_n),
                1,
                grid_block_n,
            )

            # This slice is potentially out of bounds, but we never access the
            # out of bound part in emit_pipeline.
            gmem_slice = pl.ds(group_block_starts_gmem[g_i], k)

            def acc_scope(acc_ref):
                def block_matmul(block_idx, lhs_smem, rhs_smem):
                    block_idx = block_idx[0]

                    @pl.when(block_idx == 0)
                    def _():
                        # Handles the first block of the group, where there might be
                        # data from the previous group in the beginning of the block.
                        lhs_reg = lhs_smem[...]
                        start_index = lax.rem(group_starts_gmem[g_i], block_k)
                        indices = plgpu.layout_cast(
                            jax.lax.broadcasted_iota(jnp.int32, (block_k, block_m), 0),
                            plgpu.Layout.WGMMA,
                        )
                        lhs_reg = jnp.where(
                            indices >= start_index,
                            lhs_reg.astype(jnp.float16),
                            jnp.float16(0),
                        ).astype(lhs_smem.dtype)
                        lhs_smem[...] = lhs_reg
                        plgpu.commit_smem()

                    @pl.when(block_idx == group_num_blocks_gmem[g_i] - 1)
                    def _():
                        # Handles the last block of the group, where there might be
                        # data from the next group in the end of the block.
                        lhs_reg = lhs_smem[...]
                        last_index = lax.rem(group_ends_gmem[g_i] - 1, block_k)
                        indices = plgpu.layout_cast(
                            jax.lax.broadcasted_iota(jnp.int32, (block_k, block_m), 0),
                            plgpu.Layout.WGMMA,
                        )
                        lhs_reg = jnp.where(
                            indices <= last_index,
                            lhs_reg.astype(jnp.float16),
                            jnp.float16(0),
                        ).astype(lhs_smem.dtype)
                        lhs_smem[...] = lhs_reg
                        plgpu.commit_smem()

                    plgpu.wgmma(acc_ref, plgpu.transpose_ref(lhs_smem, (1, 0)), rhs_smem)
                    if max_concurrent_steps == 1:
                        # Without delayed release, we won't have at least two separate
                        # smem blocks in flight. Therefore, we cannot rely on the implicit
                        # wait of wgmma to gaurantee that the data in smem is ready to be
                        # overwritten by the next pipeline iteration.
                        plgpu.wgmma_wait(0)

                @pl.when(group_sizes_gmem[g_i] > 0)  # Skip the group if it is empty.
                def _():
                    plgpu.emit_pipeline(
                        block_matmul,
                        grid=(group_num_blocks_gmem[g_i],),
                        in_specs=[
                            plgpu.BlockSpec(
                                (block_k, block_m),
                                lambda k_i: (k_i, m_i),
                                delay_release=1 if max_concurrent_steps > 1 else 0,
                                transforms=transforms,
                            ),
                            plgpu.BlockSpec(
                                (block_k, block_n),
                                lambda k_i: (k_i, n_i),
                                delay_release=1 if max_concurrent_steps > 1 else 0,
                                transforms=transforms,
                            ),
                        ],
                        max_concurrent_steps=max_concurrent_steps,
                    )(lhs_gmem.at[gmem_slice, :], rhs_gmem.at[gmem_slice, :])

                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(
                pl.run_scoped,
                o_smem=plgpu.SMEM(
                    (block_m, block_n),
                    dtype=o_gmem.dtype,
                    transforms=o_transforms,
                ),
            )
            def store_scope(o_smem):
                o_smem[...] = acc.astype(o_smem.dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    o_smem,
                    o_gmem.at[
                        g_i,
                        pl.ds(m_i * block_m, block_m),
                        pl.ds(n_i * block_n, block_n),
                    ],
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    # There are 132 SMs on a H100 SXM GPU.
    num_sms = jax.devices()[0].core_count
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((g, m, n), _od),
        grid=(num_sms,),
        grid_names=("sm",),
    )
    return kernel(
        group_sizes,
        group_starts,
        group_ends,
        group_num_blocks,
        group_block_starts,
        lhs,
        rhs,
    )


def mgpu_dwgrad(
    lhs_t,  # (K, T)  e4m3 -- weight-grad lhs, token dim T contiguous (dim 1)
    grad_t,  # (N, T) e5m2 -- output gradient, token dim T contiguous (dim 1)
    *,
    group_sizes,  # (G,)
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
    out_dtype=None,
    out_scale=None,
) -> jax.Array:
    """Genuine ragged FP8 weight gradient: ``dr[G,K,N] = sum_{t in g} lhs[t,K] grad[t,N]``.

    Contracts the *ragged* token dimension.  FP8 ``wgmma`` cannot transpose an
    operand at runtime, so the token (contracting) dimension must be contiguous
    for both operands: the caller cast-transposes ``lhs -> [K, T]`` (E4M3) and
    ``grad -> [N, T]`` (E5M2) out of kernel.  Inside, ``A = lhs[K, T]`` is used
    natively (token-contiguous) and ``B = transpose_ref(grad[N, T]) = [T, N]``
    uses the B-operand transpose that FP8 ``wgmma`` does support.  Mixed
    E4M3 x E5M2 is contracted genuinely (no all-E4M3/all-E5M2 shortcut).
    """
    _fp8 = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    if not (lhs_t.dtype in _fp8 and grad_t.dtype in _fp8):
        raise NotImplementedError(f"mgpu_dwgrad expects FP8 operands, got {lhs_t.dtype}, {grad_t.dtype}")
    k_dim, t = lhs_t.shape
    n_dim, t2 = grad_t.shape
    if t != t2:
        raise ValueError(f"token dim mismatch: {t} vs {t2}")
    if k_dim % block_m != 0:
        raise ValueError(f"K={k_dim} must be a multiple of block_m={block_m}")
    if n_dim % block_n != 0:
        raise ValueError(f"N={n_dim} must be a multiple of block_n={block_n}")
    g = group_sizes.shape[0]
    _od = jnp.bfloat16 if out_dtype is None else out_dtype
    _scale = jnp.ones((1,), jnp.float32) if out_scale is None else jnp.asarray(out_scale, jnp.float32).reshape(1)

    group_sizes = group_sizes.astype(int)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=int), jnp.cumsum(group_sizes)[:-1]]).astype(int)
    group_ends = jnp.cumsum(group_sizes)
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    group_num_blocks = (group_block_ends - group_block_starts) // block_k

    # Operand SMEM tiles are token-contiguous (trailing dim = block_k tokens).
    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs_t.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(lhs_t.dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle),
    )
    # Output tile [block_m(K), block_n(N)] is N-contiguous; swizzle from block_n.
    o_swizzle = plgpu.find_swizzle(block_n * jnp.dtype(_od).itemsize * 8)
    o_swizzle_elems = o_swizzle // jnp.dtype(_od).itemsize
    o_transforms = (
        plgpu.TilingTransform((8, o_swizzle_elems)),
        plgpu.SwizzleTransform(o_swizzle),
    )

    def body(
        group_sizes_gmem,
        group_starts_gmem,
        group_ends_gmem,
        group_num_blocks_gmem,
        group_block_starts_gmem,
        lhs_gmem,
        grad_gmem,
        scale_gmem,
        o_gmem,
    ):
        grid_m = pl.cdiv(k_dim, block_m)
        grid_n = pl.cdiv(n_dim, block_n)

        @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            g_i = loop_info.index[0]
            m_i, n_i = plgpu.planar_snake(loop_info.index[1], (grid_m, grid_n), 1, grid_block_n)
            # Slice the ragged token dimension (dim 1).  Potentially out of bounds,
            # but the out-of-bounds tail is never accessed in emit_pipeline.
            tok_slice = pl.ds(group_block_starts_gmem[g_i], t)

            def acc_scope(acc_ref):
                def block_matmul(block_idx, lhs_smem, grad_smem):
                    block_idx = block_idx[0]

                    @pl.when(block_idx == 0)
                    def _():
                        reg = lhs_smem[...]
                        start_index = lax.rem(group_starts_gmem[g_i], block_k)
                        indices = plgpu.layout_cast(
                            jax.lax.broadcasted_iota(jnp.int32, (block_m, block_k), 1),
                            plgpu.Layout.WGMMA,
                        )
                        reg = jnp.where(
                            indices >= start_index,
                            reg.astype(jnp.float16),
                            jnp.float16(0),
                        ).astype(lhs_smem.dtype)
                        lhs_smem[...] = reg
                        plgpu.commit_smem()

                    @pl.when(block_idx == group_num_blocks_gmem[g_i] - 1)
                    def _():
                        reg = lhs_smem[...]
                        last_index = lax.rem(group_ends_gmem[g_i] - 1, block_k)
                        indices = plgpu.layout_cast(
                            jax.lax.broadcasted_iota(jnp.int32, (block_m, block_k), 1),
                            plgpu.Layout.WGMMA,
                        )
                        reg = jnp.where(
                            indices <= last_index,
                            reg.astype(jnp.float16),
                            jnp.float16(0),
                        ).astype(lhs_smem.dtype)
                        lhs_smem[...] = reg
                        plgpu.commit_smem()

                    # A = lhs[K, T] native (token-contiguous); B = grad[N, T] -> [T, N].
                    plgpu.wgmma(acc_ref, lhs_smem, plgpu.transpose_ref(grad_smem, (1, 0)))
                    if max_concurrent_steps == 1:
                        plgpu.wgmma_wait(0)

                @pl.when(group_sizes_gmem[g_i] > 0)
                def _():
                    plgpu.emit_pipeline(
                        block_matmul,
                        grid=(group_num_blocks_gmem[g_i],),
                        in_specs=[
                            plgpu.BlockSpec(
                                (block_m, block_k),
                                lambda k_i: (m_i, k_i),
                                delay_release=1 if max_concurrent_steps > 1 else 0,
                                transforms=transforms,
                            ),
                            plgpu.BlockSpec(
                                (block_n, block_k),
                                lambda k_i: (n_i, k_i),
                                delay_release=1 if max_concurrent_steps > 1 else 0,
                                transforms=transforms,
                            ),
                        ],
                        max_concurrent_steps=max_concurrent_steps,
                    )(lhs_gmem.at[:, tok_slice], grad_gmem.at[:, tok_slice])

                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(
                pl.run_scoped,
                o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype, transforms=o_transforms),
            )
            def store_scope(o_smem):
                o_smem[...] = (acc * scale_gmem[0]).astype(o_smem.dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    o_smem,
                    o_gmem.at[
                        g_i,
                        pl.ds(m_i * block_m, block_m),
                        pl.ds(n_i * block_n, block_n),
                    ],
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.devices()[0].core_count
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((g, k_dim, n_dim), _od),
        grid=(num_sms,),
        grid_names=("sm",),
    )
    return kernel(
        group_sizes,
        group_starts,
        group_ends,
        group_num_blocks,
        group_block_starts,
        lhs_t,
        grad_t,
        _scale,
    )
