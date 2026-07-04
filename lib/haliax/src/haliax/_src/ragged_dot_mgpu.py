# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored and lightly modified from JAX (Apache-2.0):
#   jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py
# (jax 0.10.0).  These Pallas-Mosaic-GPU grouped matmuls drive the Hopper FP8
# tensor cores (wgmma.fp8) and handle non-uniform, dynamic group_sizes.
# Modifications vs upstream:
#   * `out_dtype` parameter so an FP8 contraction can emit a BF16/FP32 result
#     (upstream hardcodes the output dtype to ``lhs.dtype``);
#   * `out_scale` parameter: per-tensor dequant scale folded into the store;
#   * `num_sms` read from the device instead of hardcoded;
#   * mixed FP8 operand pairs (E5M2 x E4M3) are accepted; lowering them needs
#     jax >= 0.11.0 (mixed-dtype wgmma, jax-ml/jax#38859);
#   * the ``main``/``ref_``/profiling helpers are dropped.

import dataclasses
import functools
import math

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

_FP8_DTYPES = (jnp.float8_e4m3fn, jnp.float8_e5m2)


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
    out_dtype=None,
    out_scale=None,
) -> jax.Array:
    if lhs.dtype != rhs.dtype and not (lhs.dtype in _FP8_DTYPES and rhs.dtype in _FP8_DTYPES):
        raise NotImplementedError(
            f"lhs and rhs must have the same dtype or both be FP8 (mixed wgmma), got {lhs.dtype} and {rhs.dtype}"
        )
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

    num_sms = jax.devices()[0].core_count
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
