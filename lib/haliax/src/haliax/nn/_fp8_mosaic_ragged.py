# Copyright 2025 The JAX Authors.
#
# SPDX-License-Identifier: Apache-2.0
"""Hopper Mosaic GPU helpers for FP8 ragged dot."""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.typing import DTypeLike


@dataclasses.dataclass(frozen=True)
class _GroupInfo:
    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]
        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, boundary in enumerate(group_boundaries):
            start = end
            end = start + boundary
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
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


def mosaic_ragged_dot(
    lhs: jax.Array,
    rhs_k_major: jax.Array,
    *,
    group_sizes: jax.Array,
    out_dtype: DTypeLike,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
) -> jax.Array:
    """Ragged `lhs[M,K] @ rhs[G,N,K].T` using Hopper WGMMA."""
    fp8_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    is_mixed_fp8 = lhs.dtype in fp8_dtypes and rhs_k_major.dtype in fp8_dtypes
    if lhs.dtype != rhs_k_major.dtype and not is_mixed_fp8:
        raise NotImplementedError(f"lhs and rhs must have compatible dtypes, got {lhs.dtype} and {rhs_k_major.dtype}")

    m, k = lhs.shape
    groups, n, k2 = rhs_k_major.shape
    if group_sizes.shape[0] != groups:
        raise ValueError(f"Expected group_sizes to have shape {groups} but got {group_sizes.shape}")
    if k != k2:
        raise ValueError(f"lhs.shape={k} must match rhs.shape={k2}")
    if k % block_k != 0:
        raise ValueError(f"k={k} must be a multiple of block_k={block_k}")

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, out_gmem):
        grid_m = pl.cdiv(m, block_m) + groups - 1
        grid_n = pl.cdiv(n, block_n)

        @plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, grid_block_n)
            group_info = _GroupInfo.create(rows_per_expert_gmem, block_m, mi)

            def acc_scope(acc_ref):
                plgpu.emit_pipeline(
                    lambda _, lhs_smem, rhs_smem: plgpu.wgmma(
                        acc_ref,
                        lhs_smem,
                        plgpu.transpose_ref(rhs_smem, (1, 0)),
                    ),
                    grid=(k // block_k,),
                    in_specs=[
                        plgpu.BlockSpec(
                            (block_m, block_k),
                            lambda kk: (group_info.block, kk),
                            delay_release=1,
                        ),
                        plgpu.BlockSpec(
                            (block_n, block_k),
                            lambda kk: (ni, kk),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=max_concurrent_steps,
                )(lhs_gmem, rhs_gmem.at[group_info.group_id])
                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((block_m, block_n), dtype=out_gmem.dtype))
            def store_scope(out_smem):
                out_smem[...] = acc.astype(out_smem.dtype)
                plgpu.commit_smem()

                smem_start = group_info.start_within_block
                remaining_rows = min(block_m, m)
                while remaining_rows > 0:
                    const_rows_len = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(group_info.actual_size & const_rows_len != 0)
                    def _():
                        out_smem_slice = out_smem.at[pl.ds(smem_start, const_rows_len)]
                        out_gmem_slice = out_gmem.at[
                            pl.ds(group_info.block_start + smem_start, const_rows_len),
                            pl.ds(ni * block_n, block_n),
                        ]
                        plgpu.copy_smem_to_gmem(out_smem_slice, out_gmem_slice)

                    smem_start += group_info.actual_size & const_rows_len
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.devices()[0].core_count
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(group_sizes, lhs, rhs_k_major)
