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


def mosaic_ragged_dot_contract_major(
    lhs: jax.Array,
    rhs_contract_major: jax.Array,
    *,
    group_sizes: jax.Array,
    out_dtype: DTypeLike,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
) -> jax.Array:
    """Ragged `lhs[M,K] @ rhs[G,K,N]` using Hopper WGMMA without RHS transpose."""
    fp8_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    is_mixed_fp8 = lhs.dtype in fp8_dtypes and rhs_contract_major.dtype in fp8_dtypes
    if lhs.dtype != rhs_contract_major.dtype and not is_mixed_fp8:
        raise NotImplementedError(f"lhs and rhs must have compatible dtypes, got {lhs.dtype} and {rhs_contract_major.dtype}")

    m, k = lhs.shape
    groups, k2, n = rhs_contract_major.shape
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
                    lambda _, lhs_smem, rhs_smem: plgpu.wgmma(acc_ref, lhs_smem, rhs_smem),
                    grid=(k // block_k,),
                    in_specs=[
                        plgpu.BlockSpec((block_m, block_k), lambda kk: (group_info.block, kk), delay_release=1),
                        plgpu.BlockSpec((block_k, block_n), lambda kk: (kk, ni), delay_release=1),
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
    return kernel(group_sizes, lhs, rhs_contract_major)


def mosaic_transposed_ragged_dot(
    lhs_t: jax.Array,
    rhs: jax.Array,
    *,
    group_sizes: jax.Array,
    out_dtype: DTypeLike,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
    mask_boundaries: bool = True,
) -> jax.Array:
    """Ragged grouped `lhs_t[:, group].T @ rhs[group]` without FP8 WGMMA transposes."""
    fp8_dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2)
    is_mixed_fp8 = lhs_t.dtype in fp8_dtypes and rhs.dtype in fp8_dtypes
    if lhs_t.dtype != rhs.dtype and not is_mixed_fp8:
        raise NotImplementedError(f"lhs and rhs must have compatible dtypes, got {lhs_t.dtype} and {rhs.dtype}")

    m, tokens = lhs_t.shape
    tokens2, n = rhs.shape
    groups = group_sizes.shape[0]
    if tokens != tokens2:
        raise ValueError(f"lhs_t.shape={tokens} must match rhs.shape={tokens2}")
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

    def body(
        group_sizes_gmem,
        group_starts_gmem,
        group_ends_gmem,
        group_num_blocks_gmem,
        group_block_starts_gmem,
        lhs_t_gmem,
        rhs_gmem,
        out_gmem,
    ):
        grid_m = pl.cdiv(m, block_m)
        grid_n = pl.cdiv(n, block_n)

        @plgpu.nd_loop((groups, grid_m * grid_n), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            group = loop_info.index[0]
            m_i, n_i = plgpu.planar_snake(loop_info.index[1], (grid_m, grid_n), 1, grid_block_n)
            token_slice = pl.ds(group_block_starts_gmem[group], tokens)

            def acc_scope(acc_ref):
                def block_matmul(block_idx, lhs_smem, rhs_smem):
                    block_idx = block_idx[0]

                    if mask_boundaries:
                        @pl.when(block_idx == 0)
                        def _():
                            start_index = lax.rem(group_starts_gmem[group], block_k)
                            lhs_reg = lhs_smem[...]
                            lhs_indices = plgpu.layout_cast(
                                jax.lax.broadcasted_iota(jnp.int32, (block_m, block_k), 1),
                                plgpu.Layout.WGMMA,
                            )
                            lhs_smem[...] = jnp.where(lhs_indices >= start_index, lhs_reg, jnp.zeros_like(lhs_reg))
                            rhs_reg = rhs_smem[...]
                            rhs_indices = plgpu.layout_cast(
                                jax.lax.broadcasted_iota(jnp.int32, (block_k, block_n), 0),
                                plgpu.Layout.WGMMA,
                            )
                            rhs_smem[...] = jnp.where(rhs_indices >= start_index, rhs_reg, jnp.zeros_like(rhs_reg))
                            plgpu.commit_smem()

                        @pl.when(block_idx == group_num_blocks_gmem[group] - 1)
                        def _():
                            last_index = lax.rem(group_ends_gmem[group] - 1, block_k)
                            lhs_reg = lhs_smem[...]
                            lhs_indices = plgpu.layout_cast(
                                jax.lax.broadcasted_iota(jnp.int32, (block_m, block_k), 1),
                                plgpu.Layout.WGMMA,
                            )
                            lhs_smem[...] = jnp.where(lhs_indices <= last_index, lhs_reg, jnp.zeros_like(lhs_reg))
                            rhs_reg = rhs_smem[...]
                            rhs_indices = plgpu.layout_cast(
                                jax.lax.broadcasted_iota(jnp.int32, (block_k, block_n), 0),
                                plgpu.Layout.WGMMA,
                            )
                            rhs_smem[...] = jnp.where(rhs_indices <= last_index, rhs_reg, jnp.zeros_like(rhs_reg))
                            plgpu.commit_smem()

                    plgpu.wgmma(acc_ref, lhs_smem, rhs_smem)

                @pl.when(group_sizes_gmem[group] > 0)
                def _():
                    plgpu.emit_pipeline(
                        block_matmul,
                        grid=(group_num_blocks_gmem[group],),
                        in_specs=[
                            plgpu.BlockSpec((block_m, block_k), lambda kk: (m_i, kk), delay_release=1),
                            plgpu.BlockSpec((block_k, block_n), lambda kk: (kk, n_i), delay_release=1),
                        ],
                        max_concurrent_steps=max_concurrent_steps,
                    )(lhs_t_gmem.at[:, token_slice], rhs_gmem.at[token_slice, :])

                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((block_m, block_n), dtype=out_gmem.dtype))
            def store_scope(out_smem):
                out_smem[...] = acc.astype(out_smem.dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    out_smem,
                    out_gmem.at[group, pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n)],
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.devices()[0].core_count
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((groups, m, n), out_dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts, lhs_t, rhs)
