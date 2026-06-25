# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from jax.experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu (Apache-2.0, The JAX
# Authors) — the ragged-CONTRACTION (weight-gradient / drhs) Mosaic-GPU kernel. Two changes make it
# Hopper-f8-legal (the stock kernel works only for 16-bit grads):
#
#   1. Operands arrive CAST-TRANSPOSED so the contraction (token) axis is contiguous (last axis):
#        lhs (M=hidden, K=tokens)   rhs/grad (N=out, K=tokens).
#      The stock kernel takes token-MAJOR operands and does an in-kernel `wgmma(acc,
#      transpose_ref(lhs), rhs)`; that transpose is a REAL data transpose (tokens are strided), which
#      Hopper f8 wgmma forbids (`mosaic/gpu/wgmma.py`: `supports_transpose = bytewidth == 2`). With
#      token-contiguous operands the matmul is `wgmma(acc, lhs_smem, transpose_ref(rhs_smem))` — the
#      same shape as the forward ragged_dot's `transpose_rhs` path, whose `transpose_ref` is a FREE
#      relabel of already-K-contiguous data and is proven f8-legal (logbook GFP8-022).
#   2. Output is stored in an explicit `out_dtype` (f32/bf16 accumulation target), not truncated to f8.
#
# The ragged group-boundary masking (head/tail partial blocks, mediated through f32 so the f8 mask
# casts that Mosaic rejects are avoided) and the empty-group skip are inherited unchanged. See
# `.agents/projects/2026-06-25_fp8_ragged_wgrad_cast_transpose.md` and logbook GFP8-024/025/028.

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


@dataclasses.dataclass(frozen=True)
class WgradBlockConfig:
    """Block config for the f8 ragged weight-gradient kernel.

    ``block_k`` tiles the ragged token (contraction) axis; ``block_m``/``block_n`` tile the dense
    output ``(K=hidden, N=out)``. Default is a starting point; M2 autotunes it at the Grug shape.
    """

    block_m: int = 128
    block_n: int = 128
    block_k: int = 128
    max_concurrent_steps: int = 2
    grid_block_n: int = 1


_DEFAULT_WGRAD_CONFIG = WgradBlockConfig()


def transposed_ragged_dot(
    lhs: jax.Array,  # (M=hidden, K=tokens), token-contiguous f8
    rhs: jax.Array,  # (N=out,   K=tokens), token-contiguous f8
    group_sizes: jax.Array,  # (G,)
    *,
    out_dtype: jnp.dtype,
    config: WgradBlockConfig = _DEFAULT_WGRAD_CONFIG,
) -> jax.Array:  # (G, M=hidden, N=out)
    """f8 ragged weight-gradient: ``out[g,m,n] = sum_{k in group g} lhs[m,k] * rhs[n,k]``.

    The contraction axis ``k`` (tokens) is ragged over ``group_sizes`` and must be the contiguous
    (last) axis of both operands — produce that with a cast-transpose at the call site.
    """
    if lhs.dtype != rhs.dtype:
        raise NotImplementedError(f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}")
    block_m, block_n, block_k = config.block_m, config.block_n, config.block_k
    max_concurrent_steps, grid_block_n = config.max_concurrent_steps, config.grid_block_n

    m, k = lhs.shape
    n, k2 = rhs.shape
    g = group_sizes.shape[0]
    if k != k2:
        raise ValueError(f"contraction (token) dims must match, got lhs k={k} rhs k={k2}")
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
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))

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
            m_i, n_i = plgpu.planar_snake(loop_info.index[1], (grid_m, grid_n), 1, grid_block_n)

            # Token (contraction) slice for this group, aligned down to block_k. Potentially out of
            # bounds at the tail, but emit_pipeline never reads the out-of-bounds part.
            group_block_start = pl.multiple_of(group_block_starts_gmem[g_i], block_k)
            gmem_slice = pl.ds(group_block_start, k)

            def acc_scope(acc_ref):
                def block_matmul(block_idx, lhs_smem, rhs_smem):
                    block_idx = block_idx[0]

                    # Mask the ragged group boundary on the lhs token axis (axis 1 here). Zeroing
                    # lhs's out-of-group tokens zeroes their product, so only one operand is masked.
                    # f8 mask: Mosaic rejects i1->f8 casts, 8-bit pointwise ops, and f8->bf16; mediate
                    # the select through f32 (lossless; matmul stays f8). Boundary blocks only.
                    @pl.when(block_idx == 0)
                    def _():
                        lhs_reg = lhs_smem[...]
                        start_index = lax.rem(group_starts_gmem[g_i], block_k)
                        cols = plgpu.layout_cast(
                            jax.lax.broadcasted_iota(jnp.int32, (block_m, block_k), 1), plgpu.Layout.WGMMA
                        )
                        lhs_f = lhs_reg.astype(jnp.float32)
                        lhs_reg = jnp.where(cols >= start_index, lhs_f, jnp.zeros_like(lhs_f)).astype(lhs_smem.dtype)
                        lhs_smem[...] = lhs_reg
                        plgpu.commit_smem()

                    @pl.when(block_idx == group_num_blocks_gmem[g_i] - 1)
                    def _():
                        lhs_reg = lhs_smem[...]
                        last_index = lax.rem(group_ends_gmem[g_i] - 1, block_k)
                        cols = plgpu.layout_cast(
                            jax.lax.broadcasted_iota(jnp.int32, (block_m, block_k), 1), plgpu.Layout.WGMMA
                        )
                        lhs_f = lhs_reg.astype(jnp.float32)
                        lhs_reg = jnp.where(cols <= last_index, lhs_f, jnp.zeros_like(lhs_f)).astype(lhs_smem.dtype)
                        lhs_smem[...] = lhs_reg
                        plgpu.commit_smem()

                    # Both operands token(K)-contiguous: lhs (hidden,tok) is wgmma-A directly; rhs
                    # (out,tok) is wgmma-B via the free transpose_ref relabel (== forward transpose_rhs).
                    plgpu.wgmma(acc_ref, lhs_smem, plgpu.transpose_ref(rhs_smem, (1, 0)))
                    if max_concurrent_steps == 1:
                        plgpu.wgmma_wait(0)

                @pl.when(group_sizes_gmem[g_i] > 0)  # Skip empty groups.
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
                    )(lhs_gmem.at[:, gmem_slice], rhs_gmem.at[:, gmem_slice])

                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(
                pl.run_scoped,
                o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype, transforms=transforms),
            )
            def store_scope(o_smem):
                o_smem[...] = acc.astype(o_smem.dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    o_smem, o_gmem.at[g_i, pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n)]
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.devices()[0].core_count
    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((g, m, n), out_dtype),
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
