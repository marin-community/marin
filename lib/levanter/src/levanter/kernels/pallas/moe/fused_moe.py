# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Optional

import jax
from jax._src import ad_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np


@dataclass(frozen=True, slots=True)
class MoEBlockSizes:
    """Block sizes for fused MoE Pallas kernels."""

    block_b: int = 256
    block_h: int = 512
    block_m: int = 1024
    block_out: int = 512

    @classmethod
    def get_default(cls) -> "MoEBlockSizes":
        return cls()


def _validate_block_sizes(block_sizes: MoEBlockSizes) -> None:
    for name in ("block_b", "block_h", "block_m", "block_out"):
        value = getattr(block_sizes, name)
        if value % 128 != 0:
            raise ValueError(f"{name} must be a multiple of 128, got {value}.")


def _build_dispatch(
    topk_idx: Int[Array, "tokens topk"],
    topk_weights: Float[Array, "tokens topk"],
    *,
    block_b: int,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sort tokens by expert and pad to block_b for block-wise execution."""
    token_count, topk = topk_idx.shape
    token_ids = jnp.repeat(jnp.arange(token_count, dtype=jnp.int32), topk)
    expert_ids = topk_idx.reshape(-1).astype(jnp.int32)
    weights = topk_weights.reshape(-1)

    sort_idx = jnp.argsort(expert_ids)
    token_ids = token_ids[sort_idx]
    expert_ids = expert_ids[sort_idx]
    weights = weights[sort_idx]

    group_sizes = jnp.bincount(expert_ids, length=num_experts)
    group_sizes_rounded = (group_sizes + block_b - 1) // block_b * block_b
    total_padded = jnp.sum(group_sizes_rounded)
    total_slots = (token_count * topk + num_experts * block_b + block_b - 1) // block_b * block_b

    prefix = jnp.cumsum(group_sizes_rounded)
    offsets = prefix - group_sizes_rounded
    pos = jnp.arange(total_slots, dtype=jnp.int32)
    exp_id = jnp.searchsorted(prefix, pos, side="right").astype(jnp.int32)
    exp_id = jnp.minimum(exp_id, num_experts - 1)
    local_idx = pos - offsets[exp_id]
    within = jnp.logical_and(pos < total_padded, local_idx < group_sizes[exp_id])

    expert_start = jnp.cumsum(group_sizes) - group_sizes
    sorted_idx = expert_start[exp_id] + local_idx

    token_padded = jnp.where(within, token_ids[sorted_idx], 0)
    weight_padded = jnp.where(within, weights[sorted_idx], 0)

    block_tokens = token_padded.reshape(-1, block_b)
    block_weights = weight_padded.reshape(-1, block_b)
    block_expert = exp_id.reshape(-1, block_b)[:, 0]
    return block_tokens, block_weights, block_expert


def _scatter_blocks(
    updates: Float[Array, "blocks block_b hidden"],
    token_idx_blocks: Int[Array, "blocks block_b"],
    *,
    token_count: int,
) -> jax.Array:
    num_blocks, block_b, hidden = updates.shape
    token_idx = token_idx_blocks.reshape(-1, 1)
    updates_flat = updates.reshape(-1, hidden)
    out = jnp.zeros((token_count, hidden), dtype=updates.dtype)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return jax.lax.scatter_add(out, token_idx, updates_flat, dnums)


def _moe_mlp_kernel(
    block_expert_ref,
    x_ref,
    w1_ref,
    w2_ref,
    w3_ref,
    o_ref,
    w1_accum_ref,
    w3_accum_ref,
    out_accum_ref,
    *,
    num_h_blocks: int,
    num_m_blocks: int,
):
    """Compute one MoE MLP block (W1/W3 -> SiLU gate -> W2) with accumulation."""
    block_idx, out_idx, m_idx, h_idx = (pl.program_id(i) for i in range(4))
    del block_expert_ref

    # Per-(block_idx, out_idx) accumulator zeroed at the start of the M loop.
    @pl.when(jnp.logical_and(h_idx == 0, m_idx == 0))
    def _init_out():
        out_accum_ref[...] = jnp.zeros_like(out_accum_ref)

    # Per-(block_idx, m_idx) accumulators for W1/W3 over the H loop.
    @pl.when(h_idx == 0)
    def _init_m():
        w1_accum_ref[...] = jnp.zeros_like(w1_accum_ref)
        w3_accum_ref[...] = jnp.zeros_like(w3_accum_ref)

    # Accumulate W1/W3 projections across H tiles.
    w1_accum_ref[...] += jax.lax.dot_general(
        x_ref[0, ...],
        w1_ref[0, ...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    w3_accum_ref[...] += jax.lax.dot_general(
        x_ref[0, ...],
        w3_ref[0, ...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    # When H loop ends, apply activation and accumulate W2 for this M tile.
    @pl.when(h_idx == num_h_blocks - 1)
    def _accumulate_out():
        gated = jax.nn.silu(w1_accum_ref[...]) * w3_accum_ref[...]
        out_accum_ref[...] += jax.lax.dot_general(
            gated,
            w2_ref[0, ...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    # Write out only once the final M tile is processed.
    @pl.when(jnp.logical_and(h_idx == num_h_blocks - 1, m_idx == num_m_blocks - 1))
    def _write_out():
        o_ref[0, ...] = out_accum_ref[...].astype(o_ref.dtype)


def _pallas_forward(
    x_blocked: Float[Array, "blocks block_b hidden"],
    block_expert: Int[Array, "blocks"],
    w1: Float[Array, "experts hidden mlp_dim"],
    w2: Float[Array, "experts mlp_dim hidden"],
    w3: Float[Array, "experts hidden mlp_dim"],
    *,
    block_sizes: MoEBlockSizes,
) -> jax.Array:
    if jax.default_backend() != "tpu":
        raise NotImplementedError("Fused MoE Pallas kernel requires TPU backend.")
    _validate_block_sizes(block_sizes)
    num_blocks, block_b, hidden = x_blocked.shape
    num_experts, hidden_dim, mlp_dim = w1.shape
    if hidden_dim != hidden:
        raise ValueError("w1 hidden dim must match x hidden dim.")
    if w2.shape[0] != num_experts or w3.shape[0] != num_experts:
        raise ValueError("w2/w3 must match expert dimension.")
    if w2.shape[1] != mlp_dim or w3.shape[2] != mlp_dim:
        raise ValueError("w2/w3 must share mlp dim.")
    if w2.shape[2] != hidden:
        raise ValueError("w2 output dim must match hidden dim.")

    if hidden % block_sizes.block_h != 0 or hidden % block_sizes.block_out != 0:
        raise ValueError("hidden must be divisible by block_h and block_out.")
    if mlp_dim % block_sizes.block_m != 0:
        raise ValueError("mlp_dim must be divisible by block_m.")
    if block_b != block_sizes.block_b:
        raise ValueError("x_blocked block_b must match block_sizes.block_b.")

    num_h_blocks = hidden // block_sizes.block_h
    num_m_blocks = mlp_dim // block_sizes.block_m
    num_out_blocks = hidden // block_sizes.block_out

    def x_map(block_idx, out_idx, m_idx, h_idx, block_expert_ref):
        del out_idx, m_idx, block_expert_ref
        return (block_idx, 0, h_idx)

    def w1_map(block_idx, out_idx, m_idx, h_idx, block_expert_ref):
        del out_idx
        return (block_expert_ref[block_idx], h_idx, m_idx)

    def w3_map(block_idx, out_idx, m_idx, h_idx, block_expert_ref):
        del out_idx
        return (block_expert_ref[block_idx], h_idx, m_idx)

    def w2_map(block_idx, out_idx, m_idx, h_idx, block_expert_ref):
        del h_idx
        return (block_expert_ref[block_idx], m_idx, out_idx)

    def o_map(block_idx, out_idx, m_idx, h_idx, block_expert_ref):
        del m_idx, h_idx, block_expert_ref
        return (block_idx, 0, out_idx)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(num_blocks, num_out_blocks, num_m_blocks, num_h_blocks),
        in_specs=[
            pl.BlockSpec((1, block_sizes.block_b, block_sizes.block_h), x_map),
            pl.BlockSpec((1, block_sizes.block_h, block_sizes.block_m), w1_map),
            pl.BlockSpec((1, block_sizes.block_m, block_sizes.block_out), w2_map),
            pl.BlockSpec((1, block_sizes.block_h, block_sizes.block_m), w3_map),
        ],
        out_specs=pl.BlockSpec((1, block_sizes.block_b, block_sizes.block_out), o_map),
        scratch_shapes=(
            pltpu.VMEM((block_sizes.block_b, block_sizes.block_m), dtype=jnp.float32),
            pltpu.VMEM((block_sizes.block_b, block_sizes.block_m), dtype=jnp.float32),
            pltpu.VMEM((block_sizes.block_b, block_sizes.block_out), dtype=jnp.float32),
        ),
    )

    kernel = pl.pallas_call(
        partial(_moe_mlp_kernel, num_h_blocks=num_h_blocks, num_m_blocks=num_m_blocks),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_blocks, block_sizes.block_b, hidden), dtype=x_blocked.dtype),
    )
    return kernel(block_expert, x_blocked, w1, w2, w3)


def _pallas_forward_staged(
    x_blocked: Float[Array, "blocks block_b hidden"],
    block_expert_blocks: Int[Array, "tiles tile_b block_b"],
    block_weight_blocks: Float[Array, "tiles tile_b block_b"],
    w1: Float[Array, "experts hidden mlp_dim"],
    w2: Float[Array, "experts mlp_dim hidden"],
    w3: Float[Array, "experts hidden mlp_dim"],
    *,
    block_sizes: MoEBlockSizes,
    parallel: bool,
    block_tile: int,
) -> jax.Array:
    if jax.default_backend() != "tpu":
        raise NotImplementedError("Fused MoE Pallas kernel requires TPU backend.")
    _validate_block_sizes(block_sizes)
    num_blocks, block_b, hidden = x_blocked.shape
    num_experts, hidden_dim, mlp_dim = w1.shape
    if hidden_dim != hidden:
        raise ValueError("w1 hidden dim must match x hidden dim.")
    if w2.shape[0] != num_experts or w3.shape[0] != num_experts:
        raise ValueError("w2/w3 must match expert dimension.")
    if w2.shape[1] != mlp_dim or w3.shape[2] != mlp_dim:
        raise ValueError("w2/w3 must share mlp dim.")
    if w2.shape[2] != hidden:
        raise ValueError("w2 output dim must match hidden dim.")
    if hidden % block_sizes.block_h != 0 or hidden % block_sizes.block_out != 0:
        raise ValueError("hidden must be divisible by block_h and block_out.")
    if mlp_dim % block_sizes.block_m != 0:
        raise ValueError("mlp_dim must be divisible by block_m.")
    if block_b != block_sizes.block_b:
        raise ValueError("x_blocked block_b must match block_sizes.block_b.")

    num_h_blocks = hidden // block_sizes.block_h
    num_out_blocks = hidden // block_sizes.block_out
    num_m_blocks = mlp_dim // block_sizes.block_m

    def kernel(
        block_expert_ref,
        block_weight_ref,
        x_ref,
        w1_ref,
        w2_ref,
        w3_ref,
        o_ref,
        x_x2_ref,
        w1_x2_ref,
        w3_x2_ref,
        w2_x2_ref,
        sem_xw_0,
        sem_xw_1,
        sem_w2_0,
        sem_w2_1,
        w1_accum_ref,
        w3_accum_ref,
        out_accum_ref,
    ):
        tile_idx = pl.program_id(0)
        base_block = tile_idx * block_tile

        # Double-buffered DMA for x/w1/w3 into VMEM.
        def _start_xw(h_idx, m_idx, buf_idx, block_idx, expert):
            sem = sem_xw_0 if buf_idx == 0 else sem_xw_1
            x_src = x_ref.at[
                pl.ds(block_idx, 1),
                pl.ds(0, block_b),
                pl.ds(h_idx * block_sizes.block_h, block_sizes.block_h),
            ]
            x_dst = x_x2_ref.at[buf_idx]
            pltpu.make_async_copy(src_ref=x_src, dst_ref=x_dst, sem=sem).start()

            w1_src = w1_ref.at[
                expert,
                pl.ds(h_idx * block_sizes.block_h, block_sizes.block_h),
                pl.ds(m_idx * block_sizes.block_m, block_sizes.block_m),
            ]
            w1_dst = w1_x2_ref.at[buf_idx]
            pltpu.make_async_copy(src_ref=w1_src, dst_ref=w1_dst, sem=sem).start()

            w3_src = w3_ref.at[
                expert,
                pl.ds(h_idx * block_sizes.block_h, block_sizes.block_h),
                pl.ds(m_idx * block_sizes.block_m, block_sizes.block_m),
            ]
            w3_dst = w3_x2_ref.at[buf_idx]
            pltpu.make_async_copy(src_ref=w3_src, dst_ref=w3_dst, sem=sem).start()

        def _wait_xw(buf_idx):
            sem = sem_xw_0 if buf_idx == 0 else sem_xw_1
            x_buf = x_x2_ref.at[buf_idx]
            w1_buf = w1_x2_ref.at[buf_idx]
            w3_buf = w3_x2_ref.at[buf_idx]
            pltpu.make_async_copy(src_ref=x_buf, dst_ref=x_buf, sem=sem).wait()
            pltpu.make_async_copy(src_ref=w1_buf, dst_ref=w1_buf, sem=sem).wait()
            pltpu.make_async_copy(src_ref=w3_buf, dst_ref=w3_buf, sem=sem).wait()

        # Double-buffered DMA for W2 tiles.
        def _start_w2(m_idx, out_idx, expert, buf_idx):
            sem = sem_w2_0 if buf_idx == 0 else sem_w2_1
            dst = w2_x2_ref.at[buf_idx]
            w2_src = w2_ref.at[
                expert,
                pl.ds(m_idx * block_sizes.block_m, block_sizes.block_m),
                pl.ds(out_idx * block_sizes.block_out, block_sizes.block_out),
            ]
            pltpu.make_async_copy(src_ref=w2_src, dst_ref=dst, sem=sem).start()

        def _wait_w2(buf_idx):
            sem = sem_w2_0 if buf_idx == 0 else sem_w2_1
            buf = w2_x2_ref.at[buf_idx]
            pltpu.make_async_copy(src_ref=buf, dst_ref=buf, sem=sem).wait()

        for local_idx in range(block_tile):
            block_idx = base_block + local_idx
            expert = block_expert_ref[0, local_idx, 0]
            weights = block_weight_ref[0, local_idx, :]

            # Accumulate output for this block across M tiles.
            out_accum_ref[...] = jnp.zeros_like(out_accum_ref)
            for m_idx in range(num_m_blocks):
                # Accumulate W1/W3 across H tiles for the current M tile.
                w1_accum_ref[...] = jnp.zeros_like(w1_accum_ref)
                w3_accum_ref[...] = jnp.zeros_like(w3_accum_ref)

                _start_xw(0, m_idx, 0, block_idx, expert)
                for h_idx in range(num_h_blocks):
                    buf = h_idx & 1
                    next_h = h_idx + 1
                    if next_h < num_h_blocks:
                        _start_xw(next_h, m_idx, 1 - buf, block_idx, expert)
                    _wait_xw(buf)
                    x_block = x_x2_ref.at[buf][0, ...]
                    w1_block = w1_x2_ref.at[buf][...]
                    w3_block = w3_x2_ref.at[buf][...]
                    w1_accum_ref[...] += jax.lax.dot_general(
                        x_block,
                        w1_block,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    w3_accum_ref[...] += jax.lax.dot_general(
                        x_block,
                        w3_block,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )

                # Apply activation and stream W2 tiles, accumulating into output.
                gated = jax.nn.silu(w1_accum_ref[...]) * w3_accum_ref[...]
                _start_w2(m_idx, 0, expert, 0)
                for out_idx in range(num_out_blocks):
                    buf = out_idx & 1
                    next_out = out_idx + 1
                    if next_out < num_out_blocks:
                        _start_w2(m_idx, next_out, expert, 1 - buf)
                    _wait_w2(buf)
                    w2_block = w2_x2_ref.at[buf][...]
                    out_accum_ref[
                        :,
                        pl.ds(out_idx * block_sizes.block_out, block_sizes.block_out),
                    ] += jax.lax.dot_general(
                        gated,
                        w2_block,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )

            # Apply routing weights and write out for this block.
            weights_f32 = weights.astype(jnp.float32)
            out = (out_accum_ref[...].T * weights_f32).T.astype(o_ref.dtype)
            o_ref[0, local_idx, ...] = out

    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    def expert_map(block_idx):
        return (block_idx, 0, 0)

    def weight_map(block_idx):
        return (block_idx, 0, 0)

    def o_map(block_idx):
        return (block_idx, 0, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_blocks // block_tile,),
        in_specs=[
            pl.BlockSpec((1, block_tile, block_sizes.block_b), expert_map),
            pl.BlockSpec((1, block_tile, block_sizes.block_b), weight_map),
            hbm_block_spec,  # x
            hbm_block_spec,  # w1
            hbm_block_spec,  # w2
            hbm_block_spec,  # w3
        ],
        out_specs=pl.BlockSpec((1, block_tile, block_sizes.block_b, hidden), o_map),
        scratch_shapes=(
            pltpu.VMEM((2, 1, block_sizes.block_b, block_sizes.block_h), x_blocked.dtype),
            pltpu.VMEM((2, block_sizes.block_h, block_sizes.block_m), w1.dtype),
            pltpu.VMEM((2, block_sizes.block_h, block_sizes.block_m), w3.dtype),
            pltpu.VMEM((2, block_sizes.block_m, block_sizes.block_out), w2.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((block_sizes.block_b, block_sizes.block_m), dtype=jnp.float32),
            pltpu.VMEM((block_sizes.block_b, block_sizes.block_m), dtype=jnp.float32),
            pltpu.VMEM((block_sizes.block_b, hidden), dtype=jnp.float32),
        ),
    )

    compiler_params = None
    if parallel:
        compiler_params = pltpu.CompilerParams(dimension_semantics=("parallel",))

    kernel = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(
            (num_blocks // block_tile, block_tile, block_sizes.block_b, hidden), dtype=x_blocked.dtype
        ),
        compiler_params=compiler_params,
    )
    return kernel(block_expert_blocks, block_weight_blocks, x_blocked, w1, w2, w3)


def _pallas_forward_gather_staged(
    x: Float[Array, "tokens hidden"],
    block_tokens_blocks: Int[Array, "tiles tile_b block_b"],
    block_expert_blocks: Int[Array, "tiles tile_b block_b"],
    block_weight_blocks: Float[Array, "tiles tile_b block_b"],
    w1: Float[Array, "experts hidden mlp_dim"],
    w2: Float[Array, "experts mlp_dim hidden"],
    w3: Float[Array, "experts hidden mlp_dim"],
    *,
    block_sizes: MoEBlockSizes,
    parallel: bool,
    block_tile: int,
) -> jax.Array:
    if jax.default_backend() != "tpu":
        raise NotImplementedError("Fused MoE Pallas kernel requires TPU backend.")
    _validate_block_sizes(block_sizes)
    num_tiles, tile_b, block_b = block_tokens_blocks.shape
    if tile_b != block_tile:
        raise ValueError("block_tile mismatch for block_tokens_blocks.")
    num_experts, hidden_dim, mlp_dim = w1.shape
    hidden = x.shape[1]
    if hidden_dim != hidden:
        raise ValueError("w1 hidden dim must match x hidden dim.")
    if w2.shape[0] != num_experts or w3.shape[0] != num_experts:
        raise ValueError("w2/w3 must match expert dimension.")
    if w2.shape[1] != mlp_dim or w3.shape[2] != mlp_dim:
        raise ValueError("w2/w3 must share mlp dim.")
    if w2.shape[2] != hidden:
        raise ValueError("w2 output dim must match hidden dim.")
    if hidden % block_sizes.block_h != 0 or hidden % block_sizes.block_out != 0:
        raise ValueError("hidden must be divisible by block_h and block_out.")
    if mlp_dim % block_sizes.block_m != 0:
        raise ValueError("mlp_dim must be divisible by block_m.")

    num_h_blocks = hidden // block_sizes.block_h
    num_out_blocks = hidden // block_sizes.block_out
    num_m_blocks = mlp_dim // block_sizes.block_m

    def kernel(
        block_tokens_ref,
        block_expert_ref,
        block_weight_ref,
        x_ref,
        w1_ref,
        w2_ref,
        w3_ref,
        o_ref,
        w1_x2_ref,
        w3_x2_ref,
        w2_x2_ref,
        x_tile_ref,
        x_tile8_ref,
        sem_w1,
        sem_w3,
        sem_w2_0,
        sem_w2_1,
        sem_x,
        w1_accum_ref,
        w3_accum_ref,
        out_accum_ref,
    ):
        # Double-buffered DMA helpers for W1/W3 and W2 tiles.
        def _start_w(h_idx, m_idx, expert, sem, w_ref, buf):
            w_src = w_ref.at[
                expert,
                pl.ds(h_idx * block_sizes.block_h, block_sizes.block_h),
                pl.ds(m_idx * block_sizes.block_m, block_sizes.block_m),
            ]
            pltpu.make_async_copy(src_ref=w_src, dst_ref=buf, sem=sem).start()

        def _wait_w(sem, buf):
            pltpu.make_async_copy(src_ref=buf, dst_ref=buf, sem=sem).wait()

        # Double-buffered DMA for W2 tiles.
        def _start_w2(m_idx, out_idx, expert, buf_idx):
            sem = sem_w2_0 if buf_idx == 0 else sem_w2_1
            dst = w2_x2_ref.at[buf_idx]
            w2_src = w2_ref.at[
                expert,
                pl.ds(m_idx * block_sizes.block_m, block_sizes.block_m),
                pl.ds(out_idx * block_sizes.block_out, block_sizes.block_out),
            ]
            pltpu.make_async_copy(src_ref=w2_src, dst_ref=dst, sem=sem).start()

        def _wait_w2(buf_idx):
            sem = sem_w2_0 if buf_idx == 0 else sem_w2_1
            buf = w2_x2_ref.at[buf_idx]
            pltpu.make_async_copy(src_ref=buf, dst_ref=buf, sem=sem).wait()

        for local_idx in range(block_tile):
            token_ids = block_tokens_ref[0, local_idx, :]
            expert = block_expert_ref[0, local_idx, 0]
            weights = block_weight_ref[0, local_idx, :]

            # Gather token rows into a contiguous VMEM tile (gather path).
            out_accum_ref[...] = jnp.zeros_like(out_accum_ref)
            for m_idx in range(num_m_blocks):
                # Accumulate W1/W3 across H tiles for the current M tile.
                w1_accum_ref[...] = jnp.zeros_like(w1_accum_ref)
                w3_accum_ref[...] = jnp.zeros_like(w3_accum_ref)

                _start_w(0, m_idx, expert, sem_w1, w1_ref, w1_x2_ref)
                _start_w(0, m_idx, expert, sem_w3, w3_ref, w3_x2_ref)
                for h_idx in range(num_h_blocks):
                    _wait_w(sem_w1, w1_x2_ref)
                    _wait_w(sem_w3, w3_x2_ref)
                    for i in range(block_b):
                        token_id = token_ids[i]
                        base = token_id - (token_id % 8)
                        offset = token_id - base
                        src = x_ref.at[
                            pl.ds(base, 8),
                            pl.ds(h_idx * block_sizes.block_h, block_sizes.block_h),
                        ]
                        pltpu.make_async_copy(src_ref=src, dst_ref=x_tile8_ref, sem=sem_x).wait()
                        x_tile8 = x_tile8_ref[...]
                        mask = (jnp.arange(8, dtype=jnp.int32) == offset).astype(x_tile8.dtype)
                        x_tile_ref[pl.ds(i, 1), :] = jnp.sum(x_tile8 * mask[:, None], axis=0)[None, :]
                    x_block = x_tile_ref[...]
                    w1_accum_ref[...] += jax.lax.dot_general(
                        x_block,
                        w1_x2_ref[...],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    w3_accum_ref[...] += jax.lax.dot_general(
                        x_block,
                        w3_x2_ref[...],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    next_h = h_idx + 1
                    if next_h < num_h_blocks:
                        _start_w(next_h, m_idx, expert, sem_w1, w1_ref, w1_x2_ref)
                        _start_w(next_h, m_idx, expert, sem_w3, w3_ref, w3_x2_ref)

                gated = jax.nn.silu(w1_accum_ref[...]) * w3_accum_ref[...]
                _start_w2(m_idx, 0, expert, 0)
                for out_idx in range(num_out_blocks):
                    buf = out_idx & 1
                    next_out = out_idx + 1
                    if next_out < num_out_blocks:
                        _start_w2(m_idx, next_out, expert, 1 - buf)
                    _wait_w2(buf)
                    out_accum_ref[
                        :,
                        pl.ds(out_idx * block_sizes.block_out, block_sizes.block_out),
                    ] += jax.lax.dot_general(
                        gated,
                        w2_x2_ref.at[buf][...],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )

            weights_f32 = weights.astype(jnp.float32)
            out = (out_accum_ref[...].T * weights_f32).T.astype(o_ref.dtype)
            o_ref[0, local_idx, ...] = out

    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    def tokens_map(tile_idx):
        return (tile_idx, 0, 0)

    def expert_map(tile_idx):
        return (tile_idx, 0, 0)

    def weight_map(tile_idx):
        return (tile_idx, 0, 0)

    def o_map(tile_idx):
        return (tile_idx, 0, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((1, block_tile, block_b), tokens_map),
            pl.BlockSpec((1, block_tile, block_b), expert_map),
            pl.BlockSpec((1, block_tile, block_b), weight_map),
            hbm_block_spec,  # x
            hbm_block_spec,  # w1
            hbm_block_spec,  # w2
            hbm_block_spec,  # w3
        ],
        out_specs=pl.BlockSpec((1, block_tile, block_b, hidden), o_map),
        scratch_shapes=(
            pltpu.VMEM((block_sizes.block_h, block_sizes.block_m), w1.dtype),
            pltpu.VMEM((block_sizes.block_h, block_sizes.block_m), w3.dtype),
            pltpu.VMEM((2, block_sizes.block_m, block_sizes.block_out), w2.dtype),
            pltpu.VMEM((block_b, block_sizes.block_h), x.dtype),
            pltpu.VMEM((8, block_sizes.block_h), x.dtype),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((block_b, block_sizes.block_m), dtype=jnp.float32),
            pltpu.VMEM((block_b, block_sizes.block_m), dtype=jnp.float32),
            pltpu.VMEM((block_b, hidden), dtype=jnp.float32),
        ),
    )

    compiler_params = None
    if parallel:
        compiler_params = pltpu.CompilerParams(dimension_semantics=("parallel",))

    kernel = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tiles, block_tile, block_b, hidden), dtype=x.dtype),
        compiler_params=compiler_params,
    )
    return kernel(block_tokens_blocks, block_expert_blocks, block_weight_blocks, x, w1, w2, w3)


def _fused_moe_reference(
    x: Float[Array, "tokens hidden"],
    topk_idx: Int[Array, "tokens topk"],
    topk_weights: Float[Array, "tokens topk"],
    w1: Float[Array, "experts hidden mlp_dim"],
    w2: Float[Array, "experts mlp_dim hidden"],
    w3: Float[Array, "experts hidden mlp_dim"],
    *,
    block_sizes: MoEBlockSizes,
) -> jax.Array:
    block_tokens, block_weights, block_expert = _build_dispatch(
        topk_idx, topk_weights, block_b=block_sizes.block_b, num_experts=w1.shape[0]
    )
    x_blocked = jnp.take(x, block_tokens, axis=0)
    w1_block = w1[block_expert]
    w2_block = w2[block_expert]
    w3_block = w3[block_expert]

    up = jnp.einsum("bth,bhm->btm", x_blocked, w1_block, preferred_element_type=jnp.float32)
    gate = jnp.einsum("bth,bhm->btm", x_blocked, w3_block, preferred_element_type=jnp.float32)
    gated = jax.nn.silu(up) * gate
    out_blocked = jnp.einsum("btm,bmh->bth", gated, w2_block, preferred_element_type=jnp.float32)
    out_blocked = out_blocked * block_weights[..., None]

    return _scatter_blocks(out_blocked, block_tokens, token_count=x.shape[0])


@lru_cache(maxsize=None)
def _make_custom_vjp(block_sizes: MoEBlockSizes):
    @jax.custom_vjp
    def _fn(x, topk_idx, topk_weights, w1, w2, w3):
        return fused_moe_pallas(x, topk_idx, topk_weights, w1, w2, w3, block_sizes=block_sizes)

    def _fn_fwd(x, topk_idx, topk_weights, w1, w2, w3):
        out = fused_moe_pallas(x, topk_idx, topk_weights, w1, w2, w3, block_sizes=block_sizes)
        return out, (x, topk_idx, topk_weights, w1, w2, w3)

    def _fn_bwd(res, g):
        x, topk_idx, topk_weights, w1, w2, w3 = res
        vjp_fn = jax.vjp(
            lambda x_, w1_, w2_, w3_, tw_: _fused_moe_reference(
                x_, topk_idx, tw_, w1_, w2_, w3_, block_sizes=block_sizes
            ),
            x,
            w1,
            w2,
            w3,
            topk_weights,
        )[1]
        gx, gw1, gw2, gw3, gtw = vjp_fn(g)
        gidx = ad_util.Zero.from_value(topk_idx)
        return gx, gidx, gtw, gw1, gw2, gw3

    _fn.defvjp(_fn_fwd, _fn_bwd)
    return _fn


@partial(jax.jit, static_argnames=["block_sizes"])
def fused_moe_pallas(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_sizes: MoEBlockSizes,
) -> jax.Array:
    block_tokens, block_weights, block_expert = _build_dispatch(
        topk_idx, topk_weights, block_b=block_sizes.block_b, num_experts=w1.shape[0]
    )
    x_blocked = jnp.take(x, block_tokens, axis=0)
    out_blocked = _pallas_forward(x_blocked, block_expert, w1, w2, w3, block_sizes=block_sizes)
    out_blocked = out_blocked * block_weights[..., None]
    return _scatter_blocks(out_blocked, block_tokens, token_count=x.shape[0])


@partial(jax.jit, static_argnames=["block_sizes", "parallel"])
def fused_moe_pallas_staged(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_sizes: MoEBlockSizes,
    parallel: bool = False,
) -> jax.Array:
    block_tokens, block_weights, block_expert = _build_dispatch(
        topk_idx, topk_weights, block_b=block_sizes.block_b, num_experts=w1.shape[0]
    )
    block_tile = 8
    if block_expert.shape[0] % block_tile != 0:
        raise ValueError(f"block count {block_expert.shape[0]} must be divisible by {block_tile}")
    block_expert_padded = jnp.broadcast_to(block_expert[:, None], (block_expert.shape[0], block_sizes.block_b))
    num_tiles = block_expert.shape[0] // block_tile
    block_expert_blocks = block_expert_padded.reshape(num_tiles, block_tile, block_sizes.block_b)
    block_weight_blocks = block_weights.reshape(num_tiles, block_tile, block_sizes.block_b)
    x_blocked = jnp.take(x, block_tokens, axis=0)
    out_blocked = _pallas_forward_staged(
        x_blocked,
        block_expert_blocks,
        block_weight_blocks,
        w1,
        w2,
        w3,
        block_sizes=block_sizes,
        parallel=parallel,
        block_tile=block_tile,
    )
    out_blocked = out_blocked.reshape(block_expert.shape[0], block_sizes.block_b, x.shape[1])
    return _scatter_blocks(out_blocked, block_tokens, token_count=x.shape[0])


@partial(jax.jit, static_argnames=["block_sizes", "parallel"])
def fused_moe_pallas_gather_staged(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_sizes: MoEBlockSizes,
    parallel: bool = False,
) -> jax.Array:
    block_tokens, block_weights, block_expert = _build_dispatch(
        topk_idx, topk_weights, block_b=block_sizes.block_b, num_experts=w1.shape[0]
    )
    block_tile = 8
    if block_expert.shape[0] % block_tile != 0:
        raise ValueError(f"block count {block_expert.shape[0]} must be divisible by {block_tile}")
    num_tiles = block_expert.shape[0] // block_tile
    block_tokens_blocks = block_tokens.reshape(num_tiles, block_tile, block_sizes.block_b)
    block_expert_padded = jnp.broadcast_to(block_expert[:, None], (block_expert.shape[0], block_sizes.block_b))
    block_expert_blocks = block_expert_padded.reshape(num_tiles, block_tile, block_sizes.block_b)
    block_weight_blocks = block_weights.reshape(num_tiles, block_tile, block_sizes.block_b)

    out_blocked = _pallas_forward_gather_staged(
        x,
        block_tokens_blocks,
        block_expert_blocks,
        block_weight_blocks,
        w1,
        w2,
        w3,
        block_sizes=block_sizes,
        parallel=parallel,
        block_tile=block_tile,
    )
    out_blocked = out_blocked.reshape(block_expert.shape[0], block_sizes.block_b, x.shape[1])
    return _scatter_blocks(out_blocked, block_tokens, token_count=x.shape[0])


@lru_cache(maxsize=None)
def _make_custom_vjp_staged(block_sizes: MoEBlockSizes, parallel: bool):
    @jax.custom_vjp
    def _fn(x, topk_idx, topk_weights, w1, w2, w3):
        return fused_moe_pallas_staged(
            x, topk_idx, topk_weights, w1, w2, w3, block_sizes=block_sizes, parallel=parallel
        )

    def _fn_fwd(x, topk_idx, topk_weights, w1, w2, w3):
        out = fused_moe_pallas_staged(
            x, topk_idx, topk_weights, w1, w2, w3, block_sizes=block_sizes, parallel=parallel
        )
        return out, (x, topk_idx, topk_weights, w1, w2, w3)

    def _fn_bwd(res, g):
        x, topk_idx, topk_weights, w1, w2, w3 = res
        vjp_fn = jax.vjp(
            lambda x_, w1_, w2_, w3_, tw_: _fused_moe_reference(
                x_, topk_idx, tw_, w1_, w2_, w3_, block_sizes=block_sizes
            ),
            x,
            w1,
            w2,
            w3,
            topk_weights,
        )[1]
        gx, gw1, gw2, gw3, gtw = vjp_fn(g)
        gidx = ad_util.Zero.from_value(topk_idx)
        return gx, gidx, gtw, gw1, gw2, gw3

    _fn.defvjp(_fn_fwd, _fn_bwd)
    return _fn


def fused_moe_staged(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_sizes: Optional[MoEBlockSizes] = None,
    parallel: bool = False,
) -> jax.Array:
    if block_sizes is None:
        block_sizes = MoEBlockSizes.get_default()
    fn = _make_custom_vjp_staged(block_sizes, parallel)
    return fn(x, topk_idx, topk_weights, w1, w2, w3)


@lru_cache(maxsize=None)
def _make_custom_vjp_gather_staged(block_sizes: MoEBlockSizes, parallel: bool):
    @jax.custom_vjp
    def _fn(x, topk_idx, topk_weights, w1, w2, w3):
        return fused_moe_pallas_gather_staged(
            x, topk_idx, topk_weights, w1, w2, w3, block_sizes=block_sizes, parallel=parallel
        )

    def _fn_fwd(x, topk_idx, topk_weights, w1, w2, w3):
        out = fused_moe_pallas_gather_staged(
            x, topk_idx, topk_weights, w1, w2, w3, block_sizes=block_sizes, parallel=parallel
        )
        return out, (x, topk_idx, topk_weights, w1, w2, w3)

    def _fn_bwd(res, g):
        x, topk_idx, topk_weights, w1, w2, w3 = res
        vjp_fn = jax.vjp(
            lambda x_, w1_, w2_, w3_, tw_: _fused_moe_reference(
                x_, topk_idx, tw_, w1_, w2_, w3_, block_sizes=block_sizes
            ),
            x,
            w1,
            w2,
            w3,
            topk_weights,
        )[1]
        gx, gw1, gw2, gw3, gtw = vjp_fn(g)
        gidx = ad_util.Zero.from_value(topk_idx)
        return gx, gidx, gtw, gw1, gw2, gw3

    _fn.defvjp(_fn_fwd, _fn_bwd)
    return _fn


def fused_moe_gather_staged(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_sizes: Optional[MoEBlockSizes] = None,
    parallel: bool = False,
) -> jax.Array:
    if block_sizes is None:
        block_sizes = MoEBlockSizes.get_default()
    fn = _make_custom_vjp_gather_staged(block_sizes, parallel)
    return fn(x, topk_idx, topk_weights, w1, w2, w3)


def fused_moe_tpu_inference(
    x: jax.Array,
    router_logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    top_k: int,
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
    mesh: Optional[jax.sharding.Mesh] = None,
    ep_axis_name: str = "model",
) -> jax.Array:
    """Use the vendored TPU-inference fused MoE kernel (expects router logits)."""
    from levanter.kernels.pallas.moe.tpu_inference_v1 import fused_ep_moe

    if mesh is None:
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, -1), ("data", ep_axis_name))

    w1_packed = jnp.stack([w1, w3], axis=1)
    return fused_ep_moe(
        mesh=mesh,
        tokens=x,
        w1=w1_packed,
        w2=w2,
        gating_output=router_logits,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        ep_axis_name=ep_axis_name,
    )


def fused_moe_fused_routing(
    x: jax.Array,
    router_logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    top_k: int,
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
    mesh: Optional[jax.sharding.Mesh] = None,
    ep_axis_name: str = "model",
) -> jax.Array:
    """Fused routing+packing+MLP path (vendored TPU-inference kernel)."""
    return fused_moe_tpu_inference(
        x,
        router_logits,
        w1,
        w2,
        w3,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        mesh=mesh,
        ep_axis_name=ep_axis_name,
    )


def fused_moe_fused_routing_ep1(
    x: jax.Array,
    router_logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    top_k: int,
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    scoring_fn: str = "softmax",
    mesh: Optional[jax.sharding.Mesh] = None,
    ep_axis_name: str = "model",
) -> jax.Array:
    """Single-device fused routing+packing using the vendored kernel (EP=1)."""
    from levanter.kernels.pallas.moe.tpu_inference_v1 import fused_ep_moe

    if mesh is None:
        device = np.array(jax.devices()[:1]).reshape(1, 1)
        mesh = jax.sharding.Mesh(device, ("data", ep_axis_name))

    if mesh.shape[ep_axis_name] != 1:
        raise ValueError(f"Expected EP axis {ep_axis_name} to be size 1, got {mesh.shape}.")

    w1_packed = jnp.stack([w1, w3], axis=1)
    return fused_ep_moe(
        mesh=mesh,
        tokens=x,
        w1=w1_packed,
        w2=w2,
        gating_output=router_logits,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        ep_axis_name=ep_axis_name,
    )


def fused_moe(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    *,
    block_sizes: Optional[MoEBlockSizes] = None,
) -> jax.Array:
    if block_sizes is None:
        block_sizes = MoEBlockSizes.get_default()
    fn = _make_custom_vjp(block_sizes)
    return fn(x, topk_idx, topk_weights, w1, w2, w3)


__all__ = [
    "MoEBlockSizes",
    "fused_moe",
    "fused_moe_pallas",
    "fused_moe_pallas_staged",
    "fused_moe_staged",
    "fused_moe_pallas_gather_staged",
    "fused_moe_gather_staged",
    "fused_moe_fused_routing",
    "fused_moe_fused_routing_ep1",
    "fused_moe_tpu_inference",
]
