# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _forward_kernel(
    x_ref,
    W_ref,
    b_ref,
    h0_ref,
    y_ref,
    h_ref,
    h_scratch,
    BLOCK_SIZE_S: int,
    S: int,
    K: int,
    PAD: int,
    ACTIVATION: str | None,
) -> None:
    BLOCK_ID_S = pl.program_id(1)

    @pl.when(BLOCK_ID_S == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...]

    if h_ref is not None:
        h_ref[...] = h_scratch[...][None]

    dtype = x_ref.dtype
    H = x_ref.shape[-1]

    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    x = jnp.where(MASK_S, x_ref[...], 0).astype(dtype)
    x_f32 = x.astype(jnp.float32)
    b = jnp.zeros((1, H), dtype=jnp.float32) if b_ref is None else b_ref[...].astype(jnp.float32)

    hist_padded = jnp.pad(h_scratch[...].astype(jnp.float32), ((BLOCK_SIZE_S - PAD, 0), (0, 0)))

    taps = [
        jnp.where(BLOCK_S < shift, pltpu.roll(hist_padded, shift, axis=0), pltpu.roll(x_f32, shift, axis=0))
        for shift in (K - 1 - k for k in range(K))
    ]

    W = W_ref[...].astype(jnp.float32)
    y = jnp.sum(jnp.stack(taps, axis=0) * W[:, None, :], axis=0) + b

    if ACTIVATION in ["silu", "swish"]:
        y = y * jax.nn.sigmoid(y)

    y_ref[...] = y.astype(dtype)
    h_scratch[...] = x_f32[BLOCK_SIZE_S - PAD :, :].astype(dtype)


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "ACTIVATION"))
def _forward_core(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    BLOCK_SIZE_S: int,
    ACTIVATION: str | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    B, S, H = x.shape
    K = W.shape[0]
    PAD = _ceil_div(K - 1, 8) * 8
    NUM_BLOCKS_S = _ceil_div(S, BLOCK_SIZE_S)

    x_spec = pl.BlockSpec(
        block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0)
    )

    kernel = pl.pallas_call(
        partial(_forward_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, PAD=PAD, ACTIVATION=ACTIVATION),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((B, NUM_BLOCKS_S, PAD, H), x.dtype),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            x_spec,
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            None if b is None else pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            (
                None
                if h0 is None
                else pl.BlockSpec(
                    block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)
                )
            ),
        ),
        out_specs=(
            x_spec,
            pl.BlockSpec(
                block_shape=(None, 1, PAD, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
        ),
        scratch_shapes=[pltpu.VMEM((PAD, H), x.dtype)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(x, W, b, h0)
