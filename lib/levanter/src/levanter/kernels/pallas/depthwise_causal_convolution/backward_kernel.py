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
from .forward_kernel import _ceil_div, _pad_size


def _backward_kernel(
    x_ref,
    W_ref,
    b_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dx_ref,
    dW_ref,
    db_ref,
    dh0_ref,
    dy_scratch,
    dht_scratch,
    BLOCK_SIZE_S: int,
    S: int,
    K: int,
    PAD: int,
    NUM_BLOCKS_S: int,
    ACTIVATION: str | None,
) -> None:
    BLOCK_ID_B = pl.program_id(0)
    BLOCK_ID_S_REVERSE = pl.program_id(1)
    BLOCK_ID_S = NUM_BLOCKS_S - 1 - BLOCK_ID_S_REVERSE

    H = x_ref.shape[-1]
    dtype = x_ref.dtype
    offset = PAD - K + 1

    @pl.when(BLOCK_ID_S_REVERSE == 0)
    def _():
        dy_scratch[...] = jnp.zeros_like(dy_scratch)
        dht_scratch[...] = jnp.zeros_like(dht_scratch)
        if dht_ref is not None:
            dht_scratch[offset:, :] = dht_ref[...].astype(jnp.float32)

    @pl.when((BLOCK_ID_B == 0) & (BLOCK_ID_S_REVERSE == 0))
    def _():
        dW_ref[...] = jnp.zeros(dW_ref.shape, dtype=dW_ref.dtype)
        db_ref[...] = jnp.zeros(db_ref.shape, dtype=db_ref.dtype)

    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    PATCH_R = jax.lax.broadcasted_iota(jnp.int32, (PAD, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    MASKED = S % BLOCK_SIZE_S != 0
    if MASKED:
        dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32)
        x = jnp.where(MASK_S, x_ref[...], 0).astype(jnp.float32)
    else:
        dy = dy_ref[...].astype(jnp.float32)
        x = x_ref[...].astype(jnp.float32)

    hist = h_ref[0].astype(jnp.float32)  # hist[PAD + j] == x[j] for j < 0

    def x_tap(k: int):
        shift = K - 1 - k
        return pltpu.roll(x, shift, axis=0) if shift > 0 else x

    if ACTIVATION in ["silu", "swish"]:
        y = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
        if b_ref is not None:
            y += b_ref[...].astype(jnp.float32)

        for k in range(K):
            y += W_ref[k, :].astype(jnp.float32)[None, :] * x_tap(k)

        x_head = x[:PAD, :]
        y_head = jnp.zeros((PAD, H), dtype=jnp.float32)
        if b_ref is not None:
            y_head += b_ref[...].astype(jnp.float32)

        for k in range(K):
            shift = K - 1 - k
            if shift == 0:
                y_head += W_ref[k, :].astype(jnp.float32)[None, :] * x_head
            else:
                W = W_ref[k, :].astype(jnp.float32)[None, :]
                y_head += W * jnp.where(PATCH_R >= shift, pltpu.roll(x_head, shift, axis=0), 0)
                y_head += W * jnp.where(PATCH_R < shift, pltpu.roll(hist, shift, axis=0), 0)

        y = jnp.where(BLOCK_S < K - 1, jnp.pad(y_head, ((0, BLOCK_SIZE_S - PAD), (0, 0))), y)

        sigmoid = jax.nn.sigmoid(y)
        dy = dy * sigmoid * (1 + y * (1 - sigmoid))

    dy_bf = dy.astype(dtype)
    dx = jnp.zeros((BLOCK_SIZE_S, H), dtype=dtype)
    for k in range(K):
        W = W_ref[k, :]

        shift = K - 1 - k
        if shift == 0:
            dx += W[None, :] * dy_bf
        else:
            dx += W[None, :] * pltpu.roll(dy_bf, (BLOCK_SIZE_S - shift) % BLOCK_SIZE_S, axis=0)

    dy_tail = dy[BLOCK_SIZE_S - PAD :, :]
    carry = dy_scratch[...]
    dx_tail = jnp.zeros((PAD, H), dtype=jnp.float32)
    for k in range(K):
        W = W_ref[k, :].astype(jnp.float32)[None, :]

        shift = K - 1 - k
        if shift == 0:
            dx_tail += W * dy_tail
        else:
            up = PAD - shift
            dx_tail += W * jnp.where(
                PATCH_R < PAD - shift, pltpu.roll(dy_tail, up, axis=0), pltpu.roll(carry, up, axis=0)
            )

    dx = jnp.where(
        BLOCK_S >= BLOCK_SIZE_S - (K - 1),
        jnp.pad(dx_tail, ((BLOCK_SIZE_S - PAD, 0), (0, 0))).astype(dtype),
        dx,
    )

    ones_row = jnp.ones((1, BLOCK_SIZE_S), dtype=dtype)
    db_ref[...] += jax.lax.dot(ones_row, dy_bf, preferred_element_type=jnp.float32)

    d_head = dy[:PAD, :]
    for k in range(K):
        shift = K - 1 - k
        tap = x_tap(k)
        acc = jax.lax.dot(ones_row, (dy * tap).astype(dtype), preferred_element_type=jnp.float32)[0]
        if shift > 0:
            acc += jnp.sum(
                jnp.where(PATCH_R < shift, d_head * (pltpu.roll(hist, shift, axis=0) - tap[:PAD, :]), 0),
                axis=0,
            )
        dW_ref[k, :] += acc

    dx_ref[...] = (jnp.where(MASK_S, dx, 0) if MASKED else dx).astype(dtype)

    state_prefix = max(K - 1 - S, 0)
    x_state_start = max(S - (K - 1), 0)

    if dht_ref is not None:
        dht_rows = {}
        for p in range(state_prefix, K - 1):
            block_index, row_in_block = divmod(x_state_start + p - state_prefix, BLOCK_SIZE_S)
            dht_rows.setdefault(block_index, []).append((p, row_in_block))

        for block_index, rows in dht_rows.items():

            @pl.when(BLOCK_ID_S == block_index)
            def _(rows=rows):
                update = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
                for p, row_in_block in rows:
                    update += jnp.where(BLOCK_S == row_in_block, dht_scratch[offset + p, :], 0)

                dx_ref[...] += update.astype(dtype)

    @pl.when(BLOCK_ID_S == 0)
    def _():
        dh0 = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
        for k in range(K):
            W = W_ref[k, :].astype(jnp.float32)
            dh0 += W[None, :] * jnp.where(BLOCK_S < k, 0, pltpu.roll(dy, k, axis=0))

        dh0_ref[...] = pltpu.roll(dh0[:PAD, :], offset, axis=0)

        for p in range(state_prefix):
            dh0_ref[offset + S + p, :] += dht_scratch[offset + p, :]

    dy_scratch[...] = dy[:PAD, :]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K", "ACTIVATION"))
def _backward_core(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    BLOCK_SIZE_S: int,
    K: int,
    ACTIVATION: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, H = x.shape
    NUM_BLOCKS_S = _ceil_div(S, BLOCK_SIZE_S)
    PAD = _pad_size(K)
    state_size = K - 1

    # a block has to be long enough to carry the whole history in its leading/trailing PAD rows
    assert BLOCK_SIZE_S >= PAD, f"BLOCK_SIZE_S ({BLOCK_SIZE_S}) must be >= {PAD} for kernel_size ({K})"

    kernel = pl.pallas_call(
        partial(
            _backward_kernel,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
            K=K,
            PAD=PAD,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            ACTIVATION=ACTIVATION,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((K, H), jnp.float32),
            jax.ShapeDtypeStruct((1, H), jnp.float32),
            jax.ShapeDtypeStruct((B, PAD, H), jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            None if b is None else pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(
                block_shape=(None, 1, PAD, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, state_size, H),
                    index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0),
                )
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)),
        ),
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32), pltpu.VMEM((PAD, H), jnp.float32)],
        # dW_ref/db_ref alias the same output block across every BLOCK_ID_B and accumulate via
        # non-atomic +=, so the B dimension must run in program order like S, not in parallel
        # (megacore could otherwise race two batch programs on the same accumulator).
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
    )

    dx, dW, db, dh0 = kernel(x, W, b, h, dy, dht)

    return dx, dW, db, dh0
