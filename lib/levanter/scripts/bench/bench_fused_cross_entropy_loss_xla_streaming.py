# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import time

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


def main() -> None:
    print("devices:", jax.devices())

    batch = 8192
    embed = 4096
    vocab = 128256

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x_raw = jax.random.normal(key_x, (batch, embed), dtype=jnp.bfloat16)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=jnp.bfloat16)
    y_raw = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    block_sizes = BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)

    def loss_fn(x_in, w_in, y_in):
        return fused_cross_entropy_loss_and_logsumexp_penalty(
            x_in,
            y_in,
            w_in,
            reduction="mean",
            logsumexp_weight=0.0,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=None,
            implementation="xla",
        )

    loss_jit = jax.jit(loss_fn)

    start = time.perf_counter()
    out = loss_jit(x_raw, w_raw, y_raw)
    out.block_until_ready()
    compile_time = time.perf_counter() - start

    steps = 3
    start = time.perf_counter()
    for _ in range(steps):
        out = loss_jit(x_raw, w_raw, y_raw)
        out.block_until_ready()
    steady_time = (time.perf_counter() - start) / steps

    tokens = batch
    print("loss", float(out))
    print("compile_time_s", compile_time)
    print("steady_time_s", steady_time)
    print("tokens_per_s", tokens / steady_time)


if __name__ == "__main__":
    main()
