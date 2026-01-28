# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import time

import jax
import jax.numpy as jnp


def main() -> None:
    from absl import flags as absl_flags
    from tokamax._src.ops.linear_softmax_cross_entropy_loss.api import (
        linear_softmax_cross_entropy_loss,
    )

    absl_flags.FLAGS(["bench_tokamax_linear_softmax_ce"])
    jax.config.update("jax_default_matmul_precision", "highest")

    print("devices:", jax.devices())

    batch = int(os.getenv("TOKAMAX_BATCH", "1024"))
    embed = int(os.getenv("TOKAMAX_EMBED", "1024"))
    vocab = int(os.getenv("TOKAMAX_VOCAB", "32768"))
    dtype_name = os.getenv("TOKAMAX_DTYPE", "f32").lower()
    dtype = jnp.bfloat16 if dtype_name in {"bf16", "bfloat16"} else jnp.float32

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x_raw = jax.random.normal(key_x, (batch, embed), dtype=dtype)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=dtype)
    y_raw = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    def loss_fn(x_in, w_in, y_in):
        return linear_softmax_cross_entropy_loss(
            x_in,
            y_in,
            w_in,
            reduction="mean",
            implementation="mosaic_tpu",
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
