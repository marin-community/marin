# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P


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

    use_shard_map = os.getenv("TOKAMAX_SHARD_MAP", "0").lower() in {"1", "true", "yes"}
    data_shards = int(os.getenv("TOKAMAX_DATA_SHARDS", "0")) or len(jax.devices())

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

    if use_shard_map:
        if data_shards <= 0:
            raise ValueError("TOKAMAX_DATA_SHARDS must be positive when TOKAMAX_SHARD_MAP is enabled.")
        devices = jax.devices()[:data_shards]
        mesh = Mesh(np.array(devices), ("data",))

        def shard_loss(x_in, w_in, y_in):
            loss = linear_softmax_cross_entropy_loss(
                x_in,
                y_in,
                w_in,
                reduction=None,
                implementation="mosaic_tpu",
            )
            local_sum = jnp.sum(loss)
            total_sum = jax.lax.psum(local_sum, "data")
            total_denom = jax.lax.psum(x_in.shape[0], "data")
            return total_sum / total_denom

        loss_jit = jax.jit(
            jax.shard_map(
                shard_loss,
                in_specs=(P("data", None), P(None, None), P("data")),
                out_specs=P(),
                check_vma=False,
            )
        )
    else:
        loss_jit = jax.jit(loss_fn)

    if use_shard_map:
        set_mesh = getattr(jax, "set_mesh", None)
        token = None
        if set_mesh is None:
            token = mesh
        else:
            set_mesh(mesh)

        try:
            if token is None:
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
            else:
                with token:
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
        finally:
            if set_mesh is not None:
                pass
    else:
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
