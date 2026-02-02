# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune block sizes for fused cross-entropy kernel.")
    parser.add_argument("--batch", type=int, default=128, help="Global batch size.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--data-shards", type=int, default=4, help="Data-parallel shards to divide batch*seq.")
    parser.add_argument("--embed", type=int, default=512, help="Hidden dimension (H).")
    parser.add_argument("--vocab", type=int, default=128256, help="Vocabulary size (V).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("devices:", jax.devices())

    tokens = args.batch * args.seq_len
    if tokens % args.data_shards != 0:
        raise ValueError(f"batch*seq ({tokens}) must be divisible by data_shards ({args.data_shards}).")
    batch = tokens // args.data_shards
    embed = args.embed
    vocab = args.vocab
    print(
        "shape",
        {
            "global_batch": args.batch,
            "seq_len": args.seq_len,
            "data_shards": args.data_shards,
            "kernel_batch": batch,
            "embed": embed,
            "vocab": vocab,
        },
    )

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x_raw = jax.random.normal(key_x, (batch, embed), dtype=jnp.bfloat16)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=jnp.bfloat16)
    y_raw = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    configs = [
        BlockSizes(b_block_size=1024, h_block_size=128, v_block_size=1024),
        BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024),
        BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024),
        BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=2048),
        BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048),
        BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=4096),
        BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=4096),
        BlockSizes(b_block_size=2048, h_block_size=256, v_block_size=2048),
        BlockSizes(b_block_size=2048, h_block_size=512, v_block_size=2048),
    ]

    def make_loss_fn(block_sizes: BlockSizes):
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
                implementation="pallas_tpu",
            )

        return jax.value_and_grad(loss_fn, argnums=(0, 1))

    for cfg in configs:
        print("config", cfg)
        loss_jit = jax.jit(make_loss_fn(cfg))
        try:
            start = time.perf_counter()
            loss, out = loss_jit(x_raw, w_raw, y_raw)
            jax.block_until_ready(out)
            # out.block_until_ready()
            compile_time = time.perf_counter() - start

            steps = 3
            start = time.perf_counter()
            for _ in range(steps):
                out = loss_jit(x_raw, w_raw, y_raw)
                jax.block_until_ready(out)
            steady_time = (time.perf_counter() - start) / steps

            print("loss", float(loss))
            print("compile_time_s", compile_time)
            print("steady_time_s", steady_time)
            print("tokens_per_s", tokens / steady_time)
        except Exception as exc:
            print("failed", type(exc).__name__, exc)
            # print stack trace for debugging
            # import traceback
            # traceback.print_exc()


if __name__ == "__main__":
    main()
