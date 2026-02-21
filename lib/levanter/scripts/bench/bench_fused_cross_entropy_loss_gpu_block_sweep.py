# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError(f"empty integer list from {raw!r}")
    return tuple(values)


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _bench(
    fn,
    x: jax.Array,
    w: jax.Array,
    y: jax.Array,
    steps: int,
    warmup: int,
) -> tuple[float, float]:
    jitted = jax.jit(fn)

    start = time.perf_counter()
    value = jitted(x, w, y)
    value.block_until_ready()
    compile_time_s = time.perf_counter() - start

    for _ in range(warmup):
        jitted(x, w, y).block_until_ready()

    start = time.perf_counter()
    for _ in range(steps):
        jitted(x, w, y).block_until_ready()
    steady_time_s = (time.perf_counter() - start) / steps
    return compile_time_s, steady_time_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pos", type=int, default=2048)
    parser.add_argument("--embed", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--input-dtype", type=str, default="bfloat16")
    parser.add_argument("--accum-dtype", type=str, default="float32")
    parser.add_argument("--b-block-sizes", type=str, default="128,256,1024")
    parser.add_argument("--h-block-sizes", type=str, default="64,128,256,512")
    parser.add_argument("--v-block-sizes", type=str, default="128,256,512,1024")
    parser.add_argument("--max-dot-tiles", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compare-xla", action="store_true", default=True)
    parser.add_argument("--no-compare-xla", action="store_false", dest="compare_xla")
    args = parser.parse_args()

    print("devices:", jax.devices())

    batch = args.batch
    pos = args.pos
    embed = args.embed
    vocab = args.vocab
    tokens = batch * pos
    input_dtype = jnp.dtype(args.input_dtype)
    accum_dtype = jnp.dtype(args.accum_dtype)
    b_candidates = _parse_int_csv(args.b_block_sizes)
    h_candidates = _parse_int_csv(args.h_block_sizes)
    v_candidates = _parse_int_csv(args.v_block_sizes)

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch * pos, embed), dtype=input_dtype)
    w = jax.random.normal(key_w, (embed, vocab), dtype=input_dtype)
    y = jax.random.randint(key_y, (batch * pos,), 0, vocab, dtype=jnp.int32)

    def _loss_fn(x_in, w_in, y_in, block_sizes: BlockSizes | None, implementation: str):
        return fused_cross_entropy_loss_and_logsumexp_penalty(
            x_in,
            y_in,
            w_in,
            reduction="mean",
            logsumexp_weight=0.0,
            dtype=accum_dtype,
            block_sizes=block_sizes,
            logit_soft_cap=None,
            implementation=implementation,
        )

    if args.compare_xla:
        compile_xla, steady_xla = _bench(
            lambda x_in, w_in, y_in: _loss_fn(x_in, w_in, y_in, None, "xla"),
            x,
            w,
            y,
            args.steps,
            args.warmup,
        )
        xla_tps = tokens / steady_xla
        print(
            "xla",
            "compile_s",
            f"{compile_xla:.6f}",
            "steady_s",
            f"{steady_xla:.6f}",
            "tokens_per_s",
            f"{xla_tps:.3f}",
        )

    for b_block_size, h_block_size, v_block_size in product(b_candidates, h_candidates, v_candidates):
        dot_tiles = _ceil_div(embed, h_block_size) * _ceil_div(vocab, v_block_size)
        if dot_tiles > args.max_dot_tiles:
            print(
                "pallas_gpu",
                f"b={b_block_size}",
                f"h={h_block_size}",
                f"v={v_block_size}",
                "SKIPPED",
                f"dot_tiles={dot_tiles}>max_dot_tiles={args.max_dot_tiles}",
            )
            continue
        block_sizes = BlockSizes(
            b_block_size=b_block_size,
            h_block_size=h_block_size,
            v_block_size=v_block_size,
        )
        try:
            compile_pallas, steady_pallas = _bench(
                lambda x_in, w_in, y_in: _loss_fn(x_in, w_in, y_in, block_sizes, "pallas_gpu"),
                x,
                w,
                y,
                args.steps,
                args.warmup,
            )
        except Exception as exc:  # pragma: no cover - backend/runtime dependent
            print(
                "pallas_gpu",
                f"b={b_block_size}",
                f"h={h_block_size}",
                f"v={v_block_size}",
                "FAILED",
                exc,
            )
            continue

        pallas_tps = tokens / steady_pallas
        delta = np.nan
        if args.compare_xla:
            delta = pallas_tps / xla_tps - 1.0
        print(
            "pallas_gpu",
            f"b={b_block_size}",
            f"h={h_block_size}",
            f"v={v_block_size}",
            "compile_s",
            f"{compile_pallas:.6f}",
            "steady_s",
            f"{steady_pallas:.6f}",
            "tokens_per_s",
            f"{pallas_tps:.3f}",
            "vs_xla",
            f"{delta:+.3%}" if args.compare_xla else "n/a",
        )


if __name__ == "__main__":
    main()
