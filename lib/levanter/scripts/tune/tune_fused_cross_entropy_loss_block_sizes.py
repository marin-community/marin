# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import json
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
    parser.add_argument(
        "--implementation",
        type=str,
        default="pallas_tpu",
        choices=("pallas_tpu", "xla", "reference"),
        help="Kernel backend implementation to benchmark.",
    )
    parser.add_argument("--input-dtype", type=str, default="bfloat16", help="Input dtype for x and w.")
    parser.add_argument("--accum-dtype", type=str, default="float32", help="Accumulation dtype for loss math.")
    parser.add_argument("--steps", type=int, default=3, help="Steady-state timing iterations.")
    parser.add_argument(
        "--b-block-sizes",
        type=str,
        default=None,
        help="Comma-separated B block sizes. If set, performs a cartesian sweep.",
    )
    parser.add_argument(
        "--h-block-sizes",
        type=str,
        default=None,
        help="Comma-separated H block sizes. If set, performs a cartesian sweep.",
    )
    parser.add_argument(
        "--v-block-sizes",
        type=str,
        default=None,
        help="Comma-separated V block sizes. If set, performs a cartesian sweep.",
    )
    parser.add_argument(
        "--include-infer",
        action="store_true",
        help="Also benchmark infer/default behavior (block_sizes=None) for pallas_tpu.",
    )
    return parser.parse_args()


def _parse_csv_ints(value: str | None) -> list[int]:
    if value is None:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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

    input_dtype = jnp.dtype(args.input_dtype)
    accum_dtype = jnp.dtype(args.accum_dtype)

    x_raw = jax.random.normal(key_x, (batch, embed), dtype=input_dtype)
    w_raw = jax.random.normal(key_w, (embed, vocab), dtype=input_dtype)
    y_raw = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    configs: list[tuple[str, BlockSizes | None]]
    b_block_sizes = _parse_csv_ints(args.b_block_sizes)
    h_block_sizes = _parse_csv_ints(args.h_block_sizes)
    v_block_sizes = _parse_csv_ints(args.v_block_sizes)

    if args.implementation != "pallas_tpu":
        configs = [("none", None)]
    elif b_block_sizes or h_block_sizes or v_block_sizes:
        b_values = b_block_sizes or [BlockSizes.get_default().b_block_size]
        h_values = h_block_sizes or [BlockSizes.get_default().h_block_size]
        v_values = v_block_sizes or [BlockSizes.get_default().v_block_size]
        configs = [
            (f"b{b}_h{h}_v{v}", BlockSizes(b_block_size=b, h_block_size=h, v_block_size=v))
            for b, h, v in itertools.product(b_values, h_values, v_values)
        ]
    else:
        configs = [
            ("b1024_h128_v1024", BlockSizes(b_block_size=1024, h_block_size=128, v_block_size=1024)),
            ("b1024_h256_v1024", BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024)),
            ("b1024_h512_v1024", BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)),
            ("b1024_h256_v2048", BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=2048)),
            ("b1024_h512_v2048", BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048)),
            ("b1024_h256_v4096", BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=4096)),
            ("b1024_h512_v4096", BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=4096)),
            ("b2048_h256_v2048", BlockSizes(b_block_size=2048, h_block_size=256, v_block_size=2048)),
            ("b2048_h512_v2048", BlockSizes(b_block_size=2048, h_block_size=512, v_block_size=2048)),
        ]
    if args.include_infer and args.implementation == "pallas_tpu":
        configs.append(("infer", None))

    def make_loss_fn(block_sizes: BlockSizes | None):
        def loss_fn(x_in, w_in, y_in):
            return fused_cross_entropy_loss_and_logsumexp_penalty(
                x_in,
                y_in,
                w_in,
                reduction="mean",
                logsumexp_weight=0.0,
                block_sizes=block_sizes,
                dtype=accum_dtype,
                logit_soft_cap=None,
                implementation=args.implementation,
            )

        return jax.value_and_grad(loss_fn, argnums=(0, 1))

    for label, cfg in configs:
        print("config", label, cfg)
        loss_jit = jax.jit(make_loss_fn(cfg))
        result: dict[str, str | float | int] = {
            "label": label,
            "implementation": args.implementation,
            "status": "failed",
        }
        try:
            start = time.perf_counter()
            loss, out = loss_jit(x_raw, w_raw, y_raw)
            jax.block_until_ready(out)
            # out.block_until_ready()
            compile_time = time.perf_counter() - start

            steps = args.steps
            start = time.perf_counter()
            for _ in range(steps):
                out = loss_jit(x_raw, w_raw, y_raw)
                jax.block_until_ready(out)
            steady_time = (time.perf_counter() - start) / steps

            print("loss", float(loss))
            print("compile_time_s", compile_time)
            print("steady_time_s", steady_time)
            print("tokens_per_s", tokens / steady_time)
            result.update(
                {
                    "status": "ok",
                    "loss": float(loss),
                    "compile_time_s": compile_time,
                    "steady_time_s": steady_time,
                    "tokens_per_s": tokens / steady_time,
                }
            )
        except Exception as exc:
            print("failed", type(exc).__name__, exc)
            result.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        print("result_json", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
