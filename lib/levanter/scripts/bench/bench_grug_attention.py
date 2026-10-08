# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Time one Grug attention layer, forward and forward+backward, on one GPU.

Defaults are the June Snowball per-GPU shape at global batch 64 on 8 GPUs: 8 sequences of 4096 tokens, 20 query
heads, 5 KV heads, head_dim 128, bf16. Each mask (sliding window 2048, global causal) runs for each
implementation. TFLOP/s counts only the FLOPs the mask needs: 2 matmuls forward and 5 backward (QK^T recompute,
dP, dV, dQ, dK) over the allowed (query, key) pairs.

    python lib/levanter/scripts/bench/bench_grug_attention.py --impl reference xla_flash gpu_pallas_triton_flash
    python lib/levanter/scripts/bench/bench_grug_attention.py --impl gpu_pallas_triton_flash --sweep
"""

import argparse
import itertools
import json
import statistics
import time

import jax
import jax.numpy as jnp

from levanter.grug.attention import AttentionMask, attention
from levanter.grug.attention._pallas_triton_flash import TritonFlashBlockSizes, pallas_triton_flash_attention

WARMUP_SECONDS = 1.0
WINDOWS = 10
WINDOW_SECONDS = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=20)
    parser.add_argument("--kv-heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--impl", nargs="+", default=["reference", "xla_flash", "gpu_pallas_triton_flash"])
    parser.add_argument("--sweep", action="store_true", help="Sweep Triton tile sizes instead of comparing impls.")
    parser.add_argument("--sweep-part", choices=["fwd", "dq", "dkv"], default="fwd")
    parser.add_argument("--masks", nargs="+", choices=["window", "causal"], default=["window", "causal"])
    return parser.parse_args()


def median_seconds(fn, *args) -> float:
    """Median of ten windows of back-to-back calls, after a warmup."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_SECONDS:
        jax.block_until_ready(fn(*args))
    per_call = []
    for _ in range(WINDOWS):
        calls = 0
        start = time.perf_counter()
        while time.perf_counter() - start < WINDOW_SECONDS:
            out = fn(*args)
            calls += 1
        jax.block_until_ready(out)
        per_call.append((time.perf_counter() - start) / calls)
    return statistics.median(per_call)


def make_fns(attend):
    def forward(q, k, v):
        return attend(q, k, v)

    def loss(q, k, v, cot):
        return jnp.sum(attend(q, k, v).astype(jnp.float32) * cot)

    return jax.jit(forward), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


def inputs(args):
    kq, kk, kv, kc = jax.random.split(jax.random.key(0), 4)
    shape_q = (args.batch, args.seq_len, args.q_heads, args.head_dim)
    shape_kv = (args.batch, args.seq_len, args.kv_heads, args.head_dim)
    q = jax.random.normal(kq, shape_q, jnp.bfloat16)
    k = jax.random.normal(kk, shape_kv, jnp.bfloat16)
    v = jax.random.normal(kv, shape_kv, jnp.bfloat16)
    cot = jax.random.normal(kc, shape_q, jnp.float32)
    return q, k, v, cot


def needed_flops(args, window: int | None) -> tuple[float, float]:
    if window is None or window >= args.seq_len:
        pairs = args.seq_len * (args.seq_len + 1) // 2
    else:
        pairs = window * (window + 1) // 2 + (args.seq_len - window) * window
    per_matmul = 2 * args.batch * args.q_heads * pairs * args.head_dim
    return 2 * per_matmul, 7 * per_matmul


def report(record: dict) -> None:
    print("RESULT " + json.dumps(record), flush=True)


def compare(args) -> None:
    q, k, v, cot = inputs(args)
    for mask_name in args.masks:
        window = args.window if mask_name == "window" else None
        mask = AttentionMask.causal(sliding_window=window)
        fwd_flops, total_flops = needed_flops(args, window)
        reference_out = None
        for impl in args.impl:
            forward, grad = make_fns(lambda q, k, v, impl=impl: attention(q, k, v, mask, implementation=impl))
            try:
                out = forward(q, k, v)
                if reference_out is None:
                    reference_out = out
                max_diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - reference_out.astype(jnp.float32))))
                t_fwd = median_seconds(forward, q, k, v)
                t_all = median_seconds(grad, q, k, v, cot)
            except Exception as e:  # report and continue with the other implementations
                report({"mask": mask_name, "impl": impl, "error": repr(e)[:500]})
                continue
            report(
                {
                    "mask": mask_name,
                    "impl": impl,
                    "fwd_ms": t_fwd * 1e3,
                    "fwd_bwd_ms": t_all * 1e3,
                    "fwd_tflops": fwd_flops / t_fwd / 1e12,
                    "fwd_bwd_tflops": total_flops / t_all / 1e12,
                    "max_abs_diff_vs_first": max_diff,
                }
            )


def sweep(args) -> None:
    """Time each tile/launch config for one kernel; the other kernels keep their defaults."""
    q, k, v, cot = inputs(args)
    if args.sweep_part == "fwd":
        grid = [
            dict(block_q=bq, block_k=bk, num_warps=w, num_stages=s)
            for bq, bk, w, s in itertools.product([64, 128, 256], [32, 64, 128], [4, 8], [1, 2])
        ]
    elif args.sweep_part == "dq":
        grid = [
            dict(block_q_dq=bq, block_k_dq=bk, num_warps_dq=w, num_stages_dq=s)
            for bq, bk, w, s in itertools.product([64, 128, 256], [32, 64], [4, 8], [1])
        ]
    else:
        grid = [
            dict(block_q_dkv=bq, block_k_dkv=bk, num_warps_dkv=w, num_stages_dkv=s)
            for bq, bk, w, s in itertools.product([128, 256], [32, 64], [4, 8], [1])
        ]
    for mask_name in args.masks:
        window = args.window if mask_name == "window" else None
        mask = AttentionMask.causal(sliding_window=window)
        fwd_flops, total_flops = needed_flops(args, window)
        for overrides in grid:
            blocks = TritonFlashBlockSizes(**overrides)
            forward, grad = make_fns(
                lambda q, k, v, blocks=blocks: pallas_triton_flash_attention(q, k, v, mask, block_sizes=blocks)
            )
            try:
                if args.sweep_part == "fwd":
                    t = median_seconds(forward, q, k, v)
                    tflops = fwd_flops / t / 1e12
                else:
                    t = median_seconds(grad, q, k, v, cot)
                    tflops = total_flops / t / 1e12
            except Exception as e:
                report({"mask": mask_name, "part": args.sweep_part, **overrides, "error": repr(e)[:300]})
                continue
            report({"mask": mask_name, "part": args.sweep_part, **overrides, "ms": t * 1e3, "tflops": tflops})


def main() -> None:
    args = parse_args()
    print(f"devices={jax.devices()} default_blocks={TritonFlashBlockSizes()}", flush=True)
    if args.sweep:
        sweep(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
