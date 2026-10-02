# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure one paged decode attention layer, excluding cache updates and model work."""

import argparse
import json
import os
import statistics
import subprocess
import time
from functools import partial

import jax
import jax.numpy as jnp

from levanter.grug.attention import ragged_paged_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["reference", "tpu", "gpu_pallas"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--window", type=int)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    dtype = jnp.dtype(args.dtype)
    pages_per_sequence = (args.context + args.page_size - 1) // args.page_size
    page_count = args.batch_size * pages_per_sequence
    q_key, kv_key = jax.random.split(jax.random.key(42))
    q = jax.random.normal(q_key, (args.batch_size, args.kv_heads, args.groups, args.head_dim), dtype)
    cache = jax.random.normal(kv_key, (page_count, args.page_size, 2 * args.kv_heads, args.head_dim), dtype)
    inputs = (
        q,
        cache,
        jnp.full((args.batch_size,), args.context, jnp.int32),
        jnp.arange(page_count, dtype=jnp.int32).reshape(args.batch_size, pages_per_sequence),
        jnp.arange(args.batch_size + 1, dtype=jnp.int32),
        jnp.array(args.batch_size, jnp.int32),
    )
    jax.block_until_ready(inputs)
    fn = jax.jit(
        partial(
            ragged_paged_attention,
            sm_scale=args.head_dim**-0.5,
            sliding_window=args.window,
            implementation=args.implementation,
        )
    )
    start = time.perf_counter()
    compiled = fn.lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    compiled(*inputs).block_until_ready()
    first_run_time = time.perf_counter() - start
    for _ in range(3):
        compiled(*inputs).block_until_ready()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        compiled(*inputs).block_until_ready()
        times.append(time.perf_counter() - start)
    elapsed = statistics.median(times)
    visible = args.context if args.window is None else min(args.context, args.window)
    kv_bytes = 2 * args.batch_size * visible * args.kv_heads * args.head_dim * dtype.itemsize
    flops = 4 * args.batch_size * visible * args.kv_heads * args.groups * args.head_dim
    print(
        json.dumps(
            {
                "kernel": "grug_paged_decode",
                "implementation": args.implementation,
                "shape": vars(args),
                "dtype": args.dtype,
                "backend": jax.default_backend(),
                "device_type": jax.devices()[0].device_kind,
                "device_count": 1,
                "block_sizes": (
                    {"page_size": args.page_size, "kv_splits": min(8, pages_per_sequence)}
                    if args.implementation == "gpu_pallas"
                    else None
                ),
                "compile_time": compile_time,
                "first_run_time": first_run_time,
                "steady_state_time": elapsed,
                "error": None,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                "logical_kv_bytes": kv_bytes,
                "logical_kv_bandwidth_gb": kv_bytes / elapsed / 1e9,
                "qk_av_flops": flops,
                "qk_av_arithmetic_intensity": flops / kv_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
