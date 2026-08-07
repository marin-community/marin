# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the executable Hopper attention templates used by the prototype."""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

QUERY_HEADS = 32
KEY_VALUE_HEADS = 8
HEAD_DIMENSION = 128


def _benchmark(
    name: str,
    function: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, float | int | str]:
    compile_start = time.perf_counter()
    executable = jax.jit(function).lower(query, key, value).compile()
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(warmups):
        output = executable(query, key, value)
    output.block_until_ready()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            output = executable(query, key, value)
        output.block_until_ready()
        samples.append((time.perf_counter() - start) * 1_000 / iterations)

    return {
        "backend": name,
        "compile_seconds": compile_seconds,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "repeats": repeats,
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, action="append", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--include-segmented", action="store_true")
    parser.add_argument("--include-fa4-thd", action="store_true")
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "environment": {
                    "jax": jax.__version__,
                    "devices": [str(device) for device in jax.devices()],
                }
            }
        )
    )
    for sequence in args.sequence:
        random_keys = jax.random.split(jax.random.key(sequence), 3)
        query = jax.random.normal(
            random_keys[0],
            (args.batch, sequence, QUERY_HEADS, HEAD_DIMENSION),
            dtype=jnp.bfloat16,
        )
        key = jax.random.normal(
            random_keys[1],
            (args.batch, sequence, KEY_VALUE_HEADS, HEAD_DIMENSION),
            dtype=jnp.bfloat16,
        )
        value = jax.random.normal(
            random_keys[2],
            (args.batch, sequence, KEY_VALUE_HEADS, HEAD_DIMENSION),
            dtype=jnp.bfloat16,
        )
        candidates: list[tuple[str, Callable[[jax.Array, jax.Array, jax.Array], jax.Array]]] = [
            (
                "jax_xla",
                lambda q, k, v: jax.nn.dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    implementation="xla",
                ),
            ),
        ]
        if args.include_fa4_thd:
            from levanter.grug.attention import AttentionMask, gpu_fa4_thd_attention  # noqa: PLC0415

            segment_ids = jnp.zeros((args.batch, sequence), dtype=jnp.int32)
            thd_mask = AttentionMask.causal().with_segment_ids(segment_ids, max_segments=1)
            candidates.append(
                (
                    "flash_attn_4_thd_sm90",
                    lambda q, k, v, mask=thd_mask: gpu_fa4_thd_attention(q, k, v, mask),
                )
            )
        if args.include_segmented:
            from levanter.grug.attention import AttentionMask, gpu_fa4_cute_attention  # noqa: PLC0415

            causal_mask = AttentionMask.causal()
            candidates.insert(
                1,
                (
                    "fa4_segmented_bshd",
                    lambda q, k, v, mask=causal_mask: gpu_fa4_cute_attention(q, k, v, mask),
                ),
            )

        for name, candidate in candidates:
            result = _benchmark(
                name,
                candidate,
                query,
                key,
                value,
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=args.iterations,
            )
            result.update(
                {
                    "batch": args.batch,
                    "sequence": sequence,
                    "query_heads": QUERY_HEADS,
                    "key_value_heads": KEY_VALUE_HEADS,
                    "head_dimension": HEAD_DIMENSION,
                }
            )
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
