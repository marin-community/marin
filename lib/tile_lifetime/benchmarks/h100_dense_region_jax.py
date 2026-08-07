# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the ordinary JAX/XLA dense reference region on H100."""

import argparse
import time

import jax
import jax.numpy as jnp

from tile_lifetime.reference import DenseDebugConfig, dense_region


def _random_bf16(key, shape: tuple[int, ...], *, scale: float = 1.0):
    return (jax.random.normal(key, shape, dtype=jnp.float32) * scale).astype(jnp.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14_336)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    config = DenseDebugConfig(
        batch=args.batch,
        sequence=args.sequence,
        hidden=args.hidden,
        intermediate=args.intermediate,
        query_heads=args.query_heads,
        key_value_heads=args.kv_heads,
        head_dimension=args.head_dimension,
    )
    qkv_width = config.qkv_width
    keys = jax.random.split(jax.random.key(0), 8)
    inputs = (
        _random_bf16(keys[0], (config.tokens, config.hidden)),
        _random_bf16(keys[1], (config.hidden, qkv_width), scale=config.hidden**-0.5),
        _random_bf16(keys[2], (config.hidden, config.hidden), scale=config.hidden**-0.5),
        jnp.ones((config.hidden,), dtype=jnp.bfloat16),
        _random_bf16(keys[3], (config.hidden, 2 * config.intermediate), scale=config.hidden**-0.5),
        _random_bf16(keys[4], (config.intermediate, config.hidden), scale=config.intermediate**-0.5),
        jnp.ones((config.hidden,), dtype=jnp.bfloat16),
        _random_bf16(keys[5], (config.hidden, qkv_width), scale=config.hidden**-0.5),
        _random_bf16(keys[6], (config.sequence, config.head_dimension // 2)),
        _random_bf16(keys[7], (config.sequence, config.head_dimension // 2)),
    )
    compiled = jax.jit(dense_region(config))

    compile_start = time.perf_counter()
    jax.block_until_ready(compiled(*inputs))
    compile_seconds = time.perf_counter() - compile_start
    for _ in range(args.warmups):
        jax.block_until_ready(compiled(*inputs))

    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*inputs))
        samples.append((time.perf_counter() - start) * 1e3)

    print(f"gpu={jax.devices()[0]}")
    print(f"jax={jax.__version__}")
    print(
        f"shape=B{config.batch} S{config.sequence} H{config.hidden} I{config.intermediate} "
        f"Hq{config.query_heads} Hkv{config.key_value_heads} D{config.head_dimension}"
    )
    print(f"compile_seconds={compile_seconds:.3f}")
    print(f"median_ms={sorted(samples)[len(samples) // 2]:.4f} min_ms={min(samples):.4f}")


if __name__ == "__main__":
    main()
