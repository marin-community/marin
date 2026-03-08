# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark unique-index row scatter on TPU with XLA vs SparseCore."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from levanter.grug.grug_moe_sparsecore import sparsecore_row_scatter_set_unique


def _parse_dtype(name: str) -> jnp.dtype:
    mapping = {
        "bf16": jnp.bfloat16,
        "f32": jnp.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}") from exc


def _xla_scatter(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    out = jnp.zeros((num_rows, updates.shape[1]), dtype=updates.dtype)
    return out.at[row_indices].set(updates, mode="drop")


def _bench(fn, updates, row_indices, *, warmup: int, iters: int) -> tuple[float, float]:
    step = jax.jit(fn)

    start = time.perf_counter()
    out = step(updates, row_indices)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - start

    for _ in range(warmup):
        out = step(updates, row_indices)
        jax.block_until_ready(out)

    start = time.perf_counter()
    for _ in range(iters):
        out = step(updates, row_indices)
        jax.block_until_ready(out)
    steady_s = (time.perf_counter() - start) / iters
    return compile_s, steady_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=163840)
    parser.add_argument("--updates", type=int, default=40960)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    if jax.default_backend() != "tpu":
        raise RuntimeError("This benchmark requires TPU backend")

    dtype = _parse_dtype(args.dtype)
    key_updates, key_perm = jax.random.split(jax.random.key(0))
    updates = jax.random.normal(key_updates, (args.updates, args.width), dtype=dtype)
    row_indices = jax.random.permutation(key_perm, args.rows)[: args.updates].astype(jnp.int32)

    print("devices", jax.devices())
    print("rows", args.rows, "updates", args.updates, "width", args.width, "dtype", dtype)

    compile_s, steady_s = _bench(
        lambda x, i: _xla_scatter(x, i, num_rows=args.rows),
        updates,
        row_indices,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"xla_scatter,compile_s={compile_s:.6f},steady_s={steady_s:.6f}")

    compile_s, steady_s = _bench(
        lambda x, i: sparsecore_row_scatter_set_unique(x, i, num_rows=args.rows),
        updates,
        row_indices,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"sparsecore_scatter,compile_s={compile_s:.6f},steady_s={steady_s:.6f}")


if __name__ == "__main__":
    main()
