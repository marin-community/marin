# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _sparsecore_info() -> tuple[int, int]:
    device_kind = jax.devices()[0].device_kind
    if device_kind in ("TPU v5", "TPU v5p", "TPU v6", "TPU v6 lite"):
        return 4 if device_kind in ("TPU v5", "TPU v5p") else 2, 8
    if device_kind == "TPU7x":
        return 2, 16
    raise NotImplementedError(f"Unsupported TPU device kind {device_kind!r}")


def make_sc_add_one(x_shape: tuple[int, int]):
    _, num_lanes = _sparsecore_info()
    dma_block = (8, 128)
    if x_shape[0] % dma_block[0] != 0 or x_shape[1] % dma_block[1] != 0:
        raise ValueError(f"x_shape must be divisible by {dma_block}; got {x_shape}")

    @jax.jit
    def sc_add_one(x):
        @partial(
            pl.pallas_call,
            out_shape=jax.ShapeDtypeStruct(x_shape, x.dtype),
            grid=(x_shape[0] // dma_block[0], x_shape[1] // dma_block[1]),
            in_specs=(pl.BlockSpec(dma_block, lambda i, j: (i, j)),),
            out_specs=pl.BlockSpec(dma_block, lambda i, j: (i, j)),
            compiler_params=pltpu.CompilerParams(
                kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
                dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
            ),
        )
        def kernel(x_ref, out_ref):
            @pl.loop(0, dma_block[0])
            def _(c0):
                @pl.loop(0, dma_block[1], step=num_lanes)
                def _(c1):
                    slc = (pl.ds(c0, 1), pl.ds(c1, num_lanes))
                    out_ref.at[*slc][...] = x_ref.at[*slc][...] + 1

        return kernel(x)

    return sc_add_one


@jax.jit
def tc_add_one(x):
    return x + 1


def _timeit(fn, x, *, warmup: int, iters: int) -> float:
    out = fn(x)
    jax.block_until_ready(out)
    for _ in range(warmup):
        out = fn(x)
        jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(x)
        jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    x_shape = (args.rows, args.cols)
    if x_shape[0] % 8 != 0 or x_shape[1] % 128 != 0:
        raise ValueError(f"rows must be divisible by 8 and cols by 128; got {x_shape}")

    x = jax.random.randint(jax.random.key(0), x_shape, 0, 64, dtype=jnp.int32)
    sc_add_one = make_sc_add_one(x_shape)

    @jax.jit
    def two_add_ones(x):
        return sc_add_one(x), tc_add_one(x)

    sc_out = sc_add_one(x)
    tc_out = tc_add_one(x)
    both_out = two_add_ones(x)
    jax.tree.map(jax.block_until_ready, (sc_out, tc_out, both_out))

    if not jnp.array_equal(sc_out, x + 1):
        raise AssertionError("sc_add_one mismatch")
    if not jnp.array_equal(tc_out, x + 1):
        raise AssertionError("tc_add_one mismatch")

    sc_t = _timeit(sc_add_one, x, warmup=args.warmup, iters=args.iters)
    tc_t = _timeit(tc_add_one, x, warmup=args.warmup, iters=args.iters)
    both_t = _timeit(two_add_ones, x, warmup=args.warmup, iters=args.iters)

    print("device_kind", jax.devices()[0].device_kind)
    print("shape", x_shape)
    print(f"sc_add_one_s={sc_t:.6f}")
    print(f"tc_add_one_s={tc_t:.6f}")
    print(f"two_add_ones_s={both_t:.6f}")
    print(f"sum_s={sc_t + tc_t:.6f}")
    print(f"overlap_ratio={(sc_t + tc_t) / both_t:.4f}")


if __name__ == "__main__":
    main()
