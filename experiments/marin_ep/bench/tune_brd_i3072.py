# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autotune the Blackwell ragged-dot TuningConfig at the 8/384 hero shapes.

The 8/384 hero (E384, top-8, intermediate 3072) misses every key in
``brd_expert_mlp._CONFIGS`` (tuned for 4/192 at intermediate 6272), so all
four per-leg GEMMs run the fallback config. This sweep covers the 8/384 keys:

  w13 fwd + dx bwd : (K=6144, N=6144)   [dx transposes w13: same key]
  w2 fwd           : (K=3072, N=6144)
  dact bwd (w2^T)  : (K=6144, N=3072)

Rows per device at the canonical config: 65,536 tokens x top-8 x cf 1.1
~= 576,717, padded to 2*TILE_M; G=6 local experts. Run on one GB200:

  python experiments/marin_ep/bench/tune_brd_i3072.py [--quick]
"""

import itertools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu
from jax.experimental.pallas.ops.gpu import blackwell_ragged_dot_mgpu as brd

TOKENS_PER_DEV = 65536
TOPK = 8
CF = 1.1
G = 6
HIDDEN = 6144
INTER = 3072
ROW_ALIGN = 256  # 2 * tile_m


def _cfg(tile_n: int, grid_tile_width: int, grid_minor_dim: int, max_concurrent_steps: int) -> "brd.TuningConfig":
    return brd.TuningConfig(
        tile_m=128,
        tile_n=tile_n,
        tile_k=64,
        grid_tile_width=grid_tile_width,
        grid_minor_dim=blackwell_matmul_mgpu.MatmulDimension(grid_minor_dim),
        max_concurrent_steps=max_concurrent_steps,
        collective=True,
    )


def bench_ms(fn, *args, iters: int = 8) -> float:
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3


def main() -> None:
    quick = "--quick" in sys.argv
    rows = int(CF * TOKENS_PER_DEV * TOPK)
    rows += -rows % ROW_ALIGN
    rng = np.random.default_rng(0)
    gs = jnp.full((G,), rows // G, dtype=jnp.int32)

    shapes = {
        "w13_dx": (HIDDEN, HIDDEN),  # (K, N): w13 fwd [6144 -> 2*3072]; dx bwd shares the key
        "w2": (INTER, HIDDEN),  # w2 fwd [3072 -> 6144]
        "dact": (HIDDEN, INTER),  # dy @ w2^T [6144 -> 3072]
    }

    tile_ns = (128,) if quick else (64, 128)
    widths = (8, 12) if quick else (4, 8, 12, 16)
    minors = (0, 1)
    steps = (6,) if quick else (4, 5, 6)

    for name, (k_dim, n_dim) in shapes.items():
        a = jnp.asarray(rng.standard_normal((rows, k_dim)), dtype=jnp.bfloat16)
        b = jnp.asarray(0.02 * rng.standard_normal((G, k_dim, n_dim)), dtype=jnp.bfloat16)
        flops = 2 * rows * k_dim * n_dim
        results = []
        for tile_n, width, minor, mcs in itertools.product(tile_ns, widths, minors, steps):
            config = _cfg(tile_n, width, minor, mcs)
            fn = jax.jit(lambda a_, b_, g_, config=config: brd.ragged_dot_kernel(a_, b_, g_, config=config))
            try:
                ms = bench_ms(fn, a, b, gs)
            except Exception as exc:
                print(f"{name} tn={tile_n} w={width} m={minor} mcs={mcs}: FAIL {type(exc).__name__}", flush=True)
                continue
            tf = flops / (ms * 1e-3) / 1e12
            results.append((tf, ms, tile_n, width, minor, mcs))
            print(f"{name} tn={tile_n} w={width} m={minor} mcs={mcs}: {ms:.3f} ms = {tf:.0f} TF/s", flush=True)
        results.sort(reverse=True)
        tf, ms, tile_n, width, minor, mcs = results[0]
        print(
            f"BEST {name} (K={k_dim}, N={n_dim}): _cfg({tile_n}, {width}, {minor}, {mcs})"
            f"  # {tf:.0f} TF/s ({ms:.3f} ms)",
            flush=True,
        )


if __name__ == "__main__":
    main()
