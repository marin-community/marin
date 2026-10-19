# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU parity check and microbenchmark of the AttnRes history kernels at the per-GPU hero shape
(``[8, 16 * 4096, 6144]`` bf16 history, 2 gates) against the XLA (plain JAX) versions.

Run on one GPU: ``python -m experiments.grug.fast_track.bench_attn_res_history``. Prints, per filled-slot
count, each op's time, effective HBM bandwidth (bytes the op must move) and max error vs XLA."""

import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.attn_res_block_mix import (
    Tiles,
    block_mix_local,
    block_mix_reference,
    block_mix_weight_grads_local,
)
from experiments.grug.fast_track.attn_res_source_grads import accumulate_source_grads_local

N, G, T, D = 8, 2, 16 * 4096, 6144
ITERS = 20


def _time(fn, *args) -> float:
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / ITERS


def _time_donated(fn, buf, *args) -> float:
    """``fn``'s time when its first argument is donated: each call's output feeds the next."""
    buf = jax.block_until_ready(fn(buf, *args))
    start = time.perf_counter()
    for _ in range(ITERS):
        buf = fn(buf, *args)
    jax.block_until_ready(buf)
    return (time.perf_counter() - start) / ITERS


def _inputs(num_valid: int):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    sources = jax.random.normal(k1, (N, T, D), jnp.bfloat16)
    logits = jax.random.normal(k2, (N, G, T), jnp.float32)
    weights = jax.nn.softmax(jnp.where((jnp.arange(N) < num_valid)[:, None, None], logits, -jnp.inf), axis=0)
    d_mixed = jax.random.normal(k3, (G, T, D), jnp.bfloat16)
    return sources, weights, d_mixed


def _xla_weight_grads(sources, weights, d_mixed):
    _, vjp = jax.vjp(lambda w: block_mix_reference(sources, w), weights)
    return vjp(d_mixed)[0]


def _xla_source_grads(d_sources, weights, d_mixes):
    acc = d_sources.astype(jnp.float32)
    for g in range(G):
        acc = acc + weights[:, g][..., None] * d_mixes[g][None]
    return acc.astype(d_sources.dtype)


def _fused_source_grads(d_sources, weights, d_mixes, num_valid, tiles=None):
    kwargs = {} if tiles is None else {"tiles": tiles}
    return accumulate_source_grads_local(
        d_sources, tuple(weights[:, g] for g in range(G)), tuple(d_mixes[g] for g in range(G)), num_valid, **kwargs
    )


def sweep() -> None:
    """Time each kernel over a tile grid at 4 and 8 filled slots; print the best per kernel (sum of both)."""
    sources, weights, d_mixed = _inputs(N)
    d_mixes = d_mixed.astype(jnp.float32)
    grids = {
        "mix": [Tiles(tb, fb, w, 1) for tb in (8, 16) for fb in (64, 128) for w in (4, 8, 16)],
        "dw": [Tiles(tb, fb, w, 2) for tb in (4, 8) for fb in (64, 128) for w in (4, 8, 16)],
        "srcgrad": [Tiles(tb, fb, w, 1) for tb in (8, 16, 32) for fb in (64, 128) for w in (4, 8, 16)],
    }
    for name, tiles_grid in grids.items():
        results = []
        for tiles in tiles_grid:
            try:
                total = 0.0
                for num_valid in (4, N):
                    nv = jnp.int32(num_valid)
                    if name == "mix":
                        total += _time(jax.jit(functools.partial(block_mix_local, tiles=tiles)), sources, weights, nv)
                    elif name == "dw":
                        fn = jax.jit(functools.partial(block_mix_weight_grads_local, tiles=tiles))
                        total += _time(fn, sources, d_mixed, nv)
                    else:
                        fn = jax.jit(functools.partial(_fused_source_grads, tiles=tiles), donate_argnums=0)
                        total += _time_donated(fn, jnp.copy(sources), weights, d_mixes, nv)
                results.append((total, tiles))
            except Exception as e:  # a tile that does not compile or fit is skipped
                print(f"  {name} {tiles} failed: {type(e).__name__}")
        results.sort()
        for total, tiles in results[:4]:
            print(f"sweep {name}: {tiles} nv4+nv8 {total * 1e3:.3f} ms")


def main() -> None:
    if "--sweep" in sys.argv:
        sweep()
        return
    print("device", jax.devices()[0].device_kind)
    xla_mix = jax.jit(block_mix_reference)
    xla_dw = jax.jit(_xla_weight_grads)
    xla_sg = jax.jit(_xla_source_grads)
    mix = jax.jit(block_mix_local)
    dw = jax.jit(block_mix_weight_grads_local)
    sg = jax.jit(_fused_source_grads, donate_argnums=0)
    act = T * D * 2
    for num_valid in (1, 4, N):
        sources, weights, d_mixed = _inputs(num_valid)
        nv = jnp.int32(num_valid)
        d_mixes = d_mixed.astype(jnp.float32)
        err_mix = float(jnp.max(jnp.abs(mix(sources, weights, nv).astype(jnp.float32) - xla_mix(sources, weights))))
        ref_dw = xla_dw(sources, weights, d_mixed)
        err_dw = float(jnp.max(jnp.abs(dw(sources, d_mixed, nv) - ref_dw)[:num_valid]) / jnp.max(jnp.abs(ref_dw)))
        ref_sg = xla_sg(sources, weights, d_mixes)
        got_sg = sg(jnp.copy(sources), weights, d_mixes, nv)
        err_sg = float(jnp.max(jnp.abs(got_sg.astype(jnp.float32) - ref_sg.astype(jnp.float32))))
        rows = [
            ("mix xla", _time(xla_mix, sources, weights), N * act + G * act),
            ("mix pallas", _time(mix, sources, weights, nv), num_valid * act + G * act),
            ("dw xla", _time(xla_dw, sources, weights, d_mixed), N * act + G * act),
            ("dw pallas", _time(dw, sources, d_mixed, nv), num_valid * act + G * act),
            ("srcgrad xla", _time(xla_sg, sources, weights, d_mixes), 2 * N * act + 2 * G * act),
        ]
        buf = jnp.copy(sources)
        jax.block_until_ready(buf)
        start = time.perf_counter()
        for _ in range(ITERS):
            buf = sg(buf, weights, d_mixes, nv)
        jax.block_until_ready(buf)
        rows.append(("srcgrad pallas", (time.perf_counter() - start) / ITERS, 2 * num_valid * act + 2 * G * act))
        print(f"num_valid={num_valid} max err: mix {err_mix:.3g} dw(rel) {err_dw:.3g} srcgrad {err_sg:.3g}")
        for name, seconds, nbytes in rows:
            print(f"  {name:15s} {seconds * 1e3:7.3f} ms  {nbytes / seconds / 1e12:5.2f} TB/s (minimum bytes)")
    np.testing.assert_array_less(err_mix, 0.1)


if __name__ == "__main__":
    main()
