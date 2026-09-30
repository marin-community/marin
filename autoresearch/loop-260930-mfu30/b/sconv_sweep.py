# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch-shape sweep of the streaming Triton short-conv kernels on one GB200.

Times the shard-local forward and backward at the hero shapes ([16, 4096, C], C in {6144, 1536}) for a grid
of `TritonShortConvTiles`, each call chained ten times inside one jit (unrolled, so no loop-carry copies) so
launch overhead is amortized, and prints the effective HBM bandwidth (forward moves 2 tensors, backward 3
plus the fp32 dw partials). Elementwise XLA ops with the same traffic (2 and 3 passes) calibrate the floor.

Usage (one GB200): python autoresearch/loop-260930-mfu30/b/sconv_sweep.py [--quick]
"""

import argparse
import itertools
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from levanter.kernels.pallas.short_conv import short_conv, short_conv_reference
from levanter.kernels.pallas.short_conv.triton_gpu import (
    TritonShortConvTiles,
    short_conv_triton_bwd_local,
    short_conv_triton_fwd_local,
)

CHAIN = 10


def _time(fn, *args, iters=10):
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3 / CHAIN


def _forward_chain(tiles):
    def run(weight, x, seg):
        for _ in range(CHAIN):
            x = short_conv_triton_fwd_local(weight, x, seg, exact_reference_rounding=True, tiles=tiles)
        return x

    return jax.jit(run)


def _backward_chain(tiles):
    def run(weight, x, seg, dy):
        dw = jnp.zeros(weight.shape, jnp.float32)
        for _ in range(CHAIN):
            dy, partials = short_conv_triton_bwd_local(weight, x, seg, dy, exact_reference_rounding=True, tiles=tiles)
            dw = dw + jnp.sum(partials, axis=0)
        return dy, dw

    return jax.jit(run)


def _api_chains(implementation):
    """Forward and backward chains through `short_conv` itself (dw reduction included)."""

    def fwd(weight, x, seg):
        for _ in range(CHAIN):
            x = short_conv(weight, x, seg, implementation=implementation)
        return x

    def bwd(weight, x, seg, dy):
        _, pullback = jax.vjp(lambda w, xx: short_conv(w, xx, seg, implementation=implementation), weight, x)
        dw = jnp.zeros(weight.shape, jnp.float32)
        for _ in range(CHAIN):
            step_dw, dy = pullback(dy)
            dw = dw + step_dw.astype(jnp.float32)
        return dy, dw

    return jax.jit(fwd), jax.jit(bwd)


def _calibration_ms(x, dy):
    """Elementwise ops with the kernels' traffic: 2 passes (read, write) and 3 (two reads, one write)."""

    def two_pass(x):
        for _ in range(CHAIN):
            x = jax.lax.optimization_barrier(x * jnp.bfloat16(0.5))
        return x

    def three_pass(x, dy):
        for _ in range(CHAIN):
            dy = jax.lax.optimization_barrier(x * dy)
        return dy

    return _time(jax.jit(two_pass), x), _time(jax.jit(three_pass), x, dy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        grid = [TritonShortConvTiles(64, 1024, 4, 1, 4), TritonShortConvTiles(128, 512, 4, 1, 8)]
    else:
        grid = [
            TritonShortConvTiles(chunk, block, warps, 1, rows)
            for chunk, block, warps, rows in itertools.product((32, 64, 128, 256), (256, 512, 1024), (2, 4, 8), (4, 8))
            if block // (32 * warps) in (2, 4, 8)
        ]
    rng = np.random.default_rng(0)
    for channels in (6144, 1536):
        batch, seq = 16, 4096
        keys = jax.random.split(jax.random.key(channels), 3)
        x = jax.random.normal(keys[0], (batch, seq, channels), jnp.bfloat16)
        weight = (jax.random.normal(keys[1], (4, channels)) * 0.5).astype(jnp.bfloat16)
        dy = jax.random.normal(keys[2], (batch, seq, channels), jnp.bfloat16)
        cuts = np.sort(rng.choice(np.arange(1, seq), size=(batch, 8)), axis=1)
        seg = jnp.asarray(np.stack([np.searchsorted(c, np.arange(seq), side="right") for c in cuts]).astype(np.int32))
        tensor_bytes = batch * seq * channels * 2
        two_pass_ms, three_pass_ms = _calibration_ms(x, dy)
        print(
            json.dumps(
                dict(
                    channels=channels,
                    calibration=dict(
                        two_pass_ms=round(two_pass_ms, 4),
                        three_pass_ms=round(three_pass_ms, 4),
                        two_pass_tbps=round(2 * tensor_bytes / two_pass_ms / 1e9, 2),
                        three_pass_tbps=round(3 * tensor_bytes / three_pass_ms / 1e9, 2),
                    ),
                )
            ),
            flush=True,
        )
        for implementation in ("pallas_gpu", "triton_gpu"):
            fwd, bwd = _api_chains(implementation)
            api_fwd_ms, api_bwd_ms = _time(fwd, weight, x, seg), _time(bwd, weight, x, seg, dy)
            print(
                json.dumps(
                    dict(channels=channels, api=implementation, fwd_ms=round(api_fwd_ms, 4), bwd_ms=round(api_bwd_ms, 4))
                ),
                flush=True,
            )
        ref_out = jax.jit(short_conv_reference)(weight, x, seg)
        _, ref_pull = jax.vjp(lambda w, xx: short_conv_reference(w, xx, seg), weight, x)
        ref_dw, ref_dx = jax.jit(ref_pull)(dy)
        ref_dw = np.asarray(ref_dw, np.float32)
        rows = []
        for tiles in grid:
            record = dict(
                channels=channels,
                tiles=[tiles.chunk, tiles.max_channel_block, tiles.num_warps, tiles.num_stages, tiles.rows_per_step],
            )
            try:
                out = jax.jit(
                    lambda w, xx, s, t=tiles: short_conv_triton_fwd_local(
                        w, xx, s, exact_reference_rounding=True, tiles=t
                    )
                )(weight, x, seg)
                dx, partials = jax.jit(
                    lambda w, xx, s, g, t=tiles: short_conv_triton_bwd_local(
                        w, xx, s, g, exact_reference_rounding=True, tiles=t
                    )
                )(weight, x, seg, dy)
                dw = np.asarray(jnp.sum(partials, axis=0).astype(weight.dtype), np.float32)
                record.update(
                    out_bitwise=bool(np.array_equal(np.asarray(out), np.asarray(ref_out))),
                    dx_bitwise=bool(np.array_equal(np.asarray(dx), np.asarray(ref_dx))),
                    dw_rel=float(np.max(np.abs(dw - ref_dw)) / np.max(np.abs(ref_dw))),
                )
                fwd_ms = _time(_forward_chain(tiles), weight, x, seg)
                partial_bytes = batch * (seq // tiles.chunk) * 4 * channels * 4
                bwd_ms = _time(_backward_chain(tiles), weight, x, seg, dy)
            except Exception as exc:  # a launch shape the compiler rejects is a sweep result, not a failure
                record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(json.dumps(record), flush=True)
                continue
            record.update(
                fwd_ms=round(fwd_ms, 4),
                bwd_ms=round(bwd_ms, 4),
                fwd_tbps=round(2 * tensor_bytes / fwd_ms / 1e9, 2),
                bwd_tbps=round((3 * tensor_bytes + partial_bytes) / bwd_ms / 1e9, 2),
            )
            rows.append(record)
            print(json.dumps(record), flush=True)
        best_fwd = min(rows, key=lambda r: r["fwd_ms"])
        best_bwd = min(rows, key=lambda r: r["bwd_ms"])
        print(json.dumps(dict(channels=channels, best_fwd=best_fwd, best_bwd=best_bwd)), flush=True)


if __name__ == "__main__":
    main()
