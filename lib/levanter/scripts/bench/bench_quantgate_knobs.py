# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlapped time of ``mxfp8_quantgate_wgrad`` per (quant warps, operands quantized in the launch) at the hero chunk,
with the full-path comparison: today = x rows+tokens + d_pre requant + GEMM."""

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.quack_mxfp8 import mxfp8_token_grouped_gemm, token_scale_blocks
from levanter.grug._moe.quack_quantgate import mxfp8_quantgate_wgrad
from levanter.grug.mxfp8_pallas import quantize_rows, quantize_rows_and_row_blocks, requantize_row_blocks


def _time(fn, args, iters=10, reps=5):
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            out = jitted(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / iters * 1e3)
    return statistics.median(samples)


def main():
    rows, cap, e, h, i = 262_144, 301_568, 3, 3072, 4608
    sizes = np.full(e, rows // e)
    sizes[-1] += rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    blocks = token_scale_blocks(cu, cap)
    k = jax.random.split(jax.random.PRNGKey(3), 2)
    x = jax.random.normal(k[0], (cap, h), jnp.bfloat16)
    d_rows = jax.jit(quantize_rows)(jax.random.normal(k[1], (cap, i), jnp.bfloat16))

    def today(x, dv, de):
        x_rows, x_tok = quantize_rows_and_row_blocks(x, *blocks)
        return x_rows, mxfp8_token_grouped_gemm(x_tok, requantize_row_blocks(dv, de, *blocks), cu)

    def full(warps, claim=4, **kw):
        def run(x, dv, de):
            return quantize_rows(x), mxfp8_quantgate_wgrad(x, (dv, de), cu, quant_warps=warps, claim=claim, **kw)[0]

        return run

    def b_only(warps):
        def run(x, dv, de):
            x_rows, x_tok = quantize_rows_and_row_blocks(x, *blocks)
            return (
                x_rows,
                mxfp8_quantgate_wgrad(x, (dv, de), cu, x_tokens=x_tok, quant_warps=warps, static_split=True)[0],
            )

        return run

    def a_only(warps):  # timing only: B left unquantized in the launch, requantized beforehand
        def run(x, dv, de):
            b = requantize_row_blocks(dv, de, *blocks)
            return (
                quantize_rows(x),
                b,
                mxfp8_quantgate_wgrad(x, (dv, de), cu, quant_parts="a", quant_warps=warps, static_split=True)[0],
            )

        return run

    for rnd in range(2):
        print(f"round {rnd}: today {_time(today, (x, *d_rows)):.3f} ms", flush=True)
        print(f"  static: full {_time(full(8, static_split=True), (x, *d_rows)):.3f}", flush=True)
        for warps in (8,):
            print(
                f"  warps {warps}: full {_time(full(warps), (x, *d_rows)):.3f} | B-only {_time(b_only(warps), (x, *d_rows)):.3f}"
                f" | A-only (timing) {_time(a_only(warps), (x, *d_rows)):.3f} ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
