# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity and timing of ``quack_quantgate.mxfp8_quantgate_wgrad`` (dW_up with in-kernel token quantization) against
today's path: ``quantize_rows_and_row_blocks(x)`` (the recompute's x quantizer, rows + tokens) +
``requantize_row_blocks(d_pre)`` + ``mxfp8_token_grouped_gemm``. The fused path keeps ``quantize_rows(x)`` (the up
GEMM recompute still needs x's rows). Hero EP64 chunk by default (262144/301568 rows, three experts, 3072 x 4608).

    python lib/levanter/scripts/bench/bench_mxfp8_quantgate_wgrad.py [--check-only]
"""

import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.quack_mxfp8 import (
    blocked_scales,
    mxfp8_token_grouped_gemm,
    quantize_token_blocks,
    token_scale_blocks,
)
from levanter.grug._moe.quack_quantgate import mxfp8_quantgate_wgrad
from levanter.grug.mxfp8_pallas import quantize_rows, quantize_rows_and_row_blocks, requantize_row_blocks


def _bits(a):
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint8))


def check(rows, cap, sizes, h, i, seed, gate_all_tiles=False):
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    x = jax.random.normal(k[0], (cap, h), jnp.bfloat16) * jnp.exp(jax.random.normal(k[1], (cap, 1), jnp.bfloat16))
    d = jax.random.normal(k[1], (cap, i), jnp.bfloat16) * 3.0
    d_rows = jax.jit(quantize_rows)(d)
    blocks = token_scale_blocks(cu, cap)

    @jax.jit
    def ref(x, dv, de):
        qa = quantize_token_blocks(x, cu)
        qb = requantize_row_blocks(dv, de, *blocks)
        return mxfp8_token_grouped_gemm(qa, qb, cu), qa, qb

    dw_ref, (ra, rae), (rb, rbe) = ref(x, *d_rows)
    dw, (a, sfa, b, sfb, counts) = jax.jit(
        lambda x, dv, de: mxfp8_quantgate_wgrad(x, (dv, de), cu, gate_all_tiles=gate_all_tiles)
    )(x, *d_rows)
    counts = np.asarray(counts)
    counts = counts[:-1]
    print(
        f"  counters min {counts.min()} max {counts.max()} (need {h // 128 + i // 128}), all-tile gate {gate_all_tiles}"
    )
    dw_b = jax.jit(lambda x, dv, de: mxfp8_quantgate_wgrad(x, (dv, de), cu, x_tokens=quantize_token_blocks(x, cu))[0])(
        x, *d_rows
    )
    n = int(cu[-1])
    res = {
        "A values": (_bits(a)[:n] == _bits(ra)[:n]).mean(),
        "A scales": (_bits(sfa) == _bits(blocked_scales(rae.T[None]))).mean(),
        "B values": (_bits(b)[:n] == _bits(rb)[:n]).mean(),
        "B scales": (_bits(sfb) == _bits(blocked_scales(rbe.T[None]))).mean(),
        "dW bitwise": (_bits(dw) == _bits(dw_ref)).mean(),
        "dW bitwise (B-only variant)": (_bits(dw_b) == _bits(dw_ref)).mean(),
        "dW ref finite": float(jnp.isfinite(dw_ref.astype(jnp.float32)).mean()),
        "dW finite": float(jnp.isfinite(dw.astype(jnp.float32)).mean()),
        "dW max abs diff": float(jnp.max(jnp.abs(dw.astype(jnp.float32) - dw_ref.astype(jnp.float32)))),
    }
    print(
        f"check rows={rows} cap={cap} sizes={list(sizes)} {h}x{i}: "
        + ", ".join(f"{k} {v:.6g}" for k, v in res.items()),
        flush=True,
    )
    return all(v == 1.0 for key, v in res.items() if key.startswith(("A", "B", "dW bitwise")))


def _time(fn, args, iters=10, reps=7):
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
    print(jax.devices()[0].device_kind, flush=True)
    if "--debug" in sys.argv:
        check(4000, 4096, np.array([1000, 1, 2999]), 256, 384, 0, gate_all_tiles=True)
    ok = check(4000, 4096, np.array([1000, 1, 2999]), 256, 384, 0)
    ok &= check(5000, 5120, np.array([0, 2500, 2500]), 384, 256, 1)
    ok &= check(262_144, 301_568, np.array([87381, 87381, 87382]), 3072, 4608, 2)
    print("PARITY", "OK" if ok else "FAIL", flush=True)
    if "--check-only" in sys.argv:
        return
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

    def fused(x, dv, de):
        return quantize_rows(x), mxfp8_quantgate_wgrad(x, (dv, de), cu)[0]

    def fused_b(x, dv, de):
        x_rows, x_tok = quantize_rows_and_row_blocks(x, *blocks)
        return x_rows, mxfp8_quantgate_wgrad(x, (dv, de), cu, x_tokens=x_tok)[0]

    def gemm_only(x, dv, de):
        return mxfp8_quantgate_wgrad(x, (dv, de), cu)[0]

    q = jax.block_until_ready(
        jax.jit(lambda x, dv, de: (quantize_token_blocks(x, cu), requantize_row_blocks(dv, de, *blocks)))(x, *d_rows)
    )
    for rnd in range(3):
        t_today = _time(today, (x, *d_rows))
        t_fused = _time(fused, (x, *d_rows))
        t_fused_b = _time(fused_b, (x, *d_rows))
        x_tok = jax.block_until_ready(jax.jit(lambda x: quantize_token_blocks(x, cu))(x))
        t_qg_b = _time(
            lambda x, dv, de, xv, xe: mxfp8_quantgate_wgrad(x, (dv, de), cu, x_tokens=(xv, xe))[0],
            (x, *d_rows, *x_tok),
        )
        print(
            f"  fused B-only (x rows+tok standalone, d_pre requant in-kernel): {t_fused_b:.3f} ms; "
            f"B-only quantgate gemm alone {t_qg_b:.3f} ms",
            flush=True,
        )
        t_qg = _time(gemm_only, (x, *d_rows))
        t_gemm = _time(lambda a, b: mxfp8_token_grouped_gemm(a, b, cu), q)
        t_serial = _time(
            lambda x, dv, de: mxfp8_quantgate_wgrad(x, (dv, de), cu, gate_all_tiles=True)[0], (x, *d_rows)
        )
        for parts in ("a", "b"):
            t_part = _time(
                lambda x, dv, de, parts=parts: mxfp8_quantgate_wgrad(
                    x, (dv, de), cu, gate_all_tiles=True, quant_parts=parts
                )[0],
                (x, *d_rows),
            )
            print(f"  quant part {parts} only, serial: {t_part:.3f} -> ~{t_part - t_gemm:.3f} ms", flush=True)
        print(
            f"round {rnd}: today (x rows+tok, d requant, gemm) {t_today:.3f} ms | fused (x rows, quantgate gemm) "
            f"{t_fused:.3f} ms | quantgate gemm alone {t_qg:.3f} | plain mxfp8 gemm alone {t_gemm:.3f} | "
            f"quantgate with all-tile gate (serial) {t_serial:.3f} -> quant alone ~{t_serial - t_gemm:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
