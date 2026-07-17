# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-002b/004: Blackwell CuTe DSL MXFP8 grouped GEMM -- fwd + dgrad + wgrad + producers.

Benchmarks all three MoE ragged_dot products of the vendored cutlass v4.5.2
scaled-grouped-GEMM kernel (tcgen05 block-scaled MMA, e4m3 data + e8m0
block-32 scales) through ``mxfp8_grouped/adapter.py``, at grug row-13
per-device shapes (M = 262144 tokens, E = 64):

  w13   fwd   x[M, 2560] @ w[E, 2560, 1280] -> [M, 1280]        (2Dx3D)
        dgrad g[M, 1280] @ w^T[E, 1280, 2560] -> [M, 2560]      (2Dx3D)
        wgrad x^T[2560, M] grouped-outer g[M, 1280] -> [E, 2560, 1280]  (2Dx2D)
  w2    the mirrored K=1280/N=2560 versions

first uniformly routed (4096/expert), then a skewed Dirichlet(0.5) routing
(group sizes multiples of 32, some experts ~0).

Correctness gate per product: rel-Frobenius error < 1e-3 vs a per-expert dense
reference computed from the SAME dequantized operands (bf16-rounded), plus the
error vs the unquantized reference (~4e-2 expected quantization noise).

The producer study times the MXFP8 quantize+swizzle cost (uniform routing):
naive per-orientation producers (XLA quantize + per-group swizzle loop) vs the
fused dual-orientation producer (``dual_quantize_activation`` /
``dual_quantize_weight``), and reports a layer-quad honest total:
fwd + dgrad + wgrad GEMMs + all producer costs vs 3x the bf16 dense yardstick.

Single GPU, GB200 (sm100) only.

Usage: python bench_mxfp8_grouped.py [--tokens 262144] [--iters 50]
         [--products fwd,dgrad,wgrad,producer] [--out out.json]
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cutlass
import jax
import jax.numpy as jnp
import numpy as np
from mxfp8_grouped.adapter import (
    build_sf_wgrad,
    build_sf_wgrad_fast,
    build_sfa,
    build_sfa_fast,
    build_sfb,
    build_sfb_fast,
    dequantize_mxfp8,
    dequantize_mxfp8_tokens,
    dual_quantize_activation,
    dual_quantize_weight,
    mxfp8_grouped_dgrad,
    mxfp8_grouped_mm,
    mxfp8_grouped_wgrad,
    quantize_mxfp8,
    quantize_mxfp8_tokens,
    sf_wgrad_col_layout,
    sfa_row_gather_indices,
)

E = 64
# (label, K, N): row-13 per-device grouped GEMMs (d2560, F1280).
SHAPES = [
    ("w13_k2560_n1280", 2560, 1280),
    ("w2_k1280_n2560", 1280, 2560),
]
GROUP_ALIGN = 32  # keep group sizes multiples of the MXFP8 block size


def uniform_groups(m: int) -> list[int]:
    assert m % E == 0
    return [m // E] * E


def skewed_groups(m: int, seed: int = 0) -> list[int]:
    """Dirichlet(0.5) routing in units of 32 tokens (mirrors the vendored harness)."""
    rng = np.random.default_rng(seed)
    n_slots = m // GROUP_ALIGN
    proportions = rng.dirichlet([0.5] * E)
    raw = np.floor(proportions * n_slots).astype(int)
    deficit = n_slots - raw.sum()
    order = np.argsort(-proportions)
    i = 0
    while deficit > 0:
        raw[order[i % E]] += 1
        deficit -= 1
        i += 1
    assert raw.sum() == n_slots
    return [int(s) * GROUP_ALIGN for s in raw]


def rel_frob(a, ref):
    a = jnp.asarray(a, jnp.float32)
    ref = jnp.asarray(ref, jnp.float32)
    return float(jnp.linalg.norm(a - ref) / jnp.linalg.norm(ref))


def timed(fn, args, iters, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def as_u8(x):
    return jax.lax.bitcast_convert_type(x, jnp.uint8)


def assert_bit_equal(a, b, what: str):
    same = bool(jnp.all(as_u8(a) == as_u8(b)))
    assert same, f"{what}: fast swizzle differs from naive"


def ragged_reference(x32, w32_ekn, group_sizes: list[int]):
    """Per-expert f32 matmul reference for the 2Dx3D products.

    Plain dense dots per group instead of ``jax.lax.ragged_dot``: the f32
    ragged_dot triton autotune tries to profile ~160 GiB fusions at these
    shapes and OOMs the pool (observed on GB200, jax 0.10.1).
    """
    outs = []
    start = 0
    for ei, g in enumerate(group_sizes):
        if g > 0:
            outs.append(x32[start : start + g] @ w32_ekn[ei])
        start += g
    return jnp.concatenate(outs, axis=0)


def wgrad_reference(x32, g32, group_sizes: list[int], k: int, n: int):
    """Per-expert f32 grouped-outer reference: dw[e] = x[s_e].T @ g[s_e]."""
    outs = []
    start = 0
    for g in group_sizes:
        if g > 0:
            outs.append(x32[start : start + g].T @ g32[start : start + g])
        else:
            outs.append(jnp.zeros((k, n), jnp.float32))
        start += g
    return jnp.stack(outs)


def print_arm(label: str, arm: str, r: dict):
    print(f"  {arm:13s} {r['ms']:8.3f} ms  {r['tfs']:7.1f} TF/s  [{label}]", flush=True)


def run_case(
    label: str,
    k: int,
    n: int,
    group_sizes: list[int],
    iters: int,
    warmup: int,
    mma_tiler: tuple[int, int, int],
    products: set[str],
) -> dict:
    m = sum(group_sizes)
    gs_dev = jnp.array(group_sizes, dtype=jnp.int32)
    offs = jnp.cumsum(gs_dev).astype(jnp.int32)
    flops = 2 * m * k * n
    res: dict = {"m": m, "k": k, "n": n, "arms": {}, "errors": {}}
    print(f"\n== {label} M={m} K={k} N={n} ==", flush=True)

    key = jax.random.PRNGKey(0)
    kx, kw, kg = jax.random.split(key, 3)
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (E, k, n), dtype=jnp.bfloat16) / (k**0.5)
    g = jax.random.normal(kg, (m, n), dtype=jnp.bfloat16)  # cotangent of the fwd output

    # Host swizzle index maps (fixed per case; the row map depends only on
    # group sizes, so x and g share it).
    row_idx = jnp.asarray(sfa_row_gather_indices(group_sizes))
    col_idx_np, perm_k_np = sf_wgrad_col_layout(group_sizes, k)
    _, perm_n_np = sf_wgrad_col_layout(group_sizes, n)
    col_idx = jnp.asarray(col_idx_np)
    perm_k = jnp.asarray(perm_k_np)
    perm_n = jnp.asarray(perm_n_np)

    def bench_kernel(arm: str, fn, args, out_shape):
        out = jax.block_until_ready(fn(*args))
        assert out.shape == out_shape, (arm, out.shape, out_shape)
        t = timed(fn, args, iters, warmup)
        res["arms"][arm] = {"ms": t * 1e3, "tfs": flops / t / 1e12}
        print_arm(label, arm, res["arms"][arm])
        return out

    def check(arm: str, out, ref32):
        err_bf = rel_frob(out, ref32.astype(jnp.bfloat16))
        err_f32 = rel_frob(out, ref32)
        res["errors"][arm] = {"vs_deq_bf16ref": err_bf, "vs_deq_f32ref": err_f32}
        print(f"  {arm} err(deq ref bf16) {err_bf:.2e}  err(deq ref f32) {err_f32:.2e}", flush=True)
        assert err_bf < 1e-3, f"{label}/{arm}: rel-Frobenius {err_bf:.3e} >= 1e-3"

    # ---------------------------------------------------------------- fwd --
    if "fwd" in products:
        x_q, x_scales = jax.jit(quantize_mxfp8)(x)
        w_fwd_q, w_fwd_scales = jax.jit(quantize_mxfp8)(jnp.swapaxes(w, 1, 2))  # (E, N, K) along K
        x_sf = jax.jit(build_sfa_fast)(x_scales, row_idx)
        assert_bit_equal(x_sf, build_sfa(x_scales, group_sizes), f"{label} sfa")
        w_sf = jax.jit(build_sfb_fast)(w_fwd_scales)
        assert_bit_equal(w_sf, build_sfb(w_fwd_scales), f"{label} sfb")
        jax.block_until_ready((x_q, w_fwd_q, x_sf, w_sf))

        fwd_fn = jax.jit(lambda a, b, asf, bsf, o: mxfp8_grouped_mm(a, asf, b, bsf, o, mma_tiler_mnk=mma_tiler))
        out = bench_kernel("mxfp8_fwd", fwd_fn, (x_q, w_fwd_q, x_sf, w_sf, offs), (m, n))

        x_deq = dequantize_mxfp8(x_q, x_scales)
        w_deq = jnp.swapaxes(dequantize_mxfp8(w_fwd_q, w_fwd_scales), 1, 2)  # (E, K, N)
        ref = ragged_reference(x_deq, w_deq, group_sizes)
        check("mxfp8_fwd", out, ref)
        err_quant = rel_frob(out, ragged_reference(x.astype(jnp.float32), jnp.asarray(w, jnp.float32), group_sizes))
        res["errors"]["mxfp8_fwd"]["vs_unquantized"] = err_quant
        print(f"  mxfp8_fwd err(vs unquantized) {err_quant:.2e}", flush=True)
        del x_q, x_scales, w_fwd_q, w_fwd_scales, x_sf, w_sf, out, x_deq, w_deq, ref

        # bf16 baselines (fwd only; dgrad/wgrad have identical FLOPs and the
        # dense yardstick is layout-insensitive).
        ragged_fn = jax.jit(lambda xx, ww, gg: jax.lax.ragged_dot(xx, ww, gg, preferred_element_type=jnp.bfloat16))
        t = timed(ragged_fn, (x, w, gs_dev), iters, warmup)
        res["arms"]["bf16_ragged"] = {"ms": t * 1e3, "tfs": flops / t / 1e12}
        print_arm(label, "bf16_ragged", res["arms"]["bf16_ragged"])

        w_dense = w[0]  # (K, N); dense equal-FLOPs yardstick: 2*M*K*N
        dense_fn = jax.jit(lambda xx, ww: jnp.einsum("mk,kn->mn", xx, ww, preferred_element_type=jnp.bfloat16))
        t = timed(dense_fn, (x, w_dense), iters, warmup)
        res["arms"]["bf16_dense"] = {"ms": t * 1e3, "tfs": flops / t / 1e12}
        print_arm(label, "bf16_dense", res["arms"]["bf16_dense"])

    # -------------------------------------------------------------- dgrad --
    # dx[M, K] = g[M, N] @ w^T[E, N, K]: 2Dx3D with contraction N; g quantized
    # along N (natural), w quantized along N in its natural (E, K, N) buffer.
    if "dgrad" in products:
        g_q, g_scales = jax.jit(quantize_mxfp8)(g)
        w_dg_q, w_dg_scales = jax.jit(quantize_mxfp8)(w)  # (E, K, N) along N
        g_sf = jax.jit(build_sfa_fast)(g_scales, row_idx)
        w_dg_sf = jax.jit(build_sfb_fast)(w_dg_scales)
        jax.block_until_ready((g_q, w_dg_q, g_sf, w_dg_sf))

        dgrad_fn = jax.jit(lambda a, b, asf, bsf, o: mxfp8_grouped_dgrad(a, asf, b, bsf, o, mma_tiler_mnk=mma_tiler))
        out = bench_kernel("mxfp8_dgrad", dgrad_fn, (g_q, w_dg_q, g_sf, w_dg_sf, offs), (m, k))

        g_deq = dequantize_mxfp8(g_q, g_scales)
        w_deq_t = jnp.swapaxes(dequantize_mxfp8(w_dg_q, w_dg_scales), 1, 2)  # (E, N, K)
        ref = ragged_reference(g_deq, w_deq_t, group_sizes)
        check("mxfp8_dgrad", out, ref)
        del g_q, g_scales, w_dg_q, w_dg_scales, g_sf, w_dg_sf, out, g_deq, w_deq_t, ref

    # -------------------------------------------------------------- wgrad --
    # dw[E, K, N] = x^T grouped-outer g: 2Dx2D with contraction TOKENS; both
    # operands token-axis quantized in natural storage.
    if "wgrad" in products:
        x_qc, x_sc = jax.jit(quantize_mxfp8_tokens)(x)  # (m, k), (m//32, k)
        g_qc, g_sc = jax.jit(quantize_mxfp8_tokens)(g)  # (m, n), (m//32, n)
        x_sfw = jax.jit(build_sf_wgrad_fast)(x_sc.T, col_idx, perm_k)
        assert_bit_equal(x_sfw, build_sf_wgrad(x_sc.T, group_sizes), f"{label} sf_wgrad")
        g_sfw = jax.jit(build_sf_wgrad_fast)(g_sc.T, col_idx, perm_n)
        jax.block_until_ready((x_qc, g_qc, x_sfw, g_sfw))

        wgrad_fn = jax.jit(lambda a, b, asf, bsf, o: mxfp8_grouped_wgrad(a, asf, b, bsf, o, mma_tiler_mnk=mma_tiler))
        out = bench_kernel("mxfp8_wgrad", wgrad_fn, (x_qc, g_qc, x_sfw, g_sfw, offs), (E, k, n))

        x_deqc = dequantize_mxfp8_tokens(x_qc, x_sc)
        g_deqc = dequantize_mxfp8_tokens(g_qc, g_sc)
        ref = wgrad_reference(x_deqc, g_deqc, group_sizes, k, n)
        check("mxfp8_wgrad", out, ref)
        del x_qc, x_sc, g_qc, g_sc, x_sfw, g_sfw, out, x_deqc, g_deqc, ref

    return res


def run_producer_case(m: int, dim: int, group_sizes: list[int], iters: int, warmup: int) -> dict:
    """Producer costs for one activation/cotangent shape (m, dim), uniform routing.

    naive_row: quantize along features + per-group-loop swizzle (fwd/dgrad A).
    naive_col: token-axis quantize + per-group-loop 2Dx2D swizzle (wgrad).
    dual: single fused jit producing both orientations.
    quantize_only: XLA quantize without any swizzle (lower bound of the row leg).
    """
    t = jax.random.normal(jax.random.PRNGKey(1), (m, dim), dtype=jnp.bfloat16)
    row_idx = jnp.asarray(sfa_row_gather_indices(group_sizes))
    col_idx_np, perm_np = sf_wgrad_col_layout(group_sizes, dim)
    col_idx, perm = jnp.asarray(col_idx_np), jnp.asarray(perm_np)

    def naive_row(tt):
        q, s = quantize_mxfp8(tt)
        return q, build_sfa(s, group_sizes)

    def naive_col(tt):
        q, s = quantize_mxfp8_tokens(tt)
        return q, build_sf_wgrad(s.T, group_sizes)

    dual = jax.jit(lambda tt: dual_quantize_activation(tt, row_idx, col_idx, perm))
    naive_row_fn = jax.jit(naive_row)
    naive_col_fn = jax.jit(naive_col)
    q_row, sf_row, q_col, sf_col = dual(t)
    nr = naive_row_fn(t)
    nc = naive_col_fn(t)
    assert_bit_equal(q_row, nr[0], f"producer({dim}) q_row")
    assert_bit_equal(sf_row, nr[1], f"producer({dim}) sf_row")
    assert_bit_equal(q_col, nc[0], f"producer({dim}) q_col")
    assert_bit_equal(sf_col, nc[1], f"producer({dim}) sf_col")
    del q_row, sf_row, q_col, sf_col, nr, nc

    out = {
        "quantize_only_ms": timed(jax.jit(quantize_mxfp8), (t,), iters, warmup) * 1e3,
        "naive_row_ms": timed(naive_row_fn, (t,), iters, warmup) * 1e3,
        "naive_col_ms": timed(naive_col_fn, (t,), iters, warmup) * 1e3,
        "dual_ms": timed(dual, (t,), iters, warmup) * 1e3,
    }
    out["naive_total_ms"] = out["naive_row_ms"] + out["naive_col_ms"]
    print(
        f"  act({m}x{dim}): quantize-only {out['quantize_only_ms']:.3f}  "
        f"naive row {out['naive_row_ms']:.3f} + col {out['naive_col_ms']:.3f} "
        f"= {out['naive_total_ms']:.3f}  dual {out['dual_ms']:.3f} ms",
        flush=True,
    )
    return out


def run_producer_weight_case(k: int, n: int, iters: int, warmup: int) -> dict:
    """Producer costs for one weight shape (E, k, n): both orientations."""
    w = jax.random.normal(jax.random.PRNGKey(2), (E, k, n), dtype=jnp.bfloat16) / (k**0.5)

    def naive_fwd(ww):
        q, s = quantize_mxfp8(jnp.swapaxes(ww, 1, 2))
        return q, build_sfb(s)

    def naive_dgrad(ww):
        q, s = quantize_mxfp8(ww)
        return q, build_sfb(s)

    dual = jax.jit(dual_quantize_weight)
    naive_fwd_fn = jax.jit(naive_fwd)
    naive_dgrad_fn = jax.jit(naive_dgrad)
    outs = dual(w)
    nf = naive_fwd_fn(w)
    nd = naive_dgrad_fn(w)
    assert_bit_equal(outs[0], nf[0], f"weight({k}x{n}) q_fwd")
    assert_bit_equal(outs[1], nf[1], f"weight({k}x{n}) sfb_fwd")
    assert_bit_equal(outs[2], nd[0], f"weight({k}x{n}) q_dgrad")
    assert_bit_equal(outs[3], nd[1], f"weight({k}x{n}) sfb_dgrad")
    del outs, nf, nd

    out = {
        "naive_fwd_ms": timed(naive_fwd_fn, (w,), iters, warmup) * 1e3,
        "naive_dgrad_ms": timed(naive_dgrad_fn, (w,), iters, warmup) * 1e3,
        "dual_ms": timed(dual, (w,), iters, warmup) * 1e3,
    }
    out["naive_total_ms"] = out["naive_fwd_ms"] + out["naive_dgrad_ms"]
    print(
        f"  weight(E{E}x{k}x{n}): naive fwd {out['naive_fwd_ms']:.3f} + dgrad "
        f"{out['naive_dgrad_ms']:.3f} = {out['naive_total_ms']:.3f}  dual {out['dual_ms']:.3f} ms",
        flush=True,
    )
    return out


def quad_summary(results: dict) -> dict | None:
    """Layer-quad honest total (uniform routing): 6 MXFP8 GEMMs + producers vs 3x bf16 dense.

    The MoE layer's fwd+bwd runs each grouped product once per weight (w13, w2):
    6 GEMMs. Activation producers: x and g13 at (M, 2560)... x (M,2560) and g2
    (M,2560) dual-quantized, h and g13 at (M,1280) dual-quantized. Weight
    producers: both orientations of w13 and w2 (once per step). The bf16 side
    runs the same 6 GEMMs at dense-yardstick speed with no producer cost.
    """
    cases = results["cases"]
    prods = results.get("producers", {})
    try:
        gemm_ms = sum(
            cases[f"{lbl}_uniform"]["arms"][arm]["ms"]
            for lbl, _, _ in SHAPES
            for arm in ("mxfp8_fwd", "mxfp8_dgrad", "mxfp8_wgrad")
        )
        bf16_ms = 3 * sum(cases[f"{lbl}_uniform"]["arms"]["bf16_dense"]["ms"] for lbl, _, _ in SHAPES)
        act_dual = 2 * prods["act_2560"]["dual_ms"] + 2 * prods["act_1280"]["dual_ms"]
        act_naive = 2 * prods["act_2560"]["naive_total_ms"] + 2 * prods["act_1280"]["naive_total_ms"]
        w_dual = prods["w13"]["dual_ms"] + prods["w2"]["dual_ms"]
    except KeyError as missing:
        print(f"quad summary skipped (missing {missing})", flush=True)
        return None

    quad = {
        "gemm_ms": gemm_ms,
        "producer_act_dual_ms": act_dual,
        "producer_act_naive_ms": act_naive,
        "producer_weight_dual_ms": w_dual,
        "bf16_3x_dense_ms": bf16_ms,
        "speedup_gemm_only": bf16_ms / gemm_ms,
        "speedup_with_act_producers": bf16_ms / (gemm_ms + act_dual),
        "speedup_full": bf16_ms / (gemm_ms + act_dual + w_dual),
        "speedup_full_naive_producers": bf16_ms / (gemm_ms + act_naive + w_dual),
    }
    print("\n== layer-quad honest total (uniform routing) ==", flush=True)
    print(f"  mxfp8 GEMMs (fwd+dgrad+wgrad, w13+w2) {gemm_ms:8.3f} ms")
    print(f"  + activation producers (dual x4)      {act_dual:8.3f} ms   (naive: {act_naive:.3f})")
    print(f"  + weight producers (dual x2)          {w_dual:8.3f} ms")
    print(f"  bf16 yardstick (3x dense, w13+w2)     {bf16_ms:8.3f} ms")
    print(f"  speedup: gemm-only {quad['speedup_gemm_only']:.3f}x | +act {quad['speedup_with_act_producers']:.3f}x")
    print(
        f"  speedup: full {quad['speedup_full']:.3f}x | full w/ naive producers "
        f"{quad['speedup_full_naive_producers']:.3f}x",
        flush=True,
    )
    return quad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=262144)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--mma-tiler", default="128,256,128", help="mma tiler MNK; 128,256,128 is the measured best")
    p.add_argument(
        "--products",
        default="fwd,dgrad,wgrad,producer",
        help="comma-separated subset of fwd,dgrad,wgrad,producer",
    )
    p.add_argument("--out", default="bench_mxfp8_grouped.json")
    a = p.parse_args()
    mma_tiler = tuple(int(v) for v in a.mma_tiler.split(","))
    assert len(mma_tiler) == 3
    products = set(a.products.split(","))
    assert products <= {"fwd", "dgrad", "wgrad", "producer"}, products

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind} (cc {dev.compute_capability}), jax {jax.__version__}")
    print(f"cutlass: {cutlass.__version__}, products: {sorted(products)}")

    results = {
        "device": str(dev.device_kind),
        "jax": jax.__version__,
        "cutlass": cutlass.__version__,
        "tokens": a.tokens,
        "experts": E,
        "mma_tiler": list(mma_tiler),
        "products": sorted(products),
        "cases": {},
        "producers": {},
    }
    for label, k, n in SHAPES:
        for dist, groups in (("uniform", uniform_groups(a.tokens)), ("skewed", skewed_groups(a.tokens))):
            case = f"{label}_{dist}"
            results["cases"][case] = run_case(case, k, n, groups, a.iters, a.warmup, mma_tiler, products)

    if "producer" in products:
        print("\n== producers (uniform routing) ==", flush=True)
        groups = uniform_groups(a.tokens)
        for dim in (2560, 1280):
            results["producers"][f"act_{dim}"] = run_producer_case(a.tokens, dim, groups, a.iters, a.warmup)
        for wl, k, n in (("w13", 2560, 1280), ("w2", 1280, 2560)):
            results["producers"][wl] = run_producer_weight_case(k, n, a.iters, a.warmup)

    quad = quad_summary(results)
    if quad is not None:
        results["quad"] = quad

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
