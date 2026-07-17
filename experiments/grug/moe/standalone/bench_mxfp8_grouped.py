"""MXFP8-002: Blackwell CuTe DSL MXFP8 scaled grouped GEMM vs bf16 baselines.

Benchmarks the vendored cutlass v4.5.2 MoE scaled-grouped-GEMM kernel (tcgen05
block-scaled MMA, e4m3 data + e8m0 block-32 scales) through the
``cutlass.jax.cutlass_call`` adapter in ``mxfp8_grouped/adapter.py``, at grug
row-13 per-device shapes:

  w13  x[M, 2560] @ w[E=64, 2560, 1280] -> [M, 1280]
  w2   x[M, 1280] @ w[E=64, 1280, 2560] -> [M, 2560]

with M = 262144 total tokens, first uniformly routed (4096/expert), then a
skewed Dirichlet(0.5) routing (group sizes multiples of 32, some experts ~0).

Arms:
  mxfp8_kernel   the CuTe DSL kernel (operands pre-quantized outside timing)
  bf16_ragged    jax.lax.ragged_dot in bf16 (XLA baseline, known slow on sm100)
  bf16_dense     dense bf16 einsum of equal total FLOPs (throughput yardstick)

Correctness: rel-Frobenius error of the kernel output vs a reference computed
by DEQUANTIZING THE SAME quantized operands (isolates the GEMM from
quantization; gate < 1e-3 against the bf16-rounded reference), plus error vs
the unquantized bf16 reference (~4e-2 expected, quantization noise).

Single GPU, GB200 (sm100) only.

Usage: python bench_mxfp8_grouped.py [--tokens 262144] [--iters 50] [--out out.json]
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from mxfp8_grouped.adapter import (
    build_sfa,
    build_sfb,
    dequantize_mxfp8,
    mxfp8_grouped_mm,
    quantize_mxfp8,
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


def run_case(label: str, k: int, n: int, group_sizes: list[int], iters: int, warmup: int) -> dict:
    m = sum(group_sizes)
    gs_dev = jnp.array(group_sizes, dtype=jnp.int32)
    offs = jnp.cumsum(gs_dev).astype(jnp.int32)

    key = jax.random.PRNGKey(0)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (E, k, n), dtype=jnp.bfloat16) / (k**0.5)

    # ── Quantize (outside timing; MXFP8-001b showed XLA quantize dominates) ──
    x_q, x_scales = jax.jit(quantize_mxfp8)(x)
    w_t = jnp.swapaxes(w, 1, 2)  # (E, N, K), K contiguous after materialization
    w_q, w_scales = jax.jit(quantize_mxfp8)(w_t)
    x_sf = build_sfa(x_scales, group_sizes)
    w_sf = build_sfb(w_scales)
    jax.block_until_ready((x_q, w_q, x_sf, w_sf))

    # ── Kernel arm ──
    kernel_fn = jax.jit(lambda xq, wq, xsf, wsf, off: mxfp8_grouped_mm(xq, xsf, wq, wsf, off))
    out = jax.block_until_ready(kernel_fn(x_q, w_q, x_sf, w_sf, offs))

    # ── References ──
    # GEMM-isolating reference: dequantize the SAME quantized operands.
    x_deq = dequantize_mxfp8(x_q, x_scales)  # (M, K) f32
    w_deq = jnp.swapaxes(dequantize_mxfp8(w_q, w_scales), 1, 2)  # (E, K, N) f32
    ref_q32 = jax.lax.ragged_dot(x_deq, w_deq, gs_dev, preferred_element_type=jnp.float32)
    ref_q_bf = ref_q32.astype(jnp.bfloat16)
    err_gemm_bf = rel_frob(out, ref_q_bf)  # gate: < 1e-3 (bf16-rounded ref)
    err_gemm_f32 = rel_frob(out, ref_q32)
    # Quantization-inclusive reference: unquantized operands.
    ref_full = jax.lax.ragged_dot(
        x.astype(jnp.float32), w.astype(jnp.float32), gs_dev, preferred_element_type=jnp.float32
    )
    err_quant = rel_frob(out, ref_full)
    del x_deq, w_deq, ref_q32, ref_q_bf, ref_full

    flops = 2 * m * k * n
    res = {
        "m": m,
        "k": k,
        "n": n,
        "err_gemm_vs_deq_bf16ref": err_gemm_bf,
        "err_gemm_vs_deq_f32ref": err_gemm_f32,
        "err_vs_unquantized": err_quant,
        "arms": {},
    }

    t = timed(kernel_fn, (x_q, w_q, x_sf, w_sf, offs), iters, warmup)
    res["arms"]["mxfp8_kernel"] = {"ms": t * 1e3, "tfs": flops / t / 1e12}

    ragged_fn = jax.jit(
        lambda xx, ww, gg: jax.lax.ragged_dot(xx, ww, gg, preferred_element_type=jnp.bfloat16)
    )
    t = timed(ragged_fn, (x, w, gs_dev), iters, warmup)
    res["arms"]["bf16_ragged"] = {"ms": t * 1e3, "tfs": flops / t / 1e12}

    w_dense = w[0]  # (K, N); dense equal-FLOPs yardstick: 2*M*K*N
    dense_fn = jax.jit(lambda xx, ww: jnp.einsum("mk,kn->mn", xx, ww, preferred_element_type=jnp.bfloat16))
    t = timed(dense_fn, (x, w_dense), iters, warmup)
    res["arms"]["bf16_dense"] = {"ms": t * 1e3, "tfs": flops / t / 1e12}

    base = res["arms"]["bf16_ragged"]["ms"]
    print(f"\n== {label} M={m} K={k} N={n} ==")
    print(
        f"  err(GEMM, deq ref bf16) {err_gemm_bf:.2e}  err(GEMM, deq ref f32) {err_gemm_f32:.2e}"
        f"  err(vs unquantized) {err_quant:.2e}"
    )
    for arm, r in res["arms"].items():
        print(f"  {arm:13s} {r['ms']:8.3f} ms  {r['tfs']:7.1f} TF/s  {base / r['ms']:.3f}x vs bf16_ragged")

    assert err_gemm_bf < 1e-3, f"{label}: GEMM rel-Frobenius error {err_gemm_bf:.3e} >= 1e-3"
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=262144)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", default="bench_mxfp8_grouped.json")
    a = p.parse_args()

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind} (cc {dev.compute_capability}), jax {jax.__version__}")
    import cutlass

    print(f"cutlass: {cutlass.__version__}")

    results = {
        "device": str(dev.device_kind),
        "jax": jax.__version__,
        "cutlass": cutlass.__version__,
        "tokens": a.tokens,
        "experts": E,
        "cases": {},
    }
    for label, k, n in SHAPES:
        for dist, groups in (("uniform", uniform_groups(a.tokens)), ("skewed", skewed_groups(a.tokens))):
            case = f"{label}_{dist}"
            results["cases"][case] = run_case(case, k, n, groups, a.iters, a.warmup)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
