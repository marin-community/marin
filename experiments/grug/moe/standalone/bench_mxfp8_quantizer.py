# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-002c: fused dual-orientation MXFP8 quantizer -- bit-exactness + bench.

Gates the CuTe DSL quantizer kernel (``mxfp8_grouped/quantize_cute.py``)
against the XLA reference producer (``adapter.dual_quantize_activation``):

1. Bit-exactness of all four outputs (q_row, sf_row, q_col, sf_col) on random
   inputs plus adversarial content (all-zero blocks, bf16-denormal-scale
   blocks, huge blocks, negative-heavy blocks, exact power-of-two amaxes),
   at a small shape and at the full M=262144 shapes, uniform + skewed routing.
2. Producer timing at (M, 2560) and (M, 1280): kernel-only, kernel+swizzle
   (the honest producer), and the XLA reference; effective read TB/s.
3. Layer-quad summary using the MXFP8-002b constants: GEMMs 4.86 ms, bf16
   3x-dense yardstick 6.95 ms, XLA weight producers 1.73 ms.

Single GPU, GB200 (sm100). All results print to stdout (pod /tmp dies with
the job).

Usage: python bench_mxfp8_quantizer.py [--tokens 262144] [--iters 50]
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
    dual_quantize_activation,
    sf_wgrad_col_layout,
    sfa_row_gather_indices,
)
from mxfp8_grouped.quantize_cute import dual_quantize_activation_cute, dual_quantize_mxfp8_cute

E = 64
DIMS = (2560, 1280)
# MXFP8-002b constants (job mxfp8-002b-g1, M=262144, E=64, tile (128,256,128)):
# fwd+dgrad+wgrad GEMMs w13+w2, the 3x bf16 dense yardstick, and the XLA
# weight producers (amortizable across microbatches; stay on XLA for now).
GEMM_MS = 4.86
BF16_3X_DENSE_MS = 6.95
WEIGHT_PRODUCERS_MS = 1.73
BREAK_EVEN_PRODUCER_MS = BF16_3X_DENSE_MS - GEMM_MS  # 2.09


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


def check_bit_equal(a, b, what: str, x=None) -> int:
    """Byte-compare; on mismatch print sample diagnostics and return the count."""
    ua, ub = as_u8(a), as_u8(b)
    assert ua.shape == ub.shape, f"{what}: shape {ua.shape} vs {ub.shape}"
    n_diff = int(jnp.sum(ua != ub))
    if n_diff:
        print(f"  MISMATCH {what}: {n_diff}/{ua.size} bytes differ", flush=True)
        idx = np.argwhere(np.asarray(ua != ub))[:8]
        for loc in idx:
            loc_t = tuple(int(v) for v in loc)
            extra = ""
            if x is not None and len(loc_t) == 2 and x.shape == ua.shape:
                xv = x[loc_t]
                extra = f" x={float(xv):.6g} (0x{int(np.asarray(xv).view(np.uint16)):04x})"
            print(f"    at {loc_t}: got 0x{int(ua[loc_t]):02x} ref 0x{int(ub[loc_t]):02x}{extra}", flush=True)
    return n_diff


def uniform_groups(m: int) -> list[int]:
    return [m // E] * E


def skewed_groups(m: int, seed: int = 0) -> list[int]:
    """Dirichlet(0.5) routing in units of 32 tokens (mirrors bench_mxfp8_grouped)."""
    rng = np.random.default_rng(seed)
    n_slots = m // 32
    proportions = rng.dirichlet([0.5] * E)
    raw = np.floor(proportions * n_slots).astype(int)
    deficit = n_slots - raw.sum()
    order = np.argsort(-proportions)
    i = 0
    while deficit > 0:
        raw[order[i % E]] += 1
        deficit -= 1
        i += 1
    return [int(s) * 32 for s in raw]


def adversarial_input(m: int, k: int, seed: int = 3) -> jnp.ndarray:
    """bf16 tensor stressing the e8m0 edge cases, blockwise (32-aligned regions).

    Rows [0:32) and cols [0:32) all-zero (zero amax in BOTH orientations),
    rows [32:64) scaled by 2^-130 (bf16 denormals; amax/448 subnormal in f32),
    cols [32:64) scaled by 2^120 (huge, near-overflow scale path),
    rows [64:96) negative-heavy, rows [96:128) exact powers of two (mant == 0,
    no-round-up branch of the e8m0 cast).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, k)).astype(np.float32)
    x[96:128] = np.sign(x[96:128]) * np.exp2(np.round(np.log2(np.abs(x[96:128]) + 1e-9)))
    x[32:64] *= 2.0**-130
    x[64:96] = -np.abs(x[64:96])
    x[:, 32:64] *= 2.0**120
    x[:32] = 0.0
    x[:, :32] = 0.0
    return jnp.asarray(x, dtype=jnp.bfloat16)


def check_case(x, group_sizes: list[int], what: str) -> int:
    m, dim = x.shape
    row_idx = jnp.asarray(sfa_row_gather_indices(group_sizes))
    col_idx_np, perm_np = sf_wgrad_col_layout(group_sizes, dim)
    col_idx, perm = jnp.asarray(col_idx_np), jnp.asarray(perm_np)
    ref = jax.jit(lambda t: dual_quantize_activation(t, row_idx, col_idx, perm))(x)
    got = jax.jit(lambda t: dual_quantize_activation_cute(t, row_idx, col_idx, perm))(x)
    total = 0
    for name, a, b in zip(("q_row", "sf_row", "q_col", "sf_col"), got, ref, strict=True):
        total += check_bit_equal(a, b, f"{what} {name}", x=x if name.startswith("q") else None)
    print(f"  {'BIT-EXACT' if total == 0 else 'FAILED'}: {what} ({m}x{dim})", flush=True)
    return total


def run_correctness(tokens: int):
    print("== correctness (bit-exact vs adapter.dual_quantize_activation) ==", flush=True)
    key = jax.random.PRNGKey(0)
    m_small = 8192
    failures = 0
    for dim in DIMS:
        x = jax.random.normal(key, (m_small, dim), dtype=jnp.bfloat16)
        failures += check_case(x, uniform_groups(m_small), f"small normal d{dim}")
        failures += check_case(adversarial_input(m_small, dim), uniform_groups(m_small), f"small adversarial d{dim}")
    for dim in DIMS:
        x = jax.random.normal(jax.random.fold_in(key, dim), (tokens, dim), dtype=jnp.bfloat16)
        failures += check_case(x, uniform_groups(tokens), f"full normal uniform d{dim}")
        failures += check_case(x, skewed_groups(tokens), f"full normal skewed d{dim}")
    return failures


def run_bench(tokens: int, iters: int, warmup: int) -> dict:
    print("\n== producer timing (uniform routing) ==", flush=True)
    groups = uniform_groups(tokens)
    row_idx = jnp.asarray(sfa_row_gather_indices(groups))
    out = {}
    for dim in DIMS:
        x = jax.random.normal(jax.random.PRNGKey(1), (tokens, dim), dtype=jnp.bfloat16)
        col_idx_np, perm_np = sf_wgrad_col_layout(groups, dim)
        col_idx, perm = jnp.asarray(col_idx_np), jnp.asarray(perm_np)

        kernel_fn = jax.jit(dual_quantize_mxfp8_cute)
        full_fn = jax.jit(lambda t, c=col_idx, p=perm: dual_quantize_activation_cute(t, row_idx, c, p))
        ref_fn = jax.jit(lambda t, c=col_idx, p=perm: dual_quantize_activation(t, row_idx, c, p))

        res = {
            "kernel_ms": timed(kernel_fn, (x,), iters, warmup) * 1e3,
            "full_ms": timed(full_fn, (x,), iters, warmup) * 1e3,
            "xla_ref_ms": timed(ref_fn, (x,), iters, warmup) * 1e3,
        }
        read_gb = tokens * dim * 2 / 1e9
        # one bf16 read + two e4m3 writes + two 1/32 scale writes per element
        total_gb = tokens * dim * (2 + 2 + 2 / 32) / 1e9
        res["read_tbs_kernel"] = read_gb / res["kernel_ms"]
        res["total_tbs_kernel"] = total_gb / res["kernel_ms"]
        res["read_tbs_full"] = read_gb / res["full_ms"]
        out[f"act_{dim}"] = res
        print(
            f"  act({tokens}x{dim}): kernel {res['kernel_ms']:.3f} ms "
            f"({res['read_tbs_kernel']:.2f} read TB/s, {res['total_tbs_kernel']:.2f} total TB/s) | "
            f"kernel+swizzle {res['full_ms']:.3f} ms ({res['read_tbs_full']:.2f} read TB/s) | "
            f"XLA ref {res['xla_ref_ms']:.3f} ms",
            flush=True,
        )
    return out


def quad_summary(bench: dict) -> dict:
    act_total = 2 * bench["act_2560"]["full_ms"] + 2 * bench["act_1280"]["full_ms"]
    act_total_kernel_only = 2 * bench["act_2560"]["kernel_ms"] + 2 * bench["act_1280"]["kernel_ms"]
    xla_total = 2 * bench["act_2560"]["xla_ref_ms"] + 2 * bench["act_1280"]["xla_ref_ms"]
    quad = {
        "act_producer_total_ms": act_total,
        "act_producer_kernel_only_ms": act_total_kernel_only,
        "act_producer_xla_ms": xla_total,
        "speedup_acts_only": BF16_3X_DENSE_MS / (GEMM_MS + act_total),
        "speedup_full": BF16_3X_DENSE_MS / (GEMM_MS + act_total + WEIGHT_PRODUCERS_MS),
    }
    print("\n== layer-quad with CuTe activation producers (MXFP8-002b GEMM constants) ==", flush=True)
    print(
        f"  activation producers (4 tensors)  {act_total:7.3f} ms  (kernel-only {act_total_kernel_only:.3f};"
        f" XLA was {xla_total:.3f}; break-even budget {BREAK_EVEN_PRODUCER_MS:.2f})"
    )
    print(f"  GEMMs {GEMM_MS:.2f} + acts -> speedup vs bf16 {quad['speedup_acts_only']:.3f}x (weights amortized)")
    print(
        f"  GEMMs {GEMM_MS:.2f} + acts + XLA weights {WEIGHT_PRODUCERS_MS:.2f} -> "
        f"speedup {quad['speedup_full']:.3f}x",
        flush=True,
    )
    return quad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=262144)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--skip-correctness", action="store_true")
    a = p.parse_args()

    dev = jax.devices()[0]
    print(
        f"device: {dev.device_kind} (cc {dev.compute_capability}), jax {jax.__version__}, "
        f"cutlass {cutlass.__version__}",
        flush=True,
    )

    failures = 0
    if not a.skip_correctness:
        failures = run_correctness(a.tokens)
    bench = run_bench(a.tokens, a.iters, a.warmup)
    quad = quad_summary(bench)
    print("\nRESULTS_JSON " + json.dumps({"bench": bench, "quad": quad}), flush=True)
    assert failures == 0, f"correctness: {failures} mismatched bytes across cases"


if __name__ == "__main__":
    main()
