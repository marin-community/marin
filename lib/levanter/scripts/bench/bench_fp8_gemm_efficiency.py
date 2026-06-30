# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated single-GEMM efficiency probe: fp8 vs bf16, achieved fraction of H100 peak.

The MoE metric (`bench_ragged_fp8_autotune.py`) measures the *whole* fwd+bwd step, which folds the
GEMM together with the fp8 fixed overhead (quantize, cast-transpose, scaling) and the bf16 activation
path. This probe strips all of that away to answer one question the e2e metric cannot:

    **At the realistic per-expert GEMM shapes, what fraction of its hardware peak does the fp8 Mosaic
    ragged kernel reach, vs the bf16 Triton kernel reaching its (half-as-high) peak?**

If both are compute-bound, the ideal fp8/bf16 GEMM speedup is 2.0x. The MoE profile attributes only a
~1.38x GEMM-only ratio. This probe decides whether that 1.38x is a hardware/roofline limit or a
*kernel-efficiency* gap (fp8 leaving headroom the bf16 kernel does not).

Four variants per shape, single group (G=1, dense), pre-quantized operands, forward GEMM only:
  - ``fp8_mosaic``  — the production fp8 forward kernel: ``_mosaic_ragged_dot`` on f8 ``lhs[M,K]`` +
                      K-contiguous f8 ``rhs[1,N,K]`` (the transpose/quant happen *outside* the timed jit).
  - ``bf16_triton`` — the production bf16 baseline: ``ragged_dot(implementation="auto")`` on bf16.
  - ``fp8_dense``   — ``jax.lax.dot_general`` f8xf8 (cuBLASLt): the mature-kernel fp8 upper bound.
  - ``bf16_dense``  — ``jax.lax.dot_general`` bf16: the mature-kernel bf16 upper bound.

The gap between the ``fp8_mosaic/bf16_triton`` ratio and the ``fp8_dense/bf16_dense`` ratio is the
Mosaic-kernel headroom. Reports achieved TFLOP/s and % of H100 SXM peak for each.
"""

import argparse
import json
import os
import sys

# The bench module runs the CUDA-toolchain bootstrap at import (must precede `import jax`), and exposes
# the kernel entry points + steady-state timer + peak constants we reuse here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_ragged_fp8_autotune as B  # noqa: E402  (triggers _ensure_cuda_toolchain before jax import)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from haliax.nn.ragged_dot import MosaicBlockConfig, _mosaic_ragged_dot, ragged_dot  # noqa: E402

_BF16_PEAK = B._H100_SXM_BF16_TFLOPS_PER_S
_FP8_PEAK = B._H100_SXM_FP8_TFLOPS_PER_S

# (M, K, N) single-GEMM shapes. The two real per-expert forward GEMMs at the realistic operating point
# (d2560_e32_t1k: 1024 tok/expert, D=2560, F=1280) are dot1 (K=D, N=2F) and dot2 (K=F, N=D). Plus an
# M-scaling sweep (does fp8 efficiency recover with bigger per-expert batch?) and a big-square
# calibration point (both kernels' best-case efficiency).
_SHAPES = {
    "dot1_t1k": (1024, 2560, 2560),  # forward dot1 @ 1024 tok/expert
    "dot2_t1k": (1024, 1280, 2560),  # forward dot2 @ 1024 tok/expert
    "dot1_t512": (512, 2560, 2560),
    "dot1_t2k": (2048, 2560, 2560),
    "dot1_t4k": (4096, 2560, 2560),
    "dot1_t8k": (8192, 2560, 2560),
    "square_8k": (8192, 8192, 8192),  # calibration: large dense, both kernels near best
}


def _median_time(fn, args, *, samples, inner_steps, warmup):
    _, times = B.time_steady_state(fn, args, samples=samples, inner_steps=inner_steps, warmup=warmup)
    return float(np.median(times))


def _variants(M, K, N, cfg):
    """Build the four (label, fn, args, peak) timing variants for one (M,K,N) single GEMM."""
    rng = np.random.default_rng(0)
    gs = jnp.asarray([M], jnp.int32)

    lhs_bf16 = jnp.asarray(rng.standard_normal((M, K)), jnp.bfloat16)
    rhs_bf16 = jnp.asarray(rng.standard_normal((1, K, N)) * 0.08, jnp.bfloat16)  # [G,K,N] natural
    # f8 operands, pre-quantized OUTSIDE the timed region. Mosaic forward wants K-contiguous rhs [G,N,K].
    lhs_f8 = lhs_bf16.astype(jnp.float8_e4m3fn)
    rhs_f8_kc = jnp.swapaxes(rhs_bf16, 1, 2).astype(jnp.float8_e4m3fn)  # [G,N,K]
    lhs_f8_2d = lhs_f8
    rhs_f8_2d = rhs_bf16[0].astype(jnp.float8_e4m3fn)  # [K,N] for dense dot_general
    rhs_bf16_2d = rhs_bf16[0]

    dn = (((1,), (0,)), ((), ()))  # standard [M,K]x[K,N] contraction

    return [
        (
            "fp8_mosaic",
            lambda a, b, g: _mosaic_ragged_dot(a, b, g, cfg),
            (lhs_f8, rhs_f8_kc, gs),
            _FP8_PEAK,
        ),
        (
            "bf16_triton",
            lambda a, b, g: ragged_dot(a, b, g, implementation="auto"),
            (lhs_bf16, rhs_bf16, gs),
            _BF16_PEAK,
        ),
        (
            "fp8_dense",
            lambda a, b: jax.lax.dot_general(a, b, dn, preferred_element_type=jnp.bfloat16),
            (lhs_f8_2d, rhs_f8_2d),
            _FP8_PEAK,
        ),
        (
            "bf16_dense",
            lambda a, b: jax.lax.dot_general(a, b, dn, preferred_element_type=jnp.bfloat16),
            (lhs_bf16, rhs_bf16_2d),
            _BF16_PEAK,
        ),
    ]


def _grouped_variants(tok, E, K, N, cfg):
    """Grouped (ragged) forward GEMM over E experts at ``tok`` tokens/expert — the faithful MoE setting.

    Launch overhead amortizes over E groups (unlike the isolated single GEMM), so the achieved TFLOP/s
    reflect the real kernel. Same four variants; the dense references are single GEMMs at the equivalent
    total M=E*tok (the mature-kernel ceiling at matched total FLOP, no per-expert structure)."""
    rng = np.random.default_rng(0)
    T = E * tok
    gs = jnp.full((E,), tok, jnp.int32)

    lhs_bf16 = jnp.asarray(rng.standard_normal((T, K)), jnp.bfloat16)
    rhs_bf16 = jnp.asarray(rng.standard_normal((E, K, N)) * 0.08, jnp.bfloat16)  # [E,K,N] natural
    lhs_f8 = lhs_bf16.astype(jnp.float8_e4m3fn)
    rhs_f8_kc = jnp.swapaxes(rhs_bf16, 1, 2).astype(jnp.float8_e4m3fn)  # [E,N,K] K-contiguous

    # Single-fat-GEMM ceiling at matched total FLOP (M=T, ONE shared weight): mature cuBLAS upper bound,
    # but optimistic (one big GEMM + 1 weight reused over all rows).
    rhs_bf16_2d = jnp.asarray(rng.standard_normal((K, N)) * 0.08, jnp.bfloat16)
    rhs_f8_2d = rhs_bf16_2d.astype(jnp.float8_e4m3fn)
    dn = (((1,), (0,)), ((), ()))

    # Batched-dense ceiling: E separate [tok,K]x[K,N] with DISTINCT per-expert weights, expressed as a
    # batched dot_general (batch dim = expert). Equal groups -> single cublasGemmStridedBatchedEx launch.
    # Apples-to-apples with the grouped kernel: same per-expert M, same distinct-weight memory pattern.
    lhs_bf16_b = lhs_bf16.reshape(E, tok, K)
    lhs_f8_b = lhs_f8.reshape(E, tok, K)
    rhs_f8_b = rhs_bf16.astype(jnp.float8_e4m3fn)  # [E,K,N] natural (distinct weights)
    dn_b = (((2,), (1,)), ((0,), (0,)))  # contract K, batch over E -> [E,tok,N]

    return [
        ("fp8_mosaic", lambda a, b, g: _mosaic_ragged_dot(a, b, g, cfg), (lhs_f8, rhs_f8_kc, gs), _FP8_PEAK),
        ("bf16_triton", lambda a, b, g: ragged_dot(a, b, g, implementation="auto"), (lhs_bf16, rhs_bf16, gs), _BF16_PEAK),
        ("fp8_dense", lambda a, b: jax.lax.dot_general(a, b, dn, preferred_element_type=jnp.bfloat16), (lhs_f8, rhs_f8_2d), _FP8_PEAK),
        ("bf16_dense", lambda a, b: jax.lax.dot_general(a, b, dn, preferred_element_type=jnp.bfloat16), (lhs_bf16, rhs_bf16_2d), _BF16_PEAK),
        ("fp8_batched", lambda a, b: jax.lax.dot_general(a, b, dn_b, preferred_element_type=jnp.bfloat16), (lhs_f8_b, rhs_f8_b), _FP8_PEAK),
        ("bf16_batched", lambda a, b: jax.lax.dot_general(a, b, dn_b, preferred_element_type=jnp.bfloat16), (lhs_bf16_b, rhs_bf16), _BF16_PEAK),
    ]


def _grouped_main(args):
    cfg = MosaicBlockConfig()
    backend = jax.default_backend()
    dev = jax.devices()[0].device_kind if jax.devices() else "none"
    E, K, N = args.experts, args.contract, args.out_dim
    toks = [int(t) for t in args.tokens_per_expert.split(",")]
    print(f"backend={backend} device={dev} GROUPED E={E} K={K} N={N} mosaic_cfg={cfg}")

    # Confirm the batched-dense reference lowers to a single strided-batched cuBLAS launch (not E dots /
    # a Triton fusion) — otherwise the "apples-to-apples ceiling" would not mean what we claim.
    try:
        probe = next(v for v in _grouped_variants(1024, E, K, N, cfg) if v[0] == "fp8_batched")
        _, fn, fn_args, _ = probe
        hlo = jax.jit(fn).lower(*fn_args).compile().as_text()
        n_cublas = hlo.count('custom_call_target="__cublas')
        n_triton = hlo.lower().count("triton")
        targets = sorted({line.split('custom_call_target="')[1].split('"')[0] for line in hlo.splitlines() if "custom_call_target=" in line})
        print(f"HLO-CHECK fp8_batched: cublas_custom_calls={n_cublas} triton_mentions={n_triton} targets={targets}")
    except Exception as e:  # noqa: BLE001
        print(f"HLO-CHECK failed: {e!r}")

    results = []
    for tok in toks:
        flop = 2.0 * (E * tok) * K * N
        row = {"tokens_per_expert": tok, "experts": E, "K": K, "N": N, "total_M": E * tok, "flop": flop, "variants": {}}
        for label, fn, fn_args, peak in _grouped_variants(tok, E, K, N, cfg):
            try:
                t = _median_time(fn, fn_args, samples=args.samples, inner_steps=args.inner_steps, warmup=args.warmup)
                row["variants"][label] = {"median_s": t, "tflops_per_s": flop / t, "pct_of_peak": 100.0 * (flop / t) / peak}
                print(f"  tok/exp={tok:6d} {label:12s} {t*1e6:9.1f} us  {flop/t/1e12:7.1f} TFLOP/s  {100.0*(flop/t)/peak:5.1f}% peak")
            except Exception as e:  # noqa: BLE001
                row["variants"][label] = {"error": repr(e)}
                print(f"  tok/exp={tok:6d} {label:12s} ERROR {e!r}")
        v = row["variants"]
        if all("median_s" in v.get(k, {}) for k in ("fp8_mosaic", "bf16_triton")):
            row["mosaic_ratio"] = v["bf16_triton"]["median_s"] / v["fp8_mosaic"]["median_s"]
        print(f"  tok/exp={tok:6d} -> mosaic fp8/bf16 ratio={row.get('mosaic_ratio')}")
        results.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"backend": backend, "device": dev, "mode": "grouped", "experts": E, "K": K, "N": N, "results": results}, f, indent=2)
    print(f"result_json {json.dumps({'device': dev, 'mode': 'grouped', 'results': results})}")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "grouped"], default="single")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--contract", type=int, default=2560, help="K (grouped mode)")
    ap.add_argument("--out-dim", type=int, default=2560, help="N (grouped mode)")
    ap.add_argument("--tokens-per-expert", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--inner-steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="/app/scratch/fp8_gemm_efficiency.json")
    args = ap.parse_args()
    if args.mode == "grouped":
        return _grouped_main(args)

    cfg = MosaicBlockConfig()  # the held d2560 winner (128/128/128, steps=6, grid_block_n=4)
    backend = jax.default_backend()
    dev = jax.devices()[0].device_kind if jax.devices() else "none"
    print(f"backend={backend} device={dev} mosaic_cfg={cfg}")

    results = []
    for name, (M, K, N) in _SHAPES.items():
        flop = 2.0 * M * K * N
        row = {"shape": name, "M": M, "K": K, "N": N, "flop": flop, "variants": {}}
        for label, fn, fn_args, peak in _variants(M, K, N, cfg):
            try:
                t = _median_time(fn, fn_args, samples=args.samples, inner_steps=args.inner_steps, warmup=args.warmup)
                tflops = flop / t
                row["variants"][label] = {
                    "median_s": t,
                    "tflops_per_s": tflops,
                    "pct_of_peak": 100.0 * tflops / peak,
                }
                print(f"  {name:10s} {label:12s} {t*1e6:8.1f} us  {tflops/1e12:7.1f} TFLOP/s  {100.0*tflops/peak:5.1f}% peak")
            except Exception as e:  # noqa: BLE001  (one bad shape/variant should not sink the sweep)
                row["variants"][label] = {"error": repr(e)}
                print(f"  {name:10s} {label:12s} ERROR {e!r}")
        v = row["variants"]
        if "fp8_mosaic" in v and "bf16_triton" in v and "median_s" in v["fp8_mosaic"] and "median_s" in v["bf16_triton"]:
            row["mosaic_ratio"] = v["bf16_triton"]["median_s"] / v["fp8_mosaic"]["median_s"]
        if "fp8_dense" in v and "bf16_dense" in v and "median_s" in v["fp8_dense"] and "median_s" in v["bf16_dense"]:
            row["dense_ratio"] = v["bf16_dense"]["median_s"] / v["fp8_dense"]["median_s"]
        print(f"  {name:10s} -> mosaic fp8/bf16 ratio={row.get('mosaic_ratio')}  dense fp8/bf16 ratio={row.get('dense_ratio')}")
        results.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"backend": backend, "device": dev, "results": results}, f, indent=2)
    print(f"result_json {json.dumps({'device': dev, 'results': results})}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
