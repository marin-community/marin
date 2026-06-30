# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Does tokamax's Mosaic-GPU ragged-dot beat JAX's built-in mgpu (and triton) for bf16?

Companion to ``bench_bf16_ragged_dot.py``. That probe established that the *autotuned* JAX built-in
``ragged_dot_mgpu`` beats main's vendored Triton kernel by ~1.10x on the e2e bf16 MoE GEMM (the win
concentrated in the dlhs layout). The remaining question the user posed: can the **tokamax** Mosaic-GPU
kernel — a more developed Hopper kernel (warp-specialized + async-store SM90 variants, built-in
heuristics + autotuner) that is already a Levanter dependency — squeeze out more, especially on the
forward/drhs layouts where JAX-mgpu only reached *parity* with Triton?

Per layout (fwd / dlhs / drhs), at the realistic d2560 operating point, time:
  - ``triton``  — main's vendored Pallas-Triton kernel (baseline).
  - ``mgpu``    — JAX ``ragged_dot_mgpu`` at the winning config from ``bench_bf16_ragged_dot.py``.
  - ``tokamax`` — ``tokamax.ragged_dot_general(..., implementation="mosaic_gpu")`` with tokamax's own
                  default (heuristic) config selection. ``implementation`` is forced (not a fallback
                  list) so an unsupported layout raises instead of silently falling back to triton/xla.

Numerics-gated against the triton reference. Stock JAX (no FP8 fork)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_ragged_fp8_autotune as B  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu as _mgpu  # noqa: E402

from haliax.nn.ragged_dot import (  # noqa: E402
    _DEFAULT_DIM_NUMS,
    _DLHS_DIM_NUMS,
    _DRHS_DIM_NUMS,
    _triton_pallas_call,
)

# Import the ragged_dot api directly from _src to avoid tokamax's top-level __init__, which pulls
# autotuning/benchmarking (tensorflow/xprof). The mosaic_gpu kernel path needs only qwix/einshape.
try:
    from tokamax._src.ops.ragged_dot import api as _tok_api  # noqa: E402

    _HAS_TOKAMAX = True
    _TOK_VER = "0.0.6 (via _src.ops.ragged_dot.api)"
except Exception as e:  # noqa: BLE001
    _HAS_TOKAMAX = False
    _TOK_VER = repr(e)

_BF16_PEAK = B._H100_SXM_BF16_TFLOPS_PER_S
_GEMMS = {"dot1": (2560, 2560), "dot2": (1280, 2560)}

# Winning JAX-mgpu configs from bench_bf16_ragged_dot.py (uniform across dot1/dot2).
_MGPU_FWD = dict(block_m=128, block_n=128, block_k=64, max_concurrent_steps=6, grid_block_n=4)


def _median_time(fn, args, *, samples, inner_steps, warmup):
    _, times = B.time_steady_state(fn, args, samples=samples, inner_steps=inner_steps, warmup=warmup)
    return float(np.median(times))


def _rel_frob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _tok_rd(dims):
    """tokamax mosaic_gpu ragged_dot_general bound to a dimension-number layout."""
    def f(lhs, rhs, gs):
        return _tok_api.ragged_dot_general(
            lhs, rhs, group_sizes=gs, ragged_dot_dimension_numbers=dims,
            preferred_element_type=jnp.bfloat16, implementation="mosaic_gpu",
        )
    return f


def _layouts(K, N, tok, E):
    rng = np.random.default_rng(0)
    T = E * tok
    gs = jnp.full((E,), tok, jnp.int32)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.05, jnp.bfloat16)
    dout = jnp.asarray(rng.standard_normal((T, N)) * 0.1, jnp.bfloat16)
    flop = 2.0 * T * K * N

    ref = {
        "fwd": jax.block_until_ready(jax.jit(lambda a, b, g: _triton_pallas_call(a, b, g, _DEFAULT_DIM_NUMS))(lhs, rhs, gs)),
        "dlhs": jax.block_until_ready(jax.jit(lambda a, b, g: _triton_pallas_call(a, b, g, _DLHS_DIM_NUMS))(dout, rhs, gs)),
        "drhs": jax.block_until_ready(jax.jit(lambda a, b, g: _triton_pallas_call(a, b, g, _DRHS_DIM_NUMS))(lhs, dout, gs)),
    }
    impls = {
        "fwd": {
            "triton": (lambda a, b, g: _triton_pallas_call(a, b, g, _DEFAULT_DIM_NUMS), (lhs, rhs, gs)),
            "mgpu": (lambda a, b, g: _mgpu.ragged_dot(a, b, group_sizes=g, transpose_rhs=False, **_MGPU_FWD), (lhs, rhs, gs)),
            "tokamax": (_tok_rd(_DEFAULT_DIM_NUMS), (lhs, rhs, gs)),
        },
        "dlhs": {
            "triton": (lambda a, b, g: _triton_pallas_call(a, b, g, _DLHS_DIM_NUMS), (dout, rhs, gs)),
            "mgpu": (lambda a, b, g: _mgpu.ragged_dot(a, b, group_sizes=g, transpose_rhs=True, **_MGPU_FWD), (dout, rhs, gs)),
            "tokamax": (_tok_rd(_DLHS_DIM_NUMS), (dout, rhs, gs)),
        },
        "drhs": {
            "triton": (lambda a, b, g: _triton_pallas_call(a, b, g, _DRHS_DIM_NUMS), (lhs, dout, gs)),
            # mgpu drhs (transposed_ragged_dot) needs cast-transposed operands; covered by the prior
            # bench. Here we only contrast tokamax vs triton on drhs (mgpu omitted for brevity).
            "tokamax": (_tok_rd(_DRHS_DIM_NUMS), (lhs, dout, gs)),
        },
    }
    return impls, ref, flop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tokens-per-expert", default="1024,2048")
    ap.add_argument("--gemms", default="dot1,dot2")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--inner-steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rel-frob-tol", type=float, default=0.03)
    ap.add_argument("--out", default="/app/scratch/tokamax_ragged.json")
    args = ap.parse_args()

    dev = jax.devices()[0].device_kind if jax.devices() else "none"
    print(f"backend={jax.default_backend()} device={dev} tokamax={_TOK_VER} has_tokamax={_HAS_TOKAMAX}")
    if not _HAS_TOKAMAX:
        print("TOKAMAX UNAVAILABLE — aborting")
        return

    toks = [int(t) for t in args.tokens_per_expert.split(",")]
    results = []
    for gemm in args.gemms.split(","):
        K, N = _GEMMS[gemm]
        for tok in toks:
            impls, ref, flop = _layouts(K, N, tok, args.experts)
            for layout in ("fwd", "dlhs", "drhs"):
                row = {"gemm": gemm, "tokens_per_expert": tok, "layout": layout, "K": K, "N": N, "impls": {}}
                for name, (fn, fn_args) in impls[layout].items():
                    try:
                        val = jax.block_until_ready(jax.jit(fn)(*fn_args))
                        err = _rel_frob(val, ref[layout])
                        gated = name != "triton" and err > args.rel_frob_tol
                        if gated:
                            row["impls"][name] = {"error": f"rel_frob={err:.4f}>tol"}
                        else:
                            t = _median_time(fn, fn_args, samples=args.samples, inner_steps=args.inner_steps, warmup=args.warmup)
                            row["impls"][name] = {"median_s": t, "tflops_per_s": flop / t, "pct_peak": 100.0 * (flop / t) / _BF16_PEAK, "rel_frob": err}
                    except Exception as e:  # noqa: BLE001
                        row["impls"][name] = {"error": repr(e)[:160]}
                ti = row["impls"]
                base_t = ti.get("triton", {}).get("median_s")
                def _sp(n):
                    return round(base_t / ti[n]["median_s"], 3) if base_t and "median_s" in ti.get(n, {}) else None
                row["mgpu_speedup"] = _sp("mgpu")
                row["tokamax_speedup"] = _sp("tokamax")
                def _fmt(n):
                    d = ti.get(n, {})
                    return f"{d['tflops_per_s']/1e12:6.1f}TF {d['pct_peak']:4.1f}%" if "median_s" in d else f"ERR:{d.get('error','')[:48]}"
                print(f"  {gemm}/{layout:4s} t={tok:5d} | triton {_fmt('triton')} | mgpu {_fmt('mgpu')} ({row['mgpu_speedup']}x) | tokamax {_fmt('tokamax')} ({row['tokamax_speedup']}x)")
                results.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": dev, "tokamax_version": _TOK_VER, "results": results}, f, indent=2, default=str)
    print(f"result_json {json.dumps({'device': dev, 'results': results}, default=str)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
