# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""bf16 ragged-dot kernel shoot-out: vendored Triton vs autotuned Mosaic-GPU (mgpu).

Main's GPU ``ragged_dot`` serves the bf16 MoE expert-MLP with the vendored Pallas-Triton kernel only
(``haliax.nn.ragged_dot``). This probe asks, with no FP8 in the picture:

    **At the realistic d2560 per-expert GEMM shapes, does the *autotuned* bf16 Mosaic-GPU ragged kernel
    beat the vendored bf16 Triton kernel — and by how much, per ragged-dot layout?**

Three ragged-dot layouts make up the MoE fwd+bwd (per gated-MLP matmul):
  - ``fwd``  : ``lhs[M,K] @ rhs[G,K,N] -> out[M,N]``      (contract K)
  - ``dlhs`` : ``dout[M,N] @ rhs[G,K,N]^T -> dlhs[M,K]``  (contract N) — input gradient
  - ``drhs`` : ``lhs[M,K]^T @ dout[M,N] -> drhs[G,K,N]``  (contract ragged token axis) — weight gradient

Implementations timed per layout:
  - ``triton`` — main's vendored Pallas-Triton kernel (production baseline, its own default blocks).
  - ``mgpu``   — ``jax.experimental.pallas.ops.gpu.ragged_dot_mgpu`` (+ ``transposed_ragged_dot`` for the
                 wgrad), Mosaic-GPU, **autotuned** over a block-config grid at the operating point (the
                 winning config is then timed at every tokens/expert). Mosaic needs tuning to be
                 competitive — the stock default loses badly (logbook).
  - ``dense``  — a single ``jax.lax.dot_general`` bf16 GEMM at matched total FLOP (cuBLAS): the
                 mature-kernel roofline ceiling. (XLA ``ragged_dot_general`` on GPU is a ~2%-peak
                 fallback loop, NOT cuBLAS, so it is not a useful baseline and is omitted.)

The per-impl SUM over all 6 GEMMs (dot1 + dot2, x fwd/dlhs/drhs) is the apples-to-apples "e2e GEMM
time" the bf16 expert-MLP would pay (elementwise activation work is identical across impls). Every
Mosaic config is numerically gated against a dense reference (rel-Frobenius) before it can win.

Stock JAX only — no forked FP8 jaxlib (bf16 Mosaic is upstream). Reuses the bench module's CUDA-toolchain
bootstrap + steady-state timer.

    uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 --enable-extra-resources --extra gpu \\
        -- python lib/levanter/scripts/bench/bench_bf16_ragged_dot.py --out /app/scratch/bf16_ragged.json
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_ragged_fp8_autotune as B  # noqa: E402  (CUDA toolchain bootstrap must precede `import jax`)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu as _mgpu  # noqa: E402

from haliax._src.transposed_ragged_dot_mgpu import WgradBlockConfig, transposed_ragged_dot  # noqa: E402
from haliax.nn.ragged_dot import (  # noqa: E402
    _DEFAULT_DIM_NUMS,
    _DLHS_DIM_NUMS,
    _DRHS_DIM_NUMS,
    _triton_pallas_call,
)

_BF16_PEAK = B._H100_SXM_BF16_TFLOPS_PER_S

# The two gated-MLP forward GEMMs of the real d2560 model (D=2560 hidden, F=1280 intermediate):
#   dot1: x[T,D] @ w13[E,D,2F] -> [T,2F]   (K=D=2560,  N=2F=2560)
#   dot2: h[T,F] @ w2 [E,F,D]  -> [T,D]    (K=F=1280,  N=D=2560)
_GEMMS = {"dot1": (2560, 2560), "dot2": (1280, 2560)}


def _fwd_grid():
    """Mosaic forward/dlhs block-config candidates (ragged_dot_mgpu kwargs).

    block_n=128 dominates for our N (2560/1280); grid_block_n is the key occupancy lever for these
    shapes (logbook: stock grid_block_n=1 default loses badly), so it is swept widest.
    """
    out = []
    for bm, bn, bk, s, g in itertools.product((64, 128), (128,), (64, 128), (4, 6), (1, 2, 4)):
        out.append(dict(block_m=bm, block_n=bn, block_k=bk, max_concurrent_steps=s, grid_block_n=g))
    return out


def _wgrad_grid():
    """Mosaic wgrad (transposed_ragged_dot) WgradBlockConfig candidates (bf16 doubles smem vs f8, so
    block_k=128/steps=6 overflows H100 smem and is pruned at runtime)."""
    out = []
    for bm, bn, bk, s, g in itertools.product((64, 128), (64, 128), (64, 128), (4, 6), (1, 2)):
        out.append(WgradBlockConfig(block_m=bm, block_n=bn, block_k=bk, max_concurrent_steps=s, grid_block_n=g))
    return out


def _median_time(fn, args, *, samples, inner_steps, warmup):
    _, times = B.time_steady_state(fn, args, samples=samples, inner_steps=inner_steps, warmup=warmup)
    return float(np.median(times))


def _rel_frob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --- references (dense bf16, used both as numerics oracle and cuBLAS ceiling) --------------------
def _dense_ref_fwd(lhs, w_first_expert):  # contract K: [T,K]@[K,N]->[T,N]
    return jax.lax.dot_general(lhs, w_first_expert, (((1,), (0,)), ((), ())), preferred_element_type=jnp.bfloat16)


def _grouped_ref(fn_per_expert, lhs, rhs, gs):
    """Exact ragged reference by looping experts in numpy-land (small, for the numerics gate only)."""
    return fn_per_expert(lhs, rhs, gs)


def _build_layouts(K, N, tok, E):
    rng = np.random.default_rng(0)
    T = E * tok
    gs = jnp.full((E,), tok, jnp.int32)
    lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
    rhs = jnp.asarray(rng.standard_normal((E, K, N)) * 0.05, jnp.bfloat16)
    dout = jnp.asarray(rng.standard_normal((T, N)) * 0.1, jnp.bfloat16)
    lhs_t = jnp.swapaxes(lhs, 0, 1)  # [K,T]
    dout_t = jnp.swapaxes(dout, 0, 1)  # [N,T]
    flop = 2.0 * T * K * N

    # Triton reference outputs (already validated against XLA in the prior run; triton is the oracle).
    ref_fwd = jax.block_until_ready(jax.jit(lambda a, b, g: _triton_pallas_call(a, b, g, _DEFAULT_DIM_NUMS))(lhs, rhs, gs))
    ref_dlhs = jax.block_until_ready(jax.jit(lambda a, b, g: _triton_pallas_call(a, b, g, _DLHS_DIM_NUMS))(dout, rhs, gs))
    ref_drhs = jax.block_until_ready(jax.jit(lambda a, b, g: _triton_pallas_call(a, b, g, _DRHS_DIM_NUMS))(lhs, dout, gs))

    layouts = {
        "fwd": {
            "flop": flop,
            "ref": ref_fwd,
            "triton": (lambda a, b, g: _triton_pallas_call(a, b, g, _DEFAULT_DIM_NUMS), (lhs, rhs, gs)),
            "mgpu_builder": lambda cfg: (lambda a, b, g: _mgpu.ragged_dot(a, b, group_sizes=g, transpose_rhs=False, **cfg), (lhs, rhs, gs)),
            "mgpu_grid": "fwd",
        },
        "dlhs": {
            "flop": flop,
            "ref": ref_dlhs,
            "triton": (lambda a, b, g: _triton_pallas_call(a, b, g, _DLHS_DIM_NUMS), (dout, rhs, gs)),
            "mgpu_builder": lambda cfg: (lambda a, b, g: _mgpu.ragged_dot(a, b, group_sizes=g, transpose_rhs=True, **cfg), (dout, rhs, gs)),
            "mgpu_grid": "fwd",
        },
        "drhs": {
            "flop": flop,
            "ref": ref_drhs,
            "triton": (lambda a, b, g: _triton_pallas_call(a, b, g, _DRHS_DIM_NUMS), (lhs, dout, gs)),
            "mgpu_builder": lambda cfg: (lambda a, b, g: transposed_ragged_dot(a, b, g, out_dtype=jnp.bfloat16, config=cfg), (lhs_t, dout_t, gs)),
            "mgpu_grid": "wgrad",
        },
    }
    return layouts, flop


def _autotune_mgpu(builder, grid, ref, tol, quick):
    """Return (best_cfg, best_time, n_valid) by quick-timing every numerically-valid config in the grid."""
    best_cfg, best_t, n_valid = None, float("inf"), 0
    for cfg in grid:
        jax.clear_caches()  # bound device-memory growth across the many compiled candidates
        try:
            fn, args = builder(cfg)
            val = jax.block_until_ready(jax.jit(fn)(*args))
        except Exception:  # noqa: BLE001 — smem overflow / bad divisibility / unsupported config
            continue
        if _rel_frob(val, ref) > tol:
            continue
        n_valid += 1
        try:
            t = _median_time(fn, args, samples=quick[0], inner_steps=quick[1], warmup=quick[2])
        except Exception:  # noqa: BLE001
            continue
        if t < best_t:
            best_cfg, best_t = cfg, t
    return best_cfg, best_t, n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tokens-per-expert", default="512,1024,2048,4096")
    ap.add_argument("--tune-at", type=int, default=1024, help="tokens/expert at which to autotune the mgpu config")
    ap.add_argument("--gemms", default="dot1,dot2")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--inner-steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rel-frob-tol", type=float, default=0.03)
    ap.add_argument("--out", default="/app/scratch/bf16_ragged.json")
    args = ap.parse_args()

    backend = jax.default_backend()
    dev = jax.devices()[0].device_kind if jax.devices() else "none"
    E = args.experts
    toks = [int(t) for t in args.tokens_per_expert.split(",")]
    gemms = args.gemms.split(",")
    grids = {"fwd": _fwd_grid(), "wgrad": _wgrad_grid()}
    quick = (5, 12, 3)  # fast timing during autotune search
    print(f"backend={backend} device={dev} E={E} tune_at={args.tune_at} grid_sizes={{k:len(v) for k,v in grids.items()}}")

    # ---- Phase 1: autotune mgpu config per (gemm, layout) at the operating point -------------------
    winners = {}  # (gemm, layout) -> cfg
    for gemm in gemms:
        K, N = _GEMMS[gemm]
        layouts, _ = _build_layouts(K, N, args.tune_at, E)
        for layout, spec in layouts.items():
            grid = grids[spec["mgpu_grid"]]
            cfg, t, nv = _autotune_mgpu(spec["mgpu_builder"], grid, spec["ref"], args.rel_frob_tol, quick)
            winners[(gemm, layout)] = cfg
            print(f"AUTOTUNE {gemm}/{layout:4s} -> {cfg}  ({nv}/{len(grid)} valid, best~{t*1e6:.1f}us)")

    # ---- Phase 2: time triton vs mgpu(winner) vs dense at every tokens/expert ---------------------
    results = []
    for gemm in gemms:
        K, N = _GEMMS[gemm]
        for tok in toks:
            layouts, flop = _build_layouts(K, N, tok, E)
            T = E * tok
            row = {"gemm": gemm, "K": K, "N": N, "tokens_per_expert": tok, "experts": E, "T": T, "flop": flop, "layouts": {}}
            # dense cuBLAS ceiling (single GEMM at matched total FLOP M=T)
            rng = np.random.default_rng(1)
            d_lhs = jnp.asarray(rng.standard_normal((T, K)) * 0.1, jnp.bfloat16)
            d_w = jnp.asarray(rng.standard_normal((K, N)) * 0.05, jnp.bfloat16)
            try:
                t = _median_time(_dense_ref_fwd, (d_lhs, d_w), samples=args.samples, inner_steps=args.inner_steps, warmup=args.warmup)
                row["dense"] = {"median_s": t, "tflops_per_s": flop / t, "pct_peak": 100.0 * (flop / t) / _BF16_PEAK}
                print(f"  {gemm} t={tok:5d} dense(cuBLAS) {t*1e6:8.1f} us  {flop/t/1e12:6.1f} TF/s  {100.0*(flop/t)/_BF16_PEAK:5.1f}% peak")
            except Exception as e:  # noqa: BLE001
                row["dense"] = {"error": repr(e)}
            for layout, spec in layouts.items():
                ref = spec["ref"]
                lr = {}
                # triton
                fn, fn_args = spec["triton"]
                t = _median_time(fn, fn_args, samples=args.samples, inner_steps=args.inner_steps, warmup=args.warmup)
                lr["triton"] = {"median_s": t, "tflops_per_s": flop / t, "pct_peak": 100.0 * (flop / t) / _BF16_PEAK}
                # mgpu winner
                cfg = winners[(gemm, layout)]
                if cfg is not None:
                    fn, fn_args = spec["mgpu_builder"](cfg)
                    try:
                        val = jax.block_until_ready(jax.jit(fn)(*fn_args))
                        err = _rel_frob(val, ref)
                        t = _median_time(fn, fn_args, samples=args.samples, inner_steps=args.inner_steps, warmup=args.warmup)
                        lr["mgpu"] = {"median_s": t, "tflops_per_s": flop / t, "pct_peak": 100.0 * (flop / t) / _BF16_PEAK, "rel_frob": err, "cfg": str(cfg)}
                    except Exception as e:  # noqa: BLE001
                        lr["mgpu"] = {"error": repr(e)}
                else:
                    lr["mgpu"] = {"error": "no valid config found in autotune"}
                if "median_s" in lr.get("triton", {}) and "median_s" in lr.get("mgpu", {}):
                    lr["mgpu_speedup"] = lr["triton"]["median_s"] / lr["mgpu"]["median_s"]
                row["layouts"][layout] = lr
                tr, mg = lr["triton"], lr.get("mgpu", {})
                sp = lr.get("mgpu_speedup")
                mg_s = f"{mg.get('tflops_per_s', 0)/1e12:6.1f} TF/s {mg.get('pct_peak', 0):5.1f}%" if "median_s" in mg else f"ERR {mg.get('error', '')[:40]}"
                print(f"  {gemm}/{layout:4s} t={tok:5d} triton {tr['tflops_per_s']/1e12:6.1f} TF/s {tr['pct_peak']:5.1f}%  | mgpu {mg_s}  | speedup={sp if sp is None else round(sp,3)}")
            results.append(row)

    # ---- e2e GEMM totals (sum over all 6 GEMMs) per tokens/expert ---------------------------------
    e2e = {}
    for tok in toks:
        tot_tr = tot_mg = 0.0
        ok = True
        for row in results:
            if row["tokens_per_expert"] != tok:
                continue
            for layout, lr in row["layouts"].items():
                if "median_s" in lr.get("triton", {}) and "median_s" in lr.get("mgpu", {}):
                    tot_tr += lr["triton"]["median_s"]
                    tot_mg += lr["mgpu"]["median_s"]
                else:
                    ok = False
        if ok:
            e2e[tok] = {"triton_s": tot_tr, "mgpu_s": tot_mg, "mgpu_speedup": tot_tr / tot_mg}
            print(f"E2E-GEMM t={tok:5d}: triton={tot_tr*1e6:.1f}us mgpu={tot_mg*1e6:.1f}us -> {tot_tr/tot_mg:.3f}x")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {"backend": backend, "device": dev, "experts": E, "winners": {f"{g}/{l}": str(c) for (g, l), c in winners.items()}, "results": results, "e2e_gemm": e2e}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"result_json {json.dumps(payload, default=str)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
