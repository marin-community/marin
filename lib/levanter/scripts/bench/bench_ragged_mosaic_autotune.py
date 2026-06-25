# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Autotune the Mosaic-GPU FP8 ragged-dot block config at the real Grug regime (H100).

GFP8-026 found the M0 hybrid LOSES to bf16 end-to-end with the hardcoded block config
(128/128/64), which won the GFP8-024 sweep for a 4x-smaller shape. This sweep re-tunes the
block/scheduling config at the real Grug MoE expert-MLP GEMM shapes and reports the best
config per GEMM plus its speedup vs the bf16-Triton baseline (the kernel the bf16 e2e uses).

The expert MLP issues four grouped GEMMs the mosaic path serves in f8 (forward + dlhs; the
wgrad/drhs runs in bf16 off-path, GFP8-025):

    role     lhs            rhs              layout    M     contract  out_n
    fwd13    x[T,D]         w13[E,D,2F]      _DEFAULT  T     D         2F
    fwd2     g[T,F]         w2[E,F,D]        _DEFAULT  T     F         D
    dlhs13   dout13[T,2F]   w13[E,D,2F]      _DLHS     T     2F        D
    dlhs2    dout2[T,D]     w2[E,F,D]        _DLHS     T     D         F

For each (role, config) we steady-state-time `_mosaic_pallas_call` in-process (one jax
import / cuda bootstrap, fast iteration), capturing lowering failures as error rows. A bf16
`_triton_pallas_call` per role gives the apples-to-apples per-GEMM baseline.

Machine-readable rows (perf-workflow schema) print as `row <json>`; a summary prints the
winning config + speedup per role. Mosaic-GPU is H100-only and needs the cluster CUDA
bootstrap (see mosaic-gpu-cluster-toolchain memory) which must run before `import jax`.

    uv run --no-sync python -u lib/levanter/scripts/bench/bench_ragged_mosaic_autotune.py
"""

import argparse
import glob
import itertools
import json
import os
import shutil
import sys
import tempfile
import time


def _ensure_cuda_toolchain() -> bool:
    """Make ptxas + libdevice discoverable to XLA/Mosaic before jax import (see GFP8-022)."""
    ptxas = shutil.which("ptxas")
    libdevice = None
    if not ptxas:
        for base in sys.path:
            if base and os.path.isdir(base):
                hits = glob.glob(os.path.join(base, "nvidia", "**", "ptxas"), recursive=True)
                if hits:
                    ptxas = hits[0]
                    break
    for base in sys.path:
        if base and os.path.isdir(base):
            hits = glob.glob(os.path.join(base, "nvidia", "**", "libdevice.10.bc"), recursive=True)
            if hits:
                libdevice = hits[0]
                break
    if not ptxas or not libdevice:
        print(f"cuda toolchain: incomplete (ptxas={ptxas}, libdevice={libdevice})")
        return False
    nvvm_parent = os.path.dirname(os.path.dirname(os.path.dirname(libdevice)))
    root = tempfile.mkdtemp(prefix="xla_cuda_")
    os.symlink(os.path.dirname(ptxas), os.path.join(root, "bin"))
    os.symlink(os.path.join(nvvm_parent, "nvvm"), os.path.join(root, "nvvm"))
    os.environ["PATH"] = os.path.dirname(ptxas) + os.pathsep + os.environ.get("PATH", "")
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_cuda_data_dir={root}".strip()
    for var in ("CUDA_DIR", "CUDA_HOME", "CUDA_PATH"):
        os.environ[var] = root
    cwd_link = os.path.join(os.getcwd(), "libdevice.10.bc")
    if not os.path.exists(cwd_link):
        os.symlink(libdevice, cwd_link)
    print(f"cuda toolchain: ptxas={ptxas} libdevice={libdevice}")
    return True


_ensure_cuda_toolchain()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from haliax.nn.ragged_dot import (  # noqa: E402
    _DEFAULT_DIM_NUMS,
    _DEFAULT_MOSAIC_CONFIG,
    _DLHS_DIM_NUMS,
    MosaicBlockConfig,
    _mosaic_pallas_call,
    _mosaic_ragged_dot,
    _triton_pallas_call,
)

_E4M3 = jnp.float8_e4m3fn

# H100 SXM5 roofline constants. Dense (no 2:4 sparsity) tensor-core peak and HBM3 bandwidth — the
# ceiling "theoretical maximum" the goal measures against. f8 (e4m3) peak is ~2x the bf16 peak.
_H100_F8_PEAK_TFLOPS = 1978.9
_H100_BF16_PEAK_TFLOPS = 989.4
_H100_HBM_BW_BYTES_PER_S = 3.35e12

# Curated bounded config space (block_m, block_n, block_k, max_concurrent_steps, grid_block_n).
# GFP8-027 found 128/128/256/steps2 best across all four GEMMs, but that sweep capped
# max_concurrent_steps at 4 and every deep config used block_k>=256 — at f8 1B that is ~64KB
# smem/stage (m*k+n*k), so block_k=256 caps pipeline depth at ~3 stages before overflowing the
# H100's ~228KB smem. Smaller block_k frees smem for deeper pipelines (steps 6-8), the usual
# dominant MFU lever on these staged Mosaic kernels — the unexplored corner this sweep probes.
_CURATED_CONFIGS = [
    # Region A — GFP8-027 winner neighborhood (block_k=256, shallow depth, smem-capped).
    (128, 128, 256, 2, 1),  # GFP8-027 winner / current default
    (128, 128, 256, 3, 1),
    (128, 128, 256, 2, 2),
    (64, 128, 256, 2, 1),
    (64, 128, 256, 3, 1),
    (256, 128, 256, 2, 1),
    # Region B — block_k=128, push pipeline depth (the unexplored corner).
    (128, 128, 128, 3, 1),
    (128, 128, 128, 4, 1),
    (128, 128, 128, 6, 1),
    (128, 128, 128, 4, 2),
    (64, 128, 128, 4, 1),
    (64, 128, 128, 6, 1),
    (64, 128, 128, 8, 1),
    (256, 128, 128, 3, 1),
    (256, 128, 128, 4, 1),
    (128, 256, 128, 3, 1),
    (128, 256, 128, 4, 1),
    # Region C — block_k=64, deepest pipelines (latency-hiding extreme).
    (128, 128, 64, 4, 1),
    (128, 128, 64, 6, 1),
    (128, 128, 64, 8, 1),
    (64, 128, 64, 8, 1),
]
_FULL_GRID_AXES = {
    "block_m": (64, 128, 256),
    "block_n": (128, 256),
    "block_k": (64, 128, 256),
    "max_concurrent_steps": (2, 4, 6),
    "grid_block_n": (1, 2),
}

# Refinement grid around the GFP8-029 winner (128/128/128 steps4 gbn2): push pipeline depth at
# gbn2, larger tiles, and gbn4 — the corners the broad curated grid only sampled at gbn1.
# block_n stays a multiple of 128: f8 swizzle-128 TMA needs the trailing tile >=128 elements, so
# block_n=64 raises cuTensorMapEncodeTiled misaligned-address (a HARD CUDA error that poisons the
# in-process context for every later config, not a catchable lowering failure). 256x256 dropped
# (f32 acc register-bound). Keep to the known-lowerable shape family the broad sweep validated.
_REFINE_CONFIGS = [
    (128, 128, 128, 4, 2),  # GFP8-029 winner (control)
    (128, 128, 128, 5, 2),
    (128, 128, 128, 6, 2),
    (128, 128, 128, 4, 4),
    (128, 128, 128, 6, 4),
    (256, 128, 128, 3, 2),
    (256, 128, 128, 4, 2),
    (128, 256, 128, 3, 2),
    (128, 256, 128, 4, 2),
    (64, 128, 128, 6, 2),
    (64, 128, 128, 8, 2),
    (128, 128, 64, 6, 2),
    (128, 128, 64, 8, 2),
    (256, 128, 64, 6, 2),
]


def _roofline(tokens, contract, out_n, experts, impl):
    """Per-GEMM roofline ceiling (TFLOP/s) on H100 for the given GEMM and dtype.

    Returns (roofline_tflops, peak_tflops, bound) where ``bound`` is "compute" or "memory".
    FLOPs = 2*M*K*N; bytes = lhs(M*K) + weight(E*K*N) + out(M*N), inputs 1B (f8) / 2B (bf16),
    output 2B (bf16 accumulation target in both paths).
    """
    in_bytes = 1 if impl == "mosaic" else 2
    flops = 2.0 * tokens * contract * out_n
    byts = (tokens * contract + experts * contract * out_n) * in_bytes + tokens * out_n * 2
    peak = _H100_F8_PEAK_TFLOPS if impl == "mosaic" else _H100_BF16_PEAK_TFLOPS
    compute_time = flops / (peak * 1e12)
    memory_time = byts / _H100_HBM_BW_BYTES_PER_S
    if compute_time >= memory_time:
        return flops / compute_time / 1e12, peak, "compute"
    return flops / memory_time / 1e12, peak, "memory"


def _roles(tokens, hidden, intermediate, experts, seed=0):
    """Build f8 operands + dim-nums for the four mosaic-served GEMMs at the real Grug shapes."""
    rng = np.random.default_rng(seed)
    two_f = 2 * intermediate

    def f8(shape, scale=1.0):
        return jnp.asarray(rng.standard_normal(shape) * scale, _E4M3)

    counts = rng.multinomial(tokens, np.ones(experts) / experts)
    group_sizes = jnp.asarray(counts, jnp.int32)

    x = f8((tokens, hidden))
    g = f8((tokens, intermediate))
    w13 = f8((experts, hidden, two_f), 0.08)
    w2 = f8((experts, intermediate, hidden), 0.08)
    dout13 = f8((tokens, two_f))
    dout2 = f8((tokens, hidden))

    # (label, lhs, rhs, dim_nums, contract_dim, out_n)
    return group_sizes, [
        ("fwd13", x, w13, _DEFAULT_DIM_NUMS, hidden, two_f),
        ("fwd2", g, w2, _DEFAULT_DIM_NUMS, intermediate, hidden),
        ("dlhs13", dout13, w13, _DLHS_DIM_NUMS, two_f, hidden),
        ("dlhs2", dout2, w2, _DLHS_DIM_NUMS, hidden, intermediate),
    ]


def _time_call(fn, *args, steps, warmup):
    """Return (compile_time, mean steady-state time) in seconds, or raise on lowering failure."""
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_time = time.perf_counter() - start
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(compiled(*args))
    return compile_time, (time.perf_counter() - start) / steps


def _fairness(roles, group_sizes, tokens, steps, warmup):
    """Attribution pass at the tuned default config: isolate the f8 *dtype* benefit from the *kernel*
    benefit, and the forward's weight-transpose tax.

      f8_mosaic     : f8 operands, Mosaic kernel (the shipped f8 path)
      bf16_mosaic   : bf16 operands, SAME Mosaic kernel + SAME config (dtype-only isolation)
      bf16_triton   : bf16 operands, production Triton kernel (the deployed bf16 baseline)
      f8_kernel_only: forward roles only — Mosaic f8 with the weight pre-swapped K-major, so the
                      `swapaxes(rhs)` transpose the forward pays is OUTSIDE the timed region.
    """
    cfg = _DEFAULT_MOSAIC_CONFIG
    # bf16 operands are 2B, so the f8-tuned deep pipeline (block_k=128 steps=6 = ~393KB smem) overflows
    # the H100. Give bf16-on-mosaic its OWN feasible config set (shallower / smaller block_k, all <=228KB)
    # and take its best — a fair same-kernel dtype comparison rather than forcing the f8 config.
    bf16_cfgs = [
        MosaicBlockConfig(128, 128, 128, 3, 4),
        MosaicBlockConfig(128, 128, 128, 2, 4),
        MosaicBlockConfig(128, 128, 64, 6, 4),
        MosaicBlockConfig(128, 128, 64, 4, 4),
        MosaicBlockConfig(128, 128, 64, 4, 2),
    ]
    print(f"=== fairness @ tuned default {cfg} (bf16-mosaic gets its own feasible config) ===")
    for label, lhs, rhs, dim_nums, contract, out_n in roles:
        flops = 2 * tokens * contract * out_n
        lhs_b, rhs_b = lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16)
        row = {"role": label}
        try:
            _, row["f8_mosaic"] = _time_call(
                lambda a, b, gs: _mosaic_pallas_call(a, b, gs, dim_nums, config=cfg),
                lhs,
                rhs,
                group_sizes,
                steps=steps,
                warmup=warmup,
            )
        except Exception as exc:  # noqa: BLE001
            row["f8_mosaic"] = None
            print(f"  {label} f8_mosaic: FAILED {str(exc)[:120]}")
        # bf16 on the SAME Mosaic kernel, best of its feasible configs (dtype-isolated comparison).
        bf16_best = None
        for bc in bf16_cfgs:
            try:
                _, st = _time_call(
                    lambda a, b, gs: _mosaic_pallas_call(a, b, gs, dim_nums, config=bc),
                    lhs_b,
                    rhs_b,
                    group_sizes,
                    steps=steps,
                    warmup=warmup,
                )
                bf16_best = st if bf16_best is None else min(bf16_best, st)
            except Exception:  # noqa: BLE001
                continue
        row["bf16_mosaic"] = bf16_best
        try:
            _, row["bf16_triton"] = _time_call(
                lambda a, b, gs: _triton_pallas_call(a, b, gs, dim_nums),
                lhs_b,
                rhs_b,
                group_sizes,
                steps=steps,
                warmup=warmup,
            )
        except Exception as exc:  # noqa: BLE001
            row["bf16_triton"] = None
            print(f"  {label} bf16_triton: FAILED {str(exc)[:120]}")
        # Forward roles pay a real f8 weight transpose inside _mosaic_pallas_call; measure the kernel
        # with the weight pre-swapped (transpose excluded) to size that tax. dlhs uses rhs natural.
        if dim_nums == _DEFAULT_DIM_NUMS:
            rhs_kmajor = jnp.swapaxes(rhs, 1, 2)
            try:
                _, st = _time_call(
                    lambda a, b, gs: _mosaic_ragged_dot(a, b, gs, cfg),
                    lhs,
                    rhs_kmajor,
                    group_sizes,
                    steps=steps,
                    warmup=warmup,
                )
                row["f8_kernel_only"] = st
            except Exception:  # noqa: BLE001
                row["f8_kernel_only"] = None

        def tf(t):
            return flops / t / 1e12 if t else float("nan")

        def ms(t):
            return t * 1e3 if t else float("nan")

        f8, bm, bt = row.get("f8_mosaic"), row.get("bf16_mosaic"), row.get("bf16_triton")
        print(
            f"  {label:7} f8_mosaic={ms(f8):.3f}ms({tf(f8):.0f}TF)  bf16_mosaic={ms(bm):.3f}ms({tf(bm):.0f}TF)  "
            f"bf16_triton={ms(bt):.3f}ms({tf(bt):.0f}TF)"
        )
        spt = f"{bt/f8:.2f}x" if (f8 and bt) else "n/a"
        spm = f"{bm/f8:.2f}x" if (f8 and bm) else "n/a"
        ko = row.get("f8_kernel_only")
        kostr = f"  f8_kernel_only={ms(ko):.3f}ms (transpose tax={ms(f8)-ms(ko):.3f}ms)" if ko else ""
        print(f"          -> f8 vs bf16_triton(deployed)={spt}  f8 vs bf16_mosaic(dtype-only)={spm}{kostr}")
        print(
            "row "
            + json.dumps(
                {k: row.get(k) for k in ("role", "f8_mosaic", "bf16_mosaic", "bf16_triton", "f8_kernel_only")}
            )
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=5632)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--full-grid", action="store_true", help="full cross-product instead of curated set")
    ap.add_argument("--refine", action="store_true", help="refinement grid around the GFP8-029 winner")
    ap.add_argument("--fairness", action="store_true", help="dtype-isolated f8-vs-bf16 + forward transpose tax")
    args = ap.parse_args()

    if args.fairness:
        group_sizes, roles = _roles(args.tokens, args.hidden, args.intermediate, args.experts)
        print(f"hardware: {jax.devices()[0].device_kind}")
        _fairness(roles, group_sizes, args.tokens, args.steps, args.warmup)
        return

    if args.full_grid:
        configs = [c for c in itertools.product(*_FULL_GRID_AXES.values())]
    elif args.refine:
        configs = list(_REFINE_CONFIGS)
    else:
        configs = list(_CURATED_CONFIGS)

    device_type = jax.devices()[0].device_kind
    git_sha = os.environ.get("GIT_SHA", "")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    print(f"hardware: {device_type}  configs={len(configs)}  roles=4")

    group_sizes, roles = _roles(args.tokens, args.hidden, args.intermediate, args.experts)

    def emit(impl, role, block_sizes, compile_time, steady, contract, out_n, error=""):
        flops = 2 * args.tokens * contract * out_n
        tflops = (flops / steady / 1e12) if (steady and not error) else None
        roof_tflops, peak_tflops, bound = _roofline(args.tokens, contract, out_n, args.experts, impl)
        row = {
            "kernel": "ragged_dot_mosaic_f8",
            "implementation": impl,
            "shape": f"M{args.tokens}_K{contract}_N{out_n}",
            "role": role,
            "dtype": "e4m3" if impl == "mosaic" else "bf16",
            "backend": "gpu",
            "device_type": device_type,
            "device_count": 1,
            "block_sizes": block_sizes,
            "compile_time": compile_time,
            "steady_state_time": steady,
            "achieved_tflops_per_s": tflops,
            "roofline_tflops_per_s": roof_tflops,
            "roofline_bound": bound,
            "pct_of_roofline": (tflops / roof_tflops) if tflops else None,
            "pct_of_peak": (tflops / peak_tflops) if tflops else None,
            "error": error,
            "git_sha": git_sha,
            "xla_flags": xla_flags,
            "backend_env": "jax[cuda13]",
        }
        print("row " + json.dumps(row))
        return row

    rows = []
    # bf16-Triton baseline per role (the kernel the bf16 e2e actually runs).
    baselines = {}
    for label, lhs, rhs, dim_nums, contract, out_n in roles:
        lhs_b, rhs_b = lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16)
        try:
            ct, st = _time_call(
                lambda a, b, gs: _triton_pallas_call(a, b, gs, dim_nums),
                lhs_b,
                rhs_b,
                group_sizes,
                steps=args.steps,
                warmup=args.warmup,
            )
            baselines[label] = st
            rows.append(emit("triton_bf16", label, "auto", ct, st, contract, out_n))
        except Exception as exc:  # noqa: BLE001 — record unsupported/lowering failures as rows
            rows.append(emit("triton_bf16", label, "auto", None, None, contract, out_n, error=repr(exc)[:300]))

    # Mosaic f8 sweep per role x config.
    for label, lhs, rhs, dim_nums, contract, out_n in roles:
        for bm, bn, bk, steps_, gbn in configs:
            block_sizes = f"{bm}x{bn}x{bk}_steps{steps_}_gbn{gbn}"
            if contract % bk != 0:
                rows.append(
                    emit(
                        "mosaic",
                        label,
                        block_sizes,
                        None,
                        None,
                        contract,
                        out_n,
                        error=f"block_k {bk} !| contract {contract}",
                    )
                )
                continue
            cfg = MosaicBlockConfig(block_m=bm, block_n=bn, block_k=bk, max_concurrent_steps=steps_, grid_block_n=gbn)
            try:
                ct, st = _time_call(
                    lambda a, b, gs: _mosaic_pallas_call(a, b, gs, dim_nums, config=cfg),
                    lhs,
                    rhs,
                    group_sizes,
                    steps=args.steps,
                    warmup=args.warmup,
                )
                rows.append(emit("mosaic", label, block_sizes, ct, st, contract, out_n))
            except Exception as exc:  # noqa: BLE001
                rows.append(emit("mosaic", label, block_sizes, None, None, contract, out_n, error=repr(exc)[:300]))

    # Summary: best mosaic config per role + speedup vs bf16 baseline.
    print("=== summary: best mosaic config per role ===")
    summary = {}
    for label, _, _, _, contract, out_n in roles:
        ok = [r for r in rows if r["role"] == label and r["implementation"] == "mosaic" and not r["error"]]
        if not ok:
            print(f"  {label}: no viable mosaic config")
            continue
        best = min(ok, key=lambda r: r["steady_state_time"])
        base = baselines.get(label)
        speedup = (base / best["steady_state_time"]) if base else None
        summary[label] = {
            "best_config": best["block_sizes"],
            "mosaic_time_s": best["steady_state_time"],
            "bf16_time_s": base,
            "speedup_vs_bf16": speedup,
            "f8_pct_of_roofline": best["pct_of_roofline"],
            "f8_pct_of_peak": best["pct_of_peak"],
            "roofline_bound": best["roofline_bound"],
        }
        sp = f"{speedup:.3f}x" if speedup else "n/a"
        roof = f"{best['pct_of_roofline']*100:.0f}%roof" if best["pct_of_roofline"] else "n/a"
        print(
            f"  {label:7} best={best['block_sizes']:24} mosaic={best['steady_state_time']*1e3:.3f}ms "
            f"bf16={base*1e3 if base else float('nan'):.3f}ms speedup={sp} f8={roof}({best['roofline_bound']})"
        )
    print("summary_json " + json.dumps(summary))


if __name__ == "__main__":
    main()
