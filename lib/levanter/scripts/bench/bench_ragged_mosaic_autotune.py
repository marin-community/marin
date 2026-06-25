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
    _DLHS_DIM_NUMS,
    MosaicBlockConfig,
    _mosaic_pallas_call,
    _triton_pallas_call,
)

_E4M3 = jnp.float8_e4m3fn

# Curated bounded config space (block_m, block_n, block_k, max_concurrent_steps, grid_block_n).
# Spans the knobs that matter for f8 wgmma grouped GEMM on H100 without a full cross-product;
# expand around the winner with --full-grid if the best sits on a boundary.
_CURATED_CONFIGS = [
    (128, 128, 64, 2, 1),  # GFP8-024 default (baseline)
    (128, 256, 64, 2, 1),
    (256, 128, 64, 2, 1),
    (256, 256, 64, 2, 1),
    (128, 128, 128, 2, 1),
    (128, 256, 128, 2, 1),
    (256, 128, 128, 2, 1),
    (256, 256, 128, 2, 1),
    (128, 256, 64, 4, 1),
    (256, 256, 64, 4, 1),
    (128, 256, 128, 4, 1),
    (256, 256, 128, 4, 1),
    (128, 256, 64, 2, 2),
    (256, 256, 64, 2, 2),
    (128, 128, 256, 2, 1),
    (256, 128, 256, 2, 1),
]
_FULL_GRID_AXES = {
    "block_m": (64, 128, 256),
    "block_n": (128, 256),
    "block_k": (64, 128, 256),
    "max_concurrent_steps": (2, 4),
    "grid_block_n": (1, 2),
}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=5632)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--full-grid", action="store_true", help="full cross-product instead of curated set")
    args = ap.parse_args()

    if args.full_grid:
        configs = [
            c
            for c in itertools.product(*_FULL_GRID_AXES.values())
        ]
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
                    emit("mosaic", label, block_sizes, None, None, contract, out_n, error=f"block_k {bk} !| contract {contract}")
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
        summary[label] = {"best_config": best["block_sizes"], "mosaic_time_s": best["steady_state_time"], "bf16_time_s": base, "speedup_vs_bf16": speedup}
        sp = f"{speedup:.3f}x" if speedup else "n/a"
        print(f"  {label:7} best={best['block_sizes']:24} mosaic={best['steady_state_time']*1e3:.3f}ms bf16={base*1e3 if base else float('nan'):.3f}ms speedup={sp}")
    print("summary_json " + json.dumps(summary))


if __name__ == "__main__":
    main()
