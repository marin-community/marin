# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Decisive forward gate for the S5 Grug FP8 thesis: does a genuine Hopper f8 wgmma
grouped GEMM beat bf16?

Earlier S5 work established that quantize-around + autotune of the *pallas-triton* ragged
kernel cannot beat bf16 (fp8 fwd 0.60-0.67x bf16; fp8 backward hits two shape-independent
backend walls -- GFP8-020/021). The triton backend abstracts the MMA dtype away, so it was
never clear whether the f8 forward issued a real Hopper f8 wgmma or an internal upcast.

JAX 0.10.0 ships ``jax.experimental.pallas.ops.gpu.ragged_dot_mgpu`` -- a production
Mosaic-GPU grouped GEMM that calls ``plgpu.wgmma`` directly. Hopper wgmma accepts
``float8_e4m3fn`` operands with f32 accumulation (mosaic/gpu/wgmma.py), so feeding f8 into
this stock kernel gives a GENUINE f8 tensor-core grouped GEMM with no kernel-writing.

This script benchmarks the stock kernel at the real Grug MoE trial-model regime
(hidden=1024, intermediate=512, the two expert GEMMs, two expert-parallel layouts) for
bf16 vs f8e4m3fn, autotuning a bounded (block_m, block_n, block_k, max_concurrent_steps,
grid_block_n) grid per case. It is a pure THROUGHPUT gate: output dtype = input dtype (the
store is negligible vs the matmul), and a coarse parity check vs ``lax.ragged_dot`` guards
against timing a miscompiled kernel. If f8 does not beat bf16 here, no hand-written kernel
can rescue the thesis at this regime.

Mosaic-GPU is H100-only; run on an H100 via Iris.
"""

import argparse
import functools
import glob
import itertools
import json
import os
import shutil
import sys
import tempfile
import time


def _glob_first(name: str) -> str | None:
    """First match of ``nvidia/**/<name>`` across sys.path site-packages dirs."""
    for base in sys.path:
        if base and os.path.isdir(base):
            hits = glob.glob(os.path.join(base, "nvidia", "**", name), recursive=True)
            if hits:
                return hits[0]
    return None


def _ensure_cuda_toolchain() -> bool:
    """Assemble a CUDA dir XLA/Mosaic can use, before jax is imported.

    Mosaic-GPU compiles PTX->SASS via XLA, which needs BOTH ``ptxas`` and the libdevice
    bitcode at ``$CUDA_DIR/nvvm/libdevice/libdevice.10.bc`` (the Triton backend bundles its
    own toolchain, so it dodges this). On the cluster GPU image these ship in *separate*
    wheels (ptxas under ``nvidia/<cuver>/bin``; libdevice under a ``cuda_nvvm`` wheel), and
    jax 0.10.0 only auto-discovers the ``*-cu12`` layout -- so no single directory has both.
    We locate each piece on sys.path and symlink them into one synthetic toolkit root, then
    point XLA at it via ``--xla_gpu_cuda_data_dir``. Must run BEFORE ``import jax``.
    """
    ptxas = shutil.which("ptxas") or _glob_first("ptxas")
    libdevice = _glob_first("libdevice.10.bc")  # at <X>/nvvm/libdevice/libdevice.10.bc
    if not ptxas or not libdevice:
        print(f"cuda toolchain: incomplete (ptxas={ptxas}, libdevice={libdevice})")
        return False
    nvvm_parent = os.path.dirname(os.path.dirname(os.path.dirname(libdevice)))  # <X> with nvvm/
    root = tempfile.mkdtemp(prefix="xla_cuda_")
    os.symlink(os.path.dirname(ptxas), os.path.join(root, "bin"))
    os.symlink(os.path.join(nvvm_parent, "nvvm"), os.path.join(root, "nvvm"))
    os.environ["PATH"] = os.path.dirname(ptxas) + os.pathsep + os.environ.get("PATH", "")
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_cuda_data_dir={root}".strip()
    # Mosaic-GPU's libdevice resolution ignores --xla_gpu_cuda_data_dir and falls back to a
    # cwd-relative "./libdevice.10.bc" / $CUDA_* lookup. Cover both: env vars + a cwd symlink.
    for var in ("CUDA_DIR", "CUDA_HOME", "CUDA_PATH"):
        os.environ[var] = root
    cwd_link = os.path.join(os.getcwd(), "libdevice.10.bc")
    if not os.path.exists(cwd_link):
        os.symlink(libdevice, cwd_link)
    print(f"cuda toolchain: ptxas={ptxas}\n  libdevice={libdevice}\n  XLA_FLAGS={os.environ['XLA_FLAGS']}")
    return True


_ensure_cuda_toolchain()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu  # noqa: E402

# Real Grug MoE trial model (GRUG_MOE_TRIAL_MODEL): hidden=1024, intermediate=512, top_k=4,
# seq 4096 -> 524288 dispatched tokens globally; one of 8 devices sees 65536. The expert MLP
# is two grouped GEMMs: gate/up (K=hidden, N=2*intermediate) then down (K=intermediate, N=hidden).
_HIDDEN = 1024
_INTERMEDIATE = 512
_TOKENS_PER_DEVICE = 65536

# Two device layouts differ only in experts-per-device, hence per-expert occupancy (the key
# compute-vs-memory driver): EP packs 8 experts/device (~8192 tok/expert); DP replicates all 64.
_BUCKETS = [
    {"name": "ep8", "experts": 8},
    {"name": "dp64", "experts": 64},
]

# The two expert GEMMs, as (label, K, N). gate/up produces 2*intermediate; down consumes
# intermediate and produces hidden.
_GEMMS = [
    {"name": "gateup", "k": _HIDDEN, "n": 2 * _INTERMEDIATE},
    {"name": "down", "k": _INTERMEDIATE, "n": _HIDDEN},
]

_DTYPES = [("bf16", jnp.bfloat16), ("f8e4m3", jnp.float8_e4m3fn)]

# Bounded autotune grid: (block_m, block_n, block_k, max_concurrent_steps, grid_block_n).
# f8 operands are 1 byte so larger block_k fits the ~228KB H100 smem budget; wide block_n
# favors Hopper wgmma. Infeasible (smem OOM) configs are caught per-config below.
_CONFIGS = list(
    itertools.product(
        (64, 128),  # block_m
        (128, 256),  # block_n
        (64, 128),  # block_k
        (2, 4, 6),  # max_concurrent_steps
        (1, 2),  # grid_block_n
    )
)


def _time_ms(f, lhs, rhs, *, steps: int, warmup: int) -> float:
    """Steady-state wall-clock per-call time in ms (avoids CUPTI/profiler.measure, which
    fails on this cluster). Compiles, warms up, then times `steps` blocking calls."""
    compiled = jax.jit(f).lower(lhs, rhs).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(lhs, rhs))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(compiled(lhs, rhs))
    return (time.perf_counter() - start) / steps * 1e3


def _make_inputs(tokens: int, k: int, n: int, experts: int, dtype, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Unit-variance activations, modest weights: operands sit inside E4M3 range at unit scale,
    # the regime per-tensor delayed scaling converges to (matches bench_ragged_fp8.py).
    #
    # RHS is stored (G, N, K) -- K-contiguous -- and the kernel is driven with
    # transpose_rhs=True. This is REQUIRED for f8: Hopper f8 wgmma forbids operand transposes
    # (wgmma.py: bytewidth==1 -> b_transpose must be False), which means B must be physically
    # K-major. The transpose_rhs=True path stores B as (N, K) and applies a *logical*
    # transpose_ref (not a hardware transpose), so b_fastest==K and f8 wgmma is legal. The
    # default (G, K, N) layout makes b N-major -> b_transpose=True -> "Only f16 WGMMA supports
    # transposes". Both dtypes use this layout so the comparison varies only dtype.
    lhs = jnp.asarray(rng.standard_normal((tokens, k)), dtype)
    rhs = jnp.asarray(rng.standard_normal((experts, n, k)) * 0.08, dtype)
    counts = rng.multinomial(tokens, np.ones(experts) / experts)
    group_sizes = jnp.asarray(counts, jnp.int32)
    return lhs, rhs, group_sizes


def _autotune_one(lhs, rhs, group_sizes, *, k: int, n: int, steps: int, warmup: int) -> dict:
    """Sweep the config grid for one (shape, dtype); return best runtime + TFLOP/s + parity."""
    m = lhs.shape[0]
    best = {"runtime_ms": float("inf"), "config": None, "n_ok": 0, "n_oom": 0, "n_err": 0, "sample_err": None}
    for block_m, block_n, block_k, msteps, grid_block_n in _CONFIGS:
        if k % block_k or n % (grid_block_n * block_n):
            continue
        f = functools.partial(
            ragged_dot_mgpu.ragged_dot,
            group_sizes=group_sizes,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            max_concurrent_steps=msteps,
            grid_block_n=grid_block_n,
            transpose_rhs=True,
        )
        try:
            runtime_ms = _time_ms(f, lhs, rhs, steps=steps, warmup=warmup)
        except Exception as e:  # noqa: BLE001 -- record smem-OOM / unsupported configs, keep sweeping
            msg = str(e)
            if "shared memory" in msg.lower():
                best["n_oom"] += 1
            else:
                best["n_err"] += 1
                if best["sample_err"] is None:
                    best["sample_err"] = msg.replace("\n", " ")[:300]
            continue
        best["n_ok"] += 1
        if runtime_ms < best["runtime_ms"]:
            best.update(
                runtime_ms=runtime_ms,
                config={
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "max_concurrent_steps": msteps,
                    "grid_block_n": grid_block_n,
                },
            )

    if best["config"] is None:
        best["tflops"] = None
        best["rel_frob_vs_bf16_ref"] = None
        return best

    best["tflops"] = float(2 * m * k * n) / (best["runtime_ms"] / 1e3) / 1e12
    # Coarse correctness guard: compare best-config output to an XLA ragged_dot reference (in
    # the kernel's own dtype). For f8 this folds in f8-output quantization, so it is a sanity
    # bound, not a precision measurement -- a miscompiled kernel shows up as a large value.
    c = best["config"]
    out = ragged_dot_mgpu.ragged_dot(
        lhs,
        rhs,
        group_sizes=group_sizes,
        block_m=c["block_m"],
        block_n=c["block_n"],
        block_k=c["block_k"],
        max_concurrent_steps=c["max_concurrent_steps"],
        grid_block_n=c["grid_block_n"],
        transpose_rhs=True,
    )
    # rhs is stored (G, N, K); lax.ragged_dot wants (G, K, N), so transpose back for the ref.
    ref = lax.ragged_dot(lhs, jnp.transpose(rhs, (0, 2, 1)), group_sizes=group_sizes)
    out, ref = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    best["rel_frob_vs_bf16_ref"] = float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-9))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=None, help="path to write the raw JSON results array")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print(f"hardware: {[d.device_kind for d in jax.devices()]}")
    print(f"buckets: {[b['name'] for b in _BUCKETS]}  gemms: {[g['name'] for g in _GEMMS]}  configs: {len(_CONFIGS)}")

    records = []
    for bucket in _BUCKETS:
        for gemm in _GEMMS:
            for dt_name, dtype in _DTYPES:
                lhs, rhs, gs = _make_inputs(_TOKENS_PER_DEVICE, gemm["k"], gemm["n"], bucket["experts"], dtype)
                rec = _autotune_one(lhs, rhs, gs, k=gemm["k"], n=gemm["n"], steps=args.steps, warmup=args.warmup)
                rec.update(bucket=bucket["name"], gemm=gemm["name"], dtype=dt_name, experts=bucket["experts"])
                records.append(rec)
                if rec["tflops"] is not None:
                    c = rec["config"]
                    tag = (
                        f"{rec['tflops']:7.1f} TF/s  {rec['runtime_ms'] * 1e3:7.1f}us  "
                        f"bm{c['block_m']} bn{c['block_n']} bk{c['block_k']} "
                        f"s{c['max_concurrent_steps']} gbn{c['grid_block_n']}  "
                        f"relfrob={rec['rel_frob_vs_bf16_ref']:.3g}"
                    )
                else:
                    tag = f"NO VIABLE CONFIG (oom={rec['n_oom']} err={rec['n_err']}) :: {rec['sample_err']}"
                print(f"  {bucket['name']:5} {gemm['name']:7} {dt_name:7} -> {tag}")

    if args.artifact:
        with open(args.artifact, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nartifact written: {args.artifact}")

    # f8-vs-bf16 verdict per (bucket, gemm).
    print("\n===== VERDICT: mosaic f8 vs mosaic bf16 (same kernel, forward) =====")
    by_key = {(r["bucket"], r["gemm"], r["dtype"]): r for r in records}
    for bucket in _BUCKETS:
        for gemm in _GEMMS:
            bf16 = by_key.get((bucket["name"], gemm["name"], "bf16"))
            f8 = by_key.get((bucket["name"], gemm["name"], "f8e4m3"))
            if bf16 and f8 and bf16["tflops"] and f8["tflops"]:
                ratio = f8["tflops"] / bf16["tflops"]
                verdict = "WINS" if ratio > 1 else "loses"
                print(
                    f"  {bucket['name']:5} {gemm['name']:7}: f8 {f8['tflops']:7.1f} vs bf16 {bf16['tflops']:7.1f} "
                    f"= {ratio:.2f}x  (f8 {verdict})"
                )


if __name__ == "__main__":
    main()
