# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""M0 step 2: end-to-end fwd+bwd speed of the Mosaic-GPU FP8 grouped GEMM family.

The forward win (GFP8-022) is per-GEMM forward only. A real fwd+bwd verdict needs the two
BACKWARD grouped GEMMs too, which have different shapes and map to two stock Mosaic kernels:

  fwd   out[M,N]   = ragged_dot(X[M,K], W[G,K,N])                 -> ragged_dot_mgpu
  dgrad dX[M,K]    = dY[M,N] @ W[G,K,N]^T   (contract N)          -> ragged_dot_mgpu
  wgrad dW[G,K,N]  = X[M,K]^T @ dY[M,N]      (contract ragged M)  -> transposed_ragged_dot_mgpu

Hopper f8 wgmma forbids operand transposes, so each operand must present its CONTRACTION axis
contiguous. fwd/dgrad satisfy this via ragged_dot_mgpu's transpose_rhs=True with the right RHS
layout (fwd: W as (G,N,K); dgrad: natural (G,K,N)). The wgrad kernel transpose_ref's its lhs
internally -- whether that is f8-legal is the open H100 question this bench answers.

All-E4M3 (M0 recipe): every GEMM is e4m3 x e4m3 (same-type), so no emitter fork. Each GEMM is
validated against lax.ragged_dot_general (coarse f8 tol) BEFORE timing, so a wrong layout or an
unsupported f8 path is reported per-GEMM rather than producing a bogus number. bf16 is the bar.

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
from jax import lax  # noqa: E402
import _transposed_ragged_dot_f8 as transposed_ragged_dot_mgpu  # noqa: E402  (vendored + f8 mask patch)
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu  # noqa: E402

_HIDDEN = 1024
_INTERMEDIATE = 512
_TOKENS = 65536
_BUCKETS = [{"name": "ep8", "experts": 8}, {"name": "dp64", "experts": 64}]
# Per expert GEMM, the (K, N) of the FORWARD; the backward shapes derive from it.
_GEMMS = [{"name": "gateup", "k": _HIDDEN, "n": 2 * _INTERMEDIATE}, {"name": "down", "k": _INTERMEDIATE, "n": _HIDDEN}]
_DTYPES = [("bf16", jnp.bfloat16), ("f8e4m3", jnp.float8_e4m3fn)]

# (block_m, block_n, block_k, max_concurrent_steps, grid_block_n)
_CONFIGS = list(itertools.product((64, 128), (128, 256), (64, 128), (2, 4), (1, 2)))

# lax reference dimension numbers per pass.
_FWD_DN = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,)
)
_DGRAD_DN = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (2,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=(0,)
)
_WGRAD_DN = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())), lhs_ragged_dimensions=(0,), rhs_group_dimensions=[]
)


def _group_sizes(tokens, experts, seed=0):
    counts = np.random.default_rng(seed).multinomial(tokens, np.ones(experts) / experts)
    return jnp.asarray(counts, jnp.int32)


def _rand(shape, dtype, scale=1.0, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape) * scale, dtype)


def _mosaic_call(pass_name, a, b, gs, cfg):
    """Dispatch one pass to the right stock Mosaic kernel + operand layout."""
    bm, bn, bk, steps, gbn = cfg
    rd = functools.partial(
        ragged_dot_mgpu.ragged_dot,
        group_sizes=gs,
        block_m=bm,
        block_n=bn,
        block_k=bk,
        max_concurrent_steps=steps,
        grid_block_n=gbn,
    )
    if pass_name == "fwd":  # a=X[M,K], b=W[G,K,N] -> [M,N]; feed W as (G,N,K), transpose_rhs
        return rd(a, jnp.swapaxes(b, 1, 2), transpose_rhs=True)
    if pass_name == "dgrad":  # a=dY[M,N], b=W[G,K,N] -> dX[M,K] (contract N); natural W, transpose_rhs
        return rd(a, b, transpose_rhs=True)
    if pass_name == "wgrad":  # a=X[M,K], b=dY[M,N] -> dW[G,K,N] (contract ragged M)
        return transposed_ragged_dot_mgpu.transposed_ragged_dot(
            a, b, group_sizes=gs, block_m=bm, block_n=bn, block_k=bk, max_concurrent_steps=steps, grid_block_n=gbn
        )
    raise ValueError(pass_name)


def _ref(pass_name, a, b, gs):
    dn = {"fwd": _FWD_DN, "dgrad": _DGRAD_DN, "wgrad": _WGRAD_DN}[pass_name]
    return lax.ragged_dot_general(a, b, gs, ragged_dot_dimension_numbers=dn, preferred_element_type=jnp.float32)


def _time_ms(f, a, b, steps=15, warmup=4):
    c = jax.jit(f).lower(a, b).compile()
    for _ in range(warmup):
        jax.block_until_ready(c(a, b))
    t = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(c(a, b))
    return (time.perf_counter() - t) / steps * 1e3


def _flops(pass_name, m, k, n, experts):
    # Every pass contracts a dim of size {K or N or M}; FLOPs = 2*M*K*N regardless of layout.
    return 2 * m * k * n


def _autotune(pass_name, a, b, gs, flops):
    best = {"ms": float("inf"), "cfg": None, "ok": 0, "err": 0, "sample": None}
    ref = np.asarray(_ref(pass_name, a, b, gs), np.float32)
    for cfg in _CONFIGS:
        try:
            out = np.asarray(_mosaic_call(pass_name, a, b, gs, cfg), np.float32)
        except Exception as e:  # noqa: BLE001
            best["err"] += 1
            if best["sample"] is None:
                best["sample"] = str(e).replace("\n", " ")[:160]
            continue
        relerr = float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-30))
        if relerr > 0.25:  # coarse f8 guard: a wrong layout shows up as a large error
            best["err"] += 1
            best["sample"] = best["sample"] or f"relerr={relerr:.3f} (layout?)"
            continue
        ms = _time_ms(lambda x, y: _mosaic_call(pass_name, x, y, gs, cfg), a, b)
        best["ok"] += 1
        if ms < best["ms"]:
            best.update(ms=ms, cfg=cfg, relerr=relerr)
    if best["cfg"] is not None:
        best["tflops"] = flops / (best["ms"] / 1e3) / 1e12
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    print(f"hardware: {[d.device_kind for d in jax.devices()]}  configs: {len(_CONFIGS)}")
    records = []
    for bucket in _BUCKETS:
        E = bucket["experts"]
        gs = _group_sizes(_TOKENS, E)
        for gemm in _GEMMS:
            K, N = gemm["k"], gemm["n"]
            for dname, dt in _DTYPES:
                X = _rand((_TOKENS, K), dt, 1.0, 1)
                W = _rand((E, K, N), dt, 0.08, 2)
                dY = _rand((_TOKENS, N), dt, 1.0, 3)
                passes = {
                    "fwd": (X, W, _flops("fwd", _TOKENS, K, N, E)),
                    "dgrad": (dY, W, _flops("dgrad", _TOKENS, K, N, E)),
                    "wgrad": (X, dY, _flops("wgrad", _TOKENS, K, N, E)),
                }
                for pname, (a, b, fl) in passes.items():
                    r = _autotune(pname, a, b, gs, fl)
                    rec = {"bucket": bucket["name"], "gemm": gemm["name"], "dtype": dname, "pass": pname, **r}
                    records.append(rec)
                    if r["cfg"] is not None:
                        tag = f"{r['tflops']:7.1f} TF/s  relerr={r['relerr']:.3g}  cfg={r['cfg']}"
                    else:
                        tag = f"NO VIABLE (err={r['err']}) :: {r['sample']}"
                    print(f"  {bucket['name']:5} {gemm['name']:7} {dname:7} {pname:6} -> {tag}")

    if args.artifact:
        with open(args.artifact, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"artifact: {args.artifact}")

    # fwd+bwd verdict per (bucket, gemm): sum the three passes' time, f8 vs bf16.
    print("\n===== fwd+bwd (sum of fwd+dgrad+wgrad ms) f8 vs bf16 =====")
    by = {(r["bucket"], r["gemm"], r["dtype"], r["pass"]): r for r in records}
    for bucket in _BUCKETS:
        for gemm in _GEMMS:
            tots = {}
            for dname, _ in _DTYPES:
                ms = [
                    by.get((bucket["name"], gemm["name"], dname, p), {}).get("ms") for p in ("fwd", "dgrad", "wgrad")
                ]
                tots[dname] = sum(ms) if all(m is not None for m in ms) else None
            b16, f8 = tots["bf16"], tots["f8e4m3"]
            if b16 and f8:
                print(
                    f"  {bucket['name']:5} {gemm['name']:7}: f8 {f8 * 1e3:7.1f}us vs bf16 {b16 * 1e3:7.1f}us = {b16 / f8:.2f}x (f8 {'WINS' if f8 < b16 else 'loses'})"
                )
            else:
                print(f"  {bucket['name']:5} {gemm['name']:7}: incomplete (bf16={b16}, f8={f8})")


if __name__ == "__main__":
    main()
