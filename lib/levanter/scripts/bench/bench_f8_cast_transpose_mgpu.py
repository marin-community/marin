# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GFP8-033 M2: standalone correctness + perf for the Mosaic f8 cast-transpose kernel.

Produces both f8 layouts (rowwise q[M,K] + transposed qT[K,M]) from one read of a bf16 tile. Times
it against the XLA baselines it must beat to justify the kernel:
  - xla_transpose : q = quantize(x); qt = swapaxes(q)   (the current f8-wgrad path's transpose tax)
  - xla_recast    : q = quantize(x); qt = quantize(x).T (GFP8-030's redundant re-cast — the bad one)
  - cast_floor    : q = quantize(x) only                (read-bound floor: no transpose at all)
  - mosaic[reg|ref]: the fused kernel, both transpose strategies

Bit-exactness gate: kernel q/qT must equal (quantize(x), quantize(x).T) byte-for-byte. H100-only.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
import traceback


def _ensure_cuda_toolchain() -> bool:
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

from haliax._src.fp8 import quantize  # noqa: E402
from haliax._src.fp8_cast_transpose_mgpu import cast_transpose_mgpu  # noqa: E402

_E4M3 = jnp.float8_e4m3fn
_HBM_TBPS = 3.35  # H100 SXM HBM3 peak


def _time(fn, *args, steps, warmup):
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / steps


def _bytes(m, k):
    # one bf16 read + two f8 writes (the ideal fused traffic)
    return m * k * 2 + 2 * m * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--block-m", type=int, default=128)
    ap.add_argument("--block-k", type=int, default=128)
    args = ap.parse_args()
    print(f"hardware: {[d.device_kind for d in jax.devices()]}")
    rng = np.random.default_rng(0)
    scale = jnp.full(1, 0.5, jnp.float32)

    # Real wgrad cast-transpose operands (M=tokens=8192; K in {D, 2F, F}).
    shapes = [("act_D", 8192, 2048), ("grad_2F", 8192, 11264), ("act_F", 8192, 5632)]

    for name, m, k in shapes:
        x = jnp.asarray(rng.standard_normal((m, k)) * 3.0, jnp.bfloat16)
        q_ref = quantize(x, _E4M3, scale, jnp.bfloat16)
        qt_ref = q_ref.T
        floor_ms = _bytes(m, k) / (_HBM_TBPS * 1e12) * 1e3

        print(f"=== {name} [{m},{k}]  read-bound floor {floor_ms:.4f}ms ===")

        # Correctness + timing for the fused kernel (register-layout-cast transpose), both src layouts.
        for strat in ("f32t", "f8t"):
            try:
                fn = lambda xx: cast_transpose_mgpu(
                    xx, scale, block_m=args.block_m, block_k=args.block_k, transpose_dtype=strat
                )
                q, qt = jax.block_until_ready(jax.jit(fn)(x))
                ok_q = np.array_equal(np.asarray(q).view(np.uint8), np.asarray(q_ref).view(np.uint8))
                ok_qt = np.array_equal(np.asarray(qt).view(np.uint8), np.asarray(qt_ref).view(np.uint8))
                t = _time(fn, x, steps=args.steps, warmup=args.warmup)
                print(
                    f"  mosaic[{strat}] {t*1e3:.4f}ms  ({floor_ms/(t*1e3)*100:.0f}% of floor)  "
                    f"q_exact={ok_q} qt_exact={ok_qt}"
                )
                print(
                    "row "
                    + json.dumps(
                        {
                            "case": name,
                            "impl": f"mosaic_{strat}",
                            "ms": t * 1e3,
                            "q_exact": bool(ok_q),
                            "qt_exact": bool(ok_qt),
                        }
                    )
                )
            except Exception as e:
                print(f"  mosaic[{strat}] FAILED: {type(e).__name__}")
                print("\n".join("    " + ln for ln in traceback.format_exc().splitlines()[-14:]))

        # Baselines. The real tax the kernel removes is the f8->f8 swapaxes of an ALREADY-quantized
        # operand (what _mosaic_pallas_call does today), not a re-quantize. Measure all three.
        t_floor = _time(lambda xx: quantize(xx, _E4M3, scale, jnp.bfloat16), x, steps=args.steps, warmup=args.warmup)
        t_f8tr = _time(lambda qq: jnp.swapaxes(qq, 0, 1), q_ref, steps=args.steps, warmup=args.warmup)
        t_sep = _time(
            lambda xx: (
                quantize(xx, _E4M3, scale, jnp.bfloat16),
                jnp.swapaxes(quantize(xx, _E4M3, scale, jnp.bfloat16), 0, 1),
            ),
            x,
            steps=args.steps,
            warmup=args.warmup,
        )
        print(f"  cast_floor     {t_floor*1e3:.4f}ms  (quantize only)")
        print(f"  f8_swapaxes    {t_f8tr*1e3:.4f}ms  (transpose of already-f8 operand — the current tax)")
        print(f"  xla_separate   {t_sep*1e3:.4f}ms  (quantize + swapaxes(quantize), CSE-dependent)")
        print("row " + json.dumps({"case": name, "impl": "cast_floor", "ms": t_floor * 1e3}))
        print("row " + json.dumps({"case": name, "impl": "f8_swapaxes", "ms": t_f8tr * 1e3}))
        print("row " + json.dumps({"case": name, "impl": "xla_separate", "ms": t_sep * 1e3}))


if __name__ == "__main__":
    main()
