# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GFP8-032: what is the practical f8 "theoretical maximum" on H100?

The goal targets "within 20% of theoretical maximum" (= 80% of the 1978.9 TF/s dense f8 peak). The
grouped/ragged MoE GEMMs sit at 30-47% of that peak. This bench establishes the practical ceiling by
timing a DENSE f8 GEMM of the same FLOPs (no ragged-group overhead) on the two real f8 paths:
  - cuBLAS dense f8 : jax.lax.dot_general on f8 operands (the vendor library ceiling)
  - dense bf16      : jax.lax.dot_general bf16 (for the dtype reference)
Read: if even cuBLAS dense f8 lands well below 80% of peak, then "within 20% of dense peak" is above the
practical f8 ceiling on H100 for these shapes — not a grouped-kernel-quality gap. H100-only.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time


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

_E4M3 = jnp.float8_e4m3fn
_F8_PEAK = 1978.9
_BF16_PEAK = 989.4


def _time(fn, *args, steps, warmup):
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / steps


def _dot_f8(a, b):
    # f8 x f8 -> f32 accumulate (contract a's axis 1 with b's axis 0); cuBLAS lt f8 GEMM.
    return jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    print(f"hardware: {[d.device_kind for d in jax.devices()]}")
    rng = np.random.default_rng(0)

    # Dense GEMMs at the same FLOPs as the grouped MoE GEMMs (M = all tokens, fused expert dim).
    # fwd13: [8192,2048]x[2048,11264]; fwd2: [8192,5632]x[5632,2048]; dgrad13: [8192,11264]x[11264,2048].
    cases = [
        ("dense_fwd13", (8192, 2048), (2048, 11264)),
        ("dense_fwd2", (8192, 5632), (5632, 2048)),
        ("dense_dgrad13", (8192, 11264), (11264, 2048)),
    ]
    for name, ashape, bshape in cases:
        m, k = ashape
        n = bshape[1]
        flops = 2.0 * m * k * n
        a8 = jnp.asarray(rng.standard_normal(ashape), _E4M3)
        b8 = jnp.asarray(rng.standard_normal(bshape), _E4M3)
        a16, b16 = a8.astype(jnp.bfloat16), b8.astype(jnp.bfloat16)
        t_f8 = _time(_dot_f8, a8, b8, steps=args.steps, warmup=args.warmup)
        t_bf16 = _time(
            lambda x, y: jnp.dot(x, y, preferred_element_type=jnp.float32),
            a16,
            b16,
            steps=args.steps,
            warmup=args.warmup,
        )
        tf_f8 = flops / t_f8 / 1e12
        tf_bf16 = flops / t_bf16 / 1e12
        print(f"=== {name} [{m},{k}]x[{k},{n}] ===")
        print(f"  cublas_f8   {t_f8*1e3:.4f}ms  {tf_f8:.0f} TF/s  = {tf_f8/_F8_PEAK*100:.0f}% of f8 peak")
        print(f"  cublas_bf16 {t_bf16*1e3:.4f}ms  {tf_bf16:.0f} TF/s = {tf_bf16/_BF16_PEAK*100:.0f}% of bf16 peak")
        print(
            "row "
            + json.dumps(
                {
                    "case": name,
                    "f8_tflops": tf_f8,
                    "f8_pct_peak": tf_f8 / _F8_PEAK,
                    "bf16_tflops": tf_bf16,
                    "bf16_pct_peak": tf_bf16 / _BF16_PEAK,
                }
            )
        )


if __name__ == "__main__":
    main()
