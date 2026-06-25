# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GFP8-028 M2 diagnostic: attribute the f8-wgrad regression to transpose vs kernel.

f8 wgrad (cast-transpose) was correct but ~0.62ms slower e2e than the bf16 wgrad the hybrid uses.
This splits that per real wgrad GEMM into:
  (1) transpose-only : materialize the two cast-transposed f8 operands (XLA f8 transpose)
  (2) kernel-only    : our f8 transposed_ragged_dot on PRE-transposed operands (no transpose)
  (3) f8 full        : transpose + kernel (what the live path pays)
  (4) bf16 ref       : the Triton ragged-contracting wgrad the hybrid actually runs (_DRHS, "auto")

Read: if (2) <= (4) the kernel is competitive and the loss is the transpose -> fusing it (M2) can win;
if (2) > (4) the kernel itself loses and no transpose-fusing recovers it -> stop. H100-only.
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
    """Make ptxas + libdevice discoverable to XLA/Mosaic before jax import (GFP8-022)."""
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

from haliax.nn.ragged_dot import _DRHS_DIM_NUMS, _triton_pallas_call  # noqa: E402
from haliax._src.transposed_ragged_dot_mgpu import transposed_ragged_dot, WgradBlockConfig  # noqa: E402

_E4M3 = jnp.float8_e4m3fn


def _time_call(fn, *args, steps, warmup):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=5632)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    T, D, F, E = args.tokens, args.hidden, args.intermediate, args.experts
    two_f = 2 * F
    cfg = WgradBlockConfig()
    print(f"hardware: {[d.device_kind for d in jax.devices()]}  config={cfg}")

    rng = np.random.default_rng(0)
    gs = jnp.asarray(rng.multinomial(T, np.ones(E) / E), jnp.int32)

    def f8(shape):
        return jnp.asarray(rng.standard_normal(shape), _E4M3)

    # (label, lhs[T,K], grad[T,N])  — wgrad contracts tokens T; output [E,K,N].
    gemms = [
        ("wgrad13", f8((T, D)), f8((T, two_f))),  # dW13[E,D,2F] = x^T @ dout13
        ("wgrad2", f8((T, F)), f8((T, D))),  # dW2[E,F,D] = g^T @ dout2
    ]

    for label, lhs, grad in gemms:
        lhs_bf, grad_bf = lhs.astype(jnp.bfloat16), grad.astype(jnp.bfloat16)
        lhs_t, grad_t = jnp.swapaxes(lhs, 0, 1), jnp.swapaxes(grad, 0, 1)  # pre-transposed f8 operands

        def transpose_only(a, b):
            return jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)

        def kernel_only(at, bt):
            return transposed_ragged_dot(at, bt, gs, out_dtype=jnp.bfloat16, config=cfg)

        def f8_full(a, b):
            return transposed_ragged_dot(
                jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1), gs, out_dtype=jnp.bfloat16, config=cfg
            )

        def bf16_ref(a, b):
            return _triton_pallas_call(a, b, gs, _DRHS_DIM_NUMS, out_dtype=jnp.bfloat16)

        rows = {}
        for name, fn, fargs in [
            ("transpose_only", transpose_only, (lhs, grad)),
            ("kernel_only", kernel_only, (lhs_t, grad_t)),
            ("f8_full", f8_full, (lhs, grad)),
            ("bf16_ref", bf16_ref, (lhs_bf, grad_bf)),
        ]:
            try:
                _, steady = _time_call(fn, *fargs, steps=args.steps, warmup=args.warmup)
                rows[name] = steady
            except Exception as exc:  # noqa: BLE001
                rows[name] = None
                print(f"  {label} {name}: FAILED {type(exc).__name__}: {str(exc)[:160]}")

        def ms(x):
            return f"{x * 1e3:.4f}ms" if x is not None else "FAIL"

        print(f"=== {label} (K-contract over T={T}; out [E,K,N]) ===")
        for n in ("transpose_only", "kernel_only", "f8_full", "bf16_ref"):
            print(f"  {n:16} {ms(rows[n])}")
        if rows["kernel_only"] and rows["bf16_ref"]:
            print(f"  -> kernel_only/bf16_ref = {rows['kernel_only'] / rows['bf16_ref']:.2f}x "
                  f"(<1 means f8 kernel beats bf16 ref; transpose headroom = {ms(rows['transpose_only'])})")
        print("row " + json.dumps({"gemm": label, **{k: rows[k] for k in rows}}))


if __name__ == "__main__":
    main()
