# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GFP8-030: does XLA fuse an f8 transpose into the producing bf16->f8 cast?

The f8 wgmma needs K-contiguous operands, forcing transposes the bf16 path avoids: the forward
weight swap and the wgrad cast-transpose (the reason f8 wgrad stays parked, GFP8-029). Those
transposes are only worth eliminating if producing the transposed f8 copy AT CAST TIME is ~free
(absorbed into the cast's output write). This microbench decides it on the real wgrad operand
shapes by timing, for x[T,D] and g[T,N]:
  (1) cast_only        : bf16 -> f8                      (the unavoidable quant write)
  (2) cast_then_T      : transpose(cast(bf16))           (cast + transposed store; fused?)
  (3) standalone_T     : transpose(precast_f8)           (the separate transpose the live path pays)
Read: if (2) ~= (1) the transpose fuses into the cast -> the cast-transpose lever is ~free and worth
plumbing; if (2) ~= (1)+(3) it does not fuse -> abandon the lever (not worth the complexity). H100-only.
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


def _time(fn, *args, steps, warmup):
    compiled = jax.jit(fn).lower(*args).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=5632)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    T, D, two_f = args.tokens, args.hidden, 2 * args.intermediate
    print(f"hardware: {[d.device_kind for d in jax.devices()]}")

    rng = np.random.default_rng(0)
    operands = [("x", (T, D)), ("dout13", (T, two_f))]  # the two wgrad-lhs / wgrad-grad shapes
    for name, shape in operands:
        x_bf = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
        x_f8 = x_bf.astype(_E4M3)

        cast_only = _time(lambda a: a.astype(_E4M3), x_bf, steps=args.steps, warmup=args.warmup)
        cast_then_t = _time(lambda a: jnp.swapaxes(a.astype(_E4M3), 0, 1), x_bf, steps=args.steps, warmup=args.warmup)
        standalone_t = _time(lambda a: jnp.swapaxes(a, 0, 1), x_f8, steps=args.steps, warmup=args.warmup)

        overhead = cast_then_t - cast_only  # extra cost of the transposed store on top of the cast
        fused = overhead < 0.5 * standalone_t  # heuristic: transpose mostly absorbed into the cast
        print(f"=== {name} {shape} ===")
        print(f"  cast_only     {cast_only*1e3:.4f}ms")
        print(f"  cast_then_T   {cast_then_t*1e3:.4f}ms  (transpose overhead on cast = {overhead*1e3:.4f}ms)")
        print(f"  standalone_T  {standalone_t*1e3:.4f}ms")
        print(f"  -> fuses_into_cast={fused}  (overhead {overhead*1e3:.4f} vs standalone {standalone_t*1e3:.4f}ms)")
        print(
            "row "
            + json.dumps(
                {
                    "operand": name,
                    "cast_only_s": cast_only,
                    "cast_then_T_s": cast_then_t,
                    "standalone_T_s": standalone_t,
                    "transpose_overhead_s": overhead,
                    "fuses": fused,
                }
            )
        )


if __name__ == "__main__":
    main()
