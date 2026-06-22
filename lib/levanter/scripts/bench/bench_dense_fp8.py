# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense matmul microbench for the Grug FP8-on-H100 spike.

Benchmarks a single dense projection ``[M, K] @ [K, N]`` (forward + backward),
emitting steady-state timing, the compiled HLO, and one machine-readable
``result_json`` line. This is the BF16 baseline arm of the linear-FP8 harness;
FP8 lowering paths land on top of the same rig (task S2).

The default shape is a hidden-dim-3072 projection at seq 4096; sweep ``--k``
over 3072/4096 for the two benchmark dims. Run on an H100 via Iris and read the
HLO back through ``iris job logs`` (no storage needed):

    uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 \\
        --enable-extra-resources --extra gpu -- \\
        python lib/levanter/scripts/bench/bench_dense_fp8.py --k 4096
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
from jax import lax

# Contract K (lhs dim 1, rhs dim 0); no batch dims. Shared with the FP8 paths
# so the lowering differs only by dtype/precision, not the einsum.
_DIMENSION_NUMBERS = (((1,), (0,)), ((), ()))

# H100 SXM dense (non-sparse) BF16 matmul peak.
_H100_SXM_BF16_TFLOPS_PER_S = 989.5e12


def _bf16_peak_tflops_per_s() -> float | None:
    """Reference dense BF16 peak for the local device, or None if unrecognized."""
    if not jax.devices():
        return None
    if "h100" in jax.devices()[0].device_kind.lower():
        return _H100_SXM_BF16_TFLOPS_PER_S
    return None


def dense_dot(x: jax.Array, w: jax.Array) -> jax.Array:
    """Dense projection ``[M, K] @ [K, N] -> [M, N]``."""
    return lax.dot_general(x, w, _DIMENSION_NUMBERS)


def _configure_xla_dump_dir(xla_dump_dir: str) -> str:
    """Direct XLA HLO text dumps to ``xla_dump_dir``. Must run before backend init."""
    resolved = os.path.abspath(xla_dump_dir)
    os.makedirs(resolved, exist_ok=True)
    flags = os.environ.get("XLA_FLAGS", "").split()
    for flag in (f"--xla_dump_to={resolved}", "--xla_dump_hlo_as_text"):
        if flag not in flags:
            flags.append(flag)
    os.environ["XLA_FLAGS"] = " ".join(flags)
    return resolved


def _time_jitted(fn, *args, steps: int, warmup: int) -> tuple[float, float]:
    """Return (compile_time, mean steady-state time) in seconds."""
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))

    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(fn(*args))
    return compile_time, (time.perf_counter() - start) / steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096, help="rows / tokens")
    parser.add_argument("--k", type=int, default=3072, help="contracting / hidden dim")
    parser.add_argument("--n", type=int, default=3072, help="output dim")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--no-print-hlo", dest="print_hlo", action="store_false")
    parser.add_argument(
        "--xla-dump-dir", type=str, default=None, help="dir for XLA HLO text dumps (upload to R2 to retrieve)"
    )
    parser.add_argument(
        "--profiler-dir", type=str, default=None, help="dir for a jax.profiler trace of the timed runs"
    )
    args = parser.parse_args()

    # Dump flags must be set before the backend initializes (first JAX op below).
    xla_dump_dir = _configure_xla_dump_dir(args.xla_dump_dir) if args.xla_dump_dir else None

    dtype = jnp.dtype(args.dtype)
    print("devices:", jax.devices())

    key_x, key_w, key_c = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(key_x, (args.m, args.k), dtype=dtype)
    w = jax.random.normal(key_w, (args.k, args.n), dtype=dtype)
    # Backward through a random output cotangent, not a sum-loss's all-ones one:
    # a constant cotangent lets XLA fold an FP8 output-grad QDQ (dequant(quant(1.0))
    # is identity), which would stop a backward f8 matmul from ever lowering. The
    # scalar loss <out, cotangent> gives d(loss)/d(out) = cotangent.
    cotangent = jax.random.normal(key_c, (args.m, args.n), dtype=dtype)

    fwd = jax.jit(dense_dot)
    grad = jax.jit(jax.grad(lambda a, b: jnp.sum(dense_dot(a, b) * cotangent), argnums=(0, 1)))

    if args.print_hlo:
        print("=== forward HLO ===")
        print(fwd.lower(x, w).compile().as_text())

    # Backward runs both grad matmuls (dx, dw), so 2x the forward flop count.
    peak = _bf16_peak_tflops_per_s()
    fwd_flops = 2.0 * args.m * args.k * args.n

    fwd_compile, fwd_steady = _time_jitted(fwd, x, w, steps=args.steps, warmup=args.warmup)
    result = {
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "dtype": str(dtype),
        "fwd_compile_time_s": fwd_compile,
        "fwd_steady_time_s": fwd_steady,
        "fwd_tflops_per_s": fwd_flops / fwd_steady / 1e12,
    }
    if peak is not None:
        result["fwd_pct_bf16_peak"] = 100.0 * fwd_flops / fwd_steady / peak
    if not args.forward_only:
        bwd_compile, bwd_steady = _time_jitted(grad, x, w, steps=args.steps, warmup=args.warmup)
        result["bwd_compile_time_s"] = bwd_compile
        result["bwd_steady_time_s"] = bwd_steady
        result["bwd_tflops_per_s"] = 2.0 * fwd_flops / bwd_steady / 1e12
        if peak is not None:
            result["bwd_pct_bf16_peak"] = 100.0 * 2.0 * fwd_flops / bwd_steady / peak

    if args.profiler_dir:
        profiler_dir = os.path.abspath(args.profiler_dir)
        with jax.profiler.trace(profiler_dir):
            jax.block_until_ready(fwd(x, w))
            if not args.forward_only:
                jax.block_until_ready(grad(x, w))
        result["profiler_dir"] = profiler_dir
    if xla_dump_dir is not None:
        result["xla_dump_dir"] = xla_dump_dir

    print("result_json", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
