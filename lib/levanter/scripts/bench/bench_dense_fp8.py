# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense matmul microbench for the Grug FP8-on-H100 spike.

Benchmarks a single dense projection ``[M, K] @ [K, N]`` (forward + backward),
emitting steady-state timing, the compiled HLO, and one machine-readable
``result_json`` line. ``--path bf16`` (default) is the baseline arm; ``--path
qdq`` is S2's first FP8 arm — the existing haliax ``Fp8DotGeneralOp`` delayed-
scaling path, re-measured to see whether XLA still fuses it to an f8 cuBLASLt
matmul or silently falls back to BF16. ``--path manual`` (forward-only) is the
direct-f8 arm: genuine E4M3 operands straight into the dot, dequant on the
output — the candidate fix for the forward QDQ fallback.

The default shape is a hidden-dim-3072 projection at seq 4096; sweep ``--k``
over 3072/4096 for the two benchmark dims. Run on an H100 via Iris and grep the
HLO back through ``iris job logs`` (no storage needed) for ``__cublas$lt$matmul$f8``:

    uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 \\
        --enable-extra-resources --extra gpu -- \\
        python lib/levanter/scripts/bench/bench_dense_fp8.py --k 4096 --path qdq
"""

import argparse
import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
from jax import lax

from haliax._src.fp8 import compute_scale, get_fp8_max, quantize
from haliax.quantization import Fp8DotGeneralOp

# Contract K (lhs dim 1, rhs dim 0); no batch dims. Shared with the FP8 paths
# so the lowering differs only by dtype/precision, not the einsum.
_DIMENSION_NUMBERS = (((1,), (0,)), ((), ()))

_E4M3 = jnp.float8_e4m3fn

# H100 SXM dense (non-sparse) matmul peaks; FP8 (E4M3) is 2x the BF16 rate.
_H100_SXM_BF16_TFLOPS_PER_S = 989.5e12
_H100_SXM_FP8_TFLOPS_PER_S = 1978.9e12


def _peak_tflops_per_s(is_fp8: bool) -> float | None:
    """Reference dense matmul peak for the local device, or None if unrecognized."""
    if not jax.devices():
        return None
    if "h100" in jax.devices()[0].device_kind.lower():
        return _H100_SXM_FP8_TFLOPS_PER_S if is_fp8 else _H100_SXM_BF16_TFLOPS_PER_S
    return None


def dense_dot(x: jax.Array, w: jax.Array) -> jax.Array:
    """Dense projection ``[M, K] @ [K, N] -> [M, N]``."""
    return lax.dot_general(x, w, _DIMENSION_NUMBERS)


def _delayed_scale(amax_history: jax.Array, scale: jax.Array) -> jax.Array:
    """Live per-tensor E4M3 scale from a delayed-scaling amax window (mirrors in_qdq)."""
    return compute_scale(jnp.max(amax_history), scale, get_fp8_max(_E4M3, jnp.float32))


def manual_fp8_dot(x: jax.Array, w: jax.Array, op: Fp8DotGeneralOp) -> jax.Array:
    """Direct-f8 forward: dot genuine E4M3 operands, dequantize only the output.

    Unlike the QDQ path — which dequantizes each operand back to the compute dtype
    *before* the dot, leaving an f8->bf16 round-trip that ``simplify-fp-conversions``
    strips on the forward — here the E4M3 operands flow straight into ``dot_general``
    and only the f32 accumulator is rescaled by ``x_scale * w_scale``. This is the
    pattern XLA's FP8 GemmRewriter is meant to fuse into ``__cublas$lt$matmul$f8``.

    Scales are computed live from ``op``'s delayed-scaling histories exactly as
    ``in_qdq`` does, so this is apples-to-apples with ``--path qdq``.
    """
    comp_dtype = w.dtype
    x_scale = _delayed_scale(op.input_amax_history, op.input_scale)
    w_scale = _delayed_scale(op.kernel_amax_history, op.kernel_scale)
    qx = quantize(x, _E4M3, x_scale, comp_dtype)
    qw = quantize(w, _E4M3, w_scale, comp_dtype)
    acc = lax.dot_general(qx, qw, _DIMENSION_NUMBERS, preferred_element_type=jnp.float32)
    return (acc * (x_scale * w_scale).astype(jnp.float32)).astype(comp_dtype)


def build_dot(path: str, amax_history_length: int):
    """Return ``(dot_fn, op)`` for the requested lowering path.

    ``qdq`` calls the existing haliax ``Fp8DotGeneralOp`` verbatim: per-tensor
    delayed-scaling QDQ (E4M3 fwd weights/acts, E5M2 output grad) around a
    DEFAULT-precision dot, relying on XLA's GemmRewriter to fuse the f8 cuBLASLt
    matmul. The op is *returned*, not closed over: ``dot_fn(x, w, op)`` takes the
    state as a runtime argument so the per-tensor scales stay live operands. If
    the op is closed into the jit instead, XLA constant-folds ``compute_scale``
    over the (constant) amax history and bakes scale=const into the f8 call —
    which is the real-training delayed-scaling case faked into a trivially-easier
    one for the rewriter. ``op`` is ``None`` for bf16.

    ``manual`` is S2's direct-f8 arm (forward-only): genuine E4M3 operands into the
    dot, dequant on the output (see ``manual_fp8_dot``). It reuses the same ``op``
    state so the scales are the same live runtime operands as ``qdq``.
    """
    if path == "bf16":
        return (lambda x, w: dense_dot(x, w)), None
    op = Fp8DotGeneralOp.init(amax_history_length=amax_history_length)
    if path == "manual":
        return (lambda x, w, op: manual_fp8_dot(x, w, op)), op
    return (lambda x, w, op: op(x, w, _DIMENSION_NUMBERS)), op


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
    parser.add_argument("--dtype", type=str, default="bfloat16", help="operand dtype (QDQ quantizes internally)")
    parser.add_argument(
        "--path",
        choices=("bf16", "qdq", "manual"),
        default="bf16",
        help="bf16 baseline; qdq = existing haliax Fp8DotGeneralOp; manual = direct-f8 dot (forward-only)",
    )
    parser.add_argument("--amax-history-length", type=int, default=1024, help="delayed-scaling amax window (qdq)")
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

    # The manual arm exists to answer the open *forward* question (does a direct-f8
    # dot fire $f8 where QDQ falls back?); its backward is not the e5m2 grad path.
    if args.path == "manual" and not args.forward_only:
        parser.error("--path manual is forward-only; pass --forward-only")

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

    dot, op = build_dot(args.path, args.amax_history_length)
    if op is not None:
        # Seed the delayed-scaling histories from the actual tensors so the scales
        # are realistic non-unit *runtime* operands (a warmed-up delayed-scaling
        # state), then pass the op as a jit arg. Zeros + closed-over would collapse
        # the scale to a baked-in constant 1.0 (see build_dot).
        length = args.amax_history_length
        op = dataclasses.replace(
            op,
            input_amax_history=jnp.full((length,), jnp.max(jnp.abs(x)).astype(jnp.float32)),
            kernel_amax_history=jnp.full((length,), jnp.max(jnp.abs(w)).astype(jnp.float32)),
            output_grad_amax_history=jnp.full((length,), jnp.max(jnp.abs(cotangent)).astype(jnp.float32)),
        )
    extra = () if op is None else (op,)
    fwd = jax.jit(dot)
    grad = jax.jit(jax.grad(lambda *a: jnp.sum(dot(*a) * cotangent), argnums=(0, 1)))

    if args.print_hlo:
        print("=== forward HLO ===")
        print(fwd.lower(x, w, *extra).compile().as_text())
        # The backward holds the dx/dw matmuls (and, for FP8, the E5M2 output-grad
        # quant) — grep this section separately for __cublas$lt$matmul$f8.
        if not args.forward_only:
            print("=== backward HLO ===")
            print(grad.lower(x, w, *extra).compile().as_text())

    # Backward runs both grad matmuls (dx, dw), so 2x the forward flop count.
    peak = _peak_tflops_per_s(is_fp8=args.path != "bf16")
    fwd_flops = 2.0 * args.m * args.k * args.n

    fwd_compile, fwd_steady = _time_jitted(fwd, x, w, *extra, steps=args.steps, warmup=args.warmup)
    result = {
        "path": args.path,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "dtype": str(dtype),
        "fwd_compile_time_s": fwd_compile,
        "fwd_steady_time_s": fwd_steady,
        "fwd_tflops_per_s": fwd_flops / fwd_steady / 1e12,
    }
    if peak is not None:
        result["peak_tflops_per_s"] = peak / 1e12
        result["fwd_pct_peak"] = 100.0 * fwd_flops / fwd_steady / peak
    if not args.forward_only:
        bwd_compile, bwd_steady = _time_jitted(grad, x, w, *extra, steps=args.steps, warmup=args.warmup)
        result["bwd_compile_time_s"] = bwd_compile
        result["bwd_steady_time_s"] = bwd_steady
        result["bwd_tflops_per_s"] = 2.0 * fwd_flops / bwd_steady / 1e12
        if peak is not None:
            result["bwd_pct_peak"] = 100.0 * 2.0 * fwd_flops / bwd_steady / peak

    if args.profiler_dir:
        profiler_dir = os.path.abspath(args.profiler_dir)
        with jax.profiler.trace(profiler_dir):
            jax.block_until_ready(fwd(x, w, *extra))
            if not args.forward_only:
                jax.block_until_ready(grad(x, w, *extra))
        result["profiler_dir"] = profiler_dir
    if xla_dump_dir is not None:
        result["xla_dump_dir"] = xla_dump_dir

    print("result_json", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
