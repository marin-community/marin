# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Grouped (ragged) MoE expert-GEMM microbench for the Grug FP8-on-H100 spike (S5).

Exercises the Grug MoE expert MLP compute — the two grouped GEMMs and the gated
activation between them — comparing BF16 against per-tensor delayed-scaling FP8:

    h          = ragged_dot(x_dispatch[T,D], w13[E,D,2F])   -> [T, 2F]
    gate, up   = split(h)
    out        = ragged_dot(silu(gate) * up [T,F], w2[E,F,D]) -> [T, D]

The token dispatch/combine (gather + scatter-add) around this is pure data movement
and dtype-irrelevant, so it is intentionally excluded; the f8 question lives entirely
in the two grouped GEMMs. ``--path fp8`` routes both GEMMs through
:class:`haliax.quantization.Fp8RaggedDotOp` (the ragged analog of
``Fp8DirectDotGeneralOp``): E4M3 operands into the grouped GEMM, E5M2 output grad in
the backward, dequant on the output.

Emits steady-state fwd / fwd+bwd timing, the compiled HLO, an f8-instruction scan, and
one machine-readable ``result_json`` line. The HLO scan reports the dtypes of operands
flowing into the grouped-GEMM call (Triton custom-call on GPU, ``gemm_fusion``/ragged
loop on the XLA path) — the correct f8 signal here, since at these shapes XLA emits
``gemm_fusion_dot kind=kCustom`` rather than ``__cublas$lt$matmul$f8`` (logbook S5 probe).

Default shape is the hidden-2048 / intermediate-5632 / 8-expert Grug MoE at ~8192
dispatched tokens (seq 4096, top-2). Run on an H100 via Iris and grep the logs:

    uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 \\
        --enable-extra-resources --extra gpu -- \\
        python lib/levanter/scripts/bench/bench_ragged_fp8.py --path fp8 --implementation triton
"""

import argparse
import json
import os
import re
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from haliax.nn.ragged_dot import Implementation, ragged_dot
from haliax.quantization import Fp8RaggedDotOp

# Triton kernel tuning knobs (env-driven; see haliax.nn.ragged_dot). Echoed into
# result_json so each swept arm is self-attributing.
_TUNE_ENV_KEYS = (
    "RAGGED_DOT_BLOCK_M",
    "RAGGED_DOT_BLOCK_K",
    "RAGGED_DOT_BLOCK_N",
    "RAGGED_DOT_NUM_WARPS",
    "RAGGED_DOT_NUM_STAGES",
    "RAGGED_DOT_F8_COMPUTE",
)

# H100 SXM dense matmul peaks; FP8 (E4M3) is 2x the BF16 rate.
_H100_SXM_BF16_TFLOPS_PER_S = 989.5e12
_H100_SXM_FP8_TFLOPS_PER_S = 1978.9e12


def _peak_tflops_per_s(is_fp8: bool) -> float | None:
    if not jax.devices():
        return None
    if "h100" in jax.devices()[0].device_kind.lower():
        return _H100_SXM_FP8_TFLOPS_PER_S if is_fp8 else _H100_SXM_BF16_TFLOPS_PER_S
    return None


RaggedDot = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def _expert_mlp(
    x: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    group_sizes: jax.Array,
    *,
    dot13: RaggedDot,
    dot2: RaggedDot,
) -> jax.Array:
    """Grug MoE expert MLP: gated up-projection, SiLU, down-projection, all grouped."""
    h = dot13(x, w13, group_sizes)
    gate, up = jnp.split(h, 2, axis=-1)
    return dot2(jax.nn.silu(gate) * up, w2, group_sizes)


def _build_dots(path: str, implementation: Implementation, compute_dtype) -> tuple[RaggedDot, RaggedDot]:
    """Return the (w13, w2) grouped-matmul callables for the requested path.

    bf16 is the baseline ``ragged_dot``; fp8 wraps each GEMM in its own ``Fp8RaggedDotOp``
    (independent delayed-scaling state per expert projection, as in a real model)."""
    if path == "bf16":
        dot = lambda a, b, gs: ragged_dot(a, b, gs, implementation=implementation)  # noqa: E731
        return dot, dot
    if path == "fp8":
        op13 = Fp8RaggedDotOp.init(compute_dtype=compute_dtype, implementation=implementation)
        op2 = Fp8RaggedDotOp.init(compute_dtype=compute_dtype, implementation=implementation)
        return op13, op2
    raise ValueError(f"unknown path {path!r}")


def _make_inputs(tokens: int, hidden: int, intermediate: int, experts: int, dtype, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Unit-variance activations (post-LayerNorm) and modest weights, so operands sit well
    # inside E4M3's range at unit (cold-start) scale — the regime delayed scaling converges
    # to. Tiny magnitudes would underflow E4M3's subnormal floor and confound the f8-vs-bf16
    # comparison with cold-start scaling artifacts rather than genuine quantization error.
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), dtype)
    w13 = jnp.asarray(rng.standard_normal((experts, hidden, 2 * intermediate)) * 0.08, dtype)
    w2 = jnp.asarray(rng.standard_normal((experts, intermediate, hidden)) * 0.08, dtype)
    # Even token-to-expert assignment summing to `tokens` (a saturated dispatch).
    counts = rng.multinomial(tokens, np.ones(experts) / experts)
    group_sizes = jnp.asarray(counts, jnp.int32)
    return x, w13, w2, group_sizes


# Dtypes XLA prints for f8 operands; their presence on a GEMM operand is the f8 signal.
_F8_DTYPES = ("f8e4m3fn", "f8e5m2", "f8e4m3", "float8_e4m3fn", "float8_e5m2")
# Grouped-GEMM instruction kinds across backends: Triton custom-call, XLA gemm fusion,
# cuBLASLt f8, and the XLA ragged-dot instruction (hyphenated; the underscore form only
# appears in op_name metadata, which we exclude by requiring a real instruction line).
_GEMM_MARKERS = ("xla.gpu.triton", "__cublas$lt$matmul", "gemm_fusion", "kind=kCustom", "ragged-dot(")


def _scan_hlo_for_f8(hlo: str) -> dict:
    """Scan compiled HLO for f8 operands reaching the grouped GEMM.

    Returns global f8 dtype counts plus, for each grouped-GEMM *instruction* line, whether
    an f8 dtype appears on it (i.e. f8 operands flow into the GEMM). This is the correct
    signal for ragged f8 on this jaxlib/H100, where the GEMM is a Triton custom-call or an
    XLA gemm fusion rather than a literal ``__cublas$lt$matmul$f8``. Lines without ``=`` are
    metadata/stack frames (op_name tables) and are excluded.
    """
    f8_counts = {dt: len(re.findall(re.escape(dt), hlo)) for dt in _F8_DTYPES}
    gemm_lines = [
        line.strip() for line in hlo.splitlines() if " = " in line and any(marker in line for marker in _GEMM_MARKERS)
    ]
    gemm_lines_with_f8 = [line for line in gemm_lines if any(dt in line for dt in _F8_DTYPES)]
    return {
        "f8_dtype_counts": f8_counts,
        "total_f8_mentions": sum(f8_counts.values()),
        "num_gemm_call_lines": len(gemm_lines),
        "num_gemm_call_lines_with_f8_operand": len(gemm_lines_with_f8),
        "f8_reaches_gemm": bool(gemm_lines_with_f8),
        "sample_gemm_lines": gemm_lines_with_f8[:3] or gemm_lines[:3],
    }


def _time_jitted(fn, *args, steps: int, warmup: int) -> tuple[float, float]:
    """Return (compile_time, mean steady-state time) in seconds."""
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(compiled(*args))
    return compile_time, (time.perf_counter() - start) / steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192, help="dispatched tokens (seq x top_k)")
    parser.add_argument("--hidden", type=int, default=2048, help="model hidden dim D")
    parser.add_argument("--intermediate", type=int, default=5632, help="per-expert intermediate dim F")
    parser.add_argument("--experts", type=int, default=8, help="number of experts E")
    parser.add_argument("--path", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--implementation", choices=("auto", "triton", "xla"), default="auto")
    parser.add_argument("--dtype", default="bfloat16", help="compute dtype")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--no-print-hlo", dest="print_hlo", action="store_false")
    args = parser.parse_args()

    dtype = jnp.dtype(args.dtype)
    x, w13, w2, group_sizes = _make_inputs(args.tokens, args.hidden, args.intermediate, args.experts, dtype)
    dot13, dot2 = _build_dots(args.path, args.implementation, dtype)

    def expert_out(x, w13, w2):
        return _expert_mlp(x, w13, w2, group_sizes, dot13=dot13, dot2=dot2)

    # Consistency vs the bf16 reference (same shapes/inputs), on the same backend.
    rel_frob_vs_bf16 = None
    if args.path != "bf16":
        bf16_dot13, bf16_dot2 = _build_dots("bf16", args.implementation, dtype)
        ref = jax.jit(lambda x, w13, w2: _expert_mlp(x, w13, w2, group_sizes, dot13=bf16_dot13, dot2=bf16_dot2))
        out_ref = np.asarray(jax.block_until_ready(ref(x, w13, w2)), np.float32)
        out_path = np.asarray(jax.block_until_ready(jax.jit(expert_out)(x, w13, w2)), np.float32)
        rel_frob_vs_bf16 = float(np.linalg.norm(out_path - out_ref) / (np.linalg.norm(out_ref) + 1e-9))

    def fwd(x, w13, w2):
        return expert_out(x, w13, w2).astype(jnp.float32).sum()

    grad = jax.grad(fwd, argnums=(0, 1, 2))

    timed = fwd if args.forward_only else grad

    # Always compute the f8 scan (the verdict); the full HLO text is large at real shapes
    # and only dumped under --print-hlo to keep job logs readable.
    hlo = jax.jit(timed).lower(x, w13, w2).compile().as_text()
    scan = _scan_hlo_for_f8(hlo)
    if args.print_hlo:
        print("=== compiled HLO ===")
        print(hlo)
    print("=== f8 scan ===")
    print(json.dumps(scan, indent=2))

    compile_time, steady = _time_jitted(timed, x, w13, w2, steps=args.steps, warmup=args.warmup)

    # Per-token expert flops are independent of E (each token hits one expert).
    # Forward GEMMs: w13 = 2*T*D*2F, w2 = 2*T*F*D -> 6*T*D*F. Backward ~2x forward.
    fwd_flops = 6 * args.tokens * args.hidden * args.intermediate
    total_flops = fwd_flops if args.forward_only else 3 * fwd_flops
    peak = _peak_tflops_per_s(args.path == "fp8")
    achieved_tflops = total_flops / steady / 1e12
    mfu = (achieved_tflops * 1e12 / peak) if peak else None

    result = {
        "path": args.path,
        "implementation": args.implementation,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "experts": args.experts,
        "forward_only": args.forward_only,
        "compile_time_s": compile_time,
        "steady_time_s": steady,
        "achieved_tflops_per_s": achieved_tflops,
        "mfu": mfu,
        "rel_frob_vs_bf16": rel_frob_vs_bf16,
        "f8_reaches_gemm": scan["f8_reaches_gemm"] if scan else None,
        "total_f8_mentions": scan["total_f8_mentions"] if scan else None,
        "tune": {k: os.environ.get(k) for k in _TUNE_ENV_KEYS},
    }
    print("result_json " + json.dumps(result))


if __name__ == "__main__":
    main()
