# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""M0 step 3 end-to-end validation: the Mosaic-GPU FP8 hybrid through Fp8RaggedDotOp (H100).

GFP8-026 wired a "mosaic" backend into the FP8 grouped-dot path: forward and dgrad run as
genuine f8 wgmma grouped GEMMs (jax's ragged_dot_mgpu), while the weight-gradient (wgrad) runs
in bf16 on the dequantized f8 operands (the Hopper f8 operand-transpose wall, GFP8-025). This
script validates that hybrid end-to-end on the real Grug MoE expert MLP — both NUMERICS (forward
output and all three gradients vs a bf16 reference) and SPEED (fwd and fwd+bwd vs bf16) — closing
the loop on the ~1.3x fwd+bwd projection from the per-GEMM bench (GFP8-024).

    h        = ragged_dot(x[T,D], w13[E,D,2F])   -> [T, 2F]
    gate, up = split(h);  g = silu(gate) * up
    out      = ragged_dot(g[T,F], w2[E,F,D])      -> [T, D]

The mosaic path historically REQUIRED grad_dtype=e4m3: stock Mosaic wgmma emitted a single element
type for both operands and rejected mixed e4m3 x e5m2 (the dgrad is e5m2-grad x e4m3-rhs). With a
jaxlib carrying the mixed-fp8-wgmma patch (mcwitt/jax), grad_dtype=e5m2 lowers — both backward GEMMs
run as genuine mixed E4M3/E5M2 wgmma — so the TE-style hybrid recipe runs on the fast f8 path.

Mosaic-GPU is H100-only and needs the cluster CUDA-toolchain bootstrap (see
mosaic-gpu-cluster-toolchain memory / bench_ragged_mosaic_fp8_fwdbwd.py). Self-contained (no
import of the jax-importing bench helpers) because the bootstrap must run before `import jax`.

    uv run iris --cluster=cw-us-east-02a job run --gpu H100x1 --enable-extra-resources --extra gpu \\
        -- python lib/levanter/scripts/bench/bench_ragged_mosaic_hybrid_e2e.py --path mosaic
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

from haliax._src.fp8_ragged import MosaicWgradMode  # noqa: E402
from haliax.nn.ragged_dot import ragged_dot  # noqa: E402
from haliax.quantization import Fp8RaggedDotOp  # noqa: E402

# H100 SXM dense matmul peaks; FP8 (E4M3) is 2x the BF16 rate.
_H100_SXM_BF16_TFLOPS_PER_S = 989.5e12
_H100_SXM_FP8_TFLOPS_PER_S = 1978.9e12

_GRAD_DTYPES = {"e4m3": jnp.float8_e4m3fn, "e5m2": jnp.float8_e5m2}


def _expert_mlp(x, w13, w2, group_sizes, *, dot13, dot2):
    """Grug MoE expert MLP: gated up-projection, SiLU, down-projection, all grouped."""
    h = dot13(x, w13, group_sizes)
    gate, up = jnp.split(h, 2, axis=-1)
    return dot2(jax.nn.silu(gate) * up, w2, group_sizes)


def _make_inputs(tokens, hidden, intermediate, experts, dtype, seed=0):
    rng = np.random.default_rng(seed)
    # Unit-variance activations and modest weights so operands sit inside E4M3 range at unit
    # (cold-start) scale — the regime delayed scaling converges to (matches bench_ragged_fp8).
    x = jnp.asarray(rng.standard_normal((tokens, hidden)), dtype)
    w13 = jnp.asarray(rng.standard_normal((experts, hidden, 2 * intermediate)) * 0.08, dtype)
    w2 = jnp.asarray(rng.standard_normal((experts, intermediate, hidden)) * 0.08, dtype)
    counts = rng.multinomial(tokens, np.ones(experts) / experts)
    return x, w13, w2, jnp.asarray(counts, jnp.int32)


def _build_dots(path, compute_dtype, grad_dtype, mosaic_wgrad):
    """(w13, w2) grouped-matmul callables. bf16 is the baseline ragged_dot; the f8 paths wrap
    each GEMM in its own Fp8RaggedDotOp (independent delayed-scaling state per projection)."""
    if path == "bf16":
        dot = lambda a, b, gs: ragged_dot(a, b, gs, implementation="auto")  # noqa: E731
        return dot, dot
    impl = "mosaic" if path == "mosaic" else "triton"
    kw = dict(compute_dtype=compute_dtype, implementation=impl, grad_dtype=grad_dtype, mosaic_wgrad=mosaic_wgrad)
    return Fp8RaggedDotOp.init(**kw), Fp8RaggedDotOp.init(**kw)


def _rel_frob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _time_jitted(fn, *args, steps, warmup):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=8192, help="dispatched tokens (seq x top_k)")
    ap.add_argument("--hidden", type=int, default=2048, help="model hidden dim D")
    ap.add_argument("--intermediate", type=int, default=5632, help="per-expert intermediate dim F")
    ap.add_argument("--experts", type=int, default=8, help="number of experts E")
    ap.add_argument("--path", choices=("bf16", "mosaic", "triton"), default="mosaic")
    ap.add_argument("--grad-dtype", choices=("e4m3", "e5m2"), default="e4m3")
    ap.add_argument(
        "--mosaic-wgrad",
        choices=("bf16", "fp8"),
        default="bf16",
        help="mosaic weight-gradient: bf16 hybrid (default) or the f8 cast-transpose wgrad (GFP8-033 M3)",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--forward-only", action="store_true")
    args = ap.parse_args()

    # With a jaxlib whose Mosaic wgmma verifier allows mixed E4M3/E5M2 operands
    # (mcwitt/jax mixed-fp8-wgmma), grad_dtype=e5m2 lowers: both backward GEMMs become
    # mixed (dlhs = e5m2-grad x e4m3-rhs, wgrad = e4m3-act x e5m2-grad). On a stock jaxlib
    # this raises in the Mosaic kernels — run e5m2+mosaic only on the patched build.

    dtype = jnp.dtype(args.dtype)
    grad_dtype = _GRAD_DTYPES[args.grad_dtype]
    mosaic_wgrad = MosaicWgradMode(args.mosaic_wgrad)
    x, w13, w2, group_sizes = _make_inputs(args.tokens, args.hidden, args.intermediate, args.experts, dtype)
    dot13, dot2 = _build_dots(args.path, dtype, grad_dtype, mosaic_wgrad)

    print(
        f"hardware: {[d.device_kind for d in jax.devices()]}  path={args.path}  "
        f"grad_dtype={args.grad_dtype}  mosaic_wgrad={args.mosaic_wgrad}"
    )

    def expert_out(x, w13, w2):
        return _expert_mlp(x, w13, w2, group_sizes, dot13=dot13, dot2=dot2)

    def loss(x, w13, w2):
        return expert_out(x, w13, w2).astype(jnp.float32).sum()

    grad = jax.grad(loss, argnums=(0, 1, 2))

    # Numerics vs the bf16 reference (same shapes/inputs/backend): forward output and all grads.
    fwd_rel_frob = grad_rel_frob = None
    if args.path != "bf16":
        bf13, bf2 = _build_dots("bf16", dtype, grad_dtype, mosaic_wgrad)
        ref_out = jax.jit(lambda x, w13, w2: _expert_mlp(x, w13, w2, group_sizes, dot13=bf13, dot2=bf2))
        ref_grad = jax.grad(
            lambda x, w13, w2: _expert_mlp(x, w13, w2, group_sizes, dot13=bf13, dot2=bf2).astype(jnp.float32).sum(),
            argnums=(0, 1, 2),
        )
        o_ref = jax.block_until_ready(ref_out(x, w13, w2))
        o_path = jax.block_until_ready(jax.jit(expert_out)(x, w13, w2))
        fwd_rel_frob = _rel_frob(o_path, o_ref)
        g_ref = jax.block_until_ready(ref_grad(x, w13, w2))
        g_path = jax.block_until_ready(jax.jit(grad)(x, w13, w2))
        grad_rel_frob = {name: _rel_frob(gp, gr) for name, gp, gr in zip(("dx", "dw13", "dw2"), g_path, g_ref)}
        print("=== numerics vs bf16 ===")
        print(f"  forward rel_frob: {fwd_rel_frob:.4e}")
        for name, v in grad_rel_frob.items():
            print(f"  {name:4} rel_frob: {v:.4e}")

    timed = (lambda x, w13, w2: expert_out(x, w13, w2)) if args.forward_only else grad
    compile_time, steady = _time_jitted(timed, x, w13, w2, steps=args.steps, warmup=args.warmup)

    # Per-token expert flops (each token hits one expert): fwd = 6*T*D*F; bwd ~2x fwd.
    fwd_flops = 6 * args.tokens * args.hidden * args.intermediate
    total_flops = fwd_flops if args.forward_only else 3 * fwd_flops
    is_fp8 = args.path != "bf16"
    peak = _H100_SXM_FP8_TFLOPS_PER_S if is_fp8 else _H100_SXM_BF16_TFLOPS_PER_S
    achieved_tflops = total_flops / steady / 1e12

    result = {
        "path": args.path,
        "grad_dtype": args.grad_dtype,
        "mosaic_wgrad": args.mosaic_wgrad,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "experts": args.experts,
        "forward_only": args.forward_only,
        "compile_time_s": compile_time,
        "steady_time_s": steady,
        "achieved_tflops_per_s": achieved_tflops,
        "mfu": achieved_tflops * 1e12 / peak,
        "fwd_rel_frob_vs_bf16": fwd_rel_frob,
        "grad_rel_frob_vs_bf16": grad_rel_frob,
    }
    print("result_json " + json.dumps(result))


if __name__ == "__main__":
    main()
