# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H100 microbenchmark for the Kimi Delta Attention (KDA) chunked kernel.

Measures forward and forward+backward wall time, tokens/s, and achieved GEMM
TFLOP/s (analytic FLOP model -- ``cost_analysis`` undercounts ``lax.scan`` bodies)
for :func:`chunk_kda` at a realistic long-context shape, comparing the shipped
bf16-intra-chunk kernel against fp32 (TF32-default) and true-fp32. Pure
microbenchmark: no wandb, no data, single device. ``KDA_FULL=1`` adds a chunk-size
sweep (C in 32/128/256).

Run (H100, cw-us-east-02a)::

    NO_PROXY="*" uv run iris --cluster marin job run --no-wait \\
        --enable-extra-resources --target-cluster cw-us-east-02a \\
        --gpu 1 --extra gpu \\
        -- python -m experiments.grug.moe.bench_kda
"""

import contextlib
import functools
import math
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.moe.kda import chunk_kda, recurrent_kda

# H100 SXM peak (dense, no sparsity), TFLOP/s.
_H100_BF16_PEAK = 989.0
_H100_TF32_PEAK = 494.5
_H100_FP32_PEAK = 66.9


def _make_inputs(b, h, length, dk, dv, seed=0):
    rng = np.random.RandomState(seed)
    q = jnp.asarray(rng.randn(b, h, length, dk), jnp.float32)
    k = jnp.asarray(rng.randn(b, h, length, dk), jnp.float32)
    v = jnp.asarray(rng.randn(b, h, length, dv), jnp.float32)
    g = -0.1 * jnp.abs(jnp.asarray(rng.randn(b, h, length, dk), jnp.float32))
    beta = jnp.asarray(rng.rand(b, h, length), jnp.float32)
    return q, k, v, g, beta


def _time_fn(fn, args, iters=20, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def _analytic_matmul_flops(g_batch, length, c, d, mode):
    """Analytic matmul FLOPs for chunk_kda (cost_analysis undercounts scan bodies).

    Per (batch, chunk) instance: 10*C^2*d + 6*C*d^2 [intra-chunk + state GEMMs] plus
    the Neumann inverse (4*C^3 per doubling step). Backward ~= 2x forward for GEMMs.
    """
    iters = max(0, math.ceil(math.log2(c)) - 1)
    per_instance = 10 * c * c * d + 6 * c * d * d + 4 * c * c * c * iters
    n = length // c
    fwd = g_batch * n * per_instance
    return fwd * (3.0 if mode == "fwd_bwd" else 1.0)


RESULTS: list[str] = []


def _run_variant(name, kernel, args, chunk_size, peak, mode, precision=None):
    fwd = functools.partial(kernel, chunk_size=chunk_size)

    def fwd_out(*a):
        return fwd(*a)[0]

    def grad_out(*a):
        return jax.grad(lambda *b: jnp.sum(fwd(*b)[0].astype(jnp.float32)), argnums=(0, 1, 2, 4))(*a)

    raw = fwd_out if mode == "fwd" else grad_out
    with jax.default_matmul_precision(precision) if precision else _nullctx():
        fn = jax.jit(raw)
        med, _best = _time_fn(fn, args)
    b, h, length = args[0].shape[0], args[0].shape[1], args[0].shape[2]
    d = args[0].shape[3]
    tokens = b * h * length
    flops = _analytic_matmul_flops(b * h, length, chunk_size, d, mode)
    tflops = flops / med / 1e12
    mfu = 100.0 * tflops / peak
    line = (
        f"L={length:<6d} {name:22s} C={chunk_size:<4d} {mode:8s} "
        f"med={med*1e3:8.3f}ms tok/s={tokens/med/1e6:8.2f}M "
        f"gemmTF/s={tflops:6.1f} MFU={mfu:5.1f}%(vs{peak:.0f})"
    )
    RESULTS.append(line)
    print(line, flush=True)
    return med, tflops, mfu


@contextlib.contextmanager
def _nullctx():
    yield


def _decompose(b, h, length, dk, dv, c):
    """Attribute the fwd cost of the bf16 kernel across its components (all bf16).

    Times, at the real per-instance batch (G*n_chunks, C, C / C, d): the Neumann
    triangular inverse alone, one big batched intra-chunk q@k GEMM, and the full
    kernel -- so we can see whether the sequential scan or the parallel intra-chunk
    inverse dominates before committing to a Pallas rewrite.
    """
    from experiments.grug.moe.kda import _unit_lower_triangular_inverse  # noqa: PLC0415

    g_batch = b * h
    n = length // c
    inst = g_batch * n
    rng = np.random.RandomState(0)
    # strictly-lower bf16 matrix for the inverse
    a = jnp.asarray(rng.randn(inst, c, c), jnp.float32)
    a = jnp.where(jnp.tril(jnp.ones((c, c), bool), -1), a, 0.0)

    def neumann(a_):
        return _unit_lower_triangular_inverse(a_, c, matmul_dtype=jnp.bfloat16)

    med, _ = _time_fn(jax.jit(neumann), (a,))
    RESULTS.append(f"  [decomp C={c}] neumann_inverse only : {med*1e3:8.3f}ms (batch={inst}, C={c})")
    print(RESULTS[-1], flush=True)

    # one big batched intra-chunk q@k^T over all chunks (parallel, no scan)
    qc = jnp.asarray(rng.randn(g_batch, n, c, dk), jnp.bfloat16)
    kc = jnp.asarray(rng.randn(g_batch, n, c, dk), jnp.bfloat16)

    def bigmm(q_, k_):
        return jnp.einsum("...rd,...jd->...rj", q_, k_).astype(jnp.float32)

    med, _ = _time_fn(jax.jit(bigmm), (qc, kc))
    RESULTS.append(f"  [decomp C={c}] one batched CxC gemm  : {med*1e3:8.3f}ms")
    print(RESULTS[-1], flush=True)

    args = _make_inputs(b, h, length, dk, dv)
    med_full, _ = _time_fn(jax.jit(lambda *a_: chunk_kda(*a_, chunk_size=c, matmul_dtype=jnp.bfloat16)[0]), args)
    RESULTS.append(f"  [decomp C={c}] FULL kernel fwd       : {med_full*1e3:8.3f}ms")
    print(RESULTS[-1], flush=True)


def main():
    RESULTS.append(
        f"jax {jax.__version__} devices={jax.devices()} default_mm_prec={jax.config.jax_default_matmul_precision}"
    )
    print(RESULTS[-1], flush=True)

    b = int(os.environ.get("KDA_B", "4"))
    h = int(os.environ.get("KDA_H", "8"))
    dk = int(os.environ.get("KDA_DK", "128"))
    dv = int(os.environ.get("KDA_DV", "128"))
    lengths = [int(x) for x in os.environ.get("KDA_LENS", "8192").split(",")]

    if os.environ.get("KDA_UNROLL") == "1":
        c = int(os.environ.get("KDA_C", "128"))
        args = _make_inputs(b, h, lengths[0], dk, dv)
        for mode in ("fwd", "fwd_bwd"):
            for u in (int(x) for x in os.environ.get("KDA_UNROLLS", "1,2,4,8,16").split(",")):
                kern = functools.partial(chunk_kda, matmul_dtype=jnp.bfloat16, scan_unroll=u)
                _try(f"bf16 unroll={u}", kern, args, c, _H100_BF16_PEAK, mode)
        marker = "###KDA_RESULTS###"
        print("\n" + marker, flush=True)
        for ln in RESULTS:
            print(ln, flush=True)
        print(marker, flush=True)
        sys.stdout.flush()
        sys.exit(3)

    if os.environ.get("KDA_DECOMPOSE") == "1":
        for c in (int(x) for x in os.environ.get("KDA_SWEEP", "64,128").split(",")):
            _decompose(b, h, lengths[0], dk, dv, c)
        marker = "###KDA_RESULTS###"
        print("\n" + marker, flush=True)
        for ln in RESULTS:
            print(ln, flush=True)
        print(marker, flush=True)
        sys.stdout.flush()
        sys.exit(3)

    kda_bf16 = functools.partial(chunk_kda, matmul_dtype=jnp.bfloat16)
    kda_fp32 = functools.partial(chunk_kda, matmul_dtype=jnp.float32)

    # Correctness: default (bf16) kernel vs the sequential fp32 oracle on a small slice.
    cq, ck, cv, cg, cb = _make_inputs(2, 2, 512, dk, dv, seed=1)
    o_bf, _ = chunk_kda(cq, ck, cv, cg, cb, chunk_size=64)
    o_ref, _ = recurrent_kda(cq, ck, cv, cg, cb)
    rel = float(jnp.max(jnp.abs(o_bf - o_ref)) / (jnp.max(jnp.abs(o_ref)) + 1e-9))
    RESULTS.append(f"correctness bf16-vs-recurrent max_rel_err={rel:.4e}")
    print(RESULTS[-1], flush=True)

    full = os.environ.get("KDA_FULL") == "1"
    for length in lengths:
        args = _make_inputs(b, h, length, dk, dv)
        for mode in ("fwd", "fwd_bwd"):
            # bf16 first so it is captured even if a later variant OOMs.
            _try("bf16-mm", kda_bf16, args, 64, _H100_BF16_PEAK, mode)
            _try("fp32,default", kda_fp32, args, 64, _H100_FP32_PEAK, mode)
            _try("fp32,highest", kda_fp32, args, 64, _H100_FP32_PEAK, mode, precision="highest")
            if full:
                sweep = [int(x) for x in os.environ.get("KDA_SWEEP", "32,128,256").split(",")]
                for c in sweep:
                    _try("fp32,default", kda_fp32, args, c, _H100_FP32_PEAK, mode)
                    _try("bf16-mm", kda_bf16, args, c, _H100_BF16_PEAK, mode)

    marker = "###KDA_RESULTS###"
    print("\n" + marker, flush=True)
    for ln in RESULTS:
        print(ln, flush=True)
    print(marker, flush=True)
    # finelog cannot serve federated peer-cluster logs after termination, but iris
    # captures the stdout tail into the job error field on abnormal exit -- so exit
    # non-zero to make the results retrievable via `job summary --json`.
    sys.stdout.flush()
    sys.exit(3)


def _try(name, kernel, args, c, peak, mode, precision=None):
    try:
        return _run_variant(name, kernel, args, c, peak, mode, precision=precision)
    except Exception as e:
        msg = f"ERR {name} C={c} {mode}: {type(e).__name__}: {str(e)[:120]}"
        RESULTS.append(msg)
        print(msg, flush=True)
        return None


if __name__ == "__main__":
    main()
