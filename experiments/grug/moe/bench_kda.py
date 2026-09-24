# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H100 microbenchmark for the Kimi Delta Attention (KDA) chunked kernel.

Measures forward and forward+backward wall time, XLA-reported FLOPs (via
``cost_analysis``), achieved TFLOP/s and MFU for :func:`chunk_kda` at a realistic
long-context shape, sweeping chunk size, the intra-chunk inverse variant, and an
inline bf16-matmul prototype (fp32 state recurrence kept). Pure microbenchmark:
no wandb, no data, single device.

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
from jax import lax

from experiments.grug.moe.kda import (
    _DEFLATE_EXP_CAP,
    _prepare_qk,
    chunk_kda,
)

# H100 SXM peak (dense, no sparsity), TFLOP/s.
_H100_BF16_PEAK = 989.0
_H100_TF32_PEAK = 494.5
_H100_FP32_PEAK = 66.9


def _chunk_kda_matmul_dtype(
    q,
    k,
    v,
    g,
    beta,
    *,
    chunk_size=64,
    initial_state=None,
    use_qk_l2norm=True,
    mm_dtype=jnp.bfloat16,
):
    """Prototype: identical math to chunk_kda but the intra-chunk GEMM operands run
    in ``mm_dtype`` (bf16) while all decay/cumsum and the cross-chunk state ``S``
    stay fp32. State-touching matmuls keep fp32 for stability."""
    q, k = _prepare_qk(q, k, use_qk_l2norm)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    lead = q.shape[:-2]
    orig_len = q.shape[-2]
    dk, dv = q.shape[-1], v.shape[-1]
    pad = (-orig_len) % chunk_size
    if pad:
        q = jnp.pad(q, [(0, 0)] * (q.ndim - 2) + [(0, pad), (0, 0)])
        k = jnp.pad(k, [(0, 0)] * (k.ndim - 2) + [(0, pad), (0, 0)])
        v = jnp.pad(v, [(0, 0)] * (v.ndim - 2) + [(0, pad), (0, 0)])
        g = jnp.pad(g, [(0, 0)] * (g.ndim - 2) + [(0, pad), (0, 0)])
        beta = jnp.pad(beta, [(0, 0)] * (beta.ndim - 1) + [(0, pad)])

    n_chunks = q.shape[-2] // chunk_size
    c = chunk_size

    def to_chunks(x):
        return x.reshape(*lead, n_chunks, c, x.shape[-1])

    qc, kc, vc, gc = to_chunks(q), to_chunks(k), to_chunks(v), to_chunks(g)
    bc = beta.reshape(*lead, n_chunks, c)

    g_cum = jnp.cumsum(gc, axis=-2)
    exp_g = jnp.exp(g_cum)
    exp_ng = jnp.exp(jnp.minimum(-g_cum, _DEFLATE_EXP_CAP))

    v_beta = vc * bc[..., None]
    k_beta = kc * bc[..., None]

    def bf(x):
        return x.astype(mm_dtype)

    k_beta_inflate = k_beta * exp_g
    k_deflate = kc * exp_ng
    a_raw = -jnp.einsum("...rd,...id->...ri", bf(k_beta_inflate), bf(k_deflate)).astype(jnp.float32)
    strict_lower = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
    a_raw = jnp.where(strict_lower, a_raw, 0.0)

    a_bcc = a_raw.reshape(-1, c, c)
    # Neumann inverse in bf16 (the log-depth CxCxC product is the dominant intra-chunk cost).
    eye = jnp.eye(c, dtype=mm_dtype)
    power = a_bcc.astype(mm_dtype)
    inv = eye + power
    kk = 1
    while (1 << kk) < c:
        power = power @ power
        inv = inv @ (eye + power)
        kk += 1
    t_mat = inv.astype(jnp.float32).reshape(*lead, n_chunks, c, c)

    v_pseudo = jnp.einsum("...rj,...jd->...rd", bf(t_mat), bf(v_beta)).astype(jnp.float32)
    k_cumdecay = jnp.einsum("...rj,...jd->...rd", bf(t_mat), bf(k_beta * exp_g)).astype(jnp.float32)

    q_inflate = qc * exp_g

    if initial_state is None:
        state = jnp.zeros((*lead, dk, dv), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    strict_upper = jnp.triu(jnp.ones((c, c), dtype=bool), k=1)

    def move_chunk_front(x):
        return jnp.moveaxis(x, -3, 0)

    scan_inputs = (
        move_chunk_front(q_inflate),
        move_chunk_front(k_deflate),
        move_chunk_front(kc),
        move_chunk_front(g_cum),
        move_chunk_front(v_pseudo),
        move_chunk_front(k_cumdecay),
    )

    def chunk_step(s_prev, inp):
        q_inf, k_def, k_i, gcum_i, v_ps, k_cd = inp
        attn = jnp.einsum("...rd,...jd->...rj", bf(q_inf), bf(k_def)).astype(jnp.float32)
        attn = jnp.where(strict_upper, 0.0, attn)
        # state matmuls kept fp32 for stability
        v_prime = jnp.einsum("...rd,...dm->...rm", k_cd, s_prev)
        v_new = v_ps - v_prime
        inter = jnp.einsum("...rd,...dm->...rm", q_inf, s_prev)
        out_i = inter + jnp.einsum("...rj,...jm->...rm", bf(attn), bf(v_new)).astype(jnp.float32)
        g_tail = gcum_i[..., -1, :]
        decay_tail = jnp.exp(g_tail)
        decay_weights = jnp.exp(g_tail[..., None, :] - gcum_i)
        add = jnp.einsum("...rd,...rm->...dm", k_i * decay_weights, v_new)
        s_new = s_prev * decay_tail[..., :, None] + add
        return s_new, out_i

    state, out_chunks = lax.scan(chunk_step, state, scan_inputs)
    out = jnp.moveaxis(out_chunks, 0, -3)
    out = out.reshape(*lead, n_chunks * c, dv)
    if pad:
        out = out[..., :orig_len, :]
    return out, state


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
    jax.clear_caches()
    return med, tflops, mfu


@contextlib.contextmanager
def _nullctx():
    yield


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

    for length in lengths:
        args = _make_inputs(b, h, length, dk, dv)
        for mode in ("fwd", "fwd_bwd"):
            # Essential comparison first (so a later OOM still leaves it captured):
            # current default vs true-fp32 vs bf16 intra-chunk, all at C=64.
            _try("fp32,default", chunk_kda, args, 64, _H100_FP32_PEAK, mode)
            _try("fp32,highest", chunk_kda, args, 64, _H100_FP32_PEAK, mode, precision="highest")
            _try("bf16-mm", _chunk_kda_matmul_dtype, args, 64, _H100_BF16_PEAK, mode)
            # Chunk-size sweep for both precisions.
            for c in (32, 128, 256):
                _try("fp32,default", chunk_kda, args, c, _H100_FP32_PEAK, mode)
                _try("bf16-mm", _chunk_kda_matmul_dtype, args, c, _H100_BF16_PEAK, mode)

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
