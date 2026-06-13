# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dev harness for a TPU-friendly batched QR to replace jnp.linalg.qr in KLSOAPH.

KLSOAPH's per-step orthogonal-iteration refresh does ``Q, _ = jnp.linalg.qr(GG @ Q)``
batched over the 256-expert axis (shapes [256, n, n], n in {256, 512}). On TPU the
XLA QR lowering is the compile/runtime bottleneck (full-matrix d512 stalls in XLA
compile for >45 min). CholeskyQR2 computes the same QR factorization (orthonormal Q,
upper-tri R) using only MXU-friendly ops — matmul + Cholesky + triangular solve — and
is the standard accelerator QR algorithm.

CholeskyQR(M):  G = Mᵀ M ;  R = chol(G)ᵀ (upper) ;  Q = M R⁻¹
CholeskyQR2:    run twice (Q1 from M, Q2 from Q1) for numerical stability — the second
                pass orthonormalizes the already-near-orthonormal Q1, recovering full
                accuracy that single-pass CholeskyQR loses for ill-conditioned M.

This file is a standalone benchmark (not wired into the optimizer yet): correctness vs
jnp.linalg.qr on CPU, then compile/runtime on TPU.
"""

import argparse
import time

import jax
import jax.numpy as jnp


def cholesky_qr(m):
    """Single-pass CholeskyQR: returns (Q, R) with M = Q R, Q orthonormal, R upper-tri.

    Batched over leading axes; the trailing two dims are the matrix. Adds a tiny
    Cholesky jitter for PSD safety on the Gram Mᵀ M.
    """
    n = m.shape[-1]
    gram = jnp.einsum("...ki,...kj->...ij", m, m)  # Mᵀ M, [..., n, n]
    eye = jnp.eye(n, dtype=m.dtype)
    # Jitter scaled by the Gram trace keeps it relative; tiny so it barely perturbs R.
    jitter = 1e-6 * jnp.einsum("...ii->...", gram)[..., None, None] / n
    chol = jnp.linalg.cholesky(gram + jitter * eye)  # lower L with G = L Lᵀ
    r = jnp.swapaxes(chol, -1, -2)  # upper R = Lᵀ
    # Q = M R⁻¹  via triangular solve  (Q Rᵀ ... solve Xᵀ): solve R for Qᵀ from Mᵀ.
    # Solve Q from Q R = M  ->  Rᵀ Qᵀ = Mᵀ. Use jax.scipy triangular solve.
    qt = jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(r, -1, -2), jnp.swapaxes(m, -1, -2), lower=True
    )
    q = jnp.swapaxes(qt, -1, -2)
    return q, r


def cholesky_qr2(m):
    """CholeskyQR2: two CholeskyQR passes for stability. Returns Q (orthonormal)."""
    q1, r1 = cholesky_qr(m)
    q2, r2 = cholesky_qr(q1)
    return q2, jnp.einsum("...ij,...jk->...ik", r2, r1)


def _orthonormality_error(q):
    n = q.shape[-1]
    eye = jnp.eye(n, dtype=q.dtype)
    qtq = jnp.einsum("...ki,...kj->...ij", q, q)
    return float(jnp.max(jnp.abs(qtq - eye)))


def _reconstruction_error(q, r, m):
    return float(jnp.max(jnp.abs(jnp.einsum("...ij,...jk->...ik", q, r) - m)))


def check_correctness(batch=8, n=512, seed=0):
    key = jax.random.PRNGKey(seed)
    m = jax.random.normal(key, (batch, n, n), dtype=jnp.float32)
    q_ref, r_ref = jnp.linalg.qr(m)
    q_c, r_c = cholesky_qr2(m)
    print(f"[correctness n={n} batch={batch}]")
    print(f"  jnp.qr      : ortho_err={_orthonormality_error(q_ref):.2e} recon_err={_reconstruction_error(q_ref, r_ref, m):.2e}")
    print(f"  choleskyQR2 : ortho_err={_orthonormality_error(q_c):.2e} recon_err={_reconstruction_error(q_c, r_c, m):.2e}")
    # SOAP only needs an orthonormal basis of the same column space; signs/order may
    # differ vs Householder QR. Verify the projector Q Qᵀ matches (basis-invariant).
    proj_ref = jnp.einsum("...ij,...kj->...ik", q_ref, q_ref)
    proj_c = jnp.einsum("...ij,...kj->...ik", q_c, q_c)
    print(f"  projector match (QQᵀ) max|Δ| = {float(jnp.max(jnp.abs(proj_ref - proj_c))):.2e}")


def _time_fn(fn, m, label, iters=5):
    f = jax.jit(fn)
    t0 = time.perf_counter()
    out = f(m)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(m)
    jax.block_until_ready(out)
    t_run = (time.perf_counter() - t0) / iters
    print(f"  {label:14s} compile={t_compile:7.2f}s  run/iter={t_run*1e3:8.2f}ms")


def bench(batch=256, n=512, seed=0):
    key = jax.random.PRNGKey(seed)
    m = jax.random.normal(key, (batch, n, n), dtype=jnp.float32)
    # Symmetric PSD input for eigh (matches the SOAP Gram); reuse m for qr benches.
    sym = jnp.einsum("...ki,...kj->...ij", m, m)
    print(f"[bench n={n} batch={batch} on {jax.devices()[0].platform}]")
    _time_fn(lambda x: jnp.linalg.qr(x)[0], m, "jnp.qr")
    _time_fn(lambda x: cholesky_qr2(x)[0], m, "choleskyQR2")
    _time_fn(lambda x: jnp.linalg.eigh(x)[1], sym, "jnp.eigh")
    # SOAP per-step op exactly: Q,_ = qr(GG @ Q) vs cholesky_qr2(GG @ Q).
    q0 = jnp.linalg.qr(m)[0]
    gg = sym
    _time_fn(lambda q: jnp.linalg.qr(jnp.einsum("...ij,...jk->...ik", gg, q))[0], q0, "soap_qr")
    _time_fn(lambda q: cholesky_qr2(jnp.einsum("...ij,...jk->...ik", gg, q))[0], q0, "soap_cholqr")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "bench"], default="correctness")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()
    if args.mode == "correctness":
        check_correctness(batch=args.batch or 8, n=args.n)
    else:
        bench(batch=args.batch or 256, n=args.n)
