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
    qt = jax.scipy.linalg.solve_triangular(jnp.swapaxes(r, -1, -2), jnp.swapaxes(m, -1, -2), lower=True)
    q = jnp.swapaxes(qt, -1, -2)
    return q, r


def cholesky_qr2(m):
    """CholeskyQR2: two CholeskyQR passes for stability. Returns Q (orthonormal)."""
    q1, r1 = cholesky_qr(m)
    q2, r2 = cholesky_qr(q1)
    return q2, jnp.einsum("...ij,...jk->...ik", r2, r1)


# max|QᵀQ - I| above which we fall back to exact jnp.linalg.qr. Exact QR in f32 floors at ~9e-7 and good
# single-pass SCQR matches it; 3e-6 is ~3x that floor — tight enough that any accepted Q is essentially as
# orthonormal as exact QR, so the SOAP eigenbasis never silently degrades (1e-5/1e-4 were too soft).
_SCQR_ORTHO_TOL = 3e-6


def scqr(m, eps=1e-7):
    """SINGLE-pass Shifted Cholesky QR (Su Jianlin, su.md): one Cholesky + one triangular solve,
    with an ORTHONORMALITY-checked fallback to jnp.linalg.qr. Returns Q (orthonormal columns of `m`).

    The shift is λ = eps * gram[0,0] (the top-left Gram element as a spectral-norm proxy) — valid in
    warm-started power iteration because qᵀ·GG·q converges to diagonal with the largest eigenvalue at
    [0,0]. This regularizes the Cholesky against the cond² blow-up of forming MᵀM, so a SINGLE pass
    suffices; the prior two-pass CholeskyQR2 did ~2x the work (and benched 0.66x = slower than XLA QR).

    Fallback is gated on ‖QᵀQ - I‖, NOT isfinite: an ill-conditioned Cholesky can return a finite but
    non-orthonormal Q, which would silently degrade the SOAP eigenbasis (the failure mode that stalls
    training). Only an orthonormality check catches that — finiteness is too soft.
    """
    gram = jnp.einsum("...ki,...kj->...ij", m, m)  # Mᵀ M
    n = m.shape[-1]
    eye = jnp.eye(n, dtype=m.dtype)
    shift = eps * gram[..., :1, :1] * eye
    r = jnp.linalg.cholesky(gram + shift, upper=True)  # upper-tri R, MᵀM ≈ RᵀR
    qt = jax.scipy.linalg.solve_triangular(jnp.swapaxes(r, -1, -2), jnp.swapaxes(m, -1, -2), lower=True)
    q = jnp.swapaxes(qt, -1, -2)
    ortho_err = jnp.max(jnp.abs(jnp.einsum("...ki,...kj->...ij", q, q) - eye))
    return jax.lax.cond(ortho_err < _SCQR_ORTHO_TOL, lambda: q, lambda: jnp.linalg.qr(m)[0])


def _chol_shift(g, eps):
    """Cholesky shift λ = eps * g[0,0] (top-left = spectral-norm proxy; qᵀGGq -> diagonal as q converges)."""
    n = g.shape[-1]
    return g + eps * g[..., :1, :1] * jnp.eye(n, dtype=g.dtype)


def _orthonormalize_against_R(a, eps):
    """Given A, return (A·R⁻¹, R) where RᵀR = chol(AᵀA + λI): one CholeskyQR pass (the orthonormal factor)."""
    g = jnp.einsum("...ki,...kj->...ij", a, a)  # AᵀA
    r = jnp.linalg.cholesky(_chol_shift(g, eps), upper=True)
    q = jnp.swapaxes(
        jax.scipy.linalg.solve_triangular(jnp.swapaxes(r, -1, -2), jnp.swapaxes(a, -1, -2), lower=True), -1, -2
    )
    return q


def dual_scqr_refresh(gg, q, eps=1e-7):
    """Dual-orthogonalization SCQR eigenbasis refresh (Su Jianlin Part 2/3, eq. 4), computed from the
    Gram GG alone (no raw M). Returns q_new spanning the same space as qr(GG @ q) — projector-identical
    to the single-QR refresh — but with each Cholesky seeing cond(GG), not cond(GG)².

    Single QR `qr(GG@q)` via SCQR Choleskys (GG@q)ᵀ(GG@q) = qᵀGG²q  -> cond(GG)²  (ill-conditioned, fails).
    Dual QR `qr(GGᵀ·qr(GG@q))` reorders (eq. 4) to:
        A1 = GG @ q
        R1 = chol(qᵀ·A1 + λI)        # = chol(qᵀ GG q)        -> cond(GG)
        A2 = A1 · R1⁻¹               # whitened first level
        q_new = A2 · R2⁻¹,  R2 = chol(A2ᵀA2 + λI)             -> cond(GG)
    The extra QR is right-multiplication by an upper-triangular matrix, so it does NOT change the column
    space (lossless); it only conditions the Cholesky. Fallback to exact qr(GG@q) gated on ‖QᵀQ-I‖.
    """
    n = q.shape[-1]
    eye = jnp.eye(n, dtype=q.dtype)
    a1 = jnp.einsum("...ij,...jk->...ik", gg, q)  # GG @ q
    g1 = jnp.einsum("...ki,...kj->...ij", q, a1)  # qᵀ (GG q) = qᵀ GG q  (cond GG, not squared)
    r1 = jnp.linalg.cholesky(_chol_shift(g1, eps), upper=True)
    a2 = jnp.swapaxes(
        jax.scipy.linalg.solve_triangular(jnp.swapaxes(r1, -1, -2), jnp.swapaxes(a1, -1, -2), lower=True), -1, -2
    )
    qn = _orthonormalize_against_R(a2, eps)  # second CholeskyQR pass on the whitened A2
    ortho_err = jnp.max(jnp.abs(jnp.einsum("...ki,...kj->...ij", qn, qn) - eye))
    return jax.lax.cond(ortho_err < _SCQR_ORTHO_TOL, lambda: qn, lambda: jnp.linalg.qr(a1)[0])


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
    q_s = scqr(m)
    print(f"[correctness n={n} batch={batch}]")
    print(f"  jnp.qr      : ortho={_orthonormality_error(q_ref):.2e} recon={_reconstruction_error(q_ref, r_ref, m):.2e}")
    print(f"  choleskyQR2 : ortho={_orthonormality_error(q_c):.2e} recon={_reconstruction_error(q_c, r_c, m):.2e}")
    print(f"  scqr (1pass): ortho={_orthonormality_error(q_s):.2e}")
    # SOAP only needs an orthonormal basis of the same column space; signs/order may
    # differ vs Householder QR. Verify the projector Q Qᵀ matches (basis-invariant).
    proj_ref = jnp.einsum("...ij,...kj->...ik", q_ref, q_ref)
    proj_c = jnp.einsum("...ij,...kj->...ik", q_c, q_c)
    proj_s = jnp.einsum("...ij,...kj->...ik", q_s, q_s)
    print(f"  projector match choleskyQR2 (QQᵀ) max|Δ| = {float(jnp.max(jnp.abs(proj_ref - proj_c))):.2e}")
    print(f"  projector match scqr       (QQᵀ) max|Δ| = {float(jnp.max(jnp.abs(proj_ref - proj_s))):.2e}")


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
    _time_fn(lambda q: cholesky_qr2(jnp.einsum("...ij,...jk->...ik", gg, q))[0], q0, "soap_cholqr2")
    _time_fn(lambda q: scqr(jnp.einsum("...ij,...jk->...ik", gg, q)), q0, "soap_scqr")
    _time_fn(lambda q: dual_scqr_refresh(gg, q), q0, "soap_dual_scqr")


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
