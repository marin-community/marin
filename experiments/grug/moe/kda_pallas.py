# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kimi Delta Attention (KDA) for JAX with fused Pallas/Triton kernels (v2).

Recurrence (per batch & head; S is K x V, alpha_t = exp(g_t) per channel):

    S_t = (I - beta_t k_t k_t^T) diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T (scale * q_t)

Four kernels, all Pallas (Triton backend), chunk size C (default 64) split
into 16-token sub-chunks:

  prep_fwd  grid (B, H, N), fully parallel over chunks. Reads q, k, v, g,
            beta straight from their (B, T, H, D) layout and does, in
            registers: in-chunk cumsum of g, the decayed A_kk / A_qk scores,
            the block-triangular inverse (I + A_kk)^-1, and w / u.
  rec_fwd   grid (B, H, V/BV), sequential over chunks with the state in
            registers.  Writes o directly in (B, T, H, V) layout.  The
            chunk-start states are stored (in the matmul dtype) only when a
            gradient is being taken.
  rec_bwd   reverse scan over chunks for the recurrence gradients.
  prep_bwd  grid (B, H, N): hand-derived backward of prep_fwd (also sums the
            per-V-block partial gradients), writing dq, dk, dv, dg, dbeta
            directly in (B, T, H, D) layout.

Numerics: every exp() in the kernels has a non-positive argument, so there is
no overflow and no gate clamping for any g <= 0.  Off-diagonal sub-chunk
blocks use a reference point between the blocks; the 16x16 diagonal blocks
are computed exactly, one row at a time.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as _plt

    _TritonParams = _plt.CompilerParams
except (ImportError, AttributeError):  # older JAX
    from jax.experimental.pallas import gpu as _plg

    _TritonParams = _plg.TritonCompilerParams

f32 = jnp.float32
_L = 16  # sub-chunk length (tensor-core minimum tile)


class _Cfg(NamedTuple):
    chunk: int
    mm_dtype: str
    block_v: int
    block_v_bwd: int
    warps_prep: int
    warps_rec: int
    stages: int
    interpret: bool


# ----------------------------------------------------------------------------
# small in-kernel helpers
# ----------------------------------------------------------------------------
def _precision(dtype, interpret):
    if jnp.dtype(dtype) == f32:
        # 3xTF32 ~ fp32 accuracy on tensor cores; the CPU interpreter lacks it.
        return lax.Precision.HIGHEST if interpret else lax.DotAlgorithmPreset.TF32_TF32_F32_X3
    return None


def _mm(a, b, prec):
    return lax.dot_general(a, b, (((1,), (0,)), ((), ())), precision=prec, preferred_element_type=f32)


def _iota(shape, dim):
    return lax.broadcasted_iota(jnp.int32, shape, dim)


def _row(x, r):
    """x[r] for a 2-D value x and (possibly traced) row index r."""
    return jnp.sum(jnp.where(_iota(x.shape, 0) == r, x, 0.0), axis=0)


def _tile(ref, i):
    return ref[pl.ds(i * _L, _L), :].astype(f32)


def _vtile(ref, i):
    return ref[pl.ds(i * _L, _L)].astype(f32)


def _gate_tiles(g_ref, S):
    """In-chunk cumulative log-gates per sub-chunk.

    Returns G[i] (L,K), base[i] = G just before tile i, end[i] = G at the end
    of tile i (all <= 0, non-increasing)."""
    K = g_ref.shape[-1]
    G, base, end = [], [], []
    b = jnp.zeros((K,), f32)
    for i in range(S):
        Gi = jnp.cumsum(_tile(g_ref, i), axis=0) + b[None, :]
        G.append(Gi)
        base.append(b)
        b = _row(Gi, _L - 1)
        end.append(b)
    return G, base, end


def _inv_unit_lower(A, prec):
    """(I + A)^-1 for strictly-lower 16x16 A: (I-A)(I+A^2)(I+A^4)(I+A^8)."""
    eye = (_iota(A.shape, 0) == _iota(A.shape, 1)).astype(f32)
    P = eye - A
    Ap = A
    for _ in range(3):
        Ap = _mm(Ap, Ap, prec)
        P = _mm(P, eye + Ap, prec)
    return P


def _block_inverse(A, S, prec):
    """Blocks T[i, j] (j <= i) of (I + A)^-1 for block-strictly-lower A."""
    T = {}
    for i in range(S):
        T[i, i] = _inv_unit_lower(A[i, i], prec)
        for j in range(i - 1, -1, -1):
            acc = _mm(A[i, j], T[j, j], prec)
            for l in range(j + 1, i):
                acc = acc + _mm(A[i, l], T[l, j], prec)
            T[i, j] = -_mm(T[i, i], acc, prec)
    return T


def _diag_scores(qt, kt, Gt):
    """Exact 16x16 diagonal blocks: M[r,s] = sum_c k_r k_s e^{G_r-G_s} (s<r),
    Aq[r,s] = sum_c q_r k_s e^{G_r-G_s} (s<=r).  One row per iteration."""
    Lr, K = kt.shape
    rid = _iota((Lr, K), 0)
    r2, c2 = _iota((Lr, Lr), 0), _iota((Lr, Lr), 1)

    def body(r, carry):
        M, Aq = carry
        E = jnp.exp(jnp.where(rid <= r, _row(Gt, r)[None, :] - Gt, -jnp.inf))
        kE = kt * E
        mk = jnp.sum(_row(kt, r)[None, :] * kE, axis=1)
        mq = jnp.sum(_row(qt, r)[None, :] * kE, axis=1)
        return (jnp.where(r2 == r, mk[None, :], M), jnp.where(r2 == r, mq[None, :], Aq))

    z = jnp.zeros((Lr, Lr), f32)
    M, Aq = lax.fori_loop(0, Lr, body, (z, z))
    return jnp.where(c2 < r2, M, 0.0), Aq


def _diag_scores_bwd(qt, kt, Gt, dM, dAq):
    """Backward of _diag_scores w.r.t. the x-side (dxk, dxq) and the shared
    y-side k (dy), excluding the exp(G) chain (handled by the caller)."""
    Lr, K = kt.shape
    rid = _iota((Lr, K), 0)

    def body(r, carry):
        dxk, dxq, dy = carry
        E = jnp.exp(jnp.where(rid <= r, _row(Gt, r)[None, :] - Gt, -jnp.inf))
        zk = _row(dM, r)[:, None]
        zq = _row(dAq, r)[:, None]
        kE = kt * E
        dxk = jnp.where(rid == r, jnp.sum(zk * kE, axis=0)[None, :], dxk)
        dxq = jnp.where(rid == r, jnp.sum(zq * kE, axis=0)[None, :], dxq)
        dy = dy + (zk * _row(kt, r)[None, :] + zq * _row(qt, r)[None, :]) * E
        return dxk, dxq, dy

    z = jnp.zeros((Lr, K), f32)
    return lax.fori_loop(0, Lr, body, (z, z, z))


def _intra(q, k, beta, G, base, end, S, prec):
    """Shared by prep fwd/bwd: scores, A = diag(beta) M, and T blocks."""
    kend = [k[j] * jnp.exp(end[j][None, :] - G[j]) for j in range(S)]
    rk = [k[i] * jnp.exp(G[i] - base[i][None, :]) for i in range(S)]
    rq = [q[i] * jnp.exp(G[i] - base[i][None, :]) for i in range(S)]
    ck, M, Aq = {}, {}, {}
    for i in range(S):
        for j in range(i):
            ck[i, j] = kend[j] * jnp.exp(base[i] - end[j])[None, :]
            M[i, j] = _mm(rk[i], ck[i, j].T, prec)
            Aq[i, j] = _mm(rq[i], ck[i, j].T, prec)
        M[i, i], Aq[i, i] = _diag_scores(q[i], k[i], G[i])
    A = {ij: beta[ij[0]][:, None] * m for ij, m in M.items()}
    T = _block_inverse(A, S, prec)
    return rk, rq, ck, M, Aq, T


# ----------------------------------------------------------------------------
# prep kernels
# ----------------------------------------------------------------------------
def _prep_fwd_kernel(q_ref, k_ref, v_ref, g_ref, b_ref, qg_ref, w_ref, u_ref, a_ref, kd_ref, gl_ref, *, prec):
    C = q_ref.shape[0]
    S = C // _L
    G, base, end = _gate_tiles(g_ref, S)
    Gl = end[-1]
    q = [_tile(q_ref, i) for i in range(S)]
    k = [_tile(k_ref, i) for i in range(S)]
    beta = [_vtile(b_ref, i) for i in range(S)]
    _, _, _, _, Aq, T = _intra(q, k, beta, G, base, end, S, prec)

    EG = [jnp.exp(G[i]) for i in range(S)]
    Rw = [beta[j][:, None] * k[j] * EG[j] for j in range(S)]
    Ru = [beta[j][:, None] * _tile(v_ref, j) for j in range(S)]
    for i in range(S):
        rows = pl.ds(i * _L, _L)
        W = _mm(T[i, 0], Rw[0], prec)
        U = _mm(T[i, 0], Ru[0], prec)
        for j in range(1, i + 1):
            W = W + _mm(T[i, j], Rw[j], prec)
            U = U + _mm(T[i, j], Ru[j], prec)
        w_ref[rows, :] = W.astype(w_ref.dtype)
        u_ref[rows, :] = U.astype(u_ref.dtype)
        qg_ref[rows, :] = (q[i] * EG[i]).astype(qg_ref.dtype)
        kd_ref[rows, :] = (k[i] * jnp.exp(Gl[None, :] - G[i])).astype(kd_ref.dtype)
        for j in range(S):
            blk = Aq[i, j] if j <= i else jnp.zeros((_L, _L), f32)
            a_ref[rows, pl.ds(j * _L, _L)] = blk.astype(a_ref.dtype)
    gl_ref[...] = jnp.exp(Gl)


def _prep_bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    b_ref,
    dqg_ref,
    dw_ref,
    du_ref,
    dkd_ref,
    da_ref,
    dgl_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dg_ref,
    db_ref,
    *,
    prec,
):
    NV, C, K = dqg_ref.shape
    S = C // _L
    G, base, end = _gate_tiles(g_ref, S)
    Gl = end[-1]
    q = [_tile(q_ref, i) for i in range(S)]
    k = [_tile(k_ref, i) for i in range(S)]
    v = [_tile(v_ref, i) for i in range(S)]
    beta = [_vtile(b_ref, i) for i in range(S)]
    rk, rq, ck, M, _, T = _intra(q, k, beta, G, base, end, S, prec)
    EG = [jnp.exp(G[i]) for i in range(S)]
    Rw = [beta[j][:, None] * k[j] * EG[j] for j in range(S)]
    Ru = [beta[j][:, None] * v[j] for j in range(S)]
    W, U = [], []
    for i in range(S):
        Wi, Ui = _mm(T[i, 0], Rw[0], prec), _mm(T[i, 0], Ru[0], prec)
        for j in range(1, i + 1):
            Wi = Wi + _mm(T[i, j], Rw[j], prec)
            Ui = Ui + _mm(T[i, j], Ru[j], prec)
        W.append(Wi)
        U.append(Ui)

    def psum(ref, i):  # sum the per-V-block partial gradients
        out = ref[0, pl.ds(i * _L, _L), :].astype(f32)
        for n in range(1, NV):
            out = out + ref[n, pl.ds(i * _L, _L), :].astype(f32)
        return out

    def dAq_blk(i, j):
        out = da_ref[0, pl.ds(i * _L, _L), pl.ds(j * _L, _L)].astype(f32)
        for n in range(1, NV):
            out = out + da_ref[n, pl.ds(i * _L, _L), pl.ds(j * _L, _L)].astype(f32)
        if i == j:
            out = jnp.where(_iota((_L, _L), 1) <= _iota((_L, _L), 0), out, 0.0)
        return out

    dW = [psum(dw_ref, i) for i in range(S)]
    dU = [_tile(du_ref, i) for i in range(S)]
    strict = _iota((_L, _L), 1) < _iota((_L, _L), 0)
    part_dk, part_dG = [], []
    acc = [jnp.zeros((_L, K), f32) for _ in range(S)]

    for i in range(S):
        rows = pl.ds(i * _L, _L)
        # dR = T^T dX  (block back-substitution)
        dRw = _mm(T[i, i].T, dW[i], prec)
        dRu = _mm(T[i, i].T, dU[i], prec)
        for m in range(i + 1, S):
            dRw = dRw + _mm(T[m, i].T, dW[m], prec)
            dRu = dRu + _mm(T[m, i].T, dU[m], prec)
        dbeta = jnp.zeros((_L,), f32)
        dxk_off = jnp.zeros((_L, K), f32)
        dxq_off = jnp.zeros((_L, K), f32)
        for j in range(i + 1):
            dA = -(_mm(dRw, W[j].T, prec) + _mm(dRu, U[j].T, prec))
            if j == i:
                dA = jnp.where(strict, dA, 0.0)
            dbeta = dbeta + jnp.sum(dA * M[i, j], axis=1)
            dM = beta[i][:, None] * dA
            dAq = dAq_blk(i, j)
            if j < i:
                dxk_off = dxk_off + _mm(dM, ck[i, j], prec)
                dxq_off = dxq_off + _mm(dAq, ck[i, j], prec)
                acc[j] = acc[j] + (_mm(dM.T, rk[i], prec) + _mm(dAq.T, rq[i], prec)) * jnp.exp(base[i] - end[j])[None, :]
            else:
                ddxk, ddxq, ddy = _diag_scores_bwd(q[i], k[i], G[i], dM, dAq)
        e_i = jnp.exp(G[i] - base[i][None, :])
        dxk = dxk_off * e_i + ddxk
        dxq = dxq_off * e_i + ddxq
        dqg = psum(dqg_ref, i)
        dq_ref[rows, :] = (dxq + dqg * EG[i]).astype(dq_ref.dtype)
        dv_ref[rows, :] = (beta[i][:, None] * dRu).astype(dv_ref.dtype)
        dbeta = dbeta + jnp.sum(dRw * k[i] * EG[i], axis=1) + jnp.sum(dRu * v[i], axis=1)
        db_ref[rows] = dbeta.astype(db_ref.dtype)
        part_dk.append(dxk + ddy + dRw * beta[i][:, None] * EG[i])
        part_dG.append(k[i] * dxk + q[i] * dxq - k[i] * ddy + dRw * Rw[i] + dqg * q[i] * EG[i])

    # gradient w.r.t. the chunk-total log decay (G_last)
    dgl = dgl_ref[0, :].astype(f32)
    for n in range(1, NV):
        dgl = dgl + dgl_ref[n, :].astype(f32)
    kd = [k[i] * jnp.exp(Gl[None, :] - G[i]) for i in range(S)]
    dkd = [psum(dkd_ref, i) for i in range(S)]
    dGl = dgl * jnp.exp(Gl)
    for i in range(S):
        dGl = dGl + jnp.sum(dkd[i] * kd[i], axis=0)

    suffix = jnp.zeros((K,), f32)
    for i in reversed(range(S)):
        rows = pl.ds(i * _L, _L)
        dy_off = jnp.exp(end[i][None, :] - G[i]) * acc[i]
        dk_ref[rows, :] = (part_dk[i] + dy_off + dkd[i] * jnp.exp(Gl[None, :] - G[i])).astype(dk_ref.dtype)
        dG = part_dG[i] - k[i] * dy_off - dkd[i] * kd[i]
        cs = jnp.cumsum(dG, axis=0)
        tot = _row(cs, _L - 1)
        # dg_t = sum_{r >= t} dG_r (+ everything that flows through G_last)
        dg_ref[rows, :] = (suffix[None, :] + tot[None, :] - cs + dG + dGl[None, :]).astype(dg_ref.dtype)
        suffix = suffix + tot


# ----------------------------------------------------------------------------
# recurrence kernels
# ----------------------------------------------------------------------------
def _rec_fwd_kernel(qg_ref, w_ref, u_ref, a_ref, kd_ref, gl_ref, s0_ref, o_ref, sT_ref, *maybe_h, prec, save_h):
    N, C, _ = qg_ref.shape

    def body(i, S):
        if save_h:
            maybe_h[0][i, :, :] = S.astype(maybe_h[0].dtype)
        mmd = qg_ref.dtype
        Sm = S.astype(mmd)
        U = u_ref[i, :, :].astype(f32) - _mm(w_ref[i, :, :], Sm, prec)
        Um = U.astype(mmd)
        o = _mm(qg_ref[i, :, :], Sm, prec) + _mm(a_ref[i, :, :], Um, prec)
        o_ref[pl.ds(pl.multiple_of(i * C, C), C), :] = o.astype(o_ref.dtype)
        gl = gl_ref[i, :].astype(f32)
        return gl[:, None] * S + _mm(kd_ref[i, :, :].T, Um, prec)

    sT_ref[...] = lax.fori_loop(0, N, body, s0_ref[...].astype(f32))


def _rec_bwd_kernel(
    qg_ref,
    w_ref,
    u_ref,
    a_ref,
    kd_ref,
    gl_ref,
    h_ref,
    do_ref,
    dsT_ref,
    dqg_ref,
    dw_ref,
    dkd_ref,
    da_ref,
    dgl_ref,
    du_ref,
    ds0_ref,
    *,
    prec,
):
    N, C, _ = qg_ref.shape
    mmd = qg_ref.dtype

    def body(j, dS):
        i = N - 1 - j
        Sm = h_ref[i, :, :]
        S = Sm.astype(f32)
        w = w_ref[i, :, :]
        dO = do_ref[pl.ds(pl.multiple_of(i * C, C), C), :]
        dOm = dO.astype(mmd)
        dSm = dS.astype(mmd)
        U = u_ref[i, :, :].astype(f32) - _mm(w, Sm, prec)
        Um = U.astype(mmd)
        dU = _mm(a_ref[i, :, :].T, dOm, prec) + _mm(kd_ref[i, :, :], dSm, prec)
        dUm = dU.astype(mmd)
        da_ref[i, :, :] = _mm(dOm, Um.T, prec)
        dkd_ref[i, :, :] = _mm(Um, dSm.T, prec)
        dgl_ref[i, :] = jnp.sum(S * dS, axis=1)
        dqg_ref[i, :, :] = _mm(dOm, Sm.T, prec)
        dw_ref[i, :, :] = -_mm(dUm, Sm.T, prec)
        du_ref[i, :, :] = dU
        gl = gl_ref[i, :].astype(f32)
        return gl[:, None] * dS + _mm(qg_ref[i, :, :].T, dOm, prec) - _mm(w.T, dUm, prec)

    ds0_ref[...] = lax.fori_loop(0, N, body, dsT_ref[...].astype(f32))


# ----------------------------------------------------------------------------
# pallas_call wrappers
# ----------------------------------------------------------------------------
def _prep_fwd(q, k, v, g, beta, cfg):
    B, Tp, H, K = q.shape
    V = v.shape[-1]
    C = cfg.chunk
    N = Tp // C
    mmd = jnp.dtype(cfg.mm_dtype)
    x = lambda D: pl.BlockSpec((None, C, None, D), lambda b, h, n: (b, n, h, 0))  # noqa: E731
    bs = pl.BlockSpec((None, C, None), lambda b, h, n: (b, n, h))
    co = lambda D: pl.BlockSpec((None, None, None, C, D), lambda b, h, n: (b, h, n, 0, 0))  # noqa: E731
    glo = pl.BlockSpec((None, None, None, K), lambda b, h, n: (b, h, n, 0))
    sds = lambda D, dt: jax.ShapeDtypeStruct((B, H, N, C, D), dt)  # noqa: E731
    return pl.pallas_call(
        functools.partial(_prep_fwd_kernel, prec=_precision(f32, cfg.interpret)),
        grid=(B, H, N),
        in_specs=[x(K), x(K), x(V), x(K), bs],
        out_specs=[co(K), co(K), co(V), co(C), co(K), glo],
        out_shape=[
            sds(K, mmd),
            sds(K, mmd),
            sds(V, f32),
            sds(C, mmd),
            sds(K, mmd),
            jax.ShapeDtypeStruct((B, H, N, K), f32),
        ],
        compiler_params=_TritonParams(num_warps=cfg.warps_prep, num_stages=1),
        interpret=cfg.interpret,
        name="kda_prep_fwd",
    )(q, k, v, g, beta)


def _prep_bwd(q, k, v, g, beta, dqg, dw, du, dkd, da, dgl, cfg):
    B, Tp, H, K = q.shape
    V = v.shape[-1]
    C = cfg.chunk
    NV = dqg.shape[0]
    x = lambda D: pl.BlockSpec((None, C, None, D), lambda b, h, n: (b, n, h, 0))  # noqa: E731
    bs = pl.BlockSpec((None, C, None), lambda b, h, n: (b, n, h))
    part = lambda D: pl.BlockSpec((NV, None, None, None, C, D), lambda b, h, n: (0, b, h, n, 0, 0))  # noqa: E731
    ci = lambda D: pl.BlockSpec((None, None, None, C, D), lambda b, h, n: (b, h, n, 0, 0))  # noqa: E731
    glp = pl.BlockSpec((NV, None, None, None, K), lambda b, h, n: (0, b, h, n, 0))
    return pl.pallas_call(
        functools.partial(_prep_bwd_kernel, prec=_precision(f32, cfg.interpret)),
        grid=(B, H, Tp // C),
        in_specs=[x(K), x(K), x(V), x(K), bs, part(K), part(K), ci(V), part(K), part(C), glp],
        out_specs=[x(K), x(K), x(V), x(K), bs],
        out_shape=[
            jax.ShapeDtypeStruct((B, Tp, H, K), f32),
            jax.ShapeDtypeStruct((B, Tp, H, K), f32),
            jax.ShapeDtypeStruct((B, Tp, H, V), f32),
            jax.ShapeDtypeStruct((B, Tp, H, K), f32),
            jax.ShapeDtypeStruct((B, Tp, H), f32),
        ],
        compiler_params=_TritonParams(num_warps=cfg.warps_prep, num_stages=1),
        interpret=cfg.interpret,
        name="kda_prep_bwd",
    )(q, k, v, g, beta, dqg, dw, du, dkd, da, dgl)


def _rec_fwd(qg, w, u, A, kd, gl, s0, out_dtype, Tp, save_h, cfg):
    B, H, N, C, K = qg.shape
    V = u.shape[-1]
    bv = min(cfg.block_v, V)
    NV = V // bv
    ck = pl.BlockSpec((None, None, N, C, K), lambda b, h, v: (b, h, 0, 0, 0))
    cv = pl.BlockSpec((None, None, N, C, bv), lambda b, h, v: (b, h, 0, 0, v))
    cc = pl.BlockSpec((None, None, N, C, C), lambda b, h, v: (b, h, 0, 0, 0))
    nk = pl.BlockSpec((None, None, N, K), lambda b, h, v: (b, h, 0, 0))
    kv = pl.BlockSpec((None, None, K, bv), lambda b, h, v: (b, h, 0, v))
    tv = pl.BlockSpec((None, Tp, None, bv), lambda b, h, v: (b, 0, h, v))
    out_specs = [tv, kv]
    out_shape = [jax.ShapeDtypeStruct((B, Tp, H, V), out_dtype), jax.ShapeDtypeStruct((B, H, K, V), f32)]
    if save_h:
        out_specs.append(pl.BlockSpec((None, None, N, K, bv), lambda b, h, v: (b, h, 0, 0, v)))
        out_shape.append(jax.ShapeDtypeStruct((B, H, N, K, V), qg.dtype))
    outs = pl.pallas_call(
        functools.partial(_rec_fwd_kernel, save_h=save_h, prec=_precision(qg.dtype, cfg.interpret)),
        grid=(B, H, NV),
        in_specs=[ck, ck, cv, cc, ck, nk, kv],
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=_TritonParams(num_warps=cfg.warps_rec, num_stages=cfg.stages),
        interpret=cfg.interpret,
        name="kda_rec_fwd",
    )(qg, w, u, A, kd, gl, s0)
    return outs if save_h else (*outs, None)


def _rec_bwd(qg, w, u, A, kd, gl, h, do, dsT, cfg):
    B, H, N, C, K = qg.shape
    V = u.shape[-1]
    Tp = do.shape[1]
    bv = min(cfg.block_v_bwd, V)
    NV = V // bv
    ck = pl.BlockSpec((None, None, N, C, K), lambda b, h, v: (b, h, 0, 0, 0))
    cv = pl.BlockSpec((None, None, N, C, bv), lambda b, h, v: (b, h, 0, 0, v))
    cc = pl.BlockSpec((None, None, N, C, C), lambda b, h, v: (b, h, 0, 0, 0))
    nk = pl.BlockSpec((None, None, N, K), lambda b, h, v: (b, h, 0, 0))
    kv = pl.BlockSpec((None, None, K, bv), lambda b, h, v: (b, h, 0, v))
    hv = pl.BlockSpec((None, None, N, K, bv), lambda b, h, v: (b, h, 0, 0, v))
    tv = pl.BlockSpec((None, Tp, None, bv), lambda b, h, v: (b, 0, h, v))
    pk = lambda D: pl.BlockSpec((None, None, None, N, C, D), lambda b, h, v: (v, b, h, 0, 0, 0))  # noqa: E731
    pgl = pl.BlockSpec((None, None, None, N, K), lambda b, h, v: (v, b, h, 0, 0))
    part = lambda D: jax.ShapeDtypeStruct((NV, B, H, N, C, D), f32)  # noqa: E731
    return pl.pallas_call(
        functools.partial(_rec_bwd_kernel, prec=_precision(qg.dtype, cfg.interpret)),
        grid=(B, H, NV),
        in_specs=[ck, ck, cv, cc, ck, nk, hv, tv, kv],
        out_specs=[pk(K), pk(K), pk(K), pk(C), pgl, cv, kv],
        out_shape=[
            part(K),
            part(K),
            part(K),
            part(C),
            jax.ShapeDtypeStruct((NV, B, H, N, K), f32),
            jax.ShapeDtypeStruct((B, H, N, C, V), f32),
            jax.ShapeDtypeStruct((B, H, K, V), f32),
        ],
        compiler_params=_TritonParams(num_warps=cfg.warps_rec, num_stages=cfg.stages),
        interpret=cfg.interpret,
        name="kda_rec_bwd",
    )(qg, w, u, A, kd, gl, h, do, dsT)


# ----------------------------------------------------------------------------
# custom VJP core
# ----------------------------------------------------------------------------
def _core_impl(q, k, v, g, beta, s0, cfg, save):
    prep = _prep_fwd(q, k, v, g, beta, cfg)
    o, sT, h = _rec_fwd(*prep, s0, v.dtype, q.shape[1], save, cfg)
    return o, sT, prep, h


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _kda_core(q, k, v, g, beta, s0, cfg):
    o, sT, _, _ = _core_impl(q, k, v, g, beta, s0, cfg, save=False)
    return o, sT


def _kda_core_fwd(q, k, v, g, beta, s0, cfg):
    o, sT, prep, h = _core_impl(q, k, v, g, beta, s0, cfg, save=True)
    return (o, sT), (q, k, v, g, beta, prep, h)


def _kda_core_bwd(cfg, res, cts):
    q, k, v, g, beta, prep, h = res
    do, dsT = cts
    qg, w, u, A, kd, gl = prep
    dqg, dw, dkd, da, dgl, du, ds0 = _rec_bwd(qg, w, u, A, kd, gl, h, do, dsT.astype(f32), cfg)
    dq, dk, dv, dg, db = _prep_bwd(q, k, v, g, beta, dqg, dw, du, dkd, da, dgl, cfg)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), dg.astype(g.dtype), db.astype(beta.dtype), ds0)


_kda_core.defvjp(_kda_core_fwd, _kda_core_bwd)


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def _l2norm(x, eps=1e-6):
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm=False,
    chunk_size=64,
    mm_dtype=None,
    block_v=64,
    block_v_bwd=64,
    num_warps_prep=8,
    num_warps_rec=4,
    num_stages=2,
    interpret=False,
):
    """Chunked Kimi Delta Attention (forward + custom VJP).

    Args:
      q, k: (B, T, H, K).  v: (B, T, H, V).  g: (B, T, H, K) log-decay <= 0
        (any magnitude; no clamping needed).  beta: (B, T, H).
      initial_state: optional (B, H, K, V).
      mm_dtype: operand dtype for the recurrence matmuls (fp32 accumulate).
        Defaults to bf16 for bf16/fp16 inputs, else fp32 (3xTF32 on GPU).
        Intra-chunk prep is always fp32.
      block_v / block_v_bwd: V tile per program for the recurrence kernels.
      interpret: run the kernels in the Pallas interpreter (CPU debugging).
    K and V must be powers of two >= 16; chunk_size a multiple of 16.
    Returns o (B, T, H, V) in v.dtype [, final state (B, H, K, V) fp32].
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    for n, d in (("K", K), ("V", V)):
        assert d >= 16 and d & (d - 1) == 0, f"{n}={d} must be a power of 2 >= 16"
    assert C % _L == 0
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if mm_dtype is None:
        mm_dtype = jnp.bfloat16 if q.dtype in (jnp.bfloat16, jnp.float16) else f32
    cfg = _Cfg(C, jnp.dtype(mm_dtype).name, block_v, block_v_bwd, num_warps_prep, num_warps_rec, num_stages, interpret)

    qf, kf = q.astype(f32), k.astype(f32)
    if use_qk_l2norm:
        qf, kf = _l2norm(qf), _l2norm(kf)
    q, k = (qf * scale).astype(q.dtype), kf.astype(k.dtype)
    g, beta = g.astype(f32), beta.astype(f32)

    pad = (-T) % C  # zero padding: k = 0 (no write), g = 0 (no decay)
    if pad:
        pw = ((0, 0), (0, pad), (0, 0), (0, 0))
        q, k, v, g = (jnp.pad(x, pw) for x in (q, k, v, g))
        beta = jnp.pad(beta, pw[:3])
    s0 = jnp.zeros((B, H, K, V), f32) if initial_state is None else initial_state.astype(f32)

    o, sT = _kda_core(q, k, v, g, beta, s0, cfg)
    if pad:
        o = o[:, :T]
    return (o, sT) if output_final_state else o


def kda_reference(q, k, v, g, beta, *, scale=None, initial_state=None, use_qk_l2norm=False):
    """Naive token-by-token fp32 recurrence, for testing."""
    B, _T, H, K = q.shape
    V = v.shape[-1]
    scale = 1.0 / math.sqrt(K) if scale is None else scale
    q, k, v, g, beta = (x.astype(f32) for x in (q, k, v, g, beta))
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    q = q * scale
    S0 = jnp.zeros((B, H, K, V), f32) if initial_state is None else initial_state.astype(f32)

    def step(S, xs):
        qt, kt, vt, gt, bt = xs
        S = jnp.exp(gt)[..., None] * S
        pred = jnp.einsum("bhk,bhkv->bhv", kt, S, precision="highest")
        S = S + bt[..., None, None] * kt[..., None] * (vt - pred)[..., None, :]
        return S, jnp.einsum("bhk,bhkv->bhv", qt, S, precision="highest")

    S, o = lax.scan(step, S0, tuple(x.swapaxes(0, 1) for x in (q, k, v, g, beta)))
    return o.swapaxes(0, 1), S
