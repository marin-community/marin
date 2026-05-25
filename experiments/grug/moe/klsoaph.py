# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# KL SOAP H: SOAP-style Hessian-eigenbasis preconditioner with Adam moments
# in the eigenbasis. The "KL" qualifier refers to the scale-invariant
# ("hyperball") post-step applied in optimizer.py:_scale_invariant_hyperball_updates,
# not a KL-divergence projection. Reproduces the KLSOAPH optimizer from
# KellerJordan/modded-nanogpt PR #290 — including the whitened Gram update,
# eigen_sqrt_inv state, first-gradient initialization, post-EMA symmetrize,
# descending-eigenvalue flip at init, and QR-iteration warm-started basis
# refresh.
#
# The single intentional deviation from upstream is the default ``precond_freq``
# value: upstream uses 1, we ship 5 to amortize the per-step refresh cost on TPU.
# All other behaviour matches upstream.
#
# scale_by_klsoaph returns only the SOAP direction; the hyperball post-step
# is applied by scale_with_grug_klsoaph in optimizer.py, mirroring the
# muonh/normuonh pattern.

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard


class ScaleByKLSoapHState(NamedTuple):
    """Per-leaf SOAP state. Float32 throughout for eigh / qr stability."""

    count: chex.Array
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    gg_l: optax.Updates
    gg_r: optax.Updates
    q_l: optax.Updates
    q_r: optax.Updates
    esi_l: optax.Updates  # eigen_sqrt_inv for the row axis, shape (..., rows)
    esi_r: optax.Updates  # eigen_sqrt_inv for the column axis, shape (..., cols)


class _SoapStepResult:
    """Plain-Python wrapper for the nine outputs of one SOAP step.

    Not a registered pytree, so jax.tree.map treats it as a leaf.
    """

    __slots__ = ("direction", "esi_l", "esi_r", "exp_avg", "exp_avg_sq", "gg_l", "gg_r", "q_l", "q_r")

    def __init__(self, direction, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r):
        self.direction = direction
        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.gg_l = gg_l
        self.gg_r = gg_r
        self.q_l = q_l
        self.q_r = q_r
        self.esi_l = esi_l
        self.esi_r = esi_r


def _is_matrix_param(p) -> bool:
    return hasattr(p, "ndim") and p.ndim >= 2


def _zeros_param(p):
    if not _is_matrix_param(p):
        return None
    return jnp.zeros(p.shape, dtype=jnp.float32)


def _zeros_left(p):
    if not _is_matrix_param(p):
        return None
    n = p.shape[-2]
    return jnp.zeros((*p.shape[:-2], n, n), dtype=jnp.float32)


def _zeros_right(p):
    if not _is_matrix_param(p):
        return None
    n = p.shape[-1]
    return jnp.zeros((*p.shape[:-2], n, n), dtype=jnp.float32)


def _eye_leading(p, axis: int):
    if not _is_matrix_param(p):
        return None
    n = p.shape[axis]
    eye = jnp.eye(n, dtype=jnp.float32)
    if p.ndim > 2:
        eye = jnp.broadcast_to(eye, (*p.shape[:-2], n, n))
    return jnp.asarray(eye, dtype=jnp.float32)


def _esi_init(p, axis: int, init_factor: float):
    if not _is_matrix_param(p):
        return None
    n = p.shape[axis]
    # eigen_sqrt_inv starts at init_factor**-0.5 (upstream init_factor=0.1 → sqrt(10) ≈ 3.162).
    return jnp.full((*p.shape[:-2], n), init_factor**-0.5, dtype=jnp.float32)


def _replicated_pspec(ndim: int) -> P:
    """PartitionSpec with `ndim` None entries (fully replicated)."""
    return P(*(None,) * ndim)


def _flipped_eigh(matrix):
    """eigh on a symmetric matrix and return eigenvectors in DESCENDING order.

    Matches upstream ``_initial_orthogonal_matrix`` which does
    ``torch.linalg.eigh`` then ``torch.flip(q, dims=[1])``. The initial Gram
    is regularized with ``1e-30 * I`` to match upstream guard.

    Operates on the last two axes; batches automatically over leading axes
    when the input is replicated across the mesh.
    """
    n = matrix.shape[-1]
    eye = jnp.eye(n, dtype=jnp.float32)
    if matrix.ndim > 2:
        eye = jnp.broadcast_to(eye, matrix.shape)
    sym = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    _, q = jnp.linalg.eigh(sym + 1e-30 * eye)
    # Flip eigenvectors along the column axis to get descending order.
    return q[..., :, ::-1]


def _symmetrize(matrix):
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _klsoaph_step(
    grad: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    gg_l: jnp.ndarray,
    gg_r: jnp.ndarray,
    q_l: jnp.ndarray,
    q_r: jnp.ndarray,
    esi_l: jnp.ndarray,
    esi_r: jnp.ndarray,
    step: jnp.ndarray,
    beta1: float,
    beta2: float,
    shampoo_beta: float,
    eps: float,
    precond_freq: int,
    init_factor: float,
):
    """One SOAP step. Works for 2-D ``(rows, cols)`` gradients and 3-D
    ``(stack, rows, cols)`` gradients via einsum ellipsis.

    Replicates every intermediate fully across the mesh so contracting-dim
    sharding ambiguities cannot occur (matching the legacy MuonH replicated
    Newton-Schulz pattern in ``levanter.optim.grugmuon``). The caller's
    parameter-sharding is restored downstream by
    ``_match_named_update_sharding`` in ``optimizer.py``.

    The first call (``step == 1``) takes the "init" branch: it derives the
    initial Gram matrices, Q bases, and eigen_sqrt_inv from the first
    gradient (matching upstream ``init_2d_klsoap_state_``) and returns a
    zero direction — i.e. the first call does not move the parameter.
    """
    rows = grad.shape[-2]
    cols = grad.shape[-1]

    has_mesh = not jax.sharding.get_abstract_mesh().empty
    out_pspec = _replicated_pspec(grad.ndim) if has_mesh else None
    out_pspec_l = _replicated_pspec(grad.ndim) if has_mesh else None  # (..., rows, rows)
    out_pspec_r = _replicated_pspec(grad.ndim) if has_mesh else None  # (..., cols, cols)

    g32 = grad.astype(jnp.float32)
    if has_mesh:
        repl = _replicated_pspec(grad.ndim)
        g32 = reshard(g32, repl)
        gg_l = reshard(gg_l, repl)
        gg_r = reshard(gg_r, repl)
        q_l = reshard(q_l, repl)
        q_r = reshard(q_r, repl)
        exp_avg = reshard(exp_avg, repl)
        exp_avg_sq = reshard(exp_avg_sq, repl)
        # esi has one fewer trailing axis.
        esi_repl = _replicated_pspec(grad.ndim - 1)
        esi_l = reshard(esi_l, esi_repl)
        esi_r = reshard(esi_r, esi_repl)

    # --- First-call init branch (matches upstream init_2d_klsoap_state_) ----
    def _init_branch(_):
        # GG = [g g.T / cols, g.T g / rows].
        new_gg_l = jnp.einsum("...ik,...jk->...ij", g32, g32, out_sharding=out_pspec_l) / cols
        new_gg_r = jnp.einsum("...ki,...kj->...ij", g32, g32, out_sharding=out_pspec_r) / rows
        new_gg_l = _symmetrize(new_gg_l)
        new_gg_r = _symmetrize(new_gg_r)
        # Q = flipped eigh of initial Gram (descending eigenvalues).
        new_q_l = _flipped_eigh(new_gg_l)
        new_q_r = _flipped_eigh(new_gg_r)
        # eigen_sqrt_inv at init_factor**-0.5.
        new_esi_l = jnp.full_like(esi_l, init_factor**-0.5)
        new_esi_r = jnp.full_like(esi_r, init_factor**-0.5)
        # No parameter update on the first call.
        zero_dir = jnp.zeros_like(g32)
        return (
            zero_dir,
            jnp.zeros_like(exp_avg),  # exp_avg
            jnp.zeros_like(exp_avg_sq),  # exp_avg_sq
            new_gg_l,
            new_gg_r,
            new_q_l,
            new_q_r,
            new_esi_l,
            new_esi_r,
        )

    # --- Normal branch ----
    def _normal_branch(_):
        # 1. Project gradient via current Q: g_proj = q_l.T @ g @ q_r.
        g_qr = jnp.einsum("...ij,...jk->...ik", g32, q_r, out_sharding=out_pspec)
        g_proj = jnp.einsum("...ki,...kj->...ij", q_l, g_qr, out_sharding=out_pspec)

        # 2. Adam in projected basis.
        new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g_proj
        new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * g_proj * g_proj
        precond_proj = new_exp_avg / (jnp.sqrt(new_exp_avg_sq) + eps)

        # 3. Direction = q_l @ precond_proj @ q_r.T.
        p_qrt = jnp.einsum("...ij,...kj->...ik", precond_proj, q_r, out_sharding=out_pspec)
        direction = jnp.einsum("...ij,...jk->...ik", q_l, p_qrt, out_sharding=out_pspec)

        # 4. Whitened Gram update (matches upstream update_2d_klsoap_preconditioner_):
        #    left_target  = (g @ q_r * esi_r[None,:]) @ (g @ q_r * esi_r[None,:]).T / cols
        #    right_target = (q_l.T @ g * esi_l[:,None]).T @ (q_l.T @ g * esi_l[:,None]) / rows
        g_qr_white = g_qr * esi_r[..., None, :]  # (..., rows, cols)
        left_target = jnp.einsum("...ik,...jk->...ij", g_qr_white, g_qr_white, out_sharding=out_pspec_l) / cols
        qlT_g = jnp.einsum("...ki,...kj->...ij", q_l, g32, out_sharding=out_pspec)
        qlT_g_white = qlT_g * esi_l[..., :, None]  # (..., rows, cols)
        right_target = jnp.einsum("...ki,...kj->...ij", qlT_g_white, qlT_g_white, out_sharding=out_pspec_r) / rows
        new_gg_l = _symmetrize(shampoo_beta * gg_l + (1.0 - shampoo_beta) * left_target)
        new_gg_r = _symmetrize(shampoo_beta * gg_r + (1.0 - shampoo_beta) * right_target)

        # 5. ESI update (matches upstream _update_eigen_sqrt_inv_).
        #    left_diag  = (g_proj * esi_r[None,:]).square().mean(axis=-1)  shape (..., rows)
        #    right_diag = (g_proj * esi_l[:,None]).square().mean(axis=-2)  shape (..., cols)
        proj_col_white = g_proj * esi_r[..., None, :]
        left_diag = jnp.mean(proj_col_white * proj_col_white, axis=-1)
        proj_row_white = g_proj * esi_l[..., :, None]
        right_diag = jnp.mean(proj_row_white * proj_row_white, axis=-2)

        def _update_esi(esi, diag):
            old_eigen = jnp.reciprocal(jnp.square(esi))
            old_eigen = jnp.nan_to_num(old_eigen, nan=0.0, posinf=0.0, neginf=0.0)
            eigen = shampoo_beta * old_eigen + (1.0 - shampoo_beta) * diag
            inv_sqrt = jnp.minimum(jax.lax.rsqrt(jnp.maximum(eigen, 1e-30)), 4000.0)
            return jnp.nan_to_num(inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)

        new_esi_l = _update_esi(esi_l, left_diag)
        new_esi_r = _update_esi(esi_r, right_diag)

        # 6. Refresh basis every precond_freq steps via QR iteration on GG @ Q,
        #    and reproject exp_avg from the old basis into the new basis.
        should_refresh = jnp.equal(jnp.remainder(step, precond_freq), 0)

        def _refresh(_):
            # exp_avg lives in the OLD basis; project back to parameter space.
            ea_qrT = jnp.einsum("...ij,...kj->...ik", new_exp_avg, q_r, out_sharding=out_pspec)  # exp_avg @ q_r.T
            ea_original = jnp.einsum("...ij,...jk->...ik", q_l, ea_qrT, out_sharding=out_pspec)  # q_l @ ea_qrT
            # One step of subspace (QR) iteration warm-started from current Q.
            gg_l_q = jnp.einsum("...ij,...jk->...ik", new_gg_l, q_l, out_sharding=out_pspec_l)
            gg_r_q = jnp.einsum("...ij,...jk->...ik", new_gg_r, q_r, out_sharding=out_pspec_r)
            ql_new, _ = jnp.linalg.qr(gg_l_q)
            qr_new, _ = jnp.linalg.qr(gg_r_q)
            # Reproject exp_avg into the NEW basis.
            ea_qr_new = jnp.einsum("...ij,...jk->...ik", ea_original, qr_new, out_sharding=out_pspec)  # ea @ q_r_new
            ea_new = jnp.einsum("...ki,...kj->...ij", ql_new, ea_qr_new, out_sharding=out_pspec)  # q_l_new.T @ ...
            return ql_new, qr_new, ea_new

        def _keep(_):
            return q_l, q_r, new_exp_avg

        ql_out, qr_out, ea_out = jax.lax.cond(should_refresh, _refresh, _keep, operand=None)

        return (
            direction,
            ea_out,
            new_exp_avg_sq,
            new_gg_l,
            new_gg_r,
            ql_out,
            qr_out,
            new_esi_l,
            new_esi_r,
        )

    is_first = jnp.equal(step, 1)
    direction, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r = jax.lax.cond(
        is_first, _init_branch, _normal_branch, operand=None
    )
    return direction.astype(grad.dtype), exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r


def scale_by_klsoaph(
    beta1: float = 0.95,
    beta2: float = 0.9,
    shampoo_beta: float = 0.9,
    eps: float = 1e-8,
    precond_freq: int = 5,
    init_factor: float = 0.1,
) -> optax.GradientTransformation:
    """SOAP-style preconditioner reproducing upstream KLSOAPH.

    Returns the direction in parameter space. No learning-rate scaling and no
    hyperball normalization — both applied by scale_with_grug_klsoaph.

    State is allocated only for leaves with ndim >= 2; lower-rank leaves get
    None state slots and pass through unchanged (their gradients should be
    routed to a different optimizer group via an optax mask). Leaves with
    ndim > 2 are treated as stacks of (rows, cols) matrices over the leading
    axes via einsum ellipsis.

    Default hyperparameters match the upstream "passing" tuple
    (beta1=0.95, beta2=0.9, shampoo_beta=0.9) at the modded-nanogpt scale
    (PR #290). ``precond_freq`` defaults to 5 (upstream 1) to amortize the
    per-step QR-iteration refresh on TPU; everything else matches upstream
    bit-for-bit including the whitened Gram update, eigen_sqrt_inv state,
    descending-eigh first-step init, post-EMA symmetrize, and warm-started
    QR refresh with exp_avg reprojection.
    """

    def init_fn(params):
        # All state tensors are placeholders for non-matrix leaves (None) or
        # zeros / eye / init-factor for matrix leaves. On the first call to
        # update_fn the matrix state is overwritten from the first gradient
        # (matching upstream init_2d_klsoap_state_).
        return ScaleByKLSoapHState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=jax.tree.map(_zeros_param, params, is_leaf=lambda x: x is None),
            exp_avg_sq=jax.tree.map(_zeros_param, params, is_leaf=lambda x: x is None),
            gg_l=jax.tree.map(_zeros_left, params, is_leaf=lambda x: x is None),
            gg_r=jax.tree.map(_zeros_right, params, is_leaf=lambda x: x is None),
            q_l=jax.tree.map(lambda p: _eye_leading(p, -2), params, is_leaf=lambda x: x is None),
            q_r=jax.tree.map(lambda p: _eye_leading(p, -1), params, is_leaf=lambda x: x is None),
            esi_l=jax.tree.map(lambda p: _esi_init(p, -2, init_factor), params, is_leaf=lambda x: x is None),
            esi_r=jax.tree.map(lambda p: _esi_init(p, -1, init_factor), params, is_leaf=lambda x: x is None),
        )

    def update_fn(updates, state, params=None):
        del params
        next_count = optax.safe_increment(state.count)

        def per_leaf(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r):
            if grad is None or exp_avg is None:
                return _SoapStepResult(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r)
            out = _klsoaph_step(
                grad,
                exp_avg,
                exp_avg_sq,
                gg_l,
                gg_r,
                q_l,
                q_r,
                esi_l,
                esi_r,
                next_count,
                beta1=beta1,
                beta2=beta2,
                shampoo_beta=shampoo_beta,
                eps=eps,
                precond_freq=precond_freq,
                init_factor=init_factor,
            )
            return _SoapStepResult(*out)

        results = jax.tree.map(
            per_leaf,
            updates,
            state.exp_avg,
            state.exp_avg_sq,
            state.gg_l,
            state.gg_r,
            state.q_l,
            state.q_r,
            state.esi_l,
            state.esi_r,
            is_leaf=lambda x: x is None,
        )

        # Treat _SoapStepResult as a leaf on the way out.
        def _is_result_or_none(x):
            return x is None or isinstance(x, _SoapStepResult)

        directions = jax.tree.map(lambda r: None if r is None else r.direction, results, is_leaf=_is_result_or_none)
        new_state = ScaleByKLSoapHState(
            count=next_count,
            exp_avg=jax.tree.map(lambda r: None if r is None else r.exp_avg, results, is_leaf=_is_result_or_none),
            exp_avg_sq=jax.tree.map(lambda r: None if r is None else r.exp_avg_sq, results, is_leaf=_is_result_or_none),
            gg_l=jax.tree.map(lambda r: None if r is None else r.gg_l, results, is_leaf=_is_result_or_none),
            gg_r=jax.tree.map(lambda r: None if r is None else r.gg_r, results, is_leaf=_is_result_or_none),
            q_l=jax.tree.map(lambda r: None if r is None else r.q_l, results, is_leaf=_is_result_or_none),
            q_r=jax.tree.map(lambda r: None if r is None else r.q_r, results, is_leaf=_is_result_or_none),
            esi_l=jax.tree.map(lambda r: None if r is None else r.esi_l, results, is_leaf=_is_result_or_none),
            esi_r=jax.tree.map(lambda r: None if r is None else r.esi_r, results, is_leaf=_is_result_or_none),
        )
        return directions, new_state

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByKLSoapHState", "scale_by_klsoaph"]
