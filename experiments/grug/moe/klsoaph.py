# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# KL SOAP H: full-matrix SOAP-style Hessian-eigenbasis preconditioner with Adam
# moments in the eigenbasis. Faithful port of the upstream KLSOAPH from
# KellerJordan/modded-nanogpt PR #290, run on the full per-leaf (rows, cols)
# matrix (no block tiling). For a stacked weight (e.g. an expert tensor with a
# leading (E,) axis), the leading axes are batched and full-matrix SOAP runs
# per slice via einsum-ellipsis; the row-Gram is (rows, rows) and the col-Gram
# is (cols, cols).
#
# Algorithm follows upstream KLSOAPH exactly:
#  1. First call: initialize Q from descending-eigh of (g g.T / cols,
#     g.T g / rows) computed on the first gradient; esi = init_factor**-0.5.
#     First call returns zero direction.
#  2. Subsequent calls: project gradient via current Q, Adam in projected
#     basis, build direction.
#  3. Whitened Gram update: gg_l += (g @ q_r * esi_r)^T @ (g @ q_r * esi_r) / cols;
#     symmetric counterpart for gg_r. Update esi via projected-gradient
#     diagonals.
#  4. Refresh every ``precond_freq`` steps: warm-started QR iteration
#     (``qr(GG @ Q)``) and reproject exp_avg through old->new basis.
#
# scale_by_klsoaph returns only the SOAP direction (full parameter
# shape). The hyperball post-step is applied by ``scale_with_grug_klsoaph``
# in ``optimizer.py``, mirroring the muonh / normuonh pattern.

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard


class ScaleByKLSoapHState(NamedTuple):
    """Per-leaf full-matrix SOAP state. Float32 throughout for eigh / qr stability.

    Tensor shapes for a parameter ``p`` with shape ``(..., rows, cols)``:

      exp_avg, exp_avg_sq:    (..., rows, cols)
      gg_l, q_l:              (..., rows, rows)   -- row-Gram / its eigenbasis
      gg_r, q_r:              (..., cols, cols)   -- col-Gram / its eigenbasis
      esi_l:                  (..., rows)
      esi_r:                  (..., cols)
    """

    count: chex.Array
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    gg_l: optax.Updates
    gg_r: optax.Updates
    q_l: optax.Updates
    q_r: optax.Updates
    esi_l: optax.Updates
    esi_r: optax.Updates


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


# Full-matrix SOAP state initializers. For a leaf p with shape (..., rows, cols):
#   exp_avg, exp_avg_sq : (..., rows, cols)
#   gg_l, q_l           : (..., rows, rows)   row-Gram / its eigenbasis
#   gg_r, q_r           : (..., cols, cols)   col-Gram / its eigenbasis
#   esi_l               : (..., rows)
#   esi_r               : (..., cols)
# Leading axes (e.g. the expert axis of a stacked (E, d, m) weight) are batch.
def _zeros_mn(p):
    if not _is_matrix_param(p):
        return None
    return jnp.zeros(p.shape, dtype=jnp.float32)


def _zeros_square(p, axis: int):
    """zeros (..., n, n) where n = p.shape[axis] (axis -2 = rows, -1 = cols)."""
    if not _is_matrix_param(p):
        return None
    n = p.shape[axis]
    return jnp.zeros((*p.shape[:-2], n, n), dtype=jnp.float32)


def _eye_square(p, axis: int):
    if not _is_matrix_param(p):
        return None
    n = p.shape[axis]
    eye = jnp.eye(n, dtype=jnp.float32)
    return jnp.broadcast_to(eye, (*p.shape[:-2], n, n)).astype(jnp.float32)


def _esi_init(p, axis: int, init_factor: float):
    if not _is_matrix_param(p):
        return None
    n = p.shape[axis]
    return jnp.full((*p.shape[:-2], n), init_factor**-0.5, dtype=jnp.float32)


def _replicated_pspec(ndim: int) -> P:
    """PartitionSpec with `ndim` None entries (fully replicated)."""
    return P(*(None,) * ndim)


def _flipped_eigh(matrix):
    """eigh on a symmetric matrix; return eigenvectors in DESCENDING order.

    Matches upstream ``_initial_orthogonal_matrix``: regularize the Gram by
    ``1e-30 * I``, eigh on the symmetric part, flip eigenvectors along the
    last axis. Operates per leading axis under broadcasting.
    """
    n = matrix.shape[-1]
    eye = jnp.eye(n, dtype=jnp.float32)
    if matrix.ndim > 2:
        eye = jnp.broadcast_to(eye, matrix.shape)
    sym = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    _, q = jnp.linalg.eigh(sym + 1e-30 * eye)
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
    """Run one full-matrix SOAP step over the (..., rows, cols) gradient.

    Every leading axis is treated independently; eigh / qr operate on the
    trailing (rows, rows) / (cols, cols) Grams only. Replicates all
    intermediates across the mesh to keep contracting-dim sharding
    unambiguous; the parameter-sharding for the externally-visible direction
    is restored downstream by ``_match_named_update_sharding`` in ``optimizer.py``.
    """
    # The trailing two dims of g32 are the actual (rows, cols); gg_l/q_l are
    # (rows, rows), gg_r/q_r are (cols, cols).
    inner_rows = grad.shape[-2]
    inner_cols = grad.shape[-1]

    has_mesh = not jax.sharding.get_abstract_mesh().empty
    g32 = grad.astype(jnp.float32)
    if has_mesh:
        repl = _replicated_pspec(g32.ndim)
        g32 = reshard(g32, repl)
        gg_l = reshard(gg_l, repl)
        gg_r = reshard(gg_r, repl)
        q_l = reshard(q_l, repl)
        q_r = reshard(q_r, repl)
        exp_avg = reshard(exp_avg, repl)
        exp_avg_sq = reshard(exp_avg_sq, repl)
        esi_repl = _replicated_pspec(esi_l.ndim)
        esi_l = reshard(esi_l, esi_repl)
        esi_r = reshard(esi_r, esi_repl)
    out_p = _replicated_pspec(g32.ndim) if has_mesh else None

    def _init_branch(_):
        # GG = [g g.T / cols, g.T g / rows] on the trailing (rows, cols) of each leaf.
        new_gg_l = jnp.einsum("...ik,...jk->...ij", g32, g32, out_sharding=out_p) / inner_cols
        new_gg_r = jnp.einsum("...ki,...kj->...ij", g32, g32, out_sharding=out_p) / inner_rows
        new_gg_l = _symmetrize(new_gg_l)
        new_gg_r = _symmetrize(new_gg_r)
        new_q_l = _flipped_eigh(new_gg_l)
        new_q_r = _flipped_eigh(new_gg_r)
        new_esi_l = jnp.full_like(esi_l, init_factor**-0.5)
        new_esi_r = jnp.full_like(esi_r, init_factor**-0.5)
        zero_dir = jnp.zeros_like(g32)
        return (
            zero_dir,
            jnp.zeros_like(exp_avg),
            jnp.zeros_like(exp_avg_sq),
            new_gg_l,
            new_gg_r,
            new_q_l,
            new_q_r,
            new_esi_l,
            new_esi_r,
        )

    def _normal_branch(_):
        # 1. Project: g_proj = q_l.T @ g @ q_r.
        g_qr = jnp.einsum("...ij,...jk->...ik", g32, q_r, out_sharding=out_p)
        g_proj = jnp.einsum("...ki,...kj->...ij", q_l, g_qr, out_sharding=out_p)

        # 2. Adam in projected basis.
        new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g_proj
        new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * g_proj * g_proj
        precond_proj = new_exp_avg / (jnp.sqrt(new_exp_avg_sq) + eps)

        # 3. Direction = q_l @ precond_proj @ q_r.T.
        p_qrt = jnp.einsum("...ij,...kj->...ik", precond_proj, q_r, out_sharding=out_p)
        direction = jnp.einsum("...ij,...jk->...ik", q_l, p_qrt, out_sharding=out_p)

        # 4. Whitened Gram update.
        g_qr_white = g_qr * esi_r[..., None, :]
        left_target = jnp.einsum("...ik,...jk->...ij", g_qr_white, g_qr_white, out_sharding=out_p) / inner_cols
        qlT_g = jnp.einsum("...ki,...kj->...ij", q_l, g32, out_sharding=out_p)
        qlT_g_white = qlT_g * esi_l[..., :, None]
        right_target = jnp.einsum("...ki,...kj->...ij", qlT_g_white, qlT_g_white, out_sharding=out_p) / inner_rows
        new_gg_l = _symmetrize(shampoo_beta * gg_l + (1.0 - shampoo_beta) * left_target)
        new_gg_r = _symmetrize(shampoo_beta * gg_r + (1.0 - shampoo_beta) * right_target)

        # 5. ESI update from projected-gradient diagonals.
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

        # 6. Warm-started QR-iteration refresh + reproject exp_avg.
        should_refresh = jnp.equal(jnp.remainder(step, precond_freq), 0)

        def _refresh(_):
            ea_qrT = jnp.einsum("...ij,...kj->...ik", new_exp_avg, q_r, out_sharding=out_p)
            ea_original = jnp.einsum("...ij,...jk->...ik", q_l, ea_qrT, out_sharding=out_p)
            gg_l_q = jnp.einsum("...ij,...jk->...ik", new_gg_l, q_l, out_sharding=out_p)
            gg_r_q = jnp.einsum("...ij,...jk->...ik", new_gg_r, q_r, out_sharding=out_p)
            ql_new, _ = jnp.linalg.qr(gg_l_q)
            qr_new, _ = jnp.linalg.qr(gg_r_q)
            ea_qr_new = jnp.einsum("...ij,...jk->...ik", ea_original, qr_new, out_sharding=out_p)
            ea_new = jnp.einsum("...ki,...kj->...ij", ql_new, ea_qr_new, out_sharding=out_p)
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
    return jax.lax.cond(is_first, _init_branch, _normal_branch, operand=None)


def scale_by_klsoaph(
    beta1: float = 0.95,
    beta2: float = 0.9,
    shampoo_beta: float = 0.9,
    eps: float = 1e-8,
    precond_freq: int = 1,
    init_factor: float = 0.1,
) -> optax.GradientTransformation:
    """Full-matrix SOAP-style preconditioner (upstream KLSOAPH, de-blocked).

    For each ``(..., rows, cols)`` weight, maintains the full row-Gram
    ``gg_l = (rows, rows)`` and col-Gram ``gg_r = (cols, cols)`` and their
    eigenbases ``q_l/q_r``; Adam runs in the projected eigenbasis and the
    direction is rotated back. Whitened-Gram + ESI + warm-started QR refresh
    match KellerJordan/modded-nanogpt PR #290. The scale-invariant
    ("hyperball") post-step (applied downstream) normalizes the full update.

    Default ``(beta1, beta2, shampoo_beta) = (0.95, 0.9, 0.9)`` matches the
    upstream "passing" tuple; ``precond_freq=1`` refreshes the eigenbasis
    every step (freshest preconditioner).

    State is allocated only for leaves with ``ndim >= 2``; lower-rank leaves
    pass through unchanged (route them to another group via an optax mask).
    Leaves with ``ndim > 2`` are treated as stacks of ``(rows, cols)``
    matrices over the leading axes (per-element full-matrix SOAP via einsum
    ellipsis).
    """

    def init_fn(params):
        return ScaleByKLSoapHState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=jax.tree.map(_zeros_mn, params, is_leaf=lambda x: x is None),
            exp_avg_sq=jax.tree.map(_zeros_mn, params, is_leaf=lambda x: x is None),
            gg_l=jax.tree.map(lambda p: _zeros_square(p, -2), params, is_leaf=lambda x: x is None),
            gg_r=jax.tree.map(lambda p: _zeros_square(p, -1), params, is_leaf=lambda x: x is None),
            q_l=jax.tree.map(lambda p: _eye_square(p, -2), params, is_leaf=lambda x: x is None),
            q_r=jax.tree.map(lambda p: _eye_square(p, -1), params, is_leaf=lambda x: x is None),
            esi_l=jax.tree.map(lambda p: _esi_init(p, -2, init_factor), params, is_leaf=lambda x: x is None),
            esi_r=jax.tree.map(lambda p: _esi_init(p, -1, init_factor), params, is_leaf=lambda x: x is None),
        )

    def update_fn(updates, state, params=None):
        del params
        next_count = optax.safe_increment(state.count)

        def per_leaf(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r):
            if grad is None or exp_avg is None:
                return _SoapStepResult(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r, esi_l, esi_r)

            # Full-matrix SOAP: run the step directly on the (..., rows, cols) gradient.
            # The step replicates state internally for the eigh/qr; the parameter
            # sharding is restored downstream via _match_named_update_sharding.
            out = _klsoaph_step(
                grad.astype(jnp.float32),
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
            direction = out[0].astype(grad.dtype)
            return _SoapStepResult(direction, *out[1:])

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
