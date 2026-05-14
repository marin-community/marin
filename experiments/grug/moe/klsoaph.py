# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# KL SOAP H: SOAP-style Hessian-eigenbasis preconditioner with Adam moments
# in the eigenbasis. The "KL" qualifier refers to the scale-invariant
# ("hyperball") post-step applied in optimizer.py:_scale_invariant_hyperball_updates,
# not a KL-divergence projection. Reproduces the KLSOAPH optimizer from
# KellerJordan/modded-nanogpt PR #290; the only deviation is
# precond_freq=5 (upstream default 1).
#
# scale_by_klsoaph returns only the SOAP direction; the hyperball post-step
# is applied by scale_with_grug_klsoaph in optimizer.py, mirroring the
# muonh/normuonh pattern.

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax


class ScaleByKLSoapHState(NamedTuple):
    """Per-leaf SOAP state. Float32 throughout for eigh stability."""

    count: chex.Array
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    gg_l: optax.Updates
    gg_r: optax.Updates
    q_l: optax.Updates
    q_r: optax.Updates


class _SoapStepResult:
    """Plain-Python wrapper for the seven outputs of one SOAP step.

    Not a registered pytree, so jax.tree.map treats it as a leaf.
    """

    __slots__ = ("direction", "exp_avg", "exp_avg_sq", "gg_l", "gg_r", "q_l", "q_r")

    def __init__(self, direction, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r):
        self.direction = direction
        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.gg_l = gg_l
        self.gg_r = gg_r
        self.q_l = q_l
        self.q_r = q_r


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


def _klsoaph_step_2d(
    grad: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    gg_l: jnp.ndarray,
    gg_r: jnp.ndarray,
    q_l: jnp.ndarray,
    q_r: jnp.ndarray,
    step: jnp.ndarray,
    beta1: float,
    beta2: float,
    shampoo_beta: float,
    eps: float,
    precond_freq: int,
):
    """One SOAP step for a single 2-D (rows, cols) gradient.

    All arrays are float32. Matches upstream KLSOAPH.step() in
    KellerJordan/modded-nanogpt PR #290, sans the hyperball post-step
    (applied later by _scale_invariant_hyperball_updates).
    """
    rows = grad.shape[0]
    cols = grad.shape[1]

    g32 = grad.astype(jnp.float32)
    gg_l = shampoo_beta * gg_l + (1.0 - shampoo_beta) * (g32 @ g32.T) / cols
    gg_r = shampoo_beta * gg_r + (1.0 - shampoo_beta) * (g32.T @ g32) / rows

    should_refresh = jnp.equal(jnp.remainder(step, precond_freq), 0)

    def _refresh(_):
        _, ql_new = jnp.linalg.eigh(gg_l)
        _, qr_new = jnp.linalg.eigh(gg_r)
        return ql_new, qr_new

    def _keep(_):
        return q_l, q_r

    q_l, q_r = jax.lax.cond(should_refresh, _refresh, _keep, operand=None)

    g_proj = q_l.T @ g32 @ q_r
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * g_proj
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * g_proj * g_proj
    precond_proj = exp_avg / (jnp.sqrt(exp_avg_sq) + eps)
    direction = q_l @ precond_proj @ q_r.T

    return direction.astype(grad.dtype), exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r


def scale_by_klsoaph(
    beta1: float = 0.9,
    beta2: float = 0.9,
    shampoo_beta: float = 0.9,
    eps: float = 1e-8,
    precond_freq: int = 5,
) -> optax.GradientTransformation:
    """SOAP-style preconditioner with eigenbasis-refresh every precond_freq steps.

    Returns the direction in parameter space. No learning-rate scaling and no
    hyperball normalization — both applied by scale_with_grug_klsoaph.

    State is allocated only for leaves with ndim >= 2; lower-rank leaves get
    None state slots and pass through unchanged (their gradients should be
    routed to a different optimizer group via an optax mask). Leaves with
    ndim > 2 are treated as stacks of (rows, cols) matrices over the leading
    axes (vmapped).
    """

    def init_fn(params):
        return ScaleByKLSoapHState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=jax.tree.map(_zeros_param, params, is_leaf=lambda x: x is None),
            exp_avg_sq=jax.tree.map(_zeros_param, params, is_leaf=lambda x: x is None),
            gg_l=jax.tree.map(_zeros_left, params, is_leaf=lambda x: x is None),
            gg_r=jax.tree.map(_zeros_right, params, is_leaf=lambda x: x is None),
            q_l=jax.tree.map(lambda p: _eye_leading(p, -2), params, is_leaf=lambda x: x is None),
            q_r=jax.tree.map(lambda p: _eye_leading(p, -1), params, is_leaf=lambda x: x is None),
        )

    def update_fn(updates, state, params=None):
        del params
        next_count = optax.safe_increment(state.count)

        def per_leaf(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r):
            if grad is None or exp_avg is None:
                return _SoapStepResult(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r)
            if grad.ndim == 2:
                out = _klsoaph_step_2d(
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    gg_l,
                    gg_r,
                    q_l,
                    q_r,
                    next_count,
                    beta1=beta1,
                    beta2=beta2,
                    shampoo_beta=shampoo_beta,
                    eps=eps,
                    precond_freq=precond_freq,
                )
                return _SoapStepResult(*out)
            out = jax.vmap(
                lambda g, ea, eas, gl, gr, ql, qr: _klsoaph_step_2d(
                    g,
                    ea,
                    eas,
                    gl,
                    gr,
                    ql,
                    qr,
                    next_count,
                    beta1=beta1,
                    beta2=beta2,
                    shampoo_beta=shampoo_beta,
                    eps=eps,
                    precond_freq=precond_freq,
                )
            )(grad, exp_avg, exp_avg_sq, gg_l, gg_r, q_l, q_r)
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
        )
        return directions, new_state

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByKLSoapHState", "scale_by_klsoaph"]
