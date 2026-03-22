# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Riemannian geometry helpers for fixed-radius manifold optimization.

Provides tangent-space projection, retraction, and parallel transport
for parameters constrained to lie on a sphere of fixed radius (the
initialization norm). Used by AdamHR and MuonHR optimizers.

All operations support both 2D (single matrix) and >2D (stacked/scanned
layers) tensors, applying rowwise constraints over all axes except the
first (batch/stack) axis.
"""

import jax.numpy as jnp


NORM_EPS = 1e-10


def _rowwise_axes(p):
    """Return reduction axes for rowwise norm: all axes except the first for >2D, all axes for 2D."""
    if p.ndim == 2:
        return None  # full Frobenius norm
    return tuple(range(1, p.ndim))


def _safe_norm(p, axes):
    """Compute norm over given axes with keepdims for >2D."""
    if axes is None:
        return jnp.linalg.norm(p)
    return jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))


def project_tangent(g, p):
    """Project gradient g onto the tangent space of the sphere at p.

    Removes the radial component: g_tan = g - (g · p̂) p̂  where p̂ = p / ||p||.
    For >2D tensors the projection is applied rowwise (over all axes except the first).
    """
    axes = _rowwise_axes(p)
    if axes is None:
        # 2D: full matrix
        p_norm_sq = jnp.sum(jnp.square(p))
        radial = jnp.sum(g * p) / jnp.maximum(p_norm_sq, NORM_EPS) * p
    else:
        p_norm_sq = jnp.sum(jnp.square(p), axis=axes, keepdims=True)
        dot = jnp.sum(g * p, axis=axes, keepdims=True)
        radial = dot / jnp.maximum(p_norm_sq, NORM_EPS) * p
    return g - radial


def retract(p_old, tangent_step, learning_rate):
    """Retract from p_old along tangent_step back onto the sphere.

    Computes p_new = p_old + lr * tangent_step, then normalizes to ||p_old||.
    Returns the *delta* (p_new - p_old) so it can be used as an optax update.
    """
    axes = _rowwise_axes(p_old)
    p_new = p_old + learning_rate * tangent_step
    old_norm = _safe_norm(p_old, axes)
    new_norm = _safe_norm(p_new, axes)
    p_new = p_new / jnp.maximum(new_norm, NORM_EPS) * old_norm
    return p_new - p_old


def parallel_transport(state_vec, p_old, p_new):
    """Transport a tangent vector from T_{p_old} to T_{p_new} on the sphere.

    Uses the simple projection-based transport: project state_vec onto
    the tangent space at p_new. This is exact for infinitesimal steps
    and a good approximation for small step sizes.
    """
    return project_tangent(state_vec, p_new)
