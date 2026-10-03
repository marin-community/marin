# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Flip detector: find the few directions along which a weight matrix's updates keep reversing.

For a matrix ``W`` [a, b] (input axis first, as ``x @ W``) with applied update ``D_t = W_t - W_{t-1}``, the
detector keeps running averages over roughly ``1 / (1 - beta)`` steps of the lag-1 cross term and the update
energy, on both sides:

- input side: ``C = EMA(D_t D_{t-1}^T)``, ``P = EMA(D_t D_t^T)`` ([a, a]);
- output side: ``C = EMA(D_t^T D_{t-1})``, ``P = EMA(D_t^T D_t)`` ([b, b]).

The generalized eigenproblem ``sym(C) v = rho P v`` gives one correlation ``rho`` in about [-1, 1] per direction:
``rho`` near -1 is a direction whose update flips sign every step, near +1 steady drift. The ``k`` most negative
directions are the flagged subspace: the directions whose update component purely alternates, so they sit
orthogonal to the drift, and damping them would leave the drift alone. Each step the detector also reports the
cosine between consecutive updates restricted to the flagged subspace and to its complement (``flagged_cos`` /
``rest_cos``), and the flagged share of the update energy. Stacked tensors carry a leading layer axis and are
handled per layer.

Diagnostic only: nothing here changes the update.
"""

from __future__ import annotations

import jax.numpy as jnp

FLIP_DETECTOR_FILE = "flip_detector_{index:04d}.npz"
DEFAULT_FLIP_PATTERNS = (
    r"stacked_blocks(_tail)?\.stacked\.attn\.w_(q|dkv|uk|uv|o)$",
    r"stacked_blocks(_tail)?\.stacked\.attn\.rel_pos\.r_proj$",
)


def side_stats(d: jnp.ndarray, d_prev: jnp.ndarray, side: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(cross, energy)`` of one step for ``side`` ``"in"`` ([..., a, a]) or ``"out"`` ([..., b, b])."""
    if side == "in":
        return jnp.einsum("...ab,...cb->...ac", d, d_prev), jnp.einsum("...ab,...cb->...ac", d, d)
    return jnp.einsum("...ab,...ac->...bc", d, d_prev), jnp.einsum("...ab,...ac->...bc", d, d)


def flagged_cosines(d, d_prev, basis, side: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per layer: cosine of consecutive updates inside the flagged subspace ``basis`` ([..., n, k], orthonormal
    columns) and in its complement, and the flagged share of this update's energy."""

    def split(x):
        if side == "in":
            inside = jnp.einsum("...nk,...mk,...mb->...nb", basis, basis, x)
        else:
            inside = jnp.einsum("...an,...nk,...mk->...am", x, basis, basis)
        return inside, x - inside

    f, r = split(d)
    fp, rp = split(d_prev)

    def cos(x, y):
        num = jnp.sum(x * y, axis=(-2, -1))
        return num / jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=(-2, -1)) * jnp.sum(y * y, axis=(-2, -1)), 1e-30))

    share = jnp.sum(f * f, axis=(-2, -1)) / jnp.maximum(jnp.sum(d * d, axis=(-2, -1)), 1e-30)
    return cos(f, fp), cos(r, rp), share


def flip_directions(
    cross: jnp.ndarray, energy: jnp.ndarray, k: int, ridge: float = 1e-3
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per layer ([..., n, n] inputs): the ascending correlations ``rho`` [..., n] of ``sym(cross) v = rho energy v``
    and an orthonormal basis [..., n, k] of the ``k`` most negative directions. Runs on device (batched eigh)."""
    sym = 0.5 * (cross + jnp.swapaxes(cross, -1, -2))
    n = energy.shape[-1]
    trace = jnp.trace(energy, axis1=-2, axis2=-1)[..., None, None]
    energy = energy + ridge * trace / n * jnp.eye(n, dtype=energy.dtype)
    w, q = jnp.linalg.eigh(energy)
    inv_sqrt = jnp.einsum("...ij,...j,...kj->...ik", q, 1.0 / jnp.sqrt(jnp.maximum(w, 1e-30)), q)
    rho, u = jnp.linalg.eigh(inv_sqrt @ sym @ inv_sqrt)
    basis, _ = jnp.linalg.qr(inv_sqrt @ u[..., :k])
    return rho, basis


def subspace_overlap(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """``||a^T b||_F^2 / k`` per layer for orthonormal [..., n, k] bases: 1 for the same subspace, about k/n for
    random ones."""
    return jnp.sum(jnp.einsum("...nk,...nj->...kj", a, b) ** 2, axis=(-2, -1)) / a.shape[-1]
