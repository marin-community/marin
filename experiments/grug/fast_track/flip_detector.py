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
import numpy as np

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


def flip_directions(cross: np.ndarray, energy: np.ndarray, k: int, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """For one layer: the ascending correlations ``rho`` of ``sym(cross) v = rho energy v`` and an orthonormal
    basis [n, k] of the ``k`` most negative directions."""
    sym = 0.5 * (cross + cross.T)
    n = energy.shape[0]
    energy = energy + ridge * np.trace(energy) / n * np.eye(n)
    w, q = np.linalg.eigh(energy)
    inv_sqrt = q @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-30))) @ q.T
    rho, u = np.linalg.eigh(inv_sqrt @ sym @ inv_sqrt)
    v = inv_sqrt @ u[:, :k]
    basis, _ = np.linalg.qr(v)
    return rho, basis


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """``||a^T b||_F^2 / k`` for orthonormal [n, k] bases: 1 for the same subspace, about k/n for random ones."""
    return float(np.sum((a.T @ b) ** 2) / a.shape[1])
