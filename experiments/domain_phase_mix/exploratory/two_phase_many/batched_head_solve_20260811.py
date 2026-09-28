# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solve the surrogate head for many targets at once (GEN-052).

``fit_head`` is called once per (fold, target). Inside it, everything except the non-negative solve
depends only on the DESIGN -- the orthonormal basis for the free block, the residualised constrained
columns, their norms, and the least-squares factorisation of the free block. None of that depends on
which target is being fitted, so on a 29-target objective it was recomputed 29 times identically.
Measured on the WSD80 objective: 87 calls per evaluation, of which the SVD, projection, norms and
lstsq accounted for roughly 70 percent of the time and only the NNLS genuinely varied.

Splitting the work into a per-design ``prepare`` and a per-target ``solve`` removes that redundancy.
Results are identical to the unbatched path, not approximately so -- the same operations happen in the
same order, they are simply hoisted out of the target loop.
"""

import numpy as np
from scipy.optimize import nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import general_mixture_surrogate_20260809 as model


class PreparedHead:
    """Design-side work hoisted out of the per-target loop."""

    __slots__ = ("basis", "constrained", "free", "pooled", "pseudo_inverse", "scale", "scaled", "strength")

    def __init__(self, free: np.ndarray, constrained: np.ndarray, ridge: float, pooled: int):
        self.free = free
        self.constrained = constrained
        self.pooled = pooled
        self.basis = model.column_space(free)
        columns = constrained - self.basis @ (self.basis.T @ constrained)
        self.scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
        scaled = columns / self.scale
        self.strength = None
        if ridge > 0:
            strength = np.sqrt(ridge) * np.concatenate([np.full(pooled, 1e-3), np.ones(scaled.shape[1] - pooled)])
            scaled = np.vstack([scaled, np.diag(strength)])
            self.strength = strength
        self.scaled = scaled
        self.pseudo_inverse = np.linalg.pinv(free)

    def solve(self, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = response - self.basis @ (self.basis.T @ response)
        if self.strength is not None:
            target = np.concatenate([target, np.zeros(self.scaled.shape[1])])
        amplitudes, _ = nnls(self.scaled, target, maxiter=20000)
        amplitudes = amplitudes / self.scale
        return self.pseudo_inverse @ (response - self.constrained @ amplitudes), amplitudes
