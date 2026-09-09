# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A ridge-fitted exposure mean plus a Matérn GP over rooted mixture weights."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.spatial.distance import cdist, pdist

from experiments.datakit.mixprior.hf import Data

KERNEL_VARIANCE = 0.1
JITTER = 1e-6
OUTCOME_CAP = 8.0
MAD_SCALE = 1.482602218505602


def kernel(x: np.ndarray, y: np.ndarray, lengthscale: float) -> np.ndarray:
    distance = np.sqrt(5) * cdist(x, y) / lengthscale
    return KERNEL_VARIANCE * (1 + distance + distance**2 / 3) * np.exp(-distance)


@dataclass
class Features:
    available: np.ndarray
    budgets: np.ndarray
    membership: np.ndarray

    def __call__(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return rooted weights, the ridge design, and the fixed mean penalty."""
        share = self.available / self.available.sum()
        domain_share = share @ self.membership
        exposure = weights * (self.budgets[:, None] / self.available)
        domain_exposure = (share * exposure) @ self.membership / domain_share
        benefit = domain_share * -np.expm1(-domain_exposure)
        phase_design = benefit.sum(axis=-1)
        domain_design = benefit.sum(axis=-2)
        domain_design -= domain_design.mean(axis=-1, keepdims=True)
        design = np.column_stack([np.ones(len(weights)), phase_design, domain_design])
        penalty = np.logaddexp(0, np.log1p(domain_exposure) - np.log(2)) ** 2
        offset = -0.01 * (domain_share * penalty).sum(axis=(-2, -1))
        shifts = np.sqrt(weights) - np.sqrt(share)
        offset -= 0.25 * np.square(shifts[:, 1] - 1.5 * shifts[:, 0]).sum(axis=-1)
        return np.sqrt(weights).reshape(len(weights), -1), design, offset


@dataclass
class GP:
    features: Features
    train_x: np.ndarray
    coefficients: np.ndarray
    lengthscale: float
    cholesky: np.ndarray
    alpha: np.ndarray
    center: float
    scale: float

    def predict(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return latent posterior means and variances in objective units."""
        x, design, offset = self.features(weights)
        cross = kernel(x, self.train_x, self.lengthscale)
        mean = offset + design @ self.coefficients + cross @ self.alpha
        solved = solve_triangular(self.cholesky, cross.T, lower=True)
        variance = np.maximum(KERNEL_VARIANCE - np.square(solved).sum(axis=0), 0)
        return self.center + self.scale * mean, self.scale**2 * variance


def fit(data: Data, values: np.ndarray, variances: np.ndarray) -> GP:
    """Fit on one swarm; all mixture proposals use its component ordering."""
    if values.shape != (len(data.weights),) or variances.shape != values.shape:
        raise ValueError("Training weights, objectives, and noise variances must align")
    if not np.isfinite(values).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Objectives must be finite and noise variances positive")
    center = float(np.median(values))
    scale = MAD_SCALE * float(np.median(np.abs(values - center)))
    if scale == 0:
        scale = float(values.std())
    if scale == 0:
        raise ValueError("GP fitting requires nonconstant observations")
    y = np.clip((values - center) / scale, -OUTCOME_CAP, OUTCOME_CAP)
    domains = sorted(set(data.domains))
    membership = np.asarray([[name == domain for domain in domains] for name in data.domains], dtype=float)
    features = Features(data.available_tokens, data.phase_budgets, membership)
    x, design, offset = features(data.weights)
    phase_x = np.concatenate([np.sqrt(data.weights[:, phase]) for phase in range(2)])
    distances = pdist(phase_x, metric="sqeuclidean")
    distances = distances[distances > np.finfo(float).eps]
    if not len(distances):
        raise ValueError("Kernel calibration requires distinct mixture profiles")
    lengthscale = 2 * float(np.sqrt(np.median(distances)))
    prior_mean = np.r_[0.0, 1.0, 1.0, np.zeros(len(domains))]
    prior_precision = np.r_[1 / 25, 1.0, 1.0, np.full(len(domains), 4.0)]
    coefficients = np.linalg.solve(
        design.T @ design + np.diag(prior_precision),
        design.T @ (y - offset) + prior_precision * prior_mean,
    )
    covariance = kernel(x, x, lengthscale) + np.diag(variances / scale**2)
    # Relative jitter remains effective when the observation scale is large.
    covariance += JITTER * np.diag(np.diag(covariance))
    cholesky = np.linalg.cholesky(covariance)
    alpha = cho_solve((cholesky, True), y - offset - design @ coefficients)
    return GP(features, x, coefficients, lengthscale, cholesky, alpha, center, scale)
