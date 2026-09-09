# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An exposure-response mean, with a Gaussian process fitted to its residuals.

Mixture arrays have shape (observations, 2 phases, components). The mean uses
exposure aggregated by domain; the GP uses square roots of component weights.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.spatial.distance import cdist, pdist

from experiments.datakit.mixprior.hf import Data

KERNEL_VARIANCE = 0.1
LENGTHSCALE_MULTIPLIER = 2.0
JITTER = 1e-6
OUTCOME_CAP = 8.0
MAD_SCALE = 1.482602218505602

LOG_EXPOSURE_THRESHOLD = np.log(2.0)
EXPOSURE_PENALTY = 0.01
COOLDOWN_ALIGNMENT = 0.25
COOLDOWN_AMPLIFICATION = 1.5

MEAN_INTERCEPT_STD = 5.0
MEAN_PHASE_STD = 1.0
MEAN_DOMAIN_STD = 0.5


def kernel(x: np.ndarray, y: np.ndarray, lengthscale: float) -> np.ndarray:
    """Matérn-5/2 covariance between rows of two feature matrices."""
    distance = cdist(x, y)
    radius = np.sqrt(5) * distance / lengthscale
    return KERNEL_VARIANCE * (1 + radius + radius**2 / 3) * np.exp(-radius)


@dataclass
class Features:
    available: np.ndarray  # Available tokens per component.
    budgets: np.ndarray  # Token budget for each of the two phases.
    membership: np.ndarray  # Component-to-domain indicator matrix.

    def __call__(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return GP inputs, the mean's design matrix, and its fixed penalty."""
        component_share = self.available / self.available.sum()
        domain_share = component_share @ self.membership
        exposure_per_unit_weight = self.budgets[:, None] / self.available
        component_exposure = weights * exposure_per_unit_weight

        # Average component epochs within each domain, weighted by available tokens.
        weighted_exposure = component_share * component_exposure
        domain_exposure = (weighted_exposure @ self.membership) / domain_share

        # More exposure helps with diminishing returns. Phase and domain
        # coefficients are fitted below; this saturation curve stays fixed.
        benefit = domain_share * -np.expm1(-domain_exposure)
        phase_benefit = benefit.sum(axis=-1)
        domain_benefit = benefit.sum(axis=-2)
        domain_benefit -= domain_benefit.mean(axis=-1, keepdims=True)
        intercept = np.ones(len(weights))
        mean_design = np.column_stack([intercept, phase_benefit, domain_benefit])

        # The overexposure penalty grows after the benefit has saturated.
        log_exposure = np.log1p(domain_exposure)
        excess_exposure = np.logaddexp(0, log_exposure - LOG_EXPOSURE_THRESHOLD)
        overexposure_penalty = domain_share * excess_exposure**2
        fixed_mean = -EXPOSURE_PENALTY * overexposure_penalty.sum(axis=(-2, -1))

        # Favor a stronger version of the early mixture shift during cooldown.
        rooted_weights = np.sqrt(weights)
        shifts = rooted_weights - np.sqrt(component_share)
        early_shift = shifts[:, 0]
        late_shift = shifts[:, 1]
        cooldown_residual = late_shift - COOLDOWN_AMPLIFICATION * early_shift
        fixed_mean -= COOLDOWN_ALIGNMENT * np.square(cooldown_residual).sum(axis=-1)

        gp_inputs = rooted_weights.reshape(len(weights), -1)
        return gp_inputs, mean_design, fixed_mean


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
        inputs, mean_design, fixed_mean = self.features(weights)
        cross_covariance = kernel(inputs, self.train_x, self.lengthscale)

        exposure_mean = fixed_mean + mean_design @ self.coefficients
        residual_mean = cross_covariance @ self.alpha
        normalized_mean = exposure_mean + residual_mean

        # Var[f* | y] = Var[f*] - K* (K + noise)^-1 K*^T.
        solved = solve_triangular(self.cholesky, cross_covariance.T, lower=True)
        variance_reduction = np.square(solved).sum(axis=0)
        normalized_variance = np.maximum(KERNEL_VARIANCE - variance_reduction, 0)

        mean = self.center + self.scale * normalized_mean
        variance = self.scale**2 * normalized_variance
        return mean, variance


def kernel_lengthscale(weights: np.ndarray) -> float:
    """Calibrate distance using all observed phase profiles."""
    phase_profiles = np.concatenate([np.sqrt(weights[:, phase]) for phase in range(2)])
    squared_distances = pdist(phase_profiles, metric="sqeuclidean")
    distinct_distances = squared_distances[squared_distances > np.finfo(float).eps]
    if not len(distinct_distances):
        raise ValueError("Kernel calibration requires distinct mixture profiles")
    typical_distance = float(np.sqrt(np.median(distinct_distances)))
    return LENGTHSCALE_MULTIPLIER * typical_distance


def fit_mean(design: np.ndarray, values: np.ndarray, domain_count: int) -> np.ndarray:
    """Fit coefficients for the intercept, phase benefits, and centered domain-benefit columns."""
    prior_mean = np.array([0.0, 1.0, 1.0] + [0.0] * domain_count)
    prior_std = np.array([MEAN_INTERCEPT_STD, MEAN_PHASE_STD, MEAN_PHASE_STD] + [MEAN_DOMAIN_STD] * domain_count)
    prior_precision = 1 / prior_std**2

    precision = design.T @ design + np.diag(prior_precision)
    information = design.T @ values + prior_precision * prior_mean
    return np.linalg.solve(precision, information)


def fit(data: Data, values: np.ndarray, variances: np.ndarray) -> GP:
    """Fit on one swarm; all mixture proposals use its component ordering."""
    if values.shape != (len(data.weights),) or variances.shape != values.shape:
        raise ValueError("Training weights, objectives, and noise variances must align")
    if not np.isfinite(values).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Objectives must be finite and noise variances positive")

    # Normalize outcomes so a few failed runs do not set the model's scale.
    center = float(np.median(values))
    scale = MAD_SCALE * float(np.median(np.abs(values - center)))
    if scale == 0:
        scale = float(values.std())
    if scale == 0:
        raise ValueError("GP fitting requires nonconstant observations")
    normalized_values = np.clip((values - center) / scale, -OUTCOME_CAP, OUTCOME_CAP)
    normalized_variances = variances / scale**2

    # Build the domain exposure mean and the component-level GP inputs.
    domains = sorted(set(data.domains))
    membership = np.asarray([[name == domain for domain in domains] for name in data.domains], dtype=float)
    features = Features(available=data.available_tokens, budgets=data.phase_budgets, membership=membership)
    train_inputs, mean_design, fixed_mean = features(data.weights)
    lengthscale = kernel_lengthscale(data.weights)

    coefficients = fit_mean(mean_design, normalized_values - fixed_mean, len(domains))
    residuals = normalized_values - fixed_mean - mean_design @ coefficients

    # Condition the GP on the remaining error after fitting the exposure mean.
    covariance = kernel(train_inputs, train_inputs, lengthscale)
    covariance += np.diag(normalized_variances)
    # Relative jitter remains effective when observation variances are large.
    covariance += JITTER * np.diag(np.diag(covariance))
    cholesky = np.linalg.cholesky(covariance)
    alpha = cho_solve((cholesky, True), residuals)

    return GP(
        features=features,
        train_x=train_inputs,
        coefficients=coefficients,
        lengthscale=lengthscale,
        cholesky=cholesky,
        alpha=alpha,
        center=center,
        scale=scale,
    )
