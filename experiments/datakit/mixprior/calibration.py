# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correct metric means and uncertainty using held-out candidate residuals."""

from dataclasses import dataclass, replace

import jax
import numpy as np

from experiments.datakit.mixprior.acquisition import AdditiveMetricGP, KernelConfig, fit_additive
from experiments.datakit.mixprior.hf import Data
from experiments.datakit.mixprior.objective import Objective

CALIBRATION_FOLDS = 4
CALIBRATION_SEED = 20260912


@dataclass
class ResidualCalibration:
    bias: np.ndarray
    variance_multiplier: np.ndarray


def residual_calibration(
    mean: np.ndarray,
    variance: np.ndarray,
    measured: np.ndarray,
    noise: np.ndarray,
    counts: np.ndarray,
) -> ResidualCalibration:
    """Estimate per-metric bias and predictive variance inflation from held-out errors."""
    if mean.shape != variance.shape or mean.shape != measured.shape or len(mean) < 2:
        raise ValueError("Calibration needs aligned posterior moments and at least two held-out designs")
    residual = measured - mean
    bias = residual.mean(axis=0)
    # Estimate predictive error including uncertainty in the fitted bias. Undo
    # replicate averaging before estimating the single-run variance multiplier.
    predictive_error = (residual - bias) ** 2 * (len(mean) + 1) / (len(mean) - 1)
    multiplier = ((predictive_error + noise - noise / counts[:, None]) / (variance + noise)).mean(axis=0)
    return ResidualCalibration(bias, np.maximum(multiplier, 1))


def fit_calibrated(
    data: Data,
    objective: Objective,
    counts: np.ndarray,
    calibration_indices: np.ndarray,
    device: jax.Device,
    *,
    kernel_config: KernelConfig = KernelConfig(),
) -> AdditiveMetricGP:
    """Cross-fit candidate corrections, then condition the final GP on every design.

    Args:
        data: Grouped designs, with replicates kept together.
        objective: Fixed metric reference scales and replicate-noise estimate.
        counts: Replicate counts aligned with data.
        calibration_indices: Candidate designs representative of future proposals.
        device: JAX device for all GP fits and predictions.
        kernel_config: Covariance amplitudes shared by fold and final fits.
    """
    indices = np.asarray(calibration_indices)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(indices) < 2 * CALIBRATION_FOLDS
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(data.weights))
    ):
        raise ValueError("Calibration needs at least eight distinct, valid grouped design indices")
    all_indices = np.arange(len(data.weights))
    mean = np.empty((len(indices), len(objective.columns)))
    variance = np.empty_like(mean)
    folds = np.array_split(np.random.default_rng(CALIBRATION_SEED).permutation(len(indices)), CALIBRATION_FOLDS)
    for positions in folds:
        held_out = indices[positions]
        training = np.setdiff1d(all_indices, held_out)
        subset = replace(
            data,
            weights=data.weights[training],
            outcomes=data.outcomes[training],
            groups=[data.groups[i] for i in training],
            observation_ids=[data.observation_ids[i] for i in training],
        )
        model = fit_additive(subset, objective, counts[training], device, kernel_config=kernel_config)
        mean[positions], variance[positions] = jax.device_get(model.predict_metrics(data.weights[held_out]))
    noise = np.diag(objective.noise_covariance) / objective.scale**2
    measured = (data.outcomes[indices][:, objective.columns] - objective.mean) / objective.scale
    correction = residual_calibration(mean, variance, measured, noise, counts[indices])
    model = fit_additive(data, objective, counts, device, kernel_config=kernel_config)
    bias, multiplier = jax.device_put((correction.bias, correction.variance_multiplier), device)
    return replace(model, bias=bias, variance_multiplier=multiplier)


def new_design_indices(data: Data, previous: Data) -> np.ndarray:
    """Identify calibration designs by mixture identity, excluding previous replicates."""
    if data.components != previous.components or data.domains != previous.domains:
        raise ValueError("Calibration revisions must have identical component ordering and domains")
    if not np.array_equal(data.quality, previous.quality) or not np.array_equal(
        data.phase_budgets, previous.phase_budgets
    ):
        raise ValueError("Calibration revisions must use identical quality levels and phase budgets")
    previous_designs = {tuple(row.ravel()) for row in np.round(previous.weights, 12)}
    return np.asarray(
        [i for i, row in enumerate(np.round(data.weights, 12)) if tuple(row.ravel()) not in previous_designs], dtype=int
    )
