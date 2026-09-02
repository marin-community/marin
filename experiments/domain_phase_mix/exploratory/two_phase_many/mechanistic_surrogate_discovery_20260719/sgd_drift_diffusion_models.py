# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Moment-closed SGD drift-diffusion dynamics for two-domain policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

INTEGRATION_STEPS = 512
NUMERICAL_FLOOR = 1e-12


class Schedule(StrEnum):
    COSINE = "cosine"
    WSD = "wsd"


@dataclass(frozen=True)
class DriftDiffusionConfig:
    curvature_ratio: float
    drift_rate: float
    diffusion_scale: float
    evaluation_mix: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"curvature={self.curvature_ratio:g},drift={self.drift_rate:g},"
            f"diffusion={self.diffusion_scale:g},eval={self.evaluation_mix:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class ResponseHead:
    intercept: float
    amplitude: float


@dataclass(frozen=True)
class DriftDiffusionModel:
    config: DriftDiffusionConfig
    phase0_fraction: float
    schedule: Schedule
    head: ResponseHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        feature = response_feature(weights, self.phase0_fraction, self.schedule, self.config)
        return self.head.intercept + self.head.amplitude * feature


def learning_rate(time: float, phase0_fraction: float, schedule: Schedule) -> float:
    """Return peak-normalized LR at normalized training time."""

    if schedule is Schedule.COSINE:
        return 0.5 * (1.0 + np.cos(np.pi * time))
    if schedule is Schedule.WSD:
        if time <= phase0_fraction:
            return 1.0
        progress = (time - phase0_fraction) / max(1.0 - phase0_fraction, NUMERICAL_FLOOR)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    raise ValueError(f"Unsupported schedule {schedule}")


def moment_rhs(
    mean: np.ndarray,
    variance: np.ndarray,
    rare_weight: np.ndarray,
    eta: float,
    config: DriftDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Close the first two moments of SGD on two quadratic domain losses."""

    curvature = config.curvature_ratio
    hessian = 1.0 - rare_weight + curvature * rare_weight
    broad_gradient = mean + 0.5
    rare_gradient = curvature * (mean - 0.5)
    mixture_gradient = (1.0 - rare_weight) * broad_gradient + rare_weight * rare_gradient
    gradient_difference = rare_gradient - broad_gradient
    expected_gradient_variance = (
        rare_weight * (1.0 - rare_weight) * (gradient_difference**2 + (curvature - 1.0) ** 2 * variance)
    )
    mean_rate = -config.drift_rate * eta * mixture_gradient
    variance_rate = config.drift_rate * (
        -2.0 * eta * hessian * variance + config.diffusion_scale * eta**2 * expected_gradient_variance
    )
    return mean_rate, variance_rate


def _rk4_step(
    mean: np.ndarray,
    variance: np.ndarray,
    rare_weight: np.ndarray,
    eta: float,
    step: float,
    config: DriftDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    k1_mean, k1_variance = moment_rhs(mean, variance, rare_weight, eta, config)
    k2_mean, k2_variance = moment_rhs(
        mean + 0.5 * step * k1_mean,
        np.maximum(variance + 0.5 * step * k1_variance, 0.0),
        rare_weight,
        eta,
        config,
    )
    k3_mean, k3_variance = moment_rhs(
        mean + 0.5 * step * k2_mean,
        np.maximum(variance + 0.5 * step * k2_variance, 0.0),
        rare_weight,
        eta,
        config,
    )
    k4_mean, k4_variance = moment_rhs(
        mean + step * k3_mean,
        np.maximum(variance + step * k3_variance, 0.0),
        rare_weight,
        eta,
        config,
    )
    next_mean = mean + step * (k1_mean + 2.0 * k2_mean + 2.0 * k3_mean + k4_mean) / 6.0
    next_variance = np.maximum(
        variance + step * (k1_variance + 2.0 * k2_variance + 2.0 * k3_variance + k4_variance) / 6.0,
        0.0,
    )
    return next_mean, next_variance


def terminal_moments(
    weights: np.ndarray,
    phase0_fraction: float,
    schedule: Schedule,
    config: DriftDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [n, 2, 2] policies, got {weights.shape}")
    mean = np.zeros(len(weights), dtype=float)
    variance = np.zeros(len(weights), dtype=float)
    step = 1.0 / INTEGRATION_STEPS
    for index in range(INTEGRATION_STEPS):
        time = (index + 0.5) * step
        phase = 0 if time < phase0_fraction else 1
        rare_weight = weights[:, phase, 1]
        eta = learning_rate(time, phase0_fraction, schedule)
        mean, variance = _rk4_step(mean, variance, rare_weight, eta, step, config)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise FloatingPointError(f"Non-finite drift-diffusion state for {config.key}")
    return mean, variance


def constant_policy_moments(
    rare_weight: np.ndarray,
    phase0_fraction: float,
    schedule: Schedule,
    config: DriftDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a phase-tied policy without consulting an artificial policy boundary."""

    rare_weight = np.asarray(rare_weight, dtype=float)
    mean = np.zeros(len(rare_weight), dtype=float)
    variance = np.zeros(len(rare_weight), dtype=float)
    step = 1.0 / INTEGRATION_STEPS
    for index in range(INTEGRATION_STEPS):
        time = (index + 0.5) * step
        eta = learning_rate(time, phase0_fraction, schedule)
        mean, variance = _rk4_step(mean, variance, rare_weight, eta, step, config)
    return mean, variance


def response_feature(
    weights: np.ndarray,
    phase0_fraction: float,
    schedule: Schedule,
    config: DriftDiffusionConfig,
) -> np.ndarray:
    mean, variance = terminal_moments(weights, phase0_fraction, schedule, config)
    broad_loss = 0.5 * ((mean + 0.5) ** 2 + variance)
    rare_loss = 0.5 * config.curvature_ratio * ((mean - 0.5) ** 2 + variance)
    return (1.0 - config.evaluation_mix) * broad_loss + config.evaluation_mix * rare_loss


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    phase0_fraction: float,
    schedule: Schedule,
    config: DriftDiffusionConfig,
) -> DriftDiffusionModel:
    feature = response_feature(weights, phase0_fraction, schedule, config)
    mean = float(np.mean(feature[train]))
    scale = max(float(np.sqrt(np.mean((feature[train] - mean) ** 2))), 1e-8)
    standardized = (feature[train] - mean) / scale
    target_mean = float(np.mean(target[train]))
    centered_target = target[train] - target_mean
    coefficient = max(
        float(standardized @ centered_target / (standardized @ standardized + config.l2)),
        0.0,
    )
    amplitude = coefficient / scale
    intercept = target_mean - amplitude * mean
    return DriftDiffusionModel(config, phase0_fraction, schedule, ResponseHead(intercept, amplitude))


def tied_policy_error(
    rare_weight: np.ndarray,
    phase0_fraction: float,
    schedule: Schedule,
    config: DriftDiffusionConfig,
) -> float:
    tied = np.stack(
        [
            np.column_stack([1.0 - rare_weight, rare_weight]),
            np.column_stack([1.0 - rare_weight, rare_weight]),
        ],
        axis=1,
    )
    tied_mean, tied_variance = terminal_moments(tied, phase0_fraction, schedule, config)
    direct_mean, direct_variance = constant_policy_moments(rare_weight, phase0_fraction, schedule, config)
    return max(
        float(np.max(np.abs(tied_mean - direct_mean))),
        float(np.max(np.abs(tied_variance - direct_variance))),
    )
