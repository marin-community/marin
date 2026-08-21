# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nonlinear shared-representation task-potential gradient flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

INTEGRATION_STEPS = 512


@dataclass(frozen=True)
class NonlinearPotentialConfig:
    curvature_ratio: float
    quartic_strength: float
    quartic_ratio: float
    relaxation: float
    evaluation_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"r={self.curvature_ratio:g},h={self.quartic_strength:g},s={self.quartic_ratio:g},"
            f"k={self.relaxation:g},q={self.evaluation_weight:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class NonlinearPotentialModel:
    config: NonlinearPotentialConfig
    intercept: float
    amplitude: float

    @property
    def natural_amplitude(self) -> float:
        return self.amplitude

    def predict(self, weights: np.ndarray, alpha0: float) -> np.ndarray:
        feature = response_feature(weights, alpha0, self.config)
        return self.intercept + self.amplitude * feature


def task_gradient(state: np.ndarray, rare_weight: np.ndarray, config: NonlinearPotentialConfig) -> np.ndarray:
    broad_delta = state + 0.5
    rare_delta = state - 0.5
    broad = broad_delta + config.quartic_strength * broad_delta**3
    rare = config.curvature_ratio * rare_delta + config.quartic_strength * config.quartic_ratio * rare_delta**3
    return (1.0 - rare_weight) * broad + rare_weight * rare


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: NonlinearPotentialConfig,
) -> np.ndarray:
    """Integrate one constant-mixture phase with deterministic RK4."""
    if duration <= 0.0:
        return state.copy()
    steps = max(1, int(np.ceil(INTEGRATION_STEPS * duration)))
    step = duration / steps
    result = state.copy()
    for _ in range(steps):
        k1 = -config.relaxation * task_gradient(result, rare_weight, config)
        k2 = -config.relaxation * task_gradient(result + 0.5 * step * k1, rare_weight, config)
        k3 = -config.relaxation * task_gradient(result + 0.5 * step * k2, rare_weight, config)
        k4 = -config.relaxation * task_gradient(result + step * k3, rare_weight, config)
        result += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result


def terminal_state(weights: np.ndarray, alpha0: float, config: NonlinearPotentialConfig) -> np.ndarray:
    state = np.zeros(len(weights), dtype=float)
    state = phase_update(state, weights[:, 0, 1], alpha0, config)
    return phase_update(state, weights[:, 1, 1], 1.0 - alpha0, config)


def task_potential(state: np.ndarray, config: NonlinearPotentialConfig) -> tuple[np.ndarray, np.ndarray]:
    broad_delta = state + 0.5
    rare_delta = state - 0.5
    broad = 0.5 * broad_delta**2 + 0.25 * config.quartic_strength * broad_delta**4
    rare = (
        0.5 * config.curvature_ratio * rare_delta**2
        + 0.25 * config.quartic_strength * config.quartic_ratio * rare_delta**4
    )
    return broad, rare


def response_feature(weights: np.ndarray, alpha0: float, config: NonlinearPotentialConfig) -> np.ndarray:
    state = terminal_state(weights, alpha0, config)
    broad, rare = task_potential(state, config)
    return (1.0 - config.evaluation_weight) * broad + config.evaluation_weight * rare


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    alpha0: float,
    config: NonlinearPotentialConfig,
) -> NonlinearPotentialModel:
    feature = response_feature(weights[indices], alpha0, config)[:, None]
    feature_mean = feature.mean(axis=0, keepdims=True)
    target_mean = float(target[indices].mean())
    design = feature - feature_mean
    centered = target[indices] - target_mean
    if config.l2 > 0.0:
        design = np.vstack([design, [[np.sqrt(config.l2)]]])
        centered = np.concatenate([centered, [0.0]])
    result = lsq_linear(design, centered, bounds=(0.0, np.inf), lsmr_tol="auto")
    if not result.success:
        raise RuntimeError(result.message)
    amplitude = float(result.x[0])
    intercept = target_mean - float(feature_mean[0, 0] * amplitude)
    return NonlinearPotentialModel(config, intercept, amplitude)
