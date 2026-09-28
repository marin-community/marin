# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Projected task flow on the constant-norm manifold used by MuonH."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 192
NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class SphericalProjectedFlowConfig:
    """Frozen projected-flow geometry and terminal response."""

    angular_curvature: float
    rare_curvature: float
    relaxation: float
    evaluation_rare_weight: float

    @property
    def key(self) -> str:
        return (
            f"angular={self.angular_curvature:g},rare={self.rare_curvature:g},"
            f"relax={self.relaxation:g},eval={self.evaluation_rare_weight:g}"
        )


def validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Each phase must be a nonnegative simplex mixture")
    return np.maximum(weights, 0.0)


def task_potential(displacement: np.ndarray, angular_curvature: float) -> np.ndarray:
    """Return the chordal task debt, with a quadratic zero-curvature limit."""

    displacement = np.asarray(displacement, dtype=float)
    if angular_curvature == 0.0:
        return 0.5 * displacement**2
    return (1.0 - np.cos(angular_curvature * displacement)) / angular_curvature**2


def task_gradient(displacement: np.ndarray, angular_curvature: float) -> np.ndarray:
    """Return the tangent gradient of the chordal task debt."""

    displacement = np.asarray(displacement, dtype=float)
    if angular_curvature == 0.0:
        return displacement
    return np.sin(angular_curvature * displacement) / angular_curvature


def derivative(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: SphericalProjectedFlowConfig,
) -> np.ndarray:
    broad_gradient = task_gradient(state + 0.5, config.angular_curvature)
    rare_gradient = task_gradient(state - 0.5, config.angular_curvature)
    gradient = (1.0 - rare_weight) * broad_gradient + config.rare_curvature * rare_weight * rare_gradient
    return -config.relaxation * gradient


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: SphericalProjectedFlowConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    steps = max(1, int(np.ceil(steps_per_unit * duration)))
    step_size = duration / steps
    state = np.asarray(state, dtype=float).copy()
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        k1 = derivative(state, rare_weight, config)
        k2 = derivative(state + 0.5 * step_size * k1, rare_weight, config)
        k3 = derivative(state + 0.5 * step_size * k2, rare_weight, config)
        k4 = derivative(state + step_size * k3, rare_weight, config)
        state += step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not np.isfinite(state).all():
        raise FloatingPointError(f"Non-finite spherical projected state for {config.key}")
    return state


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: SphericalProjectedFlowConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weights = validate_weights(weights)
    state = np.zeros(len(weights), dtype=float)
    state = phase_update(
        state,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return phase_update(
        state,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: SphericalProjectedFlowConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    state = terminal_state(
        weights,
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    broad = task_potential(state + 0.5, config.angular_curvature)
    rare = config.rare_curvature * task_potential(state - 0.5, config.angular_curvature)
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad + q * rare


def tied_semigroup_error(config: SphericalProjectedFlowConfig, phase0_fraction: float) -> float:
    rng = np.random.default_rng(20260719)
    rare_weight = rng.uniform(size=64)
    initial = rng.uniform(-0.25, 0.25, size=64)
    first = phase_update(initial, rare_weight, phase0_fraction, config, steps_per_unit=768)
    split = phase_update(first, rare_weight, 1.0 - phase0_fraction, config, steps_per_unit=768)
    full = phase_update(initial, rare_weight, 1.0, config, steps_per_unit=768)
    return float(np.max(np.abs(split - full)))
