# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conservative gradient flow on a curved general-specialist task manifold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 256


@dataclass(frozen=True)
class CurvedSpecializationConfig:
    """Frozen task geometry, flow speed, and evaluation response."""

    manifold_power: int
    broad_specialist_regularization: float
    rare_general_weight: float
    speed: float
    evaluation_rare_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"power={self.manifold_power},broad_reg={self.broad_specialist_regularization:g},"
            f"rare_general={self.rare_general_weight:g},speed={self.speed:g},"
            f"eval={self.evaluation_rare_weight:g},l2={self.l2:g}"
        )


def validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Each phase must be a nonnegative simplex mixture")
    return np.maximum(weights, 0.0)


def task_gradients(
    general: np.ndarray,
    specialist: np.ndarray,
    config: CurvedSpecializationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    broad_general = general - 1.0
    broad_specialist = config.broad_specialist_regularization * specialist
    manifold = general**config.manifold_power
    displacement = specialist - manifold
    manifold_derivative = config.manifold_power * general ** (config.manifold_power - 1)
    rare_general = config.rare_general_weight * (general - 1.0) - manifold_derivative * displacement
    rare_specialist = displacement
    return broad_general, broad_specialist, rare_general, rare_specialist


def derivatives(
    general: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    config: CurvedSpecializationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    broad_g, broad_s, rare_g, rare_s = task_gradients(general, specialist, config)
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    general_gradient = (1.0 - rare_weight) * broad_g + rare_weight * rare_g
    specialist_gradient = (1.0 - rare_weight) * broad_s + rare_weight * rare_s
    return -config.speed * general_gradient, -config.speed * specialist_gradient


def phase_update(
    general: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: CurvedSpecializationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(np.ceil(INTEGRATION_STEPS_PER_UNIT * duration)))
    step_size = duration / steps
    general = np.asarray(general, dtype=float).copy()
    specialist = np.asarray(specialist, dtype=float).copy()
    for _ in range(steps):
        k1_general, k1_specialist = derivatives(general, specialist, rare_weight, config)
        k2_general, k2_specialist = derivatives(
            general + 0.5 * step_size * k1_general,
            specialist + 0.5 * step_size * k1_specialist,
            rare_weight,
            config,
        )
        k3_general, k3_specialist = derivatives(
            general + 0.5 * step_size * k2_general,
            specialist + 0.5 * step_size * k2_specialist,
            rare_weight,
            config,
        )
        k4_general, k4_specialist = derivatives(
            general + step_size * k3_general,
            specialist + step_size * k3_specialist,
            rare_weight,
            config,
        )
        general += step_size * (k1_general + 2.0 * k2_general + 2.0 * k3_general + k4_general) / 6.0
        specialist += step_size * (k1_specialist + 2.0 * k2_specialist + 2.0 * k3_specialist + k4_specialist) / 6.0
        if not np.isfinite(general).all() or not np.isfinite(specialist).all():
            raise FloatingPointError(f"Non-finite curved-specialization state for {config.key}")
    return general, specialist


def terminal_state(weights: np.ndarray, alpha0: float, config: CurvedSpecializationConfig) -> np.ndarray:
    weights = validate_weights(weights)
    general = np.zeros(len(weights), dtype=float)
    specialist = np.zeros(len(weights), dtype=float)
    general, specialist = phase_update(general, specialist, weights[:, 0, 1], alpha0, config)
    general, specialist = phase_update(general, specialist, weights[:, 1, 1], 1.0 - alpha0, config)
    return np.column_stack([general, specialist])


def task_potentials(state: np.ndarray, config: CurvedSpecializationConfig) -> tuple[np.ndarray, np.ndarray]:
    general = np.asarray(state, dtype=float)[:, 0]
    specialist = np.asarray(state, dtype=float)[:, 1]
    broad = 0.5 * (general - 1.0) ** 2 + 0.5 * config.broad_specialist_regularization * specialist**2
    rare = (
        0.5 * config.rare_general_weight * (general - 1.0) ** 2
        + 0.5 * (specialist - general**config.manifold_power) ** 2
    )
    return broad, rare


def response_feature(weights: np.ndarray, alpha0: float, config: CurvedSpecializationConfig) -> np.ndarray:
    broad, rare = task_potentials(terminal_state(weights, alpha0, config), config)
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad + q * rare
