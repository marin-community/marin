# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""General/specialist learning with competence-triggered gradient conflict."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 128


@dataclass(frozen=True)
class InterferenceConfig:
    """Frozen kinetics and response geometry."""

    general_rate: float
    rare_general_efficiency: float
    specialist_rate: float
    interference_rate: float
    threshold: float
    softness: float
    evaluation_specialist_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"general={self.general_rate:g},rare_general={self.rare_general_efficiency:g},"
            f"specialist={self.specialist_rate:g},interference={self.interference_rate:g},"
            f"threshold={self.threshold:g},softness={self.softness:g},"
            f"eval={self.evaluation_specialist_weight:g},l2={self.l2:g}"
        )


def validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Each phase must be a nonnegative simplex mixture")
    return np.maximum(weights, 0.0)


def derivatives(
    general: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    config: InterferenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    general_drive = (1.0 - rare_weight) + config.rare_general_efficiency * rare_weight
    general_derivative = config.general_rate * general_drive * (1.0 - general)
    activation = 1.0 / (1.0 + np.exp(-(general - config.threshold) / config.softness))
    specialist_acquisition = config.specialist_rate * rare_weight * general * (1.0 - specialist)
    specialist_interference = config.interference_rate * (1.0 - rare_weight) * activation * specialist
    return general_derivative, specialist_acquisition - specialist_interference


def phase_update(
    general: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: InterferenceConfig,
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
        general = np.clip(general, 0.0, 1.0)
        specialist = np.clip(specialist, 0.0, 1.0)
    return general, specialist


def terminal_state(weights: np.ndarray, alpha0: float, config: InterferenceConfig) -> np.ndarray:
    weights = validate_weights(weights)
    general = np.zeros(len(weights), dtype=float)
    specialist = np.zeros(len(weights), dtype=float)
    general, specialist = phase_update(general, specialist, weights[:, 0, 1], alpha0, config)
    general, specialist = phase_update(general, specialist, weights[:, 1, 1], 1.0 - alpha0, config)
    return np.column_stack([general, specialist])


def response_feature(weights: np.ndarray, alpha0: float, config: InterferenceConfig) -> np.ndarray:
    state = terminal_state(weights, alpha0, config)
    unresolved_general = 1.0 - state[:, 0]
    unresolved_specialist = 1.0 - state[:, 1]
    q = config.evaluation_specialist_weight
    return (1.0 - q) * unresolved_general + q * unresolved_specialist
