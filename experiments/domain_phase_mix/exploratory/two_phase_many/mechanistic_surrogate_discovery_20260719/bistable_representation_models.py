# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quartic representation flow with data-tilted bistable basins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 256
INITIAL_STATE = -1.0


@dataclass(frozen=True)
class BistableConfig:
    """Frozen quartic transition and terminal response."""

    barrier: float
    tilt: float
    speed: float
    evaluation_state: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"barrier={self.barrier:g},tilt={self.tilt:g},speed={self.speed:g},"
            f"eval={self.evaluation_state:g},l2={self.l2:g}"
        )


def validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Each phase must be a nonnegative simplex mixture")
    return np.maximum(weights, 0.0)


def derivative(state: np.ndarray, rare_weight: np.ndarray, config: BistableConfig) -> np.ndarray:
    mixture_tilt = config.tilt * (2.0 * np.asarray(rare_weight, dtype=float) - 1.0)
    return config.speed * (config.barrier * state + mixture_tilt - state**3)


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: BistableConfig,
) -> np.ndarray:
    steps = max(1, int(np.ceil(INTEGRATION_STEPS_PER_UNIT * duration)))
    step_size = duration / steps
    state = np.asarray(state, dtype=float).copy()
    for _ in range(steps):
        k1 = derivative(state, rare_weight, config)
        k2 = derivative(state + 0.5 * step_size * k1, rare_weight, config)
        k3 = derivative(state + 0.5 * step_size * k2, rare_weight, config)
        k4 = derivative(state + step_size * k3, rare_weight, config)
        state += step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite bistable state for {config.key}")
    return state


def terminal_state(weights: np.ndarray, alpha0: float, config: BistableConfig) -> np.ndarray:
    weights = validate_weights(weights)
    state = np.full(len(weights), INITIAL_STATE, dtype=float)
    state = phase_update(state, weights[:, 0, 1], alpha0, config)
    return phase_update(state, weights[:, 1, 1], 1.0 - alpha0, config)


def response_feature(weights: np.ndarray, alpha0: float, config: BistableConfig) -> np.ndarray:
    state = terminal_state(weights, alpha0, config)
    return (state - config.evaluation_state) ** 2
