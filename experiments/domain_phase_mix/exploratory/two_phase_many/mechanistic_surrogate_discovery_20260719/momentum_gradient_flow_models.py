# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Damped momentum gradient flow for two-task mixture schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    hessian_equilibrium_models as scalar_hessian,
)

BROAD_OPTIMUM = -0.5
RARE_OPTIMUM = 0.5


class Dynamics(StrEnum):
    FIRST_ORDER = "first_order"
    MOMENTUM = "momentum"


@dataclass(frozen=True)
class MomentumConfig:
    dynamics: Dynamics
    curvature_ratio: float
    relaxation: float
    damping_ratio: float
    evaluation_center: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"dynamics={self.dynamics.value},curvature={self.curvature_ratio:g},"
            f"relaxation={self.relaxation:g},damping={self.damping_ratio:g},"
            f"eval={self.evaluation_center:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class MomentumModel:
    alpha0: float
    config: MomentumConfig
    head: scalar_hessian.QuadraticHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.head.predict(response_feature(weights, self.alpha0, self.config))


def equilibrium(rare_weight: np.ndarray, curvature_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    hessian = (1.0 - rare_weight) + rare_weight * curvature_ratio
    linear = (1.0 - rare_weight) * BROAD_OPTIMUM + rare_weight * curvature_ratio * RARE_OPTIMUM
    return hessian, linear / hessian


def first_order_update(
    position: np.ndarray,
    velocity: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MomentumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    hessian, center = equilibrium(rare_weight, config.curvature_ratio)
    decay = np.exp(-config.relaxation * hessian * duration)
    result = center + (position - center) * decay
    return result, np.zeros_like(velocity)


def momentum_update(
    position: np.ndarray,
    velocity: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MomentumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    hessian, center = equilibrium(rare_weight, config.curvature_ratio)
    omega = config.relaxation
    damping = config.damping_ratio
    displacement = position - center
    decay_rate = damping * omega
    delta = omega * np.sqrt((damping**2 - hessian).astype(complex))
    scaled = delta * duration
    cosh = np.cosh(scaled)
    ratio = np.empty_like(delta)
    near_zero = np.abs(delta) < 1e-10
    ratio[near_zero] = duration
    ratio[~near_zero] = np.sinh(scaled[~near_zero]) / delta[~near_zero]
    decay = np.exp(-decay_rate * duration)
    next_displacement = decay * (displacement * cosh + (velocity + decay_rate * displacement) * ratio)
    next_velocity = decay * (velocity * cosh - (decay_rate * velocity + omega**2 * hessian * displacement) * ratio)
    return np.asarray(next_displacement.real + center, dtype=float), np.asarray(next_velocity.real, dtype=float)


def phase_update(
    position: np.ndarray,
    velocity: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MomentumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.dynamics is Dynamics.FIRST_ORDER:
        return first_order_update(position, velocity, rare_weight, duration, config)
    return momentum_update(position, velocity, rare_weight, duration, config)


def terminal_state(weights: np.ndarray, alpha0: float, config: MomentumConfig) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros(len(weights), dtype=float)
    velocity = np.zeros(len(weights), dtype=float)
    position, velocity = phase_update(position, velocity, weights[:, 0, 1], alpha0, config)
    return phase_update(position, velocity, weights[:, 1, 1], 1.0 - alpha0, config)


def response_feature(weights: np.ndarray, alpha0: float, config: MomentumConfig) -> np.ndarray:
    position, _velocity = terminal_state(weights, alpha0, config)
    return (position - config.evaluation_center) ** 2


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    alpha0: float,
    config: MomentumConfig,
) -> MomentumModel:
    feature = response_feature(weights, alpha0, config)
    head = scalar_hessian.fit_quadratic_head(feature, target, indices, config.l2)
    return MomentumModel(alpha0, config, head)
