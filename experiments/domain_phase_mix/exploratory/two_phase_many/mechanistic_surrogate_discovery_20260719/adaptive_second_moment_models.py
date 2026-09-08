# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Adaptive-second-moment gradient-flow models for two-phase policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

INTEGRATION_STEPS = 256
NUMERICAL_FLOOR = 1e-12


class Dynamics(StrEnum):
    GRADIENT_FLOW = "gradient_flow"
    ADAPTIVE_SECOND_MOMENT = "adaptive_second_moment"


@dataclass(frozen=True)
class AdaptiveMomentConfig:
    dynamics: Dynamics
    curvature_ratio: float
    speed: float
    memory_rate: float
    epsilon: float
    evaluation_mix: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"dynamics={self.dynamics.value},curvature={self.curvature_ratio:g},speed={self.speed:g},"
            f"memory={self.memory_rate:g},epsilon={self.epsilon:g},eval={self.evaluation_mix:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class ResponseHead:
    intercept: float
    amplitude: float


@dataclass(frozen=True)
class AdaptiveMomentModel:
    config: AdaptiveMomentConfig
    phase0_fraction: float
    head: ResponseHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        feature = response_feature(weights, self.phase0_fraction, self.config)
        return self.head.intercept + self.head.amplitude * feature


def mixture_gradient(position: np.ndarray, rare_weight: np.ndarray, curvature_ratio: float) -> np.ndarray:
    broad = (1.0 - rare_weight) * (position + 0.5)
    rare = curvature_ratio * rare_weight * (position - 0.5)
    return broad + rare


def gradient_flow_update(
    position: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: AdaptiveMomentConfig,
) -> np.ndarray:
    hessian = 1.0 - rare_weight + config.curvature_ratio * rare_weight
    equilibrium = 0.5 * (1.0 - rare_weight - config.curvature_ratio * rare_weight) / hessian
    return equilibrium + (position - equilibrium) * np.exp(-config.speed * hessian * duration)


def adaptive_rhs(
    position: np.ndarray,
    second_moment: np.ndarray,
    rare_weight: np.ndarray,
    config: AdaptiveMomentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    gradient = mixture_gradient(position, rare_weight, config.curvature_ratio)
    preconditioner = np.sqrt(np.maximum(second_moment, 0.0)) + config.epsilon
    position_rate = -config.speed * gradient / preconditioner
    moment_rate = config.memory_rate * (gradient**2 - second_moment)
    return position_rate, moment_rate


def adaptive_update(
    position: np.ndarray,
    second_moment: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: AdaptiveMomentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(np.ceil(INTEGRATION_STEPS * duration)))
    step = duration / steps
    current_position = position.copy()
    current_moment = second_moment.copy()
    for _ in range(steps):
        k1_position, k1_moment = adaptive_rhs(current_position, current_moment, rare_weight, config)
        k2_position, k2_moment = adaptive_rhs(
            current_position + 0.5 * step * k1_position,
            np.maximum(current_moment + 0.5 * step * k1_moment, 0.0),
            rare_weight,
            config,
        )
        k3_position, k3_moment = adaptive_rhs(
            current_position + 0.5 * step * k2_position,
            np.maximum(current_moment + 0.5 * step * k2_moment, 0.0),
            rare_weight,
            config,
        )
        k4_position, k4_moment = adaptive_rhs(
            current_position + step * k3_position,
            np.maximum(current_moment + step * k3_moment, 0.0),
            rare_weight,
            config,
        )
        current_position += step * (k1_position + 2.0 * k2_position + 2.0 * k3_position + k4_position) / 6.0
        current_moment = np.maximum(
            current_moment + step * (k1_moment + 2.0 * k2_moment + 2.0 * k3_moment + k4_moment) / 6.0,
            0.0,
        )
    if not np.isfinite(current_position).all() or not np.isfinite(current_moment).all():
        raise FloatingPointError(f"Non-finite adaptive state for {config.key}")
    return current_position, current_moment


def terminal_state(
    weights: np.ndarray,
    phase0_fraction: float,
    config: AdaptiveMomentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros(len(weights), dtype=float)
    second_moment = np.zeros(len(weights), dtype=float)
    for phase, duration in ((0, phase0_fraction), (1, 1.0 - phase0_fraction)):
        rare_weight = weights[:, phase, 1]
        if config.dynamics == Dynamics.GRADIENT_FLOW:
            position = gradient_flow_update(position, rare_weight, duration, config)
            second_moment.fill(0.0)
        else:
            position, second_moment = adaptive_update(position, second_moment, rare_weight, duration, config)
    return position, second_moment


def response_feature(
    weights: np.ndarray,
    phase0_fraction: float,
    config: AdaptiveMomentConfig,
) -> np.ndarray:
    position, _second_moment = terminal_state(weights, phase0_fraction, config)
    broad_loss = 0.5 * (position + 0.5) ** 2
    rare_loss = 0.5 * config.curvature_ratio * (position - 0.5) ** 2
    return (1.0 - config.evaluation_mix) * broad_loss + config.evaluation_mix * rare_loss


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    phase0_fraction: float,
    config: AdaptiveMomentConfig,
) -> AdaptiveMomentModel:
    feature = response_feature(weights, phase0_fraction, config)
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
    return AdaptiveMomentModel(config, phase0_fraction, ResponseHead(intercept, amplitude))


def equivalent_tied_error(config: AdaptiveMomentConfig, weights: np.ndarray, split: float) -> float:
    tied = np.repeat(weights[:, None, :], 2, axis=1)
    position, moment = terminal_state(tied, split, config)
    whole = np.repeat(weights[:, None, :], 2, axis=1)
    whole_position, whole_moment = terminal_state(whole, 1.0, config)
    return max(
        float(np.max(np.abs(position - whole_position))),
        float(np.max(np.abs(moment - whole_moment))),
    )
