# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-dimensional gradient flow with persistent coordinatewise second moments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    hessian_equilibrium_models as scalar_hessian,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    noncommuting_gradient_flow_models as vector_flow,
)

NUMERICAL_FLOOR = 1e-12
INTEGRATION_STEPS_PER_UNIT = 128


@dataclass(frozen=True)
class TaskGeometry:
    """Dimensionless broad/rare quadratic task geometry."""

    curvature_ratio: float
    anisotropy: float
    angle_degrees: float

    def vector_flow_config(self, speed: float) -> vector_flow.NoncommutingConfig:
        return vector_flow.NoncommutingConfig(
            curvature_ratio=self.curvature_ratio,
            anisotropy=self.anisotropy,
            angle_degrees=self.angle_degrees,
            relaxation=speed,
            evaluation_center=0.0,
            l2=0.0,
        )

    @property
    def key(self) -> str:
        return f"curvature={self.curvature_ratio:g},anisotropy={self.anisotropy:g},angle={self.angle_degrees:g}"


@dataclass(frozen=True)
class AdaptiveMomentConfig:
    """Frozen optimizer dynamics and evaluation response."""

    geometry: TaskGeometry
    speed: float
    memory_rate: float
    epsilon: float
    evaluation_rare_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"{self.geometry.key},speed={self.speed:g},memory={self.memory_rate:g},"
            f"epsilon={self.epsilon:g},eval={self.evaluation_rare_weight:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class GradientFlowConfig:
    """Exact no-adaptive ablation with matched geometry and response."""

    geometry: TaskGeometry
    speed: float
    evaluation_rare_weight: float
    l2: float

    @property
    def key(self) -> str:
        return f"{self.geometry.key},speed={self.speed:g},eval={self.evaluation_rare_weight:g},l2={self.l2:g}"


def task_hessians(geometry: TaskGeometry) -> tuple[np.ndarray, np.ndarray]:
    return vector_flow.task_hessians(geometry.vector_flow_config(speed=1.0))


def task_gradient(state: np.ndarray, rare_weight: np.ndarray, geometry: TaskGeometry) -> np.ndarray:
    broad, rare = task_hessians(geometry)
    return task_gradient_with_hessians(state, rare_weight, broad, rare)


def task_gradient_with_hessians(
    state: np.ndarray,
    rare_weight: np.ndarray,
    broad: np.ndarray,
    rare: np.ndarray,
) -> np.ndarray:
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    broad_gradient = (np.asarray(state, dtype=float) - vector_flow.BROAD_OPTIMUM) @ broad
    rare_gradient = (np.asarray(state, dtype=float) - vector_flow.RARE_OPTIMUM) @ rare
    return (1.0 - rare_weight[:, None]) * broad_gradient + rare_weight[:, None] * rare_gradient


def adaptive_derivative(
    state: np.ndarray,
    second_moment: np.ndarray,
    rare_weight: np.ndarray,
    config: AdaptiveMomentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    gradient = task_gradient(state, rare_weight, config.geometry)
    safe_moment = np.maximum(second_moment, 0.0)
    state_derivative = -config.speed * gradient / (np.sqrt(safe_moment) + config.epsilon)
    moment_derivative = config.memory_rate * (gradient**2 - safe_moment)
    return state_derivative, moment_derivative


def adaptive_phase_update(
    state: np.ndarray,
    second_moment: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: AdaptiveMomentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(np.ceil(INTEGRATION_STEPS_PER_UNIT * duration)))
    step_size = duration / steps
    state = np.asarray(state, dtype=float).copy()
    second_moment = np.asarray(second_moment, dtype=float).copy()
    broad, rare = task_hessians(config.geometry)

    def derivative(local_state: np.ndarray, local_moment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gradient = task_gradient_with_hessians(local_state, rare_weight, broad, rare)
        safe_moment = np.maximum(local_moment, 0.0)
        state_derivative = -config.speed * gradient / (np.sqrt(safe_moment) + config.epsilon)
        moment_derivative = config.memory_rate * (gradient**2 - safe_moment)
        return state_derivative, moment_derivative

    for _ in range(steps):
        k1_state, k1_moment = derivative(state, second_moment)
        k2_state, k2_moment = derivative(
            state + 0.5 * step_size * k1_state,
            second_moment + 0.5 * step_size * k1_moment,
        )
        k3_state, k3_moment = derivative(
            state + 0.5 * step_size * k2_state,
            second_moment + 0.5 * step_size * k2_moment,
        )
        k4_state, k4_moment = derivative(
            state + step_size * k3_state,
            second_moment + step_size * k3_moment,
        )
        state += step_size * (k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state) / 6.0
        second_moment += step_size * (k1_moment + 2.0 * k2_moment + 2.0 * k3_moment + k4_moment) / 6.0
        second_moment = np.maximum(second_moment, 0.0)
    return state, second_moment


def adaptive_terminal_state(
    weights: np.ndarray,
    alpha0: float,
    config: AdaptiveMomentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    weights = vector_flow.normalized_policy(weights)
    state = np.zeros((len(weights), 2), dtype=float)
    second_moment = np.zeros_like(state)
    state, second_moment = adaptive_phase_update(
        state,
        second_moment,
        weights[:, 0, 1],
        alpha0,
        config,
    )
    return adaptive_phase_update(
        state,
        second_moment,
        weights[:, 1, 1],
        1.0 - alpha0,
        config,
    )


def gradient_flow_terminal_state(
    weights: np.ndarray,
    alpha0: float,
    config: GradientFlowConfig,
) -> np.ndarray:
    weights = vector_flow.normalized_policy(weights)
    vector_config = config.geometry.vector_flow_config(config.speed)
    state = np.zeros((len(weights), 2), dtype=float)
    state = vector_flow.matrix_relaxation_update(state, weights[:, 0, 1], alpha0, vector_config)
    return vector_flow.matrix_relaxation_update(state, weights[:, 1, 1], 1.0 - alpha0, vector_config)


def evaluation_potential(
    state: np.ndarray,
    geometry: TaskGeometry,
    evaluation_rare_weight: float,
) -> np.ndarray:
    broad, rare = task_hessians(geometry)
    broad_displacement = np.asarray(state, dtype=float) - vector_flow.BROAD_OPTIMUM
    rare_displacement = np.asarray(state, dtype=float) - vector_flow.RARE_OPTIMUM
    broad_loss = 0.5 * np.einsum("ni,ij,nj->n", broad_displacement, broad, broad_displacement)
    rare_loss = 0.5 * np.einsum("ni,ij,nj->n", rare_displacement, rare, rare_displacement)
    return (1.0 - evaluation_rare_weight) * broad_loss + evaluation_rare_weight * rare_loss


def adaptive_response_feature(weights: np.ndarray, alpha0: float, config: AdaptiveMomentConfig) -> np.ndarray:
    state, _second_moment = adaptive_terminal_state(weights, alpha0, config)
    return evaluation_potential(state, config.geometry, config.evaluation_rare_weight)


def gradient_flow_response_feature(weights: np.ndarray, alpha0: float, config: GradientFlowConfig) -> np.ndarray:
    state = gradient_flow_terminal_state(weights, alpha0, config)
    return evaluation_potential(state, config.geometry, config.evaluation_rare_weight)


def fit_head(feature: np.ndarray, target: np.ndarray, indices: np.ndarray, l2: float) -> scalar_hessian.QuadraticHead:
    return scalar_hessian.fit_quadratic_head(feature, target, indices, l2)
