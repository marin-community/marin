# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-ordered discrete MuonH transition for a two-task matrix model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_anisotropic_polar_models as task,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_polar_matrix_models as muon,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    starcoder_optimizer_schedule as schedules,
)

SOURCE_MOMENTUM = 0.95
SOURCE_PEAK_LEARNING_RATE = 0.02
GRADIENT_CLIP_NORM = 1.0


@dataclass(frozen=True)
class SourceDiscreteMuonConfig:
    """Task geometry and whether the source momentum state is active."""

    task_angle_degrees: float
    rare_curvature: float
    input_anisotropy: float
    evaluation_rare_weight: float
    momentum: float

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},rare={self.rare_curvature:g},"
            f"anisotropy={self.input_anisotropy:g},eval={self.evaluation_rare_weight:g},"
            f"momentum={self.momentum:g}"
        )


def task_config(config: SourceDiscreteMuonConfig) -> task.MuonAnisotropicPolarConfig:
    return task.MuonAnisotropicPolarConfig(
        config.task_angle_degrees,
        config.rare_curvature,
        config.input_anisotropy,
        1.0,
        config.evaluation_rare_weight,
        "newton_schulz",
    )


def clipped_gradient(state: np.ndarray, rare_weight: np.ndarray, config: SourceDiscreteMuonConfig) -> np.ndarray:
    gradient = task.task_gradient(state, rare_weight, task_config(config))
    norm = np.linalg.norm(gradient, axis=(1, 2), keepdims=True)
    scale = np.minimum(1.0, GRADIENT_CLIP_NORM / np.maximum(norm, muon.NUMERICAL_FLOOR))
    return gradient * scale


def projected_step(state: np.ndarray, direction: np.ndarray, learning_rate: float) -> np.ndarray:
    state_norm = np.linalg.norm(state, axis=(1, 2), keepdims=True)
    direction_norm = np.linalg.norm(direction, axis=(1, 2), keepdims=True)
    intermediate = state - learning_rate * direction * state_norm / np.maximum(direction_norm, 1e-10)
    intermediate_norm = np.linalg.norm(intermediate, axis=(1, 2), keepdims=True)
    return intermediate * state_norm / np.maximum(intermediate_norm, 1e-10)


def terminal_state(
    weights: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    config: SourceDiscreteMuonConfig,
) -> np.ndarray:
    weights = muon.validate_weights(weights)
    state = muon.initial_state(len(weights))
    momentum = np.zeros_like(state)
    learning_rate = schedule.learning_rate_at_steps(np.arange(schedule.total_steps, dtype=float))
    for step in range(schedule.total_steps):
        phase = 0 if step < schedule.phase_boundary_step else 1
        gradient = clipped_gradient(state, weights[:, phase, 1], config)
        momentum = config.momentum * momentum + gradient
        nesterov = config.momentum * momentum + gradient
        direction = muon.update_direction(nesterov, "newton_schulz")
        state = projected_step(state, direction, SOURCE_PEAK_LEARNING_RATE * float(learning_rate[step]))
    if not np.isfinite(state).all():
        raise FloatingPointError(f"Non-finite source-discrete state for {config.key}")
    return state


def response_feature(
    weights: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    config: SourceDiscreteMuonConfig,
) -> np.ndarray:
    state = terminal_state(weights, schedule, config)
    config_as_task = task_config(config)
    broad_target, rare_target = task.task_targets(config_as_task)
    broad_covariance, rare_covariance = task.task_covariances(config_as_task)
    broad_debt = task.weighted_task_debt(state, broad_target, broad_covariance)
    rare_debt = task.weighted_task_debt(state, rare_target, rare_covariance)
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad_debt + q * config.rare_curvature * rare_debt
