# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite-step modified-equation dynamics for quadratic two-task flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    noncommuting_gradient_flow_models as quadratic,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    starcoder_optimizer_schedule as schedules,
)

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class FiniteStepQuadraticConfig:
    """Frozen task geometry and modified-equation order."""

    curvature_ratio: float
    anisotropy: float
    angle_degrees: float
    total_relaxation: float
    evaluation_center: float
    expansion_order: int

    @property
    def key(self) -> str:
        return (
            f"curvature={self.curvature_ratio:g},anisotropy={self.anisotropy:g},"
            f"angle={self.angle_degrees:g},relax={self.total_relaxation:g},"
            f"eval={self.evaluation_center:g},order={self.expansion_order}"
        )


def quadratic_config(config: FiniteStepQuadraticConfig) -> quadratic.NoncommutingConfig:
    return quadratic.NoncommutingConfig(
        config.curvature_ratio,
        config.anisotropy,
        config.angle_degrees,
        1.0,
        config.evaluation_center,
        0.0,
    )


def segment_learning_rates(
    schedule: schedules.OptimizerScheduleSpec,
    phase: int,
) -> np.ndarray:
    learning_rate = schedule.learning_rate_at_steps(np.arange(schedule.total_steps, dtype=float))
    if phase == 0:
        return learning_rate[: schedule.phase_boundary_step]
    if phase == 1:
        return learning_rate[schedule.phase_boundary_step :]
    raise ValueError(f"Unknown phase {phase}")


def modified_log_multiplier(
    eigenvalues: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    phase: int,
    config: FiniteStepQuadraticConfig,
) -> np.ndarray:
    """Return the truncated log of the exact GD transition multiplier."""

    learning_rate = segment_learning_rates(schedule, phase)
    total_mass = sum(schedule.phase_learning_rate_masses())
    scaled_eigenvalue = config.total_relaxation * np.asarray(eigenvalues, dtype=float)
    maximum_increment = float(np.max(scaled_eigenvalue)) * float(np.max(learning_rate)) / total_mass
    if maximum_increment >= 0.8:
        raise ValueError(f"Unstable finite-step configuration {config.key}: max increment={maximum_increment:g}")
    result = np.zeros_like(scaled_eigenvalue)
    for order in range(1, config.expansion_order + 1):
        normalized_moment = float(np.sum(learning_rate**order)) / total_mass**order
        result -= scaled_eigenvalue**order * normalized_moment / order
    return result


def exact_log_multiplier(
    eigenvalues: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    phase: int,
    config: FiniteStepQuadraticConfig,
) -> np.ndarray:
    learning_rate = segment_learning_rates(schedule, phase)
    total_mass = sum(schedule.phase_learning_rate_masses())
    increments = (
        config.total_relaxation
        * learning_rate[:, None, None]
        * np.asarray(eigenvalues, dtype=float)[None, :, :]
        / total_mass
    )
    if np.any(increments >= 1.0):
        raise ValueError(f"Exact finite-step transition is unstable for {config.key}")
    return np.sum(np.log1p(-increments), axis=0)


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    phase: int,
    config: FiniteStepQuadraticConfig,
    *,
    exact: bool = False,
) -> np.ndarray:
    hessian, equilibrium = quadratic.phase_generator(rare_weight, quadratic_config(config))
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    log_multiplier = (
        exact_log_multiplier(eigenvalues, schedule, phase, config)
        if exact
        else modified_log_multiplier(eigenvalues, schedule, phase, config)
    )
    displacement = np.asarray(state, dtype=float) - equilibrium
    projected = np.einsum("nji,nj->ni", eigenvectors, displacement)
    projected *= np.exp(log_multiplier)
    return equilibrium + np.einsum("nij,nj->ni", eigenvectors, projected)


def terminal_state(
    weights: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    config: FiniteStepQuadraticConfig,
    *,
    exact: bool = False,
) -> np.ndarray:
    weights = quadratic.normalized_policy(weights)
    state = np.zeros((len(weights), 2), dtype=float)
    state = phase_update(state, weights[:, 0, 1], schedule, 0, config, exact=exact)
    return phase_update(state, weights[:, 1, 1], schedule, 1, config, exact=exact)


def response_feature(
    weights: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    config: FiniteStepQuadraticConfig,
    *,
    exact: bool = False,
) -> np.ndarray:
    state = terminal_state(weights, schedule, config, exact=exact)
    center = np.asarray([config.evaluation_center, 0.0], dtype=float)
    return np.sum((state - center[None, :]) ** 2, axis=1)


def exact_approximation_error(
    weights: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    config: FiniteStepQuadraticConfig,
) -> float:
    approximate = response_feature(weights, schedule, config)
    exact = response_feature(weights, schedule, config, exact=True)
    return float(np.max(np.abs(approximate - exact)))


def tied_boundary_error(
    rare_weight: np.ndarray,
    schedule: schedules.OptimizerScheduleSpec,
    config: FiniteStepQuadraticConfig,
) -> float:
    """Compare split and unsplit exact products for a phase-tied policy."""

    rare_weight = np.asarray(rare_weight, dtype=float)
    hessian, equilibrium = quadratic.phase_generator(rare_weight, quadratic_config(config))
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    whole_learning_rate = schedule.learning_rate_at_steps(np.arange(schedule.total_steps, dtype=float))
    total_mass = sum(schedule.phase_learning_rate_masses())
    increments = config.total_relaxation * whole_learning_rate[:, None, None] * eigenvalues[None, :, :] / total_mass
    if np.any(increments >= 1.0):
        raise ValueError(f"Tied exact transition is unstable for {config.key}")
    whole_log = np.sum(np.log1p(-increments), axis=0)
    split_log = exact_log_multiplier(eigenvalues, schedule, 0, config) + exact_log_multiplier(
        eigenvalues, schedule, 1, config
    )
    initial = np.zeros((len(rare_weight), 2), dtype=float)
    projected = np.einsum("nji,nj->ni", eigenvectors, initial - equilibrium)
    whole = equilibrium + np.einsum("nij,nj->ni", eigenvectors, projected * np.exp(whole_log))
    split = equilibrium + np.einsum("nij,nj->ni", eigenvectors, projected * np.exp(split_log))
    return float(np.max(np.abs(whole - split)))
