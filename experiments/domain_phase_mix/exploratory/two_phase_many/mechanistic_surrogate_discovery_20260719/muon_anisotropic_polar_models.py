# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon matrix-polar flow with domain-specific anisotropic input covariance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_polar_matrix_models as isotropic,
)

INTEGRATION_STEPS_PER_UNIT = 64
NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class MuonAnisotropicPolarConfig:
    """Frozen task geometry, covariance, transition rule, and response."""

    task_angle_degrees: float
    rare_curvature: float
    input_anisotropy: float
    relaxation: float
    evaluation_rare_weight: float
    update_rule: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},rare={self.rare_curvature:g},"
            f"anisotropy={self.input_anisotropy:g},relax={self.relaxation:g},"
            f"eval={self.evaluation_rare_weight:g},rule={self.update_rule}"
        )


def task_targets(config: MuonAnisotropicPolarConfig) -> tuple[np.ndarray, np.ndarray]:
    broad = np.eye(isotropic.MATRIX_SIZE, dtype=float) / np.sqrt(isotropic.MATRIX_SIZE)
    rare = isotropic.rotation(config.task_angle_degrees) / np.sqrt(isotropic.MATRIX_SIZE)
    return broad, rare


def task_covariances(config: MuonAnisotropicPolarConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-trace broad and rare second-moment matrices."""

    scale = 2.0 / (1.0 + config.input_anisotropy)
    broad = scale * np.diag([1.0, config.input_anisotropy])
    rotation = isotropic.rotation(config.task_angle_degrees)
    rare = scale * rotation @ np.diag([config.input_anisotropy, 1.0]) @ rotation.T
    return broad, rare


def task_gradient(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: MuonAnisotropicPolarConfig,
) -> np.ndarray:
    broad_target, rare_target = task_targets(config)
    broad_covariance, rare_covariance = task_covariances(config)
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    broad_gradient = np.einsum("nij,jk->nik", state - broad_target[None, :, :], broad_covariance)
    rare_gradient = np.einsum("nij,jk->nik", state - rare_target[None, :, :], rare_covariance)
    return (1.0 - rare_weight)[:, None, None] * broad_gradient + config.rare_curvature * rare_weight[
        :, None, None
    ] * rare_gradient


def derivative(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: MuonAnisotropicPolarConfig,
) -> np.ndarray:
    gradient = task_gradient(state, rare_weight, config)
    direction = isotropic.update_direction(gradient, config.update_rule)
    radial = np.sum(state * direction, axis=(1, 2), keepdims=True)
    return -config.relaxation * (direction - radial * state)


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MuonAnisotropicPolarConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    steps = max(1, int(np.ceil(steps_per_unit * duration)))
    step_size = duration / steps
    state = isotropic.normalize_state(np.asarray(state, dtype=float).copy())
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        k1 = derivative(state, rare_weight, config)
        k2 = derivative(isotropic.normalize_state(state + 0.5 * step_size * k1), rare_weight, config)
        k3 = derivative(isotropic.normalize_state(state + 0.5 * step_size * k2), rare_weight, config)
        k4 = derivative(isotropic.normalize_state(state + step_size * k3), rare_weight, config)
        state = isotropic.normalize_state(state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
    if not np.isfinite(state).all():
        raise FloatingPointError(f"Non-finite anisotropic matrix state for {config.key}")
    return state


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MuonAnisotropicPolarConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weights = isotropic.validate_weights(weights)
    state = isotropic.initial_state(len(weights))
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


def weighted_task_debt(
    state: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    displacement = state - target[None, :, :]
    transformed = np.einsum("nij,jk->nik", displacement, covariance)
    return 0.5 * np.sum(displacement * transformed, axis=(1, 2))


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MuonAnisotropicPolarConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    state = terminal_state(
        weights,
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    broad_target, rare_target = task_targets(config)
    broad_covariance, rare_covariance = task_covariances(config)
    broad_debt = weighted_task_debt(state, broad_target, broad_covariance)
    rare_debt = weighted_task_debt(state, rare_target, rare_covariance)
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad_debt + q * config.rare_curvature * rare_debt


def trajectory_rule_separation(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MuonAnisotropicPolarConfig,
) -> float:
    normalized = MuonAnisotropicPolarConfig(
        config.task_angle_degrees,
        config.rare_curvature,
        config.input_anisotropy,
        config.relaxation,
        config.evaluation_rare_weight,
        "normalized",
    )
    polar = MuonAnisotropicPolarConfig(
        config.task_angle_degrees,
        config.rare_curvature,
        config.input_anisotropy,
        config.relaxation,
        config.evaluation_rare_weight,
        "polar",
    )
    left = terminal_state(weights, phase0_optimizer_fraction, normalized)
    right = terminal_state(weights, phase0_optimizer_fraction, polar)
    return float(np.mean(np.linalg.norm(left - right, axis=(1, 2))))


def integration_error(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MuonAnisotropicPolarConfig,
) -> float:
    coarse = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=64)
    fine = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=192)
    return float(np.max(np.abs(coarse - fine)))
