# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Balanced matrix-factorization dynamics for competing task targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 256
INITIAL_FACTOR = 0.1


@dataclass(frozen=True)
class MatrixFactorizationConfig:
    """Dimensionless target geometry and factor-learning clock."""

    task_angle_degrees: float
    relaxation: float
    rare_curvature: float
    evaluation_rare_weight: float
    transition: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},relax={self.relaxation:g},"
            f"rare_curvature={self.rare_curvature:g},eval={self.evaluation_rare_weight:g},"
            f"transition={self.transition}"
        )


def task_targets(config: MatrixFactorizationConfig) -> tuple[np.ndarray, np.ndarray]:
    broad_vector = np.asarray([1.0, 0.0])
    angle = np.deg2rad(config.task_angle_degrees)
    rare_vector = np.asarray([np.cos(angle), np.sin(angle)])
    return np.outer(broad_vector, broad_vector), np.outer(rare_vector, rare_vector)


def mixed_gradient(weight: np.ndarray, rare_weight: np.ndarray, config: MatrixFactorizationConfig) -> np.ndarray:
    broad, rare = task_targets(config)
    return (1.0 - rare_weight)[:, None, None] * (weight - broad) + config.rare_curvature * rare_weight[:, None, None] * (
        weight - rare
    )


def factor_derivative(
    left: np.ndarray,
    right: np.ndarray,
    rare_weight: np.ndarray,
    config: MatrixFactorizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.einsum("nij,nkj->nik", left, right)
    gradient = mixed_gradient(weight, rare_weight, config)
    left_derivative = -config.relaxation * np.einsum("nij,njk->nik", gradient, right)
    right_derivative = -config.relaxation * np.einsum("nij,njk->nik", np.swapaxes(gradient, 1, 2), left)
    return left_derivative, right_derivative


def factor_phase_update(
    left: np.ndarray,
    right: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MatrixFactorizationConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    clock_resolution = max(1.0, config.relaxation / 4.0)
    steps = max(1, int(np.ceil(duration * steps_per_unit * clock_resolution)))
    step_size = duration / steps
    left = np.asarray(left, dtype=float).copy()
    right = np.asarray(right, dtype=float).copy()
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        l1, r1 = factor_derivative(left, right, rare_weight, config)
        l2, r2 = factor_derivative(
            left + 0.5 * step_size * l1,
            right + 0.5 * step_size * r1,
            rare_weight,
            config,
        )
        l3, r3 = factor_derivative(
            left + 0.5 * step_size * l2,
            right + 0.5 * step_size * r2,
            rare_weight,
            config,
        )
        l4, r4 = factor_derivative(left + step_size * l3, right + step_size * r3, rare_weight, config)
        left += step_size * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
        right += step_size * (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise FloatingPointError(f"Non-finite matrix-factor state for {config.key}")
    return left, right


def direct_phase_update(
    weight: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MatrixFactorizationConfig,
) -> np.ndarray:
    broad, rare = task_targets(config)
    hessian = (1.0 - rare_weight) + config.rare_curvature * rare_weight
    equilibrium = (
        (1.0 - rare_weight)[:, None, None] * broad + config.rare_curvature * rare_weight[:, None, None] * rare
    ) / hessian[:, None, None]
    decay = np.exp(-config.relaxation * hessian * duration)
    return equilibrium + (weight - equilibrium) * decay[:, None, None]


def terminal_weight(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MatrixFactorizationConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, domain] weights, got {weights.shape}")
    if config.transition == "direct":
        weight = np.repeat((INITIAL_FACTOR**2 * np.eye(2))[None, :, :], len(weights), axis=0)
        weight = direct_phase_update(weight, weights[:, 0, 1], phase0_optimizer_fraction, config)
        return direct_phase_update(weight, weights[:, 1, 1], 1.0 - phase0_optimizer_fraction, config)
    if config.transition != "factorized":
        raise ValueError(f"Unknown transition {config.transition}")
    left = np.repeat((INITIAL_FACTOR * np.eye(2))[None, :, :], len(weights), axis=0)
    right = left.copy()
    left, right = factor_phase_update(
        left,
        right,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    left, right = factor_phase_update(
        left,
        right,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return np.einsum("nij,nkj->nik", left, right)


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MatrixFactorizationConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weight = terminal_weight(weights, phase0_optimizer_fraction, config, steps_per_unit=steps_per_unit)
    broad, rare = task_targets(config)
    broad_debt = np.sum((weight - broad) ** 2, axis=(1, 2))
    rare_debt = np.sum((weight - rare) ** 2, axis=(1, 2))
    q = config.evaluation_rare_weight
    return 0.5 * ((1.0 - q) * broad_debt + q * rare_debt)


def integration_error(weights: np.ndarray, phase0_optimizer_fraction: float, config: MatrixFactorizationConfig) -> float:
    if config.transition == "direct":
        return 0.0
    coarse = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=256)
    fine = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=768)
    return float(np.max(np.abs(coarse - fine)))
