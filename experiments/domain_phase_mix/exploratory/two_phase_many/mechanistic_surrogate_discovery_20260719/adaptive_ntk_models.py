# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained task-space kernel alignment and residual transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 192


@dataclass(frozen=True)
class AdaptiveKernelConfig:
    """Dimensionless kernel geometry and adaptation clocks."""

    task_angle_degrees: float
    anisotropy: float
    kernel_adaptation: float
    residual_relaxation: float
    rare_curvature: float
    evaluation_rare_weight: float
    transition: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},anisotropy={self.anisotropy:g},"
            f"kernel={self.kernel_adaptation:g},residual={self.residual_relaxation:g},"
            f"rare_curvature={self.rare_curvature:g},eval={self.evaluation_rare_weight:g},"
            f"transition={self.transition}"
        )


def rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def target_kernels(config: AdaptiveKernelConfig) -> tuple[np.ndarray, np.ndarray]:
    strong = config.anisotropy / (1.0 + config.anisotropy)
    weak = 1.0 / (1.0 + config.anisotropy)
    broad = np.diag([strong, weak])
    rotate = rotation(config.task_angle_degrees)
    rare = rotate @ np.diag([strong, weak]) @ rotate.T
    return broad, rare


def kernel_target(rare_weight: np.ndarray, config: AdaptiveKernelConfig) -> np.ndarray:
    broad, rare = target_kernels(config)
    return (1.0 - rare_weight)[:, None, None] * broad + rare_weight[:, None, None] * rare


def derivative(
    kernel: np.ndarray,
    residual: np.ndarray,
    rare_weight: np.ndarray,
    config: AdaptiveKernelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    target = kernel_target(rare_weight, config)
    if config.transition == "adaptive":
        kernel_derivative = config.kernel_adaptation * (target - kernel)
        active_kernel = kernel
    elif config.transition == "frozen":
        kernel_derivative = np.zeros_like(kernel)
        active_kernel = kernel
    elif config.transition == "instantaneous":
        kernel_derivative = np.zeros_like(kernel)
        active_kernel = target
    else:
        raise ValueError(f"Unknown transition {config.transition}")
    weighted_residual = residual * np.column_stack([1.0 - rare_weight, config.rare_curvature * rare_weight])
    residual_derivative = -config.residual_relaxation * np.einsum("nij,nj->ni", active_kernel, weighted_residual)
    return kernel_derivative, residual_derivative


def phase_update(
    kernel: np.ndarray,
    residual: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: AdaptiveKernelConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(np.ceil(duration * steps_per_unit)))
    step_size = duration / steps
    kernel = np.asarray(kernel, dtype=float).copy()
    residual = np.asarray(residual, dtype=float).copy()
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        k1, e1 = derivative(kernel, residual, rare_weight, config)
        k2, e2 = derivative(kernel + 0.5 * step_size * k1, residual + 0.5 * step_size * e1, rare_weight, config)
        k3, e3 = derivative(kernel + 0.5 * step_size * k2, residual + 0.5 * step_size * e2, rare_weight, config)
        k4, e4 = derivative(kernel + step_size * k3, residual + step_size * e3, rare_weight, config)
        kernel += step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        residual += step_size * (e1 + 2.0 * e2 + 2.0 * e3 + e4) / 6.0
    if not np.isfinite(kernel).all() or not np.isfinite(residual).all():
        raise FloatingPointError(f"Non-finite adaptive-kernel state for {config.key}")
    if config.transition != "instantaneous" and float(np.linalg.eigvalsh(kernel).min()) <= 0.0:
        raise FloatingPointError(f"Adaptive kernel lost positive definiteness for {config.key}")
    return kernel, residual


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: AdaptiveKernelConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, domain] weights, got {weights.shape}")
    kernel = np.repeat((0.5 * np.eye(2))[None, :, :], len(weights), axis=0)
    residual = np.ones((len(weights), 2), dtype=float)
    kernel, residual = phase_update(
        kernel,
        residual,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return phase_update(
        kernel,
        residual,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: AdaptiveKernelConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    _kernel, residual = terminal_state(weights, phase0_optimizer_fraction, config, steps_per_unit=steps_per_unit)
    q = config.evaluation_rare_weight
    return 0.5 * ((1.0 - q) * residual[:, 0] ** 2 + q * residual[:, 1] ** 2)


def integration_error(weights: np.ndarray, phase0_optimizer_fraction: float, config: AdaptiveKernelConfig) -> float:
    coarse = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=192)
    fine = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=576)
    return float(np.max(np.abs(coarse - fine)))
