# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matrix Riccati information acquisition with orthogonal interference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MATRIX_SIZE = 2
INTEGRATION_STEPS_PER_UNIT = 128


@dataclass(frozen=True)
class MatrixInformationConfig:
    """Dimensionless information geometry and process-to-information ratio."""

    task_angle_degrees: float
    information_anisotropy: float
    process_ratio: float
    relaxation: float
    evaluation_rare_weight: float
    transition: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},anisotropy={self.information_anisotropy:g},"
            f"process={self.process_ratio:g},relax={self.relaxation:g},"
            f"eval={self.evaluation_rare_weight:g},transition={self.transition}"
        )


def rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def domain_matrices(config: MatrixInformationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anisotropy = 1.0 if config.transition == "isotropic" else config.information_anisotropy
    strong = 2.0 * anisotropy / (1.0 + anisotropy)
    weak = 2.0 / (1.0 + anisotropy)
    broad_information = np.diag([strong, weak])
    broad_process = np.diag([weak, strong])
    rotate = rotation(config.task_angle_degrees)
    rare_information = rotate @ np.diag([strong, weak]) @ rotate.T
    rare_process = rotate @ np.diag([weak, strong]) @ rotate.T
    return broad_information, rare_information, broad_process, rare_process


def derivative(
    covariance: np.ndarray,
    rare_weight: np.ndarray,
    config: MatrixInformationConfig,
) -> np.ndarray:
    broad_information, rare_information, broad_process, rare_process = domain_matrices(config)
    rare_weight = np.asarray(rare_weight, dtype=float)
    information = (1.0 - rare_weight)[:, None, None] * broad_information[None, :, :] + rare_weight[
        :, None, None
    ] * rare_information[None, :, :]
    process_ratio = 0.0 if config.transition == "zero_process" else config.process_ratio
    process = process_ratio * (
        (1.0 - rare_weight)[:, None, None] * broad_process[None, :, :]
        + rare_weight[:, None, None] * rare_process[None, :, :]
    )
    contraction = np.einsum("nij,njk,nkl->nil", covariance, information, covariance)
    return config.relaxation * (process - contraction)


def phase_update(
    covariance: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MatrixInformationConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    steps = max(1, int(np.ceil(duration * steps_per_unit)))
    step_size = duration / steps
    covariance = np.asarray(covariance, dtype=float).copy()
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        k1 = derivative(covariance, rare_weight, config)
        k2 = derivative(covariance + 0.5 * step_size * k1, rare_weight, config)
        k3 = derivative(covariance + 0.5 * step_size * k2, rare_weight, config)
        k4 = derivative(covariance + step_size * k3, rare_weight, config)
        covariance += step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    if not np.isfinite(covariance).all():
        raise FloatingPointError(f"Non-finite information covariance for {config.key}")
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if minimum_eigenvalue <= 0.0:
        raise FloatingPointError(f"Information covariance lost positive definiteness ({minimum_eigenvalue})")
    return covariance


def terminal_covariance(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MatrixInformationConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, domain] weights, got {weights.shape}")
    covariance = np.repeat(np.eye(MATRIX_SIZE)[None, :, :], len(weights), axis=0)
    covariance = phase_update(
        covariance,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return phase_update(
        covariance,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MatrixInformationConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    covariance = terminal_covariance(
        weights,
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    broad_information, rare_information, _broad_process, _rare_process = domain_matrices(config)
    q = config.evaluation_rare_weight
    evaluation = (1.0 - q) * broad_information + q * rare_information
    return np.einsum("ij,nji->n", evaluation, covariance)


def integration_error(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MatrixInformationConfig,
) -> float:
    coarse = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=128)
    fine = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=384)
    return float(np.max(np.abs(coarse - fine)))
