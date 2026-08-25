# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Noncommuting residual contraction over a fixed Hessian spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurvatureSpectrumConfig:
    """Dimensionless task geometry and Hessian-spectrum quadrature."""

    task_angle_degrees: float
    anisotropy: float
    spectrum_span: float
    spectrum_tilt: float
    relaxation: float
    rare_curvature: float
    evaluation_rare_weight: float
    transition: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},anisotropy={self.anisotropy:g},"
            f"span={self.spectrum_span:g},tilt={self.spectrum_tilt:g},"
            f"relax={self.relaxation:g},rare_curvature={self.rare_curvature:g},"
            f"eval={self.evaluation_rare_weight:g},transition={self.transition}"
        )


def rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def domain_hessians(config: CurvatureSpectrumConfig) -> tuple[np.ndarray, np.ndarray]:
    strong = config.anisotropy / (1.0 + config.anisotropy)
    weak = 1.0 / (1.0 + config.anisotropy)
    broad = np.diag([strong, weak])
    rotate = rotation(config.task_angle_degrees)
    rare = config.rare_curvature * (rotate @ np.diag([strong, weak]) @ rotate.T)
    return broad, rare


def spectral_quadrature(config: CurvatureSpectrumConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.transition == "single_mode":
        return np.asarray([1.0]), np.asarray([1.0])
    rates = np.asarray(
        [config.spectrum_span**-0.5, 1.0, config.spectrum_span**0.5],
        dtype=float,
    )
    weights = rates**config.spectrum_tilt
    return rates, weights / np.sum(weights)


def matrix_exponential_apply(matrix: np.ndarray, vector: np.ndarray, scale: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    coordinates = np.einsum("nij,nj->ni", np.swapaxes(eigenvectors, 1, 2), vector)
    contracted = coordinates * np.exp(-scale * eigenvalues)
    return np.einsum("nij,nj->ni", eigenvectors, contracted)


def phase_update(
    residuals: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    rate: float,
    config: CurvatureSpectrumConfig,
) -> np.ndarray:
    broad, rare = domain_hessians(config)
    hessian = (1.0 - rare_weight)[:, None, None] * broad + rare_weight[:, None, None] * rare
    return matrix_exponential_apply(hessian, residuals, config.relaxation * rate * duration)


def terminal_residuals(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: CurvatureSpectrumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, domain] weights, got {weights.shape}")
    rates, mode_weights = spectral_quadrature(config)
    modes = []
    for rate in rates:
        residual = np.ones((len(weights), 2), dtype=float)
        residual = phase_update(
            residual,
            weights[:, 0, 1],
            phase0_optimizer_fraction,
            float(rate),
            config,
        )
        residual = phase_update(
            residual,
            weights[:, 1, 1],
            1.0 - phase0_optimizer_fraction,
            float(rate),
            config,
        )
        modes.append(residual)
    return np.stack(modes, axis=1), mode_weights


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: CurvatureSpectrumConfig,
) -> np.ndarray:
    residuals, mode_weights = terminal_residuals(weights, phase0_optimizer_fraction, config)
    broad, rare = domain_hessians(config)
    evaluation = (1.0 - config.evaluation_rare_weight) * broad + config.evaluation_rare_weight * rare
    mode_energy = np.einsum("nmi,ij,nmj->nm", residuals, evaluation, residuals)
    return mode_energy @ mode_weights
