# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact two-dimensional gradient flow with noncommuting task Hessians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    hessian_equilibrium_models as scalar_hessian,
)

NUMERICAL_FLOOR = 1e-12
BROAD_OPTIMUM = np.asarray([-0.5, 0.0], dtype=float)
RARE_OPTIMUM = np.asarray([0.5, 0.0], dtype=float)


@dataclass(frozen=True)
class NoncommutingConfig:
    """Frozen geometry for two task Hessians and one evaluation bowl."""

    curvature_ratio: float
    anisotropy: float
    angle_degrees: float
    relaxation: float
    evaluation_center: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"curvature={self.curvature_ratio:g},anisotropy={self.anisotropy:g},"
            f"angle={self.angle_degrees:g},relaxation={self.relaxation:g},"
            f"eval={self.evaluation_center:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class NoncommutingModel:
    """Fitted response over an exact ordered matrix-flow state."""

    alpha0: float
    config: NoncommutingConfig
    head: scalar_hessian.QuadraticHead

    def state(self, weights: np.ndarray) -> np.ndarray:
        return terminal_state(weights, self.alpha0, self.config)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.head.predict(response_feature(weights, self.alpha0, self.config))


def task_hessians(config: NoncommutingConfig) -> tuple[np.ndarray, np.ndarray]:
    broad = np.diag([1.0, config.anisotropy])
    angle = np.deg2rad(config.angle_degrees)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rare = config.curvature_ratio * rotation @ np.diag([1.0, config.anisotropy]) @ rotation.T
    return broad, rare


def normalized_commutator(config: NoncommutingConfig) -> float:
    broad, rare = task_hessians(config)
    commutator = broad @ rare - rare @ broad
    return float(
        np.linalg.norm(commutator, ord="fro")
        / max(np.linalg.norm(broad, ord="fro") * np.linalg.norm(rare, ord="fro"), NUMERICAL_FLOOR)
    )


def normalized_policy(weights: np.ndarray) -> np.ndarray:
    return scalar_hessian.normalized_policy(weights)


def phase_generator(rare_weight: np.ndarray, config: NoncommutingConfig) -> tuple[np.ndarray, np.ndarray]:
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    broad, rare = task_hessians(config)
    hessian = (1.0 - rare_weight)[:, None, None] * broad[None, :, :] + rare_weight[:, None, None] * rare[None, :, :]
    broad_linear = broad @ BROAD_OPTIMUM
    rare_linear = rare @ RARE_OPTIMUM
    linear = (1.0 - rare_weight)[:, None] * broad_linear[None, :] + rare_weight[:, None] * rare_linear[None, :]
    equilibrium = np.linalg.solve(hessian, linear[..., None])[..., 0]
    return hessian, equilibrium


def matrix_relaxation_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: NoncommutingConfig,
) -> np.ndarray:
    hessian, equilibrium = phase_generator(rare_weight, config)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    displacement = np.asarray(state, dtype=float) - equilibrium
    projected = np.einsum("nji,nj->ni", eigenvectors, displacement)
    duration_array = np.asarray(duration, dtype=float)
    if duration_array.ndim == 0:
        duration_array = np.full(len(eigenvalues), float(duration_array))
    projected *= np.exp(-config.relaxation * duration_array[:, None] * eigenvalues)
    return equilibrium + np.einsum("nij,nj->ni", eigenvectors, projected)


def terminal_state(weights: np.ndarray, alpha0: float, config: NoncommutingConfig) -> np.ndarray:
    weights = normalized_policy(weights)
    state = np.zeros((len(weights), 2), dtype=float)
    state = matrix_relaxation_update(state, weights[:, 0, 1], alpha0, config)
    return matrix_relaxation_update(state, weights[:, 1, 1], 1.0 - alpha0, config)


def response_feature(weights: np.ndarray, alpha0: float, config: NoncommutingConfig) -> np.ndarray:
    state = terminal_state(weights, alpha0, config)
    center = np.asarray([config.evaluation_center, 0.0], dtype=float)
    return np.sum((state - center[None, :]) ** 2, axis=1)


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    alpha0: float,
    config: NoncommutingConfig,
) -> NoncommutingModel:
    feature = response_feature(weights, alpha0, config)
    head = scalar_hessian.fit_quadratic_head(feature, target, indices, config.l2)
    return NoncommutingModel(alpha0, config, head)
