# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact reduced gradient flow under mixture-dependent quadratic curvature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class HessianConfig:
    """Nonlinear transition and response geometry."""

    curvature_ratio: float
    relaxation: float
    initial_state: float
    evaluation_optimum: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"curvature={self.curvature_ratio:g},relaxation={self.relaxation:g},"
            f"initial={self.initial_state:g},eval={self.evaluation_optimum:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class QuadraticHead:
    """A nonnegative one-feature ridge response with an intercept."""

    feature_mean: float
    feature_scale: float
    target_mean: float
    coefficient: float

    def predict(self, feature: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(feature, dtype=float) - self.feature_mean) / self.feature_scale
        return np.asarray(self.target_mean + self.coefficient * standardized, dtype=float)

    @property
    def natural_curvature(self) -> float:
        return self.coefficient / self.feature_scale


@dataclass(frozen=True)
class HessianModel:
    """Fitted gradient-flow state and convex evaluation response."""

    alpha0: float
    config: HessianConfig
    head: QuadraticHead

    def state(self, weights: np.ndarray) -> np.ndarray:
        return terminal_state(weights, self.alpha0, self.config)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        state = self.state(weights)
        feature = (state - self.config.evaluation_optimum) ** 2
        return self.head.predict(feature)


def normalized_policy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"HWER requires [n, 2, 2] policies, got {weights.shape}")
    if np.any(weights < -1e-10):
        raise ValueError("Negative mixture weight")
    clipped = np.maximum(weights, 0.0)
    totals = clipped.sum(axis=2, keepdims=True)
    if np.any(totals <= NUMERICAL_FLOOR):
        raise ValueError("Empty phase mixture")
    return clipped / totals


def equilibrium_and_hessian(rare_weight: np.ndarray, curvature_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    hessian = (1.0 - rare_weight) + curvature_ratio * rare_weight
    equilibrium = curvature_ratio * rare_weight / np.maximum(hessian, NUMERICAL_FLOOR)
    return equilibrium, hessian


def relaxation_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    curvature_ratio: float,
    relaxation: float,
) -> np.ndarray:
    equilibrium, hessian = equilibrium_and_hessian(rare_weight, curvature_ratio)
    return equilibrium + (np.asarray(state, dtype=float) - equilibrium) * np.exp(-relaxation * hessian * duration)


def terminal_state(weights: np.ndarray, alpha0: float, config: HessianConfig) -> np.ndarray:
    weights = normalized_policy(weights)
    state = np.full(len(weights), config.initial_state, dtype=float)
    state = relaxation_update(
        state,
        weights[:, 0, 1],
        alpha0,
        config.curvature_ratio,
        config.relaxation,
    )
    return relaxation_update(
        state,
        weights[:, 1, 1],
        1.0 - alpha0,
        config.curvature_ratio,
        config.relaxation,
    )


def response_feature(weights: np.ndarray, alpha0: float, config: HessianConfig) -> np.ndarray:
    state = terminal_state(weights, alpha0, config)
    return (state - config.evaluation_optimum) ** 2


def fit_quadratic_head(feature: np.ndarray, target: np.ndarray, indices: np.ndarray, l2: float) -> QuadraticHead:
    feature = np.asarray(feature, dtype=float)[indices]
    target = np.asarray(target, dtype=float)[indices]
    feature_mean = float(np.mean(feature))
    feature_scale = max(float(np.sqrt(np.mean((feature - feature_mean) ** 2))), 1e-8)
    standardized = (feature - feature_mean) / feature_scale
    target_mean = float(np.mean(target))
    numerator = float(np.sum(standardized * (target - target_mean)))
    denominator = float(np.sum(standardized**2) + l2)
    coefficient = max(numerator / max(denominator, NUMERICAL_FLOOR), 0.0)
    return QuadraticHead(feature_mean, feature_scale, target_mean, coefficient)


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    alpha0: float,
    config: HessianConfig,
) -> HessianModel:
    feature = response_feature(weights, alpha0, config)
    head = fit_quadratic_head(feature, target, indices, config.l2)
    return HessianModel(alpha0, config, head)
