# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-task gradient flow with the training optimizer's global gradient clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUMERICAL_FLOOR = 1e-12
BROAD_OPTIMUM = np.asarray([-0.5, 0.0], dtype=float)
RARE_OPTIMUM = np.asarray([0.5, 0.0], dtype=float)


@dataclass(frozen=True)
class ClippedTaskFlowConfig:
    """Frozen task geometry, optimizer flow rate, and clipping threshold."""

    curvature_ratio: float
    anisotropy: float
    angle_degrees: float
    relaxation: float
    evaluation_rare_weight: float
    clip_norm: float

    @property
    def key(self) -> str:
        clip = "inf" if np.isinf(self.clip_norm) else f"{self.clip_norm:g}"
        return (
            f"curvature={self.curvature_ratio:g},anisotropy={self.anisotropy:g},"
            f"angle={self.angle_degrees:g},relaxation={self.relaxation:g},"
            f"eval={self.evaluation_rare_weight:g},clip={clip}"
        )


def normalized_policy(weights: np.ndarray) -> np.ndarray:
    """Return nonnegative phase mixtures normalized over the two domains."""

    clipped = np.maximum(np.asarray(weights, dtype=float), 0.0)
    totals = clipped.sum(axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Every phase policy must have positive total weight")
    return clipped / totals


def task_hessians(config: ClippedTaskFlowConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return broad and rare quadratic-task Hessians."""

    broad = np.diag([1.0, config.anisotropy])
    angle = np.deg2rad(config.angle_degrees)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rare = config.curvature_ratio * rotation @ np.diag([1.0, config.anisotropy]) @ rotation.T
    return broad, rare


def task_gradient(state: np.ndarray, rare_weight: np.ndarray, config: ClippedTaskFlowConfig) -> np.ndarray:
    """Evaluate the current mixture-weighted task gradient."""

    broad, rare = task_hessians(config)
    broad_gradient = (state - BROAD_OPTIMUM[None, :]) @ broad
    rare_gradient = (state - RARE_OPTIMUM[None, :]) @ rare
    p = np.asarray(rare_weight, dtype=float)[:, None]
    return (1.0 - p) * broad_gradient + p * rare_gradient


def clipped_gradient(gradient: np.ndarray, clip_norm: float) -> np.ndarray:
    """Apply global gradient clipping independently to each policy trajectory."""

    if np.isinf(clip_norm):
        return gradient
    norm = np.linalg.norm(gradient, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norm, NUMERICAL_FLOOR))
    return gradient * scale


def _derivative(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: ClippedTaskFlowConfig,
) -> np.ndarray:
    gradient = task_gradient(state, rare_weight, config)
    return -config.relaxation * clipped_gradient(gradient, config.clip_norm)


def integrate_phase(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: ClippedTaskFlowConfig,
    *,
    steps_per_unit: int = 192,
) -> np.ndarray:
    """Integrate one constant-mixture phase with fourth-order Runge--Kutta."""

    steps = max(1, int(np.ceil(duration * steps_per_unit)))
    step = duration / steps
    current = np.asarray(state, dtype=float).copy()
    for _ in range(steps):
        k1 = _derivative(current, rare_weight, config)
        k2 = _derivative(current + 0.5 * step * k1, rare_weight, config)
        k3 = _derivative(current + 0.5 * step * k2, rare_weight, config)
        k4 = _derivative(current + step * k3, rare_weight, config)
        current += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return current


def terminal_state(
    weights: np.ndarray,
    optimizer_phase0_fraction: float,
    config: ClippedTaskFlowConfig,
    *,
    steps_per_unit: int = 192,
) -> np.ndarray:
    """Integrate the ordered two-phase policy in normalized optimizer time."""

    policy = normalized_policy(weights)
    state = np.zeros((len(policy), 2), dtype=float)
    state = integrate_phase(
        state,
        policy[:, 0, 1],
        optimizer_phase0_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return integrate_phase(
        state,
        policy[:, 1, 1],
        1.0 - optimizer_phase0_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def task_potential(state: np.ndarray, config: ClippedTaskFlowConfig) -> np.ndarray:
    """Evaluate the declared mixture of broad and rare task losses."""

    broad, rare = task_hessians(config)
    broad_error = state - BROAD_OPTIMUM[None, :]
    rare_error = state - RARE_OPTIMUM[None, :]
    broad_loss = 0.5 * np.einsum("ni,ij,nj->n", broad_error, broad, broad_error)
    rare_loss = 0.5 * np.einsum("ni,ij,nj->n", rare_error, rare, rare_error)
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad_loss + q * rare_loss


def response_feature(
    weights: np.ndarray,
    optimizer_phase0_fraction: float,
    config: ClippedTaskFlowConfig,
    *,
    steps_per_unit: int = 192,
) -> np.ndarray:
    """Return the terminal evaluation potential for each policy."""

    state = terminal_state(
        weights,
        optimizer_phase0_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return task_potential(state, config)
