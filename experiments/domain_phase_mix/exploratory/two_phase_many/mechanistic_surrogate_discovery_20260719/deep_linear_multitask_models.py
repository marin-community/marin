# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deep-linear multitask gradient flow with a shared feature and two heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 256
INITIAL_STATE = 0.1


@dataclass(frozen=True)
class DeepLinearConfig:
    """Frozen optimization dynamics and evaluation weighting."""

    shared_rate: float
    head_rate: float
    weight_decay: float
    evaluation_rare_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"shared={self.shared_rate:g},head={self.head_rate:g},decay={self.weight_decay:g},"
            f"eval={self.evaluation_rare_weight:g},l2={self.l2:g}"
        )


def validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Each phase must be a nonnegative simplex mixture")
    return np.maximum(weights, 0.0)


def derivatives(
    shared: np.ndarray,
    broad_head: np.ndarray,
    rare_head: np.ndarray,
    rare_weight: np.ndarray,
    config: DeepLinearConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    broad_weight = 1.0 - rare_weight
    broad_error = shared * broad_head - 1.0
    rare_error = shared * rare_head - 1.0
    shared_gradient = broad_weight * broad_error * broad_head + rare_weight * rare_error * rare_head
    broad_head_gradient = broad_weight * broad_error * shared
    rare_head_gradient = rare_weight * rare_error * shared
    shared_derivative = -config.shared_rate * (shared_gradient + config.weight_decay * shared)
    broad_head_derivative = -config.head_rate * (broad_head_gradient + config.weight_decay * broad_head)
    rare_head_derivative = -config.head_rate * (rare_head_gradient + config.weight_decay * rare_head)
    return shared_derivative, broad_head_derivative, rare_head_derivative


def phase_update(
    shared: np.ndarray,
    broad_head: np.ndarray,
    rare_head: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: DeepLinearConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = max(1, int(np.ceil(INTEGRATION_STEPS_PER_UNIT * duration)))
    step_size = duration / steps
    shared = np.asarray(shared, dtype=float).copy()
    broad_head = np.asarray(broad_head, dtype=float).copy()
    rare_head = np.asarray(rare_head, dtype=float).copy()
    for _ in range(steps):
        k1 = derivatives(shared, broad_head, rare_head, rare_weight, config)
        k2 = derivatives(
            shared + 0.5 * step_size * k1[0],
            broad_head + 0.5 * step_size * k1[1],
            rare_head + 0.5 * step_size * k1[2],
            rare_weight,
            config,
        )
        k3 = derivatives(
            shared + 0.5 * step_size * k2[0],
            broad_head + 0.5 * step_size * k2[1],
            rare_head + 0.5 * step_size * k2[2],
            rare_weight,
            config,
        )
        k4 = derivatives(
            shared + step_size * k3[0],
            broad_head + step_size * k3[1],
            rare_head + step_size * k3[2],
            rare_weight,
            config,
        )
        shared += step_size * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        broad_head += step_size * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        rare_head += step_size * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        if not np.isfinite(shared).all() or not np.isfinite(broad_head).all() or not np.isfinite(rare_head).all():
            raise FloatingPointError(f"Non-finite deep-linear state for {config.key}")
    return shared, broad_head, rare_head


def terminal_state(weights: np.ndarray, alpha0: float, config: DeepLinearConfig) -> np.ndarray:
    weights = validate_weights(weights)
    shared = np.full(len(weights), INITIAL_STATE, dtype=float)
    broad_head = np.full(len(weights), INITIAL_STATE, dtype=float)
    rare_head = np.full(len(weights), INITIAL_STATE, dtype=float)
    shared, broad_head, rare_head = phase_update(
        shared,
        broad_head,
        rare_head,
        weights[:, 0, 1],
        alpha0,
        config,
    )
    shared, broad_head, rare_head = phase_update(
        shared,
        broad_head,
        rare_head,
        weights[:, 1, 1],
        1.0 - alpha0,
        config,
    )
    return np.column_stack([shared, broad_head, rare_head])


def task_losses(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shared = np.asarray(state, dtype=float)[:, 0]
    broad_head = np.asarray(state, dtype=float)[:, 1]
    rare_head = np.asarray(state, dtype=float)[:, 2]
    return 0.5 * (shared * broad_head - 1.0) ** 2, 0.5 * (shared * rare_head - 1.0) ** 2


def response_feature(weights: np.ndarray, alpha0: float, config: DeepLinearConfig) -> np.ndarray:
    broad, rare = task_losses(terminal_state(weights, alpha0, config))
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad + q * rare
