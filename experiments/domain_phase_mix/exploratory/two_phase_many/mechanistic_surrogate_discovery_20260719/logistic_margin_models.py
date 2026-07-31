# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite-margin cross-entropy dynamics for broad and specialist tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.special import expit


class Clock(StrEnum):
    TOKEN = "token_time"
    OPTIMIZER = "optimizer_time"


@dataclass(frozen=True)
class Config:
    clock: Clock
    acquisition_rate: float
    weight_decay: float
    task_angle_degrees: float
    rare_rate_ratio: float
    l2: float
    integration_steps: int = 128

    @property
    def key(self) -> str:
        return (
            f"clock={self.clock.value},rate={self.acquisition_rate:g},decay={self.weight_decay:g},"
            f"angle={self.task_angle_degrees:g},rare={self.rare_rate_ratio:g},l2={self.l2:g}"
        )


def task_vectors(config: Config) -> np.ndarray:
    angle = np.deg2rad(config.task_angle_degrees)
    return np.asarray([[1.0, 0.0], [np.cos(angle), np.sin(angle)]], dtype=float)


def derivative(state: np.ndarray, rare_weight: np.ndarray, config: Config) -> np.ndarray:
    vectors = task_vectors(config)
    margins = state @ vectors.T
    gradient_scale = expit(-margins)
    broad = (1.0 - rare_weight)[:, None] * gradient_scale[:, 0, None] * vectors[0]
    rare = config.rare_rate_ratio * rare_weight[:, None] * gradient_scale[:, 1, None] * vectors[1]
    return config.acquisition_rate * (broad + rare - config.weight_decay * state)


def integrate_phase(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: Config,
) -> np.ndarray:
    steps = max(1, int(np.ceil(config.integration_steps * duration)))
    step = duration / steps
    result = state.copy()
    for _ in range(steps):
        k1 = derivative(result, rare_weight, config)
        k2 = derivative(result + 0.5 * step * k1, rare_weight, config)
        k3 = derivative(result + 0.5 * step * k2, rare_weight, config)
        k4 = derivative(result + step * k3, rare_weight, config)
        result += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result


def terminal_state(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> np.ndarray:
    if rare_phase_weights.ndim != 2 or rare_phase_weights.shape[1] != 2:
        raise ValueError(f"Expected [policy, phase] rare weights, got {rare_phase_weights.shape}")
    state = np.zeros((len(rare_phase_weights), 2), dtype=float)
    state = integrate_phase(state, rare_phase_weights[:, 0], phase0_duration, config)
    return integrate_phase(state, rare_phase_weights[:, 1], 1.0 - phase0_duration, config)


def logistic_loss_design(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> np.ndarray:
    state = terminal_state(rare_phase_weights, phase0_duration, config)
    margins = state @ task_vectors(config).T
    return np.logaddexp(0.0, -margins)


def semigroup_error(rare_weight: float, config: Config) -> float:
    state = np.zeros((1, 2), dtype=float)
    split = integrate_phase(state, np.asarray([rare_weight]), 0.37, config)
    split = integrate_phase(split, np.asarray([rare_weight]), 0.63, config)
    direct = integrate_phase(state, np.asarray([rare_weight]), 1.0, config)
    return float(np.max(np.abs(split - direct)))


def integration_error(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> float:
    fine = Config(
        clock=config.clock,
        acquisition_rate=config.acquisition_rate,
        weight_decay=config.weight_decay,
        task_angle_degrees=config.task_angle_degrees,
        rare_rate_ratio=config.rare_rate_ratio,
        l2=config.l2,
        integration_steps=2 * config.integration_steps,
    )
    coarse_state = terminal_state(rare_phase_weights, phase0_duration, config)
    fine_state = terminal_state(rare_phase_weights, phase0_duration, fine)
    return float(np.max(np.abs(coarse_state - fine_state)))
