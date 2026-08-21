# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Power-law task-loss dynamics constrained by a gradient Gram matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Clock(StrEnum):
    TOKEN = "token_time"
    OPTIMIZER = "optimizer_time"


@dataclass(frozen=True)
class Config:
    clock: Clock
    acquisition_rate: float
    decay_power: float
    gradient_correlation: float
    rare_rate_ratio: float
    l2: float
    integration_steps: int = 128

    @property
    def key(self) -> str:
        return (
            f"clock={self.clock.value},rate={self.acquisition_rate:g},power={self.decay_power:g},"
            f"corr={self.gradient_correlation:g},rare={self.rare_rate_ratio:g},l2={self.l2:g}"
        )


def derivative(log_competence: np.ndarray, rare_weight: np.ndarray, config: Config) -> np.ndarray:
    """Return d(-log excess loss)/d time from the two-task gradient Gram."""
    unresolved = np.exp(-log_competence)
    broad_unresolved = unresolved[:, 0]
    rare_unresolved = unresolved[:, 1]
    broad_rate = config.acquisition_rate
    rare_rate = config.acquisition_rate * config.rare_rate_ratio
    power = config.decay_power
    common = config.gradient_correlation * np.sqrt(broad_rate * rare_rate)
    broad_self = (1.0 - rare_weight) * broad_rate * broad_unresolved**power
    rare_self = rare_weight * rare_rate * rare_unresolved**power
    broad_cross = (
        rare_weight * common * broad_unresolved ** ((power - 1.0) / 2.0) * rare_unresolved ** ((power + 1.0) / 2.0)
    )
    rare_cross = (
        (1.0 - rare_weight)
        * common
        * rare_unresolved ** ((power - 1.0) / 2.0)
        * broad_unresolved ** ((power + 1.0) / 2.0)
    )
    return np.column_stack([broad_self + broad_cross, rare_self + rare_cross])


def integrate_phase(
    log_competence: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: Config,
) -> np.ndarray:
    steps = max(1, int(np.ceil(config.integration_steps * duration)))
    step = duration / steps
    result = log_competence.copy()
    for _ in range(steps):
        k1 = derivative(result, rare_weight, config)
        k2 = derivative(result + 0.5 * step * k1, rare_weight, config)
        k3 = derivative(result + 0.5 * step * k2, rare_weight, config)
        k4 = derivative(result + step * k3, rare_weight, config)
        result += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result


def terminal_unresolved(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> np.ndarray:
    if rare_phase_weights.ndim != 2 or rare_phase_weights.shape[1] != 2:
        raise ValueError(f"Expected [policy, phase] rare weights, got {rare_phase_weights.shape}")
    state = np.zeros((len(rare_phase_weights), 2), dtype=float)
    state = integrate_phase(state, rare_phase_weights[:, 0], phase0_duration, config)
    state = integrate_phase(state, rare_phase_weights[:, 1], 1.0 - phase0_duration, config)
    return np.exp(-state)


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
        decay_power=config.decay_power,
        gradient_correlation=config.gradient_correlation,
        rare_rate_ratio=config.rare_rate_ratio,
        l2=config.l2,
        integration_steps=2 * config.integration_steps,
    )
    return float(
        np.max(
            np.abs(
                terminal_unresolved(rare_phase_weights, phase0_duration, config)
                - terminal_unresolved(rare_phase_weights, phase0_duration, fine)
            )
        )
    )
