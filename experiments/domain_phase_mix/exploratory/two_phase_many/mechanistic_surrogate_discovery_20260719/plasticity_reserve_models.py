# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Foundation and specialist learning with a replenishable plasticity reserve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    shared_private_models,
)


@dataclass(frozen=True)
class PlasticityReserveConfig:
    foundation_rate: float
    reserve_recovery_rate: float
    specialist_rate: float
    rare_foundation_efficiency: float
    reserve_depletion_rate: float
    l2: float


def _phase_update(
    foundation: np.ndarray,
    reserve: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: PlasticityReserveConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration <= 0.0:
        return foundation, reserve, specialist

    foundation_drive = (1.0 - rare_weight) + config.rare_foundation_efficiency * rare_weight
    foundation = 1.0 - (1.0 - foundation) * np.exp(-config.foundation_rate * foundation_drive * duration)

    recovery = config.reserve_recovery_rate * (1.0 - rare_weight)
    depletion = config.reserve_depletion_rate * rare_weight
    rate = recovery + depletion
    equilibrium = np.divide(recovery, rate, out=reserve.copy(), where=rate > 1e-12)
    decay = np.exp(-rate * duration)
    transient_integral = np.divide(
        (reserve - equilibrium) * (1.0 - decay),
        rate,
        out=np.zeros_like(rate),
        where=rate > 1e-12,
    )
    integrated_reserve = np.where(
        rate > 1e-12,
        equilibrium * duration + transient_integral,
        reserve * duration,
    )
    reserve = equilibrium + (reserve - equilibrium) * decay
    specialist_hazard = config.specialist_rate * rare_weight * np.maximum(integrated_reserve, 0.0)
    specialist = 1.0 - (1.0 - specialist) * np.exp(-specialist_hazard)
    return foundation, reserve, specialist


def terminal_state(
    weights: np.ndarray,
    phase0_fraction: float,
    config: PlasticityReserveConfig,
) -> np.ndarray:
    """Return foundation competence, reserve, and specialist competence."""

    weights = shared_private_models._validate_two_domain_weights(weights)
    foundation = np.zeros(len(weights), dtype=float)
    reserve = np.zeros(len(weights), dtype=float)
    specialist = np.zeros(len(weights), dtype=float)
    for phase, duration in ((0, phase0_fraction), (1, 1.0 - phase0_fraction)):
        foundation, reserve, specialist = _phase_update(
            foundation,
            reserve,
            specialist,
            weights[:, phase, 1],
            duration,
            config,
        )
    return np.column_stack([foundation, reserve, specialist])


def design(
    weights: np.ndarray,
    phase0_fraction: float,
    phase0_epochs: np.ndarray,
    phase1_epochs: np.ndarray,
    config: PlasticityReserveConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    state = terminal_state(weights, phase0_fraction, config)
    replay = shared_private_models.literal_replay(weights, phase0_epochs, phase1_epochs)
    return np.column_stack([1.0 - state[:, 0], 1.0 - state[:, 2], replay]), (
        "foundation_error",
        "specialist_error",
        "broad_replay",
        "specialist_replay",
    )


def tied_policy_error(config: PlasticityReserveConfig) -> float:
    grid = np.linspace(0.0, 1.0, 31)
    weights = np.stack(
        [np.column_stack([1.0 - grid, grid]), np.column_stack([1.0 - grid, grid])],
        axis=1,
    )
    state_short_early = terminal_state(weights, 0.2, config)
    state_long_early = terminal_state(weights, 0.8, config)
    return float(np.max(np.abs(state_short_early - state_long_early)))
