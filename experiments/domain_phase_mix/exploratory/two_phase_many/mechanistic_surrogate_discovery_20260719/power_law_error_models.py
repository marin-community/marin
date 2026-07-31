# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Power-law error kinetics for foundation and specialist capabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    shared_private_models,
)

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class GatedPowerConfig:
    foundation_rate: float
    specialist_rate: float
    rare_foundation_efficiency: float
    prerequisite_power: float
    learning_curve_power: float
    l2: float


@dataclass(frozen=True)
class ForgettingPowerConfig:
    foundation_rate: float
    specialist_rate: float
    rare_foundation_efficiency: float
    forgetting_rate: float
    learning_curve_power: float
    l2: float


def power_error_update(error: np.ndarray, hazard: np.ndarray, power: float) -> np.ndarray:
    """Integrate de/dt=-hazard_rate*e^(1+power) over a phase."""

    error = np.maximum(np.asarray(error, dtype=float), NUMERICAL_FLOOR)
    hazard = np.maximum(np.asarray(hazard, dtype=float), 0.0)
    if power == 0.0:
        return error * np.exp(-hazard)
    return (error ** (-power) + power * hazard) ** (-1.0 / power)


def _foundation_trajectory(
    initial_error: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    foundation_rate: float,
    rare_foundation_efficiency: float,
    learning_curve_power: float,
    fractions: np.ndarray,
) -> np.ndarray:
    drive = (1.0 - rare_weight) + rare_foundation_efficiency * rare_weight
    hazard = foundation_rate * drive[:, None] * duration * fractions[None, :]
    return power_error_update(initial_error[:, None], hazard, learning_curve_power)


def gated_power_terminal_errors(
    weights: np.ndarray,
    phase0_fraction: float,
    config: GatedPowerConfig,
) -> np.ndarray:
    """Integrate power-law foundation and prerequisite-gated specialist errors."""

    weights = shared_private_models._validate_two_domain_weights(weights)
    foundation_error = np.ones(len(weights), dtype=float)
    specialist_error = np.ones(len(weights), dtype=float)
    fractions = np.linspace(0.0, 1.0, 65)
    for phase, duration in ((0, phase0_fraction), (1, 1.0 - phase0_fraction)):
        rare_weight = weights[:, phase, 1]
        trajectory = _foundation_trajectory(
            foundation_error,
            rare_weight,
            duration,
            config.foundation_rate,
            config.rare_foundation_efficiency,
            config.learning_curve_power,
            fractions,
        )
        competence = 1.0 - trajectory
        if config.prerequisite_power == 0.0:
            gated_time = np.full(len(weights), duration, dtype=float)
        else:
            gated_time = duration * np.trapezoid(competence**config.prerequisite_power, fractions, axis=1)
        foundation_error = trajectory[:, -1]
        specialist_error = power_error_update(
            specialist_error,
            config.specialist_rate * rare_weight * gated_time,
            config.learning_curve_power,
        )
    return np.column_stack([foundation_error, specialist_error])


def gated_power_design(
    weights: np.ndarray,
    phase0_fraction: float,
    phase0_epochs: np.ndarray,
    phase1_epochs: np.ndarray,
    config: GatedPowerConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    errors = gated_power_terminal_errors(weights, phase0_fraction, config)
    replay = shared_private_models.literal_replay(weights, phase0_epochs, phase1_epochs)
    return np.column_stack([errors, replay]), (
        "foundation_error",
        "specialist_error",
        "broad_replay",
        "specialist_replay",
    )


def _forgetting_derivative(
    foundation_error: np.ndarray,
    specialist_error: np.ndarray,
    rare_weight: np.ndarray,
    config: ForgettingPowerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    drive = (1.0 - rare_weight) + config.rare_foundation_efficiency * rare_weight
    foundation = -config.foundation_rate * drive * foundation_error ** (1.0 + config.learning_curve_power)
    specialist = -config.specialist_rate * rare_weight * specialist_error ** (
        1.0 + config.learning_curve_power
    ) + config.forgetting_rate * (1.0 - rare_weight) * (1.0 - specialist_error)
    return foundation, specialist


def _forgetting_phase(
    foundation_error: np.ndarray,
    specialist_error: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: ForgettingPowerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 0.0:
        return foundation_error, specialist_error
    steps = max(16, int(np.ceil(192 * duration)))
    step = duration / steps
    for _ in range(steps):
        k1_g, k1_s = _forgetting_derivative(foundation_error, specialist_error, rare_weight, config)
        k2_g, k2_s = _forgetting_derivative(
            np.clip(foundation_error + 0.5 * step * k1_g, 0.0, 1.0),
            np.clip(specialist_error + 0.5 * step * k1_s, 0.0, 1.0),
            rare_weight,
            config,
        )
        k3_g, k3_s = _forgetting_derivative(
            np.clip(foundation_error + 0.5 * step * k2_g, 0.0, 1.0),
            np.clip(specialist_error + 0.5 * step * k2_s, 0.0, 1.0),
            rare_weight,
            config,
        )
        k4_g, k4_s = _forgetting_derivative(
            np.clip(foundation_error + step * k3_g, 0.0, 1.0),
            np.clip(specialist_error + step * k3_s, 0.0, 1.0),
            rare_weight,
            config,
        )
        foundation_error = np.clip(
            foundation_error + step * (k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g) / 6.0,
            0.0,
            1.0,
        )
        specialist_error = np.clip(
            specialist_error + step * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s) / 6.0,
            0.0,
            1.0,
        )
    return foundation_error, specialist_error


def forgetting_power_terminal_errors(
    weights: np.ndarray,
    phase0_fraction: float,
    config: ForgettingPowerConfig,
) -> np.ndarray:
    """Integrate power-law acquisition with broad-induced specialist forgetting."""

    weights = shared_private_models._validate_two_domain_weights(weights)
    foundation_error = np.ones(len(weights), dtype=float)
    specialist_error = np.ones(len(weights), dtype=float)
    for phase, duration in ((0, phase0_fraction), (1, 1.0 - phase0_fraction)):
        foundation_error, specialist_error = _forgetting_phase(
            foundation_error,
            specialist_error,
            weights[:, phase, 1],
            duration,
            config,
        )
    tied = np.max(np.abs(weights[:, 0] - weights[:, 1]), axis=1) < 1e-12
    if np.any(tied):
        tied_foundation, tied_specialist = _forgetting_phase(
            np.ones(int(tied.sum()), dtype=float),
            np.ones(int(tied.sum()), dtype=float),
            weights[tied, 0, 1],
            1.0,
            config,
        )
        foundation_error[tied] = tied_foundation
        specialist_error[tied] = tied_specialist
    return np.column_stack([foundation_error, specialist_error])


def forgetting_power_design(
    weights: np.ndarray,
    phase0_fraction: float,
    phase0_epochs: np.ndarray,
    phase1_epochs: np.ndarray,
    config: ForgettingPowerConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    errors = forgetting_power_terminal_errors(weights, phase0_fraction, config)
    replay = shared_private_models.literal_replay(weights, phase0_epochs, phase1_epochs)
    return np.column_stack([errors, replay]), (
        "foundation_error",
        "specialist_error",
        "broad_replay",
        "specialist_replay",
    )


def tied_error(
    state_function: object,
    config: GatedPowerConfig | ForgettingPowerConfig,
) -> float:
    grid = np.linspace(0.0, 1.0, 31)
    weights = np.stack(
        [np.column_stack([1.0 - grid, grid]), np.column_stack([1.0 - grid, grid])],
        axis=1,
    )
    state_a = state_function(weights, 0.2, config)
    state_b = state_function(weights, 0.8, config)
    return float(np.max(np.abs(state_a - state_b)))
