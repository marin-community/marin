# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared/private competence dynamics for two-domain falsification.

The module contains two independently derived mechanisms.  The competence
cascade is an acquisition model with a prerequisite state.  The factorized
gradient flow instead derives its state transition from explicit broad and
specialist task potentials.  Both are autonomous for a phase-tied policy and
therefore ignore an artificial phase boundary exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class CascadeConfig:
    """Hyperparameters for the foundation-specialization cascade."""

    foundation_rate: float
    specialist_rate: float
    rare_foundation_efficiency: float
    prerequisite_power: float
    l2: float


@dataclass(frozen=True)
class FactorizedFlowConfig:
    """Hyperparameters for factorized-capability gradient flow."""

    speed: float
    broad_specialist_decay: float
    rare_foundation_efficiency: float
    l2: float


def _validate_two_domain_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Mixture weights must sum to one within each phase")
    return weights


def _cascade_phase(
    foundation: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: CascadeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    foundation_drive = (1.0 - rare_weight) + config.rare_foundation_efficiency * rare_weight
    rate = config.foundation_rate * foundation_drive
    survival = np.exp(-rate * duration)
    end_foundation = 1.0 - (1.0 - foundation) * survival
    integral_foundation = duration - np.divide(
        (1.0 - foundation) * (1.0 - survival),
        rate,
        out=(1.0 - foundation) * np.full_like(rate, duration),
        where=rate > NUMERICAL_FLOOR,
    )

    if config.prerequisite_power == 0.0:
        gated_time = np.full_like(integral_foundation, duration)
    elif config.prerequisite_power == 1.0:
        gated_time = integral_foundation
    else:
        # Simpson quadrature is deterministic and exact for the two special
        # cases above.  It is only needed for the nonlinear prerequisite.
        fractions = np.linspace(0.0, 1.0, 33)
        state = 1.0 - (1.0 - foundation[:, None]) * np.exp(-rate[:, None] * duration * fractions[None, :])
        gated_time = duration * np.trapezoid(state**config.prerequisite_power, fractions, axis=1)
    specialist_hazard = config.specialist_rate * rare_weight * gated_time
    end_specialist = 1.0 - (1.0 - specialist) * np.exp(-specialist_hazard)
    return end_foundation, end_specialist


def cascade_terminal_state(
    weights: np.ndarray,
    phase0_fraction: float,
    config: CascadeConfig,
) -> np.ndarray:
    """Return terminal foundation and specialist competences in [0, 1]."""

    weights = _validate_two_domain_weights(weights)
    foundation = np.zeros(len(weights), dtype=float)
    specialist = np.zeros(len(weights), dtype=float)
    for phase, duration in ((0, phase0_fraction), (1, 1.0 - phase0_fraction)):
        foundation, specialist = _cascade_phase(
            foundation,
            specialist,
            weights[:, phase, 1],
            duration,
            config,
        )
    return np.column_stack([foundation, specialist])


def cascade_design(
    weights: np.ndarray,
    phase0_fraction: float,
    config: CascadeConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return nonnegative terminal error masses for a linear BPB response."""

    state = cascade_terminal_state(weights, phase0_fraction, config)
    return 1.0 - state, ("foundation_error", "specialist_error")


def literal_replay(weights: np.ndarray, phase0_epochs: np.ndarray, phase1_epochs: np.ndarray) -> np.ndarray:
    """Return exact repeated finite-subset traversals after the first epoch."""

    weights = _validate_two_domain_weights(weights)
    exposure = weights[:, 0] * phase0_epochs[None, :] + weights[:, 1] * phase1_epochs[None, :]
    return np.maximum(exposure - 1.0, 0.0)


def cascade_replay_design(
    weights: np.ndarray,
    phase0_fraction: float,
    phase0_epochs: np.ndarray,
    phase1_epochs: np.ndarray,
    config: CascadeConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Combine terminal unresolved error with exact physical replay harm."""

    unresolved, unresolved_names = cascade_design(weights, phase0_fraction, config)
    replay = literal_replay(weights, phase0_epochs, phase1_epochs)
    return np.column_stack([unresolved, replay]), (*unresolved_names, "broad_replay", "specialist_replay")


def _factorized_gradient(
    foundation: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    config: FactorizedFlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    product_error = foundation * specialist - 1.0
    broad_weight = 1.0 - rare_weight
    foundation_gradient = broad_weight * (foundation - 1.0) + rare_weight * (
        product_error * specialist + config.rare_foundation_efficiency * (foundation - 1.0)
    )
    specialist_gradient = (
        broad_weight * config.broad_specialist_decay * specialist + rare_weight * product_error * foundation
    )
    return foundation_gradient, specialist_gradient


def _factorized_phase(
    foundation: np.ndarray,
    specialist: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: FactorizedFlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 0.0:
        return foundation, specialist
    steps = max(16, int(np.ceil(128 * duration)))
    step = duration / steps
    for _ in range(steps):
        k1_g, k1_s = _factorized_gradient(foundation, specialist, rare_weight, config)
        k2_g, k2_s = _factorized_gradient(
            foundation - 0.5 * step * config.speed * k1_g,
            specialist - 0.5 * step * config.speed * k1_s,
            rare_weight,
            config,
        )
        k3_g, k3_s = _factorized_gradient(
            foundation - 0.5 * step * config.speed * k2_g,
            specialist - 0.5 * step * config.speed * k2_s,
            rare_weight,
            config,
        )
        k4_g, k4_s = _factorized_gradient(
            foundation - step * config.speed * k3_g,
            specialist - step * config.speed * k3_s,
            rare_weight,
            config,
        )
        foundation -= step * config.speed * (k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g) / 6.0
        specialist -= step * config.speed * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s) / 6.0
    return foundation, specialist


def factorized_terminal_state(
    weights: np.ndarray,
    phase0_fraction: float,
    config: FactorizedFlowConfig,
) -> np.ndarray:
    """Integrate the shared/private task gradient flow through both phases."""

    weights = _validate_two_domain_weights(weights)
    foundation = np.zeros(len(weights), dtype=float)
    specialist = np.zeros(len(weights), dtype=float)
    for phase, duration in ((0, phase0_fraction), (1, 1.0 - phase0_fraction)):
        foundation, specialist = _factorized_phase(
            foundation,
            specialist,
            weights[:, phase, 1],
            duration,
            config,
        )
    tied = np.max(np.abs(weights[:, 0] - weights[:, 1]), axis=1) < 1e-12
    if np.any(tied):
        tied_foundation, tied_specialist = _factorized_phase(
            np.zeros(int(tied.sum()), dtype=float),
            np.zeros(int(tied.sum()), dtype=float),
            weights[tied, 0, 1],
            1.0,
            config,
        )
        foundation[tied] = tied_foundation
        specialist[tied] = tied_specialist
    return np.column_stack([foundation, specialist])


def factorized_target_potential(state: np.ndarray, config: FactorizedFlowConfig) -> np.ndarray:
    """Evaluate the specialist task potential used by the transition law."""

    foundation = state[:, 0]
    specialist = state[:, 1]
    return 0.5 * (foundation * specialist - 1.0) ** 2 + 0.5 * config.rare_foundation_efficiency * (foundation - 1.0) ** 2


def factorized_design(
    weights: np.ndarray,
    phase0_fraction: float,
    config: FactorizedFlowConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the terminal target potential for a nonnegative BPB response."""

    state = factorized_terminal_state(weights, phase0_fraction, config)
    return factorized_target_potential(state, config)[:, None], ("terminal_target_potential",)


def tied_policy_error(
    state_function: object,
    config: CascadeConfig | FactorizedFlowConfig,
) -> float:
    """Measure invariance to moving an artificial boundary under tied input."""

    grid = np.linspace(0.0, 1.0, 31)
    weights = np.stack(
        [
            np.column_stack([1.0 - grid, grid]),
            np.column_stack([1.0 - grid, grid]),
        ],
        axis=1,
    )
    state_a = state_function(weights, 0.2, config)
    state_b = state_function(weights, 0.8, config)
    return float(np.max(np.abs(state_a - state_b)))
