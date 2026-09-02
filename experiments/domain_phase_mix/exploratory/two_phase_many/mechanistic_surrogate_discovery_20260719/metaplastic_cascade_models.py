# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directed metaplastic consolidation dynamics for two-phase task mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import numpy as np
from scipy.linalg import expm


class Clock(StrEnum):
    TOKEN = "token_time"
    OPTIMIZER = "optimizer_time"


@dataclass(frozen=True)
class Config:
    clock: Clock
    levels: int
    acquisition_rate: float
    forgetting_ratio: float
    consolidation_rate: float
    depth_ratio: float
    durable_weight: float
    rare_rate_ratio: float
    l2: float

    def __post_init__(self) -> None:
        if self.levels not in (1, 3):
            raise ValueError(f"MCCF supports exactly 1 or 3 levels, got {self.levels}")
        if (
            min(
                self.acquisition_rate,
                self.forgetting_ratio,
                self.consolidation_rate,
                self.depth_ratio,
                self.rare_rate_ratio,
            )
            <= 0.0
        ):
            raise ValueError("All MCCF rates and ratios must be positive")
        if not 0.0 <= self.durable_weight <= 1.0:
            raise ValueError("durable_weight must be in [0, 1]")

    @property
    def key(self) -> str:
        return (
            f"clock={self.clock.value},K={self.levels},acq={self.acquisition_rate:g},"
            f"forget={self.forgetting_ratio:g},consolidate={self.consolidation_rate:g},"
            f"depth={self.depth_ratio:g},durable={self.durable_weight:g},"
            f"rare={self.rare_rate_ratio:g},l2={self.l2:g}"
        )


@cache
def affine_transition(
    levels: int,
    input_mass_rounded: float,
    duration_rounded: float,
    acquisition_rate: float,
    forgetting_ratio: float,
    consolidation_rate: float,
    depth_ratio: float,
) -> np.ndarray:
    """Return the exact augmented affine transition for one constant-input phase."""

    input_mass = float(input_mass_rounded)
    duration = float(duration_rounded)
    generator = np.zeros((levels + 1, levels + 1), dtype=float)
    shallow_rate = acquisition_rate * (input_mass + forgetting_ratio * (1.0 - input_mass))
    generator[0, 0] = -shallow_rate
    generator[0, -1] = acquisition_rate * input_mass
    for level in range(1, levels):
        rate = consolidation_rate / depth_ratio ** (level - 1)
        generator[level, level - 1] = rate
        generator[level, level] = -rate
    return expm(generator * duration)


def update_state(
    state: np.ndarray,
    input_mass: float,
    duration: float,
    acquisition_rate: float,
    forgetting_ratio: float,
    consolidation_rate: float,
    depth_ratio: float,
) -> np.ndarray:
    transition = affine_transition(
        len(state),
        round(float(input_mass), 12),
        round(float(duration), 12),
        float(acquisition_rate),
        float(forgetting_ratio),
        float(consolidation_rate),
        float(depth_ratio),
    )
    augmented = np.concatenate([state, np.ones(1, dtype=float)])
    return (transition @ augmented)[:-1]


def terminal_competence(
    rare_phase_weights: np.ndarray,
    phase0_duration: float,
    config: Config,
) -> np.ndarray:
    """Return broad and rare terminal competence for each policy."""

    if rare_phase_weights.ndim != 2 or rare_phase_weights.shape[1] != 2:
        raise ValueError(f"Expected [policy, phase] rare weights, got {rare_phase_weights.shape}")
    result = np.zeros((len(rare_phase_weights), 2), dtype=float)
    durations = (phase0_duration, 1.0 - phase0_duration)
    for row, (phase0_rare, phase1_rare) in enumerate(rare_phase_weights):
        phase_inputs = ((1.0 - phase0_rare, phase0_rare), (1.0 - phase1_rare, phase1_rare))
        states = [np.zeros(config.levels, dtype=float), np.zeros(config.levels, dtype=float)]
        for inputs, duration in zip(phase_inputs, durations, strict=True):
            for task, input_mass in enumerate(inputs):
                task_rate = config.acquisition_rate * (config.rare_rate_ratio if task == 1 else 1.0)
                states[task] = update_state(
                    states[task],
                    input_mass,
                    duration,
                    task_rate,
                    config.forgetting_ratio,
                    config.consolidation_rate,
                    config.depth_ratio,
                )
        for task, state in enumerate(states):
            if config.levels == 1:
                result[row, task] = state[0]
            else:
                result[row, task] = (1.0 - config.durable_weight) * state[0] + config.durable_weight * state[-1]
    return result


def unresolved_design(
    rare_phase_weights: np.ndarray,
    phase0_duration: float,
    config: Config,
) -> np.ndarray:
    competence = terminal_competence(rare_phase_weights, phase0_duration, config)
    return 1.0 - competence


def semigroup_error(rare_weight: float, config: Config) -> float:
    """Check that a tied policy is invariant to insertion of the phase boundary."""

    phase_duration = 0.37
    split = terminal_competence(np.asarray([[rare_weight, rare_weight]]), phase_duration, config)[0]
    states = [np.zeros(config.levels, dtype=float), np.zeros(config.levels, dtype=float)]
    for task, input_mass in enumerate((1.0 - rare_weight, rare_weight)):
        task_rate = config.acquisition_rate * (config.rare_rate_ratio if task == 1 else 1.0)
        states[task] = update_state(
            states[task],
            input_mass,
            1.0,
            task_rate,
            config.forgetting_ratio,
            config.consolidation_rate,
            config.depth_ratio,
        )
    uninterrupted = np.asarray(
        [
            state[0]
            if config.levels == 1
            else (1.0 - config.durable_weight) * state[0] + config.durable_weight * state[-1]
            for state in states
        ]
    )
    return float(np.max(np.abs(split - uninterrupted)))
