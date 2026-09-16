# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite feature-slot acquisition and overwrite dynamics."""

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
    finite_capacity: bool
    acquisition_rate: float
    overwrite_rate: float
    rare_rate_ratio: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"clock={self.clock.value},finite={int(self.finite_capacity)},acq={self.acquisition_rate:g},"
            f"overwrite={self.overwrite_rate:g},rare={self.rare_rate_ratio:g},l2={self.l2:g}"
        )


@cache
def occupancy_transition(
    rare_weight_rounded: float,
    duration_rounded: float,
    acquisition_rate: float,
    overwrite_rate: float,
    rare_rate_ratio: float,
) -> np.ndarray:
    rare_weight = float(rare_weight_rounded)
    duration = float(duration_rounded)
    broad_acquisition = acquisition_rate * (1.0 - rare_weight)
    rare_acquisition = acquisition_rate * rare_rate_ratio * rare_weight
    broad_overwrite = overwrite_rate * (1.0 - rare_weight)
    rare_overwrite = overwrite_rate * rare_rate_ratio * rare_weight
    generator = np.asarray(
        [
            [-(broad_acquisition + rare_acquisition), 0.0, 0.0],
            [broad_acquisition, -rare_overwrite, broad_overwrite],
            [rare_acquisition, rare_overwrite, -broad_overwrite],
        ],
        dtype=float,
    )
    return expm(generator * duration)


def finite_unresolved(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> np.ndarray:
    states = np.zeros((len(rare_phase_weights), 3), dtype=float)
    states[:, 0] = 1.0
    durations = (phase0_duration, 1.0 - phase0_duration)
    for phase, duration in enumerate(durations):
        for row, rare_weight in enumerate(rare_phase_weights[:, phase]):
            transition = occupancy_transition(
                round(float(rare_weight), 12),
                round(float(duration), 12),
                float(config.acquisition_rate),
                float(config.overwrite_rate),
                float(config.rare_rate_ratio),
            )
            states[row] = transition @ states[row]
    unallocated, broad, rare = states.T
    if np.max(np.abs(states.sum(axis=1) - 1.0)) > 1e-10 or np.min(states) < -1e-10:
        raise RuntimeError("Feature-occupancy CTMC left the probability simplex")
    return np.column_stack([unallocated + rare, unallocated + broad])


def independent_unresolved(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> np.ndarray:
    aggregate_rare = phase0_duration * rare_phase_weights[:, 0] + (1.0 - phase0_duration) * rare_phase_weights[:, 1]
    broad_exposure = 1.0 - aggregate_rare
    rare_exposure = config.rare_rate_ratio * aggregate_rare
    return np.column_stack(
        [
            np.exp(-config.acquisition_rate * broad_exposure),
            np.exp(-config.acquisition_rate * rare_exposure),
        ]
    )


def unresolved_design(rare_phase_weights: np.ndarray, phase0_duration: float, config: Config) -> np.ndarray:
    if rare_phase_weights.ndim != 2 or rare_phase_weights.shape[1] != 2:
        raise ValueError(f"Expected [policy, phase] rare weights, got {rare_phase_weights.shape}")
    if config.finite_capacity:
        return finite_unresolved(rare_phase_weights, phase0_duration, config)
    return independent_unresolved(rare_phase_weights, phase0_duration, config)


def semigroup_error(rare_weight: float, config: Config) -> float:
    split = unresolved_design(np.asarray([[rare_weight, rare_weight]]), 0.37, config)[0]
    direct = unresolved_design(np.asarray([[rare_weight, rare_weight]]), 1.0, config)[0]
    return float(np.max(np.abs(split - direct)))
