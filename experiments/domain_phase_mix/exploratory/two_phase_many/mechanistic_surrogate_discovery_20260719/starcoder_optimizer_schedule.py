# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact peak-normalized optimizer schedules for the two StarCoder surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class StarCoderSchedule(StrEnum):
    COSINE_50_50 = "starcoder_cosine_50_50"
    WSD_80_20 = "starcoder_wsd_80_20"


@dataclass(frozen=True)
class OptimizerScheduleSpec:
    """One continuous Levanter optimizer cycle expressed in training steps."""

    name: StarCoderSchedule
    total_steps: int
    phase_boundary_step: int
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    provenance: str

    def __post_init__(self) -> None:
        if self.warmup_steps + self.stable_steps + self.decay_steps != self.total_steps:
            raise ValueError("Warmup, stable, and decay steps must partition training")
        if not 0 < self.phase_boundary_step < self.total_steps:
            raise ValueError("Phase boundary must lie strictly inside training")

    @property
    def phase0_fraction(self) -> float:
        return self.phase_boundary_step / self.total_steps

    @property
    def warmup_fraction(self) -> float:
        return self.warmup_steps / self.total_steps

    def learning_rate_at_steps(self, steps: np.ndarray) -> np.ndarray:
        """Match Levanter's linear-warmup, stable, cosine-decay schedule."""

        step = np.asarray(steps, dtype=float)
        learning_rate = np.ones_like(step)
        if self.warmup_steps > 0:
            warmup = step < self.warmup_steps
            learning_rate[warmup] = step[warmup] / self.warmup_steps
        decay_start = self.warmup_steps + self.stable_steps
        decay = step >= decay_start
        progress = np.clip((step[decay] - decay_start) / self.decay_steps, 0.0, 1.0)
        learning_rate[decay] = 0.5 * (1.0 + np.cos(np.pi * progress))
        return learning_rate

    def learning_rate(self, normalized_time: float | np.ndarray) -> float | np.ndarray:
        """Return peak-normalized LR at normalized token time."""

        time = np.asarray(normalized_time, dtype=float)
        value = self.learning_rate_at_steps(time * self.total_steps)
        if np.ndim(normalized_time) == 0:
            return float(value)
        return value

    def phase_learning_rate_masses(self) -> tuple[float, float]:
        """Return discrete peak-normalized LR mass before and after the phase boundary."""

        learning_rate = self.learning_rate_at_steps(np.arange(self.total_steps, dtype=float))
        return (
            float(np.sum(learning_rate[: self.phase_boundary_step])),
            float(np.sum(learning_rate[self.phase_boundary_step :])),
        )

    def optimizer_phase0_fraction(self) -> float:
        early, late = self.phase_learning_rate_masses()
        return early / (early + late)


TOTAL_STEPS = 1_000_000_000 // (128 * 2048)

COSINE_50_50 = OptimizerScheduleSpec(
    name=StarCoderSchedule.COSINE_50_50,
    total_steps=TOTAL_STEPS,
    phase_boundary_step=1904,
    warmup_steps=1000,
    stable_steps=0,
    decay_steps=TOTAL_STEPS - 1000,
    provenance=(
        "experiments/domain_phase_mix/experiment.py DEFAULT_MUON_CONFIG; "
        "experiments/domain_phase_mix/two_phase_starcoder_experiment.py"
    ),
)

WSD_80_20 = OptimizerScheduleSpec(
    name=StarCoderSchedule.WSD_80_20,
    total_steps=TOTAL_STEPS,
    phase_boundary_step=3040,
    warmup_steps=int(TOTAL_STEPS * 0.01),
    stable_steps=3040 - int(TOTAL_STEPS * 0.01),
    decay_steps=TOTAL_STEPS - 3040,
    provenance="experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py",
)


def schedule_for_name(name: str) -> OptimizerScheduleSpec:
    if name.startswith(StarCoderSchedule.COSINE_50_50.value):
        return COSINE_50_50
    if name.startswith(StarCoderSchedule.WSD_80_20.value):
        return WSD_80_20
    raise ValueError(f"Unknown StarCoder schedule {name!r}")
