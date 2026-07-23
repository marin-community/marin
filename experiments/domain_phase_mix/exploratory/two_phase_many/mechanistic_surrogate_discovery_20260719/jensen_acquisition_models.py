# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exposure-rate-curved acquisition with physical replay harm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class JensenAcquisitionConfig:
    rate_power: float
    shortage_power: float
    shortage_offset: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"rate={self.rate_power:g},shortage={self.shortage_power:g},offset={self.shortage_offset:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class JensenAcquisitionModel:
    panel: paired.PairedPanel
    config: JensenAcquisitionConfig
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _names = design_matrix(self.panel, weights, self.config)
        return self.head.predict(design)


def acquisition_ratio(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    rate_power: float,
) -> np.ndarray:
    """Return useful acquisition relative to the proportional tied policy."""
    dose = panel.alpha0 * weights[:, 0, :] ** rate_power + panel.alpha1 * weights[:, 1, :] ** rate_power
    reference = panel.proportional_weights**rate_power
    return dose / np.maximum(reference[None, :], NUMERICAL_FLOOR)


def shortage_debt(ratio: np.ndarray, power: float, offset: float) -> np.ndarray:
    safe = np.maximum(ratio, 0.0) + offset
    reference = 1.0 + offset
    return safe ** (-power) - reference ** (-power)


def duplicate_mass(exposure: np.ndarray) -> np.ndarray:
    exposure = np.maximum(exposure, 0.0)
    return exposure - (1.0 - np.exp(-exposure))


def design_matrix(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    config: JensenAcquisitionConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, panel.m):
        raise ValueError(f"Unexpected weights shape {weights.shape}")
    ratio = acquisition_ratio(panel, weights, config.rate_power)
    debt = shortage_debt(ratio, config.shortage_power, config.shortage_offset)
    exposure = weights[:, 0, :] * panel.c0[None, :] + weights[:, 1, :] * panel.c1[None, :]
    replay = duplicate_mass(exposure) - duplicate_mass(panel.proportional_exposure)[None, :]
    names = tuple(f"shortage:{name}" for name in panel.domain_names) + tuple(
        f"physical_replay:{name}" for name in panel.domain_names
    )
    return np.column_stack([debt, replay]), names


def fit_model(
    panel: paired.PairedPanel,
    target: np.ndarray,
    indices: np.ndarray,
    config: JensenAcquisitionConfig,
) -> JensenAcquisitionModel:
    design, names = design_matrix(panel, panel.weights, config)
    head = paired.fit_linear_head(
        design[indices],
        target[indices],
        names,
        coefficient_signs=np.ones(design.shape[1], dtype=int),
        l2=config.l2,
    )
    return JensenAcquisitionModel(panel, config, head)
