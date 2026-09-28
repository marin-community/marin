# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Family-pooled relaxation of independently fitted one-phase loss components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    potential_phase_models as potential,
)

RATE_ZERO_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ComponentRelaxationConfig:
    family_rates: tuple[float, ...]

    @property
    def key(self) -> str:
        return "rates=" + ",".join(f"{rate:g}" for rate in self.family_rates)


@dataclass(frozen=True)
class ComponentRelaxationModel:
    potential: potential.ConvexPotential
    alpha0: float
    config: ComponentRelaxationConfig
    feature_family_indices: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        weights = potential.normalized_policy(weights)
        phase0, names0 = potential.potential_design(self.potential.geometry, weights[:, 0, :], self.potential.config)
        phase1, names1 = potential.potential_design(self.potential.geometry, weights[:, 1, :], self.potential.config)
        if names0 != self.potential.head.feature_names or names1 != names0:
            raise ValueError("Potential feature ordering changed during component relaxation")
        rates = np.asarray(self.config.family_rates, dtype=float)[self.feature_family_indices]
        early = early_component_weight(rates, self.alpha0)
        terminal_design = early[None, :] * phase0 + (1.0 - early)[None, :] * phase1
        return self.potential.head.predict(terminal_design)


def early_component_weight(rate: np.ndarray, alpha0: float) -> np.ndarray:
    """Return the phase-0 contribution after normalized-horizon calibration."""

    rate = np.asarray(rate, dtype=float)
    if np.any(rate < 0.0):
        raise ValueError("Relaxation rates must be nonnegative")
    alpha1 = 1.0 - alpha0
    numerator = np.exp(-rate * alpha1) * (-np.expm1(-rate * alpha0))
    denominator = -np.expm1(-rate)
    weight = np.full(rate.shape, alpha0, dtype=float)
    finite_rate = rate >= RATE_ZERO_TOLERANCE
    weight[finite_rate] = numerator[finite_rate] / denominator[finite_rate]
    return weight


def feature_family_indices(
    geometry: potential.PotentialGeometry,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    domain_to_family = {}
    for family_index, members in enumerate(geometry.family_members):
        for member in members:
            domain_to_family[geometry.domain_names[int(member)]] = family_index
    family_to_index = {name: index for index, name in enumerate(geometry.family_names)}
    indices = []
    for feature in feature_names:
        kind, name = feature.split(":", maxsplit=1)
        if kind == "bucket_debt":
            indices.append(domain_to_family[name])
        elif kind in {"family_debt", "family_duplicate_mass"}:
            indices.append(family_to_index[name])
        else:
            raise ValueError(f"No semantic-family assignment for feature {feature}")
    return np.asarray(indices, dtype=int)


def fit_component_relaxation(
    tied_potential: potential.ConvexPotential,
    alpha0: float,
    config: ComponentRelaxationConfig,
) -> ComponentRelaxationModel:
    family_count = len(tied_potential.geometry.family_names)
    if len(config.family_rates) != family_count:
        raise ValueError(f"Expected {family_count} rates, found {len(config.family_rates)}")
    indices = feature_family_indices(tied_potential.geometry, tied_potential.head.feature_names)
    return ComponentRelaxationModel(tied_potential, alpha0, config, indices)
