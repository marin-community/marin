# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Early-allocation-gated final-phase plasticity models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    potential_phase_models as potential,
)

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class PrimerGeometry:
    potential_geometry: potential.PotentialGeometry
    phase0_epoch_coefficients: np.ndarray
    phase1_epoch_coefficients: np.ndarray

    def __post_init__(self) -> None:
        expected = self.potential_geometry.proportional_weights.shape
        if self.phase0_epoch_coefficients.shape != expected or self.phase1_epoch_coefficients.shape != expected:
            raise ValueError("Primer epoch coefficients do not match the mixture geometry")


@dataclass(frozen=True)
class PrimerConfig:
    primer_rate: float
    residual_plasticity: float
    response: potential.DebtResponse
    curvature: float
    offset: float
    l2: float

    @property
    def potential_config(self) -> potential.PotentialConfig:
        return potential.PotentialConfig(self.response, self.curvature, self.offset, self.l2)

    @property
    def key(self) -> str:
        return (
            f"primer={self.primer_rate:g},floor={self.residual_plasticity:g},"
            f"response={self.response.value},curvature={self.curvature:g},offset={self.offset:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class PrimedPlasticityModel:
    geometry: PrimerGeometry
    config: PrimerConfig
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, names = primed_design(self.geometry, weights, self.config)
        if names != self.head.feature_names:
            raise ValueError("Primer feature ordering changed")
        return self.head.predict(design)


def primer_geometry(panel: paired.PairedPanel) -> PrimerGeometry:
    return PrimerGeometry(
        potential.PotentialGeometry(
            domain_names=panel.domain_names,
            family_names=panel.family_names,
            family_members=panel.family_members,
            proportional_weights=panel.proportional_weights,
            total_epoch_coefficients=panel.c0 + panel.c1,
        ),
        np.asarray(panel.c0, dtype=float),
        np.asarray(panel.c1, dtype=float),
    )


def effective_evidence(
    geometry: PrimerGeometry,
    weights: np.ndarray,
    primer_rate: float,
    residual_plasticity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if primer_rate < 0.0:
        raise ValueError("Primer rate must be nonnegative")
    if not 0.0 <= residual_plasticity <= 1.0:
        raise ValueError("Residual plasticity must lie in [0, 1]")
    weights = potential.normalized_policy(weights)
    proportional = geometry.potential_geometry.proportional_weights
    phase0_exposure = weights[:, 0, :] * geometry.phase0_epoch_coefficients[None, :]
    phase1_exposure = weights[:, 1, :] * geometry.phase1_epoch_coefficients[None, :]
    proportional_phase0 = proportional * geometry.phase0_epoch_coefficients
    normalized_phase0 = phase0_exposure / np.maximum(proportional_phase0[None, :], NUMERICAL_FLOOR)
    primer = -np.expm1(-primer_rate * normalized_phase0)
    efficiency = residual_plasticity + (1.0 - residual_plasticity) * primer
    effective = phase0_exposure + efficiency * phase1_exposure

    proportional_primer = -np.expm1(-primer_rate)
    proportional_efficiency = residual_plasticity + (1.0 - residual_plasticity) * proportional_primer
    reference = proportional_phase0 + proportional_efficiency * proportional * geometry.phase1_epoch_coefficients
    physical = phase0_exposure + phase1_exposure
    return effective, reference, physical


def primed_design(
    geometry: PrimerGeometry,
    weights: np.ndarray,
    config: PrimerConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    effective, reference, physical = effective_evidence(
        geometry,
        weights,
        config.primer_rate,
        config.residual_plasticity,
    )
    response_config = config.potential_config
    ratio = effective / np.maximum(reference[None, :], NUMERICAL_FLOOR)
    pieces = [potential.debt_response(ratio, response_config)]
    names = [f"bucket_debt:{name}" for name in geometry.potential_geometry.domain_names]

    for family_name, members in zip(
        geometry.potential_geometry.family_names,
        geometry.potential_geometry.family_members,
        strict=True,
    ):
        if len(members) <= 1:
            continue
        family_ratio = effective[:, members].sum(axis=1) / max(float(reference[members].sum()), NUMERICAL_FLOOR)
        pieces.append(potential.debt_response(family_ratio[:, None], response_config))
        names.append(f"family_debt:{family_name}")

    proportional = geometry.potential_geometry.proportional_weights
    proportional_physical = proportional * (geometry.phase0_epoch_coefficients + geometry.phase1_epoch_coefficients)
    for family_name, members in zip(
        geometry.potential_geometry.family_names,
        geometry.potential_geometry.family_members,
        strict=True,
    ):
        family_physical = physical[:, members].sum(axis=1, keepdims=True)
        family_reference = float(proportional_physical[members].sum())
        pieces.append(
            potential.duplicate_mass(family_physical) - potential.duplicate_mass(np.asarray([[family_reference]]))
        )
        names.append(f"family_duplicate_mass:{family_name}")
    return np.column_stack(pieces), tuple(names)


def fit_primed_plasticity(
    geometry: PrimerGeometry,
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    config: PrimerConfig,
) -> PrimedPlasticityModel:
    design, names = primed_design(geometry, weights, config)
    head = paired.fit_linear_head(
        design[indices],
        np.asarray(target, dtype=float)[indices],
        names,
        coefficient_signs=np.ones(design.shape[1], dtype=int),
        l2=config.l2,
    )
    return PrimedPlasticityModel(geometry, config, head)
