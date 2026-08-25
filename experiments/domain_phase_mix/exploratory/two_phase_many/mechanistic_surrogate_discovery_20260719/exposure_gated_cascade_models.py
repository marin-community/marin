# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exposure-gated feature-to-competence cascade surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class CascadeGeometry:
    """Physical exposure geometry shared by all response targets."""

    domain_names: tuple[str, ...]
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    proportional_weights: np.ndarray
    phase0_epoch_coefficients: np.ndarray
    phase1_epoch_coefficients: np.ndarray

    def __post_init__(self) -> None:
        m = len(self.domain_names)
        if self.proportional_weights.shape != (m,):
            raise ValueError("Unexpected proportional-weight shape")
        if self.phase0_epoch_coefficients.shape != (m,) or self.phase1_epoch_coefficients.shape != (m,):
            raise ValueError("Unexpected epoch-coefficient shape")
        if not np.isclose(self.proportional_weights.sum(), 1.0, atol=1e-9):
            raise ValueError("Proportional weights are not normalized")
        covered = np.concatenate(self.family_members)
        if sorted(covered.tolist()) != list(range(m)):
            raise ValueError("Families do not partition cascade domains")


@dataclass(frozen=True)
class CascadeConfig:
    """Frozen nonlinear state and response hyperparameters."""

    feature_rate: float
    conversion_rate: float
    response_offset: float
    l2: float
    include_replay_harm: bool

    @property
    def key(self) -> str:
        return (
            f"feature={self.feature_rate:g},conversion={self.conversion_rate:g},"
            f"offset={self.response_offset:g},l2={self.l2:g},replay={int(self.include_replay_harm)}"
        )


@dataclass(frozen=True)
class CascadeState:
    """Final latent state and physical exposure for a policy batch."""

    feature_readiness: np.ndarray
    competence: np.ndarray
    exposure: np.ndarray


@dataclass(frozen=True)
class CascadeModel:
    """Fitted cascade response head over an exact latent-state transition."""

    geometry: CascadeGeometry
    config: CascadeConfig
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _names, _signs = cascade_design(self.geometry, weights, self.config)
        return self.head.predict(design)


def normalized_policy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError(f"Expected [n, 2, domain] policy, got {weights.shape}")
    if np.any(weights < -1e-10):
        raise ValueError("Negative mixture weight")
    clipped = np.maximum(weights, 0.0)
    totals = clipped.sum(axis=2, keepdims=True)
    if np.any(totals <= NUMERICAL_FLOOR):
        raise ValueError("Empty phase mixture")
    return clipped / totals


def exposure_update(
    feature: np.ndarray,
    competence: np.ndarray,
    exposure: np.ndarray,
    feature_rate: float,
    conversion_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the triangular state exactly through constant exposure increments."""

    feature = np.asarray(feature, dtype=float)
    competence = np.asarray(competence, dtype=float)
    exposure = np.maximum(np.asarray(exposure, dtype=float), 0.0)
    feature_survival = np.exp(-feature_rate * exposure)
    integrated_feature = exposure - (1.0 - feature) * (-np.expm1(-feature_rate * exposure)) / feature_rate
    next_feature = 1.0 - (1.0 - feature) * feature_survival
    next_competence = 1.0 - (1.0 - competence) * np.exp(-conversion_rate * integrated_feature)
    return np.clip(next_feature, 0.0, 1.0), np.clip(next_competence, 0.0, 1.0)


def cascade_state(geometry: CascadeGeometry, weights: np.ndarray, config: CascadeConfig) -> CascadeState:
    weights = normalized_policy(weights)
    phase0_exposure = weights[:, 0, :] * geometry.phase0_epoch_coefficients[None, :]
    phase1_exposure = weights[:, 1, :] * geometry.phase1_epoch_coefficients[None, :]
    feature = np.zeros_like(phase0_exposure)
    competence = np.zeros_like(phase0_exposure)
    feature, competence = exposure_update(
        feature,
        competence,
        phase0_exposure,
        config.feature_rate,
        config.conversion_rate,
    )
    feature, competence = exposure_update(
        feature,
        competence,
        phase1_exposure,
        config.feature_rate,
        config.conversion_rate,
    )
    return CascadeState(feature, competence, phase0_exposure + phase1_exposure)


def proportional_policy(geometry: CascadeGeometry, n: int = 1) -> np.ndarray:
    weights = np.broadcast_to(geometry.proportional_weights, (n, len(geometry.domain_names)))
    return np.stack([weights, weights], axis=1).copy()


def cascade_design(
    geometry: CascadeGeometry,
    weights: np.ndarray,
    config: CascadeConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    state = cascade_state(geometry, weights, config)
    reference = cascade_state(geometry, proportional_policy(geometry), config)
    competence_debt = np.log(
        (config.response_offset + reference.competence) / (config.response_offset + state.competence)
    )
    pieces: list[np.ndarray] = [competence_debt]
    names = [f"competence_debt:{name}" for name in geometry.domain_names]
    signs = [1] * len(geometry.domain_names)

    if config.include_replay_harm:
        for family_name, members in zip(geometry.family_names, geometry.family_members, strict=True):
            replay = np.maximum(state.exposure[:, members] - 1.0, 0.0).sum(axis=1)
            reference_replay = float(np.maximum(reference.exposure[0, members] - 1.0, 0.0).sum())
            pieces.append((replay - reference_replay)[:, None])
            names.append(f"literal_replay_harm:{family_name}")
            signs.append(1)

    return np.column_stack(pieces), tuple(names), np.asarray(signs, dtype=int)


def fit_cascade(
    geometry: CascadeGeometry,
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    config: CascadeConfig,
) -> CascadeModel:
    design, names, signs = cascade_design(geometry, weights, config)
    head = paired.fit_linear_head(
        design[indices],
        np.asarray(target, dtype=float)[indices],
        names,
        coefficient_signs=signs,
        l2=config.l2,
    )
    return CascadeModel(geometry, config, head)
