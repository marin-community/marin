# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convex tied potentials and phase laws derived from their geometry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)

NUMERICAL_FLOOR = 1e-12


class DebtResponse(StrEnum):
    INVERSE_POWER = "inverse_power"
    LOGARITHMIC = "logarithmic"


class PhaseLaw(StrEnum):
    WORK_DISSIPATION = "potential_work_dissipation"
    RELAXATION = "equilibrium_stress_relaxation"


@dataclass(frozen=True)
class PotentialConfig:
    response: DebtResponse
    curvature: float
    offset: float
    l2: float

    @property
    def key(self) -> str:
        return f"response={self.response.value},curvature={self.curvature:g},offset={self.offset:g},l2={self.l2:g}"


@dataclass(frozen=True)
class WorkDissipationConfig:
    l2: float

    @property
    def key(self) -> str:
        return f"l2={self.l2:g}"


@dataclass(frozen=True)
class RelaxationConfig:
    rate: float
    l2: float

    @property
    def key(self) -> str:
        return f"rate={self.rate:g},l2={self.l2:g}"


PhaseConfig = WorkDissipationConfig | RelaxationConfig


@dataclass(frozen=True)
class PotentialGeometry:
    domain_names: tuple[str, ...]
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    proportional_weights: np.ndarray
    total_epoch_coefficients: np.ndarray

    def __post_init__(self) -> None:
        m = len(self.domain_names)
        if self.proportional_weights.shape != (m,):
            raise ValueError("Unexpected proportional-weight shape")
        if self.total_epoch_coefficients.shape != (m,):
            raise ValueError("Unexpected epoch-coefficient shape")
        if not np.isclose(self.proportional_weights.sum(), 1.0, atol=1e-9):
            raise ValueError("Proportional weights are not normalized")
        covered = np.concatenate(self.family_members)
        if sorted(covered.tolist()) != list(range(m)):
            raise ValueError("Families do not partition potential domains")


@dataclass(frozen=True)
class ConvexPotential:
    geometry: PotentialGeometry
    config: PotentialConfig
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _names = potential_design(self.geometry, weights, self.config)
        return self.head.predict(design)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return potential_gradient(self.geometry, weights, self.config, self.head.coefficients_in_natural_units)

    def bregman(self, endpoint: np.ndarray, reference: np.ndarray) -> np.ndarray:
        endpoint = normalized_weights(endpoint)
        reference = normalized_weights(reference)
        difference = endpoint - reference
        return self.predict(endpoint) - self.predict(reference) - np.sum(self.gradient(reference) * difference, axis=1)


@dataclass(frozen=True)
class ZeroInterceptHead:
    feature_names: tuple[str, ...]
    feature_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.asarray((design / self.feature_scale[None, :]) @ self.coefficients, dtype=float)

    @property
    def coefficients_in_natural_units(self) -> np.ndarray:
        return self.coefficients / self.feature_scale


@dataclass(frozen=True)
class PhasePotentialModel:
    potential: ConvexPotential
    alpha0: float
    law: PhaseLaw
    config: PhaseConfig
    head: ZeroInterceptHead

    @property
    def alpha1(self) -> float:
        return 1.0 - self.alpha0

    def predict_delta(self, weights: np.ndarray) -> np.ndarray:
        design, _names, _signs = phase_design(self.potential, weights, self.alpha0, self.law, self.config)
        return self.head.predict(design)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        weights = normalized_policy(weights)
        aggregate = self.alpha0 * weights[:, 0, :] + self.alpha1 * weights[:, 1, :]
        return self.potential.predict(aggregate) + self.predict_delta(weights)


def normalized_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"Expected [n, domain] weights, got {weights.shape}")
    if np.any(weights < -1e-10):
        raise ValueError("Negative mixture weight")
    return np.maximum(weights, 0.0) / np.maximum(weights.sum(axis=1, keepdims=True), NUMERICAL_FLOOR)


def normalized_policy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError(f"Expected [n, 2, domain] policy, got {weights.shape}")
    return np.stack([normalized_weights(weights[:, phase, :]) for phase in range(2)], axis=1)


def debt_response(ratio: np.ndarray, config: PotentialConfig) -> np.ndarray:
    safe = np.maximum(ratio, 0.0) + config.offset
    reference = 1.0 + config.offset
    if config.response is DebtResponse.INVERSE_POWER:
        return safe ** (-config.curvature) - reference ** (-config.curvature)
    if config.response is DebtResponse.LOGARITHMIC:
        return np.log(reference) - np.log(safe)
    raise ValueError(f"Unsupported response {config.response}")


def debt_derivative(ratio: np.ndarray, config: PotentialConfig) -> np.ndarray:
    safe = np.maximum(ratio, 0.0) + config.offset
    if config.response is DebtResponse.INVERSE_POWER:
        return -config.curvature * safe ** (-config.curvature - 1.0)
    if config.response is DebtResponse.LOGARITHMIC:
        return -1.0 / safe
    raise ValueError(f"Unsupported response {config.response}")


def duplicate_mass(exposure: np.ndarray) -> np.ndarray:
    exposure = np.maximum(exposure, 0.0)
    return exposure - (1.0 - np.exp(-exposure))


def duplicate_mass_derivative(exposure: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-np.maximum(exposure, 0.0))


def potential_design(
    geometry: PotentialGeometry,
    weights: np.ndarray,
    config: PotentialConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    weights = normalized_weights(weights)
    ratio = weights / np.maximum(geometry.proportional_weights[None, :], NUMERICAL_FLOOR)
    pieces = [debt_response(ratio, config)]
    names = [f"bucket_debt:{name}" for name in geometry.domain_names]

    for family_name, members in zip(geometry.family_names, geometry.family_members, strict=True):
        if len(members) <= 1:
            continue
        family_weight = weights[:, members].sum(axis=1)
        family_reference = float(geometry.proportional_weights[members].sum())
        pieces.append(debt_response(family_weight[:, None] / max(family_reference, NUMERICAL_FLOOR), config))
        names.append(f"family_debt:{family_name}")

    physical_exposure = weights * geometry.total_epoch_coefficients[None, :]
    for family_name, members in zip(geometry.family_names, geometry.family_members, strict=True):
        family_exposure = physical_exposure[:, members].sum(axis=1, keepdims=True)
        proportional_exposure = float(
            np.sum(geometry.proportional_weights[members] * geometry.total_epoch_coefficients[members])
        )
        pieces.append(duplicate_mass(family_exposure) - duplicate_mass(np.asarray([[proportional_exposure]])))
        names.append(f"family_duplicate_mass:{family_name}")

    return np.column_stack(pieces), tuple(names)


def potential_gradient(
    geometry: PotentialGeometry,
    weights: np.ndarray,
    config: PotentialConfig,
    coefficients: np.ndarray,
) -> np.ndarray:
    weights = normalized_weights(weights)
    n, m = weights.shape
    gradient = np.zeros((n, m), dtype=float)
    cursor = 0

    ratio = weights / np.maximum(geometry.proportional_weights[None, :], NUMERICAL_FLOOR)
    bucket_derivative = debt_derivative(ratio, config) / np.maximum(
        geometry.proportional_weights[None, :], NUMERICAL_FLOOR
    )
    gradient += bucket_derivative * coefficients[cursor : cursor + m][None, :]
    cursor += m

    for members in geometry.family_members:
        if len(members) <= 1:
            continue
        family_weight = weights[:, members].sum(axis=1)
        family_reference = float(geometry.proportional_weights[members].sum())
        derivative = debt_derivative(family_weight / max(family_reference, NUMERICAL_FLOOR), config)
        gradient[:, members] += coefficients[cursor] * derivative[:, None] / max(family_reference, NUMERICAL_FLOOR)
        cursor += 1

    physical_exposure = weights * geometry.total_epoch_coefficients[None, :]
    for members in geometry.family_members:
        family_exposure = physical_exposure[:, members].sum(axis=1)
        derivative = duplicate_mass_derivative(family_exposure)
        gradient[:, members] += (
            coefficients[cursor] * derivative[:, None] * geometry.total_epoch_coefficients[members][None, :]
        )
        cursor += 1

    if cursor != len(coefficients):
        raise ValueError(f"Potential gradient consumed {cursor} of {len(coefficients)} coefficients")
    return gradient


def fit_potential(
    geometry: PotentialGeometry,
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    config: PotentialConfig,
) -> ConvexPotential:
    design, names = potential_design(geometry, weights, config)
    head = paired.fit_linear_head(
        design[indices],
        np.asarray(target, dtype=float)[indices],
        names,
        coefficient_signs=np.ones(design.shape[1], dtype=int),
        l2=config.l2,
    )
    return ConvexPotential(geometry, config, head)


def transported_endpoint(weights: np.ndarray, alpha0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = normalized_policy(weights)
    alpha1 = 1.0 - alpha0
    aggregate = alpha0 * weights[:, 0, :] + alpha1 * weights[:, 1, :]
    displacement = alpha0 * alpha1 * (weights[:, 1, :] - weights[:, 0, :])
    endpoint = aggregate + displacement
    if np.any(endpoint < -1e-10) or not np.allclose(endpoint.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Phase-fiber endpoint left the simplex")
    return aggregate, np.maximum(endpoint, 0.0), displacement


def phase_design(
    potential: ConvexPotential,
    weights: np.ndarray,
    alpha0: float,
    law: PhaseLaw,
    config: PhaseConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    weights = normalized_policy(weights)
    alpha1 = 1.0 - alpha0
    aggregate, endpoint, displacement = transported_endpoint(weights, alpha0)
    if law is PhaseLaw.WORK_DISSIPATION:
        work = np.sum(potential.gradient(aggregate) * displacement, axis=1)
        dissipation = np.maximum(potential.bregman(endpoint, aggregate), 0.0)
        return (
            np.column_stack([work, dissipation]),
            ("potential_work", "bregman_dissipation"),
            np.asarray([0, 1], dtype=int),
        )
    if law is PhaseLaw.RELAXATION:
        if not isinstance(config, RelaxationConfig):
            raise TypeError("Relaxation law requires RelaxationConfig")
        rate = config.rate
        early_coefficient = np.exp(-rate * alpha1) * (1.0 - np.exp(-rate * alpha0))
        late_coefficient = 1.0 - np.exp(-rate * alpha1)
        equilibrium_gap = early_coefficient * (
            potential.predict(weights[:, 0, :]) - potential.predict(aggregate)
        ) + late_coefficient * (potential.predict(weights[:, 1, :]) - potential.predict(aggregate))
        return equilibrium_gap[:, None], ("relaxation_equilibrium_gap",), np.asarray([1], dtype=int)
    raise ValueError(f"Unsupported phase law {law}")


def fit_zero_intercept_head(
    design: np.ndarray,
    target: np.ndarray,
    feature_names: Iterable[str],
    coefficient_signs: np.ndarray,
    l2: float,
) -> ZeroInterceptHead:
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    signs = np.asarray(coefficient_signs, dtype=int)
    if design.ndim != 2 or target.shape != (len(design),):
        raise ValueError("Unexpected zero-intercept design or target shape")
    scale = np.sqrt(np.mean(design**2, axis=0))
    scale = np.where(scale > NUMERICAL_FLOOR, scale, 1.0)
    standardized = design / scale[None, :]
    if l2 > 0.0:
        standardized = np.vstack([standardized, np.sqrt(l2) * np.eye(design.shape[1])])
        target = np.concatenate([target, np.zeros(design.shape[1], dtype=float)])
    lower = np.where(signs > 0, 0.0, -np.inf)
    upper = np.where(signs < 0, 0.0, np.inf)
    solution = lsq_linear(standardized, target, bounds=(lower, upper), lsmr_tol="auto", max_iter=1000)
    if not solution.success:
        raise RuntimeError(f"Phase fit failed: {solution.message}")
    return ZeroInterceptHead(tuple(feature_names), scale, np.asarray(solution.x, dtype=float))


def fit_phase_potential(
    potential: ConvexPotential,
    weights: np.ndarray,
    phase_delta: np.ndarray,
    indices: np.ndarray,
    alpha0: float,
    law: PhaseLaw,
    config: PhaseConfig,
) -> PhasePotentialModel:
    design, names, signs = phase_design(potential, weights, alpha0, law, config)
    head = fit_zero_intercept_head(design[indices], phase_delta[indices], names, signs, config.l2)
    return PhasePotentialModel(potential, alpha0, law, config, head)
