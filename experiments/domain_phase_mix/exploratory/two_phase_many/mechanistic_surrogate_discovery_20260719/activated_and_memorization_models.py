# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Activated annealing flow and recoverable replay-memorization dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    sgd_drift_diffusion_models as schedule_models,
)

INTEGRATION_STEPS = 512
NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class ActivatedFlowConfig:
    curvature_ratio: float
    speed: float
    barrier: float
    evaluation_mix: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"curvature={self.curvature_ratio:g},speed={self.speed:g},barrier={self.barrier:g},"
            f"eval={self.evaluation_mix:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class MemorizationConfig:
    accumulation_rate: float
    recovery_rate: float
    response_offset: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"accumulate={self.accumulation_rate:g},recover={self.recovery_rate:g},"
            f"offset={self.response_offset:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class ActivatedHead:
    intercept: float
    amplitude: float


@dataclass(frozen=True)
class ActivatedModel:
    config: ActivatedFlowConfig
    phase0_fraction: float
    schedule: schedule_models.Schedule
    head: ActivatedHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        feature = activated_response_feature(weights, self.phase0_fraction, self.schedule, self.config)
        return self.head.intercept + self.head.amplitude * feature


@dataclass(frozen=True)
class MemorizationGeometry:
    domain_names: tuple[str, ...]
    phase0_epoch_coefficients: np.ndarray
    phase1_epoch_coefficients: np.ndarray
    proportional_weights: np.ndarray


@dataclass(frozen=True)
class MemorizationState:
    unique_coverage: np.ndarray
    memorization_load: np.ndarray


@dataclass(frozen=True)
class MemorizationModel:
    geometry: MemorizationGeometry
    config: MemorizationConfig
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _names = memorization_design(self.geometry, weights, self.config)
        return self.head.predict(design)


def activated_mobility(eta: float, barrier: float) -> float:
    if barrier == 0.0:
        return 1.0
    return float(np.exp(-barrier / max(eta, 1e-6)))


def activated_phase_masses(
    phase0_fraction: float,
    schedule: schedule_models.Schedule,
    barrier: float,
) -> tuple[float, float]:
    step = 1.0 / INTEGRATION_STEPS
    masses = [0.0, 0.0]
    for index in range(INTEGRATION_STEPS):
        time = (index + 0.5) * step
        phase = 0 if time < phase0_fraction else 1
        eta = schedule_models.learning_rate(time, phase0_fraction, schedule)
        masses[phase] += step * activated_mobility(eta, barrier)
    return masses[0], masses[1]


def activated_terminal_state(
    weights: np.ndarray,
    phase0_fraction: float,
    schedule: schedule_models.Schedule,
    config: ActivatedFlowConfig,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    state = np.zeros(len(weights), dtype=float)
    masses = activated_phase_masses(phase0_fraction, schedule, config.barrier)
    for phase, mass in enumerate(masses):
        rare_weight = weights[:, phase, 1]
        hessian = 1.0 - rare_weight + config.curvature_ratio * rare_weight
        equilibrium = 0.5 * (1.0 - rare_weight - config.curvature_ratio * rare_weight) / hessian
        state = equilibrium + (state - equilibrium) * np.exp(-config.speed * hessian * mass)
    return state


def activated_response_feature(
    weights: np.ndarray,
    phase0_fraction: float,
    schedule: schedule_models.Schedule,
    config: ActivatedFlowConfig,
) -> np.ndarray:
    state = activated_terminal_state(weights, phase0_fraction, schedule, config)
    broad = 0.5 * (state + 0.5) ** 2
    rare = 0.5 * config.curvature_ratio * (state - 0.5) ** 2
    return (1.0 - config.evaluation_mix) * broad + config.evaluation_mix * rare


def fit_activated_model(
    weights: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    phase0_fraction: float,
    schedule: schedule_models.Schedule,
    config: ActivatedFlowConfig,
) -> ActivatedModel:
    feature = activated_response_feature(weights, phase0_fraction, schedule, config)
    feature_mean = float(np.mean(feature[train]))
    feature_scale = max(float(np.std(feature[train])), 1e-8)
    standardized = (feature[train] - feature_mean) / feature_scale
    target_mean = float(np.mean(target[train]))
    centered_target = target[train] - target_mean
    coefficient = max(
        float(standardized @ centered_target / (standardized @ standardized + config.l2)),
        0.0,
    )
    amplitude = coefficient / feature_scale
    intercept = target_mean - amplitude * feature_mean
    return ActivatedModel(config, phase0_fraction, schedule, ActivatedHead(intercept, amplitude))


def normalized_policy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError(f"Expected [n, 2, domain] policy, got {weights.shape}")
    clipped = np.maximum(weights, 0.0)
    return clipped / np.maximum(clipped.sum(axis=2, keepdims=True), NUMERICAL_FLOOR)


def memorization_state(
    geometry: MemorizationGeometry,
    weights: np.ndarray,
    config: MemorizationConfig,
) -> MemorizationState:
    weights = normalized_policy(weights)
    n, _phases, domains = weights.shape
    exposure = np.zeros((n, domains), dtype=float)
    load = np.zeros((n, domains), dtype=float)
    step = 1.0 / INTEGRATION_STEPS
    alpha0 = float(
        np.mean(
            geometry.phase0_epoch_coefficients
            / np.maximum(
                geometry.phase0_epoch_coefficients + geometry.phase1_epoch_coefficients,
                NUMERICAL_FLOOR,
            )
        )
    )
    for index in range(INTEGRATION_STEPS):
        time = (index + 0.5) * step
        phase = 0 if time < alpha0 else 1
        phase_fraction = alpha0 if phase == 0 else 1.0 - alpha0
        epoch_coefficients = geometry.phase0_epoch_coefficients if phase == 0 else geometry.phase1_epoch_coefficients
        mixture = weights[:, phase, :]
        exposure_rate = mixture * epoch_coefficients[None, :] / max(phase_fraction, NUMERICAL_FLOOR)
        duplicate_probability = 1.0 - np.exp(-exposure)
        load_rate = (
            config.accumulation_rate * mixture * duplicate_probability * (1.0 - load)
            - config.recovery_rate * (1.0 - mixture) * load
        )
        exposure += step * exposure_rate
        load = np.clip(load + step * load_rate, 0.0, 1.0)
    return MemorizationState(1.0 - np.exp(-exposure), load)


def proportional_policy(geometry: MemorizationGeometry, n: int = 1) -> np.ndarray:
    weights = np.broadcast_to(geometry.proportional_weights, (n, len(geometry.domain_names)))
    return np.stack([weights, weights], axis=1).copy()


def memorization_design(
    geometry: MemorizationGeometry,
    weights: np.ndarray,
    config: MemorizationConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    state = memorization_state(geometry, weights, config)
    reference = memorization_state(geometry, proportional_policy(geometry), config)
    coverage_debt = np.log(
        (config.response_offset + reference.unique_coverage) / (config.response_offset + state.unique_coverage)
    )
    load_excess = state.memorization_load - reference.memorization_load
    design = np.column_stack([coverage_debt, load_excess])
    names = tuple(
        [f"unique_coverage_debt:{name}" for name in geometry.domain_names]
        + [f"recoverable_memorization:{name}" for name in geometry.domain_names]
    )
    return design, names


def fit_memorization_model(
    geometry: MemorizationGeometry,
    weights: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    config: MemorizationConfig,
) -> MemorizationModel:
    design, names = memorization_design(geometry, weights, config)
    head = paired.fit_linear_head(
        design[indices],
        np.asarray(target, dtype=float)[indices],
        names,
        coefficient_signs=np.ones(design.shape[1], dtype=int),
        l2=config.l2,
    )
    return MemorizationModel(geometry, config, head)
