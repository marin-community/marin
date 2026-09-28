# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coupled Muon feature and Adam-like task-readout dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 128
INITIAL_READOUT = 0.1
NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class DualMuonAdamConfig:
    """Dimensionless task geometry and optimizer-channel clocks."""

    task_angle_degrees: float
    muon_relaxation: float
    readout_rate_ratio: float
    evaluation_rare_weight: float
    transition: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},muon={self.muon_relaxation:g},"
            f"readout_ratio={self.readout_rate_ratio:g},eval={self.evaluation_rare_weight:g},"
            f"transition={self.transition}"
        )


def task_vectors(config: DualMuonAdamConfig) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(config.task_angle_degrees)
    return np.asarray([1.0, 0.0]), np.asarray([np.cos(angle), np.sin(angle)])


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    return feature / np.maximum(norm, NUMERICAL_FLOOR)


def initial_state(n: int) -> tuple[np.ndarray, np.ndarray]:
    feature = np.repeat(np.asarray([[1.0, 0.0]]), n, axis=0)
    readouts = np.full((n, 2), INITIAL_READOUT, dtype=float)
    return feature, readouts


def derivative(
    feature: np.ndarray,
    readouts: np.ndarray,
    rare_weight: np.ndarray,
    config: DualMuonAdamConfig,
) -> tuple[np.ndarray, np.ndarray]:
    broad, rare = task_vectors(config)
    projection_broad = feature @ broad
    projection_rare = feature @ rare
    error_broad = readouts[:, 0] * projection_broad - 1.0
    error_rare = readouts[:, 1] * projection_rare - 1.0
    gradient_feature = (1.0 - rare_weight)[:, None] * (error_broad * readouts[:, 0])[:, None] * broad[
        None, :
    ] + rare_weight[:, None] * (error_rare * readouts[:, 1])[:, None] * rare[None, :]
    tangent = gradient_feature - np.sum(feature * gradient_feature, axis=1, keepdims=True) * feature
    if config.transition == "frozen_feature":
        feature_derivative = np.zeros_like(feature)
    elif config.transition in {"split_channel", "equal_channel"}:
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        feature_derivative = -config.muon_relaxation * tangent / np.maximum(tangent_norm, 1e-5)
    else:
        raise ValueError(f"Unknown transition {config.transition}")

    ratio = 1.0 if config.transition == "equal_channel" else config.readout_rate_ratio
    readout_rate = config.muon_relaxation * ratio
    readout_derivative = np.column_stack(
        [
            -readout_rate * (1.0 - rare_weight) * error_broad * projection_broad,
            -readout_rate * rare_weight * error_rare * projection_rare,
        ]
    )
    return feature_derivative, readout_derivative


def phase_update(
    feature: np.ndarray,
    readouts: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: DualMuonAdamConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(np.ceil(duration * steps_per_unit)))
    step_size = duration / steps
    feature = normalize_feature(np.asarray(feature, dtype=float).copy())
    readouts = np.asarray(readouts, dtype=float).copy()
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        f1, h1 = derivative(feature, readouts, rare_weight, config)
        f2, h2 = derivative(
            normalize_feature(feature + 0.5 * step_size * f1),
            readouts + 0.5 * step_size * h1,
            rare_weight,
            config,
        )
        f3, h3 = derivative(
            normalize_feature(feature + 0.5 * step_size * f2),
            readouts + 0.5 * step_size * h2,
            rare_weight,
            config,
        )
        f4, h4 = derivative(
            normalize_feature(feature + step_size * f3),
            readouts + step_size * h3,
            rare_weight,
            config,
        )
        feature = normalize_feature(feature + step_size * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0)
        readouts += step_size * (h1 + 2.0 * h2 + 2.0 * h3 + h4) / 6.0
    if not np.isfinite(feature).all() or not np.isfinite(readouts).all():
        raise FloatingPointError(f"Non-finite dual-channel state for {config.key}")
    return feature, readouts


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: DualMuonAdamConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, domain] weights, got {weights.shape}")
    feature, readouts = initial_state(len(weights))
    feature, readouts = phase_update(
        feature,
        readouts,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return phase_update(
        feature,
        readouts,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: DualMuonAdamConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    feature, readouts = terminal_state(
        weights,
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    broad, rare = task_vectors(config)
    broad_error = readouts[:, 0] * (feature @ broad) - 1.0
    rare_error = readouts[:, 1] * (feature @ rare) - 1.0
    q = config.evaluation_rare_weight
    return 0.5 * ((1.0 - q) * broad_error**2 + q * rare_error**2)


def integration_error(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: DualMuonAdamConfig,
) -> float:
    coarse = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=128)
    fine = response_feature(weights, phase0_optimizer_fraction, config, steps_per_unit=384)
    return float(np.max(np.abs(coarse - fine)))
