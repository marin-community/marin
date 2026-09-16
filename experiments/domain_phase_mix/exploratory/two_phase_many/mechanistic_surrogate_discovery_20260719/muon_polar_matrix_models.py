# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matrix task flow with the polar-gradient map used by Muon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTEGRATION_STEPS_PER_UNIT = 96
NUMERICAL_FLOOR = 1e-12
MATRIX_SIZE = 2
MUON_EPSILON = 1e-5
MUON_QUINTIC_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


@dataclass(frozen=True)
class MuonPolarMatrixConfig:
    """Frozen task geometry, transition rule, and terminal response."""

    task_angle_degrees: float
    rare_curvature: float
    relaxation: float
    evaluation_rare_weight: float
    update_rule: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},rare={self.rare_curvature:g},"
            f"relax={self.relaxation:g},eval={self.evaluation_rare_weight:g},rule={self.update_rule}"
        )


def validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[1:] != (2, 2):
        raise ValueError(f"Expected [policy, phase, broad_or_rare] weights, got {weights.shape}")
    if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("Each phase must be a nonnegative simplex mixture")
    return np.maximum(weights, 0.0)


def rotation(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float)


def task_targets(config: MuonPolarMatrixConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-norm broad and rare representation targets."""

    broad = np.eye(MATRIX_SIZE, dtype=float) / np.sqrt(MATRIX_SIZE)
    rare = rotation(config.task_angle_degrees) / np.sqrt(MATRIX_SIZE)
    return broad, rare


def initial_state(num_policies: int) -> np.ndarray:
    """Use one fixed isotropic non-task-aligned representation initialization."""

    state = rotation(-90.0) / np.sqrt(MATRIX_SIZE)
    return np.broadcast_to(state, (num_policies, MATRIX_SIZE, MATRIX_SIZE)).copy()


def task_gradient(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: MuonPolarMatrixConfig,
) -> np.ndarray:
    broad, rare = task_targets(config)
    rare_weight = np.clip(np.asarray(rare_weight, dtype=float), 0.0, 1.0)
    broad_gradient = state - broad[None, :, :]
    rare_gradient = state - rare[None, :, :]
    return (1.0 - rare_weight)[:, None, None] * broad_gradient + config.rare_curvature * rare_weight[
        :, None, None
    ] * rare_gradient


def update_direction(gradient: np.ndarray, rule: str) -> np.ndarray:
    """Apply one declared Muon or exact-ablation update geometry."""

    gradient = np.asarray(gradient, dtype=float)
    norm = np.linalg.norm(gradient, axis=(1, 2), keepdims=True)
    if rule == "euclidean":
        return gradient
    if rule == "normalized":
        return gradient / np.maximum(norm, NUMERICAL_FLOOR)
    if rule == "newton_schulz":
        direction = gradient / (norm + MUON_EPSILON)
        for a, b, c in MUON_QUINTIC_COEFFICIENTS:
            gram = np.einsum("nij,nkj->nik", direction, direction)
            gram_squared = np.einsum("nij,njk->nik", gram, gram)
            transform = b * gram + c * gram_squared
            direction = a * direction + np.einsum("nij,njk->nik", transform, direction)
        direction_norm = np.linalg.norm(direction, axis=(1, 2), keepdims=True)
        return direction / np.maximum(direction_norm, NUMERICAL_FLOOR)
    if rule != "polar":
        raise ValueError(f"Unknown update rule {rule}")
    u, singular_values, vh = np.linalg.svd(gradient, full_matrices=False)
    polar = np.einsum("nij,njk->nik", u, vh) / np.sqrt(MATRIX_SIZE)
    return np.where((singular_values.sum(axis=1) > NUMERICAL_FLOOR)[:, None, None], polar, 0.0)


def derivative(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: MuonPolarMatrixConfig,
) -> np.ndarray:
    gradient = task_gradient(state, rare_weight, config)
    direction = update_direction(gradient, config.update_rule)
    # MuonH retracts onto the constant-Frobenius-norm sphere after every update.
    radial = np.sum(state * direction, axis=(1, 2), keepdims=True)
    tangent = direction - radial * state
    return -config.relaxation * tangent


def normalize_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state, axis=(1, 2), keepdims=True)
    return state / np.maximum(norm, NUMERICAL_FLOOR)


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: MuonPolarMatrixConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    """Integrate autonomous tangent flow and retract numerical drift."""

    steps = max(1, int(np.ceil(steps_per_unit * duration)))
    step_size = duration / steps
    state = normalize_state(np.asarray(state, dtype=float).copy())
    rare_weight = np.asarray(rare_weight, dtype=float)
    for _ in range(steps):
        k1 = derivative(state, rare_weight, config)
        k2 = derivative(normalize_state(state + 0.5 * step_size * k1), rare_weight, config)
        k3 = derivative(normalize_state(state + 0.5 * step_size * k2), rare_weight, config)
        k4 = derivative(normalize_state(state + step_size * k3), rare_weight, config)
        state = normalize_state(state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
    if not np.isfinite(state).all():
        raise FloatingPointError(f"Non-finite matrix state for {config.key}")
    return state


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MuonPolarMatrixConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weights = validate_weights(weights)
    state = initial_state(len(weights))
    state = phase_update(
        state,
        weights[:, 0, 1],
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    return phase_update(
        state,
        weights[:, 1, 1],
        1.0 - phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )


def response_feature(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: MuonPolarMatrixConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    state = terminal_state(
        weights,
        phase0_optimizer_fraction,
        config,
        steps_per_unit=steps_per_unit,
    )
    broad, rare = task_targets(config)
    broad_debt = 0.5 * np.sum((state - broad[None, :, :]) ** 2, axis=(1, 2))
    rare_debt = 0.5 * np.sum((state - rare[None, :, :]) ** 2, axis=(1, 2))
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad_debt + q * config.rare_curvature * rare_debt


def tied_semigroup_error(config: MuonPolarMatrixConfig, phase0_fraction: float) -> float:
    rng = np.random.default_rng(20260719)
    rare_weight = rng.uniform(size=32)
    state = normalize_state(rng.normal(size=(32, MATRIX_SIZE, MATRIX_SIZE)))
    first = phase_update(state, rare_weight, phase0_fraction, config, steps_per_unit=384)
    split = phase_update(first, rare_weight, 1.0 - phase0_fraction, config, steps_per_unit=384)
    whole = phase_update(state, rare_weight, 1.0, config, steps_per_unit=384)
    return float(np.max(np.abs(split - whole)))


def polar_separation(config: MuonPolarMatrixConfig) -> float:
    """Measure whether the matrix polar factor differs from vector normalization."""

    rng = np.random.default_rng(20260719)
    state = normalize_state(rng.normal(size=(64, MATRIX_SIZE, MATRIX_SIZE)))
    gradient = task_gradient(state, rng.uniform(size=64), config)
    polar = update_direction(gradient, "polar")
    normalized = update_direction(gradient, "normalized")
    return float(np.mean(np.linalg.norm(polar - normalized, axis=(1, 2))))
