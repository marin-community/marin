# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon flow averaged over Levanter's exact marginal batch-composition law."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import hypergeom

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_anisotropic_polar_models as task,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_polar_matrix_models as muon,
)

MIXTURE_BLOCK_SIZE = 2048
GLOBAL_BATCH_SIZE = 128
INTEGRATION_STEPS_PER_UNIT = 64


@dataclass(frozen=True)
class BatchCompositionMuonConfig:
    """Frozen task geometry and whether composition is averaged before or after Muon."""

    task_angle_degrees: float
    rare_curvature: float
    input_anisotropy: float
    relaxation: float
    evaluation_rare_weight: float
    composition_rule: str

    @property
    def key(self) -> str:
        return (
            f"angle={self.task_angle_degrees:g},rare={self.rare_curvature:g},"
            f"anisotropy={self.input_anisotropy:g},relax={self.relaxation:g},"
            f"eval={self.evaluation_rare_weight:g},composition={self.composition_rule}"
        )


def task_config(config: BatchCompositionMuonConfig) -> task.MuonAnisotropicPolarConfig:
    return task.MuonAnisotropicPolarConfig(
        config.task_angle_degrees,
        config.rare_curvature,
        config.input_anisotropy,
        config.relaxation,
        config.evaluation_rare_weight,
        "newton_schulz",
    )


def rare_count_per_block(rare_weight: float) -> int:
    """Match MixtureDataset's integer count and largest-component remainder rule."""

    rare_weight = float(np.clip(rare_weight, 0.0, 1.0))
    broad_count = int((1.0 - rare_weight) * MIXTURE_BLOCK_SIZE)
    rare_count = int(rare_weight * MIXTURE_BLOCK_SIZE)
    remainder = MIXTURE_BLOCK_SIZE - broad_count - rare_count
    if broad_count >= rare_count:
        broad_count += remainder
    else:
        rare_count += remainder
    assert broad_count + rare_count == MIXTURE_BLOCK_SIZE
    return rare_count


def composition_distribution(rare_weight: float) -> tuple[np.ndarray, np.ndarray]:
    rare_count = rare_count_per_block(rare_weight)
    lower = max(0, GLOBAL_BATCH_SIZE - (MIXTURE_BLOCK_SIZE - rare_count))
    upper = min(GLOBAL_BATCH_SIZE, rare_count)
    counts = np.arange(lower, upper + 1, dtype=int)
    probabilities = hypergeom.pmf(counts, MIXTURE_BLOCK_SIZE, rare_count, GLOBAL_BATCH_SIZE)
    probabilities /= probabilities.sum()
    return counts.astype(float) / GLOBAL_BATCH_SIZE, probabilities


def domain_gradients(
    state: np.ndarray,
    config: BatchCompositionMuonConfig,
) -> tuple[np.ndarray, np.ndarray]:
    config_as_task = task_config(config)
    broad_target, rare_target = task.task_targets(config_as_task)
    broad_covariance, rare_covariance = task.task_covariances(config_as_task)
    broad = np.einsum("nij,jk->nik", state - broad_target[None, :, :], broad_covariance)
    rare = config.rare_curvature * np.einsum(
        "nij,jk->nik",
        state - rare_target[None, :, :],
        rare_covariance,
    )
    return broad, rare


def expected_update_direction(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: BatchCompositionMuonConfig,
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    rare_weight = np.asarray(rare_weight, dtype=float)
    broad, rare = domain_gradients(state, config)
    if config.composition_rule == "mean":
        realized_weight = np.asarray(
            [rare_count_per_block(weight) / MIXTURE_BLOCK_SIZE for weight in rare_weight],
            dtype=float,
        )
        gradient = (1.0 - realized_weight)[:, None, None] * broad + realized_weight[:, None, None] * rare
        return muon.update_direction(gradient, "newton_schulz")
    if config.composition_rule != "hypergeometric":
        raise ValueError(f"Unknown composition rule {config.composition_rule}")

    expected = np.zeros_like(state)
    rounded_counts = np.asarray([rare_count_per_block(weight) for weight in rare_weight], dtype=int)
    for rare_count in np.unique(rounded_counts):
        indices = np.flatnonzero(rounded_counts == rare_count)
        fractions, probabilities = composition_distribution(rare_count / MIXTURE_BLOCK_SIZE)
        gradients = (1.0 - fractions)[None, :, None, None] * broad[indices, None, :, :] + fractions[
            None, :, None, None
        ] * rare[indices, None, :, :]
        directions = muon.update_direction(gradients.reshape(-1, 2, 2), "newton_schulz").reshape(
            len(indices), len(fractions), 2, 2
        )
        expected[indices] = np.einsum("k,nkij->nij", probabilities, directions)
    return expected


def derivative(
    state: np.ndarray,
    rare_weight: np.ndarray,
    config: BatchCompositionMuonConfig,
) -> np.ndarray:
    direction = expected_update_direction(state, rare_weight, config)
    radial = np.sum(state * direction, axis=(1, 2), keepdims=True)
    return -config.relaxation * (direction - radial * state)


def phase_update(
    state: np.ndarray,
    rare_weight: np.ndarray,
    duration: float,
    config: BatchCompositionMuonConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    steps = max(1, int(np.ceil(duration * steps_per_unit)))
    step_size = duration / steps
    state = muon.normalize_state(np.asarray(state, dtype=float).copy())
    for _ in range(steps):
        k1 = derivative(state, rare_weight, config)
        k2 = derivative(muon.normalize_state(state + 0.5 * step_size * k1), rare_weight, config)
        k3 = derivative(muon.normalize_state(state + 0.5 * step_size * k2), rare_weight, config)
        k4 = derivative(muon.normalize_state(state + step_size * k3), rare_weight, config)
        state = muon.normalize_state(state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
    return state


def terminal_state(
    weights: np.ndarray,
    phase0_optimizer_fraction: float,
    config: BatchCompositionMuonConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    weights = muon.validate_weights(weights)
    state = muon.initial_state(len(weights))
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
    config: BatchCompositionMuonConfig,
    *,
    steps_per_unit: int = INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    state = terminal_state(weights, phase0_optimizer_fraction, config, steps_per_unit=steps_per_unit)
    config_as_task = task_config(config)
    broad_target, rare_target = task.task_targets(config_as_task)
    broad_covariance, rare_covariance = task.task_covariances(config_as_task)
    broad_debt = task.weighted_task_debt(state, broad_target, broad_covariance)
    rare_debt = task.weighted_task_debt(state, rare_target, rare_covariance)
    q = config.evaluation_rare_weight
    return (1.0 - q) * broad_debt + q * config.rare_curvature * rare_debt
