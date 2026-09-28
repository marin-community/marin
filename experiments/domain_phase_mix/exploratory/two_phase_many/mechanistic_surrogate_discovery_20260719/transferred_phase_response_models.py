# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transferred phase-response basis with a separately identified tied spine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)

NUMERICAL_FLOOR = 1e-12


@dataclass(frozen=True)
class SourceConfig:
    denominator_offset: float
    l2: float

    @property
    def key(self) -> str:
        return f"tau={self.denominator_offset:g},l2={self.l2:g}"


@dataclass(frozen=True)
class TargetConfig:
    l2: float
    include_contrast_cost: bool = True

    @property
    def key(self) -> str:
        return f"l2={self.l2:g},contrast={int(self.include_contrast_cost)}"


@dataclass(frozen=True)
class ConstrainedHead:
    coefficients: np.ndarray
    feature_scale: np.ndarray

    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(design @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class SourceDirection:
    config: SourceConfig
    family_direction: np.ndarray
    source_contrast_coefficient: float
    source_head: ConstrainedHead


@dataclass(frozen=True)
class TargetPhaseModel:
    source: SourceDirection
    config: TargetConfig
    amplitude: float
    contrast_coefficient: float
    head: ConstrainedHead

    def predict_delta(self, panel: paired.PairedPanel, weights: np.ndarray) -> np.ndarray:
        design = target_design(panel, weights, self.source, self.config)
        return self.head.predict(design)


def normalized_phase_displacement(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    denominator_offset: float,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    aggregate = panel.alpha0 * weights[:, 0, :] + panel.alpha1 * weights[:, 1, :]
    displacement = panel.alpha0 * panel.alpha1 * (weights[:, 1, :] - weights[:, 0, :])
    denominator = aggregate + denominator_offset * panel.proportional_weights[None, :]
    return displacement / np.maximum(denominator, NUMERICAL_FLOOR)


def source_design(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    config: SourceConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    normalized = normalized_phase_displacement(panel, weights, config.denominator_offset)
    family = np.column_stack([normalized[:, members].sum(axis=1) for members in panel.family_members])
    contrast = np.sum(normalized**2, axis=1, keepdims=True)
    names = (*(f"signed_recency:{name}" for name in panel.family_names), "phase_contrast_cost")
    return np.column_stack([family, contrast]), names


def _fit_signed_plus_nonnegative(
    design: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    l2: float,
    signed_count: int,
) -> ConstrainedHead:
    train_design = np.asarray(design[indices], dtype=float)
    train_target = np.asarray(target[indices], dtype=float)
    scale = np.sqrt(np.mean(train_design**2, axis=0))
    scale = np.maximum(scale, 1e-8)
    standardized = train_design / scale[None, :]
    if l2 > 0.0:
        standardized = np.vstack([standardized, np.sqrt(l2) * np.eye(standardized.shape[1])])
        train_target = np.concatenate([train_target, np.zeros(standardized.shape[1], dtype=float)])
    lower = np.concatenate([np.full(signed_count, -np.inf), np.zeros(design.shape[1] - signed_count)])
    upper = np.full(design.shape[1], np.inf)
    result = lsq_linear(standardized, train_target, bounds=(lower, upper), lsmr_tol="auto")
    if not result.success:
        raise RuntimeError(f"Constrained phase fit failed: {result.message}")
    coefficients = result.x / scale
    return ConstrainedHead(coefficients=np.asarray(coefficients, dtype=float), feature_scale=scale)


def fit_source_direction(
    panel: paired.PairedPanel,
    indices: np.ndarray,
    config: SourceConfig,
) -> SourceDirection:
    indices = np.asarray(indices, dtype=int)
    indices = indices[panel.paired_mask[indices]]
    delta = panel.two_phase_target - panel.one_phase_target
    design, _names = source_design(panel, panel.weights, config)
    head = _fit_signed_plus_nonnegative(
        design,
        delta,
        indices,
        config.l2,
        signed_count=len(panel.family_names),
    )
    direction = np.asarray(head.coefficients[: len(panel.family_names)], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-10:
        raise RuntimeError("Source recency direction collapsed to zero")
    direction /= norm
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction *= -1.0
    return SourceDirection(
        config=config,
        family_direction=direction,
        source_contrast_coefficient=float(head.coefficients[-1]),
        source_head=head,
    )


def target_design(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    source: SourceDirection,
    config: TargetConfig,
) -> np.ndarray:
    normalized = normalized_phase_displacement(panel, weights, source.config.denominator_offset)
    family = np.column_stack([normalized[:, members].sum(axis=1) for members in panel.family_members])
    signed_coordinate = family @ source.family_direction
    if not config.include_contrast_cost:
        return signed_coordinate[:, None]
    contrast = np.sum(normalized**2, axis=1)
    return np.column_stack([signed_coordinate, contrast])


def fit_target_phase(
    panel: paired.PairedPanel,
    indices: np.ndarray,
    source: SourceDirection,
    config: TargetConfig,
) -> TargetPhaseModel:
    indices = np.asarray(indices, dtype=int)
    indices = indices[panel.paired_mask[indices]]
    delta = panel.two_phase_target - panel.one_phase_target
    design = target_design(panel, panel.weights, source, config)
    head = _fit_signed_plus_nonnegative(design, delta, indices, config.l2, signed_count=1)
    contrast = float(head.coefficients[1]) if config.include_contrast_cost else 0.0
    return TargetPhaseModel(
        source=source,
        config=config,
        amplitude=float(head.coefficients[0]),
        contrast_coefficient=contrast,
        head=head,
    )
