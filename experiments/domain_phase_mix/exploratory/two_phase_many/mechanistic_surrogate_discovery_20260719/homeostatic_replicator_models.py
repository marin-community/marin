# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Simplex-constrained representation capacity with homeostatic recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import nnls

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)

NUMERICAL_FLOOR = 1e-12
INTEGRATION_STEPS_PER_UNIT = 192
BENEFIT_OFFSET = 0.1


@dataclass(frozen=True)
class Config:
    selection_rate: float
    homeostasis_rate: float
    replay_log_onset: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"selection={self.selection_rate:g},homeostasis={self.homeostasis_rate:g},"
            f"replay={self.replay_log_onset:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class Model:
    panel: paired.PairedPanel
    config: Config
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _names = build_design(self.panel, weights, self.config)
        return np.asarray(self.intercept + design @ self.coefficients, dtype=float)


def _derivative(
    state: np.ndarray,
    relative_rate: np.ndarray,
    proportional: np.ndarray,
    selection_rate: float,
    homeostasis_rate: float,
) -> np.ndarray:
    mean_rate = np.sum(state * relative_rate, axis=1, keepdims=True)
    selection = selection_rate * state * (relative_rate - mean_rate)
    homeostasis = homeostasis_rate * (proportional[None, :] - state)
    return selection + homeostasis


def _integrate_phase(
    state: np.ndarray,
    weights: np.ndarray,
    duration: float,
    proportional: np.ndarray,
    selection_rate: float,
    homeostasis_rate: float,
) -> np.ndarray:
    if duration <= 0.0:
        return state
    steps = max(1, int(np.ceil(INTEGRATION_STEPS_PER_UNIT * duration)))
    step = duration / steps
    relative_rate = weights / np.maximum(proportional[None, :], NUMERICAL_FLOOR)
    current = np.asarray(state, dtype=float).copy()
    for _ in range(steps):
        k1 = _derivative(current, relative_rate, proportional, selection_rate, homeostasis_rate)
        k2 = _derivative(current + 0.5 * step * k1, relative_rate, proportional, selection_rate, homeostasis_rate)
        k3 = _derivative(current + 0.5 * step * k2, relative_rate, proportional, selection_rate, homeostasis_rate)
        k4 = _derivative(current + step * k3, relative_rate, proportional, selection_rate, homeostasis_rate)
        current += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        current = np.maximum(current, NUMERICAL_FLOOR)
        current /= current.sum(axis=1, keepdims=True)
    return current


def terminal_capacity(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    selection_rate: float,
    homeostasis_rate: float,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    proportional = panel.proportional_weights
    state = np.broadcast_to(proportional, (len(weights), panel.m)).copy()
    state = _integrate_phase(
        state,
        weights[:, 0, :],
        panel.alpha0,
        proportional,
        selection_rate,
        homeostasis_rate,
    )
    return _integrate_phase(
        state,
        weights[:, 1, :],
        panel.alpha1,
        proportional,
        selection_rate,
        homeostasis_rate,
    )


def build_design(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, tuple[str, ...]]:
    weights = np.asarray(weights, dtype=float)
    capacity = terminal_capacity(
        panel,
        weights,
        config.selection_rate,
        config.homeostasis_rate,
    )
    exposure = weights[:, 0, :] * panel.c0[None, :] + weights[:, 1, :] * panel.c1[None, :]
    unique_coverage = -np.expm1(-np.maximum(exposure, 0.0))
    capacity_ratio = capacity / np.maximum(panel.proportional_weights[None, :], NUMERICAL_FLOOR)
    effective = unique_coverage * capacity_ratio

    proportional_exposure = panel.proportional_exposure
    proportional_unique = -np.expm1(-np.maximum(proportional_exposure, 0.0))
    benefit = -np.log1p(effective / np.maximum(BENEFIT_OFFSET * proportional_unique[None, :], NUMERICAL_FLOOR))
    replay_delta = np.log1p(np.maximum(exposure, 0.0)) - config.replay_log_onset
    replay = np.logaddexp(0.0, replay_delta) ** 2
    design = np.column_stack([benefit, replay])
    names = tuple(f"capacity_weighted_benefit:{name}" for name in panel.domain_names) + tuple(
        f"physical_replay:{name}" for name in panel.domain_names
    )
    return design, names


def fit_model(
    panel: paired.PairedPanel,
    indices: np.ndarray,
    config: Config,
) -> Model:
    design, _names = build_design(panel, panel.weights, config)
    train_design = design[indices]
    train_target = panel.two_phase_target[indices]
    design_mean = train_design.mean(axis=0, keepdims=True)
    target_mean = float(train_target.mean())
    centered_design = train_design - design_mean
    centered_target = train_target - target_mean
    if config.l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(config.l2) * np.eye(centered_design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(centered_design.shape[1], dtype=float)])
    coefficients, _residual = nnls(
        centered_design,
        centered_target,
        maxiter=50 * centered_design.shape[1],
    )
    intercept = target_mean - float((design_mean @ coefficients).item())
    return Model(panel=panel, config=config, intercept=intercept, coefficients=coefficients)


def tied_semigroup_error(panel: paired.PairedPanel, config: Config) -> float:
    tied = np.stack([panel.aggregate_weights, panel.aggregate_weights], axis=1)
    capacity_two = terminal_capacity(
        panel,
        tied,
        config.selection_rate,
        config.homeostasis_rate,
    )
    one_phase_panel = replace(panel, c0=panel.c0 + panel.c1, c1=np.zeros_like(panel.c1))
    one_phase_weights = np.stack([panel.aggregate_weights, panel.aggregate_weights], axis=1)
    capacity_one = terminal_capacity(
        one_phase_panel,
        one_phase_weights,
        config.selection_rate,
        config.homeostasis_rate,
    )
    return float(np.max(np.abs(capacity_two - capacity_one)))
