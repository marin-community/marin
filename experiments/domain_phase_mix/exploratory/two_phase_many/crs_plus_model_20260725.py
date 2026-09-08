# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compact retained state plus family benefit and family-pooled overload.

This is the response law selected for Uncheatable mixture design on the 39-bucket
swarm. It contains compact retained state exactly: zeroing the family-benefit and
family-overload coefficients recovers the baseline's per-bucket benefit plus its
shared literal-replay channel, and the nonnegative head can select zero. Any gain
therefore has to come from the two added blocks earning their place.

Design blocks, in column order:

* per-bucket retained benefit, the baseline's own Weibull response on the
  revisit-gated retained state;
* family benefit, the complementarity channel that bucket-resolved family GRP and
  hierarchical phase replay carry and the baseline lacks;
* shared literal replay, the baseline's single global repetition scalar;
* family-pooled overload above a threshold in simulated epochs, which prices
  repetition per family rather than through one global scalar.

The saturation scale is expressed in epochs rather than as a bare rate because the
39-bucket panels oversample small corpora heavily: the proportional policy gives
every bucket 0.905 epochs, but the 99th percentile of the oversampling ratio is
about 117x and roughly 44 percent of (policy, bucket) cells exceed one epoch. A
saturation of 4 epochs reproduces the baseline's ``rate = 0.25``.

Measured against the best Observatory baseline on low-predicted-tail RMSE, the
accuracy over the 15 percent of policies a model ranks best, which is the quantity
that matters for proposing a mixture: 0.01440 against 0.01964 at 60M, 0.00583
against 0.00593 at 300M, and 0.00735 against 0.01422 at 3e18.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from fit_production_grp_quality_variants import Dataset
from scipy.optimize import nnls

SCREEN_SEED = 20260725
L2_GRID = (0.0, 0.01, 0.1, 1.0)
EPSILON = 1e-12


@dataclass(frozen=True)
class Shape:
    """Nonlinear response parameters, all shared across buckets."""

    saturation_epochs: float
    power: float
    late_multiplier: float
    forgetting_rate: float
    overload_threshold: float


@dataclass(frozen=True)
class Config:
    shape: Shape
    l2: float


@dataclass(frozen=True)
class Model:
    config: Config
    intercept: float
    coefficients: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    domains: tuple[str, ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(
            np.asarray(weights, dtype=float),
            self.c0,
            self.c1,
            self.config.shape,
            self.family_members,
        )
        return np.asarray(self.intercept + design @ self.coefficients, dtype=float)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return feature_names(self.domains, self.family_names)


def shape_grid() -> Iterator[Shape]:
    """Grid spanning the baseline's own selected configuration.

    ``saturation_epochs = 4`` reproduces the baseline's ``rate = 0.25`` and
    ``forgetting_rate`` includes 1.0, so the nesting is reachable by search rather
    than only in principle.
    """
    for saturation_epochs in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        for power in (0.4, 0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0, 4.0):
                for forgetting_rate in (0.0, 0.25, 1.0):
                    for overload_threshold in (1.0, 2.0, 4.0):
                        yield Shape(
                            saturation_epochs=saturation_epochs,
                            power=power,
                            late_multiplier=late_multiplier,
                            forgetting_rate=forgetting_rate,
                            overload_threshold=overload_threshold,
                        )


def feature_names(domains: tuple[str, ...], family_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        [
            *(f"retained_benefit:{domain}" for domain in domains),
            *(f"family_benefit:{family}" for family in family_names),
            "shared_literal_replay",
            *(f"family_overload:{family}" for family in family_names),
        ]
    )


def _weibull(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def _family_pool(values: np.ndarray, family_members: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.column_stack([values[:, members].sum(axis=1) for members in family_members])


def retained_state(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray, shape: Shape) -> np.ndarray:
    """Revisit-gated retained state, identical to the baseline's."""
    early = weights[:, 0, :] * c0[None, :]
    late = weights[:, 1, :] * c1[None, :]
    revisit = np.clip(weights[:, 1, :], 0.0, 1.0)
    retained = np.exp(-shape.forgetting_rate * (1.0 - revisit)) * early
    return np.maximum(retained + shape.late_multiplier * late, 0.0)


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    shape: Shape,
    family_members: tuple[np.ndarray, ...],
) -> np.ndarray:
    state = retained_state(weights, c0, c1, shape)
    rate = 1.0 / shape.saturation_epochs
    total = weights[:, 0, :] * c0[None, :] + weights[:, 1, :] * c1[None, :]
    family_state = _family_pool(state, family_members)
    # The family rate is divided by the family count so that a pooled sum enters
    # the same saturation regime as a single bucket rather than saturating at once.
    family_rate = rate / max(len(family_members), 1)
    literal_replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    overload = np.maximum(total - shape.overload_threshold, 0.0) ** 2
    return np.hstack(
        [
            -_weibull(state, rate, shape.power),
            -_weibull(family_state, family_rate, shape.power),
            literal_replay,
            _family_pool(overload, family_members),
        ]
    )


def fit_nonnegative_head(design: np.ndarray, target: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    """Column-scaled nonnegative least squares with a free intercept."""
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), EPSILON)
    scaled = design / scale
    design_mean = scaled.mean(axis=0, keepdims=True)
    target_mean = float(np.mean(target))
    centered_design = scaled - design_mean
    centered_target = target - target_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    coefficients, _ = nnls(centered_design, centered_target, maxiter=80 * design.shape[1])
    coefficients = coefficients / scale
    intercept = target_mean - float((design.mean(axis=0, keepdims=True) @ coefficients).item())
    return intercept, np.asarray(coefficients, dtype=float)


def fit_model(dataset: Dataset, config: Config, indices: np.ndarray) -> Model:
    weights = np.asarray(dataset.weights, dtype=float)[indices]
    target = np.asarray(dataset.target, dtype=float)[indices]
    c0 = np.asarray(dataset.c0, dtype=float)
    c1 = np.asarray(dataset.c1, dtype=float)
    design = design_matrix(weights, c0, c1, config.shape, dataset.family_members)
    intercept, coefficients = fit_nonnegative_head(design, target, config.l2)
    return Model(
        config=config,
        intercept=intercept,
        coefficients=coefficients,
        c0=c0,
        c1=c1,
        family_names=tuple(dataset.family_names),
        family_members=tuple(dataset.family_members),
        domains=tuple(dataset.domains),
    )


def select_config(
    dataset: Dataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    l2_grid: tuple[float, ...] = L2_GRID,
) -> tuple[Config, dict[str, Any]]:
    """Choose shape and ridge by out-of-fold RMSE on the supplied folds."""
    target = np.asarray(dataset.target, dtype=float)
    best: tuple[float, Config] | None = None
    evaluated = 0
    for shape in shape_grid():
        design = design_matrix(
            np.asarray(dataset.weights, dtype=float),
            np.asarray(dataset.c0, dtype=float),
            np.asarray(dataset.c1, dtype=float),
            shape,
            dataset.family_members,
        )
        for l2 in l2_grid:
            errors = []
            for train, test in splits:
                intercept, coefficients = fit_nonnegative_head(design[train], target[train], l2)
                errors.append(intercept + design[test] @ coefficients - target[test])
            score = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
            evaluated += 1
            if best is None or score < best[0]:
                best = (score, Config(shape=shape, l2=l2))
    assert best is not None, "empty shape grid"
    score, config = best
    sweep = {
        "shapeSweep": {
            "evaluated": evaluated,
            "selectedOofRmse": score,
            "screenSeed": SCREEN_SEED,
        }
    }
    return config, sweep
