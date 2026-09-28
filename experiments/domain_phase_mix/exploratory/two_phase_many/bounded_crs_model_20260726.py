# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compact retained state on a bounded log-deficit link.

The design block is compact retained state exactly: a shared Weibull response on
revisit-gated retained state, one amplitude per bucket, and one global literal
replay channel. Nothing is added to the response surface. The only change is how
the response is *parameterized* rather than what it can represent.

*Bounded link.* The head fits ``log(BPB - floor)`` and predicts
``floor + exp(eta)``, so the response is multiplicative in reducible loss and
cannot fall below ``floor``. Additive models can predict below any entropy floor,
which is the mechanism behind out-of-support optimism: the optimizer walks toward
a region the model says is arbitrarily good and the panel never contradicted it.
The bound is structural, not fitted. ``floor`` is ``DEFICIT_FLOOR_FRACTION`` times
the smallest observed target on the fitting panel, held fixed rather than chosen
by cross-validation, because cross-validation selects it against in-support fit.

*Panel-identified selection.* ``select_config`` cross-validates shape and ridge on
the fit panel, so every quantity the model depends on is estimated from the
supplied dataset. That is what makes its metrics comparable with the other
Observatory models, which select their own shapes the same way against
approximately the reported metric.

A shape pinned from another scale was tried and removed. It fit the panel worse,
and because four of its constants were not estimated from the panel it could not
be ranked against panel-CV models at all. The cross-scale selector disagreement it
was built to expose is a real result -- see
``reference_outputs/bounded_crs_shape_and_link_20260726/`` and
``audit_bounded_crs_shape_and_link_20260726.py`` -- but it is a claim about
selection protocol, not a surrogate, so it belongs in that report rather than in a
fit-quality ranking.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from fit_production_grp_quality_variants import Dataset
from scipy.optimize import nnls

SCREEN_SEED = 20260726
L2_GRID = (0.0, 0.01, 0.1, 1.0)
EPSILON = 1e-12

# Fraction of the smallest observed target used as the log-deficit floor. Fixed
# rather than cross-validated: CV selects it against in-support fit, and a floor
# tuned that way stops being a bound on out-of-support optimism.
DEFICIT_FLOOR_FRACTION = 0.95
# Numerical guard on the exponentiated link so a runaway linear predictor cannot
# produce an overflow instead of an obviously wrong prediction.
LINK_CLIP = 30.0


@dataclass(frozen=True)
class Shape:
    """Nonlinear response parameters, all shared across buckets."""

    rate: float
    power: float
    late_multiplier: float
    forgetting_rate: float


def shape_grid() -> Iterator[Shape]:
    """Shape candidates for panel cross-validation.

    Includes compact retained state's own ``power = 1.0``,
    ``late_multiplier = 4.0`` so that the baseline's configuration is reachable by
    search rather than only in principle.
    """
    for rate in (0.25, 1.0):
        for power in (0.4, 0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0, 4.0, 8.0):
                for forgetting_rate in (0.0, 0.25, 1.0):
                    yield Shape(
                        rate=rate,
                        power=power,
                        late_multiplier=late_multiplier,
                        forgetting_rate=forgetting_rate,
                    )


@dataclass(frozen=True)
class Config:
    shape: Shape
    l2: float
    deficit_floor_fraction: float = DEFICIT_FLOOR_FRACTION


@dataclass(frozen=True)
class Model:
    config: Config
    intercept: float
    coefficients: np.ndarray
    floor: float
    c0: np.ndarray
    c1: np.ndarray
    domains: tuple[str, ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(np.asarray(weights, dtype=float), self.c0, self.c1, self.config.shape)
        eta = self.intercept + design @ self.coefficients
        return np.asarray(self.floor + np.exp(np.clip(eta, -LINK_CLIP, LINK_CLIP)), dtype=float)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return feature_names(self.domains)


def feature_names(domains: tuple[str, ...]) -> tuple[str, ...]:
    return tuple([*(f"retained_benefit:{domain}" for domain in domains), "shared_literal_replay"])


def _weibull(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def retained_state(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray, shape: Shape) -> np.ndarray:
    """Revisit-gated retained state, identical to compact retained state's."""
    early = weights[:, 0, :] * c0[None, :]
    late = weights[:, 1, :] * c1[None, :]
    revisit = np.clip(weights[:, 1, :], 0.0, 1.0)
    retained = np.exp(-shape.forgetting_rate * (1.0 - revisit)) * early
    return np.maximum(retained + shape.late_multiplier * late, 0.0)


def design_matrix(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray, shape: Shape) -> np.ndarray:
    state = retained_state(weights, c0, c1, shape)
    total = weights[:, 0, :] * c0[None, :] + weights[:, 1, :] * c1[None, :]
    benefit = _weibull(state, shape.rate, shape.power)
    replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return np.hstack([-benefit, replay])


def link_floor(target: np.ndarray, deficit_floor_fraction: float) -> float:
    return float(deficit_floor_fraction) * float(np.min(target))


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
    floor = link_floor(target, config.deficit_floor_fraction)
    design = design_matrix(weights, c0, c1, config.shape)
    intercept, coefficients = fit_nonnegative_head(design, np.log(np.maximum(target - floor, 1e-9)), config.l2)
    return Model(
        config=config,
        intercept=intercept,
        coefficients=coefficients,
        floor=floor,
        c0=c0,
        c1=c1,
        domains=tuple(dataset.domains),
    )


def _oof_rmse(
    design: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    l2: float,
) -> float:
    """Out-of-fold RMSE in BPB after inverting the link, not in link space.

    Scoring on the link scale would optimize a different objective than the one
    the Observatory reports, and would make the ridge incomparable with the
    identity-link baselines.
    """
    errors = []
    for train, test in splits:
        floor = link_floor(target[train], DEFICIT_FLOOR_FRACTION)
        linked = np.log(np.maximum(target[train] - floor, 1e-9))
        intercept, coefficients = fit_nonnegative_head(design[train], linked, l2)
        eta = intercept + design[test] @ coefficients
        errors.append(floor + np.exp(np.clip(eta, -LINK_CLIP, LINK_CLIP)) - target[test])
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def select_config(
    dataset: Dataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    l2_grid: tuple[float, ...] = L2_GRID,
) -> tuple[Config, dict[str, Any]]:
    """Choose shape and ridge by out-of-fold RMSE on the fit panel alone.

    Everything the returned config depends on is estimated from the supplied
    dataset, which is what makes this arm comparable with the panel-CV baselines.
    """
    target = np.asarray(dataset.target, dtype=float)
    weights = np.asarray(dataset.weights, dtype=float)
    c0 = np.asarray(dataset.c0, dtype=float)
    c1 = np.asarray(dataset.c1, dtype=float)
    best: tuple[float, Config] | None = None
    evaluated = 0
    for shape in shape_grid():
        design = design_matrix(weights, c0, c1, shape)
        for l2 in l2_grid:
            score = _oof_rmse(design, target, splits, l2)
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
            "shapeSelector": "fit_panel_cv",
        }
    }
    return config, sweep
