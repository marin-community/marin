# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Evaluate an unstructured local-smoother null on the WSD80 panel.

Tensor cubic B-spline basis size and ridge are selected inside every outer
training fold. The model is a representability null, not a deployable
surrogate or evidence for a training mechanism.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import multitarget_ile_wsd80_20260806 as wsd

RPL_INTERIOR_RMSE = 0.007575
RPL_REGRET_LIMIT = 0.004842
OPTIMUM_DISTANCE_LIMIT = 0.05
GAIN_ERROR_LIMIT = harness.WSD_GAIN_ERROR_LIMIT
BASIS_COUNTS = (4, 5, 6)
RIDGES = (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
DEGREE = 3
N_FOLDS = 3
N_INNER_FOLDS = 5
SURFACE_GRID = 801
CODE = 1

PANEL, TARGETS = wsd.load_targets()
INTERIOR = wsd.interior_mask(PANEL)
PRIMARY = TARGETS.values[:, TARGETS.names.index(harness.PRIMARY_TARGET)]


@dataclass(frozen=True)
class SplineFit:
    basis_count: int
    ridge: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    response_mean: float
    coefficients: np.ndarray


def coordinates(weights: np.ndarray) -> np.ndarray:
    """Return phase-0 and phase-1 code shares."""
    return weights[:, :, CODE]


def knots(basis_count: int) -> np.ndarray:
    """Return a clamped uniform knot vector on the unit interval."""
    internal_count = basis_count - DEGREE - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    return np.concatenate([np.zeros(DEGREE + 1), internal, np.ones(DEGREE + 1)])


def one_dimensional_basis(values: np.ndarray, basis_count: int) -> np.ndarray:
    """Evaluate a cubic B-spline basis at values in the unit interval."""
    design = BSpline.design_matrix(values, knots(basis_count), DEGREE, extrapolate=False)
    return np.asarray(design.toarray())


def spline_features(points: np.ndarray, basis_count: int) -> np.ndarray:
    """Return tensor products of phase-specific cubic B-spline bases."""
    phase_0 = one_dimensional_basis(points[:, 0], basis_count)
    phase_1 = one_dimensional_basis(points[:, 1], basis_count)
    return np.einsum("ni,nj->nij", phase_0, phase_1).reshape(len(points), -1)


def fit(points: np.ndarray, response: np.ndarray, basis_count: int, ridge: float) -> SplineFit:
    """Fit a standardized tensor spline with an unpenalized response mean."""
    features = spline_features(points, basis_count)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    standardized = (features - feature_mean) / feature_scale
    response_mean = float(response.mean())
    centered = response - response_mean
    design = standardized
    target = centered
    if ridge > 0.0:
        design = np.vstack([design, np.sqrt(ridge) * np.eye(design.shape[1])])
        target = np.concatenate([target, np.zeros(design.shape[1])])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return SplineFit(basis_count, ridge, feature_mean, feature_scale, response_mean, coefficients)


def predict(fitted: SplineFit, points: np.ndarray) -> np.ndarray:
    """Predict at phase-share coordinates."""
    features = spline_features(points, fitted.basis_count)
    standardized = (features - fitted.feature_mean) / fitted.feature_scale
    return fitted.response_mean + standardized @ fitted.coefficients


def folds(points: np.ndarray, count: int, seed: int, fold_mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the campaign's random or mixture-blocked folds."""
    weights = np.empty((len(points), 2, 2))
    weights[:, :, CODE] = points
    weights[:, :, 0] = 1.0 - points
    return harness.wsd80_folds(fold_mode, weights, np.arange(len(points)), count, seed)


def select(
    points: np.ndarray, response: np.ndarray, interior: np.ndarray, seed: int, fold_mode: str
) -> tuple[int, float]:
    """Select basis size and ridge using only nested held-fold residuals."""
    inner = folds(points, N_INNER_FOLDS, seed, fold_mode)
    best: tuple[float, int, float] | None = None
    for basis_count in BASIS_COUNTS:
        for ridge in RIDGES:
            loss = 0.0
            for train, test in inner:
                fitted = fit(points[train], response[train], basis_count, ridge)
                scored = test[interior[test]]
                if len(scored):
                    residual = predict(fitted, points[scored]) - response[scored]
                    loss += float(residual @ residual)
            candidate = (loss, basis_count, -ridge)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], -best[2]


def surface_diagnostics(fitted: SplineFit) -> tuple[tuple[float, float], float, float]:
    """Return descriptive full-fit optimum, distance, and predicted gain."""
    axis = np.linspace(0.0, 1.0, SURFACE_GRID)
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    grid = np.column_stack([phase_0.ravel(), phase_1.ravel()])
    values = predict(fitted, grid)
    best = int(np.argmin(values))
    optimum = (float(grid[best, 0]), float(grid[best, 1]))

    observed_best = int(np.argmin(PRIMARY))
    observed = (float(PANEL.phase_0[observed_best, CODE]), float(PANEL.phase_1[observed_best, CODE]))
    distance = float(np.linalg.norm(np.asarray(optimum) - np.asarray(observed)))
    tied = np.column_stack([axis, axis])
    gain = float(np.min(predict(fitted, tied)) - values[best])
    return optimum, distance, gain


def evaluate(seed: int, fold_mode: str) -> dict[str, object]:
    """Run nested OOF evaluation and descriptive full-fit diagnostics."""
    points = coordinates(PANEL.weights)
    predictions = np.empty_like(PRIMARY)
    selected: list[tuple[int, float]] = []
    for train, test in folds(points, N_FOLDS, seed, fold_mode):
        basis_count, ridge = select(points[train], PRIMARY[train], INTERIOR[train], seed, fold_mode)
        selected.append((basis_count, ridge))
        fitted = fit(points[train], PRIMARY[train], basis_count, ridge)
        predictions[test] = predict(fitted, points[test])

    basis_count, ridge = select(points, PRIMARY, INTERIOR, seed, fold_mode)
    full = fit(points, PRIMARY, basis_count, ridge)
    optimum, distance, gain = surface_diagnostics(full)
    observed_best = int(np.argmin(PRIMARY))
    predicted_best = int(np.argmin(predictions))
    residual = predictions[INTERIOR] - PRIMARY[INTERIOR]
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "regret_at_1": float(PRIMARY[predicted_best] - PRIMARY[observed_best]),
        "optimum": optimum,
        "distance": distance,
        "gain": gain,
        "gain_error": abs(gain - harness.OBSERVED_WSD_GAIN),
        "outer_configs": selected,
        "full_config": (basis_count, ridge),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--fold-mode", choices=("random", "blocked"), default="random")
    args = parser.parse_args()

    print(f"Tensor spline null: bases={BASIS_COUNTS}, ridges={RIDGES}, fold_mode={args.fold_mode}")
    for seed in args.seeds:
        result = evaluate(seed, args.fold_mode)
        passes = (
            result["rmse"] <= 1.05 * RPL_INTERIOR_RMSE,
            result["regret_at_1"] <= RPL_REGRET_LIMIT,
            result["distance"] <= OPTIMUM_DISTANCE_LIMIT,
            result["gain_error"] <= GAIN_ERROR_LIMIT,
        )
        print(
            f"seed {seed}: RMSE {result['rmse']:.6f}{'P' if passes[0] else 'F'}; "
            f"Regret@1 {result['regret_at_1']:.6f}{'P' if passes[1] else 'F'}; "
            f"distance {result['distance']:.6f}{'P' if passes[2] else 'F'}; "
            f"gain error {result['gain_error']:.6f}{'P' if passes[3] else 'F'}; "
            f"optimum ({result['optimum'][0]:.3f},{result['optimum'][1]:.3f}); "
            f"gain {result['gain']:+.6f}; total {sum(passes)}/4"
        )
        print(f"         outer configs: {result['outer_configs']}; full config: {result['full_config']}")


if __name__ == "__main__":
    main()
