# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Reproduce and audit the 300M saturation-knee promotion claim.

The original SUR-111 scratch calculation selected nonlinear parameters by
grouped CV, refit on every row, and reported the refit's residual RMSE against
an OOF HPR reference. This script preserves that exact model and optimization
problem, but additionally evaluates it with an outer grouped split. It reports
both quantities so the optimistic in-sample number cannot be mistaken for an
OOF comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, nnls

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_aggregate_conditioned_replay_control_20260730 as panel,
)

N_FOLDS = 3
N_INNER_FOLDS = 3
OPTIMIZER_SEED = 20260808

# near horizon, three family exponents, log offset, damage exponent,
# damage horizon, log saturation knee in excess epochs, log ridge.
BOUNDS = (
    (0.0, 1.0),
    (0.005, 1.5),
    (0.005, 1.5),
    (0.005, 1.5),
    (-5.0, -0.3),
    (0.2, 10.0),
    (0.0, 1.0),
    (-1.0, 3.0),
    (-6.0, 1.0),
)


@dataclass(frozen=True)
class Fit:
    shape: np.ndarray
    free_amplitudes: np.ndarray
    constrained_amplitudes: np.ndarray


def family_means(values: np.ndarray, family_index: np.ndarray) -> np.ndarray:
    """Average bucket-level features within each predeclared family."""
    return np.column_stack([values[:, family_index == family].mean(axis=1) for family in np.unique(family_index)])


def design(weights: np.ndarray, data, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact SUR-111 scratch design."""
    near, gamma_0, gamma_1, gamma_2, log_offset, tau, damage_horizon, log_knee, _ = shape
    offset = 10.0**log_offset
    knee = 10.0**log_knee
    exponents = np.array([gamma_0, gamma_1, gamma_2])[data.family_index]
    total_epochs = data.c0 + data.c1

    def exposure(horizon: float) -> np.ndarray:
        return total_epochs * ((1.0 - horizon) * weights[:, 0, :] + horizon * weights[:, 1, :])

    excess = np.maximum(exposure(damage_horizon) - 1.0, 0.0)
    damage = excess**tau / (1.0 + (excess / knee) ** tau)
    near_benefit = (exposure(near) + offset) ** -exponents
    constrained = np.column_stack(
        [
            family_means(near_benefit, data.family_index),
            family_means((exposure(1.0) + offset) ** -exponents, data.family_index),
            family_means(damage, data.family_index),
            near_benefit,
        ]
    )
    return np.ones((len(weights), 1)), constrained


def fit_head(
    free: np.ndarray,
    constrained: np.ndarray,
    response: np.ndarray,
    ridge: float,
    pooled: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the partitioned free-sign and nonnegative linear head."""
    basis, _ = np.linalg.qr(free)
    columns = constrained - basis @ (basis.T @ constrained)
    target = response - basis @ (basis.T @ response)
    scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
    scaled = columns / scale
    if ridge > 0:
        strength = np.sqrt(ridge) * np.concatenate([np.full(pooled, 1e-3), np.ones(scaled.shape[1] - pooled)])
        scaled = np.vstack([scaled, np.diag(strength)])
        target = np.concatenate([target, np.zeros(scaled.shape[1])])
    amplitudes, _ = nnls(scaled, target, maxiter=20000)
    amplitudes = amplitudes / scale
    free_amplitudes = np.linalg.lstsq(free, response - constrained @ amplitudes, rcond=None)[0]
    return free_amplitudes, amplitudes


def select(data, rows: np.ndarray, seed: int) -> np.ndarray:
    """Select nonlinear parameters using only the supplied rows."""
    frame = data.frame.iloc[rows].reset_index(drop=True)
    folds = panel.grouped_folds(frame, seed, N_INNER_FOLDS)
    weights = data.weights[rows]
    response = data.y[rows]
    pooled = 3 * len(np.unique(data.family_index))

    def objective(shape: np.ndarray) -> float:
        free, constrained = design(weights, data, shape)
        if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
            return 1e6
        total = 0.0
        for train, test in folds:
            b, a = fit_head(free[train], constrained[train], response[train], 10.0 ** shape[8], pooled)
            residual = free[test] @ b + constrained[test] @ a - response[test]
            total += float(residual @ residual)
        return total

    return differential_evolution(
        objective,
        BOUNDS,
        rng=np.random.default_rng(OPTIMIZER_SEED),
        popsize=12,
        maxiter=80,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def fit(data, rows: np.ndarray, seed: int) -> Fit:
    """Select a shape and refit its head on the supplied rows."""
    shape = select(data, rows, seed)
    free, constrained = design(data.weights[rows], data, shape)
    pooled = 3 * len(np.unique(data.family_index))
    b, a = fit_head(free, constrained, data.y[rows], 10.0 ** shape[8], pooled)
    return Fit(shape=shape, free_amplitudes=b, constrained_amplitudes=a)


def predict(data, fitted: Fit, rows: np.ndarray) -> np.ndarray:
    """Predict the requested rows."""
    free, constrained = design(data.weights[rows], data, fitted.shape)
    return free @ fitted.free_amplitudes + constrained @ fitted.constrained_amplitudes


def pair_metrics(data, predictions: np.ndarray) -> tuple[float, float, float, float, int]:
    """Return predicted and observed aggregate-matched phase effects."""
    tied = np.all(np.isclose(data.weights[:, 0, :], data.weights[:, 1, :]), axis=1)
    keys = data.frame["phase_correspondence_key"].astype(str).to_numpy()
    predicted_gaps: list[float] = []
    observed_gaps: list[float] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        moved = rows[~tied[rows]]
        held = rows[tied[rows]]
        if len(moved) and len(held):
            predicted_gaps.append(float(predictions[held].min() - predictions[moved].min()))
            observed_gaps.append(float(data.y[held].min() - data.y[moved].min()))
    return (
        float(np.mean(predicted_gaps)),
        float(np.mean(observed_gaps)),
        float(np.max(predicted_gaps)),
        float(np.max(observed_gaps)),
        len(predicted_gaps),
    )


def evaluate(data, seed: int) -> dict[str, object]:
    """Compute honest outer-OOF metrics and the original in-sample diagnostic."""
    outer = panel.grouped_folds(data.frame, seed, N_FOLDS)
    predictions = np.empty_like(data.y)
    fold_knees: list[float] = []
    fold_taus: list[float] = []
    fold_damage_horizons: list[float] = []
    for train, test in outer:
        fitted = fit(data, train, seed)
        predictions[test] = predict(data, fitted, test)
        fold_knees.append(float(10.0 ** fitted.shape[7]))
        fold_taus.append(float(fitted.shape[5]))
        fold_damage_horizons.append(float(fitted.shape[6]))

    full = fit(data, np.arange(data.n), seed)
    fitted_values = predict(data, full, np.arange(data.n))
    selected = int(np.argmin(predictions))
    observed_best = int(np.argmin(data.y))
    predicted_mean, observed_mean, predicted_max, observed_max, n_pairs = pair_metrics(data, predictions)
    return {
        "seed": seed,
        "oof_rmse": float(np.sqrt(np.mean((predictions - data.y) ** 2))),
        "in_sample_rmse": float(np.sqrt(np.mean((fitted_values - data.y) ** 2))),
        "regret_at_1": float(data.y[selected] - data.y[observed_best]),
        "full_knee": float(10.0 ** full.shape[7]),
        "full_tau": float(full.shape[5]),
        "full_damage_horizon": float(full.shape[6]),
        "fold_knees": fold_knees,
        "fold_taus": fold_taus,
        "fold_damage_horizons": fold_damage_horizons,
        "predicted_pair_gain": predicted_mean,
        "observed_pair_gain": observed_mean,
        "predicted_best_pair_gain": predicted_max,
        "observed_best_pair_gain": observed_max,
        "n_pairs": n_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("uncheatable", "table9"))
    parser.add_argument("seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    data = panel.load_300m(args.target)
    print(f"SUR-111 audit: target={args.target}, rows={data.n}, outer={N_FOLDS}, inner={N_INNER_FOLDS}")
    for seed in args.seeds:
        row = evaluate(data, seed)
        knees = ",".join(f"{value:.2f}" for value in row["fold_knees"])
        taus = ",".join(f"{value:.3f}" for value in row["fold_taus"])
        horizons = ",".join(f"{value:.3f}" for value in row["fold_damage_horizons"])
        print(
            f"seed {seed}: OOF RMSE {row['oof_rmse']:.6f}; in-sample RMSE {row['in_sample_rmse']:.6f}; "
            f"Regret@1 {row['regret_at_1']:.6f}; full E* {row['full_knee']:.2f}; full tau {row['full_tau']:.3f}; "
            f"full damage horizon {row['full_damage_horizon']:.3f}"
        )
        print(f"         fold E* [{knees}]; fold tau [{taus}]; fold damage horizon [{horizons}]")
        print(
            f"         pair gain n={row['n_pairs']}: predicted mean {row['predicted_pair_gain']:+.6f}; "
            f"observed mean {row['observed_pair_gain']:+.6f}; predicted max {row['predicted_best_pair_gain']:+.6f}; "
            f"observed max {row['observed_best_pair_gain']:+.6f}"
        )


if __name__ == "__main__":
    main()
