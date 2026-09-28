# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""WSD80-SUR-102 carried to the 39-bucket 300M panel: the sharpest transfer test available.

This panel is the mirror image of WSD80. There, a two-phase policy beats every constant mixture by
0.009594 BPB and a surrogate must predict a real gain. Here, NONE of the 238 asymmetric policies beats
the best tied policy on either target, and the panel supplies an exact aggregate-matched tied counterpart
for every one of them. So the correct answer is a near-zero predicted advantage, and a large predicted
gain is a failure. A form that produces the right answer on both panels is doing something other than
fitting whatever it is shown.

The port is family-pooled rather than per-bucket. Thirty-nine buckets fall into three predeclared
families, and every nonlinear parameter is shared across buckets: one pair of horizons, one readout
exponent per family, one gate scale and sharpness, one damage horizon and exponent. Per-bucket freedom
was tried in this project before and buys conditioning trouble rather than resolution.

The signed conflict channel is NOT ported. On WSD80 it reads the decay-phase share of the single
off-domain family against a code eval; with thirty-nine buckets and no single eval domain there is no
faithful analogue, and inventing one would make the comparison between panels incomparable. Its absence
is the one structural difference from the WSD80 form and is reported as such.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, nnls  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as panel,
)

# Published HPR reference per target, from the frozen harness constants. Each target must be compared
# against its OWN reference; an earlier stage in this project was withdrawn partly for comparing a
# reconstructed macro against the reference for a different column.
HPR = {
    "uncheatable": {"all_rmse": 0.006800, "regret_at_1": 0.002678},
    "table9": {"all_rmse": 0.013001, "regret_at_1": 0.003304},
}
RMSE_SLACK = 1.05
REGRET_SLACK = 0.002
N_FOLDS = 3
N_INNER_FOLDS = 3

# phi_near, gamma_f0, gamma_f1, gamma_f2, log_offset, tau, phi_damage, log_kappa, beta, log_ridge
BOUNDS = (
    (0.0, 1.0),
    (0.005, 1.5),
    (0.005, 1.5),
    (0.005, 1.5),
    (-5.0, -0.3),
    (0.2, 10.0),
    (0.0, 1.0),
    (-2.0, 2.5),
    (0.3, 20.0),
    (-6.0, 1.0),
)


def family_sums(values: np.ndarray, family_index: np.ndarray) -> np.ndarray:
    """Mean over the buckets of each predeclared family, the project's standard pooling."""
    return np.column_stack([values[:, family_index == f].mean(axis=1) for f in np.unique(family_index)])


def design(weights: np.ndarray, data, shape) -> tuple[np.ndarray, np.ndarray]:
    near, g0, g1, g2, log_offset, tau, damage_horizon, log_kappa, beta, _ = shape
    offset, kappa = 10.0**log_offset, 10.0**log_kappa
    exponents = np.array([g0, g1, g2])[data.family_index]
    epochs = data.c0 + data.c1

    def exposure(horizon: float) -> np.ndarray:
        return epochs * ((1.0 - horizon) * weights[:, 0, :] + horizon * weights[:, 1, :])

    early = data.c0 * weights[:, 0, :]
    gate = early**beta / (early**beta + kappa**beta)
    absorbed = data.c0 * weights[:, 0, :] + data.c1 * weights[:, 1, :] * gate

    near_benefit = (exposure(near) + offset) ** -exponents
    blocks = [
        family_sums(near_benefit, data.family_index),
        family_sums((exposure(1.0) + offset) ** -exponents, data.family_index),
        family_sums((absorbed + offset) ** -exponents, data.family_index),
        family_sums(np.maximum(exposure(damage_horizon) - 1.0, 0.0) ** tau, data.family_index),
        # Per-bucket departures from the family level, on the main benefit block only. This is the
        # project's standard hierarchical pooling: three family means alone give the incumbent far more
        # per-bucket resolution than this port, and per-bucket freedom on EVERY block was already found
        # to buy conditioning trouble rather than resolution.
        near_benefit,
    ]
    return np.ones((len(weights), 1)), np.column_stack(blocks)


def n_pooled(data) -> int:
    """Columns that carry the pooled signal; everything after them is a shrunk departure."""
    return 4 * len(np.unique(data.family_index))


def fit_head(free: np.ndarray, constrained: np.ndarray, response: np.ndarray, ridge: float, pooled: int):
    """Pooled levels barely shrunk, per-bucket departures shrunk hard, intercept never shrunk."""
    basis, _ = np.linalg.qr(free)
    columns = constrained - basis @ (basis.T @ constrained)
    target = response - basis @ (basis.T @ response)
    scale = np.maximum(np.linalg.norm(columns, axis=0), 1e-300)
    scaled = columns / scale
    if ridge > 0:
        strength = np.sqrt(ridge) * np.concatenate([np.full(pooled, 1e-3), np.ones(scaled.shape[1] - pooled)])
        scaled = np.vstack([scaled, np.diag(strength)])
        target = np.concatenate([target, np.zeros(scaled.shape[1])])
    amplitudes, _ = nnls(scaled, target)
    amplitudes = amplitudes / scale
    return np.linalg.lstsq(free, response - constrained @ amplitudes, rcond=None)[0], amplitudes


def select(data, rows: np.ndarray, seed: int):
    inner = panel.grouped_folds(data.frame.iloc[rows].reset_index(drop=True), seed, N_INNER_FOLDS)
    weights, response = data.weights[rows], data.y[rows]

    def inner_error(shape) -> float:
        free, constrained = design(weights, data, shape)
        if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
            return 1e3
        total = 0.0
        for train, test in inner:
            b, a = fit_head(free[train], constrained[train], response[train], 10.0 ** shape[9], n_pooled(data))
            residual = free[test] @ b + constrained[test] @ a - response[test]
            total += float(residual @ residual)
        return total

    return differential_evolution(
        inner_error,
        BOUNDS,
        rng=np.random.default_rng(20260808),
        popsize=12,
        maxiter=80,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def evaluate(data, seed: int) -> dict:
    outer = panel.grouped_folds(data.frame, seed, N_FOLDS)
    predictions = np.empty_like(data.y)
    for train, test in outer:
        shape = select(data, train, seed)
        free, constrained = design(data.weights, data, shape)
        b, a = fit_head(free[train], constrained[train], data.y[train], 10.0 ** shape[9], n_pooled(data))
        predictions[test] = free[test] @ b + constrained[test] @ a

    tied = np.all(np.isclose(data.weights[:, 0, :], data.weights[:, 1, :]), axis=1)
    best_observed = int(np.argmin(data.y))
    selected = int(np.argmin(predictions))

    # The panel's own pairing: every asymmetric policy has an exact aggregate-matched tied counterpart in
    # its correspondence group. The advantage the model claims for going two-phase is the mean predicted
    # improvement over that counterpart, and on this panel the truth is that it should be at or below zero.
    keys = data.frame["phase_correspondence_key"].astype(str).to_numpy()
    predicted_gaps, observed_gaps = [], []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        moved, held = rows[~tied[rows]], rows[tied[rows]]
        if len(moved) and len(held):
            predicted_gaps.append(predictions[held].min() - predictions[moved].min())
            observed_gaps.append(data.y[held].min() - data.y[moved].min())

    return {
        "seed": seed,
        "rmse": float(np.sqrt(np.mean((predictions - data.y) ** 2))),
        "regret_at_1": float(data.y[selected] - data.y[best_observed]),
        "predicted_pair_gain": float(np.mean(predicted_gaps)),
        "observed_pair_gain": float(np.mean(observed_gaps)),
        "predicted_best_pair_gain": float(np.max(predicted_gaps)),
        "observed_best_pair_gain": float(np.max(observed_gaps)),
        "n_pairs": len(predicted_gaps),
    }


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "uncheatable"
    data = panel.load_300m(target)
    print(
        f"300M {target}: {data.n} rows, {len(data.domain_names)} buckets, "
        f"{len(np.unique(data.family_index))} families, grouped {N_FOLDS}x{N_INNER_FOLDS}"
    )
    reference = HPR[target]
    print(f"HPR reference for {target}: all-row RMSE {reference['all_rmse']}, Regret@1 {reference['regret_at_1']}")
    print("the panel's truth is that NO asymmetric policy beats the best tied one, so predicted pair gain")
    print("should sit at or below zero; a large positive value is a failure\n")
    for seed in (0, 1, 2):
        row = evaluate(data, seed)
        rmse_ok = row["rmse"] <= reference["all_rmse"] * RMSE_SLACK
        regret_ok = row["regret_at_1"] <= reference["regret_at_1"] + REGRET_SLACK
        print(
            f"  seed {seed}: all-row RMSE {row['rmse']:.6f}{'P' if rmse_ok else 'F'}"
            f"  Regret@1 {row['regret_at_1']:.6f}{'P' if regret_ok else 'F'}"
        )
        print(
            f"          pair gain over {row['n_pairs']} matched pairs: predicted mean"
            f" {row['predicted_pair_gain']:+.6f}  observed mean {row['observed_pair_gain']:+.6f}"
        )
        print(
            f"          worst case: predicted max {row['predicted_best_pair_gain']:+.6f}"
            f"  observed max {row['observed_best_pair_gain']:+.6f}"
        )


if __name__ == "__main__":
    main()
