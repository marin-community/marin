# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Estimate a flexible two-dimensional prediction floor for StarCoder BPB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (  # noqa: E402
    load_completed_two_phase_starcoder_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (  # noqa: E402
    STARCODER_TARGET,
)

OUTPUT_DIR = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_surface_upper_bound_20260710"
)
SEEDS = (0, 1, 2)
N_SPLITS = 5


def models(seed: int) -> dict[str, object]:
    """Return flexible two-dimensional regressors with fixed complexity."""
    return {
        "polynomial_degree_2": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-3),
        ),
        "polynomial_degree_3": make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-3),
        ),
        "polynomial_degree_4": make_pipeline(
            PolynomialFeatures(degree=4, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1e-2),
        ),
        "gaussian_process_rbf": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=(
                    ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(2), length_scale_bounds=(5e-2, 20.0))
                    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
                ),
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=seed,
            ),
        ),
        "gaussian_process_matern": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=(
                    ConstantKernel(1.0, (1e-2, 1e2))
                    * Matern(
                        length_scale=np.ones(2),
                        length_scale_bounds=(5e-2, 20.0),
                        nu=1.5,
                    )
                    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1.0))
                ),
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=seed,
            ),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packet = load_completed_two_phase_starcoder_packet()
    x = packet.frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    y = packet.frame[STARCODER_TARGET].to_numpy(dtype=float)
    metric_rows = []
    prediction_rows = []
    for seed in SEEDS:
        folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for name, model in models(seed).items():
            prediction = np.zeros(len(y), dtype=float)
            for train_indices, test_indices in folds.split(x):
                model.fit(x[train_indices], y[train_indices])
                prediction[test_indices] = model.predict(x[test_indices])
            residual = prediction - y
            metric_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "oof_rmse": float(np.sqrt(np.mean(residual**2))),
                    "oof_spearman": float(spearmanr(y, prediction).statistic),
                    "low_tail_rmse": float(np.sqrt(np.mean(residual[y <= np.quantile(y, 0.2)] ** 2))),
                }
            )
            for row, value in enumerate(prediction):
                prediction_rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "row": row,
                        "phase_0_starcoder": x[row, 0],
                        "phase_1_starcoder": x[row, 1],
                        "observed_bpb": y[row],
                        "oof_prediction": value,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            oof_rmse_mean=("oof_rmse", "mean"),
            oof_rmse_std=("oof_rmse", "std"),
            oof_spearman_mean=("oof_spearman", "mean"),
            oof_spearman_std=("oof_spearman", "std"),
            low_tail_rmse_mean=("low_tail_rmse", "mean"),
            low_tail_rmse_std=("low_tail_rmse", "std"),
        )
        .sort_values("oof_rmse_mean")
    )
    metrics.to_csv(OUTPUT_DIR / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "target": STARCODER_TARGET,
                "n_rows": len(y),
                "n_splits": N_SPLITS,
                "seeds": list(SEEDS),
                "models": summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    report = [
        "# StarCoder two-dimensional surface upper bound",
        "",
        "These flexible regressors are diagnostics, not deployable mixture surrogates. "
        "They estimate how much held-out error remains after relaxing the mechanistic form.",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"Wrote StarCoder surface diagnostic to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
