# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test whether a sparse set of physical invariants explains heldout optimism.

This is a diagnostic upper bound trained directly on development-heldout
residuals. It is not an eligible surrogate or post-hoc deployment calibrator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260717"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "multivariate_invariant_upper_bound"
ALPHAS = np.logspace(-5, -1, 17)
FEATURES = (
    "support_distance",
    "aggregate_kl_to_proportional",
    "max_epoch",
    "mean_literal_replay",
    "mass_ratio_lt_0p25",
    "weighted_undercoverage_q10",
    "family_log_ratio_variance",
    "bucket_reverse_kl",
    "bucket_importance_variance",
)


def design(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, [feature for feature in FEATURES if feature != "weighted_undercoverage_q10"]].copy()
    result["weighted_undercoverage_q10"] = 1.0 - frame["weighted_ratio_q10"]
    return result.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def metric_row(observed: np.ndarray, predicted: np.ndarray, variant: str) -> dict[str, float | str]:
    optimism = observed - predicted
    slope = float(np.polyfit(predicted, observed, 1)[0])
    rank = float(spearmanr(observed, predicted).statistic)
    selected = int(np.argmin(predicted))
    return {
        "variant": variant,
        "rmse": float(np.sqrt(np.mean(np.square(predicted - observed)))),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "spearman": rank,
        "bias_predicted_minus_observed": float(np.mean(predicted - observed)),
        "calibration_slope_observed_on_predicted": slope,
        "optimism_gt_0p05_count": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
        "selected_optimism": float(optimism[selected]),
        "regret_at_1": float(observed[selected] - np.min(observed)),
    }


def nested_oof(
    frame: pd.DataFrame,
    *,
    positive: bool,
) -> tuple[np.ndarray, pd.DataFrame, float, pd.DataFrame]:
    x = design(frame)
    y = frame["optimism"].to_numpy(dtype=float)
    groups = frame["training_series"].astype(str).to_numpy()
    outer = LeaveOneGroupOut()
    oof = np.full(len(frame), np.nan)
    fold_rows: list[dict[str, float | int]] = []
    coefficient_rows: list[pd.Series] = []
    for fold, (train_indices, test_indices) in enumerate(outer.split(x, y, groups)):
        inner_groups = groups[train_indices]
        inner = GroupKFold(n_splits=min(5, len(np.unique(inner_groups))))
        search = GridSearchCV(
            make_pipeline(StandardScaler(), Lasso(positive=positive, max_iter=100_000)),
            {"lasso__alpha": ALPHAS},
            scoring="neg_mean_squared_error",
            cv=inner,
        )
        search.fit(x.iloc[train_indices], y[train_indices], groups=inner_groups)
        oof[test_indices] = search.predict(x.iloc[test_indices])
        coefficients = pd.Series(search.best_estimator_.named_steps["lasso"].coef_, index=FEATURES)
        coefficient_rows.append(coefficients)
        fold_rows.append(
            {
                "fold": fold,
                "selected_alpha": float(search.best_params_["lasso__alpha"]),
                "active_features": int(np.sum(np.abs(coefficients) > 1e-10)),
                "test_series": len(np.unique(groups[test_indices])),
            }
        )
    if np.isnan(oof).any():
        raise ValueError("Nested grouped OOF predictions are incomplete")

    full_cv = GroupKFold(n_splits=5)
    full_search = GridSearchCV(
        make_pipeline(StandardScaler(), Lasso(positive=positive, max_iter=100_000)),
        {"lasso__alpha": ALPHAS},
        scoring="neg_mean_squared_error",
        cv=full_cv,
    )
    full_search.fit(x, y, groups=groups)
    full_coefficients = pd.Series(full_search.best_estimator_.named_steps["lasso"].coef_, index=FEATURES)
    fold_coefficients = pd.DataFrame(coefficient_rows)
    active = np.abs(fold_coefficients) > 1e-10
    positive_fraction = (fold_coefficients > 1e-10).mean(axis=0)
    negative_fraction = (fold_coefficients < -1e-10).mean(axis=0)
    stability = pd.DataFrame(
        {
            "feature": FEATURES,
            "outer_fold_active_fraction": active.mean(axis=0).to_numpy(),
            "outer_fold_positive_fraction": positive_fraction.to_numpy(),
            "outer_fold_negative_fraction": negative_fraction.to_numpy(),
            "dominant_sign_fraction": np.maximum(positive_fraction, negative_fraction).to_numpy(),
            "outer_fold_coefficient_median": fold_coefficients.median(axis=0).to_numpy(),
            "full_fit_standardized_coefficient": full_coefficients.to_numpy(),
        }
    )
    return oof, pd.DataFrame(fold_rows), float(full_search.best_params_["lasso__alpha"]), stability


def main() -> None:
    gate.assert_sealed_absent(FAILURE_ATLAS)
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(FAILURE_ATLAS)
    baseline = atlas.loc[atlas["mechanism"].eq("baseline")].copy()
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | str]] = []
    fold_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    for dataset, frame in baseline.groupby("dataset", sort=True):
        frame = frame.reset_index(drop=True)
        observed = frame["observed"].to_numpy(dtype=float)
        base_predicted = frame["predicted"].to_numpy(dtype=float)
        residual_oof, folds, full_alpha, stability = nested_oof(frame, positive=True)
        unconstrained_oof, unconstrained_folds, unconstrained_alpha, unconstrained_stability = nested_oof(
            frame, positive=False
        )
        corrected = base_predicted + residual_oof
        unconstrained = base_predicted + unconstrained_oof
        for variant, predicted in (
            ("frozen_baseline", base_predicted),
            ("monotone_sparse_upper_bound", corrected),
            ("unconstrained_sparse_upper_bound", unconstrained),
        ):
            row = metric_row(observed, predicted, variant)
            row["dataset"] = dataset
            metric_rows.append(row)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "row_id": frame["row_id"],
                    "training_series": frame["training_series"],
                    "observed": observed,
                    "baseline_predicted": base_predicted,
                    "monotone_predicted": corrected,
                    "unconstrained_predicted": unconstrained,
                    "monotone_oof_predicted_optimism": residual_oof,
                    "unconstrained_oof_predicted_optimism": unconstrained_oof,
                }
            )
        )
        folds.insert(0, "variant", "monotone")
        unconstrained_folds.insert(0, "variant", "unconstrained")
        folds.insert(0, "dataset", dataset)
        unconstrained_folds.insert(0, "dataset", dataset)
        folds["full_fit_alpha"] = full_alpha
        unconstrained_folds["full_fit_alpha"] = unconstrained_alpha
        fold_frames.extend((folds, unconstrained_folds))
        stability.insert(0, "variant", "monotone")
        unconstrained_stability.insert(0, "variant", "unconstrained")
        stability.insert(0, "dataset", dataset)
        unconstrained_stability.insert(0, "dataset", dataset)
        stability_frames.extend((stability, unconstrained_stability))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    stability = pd.concat(stability_frames, ignore_index=True)
    metrics.to_csv(DEFAULT_OUTPUT / "metrics.csv", index=False)
    predictions.to_csv(DEFAULT_OUTPUT / "oof_predictions.csv", index=False)
    folds.to_csv(DEFAULT_OUTPUT / "fold_selection.csv", index=False)
    stability.to_csv(DEFAULT_OUTPUT / "feature_stability.csv", index=False)

    long = predictions.melt(
        id_vars=["dataset", "row_id", "training_series", "observed"],
        value_vars=["baseline_predicted", "monotone_predicted", "unconstrained_predicted"],
        var_name="variant",
        value_name="predicted",
    )
    figure = px.scatter(
        long,
        x="predicted",
        y="observed",
        facet_row="dataset",
        facet_col="variant",
        color="observed",
        color_continuous_scale="RdYlGn_r",
        hover_data=["row_id", "training_series"],
        title="Sparse physical-invariant residual upper bound (nested grouped OOF)",
        height=900,
        width=1200,
    )
    for row in (1, 2):
        for column in (1, 2, 3):
            figure.add_shape(
                type="line",
                x0=0.95,
                y0=0.95,
                x1=1.5,
                y1=1.5,
                line={"color": "#607682", "dash": "dash"},
                row=row,
                col=column,
            )
    figure.update_layout(template="plotly_white", margin={"l": 65, "r": 40, "t": 90, "b": 60})
    figure.write_html(
        DEFAULT_OUTPUT / "multivariate_invariant_upper_bound.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )

    comparison = metrics.pivot(index="dataset", columns="variant")
    stability_summary = stability.groupby(["dataset", "variant"], as_index=False).agg(
        stable_feature_count=("outer_fold_active_fraction", lambda values: int(np.sum(np.asarray(values) >= 0.8))),
        stable_sign_feature_count=("dominant_sign_fraction", lambda values: int(np.sum(np.asarray(values) >= 0.8))),
        median_feature_active_fraction=("outer_fold_active_fraction", "median"),
    )
    report = f"""# Multivariate physical-invariant upper bound

This deliberately ineligible diagnostic predicts the frozen baseline's development-heldout optimism from nine prespecified physical invariants. It uses leave-one-training-series-out nested selection and no row-level leakage. The monotone variant constrains every invariant to its prespecified harmful direction; the unconstrained comparison is allowed to reverse signs only to reveal whether an arbitrary residual restatement can help. Because both are fitted to heldout residuals, neither can be promoted or used for deployment.

## Metrics

{metrics.to_markdown(index=False, floatfmt=".6f")}

## Feature stability

{stability.to_markdown(index=False, floatfmt=".6f")}

{stability_summary.to_markdown(index=False, floatfmt=".6f")}

## Interpretation

The diagnostic is useful only if it materially reduces extreme optimism with a stable sparse feature set. A small RMSE gain with changing active features instead supports the existing conclusion that deployment error is underidentified rather than governed by one omitted scalar. It must not be read as evidence for a residual-correction layer.
"""
    del comparison
    (DEFAULT_OUTPUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
