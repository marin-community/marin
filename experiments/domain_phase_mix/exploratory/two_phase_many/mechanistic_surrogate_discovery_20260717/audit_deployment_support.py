# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Measure whether frozen 3e18 heldouts lie outside mechanistic fit support.

Distances are diagnostics only. They are never used as model features or
post-hoc corrections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    analyze_failure_modes as failures,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    audit_oof_identification as identification,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_portfolio as portfolio,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.freeze_baseline_gate import (
    assert_sealed_absent,
)

DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/deployment_support_audit"
TARGETS = ("uncheatable", "table9")
MODEL = "early_family_asymmetric"
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}
FEATURES = (
    "phase_tv",
    "aggregate_tv_to_proportional",
    "aggregate_kl_to_proportional",
    "max_epoch",
    "mean_literal_replay",
    "max_literal_replay",
    "mass_ratio_lt_0p1",
    "mass_ratio_lt_0p25",
    "mass_ratio_lt_0p5",
    "weighted_ratio_q10",
    "weighted_ratio_q25",
    "min_family_ratio",
    "family_log_ratio_variance",
    "bucket_reverse_kl",
    "bucket_importance_variance",
)


def feature_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    swarm = bundle["swarms"]["delphi_3e18"]
    domains = tuple(item["id"] for item in swarm["domains"])
    proportional = np.asarray([item["proportionalWeight"] for item in swarm["domains"]], dtype=float)
    reference_epochs = np.asarray(
        [
            item["proportionalWeight"] * (item["phase0EpochFactor"] + item["phase1EpochFactor"])
            for item in swarm["domains"]
        ],
        dtype=float,
    )
    _family_names, family_members, _group_names, _group_members, _indices = portfolio.hierarchical_partition(domains)
    records = []
    for row in swarm["rows"]:
        if row["isSharedAlias"]:
            continue
        if row["policyFamily"] != "two_phase":
            continue
        if row["split"] not in {"fit", "heldout"}:
            continue
        records.append(
            {
                "row_id": row["name"],
                "split": "fit_oof" if row["split"] == "fit" else "heldout_policy_matched",
                "panel": row["panel"],
                "method": row["method"],
                **failures.row_features(row, proportional, reference_epochs, family_members, domains),
            }
        )
    return pd.DataFrame(records)


def transformed_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[list(FEATURES)].to_numpy(dtype=float)
    if np.any(values < -1e-12):
        raise ValueError("Support features must be nonnegative")
    return np.log1p(np.maximum(values, 0.0))


def support_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    fit = features.loc[features["split"].eq("fit_oof")].copy()
    heldout = features.loc[features["split"].eq("heldout_policy_matched")].copy()
    fit_x = transformed_matrix(fit)
    heldout_x = transformed_matrix(heldout)
    mean = fit_x.mean(axis=0)
    scale = fit_x.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-10, scale, 1.0)
    fit_z = (fit_x - mean) / scale
    heldout_z = (heldout_x - mean) / scale

    fit_pairwise = cdist(fit_z, fit_z)
    np.fill_diagonal(fit_pairwise, np.inf)
    fit_nearest = fit_pairwise.min(axis=1)
    heldout_nearest = cdist(heldout_z, fit_z).min(axis=1)
    nearest_scale = float(np.median(fit_nearest))

    covariance = LedoitWolf().fit(fit_z)
    precision = covariance.precision_
    fit_center = fit_z.mean(axis=0)
    heldout_centered = heldout_z - fit_center
    mahalanobis = np.sqrt(np.einsum("ni,ij,nj->n", heldout_centered, precision, heldout_centered))

    lower = np.quantile(fit_x, 0.01, axis=0)
    upper = np.quantile(fit_x, 0.99, axis=0)
    outside = (heldout_x < lower[None, :]) | (heldout_x > upper[None, :])

    output = heldout[["row_id", "split", "panel", "method", *FEATURES]].copy()
    output["nearest_fit_z_distance"] = heldout_nearest
    output["nearest_distance_over_fit_median"] = heldout_nearest / max(nearest_scale, 1e-12)
    output["shrinkage_mahalanobis"] = mahalanobis
    output["outside_fit_01_99_count"] = outside.sum(axis=1)
    output["outside_fit_features"] = [
        ";".join(feature for feature, flag in zip(FEATURES, flags, strict=True) if flag) for flags in outside
    ]
    return output


def add_predictions(support: pd.DataFrame, target: str) -> pd.DataFrame:
    selected = identification.external_predictions(target, MODEL)
    selected = selected.loc[selected["split"].eq("heldout_policy_matched")][["row_id", "observed", "predicted"]]
    output = support.merge(selected, on="row_id", how="inner", validate="one_to_one")
    output["target"] = target
    output["optimism"] = output["observed"] - output["predicted"]
    output["absolute_error"] = output["optimism"].abs()
    return output


def calibration_bins(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for target, local in frame.groupby("target", sort=True):
        bins = pd.qcut(local["nearest_distance_over_fit_median"], 5, duplicates="drop")
        for interval, group in local.groupby(bins, observed=True):
            output.append(
                {
                    "target": target,
                    "support_distance_bin": str(interval),
                    "n": len(group),
                    "mean_distance_ratio": group["nearest_distance_over_fit_median"].mean(),
                    "mean_optimism": group["optimism"].mean(),
                    "mean_absolute_error": group["absolute_error"].mean(),
                    "optimism_gt_0p05_count": int((group["optimism"] > 0.05).sum()),
                    "worst_optimism": group["optimism"].max(),
                }
            )
    return pd.DataFrame(output)


def plot(frame: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"))
    for column, target in enumerate(TARGETS, start=1):
        local = frame.loc[frame["target"].eq(target)]
        figure.add_trace(
            go.Scatter(
                x=local["nearest_distance_over_fit_median"],
                y=local["optimism"],
                mode="markers",
                marker={
                    "size": 7,
                    "color": local["outside_fit_01_99_count"],
                    "colorscale": "RdYlGn_r",
                    "showscale": column == 2,
                    "colorbar": {"title": "features<br>outside<br>fit 1--99%"},
                    "line": {"color": "#24364b", "width": 0.5},
                },
                customdata=np.column_stack([local["row_id"], local["panel"], local["outside_fit_features"]]),
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>distance / fit median=%{x:.2f}"
                    "<br>optimism=%{y:.4f}<br>outside=%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_hline(y=0.05, line_dash="dash", line_color="#b2182b", row=1, col=column)
        figure.add_hline(y=0.0, line_dash="dot", line_color="#666", row=1, col=column)
    figure.update_xaxes(title_text="Nearest fit distance / median fit-to-fit distance", type="log")
    figure.update_yaxes(title_text="Optimism (observed - predicted BPB)")
    figure.update_layout(
        title="Frozen 3e18 errors versus mechanistic fit-support distance",
        template="plotly_white",
        width=1500,
        height=650,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    assert_sealed_absent(DASHBOARD)
    bundle = json.loads(DASHBOARD.read_text())
    features = feature_frame(bundle)
    support = support_diagnostics(features)
    scored = pd.concat(
        [add_predictions(support, target) for target in TARGETS],
        ignore_index=True,
    )
    bins = calibration_bins(scored)
    correlations = []
    for target, local in scored.groupby("target", sort=True):
        for diagnostic in (
            "nearest_distance_over_fit_median",
            "shrinkage_mahalanobis",
            "outside_fit_01_99_count",
        ):
            result = spearmanr(local[diagnostic], local["optimism"])
            correlations.append(
                {
                    "target": target,
                    "diagnostic": diagnostic,
                    "spearman_with_optimism": result.statistic,
                    "pvalue": result.pvalue,
                }
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT / "heldout_support_diagnostics.csv", index=False)
    bins.to_csv(OUTPUT / "support_binned_calibration.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUTPUT / "support_error_correlations.csv", index=False)
    scored.sort_values(["target", "optimism"], ascending=[True, False]).groupby("target", as_index=False).head(
        10
    ).to_csv(OUTPUT / "worst_optimism_with_support.csv", index=False)
    plot(scored, OUTPUT / "optimism_vs_fit_support.html")

    report = [
        "# Deployment-support audit",
        "",
        "Distances are diagnostics only; no distance enters a surrogate or output correction. The predictor is the frozen early-family asymmetric baseline for both targets.",
    ]
    for target, local in scored.groupby("target", sort=True):
        far = local["nearest_distance_over_fit_median"] >= local["nearest_distance_over_fit_median"].quantile(0.9)
        near = local["nearest_distance_over_fit_median"] <= local["nearest_distance_over_fit_median"].quantile(0.5)
        report.extend(
            [
                "",
                f"## {target}",
                "",
                f"- Median heldout distance is {local['nearest_distance_over_fit_median'].median():.2f} times the median fit-to-fit nearest distance; p90 is {local['nearest_distance_over_fit_median'].quantile(0.9):.2f}.",
                f"- {int((local['outside_fit_01_99_count'] > 0).sum())}/{len(local)} heldouts leave the fit panel's 1--99% interval on at least one invariant.",
                f"- Near-half optimism mean/worst: {local.loc[near, 'optimism'].mean():+.4f}/{local.loc[near, 'optimism'].max():+.4f} BPB.",
                f"- Far-decile optimism mean/worst: {local.loc[far, 'optimism'].mean():+.4f}/{local.loc[far, 'optimism'].max():+.4f} BPB.",
            ]
        )
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
