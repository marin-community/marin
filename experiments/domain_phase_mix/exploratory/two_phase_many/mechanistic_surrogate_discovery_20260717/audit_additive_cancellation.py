# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Test whether additive channel cancellation marks frozen-heldout optimism.

The strongest frozen baseline is additive in mechanistic response channels.
Relative to each heldout policy's nearest fit design, a positive contribution
is a predicted BPB charge and a negative contribution is a predicted credit.
If missing complementarity causes the failures, large opposing charge and
credit should identify optimistic errors without using the observed target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "additive_cancellation_audit"
DECOMPOSITION_DIR = ARTIFACT_ROOT / "worst_policy_feature_decomposition"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
OPTIMISM_THRESHOLD = 0.05
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260717


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_auc(labels: np.ndarray, score: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, score))


def bootstrap_statistic(
    frame: pd.DataFrame,
    score: str,
    statistic: str,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    series_labels = frame["training_series"].astype(str).to_numpy()
    score_values = frame[score].to_numpy(dtype=float)
    optimism = frame["optimism"].to_numpy(dtype=float)
    series = np.unique(series_labels)
    blocks = [np.flatnonzero(series_labels == name) for name in series]
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in sampled])
        local_score = score_values[indices]
        local_optimism = optimism[indices]
        if statistic == "spearman":
            value = float(spearmanr(local_score, local_optimism).statistic)
        elif statistic == "auc":
            labels = local_optimism > OPTIMISM_THRESHOLD
            value = safe_auc(labels, local_score)
        else:
            raise ValueError(statistic)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]))


def cancellation_frame(summary: pd.DataFrame, details: pd.DataFrame, atlas: pd.DataFrame) -> pd.DataFrame:
    category = details.groupby(["dataset", "row_id", "category"], as_index=False)["output_bpb_contribution_delta"].sum()
    rows: list[dict[str, object]] = []
    for (dataset, row_id), local in category.groupby(["dataset", "row_id"], sort=False):
        contributions = local["output_bpb_contribution_delta"].to_numpy(dtype=float)
        charges = float(np.maximum(contributions, 0.0).sum())
        benefits = float(np.maximum(-contributions, 0.0).sum())
        opposing = min(charges, benefits)
        gross = charges + benefits
        rows.append(
            {
                "dataset": dataset,
                "row_id": row_id,
                "predicted_charge": charges,
                "predicted_credit": benefits,
                "opposing_mass": opposing,
                "cancellation_fraction": 2.0 * opposing / max(gross, 1e-12),
                "gross_channel_motion": gross,
            }
        )
    cancellation = pd.DataFrame(rows)
    baseline_atlas = atlas.loc[atlas["mechanism"].eq("baseline")].copy()
    merged = summary.merge(cancellation, on=["dataset", "row_id"], validate="one_to_one")
    merged = merged.merge(
        baseline_atlas[["dataset", "row_id", "training_series"]],
        on=["dataset", "row_id"],
        validate="one_to_one",
    )
    merged["optimism"] = merged["heldout_observed"] - merged["heldout_predicted"]
    return merged


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    diagnostics = (
        "opposing_mass",
        "cancellation_fraction",
        "predicted_credit",
        "gross_channel_motion",
        "mechanistic_design_distance",
    )
    rows: list[dict[str, object]] = []
    for dataset, local in frame.groupby("dataset", sort=False):
        labels = local["optimism"].to_numpy() > OPTIMISM_THRESHOLD
        for diagnostic in diagnostics:
            score = local[diagnostic].to_numpy(dtype=float)
            correlation = float(spearmanr(score, local["optimism"]).statistic)
            auc = safe_auc(labels, score)
            rho_low, rho_high = bootstrap_statistic(local, diagnostic, "spearman")
            auc_low, auc_high = bootstrap_statistic(local, diagnostic, "auc")
            rows.append(
                {
                    "dataset": dataset,
                    "diagnostic": diagnostic,
                    "spearman_with_optimism": correlation,
                    "spearman_ci_low": rho_low,
                    "spearman_ci_high": rho_high,
                    "optimism_gt_0p05_auc": auc,
                    "auc_ci_low": auc_low,
                    "auc_ci_high": auc_high,
                    "optimism_error_count": int(labels.sum()),
                }
            )
    return pd.DataFrame(rows)


def quartile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, diagnostic), local in (
        frame.assign(_key=1)
        .merge(
            pd.DataFrame(
                {"diagnostic": ["opposing_mass", "cancellation_fraction", "mechanistic_design_distance"], "_key": 1}
            ),
            on="_key",
        )
        .drop(columns="_key")
        .groupby(["dataset", "diagnostic"], sort=False)
    ):
        local = local.copy()
        local["quartile"] = pd.qcut(local[diagnostic], 4, labels=False, duplicates="drop")
        for quartile, bin_frame in local.groupby("quartile", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "diagnostic": diagnostic,
                    "quartile": int(quartile) + 1,
                    "count": len(bin_frame),
                    "mean_optimism": float(bin_frame["optimism"].mean()),
                    "optimism_gt_0p05_count": int((bin_frame["optimism"] > OPTIMISM_THRESHOLD).sum()),
                    "optimism_gt_0p05_rate": float((bin_frame["optimism"] > OPTIMISM_THRESHOLD).mean()),
                }
            )
    return pd.DataFrame(rows)


def render(frame: pd.DataFrame, output: Path) -> None:
    targets = list(frame["dataset"].unique())
    figure = make_subplots(
        rows=2,
        cols=len(targets),
        subplot_titles=[
            *(target.replace("delphi_3e18_", "") + " · opposing mass" for target in targets),
            *(target.replace("delphi_3e18_", "") + " · cancellation fraction" for target in targets),
        ],
    )
    for column, target in enumerate(targets, start=1):
        local = frame.loc[frame["dataset"].eq(target)]
        colors = np.where(local["optimism"] > OPTIMISM_THRESHOLD, "#d73027", "#1a9850")
        custom = local[["row_id", "training_series", "optimism"]]
        for row, diagnostic in enumerate(("opposing_mass", "cancellation_fraction"), start=1):
            figure.add_trace(
                go.Scatter(
                    x=local[diagnostic],
                    y=local["optimism"],
                    mode="markers",
                    marker={"color": colors, "size": 8, "opacity": 0.78, "line": {"width": 0.5, "color": "#183642"}},
                    customdata=custom,
                    hovertemplate=(
                        "%{customdata[0]}<br>series=%{customdata[1]}<br>diagnostic=%{x:.4f}"
                        "<br>optimism=%{customdata[2]:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            figure.add_hline(y=OPTIMISM_THRESHOLD, line_dash="dash", line_color="#d73027", row=row, col=column)
    figure.update_xaxes(title_text="Predicted opposing BPB mass", row=1)
    figure.update_xaxes(title_text="Cancellation fraction", row=2)
    figure.update_yaxes(title_text="Heldout optimism (observed - predicted)", col=1)
    figure.update_layout(
        title="Does additive channel cancellation identify frozen-heldout optimism?",
        template="plotly_white",
        width=1500,
        height=1050,
    )
    figure.write_html(output, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def main() -> None:
    args = parse_args()
    summary_path = DECOMPOSITION_DIR / "worst_policy_decomposition_summary.csv"
    details_path = DECOMPOSITION_DIR / "worst_policy_feature_contributions.csv"
    for path in (summary_path, details_path, FAILURE_ATLAS):
        gate.assert_sealed_absent(path)
    summary = pd.read_csv(summary_path)
    details = pd.read_csv(details_path)
    atlas = pd.read_csv(FAILURE_ATLAS)
    frame = cancellation_frame(summary, details, atlas)
    metrics = summarize(frame)
    quartiles = quartile_summary(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "heldout_cancellation_diagnostics.csv", index=False)
    metrics.to_csv(args.output_dir / "cancellation_metrics.csv", index=False)
    quartiles.to_csv(args.output_dir / "cancellation_quartiles.csv", index=False)
    render(frame, args.output_dir / "additive_cancellation_diagnostic.html")
    report = [
        "# Additive cancellation audit",
        "",
        "All diagnostics are computed from policy inputs and the frozen model relative to the nearest fit design. "
        "The observed target is used only to score optimism. Confidence intervals resample named validation series "
        "as blocks.",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Quartiles",
        "",
        quartiles.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
