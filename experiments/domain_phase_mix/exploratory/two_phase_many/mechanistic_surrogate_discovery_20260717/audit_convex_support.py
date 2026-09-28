# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
#   "tabulate",
# ]
# ///
"""Separate interpolation from extrapolation in the frozen mechanistic state.

Each policy's mechanistic design is projected onto the convex hull of the fit
designs. Fit points are projected leave-one-out. This tests whether frozen
heldout failures occur where the response is interpolating among observed
states or where any parametric response must extrapolate beyond them.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "convex_support_audit"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
RAW_OPTIMUM_WEIGHTS = ARTIFACT_ROOT / "raw_optimum_audit/raw_optimum_weights.csv"
VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
TARGETS = (
    base.DatasetId.DELPHI_3E18_UNCHEATABLE,
    base.DatasetId.DELPHI_3E18_TABLE9,
)
OPTIMISM_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def candidate_dataset(dataset: base.Dataset, weights: np.ndarray, target: np.ndarray) -> base.Dataset:
    return replace(dataset, weights=weights, target=target)


class ConvexProjector:
    def __init__(self, fit_design: np.ndarray):
        self.fit_design = fit_design
        self.coefficients = cp.Variable(fit_design.shape[0], nonneg=True)
        self.target = cp.Parameter(fit_design.shape[1])
        self.upper = cp.Parameter(fit_design.shape[0], nonneg=True)
        residual = fit_design.T @ self.coefficients - self.target
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(residual)),
            [cp.sum(self.coefficients) == 1.0, self.coefficients <= self.upper],
        )

    def project(self, target: np.ndarray, excluded: int | None = None) -> tuple[float, float, str]:
        upper = np.ones(self.fit_design.shape[0], dtype=float)
        if excluded is not None:
            upper[excluded] = 0.0
        self.target.value = target
        self.upper.value = upper
        self.problem.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            tol_gap_abs=1e-9,
            tol_feas=1e-9,
            tol_gap_rel=1e-9,
            max_iter=1_000,
        )
        if self.coefficients.value is None:
            raise RuntimeError(f"Convex projection failed: {self.problem.status}")
        coefficients = np.maximum(np.asarray(self.coefficients.value, dtype=float), 0.0)
        coefficients /= coefficients.sum()
        residual = self.fit_design.T @ coefficients - target
        distance = float(np.sqrt(np.mean(residual**2)))
        effective_support = float(1.0 / np.sum(coefficients**2))
        return distance, effective_support, str(self.problem.status)


def audit_target(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    atlas: pd.DataFrame,
    raw_optimum_weights: pd.DataFrame,
) -> pd.DataFrame:
    dataset = base.load_dataset(dataset_id)
    config = output_link.selected_deficit_config(dataset_id, VARIANT, source_metrics)
    fit_design = deficit.build_design(dataset, config).values
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(dataset_id)
    heldout_frame, heldout_weights, heldout_target = heldout
    baseline = atlas.loc[atlas["dataset"].eq(dataset_id.value) & atlas["mechanism"].eq("baseline")].copy()
    retained = heldout_frame["wandb_run_name"].astype(str).isin(baseline["row_id"].astype(str)).to_numpy()
    heldout_frame = heldout_frame.loc[retained].reset_index(drop=True)
    heldout_weights = heldout_weights[retained]
    heldout_target = heldout_target[retained]
    heldout_dataset = candidate_dataset(dataset, heldout_weights, heldout_target)
    heldout_design = deficit.build_design(heldout_dataset, config).values
    scale = fit_design.std(axis=0)
    active = scale > 1e-10
    mean = fit_design[:, active].mean(axis=0)
    standardized_fit = (fit_design[:, active] - mean) / scale[active]
    standardized_heldout = (heldout_design[:, active] - mean) / scale[active]
    projector = ConvexProjector(standardized_fit)

    fit_distances = np.empty(dataset.n, dtype=float)
    fit_support = np.empty(dataset.n, dtype=float)
    for index, design in enumerate(standardized_fit):
        fit_distances[index], fit_support[index], _status = projector.project(design, excluded=index)
    heldout_distances = np.empty(len(heldout_frame), dtype=float)
    heldout_support = np.empty(len(heldout_frame), dtype=float)
    statuses: list[str] = []
    for index, design in enumerate(standardized_heldout):
        heldout_distances[index], heldout_support[index], status = projector.project(design)
        statuses.append(status)

    baseline = baseline.set_index("row_id")
    rows: list[dict[str, object]] = []
    p95 = float(np.quantile(fit_distances, 0.95))
    for index, row in heldout_frame.reset_index(drop=True).iterrows():
        row_id = str(row["wandb_run_name"])
        atlas_row = baseline.loc[row_id]
        rows.append(
            {
                "dataset": dataset_id.value,
                "row_id": row_id,
                "training_series": atlas_row["training_series"],
                "observed": float(atlas_row["observed"]),
                "predicted": float(atlas_row["predicted"]),
                "optimism": float(atlas_row["optimism"]),
                "convex_hull_distance": heldout_distances[index],
                "fit_loo_distance_p95": p95,
                "distance_over_fit_p95": heldout_distances[index] / max(p95, 1e-12),
                "convex_effective_support": heldout_support[index],
                "outside_fit_loo_p95": bool(heldout_distances[index] > p95),
                "solver_status": statuses[index],
            }
        )
    result = pd.DataFrame(rows)
    optimum_rows: list[dict[str, object]] = []
    optima = raw_optimum_weights.loc[
        raw_optimum_weights["dataset"].eq(dataset_id.value) & raw_optimum_weights["model"].eq("baseline")
    ]
    for policy, optimum in optima.groupby("policy", sort=False):
        ordered = optimum.set_index("domain").loc[list(dataset.domains)]
        weights = np.asarray(
            [[ordered["phase0_weight"].to_numpy(), ordered["phase1_weight"].to_numpy()]],
            dtype=float,
        )
        optimum_dataset = candidate_dataset(dataset, weights, np.zeros(1, dtype=float))
        optimum_design = deficit.build_design(optimum_dataset, config).values[:, active]
        standardized_optimum = (optimum_design[0] - mean) / scale[active]
        distance, effective_support, status = projector.project(standardized_optimum)
        optimum_rows.append(
            {
                "dataset": dataset_id.value,
                "policy": policy,
                "convex_hull_distance": distance,
                "fit_loo_distance_p95": p95,
                "distance_over_fit_p95": distance / max(p95, 1e-12),
                "convex_effective_support": effective_support,
                "solver_status": status,
            }
        )
    fit_reference = pd.DataFrame(
        {
            "dataset": dataset_id.value,
            "fit_row": dataset.frame["run_name"].astype(str),
            "fit_loo_convex_hull_distance": fit_distances,
            "fit_loo_effective_support": fit_support,
        }
    )
    result.attrs["fit_reference"] = fit_reference
    result.attrs["raw_optima"] = pd.DataFrame(optimum_rows)
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, local in frame.groupby("dataset", sort=False):
        labels = local["optimism"].to_numpy() > OPTIMISM_THRESHOLD
        distance = local["convex_hull_distance"].to_numpy()
        rows.append(
            {
                "dataset": dataset,
                "heldout_count": len(local),
                "outside_fit_loo_p95_count": int(local["outside_fit_loo_p95"].sum()),
                "optimism_gt_0p05_count": int(labels.sum()),
                "optimism_errors_outside_count": int((labels & local["outside_fit_loo_p95"].to_numpy()).sum()),
                "spearman_distance_optimism": float(spearmanr(distance, local["optimism"]).statistic),
                "distance_auc_optimism_gt_0p05": float(roc_auc_score(labels, distance)),
                "median_distance_over_fit_p95": float(local["distance_over_fit_p95"].median()),
                "minimum_failure_distance_over_fit_p95": float(local.loc[labels, "distance_over_fit_p95"].min()),
            }
        )
    return pd.DataFrame(rows)


def stratified_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, target_frame in frame.groupby("dataset", sort=False):
        for support_region, local in target_frame.groupby("outside_fit_loo_p95", sort=True):
            observed = local["observed"].to_numpy(dtype=float)
            predicted = local["predicted"].to_numpy(dtype=float)
            centered = predicted - predicted.mean()
            slope = float(centered @ (observed - observed.mean()) / max(centered @ centered, 1e-12))
            residual = predicted - observed
            rows.append(
                {
                    "dataset": dataset,
                    "support_region": "outside" if support_region else "inside",
                    "count": len(local),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "bias_predicted_minus_observed": float(residual.mean()),
                    "observed_on_predicted_slope": slope,
                    "optimism_gt_0p05_count": int((-residual > OPTIMISM_THRESHOLD).sum()),
                    "worst_optimism": float((-residual).max()),
                }
            )
    return pd.DataFrame(rows)


def render(frame: pd.DataFrame, fit_reference: pd.DataFrame, output: Path) -> None:
    targets = list(frame["dataset"].unique())
    figure = make_subplots(
        rows=2,
        cols=len(targets),
        subplot_titles=[
            *(target.replace("delphi_3e18_", "") + " · support distance" for target in targets),
            *(target.replace("delphi_3e18_", "") + " · fit versus heldout" for target in targets),
        ],
    )
    for column, target in enumerate(targets, start=1):
        local = frame.loc[frame["dataset"].eq(target)]
        reference = fit_reference.loc[fit_reference["dataset"].eq(target)]
        colors = np.where(local["optimism"] > OPTIMISM_THRESHOLD, "#d73027", "#1a9850")
        figure.add_trace(
            go.Scatter(
                x=local["distance_over_fit_p95"],
                y=local["optimism"],
                mode="markers",
                marker={"color": colors, "size": 8, "opacity": 0.78},
                customdata=local[["row_id", "training_series", "observed", "predicted"]],
                hovertemplate=(
                    "%{customdata[0]}<br>series=%{customdata[1]}<br>distance/p95=%{x:.3f}"
                    "<br>optimism=%{y:.4f}<br>observed=%{customdata[2]:.4f}"
                    "<br>predicted=%{customdata[3]:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.add_hline(y=OPTIMISM_THRESHOLD, line_dash="dash", line_color="#d73027", row=1, col=column)
        figure.add_vline(x=1.0, line_dash="dot", line_color="#4575b4", row=1, col=column)
        figure.add_trace(
            go.Histogram(
                x=reference["fit_loo_convex_hull_distance"],
                name="fit LOO",
                marker_color="#4575b4",
                opacity=0.65,
                histnorm="probability density",
                showlegend=column == 1,
            ),
            row=2,
            col=column,
        )
        figure.add_trace(
            go.Histogram(
                x=local["convex_hull_distance"],
                name="heldout",
                marker_color="#d73027",
                opacity=0.55,
                histnorm="probability density",
                showlegend=column == 1,
            ),
            row=2,
            col=column,
        )
    figure.update_xaxes(title_text="Heldout convex distance / fit LOO p95", row=1)
    figure.update_yaxes(title_text="Heldout optimism", row=1, col=1)
    figure.update_xaxes(title_text="Standardized convex-hull distance", row=2)
    figure.update_yaxes(title_text="Density", row=2, col=1)
    figure.update_layout(
        title="Frozen mechanistic-state interpolation versus extrapolation",
        template="plotly_white",
        barmode="overlay",
        width=1500,
        height=1000,
        legend={"orientation": "h", "y": -0.12},
    )
    figure.write_html(output, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def main() -> None:
    args = parse_args()
    for path in (SOURCE_METRICS, FAILURE_ATLAS, RAW_OPTIMUM_WEIGHTS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    atlas = pd.read_csv(FAILURE_ATLAS)
    raw_optimum_weights = pd.read_csv(RAW_OPTIMUM_WEIGHTS)
    heldout_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    optimum_frames: list[pd.DataFrame] = []
    for dataset_id in TARGETS:
        audited = audit_target(dataset_id, source_metrics, atlas, raw_optimum_weights)
        fit_frames.append(audited.attrs["fit_reference"])
        optimum_frames.append(audited.attrs["raw_optima"])
        audited.attrs.clear()
        heldout_frames.append(audited)
    heldout = pd.concat(heldout_frames, ignore_index=True)
    fit_reference = pd.concat(fit_frames, ignore_index=True)
    raw_optima = pd.concat(optimum_frames, ignore_index=True)
    metrics = summarize(heldout)
    calibration = stratified_calibration(heldout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.output_dir / "heldout_convex_support.csv", index=False)
    fit_reference.to_csv(args.output_dir / "fit_loo_convex_support.csv", index=False)
    raw_optima.to_csv(args.output_dir / "raw_optimum_convex_support.csv", index=False)
    metrics.to_csv(args.output_dir / "convex_support_metrics.csv", index=False)
    calibration.to_csv(args.output_dir / "support_stratified_calibration.csv", index=False)
    render(heldout, fit_reference, args.output_dir / "convex_support_audit.html")
    failures = heldout.loc[heldout["optimism"] > OPTIMISM_THRESHOLD].sort_values(
        ["dataset", "optimism"], ascending=[True, False]
    )
    report = [
        "# Convex-support audit",
        "",
        "Distances are computed in the frozen baseline's standardized active mechanistic design. Fit support is "
        "calibrated by projecting every fit point onto the convex hull of the other 279 points.",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Calibration by empirical convex support",
        "",
        calibration.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optimum support",
        "",
        raw_optima.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimism failures",
        "",
        failures.to_markdown(index=False, floatfmt=".6f"),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
