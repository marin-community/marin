# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Quantify decision non-identifiability under the frozen surrogate baseline.

This is a diagnostic, not a candidate model. It never fits against the exposed
adversarial outcomes and never reads the running sealed confirmation panel.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719.freeze_pareto_gate import (
    BASELINE_MODELS,
    DEFAULT_DASHBOARD,
    delphi_development_layer,
    metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719.freeze_pareto_gate import (
    DEFAULT_OUTPUT as DEFAULT_FROZEN_GATE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719/round53_partial_identification"
)
REFINED_WSD80_DATA = (
    RESEARCH_DIR / "reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_observed_metrics.csv"
)
STRICT_RELATIVE_OOF_THRESHOLD = 0.05
DESCRIPTIVE_RELATIVE_OOF_THRESHOLD = 0.15
TOP_K = (1, 3, 5, 10)
SUPPORT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
COLORS = {
    "uncheatable": "#0f766e",
    "table9": "#d97706",
}


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 5) -> str:
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def write_html(figure: go.Figure, path: Path) -> None:
    figure.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def finite(values: Iterable[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)


def heldout_frame(bundle: dict[str, Any], target: str) -> pd.DataFrame:
    swarm = bundle["swarms"]["delphi_3e18"]
    records = []
    for index, row in enumerate(swarm["rows"]):
        if row["split"] != "heldout" or row["isSharedAlias"]:
            continue
        record = {
            "row_index": index,
            "row_id": row["id"],
            "name": row["name"],
            "policy_class": row["policyFamily"],
            "panel": row["panel"],
            "development_layer": delphi_development_layer(row),
            "candidate_target": row.get("candidateTarget"),
            "support_distance": float(row["diagnostics"]["supportDistance"]),
            "phase_tv": float(row["diagnostics"]["phaseTv"]),
            "aggregate_tv_to_proportional": float(row["diagnostics"]["aggregateTvToProportional"]),
            "aggregate_kl_to_proportional": float(row["diagnostics"]["aggregateKlToProportional"]),
            "max_epoch": float(row["diagnostics"]["maxEpoch"]),
            "observed": float(row["observed"][target]),
        }
        for model in BASELINE_MODELS:
            record[f"prediction::{model}"] = float(swarm["predictions"][target]["two_phase"][model]["prediction"][index])
        records.append(record)
    result = pd.DataFrame(records)
    if len(result) != 710:
        raise ValueError(f"Expected 710 coordinate-disjoint heldouts, found {len(result)}")
    result["support_quartile"] = pd.qcut(
        result["support_distance"],
        q=SUPPORT_QUANTILES,
        labels=["Q1 nearest", "Q2", "Q3", "Q4 farthest"],
        duplicates="raise",
    )
    return result


def fit_oof_model_sets(metrics_frame: pd.DataFrame, target: str) -> tuple[list[str], list[str], pd.DataFrame]:
    selected = metrics_frame.loc[
        metrics_frame["swarm"].eq("delphi_3e18")
        & metrics_frame["target"].eq(target)
        & metrics_frame["policy"].eq("two_phase")
        & metrics_frame["split"].eq("fit_oof")
        & metrics_frame["model"].isin(BASELINE_MODELS)
    ].copy()
    selected = selected.sort_values(["rmse", "model"])
    best = float(selected["rmse"].min())
    selected["relative_to_best"] = selected["rmse"] / best - 1.0
    strict = selected.loc[selected["relative_to_best"].le(STRICT_RELATIVE_OOF_THRESHOLD), "model"].tolist()
    descriptive = selected.loc[selected["relative_to_best"].le(DESCRIPTIVE_RELATIVE_OOF_THRESHOLD), "model"].tolist()
    return strict, descriptive, selected


def selected_policy_rows(frame: pd.DataFrame, target: str, model_set: str, models: list[str]) -> list[dict[str, Any]]:
    best_observed = float(frame["observed"].min())
    output = []
    for model in models:
        prediction_column = f"prediction::{model}"
        selected_index = int(frame[prediction_column].idxmin())
        row = frame.loc[selected_index]
        prediction = frame[prediction_column].to_numpy(dtype=float)
        observed = frame["observed"].to_numpy(dtype=float)
        summary, _ = metrics(observed, prediction)
        output.append(
            {
                "target": target,
                "model_set": model_set,
                "model": model,
                "selected_name": row["name"],
                "selected_policy_class": row["policy_class"],
                "selected_layer": row["development_layer"],
                "selected_observed": float(row["observed"]),
                "selected_predicted": float(row[prediction_column]),
                "selected_regret": float(row["observed"] - best_observed),
                "selected_optimism": float(row["observed"] - row[prediction_column]),
                "support_distance": float(row["support_distance"]),
                "phase_tv": float(row["phase_tv"]),
                "aggregate_tv_to_proportional": float(row["aggregate_tv_to_proportional"]),
                "max_epoch": float(row["max_epoch"]),
                "archive_rmse": float(summary["rmse"]),
                "archive_spearman": float(summary["spearman"]),
                "archive_calibration_slope": float(summary["calibration_slope_observed_on_predicted"]),
            }
        )
    return output


def top_indices(frame: pd.DataFrame, model: str, k: int) -> set[int]:
    values = frame[f"prediction::{model}"].to_numpy(dtype=float)
    return set(np.argsort(values)[: min(k, len(values))].tolist())


def pairwise_model_rows(frame: pd.DataFrame, target: str, model_set: str, models: list[str]) -> list[dict[str, Any]]:
    output = []
    observed = frame["observed"].to_numpy(dtype=float)
    best_observed = float(np.min(observed))
    for left_index, left in enumerate(models):
        left_prediction = frame[f"prediction::{left}"].to_numpy(dtype=float)
        left_selected = int(np.argmin(left_prediction))
        for right in models[left_index + 1 :]:
            right_prediction = frame[f"prediction::{right}"].to_numpy(dtype=float)
            right_selected = int(np.argmin(right_prediction))
            record: dict[str, Any] = {
                "target": target,
                "model_set": model_set,
                "left_model": left,
                "right_model": right,
                "prediction_spearman": float(spearmanr(left_prediction, right_prediction).statistic),
                "mean_absolute_prediction_difference": float(np.mean(np.abs(left_prediction - right_prediction))),
                "max_absolute_prediction_difference": float(np.max(np.abs(left_prediction - right_prediction))),
                "same_argmin": left_selected == right_selected,
                "left_selected_regret": float(observed[left_selected] - best_observed),
                "right_selected_regret": float(observed[right_selected] - best_observed),
                "absolute_selected_regret_gap": float(abs(observed[left_selected] - observed[right_selected])),
            }
            for k in TOP_K:
                left_top = top_indices(frame, left, k)
                right_top = top_indices(frame, right, k)
                record[f"top_{k}_jaccard"] = len(left_top & right_top) / len(left_top | right_top)
            output.append(record)
    return output


def support_stratified_rows(frame: pd.DataFrame, target: str, models: list[str]) -> list[dict[str, Any]]:
    output = []
    for model in models:
        prediction_column = f"prediction::{model}"
        for support_quartile, group in frame.groupby("support_quartile", observed=True, sort=False):
            summary, _ = metrics(group["observed"], group[prediction_column])
            output.append(
                {
                    "target": target,
                    "model": model,
                    "support_quartile": str(support_quartile),
                    "mean_support_distance": float(group["support_distance"].mean()),
                    **summary,
                }
            )
    return output


def disagreement_rows(frame: pd.DataFrame, target: str, models: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = frame[[f"prediction::{model}" for model in models]].to_numpy(dtype=float)
    mean_prediction = np.mean(predictions, axis=1)
    disagreement = np.std(predictions, axis=1)
    result = frame[
        [
            "name",
            "policy_class",
            "development_layer",
            "candidate_target",
            "observed",
            "support_distance",
            "phase_tv",
            "aggregate_tv_to_proportional",
            "max_epoch",
        ]
    ].copy()
    result.insert(0, "target", target)
    result["model_set"] = "fit_15pct"
    result["mean_prediction"] = mean_prediction
    result["prediction_disagreement"] = disagreement
    result["absolute_mean_residual"] = np.abs(mean_prediction - result["observed"].to_numpy(dtype=float))
    result["mean_optimism"] = result["observed"].to_numpy(dtype=float) - mean_prediction
    correlations = {
        "target": target,
        "n_models": len(models),
        "spearman_disagreement_abs_error": float(
            spearmanr(result["prediction_disagreement"], result["absolute_mean_residual"]).statistic
        ),
        "spearman_support_disagreement": float(
            spearmanr(result["support_distance"], result["prediction_disagreement"]).statistic
        ),
        "spearman_support_abs_error": float(
            spearmanr(result["support_distance"], result["absolute_mean_residual"]).statistic
        ),
    }
    return result, correlations


def maximum_empty_ball(points: np.ndarray, resolution: int = 401) -> dict[str, float]:
    triangulation = Delaunay(points)
    tree = cKDTree(points)
    axis = np.linspace(0.0, 1.0, resolution)
    x, y = np.meshgrid(axis, axis)
    candidates = np.column_stack([x.ravel(), y.ravel()])
    inside = triangulation.find_simplex(candidates) >= 0
    interior = candidates[inside]
    distances, _ = tree.query(interior, k=1)
    selected = int(np.argmax(distances))
    center = interior[selected]
    nearest_distance = float(distances[selected])
    return {
        "center_phase0": float(center[0]),
        "center_phase1": float(center[1]),
        "nearest_observation_distance": nearest_distance,
        "safe_bump_radius": 0.49 * nearest_distance,
        "grid_resolution": resolution,
    }


def smooth_bump_certificate(bundle: dict[str, Any]) -> pd.DataFrame:
    cosine_rows = bundle["swarms"]["starcoder_cosine"]["rows"]
    cosine_points = np.asarray([[float(row["phase0"][1]), float(row["phase1"][1])] for row in cosine_rows], dtype=float)
    refined_wsd = pd.read_csv(REFINED_WSD80_DATA)
    wsd_points = refined_wsd[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    surfaces = {
        "starcoder_cosine_50_50": cosine_points,
        "starcoder_wsd_80_20_refined": wsd_points,
    }
    rows = []
    for surface_name, points in surfaces.items():
        certificate = maximum_empty_ball(points)
        rows.append(
            {
                "surface": surface_name,
                "n_observations": len(points),
                **certificate,
                "bump_at_every_observation": 0.0,
                "arbitrary_center_depth_possible": True,
            }
        )
    return pd.DataFrame(rows)


def render_pairwise_heatmap(pairwise: pd.DataFrame, path: Path) -> None:
    targets = ["uncheatable", "table9"]
    figure = make_subplots(rows=1, cols=2, subplot_titles=targets, horizontal_spacing=0.12)
    for column, target in enumerate(targets, start=1):
        target_rows = pairwise.loc[pairwise["target"].eq(target) & pairwise["model_set"].eq("fit_15pct")]
        models = sorted(set(target_rows["left_model"]) | set(target_rows["right_model"]))
        matrix = np.eye(len(models))
        index = {model: i for i, model in enumerate(models)}
        for row in target_rows.itertuples(index=False):
            i = index[row.left_model]
            j = index[row.right_model]
            matrix[i, j] = matrix[j, i] = row.top_5_jaccard
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=models,
                y=models,
                zmin=0,
                zmax=1,
                colorscale="RdYlGn",
                colorbar={"title": "Top-5 Jaccard"} if column == 2 else None,
                showscale=column == 2,
                hovertemplate="%{y} vs %{x}<br>Top-5 Jaccard=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title="Fit-near-equivalent surrogates select different 3e18 policies",
        height=620,
        width=1420,
        margin={"l": 170, "r": 80, "t": 90, "b": 170},
        template="plotly_white",
    )
    write_html(figure, path)


def render_support_calibration(support: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: RMSE",
            "Table-9: RMSE",
            "Uncheatable: calibration slope",
            "Table-9: calibration slope",
        ),
        vertical_spacing=0.16,
    )
    order = ["Q1 nearest", "Q2", "Q3", "Q4 farthest"]
    for column, target in enumerate(("uncheatable", "table9"), start=1):
        target_rows = support.loc[support["target"].eq(target)]
        for model, model_rows in target_rows.groupby("model", sort=True):
            indexed = model_rows.set_index("support_quartile").reindex(order)
            figure.add_trace(
                go.Scatter(
                    x=order,
                    y=indexed["rmse"],
                    mode="lines+markers",
                    name=model,
                    legendgroup=model,
                    showlegend=column == 1,
                    hovertemplate=f"{model}<br>%{{x}}<br>RMSE=%{{y:.5f}}<extra></extra>",
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=order,
                    y=indexed["calibration_slope_observed_on_predicted"],
                    mode="lines+markers",
                    name=model,
                    legendgroup=model,
                    showlegend=False,
                    hovertemplate=f"{model}<br>%{{x}}<br>slope=%{{y:.3f}}<extra></extra>",
                ),
                row=2,
                col=column,
            )
        figure.add_hline(y=1.0, line_dash="dash", line_color="#64748b", row=2, col=column)
    figure.update_layout(
        title="Heldout error changes with distance from the 3e18 fit support",
        height=900,
        width=1450,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.17},
        margin={"l": 80, "r": 40, "t": 90, "b": 170},
    )
    write_html(figure, path)


def render_disagreement(disagreement: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"), horizontal_spacing=0.1)
    for column, target in enumerate(("uncheatable", "table9"), start=1):
        target_rows = disagreement.loc[disagreement["target"].eq(target)]
        for policy_class, symbol in (("single_phase", "circle"), ("two_phase", "diamond")):
            group = target_rows.loc[target_rows["policy_class"].eq(policy_class)]
            figure.add_trace(
                go.Scatter(
                    x=group["prediction_disagreement"],
                    y=group["absolute_mean_residual"],
                    mode="markers",
                    name=policy_class.replace("_", " "),
                    legendgroup=policy_class,
                    showlegend=column == 1,
                    marker={
                        "symbol": symbol,
                        "size": 8,
                        "opacity": 0.7,
                        "color": group["support_distance"],
                        "colorscale": "RdYlGn_r",
                        "showscale": column == 2 and policy_class == "two_phase",
                        "colorbar": {"title": "Support distance"},
                        "line": {"width": 0.5, "color": "#0f172a"},
                    },
                    customdata=np.column_stack([group["name"], group["support_distance"], group["observed"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>model disagreement=%{x:.5f}<br>absolute error=%{y:.5f}"
                        "<br>support distance=%{customdata[1]:.3f}<br>observed=%{customdata[2]:.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.update_xaxes(title_text="Prediction SD across fit-near-equivalent models")
    figure.update_yaxes(title_text="Absolute residual of model-set mean")
    figure.update_layout(
        title="Model disagreement is an extrapolation warning, not a surrogate",
        height=650,
        width=1450,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.16},
        margin={"l": 90, "r": 130, "t": 90, "b": 120},
    )
    write_html(figure, path)


def write_report(
    output: Path,
    fit_sets: pd.DataFrame,
    selections: pd.DataFrame,
    pairwise: pd.DataFrame,
    support: pd.DataFrame,
    correlations: pd.DataFrame,
    bumps: pd.DataFrame,
) -> None:
    strict_summary = fit_sets.loc[fit_sets["set"].eq("strict_5pct")]
    selection_summary = selections.loc[selections["model_set"].eq("fit_15pct")].copy()
    closest_pair = pairwise.loc[pairwise["model_set"].eq("fit_15pct")].sort_values(
        ["target", "top_5_jaccard", "absolute_selected_regret_gap"]
    )
    farthest_support = support.loc[support["support_quartile"].eq("Q4 farthest")].copy()
    theorem = r"""Let \(S=\{x_1,\ldots,x_n\}\) be the finite set of observed policies and let \(f\) be any smooth surrogate matching their observations. For any \(x_\star\notin S\), choose \(r<\min_i\lVert x_\star-x_i\rVert\) and define the compactly supported \(C^\infty\) bump

$$
b(x) = -A\,\mathbf{1}[\lVert x-x_\star\rVert<r]\exp\!\left(1-\frac{1}{1-(\lVert x-x_\star\rVert/r)^2}\right).
$$

Then \(b(x_i)=0\) for every observation, while \(b(x_\star)=-A\). For sufficiently large \(A\), \(f+b\) has a lower optimum near \(x_\star\) but is observationally indistinguishable from \(f\) on the entire finite design. Therefore neither smoothness nor interpolation identifies the raw optimum. A defensible optimum requires a structural law whose state transition and response link are independently falsified; deployment regularization cannot supply that identification."""
    report = f"""# Round 53: partial-identification and decision-instability audit

## Status

This is a diagnostic result, not a promoted surrogate. No model was fit or retuned against exposed adversarial outcomes, and the running sealed confirmation panel was not read.

## Formal non-identifiability result

{theorem}

The maximum-empty-ball calculation below makes the construction explicit even inside the convex hulls of both dense two-domain surfaces:

{markdown_table(bumps, ["surface", "n_observations", "center_phase0", "center_phase1", "nearest_observation_distance", "safe_bump_radius"])}

The theorem does **not** imply that learning an optimum is impossible. It says the evidence for an optimum comes entirely from accepted mechanistic restrictions and targeted interventions, not from finite-sample fit quality alone.

## Fit-panel equivalence sets

Sets were frozen using only Delphi 3e18 fit-panel OOF RMSE. `strict_5pct` implements the immutable gate; `fit_15pct` is a descriptive Rashomon set used only to quantify decision disagreement.

{markdown_table(strict_summary, ["target", "set", "models", "best_oof_rmse", "threshold_rmse"])}

## Archive decisions

The following models are all within 15% of the best fit-panel OOF RMSE for their target, yet they select different policies and incur different observed regret on the same 710-policy archive:

{markdown_table(selection_summary, ["target", "model", "selected_name", "selected_policy_class", "selected_observed", "selected_predicted", "selected_regret", "selected_optimism", "support_distance", "max_epoch"])}

Low top-5 overlap is direct evidence that fit-near-equivalent surfaces do not identify the same decision:

{markdown_table(closest_pair.groupby("target", as_index=False).first(), ["target", "left_model", "right_model", "prediction_spearman", "top_5_jaccard", "same_argmin", "left_selected_regret", "right_selected_regret"])}

## Support and disagreement

Error and calibration are support-dependent. Metrics on the farthest support quartile are:

{markdown_table(farthest_support, ["target", "model", "n", "mean_support_distance", "rmse", "spearman", "calibration_slope_observed_on_predicted", "regret_at_1", "optimism_gt_0p05_count", "worst_optimism"])}

Model disagreement is correlated with error, but it remains an inadmissible ensemble or output correction. It can only be used as an abstention or experimental-design diagnostic:

{markdown_table(correlations, list(correlations.columns))}

## Consequence for model discovery

The current evidence does not identify a trustworthy raw optimum. The two-domain surfaces are still decisive falsification tests: every newly tested physical transition through Round 52 either misses their shape or places the raw optimum in the wrong ordering regime. On the 39-bucket problem, fit-near-equivalent models select materially different archive policies. This blocks declaring a winner from OOF fit or archive interpolation.

The remaining scientifically defensible path is an identification experiment: hold aggregate exposure fixed and intervene on phase contrast around independently identified one-phase anchors. Such paired phase-fiber observations directly identify the phase transition while avoiding the aggregate/phase confounding present in random two-phase swarms. Until a candidate transition law predicts those untouched interventions, the correct result is negative.

## Artifacts

- `fit_equivalence_sets.csv`: OOF-only model-set definitions.
- `selected_policies.csv`: archive selections and observed regret.
- `pairwise_decision_disagreement.csv`: rank and top-k overlap.
- `support_stratified_metrics.csv`: error and calibration by support quartile.
- `heldout_model_disagreement.csv`: row-level disagreement diagnostics.
- `disagreement_correlations.csv`: warning-signal correlations.
- `smooth_bump_certificate.csv`: finite-design non-identifiability construction.
- `pairwise_top5_jaccard.html`, `support_stratified_calibration.html`, and `model_disagreement_vs_error.html`: interactive visualizations.
"""
    (output / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--frozen-gate", type=Path, default=DEFAULT_FROZEN_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(args.dashboard.read_text())
    baseline_metrics = pd.read_csv(args.frozen_gate / "baseline_metrics.csv")

    fit_set_rows = []
    selected_rows = []
    pairwise_rows = []
    support_rows = []
    disagreement_frames = []
    correlation_rows = []

    for target in ("uncheatable", "table9"):
        frame = heldout_frame(bundle, target)
        strict, descriptive, fit_metrics = fit_oof_model_sets(baseline_metrics, target)
        best = float(fit_metrics["rmse"].min())
        for set_name, models, threshold in (
            ("strict_5pct", strict, STRICT_RELATIVE_OOF_THRESHOLD),
            ("fit_15pct", descriptive, DESCRIPTIVE_RELATIVE_OOF_THRESHOLD),
        ):
            fit_set_rows.append(
                {
                    "target": target,
                    "set": set_name,
                    "models": ", ".join(models),
                    "n_models": len(models),
                    "best_oof_rmse": best,
                    "relative_threshold": threshold,
                    "threshold_rmse": best * (1.0 + threshold),
                }
            )
            selected_rows.extend(selected_policy_rows(frame, target, set_name, models))
            pairwise_rows.extend(pairwise_model_rows(frame, target, set_name, models))
        support_rows.extend(support_stratified_rows(frame, target, descriptive))
        disagreement, correlations = disagreement_rows(frame, target, descriptive)
        disagreement_frames.append(disagreement)
        correlation_rows.append(correlations)

    fit_sets = pd.DataFrame(fit_set_rows)
    selections = pd.DataFrame(selected_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    support = pd.DataFrame(support_rows)
    disagreement = pd.concat(disagreement_frames, ignore_index=True)
    correlations = pd.DataFrame(correlation_rows)
    bumps = smooth_bump_certificate(bundle)

    fit_sets.to_csv(args.output / "fit_equivalence_sets.csv", index=False)
    selections.to_csv(args.output / "selected_policies.csv", index=False)
    pairwise.to_csv(args.output / "pairwise_decision_disagreement.csv", index=False)
    support.to_csv(args.output / "support_stratified_metrics.csv", index=False)
    disagreement.to_csv(args.output / "heldout_model_disagreement.csv", index=False)
    correlations.to_csv(args.output / "disagreement_correlations.csv", index=False)
    bumps.to_csv(args.output / "smooth_bump_certificate.csv", index=False)

    render_pairwise_heatmap(pairwise, args.output / "pairwise_top5_jaccard.html")
    render_support_calibration(support, args.output / "support_stratified_calibration.html")
    render_disagreement(disagreement, args.output / "model_disagreement_vs_error.html")
    write_report(args.output, fit_sets, selections, pairwise, support, correlations, bumps)

    print(f"Wrote Round 53 partial-identification audit to {args.output}")


if __name__ == "__main__":
    main()
