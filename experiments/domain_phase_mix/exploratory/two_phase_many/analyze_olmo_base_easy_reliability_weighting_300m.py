# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Reliability-weighted OLMoBaseEval Easy Table-9 DSP diagnostics.

The headline target remains the unweighted 51-component OLMoBaseEval Easy
Table-9 BPB macro. This script asks whether domain-ablation reliability
diagnostics are useful as a denoising/training signal for Effective-exposure
DSP, without redefining the evaluation objective.

Inputs:
- the 280-row paper-faithful Table-9 fit panel;
- the domain-deletion p-value matrix calibrated against 11 proportional repeats.

Outputs:
- component crosswalk and reliability tables;
- weighted-target fit summaries evaluated against both the weighted target and
  the unweighted Table-9 macro;
- plots for reliability weights, lambda sweeps, and OOF prediction quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIT_PANEL = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "fit_panel_table9_macro.csv"
)
DEFAULT_PVALUE_CELLS = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_domain_ablation_pvalue_matrix_20260625"
    / "smooth_benchmark_deleted_domain_pvalue_matrix_cells.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_reliability_weighting_20260625"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COMPONENT_TARGET = "table9_macro_bpb"
COMPONENT_METRIC = "mean_51_table9_bpb_components"
STUDENT_DF = 10


@dataclass(frozen=True)
class MethodSummary:
    method: str
    statistic: str
    lambda_value: float
    linear_reg: float
    n_rows: int
    n_components: int
    min_component_weight: float
    max_component_weight: float
    q05_component_weight: float
    q95_component_weight: float
    effective_component_count: float
    weighted_target_train_rmse: float
    weighted_target_train_spearman: float
    weighted_target_oof_rmse: float
    weighted_target_oof_spearman: float
    headline_train_rmse: float
    headline_train_spearman: float
    headline_oof_rmse: float
    headline_oof_spearman: float
    headline_fold_mean_regret_at_1: float
    headline_lower_tail_optimism: float
    headline_low_tail_rmse: float
    headline_predicted_best_run_name: str
    headline_predicted_best_observed_value: float
    headline_predicted_best_predicted_value: float
    headline_best_observed_run_name: str
    headline_best_observed_value: float
    headline_selection_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--pvalue-cells", type=Path, default=DEFAULT_PVALUE_CELLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--statistics",
        default="harm_t_excess,two_sided_t_excess,harm_bonferroni,two_sided_bonferroni,inverse_variance",
        help="Comma-separated reliability statistics to try.",
    )
    parser.add_argument("--lambda-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--linear-reg", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    return parser.parse_args()


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def component_columns(panel: pd.DataFrame) -> list[str]:
    metadata = {"run_name", "source_experiment", "panel_source", COMPONENT_TARGET}
    components = [col for col in panel.columns if col not in metadata and not col.startswith("phase_")]
    if len(components) != 51:
        raise ValueError(f"Expected 51 Table-9 components, found {len(components)}")
    return components


def component_source_keys(component: str) -> dict[str, float]:
    if component in paper_olmix.MMLU_CATEGORY_WEIGHTS:
        return {
            paper_olmix.mmlu_metric_key(task): float(weight)
            for task, weight in paper_olmix.MMLU_CATEGORY_WEIGHTS[component].items()
        }
    return {component: 1.0}


def aggregate_component_cells(
    cells: pd.DataFrame,
    component: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source_weights = component_source_keys(component)
    source_rows: list[pd.DataFrame] = []
    crosswalk_rows: list[dict[str, Any]] = []
    for source_key, source_weight in source_weights.items():
        view = cells.loc[cells["benchmark_key"].eq(source_key)].copy()
        if view.empty:
            raise ValueError(f"Missing p-value matrix rows for component {component} source {source_key}")
        if int(view["target_domain"].nunique()) != 39:
            raise ValueError(f"Expected 39 deletion rows for {source_key}, found {view['target_domain'].nunique()}")
        view["source_weight"] = float(source_weight)
        source_rows.append(view)
        crosswalk_rows.append(
            {
                "component": component,
                "source_key": source_key,
                "source_weight": float(source_weight),
                "aggregation": "mmlu_weighted" if component in paper_olmix.MMLU_CATEGORY_WEIGHTS else "exact",
            }
        )
    merged = pd.concat(source_rows, ignore_index=True)

    if len(source_weights) == 1:
        out = merged[
            [
                "target_domain",
                "domain_deletion_utility_delta",
                "predictive_sd",
                "t_statistic",
                "p_harm",
                "p_two_sided",
            ]
        ].copy()
        out["component"] = component
        out["component_utility_delta"] = pd.to_numeric(out["domain_deletion_utility_delta"], errors="raise")
        out["component_predictive_sd"] = pd.to_numeric(out["predictive_sd"], errors="raise")
        out["component_t_statistic"] = pd.to_numeric(out["t_statistic"], errors="raise")
        out["component_p_harm"] = pd.to_numeric(out["p_harm"], errors="raise")
        out["component_p_two_sided"] = pd.to_numeric(out["p_two_sided"], errors="raise")
        return out, crosswalk_rows

    # MMLU component reliability is reconstructed from leaf deletion deltas.
    # This treats proportional repeat errors for leaves as independent; the
    # approximation is recorded in component_crosswalk.csv and should be read
    # as a diagnostic denominator, not a new benchmark estimator.
    rows: list[dict[str, Any]] = []
    grouped = merged.groupby("target_domain", sort=True)
    for target_domain, view in grouped:
        if int(view["benchmark_key"].nunique()) != len(source_weights):
            raise ValueError(f"Incomplete MMLU source coverage for {component}/{target_domain}")
        source_weight = pd.to_numeric(view["source_weight"], errors="raise").to_numpy(dtype=float)
        delta = pd.to_numeric(view["domain_deletion_utility_delta"], errors="raise").to_numpy(dtype=float)
        predictive_sd = pd.to_numeric(view["predictive_sd"], errors="raise").to_numpy(dtype=float)
        component_delta = float(np.sum(source_weight * delta))
        component_sd = float(np.sqrt(np.sum((source_weight * predictive_sd) ** 2)))
        component_t = component_delta / component_sd if component_sd > 0.0 else np.nan
        p_harm = float(student_t.cdf(component_t, df=STUDENT_DF))
        p_two_sided = float(2.0 * min(p_harm, 1.0 - p_harm))
        rows.append(
            {
                "target_domain": target_domain,
                "component": component,
                "component_utility_delta": component_delta,
                "component_predictive_sd": component_sd,
                "component_t_statistic": component_t,
                "component_p_harm": p_harm,
                "component_p_two_sided": p_two_sided,
            }
        )
    return pd.DataFrame(rows), crosswalk_rows


def component_reliability_table(cells: pd.DataFrame, components: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    crosswalk: list[dict[str, Any]] = []
    null_t2 = STUDENT_DF / (STUDENT_DF - 2.0)
    null_harm_t2 = null_t2 / 2.0
    for component in components:
        view, component_crosswalk = aggregate_component_cells(cells, component)
        crosswalk.extend(component_crosswalk)
        t_stats = pd.to_numeric(view["component_t_statistic"], errors="raise").to_numpy(dtype=float)
        p_harm = pd.to_numeric(view["component_p_harm"], errors="raise").to_numpy(dtype=float)
        p_two_sided = pd.to_numeric(view["component_p_two_sided"], errors="raise").to_numpy(dtype=float)
        pred_sd = pd.to_numeric(view["component_predictive_sd"], errors="raise").to_numpy(dtype=float)
        mean_t2 = float(np.mean(t_stats * t_stats))
        harm_t2 = float(np.mean(np.minimum(t_stats, 0.0) ** 2))
        two_sided_t_excess = max(0.0, (mean_t2 - null_t2) / mean_t2) if mean_t2 > 0.0 else 0.0
        harm_t_excess = max(0.0, (harm_t2 - null_harm_t2) / harm_t2) if harm_t2 > 0.0 else 0.0
        harm_bonferroni = 1.0 - min(1.0, float(len(p_harm)) * float(np.min(p_harm)))
        two_sided_bonferroni = 1.0 - min(1.0, float(len(p_two_sided)) * float(np.min(p_two_sided)))
        rows.append(
            {
                "component": component,
                "n_deletions": int(len(view)),
                "n_sources": int(len(component_crosswalk)),
                "mean_predictive_sd": float(np.sqrt(np.mean(pred_sd * pred_sd))),
                "inverse_variance": float(1.0 / max(float(np.mean(pred_sd * pred_sd)), 1e-18)),
                "min_p_harm": float(np.min(p_harm)),
                "min_p_two_sided": float(np.min(p_two_sided)),
                "mean_t2": mean_t2,
                "mean_harm_t2": harm_t2,
                "max_abs_t": float(np.max(np.abs(t_stats))),
                "n_raw_harm_p_lt_0p05": int(np.sum(p_harm < 0.05)),
                "n_raw_two_sided_p_lt_0p05": int(np.sum(p_two_sided < 0.05)),
                "n_bonferroni_harm_p_lt_0p05": int(np.sum(p_harm * len(p_harm) < 0.05)),
                "harm_t_excess": float(harm_t_excess),
                "two_sided_t_excess": float(two_sided_t_excess),
                "harm_bonferroni": float(harm_bonferroni),
                "two_sided_bonferroni": float(two_sided_bonferroni),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(crosswalk)


def component_weights(reliability: pd.DataFrame, statistic: str, lambda_value: float) -> np.ndarray:
    if statistic == "uniform":
        raw = np.ones(len(reliability), dtype=float)
    elif statistic == "inverse_variance":
        raw = pd.to_numeric(reliability["inverse_variance"], errors="raise").to_numpy(dtype=float)
        raw = raw / np.mean(raw)
    else:
        if statistic not in reliability.columns:
            raise ValueError(f"Unknown statistic {statistic}")
        raw = pd.to_numeric(reliability[statistic], errors="raise").to_numpy(dtype=float)
        raw = np.clip(raw, 0.0, None)
    blended = (1.0 - lambda_value) + lambda_value * raw
    if np.any(blended < 0.0) or not np.isfinite(blended).all() or float(np.sum(blended)) <= 0.0:
        raise ValueError(f"Invalid component weights for {statistic} lambda={lambda_value}")
    return blended / np.mean(blended)


def fit_effective_exposure(
    *,
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    target_name: str,
    linear_reg: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], dsp.FittedDSPModel]:
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, target_name)
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        model, _tuning = dsp.fit_variant(
            packet,
            dsp.VARIANTS["effective_exposure"],
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        train_prediction = dsp.predict(model, packet.w)
        oof_prediction, folds = top_level_dsp.fit_dsp_oof_predictions(packet, model)
    finally:
        dsp.LINEAR_REG = original_linear_reg
    return train_prediction, oof_prediction, folds, model


def regression_spearman(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(pd.Series(y).corr(pd.Series(y_hat), method="spearman"))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def summarize_method(
    *,
    method: str,
    statistic: str,
    lambda_value: float,
    linear_reg: float,
    weights: np.ndarray,
    panel: pd.DataFrame,
    target_name: str,
    train_prediction: np.ndarray,
    oof_prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> MethodSummary:
    weighted_target = pd.to_numeric(panel[target_name], errors="raise").to_numpy(dtype=float)
    headline = pd.to_numeric(panel[COMPONENT_TARGET], errors="raise").to_numpy(dtype=float)
    weighted_oof = base.predictive_diagnostics(weighted_target, oof_prediction, folds)
    headline_oof = base.predictive_diagnostics(headline, oof_prediction, folds)
    best_idx = int(np.argmin(headline))
    predicted_best_idx = int(np.argmin(oof_prediction))
    normalized = weights / np.sum(weights)
    effective_count = float(1.0 / np.sum(normalized * normalized))
    return MethodSummary(
        method=method,
        statistic=statistic,
        lambda_value=float(lambda_value),
        linear_reg=float(linear_reg),
        n_rows=int(len(panel)),
        n_components=int(len(weights)),
        min_component_weight=float(np.min(weights)),
        max_component_weight=float(np.max(weights)),
        q05_component_weight=float(np.quantile(weights, 0.05)),
        q95_component_weight=float(np.quantile(weights, 0.95)),
        effective_component_count=effective_count,
        weighted_target_train_rmse=rmse(weighted_target, train_prediction),
        weighted_target_train_spearman=regression_spearman(weighted_target, train_prediction),
        weighted_target_oof_rmse=float(weighted_oof["rmse"]),
        weighted_target_oof_spearman=float(weighted_oof["spearman"]),
        headline_train_rmse=rmse(headline, train_prediction),
        headline_train_spearman=regression_spearman(headline, train_prediction),
        headline_oof_rmse=float(headline_oof["rmse"]),
        headline_oof_spearman=float(headline_oof["spearman"]),
        headline_fold_mean_regret_at_1=float(headline_oof["fold_mean_regret_at_1"]),
        headline_lower_tail_optimism=float(headline_oof["lower_tail_optimism"]),
        headline_low_tail_rmse=float(headline_oof["low_tail_rmse"]),
        headline_predicted_best_run_name=str(panel.iloc[predicted_best_idx]["run_name"]),
        headline_predicted_best_observed_value=float(headline[predicted_best_idx]),
        headline_predicted_best_predicted_value=float(oof_prediction[predicted_best_idx]),
        headline_best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        headline_best_observed_value=float(headline[best_idx]),
        headline_selection_score=float(headline_oof["rmse"] + 0.5 * max(float(headline_oof["lower_tail_optimism"]), 0.0)),
    )


def write_weight_plots(output_dir: Path, reliability: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "One-sided harm vs two-sided sensitivity",
            "Reliability vs inverse variance",
            "Raw significant deletion counts",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=reliability["harm_t_excess"],
            y=reliability["two_sided_t_excess"],
            mode="markers+text",
            text=reliability["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False),
            textposition="top center",
            marker={"color": reliability["mean_predictive_sd"], "colorscale": "RdYlGn_r", "colorbar": {"title": "noise sd"}},
            hovertemplate="%{text}<br>harm=%{x:.3f}<br>two-sided=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reliability["two_sided_t_excess"],
            y=reliability["inverse_variance"],
            mode="markers",
            text=reliability["component"],
            hovertemplate="%{text}<br>two-sided=%{x:.3f}<br>inv-var=%{y:.3g}<extra></extra>",
            marker_color="#2f6f4e",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=reliability["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False),
            y=reliability["n_raw_harm_p_lt_0p05"],
            marker_color="#c03b2b",
            name="one-sided p_harm < 0.05",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=reliability["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False),
            y=reliability["n_raw_two_sided_p_lt_0p05"],
            marker_color="#2f5d8a",
            name="two-sided p < 0.05",
        ),
        row=1,
        col=3,
    )
    fig.update_layout(
        title="OLMoBaseEval Table-9 component reliability diagnostics",
        height=780,
        width=1800,
        barmode="group",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="harm_t_excess", row=1, col=1)
    fig.update_yaxes(title_text="two_sided_t_excess", row=1, col=1)
    fig.update_xaxes(title_text="two_sided_t_excess", row=1, col=2)
    fig.update_yaxes(title_text="inverse variance", type="log", row=1, col=2)
    fig.update_xaxes(tickangle=65, row=1, col=3)
    fig.update_yaxes(title_text="domains", row=1, col=3)
    fig.write_html(output_dir / "component_reliability_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_method_plots(output_dir: Path, summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Headline OOF Spearman",
            "Headline OOF RMSE",
            "Regret at predicted best",
            "Lower-tail optimism",
        ),
    )
    for method, group in summary.sort_values("lambda_value").groupby("method", sort=True):
        x = group["lambda_value"]
        fig.add_trace(go.Scatter(x=x, y=group["headline_oof_spearman"], mode="lines+markers", name=method), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x, y=group["headline_oof_rmse"], mode="lines+markers", name=method, showlegend=False),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=x, y=group["headline_fold_mean_regret_at_1"], mode="lines+markers", name=method, showlegend=False),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=group["headline_lower_tail_optimism"], mode="lines+markers", name=method, showlegend=False),
            row=2,
            col=2,
        )
    fig.update_layout(
        title="Reliability-weighted Effective-exposure DSP: evaluated on unweighted Table-9 macro",
        height=900,
        width=1500,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="reliability blend λ")
    fig.write_html(output_dir / "lambda_sweep_headline_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    best = summary.sort_values(["headline_selection_score", "headline_oof_rmse"], ascending=[True, True]).iloc[0]
    best_predictions = predictions.loc[predictions["method"].eq(best["method"])]
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=best_predictions[COMPONENT_TARGET],
            y=best_predictions["oof_prediction"],
            mode="markers",
            text=best_predictions["run_name"],
            marker={
                "color": best_predictions["panel_source"].map({"qsplit_signal": "#335c81", "domain_deletion": "#c75035"}),
                "size": 9,
                "line": {"color": "white", "width": 0.7},
            },
            hovertemplate="%{text}<br>observed=%{x:.5f}<br>oof=%{y:.5f}<extra></extra>",
        )
    )
    lo = float(min(best_predictions[COMPONENT_TARGET].min(), best_predictions["oof_prediction"].min()))
    hi = float(max(best_predictions[COMPONENT_TARGET].max(), best_predictions["oof_prediction"].max()))
    fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#555"}, showlegend=False))
    fig2.update_layout(
        title=f"Best reliability variant: {best['method']} OOF prediction vs unweighted macro",
        xaxis_title="Observed unweighted Table-9 macro BPB",
        yaxis_title="OOF prediction from weighted-target DSP",
        height=760,
        width=900,
        template="plotly_white",
    )
    fig2.write_html(output_dir / "best_reliability_variant_oof_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    rows = summary.sort_values(["headline_selection_score", "headline_oof_rmse"], ascending=[True, True]).head(12)
    lines = [
        "# OLMoBaseEval Easy reliability-weighted DSP investigation",
        "",
        "Headline evaluation remains the unweighted 51-component Table-9 BPB macro.",
        "Weighted targets are training-time denoising probes, not replacement objectives.",
        "",
        "## Top methods by headline selection score",
        "",
        "| method | λ | OOF Spearman | OOF RMSE | regret@1 | lower-tail optimism | predicted best | actual BPB |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.lambda_value:.2f} | {row.headline_oof_spearman:.4f} | "
            f"{row.headline_oof_rmse:.6f} | {row.headline_fold_mean_regret_at_1:.6f} | "
            f"{row.headline_lower_tail_optimism:.6f} | `{row.headline_predicted_best_run_name}` | "
            f"{row.headline_predicted_best_observed_value:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- MMLU category reliability denominators are reconstructed from leaf metrics and assume independent proportional-repeat noise across leaves.",
            "- Reliability weights use global deletion diagnostics, so this is an exploratory upper-bound screen; fold-local reliability should be added before treating gains as final.",
            "- One-sided statistics emphasize deletion harm; two-sided statistics measure any detectable sensitivity.",
            "",
            "## Artifacts",
            "",
            "- `component_crosswalk.csv`",
            "- `component_reliability.csv`",
            "- `method_summary.csv`",
            "- `method_predictions.csv`",
            "- `component_reliability_diagnostics.html`",
            "- `lambda_sweep_headline_metrics.html`",
            "- `best_reliability_variant_oof_scatter.html`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.fit_panel)
    cells = pd.read_csv(args.pvalue_cells)
    components = component_columns(panel)
    reliability, crosswalk = component_reliability_table(cells, components)
    reliability = reliability.set_index("component").loc[components].reset_index()
    crosswalk.to_csv(args.output_dir / "component_crosswalk.csv", index=False)
    reliability.to_csv(args.output_dir / "component_reliability.csv", index=False)
    write_weight_plots(args.output_dir, reliability)

    _signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    statistics = parse_csv_strings(args.statistics)
    lambda_values = parse_csv_floats(args.lambda_values)
    if 0.0 not in lambda_values:
        lambda_values = [0.0, *lambda_values]

    summaries: list[MethodSummary] = []
    prediction_frames: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    seen_methods: set[str] = set()
    for statistic in ["uniform", *statistics]:
        for lambda_value in lambda_values:
            if statistic == "uniform" and lambda_value != 0.0:
                continue
            if statistic != "uniform" and lambda_value == 0.0:
                continue
            method = "uniform" if lambda_value == 0.0 else f"{statistic}_lambda_{lambda_value:g}".replace(".", "p")
            if method in seen_methods:
                continue
            seen_methods.add(method)
            weights = component_weights(reliability, statistic, lambda_value)
            target_name = f"target_{method}"
            target_value = panel[components].astype(float).to_numpy() @ weights / float(np.sum(weights))
            fit_panel = panel.assign(**{target_name: target_value})
            weight_rows.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "statistic": statistic,
                        "lambda_value": float(lambda_value),
                        "component": components,
                        "component_weight": weights,
                    }
                )
            )
            print(f"Fitting {method} with LINEAR_REG={args.linear_reg:g}", flush=True)
            train_prediction, oof_prediction, folds, _model = fit_effective_exposure(
                panel=fit_panel,
                columns=columns,
                domains=domains,
                token_counts=token_counts,
                target_name=target_name,
                linear_reg=args.linear_reg,
                args=args,
            )
            summary = summarize_method(
                method=method,
                statistic=statistic,
                lambda_value=lambda_value,
                linear_reg=args.linear_reg,
                weights=weights,
                panel=fit_panel,
                target_name=target_name,
                train_prediction=train_prediction,
                oof_prediction=oof_prediction,
                folds=folds,
            )
            summaries.append(summary)
            pred = fit_panel[["run_name", "source_experiment", "panel_source", COMPONENT_TARGET, target_name]].copy()
            pred["method"] = method
            pred["statistic"] = statistic
            pred["lambda_value"] = float(lambda_value)
            pred["train_prediction"] = train_prediction
            pred["oof_prediction"] = oof_prediction
            pred["train_residual_vs_headline"] = train_prediction - fit_panel[COMPONENT_TARGET].to_numpy(dtype=float)
            pred["oof_residual_vs_headline"] = oof_prediction - fit_panel[COMPONENT_TARGET].to_numpy(dtype=float)
            prediction_frames.append(pred)

    summary_frame = pd.DataFrame([asdict(summary) for summary in summaries])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    component_weight_grid = pd.concat(weight_rows, ignore_index=True)
    summary_frame.to_csv(args.output_dir / "method_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "method_predictions.csv", index=False)
    component_weight_grid.to_csv(args.output_dir / "component_weight_grid.csv", index=False)
    topk = (
        predictions.sort_values(["method", "oof_prediction"], ascending=[True, True])
        .groupby("method", sort=False)
        .head(10)
        .copy()
    )
    topk["rank_within_method"] = topk.groupby("method").cumcount() + 1
    topk.to_csv(args.output_dir / "top10_predictions_by_method.csv", index=False)
    write_method_plots(args.output_dir, summary_frame, predictions)
    write_report(args.output_dir, summary_frame)
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                "fit_panel": str(args.fit_panel),
                "pvalue_cells": str(args.pvalue_cells),
                "linear_reg": args.linear_reg,
                "maxiter": args.maxiter,
                "coarse_top_k": args.coarse_top_k,
                "basin_hopping_iters": args.basin_hopping_iters,
                "statistics": statistics,
                "lambda_values": lambda_values,
                "note": "Headline evaluation is unweighted table9_macro_bpb; weighted targets are denoising probes.",
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
