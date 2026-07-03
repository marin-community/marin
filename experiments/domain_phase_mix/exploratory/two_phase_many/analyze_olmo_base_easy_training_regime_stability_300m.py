# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Training-regime and selection-stability diagnostics for OLMoBaseEval Easy DSP.

This script follows the Issue 6665 plan after first-pass reliability weighting:

1. Compare full deletion-augmented fitting against qsplit-only fitting.
2. Treat domain-deletion rows as a held-out stress test for qsplit-only models.
3. Estimate selected-mixture stability under empirical residual bootstraps.
4. Surface poorly modeled components before adding more decision complexity.

The headline target remains the unweighted 51-component Table-9 macro BPB.
Domain-ablation p-values are used only as diagnostics in this pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import entropy as scipy_entropy
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as pc,
)
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
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_training_regime_stability_300m_20260626"
DEFAULT_FIT_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "fit_panel_table9_macro.csv"
)
DEFAULT_FULL_PER_COMPONENT = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_per_component_dsp_decision_300m_20260626"
    / "per_component_oof_predictions_linear_reg_0.001.csv"
)
DEFAULT_FULL_AGGREGATE_DSP = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_predictions.csv"
)
DEFAULT_RELIABILITY = (
    REFERENCE_OUTPUTS / "olmo_base_easy_reliability_weighting_20260625" / "component_reliability.csv"
)

MACRO_TARGET = "table9_macro_bpb"
CV_SEED = 0
N_SPLITS = 5
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class RegimeDecision:
    method: str
    family: str
    train_regime: str
    eval_subset: str
    prediction_convention: str
    n_rows: int
    rmse: float
    pearson: float
    spearman: float
    regret_at_1: float
    regret_at_3: float
    regret_at_5: float
    selected_run_name: str
    selected_actual_bpb: float
    selected_prediction: float
    selected_actual_rank: int
    best_observed_run_name: str
    best_observed_bpb: float
    post_selection_optimism: float


@dataclass(frozen=True)
class BootstrapStability:
    method: str
    family: str
    train_regime: str
    eval_subset: str
    n_bootstrap: int
    residual_sd: float
    unique_selected_count: int
    selection_entropy: float
    top_selected_run_name: str
    top_selected_frequency: int
    top_selected_probability: float
    top_selected_actual_rank: int
    nominal_selected_run_name: str
    nominal_selected_actual_rank: int


@dataclass(frozen=True)
class ComponentStress:
    component: str
    qsplit_oof_spearman: float
    qsplit_oof_rmse: float
    deletion_heldout_spearman: float
    deletion_heldout_rmse: float
    deletion_bias: float
    full_panel_oof_spearman: float
    full_panel_oof_rmse: float
    two_sided_t_excess: float
    mean_predictive_sd: float
    stress_failure_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--full-per-component-predictions", type=Path, default=DEFAULT_FULL_PER_COMPONENT)
    parser.add_argument("--full-aggregate-dsp-predictions", type=Path, default=DEFAULT_FULL_AGGREGATE_DSP)
    parser.add_argument("--component-reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--linear-reg", type=float, default=0.001)
    parser.add_argument("--full-aggregate-linear-reg", type=float, default=0.0001)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--component-limit", type=int, default=None)
    return parser.parse_args()


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    residual = pred - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    pearson = float(pearsonr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    spearman = float(spearmanr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    return rmse, pearson, spearman


def subset_packet(packet: dsp.PacketData, row_indices: np.ndarray) -> dsp.PacketData:
    return dsp.PacketData(
        frame=packet.frame.iloc[row_indices].reset_index(drop=True),
        name_col=packet.name_col,
        y=packet.y[row_indices],
        w=packet.w[row_indices],
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=list(packet.domain_names),
    )


def train_subset_oof(packet: dsp.PacketData, model: dsp.FittedDSPModel) -> np.ndarray:
    folds = base.kfold_indices(len(packet.y), n_splits=N_SPLITS, seed=CV_SEED)
    out = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fold_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        out[test_idx] = dsp.predict(fold_model, packet.w[test_idx])
    return out


def fit_effective_exposure_on_subset(
    *,
    packet_full: dsp.PacketData,
    train_indices: np.ndarray,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[dsp.FittedDSPModel, np.ndarray, np.ndarray]:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = linear_reg
    try:
        packet_train = subset_packet(packet_full, train_indices)
        model, _tuning = dsp.fit_variant(
            packet_train,
            dsp.VARIANTS["effective_exposure"],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        all_pred = dsp.predict(model, packet_full.w)
        train_oof = train_subset_oof(packet_train, model)
    finally:
        dsp.LINEAR_REG = original_linear_reg
    return model, all_pred, train_oof


def prediction_with_train_oof(
    *,
    all_pred: np.ndarray,
    train_indices: np.ndarray,
    train_oof: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(all_pred, dtype=float).copy()
    pred[train_indices] = train_oof
    return pred


def topk_regret(y: np.ndarray, score: np.ndarray, indices: np.ndarray, k: int) -> float:
    order = indices[np.argsort(score[indices])]
    selected = order[: min(k, len(order))]
    return float(np.min(y[selected]) - np.min(y[indices]))


def summarize_prediction(
    *,
    method: str,
    family: str,
    train_regime: str,
    eval_subset: str,
    prediction_convention: str,
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    indices: np.ndarray,
) -> RegimeDecision:
    rmse, pearson, spearman = regression_metrics(y[indices], pred[indices])
    selected_idx = int(indices[np.argmin(pred[indices])])
    best_idx = int(indices[np.argmin(y[indices])])
    actual_order = indices[np.argsort(y[indices])]
    selected_rank = int(np.flatnonzero(actual_order == selected_idx)[0] + 1)
    return RegimeDecision(
        method=method,
        family=family,
        train_regime=train_regime,
        eval_subset=eval_subset,
        prediction_convention=prediction_convention,
        n_rows=int(len(indices)),
        rmse=rmse,
        pearson=pearson,
        spearman=spearman,
        regret_at_1=topk_regret(y, pred, indices, 1),
        regret_at_3=topk_regret(y, pred, indices, 3),
        regret_at_5=topk_regret(y, pred, indices, 5),
        selected_run_name=str(panel.iloc[selected_idx]["run_name"]),
        selected_actual_bpb=float(y[selected_idx]),
        selected_prediction=float(pred[selected_idx]),
        selected_actual_rank=selected_rank,
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_bpb=float(y[best_idx]),
        post_selection_optimism=float(max(y[selected_idx] - pred[selected_idx], 0.0)),
    )


def summarize_all_subsets(
    *,
    method: str,
    family: str,
    train_regime: str,
    prediction_convention: str,
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
) -> list[RegimeDecision]:
    subset_indices = {
        "qsplit_signal": np.flatnonzero(panel["panel_source"].eq("qsplit_signal").to_numpy(dtype=bool)),
        "domain_deletion": np.flatnonzero(panel["panel_source"].eq("domain_deletion").to_numpy(dtype=bool)),
        "full": np.arange(len(panel), dtype=int),
    }
    return [
        summarize_prediction(
            method=method,
            family=family,
            train_regime=train_regime,
            eval_subset=subset,
            prediction_convention=prediction_convention,
            panel=panel,
            y=y,
            pred=pred,
            indices=indices,
        )
        for subset, indices in subset_indices.items()
    ]


def component_columns(panel: pd.DataFrame) -> list[str]:
    components = paper_olmix.table9_component_order()
    missing = sorted(set(components).difference(panel.columns))
    if missing:
        raise ValueError(f"Fit panel missing components: {missing[:10]}")
    return components


def build_packet(
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    target_name: str,
) -> dsp.PacketData:
    return top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, target_name)


def fit_qsplit_per_component(
    *,
    panel: pd.DataFrame,
    components: list[str],
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    qsplit_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    all_pred = np.zeros((len(panel), len(components)), dtype=float)
    oof_or_heldout = np.zeros_like(all_pred)
    for idx, component in enumerate(components, start=1):
        print(f"qsplit-only component {idx}/{len(components)}: {component}", flush=True)
        packet_full = build_packet(panel, columns, domains, token_counts, component)
        _model, component_all_pred, component_train_oof = fit_effective_exposure_on_subset(
            packet_full=packet_full,
            train_indices=qsplit_indices,
            linear_reg=float(args.linear_reg),
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        component_combined_pred = prediction_with_train_oof(
            all_pred=component_all_pred,
            train_indices=qsplit_indices,
            train_oof=component_train_oof,
        )
        all_pred[:, idx - 1] = component_all_pred
        oof_or_heldout[:, idx - 1] = component_combined_pred
        q_rmse, q_pearson, q_spearman = regression_metrics(packet_full.y[qsplit_indices], component_train_oof)
        deletion_indices = np.flatnonzero(panel["panel_source"].eq("domain_deletion").to_numpy(dtype=bool))
        d_rmse, d_pearson, d_spearman = regression_metrics(
            packet_full.y[deletion_indices],
            component_all_pred[deletion_indices],
        )
        rows.append(
            {
                "component": component,
                "train_regime": "qsplit_only",
                "qsplit_oof_rmse": q_rmse,
                "qsplit_oof_pearson": q_pearson,
                "qsplit_oof_spearman": q_spearman,
                "deletion_heldout_rmse": d_rmse,
                "deletion_heldout_pearson": d_pearson,
                "deletion_heldout_spearman": d_spearman,
                "deletion_bias": float(np.mean(component_all_pred[deletion_indices] - packet_full.y[deletion_indices])),
            }
        )
    return pd.DataFrame(rows), all_pred, oof_or_heldout


def load_full_component_predictions(panel: pd.DataFrame, path: Path, components: list[str]) -> np.ndarray:
    data = pd.read_csv(path)
    merged = panel[["run_name"]].merge(data, on="run_name", how="left", validate="one_to_one")
    pred_cols = [f"pred::{component}" for component in components]
    missing = [col for col in pred_cols if col not in merged.columns]
    if missing:
        raise ValueError(f"Missing full component prediction columns: {missing[:8]}")
    if merged[pred_cols].isna().any().any():
        raise ValueError("Full component predictions have missing rows")
    return merged[pred_cols].to_numpy(dtype=float)


def load_best_full_aggregate_prediction(panel: pd.DataFrame, path: Path, *, linear_reg: float) -> np.ndarray:
    data = pd.read_csv(path)
    view = data[
        data["variant"].eq("effective_exposure") & np.isclose(data["hyperparameter_value"], linear_reg)
    ].copy()
    if view.empty:
        raise ValueError(f"No effective-exposure aggregate DSP predictions found for linear_reg={linear_reg:g}")
    merged = panel[["run_name"]].merge(
        view[["run_name", "oof_prediction"]],
        on="run_name",
        how="left",
        validate="one_to_one",
    )
    if merged["oof_prediction"].isna().any():
        raise ValueError("Missing aggregate DSP predictions")
    return merged["oof_prediction"].to_numpy(dtype=float)


def bootstrap_stability(
    *,
    method: str,
    family: str,
    train_regime: str,
    eval_subset: str,
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    indices: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> BootstrapStability:
    rng = np.random.default_rng(seed)
    residuals = pred[indices] - y[indices]
    selected_counts: dict[int, int] = {}
    for _ in range(n_bootstrap):
        sampled_residual = rng.choice(residuals, size=len(indices), replace=True)
        perturbed_score = pred[indices] + sampled_residual
        selected_local = int(np.argmin(perturbed_score))
        selected_global = int(indices[selected_local])
        selected_counts[selected_global] = selected_counts.get(selected_global, 0) + 1
    count_items = sorted(selected_counts.items(), key=lambda item: item[1], reverse=True)
    top_idx, top_count = count_items[0]
    probabilities = np.asarray([count / n_bootstrap for _idx, count in count_items], dtype=float)
    actual_order = indices[np.argsort(y[indices])]
    top_rank = int(np.flatnonzero(actual_order == top_idx)[0] + 1)
    nominal_idx = int(indices[np.argmin(pred[indices])])
    nominal_rank = int(np.flatnonzero(actual_order == nominal_idx)[0] + 1)
    return BootstrapStability(
        method=method,
        family=family,
        train_regime=train_regime,
        eval_subset=eval_subset,
        n_bootstrap=int(n_bootstrap),
        residual_sd=float(np.std(residuals, ddof=1)),
        unique_selected_count=len(selected_counts),
        selection_entropy=float(scipy_entropy(probabilities)),
        top_selected_run_name=str(panel.iloc[top_idx]["run_name"]),
        top_selected_frequency=int(top_count),
        top_selected_probability=float(top_count / n_bootstrap),
        top_selected_actual_rank=top_rank,
        nominal_selected_run_name=str(panel.iloc[nominal_idx]["run_name"]),
        nominal_selected_actual_rank=nominal_rank,
    )


def build_component_stress(
    *,
    qsplit_component_summary: pd.DataFrame,
    full_component_predictions: np.ndarray,
    panel: pd.DataFrame,
    components: list[str],
    reliability: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[ComponentStress] = []
    qsplit_indices = np.flatnonzero(panel["panel_source"].eq("qsplit_signal").to_numpy(dtype=bool))
    full_values = panel[components].to_numpy(dtype=float)
    for comp_idx, component in enumerate(components):
        full_pred = full_component_predictions[:, comp_idx]
        full_rmse, _full_pearson, full_spearman = regression_metrics(
            full_values[qsplit_indices, comp_idx],
            full_pred[qsplit_indices],
        )
        qrow = qsplit_component_summary[qsplit_component_summary["component"].eq(component)].iloc[0]
        rrow = reliability[reliability["component"].eq(component)].iloc[0]
        stress_score = float(
            max(0.0, 0.75 - float(qrow["qsplit_oof_spearman"]))
            + max(0.0, 0.75 - float(qrow["deletion_heldout_spearman"]))
            + 5.0 * abs(float(qrow["deletion_bias"]))
        )
        rows.append(
            ComponentStress(
                component=component,
                qsplit_oof_spearman=float(qrow["qsplit_oof_spearman"]),
                qsplit_oof_rmse=float(qrow["qsplit_oof_rmse"]),
                deletion_heldout_spearman=float(qrow["deletion_heldout_spearman"]),
                deletion_heldout_rmse=float(qrow["deletion_heldout_rmse"]),
                deletion_bias=float(qrow["deletion_bias"]),
                full_panel_oof_spearman=full_spearman,
                full_panel_oof_rmse=full_rmse,
                two_sided_t_excess=float(rrow["two_sided_t_excess"]),
                mean_predictive_sd=float(rrow["mean_predictive_sd"]),
                stress_failure_score=stress_score,
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def write_method_plots(output_dir: Path, decisions: pd.DataFrame, stability: pd.DataFrame, stress: pd.DataFrame) -> None:
    qsplit = decisions[decisions["eval_subset"].eq("qsplit_signal")].sort_values(["regret_at_3", "rmse"]).copy()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Qsplit Regret@3", "Qsplit RMSE", "Qsplit selected actual rank"),
    )
    fig.add_trace(go.Bar(x=qsplit["method"], y=qsplit["regret_at_3"], marker_color="#2f5d8a"), row=1, col=1)
    fig.add_trace(go.Bar(x=qsplit["method"], y=qsplit["rmse"], marker_color="#c75035"), row=1, col=2)
    fig.add_trace(go.Bar(x=qsplit["method"], y=qsplit["selected_actual_rank"], marker_color="#3d8f5f"), row=1, col=3)
    fig.update_xaxes(tickangle=60)
    fig.update_layout(
        title="OLMoBaseEval Easy qsplit decision diagnostics by training regime",
        template="plotly_white",
        width=1700,
        height=650,
        showlegend=False,
    )
    fig.write_html(output_dir / "training_regime_qsplit_decision_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residual-bootstrap top selection probability", "Unique selected count"),
    )
    fig.add_trace(go.Bar(x=stability["method"], y=stability["top_selected_probability"], marker_color="#2f5d8a"), row=1, col=1)
    fig.add_trace(go.Bar(x=stability["method"], y=stability["unique_selected_count"], marker_color="#8f4775"), row=1, col=2)
    fig.update_xaxes(tickangle=60)
    fig.update_layout(
        title="Qsplit selection stability under empirical residual perturbations",
        template="plotly_white",
        width=1500,
        height=620,
        showlegend=False,
    )
    fig.write_html(output_dir / "qsplit_bootstrap_selection_stability.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    worst = stress.sort_values("stress_failure_score", ascending=False).head(18).copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Qsplit-only component fit vs deletion stress", "Worst stress-failure score"),
    )
    short = stress["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False)
    fig.add_trace(
        go.Scatter(
            x=stress["qsplit_oof_spearman"],
            y=stress["deletion_heldout_spearman"],
            mode="markers+text",
            text=short,
            textposition="top center",
            marker={"color": stress["two_sided_t_excess"], "colorscale": "RdYlGn_r", "size": 10},
            hovertemplate="%{text}<br>qsplit=%{x:.3f}<br>deletion=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=worst["stress_failure_score"],
            y=worst["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False),
            orientation="h",
            marker_color="#c75035",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Component-level qsplit fit and domain-deletion stress diagnostics",
        template="plotly_white",
        width=1700,
        height=900,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Qsplit OOF Spearman", row=1, col=1)
    fig.update_yaxes(title_text="Deletion heldout Spearman", row=1, col=1)
    fig.write_html(output_dir / "component_deletion_stress_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_prediction_artifacts(
    output_dir: Path,
    panel: pd.DataFrame,
    components: list[str],
    *,
    macro_qsplit_combined: np.ndarray,
    qsplit_component_combined: np.ndarray,
    full_aggregate_pred: np.ndarray,
    full_component_predictions: np.ndarray,
) -> None:
    base_columns = panel[["run_name", "source_experiment", "panel_source", MACRO_TARGET]].copy()
    macro = base_columns.copy()
    macro["aggregate_dsp_effective_exposure_qsplit_only"] = macro_qsplit_combined
    macro["aggregate_dsp_effective_exposure_deletion_augmented"] = full_aggregate_pred
    macro["per_component_mean_qsplit_only"] = qsplit_component_combined.mean(axis=1)
    macro["per_component_mean_deletion_augmented"] = full_component_predictions.mean(axis=1)
    macro.to_csv(output_dir / "method_macro_predictions.csv", index=False)

    qsplit_components = base_columns.copy()
    full_components = base_columns.copy()
    for idx, component in enumerate(components):
        qsplit_components[f"pred::{component}"] = qsplit_component_combined[:, idx]
        full_components[f"pred::{component}"] = full_component_predictions[:, idx]
    qsplit_components.to_csv(output_dir / "qsplit_only_component_predictions.csv", index=False)
    full_components.to_csv(output_dir / "deletion_augmented_component_predictions.csv", index=False)


def write_report(output_dir: Path, decisions: pd.DataFrame, stability: pd.DataFrame, stress: pd.DataFrame) -> None:
    qsplit = decisions[decisions["eval_subset"].eq("qsplit_signal")].sort_values(["regret_at_3", "rmse"]).copy()
    deletion = decisions[decisions["eval_subset"].eq("domain_deletion")].sort_values(["rmse", "regret_at_3"]).copy()
    lines = [
        "# OLMoBaseEval Easy training-regime and stability diagnostics",
        "",
        "Headline target remains the unweighted 51-component Table-9 macro BPB.",
        "",
        "The qsplit-only regime fits on `qsplit_signal` rows only and treats domain-deletion rows as a held-out stress test. The proportional row is the existing 11-row reference mean in the 280-row fit panel.",
        "",
        "## Qsplit decision diagnostics",
        "",
        qsplit[
            [
                "method",
                "family",
                "train_regime",
                "rmse",
                "spearman",
                "regret_at_1",
                "regret_at_3",
                "selected_run_name",
                "selected_actual_bpb",
                "selected_actual_rank",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Domain-deletion stress diagnostics",
        "",
        deletion[
            [
                "method",
                "family",
                "train_regime",
                "rmse",
                "spearman",
                "regret_at_3",
                "selected_run_name",
                "selected_actual_rank",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Qsplit bootstrap selection stability",
        "",
        stability[
            [
                "method",
                "train_regime",
                "residual_sd",
                "unique_selected_count",
                "top_selected_run_name",
                "top_selected_probability",
                "top_selected_actual_rank",
                "nominal_selected_run_name",
                "nominal_selected_actual_rank",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Worst component stress failures",
        "",
        stress.sort_values("stress_failure_score", ascending=False)
        .head(12)[
            [
                "component",
                "qsplit_oof_spearman",
                "deletion_heldout_spearman",
                "deletion_bias",
                "full_panel_oof_spearman",
                "two_sided_t_excess",
                "stress_failure_score",
            ]
        ]
        .to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.fit_panel)
    components = component_columns(panel)
    if args.component_limit is not None:
        components = components[: int(args.component_limit)]
        panel[MACRO_TARGET] = panel[components].mean(axis=1)

    reliability = pd.DataFrame({"component": components}).merge(
        pd.read_csv(args.component_reliability),
        on="component",
        how="left",
    )
    if reliability.isna().any().any():
        raise ValueError("Missing reliability rows for components")

    _signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    y = pd.to_numeric(panel[MACRO_TARGET], errors="raise").to_numpy(dtype=float)
    qsplit_indices = np.flatnonzero(panel["panel_source"].eq("qsplit_signal").to_numpy(dtype=bool))

    macro_packet_full = build_packet(panel, columns, domains, token_counts, MACRO_TARGET)
    print("Fitting qsplit-only aggregate Effective-exposure DSP", flush=True)
    _macro_model, macro_all_pred, macro_qsplit_oof = fit_effective_exposure_on_subset(
        packet_full=macro_packet_full,
        train_indices=qsplit_indices,
        linear_reg=float(args.linear_reg),
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
        basin_hopping_iters=int(args.basin_hopping_iters),
    )
    macro_qsplit_combined = prediction_with_train_oof(
        all_pred=macro_all_pred,
        train_indices=qsplit_indices,
        train_oof=macro_qsplit_oof,
    )

    qsplit_component_summary, _qsplit_component_all_pred, qsplit_component_combined = fit_qsplit_per_component(
        panel=panel,
        components=components,
        columns=columns,
        domains=domains,
        token_counts=token_counts,
        qsplit_indices=qsplit_indices,
        args=args,
    )
    qsplit_component_macro = qsplit_component_combined.mean(axis=1)

    full_component_predictions = load_full_component_predictions(panel, args.full_per_component_predictions, components)
    full_component_macro = full_component_predictions.mean(axis=1)
    full_aggregate_pred = load_best_full_aggregate_prediction(
        panel,
        args.full_aggregate_dsp_predictions,
        linear_reg=float(args.full_aggregate_linear_reg),
    )

    decisions: list[RegimeDecision] = []
    for method, family, regime, convention, pred in (
        (
            "aggregate_dsp_effective_exposure_qsplit_only",
            "aggregate_dsp",
            "qsplit_only",
            "qsplit_oof_else_heldout",
            macro_qsplit_combined,
        ),
        (
            "per_component_mean_qsplit_only",
            "per_component_dsp",
            "qsplit_only",
            "qsplit_oof_else_heldout",
            qsplit_component_macro,
        ),
        (
            "aggregate_dsp_effective_exposure_deletion_augmented",
            "aggregate_dsp",
            "qsplit_plus_deletion",
            "full_panel_oof",
            full_aggregate_pred,
        ),
        (
            "per_component_mean_deletion_augmented",
            "per_component_dsp",
            "qsplit_plus_deletion",
            "full_panel_oof",
            full_component_macro,
        ),
    ):
        decisions.extend(
            summarize_all_subsets(
                method=method,
                family=family,
                train_regime=regime,
                prediction_convention=convention,
                panel=panel,
                y=y,
                pred=pred,
            )
        )

    qsplit_stability_indices = qsplit_indices
    stability_rows = [
        bootstrap_stability(
            method=method,
            family=family,
            train_regime=regime,
            eval_subset="qsplit_signal",
            panel=panel,
            y=y,
            pred=pred,
            indices=qsplit_stability_indices,
            n_bootstrap=int(args.bootstrap_samples),
            seed=CV_SEED,
        )
        for method, family, regime, pred in (
            (
                "aggregate_dsp_effective_exposure_qsplit_only",
                "aggregate_dsp",
                "qsplit_only",
                macro_qsplit_combined,
            ),
            (
                "per_component_mean_qsplit_only",
                "per_component_dsp",
                "qsplit_only",
                qsplit_component_macro,
            ),
            (
                "aggregate_dsp_effective_exposure_deletion_augmented",
                "aggregate_dsp",
                "qsplit_plus_deletion",
                full_aggregate_pred,
            ),
            (
                "per_component_mean_deletion_augmented",
                "per_component_dsp",
                "qsplit_plus_deletion",
                full_component_macro,
            ),
        )
    ]

    component_stress = build_component_stress(
        qsplit_component_summary=qsplit_component_summary,
        full_component_predictions=full_component_predictions,
        panel=panel,
        components=components,
        reliability=reliability,
    )
    write_prediction_artifacts(
        args.output_dir,
        panel,
        components,
        macro_qsplit_combined=macro_qsplit_combined,
        qsplit_component_combined=qsplit_component_combined,
        full_aggregate_pred=full_aggregate_pred,
        full_component_predictions=full_component_predictions,
    )

    decision_frame = pd.DataFrame([asdict(row) for row in decisions])
    stability_frame = pd.DataFrame([asdict(row) for row in stability_rows])
    qsplit_component_summary.to_csv(args.output_dir / "qsplit_only_component_dsp_summary.csv", index=False)
    decision_frame.to_csv(args.output_dir / "training_regime_decision_summary.csv", index=False)
    stability_frame.to_csv(args.output_dir / "qsplit_bootstrap_selection_stability.csv", index=False)
    component_stress.to_csv(args.output_dir / "component_deletion_stress_summary.csv", index=False)
    write_method_plots(args.output_dir, decision_frame, stability_frame, component_stress)
    write_report(args.output_dir, decision_frame, stability_frame, component_stress)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "fit_panel": str(args.fit_panel),
                "full_per_component_predictions": str(args.full_per_component_predictions),
                "full_aggregate_dsp_predictions": str(args.full_aggregate_dsp_predictions),
                "component_reliability": str(args.component_reliability),
                "linear_reg": float(args.linear_reg),
                "full_aggregate_linear_reg": float(args.full_aggregate_linear_reg),
                "maxiter": int(args.maxiter),
                "coarse_top_k": int(args.coarse_top_k),
                "basin_hopping_iters": int(args.basin_hopping_iters),
                "bootstrap_samples": int(args.bootstrap_samples),
                "headline_objective": "unweighted 51-component Table-9 macro BPB",
                "qsplit_only_convention": "qsplit rows use OOF predictions; deletion rows use held-out predictions",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(decision_frame[decision_frame["eval_subset"].eq("qsplit_signal")].to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
