# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit per-component effective-exposure DSP models and sweep KL proposals.

This is the DSP analogue of the paper-faithful OLMix baseline: fit one model
per OLMoBaseEval Easy Table-9 BPB component, select the linear-head L2 per
component by OOF diagnostics, then optimize the unweighted mean predicted
component BPB under a KL trust-region penalty.

The fit panel is the 300M deletion-augmented panel used for the current
Table-9 work: 241 ex-ante qsplit/signal rows, 39 domain-deletion rows, and a
proportional target replaced by the 11-row proportional reference mean.
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
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_l2_kl_sweep_deletion_augmented_300m as dsp_sweep,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
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
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
DEFAULT_OLMIX_SUMMARY = (
    REFERENCE_OUTPUTS / "olmo_base_easy_paper_faithful_olmix_300m_20260625" / "summary.csv"
)
DEFAULT_OLMIX_WEIGHTS = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "two_phase_adapted_delta_0p01"
    / "proposed_mixture_weights.csv"
)
DEFAULT_AGGREGATE_DSP_SWEEP = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_kl_sweep_linear_reg_0p0001"
    / "effective_exposure_table9_macro_kl_sweep_summary.csv"
)

PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
CV_SEED = 0
N_SPLITS = 5
MACRO_TARGET = "table9_macro_bpb"


@dataclass(frozen=True)
class SelectedComponent:
    component: str
    selected_linear_reg: float
    selected_oof_rmse: float
    selected_oof_spearman: float
    selected_fold_mean_regret_at_1: float
    selected_lower_tail_optimism: float
    selected_low_tail_rmse: float
    selected_selection_score: float
    train_rmse: float
    train_spearman: float


@dataclass(frozen=True)
class KLSummary:
    model_family: str
    variant: str
    target_metric: str
    kl_reg: float
    predicted_objective: float
    regularized_objective: float
    proportional_actual: float
    proportional_predicted: float
    best_observed_run_name: str
    best_observed_value: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_mean_phase_tv: float
    mean_phase_tv_to_proportional: float
    max_epoch_multiplier: float
    q95_epoch_multiplier: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_weight: float
    min_weight: float
    optimizer_status: str
    n_starts: int
    component_mean_oof_rmse: float
    component_median_oof_rmse: float
    component_mean_oof_spearman: float
    component_median_oof_spearman: float
    macro_train_rmse: float
    macro_train_spearman: float
    macro_oof_rmse: float
    macro_oof_spearman: float
    macro_fold_mean_regret_at_1: float
    macro_lower_tail_optimism: float
    macro_low_tail_rmse: float


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--linear-reg-values", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--kl-reg-values", default="0,0.001,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--component-limit", type=int, default=None)
    parser.add_argument("--olmix-summary", type=Path, default=DEFAULT_OLMIX_SUMMARY)
    parser.add_argument("--olmix-weights", type=Path, default=DEFAULT_OLMIX_WEIGHTS)
    parser.add_argument("--aggregate-dsp-sweep", type=Path, default=DEFAULT_AGGREGATE_DSP_SWEEP)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(".", "p")


def selection_score(row: pd.Series) -> float:
    return float(row["oof_rmse"] + 0.5 * row["lower_tail_optimism"])


def fit_component_for_l2(
    *,
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    component: str,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    model_dir: Path,
) -> tuple[dsp.FittedDSPModel, pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, component)
        model, tuning = dsp.fit_variant(
            packet,
            dsp.VARIANTS["effective_exposure"],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        train_pred = dsp.predict(model, packet.w)
        folds = component_dsp.panel_stratified_folds(panel, n_splits=N_SPLITS, seed=CV_SEED)
        oof_pred = component_dsp.fit_oof_with_folds(packet, model, folds)
        train_rmse, _train_mae, _train_pearson, train_spearman = component_dsp.regression_metrics(packet.y, train_pred)
        oof_rmse, _oof_mae, oof_pearson, oof_spearman = component_dsp.regression_metrics(packet.y, oof_pred)
        oof_diag = base.predictive_diagnostics(packet.y, oof_pred, folds)
        metrics = {
            "linear_reg": float(linear_reg),
            "train_rmse": float(train_rmse),
            "train_spearman": float(train_spearman),
            "oof_rmse": float(oof_rmse),
            "oof_spearman": float(oof_spearman),
            "oof_pearson": float(oof_pearson),
            "fold_mean_regret_at_1": float(oof_diag["fold_mean_regret_at_1"]),
            "lower_tail_optimism": float(oof_diag["lower_tail_optimism"]),
            "low_tail_rmse": float(oof_diag["low_tail_rmse"]),
            "nonlinear_objective": float(tuning["objective"].min()),
            "total_param_count": float(model.total_param_count),
        }
        component_model_dir = model_dir / safe_name(component) / f"linear_reg_{linear_reg:g}"
        component_model_dir.mkdir(parents=True, exist_ok=True)
        tuning.to_csv(component_model_dir / "dsp_tuning.csv", index=False)
        (component_model_dir / "model.json").write_text(
            json.dumps(
                dsp.model_to_json(
                    model,
                    {
                        "component": component,
                        "linear_reg": float(linear_reg),
                        "oof_convention": "full nonlinear geometry, fold linear head",
                    },
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return model, tuning, metrics, train_pred, oof_pred
    finally:
        dsp.LINEAR_REG = original_linear_reg


def fit_all_components(
    *,
    panel: pd.DataFrame,
    components: list[str],
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    linear_regs: list[float],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dsp.FittedDSPModel], pd.DataFrame, np.ndarray, np.ndarray]:
    model_dir = output_dir / "component_models"
    selected_models: list[dsp.FittedDSPModel] = []
    selected_rows: list[SelectedComponent] = []
    all_rows: list[dict[str, Any]] = []
    train_predictions = np.zeros((len(panel), len(components)), dtype=float)
    oof_predictions = np.zeros((len(panel), len(components)), dtype=float)
    for component_idx, component in enumerate(components, start=1):
        print(f"Fitting component {component_idx}/{len(components)}: {component}", flush=True)
        candidates: list[tuple[dsp.FittedDSPModel, dict[str, float], np.ndarray, np.ndarray]] = []
        for linear_reg in linear_regs:
            print(f"  LINEAR_REG={linear_reg:g}", flush=True)
            model, _tuning, metrics, train_pred, oof_pred = fit_component_for_l2(
                panel=panel,
                columns=columns,
                domains=domains,
                token_counts=token_counts,
                component=component,
                linear_reg=float(linear_reg),
                maxiter=int(args.maxiter),
                coarse_top_k=int(args.coarse_top_k),
                basin_hopping_iters=int(args.basin_hopping_iters),
                model_dir=model_dir,
            )
            metrics["component"] = component
            metrics["selection_score"] = selection_score(pd.Series(metrics))
            all_rows.append(metrics)
            candidates.append((model, metrics, train_pred, oof_pred))
        best_model, best_metrics, best_train, best_oof = min(candidates, key=lambda item: item[1]["selection_score"])
        selected_models.append(best_model)
        train_predictions[:, component_idx - 1] = best_train
        oof_predictions[:, component_idx - 1] = best_oof
        selected_rows.append(
            SelectedComponent(
                component=component,
                selected_linear_reg=float(best_metrics["linear_reg"]),
                selected_oof_rmse=float(best_metrics["oof_rmse"]),
                selected_oof_spearman=float(best_metrics["oof_spearman"]),
                selected_fold_mean_regret_at_1=float(best_metrics["fold_mean_regret_at_1"]),
                selected_lower_tail_optimism=float(best_metrics["lower_tail_optimism"]),
                selected_low_tail_rmse=float(best_metrics["low_tail_rmse"]),
                selected_selection_score=float(best_metrics["selection_score"]),
                train_rmse=float(best_metrics["train_rmse"]),
                train_spearman=float(best_metrics["train_spearman"]),
            )
        )
    all_summary = pd.DataFrame(all_rows)
    selected_summary = pd.DataFrame([asdict(row) for row in selected_rows])
    all_summary.to_csv(output_dir / "component_l2_sweep_summary.csv", index=False)
    selected_summary.to_csv(output_dir / "selected_component_l2_summary.csv", index=False)
    write_component_fit_plots(output_dir, all_summary, selected_summary)
    return selected_models, selected_summary, train_predictions, oof_predictions


def predict_component_matrix(models: list[dsp.FittedDSPModel], weights: np.ndarray) -> np.ndarray:
    return np.column_stack([dsp.predict(model, weights) for model in models])


def per_component_objective(
    models: list[dsp.FittedDSPModel],
    weights: np.ndarray,
    natural: np.ndarray,
    kl_reg: float,
) -> float:
    prediction = float(np.mean(predict_component_matrix(models, weights[None, :, :])))
    kl = base.weighted_multiclass_kl(weights, natural, base.PHASE_FRACTIONS)
    return prediction + float(kl_reg) * kl


def optimize_per_component_kl(
    models: list[dsp.FittedDSPModel],
    natural: np.ndarray,
    *,
    kl_reg: float,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, str]:
    m = len(natural)

    def objective(logits: np.ndarray) -> float:
        weights = dsp_sweep.softmax_pair(logits, m)
        return per_component_objective(models, weights, natural, kl_reg)

    best: Any | None = None
    for start_weights in starts:
        result = minimize(
            objective,
            dsp_sweep.weights_to_logits(start_weights),
            method="L-BFGS-B",
            options={"maxiter": 700, "ftol": 1e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Per-component DSP KL optimization failed")
    return dsp_sweep.softmax_pair(np.asarray(best.x, dtype=float), m), float(best.fun), str(best.message)


def mixture_frame(
    *,
    domains: list[str],
    natural: np.ndarray,
    weights: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> pd.DataFrame:
    reference = np.stack([natural, natural], axis=0)
    ratios = weights / np.clip(reference, 1e-12, None)
    sim_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
    frame = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": base.aggregate_phase_weights(weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "phase_0_epoch_multiplier": ratios[0],
            "phase_1_epoch_multiplier": ratios[1],
            "phase_0_delta": weights[0] - natural,
            "phase_1_delta": weights[1] - natural,
        }
    )
    frame["max_abs_delta"] = frame[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    return frame


def summarize_kl(
    *,
    models: list[dsp.FittedDSPModel],
    selected_summary: pd.DataFrame,
    panel: pd.DataFrame,
    components: list[str],
    columns: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    weights: np.ndarray,
    regularized_objective: float,
    optimizer_status: str,
    kl_reg: float,
    train_predictions: np.ndarray,
    oof_predictions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_starts: int,
) -> KLSummary:
    reference = np.stack([natural, natural], axis=0)
    phase_weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(natural))
    observed_macro = panel[components].astype(float).to_numpy().mean(axis=1)
    train_macro = train_predictions.mean(axis=1)
    oof_macro = oof_predictions.mean(axis=1)
    train_rmse, _train_mae, _train_pearson, train_spearman = component_dsp.regression_metrics(observed_macro, train_macro)
    oof_diag = base.predictive_diagnostics(observed_macro, oof_macro, folds)
    distances = base.mean_phase_tv(phase_weights, weights)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(observed_macro))
    prop_rows = panel["run_name"].eq("baseline_proportional")
    if int(prop_rows.sum()) != 1:
        raise ValueError("Expected one baseline_proportional row in fit panel")
    proposed_components = predict_component_matrix(models, weights[None, :, :])[0]
    prop_components = predict_component_matrix(models, reference[None, :, :])[0]
    ratios = weights / np.clip(reference, 1e-12, None)
    sim_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
    return KLSummary(
        model_family="DSP effective_exposure per-component",
        variant=f"per_component_effective_exposure_selected_l2_kl_{kl_reg:g}",
        target_metric="mean_51_table9_bpb_components",
        kl_reg=float(kl_reg),
        predicted_objective=float(np.mean(proposed_components)),
        regularized_objective=float(regularized_objective),
        proportional_actual=float(panel.loc[prop_rows, MACRO_TARGET].iloc[0]),
        proportional_predicted=float(np.mean(prop_components)),
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_value=float(observed_macro[best_idx]),
        nearest_observed_run_name=str(panel.iloc[nearest_idx]["run_name"]),
        nearest_observed_value=float(observed_macro[nearest_idx]),
        nearest_observed_mean_phase_tv=float(distances[nearest_idx]),
        mean_phase_tv_to_proportional=float(0.5 * np.abs(weights - reference).sum(axis=1).mean()),
        max_epoch_multiplier=float(np.max(ratios)),
        q95_epoch_multiplier=float(np.quantile(ratios, 0.95)),
        max_simulated_epoch=float(np.max(sim_epochs)),
        q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
        max_weight=float(np.max(weights)),
        min_weight=float(np.min(weights)),
        optimizer_status=optimizer_status,
        n_starts=int(n_starts),
        component_mean_oof_rmse=float(selected_summary["selected_oof_rmse"].mean()),
        component_median_oof_rmse=float(selected_summary["selected_oof_rmse"].median()),
        component_mean_oof_spearman=float(selected_summary["selected_oof_spearman"].mean()),
        component_median_oof_spearman=float(selected_summary["selected_oof_spearman"].median()),
        macro_train_rmse=float(train_rmse),
        macro_train_spearman=float(train_spearman),
        macro_oof_rmse=float(oof_diag["rmse"]),
        macro_oof_spearman=float(oof_diag["spearman"]),
        macro_fold_mean_regret_at_1=float(oof_diag["fold_mean_regret_at_1"]),
        macro_lower_tail_optimism=float(oof_diag["lower_tail_optimism"]),
        macro_low_tail_rmse=float(oof_diag["low_tail_rmse"]),
    )


def write_component_predictions(
    output_dir: Path,
    panel: pd.DataFrame,
    components: list[str],
    train_predictions: np.ndarray,
    oof_predictions: np.ndarray,
) -> None:
    base_cols = panel[["run_name", "source_experiment", "panel_source", MACRO_TARGET]].copy()
    train = base_cols.copy()
    oof = base_cols.copy()
    for idx, component in enumerate(components):
        train[f"pred::{component}"] = train_predictions[:, idx]
        oof[f"pred::{component}"] = oof_predictions[:, idx]
    train.to_csv(output_dir / "selected_component_train_predictions.csv", index=False)
    oof.to_csv(output_dir / "selected_component_oof_predictions.csv", index=False)


def load_olmix_marker(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[
        frame["variant"].eq("two_phase_adapted")
        & np.isclose(pd.to_numeric(frame["huber_delta"], errors="coerce"), 0.01)
    ].copy()
    if rows.empty:
        rows = frame[frame["variant"].eq("two_phase_adapted")].copy()
    if rows.empty:
        return pd.DataFrame()
    return rows.sort_values(["oof_macro_spearman", "oof_macro_rmse"], ascending=[False, True]).head(1)


def load_olmix_weights(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_aggregate_sweep(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["model_family"] = "DSP effective_exposure aggregate"
    return frame


def write_kl_overlay_plots(
    output_dir: Path,
    per_component: pd.DataFrame,
    aggregate: pd.DataFrame | None,
    olmix_marker: pd.DataFrame,
) -> None:
    overlay_dir = output_dir / "dsp_olmix_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    per_component.to_csv(overlay_dir / "per_component_dsp_kl_sweep_summary.csv", index=False)
    if aggregate is not None:
        aggregate.to_csv(overlay_dir / "aggregate_dsp_kl_sweep_summary.csv", index=False)
    if not olmix_marker.empty:
        olmix_marker.to_csv(overlay_dir / "olmix_two_phase_adapted_marker.csv", index=False)

    per_component_x = per_component["kl_reg"].map(lambda value: f"{float(value):g}")
    aggregate_x = None if aggregate is None else aggregate["kl_reg"].map(lambda value: f"{float(value):g}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=per_component_x,
            y=per_component["predicted_objective"],
            mode="lines+markers",
            name="per-component DSP, selected L2",
            line={"color": "#2563eb", "width": 3},
            hovertemplate="KL=%{x}<br>pred BPB=%{y:.6f}<extra></extra>",
        )
    )
    if aggregate is not None:
        fig.add_trace(
            go.Scatter(
                x=aggregate_x,
                y=aggregate["predicted_objective"],
                mode="lines+markers",
                name="aggregate DSP, L2=1e-4",
                line={"color": "#ef4444", "width": 2, "dash": "dash"},
                hovertemplate="KL=%{x}<br>pred BPB=%{y:.6f}<extra></extra>",
            )
        )
    if not olmix_marker.empty:
        marker = olmix_marker.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[f"{float(marker['kl_reg']):g}"],
                y=[float(marker["predicted_macro_bpb"])],
                mode="markers+text",
                name="paper-faithful OLMix",
                marker={"size": 14, "color": "#111827", "symbol": "diamond"},
                text=["OLMix"],
                textposition="top center",
                hovertemplate="OLMix<br>KL=%{x}<br>pred BPB=%{y:.6f}<extra></extra>",
            )
        )
    fig.update_xaxes(type="category", title_text="KL penalty λ")
    fig.update_layout(
        title="Table-9 macro predicted BPB under KL-regularized mixture search",
        yaxis_title="Predicted mean Table-9 component BPB (lower is better)",
        template="plotly_white",
        width=1100,
        height=700,
    )
    fig.write_html(
        overlay_dir / "table9_macro_kl_predicted_bpb_dsp_olmix_overlay.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    fig2 = go.Figure()
    for column, label, color in (
        ("max_simulated_epoch", "per-component max epoch", "#2563eb"),
        ("q95_simulated_epoch", "per-component q95 epoch", "#60a5fa"),
    ):
        fig2.add_trace(
            go.Scatter(
                x=per_component_x,
                y=per_component[column],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 3 if column == "max_simulated_epoch" else 2},
            )
        )
    if aggregate is not None:
        fig2.add_trace(
            go.Scatter(
                x=aggregate_x,
                y=aggregate["max_simulated_epoch"],
                mode="lines+markers",
                name="aggregate DSP max epoch",
                line={"color": "#ef4444", "width": 2, "dash": "dash"},
            )
        )
    if not olmix_marker.empty:
        marker = olmix_marker.iloc[0]
        fig2.add_trace(
            go.Scatter(
                x=[f"{float(marker['kl_reg']):g}"],
                y=[float(marker["max_simulated_epoch"])],
                mode="markers+text",
                name="paper-faithful OLMix max epoch",
                marker={"size": 14, "color": "#111827", "symbol": "diamond"},
                text=["OLMix"],
                textposition="top center",
            )
        )
    fig2.update_xaxes(type="category", title_text="KL penalty λ")
    fig2.update_layout(
        title="Materialized epochs for KL-regularized Table-9 mixture proposals",
        yaxis_title="Materialized epoch count",
        template="plotly_white",
        width=1100,
        height=700,
    )
    fig2.write_html(
        overlay_dir / "table9_macro_kl_epochs_dsp_olmix_overlay.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=per_component_x,
            y=per_component["mean_phase_tv_to_proportional"],
            mode="lines+markers",
            name="per-component DSP",
            line={"color": "#2563eb", "width": 3},
        )
    )
    if aggregate is not None:
        fig3.add_trace(
            go.Scatter(
                x=aggregate_x,
                y=aggregate["mean_phase_tv_to_proportional"],
                mode="lines+markers",
                name="aggregate DSP",
                line={"color": "#ef4444", "width": 2, "dash": "dash"},
            )
        )
    if not olmix_marker.empty:
        marker = olmix_marker.iloc[0]
        fig3.add_trace(
            go.Scatter(
                x=[f"{float(marker['kl_reg']):g}"],
                y=[float(marker["mean_phase_tv_to_proportional"])],
                mode="markers+text",
                name="paper-faithful OLMix",
                marker={"size": 14, "color": "#111827", "symbol": "diamond"},
                text=["OLMix"],
                textposition="top center",
            )
        )
    fig3.update_xaxes(type="category", title_text="KL penalty λ")
    fig3.update_layout(
        title="Distance from proportional for KL-regularized Table-9 mixture proposals",
        yaxis_title="Mean phase TV to proportional",
        template="plotly_white",
        width=1100,
        height=700,
    )
    fig3.write_html(
        overlay_dir / "table9_macro_kl_tv_dsp_olmix_overlay.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_component_fit_plots(output_dir: Path, all_summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Selected L2 by component", "Selected component fit quality"),
        horizontal_spacing=0.16,
    )
    short = selected["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False).str.replace(
        "/bpb", "", regex=False
    )
    fig.add_trace(
        go.Bar(
            x=selected["selected_linear_reg"].astype(str),
            y=short,
            orientation="h",
            marker_color="#2563eb",
            name="selected L2",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected["selected_oof_rmse"],
            y=short,
            mode="markers",
            marker={"color": selected["selected_oof_spearman"], "colorscale": "RdYlGn_r", "size": 9},
            name="OOF RMSE",
            hovertemplate="component=%{y}<br>OOF RMSE=%{x:.5f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Per-component DSP L2 selection",
        template="plotly_white",
        width=1500,
        height=1350,
        showlegend=False,
    )
    fig.update_xaxes(title_text="LINEAR_REG", row=1, col=1)
    fig.update_xaxes(title_text="OOF RMSE; color is OOF Spearman", row=1, col=2)
    fig.write_html(output_dir / "selected_component_l2_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    best_by_l2 = (
        all_summary.groupby("linear_reg", as_index=False)
        .agg(mean_oof_rmse=("oof_rmse", "mean"), median_oof_rmse=("oof_rmse", "median"), mean_oof_spearman=("oof_spearman", "mean"))
        .sort_values("linear_reg")
    )
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=best_by_l2["linear_reg"].astype(str),
            y=best_by_l2["mean_oof_rmse"],
            mode="lines+markers",
            name="mean component OOF RMSE",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=best_by_l2["linear_reg"].astype(str),
            y=best_by_l2["median_oof_rmse"],
            mode="lines+markers",
            name="median component OOF RMSE",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=best_by_l2["linear_reg"].astype(str),
            y=best_by_l2["mean_oof_spearman"],
            mode="lines+markers",
            name="mean component OOF Spearman",
            yaxis="y2",
        )
    )
    fig2.update_layout(
        title="Per-component DSP L2 grid aggregate diagnostics",
        xaxis_title="LINEAR_REG",
        yaxis_title="Component OOF RMSE",
        yaxis2={"title": "Component OOF Spearman", "overlaying": "y", "side": "right"},
        template="plotly_white",
        width=1000,
        height=650,
    )
    fig2.write_html(output_dir / "component_l2_grid_summary.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_mixture_weights(
    output_dir: Path,
    rows: pd.DataFrame,
    mixtures: dict[float, pd.DataFrame],
) -> None:
    for kl_reg, frame in mixtures.items():
        variant_dir = output_dir / f"kl_{str(kl_reg).replace('.', 'p')}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(variant_dir / "proposed_mixture_weights.csv", index=False)
    focus = [0.025, 0.05, 0.1, 0.2, 0.5]
    available = [value for value in focus if value in mixtures]
    if not available:
        return
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Phase 0 epoch multipliers", "Phase 1 epoch multipliers"),
        vertical_spacing=0.12,
    )
    for kl_reg in available:
        frame = mixtures[kl_reg].sort_values("domain")
        fig.add_trace(
            go.Bar(
                x=frame["domain"],
                y=frame["phase_0_epoch_multiplier"],
                name=f"KL {kl_reg:g}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=frame["domain"],
                y=frame["phase_1_epoch_multiplier"],
                name=f"KL {kl_reg:g}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title="Per-component DSP Table-9 proposals: selected KL epoch multipliers",
        template="plotly_white",
        width=1500,
        height=1050,
        barmode="group",
    )
    fig.update_xaxes(tickangle=65)
    fig.update_yaxes(title_text="Epoch multiplier vs proportional")
    fig.write_html(output_dir / "selected_kl_epoch_multiplier_bars.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    selected: pd.DataFrame,
    kl_summary: pd.DataFrame,
    aggregate: pd.DataFrame | None,
    olmix_marker: pd.DataFrame,
) -> None:
    key = kl_summary.sort_values(["predicted_objective", "kl_reg"]).head(8)
    lines = [
        "# Per-component DSP Table-9 KL sweep",
        "",
        "This analysis fits one effective-exposure DSP model per OLMoBaseEval Easy Table-9 BPB component. Each component chooses `LINEAR_REG` by `OOF RMSE + 0.5 * lower-tail optimism`; the proposal objective is the unweighted mean predicted component BPB plus KL to proportional.",
        "",
        f"- Components: `{len(selected)}`.",
        f"- Component-specific L2 values tried; selected counts: `{selected['selected_linear_reg'].value_counts().sort_index().to_dict()}`.",
        f"- Mean selected component OOF Spearman: `{selected['selected_oof_spearman'].mean():.4f}`.",
        f"- Median selected component OOF Spearman: `{selected['selected_oof_spearman'].median():.4f}`.",
        f"- Macro OOF Spearman induced by selected component models: `{kl_summary['macro_oof_spearman'].iloc[0]:.4f}`.",
        "",
        "Best predicted KL rows:",
        "",
        key[
            [
                "kl_reg",
                "predicted_objective",
                "regularized_objective",
                "mean_phase_tv_to_proportional",
                "max_simulated_epoch",
                "q95_simulated_epoch",
                "nearest_observed_run_name",
                "nearest_observed_value",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
    ]
    if aggregate is not None:
        agg_best = aggregate.sort_values(["predicted_objective", "kl_reg"]).head(1).iloc[0]
        lines.extend(
            [
                "",
                "Aggregate DSP reference:",
                f"- Best predicted row in the prior aggregate sweep: `KL={float(agg_best['kl_reg']):g}`, predicted BPB `{float(agg_best['predicted_objective']):.6f}`.",
            ]
        )
    if not olmix_marker.empty:
        row = olmix_marker.iloc[0]
        lines.extend(
            [
                "",
                "Paper-faithful OLMix reference:",
                f"- `two_phase_adapted`, Huber delta `{float(row['huber_delta']):g}`, KL `{float(row['kl_reg']):g}`, predicted BPB `{float(row['predicted_macro_bpb']):.6f}`, max simulated epoch `{float(row['max_simulated_epoch']):.3f}`.",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_regs = parse_float_list(str(args.linear_reg_values))
    kl_regs = parse_float_list(str(args.kl_reg_values))
    _signal, columns, domains, natural = base.load_raw_signal_panel()
    target_budget = base.load_target_budget()
    token_counts = base.load_domain_token_counts(domains)
    panel, metadata = paper_olmix.build_fit_panel(columns)
    components = list(metadata["components"])
    if args.component_limit is not None:
        components = components[: int(args.component_limit)]
        print(f"DEBUG: limiting to first {len(components)} components", flush=True)
    panel[MACRO_TARGET] = panel[components].astype(float).mean(axis=1)
    panel.to_csv(args.output_dir / "fit_panel_table9_macro.csv", index=False)
    (args.output_dir / "component_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    models, selected, train_predictions, oof_predictions = fit_all_components(
        panel=panel,
        components=components,
        columns=columns,
        domains=domains,
        token_counts=token_counts,
        linear_regs=linear_regs,
        args=args,
        output_dir=args.output_dir,
    )
    write_component_predictions(args.output_dir, panel, components, train_predictions, oof_predictions)

    reference = np.stack([natural, natural], axis=0)
    starts = [reference]
    aggregate = load_aggregate_sweep(args.aggregate_dsp_sweep)
    if aggregate is not None:
        for kl_value in (0.025, 0.05, 0.1, 0.2, 0.5):
            proposal_path = args.aggregate_dsp_sweep.parent / f"kl_{str(kl_value).replace('.', 'p')}" / "proposed_mixture_weights.csv"
            if proposal_path.exists():
                proposal = pd.read_csv(proposal_path)
                starts.append(proposal[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T)
    if args.olmix_weights.exists():
        olmix_weights = pd.read_csv(args.olmix_weights)
        starts.append(olmix_weights[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T)

    folds = component_dsp.panel_stratified_folds(panel, n_splits=N_SPLITS, seed=CV_SEED)
    kl_rows: list[KLSummary] = []
    mixture_frames: dict[float, pd.DataFrame] = {}
    for kl_reg in kl_regs:
        print(f"Optimizing per-component DSP KL={kl_reg:g}", flush=True)
        weights, regularized, status = optimize_per_component_kl(
            models,
            natural,
            kl_reg=float(kl_reg),
            starts=starts,
        )
        starts.append(weights)
        summary = summarize_kl(
            models=models,
            selected_summary=selected,
            panel=panel,
            components=components,
            columns=columns,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            weights=weights,
            regularized_objective=regularized,
            optimizer_status=status,
            kl_reg=float(kl_reg),
            train_predictions=train_predictions,
            oof_predictions=oof_predictions,
            folds=folds,
            n_starts=len(starts),
        )
        kl_rows.append(summary)
        mixture_frames[float(kl_reg)] = mixture_frame(
            domains=domains,
            natural=natural,
            weights=weights,
            token_counts=token_counts,
            target_budget=target_budget,
        )

    kl_summary = pd.DataFrame([asdict(row) for row in kl_rows]).sort_values("kl_reg").reset_index(drop=True)
    kl_summary.to_csv(args.output_dir / "per_component_dsp_kl_sweep_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "target_metric": "mean_51_table9_bpb_components",
                    "panel_rows": int(len(panel)),
                    "components": components,
                    "component_count": len(components),
                    "linear_reg_values": linear_regs,
                    "kl_reg_values": kl_regs,
                    "component_l2_selection": "oof_rmse + 0.5 * lower_tail_optimism",
                    "phase_fractions": base.PHASE_FRACTIONS.tolist(),
                    "n_proportional_reference_rows": int(metadata["n_proportional_reference_rows"]),
                    "proportional_reference_macro_mean": float(metadata["proportional_reference_macro_mean"]),
                },
                "selected_components": selected.to_dict(orient="records"),
                "kl_sweep": kl_summary.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_mixture_weights(args.output_dir, kl_summary, mixture_frames)
    olmix_marker = load_olmix_marker(args.olmix_summary)
    write_kl_overlay_plots(args.output_dir, kl_summary, aggregate, olmix_marker)
    write_report(args.output_dir, selected, kl_summary, aggregate, olmix_marker)
    print(
        kl_summary[
            [
                "kl_reg",
                "predicted_objective",
                "regularized_objective",
                "mean_phase_tv_to_proportional",
                "max_simulated_epoch",
                "macro_oof_spearman",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
