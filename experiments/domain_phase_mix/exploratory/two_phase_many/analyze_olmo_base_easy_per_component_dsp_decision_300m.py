# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Per-component OLMoBaseEval Easy DSP decision diagnostics.

This is the next-stage harness after the first reliability-weighting screen.
The headline objective remains the unweighted OLMoBaseEval Easy Table-9 macro
BPB. Reliability is used as uncertainty/shrinkage/guardrail evidence, not as a
silent replacement for the objective.

The OOF convention matches the existing DSP diagnostics: tune nonlinear DSP
geometry on the full panel, then refit only the linear head inside panel-stratified
folds. This is not full nested nonlinear CV; it is a decision-diagnostic screen
before paying for validation runs.
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
from scipy.stats import pearsonr, spearmanr

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
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_per_component_dsp_decision_300m_20260626"
DEFAULT_FIT_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "fit_panel_table9_macro.csv"
)
DEFAULT_RELIABILITY = (
    REFERENCE_OUTPUTS / "olmo_base_easy_reliability_weighting_20260625" / "component_reliability.csv"
)
DEFAULT_OLMIX_PREDICTIONS = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "macro_fit_predictions.csv"
)
DEFAULT_AGGREGATE_DSP_PREDICTIONS = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_predictions.csv"
)
DEFAULT_AGGREGATE_DSP_SUMMARY = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_fit_summary.csv"
)

MACRO_TARGET = "table9_macro_bpb"
N_SPLITS = 5
CV_SEED = 0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ComponentDspSummary:
    component: str
    linear_reg: float
    n_rows: int
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_pearson: float
    fold_mean_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float
    nonlinear_objective: float
    total_param_count: int


@dataclass(frozen=True)
class DecisionSummary:
    method: str
    family: str
    subset: str
    selection_score_name: str
    hyperparameter_name: str
    hyperparameter_value: float
    n_rows: int
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_pearson: float
    fold_mean_regret_at_1: float
    fold_mean_regret_at_3: float
    fold_mean_regret_at_5: float
    full_regret_at_1: float
    full_regret_at_3: float
    full_regret_at_5: float
    lower_tail_optimism: float
    low_tail_rmse: float
    selected_run_name: str
    selected_actual_bpb: float
    selected_oof_prediction: float
    selected_selection_score: float
    selected_actual_rank: int
    best_observed_run_name: str
    best_observed_bpb: float
    post_selection_optimism: float
    predicted_guardrail_violation_count: int
    observed_guardrail_violation_count: int
    max_predicted_guardrail_excess: float
    max_observed_guardrail_excess: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--component-reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--olmix-predictions", type=Path, default=DEFAULT_OLMIX_PREDICTIONS)
    parser.add_argument("--aggregate-dsp-predictions", type=Path, default=DEFAULT_AGGREGATE_DSP_PREDICTIONS)
    parser.add_argument("--aggregate-dsp-summary", type=Path, default=DEFAULT_AGGREGATE_DSP_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--linear-reg-values", default="0.001")
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--component-limit", type=int, default=None)
    parser.add_argument("--guardrail-lambdas", default="0.25,0.5,1.0")
    parser.add_argument("--shrink-strengths", default="0.5,1.0")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def component_columns(panel: pd.DataFrame) -> list[str]:
    components = paper_olmix.table9_component_order()
    missing = sorted(set(components).difference(panel.columns))
    if missing:
        raise ValueError(f"Fit panel is missing Table-9 components: {missing[:12]}")
    return components


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = y_hat - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    spearman = float(spearmanr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def panel_stratified_folds(panel: pd.DataFrame, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_id = np.full(len(panel), -1, dtype=int)
    for _panel_source, indices in panel.groupby("panel_source", sort=True).indices.items():
        shuffled = np.asarray(list(indices), dtype=int)
        rng.shuffle(shuffled)
        for position, row_idx in enumerate(shuffled):
            fold_id[row_idx] = position % n_splits
    if np.any(fold_id < 0):
        raise ValueError("Failed to assign all rows to folds")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        test_idx = np.flatnonzero(fold_id == fold)
        train_idx = np.flatnonzero(fold_id != fold)
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"Empty fold {fold}")
        folds.append((train_idx, test_idx))
    return folds


def fit_oof_with_folds(packet: dsp.PacketData, model: dsp.FittedDSPModel, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fold_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = dsp.predict(fold_model, packet.w[test_idx])
    return oof


def build_packet(
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    target_name: str,
) -> dsp.PacketData:
    return top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, target_name)


def fit_component_dsp(
    *,
    panel: pd.DataFrame,
    components: list[str],
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    folds = panel_stratified_folds(panel, n_splits=N_SPLITS, seed=CV_SEED)
    train_predictions = np.zeros((len(panel), len(components)), dtype=float)
    oof_predictions = np.zeros((len(panel), len(components)), dtype=float)
    rows: list[ComponentDspSummary] = []
    model_dir = output_dir / f"linear_reg_{linear_reg:g}" / "component_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        for idx, component in enumerate(components, start=1):
            print(f"  component {idx}/{len(components)}: {component}", flush=True)
            packet = build_packet(panel, columns, domains, token_counts, component)
            model, tuning = dsp.fit_variant(
                packet,
                dsp.VARIANTS["effective_exposure"],
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
                basin_hopping_iters=basin_hopping_iters,
            )
            train_pred = dsp.predict(model, packet.w)
            oof_pred = fit_oof_with_folds(packet, model, folds)
            train_predictions[:, idx - 1] = train_pred
            oof_predictions[:, idx - 1] = oof_pred
            train_rmse, _train_mae, _train_pearson, train_spearman = regression_metrics(packet.y, train_pred)
            oof_rmse, _oof_mae, oof_pearson, oof_spearman = regression_metrics(packet.y, oof_pred)
            oof_diag = base.predictive_diagnostics(packet.y, oof_pred, folds)
            rows.append(
                ComponentDspSummary(
                    component=component,
                    linear_reg=float(linear_reg),
                    n_rows=int(len(panel)),
                    train_rmse=train_rmse,
                    train_spearman=train_spearman,
                    oof_rmse=oof_rmse,
                    oof_spearman=oof_spearman,
                    oof_pearson=oof_pearson,
                    fold_mean_regret_at_1=float(oof_diag["fold_mean_regret_at_1"]),
                    lower_tail_optimism=float(oof_diag["lower_tail_optimism"]),
                    low_tail_rmse=float(oof_diag["low_tail_rmse"]),
                    nonlinear_objective=float(tuning["objective"].min()),
                    total_param_count=int(model.total_param_count),
                )
            )
            safe_component = component.replace("/", "__").replace(":", "_")
            tuning.to_csv(model_dir / f"{safe_component}_tuning.csv", index=False)
            (model_dir / f"{safe_component}_model.json").write_text(
                json.dumps(
                    dsp.model_to_json(
                        model,
                        {
                            "component": component,
                            "linear_reg": linear_reg,
                            "oof_convention": "full nonlinear geometry, fold linear head",
                        },
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        dsp.LINEAR_REG = original_linear_reg
    return pd.DataFrame([asdict(row) for row in rows]), train_predictions, oof_predictions


def topk_regret(y: np.ndarray, score: np.ndarray, candidate_indices: np.ndarray, k: int) -> float:
    if len(candidate_indices) == 0:
        return float("nan")
    order = candidate_indices[np.argsort(score[candidate_indices])]
    selected = order[: min(k, len(order))]
    return float(np.min(y[selected]) - np.min(y[candidate_indices]))


def lower_tail_optimism(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tail_count = max(5, int(np.ceil(0.15 * len(y))))
    tail_idx = np.argsort(pred)[:tail_count]
    residual = pred[tail_idx] - y[tail_idx]
    optimism = float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0)))
    tail_rmse = float(np.sqrt(np.mean(residual * residual)))
    return optimism, tail_rmse


def guardrail_counts(
    values: np.ndarray,
    prop_mean: np.ndarray,
    prop_sd: np.ndarray,
    *,
    epsilon_sd: float = 1.0,
) -> tuple[int, float]:
    threshold = prop_mean + epsilon_sd * np.nan_to_num(prop_sd, nan=0.0, posinf=0.0, neginf=0.0)
    excess = values - threshold
    return int(np.sum(excess > 0.0)), float(np.max(np.maximum(excess, 0.0)))


def summarize_decision(
    *,
    method: str,
    family: str,
    hyperparameter_name: str,
    hyperparameter_value: float,
    panel: pd.DataFrame,
    y: np.ndarray,
    train_pred: np.ndarray,
    oof_pred: np.ndarray,
    selection_score: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    subset: str,
    subset_mask: np.ndarray,
    component_values: np.ndarray | None,
    component_predictions: np.ndarray | None,
    prop_component_mean: np.ndarray,
    prop_component_sd: np.ndarray,
) -> DecisionSummary:
    indices = np.flatnonzero(subset_mask)
    train_rmse, _train_mae, _train_pearson, train_spearman = regression_metrics(y[indices], train_pred[indices])
    oof_rmse, _oof_mae, oof_pearson, oof_spearman = regression_metrics(y[indices], oof_pred[indices])
    fold_regrets: dict[int, list[float]] = {1: [], 3: [], 5: []}
    for _train_idx, test_idx in folds:
        fold_indices = np.intersect1d(test_idx, indices, assume_unique=False)
        if len(fold_indices) == 0:
            continue
        for k in fold_regrets:
            fold_regrets[k].append(topk_regret(y, selection_score, fold_indices, k))
    selected_idx = int(indices[np.argmin(selection_score[indices])])
    best_idx = int(indices[np.argmin(y[indices])])
    actual_order = indices[np.argsort(y[indices])]
    selected_rank = int(np.flatnonzero(actual_order == selected_idx)[0] + 1)
    optimism, tail_rmse = lower_tail_optimism(y[indices], oof_pred[indices])
    predicted_violation_count = 0
    observed_violation_count = 0
    max_predicted_excess = 0.0
    max_observed_excess = 0.0
    if component_predictions is not None and component_values is not None:
        predicted_violation_count, max_predicted_excess = guardrail_counts(
            component_predictions[selected_idx],
            prop_component_mean,
            prop_component_sd,
        )
        observed_violation_count, max_observed_excess = guardrail_counts(
            component_values[selected_idx],
            prop_component_mean,
            prop_component_sd,
        )
    return DecisionSummary(
        method=method,
        family=family,
        subset=subset,
        selection_score_name="selection_score",
        hyperparameter_name=hyperparameter_name,
        hyperparameter_value=float(hyperparameter_value),
        n_rows=int(len(indices)),
        train_rmse=train_rmse,
        train_spearman=train_spearman,
        oof_rmse=oof_rmse,
        oof_spearman=oof_spearman,
        oof_pearson=oof_pearson,
        fold_mean_regret_at_1=float(np.mean(fold_regrets[1])),
        fold_mean_regret_at_3=float(np.mean(fold_regrets[3])),
        fold_mean_regret_at_5=float(np.mean(fold_regrets[5])),
        full_regret_at_1=topk_regret(y, selection_score, indices, 1),
        full_regret_at_3=topk_regret(y, selection_score, indices, 3),
        full_regret_at_5=topk_regret(y, selection_score, indices, 5),
        lower_tail_optimism=optimism,
        low_tail_rmse=tail_rmse,
        selected_run_name=str(panel.iloc[selected_idx]["run_name"]),
        selected_actual_bpb=float(y[selected_idx]),
        selected_oof_prediction=float(oof_pred[selected_idx]),
        selected_selection_score=float(selection_score[selected_idx]),
        selected_actual_rank=selected_rank,
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_bpb=float(y[best_idx]),
        post_selection_optimism=float(max(y[selected_idx] - oof_pred[selected_idx], 0.0)),
        predicted_guardrail_violation_count=predicted_violation_count,
        observed_guardrail_violation_count=observed_violation_count,
        max_predicted_guardrail_excess=max_predicted_excess,
        max_observed_guardrail_excess=max_observed_excess,
    )


def readiness_score(component_summary: pd.DataFrame, reliability: pd.DataFrame, components: list[str]) -> np.ndarray:
    merged = pd.DataFrame({"component": components}).merge(component_summary, on="component", how="left").merge(
        reliability,
        on="component",
        how="left",
    )
    fit = pd.to_numeric(merged["oof_spearman"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    sensitivity = pd.to_numeric(merged["two_sided_t_excess"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    return (fit * sensitivity).to_numpy(dtype=float)


def guardrail_penalty(
    component_prediction: np.ndarray,
    prop_mean: np.ndarray,
    prop_sd: np.ndarray,
) -> np.ndarray:
    threshold = prop_mean[None, :] + np.nan_to_num(prop_sd, nan=0.0, posinf=0.0, neginf=0.0)[None, :]
    return np.mean(np.maximum(component_prediction - threshold, 0.0), axis=1)


def add_method_summaries(
    *,
    rows: list[DecisionSummary],
    method: str,
    family: str,
    hyperparameter_name: str,
    hyperparameter_value: float,
    panel: pd.DataFrame,
    y: np.ndarray,
    train_pred: np.ndarray,
    oof_pred: np.ndarray,
    selection_score: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    component_values: np.ndarray | None,
    component_predictions: np.ndarray | None,
    prop_component_mean: np.ndarray,
    prop_component_sd: np.ndarray,
) -> None:
    subsets = {
        "full": np.ones(len(panel), dtype=bool),
        "qsplit_signal": panel["panel_source"].eq("qsplit_signal").to_numpy(dtype=bool),
        "domain_deletion": panel["panel_source"].eq("domain_deletion").to_numpy(dtype=bool),
    }
    for subset, mask in subsets.items():
        rows.append(
            summarize_decision(
                method=method,
                family=family,
                hyperparameter_name=hyperparameter_name,
                hyperparameter_value=hyperparameter_value,
                panel=panel,
                y=y,
                train_pred=train_pred,
                oof_pred=oof_pred,
                selection_score=selection_score,
                folds=folds,
                subset=subset,
                subset_mask=mask,
                component_values=component_values,
                component_predictions=component_predictions,
                prop_component_mean=prop_component_mean,
                prop_component_sd=prop_component_sd,
            )
        )


def load_baseline_method_predictions(
    panel: pd.DataFrame,
    path: Path,
    *,
    train_col: str,
    oof_col: str,
    family: str,
    method_prefix: str,
    hp_name: str,
    hp_col: str,
    variant_cols: list[str],
) -> list[tuple[str, str, float, np.ndarray, np.ndarray]]:
    data = pd.read_csv(path)
    out: list[tuple[str, str, float, np.ndarray, np.ndarray]] = []
    for key, view in data.groupby([*variant_cols, hp_col], sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        hp_value = float(key[-1])
        variants = key[:-1]
        method = method_prefix + "_" + "_".join(str(value).replace(".", "p") for value in variants + (hp_value,))
        merged = panel[["run_name"]].merge(
            view[["run_name", train_col, oof_col]],
            on="run_name",
            how="left",
            validate="one_to_one",
        )
        if merged[[train_col, oof_col]].isna().any().any():
            raise ValueError(f"Missing baseline predictions for {method}")
        out.append(
            (
                method,
                hp_name,
                hp_value,
                pd.to_numeric(merged[train_col], errors="raise").to_numpy(dtype=float),
                pd.to_numeric(merged[oof_col], errors="raise").to_numpy(dtype=float),
            )
        )
    return out


def write_prediction_matrix(
    output_dir: Path,
    panel: pd.DataFrame,
    components: list[str],
    linear_reg: float,
    train_predictions: np.ndarray,
    oof_predictions: np.ndarray,
) -> None:
    base_cols = panel[["run_name", "source_experiment", "panel_source", MACRO_TARGET]].copy()
    train = base_cols.copy()
    oof = base_cols.copy()
    for idx, component in enumerate(components):
        train[f"pred::{component}"] = train_predictions[:, idx]
        oof[f"pred::{component}"] = oof_predictions[:, idx]
    train.to_csv(output_dir / f"per_component_train_predictions_linear_reg_{linear_reg:g}.csv", index=False)
    oof.to_csv(output_dir / f"per_component_oof_predictions_linear_reg_{linear_reg:g}.csv", index=False)


def write_guardrail_table(
    output_dir: Path,
    decisions: pd.DataFrame,
    panel: pd.DataFrame,
    components: list[str],
    component_values: np.ndarray,
    method_component_predictions: dict[str, np.ndarray],
    prop_mean: np.ndarray,
    prop_sd: np.ndarray,
    readiness: np.ndarray,
) -> None:
    rows: list[dict[str, Any]] = []
    qsplit = decisions[decisions["subset"].eq("qsplit_signal")].copy()
    for decision in qsplit.itertuples(index=False):
        if decision.method not in method_component_predictions:
            continue
        selected = panel.index[panel["run_name"].eq(decision.selected_run_name)].to_numpy()
        if len(selected) != 1:
            raise ValueError(f"Could not locate selected run {decision.selected_run_name}")
        idx = int(selected[0])
        predictions = method_component_predictions[decision.method][idx]
        observed = component_values[idx]
        threshold = prop_mean + np.nan_to_num(prop_sd, nan=0.0, posinf=0.0, neginf=0.0)
        for comp_idx, component in enumerate(components):
            rows.append(
                {
                    "method": decision.method,
                    "selected_run_name": decision.selected_run_name,
                    "component": component,
                    "observed_component_bpb": float(observed[comp_idx]),
                    "predicted_component_bpb": float(predictions[comp_idx]),
                    "proportional_mean_bpb": float(prop_mean[comp_idx]),
                    "proportional_sd_bpb": float(prop_sd[comp_idx]),
                    "guardrail_threshold_bpb": float(threshold[comp_idx]),
                    "observed_excess_bpb": float(max(observed[comp_idx] - threshold[comp_idx], 0.0)),
                    "predicted_excess_bpb": float(max(predictions[comp_idx] - threshold[comp_idx], 0.0)),
                    "readiness_score": float(readiness[comp_idx]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "selected_component_guardrails.csv", index=False)


def plot_decision_summary(output_dir: Path, decisions: pd.DataFrame) -> None:
    qsplit = decisions[decisions["subset"].eq("qsplit_signal")].copy()
    top = qsplit.sort_values(["fold_mean_regret_at_3", "oof_rmse", "full_regret_at_3"]).head(24)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Fold Regret@3", "OOF Spearman", "Selected actual BPB", "Guardrail violations"),
    )
    fig.add_trace(go.Bar(x=top["method"], y=top["fold_mean_regret_at_3"], marker_color="#2f5d8a"), row=1, col=1)
    fig.add_trace(go.Bar(x=top["method"], y=top["oof_spearman"], marker_color="#3d8f5f"), row=1, col=2)
    fig.add_trace(go.Bar(x=top["method"], y=top["selected_actual_bpb"], marker_color="#c75035"), row=2, col=1)
    fig.add_trace(
        go.Bar(x=top["method"], y=top["predicted_guardrail_violation_count"], marker_color="#8f4775"),
        row=2,
        col=2,
    )
    fig.update_xaxes(tickangle=60)
    fig.update_layout(
        title="Qsplit-only decision diagnostics for OLMoBaseEval Easy Table-9 macro",
        template="plotly_white",
        width=1700,
        height=1000,
        showlegend=False,
    )
    fig.write_html(output_dir / "decision_diagnostics_qsplit_top_methods.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def plot_component_summary(output_dir: Path, component_summary: pd.DataFrame, reliability: pd.DataFrame) -> None:
    if "two_sided_t_excess" in component_summary.columns:
        merged = component_summary.copy()
    else:
        merged = component_summary.merge(reliability, on="component", how="left")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Component DSP OOF fit vs sensitivity", "Component OOF Spearman"),
    )
    short = merged["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False)
    fig.add_trace(
        go.Scatter(
            x=merged["two_sided_t_excess"],
            y=merged["oof_spearman"],
            mode="markers+text",
            text=short,
            textposition="top center",
            marker={"color": merged["mean_predictive_sd"], "colorscale": "RdYlGn_r", "size": 11},
            hovertemplate="%{text}<br>sensitivity=%{x:.3f}<br>oof_spearman=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    ordered = merged.sort_values("oof_spearman")
    fig.add_trace(
        go.Bar(
            x=ordered["oof_spearman"],
            y=ordered["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False),
            orientation="h",
            marker_color="#2f5d8a",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Per-component Effective-exposure DSP readiness",
        template="plotly_white",
        width=1700,
        height=1000,
        showlegend=False,
    )
    fig.update_xaxes(title_text="two_sided_t_excess", row=1, col=1)
    fig.update_yaxes(title_text="OOF Spearman", row=1, col=1)
    fig.update_xaxes(title_text="OOF Spearman", row=1, col=2)
    fig.write_html(output_dir / "per_component_dsp_readiness.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, decisions: pd.DataFrame, component_summary: pd.DataFrame) -> None:
    qsplit = decisions[decisions["subset"].eq("qsplit_signal")].copy()
    best = qsplit.sort_values(["fold_mean_regret_at_3", "oof_rmse", "full_regret_at_3"]).head(12)
    worst_components = component_summary.sort_values("oof_spearman").head(10)
    lines = [
        "# Per-component DSP decision diagnostics for OLMoBaseEval Easy",
        "",
        "Headline objective: unweighted 51-component Table-9 macro BPB.",
        "",
        "OOF convention: full-data nonlinear DSP geometry, fold-refit linear head, panel-stratified folds.",
        "This is a decision screen, not full nested nonlinear CV.",
        "",
        "## Best qsplit-only selectors by fold Regret@3",
        "",
        best[
            [
                "method",
                "family",
                "fold_mean_regret_at_1",
                "fold_mean_regret_at_3",
                "oof_spearman",
                "oof_rmse",
                "selected_run_name",
                "selected_actual_bpb",
                "selected_actual_rank",
                "predicted_guardrail_violation_count",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Worst component DSP fits",
        "",
        worst_components[
            ["component", "oof_spearman", "oof_rmse", "lower_tail_optimism", "two_sided_t_excess", "mean_predictive_sd"]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Artifacts",
        "",
        "- `decision_summary.csv`",
        "- `component_dsp_summary.csv`",
        "- `selected_component_guardrails.csv`",
        "- `decision_diagnostics_qsplit_top_methods.html`",
        "- `per_component_dsp_readiness.html`",
        "- `per_component_*_predictions_linear_reg_*.csv`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.fit_panel)
    reliability = pd.read_csv(args.component_reliability)
    components = component_columns(panel)
    if args.component_limit is not None:
        components = components[: int(args.component_limit)]
        panel = panel[[col for col in panel.columns if col not in paper_olmix.table9_component_order() or col in components]].copy()
        panel[MACRO_TARGET] = panel[components].mean(axis=1)
    reliability = pd.DataFrame({"component": components}).merge(reliability, on="component", how="left")
    if reliability.isna().any().any():
        raise ValueError("Missing reliability rows for some components")

    _signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    folds = panel_stratified_folds(panel, n_splits=N_SPLITS, seed=CV_SEED)
    y = pd.to_numeric(panel[MACRO_TARGET], errors="raise").to_numpy(dtype=float)
    component_values = panel[components].astype(float).to_numpy()
    _reference_panel, reference_metadata = paper_olmix.build_fit_panel(columns)
    prop_component_mean = np.asarray(
        [reference_metadata["proportional_reference_component_means"][component] for component in components],
        dtype=float,
    )
    prop_component_sd = np.asarray(
        [reference_metadata["proportional_reference_component_stds"][component] for component in components],
        dtype=float,
    )

    decisions: list[DecisionSummary] = []
    method_component_predictions: dict[str, np.ndarray] = {}
    for method, hp_name, hp_value, train_pred, oof_pred in load_baseline_method_predictions(
        panel,
        args.aggregate_dsp_predictions,
        train_col="train_prediction",
        oof_col="oof_prediction",
        family="aggregate_dsp",
        method_prefix="aggregate_dsp",
        hp_name="linear_reg",
        hp_col="hyperparameter_value",
        variant_cols=["variant"],
    ):
        add_method_summaries(
            rows=decisions,
            method=method,
            family="aggregate_dsp",
            hyperparameter_name=hp_name,
            hyperparameter_value=hp_value,
            panel=panel,
            y=y,
            train_pred=train_pred,
            oof_pred=oof_pred,
            selection_score=oof_pred,
            folds=folds,
            component_values=None,
            component_predictions=None,
            prop_component_mean=prop_component_mean,
            prop_component_sd=prop_component_sd,
        )

    for method, hp_name, hp_value, train_pred, oof_pred in load_baseline_method_predictions(
        panel,
        args.olmix_predictions,
        train_col="train_pred_macro_bpb",
        oof_col="oof_pred_macro_bpb",
        family="paper_faithful_olmix",
        method_prefix="olmix",
        hp_name="huber_delta",
        hp_col="huber_delta",
        variant_cols=["variant"],
    ):
        add_method_summaries(
            rows=decisions,
            method=method,
            family="paper_faithful_olmix",
            hyperparameter_name=hp_name,
            hyperparameter_value=hp_value,
            panel=panel,
            y=y,
            train_pred=train_pred,
            oof_pred=oof_pred,
            selection_score=oof_pred,
            folds=folds,
            component_values=None,
            component_predictions=None,
            prop_component_mean=prop_component_mean,
            prop_component_sd=prop_component_sd,
        )

    all_component_summaries: list[pd.DataFrame] = []
    linear_regs = parse_float_list(args.linear_reg_values)
    guardrail_lambdas = parse_float_list(args.guardrail_lambdas)
    shrink_strengths = parse_float_list(args.shrink_strengths)
    for linear_reg in linear_regs:
        print(f"Fitting per-component Effective-exposure DSP linear_reg={linear_reg:g}", flush=True)
        component_summary, train_component_pred, oof_component_pred = fit_component_dsp(
            panel=panel,
            components=components,
            columns=columns,
            domains=domains,
            token_counts=token_counts,
            linear_reg=linear_reg,
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
            output_dir=args.output_dir,
        )
        all_component_summaries.append(component_summary)
        write_prediction_matrix(args.output_dir, panel, components, linear_reg, train_component_pred, oof_component_pred)
        macro_train = train_component_pred.mean(axis=1)
        macro_oof = oof_component_pred.mean(axis=1)
        mean_method = f"per_component_mean_linear_reg_{linear_reg:g}".replace(".", "p")
        method_component_predictions[mean_method] = oof_component_pred
        add_method_summaries(
            rows=decisions,
            method=mean_method,
            family="per_component_dsp",
            hyperparameter_name="linear_reg",
            hyperparameter_value=linear_reg,
            panel=panel,
            y=y,
            train_pred=macro_train,
            oof_pred=macro_oof,
            selection_score=macro_oof,
            folds=folds,
            component_values=component_values,
            component_predictions=oof_component_pred,
            prop_component_mean=prop_component_mean,
            prop_component_sd=prop_component_sd,
        )

        readiness = readiness_score(component_summary, reliability, components)
        for shrink_strength in shrink_strengths:
            shrink = (1.0 - shrink_strength) + shrink_strength * readiness
            train_shrunk = prop_component_mean[None, :] + shrink[None, :] * (train_component_pred - prop_component_mean[None, :])
            oof_shrunk = prop_component_mean[None, :] + shrink[None, :] * (oof_component_pred - prop_component_mean[None, :])
            method = f"per_component_shrink_s{shrink_strength:g}_linear_reg_{linear_reg:g}".replace(".", "p")
            method_component_predictions[method] = oof_shrunk
            add_method_summaries(
                rows=decisions,
                method=method,
                family="per_component_dsp_shrinkage",
                hyperparameter_name="shrink_strength",
                hyperparameter_value=shrink_strength,
                panel=panel,
                y=y,
                train_pred=train_shrunk.mean(axis=1),
                oof_pred=oof_shrunk.mean(axis=1),
                selection_score=oof_shrunk.mean(axis=1),
                folds=folds,
                component_values=component_values,
                component_predictions=oof_shrunk,
                prop_component_mean=prop_component_mean,
                prop_component_sd=prop_component_sd,
            )

        penalty = guardrail_penalty(oof_component_pred, prop_component_mean, prop_component_sd)
        for guardrail_lambda in guardrail_lambdas:
            method = f"per_component_guardrail_lam{guardrail_lambda:g}_linear_reg_{linear_reg:g}".replace(".", "p")
            method_component_predictions[method] = oof_component_pred
            add_method_summaries(
                rows=decisions,
                method=method,
                family="per_component_dsp_guardrail",
                hyperparameter_name="guardrail_lambda",
                hyperparameter_value=guardrail_lambda,
                panel=panel,
                y=y,
                train_pred=macro_train,
                oof_pred=macro_oof,
                selection_score=macro_oof + guardrail_lambda * penalty,
                folds=folds,
                component_values=component_values,
                component_predictions=oof_component_pred,
                prop_component_mean=prop_component_mean,
                prop_component_sd=prop_component_sd,
            )

    component_summary_frame = pd.concat(all_component_summaries, ignore_index=True)
    # Plot and guardrail tables use the first linear-reg slice unless more are
    # explicitly requested; all slices are still saved in component_dsp_summary.
    first_component_summary = component_summary_frame[
        component_summary_frame["linear_reg"].eq(component_summary_frame["linear_reg"].iloc[0])
    ].copy()
    component_summary_with_reliability = first_component_summary.merge(reliability, on="component", how="left")
    readiness = readiness_score(first_component_summary, reliability, components)
    decision_frame = pd.DataFrame([asdict(row) for row in decisions])

    component_summary_frame.to_csv(args.output_dir / "component_dsp_summary.csv", index=False)
    component_summary_with_reliability.to_csv(args.output_dir / "component_dsp_readiness.csv", index=False)
    decision_frame.to_csv(args.output_dir / "decision_summary.csv", index=False)
    write_guardrail_table(
        args.output_dir,
        decision_frame,
        panel,
        components,
        component_values,
        method_component_predictions,
        prop_component_mean,
        prop_component_sd,
        readiness,
    )
    plot_decision_summary(args.output_dir, decision_frame)
    plot_component_summary(args.output_dir, component_summary_with_reliability, reliability)
    write_report(args.output_dir, decision_frame, component_summary_with_reliability)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "fit_panel": str(args.fit_panel),
                "component_reliability": str(args.component_reliability),
                "olmix_predictions": str(args.olmix_predictions),
                "aggregate_dsp_predictions": str(args.aggregate_dsp_predictions),
                "linear_reg_values": linear_regs,
                "guardrail_lambdas": guardrail_lambdas,
                "shrink_strengths": shrink_strengths,
                "maxiter": int(args.maxiter),
                "coarse_top_k": int(args.coarse_top_k),
                "basin_hopping_iters": int(args.basin_hopping_iters),
                "oof_convention": "full nonlinear geometry, fold linear head, panel-stratified folds",
                "headline_objective": "unweighted 51-component Table-9 macro BPB",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    best = decision_frame[decision_frame["subset"].eq("qsplit_signal")].sort_values(
        ["fold_mean_regret_at_3", "oof_rmse", "full_regret_at_3"]
    )
    print(best.head(15).to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
