# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Nested-CV diagnostics for simple phase-separated DSP variants on Table-9.

This analysis tests whether OLMoBaseEval Table-9 benefits from a minimally more
expressive two-phase DSP form: separate global phase-1 multipliers for
saturation exposure and overexposure-penalty exposure. The key diagnostic is
nested CV: nonlinear DSP parameters are refit inside each fold instead of
freezing full-panel phase parameters and only refitting the linear head.
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
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
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
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_dsp_phase_functional_form_20260630"
MACRO_TARGET = "table9_macro_bpb"
N_SPLITS = 5
CV_SEED = 0
LOWER_TAIL_FRAC = 0.15
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class VariantSummary:
    variant_key: str
    variant_name: str
    linear_reg: float
    n_rows: int
    total_param_count: int
    train_rmse: float
    train_spearman: float
    fixed_oof_rmse: float
    fixed_oof_spearman: float
    fixed_fold_mean_regret_at_1: float
    fixed_global_regret_at_1: float
    fixed_lower_tail_optimism: float
    nested_oof_rmse: float
    nested_oof_spearman: float
    nested_fold_mean_regret_at_1: float
    nested_fold_mean_regret_at_3: float
    nested_fold_mean_regret_at_5: float
    nested_global_regret_at_1: float
    nested_global_regret_at_3: float
    nested_global_regret_at_5: float
    nested_lower_tail_optimism: float
    nested_low_tail_rmse: float
    nested_selected_run_name: str
    nested_selected_actual_bpb: float
    nested_selected_actual_rank: int
    nested_selected_component_harm_count: int
    best_observed_run_name: str
    best_observed_bpb: float
    gamma_full: float | None
    gamma_saturation_full: float | None
    gamma_penalty_full: float | None
    gamma_nested_mean: float | None
    gamma_nested_std: float | None
    gamma_saturation_nested_mean: float | None
    gamma_saturation_nested_std: float | None
    gamma_penalty_nested_mean: float | None
    gamma_penalty_nested_std: float | None


@dataclass(frozen=True)
class KLSummary:
    variant_key: str
    variant_name: str
    linear_reg: float
    kl_reg: float
    predicted_objective: float
    regularized_objective: float
    proportional_predicted: float
    best_observed_run_name: str
    best_observed_bpb: float
    nearest_observed_run_name: str
    nearest_observed_bpb: float
    nearest_observed_mean_phase_tv: float
    mean_phase_tv_to_proportional: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_weight: float
    min_weight: float
    optimizer_status: str


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default="canonical,effective_exposure,split_saturation_penalty")
    parser.add_argument("--linear-reg-values", default="0.0001,0.001")
    parser.add_argument("--kl-reg-values", default="0.025,0.05,0.1,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    return parser.parse_args()


def softmax_pair(logits: np.ndarray, m: int) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    out = np.zeros((2, m), dtype=float)
    for phase_idx in range(2):
        phase_logits = logits[phase_idx * m : (phase_idx + 1) * m]
        weights = np.exp(phase_logits - np.max(phase_logits))
        out[phase_idx] = weights / weights.sum()
    return out


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)


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


def fit_variant_with_l2(
    packet: dsp.PacketData,
    variant_key: str,
    linear_reg: float,
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[dsp.FittedDSPModel, pd.DataFrame]:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        return dsp.fit_variant(
            packet,
            dsp.VARIANTS[variant_key],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
    finally:
        dsp.LINEAR_REG = original_linear_reg


def fixed_param_oof(packet: dsp.PacketData, model: dsp.FittedDSPModel, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
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


def nested_oof(
    packet: dsp.PacketData,
    variant_key: str,
    linear_reg: float,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[np.ndarray, list[dsp.FittedDSPModel]]:
    out = np.zeros_like(packet.y, dtype=float)
    fold_models: list[dsp.FittedDSPModel] = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"    nested fold {fold_id + 1}/{len(folds)}", flush=True)
        train_packet = subset_packet(packet, train_idx)
        model, _tuning = fit_variant_with_l2(
            train_packet,
            variant_key,
            linear_reg,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        out[test_idx] = dsp.predict(model, packet.w[test_idx])
        fold_models.append(model)
    return out, fold_models


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    spearman = float(spearmanr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    return rmse, spearman


def global_regret_at_k(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    order = np.argsort(pred)[: min(k, len(y))]
    return float(np.min(y[order]) - np.min(y))


def fold_mean_regret_at_k(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], k: int) -> float:
    values: list[float] = []
    for _train_idx, test_idx in folds:
        selected = test_idx[np.argsort(pred[test_idx])[: min(k, len(test_idx))]]
        values.append(float(np.min(y[selected]) - np.min(y[test_idx])))
    return float(np.mean(values))


def lower_tail_optimism(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0)))
    rmse = float(np.sqrt(np.mean((pred[tail_idx] - y[tail_idx]) ** 2)))
    return optimism, rmse


def gamma_values(model: dsp.FittedDSPModel) -> dict[str, float | None]:
    params = model.params
    return {
        "gamma": float(params["gamma"]) if "gamma" in params else None,
        "gamma_saturation": float(params["gamma_saturation"]) if "gamma_saturation" in params else None,
        "gamma_penalty": float(params["gamma_penalty"]) if "gamma_penalty" in params else None,
    }


def summarize_gamma(fold_models: list[dsp.FittedDSPModel], key: str) -> tuple[float | None, float | None]:
    values = [gamma_values(model)[key] for model in fold_models]
    numeric = np.asarray([value for value in values if value is not None], dtype=float)
    if len(numeric) == 0:
        return None, None
    return float(np.mean(numeric)), float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0


def selected_component_harm_count(panel: pd.DataFrame, selected_idx: int, components: list[str]) -> int:
    proportional = panel.loc[panel["run_name"].eq("baseline_proportional"), components]
    if len(proportional) != 1:
        raise ValueError("Expected exactly one baseline_proportional row in fit panel")
    baseline = proportional.iloc[0]
    selected = panel.iloc[selected_idx][components]
    return int((selected.to_numpy(dtype=float) > baseline.to_numpy(dtype=float)).sum())


def summarize_variant(
    *,
    panel: pd.DataFrame,
    components: list[str],
    packet: dsp.PacketData,
    variant_key: str,
    linear_reg: float,
    full_model: dsp.FittedDSPModel,
    train_pred: np.ndarray,
    fixed_oof_pred: np.ndarray,
    nested_oof_pred: np.ndarray,
    nested_models: list[dsp.FittedDSPModel],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> VariantSummary:
    y = packet.y
    train_rmse, train_spearman = regression_metrics(y, train_pred)
    fixed_rmse, fixed_spearman = regression_metrics(y, fixed_oof_pred)
    nested_rmse, nested_spearman = regression_metrics(y, nested_oof_pred)
    fixed_optimism, _fixed_low_tail_rmse = lower_tail_optimism(y, fixed_oof_pred)
    nested_optimism, nested_low_tail_rmse = lower_tail_optimism(y, nested_oof_pred)
    selected_idx = int(np.argmin(nested_oof_pred))
    best_idx = int(np.argmin(y))
    actual_order = np.argsort(y)
    selected_rank = int(np.flatnonzero(actual_order == selected_idx)[0] + 1)
    gamma = gamma_values(full_model)
    gamma_mean, gamma_std = summarize_gamma(nested_models, "gamma")
    gamma_sat_mean, gamma_sat_std = summarize_gamma(nested_models, "gamma_saturation")
    gamma_pen_mean, gamma_pen_std = summarize_gamma(nested_models, "gamma_penalty")
    return VariantSummary(
        variant_key=variant_key,
        variant_name=full_model.variant.name,
        linear_reg=float(linear_reg),
        n_rows=int(len(y)),
        total_param_count=int(full_model.total_param_count),
        train_rmse=train_rmse,
        train_spearman=train_spearman,
        fixed_oof_rmse=fixed_rmse,
        fixed_oof_spearman=fixed_spearman,
        fixed_fold_mean_regret_at_1=fold_mean_regret_at_k(y, fixed_oof_pred, folds, 1),
        fixed_global_regret_at_1=global_regret_at_k(y, fixed_oof_pred, 1),
        fixed_lower_tail_optimism=fixed_optimism,
        nested_oof_rmse=nested_rmse,
        nested_oof_spearman=nested_spearman,
        nested_fold_mean_regret_at_1=fold_mean_regret_at_k(y, nested_oof_pred, folds, 1),
        nested_fold_mean_regret_at_3=fold_mean_regret_at_k(y, nested_oof_pred, folds, 3),
        nested_fold_mean_regret_at_5=fold_mean_regret_at_k(y, nested_oof_pred, folds, 5),
        nested_global_regret_at_1=global_regret_at_k(y, nested_oof_pred, 1),
        nested_global_regret_at_3=global_regret_at_k(y, nested_oof_pred, 3),
        nested_global_regret_at_5=global_regret_at_k(y, nested_oof_pred, 5),
        nested_lower_tail_optimism=nested_optimism,
        nested_low_tail_rmse=nested_low_tail_rmse,
        nested_selected_run_name=str(panel.iloc[selected_idx]["run_name"]),
        nested_selected_actual_bpb=float(y[selected_idx]),
        nested_selected_actual_rank=selected_rank,
        nested_selected_component_harm_count=selected_component_harm_count(panel, selected_idx, components),
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_bpb=float(y[best_idx]),
        gamma_full=gamma["gamma"],
        gamma_saturation_full=gamma["gamma_saturation"],
        gamma_penalty_full=gamma["gamma_penalty"],
        gamma_nested_mean=gamma_mean,
        gamma_nested_std=gamma_std,
        gamma_saturation_nested_mean=gamma_sat_mean,
        gamma_saturation_nested_std=gamma_sat_std,
        gamma_penalty_nested_mean=gamma_pen_mean,
        gamma_penalty_nested_std=gamma_pen_std,
    )


def write_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    fig = go.Figure()
    for variant, group in predictions.groupby("variant_key", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["actual"],
                y=group["nested_oof_prediction"],
                mode="markers",
                name=str(variant),
                text=group["run_name"],
                hovertemplate="run=%{text}<br>actual=%{x:.5f}<br>nested prediction=%{y:.5f}<extra></extra>",
            )
        )
    lo = min(float(predictions["actual"].min()), float(predictions["nested_oof_prediction"].min()))
    hi = max(float(predictions["actual"].max()), float(predictions["nested_oof_prediction"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line={"dash": "dash", "color": "#64748b"}))
    fig.update_layout(
        title="Table-9 DSP phase variants: nested OOF prediction vs actual",
        xaxis_title="Observed Table-9 macro BPB",
        yaxis_title="Nested OOF predicted Table-9 macro BPB",
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def regularized_dsp_objective(
    model: dsp.FittedDSPModel,
    weights: np.ndarray,
    natural: np.ndarray,
    kl_reg: float,
) -> float:
    prediction = float(dsp.predict(model, weights[None, :, :])[0])
    kl = base.weighted_multiclass_kl(weights, natural, base.PHASE_FRACTIONS)
    return prediction + float(kl_reg) * kl


def optimize_dsp_kl(
    model: dsp.FittedDSPModel,
    natural: np.ndarray,
    kl_reg: float,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, str]:
    m = len(natural)

    def objective(logits: np.ndarray) -> float:
        return regularized_dsp_objective(model, softmax_pair(logits, m), natural, kl_reg)

    best: Any | None = None
    for start_weights in starts:
        result = minimize(
            objective,
            weights_to_logits(start_weights),
            method="L-BFGS-B",
            options={"maxiter": 700, "ftol": 1e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("DSP KL optimization failed")
    return softmax_pair(np.asarray(best.x, dtype=float), m), float(best.fun), str(best.message)


def proposal_starts(packet: dsp.PacketData, natural: np.ndarray, train_pred: np.ndarray) -> list[np.ndarray]:
    starts: list[np.ndarray] = [np.stack([natural, natural], axis=0)]
    for order in (np.argsort(packet.y), np.argsort(train_pred)):
        for idx in order[: min(20, len(order))]:
            starts.append(packet.w[int(idx)])
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for weights in starts:
        key = np.round(weights, decimals=12).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(weights)
    return unique


def summarize_kl(
    *,
    model: dsp.FittedDSPModel,
    packet: dsp.PacketData,
    panel: pd.DataFrame,
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    regularized_objective: float,
    optimizer_status: str,
    variant_key: str,
    linear_reg: float,
    kl_reg: float,
) -> KLSummary:
    reference = np.stack([natural, natural], axis=0)
    distances = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(packet.y))
    sim_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
    return KLSummary(
        variant_key=variant_key,
        variant_name=model.variant.name,
        linear_reg=float(linear_reg),
        kl_reg=float(kl_reg),
        predicted_objective=float(dsp.predict(model, weights[None, :, :])[0]),
        regularized_objective=float(regularized_objective),
        proportional_predicted=float(dsp.predict(model, reference[None, :, :])[0]),
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_bpb=float(packet.y[best_idx]),
        nearest_observed_run_name=str(panel.iloc[nearest_idx]["run_name"]),
        nearest_observed_bpb=float(packet.y[nearest_idx]),
        nearest_observed_mean_phase_tv=float(distances[nearest_idx]),
        mean_phase_tv_to_proportional=float(0.5 * np.abs(weights - reference).sum(axis=1).mean()),
        max_simulated_epoch=float(np.max(sim_epochs)),
        q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
        max_weight=float(np.max(weights)),
        min_weight=float(np.min(weights)),
        optimizer_status=optimizer_status,
    )


def write_mixture_weights(
    output_dir: Path,
    *,
    variant_key: str,
    linear_reg: float,
    kl_reg: float,
    domains: list[str],
    natural: np.ndarray,
    weights: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> None:
    sim_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
    ratios = weights / np.clip(natural[None, :], 1e-12, None)
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
        }
    )
    safe_l2 = str(linear_reg).replace(".", "p")
    safe_kl = str(kl_reg).replace(".", "p")
    path = output_dir / "mixtures" / f"{variant_key}_l2_{safe_l2}_kl_{safe_kl}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_kl_plot(summary: pd.DataFrame, output_path: Path) -> None:
    fig = go.Figure()
    for (variant, l2), group in summary.groupby(["variant_key", "linear_reg"], sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["kl_reg"].map(lambda value: f"{float(value):g}"),
                y=group["predicted_objective"],
                mode="lines+markers",
                name=f"{variant}, L2={float(l2):g}",
                hovertemplate="KL=%{x}<br>pred BPB=%{y:.6f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Table-9 phase-DSP KL proposals: predicted BPB",
        xaxis_title="KL penalty",
        yaxis_title="Predicted Table-9 macro BPB",
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_str_list(args.variants)
    linear_regs = parse_float_list(args.linear_reg_values)
    kl_regs = parse_float_list(args.kl_reg_values)
    invalid = sorted(set(variants).difference(dsp.VARIANTS))
    if invalid:
        raise ValueError(f"Unknown DSP variants: {invalid}")

    signal, columns, domains, _natural = base.load_raw_signal_panel()
    del signal
    token_counts = base.load_domain_token_counts(domains)
    target_budget = base.load_target_budget()
    panel, metadata = paper_olmix.build_fit_panel(columns)
    components = list(metadata["components"])
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, MACRO_TARGET)
    folds = component_dsp.panel_stratified_folds(panel, n_splits=N_SPLITS, seed=CV_SEED)
    prop_rows = panel["run_name"].eq("baseline_proportional")
    if int(prop_rows.sum()) != 1:
        raise ValueError("Expected one baseline_proportional row")
    natural = packet.w[int(np.flatnonzero(prop_rows)[0]), 0].copy()

    summary_rows: list[VariantSummary] = []
    prediction_rows: list[pd.DataFrame] = []
    full_models: dict[tuple[str, float], dsp.FittedDSPModel] = {}
    train_predictions_by_model: dict[tuple[str, float], np.ndarray] = {}
    for linear_reg in linear_regs:
        for variant_key in variants:
            print(f"Fitting {variant_key} linear_reg={linear_reg:g}", flush=True)
            full_model, tuning = fit_variant_with_l2(
                packet,
                variant_key,
                linear_reg,
                maxiter=int(args.maxiter),
                coarse_top_k=int(args.coarse_top_k),
                basin_hopping_iters=int(args.basin_hopping_iters),
            )
            train_pred = dsp.predict(full_model, packet.w)
            full_models[(variant_key, float(linear_reg))] = full_model
            train_predictions_by_model[(variant_key, float(linear_reg))] = train_pred
            fixed_oof_pred = fixed_param_oof(packet, full_model, folds)
            nested_pred, nested_models = nested_oof(
                packet,
                variant_key,
                linear_reg,
                folds,
                maxiter=int(args.maxiter),
                coarse_top_k=int(args.coarse_top_k),
                basin_hopping_iters=int(args.basin_hopping_iters),
            )
            summary = summarize_variant(
                panel=panel,
                components=components,
                packet=packet,
                variant_key=variant_key,
                linear_reg=linear_reg,
                full_model=full_model,
                train_pred=train_pred,
                fixed_oof_pred=fixed_oof_pred,
                nested_oof_pred=nested_pred,
                nested_models=nested_models,
                folds=folds,
            )
            summary_rows.append(summary)
            tuning_path = args.output_dir / f"tuning_{variant_key}_l2_{linear_reg:g}.csv"
            tuning.to_csv(tuning_path, index=False)
            pred = panel[["run_name", "panel_source", MACRO_TARGET]].copy()
            pred = pred.rename(columns={MACRO_TARGET: "actual"})
            pred["variant_key"] = variant_key
            pred["linear_reg"] = float(linear_reg)
            pred["train_prediction"] = train_pred
            pred["fixed_oof_prediction"] = fixed_oof_pred
            pred["nested_oof_prediction"] = nested_pred
            pred["nested_oof_residual"] = nested_pred - packet.y
            prediction_rows.append(pred)

    summary_frame = pd.DataFrame.from_records([asdict(row) for row in summary_rows])
    summary_frame = summary_frame.sort_values(
        ["nested_fold_mean_regret_at_1", "nested_oof_rmse", "nested_lower_tail_optimism"],
        ascending=[True, True, True],
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary_frame.to_csv(args.output_dir / "phase_variant_nested_cv_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "phase_variant_nested_cv_predictions.csv", index=False)
    write_scatter(predictions, args.output_dir / "phase_variant_nested_oof_scatter.html")
    kl_rows: list[KLSummary] = []
    for row in summary_frame.itertuples(index=False):
        variant_key = str(row.variant_key)
        linear_reg = float(row.linear_reg)
        model = full_models[(variant_key, linear_reg)]
        starts = proposal_starts(packet, natural, train_predictions_by_model[(variant_key, linear_reg)])
        for kl_reg in kl_regs:
            print(f"Optimizing KL proposal {variant_key} L2={linear_reg:g} KL={kl_reg:g}", flush=True)
            weights, regularized_objective, optimizer_status = optimize_dsp_kl(
                model,
                natural,
                float(kl_reg),
                starts,
            )
            kl_summary = summarize_kl(
                model=model,
                packet=packet,
                panel=panel,
                weights=weights,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                regularized_objective=regularized_objective,
                optimizer_status=optimizer_status,
                variant_key=variant_key,
                linear_reg=linear_reg,
                kl_reg=float(kl_reg),
            )
            kl_rows.append(kl_summary)
            write_mixture_weights(
                args.output_dir,
                variant_key=variant_key,
                linear_reg=linear_reg,
                kl_reg=float(kl_reg),
                domains=domains,
                natural=natural,
                weights=weights,
                token_counts=token_counts,
                target_budget=target_budget,
            )
    kl_frame = pd.DataFrame.from_records([asdict(row) for row in kl_rows])
    kl_frame = kl_frame.sort_values(
        ["predicted_objective", "mean_phase_tv_to_proportional", "max_simulated_epoch"],
        ascending=[True, True, True],
    )
    kl_frame.to_csv(args.output_dir / "phase_variant_kl_proposal_summary.csv", index=False)
    write_kl_plot(kl_frame, args.output_dir / "phase_variant_kl_predicted_bpb.html")
    report = {
        "output_dir": str(args.output_dir),
        "variants": variants,
        "linear_regs": linear_regs,
        "kl_regs": kl_regs,
        "n_rows": int(len(panel)),
        "n_qsplit_signal_rows": int(panel["panel_source"].eq("qsplit_signal").sum()),
        "n_domain_deletion_rows": int(panel["panel_source"].eq("domain_deletion").sum()),
        "best_by_nested_regret_rmse": summary_frame.iloc[0].to_dict(),
        "best_kl_by_predicted_objective": kl_frame.iloc[0].to_dict() if not kl_frame.empty else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summary_frame.to_string(index=False), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
