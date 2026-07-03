# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose a shared-capacity late-refresh DSP term for Uncheatable BPB.

The previous per-domain phase forms failed on repaired frontier candidates. This
diagnostic tests one remaining mechanistic hypothesis: late-phase refresh is a
bounded shared resource, not an additive per-domain credit.

It freezes a no-phase DSP base, weights per-domain late refresh by the no-phase
benefit coefficients, and adds one NNLS shared feature:

    refresh = 1 - exp(-R / S)
    R = sum_i a_i^(0) * (1 - exp(-e1_i / s1))

where s1 is fixed at 1 epoch and S is selected from a predeclared grid.
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
import plotly.express as px
from scipy.optimize import minimize, nnls
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_dsp_uncheatable_eta_heldout as eta_diag  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_shared_late_capacity_20260703"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LINEAR_REG = 0.01
S1_EPOCHS = 1.0


@dataclass(frozen=True)
class SharedCapacityFit:
    s_capacity: float
    intercept: float
    coef: np.ndarray
    base_model: dsp.FittedDSPModel
    shared_coef: float


@dataclass(frozen=True)
class SummaryRow:
    model_name: str
    s_capacity: float | None
    shared_coef: float | None
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_lower_tail_optimism: float
    heldout_uncheatable_mae: float
    heldout_max_optimism: float
    heldout_gap_error: float
    heldout_best_order_correct: bool
    leave_good_rmse: float
    leave_good_signed_optimism: float
    raw_predicted_optimum: float | None
    raw_nearest_observed_tv: float | None
    raw_nearest_observed_value: float | None
    pass_gates: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-results", type=Path, default=eta_diag.DEFAULT_REPAIR_RESULTS)
    parser.add_argument("--repair-mixture-dir", type=Path, default=eta_diag.DEFAULT_REPAIR_MIXTURE_DIR)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--good-frac", type=float, default=0.15)
    return parser.parse_args()


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    residual = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    spearman = float(spearmanr(y, pred).statistic) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def lower_tail_optimism(y: np.ndarray, pred: np.ndarray) -> float:
    tail_count = max(5, int(np.ceil(dsp.LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(pred)[:tail_count]
    return float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0)))


def fit_no_phase(packet: dsp.PacketData, args: argparse.Namespace) -> dsp.FittedDSPModel:
    old_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = LINEAR_REG
    try:
        model, _trace = dsp.fit_variant(
            packet,
            dsp.VARIANTS["no_phase"],
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        return model
    finally:
        dsp.LINEAR_REG = old_reg


def shared_design(
    weights: np.ndarray,
    base_model: dsp.FittedDSPModel,
    *,
    s_capacity: float,
) -> np.ndarray:
    signal, penalty = dsp.features(weights, base_model.c0, base_model.c1, base_model.variant, base_model.params)
    e1 = weights[:, 1, :] * base_model.c1[None, :]
    kappa = np.asarray(base_model.benefit_coef, dtype=float)
    refresh_work = (1.0 - np.exp(-e1 / S1_EPOCHS)) @ kappa
    shared_feature = 1.0 - np.exp(-refresh_work / max(float(s_capacity), dsp.PHASE_EPS))
    return np.hstack([-signal, penalty, -shared_feature[:, None]])


def fit_shared_head(
    weights: np.ndarray,
    y: np.ndarray,
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    *,
    s_capacity: float,
) -> SharedCapacityFit:
    design = shared_design(weights, base_model, s_capacity=s_capacity)
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(np.mean(y))
    centered_design = design - design_mean
    centered_targets = y - target_mean
    if LINEAR_REG > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(LINEAR_REG) * np.eye(centered_design.shape[1])])
        centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
    coef, _ = nnls(centered_design, centered_targets)
    intercept = float(target_mean - (design_mean @ coef).item())
    return SharedCapacityFit(
        s_capacity=float(s_capacity),
        intercept=intercept,
        coef=np.asarray(coef, dtype=float),
        base_model=base_model,
        shared_coef=float(coef[-1]),
    )


def predict_shared(fit: SharedCapacityFit, weights: np.ndarray) -> np.ndarray:
    return np.asarray(fit.intercept + shared_design(weights, fit.base_model, s_capacity=fit.s_capacity) @ fit.coef)


def profile_score(y: np.ndarray, pred: np.ndarray) -> float:
    residual = pred - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return rmse + 0.5 * lower_tail_optimism(y, pred)


def capacity_grid(base_model: dsp.FittedDSPModel, weights: np.ndarray) -> np.ndarray:
    e1 = weights[:, 1, :] * base_model.c1[None, :]
    refresh_work = (1.0 - np.exp(-e1 / S1_EPOCHS)) @ np.asarray(base_model.benefit_coef, dtype=float)
    positive = refresh_work[refresh_work > 1e-12]
    median = float(np.median(positive)) if len(positive) else 1.0
    raw = median * np.asarray([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=float)
    return np.unique(np.clip(raw, 1e-6, 1e6))


def fit_best_shared(packet: dsp.PacketData, base_model: dsp.FittedDSPModel, idx: np.ndarray | None = None) -> SharedCapacityFit:
    if idx is None:
        train_weights = packet.w
        train_y = packet.y
    else:
        train_weights = packet.w[idx]
        train_y = packet.y[idx]
    best_fit: SharedCapacityFit | None = None
    best_score = float("inf")
    for s_capacity in capacity_grid(base_model, train_weights):
        fit = fit_shared_head(train_weights, train_y, packet, base_model, s_capacity=s_capacity)
        pred = predict_shared(fit, train_weights)
        score = profile_score(train_y, pred)
        if score < best_score:
            best_score = score
            best_fit = fit
    if best_fit is None:
        raise RuntimeError("No shared-capacity fit selected")
    return best_fit


def oof_shared(packet: dsp.PacketData, base_model: dsp.FittedDSPModel) -> np.ndarray:
    folds = eta_diag.olmix.kfold_indices(len(packet.y), n_splits=eta_diag.olmix.N_SPLITS, seed=eta_diag.olmix.CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fit = fit_best_shared(packet, base_model, train_idx)
        oof[test_idx] = predict_shared(fit, packet.w[test_idx])
    return oof


def repaired_predictions(base_model: dsp.FittedDSPModel, predict_fn: Any, heldout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(base_model, Path(row.mixture_path))
        pred = float(predict_fn(weights)[0])
        actual = float(row.uncheatable_bpb)
        rows.append(
            {
                "mixture": str(row.mixture),
                "is_uncheatable_objective": bool(row.is_uncheatable_objective),
                "actual_uncheatable_bpb": actual,
                "predicted_uncheatable_bpb": pred,
                "prediction_error": pred - actual,
                "optimism": actual - pred,
                "absolute_error": abs(pred - actual),
            }
        )
    return pd.DataFrame(rows)


def repaired_gate_stats(repaired: pd.DataFrame) -> dict[str, Any]:
    uncheatable = repaired[repaired["is_uncheatable_objective"]].copy()
    predicted_best = str(uncheatable.loc[uncheatable["predicted_uncheatable_bpb"].idxmin(), "mixture"])
    actual_best = str(uncheatable.loc[uncheatable["actual_uncheatable_bpb"].idxmin(), "mixture"])
    targeted = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_targeted")].iloc[0]
    all_deficits = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_all_deficits")].iloc[0]
    predicted_gap = float(targeted.predicted_uncheatable_bpb - all_deficits.predicted_uncheatable_bpb)
    actual_gap = float(targeted.actual_uncheatable_bpb - all_deficits.actual_uncheatable_bpb)
    return {
        "heldout_uncheatable_mae": float(uncheatable["absolute_error"].mean()),
        "heldout_max_optimism": float(uncheatable["optimism"].max()),
        "heldout_gap_error": predicted_gap - actual_gap,
        "heldout_best_order_correct": predicted_best == actual_best,
    }


def leave_good_out(packet: dsp.PacketData, base_model: dsp.FittedDSPModel, predict_kind: str) -> tuple[float, float]:
    holdout_count = max(5, int(np.ceil(0.15 * len(packet.y))))
    test_idx = np.argsort(packet.y)[:holdout_count]
    train_idx = np.setdiff1d(np.arange(len(packet.y)), test_idx)
    if predict_kind == "no_phase":
        fold_model = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, base_model.variant, base_model.params)
        pred = dsp.predict(fold_model, packet.w[test_idx])
    elif predict_kind == "shared_capacity":
        fit = fit_best_shared(packet, base_model, train_idx)
        pred = predict_shared(fit, packet.w[test_idx])
    else:
        raise ValueError(predict_kind)
    y = packet.y[test_idx]
    rmse, _mae, _pearson, _spearman = regression_metrics(y, pred)
    return rmse, float(np.mean(y - pred))


def raw_optimize_shared(fit: SharedCapacityFit, packet: dsp.PacketData) -> tuple[float, float, float]:
    n = len(fit.base_model.domain_names)

    def logits_to_weights(logits: np.ndarray) -> np.ndarray:
        logits0 = logits[:n]
        logits1 = logits[n:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)
        return np.stack([p0, p1], axis=0)

    starts = [np.zeros(2 * n, dtype=float)]
    for idx in np.linspace(0, len(packet.w) - 1, min(12, len(packet.w)), dtype=int):
        starts.append(dsp.weights_to_logits(packet.w[idx]))
    best = None
    for start in starts:
        result = minimize(
            lambda z: float(predict_shared(fit, logits_to_weights(np.asarray(z, dtype=float))[None, :, :])[0]),
            start,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("raw optimize failed")
    weights = logits_to_weights(np.asarray(best.x, dtype=float))
    distances = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    return float(best.fun), float(distances[nearest_idx]), float(packet.y[nearest_idx])


def row_passes(row: dict[str, Any]) -> bool:
    raw_ok = row["raw_predicted_optimum"] is None or float(row["raw_predicted_optimum"]) >= 0.935
    return bool(
        row["oof_spearman"] >= 0.70
        and row["heldout_uncheatable_mae"] <= 0.030
        and row["heldout_max_optimism"] <= 0.035
        and (abs(row["heldout_gap_error"]) <= 0.004)
        and raw_ok
        and row["leave_good_rmse"] <= 0.015
        and row["leave_good_signed_optimism"] <= 0.005
    )


def summarize_no_phase(packet: dsp.PacketData, base_model: dsp.FittedDSPModel, heldout: pd.DataFrame) -> tuple[SummaryRow, pd.DataFrame]:
    train_pred = dsp.predict(base_model, packet.w)
    train_rmse, _mae, _pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof, _folds = dsp_compare.fit_dsp_oof_predictions(packet, base_model)
    oof_rmse, _mae, _pearson, oof_spearman = regression_metrics(packet.y, oof)
    repaired = repaired_predictions(base_model, lambda weights: dsp.predict(base_model, weights), heldout)
    stats = repaired_gate_stats(repaired)
    leave_rmse, leave_optimism = leave_good_out(packet, base_model, "no_phase")
    raw_result, raw_weights = dsp.optimize_raw(base_model, num_starts=8, observed_start_weights=packet.w, max_observed_starts=16)
    distances = dsp.average_phase_tv_distance(packet.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    row = {
        "model_name": "no_phase",
        "s_capacity": None,
        "shared_coef": None,
        "train_rmse": float(train_rmse),
        "train_spearman": float(train_spearman),
        "oof_rmse": float(oof_rmse),
        "oof_spearman": float(oof_spearman),
        "oof_lower_tail_optimism": lower_tail_optimism(packet.y, oof),
        **stats,
        "leave_good_rmse": leave_rmse,
        "leave_good_signed_optimism": leave_optimism,
        "raw_predicted_optimum": float(raw_result.fun),
        "raw_nearest_observed_tv": float(distances[nearest_idx]),
        "raw_nearest_observed_value": float(packet.y[nearest_idx]),
    }
    row["pass_gates"] = row_passes(row)
    repaired.insert(0, "model_name", "no_phase")
    return SummaryRow(**row), repaired


def summarize_shared(packet: dsp.PacketData, base_model: dsp.FittedDSPModel, heldout: pd.DataFrame) -> tuple[SummaryRow, pd.DataFrame, pd.DataFrame]:
    fit = fit_best_shared(packet, base_model)
    train_pred = predict_shared(fit, packet.w)
    train_rmse, _mae, _pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof = oof_shared(packet, base_model)
    oof_rmse, _mae, _pearson, oof_spearman = regression_metrics(packet.y, oof)
    repaired = repaired_predictions(base_model, lambda weights: predict_shared(fit, weights), heldout)
    stats = repaired_gate_stats(repaired)
    leave_rmse, leave_optimism = leave_good_out(packet, base_model, "shared_capacity")
    raw_value, raw_tv, raw_nearest = raw_optimize_shared(fit, packet)
    row = {
        "model_name": "shared_late_capacity",
        "s_capacity": fit.s_capacity,
        "shared_coef": fit.shared_coef,
        "train_rmse": float(train_rmse),
        "train_spearman": float(train_spearman),
        "oof_rmse": float(oof_rmse),
        "oof_spearman": float(oof_spearman),
        "oof_lower_tail_optimism": lower_tail_optimism(packet.y, oof),
        **stats,
        "leave_good_rmse": leave_rmse,
        "leave_good_signed_optimism": leave_optimism,
        "raw_predicted_optimum": raw_value,
        "raw_nearest_observed_tv": raw_tv,
        "raw_nearest_observed_value": raw_nearest,
    }
    row["pass_gates"] = row_passes(row)
    repaired.insert(0, "model_name", "shared_late_capacity")

    grid_rows = []
    for s_capacity in capacity_grid(base_model, packet.w):
        grid_fit = fit_shared_head(packet.w, packet.y, packet, base_model, s_capacity=s_capacity)
        pred = predict_shared(grid_fit, packet.w)
        grid_rows.append(
            {
                "s_capacity": float(s_capacity),
                "shared_coef": grid_fit.shared_coef,
                "train_score": profile_score(packet.y, pred),
                "train_rmse": regression_metrics(packet.y, pred)[0],
                "train_spearman": regression_metrics(packet.y, pred)[3],
            }
        )
    return SummaryRow(**row), repaired, pd.DataFrame(grid_rows)


def write_outputs(output_dir: Path, summary: pd.DataFrame, heldout: pd.DataFrame, grid: pd.DataFrame) -> None:
    fig = px.scatter(
        summary,
        x="oof_spearman",
        y="heldout_uncheatable_mae",
        color="model_name",
        symbol="pass_gates",
        hover_data=["s_capacity", "shared_coef", "leave_good_rmse", "raw_predicted_optimum"],
        title="Shared-capacity late-refresh DSP diagnostic",
        template="plotly_white",
    )
    fig.add_hline(y=0.030, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.70, line_dash="dot", line_color="gray")
    fig.update_layout(width=1000, height=650)
    fig.write_html(output_dir / "shared_late_capacity_gate_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = px.bar(
        heldout,
        x="mixture",
        y="prediction_error",
        color="model_name",
        barmode="group",
        title="Shared-capacity heldout repaired-candidate errors",
        template="plotly_white",
    )
    fig2.update_layout(width=1250, height=620, xaxis_tickangle=-20)
    fig2.write_html(output_dir / "shared_late_capacity_heldout_errors.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig3 = px.line(
        grid,
        x="s_capacity",
        y="train_score",
        markers=True,
        log_x=True,
        title="Shared-capacity grid profile",
        template="plotly_white",
    )
    fig3.update_layout(width=950, height=550)
    fig3.write_html(output_dir / "shared_late_capacity_grid.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = [
        "# Shared-capacity late-refresh DSP diagnostic",
        "",
        "This tests the last pre-registered mechanistic phase form: aggregate late refresh is bounded by a shared capacity rather than additive per domain.",
        "",
        "## Gate summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Capacity grid",
        "",
        grid.to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet, *_ = eta_diag.load_packet()
    heldout = eta_diag.load_heldout(args)
    print("Fitting no-phase base", flush=True)
    base_model = fit_no_phase(packet, args)
    print("Summarizing no-phase", flush=True)
    no_phase_row, no_phase_heldout = summarize_no_phase(packet, base_model, heldout)
    print("Summarizing shared capacity", flush=True)
    shared_row, shared_heldout, grid = summarize_shared(packet, base_model, heldout)
    summary = pd.DataFrame([asdict(no_phase_row), asdict(shared_row)])
    heldout_frame = pd.concat([no_phase_heldout, shared_heldout], ignore_index=True)
    summary.to_csv(args.output_dir / "shared_late_capacity_summary.csv", index=False)
    heldout_frame.to_csv(args.output_dir / "shared_late_capacity_heldout_predictions.csv", index=False)
    grid.to_csv(args.output_dir / "shared_late_capacity_grid.csv", index=False)
    write_outputs(args.output_dir, summary, heldout_frame, grid)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "linear_reg": LINEAR_REG,
                "s1_epochs": S1_EPOCHS,
                "fit_rows": int(len(packet.y)),
                "repair_results": str(args.repair_results),
                "repair_mixture_dir": str(args.repair_mixture_dir),
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
