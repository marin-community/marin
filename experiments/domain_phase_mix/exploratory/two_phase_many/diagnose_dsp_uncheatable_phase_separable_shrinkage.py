# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose a shrinkage-tied phase-separable DSP for Uncheatable BPB.

The failed scalar-phase variants all collapse phase placement into one premium
coefficient. This diagnostic tests one final mechanistic alternative: phase 0
and phase 1 get separate nonnegative benefit channels, but late coefficients
are quadratically shrunk toward a global multiple of the early coefficients.

The nonlinear exposure shape is frozen from the no-phase DSP fit. This keeps the
test focused on whether phase placement needs separable linear response
capacity, not whether another high-dimensional nonlinear fit can chase the
heldout repair points.
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
from scipy.optimize import lsq_linear, minimize
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_shared_late_capacity as shared_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_phase_separable_shrinkage_20260703"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

LINEAR_REG = 0.01
LATE_PRIORS = (0.25, 0.5, 1.0, 2.0, 4.0)
SHRINK_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TAU_GRID = (None, np.log1p(4.0), np.log1p(8.0), np.log1p(15.0), np.log1p(30.0))


@dataclass(frozen=True)
class PhaseSeparableFit:
    intercept: float
    coef: np.ndarray
    base_model: dsp.FittedDSPModel
    late_prior: float
    shrink_lambda: float
    pooled_tau: float | None
    pooled_kappa: np.ndarray


@dataclass(frozen=True)
class SummaryRow:
    model_name: str
    late_prior: float | None
    shrink_lambda: float | None
    pooled_tau: float | None
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_lower_tail_optimism: float
    heldout_uncheatable_mae: float
    heldout_all_mae: float
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


def row_passes(row: dict[str, Any]) -> bool:
    raw_ok = row["raw_predicted_optimum"] is None or float(row["raw_predicted_optimum"]) >= 0.935
    return bool(
        row["oof_spearman"] >= 0.70
        and row["heldout_uncheatable_mae"] <= 0.030
        and row["heldout_max_optimism"] <= 0.035
        and abs(row["heldout_gap_error"]) <= 0.004
        and raw_ok
        and row["leave_good_rmse"] <= 0.015
        and row["leave_good_signed_optimism"] <= 0.005
    )


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


def pooled_kappa(base_model: dsp.FittedDSPModel) -> np.ndarray:
    kappa = np.maximum(np.asarray(base_model.benefit_coef, dtype=float), 0.0)
    total = float(kappa.sum())
    if total <= 0.0:
        return np.full_like(kappa, 1.0 / len(kappa), dtype=float)
    return kappa / total


def phase_separable_design(
    weights: np.ndarray,
    base_model: dsp.FittedDSPModel,
    *,
    pooled_tau: float | None,
    pooled_kappa_values: np.ndarray,
) -> np.ndarray:
    rho = np.asarray(base_model.params["rho"], dtype=float)[None, :]
    e0 = weights[:, 0, :] * base_model.c0[None, :]
    e1 = weights[:, 1, :] * base_model.c1[None, :]
    s0 = 1.0 - np.exp(-rho * e0)
    s1 = 1.0 - np.exp(-rho * e1)
    columns = [-s0, -s1]
    if pooled_tau is not None:
        total_exposure = e0 + e1
        pooled = (dsp.softplus(np.log1p(total_exposure) - float(pooled_tau)) ** 2) @ pooled_kappa_values
        columns.append(pooled[:, None])
    return np.hstack(columns)


def fit_phase_separable_head(
    weights: np.ndarray,
    y: np.ndarray,
    base_model: dsp.FittedDSPModel,
    *,
    late_prior: float,
    shrink_lambda: float,
    pooled_tau: float | None,
    pooled_kappa_values: np.ndarray,
) -> PhaseSeparableFit:
    design = phase_separable_design(
        weights,
        base_model,
        pooled_tau=pooled_tau,
        pooled_kappa_values=pooled_kappa_values,
    )
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(y.mean())
    matrix = design - design_mean
    target = y - target_mean
    if LINEAR_REG > 0.0:
        matrix = np.vstack([matrix, np.sqrt(LINEAR_REG) * np.eye(matrix.shape[1])])
        target = np.concatenate([target, np.zeros(matrix.shape[1], dtype=float)])
    if shrink_lambda > 0.0:
        m = len(base_model.domain_names)
        shrink = np.zeros((m, matrix.shape[1]), dtype=float)
        scale = float(np.sqrt(shrink_lambda))
        for idx in range(m):
            shrink[idx, idx] = -float(late_prior) * scale
            shrink[idx, m + idx] = scale
        matrix = np.vstack([matrix, shrink])
        target = np.concatenate([target, np.zeros(m, dtype=float)])
    result = lsq_linear(matrix, target, bounds=(0.0, np.inf), lsmr_tol="auto", max_iter=3000)
    if not result.success:
        raise RuntimeError(result.message)
    coef = np.asarray(result.x, dtype=float)
    intercept = float(target_mean - (design_mean @ coef).item())
    return PhaseSeparableFit(
        intercept=intercept,
        coef=coef,
        base_model=base_model,
        late_prior=float(late_prior),
        shrink_lambda=float(shrink_lambda),
        pooled_tau=None if pooled_tau is None else float(pooled_tau),
        pooled_kappa=np.asarray(pooled_kappa_values, dtype=float),
    )


def predict_phase_separable(fit: PhaseSeparableFit, weights: np.ndarray) -> np.ndarray:
    design = phase_separable_design(
        weights,
        fit.base_model,
        pooled_tau=fit.pooled_tau,
        pooled_kappa_values=fit.pooled_kappa,
    )
    return np.asarray(fit.intercept + design @ fit.coef, dtype=float)


def profile_score(y: np.ndarray, pred: np.ndarray) -> float:
    rmse, _mae, _pearson, _spearman = regression_metrics(y, pred)
    return rmse + 0.5 * lower_tail_optimism(y, pred)


def fit_best_phase_separable(
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    *,
    idx: np.ndarray | None = None,
) -> PhaseSeparableFit:
    train_weights = packet.w if idx is None else packet.w[idx]
    train_y = packet.y if idx is None else packet.y[idx]
    kappa = pooled_kappa(base_model)
    best_fit: PhaseSeparableFit | None = None
    best_score = float("inf")
    for late_prior in LATE_PRIORS:
        for shrink_lambda in SHRINK_LAMBDAS:
            for pooled_tau in TAU_GRID:
                fit = fit_phase_separable_head(
                    train_weights,
                    train_y,
                    base_model,
                    late_prior=late_prior,
                    shrink_lambda=shrink_lambda,
                    pooled_tau=pooled_tau,
                    pooled_kappa_values=kappa,
                )
                pred = predict_phase_separable(fit, train_weights)
                score = profile_score(train_y, pred)
                if score < best_score:
                    best_score = score
                    best_fit = fit
    if best_fit is None:
        raise RuntimeError("no phase-separable fit selected")
    return best_fit


def oof_phase_separable(packet: dsp.PacketData, base_model: dsp.FittedDSPModel) -> np.ndarray:
    folds = eta_diag.olmix.kfold_indices(len(packet.y), n_splits=eta_diag.olmix.N_SPLITS, seed=eta_diag.olmix.CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fit = fit_best_phase_separable(packet, base_model, idx=train_idx)
        oof[test_idx] = predict_phase_separable(fit, packet.w[test_idx])
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
                "absolute_error": abs(pred - actual),
                "optimism": actual - pred,
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
        "heldout_all_mae": float(repaired["absolute_error"].mean()),
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
    elif predict_kind == "phase_separable":
        fit = fit_best_phase_separable(packet, base_model, idx=train_idx)
        pred = predict_phase_separable(fit, packet.w[test_idx])
    else:
        raise ValueError(predict_kind)
    y = packet.y[test_idx]
    rmse, _mae, _pearson, _spearman = regression_metrics(y, pred)
    return rmse, float(np.mean(y - pred))


def raw_optimize_phase_separable(fit: PhaseSeparableFit, packet: dsp.PacketData) -> tuple[float, float, float]:
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
            lambda z: float(predict_phase_separable(fit, logits_to_weights(np.asarray(z, dtype=float))[None, :, :])[0]),
            start,
            method="L-BFGS-B",
            options={"maxiter": 250, "ftol": 1e-8},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("raw optimize failed")
    weights = logits_to_weights(np.asarray(best.x, dtype=float))
    distances = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    return float(best.fun), float(distances[nearest_idx]), float(packet.y[nearest_idx])


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
    row: dict[str, Any] = {
        "model_name": "no_phase",
        "late_prior": None,
        "shrink_lambda": None,
        "pooled_tau": None,
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


def summarize_phase_separable(
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    heldout: pd.DataFrame,
) -> tuple[SummaryRow, pd.DataFrame, pd.DataFrame]:
    grid_rows: list[dict[str, Any]] = []
    best_fit: PhaseSeparableFit | None = None
    best_score = float("inf")
    kappa = pooled_kappa(base_model)
    for late_prior in LATE_PRIORS:
        for shrink_lambda in SHRINK_LAMBDAS:
            for pooled_tau in TAU_GRID:
                fit = fit_phase_separable_head(
                    packet.w,
                    packet.y,
                    base_model,
                    late_prior=late_prior,
                    shrink_lambda=shrink_lambda,
                    pooled_tau=pooled_tau,
                    pooled_kappa_values=kappa,
                )
                pred = predict_phase_separable(fit, packet.w)
                rmse, _mae, _pearson, spearman = regression_metrics(packet.y, pred)
                score = profile_score(packet.y, pred)
                grid_rows.append(
                    {
                        "late_prior": float(late_prior),
                        "shrink_lambda": float(shrink_lambda),
                        "pooled_tau": None if pooled_tau is None else float(pooled_tau),
                        "train_rmse": float(rmse),
                        "train_spearman": float(spearman),
                        "selection_score": float(score),
                    }
                )
                if score < best_score:
                    best_score = score
                    best_fit = fit
    if best_fit is None:
        raise RuntimeError("no phase-separable fit selected")
    train_pred = predict_phase_separable(best_fit, packet.w)
    train_rmse, _mae, _pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof = oof_phase_separable(packet, base_model)
    oof_rmse, _mae, _pearson, oof_spearman = regression_metrics(packet.y, oof)
    repaired = repaired_predictions(base_model, lambda weights: predict_phase_separable(best_fit, weights), heldout)
    stats = repaired_gate_stats(repaired)
    leave_rmse, leave_optimism = leave_good_out(packet, base_model, "phase_separable")
    raw_value, raw_tv, raw_nearest = raw_optimize_phase_separable(best_fit, packet)
    row: dict[str, Any] = {
        "model_name": "phase_separable_shrinkage",
        "late_prior": best_fit.late_prior,
        "shrink_lambda": best_fit.shrink_lambda,
        "pooled_tau": best_fit.pooled_tau,
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
    repaired.insert(0, "model_name", "phase_separable_shrinkage")
    return SummaryRow(**row), repaired, pd.DataFrame(grid_rows)


def write_outputs(output_dir: Path, summary: pd.DataFrame, heldout: pd.DataFrame, grid: pd.DataFrame) -> None:
    fig = px.scatter(
        summary,
        x="oof_spearman",
        y="heldout_uncheatable_mae",
        color="model_name",
        symbol="pass_gates",
        hover_data=["late_prior", "shrink_lambda", "pooled_tau", "leave_good_rmse", "raw_predicted_optimum"],
        title="Phase-separable shrinkage DSP diagnostic",
        template="plotly_white",
    )
    fig.add_hline(y=0.030, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.70, line_dash="dot", line_color="gray")
    fig.update_layout(width=1000, height=650)
    fig.write_html(output_dir / "phase_separable_gate_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = px.bar(
        heldout,
        x="mixture",
        y="prediction_error",
        color="model_name",
        barmode="group",
        title="Phase-separable heldout repaired-candidate errors",
        template="plotly_white",
    )
    fig2.update_layout(width=1250, height=620, xaxis_tickangle=-20)
    fig2.write_html(output_dir / "phase_separable_heldout_errors.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig3 = px.scatter(
        grid,
        x="train_spearman",
        y="train_rmse",
        color="shrink_lambda",
        symbol="late_prior",
        facet_col="pooled_tau",
        hover_data=["selection_score"],
        title="Phase-separable shrinkage grid profile",
        template="plotly_white",
    )
    fig3.update_layout(width=1450, height=650)
    fig3.write_html(output_dir / "phase_separable_grid.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = [
        "# Phase-separable shrinkage DSP diagnostic",
        "",
        "This tests whether phase placement needs separate early and late benefit channels rather than another scalar phase premium. Nonlinear exposure shapes are frozen from no-phase DSP; late coefficients are shrunk toward a global multiple of early coefficients by pseudo-observations.",
        "",
        "## Gate summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best training-profile grid rows",
        "",
        grid.sort_values("selection_score").head(20).to_markdown(index=False, floatfmt=".6f"),
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
    print("Summarizing phase-separable shrinkage", flush=True)
    phase_row, phase_heldout, grid = summarize_phase_separable(packet, base_model, heldout)
    summary = pd.DataFrame([asdict(no_phase_row), asdict(phase_row)])
    heldout_frame = pd.concat([no_phase_heldout, phase_heldout], ignore_index=True)
    summary.to_csv(args.output_dir / "phase_separable_summary.csv", index=False)
    heldout_frame.to_csv(args.output_dir / "phase_separable_heldout_predictions.csv", index=False)
    grid.to_csv(args.output_dir / "phase_separable_grid.csv", index=False)
    write_outputs(args.output_dir, summary, heldout_frame, grid)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "linear_reg": LINEAR_REG,
                "late_priors": list(LATE_PRIORS),
                "shrink_lambdas": list(SHRINK_LAMBDAS),
                "tau_grid": [None if tau is None else float(tau) for tau in TAU_GRID],
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
