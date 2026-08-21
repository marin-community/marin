# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose support-aware DSP shrinkage for Uncheatable BPB.

This is not another benefit curve. It treats the phase-aware DSP residual over
the no-phase predictor as an uncertainty-sensitive correction: use the phase
correction inside the empirical fixed-DSP feature support, and shrink it toward
the no-phase model outside support.

Canonical support rule:

    pred(w) = pred_no_phase(w) + alpha(w) * (pred_phase(w) - pred_no_phase(w))

where `alpha=1` below the training panel's 95th percentile leave-one-out
distance in the fixed effective-exposure DSP design space, linearly tapers to
0 at the panel maximum leave-one-out distance, and stays 0 beyond that.
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
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_l2_kl_sweep_deletion_augmented_300m as l2_sweep,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_support_aware_shrinkage_20260703"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LINEAR_REG = 0.01


@dataclass(frozen=True)
class SupportScaler:
    mean: np.ndarray
    scale: np.ndarray
    low_distance: float
    high_distance: float


@dataclass(frozen=True)
class SupportAwareFit:
    no_phase_model: dsp.FittedDSPModel
    phase_model: dsp.FittedDSPModel
    scaler: SupportScaler
    train_design: np.ndarray
    train_targets: np.ndarray
    low_quantile: float
    high_quantile: float
    local_k: int
    optimism_delta: float | None
    floor_alpha_threshold: float


@dataclass(frozen=True)
class SummaryRow:
    model_name: str
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
    parser.add_argument("--low-quantile", type=float, default=0.95)
    parser.add_argument("--high-quantile", type=float, default=1.0)
    parser.add_argument("--local-k", type=int, default=5)
    parser.add_argument("--floor-quantile", type=float, default=0.95)
    parser.add_argument("--floor-alpha-threshold", type=float, default=0.5)
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
        and row["heldout_best_order_correct"]
        and raw_ok
        and row["leave_good_rmse"] <= 0.015
        and row["leave_good_signed_optimism"] <= 0.005
    )


def fit_model(packet: dsp.PacketData, variant_key: str, args: argparse.Namespace) -> dsp.FittedDSPModel:
    model, _train_metrics, _oof_metrics, _oof_pred, _tuning = l2_sweep.fit_one_dsp_model(
        packet,
        variant_key=variant_key,
        linear_reg=LINEAR_REG,
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
        basin_hopping_iters=int(args.basin_hopping_iters),
    )
    return model


def fixed_dsp_design(model: dsp.FittedDSPModel, packet: dsp.PacketData, weights: np.ndarray) -> np.ndarray:
    signal, penalty = dsp.features(weights, packet.c0, packet.c1, model.variant, model.params)
    return np.hstack([-signal, penalty])


def fit_support_scaler(design: np.ndarray, low_quantile: float, high_quantile: float) -> SupportScaler:
    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (design - mean[None, :]) / scale[None, :]
    neighbors = min(2, len(standardized))
    distances, _indices = NearestNeighbors(n_neighbors=neighbors).fit(standardized).kneighbors(standardized)
    leave_one_out = distances[:, 1] if neighbors > 1 else np.zeros(len(standardized), dtype=float)
    low_distance = float(np.quantile(leave_one_out, low_quantile))
    high_distance = float(np.quantile(leave_one_out, high_quantile))
    if high_distance <= low_distance:
        high_distance = low_distance + 1e-8
    return SupportScaler(mean=mean, scale=scale, low_distance=low_distance, high_distance=high_distance)


def support_alpha(
    train_design: np.ndarray,
    scaler: SupportScaler,
    query_design: np.ndarray,
    *,
    local_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_z = (train_design - scaler.mean[None, :]) / scaler.scale[None, :]
    query_z = (query_design - scaler.mean[None, :]) / scaler.scale[None, :]
    neighbor_count = min(max(int(local_k), 1), len(train_z))
    distance_matrix, indices = NearestNeighbors(n_neighbors=neighbor_count).fit(train_z).kneighbors(query_z)
    distance = distance_matrix[:, 0]
    alpha = np.clip((scaler.high_distance - distance) / (scaler.high_distance - scaler.low_distance), 0.0, 1.0)
    alpha = np.where(distance <= scaler.low_distance, 1.0, alpha)
    return distance, alpha, indices


def build_support_fit(
    packet: dsp.PacketData,
    no_phase_model: dsp.FittedDSPModel,
    phase_model: dsp.FittedDSPModel,
    *,
    low_quantile: float,
    high_quantile: float,
) -> SupportAwareFit:
    train_design = fixed_dsp_design(phase_model, packet, packet.w)
    scaler = fit_support_scaler(train_design, low_quantile=low_quantile, high_quantile=high_quantile)
    return SupportAwareFit(
        no_phase_model=no_phase_model,
        phase_model=phase_model,
        scaler=scaler,
        train_design=train_design,
        train_targets=np.asarray(packet.y, dtype=float),
        low_quantile=float(low_quantile),
        high_quantile=float(high_quantile),
        local_k=5,
        optimism_delta=None,
        floor_alpha_threshold=0.5,
    )


def predict_support(fit: SupportAwareFit, packet: dsp.PacketData, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_design = fixed_dsp_design(fit.phase_model, packet, weights)
    distance, alpha, indices = support_alpha(
        fit.train_design,
        fit.scaler,
        query_design,
        local_k=fit.local_k,
    )
    no_pred = dsp.predict(fit.no_phase_model, weights)
    phase_pred = dsp.predict(fit.phase_model, weights)
    pred = no_pred + alpha * (phase_pred - no_pred)
    if fit.optimism_delta is not None:
        local_best = fit.train_targets[indices].min(axis=1)
        floor = local_best - float(fit.optimism_delta)
        pred = np.where(alpha >= fit.floor_alpha_threshold, np.maximum(pred, floor), pred)
    return pred, distance, alpha


def with_floor_calibration(
    fit: SupportAwareFit,
    *,
    local_k: int,
    optimism_delta: float | None,
    floor_alpha_threshold: float,
) -> SupportAwareFit:
    return SupportAwareFit(
        no_phase_model=fit.no_phase_model,
        phase_model=fit.phase_model,
        scaler=fit.scaler,
        train_design=fit.train_design,
        train_targets=fit.train_targets,
        low_quantile=fit.low_quantile,
        high_quantile=fit.high_quantile,
        local_k=int(local_k),
        optimism_delta=None if optimism_delta is None else float(optimism_delta),
        floor_alpha_threshold=float(floor_alpha_threshold),
    )


def oof_support(packet: dsp.PacketData, no_phase_model: dsp.FittedDSPModel, phase_model: dsp.FittedDSPModel, args: argparse.Namespace) -> np.ndarray:
    folds = eta_diag.olmix.kfold_indices(len(packet.y), n_splits=eta_diag.olmix.N_SPLITS, seed=eta_diag.olmix.CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fold_no = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, no_phase_model.variant, no_phase_model.params)
        fold_phase = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, phase_model.variant, phase_model.params)
        train_design = fixed_dsp_design(fold_phase, packet, packet.w[train_idx])
        scaler = fit_support_scaler(
            train_design,
            low_quantile=float(args.low_quantile),
            high_quantile=float(args.high_quantile),
        )
        query_design = fixed_dsp_design(fold_phase, packet, packet.w[test_idx])
        _distance, alpha, indices = support_alpha(train_design, scaler, query_design, local_k=int(args.local_k))
        no_pred = dsp.predict(fold_no, packet.w[test_idx])
        phase_pred = dsp.predict(fold_phase, packet.w[test_idx])
        pred = no_pred + alpha * (phase_pred - no_pred)
        if getattr(args, "optimism_delta", None) is not None:
            local_best = packet.y[train_idx][indices].min(axis=1)
            floor = local_best - float(args.optimism_delta)
            pred = np.where(alpha >= float(args.floor_alpha_threshold), np.maximum(pred, floor), pred)
        oof[test_idx] = pred
    return oof


def calibrate_optimism_delta(
    packet: dsp.PacketData,
    no_phase_model: dsp.FittedDSPModel,
    phase_model: dsp.FittedDSPModel,
    args: argparse.Namespace,
) -> float:
    folds = eta_diag.olmix.kfold_indices(len(packet.y), n_splits=eta_diag.olmix.N_SPLITS, seed=eta_diag.olmix.CV_SEED)
    optimism: list[float] = []
    for train_idx, test_idx in folds:
        fold_no = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, no_phase_model.variant, no_phase_model.params)
        fold_phase = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, phase_model.variant, phase_model.params)
        train_design = fixed_dsp_design(fold_phase, packet, packet.w[train_idx])
        scaler = fit_support_scaler(
            train_design,
            low_quantile=float(args.low_quantile),
            high_quantile=float(args.high_quantile),
        )
        query_design = fixed_dsp_design(fold_phase, packet, packet.w[test_idx])
        _distance, alpha, indices = support_alpha(train_design, scaler, query_design, local_k=int(args.local_k))
        pred = dsp.predict(fold_no, packet.w[test_idx]) + alpha * (
            dsp.predict(fold_phase, packet.w[test_idx]) - dsp.predict(fold_no, packet.w[test_idx])
        )
        local_best = packet.y[train_idx][indices].min(axis=1)
        optimism.extend(np.maximum(local_best - pred, 0.0).tolist())
    return float(np.quantile(np.asarray(optimism, dtype=float), float(args.floor_quantile)))


def repaired_predictions(fit: SupportAwareFit, packet: dsp.PacketData, heldout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(fit.phase_model, Path(row.mixture_path))
        pred, distance, alpha = predict_support(fit, packet, weights)
        actual = float(row.uncheatable_bpb)
        rows.append(
            {
                "mixture": str(row.mixture),
                "is_uncheatable_objective": bool(row.is_uncheatable_objective),
                "actual_uncheatable_bpb": actual,
                "predicted_uncheatable_bpb": float(pred[0]),
                "prediction_error": float(pred[0]) - actual,
                "absolute_error": abs(float(pred[0]) - actual),
                "optimism": actual - float(pred[0]),
                "support_distance": float(distance[0]),
                "support_alpha": float(alpha[0]),
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


def leave_good_out(
    packet: dsp.PacketData,
    no_phase_model: dsp.FittedDSPModel,
    phase_model: dsp.FittedDSPModel,
    args: argparse.Namespace,
) -> tuple[float, float]:
    holdout_count = max(5, int(np.ceil(0.15 * len(packet.y))))
    test_idx = np.argsort(packet.y)[:holdout_count]
    train_idx = np.setdiff1d(np.arange(len(packet.y)), test_idx)
    fold_no = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, no_phase_model.variant, no_phase_model.params)
    fold_phase = dsp.fit_linear_head(packet.w[train_idx], packet.y[train_idx], packet, phase_model.variant, phase_model.params)
    train_design = fixed_dsp_design(fold_phase, packet, packet.w[train_idx])
    scaler = fit_support_scaler(train_design, low_quantile=float(args.low_quantile), high_quantile=float(args.high_quantile))
    query_design = fixed_dsp_design(fold_phase, packet, packet.w[test_idx])
    _distance, alpha, indices = support_alpha(train_design, scaler, query_design, local_k=int(args.local_k))
    pred = dsp.predict(fold_no, packet.w[test_idx]) + alpha * (
        dsp.predict(fold_phase, packet.w[test_idx]) - dsp.predict(fold_no, packet.w[test_idx])
    )
    if getattr(args, "optimism_delta", None) is not None:
        local_best = packet.y[train_idx][indices].min(axis=1)
        floor = local_best - float(args.optimism_delta)
        pred = np.where(alpha >= float(args.floor_alpha_threshold), np.maximum(pred, floor), pred)
    y = packet.y[test_idx]
    rmse, _mae, _pearson, _spearman = regression_metrics(y, pred)
    return rmse, float(np.mean(y - pred))


def raw_optimize_support(fit: SupportAwareFit, packet: dsp.PacketData) -> tuple[float, float, float]:
    n = len(fit.phase_model.domain_names)

    def logits_to_weights(logits: np.ndarray) -> np.ndarray:
        logits0 = logits[:n]
        logits1 = logits[n:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)
        return np.stack([p0, p1], axis=0)

    starts = [np.zeros(2 * n, dtype=float)]
    observed_values, _distance, _alpha = predict_support(fit, packet, packet.w)
    for idx in np.argsort(observed_values)[:16]:
        starts.append(dsp.weights_to_logits(packet.w[int(idx)]))
    best_value = float("inf")
    best_weights: np.ndarray | None = None
    for start in starts:
        result = minimize(
            lambda z: float(predict_support(fit, packet, logits_to_weights(np.asarray(z, dtype=float))[None, :, :])[0][0]),
            start,
            method="L-BFGS-B",
            options={"maxiter": 180, "ftol": 1e-8},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_weights = logits_to_weights(np.asarray(result.x, dtype=float))
    if best_weights is None:
        raise RuntimeError("raw optimize failed")
    distances = dsp.average_phase_tv_distance(packet.w, best_weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    return best_value, float(distances[nearest_idx]), float(packet.y[nearest_idx])


def write_outputs(output_dir: Path, summary: pd.DataFrame, heldout: pd.DataFrame) -> None:
    fig = px.scatter(
        summary,
        x="oof_spearman",
        y="heldout_uncheatable_mae",
        color="model_name",
        symbol="pass_gates",
        hover_data=["leave_good_rmse", "raw_predicted_optimum"],
        title="Support-aware DSP shrinkage diagnostic",
        template="plotly_white",
    )
    fig.add_hline(y=0.030, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.70, line_dash="dot", line_color="gray")
    fig.update_layout(width=1000, height=650)
    fig.write_html(output_dir / "support_aware_gate_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = px.bar(
        heldout,
        x="mixture",
        y="prediction_error",
        color="support_alpha",
        hover_data=["support_distance", "support_alpha"],
        title="Support-aware heldout repaired-candidate errors",
        template="plotly_white",
    )
    fig2.update_layout(width=1250, height=620, xaxis_tickangle=-20)
    fig2.write_html(output_dir / "support_aware_heldout_errors.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = [
        "# Support-aware DSP shrinkage diagnostic",
        "",
        "This tests a model-uncertainty correction: phase-aware DSP is used inside the empirical fixed-DSP feature support, and the phase residual is shrunk to no-phase outside support.",
        "",
        "## Gate summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Heldout repaired candidates",
        "",
        heldout.to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet, *_ = eta_diag.load_packet()
    heldout = eta_diag.load_heldout(args)
    print("Fitting no-phase model", flush=True)
    no_phase_model = fit_model(packet, "no_phase", args)
    print("Fitting effective-exposure phase model", flush=True)
    phase_model = fit_model(packet, "effective_exposure", args)
    fit = build_support_fit(
        packet,
        no_phase_model,
        phase_model,
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
    )
    optimism_delta = calibrate_optimism_delta(packet, no_phase_model, phase_model, args)
    args.optimism_delta = optimism_delta
    fit = with_floor_calibration(
        fit,
        local_k=int(args.local_k),
        optimism_delta=optimism_delta,
        floor_alpha_threshold=float(args.floor_alpha_threshold),
    )
    train_pred, _train_distance, _train_alpha = predict_support(fit, packet, packet.w)
    train_rmse, _mae, _pearson, train_spearman = regression_metrics(packet.y, train_pred)
    print("Computing OOF support-aware predictions", flush=True)
    oof = oof_support(packet, no_phase_model, phase_model, args)
    oof_rmse, _oof_mae, _oof_pearson, oof_spearman = regression_metrics(packet.y, oof)
    repaired = repaired_predictions(fit, packet, heldout)
    stats = repaired_gate_stats(repaired)
    leave_rmse, leave_optimism = leave_good_out(packet, no_phase_model, phase_model, args)
    raw_value, raw_tv, raw_nearest = raw_optimize_support(fit, packet)
    row: dict[str, Any] = {
        "model_name": "support_aware_effective_exposure",
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
    summary = pd.DataFrame([SummaryRow(**row)])
    summary.to_csv(args.output_dir / "support_aware_summary.csv", index=False)
    repaired.to_csv(args.output_dir / "support_aware_heldout_predictions.csv", index=False)
    pd.DataFrame({"oof_prediction": oof, "target": packet.y}).to_csv(args.output_dir / "support_aware_oof_predictions.csv", index=False)
    write_outputs(args.output_dir, summary, repaired)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "linear_reg": LINEAR_REG,
                "low_quantile": float(args.low_quantile),
                "high_quantile": float(args.high_quantile),
                "support_low_distance": fit.scaler.low_distance,
                "support_high_distance": fit.scaler.high_distance,
                "local_k": int(args.local_k),
                "floor_quantile": float(args.floor_quantile),
                "optimism_delta": optimism_delta,
                "floor_alpha_threshold": float(args.floor_alpha_threshold),
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
