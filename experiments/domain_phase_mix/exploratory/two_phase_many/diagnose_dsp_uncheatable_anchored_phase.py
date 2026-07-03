# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose anchored phase-residual DSP variants for Uncheatable BPB.

The previous diagnostic showed that retuning the effective-exposure phase
multiplier does not fix the repaired-candidate heldout failure. This script
tests a more mechanistic alternative:

1. Fit a no-phase aggregate-exposure DSP base.
2. Freeze the base exposure semantics.
3. Fit only a centered and bounded phase-share residual.

No jobs are launched and no remote state is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_dsp_uncheatable_eta_heldout as eta_diag  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_anchored_phase_residual_20260703"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LINEAR_REG = 0.01
EPS = 1e-12

ResidualMode = Literal["signed_ridge", "late_helps_nnls", "bidirectional_nnls"]


@dataclass(frozen=True)
class ResidualFit:
    mode: ResidualMode
    ridge: float
    clip_scale: float | None
    center: np.ndarray
    intercept: float
    coef: np.ndarray
    residual_clip: float | None


@dataclass(frozen=True)
class VariantResult:
    model_name: str
    mode: str
    ridge: float
    clip_scale: float | None
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_lower_tail_optimism: float
    oof_low_tail_rmse: float
    leave_extreme_rmse: float
    leave_extreme_spearman: float
    leave_extreme_signed_optimism: float
    heldout_all_mae: float
    heldout_uncheatable_mae: float
    heldout_predicted_best_uncheatable: str
    heldout_actual_best_uncheatable: str
    heldout_best_order_correct: bool
    residual_clip: float | None
    selection_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-results", type=Path, default=eta_diag.DEFAULT_REPAIR_RESULTS)
    parser.add_argument("--repair-mixture-dir", type=Path, default=eta_diag.DEFAULT_REPAIR_MIXTURE_DIR)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    parser.add_argument("--extreme-frac", type=float, default=0.15)
    return parser.parse_args()


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    spearman = float(spearmanr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def lower_tail_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    tail_count = max(5, int(np.ceil(dsp.LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(y_hat)[:tail_count]
    residual = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    optimism = float(np.mean(np.maximum(y[tail_idx] - y_hat[tail_idx], 0.0)))
    low_tail_rmse = float(np.sqrt(np.mean(residual[tail_idx] * residual[tail_idx])))
    return optimism, low_tail_rmse


def no_phase_signal(model: dsp.FittedDSPModel, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = weights[:, 0, :]
    p1 = weights[:, 1, :]
    e0 = p0 * model.c0[None, :]
    e1 = p1 * model.c1[None, :]
    exposure = e0 + e1
    rho = np.asarray(model.params["rho"], dtype=float)[None, :]
    signal = 1.0 - np.exp(-rho * exposure)
    return exposure, e1, signal


def residual_base_features(
    model: dsp.FittedDSPModel,
    weights: np.ndarray,
    *,
    center: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    exposure, phase1_exposure, signal = no_phase_signal(model, weights)
    phase1_share = phase1_exposure / np.maximum(exposure, EPS)
    if center is None:
        center = np.mean(phase1_share, axis=0)
    features = (phase1_share - center[None, :]) * signal
    return features, center


def design_matrix(base_features: np.ndarray, mode: ResidualMode) -> np.ndarray:
    if mode == "signed_ridge":
        return base_features
    if mode == "late_helps_nnls":
        return -base_features
    if mode == "bidirectional_nnls":
        return np.hstack([-base_features, base_features])
    raise ValueError(f"Unknown mode: {mode}")


def fit_residual(
    base_model: dsp.FittedDSPModel,
    weights: np.ndarray,
    base_pred: np.ndarray,
    target: np.ndarray,
    *,
    mode: ResidualMode,
    ridge: float,
    clip_scale: float | None,
) -> ResidualFit:
    base_features, center = residual_base_features(base_model, weights, center=None)
    design = design_matrix(base_features, mode)
    target_residual = np.asarray(target, dtype=float) - np.asarray(base_pred, dtype=float)
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(np.mean(target_residual))
    centered_design = design - design_mean
    centered_target = target_residual - target_mean
    if mode == "signed_ridge":
        lhs = centered_design.T @ centered_design + ridge * np.eye(centered_design.shape[1])
        rhs = centered_design.T @ centered_target
        coef = np.linalg.solve(lhs, rhs)
    else:
        if ridge > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(ridge) * np.eye(centered_design.shape[1])])
            centered_target = np.concatenate([centered_target, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_target)
    intercept = float(target_mean - (design_mean @ coef).item())
    fitted_residual = intercept + design @ coef
    residual_clip = None
    if clip_scale is not None:
        residual_clip = float(clip_scale * np.max(np.abs(fitted_residual)))
    return ResidualFit(
        mode=mode,
        ridge=float(ridge),
        clip_scale=clip_scale,
        center=np.asarray(center, dtype=float),
        intercept=intercept,
        coef=np.asarray(coef, dtype=float),
        residual_clip=residual_clip,
    )


def predict_residual(base_model: dsp.FittedDSPModel, weights: np.ndarray, fit: ResidualFit) -> np.ndarray:
    base_features, _center = residual_base_features(base_model, weights, center=fit.center)
    design = design_matrix(base_features, fit.mode)
    residual = fit.intercept + design @ fit.coef
    if fit.residual_clip is not None:
        residual = np.clip(residual, -fit.residual_clip, fit.residual_clip)
    return np.asarray(residual, dtype=float)


def fit_base_head(
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    train_idx: np.ndarray,
) -> dsp.FittedDSPModel:
    return dsp.fit_linear_head(
        packet.w[train_idx],
        packet.y[train_idx],
        packet,
        base_model.variant,
        base_model.params,
    )


def kfold_oof_predictions(
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    *,
    mode: ResidualMode,
    ridge: float,
    clip_scale: float | None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    folds = olmix.kfold_indices(len(packet.y), n_splits=olmix.N_SPLITS, seed=olmix.CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fold_base = fit_base_head(packet, base_model, train_idx)
        train_base_pred = dsp.predict(fold_base, packet.w[train_idx])
        test_base_pred = dsp.predict(fold_base, packet.w[test_idx])
        residual_fit = fit_residual(
            fold_base,
            packet.w[train_idx],
            train_base_pred,
            packet.y[train_idx],
            mode=mode,
            ridge=ridge,
            clip_scale=clip_scale,
        )
        oof[test_idx] = test_base_pred + predict_residual(fold_base, packet.w[test_idx], residual_fit)
    return oof, folds


def extreme_holdout_indices(
    packet: dsp.PacketData,
    natural: np.ndarray,
    *,
    extreme_frac: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    reference = np.stack([natural, natural], axis=0)
    phase_tv = 0.5 * np.sum(np.abs(packet.w - reference[None, :, :]), axis=(1, 2))
    exposure = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    max_epoch = np.max(exposure, axis=1)
    phase1_share = (packet.w[:, 1, :] * packet.c1[None, :]) / np.maximum(exposure, EPS)
    phase1_share_span = np.nanmax(phase1_share, axis=1) - np.nanmin(phase1_share, axis=1)
    score = np.array(pd.Series(phase_tv).rank(pct=True).to_numpy(), dtype=float, copy=True)
    score += pd.Series(max_epoch).rank(pct=True).to_numpy()
    score += pd.Series(phase1_share_span).rank(pct=True).to_numpy()
    holdout_count = max(5, int(np.ceil(extreme_frac * len(packet.y))))
    test_idx = np.argsort(score)[-holdout_count:]
    train_idx = np.setdiff1d(np.arange(len(packet.y)), test_idx)
    frame = pd.DataFrame(
        {
            "run_name": packet.frame[packet.name_col].to_numpy(),
            "extreme_score": score,
            "phase_tv_to_proportional": phase_tv,
            "max_simulated_epoch": max_epoch,
            "phase1_share_span": phase1_share_span,
            "is_extreme_holdout": False,
        }
    )
    frame.loc[test_idx, "is_extreme_holdout"] = True
    return train_idx, test_idx, frame


def leave_extreme_predictions(
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    natural: np.ndarray,
    *,
    mode: ResidualMode,
    ridge: float,
    clip_scale: float | None,
    extreme_frac: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_idx, test_idx, frame = extreme_holdout_indices(packet, natural, extreme_frac=extreme_frac)
    fold_base = fit_base_head(packet, base_model, train_idx)
    train_base_pred = dsp.predict(fold_base, packet.w[train_idx])
    test_base_pred = dsp.predict(fold_base, packet.w[test_idx])
    residual_fit = fit_residual(
        fold_base,
        packet.w[train_idx],
        train_base_pred,
        packet.y[train_idx],
        mode=mode,
        ridge=ridge,
        clip_scale=clip_scale,
    )
    pred = test_base_pred + predict_residual(fold_base, packet.w[test_idx], residual_fit)
    return pred, test_idx, frame


def repaired_predictions(
    base_model: dsp.FittedDSPModel,
    residual_fit: ResidualFit,
    heldout: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(base_model, Path(row.mixture_path))
        base_pred = float(dsp.predict(base_model, weights)[0])
        residual = float(predict_residual(base_model, weights, residual_fit)[0])
        actual = float(row.uncheatable_bpb)
        rows.append(
            {
                "mixture": str(row.mixture),
                "is_uncheatable_objective": bool(row.is_uncheatable_objective),
                "actual_uncheatable_bpb": actual,
                "base_prediction": base_pred,
                "phase_residual_prediction": residual,
                "predicted_uncheatable_bpb": base_pred + residual,
                "prediction_error": base_pred + residual - actual,
                "absolute_error": abs(base_pred + residual - actual),
            }
        )
    return pd.DataFrame(rows)


def signed_optimism(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)))


def summarize_variant(
    *,
    name: str,
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    natural: np.ndarray,
    heldout: pd.DataFrame,
    mode: ResidualMode,
    ridge: float,
    clip_scale: float | None,
    extreme_frac: float,
) -> tuple[VariantResult, pd.DataFrame, pd.DataFrame]:
    base_pred = dsp.predict(base_model, packet.w)
    residual_fit = fit_residual(
        base_model,
        packet.w,
        base_pred,
        packet.y,
        mode=mode,
        ridge=ridge,
        clip_scale=clip_scale,
    )
    train_pred = base_pred + predict_residual(base_model, packet.w, residual_fit)
    train_rmse, _train_mae, _train_pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof, _folds = kfold_oof_predictions(
        packet,
        base_model,
        mode=mode,
        ridge=ridge,
        clip_scale=clip_scale,
    )
    oof_rmse, _oof_mae, _oof_pearson, oof_spearman = regression_metrics(packet.y, oof)
    oof_optimism, oof_low_tail_rmse = lower_tail_metrics(packet.y, oof)
    extreme_pred, extreme_idx, _extreme_frame = leave_extreme_predictions(
        packet,
        base_model,
        natural,
        mode=mode,
        ridge=ridge,
        clip_scale=clip_scale,
        extreme_frac=extreme_frac,
    )
    extreme_y = packet.y[extreme_idx]
    leave_extreme_rmse, _mae, _pearson, leave_extreme_spearman = regression_metrics(extreme_y, extreme_pred)
    leave_extreme_signed_optimism = signed_optimism(extreme_y, extreme_pred)
    heldout_predictions = repaired_predictions(base_model, residual_fit, heldout)
    all_errors = heldout_predictions["prediction_error"].to_numpy(dtype=float)
    uncheatable = heldout_predictions[heldout_predictions["is_uncheatable_objective"]].copy()
    uncheatable_errors = uncheatable["prediction_error"].to_numpy(dtype=float)
    predicted_best = str(uncheatable.loc[uncheatable["predicted_uncheatable_bpb"].idxmin(), "mixture"])
    actual_best = str(uncheatable.loc[uncheatable["actual_uncheatable_bpb"].idxmin(), "mixture"])
    selection_score = float(
        leave_extreme_rmse
        + 0.5 * max(leave_extreme_signed_optimism, 0.0)
        + 0.5 * np.mean(np.abs(uncheatable_errors))
    )
    result = VariantResult(
        model_name=name,
        mode=mode,
        ridge=float(ridge),
        clip_scale=clip_scale,
        train_rmse=float(train_rmse),
        train_spearman=float(train_spearman),
        oof_rmse=float(oof_rmse),
        oof_spearman=float(oof_spearman),
        oof_lower_tail_optimism=float(oof_optimism),
        oof_low_tail_rmse=float(oof_low_tail_rmse),
        leave_extreme_rmse=float(leave_extreme_rmse),
        leave_extreme_spearman=float(leave_extreme_spearman),
        leave_extreme_signed_optimism=float(leave_extreme_signed_optimism),
        heldout_all_mae=float(np.mean(np.abs(all_errors))),
        heldout_uncheatable_mae=float(np.mean(np.abs(uncheatable_errors))),
        heldout_predicted_best_uncheatable=predicted_best,
        heldout_actual_best_uncheatable=actual_best,
        heldout_best_order_correct=predicted_best == actual_best,
        residual_clip=residual_fit.residual_clip,
        selection_score=selection_score,
    )
    heldout_predictions.insert(0, "model_name", name)
    train_predictions = pd.DataFrame(
        {
            "model_name": name,
            "run_name": packet.frame[packet.name_col].to_numpy(),
            "actual": packet.y,
            "train_pred": train_pred,
            "oof_pred": oof,
        }
    )
    return result, heldout_predictions, train_predictions


def base_summary(
    packet: dsp.PacketData,
    base_model: dsp.FittedDSPModel,
    natural: np.ndarray,
    heldout: pd.DataFrame,
    *,
    extreme_frac: float,
) -> tuple[VariantResult, pd.DataFrame, pd.DataFrame]:
    train_pred = dsp.predict(base_model, packet.w)
    oof, _folds = eta_diag.dsp_compare.fit_dsp_oof_predictions(packet, base_model)
    extreme_pred, extreme_idx, _frame = leave_extreme_predictions(
        packet,
        base_model,
        natural,
        mode="signed_ridge",
        ridge=1e30,
        clip_scale=0.0,
        extreme_frac=extreme_frac,
    )
    # leave_extreme_predictions above degenerates to base plus zero residual.
    extreme_y = packet.y[extreme_idx]
    train_rmse, _mae, _pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof_rmse, _mae, _pearson, oof_spearman = regression_metrics(packet.y, oof)
    oof_optimism, oof_low_tail_rmse = lower_tail_metrics(packet.y, oof)
    leave_extreme_rmse, _mae, _pearson, leave_extreme_spearman = regression_metrics(extreme_y, extreme_pred)
    heldout_predictions = []
    for row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(base_model, Path(row.mixture_path))
        pred = float(dsp.predict(base_model, weights)[0])
        actual = float(row.uncheatable_bpb)
        heldout_predictions.append(
            {
                "model_name": "no_phase_base",
                "mixture": str(row.mixture),
                "is_uncheatable_objective": bool(row.is_uncheatable_objective),
                "actual_uncheatable_bpb": actual,
                "base_prediction": pred,
                "phase_residual_prediction": 0.0,
                "predicted_uncheatable_bpb": pred,
                "prediction_error": pred - actual,
                "absolute_error": abs(pred - actual),
            }
        )
    heldout_frame = pd.DataFrame(heldout_predictions)
    uncheatable = heldout_frame[heldout_frame["is_uncheatable_objective"]].copy()
    predicted_best = str(uncheatable.loc[uncheatable["predicted_uncheatable_bpb"].idxmin(), "mixture"])
    actual_best = str(uncheatable.loc[uncheatable["actual_uncheatable_bpb"].idxmin(), "mixture"])
    all_errors = heldout_frame["prediction_error"].to_numpy(dtype=float)
    uncheatable_errors = uncheatable["prediction_error"].to_numpy(dtype=float)
    result = VariantResult(
        model_name="no_phase_base",
        mode="base",
        ridge=float("nan"),
        clip_scale=None,
        train_rmse=float(train_rmse),
        train_spearman=float(train_spearman),
        oof_rmse=float(oof_rmse),
        oof_spearman=float(oof_spearman),
        oof_lower_tail_optimism=float(oof_optimism),
        oof_low_tail_rmse=float(oof_low_tail_rmse),
        leave_extreme_rmse=float(leave_extreme_rmse),
        leave_extreme_spearman=float(leave_extreme_spearman),
        leave_extreme_signed_optimism=signed_optimism(extreme_y, extreme_pred),
        heldout_all_mae=float(np.mean(np.abs(all_errors))),
        heldout_uncheatable_mae=float(np.mean(np.abs(uncheatable_errors))),
        heldout_predicted_best_uncheatable=predicted_best,
        heldout_actual_best_uncheatable=actual_best,
        heldout_best_order_correct=predicted_best == actual_best,
        residual_clip=None,
        selection_score=float(leave_extreme_rmse + 0.5 * np.mean(np.abs(uncheatable_errors))),
    )
    train_frame = pd.DataFrame(
        {
            "model_name": "no_phase_base",
            "run_name": packet.frame[packet.name_col].to_numpy(),
            "actual": packet.y,
            "train_pred": train_pred,
            "oof_pred": oof,
        }
    )
    return result, heldout_frame, train_frame


def write_plots(output_dir: Path, summary: pd.DataFrame, heldout: pd.DataFrame) -> None:
    fig = px.scatter(
        summary,
        x="leave_extreme_rmse",
        y="heldout_uncheatable_mae",
        color="mode",
        symbol="heldout_best_order_correct",
        hover_data=["model_name", "ridge", "clip_scale", "oof_spearman", "selection_score"],
        title="Anchored phase-residual DSP diagnostics for Uncheatable BPB",
        labels={
            "leave_extreme_rmse": "leave-extreme-out RMSE",
            "heldout_uncheatable_mae": "repaired uncheatable heldout MAE",
        },
        template="plotly_white",
    )
    fig.update_layout(width=1100, height=700)
    fig.write_html(output_dir / "anchored_phase_residual_frontier.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    best_names = summary.sort_values("selection_score").head(8)["model_name"].tolist()
    plot_frame = heldout[heldout["model_name"].isin(best_names + ["no_phase_base"])].copy()
    fig2 = px.bar(
        plot_frame,
        x="mixture",
        y="prediction_error",
        color="model_name",
        barmode="group",
        title="Heldout repaired-candidate prediction errors",
        labels={"prediction_error": "prediction - actual BPB"},
        template="plotly_white",
    )
    fig2.update_layout(width=1300, height=650, xaxis_tickangle=-20)
    fig2.write_html(output_dir / "anchored_phase_residual_heldout_errors.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    best = summary.sort_values("selection_score").head(12)
    lines = [
        "# Anchored phase-residual DSP diagnostic",
        "",
        "This diagnostic freezes no-phase aggregate exposure and fits only a centered phase-share residual.",
        "",
        "## Best variants by selection score",
        "",
        best[
            [
                "model_name",
                "mode",
                "ridge",
                "clip_scale",
                "oof_spearman",
                "leave_extreme_rmse",
                "leave_extreme_signed_optimism",
                "heldout_uncheatable_mae",
                "heldout_best_order_correct",
                "selection_score",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Gate interpretation",
        "",
        "A phase residual should beat no-phase repaired-heldout MAE, preserve nontrivial OOF Spearman, and avoid leave-extreme-out optimism. If it cannot, phase information should be treated as unsupported for this target rather than deployment-regularized after optimization.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet, _panel, _domains, natural, _token_counts, _target_budget = eta_diag.load_packet()
    heldout = eta_diag.load_heldout(args)

    print("Fitting no_phase base", flush=True)
    base_model = eta_diag.fit_model(packet, "no_phase", args)
    results: list[VariantResult] = []
    heldout_frames: list[pd.DataFrame] = []
    train_frames: list[pd.DataFrame] = []
    base_result, base_heldout, base_train = base_summary(
        packet,
        base_model,
        natural,
        heldout,
        extreme_frac=float(args.extreme_frac),
    )
    results.append(base_result)
    heldout_frames.append(base_heldout)
    train_frames.append(base_train)

    ridges = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    clip_scales: list[float | None] = [None, 1.0, 0.5]
    modes: list[ResidualMode] = ["signed_ridge", "late_helps_nnls", "bidirectional_nnls"]
    for mode in modes:
        for ridge in ridges:
            for clip_scale in clip_scales:
                name = f"anchored_{mode}_ridge{ridge:g}_clip{clip_scale if clip_scale is not None else 'none'}"
                print(f"Evaluating {name}", flush=True)
                result, heldout_pred, train_pred = summarize_variant(
                    name=name,
                    packet=packet,
                    base_model=base_model,
                    natural=natural,
                    heldout=heldout,
                    mode=mode,
                    ridge=ridge,
                    clip_scale=clip_scale,
                    extreme_frac=float(args.extreme_frac),
                )
                results.append(result)
                heldout_frames.append(heldout_pred)
                train_frames.append(train_pred)

    summary = pd.DataFrame([asdict(result) for result in results]).sort_values("selection_score")
    heldout_predictions = pd.concat(heldout_frames, ignore_index=True)
    train_predictions = pd.concat(train_frames, ignore_index=True)
    _train_idx, _test_idx, extreme_frame = extreme_holdout_indices(packet, natural, extreme_frac=float(args.extreme_frac))

    summary.to_csv(args.output_dir / "anchored_phase_residual_summary.csv", index=False)
    heldout_predictions.to_csv(args.output_dir / "anchored_phase_residual_heldout_predictions.csv", index=False)
    train_predictions.to_csv(args.output_dir / "anchored_phase_residual_train_predictions.csv", index=False)
    extreme_frame.to_csv(args.output_dir / "anchored_phase_residual_extreme_holdout_rows.csv", index=False)
    write_plots(args.output_dir, summary, heldout_predictions)
    write_report(args.output_dir, summary)
    metadata = {
        "linear_reg": LINEAR_REG,
        "target_metric": olmix.UNCHEATABLE_TARGET,
        "extreme_frac": float(args.extreme_frac),
        "modes": modes,
        "ridges": ridges,
        "clip_scales": clip_scales,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("Top variants")
    print(summary.head(15).to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
