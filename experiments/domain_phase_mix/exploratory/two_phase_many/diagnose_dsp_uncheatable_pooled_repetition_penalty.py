# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose a pooled effective-epoch repetition penalty for Uncheatable BPB.

Prior phase-benefit variants improved in-support rank but overpredicted repaired
frontier candidates. This diagnostic tests the complementary mechanism: pooled
repetition harm in effective-epoch space. Per-domain penalty thresholds are
removed; a single global threshold and a single NNLS penalty coefficient transfer
overexposure evidence across domains.

The model is:

    z_i = e0_i + gamma * e1_i
    h_i = softplus(log1p(z_i) - tau)^2
    F = sum_i kappa_i h_i
    L = b0 - sum_i a_i (1 - exp(-rho_i z_i)) + P F

where kappa is frozen from the no-phase benefit coefficients.
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
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_pooled_repetition_penalty_20260703"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LINEAR_REG = 0.01
TAU_GRID = (np.log1p(4.0), np.log1p(8.0), np.log1p(15.0), np.log1p(30.0), np.log1p(60.0))


@dataclass(frozen=True)
class PooledPenaltyFit:
    intercept: float
    benefit_coef: np.ndarray
    pooled_penalty_coef: float
    rho: np.ndarray
    gamma: float
    tau: float
    kappa: np.ndarray
    domain_names: list[str]
    c0: np.ndarray
    c1: np.ndarray
    fixed_tau: bool


@dataclass(frozen=True)
class SummaryRow:
    model_name: str
    gamma: float | None
    tau: float | None
    pooled_penalty_coef: float | None
    fixed_tau: bool
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


def frozen_kappa(base_model: dsp.FittedDSPModel) -> np.ndarray:
    kappa = np.maximum(np.asarray(base_model.benefit_coef, dtype=float), 0.0)
    total = float(kappa.sum())
    if total <= 0.0:
        return np.full_like(kappa, 1.0 / len(kappa), dtype=float)
    return kappa / total


def effective_exposure(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray, gamma: float) -> np.ndarray:
    e0 = weights[:, 0, :] * c0[None, :]
    e1 = weights[:, 1, :] * c1[None, :]
    return e0 + float(gamma) * e1


def pooled_design(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    *,
    rho: np.ndarray,
    gamma: float,
    tau: float,
    kappa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = effective_exposure(weights, c0, c1, gamma)
    signal = 1.0 - np.exp(-rho[None, :] * z)
    pooled_penalty = (dsp.softplus(np.log1p(z) - float(tau)) ** 2) @ kappa
    design = np.hstack([-signal, pooled_penalty[:, None]])
    return design, signal, pooled_penalty


def fit_head(
    weights: np.ndarray,
    targets: np.ndarray,
    packet: dsp.PacketData,
    *,
    rho: np.ndarray,
    gamma: float,
    tau: float,
    kappa: np.ndarray,
    fixed_tau: bool,
) -> PooledPenaltyFit:
    design, _signal, _pooled_penalty = pooled_design(
        weights,
        packet.c0,
        packet.c1,
        rho=rho,
        gamma=gamma,
        tau=tau,
        kappa=kappa,
    )
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if LINEAR_REG > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(LINEAR_REG) * np.eye(centered_design.shape[1])])
        centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
    coef, _ = nnls(centered_design, centered_targets)
    intercept = float(target_mean - (design_mean @ coef).item())
    return PooledPenaltyFit(
        intercept=intercept,
        benefit_coef=np.asarray(coef[: packet.m], dtype=float),
        pooled_penalty_coef=float(coef[-1]),
        rho=np.asarray(rho, dtype=float),
        gamma=float(gamma),
        tau=float(tau),
        kappa=np.asarray(kappa, dtype=float),
        domain_names=list(packet.domain_names),
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
        fixed_tau=bool(fixed_tau),
    )


def predict_pooled(fit: PooledPenaltyFit, weights: np.ndarray) -> np.ndarray:
    design, _signal, _pooled_penalty = pooled_design(
        weights,
        fit.c0,
        fit.c1,
        rho=fit.rho,
        gamma=fit.gamma,
        tau=fit.tau,
        kappa=fit.kappa,
    )
    return np.asarray(fit.intercept + design @ np.concatenate([fit.benefit_coef, [fit.pooled_penalty_coef]]))


def pack_theta(rho: np.ndarray, gamma: float, tau: float | None) -> np.ndarray:
    parts = [np.log(np.clip(rho, 1e-4, 2.0)), np.asarray([np.log(max(gamma, 1e-4))], dtype=float)]
    if tau is not None:
        parts.append(np.asarray([tau], dtype=float))
    return np.concatenate(parts)


def unpack_theta(theta: np.ndarray, m: int, fixed_tau: float | None) -> tuple[np.ndarray, float, float]:
    rho = np.exp(theta[:m])
    gamma = float(np.exp(theta[m]))
    tau = float(fixed_tau) if fixed_tau is not None else float(theta[m + 1])
    return rho, gamma, tau


def theta_bounds(m: int, fixed_tau: float | None) -> list[tuple[float, float]]:
    bounds = [(np.log(1e-4), np.log(2.0))] * m
    bounds.append((np.log(0.05), np.log(64.0)))
    if fixed_tau is None:
        bounds.append((1.0, 4.5))
    return bounds


def start_bank(packet: dsp.PacketData, fixed_tau: float | None) -> list[np.ndarray]:
    raw_z = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    positive = np.where(raw_z > 1e-8, raw_z, np.nan)
    median_exposure = np.nanmedian(positive, axis=0)
    fallback = float(np.nanmedian(positive))
    median_exposure = np.where(np.isfinite(median_exposure), median_exposure, fallback)
    base_rho = np.clip(1.0 / np.maximum(median_exposure, 1e-3), 1e-4, 0.5)
    tau_start = float(np.log1p(15.0) if fixed_tau is None else fixed_tau)
    starts: list[np.ndarray] = []
    for rho_scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        for gamma in (0.5, 1.0, 4.0, 12.0, 24.0):
            tau_value = tau_start if fixed_tau is None else None
            starts.append(pack_theta(np.clip(base_rho * rho_scale, 1e-4, 2.0), gamma, tau_value))
    rng = np.random.default_rng(dsp.CV_SEED)
    for _ in range(8):
        rho = np.clip(base_rho * np.exp(rng.normal(scale=0.8, size=packet.m)), 1e-4, 2.0)
        gamma = float(np.exp(rng.normal(loc=np.log(4.0), scale=1.0)))
        tau_value = float(rng.uniform(1.2, 4.2)) if fixed_tau is None else None
        starts.append(pack_theta(rho, gamma, tau_value))
    return starts


def profile_objective(packet: dsp.PacketData, kappa: np.ndarray, theta: np.ndarray, fixed_tau: float | None) -> float:
    rho, gamma, tau = unpack_theta(np.asarray(theta, dtype=float), packet.m, fixed_tau)
    fit = fit_head(packet.w, packet.y, packet, rho=rho, gamma=gamma, tau=tau, kappa=kappa, fixed_tau=fixed_tau is not None)
    pred = predict_pooled(fit, packet.w)
    residual = pred - packet.y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return rmse + 0.5 * lower_tail_optimism(packet.y, pred)


def fit_pooled(
    packet: dsp.PacketData,
    kappa: np.ndarray,
    *,
    fixed_tau: float | None,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[PooledPenaltyFit, pd.DataFrame]:
    starts = start_bank(packet, fixed_tau)
    rows: list[dict[str, Any]] = []
    for start_id, start in enumerate(starts):
        rows.append(
            {
                "stage": "coarse",
                "start_id": start_id,
                "objective": profile_objective(packet, kappa, start, fixed_tau),
            }
        )
    ranked = sorted(rows, key=lambda row: float(row["objective"]))
    best_theta: np.ndarray | None = None
    best_objective = float("inf")
    for rank, row in enumerate(ranked[:coarse_top_k]):
        start = starts[int(row["start_id"])]
        result = minimize(
            lambda theta: profile_objective(packet, kappa, np.asarray(theta, dtype=float), fixed_tau),
            start,
            method="L-BFGS-B",
            bounds=theta_bounds(packet.m, fixed_tau),
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
        rows.append(
            {
                "stage": "refine",
                "chosen_rank": rank,
                "start_id": int(row["start_id"]),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)
    if best_theta is None:
        raise RuntimeError("No pooled repetition fit selected")
    rho, gamma, tau = unpack_theta(best_theta, packet.m, fixed_tau)
    return (
        fit_head(packet.w, packet.y, packet, rho=rho, gamma=gamma, tau=tau, kappa=kappa, fixed_tau=fixed_tau is not None),
        pd.DataFrame.from_records(rows),
    )


def oof_pooled(packet: dsp.PacketData, fit: PooledPenaltyFit) -> np.ndarray:
    kf = dsp.KFold(n_splits=dsp.N_SPLITS, shuffle=True, random_state=dsp.CV_SEED)
    oof = np.zeros_like(packet.y)
    for train_idx, test_idx in kf.split(packet.w):
        fold_fit = fit_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            rho=fit.rho,
            gamma=fit.gamma,
            tau=fit.tau,
            kappa=fit.kappa,
            fixed_tau=fit.fixed_tau,
        )
        oof[test_idx] = predict_pooled(fold_fit, packet.w[test_idx])
    return oof


def repaired_predictions(fit: PooledPenaltyFit, heldout: pd.DataFrame) -> pd.DataFrame:
    base_like = dsp.FittedDSPModel(
        variant=dsp.VARIANTS["no_phase"],
        params={"rho": fit.rho, "tau": np.full_like(fit.rho, fit.tau)},
        intercept=fit.intercept,
        benefit_coef=fit.benefit_coef,
        penalty_coef=np.zeros_like(fit.rho),
        domain_names=fit.domain_names,
        c0=fit.c0,
        c1=fit.c1,
    )
    rows: list[dict[str, Any]] = []
    for row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(base_like, Path(row.mixture_path))
        pred = float(predict_pooled(fit, weights)[0])
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
        "heldout_all_mae": float(repaired["absolute_error"].mean()),
        "heldout_max_optimism": float(uncheatable["optimism"].max()),
        "heldout_gap_error": predicted_gap - actual_gap,
        "heldout_best_order_correct": predicted_best == actual_best,
    }


def leave_good_out(packet: dsp.PacketData, fit: PooledPenaltyFit, good_frac: float) -> tuple[float, float]:
    holdout_count = max(5, int(np.ceil(good_frac * len(packet.y))))
    test_idx = np.argsort(packet.y)[:holdout_count]
    train_idx = np.setdiff1d(np.arange(len(packet.y)), test_idx)
    fold_fit = fit_head(
        packet.w[train_idx],
        packet.y[train_idx],
        packet,
        rho=fit.rho,
        gamma=fit.gamma,
        tau=fit.tau,
        kappa=fit.kappa,
        fixed_tau=fit.fixed_tau,
    )
    pred = predict_pooled(fold_fit, packet.w[test_idx])
    y = packet.y[test_idx]
    rmse, _mae, _pearson, _spearman = regression_metrics(y, pred)
    return rmse, float(np.mean(y - pred))


def raw_optimize(fit: PooledPenaltyFit, packet: dsp.PacketData) -> tuple[float, float, float]:
    n = len(fit.domain_names)

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
            lambda z: float(predict_pooled(fit, logits_to_weights(np.asarray(z, dtype=float))[None, :, :])[0]),
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
    return bool(
        row["oof_spearman"] >= 0.70
        and row["heldout_uncheatable_mae"] <= 0.030
        and row["heldout_max_optimism"] <= 0.035
        and abs(row["heldout_gap_error"]) <= 0.004
        and row["raw_predicted_optimum"] >= 0.935
        and row["leave_good_rmse"] <= 0.015
        and row["leave_good_signed_optimism"] <= 0.005
    )


def summarize_fit(
    model_name: str,
    packet: dsp.PacketData,
    fit: PooledPenaltyFit,
    heldout: pd.DataFrame,
    *,
    good_frac: float,
) -> tuple[SummaryRow, pd.DataFrame]:
    train_pred = predict_pooled(fit, packet.w)
    train_rmse, _mae, _pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof = oof_pooled(packet, fit)
    oof_rmse, _mae, _pearson, oof_spearman = regression_metrics(packet.y, oof)
    repaired = repaired_predictions(fit, heldout)
    stats = repaired_gate_stats(repaired)
    leave_rmse, leave_optimism = leave_good_out(packet, fit, good_frac)
    raw_value, raw_tv, raw_nearest = raw_optimize(fit, packet)
    row = {
        "model_name": model_name,
        "gamma": fit.gamma,
        "tau": fit.tau,
        "pooled_penalty_coef": fit.pooled_penalty_coef,
        "fixed_tau": fit.fixed_tau,
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
    repaired.insert(0, "model_name", model_name)
    return SummaryRow(**row), repaired


def transfer_visibility(packet: dsp.PacketData, fit: PooledPenaltyFit, heldout: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    base_like = dsp.FittedDSPModel(
        variant=dsp.VARIANTS["no_phase"],
        params={"rho": fit.rho, "tau": np.full_like(fit.rho, fit.tau)},
        intercept=fit.intercept,
        benefit_coef=fit.benefit_coef,
        penalty_coef=np.zeros_like(fit.rho),
        domain_names=fit.domain_names,
        c0=fit.c0,
        c1=fit.c1,
    )
    panel_z = effective_exposure(packet.w, fit.c0, fit.c1, fit.gamma)
    panel_log_margin = np.log1p(panel_z) - fit.tau
    rows: list[dict[str, Any]] = []
    for heldout_row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(base_like, Path(heldout_row.mixture_path))
        z = effective_exposure(weights, fit.c0, fit.c1, fit.gamma)[0]
        log_margin = np.log1p(z) - fit.tau
        penalty = fit.pooled_penalty_coef * fit.kappa * (dsp.softplus(log_margin) ** 2)
        order = np.argsort(penalty)[::-1][:top_k]
        for idx in order:
            rows.append(
                {
                    "mixture": str(heldout_row.mixture),
                    "domain": fit.domain_names[idx],
                    "effective_exposure_z": float(z[idx]),
                    "log1p_z_minus_tau": float(log_margin[idx]),
                    "penalty_contribution": float(penalty[idx]),
                    "panel_log_margin_p90": float(np.percentile(panel_log_margin[:, idx], 90)),
                    "panel_log_margin_p99": float(np.percentile(panel_log_margin[:, idx], 99)),
                    "panel_z_p90": float(np.percentile(panel_z[:, idx], 90)),
                    "panel_z_p99": float(np.percentile(panel_z[:, idx], 99)),
                    "kappa": float(fit.kappa[idx]),
                }
            )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, summary: pd.DataFrame, grid_trace: pd.DataFrame) -> None:
    lines = [
        "# Pooled effective-epoch repetition-penalty diagnostic",
        "",
        "This tests whether the repaired/frontier optimism is caused by per-domain overexposure penalties failing to transfer across domains. The variant removes per-domain penalty thresholds and replaces them with one pooled effective-epoch repetition penalty.",
        "",
        "## Gate summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Tau grid fits",
        "",
        grid_trace.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet, *_ = eta_diag.load_packet()
    heldout = eta_diag.load_heldout(args)
    base_model = shared_diag.fit_no_phase(packet, args)
    kappa = frozen_kappa(base_model)

    fits: list[tuple[str, PooledPenaltyFit, pd.DataFrame]] = []
    free_fit, free_trace = fit_pooled(
        packet,
        kappa,
        fixed_tau=None,
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
    )
    fits.append(("pooled_repetition_penalty_free_tau", free_fit, free_trace))
    for tau in TAU_GRID:
        fit, trace = fit_pooled(
            packet,
            kappa,
            fixed_tau=float(tau),
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
        )
        fits.append((f"pooled_repetition_penalty_tau_{tau:.3f}", fit, trace))

    summaries: list[SummaryRow] = []
    heldout_predictions: list[pd.DataFrame] = []
    trace_rows: list[pd.DataFrame] = []
    for name, fit, trace in fits:
        summary, repaired = summarize_fit(name, packet, fit, heldout, good_frac=float(args.good_frac))
        summaries.append(summary)
        heldout_predictions.append(repaired)
        trace = trace.copy()
        trace.insert(0, "model_name", name)
        trace_rows.append(trace)

    summary_frame = pd.DataFrame([asdict(row) for row in summaries])
    heldout_frame = pd.concat(heldout_predictions, ignore_index=True)
    trace_frame = pd.concat(trace_rows, ignore_index=True)
    visibility = transfer_visibility(packet, free_fit, heldout)

    summary_frame.to_csv(args.output_dir / "pooled_repetition_penalty_summary.csv", index=False)
    heldout_frame.to_csv(args.output_dir / "pooled_repetition_penalty_heldout_predictions.csv", index=False)
    trace_frame.to_csv(args.output_dir / "pooled_repetition_penalty_fit_trace.csv", index=False)
    visibility.to_csv(args.output_dir / "pooled_repetition_penalty_transfer_visibility.csv", index=False)
    metadata = {
        "linear_reg": LINEAR_REG,
        "tau_grid": [float(v) for v in TAU_GRID],
        "repair_results": str(args.repair_results),
        "repair_mixture_dir": str(args.repair_mixture_dir),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    fig = px.scatter(
        summary_frame,
        x="oof_spearman",
        y="heldout_uncheatable_mae",
        color="pass_gates",
        hover_data=[
            "model_name",
            "gamma",
            "tau",
            "pooled_penalty_coef",
            "raw_predicted_optimum",
            "leave_good_rmse",
            "heldout_max_optimism",
        ],
        title="Pooled repetition penalty gate frontier",
        color_discrete_map={True: "#138a36", False: "#c0362c"},
    )
    fig.add_vline(x=0.70, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.030, line_dash="dash", line_color="gray")
    fig.write_html(args.output_dir / "pooled_repetition_penalty_gate_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    write_report(args.output_dir, summary_frame, summary_frame[["model_name", "gamma", "tau", "pooled_penalty_coef", "train_rmse", "train_spearman"]])

    print(summary_frame.sort_values(["pass_gates", "heldout_uncheatable_mae"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    main()
