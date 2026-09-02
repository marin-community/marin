# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy"]
# ///
"""Fit OLMix log-linear baselines on the 300M deletion-augmented panel.

This is intentionally self-contained: the repo workspace currently has a uv
resolution conflict, and this script is a reproducible analysis artifact rather
than a launcher.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS

SCRIPT_DIR = Path(__file__).resolve().parent
HYDRATED_RAW_MATRIX = (
    SCRIPT_DIR
    / "reference_outputs"
    / "raw_metric_matrix_300m_training_eval_wandb_collect_20260623"
    / "pctrl_final_metric_matrix_with_training_eval.csv"
)
PCTRL_METRICS = (
    SCRIPT_DIR
    / "reference_outputs"
    / "pctrl_training_eval_wandb_collect_20260623"
    / "pctrl_final_metric_matrix_with_training_eval.csv"
)
PCTRL_MANIFEST = (
    SCRIPT_DIR / "reference_outputs" / "proportional_controllability_300m_20260520" / "training_manifest.csv"
)
OLMO_BASE_EASY_LONG = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_domain_ablation_effects_20260623"
    / "olmo_base_easy_300m_metrics_wide_source_long.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_reference_deletion_augmented_300m_20260625"

UNCHEATABLE_TARGET = "eval/uncheatable_eval/bpb"
ADAPTIVE_OLMIX_RUN_NAME = "baseline_olmix_loglinear_uncheatable_bpb"
OLMO_EASY_TOP_LEVEL_TASKS = (
    "olmobase:easy:qa:bpb",
    "olmobase:easy:code:bpb",
    "olmobase:easy:math:bpb",
)

HUBER_DELTA = 0.02
FIT_SEED = 0
FIT_N_STARTS = 48
CV_SEED = 0
N_SPLITS = 5
LOWER_TAIL_FRAC = 0.15
KL_REG = 0.05
# Canonical schedule in two_phase_dolma3_dolmino_top_level.py:
# PHASE_BOUNDARIES = [0.8], so phase fractions are 80% / 20%.
PHASE_FRACTIONS = np.asarray([0.8, 0.2], dtype=float)
REPETITION_FACTOR = 4.0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FitResult:
    target_name: str
    target_metric: str
    objective_direction: str
    n_rows: int
    n_signal_rows: int
    n_deletion_rows: int
    n_proportional_reference_rows: int
    proportional_reference_mean: float | None
    proportional_reference_std: float | None
    fit_log_c: float
    fit_huber_loss: float
    train_rmse: float
    train_mae: float
    train_pearson: float
    train_spearman: float
    oof_rmse: float
    oof_mae: float
    oof_pearson: float
    oof_spearman: float
    fold_mean_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float
    cvxpy_status: str
    kl_reg: float
    predicted_objective: float
    regularized_objective: float
    proportional_actual: float | None
    proportional_predicted: float
    best_observed_run_name: str
    best_observed_value: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_mean_phase_tv: float
    mean_phase_tv_to_proportional: float
    max_epoch_multiplier: float
    q95_epoch_multiplier: float
    repetition_factor: float | None
    target_budget_tokens: int | None
    max_simulated_epoch: float | None
    q95_simulated_epoch: float | None
    max_repetition_cap_violation: float | None
    max_weight: float
    min_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-n-starts", type=int, default=FIT_N_STARTS)
    parser.add_argument("--huber-delta", type=float, default=HUBER_DELTA)
    parser.add_argument("--kl-reg", type=float, default=KL_REG)
    parser.add_argument(
        "--include-stale-olmo-easy",
        action="store_true",
        help="Also rerun the pre-fix OLMoBaseEval Easy exploratory target.",
    )
    return parser.parse_args()


def phase_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("phase_0_")]
    columns.extend(column for column in frame.columns if column.startswith("phase_1_"))
    if len(columns) != 78:
        raise ValueError(f"Expected 78 phase columns, found {len(columns)}")
    return columns


def domain_names_from_phase_columns(columns: list[str]) -> list[str]:
    phase0 = [column.removeprefix("phase_0_") for column in columns if column.startswith("phase_0_")]
    phase1 = [column.removeprefix("phase_1_") for column in columns if column.startswith("phase_1_")]
    if phase0 != phase1:
        raise ValueError("Phase 0/1 domain names do not align")
    return phase0


def load_target_budget() -> int:
    manifest = pd.read_csv(PCTRL_MANIFEST, usecols=["target_budget"])
    values = sorted(pd.to_numeric(manifest["target_budget"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(values) != 1:
        raise ValueError(f"Expected one target_budget in {PCTRL_MANIFEST}, found {values}")
    return int(values[0])


def load_domain_token_counts(domains: list[str]) -> np.ndarray:
    missing = [domain for domain in domains if domain not in TOP_LEVEL_DOMAIN_TOKEN_COUNTS]
    if missing:
        raise ValueError(f"Missing token counts for domains: {missing}")
    return np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain] for domain in domains], dtype=float)


def repetition_weight_caps(
    token_counts: np.ndarray,
    *,
    target_budget: int,
    repetition_factor: float,
) -> np.ndarray:
    """Maximum aggregate mixture weight under a materialized epoch cap."""
    caps = float(repetition_factor) * token_counts / float(target_budget)
    return np.minimum(caps, 1.0)


def aggregate_phase_weights(weights: np.ndarray) -> np.ndarray:
    return np.einsum("p,pd->d", PHASE_FRACTIONS, weights)


def simulated_epochs(weights: np.ndarray, token_counts: np.ndarray, *, target_budget: int) -> np.ndarray:
    aggregate = aggregate_phase_weights(weights)
    return float(target_budget) * aggregate / token_counts


def huber_sum(residuals: np.ndarray, delta: float) -> float:
    abs_residuals = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (abs_residuals - 0.5 * delta)
    return float(np.where(abs_residuals <= delta, quadratic, linear).sum())


def predict(log_c: float, coefficients: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = weights.reshape(len(weights), -1)
    logits = np.clip(x @ coefficients, -50.0, 50.0)
    return np.exp(log_c) + np.exp(logits)


def fit_olmix_loglinear(
    weights: np.ndarray,
    target: np.ndarray,
    *,
    delta: float,
    seed: int,
    n_starts: int,
    verbose: bool = True,
) -> tuple[float, np.ndarray, float]:
    x = weights.reshape(len(weights), -1)
    y = np.asarray(target, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("OLMix log-linear target must be positive")
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        log_c = float(params[0])
        coefficients = params[1:]
        pred = predict(log_c, coefficients, weights)
        return huber_sum(pred - y, delta=delta)

    best_params: np.ndarray | None = None
    best_loss = math.inf
    log_c_candidates = np.linspace(np.log(max(float(np.min(y)) * 0.25, 1e-6)), np.log(max(float(np.median(y)), 1e-6)), 6)
    starts: list[np.ndarray] = []
    for log_c in log_c_candidates:
        starts.append(np.concatenate([[log_c], np.zeros(x.shape[1], dtype=float)]))
        for _ in range(max(n_starts // len(log_c_candidates) - 1, 0)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))

    for idx, start in enumerate(starts, start=1):
        result = minimize(objective, start, method="L-BFGS-B")
        if result.success or best_params is None:
            if float(result.fun) < best_loss:
                best_loss = float(result.fun)
                best_params = np.asarray(result.x, dtype=float)
        if verbose and (idx % 25 == 0 or idx == len(starts)):
            print(f"fit start {idx}/{len(starts)} best_huber={best_loss:.8f}", flush=True)

    if best_params is None:
        raise RuntimeError("OLMix log-linear fit failed")
    return float(best_params[0]), np.asarray(best_params[1:], dtype=float), best_loss


def kfold_indices(n_rows: int, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_rows < n_splits:
        raise ValueError(f"Need at least {n_splits} rows for cross validation, got {n_rows}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    fold_sizes = np.full(n_splits, n_rows // n_splits, dtype=int)
    fold_sizes[: n_rows % n_splits] += 1
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        test_idx = np.sort(indices[start:stop])
        train_idx = np.sort(np.setdiff1d(indices, test_idx, assume_unique=False))
        folds.append((train_idx, test_idx))
        start = stop
    return folds


def fit_oof_predictions(
    weights: np.ndarray,
    target: np.ndarray,
    *,
    delta: float,
    seed: int,
    n_starts: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    folds = kfold_indices(len(target), n_splits=N_SPLITS, seed=seed)
    oof = np.zeros_like(target, dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        log_c, coefficients, _loss = fit_olmix_loglinear(
            weights[train_idx],
            target[train_idx],
            delta=delta,
            seed=seed + fold_idx,
            n_starts=n_starts,
            verbose=False,
        )
        oof[test_idx] = predict(log_c, coefficients, weights[test_idx])
    return oof, folds


def weighted_multiclass_kl(weights: np.ndarray, natural: np.ndarray, phase_fractions: np.ndarray) -> float:
    eps = 1e-12
    q = np.clip(weights, eps, 1.0)
    p = np.clip(natural, eps, 1.0)
    return float(phase_fractions @ np.sum(q * (np.log(q) - np.log(p[None, :])), axis=1))


def solve_exact_kl(
    log_c: float,
    coefficients: np.ndarray,
    *,
    natural: np.ndarray,
    phase_fractions: np.ndarray,
    kl_reg: float,
    repetition_caps: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, str]:
    n_phases = len(phase_fractions)
    n_domains = len(natural)
    phase_weights = cp.Variable((n_phases, n_domains))
    coeff = coefficients.reshape(n_phases, n_domains)
    linear = cp.sum(cp.multiply(coeff, phase_weights))
    predicted = float(np.exp(log_c)) + cp.exp(linear)
    kl = cp.sum(
        [
            float(phase_fractions[phase_idx]) * cp.sum(cp.rel_entr(phase_weights[phase_idx], natural))
            for phase_idx in range(n_phases)
        ]
    )
    constraints: list[Any] = [phase_weights >= 0, cp.sum(phase_weights, axis=1) == 1]
    if repetition_caps is not None:
        if repetition_caps.shape != (n_domains,):
            raise ValueError(f"Expected repetition caps shape {(n_domains,)}, got {repetition_caps.shape}")
        aggregate = cp.sum(cp.multiply(phase_fractions[:, None], phase_weights), axis=0)
        constraints.append(aggregate <= repetition_caps)
    problem = cp.Problem(cp.Minimize(predicted + float(kl_reg) * kl), constraints)

    errors: list[str] = []
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break
        errors.append(f"{solver}: status={problem.status}")
    else:
        raise RuntimeError(f"CVXPY solve failed: {errors}")

    if phase_weights.value is None:
        raise RuntimeError("CVXPY solve returned no phase weights")
    weights = np.asarray(phase_weights.value, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum(axis=1, keepdims=True)
    predicted_value = float(predict(log_c, coefficients, weights[None, :, :])[0])
    regularized = predicted_value + kl_reg * weighted_multiclass_kl(weights, natural, phase_fractions)
    return weights, predicted_value, regularized, str(problem.status)


def solve_single_exact_kl(
    log_c: float,
    coefficients: np.ndarray,
    *,
    natural: np.ndarray,
    kl_reg: float,
    repetition_caps: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, str]:
    n_domains = len(natural)
    if coefficients.shape != (n_domains,):
        raise ValueError(f"Expected {n_domains} single-simplex coefficients, got {coefficients.shape}")
    weights = cp.Variable(n_domains)
    predicted = float(np.exp(log_c)) + cp.exp(cp.sum(cp.multiply(coefficients, weights)))
    kl = cp.sum(cp.rel_entr(weights, natural))
    constraints: list[Any] = [weights >= 0, cp.sum(weights) == 1]
    if repetition_caps is not None:
        if repetition_caps.shape != (n_domains,):
            raise ValueError(f"Expected repetition caps shape {(n_domains,)}, got {repetition_caps.shape}")
        constraints.append(weights <= repetition_caps)
    problem = cp.Problem(cp.Minimize(predicted + float(kl_reg) * kl), constraints)

    errors: list[str] = []
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break
        errors.append(f"{solver}: status={problem.status}")
    else:
        raise RuntimeError(f"Single-simplex CVXPY solve failed: {errors}")

    if weights.value is None:
        raise RuntimeError("Single-simplex CVXPY solve returned no weights")
    solved = np.asarray(weights.value, dtype=float)
    solved = np.clip(solved, 0.0, None)
    solved = solved / solved.sum()
    predicted_value = float(predict(log_c, coefficients, solved[None, :])[0])
    regularized = predicted_value + kl_reg * float(
        np.sum(solved * (np.log(np.clip(solved, 1e-12, 1.0)) - np.log(natural)))
    )
    return solved, predicted_value, regularized, str(problem.status)


def mean_phase_tv(weights: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(weights - reference[None, :, :]).sum(axis=2).mean(axis=1)


def load_raw_signal_panel() -> tuple[pd.DataFrame, list[str], list[str], np.ndarray]:
    frame = pd.read_csv(HYDRATED_RAW_MATRIX, low_memory=False)
    columns = phase_columns(frame)
    domains = domain_names_from_phase_columns(columns)
    signal = frame[frame["row_kind"].eq("signal") & ~frame["run_name"].eq(ADAPTIVE_OLMIX_RUN_NAME)].copy()
    natural_row = frame[frame["run_name"].eq("baseline_proportional")].iloc[0]
    natural = natural_row[[f"phase_0_{domain}" for domain in domains]].astype(float).to_numpy()
    natural = natural / natural.sum()
    return signal, columns, domains, natural


def load_deletion_weights(columns: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(PCTRL_MANIFEST, low_memory=False)
    deletion = manifest[manifest["intervention_type"].eq("domain_deletion")].copy()
    if "source_experiment" not in deletion.columns:
        if "source_two_phase_experiment" not in deletion.columns:
            raise ValueError("Deletion manifest has no source experiment column")
        deletion["source_experiment"] = deletion["source_two_phase_experiment"]
    keep = [
        "run_name",
        "source_experiment",
        "intervention_type",
        "target_domain",
        "base_mass",
        "tv_distance",
        *columns,
    ]
    return deletion[keep]


def replace_proportional_target_with_reference_mean(
    panel: pd.DataFrame,
    *,
    target_column: str,
    reference: pd.Series,
) -> tuple[pd.DataFrame, int, float | None, float | None]:
    values = pd.to_numeric(reference, errors="coerce").dropna()
    if values.empty:
        return panel, 0, None, None
    out = panel.copy()
    out.loc[out["run_name"].eq("baseline_proportional"), target_column] = float(values.mean())
    return out, int(len(values)), float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def build_uncheatable_panel(columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(HYDRATED_RAW_MATRIX, low_memory=False)
    signal = raw[raw["row_kind"].eq("signal") & ~raw["run_name"].eq(ADAPTIVE_OLMIX_RUN_NAME)].copy()
    signal["panel_source"] = "qsplit_signal"
    deletion_weights = load_deletion_weights(columns)
    pctrl = pd.read_csv(PCTRL_METRICS, low_memory=False)
    deletion_metrics = pctrl[pctrl["intervention_type"].eq("domain_deletion")][["run_name", UNCHEATABLE_TARGET]].copy()
    deletion = deletion_weights.merge(deletion_metrics, on="run_name", how="inner", validate="one_to_one")
    deletion["panel_source"] = "domain_deletion"
    signal_keep = ["run_name", "source_experiment", "panel_source", *columns, UNCHEATABLE_TARGET]
    deletion_keep = ["run_name", "source_experiment", "panel_source", *columns, UNCHEATABLE_TARGET]
    panel = pd.concat([signal[signal_keep], deletion[deletion_keep]], ignore_index=True)
    prop_ref = raw[
        raw["run_name"].eq("baseline_proportional") | raw["noise_anchor_run_name"].fillna("").eq("baseline_proportional")
    ][UNCHEATABLE_TARGET]
    panel, ref_n, ref_mean, ref_std = replace_proportional_target_with_reference_mean(
        panel,
        target_column=UNCHEATABLE_TARGET,
        reference=prop_ref,
    )
    panel = panel[pd.to_numeric(panel[UNCHEATABLE_TARGET], errors="coerce").notna()].reset_index(drop=True)
    expected_rows = 280
    if len(panel) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} Uncheatable fit rows after excluding {ADAPTIVE_OLMIX_RUN_NAME}, "
            f"found {len(panel)}"
        )
    if int(panel["panel_source"].eq("qsplit_signal").sum()) != 241:
        raise ValueError("Expected 241 ex-ante qsplit/signal rows")
    if int(panel["panel_source"].eq("domain_deletion").sum()) != 39:
        raise ValueError("Expected 39 domain-deletion rows")
    if int(panel[UNCHEATABLE_TARGET].notna().sum()) != expected_rows:
        raise ValueError(f"Missing {UNCHEATABLE_TARGET} in the intended fit panel")
    metadata = {
        "target_metric": UNCHEATABLE_TARGET,
        "excluded_adaptive_run_name": ADAPTIVE_OLMIX_RUN_NAME,
        "source_metric_matrix": str(HYDRATED_RAW_MATRIX),
        "n_proportional_reference_rows": ref_n,
        "proportional_reference_mean": ref_mean,
        "proportional_reference_std": ref_std,
    }
    return panel, metadata


def build_olmo_easy_macro_panel(columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, _, _, _ = load_raw_signal_panel()
    signal = signal.copy()
    signal["panel_source"] = "qsplit_signal"
    deletion_weights = load_deletion_weights(columns)
    deletion_weights["panel_source"] = "domain_deletion"

    metrics = pd.read_csv(OLMO_BASE_EASY_LONG, low_memory=False)
    metrics = metrics[metrics["scale"].eq("300m_6b") & metrics["olmo_task"].isin(OLMO_EASY_TOP_LEVEL_TASKS)].copy()
    macro = (
        metrics.groupby("run_name", as_index=False)["value_bpb"]
        .mean()
        .rename(columns={"value_bpb": "olmo_base_easy_top3_macro_bpb"})
    )

    signal_keep = ["run_name", "source_experiment", "panel_source", *columns]
    deletion_keep = ["run_name", "source_experiment", "panel_source", *columns]
    panel = pd.concat([signal[signal_keep], deletion_weights[deletion_keep]], ignore_index=True)
    panel = panel.merge(macro, on="run_name", how="inner", validate="one_to_one")
    panel = panel[pd.to_numeric(panel["olmo_base_easy_top3_macro_bpb"], errors="coerce").notna()].reset_index(drop=True)
    metadata = {
        "target_metric": "olmo_base_easy_top3_macro_bpb",
        "n_proportional_reference_rows": 1,
        "proportional_reference_mean": (
            float(panel.loc[panel["run_name"].eq("baseline_proportional"), "olmo_base_easy_top3_macro_bpb"].iloc[0])
            if panel["run_name"].eq("baseline_proportional").any()
            else None
        ),
        "proportional_reference_std": None,
        "olmo_top_level_tasks": list(OLMO_EASY_TOP_LEVEL_TASKS),
    }
    return panel, metadata


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = y_hat - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = (
        float(pearsonr(y, y_hat).statistic) if len(y) >= 3 and np.std(y) > 0 and np.std(y_hat) > 0 else float("nan")
    )
    spearman = (
        float(spearmanr(y, y_hat).statistic) if len(y) >= 3 and np.std(y) > 0 and np.std(y_hat) > 0 else float("nan")
    )
    return rmse, mae, pearson, spearman


def predictive_diagnostics(
    y: np.ndarray,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    rmse, mae, pearson, spearman = regression_metrics(y, pred)
    fold_regrets: list[float] = []
    for _train_idx, test_idx in folds:
        local_pred = pred[test_idx]
        local_y = y[test_idx]
        chosen = int(np.argmin(local_pred))
        fold_regrets.append(float(local_y[chosen] - np.min(local_y)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(pred)[:tail_count]
    tail_residual = pred[tail_idx] - y[tail_idx]
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "fold_mean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(tail_residual * tail_residual))),
    }


def fit_target(
    *,
    target_name: str,
    target_metric: str,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    columns: list[str],
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    repetition_caps: np.ndarray | None,
    repetition_factor: float | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> FitResult:
    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = pd.to_numeric(panel[target_metric], errors="coerce").to_numpy(dtype=float)
    weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    log_c, coefficients, huber_loss = fit_olmix_loglinear(
        weights,
        target,
        delta=float(args.huber_delta),
        seed=FIT_SEED,
        n_starts=int(args.fit_n_starts),
    )
    predictions = predict(log_c, coefficients, weights)
    rmse, mae, pearson, spearman = regression_metrics(target, predictions)
    oof_predictions, folds = fit_oof_predictions(
        weights,
        target,
        delta=float(args.huber_delta),
        seed=CV_SEED,
        n_starts=int(args.fit_n_starts),
    )
    oof_metrics = predictive_diagnostics(target, oof_predictions, folds)

    optimum_weights, predicted_objective, regularized_objective, cvxpy_status = solve_exact_kl(
        log_c,
        coefficients,
        natural=natural,
        phase_fractions=PHASE_FRACTIONS,
        kl_reg=float(args.kl_reg),
        repetition_caps=repetition_caps,
    )
    reference = np.stack([natural, natural])
    distances = mean_phase_tv(weights, optimum_weights)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(target))
    prop_predicted = float(predict(log_c, coefficients, reference[None, :, :])[0])
    prop_actual_rows = panel.loc[panel["run_name"].eq("baseline_proportional"), target_metric]
    prop_actual = float(prop_actual_rows.iloc[0]) if len(prop_actual_rows) else None

    ratios = optimum_weights / np.clip(reference, 1e-12, None)
    sim_epochs = simulated_epochs(optimum_weights, token_counts, target_budget=target_budget)
    cap_violation = None if repetition_factor is None else float(np.max(sim_epochs - float(repetition_factor)))
    result = FitResult(
        target_name=target_name,
        target_metric=target_metric,
        objective_direction="lower_is_better",
        n_rows=int(len(panel)),
        n_signal_rows=int(panel["panel_source"].eq("qsplit_signal").sum()),
        n_deletion_rows=int(panel["panel_source"].eq("domain_deletion").sum()),
        n_proportional_reference_rows=int(metadata.get("n_proportional_reference_rows", 0)),
        proportional_reference_mean=metadata.get("proportional_reference_mean"),
        proportional_reference_std=metadata.get("proportional_reference_std"),
        fit_log_c=float(log_c),
        fit_huber_loss=float(huber_loss),
        train_rmse=rmse,
        train_mae=mae,
        train_pearson=pearson,
        train_spearman=spearman,
        oof_rmse=float(oof_metrics["rmse"]),
        oof_mae=float(oof_metrics["mae"]),
        oof_pearson=float(oof_metrics["pearson"]),
        oof_spearman=float(oof_metrics["spearman"]),
        fold_mean_regret_at_1=float(oof_metrics["fold_mean_regret_at_1"]),
        lower_tail_optimism=float(oof_metrics["lower_tail_optimism"]),
        low_tail_rmse=float(oof_metrics["low_tail_rmse"]),
        cvxpy_status=cvxpy_status,
        kl_reg=float(args.kl_reg),
        predicted_objective=float(predicted_objective),
        regularized_objective=float(regularized_objective),
        proportional_actual=prop_actual,
        proportional_predicted=prop_predicted,
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_value=float(target[best_idx]),
        nearest_observed_run_name=str(panel.iloc[nearest_idx]["run_name"]),
        nearest_observed_value=float(target[nearest_idx]),
        nearest_observed_mean_phase_tv=float(distances[nearest_idx]),
        mean_phase_tv_to_proportional=float(0.5 * np.abs(optimum_weights - reference).sum(axis=1).mean()),
        max_epoch_multiplier=float(np.max(ratios)),
        q95_epoch_multiplier=float(np.quantile(ratios, 0.95)),
        repetition_factor=repetition_factor,
        target_budget_tokens=int(target_budget),
        max_simulated_epoch=float(np.max(sim_epochs)),
        q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
        max_repetition_cap_violation=cap_violation,
        max_weight=float(np.max(optimum_weights)),
        min_weight=float(np.min(optimum_weights)),
    )

    panel_out = panel[["run_name", "source_experiment", "panel_source", target_metric]].copy()
    panel_out["olmix_prediction"] = predictions
    panel_out["olmix_oof_prediction"] = oof_predictions
    panel_out["residual"] = predictions - target
    panel_out["oof_residual"] = oof_predictions - target
    panel_out.to_csv(target_dir / "fit_panel_predictions.csv", index=False)

    weights_out = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": optimum_weights[0],
            "phase_1_weight": optimum_weights[1],
            "aggregate_weight": aggregate_phase_weights(optimum_weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "repetition_cap_weight": repetition_caps if repetition_caps is not None else np.nan,
            "simulated_epoch_cap": repetition_factor if repetition_factor is not None else np.nan,
            "simulated_epoch_cap_slack": (
                (float(repetition_factor) - sim_epochs) if repetition_factor is not None else np.nan
            ),
            "phase_0_epoch_multiplier": ratios[0],
            "phase_1_epoch_multiplier": ratios[1],
            "phase_0_delta": optimum_weights[0] - natural,
            "phase_1_delta": optimum_weights[1] - natural,
        }
    )
    weights_out["max_abs_delta"] = weights_out[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    weights_out.to_csv(target_dir / "proposed_mixture_weights.csv", index=False)

    with (target_dir / "fit_summary.json").open("w") as f:
        json.dump(
            {
                "result": asdict(result),
                "metadata": {
                    **metadata,
                    "phase_fractions": PHASE_FRACTIONS.tolist(),
                    "target_budget_tokens": int(target_budget),
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=target,
            y=predictions,
            mode="markers",
            marker={
                "size": 8,
                "color": panel["panel_source"].map({"qsplit_signal": 0, "domain_deletion": 1}),
                "colorscale": "RdYlGn_r",
            },
            text=panel["run_name"],
            name="fit rows",
        )
    )
    lo = float(min(np.min(target), np.min(predictions)))
    hi = float(max(np.max(target), np.max(predictions)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#666"}, name="y=x"))
    fig.update_layout(
        title=f"OLMix log-linear fit: {target_name}",
        xaxis_title="Observed BPB",
        yaxis_title="Predicted BPB",
        template="plotly_white",
        width=900,
        height=720,
    )
    fig.write_html(target_dir / "fit_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    top = weights_out.sort_values("max_abs_delta", ascending=False).head(24).iloc[::-1]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=top["domain"], x=top["phase_0_epoch_multiplier"], orientation="h", name="phase 0"))
    fig2.add_trace(go.Bar(y=top["domain"], x=top["phase_1_epoch_multiplier"], orientation="h", name="phase 1"))
    fig2.add_vline(x=1.0, line_dash="dash", line_color="#444")
    fig2.update_layout(
        title=f"OLMix proposed mixture vs proportional: {target_name}",
        xaxis_title="Epoch multiplier relative to proportional",
        yaxis_title="Domain",
        template="plotly_white",
        width=1000,
        height=780,
        barmode="group",
    )
    fig2.write_html(target_dir / "proposed_mixture_epoch_multipliers.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    return result


def fit_target_single_simplex_tied_phases(
    *,
    target_name: str,
    target_metric: str,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    columns: list[str],
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    repetition_caps: np.ndarray | None,
    repetition_factor: float | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> FitResult:
    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = pd.to_numeric(panel[target_metric], errors="coerce").to_numpy(dtype=float)
    phase_weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    exposure_average = np.einsum("p,npd->nd", PHASE_FRACTIONS, phase_weights)
    log_c, coefficients, huber_loss = fit_olmix_loglinear(
        exposure_average,
        target,
        delta=float(args.huber_delta),
        seed=FIT_SEED,
        n_starts=int(args.fit_n_starts),
    )
    predictions = predict(log_c, coefficients, exposure_average)
    rmse, mae, pearson, spearman = regression_metrics(target, predictions)
    oof_predictions, folds = fit_oof_predictions(
        exposure_average,
        target,
        delta=float(args.huber_delta),
        seed=CV_SEED,
        n_starts=int(args.fit_n_starts),
    )
    oof_metrics = predictive_diagnostics(target, oof_predictions, folds)

    single_weights, predicted_objective, regularized_objective, cvxpy_status = solve_single_exact_kl(
        log_c,
        coefficients,
        natural=natural,
        kl_reg=float(args.kl_reg),
        repetition_caps=repetition_caps,
    )
    optimum_weights = np.stack([single_weights, single_weights])
    reference = np.stack([natural, natural])
    distances = mean_phase_tv(phase_weights, optimum_weights)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(target))
    prop_predicted = float(predict(log_c, coefficients, natural[None, :])[0])
    prop_actual_rows = panel.loc[panel["run_name"].eq("baseline_proportional"), target_metric]
    prop_actual = float(prop_actual_rows.iloc[0]) if len(prop_actual_rows) else None
    ratios = optimum_weights / np.clip(reference, 1e-12, None)
    sim_epochs = simulated_epochs(optimum_weights, token_counts, target_budget=target_budget)
    cap_violation = None if repetition_factor is None else float(np.max(sim_epochs - float(repetition_factor)))
    result = FitResult(
        target_name=target_name,
        target_metric=target_metric,
        objective_direction="lower_is_better",
        n_rows=int(len(panel)),
        n_signal_rows=int(panel["panel_source"].eq("qsplit_signal").sum()),
        n_deletion_rows=int(panel["panel_source"].eq("domain_deletion").sum()),
        n_proportional_reference_rows=int(metadata.get("n_proportional_reference_rows", 0)),
        proportional_reference_mean=metadata.get("proportional_reference_mean"),
        proportional_reference_std=metadata.get("proportional_reference_std"),
        fit_log_c=float(log_c),
        fit_huber_loss=float(huber_loss),
        train_rmse=rmse,
        train_mae=mae,
        train_pearson=pearson,
        train_spearman=spearman,
        oof_rmse=float(oof_metrics["rmse"]),
        oof_mae=float(oof_metrics["mae"]),
        oof_pearson=float(oof_metrics["pearson"]),
        oof_spearman=float(oof_metrics["spearman"]),
        fold_mean_regret_at_1=float(oof_metrics["fold_mean_regret_at_1"]),
        lower_tail_optimism=float(oof_metrics["lower_tail_optimism"]),
        low_tail_rmse=float(oof_metrics["low_tail_rmse"]),
        cvxpy_status=cvxpy_status,
        kl_reg=float(args.kl_reg),
        predicted_objective=float(predicted_objective),
        regularized_objective=float(regularized_objective),
        proportional_actual=prop_actual,
        proportional_predicted=prop_predicted,
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_value=float(target[best_idx]),
        nearest_observed_run_name=str(panel.iloc[nearest_idx]["run_name"]),
        nearest_observed_value=float(target[nearest_idx]),
        nearest_observed_mean_phase_tv=float(distances[nearest_idx]),
        mean_phase_tv_to_proportional=float(0.5 * np.abs(optimum_weights - reference).sum(axis=1).mean()),
        max_epoch_multiplier=float(np.max(ratios)),
        q95_epoch_multiplier=float(np.quantile(ratios, 0.95)),
        repetition_factor=repetition_factor,
        target_budget_tokens=int(target_budget),
        max_simulated_epoch=float(np.max(sim_epochs)),
        q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
        max_repetition_cap_violation=cap_violation,
        max_weight=float(np.max(optimum_weights)),
        min_weight=float(np.min(optimum_weights)),
    )

    panel_out = panel[["run_name", "source_experiment", "panel_source", target_metric]].copy()
    panel_out["olmix_prediction"] = predictions
    panel_out["olmix_oof_prediction"] = oof_predictions
    panel_out["residual"] = predictions - target
    panel_out["oof_residual"] = oof_predictions - target
    panel_out.to_csv(target_dir / "fit_panel_predictions.csv", index=False)

    weights_out = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": optimum_weights[0],
            "phase_1_weight": optimum_weights[1],
            "aggregate_weight": aggregate_phase_weights(optimum_weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "repetition_cap_weight": repetition_caps if repetition_caps is not None else np.nan,
            "simulated_epoch_cap": repetition_factor if repetition_factor is not None else np.nan,
            "simulated_epoch_cap_slack": (
                (float(repetition_factor) - sim_epochs) if repetition_factor is not None else np.nan
            ),
            "phase_0_epoch_multiplier": ratios[0],
            "phase_1_epoch_multiplier": ratios[1],
            "phase_0_delta": optimum_weights[0] - natural,
            "phase_1_delta": optimum_weights[1] - natural,
        }
    )
    weights_out["max_abs_delta"] = weights_out[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    weights_out.to_csv(target_dir / "proposed_mixture_weights.csv", index=False)

    with (target_dir / "fit_summary.json").open("w") as f:
        json.dump(
            {
                "result": asdict(result),
                "metadata": {
                    **metadata,
                    "variant": "single_simplex_tied_phases",
                    "fit_input": "phase_fractions_weighted_exposure_average",
                    "phase_fractions": PHASE_FRACTIONS.tolist(),
                    "target_budget_tokens": int(target_budget),
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=target,
            y=predictions,
            mode="markers",
            marker={
                "size": 8,
                "color": panel["panel_source"].map({"qsplit_signal": 0, "domain_deletion": 1}),
                "colorscale": "RdYlGn_r",
            },
            text=panel["run_name"],
            name="fit rows",
        )
    )
    lo = float(min(np.min(target), np.min(predictions)))
    hi = float(max(np.max(target), np.max(predictions)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#666"}, name="y=x"))
    fig.update_layout(
        title=f"OLMix single-simplex tied-phase fit: {target_name}",
        xaxis_title="Observed BPB",
        yaxis_title="Predicted BPB",
        template="plotly_white",
        width=900,
        height=720,
    )
    fig.write_html(target_dir / "fit_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    return result


def write_uncheatable_variant_comparison(
    output_dir: Path,
    *,
    two_variant: str,
    tied_variant: str,
    output_prefix: str,
    title_suffix: str,
) -> None:
    two_path = output_dir / two_variant / "proposed_mixture_weights.csv"
    tied_path = output_dir / tied_variant / "proposed_mixture_weights.csv"
    if not two_path.exists() or not tied_path.exists():
        return
    two = pd.read_csv(two_path)
    tied = pd.read_csv(tied_path)
    merged = two[
        [
            "domain",
            "proportional",
            "phase_0_epoch_multiplier",
            "phase_1_epoch_multiplier",
            "phase_0_weight",
            "phase_1_weight",
            "aggregate_weight",
            "simulated_epochs",
        ]
    ].rename(
        columns={
            "phase_0_epoch_multiplier": "two_phase_p0_multiplier",
            "phase_1_epoch_multiplier": "two_phase_p1_multiplier",
            "phase_0_weight": "two_phase_p0_weight",
            "phase_1_weight": "two_phase_p1_weight",
            "aggregate_weight": "two_phase_aggregate_weight",
            "simulated_epochs": "two_phase_simulated_epochs",
        }
    )
    tied_small = tied[
        ["domain", "phase_0_epoch_multiplier", "phase_0_weight", "aggregate_weight", "simulated_epochs"]
    ].rename(
        columns={
            "phase_0_epoch_multiplier": "single_tied_multiplier",
            "phase_0_weight": "single_tied_weight",
            "aggregate_weight": "single_tied_aggregate_weight",
            "simulated_epochs": "single_tied_simulated_epochs",
        }
    )
    merged = merged.merge(tied_small, on="domain", how="inner", validate="one_to_one")
    merged["max_abs_log2_multiplier"] = np.max(
        np.abs(
            np.log2(
                np.clip(
                    merged[["two_phase_p0_multiplier", "two_phase_p1_multiplier", "single_tied_multiplier"]].to_numpy(),
                    1e-9,
                    None,
                )
            )
        ),
        axis=1,
    )
    merged = merged.sort_values("max_abs_log2_multiplier", ascending=False)
    merged.to_csv(output_dir / f"{output_prefix}_single_vs_two_phase_weights.csv", index=False)

    top = merged.head(30).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top["domain"], x=top["two_phase_p0_multiplier"], orientation="h", name="two-phase p0"))
    fig.add_trace(go.Bar(y=top["domain"], x=top["two_phase_p1_multiplier"], orientation="h", name="two-phase p1"))
    fig.add_trace(go.Bar(y=top["domain"], x=top["single_tied_multiplier"], orientation="h", name="single tied"))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#444")
    fig.update_layout(
        title=f"Uncheatable OLMix{title_suffix}: single tied phases vs two-phase proposal",
        xaxis_title="Epoch multiplier relative to proportional",
        yaxis_title="Domain",
        template="plotly_white",
        width=1150,
        height=900,
        barmode="group",
    )
    fig.write_html(
        output_dir / f"{output_prefix}_single_vs_two_phase_epoch_multipliers.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    heat = merged.sort_values("domain")
    z = np.log2(
        np.clip(
            heat[["two_phase_p0_multiplier", "two_phase_p1_multiplier", "single_tied_multiplier"]].to_numpy(dtype=float),
            1e-9,
            None,
        )
    )
    text = np.vectorize(lambda value: f"{value:.1f}x")(
        heat[["two_phase_p0_multiplier", "two_phase_p1_multiplier", "single_tied_multiplier"]].to_numpy(dtype=float)
    )
    fig2 = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["two-phase p0", "two-phase p1", "single tied"],
            y=heat["domain"],
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "log2 epoch multiplier"},
            text=text,
            texttemplate="%{text}",
            hovertemplate="domain=%{y}<br>variant=%{x}<br>log2 multiplier=%{z:.2f}<extra></extra>",
        )
    )
    fig2.update_layout(
        title=f"Uncheatable OLMix{title_suffix}: relative exposure heatmap",
        xaxis_title="Variant",
        yaxis_title="Domain",
        template="plotly_white",
        width=900,
        height=1250,
    )
    fig2.write_html(
        output_dir / f"{output_prefix}_single_vs_two_phase_heatmap.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    epoch_top = (
        merged.sort_values(
            ["two_phase_simulated_epochs", "single_tied_simulated_epochs"],
            ascending=False,
        )
        .head(30)
        .iloc[::-1]
    )
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            y=epoch_top["domain"],
            x=epoch_top["two_phase_simulated_epochs"],
            orientation="h",
            name="two-phase aggregate",
        )
    )
    fig3.add_trace(
        go.Bar(
            y=epoch_top["domain"],
            x=epoch_top["single_tied_simulated_epochs"],
            orientation="h",
            name="single tied",
        )
    )
    fig3.add_vline(x=REPETITION_FACTOR, line_dash="dash", line_color="#444", annotation_text="cap=4")
    fig3.update_layout(
        title=f"Uncheatable OLMix{title_suffix}: simulated materialized epochs",
        xaxis_title="Simulated epochs",
        yaxis_title="Domain",
        template="plotly_white",
        width=1150,
        height=900,
        barmode="group",
    )
    fig3.write_html(
        output_dir / f"{output_prefix}_single_vs_two_phase_simulated_epochs.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_report(output_dir: Path, results: list[FitResult]) -> None:
    lines = [
        "# OLMix reference deletion-augmented 300M fits",
        "",
        "Functional form: `BPB_hat(w) = exp(c) + exp(beta^T [w0, w1])`.",
        "",
        "Solver: Huber regression fit plus KL-regularized exact convex proposer.",
        "",
        "| Target | rows | OOF Spearman | OOF RMSE | fold mean regret@1 | lower-tail optimism | low-tail RMSE | predicted BPB | proportional BPB | TV to proportional | max sim epoch | cap violation | nearest observed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        proportional = "NA" if row.proportional_actual is None else f"{row.proportional_actual:.6f}"
        max_epoch = "NA" if row.max_simulated_epoch is None else f"{row.max_simulated_epoch:.2f}"
        cap_violation = "NA" if row.max_repetition_cap_violation is None else f"{row.max_repetition_cap_violation:.3g}"
        lines.append(
            "| "
            f"`{row.target_name}` | {row.n_rows} | {row.oof_spearman:.4f} | {row.oof_rmse:.6f} | "
            f"{row.fold_mean_regret_at_1:.6f} | {row.lower_tail_optimism:.6f} | {row.low_tail_rmse:.6f} | "
            f"{row.predicted_objective:.6f} | {proportional} | {row.mean_phase_tv_to_proportional:.4f} | "
            f"{max_epoch} | {cap_violation} | `{row.nearest_observed_run_name}` |"
        )
    lines.extend(
        [
            "",
            "Hyperparameters:",
            f"- Huber delta: `{HUBER_DELTA}` by default; run value recorded in each summary JSON.",
            f"- KL regularization lambda: `{KL_REG}` by default; run value recorded in each summary JSON.",
            f"- Repetition cap for `rep_cap4` variants: aggregate simulated epoch `<= {REPETITION_FACTOR}`.",
            f"- Fit seed: `{FIT_SEED}`.",
            f"- Fit starts: `{FIT_N_STARTS}` by default; run value controlled by `--fit-n-starts`.",
            f"- Phase fractions: `{tuple(float(x) for x in PHASE_FRACTIONS)}`.",
            "- Prior for KL: 300M baseline proportional phase weights.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal, columns, domains, natural = load_raw_signal_panel()
    _ = signal
    target_budget = load_target_budget()
    token_counts = load_domain_token_counts(domains)
    repetition_caps = repetition_weight_caps(
        token_counts,
        target_budget=target_budget,
        repetition_factor=REPETITION_FACTOR,
    )
    if np.any(natural - repetition_caps > 1e-12):
        raise ValueError("Proportional baseline violates the requested repetition cap")

    results: list[FitResult] = []
    uncheatable_panel, uncheatable_metadata = build_uncheatable_panel(columns)
    results.append(
        fit_target(
            target_name="uncheatable_eval_bpb",
            target_metric=UNCHEATABLE_TARGET,
            panel=uncheatable_panel,
            metadata=uncheatable_metadata,
            columns=columns,
            domains=domains,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            repetition_caps=None,
            repetition_factor=None,
            output_dir=args.output_dir,
            args=args,
        )
    )
    results.append(
        fit_target_single_simplex_tied_phases(
            target_name="uncheatable_eval_bpb_single_simplex_tied_phases",
            target_metric=UNCHEATABLE_TARGET,
            panel=uncheatable_panel,
            metadata=uncheatable_metadata,
            columns=columns,
            domains=domains,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            repetition_caps=None,
            repetition_factor=None,
            output_dir=args.output_dir,
            args=args,
        )
    )
    results.append(
        fit_target(
            target_name="uncheatable_eval_bpb_rep_cap4",
            target_metric=UNCHEATABLE_TARGET,
            panel=uncheatable_panel,
            metadata={**uncheatable_metadata, "repetition_factor": REPETITION_FACTOR},
            columns=columns,
            domains=domains,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            repetition_caps=repetition_caps,
            repetition_factor=REPETITION_FACTOR,
            output_dir=args.output_dir,
            args=args,
        )
    )
    results.append(
        fit_target_single_simplex_tied_phases(
            target_name="uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4",
            target_metric=UNCHEATABLE_TARGET,
            panel=uncheatable_panel,
            metadata={**uncheatable_metadata, "repetition_factor": REPETITION_FACTOR},
            columns=columns,
            domains=domains,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            repetition_caps=repetition_caps,
            repetition_factor=REPETITION_FACTOR,
            output_dir=args.output_dir,
            args=args,
        )
    )

    if args.include_stale_olmo_easy:
        olmo_panel, olmo_metadata = build_olmo_easy_macro_panel(columns)
        results.append(
            fit_target(
                target_name="olmo_base_easy_top3_macro_bpb",
                target_metric="olmo_base_easy_top3_macro_bpb",
                panel=olmo_panel,
                metadata=olmo_metadata,
                columns=columns,
                domains=domains,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                repetition_caps=None,
                repetition_factor=None,
                output_dir=args.output_dir,
                args=args,
            )
        )

    pd.DataFrame([asdict(row) for row in results]).to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump([asdict(row) for row in results], f, indent=2, sort_keys=True)
    write_report(args.output_dir, results)
    write_uncheatable_variant_comparison(
        args.output_dir,
        two_variant="uncheatable_eval_bpb",
        tied_variant="uncheatable_eval_bpb_single_simplex_tied_phases",
        output_prefix="uncheatable_olmix_uncapped",
        title_suffix=" uncapped",
    )
    write_uncheatable_variant_comparison(
        args.output_dir,
        two_variant="uncheatable_eval_bpb_rep_cap4",
        tied_variant="uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4",
        output_prefix="uncheatable_olmix_rep_cap4",
        title_suffix=" rep cap 4",
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
