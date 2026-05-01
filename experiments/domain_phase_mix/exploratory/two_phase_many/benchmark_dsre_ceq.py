# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Benchmark DS-RE-CEQ and Olmix loglinear on the two-phase many-domain swarm."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    DatasetSpec,
    GENERAL_MODELS,
)
from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.static_batch_selection import (
    build_dataset_spec_from_frame,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "two_phase_many.csv"
OBJECTIVE_METRIC = "lm_eval/mmlu_5shot/bpb"
N_FOLDS = 5
CV_SEED = 42
SUBSET_SIZES = tuple(range(20, 241, 20))
FEATURE_POLICY = "feature_bayes_linear_observed"
DSRE_MODEL_NAME = "DS-RE-CEQ"
OLMIX_MODEL_NAME = "Olmix loglinear"
OLMIX_HUBER_DELTA = 0.02
OLMIX_N_STARTS = 48
OPT_SEARCH_POINTS = 8192
OPT_SEARCH_SEED = 42
MAX_WORKERS = min(6, max(1, (os.cpu_count() or 1) - 2))

SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_dsre_ceq_summary.json"
OOF_PREDICTIONS_CSV = SCRIPT_DIR / "two_phase_many_dsre_ceq_oof_predictions.csv"
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_dsre_ceq_vs_olmix_curve_points.csv"
FULL_OPTIMA_JSON = SCRIPT_DIR / "two_phase_many_model_optima.json"
DSRE_PLOT_PATH = SCRIPT_DIR / "two_phase_many_dsre_ceq_predicted_bpb_regret.png"
COMPARISON_PLOT_PATH = SCRIPT_DIR / "two_phase_many_dsre_ceq_vs_olmix_predicted_bpb_regret.png"


@dataclass(frozen=True)
class CvMetrics:
    r2: float
    rmse: float
    spearman: float
    regret_at_1: float
    n_params: int


@dataclass(frozen=True)
class FitResult:
    predict_fn: object
    n_params: int
    info: dict[str, object]


@dataclass(frozen=True)
class PredictedOptimum:
    predicted_objective: float
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class OlmixFit:
    log_c: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        matrix = np.asarray(weights, dtype=float).reshape(len(weights), -1)
        logits = np.clip(matrix @ self.coefficients, -50.0, 50.0)
        return np.exp(self.log_c) + np.exp(logits)

    def optimum(self, spec: DatasetSpec) -> PredictedOptimum:
        coefficients = self.coefficients.reshape(spec.N, spec.M)
        point = np.zeros((spec.N, spec.M), dtype=float)
        for phase_idx in range(spec.N):
            best_domain_idx = int(np.argmin(coefficients[phase_idx]))
            point[phase_idx, best_domain_idx] = 1.0
        predicted_objective = float(self.predict(point[None, :, :])[0])
        return PredictedOptimum(
            predicted_objective=predicted_objective,
            phase_weights=_phase_weights_from_point(point, spec),
        )


_SCRIPT_START = perf_counter()


def _log(message: str) -> None:
    elapsed = perf_counter() - _SCRIPT_START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def _model_map() -> dict[str, object]:
    return {model.name: model for model in GENERAL_MODELS}


def _build_loop_config() -> LoopConfig:
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="two_phase_many_analysis")
    phase_fractions = tuple(phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases)
    domain_token_counts = {domain.name: int(domain.total_weight) for domain in experiment.domains}
    return LoopConfig(
        name="two_phase_many_analysis",
        objective_metric=OBJECTIVE_METRIC,
        model_names=(DSRE_MODEL_NAME,),
        domain_token_counts=domain_token_counts,
        phase_fractions=phase_fractions,
        target_budget=experiment.target_budget,
        candidate_search_points=OPT_SEARCH_POINTS,
        candidate_search_seed=OPT_SEARCH_SEED,
    )


def _load_spec() -> tuple[pd.DataFrame, DatasetSpec, LoopConfig]:
    frame = pd.read_csv(CSV_PATH)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "completed"].reset_index(drop=True)
    loop = _build_loop_config()
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many",
        loop=loop,
    )
    return frame, spec, loop


def _fit_dsre(spec: DatasetSpec, *, seed: int = 0) -> FitResult:
    model = _model_map()[DSRE_MODEL_NAME]
    predict_fn, info = model.fit_fn(spec, seed=seed, n_restarts=8, maxiter=500)
    return FitResult(predict_fn=predict_fn, n_params=int(info["n_params"]), info=info)


def _huber_sum(residuals: np.ndarray, *, delta: float) -> float:
    abs_residuals = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (abs_residuals - 0.5 * delta)
    return float(np.where(abs_residuals <= delta, quadratic, linear).sum())


def _fit_olmix_loglinear(spec: DatasetSpec, *, seed: int = 0, n_starts: int = OLMIX_N_STARTS) -> FitResult:
    x = np.asarray(spec.weights, dtype=float).reshape(spec.R, -1)
    y = np.asarray(spec.y, dtype=float)
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        log_c = float(params[0])
        coefficients = params[1:]
        logits = np.clip(x @ coefficients, -50.0, 50.0)
        predictions = np.exp(log_c) + np.exp(logits)
        return _huber_sum(predictions - y, delta=OLMIX_HUBER_DELTA)

    best_params = None
    best_loss = float("inf")
    starts: list[np.ndarray] = []
    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    for log_c in log_c_candidates:
        starts.append(np.concatenate([[log_c], np.zeros(x.shape[1], dtype=float)]))
        for _ in range(max(n_starts // len(log_c_candidates) - 1, 0)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))

    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B")
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)

    if best_params is None:
        raise RuntimeError("Olmix loglinear fit failed")

    fit = OlmixFit(log_c=float(best_params[0]), coefficients=np.asarray(best_params[1:], dtype=float))
    return FitResult(
        predict_fn=fit.predict,
        n_params=int(best_params.size),
        info={"log_c": fit.log_c, "coefficients": fit.coefficients.copy(), "huber_loss": best_loss, "fit": fit},
    )


def _phase_weights_from_point(point: np.ndarray, spec: DatasetSpec) -> dict[str, dict[str, float]]:
    return {
        spec.phase_names[phase_idx]: {
            spec.domain_names[domain_idx]: float(point[phase_idx, domain_idx]) for domain_idx in range(spec.M)
        }
        for phase_idx in range(spec.N)
    }


def _sample_simplex_points(rng: np.random.Generator, n_points: int, n_dims: int) -> np.ndarray:
    raw = rng.exponential(1.0, size=(n_points, n_dims))
    return raw / raw.sum(axis=1, keepdims=True)


def _sample_predicted_optimum(
    predict_fn,
    spec: DatasetSpec,
    *,
    n_points: int = OPT_SEARCH_POINTS,
    seed: int = OPT_SEARCH_SEED,
) -> PredictedOptimum:
    rng = np.random.default_rng(seed)
    points = np.zeros((n_points, spec.N, spec.M), dtype=float)
    for phase_idx in range(spec.N):
        points[:, phase_idx, :] = _sample_simplex_points(rng, n_points, spec.M)

    predictions = np.asarray(predict_fn(points), dtype=float)
    finite_mask = np.isfinite(predictions)
    if not finite_mask.any():
        raise RuntimeError("Predicted optimum search found no finite predictions")

    best_idx = int(np.flatnonzero(finite_mask)[np.argmin(predictions[finite_mask])])
    return PredictedOptimum(
        predicted_objective=float(predictions[best_idx]),
        phase_weights=_phase_weights_from_point(points[best_idx], spec),
    )


def _kfold_indices(n_rows: int, *, n_folds: int = N_FOLDS, seed: int = CV_SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_rows)
    return [np.asarray(indices, dtype=int) for indices in np.array_split(permuted, n_folds)]


def _cross_validate_fold(
    payload: tuple[DatasetSpec, str, list[np.ndarray], int],
) -> tuple[int, np.ndarray, np.ndarray, float, int, float]:
    spec, fit_kind, folds, fold_index = payload
    start = perf_counter()
    test_idx = folds[fold_index]
    train_idx = np.concatenate([indices for current_fold, indices in enumerate(folds) if current_fold != fold_index])
    train_spec = spec.subset(train_idx)
    fit = _fit_dsre(train_spec, seed=0) if fit_kind == "dsre" else _fit_olmix_loglinear(train_spec, seed=0)
    fold_predictions = np.asarray(fit.predict_fn(spec.weights[test_idx]), dtype=float)
    y_test = spec.y[test_idx]
    chosen_idx = int(np.argmin(fold_predictions))
    regret = float(y_test[chosen_idx] - np.min(y_test))
    return fold_index, test_idx, fold_predictions, regret, fit.n_params, perf_counter() - start


def _cross_validate(spec: DatasetSpec, *, fit_kind: str) -> tuple[CvMetrics, np.ndarray]:
    _log(f"Starting 5-fold CV for {DSRE_MODEL_NAME if fit_kind == 'dsre' else OLMIX_MODEL_NAME}")
    all_predictions = np.full(spec.R, np.nan, dtype=float)
    fold_regrets: list[float] = []
    n_params: int | None = None
    folds = _kfold_indices(spec.R)
    payloads = [(spec, fit_kind, folds, fold_index) for fold_index in range(len(folds))]

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(folds))) as executor:
        futures = [executor.submit(_cross_validate_fold, payload) for payload in payloads]
        for future in as_completed(futures):
            fold_index, test_idx, fold_predictions, regret, fold_n_params, duration = future.result()
            n_params = fold_n_params
            all_predictions[test_idx] = fold_predictions
            fold_regrets.append(regret)
            _log(
                f"Finished {DSRE_MODEL_NAME if fit_kind == 'dsre' else OLMIX_MODEL_NAME} CV fold "
                f"{fold_index + 1}/{len(folds)} in {duration:.1f}s"
            )

    residuals = spec.y - all_predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((spec.y - np.mean(spec.y)) ** 2))
    spearman = float(spearmanr(spec.y, all_predictions)[0])
    metrics = CvMetrics(
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        spearman=spearman,
        regret_at_1=float(np.mean(fold_regrets)),
        n_params=int(n_params or 0),
    )
    return metrics, all_predictions


def _full_fit_optimum(spec: DatasetSpec, *, fit_kind: str) -> tuple[FitResult, PredictedOptimum]:
    _log(f"Fitting full-data {DSRE_MODEL_NAME if fit_kind == 'dsre' else OLMIX_MODEL_NAME}")
    fit = _fit_dsre(spec) if fit_kind == "dsre" else _fit_olmix_loglinear(spec)
    if fit_kind == "olmix":
        optimum = fit.info["fit"].optimum(spec)
    else:
        _log(f"Searching predicted optimum for {DSRE_MODEL_NAME}")
        optimum = _sample_predicted_optimum(fit.predict_fn, spec, seed=OPT_SEARCH_SEED)
    _log(f"Finished full-data {DSRE_MODEL_NAME if fit_kind == 'dsre' else OLMIX_MODEL_NAME}")
    return fit, optimum


def _subset_curve_rows_for_k(payload: tuple[DatasetSpec, int]) -> tuple[int, list[dict[str, float | int | str]], float]:
    spec, subset_size = payload
    start = perf_counter()
    selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
    train_spec = spec.subset(np.asarray(selection.selected_indices, dtype=int))
    rows: list[dict[str, float | int | str]] = []
    for model_name, fit_kind in ((DSRE_MODEL_NAME, "dsre"), (OLMIX_MODEL_NAME, "olmix")):
        fit = _fit_dsre(train_spec) if fit_kind == "dsre" else _fit_olmix_loglinear(train_spec)
        pool_predictions = np.asarray(fit.predict_fn(spec.weights), dtype=float)
        chosen_idx = int(np.argmin(pool_predictions))
        if fit_kind == "olmix":
            optimum = fit.info["fit"].optimum(spec)
        else:
            optimum = _sample_predicted_optimum(fit.predict_fn, spec, seed=OPT_SEARCH_SEED + subset_size)
        rows.append(
            {
                "subset_size": subset_size,
                "policy": FEATURE_POLICY,
                "model_name": model_name,
                "predicted_bpb": float(optimum.predicted_objective),
                "regret_at_1": float(spec.y[chosen_idx] - np.min(spec.y)),
                "n_params": fit.n_params,
            }
        )
    return subset_size, rows, perf_counter() - start


def _subset_curve_rows(spec: DatasetSpec) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    payloads = [(spec, subset_size) for subset_size in SUBSET_SIZES]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(SUBSET_SIZES))) as executor:
        futures = [executor.submit(_subset_curve_rows_for_k, payload) for payload in payloads]
        for future in as_completed(futures):
            subset_size, subset_rows, duration = future.result()
            rows.extend(subset_rows)
            pd.DataFrame(rows).to_csv(CURVE_POINTS_CSV, index=False)
            _log(f"Finished subset size k={subset_size} in {duration:.1f}s")
    return pd.DataFrame(rows)


def _top_phase_weights(
    phase_weights: dict[str, dict[str, float]], *, top_k: int = 8
) -> dict[str, list[dict[str, float | str]]]:
    top: dict[str, list[dict[str, float | str]]] = {}
    for phase_name, weights in phase_weights.items():
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_k]
        top[phase_name] = [{"domain": domain, "weight": float(weight)} for domain, weight in ranked]
    return top


def _write_summary(
    *,
    metrics: CvMetrics,
    optimum: PredictedOptimum,
    comparison_metrics: CvMetrics,
    comparison_optimum: PredictedOptimum,
) -> None:
    payload = {
        "objective_metric": OBJECTIVE_METRIC,
        "dsre_ceq": {
            "r2": metrics.r2,
            "rmse": metrics.rmse,
            "spearman": metrics.spearman,
            "regret_at_1": metrics.regret_at_1,
            "n_params": metrics.n_params,
            "predicted_bpb": optimum.predicted_objective,
            "top_phase_weights": _top_phase_weights(optimum.phase_weights),
        },
        "olmix_loglinear": {
            "r2": comparison_metrics.r2,
            "rmse": comparison_metrics.rmse,
            "spearman": comparison_metrics.spearman,
            "regret_at_1": comparison_metrics.regret_at_1,
            "n_params": comparison_metrics.n_params,
            "predicted_bpb": comparison_optimum.predicted_objective,
            "top_phase_weights": _top_phase_weights(comparison_optimum.phase_weights),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    FULL_OPTIMA_JSON.write_text(
        json.dumps(
            {
                "dsre_ceq": optimum.phase_weights,
                "olmix_loglinear": comparison_optimum.phase_weights,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _plot_dsre_only(curves: pd.DataFrame, best_observed_bpb: float) -> None:
    frame = curves[curves["model_name"] == DSRE_MODEL_NAME].sort_values("subset_size")
    cmap = plt.colormaps["RdYlGn_r"]
    fig, ax_bpb = plt.subplots(figsize=(10, 6), dpi=180)
    ax_regret = ax_bpb.twinx()

    ax_bpb.plot(
        frame["subset_size"],
        frame["predicted_bpb"],
        color=cmap(0.18),
        marker="o",
        linewidth=2.2,
        linestyle="--",
        label="DS-RE-CEQ predicted BPB",
    )
    ax_bpb.axhline(
        best_observed_bpb,
        color=cmap(0.55),
        linewidth=1.8,
        linestyle=":",
        label=f"Best observed BPB ({best_observed_bpb:.4f})",
    )
    ax_regret.plot(
        frame["subset_size"],
        frame["regret_at_1"],
        color=cmap(0.82),
        marker="s",
        linewidth=2.2,
        linestyle="-",
        label="DS-RE-CEQ Regret@1",
    )

    ax_bpb.set_title("Two-phase many-domain: DS-RE-CEQ predicted BPB and Regret@1")
    ax_bpb.set_xlabel("Observed runs used for fitting")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_bpb.set_xticks(list(SUBSET_SIZES))
    ax_bpb.set_xlim(min(SUBSET_SIZES), max(SUBSET_SIZES))
    ax_bpb.grid(True, alpha=0.25)

    handles = ax_bpb.get_lines() + ax_regret.get_lines()
    labels = [handle.get_label() for handle in handles]
    ax_bpb.legend(handles, labels, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(DSRE_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(curves: pd.DataFrame, best_observed_bpb: float) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    fig, ax_bpb = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    ax_regret = ax_bpb.twinx()

    styles = {
        DSRE_MODEL_NAME: {
            "pred_color": cmap(0.18),
            "regret_color": cmap(0.82),
            "pred_marker": "o",
            "regret_marker": "s",
        },
        OLMIX_MODEL_NAME: {
            "pred_color": cmap(0.35),
            "regret_color": cmap(0.95),
            "pred_marker": "^",
            "regret_marker": "D",
        },
    }

    for model_name in (DSRE_MODEL_NAME, OLMIX_MODEL_NAME):
        frame = curves[curves["model_name"] == model_name].sort_values("subset_size")
        style = styles[model_name]
        ax_bpb.plot(
            frame["subset_size"],
            frame["predicted_bpb"],
            color=style["pred_color"],
            marker=style["pred_marker"],
            linewidth=2.1,
            linestyle="--",
            label=f"{model_name} predicted BPB",
        )
        ax_regret.plot(
            frame["subset_size"],
            frame["regret_at_1"],
            color=style["regret_color"],
            marker=style["regret_marker"],
            linewidth=2.1,
            linestyle="-",
            label=f"{model_name} Regret@1",
        )

    ax_bpb.axhline(
        best_observed_bpb,
        color=cmap(0.58),
        linewidth=1.8,
        linestyle=":",
        label=f"Best observed BPB ({best_observed_bpb:.4f})",
    )

    ax_bpb.set_title("Two-phase many-domain: DS-RE-CEQ vs Olmix loglinear")
    ax_bpb.set_xlabel("Observed runs used for fitting")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_bpb.set_xticks(list(SUBSET_SIZES))
    ax_bpb.set_xlim(min(SUBSET_SIZES), max(SUBSET_SIZES))
    ax_bpb.grid(True, alpha=0.25)

    handles = ax_bpb.get_lines() + ax_regret.get_lines()
    labels = [handle.get_label() for handle in handles]
    ax_bpb.legend(handles, labels, loc="best", frameon=True, ncol=1)
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _log(f"Loading {CSV_PATH}")
    frame, spec, _loop = _load_spec()
    dsre_metrics, dsre_oof_predictions = _cross_validate(spec, fit_kind="dsre")
    olmix_metrics, _ = _cross_validate(spec, fit_kind="olmix")
    _dsre_fit, dsre_optimum = _full_fit_optimum(spec, fit_kind="dsre")
    _olmix_fit, olmix_optimum = _full_fit_optimum(spec, fit_kind="olmix")

    oof_frame = frame[["run_name", OBJECTIVE_METRIC]].copy()
    oof_frame["dsre_ceq_oof_prediction"] = dsre_oof_predictions
    oof_frame.to_csv(OOF_PREDICTIONS_CSV, index=False)
    _log(f"Wrote {OOF_PREDICTIONS_CSV}")

    curve_points = _subset_curve_rows(spec)
    curve_points.to_csv(CURVE_POINTS_CSV, index=False)
    _log(f"Wrote {CURVE_POINTS_CSV}")

    _write_summary(
        metrics=dsre_metrics,
        optimum=dsre_optimum,
        comparison_metrics=olmix_metrics,
        comparison_optimum=olmix_optimum,
    )
    best_observed_bpb = float(np.min(spec.y))
    _plot_dsre_only(curve_points, best_observed_bpb)
    _plot_comparison(curve_points, best_observed_bpb)
    _log(f"Wrote plots to {DSRE_PLOT_PATH} and {COMPARISON_PLOT_PATH}")

    print(
        json.dumps(
            {
                "dsre_ceq": {
                    "r2": dsre_metrics.r2,
                    "rmse": dsre_metrics.rmse,
                    "spearman": dsre_metrics.spearman,
                    "regret_at_1": dsre_metrics.regret_at_1,
                    "n_params": dsre_metrics.n_params,
                    "predicted_bpb": dsre_optimum.predicted_objective,
                },
                "olmix_loglinear": {
                    "r2": olmix_metrics.r2,
                    "rmse": olmix_metrics.rmse,
                    "spearman": olmix_metrics.spearman,
                    "regret_at_1": olmix_metrics.regret_at_1,
                    "n_params": olmix_metrics.n_params,
                    "predicted_bpb": olmix_optimum.predicted_objective,
                },
                "artifacts": {
                    "summary_json": str(SUMMARY_JSON),
                    "curve_points_csv": str(CURVE_POINTS_CSV),
                    "dsre_plot": str(DSRE_PLOT_PATH),
                    "comparison_plot": str(COMPARISON_PLOT_PATH),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
