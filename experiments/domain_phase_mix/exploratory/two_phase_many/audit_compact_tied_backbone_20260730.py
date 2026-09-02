# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Phase-blind stability audit for the Compact Retained State aggregate backbone.

The 300M audit uses only the 282 physically tied policies from the original
two-phase fit panel plus the qsplit240 exposure-average ablation. The 238
aggregate-matched asymmetric policies are never read by model selection or
scoring. WSD80 likewise uses only the sampled tied diagonal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_retained_weibull_replay_20260713 as compact,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "compact_tied_backbone_audit_20260730"
L2_GRID = (0.0, 0.001, 0.01, 0.1, 1.0)
TARGETS = ("uncheatable", "table9")
OUTER_SPLITS = 3
INNER_SPLITS = 3
OPTIMIZER_STARTS = 16
ZERO_TOLERANCE = 1e-10
WEIGHT_ZERO_TOLERANCE = 1e-6
GATES = {
    "uncheatable_oof_rmse": 0.0056,
    "table9_oof_rmse": 0.0125,
    "median_optimum_l1": 0.05,
    "maximum_zero_amplitudes": 8,
    "wsd_tied_optimum_distance": 0.05,
    "wsd_minimum_predicted_optimum_bpb": 0.940429,
}


@dataclass(frozen=True)
class NestedResult:
    """Nested-CV predictions and fitted outer-fold optima."""

    prediction: np.ndarray
    selected_l2: tuple[float, ...]
    optima: tuple[np.ndarray, ...]
    zero_amplitudes: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def as_pooled_dataset(
    name: str,
    frame: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    domain_names: tuple[str, ...] | list[str],
) -> pooled.Dataset:
    """Build the dataset interface used by the maintained Compact fitter."""
    return pooled.Dataset(
        name=name,
        frame=frame.reset_index(drop=True),
        y=np.asarray(target, dtype=float),
        weights=np.asarray(weights, dtype=float),
        c0=np.asarray(c0, dtype=float),
        c1=np.asarray(c1, dtype=float),
        domain_names=list(domain_names),
    )


def fit_compact(dataset: pooled.Dataset, indices: np.ndarray, l2: float) -> compact.FittedModel:
    """Fit the independently restricted one-phase Compact model."""
    return compact.fit_model(
        dataset,
        np.asarray(indices, dtype=int),
        observatory.COMPACT_ONE_PHASE_CONFIG,
        l2,
        maxiter=24,
        top_k=2,
    )


def group_folds(
    indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Split row indices while keeping aliases of one policy together."""
    subset_groups = np.asarray(groups, dtype=str)[indices]
    unique = np.unique(subset_groups)
    local_frame = pd.DataFrame({"phase_correspondence_key": unique})
    local = benchmark.grouped_folds(local_frame, seed, min(n_splits, len(unique)))
    folds = []
    for train_groups, test_groups in local:
        train_names = set(unique[train_groups])
        test_names = set(unique[test_groups])
        train = indices[np.fromiter((group in train_names for group in subset_groups), dtype=bool)]
        test = indices[np.fromiter((group in test_names for group in subset_groups), dtype=bool)]
        folds.append((train, test))
    return tuple(folds)


def select_l2(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, list[dict[str, float]]]:
    """Select ridge only from predictions inside the supplied training split."""
    rows = []
    for l2 in L2_GRID:
        prediction = np.full(dataset.n, np.nan)
        for train, test in folds:
            model = fit_compact(dataset, train, l2)
            prediction[test] = model.predict(dataset.weights[test])
        if not np.isfinite(prediction[indices]).all():
            raise ValueError("incomplete inner Compact prediction")
        residual = prediction[indices] - dataset.y[indices]
        rows.append({"l2": l2, "rmse": float(np.sqrt(np.mean(residual**2)))})
    selected = min(rows, key=lambda row: (row["rmse"], row["l2"]))
    return float(selected["l2"]), rows


def tied_prediction_and_gradient(model: compact.FittedModel, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate a tied Compact policy and its gradient in simplex coordinates."""
    weights = np.asarray(weights, dtype=float)
    exposure_scale = model.c0 + model.c1
    exposure = np.maximum(weights * exposure_scale, 0.0)
    scaled_power = (model.shape.rate * exposure) ** model.shape.power
    signal = -np.expm1(-scaled_power)
    prediction = float(model.intercept - signal @ model.signal_coef)
    derivative = np.divide(
        model.shape.power * scaled_power * np.exp(-scaled_power),
        exposure,
        out=np.zeros_like(exposure),
        where=exposure > 1e-12,
    )
    gradient = -model.signal_coef * derivative * exposure_scale
    if len(model.replay_coef):
        repeated = np.maximum(exposure - 1.0, 0.0)
        replay_coefficient = float(model.replay_coef[0])
        prediction += replay_coefficient * float(np.sum(repeated**2))
        gradient += 2.0 * replay_coefficient * repeated * exposure_scale
    tied = np.stack([weights, weights], axis=0)[None, :, :]
    direct = float(model.predict(tied)[0])
    if not np.isclose(prediction, direct, atol=1e-10, rtol=1e-10):
        raise AssertionError(f"analytic Compact prediction mismatch: {prediction} != {direct}")
    return prediction, gradient


def optimizer_starts(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    seed: int,
    count: int,
) -> tuple[np.ndarray, ...]:
    """Deterministic, support-spanning starts for raw tied optimization."""
    tied_weights = dataset.weights[indices, 0, :]
    observed_best = tied_weights[int(np.argmin(dataset.y[indices]))]
    equal_epoch = 1.0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)
    equal_epoch /= equal_epoch.sum()
    starts = [observed_best, np.full(dataset.m, 1.0 / dataset.m), equal_epoch]
    generator = np.random.default_rng(seed)
    while len(starts) < count:
        starts.append(generator.dirichlet(np.ones(dataset.m)))
    return tuple(np.asarray(start, dtype=float) for start in starts)


def optimize_tied(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    model: compact.FittedModel,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Optimize the raw tied surrogate with no deployment regularization."""

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return tied_prediction_and_gradient(model, weights)

    constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda weights: np.ones_like(weights),
    }
    candidates = []
    for start in optimizer_starts(dataset, indices, seed, OPTIMIZER_STARTS):
        result = minimize(
            objective,
            start,
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * dataset.m,
            constraints=[constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
            weights /= weights.sum()
            candidates.append((float(result.fun), weights))
    if not candidates:
        raise RuntimeError(f"no finite tied optimum for {dataset.name}")
    return min(candidates, key=lambda candidate: candidate[0])[1], min(candidate[0] for candidate in candidates)


def nested_compact(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    groups: np.ndarray,
    seed: int,
) -> NestedResult:
    """Nested ridge selection with a raw optimum for every outer training fold."""
    prediction = np.full(dataset.n, np.nan)
    selected_l2 = []
    optima = []
    zero_amplitudes = []
    for fold_id, (train, test) in enumerate(outer_folds):
        inner = group_folds(train, groups, INNER_SPLITS, seed + 100 + fold_id)
        l2, _rows = select_l2(dataset, train, inner)
        model = fit_compact(dataset, train, l2)
        prediction[test] = model.predict(dataset.weights[test])
        optimum, _value = optimize_tied(dataset, train, model, seed + 1000 + fold_id)
        selected_l2.append(l2)
        optima.append(optimum)
        zero_amplitudes.append(int(np.sum(model.signal_coef <= ZERO_TOLERANCE)))
    if not np.isfinite(prediction[indices]).all():
        raise ValueError("incomplete outer Compact prediction")
    return NestedResult(
        prediction=prediction,
        selected_l2=tuple(selected_l2),
        optima=tuple(optima),
        zero_amplitudes=tuple(zero_amplitudes),
    )


def full_compact(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[compact.FittedModel, float, np.ndarray, float, list[dict[str, float]]]:
    """Select ridge by CV and fit the full phase-blind backbone."""
    folds = group_folds(indices, groups, INNER_SPLITS, seed + 500)
    l2, sweep = select_l2(dataset, indices, folds)
    model = fit_compact(dataset, indices, l2)
    optimum, value = optimize_tied(dataset, indices, model, seed + 1500)
    return model, l2, optimum, value, sweep


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    fitted = intercept + slope * predicted
    centered = observed - np.mean(observed)
    total_sum_squares = float(centered @ centered)
    residual_sum_squares = float((observed - fitted) @ (observed - fitted))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias": float(np.mean(residual)),
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
        "observed_on_predicted_r2": float(1.0 - residual_sum_squares / total_sum_squares),
    }


def audit_300m(target: str, seed: int) -> tuple[dict[str, float | int | str | bool], pd.DataFrame, pd.DataFrame]:
    source = benchmark.load_300m(target)
    tied = benchmark.replay_control.tied_rows(source.weights)
    indices = np.flatnonzero(tied)
    if len(indices) != 282:
        raise ValueError(f"expected 282 tied rows, found {len(indices)}")
    dataset = as_pooled_dataset(
        source.name,
        source.frame,
        source.y,
        source.weights,
        source.c0,
        source.c1,
        source.domain_names,
    )
    groups = source.frame["phase_correspondence_key"].astype(str).to_numpy()
    outer = group_folds(indices, groups, OUTER_SPLITS, seed)
    nested = nested_compact(dataset, indices, outer, groups, seed)
    model, l2, full_optimum, predicted_optimum, sweep = full_compact(dataset, indices, groups, seed)
    outer_distances = np.asarray([np.abs(optimum - full_optimum).sum() for optimum in nested.optima])
    observed_best_index = indices[int(np.argmin(dataset.y[indices]))]
    observed_best = dataset.weights[observed_best_index, 0, :]
    metrics = metric_summary(dataset.y[indices], nested.prediction[indices])
    exposure = full_optimum * (dataset.c0 + dataset.c1)
    zero_amplitudes = int(np.sum(model.signal_coef <= ZERO_TOLERANCE))
    threshold = GATES[f"{target}_oof_rmse"]
    row: dict[str, float | int | str | bool] = {
        "target": target,
        "n_tied": len(indices),
        "selected_l2": l2,
        "oof_rmse": metrics["rmse"],
        "oof_spearman": metrics["spearman"],
        "oof_bias": metrics["bias"],
        "predicted_optimum_bpb": predicted_optimum,
        "observed_best_tied_bpb": float(dataset.y[observed_best_index]),
        "optimum_l1_to_observed_best": float(np.abs(full_optimum - observed_best).sum()),
        "median_fold_to_full_optimum_l1": float(np.median(outer_distances)),
        "maximum_fold_to_full_optimum_l1": float(np.max(outer_distances)),
        "zero_amplitudes": zero_amplitudes,
        "maximum_optimum_weight": float(np.max(full_optimum)),
        "maximum_optimum_epochs": float(np.max(exposure)),
        "near_zero_optimum_weights": int(np.sum(full_optimum <= WEIGHT_ZERO_TOLERANCE)),
        "passes_oof_gate": bool(metrics["rmse"] <= threshold),
        "passes_stability_gate": bool(np.median(outer_distances) <= GATES["median_optimum_l1"]),
        "passes_amplitude_gate": bool(zero_amplitudes <= GATES["maximum_zero_amplitudes"]),
    }
    fold_rows = pd.DataFrame(
        [
            {
                "target": target,
                "fold": fold_id,
                "selected_l2": nested.selected_l2[fold_id],
                "fold_to_full_optimum_l1": outer_distances[fold_id],
                "zero_amplitudes": nested.zero_amplitudes[fold_id],
                **{f"weight_{name}": value for name, value in zip(dataset.domain_names, optimum, strict=True)},
            }
            for fold_id, optimum in enumerate(nested.optima)
        ]
    )
    sweep_frame = pd.DataFrame([{"target": target, **entry} for entry in sweep])
    return row, fold_rows, sweep_frame


def wsd_dataset() -> tuple[pooled.Dataset, np.ndarray]:
    panel = wsd80.load_surface()
    tied = np.flatnonzero(np.isclose(panel.weights[:, 0, 1], panel.weights[:, 1, 1]))
    dataset = as_pooled_dataset(
        panel.name,
        panel.frame,
        panel.y,
        panel.weights,
        panel.c0,
        panel.c1,
        panel.domain_names,
    )
    return dataset, tied


def wsd_cv(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    protocol: str,
    seed: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    outer = benchmark.wsd_folds(dataset.weights, indices, OUTER_SPLITS, seed, protocol)
    prediction = np.full(dataset.n, np.nan)
    selected = []
    for fold_id, (train, test) in enumerate(outer):
        inner = benchmark.wsd_folds(
            dataset.weights,
            train,
            min(INNER_SPLITS, len(train)),
            seed + 100 + fold_id,
            protocol,
        )
        l2, _rows = select_l2(dataset, train, inner)
        prediction[test] = fit_compact(dataset, train, l2).predict(dataset.weights[test])
        selected.append(l2)
    if not np.isfinite(prediction[indices]).all():
        raise ValueError(f"incomplete WSD {protocol} prediction")
    return prediction, tuple(selected)


def audit_wsd(
    seed: int,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame, pd.DataFrame]:
    dataset, tied = wsd_dataset()
    random_prediction, random_l2 = wsd_cv(dataset, tied, "random", seed)
    blocked_prediction, blocked_l2 = wsd_cv(dataset, tied, "blocked", seed)
    groups = np.asarray([f"wsd_tied_{index}" for index in range(dataset.n)])
    model, l2, _optimum, _value, sweep = full_compact(dataset, tied, groups, seed)
    axis = np.linspace(0.0, 1.0, 10001)
    tied_weights = benchmark.grid_weights(axis, axis)
    values = model.predict(tied_weights)
    optimum_index = int(np.argmin(values))
    optimum_share = float(axis[optimum_index])
    observed_index = tied[int(np.argmin(dataset.y[tied]))]
    observed_share = float(dataset.weights[observed_index, 0, 1])
    random_metrics = metric_summary(dataset.y[tied], random_prediction[tied])
    blocked_metrics = metric_summary(dataset.y[tied], blocked_prediction[tied])
    row: dict[str, float | int | str | bool] = {
        "target": "starcoder_bpb",
        "n_tied": len(tied),
        "selected_l2": l2,
        "random_oof_rmse": random_metrics["rmse"],
        "blocked_oof_rmse": blocked_metrics["rmse"],
        "random_oof_spearman": random_metrics["spearman"],
        "blocked_oof_spearman": blocked_metrics["spearman"],
        "random_observed_on_predicted_slope": random_metrics["observed_on_predicted_slope"],
        "random_observed_on_predicted_r2": random_metrics["observed_on_predicted_r2"],
        "blocked_observed_on_predicted_slope": blocked_metrics["observed_on_predicted_slope"],
        "blocked_observed_on_predicted_r2": blocked_metrics["observed_on_predicted_r2"],
        "predicted_tied_optimum_share": optimum_share,
        "predicted_tied_optimum_bpb": float(values[optimum_index]),
        "observed_tied_optimum_share": observed_share,
        "observed_tied_optimum_bpb": float(dataset.y[observed_index]),
        "zero_amplitudes": int(np.sum(model.signal_coef <= ZERO_TOLERANCE)),
        "passes_optimum_location_gate": bool(abs(optimum_share - observed_share) <= GATES["wsd_tied_optimum_distance"]),
        "passes_optimum_value_gate": bool(values[optimum_index] >= GATES["wsd_minimum_predicted_optimum_bpb"]),
    }
    sweep_frame = pd.DataFrame([{"target": "starcoder_bpb", **entry} for entry in sweep])
    sweep_frame["random_outer_l2_json"] = json.dumps(random_l2)
    sweep_frame["blocked_outer_l2_json"] = json.dumps(blocked_l2)
    prediction_frame = pd.DataFrame(
        {
            "row": tied,
            "starcoder_weight": dataset.weights[tied, 0, 1],
            "observed": dataset.y[tied],
            "random_oof_prediction": random_prediction[tied],
            "blocked_oof_prediction": blocked_prediction[tied],
        }
    )
    return row, sweep_frame, prediction_frame


def write_report(metrics: pd.DataFrame, output_dir: Path) -> Path:
    status_columns = [column for column in metrics if column.startswith("passes_")]
    metrics["passes_all_gates"] = metrics[status_columns].fillna(True).all(axis=1)
    lines = [
        "# Compact tied-backbone stability audit",
        "",
        "Only physically tied rows were used. No asymmetric outcome entered fitting, selection, or scoring.",
        "",
        metrics.to_markdown(index=False),
        "",
        "The earlier WSD RMSE thresholds were invalidated because they came from an incumbent's",
        "full-panel in-sample residuals rather than nested OOF predictions. WSD RMSE is reported",
        "descriptively here and is not an acceptance gate.",
        "",
        "The backbone is eligible for a phase-mechanism iteration only if every applicable gate passes.",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    fold_frames = []
    sweep_frames = []
    for target in TARGETS:
        row, folds, sweep = audit_300m(target, args.seed)
        metric_rows.append(row)
        fold_frames.append(folds)
        sweep_frames.append(sweep)
    wsd_row, wsd_sweep, wsd_predictions = audit_wsd(args.seed)
    metric_rows.append(wsd_row)
    sweep_frames.append(wsd_sweep)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(args.output_dir / "fold_optima_300m.csv", index=False)
    pd.concat(sweep_frames, ignore_index=True).to_csv(args.output_dir / "l2_sweeps.csv", index=False)
    wsd_predictions.to_csv(args.output_dir / "wsd_tied_oof_predictions.csv", index=False)
    (args.output_dir / "gate.json").write_text(json.dumps(GATES, indent=2, sort_keys=True) + "\n")
    report = write_report(metrics, args.output_dir)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
