# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predict final validation loss from early prefixes of Delphi midtraining runs.

This is the within-run companion to ``delphi_small_final_loss_scaling.py``.
It caches validation trajectories, tunes prefix-prediction rules on the clean
small ladder through ``3e20``, and evaluates the same rules on held-out
``1e21``/``1e22`` runs.

Outputs:
    midtrain_analysis_outputs/small_final_loss_scaling/trajectory_points.csv
    midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prefix_predictions.csv
    midtrain_analysis_outputs/small_final_loss_scaling/trajectory_prefix_summary.csv
    midtrain_analysis_outputs/small_final_loss_scaling/trajectory_method_selection.csv

Run:
    uv run python scripts/analysis/delphi_within_run_prediction.py
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from delphi_small_final_loss_scaling import (
    OUT_DIR,
    PROJECT,
)
from scipy.optimize import Bounds, least_squares, minimize

logger = logging.getLogger("delphi_within_run_prediction")

TRAJECTORY_POINTS_PATH = OUT_DIR / "trajectory_points.csv"
TRAJECTORY_PREDICTIONS_PATH = OUT_DIR / "trajectory_prefix_predictions.csv"
TRAJECTORY_SUMMARY_PATH = OUT_DIR / "trajectory_prefix_summary.csv"
TRAJECTORY_SELECTION_PATH = OUT_DIR / "trajectory_method_selection.csv"

ENDPOINTS_PATH = OUT_DIR / "endpoints.csv"
TARGETS_PATH = OUT_DIR / "extrapolation_targets.csv"

VALIDATION_METRICS = {
    "eval/nemotron_cc_math_v1/4plus/loss": "math_val_loss",
    "eval/loss": "eval_loss",
    "eval/paloma/macro_loss": "paloma_macro_loss",
    "eval/paloma/c4_en/loss": "paloma_c4_loss",
}
PREFIX_FRACS = tuple(round(percent / 100, 2) for percent in range(10, 91, 5))
PARAMETRIC_BASE_METHODS = ("curve_log", "curve_exp", "curve_power", "curve_rational")
PARAMETRIC_FIT_LOSSES = ("mae", "huber")
METHODS = (
    "last_value",
    "linear_tau",
    "template_global",
    "template_by_mix",
    "template_by_recipe",
    *(f"{method}_{fit_loss}" for method in PARAMETRIC_BASE_METHODS for fit_loss in PARAMETRIC_FIT_LOSSES),
)
MIN_POINTS_FOR_LINEAR = 2
MIN_POINTS_FOR_PARAMETRIC = 3
MIN_TEMPLATE_RUNS = 3
SELECTION_REL_TOLERANCE = 0.10
SELECTION_ABS_TOLERANCE = 0.002
HUBER_DELTA_FRACTION = 0.10
SCIPY_MAXITER = 200
SCIPY_MAX_NFEV = 300
PARAMETRIC_METRIC_LABELS = {"math_val_loss"}

PARAMETRIC_SHAPE_GRID: dict[str, tuple[tuple[float, float | None], ...]] = {
    "curve_log": tuple((shift, None) for shift in (0.01, 0.03, 0.10, 0.30, 1.00)),
    "curve_exp": tuple((rate, None) for rate in (1.0, 3.0, 8.0, 18.0)),
    "curve_power": tuple((shift, exponent) for shift in (0.03, 0.10, 0.30) for exponent in (0.50, 1.0, 2.0, 4.0)),
    "curve_rational": tuple((t0, beta) for t0 in (0.10, 0.30, 0.80, 1.50) for beta in (0.75, 1.0, 2.0, 4.0)),
}
PARAMETRIC_SHAPE_BOUNDS: dict[str, tuple[tuple[float, float], ...]] = {
    "curve_log": ((1e-4, 10.0),),
    "curve_exp": ((0.01, 50.0),),
    "curve_power": ((1e-4, 10.0), (0.05, 10.0)),
    "curve_rational": ((0.01, 10.0), (0.10, 10.0)),
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_name: str
    url: str
    scale: str
    mix: str
    lr: str
    state: str
    eval_split: str
    target_kind: str
    complete: bool
    final_step: int

    @property
    def recipe(self) -> str:
        return f"{self.mix}-lr{self.lr}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cache", action="store_true", help="Reuse trajectory_points.csv instead of querying W&B")
    parser.add_argument("--project", default=PROJECT, help="W&B entity/project")
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value):
        return bool(value)
    return False


def final_step_from_expected(value: Any, fallback: Any = None) -> int:
    expected = finite_float(value)
    if expected is not None and expected > 1:
        return int(expected) - 1
    fallback_value = finite_float(fallback)
    if fallback_value is not None:
        return int(fallback_value)
    raise ValueError("missing expected final step")


def required_outputs_exist() -> bool:
    return ENDPOINTS_PATH.exists() and TARGETS_PATH.exists()


def load_run_specs_and_targets() -> tuple[list[RunSpec], pd.DataFrame]:
    if not required_outputs_exist():
        raise FileNotFoundError(
            f"Missing {ENDPOINTS_PATH} or {TARGETS_PATH}. Run delphi_small_final_loss_scaling.py first."
        )

    endpoints = pd.read_csv(ENDPOINTS_PATH, dtype={"scale": str, "lr": str})
    targets = pd.read_csv(TARGETS_PATH, dtype={"scale": str, "lr": str})
    target_values = pd.concat(
        [
            endpoints.assign(eval_split="small_cv", target_kind="complete"),
            targets.assign(eval_split="heldout_large"),
        ],
        ignore_index=True,
    )
    target_values = target_values[target_values["metric"].isin(VALIDATION_METRICS)].copy()
    target_values["metric_label"] = target_values["metric"].map(VALIDATION_METRICS)

    specs: list[RunSpec] = []
    small_runs = endpoints[["run_id", "run_name", "url", "scale", "mix", "lr", "state", "complete", "global_step"]]
    small_runs = small_runs.drop_duplicates("run_id")
    for _, row in small_runs.iterrows():
        specs.append(
            RunSpec(
                run_id=row["run_id"],
                run_name=row["run_name"],
                url=row["url"],
                scale=row["scale"],
                mix=row["mix"],
                lr=row["lr"],
                state=row["state"],
                eval_split="small_cv",
                target_kind="complete",
                complete=bool_value(row["complete"]),
                final_step=int(row["global_step"]),
            )
        )

    large_runs = targets[
        [
            "run_id",
            "run_name",
            "url",
            "scale",
            "mix",
            "lr",
            "state",
            "complete",
            "target_kind",
            "expected_steps",
            "global_step",
        ]
    ].drop_duplicates("run_id")
    for _, row in large_runs.iterrows():
        specs.append(
            RunSpec(
                run_id=row["run_id"],
                run_name=row["run_name"],
                url=row["url"],
                scale=row["scale"],
                mix=row["mix"],
                lr=row["lr"],
                state=row["state"],
                eval_split="heldout_large",
                target_kind=row["target_kind"],
                complete=bool_value(row["complete"]),
                final_step=final_step_from_expected(row["expected_steps"], row["global_step"]),
            )
        )

    return specs, target_values


def history_rows_for_run(api, project: str, spec: RunSpec) -> list[dict[str, Any]]:
    run = api.run(f"{project}/{spec.run_id}")
    metric_keys = list(VALIDATION_METRICS)
    key_sets = [
        ["global_step", "_step", *metric_keys],
        ["global_step", *metric_keys],
        ["_step", *metric_keys],
    ]
    for keys in key_sets:
        rows = list(run.scan_history(keys=keys, page_size=4000))
        if rows:
            return rows
    return []


def fetch_trajectory_points(specs: list[RunSpec], project: str, target_values: pd.DataFrame) -> pd.DataFrame:
    api = wandb.Api(timeout=60)
    target_lookup = {
        (row.run_id, row.metric): row.value
        for row in target_values[["run_id", "metric", "value"]].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        logger.info("Fetching trajectory %d/%d: %s", index, len(specs), spec.run_name)
        for raw in history_rows_for_run(api, project, spec):
            step = finite_float(raw.get("global_step", raw.get("_step")))
            if step is None:
                continue
            step_i = int(step)
            tau = step_i / max(spec.final_step, 1)
            for metric, metric_label in VALIDATION_METRICS.items():
                value = finite_float(raw.get(metric))
                if value is None:
                    continue
                final_value = finite_float(target_lookup.get((spec.run_id, metric)))
                if final_value is None:
                    continue
                rows.append(
                    {
                        "run_id": spec.run_id,
                        "run_name": spec.run_name,
                        "url": spec.url,
                        "scale": spec.scale,
                        "mix": spec.mix,
                        "lr": spec.lr,
                        "recipe": spec.recipe,
                        "state": spec.state,
                        "eval_split": spec.eval_split,
                        "target_kind": spec.target_kind,
                        "complete": spec.complete,
                        "step": step_i,
                        "final_step": spec.final_step,
                        "tau": tau,
                        "metric": metric,
                        "metric_label": metric_label,
                        "value": value,
                        "final_value": final_value,
                    }
                )
    if not rows:
        return pd.DataFrame()
    points = pd.DataFrame(rows).drop_duplicates(["run_id", "metric", "step"], keep="last")
    points = points.sort_values(["eval_split", "scale", "mix", "lr", "metric_label", "step"])
    first_values = (
        points.sort_values("step")
        .groupby(["run_id", "metric"], observed=True)["value"]
        .first()
        .rename("baseline_value")
        .reset_index()
    )
    points = points.merge(first_values, on=["run_id", "metric"], how="left")
    return points.reset_index(drop=True)


def update_trajectory_points_cache(
    specs: list[RunSpec],
    project: str,
    target_values: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    if not cache_path.exists():
        return fetch_trajectory_points(specs, project, target_values)

    cached = pd.read_csv(cache_path, dtype={"scale": str, "lr": str})
    cached_run_ids = set(cached["run_id"].dropna().astype(str))
    missing_specs = [spec for spec in specs if spec.run_id not in cached_run_ids]
    if not missing_specs:
        logger.info("Trajectory cache is already current for all %d run specs.", len(specs))
        return cached

    incomplete_missing = [spec for spec in missing_specs if not spec.complete]
    fetch_specs = [spec for spec in missing_specs if spec.complete]
    if incomplete_missing:
        logger.info("Skipping %d incomplete missing runs while refreshing trajectory cache.", len(incomplete_missing))
    if not fetch_specs:
        logger.info("No completed missing runs to fetch; reusing trajectory cache.")
        return cached

    fetched = fetch_trajectory_points(fetch_specs, project, target_values)
    if fetched.empty:
        logger.warning("Fetched no new trajectory points; reusing trajectory cache.")
        return cached

    points = pd.concat([cached, fetched], ignore_index=True)
    points = points.drop_duplicates(["run_id", "metric", "step"], keep="last")
    points = points.sort_values(["eval_split", "scale", "mix", "lr", "metric_label", "step"])
    return points.reset_index(drop=True)


def prefix_point(group: pd.DataFrame, prefix: float) -> pd.Series | None:
    sub = group[group["tau"].le(prefix)].sort_values("tau")
    if sub.empty:
        return None
    return sub.iloc[-1]


def linear_tau_prediction(group: pd.DataFrame, prefix: float) -> tuple[float, int] | None:
    sub = group[group["tau"].le(prefix)].sort_values("tau")
    if len(sub) < MIN_POINTS_FOR_LINEAR:
        return None
    x = sub["tau"].to_numpy(dtype=float)
    y = sub["value"].to_numpy(dtype=float)
    if len(np.unique(x)) < MIN_POINTS_FOR_LINEAR:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept + slope), len(sub)


def parametric_shape_values(method: str, tau: np.ndarray, shape_1: float, shape_2: float | None) -> np.ndarray | None:
    if method == "curve_log":
        shift = shape_1
        denominator = math.log((1 + shift) / shift)
        return np.log((1 + shift) / (tau + shift)) / denominator
    if method == "curve_exp":
        rate = shape_1
        denominator = 1 - math.exp(-rate)
        return (np.exp(-rate * tau) - math.exp(-rate)) / denominator
    if method == "curve_power":
        if shape_2 is None:
            return None
        shift = shape_1
        exponent = shape_2
        raw = np.power(tau + shift, -exponent)
        raw_start = shift**-exponent
        raw_end = (1 + shift) ** -exponent
        return (raw - raw_end) / (raw_start - raw_end)
    if method == "curve_rational":
        if shape_2 is None:
            return None
        t0 = shape_1
        beta = shape_2
        raw = 1 / (1 + np.power(tau / t0, beta))
        raw_end = 1 / (1 + (1 / t0) ** beta)
        return (raw - raw_end) / (1 - raw_end)
    raise ValueError(f"not a parametric method: {method}")


def huber_delta(y: np.ndarray) -> float:
    return max(0.002, HUBER_DELTA_FRACTION * float(np.max(y) - np.min(y)))


def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residuals)
    return np.where(absolute <= delta, 0.5 * np.square(residuals), delta * (absolute - 0.5 * delta))


def huber_location(values: np.ndarray, delta: float) -> float:
    lower = float(np.min(values) - delta)
    upper = float(np.max(values) + delta)
    for _ in range(20):
        midpoint = (lower + upper) / 2
        score = float(np.sum(np.clip(values - midpoint, -delta, delta)))
        if score > 0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def fit_parametric_shape(
    y: np.ndarray,
    shape_values: np.ndarray,
    fit_loss: str,
) -> tuple[float, float, float, float] | None:
    shape_range = float(np.max(shape_values) - np.min(shape_values))
    if shape_range <= 1e-8:
        return None

    centered_shape = shape_values - np.mean(shape_values)
    denominator = float(np.dot(centered_shape, centered_shape))
    candidates: list[float] = []
    if denominator > 1e-12:
        candidates.append(float(np.dot(centered_shape, y - np.mean(y)) / denominator))
    candidates.append(float((y[0] - y[-1]) / (shape_values[0] - shape_values[-1])))
    candidates.append(float((np.max(y) - np.min(y)) / shape_range))

    adjacent_slopes = []
    for left in range(len(y) - 1):
        shape_delta = shape_values[left] - shape_values[left + 1]
        if abs(shape_delta) > 1e-8:
            adjacent_slopes.append(float((y[left] - y[left + 1]) / shape_delta))
    positive_adjacent_slopes = [value for value in adjacent_slopes if math.isfinite(value) and value > 0]
    if positive_adjacent_slopes:
        candidates.append(float(np.median(positive_adjacent_slopes)))

    best: tuple[float, float, float, float] | None = None
    floor_lower = 1e-8
    floor_upper = max(float(np.min(y)) - 1e-8, floor_lower)
    delta = huber_delta(y)
    for candidate in candidates:
        if not math.isfinite(candidate) or candidate <= 0:
            continue
        for multiplier in (0.75, 1.0, 1.25):
            amplitude = candidate * multiplier
            residual_location_values = y - amplitude * shape_values
            if fit_loss == "mae":
                floor = float(np.median(residual_location_values))
            elif fit_loss == "huber":
                floor = huber_location(residual_location_values, delta)
            else:
                raise ValueError(f"unknown fit loss: {fit_loss}")
            floor = min(max(floor, floor_lower), floor_upper)
            predictions = floor + amplitude * shape_values
            prefix_fit_mae = float(np.mean(np.abs(predictions - y)))
            prefix_fit_loss = prefix_fit_mae if fit_loss == "mae" else float(np.mean(huber_loss(predictions - y, delta)))
            if best is None or prefix_fit_loss < best[3]:
                best = (floor, float(amplitude), prefix_fit_mae, prefix_fit_loss)
    return best


def parametric_model_values(base_method: str, params: np.ndarray, tau: np.ndarray) -> np.ndarray | None:
    shape_2 = float(params[3]) if len(params) > 3 else None
    shape_values = parametric_shape_values(base_method, tau, float(params[2]), shape_2)
    if shape_values is None or not np.all(np.isfinite(shape_values)):
        return None
    return params[0] + params[1] * shape_values


def parametric_residuals(base_method: str, params: np.ndarray, tau: np.ndarray, y: np.ndarray) -> np.ndarray:
    predicted = parametric_model_values(base_method, params, tau)
    if predicted is None:
        return np.full_like(y, 1e6, dtype=float)
    return predicted - y


def parametric_parameter_bounds(base_method: str, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    floor_upper = max(float(np.min(y)) - 1e-8, 1e-8)
    amplitude_upper = max(10.0, 10.0 * float(np.max(y) - np.min(y)), 2.0 * float(np.max(y)))
    lower = [1e-8, 0.0]
    upper = [floor_upper, amplitude_upper]
    for lower_shape, upper_shape in PARAMETRIC_SHAPE_BOUNDS[base_method]:
        lower.append(lower_shape)
        upper.append(upper_shape)
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def parametric_loss_value(base_method: str, params: np.ndarray, tau: np.ndarray, y: np.ndarray, fit_loss: str) -> float:
    residual = parametric_residuals(base_method, params, tau, y)
    if not np.all(np.isfinite(residual)):
        return 1e6
    if fit_loss == "mae":
        return float(np.mean(np.abs(residual)))
    if fit_loss == "huber":
        return float(np.mean(huber_loss(residual, huber_delta(y))))
    raise ValueError(f"unknown fit loss: {fit_loss}")


def scipy_parametric_fit(
    base_method: str,
    tau: np.ndarray,
    y: np.ndarray,
    fit_loss: str,
    floor: float,
    amplitude: float,
    shape_1: float,
    shape_2: float | None,
) -> tuple[np.ndarray, bool] | None:
    initial_values = [floor, amplitude, shape_1]
    if len(PARAMETRIC_SHAPE_BOUNDS[base_method]) == 2:
        if shape_2 is None:
            return None
        initial_values.append(shape_2)

    lower, upper = parametric_parameter_bounds(base_method, y)
    initial = np.clip(np.array(initial_values, dtype=float), lower + 1e-12, upper - 1e-12)
    if not np.all(np.isfinite(initial)):
        return None

    try:
        if fit_loss == "huber":
            result = least_squares(
                lambda params: parametric_residuals(base_method, params, tau, y),
                initial,
                bounds=(lower, upper),
                loss="huber",
                f_scale=huber_delta(y),
                max_nfev=SCIPY_MAX_NFEV,
            )
        elif fit_loss == "mae":
            result = minimize(
                lambda params: parametric_loss_value(base_method, params, tau, y, fit_loss),
                initial,
                method="L-BFGS-B",
                bounds=Bounds(lower, upper),
                options={"maxiter": SCIPY_MAXITER},
            )
        else:
            raise ValueError(f"unknown fit loss: {fit_loss}")
    except ValueError:
        return None

    params = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(params)):
        return None
    return params, bool(result.success)


def parametric_predictions(group: pd.DataFrame, prefix: float) -> list[dict[str, Any]]:
    sub = group[group["tau"].le(prefix)].sort_values("tau")
    if len(sub) < MIN_POINTS_FOR_PARAMETRIC:
        return []
    tau = sub["tau"].to_numpy(dtype=float)
    y = sub["value"].to_numpy(dtype=float)
    if len(np.unique(tau)) < MIN_POINTS_FOR_PARAMETRIC:
        return []

    rows: list[dict[str, Any]] = []
    for base_method, shapes in PARAMETRIC_SHAPE_GRID.items():
        for fit_loss in PARAMETRIC_FIT_LOSSES:
            best: dict[str, Any] | None = None
            for shape_1, shape_2 in shapes:
                shape_values = parametric_shape_values(base_method, tau, shape_1, shape_2)
                if shape_values is None or not np.all(np.isfinite(shape_values)):
                    continue
                fit = fit_parametric_shape(y, shape_values, fit_loss)
                if fit is None:
                    continue
                floor, amplitude, prefix_fit_mae, prefix_fit_loss = fit
                if best is None or prefix_fit_loss < best["heuristic_prefix_fit_loss"]:
                    best = {
                        "method": f"{base_method}_{fit_loss}",
                        "fit_loss": fit_loss,
                        "heuristic_floor": floor,
                        "heuristic_amplitude": amplitude,
                        "heuristic_shape_1": shape_1,
                        "heuristic_shape_2": shape_2,
                        "fit_n": len(sub),
                        "heuristic_prefix_fit_mae": prefix_fit_mae,
                        "heuristic_prefix_fit_loss": prefix_fit_loss,
                    }
            if best is not None:
                scipy_fit = scipy_parametric_fit(
                    base_method,
                    tau,
                    y,
                    fit_loss,
                    best["heuristic_floor"],
                    best["heuristic_amplitude"],
                    best["heuristic_shape_1"],
                    best["heuristic_shape_2"],
                )
                if scipy_fit is None:
                    floor = best["heuristic_floor"]
                    amplitude = best["heuristic_amplitude"]
                    shape_1 = best["heuristic_shape_1"]
                    shape_2 = best["heuristic_shape_2"]
                    prefix_fit_mae = best["heuristic_prefix_fit_mae"]
                    prefix_fit_loss = best["heuristic_prefix_fit_loss"]
                    optimizer = "heuristic_fallback"
                    optimizer_success = False
                else:
                    params, optimizer_success = scipy_fit
                    floor = float(params[0])
                    amplitude = float(params[1])
                    shape_1 = float(params[2])
                    shape_2 = float(params[3]) if len(params) > 3 else None
                    residual = parametric_residuals(base_method, params, tau, y)
                    prefix_fit_mae = float(np.mean(np.abs(residual)))
                    prefix_fit_loss = parametric_loss_value(base_method, params, tau, y, fit_loss)
                    optimizer = "scipy"

                rows.append(
                    {
                        "method": best["method"],
                        "fit_loss": fit_loss,
                        "optimizer": optimizer,
                        "optimizer_success": optimizer_success,
                        "predicted": floor,
                        "fit_n": len(sub),
                        "param_floor": floor,
                        "param_amplitude": amplitude,
                        "param_shape_1": shape_1,
                        "param_shape_2": np.nan if shape_2 is None else shape_2,
                        "prefix_fit_mae": prefix_fit_mae,
                        "prefix_fit_loss": prefix_fit_loss,
                    }
                )
    return rows


def fraction_record(group: pd.DataFrame, prefix: float) -> dict[str, Any] | None:
    row = prefix_point(group, prefix)
    if row is None:
        return None
    baseline = float(row["baseline_value"])
    final = float(row["final_value"])
    prefix_value = float(row["value"])
    final_delta = final - baseline
    prefix_delta = prefix_value - baseline
    if abs(final_delta) < 1e-8:
        return None
    fraction = prefix_delta / final_delta
    if not math.isfinite(fraction) or fraction <= 0:
        return None
    return {
        "run_id": row["run_id"],
        "scale": row["scale"],
        "mix": row["mix"],
        "lr": row["lr"],
        "recipe": row["recipe"],
        "metric": row["metric"],
        "metric_label": row["metric_label"],
        "prefix": prefix,
        "prefix_actual_tau": float(row["tau"]),
        "fraction": fraction,
    }


def template_group_columns(method: str) -> list[str]:
    if method == "template_global":
        return ["metric_label", "prefix"]
    if method == "template_by_mix":
        return ["metric_label", "prefix", "mix"]
    if method == "template_by_recipe":
        return ["metric_label", "prefix", "mix", "lr"]
    raise ValueError(f"not a template method: {method}")


def template_key(row: pd.Series, method: str) -> tuple[Any, ...]:
    return tuple(row[column] for column in template_group_columns(method))


def build_template_index(fraction_table: pd.DataFrame) -> dict[str, dict[tuple[Any, ...], list[tuple[str, float]]]]:
    index: dict[str, dict[tuple[Any, ...], list[tuple[str, float]]]] = {}
    for method in [item for item in METHODS if item.startswith("template_")]:
        method_index: dict[tuple[Any, ...], list[tuple[str, float]]] = {}
        for key, group in fraction_table.groupby(template_group_columns(method), observed=True, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            method_index[key] = [
                (str(row.run_id), float(row.fraction)) for row in group[["run_id", "fraction"]].itertuples(index=False)
            ]
        index[method] = method_index
    return index


def template_fraction(
    template_index: dict[str, dict[tuple[Any, ...], list[tuple[str, float]]]],
    target_row: pd.Series,
    method: str,
    *,
    exclude_run_id: str | None,
) -> tuple[float, int] | None:
    candidates = template_index.get(method, {}).get(template_key(target_row, method), [])
    if exclude_run_id is not None:
        candidates = [(run_id, fraction) for run_id, fraction in candidates if run_id != exclude_run_id]
    if len(candidates) < MIN_TEMPLATE_RUNS:
        return None
    fraction = float(np.median([fraction for _, fraction in candidates]))
    if not math.isfinite(fraction) or fraction <= 0:
        return None
    return fraction, len(candidates)


def prediction_rows_for_run(
    group: pd.DataFrame,
    prefix: float,
    template_index: dict[str, dict[tuple[Any, ...], list[tuple[str, float]]]],
) -> list[dict[str, Any]]:
    prefix_row = prefix_point(group, prefix)
    if prefix_row is None:
        return []

    baseline = float(prefix_row["baseline_value"])
    prefix_value = float(prefix_row["value"])
    target = float(prefix_row["final_value"])
    base = {
        "run_id": prefix_row["run_id"],
        "run_name": prefix_row["run_name"],
        "url": prefix_row["url"],
        "scale": prefix_row["scale"],
        "mix": prefix_row["mix"],
        "lr": prefix_row["lr"],
        "recipe": prefix_row["recipe"],
        "eval_split": prefix_row["eval_split"],
        "target_kind": prefix_row["target_kind"],
        "complete": prefix_row["complete"],
        "metric": prefix_row["metric"],
        "metric_label": prefix_row["metric_label"],
        "prefix": prefix,
        "prefix_actual_tau": float(prefix_row["tau"]),
        "prefix_step": int(prefix_row["step"]),
        "final_step": int(prefix_row["final_step"]),
        "baseline_value": baseline,
        "prefix_value": prefix_value,
        "target": target,
    }
    rows: list[dict[str, Any]] = []

    def add_prediction(method: str, predicted: float, fit_n: int, extra: dict[str, Any] | None = None) -> None:
        error = predicted - target
        rows.append(
            {
                **base,
                "method": method,
                "predicted": predicted,
                "error": error,
                "abs_error": abs(error),
                "pct_error": 100 * error / target,
                "fit_n": fit_n,
                **(extra or {}),
            }
        )

    add_prediction("last_value", prefix_value, 1)

    linear_prediction = linear_tau_prediction(group, prefix)
    if linear_prediction is not None:
        predicted, fit_n = linear_prediction
        add_prediction("linear_tau", predicted, fit_n)

    if prefix_row["metric_label"] in PARAMETRIC_METRIC_LABELS:
        for prediction in parametric_predictions(group, prefix):
            add_prediction(
                prediction["method"],
                prediction["predicted"],
                prediction["fit_n"],
                {
                    "param_floor": prediction["param_floor"],
                    "param_amplitude": prediction["param_amplitude"],
                    "param_shape_1": prediction["param_shape_1"],
                    "param_shape_2": prediction["param_shape_2"],
                    "prefix_fit_mae": prediction["prefix_fit_mae"],
                    "prefix_fit_loss": prediction["prefix_fit_loss"],
                    "fit_loss": prediction["fit_loss"],
                    "optimizer": prediction["optimizer"],
                    "optimizer_success": prediction["optimizer_success"],
                },
            )

    template_target = pd.Series(base)
    exclude_run_id = str(prefix_row["run_id"]) if prefix_row["eval_split"] == "small_cv" else None
    for method in [m for m in METHODS if m.startswith("template_")]:
        fitted_fraction = template_fraction(template_index, template_target, method, exclude_run_id=exclude_run_id)
        if fitted_fraction is None:
            continue
        fraction, fit_n = fitted_fraction
        predicted = baseline + (prefix_value - baseline) / fraction
        add_prediction(method, predicted, fit_n)

    return rows


def make_fraction_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    small = points[points["eval_split"].eq("small_cv") & points["complete"]].copy()
    for _, group in small.groupby(["run_id", "metric"], observed=True, sort=False):
        for prefix in PREFIX_FRACS:
            record = fraction_record(group, prefix)
            if record is not None:
                rows.append(record)
    return pd.DataFrame(rows)


def predict_prefixes(points: pd.DataFrame) -> pd.DataFrame:
    fraction_table = make_fraction_table(points)
    template_index = build_template_index(fraction_table)
    rows: list[dict[str, Any]] = []
    for _, group in points.groupby(["run_id", "metric"], observed=True, sort=False):
        for prefix in PREFIX_FRACS:
            rows.extend(prediction_rows_for_run(group.sort_values("tau"), prefix, template_index))
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return predictions
    return predictions.sort_values(["eval_split", "metric_label", "prefix", "method", "scale", "mix", "lr"])


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    complete_predictions = predictions[predictions["complete"]].copy()
    if complete_predictions.empty:
        return pd.DataFrame()
    grouped = complete_predictions.groupby(
        ["eval_split", "target_kind", "metric_label", "method", "prefix"],
        observed=True,
    )
    summary = grouped.agg(
        n=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        max_abs_error=("abs_error", "max"),
        mean_abs_pct_error=("pct_error", lambda values: values.abs().mean()),
        median_prefix_actual_tau=("prefix_actual_tau", "median"),
    ).reset_index()
    return summary.sort_values(["metric_label", "eval_split", "prefix", "mean_abs_error"])


def select_methods(summary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    small = summary[summary["eval_split"].eq("small_cv") & summary["target_kind"].eq("complete")].copy()
    for metric_label, group in small.groupby("metric_label", observed=True):
        best_mae = float(group["mean_abs_error"].min())
        tolerance = best_mae * (1 + SELECTION_REL_TOLERANCE) + SELECTION_ABS_TOLERANCE
        eligible = group[group["mean_abs_error"].le(tolerance)].sort_values(["prefix", "mean_abs_error"])
        if eligible.empty:
            eligible = group.sort_values(["mean_abs_error", "prefix"])
        selected = eligible.iloc[0]
        heldout = predictions[
            predictions["eval_split"].eq("heldout_large")
            & predictions["complete"]
            & predictions["metric_label"].eq(metric_label)
            & predictions["method"].eq(selected["method"])
            & predictions["prefix"].eq(selected["prefix"])
        ]
        rows.append(
            {
                "metric_label": metric_label,
                "selected_method": selected["method"],
                "selected_prefix": selected["prefix"],
                "small_cv_mean_abs_error": selected["mean_abs_error"],
                "small_cv_n": int(selected["n"]),
                "small_cv_best_mean_abs_error": best_mae,
                "selection_tolerance": tolerance,
                "heldout_complete_n": len(heldout),
                "heldout_complete_mean_abs_error": float(heldout["abs_error"].mean()) if not heldout.empty else np.nan,
                "heldout_complete_median_abs_error": (
                    float(heldout["abs_error"].median()) if not heldout.empty else np.nan
                ),
                "heldout_complete_mean_abs_pct_error": (
                    float(heldout["pct_error"].abs().mean()) if not heldout.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("metric_label")


def write_markdown_summary(summary: pd.DataFrame, selection: pd.DataFrame) -> None:
    lines = [
        "# Within-Run Validation Prediction",
        "",
        "Train/tune split: methods are tuned on clean small-ladder runs through `3e20`.",
        "`1e21` and `1e22` are held out for generalization checks.",
        "",
        "Methods:",
        "",
        "- `last_value`: carry forward the last validation point in the prefix.",
        "- `linear_tau`: fit a line in normalized training progress `tau` and evaluate at `tau=1`.",
        "- `template_*`: learn the median fraction of final improvement "
        "achieved by the prefix on small runs, then apply that fraction to "
        "the target run's observed prefix improvement.",
        "- `curve_*_mae` and `curve_*_huber`: fit a bounded monotone-decay "
        "curve to the observed prefix and read its `tau=1` floor as the "
        "final-loss prediction. The fixed shape grid supplies initial values; "
        "SciPy then optimizes the endpoint floor, amplitude, and shape "
        "parameters under the named prefix fit loss. Final evaluation is still MAE.",
        "",
    ]
    if not selection.empty:
        lines.extend(
            [
                "## Selected Small-Scale Recipes",
                "",
                "| metric | method | prefix | small MAE | held-out MAE | held-out MAPE |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in selection.itertuples(index=False):
            lines.append(
                "| "
                f"{row.metric_label} | {row.selected_method} | {row.selected_prefix:.2f} | "
                f"{row.small_cv_mean_abs_error:.5f} | {row.heldout_complete_mean_abs_error:.5f} | "
                f"{row.heldout_complete_mean_abs_pct_error:.2f}% |"
            )
    if not summary.empty:
        math = summary[
            summary["metric_label"].eq("math_val_loss")
            & summary["eval_split"].eq("small_cv")
            & summary["target_kind"].eq("complete")
        ]
        math = math.sort_values(["prefix", "mean_abs_error"])
        lines.extend(["", "## Math Small-CV Leaderboard", ""])
        lines.extend(
            [
                "| prefix | method | n | mean abs error | median abs error |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in math.head(30).itertuples(index=False):
            lines.append(
                f"| {row.prefix:.2f} | {row.method} | {row.n} | "
                f"{row.mean_abs_error:.5f} | {row.median_abs_error:.5f} |"
            )
    (OUT_DIR / "trajectory_prediction_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs, target_values = load_run_specs_and_targets()

    if args.use_cache and TRAJECTORY_POINTS_PATH.exists():
        points = pd.read_csv(TRAJECTORY_POINTS_PATH, dtype={"scale": str, "lr": str})
    else:
        points = update_trajectory_points_cache(specs, args.project, target_values, TRAJECTORY_POINTS_PATH)
        points.to_csv(TRAJECTORY_POINTS_PATH, index=False)

    predictions = predict_prefixes(points)
    predictions.to_csv(TRAJECTORY_PREDICTIONS_PATH, index=False)
    summary = summarize_predictions(predictions)
    summary.to_csv(TRAJECTORY_SUMMARY_PATH, index=False)
    selection = select_methods(summary, predictions)
    selection.to_csv(TRAJECTORY_SELECTION_PATH, index=False)
    write_markdown_summary(summary, selection)

    print("Trajectory coverage:")
    coverage = points[["eval_split", "scale", "mix", "lr", "complete"]].drop_duplicates()
    print(coverage.groupby(["eval_split", "scale"], observed=True)["complete"].agg(["sum", "count"]).to_string())
    print("\nSelected methods:")
    print(selection.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'trajectory_prediction_summary.md'}")


if __name__ == "__main__":
    main()
