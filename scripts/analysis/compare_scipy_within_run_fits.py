# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare current grid trajectory fits against SciPy-optimized fits.

This is an experimental sidecar: it does not replace the current notebook or
report outputs. It reads the cached math validation trajectory predictions,
starts from the current grid-fit parameters, and lets scipy.optimize refine the
same parametric forms.

Run:
    uv run python scripts/analysis/compare_scipy_within_run_fits.py --prefix-min 0.50
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from delphi_within_run_prediction import (
    MIN_POINTS_FOR_PARAMETRIC,
    OUT_DIR,
    TRAJECTORY_POINTS_PATH,
    TRAJECTORY_PREDICTIONS_PATH,
    huber_delta,
    huber_loss,
    parametric_shape_values,
)
from scipy.optimize import Bounds, least_squares, minimize

logger = logging.getLogger("compare_scipy_within_run_fits")

OUTPUT_PREDICTIONS_PATH = OUT_DIR / "trajectory_prefix_predictions_scipy_compare.csv"
OUTPUT_SUMMARY_PATH = OUT_DIR / "trajectory_prefix_summary_scipy_compare.csv"
METRIC_LABEL = "math_val_loss"
SHAPE_BOUNDS: dict[str, tuple[tuple[float, float], ...]] = {
    "curve_log": ((1e-4, 10.0),),
    "curve_exp": ((0.01, 50.0),),
    "curve_power": ((1e-4, 10.0), (0.05, 10.0)),
    "curve_rational": ((0.01, 10.0), (0.10, 10.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=TRAJECTORY_POINTS_PATH)
    parser.add_argument("--predictions", type=Path, default=TRAJECTORY_PREDICTIONS_PATH)
    parser.add_argument("--prefix-min", type=float, default=0.05)
    parser.add_argument("--prefix-max", type=float, default=0.90)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--max-nfev", type=int, default=200)
    return parser.parse_args()


def method_parts(method: str) -> tuple[str, str]:
    if method.endswith("_mae"):
        return method.removesuffix("_mae"), "mae"
    if method.endswith("_huber"):
        return method.removesuffix("_huber"), "huber"
    raise ValueError(f"unsupported method: {method}")


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def initial_parameters(row: pd.Series) -> np.ndarray | None:
    base_method, _ = method_parts(str(row["method"]))
    floor = finite_float(row.get("param_floor"))
    amplitude = finite_float(row.get("param_amplitude"))
    shape_1 = finite_float(row.get("param_shape_1"))
    shape_2 = finite_float(row.get("param_shape_2"))
    if floor is None or amplitude is None or shape_1 is None:
        return None
    if len(SHAPE_BOUNDS[base_method]) == 1:
        return np.array([floor, amplitude, shape_1], dtype=float)
    if shape_2 is None:
        return None
    return np.array([floor, amplitude, shape_1, shape_2], dtype=float)


def parameter_bounds(base_method: str, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    floor_upper = max(float(np.min(y)) - 1e-8, 1e-8)
    amplitude_upper = max(10.0, 10.0 * float(np.max(y) - np.min(y)), 2.0 * float(np.max(y)))
    lower = [1e-8, 0.0]
    upper = [floor_upper, amplitude_upper]
    for lo, hi in SHAPE_BOUNDS[base_method]:
        lower.append(lo)
        upper.append(hi)
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def model_values(base_method: str, params: np.ndarray, tau: np.ndarray) -> np.ndarray | None:
    floor = params[0]
    amplitude = params[1]
    shape_1 = params[2]
    shape_2 = float(params[3]) if len(params) > 3 else None
    shape = parametric_shape_values(base_method, tau, float(shape_1), shape_2)
    if shape is None or not np.all(np.isfinite(shape)):
        return None
    return floor + amplitude * shape


def residuals(base_method: str, params: np.ndarray, tau: np.ndarray, y: np.ndarray) -> np.ndarray:
    predicted = model_values(base_method, params, tau)
    if predicted is None:
        return np.full_like(y, 1e6, dtype=float)
    return predicted - y


def mae_objective(base_method: str, tau: np.ndarray, y: np.ndarray):
    def objective(params: np.ndarray) -> float:
        values = residuals(base_method, params, tau, y)
        if not np.all(np.isfinite(values)):
            return 1e6
        return float(np.mean(np.abs(values)))

    return objective


def optimize_row(row: pd.Series, group: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any] | None:
    prefix = float(row["prefix"])
    sub = group[group["tau"].le(prefix)].sort_values("tau")
    if len(sub) < MIN_POINTS_FOR_PARAMETRIC:
        return None
    tau = sub["tau"].to_numpy(dtype=float)
    y = sub["value"].to_numpy(dtype=float)
    if len(np.unique(tau)) < MIN_POINTS_FOR_PARAMETRIC:
        return None

    base_method, fit_loss = method_parts(str(row["method"]))
    initial = initial_parameters(row)
    if initial is None:
        return None

    lower, upper = parameter_bounds(base_method, y)
    clipped_initial = np.clip(initial, lower + 1e-12, upper - 1e-12)
    if not np.all(np.isfinite(clipped_initial)):
        return None

    success = False
    message = ""
    try:
        if fit_loss == "huber":
            delta = huber_delta(y)
            result = least_squares(
                lambda params: residuals(base_method, params, tau, y),
                clipped_initial,
                bounds=(lower, upper),
                loss="huber",
                f_scale=delta,
                max_nfev=args.max_nfev,
            )
            params = result.x
            success = bool(result.success)
            message = str(result.message)
            prefix_fit_loss = float(np.mean(huber_loss(residuals(base_method, params, tau, y), delta)))
        else:
            result = minimize(
                mae_objective(base_method, tau, y),
                clipped_initial,
                method="Powell",
                bounds=Bounds(lower, upper),
                options={"maxiter": args.maxiter, "xtol": 1e-8, "ftol": 1e-8},
            )
            params = np.asarray(result.x, dtype=float)
            success = bool(result.success)
            message = str(result.message)
            prefix_fit_loss = float(mae_objective(base_method, tau, y)(params))
    except ValueError as error:
        return {"success": False, "message": str(error)}

    prefix_residuals = residuals(base_method, params, tau, y)
    if not np.all(np.isfinite(prefix_residuals)):
        return None
    scipy_predicted = float(params[0])
    target = float(row["target"])
    scipy_error = scipy_predicted - target
    current_abs_error = float(row["abs_error"])
    scipy_abs_error = abs(scipy_error)
    return {
        "success": success,
        "message": message,
        "scipy_predicted": scipy_predicted,
        "scipy_error": scipy_error,
        "scipy_abs_error": scipy_abs_error,
        "current_predicted": float(row["predicted"]),
        "current_error": float(row["error"]),
        "current_abs_error": current_abs_error,
        "delta_abs_error": scipy_abs_error - current_abs_error,
        "scipy_prefix_fit_mae": float(np.mean(np.abs(prefix_residuals))),
        "scipy_prefix_fit_loss": prefix_fit_loss,
        "scipy_param_floor": float(params[0]),
        "scipy_param_amplitude": float(params[1]),
        "scipy_param_shape_1": float(params[2]),
        "scipy_param_shape_2": float(params[3]) if len(params) > 3 else np.nan,
        "fit_n": len(sub),
    }


def comparison_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    completed = comparison[comparison["complete"]].copy()
    if completed.empty:
        return pd.DataFrame()
    grouped = completed.groupby(["eval_split", "method", "prefix"], observed=True)
    summary = grouped.agg(
        n=("scipy_abs_error", "size"),
        scipy_success_rate=("scipy_success", "mean"),
        current_mean_abs_error=("current_abs_error", "mean"),
        scipy_mean_abs_error=("scipy_abs_error", "mean"),
        current_max_abs_error=("current_abs_error", "max"),
        scipy_max_abs_error=("scipy_abs_error", "max"),
        scipy_better_fraction=("delta_abs_error", lambda values: float((values < 0).mean())),
        mean_delta_abs_error=("delta_abs_error", "mean"),
        median_delta_abs_error=("delta_abs_error", "median"),
    ).reset_index()
    summary["mean_abs_error_improvement"] = summary["current_mean_abs_error"] - summary["scipy_mean_abs_error"]
    summary["max_abs_error_improvement"] = summary["current_max_abs_error"] - summary["scipy_max_abs_error"]
    return summary.sort_values(["eval_split", "prefix", "scipy_mean_abs_error", "method"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    points = pd.read_csv(args.points, dtype={"scale": str, "lr": str})
    predictions = pd.read_csv(args.predictions, dtype={"scale": str, "lr": str}, low_memory=False)
    predictions = predictions[
        predictions["metric_label"].eq(METRIC_LABEL)
        & predictions["method"].astype(str).str.startswith("curve_")
        & predictions["prefix"].ge(args.prefix_min)
        & predictions["prefix"].le(args.prefix_max)
    ].copy()
    points = points[points["metric_label"].eq(METRIC_LABEL)].copy()
    point_groups = {run_id: group for run_id, group in points.groupby("run_id", observed=True, sort=False)}

    rows: list[dict[str, Any]] = []
    total = len(predictions)
    for index, row in enumerate(predictions.itertuples(index=False), start=1):
        if index == 1 or index % 500 == 0:
            logger.info("Optimizing scipy fit %d/%d", index, total)
        row_series = pd.Series(row._asdict())
        group = point_groups.get(row_series["run_id"])
        if group is None:
            continue
        fit = optimize_row(row_series, group, args)
        if fit is None:
            continue
        rows.append(
            {
                "run_id": row_series["run_id"],
                "run_name": row_series["run_name"],
                "scale": row_series["scale"],
                "mix": row_series["mix"],
                "lr": row_series["lr"],
                "recipe": row_series["recipe"],
                "eval_split": row_series["eval_split"],
                "target_kind": row_series["target_kind"],
                "complete": bool(row_series["complete"]),
                "prefix": float(row_series["prefix"]),
                "method": row_series["method"],
                "target": float(row_series["target"]),
                "scipy_success": bool(fit.get("success", False)),
                "scipy_message": fit.get("message", ""),
                **{key: value for key, value in fit.items() if key not in {"success", "message"}},
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    summary = comparison_summary(comparison)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print(f"Wrote {OUTPUT_PREDICTIONS_PATH}")
    print(f"Wrote {OUTPUT_SUMMARY_PATH}")
    if not summary.empty:
        for split in ["small_cv", "heldout_large"]:
            split_summary = summary[summary["eval_split"].eq(split)].copy()
            if split_summary.empty:
                continue
            best = split_summary.sort_values("scipy_mean_abs_error").head(10)
            print(f"\nBest SciPy rows for {split}:")
            print(
                best[
                    [
                        "prefix",
                        "method",
                        "n",
                        "current_mean_abs_error",
                        "scipy_mean_abs_error",
                        "mean_abs_error_improvement",
                        "current_max_abs_error",
                        "scipy_max_abs_error",
                    ]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
