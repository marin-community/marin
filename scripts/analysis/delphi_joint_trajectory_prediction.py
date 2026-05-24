# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit joint trajectory regressions over tau, flops, mix, and LR.

This complements ``delphi_within_run_prediction.py``. The within-run script
fits one trajectory curve per run/cell. This script asks whether a shared
trajectory model across cells helps endpoint prediction.

Two scopes are emitted:

- ``global``: one model per prefix/form using all completed runs.
- ``by_flop``: one model per prefix/form/scale using runs at that scale.

For a prefix p, the model is fit only on validation points with tau <= p, then
evaluated at tau=1 for each completed run in the scope.

Run:
    uv run python scripts/analysis/delphi_joint_trajectory_prediction.py
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from delphi_small_final_loss_scaling import ALL_SCALE_FLOPS, OUT_DIR
from delphi_within_run_prediction import PREFIX_FRACS, TRAJECTORY_POINTS_PATH, huber_delta
from scipy.optimize import least_squares

logger = logging.getLogger("delphi_joint_trajectory_prediction")

OUTPUT_PREDICTIONS_PATH = OUT_DIR / "trajectory_joint_prefix_predictions.csv"
OUTPUT_SUMMARY_PATH = OUT_DIR / "trajectory_joint_prefix_summary.csv"
OUTPUT_MODELS_PATH = OUT_DIR / "trajectory_joint_prefix_models.csv"
METRIC_LABEL = "math_val_loss"
MAX_NFEV = 500
RIDGE_PENALTY = 1e-4


@dataclass(frozen=True)
class JointForm:
    name: str
    label: str
    initial_theta: tuple[float, ...]
    lower_theta: tuple[float, ...]
    upper_theta: tuple[float, ...]


JOINT_FORMS = (
    JointForm(
        name="exp_drift",
        label="exp + drift",
        initial_theta=(math.log(3.0),),
        lower_theta=(math.log(0.05),),
        upper_theta=(math.log(50.0),),
    ),
    JointForm(
        name="power_drift",
        label="power + drift",
        initial_theta=(math.log(0.10), math.log(1.0)),
        lower_theta=(math.log(1e-3), math.log(0.05)),
        upper_theta=(math.log(10.0), math.log(10.0)),
    ),
    JointForm(
        name="gompertz_shoulder",
        label="Gompertz shoulder",
        initial_theta=(math.log(1.0), math.log(3.0)),
        lower_theta=(math.log(0.05), math.log(0.05)),
        upper_theta=(math.log(10.0), math.log(50.0)),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=TRAJECTORY_POINTS_PATH)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    return parser.parse_args()


def mix_mid_fraction(mix: str) -> float:
    if mix == "p33m67":
        return 0.67
    if mix == "p50m50":
        return 0.50
    if mix == "p67m33":
        return 0.33
    raise ValueError(f"unknown mix: {mix}")


def static_features(frame: pd.DataFrame, *, include_flops: bool) -> np.ndarray:
    mid = frame["mix"].map(mix_mid_fraction).to_numpy(dtype=float) - 0.50
    lr = frame["lr"].astype(float).to_numpy(dtype=float) / 100 - 0.58
    columns = [np.ones(len(frame)), mid, lr, mid * lr]
    if include_flops:
        log_flops = np.log10(frame["scale"].map(ALL_SCALE_FLOPS).to_numpy(dtype=float))
        log_flops = log_flops - np.log10(3e20)
        columns.extend([log_flops, log_flops * mid, log_flops * lr])
    return np.column_stack(columns)


def trajectory_bases(tau: np.ndarray, form: JointForm, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if form.name == "exp_drift":
        rate = math.exp(float(theta[0]))
        return np.exp(-rate * tau), tau
    if form.name == "power_drift":
        shift = math.exp(float(theta[0]))
        exponent = math.exp(float(theta[1]))
        return np.power(tau + shift, -exponent), tau
    if form.name == "gompertz_shoulder":
        amplitude = math.exp(float(theta[0]))
        rate = math.exp(float(theta[1]))
        return np.exp(-amplitude * np.exp(-rate * tau)), tau
    raise ValueError(f"unknown form: {form.name}")


def design_matrix(frame: pd.DataFrame, form: JointForm, theta: np.ndarray, *, include_flops: bool) -> np.ndarray:
    z = static_features(frame, include_flops=include_flops)
    basis_1, basis_2 = trajectory_bases(frame["tau"].to_numpy(dtype=float), form, theta)
    return np.concatenate([z, z * basis_1[:, None], z * basis_2[:, None]], axis=1)


def ridge_initial_beta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    penalty = RIDGE_PENALTY * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def fit_joint_model(
    train: pd.DataFrame,
    form: JointForm,
    *,
    include_flops: bool,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    y = train["value"].to_numpy(dtype=float)
    if len(train) < 30:
        return None

    theta0 = np.array(form.initial_theta, dtype=float)
    x0 = design_matrix(train, form, theta0, include_flops=include_flops)
    beta0 = ridge_initial_beta(x0, y)
    params0 = np.concatenate([beta0, theta0])
    beta_bounds = np.full(len(beta0), np.inf)
    lower = np.concatenate([-beta_bounds, np.array(form.lower_theta, dtype=float)])
    upper = np.concatenate([beta_bounds, np.array(form.upper_theta, dtype=float)])
    delta = huber_delta(y)

    def residuals(params: np.ndarray) -> np.ndarray:
        beta = params[: len(beta0)]
        theta = params[len(beta0) :]
        x = design_matrix(train, form, theta, include_flops=include_flops)
        data_residual = x @ beta - y
        ridge_residual = math.sqrt(RIDGE_PENALTY) * beta
        return np.concatenate([data_residual, ridge_residual])

    result = least_squares(
        residuals,
        params0,
        bounds=(lower, upper),
        loss="huber",
        f_scale=delta,
        max_nfev=max_nfev,
    )
    beta = result.x[: len(beta0)]
    theta = result.x[len(beta0) :]
    return beta, theta, bool(result.success)


def predict_endpoint(
    targets: pd.DataFrame,
    form: JointForm,
    beta: np.ndarray,
    theta: np.ndarray,
    *,
    include_flops: bool,
) -> np.ndarray:
    endpoint = targets.copy()
    endpoint["tau"] = 1.0
    x = design_matrix(endpoint, form, theta, include_flops=include_flops)
    return x @ beta


def completed_run_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in points[points["complete"]].groupby("run_id", observed=True, sort=False):
        final_row = group.sort_values("tau").iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "run_name": final_row["run_name"],
                "scale": final_row["scale"],
                "mix": final_row["mix"],
                "lr": final_row["lr"],
                "recipe": final_row["recipe"],
                "eval_split": final_row["eval_split"],
                "complete": bool(final_row["complete"]),
                "target": float(final_row["final_value"]),
            }
        )
    return pd.DataFrame(rows)


def prediction_rows_for_scope(
    train_pool: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    scope: str,
    prefix: float,
    include_flops: bool,
    max_nfev: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = train_pool[train_pool["tau"].le(prefix)].copy()
    rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    model_scale = "all" if include_flops else str(targets["scale"].iloc[0])
    for form in JOINT_FORMS:
        fit = fit_joint_model(train, form, include_flops=include_flops, max_nfev=max_nfev)
        if fit is None:
            continue
        beta, theta, success = fit
        predicted = predict_endpoint(targets, form, beta, theta, include_flops=include_flops)
        method = f"joint_{scope}_{form.name}"
        method_label = f"{scope.replace('_', '-')} {form.label}"
        model_rows.append(
            {
                "scale": model_scale,
                "prefix": prefix,
                "scope": scope,
                "form": form.name,
                "method": method,
                "method_label": method_label,
                "include_flops": include_flops,
                "beta_json": list(map(float, beta)),
                "theta_json": list(map(float, theta)),
                "fit_n": len(train),
                "optimizer": "scipy_huber",
                "optimizer_success": success,
            }
        )
        for row, value in zip(targets.itertuples(index=False), predicted, strict=True):
            error = float(value) - float(row.target)
            rows.append(
                {
                    "run_id": row.run_id,
                    "run_name": row.run_name,
                    "scale": row.scale,
                    "mix": row.mix,
                    "lr": row.lr,
                    "recipe": row.recipe,
                    "eval_split": row.eval_split,
                    "complete": row.complete,
                    "prefix": prefix,
                    "scope": scope,
                    "form": form.name,
                    "method": method,
                    "method_label": method_label,
                    "target": float(row.target),
                    "predicted": float(value),
                    "error": error,
                    "abs_error": abs(error),
                    "fit_n": len(train),
                    "optimizer": "scipy_huber",
                    "optimizer_success": success,
                }
            )
    return rows, model_rows


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    grouped = predictions.groupby(["scope", "form", "method", "method_label", "prefix"], observed=True)
    return (
        grouped.agg(
            n=("run_id", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
            optimizer_success_rate=("optimizer_success", "mean"),
        )
        .reset_index()
        .sort_values(["prefix", "mean_abs_error", "method"])
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    points = pd.read_csv(args.points, dtype={"scale": str, "lr": str}, low_memory=False)
    points = points[points["metric_label"].eq(METRIC_LABEL) & points["complete"]].copy()
    targets = completed_run_table(points)
    rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for prefix in PREFIX_FRACS:
        logger.info("Fitting global joint forms at prefix %.2f", prefix)
        new_rows, new_model_rows = prediction_rows_for_scope(
            points,
            targets,
            scope="global",
            prefix=prefix,
            include_flops=True,
            max_nfev=args.max_nfev,
        )
        rows.extend(new_rows)
        model_rows.extend(new_model_rows)

        for scale, scale_points in points.groupby("scale", observed=True, sort=False):
            scale_targets = targets[targets["scale"].eq(scale)].copy()
            logger.info("Fitting by-flop joint forms at prefix %.2f scale %s", prefix, scale)
            new_rows, new_model_rows = prediction_rows_for_scope(
                scale_points,
                scale_targets,
                scope="by_flop",
                prefix=prefix,
                include_flops=False,
                max_nfev=args.max_nfev,
            )
            rows.extend(new_rows)
            model_rows.extend(new_model_rows)

    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    models = pd.DataFrame(model_rows)
    models.to_csv(OUTPUT_MODELS_PATH, index=False)
    summary = summarize(predictions)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print(f"Wrote {OUTPUT_PREDICTIONS_PATH}")
    print(f"Wrote {OUTPUT_MODELS_PATH}")
    print(f"Wrote {OUTPUT_SUMMARY_PATH}")
    if not summary.empty:
        print(summary.sort_values(["mean_abs_error", "max_abs_error"]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
