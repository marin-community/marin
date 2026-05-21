# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predict final validation loss from early prefixes of Delphi midtraining runs.

This is the within-run companion to ``delphi_small_final_loss_scaling.py``.
It caches validation trajectories, tunes prefix-prediction rules on the clean
small ladder through ``2e20``, and evaluates the same rules on held-out
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
from typing import Any

import numpy as np
import pandas as pd
import wandb
from delphi_small_final_loss_scaling import (
    OUT_DIR,
    PROJECT,
)

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
PREFIX_FRACS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
METHODS = (
    "last_value",
    "linear_tau",
    "template_global",
    "template_by_mix",
    "template_by_recipe",
)
MIN_POINTS_FOR_LINEAR = 2
MIN_TEMPLATE_RUNS = 3
SELECTION_REL_TOLERANCE = 0.10
SELECTION_ABS_TOLERANCE = 0.002


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


def template_fraction(
    fraction_table: pd.DataFrame,
    target_row: pd.Series,
    method: str,
    *,
    exclude_run_id: str | None,
) -> tuple[float, int] | None:
    group_columns = template_group_columns(method)
    candidates = fraction_table.copy()
    if exclude_run_id is not None:
        candidates = candidates[~candidates["run_id"].eq(exclude_run_id)]
    for column in group_columns:
        candidates = candidates[candidates[column].eq(target_row[column])]
    if len(candidates) < MIN_TEMPLATE_RUNS:
        return None
    fraction = float(candidates["fraction"].median())
    if not math.isfinite(fraction) or fraction <= 0:
        return None
    return fraction, len(candidates)


def prediction_rows_for_run(
    group: pd.DataFrame,
    prefix: float,
    fraction_table: pd.DataFrame,
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

    def add_prediction(method: str, predicted: float, fit_n: int) -> None:
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
            }
        )

    add_prediction("last_value", prefix_value, 1)

    linear_prediction = linear_tau_prediction(group, prefix)
    if linear_prediction is not None:
        predicted, fit_n = linear_prediction
        add_prediction("linear_tau", predicted, fit_n)

    template_target = pd.Series(base)
    exclude_run_id = str(prefix_row["run_id"]) if prefix_row["eval_split"] == "small_cv" else None
    for method in [m for m in METHODS if m.startswith("template_")]:
        fitted_fraction = template_fraction(fraction_table, template_target, method, exclude_run_id=exclude_run_id)
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
    rows: list[dict[str, Any]] = []
    for _, group in points.groupby(["run_id", "metric"], observed=True, sort=False):
        for prefix in PREFIX_FRACS:
            rows.extend(prediction_rows_for_run(group.sort_values("tau"), prefix, fraction_table))
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
        "Train/tune split: methods are tuned on clean small-ladder runs through `2e20`.",
        "`1e21` and `1e22` are held out for generalization checks.",
        "",
        "Methods:",
        "",
        "- `last_value`: carry forward the last validation point in the prefix.",
        "- `linear_tau`: fit a line in normalized training progress `tau` and evaluate at `tau=1`.",
        "- `template_*`: learn the median fraction of final improvement "
        "achieved by the prefix on small runs, then apply that fraction to "
        "the target run's observed prefix improvement.",
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
        points = fetch_trajectory_points(specs, args.project, target_values)
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
