# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Codex retrospective checks for Delphi midtraining 1e22 prediction.

This script is intentionally separate from the main interactive report. It reads
the cached endpoint and trajectory exports, then writes independent claim-audit
tables, prediction experiments, and plots under:

    midtrain_analysis_outputs/codex_midtrain_prediction_retro/
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path("midtrain_analysis_outputs/small_final_loss_scaling")
CONFIG_DIR = Path("midtrain_wandb_data/runs")
OUT_DIR = Path("midtrain_analysis_outputs/codex_midtrain_prediction_retro")

MATH_METRIC = "math_val_loss"
RAW_MATH_METRIC = "eval/nemotron_cc_math_v1/4plus/loss"
SCALE_ORDER = ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20", "1e21", "1e22"]
SMALL_SCALES = SCALE_ORDER[:7]
HELDOUT_SCALES = ["1e21", "1e22"]
MIX_ORDER = ["p67m33", "p50m50", "p33m67"]
LR_ORDER = ["33", "50", "67", "83"]
MIN_POINTS_FOR_FIT = 3

ALL_SCALE_FLOPS = {
    "3e18": 3e18,
    "9e18": 9e18,
    "2e19": 2e19,
    "3e19": 3e19,
    "9e19": 9e19,
    "2e20": 2e20,
    "3e20": 3e20,
    "1e21": 1e21,
    "1e22": 1e22,
}
SCALE_PARAMS_B = {
    "3e18": 0.447,
    "9e18": 0.550,
    "2e19": 0.837,
    "3e19": 0.998,
    "9e19": 1.4,
    "2e20": 1.9,
    "3e20": 2.5,
    "1e21": 3.4,
    "1e22": 9.7,
}
SCALE_PRETRAIN_TOKENS_B = {
    "3e18": 1.2,
    "9e18": 2.9,
    "2e19": 3.6,
    "3e19": 5.0,
    "9e19": 10.6,
    "2e20": 14.8,
    "3e20": 18.6,
    "1e21": 46.3,
    "1e22": 160.0,
}
MIDTRAIN_BUDGET_FRACTION = 0.20
MATH_FRACTION = {"p67m33": 0.33, "p50m50": 0.50, "p33m67": 0.67}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def scale_label(value: Any) -> str:
    if isinstance(value, str) and value in ALL_SCALE_FLOPS:
        return value
    numeric = float(value)
    return min(ALL_SCALE_FLOPS, key=lambda scale: abs(math.log10(ALL_SCALE_FLOPS[scale]) - math.log10(numeric)))


def attach_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scale"] = out["scale"].map(scale_label)
    out["scale_flops"] = out["scale"].map(ALL_SCALE_FLOPS).astype(float)
    lr_num = pd.to_numeric(out["lr"], errors="coerce")
    out["lr_factor"] = np.where(lr_num > 1.0, lr_num / 100.0, lr_num)
    out["lr_key"] = (100 * out["lr_factor"]).round().astype(int).astype(str)
    out["recipe"] = out["mix"] + "-lr" + out["lr_key"]
    out["recipe_key"] = out["mix"] + "-" + out["lr_key"]
    out["math_frac"] = out["mix"].map(MATH_FRACTION).astype(float)
    out["params_b"] = out["scale"].map(SCALE_PARAMS_B).astype(float)
    out["midtrain_tokens_b"] = out["scale"].map(SCALE_PRETRAIN_TOKENS_B).astype(float) * MIDTRAIN_BUDGET_FRACTION
    out["dmath_b"] = out["midtrain_tokens_b"] * out["math_frac"]
    out["c"] = out["scale_flops"] / 1e18
    out["log_c"] = np.log(out["c"])
    out["log_n"] = np.log(out["params_b"])
    out["log_dmath"] = np.log(out["dmath_b"])
    return out


def load_math_points() -> pd.DataFrame:
    endpoints = pd.read_csv(BASE_DIR / "endpoints.csv", dtype={"scale": str, "lr": str})
    targets = pd.read_csv(BASE_DIR / "extrapolation_targets.csv", dtype={"scale": str, "lr": str})

    small = endpoints[bool_series(endpoints["complete"]) & endpoints["metric_label"].eq(MATH_METRIC)].copy()
    small["is_heldout"] = False
    small["target_kind"] = "small_ladder"
    held = targets[bool_series(targets["complete"]) & targets["metric_label"].eq(MATH_METRIC)].copy()
    held["is_heldout"] = True
    cols = ["scale", "scale_flops", "mix", "lr", "value", "is_heldout", "target_kind", "run_id", "run_name"]
    points = pd.concat([small[cols], held[cols]], ignore_index=True)
    points = attach_features(points)
    points["scale"] = pd.Categorical(points["scale"], categories=SCALE_ORDER, ordered=True)
    points["mix"] = pd.Categorical(points["mix"], categories=MIX_ORDER, ordered=True)
    points["lr_key"] = pd.Categorical(points["lr_key"], categories=LR_ORDER, ordered=True)
    return points.sort_values(["scale", "mix", "lr_key"]).reset_index(drop=True)


def fit_log_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(intercept), float(slope)


def predict_log_power(intercept: float, slope: float, x: np.ndarray | float) -> np.ndarray:
    return np.exp(intercept + slope * np.log(x))


def fit_per_recipe_power(train: pd.DataFrame, value_col: str = "value") -> dict[str, tuple[float, float]]:
    fits: dict[str, tuple[float, float]] = {}
    for recipe, group in train.groupby("recipe_key", observed=True):
        group = group.sort_values("scale_flops")
        if len(group) < MIN_POINTS_FOR_FIT or (group[value_col] <= 0).any():
            continue
        fits[str(recipe)] = fit_log_power(group["c"].to_numpy(dtype=float), group[value_col].to_numpy(dtype=float))
    return fits


def predict_per_recipe_power(fits: dict[str, tuple[float, float]], df: pd.DataFrame) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        fit = fits.get(str(row.recipe_key))
        if fit is None:
            continue
        pred[i] = float(predict_log_power(fit[0], fit[1], float(row.c)))
    return pred


def recipe_intercept_design(df: pd.DataFrame, extra_columns: list[np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    recipes = sorted(df["recipe_key"].astype(str).unique())
    recipe_index = {recipe: i for i, recipe in enumerate(recipes)}
    x = np.zeros((len(df), len(recipes) + len(extra_columns)))
    rows = df["recipe_key"].astype(str).map(recipe_index).to_numpy()
    x[np.arange(len(df)), rows] = 1.0
    for offset, column in enumerate(extra_columns):
        x[:, len(recipes) + offset] = column
    return x, recipe_index


@dataclass(frozen=True)
class OlsModel:
    beta: np.ndarray
    recipe_index: dict[str, int]
    columns: list[str]
    transform: Callable[[pd.DataFrame], list[np.ndarray]]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        extra = self.transform(df)
        x, _ = recipe_intercept_design_with_index(df, extra, self.recipe_index)
        return np.exp(x @ self.beta)


def recipe_intercept_design_with_index(
    df: pd.DataFrame, extra_columns: list[np.ndarray], recipe_index: dict[str, int]
) -> tuple[np.ndarray, dict[str, int]]:
    x = np.zeros((len(df), len(recipe_index) + len(extra_columns)))
    rows = df["recipe_key"].astype(str).map(lambda recipe: recipe_index.get(recipe, -1)).to_numpy()
    valid = rows >= 0
    safe_rows = np.clip(rows, 0, max(len(recipe_index) - 1, 0))
    x[np.arange(len(df))[valid], safe_rows[valid]] = 1.0
    for offset, column in enumerate(extra_columns):
        x[:, len(recipe_index) + offset] = column
    x[~valid, :] = np.nan
    return x, recipe_index


def fit_ols_form(
    train: pd.DataFrame, columns: list[str], transform: Callable[[pd.DataFrame], list[np.ndarray]]
) -> OlsModel:
    extra = transform(train)
    x, recipe_index = recipe_intercept_design(train, extra)
    beta, *_ = np.linalg.lstsq(x, np.log(train["value"].to_numpy(dtype=float)), rcond=None)
    return OlsModel(beta=beta, recipe_index=recipe_index, columns=columns, transform=transform)


def per_recipe_power_predictions(points: pd.DataFrame) -> pd.DataFrame:
    train = points[points["scale"].isin(SMALL_SCALES)]
    held = points[points["scale"].isin(HELDOUT_SCALES)].copy()
    fits = fit_per_recipe_power(train)
    held["predicted"] = predict_per_recipe_power(fits, held)
    held["experiment"] = "absolute_per_recipe_power_small_only"
    return score_prediction_frame(held)


def pooled_form_predictions(points: pd.DataFrame) -> pd.DataFrame:
    train = points[points["scale"].isin(SMALL_SCALES)]
    held = points[points["scale"].isin(HELDOUT_SCALES)].copy()
    forms: list[tuple[str, list[str], Callable[[pd.DataFrame], list[np.ndarray]]]] = [
        (
            "absolute_pooled_recipe_intercept_shared_slope",
            ["log_c"],
            lambda df: [df["log_c"].to_numpy(dtype=float)],
        ),
        (
            "absolute_pooled_mix_slope",
            ["log_c", "log_c_x_math_frac"],
            lambda df: [
                df["log_c"].to_numpy(dtype=float),
                df["log_c"].to_numpy(dtype=float) * df["math_frac"].to_numpy(dtype=float),
            ],
        ),
        (
            "absolute_pooled_mix_lr_slope",
            ["log_c", "log_c_x_math_frac", "log_c_x_lr"],
            lambda df: [
                df["log_c"].to_numpy(dtype=float),
                df["log_c"].to_numpy(dtype=float) * df["math_frac"].to_numpy(dtype=float),
                df["log_c"].to_numpy(dtype=float) * df["lr_factor"].to_numpy(dtype=float),
            ],
        ),
        (
            "absolute_pooled_mechanistic_linear",
            ["log_n", "log_dmath", "lr_factor"],
            lambda df: [
                df["log_n"].to_numpy(dtype=float),
                df["log_dmath"].to_numpy(dtype=float),
                df["lr_factor"].to_numpy(dtype=float),
            ],
        ),
        (
            "absolute_pooled_mechanistic_mix_interaction",
            ["log_n", "log_dmath", "lr_factor", "log_dmath_x_math_frac"],
            lambda df: [
                df["log_n"].to_numpy(dtype=float),
                df["log_dmath"].to_numpy(dtype=float),
                df["lr_factor"].to_numpy(dtype=float),
                df["log_dmath"].to_numpy(dtype=float) * df["math_frac"].to_numpy(dtype=float),
            ],
        ),
    ]
    rows: list[pd.DataFrame] = []
    for name, columns, transform in forms:
        model = fit_ols_form(train, columns, transform)
        pred = held.copy()
        pred["predicted"] = model.predict(pred)
        pred["experiment"] = name
        rows.append(score_prediction_frame(pred))
    return pd.concat(rows, ignore_index=True)


def score_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["observed"] = out["value"].astype(float)
    out["error"] = out["observed"] - out["predicted"]
    out["signed_pct_error"] = 100.0 * out["error"] / out["observed"]
    out["abs_pct_error"] = out["signed_pct_error"].abs()
    out["target_scale"] = out["scale"].astype(str)
    out["extrapolation_multiple_from_small_max"] = out["scale_flops"].astype(float) / ALL_SCALE_FLOPS["3e20"]
    return out[
        [
            "experiment",
            "target_scale",
            "mix",
            "lr_key",
            "recipe",
            "observed",
            "predicted",
            "error",
            "signed_pct_error",
            "abs_pct_error",
            "extrapolation_multiple_from_small_max",
        ]
    ]


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["abs_pct_error"])
        .groupby(["experiment", "target_scale"], observed=True)
        .agg(
            n=("abs_pct_error", "size"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            median_abs_pct_error=("abs_pct_error", "median"),
            max_abs_pct_error=("abs_pct_error", "max"),
            signed_mean_pct_error=("signed_pct_error", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["target_scale", "mean_abs_pct_error", "experiment"])


def local_slopes(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (mix, lr_key), group in points.groupby(["mix", "lr_key"], observed=True):
        group = group.sort_values("scale_flops")
        records = group.to_dict("records")
        for left, right in pairwise(records):
            slope = (math.log(float(right["value"])) - math.log(float(left["value"]))) / (
                math.log(float(right["scale_flops"])) - math.log(float(left["scale_flops"]))
            )
            rows.append(
                {
                    "mix": mix,
                    "lr_key": lr_key,
                    "recipe": f"{mix}-lr{lr_key}",
                    "left_scale": left["scale"],
                    "right_scale": right["scale"],
                    "interval": f"{left['scale']}->{right['scale']}",
                    "mid_log_flops": (
                        0.5 * (math.log(float(left["scale_flops"])) + math.log(float(right["scale_flops"])))
                    ),
                    "slope": slope,
                }
            )
    return pd.DataFrame(rows)


def anchored_predictions(points: pd.DataFrame) -> pd.DataFrame:
    train_small = points[points["scale"].isin(SMALL_SCALES)]
    target = points[points["scale"].eq("1e22")].copy()
    anchor = points[points["scale"].eq("1e21")][["recipe_key", "value", "c"]].rename(
        columns={"value": "anchor_value", "c": "anchor_c"}
    )
    target = target.merge(anchor, on="recipe_key", how="left")
    fits = fit_per_recipe_power(train_small)
    rows: list[pd.DataFrame] = []

    small_power = target.copy()
    small_power["predicted"] = [
        (
            row.anchor_value * (row.c / row.anchor_c) ** fits[str(row.recipe_key)][1]
            if str(row.recipe_key) in fits and np.isfinite(row.anchor_value)
            else np.nan
        )
        for row in small_power.itertuples(index=False)
    ]
    small_power["experiment"] = "anchor_1e21_small_power_slope"
    rows.append(score_prediction_frame(small_power))

    slopes = local_slopes(points[points["scale"].isin([*SMALL_SCALES, "1e21"])])
    last_slopes = slopes[slopes["interval"].eq("3e20->1e21")][["recipe", "slope"]]
    last = target.merge(last_slopes, on="recipe", how="left")
    last["predicted"] = last["anchor_value"] * (last["c"] / last["anchor_c"]) ** last["slope"]
    last["experiment"] = "anchor_1e21_last_segment_slope"
    rows.append(score_prediction_frame(last))

    trend_rows = []
    for recipe, slope_group in slopes.groupby("recipe", observed=True):
        slope_group = slope_group.sort_values("mid_log_flops")
        if len(slope_group) < 3:
            continue
        b, a = np.polyfit(slope_group["mid_log_flops"].to_numpy(), slope_group["slope"].to_numpy(), 1)
        target_mid = 0.5 * (math.log(ALL_SCALE_FLOPS["1e21"]) + math.log(ALL_SCALE_FLOPS["1e22"]))
        trend_rows.append({"recipe": recipe, "trend_slope": float(a + b * target_mid)})
    trend = target.merge(pd.DataFrame(trend_rows), on="recipe", how="left")
    trend["predicted"] = trend["anchor_value"] * (trend["c"] / trend["anchor_c"]) ** trend["trend_slope"]
    trend["experiment"] = "anchor_1e21_recipe_slope_trend"
    rows.append(score_prediction_frame(trend))

    drift_rows = []
    small_fit_rows = [
        {"recipe": recipe.replace("-", "-lr"), "recipe_key": recipe, "small_slope": fit[1]}
        for recipe, fit in fits.items()
    ]
    small_fit = pd.DataFrame(small_fit_rows)
    last_for_drift = last_slopes.merge(small_fit[["recipe", "small_slope"]], on="recipe", how="left")
    last_for_drift["next_slope"] = last_for_drift["slope"] + (last_for_drift["slope"] - last_for_drift["small_slope"])
    drift_rows = last_for_drift[["recipe", "next_slope"]]
    drift = target.merge(drift_rows, on="recipe", how="left")
    drift["predicted"] = drift["anchor_value"] * (drift["c"] / drift["anchor_c"]) ** drift["next_slope"]
    drift["experiment"] = "anchor_1e21_repeat_slope_drift"
    rows.append(score_prediction_frame(drift))

    return pd.concat(rows, ignore_index=True)


def endpoint_table_from_trajectory() -> pd.DataFrame:
    traj = pd.read_csv(BASE_DIR / "trajectory_points.csv", dtype={"scale": str, "lr": str}, low_memory=False)
    traj = traj[traj["metric_label"].eq(MATH_METRIC) & bool_series(traj["complete"])].copy()
    traj = attach_features(traj)
    final_rows = (
        traj.sort_values(["run_id", "tau"])
        .groupby(["run_id", "scale", "mix", "lr_key", "recipe", "eval_split"], observed=True)
        .tail(1)
        .copy()
    )
    final_rows["final"] = final_rows["final_value"].astype(float)
    final_rows["baseline"] = final_rows["baseline_value"].astype(float)
    final_rows["improvement"] = final_rows["baseline"] - final_rows["final"]
    final_rows["relative_improvement"] = final_rows["improvement"] / final_rows["baseline"]
    final_rows["final_ratio"] = final_rows["final"] / final_rows["baseline"]
    final_rows["is_heldout"] = final_rows["scale"].isin(HELDOUT_SCALES)
    return final_rows


def transformed_delta_predictions() -> pd.DataFrame:
    endpoints = endpoint_table_from_trajectory()
    train = endpoints[endpoints["scale"].isin(SMALL_SCALES)]
    held = endpoints[endpoints["scale"].isin(HELDOUT_SCALES)].copy()
    specs = [
        ("delta_raw_improvement", "improvement", lambda row, pred: row.baseline - pred),
        ("delta_relative_improvement", "relative_improvement", lambda row, pred: row.baseline * (1.0 - pred)),
        ("delta_final_ratio", "final_ratio", lambda row, pred: row.baseline * pred),
    ]
    rows: list[pd.DataFrame] = []
    for experiment, value_col, reconstruct in specs:
        fit_train = train[train[value_col] > 0].copy()
        fit_train["value"] = fit_train[value_col].astype(float)
        fits = fit_per_recipe_power(fit_train, value_col="value")
        fit_held = held.copy()
        fit_held["value"] = fit_held[value_col].astype(float)
        pred_values = predict_per_recipe_power(fits, fit_held)
        pred = held.copy()
        pred["predicted"] = [
            reconstruct(row, pred_value)
            for row, pred_value in zip(pred.itertuples(index=False), pred_values, strict=True)
        ]
        pred["value"] = pred["final"]
        pred["experiment"] = experiment
        rows.append(score_prediction_frame(pred))
    return pd.concat(rows, ignore_index=True)


def mix_gap_predictions(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    train = points[points["scale"].isin(SMALL_SCALES)].copy()
    target = points[points["scale"].isin(HELDOUT_SCALES)].copy()

    baseline = per_recipe_power_predictions(points)
    baseline_pred = baseline[["target_scale", "mix", "lr_key", "predicted"]].rename(
        columns={"predicted": "baseline_predicted"}
    )
    target = target.merge(
        baseline_pred,
        left_on=["scale", "mix", "lr_key"],
        right_on=["target_scale", "mix", "lr_key"],
        how="left",
    )

    def gap_frame(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
        pivot = frame.pivot_table(index=["scale", "lr_key"], columns="mix", values=value_col, observed=True)
        out_rows = []
        for (scale, lr_key), row in pivot.iterrows():
            if not np.isfinite(row.get("p67m33", np.nan)):
                continue
            for mix in ["p50m50", "p33m67"]:
                if np.isfinite(row.get(mix, np.nan)):
                    out_rows.append(
                        {
                            "scale": scale,
                            "lr": lr_key,
                            "lr_key": lr_key,
                            "mix": mix,
                            "gap_from_p67": float(row[mix] - row["p67m33"]),
                        }
                    )
        return attach_features(pd.DataFrame(out_rows))

    train_gap = gap_frame(train, "value")
    fit_gap = train_gap.copy()
    fit_gap["gap_magnitude"] = -fit_gap["gap_from_p67"]
    gap_fits = fit_per_recipe_power(fit_gap.rename(columns={"gap_magnitude": "value"}), value_col="value")

    target_gap = gap_frame(target, "value")
    target_gap["gap_magnitude_pred"] = predict_per_recipe_power(
        gap_fits, target_gap.rename(columns={"gap_magnitude": "value"})
    )
    target_gap["gap_pred"] = -target_gap["gap_magnitude_pred"]

    p67_pred = baseline[baseline["mix"].astype(str).eq("p67m33")][
        ["target_scale", "lr_key", "predicted", "observed"]
    ].rename(columns={"predicted": "p67_predicted", "observed": "p67_observed"})
    gap_scored = target_gap.merge(p67_pred, left_on=["scale", "lr_key"], right_on=["target_scale", "lr_key"])
    observed_lookup = target[["scale", "mix", "lr_key", "value", "recipe"]].rename(columns={"value": "observed"})
    gap_scored = gap_scored.merge(observed_lookup, on=["scale", "mix", "lr_key"], how="left")
    if "recipe_x" in gap_scored.columns:
        gap_scored["recipe"] = gap_scored["recipe_x"]
    elif "recipe_y" in gap_scored.columns:
        gap_scored["recipe"] = gap_scored["recipe_y"]
    gap_scored["value"] = gap_scored["observed"]
    gap_scored["predicted"] = gap_scored["p67_predicted"] + gap_scored["gap_pred"]
    gap_scored["experiment"] = "mix_gap_small_power_plus_predicted_p67"
    rows.append(score_prediction_frame(gap_scored))

    gap_anchor = gap_scored.copy()
    gap_anchor["predicted"] = gap_anchor["p67_observed"] + gap_anchor["gap_pred"]
    gap_anchor["experiment"] = "mix_gap_small_power_plus_observed_p67_anchor"
    rows.append(score_prediction_frame(gap_anchor))
    return pd.concat(rows, ignore_index=True)


def rank_predictability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (experiment, target_scale), group in predictions.dropna(subset=["predicted"]).groupby(
        ["experiment", "target_scale"], observed=True
    ):
        if len(group) < 3:
            continue
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        rank_obs = pd.Series(observed).rank(method="average", ascending=True).to_numpy()
        rank_pred = pd.Series(predicted).rank(method="average", ascending=True).to_numpy()
        spearman = float(np.corrcoef(rank_obs, rank_pred)[0, 1])
        best_pred_row = group.iloc[int(np.argmin(predicted))]
        best_obs = float(np.min(observed))
        rows.append(
            {
                "experiment": experiment,
                "target_scale": target_scale,
                "n": len(group),
                "spearman_rank": spearman,
                "predicted_best_recipe": best_pred_row["recipe"],
                "predicted_best_observed_loss": float(best_pred_row["observed"]),
                "actual_best_loss": best_obs,
                "selection_regret_abs": float(best_pred_row["observed"] - best_obs),
                "selection_regret_pct": 100.0 * float(best_pred_row["observed"] - best_obs) / best_obs,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_scale", "selection_regret_pct", "experiment"])


def bootstrap_power_intervals(points: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train = points[points["scale"].isin(SMALL_SCALES)]
    held = points[points["scale"].isin(HELDOUT_SCALES)].copy()
    rows: list[dict[str, Any]] = []
    for recipe, group in train.groupby("recipe_key", observed=True):
        group = group.sort_values("c")
        if len(group) < MIN_POINTS_FOR_FIT:
            continue
        x = np.log(group["c"].to_numpy(dtype=float))
        y = np.log(group["value"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        residual = y - fitted
        recipe_held = held[held["recipe_key"].astype(str).eq(str(recipe))]
        for target in recipe_held.itertuples(index=False):
            log_x_target = math.log(float(target.c))
            samples = []
            for _ in range(n_boot):
                resampled = rng.choice(residual, size=len(residual), replace=True)
                b_slope, b_intercept = np.polyfit(x, fitted + resampled, 1)
                samples.append(math.exp(b_intercept + b_slope * log_x_target))
            arr = np.array(samples)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            rows.append(
                {
                    "recipe": target.recipe,
                    "target_scale": target.scale,
                    "observed": float(target.value),
                    "point_predicted": math.exp(intercept + slope * log_x_target),
                    "bootstrap_p025": float(lo),
                    "bootstrap_p975": float(hi),
                    "covered": bool(lo <= float(target.value) <= hi),
                    "interval_width_pct_of_observed": 100.0 * float(hi - lo) / float(target.value),
                }
            )
    return pd.DataFrame(rows)


def protocol_audit(points: pd.DataFrame) -> pd.DataFrame:
    run_ids = sorted(points[points["scale"].isin(HELDOUT_SCALES)]["run_id"].dropna().astype(str).unique())
    rows = []
    for run_id in run_ids:
        path = CONFIG_DIR / run_id / "config.json"
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        trainer = config.get("trainer") if isinstance(config.get("trainer"), dict) else {}
        optimizer = config.get("optimizer") if isinstance(config.get("optimizer"), dict) else {}
        num_steps = trainer.get("num_train_steps")
        warmup = optimizer.get("warmup")
        warmup_fraction = None
        if isinstance(num_steps, (int, float)) and isinstance(warmup, (int, float)) and num_steps:
            warmup_fraction = float(warmup) / float(num_steps)
        rows.append(
            {
                "run_id": run_id,
                "checkpoint_init_mode": config.get("checkpoint_init_mode"),
                "initialize_from_checkpoint_path": config.get("initialize_from_checkpoint_path"),
                "trainer_load_checkpoint_path": trainer.get("load_checkpoint_path"),
                "trainer_num_train_steps": num_steps,
                "optimizer_warmup": warmup,
                "optimizer_warmup_fraction": warmup_fraction,
                "optimizer_cooldown": optimizer.get("cooldown"),
                "optimizer_rewarmup": optimizer.get("rewarmup"),
                "optimizer_lr_schedule": optimizer.get("lr_schedule"),
            }
        )
    return pd.DataFrame(rows)


def prefix_summary() -> pd.DataFrame:
    path = BASE_DIR / "trajectory_prefix_predictions.csv"
    pred = pd.read_csv(path, dtype={"scale": str, "lr": str}, low_memory=False)
    pred = pred[
        pred["metric_label"].eq(MATH_METRIC) & pred["eval_split"].eq("heldout_large") & bool_series(pred["complete"])
    ].copy()
    pred["scale"] = pred["scale"].map(scale_label)
    grouped = (
        pred.groupby(["scale", "prefix", "method"], observed=True)
        .agg(
            n=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            mean_abs_pct_error=("pct_error", lambda s: s.abs().mean()),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(["scale", "prefix", "mean_abs_pct_error", "method"])


def baseline_metric_summary() -> pd.DataFrame:
    predictions = pd.read_csv(BASE_DIR / "extrapolation_predictions.csv")
    predictions = predictions[
        predictions["fit_kind"].eq("log_loss_vs_log_compute") & bool_series(predictions["target_complete"])
    ]
    grouped = (
        predictions.groupby(["metric_label", "target_scale"], observed=True)
        .agg(
            n=("pct_error", "size"),
            mean_abs_pct_error=("pct_error", lambda s: s.abs().mean()),
            signed_mean_pct_error=("pct_error", "mean"),
            max_abs_pct_error=("pct_error", lambda s: s.abs().max()),
        )
        .reset_index()
        .sort_values(["target_scale", "mean_abs_pct_error"])
    )
    grouped["target_scale"] = grouped["target_scale"].map(scale_label)
    return grouped


def tradeoff_residuals() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(BASE_DIR / "extrapolation_predictions.csv")
    predictions = predictions[
        predictions["fit_kind"].eq("log_loss_vs_log_compute") & bool_series(predictions["target_complete"])
    ].copy()
    predictions["target_scale"] = predictions["target_scale"].map(scale_label)
    wide = (
        predictions.pivot_table(
            index=["target_scale", "mix", "lr"],
            columns="metric_label",
            values="pct_error",
            observed=True,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    metric_columns = [
        column
        for column in ["math_val_loss", "eval_loss", "paloma_macro_loss", "paloma_c4_loss", "train_loss"]
        if column in wide.columns
    ]
    by_mix = (
        wide.groupby(["target_scale", "mix"], observed=True)[metric_columns]
        .mean()
        .reset_index()
        .sort_values(["target_scale", "mix"])
    )
    return wide, by_mix


def score_envelope_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out["observed"] - out["predicted"]
    out["signed_pct_error"] = 100.0 * out["error"] / out["observed"]
    out["abs_pct_error"] = out["signed_pct_error"].abs()
    return out[
        [
            "experiment",
            "target_scale",
            "scope",
            "mix",
            "observed",
            "predicted",
            "error",
            "signed_pct_error",
            "abs_pct_error",
        ]
    ]


def envelope_predictions(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict best-LR and best-overall envelopes instead of every LR cell."""
    best_mix = (
        points.sort_values("value").groupby(["scale", "mix"], observed=True).head(1).sort_values(["scale", "mix"]).copy()
    )
    best_mix["scope"] = "best_lr_per_mix"
    best_mix["envelope_key"] = best_mix["mix"].astype(str)

    best_overall = points.sort_values("value").groupby("scale", observed=True).head(1).copy()
    best_overall["scope"] = "best_overall"
    best_overall["envelope_key"] = "overall"
    best_overall["mix"] = "overall"

    envelope = pd.concat([best_mix, best_overall], ignore_index=True)
    train = envelope[envelope["scale"].isin(SMALL_SCALES)]
    held = envelope[envelope["scale"].isin(HELDOUT_SCALES)]

    rows: list[dict[str, Any]] = []
    for (scope, key), group in train.groupby(["scope", "envelope_key"], observed=True):
        if len(group) < MIN_POINTS_FOR_FIT:
            continue
        intercept, slope = fit_log_power(group["c"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float))
        target_rows = held[held["scope"].eq(scope) & held["envelope_key"].eq(key)]
        for target in target_rows.itertuples(index=False):
            rows.append(
                {
                    "experiment": "envelope_small_only_power",
                    "target_scale": target.scale,
                    "scope": scope,
                    "mix": target.mix,
                    "observed": float(target.value),
                    "predicted": float(predict_log_power(intercept, slope, float(target.c))),
                }
            )

        anchor_rows = envelope[
            envelope["scale"].eq("1e21") & envelope["scope"].eq(scope) & envelope["envelope_key"].eq(key)
        ]
        target_1e22 = held[held["scale"].eq("1e22") & held["scope"].eq(scope) & held["envelope_key"].eq(key)]
        if not anchor_rows.empty and not target_1e22.empty:
            anchor = anchor_rows.iloc[0]
            target = target_1e22.iloc[0]
            rows.append(
                {
                    "experiment": "envelope_anchor_1e21_small_slope",
                    "target_scale": target["scale"],
                    "scope": scope,
                    "mix": target["mix"],
                    "observed": float(target["value"]),
                    "predicted": float(anchor["value"]) * (float(target["c"]) / float(anchor["c"])) ** slope,
                }
            )

    scored = score_envelope_frame(pd.DataFrame(rows))
    summary = (
        scored.groupby(["experiment", "target_scale", "scope"], observed=True)
        .agg(
            n=("abs_pct_error", "size"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            max_abs_pct_error=("abs_pct_error", "max"),
            signed_mean_pct_error=("signed_pct_error", "mean"),
        )
        .reset_index()
        .sort_values(["target_scale", "scope", "mean_abs_pct_error"])
    )
    return scored, summary


def anchored_slope_drift_cv(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin check for the 1e21-anchored slope-drift heuristic."""
    rows: list[dict[str, Any]] = []
    c_by_scale = {scale: ALL_SCALE_FLOPS[scale] / 1e18 for scale in SCALE_ORDER}
    for test_index in range(3, len(SCALE_ORDER)):
        test_scale = SCALE_ORDER[test_index]
        anchor_scale = SCALE_ORDER[test_index - 1]
        previous_scale = SCALE_ORDER[test_index - 2]
        prior_scales = SCALE_ORDER[: test_index - 1]
        for _, group in points.groupby("recipe_key", observed=True):
            by_scale = {row.scale: row for row in group.itertuples(index=False)}
            if test_scale not in by_scale or anchor_scale not in by_scale:
                continue
            if previous_scale not in by_scale or any(scale not in by_scale for scale in prior_scales):
                continue
            x = np.log([c_by_scale[scale] for scale in prior_scales])
            y = np.log([float(by_scale[scale].value) for scale in prior_scales])
            small_slope, _ = np.polyfit(x, y, 1)
            last_slope = (
                math.log(float(by_scale[anchor_scale].value)) - math.log(float(by_scale[previous_scale].value))
            ) / (math.log(c_by_scale[anchor_scale]) - math.log(c_by_scale[previous_scale]))
            methods = {
                "anchor_small_slope": small_slope,
                "anchor_last_segment": last_slope,
                "anchor_repeat_drift": last_slope + (last_slope - small_slope),
            }
            for method, slope in methods.items():
                observed = float(by_scale[test_scale].value)
                predicted = (
                    float(by_scale[anchor_scale].value) * (c_by_scale[test_scale] / c_by_scale[anchor_scale]) ** slope
                )
                signed_pct_error = 100.0 * (observed - predicted) / observed
                rows.append(
                    {
                        "method": method,
                        "test_scale": test_scale,
                        "recipe": by_scale[test_scale].recipe,
                        "mix": by_scale[test_scale].mix,
                        "lr_key": by_scale[test_scale].lr_key,
                        "observed": observed,
                        "predicted": predicted,
                        "signed_pct_error": signed_pct_error,
                        "abs_pct_error": abs(signed_pct_error),
                        "jump_multiple": c_by_scale[test_scale] / c_by_scale[anchor_scale],
                    }
                )
    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby(["method", "test_scale"], observed=True)
        .agg(
            n=("abs_pct_error", "size"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            signed_mean_pct_error=("signed_pct_error", "mean"),
            max_abs_pct_error=("abs_pct_error", "max"),
            jump_multiple=("jump_multiple", "first"),
        )
        .reset_index()
        .sort_values(["test_scale", "mean_abs_pct_error"])
    )
    return cv, summary


def prefix_fraction_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare how much final improvement has appeared by each prefix."""
    trajectory = pd.read_csv(BASE_DIR / "trajectory_points.csv", dtype={"scale": str, "lr": str}, low_memory=False)
    trajectory = trajectory[trajectory["metric_label"].eq(MATH_METRIC) & bool_series(trajectory["complete"])].copy()
    trajectory["scale"] = trajectory["scale"].map(scale_label)
    prefixes = [round(value, 2) for value in np.arange(0.10, 0.95, 0.05)]
    rows: list[dict[str, Any]] = []
    for (run_id, _metric), group in trajectory.groupby(["run_id", "metric"], observed=True):
        group = group.sort_values("tau")
        first = group.iloc[0]
        baseline = float(first["baseline_value"])
        final = float(first["final_value"])
        denominator = baseline - final
        if denominator <= 0:
            continue
        for prefix in prefixes:
            prefix_points = group[group["tau"].le(prefix)]
            if prefix_points.empty:
                continue
            row = prefix_points.iloc[-1]
            fraction = (baseline - float(row["value"])) / denominator
            rows.append(
                {
                    "run_id": run_id,
                    "scale": row["scale"],
                    "mix": row["mix"],
                    "lr": row["lr"],
                    "recipe": row["recipe"],
                    "eval_split": row["eval_split"],
                    "prefix": prefix,
                    "prefix_actual_tau": float(row["tau"]),
                    "fraction_of_final_improvement": fraction,
                }
            )
    fractions = pd.DataFrame(rows)
    small = (
        fractions[fractions["eval_split"].eq("small_cv")]
        .groupby("prefix", observed=True)["fraction_of_final_improvement"]
        .mean()
        .rename("small_mean_fraction")
    )
    large = (
        fractions[fractions["scale"].isin(HELDOUT_SCALES)]
        .groupby(["scale", "prefix"], observed=True)["fraction_of_final_improvement"]
        .mean()
        .rename("large_mean_fraction")
        .reset_index()
    )
    comparison = large.merge(small, on="prefix")
    comparison["large_minus_small"] = comparison["large_mean_fraction"] - comparison["small_mean_fraction"]
    return fractions, comparison


def write_plots(
    points: pd.DataFrame,
    slopes: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    rank: pd.DataFrame,
    prefix: pd.DataFrame,
    intervals: pd.DataFrame,
    metric_summary: pd.DataFrame,
    envelope_summary: pd.DataFrame,
    slope_drift_cv_summary: pd.DataFrame,
    prefix_fraction_comparison: pd.DataFrame,
    tradeoff_by_cell: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = px.line(
        points,
        x="scale_flops",
        y="value",
        color="mix",
        line_dash="lr_key",
        markers=True,
        log_x=True,
        category_orders={"mix": MIX_ORDER, "lr_key": LR_ORDER},
        title="Math validation loss endpoints by scale",
    )
    fig.add_vline(x=ALL_SCALE_FLOPS["3e20"], line_dash="dot", annotation_text="small-fit max")
    fig.write_html(OUT_DIR / "math_loss_endpoints.html", include_plotlyjs="cdn")

    slope_avg = slopes.groupby(["interval", "mix"], observed=True)["slope"].mean().reset_index()
    slope_avg["interval"] = pd.Categorical(
        slope_avg["interval"],
        categories=[f"{a}->{b}" for a, b in pairwise(SCALE_ORDER)],
        ordered=True,
    )
    fig = px.line(
        slope_avg.sort_values("interval"),
        x="interval",
        y="slope",
        color="mix",
        markers=True,
        category_orders={"mix": MIX_ORDER},
        title="Average local log-log slope by mix",
    )
    fig.write_html(OUT_DIR / "local_slopes_by_mix.html", include_plotlyjs="cdn")

    baseline = predictions[
        predictions["experiment"].eq("absolute_per_recipe_power_small_only") & predictions["target_scale"].eq("1e22")
    ].copy()
    if not baseline.empty:
        heat = baseline.pivot_table(index="mix", columns="lr_key", values="signed_pct_error", observed=True)
        fig = px.imshow(
            heat.loc[MIX_ORDER, LR_ORDER],
            text_auto=".1f",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="1e22 signed percent error: small-only per-recipe power",
            labels={"color": "signed % error"},
        )
        fig.write_html(OUT_DIR / "baseline_1e22_error_heatmap.html", include_plotlyjs="cdn")

    fig = px.bar(
        summary[summary["target_scale"].isin(HELDOUT_SCALES)],
        x="experiment",
        y="mean_abs_pct_error",
        color="target_scale",
        barmode="group",
        title="Prediction experiments: mean absolute percent error",
    )
    fig.update_layout(xaxis_tickangle=-35)
    fig.write_html(OUT_DIR / "prediction_experiment_summary.html", include_plotlyjs="cdn")

    rank_1e22 = rank[rank["target_scale"].eq("1e22")].copy()
    if not rank_1e22.empty:
        fig = px.scatter(
            rank_1e22,
            x="spearman_rank",
            y="selection_regret_pct",
            color="experiment",
            hover_data=["predicted_best_recipe", "predicted_best_observed_loss", "actual_best_loss"],
            title="1e22 recipe selection: rank correlation vs regret",
        )
        fig.write_html(OUT_DIR / "rank_selection_1e22.html", include_plotlyjs="cdn")

    best_prefix = (
        prefix[prefix["scale"].eq("1e22")].sort_values(["prefix", "mean_abs_pct_error"]).groupby("prefix").head(3)
    )
    if not best_prefix.empty:
        fig = px.line(
            best_prefix,
            x="prefix",
            y="mean_abs_pct_error",
            color="method",
            markers=True,
            title="1e22 final math loss from early prefixes: top 3 methods per prefix",
        )
        fig.write_html(OUT_DIR / "prefix_prediction_1e22.html", include_plotlyjs="cdn")

    if not intervals.empty:
        plot_intervals = intervals[intervals["target_scale"].eq("1e22")].copy()
        plot_intervals["covered_label"] = plot_intervals["covered"].map({True: "covered", False: "missed"})
        fig = go.Figure()
        for _, row in plot_intervals.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["bootstrap_p025"], row["bootstrap_p975"]],
                    y=[row["recipe"], row["recipe"]],
                    mode="lines",
                    line={"color": "#888"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=plot_intervals["point_predicted"],
                y=plot_intervals["recipe"],
                mode="markers",
                name="point prediction",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_intervals["observed"],
                y=plot_intervals["recipe"],
                mode="markers",
                name="observed",
            )
        )
        fig.update_layout(title="1e22 residual-bootstrap intervals from small-ladder power fits", xaxis_title="loss")
        fig.write_html(OUT_DIR / "bootstrap_intervals_1e22.html", include_plotlyjs="cdn")

    if not metric_summary.empty:
        fig = px.bar(
            metric_summary,
            x="metric_label",
            y="mean_abs_pct_error",
            color="target_scale",
            barmode="group",
            title="Baseline small-only power error by metric",
        )
        fig.update_layout(xaxis_tickangle=-25)
        fig.write_html(OUT_DIR / "baseline_metric_errors.html", include_plotlyjs="cdn")

    if not envelope_summary.empty:
        fig = px.bar(
            envelope_summary,
            x="scope",
            y="mean_abs_pct_error",
            color="experiment",
            facet_col="target_scale",
            barmode="group",
            title="Envelope prediction error: best LR per mix and best overall",
        )
        fig.write_html(OUT_DIR / "envelope_prediction_summary.html", include_plotlyjs="cdn")

    if not slope_drift_cv_summary.empty:
        fig = px.bar(
            slope_drift_cv_summary,
            x="test_scale",
            y="mean_abs_pct_error",
            color="method",
            barmode="group",
            hover_data=["jump_multiple", "signed_mean_pct_error", "max_abs_pct_error"],
            title="Rolling-origin CV for anchored slope-drift heuristics",
        )
        fig.write_html(OUT_DIR / "anchored_slope_drift_cv.html", include_plotlyjs="cdn")

    if not prefix_fraction_comparison.empty:
        fig = px.line(
            prefix_fraction_comparison,
            x="prefix",
            y="large_minus_small",
            color="scale",
            markers=True,
            title="Large-run prefix improvement fraction minus small-run mean",
        )
        fig.add_hline(y=0.0, line_dash="dot")
        fig.write_html(OUT_DIR / "prefix_fraction_large_vs_small.html", include_plotlyjs="cdn")

    if not tradeoff_by_cell.empty and {"math_val_loss", "paloma_macro_loss"}.issubset(tradeoff_by_cell.columns):
        fig = px.scatter(
            tradeoff_by_cell,
            x="math_val_loss",
            y="paloma_macro_loss",
            color="mix",
            symbol="target_scale",
            hover_data=["lr", "eval_loss", "paloma_c4_loss"],
            title="Baseline signed percent errors: math vs Paloma macro",
            labels={
                "math_val_loss": "math signed % error",
                "paloma_macro_loss": "Paloma macro signed % error",
            },
        )
        fig.add_vline(x=0.0, line_dash="dot")
        fig.add_hline(y=0.0, line_dash="dot")
        fig.write_html(OUT_DIR / "math_vs_paloma_residual_tradeoff.html", include_plotlyjs="cdn")


def write_markdown_summary(
    summary: pd.DataFrame,
    slopes: pd.DataFrame,
    rank: pd.DataFrame,
    prefix: pd.DataFrame,
    intervals: pd.DataFrame,
    protocol: pd.DataFrame,
    metric_summary: pd.DataFrame,
    envelope_summary: pd.DataFrame,
    slope_drift_cv_summary: pd.DataFrame,
    prefix_fraction_comparison: pd.DataFrame,
    tradeoff_by_mix: pd.DataFrame,
) -> None:
    lines = [
        "# Codex Delphi Midtraining Prediction Retro",
        "",
        "This is an independent analysis over cached endpoint/trajectory exports.",
        "",
        "## Prediction Experiments",
        "",
        "| experiment | target | n | mean abs % | median abs % | max abs % | signed mean % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['target_scale']} | {int(row['n'])} | "
            f"{row['mean_abs_pct_error']:.2f} | {row['median_abs_pct_error']:.2f} | "
            f"{row['max_abs_pct_error']:.2f} | {row['signed_mean_pct_error']:+.2f} |"
        )

    slope_focus = (
        slopes[slopes["interval"].isin(["3e20->1e21", "1e21->1e22"])]
        .groupby(["interval", "mix"], observed=True)["slope"]
        .mean()
        .reset_index()
        .sort_values(["interval", "mix"])
    )
    lines.extend(["", "## Local Slopes", "", "| interval | mix | mean slope |", "|---|---|---:|"])
    for _, row in slope_focus.iterrows():
        lines.append(f"| {row['interval']} | {row['mix']} | {row['slope']:.4f} |")

    rank_1e22 = rank[rank["target_scale"].eq("1e22")].head(12)
    lines.extend(
        [
            "",
            "## 1e22 Rank And Selection",
            "",
            "| experiment | Spearman | predicted best | regret % |",
            "|---|---:|---|---:|",
        ]
    )
    for _, row in rank_1e22.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['spearman_rank']:.3f} | "
            f"{row['predicted_best_recipe']} | {row['selection_regret_pct']:.2f} |"
        )

    best_prefix = (
        prefix[prefix["scale"].eq("1e22")].sort_values(["prefix", "mean_abs_pct_error"]).groupby("prefix").head(1)
    )
    lines.extend(
        [
            "",
            "## 1e22 Prefix Calibration",
            "",
            "| prefix | best method | n | mean abs % | max abs error |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for _, row in best_prefix.iterrows():
        lines.append(
            f"| {row['prefix']:.2f} | {row['method']} | {int(row['n'])} | "
            f"{row['mean_abs_pct_error']:.2f} | {row['max_abs_error']:.4f} |"
        )

    if not intervals.empty:
        coverage = intervals.groupby("target_scale", observed=True)["covered"].agg(["sum", "count"]).reset_index()
        lines.extend(["", "## Bootstrap Coverage", "", "| target | covered | total |", "|---|---:|---:|"])
        for _, row in coverage.iterrows():
            lines.append(f"| {row['target_scale']} | {int(row['sum'])} | {int(row['count'])} |")

    if not metric_summary.empty:
        lines.extend(
            [
                "",
                "## Baseline Error By Metric",
                "",
                "| metric | target | n | mean abs % | signed mean % | max abs % |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in metric_summary.iterrows():
            lines.append(
                f"| {row['metric_label']} | {row['target_scale']} | {int(row['n'])} | "
                f"{row['mean_abs_pct_error']:.2f} | {row['signed_mean_pct_error']:+.2f} | "
                f"{row['max_abs_pct_error']:.2f} |"
            )

    if not envelope_summary.empty:
        lines.extend(
            [
                "",
                "## Envelope Prediction",
                "",
                "| experiment | target | scope | n | mean abs % | signed mean % | max abs % |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in envelope_summary.iterrows():
            lines.append(
                f"| {row['experiment']} | {row['target_scale']} | {row['scope']} | {int(row['n'])} | "
                f"{row['mean_abs_pct_error']:.2f} | {row['signed_mean_pct_error']:+.2f} | "
                f"{row['max_abs_pct_error']:.2f} |"
            )

    if not slope_drift_cv_summary.empty:
        focus = slope_drift_cv_summary[slope_drift_cv_summary["test_scale"].isin(["1e21", "1e22"])]
        lines.extend(
            [
                "",
                "## Anchored Slope-Drift CV",
                "",
                "| method | test | jump | n | mean abs % | signed mean % | max abs % |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in focus.iterrows():
            lines.append(
                f"| {row['method']} | {row['test_scale']} | {row['jump_multiple']:.1f} | {int(row['n'])} | "
                f"{row['mean_abs_pct_error']:.2f} | {row['signed_mean_pct_error']:+.2f} | "
                f"{row['max_abs_pct_error']:.2f} |"
            )

    if not prefix_fraction_comparison.empty:
        focus = prefix_fraction_comparison[
            prefix_fraction_comparison["prefix"].isin([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90])
        ]
        lines.extend(
            [
                "",
                "## Prefix Fraction Diagnostics",
                "",
                "| scale | prefix | large mean fraction | small mean fraction | large - small |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in focus.iterrows():
            lines.append(
                f"| {row['scale']} | {row['prefix']:.2f} | {row['large_mean_fraction']:.3f} | "
                f"{row['small_mean_fraction']:.3f} | {row['large_minus_small']:+.3f} |"
            )

    if not tradeoff_by_mix.empty:
        metric_columns = [
            column
            for column in ["math_val_loss", "eval_loss", "paloma_macro_loss", "paloma_c4_loss"]
            if column in tradeoff_by_mix.columns
        ]
        lines.extend(
            [
                "",
                "## Residual Tradeoff By Mix",
                "",
                "| target | mix | " + " | ".join(metric_columns) + " |",
                "|---|---|" + "|".join("---:" for _ in metric_columns) + "|",
            ]
        )
        for _, row in tradeoff_by_mix.iterrows():
            values = " | ".join(f"{row[column]:+.2f}" for column in metric_columns)
            lines.append(f"| {row['target_scale']} | {row['mix']} | {values} |")

    if not protocol.empty:
        proto_summary = (
            protocol.groupby(
                [
                    "checkpoint_init_mode",
                    "trainer_load_checkpoint_path",
                    "optimizer_cooldown",
                    "optimizer_rewarmup",
                    "optimizer_lr_schedule",
                ],
                dropna=False,
                observed=True,
            )
            .agg(
                n=("run_id", "size"),
                mean_warmup_fraction=("optimizer_warmup_fraction", "mean"),
                min_warmup_fraction=("optimizer_warmup_fraction", "min"),
                max_warmup_fraction=("optimizer_warmup_fraction", "max"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "",
                "## Held-Out Config Protocol Audit",
                "",
                "| init mode | load ckpt path | cooldown | rewarmup | schedule | n | warmup frac mean | range |",
                "|---|---|---|---|---|---:|---:|---|",
            ]
        )
        for _, row in proto_summary.iterrows():
            lines.append(
                f"| {row['checkpoint_init_mode']} | {row['trainer_load_checkpoint_path']} | "
                f"{row['optimizer_cooldown']} | {row['optimizer_rewarmup']} | {row['optimizer_lr_schedule']} | "
                f"{int(row['n'])} | {row['mean_warmup_fraction']:.3f} | "
                f"{row['min_warmup_fraction']:.3f}-{row['max_warmup_fraction']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- `math_loss_endpoints.html`",
            "- `local_slopes_by_mix.html`",
            "- `baseline_1e22_error_heatmap.html`",
            "- `prediction_experiment_summary.html`",
            "- `rank_selection_1e22.html`",
            "- `prefix_prediction_1e22.html`",
            "- `bootstrap_intervals_1e22.html`",
            "- `baseline_metric_errors.html`",
            "- `envelope_prediction_summary.html`",
            "- `anchored_slope_drift_cv.html`",
            "- `prefix_fraction_large_vs_small.html`",
            "- `math_vs_paloma_residual_tradeoff.html`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_math_points()
    slopes = local_slopes(points)

    prediction_frames = [
        per_recipe_power_predictions(points),
        pooled_form_predictions(points),
        anchored_predictions(points),
        transformed_delta_predictions(),
        mix_gap_predictions(points),
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_predictions(predictions)
    rank = rank_predictability(predictions)
    intervals = bootstrap_power_intervals(points)
    prefix = prefix_summary()
    protocol = protocol_audit(points)
    metric_summary = baseline_metric_summary()
    envelope, envelope_summary = envelope_predictions(points)
    slope_drift_cv, slope_drift_cv_summary = anchored_slope_drift_cv(points)
    prefix_fractions, prefix_fraction_comparison = prefix_fraction_diagnostics()
    tradeoff_by_cell, tradeoff_by_mix = tradeoff_residuals()

    points.to_csv(OUT_DIR / "math_points.csv", index=False)
    slopes.to_csv(OUT_DIR / "local_slopes.csv", index=False)
    predictions.to_csv(OUT_DIR / "prediction_experiments.csv", index=False)
    summary.to_csv(OUT_DIR / "prediction_experiment_summary.csv", index=False)
    rank.to_csv(OUT_DIR / "rank_predictability.csv", index=False)
    intervals.to_csv(OUT_DIR / "bootstrap_power_intervals.csv", index=False)
    prefix.to_csv(OUT_DIR / "prefix_prediction_summary.csv", index=False)
    protocol.to_csv(OUT_DIR / "heldout_protocol_audit.csv", index=False)
    metric_summary.to_csv(OUT_DIR / "baseline_metric_summary.csv", index=False)
    envelope.to_csv(OUT_DIR / "envelope_predictions.csv", index=False)
    envelope_summary.to_csv(OUT_DIR / "envelope_prediction_summary.csv", index=False)
    slope_drift_cv.to_csv(OUT_DIR / "anchored_slope_drift_cv.csv", index=False)
    slope_drift_cv_summary.to_csv(OUT_DIR / "anchored_slope_drift_cv_summary.csv", index=False)
    prefix_fractions.to_csv(OUT_DIR / "prefix_fraction_diagnostics.csv", index=False)
    prefix_fraction_comparison.to_csv(OUT_DIR / "prefix_fraction_large_vs_small.csv", index=False)
    tradeoff_by_cell.to_csv(OUT_DIR / "tradeoff_residuals_by_cell.csv", index=False)
    tradeoff_by_mix.to_csv(OUT_DIR / "tradeoff_residuals_by_mix.csv", index=False)

    write_plots(
        points,
        slopes,
        predictions,
        summary,
        rank,
        prefix,
        intervals,
        metric_summary,
        envelope_summary,
        slope_drift_cv_summary,
        prefix_fraction_comparison,
        tradeoff_by_cell,
    )
    write_markdown_summary(
        summary,
        slopes,
        rank,
        prefix,
        intervals,
        protocol,
        metric_summary,
        envelope_summary,
        slope_drift_cv_summary,
        prefix_fraction_comparison,
        tradeoff_by_mix,
    )

    print(summary.to_string(index=False))
    print("\nTop 1e22 prefix methods:")
    best_prefix = (
        prefix[prefix["scale"].eq("1e22")].sort_values(["prefix", "mean_abs_pct_error"]).groupby("prefix").head(1)
    )
    print(best_prefix[["prefix", "method", "n", "mean_abs_pct_error", "max_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
