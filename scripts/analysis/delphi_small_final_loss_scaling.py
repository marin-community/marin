# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit endpoint validation-loss scaling curves for small Delphi CPT runs.

This is a deliberately simple first pass: fetch W&B summaries for the
3e18 -> 3e20 K=0.20 CPT ladder, select the best non-probe attempt for each
cell, and fit final observed validation loss across compute scale.

Outputs:
    midtrain_analysis_outputs/small_final_loss_scaling/endpoints.csv
    midtrain_analysis_outputs/small_final_loss_scaling/fit_summary.csv
    midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_targets.csv
    midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_predictions.csv
    midtrain_analysis_outputs/small_final_loss_scaling/summary.md
    midtrain_analysis_outputs/small_final_loss_scaling/*.html

Run:
    uv run python scripts/analysis/delphi_small_final_loss_scaling.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb
from scipy.optimize import curve_fit, least_squares

logger = logging.getLogger("delphi_small_final_loss_scaling")

PROJECT = "marin-community/delphi-midtraining"
OUT_DIR = Path("midtrain_analysis_outputs/small_final_loss_scaling")
RUN_REGISTRY_PATH = Path("midtrain_analysis_outputs/midtrain_run_registry.csv")
TRAJECTORY_DELTAS_PATH = Path("midtrain_analysis_outputs/midtrain_trajectory_deltas.csv")

RUN_PATTERN = re.compile(
    r"^delphi-(?P<scale>3e18|9e18|2e19|3e19|9e19|2e20|3e20)-"
    r"(?P<mix>p33m67|p50m50|p67m33)-k0p20-lr(?P<lr>33|50|67|83)-a(?P<attempt>\d{3})$"
)
HELDOUT_RUN_PATTERN = re.compile(
    r"^delphi-(?P<scale>1e21|1e22)-"
    r"(?P<mix>p33m67|p50m50|p67m33)-(?P<budget>9p25b|32p07b)-"
    r"lr(?P<lr>0\.33|0\.5|0\.67|0\.83)-(?P<suffix>[a-z0-9]+)$"
)

SCALE_FLOPS = {
    "3e18": 3e18,
    "9e18": 9e18,
    "2e19": 2e19,
    "3e19": 3e19,
    "9e19": 9e19,
    "2e20": 2e20,
    "3e20": 3e20,
}
HELDOUT_SCALE_FLOPS = {
    "1e21": 1e21,
    "1e22": 1e22,
}
ALL_SCALE_FLOPS = {**SCALE_FLOPS, **HELDOUT_SCALE_FLOPS}
SCALE_ORDER = list(SCALE_FLOPS)
HELDOUT_SCALE_ORDER = list(HELDOUT_SCALE_FLOPS)
ALL_SCALE_ORDER = SCALE_ORDER + HELDOUT_SCALE_ORDER
MIX_ORDER = ["p67m33", "p50m50", "p33m67"]
LR_ORDER = ["33", "50", "67", "83"]

METRICS = {
    "eval/nemotron_cc_math_v1/4plus/loss": "math_val_loss",
    "eval/loss": "eval_loss",
    "eval/paloma/macro_loss": "paloma_macro_loss",
    "eval/paloma/c4_en/loss": "paloma_c4_loss",
    "train/loss": "train_loss",
}

COMPLETION_TOLERANCE = 5
MIN_POINTS_FOR_LOG_LINEAR = 3
MIN_POINTS_FOR_FLOOR_POWER = 5

# Per-scale model size and pretrain-token budget, read from the canonical Delphi model
# registry (experiments/delphi_models.py HF repo names, e.g.
# "delphi-3e18-447Mparams-1.2Btokens"). Midtraining spends K=0.20 of the pretrain token
# budget (midtraining_mixes.MIDTRAIN_BUDGET_FRACTION); the math tokens a recipe sees in
# midtraining are mathfrac * K * pretrain_tokens. These two axes (model size N and math
# data D_math) co-move along the single `scale_flops` ladder, so a 1-D power law in FLOPs
# cannot separate them -- the multi-axis fits below do.
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

# Forms compared head-to-head on held-out 1e21/1e22 and in rolling-origin CV. The first is
# the canonical per-recipe single power law; the rest are the new candidates.
FORM_BASELINE = "per_recipe_power"

TARGET_COLUMNS = [
    "run_id",
    "run_name",
    "state",
    "created_at",
    "url",
    "scale",
    "scale_flops",
    "log10_flops",
    "mix",
    "lr",
    "lr_factor",
    "budget",
    "global_step",
    "expected_steps",
    "progress",
    "complete",
    "target_kind",
    "canonical_for_cell",
    "best_prefix_for_cell",
    "last_metric_step",
    "metric",
    "metric_label",
    "value",
    "recipe",
    "source",
]

PREDICTION_COLUMNS = [
    "metric",
    "metric_label",
    "fit_kind",
    "mix",
    "lr",
    "recipe",
    "target_scale",
    "target_scale_flops",
    "target_kind",
    "target_complete",
    "target_progress",
    "observed",
    "predicted",
    "error",
    "abs_error",
    "pct_error",
    "train_scales",
    "train_max_scale",
    "extrapolation_multiple",
    "fit_n",
    "fit_exponent",
    "fit_r2",
    "fit_rmse",
    "fit_loocv_rmse",
    "run_id",
    "run_name",
    "url",
]


@dataclass(frozen=True)
class RunEndpoint:
    run_id: str
    run_name: str
    state: str
    created_at: str
    url: str
    scale: str
    scale_flops: float
    mix: str
    lr: str
    lr_factor: float
    attempt: int
    global_step: int | None
    expected_steps: int | None
    complete: bool
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cache", action="store_true", help="Read endpoints.csv instead of querying W&B")
    parser.add_argument("--project", default=PROJECT, help="W&B entity/project")
    return parser.parse_args()


def summary_dict(run) -> dict[str, Any]:
    if hasattr(run.summary, "_json_dict"):
        return dict(run.summary._json_dict)
    return dict(run.summary)


def expected_steps_from_config(config: dict[str, Any]) -> int | None:
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return None
    value = trainer.get("num_train_steps")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return int(value)
    return None


def global_step_from_summary(summary: dict[str, Any]) -> int | None:
    for key in ("global_step", "_step"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return int(value)
    return None


def is_complete(state: str, global_step: int | None, expected_steps: int | None) -> bool:
    if state == "finished":
        return True
    if global_step is None or expected_steps is None:
        return False
    return global_step >= expected_steps - 1 - COMPLETION_TOLERANCE


def fetch_run_endpoints(project: str) -> list[RunEndpoint]:
    api = wandb.Api(timeout=60)
    filters = {"display_name": {"$regex": r"^delphi-(3e18|9e18|2e19|3e19|9e19|2e20|3e20)-"}}
    candidates: list[RunEndpoint] = []

    for run in api.runs(project, filters=filters):
        match = RUN_PATTERN.match(run.name)
        if match is None:
            continue
        groups = match.groupdict()
        summary = summary_dict(run)
        expected_steps = expected_steps_from_config(dict(run.config))
        global_step = global_step_from_summary(summary)
        candidates.append(
            RunEndpoint(
                run_id=run.id,
                run_name=run.name,
                state=run.state,
                created_at=run.created_at,
                url=run.url,
                scale=groups["scale"],
                scale_flops=SCALE_FLOPS[groups["scale"]],
                mix=groups["mix"],
                lr=groups["lr"],
                lr_factor=int(groups["lr"]) / 100,
                attempt=int(groups["attempt"]),
                global_step=global_step,
                expected_steps=expected_steps,
                complete=is_complete(run.state, global_step, expected_steps),
                summary=summary,
            )
        )

    selected: dict[tuple[str, str, str], RunEndpoint] = {}
    for endpoint in candidates:
        key = (endpoint.scale, endpoint.mix, endpoint.lr)
        current = selected.get(key)
        if current is None or endpoint_rank(endpoint) > endpoint_rank(current):
            selected[key] = endpoint

    return sorted(
        selected.values(),
        key=lambda item: (
            SCALE_ORDER.index(item.scale),
            MIX_ORDER.index(item.mix),
            LR_ORDER.index(item.lr),
        ),
    )


def endpoint_rank(endpoint: RunEndpoint) -> tuple[int, int, str]:
    status_rank = 2 if endpoint.complete else 1 if endpoint.state == "running" else 0
    return (status_rank, endpoint.attempt, endpoint.created_at)


def endpoints_to_long(endpoints: list[RunEndpoint]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for metric, metric_label in METRICS.items():
            value = endpoint.summary.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            rows.append(
                {
                    "run_id": endpoint.run_id,
                    "run_name": endpoint.run_name,
                    "state": endpoint.state,
                    "created_at": endpoint.created_at,
                    "url": endpoint.url,
                    "scale": endpoint.scale,
                    "scale_flops": endpoint.scale_flops,
                    "log10_flops": math.log10(endpoint.scale_flops),
                    "mix": endpoint.mix,
                    "lr": endpoint.lr,
                    "lr_factor": endpoint.lr_factor,
                    "attempt": endpoint.attempt,
                    "global_step": endpoint.global_step,
                    "expected_steps": endpoint.expected_steps,
                    "complete": endpoint.complete,
                    "metric": metric,
                    "metric_label": metric_label,
                    "value": float(value),
                    "recipe": f"{endpoint.mix}-lr{endpoint.lr}",
                }
            )
    return pd.DataFrame(rows)


def lr_label(value: Any) -> str:
    text = str(value).strip()
    if text in LR_ORDER:
        return text
    number = float(text)
    return str(round(number * 100))


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value):
        return bool(value)
    return False


def target_complete(registry_row: pd.Series) -> bool:
    if bool_value(registry_row.get("canonical_for_cell")):
        return True
    state = str(registry_row.get("state", ""))
    if state == "finished":
        return True
    max_step = registry_row.get("max_step")
    expected_step = registry_row.get("expected_final_step")
    if isinstance(max_step, (int, float)) and isinstance(expected_step, (int, float)):
        if math.isfinite(max_step) and math.isfinite(expected_step):
            return max_step >= expected_step - 1 - COMPLETION_TOLERANCE
    return False


def target_rank(row: dict[str, Any]) -> tuple[int, float, str]:
    status_rank = 2 if bool_value(row["complete"]) else 1 if row["state"] in {"running", "crashed"} else 0
    progress = row.get("progress")
    progress_rank = float(progress) if isinstance(progress, (int, float, np.integer, np.floating)) else 0.0
    return (status_rank, progress_rank, str(row["created_at"]))


def empty_targets() -> pd.DataFrame:
    return pd.DataFrame(columns=TARGET_COLUMNS)


def heldout_targets_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return empty_targets()
    targets = pd.DataFrame(rows)
    targets["scale"] = pd.Categorical(targets["scale"], categories=HELDOUT_SCALE_ORDER, ordered=True)
    targets["mix"] = pd.Categorical(targets["mix"], categories=MIX_ORDER, ordered=True)
    targets["lr"] = pd.Categorical(targets["lr"], categories=LR_ORDER, ordered=True)
    return targets.sort_values(["scale", "mix", "lr", "metric_label"]).reset_index(drop=True)


def fetch_wandb_heldout_targets(project: str) -> pd.DataFrame:
    api = wandb.Api(timeout=60)
    filters = {"display_name": {"$regex": r"^delphi-(1e21|1e22)-"}}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}

    for run in api.runs(project, filters=filters):
        match = HELDOUT_RUN_PATTERN.match(run.name)
        if match is None:
            continue
        groups = match.groupdict()
        summary = summary_dict(run)
        expected_steps = expected_steps_from_config(dict(run.config))
        global_step = global_step_from_summary(summary)
        complete = is_complete(run.state, global_step, expected_steps)
        scale = groups["scale"]
        lr = lr_label(groups["lr"])
        progress = (
            global_step / (expected_steps - 1)
            if global_step is not None and expected_steps is not None and expected_steps > 1
            else float("nan")
        )
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
            "scale": scale,
            "scale_flops": HELDOUT_SCALE_FLOPS[scale],
            "log10_flops": math.log10(HELDOUT_SCALE_FLOPS[scale]),
            "mix": groups["mix"],
            "lr": lr,
            "lr_factor": int(lr) / 100,
            "budget": groups["budget"],
            "global_step": global_step,
            "expected_steps": expected_steps,
            "progress": progress,
            "complete": complete,
            "target_kind": "complete" if complete else "best_prefix",
            "canonical_for_cell": complete,
            "best_prefix_for_cell": True,
            "last_metric_step": global_step,
            "source": "wandb_live",
            "summary": summary,
        }
        key = (scale, groups["mix"], lr)
        current = selected.get(key)
        if current is None or target_rank(row) > target_rank(current):
            selected[key] = row

    rows: list[dict[str, Any]] = []
    for selected_row in selected.values():
        selected_row = dict(selected_row)
        summary = selected_row.pop("summary")
        for metric, metric_label in METRICS.items():
            value = summary.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                continue
            rows.append(
                {
                    **selected_row,
                    "metric": metric,
                    "metric_label": metric_label,
                    "value": float(value),
                    "recipe": f"{selected_row['mix']}-lr{selected_row['lr']}",
                }
            )

    return heldout_targets_from_rows(rows)


def load_local_heldout_targets() -> pd.DataFrame:
    if not RUN_REGISTRY_PATH.exists() or not TRAJECTORY_DELTAS_PATH.exists():
        logger.warning("Missing local large-scale registry or trajectory file; skipping held-out extrapolation targets.")
        return empty_targets()

    registry = pd.read_csv(
        RUN_REGISTRY_PATH,
        dtype={"run_id": str, "name": str, "scale": str, "mix": str, "budget": str},
    )
    registry = registry[
        registry["scale"].isin(HELDOUT_SCALE_ORDER)
        & registry["mix"].isin(MIX_ORDER)
        & registry["best_prefix_for_cell"].map(bool_value)
    ].copy()
    if registry.empty:
        return empty_targets()

    registry["lr"] = registry["lr"].map(lr_label)
    registry = registry[registry["lr"].isin(LR_ORDER)].copy()
    registry["complete"] = registry.apply(target_complete, axis=1)
    registry = registry.drop_duplicates("run_id", keep="last").set_index("run_id")

    trajectory = pd.read_csv(
        TRAJECTORY_DELTAS_PATH,
        usecols=["run_id", "step", "metric", "value", "final_observed_value"],
        dtype={"run_id": str, "metric": str},
    )
    trajectory = trajectory[trajectory["run_id"].isin(registry.index) & trajectory["metric"].isin(METRICS)].copy()
    if trajectory.empty:
        return empty_targets()

    rows: list[dict[str, Any]] = []
    for (run_id, metric), group in trajectory.groupby(["run_id", "metric"], observed=True, sort=False):
        values = group["final_observed_value"].dropna()
        if values.empty:
            values = group.sort_values("step")["value"].dropna()
        if values.empty:
            continue
        value = float(values.iloc[-1])
        if not math.isfinite(value) or value <= 0:
            continue

        registry_row = registry.loc[run_id]
        scale = str(registry_row["scale"])
        lr = str(registry_row["lr"])
        complete = bool_value(registry_row["complete"])
        rows.append(
            {
                "run_id": run_id,
                "run_name": registry_row["name"],
                "state": registry_row["state"],
                "created_at": registry_row["created_at"],
                "url": registry_row["url"],
                "scale": scale,
                "scale_flops": HELDOUT_SCALE_FLOPS[scale],
                "log10_flops": math.log10(HELDOUT_SCALE_FLOPS[scale]),
                "mix": registry_row["mix"],
                "lr": lr,
                "lr_factor": int(lr) / 100,
                "budget": registry_row["budget"],
                "global_step": registry_row["summary_step"],
                "expected_steps": registry_row["expected_final_step"],
                "progress": registry_row["progress"],
                "complete": complete,
                "target_kind": "complete" if complete else "best_prefix",
                "canonical_for_cell": bool_value(registry_row["canonical_for_cell"]),
                "best_prefix_for_cell": bool_value(registry_row["best_prefix_for_cell"]),
                "last_metric_step": float(group["step"].max()),
                "metric": metric,
                "metric_label": METRICS[metric],
                "value": value,
                "recipe": f"{registry_row['mix']}-lr{lr}",
                "source": str(TRAJECTORY_DELTAS_PATH),
            }
        )

    return heldout_targets_from_rows(rows)


def load_heldout_targets(project: str) -> pd.DataFrame:
    targets = fetch_wandb_heldout_targets(project)
    target_cells = targets[["scale", "mix", "lr"]].drop_duplicates() if not targets.empty else targets
    if len(target_cells) >= len(HELDOUT_SCALE_ORDER) * len(MIX_ORDER) * len(LR_ORDER):
        return targets

    logger.warning("Live W&B held-out target coverage is incomplete; falling back to local held-out target dump.")
    return load_local_heldout_targets()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def loocv_rmse_log_linear(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 2:
        return float("nan")
    preds: list[float] = []
    targets: list[float] = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        if np.any(y[mask] <= 0):
            return float("nan")
        coeffs = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
        preds.append(float(np.exp(coeffs[1] + coeffs[0] * np.log(x[i]))))
        targets.append(float(y[i]))
    return float(np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2)))


def fit_log_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    exponent = float(coeffs[0])
    intercept = float(coeffs[1])
    pred = np.exp(intercept + exponent * np.log(x))
    return {
        "fit_kind": "log_loss_vs_log_compute",
        "n": float(len(x)),
        "floor": float("nan"),
        "amplitude": float(math.exp(intercept)),
        "exponent": exponent,
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "loocv_rmse": loocv_rmse_log_linear(x, y),
    }


def floor_power_model(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -alpha)


def fit_floor_power(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    y_min = float(np.min(y))
    y_span = float(np.max(y) - np.min(y))
    if y_min <= 0 or y_span <= 1e-8:
        return None

    floor0 = max(1e-8, y_min - 0.2 * y_span)
    amplitude0 = max(y_span, 1e-6)
    alpha0 = 0.1
    try:
        params, _ = curve_fit(
            floor_power_model,
            x,
            y,
            p0=(floor0, amplitude0, alpha0),
            bounds=([0.0, 0.0, 0.0], [y_min * 0.999, max(float(np.max(y)) * 100, 1.0), 5.0]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    pred = floor_power_model(x, *params)
    return {
        "fit_kind": "floor_plus_power",
        "n": float(len(x)),
        "floor": float(params[0]),
        "amplitude": float(params[1]),
        "exponent": float(params[2]),
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "loocv_rmse": float("nan"),
    }


def fit_groups(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = endpoints[endpoints["complete"]].copy()
    for (metric, metric_label, mix, lr), group in complete.groupby(
        ["metric", "metric_label", "mix", "lr"],
        observed=True,
        sort=False,
    ):
        group = group.sort_values("scale_flops")
        x = (group["scale_flops"].to_numpy(dtype=float) / 1e18).astype(float)
        y = group["value"].to_numpy(dtype=float)
        if len(group) < MIN_POINTS_FOR_LOG_LINEAR or np.any(y <= 0):
            continue
        base = {
            "metric": metric,
            "metric_label": metric_label,
            "mix": mix,
            "lr": lr,
            "lr_factor": int(lr) / 100,
            "scales": ",".join(group["scale"].tolist()),
            "min_scale": group.iloc[0]["scale"],
            "max_scale": group.iloc[-1]["scale"],
            "first_value": float(y[0]),
            "last_value": float(y[-1]),
            "delta_first_to_last": float(y[-1] - y[0]),
            "monotone_nonincreasing": bool(np.all(np.diff(y) <= 1e-8)),
        }
        rows.append({**base, **fit_log_linear(x, y)})
        if len(group) >= MIN_POINTS_FOR_FLOOR_POWER:
            floor_fit = fit_floor_power(x, y)
            if floor_fit is not None:
                rows.append({**base, **floor_fit})
    return pd.DataFrame(rows)


def predict_from_fit(fit_row: pd.Series, scale_flops: float) -> float:
    compute_x = scale_flops / 1e18
    if fit_row["fit_kind"] == "log_loss_vs_log_compute":
        return float(fit_row["amplitude"] * math.pow(compute_x, fit_row["exponent"]))
    return float(fit_row["floor"] + fit_row["amplitude"] * math.pow(compute_x, -fit_row["exponent"]))


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def predict_heldout_targets(fits: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if fits.empty or targets.empty:
        return empty_predictions()

    rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        matching_fits = fits[
            fits["metric"].eq(target["metric"]) & fits["mix"].eq(target["mix"]) & fits["lr"].eq(target["lr"])
        ]
        for _, fit_row in matching_fits.iterrows():
            predicted = predict_from_fit(fit_row, float(target["scale_flops"]))
            observed = float(target["value"])
            error = observed - predicted
            train_max_flops = SCALE_FLOPS.get(str(fit_row["max_scale"]), float("nan"))
            rows.append(
                {
                    "metric": target["metric"],
                    "metric_label": target["metric_label"],
                    "fit_kind": fit_row["fit_kind"],
                    "mix": target["mix"],
                    "lr": target["lr"],
                    "recipe": target["recipe"],
                    "target_scale": target["scale"],
                    "target_scale_flops": target["scale_flops"],
                    "target_kind": target["target_kind"],
                    "target_complete": target["complete"],
                    "target_progress": target["progress"],
                    "observed": observed,
                    "predicted": predicted,
                    "error": error,
                    "abs_error": abs(error),
                    "pct_error": 100 * error / observed,
                    "train_scales": fit_row["scales"],
                    "train_max_scale": fit_row["max_scale"],
                    "extrapolation_multiple": float(target["scale_flops"]) / train_max_flops,
                    "fit_n": fit_row["n"],
                    "fit_exponent": fit_row["exponent"],
                    "fit_r2": fit_row["r2"],
                    "fit_rmse": fit_row["rmse"],
                    "fit_loocv_rmse": fit_row["loocv_rmse"],
                    "run_id": target["run_id"],
                    "run_name": target["run_name"],
                    "url": target["url"],
                }
            )
    if not rows:
        return empty_predictions()
    return pd.DataFrame(rows).sort_values(["metric_label", "fit_kind", "target_scale_flops", "mix", "lr"])


# --------------------------------------------------------------------------------------
# Multi-axis / curvature scaling forms.
#
# The per-recipe single power law extrapolates 1e21 well (~1%) but is systematically
# over-pessimistic at 1e22 (~10%, worst for math-heavy mixes) because the loss-vs-compute
# curve *accelerates* at the top of the ladder -- the local log-log slope steepens from
# ~-0.10 to -0.12/-0.14/-0.15 for p67m33/p50m50/p33m67. A Chinchilla floor bends the curve
# the wrong way (it flattens), so it cannot fix this. The forms below either separate the
# model-size (N) and math-token (D_math) axes that the 1-D FLOPs ladder conflates, or allow
# the slope itself to change with compute.
# --------------------------------------------------------------------------------------

MIN_CV_TRAIN_SCALES = 3


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def attach_scaling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-size, math-token, compute, and LR features used by the multi-axis fits.

    Compute is in units of 1e18 FLOPs; ``log_*`` columns are natural logs. ``recipe_key``
    is a mix/LR identifier that is stable across the endpoints and held-out target frames
    (which store ``lr`` as ``"33"`` and ``"0.33"`` respectively).
    """
    out = df.copy()
    lr_num = pd.to_numeric(out["lr"], errors="coerce")
    out["lr_factor_num"] = np.where(lr_num > 1.0, lr_num / 100.0, lr_num)
    out["recipe_key"] = out["mix"] + "-" + (out["lr_factor_num"] * 100).round().astype(int).astype(str)
    out["math_frac"] = out["mix"].map(MATH_FRACTION).astype(float)
    params_b = out["scale"].map(SCALE_PARAMS_B).astype(float)
    midtrain_b = out["scale"].map(SCALE_PRETRAIN_TOKENS_B).astype(float) * MIDTRAIN_BUDGET_FRACTION
    out["dmath_b"] = midtrain_b * out["math_frac"]
    out["c"] = out["scale_flops"].astype(float) / 1e18
    out["log_c"] = np.log(out["c"])
    out["log_n"] = np.log(params_b)
    out["log_dmath"] = np.log(out["dmath_b"])
    return out


@dataclass(frozen=True)
class EndpointForm:
    """A candidate endpoint scaling form.

    ``fit`` consumes the training rows (with :func:`attach_scaling_features` columns) and
    returns opaque fitted state; ``predict`` maps that state plus a frame of rows to a
    predicted-loss array aligned to the frame's row order. Per-recipe forms fit one model
    per ``recipe_key``; pooled forms fit once across all cells and rely on the N / D_math /
    LR features (and per-recipe intercepts) to separate cells.
    """

    name: str
    fit: Callable[[pd.DataFrame], Any]
    predict: Callable[[Any, pd.DataFrame], np.ndarray]
    label: str


def _fit_per_recipe_power(train: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        rk: np.polyfit(g["log_c"].to_numpy(), np.log(g["value"].to_numpy()), 1)
        for rk, g in train.groupby("recipe_key")
        if len(g) >= MIN_POINTS_FOR_LOG_LINEAR
    }


def _predict_per_recipe_power(state: dict[str, np.ndarray], df: pd.DataFrame) -> np.ndarray:
    slope = df["recipe_key"].map(lambda rk: state[rk][0] if rk in state else np.nan).to_numpy()
    intercept = df["recipe_key"].map(lambda rk: state[rk][1] if rk in state else np.nan).to_numpy()
    return np.exp(intercept + slope * df["log_c"].to_numpy())


def _fit_per_recipe_floor(train: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for rk, g in train.groupby("recipe_key"):
        fit = fit_floor_power(g["c"].to_numpy(dtype=float), g["value"].to_numpy(dtype=float))
        if fit is not None:
            out[rk] = (fit["floor"], fit["amplitude"], fit["exponent"])
    return out


def _predict_per_recipe_floor(state: dict[str, tuple[float, float, float]], df: pd.DataFrame) -> np.ndarray:
    def one(rk: str, c: float) -> float:
        params = state.get(rk)
        return params[0] + params[1] * c ** (-params[2]) if params is not None else float("nan")

    return np.array([one(rk, c) for rk, c in zip(df["recipe_key"], df["c"], strict=True)])


def _fit_pooled_curvature(train: pd.DataFrame) -> dict[str, Any]:
    recipes = sorted(train["recipe_key"].unique())
    idx = {rk: i for i, rk in enumerate(recipes)}
    n_recipes = len(recipes)
    design = np.zeros((len(train), n_recipes + 2))
    rows = train["recipe_key"].map(idx).to_numpy()
    design[np.arange(len(train)), rows] = 1.0
    design[:, n_recipes] = train["log_c"].to_numpy()
    design[:, n_recipes + 1] = train["log_c"].to_numpy() ** 2
    beta, *_ = np.linalg.lstsq(design, np.log(train["value"].to_numpy()), rcond=None)
    return {"idx": idx, "beta": beta, "n_recipes": n_recipes}


def _predict_pooled_curvature(state: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    beta = state["beta"]
    idx = state["idx"]
    n_recipes = state["n_recipes"]
    intercept = df["recipe_key"].map(lambda rk: beta[idx[rk]] if rk in idx else np.nan).to_numpy()
    log_c = df["log_c"].to_numpy()
    return np.exp(intercept + beta[n_recipes] * log_c + beta[n_recipes + 1] * log_c**2)


def _design_mechanistic(df: pd.DataFrame) -> np.ndarray:
    # Linear (multiplicative power law) in model size and math tokens, plus a small LR offset.
    # A (log D_math)^2 curvature term was tested and dropped: it overfits the small ladder
    # (which is straight in log-log) and worsens the 1e22 extrapolation, because the
    # acceleration only appears *at* 1e22 and is not identifiable from below.
    return np.column_stack(
        [
            np.ones(len(df)),
            df["log_n"].to_numpy(),
            df["log_dmath"].to_numpy(),
            df["lr_factor_num"].to_numpy(),
        ]
    )


def _fit_pooled_mechanistic(train: pd.DataFrame) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(_design_mechanistic(train), np.log(train["value"].to_numpy()), rcond=None)
    return beta


def _predict_pooled_mechanistic(beta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    return np.exp(_design_mechanistic(df) @ beta)


def _fit_pooled_broken_power(train: pd.DataFrame) -> dict[str, Any] | None:
    """Smoothly-broken power law in compute with per-recipe amplitude, shared bend.

    ``log L = logA[recipe] - a*z - db * delta * log(1 + exp((z - logC0)/delta))`` where
    ``z = log C``. Low-C slope is ``-a``; high-C slope is ``-(a+db)``, so ``db > 0`` is the
    acceleration. The bend (``a, db, logC0, delta``) is shared across recipes for stability.
    """
    recipes = sorted(train["recipe_key"].unique())
    idx = {rk: i for i, rk in enumerate(recipes)}
    n_recipes = len(recipes)
    z = train["log_c"].to_numpy()
    y = np.log(train["value"].to_numpy())
    rows = train["recipe_key"].map(idx).to_numpy()

    def residual(params: np.ndarray) -> np.ndarray:
        log_a = params[:n_recipes]
        a, db, log_c0, delta = params[n_recipes:]
        bend = delta * np.logaddexp(0.0, (z - log_c0) / delta)
        return (log_a[rows] - a * z - db * bend) - y

    p0 = np.concatenate([np.full(n_recipes, float(np.mean(y))), [0.1, 0.05, float(np.median(z)), 1.0]])
    lower = np.concatenate([np.full(n_recipes, -10.0), [0.0, 0.0, 0.0, 0.05]])
    upper = np.concatenate([np.full(n_recipes, 10.0), [1.0, 1.0, float(np.log(2e4)), 5.0]])
    try:
        result = least_squares(residual, p0, bounds=(lower, upper), loss="huber", f_scale=0.05, max_nfev=20_000)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    return {"idx": idx, "n_recipes": n_recipes, "params": result.x}


def _predict_pooled_broken_power(state: dict[str, Any] | None, df: pd.DataFrame) -> np.ndarray:
    if state is None:
        return np.full(len(df), np.nan)
    idx = state["idx"]
    n_recipes = state["n_recipes"]
    params = state["params"]
    log_a = params[:n_recipes]
    a, db, log_c0, delta = params[n_recipes:]
    z = df["log_c"].to_numpy()
    bend = delta * np.logaddexp(0.0, (z - log_c0) / delta)
    recipe_idx = df["recipe_key"].map(lambda rk: idx.get(rk, -1)).to_numpy()
    valid = recipe_idx >= 0
    pred = np.full(len(df), np.nan)
    safe = np.clip(recipe_idx, 0, n_recipes - 1)
    pred[valid] = (log_a[safe] - a * z - db * bend)[valid]
    return np.exp(pred)


ENDPOINT_FORMS: list[EndpointForm] = [
    EndpointForm(
        "per_recipe_power", _fit_per_recipe_power, _predict_per_recipe_power, "per-recipe single power law (baseline)"
    ),
    EndpointForm(
        "per_recipe_floor", _fit_per_recipe_floor, _predict_per_recipe_floor, "per-recipe Chinchilla floor + power"
    ),
    EndpointForm(
        "pooled_curvature", _fit_pooled_curvature, _predict_pooled_curvature, "pooled log-quadratic in compute"
    ),
    EndpointForm(
        "pooled_broken_power",
        _fit_pooled_broken_power,
        _predict_pooled_broken_power,
        "pooled broken power law in compute (BNSL-style)",
    ),
    EndpointForm(
        "pooled_mechanistic",
        _fit_pooled_mechanistic,
        _predict_pooled_mechanistic,
        "pooled N x D_math + LR (separates model-size and math-token axes)",
    ),
]


def math_scaling_points(endpoints: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Complete `math_val_loss` rows (small ladder + held-out), with scaling features.

    Held-out cells that are only `best_prefix` forecasts (not observed endpoints) are
    dropped so they never enter training or scoring.
    """
    cols = ["scale", "scale_flops", "mix", "lr", "value"]
    small = endpoints[_truthy(endpoints["complete"]) & endpoints["metric_label"].eq("math_val_loss")][cols].copy()
    small["is_heldout"] = False
    if targets.empty:
        held = pd.DataFrame(columns=[*cols, "is_heldout"])
    else:
        held = targets[_truthy(targets["complete"]) & targets["metric_label"].eq("math_val_loss")][cols].copy()
        held["is_heldout"] = True
    points = pd.concat([small, held], ignore_index=True)
    points = points[points["value"] > 0]
    return attach_scaling_features(points)


def compare_forms_heldout(points: pd.DataFrame, forms: list[EndpointForm]) -> pd.DataFrame:
    """Fit each form on the small ladder and score it on the held-out 1e21/1e22 cells."""
    train = points[~points["is_heldout"]]
    held = points[points["is_heldout"]]
    rows: list[dict[str, Any]] = []
    if train.empty or held.empty:
        return pd.DataFrame(rows)
    for form in forms:
        state = form.fit(train)
        predicted = form.predict(state, held)
        for (_, row), pred in zip(held.iterrows(), predicted, strict=True):
            if not np.isfinite(pred):
                continue
            observed = float(row["value"])
            rows.append(
                {
                    "form": form.name,
                    "form_label": form.label,
                    "target_scale": row["scale"],
                    "mix": row["mix"],
                    "lr": row["lr"],
                    "observed": observed,
                    "predicted": float(pred),
                    "signed_pct_error": 100.0 * (observed - pred) / observed,
                    "abs_pct_error": abs(100.0 * (observed - pred) / observed),
                }
            )
    return pd.DataFrame(rows)


def summarize_form_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    grouped = (
        comparison.groupby(["form", "form_label", "target_scale"], observed=True)
        .agg(
            n=("abs_pct_error", "size"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            max_abs_pct_error=("abs_pct_error", "max"),
            signed_mean_pct_error=("signed_pct_error", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["target_scale", "mean_abs_pct_error"])


def rolling_origin_cv(points: pd.DataFrame, forms: list[EndpointForm]) -> pd.DataFrame:
    """Fit on every scale up to a cutoff and predict the next scale up, for each form.

    Produces an error-vs-extrapolation-distance curve that states how far each form can be
    trusted: the trustworthy horizon is the largest ``extrapolation_multiple`` whose mean
    error stays acceptable.
    """
    order = [s for s in ALL_SCALE_ORDER if (points["scale"] == s).any()]
    flops = {s: float(points.loc[points["scale"] == s, "scale_flops"].iloc[0]) for s in order}
    rows: list[dict[str, Any]] = []
    for k in range(MIN_CV_TRAIN_SCALES - 1, len(order) - 1):
        train_scales = order[: k + 1]
        test_scale = order[k + 1]
        train = points[points["scale"].isin(train_scales)]
        test = points[points["scale"] == test_scale]
        if test.empty:
            continue
        train_max_flops = max(flops[s] for s in train_scales)
        for form in forms:
            state = form.fit(train)
            predicted = form.predict(state, test)
            observed = test["value"].to_numpy()
            abs_pct = np.abs(100.0 * (observed - predicted) / observed)
            abs_pct = abs_pct[np.isfinite(abs_pct)]
            if abs_pct.size == 0:
                continue
            rows.append(
                {
                    "form": form.name,
                    "train_max_scale": train_scales[-1],
                    "test_scale": test_scale,
                    "extrapolation_multiple": flops[test_scale] / train_max_flops,
                    "n": int(abs_pct.size),
                    "mean_abs_pct_error": float(np.mean(abs_pct)),
                    "max_abs_pct_error": float(np.max(abs_pct)),
                }
            )
    return pd.DataFrame(rows)


def full_ladder_grid() -> pd.DataFrame:
    """Features for every (scale, mix, lr) cell across the full ladder, for curve overlays."""
    rows = [
        {"scale": scale, "scale_flops": ALL_SCALE_FLOPS[scale], "mix": mix, "lr": lr}
        for scale in ALL_SCALE_ORDER
        for mix in MIX_ORDER
        for lr in LR_ORDER
    ]
    return attach_scaling_features(pd.DataFrame(rows))


def select_best_form(comparison_summary: pd.DataFrame, target_scale: str = "1e22") -> str | None:
    """Form with the lowest mean abs % error at ``target_scale`` (the hardest extrapolation)."""
    if comparison_summary.empty:
        return None
    at_target = comparison_summary[comparison_summary["target_scale"].eq(target_scale)]
    if at_target.empty:
        return None
    return str(at_target.sort_values("mean_abs_pct_error").iloc[0]["form"])


def plot_endpoint_scaling(
    endpoints: pd.DataFrame,
    fits: pd.DataFrame,
    targets: pd.DataFrame,
    metric_label: str,
    output_name: str,
    overlay: pd.DataFrame | None = None,
    overlay_label: str | None = None,
) -> None:
    metric_df = endpoints[endpoints["metric_label"].eq(metric_label)].copy()
    if metric_df.empty:
        return
    metric_df["fit_status"] = np.where(metric_df["complete"], "fit", "partial")
    target_df = targets[targets["metric_label"].eq(metric_label)].copy()
    if not target_df.empty:
        target_df["fit_status"] = np.where(target_df["complete"], "heldout", "heldout prefix")
    fig = px.scatter(
        metric_df,
        x="scale_flops",
        y="value",
        color="recipe",
        symbol="fit_status",
        hover_data=["run_name", "state", "global_step", "expected_steps", "url"],
        log_x=True,
        category_orders={"scale": ALL_SCALE_ORDER, "mix": MIX_ORDER, "lr": LR_ORDER},
        title=f"Small Delphi CPT endpoint scaling + held-out extrapolation: {metric_label}",
        labels={"scale_flops": "isoflop scale", "value": metric_label},
    )
    max_scale_flops = max(SCALE_FLOPS.values())
    if not target_df.empty:
        max_scale_flops = max(max_scale_flops, float(target_df["scale_flops"].max()))
        fig.add_trace(
            go.Scatter(
                x=target_df["scale_flops"],
                y=target_df["value"],
                mode="markers",
                name="1e21/1e22 held-out targets",
                marker={"size": 12, "symbol": np.where(target_df["complete"], "diamond", "x-open")},
                customdata=target_df[
                    ["recipe", "scale", "target_kind", "global_step", "expected_steps", "url"]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]} %{customdata[2]}<br>"
                    "step=%{customdata[3]}/%{customdata[4]}<br>"
                    "loss=%{y:.5f}<extra></extra>"
                ),
            )
        )

    x_grid = np.geomspace(min(SCALE_FLOPS.values()), max_scale_flops, 240) / 1e18
    metric_fits = fits[fits["metric_label"].eq(metric_label) & fits["fit_kind"].eq("log_loss_vs_log_compute")]
    for _, row in metric_fits.iterrows():
        y_grid = row["amplitude"] * np.power(x_grid, row["exponent"])
        fig.add_trace(
            go.Scatter(
                x=x_grid * 1e18,
                y=y_grid,
                mode="lines",
                name=f"{row['mix']}-lr{row['lr']} log-linear fit",
                legendgroup=f"{row['mix']}-lr{row['lr']}",
                showlegend=False,
                hovertemplate=(
                    f"{row['mix']}-lr{row['lr']}<br>"
                    f"exponent={row['exponent']:.4f}<br>"
                    f"R2={row['r2']:.3f}<extra></extra>"
                ),
            )
        )
    if overlay is not None and not overlay.empty:
        overlay = overlay.sort_values("scale_flops")
        legend_shown = False
        for (mix, lr), grp in overlay.groupby(["mix", "lr"]):
            fig.add_trace(
                go.Scatter(
                    x=grp["scale_flops"],
                    y=grp["predicted"],
                    mode="lines",
                    line={"dash": "dash", "width": 1.5},
                    name=overlay_label or "best form",
                    legendgroup="overlay",
                    showlegend=not legend_shown,
                    hovertemplate=(f"{mix}-lr{lr} {overlay_label or 'best form'}<br>loss=%{{y:.5f}}<extra></extra>"),
                )
            )
            legend_shown = True
    fig.add_vline(
        x=max(SCALE_FLOPS.values()),
        line_dash="dot",
        line_color="#666",
        annotation_text="fit max",
        annotation_position="top left",
    )
    fig.update_layout(height=720, legend_title_text="recipe")
    fig.write_html(OUT_DIR / output_name, include_plotlyjs="cdn")


def write_summary(
    endpoints: pd.DataFrame,
    fits: pd.DataFrame,
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    cv: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Small Delphi Final-Loss Scaling First Pass",
        "",
        "Endpoint rows are W&B summary values from the selected non-probe attempt for each cell.",
        "Fits use completed cells only; partial/running cells are kept in `endpoints.csv` but excluded from fits.",
        "Held-out `1e21`/`1e22` targets are live W&B summary rows when available, with the "
        "prior local Delphi dump used only as a fallback. They are never used for fitting.",
        "",
    ]
    complete_cells = endpoints[["scale", "mix", "lr", "complete"]].drop_duplicates()
    complete_cells["scale"] = pd.Categorical(complete_cells["scale"], categories=SCALE_ORDER, ordered=True)
    counts = complete_cells.groupby("scale", observed=True)["complete"].agg(["sum", "count"]).reset_index()
    lines.extend(["## Coverage", ""])
    for _, row in counts.iterrows():
        lines.append(f"- `{row['scale']}`: {int(row['sum'])}/{int(row['count'])} complete selected cells")

    if not targets.empty:
        target_cells = targets[["scale", "mix", "lr", "complete"]].drop_duplicates()
        target_cells["scale"] = pd.Categorical(target_cells["scale"], categories=HELDOUT_SCALE_ORDER, ordered=True)
        target_counts = target_cells.groupby("scale", observed=True)["complete"].agg(["sum", "count"]).reset_index()
        lines.extend(["", "## Held-Out Target Coverage", ""])
        for _, row in target_counts.iterrows():
            lines.append(f"- `{row['scale']}`: {int(row['sum'])}/{int(row['count'])} complete-like targets")

    math_fits = fits[fits["metric_label"].eq("math_val_loss") & fits["fit_kind"].eq("log_loss_vs_log_compute")].copy()
    if not math_fits.empty:
        math_fits = math_fits.sort_values("r2", ascending=False)
        lines.extend(
            [
                "",
                "## Math Validation Log-Linear Fits",
                "",
                "| recipe | n | scales | exponent | R2 | RMSE | LOOCV RMSE | monotone | first -> last |",
                "|---|---:|---|---:|---:|---:|---:|---|---:|",
            ]
        )
        for _, row in math_fits.iterrows():
            recipe = f"{row['mix']}-lr{row['lr']}"
            lines.append(
                "| "
                f"{recipe} | {int(row['n'])} | {row['scales']} | "
                f"{row['exponent']:.4f} | {row['r2']:.3f} | {row['rmse']:.4f} | "
                f"{row['loocv_rmse']:.4f} | {row['monotone_nonincreasing']} | "
                f"{row['first_value']:.4f} -> {row['last_value']:.4f} |"
            )

    floor_fits = fits[fits["metric_label"].eq("math_val_loss") & fits["fit_kind"].eq("floor_plus_power")].copy()
    if not floor_fits.empty:
        floor_fits = floor_fits.sort_values("r2", ascending=False)
        lines.extend(
            [
                "",
                "## Math Validation Floor + Power Diagnostics",
                "",
                "These 3-parameter fits are diagnostic only with 5-6 scales; the floor can be unstable.",
                "",
                "| recipe | n | floor | exponent alpha | R2 | RMSE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in floor_fits.iterrows():
            recipe = f"{row['mix']}-lr{row['lr']}"
            lines.append(
                f"| {recipe} | {int(row['n'])} | {row['floor']:.4f} | "
                f"{row['exponent']:.4f} | {row['r2']:.3f} | {row['rmse']:.4f} |"
            )

    math_predictions = predictions[
        predictions["metric_label"].eq("math_val_loss") & predictions["fit_kind"].eq("log_loss_vs_log_compute")
    ].copy()
    if not math_predictions.empty:
        math_predictions = math_predictions.sort_values(
            ["target_scale_flops", "target_complete", "abs_error", "mix", "lr"],
            ascending=[True, False, True, True, True],
        )
        lines.extend(
            [
                "",
                "## Math Validation Held-Out Extrapolation",
                "",
                "These rows fit only the small ladder, then evaluate at `1e21`/`1e22`.",
                "",
                "| target | recipe | status | observed | predicted | error | pct error | train max |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in math_predictions.iterrows():
            lines.append(
                "| "
                f"{row['target_scale']} | {row['recipe']} | {row['target_kind']} | "
                f"{row['observed']:.4f} | {row['predicted']:.4f} | {row['error']:.4f} | "
                f"{row['pct_error']:.2f}% | {row['train_max_scale']} |"
            )

    if not comparison_summary.empty:
        lines.extend(
            [
                "",
                "## Math Validation Form Comparison On Held-Out",
                "",
                "Each form is fit on the small ladder (`3e18`-`3e20`) and scored on the observed "
                "`1e21`/`1e22` cells (the one `best_prefix` 1e22 forecast cell is excluded). Lower "
                "mean abs % error is better; signed mean % < 0 means the form is over-pessimistic "
                "(predicts a higher loss than observed).",
                "",
                "| form | target | n | mean abs % | max abs % | signed mean % |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in comparison_summary.iterrows():
            lines.append(
                f"| {row['form']} | {row['target_scale']} | {int(row['n'])} | "
                f"{row['mean_abs_pct_error']:.2f}% | {row['max_abs_pct_error']:.2f}% | "
                f"{row['signed_mean_pct_error']:+.2f}% |"
            )

    if not cv.empty:
        baseline_horizon = cv[cv["form"].eq(FORM_BASELINE)].sort_values("extrapolation_multiple")
        lines.extend(
            [
                "",
                "## Trustworthy Extrapolation Horizon",
                "",
                "Rolling-origin CV: fit on every scale up to a cutoff, predict the next scale up. "
                "`extrapolation_multiple` is the FLOPs jump from the training max to the test scale.",
                "",
                "| form | train max | test | x multiple | mean abs % | max abs % |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for _, row in cv.sort_values(["form", "extrapolation_multiple"]).iterrows():
            lines.append(
                f"| {row['form']} | {row['train_max_scale']} | {row['test_scale']} | "
                f"{row['extrapolation_multiple']:.1f}x | {row['mean_abs_pct_error']:.2f}% | "
                f"{row['max_abs_pct_error']:.2f}% |"
            )
        if not baseline_horizon.empty:
            within_1pct = baseline_horizon[baseline_horizon["mean_abs_pct_error"] <= 2.0]
            safe_mult = within_1pct["extrapolation_multiple"].max() if not within_1pct.empty else float("nan")
            lines.extend(
                [
                    "",
                    f"Baseline (`{FORM_BASELINE}`) stays within ~2% mean error out to about "
                    f"{safe_mult:.0f}x the training-max compute, then degrades. Read predictions "
                    "beyond that multiple as directional, not precise.",
                ]
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `endpoints.csv`: selected endpoint metric values.",
            "- `fit_summary.csv`: per-recipe fit parameters and errors.",
            "- `extrapolation_targets.csv`: observed `1e21`/`1e22` held-out target endpoints.",
            "- `extrapolation_predictions.csv`: predictions and errors from small-ladder fits evaluated "
            "on held-out targets.",
            "- `endpoint_form_comparison.csv` / `endpoint_form_comparison_summary.csv`: per-cell and "
            "aggregated held-out errors for every candidate form.",
            "- `endpoint_cv_by_distance.csv`: rolling-origin CV error vs extrapolation distance per form.",
            "- `endpoint_math_val_loss.html`: scatter + log-linear fit overlay for the held-out math validation loss.",
            "- `endpoint_eval_loss.html`: same for aggregate eval/loss.",
            "- `endpoint_paloma_macro_loss.html`: same for Paloma macro retention.",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    endpoints_path = OUT_DIR / "endpoints.csv"
    if args.use_cache and endpoints_path.exists():
        endpoints = pd.read_csv(endpoints_path, dtype={"scale": str, "lr": str})
    else:
        run_endpoints = fetch_run_endpoints(args.project)
        endpoints = endpoints_to_long(run_endpoints)
        endpoints.to_csv(endpoints_path, index=False)
        (OUT_DIR / "selected_runs.json").write_text(
            json.dumps([endpoint.__dict__ for endpoint in run_endpoints], indent=2, default=str) + "\n"
        )

    fits = fit_groups(endpoints)
    fits.to_csv(OUT_DIR / "fit_summary.csv", index=False)

    targets = load_heldout_targets(args.project)
    targets.to_csv(OUT_DIR / "extrapolation_targets.csv", index=False)
    predictions = predict_heldout_targets(fits, targets)
    predictions.to_csv(OUT_DIR / "extrapolation_predictions.csv", index=False)

    # Multi-axis / curvature forms: head-to-head on held-out + rolling-origin CV (math only).
    points = math_scaling_points(endpoints, targets)
    comparison = compare_forms_heldout(points, ENDPOINT_FORMS)
    comparison.to_csv(OUT_DIR / "endpoint_form_comparison.csv", index=False)
    comparison_summary = summarize_form_comparison(comparison)
    comparison_summary.to_csv(OUT_DIR / "endpoint_form_comparison_summary.csv", index=False)
    cv = rolling_origin_cv(points, ENDPOINT_FORMS)
    cv.to_csv(OUT_DIR / "endpoint_cv_by_distance.csv", index=False)

    overlay: pd.DataFrame | None = None
    overlay_label: str | None = None
    best_form_name = select_best_form(comparison_summary)
    if best_form_name is not None:
        best_form = next(form for form in ENDPOINT_FORMS if form.name == best_form_name)
        train_points = points[~points["is_heldout"]]
        state = best_form.fit(train_points)
        grid = full_ladder_grid()
        grid["predicted"] = best_form.predict(state, grid)
        overlay = grid.dropna(subset=["predicted"])
        overlay_label = f"{best_form.name} fit"
        logger.info("Best held-out form at 1e22: %s", best_form_name)

    plot_endpoint_scaling(
        endpoints, fits, targets, "math_val_loss", "endpoint_math_val_loss.html", overlay, overlay_label
    )
    plot_endpoint_scaling(endpoints, fits, targets, "eval_loss", "endpoint_eval_loss.html")
    plot_endpoint_scaling(endpoints, fits, targets, "paloma_macro_loss", "endpoint_paloma_macro_loss.html")
    write_summary(endpoints, fits, targets, predictions, comparison_summary, cv)

    complete_cells = endpoints[["scale", "mix", "lr", "complete"]].drop_duplicates()
    complete_cells["scale"] = pd.Categorical(complete_cells["scale"], categories=SCALE_ORDER, ordered=True)
    print("Coverage by scale:")
    print(complete_cells.groupby("scale", observed=True)["complete"].agg(["sum", "count"]).to_string())
    if not targets.empty:
        target_cells = targets[["scale", "mix", "lr", "complete"]].drop_duplicates()
        target_cells["scale"] = pd.Categorical(target_cells["scale"], categories=HELDOUT_SCALE_ORDER, ordered=True)
        print("\nHeld-out target coverage by scale:")
        print(target_cells.groupby("scale", observed=True)["complete"].agg(["sum", "count"]).to_string())
    print(f"\nWrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
