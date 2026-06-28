# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a K=0.20-only fit-family report for the clean-seen validation sweep.

The input is the completed p33m67 K=0.20 clean-seen eval summary:

    gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv

Default split matches the other Delphi reports: fit through 3e20 and hold out
1e21/1e22. The report compares per-LR extrapolation fits, pooled LR-aware fits,
and calibration fits that map the original 4plus math val loss to clean-seen
loss.

Run:
    uv run --with scipy --with plotly --with pandas --with gcsfs \
      python scripts/analysis/delphi_k020_clean_seen_fit_family_report.py
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from build_delphi_midtraining_interactive_report import finite_or_none
from delphi_isotoken_endpoint_scaling import ALL_SCALE_FLOPS, DEFAULT_CUTOFF_SCALE, HELD_OUT_SCALES, SCALE_ORDER
from delphi_small_final_loss_scaling import (
    MATH_FRACTION,
    MIDTRAIN_BUDGET_FRACTION,
    SCALE_PARAMS_B,
    SCALE_PRETRAIN_TOKENS_B,
)
from marin.scaling_laws.scaling_plots import MARKERS, PALETTE
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

DEFAULT_INPUT = (
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv"
)
DEFAULT_OUTPUT_DIR = Path("sk_midtrain_analysis_fable")
DEFAULT_OUTPUT_STEM = "delphi_k020_clean_seen_fit_family_report"
TARGET_METRIC = "clean_seen_loss"
ANCHOR_METRIC = "anchor_4plus_loss"
MIX = "p33m67"
LR_ORDER = ("0.33", "0.50", "0.67", "0.83")
LR_LABELS = {"0.33": "lr0.33", "0.50": "lr0.50", "0.67": "lr0.67", "0.83": "lr0.83"}


@dataclass(frozen=True)
class FitSpec:
    key: str
    label: str
    family: str
    scale_feature: str
    description: str


@dataclass(frozen=True)
class FittedModel:
    spec: FitSpec
    parameters: dict[str, Any]
    train_n: int
    fit_r2: float | None
    fit_rmse: float


@dataclass(frozen=True)
class FailedFit:
    spec: FitSpec
    error: str


RESOURCE_LABELS = {
    "pretrain_flops_e18": "C_pre",
    "params_b": "N",
    "pretrain_tokens_b": "D_pre",
    "dmath_b": "D_math",
    "midtrain_math_flops_e18": "C_mid_math",
    ANCHOR_METRIC: "anchor 4plus loss",
}

RESOURCE_DESCRIPTIONS = {
    "C_pre": "base pretraining FLOPs, normalized to 1e18 FLOPs",
    "N": "trainable parameters in billions",
    "D_pre": "pretraining tokens in billions",
    "D_math": "math midtraining tokens in billions; for p33m67 K=0.20 this is 0.67 * 0.20 * D_pre",
    "C_mid_math": "estimated math-component midtraining FLOPs: C_pre * D_math / D_pre",
    "anchor 4plus loss": "the original eval/nemotron_cc_math_v1/4plus/loss measured in the same eval job",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV path or gs:// URI for the clean-seen summary.")
    parser.add_argument(
        "--fit-through-scale",
        choices=SCALE_ORDER[:-1],
        default=DEFAULT_CUTOFF_SCALE,
        help="Largest scale included in the training split.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def scale_label_for(value: Any) -> str:
    if isinstance(value, str) and value in ALL_SCALE_FLOPS:
        return value
    numeric = float(value)
    for label, flops in ALL_SCALE_FLOPS.items():
        if math.isclose(numeric, flops, rel_tol=1e-9):
            return label
    raise ValueError(f"Unknown scale value: {value!r}")


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_lr(value: Any) -> str:
    numeric = float(value)
    return f"{numeric:.2f}"


def load_clean_seen_summary(path: str) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"scale", "lr_factor", "run", "step", TARGET_METRIC, ANCHOR_METRIC}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    out = frame.copy()
    out["scale"] = out["scale"].map(scale_label_for)
    out["scale_flops"] = out["scale"].map(ALL_SCALE_FLOPS).astype(float)
    out["lr"] = out["lr_factor"].map(normalize_lr)
    out["lr_numeric"] = out["lr"].astype(float)
    out["params_b"] = out["scale"].map(SCALE_PARAMS_B).astype(float)
    out["pretrain_tokens_b"] = out["scale"].map(SCALE_PRETRAIN_TOKENS_B).astype(float)
    out["midtrain_tokens_b"] = out["pretrain_tokens_b"] * MIDTRAIN_BUDGET_FRACTION
    out["math_fraction"] = MATH_FRACTION[MIX]
    out["dmath_b"] = out["midtrain_tokens_b"] * out["math_fraction"]
    out["pretrain_flops_e18"] = out["scale_flops"] / 1e18
    out["midtrain_math_flops_e18"] = out["pretrain_flops_e18"] * out["dmath_b"] / out["pretrain_tokens_b"]
    out["split"] = np.where(out["scale"].isin(HELD_OUT_SCALES), "heldout", "train")
    return out.sort_values(["scale_flops", "lr_numeric"]).reset_index(drop=True)


def split_points(points: pd.DataFrame, fit_through_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = ALL_SCALE_FLOPS[fit_through_scale]
    train = points[points["scale_flops"] <= cutoff + 1.0].copy()
    heldout = points[points["scale"].isin(HELD_OUT_SCALES) & (points["scale_flops"] > cutoff + 1.0)].copy()
    if train.empty:
        raise ValueError(f"No training rows at or below {fit_through_scale}")
    if heldout.empty:
        raise ValueError(f"No heldout rows above {fit_through_scale}")
    return train, heldout


def floor_power_model(x: np.ndarray, floor: float, amplitude: float, exponent: float) -> np.ndarray:
    return floor + amplitude * np.power(x, -exponent)


def pooled_power_lr_offset_model(
    features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    floor: float,
    amplitude: float,
    exponent: float,
    offset_lr50: float,
    offset_lr67: float,
    offset_lr83: float,
) -> np.ndarray:
    x, lr50, lr67, lr83 = features
    return floor + amplitude * np.power(x, -exponent) + offset_lr50 * lr50 + offset_lr67 * lr67 + offset_lr83 * lr83


def pooled_power_lr_quadratic_model(
    features: tuple[np.ndarray, np.ndarray],
    floor: float,
    amplitude: float,
    exponent: float,
    lr_linear: float,
    lr_quadratic: float,
) -> np.ndarray:
    x, lr = features
    return floor + amplitude * np.power(x, -exponent) + lr_linear * lr + lr_quadratic * np.square(lr)


def pooled_multi_power_lr_quadratic_model(
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
    floor: float,
    params_amplitude: float,
    params_exponent: float,
    data_amplitude: float,
    data_exponent: float,
    lr_linear: float,
    lr_quadratic: float,
) -> np.ndarray:
    params_b, pretrain_tokens_b, lr = features
    return (
        floor
        + params_amplitude * np.power(params_b, -params_exponent)
        + data_amplitude * np.power(pretrain_tokens_b, -data_exponent)
        + lr_linear * lr
        + lr_quadratic * np.square(lr)
    )


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


def model_quality(spec: FitSpec, parameters: dict[str, Any], train: pd.DataFrame) -> FittedModel:
    actual = train[TARGET_METRIC].to_numpy(dtype=float)
    predicted = predict_model(spec, parameters, train)
    return FittedModel(
        spec=spec,
        parameters=parameters,
        train_n=len(train),
        fit_r2=r2_score(actual, predicted),
        fit_rmse=math.sqrt(float(np.mean((actual - predicted) ** 2))),
    )


def fit_per_lr_power(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    params_by_lr: dict[str, dict[str, float]] = {}
    for lr, group in train.groupby("lr", sort=False):
        x = group[spec.scale_feature].to_numpy(dtype=float)
        y = group[TARGET_METRIC].to_numpy(dtype=float)
        spread = max(float(y.max() - y.min()), 1e-3)
        initial = (float(y.min()) * 0.35, spread, 0.1)
        fitted, _ = curve_fit(
            floor_power_model,
            x,
            y,
            p0=initial,
            bounds=([0.0, 0.0, 0.0], [float(y.min()) * 0.999, np.inf, 5.0]),
            maxfev=100_000,
        )
        params_by_lr[str(lr)] = {
            "floor": float(fitted[0]),
            "amplitude": float(fitted[1]),
            "exponent": float(fitted[2]),
        }
    return model_quality(spec, {"by_lr": params_by_lr}, train)


def fit_per_lr_loglog(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    params_by_lr: dict[str, dict[str, float]] = {}
    for lr, group in train.groupby("lr", sort=False):
        x = group[spec.scale_feature].to_numpy(dtype=float)
        y = group[TARGET_METRIC].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), np.log(x)])
        fitted, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
        params_by_lr[str(lr)] = {"intercept": float(fitted[0]), "slope": float(fitted[1])}
    return model_quality(spec, {"by_lr": params_by_lr}, train)


def fit_pooled_power_lr_offsets(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    x = train[spec.scale_feature].to_numpy(dtype=float)
    y = train[TARGET_METRIC].to_numpy(dtype=float)
    lr = train["lr"].astype(str)
    spread = max(float(y.max() - y.min()), 1e-3)
    initial = (float(y.min()) * 0.35, spread, 0.1, 0.0, 0.0, 0.0)
    lower = (0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf)
    upper = (float(y.min()) * 0.999, np.inf, 5.0, np.inf, np.inf, np.inf)
    fitted, _ = curve_fit(
        pooled_power_lr_offset_model,
        (
            x,
            lr.eq("0.50").astype(float).to_numpy(),
            lr.eq("0.67").astype(float).to_numpy(),
            lr.eq("0.83").astype(float).to_numpy(),
        ),
        y,
        p0=initial,
        bounds=(lower, upper),
        maxfev=100_000,
    )
    parameters = {
        "floor": float(fitted[0]),
        "amplitude": float(fitted[1]),
        "exponent": float(fitted[2]),
        "offset_lr50": float(fitted[3]),
        "offset_lr67": float(fitted[4]),
        "offset_lr83": float(fitted[5]),
    }
    return model_quality(spec, parameters, train)


def fit_pooled_power_lr_quadratic(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    x = train[spec.scale_feature].to_numpy(dtype=float)
    lr = train["lr_numeric"].to_numpy(dtype=float)
    y = train[TARGET_METRIC].to_numpy(dtype=float)
    spread = max(float(y.max() - y.min()), 1e-3)
    initial = (float(y.min()) * 0.35, spread, 0.1, 0.0, 0.0)
    lower = (0.0, 0.0, 0.0, -np.inf, -np.inf)
    upper = (float(y.min()) * 0.999, np.inf, 5.0, np.inf, np.inf)
    fitted, _ = curve_fit(
        pooled_power_lr_quadratic_model,
        (x, lr),
        y,
        p0=initial,
        bounds=(lower, upper),
        maxfev=100_000,
    )
    parameters = {
        "floor": float(fitted[0]),
        "amplitude": float(fitted[1]),
        "exponent": float(fitted[2]),
        "lr_linear": float(fitted[3]),
        "lr_quadratic": float(fitted[4]),
    }
    return model_quality(spec, parameters, train)


def fit_pooled_loglog_lr_quadratic(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    x = train[spec.scale_feature].to_numpy(dtype=float)
    lr = train["lr_numeric"].to_numpy(dtype=float)
    y = train[TARGET_METRIC].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), np.log(x), lr, np.square(lr)])
    fitted, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
    parameters = {
        "intercept": float(fitted[0]),
        "slope": float(fitted[1]),
        "lr_linear": float(fitted[2]),
        "lr_quadratic": float(fitted[3]),
    }
    return model_quality(spec, parameters, train)


def fit_pooled_multi_power_lr_quadratic(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    params_b = train["params_b"].to_numpy(dtype=float)
    pretrain_tokens_b = train["pretrain_tokens_b"].to_numpy(dtype=float)
    lr = train["lr_numeric"].to_numpy(dtype=float)
    y = train[TARGET_METRIC].to_numpy(dtype=float)
    spread = max(float(y.max() - y.min()), 1e-3)
    initial = (float(y.min()) * 0.35, spread / 2.0, 0.1, spread / 2.0, 0.1, 0.0, 0.0)
    lower = (0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf)
    upper = (float(y.min()) * 0.999, np.inf, 5.0, np.inf, 5.0, np.inf, np.inf)
    fitted, _ = curve_fit(
        pooled_multi_power_lr_quadratic_model,
        (params_b, pretrain_tokens_b, lr),
        y,
        p0=initial,
        bounds=(lower, upper),
        maxfev=200_000,
    )
    parameters = {
        "floor": float(fitted[0]),
        "params_amplitude": float(fitted[1]),
        "params_exponent": float(fitted[2]),
        "pretrain_data_amplitude": float(fitted[3]),
        "pretrain_data_exponent": float(fitted[4]),
        "lr_linear": float(fitted[5]),
        "lr_quadratic": float(fitted[6]),
    }
    return model_quality(spec, parameters, train)


def fit_anchor_linear(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    anchor = train[ANCHOR_METRIC].to_numpy(dtype=float)
    y = train[TARGET_METRIC].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(anchor), anchor])
    fitted, *_ = np.linalg.lstsq(design, y, rcond=None)
    parameters = {"intercept": float(fitted[0]), "anchor_slope": float(fitted[1])}
    return model_quality(spec, parameters, train)


def fit_anchor_scale_lr(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    anchor = train[ANCHOR_METRIC].to_numpy(dtype=float)
    x = train["pretrain_flops_e18"].to_numpy(dtype=float)
    lr = train["lr_numeric"].to_numpy(dtype=float)
    y = train[TARGET_METRIC].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(anchor), anchor, np.log(x), lr, np.square(lr)])
    fitted, *_ = np.linalg.lstsq(design, y, rcond=None)
    parameters = {
        "intercept": float(fitted[0]),
        "anchor_slope": float(fitted[1]),
        "log_scale_slope": float(fitted[2]),
        "lr_linear": float(fitted[3]),
        "lr_quadratic": float(fitted[4]),
    }
    return model_quality(spec, parameters, train)


def fit_model(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    if spec.family == "per_lr_power":
        return fit_per_lr_power(spec, train)
    if spec.family == "per_lr_loglog":
        return fit_per_lr_loglog(spec, train)
    if spec.family == "pooled_power_lr_offsets":
        return fit_pooled_power_lr_offsets(spec, train)
    if spec.family == "pooled_power_lr_quadratic":
        return fit_pooled_power_lr_quadratic(spec, train)
    if spec.family == "pooled_loglog_lr_quadratic":
        return fit_pooled_loglog_lr_quadratic(spec, train)
    if spec.family == "pooled_multi_power_lr_quadratic":
        return fit_pooled_multi_power_lr_quadratic(spec, train)
    if spec.family == "anchor_linear":
        return fit_anchor_linear(spec, train)
    if spec.family == "anchor_scale_lr":
        return fit_anchor_scale_lr(spec, train)
    raise ValueError(f"Unknown fit family: {spec.family}")


def predict_model(spec: FitSpec, parameters: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if spec.family == "per_lr_power":
        rows = []
        for _, row in frame.iterrows():
            lr_params = parameters["by_lr"][str(row["lr"])]
            rows.append(
                floor_power_model(
                    np.asarray([float(row[spec.scale_feature])]),
                    lr_params["floor"],
                    lr_params["amplitude"],
                    lr_params["exponent"],
                )[0]
            )
        return np.asarray(rows, dtype=float)
    if spec.family == "per_lr_loglog":
        rows = []
        for _, row in frame.iterrows():
            lr_params = parameters["by_lr"][str(row["lr"])]
            rows.append(math.exp(lr_params["intercept"] + lr_params["slope"] * math.log(float(row[spec.scale_feature]))))
        return np.asarray(rows, dtype=float)
    if spec.family == "pooled_power_lr_offsets":
        lr = frame["lr"].astype(str)
        return pooled_power_lr_offset_model(
            (
                frame[spec.scale_feature].to_numpy(dtype=float),
                lr.eq("0.50").astype(float).to_numpy(),
                lr.eq("0.67").astype(float).to_numpy(),
                lr.eq("0.83").astype(float).to_numpy(),
            ),
            parameters["floor"],
            parameters["amplitude"],
            parameters["exponent"],
            parameters["offset_lr50"],
            parameters["offset_lr67"],
            parameters["offset_lr83"],
        )
    if spec.family == "pooled_power_lr_quadratic":
        return pooled_power_lr_quadratic_model(
            (frame[spec.scale_feature].to_numpy(dtype=float), frame["lr_numeric"].to_numpy(dtype=float)),
            parameters["floor"],
            parameters["amplitude"],
            parameters["exponent"],
            parameters["lr_linear"],
            parameters["lr_quadratic"],
        )
    if spec.family == "pooled_loglog_lr_quadratic":
        x = frame[spec.scale_feature].to_numpy(dtype=float)
        lr = frame["lr_numeric"].to_numpy(dtype=float)
        return np.exp(
            parameters["intercept"]
            + parameters["slope"] * np.log(x)
            + parameters["lr_linear"] * lr
            + parameters["lr_quadratic"] * np.square(lr)
        )
    if spec.family == "pooled_multi_power_lr_quadratic":
        return pooled_multi_power_lr_quadratic_model(
            (
                frame["params_b"].to_numpy(dtype=float),
                frame["pretrain_tokens_b"].to_numpy(dtype=float),
                frame["lr_numeric"].to_numpy(dtype=float),
            ),
            parameters["floor"],
            parameters["params_amplitude"],
            parameters["params_exponent"],
            parameters["pretrain_data_amplitude"],
            parameters["pretrain_data_exponent"],
            parameters["lr_linear"],
            parameters["lr_quadratic"],
        )
    if spec.family == "anchor_linear":
        return parameters["intercept"] + parameters["anchor_slope"] * frame[ANCHOR_METRIC].to_numpy(dtype=float)
    if spec.family == "anchor_scale_lr":
        anchor = frame[ANCHOR_METRIC].to_numpy(dtype=float)
        x = frame["pretrain_flops_e18"].to_numpy(dtype=float)
        lr = frame["lr_numeric"].to_numpy(dtype=float)
        return (
            parameters["intercept"]
            + parameters["anchor_slope"] * anchor
            + parameters["log_scale_slope"] * np.log(x)
            + parameters["lr_linear"] * lr
            + parameters["lr_quadratic"] * np.square(lr)
        )
    raise ValueError(f"Unknown fit family: {spec.family}")


def fit_specs() -> list[FitSpec]:
    scale_features = ("pretrain_flops_e18", "params_b", "pretrain_tokens_b", "dmath_b", "midtrain_math_flops_e18")
    specs: list[FitSpec] = []
    for feature in scale_features:
        label = RESOURCE_LABELS[feature]
        specs.extend(
            [
                FitSpec(
                    key=f"per_lr_power_{feature}",
                    label=f"per-LR Chinchilla: {label}",
                    family="per_lr_power",
                    scale_feature=feature,
                    description="Independent floor+power fit for each LR; no information shared across LR curves.",
                ),
                FitSpec(
                    key=f"per_lr_loglog_{feature}",
                    label=f"per-LR log-log: {label}",
                    family="per_lr_loglog",
                    scale_feature=feature,
                    description="Independent pure power-law fit for each LR; no floor.",
                ),
                FitSpec(
                    key=f"pooled_power_lr_offsets_{feature}",
                    label=f"pooled Chinchilla + LR offsets: {label}",
                    family="pooled_power_lr_offsets",
                    scale_feature=feature,
                    description="One floor+power curve shared across LRs plus additive LR-specific offsets.",
                ),
                FitSpec(
                    key=f"pooled_power_lr_quadratic_{feature}",
                    label=f"pooled Chinchilla + LR quadratic: {label}",
                    family="pooled_power_lr_quadratic",
                    scale_feature=feature,
                    description="One floor+power curve with continuous linear/quadratic LR terms.",
                ),
                FitSpec(
                    key=f"pooled_loglog_lr_quadratic_{feature}",
                    label=f"pooled log-log + LR quadratic: {label}",
                    family="pooled_loglog_lr_quadratic",
                    scale_feature=feature,
                    description="Pure power law with continuous linear/quadratic LR terms.",
                ),
            ]
        )
    specs.append(
        FitSpec(
            key="pooled_multi_power_lr_quadratic_N_Dpre",
            label="pooled Chinchilla + LR quadratic: N + D_pre",
            family="pooled_multi_power_lr_quadratic",
            scale_feature="params_b",
            description="Chinchilla-style N and D_pre terms plus continuous linear/quadratic LR terms.",
        )
    )
    specs.extend(
        [
            FitSpec(
                key="anchor_linear",
                label="anchor calibration: clean = a + b * anchor",
                family="anchor_linear",
                scale_feature=ANCHOR_METRIC,
                description="Calibration from original 4plus loss to clean-seen loss using same-checkpoint anchor loss.",
            ),
            FitSpec(
                key="anchor_scale_lr",
                label="anchor calibration + scale/LR",
                family="anchor_scale_lr",
                scale_feature=ANCHOR_METRIC,
                description="Linear calibration using anchor loss, log scale, LR, and LR^2.",
            ),
        ]
    )
    return specs


def fit_all(
    points: pd.DataFrame, fit_through_scale: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[FittedModel], list[FailedFit]]:
    train, _ = split_points(points, fit_through_scale)
    prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    models: list[FittedModel] = []
    failures: list[FailedFit] = []
    for spec in fit_specs():
        try:
            model = fit_model(spec, train)
            predictions = prediction_frame(model, points)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(FailedFit(spec=spec, error=str(exc)))
            continue
        models.append(model)
        prediction_frames.append(predictions)
        fit_rows.append(
            {
                "model_key": spec.key,
                "model_label": spec.label,
                "family": spec.family,
                "resource": RESOURCE_LABELS.get(spec.scale_feature, spec.scale_feature),
                "train_n": model.train_n,
                "fit_r2": model.fit_r2,
                "fit_rmse": model.fit_rmse,
                "parameters": json.dumps(model.parameters, sort_keys=True),
            }
        )
    if not prediction_frames:
        raise ValueError("No fits succeeded")
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(fit_rows), models, failures


def prediction_frame(model: FittedModel, points: pd.DataFrame) -> pd.DataFrame:
    rows = points.copy()
    rows["model_key"] = model.spec.key
    rows["model_label"] = model.spec.label
    rows["family"] = model.spec.family
    rows["resource"] = RESOURCE_LABELS.get(model.spec.scale_feature, model.spec.scale_feature)
    rows["prediction"] = predict_model(model.spec, model.parameters, rows)
    rows["error"] = rows["prediction"] - rows[TARGET_METRIC]
    rows["error_pct"] = (rows["prediction"] / rows[TARGET_METRIC] - 1.0) * 100.0
    rows["abs_error_pct"] = rows["error_pct"].abs()
    return rows.sort_values(["model_key", "split", "scale_flops", "lr_numeric"]).reset_index(drop=True)


def summarize_predictions(predictions: pd.DataFrame, fit_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_key, model_label, family, resource), frame in predictions.groupby(
        ["model_key", "model_label", "family", "resource"],
        sort=False,
    ):
        heldout = frame[frame["split"].eq("heldout")]
        one_e22 = heldout[heldout["scale"].eq("1e22")]
        fit_meta = fit_table[fit_table["model_key"].eq(model_key)].iloc[0]
        errors = heldout["error_pct"].to_numpy(dtype=float)
        loss_errors = heldout["error"].to_numpy(dtype=float)
        one_e22_errors = one_e22["error_pct"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "model_key": model_key,
            "model_label": model_label,
            "family": family,
            "resource": resource,
            "train_n": int(fit_meta["train_n"]),
            "heldout_n": len(heldout),
            "heldout_mae_pct": float(np.mean(np.abs(errors))),
            "heldout_rmse_pct": math.sqrt(float(np.mean(errors**2))),
            "heldout_bias_pct": float(np.mean(errors)),
            "heldout_loss_rmse": math.sqrt(float(np.mean(loss_errors**2))),
            "1e22_mae_pct": float(np.mean(np.abs(one_e22_errors))),
            "1e22_rmse_pct": math.sqrt(float(np.mean(one_e22_errors**2))),
            "1e22_bias_pct": float(np.mean(one_e22_errors)),
            "fit_r2": None if pd.isna(fit_meta["fit_r2"]) else float(fit_meta["fit_r2"]),
            "fit_rmse": float(fit_meta["fit_rmse"]),
        }
        for lr in LR_ORDER:
            cell = one_e22[one_e22["lr"].eq(lr)]
            if cell.empty:
                continue
            label = lr.replace(".", "")
            row[f"1e22_{label}_actual"] = float(cell[TARGET_METRIC].iloc[0])
            row[f"1e22_{label}_prediction"] = float(cell["prediction"].iloc[0])
            row[f"1e22_{label}_error_pct"] = float(cell["error_pct"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["heldout_mae_pct", "1e22_mae_pct", "model_label"]).reset_index(drop=True)


def one_e22_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions[predictions["scale"].eq("1e22")].copy()
    return (
        rows[
            [
                "model_label",
                "family",
                "resource",
                "lr",
                TARGET_METRIC,
                "prediction",
                "error_pct",
                "abs_error_pct",
                ANCHOR_METRIC,
                "run",
            ]
        ]
        .sort_values(["abs_error_pct", "lr", "model_label"])
        .reset_index(drop=True)
    )


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    return [{key: finite_or_none(value) for key, value in record.items()} for record in records]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(str(path), "w") as handle:
        handle.write(text)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_text(path, frame.to_csv(index=False))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, default=finite_or_none))


def format_float(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def sort_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        return html.escape(value.lower(), quote=True)
    return html.escape(str(float(value)), quote=True)


def table_html(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, int, str]],
    table_class: str = "sortable",
    default_sort_key: str | None = None,
    default_sort_direction: str = "asc",
) -> str:
    default_sort_index = next((index for index, (key, _, _, _) in enumerate(columns) if key == default_sort_key), None)
    default_sort_attrs = ""
    if default_sort_index is not None:
        default_sort_attrs = (
            f' data-default-sort-index="{default_sort_index}"'
            f' data-default-sort-direction="{html.escape(default_sort_direction, quote=True)}"'
        )
    class_attr = f' class="{html.escape(table_class, quote=True)}"' if table_class else ""
    header = "".join(
        f'<th data-sort-index="{index}"><button type="button">{html.escape(label)}'
        '<span class="sort-indicator"></span></button></th>'
        for index, (_, label, _, _) in enumerate(columns)
    )
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, digits, suffix in columns:
            value = row.get(key)
            if isinstance(value, str):
                rendered = html.escape(value)
            else:
                rendered = format_float(value, digits=digits, suffix=suffix)
            cells.append(f'<td data-sort-value="{sort_value(value)}">{rendered}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table{class_attr}{default_sort_attrs}><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def failure_table_html(failures: list[FailedFit]) -> str:
    if not failures:
        return "<p>No fit failures.</p>"
    rows = []
    for failure in failures:
        rows.append(
            "<tr>"
            f"<td>{html.escape(failure.spec.label)}</td>"
            f"<td>{html.escape(failure.spec.family)}</td>"
            f"<td><code>{html.escape(failure.error[:500])}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>model</th><th>family</th><th>reason</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_figure(predictions: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Actual vs predicted clean-seen loss", "Heldout prediction error", "1e22 actual vs predicted"),
        horizontal_spacing=0.09,
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]],
    )
    initial_model = str(summary.iloc[0]["model_key"])
    trace_keys: list[str] = []
    lr_colors = {lr: PALETTE[index % len(PALETTE)] for index, lr in enumerate(LR_ORDER)}

    def add_model_trace(trace: Any, model_key: str, row: int, col: int) -> None:
        trace.visible = model_key == initial_model
        fig.add_trace(trace, row=row, col=col)
        trace_keys.append(model_key)

    def add_always_trace(trace: Any, row: int, col: int) -> None:
        trace.visible = True
        fig.add_trace(trace, row=row, col=col)
        trace_keys.append("always")

    for model_key, model_frame in predictions.groupby("model_key", sort=False):
        for lr in LR_ORDER:
            for split in ("train", "heldout"):
                frame = model_frame[model_frame["lr"].eq(lr) & model_frame["split"].eq(split)]
                if frame.empty:
                    continue
                add_model_trace(
                    go.Scatter(
                        x=frame[TARGET_METRIC],
                        y=frame["prediction"],
                        mode="markers",
                        name=f"{LR_LABELS[lr]} {split}",
                        legendgroup=f"{model_key}-{lr}-{split}",
                        marker={
                            "symbol": MARKERS[0] if split == "train" else MARKERS[9],
                            "size": 9 if split == "train" else 13,
                            "color": lr_colors[lr],
                            "line": {"width": 1, "color": "#111827"},
                        },
                        customdata=np.stack(
                            [
                                frame["model_label"].astype(str),
                                frame["scale"].astype(str),
                                frame["lr"].astype(str),
                                frame["anchor_4plus_loss"].astype(float),
                                frame["error_pct"].astype(float),
                                frame["run"].astype(str),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[5]}<br>scale=%{customdata[1]} lr=%{customdata[2]}"
                            "<br>anchor 4plus=%{customdata[3]:.5f}<br>actual clean=%{x:.5f}<br>pred=%{y:.5f}"
                            "<br>error=%{customdata[4]:+.2f}%<extra></extra>"
                        ),
                    ),
                    model_key,
                    row=1,
                    col=1,
                )
        heldout = model_frame[model_frame["split"].eq("heldout")]
        for lr in LR_ORDER:
            frame = heldout[heldout["lr"].eq(lr)].sort_values("scale_flops")
            if frame.empty:
                continue
            add_model_trace(
                go.Scatter(
                    x=frame["scale"],
                    y=frame["error_pct"],
                    mode="lines+markers",
                    name=f"{LR_LABELS[lr]} heldout error",
                    legendgroup=f"{model_key}-{lr}-error",
                    marker={"symbol": MARKERS[int(float(lr) * 100) % len(MARKERS)], "size": 10, "color": lr_colors[lr]},
                    line={"color": lr_colors[lr]},
                    customdata=np.stack(
                        [
                            frame["model_label"].astype(str),
                            frame[TARGET_METRIC].astype(float),
                            frame["prediction"].astype(float),
                            frame["anchor_4plus_loss"].astype(float),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{fullData.name}<br>scale=%{x}<br>anchor 4plus=%{customdata[3]:.5f}"
                        "<br>actual clean=%{customdata[1]:.5f}<br>pred=%{customdata[2]:.5f}<br>error=%{y:+.2f}%<extra></extra>"
                    ),
                ),
                model_key,
                row=1,
                col=2,
            )
        one_e22 = model_frame[model_frame["scale"].eq("1e22")].sort_values("lr_numeric")
        if not one_e22.empty:
            add_model_trace(
                go.Bar(
                    x=one_e22["lr"],
                    y=one_e22[TARGET_METRIC],
                    name="1e22 actual",
                    marker={"color": "#94a3b8"},
                    offsetgroup="actual",
                    legendgroup=f"{model_key}-1e22",
                    hovertemplate="lr=%{x}<br>actual clean=%{y:.5f}<extra></extra>",
                ),
                model_key,
                row=1,
                col=3,
            )
            add_model_trace(
                go.Bar(
                    x=one_e22["lr"],
                    y=one_e22["prediction"],
                    name="1e22 predicted",
                    marker={"color": "#1877F2"},
                    offsetgroup="predicted",
                    legendgroup=f"{model_key}-1e22",
                    customdata=one_e22["error_pct"],
                    hovertemplate="lr=%{x}<br>pred=%{y:.5f}<br>error=%{customdata:+.2f}%<extra></extra>",
                ),
                model_key,
                row=1,
                col=3,
            )
    lower = float(min(predictions[TARGET_METRIC].min(), predictions["prediction"].min()))
    upper = float(max(predictions[TARGET_METRIC].max(), predictions["prediction"].max()))
    add_always_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="perfect prediction",
            line={"color": "#475569", "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line={"color": "#475569", "width": 1}, row=1, col=2)
    label_by_key = dict(zip(summary["model_key"], summary["model_label"], strict=False))
    buttons = []
    for model_key, label in label_by_key.items():
        visible = [trace_key in ("always", model_key) for trace_key in trace_keys]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"K=0.20 clean-seen fit-family comparison<br><sup>{html.escape(label)}</sup>"},
                ],
            }
        )
    fig.update_xaxes(title_text="actual clean-seen loss", row=1, col=1)
    fig.update_yaxes(title_text="predicted clean-seen loss", row=1, col=1)
    fig.update_xaxes(title_text="heldout scale", categoryorder="array", categoryarray=SCALE_ORDER, row=1, col=2)
    fig.update_yaxes(title_text="prediction error (pred / actual - 1) [%]", row=1, col=2)
    fig.update_xaxes(title_text="1e22 LR factor", row=1, col=3)
    fig.update_yaxes(title_text="clean-seen loss", row=1, col=3)
    fig.update_layout(
        title=f"K=0.20 clean-seen fit-family comparison<br><sup>{label_by_key[initial_model]}</sup>",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.2,
                "yanchor": "top",
            }
        ],
        barmode="group",
        width=1550,
        height=700,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 70, "r": 30, "t": 110, "b": 150},
    )
    return fig


def resource_definition_html() -> str:
    items = []
    for label, description in RESOURCE_DESCRIPTIONS.items():
        items.append(f"<li><code>{html.escape(label)}</code>: {html.escape(description)}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def build_report_html(
    fig: go.Figure,
    summary: pd.DataFrame,
    one_e22: pd.DataFrame,
    failures: list[FailedFit],
    input_path: str,
    fit_through_scale: str,
) -> str:
    figure_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="k020-clean-seen-fit-family")
    summary_columns = [
        ("model_label", "model", 3, ""),
        ("family", "family", 3, ""),
        ("resource", "resource", 3, ""),
        ("heldout_n", "heldout n", 0, ""),
        ("heldout_mae_pct", "heldout MAE", 2, "%"),
        ("heldout_rmse_pct", "heldout RMSE", 2, "%"),
        ("1e22_mae_pct", "1e22 MAE", 2, "%"),
        ("1e22_033_prediction", "1e22 lr0.33 pred", 5, ""),
        ("1e22_050_prediction", "1e22 lr0.50 pred", 5, ""),
        ("1e22_067_prediction", "1e22 lr0.67 pred", 5, ""),
        ("1e22_083_prediction", "1e22 lr0.83 pred", 5, ""),
        ("1e22_067_actual", "1e22 lr0.67 actual", 5, ""),
        ("1e22_067_error_pct", "1e22 lr0.67 err", 2, "%"),
    ]
    one_e22_columns = [
        ("model_label", "model", 3, ""),
        ("family", "family", 3, ""),
        ("resource", "resource", 3, ""),
        ("lr", "LR", 2, ""),
        (TARGET_METRIC, "actual", 5, ""),
        ("prediction", "prediction", 5, ""),
        ("error_pct", "signed error", 2, "%"),
        ("abs_error_pct", "abs error", 2, "%"),
        (ANCHOR_METRIC, "anchor", 5, ""),
        ("run", "run", 3, ""),
    ]
    failure_html = failure_table_html(failures)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi K=0.20 Clean-Seen Fit Family Report</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #172033;
      --muted: #5f6b7a;
      --border: #d8dee8;
      --panel: #f7f9fc;
      --accent: #1877f2;
    }}
    body {{
      margin: 0;
      color: var(--text);
      background: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1580px;
      margin: 0 auto;
      padding: 28px 32px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 10px;
    }}
    code {{
      padding: 1px 4px;
      border-radius: 4px;
      background: #edf1f7;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
    }}
    .lede {{
      max-width: 1120px;
      color: var(--muted);
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin: 20px 0 18px;
    }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px 16px;
    }}
    .panel ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .panel li {{
      margin: 5px 0;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #f2f7ff;
      padding: 12px 14px;
      margin: 18px 0;
      max-width: 1180px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 8px 0 20px;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #f8fafc;
      position: sticky;
      top: 0;
    }}
    table.sortable th {{
      cursor: pointer;
      user-select: none;
    }}
    table.sortable th button {{
      appearance: none;
      border: 0;
      padding: 0;
      margin: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      cursor: pointer;
      text-align: left;
      white-space: nowrap;
    }}
    .sort-indicator {{
      display: inline-block;
      width: 3ch;
      margin-left: 5px;
      color: var(--accent);
    }}
    th[aria-sort="ascending"] .sort-indicator::after {{
      content: "asc";
    }}
    th[aria-sort="descending"] .sort-indicator::after {{
      content: "desc";
    }}
    details {{
      margin: 18px 0;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    .plot-wrap {{
      margin-top: 14px;
      border-top: 1px solid var(--border);
      padding-top: 14px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Delphi K=0.20 clean-seen fit-family comparison</h1>
  <p class="lede">
    This report fits only the p33m67 K=0.20 ladder evaluated on the clean-seen 1e22 validation set.
    Input: <code>{html.escape(input_path)}</code>. Target metric: <code>{TARGET_METRIC}</code>.
    Fits train through <code>{html.escape(fit_through_scale)}</code>; <code>1e21</code> and <code>1e22</code>
    are held out.
  </p>
  <div class="callout">
    <strong>Interpretation:</strong> scale-only resources are mostly alternative parameterizations of the same K=0.20
    ladder because <code>D_math = 0.67 * 0.20 * D_pre</code>. Anchor-calibration fits use the same-checkpoint
    original 4plus validation loss, so they answer a different question: how well clean-seen loss can be calibrated
    from the old validation metric.
  </div>
  <div class="grid">
    <section class="panel">
      <h2>Fit Families</h2>
      <ul>
        <li><code>per-LR Chinchilla</code>: independent <code>L=E+A*x^-alpha</code> fit for each LR.</li>
        <li><code>per-LR log-log</code>: independent pure power law for each LR.</li>
        <li><code>pooled Chinchilla + LR offsets</code>: shared scale curve plus LR-specific offsets.</li>
        <li><code>pooled Chinchilla/log-log + LR quadratic</code>: shared scale curve plus continuous LR terms.</li>
        <li><code>anchor calibration</code>: maps original 4plus loss to clean-seen loss.</li>
      </ul>
    </section>
    <section class="panel">
      <h2>Resources</h2>
      {resource_definition_html()}
    </section>
  </div>
  <h2>Heldout Summary</h2>
  <div class="table-wrap">
    {table_html(summary, summary_columns, default_sort_key="heldout_mae_pct")}
  </div>
  <div class="plot-wrap">
    {figure_html}
  </div>
  <details open>
    <summary>All 1e22 predictions</summary>
    <div class="table-wrap">
      {table_html(one_e22, one_e22_columns, default_sort_key="abs_error_pct")}
    </div>
  </details>
  <details>
    <summary>Fit failures</summary>
    <div class="table-wrap">{failure_html}</div>
  </details>
</main>
<script>
  function cellSortValue(row, index) {{
    const cell = row.cells[index];
    if (!cell) {{
      return {{missing: true, text: "", number: Number.NaN}};
    }}
    const raw = cell.dataset.sortValue || "";
    if (raw === "") {{
      return {{missing: true, text: "", number: Number.NaN}};
    }}
    const number = Number(raw);
    return {{
      missing: false,
      text: raw,
      number: Number.isFinite(number) ? number : Number.NaN,
    }};
  }}

  function compareRows(left, right, index, direction) {{
    const leftValue = cellSortValue(left, index);
    const rightValue = cellSortValue(right, index);
    if (leftValue.missing && rightValue.missing) {{
      return 0;
    }}
    if (leftValue.missing) {{
      return 1;
    }}
    if (rightValue.missing) {{
      return -1;
    }}
    const bothNumeric = Number.isFinite(leftValue.number) && Number.isFinite(rightValue.number);
    const result = bothNumeric
      ? leftValue.number - rightValue.number
      : leftValue.text.localeCompare(rightValue.text, undefined, {{numeric: true, sensitivity: "base"}});
    return direction === "asc" ? result : -result;
  }}

  function sortTable(table, index, direction) {{
    const body = table.tBodies[0];
    const rows = Array.from(body.rows);
    rows.sort((left, right) => compareRows(left, right, index, direction));
    for (const row of rows) {{
      body.appendChild(row);
    }}
    for (const header of table.tHead.rows[0].cells) {{
      header.removeAttribute("aria-sort");
      header.dataset.direction = "";
    }}
    const activeHeader = table.tHead.rows[0].cells[index];
    activeHeader.setAttribute("aria-sort", direction === "asc" ? "ascending" : "descending");
    activeHeader.dataset.direction = direction;
  }}

  function installSortableTables() {{
    for (const table of document.querySelectorAll("table.sortable")) {{
      const headers = Array.from(table.tHead.rows[0].cells);
      for (const header of headers) {{
        const index = Number(header.dataset.sortIndex);
        header.addEventListener("click", () => {{
          const current = header.dataset.direction;
          const next = current === "asc" ? "desc" : "asc";
          sortTable(table, index, next);
        }});
      }}
      const defaultIndex = Number(table.dataset.defaultSortIndex);
      if (Number.isInteger(defaultIndex)) {{
        sortTable(table, defaultIndex, table.dataset.defaultSortDirection || "asc");
      }}
    }}
  }}

  installSortableTables();
</script>
</body>
</html>
"""


def build_outputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FailedFit], Path]:
    points = load_clean_seen_summary(args.input)
    predictions, fit_table, models, failures = fit_all(points, args.fit_through_scale)
    summary = summarize_predictions(predictions, fit_table)
    one_e22 = one_e22_predictions(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    html_path = stem.with_suffix(".html")
    write_csv(stem.with_name(f"{stem.name}_points.csv"), points)
    write_csv(stem.with_name(f"{stem.name}_predictions.csv"), predictions)
    write_csv(stem.with_name(f"{stem.name}_summary.csv"), summary)
    write_csv(stem.with_name(f"{stem.name}_fit_table.csv"), fit_table)
    write_csv(stem.with_name(f"{stem.name}_1e22_predictions.csv"), one_e22)
    write_json(
        stem.with_name(f"{stem.name}_fit.json"),
        {
            "input": args.input,
            "fit_through_scale": args.fit_through_scale,
            "target_metric": TARGET_METRIC,
            "anchor_metric": ANCHOR_METRIC,
            "succeeded_models": [model.spec.key for model in models],
            "failed_models": [
                {
                    "model_key": failure.spec.key,
                    "model_label": failure.spec.label,
                    "family": failure.spec.family,
                    "error": failure.error,
                }
                for failure in failures
            ],
            "summary": records_for_json(summary),
        },
    )
    write_text(
        html_path,
        build_report_html(
            build_figure(predictions, summary),
            summary,
            one_e22,
            failures,
            args.input,
            args.fit_through_scale,
        ),
    )
    return points, predictions, summary, one_e22, failures, html_path


def main() -> None:
    args = parse_args()
    _, _, summary, one_e22, failures, html_path = build_outputs(args)
    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(one_e22.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if failures:
        print()
        print(f"{len(failures)} fit(s) failed; see fit JSON for details")
    print()
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
