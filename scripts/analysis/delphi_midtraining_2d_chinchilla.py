# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit Delphi midtraining loss as a function of pretraining FLOPs and midtraining tokens.

The existing endpoint plots fit one Chinchilla-style curve per fixed-token budget:

    L(C) = E + A * (C / 1e18)^(-alpha)

where C is base-model pretraining FLOPs. This script fits less constrained
two-resource forms:

    L(C, D) = E + A * (C / 1e18)^(-alpha) + B * (D / 1e9)^(-beta)

and the corresponding pure power-law log-log forms, where D is midtraining
tokens. The default training split matches the previous analysis: train on
3e18 through 3e20 and report held-out error on 1e21/1e22.

The script reuses the validated endpoint loaders from
``delphi_isotoken_endpoint_scaling.py`` and the Marin scaling-law plotting
constants from ``marin.scaling_laws.scaling_plots``.

It compares four default fit modes:

1. Chinchilla endpoints only: the original two-resource fit on endpoints.
2. Chinchilla separate base component: fit a base-loss curve plus a midtraining
   improvement term that is exactly zero at D=0.
3. Log-log endpoints only: pure power law in C and D on endpoints.
4. Log-log separate base component: pure power-law base loss plus pure
   power-law midtraining improvement.

Passing ``--include-parameter-data-chinchilla`` adds a fifth endpoint-only
Chinchilla fit that replaces pretraining FLOPs with model-size and
pretraining-token resources:

    L(N, D_pre, D_mid) = E + A * N^-alpha + B * D_pre^-beta + G * D_mid^-gamma

where ``D_pre`` is pretraining tokens and ``D_mid`` is midtraining tokens.

Run:
    uv run python scripts/analysis/delphi_midtraining_2d_chinchilla.py
    uv run python scripts/analysis/delphi_midtraining_2d_chinchilla.py --refresh-cache
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from build_delphi_midtraining_interactive_report import finite_or_none, fit_floor_power
from delphi_isotoken_endpoint_scaling import (
    ALL_SCALE_FLOPS,
    DEFAULT_CUTOFF_SCALE,
    HELD_OUT_SCALES,
    ISOFLOP_SERIES,
    MIDTRAIN_BUDGET_FRACTION,
    OUT_DIR,
    SCALE_ORDER,
    SCALE_PRETRAIN_TOKENS_B,
    budget_tokens,
    load_endpoints,
    load_isoflop_endpoints,
    refresh_cache,
    refresh_isoflop_cache,
)
from marin.scaling_laws.scaling_plots import MARKERS, PALETTE
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from experiments.delphi_models import DELPHI_BY_FLOPS_KEY

logger = logging.getLogger("delphi_midtraining_2d_chinchilla")

FLOP_NORM = 1e18
TOKEN_NORM = 1e9
PARAM_NORM = 1e9
DEFAULT_OUTPUT_STEM = "delphi_midtraining_2d_chinchilla"
PARAMETER_DATA_OUTPUT_STEM = "delphi_midtraining_param_data_chinchilla"
TRAJECTORY_POINTS_PATH = Path("midtrain_analysis_outputs/small_final_loss_scaling/trajectory_points.csv")
METRIC_LABEL = "math_val_loss"

MODEL_LABELS = {
    "chinchilla_endpoints": "1) Chinchilla endpoints only",
    "chinchilla_separate_base": "2) Chinchilla separate base component",
    "loglog_endpoints": "3) log-log endpoints only",
    "loglog_separate_base": "4) log-log separate base component",
}
PARAMETER_DATA_MODEL_KEY = "parameter_data_chinchilla_endpoints"
PARAMETER_DATA_MODEL_LABEL = "5) Chinchilla params + data"


@dataclass(frozen=True)
class TwoResourceFit:
    """Parameters for L(C, D) = E + A C^-alpha + B D^-beta."""

    floor: float
    flops_amplitude: float
    flops_exponent: float
    tokens_amplitude: float
    tokens_exponent: float
    r2: float | None
    rmse: float
    n: int


@dataclass(frozen=True)
class ParameterDataChinchillaFit:
    """Parameters for L(N,D_pre,D_mid)=E+A N^-a+B D_pre^-b+G D_mid^-g."""

    floor: float
    params_amplitude: float
    params_exponent: float
    pretrain_data_amplitude: float
    pretrain_data_exponent: float
    midtrain_data_amplitude: float
    midtrain_data_exponent: float
    r2: float | None
    rmse: float
    n: int


@dataclass(frozen=True)
class SeparateBaseFit:
    """Parameters for L(C,D)=L_base(C)-I(C,D), with I(C,0)=0."""

    base_floor: float
    base_amplitude: float
    base_exponent: float
    improvement_amplitude: float
    improvement_flops_exponent: float
    improvement_tokens_exponent: float
    improvement_token_scale_b: float
    r2: float | None
    rmse: float
    n: int


@dataclass(frozen=True)
class LogLogFit:
    """Parameters for log L = a + b log C + c log D."""

    intercept: float
    flops_slope: float
    tokens_slope: float
    r2: float | None
    rmse: float
    rmse_log: float
    n: int


@dataclass(frozen=True)
class LogLogSeparateBaseFit:
    """Parameters for L(C,D)=L_base(C)-I(C,D), both pure power laws."""

    base_intercept: float
    base_flops_slope: float
    improvement_intercept: float
    improvement_flops_slope: float
    improvement_tokens_slope: float
    r2: float | None
    rmse: float
    rmse_log: float | None
    n: int


@dataclass(frozen=True)
class K020ChinchillaFit:
    """Parameters for L(C) fit only on the K=0.20 ladder."""

    floor: float
    amplitude: float
    exponent: float
    r2: float | None
    rmse: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default="p33m67", help="Midtraining mix to fit.")
    parser.add_argument("--lr", default="50", help="LR factor suffix to fit, e.g. 50 for lr0.5.")
    parser.add_argument(
        "--fit-through-scale",
        choices=SCALE_ORDER[:-1],
        default=DEFAULT_CUTOFF_SCALE,
        help="Largest scale included in the training split.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh W&B cache before fitting. Default is local cache/CSV only.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="When refreshing W&B cache, skip history downloads and use summaries only.",
    )
    parser.add_argument(
        "--exclude-isoflop",
        action="store_true",
        help="Fit only fixed-token iso-token points, excluding K=0.20 iso-FLOP points.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename stem. Defaults to an all-series or iso-token-only stem.",
    )
    parser.add_argument(
        "--include-parameter-data-chinchilla",
        action="store_true",
        help="Add a Chinchilla fit with parameter count N and total data tokens D.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def two_resource_model(
    features: tuple[np.ndarray, np.ndarray], floor: float, a: float, alpha: float, b: float, beta: float
):
    flops, tokens = features
    return floor + a * np.power(flops, -alpha) + b * np.power(tokens, -beta)


def parameter_data_chinchilla_model(
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
    floor: float,
    params_amplitude: float,
    params_exponent: float,
    pretrain_data_amplitude: float,
    pretrain_data_exponent: float,
    midtrain_data_amplitude: float,
    midtrain_data_exponent: float,
) -> np.ndarray:
    params, pretrain_tokens, midtrain_tokens = features
    return (
        floor
        + params_amplitude * np.power(params, -params_exponent)
        + pretrain_data_amplitude * np.power(pretrain_tokens, -pretrain_data_exponent)
        + midtrain_data_amplitude * np.power(midtrain_tokens, -midtrain_data_exponent)
    )


def loglog_model(features: tuple[np.ndarray, np.ndarray], intercept: float, flops_slope: float, tokens_slope: float):
    flops, tokens = features
    return np.exp(intercept + flops_slope * np.log(flops) + tokens_slope * np.log(tokens))


def separate_base_model(
    features: tuple[np.ndarray, np.ndarray],
    base_floor: float,
    base_amplitude: float,
    base_exponent: float,
    improvement_amplitude: float,
    improvement_flops_exponent: float,
    improvement_tokens_exponent: float,
    improvement_token_scale_b: float,
) -> np.ndarray:
    flops, tokens = features
    base_loss = base_floor + base_amplitude * np.power(flops, -base_exponent)
    token_progress = 1.0 - np.power(1.0 + tokens / improvement_token_scale_b, -improvement_tokens_exponent)
    improvement = improvement_amplitude * np.power(flops, -improvement_flops_exponent) * token_progress
    return base_loss - improvement


def loglog_separate_base_model(
    features: tuple[np.ndarray, np.ndarray],
    base_intercept: float,
    base_flops_slope: float,
    improvement_intercept: float,
    improvement_flops_slope: float,
    improvement_tokens_slope: float,
) -> np.ndarray:
    flops, tokens = features
    base_loss = np.exp(base_intercept + base_flops_slope * np.log(flops))
    improvement = np.zeros_like(base_loss)
    positive = tokens > 0
    improvement[positive] = np.exp(
        improvement_intercept
        + improvement_flops_slope * np.log(flops[positive])
        + improvement_tokens_slope * np.log(tokens[positive])
    )
    return base_loss - improvement


def predict_two_resource(fit: TwoResourceFit, frame: pd.DataFrame) -> np.ndarray:
    flops = frame["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = frame["tokens_b"].to_numpy(dtype=float)
    return two_resource_model(
        (flops, tokens),
        fit.floor,
        fit.flops_amplitude,
        fit.flops_exponent,
        fit.tokens_amplitude,
        fit.tokens_exponent,
    )


def predict_parameter_data_chinchilla(fit: ParameterDataChinchillaFit, frame: pd.DataFrame) -> np.ndarray:
    params = frame["params_b"].to_numpy(dtype=float)
    pretrain_tokens = frame["pretrain_tokens_b"].to_numpy(dtype=float)
    midtrain_tokens = frame["tokens_b"].to_numpy(dtype=float)
    return parameter_data_chinchilla_model(
        (params, pretrain_tokens, midtrain_tokens),
        fit.floor,
        fit.params_amplitude,
        fit.params_exponent,
        fit.pretrain_data_amplitude,
        fit.pretrain_data_exponent,
        fit.midtrain_data_amplitude,
        fit.midtrain_data_exponent,
    )


def predict_loglog(fit: LogLogFit, frame: pd.DataFrame) -> np.ndarray:
    flops = frame["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = frame["tokens_b"].to_numpy(dtype=float)
    return loglog_model((flops, tokens), fit.intercept, fit.flops_slope, fit.tokens_slope)


def predict_separate_base(fit: SeparateBaseFit, frame: pd.DataFrame) -> np.ndarray:
    flops = frame["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = frame["tokens_b"].to_numpy(dtype=float)
    return separate_base_model(
        (flops, tokens),
        fit.base_floor,
        fit.base_amplitude,
        fit.base_exponent,
        fit.improvement_amplitude,
        fit.improvement_flops_exponent,
        fit.improvement_tokens_exponent,
        fit.improvement_token_scale_b,
    )


def predict_loglog_separate_base(fit: LogLogSeparateBaseFit, frame: pd.DataFrame) -> np.ndarray:
    flops = frame["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = frame["tokens_b"].to_numpy(dtype=float)
    return loglog_separate_base_model(
        (flops, tokens),
        fit.base_intercept,
        fit.base_flops_slope,
        fit.improvement_intercept,
        fit.improvement_flops_slope,
        fit.improvement_tokens_slope,
    )


def predict_k020_chinchilla(fit: K020ChinchillaFit, frame: pd.DataFrame) -> np.ndarray:
    flops = frame["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    return fit.floor + fit.amplitude * np.power(flops, -fit.exponent)


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


def rmse_log_score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if np.any(actual <= 0) or np.any(predicted <= 0):
        return None
    return math.sqrt(float(np.mean((np.log(actual) - np.log(predicted)) ** 2)))


def fit_loglog(points: pd.DataFrame) -> LogLogFit:
    flops = points["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = points["tokens_b"].to_numpy(dtype=float)
    loss = points["value"].to_numpy(dtype=float)
    if np.any(tokens <= 0):
        raise ValueError("Log-log endpoint fit requires positive midtraining tokens")
    design = np.column_stack([np.ones_like(flops), np.log(flops), np.log(tokens)])
    params, *_ = np.linalg.lstsq(design, np.log(loss), rcond=None)
    predicted = loglog_model((flops, tokens), float(params[0]), float(params[1]), float(params[2]))
    return LogLogFit(
        intercept=float(params[0]),
        flops_slope=float(params[1]),
        tokens_slope=float(params[2]),
        r2=r2_score(loss, predicted),
        rmse=math.sqrt(float(np.mean((loss - predicted) ** 2))),
        rmse_log=float(rmse_log_score(loss, predicted)),
        n=int(loss.size),
    )


def fit_parameter_data_chinchilla(points: pd.DataFrame) -> ParameterDataChinchillaFit:
    params = points["params_b"].to_numpy(dtype=float)
    pretrain_tokens = points["pretrain_tokens_b"].to_numpy(dtype=float)
    midtrain_tokens = points["tokens_b"].to_numpy(dtype=float)
    loss = points["value"].to_numpy(dtype=float)
    floor0 = float(loss.min()) * 0.3
    spread = max(float(loss.max() - loss.min()), 1e-3)
    initial = (floor0, spread / 3.0, 0.1, spread / 3.0, 0.08, spread / 3.0, 0.15)
    lower = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    upper = (float(loss.min()) * 0.999, np.inf, 5.0, np.inf, 5.0, np.inf, 5.0)
    fitted, _ = curve_fit(
        parameter_data_chinchilla_model,
        (params, pretrain_tokens, midtrain_tokens),
        loss,
        p0=initial,
        bounds=(lower, upper),
        maxfev=200_000,
    )
    predicted = parameter_data_chinchilla_model((params, pretrain_tokens, midtrain_tokens), *fitted)
    rmse = math.sqrt(float(np.mean((loss - predicted) ** 2)))
    return ParameterDataChinchillaFit(
        floor=float(fitted[0]),
        params_amplitude=float(fitted[1]),
        params_exponent=float(fitted[2]),
        pretrain_data_amplitude=float(fitted[3]),
        pretrain_data_exponent=float(fitted[4]),
        midtrain_data_amplitude=float(fitted[5]),
        midtrain_data_exponent=float(fitted[6]),
        r2=r2_score(loss, predicted),
        rmse=rmse,
        n=int(loss.size),
    )


def fit_two_resource(points: pd.DataFrame) -> TwoResourceFit:
    flops = points["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = points["tokens_b"].to_numpy(dtype=float)
    loss = points["value"].to_numpy(dtype=float)
    floor0 = float(loss.min()) * 0.35
    spread = max(float(loss.max() - loss.min()), 1e-3)
    initial = (floor0, spread * 0.7, 0.08, spread * 0.3, 0.15)
    lower = (0.0, 0.0, 0.0, 0.0, 0.0)
    upper = (float(loss.min()) * 0.999, np.inf, 5.0, np.inf, 5.0)
    params, _ = curve_fit(
        two_resource_model,
        (flops, tokens),
        loss,
        p0=initial,
        bounds=(lower, upper),
        maxfev=100_000,
    )
    predicted = two_resource_model((flops, tokens), *params)
    rmse = math.sqrt(float(np.mean((loss - predicted) ** 2)))
    return TwoResourceFit(
        floor=float(params[0]),
        flops_amplitude=float(params[1]),
        flops_exponent=float(params[2]),
        tokens_amplitude=float(params[3]),
        tokens_exponent=float(params[4]),
        r2=r2_score(loss, predicted),
        rmse=rmse,
        n=int(loss.size),
    )


def base_curve_fit(base_points: pd.DataFrame) -> dict[str, float]:
    fit = fit_floor_power(
        base_points["scale_flops"].to_numpy(dtype=float),
        base_points["value"].to_numpy(dtype=float),
    )
    if fit is None:
        raise ValueError("Need at least three base points for the base-loss curve")
    return {
        "floor": float(fit["floor"]),
        "amplitude": float(fit["amplitude"]),
        "exponent": float(fit["alpha"]),
    }


def fit_base_loglog(base_points: pd.DataFrame) -> tuple[float, float]:
    flops = base_points["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    loss = base_points["value"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(flops), np.log(flops)])
    params, *_ = np.linalg.lstsq(design, np.log(loss), rcond=None)
    return float(params[0]), float(params[1])


def fit_separate_base(points: pd.DataFrame) -> SeparateBaseFit:
    base_points = points[points["series"].eq("base")]
    if base_points.empty:
        raise ValueError("Separate-base fit requires base series points")
    initial_base = base_curve_fit(base_points)
    flops = points["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = points["tokens_b"].to_numpy(dtype=float)
    loss = points["value"].to_numpy(dtype=float)
    positive_tokens = tokens[tokens > 0]
    if positive_tokens.size == 0:
        raise ValueError("Separate-base fit requires at least one positive-token point")
    base_prediction = initial_base["floor"] + initial_base["amplitude"] * np.power(flops, -initial_base["exponent"])
    improvement0 = max(float(np.median(np.maximum(base_prediction - loss, 0.0))), 1e-3)
    token_scale0 = float(np.median(positive_tokens))
    initial = (
        initial_base["floor"],
        initial_base["amplitude"],
        initial_base["exponent"],
        improvement0,
        0.05,
        0.25,
        token_scale0,
    )
    lower = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-9)
    upper = (float(loss.max()) * 1.5, np.inf, 5.0, np.inf, 5.0, 5.0, float(positive_tokens.max()) * 100.0)

    def model(
        features,
        base_floor,
        base_amplitude,
        base_exponent,
        improvement_amplitude,
        flops_exponent,
        tokens_exponent,
        token_scale,
    ):
        return separate_base_model(
            features,
            base_floor,
            base_amplitude,
            base_exponent,
            improvement_amplitude,
            flops_exponent,
            tokens_exponent,
            token_scale,
        )

    params, _ = curve_fit(
        model,
        (flops, tokens),
        loss,
        p0=initial,
        bounds=(lower, upper),
        maxfev=100_000,
    )
    predicted = model((flops, tokens), *params)
    rmse = math.sqrt(float(np.mean((loss - predicted) ** 2)))
    return SeparateBaseFit(
        base_floor=float(params[0]),
        base_amplitude=float(params[1]),
        base_exponent=float(params[2]),
        improvement_amplitude=float(params[3]),
        improvement_flops_exponent=float(params[4]),
        improvement_tokens_exponent=float(params[5]),
        improvement_token_scale_b=float(params[6]),
        r2=r2_score(loss, predicted),
        rmse=rmse,
        n=int(loss.size),
    )


def fit_loglog_separate_base(points: pd.DataFrame) -> LogLogSeparateBaseFit:
    base_points = points[points["series"].eq("base")]
    if base_points.empty:
        raise ValueError("Log-log separate-base fit requires base series points")
    base_intercept, base_flops_slope = fit_base_loglog(base_points)
    endpoint_points = points[points["tokens_b"].gt(0)].copy()
    if endpoint_points.empty:
        raise ValueError("Log-log separate-base fit requires positive-token endpoint points")

    endpoint_flops = endpoint_points["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    endpoint_tokens = endpoint_points["tokens_b"].to_numpy(dtype=float)
    endpoint_loss = endpoint_points["value"].to_numpy(dtype=float)
    endpoint_base_prediction = np.exp(base_intercept + base_flops_slope * np.log(endpoint_flops))
    improvement = endpoint_base_prediction - endpoint_loss
    if np.any(improvement <= 0):
        bad = endpoint_points.loc[improvement <= 0, ["series", "scale", "tokens_b", "value"]]
        raise ValueError(f"Log-log improvement is non-positive for rows:\n{bad.to_string(index=False)}")

    design = np.column_stack([np.ones_like(endpoint_flops), np.log(endpoint_flops), np.log(endpoint_tokens)])
    initial, *_ = np.linalg.lstsq(design, np.log(improvement), rcond=None)

    def log_improvement_model(features, intercept, flops_slope, tokens_slope):
        flops, tokens = features
        return intercept + flops_slope * np.log(flops) + tokens_slope * np.log(tokens)

    params, _ = curve_fit(
        log_improvement_model,
        (endpoint_flops, endpoint_tokens),
        np.log(improvement),
        p0=(float(initial[0]), float(initial[1]), max(float(initial[2]), 0.1)),
        bounds=([-np.inf, -5.0, 0.0], [np.inf, 5.0, 5.0]),
        maxfev=100_000,
    )

    flops = points["scale_flops"].to_numpy(dtype=float) / FLOP_NORM
    tokens = points["tokens_b"].to_numpy(dtype=float)
    loss = points["value"].to_numpy(dtype=float)
    predicted = loglog_separate_base_model(
        (flops, tokens),
        base_intercept,
        base_flops_slope,
        float(params[0]),
        float(params[1]),
        float(params[2]),
    )
    return LogLogSeparateBaseFit(
        base_intercept=base_intercept,
        base_flops_slope=base_flops_slope,
        improvement_intercept=float(params[0]),
        improvement_flops_slope=float(params[1]),
        improvement_tokens_slope=float(params[2]),
        r2=r2_score(loss, predicted),
        rmse=math.sqrt(float(np.mean((loss - predicted) ** 2))),
        rmse_log=rmse_log_score(loss, predicted),
        n=int(loss.size),
    )


def fit_k020_chinchilla(points: pd.DataFrame) -> K020ChinchillaFit | None:
    k020 = points[points["series"].eq(ISOFLOP_SERIES)]
    if len(k020) < 3:
        return None
    fit = fit_floor_power(
        k020["scale_flops"].to_numpy(dtype=float),
        k020["value"].to_numpy(dtype=float),
    )
    if fit is None:
        return None
    return K020ChinchillaFit(
        floor=float(fit["floor"]),
        amplitude=float(fit["amplitude"]),
        exponent=float(fit["alpha"]),
        r2=None if fit["r2"] is None else float(fit["r2"]),
        rmse=float(fit["rmse"]),
        n=int(fit["n"]),
    )


def scale_label_for(value: Any) -> str:
    if isinstance(value, str) and value in ALL_SCALE_FLOPS:
        return value
    numeric = float(value)
    for label, flops in ALL_SCALE_FLOPS.items():
        if math.isclose(numeric, flops, rel_tol=1e-9):
            return label
    raise ValueError(f"Unknown scale value: {value!r}")


def params_b_for_scale(scale: str) -> float:
    model = DELPHI_BY_FLOPS_KEY.get(scale)
    if model is None:
        raise ValueError(f"No Delphi model registered for scale {scale!r}")
    return model.params / PARAM_NORM


def pretrain_tokens_b_for_scale(scale: str) -> float:
    return SCALE_PRETRAIN_TOKENS_B[scale]


def tokens_for_row(row: pd.Series) -> float:
    tokens_b = row.get("tokens_b")
    if pd.notna(tokens_b) and float(tokens_b) > 0:
        return float(tokens_b)
    series = str(row["series"])
    scale = str(row["scale"])
    if series.startswith("tok"):
        return budget_tokens(series.removeprefix("tok")) / TOKEN_NORM
    if series == ISOFLOP_SERIES:
        return MIDTRAIN_BUDGET_FRACTION * SCALE_PRETRAIN_TOKENS_B[scale]
    raise ValueError(f"Cannot infer midtraining tokens for series {series!r}")


def normalize_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["scale"] = normalized["scale"].map(scale_label_for)
    normalized["scale_flops"] = normalized["scale"].map(ALL_SCALE_FLOPS).astype(float)
    normalized["params_b"] = normalized["scale"].map(params_b_for_scale).astype(float)
    normalized["pretrain_tokens_b"] = normalized["scale"].map(pretrain_tokens_b_for_scale).astype(float)
    normalized["lr"] = normalized["lr"].astype(str)
    normalized["tokens_b"] = normalized.apply(tokens_for_row, axis=1)
    normalized["total_tokens_b"] = normalized["pretrain_tokens_b"] + normalized["tokens_b"]
    return normalized.sort_values(["series", "tokens_b", "scale_flops"]).reset_index(drop=True)


def csv_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "isotoken_endpoints.csv", output_dir / "isoflop_k020_endpoints.csv"


def load_from_csv(output_dir: Path) -> pd.DataFrame:
    isotoken_path, isoflop_path = csv_paths(output_dir)
    if not isotoken_path.exists() or not isoflop_path.exists():
        raise FileNotFoundError(
            f"Missing {isotoken_path} or {isoflop_path}. Run delphi_isotoken_endpoint_scaling.py "
            "or pass --refresh-cache."
        )
    isotoken = pd.read_csv(isotoken_path, dtype={"lr": str})
    isoflop = pd.read_csv(isoflop_path, dtype={"lr": str})
    return normalize_endpoints(pd.concat([isotoken, isoflop], ignore_index=True))


def load_endpoint_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.refresh_cache:
        refresh_cache(with_history=not args.no_history)
        isotoken = load_endpoints()
        refresh_isoflop_cache(isotoken[["mix", "lr"]].drop_duplicates(), with_history=not args.no_history)
        return normalize_endpoints(pd.concat([isotoken, load_isoflop_endpoints()], ignore_index=True))

    try:
        return normalize_endpoints(pd.concat([load_endpoints(), load_isoflop_endpoints()], ignore_index=True))
    except ValueError as exc:
        logger.info("cache load failed (%s); falling back to CSV endpoints", exc)
        return load_from_csv(args.output_dir)


def load_base_points(args: argparse.Namespace) -> pd.DataFrame:
    if not TRAJECTORY_POINTS_PATH.exists():
        raise FileNotFoundError(f"Missing base-loss trajectory file: {TRAJECTORY_POINTS_PATH}")
    source = pd.read_csv(TRAJECTORY_POINTS_PATH, dtype={"lr": str})
    mask = (
        source["metric_label"].eq(METRIC_LABEL)
        & source["step"].eq(0)
        & source["mix"].eq(args.mix)
        & source["lr"].astype(str).eq(str(args.lr))
    )
    base = source[mask].copy()
    if base.empty:
        raise ValueError(
            f"No step-0 {METRIC_LABEL} base rows for mix={args.mix!r}, lr={args.lr!r} " f"in {TRAJECTORY_POINTS_PATH}"
        )
    grouped = (
        base.groupby("scale", as_index=False)
        .agg(value=("value", "mean"), source_n=("value", "size"), source_std=("value", "std"))
        .reset_index(drop=True)
    )
    grouped["scale"] = grouped["scale"].map(scale_label_for)
    grouped["scale_flops"] = grouped["scale"].map(ALL_SCALE_FLOPS).astype(float)
    grouped["params_b"] = grouped["scale"].map(params_b_for_scale).astype(float)
    grouped["pretrain_tokens_b"] = grouped["scale"].map(pretrain_tokens_b_for_scale).astype(float)
    grouped["run_id"] = "base-step0-" + grouped["scale"].astype(str)
    grouped["run_name"] = grouped["run_id"]
    grouped["url"] = ""
    grouped["mix"] = args.mix
    grouped["lr"] = str(args.lr)
    grouped["series"] = "base"
    grouped["tokens_b"] = 0.0
    grouped["total_tokens_b"] = grouped["pretrain_tokens_b"]
    grouped["final_step"] = 0
    grouped["source_std"] = grouped["source_std"].fillna(0.0)
    columns = [
        "run_id",
        "run_name",
        "url",
        "scale",
        "mix",
        "lr",
        "series",
        "tokens_b",
        "final_step",
        "value",
        "scale_flops",
        "params_b",
        "pretrain_tokens_b",
        "total_tokens_b",
        "source_n",
        "source_std",
    ]
    return grouped[columns].sort_values("scale_flops").reset_index(drop=True)


def split_points(points: pd.DataFrame, fit_through_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = ALL_SCALE_FLOPS[fit_through_scale]
    train = points[points["scale_flops"] <= cutoff + 1.0].copy()
    heldout = points[points["scale"].isin(HELD_OUT_SCALES) & (points["scale_flops"] > cutoff + 1.0)].copy()
    if train.empty:
        raise ValueError(f"No training points at or below {fit_through_scale}")
    if heldout.empty:
        raise ValueError(f"No held-out points above {fit_through_scale}")
    return train, heldout


FitResult = TwoResourceFit | ParameterDataChinchillaFit | SeparateBaseFit | LogLogFit | LogLogSeparateBaseFit


def model_labels(include_parameter_data_chinchilla: bool) -> dict[str, str]:
    labels = dict(MODEL_LABELS)
    if include_parameter_data_chinchilla:
        labels[PARAMETER_DATA_MODEL_KEY] = PARAMETER_DATA_MODEL_LABEL
    return labels


def fit_model(model_key: str, train: pd.DataFrame) -> FitResult:
    if model_key == "chinchilla_endpoints":
        return fit_two_resource(train)
    if model_key == PARAMETER_DATA_MODEL_KEY:
        return fit_parameter_data_chinchilla(train)
    if model_key == "chinchilla_separate_base":
        return fit_separate_base(train)
    if model_key == "loglog_endpoints":
        return fit_loglog(train)
    if model_key == "loglog_separate_base":
        return fit_loglog_separate_base(train)
    raise ValueError(f"Unknown model key: {model_key}")


def predict_model(
    model_key: str,
    fit: FitResult,
    points: pd.DataFrame,
) -> np.ndarray:
    if model_key == "chinchilla_endpoints":
        assert isinstance(fit, TwoResourceFit)
        return predict_two_resource(fit, points)
    if model_key == PARAMETER_DATA_MODEL_KEY:
        assert isinstance(fit, ParameterDataChinchillaFit)
        return predict_parameter_data_chinchilla(fit, points)
    if model_key == "chinchilla_separate_base":
        assert isinstance(fit, SeparateBaseFit)
        return predict_separate_base(fit, points)
    if model_key == "loglog_endpoints":
        assert isinstance(fit, LogLogFit)
        return predict_loglog(fit, points)
    if model_key == "loglog_separate_base":
        assert isinstance(fit, LogLogSeparateBaseFit)
        return predict_loglog_separate_base(fit, points)
    raise ValueError(f"Unknown model key: {model_key}")


def points_for_model(model_key: str, endpoints: pd.DataFrame, base_points: pd.DataFrame) -> pd.DataFrame:
    if model_key in {"chinchilla_endpoints", "loglog_endpoints", PARAMETER_DATA_MODEL_KEY}:
        return endpoints.reset_index(drop=True)
    return pd.concat([endpoints, base_points], ignore_index=True).reset_index(drop=True)


def prediction_frame_for_model(
    model_key: str,
    points: pd.DataFrame,
    train: pd.DataFrame,
    fit: FitResult,
    labels: dict[str, str],
) -> pd.DataFrame:
    rows = points.copy()
    train_keys = set(train.index)
    rows["split"] = ["train" if idx in train_keys else "heldout" for idx in rows.index]
    rows["model_key"] = model_key
    rows["model_label"] = labels[model_key]
    rows["is_baseline"] = False
    rows["prediction"] = predict_model(model_key, fit, rows)
    rows["error_pct"] = (rows["prediction"] / rows["value"] - 1.0) * 100
    return rows.sort_values(["model_key", "split", "series", "tokens_b", "scale_flops"]).reset_index(drop=True)


def k020_baseline_predictions(
    endpoints: pd.DataFrame, train: pd.DataFrame
) -> tuple[pd.DataFrame, K020ChinchillaFit | None]:
    fit = fit_k020_chinchilla(train)
    if fit is None:
        return pd.DataFrame(), None
    rows = endpoints[endpoints["series"].eq(ISOFLOP_SERIES)].copy()
    train_keys = set(train.index)
    rows["split"] = ["train" if idx in train_keys else "heldout" for idx in rows.index]
    rows["model_key"] = "k020_chinchilla"
    rows["model_label"] = "K=0.20-only Chinchilla"
    rows["is_baseline"] = True
    rows["prediction"] = predict_k020_chinchilla(fit, rows)
    rows["error_pct"] = (rows["prediction"] / rows["value"] - 1.0) * 100
    return rows.sort_values(["split", "scale_flops"]).reset_index(drop=True), fit


def fit_all_models(
    endpoints: pd.DataFrame,
    base_points: pd.DataFrame,
    args: argparse.Namespace,
    labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FitResult | K020ChinchillaFit | None]]:
    predictions: list[pd.DataFrame] = []
    fits: dict[str, FitResult | K020ChinchillaFit | None] = {}
    for model_key in labels:
        points = points_for_model(model_key, endpoints, base_points)
        train, _ = split_points(points, args.fit_through_scale)
        fit = fit_model(model_key, train)
        fits[model_key] = fit
        predictions.append(prediction_frame_for_model(model_key, points, train, fit, labels))

    endpoints_train, _ = split_points(endpoints, args.fit_through_scale)
    baseline_predictions, k020_fit = k020_baseline_predictions(endpoints, endpoints_train)
    fits["k020_chinchilla"] = k020_fit
    if not baseline_predictions.empty:
        predictions.append(baseline_predictions)

    all_predictions = pd.concat(predictions, ignore_index=True)
    summary = summarize_predictions(all_predictions)
    return all_predictions, summary, fits


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_key", "model_label", "split", "is_baseline"]
    for (model_key, model_label, split, is_baseline), frame in predictions.groupby(group_cols, sort=False):
        actual = frame["value"].to_numpy(dtype=float)
        pred = frame["prediction"].to_numpy(dtype=float)
        error = frame["error_pct"].to_numpy(dtype=float)
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "split": split,
                "is_baseline": is_baseline,
                "n": len(frame),
                "mae_pct": float(np.mean(np.abs(error))),
                "rmse_pct": math.sqrt(float(np.mean(error**2))),
                "bias_pct": float(np.mean(error)),
                "loss_rmse": math.sqrt(float(np.mean((actual - pred) ** 2))),
                "r2": r2_score(actual, pred),
            }
        )
    return pd.DataFrame(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(str(path), "w") as handle:
        handle.write(text)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_text(path, frame.to_csv(index=False))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, default=finite_or_none))


def heldout_summary_rows(summary: pd.DataFrame) -> str:
    heldout = summary[summary["split"].eq("heldout")].copy()
    rows = []
    for _, row in heldout.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td>{int(row['n'])}</td>"
            f"<td>{float(row['mae_pct']):.2f}%</td>"
            f"<td>{float(row['rmse_pct']):.2f}%</td>"
            f"<td>{float(row['bias_pct']):+.2f}%</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_report_html(
    fig: go.Figure,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    included_series: list[str],
    labels: dict[str, str],
) -> str:
    has_isoflop = ISOFLOP_SERIES in included_series
    scope_label = "iso-token-only" if args.exclude_isoflop else "iso-token plus K=0.20"
    baseline_note = (
        "The black dashed <code>K=0.20-only Chinchilla</code> line is the old baseline fit using only the K=0.20 "
        "ladder; it stays visible as a comparison, not as a dropdown mode."
        if has_isoflop
        else "This iso-token-only report excludes the K=0.20 iso-FLOP ladder, so no K=0.20 baseline is shown."
    )
    series_items = ["<li><code>tok500m</code>, <code>tok1b</code>, ...: fixed midtraining-token budgets.</li>"]
    if has_isoflop:
        series_items.append("<li><code>K=0.20</code>: iso-FLOP budget; <code>D</code> grows with base scale.</li>")
    series_items.append("<li><code>base</code>: the model before midtraining, so <code>D=0</code>.</li>")
    parameter_mode_card = (
        """
    <section class="panel">
      <h2>Mode 5: Chinchilla params + data</h2>
      <p>
        Fits endpoints with the Chinchilla-style parameter/data form:
        <code>L(N,D_pre,D_mid)=E + A*N^-alpha + B*D_pre^-beta + G*D_mid^-gamma</code>.
      </p>
      <p>
        <code>N</code> is trainable parameters in billions from the canonical Delphi registry, and
        <code>D_pre</code> and <code>D_mid</code> are pretraining and midtraining tokens in billions.
      </p>
    </section>
"""
        if PARAMETER_DATA_MODEL_KEY in labels
        else ""
    )
    figure_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="midtraining-2d-fit")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi Midtraining 2D Chinchilla Fit</title>
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
      background: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 32px 44px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 17px;
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
      max-width: 980px;
      color: var(--muted);
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 22px 0 18px;
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
      max-width: 1040px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      max-width: 900px;
      margin: 8px 0 18px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 7px 8px;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .plot-wrap {{
      margin-top: 12px;
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 20px 16px 36px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Delphi midtraining prediction error by fit mode</h1>
  <p class="lede">
    This report tests whether a Chinchilla-style scaling fit should use only midtraining endpoints,
    or also include base-model math validation losses at zero midtraining tokens. It also compares
    the same two data choices under pure power-law log-log fits.
    Scope: <code>{html.escape(scope_label)}</code>.
    Here <code>C</code> is base pretraining FLOPs and <code>D_mid</code> is midtraining tokens in billions.
    The params+data mode replaces <code>C</code> with <code>N</code> parameters and
    <code>D_pre</code> pretraining tokens.
    The fit trains on scales up through <code>{html.escape(args.fit_through_scale)}</code>; <code>1e21</code>
    and <code>1e22</code> are held out.
  </p>

  <div class="callout">
    <strong>How to read the right panel:</strong>
    error is <code>prediction / actual - 1</code>. Positive means the model predicted too much loss;
    negative means it predicted too little loss. Values closer to zero are better.
    {baseline_note}
  </div>

  <div class="grid">
    <section class="panel">
      <h2>Mode 1: Chinchilla endpoints only</h2>
      <p>
        Fits only completed midtraining endpoints with the additive two-resource form:
        <code>L(C,D)=E + A*C^-alpha + B*D^-beta</code>.
      </p>
      <p>This is the current unconstrained model.</p>
    </section>
    <section class="panel">
      <h2>Mode 2: Chinchilla separate base</h2>
      <p>
        Fits a base-loss curve <code>L_base(C)</code> and subtracts a midtraining improvement term
        <code>I(C,D)</code>.
      </p>
      <p>The improvement is forced to be exactly zero at <code>D=0</code>.</p>
    </section>
    <section class="panel">
      <h2>Mode 3: log-log endpoints only</h2>
      <p>
        Fits endpoint losses as a pure power law:
        <code>log L = a + b*log C + c*log D</code>.
      </p>
      <p>This is the no-floor alternative to mode 1.</p>
    </section>
    <section class="panel">
      <h2>Mode 4: log-log separate base</h2>
      <p>
        Fits <code>L_base(C)</code> as a pure power law, then fits the midtraining improvement
        <code>L_base(C)-L(C,D)</code> as a pure power law in <code>C</code> and <code>D</code>.
      </p>
    </section>
    {parameter_mode_card}
  </div>

  <div class="grid">
    <section class="panel">
      <h2>Series</h2>
      <ul>
        {"".join(series_items)}
      </ul>
    </section>
    <section class="panel">
      <h2>Left Panel</h2>
      <p>
        Actual loss is on the x-axis and predicted loss is on the y-axis.
        Points on the dashed diagonal are perfectly predicted.
      </p>
      <p>Circle markers are training points; hexagon markers are held-out points.</p>
    </section>
    <section class="panel">
      <h2>Held-Out Summary</h2>
      <p>
        Separate-base modes include held-out base points, so their held-out <code>n</code> is larger than the
        endpoints-only modes.
      </p>
    </section>
  </div>

  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Held-out n</th>
        <th>MAE</th>
        <th>RMSE</th>
        <th>Bias</th>
      </tr>
    </thead>
    <tbody>
      {heldout_summary_rows(summary)}
    </tbody>
  </table>

  <div class="plot-wrap">
    {figure_html}
  </div>
</main>
</body>
</html>
"""


def build_figure(predictions: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Actual vs predicted loss", "Held-out prediction error"),
        horizontal_spacing=0.11,
    )
    model_predictions = predictions[~predictions["is_baseline"]].copy()
    baseline_predictions = predictions[predictions["is_baseline"]].copy()
    initial_model = "chinchilla_endpoints"
    trace_model_keys: list[str] = []
    series_order = sorted(
        model_predictions["series"].unique(),
        key=lambda series: (
            series == ISOFLOP_SERIES,
            float(model_predictions[model_predictions["series"].eq(series)]["tokens_b"].median()),
        ),
    )
    color_by_series = {series: PALETTE[i % len(PALETTE)] for i, series in enumerate(series_order)}
    marker_by_split = {"train": MARKERS[0], "heldout": MARKERS[10]}

    def add_model_trace(trace: go.Scatter, model_key: str, row: int, col: int) -> None:
        trace.visible = model_key == initial_model
        fig.add_trace(trace, row=row, col=col)
        trace_model_keys.append(model_key)

    def add_always_trace(trace: go.Scatter, row: int, col: int) -> None:
        trace.visible = True
        fig.add_trace(trace, row=row, col=col)
        trace_model_keys.append("always")

    for model_key in labels:
        mode_predictions = model_predictions[model_predictions["model_key"].eq(model_key)]
        for series in series_order:
            for split in ("train", "heldout"):
                frame = mode_predictions[mode_predictions["series"].eq(series) & mode_predictions["split"].eq(split)]
                if frame.empty:
                    continue
                add_model_trace(
                    go.Scatter(
                        x=frame["value"],
                        y=frame["prediction"],
                        mode="markers",
                        name=f"{series} {split}",
                        legendgroup=f"{model_key}-{series}-{split}",
                        marker={
                            "symbol": marker_by_split[split],
                            "size": 10 if split == "train" else 13,
                            "color": color_by_series[series],
                            "line": {"width": 1, "color": "#1f2937"},
                        },
                        customdata=np.stack(
                            [
                                frame["scale"].astype(str),
                                frame["tokens_b"].astype(float),
                                frame["pretrain_tokens_b"].astype(float),
                                frame["total_tokens_b"].astype(float),
                                frame["params_b"].astype(float),
                                frame["error_pct"].astype(float),
                                frame["model_label"].astype(str),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{customdata[6]}<br>series=%{fullData.name}<br>scale=%{customdata[0]}"
                            "<br>D_mid=%{customdata[1]:.2f}B<br>D_pre=%{customdata[2]:.2f}B"
                            "<br>D_total=%{customdata[3]:.2f}B<br>N=%{customdata[4]:.2f}B params"
                            "<br>actual=%{x:.5f}<br>pred=%{y:.5f}"
                            "<br>err=%{customdata[5]:+.2f}%<extra></extra>"
                        ),
                    ),
                    model_key,
                    row=1,
                    col=1,
                )

    lower = float(min(model_predictions["value"].min(), model_predictions["prediction"].min()))
    upper = float(max(model_predictions["value"].max(), model_predictions["prediction"].max()))
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="perfect prediction",
            line={"color": "#667085", "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    trace_model_keys.append("always")

    for model_key in labels:
        heldout = model_predictions[
            model_predictions["model_key"].eq(model_key) & model_predictions["split"].eq("heldout")
        ]
        for i, series in enumerate(series_order):
            frame = heldout[heldout["series"].eq(series)].sort_values("scale_flops")
            if frame.empty:
                continue
            add_model_trace(
                go.Scatter(
                    x=frame["scale"],
                    y=frame["error_pct"],
                    mode="lines+markers",
                    name=f"{series} heldout error",
                    legendgroup=f"{model_key}-{series}-error",
                    marker={"symbol": MARKERS[i % len(MARKERS)], "size": 10, "color": color_by_series[series]},
                    line={"color": color_by_series[series]},
                    customdata=np.stack(
                        [
                            frame["tokens_b"].astype(float),
                            frame["pretrain_tokens_b"].astype(float),
                            frame["total_tokens_b"].astype(float),
                            frame["params_b"].astype(float),
                            frame["value"].astype(float),
                            frame["prediction"].astype(float),
                            frame["model_label"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[6]}<br>series=%{fullData.name}<br>scale=%{x}"
                        "<br>D_mid=%{customdata[0]:.2f}B<br>D_pre=%{customdata[1]:.2f}B"
                        "<br>D_total=%{customdata[2]:.2f}B<br>N=%{customdata[3]:.2f}B params"
                        "<br>actual=%{customdata[4]:.5f}<br>pred=%{customdata[5]:.5f}"
                        "<br>error=%{y:+.2f}%<extra></extra>"
                    ),
                ),
                model_key,
                row=1,
                col=2,
            )

    k020_baseline = baseline_predictions[baseline_predictions["split"].eq("heldout")]
    if not k020_baseline.empty:
        add_always_trace(
            go.Scatter(
                x=k020_baseline["scale"],
                y=k020_baseline["error_pct"],
                mode="lines+markers",
                name="K=0.20-only Chinchilla",
                legendgroup="k0p20-only-error",
                marker={
                    "symbol": "diamond-open",
                    "size": 13,
                    "color": "#111827",
                    "line": {"width": 2, "color": "#111827"},
                },
                line={"color": "#111827", "dash": "dash", "width": 2},
                customdata=np.stack(
                    [
                        k020_baseline["tokens_b"].astype(float),
                        k020_baseline["pretrain_tokens_b"].astype(float),
                        k020_baseline["total_tokens_b"].astype(float),
                        k020_baseline["params_b"].astype(float),
                        k020_baseline["value"].astype(float),
                        k020_baseline["prediction"].astype(float),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "K=0.20-only Chinchilla<br>scale=%{x}<br>D_mid=%{customdata[0]:.2f}B"
                    "<br>D_pre=%{customdata[1]:.2f}B<br>D_total=%{customdata[2]:.2f}B"
                    "<br>N=%{customdata[3]:.2f}B params"
                    "<br>actual=%{customdata[4]:.5f}<br>pred=%{customdata[5]:.5f}"
                    "<br>error=%{y:+.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    buttons = []
    for model_key, label in labels.items():
        visible = [trace_key in ("always", model_key) for trace_key in trace_model_keys]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": "Delphi midtraining prediction error by fit mode<br>" f"<sup>{label}</sup>"},
                ],
            }
        )
    fig.add_hline(y=0, line={"color": "#667085", "width": 1}, row=1, col=2)
    fig.update_xaxes(title_text="actual loss", row=1, col=1)
    fig.update_yaxes(title_text="predicted loss", row=1, col=1)
    fig.update_xaxes(title_text="held-out scale", categoryorder="array", categoryarray=SCALE_ORDER, row=1, col=2)
    fig.update_yaxes(title_text="prediction error (pred / actual - 1) [%]", row=1, col=2)
    fig.update_layout(
        title=("Delphi midtraining prediction error by fit mode<br>" f"<sup>{labels[initial_model]}</sup>"),
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.17,
                "yanchor": "top",
            }
        ],
        width=1300,
        height=650,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 70, "r": 30, "t": 90, "b": 140},
    )
    return fig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    endpoints = load_endpoint_data(args)
    filtered = endpoints[endpoints["mix"].eq(args.mix) & endpoints["lr"].eq(args.lr)].copy()
    if args.exclude_isoflop:
        filtered = filtered[filtered["series"].ne(ISOFLOP_SERIES)].copy()
    if filtered.empty:
        raise ValueError(f"No endpoint rows for mix={args.mix!r}, lr={args.lr!r}")

    included_series = sorted(filtered["series"].unique())
    labels = model_labels(args.include_parameter_data_chinchilla)
    base_points = load_base_points(args)
    predictions, summary, fits = fit_all_models(filtered, base_points, args, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem
    if output_stem is None:
        base_stem = PARAMETER_DATA_OUTPUT_STEM if args.include_parameter_data_chinchilla else DEFAULT_OUTPUT_STEM
        output_stem = f"{base_stem}_isotoken_only" if args.exclude_isoflop else base_stem
    stem = args.output_dir / output_stem
    write_csv(stem.with_name(f"{stem.name}_predictions.csv"), predictions)
    write_csv(stem.with_name(f"{stem.name}_summary.csv"), summary)
    write_json(
        stem.with_name(f"{stem.name}_fit.json"),
        {
            "fit_through_scale": args.fit_through_scale,
            "mix": args.mix,
            "lr": args.lr,
            "exclude_isoflop": bool(args.exclude_isoflop),
            "included_series": included_series,
            "included_models": labels,
            "base_points": {
                "path": str(TRAJECTORY_POINTS_PATH),
                "metric_label": METRIC_LABEL,
                "n": len(base_points),
            },
            "fits": {key: None if value is None else asdict(value) for key, value in fits.items()},
        },
    )
    html_path = stem.with_suffix(".html")
    write_text(html_path, build_report_html(build_figure(predictions, labels), summary, args, included_series, labels))

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    heldout_cols = [
        "model_label",
        "series",
        "scale",
        "tokens_b",
        "value",
        "prediction",
        "error_pct",
    ]
    print()
    print(
        predictions[predictions["split"].eq("heldout")][heldout_cols].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print()
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
