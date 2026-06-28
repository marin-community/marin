# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a broad Delphi midtraining fit-family comparison report.

This report expands ``delphi_midtraining_2d_chinchilla.py`` from a small set of
hand-picked models into a registry of fit families and resource choices. It keeps
the same train/heldout split: fit through 3e20 by default, hold out 1e21/1e22.

Run:
    uv run --with scipy --with plotly --with pandas --with wandb \
      python scripts/analysis/delphi_midtraining_fit_family_report.py
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from build_delphi_midtraining_interactive_report import finite_or_none
from delphi_isotoken_endpoint_scaling import (
    DEFAULT_CUTOFF_SCALE,
    ISOFLOP_SERIES,
    OUT_DIR,
    SCALE_ORDER,
)
from delphi_midtraining_2d_chinchilla import (
    load_base_points,
    load_endpoint_data,
    split_points,
)
from delphi_small_final_loss_scaling import MATH_FRACTION
from marin.scaling_laws.scaling_plots import MARKERS, PALETTE
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

logger = logging.getLogger("delphi_midtraining_fit_family_report")

DEFAULT_OUTPUT_STEM = "delphi_midtraining_fit_family_report"


@dataclass(frozen=True)
class FitSpec:
    """One candidate model family and resource choice."""

    key: str
    label: str
    family: str
    features: tuple[str, ...]
    equation: str
    description: str
    include_base: bool = False
    base_features: tuple[str, ...] = ()
    scale_feature: str | None = None
    progress_feature: str | None = None


@dataclass(frozen=True)
class FittedModel:
    """Fitted model state used for prediction and serialization."""

    spec: FitSpec
    parameters: dict[str, float]
    r2: float | None
    rmse: float
    rmse_log: float | None
    n: int


@dataclass(frozen=True)
class FailedFit:
    """Fit failure recorded in the report instead of crashing the whole run."""

    spec: FitSpec
    error: str


RESOURCE_LABELS = {
    "pretrain_flops_e18": "C_pre",
    "params_b": "N",
    "pretrain_tokens_b": "D_pre",
    "tokens_b": "D_mid",
    "dmath_b": "D_math",
    "total_tokens_b": "D_total",
    "k_mid": "K_mid",
    "k_math": "K_math",
    "d_mid_per_param": "D_mid/N",
    "d_math_per_param": "D_math/N",
    "midtrain_flops_e18": "C_mid",
    "midtrain_math_flops_e18": "C_mid_math",
    "total_flops_e18": "C_total",
}

RESOURCE_DESCRIPTIONS = {
    "C_pre": "base pretraining FLOPs, normalized to 1e18 FLOPs",
    "N": "trainable parameters in billions",
    "D_pre": "pretraining tokens in billions",
    "D_mid": "total midtraining tokens in billions, including both prose and math mixture components",
    "D_math": "math midtraining tokens in billions: D_mid times the mixture's math fraction",
    "D_total": "D_pre + D_mid in billions",
    "K_mid": "D_mid / D_pre, the total midtraining-token budget as a pretraining-token fraction",
    "K_math": "D_math / D_pre, the math-token budget as a pretraining-token fraction",
    "D_mid/N": "total midtraining tokens per trainable parameter, both in billions",
    "D_math/N": "math midtraining tokens per trainable parameter, both in billions",
    "C_mid": "estimated midtraining FLOPs: C_pre * D_mid / D_pre",
    "C_mid_math": "estimated math-component midtraining FLOPs: C_pre * D_math / D_pre",
    "C_total": "C_pre + C_mid, normalized to 1e18 FLOPs",
}


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
        "--focus-series",
        default=None,
        help="Series used for the summary table's 1e22 prediction columns. Defaults to k0p20 when present.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def resource_label(columns: tuple[str, ...] | list[str]) -> str:
    unique_columns = list(dict.fromkeys(columns))
    return " + ".join(RESOURCE_LABELS[column] for column in unique_columns)


def finite_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    return frame[mask].copy()


def additive_inverse_power(feature_matrix: np.ndarray, *theta: float) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    out = np.full(features.shape[1], float(theta[0]))
    for index in range(features.shape[0]):
        amplitude = float(theta[1 + 2 * index])
        exponent = float(theta[2 + 2 * index])
        out += amplitude * np.power(features[index], -exponent)
    return out


def fit_additive_inverse_power(
    points: pd.DataFrame,
    features: tuple[str, ...],
    floor_upper: float | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    fit_points = finite_frame(points, features)
    if len(fit_points) != len(points):
        raise ValueError(f"nonpositive or nonfinite resource values for {resource_label(features)}")
    values = fit_points["value"].to_numpy(dtype=float)
    matrix = fit_points.loc[:, list(features)].to_numpy(dtype=float).T
    spread = max(float(values.max() - values.min()), 1e-3)
    floor0 = float(values.min()) * 0.3
    upper_floor = float(values.min()) * 0.999 if floor_upper is None else floor_upper
    initial = [floor0]
    lower = [0.0]
    upper = [upper_floor]
    for _ in features:
        initial.extend([spread / max(len(features), 1), 0.1])
        lower.extend([0.0, 0.0])
        upper.extend([np.inf, 5.0])
    fitted, _ = curve_fit(
        additive_inverse_power,
        matrix,
        values,
        p0=tuple(initial),
        bounds=(tuple(lower), tuple(upper)),
        maxfev=200_000,
    )
    parameters = {"floor": float(fitted[0])}
    for index, feature in enumerate(features):
        label = RESOURCE_LABELS[feature]
        parameters[f"{label}_amplitude"] = float(fitted[1 + 2 * index])
        parameters[f"{label}_exponent"] = float(fitted[2 + 2 * index])
    return parameters, np.asarray(fitted, dtype=float)


def predict_additive_inverse_power(
    frame: pd.DataFrame, features: tuple[str, ...], parameters: dict[str, float]
) -> np.ndarray:
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float).T
    theta = [parameters["floor"]]
    for feature in features:
        label = RESOURCE_LABELS[feature]
        theta.extend([parameters[f"{label}_amplitude"], parameters[f"{label}_exponent"]])
    return additive_inverse_power(matrix, *theta)


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


def fit_endpoint_chinchilla(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    parameters, _ = fit_additive_inverse_power(train, spec.features)
    actual = train["value"].to_numpy(dtype=float)
    predicted = predict_additive_inverse_power(train, spec.features, parameters)
    return FittedModel(
        spec=spec,
        parameters=parameters,
        r2=r2_score(actual, predicted),
        rmse=math.sqrt(float(np.mean((actual - predicted) ** 2))),
        rmse_log=rmse_log_score(actual, predicted),
        n=len(train),
    )


def fit_endpoint_loglog(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    fit_points = finite_frame(train, spec.features)
    if len(fit_points) != len(train):
        raise ValueError(f"log-log fit has nonpositive resource values for {resource_label(spec.features)}")
    actual = fit_points["value"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(fit_points))] + [np.log(fit_points[feature].to_numpy(dtype=float)) for feature in spec.features]
    )
    theta, *_ = np.linalg.lstsq(design, np.log(actual), rcond=None)
    parameters = {"intercept": float(theta[0])}
    for index, feature in enumerate(spec.features):
        parameters[f"{RESOURCE_LABELS[feature]}_slope"] = float(theta[index + 1])
    predicted = predict_loglog(fit_points, spec.features, parameters)
    return FittedModel(
        spec=spec,
        parameters=parameters,
        r2=r2_score(actual, predicted),
        rmse=math.sqrt(float(np.mean((actual - predicted) ** 2))),
        rmse_log=rmse_log_score(actual, predicted),
        n=len(fit_points),
    )


def predict_loglog(frame: pd.DataFrame, features: tuple[str, ...], parameters: dict[str, float]) -> np.ndarray:
    log_prediction = np.full(len(frame), parameters["intercept"], dtype=float)
    for feature in features:
        log_prediction += parameters[f"{RESOURCE_LABELS[feature]}_slope"] * np.log(frame[feature].to_numpy(dtype=float))
    return np.exp(log_prediction)


def separate_chinchilla_model(
    feature_matrix: np.ndarray,
    num_base_features: int,
    floor: float,
    *theta: float,
) -> np.ndarray:
    features = np.asarray(feature_matrix, dtype=float)
    base = np.full(features.shape[1], floor)
    offset = 0
    for index in range(num_base_features):
        amplitude = theta[offset]
        exponent = theta[offset + 1]
        base += amplitude * np.power(features[index], -exponent)
        offset += 2
    scale_feature = features[num_base_features]
    progress_feature = features[num_base_features + 1]
    improvement_amplitude = theta[offset]
    improvement_scale_exponent = theta[offset + 1]
    improvement_progress_exponent = theta[offset + 2]
    improvement_progress_scale = theta[offset + 3]
    progress = 1.0 - np.power(1.0 + progress_feature / improvement_progress_scale, -improvement_progress_exponent)
    improvement = improvement_amplitude * np.power(scale_feature, -improvement_scale_exponent) * progress
    return base - improvement


def fit_separate_chinchilla(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    if not spec.base_features or spec.scale_feature is None or spec.progress_feature is None:
        raise ValueError("separate Chinchilla spec is missing base/scale/progress features")
    all_features = (*spec.base_features, spec.scale_feature, spec.progress_feature)
    train = train.copy()
    values = train["value"].to_numpy(dtype=float)
    positive_progress = train.loc[train[spec.progress_feature].gt(0), spec.progress_feature].to_numpy(dtype=float)
    if positive_progress.size == 0:
        raise ValueError("separate Chinchilla fit needs positive-progress endpoint rows")
    base_points = train[train["series"].eq("base")].copy()
    if base_points.empty:
        raise ValueError("separate Chinchilla fit needs base rows")
    base_parameters, _ = fit_additive_inverse_power(
        base_points, spec.base_features, floor_upper=float(values.max()) * 1.5
    )
    initial = [base_parameters["floor"]]
    lower = [0.0]
    upper = [float(values.max()) * 1.5]
    for feature in spec.base_features:
        label = RESOURCE_LABELS[feature]
        initial.extend([base_parameters[f"{label}_amplitude"], base_parameters[f"{label}_exponent"]])
        lower.extend([0.0, 0.0])
        upper.extend([np.inf, 5.0])
    base_pred = predict_additive_inverse_power(train, spec.base_features, base_parameters)
    improvement0 = max(float(np.median(np.maximum(base_pred - values, 0.0))), 1e-3)
    initial.extend([improvement0, 0.05, 0.25, float(np.median(positive_progress))])
    lower.extend([0.0, 0.0, 0.0, 1e-12])
    upper.extend([np.inf, 5.0, 5.0, float(positive_progress.max()) * 100.0])
    matrix = train.loc[:, list(all_features)].to_numpy(dtype=float).T

    def model(feature_matrix: np.ndarray, *theta: float) -> np.ndarray:
        return separate_chinchilla_model(
            feature_matrix,
            len(spec.base_features),
            float(theta[0]),
            *theta[1:],
        )

    fitted, _ = curve_fit(
        model,
        matrix,
        values,
        p0=tuple(initial),
        bounds=(tuple(lower), tuple(upper)),
        maxfev=200_000,
    )
    parameters = {"base_floor": float(fitted[0])}
    offset = 1
    for feature in spec.base_features:
        label = RESOURCE_LABELS[feature]
        parameters[f"base_{label}_amplitude"] = float(fitted[offset])
        parameters[f"base_{label}_exponent"] = float(fitted[offset + 1])
        offset += 2
    parameters.update(
        {
            "improvement_amplitude": float(fitted[offset]),
            f"improvement_{RESOURCE_LABELS[spec.scale_feature]}_exponent": float(fitted[offset + 1]),
            f"improvement_{RESOURCE_LABELS[spec.progress_feature]}_exponent": float(fitted[offset + 2]),
            f"improvement_{RESOURCE_LABELS[spec.progress_feature]}_scale": float(fitted[offset + 3]),
        }
    )
    predicted = predict_separate_chinchilla(train, spec, parameters)
    return FittedModel(
        spec=spec,
        parameters=parameters,
        r2=r2_score(values, predicted),
        rmse=math.sqrt(float(np.mean((values - predicted) ** 2))),
        rmse_log=rmse_log_score(values, predicted),
        n=len(train),
    )


def predict_separate_chinchilla(frame: pd.DataFrame, spec: FitSpec, parameters: dict[str, float]) -> np.ndarray:
    if spec.scale_feature is None or spec.progress_feature is None:
        raise ValueError("separate Chinchilla spec is missing scale/progress features")
    base = np.full(len(frame), parameters["base_floor"], dtype=float)
    for feature in spec.base_features:
        label = RESOURCE_LABELS[feature]
        base += parameters[f"base_{label}_amplitude"] * np.power(
            frame[feature].to_numpy(dtype=float),
            -parameters[f"base_{label}_exponent"],
        )
    progress_feature = frame[spec.progress_feature].to_numpy(dtype=float)
    progress_label = RESOURCE_LABELS[spec.progress_feature]
    scale_label = RESOURCE_LABELS[spec.scale_feature]
    progress = 1.0 - np.power(
        1.0 + progress_feature / parameters[f"improvement_{progress_label}_scale"],
        -parameters[f"improvement_{progress_label}_exponent"],
    )
    improvement = (
        parameters["improvement_amplitude"]
        * np.power(frame[spec.scale_feature].to_numpy(dtype=float), -parameters[f"improvement_{scale_label}_exponent"])
        * progress
    )
    return base - improvement


def fit_separate_log_improvement(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    if not spec.base_features or spec.scale_feature is None or spec.progress_feature is None:
        raise ValueError("separate log-improvement spec is missing base/scale/progress features")
    base_points = train[train["series"].eq("base")].copy()
    endpoints = train[train[spec.progress_feature].gt(0)].copy()
    if base_points.empty or endpoints.empty:
        raise ValueError("separate log-improvement fit needs base rows and positive-progress endpoint rows")
    base_parameters, _ = fit_additive_inverse_power(
        base_points,
        spec.base_features,
        floor_upper=float(base_points["value"].max()) * 1.5,
    )
    endpoint_base = predict_additive_inverse_power(endpoints, spec.base_features, base_parameters)
    improvement = endpoint_base - endpoints["value"].to_numpy(dtype=float)
    if np.any(improvement <= 0):
        bad = endpoints.loc[improvement <= 0, ["series", "scale", "value", spec.progress_feature]]
        raise ValueError(f"nonpositive improvement rows:\n{bad.to_string(index=False)}")
    design = np.column_stack(
        [
            np.ones(len(endpoints)),
            np.log(endpoints[spec.scale_feature].to_numpy(dtype=float)),
            np.log(endpoints[spec.progress_feature].to_numpy(dtype=float)),
        ]
    )
    theta, *_ = np.linalg.lstsq(design, np.log(improvement), rcond=None)
    parameters = {f"base_{key}": value for key, value in base_parameters.items()}
    parameters.update(
        {
            "improvement_intercept": float(theta[0]),
            f"improvement_{RESOURCE_LABELS[spec.scale_feature]}_slope": float(theta[1]),
            f"improvement_{RESOURCE_LABELS[spec.progress_feature]}_slope": float(theta[2]),
        }
    )
    actual = train["value"].to_numpy(dtype=float)
    predicted = predict_separate_log_improvement(train, spec, parameters)
    return FittedModel(
        spec=spec,
        parameters=parameters,
        r2=r2_score(actual, predicted),
        rmse=math.sqrt(float(np.mean((actual - predicted) ** 2))),
        rmse_log=rmse_log_score(actual, predicted),
        n=len(train),
    )


def predict_separate_log_improvement(frame: pd.DataFrame, spec: FitSpec, parameters: dict[str, float]) -> np.ndarray:
    if spec.scale_feature is None or spec.progress_feature is None:
        raise ValueError("separate log-improvement spec is missing scale/progress features")
    base_parameters = {key.removeprefix("base_"): value for key, value in parameters.items() if key.startswith("base_")}
    base = predict_additive_inverse_power(frame, spec.base_features, base_parameters)
    improvement = np.zeros(len(frame), dtype=float)
    positive = frame[spec.progress_feature].to_numpy(dtype=float) > 0
    if positive.any():
        scale_label = RESOURCE_LABELS[spec.scale_feature]
        progress_label = RESOURCE_LABELS[spec.progress_feature]
        improvement[positive] = np.exp(
            parameters["improvement_intercept"]
            + parameters[f"improvement_{scale_label}_slope"]
            * np.log(frame.loc[positive, spec.scale_feature].to_numpy(dtype=float))
            + parameters[f"improvement_{progress_label}_slope"]
            * np.log(frame.loc[positive, spec.progress_feature].to_numpy(dtype=float))
        )
    return base - improvement


def fit_model(spec: FitSpec, train: pd.DataFrame) -> FittedModel:
    if spec.family == "endpoint_chinchilla":
        return fit_endpoint_chinchilla(spec, train)
    if spec.family == "endpoint_loglog":
        return fit_endpoint_loglog(spec, train)
    if spec.family == "separate_chinchilla":
        return fit_separate_chinchilla(spec, train)
    if spec.family == "separate_log_improvement":
        return fit_separate_log_improvement(spec, train)
    raise ValueError(f"Unknown fit family: {spec.family}")


def predict_model(model: FittedModel, frame: pd.DataFrame) -> np.ndarray:
    spec = model.spec
    if spec.family == "endpoint_chinchilla":
        return predict_additive_inverse_power(frame, spec.features, model.parameters)
    if spec.family == "endpoint_loglog":
        return predict_loglog(frame, spec.features, model.parameters)
    if spec.family == "separate_chinchilla":
        return predict_separate_chinchilla(frame, spec, model.parameters)
    if spec.family == "separate_log_improvement":
        return predict_separate_log_improvement(frame, spec, model.parameters)
    raise ValueError(f"Unknown fit family: {spec.family}")


def endpoint_chinchilla_specs() -> list[FitSpec]:
    resources = [
        ("pretrain_flops_e18", "tokens_b"),
        ("pretrain_flops_e18", "dmath_b"),
        ("pretrain_flops_e18", "k_mid"),
        ("pretrain_flops_e18", "k_math"),
        ("pretrain_flops_e18", "midtrain_flops_e18"),
        ("pretrain_flops_e18", "midtrain_math_flops_e18"),
        ("pretrain_tokens_b", "tokens_b"),
        ("pretrain_tokens_b", "dmath_b"),
        ("pretrain_tokens_b", "k_math"),
        ("params_b", "tokens_b"),
        ("params_b", "dmath_b"),
        ("params_b", "d_math_per_param"),
        ("params_b", "pretrain_tokens_b", "tokens_b"),
        ("params_b", "pretrain_tokens_b", "dmath_b"),
        ("params_b", "pretrain_tokens_b", "k_math"),
        ("params_b", "pretrain_tokens_b", "d_math_per_param"),
        ("params_b", "pretrain_tokens_b", "midtrain_math_flops_e18"),
        ("params_b", "total_tokens_b"),
    ]
    specs = []
    for features in resources:
        label = resource_label(features)
        specs.append(
            FitSpec(
                key=f"chinchilla_{'_'.join(features)}",
                label=f"Chinchilla endpoints: {label}",
                family="endpoint_chinchilla",
                features=features,
                equation=f"L = E + additive inverse powers over {label}",
                description="Endpoint-only Chinchilla-style floor plus one inverse-power term per resource.",
            )
        )
    return specs


def endpoint_loglog_specs() -> list[FitSpec]:
    specs = []
    for base in endpoint_chinchilla_specs():
        label = resource_label(base.features)
        specs.append(
            FitSpec(
                key=base.key.replace("chinchilla_", "loglog_"),
                label=f"log-log endpoints: {label}",
                family="endpoint_loglog",
                features=base.features,
                equation=f"log L = intercept + slopes over log({label})",
                description="Endpoint-only pure power law with no irreducible floor.",
            )
        )
    return specs


def separate_specs() -> list[FitSpec]:
    saturating = [
        ("pretrain_flops_e18", ("pretrain_flops_e18",), "pretrain_flops_e18", "tokens_b"),
        ("pretrain_flops_e18", ("pretrain_flops_e18",), "pretrain_flops_e18", "dmath_b"),
        ("pretrain_flops_e18", ("pretrain_flops_e18",), "pretrain_flops_e18", "k_math"),
        ("pretrain_tokens_b", ("pretrain_tokens_b",), "pretrain_tokens_b", "dmath_b"),
        ("params_b", ("params_b", "pretrain_tokens_b"), "params_b", "dmath_b"),
        ("params_b", ("params_b", "pretrain_tokens_b"), "params_b", "midtrain_math_flops_e18"),
    ]
    specs = []
    for _, base_features, scale_feature, progress_feature in saturating:
        base_label = resource_label(base_features)
        progress_label = RESOURCE_LABELS[progress_feature]
        specs.append(
            FitSpec(
                key=f"separate_chinchilla_{'_'.join(base_features)}_{progress_feature}",
                label=f"separate Chinchilla: base {base_label}, progress {progress_label}",
                family="separate_chinchilla",
                features=(*base_features, scale_feature, progress_feature),
                base_features=base_features,
                scale_feature=scale_feature,
                progress_feature=progress_feature,
                include_base=True,
                equation=f"L = L_base({base_label}) - I({RESOURCE_LABELS[scale_feature]}, {progress_label})",
                description="Fits base rows at D=0 and a saturating midtraining improvement that is exactly zero at D=0.",
            )
        )
    log_improvement = [
        (("pretrain_flops_e18",), "pretrain_flops_e18", "tokens_b"),
        (("pretrain_flops_e18",), "pretrain_flops_e18", "dmath_b"),
        (("pretrain_flops_e18",), "pretrain_flops_e18", "k_math"),
        (("pretrain_tokens_b",), "pretrain_tokens_b", "dmath_b"),
        (("params_b", "pretrain_tokens_b"), "params_b", "dmath_b"),
        (("params_b", "pretrain_tokens_b"), "params_b", "midtrain_math_flops_e18"),
    ]
    for base_features, scale_feature, progress_feature in log_improvement:
        base_label = resource_label(base_features)
        progress_label = RESOURCE_LABELS[progress_feature]
        specs.append(
            FitSpec(
                key=f"separate_log_{'_'.join(base_features)}_{progress_feature}",
                label=f"separate log-improvement: base {base_label}, progress {progress_label}",
                family="separate_log_improvement",
                features=(*base_features, scale_feature, progress_feature),
                base_features=base_features,
                scale_feature=scale_feature,
                progress_feature=progress_feature,
                include_base=True,
                equation=f"L = L_base({base_label}) - exp(a + b log {RESOURCE_LABELS[scale_feature]} + c log {progress_label})",
                description="Fits base rows first, then fits the loss improvement as a power law on positive-progress endpoints.",
            )
        )
    return specs


def all_specs() -> list[FitSpec]:
    return [*endpoint_chinchilla_specs(), *endpoint_loglog_specs(), *separate_specs()]


def enrich_resources(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["math_fraction"] = out["mix"].map(MATH_FRACTION).astype(float)
    out["pretrain_flops_e18"] = out["scale_flops"].astype(float) / 1e18
    out["dmath_b"] = out["tokens_b"].astype(float) * out["math_fraction"]
    out["total_tokens_b"] = out["pretrain_tokens_b"].astype(float) + out["tokens_b"].astype(float)
    out["k_mid"] = out["tokens_b"].astype(float) / out["pretrain_tokens_b"].astype(float)
    out["k_math"] = out["dmath_b"] / out["pretrain_tokens_b"].astype(float)
    out["d_mid_per_param"] = out["tokens_b"].astype(float) / out["params_b"].astype(float)
    out["d_math_per_param"] = out["dmath_b"] / out["params_b"].astype(float)
    token_ratio = out["tokens_b"].astype(float) / out["pretrain_tokens_b"].astype(float)
    math_token_ratio = out["dmath_b"] / out["pretrain_tokens_b"].astype(float)
    out["midtrain_flops_e18"] = out["pretrain_flops_e18"] * token_ratio
    out["midtrain_math_flops_e18"] = out["pretrain_flops_e18"] * math_token_ratio
    out["total_flops_e18"] = out["pretrain_flops_e18"] + out["midtrain_flops_e18"]
    return out


def default_focus_series(endpoints: pd.DataFrame) -> str:
    if ISOFLOP_SERIES in set(endpoints["series"]):
        return ISOFLOP_SERIES
    candidates = endpoints[endpoints["scale"].eq("1e22")]
    if candidates.empty:
        candidates = endpoints
    tokens = candidates.groupby("series")["tokens_b"].median().sort_values()
    return str(tokens.index[-1])


def split_labels(points: pd.DataFrame, train: pd.DataFrame) -> list[str]:
    train_keys = set(train.index)
    return ["train" if index in train_keys else "heldout" for index in points.index]


def prediction_frame_for_model(model: FittedModel, points: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows = points.copy()
    rows["split"] = split_labels(rows, train)
    rows["model_key"] = model.spec.key
    rows["model_label"] = model.spec.label
    rows["family"] = model.spec.family
    rows["resources"] = resource_label(model.spec.features)
    rows["prediction"] = predict_model(model, rows)
    rows["error_pct"] = (rows["prediction"] / rows["value"] - 1.0) * 100.0
    return rows.sort_values(["model_key", "split", "series", "tokens_b", "scale_flops"]).reset_index(drop=True)


def fit_all(
    endpoints: pd.DataFrame,
    base_points: pd.DataFrame,
    fit_through_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FittedModel], list[FailedFit]]:
    predictions: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    fitted_models: list[FittedModel] = []
    failures: list[FailedFit] = []
    for spec in all_specs():
        points = pd.concat([endpoints, base_points], ignore_index=True) if spec.include_base else endpoints.copy()
        train, _ = split_points(points, fit_through_scale)
        try:
            model = fit_model(spec, train)
            frame = prediction_frame_for_model(model, points, train)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(FailedFit(spec=spec, error=str(exc)))
            continue
        predictions.append(frame)
        fitted_models.append(model)
        fit_rows.append(
            {
                "model_key": spec.key,
                "model_label": spec.label,
                "family": spec.family,
                "resources": resource_label(spec.features),
                "equation": spec.equation,
                "train_n": model.n,
                "fit_r2": model.r2,
                "fit_rmse": model.rmse,
                "fit_rmse_log": model.rmse_log,
                "parameters": json.dumps(model.parameters, sort_keys=True),
            }
        )
    if not predictions:
        raise ValueError("No candidate fits succeeded")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fit_rows), fitted_models, failures


def summarize_predictions(predictions: pd.DataFrame, fit_table: pd.DataFrame, focus_series: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_key, model_label, family, resources), frame in predictions.groupby(
        ["model_key", "model_label", "family", "resources"],
        sort=False,
    ):
        endpoint_heldout = frame[frame["split"].eq("heldout") & frame["series"].ne("base")]
        base_heldout = frame[frame["split"].eq("heldout") & frame["series"].eq("base")]
        fit_meta = fit_table[fit_table["model_key"].eq(model_key)].iloc[0]
        error = endpoint_heldout["error_pct"].to_numpy(dtype=float)
        loss_error = endpoint_heldout["prediction"].to_numpy(dtype=float) - endpoint_heldout["value"].to_numpy(
            dtype=float
        )
        focus = endpoint_heldout[endpoint_heldout["scale"].eq("1e22") & endpoint_heldout["series"].eq(focus_series)]
        one_e22 = endpoint_heldout[endpoint_heldout["scale"].eq("1e22")]
        row = {
            "model_key": model_key,
            "model_label": model_label,
            "family": family,
            "resources": resources,
            "train_n": int(fit_meta["train_n"]),
            "heldout_endpoint_n": len(endpoint_heldout),
            "heldout_mae_pct": float(np.mean(np.abs(error))) if error.size else None,
            "heldout_rmse_pct": math.sqrt(float(np.mean(error**2))) if error.size else None,
            "heldout_bias_pct": float(np.mean(error)) if error.size else None,
            "heldout_loss_rmse": math.sqrt(float(np.mean(loss_error**2))) if loss_error.size else None,
            "base_heldout_mae_pct": (
                float(np.mean(np.abs(base_heldout["error_pct"]))) if not base_heldout.empty else None
            ),
            "1e22_mae_pct": float(np.mean(np.abs(one_e22["error_pct"]))) if not one_e22.empty else None,
            "1e22_bias_pct": float(np.mean(one_e22["error_pct"])) if not one_e22.empty else None,
            "focus_series": focus_series,
            "1e22_focus_actual": float(focus["value"].iloc[0]) if not focus.empty else None,
            "1e22_focus_prediction": float(focus["prediction"].iloc[0]) if not focus.empty else None,
            "1e22_focus_error_pct": float(focus["error_pct"].iloc[0]) if not focus.empty else None,
            "fit_r2": None if pd.isna(fit_meta["fit_r2"]) else float(fit_meta["fit_r2"]),
            "fit_rmse": float(fit_meta["fit_rmse"]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["heldout_mae_pct", "1e22_mae_pct", "model_label"]).reset_index(drop=True)


def one_e22_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions[predictions["scale"].eq("1e22") & predictions["series"].ne("base")].copy()
    rows["abs_error_pct"] = rows["error_pct"].abs()
    return (
        rows[
            [
                "model_label",
                "family",
                "resources",
                "series",
                "value",
                "prediction",
                "error_pct",
                "abs_error_pct",
                "tokens_b",
                "dmath_b",
                "params_b",
                "pretrain_tokens_b",
            ]
        ]
        .sort_values(["abs_error_pct", "series", "model_label"])
        .reset_index(drop=True)
    )


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
    limit: int | None = None,
    table_class: str = "",
    default_sort_key: str | None = None,
    default_sort_direction: str = "asc",
) -> str:
    shown = frame if limit is None else frame.head(limit)
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
    for _, row in shown.iterrows():
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
            f"<td>{html.escape(resource_label(failure.spec.features))}</td>"
            f"<td><code>{html.escape(failure.error[:500])}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>model</th><th>family</th><th>resources</th><th>reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_figure(predictions: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Actual vs predicted loss", "Held-out endpoint error", "1e22 actual vs predicted"),
        horizontal_spacing=0.09,
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]],
    )
    initial_model = str(summary.iloc[0]["model_key"])
    trace_model_keys: list[str] = []
    series_order = sorted(
        predictions["series"].unique(),
        key=lambda series: (
            series == "base",
            series == ISOFLOP_SERIES,
            float(predictions[predictions["series"].eq(series)]["tokens_b"].median()),
        ),
    )
    color_by_series = {series: PALETTE[index % len(PALETTE)] for index, series in enumerate(series_order)}
    marker_by_split = {"train": MARKERS[0], "heldout": MARKERS[9]}

    def add_model_trace(trace: go.BaseTraceType, model_key: str, row: int, col: int) -> None:
        trace.visible = model_key == initial_model
        fig.add_trace(trace, row=row, col=col)
        trace_model_keys.append(model_key)

    def add_always_trace(trace: go.BaseTraceType, row: int, col: int) -> None:
        trace.visible = True
        fig.add_trace(trace, row=row, col=col)
        trace_model_keys.append("always")

    for model_key, model_frame in predictions.groupby("model_key", sort=False):
        for series in series_order:
            for split in ("train", "heldout"):
                frame = model_frame[model_frame["series"].eq(series) & model_frame["split"].eq(split)]
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
                            "size": 9 if split == "train" else 13,
                            "color": "#64748b" if series == "base" else color_by_series[series],
                            "line": {"width": 1, "color": "#111827"},
                        },
                        customdata=np.stack(
                            [
                                frame["model_label"].astype(str),
                                frame["scale"].astype(str),
                                frame["series"].astype(str),
                                frame["params_b"].astype(float),
                                frame["pretrain_tokens_b"].astype(float),
                                frame["tokens_b"].astype(float),
                                frame["dmath_b"].astype(float),
                                frame["k_math"].astype(float),
                                frame["error_pct"].astype(float),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{customdata[0]}<br>series=%{customdata[2]} %{fullData.name}<br>scale=%{customdata[1]}"
                            "<br>N=%{customdata[3]:.3f}B params<br>D_pre=%{customdata[4]:.3f}B"
                            "<br>D_mid=%{customdata[5]:.3f}B<br>D_math=%{customdata[6]:.3f}B"
                            "<br>K_math=%{customdata[7]:.5f}<br>actual=%{x:.5f}<br>pred=%{y:.5f}"
                            "<br>err=%{customdata[8]:+.2f}%<extra></extra>"
                        ),
                    ),
                    model_key,
                    row=1,
                    col=1,
                )

        heldout = model_frame[model_frame["split"].eq("heldout") & model_frame["series"].ne("base")]
        for index, series in enumerate(series_order):
            if series == "base":
                continue
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
                    marker={"symbol": MARKERS[index % len(MARKERS)], "size": 10, "color": color_by_series[series]},
                    line={"color": color_by_series[series]},
                    customdata=np.stack(
                        [
                            frame["model_label"].astype(str),
                            frame["value"].astype(float),
                            frame["prediction"].astype(float),
                            frame["tokens_b"].astype(float),
                            frame["dmath_b"].astype(float),
                            frame["k_math"].astype(float),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>series=%{fullData.name}<br>scale=%{x}"
                        "<br>D_mid=%{customdata[3]:.3f}B<br>D_math=%{customdata[4]:.3f}B"
                        "<br>K_math=%{customdata[5]:.5f}<br>actual=%{customdata[1]:.5f}"
                        "<br>pred=%{customdata[2]:.5f}<br>error=%{y:+.2f}%<extra></extra>"
                    ),
                ),
                model_key,
                row=1,
                col=2,
            )

        one_e22 = model_frame[model_frame["scale"].eq("1e22") & model_frame["series"].ne("base")].copy()
        if not one_e22.empty:
            one_e22 = one_e22.sort_values(["tokens_b", "series"])
            add_model_trace(
                go.Bar(
                    x=one_e22["series"],
                    y=one_e22["value"],
                    name="1e22 actual",
                    legendgroup=f"{model_key}-1e22",
                    marker={"color": "#94a3b8"},
                    offsetgroup="actual",
                    customdata=np.stack(
                        [
                            one_e22["model_label"].astype(str),
                            one_e22["tokens_b"].astype(float),
                            one_e22["dmath_b"].astype(float),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>series=%{x}<br>D_mid=%{customdata[1]:.3f}B"
                        "<br>D_math=%{customdata[2]:.3f}B<br>actual=%{y:.5f}<extra></extra>"
                    ),
                ),
                model_key,
                row=1,
                col=3,
            )
            add_model_trace(
                go.Bar(
                    x=one_e22["series"],
                    y=one_e22["prediction"],
                    name="1e22 predicted",
                    legendgroup=f"{model_key}-1e22",
                    marker={"color": "#1877F2"},
                    offsetgroup="predicted",
                    customdata=np.stack(
                        [
                            one_e22["model_label"].astype(str),
                            one_e22["error_pct"].astype(float),
                            one_e22["tokens_b"].astype(float),
                            one_e22["dmath_b"].astype(float),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>series=%{x}<br>D_mid=%{customdata[2]:.3f}B"
                        "<br>D_math=%{customdata[3]:.3f}B<br>pred=%{y:.5f}"
                        "<br>error=%{customdata[1]:+.2f}%<extra></extra>"
                    ),
                ),
                model_key,
                row=1,
                col=3,
            )

    lower = float(min(predictions["value"].min(), predictions["prediction"].min()))
    upper = float(max(predictions["value"].max(), predictions["prediction"].max()))
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
    buttons = []
    labels = dict(zip(summary["model_key"], summary["model_label"], strict=False))
    for model_key, label in labels.items():
        visible = [trace_key in ("always", model_key) for trace_key in trace_model_keys]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Delphi midtraining fit-family comparison<br><sup>{html.escape(label)}</sup>"},
                ],
            }
        )
    fig.update_xaxes(title_text="actual loss", row=1, col=1)
    fig.update_yaxes(title_text="predicted loss", row=1, col=1)
    fig.update_xaxes(title_text="held-out scale", categoryorder="array", categoryarray=SCALE_ORDER, row=1, col=2)
    fig.update_yaxes(title_text="prediction error (pred / actual - 1) [%]", row=1, col=2)
    fig.update_xaxes(title_text="1e22 series", row=1, col=3)
    fig.update_yaxes(title_text="math val loss", row=1, col=3)
    fig.update_layout(
        title=f"Delphi midtraining fit-family comparison<br><sup>{labels[initial_model]}</sup>",
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
    args: argparse.Namespace,
    focus_series: str,
    included_series: list[str],
) -> str:
    figure_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="delphi-fit-family")
    summary_columns = [
        ("model_label", "model", 3, ""),
        ("family", "family", 3, ""),
        ("resources", "resources", 3, ""),
        ("heldout_endpoint_n", "heldout n", 0, ""),
        ("heldout_mae_pct", "heldout MAE", 2, "%"),
        ("heldout_rmse_pct", "heldout RMSE", 2, "%"),
        ("1e22_mae_pct", "1e22 MAE", 2, "%"),
        ("1e22_focus_prediction", f"1e22 pred ({focus_series})", 5, ""),
        ("1e22_focus_actual", f"1e22 actual ({focus_series})", 5, ""),
        ("1e22_focus_error_pct", f"1e22 err ({focus_series})", 2, "%"),
        ("base_heldout_mae_pct", "base heldout MAE", 2, "%"),
    ]
    one_e22_columns = [
        ("model_label", "model", 3, ""),
        ("series", "series", 3, ""),
        ("resources", "resources", 3, ""),
        ("value", "actual", 5, ""),
        ("prediction", "prediction", 5, ""),
        ("error_pct", "signed error", 2, "%"),
        ("abs_error_pct", "abs error", 2, "%"),
        ("tokens_b", "D_mid B", 3, ""),
        ("dmath_b", "D_math B", 3, ""),
    ]
    scope = "iso-token only" if args.exclude_isoflop else "iso-token plus K=0.20"
    series_text = ", ".join(f"<code>{html.escape(series)}</code>" for series in included_series)
    failure_html = failure_table_html(failures)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi Midtraining Fit Family Report</title>
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
  <h1>Delphi midtraining fit-family comparison</h1>
  <p class="lede">
    This report compares endpoint-only and base-constrained fits for <code>math_val_loss</code>.
    Scope: <code>{html.escape(scope)}</code>; mix <code>{html.escape(args.mix)}</code>; LR suffix
    <code>{html.escape(str(args.lr))}</code>; fit through <code>{html.escape(args.fit_through_scale)}</code>.
    The heldout scales are <code>1e21</code> and <code>1e22</code>. The summary table is sorted by heldout endpoint MAE
    and includes explicit <code>1e22</code> prediction columns for focus series <code>{html.escape(focus_series)}</code>.
  </p>

  <div class="callout">
    <strong>Important:</strong> in the current default data slice, <code>{html.escape(args.mix)}</code> has a fixed
    math fraction, so <code>D_math</code> is a constant rescaling of <code>D_mid</code>. With a free power-law amplitude,
    those two fits can make the same predictions. The math-token axis is still included because it is the semantically
    correct resource for math validation and becomes identifiable when pooling mixes.
  </div>

  <div class="grid">
    <section class="panel">
      <h2>Fit Families</h2>
      <ul>
        <li><code>endpoint Chinchilla</code>: <code>L = E + sum_i A_i x_i^-alpha_i</code> on finished midtraining endpoints.</li>
        <li><code>endpoint log-log</code>: <code>log L = a + sum_i b_i log x_i</code>, no floor.</li>
        <li><code>separate Chinchilla</code>: fits <code>L_base</code> using step-0 rows and subtracts a saturating improvement that is zero at <code>D=0</code>.</li>
        <li><code>separate log-improvement</code>: fits <code>L_base - L_midtrained</code> as a power law on positive-token endpoints.</li>
      </ul>
    </section>
    <section class="panel">
      <h2>Resources</h2>
      {resource_definition_html()}
    </section>
    <section class="panel">
      <h2>Series</h2>
      <p>{series_text}</p>
      <p><code>base</code> appears only in separate-base fits and comes from step-0 math validation rows.</p>
    </section>
  </div>

  <h2>Heldout Summary</h2>
  <p class="lede">
    Main error columns are computed only on heldout endpoint rows, so separate-base fits are comparable to endpoint-only
    fits. The optional base-heldout column reports how well modes that include step-0 rows extrapolate base losses.
  </p>
  <div class="table-wrap">
    {table_html(summary, summary_columns, table_class="sortable")}
  </div>

  <div class="plot-wrap">
    {figure_html}
  </div>

  <details>
    <summary>All 1e22 endpoint predictions</summary>
    <div class="table-wrap">
      {table_html(one_e22, one_e22_columns, table_class="sortable", default_sort_key="abs_error_pct")}
    </div>
  </details>

  <details>
    <summary>Fit failures</summary>
    <div class="table-wrap">
      {failure_html}
    </div>
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    endpoints = enrich_resources(load_endpoint_data(args))
    filtered = endpoints[endpoints["mix"].eq(args.mix) & endpoints["lr"].astype(str).eq(str(args.lr))].copy()
    if args.exclude_isoflop:
        filtered = filtered[filtered["series"].ne(ISOFLOP_SERIES)].copy()
    if filtered.empty:
        raise ValueError(f"No endpoint rows for mix={args.mix!r}, lr={args.lr!r}")
    base_points = enrich_resources(load_base_points(args))
    included_series = sorted(
        filtered["series"].unique(),
        key=lambda series: (
            series == ISOFLOP_SERIES,
            float(filtered[filtered["series"].eq(series)]["tokens_b"].median()),
        ),
    )
    focus_series = args.focus_series or default_focus_series(filtered)
    predictions, fit_table, models, failures = fit_all(filtered, base_points, args.fit_through_scale)
    summary = summarize_predictions(predictions, fit_table, focus_series)
    one_e22 = one_e22_predictions(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{args.output_stem}_isotoken_only" if args.exclude_isoflop else args.output_stem
    stem = args.output_dir / output_stem
    html_path = stem.with_suffix(".html")
    write_csv(stem.with_name(f"{stem.name}_predictions.csv"), predictions)
    write_csv(stem.with_name(f"{stem.name}_summary.csv"), summary)
    write_csv(stem.with_name(f"{stem.name}_fit_table.csv"), fit_table)
    write_csv(stem.with_name(f"{stem.name}_1e22_predictions.csv"), one_e22)
    write_json(
        stem.with_name(f"{stem.name}_fit.json"),
        {
            "fit_through_scale": args.fit_through_scale,
            "mix": args.mix,
            "lr": args.lr,
            "exclude_isoflop": bool(args.exclude_isoflop),
            "focus_series": focus_series,
            "included_series": included_series,
            "succeeded_models": [model.spec.key for model in models],
            "failed_models": [
                {
                    "model_key": failure.spec.key,
                    "model_label": failure.spec.label,
                    "family": failure.spec.family,
                    "resources": resource_label(failure.spec.features),
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
            build_figure(predictions, summary), summary, one_e22, failures, args, focus_series, included_series
        ),
    )

    print(summary.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if failures:
        print()
        print(f"{len(failures)} fit(s) failed; see {stem.with_name(f'{stem.name}_fit.json')}")
    print()
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
