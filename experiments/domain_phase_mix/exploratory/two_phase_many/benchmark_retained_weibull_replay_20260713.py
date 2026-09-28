# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

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
"""Benchmark compact retained-response surrogates.

The main model separates useful retained learning from literal data replay:

    z_i = exp(-lambda (1 - w_i^(1))) e_i^(0) + eta e_i^(1)
    q_i = e_i^(0) + e_i^(1)
    L = b - sum_i a_i (1 - exp(-(rho z_i)^p))
          + c sum_i [q_i - 1]_+^2.

The forgetting rate lambda says how quickly phase-0 learning is lost when a
bucket is not revisited in phase 1, while eta is the relative value of a late
epoch. The shared exponent p represents a distribution of learning timescales;
p=1 recovers exponential saturation. Since exposure is measured in simulated
epochs, one shared rho is the parsimonious default. Repetition harm depends on
total exposure, begins only after a bucket is replayed, and has one global
coefficient rather than an unidentifiable coefficient per bucket.

All bucket-specific parameters are nonnegative benefit amplitudes. The script
compares constant retention, revisit-dependent retention, exponential and
unsaturated-power responses, and a shared replay-harm ablation using refit
out-of-fold predictions and untouched 300M policies. It does not average or
stack surrogate predictions. Completed OOF predictions are checkpointed and
reused on resubmission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_debugger_300m as legacy_exporter,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/retained_weibull_replay_20260713"
DASHBOARD_DATA_PATH = SCRIPT_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DATASET_NAMES = ("300m_uncheatable", "300m_table9", "production_uncheatable")
CONFIRMATORY_MODELS = (
    "revisit_retention_power",
    "revisit_retention_weibull_shared_replay",
    "revisit_retention_weibull_token_replay",
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
POWER_BOUNDS = (0.2, 1.0)
LOG_RATE_BOUNDS = (math.log(0.05), math.log(20.0))
LOG_LATE_MULTIPLIER_BOUNDS = (math.log(0.1), math.log(20.0))
FORGETTING_RATE_BOUNDS = (0.0, 8.0)


def observatory_api():
    """Load the Observatory lazily so it can import these model primitives."""
    from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
        export_mixture_fit_observatory as observatory,
    )

    return observatory


class ResponseKind(StrEnum):
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    POWER = "power"


class SignalKind(StrEnum):
    RETAINED_STATE = "retained_state"
    TOTAL_EXPOSURE = "total_exposure"
    SEPARATE_PHASES = "separate_phases"
    CUMULATIVE_RECENCY = "cumulative_recency"


class RetentionKind(StrEnum):
    CONSTANT = "constant"
    REVISIT_GATED = "revisit_gated"
    POISSON_REFRESH = "poisson_refresh"


class ReplayPenaltyKind(StrEnum):
    NONE = "none"
    SHARED = "shared"
    TOKEN_WEIGHTED = "token_weighted"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    signal: SignalKind
    response: ResponseKind
    retention: RetentionKind
    replay_penalty: ReplayPenaltyKind
    family_coverage: bool = False


@dataclass(frozen=True)
class Shape:
    rate: float
    late_rate: float
    power: float
    late_multiplier: float
    forgetting_rate: float


@dataclass(frozen=True)
class FittedModel:
    config: ModelConfig
    shape: Shape
    intercept: float
    signal_coef: np.ndarray
    replay_coef: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    family_members: tuple[np.ndarray, ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        signal, replay = response_features(
            weights,
            self.c0,
            self.c1,
            self.config,
            self.shape,
            self.family_members,
        )
        prediction = self.intercept - signal @ self.signal_coef
        if self.config.replay_penalty is not ReplayPenaltyKind.NONE:
            prediction = prediction + replay @ self.replay_coef
        return np.asarray(prediction, dtype=float)


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    model: str
    l2: float
    seed: int
    nominal_parameter_count: int
    oof_rmse: float
    oof_spearman: float
    fold_mean_regret_at_1: float
    global_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float


CONFIGS = (
    ModelConfig(
        "constant_retention_weibull",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.CONSTANT,
        ReplayPenaltyKind.NONE,
    ),
    ModelConfig(
        "revisit_retention_weibull",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.REVISIT_GATED,
        ReplayPenaltyKind.NONE,
    ),
    ModelConfig(
        "revisit_retention_exponential",
        SignalKind.RETAINED_STATE,
        ResponseKind.EXPONENTIAL,
        RetentionKind.REVISIT_GATED,
        ReplayPenaltyKind.NONE,
    ),
    ModelConfig(
        "revisit_retention_power",
        SignalKind.RETAINED_STATE,
        ResponseKind.POWER,
        RetentionKind.REVISIT_GATED,
        ReplayPenaltyKind.NONE,
    ),
    ModelConfig(
        "revisit_retention_weibull_shared_replay",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.REVISIT_GATED,
        ReplayPenaltyKind.SHARED,
    ),
    ModelConfig(
        "poisson_refresh_weibull_shared_replay",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.POISSON_REFRESH,
        ReplayPenaltyKind.SHARED,
    ),
    ModelConfig(
        "revisit_retention_weibull_token_replay",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.REVISIT_GATED,
        ReplayPenaltyKind.TOKEN_WEIGHTED,
    ),
    ModelConfig(
        "poisson_refresh_weibull_token_replay",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.POISSON_REFRESH,
        ReplayPenaltyKind.TOKEN_WEIGHTED,
    ),
    ModelConfig(
        "revisit_retention_weibull_family_replay",
        SignalKind.RETAINED_STATE,
        ResponseKind.WEIBULL,
        RetentionKind.REVISIT_GATED,
        ReplayPenaltyKind.SHARED,
        True,
    ),
    ModelConfig(
        "separate_phase_weibull_shared_replay",
        SignalKind.SEPARATE_PHASES,
        ResponseKind.WEIBULL,
        RetentionKind.CONSTANT,
        ReplayPenaltyKind.SHARED,
    ),
    ModelConfig(
        "cumulative_recency_weibull_shared_replay",
        SignalKind.CUMULATIVE_RECENCY,
        ResponseKind.WEIBULL,
        RetentionKind.CONSTANT,
        ReplayPenaltyKind.SHARED,
    ),
)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_string_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def dataset_by_name(name: str) -> pooled.Dataset:
    if name == "300m_uncheatable":
        return pooled.load_300m_dataset("uncheatable")
    if name == "300m_table9":
        return pooled.load_300m_dataset("table9")
    if name == "production_uncheatable":
        return pooled.load_production_dataset()
    raise ValueError(f"Unknown dataset {name!r}")


def family_members(dataset: pooled.Dataset, config: ModelConfig) -> tuple[np.ndarray, ...]:
    if not config.family_coverage or not dataset.name.startswith("300m_"):
        return ()
    family_map = legacy_exporter.grp_packet(dataset).family_map
    return tuple(np.asarray(indices, dtype=int) for _name, indices in sorted(family_map.items()))


def response_features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    config: ModelConfig,
    shape: Shape,
    family_members: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    e0 = weights[:, 0, :] * c0[None, :]
    e1 = weights[:, 1, :] * c1[None, :]
    total_exposure = e0 + e1
    if config.signal is SignalKind.TOTAL_EXPOSURE:
        signal = response_link(total_exposure, shape.rate, shape.power, config.response)
        if config.family_coverage and family_members:
            family_signal = np.column_stack(
                [
                    response_link(
                        total_exposure[:, members].mean(axis=1, keepdims=True),
                        shape.rate,
                        shape.power,
                        config.response,
                    )[:, 0]
                    for members in family_members
                ]
            )
            signal = np.hstack([signal, family_signal])
    elif config.signal is SignalKind.RETAINED_STATE:
        if config.retention is RetentionKind.CONSTANT:
            retained_phase0 = e0
        elif config.retention is RetentionKind.REVISIT_GATED:
            revisit = np.clip(weights[:, 1, :], 0.0, 1.0)
            retained_phase0 = np.exp(-shape.forgetting_rate * (1.0 - revisit)) * e0
        elif config.retention is RetentionKind.POISSON_REFRESH:
            retention_floor = np.exp(-shape.forgetting_rate)
            retained_fraction = 1.0 - (1.0 - retention_floor) * np.exp(-e1)
            retained_phase0 = retained_fraction * e0
        else:
            raise ValueError(f"Unsupported retention {config.retention}")
        retained = np.maximum(retained_phase0 + shape.late_multiplier * e1, 0.0)
        signal = response_link(retained, shape.rate, shape.power, config.response)
        if config.family_coverage and family_members:
            family_signal = np.column_stack(
                [
                    response_link(
                        retained[:, members].mean(axis=1, keepdims=True),
                        shape.rate,
                        shape.power,
                        config.response,
                    )[:, 0]
                    for members in family_members
                ]
            )
            signal = np.hstack([signal, family_signal])
    elif config.signal is SignalKind.SEPARATE_PHASES:
        signal = np.hstack(
            [
                response_link(e0, shape.rate, shape.power, config.response),
                response_link(e1, shape.late_rate, shape.power, config.response),
            ]
        )
    elif config.signal is SignalKind.CUMULATIVE_RECENCY:
        signal = np.hstack(
            [
                response_link(total_exposure, shape.rate, shape.power, config.response),
                response_link(e1, shape.late_rate, shape.power, config.response),
            ]
        )
    else:
        raise ValueError(f"Unsupported signal {config.signal}")
    repeated_epochs = np.maximum(total_exposure - 1.0, 0.0) ** 2
    if config.replay_penalty is ReplayPenaltyKind.TOKEN_WEIGHTED:
        relative_bucket_tokens = 1.0 / np.maximum(c0 + c1, 1e-12)
        relative_bucket_tokens /= relative_bucket_tokens.mean()
        replay = repeated_epochs @ relative_bucket_tokens[:, None]
    else:
        replay = np.sum(repeated_epochs, axis=1, keepdims=True)
    return signal, replay


def response_link(exposure: np.ndarray, rate: float, power: float, response: ResponseKind) -> np.ndarray:
    exposure = np.maximum(exposure, 0.0)
    if response is ResponseKind.WEIBULL:
        return -np.expm1(-((rate * exposure) ** power))
    if response is ResponseKind.EXPONENTIAL:
        return -np.expm1(-(rate * exposure))
    if response is ResponseKind.POWER:
        return np.maximum(exposure, 1e-12) ** power
    raise ValueError(f"Unsupported response {response}")


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    config: ModelConfig,
    shape: Shape,
    family_members: tuple[np.ndarray, ...],
) -> np.ndarray:
    signal, replay = response_features(weights, c0, c1, config, shape, family_members)
    pieces = [-signal]
    if config.replay_penalty is not ReplayPenaltyKind.NONE:
        pieces.append(replay)
    return np.hstack(pieces)


def fit_nonnegative_head(design: np.ndarray, target: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    feature_scale = np.sqrt(np.mean(design**2, axis=0))
    feature_scale = np.maximum(feature_scale, 1e-8)
    scaled_design = design / feature_scale[None, :]
    design_mean = scaled_design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = scaled_design - design_mean
    centered_target = target - target_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    scaled_coef, _residual = nnls(centered_design, centered_target, maxiter=50 * design.shape[1])
    coef = scaled_coef / feature_scale
    intercept = target_mean - float((design.mean(axis=0, keepdims=True) @ coef).item())
    return intercept, np.asarray(coef, dtype=float)


def decode_shape(theta: np.ndarray, config: ModelConfig) -> Shape:
    if config.signal is SignalKind.TOTAL_EXPOSURE:
        return Shape(
            rate=float(np.exp(theta[0])),
            late_rate=float(np.exp(theta[0])),
            power=float(theta[1]),
            late_multiplier=1.0,
            forgetting_rate=0.0,
        )
    if config.signal is not SignalKind.RETAINED_STATE:
        return Shape(
            rate=float(np.exp(theta[0])),
            late_rate=float(np.exp(theta[1])),
            power=float(theta[2]),
            late_multiplier=1.0,
            forgetting_rate=0.0,
        )
    cursor = 0
    if config.response in (ResponseKind.WEIBULL, ResponseKind.EXPONENTIAL):
        rate = float(np.exp(theta[cursor]))
        cursor += 1
    else:
        rate = 1.0
    if config.response in (ResponseKind.WEIBULL, ResponseKind.POWER):
        power = float(theta[cursor])
        cursor += 1
    else:
        power = 1.0
    late_multiplier = float(np.exp(theta[cursor]))
    cursor += 1
    forgetting_rate = (
        float(theta[cursor]) if config.retention in (RetentionKind.REVISIT_GATED, RetentionKind.POISSON_REFRESH) else 0.0
    )
    return Shape(
        rate=rate,
        late_rate=rate,
        power=power,
        late_multiplier=late_multiplier,
        forgetting_rate=forgetting_rate,
    )


def shape_bounds(config: ModelConfig) -> list[tuple[float, float]]:
    if config.signal is SignalKind.TOTAL_EXPOSURE:
        return [LOG_RATE_BOUNDS, POWER_BOUNDS]
    if config.signal is not SignalKind.RETAINED_STATE:
        return [LOG_RATE_BOUNDS, LOG_RATE_BOUNDS, POWER_BOUNDS]
    bounds: list[tuple[float, float]] = []
    if config.response in (ResponseKind.WEIBULL, ResponseKind.EXPONENTIAL):
        bounds.append(LOG_RATE_BOUNDS)
    if config.response in (ResponseKind.WEIBULL, ResponseKind.POWER):
        bounds.append(POWER_BOUNDS)
    bounds.append(LOG_LATE_MULTIPLIER_BOUNDS)
    if config.retention in (RetentionKind.REVISIT_GATED, RetentionKind.POISSON_REFRESH):
        bounds.append(FORGETTING_RATE_BOUNDS)
    return bounds


def shape_starts(config: ModelConfig) -> tuple[np.ndarray, ...]:
    if config.signal is SignalKind.TOTAL_EXPOSURE:
        return tuple(
            np.asarray([math.log(rate), power], dtype=float) for rate in (0.25, 1.0, 4.0) for power in (0.34, 0.67, 1.0)
        )
    if config.signal is not SignalKind.RETAINED_STATE:
        return tuple(
            np.asarray([math.log(rate), math.log(late_rate), power], dtype=float)
            for rate in (0.25, 1.0, 4.0)
            for late_rate in (0.25, 1.0, 4.0)
            for power in (0.34, 0.67, 1.0)
        )
    rates = (0.25, 1.0, 4.0) if config.response in (ResponseKind.WEIBULL, ResponseKind.EXPONENTIAL) else (1.0,)
    powers = (0.34, 0.67, 1.0) if config.response in (ResponseKind.WEIBULL, ResponseKind.POWER) else (1.0,)
    late_multipliers = (1.0, 4.0, 10.0)
    forgetting_rates = (
        (0.0, 1.0, 3.0) if config.retention in (RetentionKind.REVISIT_GATED, RetentionKind.POISSON_REFRESH) else (0.0,)
    )
    starts = []
    for rate in rates:
        for power in powers:
            for late_multiplier in late_multipliers:
                for forgetting_rate in forgetting_rates:
                    values = []
                    if config.response in (ResponseKind.WEIBULL, ResponseKind.EXPONENTIAL):
                        values.append(math.log(rate))
                    if config.response in (ResponseKind.WEIBULL, ResponseKind.POWER):
                        values.append(power)
                    values.append(math.log(late_multiplier))
                    if config.retention in (RetentionKind.REVISIT_GATED, RetentionKind.POISSON_REFRESH):
                        values.append(forgetting_rate)
                    starts.append(np.asarray(values, dtype=float))
    return tuple(starts)


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: ModelConfig,
    l2: float,
    *,
    maxiter: int,
    top_k: int,
) -> FittedModel:
    weights = dataset.weights[indices]
    target = dataset.y[indices]
    families = family_members(dataset, config)

    def objective(theta: np.ndarray) -> float:
        shape = decode_shape(np.asarray(theta, dtype=float), config)
        design = design_matrix(weights, dataset.c0, dataset.c1, config, shape, families)
        intercept, coef = fit_nonnegative_head(design, target, l2)
        residual = intercept + design @ coef - target
        return float(np.mean(residual**2))

    scored = sorted(
        ((objective(start), start) for start in shape_starts(config)),
        key=lambda candidate: candidate[0],
    )
    best_value, best_theta = scored[0]
    for _score, start in scored[:top_k]:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=shape_bounds(config),
            options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 30},
        )
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)
    shape = decode_shape(best_theta, config)
    design = design_matrix(weights, dataset.c0, dataset.c1, config, shape, families)
    intercept, coef = fit_nonnegative_head(design, target, l2)
    split = signal_width(dataset, config)
    replay_coef = (
        np.asarray(coef[split:], dtype=float) if config.replay_penalty is not ReplayPenaltyKind.NONE else np.zeros(1)
    )
    return FittedModel(
        config=config,
        shape=shape,
        intercept=intercept,
        signal_coef=np.asarray(coef[:split], dtype=float),
        replay_coef=replay_coef,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        family_members=families,
    )


def nominal_parameter_count(dataset: pooled.Dataset, config: ModelConfig) -> int:
    if config.signal is SignalKind.TOTAL_EXPOSURE:
        linear = signal_width(dataset, config) + 1 + int(config.replay_penalty is not ReplayPenaltyKind.NONE)
        return linear + 2
    if config.signal is not SignalKind.RETAINED_STATE:
        linear = signal_width(dataset, config) + 1 + int(config.replay_penalty is not ReplayPenaltyKind.NONE)
        return linear + 3
    nonlinear = 1
    if config.response in (ResponseKind.WEIBULL, ResponseKind.EXPONENTIAL):
        nonlinear += 1
    if config.response in (ResponseKind.WEIBULL, ResponseKind.POWER):
        nonlinear += 1
    nonlinear += int(config.retention in (RetentionKind.REVISIT_GATED, RetentionKind.POISSON_REFRESH))
    linear = signal_width(dataset, config) + 1 + int(config.replay_penalty is not ReplayPenaltyKind.NONE)
    return nonlinear + linear


def signal_width(dataset: pooled.Dataset, config: ModelConfig) -> int:
    if config.signal in (SignalKind.RETAINED_STATE, SignalKind.TOTAL_EXPOSURE):
        return dataset.m + len(family_members(dataset, config))
    return 2 * dataset.m


def metric_row(
    dataset: pooled.Dataset,
    config: ModelConfig,
    l2: float,
    prediction: np.ndarray,
    seed: int,
) -> MetricRow:
    observatory = observatory_api()
    fold_indices = [test for _train, test in observatory.folds(dataset, seed)]
    summary = observatory.metric_summary(dataset.y, prediction, fold_test_indices=fold_indices)
    required = ("rmse", "spearman", "foldMeanRegretAt1", "regretAt1", "lowerTailOptimism", "lowTailRmse")
    if any(summary[key] is None for key in required):
        raise ValueError(f"Incomplete metrics for {dataset.name}/{config.name}: {summary}")
    return MetricRow(
        dataset=dataset.name,
        model=config.name,
        l2=l2,
        seed=seed,
        nominal_parameter_count=nominal_parameter_count(dataset, config),
        oof_rmse=float(summary["rmse"]),
        oof_spearman=float(summary["spearman"]),
        fold_mean_regret_at_1=float(summary["foldMeanRegretAt1"]),
        global_regret_at_1=float(summary["regretAt1"]),
        lower_tail_optimism=float(summary["lowerTailOptimism"]),
        low_tail_rmse=float(summary["lowTailRmse"]),
    )


def checkpoint_stem(dataset: str, config: ModelConfig, l2: float, seed: int) -> str:
    l2_key = str(l2).replace(".", "p")
    return f"{dataset}__{config.name}__l2_{l2_key}__seed_{seed}"


def oof_prediction(
    dataset: pooled.Dataset,
    config: ModelConfig,
    l2: float,
    seed: int,
    output_dir: Path,
    *,
    maxiter: int,
    top_k: int,
    force: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    observatory = observatory_api()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_stem(dataset.name, config, l2, seed)
    prediction_path = checkpoint_dir / f"{stem}.npy"
    parameter_path = checkpoint_dir / f"{stem}.json"
    if not force and prediction_path.exists() and parameter_path.exists():
        return np.load(prediction_path), json.loads(parameter_path.read_text())
    prediction = np.full(dataset.n, np.nan, dtype=float)
    parameters = []
    folds = observatory.folds(dataset, seed)
    for fold_id, (train, test) in enumerate(folds):
        print(
            f"{dataset.name}/{config.name}/l2={l2:g}/seed={seed}: fold={fold_id + 1}/{len(folds)}",
            flush=True,
        )
        model = fit_model(dataset, train, config, l2, maxiter=maxiter, top_k=top_k)
        prediction[test] = model.predict(dataset.weights[test])
        parameters.append(
            {
                "dataset": dataset.name,
                "model": config.name,
                "l2": l2,
                "seed": seed,
                "fold": fold_id,
                "rate": model.shape.rate,
                "late_rate": model.shape.late_rate,
                "power": model.shape.power,
                "late_multiplier": model.shape.late_multiplier,
                "forgetting_rate": model.shape.forgetting_rate,
                "active_signal_count": int(np.sum(model.signal_coef > 1e-10)),
                "active_replay_count": int(np.sum(model.replay_coef > 1e-10)),
            }
        )
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete OOF prediction for {stem}")
    np.save(prediction_path, prediction)
    parameter_path.write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n")
    return prediction, parameters


def evaluate_untouched_policies(
    fitted_models: dict[tuple[str, str, float], FittedModel],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observatory = observatory_api()
    dashboard = json.loads(DASHBOARD_DATA_PATH.read_text())
    swarm = dashboard["swarms"]["300m"]
    rows = swarm["rows"]
    weights = np.stack([np.asarray([row["phase0"], row["phase1"]], dtype=float) for row in rows])
    metric_rows = []
    prediction_rows = []
    for target, dataset_name in (("uncheatable", "300m_uncheatable"), ("table9", "300m_table9")):
        observed = np.asarray(
            [np.nan if row["observed"].get(target) is None else row["observed"][target] for row in rows],
            dtype=float,
        )
        masks = {
            "all": np.asarray(
                [row["split"] == "heldout" and row["phaseFamily"] == observatory.TWO_PHASE for row in rows]
            ),
        }
        panels = {
            row["panel"] for row in rows if row["split"] == "heldout" and row["phaseFamily"] == observatory.TWO_PHASE
        }
        for panel in sorted(panels):
            masks[panel] = np.asarray(
                [
                    row["split"] == "heldout" and row["phaseFamily"] == observatory.TWO_PHASE and row["panel"] == panel
                    for row in rows
                ]
            )
        references = swarm["predictions"][target][observatory.TWO_PHASE]
        predictions = {
            "effective_exposure": np.asarray(references["effective_exposure"]["fullFitPrediction"], dtype=float),
            "grp": np.asarray(references["grp"]["fullFitPrediction"], dtype=float),
        }
        for (candidate_dataset, model_name, l2), model in fitted_models.items():
            if candidate_dataset == dataset_name:
                predictions[f"{model_name}_l2_{l2:g}"] = model.predict(weights)
        all_mask = masks["all"] & np.isfinite(observed)
        for model_name, prediction in predictions.items():
            for row_index in np.flatnonzero(all_mask & np.isfinite(prediction)):
                row = rows[row_index]
                prediction_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "row_id": row["id"],
                        "run_name": row["name"],
                        "panel": row["panel"],
                        "intervention_type": row["interventionType"],
                        "target_domain": row["targetDomain"],
                        "observed": observed[row_index],
                        "prediction": prediction[row_index],
                        "residual": prediction[row_index] - observed[row_index],
                        "support_distance": row["diagnostics"]["supportDistance"],
                        "max_epoch": row["diagnostics"]["maxEpoch"],
                    }
                )
        for panel, base_mask in masks.items():
            mask = base_mask & np.isfinite(observed)
            if not mask.any():
                continue
            for model_name, prediction in predictions.items():
                summary = observatory.metric_summary(observed[mask], prediction[mask])
                metric_rows.append(
                    {
                        "target": target,
                        "panel": panel,
                        "model": model_name,
                        "n": int(mask.sum()),
                        "rmse": summary["rmse"],
                        "spearman": summary["spearman"],
                        "regret_at_1": summary["regretAt1"],
                        "lower_tail_optimism": summary["lowerTailOptimism"],
                        "low_tail_rmse": summary["lowTailRmse"],
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    aggregate = aggregate_metrics(metrics)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("OOF RMSE", "OOF Spearman"))
    colors = dict(
        zip(
            aggregate["dataset"].drop_duplicates(),
            reversed(px.colors.diverging.RdYlGn),
            strict=False,
        )
    )
    for dataset, frame in aggregate.groupby("dataset", sort=False):
        labels = frame["model"] + " / L2=" + frame["l2"].astype(str)
        for metric, column in (("oof_rmse", 1), ("oof_spearman", 2)):
            error_y = None
            if metric == "oof_rmse":
                error_y = {"type": "data", "array": frame["oof_rmse_sd"].fillna(0.0), "visible": True}
            figure.add_bar(
                x=labels,
                y=frame[metric],
                error_y=error_y,
                name=dataset,
                legendgroup=dataset,
                showlegend=column == 1,
                marker_color=colors[dataset],
                row=1,
                col=column,
            )
    figure.update_layout(
        title="Compact retained-response surrogate screen",
        template="plotly_white",
        barmode="group",
        height=620,
    )
    figure.write_html(output_dir / "clean_response_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["dataset", "model", "l2", "nominal_parameter_count"], as_index=False)
        .agg(
            num_cv_seeds=("seed", "nunique"),
            oof_rmse=("oof_rmse", "mean"),
            oof_rmse_sd=("oof_rmse", "std"),
            oof_spearman=("oof_spearman", "mean"),
            fold_mean_regret_at_1=("fold_mean_regret_at_1", "mean"),
            global_regret_at_1=("global_regret_at_1", "mean"),
            lower_tail_optimism=("lower_tail_optimism", "mean"),
            low_tail_rmse=("low_tail_rmse", "mean"),
        )
        .sort_values(["dataset", "oof_rmse", "lower_tail_optimism"])
    )


def observatory_baselines() -> pd.DataFrame:
    observatory = observatory_api()
    dashboard = json.loads(DASHBOARD_DATA_PATH.read_text())
    rows = []
    for swarm, target, dataset in (
        ("300m", "uncheatable", "300m_uncheatable"),
        ("300m", "table9", "300m_table9"),
        ("production", "uncheatable", "production_uncheatable"),
    ):
        for model_name, fit in dashboard["swarms"][swarm]["fits"][target][observatory.TWO_PHASE].items():
            diagnostics = fit["diagnostics"]["oof"]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "nominal_parameter_count": fit["parameterCount"],
                    "oof_rmse": diagnostics["rmse"],
                    "oof_spearman": diagnostics["spearman"],
                    "fold_mean_regret_at_1": diagnostics["foldMeanRegretAt1"],
                    "global_regret_at_1": diagnostics["regretAt1"],
                    "lower_tail_optimism": diagnostics["lowerTailOptimism"],
                    "low_tail_rmse": diagnostics["lowTailRmse"],
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "oof_rmse"])


def write_report(
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    heldout: pd.DataFrame,
    heldout_predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    aggregate = aggregate_metrics(metrics)
    selected = (
        aggregate.loc[aggregate["model"].eq("revisit_retention_weibull_shared_replay")]
        .sort_values(["dataset", "oof_rmse"])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    shape_summary = (
        parameters.groupby(["dataset", "model", "l2"], as_index=False)
        .agg(
            mean_rate=("rate", "mean"),
            mean_late_rate=("late_rate", "mean"),
            mean_power=("power", "mean"),
            mean_late_multiplier=("late_multiplier", "mean"),
            late_multiplier_sd=("late_multiplier", "std"),
            mean_forgetting_rate=("forgetting_rate", "mean"),
            forgetting_rate_sd=("forgetting_rate", "std"),
        )
        .sort_values(["dataset", "model", "l2"])
    )
    heldout_all = heldout.loc[heldout["panel"].eq("all")]
    candidate_predictions = heldout_predictions.loc[
        heldout_predictions["model"].str.startswith("revisit_retention_weibull_shared_replay")
    ].copy()
    candidate_predictions["absolute_residual"] = candidate_predictions["residual"].abs()
    largest_residuals = (
        candidate_predictions.sort_values(["target", "absolute_residual"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(8)
        .loc[
            :,
            [
                "target",
                "model",
                "run_name",
                "panel",
                "observed",
                "prediction",
                "residual",
                "support_distance",
                "max_epoch",
            ],
        ]
    )
    baselines = observatory_baselines()
    effective_exposure = baselines.loc[baselines["model"].eq("effective_exposure")]
    selected_seed0 = selected.loc[:, ["dataset", "l2", "nominal_parameter_count"]].merge(
        metrics.loc[
            metrics["model"].eq("revisit_retention_weibull_shared_replay") & metrics["seed"].eq(0),
            ["dataset", "l2", "oof_rmse", "oof_spearman"],
        ],
        on=["dataset", "l2"],
    )
    candidate_comparison = selected_seed0.merge(
        effective_exposure.loc[:, ["dataset", "oof_rmse", "oof_spearman"]],
        on="dataset",
        suffixes=("_retained", "_effective_exposure"),
    )
    candidate_comparison["rmse_delta_vs_effective_exposure"] = (
        candidate_comparison["oof_rmse_retained"] - candidate_comparison["oof_rmse_effective_exposure"]
    )
    candidate_comparison["spearman_delta_vs_effective_exposure"] = (
        candidate_comparison["oof_spearman_retained"] - candidate_comparison["oof_spearman_effective_exposure"]
    )
    comparison_columns = [
        "dataset",
        "l2",
        "nominal_parameter_count",
        "oof_rmse_retained",
        "oof_rmse_effective_exposure",
        "rmse_delta_vs_effective_exposure",
        "oof_spearman_retained",
        "oof_spearman_effective_exposure",
        "spearman_delta_vs_effective_exposure",
    ]
    lines = [
        "# Compact retained-state surrogate screen",
        "",
        "## Model",
        "",
        "$$z_i=e^{-\\lambda(1-w_i^{(1)})}e_i^{(0)}+\\eta e_i^{(1)}," "\\qquad q_i=e_i^{(0)}+e_i^{(1)}$$",
        "",
        "$$\\hat L=b-\\sum_i a_i\\left(1-e^{-(\\rho z_i)^p}\\right)" "+c\\sum_i[q_i-1]_+^2.$$",
        "",
        "The full model has four shared nonlinear parameters and no bucket-specific nonlinear parameters. "
        "All bucket-specific coefficients are nonnegative benefit amplitudes; the optional replay term has one "
        "shared nonnegative coefficient. No ensemble, calibration layer, or parameter floor is used.",
        "",
        "The ablations ask one mechanical question at a time: whether phase-0 learning is retained without "
        "late revisits, whether learning saturates, and whether replay beyond one simulated epoch causes shared "
        "quadratic harm.",
        "",
        "Two equally sized alternatives were also screened. Poisson refresh replaces the retained fraction by",
        "",
        "$$r_i=1-(1-e^{-\\lambda})e^{-e_i^{(1)}},$$",
        "",
        "and token-weighted replay replaces the bucket-balanced sum by "
        "$$\\sum_i (N_i/\\bar N)[q_i-1]_+^2$$. Neither improves consistently across targets. "
        "Independent phase heads and cumulative-plus-recency channels are reported as richer negative controls, "
        "not candidate models.",
        "",
        "## OOF metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Existing Observatory baselines",
        "",
        baselines.to_markdown(index=False),
        "",
        "## Selected retained candidate versus effective-exposure DSP",
        "",
        candidate_comparison.loc[:, comparison_columns].to_markdown(index=False),
        "",
        "## Fitted shared shapes",
        "",
        shape_summary.to_markdown(index=False),
        "",
        "## Untouched 300M policies",
        "",
        heldout_all.to_markdown(index=False),
        "",
        "## Largest untouched-policy residuals for the retained model",
        "",
        largest_residuals.to_markdown(index=False),
        "",
        "## Reproduce",
        "",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "benchmark_retained_weibull_replay_20260713.py",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default=",".join(DATASET_NAMES))
    parser.add_argument("--models", default=",".join(config.name for config in CONFIGS))
    parser.add_argument("--l2-values", default="0.1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--confirmatory-models", default=",".join(CONFIRMATORY_MODELS))
    parser.add_argument("--confirmatory-l2-values", default="0.1,1.0")
    parser.add_argument("--confirmatory-seeds", default="0,1,2")
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = parse_string_list(args.datasets)
    model_names = set(parse_string_list(args.models))
    configs = [config for config in CONFIGS if config.name in model_names]
    if model_names != {config.name for config in configs}:
        raise ValueError(f"Unknown models: {sorted(model_names - {config.name for config in configs})}")
    l2_values = parse_float_list(args.l2_values)
    seeds = parse_int_list(args.seeds)
    confirmatory_model_names = set(parse_string_list(args.confirmatory_models))
    if not confirmatory_model_names.issubset({config.name for config in configs}):
        missing = confirmatory_model_names - {config.name for config in configs}
        raise ValueError(f"Confirmatory models are not in --models: {sorted(missing)}")
    confirmatory_l2_values = parse_float_list(args.confirmatory_l2_values)
    confirmatory_seeds = parse_int_list(args.confirmatory_seeds)

    metric_rows = []
    parameter_rows: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, str, float], FittedModel] = {}
    for dataset_name in dataset_names:
        dataset = dataset_by_name(dataset_name)
        for config in configs:
            cases = {(l2, seed) for l2 in l2_values for seed in seeds}
            if config.name in confirmatory_model_names:
                cases.update((l2, seed) for l2 in confirmatory_l2_values for seed in confirmatory_seeds)
            for l2, seed in sorted(cases):
                prediction, parameters = oof_prediction(
                    dataset,
                    config,
                    l2,
                    seed,
                    args.output_dir,
                    maxiter=args.maxiter,
                    top_k=args.top_k,
                    force=args.force,
                )
                parameter_rows.extend(parameters)
                metric_rows.append(asdict(metric_row(dataset, config, l2, prediction, seed)))
            for l2 in sorted({case[0] for case in cases}):
                fitted_models[(dataset.name, config.name, l2)] = fit_model(
                    dataset,
                    np.arange(dataset.n),
                    config,
                    l2,
                    maxiter=args.maxiter,
                    top_k=args.top_k,
                )

    metrics = pd.DataFrame(metric_rows)
    parameters = pd.DataFrame(parameter_rows)
    heldout, heldout_predictions = evaluate_untouched_policies(fitted_models)
    metrics.to_csv(args.output_dir / "clean_response_metrics.csv", index=False)
    parameters.to_csv(args.output_dir / "clean_response_parameters.csv", index=False)
    heldout.to_csv(args.output_dir / "untouched_policy_metrics.csv", index=False)
    heldout_predictions.to_csv(args.output_dir / "untouched_policy_predictions.csv", index=False)
    plot_metrics(metrics, args.output_dir)
    write_report(metrics, parameters, heldout, heldout_predictions, args.output_dir)
    print(metrics.sort_values(["dataset", "oof_rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
