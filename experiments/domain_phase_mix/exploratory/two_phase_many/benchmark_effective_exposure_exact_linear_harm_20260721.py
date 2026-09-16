# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Benchmark exact linear-past-threshold harm in effective-exposure DSP.

The useful-learning state remains effective-exposure DSP:

    z_i = e_i^(0) + gamma e_i^(1)
    B_i = 1 - exp(-rho_i z_i)

The experiment crosses two repetition-harm states with two response laws:

    state: effective z_i, or physical E_i = e_i^(0) + e_i^(1)
    law:   softplus(log(1 + state_i) - tau_i)^2, or
           max(state_i - T_i, 0)

Both tau_i/T_i and the nonnegative harm amplitude are learned per bucket. The
hard hinge is the exact collaborator-proposed law, not a softened asymptotic
approximation. This script intentionally uses fit panels only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as panels,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/effective_exposure_exact_linear_harm_20260721"
DEFAULT_DATASETS = (
    panels.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    panels.DatasetId.THREE_HUNDRED_M_TABLE9,
    panels.DatasetId.DELPHI_3E18_UNCHEATABLE,
    panels.DatasetId.DELPHI_3E18_TABLE9,
    panels.DatasetId.PRODUCTION_UNCHEATABLE,
)
CV_SEED = 7211
N_SPLITS = 5
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
OPTIMISM_THRESHOLD = 0.05
PROFILE_OPTIMISM_WEIGHT = 0.5
HINGE_LOG_THRESHOLD_BOUNDS = (math.log(1e-3), math.log(512.0))
CURVED_THRESHOLD_BOUNDS = (-2.0, 8.0)
RHO_BOUNDS = (math.log(1e-4), math.log(2.0))
GAMMA_BOUNDS = (math.log(1e-4), math.log(100.0))
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class HarmState(StrEnum):
    EFFECTIVE = "effective"
    PHYSICAL = "physical"


class HarmLaw(StrEnum):
    CURVED = "curved_softplus_squared"
    EXACT_LINEAR = "exact_linear_hinge"


@dataclass(frozen=True)
class Variant:
    name: str
    harm_state: HarmState
    harm_law: HarmLaw


@dataclass(frozen=True)
class Packet:
    frame: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domains: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def m(self) -> int:
        return len(self.domains)


@dataclass(frozen=True)
class Model:
    variant: Variant
    rho: np.ndarray
    threshold_coordinate: np.ndarray
    gamma: float
    intercept: float
    benefit: np.ndarray
    harm: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domains: tuple[str, ...]

    @property
    def threshold_epochs(self) -> np.ndarray:
        if self.variant.harm_law is HarmLaw.EXACT_LINEAR:
            return np.exp(self.threshold_coordinate)
        return np.maximum(np.expm1(self.threshold_coordinate), 0.0)

    @property
    def parameter_count(self) -> int:
        return 4 * len(self.domains) + 2


@dataclass(frozen=True)
class FitResult:
    model: Model
    tuning: tuple[dict[str, Any], ...]


VARIANTS = (
    Variant("curved_effective", HarmState.EFFECTIVE, HarmLaw.CURVED),
    Variant("curved_physical", HarmState.PHYSICAL, HarmLaw.CURVED),
    Variant("linear_effective", HarmState.EFFECTIVE, HarmLaw.EXACT_LINEAR),
    Variant("linear_physical", HarmState.PHYSICAL, HarmLaw.EXACT_LINEAR),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=",".join(dataset.value for dataset in DEFAULT_DATASETS),
        help="Comma-separated fit-panel dataset IDs.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(variant.name for variant in VARIANTS),
        help="Comma-separated variant names.",
    )
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    return parser.parse_args()


def preregistration(
    dataset_ids: tuple[panels.DatasetId, ...],
    variants: tuple[Variant, ...],
) -> dict[str, Any]:
    return {
        "frozen_at": datetime.now(UTC).isoformat(),
        "data_boundary": "fit panels only; append-only 3e18 development heldouts are not loaded",
        "datasets": [dataset.value for dataset in dataset_ids],
        "variants": [variant.name for variant in variants],
        "useful_state": "z_i = e0_i + gamma * e1_i",
        "useful_response": "-a_i * (1 - exp(-rho_i * z_i)); a_i >= 0",
        "harm_states": {
            HarmState.EFFECTIVE.value: "z_i = e0_i + gamma * e1_i",
            HarmState.PHYSICAL.value: "E_i = e0_i + e1_i",
        },
        "harm_laws": {
            HarmLaw.CURVED.value: "+p_i * softplus(log(1 + state_i) - tau_i)^2",
            HarmLaw.EXACT_LINEAR.value: "+p_i * max(state_i - T_i, 0)",
        },
        "identification": "rho_i, threshold_i, gamma are profiled nonlinearly; intercept, a_i, p_i use NNLS",
        "linear_regularization": {
            "production": 1e-6,
            "other_panels": 0.01,
            "reason": "match the Observatory effective-exposure DSP convention without target retuning",
        },
        "promotion_gate": {
            "oof_rmse": "no panel more than 5% worse than curved_effective",
            "regret_at_1": "no panel more than 0.002 BPB worse than curved_effective",
            "material_gain": "at least 1% OOF RMSE improvement on two independent panels",
            "identifiability": "nonzero harm on >= 25% of buckets and fold threshold IQR finite",
        },
    }


def packet_from_dataset(dataset: family_grp.Dataset) -> Packet:
    return Packet(
        frame=dataset.frame.reset_index(drop=True),
        y=np.asarray(dataset.target, dtype=float),
        weights=np.asarray(dataset.weights, dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domains=tuple(dataset.domains),
    )


def subset(packet: Packet, indices: np.ndarray) -> Packet:
    return Packet(
        frame=packet.frame.iloc[indices].reset_index(drop=True),
        y=packet.y[indices],
        weights=packet.weights[indices],
        c0=packet.c0,
        c1=packet.c1,
        domains=packet.domains,
    )


def exposure_states(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    e0 = weights[:, 0, :] * c0[None, :]
    e1 = weights[:, 1, :] * c1[None, :]
    return e0 + gamma * e1, e0 + e1


def feature_matrices(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    variant: Variant,
    rho: np.ndarray,
    threshold_coordinate: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    effective, physical = exposure_states(weights, c0, c1, gamma)
    signal = 1.0 - np.exp(-effective * rho[None, :])
    harm_state = effective if variant.harm_state is HarmState.EFFECTIVE else physical
    if variant.harm_law is HarmLaw.EXACT_LINEAR:
        threshold = np.exp(threshold_coordinate)[None, :]
        harm = np.maximum(harm_state - threshold, 0.0)
    else:
        tau = threshold_coordinate[None, :]
        harm = dsp.softplus(np.log1p(harm_state) - tau) ** 2
    return signal, harm


def fit_head(
    packet: Packet,
    variant: Variant,
    rho: np.ndarray,
    threshold_coordinate: np.ndarray,
    gamma: float,
    l2: float,
) -> Model:
    signal, harm = feature_matrices(
        packet.weights,
        packet.c0,
        packet.c1,
        variant,
        rho,
        threshold_coordinate,
        gamma,
    )
    design = np.hstack([-signal, harm])
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(packet.y.mean())
    centered_design = design - design_mean
    centered_target = packet.y - target_mean
    if l2 > 0.0:
        num_features = centered_design.shape[1]
        centered_design = np.vstack([centered_design, math.sqrt(l2) * np.eye(num_features)])
        centered_target = np.concatenate([centered_target, np.zeros(num_features, dtype=float)])
    coefficients, _residual = nnls(centered_design, centered_target, maxiter=40 * centered_design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
    return Model(
        variant=variant,
        rho=np.asarray(rho, dtype=float),
        threshold_coordinate=np.asarray(threshold_coordinate, dtype=float),
        gamma=float(gamma),
        intercept=intercept,
        benefit=np.asarray(coefficients[: packet.m], dtype=float),
        harm=np.asarray(coefficients[packet.m :], dtype=float),
        c0=packet.c0,
        c1=packet.c1,
        domains=packet.domains,
    )


def predict(model: Model, weights: np.ndarray) -> np.ndarray:
    signal, harm = feature_matrices(
        weights,
        model.c0,
        model.c1,
        model.variant,
        model.rho,
        model.threshold_coordinate,
        model.gamma,
    )
    return np.asarray(model.intercept - signal @ model.benefit + harm @ model.harm, dtype=float)


def unpack_theta(theta: np.ndarray, packet: Packet) -> tuple[np.ndarray, np.ndarray, float]:
    rho = np.exp(theta[: packet.m])
    threshold = theta[packet.m : 2 * packet.m]
    gamma = float(np.exp(theta[-1]))
    return rho, threshold, gamma


def theta_bounds(variant: Variant, packet: Packet) -> list[tuple[float, float]]:
    threshold_bounds = (
        HINGE_LOG_THRESHOLD_BOUNDS if variant.harm_law is HarmLaw.EXACT_LINEAR else CURVED_THRESHOLD_BOUNDS
    )
    return [RHO_BOUNDS] * packet.m + [threshold_bounds] * packet.m + [GAMMA_BOUNDS]


def start_bank(packet: Packet, variant: Variant) -> tuple[np.ndarray, ...]:
    starts: list[np.ndarray] = []
    physical = exposure_states(packet.weights, packet.c0, packet.c1, 1.0)[1]
    positive_physical = np.where(physical > 1e-8, physical, np.nan)
    rng = np.random.default_rng(CV_SEED)
    recipes = (
        (0.5, 0.50, 0.5),
        (1.0, 0.70, 0.75),
        (4.0, 0.85, 1.0),
        (16.0, 0.90, 1.5),
        (32.0, 0.95, 2.0),
    )
    for gamma, quantile, rho_scale in recipes:
        effective, _physical = exposure_states(packet.weights, packet.c0, packet.c1, gamma)
        positive_effective = np.where(effective > 1e-8, effective, np.nan)
        median = np.nanmedian(positive_effective, axis=0)
        global_median = float(np.nanmedian(positive_effective))
        median = np.where(np.isfinite(median), median, global_median)
        rho = np.clip(rho_scale / np.maximum(median, 1e-3), np.exp(RHO_BOUNDS[0]), np.exp(RHO_BOUNDS[1]))
        state = positive_effective if variant.harm_state is HarmState.EFFECTIVE else positive_physical
        onset = np.nanquantile(state, quantile, axis=0)
        global_onset = float(np.nanquantile(state, quantile))
        onset = np.where(np.isfinite(onset), onset, global_onset)
        onset = np.maximum(onset, np.exp(HINGE_LOG_THRESHOLD_BOUNDS[0]))
        threshold_coordinate = np.log(onset) if variant.harm_law is HarmLaw.EXACT_LINEAR else np.log1p(onset)
        starts.append(np.concatenate([np.log(rho), threshold_coordinate, np.asarray([math.log(gamma)])]))

    for _ in range(3):
        base = starts[int(rng.integers(0, len(starts)))].copy()
        base[: packet.m] += rng.normal(scale=0.5, size=packet.m)
        base[packet.m : 2 * packet.m] += rng.normal(scale=0.5, size=packet.m)
        base[-1] += float(rng.normal(scale=0.7))
        starts.append(
            np.asarray(
                [np.clip(value, *bound) for value, bound in zip(base, theta_bounds(variant, packet), strict=True)]
            )
        )
    return tuple(starts)


def profile_objective(packet: Packet, variant: Variant, theta: np.ndarray, l2: float) -> float:
    rho, threshold, gamma = unpack_theta(theta, packet)
    model = fit_head(packet, variant, rho, threshold, gamma, l2)
    prediction = predict(model, packet.weights)
    residual = prediction - packet.y
    rmse = float(np.sqrt(np.mean(residual**2)))
    tail_count = min(packet.n, max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * packet.n)))
    tail = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail] - prediction[tail], 0.0)))
    return rmse + PROFILE_OPTIMISM_WEIGHT * optimism


def fit_model(
    packet: Packet,
    variant: Variant,
    l2: float,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> FitResult:
    starts = start_bank(packet, variant)
    scored = [profile_objective(packet, variant, start, l2) for start in starts]
    order = np.argsort(scored)[:coarse_top_k]
    records: list[dict[str, Any]] = [
        {"stage": "coarse", "start": index, "objective": value} for index, value in enumerate(scored)
    ]
    best_start = int(np.argmin(scored))
    best_theta: np.ndarray | None = np.asarray(starts[best_start], dtype=float)
    best_objective = float(scored[best_start])
    if maxiter <= 0:
        rho, threshold, gamma = unpack_theta(best_theta, packet)
        return FitResult(fit_head(packet, variant, rho, threshold, gamma, l2), tuple(records))
    for rank, start_index in enumerate(order):
        result = minimize(
            lambda theta: profile_objective(packet, variant, np.asarray(theta, dtype=float), l2),
            starts[int(start_index)],
            method="L-BFGS-B",
            bounds=theta_bounds(variant, packet),
            options={"maxiter": maxiter, "ftol": 1e-8, "maxls": 30},
        )
        records.append(
            {
                "stage": "refine",
                "rank": rank,
                "start": int(start_index),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)
    if best_theta is None:
        raise RuntimeError(f"No fit result for {variant.name}")
    rho, threshold, gamma = unpack_theta(best_theta, packet)
    return FitResult(fit_head(packet, variant, rho, threshold, gamma, l2), tuple(records))


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = prediction - observed
    tail_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    tail = np.argsort(prediction)[:tail_count]
    tail_error = observed[tail] - prediction[tail]
    order = np.argsort(prediction)
    pred_variance = float(np.var(prediction))
    calibration_slope = (
        float(np.cov(prediction, observed, ddof=0)[0, 1] / pred_variance) if pred_variance > 0.0 else float("nan")
    )
    out: dict[str, float | int] = {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "calibration_slope": calibration_slope,
        "regret_at_1": float(observed[order[0]] - np.min(observed)),
        "lower_tail_optimism": float(np.mean(np.maximum(tail_error, 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(tail_error**2))),
        "optimism_gt_0p05": int(np.sum(observed - prediction > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(observed - prediction)),
    }
    for k in (3, 5):
        out[f"regret_at_{k}"] = float(np.min(observed[order[:k]]) - np.min(observed))
    return out


def nested_oof(
    packet: Packet,
    dataset_id: panels.DatasetId,
    variant: Variant,
    l2: float,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[np.ndarray, tuple[Model, ...]]:
    indices = np.arange(packet.n)
    raw_dataset = panels.load_dataset(dataset_id)
    splits = panels.split_indices(raw_dataset, dataset_id, indices, CV_SEED)
    prediction = np.full(packet.n, np.nan, dtype=float)
    models: list[Model] = []
    for fold_id, (train, test) in enumerate(splits):
        print(f"    {variant.name}: fold {fold_id + 1}/{len(splits)}", flush=True)
        fit = fit_model(
            subset(packet, train),
            variant,
            l2,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        prediction[test] = predict(fit.model, packet.weights[test])
        models.append(fit.model)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {dataset_id.value}/{variant.name}")
    return prediction, tuple(models)


def parameter_summary(dataset_id: panels.DatasetId, variant: Variant, models: tuple[Model, ...]) -> dict[str, Any]:
    gamma = np.asarray([model.gamma for model in models], dtype=float)
    active_fraction = np.asarray([np.mean(model.harm > 1e-8) for model in models], dtype=float)
    thresholds = np.vstack([model.threshold_epochs for model in models])
    threshold_iqr = np.nanpercentile(thresholds, 75, axis=0) - np.nanpercentile(thresholds, 25, axis=0)
    threshold_median = np.nanmedian(thresholds, axis=0)
    relative_iqr = threshold_iqr / np.maximum(threshold_median, 1e-6)
    return {
        "dataset": dataset_id.value,
        "variant": variant.name,
        "gamma_mean": float(np.mean(gamma)),
        "gamma_std": float(np.std(gamma, ddof=1)),
        "active_harm_bucket_fraction_mean": float(np.mean(active_fraction)),
        "threshold_epoch_median": float(np.nanmedian(thresholds)),
        "threshold_epoch_q05": float(np.nanpercentile(thresholds, 5)),
        "threshold_epoch_q95": float(np.nanpercentile(thresholds, 95)),
        "threshold_relative_iqr_median": float(np.nanmedian(relative_iqr)),
        "threshold_relative_iqr_q90": float(np.nanpercentile(relative_iqr, 90)),
    }


def comparison_plot(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    datasets = list(dict.fromkeys(metrics_frame["dataset"].tolist()))
    figure = make_subplots(
        rows=2,
        cols=len(datasets),
        subplot_titles=datasets,
        vertical_spacing=0.18,
    )
    colors = {
        "curved_effective": "#17384c",
        "curved_physical": "#4f8a78",
        "linear_effective": "#e46f34",
        "linear_physical": "#efb431",
    }
    for column, dataset in enumerate(datasets, start=1):
        selected = metrics_frame.loc[metrics_frame["dataset"].eq(dataset)]
        figure.add_trace(
            go.Bar(
                x=selected["variant"],
                y=selected["rmse"],
                marker_color=[colors[name] for name in selected["variant"]],
                showlegend=False,
                hovertemplate="%{x}<br>OOF RMSE %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Bar(
                x=selected["variant"],
                y=selected["regret_at_1"],
                marker_color=[colors[name] for name in selected["variant"]],
                showlegend=False,
                hovertemplate="%{x}<br>Regret@1 %{y:.6f}<extra></extra>",
            ),
            row=2,
            col=column,
        )
    figure.update_yaxes(title_text="Nested OOF RMSE", row=1, col=1)
    figure.update_yaxes(title_text="Nested OOF Regret@1", row=2, col=1)
    figure.update_layout(
        title="Effective-exposure DSP: exact per-bucket linear-threshold harm",
        template="plotly_white",
        width=max(1200, 360 * len(datasets)),
        height=760,
        margin={"l": 80, "r": 30, "t": 100, "b": 130},
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def report(
    metrics_frame: pd.DataFrame,
    parameter_frame: pd.DataFrame,
    gate_frame: pd.DataFrame,
) -> str:
    columns = [
        "dataset",
        "variant",
        "rmse",
        "spearman",
        "regret_at_1",
        "lower_tail_optimism",
        "calibration_slope",
        "optimism_gt_0p05",
    ]
    parameter_columns = [
        "dataset",
        "variant",
        "gamma_mean",
        "gamma_std",
        "active_harm_bucket_fraction_mean",
        "threshold_epoch_median",
        "threshold_relative_iqr_median",
    ]
    return "\n".join(
        [
            "# Exact linear-threshold harm in effective-exposure DSP",
            "",
            "This is a fit-panel-only nested-CV ablation. No append-only Delphi heldout outcomes were loaded.",
            "",
            "## Equations",
            "",
            r"Useful state: $z_i=e_i^{(0)}+\gamma e_i^{(1)}$ and benefit " r"$-a_i(1-\exp[-\rho_i z_i])$.",
            "",
            r"Exact collaborator harm: $+p_i[E_i-\tau_i]_+$ with "
            r"$E_i=e_i^{(0)}+e_i^{(1)}$, $p_i\geq0$, and one learned $\tau_i>0$ per bucket.",
            "",
            (
                "The crossed controls determine whether any difference is due to the exact linear law or to "
                "using literal physical epochs for repetition harm."
            ),
            "",
            "## Nested OOF metrics",
            "",
            metrics_frame[columns].to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Parameter stability",
            "",
            parameter_frame[parameter_columns].to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Frozen promotion gate",
            "",
            gate_frame.to_markdown(index=False, floatfmt=".6f"),
            "",
            (
                "A hard-hinge threshold is not identified when its fitted harm amplitude is zero. The "
                "active-bucket fraction and fold threshold IQR therefore matter as much as nominal fit error."
            ),
        ]
    )


def linear_reg(dataset_id: panels.DatasetId) -> float:
    return 1e-6 if dataset_id is panels.DatasetId.PRODUCTION_UNCHEATABLE else 0.01


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_ids = tuple(panels.DatasetId(value.strip()) for value in args.datasets.split(",") if value.strip())
    variant_by_name = {variant.name: variant for variant in VARIANTS}
    variants = tuple(variant_by_name[value.strip()] for value in args.variants.split(",") if value.strip())
    (output_dir / "preregistration.json").write_text(
        json.dumps(preregistration(dataset_ids, variants), indent=2, sort_keys=True) + "\n"
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    full_parameter_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        print(f"Loading {dataset_id.value}", flush=True)
        raw_dataset = panels.load_dataset(dataset_id)
        packet = packet_from_dataset(raw_dataset)
        for variant in variants:
            print(f"  Fitting {variant.name}", flush=True)
            l2 = linear_reg(dataset_id)
            oof, fold_models = nested_oof(
                packet,
                dataset_id,
                variant,
                l2,
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
            )
            full_fit = fit_model(
                packet,
                variant,
                l2,
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
            )
            row = {
                "dataset": dataset_id.value,
                "variant": variant.name,
                "harm_state": variant.harm_state.value,
                "harm_law": variant.harm_law.value,
                "l2": l2,
                "parameter_count": full_fit.model.parameter_count,
                **metrics(packet.y, oof),
            }
            metric_rows.append(row)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.name,
                        "row": np.arange(packet.n),
                        "observed": packet.y,
                        "prediction": oof,
                        "residual": oof - packet.y,
                    }
                )
            )
            parameter_rows.append(parameter_summary(dataset_id, variant, fold_models))
            for domain, rho, threshold, benefit, harm in zip(
                packet.domains,
                full_fit.model.rho,
                full_fit.model.threshold_epochs,
                full_fit.model.benefit,
                full_fit.model.harm,
                strict=True,
            ):
                full_parameter_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.name,
                        "domain": domain,
                        "rho": rho,
                        "threshold_epochs": threshold,
                        "benefit_amplitude": benefit,
                        "harm_slope": harm,
                        "gamma": full_fit.model.gamma,
                    }
                )
            for tuning in full_fit.tuning:
                tuning_rows.append({"dataset": dataset_id.value, "variant": variant.name, **tuning})

    metrics_frame = pd.DataFrame.from_records(metric_rows)
    parameter_frame = pd.DataFrame.from_records(parameter_rows)
    baselines = metrics_frame.loc[metrics_frame["variant"].eq("curved_effective")].set_index("dataset")
    gate_rows: list[dict[str, Any]] = []
    for row in metrics_frame.itertuples(index=False):
        baseline = baselines.loc[row.dataset]
        rmse_ratio = float(row.rmse / baseline["rmse"])
        regret_delta = float(row.regret_at_1 - baseline["regret_at_1"])
        gate_rows.append(
            {
                "dataset": row.dataset,
                "variant": row.variant,
                "rmse_ratio_vs_curved_effective": rmse_ratio,
                "regret_delta_vs_curved_effective": regret_delta,
                "rmse_gate": rmse_ratio <= 1.05,
                "regret_gate": regret_delta <= 0.002,
                "material_rmse_gain": rmse_ratio <= 0.99,
            }
        )
    gate_frame = pd.DataFrame.from_records(gate_rows)

    metrics_frame.to_csv(output_dir / "nested_oof_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(output_dir / "nested_oof_predictions.csv", index=False)
    parameter_frame.to_csv(output_dir / "fold_parameter_stability.csv", index=False)
    pd.DataFrame.from_records(full_parameter_rows).to_csv(output_dir / "full_fit_parameters.csv", index=False)
    pd.DataFrame.from_records(tuning_rows).to_csv(output_dir / "full_fit_tuning.csv", index=False)
    gate_frame.to_csv(output_dir / "promotion_gate.csv", index=False)
    comparison_plot(metrics_frame, output_dir / "fit_panel_comparison.html")
    (output_dir / "report.md").write_text(report(metrics_frame, parameter_frame, gate_frame) + "\n")
    print(metrics_frame.to_string(index=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
