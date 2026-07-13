# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Calibrate cumulative-recency response timescales across swarms.

The cumulative-recency model derives one benefit scale and one overexposure
threshold per bucket from observed exposure statistics, then fits only a
global shift. That is parsimonious, but it assumes the empirical prior has the
correct dispersion across buckets. This benchmark learns either a positive
affine calibration of that dispersion or one mean-zero, hierarchically shrunk
sample-efficiency offset per bucket. The latter is shared across cumulative
and recency channels: a bucket that learns faster also reaches its repetition
threshold earlier. Every family preserves the existing model at zero offset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_cumulative_recency_crossswarm as cross,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    benchmark_cumulative_recency_starcoder as recency,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "calibrated_cumulative_recency_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
SHIFT_BOUND = 3.0
LOG_SPREAD_BOUND = 1.5
LOG_POWER_BOUND = 1.0
HEAD_L2_BY_DATASET = {
    centered.STARCODER_NAME: 1e-5,
    "300m_uncheatable": 0.01,
    "300m_table9": 0.1,
    "production_uncheatable": 0.1,
}


class CalibrationKind(StrEnum):
    """Nested response-scale calibration families."""

    SHIFT_ONLY = "shift_only"
    SHARED_SPREAD = "shared_spread"
    CHANNEL_SPREAD = "channel_spread"
    DOMAIN_EFFICIENCY = "domain_efficiency"
    SHARED_POWER = "shared_power"
    CHANNEL_POWER = "channel_power"
    SHARED_POWER_TV = "shared_power_tv"
    SHARED_POWER_HHI = "shared_power_hhi"
    SHARED_POWER_GEOMETRY = "shared_power_geometry"


@dataclass(frozen=True)
class ModelConfig:
    """One response-scale calibration configuration."""

    kind: CalibrationKind
    spread_l2: float

    @property
    def name(self) -> str:
        return f"{self.kind.value}_spread_l2_{self.spread_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Fitted cumulative-recency model with calibrated response dispersion."""

    config: ModelConfig
    cumulative_base: recency.ChannelBase
    recency_base: recency.ChannelBase
    shape_parameters: np.ndarray
    intercept: float
    coef: np.ndarray
    head_l2: float
    c0: np.ndarray
    c1: np.ndarray

    @property
    def num_domains(self) -> int:
        return len(self.c0)

    @property
    def alpha0(self) -> float:
        ratio = float(np.median(self.c0 / self.c1))
        return ratio / (1.0 + ratio)

    @property
    def alpha1(self) -> float:
        return 1.0 - self.alpha0

    @property
    def parameter_count(self) -> int:
        extra = {
            CalibrationKind.SHIFT_ONLY: 0,
            CalibrationKind.SHARED_SPREAD: 2,
            CalibrationKind.CHANNEL_SPREAD: 4,
            CalibrationKind.DOMAIN_EFFICIENCY: self.num_domains,
            CalibrationKind.SHARED_POWER: 2,
            CalibrationKind.CHANNEL_POWER: 4,
            CalibrationKind.SHARED_POWER_TV: 3,
            CalibrationKind.SHARED_POWER_HHI: 3,
            CalibrationKind.SHARED_POWER_GEOMETRY: 4,
        }[self.config.kind]
        return 4 * self.num_domains + 5 + extra

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(
            weights,
            self.c0,
            self.c1,
            self.cumulative_base,
            self.recency_base,
            self.config.kind,
            self.shape_parameters,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def calibrated_shape(
    base: recency.ChannelBase,
    rho_shift: float,
    tau_shift: float,
    rho_log_spread: float,
    tau_log_spread: float,
    efficiency_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the location and dispersion of one empirical shape prior."""
    log_rho = np.log(np.clip(base.rho, 1e-8, None))
    log_rho_center = float(np.median(log_rho))
    tau_center = float(np.median(base.tau))
    rho = np.exp(log_rho_center + rho_shift + np.exp(rho_log_spread) * (log_rho - log_rho_center) + efficiency_offsets)
    tau = tau_center + tau_shift + np.exp(tau_log_spread) * (base.tau - tau_center) - efficiency_offsets
    return np.clip(rho, 1e-4, 2.0), np.clip(tau, -2.0, 8.0)


def channel_parameters(
    kind: CalibrationKind,
    shape_parameters: np.ndarray,
    num_domains: int,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    np.ndarray,
]:
    """Decode cumulative and recency channel parameters."""
    if len(shape_parameters) < 4:
        raise ValueError("Shape parameter vector must include four shifts")
    cumulative_shift = (float(shape_parameters[0]), float(shape_parameters[1]))
    recency_shift = (float(shape_parameters[2]), float(shape_parameters[3]))
    if kind in (
        CalibrationKind.SHIFT_ONLY,
        CalibrationKind.DOMAIN_EFFICIENCY,
        CalibrationKind.SHARED_POWER,
        CalibrationKind.CHANNEL_POWER,
        CalibrationKind.SHARED_POWER_TV,
        CalibrationKind.SHARED_POWER_HHI,
        CalibrationKind.SHARED_POWER_GEOMETRY,
    ):
        cumulative_spread = (0.0, 0.0)
        recency_spread = (0.0, 0.0)
    elif kind is CalibrationKind.SHARED_SPREAD:
        cumulative_spread = (float(shape_parameters[4]), float(shape_parameters[5]))
        recency_spread = cumulative_spread
    elif kind is CalibrationKind.CHANNEL_SPREAD:
        cumulative_spread = (float(shape_parameters[4]), float(shape_parameters[5]))
        recency_spread = (float(shape_parameters[6]), float(shape_parameters[7]))
    else:
        raise ValueError(f"Unknown calibration kind {kind!r}")
    if kind is CalibrationKind.DOMAIN_EFFICIENCY:
        offsets = np.asarray(shape_parameters[4:], dtype=float)
        if len(offsets) != num_domains:
            raise ValueError(f"Expected {num_domains} domain offsets, got {len(offsets)}")
        offsets = offsets - float(np.mean(offsets))
    else:
        offsets = np.zeros(num_domains, dtype=float)
    return (
        (*cumulative_shift, *cumulative_spread),
        (*recency_shift, *recency_spread),
        offsets,
    )


def spread_parameters(kind: CalibrationKind, shape_parameters: np.ndarray) -> np.ndarray:
    """Return only parameters shrunk toward the nested slope-one model."""
    if kind is CalibrationKind.SHIFT_ONLY:
        return np.empty(0, dtype=float)
    return np.asarray(shape_parameters[4:], dtype=float)


def response_powers(
    kind: CalibrationKind,
    shape_parameters: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Decode benefit and penalty powers for both exposure channels."""
    if kind in (
        CalibrationKind.SHARED_POWER,
        CalibrationKind.SHARED_POWER_TV,
        CalibrationKind.SHARED_POWER_HHI,
        CalibrationKind.SHARED_POWER_GEOMETRY,
    ):
        shared = (float(np.exp(shape_parameters[4])), float(2.0 * np.exp(shape_parameters[5])))
        return shared, shared
    if kind is CalibrationKind.CHANNEL_POWER:
        return (
            (float(np.exp(shape_parameters[4])), float(2.0 * np.exp(shape_parameters[5]))),
            (float(np.exp(shape_parameters[6])), float(2.0 * np.exp(shape_parameters[7]))),
        )
    return (1.0, 2.0), (1.0, 2.0)


def channel_features(
    exposure: np.ndarray,
    rho: np.ndarray,
    tau: np.ndarray,
    benefit_power: float,
    penalty_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return monotone saturating benefit and powered overexposure harm."""
    scaled_exposure = np.maximum(rho[None, :] * exposure, 0.0)
    benefit = -np.expm1(-(scaled_exposure**benefit_power))
    penalty = recency.softplus(np.log1p(exposure) - tau[None, :]) ** penalty_power
    return benefit, penalty


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    cumulative_base: recency.ChannelBase,
    recency_base: recency.ChannelBase,
    kind: CalibrationKind,
    shape_parameters: np.ndarray,
) -> np.ndarray:
    """Build cumulative and recency response features."""
    cumulative, late = recency.exposures(weights, c0, c1)
    cumulative_parameters, recency_parameters, efficiency_offsets = channel_parameters(
        kind,
        shape_parameters,
        len(c0),
    )
    cumulative_rho, cumulative_tau = calibrated_shape(
        cumulative_base,
        *cumulative_parameters,
        efficiency_offsets,
    )
    late_rho, late_tau = calibrated_shape(
        recency_base,
        *recency_parameters,
        efficiency_offsets,
    )
    cumulative_powers, recency_powers = response_powers(kind, shape_parameters)
    cumulative_benefit, cumulative_penalty = channel_features(
        cumulative,
        cumulative_rho,
        cumulative_tau,
        *cumulative_powers,
    )
    late_benefit, late_penalty = channel_features(
        late,
        late_rho,
        late_tau,
        *recency_powers,
    )
    design_parts = [-cumulative_benefit, cumulative_penalty, -late_benefit, late_penalty]
    if kind in (CalibrationKind.SHARED_POWER_TV, CalibrationKind.SHARED_POWER_GEOMETRY):
        phase_tv = 0.5 * np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1)
        design_parts.append(phase_tv[:, None])
    if kind in (CalibrationKind.SHARED_POWER_HHI, CalibrationKind.SHARED_POWER_GEOMETRY):
        alpha0 = float(np.median(c0 / (c0 + c1)))
        aggregate = alpha0 * weights[:, 0, :] + (1.0 - alpha0) * weights[:, 1, :]
        aggregate_hhi = np.sum(aggregate**2, axis=1)
        design_parts.append(aggregate_hhi[:, None])
    return np.hstack(design_parts)


def initial_shape_parameters(kind: CalibrationKind, num_domains: int) -> list[np.ndarray]:
    """Build deterministic multistarts around the nested model."""
    extra = {
        CalibrationKind.SHIFT_ONLY: 0,
        CalibrationKind.SHARED_SPREAD: 2,
        CalibrationKind.CHANNEL_SPREAD: 4,
        CalibrationKind.DOMAIN_EFFICIENCY: num_domains,
        CalibrationKind.SHARED_POWER: 2,
        CalibrationKind.CHANNEL_POWER: 4,
        CalibrationKind.SHARED_POWER_TV: 2,
        CalibrationKind.SHARED_POWER_HHI: 2,
        CalibrationKind.SHARED_POWER_GEOMETRY: 2,
    }[kind]
    starts = [np.concatenate([base, np.zeros(extra, dtype=float)]) for base in recency.shape_starts()]
    if kind in (
        CalibrationKind.SHARED_SPREAD,
        CalibrationKind.CHANNEL_SPREAD,
        CalibrationKind.SHARED_POWER,
        CalibrationKind.CHANNEL_POWER,
        CalibrationKind.SHARED_POWER_TV,
        CalibrationKind.SHARED_POWER_HHI,
        CalibrationKind.SHARED_POWER_GEOMETRY,
    ):
        for value in (-0.5, 0.5):
            start = np.zeros(4 + extra, dtype=float)
            start[4:] = value
            starts.append(start)
    return starts


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: ModelConfig,
    head_l2: float,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> FittedModel:
    """Fit calibrated nonlinear shapes by profiling a nonnegative ridge head."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    cumulative, late = recency.exposures(weights, dataset.c0, dataset.c1)
    cumulative_base = recency.channel_base(cumulative)
    recency_base = recency.channel_base(late)

    def objective(shape_parameters: np.ndarray) -> float:
        design = design_matrix(
            weights,
            dataset.c0,
            dataset.c1,
            cumulative_base,
            recency_base,
            config.kind,
            np.asarray(shape_parameters, dtype=float),
        )
        intercept, coef = cross.fit_head(design, targets, head_l2)
        residual = intercept + design @ coef - targets
        spread = spread_parameters(config.kind, np.asarray(shape_parameters, dtype=float))
        penalty = config.spread_l2 * float(spread @ spread) / len(targets)
        return float(np.mean(residual**2) + penalty)

    starts = initial_shape_parameters(config.kind, dataset.m)
    scored_starts = sorted(
        ((objective(start), start) for start in starts),
        key=lambda item: item[0],
    )
    extra_bound = (
        LOG_POWER_BOUND
        if config.kind
        in (
            CalibrationKind.SHARED_POWER,
            CalibrationKind.CHANNEL_POWER,
            CalibrationKind.SHARED_POWER_TV,
            CalibrationKind.SHARED_POWER_HHI,
            CalibrationKind.SHARED_POWER_GEOMETRY,
        )
        else LOG_SPREAD_BOUND
    )
    bounds = [(-SHIFT_BOUND, SHIFT_BOUND)] * 4 + [(-extra_bound, extra_bound)] * (len(scored_starts[0][1]) - 4)
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "maxls": 30},
        )
        for _score, start in scored_starts[:coarse_top_k]
    ]
    best = min(results, key=lambda result: float(result.fun))
    shape_parameters = np.asarray(best.x, dtype=float)
    design = design_matrix(
        weights,
        dataset.c0,
        dataset.c1,
        cumulative_base,
        recency_base,
        config.kind,
        shape_parameters,
    )
    intercept, coef = cross.fit_head(design, targets, head_l2)
    return FittedModel(
        config=config,
        cumulative_base=cumulative_base,
        recency_base=recency_base,
        shape_parameters=shape_parameters,
        intercept=intercept,
        coef=coef,
        head_l2=head_l2,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
    )


def benchmark(
    dataset: pooled.Dataset,
    configs: list[ModelConfig],
    seeds: list[int],
    n_splits: int,
    *,
    head_l2: float,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for every configuration."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            folds = cross.folds_for(dataset, seed, n_splits)
            prediction = np.zeros(dataset.n, dtype=float)
            for fold_id, (train_indices, test_indices) in enumerate(folds):
                print(
                    f"{dataset.name}/{config.name}: seed={seed} " f"fold={fold_id + 1}/{n_splits}",
                    flush=True,
                )
                model = fit_model(
                    dataset,
                    train_indices,
                    config,
                    head_l2,
                    maxiter=maxiter,
                    coarse_top_k=coarse_top_k,
                )
                prediction[test_indices] = model.predict(dataset.weights[test_indices])
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "shape_parameters": json.dumps(model.shape_parameters.tolist()),
                        "shrunk_shape_parameters": json.dumps(
                            spread_parameters(config.kind, model.shape_parameters).tolist()
                        ),
                        "coef_norm": float(np.linalg.norm(model.coef)),
                        "parameter_count": model.parameter_count,
                    }
                )
            metric = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            metric["nominal_param_count"] = model.parameter_count
            metric["head_l2"] = head_l2
            metric["spread_l2"] = config.spread_l2
            metric_rows.append(metric)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def full_fit_slice_metrics(
    dataset: pooled.Dataset,
    configs: list[ModelConfig],
    *,
    head_l2: float,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure the dense StarCoder phase-0-Nemotron slice after a full fit."""
    if dataset.name != centered.STARCODER_NAME:
        return pd.DataFrame()
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for config in configs:
        model = fit_model(
            dataset,
            np.arange(dataset.n),
            config,
            head_l2,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        prediction = model.predict(dataset.weights[mask])
        targets = dataset.y[mask]
        phase1 = dataset.frame.loc[mask, "phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "dataset": dataset.name,
                "model": config.name,
                "slice_rows": int(mask.sum()),
                "slice_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                "slice_spearman": float(spearmanr(targets, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def external_intervention_metrics(
    datasets: dict[str, pooled.Dataset],
    external: dict[str, pooled.Dataset],
    configs: list[ModelConfig],
    *,
    head_l2_override: float | None,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Fit full 300M panels and score untouched two-phase interventions."""
    rows: list[dict[str, Any]] = []
    for dataset_name, external_dataset in external.items():
        dataset = datasets[dataset_name]
        head_l2 = HEAD_L2_BY_DATASET[dataset_name] if head_l2_override is None else head_l2_override
        for config in configs:
            model = fit_model(
                dataset,
                np.arange(dataset.n),
                config,
                head_l2,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            row = joint.external_metrics(
                config.name,
                external_dataset.y,
                model.predict(external_dataset.weights),
            )
            row["dataset"] = dataset_name
            row["external_rows"] = external_dataset.n
            rows.append(row)
    return pd.DataFrame(rows)


def parse_configs(kinds: str, spread_l2_values: str) -> list[ModelConfig]:
    """Parse a compact Cartesian configuration grid."""
    parsed_kinds = [CalibrationKind(value.strip()) for value in kinds.split(",")]
    spread_values = pooled.parse_float_list(spread_l2_values)
    configs = []
    for kind in parsed_kinds:
        values = [0.0] if kind is CalibrationKind.SHIFT_ONLY else spread_values
        configs.extend(ModelConfig(kind=kind, spread_l2=value) for value in values)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=(f"{centered.STARCODER_NAME},300m_uncheatable," "300m_table9,production_uncheatable"),
    )
    parser.add_argument(
        "--kinds",
        default="shift_only,shared_spread,channel_spread",
    )
    parser.add_argument("--spread-l2-values", default="0,0.001")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--head-l2", type=float)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, external = centered.load_datasets()
    names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    configs = parse_configs(args.kinds, args.spread_l2_values)
    seeds = pooled.parse_int_list(args.seeds)
    metric_frames = []
    parameter_frames = []
    slice_frames = []
    for name in names:
        dataset = datasets[name]
        head_l2 = HEAD_L2_BY_DATASET[name] if args.head_l2 is None else args.head_l2
        metrics, parameters = benchmark(
            dataset,
            configs,
            seeds,
            args.n_splits,
            head_l2=head_l2,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
        slice_frame = full_fit_slice_metrics(
            dataset,
            configs,
            head_l2=head_l2,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        if not slice_frame.empty:
            slice_frames.append(slice_frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    slices = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    external_frame = external_intervention_metrics(
        datasets,
        external,
        configs,
        head_l2_override=args.head_l2,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameter_diagnostics.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    external_frame.to_csv(args.output_dir / "external_two_phase_summary.csv", index=False)
    figure = px.scatter(
        summary,
        x="oof_rmse_mean",
        y="oof_spearman_mean",
        color="model",
        facet_col="dataset",
        hover_data=["nominal_param_count", "fold_mean_regret_at_1_mean"],
        title="Calibrated cumulative-recency grouped-CV comparison",
    )
    figure.write_html(
        args.output_dir / "cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    report = [
        "# Calibrated cumulative-recency response scales",
        "",
        "The nested baseline uses empirical bucket response scales with global shifts. "
        "Calibrated variants either learn positive dispersion slopes or a shared-channel, "
        "mean-zero sample-efficiency offset per bucket. All added parameters are shrunk "
        "toward zero, which exactly recovers the baseline.",
        "",
        "## Grouped-CV metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## StarCoder phase-0 Nemotron slice",
        "",
        slices.to_markdown(index=False) if not slices.empty else "Not evaluated.",
        "",
        "## Untouched 300M two-phase interventions",
        "",
        external_frame.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    if not slices.empty:
        print(slices.to_string(index=False))
    print(f"Wrote calibrated cumulative-recency benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
