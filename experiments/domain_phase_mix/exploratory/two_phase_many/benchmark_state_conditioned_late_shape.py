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
"""Condition late learning and overexposure shapes on phase-0 capability.

The late response to a fixed dose need not be identical after different
phase-0 curricula. A high-capability phase-0 state can change both the rate at
which late examples are learned and the dose at which repetition becomes
harmful. This model adds at most two global interactions to the shared-power
cumulative-recency DSP:

    rho_late(x) = rho_late * exp(beta_rho * z0)
    tau_late(x) = tau_late + beta_tau * z0,

where ``z0`` is a centered scalar phase-0 capability score derived from the
incumbent cumulative-benefit head. The incumbent is nested exactly at zero.
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
    benchmark_calibrated_cumulative_recency as generalized,
)
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "state_conditioned_late_shape_20260710"
SHARED_POWER_CONFIG = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, 0.001)
BETA_BOUND = 3.0
SCALE_FLOOR = 1e-12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class GateKind(StrEnum):
    """Late response quantities conditioned on phase-0 state."""

    RATE = "rate"
    THRESHOLD = "threshold"
    BOTH = "rate_and_threshold"


@dataclass(frozen=True)
class Config:
    """State-conditioned mechanism and scalar shrinkage."""

    kind: GateKind
    beta_l2: float

    @property
    def name(self) -> str:
        return f"state_{self.kind.value}_l2_{self.beta_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Frozen response-scale prior with state-conditioned late shapes."""

    config: Config
    base: generalized.FittedModel
    beta_rho: float
    beta_tau: float
    capability_mean: float
    intercept: float
    coef: np.ndarray

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count + (2 if self.config.kind is GateKind.BOTH else 1)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _capability = design_matrix(
            self.base,
            weights,
            self.beta_rho,
            self.beta_tau,
            self.capability_mean,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def base_shapes(
    model: generalized.FittedModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """Decode incumbent channel geometry."""
    cumulative_parameters, late_parameters, efficiency_offsets = generalized.channel_parameters(
        generalized.CalibrationKind.SHARED_POWER,
        model.shape_parameters,
        model.num_domains,
    )
    cumulative_rho, cumulative_tau = generalized.calibrated_shape(
        model.cumulative_base,
        *cumulative_parameters,
        efficiency_offsets,
    )
    late_rho, late_tau = generalized.calibrated_shape(
        model.recency_base,
        *late_parameters,
        efficiency_offsets,
    )
    cumulative_powers, late_powers = generalized.response_powers(
        generalized.CalibrationKind.SHARED_POWER,
        model.shape_parameters,
    )
    return cumulative_rho, cumulative_tau, late_rho, late_tau, cumulative_powers, late_powers


def phase0_capability(
    model: generalized.FittedModel,
    weights: np.ndarray,
    cumulative_rho: np.ndarray,
    cumulative_tau: np.ndarray,
    cumulative_powers: tuple[float, float],
) -> np.ndarray:
    """Compute the normalized incumbent phase-0 learning state."""
    exposure0 = weights[:, 0, :] * model.c0[None, :]
    benefit, _penalty = generalized.channel_features(
        exposure0,
        cumulative_rho,
        cumulative_tau,
        *cumulative_powers,
    )
    amplitudes = model.coef[: model.num_domains]
    denominator = max(float(amplitudes.sum()), SCALE_FLOOR)
    return np.asarray(benefit @ amplitudes / denominator, dtype=float)


def design_matrix(
    model: generalized.FittedModel,
    weights: np.ndarray,
    beta_rho: float,
    beta_tau: float,
    capability_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cumulative features and state-conditioned late features."""
    cumulative, late = recency.exposures(weights, model.c0, model.c1)
    cumulative_rho, cumulative_tau, late_rho, late_tau, cumulative_powers, late_powers = base_shapes(model)
    capability = phase0_capability(
        model,
        weights,
        cumulative_rho,
        cumulative_tau,
        cumulative_powers,
    )
    centered_capability = capability - capability_mean
    cumulative_benefit, cumulative_penalty = generalized.channel_features(
        cumulative,
        cumulative_rho,
        cumulative_tau,
        *cumulative_powers,
    )
    scaled_late = np.maximum(
        late_rho[None, :] * late * np.exp(beta_rho * centered_capability[:, None]),
        0.0,
    )
    late_benefit = -np.expm1(-(scaled_late ** late_powers[0]))
    shifted_tau = late_tau[None, :] + beta_tau * centered_capability[:, None]
    late_penalty = recency.softplus(np.log1p(late) - shifted_tau) ** late_powers[1]
    design = np.hstack(
        [
            -cumulative_benefit,
            cumulative_penalty,
            -late_benefit,
            late_penalty,
        ]
    )
    return design, capability


def decode_beta(parameters: np.ndarray, kind: GateKind) -> tuple[float, float]:
    """Decode the active state-conditioned interactions."""
    if kind is GateKind.RATE:
        return float(parameters[0]), 0.0
    if kind is GateKind.THRESHOLD:
        return 0.0, float(parameters[0])
    return float(parameters[0]), float(parameters[1])


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
    base: generalized.FittedModel,
    *,
    maxiter: int,
) -> FittedModel:
    """Profile one or two bounded state interactions over a nonnegative head."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    cumulative_rho, cumulative_tau, _late_rho, _late_tau, cumulative_powers, _late_powers = base_shapes(base)
    capability = phase0_capability(
        base,
        weights,
        cumulative_rho,
        cumulative_tau,
        cumulative_powers,
    )
    capability_mean = float(np.mean(capability))

    def objective(parameters: np.ndarray) -> float:
        beta_rho, beta_tau = decode_beta(np.asarray(parameters, dtype=float), config.kind)
        design, _capability = design_matrix(base, weights, beta_rho, beta_tau, capability_mean)
        intercept, coef = cross.fit_head(design, targets, base.head_l2)
        residual = intercept + design @ coef - targets
        penalty = config.beta_l2 * float(parameters @ parameters) / len(targets)
        return float(np.mean(residual**2) + penalty)

    parameter_count = 2 if config.kind is GateKind.BOTH else 1
    starts = [np.full(parameter_count, value, dtype=float) for value in (-1.0, 0.0, 1.0)]
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-BETA_BOUND, BETA_BOUND)] * parameter_count,
            options={"maxiter": maxiter, "ftol": 1e-12, "maxls": 30},
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    beta_rho, beta_tau = decode_beta(np.asarray(best.x, dtype=float), config.kind)
    design, _capability = design_matrix(base, weights, beta_rho, beta_tau, capability_mean)
    intercept, coef = cross.fit_head(design, targets, base.head_l2)
    return FittedModel(
        config=config,
        base=base,
        beta_rho=beta_rho,
        beta_tau=beta_tau,
        capability_mean=capability_mean,
        intercept=intercept,
        coef=coef,
    )


def fit_base(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> generalized.FittedModel:
    """Fit the incumbent shared-power model once per fold."""
    return generalized.fit_model(
        dataset,
        indices,
        SHARED_POWER_CONFIG,
        generalized.HEAD_L2_BY_DATASET[dataset.name],
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[Config],
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    labels = ["shared_power_backbone", *[config.name for config in configs]]
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = centered.folds_for(dataset, seed, n_splits)
        predictions = {label: np.zeros(dataset.n, dtype=float) for label in labels}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            base = fit_base(
                dataset,
                train_indices,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            predictions[labels[0]][test_indices] = base.predict(dataset.weights[test_indices])
            for config in configs:
                model = fit_model(dataset, train_indices, config, base, maxiter=maxiter)
                predictions[config.name][test_indices] = model.predict(dataset.weights[test_indices])
                _design, capability = design_matrix(
                    base,
                    dataset.weights[train_indices],
                    model.beta_rho,
                    model.beta_tau,
                    model.capability_mean,
                )
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "beta_rho": model.beta_rho,
                        "beta_tau": model.beta_tau,
                        "capability_sd": float(np.std(capability)),
                        "shape_parameters": json.dumps(base.shape_parameters.tolist()),
                    }
                )
        for label in labels:
            row = asdict(pooled.metrics(dataset, label, seed, predictions[label], folds))
            extra = 0 if label == labels[0] else (2 if "rate_and_threshold" in label else 1)
            row["nominal_param_count"] = 4 * dataset.m + 7 + extra
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure full-fit and leave-slice-out StarCoder response."""
    if dataset.name != centered.STARCODER_NAME:
        return pd.DataFrame()
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    slice_indices = np.flatnonzero(mask)
    outside_indices = np.flatnonzero(~mask)
    rows = []
    for protocol, train_indices in (("full_fit", np.arange(dataset.n)), ("leave_slice_out", outside_indices)):
        base = fit_base(dataset, train_indices, maxiter=maxiter, coarse_top_k=coarse_top_k)
        candidates: list[tuple[str, Any]] = [("shared_power_backbone", base)]
        candidates.extend(
            (config.name, fit_model(dataset, train_indices, config, base, maxiter=maxiter)) for config in configs
        )
        for label, model in candidates:
            prediction = model.predict(dataset.weights[slice_indices])
            targets = dataset.y[slice_indices]
            phase1 = dataset.frame.iloc[slice_indices]["phase_1_starcoder"].to_numpy(dtype=float)
            minimum = int(np.argmin(prediction))
            rows.append(
                {
                    "protocol": protocol,
                    "model": label,
                    "slice_rows": len(slice_indices),
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
    configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Score untouched 300M interventions after full-panel fits."""
    rows = []
    for dataset_name, external_dataset in external.items():
        dataset = datasets[dataset_name]
        base = fit_base(dataset, np.arange(dataset.n), maxiter=maxiter, coarse_top_k=coarse_top_k)
        candidates: list[tuple[str, Any]] = [("shared_power_backbone", base)]
        candidates.extend(
            (config.name, fit_model(dataset, np.arange(dataset.n), config, base, maxiter=maxiter)) for config in configs
        )
        for label, model in candidates:
            row = joint.external_metrics(label, external_dataset.y, model.predict(external_dataset.weights))
            row["dataset"] = dataset_name
            row["external_rows"] = external_dataset.n
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--kinds", default="rate,threshold,rate_and_threshold")
    parser.add_argument("--beta-l2-values", default="0,0.01,0.1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, external = centered.load_datasets()
    names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = sorted(set(names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    configs = [
        Config(GateKind(kind.strip()), l2)
        for kind in args.kinds.split(",")
        if kind.strip()
        for l2 in pooled.parse_float_list(args.beta_l2_values)
    ]
    metric_frames = []
    parameter_frames = []
    slice_frames = []
    for name in names:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
        slices = starcoder_slice_summary(
            datasets[name],
            configs,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        if not slices.empty:
            slice_frames.append(slices)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    external_frame = external_intervention_metrics(
        datasets,
        external,
        configs,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    external_frame.to_csv(args.output_dir / "external_two_phase_summary.csv", index=False)
    figure = px.scatter(
        summary,
        x="oof_rmse_mean",
        y="oof_spearman_mean",
        color="model",
        facet_col="dataset",
        hover_data=["fold_mean_regret_at_1_mean", "low_tail_rmse_mean"],
        title="State-conditioned late-shape grouped-CV comparison",
    )
    figure.write_html(
        args.output_dir / "cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    report = [
        "# State-conditioned late response",
        "",
        "One or two global parameters let phase-0 capability alter late learning speed or repetition tolerance.",
        "",
        "## Grouped-CV metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Parameter stability",
        "",
        parameters.groupby(["dataset", "model"])[["beta_rho", "beta_tau", "capability_sd"]]
        .agg(["mean", "std"])
        .reset_index()
        .to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False) if not slices.empty else "Not evaluated.",
        "",
        "## Untouched 300M interventions",
        "",
        external_frame.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    if not slices.empty:
        print(slices.to_string(index=False))
    print(f"Wrote state-conditioned late-shape benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
