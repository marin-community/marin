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
"""Screen a retained-state extension of cumulative-recency DSP.

The cumulative-recency model uses total exposure for durable learning and only
phase-1 exposure for final-state utility. This script replaces the latter with

    R_lambda = lambda * e0 + e1,  0 <= lambda <= 1.

``lambda`` is the retained fraction of phase-0 state at the final checkpoint.
The current model is nested exactly at ``lambda=0``; ``lambda=1`` removes the
distinction between early and late exposure. One global scalar adds a direct,
mechanistic phase interaction without introducing bucket-pair parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "retained_state_cumulative_recency_20260710"
SHARED_POWER_CONFIG = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, 0.001)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
RETENTION_STARTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Config:
    """One retention-shrinkage setting."""

    retention_l2: float

    @property
    def name(self) -> str:
        return f"retained_state_l2_{self.retention_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Cumulative learning plus a retained final-state channel."""

    config: Config | None
    cumulative_base: recency.ChannelBase
    retained_base: recency.ChannelBase
    shape_parameters: np.ndarray
    retention: float
    intercept: float
    coef: np.ndarray
    head_l2: float
    c0: np.ndarray
    c1: np.ndarray

    @property
    def name(self) -> str:
        return "shared_power_lambda_0" if self.config is None else self.config.name

    @property
    def parameter_count(self) -> int:
        return 4 * len(self.c0) + 7 + int(self.config is not None)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(
            weights,
            self.c0,
            self.c1,
            self.cumulative_base,
            self.retained_base,
            self.shape_parameters,
            self.retention,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    cumulative_base: recency.ChannelBase,
    retained_base: recency.ChannelBase,
    shape_parameters: np.ndarray,
    retention: float,
) -> np.ndarray:
    """Build durable-learning and retained-final-state response features."""
    phase0 = weights[:, 0, :] * c0[None, :]
    phase1 = weights[:, 1, :] * c1[None, :]
    cumulative = phase0 + phase1
    retained = retention * phase0 + phase1
    cumulative_parameters, retained_parameters, efficiency_offsets = generalized.channel_parameters(
        generalized.CalibrationKind.SHARED_POWER,
        shape_parameters,
        len(c0),
    )
    cumulative_rho, cumulative_tau = generalized.calibrated_shape(
        cumulative_base,
        *cumulative_parameters,
        efficiency_offsets,
    )
    retained_rho, retained_tau = generalized.calibrated_shape(
        retained_base,
        *retained_parameters,
        efficiency_offsets,
    )
    cumulative_powers, retained_powers = generalized.response_powers(
        generalized.CalibrationKind.SHARED_POWER,
        shape_parameters,
    )
    cumulative_benefit, cumulative_penalty = generalized.channel_features(
        cumulative,
        cumulative_rho,
        cumulative_tau,
        *cumulative_powers,
    )
    retained_benefit, retained_penalty = generalized.channel_features(
        retained,
        retained_rho,
        retained_tau,
        *retained_powers,
    )
    return np.hstack(
        [
            -cumulative_benefit,
            cumulative_penalty,
            -retained_benefit,
            retained_penalty,
        ]
    )


def wrap_baseline(model: generalized.FittedModel) -> FittedModel:
    """Expose the exact lambda-zero incumbent through the retained-state API."""
    return FittedModel(
        config=None,
        cumulative_base=model.cumulative_base,
        retained_base=model.recency_base,
        shape_parameters=model.shape_parameters,
        retention=0.0,
        intercept=model.intercept,
        coef=model.coef,
        head_l2=model.head_l2,
        c0=model.c0,
        c1=model.c1,
    )


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
    baseline: generalized.FittedModel,
    *,
    maxiter: int,
) -> FittedModel:
    """Profile one retained-state scalar jointly with the shared response shape."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    cumulative_base = baseline.cumulative_base
    retained_base = baseline.recency_base
    head_l2 = baseline.head_l2

    def objective(parameters: np.ndarray) -> float:
        shape_parameters = np.asarray(parameters[:-1], dtype=float)
        retention = float(parameters[-1])
        design = design_matrix(
            weights,
            dataset.c0,
            dataset.c1,
            cumulative_base,
            retained_base,
            shape_parameters,
            retention,
        )
        intercept, coef = cross.fit_head(design, targets, head_l2)
        residual = intercept + design @ coef - targets
        shape_penalty = SHARED_POWER_CONFIG.spread_l2 * float(shape_parameters[4:] @ shape_parameters[4:])
        retention_penalty = config.retention_l2 * retention**2
        return float(np.mean(residual**2) + (shape_penalty + retention_penalty) / len(targets))

    starts = [np.r_[baseline.shape_parameters, retention] for retention in RETENTION_STARTS]
    bounds = [
        *[(-generalized.SHIFT_BOUND, generalized.SHIFT_BOUND)] * 4,
        *[(-generalized.LOG_POWER_BOUND, generalized.LOG_POWER_BOUND)] * 2,
        (0.0, 1.0),
    ]
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "maxls": 30},
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(result.fun))
    shape_parameters = np.asarray(best.x[:-1], dtype=float)
    retention = float(best.x[-1])
    design = design_matrix(
        weights,
        dataset.c0,
        dataset.c1,
        cumulative_base,
        retained_base,
        shape_parameters,
        retention,
    )
    intercept, coef = cross.fit_head(design, targets, head_l2)
    return FittedModel(
        config=config,
        cumulative_base=cumulative_base,
        retained_base=retained_base,
        shape_parameters=shape_parameters,
        retention=retention,
        intercept=intercept,
        coef=coef,
        head_l2=head_l2,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
    )


def fit_fold_models(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> list[FittedModel]:
    """Fit the incumbent once, then all nested retention settings."""
    baseline = generalized.fit_model(
        dataset,
        indices,
        SHARED_POWER_CONFIG,
        generalized.HEAD_L2_BY_DATASET[dataset.name],
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )
    return [wrap_baseline(baseline)] + [
        fit_model(dataset, indices, config, baseline, maxiter=maxiter) for config in configs
    ]


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
    labels = ["shared_power_lambda_0", *[config.name for config in configs]]
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = cross.folds_for(dataset, seed, n_splits)
        predictions = {label: np.zeros(dataset.n, dtype=float) for label in labels}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            models = fit_fold_models(
                dataset,
                train_indices,
                configs,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            for model in models:
                predictions[model.name][test_indices] = model.predict(dataset.weights[test_indices])
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": model.name,
                        "seed": seed,
                        "fold": fold_id,
                        "retention": model.retention,
                        "shape_parameters": json.dumps(model.shape_parameters.tolist()),
                        "coef_norm": float(np.linalg.norm(model.coef)),
                    }
                )
        for label in labels:
            row = asdict(pooled.metrics(dataset, label, seed, predictions[label], folds))
            row["nominal_param_count"] = 4 * dataset.m + 7 + int(label != "shared_power_lambda_0")
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure full-fit fidelity on the dense phase-0-Nemotron slice."""
    if dataset.name != centered.STARCODER_NAME:
        return pd.DataFrame()
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    models = fit_fold_models(
        dataset,
        np.arange(dataset.n),
        configs,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )
    rows = []
    for model in models:
        prediction = model.predict(dataset.weights[mask])
        targets = dataset.y[mask]
        phase1 = dataset.frame.loc[mask, "phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": model.name,
                "retention": model.retention,
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
    configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Score untouched 300M interventions after full-panel fits."""
    rows = []
    for dataset_name, external_dataset in external.items():
        dataset = datasets[dataset_name]
        models = fit_fold_models(
            dataset,
            np.arange(dataset.n),
            configs,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        for model in models:
            row = joint.external_metrics(
                model.name,
                external_dataset.y,
                model.predict(external_dataset.weights),
            )
            row["dataset"] = dataset_name
            row["external_rows"] = external_dataset.n
            row["retention"] = model.retention
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--retention-l2-values", default="0,0.001,0.01,0.1,1")
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
    configs = [Config(value) for value in pooled.parse_float_list(args.retention_l2_values)]
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
        title="Retained-state cumulative-recency grouped-CV comparison",
    )
    figure.write_html(
        args.output_dir / "cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    report = [
        "# Retained-state cumulative-recency DSP",
        "",
        "The single added parameter is the retained fraction of phase-0 state in the final-state channel. "
        "The current cumulative-recency model is recovered exactly at lambda=0.",
        "",
        "**Interpretation caveat:** retained-state variants receive additional nonlinear optimization after the "
        "incumbent fit. When fitted lambda is near zero, any apparent gain is optimization-budget confounding, "
        "not evidence for retained state. Compare only against a matched-budget incumbent refit.",
        "",
        "## Grouped-CV metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Retention stability",
        "",
        parameters.groupby(["dataset", "model"])["retention"]
        .agg(["mean", "std"])
        .reset_index()
        .to_markdown(index=False),
        "",
        "## StarCoder phase-0 Nemotron slice",
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
    print(parameters.groupby(["dataset", "model"])["retention"].agg(["mean", "std"]).to_string())
    if not slices.empty:
        print(slices.to_string(index=False))
    print(f"Wrote retained-state benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
