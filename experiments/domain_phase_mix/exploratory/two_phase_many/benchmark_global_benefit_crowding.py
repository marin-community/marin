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
"""Test nested cross-bucket diminishing returns on cumulative-recency DSP.

Additive DSP can collect a separate predicted benefit from every bucket when
optimized far from the sampled panel. This benchmark leaves all per-bucket
response curves intact and adds a nonnegative quadratic correction to the
benefit score of a preliminary additive fit. The correction is exactly nested
at zero and is fit inside every outer fold.
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
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "global_benefit_crowding_20260710"
POWER_L2 = 0.001
SCALE_QUANTILE = 0.9
SCALE_FLOOR = 1e-8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class CrowdingKind(StrEnum):
    """Nested global diminishing-return corrections."""

    NONE = "none"
    TOTAL_QUADRATIC = "total_quadratic"
    CHANNEL_QUADRATIC = "channel_quadratic"


@dataclass(frozen=True)
class FittedCrowdingModel:
    """Shared-power DSP plus frozen-score quadratic correction."""

    base: generalized.FittedModel
    kind: CrowdingKind
    intercept: float
    coef: np.ndarray
    score_coef: np.ndarray
    score_scale: np.ndarray

    @property
    def num_domains(self) -> int:
        return self.base.num_domains

    @property
    def crowding_coef(self) -> np.ndarray:
        return self.coef[4 * self.num_domains :]

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count + len(self.crowding_coef)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = augmented_design(
            self.base,
            weights,
            self.kind,
            self.score_coef,
            self.score_scale,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def base_design(model: generalized.FittedModel, weights: np.ndarray) -> np.ndarray:
    """Build the additive shared-power design at fixed nonlinear shapes."""
    return generalized.design_matrix(
        weights,
        model.c0,
        model.c1,
        model.cumulative_base,
        model.recency_base,
        model.config.kind,
        model.shape_parameters,
    )


def benefit_scores(design: np.ndarray, score_coef: np.ndarray, num_domains: int) -> np.ndarray:
    """Return cumulative and recency benefits from a preliminary additive head."""
    cumulative = (-design[:, :num_domains]) @ score_coef[:num_domains]
    late = (-design[:, 2 * num_domains : 3 * num_domains]) @ score_coef[2 * num_domains : 3 * num_domains]
    return np.column_stack([cumulative, late])


def score_scales(scores: np.ndarray) -> np.ndarray:
    """Normalize score magnitudes using train-only robust upper quantiles."""
    scales = np.quantile(scores, SCALE_QUANTILE, axis=0)
    total = float(np.quantile(scores.sum(axis=1), SCALE_QUANTILE))
    return np.maximum(np.asarray([scales[0], scales[1], total]), SCALE_FLOOR)


def crowding_features(scores: np.ndarray, scales: np.ndarray, kind: CrowdingKind) -> np.ndarray:
    """Build nested convex corrections to accumulated predicted benefit."""
    if kind is CrowdingKind.NONE:
        return np.empty((len(scores), 0), dtype=float)
    if kind is CrowdingKind.TOTAL_QUADRATIC:
        return (scores.sum(axis=1, keepdims=True) / scales[2]) ** 2
    if kind is CrowdingKind.CHANNEL_QUADRATIC:
        return (scores / scales[None, :2]) ** 2
    raise ValueError(f"Unknown crowding kind {kind!r}")


def augmented_design(
    base: generalized.FittedModel,
    weights: np.ndarray,
    kind: CrowdingKind,
    score_coef: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Build the additive design plus cross-bucket diminishing returns."""
    design = base_design(base, weights)
    scores = benefit_scores(design, score_coef, base.num_domains)
    return np.hstack([design, crowding_features(scores, scales, kind)])


def fit_base_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> generalized.FittedModel:
    """Fit the shared nonlinear backbone used by every nested correction."""
    config = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, POWER_L2)
    head_l2 = generalized.HEAD_L2_BY_DATASET[dataset.name]
    return generalized.fit_model(
        dataset,
        indices,
        config,
        head_l2,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )


def fit_from_base(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    kind: CrowdingKind,
    base: generalized.FittedModel,
) -> FittedCrowdingModel:
    """Refit one nested crowding head at fixed nonlinear shapes."""
    head_l2 = generalized.HEAD_L2_BY_DATASET[dataset.name]
    train_design = base_design(base, dataset.weights[indices])
    score_coef = np.asarray(base.coef, dtype=float)
    scores = benefit_scores(train_design, score_coef, dataset.m)
    scales = score_scales(scores)
    design = np.hstack([train_design, crowding_features(scores, scales, kind)])
    intercept, coef = cross.fit_head(design, dataset.y[indices], head_l2)
    return FittedCrowdingModel(
        base=base,
        kind=kind,
        intercept=intercept,
        coef=coef,
        score_coef=score_coef,
        score_scale=scales,
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    kinds: list[CrowdingKind],
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped outer CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = cross.folds_for(dataset, seed, n_splits)
        predictions = {kind: np.zeros(dataset.n, dtype=float) for kind in kinds}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            base = fit_base_model(
                dataset,
                train_indices,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            for kind in kinds:
                model = fit_from_base(
                    dataset,
                    train_indices,
                    kind,
                    base,
                )
                predictions[kind][test_indices] = model.predict(dataset.weights[test_indices])
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": kind.value,
                        "seed": seed,
                        "fold": fold_id,
                        "benefit_power": generalized.response_powers(
                            model.base.config.kind,
                            model.base.shape_parameters,
                        )[0][0],
                        "penalty_power": generalized.response_powers(
                            model.base.config.kind,
                            model.base.shape_parameters,
                        )[0][1],
                        "crowding_coef": json.dumps(model.crowding_coef.tolist()),
                        "score_scale": json.dumps(model.score_scale.tolist()),
                    }
                )
        for kind in kinds:
            row = asdict(pooled.metrics(dataset, kind.value, seed, predictions[kind], folds))
            row["nominal_param_count"] = (
                model.base.parameter_count
                + (kind is CrowdingKind.TOTAL_QUADRATIC)
                + 2 * (kind is CrowdingKind.CHANNEL_QUADRATIC)
            )
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def full_fit_summary(
    datasets: dict[str, pooled.Dataset],
    names: list[str],
    kinds: list[CrowdingKind],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Record full-data curvature magnitudes before proposal optimization."""
    rows: list[dict[str, Any]] = []
    for name in names:
        dataset = datasets[name]
        indices = np.arange(dataset.n)
        base = fit_base_model(
            dataset,
            indices,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        for kind in kinds:
            print(f"Full fit {name}/{kind.value}", flush=True)
            model = fit_from_base(
                dataset,
                indices,
                kind,
                base,
            )
            prediction = model.predict(dataset.weights)
            rows.append(
                {
                    "dataset": name,
                    "model": kind.value,
                    "train_rmse": float(np.sqrt(np.mean((prediction - dataset.y) ** 2))),
                    "crowding_coef": json.dumps(model.crowding_coef.tolist()),
                    "crowding_active": bool(np.any(model.crowding_coef > 1e-10)),
                    "score_scale": json.dumps(model.score_scale.tolist()),
                }
            )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a compact cross-swarm OOF comparison."""
    long = summary.melt(
        id_vars=["dataset", "model"],
        value_vars=[
            "oof_rmse_mean",
            "oof_spearman_mean",
            "fold_mean_regret_at_1_mean",
            "lower_tail_optimism_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    figure = px.bar(
        long,
        x="model",
        y="value",
        color="model",
        facet_row="dataset",
        facet_col="metric",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Shared-power DSP with nested global benefit crowding",
    )
    figure.update_layout(showlegend=False, height=1000)
    figure.write_html(
        output_dir / "crossswarm_cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--kinds", default="none,total_quadratic,channel_quadratic")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    kinds = [CrowdingKind(value.strip()) for value in args.kinds.split(",") if value.strip()]
    seeds = pooled.parse_int_list(args.seeds)
    metric_frames = []
    parameter_frames = []
    for name in names:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            kinds,
            seeds,
            args.n_splits,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    full = full_fit_summary(
        datasets,
        names,
        kinds,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    full.to_csv(args.output_dir / "full_fit_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Global benefit-crowding benchmark",
        "",
        "The quadratic correction is exactly nested at zero and is built from train-fold-only additive benefit scores.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Full-fit curvature",
        "",
        full.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(full.to_string(index=False))
    print(f"Wrote global benefit-crowding benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
