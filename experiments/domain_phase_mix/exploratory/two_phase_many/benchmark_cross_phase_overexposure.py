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
"""Test persistent same-bucket overexposure on top of separate phase heads.

Separate heads assume that the two phase responses add. This benchmark adds
only nested, nonnegative penalties that become active when a bucket is
overexposed across the full schedule:

* ``overlap`` is the product of the phase-specific excess log exposures;
* ``aggregate`` is a bowl on total exposure;
* ``both`` contains both terms.

The overlap term is the smallest mechanistic correction for persistent
repetition: it is zero unless the same bucket is above its response center in
both phases. The aggregate term tests the broader hypothesis that total
repetition, rather than phase-local repetition, drives the missing curvature.
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
from scipy.optimize import nnls
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "cross_phase_overexposure_20260710"
BASE_L2 = 0.1
SCALE_FLOOR = 1e-8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class InteractionKind(StrEnum):
    """Nested departures from additive phase heads."""

    NONE = "none"
    OVERLAP = "overlap"
    AGGREGATE = "aggregate"
    BOTH = "both"


@dataclass(frozen=True)
class Config:
    """Interaction family and ridge strength."""

    kind: InteractionKind
    interaction_l2: float

    @property
    def name(self) -> str:
        if self.kind is InteractionKind.NONE:
            return "separate_heads"
        return f"separate_heads_{self.kind.value}_l2_{self.interaction_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Separate phase bowls plus optional persistent-overexposure terms."""

    config: Config
    mu0: np.ndarray
    mu1: np.ndarray
    mu_aggregate: np.ndarray
    interaction_scale: np.ndarray
    intercept: float
    coef0: np.ndarray
    coef1: np.ndarray
    interaction_coef: np.ndarray

    @property
    def num_domains(self) -> int:
        return len(self.mu0)

    @property
    def parameter_count(self) -> int:
        interaction_count = {
            InteractionKind.NONE: 0,
            InteractionKind.OVERLAP: self.num_domains,
            InteractionKind.AGGREGATE: self.num_domains,
            InteractionKind.BOTH: 2 * self.num_domains,
        }[self.config.kind]
        return 4 * self.num_domains + interaction_count + 3


def interaction_design(
    exposure0: np.ndarray,
    exposure1: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    mu_aggregate: np.ndarray,
    kind: InteractionKind,
) -> np.ndarray:
    """Build nonnegative cross-phase repetition features."""
    if kind is InteractionKind.NONE:
        return np.empty((len(exposure0), 0), dtype=float)
    excess0 = np.maximum(np.log1p(exposure0) - mu0[None, :], 0.0)
    excess1 = np.maximum(np.log1p(exposure1) - mu1[None, :], 0.0)
    pieces = []
    if kind in (InteractionKind.OVERLAP, InteractionKind.BOTH):
        pieces.append(excess0 * excess1)
    if kind in (InteractionKind.AGGREGATE, InteractionKind.BOTH):
        aggregate_delta = np.log1p(exposure0 + exposure1) - mu_aggregate[None, :]
        pieces.append(np.maximum(aggregate_delta, 0.0) ** 2)
    return np.hstack(pieces)


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
) -> FittedModel:
    """Fit all nonnegative heads inside one fold."""
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    target = dataset.y[indices]
    mu0 = pooled.selected_mu(exposure0, target)
    mu1 = pooled.selected_mu(exposure1, target)
    mu_aggregate = pooled.selected_mu(exposure0 + exposure1, target)
    design0 = pooled.bowl_design(exposure0, mu0)
    design1 = pooled.bowl_design(exposure1, mu1)
    interaction = interaction_design(exposure0, exposure1, mu0, mu1, mu_aggregate, config.kind)
    interaction_scale = np.maximum(np.sqrt(np.mean(interaction**2, axis=0)), SCALE_FLOOR)
    if interaction.shape[1]:
        interaction = interaction / interaction_scale[None, :]
    design = np.hstack([design0, design1, interaction])
    center = design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = design - center
    centered_target = target - target_mean
    phase_count = design0.shape[1] + design1.shape[1]
    penalties = np.full(design.shape[1], np.sqrt(BASE_L2), dtype=float)
    penalties[phase_count:] = np.sqrt(config.interaction_l2)
    augmented_design = np.vstack([centered_design, np.diag(penalties)])
    augmented_target = np.concatenate([centered_target, np.zeros(design.shape[1], dtype=float)])
    coef, _residual = nnls(augmented_design, augmented_target, maxiter=20 * design.shape[1])
    intercept = target_mean - float((center @ coef).item())
    per_phase = design0.shape[1]
    return FittedModel(
        config=config,
        mu0=mu0,
        mu1=mu1,
        mu_aggregate=mu_aggregate,
        interaction_scale=interaction_scale,
        intercept=intercept,
        coef0=np.asarray(coef[:per_phase], dtype=float),
        coef1=np.asarray(coef[per_phase : 2 * per_phase], dtype=float),
        interaction_coef=np.asarray(coef[2 * per_phase :], dtype=float),
    )


def predict(model: FittedModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    """Predict held-out rows."""
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    prediction = (
        model.intercept
        + pooled.bowl_design(exposure0, model.mu0) @ model.coef0
        + pooled.bowl_design(exposure1, model.mu1) @ model.coef1
    )
    interaction = interaction_design(
        exposure0,
        exposure1,
        model.mu0,
        model.mu1,
        model.mu_aggregate,
        model.config.kind,
    )
    if interaction.shape[1]:
        prediction += (interaction / model.interaction_scale[None, :]) @ model.interaction_coef
    return np.asarray(prediction, dtype=float)


def configs(l2_values: list[float]) -> list[Config]:
    """Return the additive control and nested interaction sweep."""
    out = [Config(InteractionKind.NONE, BASE_L2)]
    out.extend(Config(kind, l2) for kind in tuple(InteractionKind)[1:] for l2 in l2_values)
    return out


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: list[Config],
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = centered.folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in model_configs:
                model = fit_model(dataset, train_indices, config)
                predictions[config.name][test_indices] = predict(model, dataset, test_indices)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "interaction_coef_l1": float(model.interaction_coef.sum()),
                        "active_interactions": int(np.sum(model.interaction_coef > 1e-10)),
                    }
                )
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = fit_model(dataset, np.arange(dataset.n), config).parameter_count
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(dataset: pooled.Dataset, model_configs: list[Config]) -> pd.DataFrame:
    """Measure full-fit response on the dense phase-0-Nemotron slice."""
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    indices = np.flatnonzero(mask)
    rows = []
    for config in model_configs:
        model = fit_model(dataset, np.arange(dataset.n), config)
        prediction = predict(model, dataset, indices)
        target = dataset.y[indices]
        phase1 = dataset.frame.iloc[indices]["phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": config.name,
                "slice_rows": len(indices),
                "slice_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
                "slice_spearman": float(spearmanr(target, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a compact cross-swarm comparison."""
    long = summary.melt(
        id_vars=["dataset", "model"],
        value_vars=["oof_rmse_mean", "oof_spearman_mean", "fold_mean_regret_at_1_mean"],
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
        title="Same-bucket cross-phase overexposure",
    )
    figure.update_layout(showlegend=False, height=1000)
    figure.write_html(output_dir / "crossswarm_cv_comparison.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--interaction-l2-values", default="0.001,0.01,0.1,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    selected = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    model_configs = configs(pooled.parse_float_list(args.interaction_l2_values))
    metric_frames = []
    parameter_frames = []
    for name in selected:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            model_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    slices = starcoder_slice_summary(datasets[centered.STARCODER_NAME], model_configs)
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Same-bucket cross-phase overexposure",
        "",
        "All interactions are nested nonnegative penalties on top of the established separate-heads model.",
        "",
        summary.to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False),
        "",
        "## Fold interaction activity",
        "",
        parameters.groupby(["dataset", "model"])[["interaction_coef_l1", "active_interactions"]]
        .mean()
        .reset_index()
        .to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    (args.output_dir / "model_configs.json").write_text(
        json.dumps([asdict(config) for config in model_configs], indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(slices.to_string(index=False))
    print(f"Wrote cross-phase overexposure benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
