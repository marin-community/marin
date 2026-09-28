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
"""Test minimal cross-phase state-transition terms on separate heads.

Let ``s0_i`` and ``s1_i`` be saturating learned-state proxies for bucket ``i``
after each phase's direct exposure. Additive phase heads cannot represent that
phase-1 learning acts on a state created in phase 0. This benchmark tests:

* ``overlap``: signed ``s0_i s1_i`` synergy or redundancy;
* ``retention``: nonnegative harm ``s0_i (1 - w1_i)`` from exposing an early
  learned state to nonmatching late data;
* ``both``: both nested mechanisms.

The features add at most two coefficients per bucket and use empirical DSP
response rates, so they are a low-parameter state-transition correction rather
than a generic pairwise interaction tensor.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear
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
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    benchmark_cumulative_recency_starcoder as recency,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "state_transition_interactions_20260710"
BASE_L2 = 0.1
SCALE_FLOOR = 1e-8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class InteractionKind(StrEnum):
    """Nested state-transition mechanisms."""

    NONE = "none"
    OVERLAP = "overlap"
    RETENTION = "retention"
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
        return f"state_{self.kind.value}_l2_{self.interaction_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Separate phase bowls plus state-transition interactions."""

    config: Config
    mu0: np.ndarray
    mu1: np.ndarray
    rho0: np.ndarray
    rho1: np.ndarray
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
        interaction_multiplier = {
            InteractionKind.NONE: 0,
            InteractionKind.OVERLAP: 1,
            InteractionKind.RETENTION: 1,
            InteractionKind.BOTH: 2,
        }[self.config.kind]
        return (4 + interaction_multiplier) * self.num_domains + 3


def interaction_design(
    weights: np.ndarray,
    exposure0: np.ndarray,
    exposure1: np.ndarray,
    rho0: np.ndarray,
    rho1: np.ndarray,
    kind: InteractionKind,
) -> np.ndarray:
    """Build learned-state overlap and late-interference features."""
    if kind is InteractionKind.NONE:
        return np.empty((len(weights), 0), dtype=float)
    state0 = -np.expm1(-np.maximum(exposure0 * rho0[None, :], 0.0))
    state1 = -np.expm1(-np.maximum(exposure1 * rho1[None, :], 0.0))
    pieces = []
    if kind in (InteractionKind.OVERLAP, InteractionKind.BOTH):
        pieces.append(state0 * state1)
    if kind in (InteractionKind.RETENTION, InteractionKind.BOTH):
        pieces.append(state0 * (1.0 - weights[:, 1, :]))
    return np.hstack(pieces)


def interaction_lower_bounds(kind: InteractionKind, num_domains: int) -> np.ndarray:
    """Constrain retention harm while allowing overlap synergy or redundancy."""
    if kind is InteractionKind.NONE:
        return np.empty(0, dtype=float)
    if kind is InteractionKind.OVERLAP:
        return np.full(num_domains, -np.inf)
    if kind is InteractionKind.RETENTION:
        return np.zeros(num_domains, dtype=float)
    return np.concatenate([np.full(num_domains, -np.inf), np.zeros(num_domains, dtype=float)])


def fit_model(dataset: pooled.Dataset, indices: np.ndarray, config: Config) -> FittedModel:
    """Fit constrained phase heads and state-transition coefficients."""
    weights = dataset.weights[indices]
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    target = dataset.y[indices]
    mu0 = pooled.selected_mu(exposure0, target)
    mu1 = pooled.selected_mu(exposure1, target)
    rho0 = recency.channel_base(exposure0).rho
    rho1 = recency.channel_base(exposure1).rho
    design0 = pooled.bowl_design(exposure0, mu0)
    design1 = pooled.bowl_design(exposure1, mu1)
    interaction = interaction_design(weights, exposure0, exposure1, rho0, rho1, config.kind)
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
    lower = np.concatenate([np.zeros(phase_count), interaction_lower_bounds(config.kind, dataset.m)])
    upper = np.full(design.shape[1], np.inf)
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not result.success:
        raise RuntimeError(f"State-transition fit failed: {result.message}")
    coef = np.asarray(result.x, dtype=float)
    intercept = target_mean - float((center @ coef).item())
    per_phase = design0.shape[1]
    return FittedModel(
        config=config,
        mu0=mu0,
        mu1=mu1,
        rho0=np.asarray(rho0, dtype=float),
        rho1=np.asarray(rho1, dtype=float),
        interaction_scale=interaction_scale,
        intercept=intercept,
        coef0=coef[:per_phase],
        coef1=coef[per_phase : 2 * per_phase],
        interaction_coef=coef[2 * per_phase :],
    )


def predict(model: FittedModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    """Predict held-out rows."""
    weights = dataset.weights[indices]
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    prediction = (
        model.intercept
        + pooled.bowl_design(exposure0, model.mu0) @ model.coef0
        + pooled.bowl_design(exposure1, model.mu1) @ model.coef1
    )
    interaction = interaction_design(
        weights,
        exposure0,
        exposure1,
        model.rho0,
        model.rho1,
        model.config.kind,
    )
    if interaction.shape[1]:
        prediction += (interaction / model.interaction_scale[None, :]) @ model.interaction_coef
    return np.asarray(prediction, dtype=float)


def configs(l2_values: list[float]) -> list[Config]:
    """Return additive control plus nested interaction sweep."""
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
                        "interaction_coef_l2": float(np.linalg.norm(model.interaction_coef)),
                        "active_interactions": int(np.sum(np.abs(model.interaction_coef) > 1e-10)),
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
        title="State-transition interactions",
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
    names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    model_configs = configs(pooled.parse_float_list(args.interaction_l2_values))
    metric_frames = []
    parameter_frames = []
    for name in names:
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
        "# State-transition interactions",
        "",
        "The tested interactions are nested low-parameter corrections to separate phase heads.",
        "",
        summary.to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(slices.to_string(index=False))
    print(f"Wrote state-transition benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
