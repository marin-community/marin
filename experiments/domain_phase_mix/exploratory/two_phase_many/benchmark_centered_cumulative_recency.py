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
"""Fit DSP in aggregate-exposure plus centered phase-order coordinates.

For total exposure ``E = e0 + e1`` and the phase-1 exposure of the exact tied
schedule ``R_tied = E * c1 / (c0 + c1)``, the model is

    L = b - a S(E) + p H(E)
          - r [S(e1) - S(R_tied)] + q [H(e1) - H(R_tied)].

The phase-order channel vanishes exactly for a constant mixture. Unlike the
uncentered cumulative-recency form, it cannot improve aggregate exposure by
collecting a second independent late-benefit vector. Nonnegative heads retain
the interpretation that extra late learning helps and extra late repetition
hurts relative to the tied schedule.
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
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    benchmark_cumulative_recency_starcoder as recency,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_cumulative_recency_20260710"
POWER_L2 = 0.001
SHAPE_BOUND = 3.0
POWER_BOUND = 1.0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FittedCenteredModel:
    """Aggregate DSP plus a zero-at-tied order channel."""

    config: generalized.ModelConfig
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
    def parameter_count(self) -> int:
        extra = 2 if self.config.kind is generalized.CalibrationKind.SHARED_POWER else 0
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


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    cumulative_base: recency.ChannelBase,
    recency_base: recency.ChannelBase,
    kind: generalized.CalibrationKind,
    shape_parameters: np.ndarray,
) -> np.ndarray:
    """Build aggregate response and centered late-response differences."""
    cumulative, late = recency.exposures(weights, c0, c1)
    tied_late = cumulative * (c1 / (c0 + c1))[None, :]
    cumulative_parameters, recency_parameters, efficiency_offsets = generalized.channel_parameters(
        kind,
        shape_parameters,
        len(c0),
    )
    cumulative_rho, cumulative_tau = generalized.calibrated_shape(
        cumulative_base,
        *cumulative_parameters,
        efficiency_offsets,
    )
    late_rho, late_tau = generalized.calibrated_shape(
        recency_base,
        *recency_parameters,
        efficiency_offsets,
    )
    cumulative_powers, recency_powers = generalized.response_powers(kind, shape_parameters)
    cumulative_benefit, cumulative_penalty = generalized.channel_features(
        cumulative,
        cumulative_rho,
        cumulative_tau,
        *cumulative_powers,
    )
    late_benefit, late_penalty = generalized.channel_features(
        late,
        late_rho,
        late_tau,
        *recency_powers,
    )
    tied_benefit, tied_penalty = generalized.channel_features(
        tied_late,
        late_rho,
        late_tau,
        *recency_powers,
    )
    return np.hstack(
        [
            -cumulative_benefit,
            cumulative_penalty,
            -(late_benefit - tied_benefit),
            late_penalty - tied_penalty,
        ]
    )


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: generalized.ModelConfig,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> FittedCenteredModel:
    """Profile centered nonnegative heads over shared response shapes."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    cumulative, late = recency.exposures(weights, dataset.c0, dataset.c1)
    tied_late = cumulative * (dataset.c1 / (dataset.c0 + dataset.c1))[None, :]
    cumulative_base = recency.channel_base(cumulative)
    recency_base = recency.channel_base(np.vstack([late, tied_late]))
    head_l2 = generalized.HEAD_L2_BY_DATASET[dataset.name]

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
        shrunk = generalized.spread_parameters(config.kind, np.asarray(shape_parameters, dtype=float))
        penalty = config.spread_l2 * float(shrunk @ shrunk) / len(targets)
        return float(np.mean(residual**2) + penalty)

    starts = generalized.initial_shape_parameters(config.kind, dataset.m)
    scored = [(objective(start), start) for start in starts]
    scored.sort(key=lambda item: item[0])
    selected = [start for _score, start in scored[:coarse_top_k]]
    extra_bound = POWER_BOUND if config.kind is generalized.CalibrationKind.SHARED_POWER else SHAPE_BOUND
    bounds = [(-SHAPE_BOUND, SHAPE_BOUND)] * 4 + [(-extra_bound, extra_bound)] * (len(selected[0]) - 4)
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "maxls": 30},
        )
        for start in selected
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
    return FittedCenteredModel(
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


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[generalized.ModelConfig],
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            folds = cross.folds_for(dataset, seed, n_splits)
            prediction = np.zeros(dataset.n, dtype=float)
            for fold_id, (train_indices, test_indices) in enumerate(folds):
                print(f"{dataset.name}/{config.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
                model = fit_model(
                    dataset,
                    train_indices,
                    config,
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
                        "coef_norm": float(np.linalg.norm(model.coef)),
                        "active_order_coef": int(np.sum(model.coef[2 * dataset.m :] > 1e-10)),
                    }
                )
            row = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            row["nominal_param_count"] = model.parameter_count
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    configs: list[generalized.ModelConfig],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure the dense phase-0-Nemotron StarCoder slice after full fits."""
    if dataset.name != centered.STARCODER_NAME:
        return pd.DataFrame()
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    rows = []
    for config in configs:
        model = fit_model(
            dataset,
            np.arange(dataset.n),
            config,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        prediction = model.predict(dataset.weights[mask])
        targets = dataset.y[mask]
        phase1 = dataset.frame.loc[mask, "phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": config.name,
                "slice_rows": int(mask.sum()),
                "slice_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                "slice_spearman": float(spearmanr(targets, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a compact cross-swarm comparison."""
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
        title="Centered cumulative-recency DSP",
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
    parser.add_argument("--kinds", default="shift_only,shared_power")
    parser.add_argument("--power-l2-values", default=str(POWER_L2))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    kinds = [generalized.CalibrationKind(value.strip()) for value in args.kinds.split(",") if value.strip()]
    configs = []
    for kind in kinds:
        values = (
            [0.0] if kind is generalized.CalibrationKind.SHIFT_ONLY else pooled.parse_float_list(args.power_l2_values)
        )
        configs.extend(generalized.ModelConfig(kind, value) for value in values)
    seeds = pooled.parse_int_list(args.seeds)
    metric_frames = []
    parameter_frames = []
    slice_frames = []
    for name in names:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            configs,
            seeds,
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
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Centered cumulative-recency DSP",
        "",
        (
            "The order channel is exactly zero for tied schedules and uses no more parameters than uncentered "
            "cumulative-recency DSP."
        ),
        "",
        summary.to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False) if not slices.empty else "Not evaluated.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(slices.to_string(index=False))
    print(f"Wrote centered cumulative-recency benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
