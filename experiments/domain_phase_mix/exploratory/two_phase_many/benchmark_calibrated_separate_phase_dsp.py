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
"""Fit calibrated DSP learning and overexposure curves separately by phase.

For phase exposures ``e0`` and ``e1``, the model is

    L = b - a0 S0(e0) + p0 H0(e0) - a1 S1(e1) + p1 H1(e1).

Each bucket has nonnegative phase-specific learning and overexposure
amplitudes. Response locations are empirical exposure priors calibrated by a
small number of global shifts, spreads, or powers. This preserves DSP's
mechanistic U-shaped response while avoiding both effective-exposure's scalar
phase multiplier and the quadratic approximation used by separate heads.
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
    benchmark_calibrated_cumulative_recency as calibrated,
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "calibrated_separate_phase_dsp_20260710"
SHAPE_BOUND = 3.0
POWER_BOUND = 1.0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FittedModel:
    """Calibrated DSP curves with independent phase heads."""

    config: calibrated.ModelConfig
    phase0_base: recency.ChannelBase
    phase1_base: recency.ChannelBase
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
        extra = {
            calibrated.CalibrationKind.SHIFT_ONLY: 0,
            calibrated.CalibrationKind.SHARED_SPREAD: 2,
            calibrated.CalibrationKind.CHANNEL_SPREAD: 4,
            calibrated.CalibrationKind.DOMAIN_EFFICIENCY: self.num_domains,
            calibrated.CalibrationKind.SHARED_POWER: 2,
            calibrated.CalibrationKind.CHANNEL_POWER: 4,
        }[self.config.kind]
        return 4 * self.num_domains + 5 + extra

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(
            weights,
            self.c0,
            self.c1,
            self.phase0_base,
            self.phase1_base,
            self.config.kind,
            self.shape_parameters,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    phase0_base: recency.ChannelBase,
    phase1_base: recency.ChannelBase,
    kind: calibrated.CalibrationKind,
    shape_parameters: np.ndarray,
) -> np.ndarray:
    """Build phase-specific learning and overexposure features."""
    exposure0 = weights[:, 0, :] * c0[None, :]
    exposure1 = weights[:, 1, :] * c1[None, :]
    phase0_parameters, phase1_parameters, efficiency_offsets = calibrated.channel_parameters(
        kind,
        shape_parameters,
        len(c0),
    )
    rho0, tau0 = calibrated.calibrated_shape(phase0_base, *phase0_parameters, efficiency_offsets)
    rho1, tau1 = calibrated.calibrated_shape(phase1_base, *phase1_parameters, efficiency_offsets)
    powers0, powers1 = calibrated.response_powers(kind, shape_parameters)
    benefit0, penalty0 = calibrated.channel_features(exposure0, rho0, tau0, *powers0)
    benefit1, penalty1 = calibrated.channel_features(exposure1, rho1, tau1, *powers1)
    return np.hstack([-benefit0, penalty0, -benefit1, penalty1])


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: calibrated.ModelConfig,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> FittedModel:
    """Profile nonnegative phase heads over global response calibrations."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    phase0_base = recency.channel_base(exposure0)
    phase1_base = recency.channel_base(exposure1)
    head_l2 = calibrated.HEAD_L2_BY_DATASET[dataset.name]

    def objective(shape_parameters: np.ndarray) -> float:
        design = design_matrix(
            weights,
            dataset.c0,
            dataset.c1,
            phase0_base,
            phase1_base,
            config.kind,
            np.asarray(shape_parameters, dtype=float),
        )
        intercept, coef = cross.fit_head(design, targets, head_l2)
        residual = intercept + design @ coef - targets
        shrunk = calibrated.spread_parameters(config.kind, np.asarray(shape_parameters, dtype=float))
        penalty = config.spread_l2 * float(shrunk @ shrunk) / len(targets)
        return float(np.mean(residual**2) + penalty)

    starts = calibrated.initial_shape_parameters(config.kind, dataset.m)
    scored = [(objective(start), start) for start in starts]
    scored.sort(key=lambda item: item[0])
    selected = [start for _score, start in scored[:coarse_top_k]]
    extra_bound = (
        POWER_BOUND
        if config.kind
        in (
            calibrated.CalibrationKind.SHARED_POWER,
            calibrated.CalibrationKind.CHANNEL_POWER,
        )
        else SHAPE_BOUND
    )
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
        phase0_base,
        phase1_base,
        config.kind,
        shape_parameters,
    )
    intercept, coef = cross.fit_head(design, targets, head_l2)
    return FittedModel(
        config=config,
        phase0_base=phase0_base,
        phase1_base=phase1_base,
        shape_parameters=shape_parameters,
        intercept=intercept,
        coef=coef,
        head_l2=head_l2,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[calibrated.ModelConfig],
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
            folds = centered.folds_for(dataset, seed, n_splits)
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
                    }
                )
            row = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            row["nominal_param_count"] = model.parameter_count
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    configs: list[calibrated.ModelConfig],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure the dense phase-0-Nemotron StarCoder slice after full fits."""
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    indices = np.flatnonzero(mask)
    rows = []
    for config in configs:
        model = fit_model(
            dataset,
            np.arange(dataset.n),
            config,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        prediction = model.predict(dataset.weights[indices])
        targets = dataset.y[indices]
        phase1 = dataset.frame.iloc[indices]["phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": config.name,
                "slice_rows": len(indices),
                "slice_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                "slice_spearman": float(spearmanr(targets, prediction).statistic),
                "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                "predicted_min_bpb": float(prediction[minimum]),
            }
        )
    return pd.DataFrame(rows)


def parse_configs(kinds: str, spread_l2_values: str) -> list[calibrated.ModelConfig]:
    """Parse only calibration kinds supported by the separate-phase form."""
    supported = {
        calibrated.CalibrationKind.SHIFT_ONLY,
        calibrated.CalibrationKind.SHARED_SPREAD,
        calibrated.CalibrationKind.CHANNEL_SPREAD,
        calibrated.CalibrationKind.DOMAIN_EFFICIENCY,
        calibrated.CalibrationKind.SHARED_POWER,
        calibrated.CalibrationKind.CHANNEL_POWER,
    }
    values = pooled.parse_float_list(spread_l2_values)
    configs = []
    for raw in (part.strip() for part in kinds.split(",") if part.strip()):
        kind = calibrated.CalibrationKind(raw)
        if kind not in supported:
            raise ValueError(f"Unsupported separate-phase calibration {kind}")
        l2_values = [0.0] if kind is calibrated.CalibrationKind.SHIFT_ONLY else values
        configs.extend(calibrated.ModelConfig(kind, l2) for l2 in l2_values)
    return configs


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
        title="Calibrated separate-phase DSP",
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
    parser.add_argument(
        "--kinds",
        default="shift_only,shared_spread,channel_spread,shared_power,channel_power",
    )
    parser.add_argument("--spread-l2-values", default="0.001")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    configs = parse_configs(args.kinds, args.spread_l2_values)
    metric_frames = []
    parameter_frames = []
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
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    slices = starcoder_slice_summary(
        datasets[centered.STARCODER_NAME],
        configs,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Calibrated separate-phase DSP",
        "",
        "Each phase receives a mechanistic saturating-learning plus overexposure response curve.",
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
    print(f"Wrote calibrated separate-phase benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
