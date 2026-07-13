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
"""Benchmark a two-timescale learning response in separate phase heads.

The StarCoder slice has a sharp initial learning regime followed by a slower
regime and eventual overexposure. A single saturating benefit term cannot fit
that shape. This benchmark adds one nested second learning timescale while
retaining DSP's nonnegative learning and overexposure heads:

    L = b - sum_{t,i,k} a[t,i,k] S[k](e[t,i])
          + sum_{t,i} p[t,i] H[t,i](e[t,i]).

``late_two`` adds the second timescale only to phase 1; ``both_two`` adds it to
both phases. Rates are empirical per-bucket priors multiplied by two shared
global scales, so the extra mechanism costs one amplitude per affected bucket
and two global nonlinear parameters rather than an unconstrained response
curve per bucket.
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
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    benchmark_cumulative_recency_starcoder as recency,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "two_timescale_separate_phase_dsp_20260710"
RATE_SCALE_MIN = 0.01
RATE_SCALE_MAX = 20.0
TAU_SHIFT_BOUND = 3.0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class TimescaleKind(StrEnum):
    """Nested allocations of the second learning timescale."""

    ONE = "one_timescale"
    LATE_TWO = "late_two_timescale"
    BOTH_TWO = "both_two_timescale"


@dataclass(frozen=True)
class Config:
    """Timescale family and ridge strength."""

    kind: TimescaleKind
    head_l2: float

    @property
    def name(self) -> str:
        return f"{self.kind.value}_head_l2_{self.head_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Separate-phase DSP with one or two shared learning timescales."""

    config: Config
    phase0_base: recency.ChannelBase
    phase1_base: recency.ChannelBase
    rate_scales: np.ndarray
    tau_shifts: np.ndarray
    intercept: float
    coef: np.ndarray
    c0: np.ndarray
    c1: np.ndarray

    @property
    def num_domains(self) -> int:
        return len(self.c0)

    @property
    def parameter_count(self) -> int:
        head_multiplier = {
            TimescaleKind.ONE: 4,
            TimescaleKind.LATE_TWO: 5,
            TimescaleKind.BOTH_TWO: 6,
        }[self.config.kind]
        nonlinear = 4 if self.config.kind is not TimescaleKind.ONE else 2
        return head_multiplier * self.num_domains + nonlinear + 1

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(
            weights,
            self.c0,
            self.c1,
            self.phase0_base,
            self.phase1_base,
            self.config.kind,
            self.rate_scales,
            self.tau_shifts,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def benefit(exposure: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """DSP saturating learning response."""
    return -np.expm1(-np.maximum(exposure * rho[None, :], 0.0))


def penalty(exposure: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """DSP overexposure response."""
    return recency.softplus(np.log1p(exposure) - tau[None, :]) ** 2


def design_matrix(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    phase0_base: recency.ChannelBase,
    phase1_base: recency.ChannelBase,
    kind: TimescaleKind,
    rate_scales: np.ndarray,
    tau_shifts: np.ndarray,
) -> np.ndarray:
    """Build nested one- or two-timescale phase features."""
    exposure0 = weights[:, 0, :] * c0[None, :]
    exposure1 = weights[:, 1, :] * c1[None, :]
    slow, fast = rate_scales
    pieces = []
    if kind is TimescaleKind.BOTH_TWO:
        pieces.extend(
            [
                -benefit(exposure0, slow * phase0_base.rho),
                -benefit(exposure0, fast * phase0_base.rho),
            ]
        )
    else:
        pieces.append(-benefit(exposure0, fast * phase0_base.rho))
    pieces.append(penalty(exposure0, phase0_base.tau + tau_shifts[0]))
    if kind in (TimescaleKind.LATE_TWO, TimescaleKind.BOTH_TWO):
        pieces.extend(
            [
                -benefit(exposure1, slow * phase1_base.rho),
                -benefit(exposure1, fast * phase1_base.rho),
            ]
        )
    else:
        pieces.append(-benefit(exposure1, fast * phase1_base.rho))
    pieces.append(penalty(exposure1, phase1_base.tau + tau_shifts[1]))
    return np.hstack(pieces)


def decode_shape(theta: np.ndarray, kind: TimescaleKind) -> tuple[np.ndarray, np.ndarray]:
    """Decode ordered rate scales and phase-specific penalty shifts."""
    if kind is TimescaleKind.ONE:
        scale = float(np.exp(np.clip(theta[0], np.log(RATE_SCALE_MIN), np.log(RATE_SCALE_MAX))))
        return np.asarray([scale, scale]), np.asarray(theta[1:3], dtype=float)
    rates = np.sort(np.exp(np.clip(theta[:2], np.log(RATE_SCALE_MIN), np.log(RATE_SCALE_MAX))))
    return np.asarray(rates, dtype=float), np.asarray(theta[2:4], dtype=float)


def shape_starts(kind: TimescaleKind) -> list[np.ndarray]:
    """Deterministic starts spanning a fast and slow response."""
    if kind is TimescaleKind.ONE:
        return [
            np.asarray([np.log(scale), shift0, shift1])
            for scale in (0.5, 1.0, 2.0)
            for shift0, shift1 in ((0.0, 0.0), (-1.0, -1.0), (1.0, 1.0))
        ]
    return [
        np.asarray([np.log(slow), np.log(fast), shift0, shift1])
        for slow, fast in ((0.05, 1.0), (0.1, 2.0), (0.25, 4.0), (0.5, 8.0))
        for shift0, shift1 in ((0.0, 0.0), (-1.0, -1.0), (1.0, 1.0))
    ]


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> FittedModel:
    """Profile nonnegative heads over shared timescale parameters."""
    weights = dataset.weights[indices]
    targets = dataset.y[indices]
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    phase0_base = recency.channel_base(exposure0)
    phase1_base = recency.channel_base(exposure1)

    def objective(theta: np.ndarray) -> float:
        rate_scales, tau_shifts = decode_shape(np.asarray(theta, dtype=float), config.kind)
        design = design_matrix(
            weights,
            dataset.c0,
            dataset.c1,
            phase0_base,
            phase1_base,
            config.kind,
            rate_scales,
            tau_shifts,
        )
        intercept, coef = cross.fit_head(design, targets, config.head_l2)
        residual = intercept + design @ coef - targets
        tail_count = max(5, int(np.ceil(0.15 * len(targets))))
        tail = np.argsort(intercept + design @ coef)[:tail_count]
        optimism = float(np.mean(np.maximum(-residual[tail], 0.0)))
        return float(np.sqrt(np.mean(residual**2)) + 0.25 * optimism)

    starts = shape_starts(config.kind)
    scored = [(objective(start), start) for start in starts]
    scored.sort(key=lambda item: item[0])
    parameter_count = 3 if config.kind is TimescaleKind.ONE else 4
    bounds = [(np.log(RATE_SCALE_MIN), np.log(RATE_SCALE_MAX))] * (parameter_count - 2)
    bounds.extend([(-TAU_SHIFT_BOUND, TAU_SHIFT_BOUND)] * 2)
    best_value, best_theta = scored[0]
    for _value, start in scored[:coarse_top_k]:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 30},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)
    rate_scales, tau_shifts = decode_shape(best_theta, config.kind)
    design = design_matrix(
        weights,
        dataset.c0,
        dataset.c1,
        phase0_base,
        phase1_base,
        config.kind,
        rate_scales,
        tau_shifts,
    )
    intercept, coef = cross.fit_head(design, targets, config.head_l2)
    return FittedModel(
        config=config,
        phase0_base=phase0_base,
        phase1_base=phase1_base,
        rate_scales=rate_scales,
        tau_shifts=tau_shifts,
        intercept=intercept,
        coef=coef,
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: list[Config],
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for config in model_configs:
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
                        "rate_scales": json.dumps(model.rate_scales.tolist()),
                        "tau_shifts": json.dumps(model.tau_shifts.tolist()),
                        "coef_norm": float(np.linalg.norm(model.coef)),
                    }
                )
            row = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            row["nominal_param_count"] = model.parameter_count
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    model_configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure the dense phase-0-Nemotron StarCoder slice after full fits."""
    mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    indices = np.flatnonzero(mask)
    rows = []
    for config in model_configs:
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


def configs(kinds: str, head_l2_values: str) -> list[Config]:
    """Parse a nested timescale and head-ridge sweep."""
    return [
        Config(TimescaleKind(kind.strip()), l2)
        for kind in kinds.split(",")
        if kind.strip()
        for l2 in pooled.parse_float_list(head_l2_values)
    ]


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
        title="Two-timescale separate-phase DSP",
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
    parser.add_argument("--kinds", default="one_timescale,late_two_timescale,both_two_timescale")
    parser.add_argument("--head-l2-values", default="0.01,0.1")
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
    model_configs = configs(args.kinds, args.head_l2_values)
    metric_frames = []
    parameter_frames = []
    for name in names:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            model_configs,
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
        model_configs,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Two-timescale separate-phase DSP",
        "",
        "The extra response term models fast memorization plus slower generalization without unconstrained splines.",
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
    print(f"Wrote two-timescale separate-phase benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
