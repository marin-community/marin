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
"""Pool additive late-phase benefits through a support-calibrated bottleneck.

The cumulative-recency model currently learns one independent late-benefit
head per bucket. Its frontier failure is dominated by collecting many such
benefits simultaneously while the late overexposure penalty remains nearly
unchanged. This benchmark preserves the preliminary additive model's relative
bucket values but replaces its late-benefit block with one scalar utility:

    c * (1 - exp(-u / c)),  u = late_benefit / train_q90.

The infinite-cap control is exactly the preliminary additive late benefit up
to a refitted scalar. Finite caps add no fitted parameter and express that a
shared final-state utility cannot grow additively without bound.
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
    benchmark_global_benefit_crowding as crowding,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "late_utility_bottleneck_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class BottleneckConfig:
    """One fixed support-relative late-utility cap."""

    cap_factor: float

    @property
    def name(self) -> str:
        return "late_linear" if np.isinf(self.cap_factor) else f"late_softcap_{self.cap_factor:g}"


@dataclass(frozen=True)
class FittedBottleneckModel:
    """Shared-power DSP with one pooled late-benefit utility."""

    base: crowding.generalized.FittedModel
    config: BottleneckConfig
    intercept: float
    coef: np.ndarray
    late_score_coef: np.ndarray
    late_score_scale: float

    @property
    def num_domains(self) -> int:
        return self.base.num_domains

    @property
    def late_utility_coef(self) -> float:
        return float(self.coef[-1])

    @property
    def parameter_count(self) -> int:
        return 3 * self.num_domains + 8

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = bottleneck_design(
            self.base,
            weights,
            self.config,
            self.late_score_coef,
            self.late_score_scale,
        )
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def parse_cap_factors(value: str) -> list[float]:
    """Parse comma-separated positive caps, accepting `inf`."""
    caps = [float(part.strip()) for part in value.split(",") if part.strip()]
    if any(cap <= 0.0 for cap in caps):
        raise ValueError("Late utility cap factors must be positive")
    return caps


def late_utility(score: np.ndarray, scale: float, cap_factor: float) -> np.ndarray:
    """Return a linear or concave support-normalized late utility."""
    normalized = np.maximum(score / scale, 0.0)
    if np.isinf(cap_factor):
        return normalized
    return -cap_factor * np.expm1(-normalized / cap_factor)


def bottleneck_design(
    base: crowding.generalized.FittedModel,
    weights: np.ndarray,
    config: BottleneckConfig,
    late_score_coef: np.ndarray,
    late_score_scale: float,
) -> np.ndarray:
    """Replace the per-bucket late-benefit block with one pooled utility."""
    design = crowding.base_design(base, weights)
    m = base.num_domains
    late_score = (-design[:, 2 * m : 3 * m]) @ late_score_coef
    utility = late_utility(late_score, late_score_scale, config.cap_factor)
    return np.hstack(
        [
            design[:, : 2 * m],
            design[:, 3 * m : 4 * m],
            -utility[:, None],
        ]
    )


def fit_from_base(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: BottleneckConfig,
    base: crowding.generalized.FittedModel,
) -> FittedBottleneckModel:
    """Fit the reduced nonnegative head at fixed nonlinear shapes."""
    base_train_design = crowding.base_design(base, dataset.weights[indices])
    m = dataset.m
    late_score_coef = np.asarray(base.coef[2 * m : 3 * m], dtype=float)
    late_score = (-base_train_design[:, 2 * m : 3 * m]) @ late_score_coef
    late_score_scale = max(float(np.quantile(late_score, crowding.SCALE_QUANTILE)), crowding.SCALE_FLOOR)
    design = bottleneck_design(
        base,
        dataset.weights[indices],
        config,
        late_score_coef,
        late_score_scale,
    )
    head_l2 = crowding.generalized.HEAD_L2_BY_DATASET[dataset.name]
    intercept, coef = cross.fit_head(design, dataset.y[indices], head_l2)
    return FittedBottleneckModel(
        base=base,
        config=config,
        intercept=intercept,
        coef=coef,
        late_score_coef=late_score_coef,
        late_score_scale=late_score_scale,
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[BottleneckConfig],
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = cross.folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in configs}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            base = crowding.fit_base_model(
                dataset,
                train_indices,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            for config in configs:
                model = fit_from_base(dataset, train_indices, config, base)
                predictions[config.name][test_indices] = model.predict(dataset.weights[test_indices])
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "late_utility_coef": model.late_utility_coef,
                        "late_score_scale": model.late_score_scale,
                        "late_score_coef_norm": float(np.linalg.norm(model.late_score_coef)),
                    }
                )
        for config in configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = 3 * dataset.m + 8
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def full_fit_summary(
    datasets: dict[str, pooled.Dataset],
    names: list[str],
    configs: list[BottleneckConfig],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Record full-data fit and late-utility calibration."""
    rows: list[dict[str, Any]] = []
    for name in names:
        dataset = datasets[name]
        indices = np.arange(dataset.n)
        base = crowding.fit_base_model(
            dataset,
            indices,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        for config in configs:
            model = fit_from_base(dataset, indices, config, base)
            prediction = model.predict(dataset.weights)
            rows.append(
                {
                    "dataset": name,
                    "model": config.name,
                    "train_rmse": float(np.sqrt(np.mean((prediction - dataset.y) ** 2))),
                    "late_utility_coef": model.late_utility_coef,
                    "late_score_scale": model.late_score_scale,
                    "late_score_coef": json.dumps(model.late_score_coef.tolist()),
                }
            )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write the cap-factor cross-swarm comparison."""
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
    figure = px.line(
        long,
        x="model",
        y="value",
        color="dataset",
        facet_col="metric",
        facet_col_wrap=2,
        markers=True,
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Support-calibrated late-utility bottleneck",
    )
    figure.update_layout(height=750)
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
    parser.add_argument("--cap-factors", default="inf,8,4,2,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    configs = [BottleneckConfig(cap) for cap in parse_cap_factors(args.cap_factors)]
    seeds = pooled.parse_int_list(args.seeds)
    metric_frames = []
    parameter_frames = []
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
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    full = full_fit_summary(
        datasets,
        names,
        configs,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    full.to_csv(args.output_dir / "full_fit_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Late-utility bottleneck benchmark",
        "",
        "The infinite-cap control pools late benefit linearly; finite caps impose support-relative diminishing returns.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Full fits",
        "",
        full.drop(columns=["late_score_coef"]).to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(full.drop(columns=["late_score_coef"]).to_string(index=False))
    print(f"Wrote late-utility bottleneck benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
