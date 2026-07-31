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
"""Cap unsupported additive late benefit without refitting its amplitude.

The refitted late-utility bottleneck can compensate for stronger curvature by
increasing its scalar coefficient. This anchored variant has no compensation
path. It withdraws only the amount by which the additive late-benefit score
exceeds a support-scaled concave utility, then recenters that correction on the
training fold so mean calibration is unchanged.
"""

from __future__ import annotations

import argparse
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
    benchmark_late_utility_bottleneck as bottleneck,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "anchored_late_utility_cap_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FittedAnchoredCap:
    """Additive shared-power model plus a centered fixed late-benefit cap."""

    base: crowding.generalized.FittedModel
    config: bottleneck.BottleneckConfig
    late_score_coef: np.ndarray
    late_score_scale: float
    correction_center: float

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = crowding.base_design(self.base, weights)
        m = self.base.num_domains
        score = (-design[:, 2 * m : 3 * m]) @ self.late_score_coef
        correction = cap_correction(score, self.late_score_scale, self.config.cap_factor)
        return np.asarray(self.base.predict(weights) + correction - self.correction_center, dtype=float)


def cap_correction(score: np.ndarray, scale: float, cap_factor: float) -> np.ndarray:
    """Return additive benefit withdrawn by a concave utility cap."""
    if np.isinf(cap_factor):
        return np.zeros_like(score, dtype=float)
    utility = scale * bottleneck.late_utility(score, scale, cap_factor)
    return np.maximum(score - utility, 0.0)


def fit_from_base(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: bottleneck.BottleneckConfig,
    base: crowding.generalized.FittedModel,
) -> FittedAnchoredCap:
    """Calibrate only the support scale and train-fold correction center."""
    design = crowding.base_design(base, dataset.weights[indices])
    m = dataset.m
    late_score_coef = np.asarray(base.coef[2 * m : 3 * m], dtype=float)
    score = (-design[:, 2 * m : 3 * m]) @ late_score_coef
    scale = max(float(np.quantile(score, crowding.SCALE_QUANTILE)), crowding.SCALE_FLOOR)
    center = float(np.mean(cap_correction(score, scale, config.cap_factor)))
    return FittedAnchoredCap(
        base=base,
        config=config,
        late_score_coef=late_score_coef,
        late_score_scale=scale,
        correction_center=center,
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[bottleneck.BottleneckConfig],
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
                        "late_score_scale": model.late_score_scale,
                        "correction_center": model.correction_center,
                    }
                )
        for config in configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = model.parameter_count
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def full_fit_summary(
    datasets: dict[str, pooled.Dataset],
    names: list[str],
    configs: list[bottleneck.BottleneckConfig],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Record full-data calibration and incumbent-independent fit metrics."""
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
                    "late_score_scale": model.late_score_scale,
                    "correction_center": model.correction_center,
                    "max_train_correction": float(
                        np.max(
                            cap_correction(
                                (-crowding.base_design(base, dataset.weights)[:, 2 * dataset.m : 3 * dataset.m])
                                @ model.late_score_coef,
                                model.late_score_scale,
                                config.cap_factor,
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write the support-cap cross-swarm comparison."""
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
        title="Anchored late-utility cap",
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
    parser.add_argument("--cap-factors", default="inf,16,8,4,2,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [value.strip() for value in args.datasets.split(",") if value.strip()]
    configs = [bottleneck.BottleneckConfig(cap) for cap in bottleneck.parse_cap_factors(args.cap_factors)]
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
        "# Anchored late-utility cap benchmark",
        "",
        "The correction has no fitted amplitude and is centered on each train fold.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Full fits",
        "",
        full.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(full.to_string(index=False))
    print(f"Wrote anchored late-utility benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
