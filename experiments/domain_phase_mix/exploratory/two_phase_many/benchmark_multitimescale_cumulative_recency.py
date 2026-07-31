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
"""Add a fast/slow learning dictionary to cumulative-recency DSP.

The incumbent shared-power model assigns one saturating learning response to
each bucket in each of the durable and late-state channels. A benchmark can
contain both fast and slow subskills, so one timescale can force a compromise
between the initial drop and the broad shoulder of a Nike-swoosh response.

This benchmark freezes the incumbent nonlinear geometry inside every fold and
adds a second log-symmetric response rate with nonnegative amplitudes:

    S_slow(e; rho / sqrt(r)), S_fast(e; rho * sqrt(r)).

The ratio ``r`` is a small, fixed dictionary hyperparameter. ``r=1`` recovers
the incumbent function span exactly. No bucket-specific nonlinear parameter is
added; ridge regularizes the two amplitudes per bucket.
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "multitimescale_cumulative_recency_20260710"
SHARED_POWER_CONFIG = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, 0.001)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Config:
    """Dictionary spacing and head-ridge multiplier."""

    rate_ratio: float
    head_l2_multiplier: float

    @property
    def name(self) -> str:
        return f"two_timescale_ratio_{self.rate_ratio:g}_l2x_{self.head_l2_multiplier:g}"


@dataclass(frozen=True)
class FittedModel:
    """Frozen nonlinear geometry with a two-timescale nonnegative head."""

    config: Config
    base: generalized.FittedModel
    intercept: float
    coef: np.ndarray

    @property
    def parameter_count(self) -> int:
        return 6 * self.base.num_domains + 7

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = design_matrix(self.base, weights, self.config.rate_ratio)
        return np.asarray(self.intercept + design @ self.coef, dtype=float)


def benefit_features(
    exposure: np.ndarray,
    rho: np.ndarray,
    power: float,
    rate_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-symmetric slow and fast saturating responses."""
    scale = np.sqrt(rate_ratio)
    slow = generalized.channel_features(
        exposure,
        rho / scale,
        np.full_like(rho, 100.0),
        power,
        2.0,
    )[0]
    fast = generalized.channel_features(
        exposure,
        rho * scale,
        np.full_like(rho, 100.0),
        power,
        2.0,
    )[0]
    return slow, fast


def design_matrix(
    base: generalized.FittedModel,
    weights: np.ndarray,
    rate_ratio: float,
) -> np.ndarray:
    """Build two learning timescales and one harm response per channel."""
    cumulative, late = recency.exposures(weights, base.c0, base.c1)
    cumulative_parameters, late_parameters, efficiency_offsets = generalized.channel_parameters(
        generalized.CalibrationKind.SHARED_POWER,
        base.shape_parameters,
        base.num_domains,
    )
    cumulative_rho, cumulative_tau = generalized.calibrated_shape(
        base.cumulative_base,
        *cumulative_parameters,
        efficiency_offsets,
    )
    late_rho, late_tau = generalized.calibrated_shape(
        base.recency_base,
        *late_parameters,
        efficiency_offsets,
    )
    cumulative_powers, late_powers = generalized.response_powers(
        generalized.CalibrationKind.SHARED_POWER,
        base.shape_parameters,
    )
    cumulative_slow, cumulative_fast = benefit_features(
        cumulative,
        cumulative_rho,
        cumulative_powers[0],
        rate_ratio,
    )
    late_slow, late_fast = benefit_features(
        late,
        late_rho,
        late_powers[0],
        rate_ratio,
    )
    cumulative_penalty = generalized.channel_features(
        cumulative,
        cumulative_rho,
        cumulative_tau,
        cumulative_powers[0],
        cumulative_powers[1],
    )[1]
    late_penalty = generalized.channel_features(
        late,
        late_rho,
        late_tau,
        late_powers[0],
        late_powers[1],
    )[1]
    return np.hstack(
        [
            -cumulative_slow,
            -cumulative_fast,
            cumulative_penalty,
            -late_slow,
            -late_fast,
            late_penalty,
        ]
    )


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
    base: generalized.FittedModel,
) -> FittedModel:
    """Fit only the expanded nonnegative head."""
    design = design_matrix(base, dataset.weights[indices], config.rate_ratio)
    head_l2 = base.head_l2 * config.head_l2_multiplier
    intercept, coef = cross.fit_head(design, dataset.y[indices], head_l2)
    return FittedModel(config=config, base=base, intercept=intercept, coef=coef)


def fit_base(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> generalized.FittedModel:
    """Fit the incumbent shared-power geometry once per fold."""
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
    labels = ["shared_power_one_timescale", *[config.name for config in configs]]
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = cross.folds_for(dataset, seed, n_splits)
        predictions = {label: np.zeros(dataset.n, dtype=float) for label in labels}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            base = fit_base(
                dataset,
                train_indices,
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            predictions["shared_power_one_timescale"][test_indices] = base.predict(dataset.weights[test_indices])
            for config in configs:
                model = fit_model(dataset, train_indices, config, base)
                predictions[config.name][test_indices] = model.predict(dataset.weights[test_indices])
                m = base.num_domains
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "rate_ratio": config.rate_ratio,
                        "head_l2": base.head_l2 * config.head_l2_multiplier,
                        "active_cumulative_slow": int(np.sum(model.coef[:m] > 1e-10)),
                        "active_cumulative_fast": int(np.sum(model.coef[m : 2 * m] > 1e-10)),
                        "active_late_slow": int(np.sum(model.coef[3 * m : 4 * m] > 1e-10)),
                        "active_late_fast": int(np.sum(model.coef[4 * m : 5 * m] > 1e-10)),
                        "shape_parameters": json.dumps(base.shape_parameters.tolist()),
                    }
                )
        for label in labels:
            row = asdict(pooled.metrics(dataset, label, seed, predictions[label], folds))
            row["nominal_param_count"] = 4 * dataset.m + 7 if label == labels[0] else 6 * dataset.m + 7
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
    base = fit_base(dataset, np.arange(dataset.n), maxiter=maxiter, coarse_top_k=coarse_top_k)
    models = [fit_model(dataset, np.arange(dataset.n), config, base) for config in configs]
    rows = []
    for label, prediction in [
        ("shared_power_one_timescale", base.predict(dataset.weights[mask])),
        *[(model.config.name, model.predict(dataset.weights[mask])) for model in models],
    ]:
        targets = dataset.y[mask]
        phase1 = dataset.frame.loc[mask, "phase_1_starcoder"].to_numpy(dtype=float)
        minimum = int(np.argmin(prediction))
        rows.append(
            {
                "model": label,
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
        base = fit_base(dataset, np.arange(dataset.n), maxiter=maxiter, coarse_top_k=coarse_top_k)
        predictions = [("shared_power_one_timescale", base.predict(external_dataset.weights))]
        predictions.extend(
            (model.config.name, model.predict(external_dataset.weights))
            for model in [fit_model(dataset, np.arange(dataset.n), config, base) for config in configs]
        )
        for label, prediction in predictions:
            row = joint.external_metrics(label, external_dataset.y, prediction)
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
    parser.add_argument("--rate-ratios", default="2,4,8")
    parser.add_argument("--head-l2-multipliers", default="1,10")
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
        Config(ratio, multiplier)
        for ratio in pooled.parse_float_list(args.rate_ratios)
        for multiplier in pooled.parse_float_list(args.head_l2_multipliers)
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
        title="Fast/slow cumulative-recency grouped-CV comparison",
    )
    figure.write_html(
        args.output_dir / "cv_comparison.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    report = [
        "# Fast/slow cumulative-recency dictionary",
        "",
        "Two fixed log-spaced learning responses approximate fast and slow subskills while retaining nonnegative heads.",
        "",
        "## Grouped-CV metrics",
        "",
        summary.to_markdown(index=False),
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
    if not slices.empty:
        print(slices.to_string(index=False))
    print(f"Wrote multitimescale benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
