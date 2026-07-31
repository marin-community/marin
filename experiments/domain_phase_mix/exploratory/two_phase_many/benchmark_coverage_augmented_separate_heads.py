# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark whether coverage terms repair separate-heads phase bowls.

The separate-heads model fits independent phase-0 and phase-1 response bowls.
This benchmark adds the same three nonnegative global terms used by the
coverage-augmented effective-exposure DSP comparison:

    TV(w0, w1), HHI(alpha0 w0 + alpha1 w1), HHI(w1).

All exposure centers and linear heads are refit inside every fold. The goal is
to test whether coverage terms preserve separate-heads' phase flexibility while
repairing its poor transfer to the production swarm.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "coverage_augmented_separate_heads_20260709"
BASE_L2 = 0.1
COVERAGE_L2 = 0.01


@dataclass(frozen=True)
class SeparateCoverageModel:
    mu0: np.ndarray
    mu1: np.ndarray
    intercept: float
    coef0: np.ndarray
    coef1: np.ndarray
    coverage_coef: np.ndarray


def fit_head(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    *,
    use_coverage: bool,
    alpha0: float,
    alpha1: float,
) -> SeparateCoverageModel:
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    target = dataset.y[indices]
    mu0 = pooled.selected_mu(exposure0, target)
    mu1 = pooled.selected_mu(exposure1, target)
    design0 = pooled.bowl_design(exposure0, mu0)
    design1 = pooled.bowl_design(exposure1, mu1)
    pieces = [design0, design1]
    if use_coverage:
        pieces.append(coverage.coverage_features(dataset.weights[indices], alpha0, alpha1))
    design = np.hstack(pieces)
    center = design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    augmented_design = design - center
    augmented_target = target - target_mean
    phase_feature_count = design0.shape[1] + design1.shape[1]
    penalties = np.full(design.shape[1], np.sqrt(BASE_L2), dtype=float)
    penalties[phase_feature_count:] = np.sqrt(COVERAGE_L2)
    augmented_design = np.vstack([augmented_design, np.diag(penalties)])
    augmented_target = np.concatenate([augmented_target, np.zeros(design.shape[1])])
    coef, _residual = nnls(augmented_design, augmented_target, maxiter=20 * design.shape[1])
    intercept = target_mean - float((center @ coef).item())
    per_phase = design0.shape[1]
    return SeparateCoverageModel(
        mu0=mu0,
        mu1=mu1,
        intercept=intercept,
        coef0=np.asarray(coef[:per_phase], dtype=float),
        coef1=np.asarray(coef[per_phase : 2 * per_phase], dtype=float),
        coverage_coef=np.asarray(coef[2 * per_phase :], dtype=float),
    )


def predict(
    model: SeparateCoverageModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    exposure0, exposure1 = pooled.phase_exposures(dataset, indices)
    prediction = (
        model.intercept
        + pooled.bowl_design(exposure0, model.mu0) @ model.coef0
        + pooled.bowl_design(exposure1, model.mu1) @ model.coef1
    )
    if len(model.coverage_coef):
        prediction += coverage.coverage_features(dataset.weights[indices], alpha0, alpha1) @ model.coverage_coef
    return prediction


def benchmark_dataset(
    dataset: pooled.Dataset,
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage.assert_unique_mixtures(dataset)
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    metric_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        folds = pooled.dataset_folds(dataset, seed, n_splits)
        predictions = {
            "separate_heads": np.zeros(dataset.n, dtype=float),
            "separate_heads_coverage": np.zeros(dataset.n, dtype=float),
        }
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for name, use_coverage in (("separate_heads", False), ("separate_heads_coverage", True)):
                model = fit_head(
                    dataset,
                    train_idx,
                    use_coverage=use_coverage,
                    alpha0=alpha0,
                    alpha1=alpha1,
                )
                predictions[name][test_idx] = predict(model, dataset, test_idx, alpha0, alpha1)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": name,
                        "seed": seed,
                        "fold": fold_id,
                        "theta_tv": float(model.coverage_coef[0]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_aggregate": float(model.coverage_coef[1]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_phase1": float(model.coverage_coef[2]) if len(model.coverage_coef) else 0.0,
                    }
                )
        for name, prediction in predictions.items():
            row = pooled.metrics(dataset, name, seed, prediction, folds).__dict__.copy()
            row["nominal_param_count"] = 4 * dataset.m + 3 + 3 * int(name.endswith("_coverage"))
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "300m_uncheatable": lambda: pooled.load_300m_dataset("uncheatable"),
        "300m_table9": lambda: pooled.load_300m_dataset("table9"),
        "production_uncheatable": pooled.load_production_dataset,
    }
    selected = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected).difference(loaders))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    metric_frames = []
    parameter_frames = []
    for name in selected:
        metrics, parameters = benchmark_dataset(loaders[name](), pooled.parse_int_list(args.seeds), args.n_splits)
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(
        parameters.groupby(["dataset", "model"])[["theta_tv", "theta_hhi_aggregate", "theta_hhi_phase1"]]
        .agg(["mean", "std"])
        .to_string()
    )
    print(f"Wrote benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
