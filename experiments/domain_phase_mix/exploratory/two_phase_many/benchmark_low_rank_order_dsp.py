# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark a low-rank phase-order residual on a phase-invariant DSP backbone.

The aggregate DSP is fit on tied schedules. Phase ordering is represented by
the leading uncentered singular vectors of the bounded late/early contrast
observed in the training designs. A nonnegative TV cost plus ``rank`` signed
latent coefficients are fit to matched two-phase-minus-tied outcomes.

The representation is learned without benchmark labels and is refit inside
every fold. It therefore tests whether phase ordering is low-dimensional
without paying for independent per-bucket phase response curves.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_order_dsp as aggregate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_matched_phase_ordering as ordering,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "low_rank_order_dsp_20260710"
SCALE_FLOOR = 1e-10


@dataclass(frozen=True)
class Config:
    name: str
    rank: int
    l2: float = 0.1


@dataclass(frozen=True)
class Model:
    aggregate_model: aggregate.AggregateOrderModel
    config: Config
    basis: np.ndarray
    feature_scale: np.ndarray
    coef: np.ndarray


def configs() -> tuple[Config, ...]:
    return tuple(Config(f"low_rank_order_r{rank}", rank) for rank in (2, 4, 8, 16))


def recency_contrast(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    weights = dataset.weights[indices]
    e0 = weights[:, 0, :] * dataset.c0[None, :]
    e1 = weights[:, 1, :] * dataset.c1[None, :]
    return np.tanh(np.log1p(e1 / alpha1) - np.log1p(e0 / alpha0))


def order_design(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    alpha0, alpha1 = aggregate.geometry.phase_fractions(dataset)
    contrast = recency_contrast(dataset, indices, alpha0, alpha1)
    weights = dataset.weights[indices]
    tv = 0.5 * np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1, keepdims=True)
    return np.hstack([tv, contrast @ basis.T])


def fit_head(
    design: np.ndarray,
    target: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), SCALE_FLOOR)
    normalized = design / scale[None, :]
    augmented_design = np.vstack([normalized, np.sqrt(l2) * np.eye(design.shape[1])])
    augmented_target = np.concatenate([target, np.zeros(design.shape[1], dtype=float)])
    lower = np.concatenate([[0.0], np.full(design.shape[1] - 1, -np.inf)])
    upper = np.full(design.shape[1], np.inf)
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=500,
    )
    if not result.success:
        raise RuntimeError(f"Low-rank order fit failed: {result.message}")
    return scale, np.asarray(result.x, dtype=float)


def fit_model(
    dataset: pooled.Dataset,
    train_indices: np.ndarray,
    config: Config,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> Model:
    aggregate_model = aggregate.fit_model(
        dataset,
        train_indices,
        aggregate.configs()[0],
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
    )
    has_pairs = "policy_family" in dataset.frame.columns
    if has_pairs:
        order_indices, target = aggregate.matched_order_training_rows(dataset, train_indices)
    else:
        order_indices = train_indices
        target = dataset.y[order_indices] - aggregate.predict(aggregate_model, dataset, order_indices)
    contrast = recency_contrast(
        dataset,
        order_indices,
        aggregate_model.alpha0,
        aggregate_model.alpha1,
    )
    _left, _singular, right = np.linalg.svd(contrast, full_matrices=False)
    rank = min(config.rank, right.shape[0])
    basis = np.asarray(right[:rank], dtype=float)
    design = order_design(dataset, order_indices, basis)
    feature_scale, coef = fit_head(design, target, config.l2)
    return Model(
        aggregate_model=aggregate_model,
        config=config,
        basis=basis,
        feature_scale=feature_scale,
        coef=coef,
    )


def predict(model: Model, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    base = aggregate.predict(model.aggregate_model, dataset, indices)
    design = order_design(dataset, indices, model.basis)
    return base + (design / model.feature_scale[None, :]) @ model.coef


def benchmark_dataset(
    dataset: pooled.Dataset,
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pair_rows = []
    parameter_rows = []
    for seed in seeds:
        folds = aggregate.folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in configs()}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in configs():
                model = fit_model(
                    dataset,
                    train_idx,
                    config,
                    maxiter=maxiter,
                    coarse_top_k=coarse_top_k,
                )
                predictions[config.name][test_idx] = predict(model, dataset, test_idx)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "rank": model.basis.shape[0],
                        "theta_tv": float(model.coef[0]),
                        "latent_coef_l2": float(np.linalg.norm(model.coef[1:])),
                    }
                )
        for config in configs():
            prediction = predictions[config.name]
            row = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            row["nominal_param_count"] = 4 * dataset.m + 3 + config.rank
            metric_rows.append(row)
            if dataset.name in aggregate.NOISE_SD:
                pairs = ordering.pair_frame(dataset, prediction, config.name)
                pair_metric = ordering.pair_metrics(pairs, aggregate.NOISE_SD[dataset.name])
                pair_metric["seed"] = seed
                pair_rows.append(pair_metric)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(parameter_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter-300m", type=int, default=4)
    parser.add_argument("--maxiter-production", type=int, default=0)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    dataset_by_name = {}
    for objective, target in joint.TARGET_COLUMNS.items():
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        dataset_by_name[dataset.name] = dataset
    production = pooled.load_production_dataset()
    dataset_by_name[production.name] = production
    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected_names).difference(dataset_by_name))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    datasets = [dataset_by_name[name] for name in selected_names]

    metric_frames = []
    pair_frames = []
    parameter_frames = []
    for dataset in datasets:
        maxiter = args.maxiter_production if dataset.name == "production_uncheatable" else args.maxiter_300m
        metrics, pairs, parameters = benchmark_dataset(
            dataset,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        if not pairs.empty:
            pair_frames.append(pairs)
        parameter_frames.append(parameters)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    pairs = pd.concat(pair_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    pairs.to_csv(args.output_dir / "matched_pair_delta_metrics.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    print(summary.to_string(index=False))
    print(pairs.groupby(["dataset", "model"])[["delta_rmse", "delta_spearman", "sign_accuracy"]].mean().to_string())
    print(f"Wrote low-rank order benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
