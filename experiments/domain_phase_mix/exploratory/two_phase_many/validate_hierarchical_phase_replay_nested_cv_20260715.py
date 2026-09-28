# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Nested-CV gate for the hierarchical phase/member-replay GRP candidate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_phase_replay_nested_cv_20260715"
DEFAULT_SEEDS = (7151, 7157)
VARIANTS = (
    benchmark.Variant.HIERARCHICAL,
    benchmark.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(dataset.value for dataset in benchmark.DatasetId),
        help="Comma-separated dataset IDs.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated outer-fold seeds.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=12)
    parser.add_argument("--top-shapes", type=int, default=3)
    return parser.parse_args()


def subset_oof_prediction(
    dataset: family_grp.Dataset,
    config: benchmark.Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
    indices: np.ndarray,
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = benchmark.fit_model(dataset, config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction[indices]).all():
        raise RuntimeError(f"Incomplete inner prediction for {config.variant}")
    return prediction[indices]


def best_config(
    dataset: family_grp.Dataset,
    dataset_id: benchmark.DatasetId,
    variant: benchmark.Variant,
    indices: np.ndarray,
    seed: int,
    num_shapes: int,
    top_shapes: int,
) -> tuple[benchmark.Config, dict[str, float | int]]:
    shapes = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    splits = benchmark.split_indices(dataset, dataset_id, indices, seed)
    baseline_scores: list[tuple[float, benchmark.Config]] = []
    best_by_shape: dict[int, float] = {}
    for config in benchmark.baseline_configs(shapes):
        prediction = subset_oof_prediction(dataset, config, splits, indices)
        metrics = benchmark.metric_summary(dataset.target[indices], prediction)
        score = float(metrics["rmse"])
        baseline_scores.append((score, config))
        best_by_shape[config.shape_index] = min(best_by_shape.get(config.shape_index, float("inf")), score)
    shape_indices = [index for index, _ in sorted(best_by_shape.items(), key=lambda item: item[1])[:top_shapes]]
    configs = (
        [config for _score, config in baseline_scores]
        if variant is benchmark.Variant.BUCKET_RESOLVED
        else benchmark.structural_configs(variant, shapes, shape_indices)
    )
    best: tuple[float, float, benchmark.Config, dict[str, float | int]] | None = None
    for config in configs:
        prediction = subset_oof_prediction(dataset, config, splits, indices)
        metrics = benchmark.metric_summary(dataset.target[indices], prediction)
        candidate = (float(metrics["rmse"]), -float(metrics["spearman"]), config, metrics)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No nested configuration for {dataset_id}/{variant}")
    return best[2], best[3]


def nested_oof(
    dataset_id: benchmark.DatasetId,
    variant: benchmark.Variant,
    seed: int,
    num_shapes: int,
    top_shapes: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    dataset = benchmark.load_dataset(dataset_id)
    all_indices = np.arange(dataset.n)
    outer_splits = benchmark.split_indices(dataset, dataset_id, all_indices, seed)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(outer_splits):
        print(f"{dataset_id.value}/{variant.value}/seed{seed}: fold {fold + 1}/{len(outer_splits)}", flush=True)
        config, inner_metrics = best_config(
            dataset,
            dataset_id,
            variant,
            train,
            seed + 1000 + fold,
            num_shapes,
            top_shapes,
        )
        model = benchmark.fit_model(dataset, config, train)
        prediction[test] = model.predict(dataset.weights[test])
        selections.append(
            {
                "dataset": dataset_id.value,
                "variant": variant.value,
                "seed": seed,
                "outer_fold": fold,
                "train_count": len(train),
                "test_count": len(test),
                "inner_rmse": inner_metrics["rmse"],
                "inner_spearman": inner_metrics["spearman"],
                "shape_index": config.shape_index,
                **asdict(config.shape),
                "l2": config.l2,
                "residual_shrink": config.residual_shrink,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested OOF prediction for {dataset_id}/{variant}/seed{seed}")
    return prediction, selections


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "variant",
        "rmse_mean",
        "rmse_std",
        "spearman_mean",
        "regret_at_1_mean",
        "lower_tail_optimism_mean",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[column]) if column in {"dataset", "variant"} else f"{float(row[column]):.6f}"
                for column in columns
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_ids = tuple(benchmark.DatasetId(value) for value in args.datasets.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for dataset_id in dataset_ids:
        dataset = benchmark.load_dataset(dataset_id)
        for variant in VARIANTS:
            for seed in seeds:
                prediction, selections = nested_oof(
                    dataset_id,
                    variant,
                    seed,
                    args.num_shapes,
                    args.top_shapes,
                )
                metrics = benchmark.metric_summary(dataset.target, prediction)
                metric_rows.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.value,
                        "seed": seed,
                        **metrics,
                    }
                )
                selection_rows.extend(selections)
                prediction_rows.extend(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.value,
                        "seed": seed,
                        "row_index": index,
                        "observed": observed,
                        "predicted": predicted,
                    }
                    for index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True))
                )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["dataset", "variant"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            spearman_mean=("spearman", "mean"),
            regret_at_1_mean=("regret_at_1", "mean"),
            lower_tail_optimism_mean=("lower_tail_optimism", "mean"),
            low_tail_rmse_mean=("low_tail_rmse", "mean"),
        )
        .fillna(0.0)
    )
    metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output_dir / "outer_fold_selections.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "datasets": [dataset.value for dataset in dataset_ids],
                "variants": [variant.value for variant in VARIANTS],
                "seeds": list(seeds),
                "protocol": (
                    "outer five-fold OOF; shape, ridge, and residual shrinkage selected only inside each outer "
                    "training fold"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "report.md").write_text(
        "# Hierarchical phase/member-replay GRP nested-CV gate\n\n"
        "All nonlinear shapes and linear shrinkage values are reselected inside each outer training fold. "
        "No 3e18 heldout result participates in this gate.\n\n" + markdown_table(summary) + "\n"
    )


if __name__ == "__main__":
    main()
