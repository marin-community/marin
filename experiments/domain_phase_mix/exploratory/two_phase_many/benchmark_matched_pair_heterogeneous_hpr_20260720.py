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
#   "tabulate",
# ]
# ///
"""Compare fixed-cost matched-pair plus phase-fiber HPR designs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_heterogeneous_design_aware_hpr_20260719 as fitting,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/matched_pair_heterogeneous_hpr_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
TOTAL_ROWS = 280


@dataclass(frozen=True)
class PairAllocation:
    name: str
    tied: int
    pairs: int
    fiber: int
    fiber_mode: str

    def __post_init__(self) -> None:
        if self.tied + 2 * self.pairs + self.fiber != TOTAL_ROWS:
            raise ValueError(f"{self.name} does not use exactly {TOTAL_ROWS} checkpoints")


ALLOCATIONS = (
    PairAllocation("t42_p119", 42, 119, 0, "both"),
    PairAllocation("p140", 0, 140, 0, "both"),
    PairAllocation("p100_f80_matched", 0, 100, 80, "target_matched"),
    PairAllocation("p90_f100_matched", 0, 90, 100, "target_matched"),
    PairAllocation("t40_p70_f100_matched", 40, 70, 100, "target_matched"),
    PairAllocation("p70_f140_both", 0, 70, 140, "both"),
    PairAllocation("p50_f180_both", 0, 50, 180, "both"),
)


@dataclass(frozen=True)
class MatchedSources:
    sources: composition.Sources
    pair_frame: pd.DataFrame
    tied_broad_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    return parser.parse_args()


def matched_sources() -> MatchedSources:
    sources = composition.load_sources()
    alpha = float(np.mean(sources.reference.c0 / (sources.reference.c0 + sources.reference.c1)))
    aggregate = alpha * sources.broad.weights[:, 0, :] + (1.0 - alpha) * sources.broad.weights[:, 1, :]
    single = sources.single.weights[:, 0, :]
    distance = np.max(np.abs(aggregate[:, None, :] - single[None, :, :]), axis=2)
    nearest = np.argmin(distance, axis=1)
    minimum = np.min(distance, axis=1)
    matched = minimum < 1e-10
    tied = np.max(np.abs(sources.broad.weights[:, 0, :] - sources.broad.weights[:, 1, :]), axis=1) < 1e-12
    if int(matched.sum()) != 238 or len(set(nearest[matched].tolist())) != 238:
        raise ValueError("Expected 238 bijective aggregate matches")
    if int(tied.sum()) != 42 or np.any(tied & matched) or np.any(~tied & ~matched):
        raise ValueError("Expected the remaining 42 broad policies to be phase tied")

    broad_indices = np.flatnonzero(matched)
    single_indices = nearest[matched]
    broad_seed = sources.broad.frame.iloc[broad_indices]["data_seed"].to_numpy(dtype=int)
    single_seed = sources.single.frame.iloc[single_indices]["data_seed"].to_numpy(dtype=int)
    if not np.array_equal(broad_seed, single_seed):
        raise ValueError("Matched one- and two-phase policies do not share data seeds")

    pair_frame = sources.broad.frame.iloc[broad_indices].copy().reset_index(drop=True)
    pair_frame["broad_index"] = broad_indices
    pair_frame["single_index"] = single_indices
    pair_frame["pair_id"] = pair_frame["run_name"].astype(str)
    return MatchedSources(sources, pair_frame, np.flatnonzero(tied))


def sampled_rows(
    matched: MatchedSources,
    allocation: PairAllocation,
    target: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed + 10_000 * fitting.TARGETS.index(target))
    pair_positions = composition.stratified_indices(matched.pair_frame, allocation.pairs, "panel_source", rng)
    pair_records = matched.pair_frame.iloc[pair_positions]
    broad_indices = pair_records["broad_index"].to_numpy(dtype=int)
    single_indices = pair_records["single_index"].to_numpy(dtype=int)

    tied_frame = matched.sources.broad.frame.iloc[matched.tied_broad_indices].copy()
    tied_positions = composition.stratified_indices(tied_frame, allocation.tied, "panel_source", rng)
    tied_indices = matched.tied_broad_indices[tied_positions]
    selected_fiber = composition.fiber_indices(
        matched.sources.fiber.frame,
        allocation.fiber,
        allocation.fiber_mode,
        target,
        rng,
    )

    phase_frame = matched.sources.broad.frame.iloc[broad_indices].copy()
    aggregate_frame = matched.sources.single.frame.iloc[single_indices].copy()
    pair_ids = pair_records["pair_id"].to_numpy(dtype=str)
    phase_frame["source_pool"] = "matched_pair"
    phase_frame["pair_role"] = "phase"
    phase_frame["pair_id"] = pair_ids
    aggregate_frame["source_pool"] = "matched_pair"
    aggregate_frame["pair_role"] = "aggregate"
    aggregate_frame["pair_id"] = pair_ids
    phase_frame["matched_delta"] = phase_frame[fitting.TARGET_COLUMNS[target]].to_numpy(dtype=float) - aggregate_frame[
        fitting.TARGET_COLUMNS[target]
    ].to_numpy(dtype=float)
    aggregate_frame["matched_delta"] = np.nan

    tied_selected = matched.sources.broad.frame.iloc[tied_indices].copy()
    tied_selected["source_pool"] = "tied_control"
    tied_selected["pair_role"] = ""
    tied_selected["pair_id"] = ""
    tied_selected["matched_delta"] = np.nan
    fiber_frame = matched.sources.fiber.frame.iloc[selected_fiber].copy()
    fiber_frame["pair_role"] = ""
    fiber_frame["pair_id"] = ""
    fiber_frame["matched_delta"] = np.nan

    frame = pd.concat([tied_selected, phase_frame, aggregate_frame, fiber_frame], ignore_index=True, sort=False)
    weights = np.concatenate(
        [
            matched.sources.broad.weights[tied_indices],
            matched.sources.broad.weights[broad_indices],
            matched.sources.single.weights[single_indices],
            matched.sources.fiber.weights[selected_fiber],
        ],
        axis=0,
    )
    if len(frame) != TOTAL_ROWS or len(weights) != TOTAL_ROWS:
        raise AssertionError("Matched allocation did not preserve the checkpoint budget")
    return frame, weights


def source_holdout_metrics(
    model: fitting.StructuredModel,
    matched: MatchedSources,
    selected: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:
    selected_ids = set(selected["coordinate_hash"].astype(str))
    rows: list[dict[str, Any]] = []
    for source_name, pool in (
        ("unused_single", matched.sources.single),
        ("unused_fiber", matched.sources.fiber),
    ):
        keep = ~pool.frame["coordinate_hash"].astype(str).isin(selected_ids).to_numpy()
        if np.sum(keep) < 3:
            continue
        observed = pool.frame.loc[keep, fitting.TARGET_COLUMNS[target]].to_numpy(dtype=float)
        predicted = model.predict(pool.weights[keep])
        rows.append({"scope": source_name, **composition.prediction_metrics(observed, predicted)})
    return rows


def write_report(metrics: pd.DataFrame, coupling: pd.DataFrame, output_dir: Path) -> None:
    common = metrics.loc[metrics["scope"].eq("common_all")]
    summary = (
        common.groupby(["target", "allocation", "estimator"], sort=True)
        .agg(
            replicates=("seed", "size"),
            rmse=("rmse", "mean"),
            spearman=("spearman", "mean"),
            calibration_slope=("calibration_slope", "mean"),
            regret_at_1=("regret_at_1", "mean"),
            optimism_gt_0p05=("optimism_gt_0p05", "mean"),
            worst_optimism=("worst_optimism", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "common_archive_summary.csv", index=False)
    selected = coupling.loc[coupling["selected"]].copy()
    lines = [
        "# Matched-pair heterogeneous HPR",
        "",
        "All designs use exactly 280 trained checkpoints. A complete pair contributes one independently trained "
        "phase-tied aggregate policy and its aggregate-matched two-phase policy with the same data seed. The "
        "aggregate member is an absolute-level equation; their difference is an exact phase-contrast equation. "
        "Frontier fibers use the same shared-center GLS treatment as the preregistered first batch.",
        "",
        "## Common archive",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Training-only coupling choices",
        "",
        selected[["target", "allocation", "seed", "coupling", "rmse", "delta_rmse"]].to_markdown(
            index=False, floatfmt=".6f"
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == DEFAULT_OUTPUT_DIR and not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing frozen preregistration {PREREGISTRATION_PATH}")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    matched = matched_sources()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    for target in fitting.TARGETS:
        config = composition.hpr_config(target)
        common_observed = matched.sources.common.frame[fitting.TARGET_COLUMNS[target]].to_numpy(dtype=float)
        for allocation in ALLOCATIONS:
            for seed in seeds:
                print(f"Fitting {target}/{allocation.name}/seed={seed}", flush=True)
                frame, weights = sampled_rows(matched, allocation, target, seed)
                dataset = composition.custom_dataset(
                    matched.sources.reference,
                    frame,
                    weights,
                    target,
                    f"matched_hpr_{target}_{allocation.name}_{seed}",
                )
                selected_coupling, partial_oof, grid = fitting.select_coupling(dataset, frame, config, target)
                for row in grid:
                    coupling_rows.append(
                        {
                            "target": target,
                            "allocation": allocation.name,
                            "seed": seed,
                            "selected": float(row["coupling"]) == selected_coupling,
                            **row,
                        }
                    )
                oof_by_estimator = {
                    fitting.Estimator.POOLED_LEVELS: fitting.oof_candidate(
                        dataset, frame, config, target, fitting.Estimator.POOLED_LEVELS, math.inf
                    ),
                    fitting.Estimator.SHARED_ORTHOGONAL_MOMENTS: fitting.oof_candidate(
                        dataset,
                        frame,
                        config,
                        target,
                        fitting.Estimator.SHARED_ORTHOGONAL_MOMENTS,
                        math.inf,
                    ),
                    fitting.Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS: partial_oof,
                }
                for estimator, oof in oof_by_estimator.items():
                    coupling_value = (
                        selected_coupling
                        if estimator is fitting.Estimator.PARTIAL_PHASE_ORTHOGONAL_MOMENTS
                        else math.inf
                    )
                    full = fitting.fit_candidate(
                        dataset,
                        frame,
                        config,
                        np.arange(dataset.n),
                        target,
                        estimator,
                        coupling_value,
                    )
                    base = {
                        "target": target,
                        "allocation": allocation.name,
                        "seed": seed,
                        "estimator": estimator.value,
                        "coupling": coupling_value,
                    }
                    metric_rows.append(
                        {**base, "scope": "train_oof", **composition.prediction_metrics(dataset.target, oof.prediction)}
                    )
                    delta_mask = oof.phase_delta_mask
                    metric_rows.append(
                        {
                            **base,
                            "scope": "train_phase_delta_oof",
                            **composition.prediction_metrics(
                                oof.phase_delta_observed[delta_mask], oof.phase_delta_prediction[delta_mask]
                            ),
                        }
                    )
                    common_prediction = full.predict(matched.sources.common.weights)
                    fitting.append_metrics(
                        metric_rows,
                        base,
                        matched.sources.common.frame,
                        common_observed,
                        common_prediction,
                        target,
                    )
                    for row in source_holdout_metrics(full, matched, frame, target):
                        metric_rows.append({**base, **row})
                    for index, (observed, predicted) in enumerate(zip(common_observed, common_prediction, strict=True)):
                        prediction_rows.append(
                            {
                                **base,
                                "row_id": matched.sources.common.frame.iloc[index]["row_id"],
                                "training_series": matched.sources.common.frame.iloc[index]["training_series"],
                                "policy_class": matched.sources.common.frame.iloc[index]["policy_class"],
                                "objective": matched.sources.common.frame.iloc[index]["objective"],
                                "observed": observed,
                                "predicted": predicted,
                                "residual": predicted - observed,
                            }
                        )
                    for block, values in (
                        ("aggregate", oof.aggregate_coefficients),
                        ("phase", oof.phase_coefficients),
                    ):
                        stability_rows.append(
                            {**base, "coefficient_block": block, **fitting.coefficient_stability(values)}
                        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    coupling = pd.DataFrame(coupling_rows)
    stability = pd.DataFrame(stability_rows)
    metrics.to_csv(output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(output_dir / "common_archive_predictions.csv", index=False)
    coupling.to_csv(output_dir / "coupling_selection.csv", index=False)
    stability.to_csv(output_dir / "coefficient_stability.csv", index=False)
    fitting.render(metrics, predictions, output_dir)
    write_report(metrics, coupling, output_dir)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "total_rows": TOTAL_ROWS,
                "allocations": [allocation.__dict__ for allocation in ALLOCATIONS],
                "seeds": seeds,
                "matched_pair_count": len(matched.pair_frame),
                "tied_broad_count": len(matched.tied_broad_indices),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
