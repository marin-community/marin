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
"""Fit-panel stability audit for the closest nested mechanisms.

This audit never reads the sealed stress panel or historical heldout targets.
For each perturbation, one panel-stratified fold is withheld, every nonlinear
setting is refit on the remaining rows, and the setting is selected by the
withheld fit-panel RMSE. This measures identification rather than deployment
performance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_kish_collision_invariant as collision,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_nested_support_invariants as support,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_phase_boundary_adaptation as phase,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
DEFAULT_OUTPUT = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/closest_candidate_stability"
)
N_REPEATS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--link-metrics", type=Path, default=DEFAULT_LINK_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    return parser.parse_args()


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def selected_configs(
    dataset_id: base.DatasetId,
    source: pd.DataFrame,
    links: pd.DataFrame,
    repeats: int,
) -> list[dict[str, object]]:
    dataset = base.load_dataset(dataset_id)
    deficit_config = output_link.selected_deficit_config(dataset_id, collision.DEFICIT_VARIANT, source)
    link_config = support.selected_link_config(dataset_id, links)
    all_indices = np.arange(dataset.n)
    collision_configs = collision.configs()
    phase_configs: tuple[phase.Config | None, ...] = (
        None,
        *tuple(phase.Config(mechanism, smoothing) for mechanism in phase.Mechanism for smoothing in phase.SMOOTHING),
    )
    rows: list[dict[str, object]] = []
    for seed in range(repeats):
        train, test = base.split_indices(dataset, dataset_id, all_indices, seed)[seed % base.N_SPLITS]
        for family, candidates in (("collision", collision_configs), ("phase_boundary", phase_configs)):
            scored: list[tuple[float, str, float]] = []
            for config in candidates:
                if family == "collision":
                    model = collision.fit_model(dataset, deficit_config, link_config, config, train)
                    key = config.key
                else:
                    model = phase.fit_model(dataset, deficit_config, link_config, config, train)
                    key = "baseline" if config is None else config.key
                prediction = model.predict(dataset.weights[test])
                amplitude = 0.0 if key == "baseline" else float(model.coefficients[-1])
                scored.append((rmse(dataset.target[test], prediction), key, amplitude))
            score, key, amplitude = min(scored)
            baseline_score = next(value for value, candidate_key, _amplitude in scored if candidate_key == "baseline")
            rows.append(
                {
                    "dataset": dataset_id.value,
                    "family": family,
                    "seed": seed,
                    "config": key,
                    "held_fit_fold_rmse": score,
                    "baseline_rmse": baseline_score,
                    "relative_rmse": score / baseline_score,
                    "extra_amplitude": amplitude,
                    "extra_active": amplitude > 1e-10,
                }
            )
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, family), group in frame.groupby(["dataset", "family"], sort=False):
        counts = Counter(group["config"].astype(str))
        selected_nonbaseline = group.loc[group["config"].ne("baseline")]
        amplitudes = selected_nonbaseline["extra_amplitude"].to_numpy(dtype=float)
        median = float(np.median(amplitudes)) if len(amplitudes) else 0.0
        mad = float(np.median(np.abs(amplitudes - median))) if len(amplitudes) else 0.0
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "n_repeats": len(group),
                "modal_config": counts.most_common(1)[0][0],
                "modal_frequency": counts.most_common(1)[0][1] / len(group),
                "baseline_selection_frequency": counts["baseline"] / len(group),
                "unique_selected_configs": len(counts),
                "mean_relative_rmse": float(group["relative_rmse"].mean()),
                "extra_active_frequency": float(group["extra_active"].mean()),
                "selected_extra_median": median,
                "selected_extra_mad": mad,
                "mad_over_abs_median": mad / max(abs(median), 1e-12),
                "selection_counts": "; ".join(f"{key}:{count}" for key, count in counts.most_common()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.source_metrics)
    gate.assert_sealed_absent(args.link_metrics)
    source = pd.read_csv(args.source_metrics)
    links = pd.read_csv(args.link_metrics)
    rows: list[dict[str, object]] = []
    for dataset_id in (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9):
        rows.extend(selected_configs(dataset_id, source, links, args.repeats))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(rows)
    summary = summarize(records)
    records.to_csv(args.output_dir / "selection_records.csv", index=False)
    summary.to_csv(args.output_dir / "stability_summary.csv", index=False)
    (args.output_dir / "report.md").write_text(
        "# Closest-candidate fit-panel stability\n\n"
        "Each repeat selects a nonlinear setting on a fresh panel-stratified held-out fit fold. "
        "No deployment heldouts are used.\n\n" + summary.to_markdown(index=False, floatfmt=".6f") + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
