# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
# ]
# ///
"""Freeze the outcome-blind candidate pool and coverage tranche for Delphi Wave 2.

Run from the repository root with::

    PYTHONPATH=. uv run \
      experiments/domain_phase_mix/exploratory/two_phase_many/\
design_delphi_phase1_kl0p05_wave2_pool_20260825.py

The raw 50,000-row pool exceeds the repository's per-file size limit and is
therefore regenerated rather than committed. The checked-in manifest and the
Wave-2 selector pin the exact hashes of every generated file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave2_pool_20260825"
ORIGINAL_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase1_common_branches_20260824" / "continuation_weights.csv"
EXTENSION_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave1_extension_20260825" / "continuation_weights.csv"
EXPECTED_ORIGINAL_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
EXPECTED_EXTENSION_SHA256 = "2860d0e1f177f1728580ec1cdda05e049734e7977b868a8c0abd05d9d8bd0ec3"
EXPECTED_WAVE1_FIT_ROWS = 100
POOL_DESIGN_SEED = 20_260_826
DIRICHLET_DRAWS_PER_CONCENTRATION = 100_000
POOL_ROWS = 50_000
NEAR_COVERAGE_ROWS = 24
GLOBAL_COVERAGE_ROWS = 16
REFEREE_ROWS = 8
REFEREE_ROWS_PER_TRANCHE = REFEREE_ROWS // 2
NEAR_TV_MAX = 0.25
NEAR_PHASE_1_EPOCH_MAX = 15.0
TARGET_PREFIX = "shared_bounded_ensemble_kl0p05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix-weights", type=Path, default=base.DEFAULT_PREFIX_WEIGHTS)
    parser.add_argument("--original-weights", type=Path, default=ORIGINAL_WEIGHTS)
    parser.add_argument("--extension-weights", type=Path, default=EXTENSION_WEIGHTS)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wave1_fit_weights(
    paths_and_hashes: tuple[tuple[Path, str], ...], buckets: tuple[str, ...]
) -> tuple[tuple[str, ...], np.ndarray]:
    identities = []
    rows = []
    for path, expected_sha256 in paths_and_hashes:
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Wave 1 continuation weights changed: {actual_sha256} != {expected_sha256}")
        frame = pd.read_csv(path)
        fit = frame[frame.fit_budget.astype(str).str.lower().eq("true")]
        for continuation_id, group in fit.groupby("continuation_id", sort=False):
            if tuple(group.bucket) != buckets:
                raise ValueError(f"Wave 1 bucket order changed for {continuation_id}")
            weights = group.phase_1_weight.to_numpy(dtype=float)
            if not np.array_equal(base.runtime_counts(weights), group.phase_1_count.to_numpy(dtype=np.int64)):
                raise ValueError(f"Wave 1 continuation {continuation_id} is not runtime exact")
            identities.append(str(continuation_id))
            rows.append(weights)
    if len(rows) != EXPECTED_WAVE1_FIT_ROWS or len({tuple(base.runtime_counts(row)) for row in rows}) != len(rows):
        raise ValueError("Combined Wave 1 fit design changed")
    return tuple(identities), np.stack(rows)


def deterministic_pool_subset(
    pool: np.ndarray,
    sources: list[str],
    concentrations: list[float | None],
) -> tuple[np.ndarray, list[str], list[float | None]]:
    if len(pool) < POOL_ROWS:
        raise ValueError(f"Only {len(pool)} candidates survived; need {POOL_ROWS}")
    generator = np.random.default_rng(POOL_DESIGN_SEED + 1)
    historical = np.flatnonzero(np.asarray([value is None for value in concentrations]))
    if len(historical) >= POOL_ROWS:
        raise ValueError("Historical rows unexpectedly fill the candidate pool")
    selected = list(historical)
    remaining = POOL_ROWS - len(selected)
    concentration_values = tuple(base.DIRICHLET_CONCENTRATIONS)
    capacities = np.asarray(
        [sum(value == concentration for value in concentrations) for concentration in concentration_values],
        dtype=int,
    )
    quotas = np.zeros(len(concentration_values), dtype=int)
    while remaining:
        active = np.flatnonzero(quotas < capacities)
        if not len(active):
            raise ValueError("Candidate-pool strata cannot fill the requested subset")
        share, extra = divmod(remaining, len(active))
        requested = np.full(len(active), share, dtype=int)
        requested[:extra] += 1
        allocated = np.minimum(requested, capacities[active] - quotas[active])
        quotas[active] += allocated
        remaining -= int(allocated.sum())
    for concentration, quota in zip(concentration_values, quotas, strict=True):
        group = np.flatnonzero(np.asarray([value == concentration for value in concentrations]))
        selected.extend(int(value) for value in generator.choice(group, size=int(quota), replace=False))
    selected_array = np.asarray(selected, dtype=int)
    if len(selected_array) != POOL_ROWS or len(np.unique(selected_array)) != POOL_ROWS:
        raise ValueError("Candidate-pool subsample changed size or contains duplicates")
    return (
        pool[selected_array],
        [sources[index] for index in selected_array],
        [concentrations[index] for index in selected_array],
    )


def minimum_distances(coordinates: np.ndarray, references: np.ndarray, chunk_size: int = 2_000) -> np.ndarray:
    result = np.empty(len(coordinates), dtype=float)
    for start in range(0, len(coordinates), chunk_size):
        chunk = coordinates[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - references[None, :, :], axis=2)
        result[start : start + len(chunk)] = distances.min(axis=1)
    return result


def maximin_indices(
    coordinates: np.ndarray,
    references: np.ndarray,
    eligible: np.ndarray,
    count: int,
) -> np.ndarray:
    available = eligible.copy()
    minimum = minimum_distances(coordinates, references)
    selected = []
    for _ in range(count):
        candidates = np.flatnonzero(available)
        if not len(candidates):
            raise ValueError("Maximin candidate set exhausted")
        pick = int(candidates[np.argmax(minimum[candidates])])
        selected.append(pick)
        available[pick] = False
        distance = np.linalg.norm(coordinates - coordinates[pick], axis=1)
        minimum = np.minimum(minimum, distance)
    return np.asarray(selected, dtype=int)


def referee_indices(coordinates: np.ndarray, clusters: int) -> np.ndarray:
    labels = KMeans(n_clusters=clusters, random_state=POOL_DESIGN_SEED, n_init=50).fit_predict(coordinates)
    selected = []
    for label in range(clusters):
        members = np.flatnonzero(labels == label)
        centre = coordinates[members].mean(axis=0)
        selected.append(int(members[np.argmin(np.linalg.norm(coordinates[members] - centre, axis=1))]))
    if len(set(selected)) != clusters:
        raise ValueError("Referee medoids are not unique")
    return np.asarray(selected, dtype=int)


def build_design(
    prefix_weights_path: Path,
    original_weights_path: Path,
    extension_weights_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = base.load_canonical_panel_geometry()
    buckets = panel.buckets
    proportional = base.runtime_weights(panel.proportional)
    unimax8 = base.runtime_weights(base.unimax_weights(panel.proportional, 8.0))
    incumbent = base.runtime_weights(panel.phase1[panel.row_id.index(base.INCUMBENT_ROW_ID)])
    candidate_ids, prefix_weights = base.load_prefix_weights(prefix_weights_path, buckets)
    wave1_ids, wave1 = load_wave1_fit_weights(
        (
            (original_weights_path, EXPECTED_ORIGINAL_SHA256),
            (extension_weights_path, EXPECTED_EXTENSION_SHA256),
        ),
        buckets,
    )

    phase_1_exposure = panel.phase1 * panel.c1[None, :]
    total_exposure = panel.phase0 * panel.c0[None, :] + phase_1_exposure
    phase_1_bucket_caps = phase_1_exposure.max(axis=0)
    total_bucket_caps = total_exposure.max(axis=0)
    controls = (proportional, unimax8, incumbent)
    forbidden = {tuple(base.runtime_counts(row)) for row in (*controls, *wave1)}
    full_pool, sources, concentrations, rejection_counts = base.pool_candidates(
        historical_phase_1=panel.phase1,
        proportional=proportional,
        prefix_weights=prefix_weights,
        phase_0_scales=panel.c0,
        phase_1_scales=panel.c1,
        phase_1_bucket_caps=phase_1_bucket_caps,
        total_bucket_caps=total_bucket_caps,
        forbidden=forbidden,
        design_seed=POOL_DESIGN_SEED,
        dirichlet_draws_per_concentration=DIRICHLET_DRAWS_PER_CONCENTRATION,
    )
    pool, sources, concentrations = deterministic_pool_subset(full_pool, sources, concentrations)
    counts = np.stack([base.runtime_counts(row) for row in pool]).astype(np.uint16)
    if not np.array_equal(pool, counts.astype(float) / base.MIXTURE_BLOCK_SIZE):
        raise ValueError("Frozen candidate pool is not runtime exact")
    if len({tuple(row) for row in counts}) != len(counts):
        raise ValueError("Frozen candidate pool contains duplicate runtime mixtures")

    coordinates = np.sqrt(pool) / np.sqrt(2.0)
    reference_weights = np.vstack([wave1, *controls])
    reference_coordinates = np.sqrt(reference_weights) / np.sqrt(2.0)
    tv = 0.5 * np.abs(pool - proportional).sum(axis=1)
    maximum_exposure = np.max(pool * panel.c1[None, :], axis=1)
    near_eligible = (tv <= NEAR_TV_MAX + 1e-12) & (maximum_exposure <= NEAR_PHASE_1_EPOCH_MAX + 1e-12)
    near = maximin_indices(coordinates, reference_coordinates, near_eligible, NEAR_COVERAGE_ROWS)
    near_coordinates = coordinates[near]
    global_eligible = np.ones(len(pool), dtype=bool)
    global_eligible[near] = False
    global_rows = maximin_indices(
        coordinates,
        np.vstack([reference_coordinates, near_coordinates]),
        global_eligible,
        GLOBAL_COVERAGE_ROWS,
    )
    coverage_pool_indices = np.concatenate([near, global_rows])
    coverage_coordinates = coordinates[coverage_pool_indices]
    near_referees = referee_indices(coverage_coordinates[:NEAR_COVERAGE_ROWS], REFEREE_ROWS_PER_TRANCHE)
    global_referees = referee_indices(coverage_coordinates[NEAR_COVERAGE_ROWS:], REFEREE_ROWS_PER_TRANCHE)
    referee_positions = {
        *(int(value) for value in near_referees),
        *(NEAR_COVERAGE_ROWS + int(value) for value in global_referees),
    }

    metadata_rows = []
    for pool_index, (source, concentration) in enumerate(zip(sources, concentrations, strict=True)):
        metadata_rows.append(
            {
                "pool_index": pool_index,
                "source": source,
                "concentration": concentration,
                "tv_to_proportional": float(tv[pool_index]),
                "hellinger_to_proportional": base.hellinger(pool[pool_index], proportional),
                "max_phase_1_materialized_epoch": float(maximum_exposure[pool_index]),
            }
        )
    pool_metadata = pd.DataFrame(metadata_rows)

    coverage_rows = []
    coverage_weights = []
    for position, pool_index in enumerate(coverage_pool_indices):
        role = "wave2_near_fill" if position < NEAR_COVERAGE_ROWS else "wave2_global_maximin"
        continuation_id = f"fit_{role}_{position:02d}"
        weights = pool[pool_index]
        coverage_rows.append(
            {
                "continuation_id": continuation_id,
                "role": role,
                "fit_budget": True,
                "referee_holdout": position in referee_positions,
                "pool_index": int(pool_index),
                "tv_to_proportional": float(tv[pool_index]),
                "hellinger_to_proportional": base.hellinger(weights, proportional),
                "max_phase_1_materialized_epoch": float(maximum_exposure[pool_index]),
                "weights_json": json.dumps(dict(zip(buckets, weights, strict=True)), sort_keys=True),
            }
        )
        for bucket_position, (bucket, count, weight) in enumerate(
            zip(buckets, counts[pool_index], weights, strict=True)
        ):
            coverage_weights.append(
                {
                    "continuation_id": continuation_id,
                    "role": role,
                    "fit_budget": True,
                    "referee_holdout": position in referee_positions,
                    "pool_index": int(pool_index),
                    "bucket": bucket,
                    "phase_1_count": int(count),
                    "phase_1_weight": float(weight),
                    "phase_1_materialized_epochs": float(weight * panel.c1[bucket_position]),
                    "historical_phase_1_bucket_epoch_cap": float(phase_1_bucket_caps[bucket_position]),
                    "historical_total_bucket_epoch_cap": float(total_bucket_caps[bucket_position]),
                }
            )
    coverage_summary = pd.DataFrame(coverage_rows)
    coverage_weights_frame = pd.DataFrame(coverage_weights)
    combined_coordinates = np.vstack([reference_coordinates, coverage_coordinates])
    combined_distances = np.linalg.norm(combined_coordinates[:, None, :] - combined_coordinates[None, :, :], axis=2)
    np.fill_diagonal(combined_distances, np.inf)
    manifest: dict[str, object] = {
        "design_stage": "wave2_outcome_blind_pool_and_coverage",
        "target_prefix_candidate": TARGET_PREFIX,
        "endpoint_metric_values_used_for_selection": False,
        "pool_design_seed": POOL_DESIGN_SEED,
        "dirichlet_draws_per_concentration": DIRICHLET_DRAWS_PER_CONCENTRATION,
        "candidate_pool_rows": len(pool),
        "candidate_pool_rows_by_concentration": {
            "historical": sum(value is None for value in concentrations),
            **{
                f"dirichlet_{concentration:g}": sum(value == concentration for value in concentrations)
                for concentration in base.DIRICHLET_CONCENTRATIONS
            },
        },
        "candidate_pool_runtime_block_size": base.MIXTURE_BLOCK_SIZE,
        "wave1_fit_repellers": len(wave1),
        "wave1_fit_continuation_ids": list(wave1_ids),
        "control_repellers": ["proportional", "unimax8", "historical_incumbent"],
        "coverage_rows": len(coverage_summary),
        "near_coverage_rows": NEAR_COVERAGE_ROWS,
        "global_coverage_rows": GLOBAL_COVERAGE_ROWS,
        "referee_holdout_rows": int(coverage_summary.referee_holdout.sum()),
        "near_constraints": {
            "tv_to_proportional_max": NEAR_TV_MAX,
            "max_phase_1_materialized_epoch": NEAR_PHASE_1_EPOCH_MAX,
        },
        "selection_geometry": "Hellinger distance on runtime-exact phase-1 mixtures",
        "minimum_combined_wave1_control_coverage_hellinger": float(combined_distances.min()),
        "candidate_prefix_ids_for_support_checks": list(candidate_ids),
        "candidate_prefix_weights_sha256": file_sha256(prefix_weights_path),
        "original_continuation_weights_sha256": file_sha256(original_weights_path),
        "extension_continuation_weights_sha256": file_sha256(extension_weights_path),
        "rejection_counts_before_pool_subsample": rejection_counts,
        "referee_selection": (
            "four Hellinger medoids from an outcome-blind KMeans partition within each fixed-coverage "
            "tranche; these eight rows are withheld from final adaptive-model fitting"
        ),
    }
    return counts, pool_metadata, coverage_summary, coverage_weights_frame, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, pool_metadata, coverage_summary, coverage_weights, manifest = build_design(
        args.prefix_weights,
        args.original_weights,
        args.extension_weights,
    )
    counts_path = args.output_dir / "candidate_pool_counts.npy"
    metadata_path = args.output_dir / "candidate_pool_metadata.csv"
    coverage_summary_path = args.output_dir / "coverage_summary.csv"
    coverage_weights_path = args.output_dir / "coverage_weights.csv"
    np.save(counts_path, counts, allow_pickle=False)
    pool_metadata.to_csv(metadata_path, index=False)
    coverage_summary.to_csv(coverage_summary_path, index=False)
    coverage_weights.to_csv(coverage_weights_path, index=False)
    manifest.update(
        {
            "candidate_pool_counts_sha256": file_sha256(counts_path),
            "candidate_pool_metadata_sha256": file_sha256(metadata_path),
            "coverage_summary_sha256": file_sha256(coverage_summary_path),
            "coverage_weights_sha256": file_sha256(coverage_weights_path),
        }
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(coverage_summary.groupby(["role", "referee_holdout"]).size().to_string())
    print("\n", json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
