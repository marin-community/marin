# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas"]
# ///
"""Freeze 50 additional outcome-blind continuations for the KL0.05 prefix.

The extension preserves the first wave's exposure and radial quotas while
using its 50 fit mixtures as fixed maximin repellers. It is frozen before any
KL0.05 branch outcomes are inspected and uses the same canonical-panel support
envelopes as the original design.

Run from the repository root with::

    PYTHONPATH=. uv run \
      experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_kl0p05_wave1_extension_20260825.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_kl0p05_wave1_extension_20260825"
DEFAULT_EXISTING_CONTINUATION_WEIGHTS = (
    SCRIPT_DIR / "reference_outputs" / "delphi_phase1_common_branches_20260824" / "continuation_weights.csv"
)
EXPECTED_EXISTING_CONTINUATION_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
EXPECTED_EXISTING_FIT_COUNT = 50
EXTENSION_FIT_COUNT = 50
EXTENSION_DESIGN_SEED = 20_260_825
EXTENSION_DIRICHLET_DRAWS_PER_CONCENTRATION = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix-weights", type=Path, default=base.DEFAULT_PREFIX_WEIGHTS)
    parser.add_argument(
        "--existing-continuation-weights",
        type=Path,
        default=DEFAULT_EXISTING_CONTINUATION_WEIGHTS,
    )
    return parser.parse_args()


def load_existing_fit_weights(path: Path, buckets: tuple[str, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    actual_sha256 = base.file_sha256(path)
    if actual_sha256 != EXPECTED_EXISTING_CONTINUATION_SHA256:
        raise ValueError(
            f"Existing continuation design changed: {actual_sha256} != {EXPECTED_EXISTING_CONTINUATION_SHA256}"
        )
    frame = pd.read_csv(path)
    fit_mask = frame.fit_budget.astype(str).str.lower().eq("true")
    fit_frame = frame[fit_mask]
    continuation_ids = tuple(fit_frame.continuation_id.drop_duplicates())
    if len(continuation_ids) != EXPECTED_EXISTING_FIT_COUNT:
        raise ValueError(f"Expected {EXPECTED_EXISTING_FIT_COUNT} existing fit rows; found {len(continuation_ids)}")

    rows = []
    for continuation_id in continuation_ids:
        group = fit_frame[fit_frame.continuation_id.eq(continuation_id)]
        if tuple(group.bucket) != buckets:
            raise ValueError(f"Bucket order changed for existing continuation {continuation_id}")
        counts = group.phase_1_count.to_numpy(dtype=np.int64)
        weights = group.phase_1_weight.to_numpy(dtype=float)
        if not np.array_equal(counts, base.runtime_counts(weights)):
            raise ValueError(f"Existing continuation {continuation_id} is not runtime exact")
        rows.append(weights)
    return continuation_ids, np.stack(rows)


def build_design(
    prefix_weights_path: Path,
    existing_continuation_weights_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = base.load_canonical_panel_geometry()
    buckets = panel.buckets
    proportional = base.runtime_weights(panel.proportional)
    unimax8 = base.runtime_weights(base.unimax_weights(panel.proportional, 8.0))
    incumbent_index = panel.row_id.index(base.INCUMBENT_ROW_ID)
    incumbent_continuation = base.runtime_weights(panel.phase1[incumbent_index])
    candidate_ids, prefix_weights = base.load_prefix_weights(prefix_weights_path, buckets)
    existing_ids, existing_weights = load_existing_fit_weights(existing_continuation_weights_path, buckets)
    wave1a_anchor_weights = existing_weights[0]
    wave1a_anchor_id = existing_ids[0]

    phase_0_scales = panel.c0
    phase_1_scales = panel.c1
    historical_phase_1_exposure = panel.phase1 * phase_1_scales[None, :]
    historical_total_exposure = panel.phase0 * phase_0_scales[None, :] + historical_phase_1_exposure
    phase_1_bucket_caps = np.max(historical_phase_1_exposure, axis=0)
    total_bucket_caps = np.max(historical_total_exposure, axis=0)
    phase_1_cap = float(phase_1_bucket_caps.max())
    total_cap = float(total_bucket_caps.max())

    controls = [
        ("control_proportional", "operator_control", proportional),
        (f"control_wave1a_anchor_{wave1a_anchor_id}", "cross_wave_anchor", wave1a_anchor_weights),
        ("control_incumbent_planned", "historical_incumbent_control", incumbent_continuation),
    ]
    forbidden = {tuple(base.runtime_counts(weights)) for weights in existing_weights}
    forbidden.update(tuple(base.runtime_counts(weights)) for _, _, weights in controls)
    forbidden.add(tuple(base.runtime_counts(unimax8)))
    pool, sources, concentrations, rejection_counts = base.pool_candidates(
        historical_phase_1=panel.phase1,
        proportional=proportional,
        prefix_weights=prefix_weights,
        phase_0_scales=phase_0_scales,
        phase_1_scales=phase_1_scales,
        phase_1_bucket_caps=phase_1_bucket_caps,
        total_bucket_caps=total_bucket_caps,
        forbidden=forbidden,
        design_seed=EXTENSION_DESIGN_SEED,
        dirichlet_draws_per_concentration=EXTENSION_DIRICHLET_DRAWS_PER_CONCENTRATION,
    )
    directions = np.stack([base.exposure_direction(weights, phase_1_scales) for weights in pool])
    maximum_exposure = np.max(pool * phase_1_scales[None, :], axis=1)
    tv_to_proportional = 0.5 * np.abs(pool - proportional).sum(axis=1)
    cells = base.exposure_radial_cells(maximum_exposure, tv_to_proportional, phase_1_cap)

    existing_directions = np.stack([base.exposure_direction(weights, phase_1_scales) for weights in existing_weights])
    anchor_directions = np.stack(
        [base.exposure_direction(weights, phase_1_scales) for weights in (proportional, unimax8)]
    )
    selected = base.stratified_maximin_indices(
        directions,
        cells,
        np.vstack([existing_directions, anchor_directions]),
    )
    if len(selected) != EXTENSION_FIT_COUNT:
        raise ValueError(f"Expected {EXTENSION_FIT_COUNT} extension rows; selected {len(selected)}")

    selected_directions = directions[selected]
    combined_directions = np.vstack([existing_directions, selected_directions])
    minimum_extension_distance = base.minimum_pairwise_distance(selected_directions)
    minimum_combined_distance = base.minimum_pairwise_distance(combined_directions)
    minimum_extension_to_existing_distance = float(
        np.linalg.norm(selected_directions[:, None, :] - existing_directions[None, :, :], axis=2).min()
    )
    if minimum_combined_distance < base.MINIMUM_PAIRWISE_DIRECTION_DISTANCE:
        raise ValueError(
            f"Combined minimum direction distance {minimum_combined_distance:.4f} is below "
            f"{base.MINIMUM_PAIRWISE_DIRECTION_DISTANCE:.4f}"
        )

    selected_rows: list[tuple[str, str, bool, str, float | None, np.ndarray]] = []
    for position, index in enumerate(selected):
        selected_rows.append(
            (
                f"fit_wave1_extension_{position:02d}",
                "wave1_extension_stratified_maximin",
                True,
                sources[index],
                concentrations[index],
                pool[index],
            )
        )
    selected_rows.extend(
        (continuation_id, role, False, "fixed_control", None, weights) for continuation_id, role, weights in controls
    )

    all_fit_keys = [tuple(base.runtime_counts(weights)) for weights in existing_weights]
    all_fit_keys.extend(tuple(base.runtime_counts(row[5])) for row in selected_rows if row[2])
    if len(all_fit_keys) != EXPECTED_EXISTING_FIT_COUNT + EXTENSION_FIT_COUNT:
        raise ValueError("Combined first-wave fit count changed")
    if len(all_fit_keys) != len(set(all_fit_keys)):
        raise ValueError("The extension duplicates an existing runtime mixture")

    summaries = []
    long_rows = []
    for continuation_id, role, fit_budget, source, concentration, weights in selected_rows:
        counts = base.runtime_counts(weights)
        phase_1_exposure = weights * phase_1_scales
        total_exposure = prefix_weights * phase_0_scales[None, :] + phase_1_exposure
        if np.any(phase_1_exposure > phase_1_bucket_caps + 1e-12):
            raise ValueError(f"Continuation {continuation_id} exceeds a per-bucket phase-1 support cap")
        if np.any(total_exposure > total_bucket_caps[None, :] + 1e-12):
            raise ValueError(f"Continuation {continuation_id} exceeds a per-bucket total-exposure support cap")
        summaries.append(
            {
                "continuation_id": continuation_id,
                "role": role,
                "fit_budget": fit_budget,
                "source": source,
                "concentration": concentration,
                "tv_to_proportional": base.total_variation(weights, proportional),
                "hellinger_to_proportional": base.hellinger(weights, proportional),
                "max_phase_1_materialized_epoch": base.max_phase_exposure(weights, phase_1_scales),
                "max_total_materialized_epoch_across_candidate_prefixes": float(np.max(total_exposure)),
                "weights_json": json.dumps(dict(zip(buckets, weights, strict=True)), sort_keys=True),
            }
        )
        for bucket_position, (bucket, count, weight, scale) in enumerate(
            zip(buckets, counts, weights, phase_1_scales, strict=True)
        ):
            long_rows.append(
                {
                    "continuation_id": continuation_id,
                    "role": role,
                    "fit_budget": fit_budget,
                    "source": source,
                    "concentration": concentration,
                    "bucket": bucket,
                    "phase_1_count": int(count),
                    "phase_1_weight": float(weight),
                    "phase_1_materialized_epochs": float(weight * scale),
                    "historical_phase_1_bucket_epoch_cap": float(phase_1_bucket_caps[bucket_position]),
                    "historical_total_bucket_epoch_cap": float(total_bucket_caps[bucket_position]),
                }
            )

    summary = pd.DataFrame(summaries)
    weights = pd.DataFrame(long_rows)
    fit_summary = summary[summary.fit_budget]
    exposure_edges = (*base.EXPOSURE_BIN_EDGES, phase_1_cap + 1e-9)
    exposure_counts = {
        f"[{lower:g},{upper:g})": int(
            (
                (fit_summary.max_phase_1_materialized_epoch >= lower)
                & (fit_summary.max_phase_1_materialized_epoch < upper)
            ).sum()
        )
        for lower, upper in itertools.pairwise(exposure_edges)
    }
    tv_counts = {
        f"[{lower:g},{upper:g})": int(
            ((fit_summary.tv_to_proportional >= lower) & (fit_summary.tv_to_proportional < upper)).sum()
        )
        for lower, upper in itertools.pairwise(base.TV_BIN_EDGES)
    }
    candidate_counts_by_cell = {
        f"{exposure_position}/{radial_position}": len(indices)
        for exposure_position, radial_position, _quota, indices in cells
    }
    forced_cells = [
        f"{exposure_position}/{radial_position}"
        for exposure_position, radial_position, quota, indices in cells
        if quota > 0 and len(indices) == quota
    ]
    cross_wave_anchor = {
        "continuation_id": f"control_wave1a_anchor_{wave1a_anchor_id}",
        "repeats_wave1a_continuation_id": wave1a_anchor_id,
        "fit_budget": False,
    }
    manifest: dict[str, object] = {
        "design_seed": EXTENSION_DESIGN_SEED,
        "dirichlet_draws_per_concentration": EXTENSION_DIRICHLET_DRAWS_PER_CONCENTRATION,
        "design_stage": "wave1_nonadaptive_extension",
        "target_prefix_candidate": "shared_bounded_ensemble_kl0p05",
        "mixture_block_size": base.MIXTURE_BLOCK_SIZE,
        "extension_fit_continuations": int(fit_summary.shape[0]),
        "combined_wave1_fit_continuations": EXPECTED_EXISTING_FIT_COUNT + int(fit_summary.shape[0]),
        "control_continuations": int((~summary.fit_budget).sum()),
        "existing_continuation_weights_sha256": base.file_sha256(existing_continuation_weights_path),
        "existing_fit_continuation_ids": list(existing_ids),
        "candidate_prefix_ids_for_support_checks": list(candidate_ids),
        "candidate_prefix_weights_sha256": base.file_sha256(prefix_weights_path),
        "incumbent_row_id": base.INCUMBENT_ROW_ID,
        "exposure_bin_edges": [*base.EXPOSURE_BIN_EDGES, phase_1_cap],
        "fit_per_exposure_bin": base.FIT_PER_EXPOSURE_BIN,
        "fit_exposure_bin_counts": exposure_counts,
        "tv_bin_edges": list(base.TV_BIN_EDGES),
        "fit_tv_bin_counts": tv_counts,
        "tv_quotas_by_exposure_bin": [list(row) for row in base.TV_QUOTAS_BY_EXPOSURE_BIN],
        "candidate_counts_by_exposure_tv_cell": candidate_counts_by_cell,
        "forced_candidate_cells": forced_cells,
        "dirichlet_concentrations": list(base.DIRICHLET_CONCENTRATIONS),
        "candidate_pool_size": len(pool),
        "candidate_pool_expansion_reason": (
            "After excluding the frozen Wave 1A rows, the original 20,000-draw pool left cells 1/1 and 4/3 "
            "empty and cell 2/2 with exactly one candidate for its one-row quota. The outcome-blind pool was "
            "regenerated with a new frozen seed and 100,000 draws per concentration while retaining every "
            "preregistered exposure and TV quota."
        ),
        "historical_phase_1_epoch_cap": phase_1_cap,
        "historical_total_epoch_cap": total_cap,
        "per_bucket_phase_1_and_total_support_enforced": True,
        "selection_geometry": (
            "original exposure-by-TV quotas with round-robin maximin over unit sqrt(c1 * w1) directions; "
            "the original 50 fit rows plus proportional and UniMax-8 are fixed repellers"
        ),
        "endpoint_metric_values_used_for_fit_selection": False,
        "first_wave_outcomes_inspected_before_freeze": False,
        "cross_wave_anchor": cross_wave_anchor,
        "minimum_extension_direction_distance": minimum_extension_distance,
        "minimum_extension_to_existing_direction_distance": minimum_extension_to_existing_distance,
        "minimum_combined_wave1_direction_distance": minimum_combined_distance,
        "rejection_counts": rejection_counts,
    }
    return summary, weights, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, weights, manifest = build_design(args.prefix_weights, args.existing_continuation_weights)
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    manifest["continuation_summary_sha256"] = base.file_sha256(summary_path)
    manifest["continuation_weights_sha256"] = base.file_sha256(weights_path)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cross_wave_anchor = cast(dict[str, object], manifest["cross_wave_anchor"])
    report = f"""# Delphi KL0.05 phase-1 Wave 1 extension

This freezes 50 additional fit-budget continuations for the KL0.05 prefix before inspecting any branch
outcomes. It repeats the original five exposure-stratum and TV quotas, but treats all original 50 fit
directions plus proportional and UniMax-8 as fixed maximin repellers. The combined first wave therefore
contains 100 distinct runtime mixtures rather than a second independent draw of the same design.

All rows retain the original per-bucket phase-1 and total-exposure support checks across all five frozen
prefix candidates. Wave 1B launches the 50 fit rows plus one non-fit cross-wave anchor. That anchor repeats
Wave 1A's `{cross_wave_anchor['repeats_wave1a_continuation_id']}` mixture with a distinct
continuation identity, isolating wave-to-wave execution drift without spending fit budget.

The candidate pool increased from 20,000 to 100,000 draws per concentration because excluding Wave 1A left
cells 1/1 and 4/3 empty and cell 2/2 forced. This was detected before reading endpoint outcomes; all exposure
and TV quotas remained frozen. Forced candidate cells in the enlarged pool:
`{json.dumps(manifest['forced_candidate_cells'])}`.

Minimum extension-to-existing direction distance: {manifest['minimum_extension_to_existing_direction_distance']:.4f}

Minimum combined Wave 1 direction distance: {manifest['minimum_combined_wave1_direction_distance']:.4f}

Continuation weights SHA-256: `{manifest['continuation_weights_sha256']}`
"""
    (args.output_dir / "report.md").write_text(report)
    print(summary.groupby(["fit_budget", "role"]).size().to_string())
    print("\n", json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
