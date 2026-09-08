# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas"]
# ///
"""Freeze an exposure-aware Delphi phase-1 continuation panel.

Round 1 crosses the same 50 fit-budget continuations with every selected prefix.
The panel spans the complete historical phase-1 exposure range while reserving
explicit radial coverage near proportional. Fit selection uses no endpoint
metric values: exact mixtures are stratified jointly by maximum materialized
phase-1 epochs and total variation from proportional, then greedily spread in
phase-weighted square-root geometry within each cell. The label-selected
historical incumbent is retained only as a fixed control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_common_branches_20260824"
DEFAULT_PREFIX_WEIGHTS = (
    SCRIPT_DIR / "reference_outputs" / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
)
MIXTURE_BLOCK_SIZE = 2_048
DESIGN_SEED = 20_260_824
FIT_PER_EXPOSURE_BIN = 10
EXPOSURE_BIN_EDGES = (0.0, 5.0, 15.0, 25.0, 35.0)
TV_BIN_EDGES = (0.0, 0.05, 0.15, 0.25, 0.5, 0.75, 1.000001)
TV_QUOTAS_BY_EXPOSURE_BIN = (
    (1, 2, 3, 3, 1, 0),
    (0, 1, 3, 4, 2, 0),
    (0, 0, 1, 6, 3, 0),
    (0, 0, 0, 5, 5, 0),
    (0, 0, 0, 3, 7, 0),
)
DIRICHLET_CONCENTRATIONS = (500.0, 100.0, 20.0, 5.0, 1.0)
DIRICHLET_DRAWS_PER_CONCENTRATION = 20_000
EXPECTED_PREFIX_COUNT = 5
INCUMBENT_ROW_ID = "run_00125"
MINIMUM_PAIRWISE_DIRECTION_DISTANCE = 0.03
PROPORTIONAL_POLICY_EPOCHS = 0.905353
SOURCE_PANEL_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714/"
    "source/fit_panel_table9_macro-4f283bacb4ef269c.csv"
)
SOURCE_PANEL_SHA256 = "4f283bacb4ef269c396277cbd518ef74212a51741c909a1e1e9ace040751d507"
EXPECTED_CANONICAL_ROWS = 280
EXPECTED_PREFIX_TRAIN_STEPS = 2_400
EXPECTED_FULL_TRAIN_STEPS = 3_007


@dataclass(frozen=True)
class CanonicalPanelGeometry:
    buckets: tuple[str, ...]
    phase0: np.ndarray
    phase1: np.ndarray
    row_id: tuple[str, ...]
    c0: np.ndarray
    c1: np.ndarray
    proportional: np.ndarray


@dataclass
class AvailableCell:
    exposure_position: int
    radial_position: int
    quota: int
    indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix-weights", type=Path, default=DEFAULT_PREFIX_WEIGHTS)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_counts(weights: np.ndarray) -> np.ndarray:
    """Map a mixture to the exact largest-remainder 2048-sequence block."""
    target = np.asarray(weights, dtype=float) * MIXTURE_BLOCK_SIZE
    counts = np.floor(target).astype(np.int64)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    if remaining:
        order = np.argsort(-(target - counts), kind="stable")
        counts[order[:remaining]] += 1
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE or int(counts.min()) < 0:
        raise ValueError("Invalid runtime mixture counts")
    return counts


def runtime_weights(weights: np.ndarray) -> np.ndarray:
    return runtime_counts(weights) / MIXTURE_BLOCK_SIZE


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def hellinger(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.sqrt(left) - np.sqrt(right)) / np.sqrt(2.0))


def max_phase_exposure(weights: np.ndarray, phase_scales: np.ndarray) -> float:
    return float(np.max(weights * phase_scales))


def unimax_weights(proportional: np.ndarray, epoch_cap: float) -> np.ndarray:
    """Return the most uniform mixture satisfying a per-bucket epoch cap."""
    ceiling = proportional * (epoch_cap / PROPORTIONAL_POLICY_EPOCHS)
    weights = np.full(len(proportional), 1.0 / len(proportional))
    free = np.ones(len(proportional), dtype=bool)
    for _ in range(len(proportional)):
        over = free & (weights > ceiling)
        if not over.any():
            break
        weights[over] = ceiling[over]
        free &= ~over
        remaining = 1.0 - weights[~free].sum()
        if not free.any() or remaining <= 0.0:
            break
        weights[free] = remaining / free.sum()
    return weights / weights.sum()


def load_canonical_panel_geometry() -> CanonicalPanelGeometry:
    """Reconstruct label-free geometry from the pinned canonical source panel."""
    with fsspec.open(SOURCE_PANEL_URI, "rb") as handle:
        source_bytes = handle.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_PANEL_SHA256:
        raise ValueError(f"Source panel changed: {source_sha256} != {SOURCE_PANEL_SHA256}")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    if len(rows) != EXPECTED_CANONICAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_CANONICAL_ROWS} canonical rows; found {len(rows)}")

    buckets = tuple(column.removeprefix("phase_0_") for column in rows[0] if column.startswith("phase_0_"))
    phase0 = np.asarray([[float(row[f"phase_0_{bucket}"]) for bucket in buckets] for row in rows])
    phase1 = np.asarray([[float(row[f"phase_1_{bucket}"]) for bucket in buckets] for row in rows])
    if not np.allclose(phase0.sum(axis=1), 1.0, atol=1e-8) or not np.allclose(phase1.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Canonical phase mixtures no longer sum to one")

    if rows[0]["run_name"] != "baseline_proportional":
        raise ValueError(f"Canonical proportional row changed: {rows[0]['run_name']}")
    raw_proportional = phase0[0]
    if not np.allclose(raw_proportional, phase1[0], atol=1e-12) or np.any(raw_proportional <= 0.0):
        raise ValueError("Canonical baseline is no longer a strictly positive tied proportional mixture")
    alpha = EXPECTED_PREFIX_TRAIN_STEPS / EXPECTED_FULL_TRAIN_STEPS
    c0 = PROPORTIONAL_POLICY_EPOCHS * alpha / raw_proportional
    c1 = PROPORTIONAL_POLICY_EPOCHS * (1.0 - alpha) / raw_proportional
    return CanonicalPanelGeometry(
        buckets=buckets,
        phase0=phase0,
        phase1=phase1,
        row_id=tuple(row["run_name"] for row in rows),
        c0=c0,
        c1=c1,
        proportional=raw_proportional,
    )


def load_prefix_weights(path: Path, buckets: tuple[str, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    frame = pd.read_csv(path)
    required = {"candidate_id", "bucket", "phase_0_count", "phase_0_weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prefix weights are missing columns: {sorted(required - set(frame.columns))}")
    candidate_ids = tuple(frame.candidate_id.drop_duplicates())
    if len(candidate_ids) != EXPECTED_PREFIX_COUNT:
        raise ValueError(f"Expected {EXPECTED_PREFIX_COUNT} candidate prefixes; found {len(candidate_ids)}")
    rows = []
    for candidate_id in candidate_ids:
        group = frame[frame.candidate_id.eq(candidate_id)]
        if tuple(group.bucket) != buckets:
            raise ValueError(f"Bucket order changed for prefix {candidate_id}")
        counts = group.phase_0_count.to_numpy(dtype=np.int64)
        weights = group.phase_0_weight.to_numpy(dtype=float)
        if not np.array_equal(counts, runtime_counts(weights)):
            raise ValueError(f"Prefix {candidate_id} is not runtime exact")
        rows.append(weights)
    return candidate_ids, np.stack(rows)


def exposure_direction(weights: np.ndarray, phase_1_scales: np.ndarray) -> np.ndarray:
    """Return a unit direction in square-root materialized-exposure geometry."""
    coordinates = np.sqrt(weights * phase_1_scales)
    norm = float(np.linalg.norm(coordinates))
    if norm <= 0.0:
        raise ValueError("Continuation has zero phase-1 exposure")
    return coordinates / norm


def pool_candidates(
    *,
    historical_phase_1: np.ndarray,
    proportional: np.ndarray,
    prefix_weights: np.ndarray,
    phase_0_scales: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_bucket_caps: np.ndarray,
    total_bucket_caps: np.ndarray,
    forbidden: set[tuple[int, ...]],
    design_seed: int = DESIGN_SEED,
    dirichlet_draws_per_concentration: int = DIRICHLET_DRAWS_PER_CONCENTRATION,
) -> tuple[np.ndarray, list[str], list[float | None], dict[str, int]]:
    """Build an exact candidate union from the archive and deterministic Dirichlet draws."""
    candidates: dict[tuple[int, ...], tuple[np.ndarray, str, float | None]] = {}
    rejection_counts = {"duplicate_or_control": 0, "phase_1_cap": 0, "total_cap": 0}

    def consider(weights: np.ndarray, source: str, concentration: float | None) -> None:
        exact = runtime_weights(weights)
        key = tuple(runtime_counts(exact))
        if key in forbidden or key in candidates:
            rejection_counts["duplicate_or_control"] += 1
            return
        phase_1_exposure = exact * phase_1_scales
        if np.any(phase_1_exposure > phase_1_bucket_caps + 1e-12):
            rejection_counts["phase_1_cap"] += 1
            return
        total_exposure = prefix_weights * phase_0_scales[None, :] + phase_1_exposure
        if np.any(total_exposure > total_bucket_caps[None, :] + 1e-12):
            rejection_counts["total_cap"] += 1
            return
        candidates[key] = (exact, source, concentration)

    for weights in historical_phase_1:
        consider(weights, "historical_panel", None)

    generator = np.random.default_rng(design_seed)
    for concentration in DIRICHLET_CONCENTRATIONS:
        draws = generator.dirichlet(concentration * proportional, size=dirichlet_draws_per_concentration)
        for weights in draws:
            consider(weights, "dirichlet", concentration)

    ordered = list(candidates.values())
    if not ordered:
        raise ValueError("No continuation candidates survived the historical-support constraints")
    return (
        np.stack([row[0] for row in ordered]),
        [row[1] for row in ordered],
        [row[2] for row in ordered],
        rejection_counts,
    )


def exposure_radial_cells(
    maximum_exposure: np.ndarray,
    tv_to_proportional: np.ndarray,
    phase_1_cap: float,
) -> list[tuple[int, int, int, np.ndarray]]:
    exposure_edges = (*EXPOSURE_BIN_EDGES, phase_1_cap + 1e-9)
    cells = []
    for exposure_position, (exposure_lower, exposure_upper) in enumerate(itertools.pairwise(exposure_edges)):
        quotas = TV_QUOTAS_BY_EXPOSURE_BIN[exposure_position]
        if sum(quotas) != FIT_PER_EXPOSURE_BIN:
            raise ValueError(f"Exposure-bin radial quotas do not sum to {FIT_PER_EXPOSURE_BIN}")
        for radial_position, (tv_lower, tv_upper) in enumerate(itertools.pairwise(TV_BIN_EDGES)):
            quota = quotas[radial_position]
            indices = np.flatnonzero(
                (maximum_exposure >= exposure_lower)
                & (maximum_exposure < exposure_upper)
                & (tv_to_proportional >= tv_lower)
                & (tv_to_proportional < tv_upper)
            )
            if len(indices) < quota:
                raise ValueError(
                    f"Exposure/radial cell {exposure_position}/{radial_position} has "
                    f"{len(indices)} candidates; need {quota}"
                )
            cells.append((exposure_position, radial_position, quota, indices))
    return cells


def stratified_maximin_indices(
    directions: np.ndarray,
    cells: list[tuple[int, int, int, np.ndarray]],
    fixed_directions: np.ndarray,
) -> np.ndarray:
    """Round-robin maximin selection keeps exposure and radial strata balanced."""
    selected: list[int] = []
    available = [AvailableCell(exposure, radial, quota, indices.copy()) for exposure, radial, quota, indices in cells]
    reference = fixed_directions.copy()
    for round_index in range(max(cell.quota for cell in available)):
        for cell in available:
            if round_index >= cell.quota:
                continue
            distances = np.linalg.norm(directions[cell.indices, None, :] - reference[None, :, :], axis=2)
            pick_position = int(np.argmax(distances.min(axis=1)))
            pick = int(cell.indices[pick_position])
            selected.append(pick)
            cell.indices = np.delete(cell.indices, pick_position)
            reference = np.vstack([reference, directions[pick]])
    return np.asarray(selected, dtype=int)


def minimum_pairwise_distance(directions: np.ndarray) -> float:
    distances = np.linalg.norm(directions[:, None, :] - directions[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def build_design(prefix_weights_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = load_canonical_panel_geometry()
    buckets = panel.buckets
    proportional = runtime_weights(panel.proportional)
    unimax8 = runtime_weights(unimax_weights(panel.proportional, 8.0))
    incumbent_index = panel.row_id.index(INCUMBENT_ROW_ID)
    incumbent_continuation = runtime_weights(panel.phase1[incumbent_index])
    candidate_ids, prefix_weights = load_prefix_weights(prefix_weights_path, buckets)

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
        ("control_unimax8", "operator_control", unimax8),
        ("control_incumbent_planned", "historical_incumbent_control", incumbent_continuation),
    ]
    forbidden = {tuple(runtime_counts(weights)) for _, _, weights in controls}
    pool, sources, concentrations, rejection_counts = pool_candidates(
        historical_phase_1=panel.phase1,
        proportional=proportional,
        prefix_weights=prefix_weights,
        phase_0_scales=phase_0_scales,
        phase_1_scales=phase_1_scales,
        phase_1_bucket_caps=phase_1_bucket_caps,
        total_bucket_caps=total_bucket_caps,
        forbidden=forbidden,
    )
    directions = np.stack([exposure_direction(weights, phase_1_scales) for weights in pool])
    maximum_exposure = np.max(pool * phase_1_scales[None, :], axis=1)
    tv_to_proportional = 0.5 * np.abs(pool - proportional).sum(axis=1)
    cells = exposure_radial_cells(maximum_exposure, tv_to_proportional, phase_1_cap)
    # Proportional and UniMax-8 are label-blind anchors. The outcome-selected
    # incumbent remains a control, but must not repel fit points.
    fixed_directions = np.stack([exposure_direction(weights, phase_1_scales) for _, _, weights in controls[:2]])
    selected = stratified_maximin_indices(directions, cells, fixed_directions)

    fit_directions = directions[selected]
    minimum_distance = minimum_pairwise_distance(fit_directions)
    if minimum_distance < MINIMUM_PAIRWISE_DIRECTION_DISTANCE:
        raise ValueError(
            f"Minimum fit-direction distance {minimum_distance:.4f} is below "
            f"{MINIMUM_PAIRWISE_DIRECTION_DISTANCE:.4f}"
        )

    selected_rows: list[tuple[str, str, bool, str, float | None, np.ndarray]] = []
    for position, index in enumerate(selected):
        selected_rows.append(
            (
                f"fit_maximin_{position:02d}",
                "exposure_radial_stratified_maximin",
                True,
                sources[index],
                concentrations[index],
                pool[index],
            )
        )
    selected_rows.extend(
        (continuation_id, role, False, "fixed_control", None, weights) for continuation_id, role, weights in controls
    )

    if sum(row[2] for row in selected_rows) != FIT_PER_EXPOSURE_BIN * len(EXPOSURE_BIN_EDGES):
        raise ValueError("Round-1 fit budget changed")
    runtime_keys = [tuple(runtime_counts(row[5])) for row in selected_rows]
    if len(runtime_keys) != len(set(runtime_keys)):
        raise ValueError("Continuation design contains duplicate runtime mixtures")

    summaries = []
    long_rows = []
    for continuation_id, role, fit_budget, source, concentration, weights in selected_rows:
        counts = runtime_counts(weights)
        maximum = max_phase_exposure(weights, phase_1_scales)
        phase_1_exposure = weights * phase_1_scales
        total_exposure = prefix_weights * phase_0_scales[None, :] + phase_1_exposure
        if np.any(phase_1_exposure > phase_1_bucket_caps + 1e-12):
            raise ValueError(f"Continuation {continuation_id} exceeds a per-bucket phase-1 support cap")
        if np.any(total_exposure > total_bucket_caps[None, :] + 1e-12):
            raise ValueError(f"Continuation {continuation_id} exceeds a per-bucket total-exposure support cap")
        total_maximum = float(np.max(total_exposure))
        summaries.append(
            {
                "continuation_id": continuation_id,
                "role": role,
                "fit_budget": fit_budget,
                "source": source,
                "concentration": concentration,
                "tv_to_proportional": total_variation(weights, proportional),
                "hellinger_to_proportional": hellinger(weights, proportional),
                "max_phase_1_materialized_epoch": maximum,
                "max_total_materialized_epoch_across_candidate_prefixes": total_maximum,
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
    edges = (*EXPOSURE_BIN_EDGES, phase_1_cap + 1e-9)
    exposure_counts = {
        f"[{lower:g},{upper:g})": int(
            (
                (fit_summary.max_phase_1_materialized_epoch >= lower)
                & (fit_summary.max_phase_1_materialized_epoch < upper)
            ).sum()
        )
        for lower, upper in itertools.pairwise(edges)
    }
    tv_counts = {
        f"[{lower:g},{upper:g})": int(
            ((fit_summary.tv_to_proportional >= lower) & (fit_summary.tv_to_proportional < upper)).sum()
        )
        for lower, upper in itertools.pairwise(TV_BIN_EDGES)
    }
    manifest: dict[str, object] = {
        "design_seed": DESIGN_SEED,
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "fit_continuations": int(fit_summary.shape[0]),
        "control_continuations": int((~summary.fit_budget).sum()),
        "exposure_bin_edges": [*EXPOSURE_BIN_EDGES, phase_1_cap],
        "fit_per_exposure_bin": FIT_PER_EXPOSURE_BIN,
        "fit_exposure_bin_counts": exposure_counts,
        "tv_bin_edges": list(TV_BIN_EDGES),
        "tv_quotas_by_exposure_bin": [list(row) for row in TV_QUOTAS_BY_EXPOSURE_BIN],
        "fit_tv_bin_counts": tv_counts,
        "candidate_counts_by_exposure_tv_cell": {
            f"{exposure_position}/{radial_position}": len(indices)
            for exposure_position, radial_position, _quota, indices in cells
        },
        "dirichlet_concentrations": list(DIRICHLET_CONCENTRATIONS),
        "dirichlet_draws_per_concentration": DIRICHLET_DRAWS_PER_CONCENTRATION,
        "candidate_pool_size": len(pool),
        "historical_phase_1_epoch_cap": phase_1_cap,
        "historical_total_epoch_cap": total_cap,
        "per_bucket_phase_1_and_total_support_enforced": True,
        "candidate_prefix_ids": list(candidate_ids),
        "candidate_prefix_weights_sha256": file_sha256(prefix_weights_path),
        "incumbent_row_id": INCUMBENT_ROW_ID,
        "selection_geometry": (
            "exposure-by-TV stratification, then round-robin maximin over unit sqrt(c1 * w1) directions"
        ),
        "endpoint_metric_values_used_for_fit_selection": False,
        "outcome_selected_incumbent_used_only_as_control": True,
        "outcome_selected_incumbent_used_as_fit_repeller": False,
        "minimum_fit_direction_distance": minimum_distance,
        "rejection_counts": rejection_counts,
        "fit_max_phase_1_epoch_range": [
            float(fit_summary.max_phase_1_materialized_epoch.min()),
            float(fit_summary.max_phase_1_materialized_epoch.max()),
        ],
        "fit_max_total_epoch_range": [
            float(fit_summary.max_total_materialized_epoch_across_candidate_prefixes.min()),
            float(fit_summary.max_total_materialized_epoch_across_candidate_prefixes.max()),
        ],
        "fit_tv_to_proportional_quantiles": {
            str(quantile): float(fit_summary.tv_to_proportional.quantile(quantile))
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
    }
    return summary, weights, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, weights, manifest = build_design(args.prefix_weights)
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    manifest["continuation_summary_sha256"] = file_sha256(summary_path)
    manifest["continuation_weights_sha256"] = file_sha256(weights_path)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = f"""# Delphi phase-1 common continuation design

Round 1 crosses the same {manifest['fit_continuations']} fit-budget continuations with every selected
prefix. Ten continuations are selected in each of five maximum phase-1 exposure strata. Within each
exposure stratum, frozen total-variation quotas preserve local, intermediate, and global coverage before
deterministic maximin selection over unit `sqrt(c1 * w1)` directions. The pool combines all 280
historical phase-1 coordinates with deterministic proportional-centered Dirichlet draws, after exact
`1/{MIXTURE_BLOCK_SIZE}` runtime materialization.

Three controls do not consume fit budget: proportional, UniMax-8, and the historical continuation paired
with the observed cap-safe prefix incumbent `{INCUMBENT_ROW_ID}`. The incumbent is outcome-selected, so it
is not used as a maximin repeller for the fit panel. The launcher also adds prefix-specific tied controls,
three prefix-seed stability sentinels per selected prefix, and four same-checkpoint phase-1 data-seed
replicates; these remain outside fit budget.

Every fit and common-control continuation stays within each bucket's observed canonical-panel phase-1 and
total materialized-exposure envelopes for every frozen candidate prefix. Prefix-specific tied controls and
seed sentinels are natural policy controls rather than support-filtered fit rows; one tied control exceeds one
phase-1 bucket envelope by 0.016 epoch. The largest bucket-wise caps for the filtered continuation panel are
{manifest['historical_phase_1_epoch_cap']:.6f} phase-1 epochs and
{manifest['historical_total_epoch_cap']:.6f} total epochs. These remain coordinate-wise support guardrails,
not a claim that every joint mixture is in-distribution.

Historical support caps are measured before runtime lattice materialization, while every candidate is
checked after exact materialization. This conservative asymmetry accounts for the rejected Dirichlet draws.
The total-exposure cap is a verification guardrail and did not reject a candidate in this frozen pool.

Fit exposure-bin counts: `{json.dumps(manifest['fit_exposure_bin_counts'], sort_keys=True)}`

Fit TV-bin counts: `{json.dumps(manifest['fit_tv_bin_counts'], sort_keys=True)}`

Minimum pairwise fit-direction distance: {manifest['minimum_fit_direction_distance']:.4f}

Continuation weights SHA-256: `{manifest['continuation_weights_sha256']}`
"""
    (args.output_dir / "report.md").write_text(report)
    print(summary.groupby(["fit_budget", "role"]).size().to_string())
    print("\nFit exposure coverage:")
    print(
        summary[summary.fit_budget]
        .max_phase_1_materialized_epoch.describe()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    print("\n", json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
