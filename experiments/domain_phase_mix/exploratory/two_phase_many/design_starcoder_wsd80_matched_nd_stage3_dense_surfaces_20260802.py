# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///

"""Freeze systematic Stage-3 surface coverage for the matched N-D WSD80 panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
STAGE2_RESULTS_DIR = PANEL_DIR / "stage2_results_20260801"
COMBINED_OBSERVATIONS_PATH = STAGE2_RESULTS_DIR / "combined_discovery_observations.csv"
STAGE2_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage2_design_20260801.json"
OUTPUT_DIR = PANEL_DIR / "stage3_dense_surface_design_20260802"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage3_dense_surface_design_20260802.json"
STAGE3_LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_matched_nd_stage3_dense_surfaces.py"
STAGE3_ANALYZER_PATH = SCRIPT_DIR / "analyze_starcoder_wsd80_matched_nd_stage3_dense_surfaces_20260802.py"

DESIGN_VERSION = "2026-08-02"
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 1.0 - PHASE_0_FRACTION
REFERENCE_SEED = 20_260_711
EXPECTED_CELLS = 10
EXPECTED_EXISTING_ROWS_PER_CELL = 23
EXPECTED_EXISTING_TIED_PER_CELL = 15
EXPECTED_EXISTING_UNTIED_PER_CELL = 8

PRIMARY_FIBER_COUNT = 16
SECONDARY_FIBER_COUNT = 8
COMMON_POSITIVE_COUNT = 12
COMMON_NEGATIVE_COUNT = 4
LOCAL_OPTIMUM_COUNT = 8
UNTIED_RUNS_PER_CELL = (
    PRIMARY_FIBER_COUNT + SECONDARY_FIBER_COUNT + COMMON_POSITIVE_COUNT + COMMON_NEGATIVE_COUNT + LOCAL_OPTIMUM_COUNT
)
EXPECTED_UNTIED_RUNS = EXPECTED_CELLS * UNTIED_RUNS_PER_CELL

PRIMARY_BASE_NORMALIZED_CONTRASTS = (0.03, 0.08, 0.16, 0.28, 0.44, 0.62, 0.80, 0.96)
PRIMARY_LOCAL_NORMALIZED_OFFSETS = (-0.18, -0.12, -0.075, -0.035, 0.035, 0.075, 0.12, 0.18)
SECONDARY_NORMALIZED_CONTRASTS = (0.08, 0.16, 0.28, 0.42, 0.58, 0.74, 0.88, 0.96)
COMMON_AGGREGATES = (0.18, 0.35, 0.55, 0.75)
COMMON_POSITIVE_NORMALIZED_CONTRASTS = (0.12, 0.32, 0.72)
COMMON_NEGATIVE_NORMALIZED_CONTRAST = -0.35
LOCAL_AGGREGATE_OFFSETS = (-0.06, -0.025, 0.025, 0.06)
LOCAL_NORMALIZED_CONTRAST_OFFSETS = (-0.08, 0.08)
MIN_POLICY_DISTANCE = 0.01
POLICY_INTERIOR_BOUND = 0.01

LAUNCH_FIELDS = (
    "run_name",
    "cell_id",
    "acquisition_kind",
    "phase_0_starcoder",
    "phase_1_starcoder",
    "total_steps",
    "boundary_step",
    "data_seed",
    "simulated_epoch_subset_seed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(p0: float, p1: float) -> float:
    return PHASE_0_FRACTION * p0 + PHASE_1_FRACTION * p1


def _contrast(p0: float, p1: float) -> float:
    return p1 - p0


def _positive_contrast_limit(aggregate: float) -> float:
    return min(aggregate / PHASE_1_FRACTION, (1.0 - aggregate) / PHASE_0_FRACTION)


def _negative_contrast_limit(aggregate: float) -> float:
    return min((1.0 - aggregate) / PHASE_1_FRACTION, aggregate / PHASE_0_FRACTION)


def _policy_from_aggregate_contrast(aggregate: float, contrast: float) -> tuple[float, float]:
    p0 = aggregate - PHASE_1_FRACTION * contrast
    p1 = aggregate + PHASE_0_FRACTION * contrast
    if min(p0, p1) < -1e-10 or max(p0, p1) > 1.0 + 1e-10:
        raise ValueError(f"Infeasible aggregate/contrast coordinate: a={aggregate}, d={contrast}")
    return round(float(np.clip(p0, 0.0, 1.0)), 6), round(float(np.clip(p1, 0.0, 1.0)), 6)


def _policy_from_aggregate_normalized_contrast(aggregate: float, normalized: float) -> tuple[float, float]:
    limit = _positive_contrast_limit(aggregate) if normalized >= 0.0 else _negative_contrast_limit(aggregate)
    return _policy_from_aggregate_contrast(aggregate, normalized * limit)


def _normalized_contrast(p0: float, p1: float) -> float:
    aggregate = _aggregate(p0, p1)
    contrast = _contrast(p0, p1)
    limit = _positive_contrast_limit(aggregate) if contrast >= 0.0 else _negative_contrast_limit(aggregate)
    return contrast / limit


def _point(row: pd.Series | dict[str, Any]) -> tuple[float, float]:
    return round(float(row["phase_0_starcoder"]), 6), round(float(row["phase_1_starcoder"]), 6)


def _distance(point: tuple[float, float], references: set[tuple[float, float]]) -> float:
    if not references:
        return float("inf")
    values = np.asarray(tuple(references), dtype=float)
    return float(np.sqrt(np.square(values - np.asarray(point, dtype=float)).sum(axis=1)).min())


def _nearest_available_on_fiber(
    *,
    aggregate: float,
    target_normalized_contrast: float,
    used: set[tuple[float, float]],
) -> tuple[float, float, float]:
    limit = _positive_contrast_limit(aggregate)
    normalized_grid = np.linspace(0.01, 0.99, 981)
    order = np.argsort(np.abs(normalized_grid - target_normalized_contrast), kind="stable")
    for index in order:
        normalized = float(normalized_grid[index])
        contrast = normalized * limit
        point = _policy_from_aggregate_contrast(aggregate, contrast)
        if abs(_contrast(*point)) < 1e-6:
            continue
        if min(point) < POLICY_INTERIOR_BOUND or max(point) > 1.0 - POLICY_INTERIOR_BOUND:
            continue
        if point in used or _distance(point, used) < MIN_POLICY_DISTANCE:
            continue
        return point[0], point[1], normalized
    raise ValueError(f"Could not place a new point on the a={aggregate:.6f} fiber")


def _secondary_aggregate(primary: float, tied_optimum: float) -> float:
    if abs(tied_optimum - primary) >= 0.04:
        return round(tied_optimum, 6)
    candidates = (
        round(0.6 * tied_optimum, 2),
        round(tied_optimum - 0.12, 2),
        round(tied_optimum + 0.12, 2),
        0.18,
        0.35,
        0.55,
    )
    for candidate in candidates:
        if 0.08 <= candidate <= 0.82 and abs(candidate - primary) >= 0.06:
            return round(candidate, 6)
    raise ValueError(f"Could not construct a secondary aggregate distinct from {primary:.6f}")


def _nearest_available_local_point(
    *,
    aggregate: float,
    normalized: float,
    used: set[tuple[float, float]],
) -> tuple[float, float, float, float]:
    aggregate_offsets = (0.0, -0.002, 0.002, -0.004, 0.004, -0.006, 0.006, -0.008, 0.008)
    normalized_offsets = (
        0.0,
        *(value for step in range(1, 41) for value in (-0.005 * step, 0.005 * step)),
    )
    candidates: list[tuple[float, float, float, float, float]] = []
    for aggregate_offset in aggregate_offsets:
        candidate_aggregate = round(float(np.clip(aggregate + aggregate_offset, 0.06, 0.86)), 6)
        for normalized_offset in normalized_offsets:
            candidate_normalized = float(np.clip(normalized + normalized_offset, 0.03, 0.97))
            point = _policy_from_aggregate_normalized_contrast(candidate_aggregate, candidate_normalized)
            if min(point) < POLICY_INTERIOR_BOUND or max(point) > 1.0 - POLICY_INTERIOR_BOUND:
                continue
            if point in used or _distance(point, used) < MIN_POLICY_DISTANCE:
                continue
            score = abs(aggregate_offset) / 0.01 + abs(normalized_offset)
            candidates.append((score, point[0], point[1], candidate_aggregate, candidate_normalized))
    if not candidates:
        raise ValueError(f"Could not place a local point near a={aggregate:.6f}, u={normalized:.6f}")
    _, p0, p1, actual_aggregate, actual_normalized = min(candidates)
    return p0, p1, actual_aggregate, actual_normalized


def _weight_slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def launch_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the design onto fields that determine the launched training runs."""
    return [{field: row[field] for field in LAUNCH_FIELDS} for row in rows]


def _base_row(
    *,
    cell: pd.DataFrame,
    acquisition_kind: str,
    p0: float,
    p1: float,
    local_index: int,
    primary_aggregate: float,
    primary_best_untied: pd.Series,
    primary_best_tied: pd.Series,
    normalized_contrast: float | None,
) -> dict[str, Any]:
    metadata = cell.iloc[0]
    aggregate = _aggregate(p0, p1)
    contrast = _contrast(p0, p1)
    kind_slug = {
        "primary_fiber": "pf",
        "secondary_fiber": "sf",
        "common_positive": "cp",
        "common_negative": "cn",
        "local_optimum": "lo",
        "primary_tied_anchor": "ta",
        "secondary_tied_anchor": "sa",
    }[acquisition_kind]
    run_name = f"s3_{metadata['cell_id']}_{kind_slug}{local_index:02d}_" f"p0{_weight_slug(p0)}_p1{_weight_slug(p1)}"
    return {
        "run_name": run_name,
        "cell_id": str(metadata["cell_id"]),
        "rung": int(metadata["rung"]),
        "track_memberships": metadata["track_memberships"],
        "hidden_size": int(metadata["hidden_size"]),
        "total_steps": int(metadata["total_steps"]),
        "boundary_step": int(int(metadata["total_steps"]) * PHASE_0_FRACTION),
        "materialized_tokens": int(metadata["materialized_tokens"]),
        "total_parameters": int(metadata["total_parameters"]),
        "non_embedding_parameters": int(metadata["non_embedding_parameters"]),
        "acquisition_kind": acquisition_kind,
        "acquisition_index": local_index,
        "phase_0_starcoder": p0,
        "phase_1_starcoder": p1,
        "aggregate_starcoder": aggregate,
        "phase_contrast": contrast,
        "normalized_phase_contrast": normalized_contrast,
        "primary_fiber_aggregate": primary_aggregate,
        "source_best_untied_p0": float(primary_best_untied["phase_0_starcoder"]),
        "source_best_untied_p1": float(primary_best_untied["phase_1_starcoder"]),
        "source_best_untied_bpb": float(primary_best_untied["starcoder_bpb"]),
        "source_best_tied_weight": float(primary_best_tied["phase_0_starcoder"]),
        "source_best_tied_bpb": float(primary_best_tied["starcoder_bpb"]),
        "data_seed": REFERENCE_SEED,
        "simulated_epoch_subset_seed": REFERENCE_SEED,
    }


def _cell_rows(cell: pd.DataFrame) -> list[dict[str, Any]]:
    existing = {_point(row) for _, row in cell.iterrows()}
    used = set(existing)
    tied = cell.loc[np.isclose(cell["phase_0_starcoder"], cell["phase_1_starcoder"])]
    untied = cell.loc[~np.isclose(cell["phase_0_starcoder"], cell["phase_1_starcoder"])]
    best_tied = tied.loc[tied["starcoder_bpb"].idxmin()]
    best_untied = untied.loc[untied["starcoder_bpb"].idxmin()]
    primary_aggregate = round(
        _aggregate(float(best_untied["phase_0_starcoder"]), float(best_untied["phase_1_starcoder"])), 6
    )
    primary_normalized = _normalized_contrast(
        float(best_untied["phase_0_starcoder"]), float(best_untied["phase_1_starcoder"])
    )
    secondary_aggregate = _secondary_aggregate(primary_aggregate, float(best_tied["phase_0_starcoder"]))

    rows: list[dict[str, Any]] = []

    # Reserve an identical interior scaffold in every cell before placing adaptive points.
    for aggregate_index, aggregate in enumerate(COMMON_AGGREGATES):
        for contrast_index, normalized in enumerate(COMMON_POSITIVE_NORMALIZED_CONTRASTS):
            p0, p1 = _policy_from_aggregate_normalized_contrast(aggregate, normalized)
            if (p0, p1) in used or _distance((p0, p1), used) < MIN_POLICY_DISTANCE:
                raise ValueError(f"Common positive scaffold collides in {cell.iloc[0]['cell_id']}: {(p0, p1)}")
            rows.append(
                _base_row(
                    cell=cell,
                    acquisition_kind="common_positive",
                    p0=p0,
                    p1=p1,
                    local_index=aggregate_index * len(COMMON_POSITIVE_NORMALIZED_CONTRASTS) + contrast_index + 1,
                    primary_aggregate=primary_aggregate,
                    primary_best_untied=best_untied,
                    primary_best_tied=best_tied,
                    normalized_contrast=normalized,
                )
            )
            used.add((p0, p1))
        p0, p1 = _policy_from_aggregate_normalized_contrast(aggregate, COMMON_NEGATIVE_NORMALIZED_CONTRAST)
        if (p0, p1) in used or _distance((p0, p1), used) < MIN_POLICY_DISTANCE:
            raise ValueError(f"Common negative scaffold collides in {cell.iloc[0]['cell_id']}: {(p0, p1)}")
        rows.append(
            _base_row(
                cell=cell,
                acquisition_kind="common_negative",
                p0=p0,
                p1=p1,
                local_index=aggregate_index + 1,
                primary_aggregate=primary_aggregate,
                primary_best_untied=best_untied,
                primary_best_tied=best_tied,
                normalized_contrast=COMMON_NEGATIVE_NORMALIZED_CONTRAST,
            )
        )
        used.add((p0, p1))

    for acquisition_kind, aggregate in (
        ("primary_tied_anchor", primary_aggregate),
        ("secondary_tied_anchor", secondary_aggregate),
    ):
        tied_anchor = (aggregate, aggregate)
        if tied_anchor in used:
            continue
        rows.append(
            _base_row(
                cell=cell,
                acquisition_kind=acquisition_kind,
                p0=aggregate,
                p1=aggregate,
                local_index=1,
                primary_aggregate=primary_aggregate,
                primary_best_untied=best_untied,
                primary_best_tied=best_tied,
                normalized_contrast=0.0,
            )
        )
        used.add(tied_anchor)

    primary_targets = PRIMARY_BASE_NORMALIZED_CONTRASTS + tuple(
        float(np.clip(primary_normalized + offset, 0.02, 0.98)) for offset in PRIMARY_LOCAL_NORMALIZED_OFFSETS
    )
    for index, target in enumerate(primary_targets, start=1):
        p0, p1, normalized = _nearest_available_on_fiber(
            aggregate=primary_aggregate,
            target_normalized_contrast=float(target),
            used=used,
        )
        rows.append(
            _base_row(
                cell=cell,
                acquisition_kind="primary_fiber",
                p0=p0,
                p1=p1,
                local_index=index,
                primary_aggregate=primary_aggregate,
                primary_best_untied=best_untied,
                primary_best_tied=best_tied,
                normalized_contrast=normalized,
            )
        )
        used.add((p0, p1))

    for index, target in enumerate(SECONDARY_NORMALIZED_CONTRASTS, start=1):
        p0, p1, normalized = _nearest_available_on_fiber(
            aggregate=secondary_aggregate,
            target_normalized_contrast=float(target),
            used=used,
        )
        rows.append(
            _base_row(
                cell=cell,
                acquisition_kind="secondary_fiber",
                p0=p0,
                p1=p1,
                local_index=index,
                primary_aggregate=primary_aggregate,
                primary_best_untied=best_untied,
                primary_best_tied=best_tied,
                normalized_contrast=normalized,
            )
        )
        used.add((p0, p1))

    local_index = 0
    for aggregate_offset in LOCAL_AGGREGATE_OFFSETS:
        for normalized_offset in LOCAL_NORMALIZED_CONTRAST_OFFSETS:
            local_index += 1
            p0, p1, _, normalized = _nearest_available_local_point(
                aggregate=primary_aggregate + aggregate_offset,
                normalized=primary_normalized + normalized_offset,
                used=used,
            )
            rows.append(
                _base_row(
                    cell=cell,
                    acquisition_kind="local_optimum",
                    p0=p0,
                    p1=p1,
                    local_index=local_index,
                    primary_aggregate=primary_aggregate,
                    primary_best_untied=best_untied,
                    primary_best_tied=best_tied,
                    normalized_contrast=normalized,
                )
            )
            used.add((p0, p1))

    untied_rows = [row for row in rows if not row["acquisition_kind"].endswith("tied_anchor")]
    if len(untied_rows) != UNTIED_RUNS_PER_CELL:
        raise ValueError(f"{cell.iloc[0]['cell_id']}: expected {UNTIED_RUNS_PER_CELL} untied rows")
    if any(abs(float(row["phase_contrast"])) < 1e-6 for row in untied_rows):
        raise ValueError(f"{cell.iloc[0]['cell_id']}: an untied allocation is phase-tied")
    return rows


def build_rows(observations: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a systematic aggregate/contrast design independently in each cell."""
    if observations["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError("The completed discovery panel does not contain ten cells")
    counts = observations.groupby("cell_id").size()
    if not counts.eq(EXPECTED_EXISTING_ROWS_PER_CELL).all():
        raise ValueError(f"Unexpected completed rows per cell: {counts.to_dict()}")
    tied_counts = observations.groupby("cell_id").apply(
        lambda frame: int(np.isclose(frame["phase_0_starcoder"], frame["phase_1_starcoder"]).sum()),
        include_groups=False,
    )
    if not tied_counts.eq(EXPECTED_EXISTING_TIED_PER_CELL).all():
        raise ValueError(f"Unexpected existing tied counts: {tied_counts.to_dict()}")
    if not (counts - tied_counts).eq(EXPECTED_EXISTING_UNTIED_PER_CELL).all():
        raise ValueError("The completed panel does not contain eight untied coordinates per cell")

    rows: list[dict[str, Any]] = []
    for _, cell in observations.groupby("cell_id", sort=True):
        rows.extend(_cell_rows(cell))
    if len({str(row["run_name"]) for row in rows}) != len(rows):
        raise ValueError("Stage-3 run names are not unique")
    untied_count = sum(not str(row["acquisition_kind"]).endswith("tied_anchor") for row in rows)
    if untied_count != EXPECTED_UNTIED_RUNS:
        raise ValueError(f"Expected {EXPECTED_UNTIED_RUNS} untied Stage-3 runs, got {untied_count}")
    frame = pd.DataFrame(rows)
    common = frame.loc[frame["acquisition_kind"].isin(("common_positive", "common_negative"))]
    scaffold_variants = common.groupby(["acquisition_kind", "acquisition_index"])[
        ["phase_0_starcoder", "phase_1_starcoder"]
    ].nunique()
    if not scaffold_variants.eq(1).all().all():
        raise ValueError("The cross-cell scaffold is not coordinate-identical across all ten cells")
    scaffold_counts = common.groupby(["acquisition_kind", "acquisition_index"])["cell_id"].nunique()
    if not scaffold_counts.eq(EXPECTED_CELLS).all():
        raise ValueError("The cross-cell scaffold is incomplete")
    return rows


def write_outputs() -> None:
    """Persist the adaptive design, launch manifest, and review table."""
    observations = pd.read_csv(COMBINED_OBSERVATIONS_PATH)
    rows = build_rows(observations)
    frame = pd.DataFrame(rows)
    stage2_design = json.loads(STAGE2_DESIGN_PATH.read_text(encoding="utf-8"))
    source_paths = (
        Path(__file__).resolve(),
        COMBINED_OBSERVATIONS_PATH,
        STAGE2_DESIGN_PATH,
        STAGE3_LAUNCHER_PATH,
        STAGE3_ANALYZER_PATH,
    )
    per_cell = (
        frame.groupby(["cell_id", "acquisition_kind"], as_index=False)
        .size()
        .pivot(index="cell_id", columns="acquisition_kind", values="size")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    payload = {
        "design_version": DESIGN_VERSION,
        "description": "Systematic 10-cell WSD80 surface expansion in aggregate/contrast coordinates.",
        "objective_metric": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
        "phase_0_fraction": PHASE_0_FRACTION,
        "expected_run_count": len(rows),
        "expected_untied_run_count": EXPECTED_UNTIED_RUNS,
        "cell_count": EXPECTED_CELLS,
        "untied_runs_per_cell": UNTIED_RUNS_PER_CELL,
        "data_use": {
            "completed_discovery_rows": len(observations),
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
        },
        "training_environment": {
            "tpu_type": "v5p-8",
            "tpu_region": "us-central1",
            "tpu_zone": "us-central1-a",
            "marin_prefix": "gs://marin-us-central1",
        },
        "allocation": {
            "primary_fiber": (
                "Sixteen late-code points combining broad coverage with dense local bracketing on the exact "
                "aggregate fiber through the lowest observed untied coordinate."
            ),
            "secondary_fiber": (
                "Eight late-code points on the best observed tied aggregate, unless that coincides with the primary "
                "fiber, in which case a deterministic lower aggregate is used."
            ),
            "common_scaffold": (
                "Sixteen interior coordinates shared exactly across all cells: three positive contrasts and one "
                "negative contrast at each of four aggregate levels."
            ),
            "local_optimum": (
                "Eight off-fiber points around the observed untied minimum estimate local aggregate-contrast "
                "curvature and their interaction."
            ),
            "tied_anchors": (
                "Exact tied controls at primary and secondary fiber aggregates are added only when absent; they are "
                "additional to the 48 untied coordinates."
            ),
            "excluded_boundary_guards": (
                "No maximin corner guards are used because the observed boundaries are heteroskedastic and not "
                "decision-relevant; the fixed interior scaffold supplies basin protection instead."
            ),
            "per_cell": per_cell.to_dict(orient="records"),
        },
        "design_provenance": (
            "This adaptive discovery panel was designed after observing the 230-row Stage-1-plus-Stage-2 panel. "
            "The primary fiber in each cell is anchored by that cell's lowest observed untied coordinate. No "
            "Stage-3 outcome was available when the coordinates and allocation were frozen. All spatial rows use "
            "the original common reference training stream; fresh-seed confirmation remains a separate inference "
            "stage."
        ),
        "noise_policy": (
            "All surface coordinates use one common reference stream to isolate spatial response. Fresh-seed "
            "replication is reserved for separately frozen selected-policy confirmations, including the active "
            "eight-seed high-TPP comparison, rather than reducing unique surface coverage."
        ),
        "interpretation_boundary": (
            "The panel provides structured surface resolution and a discrete optimum search, not proof of a "
            "continuous global optimum. Reference-seed minima are selection-biased and cannot establish a phase "
            "advantage without fresh same-seed candidate/comparator confirmation."
        ),
        "followup_boundary": {
            "selection": (
                "After complete durable coverage, fit the preregistered degree-four ridge surface in aggregate and "
                "raw phase contrast separately within each cell. Ridge strength is selected by the frozen "
                "five-fold spatial-CV rule, and the candidate is the fitted minimum inside the empirical convex "
                "hull with absolute phase contrast at least 0.04. Compare it with the same fitted surface restricted "
                "to the tied diagonal. Raw discrete minima remain descriptive and cannot select a confirmation "
                "candidate."
            ),
            "promotion": (
                "A cell is eligible for a fresh-seed confirmation only if the fitted tied-minus-untied gain is at "
                "least 0.005 BPB and at least 80 percent of the frozen residual-bootstrap fits preserve a positive "
                "gain. The reference-seed panel is discovery only; confirmation design and multiplicity control "
                "must be frozen before any fresh-seed launch. Bootstrap residuals are leverage-corrected; ridge "
                "selection is held fixed across bootstrap replicates, so this diagnostic omits ridge-selection "
                "uncertainty."
            ),
            "scaling_estimator": (
                "The primary cross-cell scaling estimate uses only the 16 byte-identical scaffold coordinates and "
                "the tied coordinates shared by all ten cells. It subtracts a linearly interpolated common-grid tied "
                "profile at each scaffold aggregate, then fits coordinate fixed effects plus log-N, log-D, and "
                "log-N-by-log-D terms with cell-clustered standard errors. Optima from the cell-adaptive fibers and "
                "rings are secondary, explicitly selection-biased sensitivity analyses."
            ),
            "claim_limit": (
                "A passing confirmation supports only the selected discrete comparison. Surface and scaling-law "
                "claims must report coordinate resolution and bootstrap/seed uncertainty."
            ),
        },
        "design": {"launch_manifest_sha256": stream_identity.canonical_sha256(launch_manifest(rows))},
        "source_cells": stage2_design["source_cells"],
        "runs": rows,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_DESIGN_PATH.write_text(serialized, encoding="utf-8")
    (OUTPUT_DIR / "design_manifest.json").write_text(serialized, encoding="utf-8")
    frame.to_csv(OUTPUT_DIR / "run_manifest.csv", index=False)
    report = [
        "# StarCoder WSD80 matched-N,D Stage-3 dense-surface design",
        "",
        f"- New runs: {len(frame)} ({EXPECTED_UNTIED_RUNS} untied plus "
        f"{int(frame['acquisition_kind'].str.endswith('tied_anchor').sum())} missing exact tied anchors).",
        f"- Per cell: {UNTIED_RUNS_PER_CELL} untied coordinates.",
        "- Parameterization: aggregate a = 0.8 p0 + 0.2 p1 and contrast d = p1 - p0.",
        "- Every row uses the original reference seed for both model/data order and simulated-epoch subset.",
        "",
        "## Allocation",
        "",
        per_cell.to_markdown(index=False),
        "",
        "## Primary and secondary fibers",
        "",
        frame.loc[
            frame["acquisition_kind"].isin(
                ("primary_tied_anchor", "primary_fiber", "secondary_tied_anchor", "secondary_fiber")
            )
        ][
            [
                "cell_id",
                "acquisition_kind",
                "phase_0_starcoder",
                "phase_1_starcoder",
                "aggregate_starcoder",
                "phase_contrast",
                "normalized_phase_contrast",
            ]
        ].to_markdown(
            index=False, floatfmt=".6f"
        ),
        "",
        "## Interpretation boundary",
        "",
        str(payload["interpretation_boundary"]),
        "",
        "## Frozen follow-up boundary",
        "",
        f"- Selection: {payload['followup_boundary']['selection']}",
        f"- Promotion: {payload['followup_boundary']['promotion']}",
        f"- Claim limit: {payload['followup_boundary']['claim_limit']}",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
