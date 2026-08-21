# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "wandb",
# ]
# ///
"""Collect and draw the measured StarCoder 80/20 WSD fixed-aggregate fiber.

``analyze_starcoder_wsd_80_20_surface.py`` owns the figure, but its ``main`` rewrites every artifact
in the output directory. This focused reproducer collects the completed 31-point fiber and its
five-coordinate repeat panel, merges only unique reference-seed coordinates into the persisted
surface, and redraws the 3D artifact.

The trace is the constant-aggregate line through the best sampled tied mixture. At an 80/20 split
its slope is -4, because a token removed from the long phase must be repaid four times over in the
short one. The surface uses one reference-seed value per coordinate; repeats remain a separate
uncertainty layer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_starcoder_wsd_80_20_surface import (  # noqa: E402
    PHASE_0_FRACTION,
    REFERENCE_DATA_SEED,
    WANDB_PATH,
    WSD80_TARGET,
    Surface,
    _render_wsd80_surface,
    _surface_frame,
)

DEFAULT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "starcoder_wsd80_surface_refined_20260714"
FIBER_OBSERVATIONS = "wsd80_fixed_aggregate_fiber_observations.csv"
FIBER_SUMMARY = "wsd80_fixed_aggregate_fiber_summary.json"
EXPECTED_PRIOR_COORDINATES = 107
EXPECTED_UPDATED_COORDINATES = 136
EXPECTED_FINAL_COORDINATES = 166
EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES = 346
FIBER_WANDB_GROUP = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_fixedagg03_fiber31_repeat5x4_20260727"
FIBER_PANEL_TAG = "fixedagg03_fiber31_repeat5x4"
TIED_STARCODER_WEIGHT = 0.30
NUM_FIBER_POINTS = 31
TIED_FIBER_INDEX = 9
REPEATED_FIBER_INDICES = (3, 6, TIED_FIBER_INDEX, 12, 15)
REUSED_REFERENCE_INDEX = 15
FIRST_SELECTION_RANK = 109
NUM_WANDB_RUNS = 50
COORDINATE_TOLERANCE = 1e-12
REFERENCE_RUN_PATTERN = re.compile(r"fiber31_i(?P<index>\d{2})_p0_.*")
REPEAT_RUN_PATTERN = re.compile(r"fiberrep_i(?P<index>\d{2})_seed(?P<seed>\d+)")
OPTIMUM_FIBER_OBSERVATIONS = "wsd80_global_optimum_fiber_observations.csv"
OPTIMUM_FIBER_SUMMARY = "wsd80_global_optimum_fiber_summary.json"
COMBINED_FIBER_OBSERVATIONS = "wsd80_measured_fiber_observations.csv"
OPTIMUM_FIBER_WANDB_GROUP = (
    "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_fixedagg018_optfiber32_repeat6x4_20260728"
)
OPTIMUM_FIBER_PANEL_TAG = "fixedagg018_optfiber32_repeat6x4"
OPTIMUM_TIED_WEIGHT = 0.18
OPTIMUM_NUM_FIBER_POINTS = 32
OPTIMUM_PHASE_1_STEP = 0.03
OPTIMUM_OBSERVED_COORDINATE = (0.10, 0.50)
OPTIMUM_REPEATED_PHASE_1_WEIGHTS = (0.09, 0.15, 0.18, 0.21, 0.27, 0.50)
OPTIMUM_TIED_INDEX = 6
OPTIMUM_REPEATED_FIBER_INDICES = (3, 5, OPTIMUM_TIED_INDEX, 7, 9, 17)
OPTIMUM_REUSED_REFERENCE_INDICES = (17, 31)
OPTIMUM_FIRST_SELECTION_RANK = 160
OPTIMUM_NUM_WANDB_RUNS = 54
OPTIMUM_REFERENCE_RUN_PATTERN = re.compile(r"optfiber_i(?P<index>\d{2})_p0_.*")
OPTIMUM_REPEAT_RUN_PATTERN = re.compile(r"optfiberrep_i(?P<index>\d{2})_seed(?P<seed>\d+)")
HIGH_AGGREGATE_FIBER_OBSERVATIONS = "wsd80_high_aggregate_fiber_observations.csv"
HIGH_AGGREGATE_FIBER_SUMMARY = "wsd80_high_aggregate_fiber_summary.json"
HIGH_AGGREGATE_WEIGHTS = (0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
HIGH_AGGREGATE_REUSED_COORDINATES = (
    (0.40, 0.40),
    (0.50, 0.50),
    (0.60, 0.60),
    (0.70, 0.70),
    (0.80, 0.80),
    (1.00, 0.00),
)
HIGH_AGGREGATE_FIRST_SELECTION_RANK = 260
HIGH_AGGREGATE_POINTS_PER_FIBER = 31
HIGH_AGGREGATE_WANDB_GROUP = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_highagg_fibers6x31_20260728"
HIGH_AGGREGATE_PANEL_TAG = "highagg_fibers6x31"
HIGH_AGGREGATE_NUM_WANDB_RUNS = 180
HIGH_AGGREGATE_RUN_PATTERN = re.compile(r"highagg_a(?P<aggregate>\d+p\d+)_i(?P<index>\d{2})_p0_.*")
SURFACE_COLUMNS_WITH_NAN = (
    "cosine_bpb",
    "wsd_50_50_bpb",
    "insertion_distance",
    "old_row_index",
)


def _optimum_fiber_coordinates() -> tuple[tuple[float, float], ...]:
    """Return the slope -4 aggregate-0.18 fiber used by the completed launcher."""
    coordinates = [
        (
            (OPTIMUM_TIED_WEIGHT - (1.0 - PHASE_0_FRACTION) * (index * OPTIMUM_PHASE_1_STEP)) / PHASE_0_FRACTION,
            index * OPTIMUM_PHASE_1_STEP,
        )
        for index in range(31)
    ]
    if OPTIMUM_OBSERVED_COORDINATE not in coordinates:
        coordinates.append(OPTIMUM_OBSERVED_COORDINATE)
    return tuple(sorted(coordinates, key=lambda coordinate: coordinate[1]))


def _parse_weight_slug(value: str) -> float:
    return float(value.replace("p", "."))


def _coordinates_match(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1])) <= COORDINATE_TOLERANCE


def high_aggregate_coordinates(aggregate: float) -> tuple[tuple[float, float], ...]:
    phase_1_weights = [index / (HIGH_AGGREGATE_POINTS_PER_FIBER - 1) for index in range(HIGH_AGGREGATE_POINTS_PER_FIBER)]
    if not any(abs(weight - aggregate) <= COORDINATE_TOLERANCE for weight in phase_1_weights):
        nearest_index = min(
            range(len(phase_1_weights)),
            key=lambda index: abs(phase_1_weights[index] - aggregate),
        )
        phase_1_weights[nearest_index] = aggregate
    return tuple(
        (
            (aggregate - (1.0 - PHASE_0_FRACTION) * phase_1_weight) / PHASE_0_FRACTION,
            phase_1_weight,
        )
        for phase_1_weight in sorted(phase_1_weights)
    )


def _fixed_aggregate_coordinates() -> tuple[tuple[float, float], ...]:
    """Return the 31 coordinates materialized by the training launcher."""
    coordinates = []
    for index in range(NUM_FIBER_POINTS):
        phase_1_starcoder = index / (NUM_FIBER_POINTS - 1)
        phase_0_starcoder = (TIED_STARCODER_WEIGHT - (1.0 - PHASE_0_FRACTION) * phase_1_starcoder) / PHASE_0_FRACTION
        coordinates.append((phase_0_starcoder, phase_1_starcoder))
    return tuple(coordinates)


def _completed_run_row(run: object, fiber_index: int, data_seed: int, source: str) -> dict[str, object]:
    target = run.summary.get(WSD80_TARGET)
    if run.state != "finished" or target is None:
        raise ValueError(f"Incomplete W&B run {run.name}: state={run.state}, target={target}")
    phase_0, phase_1 = _fixed_aggregate_coordinates()[fiber_index]
    return {
        "fiber_index": fiber_index,
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder_share_80_20": PHASE_0_FRACTION * phase_0 + (1.0 - PHASE_0_FRACTION) * phase_1,
        "ordering_contrast_p1_minus_p0": phase_1 - phase_0,
        "data_seed": data_seed,
        "wsd80_bpb": float(target),
        "eval_loss": float(run.summary["eval/loss"]),
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_url": run.url,
        "wandb_state": run.state,
        "wandb_group": FIBER_WANDB_GROUP,
        "source": source,
    }


def _reused_reference_row(surface_metrics: pd.DataFrame) -> dict[str, object]:
    fiber_index = REUSED_REFERENCE_INDEX
    phase_0, phase_1 = _fixed_aggregate_coordinates()[fiber_index]
    matched = surface_metrics.loc[
        np.isclose(surface_metrics["phase_0_starcoder"], phase_0, atol=COORDINATE_TOLERANCE)
        & np.isclose(surface_metrics["phase_1_starcoder"], phase_1, atol=COORDINATE_TOLERANCE)
    ]
    if len(matched) != 1:
        raise ValueError(f"Expected one reusable surface coordinate at fiber index {fiber_index}, found {len(matched)}")
    row = matched.iloc[0]
    return {
        "fiber_index": fiber_index,
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder_share_80_20": PHASE_0_FRACTION * phase_0 + (1.0 - PHASE_0_FRACTION) * phase_1,
        "ordering_contrast_p1_minus_p0": phase_1 - phase_0,
        "data_seed": REFERENCE_DATA_SEED,
        "wsd80_bpb": float(row["wsd80_bpb"]),
        "eval_loss": float(row["eval_loss"]),
        "wandb_run_id": row["wandb_run_id"],
        "wandb_run_name": row["wandb_run_name"],
        "wandb_url": row["wandb_url"],
        "wandb_state": row["wandb_state"],
        "wandb_group": "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_20_refinement44_20260714",
        "source": "reused prior reference-seed coordinate",
    }


def _collect_fiber_observations(surface_metrics: pd.DataFrame) -> pd.DataFrame:
    runs = list(
        wandb.Api(timeout=90).runs(
            WANDB_PATH,
            filters={"group": FIBER_WANDB_GROUP},
            per_page=100,
        )
    )
    if len(runs) != NUM_WANDB_RUNS:
        raise ValueError(f"Expected {NUM_WANDB_RUNS} W&B runs in {FIBER_WANDB_GROUP!r}, found {len(runs)}")

    rows: list[dict[str, object]] = []
    for run in runs:
        reference_match = REFERENCE_RUN_PATTERN.fullmatch(run.name)
        repeat_match = REPEAT_RUN_PATTERN.fullmatch(run.name)
        if reference_match is not None:
            fiber_index = int(reference_match.group("index"))
            data_seed = int(run.config["data_seed"])
            if data_seed != REFERENCE_DATA_SEED:
                raise ValueError(f"Reference fiber run {run.name} has unexpected data seed {data_seed}")
            rows.append(_completed_run_row(run, fiber_index, data_seed, "new reference-seed fiber run"))
            continue
        if repeat_match is not None:
            fiber_index = int(repeat_match.group("index"))
            data_seed = int(repeat_match.group("seed"))
            if data_seed != int(run.config["data_seed"]):
                raise ValueError(f"Repeat run name/config seed mismatch for {run.name}")
            rows.append(_completed_run_row(run, fiber_index, data_seed, "matched fiber repeat"))
            continue
        raise ValueError(f"Unexpected fixed-aggregate fiber run name: {run.name!r}")

    rows.append(_reused_reference_row(surface_metrics))
    observations = pd.DataFrame(rows).sort_values(["fiber_index", "data_seed"]).reset_index(drop=True)
    reference = observations.loc[observations["data_seed"].eq(REFERENCE_DATA_SEED)]
    expected_indices = set(range(NUM_FIBER_POINTS))
    if len(reference) != NUM_FIBER_POINTS or set(reference["fiber_index"]) != expected_indices:
        raise ValueError("Reference-seed fiber does not cover all 31 coordinates")
    expected_counts = pd.Series(1, index=range(NUM_FIBER_POINTS), dtype=int)
    expected_counts.loc[list(REPEATED_FIBER_INDICES)] = 5
    actual_counts = observations.groupby("fiber_index").size().reindex(expected_counts.index, fill_value=0)
    if not actual_counts.equals(expected_counts):
        raise ValueError(f"Unexpected per-coordinate observation counts: {actual_counts.to_dict()}")
    if not np.allclose(
        observations["aggregate_starcoder_share_80_20"],
        TIED_STARCODER_WEIGHT,
        atol=COORDINATE_TOLERANCE,
    ):
        raise ValueError("A collected point leaves the fixed-aggregate fiber")
    return observations


def _optimum_completed_run_row(run: object, fiber_index: int, data_seed: int, source: str) -> dict[str, object]:
    target = run.summary.get(WSD80_TARGET)
    if run.state != "finished" or target is None:
        raise ValueError(f"Incomplete W&B run {run.name}: state={run.state}, target={target}")
    phase_0, phase_1 = _optimum_fiber_coordinates()[fiber_index]
    return {
        "fiber_index": fiber_index,
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder_share_80_20": PHASE_0_FRACTION * phase_0 + (1.0 - PHASE_0_FRACTION) * phase_1,
        "ordering_contrast_p1_minus_p0": phase_1 - phase_0,
        "data_seed": data_seed,
        "wsd80_bpb": float(target),
        "eval_loss": float(run.summary["eval/loss"]),
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_url": run.url,
        "wandb_state": run.state,
        "wandb_group": OPTIMUM_FIBER_WANDB_GROUP,
        "source": source,
    }


def _optimum_reused_reference_rows(surface_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    coordinates = _optimum_fiber_coordinates()
    for fiber_index in OPTIMUM_REUSED_REFERENCE_INDICES:
        phase_0, phase_1 = coordinates[fiber_index]
        matched = surface_metrics.loc[
            np.isclose(surface_metrics["phase_0_starcoder"], phase_0, atol=COORDINATE_TOLERANCE)
            & np.isclose(surface_metrics["phase_1_starcoder"], phase_1, atol=COORDINATE_TOLERANCE)
        ]
        if len(matched) != 1:
            raise ValueError(
                f"Expected one reusable optimum-fiber coordinate at index {fiber_index}, found {len(matched)}"
            )
        row = matched.iloc[0]
        rows.append(
            {
                "fiber_index": fiber_index,
                "phase_0_starcoder": phase_0,
                "phase_1_starcoder": phase_1,
                "aggregate_starcoder_share_80_20": PHASE_0_FRACTION * phase_0 + (1.0 - PHASE_0_FRACTION) * phase_1,
                "ordering_contrast_p1_minus_p0": phase_1 - phase_0,
                "data_seed": REFERENCE_DATA_SEED,
                "wsd80_bpb": float(row["wsd80_bpb"]),
                "eval_loss": float(row["eval_loss"]),
                "wandb_run_id": row["wandb_run_id"],
                "wandb_run_name": row["wandb_run_name"],
                "wandb_url": row["wandb_url"],
                "wandb_state": row["wandb_state"],
                "wandb_group": row["wandb_group"] if "wandb_group" in row and pd.notna(row["wandb_group"]) else "",
                "source": "reused prior reference-seed coordinate",
            }
        )
    return rows


def _collect_optimum_fiber_observations(surface_metrics: pd.DataFrame) -> pd.DataFrame:
    coordinates = _optimum_fiber_coordinates()
    if len(coordinates) != OPTIMUM_NUM_FIBER_POINTS:
        raise ValueError(f"Expected {OPTIMUM_NUM_FIBER_POINTS} optimum-fiber coordinates, found {len(coordinates)}")
    for phase_1_weight, expected_index in zip(
        OPTIMUM_REPEATED_PHASE_1_WEIGHTS,
        OPTIMUM_REPEATED_FIBER_INDICES,
        strict=True,
    ):
        if not np.isclose(coordinates[expected_index][1], phase_1_weight, atol=COORDINATE_TOLERANCE):
            raise ValueError(f"Repeat index {expected_index} does not match phase-1 weight {phase_1_weight}")

    runs = list(
        wandb.Api(timeout=90).runs(
            WANDB_PATH,
            filters={"group": OPTIMUM_FIBER_WANDB_GROUP},
            per_page=100,
        )
    )
    if len(runs) != OPTIMUM_NUM_WANDB_RUNS:
        raise ValueError(f"Expected {OPTIMUM_NUM_WANDB_RUNS} W&B runs, found {len(runs)}")

    rows: list[dict[str, object]] = []
    for run in runs:
        reference_match = OPTIMUM_REFERENCE_RUN_PATTERN.fullmatch(run.name)
        repeat_match = OPTIMUM_REPEAT_RUN_PATTERN.fullmatch(run.name)
        if reference_match is not None:
            fiber_index = int(reference_match.group("index"))
            data_seed = int(run.config["data_seed"])
            if data_seed != REFERENCE_DATA_SEED:
                raise ValueError(f"Reference optimum-fiber run {run.name} has unexpected data seed {data_seed}")
            rows.append(_optimum_completed_run_row(run, fiber_index, data_seed, "new reference-seed optimum fiber"))
            continue
        if repeat_match is not None:
            fiber_index = int(repeat_match.group("index"))
            data_seed = int(repeat_match.group("seed"))
            if data_seed != int(run.config["data_seed"]):
                raise ValueError(f"Repeat run name/config seed mismatch for {run.name}")
            rows.append(_optimum_completed_run_row(run, fiber_index, data_seed, "matched optimum-fiber repeat"))
            continue
        raise ValueError(f"Unexpected optimum-fiber run name: {run.name!r}")

    rows.extend(_optimum_reused_reference_rows(surface_metrics))
    observations = pd.DataFrame(rows).sort_values(["fiber_index", "data_seed"]).reset_index(drop=True)
    reference = observations.loc[observations["data_seed"].eq(REFERENCE_DATA_SEED)]
    expected_indices = set(range(OPTIMUM_NUM_FIBER_POINTS))
    if len(reference) != OPTIMUM_NUM_FIBER_POINTS or set(reference["fiber_index"]) != expected_indices:
        raise ValueError("Reference-seed optimum fiber does not cover all coordinates")
    expected_counts = pd.Series(1, index=range(OPTIMUM_NUM_FIBER_POINTS), dtype=int)
    expected_counts.loc[list(OPTIMUM_REPEATED_FIBER_INDICES)] = 5
    actual_counts = observations.groupby("fiber_index").size().reindex(expected_counts.index, fill_value=0)
    if not actual_counts.equals(expected_counts):
        raise ValueError(f"Unexpected optimum-fiber observation counts: {actual_counts.to_dict()}")
    if not np.allclose(
        observations["aggregate_starcoder_share_80_20"],
        OPTIMUM_TIED_WEIGHT,
        atol=COORDINATE_TOLERANCE,
    ):
        raise ValueError("A collected optimum-fiber point leaves the fixed aggregate")
    return observations


def _high_aggregate_completed_run_row(
    run: object,
    aggregate: float,
    fiber_index: int,
    source: str,
) -> dict[str, object]:
    target = run.summary.get(WSD80_TARGET)
    if run.state != "finished" or target is None:
        raise ValueError(f"Incomplete W&B run {run.name}: state={run.state}, target={target}")
    phase_0, phase_1 = high_aggregate_coordinates(aggregate)[fiber_index]
    return {
        "fiber_index": fiber_index,
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder_share_80_20": aggregate,
        "ordering_contrast_p1_minus_p0": phase_1 - phase_0,
        "data_seed": int(run.config["data_seed"]),
        "wsd80_bpb": float(target),
        "eval_loss": float(run.summary["eval/loss"]),
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_url": run.url,
        "wandb_state": run.state,
        "wandb_group": HIGH_AGGREGATE_WANDB_GROUP,
        "source": source,
    }


def _high_aggregate_reused_rows(surface_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for aggregate in HIGH_AGGREGATE_WEIGHTS:
        for fiber_index, (phase_0, phase_1) in enumerate(high_aggregate_coordinates(aggregate)):
            if not any(_coordinates_match((phase_0, phase_1), reused) for reused in HIGH_AGGREGATE_REUSED_COORDINATES):
                continue
            matched = surface_metrics.loc[
                np.isclose(surface_metrics["phase_0_starcoder"], phase_0, atol=COORDINATE_TOLERANCE)
                & np.isclose(surface_metrics["phase_1_starcoder"], phase_1, atol=COORDINATE_TOLERANCE)
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"Expected one reusable high-aggregate coordinate at {(phase_0, phase_1)}, found {len(matched)}"
                )
            row = matched.iloc[0]
            rows.append(
                {
                    "fiber_index": fiber_index,
                    "phase_0_starcoder": phase_0,
                    "phase_1_starcoder": phase_1,
                    "aggregate_starcoder_share_80_20": aggregate,
                    "ordering_contrast_p1_minus_p0": phase_1 - phase_0,
                    "data_seed": REFERENCE_DATA_SEED,
                    "wsd80_bpb": float(row["wsd80_bpb"]),
                    "eval_loss": float(row["eval_loss"]),
                    "wandb_run_id": row["wandb_run_id"],
                    "wandb_run_name": row["wandb_run_name"],
                    "wandb_url": row["wandb_url"],
                    "wandb_state": row["wandb_state"],
                    "wandb_group": row["wandb_group"] if "wandb_group" in row and pd.notna(row["wandb_group"]) else "",
                    "source": "reused prior reference-seed coordinate",
                }
            )
    if len(rows) != len(HIGH_AGGREGATE_REUSED_COORDINATES):
        raise ValueError(f"Expected {len(HIGH_AGGREGATE_REUSED_COORDINATES)} reused rows, found {len(rows)}")
    return rows


def _collect_high_aggregate_observations(surface_metrics: pd.DataFrame) -> pd.DataFrame:
    runs = list(
        wandb.Api(timeout=90).runs(
            WANDB_PATH,
            filters={"group": HIGH_AGGREGATE_WANDB_GROUP},
            per_page=250,
        )
    )
    if len(runs) != HIGH_AGGREGATE_NUM_WANDB_RUNS:
        raise ValueError(f"Expected {HIGH_AGGREGATE_NUM_WANDB_RUNS} W&B runs, found {len(runs)}")

    rows: list[dict[str, object]] = []
    for run in runs:
        match = HIGH_AGGREGATE_RUN_PATTERN.fullmatch(run.name)
        if match is None:
            raise ValueError(f"Unexpected high-aggregate fiber run name: {run.name!r}")
        aggregate = _parse_weight_slug(match.group("aggregate"))
        if not any(np.isclose(aggregate, expected, atol=COORDINATE_TOLERANCE) for expected in HIGH_AGGREGATE_WEIGHTS):
            raise ValueError(f"Unexpected aggregate {aggregate} in run {run.name}")
        fiber_index = int(match.group("index"))
        data_seed = int(run.config["data_seed"])
        if data_seed != REFERENCE_DATA_SEED:
            raise ValueError(f"High-aggregate run {run.name} has unexpected data seed {data_seed}")
        rows.append(
            _high_aggregate_completed_run_row(
                run,
                aggregate,
                fiber_index,
                "new reference-seed high-aggregate fiber run",
            )
        )

    rows.extend(_high_aggregate_reused_rows(surface_metrics))
    observations = (
        pd.DataFrame(rows).sort_values(["aggregate_starcoder_share_80_20", "fiber_index"]).reset_index(drop=True)
    )
    if len(observations) != len(HIGH_AGGREGATE_WEIGHTS) * HIGH_AGGREGATE_POINTS_PER_FIBER:
        raise ValueError(f"Expected 186 high-aggregate observations, found {len(observations)}")
    for aggregate, block in observations.groupby("aggregate_starcoder_share_80_20"):
        if len(block) != HIGH_AGGREGATE_POINTS_PER_FIBER:
            raise ValueError(f"Aggregate {aggregate:.2f} has {len(block)} coordinates")
        if set(block["fiber_index"]) != set(range(HIGH_AGGREGATE_POINTS_PER_FIBER)):
            raise ValueError(f"Aggregate {aggregate:.2f} does not cover every fiber index")
        if not np.allclose(
            block["aggregate_starcoder_share_80_20"],
            aggregate,
            atol=COORDINATE_TOLERANCE,
        ):
            raise ValueError(f"Aggregate {aggregate:.2f} leaves its fixed-aggregate fiber")
        tied = block.loc[np.isclose(block["phase_0_starcoder"], block["phase_1_starcoder"], atol=COORDINATE_TOLERANCE)]
        if len(tied) != 1:
            raise ValueError(f"Aggregate {aggregate:.2f} must contain exactly one tied control")
    return observations


def _surface_rows(observations: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    reference = observations.loc[observations["data_seed"].eq(REFERENCE_DATA_SEED)].copy()
    rows = pd.DataFrame(index=reference.index, columns=columns)
    rows["selection_rank"] = FIRST_SELECTION_RANK + reference["fiber_index"]
    rows["phase_0_starcoder"] = reference["phase_0_starcoder"]
    rows["phase_1_starcoder"] = reference["phase_1_starcoder"]
    rows["aggregate_starcoder_share_80_20"] = reference["aggregate_starcoder_share_80_20"]
    rows["ordering_contrast_p1_minus_p0"] = reference["ordering_contrast_p1_minus_p0"]
    rows["source"] = "fixed_aggregate_fiber"
    rows["forced"] = True
    rows["forced_reasons"] = "fixed_aggregate_fiber"
    for column in SURFACE_COLUMNS_WITH_NAN:
        if column in rows.columns:
            rows[column] = np.nan
    for column in ("wandb_run_id", "wandb_run_name", "wandb_url", "wandb_state", "wsd80_bpb", "eval_loss"):
        rows[column] = reference[column]
    rows["panel"] = FIBER_PANEL_TAG
    return rows


def _optimum_surface_rows(observations: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    reference = observations.loc[observations["data_seed"].eq(REFERENCE_DATA_SEED)].copy()
    rows = pd.DataFrame(index=reference.index, columns=columns)
    rows["selection_rank"] = OPTIMUM_FIRST_SELECTION_RANK + reference["fiber_index"]
    rows["phase_0_starcoder"] = reference["phase_0_starcoder"]
    rows["phase_1_starcoder"] = reference["phase_1_starcoder"]
    rows["aggregate_starcoder_share_80_20"] = reference["aggregate_starcoder_share_80_20"]
    rows["ordering_contrast_p1_minus_p0"] = reference["ordering_contrast_p1_minus_p0"]
    rows["source"] = "global_optimum_fixed_aggregate_fiber"
    rows["forced"] = True
    rows["forced_reasons"] = "global_optimum_fixed_aggregate_fiber"
    for column in SURFACE_COLUMNS_WITH_NAN:
        if column in rows.columns:
            rows[column] = np.nan
    for column in ("wandb_run_id", "wandb_run_name", "wandb_url", "wandb_state", "wsd80_bpb", "eval_loss"):
        rows[column] = reference[column]
    rows["panel"] = OPTIMUM_FIBER_PANEL_TAG
    return rows


def _high_aggregate_surface_rows(observations: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    rows = pd.DataFrame(index=observations.index, columns=columns)
    aggregate_positions = {aggregate: position for position, aggregate in enumerate(HIGH_AGGREGATE_WEIGHTS)}
    rows["selection_rank"] = [
        HIGH_AGGREGATE_FIRST_SELECTION_RANK
        + aggregate_positions[float(aggregate)] * HIGH_AGGREGATE_POINTS_PER_FIBER
        + int(fiber_index)
        for aggregate, fiber_index in zip(
            observations["aggregate_starcoder_share_80_20"],
            observations["fiber_index"],
            strict=True,
        )
    ]
    rows["phase_0_starcoder"] = observations["phase_0_starcoder"]
    rows["phase_1_starcoder"] = observations["phase_1_starcoder"]
    rows["aggregate_starcoder_share_80_20"] = observations["aggregate_starcoder_share_80_20"]
    rows["ordering_contrast_p1_minus_p0"] = observations["ordering_contrast_p1_minus_p0"]
    rows["source"] = "high_aggregate_fixed_aggregate_fiber"
    rows["forced"] = True
    rows["forced_reasons"] = "high_aggregate_fixed_aggregate_fiber"
    for column in SURFACE_COLUMNS_WITH_NAN:
        if column in rows.columns:
            rows[column] = np.nan
    for column in ("wandb_run_id", "wandb_run_name", "wandb_url", "wandb_state", "wsd80_bpb", "eval_loss"):
        rows[column] = observations[column]
    rows["panel"] = HIGH_AGGREGATE_PANEL_TAG
    return rows


def _merge_surface_metrics(surface_metrics: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if len(surface_metrics) not in {
        EXPECTED_PRIOR_COORDINATES,
        EXPECTED_UPDATED_COORDINATES,
        EXPECTED_FINAL_COORDINATES,
        EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES,
    }:
        raise ValueError(
            f"Expected {EXPECTED_PRIOR_COORDINATES}, {EXPECTED_UPDATED_COORDINATES}, "
            f"{EXPECTED_FINAL_COORDINATES}, or {EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES} coordinates"
        )
    combined = pd.concat([surface_metrics, _surface_rows(observations, surface_metrics.columns)], ignore_index=True)
    key_columns = ["_phase_0_key", "_phase_1_key"]
    combined[key_columns[0]] = combined["phase_0_starcoder"].round(12)
    combined[key_columns[1]] = combined["phase_1_starcoder"].round(12)
    for _coordinate, block in combined.groupby(key_columns):
        if len(block) > 1 and block["wsd80_bpb"].max() - block["wsd80_bpb"].min() > 1e-12:
            raise ValueError(f"Conflicting reference-seed values at {_coordinate}: {block['wsd80_bpb'].tolist()}")
    merged = (
        combined.drop_duplicates(key_columns, keep="first")
        .drop(columns=key_columns)
        .sort_values(["phase_0_starcoder", "phase_1_starcoder"])
        .reset_index(drop=True)
    )
    expected_coordinates = {
        EXPECTED_PRIOR_COORDINATES: EXPECTED_UPDATED_COORDINATES,
        EXPECTED_UPDATED_COORDINATES: EXPECTED_UPDATED_COORDINATES,
        EXPECTED_FINAL_COORDINATES: EXPECTED_FINAL_COORDINATES,
        EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES: EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES,
    }[len(surface_metrics)]
    if len(merged) != expected_coordinates:
        raise ValueError(f"Expected {expected_coordinates} unique coordinates, found {len(merged)}")
    return merged


def _merge_optimum_surface_metrics(surface_metrics: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if len(surface_metrics) not in {
        EXPECTED_UPDATED_COORDINATES,
        EXPECTED_FINAL_COORDINATES,
        EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES,
    }:
        raise ValueError(
            f"Expected {EXPECTED_UPDATED_COORDINATES}, {EXPECTED_FINAL_COORDINATES}, "
            f"or {EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES} coordinates"
        )
    combined = pd.concat(
        [surface_metrics, _optimum_surface_rows(observations, surface_metrics.columns)],
        ignore_index=True,
    )
    key_columns = ["_phase_0_key", "_phase_1_key"]
    combined[key_columns[0]] = combined["phase_0_starcoder"].round(12)
    combined[key_columns[1]] = combined["phase_1_starcoder"].round(12)
    for coordinate, block in combined.groupby(key_columns):
        if len(block) > 1 and block["wsd80_bpb"].max() - block["wsd80_bpb"].min() > 1e-12:
            raise ValueError(f"Conflicting reference-seed values at {coordinate}: {block['wsd80_bpb'].tolist()}")
    merged = (
        combined.drop_duplicates(key_columns, keep="first")
        .drop(columns=key_columns)
        .sort_values(["phase_0_starcoder", "phase_1_starcoder"])
        .reset_index(drop=True)
    )
    expected_coordinates = (
        EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES
        if len(surface_metrics) == EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES
        else EXPECTED_FINAL_COORDINATES
    )
    if len(merged) != expected_coordinates:
        raise ValueError(f"Expected {expected_coordinates} unique coordinates, found {len(merged)}")
    return merged


def _merge_high_aggregate_surface_metrics(
    surface_metrics: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if len(surface_metrics) not in {
        EXPECTED_FINAL_COORDINATES,
        EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES,
    }:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_COORDINATES} or " f"{EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES} coordinates"
        )
    combined = pd.concat(
        [surface_metrics, _high_aggregate_surface_rows(observations, surface_metrics.columns)],
        ignore_index=True,
    )
    key_columns = ["_phase_0_key", "_phase_1_key"]
    combined[key_columns[0]] = combined["phase_0_starcoder"].round(12)
    combined[key_columns[1]] = combined["phase_1_starcoder"].round(12)
    for coordinate, block in combined.groupby(key_columns):
        if len(block) > 1 and block["wsd80_bpb"].max() - block["wsd80_bpb"].min() > 1e-12:
            raise ValueError(f"Conflicting reference-seed values at {coordinate}: {block['wsd80_bpb'].tolist()}")
    merged = (
        combined.drop_duplicates(key_columns, keep="first")
        .drop(columns=key_columns)
        .sort_values(["phase_0_starcoder", "phase_1_starcoder"])
        .reset_index(drop=True)
    )
    if len(merged) != EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES:
        raise ValueError(f"Expected {EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES} unique coordinates, found {len(merged)}")
    return merged


def _fiber_summary(
    observations: pd.DataFrame,
    *,
    wandb_group: str = FIBER_WANDB_GROUP,
    tied_fiber_index: int = TIED_FIBER_INDEX,
    repeated_fiber_indices: tuple[int, ...] = REPEATED_FIBER_INDICES,
) -> dict[str, object]:
    reference = observations.loc[observations["data_seed"].eq(REFERENCE_DATA_SEED)].sort_values("fiber_index")
    reference_best = reference.loc[reference["wsd80_bpb"].idxmin()]
    repeated = observations.loc[observations["fiber_index"].isin(repeated_fiber_indices)]
    pivot = repeated.pivot(index="data_seed", columns="fiber_index", values="wsd80_bpb").sort_index()
    tied = pivot[tied_fiber_index]
    repeated_summaries = []
    for fiber_index in repeated_fiber_indices:
        values = pivot[fiber_index]
        delta = values - tied
        repeated_summaries.append(
            {
                "fiber_index": fiber_index,
                "phase_0_starcoder": float(
                    reference.loc[reference["fiber_index"].eq(fiber_index), "phase_0_starcoder"].iloc[0]
                ),
                "phase_1_starcoder": float(
                    reference.loc[reference["fiber_index"].eq(fiber_index), "phase_1_starcoder"].iloc[0]
                ),
                "mean_bpb": float(values.mean()),
                "sd_bpb": float(values.std(ddof=1)),
                "mean_delta_vs_tied_bpb": float(delta.mean()),
                "fraction_better_than_tied": float((delta < 0).mean()),
            }
        )
    return {
        "wandb_group": wandb_group,
        "target": WSD80_TARGET,
        "fiber_coordinates": len(reference),
        "observation_rows": len(observations),
        "reference_seed": REFERENCE_DATA_SEED,
        "reference_seed_best": {
            "fiber_index": int(reference_best["fiber_index"]),
            "phase_0_starcoder": float(reference_best["phase_0_starcoder"]),
            "phase_1_starcoder": float(reference_best["phase_1_starcoder"]),
            "bpb": float(reference_best["wsd80_bpb"]),
        },
        "repeated_coordinates": repeated_summaries,
    }


def _high_aggregate_summary(observations: pd.DataFrame) -> dict[str, object]:
    fibers = []
    for aggregate, block in observations.groupby("aggregate_starcoder_share_80_20", sort=True):
        block = block.sort_values("fiber_index")
        tied = block.loc[np.isclose(block["ordering_contrast_p1_minus_p0"], 0.0, atol=COORDINATE_TOLERANCE)]
        if len(tied) != 1:
            raise ValueError(f"Aggregate {aggregate:.2f} must have one tied row")
        tied_bpb = float(tied["wsd80_bpb"].iloc[0])
        best = block.loc[block["wsd80_bpb"].idxmin()]
        contrasts = block["ordering_contrast_p1_minus_p0"].to_numpy(dtype=float)
        positive_pairs = [
            value
            for value in contrasts
            if value > COORDINATE_TOLERANCE and np.isclose(contrasts, -value, atol=COORDINATE_TOLERANCE).any()
        ]
        fibers.append(
            {
                "aggregate": float(aggregate),
                "coordinates": len(block),
                "antithetic_pairs": len(positive_pairs),
                "tied_bpb": tied_bpb,
                "best": {
                    "fiber_index": int(best["fiber_index"]),
                    "phase_0_starcoder": float(best["phase_0_starcoder"]),
                    "phase_1_starcoder": float(best["phase_1_starcoder"]),
                    "contrast": float(best["ordering_contrast_p1_minus_p0"]),
                    "bpb": float(best["wsd80_bpb"]),
                    "gain_vs_tied_bpb": tied_bpb - float(best["wsd80_bpb"]),
                },
            }
        )
    return {
        "wandb_group": HIGH_AGGREGATE_WANDB_GROUP,
        "target": WSD80_TARGET,
        "fiber_count": len(fibers),
        "fiber_coordinates": len(observations),
        "observation_rows": len(observations),
        "reference_seed": REFERENCE_DATA_SEED,
        "fibers": fibers,
    }


def _update_surface_summary(
    output_dir: Path,
    fiber_summary: dict[str, object],
    optimum_fiber_summary: dict[str, object],
    high_aggregate_summary: dict[str, object],
) -> None:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    wsd80_surfaces = [surface for surface in summary["surfaces"] if surface["name"] == "WSD, 80/20 phases"]
    if len(wsd80_surfaces) != 1:
        raise ValueError(f"Expected one WSD 80/20 surface summary, found {len(wsd80_surfaces)}")
    wsd80_surfaces[0]["n"] = EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES
    summary["fixed_aggregate_fiber"] = fiber_summary
    summary["fixed_aggregate_fibers"] = [
        fiber_summary,
        optimum_fiber_summary,
        high_aggregate_summary,
    ]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    metrics_path = args.output_dir / "wsd80_observed_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    observations = _collect_fiber_observations(metrics)
    observations.to_csv(args.output_dir / FIBER_OBSERVATIONS, index=False)
    fiber_summary = _fiber_summary(observations)
    (args.output_dir / FIBER_SUMMARY).write_text(json.dumps(fiber_summary, indent=2) + "\n")
    metrics = _merge_surface_metrics(metrics, observations)

    optimum_observations = _collect_optimum_fiber_observations(metrics)
    optimum_observations.to_csv(args.output_dir / OPTIMUM_FIBER_OBSERVATIONS, index=False)
    optimum_fiber_summary = _fiber_summary(
        optimum_observations,
        wandb_group=OPTIMUM_FIBER_WANDB_GROUP,
        tied_fiber_index=OPTIMUM_TIED_INDEX,
        repeated_fiber_indices=OPTIMUM_REPEATED_FIBER_INDICES,
    )
    (args.output_dir / OPTIMUM_FIBER_SUMMARY).write_text(json.dumps(optimum_fiber_summary, indent=2) + "\n")
    metrics = _merge_optimum_surface_metrics(metrics, optimum_observations)

    high_aggregate_observations = _collect_high_aggregate_observations(metrics)
    high_aggregate_observations.to_csv(
        args.output_dir / HIGH_AGGREGATE_FIBER_OBSERVATIONS,
        index=False,
    )
    high_aggregate_summary = _high_aggregate_summary(high_aggregate_observations)
    (args.output_dir / HIGH_AGGREGATE_FIBER_SUMMARY).write_text(json.dumps(high_aggregate_summary, indent=2) + "\n")
    metrics = _merge_high_aggregate_surface_metrics(metrics, high_aggregate_observations)
    metrics.to_csv(metrics_path, index=False)
    _update_surface_summary(
        args.output_dir,
        fiber_summary,
        optimum_fiber_summary,
        high_aggregate_summary,
    )

    observations = observations.assign(
        fiber_id="aggregate_0p30",
        fiber_label="Best tied-mixture fiber",
    )
    optimum_observations = optimum_observations.assign(
        fiber_id="aggregate_0p18",
        fiber_label="Observed-optimum fiber",
    )
    high_aggregate_observations = high_aggregate_observations.assign(
        fiber_id=high_aggregate_observations["aggregate_starcoder_share_80_20"].map(
            lambda aggregate: f"aggregate_{aggregate:.2f}".replace(".", "p")
        ),
        fiber_label=high_aggregate_observations["aggregate_starcoder_share_80_20"].map(
            lambda aggregate: f"High-aggregate fiber {aggregate:.2f}"
        ),
    )
    combined_observations = pd.concat(
        [observations, optimum_observations, high_aggregate_observations],
        ignore_index=True,
    )
    combined_observations.to_csv(args.output_dir / COMBINED_FIBER_OBSERVATIONS, index=False)
    frame = _surface_frame(metrics, target="wsd80_bpb", url_column="wandb_url")
    if len(frame) != EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES:
        raise ValueError(f"Expected {EXPECTED_HIGH_AGGREGATE_FINAL_COORDINATES} unique coordinates, found {len(frame)}")

    surface = Surface("WSD, 80/20 phases", frame, PHASE_0_FRACTION)
    _render_wsd80_surface(surface, args.output_dir, combined_observations)
    print(args.output_dir / "starcoder_wsd80_surface.html")


if __name__ == "__main__":
    main()
