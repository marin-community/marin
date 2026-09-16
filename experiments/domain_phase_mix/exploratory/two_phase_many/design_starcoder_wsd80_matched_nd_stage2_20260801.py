# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///

"""Freeze the 50-run Stage-2 acquisition batch for the matched N-D panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import norm, rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
STAGE1_RESULTS_DIR = PANEL_DIR / "results_20260801"
STAGE1_OBSERVATIONS_PATH = STAGE1_RESULTS_DIR / "stage1_observations.csv"
STAGE1_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage1_design_20260731.json"
OUTPUT_DIR = PANEL_DIR / "stage2_design_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage2_design_20260801.json"
STAGE1_ANALYZER_PATH = SCRIPT_DIR / "analyze_starcoder_wsd80_matched_nd_stage1_20260801.py"
STAGE1_LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_matched_nd_stage1.py"
STAGE2_LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_matched_nd_stage2.py"
STREAM_IDENTITY_PATH = SCRIPT_DIR / "starcoder_wsd80_training_identity.py"
NOISE_REFERENCE_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "starcoder_wsd80_fixed_model_tied_diagonal_20260730"
    / "results_20260731"
    / "repeat_noise.csv"
)

DESIGN_VERSION = "2026-08-01"
PHASE_0_FRACTION = 0.8
REFERENCE_SEED = 20_260_711
EXPECTED_CELLS = 10
RUNS_PER_CELL = 5
EXPECTED_RUNS = EXPECTED_CELLS * RUNS_PER_CELL
GRID_STEP = 0.01
GRID_MIN = 0.01
GRID_MAX = 0.99
MIN_EXISTING_DISTANCE = 0.035
MIN_BATCH_DISTANCE = 0.06
MIN_UNTIED_CONTRAST = 0.04
LOCAL_RANK_VETO = 0.50
NOISE_SD = 0.005
TIED_UNCERTAINTY_BPB_MARGIN = 0.04
SURFACE_UNCERTAINTY_BPB_MARGIN = 0.06
CONFIRMATION_DISCOVERY_GAIN_THRESHOLD = 0.005
CONFIRMATION_SEEDS = tuple(range(20_260_811, 20_260_819))

# Features are min-max normalized aggregate, contrast, log parameters, and log tokens.
GLOBAL_LENGTH_SCALES = (
    (0.08, 0.08, 0.45, 0.45),
    (0.15, 0.15, 0.65, 0.65),
    (0.25, 0.25, 1.00, 1.00),
)
LOCAL_LENGTH_SCALES = tuple(values[:2] for values in GLOBAL_LENGTH_SCALES)
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


def _aggregate(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return PHASE_0_FRACTION * p0 + (1.0 - PHASE_0_FRACTION) * p1


def _contrast(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return p1 - p0


def _scale(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not upper > lower:
        raise ValueError(f"Invalid scaling interval [{lower}, {upper}]")
    return (value - lower) / (upper - lower)


def _global_features(
    frame: pd.DataFrame, log_n_bounds: tuple[float, float], log_d_bounds: tuple[float, float]
) -> np.ndarray:
    p0 = frame["phase_0_starcoder"].to_numpy(dtype=float)
    p1 = frame["phase_1_starcoder"].to_numpy(dtype=float)
    log_n = np.log(frame["total_parameters"].to_numpy(dtype=float))
    log_d = np.log(frame["materialized_tokens"].to_numpy(dtype=float))
    return np.column_stack(
        [
            _aggregate(p0, p1),
            (_contrast(p0, p1) + 1.0) / 2.0,
            _scale(log_n, *log_n_bounds),
            _scale(log_d, *log_d_bounds),
        ]
    )


def _local_features(frame: pd.DataFrame) -> np.ndarray:
    p0 = frame["phase_0_starcoder"].to_numpy(dtype=float)
    p1 = frame["phase_1_starcoder"].to_numpy(dtype=float)
    return np.column_stack([_aggregate(p0, p1), (_contrast(p0, p1) + 1.0) / 2.0])


def _fit_committee(
    x_train: np.ndarray,
    y_train: np.ndarray,
    length_scales: tuple[tuple[float, ...], ...],
) -> tuple[GaussianProcessRegressor, ...]:
    response_sd = float(np.std(y_train, ddof=0))
    if response_sd <= 0.0:
        raise ValueError("Cannot fit a GP committee to a constant response")
    normalized_alpha = (NOISE_SD / response_sd) ** 2
    models = []
    for length_scale in length_scales:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=length_scale,
            length_scale_bounds="fixed",
            nu=2.5,
        )
        models.append(
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=normalized_alpha,
                normalize_y=True,
                optimizer=None,
            ).fit(x_train, y_train)
        )
    return tuple(models)


def _committee_predictions(
    models: tuple[GaussianProcessRegressor, ...], x: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    means = []
    standard_deviations = []
    for model in models:
        mean, standard_deviation = cast(tuple[np.ndarray, np.ndarray], model.predict(x, return_std=True))
        means.append(mean)
        standard_deviations.append(standard_deviation)
    return means, standard_deviations


def _expected_improvement(mean: np.ndarray, standard_deviation: np.ndarray, best: float) -> np.ndarray:
    safe_standard_deviation = np.maximum(standard_deviation, 1e-12)
    z = (best - mean) / safe_standard_deviation
    return (best - mean) * norm.cdf(z) + safe_standard_deviation * norm.pdf(z)


def _rank_fraction(values: list[np.ndarray], *, larger_is_better: bool) -> np.ndarray:
    ranked = [rankdata(-value if larger_is_better else value, method="average") / len(value) for value in values]
    return np.mean(ranked, axis=0)


def _distance_to(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None, :] - references[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def _candidate_grid(*, tied: bool) -> pd.DataFrame:
    weights = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2.0, GRID_STEP)
    if tied:
        return pd.DataFrame({"phase_0_starcoder": weights, "phase_1_starcoder": weights})
    p0, p1 = np.meshgrid(weights, weights, indexing="ij")
    frame = pd.DataFrame({"phase_0_starcoder": p0.ravel(), "phase_1_starcoder": p1.ravel()})
    return frame.loc[
        np.abs(frame["phase_1_starcoder"] - frame["phase_0_starcoder"]).ge(MIN_UNTIED_CONTRAST)
    ].reset_index(drop=True)


def _select_index(
    score: np.ndarray,
    candidates: pd.DataFrame,
    existing: np.ndarray,
    selected: list[np.ndarray],
    eligible: np.ndarray,
) -> int:
    points = candidates[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    allowed = eligible & (_distance_to(points, existing) >= MIN_EXISTING_DISTANCE)
    if selected:
        allowed &= _distance_to(points, np.vstack(selected)) >= MIN_BATCH_DISTANCE
    masked = np.where(allowed, score, np.inf)
    index = int(masked.argmin())
    if not np.isfinite(masked[index]):
        raise ValueError("Acquisition constraints leave no eligible candidate")
    return index


def _annotate_candidates(
    candidates: pd.DataFrame,
    cell_observations: pd.DataFrame,
    global_models: tuple[GaussianProcessRegressor, ...],
    local_models: tuple[GaussianProcessRegressor, ...],
    log_n_bounds: tuple[float, float],
    log_d_bounds: tuple[float, float],
) -> pd.DataFrame:
    candidates = candidates.copy()
    metadata = cell_observations.iloc[0]
    candidates["total_parameters"] = float(metadata["total_parameters"])
    candidates["materialized_tokens"] = float(metadata["materialized_tokens"])
    global_means, global_standard_deviations = _committee_predictions(
        global_models, _global_features(candidates, log_n_bounds, log_d_bounds)
    )
    local_means, local_standard_deviations = _committee_predictions(local_models, _local_features(candidates))
    best = float(cell_observations["starcoder_bpb"].min())
    global_ei = [
        _expected_improvement(mean, standard_deviation, best)
        for mean, standard_deviation in zip(global_means, global_standard_deviations, strict=True)
    ]
    local_ei = [
        _expected_improvement(mean, standard_deviation, best)
        for mean, standard_deviation in zip(local_means, local_standard_deviations, strict=True)
    ]
    candidates["global_mean_bpb"] = np.mean(global_means, axis=0)
    candidates["global_mean_sd"] = np.mean(global_standard_deviations, axis=0)
    candidates["global_ei_rank_fraction"] = _rank_fraction(global_ei, larger_is_better=True)
    candidates["global_sd_rank_fraction"] = _rank_fraction(global_standard_deviations, larger_is_better=True)
    candidates["local_mean_bpb"] = np.mean(local_means, axis=0)
    candidates["local_mean_sd"] = np.mean(local_standard_deviations, axis=0)
    candidates["local_ei_rank_fraction"] = _rank_fraction(local_ei, larger_is_better=True)
    candidates["local_sd_rank_fraction"] = _rank_fraction(local_standard_deviations, larger_is_better=True)
    points = candidates[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    existing = cell_observations[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    candidates["distance_to_existing"] = _distance_to(points, existing)
    candidates["inside_existing_convex_hull"] = Delaunay(existing).find_simplex(points) >= 0
    return candidates


def _selected_row(candidates: pd.DataFrame, index: int, acquisition_kind: str) -> dict[str, object]:
    row = candidates.iloc[index]
    return {
        "acquisition_kind": acquisition_kind,
        "phase_0_starcoder": round(float(row["phase_0_starcoder"]), 6),
        "phase_1_starcoder": round(float(row["phase_1_starcoder"]), 6),
        "aggregate_starcoder": float(
            PHASE_0_FRACTION * row["phase_0_starcoder"] + (1.0 - PHASE_0_FRACTION) * row["phase_1_starcoder"]
        ),
        "phase_contrast": float(row["phase_1_starcoder"] - row["phase_0_starcoder"]),
        "global_mean_bpb": float(row["global_mean_bpb"]),
        "global_mean_sd": float(row["global_mean_sd"]),
        "global_ei_rank_fraction": float(row["global_ei_rank_fraction"]),
        "global_sd_rank_fraction": float(row["global_sd_rank_fraction"]),
        "local_mean_bpb": float(row["local_mean_bpb"]),
        "local_mean_sd": float(row["local_mean_sd"]),
        "local_ei_rank_fraction": float(row["local_ei_rank_fraction"]),
        "local_sd_rank_fraction": float(row["local_sd_rank_fraction"]),
        "distance_to_existing": float(row["distance_to_existing"]),
        "inside_existing_convex_hull": bool(row["inside_existing_convex_hull"]),
    }


def _policy_point(candidates: pd.DataFrame, index: int) -> np.ndarray:
    """Return one candidate coordinate without pandas scalar-index ambiguity."""
    return candidates[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)[index]


def _weight_slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def launch_manifest(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the launch-relevant fields covered by the frozen manifest hash."""
    return [{field: row[field] for field in LAUNCH_FIELDS} for row in rows]


def build_rows(observations: pd.DataFrame) -> list[dict[str, object]]:
    """Apply the frozen across-cell acquisition and local-audit rule."""
    if len(observations) != 180 or observations["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError("Stage-1 observations must contain 180 rows over ten cells")
    if observations["run_name"].duplicated().any() or not np.isfinite(observations["starcoder_bpb"]).all():
        raise ValueError("Stage-1 observations are incomplete or duplicated")

    log_n = np.log(observations["total_parameters"].to_numpy(dtype=float))
    log_d = np.log(observations["materialized_tokens"].to_numpy(dtype=float))
    log_n_bounds = (float(log_n.min()), float(log_n.max()))
    log_d_bounds = (float(log_d.min()), float(log_d.max()))
    global_models = _fit_committee(
        _global_features(observations, log_n_bounds, log_d_bounds),
        observations["starcoder_bpb"].to_numpy(dtype=float),
        GLOBAL_LENGTH_SCALES,
    )

    rows: list[dict[str, object]] = []
    for cell_id, cell_observations in observations.groupby("cell_id", sort=True):
        local_models = _fit_committee(
            _local_features(cell_observations),
            cell_observations["starcoder_bpb"].to_numpy(dtype=float),
            LOCAL_LENGTH_SCALES,
        )
        existing = cell_observations[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        selected_points: list[np.ndarray] = []
        selected_rows: list[dict[str, object]] = []
        best_observed = float(cell_observations["starcoder_bpb"].min())

        tied = _annotate_candidates(
            _candidate_grid(tied=True),
            cell_observations,
            global_models,
            local_models,
            log_n_bounds,
            log_d_bounds,
        )
        tied_ei_eligible = (
            tied["local_ei_rank_fraction"].le(LOCAL_RANK_VETO) & tied["inside_existing_convex_hull"]
        ).to_numpy()
        index = _select_index(
            tied["global_ei_rank_fraction"].to_numpy(), tied, existing, selected_points, tied_ei_eligible
        )
        selected_rows.append(_selected_row(tied, index, "tied_expected_improvement"))
        selected_points.append(_policy_point(tied, index))

        tied_sd_eligible = (
            tied["local_sd_rank_fraction"].le(LOCAL_RANK_VETO)
            & tied["global_mean_bpb"].le(best_observed + TIED_UNCERTAINTY_BPB_MARGIN)
            & tied["inside_existing_convex_hull"]
        ).to_numpy()
        index = _select_index(
            tied["global_sd_rank_fraction"].to_numpy(), tied, existing, selected_points, tied_sd_eligible
        )
        selected_rows.append(_selected_row(tied, index, "tied_uncertainty"))
        selected_points.append(_policy_point(tied, index))

        surface = _annotate_candidates(
            _candidate_grid(tied=False),
            cell_observations,
            global_models,
            local_models,
            log_n_bounds,
            log_d_bounds,
        )
        surface_ei_eligible = (
            surface["local_ei_rank_fraction"].le(LOCAL_RANK_VETO) & surface["inside_existing_convex_hull"]
        ).to_numpy()
        for acquisition_index in range(1, 3):
            index = _select_index(
                surface["global_ei_rank_fraction"].to_numpy(),
                surface,
                existing,
                selected_points,
                surface_ei_eligible,
            )
            selected_rows.append(_selected_row(surface, index, f"surface_expected_improvement_{acquisition_index}"))
            selected_points.append(_policy_point(surface, index))

        surface_sd_eligible = (
            surface["local_sd_rank_fraction"].le(LOCAL_RANK_VETO)
            & surface["global_mean_bpb"].le(best_observed + SURFACE_UNCERTAINTY_BPB_MARGIN)
            & surface["inside_existing_convex_hull"]
        ).to_numpy()
        index = _select_index(
            surface["global_sd_rank_fraction"].to_numpy(),
            surface,
            existing,
            selected_points,
            surface_sd_eligible,
        )
        selected_rows.append(_selected_row(surface, index, "surface_uncertainty"))

        metadata = cell_observations.iloc[0]
        for row in selected_rows:
            row.update(
                {
                    "cell_id": cell_id,
                    "rung": int(metadata["rung"]),
                    "track_memberships": metadata["track_memberships"],
                    "hidden_size": int(metadata["hidden_size"]),
                    "total_steps": int(metadata["total_steps"]),
                    "boundary_step": int(int(metadata["total_steps"]) * PHASE_0_FRACTION),
                    "materialized_tokens": int(metadata["materialized_tokens"]),
                    "total_parameters": int(metadata["total_parameters"]),
                    "non_embedding_parameters": int(metadata["non_embedding_parameters"]),
                    "data_seed": REFERENCE_SEED,
                    "simulated_epoch_subset_seed": REFERENCE_SEED,
                }
            )
            p0 = float(row["phase_0_starcoder"])
            p1 = float(row["phase_1_starcoder"])
            kind = str(row["acquisition_kind"]).replace("_expected_improvement", "_ei")
            row["run_name"] = f"s2_{cell_id}_{kind}_p0{_weight_slug(p0)}_p1{_weight_slug(p1)}"
            rows.append(row)

    if len(rows) != EXPECTED_RUNS or len({str(row["run_name"]) for row in rows}) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} unique Stage-2 rows")
    return rows


def write_outputs() -> None:
    """Persist the frozen Stage-2 launch design and audit table."""
    observations = pd.read_csv(STAGE1_OBSERVATIONS_PATH)
    rows = build_rows(observations)
    frame = pd.DataFrame(rows)
    source_design = json.loads(STAGE1_DESIGN_PATH.read_text(encoding="utf-8"))
    noise_reference = pd.read_csv(NOISE_REFERENCE_PATH)
    unique_stage1_points = observations[["phase_0_starcoder", "phase_1_starcoder"]].drop_duplicates()
    hull = ConvexHull(unique_stage1_points.to_numpy(dtype=float))
    hull_vertices = unique_stage1_points.iloc[hull.vertices].to_dict(orient="records")
    best_by_cell = observations.groupby("cell_id")["starcoder_bpb"].min()
    surface_local_margin_excess = {
        str(row.cell_id): float(row.local_mean_bpb - best_by_cell.loc[row.cell_id] - SURFACE_UNCERTAINTY_BPB_MARGIN)
        for row in frame.loc[frame["acquisition_kind"].eq("surface_uncertainty")].itertuples()
        if row.local_mean_bpb > best_by_cell.loc[row.cell_id] + SURFACE_UNCERTAINTY_BPB_MARGIN
    }
    tied_local_margin_excess = {
        str(row.cell_id): float(row.local_mean_bpb - best_by_cell.loc[row.cell_id] - TIED_UNCERTAINTY_BPB_MARGIN)
        for row in frame.loc[frame["acquisition_kind"].eq("tied_uncertainty")].itertuples()
        if row.local_mean_bpb > best_by_cell.loc[row.cell_id] + TIED_UNCERTAINTY_BPB_MARGIN
    }
    ei_pair_distances = []
    for _, cell_rows in frame.groupby("cell_id", sort=True):
        pair = cell_rows.loc[
            cell_rows["acquisition_kind"].isin(("surface_expected_improvement_1", "surface_expected_improvement_2")),
            ["phase_0_starcoder", "phase_1_starcoder"],
        ].to_numpy(dtype=float)
        if pair.shape != (2, 2):
            raise ValueError("Each cell must contain exactly two surface expected-improvement probes")
        ei_pair_distances.append(float(np.linalg.norm(pair[0] - pair[1])))
    source_paths = (
        Path(__file__).resolve(),
        STAGE1_ANALYZER_PATH,
        STAGE1_DESIGN_PATH,
        STAGE1_OBSERVATIONS_PATH,
        STAGE1_LAUNCHER_PATH,
        STAGE2_LAUNCHER_PATH,
        STREAM_IDENTITY_PATH,
        NOISE_REFERENCE_PATH,
    )
    payload = {
        "design_version": DESIGN_VERSION,
        "description": "Frozen 50-run Stage-2 acquisition batch for the matched-compute WSD80 N-D panel.",
        "objective_metric": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
        "phase_0_fraction": PHASE_0_FRACTION,
        "expected_run_count": EXPECTED_RUNS,
        "cell_count": EXPECTED_CELLS,
        "runs_per_cell": RUNS_PER_CELL,
        "data_use": {
            "stage1_observation_count": len(observations),
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
        },
        "training_environment": {
            "tpu_type": "v5p-8",
            "tpu_region": "us-central1",
            "tpu_zone": "us-central1-a",
            "marin_prefix": "gs://marin-us-central1",
        },
        "acquisition": {
            "global_features": ["aggregate", "contrast", "log_total_parameters", "log_materialized_tokens"],
            "global_length_scales": GLOBAL_LENGTH_SCALES,
            "local_features": ["aggregate", "contrast"],
            "local_length_scales": LOCAL_LENGTH_SCALES,
            "noise_sd_bpb": NOISE_SD,
            "gp_observation_variance": (
                "Within each GP fit, (noise_sd_bpb / response_sd_bpb)^2 because normalize_y=True."
            ),
            "grid_step": GRID_STEP,
            "grid_bounds": [GRID_MIN, GRID_MAX],
            "minimum_existing_distance": MIN_EXISTING_DISTANCE,
            "minimum_batch_distance": MIN_BATCH_DISTANCE,
            "minimum_untied_contrast": MIN_UNTIED_CONTRAST,
            "local_rank_veto": LOCAL_RANK_VETO,
            "support_gate": (
                "Every acquisition must lie inside the common empirical Stage-1 convex hull. All cells use the "
                "same 18 coordinates, so hull membership bounds extrapolation but does not establish local support."
            ),
            "support_summary": {
                "common_hull_vertices": hull_vertices,
                "common_hull_area_fraction_of_unit_square": float(hull.volume),
                "maximum_nearest_observation_distance": float(frame["distance_to_existing"].max()),
                "off_diagonal_stage1_coordinates": int(
                    unique_stage1_points["phase_0_starcoder"].ne(unique_stage1_points["phase_1_starcoder"]).sum()
                ),
            },
            "noise_sd_provenance": {
                "status": "assumed from prior same-schedule five-seed tied repeats, not measured within this panel",
                "source": str(NOISE_REFERENCE_PATH.relative_to(REPO_ROOT)),
                "reference_repeat_sd_bpb_range": [
                    float(noise_reference["repeat_sd_bpb"].min()),
                    float(noise_reference["repeat_sd_bpb"].max()),
                ],
            },
            "global_mean_margin_only": {
                "tied_local_fit_excess_bpb": tied_local_margin_excess,
                "surface_local_fit_excess_bpb": surface_local_margin_excess,
            },
            "surface_ei_pair_distance_range": [min(ei_pair_distances), max(ei_pair_distances)],
            "allocation_per_cell": [
                "tied_expected_improvement",
                "tied_uncertainty",
                "surface_expected_improvement_1",
                "surface_expected_improvement_2",
                "surface_uncertainty",
            ],
        },
        "design_provenance": (
            "Before Stage-1 outcomes existed, the Stage-1 review packet predeclared only the per-cell allocation: "
            "two tied-diagonal and three full-surface Bayesian acquisitions, or 50 runs over ten cells. The GP "
            "kernel and length-scale committee, the two-EI/one-uncertainty split, local rank veto, BPB margins, "
            "distance thresholds, assumed noise SD, and convex-hull gate were chosen after Stage-1 outcomes were "
            "available. An initial pre-launch draft selected out-of-hull corners; the hull gate was then added before "
            "any Stage-2 run or outcome existed. This is an outcome-informed adaptive discovery batch, not a "
            "confirmatory or fully preregistered design."
        ),
        "confirmation_boundary": {
            "promotion_rule": (
                "Within each cell, compare the lowest observed untied candidate with the lowest observed tied "
                "candidate across Stages 1 and 2. Promote at most one untied candidate per cell, and only when its "
                "single-seed discovery gain is at least 0.005 BPB."
            ),
            "fresh_seeds": CONFIRMATION_SEEDS,
            "paired_runs_per_promoted_cell": 2 * len(CONFIRMATION_SEEDS),
            "success_rule": (
                "Using the same eight fresh seeds for candidate and tied comparator, require at least seven of eight "
                "paired wins, a positive lower endpoint for the 95% paired-t confidence interval on tied-minus-untied "
                "BPB, and Holm-adjusted one-sided paired-t p<0.05 across promoted cells."
            ),
            "claim_limit": (
                "Passing confirms the selected discrete policy against its selected tied comparator; it does not prove "
                "the continuous global optimum. No cell below the discovery threshold receives a positive phase-gap "
                "claim."
            ),
        },
        "design": {
            "launch_manifest_sha256": stream_identity.canonical_sha256(launch_manifest(rows)),
        },
        "interpretation_boundary": (
            "This is the only adaptive discovery stage. Across-cell Gaussian processes borrow strength over N and D; "
            "independent per-cell fits veto candidates in the worse half of the corresponding local acquisition "
            "ranking. Stage 2 adds no new N,D cell and refines only the response surface within the existing ten "
            "cells. Stage-2 outcomes must not alter this frozen 230-run Stage-1-plus-Stage-2 discovery panel. Any "
            "promoted policy must pass the frozen fresh-seed confirmation rule above."
        ),
        "source_cells": source_design["cells"],
        "runs": rows,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "design_manifest.json").write_text(serialized, encoding="utf-8")
    FROZEN_DESIGN_PATH.write_text(serialized, encoding="utf-8")
    frame.to_csv(OUTPUT_DIR / "run_manifest.csv", index=False)
    report = [
        "# StarCoder WSD80 matched-N,D Stage-2 design",
        "",
        f"- Frozen run count: {len(frame)} across {frame['cell_id'].nunique()} cells.",
        "- Each cell receives two tied-diagonal and three off-diagonal two-phase acquisitions; the off-diagonal grid "
        f"requires |phase contrast| >= {MIN_UNTIED_CONTRAST:.2f}.",
        "- Expected-improvement probes exploit the across-cell response model. The second EI probe is a locally "
        f"paired measurement {min(ei_pair_distances):.3f}-{max(ei_pair_distances):.3f} from the first, not an "
        "independent surface mode.",
        "- Uncertainty probes satisfy a predicted-BPB margin under the across-cell model only; the independent "
        "per-cell fit exceeds that margin in the cases listed below.",
        "- Every proposal lies inside one common empirical convex hull because all cells share the same 18 Stage-1 "
        f"coordinates. Hull membership limits extrapolation but does not imply local support; the maximum nearest-"
        f"observation distance is {frame['distance_to_existing'].max():.3f}.",
        "- The per-cell rank veto and all acquisition hyperparameters are outcome-informed Stage-2 design choices, "
        "not preregistered tests.",
        "",
        "## Design provenance",
        "",
        str(payload["design_provenance"]),
        "",
        "## Support and acquisition limits",
        "",
        f"- Common hull vertices: `{json.dumps(hull_vertices, sort_keys=True)}`.",
        f"- Hull area: {float(hull.volume):.3f} of the unit square; only "
        f"{payload['acquisition']['support_summary']['off_diagonal_stage1_coordinates']} Stage-1 coordinates are "
        "genuinely off diagonal, and boundary points are admitted.",
        "- The predicted-BPB margin is applied to the across-cell mean only. Independent per-cell excesses above the "
        f"tied margin: `{json.dumps(tied_local_margin_excess, sort_keys=True)}`; above the surface margin: "
        f"`{json.dumps(surface_local_margin_excess, sort_keys=True)}`.",
        f"- The {NOISE_SD:.3f}-BPB noise SD is an assumption based on prior same-schedule five-seed tied repeats "
        f"spanning {noise_reference['repeat_sd_bpb'].min():.6f}-{noise_reference['repeat_sd_bpb'].max():.6f} BPB; "
        "this panel has no coordinate repeats, so its own run-to-run SD is unidentified.",
        "- All Stage-2 surface probes have positive phase contrast. The new batch cannot independently establish a "
        "general code-late versus code-early ordering law.",
        "- Stage 2 adds no N,D cell. It refines within-cell response surfaces at the existing ten cells; after Stage 2, "
        "the discovery panel contains 230 runs but still only ten N,D design points.",
        "",
        "## Acquisitions",
        "",
        frame[
            [
                "cell_id",
                "acquisition_kind",
                "phase_0_starcoder",
                "phase_1_starcoder",
                "aggregate_starcoder",
                "phase_contrast",
                "global_ei_rank_fraction",
                "local_ei_rank_fraction",
                "global_sd_rank_fraction",
                "local_sd_rank_fraction",
                "distance_to_existing",
                "inside_existing_convex_hull",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation boundary",
        "",
        str(payload["interpretation_boundary"]),
        "",
        "## Frozen fresh-seed confirmation",
        "",
        f"- Promotion: {payload['confirmation_boundary']['promotion_rule']}",
        f"- Seeds: `{list(CONFIRMATION_SEEDS)}` ({len(CONFIRMATION_SEEDS)} paired seeds per promoted cell).",
        f"- Success: {payload['confirmation_boundary']['success_rule']}",
        f"- Claim limit: {payload['confirmation_boundary']['claim_limit']}",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
