# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Render an interactive completed-surface audit for the matched-N,D StarCoder grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from starcoder_wsd80_epoch_accounting import (
    SIMULATED_EPOCH_TARGET_BUDGET,
    simulated_materialized_epochs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
STAGE3_RESULTS = PANEL_DIR / "stage3_dense_surface_results_20260802"
DEFAULT_OBSERVATIONS = STAGE3_RESULTS / "combined_discovery_observations.csv"
DEFAULT_CANDIDATES = STAGE3_RESULTS / "fitted_surface_candidates.csv"
DEFAULT_SOURCE_DESIGN = PANEL_DIR / "stage2_results_20260801" / "source_design.json"
DEFAULT_HISTORICAL_DENSE = (
    SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_surface_refined_20260714" / "wsd80_observed_metrics.csv"
)
DEFAULT_OUTPUT_DIR = PANEL_DIR / "surface_explorer_20260802"

TRACK_ORDER = ("increase_d", "increase_n", "increase_nd")
TRACK_LABELS = {
    "increase_d": "Fixed N · increase D",
    "increase_n": "Fixed D · increase N",
    "increase_nd": "Increase N and D",
}
PHASE_0_FRACTION = 0.8
EXPECTED_CELL_COUNT = 10
EXPECTED_TOTAL_RUNS = 714
EXPECTED_UNTIED_PER_CELL = 56
LOCAL_RADIUS = 0.15
HISTORICAL_DENSE_TOTAL_PARAMETERS = 157_527_552
HISTORICAL_DENSE_NON_EMBEDDING_PARAMETERS = 59_009_280
HISTORICAL_DENSE_TOKENS = 999_817_216
HISTORICAL_PAIRED_GAIN = 0.005339


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--historical-dense", type=Path, default=DEFAULT_HISTORICAL_DENSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _track_memberships(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value.replace("'", '"'))
    if not isinstance(value, list):
        raise ValueError(f"Invalid track memberships: {value!r}")
    tracks = tuple(str(item) for item in value)
    unknown = set(tracks) - set(TRACK_ORDER)
    if unknown:
        raise ValueError(f"Unknown track memberships: {sorted(unknown)}")
    return tracks


def _load_cells(path: Path) -> pd.DataFrame:
    design = json.loads(path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(design.get("source_cells", design.get("cells")))
    required = {
        "cell_id",
        "compute_flops",
        "hidden_size",
        "materialized_tokens",
        "non_embedding_parameters",
        "num_layers",
        "rung",
        "total_parameters",
        "track_memberships",
    }
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Source design is missing fields: {sorted(missing)}")
    if len(cells) != EXPECTED_CELL_COUNT or cells["cell_id"].nunique() != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected {EXPECTED_CELL_COUNT} unique N,D cells")
    cells["track_memberships"] = cells["track_memberships"].map(_track_memberships)
    return cells


def _load_observations(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "cell_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "policy_class",
        "run_name",
        "selection_label",
        "source_stage",
        "starcoder_bpb",
        "wandb_url",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Observation table is missing fields: {sorted(missing)}")
    if set(frame["cell_id"]) != set(cells["cell_id"]):
        raise ValueError("Observation and source-design cell IDs disagree")
    numeric = frame[["phase_0_starcoder", "phase_1_starcoder", "starcoder_bpb"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Observations contain non-finite coordinates or BPB")
    if not frame["phase_0_starcoder"].between(0.0, 1.0).all():
        raise ValueError("Phase-0 weights must be in [0,1]")
    if not frame["phase_1_starcoder"].between(0.0, 1.0).all():
        raise ValueError("Phase-1 weights must be in [0,1]")
    if len(frame) != EXPECTED_TOTAL_RUNS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_RUNS} completed discovery outcomes, found {len(frame)}")
    for cell_id, group in frame.groupby("cell_id"):
        if group.duplicated(["phase_0_starcoder", "phase_1_starcoder"]).any():
            raise ValueError(f"{cell_id}: coordinates are not unique")
        counts = group["policy_class"].value_counts().to_dict()
        if counts.get("untied") != EXPECTED_UNTIED_PER_CELL or counts.get("tied") not in {15, 16}:
            raise ValueError(f"{cell_id}: unexpected policy counts {counts}")
    return frame


def _load_candidates(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "cell_id",
        "fitted_untied_p0",
        "fitted_untied_p1",
        "fitted_untied_bpb",
        "fitted_tied_weight",
        "fitted_tied_bpb",
        "fitted_gain_tied_minus_untied_bpb",
        "bootstrap_gain_p05",
        "bootstrap_gain_p95",
        "bootstrap_positive_gain_probability",
        "bootstrap_candidate_l2_p90",
        "confirmation_eligible",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fitted-candidate table is missing fields: {sorted(missing)}")
    if set(frame["cell_id"]) != set(cells["cell_id"]):
        raise ValueError("Fitted candidates and source-design cell IDs disagree")
    if len(frame) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected one fitted candidate per cell, found {len(frame)}")
    return frame


def _load_historical_dense(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"phase_0_starcoder", "phase_1_starcoder", "wsd80_bpb"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical dense panel is missing fields: {sorted(missing)}")
    frame = frame.loc[np.isfinite(frame["wsd80_bpb"])].copy()
    if frame.empty:
        raise ValueError("Historical dense panel has no finite WSD80 observations")
    return frame


def _triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    ab = b - a
    ac = c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def _unique_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for triangle in triangles:
        for left, right in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges.add(tuple(sorted((int(left), int(right)))))
    return sorted(edges)


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _sampling_read(
    *,
    stage1_gain: float,
    raw_gain: float,
    fitted_gain: float,
    bootstrap_gain_p05: float,
    bootstrap_gain_p95: float,
    bootstrap_positive_gain_probability: float,
    bootstrap_candidate_l2_p90: float,
    confirmation_eligible: bool,
    best_untied_on_hull: bool,
    prior_local_untied_neighbors: int,
    local_untied_neighbors: int,
    untied_median_nearest_neighbor: float,
    largest_triangle_hull_fraction: float,
    untied_margin: float,
) -> list[str]:
    findings = [
        f"The completed mesh contains {EXPECTED_UNTIED_PER_CELL} untied outcomes. Median untied nearest-neighbor "
        f"spacing is {untied_median_nearest_neighbor:.3f}, and {local_untied_neighbors} points lie within L2 radius "
        f"{LOCAL_RADIUS:.2f} of the raw untied winner (up from {prior_local_untied_neighbors} before Stage 3).",
        f"The largest interpolation triangle is now {100 * largest_triangle_hull_fraction:.1f}% of the sampled hull.",
        f"The frozen smooth fit estimates a {fitted_gain:.6f}-BPB gain with bootstrap p05/p95 "
        f"[{bootstrap_gain_p05:.6f}, {bootstrap_gain_p95:.6f}] and P(gain>0)={bootstrap_positive_gain_probability:.3f}.",
        f"Candidate-location bootstrap displacement is L2 p90={bootstrap_candidate_l2_p90:.3f}; "
        + (
            "this cell clears the frozen confirmation gate."
            if confirmation_eligible
            else "this cell does not clear the frozen confirmation gate."
        ),
    ]
    gain_change = raw_gain - stage1_gain
    if abs(gain_change) >= 5e-4:
        findings.append(
            f"Densification changed the raw min-vs-min gain from {stage1_gain:.6f} to {raw_gain:.6f} BPB "
            f"({gain_change:+.6f}); raw minima remain selection-biased."
        )
    if best_untied_on_hull:
        findings.append(
            "The best untied observation is on the sampled convex-hull edge, so the optimum may lie beyond current support."
        )
    if untied_margin < 0.002:
        findings.append(
            f"The best and second-best untied observations differ by only {untied_margin:.6f} BPB; their ordering is fragile at this resolution."
        )
    return findings


def _cell_payload(
    cell: pd.Series,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    frame = observations.loc[observations["cell_id"].eq(cell["cell_id"])].copy().reset_index(drop=True)
    candidate = candidates.loc[candidates["cell_id"].eq(cell["cell_id"])].iloc[0]
    points = frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    triangulation = Delaunay(points)
    triangles = triangulation.simplices.astype(int)
    triangle_areas = _triangle_areas(points, triangles)
    hull = ConvexHull(points)
    hull_vertices = set(int(index) for index in hull.vertices)

    prior = frame.loc[~frame["source_stage"].eq("stage3")]
    prior_points = prior[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    prior_triangles = Delaunay(prior_points).simplices.astype(int)
    prior_hull = ConvexHull(prior_points)
    prior_triangle_areas = _triangle_areas(prior_points, prior_triangles)

    untied_indices = np.flatnonzero(frame["policy_class"].eq("untied").to_numpy())
    untied_points = points[untied_indices]
    untied_neighbor_distances = _nearest_neighbor_distances(untied_points)
    prior_untied_points = prior.loc[
        prior["policy_class"].eq("untied"), ["phase_0_starcoder", "phase_1_starcoder"]
    ].to_numpy(dtype=float)

    tied = frame.loc[frame["policy_class"].eq("tied")]
    untied = frame.loc[frame["policy_class"].eq("untied")]
    best_tied_index = int(tied["starcoder_bpb"].idxmin())
    best_untied_index = int(untied["starcoder_bpb"].idxmin())
    best_tied = frame.loc[best_tied_index]
    best_untied = frame.loc[best_untied_index]
    raw_gain = max(0.0, float(best_tied["starcoder_bpb"] - best_untied["starcoder_bpb"]))
    winner_index = best_untied_index if raw_gain > 0.0 else best_tied_index

    stage1 = frame.loc[frame["source_stage"].eq("stage1")]
    stage1_tied = stage1.loc[stage1["policy_class"].eq("tied"), "starcoder_bpb"].min()
    stage1_untied = stage1.loc[stage1["policy_class"].eq("untied"), "starcoder_bpb"].min()
    stage1_gain = max(0.0, float(stage1_tied - stage1_untied))

    ordered_untied = untied.sort_values("starcoder_bpb")
    untied_margin = float(ordered_untied.iloc[1]["starcoder_bpb"] - ordered_untied.iloc[0]["starcoder_bpb"])
    best_untied_point = points[best_untied_index]
    distances_from_best = np.linalg.norm(untied_points - best_untied_point, axis=1)
    local_untied_neighbors = int(np.sum((distances_from_best > 1e-12) & (distances_from_best <= LOCAL_RADIUS)))
    prior_distances_from_best = np.linalg.norm(prior_untied_points - best_untied_point, axis=1)
    prior_local_untied_neighbors = int(
        np.sum((prior_distances_from_best > 1e-12) & (prior_distances_from_best <= LOCAL_RADIUS))
    )

    aggregate = PHASE_0_FRACTION * frame["phase_0_starcoder"] + (1.0 - PHASE_0_FRACTION) * frame["phase_1_starcoder"]
    frame["aggregate_starcoder"] = aggregate
    frame["phase_contrast"] = frame["phase_1_starcoder"] - frame["phase_0_starcoder"]
    epoch_rows = [
        simulated_materialized_epochs(row.phase_0_starcoder, row.phase_1_starcoder)
        for row in frame.itertuples(index=False)
    ]
    frame["starcoder_phase_0_simulated_epochs"] = [row.starcoder.phase_0 for row in epoch_rows]
    frame["starcoder_phase_1_simulated_epochs"] = [row.starcoder.phase_1 for row in epoch_rows]
    frame["starcoder_total_simulated_epochs"] = [row.starcoder.total for row in epoch_rows]
    frame["nemotron_phase_0_simulated_epochs"] = [row.nemotron.phase_0 for row in epoch_rows]
    frame["nemotron_phase_1_simulated_epochs"] = [row.nemotron.phase_1 for row in epoch_rows]
    frame["nemotron_total_simulated_epochs"] = [row.nemotron.total for row in epoch_rows]
    hover = [
        "<br>".join(
            [
                f"{row.source_stage.title()} · {row.policy_class}",
                f"{row.selection_label}",
                f"Phase 0 StarCoder: {row.phase_0_starcoder:.4f}",
                f"Phase 1 StarCoder: {row.phase_1_starcoder:.4f}",
                f"80/20 aggregate: {row.aggregate_starcoder:.4f}",
                f"Phase contrast p1-p0: {row.phase_contrast:+.4f}",
                f"<br>Simulated materialized epochs (target {SIMULATED_EPOCH_TARGET_BUDGET / 1e12:.3f}T)",
                (
                    f"StarCoder: {row.starcoder_phase_0_simulated_epochs:.3f} early + "
                    f"{row.starcoder_phase_1_simulated_epochs:.3f} late = "
                    f"{row.starcoder_total_simulated_epochs:.3f}"
                ),
                (
                    f"Nemotron: {row.nemotron_phase_0_simulated_epochs:.3f} early + "
                    f"{row.nemotron_phase_1_simulated_epochs:.3f} late = "
                    f"{row.nemotron_total_simulated_epochs:.3f}"
                ),
                "<br>Outcome",
                f"Programming BPB: {row.starcoder_bpb:.6f}",
                f"Run: {row.run_name}",
            ]
        )
        for row in frame.itertuples(index=False)
    ]
    customdata = [
        [row.run_name, row.source_stage, row.selection_label, row.policy_class, row.wandb_url]
        for row in frame.itertuples(index=False)
    ]
    records = {
        "cellId": str(cell["cell_id"]),
        "rung": int(cell["rung"]),
        "hiddenSize": int(cell["hidden_size"]),
        "layers": int(cell["num_layers"]),
        "parameters": int(cell["total_parameters"]),
        "nonEmbeddingParameters": int(cell["non_embedding_parameters"]),
        "tokens": int(cell["materialized_tokens"]),
        "computeFlops": float(cell["compute_flops"]),
        "totalTpp": float(cell["materialized_tokens"] / cell["total_parameters"]),
        "nonEmbeddingTpp": float(cell["materialized_tokens"] / cell["non_embedding_parameters"]),
        "x": frame["phase_0_starcoder"].astype(float).tolist(),
        "y": frame["phase_1_starcoder"].astype(float).tolist(),
        "z": frame["starcoder_bpb"].astype(float).tolist(),
        "colorMax": float(min(frame["starcoder_bpb"].max(), frame["starcoder_bpb"].min() + 0.05)),
        "hover": hover,
        "customdata": customdata,
        "triangles": triangles.tolist(),
        "edges": _unique_edges(triangles),
        "hull": [*map(int, hull.vertices), int(hull.vertices[0])],
        "bestTiedIndex": best_tied_index,
        "bestUntiedIndex": best_untied_index,
        "winnerIndex": winner_index,
        "bestTiedBpb": float(best_tied["starcoder_bpb"]),
        "bestUntiedBpb": float(best_untied["starcoder_bpb"]),
        "rawGain": raw_gain,
        "fittedGain": float(candidate["fitted_gain_tied_minus_untied_bpb"]),
        "bootstrapGainP05": float(candidate["bootstrap_gain_p05"]),
        "bootstrapGainP95": float(candidate["bootstrap_gain_p95"]),
        "bootstrapPositiveGainProbability": float(candidate["bootstrap_positive_gain_probability"]),
        "bootstrapCandidateL2P90": float(candidate["bootstrap_candidate_l2_p90"]),
        "confirmationEligible": bool(candidate["confirmation_eligible"]),
        "stage1Gain": stage1_gain,
        "hullArea": float(hull.volume),
        "priorHullArea": float(prior_hull.volume),
        "largestTriangleArea": float(triangle_areas.max()),
        "largestTriangleHullFraction": float(triangle_areas.max() / hull.volume),
        "priorLargestTriangleHullFraction": float(prior_triangle_areas.max() / prior_hull.volume),
        "untiedMedianNearestNeighbor": float(np.median(untied_neighbor_distances)),
        "bestUntiedOnHull": best_untied_index in hull_vertices,
        "localUntiedNeighbors": local_untied_neighbors,
        "priorLocalUntiedNeighbors": prior_local_untied_neighbors,
        "untiedMargin": untied_margin,
        "samplingRead": _sampling_read(
            stage1_gain=stage1_gain,
            raw_gain=raw_gain,
            fitted_gain=float(candidate["fitted_gain_tied_minus_untied_bpb"]),
            bootstrap_gain_p05=float(candidate["bootstrap_gain_p05"]),
            bootstrap_gain_p95=float(candidate["bootstrap_gain_p95"]),
            bootstrap_positive_gain_probability=float(candidate["bootstrap_positive_gain_probability"]),
            bootstrap_candidate_l2_p90=float(candidate["bootstrap_candidate_l2_p90"]),
            confirmation_eligible=bool(candidate["confirmation_eligible"]),
            best_untied_on_hull=best_untied_index in hull_vertices,
            prior_local_untied_neighbors=prior_local_untied_neighbors,
            local_untied_neighbors=local_untied_neighbors,
            untied_median_nearest_neighbor=float(np.median(untied_neighbor_distances)),
            largest_triangle_hull_fraction=float(triangle_areas.max() / hull.volume),
            untied_margin=untied_margin,
        ),
    }
    diagnostic = {
        "cell_id": cell["cell_id"],
        "rung": int(cell["rung"]),
        "total_parameters": int(cell["total_parameters"]),
        "materialized_tokens": int(cell["materialized_tokens"]),
        "compute_flops": float(cell["compute_flops"]),
        "total_parameter_tpp": records["totalTpp"],
        "coordinate_count": len(frame),
        "tied_coordinate_count": int(frame["policy_class"].eq("tied").sum()),
        "untied_coordinate_count": int(frame["policy_class"].eq("untied").sum()),
        "stage2_coordinate_count": int(frame["source_stage"].eq("stage2").sum()),
        "stage3_coordinate_count": int(frame["source_stage"].eq("stage3").sum()),
        "convex_hull_area_fraction_of_unit_square": records["hullArea"],
        "prior_convex_hull_area_fraction_of_unit_square": records["priorHullArea"],
        "largest_triangle_area_fraction_of_hull": records["largestTriangleHullFraction"],
        "prior_largest_triangle_area_fraction_of_hull": records["priorLargestTriangleHullFraction"],
        "untied_median_nearest_neighbor_l2": records["untiedMedianNearestNeighbor"],
        "best_untied_on_sampled_hull": records["bestUntiedOnHull"],
        "untied_neighbors_within_l2_0p15": records["localUntiedNeighbors"],
        "prior_untied_neighbors_within_l2_0p15": records["priorLocalUntiedNeighbors"],
        "best_to_second_best_untied_margin_bpb": records["untiedMargin"],
        "stage1_nested_two_phase_gain_bpb": records["stage1Gain"],
        "raw_nested_two_phase_gain_bpb": records["rawGain"],
        "fitted_nested_two_phase_gain_bpb": records["fittedGain"],
        "bootstrap_candidate_l2_p90": records["bootstrapCandidateL2P90"],
        "confirmation_eligible": records["confirmationEligible"],
    }
    return records, diagnostic


def _comparison_payload(
    cells: pd.DataFrame,
    observations: pd.DataFrame,
    historical_dense: pd.DataFrame,
) -> dict[str, object]:
    historical = historical_dense.groupby(["phase_0_starcoder", "phase_1_starcoder"], as_index=False)["wsd80_bpb"].mean()
    historical_tied = historical.loc[np.isclose(historical["phase_0_starcoder"], historical["phase_1_starcoder"])]
    historical_untied = historical.loc[~np.isclose(historical["phase_0_starcoder"], historical["phase_1_starcoder"])]
    historical_best_tied = historical_tied.loc[historical_tied["wsd80_bpb"].idxmin()]
    historical_best_untied = historical_untied.loc[historical_untied["wsd80_bpb"].idxmin()]

    r0_cell = cells.loc[cells["cell_id"].eq("r0_shared_h0640_s03820")].iloc[0]
    r0 = observations.loc[observations["cell_id"].eq(r0_cell["cell_id"])]
    r0_tied = r0.loc[r0["policy_class"].eq("tied")]
    r0_untied = r0.loc[r0["policy_class"].eq("untied")]
    r0_best_tied = r0_tied.loc[r0_tied["starcoder_bpb"].idxmin()]
    r0_best_untied = r0_untied.loc[r0_untied["starcoder_bpb"].idxmin()]
    exact_historical_policy = r0.loc[
        np.isclose(r0["phase_0_starcoder"], float(historical_best_untied["phase_0_starcoder"]))
        & np.isclose(r0["phase_1_starcoder"], float(historical_best_untied["phase_1_starcoder"]))
    ]
    if len(exact_historical_policy) != 1:
        raise ValueError("Matched-grid r0 does not contain the historical dense winner exactly once")
    exact_historical_policy = exact_historical_policy.iloc[0]
    return {
        "historical": {
            "parameters": HISTORICAL_DENSE_TOTAL_PARAMETERS,
            "nonEmbeddingParameters": HISTORICAL_DENSE_NON_EMBEDDING_PARAMETERS,
            "tokens": HISTORICAL_DENSE_TOKENS,
            "totalTpp": HISTORICAL_DENSE_TOKENS / HISTORICAL_DENSE_TOTAL_PARAMETERS,
            "nonEmbeddingTpp": HISTORICAL_DENSE_TOKENS / HISTORICAL_DENSE_NON_EMBEDDING_PARAMETERS,
            "bestTied": {
                "p0": float(historical_best_tied["phase_0_starcoder"]),
                "p1": float(historical_best_tied["phase_1_starcoder"]),
                "bpb": float(historical_best_tied["wsd80_bpb"]),
            },
            "bestUntied": {
                "p0": float(historical_best_untied["phase_0_starcoder"]),
                "p1": float(historical_best_untied["phase_1_starcoder"]),
                "bpb": float(historical_best_untied["wsd80_bpb"]),
            },
            "referenceSeedObservedGain": float(historical_best_tied["wsd80_bpb"] - historical_best_untied["wsd80_bpb"]),
            "freshSeedPairedGain": HISTORICAL_PAIRED_GAIN,
        },
        "matchedR0": {
            "parameters": int(r0_cell["total_parameters"]),
            "nonEmbeddingParameters": int(r0_cell["non_embedding_parameters"]),
            "tokens": int(r0_cell["materialized_tokens"]),
            "totalTpp": float(r0_cell["materialized_tokens"] / r0_cell["total_parameters"]),
            "nonEmbeddingTpp": float(r0_cell["materialized_tokens"] / r0_cell["non_embedding_parameters"]),
            "layers": int(r0_cell["num_layers"]),
            "hiddenSize": int(r0_cell["hidden_size"]),
            "bestTied": {
                "p0": float(r0_best_tied["phase_0_starcoder"]),
                "p1": float(r0_best_tied["phase_1_starcoder"]),
                "bpb": float(r0_best_tied["starcoder_bpb"]),
            },
            "bestUntied": {
                "p0": float(r0_best_untied["phase_0_starcoder"]),
                "p1": float(r0_best_untied["phase_1_starcoder"]),
                "bpb": float(r0_best_untied["starcoder_bpb"]),
            },
            "observedGain": max(0.0, float(r0_best_tied["starcoder_bpb"] - r0_best_untied["starcoder_bpb"])),
            "historicalWinnerBpb": float(exact_historical_policy["starcoder_bpb"]),
            "historicalWinnerMinusBestTied": float(
                exact_historical_policy["starcoder_bpb"] - r0_best_tied["starcoder_bpb"]
            ),
        },
    }


def _payload(
    cells: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    historical_dense: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    cell_payloads: dict[str, dict[str, object]] = {}
    diagnostics = []
    for _, cell in cells.iterrows():
        cell_payload, diagnostic = _cell_payload(cell, observations, candidates)
        cell_payloads[str(cell["cell_id"])] = cell_payload
        diagnostics.append(diagnostic)
    tracks = {}
    for track in TRACK_ORDER:
        selected = cells.loc[
            cells["track_memberships"].map(lambda memberships, selected_track=track: selected_track in memberships)
        ].sort_values("rung")
        if selected["rung"].tolist() != [0, 1, 2, 3]:
            raise ValueError(f"{track}: expected one cell at each rung")
        tracks[track] = {
            "label": TRACK_LABELS[track],
            "cells": selected["cell_id"].astype(str).tolist(),
        }
    return {
        "trackOrder": list(TRACK_ORDER),
        "tracks": tracks,
        "cells": cell_payloads,
        "comparison": _comparison_payload(cells, observations, historical_dense),
    }, pd.DataFrame(diagnostics)


def _html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StarCoder matched-N,D sampling explorer</title>
  <style>
    :root {{
      --ink: #17324d;
      --muted: #617386;
      --paper: #f7f3e8;
      --panel: #fffdf8;
      --line: #d8d1c2;
      --orange: #d95f32;
      --green: #1d8a72;
      --gold: #d9a521;
      --navy: #11344b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background:
      linear-gradient(rgba(23,50,77,.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(23,50,77,.035) 1px, transparent 1px), var(--paper);
      background-size: 36px 36px; font-family: Avenir Next, Avenir, Helvetica Neue, sans-serif; }}
    header {{ padding: 34px 42px 22px; border-bottom: 1px solid var(--line); background: rgba(247,243,232,.94); }}
    .eyebrow {{ color: var(--orange); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 8px; max-width: 1100px; font: 600 35px/1.12 Georgia, serif; }}
    .dek {{ max-width: 1080px; margin: 0; color: var(--muted); font-size: 16px; line-height: 1.55; }}
    .controls {{ display: grid; grid-template-columns: minmax(260px, .7fr) minmax(440px, 1.3fr) 250px; gap: 22px;
      padding: 20px 42px; align-items: end; background: rgba(255,253,248,.9); border-bottom: 1px solid var(--line); }}
    label {{ display: block; margin-bottom: 7px; color: var(--muted); font-size: 11px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }}
    select {{ width: 100%; padding: 13px 14px; color: var(--ink); border: 1px solid #bcb3a3; border-radius: 0; background: var(--panel); font: 700 15px Avenir Next, sans-serif; }}
    input[type=range] {{ width: 100%; accent-color: var(--orange); }}
    .rung-marks {{ display: flex; justify-content: space-between; margin-top: 2px; color: var(--muted); font-size: 11px; }}
    .rung-marks button {{ padding: 2px 5px; border: 0; color: inherit; background: transparent; cursor: pointer; }}
    .rung-marks button.active {{ color: var(--orange); font-weight: 800; }}
    .check {{ display: flex; align-items: center; gap: 8px; min-height: 42px; padding: 0 4px; color: var(--ink); font-size: 13px; font-weight: 700; }}
    .check input {{ accent-color: var(--orange); }}
    .checks {{ display: grid; grid-template-columns: 1fr; gap: 0; }}
    main {{ padding: 22px 28px 52px; }}
    .facts {{ display: grid; grid-template-columns: repeat(7, minmax(120px, 1fr)); border: 1px solid var(--line); background: var(--panel); }}
    .fact {{ min-height: 82px; padding: 14px 16px; border-right: 1px solid var(--line); }}
    .fact:last-child {{ border-right: 0; }}
    .fact-name {{ color: var(--muted); font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
    .fact-value {{ margin-top: 5px; font: 600 21px/1.15 Georgia, serif; }}
    .fact-sub {{ margin-top: 4px; color: var(--muted); font-size: 11px; }}
    .plots {{ display: grid; grid-template-columns: minmax(650px, 1.65fr) minmax(390px, .85fr); gap: 18px; margin-top: 18px; }}
    .panel {{ border: 1px solid var(--line); background: var(--panel); box-shadow: 0 10px 28px rgba(23,50,77,.06); }}
    .panel-head {{ display: flex; justify-content: space-between; gap: 20px; min-height: 72px; padding: 14px 18px; border-bottom: 1px solid var(--line); }}
    .panel-head h2 {{ margin: 0; font: 600 21px/1.2 Georgia, serif; }}
    .panel-head p {{ margin: 5px 0 0; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    #surface-plot {{ height: 690px; }}
    #map-plot {{ height: 560px; }}
    .readout {{ padding: 16px 18px 20px; border-top: 1px solid var(--line); }}
    .readout h3 {{ margin: 0 0 8px; color: var(--orange); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; }}
    .readout ul {{ margin: 0; padding-left: 18px; color: var(--ink); font-size: 13px; line-height: 1.5; }}
    .readout li + li {{ margin-top: 5px; }}
    .inspector {{ min-height: 92px; padding: 14px 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; line-height: 1.5; }}
    .inspector strong {{ color: var(--ink); }}
    .inspector a {{ color: var(--orange); font-weight: 800; text-decoration: none; }}
    .method {{ margin-top: 18px; padding: 18px 20px; border-left: 4px solid var(--orange); background: #fff8ec; color: var(--muted); font-size: 13px; line-height: 1.55; }}
    .method strong {{ color: var(--ink); }}
    .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin-top: 18px; border: 1px solid var(--line); background: var(--panel); }}
    .comparison-head {{ grid-column: 1 / -1; padding: 16px 20px; border-bottom: 1px solid var(--line); }}
    .comparison-head h2 {{ margin: 0; font: 600 22px/1.2 Georgia, serif; }}
    .comparison-head p {{ margin: 6px 0 0; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .comparison-card {{ padding: 18px 20px; }}
    .comparison-card + .comparison-card {{ border-left: 1px solid var(--line); }}
    .comparison-card h3 {{ margin: 0 0 9px; font: 600 18px/1.2 Georgia, serif; }}
    .comparison-card p {{ margin: 5px 0; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .comparison-card strong {{ color: var(--ink); }}
    @media (max-width: 1100px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .facts {{ grid-template-columns: repeat(2, 1fr); }}
      .fact {{ border-bottom: 1px solid var(--line); }}
      .plots {{ grid-template-columns: 1fr; }}
      .comparison {{ grid-template-columns: 1fr; }}
      .comparison-card + .comparison-card {{ border-left: 0; border-top: 1px solid var(--line); }}
      #surface-plot {{ height: 600px; }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header>
    <div class="eyebrow">Completed surface audit · 714 observed checkpoints</div>
    <h1>StarCoder 80/20 WSD matched-compute N-D surfaces</h1>
    <p class="dek">Select a scaling intervention, then move through its four rungs. The 3D mesh uses every completed Stage-1/2/3 outcome and labels the observed optima of the tied and untied policy classes. Smooth-fit inference and fresh-confirmation eligibility remain in the facts and sampling read.</p>
  </header>
  <section class="controls">
    <div><label for="track-select">Scaling setting</label><select id="track-select"></select></div>
    <div>
      <label for="rung-slider">Scaling rung</label>
      <input id="rung-slider" type="range" min="0" max="3" step="1" value="0">
      <div class="rung-marks" id="rung-marks"></div>
    </div>
    <div class="checks"><label class="check"><input id="focus-optimum" type="checkbox"> Focus vertical axis near optimum</label></div>
  </section>
  <main>
    <section class="facts" id="facts"></section>
    <section class="plots">
      <article class="panel">
        <div class="panel-head">
          <div><h2 id="surface-title">Observed response surface</h2><p>Drag to rotate; scroll to zoom; click an observed checkpoint to open its exact W&amp;B run. Vertical geometry is exaggerated for readability.</p></div>
        </div>
        <div id="surface-plot"></div>
      </article>
      <article class="panel">
        <div class="panel-head"><div><h2>Completed sampling footprint</h2><p>All markers have measured BPB. Click a checkpoint to open the exact W&amp;B run.</p></div></div>
        <div id="map-plot"></div>
        <div class="readout"><h3>Sampling read</h3><ul id="sampling-read"></ul></div>
        <div class="inspector" id="point-inspector">Click an observed point to open W&amp;B; the selected run remains summarized here.</div>
      </article>
    </section>
    <section class="comparison" id="dense-comparison">
      <div class="comparison-head"><h2>Same 1B tokens, different scaling cell</h2><p>The canonical dense panel and matched-grid r0 should not be treated as replications: the architecture, total and non-embedding parameter counts, and both TPP conventions differ.</p></div>
      <div class="comparison-card" id="historical-card"></div>
      <div class="comparison-card" id="matched-card"></div>
    </section>
    <section class="method"><strong>Interpretation boundary.</strong> Each cell has 71 or 72 observed coordinates, including 56 untied policies. The mesh is linear triangulation of measurements, not the fitted surrogate. The labeled observed optima are selection-biased; smooth quartic-ridge estimates and bootstrap diagnostics appear above and in the sampling read. Bootstrap gain support does not imply a stable exact location: inspect candidate-location L2 p90 before interpreting the coordinates.</section>
  </main>
  <script>
    const DATA = {payload_json};
    const COLORSCALE = 'RdYlGn_r';
    const NAVY = '#17324d';
    const PANEL = '#fffdf8';
    const GRID = '#ded6c7';
    const trackSelect = document.getElementById('track-select');
    const rungSlider = document.getElementById('rung-slider');
    const focusOptimum = document.getElementById('focus-optimum');

    const fmt = (value, digits=2) => Number(value).toLocaleString(undefined, {{minimumFractionDigits: digits, maximumFractionDigits: digits}});
    const sci = value => Number(value).toExponential(2);
    const currentCell = () => DATA.cells[DATA.tracks[trackSelect.value].cells[Number(rungSlider.value)]];

    function populateControls() {{
      DATA.trackOrder.forEach(track => {{
        const option = document.createElement('option');
        option.value = track;
        option.textContent = DATA.tracks[track].label;
        trackSelect.appendChild(option);
      }});
      renderRungMarks();
    }}

    function renderRungMarks() {{
      const marks = document.getElementById('rung-marks');
      marks.innerHTML = '';
      DATA.tracks[trackSelect.value].cells.forEach((cellId, rung) => {{
        const cell = DATA.cells[cellId];
        const button = document.createElement('button');
        button.type = 'button';
        button.className = rung === Number(rungSlider.value) ? 'active' : '';
        button.innerHTML = `r${{rung}} · ${{fmt(cell.parameters / 1e6, 0)}}M / ${{fmt(cell.tokens / 1e9, 2)}}B`;
        button.onclick = () => {{ rungSlider.value = rung; render(); }};
        marks.appendChild(button);
      }});
    }}

    function fact(name, value, sub='') {{
      return `<div class="fact"><div class="fact-name">${{name}}</div><div class="fact-value">${{value}}</div><div class="fact-sub">${{sub}}</div></div>`;
    }}

    function renderFacts(cell) {{
      const gate = cell.confirmationEligible ? 'passes gate' : 'does not pass';
      document.getElementById('facts').innerHTML = [
        fact('Architecture', `${{fmt(cell.parameters / 1e6, 1)}}M`, `${{cell.layers}} layers · width ${{cell.hiddenSize}}`),
        fact('Tokens D', `${{fmt(cell.tokens / 1e9, 3)}}B`, `total TPP ${{fmt(cell.totalTpp, 2)}}`),
        fact('Compute', `${{fmt(cell.computeFlops / 1e18, 3)}}e18`, `non-embed TPP ${{fmt(cell.nonEmbeddingTpp, 2)}}`),
        fact('Coordinates', `${{cell.x.length}} observed`, `56 untied · ${{cell.x.length - 56}} tied`),
        fact('Smooth phase gain', `${{fmt(cell.fittedGain, 6)}} BPB`, `${{gate}} · P(gain>0) ${{fmt(cell.bootstrapPositiveGainProbability, 3)}}`),
        fact('Raw min gain', `${{fmt(cell.rawGain, 6)}} BPB`, `Stage 1 ${{fmt(cell.stage1Gain, 6)}} · selection-biased`),
        fact('Location stability', `L2 p90 ${{fmt(cell.bootstrapCandidateL2P90, 3)}}`, `gain p05/p95 ${{fmt(cell.bootstrapGainP05, 4)}} / ${{fmt(cell.bootstrapGainP95, 4)}}`),
      ].join('');
      document.getElementById('surface-title').textContent = `${{DATA.tracks[trackSelect.value].label}} · rung ${{cell.rung}}`;
      document.getElementById('sampling-read').innerHTML = cell.samplingRead.map(item => `<li>${{item}}</li>`).join('');
    }}

    function observedPointTrace(cell, is3d) {{
      const indices = cell.x.map((_, index) => index);
      const trace = {{
        type: is3d ? 'scatter3d' : 'scatter',
        mode: 'markers',
        x: indices.map(index => cell.x[index]),
        y: indices.map(index => cell.y[index]),
        text: indices.map(index => cell.hover[index]),
        customdata: indices.map(index => cell.customdata[index]),
        hovertemplate: '%{{text}}<extra></extra>',
        name: 'observed checkpoints',
        marker: {{
          size: is3d ? 5.5 : 9,
          color: indices.map(index => cell.z[index]),
          colorscale: COLORSCALE,
          cmin: Math.min(...cell.z),
          cmax: cell.colorMax,
          showscale: false,
          symbol: 'circle',
          line: {{color: PANEL, width: 1.1}},
        }},
      }};
      if (is3d) trace.z = indices.map(index => cell.z[index]);
      return trace;
    }}

    function edgeCoordinates(cell, includeZ) {{
      const x = [], y = [], z = [];
      cell.edges.forEach(([left, right]) => {{
        x.push(cell.x[left], cell.x[right], null);
        y.push(cell.y[left], cell.y[right], null);
        if (includeZ) z.push(cell.z[left], cell.z[right], null);
      }});
      return {{x, y, z}};
    }}

    function highlightTrace(cell, index, name, symbol, color, is3d, size, textposition) {{
      const trace = {{
        type: is3d ? 'scatter3d' : 'scatter', mode: 'markers+text', x: [cell.x[index]], y: [cell.y[index]],
        text: [name], textposition,
        textfont: {{family: 'Avenir Next, sans-serif', size: is3d ? 11 : 12, color: NAVY}},
        hovertext: [cell.hover[index]], customdata: [cell.customdata[index]],
        hovertemplate: '%{{hovertext}}<extra></extra>', name,
        marker: {{symbol, size, color, line: {{color: PANEL, width: 2.2}}}},
      }};
      if (is3d) trace.z = [cell.z[index]];
      return trace;
    }}

    function surfaceTraces(cell) {{
      const edges = edgeCoordinates(cell, true);
      const triangles = cell.triangles;
      return [
        {{type: 'mesh3d', x: cell.x, y: cell.y, z: cell.z,
          i: triangles.map(t => t[0]), j: triangles.map(t => t[1]), k: triangles.map(t => t[2]),
          intensity: cell.z, colorscale: COLORSCALE, cmin: Math.min(...cell.z), cmax: cell.colorMax,
          opacity: .42, flatshading: false, hoverinfo: 'skip', name: 'linear triangulation', showscale: true,
          colorbar: {{title: 'BPB', len: .53, thickness: 14, x: .99}}}},
        {{type: 'scatter3d', mode: 'lines', ...edges, line: {{color: 'rgba(23,50,77,.34)', width: 2}}, hoverinfo: 'skip', name: 'triangulation edges'}},
        observedPointTrace(cell, true),
        highlightTrace(cell, cell.bestTiedIndex, 'observed 1p optimum', 'x', '#e7b416', true, 9, 'top left'),
        highlightTrace(cell, cell.bestUntiedIndex, 'observed 2p optimum', 'diamond-open', '#0b6e69', true, 10, 'bottom right'),
      ];
    }}

    function mapTraces(cell) {{
      const edges = edgeCoordinates(cell, false);
      const hullX = cell.hull.map(index => cell.x[index]);
      const hullY = cell.hull.map(index => cell.y[index]);
      const traces = [];
      traces.push(
        {{type: 'scatter', mode: 'lines', x: hullX, y: hullY, fill: 'toself', fillcolor: 'rgba(217,165,33,.08)',
          line: {{color: 'rgba(217,165,33,.5)', width: 2}}, hoverinfo: 'skip', name: 'observed hull'}},
        {{type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: {{color: 'rgba(23,50,77,.55)', width: 2, dash: 'dash'}}, hoverinfo: 'skip', name: 'tied diagonal'}},
        {{type: 'scatter', mode: 'lines', ...edges, line: {{color: 'rgba(23,50,77,.26)', width: 1}}, hoverinfo: 'skip', name: 'triangulation'}},
      );
      traces.push(
        observedPointTrace(cell, false),
        highlightTrace(cell, cell.bestTiedIndex, 'observed 1p optimum', 'x', '#e7b416', false, 15, 'bottom left'),
        highlightTrace(cell, cell.bestUntiedIndex, 'observed 2p optimum', 'diamond-open', '#0b6e69', false, 16, 'top right'),
      );
      return traces;
    }}

    function renderPlots(cell) {{
      const minimum = Math.min(...cell.z), maximum = Math.max(...cell.z), span = Math.max(maximum - minimum, .01);
      const focusMax = Math.min(maximum, minimum + Math.max(.045, .28 * span));
      const zRange = focusOptimum.checked ? [minimum - .025 * span, focusMax] : [minimum - .035 * span, maximum + .035 * span];
      const surfaceLayout = {{
        paper_bgcolor: PANEL, plot_bgcolor: PANEL, margin: {{l: 0, r: 30, t: 10, b: 0}},
        font: {{family: 'Avenir Next, sans-serif', color: NAVY, size: 12}}, showlegend: false, uirevision: 'matched-nd-surface',
        scene: {{
          xaxis: {{title: {{text: 'Phase 0 StarCoder weight'}}, range: [0, 1], gridcolor: GRID, backgroundcolor: '#eef2f4'}},
          yaxis: {{title: {{text: 'Phase 1 StarCoder weight'}}, range: [0, 1], gridcolor: GRID, backgroundcolor: '#eef2f4'}},
          zaxis: {{title: {{text: 'Programming BPB'}}, range: zRange, gridcolor: '#fff', backgroundcolor: '#e5edf2'}},
          aspectmode: 'manual', aspectratio: {{x: 1, y: 1, z: 1.35}},
          camera: {{eye: {{x: 1.5, y: 1.45, z: 1.05}}}},
        }},
      }};
      const mapLayout = {{
        paper_bgcolor: PANEL, plot_bgcolor: '#fbfaf5', margin: {{l: 62, r: 18, t: 18, b: 58}}, showlegend: false,
        font: {{family: 'Avenir Next, sans-serif', color: NAVY, size: 11}}, uirevision: 'matched-nd-map',
        xaxis: {{title: 'Phase 0 StarCoder weight', range: [-.02, 1.02], dtick: .2, gridcolor: GRID, zeroline: false}},
        yaxis: {{title: 'Phase 1 StarCoder weight', range: [-.02, 1.02], dtick: .2, gridcolor: GRID, zeroline: false, scaleanchor: 'x', scaleratio: 1}},
      }};
      const config = {{displaylogo: false, responsive: true, toImageButtonOptions: {{format: 'png', scale: 4}}}};
      Plotly.react('surface-plot', surfaceTraces(cell), surfaceLayout, config);
      Plotly.react('map-plot', mapTraces(cell), mapLayout, config);
      bindPointInspector('surface-plot'); bindPointInspector('map-plot');
    }}

    function bindPointInspector(plotId) {{
      const plot = document.getElementById(plotId);
      plot.removeAllListeners('plotly_click');
      plot.on('plotly_click', event => {{
        const point = event.points[0];
        if (!point || !point.customdata) return;
        const [run, stage, label, policy, url] = point.customdata;
        const link = url ? `<br><a href="${{url}}" target="_blank" rel="noopener">Open exact W&amp;B run ↗</a>` : '<br>Training outcome not yet included.';
        document.getElementById('point-inspector').innerHTML = `<strong>${{stage}} · ${{policy}}</strong><br>${{label}}<br>${{run}}${{link}}`;
        if (url) window.open(url, '_blank', 'noopener,noreferrer');
      }});
    }}

    function renderComparison() {{
      const historical = DATA.comparison.historical;
      const matched = DATA.comparison.matchedR0;
      document.getElementById('historical-card').innerHTML = `
        <h3>Canonical dense 1B · RegMix proxy</h3>
        <p><strong>${{fmt(historical.parameters / 1e6, 1)}}M total / ${{fmt(historical.nonEmbeddingParameters / 1e6, 1)}}M non-embedding</strong><br>
        TPP ${{fmt(historical.totalTpp, 2)}} / ${{fmt(historical.nonEmbeddingTpp, 2)}}</p>
        <p>Best tied (${{fmt(historical.bestTied.p0, 2)}}, ${{fmt(historical.bestTied.p1, 2)}}): ${{fmt(historical.bestTied.bpb, 6)}} BPB<br>
        Best observed untied (${{fmt(historical.bestUntied.p0, 2)}}, ${{fmt(historical.bestUntied.p1, 2)}}): ${{fmt(historical.bestUntied.bpb, 6)}} BPB</p>
        <p><strong>Reference-seed gap: ${{fmt(historical.referenceSeedObservedGain, 6)}} BPB.</strong> The separately selected phased schedule retained a ${{fmt(historical.freshSeedPairedGain, 6)}} mean gain over five matched seeds.</p>`;
      document.getElementById('matched-card').innerHTML = `
        <h3>Matched-grid r0 · Delphi architecture</h3>
        <p><strong>${{fmt(matched.parameters / 1e6, 1)}}M total / ${{fmt(matched.nonEmbeddingParameters / 1e6, 1)}}M non-embedding</strong><br>
        ${{matched.layers}} layers · width ${{matched.hiddenSize}} · TPP ${{fmt(matched.totalTpp, 2)}} / ${{fmt(matched.nonEmbeddingTpp, 2)}}</p>
        <p>Best tied (${{fmt(matched.bestTied.p0, 2)}}, ${{fmt(matched.bestTied.p1, 2)}}): ${{fmt(matched.bestTied.bpb, 6)}} BPB<br>
        Best observed untied (${{fmt(matched.bestUntied.p0, 2)}}, ${{fmt(matched.bestUntied.p1, 2)}}): ${{fmt(matched.bestUntied.bpb, 6)}} BPB</p>
        <p><strong>Current observed gain: ${{fmt(matched.observedGain, 6)}} BPB.</strong> The exact historical (0.10, 0.50) policy is ${{fmt(matched.historicalWinnerMinusBestTied, 6)}} BPB worse than r0's tied winner.</p>`;
    }}

    function render() {{
      renderRungMarks();
      const cell = currentCell();
      renderFacts(cell); renderPlots(cell);
    }}

    trackSelect.addEventListener('change', () => {{ rungSlider.value = 0; render(); }});
    rungSlider.addEventListener('input', render);
    focusOptimum.addEventListener('change', render);
    populateControls(); renderComparison(); render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    cells = _load_cells(args.source_design)
    observations = _load_observations(args.observations, cells)
    candidates = _load_candidates(args.candidates, cells)
    historical_dense = _load_historical_dense(args.historical_dense)
    payload, diagnostics = _payload(cells, observations, candidates, historical_dense)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "starcoder_wsd80_matched_nd_surface_explorer.html"
    output_path.write_text(_html(payload), encoding="utf-8")
    diagnostics.to_csv(args.output_dir / "cell_sampling_diagnostics.csv", index=False)
    print(output_path)


if __name__ == "__main__":
    main()
