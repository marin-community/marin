# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "google-cloud-storage",
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///

"""Render atomic BPB response surfaces across StarCoder WSD80 replay regimes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from google.cloud import storage
from plotly.subplots import make_subplots
from render_starcoder_wsd80_matched_nd_surface_explorer_20260802 import render_surface_explorer_html
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import t as student_t
from starcoder_wsd80_atomic_metrics import ATOMIC_METRICS, DEFAULT_METRIC_KEY, METRIC_KEYS, final_atomic_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811"
DEFAULT_COVERAGE = SOURCE_DIR / "coverage_observations.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811"
CONFIRMATION_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811"
DEFAULT_CONFIRMATION_OBSERVATIONS = CONFIRMATION_DIR / "confirmation_observations.csv"
DEFAULT_CONFIRMATION_SUMMARY = CONFIRMATION_DIR / "block_summary.csv"

GCS_BUCKET = "marin-us-central1"
GCS_PREFIX = "checkpoints/pinlin_calvin_xu/data_mixture/" "starcoder_wsd80_dense_support_surfaces_20260808"
CHECKPOINT_VERSION = "2026.07.11"
PHASE_0_FRACTION = 0.8
EXPECTED_HORIZONS = 4
EXPECTED_COORDINATES = 125
SUPPORT_ORDER = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
SUPPORT_LABELS = {
    "full": "Full physical pool (no forced replay)",
    "m0125": "0.125x simulated epoching repetition",
    "m025": "0.25x simulated epoching repetition",
    "m050": "0.5x simulated epoching repetition",
    "m100": "1x simulated epoching repetition",
    "m200": "2x simulated epoching repetition",
    "m400": "4x simulated epoching repetition",
}
SUPPORT_SHORT_LABELS = {
    "full": "full",
    "m0125": ".125x",
    "m025": ".25x",
    "m050": ".5x",
    "m100": "1x",
    "m200": "2x",
    "m400": "4x",
}
TRACES_PER_SCENE = 7
SCENE_COUNT = 4
TRACES_PER_METRIC = TRACES_PER_SCENE * SCENE_COUNT

PAPER_BACKGROUND = "#FFFDF8"
PAPER_TEXT = "#17324D"
PANE_BACKGROUND = "#E8EEF6"
GRID_COLOR = "#D8D1C2"
OBSERVED_COLOR = "#17324D"
TIED_COLOR = "#F2B134"
GLOBAL_COLOR = "#D95F3B"
UNTIED_COLOR = "#0B6E69"
VERTICAL_ASPECT_RATIO = 2.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confirmation-observations", type=Path, default=DEFAULT_CONFIRMATION_OBSERVATIONS)
    parser.add_argument("--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY)
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--write-static-image", action="store_true")
    return parser.parse_args()


def _load_confirmations(observations_path: Path, summary_path: Path) -> dict[tuple[str, str], dict[str, object]]:
    observations = pd.read_csv(observations_path)
    required_observation_fields = {
        "cell_id",
        "support_id",
        "policy_class",
        "coordinate_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "observed_bpb",
    }
    missing_observation_fields = required_observation_fields - set(observations.columns)
    if missing_observation_fields:
        raise ValueError(f"Confirmation observations are missing fields: {sorted(missing_observation_fields)}")

    grouped = observations.groupby(
        [
            "cell_id",
            "support_id",
            "policy_class",
            "coordinate_id",
            "phase_0_starcoder",
            "phase_1_starcoder",
        ],
        sort=False,
        dropna=False,
    )["observed_bpb"]
    policy_summary = grouped.agg(["count", "mean", "std"]).reset_index()
    if not policy_summary["count"].eq(5).all():
        raise ValueError("Every selected policy must have exactly five fresh repeat seeds")
    critical = student_t.ppf(0.975, policy_summary["count"] - 1)
    half_width = critical * policy_summary["std"] / np.sqrt(policy_summary["count"])
    policy_summary["ci95_low"] = policy_summary["mean"] - half_width
    policy_summary["ci95_high"] = policy_summary["mean"] + half_width

    summary = pd.read_csv(summary_path)
    required_summary_fields = {
        "cell_id",
        "support_id",
        "mean_gain_bpb",
        "ci95_low",
        "ci95_high",
        "paired_t_holm_p",
        "holm_positive",
        "tied_coordinate_id",
        "untied_coordinate_id",
    }
    missing_summary_fields = required_summary_fields - set(summary.columns)
    if missing_summary_fields:
        raise ValueError(f"Confirmation summary is missing fields: {sorted(missing_summary_fields)}")
    if summary.duplicated(["cell_id", "support_id"]).any():
        raise ValueError("Confirmation summary must contain one row per cell and support regime")

    confirmations: dict[tuple[str, str], dict[str, object]] = {}
    for block in summary.itertuples(index=False):
        block_policies = policy_summary.loc[
            policy_summary["cell_id"].eq(block.cell_id) & policy_summary["support_id"].eq(block.support_id)
        ]
        if set(block_policies["policy_class"]) != {"tied", "untied"} or len(block_policies) != 2:
            raise ValueError(f"{block.cell_id}/{block.support_id}: expected one tied and one untied repeat group")

        policies: dict[str, dict[str, object]] = {}
        for policy in block_policies.itertuples(index=False):
            expected_coordinate = (
                block.tied_coordinate_id if policy.policy_class == "tied" else block.untied_coordinate_id
            )
            if policy.coordinate_id != expected_coordinate:
                raise ValueError(
                    f"{block.cell_id}/{block.support_id}: {policy.policy_class} coordinate does not match "
                    "frozen selection"
                )
            policies[str(policy.policy_class)] = {
                "coordinateId": str(policy.coordinate_id),
                "phase0": float(policy.phase_0_starcoder),
                "phase1": float(policy.phase_1_starcoder),
                "n": int(policy.count),
                "mean": float(policy.mean),
                "ci95Low": float(policy.ci95_low),
                "ci95High": float(policy.ci95_high),
            }
        confirmations[(str(block.cell_id), str(block.support_id))] = {
            "policies": policies,
            "pairedGain": float(block.mean_gain_bpb),
            "pairedCi95Low": float(block.ci95_low),
            "pairedCi95High": float(block.ci95_high),
            "holmP": float(block.paired_t_holm_p),
            "holmPositive": bool(block.holm_positive),
        }
    expected_blocks = len(SUPPORT_ORDER) * EXPECTED_HORIZONS
    if len(confirmations) != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} fresh-repeat blocks, got {len(confirmations)}")
    return confirmations


def _load_coverage(path: Path) -> pd.DataFrame:
    coverage = pd.read_csv(path)
    required = {
        "cell_id",
        "coordinate_id",
        "run_name",
        "support_id",
        "support_role",
        "epoch_multiplier",
        "is_alias",
        "alias_of_run_name",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "aggregate_starcoder",
        "materialized_tokens",
    }
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing fields: {sorted(missing)}")
    unknown_supports = set(coverage["support_id"]) - set(SUPPORT_ORDER)
    if unknown_supports:
        raise ValueError(f"Unknown replay regimes: {sorted(unknown_supports)}")
    counts = coverage.groupby(["support_id", "cell_id"]).size()
    expected_surfaces = len(SUPPORT_ORDER) * EXPECTED_HORIZONS
    if len(counts) != expected_surfaces or not counts.eq(EXPECTED_COORDINATES).all():
        raise ValueError(
            f"Expected {EXPECTED_COORDINATES} coordinates in each of {expected_surfaces} replay-by-horizon surfaces"
        )
    if coverage["run_name"].duplicated().any():
        raise ValueError("Replay-surface run names must be unique")
    coverage["metric_run_name"] = coverage["run_name"].where(~coverage["is_alias"], coverage["alias_of_run_name"])
    if coverage["metric_run_name"].isna().any():
        raise ValueError("Every aliased design row must name its canonical checkpoint")
    coverage["materialized_tokens_b"] = coverage["materialized_tokens"] / 1e9
    coverage["policy_class"] = np.where(
        np.isclose(coverage["phase_0_starcoder"], coverage["phase_1_starcoder"]),
        "tied",
        "untied",
    )
    coverage["wandb_url"] = coverage["metric_run_name"].map(
        lambda run_name: f"https://wandb.ai/marin-community/marin/runs/{run_name}"
    )
    return coverage


def _metric_blob_name(run_name: str) -> str:
    return f"{GCS_PREFIX}/{run_name}/{CHECKPOINT_VERSION}/checkpoints/eval_metrics.jsonl"


def _fetch_atomic_metrics(bucket: storage.Bucket, run_name: str) -> dict[str, object]:
    text = bucket.blob(_metric_blob_name(run_name)).download_as_text()
    return {"run_name": run_name, **final_atomic_metrics(text, source=run_name)}


def _materialize_metrics(
    coverage: pd.DataFrame,
    path: Path,
    *,
    refresh: bool,
    workers: int,
) -> pd.DataFrame:
    expected_runs = set(coverage["metric_run_name"])
    if path.exists() and not refresh:
        metrics = pd.read_csv(path)
        if not set(METRIC_KEYS).issubset(metrics.columns):
            raise ValueError(f"Existing metric cache is missing atomic metrics: {path}")
        metrics = metrics.loc[metrics["run_name"].isin(expected_runs)].drop_duplicates("run_name", keep="last")
        if metrics[list(METRIC_KEYS)].isna().any().any():
            raise ValueError(f"Existing metric cache contains missing atomic metrics: {path}")
    else:
        metrics = pd.DataFrame(columns=["run_name", *METRIC_KEYS])

    bucket = storage.Client().bucket(GCS_BUCKET)
    missing_runs = sorted(expected_runs - set(metrics["run_name"]))
    if missing_runs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda run_name: _fetch_atomic_metrics(bucket, run_name), missing_runs))
        metrics = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)
    metrics = metrics.sort_values("run_name").reset_index(drop=True)
    if len(metrics) != len(expected_runs) or set(metrics["run_name"]) != expected_runs:
        raise ValueError("Materialized metric rows do not match the replay-by-horizon design")
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
    return metrics


def load_observations(
    coverage_path: Path,
    output_dir: Path,
    *,
    refresh: bool,
    workers: int,
) -> pd.DataFrame:
    coverage = _load_coverage(coverage_path)
    metrics = _materialize_metrics(
        coverage,
        output_dir / "full_pool_atomic_metric_observations.csv",
        refresh=refresh,
        workers=workers,
    )
    observations = coverage.merge(
        metrics,
        left_on="metric_run_name",
        right_on="run_name",
        validate="many_to_one",
        suffixes=("", "_metric"),
    ).drop(columns="run_name_metric")
    expected_aggregate = (
        PHASE_0_FRACTION * observations["phase_0_starcoder"]
        + (1.0 - PHASE_0_FRACTION) * observations["phase_1_starcoder"]
    )
    if not np.allclose(expected_aggregate, observations["aggregate_starcoder"], atol=1e-6, rtol=0.0):
        raise ValueError("Aggregate StarCoder weights do not match the 80/20 phase fractions")
    observations["support_order"] = observations["support_id"].map(
        {support_id: index for index, support_id in enumerate(SUPPORT_ORDER)}
    )
    return observations.sort_values(["support_order", "materialized_tokens", "coordinate_id"]).reset_index(drop=True)


def _triangles(group: pd.DataFrame) -> np.ndarray:
    coordinates = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    if len(np.unique(coordinates, axis=0)) != len(group):
        raise ValueError("Surface coordinates must be unique within each horizon")
    return Delaunay(coordinates).simplices


def _unique_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for triangle in triangles:
        for left, right in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges.add(tuple(sorted((int(left), int(right)))))
    return sorted(edges)


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _metric_payload(
    group: pd.DataFrame,
    points: np.ndarray,
    hull_vertices: set[int],
    confirmation: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    tied = group.loc[group["policy_class"].eq("tied")]
    untied = group.loc[group["policy_class"].eq("untied")]
    untied_points = points[np.flatnonzero(group["policy_class"].eq("untied").to_numpy())]
    payloads: dict[str, dict[str, object]] = {}
    for metric_key, metric_label in ATOMIC_METRICS:
        values = group[metric_key].to_numpy(dtype=float)
        best_tied_index = int(tied[metric_key].idxmin())
        best_untied_index = int(untied[metric_key].idxmin())
        best_tied_bpb = float(group.loc[best_tied_index, metric_key])
        best_untied_bpb = float(group.loc[best_untied_index, metric_key])
        ordered_untied = untied.sort_values(metric_key)
        best_point = points[best_untied_index]
        local_distances = np.linalg.norm(untied_points - best_point, axis=1)
        span = float(values.max() - values.min())
        payloads[metric_key] = {
            "label": metric_label,
            "z": values.tolist(),
            "colorMax": float(min(values.max(), values.min() + max(0.05, 0.35 * span))),
            "bestTiedIndex": best_tied_index,
            "bestUntiedIndex": best_untied_index,
            "bestTiedBpb": best_tied_bpb,
            "bestUntiedBpb": best_untied_bpb,
            "rawGain": best_tied_bpb - best_untied_bpb,
            "untiedMargin": float(ordered_untied.iloc[1][metric_key] - ordered_untied.iloc[0][metric_key]),
            "bestUntiedOnHull": best_untied_index in hull_vertices,
            "localUntiedNeighbors": int(np.sum((local_distances > 1e-12) & (local_distances <= 0.15))),
            "repeatConfirmation": None,
        }

        if metric_key != DEFAULT_METRIC_KEY or confirmation is None:
            continue

        coordinate_indices = {str(coordinate_id): index for index, coordinate_id in enumerate(group["coordinate_id"])}
        source_policies = cast(dict[str, dict[str, object]], confirmation["policies"])
        policies_payload: dict[str, dict[str, object]] = {}
        for policy_class in ("tied", "untied"):
            policy = dict(source_policies[policy_class])
            coordinate_id = str(policy["coordinateId"])
            if coordinate_id not in coordinate_indices:
                raise ValueError(f"Confirmation coordinate {coordinate_id} is absent from the discovery surface")
            policy["index"] = coordinate_indices[coordinate_id]
            policy["discoveryBpb"] = float(values[int(policy["index"])])
            policies_payload[policy_class] = policy

        tied_confirmation_index = int(policies_payload["tied"]["index"])
        untied_confirmation_index = int(policies_payload["untied"]["index"])
        if tied_confirmation_index != best_tied_index:
            raise ValueError("Fresh tied repeats do not match the Programming Languages discovery-selected optimum")
        payloads[metric_key]["repeatConfirmation"] = {
            **confirmation,
            "policies": policies_payload,
            "matchesRawTied": True,
            "matchesRawUntied": untied_confirmation_index == best_untied_index,
            "minimumEligibleContrast": 0.04,
        }
    return payloads


def _explorer_cell_payload(
    group: pd.DataFrame,
    rung: int,
    confirmation: dict[str, object] | None,
) -> dict[str, object]:
    group = group.sort_values("coordinate_id").reset_index(drop=True)
    points = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    triangles = _triangles(group).astype(int)
    hull = ConvexHull(points)
    hull_vertices = set(int(index) for index in hull.vertices)
    untied_points = points[np.flatnonzero(group["policy_class"].eq("untied").to_numpy())]
    metadata = group.iloc[0]
    support_id = str(metadata["support_id"])
    support_label = SUPPORT_LABELS[support_id]
    hover_base = [
        "<br>".join(
            [
                f"{support_label} · {row.policy_class}",
                f"Coordinate: {row.coordinate_id}",
                f"Phase 0 StarCoder: {row.phase_0_starcoder:.4f}",
                f"Phase 1 StarCoder: {row.phase_1_starcoder:.4f}",
                f"80/20 aggregate: {row.aggregate_starcoder:.4f}",
                f"Phase contrast p1-p0: {row.phase_1_starcoder - row.phase_0_starcoder:+.4f}",
                f"Checkpoint run: {row.metric_run_name}",
                *([f"Design alias: {row.run_name}"] if row.is_alias else []),
            ]
        )
        for row in group.itertuples(index=False)
    ]
    return {
        "cellId": str(metadata["cell_id"]),
        "supportId": support_id,
        "supportLabel": support_label,
        "supportShortLabel": SUPPORT_SHORT_LABELS[support_id],
        "epochMultiplier": None if pd.isna(metadata["epoch_multiplier"]) else float(metadata["epoch_multiplier"]),
        "rung": rung,
        "hiddenSize": int(metadata["hidden_size"]),
        "layers": 7,
        "parameters": int(metadata["total_parameters"]),
        "nonEmbeddingParameters": int(metadata["non_embedding_parameters"]),
        "tokens": int(metadata["materialized_tokens"]),
        "totalTpp": float(metadata["materialized_tokens"] / metadata["total_parameters"]),
        "nonEmbeddingTpp": float(metadata["materialized_tokens"] / metadata["non_embedding_parameters"]),
        "x": group["phase_0_starcoder"].astype(float).tolist(),
        "y": group["phase_1_starcoder"].astype(float).tolist(),
        "hoverBase": hover_base,
        "customdata": [
            [row.metric_run_name, "dense support", row.coordinate_id, row.policy_class, row.wandb_url]
            for row in group.itertuples(index=False)
        ],
        "triangles": triangles.tolist(),
        "edges": _unique_edges(triangles),
        "hull": [*map(int, hull.vertices), int(hull.vertices[0])],
        "metrics": _metric_payload(group, points, hull_vertices, confirmation),
        "untiedMedianNearestNeighbor": float(np.median(_nearest_neighbor_distances(untied_points))),
        "tiedCount": int(group["policy_class"].eq("tied").sum()),
        "untiedCount": int(group["policy_class"].eq("untied").sum()),
    }


def build_explorer_payload(
    observations: pd.DataFrame,
    confirmations: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    replay_regimes: dict[str, dict[str, object]] = {}
    for support_id in SUPPORT_ORDER:
        support = observations.loc[observations["support_id"].eq(support_id)]
        cell_ids: list[str] = []
        for rung, (_tokens, group) in enumerate(support.groupby("materialized_tokens", sort=True)):
            cell_id = str(group.iloc[0]["cell_id"])
            confirmation = confirmations.get((cell_id, support_id))
            if confirmation is None:
                raise ValueError(f"Missing fresh-repeat confirmation for {cell_id}/{support_id}")
            cell = _explorer_cell_payload(group, rung, confirmation)
            cell_key = f"{support_id}:{cell['cellId']}"
            cells[cell_key] = cell
            cell_ids.append(cell_key)
        if len(cell_ids) != EXPECTED_HORIZONS:
            raise ValueError(f"{support_id}: expected {EXPECTED_HORIZONS} horizons, got {len(cell_ids)}")
        replay_regimes[support_id] = {
            "label": SUPPORT_LABELS[support_id],
            "shortLabel": SUPPORT_SHORT_LABELS[support_id],
            "cells": cell_ids,
        }
    full_pool_cells = replay_regimes["full"]["cells"]
    return {
        "mode": "full_pool",
        "page": {
            "eyebrow": "Replay-by-horizon surface audit · 3,500 observed checkpoints",
            "title": "StarCoder 80/20 WSD atomic response surfaces",
            "dek": (
                "Switch among 23 atomic dataset-level BPB metrics, seven simulated-epoching repetition regimes, "
                "and four fixed-N token horizons. Every view uses the same complete 125-coordinate policy grid; "
                "Programming Languages also overlays fresh five-seed means and 95% intervals."
            ),
            "method": (
                "<strong>Interpretation boundary.</strong> The replay slider changes the simulated-epoching repetition "
                "target while holding the 125 policy coordinates fixed within each token horizon. Each mesh is a "
                "linear triangulation through one-seed observations, not a smooth expected-response estimate. Raw "
                "extrema and global two-phase gains are selection-biased. For Programming Languages only, vertical "
                "whiskers show marginal 95% t-intervals for the independently repeated selected tied and strictly "
                "untied coordinates. Statistical evidence comes from their paired gain interval and Holm-adjusted "
                "test in the readout, not from visual overlap of marginal intervals. Other metrics were not repeated."
            ),
        },
        "metrics": [{"key": key, "label": label} for key, label in ATOMIC_METRICS],
        "defaultMetricKey": DEFAULT_METRIC_KEY,
        "trackOrder": ["full_pool"],
        "tracks": {"full_pool": {"label": "Fixed N · increase D", "cells": full_pool_cells}},
        "replayOrder": list(SUPPORT_ORDER),
        "replayRegimes": replay_regimes,
        "cells": cells,
        "comparison": None,
    }


def _observed_customdata(group: pd.DataFrame, metric_key: str) -> np.ndarray:
    return np.column_stack(
        (
            group["coordinate_id"],
            group["run_name"],
            group["aggregate_starcoder"],
            group["policy_class"],
            group[metric_key],
            group["materialized_tokens_b"],
            group["wandb_url"],
        )
    )


def _surface_traces(
    group: pd.DataFrame,
    metric_key: str,
    confirmation: dict[str, object] | None,
    *,
    scene: str,
    visible: bool,
    show_legend: bool,
    color_min: float,
    color_max: float,
) -> list[go.BaseTraceType]:
    triangles = _triangles(group)
    metric = group[metric_key].to_numpy(dtype=float)
    tied = group.loc[group["policy_class"].eq("tied")].sort_values("phase_0_starcoder")
    untied = group.loc[group["policy_class"].eq("untied")]
    tied_minimum = tied.loc[tied[metric_key].idxmin()]
    untied_minimum = untied.loc[untied[metric_key].idxmin()]

    mesh = go.Mesh3d(
        x=group["phase_0_starcoder"],
        y=group["phase_1_starcoder"],
        z=metric,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=metric,
        colorscale="RdYlGn_r",
        cmin=color_min,
        cmax=color_max,
        opacity=0.55,
        showscale=show_legend,
        colorbar={"title": "BPB", "len": 0.30, "thickness": 14, "x": 1.015, "y": 0.26},
        hoverinfo="skip",
        name="Linear triangulation",
        legendgroup="surface",
        showlegend=show_legend,
        visible=visible,
        scene=scene,
    )
    observed = go.Scatter3d(
        x=group["phase_0_starcoder"],
        y=group["phase_1_starcoder"],
        z=metric,
        mode="markers",
        marker={
            "size": 3.5,
            "color": metric,
            "colorscale": "RdYlGn_r",
            "cmin": color_min,
            "cmax": color_max,
            "line": {"color": OBSERVED_COLOR, "width": 0.6},
            "showscale": False,
        },
        customdata=_observed_customdata(group, metric_key),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Run: %{customdata[1]}<br>"
            "Phase 0 StarCoder: %{x:.4f}<br>"
            "Phase 1 StarCoder: %{y:.4f}<br>"
            "Aggregate StarCoder: %{customdata[2]:.4f}<br>"
            "Policy: %{customdata[3]}<br>"
            "BPB: %{z:.6f}<br>"
            "Click to open W&B<extra></extra>"
        ),
        name="Observed policy",
        legendgroup="observed",
        showlegend=show_legend,
        visible=visible,
        scene=scene,
    )
    tied_curve = go.Scatter3d(
        x=tied["phase_0_starcoder"],
        y=tied["phase_1_starcoder"],
        z=tied[metric_key],
        mode="lines+markers",
        line={"color": TIED_COLOR, "width": 5},
        marker={"color": TIED_COLOR, "size": 2.5},
        hovertemplate="Tied policy<br>p=%{x:.4f}<br>BPB=%{z:.6f}<extra></extra>",
        name="Observed tied spine",
        legendgroup="tied-spine",
        showlegend=show_legend,
        visible=visible,
        scene=scene,
    )
    tied_marker = go.Scatter3d(
        x=[tied_minimum["phase_0_starcoder"]],
        y=[tied_minimum["phase_1_starcoder"]],
        z=[tied_minimum[metric_key]],
        mode="markers",
        marker={"symbol": "diamond", "size": 8, "color": TIED_COLOR, "line": {"color": "white", "width": 1}},
        hovertemplate="Best observed tied<br>p=%{x:.4f}<br>BPB=%{z:.6f}<extra></extra>",
        name="Best observed tied",
        legendgroup="tied-minimum",
        showlegend=show_legend,
        visible=visible,
        scene=scene,
    )
    untied_marker = go.Scatter3d(
        x=[untied_minimum["phase_0_starcoder"]],
        y=[untied_minimum["phase_1_starcoder"]],
        z=[untied_minimum[metric_key]],
        mode="markers",
        marker={"symbol": "diamond", "size": 9, "color": GLOBAL_COLOR, "line": {"color": "white", "width": 1}},
        hovertemplate=("Best observed strictly untied<br>p0=%{x:.4f}<br>p1=%{y:.4f}<br>BPB=%{z:.6f}<extra></extra>"),
        name="Best observed strictly untied",
        legendgroup="untied-minimum",
        showlegend=show_legend,
        visible=visible,
        scene=scene,
    )

    confirmation_payload = None
    if confirmation is not None:
        source_policies = cast(dict[str, dict[str, object]], confirmation["policies"])
        policies_payload: dict[str, dict[str, object]] = {}
        for policy_class, source_policy in source_policies.items():
            policy = dict(source_policy)
            coordinate_id = str(policy["coordinateId"])
            coordinate = group.loc[group["coordinate_id"].eq(coordinate_id)]
            if len(coordinate) != 1:
                raise ValueError(f"Expected one discovery row for confirmation coordinate {coordinate_id}")
            policy["discoveryBpb"] = float(coordinate.iloc[0][metric_key])
            policies_payload[policy_class] = policy
        confirmation_payload = {
            **confirmation,
            "policies": policies_payload,
            "matchesRawUntied": str(source_policies["untied"]["coordinateId"]) == str(untied_minimum["coordinate_id"]),
            "minimumEligibleContrast": 0.04,
        }
    fresh_traces = [
        _fresh_mean_trace(
            confirmation_payload,
            policy_class=policy_class,
            scene=scene,
            visible=visible,
            show_legend=show_legend,
        )
        for policy_class in ("tied", "untied")
    ]
    return [mesh, observed, tied_curve, tied_marker, untied_marker, *fresh_traces]


def _fresh_mean_trace(
    confirmation: dict[str, object] | None,
    *,
    policy_class: str,
    scene: str,
    visible: bool,
    show_legend: bool,
) -> go.Scatter3d:
    if confirmation is None:
        return go.Scatter3d(x=[], y=[], z=[], visible=visible, showlegend=False, scene=scene)

    policies = confirmation["policies"]
    if not isinstance(policies, dict):
        raise TypeError("Confirmation policies must be a mapping")
    policy = policies[policy_class]
    if not isinstance(policy, dict):
        raise TypeError("Confirmation policy must be a mapping")

    mean = float(policy["mean"])
    ci95_low = float(policy["ci95Low"])
    ci95_high = float(policy["ci95High"])
    paired_gain = float(confirmation["pairedGain"])
    paired_low = float(confirmation["pairedCi95Low"])
    paired_high = float(confirmation["pairedCi95High"])
    holm_p = float(confirmation["holmP"])
    holm_positive = bool(confirmation["holmPositive"])
    discovery_bpb = float(policy["discoveryBpb"])
    color = TIED_COLOR if policy_class == "tied" else UNTIED_COLOR
    interval_color = "#573700" if policy_class == "tied" else "#003f3b"
    symbol = "square" if policy_class == "tied" else "diamond"
    label = "tied" if policy_class == "tied" else "strictly untied"
    return go.Scatter3d(
        x=[float(policy["phase0"])],
        y=[float(policy["phase1"])],
        z=[mean],
        mode="markers",
        marker={"symbol": symbol, "size": 11, "color": color, "line": {"color": PAPER_TEXT, "width": 2.5}},
        error_z={
            "type": "data",
            "symmetric": False,
            "array": [ci95_high - mean],
            "arrayminus": [mean - ci95_low],
            "color": interval_color,
            "thickness": 10,
            "width": 18,
        },
        customdata=[
            [
                int(policy["n"]),
                ci95_low,
                ci95_high,
                paired_gain,
                paired_low,
                paired_high,
                holm_p,
                "positive" if holm_positive else "not positive",
                discovery_bpb,
                mean - discovery_bpb,
                (
                    "matches raw nonzero-contrast minimum"
                    if bool(confirmation["matchesRawUntied"])
                    else f"uses frozen |contrast| >= {float(confirmation['minimumEligibleContrast']):.2f} rule"
                ),
            ]
        ],
        hovertemplate=(
            f"Fresh {label} mean"
            "<br>p0=%{x:.4f}<br>p1=%{y:.4f}<br>mean BPB=%{z:.6f}"
            "<br>marginal 95% t-CI=[%{customdata[1]:.6f}, %{customdata[2]:.6f}]"
            "<br>n=%{customdata[0]} fresh paired seeds"
            "<br>one-seed discovery BPB=%{customdata[8]:.6f}"
            "<br>fresh mean - discovery=%{customdata[9]:+.6f} BPB"
            "<br><br>Paired gain (tied - untied)=%{customdata[3]:+.6f} BPB"
            "<br>paired 95% t-CI=[%{customdata[4]:+.6f}, %{customdata[5]:+.6f}]"
            "<br>Holm p=%{customdata[6]:.4g} · %{customdata[7]}"
            "<br>%{customdata[10]}<extra></extra>"
        ),
        name=f"Fresh n=5 {label} mean ± 95% CI",
        legendgroup=f"fresh-{policy_class}",
        showlegend=show_legend,
        visible=visible,
        scene=scene,
    )


def _padded_range(z_min: float, z_max: float) -> list[float]:
    padding = max(0.015 * (z_max - z_min), 1e-5)
    return [z_min - padding, z_max + padding]


def _confirmation_focused_range(
    group: pd.DataFrame,
    confirmation: dict[str, object],
) -> list[float]:
    tied = group.loc[group["policy_class"].eq("tied")]
    untied = group.loc[group["policy_class"].eq("untied")]
    policies = cast(dict[str, dict[str, object]], confirmation["policies"])
    candidates = [
        float(tied[DEFAULT_METRIC_KEY].min()),
        float(untied[DEFAULT_METRIC_KEY].min()),
        float(policies["tied"]["ci95Low"]),
        float(policies["tied"]["ci95High"]),
        float(policies["untied"]["ci95Low"]),
        float(policies["untied"]["ci95High"]),
    ]
    candidate_min = min(candidates)
    candidate_max = max(candidates)
    candidate_span = max(candidate_max - candidate_min, 0.025)
    return [candidate_min - 0.24 * candidate_span, candidate_max + 0.32 * candidate_span]


def _scene_layout(z_min: float, z_max: float) -> dict[str, object]:
    axis = {
        "range": [0.0, 1.0],
        "gridcolor": "white",
        "backgroundcolor": PANE_BACKGROUND,
        "showbackground": True,
        "zeroline": False,
    }
    return {
        "xaxis": {**axis, "title": "Phase 0 StarCoder"},
        "yaxis": {**axis, "title": "Phase 1 StarCoder"},
        "zaxis": {
            "title": "BPB",
            "range": _padded_range(z_min, z_max),
            "gridcolor": "white",
            "backgroundcolor": PANE_BACKGROUND,
            "showbackground": True,
            "zeroline": False,
        },
        "camera": {"eye": {"x": -1.55, "y": -1.55, "z": 1.25}},
        "uirevision": "full-pool-atomic-surface",
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": VERTICAL_ASPECT_RATIO},
    }


def _title(metric_label: str) -> str:
    repeat_note = " · candidate-focused z-axis · fresh n=5 intervals" if metric_label == ATOMIC_METRICS[0][1] else ""
    return (
        "<b>StarCoder WSD80 full-pool atomic response surfaces</b><br>"
        f"<sup>{metric_label} BPB · 125 observed policies per horizon · full physical pool · linear triangulation"
        f"{repeat_note}</sup>"
    )


def build_figure(
    observations: pd.DataFrame,
    confirmations: dict[tuple[str, str], dict[str, object]],
) -> go.Figure:
    horizons = [group for _tokens, group in observations.groupby("materialized_tokens", sort=True)]
    if len(horizons) != SCENE_COUNT:
        raise ValueError(f"Expected {SCENE_COUNT} horizons, got {len(horizons)}")

    subplot_titles = tuple(
        f"<b>{group['materialized_tokens_b'].iloc[0]:.2f}B materialized tokens</b>" for group in horizons
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.015,
        vertical_spacing=0.07,
        subplot_titles=subplot_titles,
    )

    scene_z_ranges: dict[tuple[str, int], list[float]] = {}
    for metric_index, (metric_key, _metric_label) in enumerate(ATOMIC_METRICS):
        color_min = float(observations[metric_key].min())
        color_max = float(observations[metric_key].max())
        for scene_index, group in enumerate(horizons):
            cell_id = str(group.iloc[0]["cell_id"])
            support_id = str(group.iloc[0]["support_id"])
            confirmation = confirmations.get((cell_id, support_id))
            if confirmation is None:
                raise ValueError(f"Missing fresh-repeat confirmation for {cell_id}/{support_id}")
            scene_z_ranges[(metric_key, scene_index)] = (
                _confirmation_focused_range(group, confirmation)
                if metric_key == DEFAULT_METRIC_KEY
                else _padded_range(color_min, color_max)
            )
            traces = _surface_traces(
                group,
                metric_key,
                confirmation if metric_key == DEFAULT_METRIC_KEY else None,
                scene=f"scene{scene_index + 1}" if scene_index else "scene",
                visible=metric_index == 0,
                show_legend=scene_index == 0,
                color_min=color_min,
                color_max=color_max,
            )
            if len(traces) != TRACES_PER_SCENE:
                raise AssertionError("Unexpected scene trace count")
            row, col = divmod(scene_index, 2)
            for trace in traces:
                figure.add_trace(trace, row=row + 1, col=col + 1)

    total_traces = len(ATOMIC_METRICS) * TRACES_PER_METRIC
    buttons = []
    for metric_index, (metric_key, metric_label) in enumerate(ATOMIC_METRICS):
        visible = [False] * total_traces
        start = metric_index * TRACES_PER_METRIC
        visible[start : start + TRACES_PER_METRIC] = [True] * TRACES_PER_METRIC
        scene_updates = {
            f"{('scene' if scene_index == 0 else f'scene{scene_index + 1}')}.zaxis.range": scene_z_ranges[
                (metric_key, scene_index)
            ]
            for scene_index in range(SCENE_COUNT)
        }
        buttons.append(
            {
                "label": metric_label,
                "method": "update",
                "args": [{"visible": visible}, {"title.text": _title(metric_label), **scene_updates}],
            }
        )

    default_key, default_label = ATOMIC_METRICS[0]
    figure.update_layout(
        title={
            "text": _title(default_label),
            "x": 0.035,
            "xanchor": "left",
            "font": {"size": 29, "family": "Georgia, Times New Roman, serif", "color": PAPER_TEXT},
        },
        width=1900,
        height=1500,
        paper_bgcolor=PAPER_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": PAPER_TEXT},
        margin={"l": 30, "r": 250, "t": 175, "b": 95},
        updatemenus=[
            {
                "buttons": buttons,
                "active": 0,
                "direction": "down",
                "showactive": True,
                "x": 1.005,
                "xanchor": "left",
                "y": 0.86,
                "yanchor": "top",
                "bgcolor": PAPER_BACKGROUND,
                "bordercolor": GRID_COLOR,
                "font": {"size": 12},
            }
        ],
        legend={
            "x": 1.005,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,253,248,0.96)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
            "font": {"size": 12},
        },
        annotations=[
            *figure.layout.annotations,
            {
                "text": (
                    "Each surface interpolates the raw one-seed observations and is not a smoothed estimate "
                    "of expected BPB. Programming Languages overlays fresh n=5 means with marginal 95% t-CIs; "
                    "use the paired gain CI and Holm test in hover for significance. Other metrics were not repeated. "
                    "Click an observed point to open its W&B run."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.04,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "font": {"size": 13, "color": PAPER_TEXT},
            },
        ],
    )
    for scene_index in range(SCENE_COUNT):
        scene_name = "scene" if scene_index == 0 else f"scene{scene_index + 1}"
        default_range = scene_z_ranges[(default_key, scene_index)]
        figure.layout[scene_name].update(_scene_layout(default_range[0], default_range[1]))
    return figure


def summarize_optima(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_key, metric_label in ATOMIC_METRICS:
        for (support_id, _tokens), group in observations.groupby(["support_id", "materialized_tokens"], sort=False):
            tied = group.loc[group["policy_class"].eq("tied")]
            tied_minimum = tied.loc[tied[metric_key].idxmin()]
            global_minimum = group.loc[group[metric_key].idxmin()]
            rows.append(
                {
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "support_id": support_id,
                    "support_label": SUPPORT_LABELS[str(support_id)],
                    "materialized_tokens_b": global_minimum["materialized_tokens_b"],
                    "global_p0": global_minimum["phase_0_starcoder"],
                    "global_p1": global_minimum["phase_1_starcoder"],
                    "global_bpb": global_minimum[metric_key],
                    "global_policy_class": global_minimum["policy_class"],
                    "tied_p": tied_minimum["phase_0_starcoder"],
                    "tied_bpb": tied_minimum[metric_key],
                    "raw_global_two_phase_gain_bpb": tied_minimum[metric_key] - global_minimum[metric_key],
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    optima: pd.DataFrame,
    confirmations: dict[tuple[str, str], dict[str, object]],
) -> None:
    uncheatable = optima.loc[optima["metric_label"].str.startswith("Uncheatable")]
    rows = len(uncheatable)
    untied = int(uncheatable["global_policy_class"].eq("untied").sum())
    positive_blocks = sum(bool(block["holmPositive"]) for block in confirmations.values())
    full_pool_blocks = [block for (cell_id, support_id), block in confirmations.items() if support_id == "full"]
    full_pool_positive = sum(bool(block["holmPositive"]) for block in full_pool_blocks)
    lines = [
        "# StarCoder WSD80 replay-by-horizon atomic response surfaces",
        "",
        (
            "This explorer renders every named dataset-level BPB response across seven repetition regimes and four "
            "horizons."
        ),
        "",
        (
            f"- Each of the {len(SUPPORT_ORDER) * EXPECTED_HORIZONS} surfaces contains "
            f"{EXPECTED_COORDINATES} observed policies."
        ),
        "- The replay slider spans the full physical pool and 0.125x through 4x simulated-epoching repetition targets.",
        "- Generic Paloma and Uncheatable micro/macro aggregates are excluded.",
        "- The mesh is a linear Delaunay triangulation through the observations, not a smooth expected-response fit.",
        f"- Uncheatable atomic components select an untied raw minimum in {untied}/{rows} metric-by-horizon cells.",
        (
            "- Programming Languages overlays fresh five-seed means and marginal 95% t-intervals for each "
            "discovery-selected tied and strictly untied coordinate."
        ),
        (
            f"- The paired tied-minus-untied gain is positive after Holm correction in {positive_blocks}/"
            f"{len(confirmations)} replay-by-horizon blocks and {full_pool_positive}/{len(full_pool_blocks)} "
            "full-pool horizons."
        ),
        "- Other atomic-metric extrema lack candidate-specific repeat confirmation and remain descriptive.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = load_observations(
        args.coverage,
        args.output_dir,
        refresh=args.refresh_metrics,
        workers=args.fetch_workers,
    )
    confirmations = _load_confirmations(args.confirmation_observations, args.confirmation_summary)
    optima = summarize_optima(observations)
    optima.to_csv(args.output_dir / "atomic_metric_raw_optima.csv", index=False)
    explorer_payload = build_explorer_payload(observations, confirmations)

    html_path = args.output_dir / "starcoder_wsd80_full_pool_atomic_surface_explorer.html"
    html_path.write_text(render_surface_explorer_html(explorer_payload), encoding="utf-8")
    if args.write_static_image:
        full_pool = observations.loc[observations["support_id"].eq("full")]
        figure = build_figure(full_pool, confirmations)
        figure.write_image(
            args.output_dir / "starcoder_wsd80_full_pool_atomic_surface_explorer.png",
            width=1900,
            height=1500,
            scale=2,
        )
    write_report(args.output_dir, optima, confirmations)
    print(html_path)


if __name__ == "__main__":
    main()
