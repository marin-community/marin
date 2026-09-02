# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
#   "wandb",
# ]
# ///

"""Collect and audit the frozen matched-N,D Stage-3 dense-surface panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_scale_bo_stage1_20260801 as scale_analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_matched_nd_stage3_dense_surfaces_20260802 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "stage3_dense_surface_results_20260802"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage3_dense_surface_design_20260802.json"
COMBINED_DISCOVERY_PATH = PANEL_DIR / "stage2_results_20260801" / "combined_discovery_observations.csv"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_matched_nd_stage3"
EXPECTED_STAGE3_RUNS = 484
EXPECTED_STAGE3_UNTIED_RUNS = 480
EXPECTED_CELLS = 10
PROMOTION_GAIN_THRESHOLD = 0.005
PROMOTION_BOOTSTRAP_PROBABILITY = 0.80
SURFACE_DEGREE = 4
SURFACE_FOLDS = 5
SURFACE_RIDGE_GRID = np.logspace(-6, 2, 17)
SURFACE_GRID_SIZE = 197
SURFACE_BOOTSTRAPS = 500
SURFACE_BOOTSTRAP_BATCH = 50
POLICY_INTERIOR_BOUND = 0.01
MIN_UNTIED_CONTRAST = 0.04
NEAR_REPLICATE_RADIUS = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=FROZEN_DESIGN_PATH)
    parser.add_argument("--existing", type=Path, default=COMBINED_DISCOVERY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def _verify_design(design: dict[str, Any], existing_path: Path) -> pd.DataFrame:
    if design.get("design_version") != "2026-08-02":
        raise ValueError("Unexpected Stage-3 design version")
    if design.get("expected_run_count") != EXPECTED_STAGE3_RUNS:
        raise ValueError("Unexpected Stage-3 run count")
    if design.get("expected_untied_run_count") != EXPECTED_STAGE3_UNTIED_RUNS:
        raise ValueError("Unexpected Stage-3 untied run count")
    if design.get("cell_count") != EXPECTED_CELLS:
        raise ValueError("Unexpected Stage-3 cell count")
    rows = design.get("runs")
    if not isinstance(rows, list) or len(rows) != EXPECTED_STAGE3_RUNS:
        raise ValueError("Frozen Stage-3 design has invalid run rows")
    manifest = pd.DataFrame(rows)
    if manifest["run_name"].duplicated().any() or manifest["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError("Frozen Stage-3 design has invalid run names or cell coverage")
    actual_manifest_hash = stream_identity.canonical_sha256(frozen_designer.launch_manifest(rows))
    expected_manifest_hash = design.get("design", {}).get("launch_manifest_sha256")
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("Frozen Stage-3 launch-manifest hash is invalid")
    source_hashes = design.get("data_use", {}).get("source_sha256", {})
    if not source_hashes:
        raise ValueError("Frozen Stage-3 design has no source hashes")
    for relative_path, expected in source_hashes.items():
        path = REPO_ROOT / relative_path
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen Stage-3 source changed: {relative_path}; {actual} != {expected}")
    existing_key = str(existing_path.relative_to(REPO_ROOT))
    if source_hashes.get(existing_key) != _sha256(existing_path):
        raise ValueError("The requested existing discovery table is not the frozen Stage-3 input")
    return manifest


def _ordered_runs(manifest: pd.DataFrame, timeout: int) -> list[Any]:
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=600))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)
    ordered = []
    for run_name in manifest["run_name"]:
        candidates = by_name.get(str(run_name), [])
        if len(candidates) != 1:
            raise ValueError(f"{run_name}: expected exactly one W&B run, found {len(candidates)}")
        ordered.append(candidates[0])
    return ordered


def _verify_streams(manifest: pd.DataFrame, runs: list[Any], existing: pd.DataFrame) -> list[str]:
    digests: list[str] = []
    by_cell: dict[str, set[str]] = {}
    for row, run in zip(manifest.to_dict("records"), runs, strict=True):
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": float(row["phase_0_starcoder"])},
            {"boundary_step": int(row["boundary_step"]), "starcoder_weight": float(row["phase_1_starcoder"])},
        ]
        differences = stream_identity.identity_differences(
            stream_identity.policy_coordinates(run.config), expected_policy
        )
        if differences:
            raise ValueError(f"{row['run_name']}: observed policy differs from the frozen design: {differences}")
        digest = stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(run.config))
        digests.append(digest)
        by_cell.setdefault(str(row["cell_id"]), set()).add(digest)
    inconsistent = {cell_id: values for cell_id, values in by_cell.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Stage-3 rows do not share one stream per cell: {inconsistent}")
    existing_streams = existing.groupby("cell_id")["stream_identity_sha256"].agg(lambda values: set(values))
    for cell_id, values in by_cell.items():
        if values != existing_streams.loc[cell_id]:
            raise ValueError(f"Stage-3 stream differs from Stages 1-2 in {cell_id}")
    return digests


def collect_stage3(
    design_path: Path,
    existing_path: Path,
    timeout: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join the frozen Stage-3 manifest to durable final metrics."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    manifest = _verify_design(design, existing_path)
    existing = pd.read_csv(existing_path)
    if len(existing) != 230 or existing["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError("The frozen existing discovery input must contain 230 rows over ten cells")
    runs = _ordered_runs(manifest, timeout)
    stream_digests = _verify_streams(manifest, runs, existing)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(scale_analysis.persisted_final_metric, runs))

    observations = manifest.copy()
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["expected_final_metric_step"] = observations["total_steps"].astype(int) - 1
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in runs]
    observations["wandb_state"] = [str(run.state) for run in runs]
    observations["wandb_url"] = [str(run.url) for run in runs]
    observations["stream_identity_sha256"] = stream_digests
    if observations["metric_uri"].nunique() != EXPECTED_STAGE3_RUNS:
        raise ValueError("Stage-3 rows do not resolve to distinct durable metric files")
    metric_is_placed = observations.apply(
        lambda row: str(row["run_name"]) in str(row["metric_uri"]),
        axis=1,
    )
    misplaced = observations.loc[~metric_is_placed]
    if not misplaced.empty:
        raise ValueError(f"Stage-3 metric path is misplaced: {misplaced[['run_name', 'metric_uri']].to_dict('records')}")
    incomplete = observations.loc[observations["final_metric_step"].ne(observations["expected_final_metric_step"])]
    if not incomplete.empty:
        raise ValueError(
            "Stage-3 contains partial checkpoints: "
            f"{incomplete[['run_name', 'final_metric_step', 'expected_final_metric_step']].to_dict('records')}"
        )
    if not np.isfinite(observations["starcoder_bpb"].to_numpy(dtype=float)).all():
        raise ValueError("Stage-3 contains non-finite BPB")
    return observations, existing, design


def _surface_features(coordinates: np.ndarray) -> np.ndarray:
    """Return a fixed quartic basis in aggregate and raw phase contrast."""
    p0 = coordinates[:, 0]
    p1 = coordinates[:, 1]
    aggregate = frozen_designer.PHASE_0_FRACTION * p0 + frozen_designer.PHASE_1_FRACTION * p1
    contrast = p1 - p0
    x = (aggregate - 0.5) / 0.5
    terms = []
    for degree in range(SURFACE_DEGREE + 1):
        for aggregate_degree in range(degree + 1):
            contrast_degree = degree - aggregate_degree
            terms.append(x**aggregate_degree * contrast**contrast_degree)
    return np.column_stack(terms)


def _ridge_operator(features: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(features.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + penalty, features.T)


def _spatial_folds(coordinates: np.ndarray) -> np.ndarray:
    x_bin = np.floor(coordinates[:, 0] * 5).astype(int)
    y_bin = np.floor(coordinates[:, 1] * 5).astype(int)
    return (x_bin + 2 * y_bin) % SURFACE_FOLDS


def _select_surface_ridge(coordinates: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    folds = _spatial_folds(coordinates)
    if set(folds) != set(range(SURFACE_FOLDS)):
        raise ValueError("The frozen spatial-CV rule produced an empty fold")
    features = _surface_features(coordinates)
    scores = []
    for ridge in SURFACE_RIDGE_GRID:
        squared_errors = []
        for fold in range(SURFACE_FOLDS):
            train = folds != fold
            test = ~train
            coefficients = _ridge_operator(features[train], float(ridge)) @ target[train]
            squared_errors.extend(np.square(features[test] @ coefficients - target[test]))
        scores.append(float(np.sqrt(np.mean(squared_errors))))
    selected = int(np.argmin(scores))
    return float(SURFACE_RIDGE_GRID[selected]), scores[selected]


def _convex_hull(coordinates: np.ndarray) -> np.ndarray:
    points = sorted({(float(row[0]), float(row[1])) for row in coordinates})
    if len(points) < 3:
        raise ValueError("At least three unique coordinates are required for a surface hull")

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _inside_convex_hull(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge = end - start
        cross = edge[0] * (points[:, 1] - start[1]) - edge[1] * (points[:, 0] - start[0])
        inside &= cross >= -1e-12
    return inside


def _surface_grid(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(POLICY_INTERIOR_BOUND, 1.0 - POLICY_INTERIOR_BOUND, SURFACE_GRID_SIZE)
    p0, p1 = np.meshgrid(axis, axis, indexing="ij")
    candidates = np.column_stack((p0.ravel(), p1.ravel()))
    candidates = candidates[np.abs(candidates[:, 1] - candidates[:, 0]) >= MIN_UNTIED_CONTRAST]
    hull = _convex_hull(coordinates)
    candidates = candidates[_inside_convex_hull(candidates, hull)]
    tied = np.column_stack((axis, axis))
    tied = tied[_inside_convex_hull(tied, hull)]
    if candidates.size == 0 or tied.size == 0:
        raise ValueError("The empirical hull contains no frozen surface-grid candidates")
    return candidates, tied


def fitted_surface_candidates(combined: pd.DataFrame) -> pd.DataFrame:
    """Select discovery candidates from a frozen smooth surface, never raw minima."""
    rows: list[dict[str, Any]] = []
    for cell_id, group in combined.groupby("cell_id", sort=True):
        coordinates = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        target = group["starcoder_bpb"].to_numpy(dtype=float)
        ridge, spatial_cv_rmse = _select_surface_ridge(coordinates, target)
        features = _surface_features(coordinates)
        operator = _ridge_operator(features, ridge)
        coefficients = operator @ target
        fitted = features @ coefficients
        candidates, tied = _surface_grid(coordinates)
        candidate_features = _surface_features(candidates)
        tied_features = _surface_features(tied)
        candidate_prediction = candidate_features @ coefficients
        tied_prediction = tied_features @ coefficients
        candidate_index = int(np.argmin(candidate_prediction))
        tied_index = int(np.argmin(tied_prediction))
        selected_candidate = candidates[candidate_index]
        selected_tied = tied[tied_index]
        predicted_gain = float(tied_prediction[tied_index] - candidate_prediction[candidate_index])

        leverage = np.sum(features * operator.T, axis=1)
        residuals = (target - fitted) / np.sqrt(np.maximum(1.0 - leverage, 0.05))
        residuals -= residuals.mean()
        seed = 20260802 + int(hashlib.sha256(str(cell_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        bootstrap_gains = []
        bootstrap_candidates = []
        for start in range(0, SURFACE_BOOTSTRAPS, SURFACE_BOOTSTRAP_BATCH):
            batch = min(SURFACE_BOOTSTRAP_BATCH, SURFACE_BOOTSTRAPS - start)
            sampled = rng.integers(0, len(residuals), size=(len(residuals), batch))
            bootstrap_target = fitted[:, None] + residuals[sampled]
            bootstrap_coefficients = operator @ bootstrap_target
            candidate_bootstrap = candidate_features @ bootstrap_coefficients
            tied_bootstrap = tied_features @ bootstrap_coefficients
            candidate_indices = np.argmin(candidate_bootstrap, axis=0)
            tied_indices = np.argmin(tied_bootstrap, axis=0)
            columns = np.arange(batch)
            bootstrap_gains.extend(
                (tied_bootstrap[tied_indices, columns] - candidate_bootstrap[candidate_indices, columns]).tolist()
            )
            bootstrap_candidates.extend(candidates[candidate_indices].tolist())
        gain_samples = np.asarray(bootstrap_gains, dtype=float)
        candidate_samples = np.asarray(bootstrap_candidates, dtype=float)
        candidate_distances = np.linalg.norm(candidate_samples - selected_candidate, axis=1)
        positive_probability = float(np.mean(gain_samples > 0.0))
        rows.append(
            {
                "cell_id": cell_id,
                "selected_ridge": ridge,
                "spatial_cv_rmse": spatial_cv_rmse,
                "fitted_untied_p0": float(selected_candidate[0]),
                "fitted_untied_p1": float(selected_candidate[1]),
                "fitted_untied_bpb": float(candidate_prediction[candidate_index]),
                "fitted_tied_weight": float(selected_tied[0]),
                "fitted_tied_bpb": float(tied_prediction[tied_index]),
                "fitted_gain_tied_minus_untied_bpb": predicted_gain,
                "bootstrap_gain_p05": float(np.quantile(gain_samples, 0.05)),
                "bootstrap_gain_p50": float(np.quantile(gain_samples, 0.50)),
                "bootstrap_gain_p95": float(np.quantile(gain_samples, 0.95)),
                "bootstrap_positive_gain_probability": positive_probability,
                "bootstrap_candidate_l2_p90": float(np.quantile(candidate_distances, 0.90)),
                "confirmation_eligible": bool(
                    predicted_gain >= PROMOTION_GAIN_THRESHOLD
                    and positive_probability >= PROMOTION_BOOTSTRAP_PROBABILITY
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def near_replicate_differences(combined: pd.DataFrame) -> pd.DataFrame:
    """Measure the local difference scale from distinct nearby coordinates."""
    rows = []
    for cell_id, group in combined.groupby("cell_id", sort=True):
        records = group.reset_index(drop=True)
        coordinates = records[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
                if not 0.0 < distance <= NEAR_REPLICATE_RADIUS:
                    continue
                left_bpb = float(records.loc[left, "starcoder_bpb"])
                right_bpb = float(records.loc[right, "starcoder_bpb"])
                rows.append(
                    {
                        "cell_id": cell_id,
                        "left_run": str(records.loc[left, "run_name"]),
                        "right_run": str(records.loc[right, "run_name"]),
                        "coordinate_l2": distance,
                        "absolute_bpb_difference": abs(left_bpb - right_bpb),
                    }
                )
    return pd.DataFrame(rows)


def common_support_scaling(
    stage3: pd.DataFrame, combined: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Estimate the primary scaling trend only on cross-cell-common support."""
    tied_sets = []
    for _, group in combined.groupby("cell_id", sort=True):
        tied = group.loc[np.isclose(group["phase_0_starcoder"], group["phase_1_starcoder"])]
        tied_sets.append({round(float(value), 8) for value in tied["phase_0_starcoder"]})
    common_tied = sorted(set.intersection(*tied_sets))
    if len(common_tied) < 4:
        raise ValueError("The ten cells do not share enough tied support for the frozen scaling estimator")

    scaffold = stage3.loc[stage3["acquisition_kind"].isin(("common_positive", "common_negative"))].copy()
    if len(scaffold) != 16 * EXPECTED_CELLS:
        raise ValueError("The frozen common scaffold is incomplete")
    rows = []
    for cell_id, cell_scaffold in scaffold.groupby("cell_id", sort=True):
        tied = combined.loc[
            combined["cell_id"].eq(cell_id) & np.isclose(combined["phase_0_starcoder"], combined["phase_1_starcoder"])
        ].copy()
        tied["rounded_weight"] = tied["phase_0_starcoder"].round(8)
        tied = tied.loc[tied["rounded_weight"].isin(common_tied)].sort_values("rounded_weight")
        if tied["rounded_weight"].duplicated().any() or len(tied) != len(common_tied):
            raise ValueError(f"Cell {cell_id} does not have one row at every shared tied coordinate")
        for row in cell_scaffold.to_dict("records"):
            tied_bpb = float(
                np.interp(
                    float(row["aggregate_starcoder"]),
                    tied["rounded_weight"].to_numpy(dtype=float),
                    tied["starcoder_bpb"].to_numpy(dtype=float),
                )
            )
            rows.append(
                {
                    "cell_id": cell_id,
                    "coordinate_id": (
                        f"p0={float(row['phase_0_starcoder']):.6f},p1={float(row['phase_1_starcoder']):.6f}"
                    ),
                    "phase_0_starcoder": float(row["phase_0_starcoder"]),
                    "phase_1_starcoder": float(row["phase_1_starcoder"]),
                    "aggregate_starcoder": float(row["aggregate_starcoder"]),
                    "phase_contrast": float(row["phase_contrast"]),
                    "starcoder_bpb": float(row["starcoder_bpb"]),
                    "interpolated_tied_bpb": tied_bpb,
                    "phase_effect_bpb": float(row["starcoder_bpb"] - tied_bpb),
                    "total_parameters": int(row["total_parameters"]),
                    "materialized_tokens": int(row["materialized_tokens"]),
                }
            )
    effects = pd.DataFrame(rows)
    coordinate_ids = sorted(effects["coordinate_id"].unique())
    reference_parameters = float(effects["total_parameters"].min())
    reference_tokens = float(effects["materialized_tokens"].min())
    columns = [np.ones(len(effects))]
    names = ["intercept"]
    for coordinate_id in coordinate_ids[1:]:
        columns.append(effects["coordinate_id"].eq(coordinate_id).to_numpy(dtype=float))
        names.append(f"coordinate[{coordinate_id}]")
    log_n = np.log(effects["total_parameters"].to_numpy(dtype=float) / reference_parameters)
    log_d = np.log(effects["materialized_tokens"].to_numpy(dtype=float) / reference_tokens)
    columns.extend((log_n, log_d, log_n * log_d))
    names.extend(("log_total_parameters", "log_materialized_tokens", "log_n_x_log_d"))
    matrix = np.column_stack(columns)
    target = effects["phase_effect_bpb"].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    residuals = target - matrix @ coefficients
    bread = np.linalg.pinv(matrix.T @ matrix)
    meat = np.zeros((matrix.shape[1], matrix.shape[1]))
    for cell_id in effects["cell_id"].unique():
        mask = effects["cell_id"].eq(cell_id).to_numpy()
        score = matrix[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    clusters = effects["cell_id"].nunique()
    correction = clusters / (clusters - 1) * (len(effects) - 1) / (len(effects) - matrix.shape[1])
    covariance = correction * bread @ meat @ bread
    regression = pd.DataFrame(
        {
            "term": names,
            "estimate": coefficients,
            "cell_clustered_se": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        }
    )
    return effects, regression, common_tied


def summarize(stage3: pd.DataFrame, existing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize descriptive discrete optima and fiber profiles after Stage 3."""
    first = existing.copy()
    first["source_stage"] = first["source_stage"].astype(str)
    second = stage3.copy()
    second["source_stage"] = "stage3"
    second["selection_label"] = second["acquisition_kind"].astype(str)
    shared = sorted(set(first.columns) & set(second.columns))
    combined = pd.concat([first[shared], second[shared]], ignore_index=True)
    combined = combined.assign(
        policy_class=pd.Series(
            np.where(np.isclose(combined["phase_0_starcoder"], combined["phase_1_starcoder"]), "tied", "untied"),
            index=combined.index,
            dtype=str,
        )
    )

    rows: list[dict[str, Any]] = []
    for cell_id, group in combined.groupby("cell_id", sort=True):
        tied = group.loc[group["policy_class"].eq("tied")]
        untied = group.loc[group["policy_class"].eq("untied")]
        best_tied = tied.loc[tied["starcoder_bpb"].idxmin()]
        best_untied = untied.loc[untied["starcoder_bpb"].idxmin()]
        gain = float(best_tied["starcoder_bpb"] - best_untied["starcoder_bpb"])
        rows.append(
            {
                "cell_id": cell_id,
                "rung": int(best_tied["rung"]),
                "hidden_size": int(best_tied["hidden_size"]),
                "materialized_tokens": int(best_tied["materialized_tokens"]),
                "total_parameters": int(best_tied["total_parameters"]),
                "total_parameter_tpp": float(best_tied["materialized_tokens"] / best_tied["total_parameters"]),
                "best_tied_source_stage": str(best_tied["source_stage"]),
                "best_tied_label": str(best_tied["selection_label"]),
                "best_tied_weight": float(best_tied["phase_0_starcoder"]),
                "best_tied_bpb": float(best_tied["starcoder_bpb"]),
                "best_untied_source_stage": str(best_untied["source_stage"]),
                "best_untied_label": str(best_untied["selection_label"]),
                "best_untied_p0": float(best_untied["phase_0_starcoder"]),
                "best_untied_p1": float(best_untied["phase_1_starcoder"]),
                "best_untied_aggregate": float(
                    frozen_designer.PHASE_0_FRACTION * best_untied["phase_0_starcoder"]
                    + frozen_designer.PHASE_1_FRACTION * best_untied["phase_1_starcoder"]
                ),
                "best_untied_contrast": float(best_untied["phase_1_starcoder"] - best_untied["phase_0_starcoder"]),
                "best_untied_bpb": float(best_untied["starcoder_bpb"]),
                "discovery_gain_tied_minus_untied_bpb": gain,
            }
        )
    cell_summary = pd.DataFrame(rows).sort_values(["rung", "cell_id"]).reset_index(drop=True)
    primary_fibers = stage3.loc[
        stage3["acquisition_kind"].isin(
            ("primary_tied_anchor", "primary_fiber", "secondary_tied_anchor", "secondary_fiber")
        )
    ].sort_values(["cell_id", "acquisition_kind", "phase_contrast"])
    return combined, cell_summary, primary_fibers


def write_report(
    output_dir: Path,
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    surface_candidates: pd.DataFrame,
    promotions: pd.DataFrame,
    primary_fibers: pd.DataFrame,
    near_replicates: pd.DataFrame,
    scaling_regression: pd.DataFrame,
    common_tied: list[float],
    design: dict[str, Any],
) -> None:
    local_difference_scale = (
        float(np.sqrt(np.mean(np.square(near_replicates["absolute_bpb_difference"]))) / np.sqrt(2.0))
        if not near_replicates.empty
        else float("nan")
    )
    scaling_terms = scaling_regression.loc[
        scaling_regression["term"].isin(("log_total_parameters", "log_materialized_tokens", "log_n_x_log_d"))
    ]
    lines = [
        "# StarCoder WSD80 matched-N,D Stage-3 dense-surface outcomes",
        "",
        f"- Complete discovery panel: {len(combined)} rows over {combined['cell_id'].nunique()} cells.",
        (
            f"- Complete Stage-3 panel: {EXPECTED_STAGE3_RUNS} rows, including "
            f"{EXPECTED_STAGE3_UNTIED_RUNS} untied coordinates."
        ),
        "- Final BPB comes from durable checkpoint `eval_metrics.jsonl`; W&B supplies identity and checkpoint roots.",
        (
            "- Fresh-confirmation eligibility uses the frozen quartic ridge surface: fitted gain at least "
            f"{PROMOTION_GAIN_THRESHOLD:.3f} BPB and residual-bootstrap positive-gain probability at least "
            f"{PROMOTION_BOOTSTRAP_PROBABILITY:.0%}."
        ),
        f"- Eligible cells: {len(promotions)}/{EXPECTED_CELLS}.",
        "",
        "## Smooth fitted-surface candidates",
        "",
        surface_candidates.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Descriptive discrete minima",
        "",
        "These minima are selection-biased and do not determine confirmation eligibility.",
        "",
        summary.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Profiled-fiber minima",
        "",
        primary_fibers.loc[primary_fibers.groupby("cell_id")["starcoder_bpb"].idxmin()][
            [
                "cell_id",
                "phase_0_starcoder",
                "phase_1_starcoder",
                "aggregate_starcoder",
                "phase_contrast",
                "starcoder_bpb",
                "wandb_url",
            ]
        ].to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Common-support scaling estimate",
        "",
        f"- Shared tied coordinates: {common_tied}",
        "- Primary estimator uses only the 16 shared scaffold coordinates and this shared tied grid.",
        "",
        scaling_terms.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Local difference scale",
        "",
        (
            f"- {len(near_replicates)} distinct coordinate pairs lie within L2 <= {NEAR_REPLICATE_RADIUS:.3f}; "
            f"their RMS difference / sqrt(2) is {local_difference_scale:.7f} BPB."
        ),
        "- This is an upper bound on a local noise floor because nearby coordinates can have different true means.",
        "",
        "## Frozen interpretation boundary",
        "",
        str(design["interpretation_boundary"]),
        "",
        "## Follow-up boundary",
        "",
        f"- Selection: {design['followup_boundary']['selection']}",
        f"- Promotion: {design['followup_boundary']['promotion']}",
        f"- Scaling estimator: {design['followup_boundary']['scaling_estimator']}",
        f"- Claim limit: {design['followup_boundary']['claim_limit']}",
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    stage3, existing, design = collect_stage3(args.design, args.existing, args.wandb_timeout, args.workers)
    combined, summary, primary_fibers = summarize(stage3, existing)
    surface_candidates = fitted_surface_candidates(combined)
    promotions = surface_candidates.loc[surface_candidates["confirmation_eligible"]].copy()
    near_replicates = near_replicate_differences(combined)
    common_effects, scaling_regression, common_tied = common_support_scaling(stage3, combined)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage3.to_csv(args.output_dir / "stage3_observations.csv", index=False)
    combined.to_csv(args.output_dir / "combined_discovery_observations.csv", index=False)
    summary.to_csv(args.output_dir / "cell_discovery_summary.csv", index=False)
    surface_candidates.to_csv(args.output_dir / "fitted_surface_candidates.csv", index=False)
    promotions.to_csv(args.output_dir / "confirmation_eligible_cells.csv", index=False)
    primary_fibers.to_csv(args.output_dir / "primary_fiber_observations.csv", index=False)
    near_replicates.to_csv(args.output_dir / "near_replicate_differences.csv", index=False)
    common_effects.to_csv(args.output_dir / "common_support_phase_effects.csv", index=False)
    scaling_regression.to_csv(args.output_dir / "common_support_scaling_regression.csv", index=False)
    (args.output_dir / "source_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    write_report(
        args.output_dir,
        combined,
        summary,
        surface_candidates,
        promotions,
        primary_fibers,
        near_replicates,
        scaling_regression,
        common_tied,
        design,
    )


if __name__ == "__main__":
    main()
