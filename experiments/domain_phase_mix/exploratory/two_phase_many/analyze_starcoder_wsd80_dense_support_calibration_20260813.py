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
#   "tabulate",
# ]
# ///

"""Apply the frozen heteroskedastic analysis to the dense WSD80 support panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from google.cloud import storage
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_starcoder_wsd80_dense_horizon_replay_gain_20260811 as raw_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_starcoder_wsd80_dense_horizon_replay_surface_sensitivity_20260811 as surface_plot,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_dense_support_surface_design_20260808.json"
DEFAULT_COVERAGE = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811"
    / "coverage_observations.csv"
)
DEFAULT_CONFIRMATION = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811"
    / "block_summary.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_dense_support_calibration_results_20260813"

DESIGN_VERSION = "2026-08-08-v5"
EXPECTED_DESIGN_SHA256 = "d4ffb9079f969af808230c623555315262cb314434a21db6d36e9651b747cd48"
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
GCS_BUCKET = "marin-us-central1"
CHECKPOINT_ROOT = "checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_surfaces_20260808"
CHECKPOINT_VERSION = "2026.07.11"
REFERENCE_SEED = 20260711
EXPECTED_REPEAT_RUNS = 564
EXPECTED_CANONICAL_GROUPS = 188
EXPECTED_NOMINAL_GROUPS = 224
EXPECTED_BLOCKS = 28
EXPECTED_REPEATS_PER_GROUP = 3
EXPECTED_SEEDS_PER_GROUP = 4
CALIBRATION_COORDINATES_PER_BLOCK = 8
VARIANCE_FLOOR = 1e-8
VARIANCE_SHAPE_RIDGE = 1.0
WEIGHT_RATIO_CAP = 100.0
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_BASE_SEED = 20260808
FRESH_CONFIRMATION_MIN_GAIN = 0.005
FRESH_CONFIRMATION_MIN_POSITIVE_FRACTION = 0.8


@dataclass(frozen=True)
class PersistedMetric:
    """One exact final-step metric recovered from durable checkpoint output."""

    value: float
    step: int
    uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--selected-policies", type=Path, default=raw_plot.DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_design(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load and verify the frozen v5 design and calibration structure."""
    design = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = design.pop("design_sha256", None)
    observed_hash = _canonical_sha256(design)
    design["design_sha256"] = claimed_hash
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design hash mismatch: claimed={claimed_hash}, observed={observed_hash}")
    if design.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version: {design.get('design_version')}")
    contract = design["analysis_contract"]
    if not contract["calibration_repeats_may_not_select_models_or_optima_or_estimate_mean_response"]:
        raise ValueError("The frozen contract does not protect mean-response selection")
    if not contract["calibration_repeats_may_only_estimate_variance_and_seed_nuisance"]:
        raise ValueError("The frozen contract does not restrict calibration outcomes to nuisance estimation")
    if contract["surface_estimator"]["weight_ratio_cap"] != WEIGHT_RATIO_CAP:
        raise ValueError("Unexpected frozen weight-ratio cap")
    if contract["uncertainty"]["replicates"] != BOOTSTRAP_REPLICATES:
        raise ValueError("Unexpected frozen bootstrap replicate count")

    repeats = pd.DataFrame(row for row in design["runs"] if row.get("replicate_kind") == "calibration_repeat")
    aliases = pd.DataFrame(
        row for row in design["deterministic_aliases"] if row.get("replicate_kind") == "calibration_repeat"
    )
    if len(repeats) != EXPECTED_REPEAT_RUNS or repeats["run_name"].duplicated().any():
        raise ValueError("Calibration-repeat manifest is incomplete or duplicated")
    if set(repeats["data_seed"].astype(int)) != set(map(int, design["repeat_seeds"])):
        raise ValueError("Calibration-repeat seeds differ from the frozen manifest")
    canonical_counts = repeats.groupby(["cell_id", "support_id", "coordinate_id"], sort=False).size()
    if len(canonical_counts) != EXPECTED_CANONICAL_GROUPS or not canonical_counts.eq(EXPECTED_REPEATS_PER_GROUP).all():
        raise ValueError("Canonical calibration groups are incomplete")
    if len(aliases) != design["calibration_alias_group_count"] * EXPECTED_REPEATS_PER_GROUP:
        raise ValueError("Calibration alias rows differ from the frozen manifest")
    return design, repeats, aliases


def _metric_blob_name(run_name: str) -> str:
    return f"{CHECKPOINT_ROOT}/{run_name}/{CHECKPOINT_VERSION}/checkpoints/eval_metrics.jsonl"


def _persisted_final_metric(bucket: Any, row: dict[str, Any]) -> PersistedMetric:
    blob_name = _metric_blob_name(str(row["run_name"]))
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise ValueError(f"{row['run_name']}: missing gs://{GCS_BUCKET}/{blob_name}")
    payload = [json.loads(line) for line in blob.download_as_text().splitlines() if line.strip()]
    finite = [
        item for item in payload if item.get(PRIMARY_METRIC) is not None and math.isfinite(float(item[PRIMARY_METRIC]))
    ]
    if not finite:
        raise ValueError(f"{row['run_name']}: no finite {PRIMARY_METRIC}")
    final = max(finite, key=lambda item: int(item["step"]))
    expected_step = int(row["total_steps"]) - 1
    if int(final["step"]) != expected_step:
        raise ValueError(f"{row['run_name']}: final metric step {final['step']} != {expected_step}")
    return PersistedMetric(
        value=float(final[PRIMARY_METRIC]),
        step=int(final["step"]),
        uri=f"gs://{GCS_BUCKET}/{blob_name}",
    )


def collect_repeat_observations(
    manifest: pd.DataFrame,
    output_path: Path,
    workers: int,
    refresh: bool,
) -> pd.DataFrame:
    """Collect or verify all exact final-step calibration-repeat metrics."""
    if output_path.exists() and not refresh:
        observations = pd.read_csv(output_path)
    else:
        if workers < 1:
            raise ValueError("--workers must be positive")
        bucket = storage.Client().bucket(GCS_BUCKET)
        rows = manifest.to_dict("records")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            metrics = list(executor.map(lambda row: _persisted_final_metric(bucket, row), rows))
        observations = manifest.copy()
        observations["bpb"] = [metric.value for metric in metrics]
        observations["final_step"] = [metric.step for metric in metrics]
        observations["metric_uri"] = [metric.uri for metric in metrics]
        observations["metric_source"] = "persisted exact-final-step eval_metrics.jsonl"
        observations.to_csv(output_path, index=False)

    required = {"run_name", "cell_id", "support_id", "coordinate_id", "data_seed", "bpb", "final_step"}
    if required - set(observations.columns):
        raise ValueError("Cached calibration observations lack required columns")
    if len(observations) != EXPECTED_REPEAT_RUNS or observations["run_name"].duplicated().any():
        raise ValueError("Calibration observations are incomplete or duplicated")
    expected_steps = observations["total_steps"].astype(int) - 1
    if not observations["final_step"].astype(int).eq(expected_steps).all():
        raise ValueError("At least one calibration metric is not at the exact final step")
    if not np.isfinite(observations["bpb"].to_numpy(dtype=float)).all():
        raise ValueError("At least one calibration BPB is non-finite")
    return observations


def build_calibration_groups(
    repeats: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine each canonical group with its independent discovery-seed observation."""
    keys = ["cell_id", "support_id", "coordinate_id"]
    canonical = repeats[keys].drop_duplicates()
    discovery = coverage.merge(canonical, on=keys, how="inner", validate="one_to_one")
    if len(discovery) != EXPECTED_CANONICAL_GROUPS or discovery["is_alias"].astype(bool).any():
        raise ValueError("Canonical calibration groups do not map to 188 launched discovery rows")
    if set(discovery["data_seed"].astype(int)) != {REFERENCE_SEED}:
        raise ValueError("Discovery calibration rows do not use the frozen reference seed")

    repeat_long = repeats.copy()
    repeat_long["seed_role"] = "calibration_repeat"
    discovery_long = discovery.copy()
    discovery_long["seed_role"] = "discovery"
    long = pd.concat([discovery_long, repeat_long], ignore_index=True, sort=False)
    counts = long.groupby(keys, sort=False).size()
    if len(counts) != EXPECTED_CANONICAL_GROUPS or not counts.eq(EXPECTED_SEEDS_PER_GROUP).all():
        raise ValueError("Calibration groups do not contain four aligned seeds")

    rows: list[dict[str, Any]] = []
    metadata_columns = [
        "cell_slug",
        "rung",
        "epoch_multiplier",
        "materialized_tokens",
        "aggregate_starcoder",
        "phase_contrast",
        "phase_0_starcoder",
        "phase_1_starcoder",
    ]
    for group_key, group in long.groupby(keys, sort=True):
        values = group["bpb"].to_numpy(dtype=float)
        row: dict[str, Any] = dict(zip(keys, group_key, strict=True))
        for column in metadata_columns:
            row[column] = group[column].iloc[0]
        row.update(
            {
                "seed_count": len(values),
                "sample_mean_bpb_variance_only": float(values.mean()),
                "sample_variance_bpb2": float(values.var(ddof=1)),
                "sample_sd_bpb": float(values.std(ddof=1)),
                "min_bpb": float(values.min()),
                "max_bpb": float(values.max()),
            }
        )
        rows.append(row)
    groups = pd.DataFrame(rows)
    if len(groups) != EXPECTED_CANONICAL_GROUPS:
        raise ValueError("Unexpected canonical calibration-group count")
    return long, groups


def _block_id(frame: pd.DataFrame) -> pd.Series:
    return frame["cell_id"].astype(str).str.cat(frame["support_id"].astype(str), sep=":")


def _variance_features(frame: pd.DataFrame, blocks: list[str]) -> np.ndarray:
    block = _block_id(frame)
    one_hot = np.column_stack([block.eq(value).to_numpy(dtype=float) for value in blocks])
    aggregate = frame["aggregate_starcoder"].to_numpy(dtype=float)
    contrast = frame["phase_contrast"].to_numpy(dtype=float)
    shape = np.column_stack([aggregate, aggregate**2, np.abs(contrast), contrast**2])
    return np.column_stack([one_hot, shape])


def fit_variance_model(
    groups: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fit the frozen pooled log-variance model and derive capped inverse-variance weights."""
    blocks = sorted(_block_id(coverage).unique().tolist())
    if len(blocks) != EXPECTED_BLOCKS:
        raise ValueError("Coverage does not contain 28 cell-support blocks")
    features = _variance_features(groups, blocks)
    target = np.log(np.maximum(groups["sample_variance_bpb2"].to_numpy(dtype=float), VARIANCE_FLOOR))
    penalty = np.zeros((features.shape[1], features.shape[1]), dtype=float)
    penalty[len(blocks) :, len(blocks) :] = np.eye(4) * VARIANCE_SHAPE_RIDGE
    coefficients = np.linalg.solve(features.T @ features + penalty, features.T @ target)
    fitted_log_variance = features @ coefficients

    diagnostics = groups.copy()
    diagnostics["observed_log_variance"] = target
    diagnostics["fitted_log_variance"] = fitted_log_variance
    diagnostics["fitted_variance_bpb2"] = np.maximum(np.exp(fitted_log_variance), VARIANCE_FLOOR)
    diagnostics["fitted_sd_bpb"] = np.sqrt(diagnostics["fitted_variance_bpb2"])
    diagnostics["log_variance_residual"] = target - fitted_log_variance

    coefficient_rows = [
        {"parameter": f"block_intercept[{block}]", "coefficient": float(coefficients[index]), "penalized": False}
        for index, block in enumerate(blocks)
    ]
    coefficient_rows.extend(
        {
            "parameter": name,
            "coefficient": float(coefficients[len(blocks) + index]),
            "penalized": True,
        }
        for index, name in enumerate(("aggregate", "aggregate_squared", "absolute_contrast", "contrast_squared"))
    )
    coefficient_table = pd.DataFrame(coefficient_rows)

    weighted_coverage = coverage.copy()
    # Exact no-wrap aliases inherit the variance of their full-pool source, not a duplicated finite-support block.
    variance_frame = weighted_coverage.copy()
    variance_frame.loc[variance_frame["is_alias"].astype(bool), "support_id"] = "full"
    coverage_features = _variance_features(variance_frame, blocks)
    predicted_variance = np.maximum(np.exp(coverage_features @ coefficients), VARIANCE_FLOOR)
    weighted_coverage["predicted_variance_bpb2"] = predicted_variance
    weighted_coverage["predicted_sd_bpb"] = np.sqrt(predicted_variance)

    # Contract completion: cap unusually precise rows at 100x the noisiest row in each fitted block,
    # then normalize to mean one so the preregistered ridge grid has a stable scale.
    weighted_coverage["raw_inverse_variance_weight"] = 1.0 / predicted_variance
    weighted_coverage["surface_weight"] = np.nan
    for _, index in weighted_coverage.groupby(["cell_id", "support_id"], sort=False).groups.items():
        raw = weighted_coverage.loc[index, "raw_inverse_variance_weight"].to_numpy(dtype=float)
        capped = np.minimum(raw, raw.min() * WEIGHT_RATIO_CAP)
        weighted_coverage.loc[index, "surface_weight"] = capped / capped.mean()

    residual = target - fitted_log_variance
    total = target - target.mean()
    summary = {
        "log_variance_rmse": float(np.sqrt(np.mean(residual**2))),
        "log_variance_r2": float(1.0 - np.sum(residual**2) / np.sum(total**2)),
        "sample_sd_median": float(diagnostics["sample_sd_bpb"].median()),
        "sample_sd_min": float(diagnostics["sample_sd_bpb"].min()),
        "sample_sd_max": float(diagnostics["sample_sd_bpb"].max()),
        "predicted_sd_median": float(weighted_coverage["predicted_sd_bpb"].median()),
        "weight_ratio_max": float(
            weighted_coverage.groupby(["cell_id", "support_id"])["surface_weight"]
            .apply(lambda values: values.max() / values.min())
            .max()
        ),
    }
    return diagnostics, coefficient_table, weighted_coverage, summary


def _weighted_ridge_operator(features: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(features.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ (features * weights[:, None]) + penalty, features.T * weights)


def _select_weighted_ridge(
    coordinates: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    folds = surface_plot._spatial_folds(coordinates)
    features = surface_plot._coordinates_to_features(coordinates)
    scores: list[float] = []
    for ridge in surface_plot.SURFACE_RIDGE_GRID:
        numerator = 0.0
        denominator = 0.0
        for fold in range(surface_plot.SURFACE_FOLDS):
            train = folds != fold
            test = ~train
            if not train.any() or not test.any():
                raise ValueError(f"Spatial fold {fold} is empty")
            coefficients = _weighted_ridge_operator(features[train], weights[train], float(ridge)) @ target[train]
            errors = features[test] @ coefficients - target[test]
            numerator += float(np.sum(weights[test] * errors**2))
            denominator += float(np.sum(weights[test]))
        scores.append(math.sqrt(numerator / denominator))
    selected = int(np.argmin(scores))
    return float(surface_plot.SURFACE_RIDGE_GRID[selected]), float(scores[selected])


def _bootstrap_seed(cell_id: str, support_id: str) -> int:
    digest = hashlib.sha256(f"{cell_id}:{support_id}".encode()).hexdigest()
    return BOOTSTRAP_BASE_SEED + int(digest[:8], 16)


def fit_weighted_surfaces(weighted_coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and optimize the frozen weighted surfaces with wild-bootstrap uncertainty."""
    summaries: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for (cell_id, support_id), group in weighted_coverage.groupby(["cell_id", "support_id"], sort=True):
        if len(group) != surface_plot.EXPECTED_COORDINATES_PER_BLOCK or group["coordinate_id"].duplicated().any():
            raise ValueError(f"{cell_id}/{support_id}: incomplete coverage block")
        coordinates = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        target = group["bpb"].to_numpy(dtype=float)
        weights = group["surface_weight"].to_numpy(dtype=float)
        features = surface_plot._coordinates_to_features(coordinates)
        ridge, cv_rmse = _select_weighted_ridge(coordinates, target, weights)
        operator = _weighted_ridge_operator(features, weights, ridge)
        coefficients = operator @ target
        fitted = features @ coefficients
        untied_grid, tied_grid = surface_plot._optimization_grids(coordinates)
        untied_features = surface_plot._coordinates_to_features(untied_grid)
        tied_features = surface_plot._coordinates_to_features(tied_grid)
        untied_prediction = untied_features @ coefficients
        tied_prediction = tied_features @ coefficients
        untied_index = int(np.argmin(untied_prediction))
        tied_index = int(np.argmin(tied_prediction))
        untied = untied_grid[untied_index]
        tied = tied_grid[tied_index]
        gain = float(tied_prediction[tied_index] - untied_prediction[untied_index])

        penalty = np.eye(features.shape[1], dtype=float) * ridge
        penalty[0, 0] = 0.0
        inverse = np.linalg.inv(features.T @ (features * weights[:, None]) + penalty)
        leverage = np.sum((features @ inverse) * (features * weights[:, None]), axis=1)
        adjusted_residual = (target - fitted) / np.sqrt(np.maximum(1.0 - leverage, 1e-6))
        rng = np.random.default_rng(_bootstrap_seed(str(cell_id), str(support_id)))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(len(target), BOOTSTRAP_REPLICATES))
        bootstrap_target = fitted[:, None] + adjusted_residual[:, None] * signs
        bootstrap_coefficients = operator @ bootstrap_target
        tied_bootstrap = tied_features @ bootstrap_coefficients
        untied_bootstrap = untied_features @ bootstrap_coefficients
        tied_indices = np.argmin(tied_bootstrap, axis=0)
        untied_indices = np.argmin(untied_bootstrap, axis=0)
        columns = np.arange(BOOTSTRAP_REPLICATES)
        gains = tied_bootstrap[tied_indices, columns] - untied_bootstrap[untied_indices, columns]
        percentile_low = float(np.quantile(gains, 0.025))
        percentile_high = float(np.quantile(gains, 0.975))
        bootstrap_mean = float(gains.mean())

        for replicate in range(BOOTSTRAP_REPLICATES):
            bootstrap_rows.append(
                {
                    "cell_id": cell_id,
                    "support_id": support_id,
                    "replicate": replicate,
                    "gain_bpb": float(gains[replicate]),
                    "tied_p": float(tied_grid[tied_indices[replicate], 0]),
                    "untied_p0": float(untied_grid[untied_indices[replicate], 0]),
                    "untied_p1": float(untied_grid[untied_indices[replicate], 1]),
                }
            )

        summaries.append(
            {
                "cell_id": cell_id,
                "support_id": support_id,
                "selected_ridge": ridge,
                "weighted_spatial_cv_rmse": cv_rmse,
                "effective_degrees_of_freedom": float(leverage.sum()),
                "maximum_leverage": float(leverage.max()),
                "weighted_surface_tied_p": float(tied[0]),
                "weighted_surface_tied_bpb": float(tied_prediction[tied_index]),
                "weighted_surface_untied_p0": float(untied[0]),
                "weighted_surface_untied_p1": float(untied[1]),
                "weighted_surface_untied_bpb": float(untied_prediction[untied_index]),
                "weighted_surface_gain_bpb": gain,
                "weighted_surface_tied_nearest_design_l2": surface_plot._nearest_distance(tied, coordinates),
                "weighted_surface_untied_nearest_design_l2": surface_plot._nearest_distance(untied, coordinates),
                "bootstrap_gain_mean_bpb": bootstrap_mean,
                "bootstrap_gain_median_bpb": float(np.median(gains)),
                "bootstrap_gain_ci95_low": percentile_low,
                "bootstrap_gain_ci95_high": percentile_high,
                "bootstrap_positive_fraction": float(np.mean(gains > 0.0)),
                "bootstrap_gain_ge_0p005_fraction": float(np.mean(gains >= FRESH_CONFIRMATION_MIN_GAIN)),
                "bootstrap_optimization_bias_bpb": bootstrap_mean - gain,
                "bootstrap_bias_corrected_gain_bpb": 2.0 * gain - bootstrap_mean,
                "bootstrap_basic_ci95_low": 2.0 * gain - percentile_high,
                "bootstrap_basic_ci95_high": 2.0 * gain - percentile_low,
            }
        )
    summary = pd.DataFrame(summaries)
    bootstrap = pd.DataFrame(bootstrap_rows)
    if len(summary) != EXPECTED_BLOCKS or len(bootstrap) != EXPECTED_BLOCKS * BOOTSTRAP_REPLICATES:
        raise ValueError("Weighted-surface output has unexpected dimensions")
    return summary, bootstrap


def materialize_nominal_seed_panel(
    canonical_long: pd.DataFrame,
    aliases: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve exact aliases to form the nominal 28 by 8 by 4 aligned-seed panel."""
    repeat_lookup = canonical_long.loc[canonical_long["seed_role"].eq("calibration_repeat")].set_index("run_name")
    alias_rows: list[dict[str, Any]] = []
    for row in aliases.to_dict("records"):
        source = repeat_lookup.loc[str(row["alias_of_run_name"])]
        alias = dict(row)
        alias["bpb"] = float(source["bpb"])
        alias["seed_role"] = "calibration_repeat_alias"
        alias_rows.append(alias)
    calibration_coordinate_ids = set(canonical_long["coordinate_id"].astype(str))
    discovery_aliases = coverage.loc[
        coverage["is_alias"].astype(bool) & coverage["coordinate_id"].astype(str).isin(calibration_coordinate_ids)
    ].copy()
    if len(discovery_aliases) != EXPECTED_NOMINAL_GROUPS - EXPECTED_CANONICAL_GROUPS:
        raise ValueError("Unexpected discovery-seed calibration alias count")
    discovery_aliases["seed_role"] = "discovery_alias"
    nominal = pd.concat(
        [canonical_long, pd.DataFrame(alias_rows), discovery_aliases],
        ignore_index=True,
        sort=False,
    )
    keys = ["cell_id", "support_id", "coordinate_id", "data_seed"]
    if len(nominal) != EXPECTED_NOMINAL_GROUPS * EXPECTED_SEEDS_PER_GROUP or nominal.duplicated(keys).any():
        raise ValueError("Nominal aligned-seed panel is incomplete or duplicated")
    block_seed = (
        nominal.groupby(["cell_id", "support_id", "data_seed"], as_index=False)["bpb"]
        .mean()
        .rename(columns={"bpb": "mean_calibration_bpb"})
    )
    block_seed["centered_seed_offset_bpb"] = block_seed["mean_calibration_bpb"] - block_seed.groupby(
        ["cell_id", "support_id"]
    )["mean_calibration_bpb"].transform("mean")
    block_summary = (
        block_seed.groupby(["cell_id", "support_id"], as_index=False)["centered_seed_offset_bpb"]
        .std(ddof=1)
        .rename(columns={"centered_seed_offset_bpb": "block_seed_offset_sd_bpb"})
    )
    return block_seed, block_summary


def paired_variance_ratios(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compare finite-replay seed SD with the same full-pool cell and coordinate."""
    index = ["cell_id", "coordinate_id"]
    wide = diagnostics.pivot(index=index, columns="support_id", values="sample_sd_bpb")
    rows: list[dict[str, Any]] = []
    for support_id in raw_plot.SUPPORT_ORDER:
        if support_id == "full":
            continue
        paired = wide[[support_id, "full"]].dropna().reset_index()
        for row in paired.itertuples(index=False):
            rows.append(
                {
                    "cell_id": row.cell_id,
                    "coordinate_id": row.coordinate_id,
                    "support_id": support_id,
                    "support_label": raw_plot.SUPPORT_LABELS[support_id],
                    "finite_support_sd_bpb": float(getattr(row, support_id)),
                    "full_pool_sd_bpb": float(row.full),
                    "sd_ratio_to_full": float(getattr(row, support_id) / row.full),
                }
            )
    return pd.DataFrame(rows)


def build_comparison(
    selected_policies_path: Path,
    coverage_path: Path,
    design_path: Path,
    weighted: pd.DataFrame,
    confirmation_path: Path,
) -> pd.DataFrame:
    """Join raw, unweighted, weighted, and fresh selected-policy evidence."""
    raw = raw_plot.load_summary(selected_policies_path, coverage_path, design_path)
    unweighted = surface_plot.fit_unweighted_surfaces(coverage_path)
    confirmation = pd.read_csv(confirmation_path)[
        ["cell_id", "support_id", "mean_gain_bpb", "ci95_low", "ci95_high", "paired_t_holm_p", "holm_positive"]
    ].rename(
        columns={
            "mean_gain_bpb": "fresh_selected_gain_bpb",
            "ci95_low": "fresh_selected_ci95_low",
            "ci95_high": "fresh_selected_ci95_high",
            "paired_t_holm_p": "fresh_selected_holm_p",
            "holm_positive": "fresh_selected_holm_positive",
        }
    )
    comparison = raw.merge(unweighted, on=["cell_id", "support_id"], validate="one_to_one")
    comparison = comparison.merge(weighted, on=["cell_id", "support_id"], validate="one_to_one")
    comparison = comparison.merge(confirmation, on=["cell_id", "support_id"], validate="one_to_one")
    comparison["weighted_minus_raw_gain_bpb"] = (
        comparison["weighted_surface_gain_bpb"] - comparison["raw_two_phase_gain_bpb"]
    )
    comparison["weighted_minus_fresh_selected_gain_bpb"] = (
        comparison["weighted_surface_gain_bpb"] - comparison["fresh_selected_gain_bpb"]
    )
    return comparison.sort_values(["support_order", "rung"]).reset_index(drop=True)


def _comparison_customdata(group: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            group["cell_id"],
            group["support_id"].map(raw_plot.SUPPORT_LABELS),
            group["total_parameter_tpp"],
            group["raw_two_phase_gain_bpb"],
            group["surface_global_two_phase_gain_bpb"],
            group["weighted_surface_gain_bpb"],
            group["bootstrap_gain_ci95_low"],
            group["bootstrap_gain_ci95_high"],
            group["bootstrap_positive_fraction"],
            group["fresh_selected_gain_bpb"],
            group["fresh_selected_ci95_low"],
            group["fresh_selected_ci95_high"],
            group["weighted_spatial_cv_rmse"],
            group["selected_ridge_y"],
            group["weighted_surface_tied_p"],
            group["weighted_surface_untied_p0"],
            group["weighted_surface_untied_p1"],
            group["weighted_surface_untied_nearest_design_l2"],
            group["bootstrap_optimization_bias_bpb"],
            group["bootstrap_bias_corrected_gain_bpb"],
            group["bootstrap_basic_ci95_low"],
            group["bootstrap_basic_ci95_high"],
        ]
    )


def build_surface_comparison_figure(comparison: pd.DataFrame) -> go.Figure:
    """Plot raw, unweighted, and frozen weighted gain estimates side by side."""
    figure = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.055,
        subplot_titles=(
            "<b>Raw common-grid minima</b><br><sup>one seed · selection-biased</sup>",
            "<b>Unweighted quartic ridge</b><br><sup>diagnostic only</sup>",
            "<b>Calibration-weighted quartic ridge</b><br><sup>fitted, bias-corrected, and fresh evidence</sup>",
        ),
    )
    y_columns = ("raw_two_phase_gain_bpb", "surface_global_two_phase_gain_bpb", "weighted_surface_gain_bpb")
    for support_id in raw_plot.SUPPORT_ORDER:
        group = comparison.loc[comparison["support_id"].eq(support_id)].sort_values("materialized_tokens_b")
        customdata = _comparison_customdata(group)
        for column, y_column in enumerate(y_columns, start=1):
            figure.add_trace(
                go.Scatter(
                    x=group["materialized_tokens_b"],
                    y=group[y_column],
                    mode="lines+markers+text",
                    name=raw_plot.SUPPORT_LABELS[support_id],
                    legendgroup=support_id,
                    showlegend=column == 3,
                    line={"color": raw_plot.SUPPORT_COLORS[support_id], "width": 2.5},
                    marker={
                        "color": raw_plot.PLOT_BACKGROUND,
                        "size": 25,
                        "line": {"color": raw_plot.SUPPORT_COLORS[support_id], "width": 3.2},
                    },
                    text=[raw_plot.SUPPORT_MARKER_LABELS[support_id]] * len(group),
                    textposition="middle center",
                    textfont={"color": raw_plot.SUPPORT_COLORS[support_id], "size": 8},
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                        "Materialized tokens: %{x:.3f}B<br>Total-parameter TPP: %{customdata[2]:.2f}<br><br>"
                        "Raw grid gain: %{customdata[3]:+.6f} BPB<br>"
                        "Unweighted surface gain: %{customdata[4]:+.6f} BPB<br>"
                        "Weighted surface gain: %{customdata[5]:+.6f} BPB<br>"
                        "Wild-bootstrap 95% interval: [%{customdata[6]:+.6f}, %{customdata[7]:+.6f}]<br>"
                        "Wild-bootstrap positive fraction: %{customdata[8]:.3f}<br>"
                        "Bootstrap optimization bias: %{customdata[18]:+.6f} BPB<br>"
                        "Basic-bootstrap corrected gain: %{customdata[19]:+.6f} "
                        "[%{customdata[20]:+.6f}, %{customdata[21]:+.6f}]<br>"
                        "Fresh selected-policy gain: %{customdata[9]:+.6f} "
                        "[%{customdata[10]:+.6f}, %{customdata[11]:+.6f}]<br><br>"
                        "Weighted spatial-CV RMSE: %{customdata[12]:.6f}<br>"
                        "Selected ridge: %{customdata[13]:.6g}<br>"
                        "Weighted tied optimum: p=%{customdata[14]:.4f}<br>"
                        "Weighted untied optimum: (%{customdata[15]:.4f}, %{customdata[16]:.4f})<br>"
                        "Untied nearest sampled L2: %{customdata[17]:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=group["materialized_tokens_b"],
                y=group["bootstrap_bias_corrected_gain_bpb"],
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 9,
                    "color": raw_plot.SUPPORT_COLORS[support_id],
                    "line": {"color": raw_plot.PAPER_TEXT, "width": 0.7},
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": group["bootstrap_basic_ci95_high"] - group["bootstrap_bias_corrected_gain_bpb"],
                    "arrayminus": group["bootstrap_bias_corrected_gain_bpb"] - group["bootstrap_basic_ci95_low"],
                    "color": raw_plot.SUPPORT_COLORS[support_id],
                    "thickness": 1.2,
                    "width": 2,
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]} / %{customdata[1]}</b><br>"
                    "Basic-bootstrap corrected gain: %{y:+.6f} BPB<br>"
                    "Basic 95% interval: [%{customdata[20]:+.6f}, %{customdata[21]:+.6f}]<br>"
                    "Bootstrap optimization bias: %{customdata[18]:+.6f} BPB<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    fresh_customdata = np.column_stack([comparison["cell_id"], comparison["support_id"].map(raw_plot.SUPPORT_LABELS)])
    figure.add_trace(
        go.Scatter(
            x=comparison["materialized_tokens_b"],
            y=comparison["fresh_selected_gain_bpb"],
            mode="markers",
            marker={"symbol": "x", "size": 8, "color": raw_plot.PAPER_TEXT},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": comparison["fresh_selected_ci95_high"] - comparison["fresh_selected_gain_bpb"],
                "arrayminus": comparison["fresh_selected_gain_bpb"] - comparison["fresh_selected_ci95_low"],
                "color": raw_plot.PAPER_TEXT,
                "thickness": 1.0,
                "width": 1.5,
            },
            customdata=fresh_customdata,
            hovertemplate=(
                "<b>%{customdata[0]} / %{customdata[1]}</b><br>"
                "Fresh selected-policy gain: %{y:+.6f} BPB<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    for column in range(1, 4):
        figure.add_hline(y=0.0, line={"color": raw_plot.PAPER_TEXT, "width": 1.3}, row=1, col=column)
        figure.update_xaxes(
            type="log",
            title_text="Materialized training tokens D",
            gridcolor=raw_plot.GRID_COLOR,
            showline=True,
            linecolor=raw_plot.PAPER_TEXT,
            row=1,
            col=column,
        )
    panel_values = {
        1: comparison["raw_two_phase_gain_bpb"].to_numpy(dtype=float),
        2: comparison["surface_global_two_phase_gain_bpb"].to_numpy(dtype=float),
        3: np.concatenate(
            [
                comparison["weighted_surface_gain_bpb"].to_numpy(dtype=float),
                comparison["bootstrap_basic_ci95_low"].to_numpy(dtype=float),
                comparison["bootstrap_basic_ci95_high"].to_numpy(dtype=float),
                comparison["fresh_selected_ci95_low"].to_numpy(dtype=float),
                comparison["fresh_selected_ci95_high"].to_numpy(dtype=float),
            ]
        ),
    }
    for column, values in panel_values.items():
        padding = max(0.002, (values.max() - values.min()) * 0.08)
        figure.update_yaxes(
            title_text="Global two-phase gain (BPB)<br><sup>higher is better</sup>",
            range=[float(min(0.0, values.min()) - padding), float(max(0.0, values.max()) + padding)],
            tickformat="+.3f",
            gridcolor=raw_plot.GRID_COLOR,
            showline=True,
            linecolor=raw_plot.PAPER_TEXT,
            row=1,
            col=column,
        )
    figure.update_layout(
        title={
            "text": (
                "<b>StarCoder WSD80 dense horizon-by-replay surface audit</b><br>"
                "<sup>564 exact-step repeats · calibration BPB enters only through variance weights</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
            "font": {"size": 27, "color": raw_plot.PAPER_TEXT, "family": "Georgia, Times New Roman, serif"},
        },
        width=2300,
        height=1180,
        paper_bgcolor=raw_plot.PAPER_BACKGROUND,
        plot_bgcolor=raw_plot.PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": raw_plot.PAPER_TEXT},
        margin={"l": 120, "r": 390, "t": 150, "b": 150},
        legend={
            "title": {"text": "<b>StarCoder replay multiplier</b>"},
            "x": 1.01,
            "y": 0.98,
            "bgcolor": "rgba(255,253,248,0.96)",
            "bordercolor": raw_plot.GRID_COLOR,
            "borderwidth": 1.5,
        },
        annotations=[
            *figure.layout.annotations,
            {
                "text": (
                    "Circles: fitted gains. Diamonds: basic-bootstrap bias correction and interval. "
                    "Black x: fresh five-seed selected-policy evidence. Coverage BPB is the sole response target."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.11,
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 13, "color": raw_plot.PAPER_TEXT},
            },
        ],
    )
    return figure


def build_replay_variance_figure(ratios: pd.DataFrame) -> go.Figure:
    """Show coordinate-matched seed-SD inflation under finite-source replay."""
    figure = go.Figure()
    for support_id in raw_plot.SUPPORT_ORDER:
        if support_id == "full":
            continue
        group = ratios.loc[ratios["support_id"].eq(support_id)]
        figure.add_trace(
            go.Box(
                x=[raw_plot.SUPPORT_LABELS[support_id]] * len(group),
                y=group["sd_ratio_to_full"],
                name=raw_plot.SUPPORT_LABELS[support_id],
                marker={"color": raw_plot.SUPPORT_COLORS[support_id], "size": 7, "opacity": 0.72},
                line={"color": raw_plot.SUPPORT_COLORS[support_id], "width": 2.2},
                fillcolor="rgba(255,253,248,0.65)",
                boxpoints="all",
                jitter=0.35,
                pointpos=0.0,
                customdata=np.column_stack([group["cell_id"], group["coordinate_id"]]),
                hovertemplate=(
                    "<b>%{customdata[0]} / %{customdata[1]}</b><br>"
                    "Seed-SD ratio to full pool: %{y:.3f}x<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.add_hline(
        y=1.0,
        line={"color": raw_plot.PAPER_TEXT, "width": 1.5, "dash": "dash"},
        annotation_text="same variance as full pool",
        annotation_position="top left",
    )
    figure.update_layout(
        title={
            "text": (
                "<b>Finite-source replay increases across-seed variability</b><br>"
                "<sup>coordinate- and horizon-matched SD ratios; four aligned seeds</sup>"
            ),
            "x": 0.04,
            "font": {"size": 27, "family": "Georgia, Times New Roman, serif"},
        },
        width=1500,
        height=900,
        paper_bgcolor=raw_plot.PAPER_BACKGROUND,
        plot_bgcolor=raw_plot.PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": raw_plot.PAPER_TEXT},
        margin={"l": 120, "r": 80, "t": 140, "b": 120},
        xaxis={"title": "StarCoder simulated-epoching repetition multiplier"},
        yaxis={
            "title": "Across-seed SD / matched full-pool SD",
            "type": "log",
            "tickmode": "array",
            "tickvals": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            "ticktext": ["0.5x", "1x", "2x", "4x", "8x", "16x"],
            "gridcolor": raw_plot.GRID_COLOR,
            "showline": True,
            "linecolor": raw_plot.PAPER_TEXT,
        },
    )
    return figure


def build_variance_figure(diagnostics: pd.DataFrame) -> go.Figure:
    """Visualize empirical and fitted calibration noise over aggregate and contrast."""
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.11,
        subplot_titles=(
            "<b>Observed seed SD</b><br><sup>four aligned seeds per canonical group</sup>",
            "<b>Variance-model fit</b><br><sup>observed versus fitted SD</sup>",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=diagnostics["aggregate_starcoder"],
            y=diagnostics["sample_sd_bpb"],
            mode="markers",
            marker={
                "size": 10 + 14 * np.abs(diagnostics["phase_contrast"]),
                "color": np.abs(diagnostics["phase_contrast"]),
                "colorscale": "RdYlGn_r",
                "colorbar": {"title": "|phase contrast|", "x": 0.45},
                "line": {"color": raw_plot.PAPER_TEXT, "width": 0.7},
                "opacity": 0.82,
            },
            customdata=np.column_stack(
                [
                    diagnostics["cell_id"],
                    diagnostics["support_id"],
                    diagnostics["coordinate_id"],
                    diagnostics["phase_contrast"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]} / %{customdata[1]}</b><br>%{customdata[2]}<br>"
                "Aggregate: %{x:.3f}<br>Contrast: %{customdata[3]:+.3f}<br>Observed seed SD: %{y:.6f} BPB"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    maximum = float(max(diagnostics["sample_sd_bpb"].max(), diagnostics["fitted_sd_bpb"].max()))
    figure.add_trace(
        go.Scatter(
            x=diagnostics["fitted_sd_bpb"],
            y=diagnostics["sample_sd_bpb"],
            mode="markers",
            marker={
                "size": 10,
                "color": diagnostics["aggregate_starcoder"],
                "colorscale": "RdYlGn_r",
                "line": {"color": raw_plot.PAPER_TEXT, "width": 0.7},
                "opacity": 0.82,
            },
            customdata=np.column_stack(
                [diagnostics["cell_id"], diagnostics["support_id"], diagnostics["coordinate_id"]]
            ),
            hovertemplate=(
                "<b>%{customdata[0]} / %{customdata[1]}</b><br>%{customdata[2]}<br>"
                "Fitted SD: %{x:.6f} BPB<br>Observed SD: %{y:.6f} BPB<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=[0.0, maximum],
            y=[0.0, maximum],
            mode="lines",
            line={"color": raw_plot.PAPER_TEXT, "dash": "dash"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Aggregate StarCoder weight", row=1, col=1)
    figure.update_yaxes(title_text="Observed across-seed SD (BPB)", row=1, col=1)
    figure.update_xaxes(title_text="Fitted SD (BPB)", row=1, col=2)
    figure.update_yaxes(title_text="Observed SD (BPB)", row=1, col=2)
    figure.update_layout(
        title={
            "text": "<b>StarCoder WSD80 calibration variance audit</b>",
            "x": 0.04,
            "font": {"size": 27, "family": "Georgia, Times New Roman, serif"},
        },
        width=1800,
        height=900,
        paper_bgcolor=raw_plot.PAPER_BACKGROUND,
        plot_bgcolor=raw_plot.PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": raw_plot.PAPER_TEXT},
        margin={"l": 115, "r": 120, "t": 130, "b": 100},
    )
    figure.update_xaxes(gridcolor=raw_plot.GRID_COLOR, showline=True, linecolor=raw_plot.PAPER_TEXT)
    figure.update_yaxes(gridcolor=raw_plot.GRID_COLOR, showline=True, linecolor=raw_plot.PAPER_TEXT)
    return figure


def write_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    variance_summary: dict[str, float],
    block_seed_summary: pd.DataFrame,
    variance_ratios: pd.DataFrame,
) -> None:
    weighted_positive = int(comparison["weighted_surface_gain_bpb"].gt(0.0).sum())
    unweighted_positive = int(comparison["surface_global_two_phase_gain_bpb"].gt(0.0).sum())
    raw_positive = int(comparison["raw_two_phase_gain_bpb"].gt(0.0).sum())
    fresh_positive = int(comparison["fresh_selected_gain_bpb"].gt(0.0).sum())
    fresh_holm_positive = int(comparison["fresh_selected_holm_positive"].astype(str).str.lower().eq("true").sum())
    positive_fraction_gate = int(
        comparison["bootstrap_positive_fraction"].ge(FRESH_CONFIRMATION_MIN_POSITIVE_FRACTION).sum()
    )
    nominal_gate = int(
        (
            comparison["weighted_surface_gain_bpb"].ge(FRESH_CONFIRMATION_MIN_GAIN)
            & comparison["bootstrap_positive_fraction"].ge(FRESH_CONFIRMATION_MIN_POSITIVE_FRACTION)
        ).sum()
    )
    bootstrap_upward = int(comparison["bootstrap_gain_mean_bpb"].gt(comparison["weighted_surface_gain_bpb"]).sum())
    corrected_negative = int(comparison["bootstrap_bias_corrected_gain_bpb"].lt(0.0).sum())
    percentile_excludes_zero = int(
        (comparison["bootstrap_gain_ci95_low"].gt(0.0) | comparison["bootstrap_gain_ci95_high"].lt(0.0)).sum()
    )
    basic_excludes_zero = int(
        (comparison["bootstrap_basic_ci95_low"].gt(0.0) | comparison["bootstrap_basic_ci95_high"].lt(0.0)).sum()
    )
    replay_ratio_summary = (
        variance_ratios.groupby(["support_id", "support_label"], as_index=False)
        .agg(
            matched_groups=("sd_ratio_to_full", "size"),
            median_sd_ratio=("sd_ratio_to_full", "median"),
            groups_noisier_than_full=("sd_ratio_to_full", lambda values: int((values > 1.0).sum())),
        )
        .set_index("support_id")
    )
    replay_ratio_table = replay_ratio_summary.reset_index()[
        ["support_label", "matched_groups", "median_sd_ratio", "groups_noisier_than_full"]
    ]
    calibration_max_aggregate = float(diagnostics["aggregate_starcoder"].max())
    untied_aggregate = 0.8 * comparison["weighted_surface_untied_p0"] + 0.2 * comparison["weighted_surface_untied_p1"]
    tied_beyond_calibration = int(comparison["weighted_surface_tied_p"].gt(calibration_max_aggregate).sum())
    untied_beyond_calibration = int(untied_aggregate.gt(calibration_max_aggregate).sum())
    boundary_optima = int(comparison["weighted_surface_untied_p0"].eq(0.0).sum())
    c109 = diagnostics.loc[diagnostics["coordinate_id"].eq("c109")].copy()
    replay_order = {"full": 0.0, "m0125": 0.125, "m025": 0.25, "m050": 0.5, "m100": 1.0, "m200": 2.0, "m400": 4.0}
    c109["replay_order"] = c109["support_id"].map(replay_order)
    c109_rank_correlations = [
        float(group["replay_order"].rank().corr(group["log_variance_residual"].rank()))
        for _, group in c109.groupby("cell_id")
    ]
    variance_ratio = diagnostics["sample_variance_bpb2"] / diagnostics["fitted_variance_bpb2"]
    worst_variance_row = diagnostics.loc[int(variance_ratio.idxmax())]
    table = comparison[
        [
            "cell_id",
            "support_id",
            "raw_two_phase_gain_bpb",
            "surface_global_two_phase_gain_bpb",
            "weighted_surface_gain_bpb",
            "bootstrap_optimization_bias_bpb",
            "bootstrap_bias_corrected_gain_bpb",
            "bootstrap_gain_ci95_low",
            "bootstrap_gain_ci95_high",
            "bootstrap_basic_ci95_low",
            "bootstrap_basic_ci95_high",
            "bootstrap_positive_fraction",
            "fresh_selected_gain_bpb",
            "fresh_selected_holm_positive",
            "weighted_spatial_cv_rmse",
            "weighted_surface_untied_nearest_design_l2",
        ]
    ]
    lines = [
        "# StarCoder WSD80 dense-support calibration analysis",
        "",
        "## Data integrity",
        "",
        (
            f"- Recovered {EXPECTED_REPEAT_RUNS}/{EXPECTED_REPEAT_RUNS} launched calibration metrics at their "
            "exact final step."
        ),
        (
            f"- Combined the three repeat seeds with the independent discovery seed into "
            f"{EXPECTED_CANONICAL_GROUPS} canonical four-seed variance groups."
        ),
        (
            "- Resolved 36 nominal finite-support calibration groups as exact full-pool aliases; they were not "
            "double-counted in the variance fit."
        ),
        (
            "- Calibration BPB values enter only through estimated variances and weights. Coverage BPB remains the "
            "sole response target, and calibration rows are never candidate coordinates."
        ),
        (
            "- The discovery seed contributes both the coverage response and one of four variance observations, so "
            "weights are contract-compliant but not statistically independent of the fitted response."
        ),
        "",
        "## Main result",
        "",
        (
            "Heteroskedastic weighting does not rescue the preregistered global quartic surface. The weighted fit "
            "is unusable as a global two-phase gain or optimum estimator on this panel."
        ),
        "",
        (
            f"- Weighted and unweighted quartic surfaces return positive global gains in "
            f"{weighted_positive}/{len(comparison)} and {unweighted_positive}/{len(comparison)} blocks. Raw common-grid "
            f"minima are positive in {raw_positive}/{len(comparison)}, and fresh selected-policy means in "
            f"{fresh_positive}/{len(comparison)}; only {fresh_holm_positive}/{len(comparison)} fresh comparisons are "
            "Holm-significant."
        ),
        (
            "- A minimum over the two-dimensional untied region has more opportunities to exploit fitted noise than "
            "a minimum over the tied diagonal. The quartic estimator reports this search advantage as policy gain."
        ),
        (
            f"- The bootstrap mean exceeds the fitted gain in {bootstrap_upward}/{len(comparison)} blocks by median "
            f"{comparison['bootstrap_optimization_bias_bpb'].median():.6f} BPB. A first-order bootstrap bias correction "
            f"makes {corrected_negative}/{len(comparison)} gains negative."
        ),
        (
            f"- Percentile 95% intervals exclude zero in {percentile_excludes_zero}/{len(comparison)} blocks; basic "
            f"reflected intervals exclude zero in only {basic_excludes_zero}/{len(comparison)}. The frozen contract did "
            "not select between these interval conventions."
        ),
        (
            f"- The nominal frozen gate is triggered in {nominal_gate}/{len(comparison)} blocks, and "
            f"{positive_fraction_gate}/{len(comparison)} have bootstrap positive fraction at least "
            f"{FRESH_CONFIRMATION_MIN_POSITIVE_FRACTION:.1f}. These counts are artifacts of the upward-biased gain "
            "functional, not promotion evidence."
        ),
        (
            f"- Weighted spatial-CV RMSE spans {comparison['weighted_spatial_cv_rmse'].min():.6f} to "
            f"{comparison['weighted_spatial_cv_rmse'].max():.6f} BPB; median "
            f"{comparison['weighted_spatial_cv_rmse'].median():.6f}."
        ),
        (
            f"- The fitted surfaces retain median effective degrees of freedom "
            f"{comparison['effective_degrees_of_freedom'].median():.2f}/15; spatial CV therefore selects nearly "
            "unregularized quartics rather than a stable low-complexity response."
        ),
        (
            f"- The weighted continuous gain differs from the fresh selected-policy gain by median absolute "
            f"{comparison['weighted_minus_fresh_selected_gain_bpb'].abs().median():.6f} BPB."
        ),
        (
            f"- {boundary_optima} weighted untied optima sit on the p0 = 0 boundary; maximum leverage ranges from "
            f"{comparison['maximum_leverage'].min():.3f} to {comparison['maximum_leverage'].max():.3f}."
        ),
        "",
        (
            "The fresh five-seed selected-policy panel remains the inferential result for the already selected grid "
            "coordinates. It confirmed positive gains only at 7.41B tokens for 1x, 2x, and 4x replay after Holm "
            "correction; calibration repeats do not alter that conclusion."
        ),
        "",
        "## Noise model",
        "",
        (
            f"- Across-group seed SD: median {variance_summary['sample_sd_median']:.6f} BPB, range "
            f"[{variance_summary['sample_sd_min']:.6f}, {variance_summary['sample_sd_max']:.6f}]."
        ),
        (
            f"- Pooled log-variance model: R2 {variance_summary['log_variance_r2']:.3f}, RMSE "
            f"{variance_summary['log_variance_rmse']:.3f} log-variance units."
        ),
        (
            f"- Median fitted per-coordinate SD is {variance_summary['predicted_sd_median']:.6f} BPB; "
            f"the largest raw within-block inverse-variance weight ratio is "
            f"{variance_summary['weight_ratio_max']:.1f}, so the frozen 100x cap never activates."
        ),
        (
            f"- Median block-level aligned-seed offset SD across the eight calibration coordinates is "
            f"{block_seed_summary['block_seed_offset_sd_bpb'].median():.6f} BPB; maximum "
            f"{block_seed_summary['block_seed_offset_sd_bpb'].max():.6f}."
        ),
        (
            f"- At the 4x replay target, the coordinate-matched median seed-SD ratio versus full-pool training is "
            f"{replay_ratio_summary.loc['m400', 'median_sd_ratio']:.2f}x; "
            f"{int(replay_ratio_summary.loc['m400', 'groups_noisier_than_full'])}/"
            f"{int(replay_ratio_summary.loc['m400', 'matched_groups'])} matched groups are noisier."
        ),
        (
            f"- The calibration design ends at aggregate {calibration_max_aggregate:.2f}; "
            f"{tied_beyond_calibration}/{len(comparison)} tied and {untied_beyond_calibration}/{len(comparison)} untied "
            "weighted optima lie beyond that variance-calibration support."
        ),
        (
            f"- At calibration coordinate c109 (aggregate 0.70), replay multiplier and log-variance residual have "
            f"within-cell rank correlations from {min(c109_rank_correlations):.3f} to "
            f"{max(c109_rank_correlations):.3f}. The worst row is "
            f"{worst_variance_row['cell_id']}/{worst_variance_row['support_id']}/"
            f"{worst_variance_row['coordinate_id']}: observed variance is "
            f"{variance_ratio.max():.1f}x fitted variance. The frozen common shape misses replay-dependent boundary "
            "noise."
        ),
        "",
        "Coordinate-matched seed-SD ratios versus full-pool training:",
        "",
        replay_ratio_table.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Preregistration caveat",
        "",
        (
            "The frozen contract specified inverse-variance weights and a 100x ratio cap, but not the cap direction "
            "or weight normalization. Before computing surfaces, this implementation fixed the conventional rule: "
            "cap high precision at 100 times the noisiest row within each block, then normalize block weights to mean "
            "one. The cap direction is immaterial here because no block reaches 100x, but weight normalization remains "
            "an operational completion."
        ),
        (
            "The contract also left the bootstrap interval convention unspecified. This implementation records both "
            "percentile and basic reflected intervals; their sharply different conclusions are part of the result, not "
            "a post-hoc choice of a preferred interval. Exact final-step collection uses zero-based step "
            "num_train_steps - 1."
        ),
        (
            "Exact no-wrap aliases inherit full-pool variance. That is mechanistically correct, but it creates a "
            "discrete weight-regime change inside finite-support blocks and tilts their fit toward rows where replay is "
            "inactive."
        ),
        "",
        "## Block table",
        "",
        table.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Interpretation",
        "",
        (
            "The calibration run succeeds at its intended purpose: it quantifies strong, coordinate-dependent noise "
            "and reveals that finite-source replay itself amplifies seed variability. The frozen pooled variance shape "
            "misses "
            "replay-dependent boundary noise and extrapolates weights beyond aggregate 0.70. More importantly, the "
            "quartic gain functional turns search dimensionality and fitted noise into an apparent two-phase benefit. "
            "The replay-by-horizon result remains supported only by raw-grid and fresh paired evidence, not by a "
            "trustworthy global smooth optimum."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_figure(figure: go.Figure, html_path: Path, png_path: Path, width: int, height: int) -> None:
    figure.write_html(
        html_path,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": html_path.stem,
                "height": height * 2,
                "width": width * 2,
                "scale": 4,
            },
        },
    )
    figure.write_image(png_path, width=width, height=height, scale=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design, repeat_manifest, aliases = load_design(args.design)
    coverage = pd.read_csv(args.coverage)
    observations_path = args.output_dir / "calibration_repeat_observations.csv"
    repeats = collect_repeat_observations(
        repeat_manifest,
        observations_path,
        workers=args.workers,
        refresh=args.refresh,
    )
    canonical_long, groups = build_calibration_groups(repeats, coverage)
    diagnostics, coefficients, weighted_coverage, variance_summary = fit_variance_model(groups, coverage)
    weighted_summary, bootstrap = fit_weighted_surfaces(weighted_coverage)
    block_seed, block_seed_summary = materialize_nominal_seed_panel(canonical_long, aliases, coverage)
    variance_ratios = paired_variance_ratios(diagnostics)
    comparison = build_comparison(
        args.selected_policies,
        args.coverage,
        args.design,
        weighted_summary,
        args.confirmation,
    )

    completion = {
        "design_sha256": design["design_sha256"],
        "analysis_date": "2026-08-13",
        "weight_cap_completion_rule": "within_block_cap_high_precision_at_100x_minimum_weight_then_normalize_mean_one",
        "ridge_weighted_objective": (
            "sum_i weight_i*(prediction_i-observation_i)^2 + ridge*sum_nonintercept(coefficient^2)"
        ),
        "weighted_cv_metric": "sqrt(sum_test weight_i*error_i^2 / sum_test weight_i)",
        "alias_weight_rule": "exact_no_wrap_aliases_inherit_full_pool_source_variance",
        "bootstrap_residual_correction": "residual/sqrt(max(1-leverage,1e-6))",
        "bootstrap_interval_completion": [
            "percentile_quantiles_of_reoptimized_gain",
            "basic_reflected_interval_around_fitted_gain",
        ],
        "bootstrap_positive_summary": "fraction_of_reoptimized_bootstrap_gains_strictly_above_zero",
        "metric_step_indexing": "zero_based_final_step_equals_num_train_steps_minus_one",
        "calibration_means_forbidden": True,
    }
    completion["analysis_completion_sha256"] = _canonical_sha256(completion)
    (args.output_dir / "analysis_contract_completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    canonical_long.to_csv(args.output_dir / "calibration_aligned_seed_observations.csv", index=False)
    groups.to_csv(args.output_dir / "calibration_variance_groups.csv", index=False)
    diagnostics.to_csv(args.output_dir / "variance_model_diagnostics.csv", index=False)
    coefficients.to_csv(args.output_dir / "variance_model_coefficients.csv", index=False)
    weighted_coverage.to_csv(args.output_dir / "coverage_with_calibration_weights.csv", index=False)
    weighted_summary.to_csv(args.output_dir / "weighted_surface_summary.csv", index=False)
    bootstrap.to_csv(args.output_dir / "weighted_surface_bootstrap.csv", index=False)
    block_seed.to_csv(args.output_dir / "aligned_block_seed_offsets.csv", index=False)
    block_seed_summary.to_csv(args.output_dir / "aligned_block_seed_offset_summary.csv", index=False)
    variance_ratios.to_csv(args.output_dir / "paired_seed_sd_ratio_to_full.csv", index=False)
    comparison.to_csv(args.output_dir / "surface_evidence_comparison.csv", index=False)

    surface_figure = build_surface_comparison_figure(comparison)
    variance_figure = build_variance_figure(diagnostics)
    replay_variance_figure = build_replay_variance_figure(variance_ratios)
    _write_figure(
        surface_figure,
        args.output_dir / "starcoder_wsd80_calibration_weighted_surface_audit.html",
        args.output_dir / "starcoder_wsd80_calibration_weighted_surface_audit.png",
        2300,
        1180,
    )
    _write_figure(
        variance_figure,
        args.output_dir / "starcoder_wsd80_calibration_variance_audit.html",
        args.output_dir / "starcoder_wsd80_calibration_variance_audit.png",
        1800,
        900,
    )
    _write_figure(
        replay_variance_figure,
        args.output_dir / "starcoder_wsd80_replay_seed_variance.html",
        args.output_dir / "starcoder_wsd80_replay_seed_variance.png",
        1500,
        900,
    )
    write_report(
        args.output_dir,
        comparison,
        diagnostics,
        variance_summary,
        block_seed_summary,
        variance_ratios,
    )


if __name__ == "__main__":
    main()
