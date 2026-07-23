# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit fixed-budget mixtures of broad, tied, and frontier-fiber evidence.

Every design uses exactly 280 training checkpoints. Hyperparameters are frozen
to the existing 280-row Hierarchical phase replay fit so that the comparison
isolates acquisition composition rather than retuning the surrogate against
newly exposed outcomes.

The primary evaluation set is coordinate-disjoint from all three acquisition
pools. Composition-specific unused rows are reported only as secondary
diagnostics. Raw optima are audited for plausibility and stability; they are not
treated as validated policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    audit_raw_optima as raw_optima,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
HELDOUT_PATH = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
FIBER_RESULTS_PATH = REFERENCE_OUTPUTS / ("delphi_3e18_frontier_phase_fiber_results_20260719/observed_results.csv")
HPR_METRICS_PATH = REFERENCE_OUTPUTS / "hierarchical_coverage_grp_20260715/metrics.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_fixed_budget_frontier_composition_20260719"

ONE_PHASE_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
FIBER_SERIES = "delphi_3e18_frontier_phase_fiber_20260719"
ADVERSARIAL_SERIES = "delphi_3e18_adversarial_stress_panel_20260716"
SOURCE_SERIES = frozenset({ONE_PHASE_SERIES, FIBER_SERIES})
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
TARGET_ANCHORS = {"uncheatable": "uncheatable_frontier", "table9": "table9_frontier"}
MODEL_VARIANT = hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
MODEL_VARIANT_NAME = MODEL_VARIANT.value
TOTAL_ROWS = 280
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Allocation:
    """One fixed-cost evidence composition."""

    name: str
    broad: int
    single: int
    fiber: int
    fiber_anchor_mode: str = "both"

    def __post_init__(self) -> None:
        if self.broad + self.single + self.fiber != TOTAL_ROWS:
            raise ValueError(f"{self.name} has {self.broad + self.single + self.fiber} rows, not {TOTAL_ROWS}")
        if self.fiber_anchor_mode not in {"both", "target_matched"}:
            raise ValueError(f"Unknown fiber anchor mode {self.fiber_anchor_mode}")


ALLOCATIONS = (
    Allocation("b280_baseline", 280, 0, 0),
    Allocation("b140_s140", 140, 140, 0),
    Allocation("b180_f100_matched", 180, 0, 100, "target_matched"),
    Allocation("s180_f100_matched", 0, 180, 100, "target_matched"),
    Allocation("b100_s80_f100_matched", 100, 80, 100, "target_matched"),
    Allocation("b140_s70_f70_matched", 140, 70, 70, "target_matched"),
    Allocation("b100_s80_f100_both", 100, 80, 100, "both"),
    Allocation("s140_f140_both", 0, 140, 140, "both"),
    Allocation("b42_s238", 42, 238, 0),
    Allocation("b80_f200", 80, 0, 200, "both"),
    Allocation("s80_f200", 0, 80, 200, "both"),
)


@dataclass(frozen=True)
class Pool:
    """Policies and outcomes from one acquisition source."""

    frame: pd.DataFrame
    weights: np.ndarray


@dataclass(frozen=True)
class Sources:
    """All acquisition pools and the common external evaluation archive."""

    reference: pooled.Dataset
    broad: Pool
    single: Pool
    fiber: Pool
    common: Pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--optimizer-starts", type=int, default=6)
    parser.add_argument("--skip-optima", action="store_true")
    return parser.parse_args()


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seeds:
        raise ValueError("At least one subset seed is required")
    return seeds


def policy_hash(weights: np.ndarray) -> str:
    """Return a stable coordinate hash independent of row provenance."""
    rounded = np.round(np.asarray(weights, dtype=np.float64), decimals=14)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def parse_weights(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    def phase(column: str) -> np.ndarray:
        rows = []
        for value in frame[column]:
            record = json.loads(str(value))
            rows.append([float(record[domain]) for domain in domains])
        return np.asarray(rows, dtype=float)

    weights = np.stack([phase("phase_0_weights_json"), phase("phase_1_weights_json")], axis=1)
    if not np.allclose(weights.sum(axis=2), 1.0):
        raise ValueError("Heldout mixture weights are not normalized")
    return weights


def coordinate_mean_pool(pool: Pool) -> Pool:
    """Collapse repeat runs to one equally weighted policy observation."""
    rows: list[dict[str, Any]] = []
    weights: list[np.ndarray] = []
    for coordinate, indices in pool.frame.groupby("coordinate_hash", sort=True).indices.items():
        local = pool.frame.iloc[np.asarray(indices, dtype=int)]
        first = local.iloc[0]
        rows.append(
            {
                **first.to_dict(),
                "coordinate_hash": coordinate,
                "repeat_count": len(local),
                "training_series": ";".join(sorted(set(local["training_series"].astype(str)))),
                "uncheatable_bpb": float(local["uncheatable_bpb"].mean()),
                "table9_macro_bpb": float(local["table9_macro_bpb"].mean()),
            }
        )
        weights.append(pool.weights[int(indices[0])])
    return Pool(pd.DataFrame(rows).reset_index(drop=True), np.asarray(weights, dtype=float))


def load_sources() -> Sources:
    fit_uncheatable = observatory.load_delphi_3e18_fit_dataset("uncheatable")
    fit_table9 = observatory.load_delphi_3e18_fit_dataset("table9")
    if not np.allclose(fit_uncheatable.weights, fit_table9.weights):
        raise ValueError("The two target fit panels have different policy coordinates")
    domains = list(fit_uncheatable.domain_names)

    broad_frame = fit_uncheatable.frame.copy().reset_index(drop=True)
    broad_frame["row_id"] = broad_frame["run_name"].astype(str)
    broad_frame["source_pool"] = "broad_two_phase"
    broad_frame["coordinate_hash"] = [policy_hash(weights) for weights in fit_uncheatable.weights]
    broad_frame["training_series"] = "delphi_3e18_augmented_swarm_20260714"
    broad_frame["policy_class"] = "two_phase"
    broad_frame["uncheatable_bpb"] = fit_uncheatable.y
    broad_frame["table9_macro_bpb"] = fit_table9.y
    broad = Pool(broad_frame, np.asarray(fit_uncheatable.weights, dtype=float))

    heldout = pd.read_csv(HELDOUT_PATH)
    complete = heldout["training_state"].eq("finished") & heldout["checkpoint_declared_complete"].eq(1)
    heldout = heldout.loc[complete].reset_index(drop=True)
    heldout_weights = parse_weights(heldout, domains)
    heldout["coordinate_hash"] = heldout["mixture_sha256"].astype(str)
    heldout["row_id"] = heldout["wandb_run_name"].astype(str)

    single_mask = heldout["training_series"].eq(ONE_PHASE_SERIES)
    single_frame = heldout.loc[single_mask].copy().reset_index(drop=True)
    single_frame["source_pool"] = "single_phase"
    single = Pool(single_frame, heldout_weights[single_mask.to_numpy()])
    if len(single.frame) != 238 or single.frame["coordinate_hash"].nunique() != 238:
        raise ValueError("Expected 238 unique independently trained one-phase policies")

    fiber_mask = heldout["training_series"].eq(FIBER_SERIES)
    fiber_frame = heldout.loc[fiber_mask].copy().reset_index(drop=True)
    fiber_weights = heldout_weights[fiber_mask.to_numpy()]
    fiber_metadata = pd.read_csv(FIBER_RESULTS_PATH).rename(columns={"training_wandb_name": "wandb_run_name"})
    metadata_columns = [
        "wandb_run_name",
        "candidate_id",
        "anchor_id",
        "contrast_family",
        "direction_id",
        "direction_label",
        "sign",
        "seed_block",
        "uncheatable_delta_vs_same_seed_center",
        "table9_delta_vs_same_seed_center",
    ]
    fiber_frame = fiber_frame.merge(
        fiber_metadata[metadata_columns],
        on="wandb_run_name",
        how="left",
        validate="one_to_one",
    )
    if fiber_frame["anchor_id"].isna().any():
        raise ValueError("Fiber rows did not match the observed-results manifest")
    fiber_frame["source_pool"] = "frontier_fiber"
    fiber = Pool(fiber_frame, fiber_weights)
    if len(fiber.frame) != 200:
        raise ValueError(f"Expected 200 frontier-fiber runs, found {len(fiber.frame)}")

    common_mask = heldout["fit_panel_overlap"].eq("coordinate_disjoint") & ~heldout["training_series"].isin(
        SOURCE_SERIES
    )
    common_frame = heldout.loc[common_mask].copy().reset_index(drop=True)
    common_frame["source_pool"] = "common_archive"
    common = coordinate_mean_pool(Pool(common_frame, heldout_weights[common_mask.to_numpy()]))
    if len(common.frame) != 452:
        raise ValueError(f"Expected 452 common coordinate-disjoint policies, found {len(common.frame)}")

    return Sources(fit_uncheatable, broad, single, fiber, common)


def largest_remainder_quotas(counts: pd.Series, total: int) -> dict[Any, int]:
    if total > int(counts.sum()):
        raise ValueError(f"Cannot sample {total} rows from {int(counts.sum())}")
    ideal = total * counts / counts.sum()
    quotas = np.floor(ideal).astype(int)
    remaining = total - int(quotas.sum())
    order = (ideal - quotas).sort_values(ascending=False).index.tolist()
    for key in order[:remaining]:
        quotas.loc[key] += 1
    if (quotas > counts).any():
        raise ValueError("A stratified quota exceeds source capacity")
    return {key: int(value) for key, value in quotas.items()}


def stratified_indices(frame: pd.DataFrame, total: int, column: str, rng: np.random.Generator) -> np.ndarray:
    if total == 0:
        return np.asarray([], dtype=int)
    if total == len(frame):
        return np.arange(len(frame), dtype=int)
    quotas = largest_remainder_quotas(frame[column].value_counts(sort=False), total)
    selected: list[int] = []
    for key, quota in quotas.items():
        candidates = np.flatnonzero(frame[column].eq(key).to_numpy())
        selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def fiber_indices(
    frame: pd.DataFrame,
    total: int,
    mode: str,
    target: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample complete signed directions plus all available center controls."""
    if total == 0:
        return np.asarray([], dtype=int)
    candidate_mask = np.ones(len(frame), dtype=bool)
    if mode == "target_matched":
        candidate_mask = frame["anchor_id"].eq(TARGET_ANCHORS[target]).to_numpy()
    candidates = frame.loc[candidate_mask].copy()
    if total > len(candidates):
        raise ValueError(f"Cannot sample {total} {mode} fiber rows for {target} from {len(candidates)}")
    if total == len(candidates):
        return np.flatnonzero(candidate_mask)

    centers = candidates.loc[candidates["contrast_family"].eq("center_control")]
    center_indices = centers.index.to_numpy(dtype=int)
    remaining = total - len(center_indices)
    if remaining < 0 or remaining % 2:
        raise ValueError(f"Fiber allocation {total} cannot preserve {len(center_indices)} centers and signed pairs")

    tilted = candidates.loc[~candidates["contrast_family"].eq("center_control")].copy()
    tilted["direction_key"] = (
        tilted["anchor_id"].astype(str)
        + "::"
        + tilted["contrast_family"].astype(str)
        + "::"
        + tilted["direction_id"].astype(str)
    )
    direction_rows = (
        tilted.groupby("direction_key", sort=True)
        .agg(anchor_id=("anchor_id", "first"), contrast_family=("contrast_family", "first"), count=("sign", "size"))
        .reset_index()
    )
    if not direction_rows["count"].eq(2).all():
        raise ValueError("Every sampled frontier direction must have exactly two signs")
    direction_rows["stratum"] = direction_rows["anchor_id"] + "::" + direction_rows["contrast_family"]
    selected_direction_positions = stratified_indices(
        direction_rows,
        remaining // 2,
        "stratum",
        rng,
    )
    keys = set(direction_rows.iloc[selected_direction_positions]["direction_key"])
    tilt_indices = tilted.index[tilted["direction_key"].isin(keys)].to_numpy(dtype=int)
    selected = np.sort(np.concatenate([center_indices, tilt_indices]))
    if len(selected) != total:
        raise AssertionError(f"Expected {total} fiber rows, selected {len(selected)}")
    return selected


def allocation_seeds(allocation: Allocation, seeds: tuple[int, ...]) -> tuple[int, ...]:
    if allocation.broad == TOTAL_ROWS:
        return (seeds[0],)
    return seeds


def selected_rows(
    sources: Sources,
    allocation: Allocation,
    target: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed + 10_000 * TARGETS.index(target))
    broad_indices = stratified_indices(sources.broad.frame, allocation.broad, "panel_source", rng)
    single_indices = stratified_indices(sources.single.frame, allocation.single, "source_pool", rng)
    selected_fiber = fiber_indices(
        sources.fiber.frame,
        allocation.fiber,
        allocation.fiber_anchor_mode,
        target,
        rng,
    )
    frames = [
        sources.broad.frame.iloc[broad_indices],
        sources.single.frame.iloc[single_indices],
        sources.fiber.frame.iloc[selected_fiber],
    ]
    weights = [
        sources.broad.weights[broad_indices],
        sources.single.weights[single_indices],
        sources.fiber.weights[selected_fiber],
    ]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    combined = np.concatenate(weights, axis=0)
    if len(frame) != TOTAL_ROWS or len(combined) != TOTAL_ROWS:
        raise AssertionError(f"{allocation.name} did not produce {TOTAL_ROWS} rows")
    return frame, combined


def hpr_config(target: str) -> hierarchical.Config:
    metrics = pd.read_csv(HPR_METRICS_PATH)
    dataset_name = f"delphi_3e18_{target}"
    selected = metrics.loc[
        metrics["dataset"].eq(dataset_name) & metrics["variant"].eq(MODEL_VARIANT_NAME) & metrics["split"].eq("fit_oof")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one frozen HPR configuration for {dataset_name}, found {len(selected)}")
    row = selected.iloc[0]
    shape = family_grp.Shape(
        exponent=float(row["exponent"]),
        late_multiplier=float(row["late_multiplier"]),
        forgetting_rate=float(row["forgetting_rate"]),
        penalty_threshold=float(row["penalty_threshold"]),
        quality_discount=float(row.get("quality_discount", 1.0)),
    )
    return hierarchical.Config(
        variant=MODEL_VARIANT,
        shape_index=int(row["shape_index"]),
        shape=shape,
        l2=float(row["l2"]),
        residual_shrink=float(row["residual_shrink"]),
        undercoverage_fraction=0.0,
        coverage_gate_ratio=0.0,
    )


def custom_dataset(
    reference: pooled.Dataset,
    frame: pd.DataFrame,
    weights: np.ndarray,
    target: str,
    name: str,
) -> family_grp.Dataset:
    raw = pooled.Dataset(
        name=name,
        frame=frame.reset_index(drop=True),
        y=frame[TARGET_COLUMNS[target]].to_numpy(dtype=float),
        weights=np.asarray(weights, dtype=float),
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(reference.domain_names),
    )
    return hierarchical.family_dataset(raw)


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) < 1e-12 or np.std(predicted) < 1e-12:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    optimism = observed - predicted
    centered_prediction = predicted - predicted.mean()
    denominator = float(centered_prediction @ centered_prediction)
    slope = (
        float(centered_prediction @ (observed - observed.mean()) / denominator) if denominator > 1e-15 else float("nan")
    )
    tail_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    tail = np.argsort(predicted)[:tail_count]
    selected = int(np.argmin(predicted))
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": safe_spearman(observed, predicted),
        "bias": float(np.mean(residual)),
        "calibration_slope": slope,
        "regret_at_1": regret_at_k(observed, predicted, 1),
        "regret_at_3": regret_at_k(observed, predicted, 3),
        "regret_at_5": regret_at_k(observed, predicted, 5),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "optimism_gt_0p05": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
        "selected_optimism": float(optimism[selected]),
        "selected_observed": float(observed[selected]),
        "selected_predicted": float(predicted[selected]),
    }


def grouped_oof_prediction(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    groups: np.ndarray,
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(np.arange(dataset.n), groups=groups):
        model = hierarchical.fit_model(dataset, config, np.asarray(train, dtype=int))
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Grouped OOF prediction is incomplete")
    return prediction


def scope_masks(frame: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    adversarial = frame["training_series"].astype(str).str.contains(ADVERSARIAL_SERIES, regex=False).to_numpy()
    objective = frame["objective"].astype(str).to_numpy()
    policy = frame["policy_class"].astype(str).to_numpy()
    return {
        "common_all": np.ones(len(frame), dtype=bool),
        "common_two_phase": policy == "two_phase",
        "common_single_phase": policy == "single_phase_tied",
        "adversarial_target_matched": adversarial & (objective == target),
        "adversarial_cross_target": adversarial & np.isin(objective, list(TARGETS)) & (objective != target),
        "historical_non_adversarial": ~adversarial,
    }


def append_scoped_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    target: str,
) -> None:
    for scope, mask in scope_masks(frame, target).items():
        if np.sum(mask) < 3:
            continue
        rows.append({**base, "scope": scope, **prediction_metrics(observed[mask], predicted[mask])})


def source_holdout_pool(sources: Sources, selected_frame: pd.DataFrame) -> Pool:
    source_frame = pd.concat([sources.single.frame, sources.fiber.frame], ignore_index=True, sort=False)
    source_weights = np.concatenate([sources.single.weights, sources.fiber.weights], axis=0)
    selected_coordinates = set(selected_frame["coordinate_hash"].astype(str))
    keep = ~source_frame["coordinate_hash"].astype(str).isin(selected_coordinates)
    return coordinate_mean_pool(Pool(source_frame.loc[keep].reset_index(drop=True), source_weights[keep.to_numpy()]))


def fiber_delta_metrics(
    model: hierarchical.Model,
    sources: Sources,
    selected_frame: pd.DataFrame,
    target: str,
    allocation: Allocation,
    seed: int,
) -> list[dict[str, Any]]:
    frame = sources.fiber.frame.copy()
    prediction = model.predict(sources.fiber.weights)
    frame["predicted"] = prediction
    selected_coordinates = set(selected_frame["coordinate_hash"].astype(str))
    centers = frame.loc[frame["contrast_family"].eq("center_control")].groupby("anchor_id")["predicted"].mean()
    delta_column = (
        f"{target}_delta_vs_same_seed_center" if target == "uncheatable" else "table9_delta_vs_same_seed_center"
    )
    tilted = frame.loc[~frame["contrast_family"].eq("center_control")].copy()
    tilted = tilted.loc[~tilted["coordinate_hash"].astype(str).isin(selected_coordinates)].copy()
    if tilted.empty:
        return []
    tilted["predicted_delta"] = tilted["predicted"] - tilted["anchor_id"].map(centers)
    rows = []
    for scope, local in (
        ("unused_fiber_all", tilted),
        ("unused_fiber_target_matched", tilted.loc[tilted["anchor_id"].eq(TARGET_ANCHORS[target])]),
    ):
        if len(local) < 3:
            continue
        observed = local[delta_column].to_numpy(dtype=float)
        predicted = local["predicted_delta"].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "allocation": allocation.name,
                "seed": seed,
                "scope": scope,
                "n": len(local),
                "delta_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "delta_bias": float(np.mean(predicted - observed)),
                "delta_spearman": safe_spearman(observed, predicted),
                "delta_sign_accuracy": float(np.mean(np.signbit(observed) == np.signbit(predicted))),
            }
        )
    return rows


def optimize_model(
    model: hierarchical.Model,
    dataset: family_grp.Dataset,
    seed: int,
    starts: int,
) -> tuple[np.ndarray, float, bool]:
    initial = raw_optima.optimization_starts(dataset, "two_phase", seed, starts)
    return raw_optima.optimize(raw_optima.Fitted(MODEL_VARIANT_NAME, model), dataset, "two_phase", initial)


def optimum_record(
    model: hierarchical.Model,
    dataset: family_grp.Dataset,
    common: Pool,
    target: str,
    allocation: Allocation,
    seed: int,
    starts: int,
) -> dict[str, Any]:
    weights, prediction, converged = optimize_model(model, dataset, seed, starts)
    exposure = weights[0] * dataset.c0 + weights[1] * dataset.c1
    common_distances = 0.25 * np.abs(common.weights - weights[None, :, :]).sum(axis=(1, 2))
    nearest = int(np.argmin(common_distances))
    proportional = hierarchical.proportional_weights(dataset)
    proportional_weights = np.stack([proportional, proportional], axis=0)
    proportional_prediction = float(model.predict(proportional_weights[None, :, :])[0])
    return {
        "target": target,
        "allocation": allocation.name,
        "seed": seed,
        "predicted_bpb": prediction,
        "optimizer_converged": converged,
        "predicted_gain_vs_proportional": proportional_prediction - prediction,
        "max_bucket_weight": float(np.max(weights)),
        "max_simulated_epochs": float(np.max(exposure)),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "fit_support_distance": raw_optima.support_distance(dataset, weights),
        "nearest_common_tv": float(common_distances[nearest]),
        "nearest_common_observed": float(common.frame.iloc[nearest][TARGET_COLUMNS[target]]),
        "nearest_common_row": str(common.frame.iloc[nearest]["row_id"]),
        "phase_0_weights_json": json.dumps(
            dict(zip(dataset.domains, weights[0].tolist(), strict=True)), separators=(",", ":")
        ),
        "phase_1_weights_json": json.dumps(
            dict(zip(dataset.domains, weights[1].tolist(), strict=True)), separators=(",", ":")
        ),
    }


def aggregate_summary(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(keys, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, values, strict=True))
        row["replicates"] = len(group)
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce").dropna()
            if numeric.empty:
                continue
            row[f"{metric}_mean"] = float(numeric.mean())
            row[f"{metric}_p10"] = float(numeric.quantile(0.10))
            row[f"{metric}_p90"] = float(numeric.quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def optimum_stability(optima: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, allocation), group in optima.groupby(["target", "allocation"], sort=True):
        vectors = []
        for row in group.itertuples(index=False):
            p0 = np.asarray(list(json.loads(row.phase_0_weights_json).values()), dtype=float)
            p1 = np.asarray(list(json.loads(row.phase_1_weights_json).values()), dtype=float)
            vectors.append(np.stack([p0, p1], axis=0))
        pairwise = [
            float(0.25 * np.abs(vectors[left] - vectors[right]).sum())
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
        ]
        rows.append(
            {
                "target": target,
                "allocation": allocation,
                "replicates": len(group),
                "pairwise_optimum_tv_mean": float(np.mean(pairwise)) if pairwise else 0.0,
                "pairwise_optimum_tv_max": float(np.max(pairwise)) if pairwise else 0.0,
            }
        )
    return pd.DataFrame(rows)


def render(metric_runs: pd.DataFrame, optima: pd.DataFrame, output_dir: Path) -> None:
    common = metric_runs.loc[metric_runs["scope"].eq("common_all")].copy()
    order = [allocation.name for allocation in ALLOCATIONS]
    colors = {"uncheatable": "#d73027", "table9": "#1a9850"}
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Common archive RMSE", "Common archive Regret@1", "Calibration slope", "Worst optimism"),
        vertical_spacing=0.18,
    )
    for target in TARGETS:
        selected = common.loc[common["target"].eq(target)]
        for column, metric in enumerate(("rmse", "regret_at_1"), start=1):
            figure.add_trace(
                go.Box(
                    x=selected["allocation"],
                    y=selected[metric],
                    name=target,
                    legendgroup=target,
                    marker_color=colors[target],
                    boxpoints="all",
                    jitter=0.25,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        for column, metric in enumerate(("calibration_slope", "worst_optimism"), start=1):
            figure.add_trace(
                go.Box(
                    x=selected["allocation"],
                    y=selected[metric],
                    name=target,
                    legendgroup=target,
                    marker_color=colors[target],
                    boxpoints="all",
                    jitter=0.25,
                    showlegend=False,
                ),
                row=2,
                col=column,
            )
    for row in (1, 2):
        for column in (1, 2):
            figure.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-35, row=row, col=column)
    figure.add_hline(y=1.0, line_dash="dot", line_color="#666", row=2, col=1)
    figure.update_layout(
        title="Fixed 280-row evidence composition: common heldout diagnostics",
        template="plotly_white",
        width=1700,
        height=1050,
        legend={"orientation": "h", "y": 1.08},
        margin={"b": 190},
    )
    figure.write_html(output_dir / "common_heldout_composition_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    if optima.empty:
        return
    optimum_figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Predicted raw optimum", "Phase total variation", "Max simulated epochs", "Support distance"),
        vertical_spacing=0.18,
    )
    for target in TARGETS:
        selected = optima.loc[optima["target"].eq(target)]
        for position, metric in enumerate(
            ("predicted_bpb", "phase_total_variation", "max_simulated_epochs", "fit_support_distance")
        ):
            row, column = divmod(position, 2)
            optimum_figure.add_trace(
                go.Box(
                    x=selected["allocation"],
                    y=selected[metric],
                    name=target,
                    legendgroup=target,
                    marker_color=colors[target],
                    boxpoints="all",
                    jitter=0.25,
                    showlegend=position == 0,
                ),
                row=row + 1,
                col=column + 1,
            )
    for row in (1, 2):
        for column in (1, 2):
            optimum_figure.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-35, row=row, col=column)
    optimum_figure.update_layout(
        title="Raw-optimum plausibility and subsample stability",
        template="plotly_white",
        width=1700,
        height=1050,
        legend={"orientation": "h", "y": 1.08},
        margin={"b": 190},
    )
    optimum_figure.write_html(
        output_dir / "raw_optimum_composition_audit.html", include_plotlyjs="cdn", config=PLOT_CONFIG
    )


def write_report(
    metric_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    optimum_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    common = metric_summary.loc[metric_summary["scope"].eq("common_all")].copy()
    baseline = common.loc[common["allocation"].eq("b280_baseline")].set_index("target")
    comparison_rows = []
    for row in common.itertuples(index=False):
        base = baseline.loc[row.target]
        comparison_rows.append(
            {
                "target": row.target,
                "allocation": row.allocation,
                "replicates": row.replicates,
                "rmse": row.rmse_mean,
                "delta_rmse_vs_b280": row.rmse_mean - base.rmse_mean,
                "spearman": row.spearman_mean,
                "regret_at_1": row.regret_at_1_mean,
                "delta_regret_vs_b280": row.regret_at_1_mean - base.regret_at_1_mean,
                "calibration_slope": row.calibration_slope_mean,
                "optimism_gt_0p05": row.optimism_gt_0p05_mean,
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(["target", "rmse", "regret_at_1"])
    comparison.to_csv(output_dir / "common_archive_comparison.csv", index=False)
    best_lines = []
    for target in TARGETS:
        local = comparison.loc[comparison["target"].eq(target)]
        best_rmse = local.iloc[int(np.argmin(local["rmse"].to_numpy()))]
        best_regret = local.iloc[int(np.argmin(local["regret_at_1"].to_numpy()))]
        best_lines.append(
            f"- **{target}:** best RMSE `{best_rmse['allocation']}` ({best_rmse['rmse']:.5f}; "
            f"delta {best_rmse['delta_rmse_vs_b280']:+.5f}); best Regret@1 `{best_regret['allocation']}` "
            f"({best_regret['regret_at_1']:.5f}; delta {best_regret['delta_regret_vs_b280']:+.5f})."
        )
    lines = [
        "# Delphi 3e18 fixed-budget frontier-composition audit",
        "",
        "Every fit uses exactly 280 checkpoint observations. The HPR nonlinear shape and ridge settings are frozen "
        "to the original broad two-phase fit; only the evidence rows and fitted nonnegative coefficients change.",
        "",
        "## Primary common-archive result",
        "",
        *best_lines,
        "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The common archive excludes the original broad fit panel, all 238 independently trained one-phase rows, "
        "and all 200 frontier-fiber runs. It therefore stays identical for every allocation. Repeat checkpoints in "
        "that archive are averaged by policy coordinate before scoring.",
        "",
        "## Held-out phase-fiber deltas",
        "",
        (
            delta_summary.to_markdown(index=False, floatfmt=".6f")
            if not delta_summary.empty
            else "No unused fibers remain."
        ),
        "",
        "## Raw-optimum audit",
        "",
        (
            optimum_summary.to_markdown(index=False, floatfmt=".6f")
            if not optimum_summary.empty
            else "Raw optimization skipped."
        ),
        "",
        "Raw-optimum numbers are diagnostics, not validated performance. The observable decision test is post-selection "
        "regret on the fixed common archive; continuous optima are judged only by plausibility, support, and stability.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources()

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        config = hpr_config(target)
        common_target = sources.common.frame[TARGET_COLUMNS[target]].to_numpy(dtype=float)
        for allocation in ALLOCATIONS:
            for seed in allocation_seeds(allocation, seeds):
                print(f"Fitting {target}/{allocation.name}/seed={seed}", flush=True)
                frame, weights = selected_rows(sources, allocation, target, seed)
                dataset = custom_dataset(
                    sources.reference,
                    frame,
                    weights,
                    target,
                    f"delphi_3e18_{target}_{allocation.name}_{seed}",
                )
                groups = frame["coordinate_hash"].astype(str).to_numpy()
                oof = grouped_oof_prediction(dataset, config, groups)
                full_model = hierarchical.fit_model(dataset, config, np.arange(dataset.n))
                base = {
                    "target": target,
                    "allocation": allocation.name,
                    "broad_rows": allocation.broad,
                    "single_rows": allocation.single,
                    "fiber_rows": allocation.fiber,
                    "fiber_anchor_mode": allocation.fiber_anchor_mode,
                    "seed": seed,
                }
                metric_rows.append(
                    {
                        **base,
                        "scope": "train_grouped_oof",
                        **prediction_metrics(dataset.target, oof),
                    }
                )
                common_prediction = full_model.predict(sources.common.weights)
                append_scoped_metrics(
                    metric_rows,
                    base,
                    sources.common.frame,
                    common_target,
                    common_prediction,
                    target,
                )
                for index, (observed, predicted) in enumerate(zip(common_target, common_prediction, strict=True)):
                    prediction_rows.append(
                        {
                            **base,
                            "row_id": sources.common.frame.iloc[index]["row_id"],
                            "training_series": sources.common.frame.iloc[index]["training_series"],
                            "policy_class": sources.common.frame.iloc[index]["policy_class"],
                            "objective": sources.common.frame.iloc[index]["objective"],
                            "observed": observed,
                            "predicted": predicted,
                            "residual": predicted - observed,
                        }
                    )

                unused = source_holdout_pool(sources, frame)
                if len(unused.frame) >= 3:
                    unused_prediction = full_model.predict(unused.weights)
                    unused_observed = unused.frame[TARGET_COLUMNS[target]].to_numpy(dtype=float)
                    for source_pool in ("single_phase", "frontier_fiber"):
                        mask = unused.frame["source_pool"].eq(source_pool).to_numpy()
                        if np.sum(mask) >= 3:
                            metric_rows.append(
                                {
                                    **base,
                                    "scope": f"unused_{source_pool}",
                                    **prediction_metrics(unused_observed[mask], unused_prediction[mask]),
                                }
                            )
                delta_rows.extend(fiber_delta_metrics(full_model, sources, frame, target, allocation, seed))
                if not args.skip_optima:
                    optimum_rows.append(
                        optimum_record(
                            full_model,
                            dataset,
                            sources.common,
                            target,
                            allocation,
                            seed + 1_000 * TARGETS.index(target),
                            args.optimizer_starts,
                        )
                    )
                for row in frame.itertuples(index=False):
                    manifest_rows.append(
                        {
                            **base,
                            "row_id": row.row_id,
                            "source_pool": row.source_pool,
                            "coordinate_hash": row.coordinate_hash,
                        }
                    )

    metric_runs = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    deltas = pd.DataFrame(delta_rows)
    optima = pd.DataFrame(optimum_rows)
    manifest = pd.DataFrame(manifest_rows)
    metric_runs.to_csv(output_dir / "metric_runs.csv", index=False)
    predictions.to_csv(output_dir / "common_archive_predictions.csv", index=False)
    deltas.to_csv(output_dir / "fiber_delta_metric_runs.csv", index=False)
    optima.to_csv(output_dir / "raw_optima.csv", index=False)
    manifest.to_csv(output_dir / "training_manifest.csv", index=False)

    metric_names = [
        "rmse",
        "mae",
        "spearman",
        "bias",
        "calibration_slope",
        "regret_at_1",
        "regret_at_3",
        "regret_at_5",
        "lower_tail_optimism",
        "low_tail_rmse",
        "optimism_gt_0p05",
        "worst_optimism",
        "selected_optimism",
        "selected_observed",
        "selected_predicted",
    ]
    metric_summary = aggregate_summary(metric_runs, ["target", "allocation", "scope"], metric_names)
    metric_summary.to_csv(output_dir / "metric_summary.csv", index=False)
    delta_summary = (
        aggregate_summary(
            deltas,
            ["target", "allocation", "scope"],
            ["delta_rmse", "delta_bias", "delta_spearman", "delta_sign_accuracy"],
        )
        if not deltas.empty
        else pd.DataFrame()
    )
    delta_summary.to_csv(output_dir / "fiber_delta_metric_summary.csv", index=False)
    if not optima.empty:
        optimum_summary = aggregate_summary(
            optima,
            ["target", "allocation"],
            [
                "predicted_bpb",
                "predicted_gain_vs_proportional",
                "max_bucket_weight",
                "max_simulated_epochs",
                "phase_total_variation",
                "fit_support_distance",
                "nearest_common_tv",
                "nearest_common_observed",
            ],
        ).merge(optimum_stability(optima), on=["target", "allocation", "replicates"], how="left")
    else:
        optimum_summary = pd.DataFrame()
    optimum_summary.to_csv(output_dir / "raw_optimum_summary.csv", index=False)

    metadata = {
        "model": MODEL_VARIANT_NAME,
        "model_hyperparameters": "frozen from original broad 280-row HPR fit separately for each target",
        "total_training_rows": TOTAL_ROWS,
        "subset_seeds": list(seeds),
        "common_archive_runs_before_coordinate_collapse": 472,
        "common_archive_unique_policies": len(sources.common.frame),
        "source_pool_rows": {
            "broad_two_phase": len(sources.broad.frame),
            "single_phase": len(sources.single.frame),
            "frontier_fiber": len(sources.fiber.frame),
        },
        "allocations": [allocation.__dict__ for allocation in ALLOCATIONS],
        "data_use": {
            "frontier_fiber_status": "exposed development evidence; permitted as fit data in this audit",
            "primary_evaluation": "coordinate-disjoint archive excluding all three acquisition pools",
            "raw_optimum_status": "unvalidated diagnostic",
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    render(metric_runs, optima, output_dir)
    write_report(metric_summary, delta_summary, optimum_summary, output_dir)
    print(f"Wrote fixed-budget composition audit to {output_dir}")


if __name__ == "__main__":
    main()
