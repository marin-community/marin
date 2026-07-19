# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Test a fixed-budget tied-scaffold then phase-contrast swarm design.

The two-stage design spends ``n_tied`` checkpoints on phase-tied policies,
then spends ``n_contrast`` checkpoints on two-phase policies whose aggregate
mixtures have observed tied controls. The accounting constraint is
``n_tied + n_contrast = total_budget``.

This is a local design audit, not a new surrogate benchmark. It reuses the
staged aggregate/ordering models so that acquisition policy is the primary
variable. The dense StarCoder WSD surface supplies an independent two-domain
check of whether tied-first acquisition can find a phased frontier efficiently.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_staged_mechanistic_phase_zoo as staged,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "two_stage_phase_contrast_design_20260718"
DEFAULT_STARCODER_SOURCE = (
    SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_surface_refined_20260714" / "wsd80_observed_metrics.csv"
)
DEFAULT_ALLOCATIONS = "140:140,160:120,180:100,200:80,220:60"
DEFAULT_MODELS = "effective_benefit_tv,phase_separable"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"
FRONTIER_FRACTION = 0.20
FRONTIER_DESIGN_FRACTION = 0.70
PLOTLY_CONFIG = {"toImageButtonOptions": {"scale": 4}}


@dataclass(frozen=True)
class Allocation:
    """One fixed-budget split between tied and contrast checkpoints."""

    n_tied: int
    n_contrast: int

    @property
    def total(self) -> int:
        return self.n_tied + self.n_contrast


def parse_allocations(raw: str) -> list[Allocation]:
    """Parse comma-separated ``n_tied:n_contrast`` allocations."""
    allocations = []
    for item in raw.split(","):
        left, right = item.strip().split(":", maxsplit=1)
        allocation = Allocation(int(left), int(right))
        if allocation.n_contrast > allocation.n_tied:
            raise ValueError(f"Matched contrasts require n_contrast <= n_tied: {allocation}")
        allocations.append(allocation)
    totals = {allocation.total for allocation in allocations}
    if len(totals) != 1:
        raise ValueError(f"All allocations must have one fixed budget, got {sorted(totals)}")
    return allocations


def parse_ints(raw: str) -> list[int]:
    """Parse a comma-separated integer list."""
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def load_300m_dataset(objective: str, packet_path: Path, one_phase_path: Path) -> pooled.Dataset:
    """Load the exact matched 300M tied/two-phase panel."""
    frame = pd.read_csv(packet_path)
    domains = pooled.load_300m_dataset(objective).domain_names
    frame = joint.attach_single_phase_weights(frame, one_phase_path, domains)
    frame = frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy()
    return joint.dataset_from_frame(objective, frame, joint.TARGET_COLUMNS[objective])


def load_300m_external_dataset(objective: str, packet_path: Path, one_phase_path: Path) -> pooled.Dataset:
    """Load two-phase checkpoints excluded from the augmented fit panel."""
    frame = pd.read_csv(packet_path)
    domains = pooled.load_300m_dataset(objective).domain_names
    frame = joint.attach_single_phase_weights(frame, one_phase_path, domains)
    frame = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
    return joint.dataset_from_frame(objective, frame, joint.TARGET_COLUMNS[objective])


def aggregate_coordinates(dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    """Return square-root aggregate weights for Hellinger-like geometry."""
    aggregate = staged.aggregate_weights(dataset, indices)
    return np.sqrt(np.clip(aggregate, 0.0, None))


def farthest_point_subset(
    coordinates: np.ndarray,
    candidates: np.ndarray,
    size: int,
    *,
    forced: np.ndarray | None = None,
) -> np.ndarray:
    """Select a deterministic maximin subset from candidate row positions."""
    candidates = np.asarray(candidates, dtype=int)
    if size >= len(candidates):
        return np.sort(candidates)
    candidate_set = set(candidates.tolist())
    selected = []
    if forced is not None:
        selected.extend(int(index) for index in forced if int(index) in candidate_set)
    selected = list(dict.fromkeys(selected))[:size]
    if not selected:
        centroid = coordinates[candidates].mean(axis=0)
        selected.append(int(candidates[np.argmin(np.linalg.norm(coordinates[candidates] - centroid, axis=1))]))
    min_distance = np.linalg.norm(
        coordinates[candidates, None, :] - coordinates[np.asarray(selected)][None, :, :],
        axis=2,
    ).min(axis=1)
    while len(selected) < size:
        for position, candidate in enumerate(candidates):
            if int(candidate) in selected:
                min_distance[position] = -1.0
        next_position = int(np.argmax(min_distance))
        next_index = int(candidates[next_position])
        selected.append(next_index)
        distance = np.linalg.norm(coordinates[candidates] - coordinates[next_index], axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.sort(np.asarray(selected, dtype=int))


def baseline_pair_positions(dataset: pooled.Dataset, pairs: staged.PairIndices) -> np.ndarray:
    """Return pair positions for predeclared baseline controls."""
    keys = pd.Series(pairs.keys)
    baseline = keys.str.startswith("baseline_").to_numpy()
    return np.flatnonzero(baseline)


def select_tied_scaffold(
    dataset: pooled.Dataset,
    pairs: staged.PairIndices,
    allocation: Allocation,
    method: str,
    seed: int,
    candidate_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Select pair positions whose tied checkpoints form stage one."""
    positions = (
        np.arange(len(pairs.keys), dtype=int)
        if candidate_positions is None
        else np.asarray(candidate_positions, dtype=int)
    )
    if allocation.n_tied > len(positions):
        raise ValueError(f"Cannot select {allocation.n_tied} tied rows from {len(positions)} candidates")
    if method == "random":
        return np.sort(np.random.default_rng(seed).choice(positions, size=allocation.n_tied, replace=False))
    if method == "maximin":
        coordinates = aggregate_coordinates(dataset, pairs.single)
        return farthest_point_subset(
            coordinates,
            positions,
            allocation.n_tied,
            forced=baseline_pair_positions(dataset, pairs),
        )
    raise ValueError(f"Unknown tied-scaffold method: {method}")


def select_contrasts(
    dataset: pooled.Dataset,
    pairs: staged.PairIndices,
    scaffold_positions: np.ndarray,
    tied_prediction: np.ndarray,
    allocation: Allocation,
    method: str,
    seed: int,
) -> np.ndarray:
    """Select stage-two contrasts from aggregate-matched scaffold anchors."""
    rng = np.random.default_rng(seed + 10_000)
    if method == "random":
        return np.sort(rng.choice(scaffold_positions, size=allocation.n_contrast, replace=False))
    ordered = scaffold_positions[np.argsort(tied_prediction)]
    if method == "frontier":
        return np.sort(ordered[: allocation.n_contrast])
    if method != "frontier_diverse":
        raise ValueError(f"Unknown contrast method: {method}")

    coordinates = aggregate_coordinates(dataset, pairs.single)
    frontier_count = max(1, round(FRONTIER_DESIGN_FRACTION * allocation.n_contrast))
    pool_count = min(len(ordered), max(frontier_count, int(np.ceil(1.5 * allocation.n_contrast))))
    frontier_selected = farthest_point_subset(
        coordinates,
        ordered[:pool_count],
        frontier_count,
        forced=baseline_pair_positions(dataset, pairs),
    )
    remaining_count = allocation.n_contrast - len(frontier_selected)
    if remaining_count == 0:
        return frontier_selected
    remaining = np.asarray(
        sorted(set(scaffold_positions.tolist()).difference(frontier_selected.tolist())),
        dtype=int,
    )
    coverage_selected = farthest_point_subset(coordinates, remaining, remaining_count)
    return np.sort(np.concatenate([frontier_selected, coverage_selected]))


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return Spearman correlation or NaN for a degenerate subset."""
    if len(observed) < 3 or np.allclose(observed, observed[0]) or np.allclose(predicted, predicted[0]):
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Compute prediction and selection diagnostics for lower-is-better BPB."""
    residual = predicted - observed
    selected = int(np.argmin(predicted))
    return {
        "n_eval": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": safe_spearman(observed, predicted),
        "regret_at_1": float(observed[selected] - np.min(observed)),
        "selected_optimism": float(observed[selected] - predicted[selected]),
        "optimism_gt_0p05": int(np.sum(observed - predicted > 0.05)),
        "worst_optimism": float(np.max(observed - predicted)),
    }


def delta_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Compute phase-delta diagnostics."""
    residual = predicted - observed
    return {
        "delta_rmse": float(np.sqrt(np.mean(residual**2))),
        "delta_bias": float(np.mean(residual)),
        "delta_spearman": safe_spearman(observed, predicted),
        "delta_sign_accuracy": float(np.mean(np.signbit(observed) == np.signbit(predicted))),
    }


def selected_pair_indices(pairs: staged.PairIndices, positions: np.ndarray) -> staged.PairIndices:
    """Slice matched pair indices by pair position."""
    return staged.PairIndices(
        keys=pairs.keys[positions],
        single=pairs.single[positions],
        two=pairs.two[positions],
    )


def evaluate_design(
    dataset: pooled.Dataset,
    pairs: staged.PairIndices,
    allocation: Allocation,
    tied_method: str,
    contrast_method: str,
    model_config: staged.OrderConfig,
    seed: int,
    maxiter: int,
) -> list[dict[str, object]]:
    """Fit one two-stage design and score unseen matched contrasts."""
    scaffold_positions = select_tied_scaffold(dataset, pairs, allocation, tied_method, seed)
    scaffold_single = pairs.single[scaffold_positions]
    aggregate_model = staged.fit_aggregate_model(dataset, scaffold_single, maxiter, coarse_top_k=1)
    tied_prediction = staged.aggregate_prediction(aggregate_model, dataset, scaffold_single)
    contrast_positions = select_contrasts(
        dataset,
        pairs,
        scaffold_positions,
        tied_prediction,
        allocation,
        contrast_method,
        seed,
    )
    order_model = staged.fit_order_model(
        model_config,
        aggregate_model,
        dataset,
        selected_pair_indices(pairs, contrast_positions),
    )

    evaluation_positions = np.asarray(
        sorted(set(range(len(pairs.keys))).difference(contrast_positions.tolist())),
        dtype=int,
    )
    aggregate_evaluation_positions = np.asarray(
        sorted(set(range(len(pairs.keys))).difference(scaffold_positions.tolist())),
        dtype=int,
    )
    aggregate_evaluation_indices = pairs.single[aggregate_evaluation_positions]
    aggregate_evaluation = prediction_metrics(
        dataset.y[aggregate_evaluation_indices],
        staged.aggregate_prediction(aggregate_model, dataset, aggregate_evaluation_indices),
    )
    two_indices = pairs.two[evaluation_positions]
    single_indices = pairs.single[evaluation_positions]
    predicted_delta = staged.order_prediction(order_model, aggregate_model, dataset, two_indices)
    predicted_two = staged.aggregate_prediction(aggregate_model, dataset, two_indices) + predicted_delta
    observed_delta = dataset.y[two_indices] - dataset.y[single_indices]
    observed_two = dataset.y[two_indices]

    true_frontier_threshold = float(np.quantile(dataset.y[pairs.single], FRONTIER_FRACTION))
    frontier_mask = dataset.y[single_indices] <= true_frontier_threshold
    selected_frontier_fraction = float(np.mean(dataset.y[pairs.single[contrast_positions]] <= true_frontier_threshold))
    rows = []
    aggregate_param_count = 3 * dataset.m + 3
    phase_param_count = staged.nominal_parameter_count(dataset, model_config) - aggregate_param_count
    for scope, mask in (
        ("all_unseen", np.ones(len(evaluation_positions), dtype=bool)),
        ("tied_frontier_unseen", frontier_mask),
    ):
        if np.sum(mask) < 5:
            continue
        rows.append(
            {
                "dataset": dataset.name,
                "model": model_config.name,
                "seed": seed,
                "n_tied": allocation.n_tied,
                "n_contrast": allocation.n_contrast,
                "total_budget": allocation.total,
                "aggregate_param_count": aggregate_param_count,
                "phase_param_count": phase_param_count,
                "tied_rows_per_aggregate_param": allocation.n_tied / aggregate_param_count,
                "contrast_rows_per_phase_param": (
                    allocation.n_contrast / phase_param_count if phase_param_count else float("inf")
                ),
                "tied_method": tied_method,
                "contrast_method": contrast_method,
                "scope": scope,
                "selected_frontier_fraction": selected_frontier_fraction,
                **{f"aggregate_{key}": value for key, value in aggregate_evaluation.items()},
                **prediction_metrics(observed_two[mask], predicted_two[mask]),
                **delta_metrics(observed_delta[mask], predicted_delta[mask]),
            }
        )
    return rows


def external_fixed_budget_rows(
    dataset: pooled.Dataset,
    external: pooled.Dataset,
    pairs: staged.PairIndices,
    allocation: Allocation,
    seeds: list[int],
    maxiter: int,
) -> list[dict[str, object]]:
    """Compare one-stage and two-stage acquisition on common external rows."""
    config = next(config for config in staged.configs() if config.name == "phase_separable")
    joint_config = coverage.FitConfig("effective_exposure", False)
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    external_indices = np.arange(external.n, dtype=int)

    one_stage_model = coverage.fit_model(
        dataset,
        pairs.two,
        joint_config,
        linear_reg=coverage.dataset_linear_reg(dataset),
        maxiter=maxiter,
        coarse_top_k=1,
    )
    one_stage_prediction = coverage.predict(one_stage_model, external.weights, alpha0, alpha1)

    rows = []
    stage_prefix = f"two_stage_{allocation.n_tied}_{allocation.n_contrast}"
    for seed in seeds:
        scaffold_positions = select_tied_scaffold(dataset, pairs, allocation, "random", seed)
        scaffold_single = pairs.single[scaffold_positions]
        aggregate_model = staged.fit_aggregate_model(dataset, scaffold_single, maxiter, coarse_top_k=1)
        tied_prediction = staged.aggregate_prediction(aggregate_model, dataset, scaffold_single)
        contrast_positions = select_contrasts(
            dataset,
            pairs,
            scaffold_positions,
            tied_prediction,
            allocation,
            "frontier_diverse",
            seed,
        )
        contrast_two = pairs.two[contrast_positions]
        two_stage_indices = np.concatenate([scaffold_single, contrast_two])
        if len(two_stage_indices) != allocation.total:
            raise AssertionError(f"Expected {allocation.total} rows, got {len(two_stage_indices)}")

        joint_model = coverage.fit_model(
            dataset,
            two_stage_indices,
            joint_config,
            linear_reg=coverage.dataset_linear_reg(dataset),
            maxiter=maxiter,
            coarse_top_k=1,
        )
        joint_prediction = coverage.predict(joint_model, external.weights, alpha0, alpha1)

        order_model = staged.fit_order_model(
            config,
            aggregate_model,
            dataset,
            selected_pair_indices(pairs, contrast_positions),
        )
        staged_prediction = staged.aggregate_prediction(aggregate_model, external, external_indices)
        staged_prediction += staged.order_prediction(order_model, aggregate_model, external, external_indices)

        for method, prediction in (
            ("one_stage_280_two_phase", one_stage_prediction),
            (f"{stage_prefix}_joint", joint_prediction),
            (f"{stage_prefix}_decomposed", staged_prediction),
        ):
            rows.append(
                {
                    "dataset": dataset.name,
                    "method": method,
                    "seed": seed,
                    "n_tied": 0 if method == "one_stage_280_two_phase" else allocation.n_tied,
                    "n_contrast": allocation.total if method == "one_stage_280_two_phase" else allocation.n_contrast,
                    "total_budget": allocation.total,
                    "external_rows": external.n,
                    **prediction_metrics(external.y, prediction),
                }
            )
    return rows


def summarize_external_fixed_budget(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-budget external comparisons across subset seeds."""
    keys = ["dataset", "method", "n_tied", "n_contrast", "total_budget", "external_rows"]
    metrics = [
        "rmse",
        "bias",
        "spearman",
        "regret_at_1",
        "selected_optimism",
        "optimism_gt_0p05",
        "worst_optimism",
    ]
    rows = []
    for group_key, frame in raw.groupby(keys, sort=True):
        row = dict(zip(keys, group_key, strict=True))
        row["replicates"] = len(frame)
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_p10"] = float(frame[metric].quantile(0.10))
            row[f"{metric}_p90"] = float(frame[metric].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def cross_validated_acquisition_rows(
    dataset: pooled.Dataset,
    pairs: staged.PairIndices,
    target_allocation: Allocation,
    seeds: list[int],
    maxiter: int,
    n_splits: int = 5,
) -> list[dict[str, object]]:
    """Compare acquisition strategies on common unseen matched pairs."""
    order_config = next(config for config in staged.configs() if config.name == "phase_separable")
    joint_config = coverage.FitConfig("effective_exposure", False)
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    positions = np.arange(len(pairs.keys), dtype=int)
    rows = []
    for seed in seeds:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_positions, test_positions) in enumerate(splitter.split(positions)):
            train_positions = positions[train_positions]
            test_positions = positions[test_positions]
            n_tied = round(target_allocation.n_tied / target_allocation.total * len(train_positions))
            fold_allocation = Allocation(n_tied, len(train_positions) - n_tied)

            one_stage_model = coverage.fit_model(
                dataset,
                pairs.two[train_positions],
                joint_config,
                linear_reg=coverage.dataset_linear_reg(dataset),
                maxiter=maxiter,
                coarse_top_k=1,
            )
            one_stage_prediction = coverage.predict(
                one_stage_model,
                dataset.weights[pairs.two[test_positions]],
                alpha0,
                alpha1,
            )

            scaffold_positions = select_tied_scaffold(
                dataset,
                pairs,
                fold_allocation,
                "random",
                seed * n_splits + fold,
                candidate_positions=train_positions,
            )
            scaffold_single = pairs.single[scaffold_positions]
            aggregate_model = staged.fit_aggregate_model(dataset, scaffold_single, maxiter, coarse_top_k=1)
            tied_prediction = staged.aggregate_prediction(aggregate_model, dataset, scaffold_single)
            contrast_positions = select_contrasts(
                dataset,
                pairs,
                scaffold_positions,
                tied_prediction,
                fold_allocation,
                "frontier_diverse",
                seed * n_splits + fold,
            )
            two_stage_indices = np.concatenate([scaffold_single, pairs.two[contrast_positions]])
            joint_model = coverage.fit_model(
                dataset,
                two_stage_indices,
                joint_config,
                linear_reg=coverage.dataset_linear_reg(dataset),
                maxiter=maxiter,
                coarse_top_k=1,
            )
            joint_prediction = coverage.predict(
                joint_model,
                dataset.weights[pairs.two[test_positions]],
                alpha0,
                alpha1,
            )
            order_model = staged.fit_order_model(
                order_config,
                aggregate_model,
                dataset,
                selected_pair_indices(pairs, contrast_positions),
            )
            decomposed_prediction = staged.aggregate_prediction(aggregate_model, dataset, pairs.two[test_positions])
            decomposed_prediction += staged.order_prediction(
                order_model,
                aggregate_model,
                dataset,
                pairs.two[test_positions],
            )

            observed = dataset.y[pairs.two[test_positions]]
            for method, prediction in (
                ("one_stage_two_phase", one_stage_prediction),
                ("two_stage_joint", joint_prediction),
                ("two_stage_decomposed", decomposed_prediction),
            ):
                rows.append(
                    {
                        "dataset": dataset.name,
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "n_train": len(train_positions),
                        "n_test": len(test_positions),
                        "n_tied": 0 if method == "one_stage_two_phase" else fold_allocation.n_tied,
                        "n_contrast": (
                            len(train_positions) if method == "one_stage_two_phase" else fold_allocation.n_contrast
                        ),
                        **prediction_metrics(observed, prediction),
                    }
                )
    return rows


def summarize_cross_validated_acquisition(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched-pair acquisition CV."""
    keys = ["dataset", "method", "n_train", "n_test", "n_tied", "n_contrast"]
    metrics = [
        "rmse",
        "bias",
        "spearman",
        "regret_at_1",
        "selected_optimism",
        "optimism_gt_0p05",
        "worst_optimism",
    ]
    rows = []
    for group_key, frame in raw.groupby(keys, sort=True):
        row = dict(zip(keys, group_key, strict=True))
        row["replicates"] = len(frame)
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_p10"] = float(frame[metric].quantile(0.10))
            row[f"{metric}_p90"] = float(frame[metric].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_design(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize bootstrap runs with central 80% intervals."""
    keys = [
        "dataset",
        "model",
        "n_tied",
        "n_contrast",
        "total_budget",
        "aggregate_param_count",
        "phase_param_count",
        "tied_rows_per_aggregate_param",
        "contrast_rows_per_phase_param",
        "tied_method",
        "contrast_method",
        "scope",
    ]
    metrics = [
        "selected_frontier_fraction",
        "aggregate_rmse",
        "aggregate_bias",
        "aggregate_spearman",
        "aggregate_regret_at_1",
        "aggregate_selected_optimism",
        "aggregate_optimism_gt_0p05",
        "aggregate_worst_optimism",
        "rmse",
        "bias",
        "spearman",
        "regret_at_1",
        "selected_optimism",
        "optimism_gt_0p05",
        "worst_optimism",
        "delta_rmse",
        "delta_bias",
        "delta_spearman",
        "delta_sign_accuracy",
    ]
    rows = []
    for group_key, frame in raw.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, group_key, strict=True))
        row["replicates"] = len(frame)
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_p10"] = float(frame[metric].quantile(0.10))
            row[f"{metric}_p90"] = float(frame[metric].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def starcoder_tied_indices(frame: pd.DataFrame) -> np.ndarray:
    """Return phase-tied StarCoder coordinate indices."""
    return np.flatnonzero(
        np.isclose(
            frame["phase_0_starcoder"].to_numpy(dtype=float),
            frame["phase_1_starcoder"].to_numpy(dtype=float),
            atol=1e-10,
        )
    )


def starcoder_stage_two_indices(
    frame: pd.DataFrame,
    tied_train: np.ndarray,
    n_contrast: int,
) -> np.ndarray:
    """Select aggregate-local, contrast-diverse StarCoder coordinates."""
    tied = frame.iloc[tied_train].sort_values("phase_0_starcoder")
    x = tied["phase_0_starcoder"].to_numpy(dtype=float)
    y = tied["wsd80_bpb"].to_numpy(dtype=float)
    if len(x) >= 4:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        grid = np.linspace(x.min(), x.max(), 1001)
        prediction = interpolator(grid)
        finite = np.isfinite(prediction)
        candidate_anchors = grid[finite][np.argsort(prediction[finite])[: max(3, n_contrast // 4)]]
    else:
        candidate_anchors = x[np.argsort(y)[: min(3, len(x))]]
    off_tied = np.flatnonzero(~np.isin(np.arange(len(frame)), starcoder_tied_indices(frame)))
    aggregate = frame["aggregate_starcoder_share_80_20"].to_numpy(dtype=float)
    contrast = frame["ordering_contrast_p1_minus_p0"].to_numpy(dtype=float)
    aggregate_distance = np.min(np.abs(aggregate[off_tied, None] - candidate_anchors[None, :]), axis=1)
    pool_size = min(len(off_tied), max(n_contrast, 4 * n_contrast))
    pool = off_tied[np.argsort(aggregate_distance)[:pool_size]]
    scale_a = max(float(np.std(aggregate[pool])), 1e-3)
    scale_d = max(float(np.std(contrast[pool])), 1e-3)
    coordinates = np.column_stack([aggregate / scale_a, contrast / scale_d])
    return farthest_point_subset(coordinates, pool, n_contrast)


def starcoder_acquisition_audit(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Compare two-stage and random acquisition on the dense WSD surface."""
    tied_indices = starcoder_tied_indices(frame)
    if len(tied_indices) < 8:
        raise ValueError(f"Expected a dense tied slice, found {len(tied_indices)} rows")
    global_best = float(frame["wsd80_bpb"].min())
    rows = []
    for total_budget in (16, 24, 32):
        n_tied = int(np.ceil(0.625 * total_budget))
        n_contrast = total_budget - n_tied
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tied_train = np.sort(rng.choice(tied_indices, size=min(n_tied, len(tied_indices)), replace=False))
            contrast_train = starcoder_stage_two_indices(frame, tied_train, n_contrast)
            staged_indices = np.unique(np.concatenate([tied_train, contrast_train]))
            random_indices = np.sort(rng.choice(len(frame), size=total_budget, replace=False))
            for method, indices in (("two_stage", staged_indices), ("random_two_phase", random_indices)):
                best_observed = float(frame.iloc[indices]["wsd80_bpb"].min())
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "total_budget": total_budget,
                        "n_tied": n_tied if method == "two_stage" else 0,
                        "n_contrast": n_contrast if method == "two_stage" else total_budget,
                        "best_observed": best_observed,
                        "best_observed_regret": best_observed - global_best,
                        "within_0p01": best_observed - global_best <= 0.01,
                    }
                )
    return pd.DataFrame(rows)


def starcoder_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize the StarCoder acquisition audit."""
    return (
        raw.groupby(["method", "total_budget", "n_tied", "n_contrast"], as_index=False)
        .agg(
            replicates=("seed", "size"),
            mean_regret=("best_observed_regret", "mean"),
            median_regret=("best_observed_regret", "median"),
            p90_regret=("best_observed_regret", lambda values: values.quantile(0.90)),
            within_0p01_rate=("within_0p01", "mean"),
        )
        .sort_values(["total_budget", "method"])
    )


def write_plots(summary: pd.DataFrame, starcoder: pd.DataFrame, output_dir: Path) -> None:
    """Write compact interactive decision plots."""
    frontier = summary.loc[summary["scope"].eq("tied_frontier_unseen") & summary["tied_method"].eq("random")].copy()
    frontier["design"] = frontier["contrast_method"] + " / " + frontier["model"]
    figure = px.line(
        frontier,
        x="n_contrast",
        y="delta_rmse_mean",
        color="design",
        facet_col="dataset",
        markers=True,
        title="Fixed-budget phase-delta error near the tied frontier",
        labels={"n_contrast": "Stage-two contrast checkpoints", "delta_rmse_mean": "Heldout phase-delta RMSE"},
    )
    figure.update_layout(template="plotly_white")
    figure.write_html(
        output_dir / "frontier_delta_rmse_by_allocation.html",
        include_plotlyjs="cdn",
        config=PLOTLY_CONFIG,
    )

    starcoder_figure = px.line(
        starcoder,
        x="total_budget",
        y="mean_regret",
        color="method",
        markers=True,
        title="Dense StarCoder WSD: best observed regret under fixed acquisition budgets",
        labels={"total_budget": "Total checkpoints", "mean_regret": "Mean best-observed BPB regret"},
    )
    starcoder_figure.update_layout(template="plotly_white")
    starcoder_figure.write_html(
        output_dir / "starcoder_acquisition_regret.html",
        include_plotlyjs="cdn",
        config=PLOTLY_CONFIG,
    )


def best_design_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best allocation by frontier phase-delta RMSE per objective/model."""
    eligible = summary.loc[summary["scope"].eq("tied_frontier_unseen") & summary["tied_method"].eq("random")].copy()
    indices = eligible.groupby(["dataset", "model"])["delta_rmse_mean"].idxmin()
    columns = [
        "dataset",
        "model",
        "n_tied",
        "n_contrast",
        "tied_rows_per_aggregate_param",
        "contrast_rows_per_phase_param",
        "contrast_method",
        "aggregate_rmse_mean",
        "aggregate_spearman_mean",
        "delta_rmse_mean",
        "delta_spearman_mean",
        "regret_at_1_mean",
        "selected_frontier_fraction_mean",
    ]
    return eligible.loc[indices, columns].sort_values(["dataset", "model"])


def write_report(
    summary: pd.DataFrame,
    starcoder: pd.DataFrame,
    external: pd.DataFrame,
    acquisition_cv: pd.DataFrame,
    allocations: list[Allocation],
    output_dir: Path,
) -> None:
    """Write a concise interpretation and a concrete two-stage design."""
    best = best_design_rows(summary)
    starcoder_table = starcoder[["method", "total_budget", "mean_regret", "p90_regret", "within_0p01_rate"]]
    lines = [
        "# Two-stage tied-spine / phase-contrast design audit",
        "",
        "## Frozen design question",
        "",
        "Can a fixed budget be used more efficiently by first learning the phase-tied aggregate surface, then "
        "collecting one non-adaptive batch of aggregate-matched phase contrasts? Every tested allocation obeys "
        "`n_tied + n_contrast = 280`; no third acquisition round is allowed.",
        "",
        "## Verdict",
        "",
        "**The current staged design fails the local fixed-budget acceptance gate and should not replace the "
        "one-stage 280-row two-phase swarm.** In matched-pair CV, the staged predictors do not improve RMSE or "
        "Regret@1 consistently on either target, and the independent StarCoder acquisition audit favors random "
        "two-phase sampling. A future Stage-2 panel would therefore be an explicitly exploratory causal DOE, not "
        "a validated sample-efficiency improvement.",
        "",
        "## Local 300M result",
        "",
        best.to_markdown(index=False, floatfmt=".6f"),
        "",
        "These rows choose allocations only by heldout phase-delta RMSE near the independently defined tied "
        "frontier. They are a local design diagnostic, not permission to tune against future 3e18 confirmatory "
        "checkpoints.",
        "",
        "## Apples-to-apples matched-pair CV",
        "",
        acquisition_cv[
            [
                "dataset",
                "method",
                "n_train",
                "n_tied",
                "n_contrast",
                "rmse_mean",
                "bias_mean",
                "spearman_mean",
                "regret_at_1_mean",
                "optimism_gt_0p05_mean",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "Each fold trains on the same 224 correspondence groups and scores the same 56 unseen two-phase "
        "schedules. The staged split scales 180/100 to 144 tied plus 80 contrasts. This is the direct local "
        "fixed-budget comparison; it does not reuse phase-tied diagnostic rows as phase-order evidence. The "
        "decomposed staged fit is clearly worse in RMSE and rank, while the joint staged fit is roughly tied in "
        "Table-9 RMSE but worse in Regret@1 and is worse on all primary Uncheatable diagnostics.",
        "",
        "## Aggregate-only external heldout check",
        "",
        external[
            [
                "dataset",
                "method",
                "n_tied",
                "n_contrast",
                "external_rows",
                "rmse_mean",
                "bias_mean",
                "spearman_mean",
                "regret_at_1_mean",
                "optimism_gt_0p05_mean",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "These packet rows are called two-phase by source, but every one is phase-tied (`phase TV = 0`). This "
        "table therefore checks aggregate-surface transfer only. It must not be interpreted as evidence that "
        "one acquisition strategy identifies phase-order effects better.",
        "",
        "## Independent two-domain shape check",
        "",
        starcoder_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "At every tested budget, random two-phase acquisition has lower best-observed regret than the staged "
        "procedure. This does not disprove targeted contrasts in the 39-bucket setting, but it blocks a generic "
        "sample-efficiency claim.",
        "",
        "## Revised protocol if the failed gate is deliberately reopened",
        "",
        "1. Stage 1 uses 180 score-blind tied rows selected by maximin Hellinger distance, with proportional, "
        "UniMax-8, and stratified baselines forced in. The remaining 100 completed tied rows are untouched "
        "evaluation data. Bootstrap only within the selected 180 rows.",
        "2. Freeze 25 bootstrap-stable anchors: 15 selected for global mixture-space coverage and 10 from the "
        "predicted tied frontier. Do not select anchors from observed heldout scores.",
        "3. Stage 2 contains 100 exact aggregate-matched contrasts: two tangent-simplex directions per anchor, "
        "both signs, at one preregistered radius. For phase fractions 0.8/0.2, parameterize "
        "`w0 = a - 0.2 d` and `w1 = a + 0.8 d`, with `sum(d) = 0`. Multiple contrasts may share one tied "
        "control; only the number of distinct anchors is bounded by the Stage-1 rows.",
        "4. Seed-match each contrast to its tied control. Fit `Y(a,d) = f(a) + g(a,d)` with `g(a,0)=0`, and also "
        "report a jointly fitted predictor. Report aggregate and matched-delta diagnostics separately.",
        "5. Stage 2 is the only adaptive acquisition. Refitting after Stage 2 is analysis, not a third acquisition "
        "stage; no further checkpoint tranche is allowed.",
        "",
        "## Launch gate",
        "",
        "Do not materialize Stage 2 unless all gates pass after the one-phase 3e18 export completes:",
        "",
        "- **G1 tied transfer:** the 180-row tied fit predicts the untouched 100 tied rows with RMSE at most 0.007 "
        "on Uncheatable and 0.016 on Table-9, and Spearman at least 0.90 and 0.85 respectively.",
        "- **G2 anchor stability:** bootstrap anchor-set Jaccard is at least 0.60.",
        "- **G3 construction:** every contrast is simplex-feasible and exactly aggregate-matched without clipping.",
        "- **G4 acquisition:** the staged design beats the one-stage random two-phase control in matched-pair local "
        "CV. **G4 currently fails.**",
        "",
        "The deterministic Stage-1 subset, anchor rule, contrast directions, radius, and total row count must be "
        "frozen before launch. Existing two-phase swarms and 3e18 validation checkpoints remain evaluation "
        "evidence, not acquisition inputs.",
        "",
        "## Tested allocations",
        "",
        ", ".join(f"{allocation.n_tied}/{allocation.n_contrast}" for allocation in allocations),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--starcoder-source", type=Path, default=DEFAULT_STARCODER_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--allocations", default=DEFAULT_ALLOCATIONS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--maxiter", type=int, default=0)
    parser.add_argument("--external-maxiter", type=int, default=16)
    parser.add_argument("--acquisition-cv-seeds", default="0,1,2")
    parser.add_argument("--acquisition-cv-maxiter", type=int, default=12)
    parser.add_argument("--acquisition-cv-only", action="store_true")
    args = parser.parse_args()

    allocations = parse_allocations(args.allocations)
    seeds = parse_ints(args.seeds)
    acquisition_cv_seeds = parse_ints(args.acquisition_cv_seeds)
    config_by_name = {config.name: config for config in staged.configs()}
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(model_names).difference(config_by_name))
    if unknown:
        raise ValueError(f"Unknown staged models: {unknown}")
    model_configs = [config_by_name[name] for name in model_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.acquisition_cv_only:
        acquisition_cv_rows = []
        for objective in [item.strip() for item in args.objectives.split(",") if item.strip()]:
            dataset = load_300m_dataset(objective, args.packet, args.one_phase_source)
            acquisition_cv_rows.extend(
                cross_validated_acquisition_rows(
                    dataset,
                    staged.pair_indices(dataset),
                    Allocation(180, 100),
                    acquisition_cv_seeds,
                    args.acquisition_cv_maxiter,
                )
            )
        acquisition_cv_raw = pd.DataFrame(acquisition_cv_rows)
        acquisition_cv = summarize_cross_validated_acquisition(acquisition_cv_raw)
        acquisition_cv_raw.to_csv(args.output_dir / "matched_pair_acquisition_cv_runs.csv", index=False)
        acquisition_cv.to_csv(args.output_dir / "matched_pair_acquisition_cv_summary.csv", index=False)
        summary = pd.read_csv(args.output_dir / "allocation_summary.csv")
        starcoder = pd.read_csv(args.output_dir / "starcoder_acquisition_summary.csv")
        external = pd.read_csv(args.output_dir / "fixed_budget_external_summary.csv")
        write_report(summary, starcoder, external, acquisition_cv, allocations, args.output_dir)
        print(acquisition_cv.to_string(index=False))
        return

    rows = []
    external_rows = []
    acquisition_cv_rows = []
    comparison_allocations = (Allocation(160, 120), Allocation(180, 100), Allocation(220, 60))
    for objective in [item.strip() for item in args.objectives.split(",") if item.strip()]:
        dataset = load_300m_dataset(objective, args.packet, args.one_phase_source)
        external_dataset = load_300m_external_dataset(objective, args.packet, args.one_phase_source)
        pairs = staged.pair_indices(dataset)
        if len(pairs.keys) < max(allocation.n_tied for allocation in allocations):
            raise ValueError(f"{dataset.name} has only {len(pairs.keys)} matched pairs")
        for allocation in allocations:
            for seed in seeds:
                for tied_method in ("random", "maximin"):
                    if tied_method == "maximin" and seed != seeds[0]:
                        continue
                    for contrast_method in ("random", "frontier", "frontier_diverse"):
                        for config in model_configs:
                            print(
                                f"{dataset.name} {allocation.n_tied}/{allocation.n_contrast} "
                                f"seed={seed} {tied_method}/{contrast_method}/{config.name}",
                                flush=True,
                            )
                            rows.extend(
                                evaluate_design(
                                    dataset,
                                    pairs,
                                    allocation,
                                    tied_method,
                                    contrast_method,
                                    config,
                                    seed,
                                    args.maxiter,
                                )
                            )
        for comparison_allocation in comparison_allocations:
            external_rows.extend(
                external_fixed_budget_rows(
                    dataset,
                    external_dataset,
                    pairs,
                    comparison_allocation,
                    seeds,
                    args.external_maxiter,
                )
            )
        acquisition_cv_rows.extend(
            cross_validated_acquisition_rows(
                dataset,
                pairs,
                Allocation(180, 100),
                acquisition_cv_seeds,
                args.acquisition_cv_maxiter,
            )
        )

    raw = pd.DataFrame(rows)
    summary = summarize_design(raw)
    external_raw = pd.DataFrame(external_rows).drop_duplicates(["dataset", "method", "seed"])
    external = summarize_external_fixed_budget(external_raw)
    acquisition_cv_raw = pd.DataFrame(acquisition_cv_rows)
    acquisition_cv = summarize_cross_validated_acquisition(acquisition_cv_raw)
    starcoder_raw = starcoder_acquisition_audit(pd.read_csv(args.starcoder_source), seeds)
    starcoder = starcoder_summary(starcoder_raw)
    raw.to_csv(args.output_dir / "bootstrap_design_runs.csv", index=False)
    summary.to_csv(args.output_dir / "allocation_summary.csv", index=False)
    best_design_rows(summary).to_csv(args.output_dir / "best_local_allocations.csv", index=False)
    external_raw.to_csv(args.output_dir / "fixed_budget_external_runs.csv", index=False)
    external.to_csv(args.output_dir / "fixed_budget_external_summary.csv", index=False)
    acquisition_cv_raw.to_csv(args.output_dir / "matched_pair_acquisition_cv_runs.csv", index=False)
    acquisition_cv.to_csv(args.output_dir / "matched_pair_acquisition_cv_summary.csv", index=False)
    starcoder_raw.to_csv(args.output_dir / "starcoder_acquisition_runs.csv", index=False)
    starcoder.to_csv(args.output_dir / "starcoder_acquisition_summary.csv", index=False)
    metadata = {
        "allocations": [asdict(allocation) for allocation in allocations],
        "seeds": seeds,
        "models": model_names,
        "objectives": [item.strip() for item in args.objectives.split(",") if item.strip()],
        "maxiter": args.maxiter,
        "external_maxiter": args.external_maxiter,
        "acquisition_cv_seeds": acquisition_cv_seeds,
        "acquisition_cv_maxiter": args.acquisition_cv_maxiter,
        "frontier_fraction": FRONTIER_FRACTION,
        "frontier_design_fraction": FRONTIER_DESIGN_FRACTION,
        "packet": str(args.packet),
        "one_phase_source": str(args.one_phase_source),
        "starcoder_source": str(args.starcoder_source),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_plots(summary, starcoder, args.output_dir)
    write_report(summary, starcoder, external, acquisition_cv, allocations, args.output_dir)
    print(best_design_rows(summary).to_string(index=False))
    print(starcoder.to_string(index=False))
    print(f"Wrote two-stage design audit to {args.output_dir}")


if __name__ == "__main__":
    main()
