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
"""Fit one aggregate spine and one conservative phase potential to heterogeneous designs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_heterogeneous_design_aware_hpr_20260719 as heterogeneous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_pair,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_plasticity_potential_transport_20260720 as first_order,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/finite_potential_joint_surrogate_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
TAU_GRID = (0.1, 0.3, 1.0, 3.0)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
OUTER_FOLDS = 4
INNER_FOLDS = 3
PAIR_TOLERANCE = 1.01
FIBER_IMPROVEMENT = 0.05
STABILITY_COSINE = 0.5
STABILITY_SIGN = 0.6
REGRET_TOLERANCE = 0.002
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Geometry(StrEnum):
    LINEAR = "linear"
    FINITE = "finite"


@dataclass(frozen=True)
class Candidate:
    name: str
    geometry: Geometry
    use_fiber: bool
    positive_semidefinite: bool = False


CANDIDATES = (
    Candidate("linear_pair_only", Geometry.LINEAR, False),
    Candidate("linear_joint", Geometry.LINEAR, True),
    Candidate("finite_pair_only", Geometry.FINITE, False),
    Candidate("finite_joint", Geometry.FINITE, True),
    Candidate("finite_psd_joint", Geometry.FINITE, True, positive_semidefinite=True),
)


@dataclass(frozen=True)
class PhaseData:
    pair_dataset: family_grp.Dataset
    pair_target: np.ndarray
    pair_groups: np.ndarray
    fiber_dataset: family_grp.Dataset
    fiber_frame: pd.DataFrame
    fiber_target: np.ndarray
    fiber_groups: np.ndarray
    fiber_blocks: np.ndarray


@dataclass(frozen=True)
class PhaseFit:
    candidate: Candidate
    tau: float
    ridge: float
    coefficients: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class FullModel:
    aggregate: hierarchical.Model
    phase: PhaseFit
    phase_dataset: family_grp.Dataset

    def predict_phase(self, weights: np.ndarray) -> np.ndarray:
        design, names = phase_design(weights, self.phase_dataset, self.phase.candidate.geometry, self.phase.tau)
        if names != self.phase.feature_names:
            raise ValueError("Phase feature order changed between fit and prediction")
        return design @ self.phase.coefficients

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.aggregate.predict(first_order.tied_policy(weights, self.phase_dataset)) + self.predict_phase(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-only", action="store_true")
    return parser.parse_args()


def finite_potential_design(
    weights: np.ndarray,
    dataset: family_grp.Dataset,
    tau: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    state, displacement, variation = first_order.phase_coordinates(weights, dataset, tau)
    family_count = len(dataset.family_names)
    pieces = [displacement]
    names = [f"potential_gradient:{name}" for name in dataset.family_names]

    curvature = []
    for left, right in itertools.combinations_with_replacement(range(family_count), 2):
        if left == right:
            feature = state[:, left] * displacement[:, left] + 0.5 * displacement[:, left] ** 2
        else:
            feature = (
                state[:, left] * displacement[:, right]
                + state[:, right] * displacement[:, left]
                + displacement[:, left] * displacement[:, right]
            )
        curvature.append(feature)
        names.append(f"potential_curvature:{dataset.family_names[left]}:{dataset.family_names[right]}")
    pieces.append(np.column_stack(curvature))
    pieces.append(variation[:, None])
    names.append("finite_phase_variation_cost")
    return np.hstack(pieces), tuple(names)


def phase_design(
    weights: np.ndarray,
    dataset: family_grp.Dataset,
    geometry: Geometry,
    tau: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if geometry is Geometry.FINITE:
        return finite_potential_design(weights, dataset, tau)
    return first_order.phase_design(
        weights,
        dataset,
        first_order.Candidate("linear", "symmetric", even_cost=True),
        tau,
    )


def load_phase_data(matched: matched_pair.MatchedSources, target: str) -> PhaseData:
    pair_dataset, pair_target, pair_frame = first_order.phase_pair_dataset(matched, target)
    fiber_dataset = composition.custom_dataset(
        matched.sources.reference,
        matched.sources.fiber.frame,
        matched.sources.fiber.weights,
        target,
        f"finite_potential_fiber_{target}",
    )
    center = matched.sources.fiber.frame["contrast_family"].eq("center_control").to_numpy()
    fiber_frame = matched.sources.fiber.frame.loc[~center].copy().reset_index(drop=True)
    fiber_target = fiber_frame[heterogeneous.fiber_delta_column(target)].to_numpy(dtype=float)
    fiber_groups = (fiber_frame["anchor_id"].astype(str) + "::" + fiber_frame["direction_id"].astype(str)).to_numpy(
        dtype=str
    )
    fiber_blocks = (fiber_frame["anchor_id"].astype(str) + "::" + fiber_frame["seed_block"].astype(str)).to_numpy(
        dtype=str
    )
    return PhaseData(
        pair_dataset=pair_dataset,
        pair_target=pair_target,
        pair_groups=pair_frame["pair_id"].to_numpy(dtype=str),
        fiber_dataset=fiber_dataset,
        fiber_frame=fiber_frame,
        fiber_target=fiber_target,
        fiber_groups=fiber_groups,
        fiber_blocks=fiber_blocks,
    )


def fiber_contrast_design(data: PhaseData, geometry: Geometry, tau: float) -> tuple[np.ndarray, tuple[str, ...]]:
    absolute, names = phase_design(data.fiber_dataset.weights, data.fiber_dataset, geometry, tau)
    frame = data.fiber_dataset.frame
    result = np.full((len(data.fiber_frame), absolute.shape[1]), np.nan, dtype=float)
    row_lookup = {str(row_id): index for index, row_id in enumerate(frame["row_id"].astype(str))}
    output_lookup = {str(row_id): index for index, row_id in enumerate(data.fiber_frame["row_id"].astype(str))}
    for (anchor, block), local in frame.groupby(["anchor_id", "seed_block"], sort=True):
        centers = local.loc[local["contrast_family"].eq("center_control")]
        if len(centers) != 1:
            raise ValueError(f"Expected one center for {anchor}/{block}, found {len(centers)}")
        center_index = row_lookup[str(centers.iloc[0]["row_id"])]
        for row_id in local.loc[~local["contrast_family"].eq("center_control"), "row_id"].astype(str):
            result[output_lookup[row_id]] = absolute[row_lookup[row_id]] - absolute[center_index]
    if not np.isfinite(result).all():
        raise RuntimeError("Incomplete fiber contrast design")
    return result, names


def stratified_fold_ids(groups: np.ndarray, count: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    mapping: dict[str, int] = {}
    for fold, (_train, test) in enumerate(splitter.split(unique)):
        for value in unique[test]:
            mapping[str(value)] = fold
    return np.asarray([mapping[str(value)] for value in groups], dtype=int)


def phase_fold_ids(data: PhaseData, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pair = stratified_fold_ids(data.pair_groups, count, seed)
    # Hold out complete shared-center blocks, not individual directions.
    unique_blocks = np.unique(data.fiber_blocks)
    if len(unique_blocks) % count != 0:
        raise ValueError(f"Cannot split {len(unique_blocks)} fiber blocks into {count} folds")
    block_order = np.random.default_rng(seed + 1).permutation(unique_blocks)
    mapping = {str(block): index % count for index, block in enumerate(block_order)}
    fiber = np.asarray([mapping[str(block)] for block in data.fiber_blocks], dtype=int)
    return pair, fiber


def weighted_equations(
    pair_design: np.ndarray,
    pair_target: np.ndarray,
    pair_indices: np.ndarray,
    fiber_design: np.ndarray,
    fiber_target: np.ndarray,
    fiber_blocks: np.ndarray,
    fiber_indices: np.ndarray,
    use_fiber: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_pair = pair_design[pair_indices]
    selected_pair_target = pair_target[pair_indices]
    pieces = [selected_pair]
    targets = [selected_pair_target]
    source_sizes = [len(selected_pair)]

    if use_fiber:
        selected = np.zeros(len(fiber_target), dtype=bool)
        selected[fiber_indices] = True
        fiber_pieces = []
        fiber_targets = []
        for block in np.unique(fiber_blocks[selected]):
            local = np.flatnonzero(selected & (fiber_blocks == block))
            whitening = heterogeneous.inverse_sqrt_shared_center_covariance(len(local))
            fiber_pieces.append(whitening @ fiber_design[local])
            fiber_targets.append(whitening @ fiber_target[local])
        whitened_design = np.vstack(fiber_pieces)
        whitened_target = np.concatenate(fiber_targets)
        pieces.append(whitened_design)
        targets.append(whitened_target)
        source_sizes.append(len(whitened_design))

    scale_basis = np.vstack(pieces)
    scale = np.maximum(np.sqrt(np.mean(scale_basis**2, axis=0)), 1e-10)
    weighted_design = []
    weighted_target = []
    source_weight = 1.0 / len(pieces)
    for design, target, size in zip(pieces, targets, source_sizes, strict=True):
        row_weight = math.sqrt(source_weight / size)
        weighted_design.append(row_weight * design / scale[None, :])
        weighted_target.append(row_weight * target)
    return np.vstack(weighted_design), np.concatenate(weighted_target), scale


def curvature_matrix(coefficients: cp.Expression, family_count: int) -> cp.Expression:
    rows: list[list[cp.Expression]] = [[cp.Constant(0.0) for _ in range(family_count)] for _ in range(family_count)]
    offset = family_count
    for value_index, (left, right) in enumerate(itertools.combinations_with_replacement(range(family_count), 2)):
        value = coefficients[offset + value_index]
        rows[left][right] = value
        rows[right][left] = value
    return cp.bmat(rows)


def fit_coefficients(
    data: PhaseData,
    candidate: Candidate,
    tau: float,
    ridge: float,
    pair_indices: np.ndarray,
    fiber_indices: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    pair_design, names = phase_design(data.pair_dataset.weights, data.pair_dataset, candidate.geometry, tau)
    fiber_design, fiber_names = fiber_contrast_design(data, candidate.geometry, tau)
    if names != fiber_names:
        raise ValueError("Pair and fiber feature order differs")
    design, target, scale = weighted_equations(
        pair_design,
        data.pair_target,
        pair_indices,
        fiber_design,
        data.fiber_target,
        data.fiber_blocks,
        fiber_indices,
        candidate.use_fiber,
    )
    augmented_design = np.vstack([design, math.sqrt(ridge) * np.eye(design.shape[1])])
    augmented_target = np.concatenate([target, np.zeros(design.shape[1], dtype=float)])

    if not candidate.positive_semidefinite:
        lower = np.full(design.shape[1], -np.inf, dtype=float)
        upper = np.full(design.shape[1], np.inf, dtype=float)
        lower[-1] = 0.0
        result = lsq_linear(augmented_design, augmented_target, bounds=(lower, upper), lsmr_tol="auto")
        if not result.success:
            raise RuntimeError(f"Phase solve failed: {result.message}")
        return np.asarray(result.x / scale, dtype=float), names

    scaled_coefficients = cp.Variable(design.shape[1])
    physical_coefficients = cp.multiply(1.0 / scale, scaled_coefficients)
    family_count = len(data.pair_dataset.family_names)
    constraints = [physical_coefficients[-1] >= 0.0, curvature_matrix(physical_coefficients, family_count) >> 0]
    objective = cp.Minimize(cp.sum_squares(augmented_design @ scaled_coefficients - augmented_target))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or scaled_coefficients.value is None:
        raise RuntimeError(f"PSD phase solve failed: {problem.status}")
    return np.asarray(scaled_coefficients.value / scale, dtype=float), names


def source_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    baseline = float(np.sqrt(np.mean(observed**2)))
    if baseline <= 1e-12:
        return 1.0
    return float(np.sqrt(np.mean((predicted - observed) ** 2)) / baseline)


def select_hyperparameters(
    data: PhaseData,
    candidate: Candidate,
    pair_indices: np.ndarray,
    fiber_indices: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, float, float]:
    local_pair_groups = data.pair_groups[pair_indices]
    local_fiber_blocks = data.fiber_blocks[fiber_indices]
    pair_fold = stratified_fold_ids(local_pair_groups, folds, seed)
    unique_blocks = np.unique(local_fiber_blocks)
    if len(unique_blocks) < folds:
        raise ValueError("Too few fiber blocks for nested selection")
    block_fold = stratified_fold_ids(unique_blocks, folds, seed + 1)
    block_to_fold = dict(zip(unique_blocks, block_fold, strict=True))
    fiber_fold = np.asarray([block_to_fold[block] for block in local_fiber_blocks], dtype=int)

    records = []
    for tau, ridge in itertools.product(TAU_GRID, RIDGE_GRID):
        pair_prediction = np.full(len(pair_indices), np.nan, dtype=float)
        fiber_prediction = np.full(len(fiber_indices), np.nan, dtype=float)
        pair_design, _ = phase_design(data.pair_dataset.weights, data.pair_dataset, candidate.geometry, tau)
        fiber_design, _ = fiber_contrast_design(data, candidate.geometry, tau)
        for fold in range(folds):
            pair_train = pair_indices[pair_fold != fold]
            pair_test_local = np.flatnonzero(pair_fold == fold)
            pair_test = pair_indices[pair_test_local]
            fiber_train = fiber_indices[fiber_fold != fold]
            fiber_test_local = np.flatnonzero(fiber_fold == fold)
            fiber_test = fiber_indices[fiber_test_local]
            coefficient, _ = fit_coefficients(
                data,
                candidate,
                tau,
                ridge,
                pair_train,
                fiber_train,
            )
            pair_prediction[pair_test_local] = pair_design[pair_test] @ coefficient
            fiber_prediction[fiber_test_local] = fiber_design[fiber_test] @ coefficient
        if not np.isfinite(pair_prediction).all() or not np.isfinite(fiber_prediction).all():
            raise RuntimeError("Incomplete inner-CV prediction")
        scores = [source_score(data.pair_target[pair_indices], pair_prediction)]
        if candidate.use_fiber:
            scores.append(source_score(data.fiber_target[fiber_indices], fiber_prediction))
        records.append((max(scores), tau, ridge))
    best = min(record[0] for record in records)
    eligible = [record for record in records if record[0] <= 1.01 * best]
    score, tau, ridge = sorted(eligible, key=lambda record: (-record[2], -record[1], record[0]))[0]
    return tau, ridge, score


def nested_predictions(
    data: PhaseData,
    candidate: Candidate,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    pair_prediction = np.full(len(data.pair_target), np.nan, dtype=float)
    fiber_prediction = np.full(len(data.fiber_target), np.nan, dtype=float)
    coefficients = []
    selections = []
    pair_fold, fiber_fold = phase_fold_ids(data, OUTER_FOLDS, 10_000 + target_index)
    for fold in range(OUTER_FOLDS):
        pair_train = np.flatnonzero(pair_fold != fold)
        pair_test = np.flatnonzero(pair_fold == fold)
        fiber_train = np.flatnonzero(fiber_fold != fold)
        fiber_test = np.flatnonzero(fiber_fold == fold)
        tau, ridge, inner_score = select_hyperparameters(
            data,
            candidate,
            pair_train,
            fiber_train,
            INNER_FOLDS,
            20_000 + 100 * target_index + fold,
        )
        coefficient, _ = fit_coefficients(data, candidate, tau, ridge, pair_train, fiber_train)
        pair_design, _ = phase_design(data.pair_dataset.weights, data.pair_dataset, candidate.geometry, tau)
        fiber_design, _ = fiber_contrast_design(data, candidate.geometry, tau)
        pair_prediction[pair_test] = pair_design[pair_test] @ coefficient
        fiber_prediction[fiber_test] = fiber_design[fiber_test] @ coefficient
        coefficients.append(coefficient)
        selections.append({"outer_fold": fold, "tau": tau, "ridge": ridge, "inner_score": inner_score})
    if not np.isfinite(pair_prediction).all() or not np.isfinite(fiber_prediction).all():
        raise RuntimeError("Incomplete outer-CV prediction")
    return pair_prediction, fiber_prediction, coefficients, selections


def stability(coefficients: list[np.ndarray]) -> dict[str, float]:
    matrix = np.vstack(coefficients)
    cosines = []
    for left, right in itertools.combinations(range(len(matrix)), 2):
        denominator = float(np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right]))
        if denominator > 1e-12:
            cosines.append(float(matrix[left] @ matrix[right] / denominator))
    signs = np.sign(np.where(np.abs(matrix) < 1e-8, 0.0, matrix))
    sign_agreement = [max(float(np.mean(column == value)) for value in (-1.0, 0.0, 1.0)) for column in signs.T]
    return {
        "median_pairwise_cosine": float(np.median(cosines)),
        "median_sign_agreement": float(np.median(sign_agreement)),
    }


def full_phase_fit(data: PhaseData, candidate: Candidate, target_index: int) -> PhaseFit:
    pair_indices = np.arange(len(data.pair_target))
    fiber_indices = np.arange(len(data.fiber_target))
    tau, ridge, _score = select_hyperparameters(
        data,
        candidate,
        pair_indices,
        fiber_indices,
        OUTER_FOLDS,
        30_000 + target_index,
    )
    coefficients, names = fit_coefficients(data, candidate, tau, ridge, pair_indices, fiber_indices)
    return PhaseFit(candidate, tau, ridge, coefficients, names)


def stage1_metrics(matched: matched_pair.MatchedSources) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    selection_rows = []
    prediction_rows = []
    for target_index, target in enumerate(composition.TARGETS):
        data = load_phase_data(matched, target)
        for candidate in CANDIDATES:
            print(f"Stage 1: {target}/{candidate.name}", flush=True)
            pair_prediction, fiber_prediction, coefficients, selections = nested_predictions(
                data, candidate, target_index
            )
            pair_metrics = composition.prediction_metrics(data.pair_target, pair_prediction)
            fiber_metrics = composition.prediction_metrics(data.fiber_target, fiber_prediction)
            stable = stability(coefficients)
            metric_rows.extend(
                [
                    {"target": target, "candidate": candidate.name, "source": "global_pair", **pair_metrics, **stable},
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "source": "frontier_fiber",
                        **fiber_metrics,
                        **stable,
                    },
                ]
            )
            for selection in selections:
                selection_rows.append({"target": target, "candidate": candidate.name, **selection})
            for source, observed, predicted, identifiers in (
                ("global_pair", data.pair_target, pair_prediction, data.pair_groups),
                ("frontier_fiber", data.fiber_target, fiber_prediction, data.fiber_groups),
            ):
                for identifier, actual, estimate in zip(identifiers, observed, predicted, strict=True):
                    prediction_rows.append(
                        {
                            "target": target,
                            "candidate": candidate.name,
                            "source": source,
                            "row_id": identifier,
                            "observed": actual,
                            "predicted": estimate,
                            "residual": estimate - actual,
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(selection_rows), pd.DataFrame(prediction_rows)


def stage1_gate(metrics: pd.DataFrame) -> dict[str, bool]:
    result = {}
    for candidate in CANDIDATES:
        passed = True
        for target in composition.TARGETS:
            pair = metrics.loc[
                (metrics["target"] == target)
                & (metrics["candidate"] == candidate.name)
                & (metrics["source"] == "global_pair")
            ].iloc[0]
            pair_baseline = metrics.loc[
                (metrics["target"] == target)
                & (metrics["candidate"] == "linear_pair_only")
                & (metrics["source"] == "global_pair"),
                "rmse",
            ].iloc[0]
            fiber = metrics.loc[
                (metrics["target"] == target)
                & (metrics["candidate"] == candidate.name)
                & (metrics["source"] == "frontier_fiber")
            ].iloc[0]
            fiber_observed = load_phase_data(matched_pair.matched_sources(), target).fiber_target
            fiber_baseline = float(np.sqrt(np.mean(fiber_observed**2)))
            passed &= float(pair["rmse"]) <= PAIR_TOLERANCE * float(pair_baseline)
            passed &= float(fiber["rmse"]) <= (1.0 - FIBER_IMPROVEMENT) * fiber_baseline
            passed &= float(pair["median_pairwise_cosine"]) >= STABILITY_COSINE
            passed &= float(pair["median_sign_agreement"]) >= STABILITY_SIGN
        result[candidate.name] = bool(passed)
    return result


def evaluate_development(
    matched: matched_pair.MatchedSources,
    promoted: list[Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], FullModel]]:
    metric_rows = []
    prediction_rows = []
    models: dict[tuple[str, str], FullModel] = {}
    for target_index, target in enumerate(composition.TARGETS):
        spine_dataset = first_order.aggregate_spine_dataset(matched, target)
        aggregate = hierarchical.fit_model(spine_dataset, composition.hpr_config(target), np.arange(spine_dataset.n))
        common_observed = matched.sources.common.frame[composition.TARGET_COLUMNS[target]].to_numpy(dtype=float)
        zero_prediction = aggregate.predict(first_order.tied_policy(matched.sources.common.weights, spine_dataset))
        metric_rows.append(
            {
                "target": target,
                "candidate": "zero_phase",
                "scope": "common_all",
                **composition.prediction_metrics(common_observed, zero_prediction),
            }
        )
        for row_id, observed, predicted in zip(
            matched.sources.common.frame["row_id"], common_observed, zero_prediction, strict=True
        ):
            prediction_rows.append(
                {
                    "target": target,
                    "candidate": "zero_phase",
                    "row_id": row_id,
                    "observed": observed,
                    "predicted": predicted,
                    "residual": predicted - observed,
                }
            )
        data = load_phase_data(matched, target)
        for candidate in promoted:
            phase = full_phase_fit(data, candidate, target_index)
            model = FullModel(aggregate, phase, data.pair_dataset)
            models[(target, candidate.name)] = model
            prediction = model.predict(matched.sources.common.weights)
            metric_rows.append(
                {
                    "target": target,
                    "candidate": candidate.name,
                    "scope": "common_all",
                    **composition.prediction_metrics(common_observed, prediction),
                }
            )
            for row_id, observed, predicted in zip(
                matched.sources.common.frame["row_id"], common_observed, prediction, strict=True
            ):
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "row_id": row_id,
                        "observed": observed,
                        "predicted": predicted,
                        "residual": predicted - observed,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), models


def stage2_gate(metrics: pd.DataFrame, promoted: list[Candidate]) -> dict[str, bool]:
    result = {}
    for candidate in promoted:
        passed = True
        for target in composition.TARGETS:
            baseline = metrics.loc[(metrics["target"] == target) & (metrics["candidate"] == "zero_phase")].iloc[0]
            row = metrics.loc[(metrics["target"] == target) & (metrics["candidate"] == candidate.name)].iloc[0]
            passed &= float(row["rmse"]) <= 1.05 * float(baseline["rmse"])
            passed &= float(row["regret_at_1"]) <= float(baseline["regret_at_1"]) + REGRET_TOLERANCE
            passed &= int(row["optimism_gt_0p05"]) <= int(baseline["optimism_gt_0p05"])
            passed &= abs(float(row["calibration_slope"]) - 1.0) <= abs(float(baseline["calibration_slope"]) - 1.0)
        result[candidate.name] = bool(passed)
    return result


def render_scatter(predictions: pd.DataFrame, title: str, path: Path) -> None:
    figure = px.scatter(
        predictions,
        x="observed",
        y="predicted",
        color="candidate",
        facet_col="target",
        hover_data=["row_id", "residual"],
        color_discrete_sequence=px.colors.qualitative.Safe,
        title=title,
    )
    bound = float(max(predictions["observed"].abs().max(), predictions["predicted"].abs().max()))
    figure.add_shape(type="line", x0=-bound, y0=-bound, x1=bound, y1=bound, line={"dash": "dash"})
    figure.update_layout(template="plotly_white", height=620, legend={"orientation": "h", "y": -0.18})
    figure.write_html(path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_registry(stage1: dict[str, bool], stage2: dict[str, bool], output_dir: Path) -> None:
    rows = []
    for candidate in CANDIDATES:
        status = "rejected_stage1"
        evidence = "Failed the frozen global/local contrast gate."
        if stage1.get(candidate.name, False):
            status = "promoted_stage1"
            evidence = "Passed the frozen global/local contrast gate."
        if stage2.get(candidate.name, False):
            status = "provisional_survivor"
            evidence = "Passed the frozen common-archive gate; StarCoder and optimization audits remain."
        elif candidate.name in stage2:
            status = "rejected_stage2"
            evidence = "Passed contrast identification but failed the frozen common-archive gate."
        rows.append(
            {
                "id": f"FPJS-{candidate.name}",
                "family": "Finite conservative phase potential",
                "relationship_to_prior": (
                    "Extends first-order PPT with an exact finite potential difference and a joint global/local "
                    "identification design."
                ),
                "materially_new_mechanism": (
                    "One scalar potential must jointly generate broad finite phase effects and local same-seed "
                    "directional effects."
                ),
                "governing_equations": "Y=F(a)+V(z+r)-V(z)+chi*q; V(z)=b^Tz+0.5*z^TMz.",
                "additional_degrees_of_freedom": 10,
                "status": status,
                "status_evidence": evidence,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "approach_registry.csv", index=False)


def write_report(
    stage1_metrics_frame: pd.DataFrame,
    stage1: dict[str, bool],
    development: pd.DataFrame,
    stage2: dict[str, bool],
    output_dir: Path,
) -> None:
    stage1_table = stage1_metrics_frame.copy()
    stage1_table["stage1_pass"] = stage1_table["candidate"].map(stage1)
    lines = [
        "# Finite-potential joint surrogate",
        "",
        "## Model",
        "",
        "The independently fitted tied-policy surface is combined with one conservative phase-response potential:",
        "",
        "$$Y(w^{(0)},w^{(1)})=F(a)+V(z+r)-V(z)+\\chi q,\\qquad V(z)=b^Tz+\\tfrac12 z^TMz.$ $".replace("$ $", "$$"),
        "",
        (
            "Global same-seed one/two-phase pairs identify finite differences of the potential. Frontier fibers "
            "identify local directional differences around two fitted anchors. The joint candidates fit both "
            "equation blocks with equal mean-squared-error weight and shared-center GLS. Phase tying gives $r=q=0$ "
            "exactly, so the "
            "single-phase restriction is the independently fitted $F(a)$ rather than an averaged two-phase fit."
        ),
        "",
        "## Stage 1: global and local phase identification",
        "",
        stage1_table[
            [
                "target",
                "candidate",
                "source",
                "rmse",
                "spearman",
                "calibration_slope",
                "median_pairwise_cosine",
                "median_sign_agreement",
                "stage1_pass",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Stage 2: frozen common archive",
        "",
    ]
    if development.empty:
        lines.append("No candidate cleared Stage 1; development outcomes were not inspected.")
    else:
        table = development.copy()
        table["stage2_pass"] = table["candidate"].map(stage2).fillna(False)
        lines.append(
            table[
                [
                    "target",
                    "candidate",
                    "rmse",
                    "spearman",
                    "calibration_slope",
                    "regret_at_1",
                    "optimism_gt_0p05",
                    "worst_optimism",
                    "stage2_pass",
                ]
            ].to_markdown(index=False, floatfmt=".5f")
        )
    survivors = [name for name, passed in stage2.items() if passed]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "Provisional survivors: "
                + ", ".join(survivors)
                + ". They still require StarCoder and raw-optimization audits."
                if survivors
                else "No candidate is promoted as a headline surrogate."
            ),
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing frozen preregistration: {PREREGISTRATION_PATH}")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text())
    if not preregistration.get("frozen_before_development_evaluation"):
        raise ValueError("Preregistration is not frozen")

    matched = matched_pair.matched_sources()
    metrics, selections, predictions = stage1_metrics(matched)
    metrics.to_csv(output_dir / "stage1_metrics.csv", index=False)
    selections.to_csv(output_dir / "stage1_hyperparameters.csv", index=False)
    predictions.to_csv(output_dir / "stage1_predictions.csv", index=False)
    render_scatter(
        predictions, "Nested OOF global and local phase contrasts", output_dir / "stage1_contrast_scatter.html"
    )
    stage1 = stage1_gate(metrics)

    promoted = [candidate for candidate in CANDIDATES if stage1[candidate.name]]
    development = pd.DataFrame()
    development_predictions = pd.DataFrame()
    stage2: dict[str, bool] = {}
    if promoted and not args.stage1_only:
        development, development_predictions, _models = evaluate_development(matched, promoted)
        development.to_csv(output_dir / "development_metrics.csv", index=False)
        development_predictions.to_csv(output_dir / "development_predictions.csv", index=False)
        render_scatter(
            development_predictions,
            "Frozen common-archive calibration",
            output_dir / "development_calibration_scatter.html",
        )
        stage2 = stage2_gate(development, promoted)

    write_registry(stage1, stage2, output_dir)
    write_report(metrics, stage1, development, stage2, output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "preregistration": str(PREREGISTRATION_PATH),
                "candidate_count": len(CANDIDATES),
                "stage1": stage1,
                "stage2": stage2,
                "development_evaluated": bool(not development.empty),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print((output_dir / "report.md").resolve())


if __name__ == "__main__":
    main()
