# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
# ]
# ///
"""Freeze the post-Wave-1 local KL0.05 phase-1 acquisition panel.

The first 100 fit branches were all far from the tied continuation and only
identified gross harm. Wave 2B therefore samples antithetic log-ratio rays
around tied. The redesign is explicitly post-outcome: Wave-1 outcomes motivated
the support change, but do not rank the geometry-only rays generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as branch_design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_kl0p05_local_wave2b_20260825"
DEFAULT_PREFIX_WEIGHTS = (
    SCRIPT_DIR / "reference_outputs" / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
)
TARGET_PREFIX = "shared_bounded_ensemble_kl0p05"
TARGET_METRIC = "uncheatable_bpb"
MIXTURE_BLOCK_SIZE = 2_048
DESIGN_SEED = 20_260_825
FIT_ROWS = 80
REFEREE_ROWS = 8
TIED_CONTROL_ROWS = 8
TOTAL_ROWS = FIT_ROWS + REFEREE_ROWS + TIED_CONTROL_ROWS
HISTORICAL_RAY_RADII = (0.08, 0.15, 0.23)
SPARSE_RAY_RADII = (0.08, 0.15)
DENSE_RAY_RADIUS = 0.15
HISTORICAL_RAY_COUNT = 2
SPARSE_RAY_COUNT = 8
DENSE_FIT_RAY_COUNT = 18
REFEREE_RAY_COUNT = 4
RADIUS_TOLERANCE = 0.003
SUPPORT_SLACK_EPOCHS = 1.0
RANDOM_POOL_ROWS = 20_000
MINIMUM_PREFIX_WEIGHT_FOR_SPARSE_RAY = 0.005
MINIMUM_LINE_DISTANCE = 0.45
PREFIX_WEIGHTS_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
WAVE1_FIT_MATRIX_SHA256 = "399ec79150a4f88de6d31917ac7fc1807410804f69c84400611a9eeaa6636e3c"
WAVE1_MATERIALIZATION_MANIFEST_SHA256 = "0d3f239002637768f696decb095877591bd7406193ba7573ffa6fc0e87ed5ebc"
SUPERSEDED_WAVE2_CONTRACT_SHA256 = "0ba8d66e1b58e351f747cdfa8fd037ecd60d20ea315965ed732c4466d6d61b91"
FRONTIER_MIXTURE_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_validation_20260712/mixtures/dphase_unch05_eff_e0p005.csv"
)
FRONTIER_MIXTURE_SHA256 = "57a2aa39a5b0e07d40fc6f55f14aaa86327c332e9ef86738b1cca547924c4a59"
TIED_DATA_SEED_BASE = 964_000


@dataclass(frozen=True)
class Direction:
    direction_id: str
    family: str
    values: np.ndarray
    radii: tuple[float, ...]
    outcome_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-weights", type=Path, default=DEFAULT_PREFIX_WEIGHTS)
    parser.add_argument("--wave1-fit-matrix", type=Path, required=True)
    parser.add_argument("--expected-wave1-fit-sha256", default=WAVE1_FIT_MATRIX_SHA256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_direction(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    norm = float(np.linalg.norm(centered))
    if norm <= 0.0:
        raise ValueError("A log-ratio direction has zero norm")
    return centered / norm


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    result = np.exp(shifted)
    return result / result.sum()


def runtime_weights(weights: np.ndarray) -> np.ndarray:
    return branch_design.runtime_counts(weights) / MIXTURE_BLOCK_SIZE


def hellinger(left: np.ndarray, right: np.ndarray) -> float:
    return branch_design.hellinger(left, right)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return branch_design.total_variation(left, right)


def mixture_on_ray(center: np.ndarray, direction: np.ndarray, signed_step: float) -> np.ndarray:
    return softmax(np.log(center) + signed_step * normalized_direction(direction))


def materialize_at_radius(
    center: np.ndarray,
    direction: np.ndarray,
    sign: int,
    radius: float,
) -> np.ndarray:
    lower = 0.0
    upper = 1.0
    while hellinger(center, mixture_on_ray(center, direction, sign * upper)) < radius:
        upper *= 2.0
        if upper > 1e6:
            raise ValueError(f"Direction cannot reach Hellinger radius {radius}")
    for _ in range(96):
        midpoint = 0.5 * (lower + upper)
        if hellinger(center, mixture_on_ray(center, direction, sign * midpoint)) < radius:
            lower = midpoint
        else:
            upper = midpoint
    result = runtime_weights(mixture_on_ray(center, direction, sign * 0.5 * (lower + upper)))
    achieved = hellinger(center, result)
    if abs(achieved - radius) > RADIUS_TOLERANCE:
        raise ValueError(f"Runtime lattice missed radius {radius}: achieved {achieved}")
    return result


def load_prefix(path: Path, buckets: tuple[str, ...]) -> np.ndarray:
    if file_sha256(path) != PREFIX_WEIGHTS_SHA256:
        raise ValueError("Prefix candidate weights changed")
    frame = pd.read_csv(path)
    group = frame[frame.candidate_id.eq(TARGET_PREFIX)]
    if tuple(group.bucket) != buckets:
        raise ValueError("KL0.05 prefix bucket order changed")
    weights = group.phase_0_weight.to_numpy(dtype=float)
    counts = group.phase_0_count.to_numpy(dtype=int)
    if not np.array_equal(counts, branch_design.runtime_counts(weights)):
        raise ValueError("KL0.05 prefix is not runtime exact")
    return weights


def load_frontier(buckets: tuple[str, ...]) -> np.ndarray:
    with fsspec.open(FRONTIER_MIXTURE_URI, "rb") as handle:
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != FRONTIER_MIXTURE_SHA256:
        raise ValueError("Historical frontier mixture changed")
    frame = pd.read_csv(io.BytesIO(payload)).set_index("domain")
    if set(frame.index) != set(buckets):
        raise ValueError("Historical frontier bucket set changed")
    weights = frame.loc[list(buckets), "phase_1_weight"].to_numpy(dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("Historical frontier phase-1 weights no longer sum to one")
    return weights


def load_wave1_weights(path: Path, buckets: tuple[str, ...], expected_sha256: str) -> np.ndarray:
    if file_sha256(path) != expected_sha256:
        raise ValueError("Wave-1 fit matrix changed")
    frame = pd.read_csv(path)
    if len(frame) != 100 or not frame.fit_budget.astype(bool).all():
        raise ValueError("Expected the complete 100-row Wave-1 fit matrix")
    columns = [f"phase_1_{bucket}" for bucket in buckets]
    if not set(columns).issubset(frame.columns):
        raise ValueError("Wave-1 fit matrix is missing phase-1 weights")
    weights = frame[columns].to_numpy(dtype=float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Wave-1 phase-1 mixtures no longer sum to one")
    return weights


def line_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = abs(float(np.dot(normalized_direction(left), normalized_direction(right))))
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, cosine) ** 2)))


def minimum_line_distance(direction: np.ndarray, references: list[np.ndarray]) -> float:
    return min(line_distance(direction, reference) for reference in references)


def support_ok(
    weights: np.ndarray,
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
) -> bool:
    phase_1_exposure = weights * phase_1_scales
    return bool(
        np.all(phase_1_exposure <= phase_1_caps + 1e-12)
        and np.all(phase_0_exposure + phase_1_exposure <= total_caps + 1e-12)
    )


def ray_is_feasible(
    center: np.ndarray,
    direction: np.ndarray,
    radii: tuple[float, ...],
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
) -> bool:
    try:
        points = [materialize_at_radius(center, direction, sign, radius) for radius in radii for sign in (-1, 1)]
    except ValueError:
        return False
    return all(support_ok(point, phase_0_exposure, phase_1_scales, phase_1_caps, total_caps) for point in points)


def select_maximin_lines(
    candidates: list[np.ndarray],
    references: list[np.ndarray],
    count: int,
) -> list[np.ndarray]:
    candidate_matrix = np.stack([normalized_direction(candidate) for candidate in candidates])
    reference_matrix = np.stack([normalized_direction(reference) for reference in references])
    max_absolute_cosine = np.max(np.abs(candidate_matrix @ reference_matrix.T), axis=1)
    available = np.ones(len(candidate_matrix), dtype=bool)
    selected: list[np.ndarray] = []
    for _ in range(count):
        scores = np.sqrt(np.maximum(0.0, 1.0 - max_absolute_cosine**2))
        scores[~available] = -np.inf
        position = int(np.argmax(scores))
        if scores[position] < MINIMUM_LINE_DISTANCE:
            raise ValueError(f"Cannot select {count} sufficiently distinct lines; best remaining={scores[position]:.4f}")
        selected_direction = candidate_matrix[position]
        selected.append(selected_direction)
        available[position] = False
        max_absolute_cosine = np.maximum(max_absolute_cosine, np.abs(candidate_matrix @ selected_direction))
    return selected


def sparse_candidate_pool(
    generator: np.random.Generator,
    center: np.ndarray,
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
) -> list[np.ndarray]:
    eligible = np.flatnonzero(center >= MINIMUM_PREFIX_WEIGHT_FOR_SPARSE_RAY)
    candidates: list[np.ndarray] = []
    for _ in range(RANDOM_POOL_ROWS):
        chosen = generator.choice(eligible, size=8, replace=False)
        values = np.zeros(len(center))
        values[chosen[:4]] = 1.0
        values[chosen[4:]] = -1.0
        direction = normalized_direction(values)
        if ray_is_feasible(
            center,
            direction,
            SPARSE_RAY_RADII,
            phase_0_exposure,
            phase_1_scales,
            phase_1_caps,
            total_caps,
        ):
            candidates.append(direction)
    if len(candidates) < SPARSE_RAY_COUNT:
        raise ValueError("Insufficient feasible sparse local directions")
    return candidates


def dense_candidate_pool(
    generator: np.random.Generator,
    center: np.ndarray,
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for _ in range(RANDOM_POOL_ROWS):
        direction = normalized_direction(generator.normal(size=len(center)))
        if ray_is_feasible(
            center,
            direction,
            (DENSE_RAY_RADIUS,),
            phase_0_exposure,
            phase_1_scales,
            phase_1_caps,
            total_caps,
        ):
            candidates.append(direction)
    if len(candidates) < DENSE_FIT_RAY_COUNT + REFEREE_RAY_COUNT:
        raise ValueError("Insufficient feasible dense local directions")
    return candidates


def build_directions(
    center: np.ndarray,
    frontier: np.ndarray,
    proportional: np.ndarray,
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
) -> tuple[list[Direction], list[Direction]]:
    historical = [
        Direction(
            "historical_frontier",
            "historical_ray",
            normalized_direction(np.log(frontier) - np.log(center)),
            HISTORICAL_RAY_RADII,
            "historical outcome-selected reference",
        ),
        Direction(
            "proportional",
            "historical_ray",
            normalized_direction(np.log(proportional) - np.log(center)),
            HISTORICAL_RAY_RADII,
            "operator-defined control reference",
        ),
    ]
    for direction in historical:
        if not ray_is_feasible(
            center,
            direction.values,
            direction.radii,
            phase_0_exposure,
            phase_1_scales,
            phase_1_caps,
            total_caps,
        ):
            raise ValueError(f"Historical direction {direction.direction_id} is infeasible")

    generator = np.random.default_rng(DESIGN_SEED)
    references = [direction.values for direction in historical]
    sparse_values = select_maximin_lines(
        sparse_candidate_pool(
            generator,
            center,
            phase_0_exposure,
            phase_1_scales,
            phase_1_caps,
            total_caps,
        ),
        references,
        SPARSE_RAY_COUNT,
    )
    sparse = [
        Direction(
            f"sparse_{position:02d}",
            "sparse_geometry",
            values,
            SPARSE_RAY_RADII,
            "outcome-blind deterministic geometry",
        )
        for position, values in enumerate(sparse_values)
    ]
    references.extend(sparse_values)
    dense_values = select_maximin_lines(
        dense_candidate_pool(
            generator,
            center,
            phase_0_exposure,
            phase_1_scales,
            phase_1_caps,
            total_caps,
        ),
        references,
        DENSE_FIT_RAY_COUNT + REFEREE_RAY_COUNT,
    )
    dense_fit = [
        Direction(
            f"dense_{position:02d}",
            "dense_geometry",
            values,
            (DENSE_RAY_RADIUS,),
            "outcome-blind deterministic geometry",
        )
        for position, values in enumerate(dense_values[:DENSE_FIT_RAY_COUNT])
    ]
    referee = [
        Direction(
            f"referee_{position:02d}",
            "referee_geometry",
            values,
            (DENSE_RAY_RADIUS,),
            "sealed outcome-blind deterministic geometry",
        )
        for position, values in enumerate(dense_values[DENSE_FIT_RAY_COUNT:])
    ]
    return [*historical, *sparse, *dense_fit], referee


def append_direction_rows(
    summary_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    *,
    direction: Direction,
    center: np.ndarray,
    buckets: tuple[str, ...],
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
    fit_budget: bool,
    referee_holdout: bool,
) -> None:
    for radius in direction.radii:
        radius_label = f"{round(radius * 1_000):03d}"
        for sign, sign_label in ((-1, "minus"), (1, "plus")):
            weights = materialize_at_radius(center, direction.values, sign, radius)
            continuation_id = f"{direction.direction_id}_h{radius_label}_{sign_label}"
            counts = branch_design.runtime_counts(weights)
            phase_1_exposure = weights * phase_1_scales
            if not support_ok(weights, phase_0_exposure, phase_1_scales, phase_1_caps, total_caps):
                raise ValueError(f"Support changed for {continuation_id}")
            summary_rows.append(
                {
                    "continuation_id": continuation_id,
                    "role": f"local_wave2b_{direction.family}",
                    "selection_tranche": direction.family,
                    "fit_budget": fit_budget,
                    "referee_holdout": referee_holdout,
                    "prefix_repeat_seed": 0,
                    "data_seed": 930_000,
                    "direction_id": direction.direction_id,
                    "direction_family": direction.family,
                    "direction_outcome_status": direction.outcome_status,
                    "sign": sign_label,
                    "target_hellinger": radius,
                    "achieved_hellinger_to_tied": hellinger(center, weights),
                    "tv_to_tied": total_variation(center, weights),
                    "max_phase_1_materialized_epoch": float(phase_1_exposure.max()),
                    "max_total_materialized_epoch": float((phase_0_exposure + phase_1_exposure).max()),
                    "weights_json": json.dumps(dict(zip(buckets, weights, strict=True)), sort_keys=True),
                }
            )
            for position, (bucket, count, weight) in enumerate(zip(buckets, counts, weights, strict=True)):
                weight_rows.append(
                    {
                        "continuation_id": continuation_id,
                        "role": f"local_wave2b_{direction.family}",
                        "selection_tranche": direction.family,
                        "fit_budget": fit_budget,
                        "referee_holdout": referee_holdout,
                        "prefix_repeat_seed": 0,
                        "data_seed": 930_000,
                        "direction_id": direction.direction_id,
                        "direction_family": direction.family,
                        "direction_outcome_status": direction.outcome_status,
                        "sign": sign_label,
                        "target_hellinger": radius,
                        "bucket": bucket,
                        "phase_1_count": int(count),
                        "phase_1_weight": float(weight),
                        "phase_1_materialized_epochs": float(phase_1_exposure[position]),
                        "phase_1_support_cap": float(phase_1_caps[position]),
                        "total_support_cap": float(total_caps[position]),
                    }
                )


def append_tied_controls(
    summary_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    center: np.ndarray,
    buckets: tuple[str, ...],
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
    phase_1_caps: np.ndarray,
    total_caps: np.ndarray,
) -> None:
    counts = branch_design.runtime_counts(center)
    phase_1_exposure = center * phase_1_scales
    for prefix_seed in (0, 1):
        for repeat_position in range(4):
            continuation_id = f"tied_seed{prefix_seed}_repeat{repeat_position + 1}"
            data_seed = TIED_DATA_SEED_BASE + 100 * prefix_seed + repeat_position
            summary_rows.append(
                {
                    "continuation_id": continuation_id,
                    "role": "local_wave2b_tied_control",
                    "selection_tranche": "tied_control",
                    "fit_budget": False,
                    "referee_holdout": False,
                    "prefix_repeat_seed": prefix_seed,
                    "data_seed": data_seed,
                    "direction_id": "tied",
                    "direction_family": "tied_control",
                    "direction_outcome_status": "fresh non-fit control",
                    "sign": "zero",
                    "target_hellinger": 0.0,
                    "achieved_hellinger_to_tied": 0.0,
                    "tv_to_tied": 0.0,
                    "max_phase_1_materialized_epoch": float(phase_1_exposure.max()),
                    "max_total_materialized_epoch": float((phase_0_exposure + phase_1_exposure).max()),
                    "weights_json": json.dumps(dict(zip(buckets, center, strict=True)), sort_keys=True),
                }
            )
            for position, (bucket, count, weight) in enumerate(zip(buckets, counts, center, strict=True)):
                weight_rows.append(
                    {
                        "continuation_id": continuation_id,
                        "role": "local_wave2b_tied_control",
                        "selection_tranche": "tied_control",
                        "fit_budget": False,
                        "referee_holdout": False,
                        "prefix_repeat_seed": prefix_seed,
                        "data_seed": data_seed,
                        "direction_id": "tied",
                        "direction_family": "tied_control",
                        "direction_outcome_status": "fresh non-fit control",
                        "sign": "zero",
                        "target_hellinger": 0.0,
                        "bucket": bucket,
                        "phase_1_count": int(count),
                        "phase_1_weight": float(weight),
                        "phase_1_materialized_epochs": float(phase_1_exposure[position]),
                        "phase_1_support_cap": float(phase_1_caps[position]),
                        "total_support_cap": float(total_caps[position]),
                    }
                )


def build_design(
    prefix_weights_path: Path,
    wave1_fit_matrix_path: Path,
    expected_wave1_fit_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = branch_design.load_canonical_panel_geometry()
    buckets = panel.buckets
    center = load_prefix(prefix_weights_path, buckets)
    frontier = load_frontier(buckets)
    proportional = runtime_weights(panel.proportional)
    wave1 = load_wave1_weights(wave1_fit_matrix_path, buckets, expected_wave1_fit_sha256)

    historical_phase_1_exposure = panel.phase1 * panel.c1[None, :]
    historical_total_exposure = panel.phase0 * panel.c0[None, :] + historical_phase_1_exposure
    phase_0_exposure = center * panel.c0
    tied_phase_1_exposure = center * panel.c1
    phase_1_caps = np.maximum(historical_phase_1_exposure.max(axis=0), tied_phase_1_exposure) + SUPPORT_SLACK_EPOCHS
    total_caps = (
        np.maximum(
            historical_total_exposure.max(axis=0),
            phase_0_exposure + tied_phase_1_exposure,
        )
        + SUPPORT_SLACK_EPOCHS
    )

    fit_directions, referee_directions = build_directions(
        center,
        frontier,
        proportional,
        phase_0_exposure,
        panel.c1,
        phase_1_caps,
        total_caps,
    )
    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for direction in fit_directions:
        append_direction_rows(
            summary_rows,
            weight_rows,
            direction=direction,
            center=center,
            buckets=buckets,
            phase_0_exposure=phase_0_exposure,
            phase_1_scales=panel.c1,
            phase_1_caps=phase_1_caps,
            total_caps=total_caps,
            fit_budget=True,
            referee_holdout=False,
        )
    for direction in referee_directions:
        append_direction_rows(
            summary_rows,
            weight_rows,
            direction=direction,
            center=center,
            buckets=buckets,
            phase_0_exposure=phase_0_exposure,
            phase_1_scales=panel.c1,
            phase_1_caps=phase_1_caps,
            total_caps=total_caps,
            fit_budget=False,
            referee_holdout=True,
        )
    append_tied_controls(
        summary_rows,
        weight_rows,
        center,
        buckets,
        phase_0_exposure,
        panel.c1,
        phase_1_caps,
        total_caps,
    )

    summary = pd.DataFrame(summary_rows)
    weights = pd.DataFrame(weight_rows)
    direction_rows = [
        {
            "direction_id": direction.direction_id,
            "direction_family": direction.family,
            "direction_outcome_status": direction.outcome_status,
            "radii_json": json.dumps(direction.radii),
            **{f"clr_direction_{bucket}": float(value) for bucket, value in zip(buckets, direction.values, strict=True)},
        }
        for direction in [*fit_directions, *referee_directions]
    ]
    directions = pd.DataFrame(direction_rows)

    if len(summary) != TOTAL_ROWS or int(summary.fit_budget.sum()) != FIT_ROWS:
        raise ValueError(f"Local Wave-2B row budget changed: {len(summary)} total, {summary.fit_budget.sum()} fit")
    if int(summary.referee_holdout.sum()) != REFEREE_ROWS:
        raise ValueError("Local Wave-2B referee budget changed")
    if int(summary.selection_tranche.eq("tied_control").sum()) != TIED_CONTROL_ROWS:
        raise ValueError("Local Wave-2B tied-control budget changed")
    fit_and_referee = summary[~summary.selection_tranche.eq("tied_control")]
    runtime_keys = [
        tuple(branch_design.runtime_counts(np.asarray([json.loads(value)[bucket] for bucket in buckets], dtype=float)))
        for value in fit_and_referee.weights_json
    ]
    if len(runtime_keys) != len(set(runtime_keys)):
        raise ValueError("Local fit or referee rows collide after runtime materialization")
    wave1_keys = {tuple(branch_design.runtime_counts(row)) for row in wave1}
    if set(runtime_keys) & wave1_keys:
        raise ValueError("Local Wave-2B collides with a Wave-1 fit row")

    direction_values = [direction.values for direction in [*fit_directions, *referee_directions]]
    minimum_direction_distance = min(
        line_distance(left, right)
        for left_position, left in enumerate(direction_values)
        for right in direction_values[left_position + 1 :]
    )
    fit_weights = np.stack(
        [
            np.asarray([json.loads(value)[bucket] for bucket in buckets], dtype=float)
            for value in summary[summary.fit_budget].weights_json
        ]
    )
    minimum_wave1_distance = float(min(hellinger(local, broad) for local in fit_weights for broad in wave1))
    manifest: dict[str, object] = {
        "contract_version": "delphi_phase1_kl0p05_local_wave2b_20260825_v1",
        "selection_mode": "post_wave1_local_redesign",
        "target_prefix": TARGET_PREFIX,
        "target_metric": TARGET_METRIC,
        "wave1_endpoint_outcomes_used": True,
        "wave1_outcome_use": (
            "Wave-1 outcomes established the local-support failure and motivated replacing the broad fallback. "
            "They do not rank the deterministic sparse or dense geometry rays."
        ),
        "wave2_endpoint_outcomes_used": False,
        "superseded_wave2_contract_sha256": SUPERSEDED_WAVE2_CONTRACT_SHA256,
        "rows": {"fit": FIT_ROWS, "referee": REFEREE_ROWS, "tied_control": TIED_CONTROL_ROWS, "total": TOTAL_ROWS},
        "directions": {
            "historical": HISTORICAL_RAY_COUNT,
            "sparse_geometry": SPARSE_RAY_COUNT,
            "dense_fit_geometry": DENSE_FIT_RAY_COUNT,
            "sealed_referee_geometry": REFEREE_RAY_COUNT,
            "minimum_projective_line_distance": minimum_direction_distance,
        },
        "radii": {
            "historical": list(HISTORICAL_RAY_RADII),
            "sparse": list(SPARSE_RAY_RADII),
            "dense": DENSE_RAY_RADIUS,
            "metric": "Hellinger distance from the runtime-exact tied continuation",
            "construction": "antithetic centered-log-ratio rays, then 1/2048 runtime materialization",
            "tolerance": RADIUS_TOLERANCE,
        },
        "support": {
            "historical_per_bucket_phase1_and_total_envelopes": True,
            "tied_continuation_included_in_envelope": True,
            "per_bucket_slack_epochs": SUPPORT_SLACK_EPOCHS,
            "minimum_hellinger_to_wave1_fit_support": minimum_wave1_distance,
        },
        "controls": {
            "seed0_fresh_data_repeats": 4,
            "seed1_fresh_data_repeats": 4,
            "fit_budget": False,
        },
        "provenance": {
            "design_seed": DESIGN_SEED,
            "prefix_weights_sha256": PREFIX_WEIGHTS_SHA256,
            "wave1_fit_matrix_sha256": expected_wave1_fit_sha256,
            "wave1_materialization_manifest_sha256": WAVE1_MATERIALIZATION_MANIFEST_SHA256,
            "historical_frontier_uri": FRONTIER_MIXTURE_URI,
            "historical_frontier_sha256": FRONTIER_MIXTURE_SHA256,
        },
    }
    return summary, weights, directions, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, weights, directions, manifest = build_design(
        args.prefix_weights,
        args.wave1_fit_matrix,
        args.expected_wave1_fit_sha256,
    )
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    directions_path = args.output_dir / "directions.csv"
    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    directions.to_csv(directions_path, index=False)
    contract = {
        **manifest,
        "artifacts": {
            summary_path.name: file_sha256(summary_path),
            weights_path.name: file_sha256(weights_path),
            directions_path.name: file_sha256(directions_path),
        },
    }
    contract_path = args.output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    launch_manifest = {
        "contract_version": contract["contract_version"],
        "contract_sha256": file_sha256(contract_path),
        "selection_mode": contract["selection_mode"],
        "target_prefix": contract["target_prefix"],
        "rows": contract["rows"],
        "artifacts": contract["artifacts"],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(launch_manifest, indent=2, sort_keys=True) + "\n")
    report = f"""# Delphi KL0.05 local Wave 2B

This post-Wave-1 redesign replaces the paused broad Wave-2 contract `{SUPERSEDED_WAVE2_CONTRACT_SHA256}`.
All 100 existing fit branches were far from tied and worse than tied, so they identify gross harm but not the
local phase-1 gradient. This panel spends all 80 remaining fit rows on paired local measurements.

- 12 rows: historical-frontier and proportional rays, each at Hellinger 0.08, 0.15, and 0.23 in both directions.
- 32 rows: eight deterministic sparse geometry rays at 0.08 and 0.15 in both directions.
- 36 rows: eighteen deterministic dense maximin rays at 0.15 in both directions.
- 8 non-fit rows: four sealed referee rays at 0.15 in both directions.
- 8 non-fit rows: fresh tied controls, four each from prefix seeds 0 and 1.

The acquisition identifies directional slopes and coarse curvature in a prespecified 28-line subspace. It does
not claim to identify an unrestricted 38-dimensional optimum or Hessian. Finalist confirmation remains a later,
separate experiment and does not consume the 180-row fit budget.

Continuation weights SHA-256: `{contract['artifacts']['continuation_weights.csv']}`
Contract SHA-256: `{launch_manifest['contract_sha256']}`
Manifest SHA-256: use the generated file hash together with the contract hash at launch.
"""
    (args.output_dir / "report.md").write_text(report)
    print(summary.groupby(["fit_budget", "referee_holdout", "selection_tranche"]).size().to_string())
    print(json.dumps(launch_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
