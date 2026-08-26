# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scipy==1.17.0",
# ]
# ///
"""Select the frozen adaptive and outcome-blind proportional-prefix Wave 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_delphi_phase1_harsh_cap_branch_response as response,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_WAVE1_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_results_20260826"
DEFAULT_WAVE1_DESIGN = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_20260825"
DEFAULT_WAVE1_MANIFEST = DEFAULT_WAVE1_DESIGN / "manifest.json"
DEFAULT_VALIDATED_FRONTIER_CONTRACT = DEFAULT_WAVE1_DESIGN / "validated_frontier_contract.json"
DEFAULT_MODEL_FIT = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_fit_20260826"
DEFAULT_CANDIDATE_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
DEFAULT_SELECTED_PREFIXES = REFERENCE_OUTPUTS / "delphi_phase0_proportional_prefix_20260825" / "selected_prefixes.json"
DEFAULT_WAVE2_CONTRACT = (
    REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_contract_20260826" / "contract.json"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave2_20260826"
TARGET_PREFIX = "proportional_control"
FIT_DATA_SEED = design.FIT_DATA_SEED
ADAPTIVE_ROWS = 40
COVERAGE_ROWS = 40
EXPLOIT_ROWS = 16
LOCAL_REFINEMENT_ROWS = 12
DISAGREEMENT_ROWS = 12
LOCAL_TRANSFER_AMOUNTS = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_WAVE1_RESULTS / "branch_results.csv")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_WAVE1_RESULTS / "coverage.json")
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_MODEL_FIT / "frozen_model_contract.json")
    parser.add_argument("--wave1-summary", type=Path, default=DEFAULT_WAVE1_DESIGN / "continuation_summary.csv")
    parser.add_argument("--wave1-weights", type=Path, default=DEFAULT_WAVE1_DESIGN / "continuation_weights.csv")
    parser.add_argument("--wave1-manifest", type=Path, default=DEFAULT_WAVE1_MANIFEST)
    parser.add_argument(
        "--validated-frontier-contract",
        type=Path,
        default=DEFAULT_VALIDATED_FRONTIER_CONTRACT,
    )
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--selected-prefixes", type=Path, default=DEFAULT_SELECTED_PREFIXES)
    parser.add_argument("--wave2-contract", type=Path, default=DEFAULT_WAVE2_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different frozen artifact: {path}")
        return
    path.write_bytes(payload)


def count_key(weights: np.ndarray) -> tuple[int, ...]:
    return tuple(design.common_design.runtime_counts(weights).tolist())


def load_wave1(
    results_path: Path,
    coverage_path: Path,
    summary_path: Path,
    weights_path: Path,
    candidate_weights_path: Path,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, set[tuple[int, ...]]]:
    results = pd.read_csv(results_path)
    response.validate_sealed_input(results, coverage_path)
    fit_rows = results[results.fit_budget.astype(bool)].sort_values("run_order")
    if len(fit_rows) != design.FIT_ROWS_PER_PREFIX:
        raise ValueError(f"Expected {design.FIT_ROWS_PER_PREFIX} Wave-1 fit rows, got {len(fit_rows)}")
    continuation_ids = tuple(fit_rows.continuation_id)
    buckets, measured = response.load_weights(weights_path, TARGET_PREFIX, continuation_ids)
    center = response.tied_center(candidate_weights_path, TARGET_PREFIX, buckets)
    baselines = response.control_baselines(results)
    effects = fit_rows[response.TARGET].to_numpy(dtype=float) - float(baselines["matched_tied_bpb"])
    all_weights = pd.read_csv(weights_path)
    all_keys = {
        tuple(rows.phase_1_count.to_numpy(dtype=int)) for _, rows in all_weights.groupby("continuation_id", sort=False)
    }
    summary = pd.read_csv(summary_path)
    deployment_anchors = summary[
        summary.fit_budget.astype(bool) & summary.source.eq("deployment_anchor:validated_cap4_frontier:1.00")
    ]
    if len(deployment_anchors) != 1:
        raise ValueError(f"Expected one validated-frontier deployment anchor, got {len(deployment_anchors)}")
    frontier_id = str(deployment_anchors.continuation_id.iloc[0])
    frontier_position = continuation_ids.index(frontier_id)
    frontier_weights = measured[frontier_position]
    referee_ids = tuple(summary[summary.role.eq("sealed_geometry_referee")].continuation_id)
    _, referee_weights = response.load_weights(weights_path, TARGET_PREFIX, referee_ids)
    all_keys.update(count_key(row) for row in referee_weights)
    return buckets, center, measured, effects, frontier_weights, all_keys


def candidate_geometry(
    center: np.ndarray,
    buckets: tuple[str, ...],
    excluded: set[tuple[int, ...]],
) -> tuple[np.ndarray, list[str]]:
    panel = design.common_design.load_canonical_panel_geometry()
    if panel.buckets != buckets:
        raise ValueError("Canonical branch geometry bucket order changed")
    anchors = design.anchor_mixtures(buckets, design.runtime_weights(panel.proportional))
    points, _ = design.generate_pool(center, anchors, center * panel.c0, panel.c1)
    rows = [(point.weights, point.source) for point in points.values() if point.counts not in excluded]
    if not rows:
        raise ValueError("Wave-2 candidate pool is empty")
    weights = np.stack([row[0] for row in rows])
    sources = [row[1] for row in rows]
    if design.rank(design.feature_matrix(weights, center, "sqrt")) != len(center):
        raise ValueError("Wave-2 candidate pool lost square-root-coordinate rank")
    return weights, sources


def select_coverage(weights: np.ndarray, center: np.ndarray) -> list[int]:
    features = design.feature_matrix(weights, center, "sqrt")
    pivots = design.qr_rows(features, len(center))
    selected = list(dict.fromkeys(pivots))
    selected.extend(
        design.maximin_rows(
            features,
            np.ones(len(weights), dtype=bool),
            selected,
            COVERAGE_ROWS - len(selected),
        )
    )
    if len(selected) != COVERAGE_ROWS:
        raise ValueError(f"Could select only {len(selected)} outcome-blind coverage rows")
    return selected


def fit_models(
    measured: np.ndarray,
    effects: np.ndarray,
    center: np.ndarray,
) -> tuple[pd.DataFrame, list[response.ResponseModel]]:
    folds = response.geometric_fold_ids(measured, center, response.OUTER_FOLDS, response.CV_SEED)
    metrics = response.parameter_cv(measured, effects, center, folds)
    eligible = response.eligible_parameters(metrics)
    representatives = eligible.groupby("feature_kind", sort=False).head(1)
    models = [
        response.fit_model(measured, effects, center, str(row.feature_kind), float(row.alpha))
        for row in representatives.itertuples(index=False)
    ]
    return metrics, models


def select_exploitation(
    weights: np.ndarray,
    center: np.ndarray,
    models: list[response.ResponseModel],
    penalties: tuple[float, ...],
    unavailable: set[int],
) -> list[int]:
    radius2 = response.hellinger(weights, center) ** 2
    predictions = np.stack([response.predict(model, weights, center) for model in models])
    selected: list[int] = []
    for model_predictions in predictions:
        for penalty in penalties:
            if len(selected) == EXPLOIT_ROWS:
                break
            for index in np.argsort(model_predictions + penalty * radius2):
                position = int(index)
                if position not in unavailable and position not in selected:
                    selected.append(position)
                    break
        if len(selected) == EXPLOIT_ROWS:
            break
    selected_model = models[0]
    for index in np.argsort(response.predict(selected_model, weights, center)):
        if len(selected) == EXPLOIT_ROWS:
            break
        position = int(index)
        if position not in unavailable and position not in selected:
            selected.append(position)
    if len(selected) != EXPLOIT_ROWS:
        raise ValueError(f"Could select only {len(selected)} exploitation rows")
    return selected


def local_neighbors(
    seeds: np.ndarray,
    center: np.ndarray,
    excluded: set[tuple[int, ...]],
) -> tuple[np.ndarray, list[str]]:
    panel = design.common_design.load_canonical_panel_geometry()
    phase0_exposure = center * panel.c0
    points: dict[tuple[int, ...], tuple[np.ndarray, str]] = {}
    for seed_position, seed in enumerate(seeds):
        counts = design.common_design.runtime_counts(seed)
        donors = np.flatnonzero(counts > 0)
        for donor in donors:
            for recipient in range(len(counts)):
                if donor == recipient:
                    continue
                for amount in LOCAL_TRANSFER_AMOUNTS:
                    if counts[donor] < amount:
                        continue
                    moved = counts.copy()
                    moved[donor] -= amount
                    moved[recipient] += amount
                    key = tuple(moved.tolist())
                    weights = moved / design.MIXTURE_BLOCK_SIZE
                    if key in excluded or not design.support_ok(weights, phase0_exposure, panel.c1):
                        continue
                    if float(response.hellinger(weights[None, :], center)[0]) < design.MINIMUM_HELLINGER:
                        continue
                    points.setdefault(key, (weights, f"adaptive_local:{seed_position:02d}:{amount}"))
    if not points:
        raise ValueError("No feasible local-refinement neighbors were generated")
    return np.stack([row[0] for row in points.values()]), [row[1] for row in points.values()]


def select_local_refinements(
    seed_weights: np.ndarray,
    center: np.ndarray,
    excluded: set[tuple[int, ...]],
) -> tuple[np.ndarray, list[str]]:
    neighbors, sources = local_neighbors(seed_weights, center, excluded)
    combined = np.vstack([seed_weights, neighbors])
    features = design.feature_matrix(combined, center, "sqrt")
    available = np.zeros(len(combined), dtype=bool)
    available[len(seed_weights) :] = True
    indices = design.maximin_rows(
        features,
        available,
        list(range(len(seed_weights))),
        LOCAL_REFINEMENT_ROWS,
    )
    local_indices = [index - len(seed_weights) for index in indices]
    if len(local_indices) != LOCAL_REFINEMENT_ROWS:
        raise ValueError(f"Could select only {len(local_indices)} local-refinement rows")
    return neighbors[local_indices], [sources[index] for index in local_indices]


def select_disagreement(
    fold_predictions: np.ndarray,
    unavailable: set[int],
) -> list[int]:
    if fold_predictions.ndim != 2 or fold_predictions.shape[0] < 2:
        raise ValueError("Model-disagreement acquisition requires at least two fold models")
    disagreement = fold_predictions.std(axis=0, ddof=1)
    selected = []
    for index in np.argsort(-disagreement):
        position = int(index)
        if position not in unavailable:
            selected.append(position)
        if len(selected) == DISAGREEMENT_ROWS:
            break
    if len(selected) != DISAGREEMENT_ROWS:
        raise ValueError(f"Could select only {len(selected)} disagreement rows")
    return selected


def selected_points(
    center: np.ndarray,
    pool: np.ndarray,
    sources: list[str],
    coverage_indices: list[int],
    models: list[response.ResponseModel],
    fold_predictions: np.ndarray,
    penalties: tuple[float, ...],
    excluded: set[tuple[int, ...]],
    local_seed_weights: np.ndarray,
) -> list[design.CandidatePoint]:
    unavailable = set(coverage_indices)
    exploit = select_exploitation(pool, center, models, penalties, unavailable)
    unavailable.update(exploit)
    selected_pool_keys = {count_key(pool[index]) for index in unavailable}
    local_weights, local_sources = select_local_refinements(
        np.vstack([local_seed_weights, pool[exploit]]),
        center,
        excluded | selected_pool_keys,
    )
    excluded.update(count_key(row) for row in local_weights)
    unavailable.update(index for index, row in enumerate(pool) if count_key(row) in excluded)
    disagreement = select_disagreement(fold_predictions, unavailable)
    adaptive = [design.CandidatePoint(count_key(pool[index]), f"adaptive_exploit:{sources[index]}") for index in exploit]
    adaptive.extend(
        design.CandidatePoint(count_key(weights), source)
        for weights, source in zip(local_weights, local_sources, strict=True)
    )
    adaptive.extend(
        design.CandidatePoint(count_key(pool[index]), f"adaptive_disagreement:{sources[index]}")
        for index in disagreement
    )
    coverage = [
        design.CandidatePoint(count_key(pool[index]), f"coverage_maximin:{sources[index]}") for index in coverage_indices
    ]
    rows = [*adaptive, *coverage]
    if len(adaptive) != ADAPTIVE_ROWS or len(coverage) != COVERAGE_ROWS or len({row.counts for row in rows}) != 80:
        raise ValueError("Wave-2 allocation contains duplicate or missing rows")
    return rows


def build_design(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    wave2_contract = cast(dict[str, object], json.loads(args.wave2_contract.read_text()))
    if wave2_contract.get("contract_version") != "delphi_phase1_proportional_prefix_wave2_20260826_v1":
        raise ValueError("Wave-2 contract changed")
    provenance = cast(dict[str, object], wave2_contract.get("provenance", {}))
    expected_provenance = {
        "selected_prefixes_sha256": file_sha256(args.selected_prefixes),
        "validated_frontier_contract_sha256": file_sha256(args.validated_frontier_contract),
        "wave1_continuation_summary_sha256": file_sha256(args.wave1_summary),
        "wave1_continuation_weights_sha256": file_sha256(args.wave1_weights),
        "wave1_design_manifest_sha256": file_sha256(args.wave1_manifest),
    }
    if provenance != expected_provenance:
        raise ValueError("Wave-2 contract provenance does not match the frozen Wave-1 artifacts")
    eligibility = cast(dict[str, object], wave2_contract.get("eligibility", {}))
    if eligibility.get("fit_data_seed") != FIT_DATA_SEED:
        raise ValueError("Wave-2 fit data seed changed")
    if eligibility.get("runtime_mixture_block_size") != design.MIXTURE_BLOCK_SIZE:
        raise ValueError("Wave-2 runtime mixture lattice changed")
    if eligibility.get("maximum_total_materialized_epoch") != design.TOTAL_MATERIALIZED_EPOCH_CAP:
        raise ValueError("Wave-2 materialized-epoch cap changed")
    acquisition = cast(dict[str, object], wave2_contract.get("acquisition", {}))
    adaptive_policy = cast(dict[str, object], acquisition.get("adaptive_policy", {}))
    expected_allocation = {
        "total_fit_rows": ADAPTIVE_ROWS + COVERAGE_ROWS,
        "adaptive_fit_rows": ADAPTIVE_ROWS,
        "outcome_blind_coverage_rows": COVERAGE_ROWS,
    }
    if any(acquisition.get(key) != value for key, value in expected_allocation.items()):
        raise ValueError("Wave-2 top-level row allocation changed")
    expected_adaptive_allocation = {
        "exploitation_rows": EXPLOIT_ROWS,
        "local_refinement_rows": LOCAL_REFINEMENT_ROWS,
        "model_disagreement_rows": DISAGREEMENT_ROWS,
    }
    if any(adaptive_policy.get(key) != value for key, value in expected_adaptive_allocation.items()):
        raise ValueError("Wave-2 adaptive row allocation changed")
    model_contract = cast(dict[str, object], json.loads(args.model_contract.read_text()))
    seal = cast(dict[str, object], model_contract.get("seal", {}))
    if seal.get("referee_outcomes_present_in_fit_input") is not False:
        raise ValueError("Wave-1 referee outcomes leaked into the model contract")
    buckets, center, measured, effects, frontier_weights, excluded = load_wave1(
        args.results,
        args.coverage,
        args.wave1_summary,
        args.wave1_weights,
        args.candidate_weights,
    )
    contract_inputs = cast(dict[str, object], model_contract.get("inputs", {}))
    expected_input_hashes = {
        "results_sha256": file_sha256(args.results),
        "coverage_sha256": file_sha256(args.coverage),
        "design_summary_sha256": file_sha256(args.wave1_summary),
        "design_weights_sha256": file_sha256(args.wave1_weights),
        "candidate_weights_sha256": file_sha256(args.candidate_weights),
    }
    if any(contract_inputs.get(key) != value for key, value in expected_input_hashes.items()):
        raise ValueError("Frozen Wave-1 model contract does not match the Wave-1 inputs")
    pool, sources = candidate_geometry(center, buckets, excluded)
    coverage_indices = select_coverage(pool, center)
    parameter_metrics, models = fit_models(measured, effects, center)
    selected_feature, selected_alpha = response.selected_parameter(parameter_metrics)
    frozen_candidates = cast(dict[str, object], model_contract.get("frozen_candidates", {}))
    frozen_candidate = cast(dict[str, object], frozen_candidates.get(TARGET_PREFIX, {}))
    if (frozen_candidate.get("feature_kind"), frozen_candidate.get("ridge_alpha")) != (
        selected_feature,
        selected_alpha,
    ):
        raise ValueError("Wave-2 parameter selection does not reproduce the frozen Wave-1 model")
    fold_predictions, _ = response.fold_ensemble_predictions(measured, effects, center, pool)
    eligible_predictions = np.stack([response.predict(model, pool, center) for model in models])
    disagreement_predictions = eligible_predictions if len(models) > 1 else fold_predictions
    model_selection = cast(dict[str, object], wave2_contract["model_selection"])
    penalties = tuple(float(value) for value in cast(list[float], model_selection["hellinger_squared_penalty_grid"]))
    points = selected_points(
        center,
        pool,
        sources,
        coverage_indices,
        models,
        disagreement_predictions,
        penalties,
        excluded,
        frontier_weights[None, :],
    )

    panel = design.common_design.load_canonical_panel_geometry()
    summary_rows, weight_rows = design.design_rows(
        TARGET_PREFIX,
        center,
        points,
        [],
        buckets,
        panel.c0,
        panel.c1,
    )
    summary = pd.DataFrame(summary_rows)
    weights = pd.DataFrame(weight_rows)
    summary = summary[summary.fit_budget.astype(bool)].copy().reset_index(drop=True)
    weights = weights[weights.fit_budget.astype(bool)].copy()
    role_by_id = {
        str(row.continuation_id): (
            "adaptive_model_fit" if str(row.source).startswith("adaptive_") else "outcome_blind_coverage_fit"
        )
        for row in summary.itertuples(index=False)
    }
    summary["role"] = summary.continuation_id.map(role_by_id)
    weights["role"] = weights.continuation_id.map(role_by_id)
    summary["data_seed"] = FIT_DATA_SEED
    weights["data_seed"] = FIT_DATA_SEED
    id_by_previous_id = {
        str(continuation_id): f"wave2_{position:03d}" for position, continuation_id in enumerate(summary.continuation_id)
    }
    summary["continuation_id"] = summary.continuation_id.map(id_by_previous_id)
    weights["continuation_id"] = weights.continuation_id.map(id_by_previous_id)

    selected_weights = np.stack(
        [
            weights.loc[weights.continuation_id.eq(continuation_id), "phase_1_weight"].to_numpy(dtype=float)
            for continuation_id in summary.continuation_id
        ]
    )
    diagnostics = {
        "direct_feature_rank": design.rank(design.feature_matrix(selected_weights, center, "direct")),
        "sqrt_feature_rank": design.rank(design.feature_matrix(selected_weights, center, "sqrt")),
        "adaptive_rows": int(summary.role.eq("adaptive_model_fit").sum()),
        "outcome_blind_coverage_rows": int(summary.role.eq("outcome_blind_coverage_fit").sum()),
        "hellinger_min": float(summary.hellinger_to_tied.min()),
        "hellinger_median": float(summary.hellinger_to_tied.median()),
        "hellinger_max": float(summary.hellinger_to_tied.max()),
    }
    if (diagnostics["direct_feature_rank"], diagnostics["sqrt_feature_rank"]) != (38, 39):
        raise ValueError(f"Wave-2 design is rank deficient: {diagnostics}")
    manifest: dict[str, object] = {
        "contract_version": "delphi_phase1_proportional_prefix_wave2_design_20260826_v1",
        "selected_candidate_ids": [TARGET_PREFIX],
        "rows": {
            "controls_per_prefix": 0,
            "fit_per_prefix": 80,
            "sealed_referees_per_prefix": 0,
            "total": 80,
        },
        "role_counts_per_prefix": summary.role.value_counts().to_dict(),
        "diagnostics": {TARGET_PREFIX: diagnostics},
        "inputs": {
            "wave1_results_sha256": file_sha256(args.results),
            "wave1_coverage_sha256": file_sha256(args.coverage),
            "wave1_summary_sha256": file_sha256(args.wave1_summary),
            "wave1_weights_sha256": file_sha256(args.wave1_weights),
            "candidate_weights_sha256": file_sha256(args.candidate_weights),
            "selected_prefixes_sha256": file_sha256(args.selected_prefixes),
            "validated_frontier_contract_sha256": file_sha256(args.validated_frontier_contract),
            "wave1_design_manifest_sha256": file_sha256(args.wave1_manifest),
            "wave1_model_contract_sha256": file_sha256(args.model_contract),
            "wave2_contract_sha256": file_sha256(args.wave2_contract),
        },
        "model_selection": {
            "eligible_settings": (
                response.eligible_parameters(parameter_metrics)[
                    ["feature_kind", "alpha", "rmse_bpb", "fold_rmse_se_bpb", "gain_sign_reversals"]
                ].to_dict(orient="records")
            ),
            "model_representatives": [{"feature_kind": model.feature_kind, "alpha": model.alpha} for model in models],
            "sealed_referees_used": False,
        },
    }
    return summary, weights, manifest


def main() -> None:
    args = parse_args()
    summary, weights, manifest = build_design(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    write_bytes_exact(summary_path, summary.to_csv(index=False).encode())
    write_bytes_exact(
        weights_path,
        weights.loc[:, list(design.WEIGHT_ARTIFACT_COLUMNS)].to_csv(index=False).encode(),
    )
    payload = {
        **manifest,
        "artifacts": {
            summary_path.name: file_sha256(summary_path),
            weights_path.name: file_sha256(weights_path),
        },
    }
    write_bytes_exact(
        args.output_dir / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
