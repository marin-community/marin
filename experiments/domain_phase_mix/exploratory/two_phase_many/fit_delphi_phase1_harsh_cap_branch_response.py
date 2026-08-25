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
"""Fit one state-conditioned harsh-cap Delphi phase-1 response."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import optimize, stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branch_results_20260825" / "branch_results.csv"
DEFAULT_COVERAGE = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branch_results_20260825" / "coverage.json"
DEFAULT_DESIGN_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branches_20260825"
DEFAULT_DESIGN_SUMMARY = DEFAULT_DESIGN_DIR / "continuation_summary.csv"
DEFAULT_DESIGN_WEIGHTS = DEFAULT_DESIGN_DIR / "continuation_weights.csv"
DEFAULT_DESIGN_MANIFEST = DEFAULT_DESIGN_DIR / "manifest.json"
DEFAULT_CANDIDATE_WEIGHTS = design.DEFAULT_CANDIDATE_WEIGHTS
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branch_fit_20260825"
TARGET = "bpb"
FEATURE_KINDS = ("direct", "sqrt")
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
OUTER_FOLDS = 5
INNER_FOLDS = 4
CV_SEED = 20_260_825
MINIMUM_PROBABILITY = 0.70
FRONTIER_BPB = 0.9824552536
STABILITY_STANDARD_DEVIATIONS = 1.0


@dataclass(frozen=True)
class ResponseModel:
    feature_kind: str
    alpha: float
    coefficients: tuple[float, ...]
    damage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN_SUMMARY)
    parser.add_argument("--design-weights", type=Path, default=DEFAULT_DESIGN_WEIGHTS)
    parser.add_argument("--design-manifest", type=Path, default=DEFAULT_DESIGN_MANIFEST)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--candidate-id")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_exact(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Refusing to replace a different frozen artifact: {path}")
        return
    path.write_bytes(encoded)


def validate_sealed_input(results: pd.DataFrame, coverage_path: Path) -> dict[str, object]:
    coverage = json.loads(coverage_path.read_text())
    expected_rows = int(coverage.get("expected_rows", -1))
    sealed_rows = int(coverage.get("sealed_referee_rows", -1))
    if coverage.get("status") != "complete" or int(coverage.get("missing_rows", -1)) != 0:
        raise ValueError("Branch materialization is not complete")
    if coverage.get("referee_outcomes_opened") is not False:
        raise ValueError("Referee outcomes were opened before model freeze")
    if len(results) != expected_rows - sealed_rows:
        raise ValueError(
            f"Sealed result coverage changed: {len(results)} visible rows != {expected_rows} - {sealed_rows}"
        )
    if results.role.eq("sealed_geometry_referee").any():
        raise ValueError("Referee outcomes are present in the model input")
    return cast(dict[str, object], coverage)


def validate_frozen_inputs(
    coverage: dict[str, object],
    design_summary_path: Path,
    design_weights_path: Path,
    design_manifest_path: Path,
    candidate_weights_path: Path,
) -> None:
    observed = {
        "continuation_summary_sha256": file_sha256(design_summary_path),
        "continuation_weights_sha256": file_sha256(design_weights_path),
        "design_manifest_sha256": file_sha256(design_manifest_path),
        "candidate_weights_sha256": file_sha256(candidate_weights_path),
    }
    for key, value in observed.items():
        if coverage.get(key) != value:
            raise ValueError(f"Frozen branch input changed: {key}={value}")


def hellinger(weights: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.sqrt(weights) - np.sqrt(center), axis=1) / np.sqrt(2.0)


def feature_map(weights: np.ndarray, center: np.ndarray, kind: str) -> np.ndarray:
    if kind == "direct":
        return weights - center
    if kind == "sqrt":
        center_root = np.sqrt(center)
        displacement = np.sqrt(weights) - center_root
        return displacement - (displacement @ center_root)[:, None] * center_root
    raise ValueError(f"Unknown feature kind: {kind}")


def fit_model(
    weights: np.ndarray,
    effects: np.ndarray,
    center: np.ndarray,
    feature_kind: str,
    alpha: float,
) -> ResponseModel:
    features = feature_map(weights, center, feature_kind)
    radius2 = hellinger(weights, center) ** 2
    penalty = np.zeros((features.shape[1], features.shape[1] + 1))
    penalty[:, :-1] = np.sqrt(alpha) * np.eye(features.shape[1])
    matrix = np.vstack([np.column_stack([features, radius2]), penalty])
    target = np.concatenate([effects, np.zeros(features.shape[1])])
    lower = np.concatenate([np.full(features.shape[1], -np.inf), [0.0]])
    upper = np.full(features.shape[1] + 1, np.inf)
    solution = optimize.lsq_linear(matrix, target, bounds=(lower, upper), lsmr_tol="auto")
    if not solution.success:
        raise ValueError(f"Constrained local response fit failed: {solution.message}")
    return ResponseModel(
        feature_kind=feature_kind,
        alpha=alpha,
        coefficients=tuple(float(value) for value in solution.x[:-1]),
        damage=float(solution.x[-1]),
    )


def predict(model: ResponseModel, weights: np.ndarray, center: np.ndarray) -> np.ndarray:
    features = feature_map(weights, center, model.feature_kind)
    return features @ np.asarray(model.coefficients) + model.damage * hellinger(weights, center) ** 2


def geometric_fold_ids(weights: np.ndarray, center: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if len(weights) < folds:
        raise ValueError(f"Need at least {folds} rows, got {len(weights)}")
    features = feature_map(weights, center, "sqrt")
    generator = np.random.default_rng(seed)
    first = int(np.argmax(features @ generator.normal(size=features.shape[1])))
    centers = [first]
    while len(centers) < folds:
        distances = np.min(
            np.linalg.norm(features[:, None, :] - features[np.asarray(centers)][None, :, :], axis=2),
            axis=1,
        )
        distances[centers] = -np.inf
        centers.append(int(np.argmax(distances)))
    capacities = np.full(folds, len(weights) // folds, dtype=int)
    capacities[: len(weights) % folds] += 1
    slots = np.repeat(np.arange(folds), capacities)
    costs = np.linalg.norm(features[:, None, :] - features[np.asarray(centers)][slots][None, :, :], axis=2)
    row_indices, slot_indices = optimize.linear_sum_assignment(costs)
    labels = np.empty(len(weights), dtype=int)
    labels[row_indices] = slots[slot_indices]
    if set(labels) != set(range(folds)):
        raise ValueError("Geometric fold construction lost a held-out region")
    if not np.array_equal(np.bincount(labels, minlength=folds), capacities):
        raise ValueError("Geometric fold construction violated balanced capacities")
    return labels


def parameter_cv(
    weights: np.ndarray,
    effects: np.ndarray,
    center: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for feature_kind in FEATURE_KINDS:
        for alpha in RIDGE_ALPHAS:
            predictions = np.empty(len(effects))
            for fold in sorted(set(folds)):
                train = folds != fold
                heldout = folds == fold
                model = fit_model(weights[train], effects[train], center, feature_kind, alpha)
                predictions[heldout] = predict(model, weights[heldout], center)
            residual = predictions - effects
            rows.append(
                {
                    "feature_kind": feature_kind,
                    "alpha": alpha,
                    "rmse_bpb": float(np.sqrt(np.mean(residual**2))),
                    "spearman": float(stats.spearmanr(predictions, effects).statistic),
                    "sign_accuracy": float(np.mean(np.sign(predictions) == np.sign(effects))),
                }
            )
    return pd.DataFrame(rows).sort_values(["rmse_bpb", "feature_kind", "alpha"]).reset_index(drop=True)


def selected_parameter(metrics: pd.DataFrame) -> tuple[str, float]:
    row = metrics.sort_values(["rmse_bpb", "feature_kind", "alpha"]).iloc[0]
    return str(row.feature_kind), float(row.alpha)


def nested_crossfit(
    weights: np.ndarray,
    effects: np.ndarray,
    center: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = geometric_fold_ids(weights, center, OUTER_FOLDS, CV_SEED)
    predictions = np.empty(len(effects))
    selections = []
    for fold in range(OUTER_FOLDS):
        train_indices = np.flatnonzero(outer != fold)
        heldout = outer == fold
        inner = geometric_fold_ids(weights[train_indices], center, INNER_FOLDS, CV_SEED + fold + 1)
        inner_metrics = parameter_cv(weights[train_indices], effects[train_indices], center, inner)
        feature_kind, alpha = selected_parameter(inner_metrics)
        model = fit_model(weights[train_indices], effects[train_indices], center, feature_kind, alpha)
        predictions[heldout] = predict(model, weights[heldout], center)
        selections.append(
            {
                "outer_fold": fold,
                "feature_kind": feature_kind,
                "alpha": alpha,
                "inner_rmse_bpb": float(inner_metrics.iloc[0].rmse_bpb),
            }
        )
    prediction_frame = pd.DataFrame(
        {
            "row": np.arange(len(effects)),
            "outer_fold": outer,
            "observed_effect_bpb": effects,
            "predicted_effect_bpb": predictions,
            "residual_bpb": predictions - effects,
        }
    )
    return prediction_frame, pd.DataFrame(selections)


def fold_ensemble_predictions(
    weights: np.ndarray,
    effects: np.ndarray,
    center: np.ndarray,
    candidate_weights: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    outer = geometric_fold_ids(weights, center, OUTER_FOLDS, CV_SEED)
    predictions = []
    selections = []
    for fold in range(OUTER_FOLDS):
        train_indices = np.flatnonzero(outer != fold)
        inner = geometric_fold_ids(weights[train_indices], center, INNER_FOLDS, CV_SEED + fold + 1)
        inner_metrics = parameter_cv(weights[train_indices], effects[train_indices], center, inner)
        feature_kind, alpha = selected_parameter(inner_metrics)
        model = fit_model(weights[train_indices], effects[train_indices], center, feature_kind, alpha)
        predictions.append(predict(model, candidate_weights, center))
        selections.append(
            {
                "outer_fold": fold,
                "feature_kind": feature_kind,
                "alpha": alpha,
                "inner_rmse_bpb": float(inner_metrics.iloc[0].rmse_bpb),
            }
        )
    return np.stack(predictions), pd.DataFrame(selections)


def load_weights(
    design_weights_path: Path,
    candidate_id: str,
    continuation_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    frame = pd.read_csv(design_weights_path)
    frame = frame[frame.prefix_candidate_id.eq(candidate_id)]
    first = frame[frame.continuation_id.eq(continuation_ids[0])]
    buckets = tuple(first.bucket)
    matrices = []
    for continuation_id in continuation_ids:
        rows = frame[frame.continuation_id.eq(continuation_id)]
        if tuple(rows.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {continuation_id}")
        matrices.append(rows.phase_1_weight.to_numpy(dtype=float))
    return buckets, np.stack(matrices)


def tied_center(candidate_weights_path: Path, candidate_id: str, buckets: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(candidate_weights_path)
    rows = frame[frame.candidate_id.eq(candidate_id)]
    if tuple(rows.bucket) != buckets:
        raise ValueError(f"Candidate bucket order changed for {candidate_id}")
    return cast(np.ndarray, rows.phase_0_weight.to_numpy(dtype=float))


def control_baselines(results: pd.DataFrame) -> dict[str, float | int]:
    matched = results[results.role.eq("common_random_tied_control")]
    expected = results[
        results.prefix_repeat_seed.eq(0) & results.role.isin(("common_random_tied_control", "fresh_tied_control"))
    ]
    stability = results[results.role.eq("prefix_state_tied_control")]
    if len(matched) != 1 or len(expected) != 4 or len(stability) != 4:
        raise ValueError(
            f"Tied controls are incomplete: matched={len(matched)}, expected={len(expected)}, stability={len(stability)}"
        )
    return {
        "matched_tied_bpb": float(matched[TARGET].iloc[0]),
        "expected_tied_bpb": float(expected[TARGET].mean()),
        "expected_tied_sd_bpb": float(expected[TARGET].std(ddof=1)),
        "expected_tied_sem_bpb": float(expected[TARGET].sem(ddof=1)),
        "stability_tied_bpb": float(stability[TARGET].mean()),
        "stability_minus_primary_bpb": float(stability[TARGET].mean() - expected[TARGET].mean()),
    }


def candidate_pool(
    center: np.ndarray,
    buckets: tuple[str, ...],
    measured: np.ndarray,
    excluded_counts: set[tuple[int, ...]],
) -> tuple[np.ndarray, list[str]]:
    panel = design.common_design.load_canonical_panel_geometry()
    if panel.buckets != buckets:
        raise ValueError("Canonical branch geometry bucket order changed")
    anchors = design.anchor_mixtures(buckets, design.runtime_weights(panel.proportional))
    points, _ = design.generate_pool(center, anchors, center * panel.c0, panel.c1)
    pool = [point.weights for point in points.values()]
    sources = [point.source for point in points.values()]
    for index, weights in enumerate(measured):
        pool.append(weights)
        sources.append(f"measured_fit:{index:03d}")
    pool.append(center)
    sources.append("tied")
    deduplicated: dict[tuple[int, ...], tuple[np.ndarray, str]] = {}
    for weights, source in zip(pool, sources, strict=True):
        counts = tuple(design.common_design.runtime_counts(weights).tolist())
        if counts in excluded_counts:
            continue
        deduplicated.setdefault(counts, (weights, source))
    return np.stack([value[0] for value in deduplicated.values()]), [value[1] for value in deduplicated.values()]


def fit_candidate(
    results: pd.DataFrame,
    design_summary_path: Path,
    design_weights_path: Path,
    candidate_weights_path: Path,
    candidate_id: str,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    candidate_results = results[results.prefix_candidate_id.eq(candidate_id)].copy()
    if candidate_results.role.eq("sealed_geometry_referee").any():
        raise ValueError("Referee outcomes were opened before model freeze")
    fit_rows = candidate_results[candidate_results.fit_budget.astype(bool)].sort_values("run_order")
    if len(fit_rows) != design.FIT_ROWS_PER_PREFIX:
        raise ValueError(f"Expected {design.FIT_ROWS_PER_PREFIX} fit rows, got {len(fit_rows)}")
    continuation_ids = tuple(fit_rows.continuation_id)
    buckets, weights = load_weights(design_weights_path, candidate_id, continuation_ids)
    summary = pd.read_csv(design_summary_path)
    referee_ids = tuple(
        summary[
            summary.prefix_candidate_id.eq(candidate_id) & summary.role.eq("sealed_geometry_referee")
        ].continuation_id
    )
    if len(referee_ids) != design.REFEREE_ROWS_PER_PREFIX:
        raise ValueError(f"Expected {design.REFEREE_ROWS_PER_PREFIX} sealed referee coordinates, got {len(referee_ids)}")
    _, referee_weights = load_weights(design_weights_path, candidate_id, referee_ids)
    referee_counts = {tuple(design.common_design.runtime_counts(row).tolist()) for row in referee_weights}
    center = tied_center(candidate_weights_path, candidate_id, buckets)
    baselines = control_baselines(candidate_results)
    effects = fit_rows[TARGET].to_numpy(dtype=float) - float(baselines["matched_tied_bpb"])

    nested_predictions, nested_selections = nested_crossfit(weights, effects, center)
    nested_predictions.insert(0, "continuation_id", continuation_ids)
    nested_rmse = float(np.sqrt(np.mean(nested_predictions.residual_bpb**2)))
    zero_rmse = float(np.sqrt(np.mean(effects**2)))
    parameter_metrics = parameter_cv(
        weights,
        effects,
        center,
        geometric_fold_ids(weights, center, OUTER_FOLDS, CV_SEED),
    )
    feature_kind, alpha = selected_parameter(parameter_metrics)
    model = fit_model(weights, effects, center, feature_kind, alpha)

    pool, sources = candidate_pool(center, buckets, weights, referee_counts)
    predicted_effects = predict(model, pool, center)
    fold_predictions, fold_selections = fold_ensemble_predictions(weights, effects, center, pool)
    fold_mean = fold_predictions.mean(axis=0)
    fold_sd = fold_predictions.std(axis=0, ddof=1)
    stability_score = fold_mean + STABILITY_STANDARD_DEVIATIONS * fold_sd
    order = np.argsort(stability_score)
    predictive_sd = math.sqrt(nested_rmse**2 + float(baselines["expected_tied_sem_bpb"]) ** 2)
    predictions = pd.DataFrame(
        {
            "candidate_rank": np.arange(len(pool)),
            "source": np.asarray(sources)[order],
            "full_model_predicted_effect_bpb": predicted_effects[order],
            "fold_mean_predicted_effect_bpb": fold_mean[order],
            "fold_sd_predicted_effect_bpb": fold_sd[order],
            "stability_score_bpb": stability_score[order],
            "fold_fraction_predicting_improvement": np.mean(fold_predictions[:, order] < 0.0, axis=0),
            "predicted_expected_endpoint_bpb": float(baselines["expected_tied_bpb"]) + fold_mean[order],
            "hellinger_to_tied": hellinger(pool[order], center),
        }
    )
    point_index = int(np.argmin(predicted_effects))
    stable_index = int(np.argmin(stability_score))
    observed_index = int(np.argmin(effects))
    measured_keys = [tuple(design.common_design.runtime_counts(row)) for row in weights]
    pool_positions = {tuple(design.common_design.runtime_counts(row)): index for index, row in enumerate(pool)}
    observed_pool_index = pool_positions[measured_keys[observed_index]]

    role_indices = (
        ("best_observed", observed_pool_index),
        ("full_model_argmin", point_index),
        ("fold_stable_argmin", stable_index),
    )
    acquisition_by_index: dict[int, dict[str, object]] = {}
    for role, index in role_indices:
        row = acquisition_by_index.setdefault(
            index,
            {
                "roles": [],
                "source": sources[index],
                "weights": dict(zip(buckets, pool[index], strict=True)),
                "hellinger_to_tied": float(hellinger(pool[[index]], center)[0]),
                "full_model_predicted_effect_bpb": float(predicted_effects[index]),
                "fold_mean_predicted_effect_bpb": float(fold_mean[index]),
                "fold_sd_predicted_effect_bpb": float(fold_sd[index]),
                "stability_score_bpb": float(stability_score[index]),
                "fold_fraction_predicting_improvement": float(np.mean(fold_predictions[:, index] < 0.0)),
            },
        )
        cast(list[str], row["roles"]).append(role)
        if role == "best_observed":
            row["observed_effect_vs_matched_tied_bpb"] = float(effects[observed_index])
            row["observed_expected_endpoint_bpb"] = float(baselines["expected_tied_bpb"]) + float(
                effects[observed_index]
            )

    best_effect = float(fold_mean[stable_index])
    expected_endpoint = float(baselines["expected_tied_bpb"]) + best_effect
    probability_beat_tied = float(stats.norm.cdf(-best_effect / predictive_sd))
    probability_beat_frontier = float(stats.norm.cdf((FRONTIER_BPB - expected_endpoint) / predictive_sd))
    candidate: dict[str, object] = {
        "candidate_id": candidate_id,
        "target": "Uncheatable BPB",
        "feature_kind": model.feature_kind,
        "ridge_alpha": model.alpha,
        "damage_coefficient": model.damage,
        "coefficients": dict(zip(buckets, model.coefficients, strict=True)),
        "primary_acquisition_role": "fold_stable_argmin",
        "weights": dict(zip(buckets, pool[stable_index], strict=True)),
        "source": sources[stable_index],
        "hellinger_to_tied": float(hellinger(pool[[stable_index]], center)[0]),
        "predicted_effect_vs_matched_tied_bpb": best_effect,
        "predicted_expected_endpoint_bpb": expected_endpoint,
        "nested_crossfit_rmse_bpb": nested_rmse,
        "zero_effect_rmse_bpb": zero_rmse,
        "predictive_sd_bpb": predictive_sd,
        "probability_beat_expected_tied": probability_beat_tied,
        "probability_beat_frontier": probability_beat_frontier,
        "probability_note": (
            "Normal approximation using nested-CV RMSE; descriptive after candidate selection, not a confidence interval"
        ),
        "stability_rule": (
            "Minimize the mean prediction from five leave-region fits plus one standard deviation; "
            "this is an acquisition heuristic, not an uncertainty interval"
        ),
        "acquisition_shortlist": list(acquisition_by_index.values()),
        "eligible_for_measurement": (
            nested_rmse < zero_rmse
            and (
                float(effects[observed_index]) < 0.0
                or (best_effect < 0.0 and probability_beat_tied >= MINIMUM_PROBABILITY)
            )
        ),
        "baselines": baselines,
    }
    return candidate, {
        "parameter_cv": parameter_metrics,
        "nested_predictions": nested_predictions,
        "nested_selections": nested_selections,
        "fold_ensemble_selections": fold_selections,
        "candidate_predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    coverage = validate_sealed_input(results, args.coverage)
    validate_frozen_inputs(
        coverage,
        args.design_summary,
        args.design_weights,
        args.design_manifest,
        args.candidate_weights,
    )
    candidate_ids = tuple(results.prefix_candidate_id.drop_duplicates())
    if args.candidate_id is not None:
        candidate_ids = (args.candidate_id,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "contract_version": "delphi_phase1_harsh_cap_branch_response_20260825_v1",
        "target": "Uncheatable BPB",
        "candidate_ids": list(candidate_ids),
        "model_class": "local direct-or-square-root ridge plus nonnegative scalar Hellinger-squared damage",
        "selection": "nested row-level cross-validation; referee outcomes excluded",
        "frontier_bpb": FRONTIER_BPB,
        "inputs": {
            "results_sha256": file_sha256(args.results),
            "coverage_sha256": file_sha256(args.coverage),
            "design_summary_sha256": file_sha256(args.design_summary),
            "design_weights_sha256": file_sha256(args.design_weights),
            "design_manifest_sha256": file_sha256(args.design_manifest),
            "candidate_weights_sha256": file_sha256(args.candidate_weights),
        },
        "candidates": {},
        "seal": {
            "referee_outcomes_present_in_fit_input": bool(results.role.eq("sealed_geometry_referee").any()),
            "materialization_referee_outcomes_opened": coverage["referee_outcomes_opened"],
            "sealed_referee_rows": coverage["sealed_referee_rows"],
            "manifest_sha256": coverage["manifest_sha256"],
            "sealed_coordinates_excluded_from_candidate_pool": True,
        },
    }
    frozen_candidates = {}
    for candidate_id in candidate_ids:
        candidate, artifacts = fit_candidate(
            results,
            args.design_summary,
            args.design_weights,
            args.candidate_weights,
            candidate_id,
        )
        candidate_dir = args.output_dir / candidate_id
        candidate_dir.mkdir(exist_ok=True)
        write_json_exact(candidate_dir / "predicted_optimum.json", candidate)
        for name, frame in artifacts.items():
            frame.to_csv(candidate_dir / f"{name}.csv", index=False)
        status["candidates"][candidate_id] = {
            key: value for key, value in candidate.items() if key not in {"coefficients", "weights", "baselines"}
        }
        frozen_candidates[candidate_id] = candidate
    status_path = args.output_dir / "status.json"
    write_json_exact(status_path, status)
    contract = {
        **status,
        "status_sha256": file_sha256(status_path),
        "frozen_candidates": frozen_candidates,
        "referee_opening_rule": (
            "Freeze this contract and all predicted optima before explicitly opening the eight sealed outcomes."
        ),
        "referee_scoring_entrypoint": "score_delphi_phase1_harsh_cap_referees.py",
    }
    contract_path = args.output_dir / "frozen_model_contract.json"
    write_json_exact(contract_path, contract)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
