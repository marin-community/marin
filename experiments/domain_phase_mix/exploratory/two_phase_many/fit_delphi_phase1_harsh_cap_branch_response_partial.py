# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scipy==1.17.0",
# ]
# ///
"""Fit a provisional harsh-cap branch response before tied controls finish."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import optimize, stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_delphi_phase1_harsh_cap_branch_response as final_fit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_RESULTS_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branch_results_20260825"
DEFAULT_DESIGN_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap4_branches_20260825"
DEFAULT_CANDIDATE_WEIGHTS = design.DEFAULT_CANDIDATE_WEIGHTS
MINIMUM_FIT_ROWS = 20
EARLY_CONFIRMATION_ROWS = 60


@dataclass(frozen=True)
class ProvisionalResponseModel:
    feature_kind: str
    alpha: float
    intercept: float
    coefficients: tuple[float, ...]
    damage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR / "branch_results.csv")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_RESULTS_DIR / "coverage.json")
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN_DIR / "continuation_summary.csv")
    parser.add_argument("--design-weights", type=Path, default=DEFAULT_DESIGN_DIR / "continuation_weights.csv")
    parser.add_argument("--design-manifest", type=Path, default=DEFAULT_DESIGN_DIR / "manifest.json")
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fit_model(
    weights: np.ndarray,
    endpoints: np.ndarray,
    center: np.ndarray,
    feature_kind: str,
    alpha: float,
) -> ProvisionalResponseModel:
    features = final_fit.feature_map(weights, center, feature_kind)
    radius2 = final_fit.hellinger(weights, center) ** 2
    design_matrix = np.column_stack([np.ones(len(weights)), features, radius2])
    penalty = np.zeros((features.shape[1], design_matrix.shape[1]))
    penalty[:, 1:-1] = np.sqrt(alpha) * np.eye(features.shape[1])
    matrix = np.vstack([design_matrix, penalty])
    target = np.concatenate([endpoints, np.zeros(features.shape[1])])
    lower = np.concatenate([[float("-inf")], np.full(features.shape[1], -np.inf), [0.0]])
    upper = np.full(design_matrix.shape[1], np.inf)
    solution = optimize.lsq_linear(matrix, target, bounds=(lower, upper), lsmr_tol="auto")
    if not solution.success:
        raise ValueError(f"Constrained provisional response fit failed: {solution.message}")
    return ProvisionalResponseModel(
        feature_kind=feature_kind,
        alpha=alpha,
        intercept=float(solution.x[0]),
        coefficients=tuple(float(value) for value in solution.x[1:-1]),
        damage=float(solution.x[-1]),
    )


def predict(model: ProvisionalResponseModel, weights: np.ndarray, center: np.ndarray) -> np.ndarray:
    features = final_fit.feature_map(weights, center, model.feature_kind)
    return (
        model.intercept
        + features @ np.asarray(model.coefficients)
        + model.damage * final_fit.hellinger(weights, center) ** 2
    )


def parameter_cv(
    weights: np.ndarray,
    endpoints: np.ndarray,
    center: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for feature_kind in final_fit.FEATURE_KINDS:
        for alpha in final_fit.RIDGE_ALPHAS:
            predictions = np.empty(len(endpoints))
            for fold in sorted(set(folds)):
                train = folds != fold
                heldout = folds == fold
                model = fit_model(weights[train], endpoints[train], center, feature_kind, alpha)
                predictions[heldout] = predict(model, weights[heldout], center)
            residual = predictions - endpoints
            rows.append(
                {
                    "feature_kind": feature_kind,
                    "alpha": alpha,
                    "rmse_bpb": float(np.sqrt(np.mean(residual**2))),
                    "spearman": float(stats.spearmanr(predictions, endpoints).statistic),
                }
            )
    return pd.DataFrame(rows).sort_values(["rmse_bpb", "feature_kind", "alpha"]).reset_index(drop=True)


def selected_parameter(metrics: pd.DataFrame) -> tuple[str, float]:
    row = metrics.iloc[0]
    return str(row.feature_kind), float(row.alpha)


def nested_crossfit(
    weights: np.ndarray,
    endpoints: np.ndarray,
    center: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = final_fit.geometric_fold_ids(weights, center, final_fit.OUTER_FOLDS, final_fit.CV_SEED)
    predictions = np.empty(len(endpoints))
    constant_predictions = np.empty(len(endpoints))
    selections = []
    for fold in range(final_fit.OUTER_FOLDS):
        train_indices = np.flatnonzero(outer != fold)
        heldout = outer == fold
        inner = final_fit.geometric_fold_ids(
            weights[train_indices], center, final_fit.INNER_FOLDS, final_fit.CV_SEED + fold + 1
        )
        inner_metrics = parameter_cv(weights[train_indices], endpoints[train_indices], center, inner)
        feature_kind, alpha = selected_parameter(inner_metrics)
        model = fit_model(weights[train_indices], endpoints[train_indices], center, feature_kind, alpha)
        predictions[heldout] = predict(model, weights[heldout], center)
        constant_predictions[heldout] = float(np.mean(endpoints[train_indices]))
        selections.append(
            {
                "outer_fold": fold,
                "feature_kind": feature_kind,
                "alpha": alpha,
                "inner_rmse_bpb": float(inner_metrics.iloc[0].rmse_bpb),
            }
        )
    return (
        pd.DataFrame(
            {
                "row": np.arange(len(endpoints)),
                "outer_fold": outer,
                "observed_endpoint_bpb": endpoints,
                "predicted_endpoint_bpb": predictions,
                "constant_predicted_endpoint_bpb": constant_predictions,
                "residual_bpb": predictions - endpoints,
                "constant_residual_bpb": constant_predictions - endpoints,
            }
        ),
        pd.DataFrame(selections),
    )


def fold_ensemble_predictions(
    weights: np.ndarray,
    endpoints: np.ndarray,
    center: np.ndarray,
    candidate_weights: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    outer = final_fit.geometric_fold_ids(weights, center, final_fit.OUTER_FOLDS, final_fit.CV_SEED)
    predictions = []
    selections = []
    for fold in range(final_fit.OUTER_FOLDS):
        train_indices = np.flatnonzero(outer != fold)
        inner = final_fit.geometric_fold_ids(
            weights[train_indices], center, final_fit.INNER_FOLDS, final_fit.CV_SEED + fold + 1
        )
        metrics = parameter_cv(weights[train_indices], endpoints[train_indices], center, inner)
        feature_kind, alpha = selected_parameter(metrics)
        model = fit_model(weights[train_indices], endpoints[train_indices], center, feature_kind, alpha)
        predictions.append(predict(model, candidate_weights, center))
        selections.append(
            {
                "outer_fold": fold,
                "feature_kind": feature_kind,
                "alpha": alpha,
                "inner_rmse_bpb": float(metrics.iloc[0].rmse_bpb),
            }
        )
    return np.stack(predictions), pd.DataFrame(selections)


def validate_inputs(args: argparse.Namespace, results: pd.DataFrame) -> dict[str, object]:
    coverage = json.loads(args.coverage.read_text())
    if coverage.get("status") not in {"provisional_incomplete", "complete"}:
        raise ValueError(f"Unexpected materialization status: {coverage.get('status')}")
    if coverage.get("referee_outcomes_opened") is not False:
        raise ValueError("Referee outcomes were opened before provisional model freeze")
    if results.role.eq("sealed_geometry_referee").any():
        raise ValueError("Referee outcomes are present in the provisional model input")
    if len(results) != int(coverage.get("visible_result_rows", -1)):
        raise ValueError("Materialized result count and coverage disagree")
    final_fit.validate_frozen_inputs(
        coverage,
        args.design_summary,
        args.design_weights,
        args.design_manifest,
        args.candidate_weights,
    )
    return cast(dict[str, object], coverage)


def referee_counts(args: argparse.Namespace, candidate_id: str) -> set[tuple[int, ...]]:
    summary = pd.read_csv(args.design_summary)
    continuation_ids = tuple(
        summary[
            summary.prefix_candidate_id.eq(candidate_id) & summary.role.eq("sealed_geometry_referee")
        ].continuation_id
    )
    if len(continuation_ids) != design.REFEREE_ROWS_PER_PREFIX:
        raise ValueError(f"Expected {design.REFEREE_ROWS_PER_PREFIX} sealed referee coordinates")
    _, weights = final_fit.load_weights(args.design_weights, candidate_id, continuation_ids)
    return {tuple(design.common_design.runtime_counts(row).tolist()) for row in weights}


def fit_partial(args: argparse.Namespace, results: pd.DataFrame) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    fit_rows = results[results.prefix_candidate_id.eq(args.candidate_id) & results.fit_budget.astype(bool)].sort_values(
        "run_order"
    )
    if not MINIMUM_FIT_ROWS <= len(fit_rows) <= design.FIT_ROWS_PER_PREFIX:
        raise ValueError(
            f"Expected {MINIMUM_FIT_ROWS}..{design.FIT_ROWS_PER_PREFIX} completed fit rows, got {len(fit_rows)}"
        )
    if fit_rows.continuation_id.duplicated().any() or fit_rows.run_order.duplicated().any():
        raise ValueError("Provisional fit input contains duplicate branch identities")
    continuation_ids = tuple(fit_rows.continuation_id)
    buckets, weights = final_fit.load_weights(args.design_weights, args.candidate_id, continuation_ids)
    center = final_fit.tied_center(args.candidate_weights, args.candidate_id, buckets)
    endpoints = fit_rows[final_fit.TARGET].to_numpy(dtype=float)

    nested_predictions, nested_selections = nested_crossfit(weights, endpoints, center)
    nested_predictions.insert(0, "continuation_id", continuation_ids)
    nested_rmse = float(np.sqrt(np.mean(nested_predictions.residual_bpb**2)))
    constant_rmse = float(np.sqrt(np.mean(nested_predictions.constant_residual_bpb**2)))
    folds = final_fit.geometric_fold_ids(weights, center, final_fit.OUTER_FOLDS, final_fit.CV_SEED)
    metrics = parameter_cv(weights, endpoints, center, folds)
    feature_kind, alpha = selected_parameter(metrics)
    model = fit_model(weights, endpoints, center, feature_kind, alpha)

    pool, sources = final_fit.candidate_pool(center, buckets, weights, referee_counts(args, args.candidate_id))
    full_predictions = predict(model, pool, center)
    fold_predictions, fold_selections = fold_ensemble_predictions(weights, endpoints, center, pool)
    fold_mean = fold_predictions.mean(axis=0)
    fold_sd = fold_predictions.std(axis=0, ddof=1)
    stability_score = fold_mean + final_fit.STABILITY_STANDARD_DEVIATIONS * fold_sd
    order = np.argsort(stability_score)
    predictions = pd.DataFrame(
        {
            "candidate_rank": np.arange(len(pool)),
            "source": np.asarray(sources)[order],
            "full_model_predicted_endpoint_bpb": full_predictions[order],
            "fold_mean_predicted_endpoint_bpb": fold_mean[order],
            "fold_sd_predicted_endpoint_bpb": fold_sd[order],
            "stability_score_bpb": stability_score[order],
            "hellinger_to_tied": final_fit.hellinger(pool[order], center),
        }
    )

    stable_index = int(np.argmin(stability_score))
    point_index = int(np.argmin(full_predictions))
    observed_index = int(np.argmin(endpoints))
    measured_keys = [tuple(design.common_design.runtime_counts(row)) for row in weights]
    pool_positions = {tuple(design.common_design.runtime_counts(row)): index for index, row in enumerate(pool)}
    observed_pool_index = pool_positions[measured_keys[observed_index]]
    argmin_hellinger = float(final_fit.hellinger(pool[[point_index]], pool[stable_index])[0])
    shortlist = []
    for role, index in (
        ("best_observed", observed_pool_index),
        ("full_model_argmin", point_index),
        ("fold_stable_argmin", stable_index),
    ):
        shortlist.append(
            {
                "role": role,
                "source": sources[index],
                "weights": dict(zip(buckets, pool[index], strict=True)),
                "hellinger_to_tied": float(final_fit.hellinger(pool[[index]], center)[0]),
                "full_model_predicted_endpoint_bpb": float(full_predictions[index]),
                "fold_mean_predicted_endpoint_bpb": float(fold_mean[index]),
                "fold_sd_predicted_endpoint_bpb": float(fold_sd[index]),
                "observed_endpoint_bpb": float(endpoints[observed_index]) if role == "best_observed" else None,
            }
        )
    status = {
        "contract_version": "delphi_phase1_harsh_cap_partial_branch_response_20260825_v1",
        "status": "provisional_not_for_claims",
        "candidate_id": args.candidate_id,
        "target": "Uncheatable BPB",
        "fit_rows": len(fit_rows),
        "minimum_fit_rows": MINIMUM_FIT_ROWS,
        "early_confirmation_rows": EARLY_CONFIRMATION_ROWS,
        "confirmation_readiness": (
            "requires_cross-checkpoint_stability_review"
            if len(fit_rows) >= EARLY_CONFIRMATION_ROWS
            else "not_before_60_fit_rows"
        ),
        "model_class": (
            "raw endpoint BPB with unpenalized intercept, local direct-or-square-root ridge, "
            "and nonnegative scalar Hellinger-squared damage"
        ),
        "feature_kind": model.feature_kind,
        "ridge_alpha": model.alpha,
        "intercept": model.intercept,
        "damage_coefficient": model.damage,
        "nested_crossfit_rmse_bpb": nested_rmse,
        "constant_crossfit_rmse_bpb": constant_rmse,
        "nested_crossfit_spearman": float(
            stats.spearmanr(
                nested_predictions.predicted_endpoint_bpb,
                nested_predictions.observed_endpoint_bpb,
            ).statistic
        ),
        "full_vs_fold_stable_argmin_hellinger": argmin_hellinger,
        "primary_acquisition_role": "fold_stable_argmin",
        "primary_candidate": shortlist[-1],
        "shortlist": shortlist,
        "coefficients": dict(zip(buckets, model.coefficients, strict=True)),
        "caveat": (
            "Tied controls are not yet available. The intercept identifies the raw BPB level, but this provisional fit "
            "cannot estimate paired gain or support a noninferiority claim."
        ),
    }
    return status, {
        "parameter_cv": metrics,
        "nested_predictions": nested_predictions,
        "nested_selections": nested_selections,
        "fold_ensemble_selections": fold_selections,
        "candidate_predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    coverage = validate_inputs(args, results)
    status, artifacts = fit_partial(args, results)
    status["inputs"] = {
        "results_sha256": final_fit.file_sha256(args.results),
        "coverage_sha256": final_fit.file_sha256(args.coverage),
        "manifest_sha256": coverage["manifest_sha256"],
        "design_manifest_sha256": final_fit.file_sha256(args.design_manifest),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_fit.write_json_exact(args.output_dir / "provisional_status.json", status)
    for name, frame in artifacts.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
