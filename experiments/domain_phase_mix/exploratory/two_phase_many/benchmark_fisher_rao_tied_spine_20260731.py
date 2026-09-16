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
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit a tied-identified Fisher-Rao aggregate spine at 300M.

The candidate is fit only to physically tied policies:

    A(w) = c - sum_i a_i sqrt(w_i) + h R(w),  a_i >= 0, h >= 0,
    R(w) = sum_i p_i max((c0_i + c1_i) w_i - 1, 0).

It is equivalent to a bounded Bhattacharyya-overlap deficit with
``M = ||a||_2`` and ``q_i = a_i^2 / M^2``. The proportional antithetic
intervention pairs are reserved for external odd/even shape falsification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import lsq_linear

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_phase_blind_rpl_tied_spine_20260731 as tied_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_physical_hpr_tied_spine_20260731 as physical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_unique_evidence_demand_allocation_20260731 as intervention,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "fisher_rao_tied_spine_20260731"
CANDIDATE_ID = "WSD80-SUR-067"
MODEL_ID = "fisher_rao_tied_spine"
PROTOCOL_VERSION = "fisher-rao-tied-spine-v1"
TARGETS = ("uncheatable", "table9")
RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
OUTER_SPLITS = 5
INNER_SPLITS = 3
OUTER_SEED = 7_316_701
INNER_SEED_BASE = 7_316_710
FULL_FIT_SEED = 7_316_720
PAIR_BOOTSTRAP_DRAWS = 4_000
PAIR_BOOTSTRAP_SEED = 7_316_730
OPTIMUM_BOOTSTRAP_REPLICATES = 100
OPTIMUM_BOOTSTRAP_SEED = 7_316_740
ACTIVE_TOLERANCE = 1e-10
ZERO_WEIGHT_TOLERANCE = 1e-7
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

GATES = {
    "tied_oof_relative_rmse_max": 0.05,
    "pair_odd_rmse_improvement_min": 0.20,
    "pair_even_rmse_improvement_min": 0.05,
    "pair_even_rmse_improvement_ci_low_min": 0.0,
    "pair_even_spearman_min": 0.25,
    "pair_even_sign_accuracy_min": 0.60,
    "predicted_gain_beyond_observed_frontier_max": 0.02,
    "nearest_aggregate_tv_max": 0.20,
    "maximum_weight_support_slack": 0.05,
    "maximum_epoch_support_ratio": 1.25,
    "fold_optimum_median_l1_max": 0.25,
    "fold_optimum_maximum_l1_max": 0.75,
    "bootstrap_optimum_median_l1_max": 0.25,
    "bootstrap_optimum_maximum_l1_max": 0.75,
    "cross_fold_demand_cosine_median_min": 0.80,
}


@dataclass(frozen=True)
class Fitted:
    """One target-specific convex aggregate response."""

    intercept: float
    amplitudes: np.ndarray
    replay_coefficient: float
    ridge: float
    feature_scale: np.ndarray
    effective_df: float
    c_total: np.ndarray
    proportional: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = aggregate_features(weights, self.c_total, self.proportional)
        coefficients = np.concatenate([self.amplitudes, [self.replay_coefficient]])
        return self.intercept + design @ coefficients

    def demand(self) -> np.ndarray:
        squared = self.amplitudes**2
        total = float(squared.sum())
        return squared / total if total > 1e-20 else np.zeros_like(squared)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=OPTIMUM_BOOTSTRAP_REPLICATES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return baseline.json_ready(value)


def source_hash(path: Path) -> str:
    return baseline.file_hash(path)


def protocol_payload(bootstrap_replicates: int) -> dict[str, Any]:
    sources = (
        Path(__file__),
        Path(expanded.__file__),
        Path(baseline.__file__),
        Path(tied_audit.__file__),
        Path(physical.__file__),
        Path(intervention.__file__),
        expanded.PACKET,
        expanded.ONE_PHASE_SOURCE,
        intervention.transfer.MANIFEST_PATH,
        intervention.transfer.UNCHEATABLE_INTERVENTIONS_PATH,
        intervention.transfer.UNCHEATABLE_CONTROLS_PATH,
        intervention.transfer.table9.OLMO_FULL_WIDE,
    )
    payload: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "version": PROTOCOL_VERSION,
        "equation": "A(w)=c-sum_i a_i sqrt(w_i)+h sum_i p_i max(E_i(w)-1,0)",
        "constraints": "a_i>=0; h>=0; intercept c unconstrained",
        "derived_parameters": "M=||a||_2; q_i=a_i^2/M^2; b=c-M",
        "units": {
            "weights_epochs_overlap_replay": "dimensionless",
            "intercept_amplitudes_replay_coefficient": "BPB",
        },
        "fit_data": "282 physically tied 300M policies only",
        "external_shape_data": "39 proportional antithetic log-tilt pairs and 11 fresh controls",
        "excluded_fit_data": "all asymmetric policies and all proportional intervention outcomes",
        "outer_folds": {"count": OUTER_SPLITS, "seed": OUTER_SEED, "group": "phase_correspondence_key"},
        "inner_folds": {"count": INNER_SPLITS, "seed_base": INNER_SEED_BASE},
        "ridge_grid": RIDGE_GRID,
        "ridge_selection": "minimum inner correspondence-grouped RMSE; numeric ties prefer smaller ridge",
        "pair_bootstrap": {"draws": PAIR_BOOTSTRAP_DRAWS, "seed": PAIR_BOOTSTRAP_SEED},
        "optimum_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": OPTIMUM_BOOTSTRAP_SEED,
            "unit": "phase_correspondence_key",
            "ridge_reselected": False,
        },
        "frozen_reference_rmse": physical.FROZEN_REFERENCE_RMSE,
        "gates": GATES,
        "decision": (
            "Both targets must pass tied OOF, external odd/even shape, raw optimum, and stability gates. "
            "A pass licenses temporal-state work but does not promote a full two-phase surrogate."
        ),
        "source_hashes": {str(path.relative_to(REPO_ROOT)): source_hash(path) for path in sources},
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def tied_panel(target: str) -> baseline.pooled.Dataset:
    dataset, _family_index = tied_audit.tied_panel(target)
    return dataset


def proportional_policy(c0: np.ndarray) -> np.ndarray:
    inverse = 1.0 / c0
    return inverse / inverse.sum()


def repeated_mass(
    weights: np.ndarray,
    c_total: np.ndarray,
    proportional: np.ndarray,
) -> np.ndarray:
    epochs = weights * c_total[None, :]
    return (proportional[None, :] * np.maximum(epochs - 1.0, 0.0)).sum(axis=1)


def aggregate_features(
    weights: np.ndarray,
    c_total: np.ndarray,
    proportional: np.ndarray,
) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(c_total):
        raise ValueError("Expected a [rows, domains] tied-mixture matrix")
    if np.any(values < -1e-10) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Invalid tied mixture")
    clipped = np.maximum(values, 0.0)
    return np.column_stack([-np.sqrt(clipped), repeated_mass(clipped, c_total, proportional)])


def solve_model(
    weights: np.ndarray,
    target: np.ndarray,
    c_total: np.ndarray,
    proportional: np.ndarray,
    ridge: float,
) -> Fitted:
    design = aggregate_features(weights, c_total, proportional)
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-12)
    scaled = design / scale[None, :]
    observed_design = np.column_stack([np.ones(len(target)), scaled])
    if ridge > 0.0:
        penalty = np.column_stack([np.zeros(design.shape[1]), np.sqrt(ridge) * np.eye(design.shape[1])])
        fitted_design = np.vstack([observed_design, penalty])
        fitted_target = np.concatenate([target, np.zeros(design.shape[1])])
    else:
        fitted_design = observed_design
        fitted_target = target
    lower = np.concatenate([[-np.inf], np.zeros(design.shape[1])])
    upper = np.full(design.shape[1] + 1, np.inf)
    result = lsq_linear(
        fitted_design,
        fitted_target,
        bounds=(lower, upper),
        method="trf",
        max_iter=5_000,
        tol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"Constrained aggregate fit failed: {result.message}")
    coefficients = result.x[1:] / scale
    active = result.x[1:] > ACTIVE_TOLERANCE
    if np.any(active):
        centered = scaled[:, active] - scaled[:, active].mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        effective_df = 1.0 + float(np.sum(singular_values**2 / (singular_values**2 + ridge)))
    else:
        effective_df = 1.0
    return Fitted(
        intercept=float(result.x[0]),
        amplitudes=np.asarray(coefficients[:-1], dtype=float),
        replay_coefficient=float(coefficients[-1]),
        ridge=ridge,
        feature_scale=scale,
        effective_df=effective_df,
        c_total=np.asarray(c_total, dtype=float),
        proportional=np.asarray(proportional, dtype=float),
    )


def select_ridge(
    dataset: baseline.pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, pd.DataFrame]:
    weights = dataset.weights[:, 0, :]
    proportional = proportional_policy(dataset.c0)
    rows: list[dict[str, float | bool]] = []
    for ridge in RIDGE_GRID:
        prediction = np.full(dataset.n, np.nan)
        for train, test in folds:
            model = solve_model(
                weights[train],
                dataset.y[train],
                dataset.c0 + dataset.c1,
                proportional,
                ridge,
            )
            prediction[test] = model.predict(weights[test])
        if not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete inner-fold prediction")
        rmse = float(np.sqrt(np.mean((prediction - dataset.y) ** 2)))
        rows.append({"ridge": ridge, "inner_rmse": rmse, "selected": False})
    best = min(rows, key=lambda row: (float(row["inner_rmse"]), float(row["ridge"])))
    best["selected"] = True
    return float(best["ridge"]), pd.DataFrame(rows)


def fit_nested(
    dataset: baseline.pooled.Dataset,
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, list[Fitted], pd.DataFrame]:
    prediction = np.full(dataset.n, np.nan)
    models: list[Fitted] = []
    selection_rows: list[pd.DataFrame] = []
    for fold_id, (train, test) in enumerate(outer_folds):
        local = baseline.subset_dataset(dataset, train, f"outer_{fold_id}_train")
        inner = baseline.correspondence_folds(local.frame, INNER_SEED_BASE + fold_id, INNER_SPLITS)
        ridge, selection = select_ridge(local, inner)
        model = solve_model(
            local.weights[:, 0, :],
            local.y,
            local.c0 + local.c1,
            proportional_policy(local.c0),
            ridge,
        )
        prediction[test] = model.predict(dataset.weights[test, 0, :])
        models.append(model)
        selection.insert(0, "outer_fold", fold_id)
        selection_rows.append(selection)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete outer-fold prediction")
    return prediction, models, pd.concat(selection_rows, ignore_index=True)


def parameter_record(model: Fitted, domains: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    magnitude = float(np.linalg.norm(model.amplitudes))
    demand = model.demand()
    payload = {
        "intercept_c_bpb": model.intercept,
        "overlap_magnitude_M_bpb": magnitude,
        "lower_bound_b_bpb": model.intercept - magnitude,
        "replay_coefficient_h_bpb": model.replay_coefficient,
        "ridge": model.ridge,
        "nominal_parameter_count": len(model.amplitudes) + 2,
        "active_parameter_count": (
            int(np.sum(model.amplitudes > ACTIVE_TOLERANCE)) + int(model.replay_coefficient > ACTIVE_TOLERANCE) + 1
        ),
        "effective_df_active_set": model.effective_df,
        "active_demand_buckets": int(np.sum(demand > ACTIVE_TOLERANCE)),
        "maximum_demand_weight": float(np.max(demand)),
        "demand_entropy": float(stats.entropy(demand + 1e-30)),
    }
    frame = pd.DataFrame(
        {
            "domain": domains,
            "amplitude_bpb": model.amplitudes,
            "demand_weight": demand,
        }
    )
    return payload, frame


def metric_summary(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> dict[str, float | int]:
    return tied_audit.metric_summary(observed, predicted, folds)


def pairwise_cosine(vectors: list[np.ndarray]) -> dict[str, float]:
    values = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            denominator = float(np.linalg.norm(vectors[left]) * np.linalg.norm(vectors[right]))
            values.append(float(np.dot(vectors[left], vectors[right]) / denominator) if denominator > 1e-20 else 0.0)
    array = np.asarray(values, dtype=float)
    return {
        "cross_fold_demand_cosine_median": float(np.median(array)),
        "cross_fold_demand_cosine_minimum": float(np.min(array)),
    }


def optimize_raw(model: Fitted) -> tuple[np.ndarray, float]:
    domains = len(model.amplitudes)
    weights = cp.Variable(domains, nonneg=True)
    replay = cp.sum(
        cp.multiply(
            model.proportional,
            cp.pos(cp.multiply(model.c_total, weights) - 1.0),
        )
    )
    objective = cp.Minimize(
        model.intercept - cp.sum(cp.multiply(model.amplitudes, cp.sqrt(weights))) + model.replay_coefficient * replay
    )
    problem = cp.Problem(objective, [cp.sum(weights) == 1.0])
    if not problem.is_dcp():
        raise ValueError("Raw Fisher-Rao optimum is not DCP")
    problem.solve(solver="CLARABEL", max_iter=2_000)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise RuntimeError(f"Convex raw optimization failed: {problem.status}")
    optimum = np.maximum(np.asarray(weights.value, dtype=float), 0.0)
    optimum /= optimum.sum()
    prediction = float(model.predict(optimum[None, :])[0])
    return optimum, prediction


def optimum_record(
    model: Fitted,
    dataset: baseline.pooled.Dataset,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    optimum, prediction = optimize_raw(model)
    observed_best_index = int(np.argmin(dataset.y))
    observed_best = dataset.weights[observed_best_index, 0, :]
    observed_epochs = dataset.weights[:, 0, :] * (dataset.c0 + dataset.c1)
    epochs = optimum * (dataset.c0 + dataset.c1)
    beta0 = float(np.median(dataset.c0 / (dataset.c0 + dataset.c1)))
    support = baseline.support_distances(np.stack([optimum, optimum]), dataset.weights, beta0)
    diagnostics: dict[str, float | int] = {
        "predicted_bpb": prediction,
        "observed_best_tied_bpb": float(dataset.y[observed_best_index]),
        "predicted_gain_beyond_observed_frontier": float(dataset.y[observed_best_index] - prediction),
        "l1_to_observed_best": float(np.abs(optimum - observed_best).sum()),
        "maximum_bucket_weight": float(np.max(optimum)),
        "observed_maximum_bucket_weight": float(np.max(dataset.weights[:, 0, :])),
        "maximum_materialized_epochs": float(np.max(epochs)),
        "observed_maximum_materialized_epochs": float(np.max(observed_epochs)),
        "near_zero_bucket_count": int(np.sum(optimum <= ZERO_WEIGHT_TOLERANCE)),
        **support,
    }
    policy = pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "weight": optimum,
            "materialized_epochs": epochs,
            "observed_best_weight": observed_best,
        }
    )
    return diagnostics, policy


def bootstrap_indices(frame: pd.DataFrame, generator: np.random.Generator) -> np.ndarray:
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    unique = np.unique(groups)
    sampled = generator.choice(unique, size=len(unique), replace=True)
    rows = {group: np.flatnonzero(groups == group) for group in unique}
    return np.concatenate([rows[group] for group in sampled])


def bootstrap_optima(
    dataset: baseline.pooled.Dataset,
    template: Fitted,
    full_optimum: np.ndarray,
    replicates: int,
) -> pd.DataFrame:
    generator = np.random.default_rng(OPTIMUM_BOOTSTRAP_SEED)
    rows: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        indices = bootstrap_indices(dataset.frame, generator)
        sampled = baseline.subset_dataset(dataset, indices, f"bootstrap_{replicate}")
        model = solve_model(
            sampled.weights[:, 0, :],
            sampled.y,
            sampled.c0 + sampled.c1,
            proportional_policy(sampled.c0),
            template.ridge,
        )
        optimum, prediction = optimize_raw(model)
        rows.append(
            {
                "replicate": replicate,
                "predicted_bpb": prediction,
                "l1_to_full_optimum": float(np.abs(optimum - full_optimum).sum()),
                "maximum_bucket_weight": float(np.max(optimum)),
                "near_zero_bucket_count": int(np.sum(optimum <= ZERO_WEIGHT_TOLERANCE)),
                **{f"weight_{domain}": value for domain, value in zip(dataset.domain_names, optimum, strict=True)},
            }
        )
    return pd.DataFrame(rows)


def external_pair_metrics(
    model: Fitted,
    target: str,
    domains: list[str],
    manifest: pd.DataFrame,
    effect: intervention.TargetEffects,
) -> tuple[dict[str, float], pd.DataFrame]:
    plus, minus = intervention.paired_rows(manifest, domains)
    columns = [f"phase_0_{domain}" for domain in domains]
    plus_prediction = model.predict(plus[columns].to_numpy(float))
    minus_prediction = model.predict(minus[columns].to_numpy(float))
    anchor_prediction = float(model.predict(model.proportional[None, :])[0])
    predicted_odd = 0.5 * (plus_prediction - minus_prediction)
    predicted_even = 0.5 * (plus_prediction + minus_prediction) - anchor_prediction
    odd = intervention.effect_metrics(effect.odd, predicted_odd)
    even = intervention.effect_metrics(effect.even, predicted_even)
    generator = np.random.default_rng(PAIR_BOOTSTRAP_SEED + TARGETS.index(target))
    intervals = intervention.metric_bootstrap(effect.even, predicted_even, generator)
    metrics = {
        "pair_odd_rmse": odd["rmse"],
        "pair_odd_null_rmse": odd["null_rmse"],
        "pair_odd_rmse_improvement": odd["rmse_improvement"],
        "pair_odd_spearman": odd["spearman"],
        "pair_odd_sign_accuracy": odd["sign_accuracy"],
        "pair_even_rmse": even["rmse"],
        "pair_even_null_rmse": even["null_rmse"],
        "pair_even_rmse_improvement": even["rmse_improvement"],
        "pair_even_rmse_improvement_ci_low": intervals["rmse_improvement_ci"][0],
        "pair_even_rmse_improvement_ci_high": intervals["rmse_improvement_ci"][1],
        "pair_even_spearman": even["spearman"],
        "pair_even_sign_accuracy": even["sign_accuracy"],
        "pair_even_bias": even["bias"],
        "pair_even_observed_on_predicted_slope": even["observed_on_predicted_slope"],
        "external_anchor_observed_bpb": effect.anchor,
        "external_anchor_prediction_bpb": anchor_prediction,
    }
    frame = pd.DataFrame(
        {
            "domain": domains,
            "observed_odd": effect.odd,
            "predicted_odd": predicted_odd,
            "observed_even": effect.even,
            "predicted_even": predicted_even,
            "plus_observed": effect.plus,
            "plus_predicted": plus_prediction,
            "minus_observed": effect.minus,
            "minus_predicted": minus_prediction,
        }
    )
    return metrics, frame


def target_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "tied_oof": metrics["relative_rmse_to_reference"] <= GATES["tied_oof_relative_rmse_max"],
        "pair_odd": metrics["pair_odd_rmse_improvement"] >= GATES["pair_odd_rmse_improvement_min"],
        "pair_even_improvement": metrics["pair_even_rmse_improvement"] >= GATES["pair_even_rmse_improvement_min"],
        "pair_even_uncertainty": (
            metrics["pair_even_rmse_improvement_ci_low"] >= GATES["pair_even_rmse_improvement_ci_low_min"]
        ),
        "pair_even_spearman": metrics["pair_even_spearman"] >= GATES["pair_even_spearman_min"],
        "pair_even_sign": metrics["pair_even_sign_accuracy"] >= GATES["pair_even_sign_accuracy_min"],
        "frontier_optimism": (
            metrics["predicted_gain_beyond_observed_frontier"] <= GATES["predicted_gain_beyond_observed_frontier_max"]
        ),
        "support_tv": metrics["nearest_aggregate_tv"] <= GATES["nearest_aggregate_tv_max"],
        "support_weight": (
            metrics["maximum_bucket_weight"]
            <= metrics["observed_maximum_bucket_weight"] + GATES["maximum_weight_support_slack"]
        ),
        "support_epochs": (
            metrics["maximum_materialized_epochs"]
            <= GATES["maximum_epoch_support_ratio"] * metrics["observed_maximum_materialized_epochs"]
        ),
        "fold_median": metrics["fold_optimum_median_l1"] <= GATES["fold_optimum_median_l1_max"],
        "fold_maximum": metrics["fold_optimum_maximum_l1"] <= GATES["fold_optimum_maximum_l1_max"],
        "bootstrap_median": metrics["bootstrap_optimum_median_l1"] <= GATES["bootstrap_optimum_median_l1_max"],
        "bootstrap_maximum": metrics["bootstrap_optimum_maximum_l1"] <= GATES["bootstrap_optimum_maximum_l1_max"],
        "demand_stability": metrics["cross_fold_demand_cosine_median"] >= GATES["cross_fold_demand_cosine_median_min"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def preflight_payload() -> dict[str, Any]:
    manifest, domains, _c0, _c1, proportional = intervention.geometry()
    plus, minus = intervention.paired_rows(manifest, domains)
    columns = [f"phase_0_{domain}" for domain in domains]
    plus_weights = plus[columns].to_numpy(float)
    minus_weights = minus[columns].to_numpy(float)
    target_rows: dict[str, Any] = {}
    for target in TARGETS:
        dataset = tied_panel(target)
        tied_weights = dataset.weights[:, 0, :]
        plus_distance = np.max(np.abs(plus_weights[:, None, :] - tied_weights[None, :, :]), axis=2)
        minus_distance = np.max(np.abs(minus_weights[:, None, :] - tied_weights[None, :, :]), axis=2)
        anchor_distance = np.max(np.abs(proportional[None, :] - tied_weights), axis=1)
        target_rows[target] = {
            "tied_rows": dataset.n,
            "correspondence_groups": int(dataset.frame["phase_correspondence_key"].nunique()),
            "external_plus_coordinate_overlaps": int(np.sum(np.min(plus_distance, axis=1) <= 1e-12)),
            "external_minus_coordinate_overlaps": int(np.sum(np.min(minus_distance, axis=1) <= 1e-12)),
            "external_anchor_coordinate_overlaps": int(np.sum(anchor_distance <= 1e-12)),
            "domain_order_matches": list(dataset.domain_names) == domains,
        }
    return {
        "targets": target_rows,
        "external_pairs": len(domains),
        "external_controls": 11,
        "fit_uses_asymmetric_rows": False,
        "external_outcomes_select_parameters": False,
        "nominal_parameter_count": len(domains) + 2,
        "raw_problem_is_convex": True,
    }


def freeze_protocol(output_dir: Path, bootstrap_replicates: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload(bootstrap_replicates)
    path = output_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json_ready(payload):
        raise ValueError(f"Frozen protocol differs from current source: {path}")
    if not path.exists():
        baseline.write_json(path, payload)
    preflight = preflight_payload()
    baseline.write_json(output_dir / "preflight.json", preflight)
    print(json.dumps(json_ready({"protocol": payload, "preflight": preflight}), indent=2, sort_keys=True))


def verify_protocol(output_dir: Path, bootstrap_replicates: int) -> dict[str, Any]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the protocol before evaluation: {path}")
    frozen = json.loads(path.read_text())
    current = json_ready(protocol_payload(bootstrap_replicates))
    if frozen != current:
        raise ValueError("Current source, data, or settings differ from the frozen protocol")
    return frozen


def target_complete(path: Path, protocol_hash: str) -> bool:
    marker = path / "complete.json"
    if not marker.exists():
        return False
    return json.loads(marker.read_text()).get("protocol_sha256") == protocol_hash


def run_target(
    output_dir: Path,
    protocol: dict[str, Any],
    target: str,
    bootstrap_replicates: int,
    force: bool,
) -> None:
    path = output_dir / "cells" / target
    if not force and target_complete(path, str(protocol["protocol_sha256"])):
        print(f"skip complete {target}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    dataset = tied_panel(target)
    outer_folds = baseline.correspondence_folds(dataset.frame, OUTER_SEED, OUTER_SPLITS)
    oof_prediction, fold_models, selections = fit_nested(dataset, outer_folds)
    selections.to_csv(path / "ridge_selection.csv", index=False)
    predictions = dataset.frame[["run_name", "phase_correspondence_key", "policy_family", "source_panel"]].copy()
    predictions["observed"] = dataset.y
    predictions["predicted"] = oof_prediction
    predictions["residual"] = oof_prediction - dataset.y
    predictions["optimism"] = dataset.y - oof_prediction
    predictions.to_csv(path / "oof_predictions.csv", index=False)

    full_inner = baseline.correspondence_folds(dataset.frame, FULL_FIT_SEED, INNER_SPLITS)
    selected_ridge, full_selection = select_ridge(dataset, full_inner)
    full_selection.to_csv(path / "full_ridge_selection.csv", index=False)
    full_model = solve_model(
        dataset.weights[:, 0, :],
        dataset.y,
        dataset.c0 + dataset.c1,
        proportional_policy(dataset.c0),
        selected_ridge,
    )
    parameter_payload, parameters = parameter_record(full_model, dataset.domain_names)
    parameters.to_csv(path / "parameters.csv", index=False)

    manifest, domains, _c0, _c1, _proportional = intervention.geometry()
    if domains != list(dataset.domain_names):
        raise ValueError("Intervention and tied-panel domain order differ")
    effects = intervention.target_effects(manifest, domains)
    pair_metrics, pair_predictions = external_pair_metrics(full_model, target, domains, manifest, effects[target])
    pair_predictions.to_csv(path / "external_pair_predictions.csv", index=False)

    optimum_metrics, optimum_policy = optimum_record(full_model, dataset)
    optimum_policy.to_csv(path / "raw_optimum_policy.csv", index=False)
    fold_optimum_rows = []
    for fold, model in enumerate(fold_models):
        optimum, prediction = optimize_raw(model)
        fold_optimum_rows.append(
            {
                "fold": fold,
                "predicted_bpb": prediction,
                **{f"weight_{domain}": value for domain, value in zip(dataset.domain_names, optimum, strict=True)},
            }
        )
    fold_optima = pd.DataFrame(fold_optimum_rows)
    weight_columns = [f"weight_{domain}" for domain in dataset.domain_names]
    full_weights = optimum_policy["weight"].to_numpy(float)
    fold_optima["l1_to_full_optimum"] = np.abs(fold_optima[weight_columns].to_numpy(float) - full_weights).sum(axis=1)
    fold_optima.to_csv(path / "fold_optima.csv", index=False)

    bootstraps = bootstrap_optima(dataset, full_model, full_weights, bootstrap_replicates)
    bootstraps.to_csv(path / "bootstrap_optima.csv", index=False)
    demand_stability = pairwise_cosine([model.demand() for model in fold_models])
    metrics: dict[str, Any] = {
        "target": target,
        "model": MODEL_ID,
        "tied_rows": dataset.n,
        **metric_summary(dataset.y, oof_prediction, outer_folds),
        **pair_metrics,
        **optimum_metrics,
        **demand_stability,
        "fold_optimum_median_l1": float(fold_optima["l1_to_full_optimum"].median()),
        "fold_optimum_maximum_l1": float(fold_optima["l1_to_full_optimum"].max()),
        "bootstrap_optimum_median_l1": float(bootstraps["l1_to_full_optimum"].median()),
        "bootstrap_optimum_maximum_l1": float(bootstraps["l1_to_full_optimum"].max()),
        **parameter_payload,
    }
    reference = physical.FROZEN_REFERENCE_RMSE[target]
    metrics["frozen_reference_rmse"] = reference
    metrics["relative_rmse_to_reference"] = float(metrics["rmse"]) / reference - 1.0
    decision = target_decision(metrics)
    baseline.write_json(path / "metrics.json", metrics)
    baseline.write_json(path / "decision.json", decision)
    baseline.write_json(
        path / "complete.json",
        {"protocol_sha256": protocol["protocol_sha256"], "passed": decision["passed"]},
    )


def write_plots(output_dir: Path, targets: tuple[str, ...]) -> None:
    figure = make_subplots(
        rows=len(targets),
        cols=3,
        subplot_titles=tuple(
            title
            for target in targets
            for title in (f"{target}: tied OOF", f"{target}: external odd", f"{target}: external even")
        ),
    )
    for row, target in enumerate(targets, start=1):
        path = output_dir / "cells" / target
        oof = pd.read_csv(path / "oof_predictions.csv")
        pairs = pd.read_csv(path / "external_pair_predictions.csv")
        panels = (
            (oof["observed"], oof["predicted"], oof["residual"]),
            (pairs["observed_odd"], pairs["predicted_odd"], pairs["predicted_odd"] - pairs["observed_odd"]),
            (pairs["observed_even"], pairs["predicted_even"], pairs["predicted_even"] - pairs["observed_even"]),
        )
        for column, (observed, predicted, residual) in enumerate(panels, start=1):
            figure.add_trace(
                go.Scatter(
                    x=observed,
                    y=predicted,
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": residual,
                        "colorscale": "RdYlGn_r",
                        "showscale": column == 3,
                    },
                    showlegend=False,
                    hovertemplate="observed=%{x:.6f}<br>predicted=%{y:.6f}<extra></extra>",
                ),
                row=row,
                col=column,
            )
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            figure.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    line={"color": "#333", "dash": "dash"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=column,
            )
    figure.update_layout(
        title="Fisher-Rao tied aggregate: grouped OOF and external shape tests",
        template="plotly_white",
        width=1500,
        height=560 * len(targets),
    )
    figure.write_html(output_dir / "aggregate_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)

    policy_figure = make_subplots(rows=len(targets), cols=1, subplot_titles=targets, shared_xaxes=True)
    for row, target in enumerate(targets, start=1):
        policy = pd.read_csv(output_dir / "cells" / target / "raw_optimum_policy.csv")
        for column, name, color in (
            ("weight", "predicted raw optimum", "#d73027"),
            ("observed_best_weight", "observed tied frontier", "#1a9850"),
        ):
            policy_figure.add_trace(
                go.Bar(x=policy["domain"], y=policy[column], name=f"{target}: {name}", marker_color=color),
                row=row,
                col=1,
            )
    policy_figure.update_layout(
        title="Raw tied optima versus observed tied frontiers",
        template="plotly_white",
        barmode="group",
        width=1500,
        height=520 * len(targets),
    )
    policy_figure.write_html(output_dir / "raw_optimum_mixtures.html", include_plotlyjs=True, config=PLOT_CONFIG)


def collect(output_dir: Path, protocol: dict[str, Any]) -> None:
    metric_rows = []
    decisions: dict[str, Any] = {}
    for target in TARGETS:
        path = output_dir / "cells" / target
        metric_rows.append(json.loads((path / "metrics.json").read_text()))
        decisions[target] = json.loads((path / "decision.json").read_text())
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    passed = all(decision["passed"] for decision in decisions.values())
    decision = {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "passed": passed,
        "targets": decisions,
        "decision": "PASS: temporal-state work licensed" if passed else "FAIL: aggregate route rejected",
    }
    baseline.write_json(output_dir / "decision.json", decision)
    write_plots(output_dir, TARGETS)
    lines = [
        "# Tied-identified Fisher-Rao aggregate-spine audit",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        (
            "The model and ridge were fit only on 282 physically tied policies. "
            "The proportional antithetic intervention outcomes were external to fitting and selection."
        ),
        "",
        metrics.to_markdown(index=False),
        "",
        (
            "A pass licenses a separately identified temporal-state stage; "
            "it does not promote a complete two-phase surrogate."
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(json_ready(decision), indent=2, sort_keys=True))


def evaluate(output_dir: Path, bootstrap_replicates: int, force: bool) -> None:
    protocol = verify_protocol(output_dir, bootstrap_replicates)
    for target in TARGETS:
        run_target(output_dir, protocol, target, bootstrap_replicates, force)
    collect(output_dir, protocol)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    if args.mode == "prepare":
        freeze_protocol(args.output_dir, args.bootstrap_replicates)
        return
    evaluate(args.output_dir, args.bootstrap_replicates, args.force)


if __name__ == "__main__":
    main()
