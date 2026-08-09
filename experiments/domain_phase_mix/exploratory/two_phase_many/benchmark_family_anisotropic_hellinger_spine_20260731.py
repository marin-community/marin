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
"""Audit a family-anisotropic Hellinger aggregate spine at 300M.

The candidate is fit only to physically tied policies. In its directly
identified gauge the response is

    A(w) = c + sum_f k_f W_f - sum_i a_i sqrt(w_i) + h R(w),

where ``sum_f k_f = 0``, ``a_i >= 0``, and ``h >= 0``. The two identifiable
family-mass contrasts are not arbitrary bucketwise linear credits. When there
is a scalar ``lambda > -min_f k_f`` satisfying

    sum_i [a_i / (2 (k_f(i) + lambda))]^2 = 1,

the same response is, up to its intercept,

    sum_f K_f sum_{i in f} (sqrt(w_i) - sqrt(q_i))^2 + h R(w),

with ``K_f = k_f + lambda > 0`` and
``q_i = [a_i / (2 K_f(i))]^2``. Thus the added family contrasts are the
anisotropy of a bounded Hellinger bowl. The isotropic Fisher-Rao model is the
exact ablation ``k_f = 0``.
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
from scipy.linalg import helmert
from scipy.optimize import brentq, lsq_linear

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_fisher_rao_tied_spine_20260731 as fisher,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_physical_hpr_tied_spine_20260731 as physical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_unique_evidence_demand_allocation_20260731 as intervention,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "family_anisotropic_hellinger_spine_20260731"
CANDIDATE_ID = "WSD80-SUR-072"
MODEL_ID = "family_anisotropic_hellinger_spine"
ABLATION_ID = fisher.MODEL_ID
PROTOCOL_VERSION = "family-anisotropic-hellinger-spine-v1"
TARGETS = fisher.TARGETS
RIDGE_GRID = fisher.RIDGE_GRID
OUTER_SPLITS = fisher.OUTER_SPLITS
INNER_SPLITS = fisher.INNER_SPLITS
OUTER_SEED = 7_317_201
INNER_SEED_BASE = 7_317_210
FULL_FIT_SEED = 7_317_220
PAIR_BOOTSTRAP_DRAWS = fisher.PAIR_BOOTSTRAP_DRAWS
PAIR_BOOTSTRAP_SEED = 7_317_230
ABLATION_BOOTSTRAP_DRAWS = 4_000
ABLATION_BOOTSTRAP_SEED = 7_317_240
OPTIMUM_BOOTSTRAP_REPLICATES = fisher.OPTIMUM_BOOTSTRAP_REPLICATES
OPTIMUM_BOOTSTRAP_SEED = 7_317_250
ACTIVE_TOLERANCE = fisher.ACTIVE_TOLERANCE
ZERO_WEIGHT_TOLERANCE = fisher.ZERO_WEIGHT_TOLERANCE
PLOT_CONFIG = fisher.PLOT_CONFIG

GATES = {
    **fisher.GATES,
    "ablation_rmse_difference_ci_high_max": 0.0,
    "ablation_fold_wins_min": 4,
    "all_bucket_amplitudes_active": True,
    "all_fold_bowl_mappings_valid": True,
}


@dataclass(frozen=True)
class BowlMapping:
    """Gauge-fixed family curvatures and the implied demand center."""

    valid: bool
    gauge_shift: float
    family_curvatures: np.ndarray
    center: np.ndarray


@dataclass(frozen=True)
class Fitted:
    """One target-specific convex anisotropic Hellinger response."""

    intercept: float
    amplitudes: np.ndarray
    family_contrasts: np.ndarray
    replay_coefficient: float
    ridge: float
    feature_scale: np.ndarray
    effective_df: float
    c_total: np.ndarray
    proportional: np.ndarray
    family_names: tuple[str, ...]
    family_members: tuple[np.ndarray, ...]
    contrast_basis: np.ndarray

    def family_slopes(self) -> np.ndarray:
        return self.contrast_basis.T @ self.family_contrasts

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design = aggregate_features(
            weights,
            self.c_total,
            self.proportional,
            self.family_members,
            self.contrast_basis,
        )
        coefficients = np.concatenate([self.amplitudes, self.family_contrasts, [self.replay_coefficient]])
        return self.intercept + design @ coefficients

    def bowl_mapping(self) -> BowlMapping:
        return recover_bowl_mapping(self.amplitudes, self.family_slopes(), self.family_members)

    def demand(self) -> np.ndarray:
        mapping = self.bowl_mapping()
        return mapping.center if mapping.valid else np.zeros_like(self.amplitudes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=OPTIMUM_BOOTSTRAP_REPLICATES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return fisher.json_ready(value)


def source_hash(path: Path) -> str:
    return fisher.source_hash(path)


def protocol_payload(bootstrap_replicates: int) -> dict[str, Any]:
    parent = fisher.protocol_payload(bootstrap_replicates)
    sources = (
        Path(__file__),
        Path(fisher.__file__),
        Path(physical.__file__),
        Path(intervention.__file__),
        Path(observatory.__file__),
    )
    payload: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "version": PROTOCOL_VERSION,
        "equation_identified_gauge": "A(w)=c+sum_f k_f W_f-sum_i a_i sqrt(w_i)+h R(w)",
        "equation_mechanistic": "A(w)=b+sum_f K_f sum_{i in f}(sqrt(w_i)-sqrt(q_i))^2+h R(w)",
        "constraints": "a_i>=0; h>=0; sum_f k_f=0; K_f=k_f+lambda>0; " "q_i=[a_i/(2K_f(i))]^2; sum_i q_i=1",
        "parameter_symmetry": (
            "family slopes are identified only modulo a common constant because sum_f W_f=1; "
            "orthonormal Helmert contrasts fix the zero-sum gauge, and lambda recovers positive curvatures"
        ),
        "exact_ablation": "k_f=0 recovers WSD80-SUR-067 isotropic Fisher-Rao demand overlap",
        "units": {
            "weights_overlap_replay_center": "dimensionless",
            "intercept_amplitudes_family_curvatures_replay_coefficient": "BPB",
        },
        "fit_data": "282 physically tied 300M policies only",
        "external_shape_data": "39 proportional antithetic log-tilt pairs and 11 fresh controls",
        "excluded_fit_data": "all asymmetric policies and all proportional intervention outcomes",
        "outer_folds": {"count": OUTER_SPLITS, "seed": OUTER_SEED, "group": "phase_correspondence_key"},
        "inner_folds": {"count": INNER_SPLITS, "seed_base": INNER_SEED_BASE},
        "ridge_grid": RIDGE_GRID,
        "ridge_selection": "minimum inner correspondence-grouped RMSE; numeric ties prefer smaller ridge",
        "ablation_comparison": {
            "model": ABLATION_ID,
            "draws": ABLATION_BOOTSTRAP_DRAWS,
            "seed": ABLATION_BOOTSTRAP_SEED,
            "unit": "phase_correspondence_key",
        },
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
            "Both targets must pass the inherited tied OOF, external shape, raw optimum, and stability gates; "
            "the candidate must also improve its isotropic ablation beyond paired-bootstrap uncertainty, "
            "win at least four of five outer folds, activate every bucket amplitude, and admit a positive-curvature "
            "Hellinger-bowl mapping in the full fit and every outer fold."
        ),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "source_hashes": {str(path.relative_to(REPO_ROOT)): source_hash(path) for path in sources},
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def family_geometry(
    dataset: fisher.baseline.pooled.Dataset,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    structured = observatory.family_dataset(dataset)
    names = tuple(str(name) for name in structured.family_names)
    members = tuple(np.asarray(indices, dtype=int) for indices in structured.family_members)
    if sorted(np.concatenate(members).tolist()) != list(range(dataset.m)):
        raise ValueError("Family partition does not cover each bucket exactly once")
    basis = np.asarray(helmert(len(names), full=False), dtype=float)
    if not np.allclose(basis @ np.ones(len(names)), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("Family contrast basis is not tangent to the family simplex")
    if not np.allclose(basis @ basis.T, np.eye(len(names) - 1), atol=1e-12, rtol=0.0):
        raise ValueError("Family contrast basis is not orthonormal")
    return names, members, basis


def family_masses(weights: np.ndarray, members: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.column_stack([weights[:, indices].sum(axis=1) for indices in members])


def aggregate_features(
    weights: np.ndarray,
    c_total: np.ndarray,
    proportional: np.ndarray,
    members: tuple[np.ndarray, ...],
    basis: np.ndarray,
) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(c_total):
        raise ValueError("Expected a [rows, domains] tied-mixture matrix")
    if np.any(values < -1e-10) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Invalid tied mixture")
    clipped = np.maximum(values, 0.0)
    contrasts = family_masses(clipped, members) @ basis.T
    replay = fisher.repeated_mass(clipped, c_total, proportional)
    return np.column_stack([-np.sqrt(clipped), contrasts, replay])


def recover_bowl_mapping(
    amplitudes: np.ndarray,
    family_slopes: np.ndarray,
    members: tuple[np.ndarray, ...],
) -> BowlMapping:
    bucket_family = np.empty(len(amplitudes), dtype=int)
    for family, indices in enumerate(members):
        bucket_family[indices] = family
    minimum = float(np.min(family_slopes))
    lower = -minimum + 1e-12

    def normalization(shift: float) -> float:
        curvature = family_slopes[bucket_family] + shift
        return float(np.sum((amplitudes / (2.0 * curvature)) ** 2) - 1.0)

    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < -1e-12):
        return BowlMapping(False, float("nan"), np.full_like(family_slopes, np.nan), np.zeros_like(amplitudes))
    low_value = normalization(lower)
    if not np.isfinite(low_value) or low_value <= 0.0:
        return BowlMapping(False, float("nan"), np.full_like(family_slopes, np.nan), np.zeros_like(amplitudes))
    upper = max(1.0, lower + 1.0)
    while normalization(upper) > 0.0 and upper < 1e12:
        upper *= 2.0
    if normalization(upper) >= 0.0:
        return BowlMapping(False, float("nan"), np.full_like(family_slopes, np.nan), np.zeros_like(amplitudes))
    shift = float(brentq(normalization, lower, upper, xtol=1e-13, rtol=1e-13))
    curvatures = family_slopes + shift
    center = (amplitudes / (2.0 * curvatures[bucket_family])) ** 2
    valid = bool(
        np.all(curvatures > 0.0) and np.all(center >= 0.0) and np.isclose(center.sum(), 1.0, atol=1e-9, rtol=0.0)
    )
    if valid:
        center /= center.sum()
    return BowlMapping(valid, shift, curvatures, center)


def solve_model(
    weights: np.ndarray,
    target: np.ndarray,
    c_total: np.ndarray,
    proportional: np.ndarray,
    family_names: tuple[str, ...],
    family_members: tuple[np.ndarray, ...],
    contrast_basis: np.ndarray,
    ridge: float,
) -> Fitted:
    design = aggregate_features(weights, c_total, proportional, family_members, contrast_basis)
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
    bucket_count = weights.shape[1]
    contrast_count = contrast_basis.shape[0]
    lower = np.concatenate([[-np.inf], np.zeros(bucket_count), np.full(contrast_count, -np.inf), [0.0]])
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
    active = np.concatenate(
        [
            result.x[1 : 1 + bucket_count] > ACTIVE_TOLERANCE,
            np.abs(result.x[1 + bucket_count : 1 + bucket_count + contrast_count]) > ACTIVE_TOLERANCE,
            result.x[-1:] > ACTIVE_TOLERANCE,
        ]
    )
    if np.any(active):
        centered = scaled[:, active] - scaled[:, active].mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        effective_df = 1.0 + float(np.sum(singular_values**2 / (singular_values**2 + ridge)))
    else:
        effective_df = 1.0
    return Fitted(
        intercept=float(result.x[0]),
        amplitudes=np.asarray(coefficients[:bucket_count], dtype=float),
        family_contrasts=np.asarray(coefficients[bucket_count : bucket_count + contrast_count], dtype=float),
        replay_coefficient=float(coefficients[-1]),
        ridge=ridge,
        feature_scale=scale,
        effective_df=effective_df,
        c_total=np.asarray(c_total, dtype=float),
        proportional=np.asarray(proportional, dtype=float),
        family_names=family_names,
        family_members=family_members,
        contrast_basis=contrast_basis,
    )


def select_ridge(
    dataset: fisher.baseline.pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[float, pd.DataFrame]:
    weights = dataset.weights[:, 0, :]
    proportional = fisher.proportional_policy(dataset.c0)
    family_names, family_members, basis = family_geometry(dataset)
    rows: list[dict[str, float | bool]] = []
    for ridge in RIDGE_GRID:
        prediction = np.full(dataset.n, np.nan)
        for train, test in folds:
            model = solve_model(
                weights[train],
                dataset.y[train],
                dataset.c0 + dataset.c1,
                proportional,
                family_names,
                family_members,
                basis,
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
    dataset: fisher.baseline.pooled.Dataset,
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, list[Fitted], pd.DataFrame]:
    prediction = np.full(dataset.n, np.nan)
    models: list[Fitted] = []
    selections: list[pd.DataFrame] = []
    for fold_id, (train, test) in enumerate(outer_folds):
        local = fisher.baseline.subset_dataset(dataset, train, f"outer_{fold_id}_train")
        inner = fisher.baseline.correspondence_folds(local.frame, INNER_SEED_BASE + fold_id, INNER_SPLITS)
        ridge, selection = select_ridge(local, inner)
        names, members, basis = family_geometry(local)
        model = solve_model(
            local.weights[:, 0, :],
            local.y,
            local.c0 + local.c1,
            fisher.proportional_policy(local.c0),
            names,
            members,
            basis,
            ridge,
        )
        prediction[test] = model.predict(dataset.weights[test, 0, :])
        models.append(model)
        selection.insert(0, "outer_fold", fold_id)
        selections.append(selection)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete outer-fold prediction")
    return prediction, models, pd.concat(selections, ignore_index=True)


def paired_ablation_metrics(
    dataset: fisher.baseline.pooled.Dataset,
    candidate: np.ndarray,
    ablation: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    target: str,
) -> dict[str, float | int]:
    candidate_rmse = float(np.sqrt(np.mean((candidate - dataset.y) ** 2)))
    ablation_rmse = float(np.sqrt(np.mean((ablation - dataset.y) ** 2)))
    differences = []
    generator = np.random.default_rng(ABLATION_BOOTSTRAP_SEED + TARGETS.index(target))
    for _ in range(ABLATION_BOOTSTRAP_DRAWS):
        indices = fisher.bootstrap_indices(dataset.frame, generator)
        candidate_value = float(np.sqrt(np.mean((candidate[indices] - dataset.y[indices]) ** 2)))
        ablation_value = float(np.sqrt(np.mean((ablation[indices] - dataset.y[indices]) ** 2)))
        differences.append(candidate_value - ablation_value)
    interval = np.quantile(np.asarray(differences), [0.025, 0.975])
    fold_wins = sum(
        np.sqrt(np.mean((candidate[test] - dataset.y[test]) ** 2))
        < np.sqrt(np.mean((ablation[test] - dataset.y[test]) ** 2))
        for _train, test in folds
    )
    return {
        "ablation_rmse": ablation_rmse,
        "candidate_minus_ablation_rmse": candidate_rmse - ablation_rmse,
        "candidate_minus_ablation_rmse_ci_low": float(interval[0]),
        "candidate_minus_ablation_rmse_ci_high": float(interval[1]),
        "candidate_ablation_fold_wins": int(fold_wins),
    }


def optimize_raw(model: Fitted) -> tuple[np.ndarray, float]:
    weights = cp.Variable(len(model.amplitudes), nonneg=True)
    family_mass = cp.hstack([cp.sum(weights[indices]) for indices in model.family_members])
    replay = cp.sum(
        cp.multiply(
            model.proportional,
            cp.pos(cp.multiply(model.c_total, weights) - 1.0),
        )
    )
    objective = cp.Minimize(
        model.intercept
        + model.family_slopes() @ family_mass
        - cp.sum(cp.multiply(model.amplitudes, cp.sqrt(weights)))
        + model.replay_coefficient * replay
    )
    problem = cp.Problem(objective, [cp.sum(weights) == 1.0])
    if not problem.is_dcp():
        raise ValueError("Raw anisotropic Hellinger optimum is not DCP")
    problem.solve(solver="CLARABEL", max_iter=2_000)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise RuntimeError(f"Convex raw optimization failed: {problem.status}")
    optimum = np.maximum(np.asarray(weights.value, dtype=float), 0.0)
    optimum /= optimum.sum()
    return optimum, float(model.predict(optimum[None, :])[0])


def optimum_record(
    model: Fitted,
    dataset: fisher.baseline.pooled.Dataset,
) -> tuple[dict[str, float | int | bool], pd.DataFrame]:
    optimum, prediction = optimize_raw(model)
    observed_best_index = int(np.argmin(dataset.y))
    observed_best = dataset.weights[observed_best_index, 0, :]
    observed_epochs = dataset.weights[:, 0, :] * (dataset.c0 + dataset.c1)
    epochs = optimum * (dataset.c0 + dataset.c1)
    beta0 = float(np.median(dataset.c0 / (dataset.c0 + dataset.c1)))
    support = fisher.baseline.support_distances(np.stack([optimum, optimum]), dataset.weights, beta0)
    mapping = model.bowl_mapping()
    diagnostics: dict[str, float | int | bool] = {
        "predicted_bpb": prediction,
        "observed_best_tied_bpb": float(dataset.y[observed_best_index]),
        "predicted_gain_beyond_observed_frontier": float(dataset.y[observed_best_index] - prediction),
        "l1_to_observed_best": float(np.abs(optimum - observed_best).sum()),
        "maximum_bucket_weight": float(np.max(optimum)),
        "observed_maximum_bucket_weight": float(np.max(dataset.weights[:, 0, :])),
        "maximum_materialized_epochs": float(np.max(epochs)),
        "observed_maximum_materialized_epochs": float(np.max(observed_epochs)),
        "near_zero_bucket_count": int(np.sum(optimum <= ZERO_WEIGHT_TOLERANCE)),
        "bowl_mapping_valid": mapping.valid,
        "l1_raw_optimum_to_unpenalized_center": (
            float(np.abs(optimum - mapping.center).sum()) if mapping.valid else float("nan")
        ),
        **support,
    }
    policy = pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "weight": optimum,
            "materialized_epochs": epochs,
            "observed_best_weight": observed_best,
            "unpenalized_center_weight": mapping.center,
        }
    )
    return diagnostics, policy


def parameter_record(model: Fitted, domains: list[str]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    mapping = model.bowl_mapping()
    payload = {
        "intercept_c_bpb": model.intercept,
        "replay_coefficient_h_bpb": model.replay_coefficient,
        "ridge": model.ridge,
        "nominal_parameter_count": len(model.amplitudes) + len(model.family_contrasts) + 2,
        "active_parameter_count": (
            int(np.sum(model.amplitudes > ACTIVE_TOLERANCE))
            + int(np.sum(np.abs(model.family_contrasts) > ACTIVE_TOLERANCE))
            + int(model.replay_coefficient > ACTIVE_TOLERANCE)
            + 1
        ),
        "active_bucket_amplitudes": int(np.sum(model.amplitudes > ACTIVE_TOLERANCE)),
        "effective_df_active_set": model.effective_df,
        "bowl_mapping_valid": mapping.valid,
        "bowl_gauge_shift_bpb": mapping.gauge_shift,
        "minimum_family_curvature_bpb": float(np.min(mapping.family_curvatures)) if mapping.valid else float("nan"),
        "maximum_family_curvature_bpb": float(np.max(mapping.family_curvatures)) if mapping.valid else float("nan"),
        "center_entropy": float(stats.entropy(mapping.center + 1e-30)) if mapping.valid else float("nan"),
        "maximum_center_weight": float(np.max(mapping.center)) if mapping.valid else float("nan"),
    }
    bucket_family = np.empty(len(domains), dtype=int)
    for family, indices in enumerate(model.family_members):
        bucket_family[indices] = family
    buckets = pd.DataFrame(
        {
            "domain": domains,
            "family": [model.family_names[index] for index in bucket_family],
            "amplitude_bpb": model.amplitudes,
            "center_weight": mapping.center,
        }
    )
    families = pd.DataFrame(
        {
            "family": model.family_names,
            "identified_slope_bpb": model.family_slopes(),
            "curvature_bpb": mapping.family_curvatures,
            "center_mass": [mapping.center[indices].sum() for indices in model.family_members],
        }
    )
    return payload, buckets, families


def bootstrap_optima(
    dataset: fisher.baseline.pooled.Dataset,
    template: Fitted,
    full_optimum: np.ndarray,
    replicates: int,
) -> pd.DataFrame:
    generator = np.random.default_rng(OPTIMUM_BOOTSTRAP_SEED)
    rows: list[dict[str, float | int | bool]] = []
    for replicate in range(replicates):
        indices = fisher.bootstrap_indices(dataset.frame, generator)
        sampled = fisher.baseline.subset_dataset(dataset, indices, f"bootstrap_{replicate}")
        names, members, basis = family_geometry(sampled)
        model = solve_model(
            sampled.weights[:, 0, :],
            sampled.y,
            sampled.c0 + sampled.c1,
            fisher.proportional_policy(sampled.c0),
            names,
            members,
            basis,
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
                "bowl_mapping_valid": model.bowl_mapping().valid,
                **{f"weight_{domain}": value for domain, value in zip(dataset.domain_names, optimum, strict=True)},
            }
        )
    return pd.DataFrame(rows)


def target_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    inherited = fisher.target_decision(metrics)["checks"]
    checks = {
        **inherited,
        "ablation_uncertainty": (
            metrics["candidate_minus_ablation_rmse_ci_high"] <= GATES["ablation_rmse_difference_ci_high_max"]
        ),
        "ablation_fold_wins": metrics["candidate_ablation_fold_wins"] >= GATES["ablation_fold_wins_min"],
        "all_bucket_amplitudes_active": metrics["active_bucket_amplitudes"] == metrics["bucket_count"],
        "full_bowl_mapping": bool(metrics["bowl_mapping_valid"]),
        "all_fold_bowl_mappings": bool(metrics["all_fold_bowl_mappings_valid"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def preflight_payload() -> dict[str, Any]:
    parent = fisher.preflight_payload()
    rows: dict[str, Any] = {}
    for target in TARGETS:
        dataset = fisher.tied_panel(target)
        names, members, basis = family_geometry(dataset)
        design = aggregate_features(
            dataset.weights[:, 0, :],
            dataset.c0 + dataset.c1,
            fisher.proportional_policy(dataset.c0),
            members,
            basis,
        )
        rows[target] = {
            "families": names,
            "family_sizes": [len(indices) for indices in members],
            "family_contrast_count": basis.shape[0],
            "design_columns": design.shape[1],
            "design_rank_with_intercept": int(np.linalg.matrix_rank(np.column_stack([np.ones(dataset.n), design]))),
        }
    return {
        **parent,
        "candidate_targets": rows,
        "nominal_parameter_count": 43,
        "exact_ablation_parameter_count": 41,
        "raw_problem_is_convex": True,
    }


def freeze_protocol(output_dir: Path, bootstrap_replicates: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload(bootstrap_replicates)
    path = output_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json_ready(payload):
        raise ValueError(f"Frozen protocol differs from current source: {path}")
    if not path.exists():
        fisher.baseline.write_json(path, payload)
    fisher.baseline.write_json(output_dir / "preflight.json", preflight_payload())
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


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
    return marker.exists() and json.loads(marker.read_text()).get("protocol_sha256") == protocol_hash


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
    dataset = fisher.tied_panel(target)
    outer_folds = fisher.baseline.correspondence_folds(dataset.frame, OUTER_SEED, OUTER_SPLITS)
    oof_prediction, fold_models, selections = fit_nested(dataset, outer_folds)
    ablation_prediction, _ablation_models, ablation_selections = fisher.fit_nested(dataset, outer_folds)
    selections.to_csv(path / "ridge_selection.csv", index=False)
    ablation_selections.to_csv(path / "ablation_ridge_selection.csv", index=False)
    predictions = dataset.frame[["run_name", "phase_correspondence_key", "policy_family", "source_panel"]].copy()
    predictions["observed"] = dataset.y
    predictions["predicted"] = oof_prediction
    predictions["ablation_predicted"] = ablation_prediction
    predictions["residual"] = oof_prediction - dataset.y
    predictions["optimism"] = dataset.y - oof_prediction
    predictions.to_csv(path / "oof_predictions.csv", index=False)

    full_inner = fisher.baseline.correspondence_folds(dataset.frame, FULL_FIT_SEED, INNER_SPLITS)
    selected_ridge, full_selection = select_ridge(dataset, full_inner)
    full_selection.to_csv(path / "full_ridge_selection.csv", index=False)
    names, members, basis = family_geometry(dataset)
    full_model = solve_model(
        dataset.weights[:, 0, :],
        dataset.y,
        dataset.c0 + dataset.c1,
        fisher.proportional_policy(dataset.c0),
        names,
        members,
        basis,
        selected_ridge,
    )
    parameter_payload, buckets, families = parameter_record(full_model, dataset.domain_names)
    buckets.to_csv(path / "bucket_parameters.csv", index=False)
    families.to_csv(path / "family_parameters.csv", index=False)

    manifest, domains, _c0, _c1, _proportional = intervention.geometry()
    if domains != list(dataset.domain_names):
        raise ValueError("Intervention and tied-panel domain order differ")
    effects = intervention.target_effects(manifest, domains)
    pair_metrics, pair_predictions = fisher.external_pair_metrics(full_model, target, domains, manifest, effects[target])
    pair_predictions.to_csv(path / "external_pair_predictions.csv", index=False)

    optimum_metrics, optimum_policy = optimum_record(full_model, dataset)
    optimum_policy.to_csv(path / "raw_optimum_policy.csv", index=False)
    fold_rows = []
    for fold, model in enumerate(fold_models):
        optimum, prediction = optimize_raw(model)
        fold_rows.append(
            {
                "fold": fold,
                "predicted_bpb": prediction,
                "bowl_mapping_valid": model.bowl_mapping().valid,
                **{f"weight_{domain}": value for domain, value in zip(dataset.domain_names, optimum, strict=True)},
            }
        )
    fold_optima = pd.DataFrame(fold_rows)
    weight_columns = [f"weight_{domain}" for domain in dataset.domain_names]
    full_weights = optimum_policy["weight"].to_numpy(float)
    fold_optima["l1_to_full_optimum"] = np.abs(fold_optima[weight_columns].to_numpy(float) - full_weights).sum(axis=1)
    fold_optima.to_csv(path / "fold_optima.csv", index=False)

    bootstraps = bootstrap_optima(dataset, full_model, full_weights, bootstrap_replicates)
    bootstraps.to_csv(path / "bootstrap_optima.csv", index=False)
    demand_stability = fisher.pairwise_cosine([model.demand() for model in fold_models])
    ablation_metrics = paired_ablation_metrics(dataset, oof_prediction, ablation_prediction, outer_folds, target)
    metrics: dict[str, Any] = {
        "target": target,
        "model": MODEL_ID,
        "tied_rows": dataset.n,
        "bucket_count": dataset.m,
        **fisher.metric_summary(dataset.y, oof_prediction, outer_folds),
        **ablation_metrics,
        **pair_metrics,
        **optimum_metrics,
        **demand_stability,
        "fold_optimum_median_l1": float(fold_optima["l1_to_full_optimum"].median()),
        "fold_optimum_maximum_l1": float(fold_optima["l1_to_full_optimum"].max()),
        "bootstrap_optimum_median_l1": float(bootstraps["l1_to_full_optimum"].median()),
        "bootstrap_optimum_maximum_l1": float(bootstraps["l1_to_full_optimum"].max()),
        "all_fold_bowl_mappings_valid": bool(fold_optima["bowl_mapping_valid"].all()),
        "bootstrap_bowl_mapping_valid_fraction": float(bootstraps["bowl_mapping_valid"].mean()),
        **parameter_payload,
    }
    reference = physical.FROZEN_REFERENCE_RMSE[target]
    metrics["frozen_reference_rmse"] = reference
    metrics["relative_rmse_to_reference"] = float(metrics["rmse"]) / reference - 1.0
    decision = target_decision(metrics)
    fisher.baseline.write_json(path / "metrics.json", metrics)
    fisher.baseline.write_json(path / "decision.json", decision)
    fisher.baseline.write_json(
        path / "complete.json",
        {"protocol_sha256": protocol["protocol_sha256"], "passed": decision["passed"]},
    )


def write_plots(output_dir: Path) -> None:
    figure = make_subplots(
        rows=len(TARGETS),
        cols=2,
        subplot_titles=tuple(title for target in TARGETS for title in (f"{target}: candidate", f"{target}: ablation")),
    )
    for row, target in enumerate(TARGETS, start=1):
        frame = pd.read_csv(output_dir / "cells" / target / "oof_predictions.csv")
        for column, prediction_name in enumerate(("predicted", "ablation_predicted"), start=1):
            residual = frame[prediction_name] - frame["observed"]
            figure.add_trace(
                go.Scatter(
                    x=frame["observed"],
                    y=frame[prediction_name],
                    mode="markers",
                    marker={"size": 8, "color": residual, "colorscale": "RdYlGn_r"},
                    showlegend=False,
                    hovertemplate="observed=%{x:.6f}<br>predicted=%{y:.6f}<extra></extra>",
                ),
                row=row,
                col=column,
            )
            lower = float(min(frame["observed"].min(), frame[prediction_name].min()))
            upper = float(max(frame["observed"].max(), frame[prediction_name].max()))
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
        title="Family-anisotropic Hellinger bowl versus isotropic ablation",
        template="plotly_white",
        width=1300,
        height=560 * len(TARGETS),
    )
    figure.write_html(output_dir / "aggregate_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)

    policies = make_subplots(rows=len(TARGETS), cols=1, subplot_titles=TARGETS, shared_xaxes=True)
    for row, target in enumerate(TARGETS, start=1):
        frame = pd.read_csv(output_dir / "cells" / target / "raw_optimum_policy.csv")
        for column, name, color in (
            ("weight", "raw optimum", "#d73027"),
            ("unpenalized_center_weight", "Hellinger center", "#fdae61"),
            ("observed_best_weight", "observed tied frontier", "#1a9850"),
        ):
            policies.add_trace(
                go.Bar(x=frame["domain"], y=frame[column], name=f"{target}: {name}", marker_color=color),
                row=row,
                col=1,
            )
    policies.update_layout(
        title="Anisotropic Hellinger centers and raw tied optima",
        template="plotly_white",
        barmode="group",
        width=1500,
        height=520 * len(TARGETS),
    )
    policies.write_html(output_dir / "raw_optimum_mixtures.html", include_plotlyjs=True, config=PLOT_CONFIG)


def collect(output_dir: Path, protocol: dict[str, Any]) -> None:
    rows = []
    decisions: dict[str, Any] = {}
    for target in TARGETS:
        path = output_dir / "cells" / target
        rows.append(json.loads((path / "metrics.json").read_text()))
        decisions[target] = json.loads((path / "decision.json").read_text())
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    passed = all(decision["passed"] for decision in decisions.values())
    decision = {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "passed": passed,
        "targets": decisions,
        "decision": "PASS: temporal-state work licensed" if passed else "FAIL: anisotropic aggregate route rejected",
    }
    fisher.baseline.write_json(output_dir / "decision.json", decision)
    write_plots(output_dir)
    columns = [
        "target",
        "rmse",
        "frozen_reference_rmse",
        "relative_rmse_to_reference",
        "ablation_rmse",
        "candidate_minus_ablation_rmse_ci_low",
        "candidate_minus_ablation_rmse_ci_high",
        "candidate_ablation_fold_wins",
        "pair_odd_rmse_improvement",
        "pair_even_rmse_improvement",
        "predicted_bpb",
        "observed_best_tied_bpb",
        "nearest_aggregate_tv",
        "maximum_bucket_weight",
        "active_bucket_amplitudes",
        "bowl_mapping_valid",
        "all_fold_bowl_mappings_valid",
    ]
    lines = [
        "# Family-anisotropic Hellinger aggregate-spine audit",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        (
            "The candidate and its isotropic ablation were fit only on 282 physically tied policies. "
            "The proportional antithetic intervention outcomes were external to fitting and selection."
        ),
        "",
        metrics[columns].to_markdown(index=False),
        "",
        (
            "The two added family contrasts are accepted as mechanistic only when they map to positive "
            "family curvatures and a normalized Hellinger center. A pass licenses a separately identified "
            "temporal-state stage; it does not promote a complete two-phase surrogate."
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
