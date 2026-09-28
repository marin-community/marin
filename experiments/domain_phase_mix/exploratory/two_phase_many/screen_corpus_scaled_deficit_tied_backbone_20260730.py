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
"""Screen a corpus-scaled inverse-power aggregate backbone without phase outcomes.

The new mechanism is one global relation between finite-pool scale and
diminishing-return curvature. For a tied aggregate mixture ``a``, define

    k_i = c_i^(0) + c_i^(1)
    alpha_i = alpha_0 * (k_i / k_g)^nu
    D_i(a_i) = (a_i + E_0)^(-alpha_i)

where ``k_i`` is the number of materialized epochs induced by unit mixture
share and ``k_g`` is its geometric mean. A smaller corpus has larger ``k_i``;
positive ``nu`` therefore gives it faster curvature in mixture-share space.
The exact ``nu=0`` ablation recovers a shared inverse-power exponent.

This is deliberately a phase-blind screen. WSD80 uses only the tied diagonal.
The 300M gate, reached only after WSD80 passes, uses only the 282 physically
tied policies from the original two-phase panel and qsplit240 exposure-average
ablation. No asymmetric outcome, Delphi row, heldout, or deployment regularizer
enters selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_compact_tied_backbone_20260730 as compact_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "corpus_scaled_deficit_tied_screen_20260730"
BENEFIT_EXPONENTS = (0.25, 0.5, 1.0)
BENEFIT_OFFSETS = (0.01, 0.1)
DAMAGE_EXPONENTS = (1.5, 2.0, 3.0)
NU_GRID = (0.0, 0.25, 0.5, 1.0)
POSITIVE_NU_GRID = tuple(value for value in NU_GRID if value > 0.0)
RIDGE_GRID = (1e-4, 1e-2, 1.0)
OUTER_SPLITS = 3
INNER_SPLITS = 3
OPTIMIZER_STARTS = 16
TARGETS = ("uncheatable", "table9")
ZERO_TOLERANCE = 1e-10
WEIGHT_ZERO_TOLERANCE = 1e-6
WSD_MINIMUM_RELATIVE_IMPROVEMENT = 0.05
WSD_MAXIMUM_OPTIMUM_DISTANCE = 0.05
WSD_MINIMUM_PREDICTED_OPTIMUM_BPB = 0.940429
WSD_CRS_NESTED_RMSE = {"random": 0.08759153295999832, "blocked": 0.09903206961101599}
MAXIMUM_NU = max(NU_GRID)
GATES_300M = {
    "uncheatable_oof_rmse": 0.0056,
    "table9_oof_rmse": 0.0125,
    "median_optimum_l1": 0.05,
    "maximum_optimum_l1": 0.10,
    "maximum_zero_amplitudes": 8,
}


@dataclass(frozen=True)
class Shape:
    """Nonlinear response shape."""

    benefit_exponent: float
    benefit_offset: float
    damage_exponent: float
    corpus_exponent: float


@dataclass(frozen=True)
class Fitted:
    """Fitted nonnegative response head."""

    shape: Shape
    ridge: float
    intercept: float
    coefficients: np.ndarray
    geometry: retained.Geometry

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept + design_matrix(weights, self.geometry, self.shape) @ self.coefficients


@dataclass(frozen=True)
class NestedResult:
    """Nested OOF predictions and fold-specific raw optima."""

    prediction: np.ndarray
    shapes: tuple[Shape, ...]
    ridges: tuple[float, ...]
    optima: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def shape_grid(corpus_exponents: tuple[float, ...]) -> tuple[Shape, ...]:
    """Frozen candidate shapes for one corpus-exponent class."""
    return tuple(
        Shape(*values)
        for values in product(
            BENEFIT_EXPONENTS,
            BENEFIT_OFFSETS,
            DAMAGE_EXPONENTS,
            corpus_exponents,
        )
    )


def epoch_scale(geometry: retained.Geometry) -> np.ndarray:
    """Materialized epochs induced by unit aggregate mixture share."""
    return np.asarray(geometry.c0 + geometry.c1, dtype=float)


def domain_exponents(geometry: retained.Geometry, shape: Shape) -> np.ndarray:
    """Per-domain inverse-power exponents tied to finite-pool scale."""
    scale = epoch_scale(geometry)
    geometric_mean = float(np.exp(np.mean(np.log(np.maximum(scale, 1e-12)))))
    return shape.benefit_exponent * (scale / geometric_mean) ** shape.corpus_exponent


def family_totals(values: np.ndarray, geometry: retained.Geometry) -> np.ndarray:
    """Sum domain values within each predeclared family."""
    families = geometry.families
    return np.stack([values[:, families == family].sum(axis=1) for family in np.unique(families)], axis=1)


def hierarchical_block(values: np.ndarray, geometry: retained.Geometry) -> np.ndarray:
    """Family response plus shrunk bucket-level excess columns."""
    pooled = family_totals(values, geometry)
    if not len(geometry.excess_domains):
        return pooled
    return np.column_stack([pooled, values[:, geometry.excess_domains]])


def aggregate_weights(weights: np.ndarray, geometry: retained.Geometry) -> np.ndarray:
    """Token-weighted mixture aggregate; identical to either phase when tied."""
    beta0 = geometry.phase_0_fraction
    return beta0 * weights[:, 0, :] + (1.0 - beta0) * weights[:, 1, :]


def design_matrix(weights: np.ndarray, geometry: retained.Geometry, shape: Shape) -> np.ndarray:
    """Inverse-power shortage and physical repetition-damage columns."""
    aggregate = aggregate_weights(weights, geometry)
    exponents = domain_exponents(geometry, shape)
    benefit = (aggregate + shape.benefit_offset) ** (-exponents[None, :])
    epochs = aggregate * epoch_scale(geometry)[None, :]
    damage = epochs**shape.damage_exponent
    return np.column_stack([hierarchical_block(benefit, geometry), hierarchical_block(damage, geometry)])


def penalty_multipliers(geometry: retained.Geometry) -> np.ndarray:
    """Shrink bucket departures toward family responses, not family amplitudes."""
    family_count = len(np.unique(geometry.families))
    if len(geometry.excess_domains):
        block = np.concatenate([np.zeros(family_count), np.ones(len(geometry.excess_domains))])
    else:
        # With singleton families there are no departure columns. Penalize the
        # domain amplitudes directly so the advertised ridge remains active.
        block = np.ones(family_count)
    return np.concatenate([block, block])


def fit_shape(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: retained.Geometry,
    indices: np.ndarray,
    shape: Shape,
    ridge: float,
) -> Fitted:
    """Fit one frozen shape on selected rows."""
    design = design_matrix(weights[indices], geometry, shape)
    intercept, coefficients = retained.solve_head(
        design,
        target[indices],
        ridge,
        penalty_multipliers(geometry),
    )
    return Fitted(
        shape=shape,
        ridge=ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )


def select_model(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: retained.Geometry,
    indices: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    corpus_exponents: tuple[float, ...],
) -> tuple[Fitted, pd.DataFrame]:
    """Select shape and ridge by OOF error inside the supplied rows."""
    rows = []
    for shape in shape_grid(corpus_exponents):
        for ridge in RIDGE_GRID:
            errors = []
            for train, test in folds:
                model = fit_shape(weights, target, geometry, train, shape, ridge)
                errors.append(model.predict(weights[test]) - target[test])
            rows.append(
                {
                    "benefit_exponent": shape.benefit_exponent,
                    "benefit_offset": shape.benefit_offset,
                    "damage_exponent": shape.damage_exponent,
                    "corpus_exponent": shape.corpus_exponent,
                    "ridge": ridge,
                    "rmse": float(np.sqrt(np.mean(np.concatenate(errors) ** 2))),
                }
            )
    sweep = pd.DataFrame(rows).sort_values(
        ["rmse", "corpus_exponent", "benefit_exponent", "benefit_offset", "damage_exponent", "ridge"]
    )
    selected = sweep.iloc[0]
    shape = Shape(
        benefit_exponent=float(selected["benefit_exponent"]),
        benefit_offset=float(selected["benefit_offset"]),
        damage_exponent=float(selected["damage_exponent"]),
        corpus_exponent=float(selected["corpus_exponent"]),
    )
    model = fit_shape(weights, target, geometry, indices, shape, float(selected["ridge"]))
    return model, sweep


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Prediction and observed-on-predicted calibration metrics."""
    residual = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    calibrated = intercept + slope * predicted
    centered = observed - np.mean(observed)
    total_sum_squares = float(centered @ centered)
    residual_sum_squares = float((observed - calibrated) @ (observed - calibrated))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias": float(np.mean(residual)),
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
        "observed_on_predicted_r2": float(1.0 - residual_sum_squares / total_sum_squares),
    }


def nested_predictions(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: retained.Geometry,
    indices: np.ndarray,
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    inner_folds,
    corpus_exponents: tuple[float, ...],
    optimize,
) -> NestedResult:
    """Nested shape selection, OOF prediction, and optional raw optimization."""
    prediction = np.full(len(target), np.nan)
    shapes = []
    ridges = []
    optima = []
    for fold_id, (train, test) in enumerate(outer_folds):
        inner = inner_folds(train, fold_id)
        model, _sweep = select_model(weights, target, geometry, train, inner, corpus_exponents)
        prediction[test] = model.predict(weights[test])
        shapes.append(model.shape)
        ridges.append(model.ridge)
        if optimize is not None:
            optima.append(optimize(model, train, fold_id))
    if not np.isfinite(prediction[indices]).all():
        raise ValueError("nested screen left incomplete predictions")
    return NestedResult(
        prediction=prediction,
        shapes=tuple(shapes),
        ridges=tuple(ridges),
        optima=tuple(optima),
    )


def wsd_tied_indices(panel: wsd80.Panel) -> np.ndarray:
    """Physically tied WSD80 policies."""
    return np.flatnonzero(np.isclose(panel.weights[:, 0, 1], panel.weights[:, 1, 1]))


def wsd_nested(
    panel: wsd80.Panel,
    indices: np.ndarray,
    protocol: str,
    corpus_exponents: tuple[float, ...],
    seed: int,
) -> NestedResult:
    """Nested WSD tied-diagonal predictions under one fold protocol."""
    geometry = retained.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    outer = benchmark.wsd_folds(panel.weights, indices, OUTER_SPLITS, seed, protocol)

    def inner(train: np.ndarray, fold_id: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return benchmark.wsd_folds(
            panel.weights,
            train,
            min(INNER_SPLITS, len(train)),
            seed + 100 + fold_id,
            protocol,
        )

    return nested_predictions(
        panel.weights,
        panel.y,
        geometry,
        indices,
        outer,
        inner,
        corpus_exponents,
        optimize=None,
    )


def fit_full_wsd(
    panel: wsd80.Panel,
    indices: np.ndarray,
    protocol: str,
    corpus_exponents: tuple[float, ...],
    seed: int,
) -> tuple[Fitted, pd.DataFrame]:
    """Select a full WSD tied model without asymmetric rows."""
    geometry = retained.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    folds = benchmark.wsd_folds(panel.weights, indices, OUTER_SPLITS, seed + 500, protocol)
    return select_model(panel.weights, panel.y, geometry, indices, folds, corpus_exponents)


def optimize_wsd_tied(model: Fitted) -> tuple[float, float]:
    """Dense raw optimization of the two-bucket tied diagonal."""
    axis = np.linspace(0.0, 1.0, 10001)
    values = model.predict(benchmark.grid_weights(axis, axis))
    best = int(np.argmin(values))
    return float(axis[best]), float(values[best])


def relative_improvement(baseline: float, candidate: float) -> float:
    """Fractional RMSE reduction, positive when the candidate is better."""
    return (baseline - candidate) / baseline


def audit_wsd(
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Run the frozen tied-only WSD screen against the exact nu=0 ablation."""
    panel = wsd80.load_surface()
    indices = wsd_tied_indices(panel)
    observed_index = indices[int(np.argmin(panel.y[indices]))]
    observed_share = float(panel.weights[observed_index, 0, 1])
    metric_rows = []
    prediction_rows = []
    selection_rows = []
    sweep_frames = []
    for protocol in ("random", "blocked"):
        ablation = wsd_nested(panel, indices, protocol, (0.0,), seed)
        candidate = wsd_nested(panel, indices, protocol, POSITIVE_NU_GRID, seed)
        full_candidate, sweep = fit_full_wsd(panel, indices, protocol, POSITIVE_NU_GRID, seed)
        optimum_share, optimum_value = optimize_wsd_tied(full_candidate)
        ablation_metrics = metric_summary(panel.y[indices], ablation.prediction[indices])
        candidate_metrics = metric_summary(panel.y[indices], candidate.prediction[indices])
        improvement = relative_improvement(ablation_metrics["rmse"], candidate_metrics["rmse"])
        selected_nu = full_candidate.shape.corpus_exponent
        row = {
            "panel": "wsd80_tied",
            "protocol": protocol,
            "n_rows": len(indices),
            "ablation_rmse": ablation_metrics["rmse"],
            "candidate_rmse": candidate_metrics["rmse"],
            "relative_rmse_improvement": improvement,
            "ablation_spearman": ablation_metrics["spearman"],
            "candidate_spearman": candidate_metrics["spearman"],
            "candidate_observed_on_predicted_slope": candidate_metrics["observed_on_predicted_slope"],
            "candidate_observed_on_predicted_r2": candidate_metrics["observed_on_predicted_r2"],
            "selected_corpus_exponent": selected_nu,
            "selected_benefit_exponent": full_candidate.shape.benefit_exponent,
            "selected_benefit_offset": full_candidate.shape.benefit_offset,
            "selected_damage_exponent": full_candidate.shape.damage_exponent,
            "selected_ridge": full_candidate.ridge,
            "predicted_tied_optimum_share": optimum_share,
            "predicted_tied_optimum_bpb": optimum_value,
            "observed_tied_optimum_share": observed_share,
            "observed_tied_optimum_bpb": float(panel.y[observed_index]),
            "passes_improvement_gate": improvement >= WSD_MINIMUM_RELATIVE_IMPROVEMENT,
            "passes_absolute_rmse_gate": candidate_metrics["rmse"] <= WSD_CRS_NESTED_RMSE[protocol],
            "passes_nu_identification_gate": 0.0 < selected_nu < MAXIMUM_NU,
            "passes_optimum_location_gate": abs(optimum_share - observed_share) <= WSD_MAXIMUM_OPTIMUM_DISTANCE,
            "passes_optimum_value_gate": optimum_value >= WSD_MINIMUM_PREDICTED_OPTIMUM_BPB,
        }
        metric_rows.append(row)
        for model_class, nested in (("nu_zero_ablation", ablation), ("positive_nu_candidate", candidate)):
            selection_rows.extend(
                {
                    "protocol": protocol,
                    "model_class": model_class,
                    "outer_fold": fold_id,
                    "benefit_exponent": shape.benefit_exponent,
                    "benefit_offset": shape.benefit_offset,
                    "damage_exponent": shape.damage_exponent,
                    "corpus_exponent": shape.corpus_exponent,
                    "ridge": nested.ridges[fold_id],
                }
                for fold_id, shape in enumerate(nested.shapes)
            )
        prediction_rows.extend(
            {
                "protocol": protocol,
                "row": int(index),
                "starcoder_weight": float(panel.weights[index, 0, 1]),
                "observed": float(panel.y[index]),
                "ablation_oof_prediction": float(ablation.prediction[index]),
                "candidate_oof_prediction": float(candidate.prediction[index]),
            }
            for index in indices
        )
        local_sweep = sweep.copy()
        local_sweep.insert(0, "protocol", protocol)
        sweep_frames.append(local_sweep)
    metrics = pd.DataFrame(metric_rows)
    gate_columns = [column for column in metrics if column.startswith("passes_")]
    passed = bool(metrics[gate_columns].all(axis=None))
    return (
        metrics,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selection_rows),
        pd.concat(sweep_frames, ignore_index=True),
        passed,
    )


def geometry_300m(dataset: benchmark.Dataset) -> retained.Geometry:
    """Retained-model geometry for the 39-bucket panel."""
    beta0 = float(np.median(dataset.c0 / (dataset.c0 + dataset.c1)))
    return retained.Geometry(
        c0=dataset.c0,
        c1=dataset.c1,
        phase_0_fraction=beta0,
        family_index=dataset.family_index,
    )


def domain_coefficients(model: Fitted, block: int) -> np.ndarray:
    """Expand one hierarchical response block to an effective coefficient per domain."""
    geometry = model.geometry
    family_count = len(np.unique(geometry.families))
    block_size = family_count + len(geometry.excess_domains)
    start = block * block_size
    local = model.coefficients[start : start + block_size]
    coefficients = local[geometry.families].copy()
    for excess_index, domain in enumerate(geometry.excess_domains):
        coefficients[domain] += local[family_count + excess_index]
    return coefficients


def tied_prediction_and_gradient(model: Fitted, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate one tied policy and its simplex gradient."""
    weights = np.asarray(weights, dtype=float)
    exponents = domain_exponents(model.geometry, model.shape)
    scale = epoch_scale(model.geometry)
    benefit_coefficients = domain_coefficients(model, 0)
    damage_coefficients = domain_coefficients(model, 1)
    shifted = weights + model.shape.benefit_offset
    epochs = weights * scale
    benefit = shifted**-exponents
    damage = epochs**model.shape.damage_exponent
    prediction = float(model.intercept + benefit_coefficients @ benefit + damage_coefficients @ damage)
    gradient = -benefit_coefficients * exponents * shifted ** (
        -exponents - 1.0
    ) + damage_coefficients * model.shape.damage_exponent * scale * np.maximum(epochs, 0.0) ** (
        model.shape.damage_exponent - 1.0
    )
    direct = float(model.predict(np.stack([weights, weights], axis=0)[None, :, :])[0])
    if not np.isclose(prediction, direct, atol=1e-10, rtol=1e-10):
        raise AssertionError(f"analytic prediction mismatch: {prediction} != {direct}")
    return prediction, gradient


def optimizer_starts(
    dataset: benchmark.Dataset,
    indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Deterministic support-spanning starts for raw tied optimization."""
    tied_weights = dataset.weights[indices, 0, :]
    observed_best = tied_weights[int(np.argmin(dataset.y[indices]))]
    equal_epoch = 1.0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)
    equal_epoch /= equal_epoch.sum()
    starts = [observed_best, np.full(len(dataset.domain_names), 1.0 / len(dataset.domain_names)), equal_epoch]
    generator = np.random.default_rng(seed)
    while len(starts) < OPTIMIZER_STARTS:
        starts.append(generator.dirichlet(np.ones(len(dataset.domain_names))))
    return tuple(np.asarray(start, dtype=float) for start in starts)


def optimize_300m(
    dataset: benchmark.Dataset,
    indices: np.ndarray,
    model: Fitted,
    seed: int,
) -> np.ndarray:
    """Optimize the raw tied model without a trust region or deployment prior."""
    constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda weights: np.ones_like(weights),
    }
    candidates = []
    for start in optimizer_starts(dataset, indices, seed):
        result = minimize(
            lambda weights: tied_prediction_and_gradient(model, weights),
            start,
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * len(dataset.domain_names),
            constraints=[constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
            weights /= weights.sum()
            candidates.append((float(result.fun), weights))
    if not candidates:
        raise RuntimeError(f"no finite tied optimum for {dataset.name}")
    return min(candidates, key=lambda item: item[0])[1]


def audit_300m_target(
    target_name: str,
    seed: int,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame, pd.DataFrame]:
    """Run the tied-only 300M screen after WSD eligibility."""
    dataset = benchmark.load_300m(target_name)
    geometry = geometry_300m(dataset)
    tied = np.flatnonzero(benchmark.replay_control.tied_rows(dataset.weights))
    groups = dataset.frame["phase_correspondence_key"].astype(str).to_numpy()
    outer = compact_audit.group_folds(tied, groups, OUTER_SPLITS, seed)

    def inner(train: np.ndarray, fold_id: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return compact_audit.group_folds(train, groups, INNER_SPLITS, seed + 100 + fold_id)

    def optimize(model: Fitted, train: np.ndarray, fold_id: int) -> np.ndarray:
        return optimize_300m(dataset, train, model, seed + 1000 + fold_id)

    ablation = nested_predictions(
        dataset.weights,
        dataset.y,
        geometry,
        tied,
        outer,
        inner,
        (0.0,),
        optimize=None,
    )
    candidate = nested_predictions(
        dataset.weights,
        dataset.y,
        geometry,
        tied,
        outer,
        inner,
        POSITIVE_NU_GRID,
        optimize=optimize,
    )
    full_folds = compact_audit.group_folds(tied, groups, INNER_SPLITS, seed + 500)
    full_candidate, sweep = select_model(
        dataset.weights,
        dataset.y,
        geometry,
        tied,
        full_folds,
        POSITIVE_NU_GRID,
    )
    full_optimum = optimize_300m(dataset, tied, full_candidate, seed + 1500)
    fold_distances = np.asarray([np.abs(optimum - full_optimum).sum() for optimum in candidate.optima])
    candidate_metrics = metric_summary(dataset.y[tied], candidate.prediction[tied])
    ablation_metrics = metric_summary(dataset.y[tied], ablation.prediction[tied])
    family_count = len(np.unique(geometry.families))
    block_size = family_count + len(geometry.excess_domains)
    zero_amplitudes = int(np.sum(full_candidate.coefficients[:block_size] <= ZERO_TOLERANCE))
    exposure = full_optimum * epoch_scale(geometry)
    threshold = GATES_300M[f"{target_name}_oof_rmse"]
    row: dict[str, float | int | str | bool] = {
        "target": target_name,
        "n_tied": len(tied),
        "ablation_oof_rmse": ablation_metrics["rmse"],
        "candidate_oof_rmse": candidate_metrics["rmse"],
        "relative_oof_improvement": relative_improvement(
            ablation_metrics["rmse"],
            candidate_metrics["rmse"],
        ),
        "candidate_oof_spearman": candidate_metrics["spearman"],
        "candidate_observed_on_predicted_slope": candidate_metrics["observed_on_predicted_slope"],
        "candidate_observed_on_predicted_r2": candidate_metrics["observed_on_predicted_r2"],
        "selected_corpus_exponent": full_candidate.shape.corpus_exponent,
        "selected_benefit_exponent": full_candidate.shape.benefit_exponent,
        "selected_benefit_offset": full_candidate.shape.benefit_offset,
        "selected_damage_exponent": full_candidate.shape.damage_exponent,
        "selected_ridge": full_candidate.ridge,
        "median_fold_to_full_optimum_l1": float(np.median(fold_distances)),
        "maximum_fold_to_full_optimum_l1": float(np.max(fold_distances)),
        "maximum_optimum_weight": float(np.max(full_optimum)),
        "maximum_optimum_epochs": float(np.max(exposure)),
        "near_zero_optimum_weights": int(np.sum(full_optimum <= WEIGHT_ZERO_TOLERANCE)),
        "zero_benefit_amplitudes": zero_amplitudes,
        "passes_oof_gate": candidate_metrics["rmse"] <= threshold,
        "passes_median_stability_gate": np.median(fold_distances) <= GATES_300M["median_optimum_l1"],
        "passes_maximum_stability_gate": np.max(fold_distances) <= GATES_300M["maximum_optimum_l1"],
        "passes_amplitude_gate": zero_amplitudes <= GATES_300M["maximum_zero_amplitudes"],
        "passes_ablation_gate": candidate_metrics["rmse"] < ablation_metrics["rmse"],
    }
    predictions = pd.DataFrame(
        {
            "target": target_name,
            "row": tied,
            "run_name": dataset.frame.iloc[tied]["run_name"].astype(str).to_numpy(),
            "observed": dataset.y[tied],
            "ablation_oof_prediction": ablation.prediction[tied],
            "candidate_oof_prediction": candidate.prediction[tied],
        }
    )
    optima = pd.DataFrame(
        [
            {
                "target": target_name,
                "fit": "full",
                "fold": -1,
                **{f"weight_{name}": value for name, value in zip(dataset.domain_names, full_optimum, strict=True)},
            }
        ]
        + [
            {
                "target": target_name,
                "fit": "outer",
                "fold": fold_id,
                **{f"weight_{name}": value for name, value in zip(dataset.domain_names, optimum, strict=True)},
            }
            for fold_id, optimum in enumerate(candidate.optima)
        ]
    )
    return row, predictions, optima.assign(selected_sweep_rows=len(sweep))


def write_report(
    wsd_metrics: pd.DataFrame,
    passed_wsd: bool,
    metrics_300m: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write the bounded screen decision."""
    lines = [
        "# Corpus-scaled inverse-power tied-backbone screen",
        "",
        "No asymmetric outcome entered fitting, selection, or scoring.",
        "",
        "## WSD80 tied-only eligibility",
        "",
        wsd_metrics.to_markdown(index=False),
        "",
        f"**WSD80 gate:** {'PASS' if passed_wsd else 'REJECT'}.",
        "",
    ]
    if passed_wsd:
        lines.extend(
            [
                "## 300M high-TPP tied gate",
                "",
                metrics_300m.to_markdown(index=False),
                "",
                "The model is eligible for a separately preregistered phase-mechanism test only if",
                "every 300M gate passes. Eligibility is not promotion.",
            ]
        )
    else:
        lines.extend(
            [
                "The candidate was rejected before reading any 300M asymmetric outcome. The",
                "conditional 300M tied gate was not run.",
            ]
        )
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wsd_metrics, wsd_predictions, wsd_selections, wsd_sweep, passed_wsd = audit_wsd(args.seed)
    wsd_metrics.to_csv(args.output_dir / "wsd_metrics.csv", index=False)
    wsd_predictions.to_csv(args.output_dir / "wsd_predictions.csv", index=False)
    wsd_selections.to_csv(args.output_dir / "wsd_fold_selections.csv", index=False)
    wsd_sweep.to_csv(args.output_dir / "wsd_shape_sweep.csv", index=False)

    metric_rows = []
    prediction_frames = []
    optimum_frames = []
    if passed_wsd:
        for target_name in TARGETS:
            row, predictions, optima = audit_300m_target(target_name, args.seed)
            metric_rows.append(row)
            prediction_frames.append(predictions)
            optimum_frames.append(optima)
    metrics_300m = pd.DataFrame(metric_rows)
    metrics_300m.to_csv(args.output_dir / "metrics_300m.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            args.output_dir / "predictions_300m.csv",
            index=False,
        )
        pd.concat(optimum_frames, ignore_index=True).to_csv(
            args.output_dir / "optima_300m.csv",
            index=False,
        )
    gate = {
        "wsd_minimum_relative_improvement": WSD_MINIMUM_RELATIVE_IMPROVEMENT,
        "wsd_crs_nested_rmse": WSD_CRS_NESTED_RMSE,
        "wsd_maximum_optimum_distance": WSD_MAXIMUM_OPTIMUM_DISTANCE,
        "wsd_minimum_predicted_optimum_bpb": WSD_MINIMUM_PREDICTED_OPTIMUM_BPB,
        "maximum_selected_corpus_exponent": MAXIMUM_NU,
        "gates_300m": GATES_300M,
    }
    (args.output_dir / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report = write_report(wsd_metrics, passed_wsd, metrics_300m, args.output_dir)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
