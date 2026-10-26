# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Conditional-geometry raw-loss probe for the archived MARINER harm screen.

The raw-loss form is ``phi + exp(c - B @ alpha) + H @ beta``. Its matched
inside-exponential control is ``phi + exp(c - B @ alpha + H @ beta)``. Both use
raw-MSE fitting, nonnegative amplitudes, and a fixed floor multiplier of 1.5.
The caller supplies geometry
selected exclusively on the outer training rows. This is not a free-geometry
MARINER comparison: inner CV selects ridge conditional on that supplied geometry.
The floor gap and all numerical column scales are recomputed on each fit's own
training rows. A supplied proportional anchor/noise estimate must also exclude
outer test rows; when no anchor is supplied, the training median is used.

For n training rows the objective is mean(raw residual**2) plus ridge times
``||alpha * rms(B)||**2 + ||beta * rms(H)||**2``. The intercept is unpenalized.
Thus the objective does not silently change regularization with fold size.
Ridge is selected separately for this raw-loss objective; these numerical ridge
values are not equivalent to the canonical unnormalized log-loss ridge.
Across placements beta has different units, so equal coordinate penalties are
not an identical function-space prior; each placement selects its own ridge.

The exact exponential is evaluated without clipping or a training-dependent
prediction cap. Only the intercept has numerical bounds, chosen to leave room
for squaring float64 residuals. For the inside-exp control, the solver intercept
is q=c+max_train(H)@beta: its exponent is q-B@alpha-(max_train(H)-H)@beta.
This exact reparameterization keeps every training exponent below q. The stored
head uses the original c. Start termination, evaluations, runtime, and bound-hit
counts are retained. Failed starts are never scored as successful fits.
"""

from __future__ import annotations

import dataclasses
import math
import time
from enum import StrEnum

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as harm_models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)

FLOOR_MULTIPLIER = 1.5
START_COUNT = 2
MAX_SAFE_INTERCEPT = 0.5 * math.log(np.finfo(float).max) - 8.0
SOLVER_TOLERANCE = 1e-9


class RawHarm(StrEnum):
    SQUARED_LOG = "squared_log"
    UNSQUARED_LOG = "unsquared_log"
    CONVEX_LINEAR = "convex_linear"
    BOUNDED_RATIONAL_C1 = "bounded_rational_c1"


class HarmPlacement(StrEnum):
    RAW_LOSS = "raw_loss"
    LOG_DEFICIT = "log_deficit"


@dataclasses.dataclass(frozen=True)
class SolverRun:
    start: int
    converged: bool
    status: int
    nfev: int
    njev: int
    objective: float
    optimality: float
    elapsed_seconds: float
    intercept_bound_hit: bool
    exponential_underflow_rows: int
    failure: str = ""


@dataclasses.dataclass(frozen=True)
class RawLossHead:
    intercept: float
    alpha: np.ndarray
    beta: np.ndarray
    floor: float
    benefit_scale: np.ndarray
    harm_scale: np.ndarray
    selected_start: int
    runs: tuple[SolverRun, ...]
    placement: HarmPlacement
    inner_fits: tuple[RawLossFoldFit, ...] = ()
    search_trials: tuple[RawLossSearchTrial, ...] = ()


@dataclasses.dataclass(frozen=True)
class RawLossFoldFit:
    train: np.ndarray
    validation: np.ndarray
    head: RawLossHead
    prediction: np.ndarray


@dataclasses.dataclass(frozen=True)
class RawLossSearchTrial:
    ridge: float
    fold: int
    head: RawLossHead | None
    runs: tuple[SolverRun, ...]
    failure: str = ""


class RawLossFitError(RuntimeError):
    """Neither deterministic start produced a converged, finite fit."""

    def __init__(self, runs: tuple[SolverRun, ...], trials: tuple[RawLossSearchTrial, ...] = ()):
        super().__init__("No converged raw-loss start; inspect runs for termination diagnostics")
        self.runs = runs
        self.trials = trials


def basis(features: base.Features, shape: base.Shape, harm: RawHarm) -> tuple[np.ndarray, np.ndarray]:
    """Return the existing Weibull benefit and an E=0-centered harm basis."""
    if base.is_per_bucket_shape(shape):
        columns = []
        for index in range(features.buckets):
            bucket_features = dataclasses.replace(
                features,
                exposures=features.exposures[:, index : index + 1],
                weights=features.weights[:, index : index + 1],
                inventory=features.inventory[index : index + 1],
                early_fraction=features.early_fraction[index : index + 1],
                families=base.families_from_buckets((f"bucket_{index}",)),
                buckets_names=(f"bucket_{index}",),
            )
            bucket_shape = {key: shape[f"{key}:{index}"] for key in base.PER_BUCKET_SHAPE_KEYS}
            columns.append(basis(bucket_features, bucket_shape, harm))
        return np.concatenate([item[0] for item in columns], axis=1), np.concatenate(
            [item[1] for item in columns], axis=1
        )
    if harm is RawHarm.SQUARED_LOG:
        design = harm_models.log_harm_design(features, shape, (("harm_power", 2.0),))
    elif harm is RawHarm.UNSQUARED_LOG:
        design = harm_models.log_harm_design(features, shape, (("harm_power", 1.0),))
    elif harm is RawHarm.CONVEX_LINEAR:
        design = harm_models.convex_linear_design(features, shape, ())
    elif harm is RawHarm.BOUNDED_RATIONAL_C1:
        design = harm_models.rational_bounded_design(features, shape, (("cap", 1.0),))
    else:
        raise ValueError(f"Unsupported raw-loss harm: {harm}")
    benefit, damage = -design.values[:, : features.buckets], design.values[:, features.buckets :]
    if not np.isfinite(design.values).all() or np.any(benefit < 0.0) or np.any(damage < 0.0):
        raise ValueError("Raw-loss bases must be finite and nonnegative")
    return benefit, damage


def predict_head(head: RawLossHead, benefit: np.ndarray, harm: np.ndarray) -> np.ndarray:
    """Evaluate the exact raw-loss hypothesis; never cap predictions."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        signal = head.intercept - benefit @ head.alpha
        damage = harm @ head.beta
        if head.placement is HarmPlacement.LOG_DEFICIT:
            prediction = head.floor + np.exp(signal + damage)
        else:
            prediction = head.floor + np.exp(signal) + damage
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite raw-loss prediction")
    return prediction


def residual_and_jacobian(
    parameters: np.ndarray,
    benefit: np.ndarray,
    harm: np.ndarray,
    target: np.ndarray,
    floor: float,
    ridge: float,
    placement: HarmPlacement = HarmPlacement.RAW_LOSS,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw residual and analytic Jacobian, including row-normalized ridge."""
    buckets = benefit.shape[1]
    signal = parameters[0] - benefit @ parameters[1 : buckets + 1]
    damage = harm @ parameters[buckets + 1 :]
    if placement is HarmPlacement.LOG_DEFICIT:
        exponential = np.exp(signal + damage)
        prediction = floor + exponential
        harm_jacobian = exponential[:, None] * harm
    else:
        exponential = np.exp(signal)
        prediction = floor + exponential + damage
        harm_jacobian = harm
    root_rows = math.sqrt(len(target))
    residual = np.concatenate(((prediction - target) / root_rows, math.sqrt(ridge) * parameters[1:]))
    data_jacobian = np.column_stack((exponential, -exponential[:, None] * benefit, harm_jacobian)) / root_rows
    penalty_jacobian = np.column_stack((np.zeros(2 * buckets), math.sqrt(ridge) * np.eye(2 * buckets)))
    return residual, np.vstack((data_jacobian, penalty_jacobian))


def _initial_parameters(
    benefit: np.ndarray,
    harm: np.ndarray,
    response: np.ndarray,
    floor: float,
    ridge: float,
    placement: HarmPlacement,
) -> tuple[np.ndarray, np.ndarray]:
    buckets = benefit.shape[1]
    deficit = response - floor
    if np.any(deficit <= 0.0):
        raise ValueError("The calibrated floor must be strictly below every training response")
    design = base.Design(
        np.concatenate((-benefit, harm), axis=1),
        np.ones(2 * buckets),
        tuple(f"column_{index}" for index in range(2 * buckets)),
    )
    warm = base.fit_head(design, np.log(deficit), len(response) * ridge, base.HeadSpec())
    alpha = warm.coefficients[:buckets]
    beta = warm.coefficients[buckets:]
    if placement is HarmPlacement.LOG_DEFICIT:
        intercept = math.log(float(np.mean(deficit))) - float(
            logsumexp(-benefit @ alpha + harm @ beta) - math.log(len(response))
        )
    else:
        beta = beta * float(np.median(deficit))
        mean_harm = float(np.mean(harm @ beta))
        if mean_harm > 0.5 * float(np.mean(deficit)):
            beta *= 0.5 * float(np.mean(deficit)) / mean_harm
        remaining = float(np.mean(deficit - harm @ beta))
        intercept = math.log(remaining) - float(logsumexp(-benefit @ alpha) - math.log(len(response)))
    warm_start = np.concatenate(([intercept], alpha, beta))
    constant_start = np.concatenate(([math.log(float(np.mean(deficit)))], np.zeros(2 * buckets)))
    for start in (warm_start, constant_start):
        if abs(float(start[0])) >= MAX_SAFE_INTERCEPT:
            raise ValueError("Initial intercept is outside the float64 square-safe numerical domain")
    return warm_start, constant_start


def _fit_head(
    benefit: np.ndarray,
    harm: np.ndarray,
    response: np.ndarray,
    ridge: float,
    spec: base.HeadSpec,
    max_nfev: int,
    placement: HarmPlacement,
) -> RawLossHead:
    anchor, gap = base.floor_anchor_and_gap(response, spec)
    floor = base.fitted_floor(FLOOR_MULTIPLIER, anchor, gap, spec)
    benefit_scale = np.sqrt(np.mean(benefit**2, axis=0))
    harm_scale = np.sqrt(np.mean(harm**2, axis=0))
    benefit_scale = np.where(benefit_scale > 0.0, benefit_scale, 1.0)
    harm_scale = np.where(harm_scale > 0.0, harm_scale, 1.0)
    normalized_benefit, normalized_harm = benefit / benefit_scale, harm / harm_scale
    starts = _initial_parameters(normalized_benefit, normalized_harm, response, floor, ridge, placement)
    harm_origin = np.zeros(harm.shape[1])
    if placement is HarmPlacement.LOG_DEFICIT:
        harm_origin = np.max(normalized_harm, axis=0)
        normalized_harm = normalized_harm - harm_origin
        for start in starts:
            start[0] += float(harm_origin @ start[benefit.shape[1] + 1 :])
    lower = np.concatenate(([-MAX_SAFE_INTERCEPT], np.zeros(2 * benefit.shape[1])))
    upper = np.concatenate(([MAX_SAFE_INTERCEPT], np.full(2 * benefit.shape[1], np.inf)))
    runs = []
    candidates = []
    for start_index, initial in enumerate(starts):
        started = time.perf_counter()
        try:
            result = least_squares(
                lambda value: residual_and_jacobian(
                    value, normalized_benefit, normalized_harm, response, floor, ridge, placement
                )[0],
                initial,
                jac=lambda value: residual_and_jacobian(
                    value, normalized_benefit, normalized_harm, response, floor, ridge, placement
                )[1],
                bounds=(lower, upper),
                max_nfev=max_nfev,
                ftol=SOLVER_TOLERANCE,
                xtol=SOLVER_TOLERANCE,
                gtol=SOLVER_TOLERANCE,
            )
        except (FloatingPointError, ValueError) as error:
            runs.append(
                SolverRun(
                    start_index,
                    False,
                    -99,
                    0,
                    0,
                    math.inf,
                    math.inf,
                    time.perf_counter() - started,
                    False,
                    0,
                    str(error),
                )
            )
            continue
        objective = float(2.0 * result.cost)
        converged = bool(result.success and np.isfinite(objective) and np.isfinite(result.x).all())
        exponent = result.x[0] - normalized_benefit @ result.x[1 : benefit.shape[1] + 1]
        if placement is HarmPlacement.LOG_DEFICIT:
            exponent += normalized_harm @ result.x[benefit.shape[1] + 1 :]
        runs.append(
            SolverRun(
                start_index,
                converged,
                int(result.status),
                int(result.nfev),
                int(result.njev or 0),
                objective,
                float(result.optimality),
                time.perf_counter() - started,
                bool(result.active_mask[0]),
                int(np.count_nonzero(exponent < math.log(np.nextafter(0.0, 1.0)))),
                "" if converged else str(result.message),
            )
        )
        if converged:
            candidates.append((objective, start_index, result.x))
    if not candidates:
        raise RawLossFitError(tuple(runs))
    _, selected_start, parameters = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    buckets = benefit.shape[1]
    return RawLossHead(
        float(parameters[0] - harm_origin @ parameters[buckets + 1 :]),
        parameters[1 : buckets + 1] / benefit_scale,
        parameters[buckets + 1 :] / harm_scale,
        floor,
        benefit_scale,
        harm_scale,
        selected_start,
        tuple(runs),
        placement,
    )


def _diagnostics(runs: tuple[SolverRun, ...], elapsed: float, seed: int) -> dict[str, float | int | bool | str]:
    return {
        "comparison_scope": "conditional_outer_train_geometry",
        "objective": "raw_mse_plus_ridge_rms_amplitudes",
        "floor_multiplier": FLOOR_MULTIPLIER,
        "prediction_link": "exact_exp_no_clipping_or_cap",
        "harm_origin": "E0_zero; rational_transform_after_base_centering",
        "deterministic_starts_per_fit": START_COUNT,
        "solver_start_count": len(runs),
        "solver_converged_count": sum(run.converged for run in runs),
        "solver_failed_count": sum(not run.converged for run in runs),
        "solver_nfev": sum(run.nfev for run in runs),
        "solver_njev": sum(run.njev for run in runs),
        "intercept_bound_hits": sum(run.intercept_bound_hit for run in runs),
        "exponential_underflow_rows": sum(run.exponential_underflow_rows for run in runs),
        "runtime_seconds": elapsed,
        "seed": seed,
    }


@dataclasses.dataclass(frozen=True)
class RawLossProbe:
    harm: RawHarm
    head_spec: base.HeadSpec
    ridge_grid: tuple[float, ...]
    max_nfev: int = 300
    placement: HarmPlacement = HarmPlacement.RAW_LOSS

    @property
    def model_id(self) -> str:
        return f"hs17_{self.placement.value}_{self.harm.value}"

    def predict(self, fitted: base.Fitted, features: base.Features, rows: np.ndarray) -> np.ndarray:
        benefit, harm = basis(features, fitted.shape, self.harm)
        return predict_head(fitted.head, benefit[rows], harm[rows])

    def fit_at_ridge(
        self,
        features: base.Features,
        response: np.ndarray,
        train: np.ndarray,
        shape: base.Shape,
        ridge: float,
        seed: int,
    ) -> base.Fitted:
        """Fit one fixed ridge, recomputing floor calibration from these training rows."""
        if ridge < 0.0 or not np.isfinite(ridge):
            raise ValueError("Ridge must be finite and nonnegative")
        started = time.perf_counter()
        benefit, harm = basis(features, shape, self.harm)
        head = _fit_head(
            benefit[train], harm[train], response[train], ridge, self.head_spec, self.max_nfev, self.placement
        )
        diagnostics = _diagnostics(head.runs, time.perf_counter() - started, seed)
        diagnostics.update({"harm": self.harm.value, "harm_placement": self.placement.value})
        return base.Fitted(dict(shape), ridge, head, diagnostics)

    def fit(
        self,
        features: base.Features,
        response: np.ndarray,
        train: np.ndarray,
        inner: base.InnerFolds,
        shape: base.Shape,
        seed: int,
    ) -> base.Fitted:
        """Select this form's ridge by three-fold raw-loss CV at fixed geometry."""
        if len(inner) != 3 or not self.ridge_grid:
            raise ValueError("The screen requires three inner folds and a nonempty ridge grid")
        for fit_rows, validation in inner:
            if (
                not len(fit_rows)
                or not len(validation)
                or np.intersect1d(fit_rows, validation).size
                or not np.isin(fit_rows, train).all()
                or not np.isin(validation, train).all()
            ):
                raise ValueError("Inner training and validation rows must be disjoint subsets of outer train")
        started = time.perf_counter()
        benefit, harm = basis(features, shape, self.harm)
        scores = []
        ridge_folds = []
        all_runs = []
        search_trials = []
        for ridge in self.ridge_grid:
            if ridge < 0.0 or not np.isfinite(ridge):
                raise ValueError("Ridge must be finite and nonnegative")
            folds = []
            failed = False
            for fold_index, (fit_rows, validation) in enumerate(inner):
                try:
                    head = _fit_head(
                        benefit[fit_rows],
                        harm[fit_rows],
                        response[fit_rows],
                        ridge,
                        self.head_spec,
                        self.max_nfev,
                        self.placement,
                    )
                except RawLossFitError as error:
                    all_runs.extend(error.runs)
                    search_trials.append(RawLossSearchTrial(ridge, fold_index, None, error.runs))
                    failed = True
                    continue
                all_runs.extend(head.runs)
                try:
                    prediction = predict_head(head, benefit[validation], harm[validation])
                except FloatingPointError as error:
                    search_trials.append(RawLossSearchTrial(ridge, fold_index, head, head.runs, str(error)))
                    failed = True
                    continue
                search_trials.append(RawLossSearchTrial(ridge, fold_index, head, head.runs))
                folds.append(RawLossFoldFit(fit_rows.copy(), validation.copy(), head, prediction))
            score = (
                math.inf
                if failed
                else math.sqrt(
                    sum(float(np.sum((fold.prediction - response[fold.validation]) ** 2)) for fold in folds)
                    / sum(len(fold.validation) for fold in folds)
                )
            )
            scores.append(score)
            ridge_folds.append(tuple(folds))
        selected = int(np.argmin(scores))
        if not np.isfinite(scores[selected]):
            raise RawLossFitError(tuple(all_runs), tuple(search_trials))
        ridge = self.ridge_grid[selected]
        try:
            final_head = _fit_head(
                benefit[train], harm[train], response[train], ridge, self.head_spec, self.max_nfev, self.placement
            )
        except RawLossFitError as error:
            raise RawLossFitError((*all_runs, *error.runs), tuple(search_trials)) from error
        all_runs.extend(final_head.runs)
        final_head = dataclasses.replace(
            final_head, inner_fits=ridge_folds[selected], search_trials=tuple(search_trials)
        )
        reconstructed = math.sqrt(
            sum(
                float(
                    np.sum(
                        (
                            predict_head(fold.head, benefit[fold.validation], harm[fold.validation])
                            - response[fold.validation]
                        )
                        ** 2
                    )
                )
                for fold in final_head.inner_fits
            )
            / sum(len(fold.validation) for fold in final_head.inner_fits)
        )
        diagnostics = _diagnostics(tuple(all_runs), time.perf_counter() - started, seed)
        diagnostics.update(
            {
                "inner_rmse": scores[selected],
                "inner_reconstructed_rmse": reconstructed,
                "inner_score_reconstruction_error": abs(scores[selected] - reconstructed),
                "ridge_candidates": len(self.ridge_grid),
                "invalid_ridge_candidates": sum(not math.isfinite(s) for s in scores),
                "harm": self.harm.value,
                "harm_placement": self.placement.value,
            }
        )
        return base.Fitted(dict(shape), ridge, final_head, diagnostics, np.asarray([scores]))
