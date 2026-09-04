# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pooled effective-data law for single-phase mixtures (round 4, after Sedova et al. 2026).

Prediction: y(w) = c + D_eff(w)^(-alpha) + sum_b gamma_b w_b, with
D_eff(w) = sum_b tau_b U_b(w) (1 + rho(E_b(w))), U_b = min(w_b, 1 / inventory_b) the unique share of the budget
the bucket contributes, E_b the materialized epochs, rho(E) = r1 (1 - exp(-(E - 1) / r1)) for E > 1 and 0
otherwise (repeated tokens count with diminishing value), tau_b >= 0 the data value of a bucket and gamma_b >= 0 a
linear share penalty. The scale of the power law is absorbed into tau. alpha, r1 and a ridge toward tau = 1 are
chosen by inner cross-validation (the ridge is a joint prior, tau toward 1 and gamma toward 0); tau, gamma and c by
bounded least squares. Concave pooling of the whole
effective-data total is the mechanism aimed at the successor's optimism far from the panel.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)

POOLED_REVISION = 4
POOLED_ALPHAS = (0.1, 0.3, 0.6, 1.0)
POOLED_REPETITION_SCALES = (2.0, 6.0, 20.0)
POOLED_RIDGE_GRID = (0.0, 1e-3, 1e-2)
POOLED_MAX_EVALUATIONS = 2000
POOLED_UNIQUE_SCALE = 10.0
EFFECTIVE_FLOOR = 1e-6


def repetition_credit(epochs: np.ndarray, scale: float) -> np.ndarray:
    excess = np.maximum(epochs - 1.0, 0.0)
    return scale * (1.0 - np.exp(-excess / scale))


def unique_share(features: models.Features) -> np.ndarray:
    return (
        np.minimum(features.weights, 1.0 / np.maximum(features.inventory, models.EPSILON)[None, :]) * POOLED_UNIQUE_SCALE
    )


def effective_terms(features: models.Features, scale: float) -> np.ndarray:
    """Per-bucket effective-data contributions before the data-value weights: U_b (1 + rho(E_b))."""
    return unique_share(features) * (1.0 + repetition_credit(features.exposures, scale))


def predict_rows(terms: np.ndarray, weights: np.ndarray, parameters: np.ndarray, alpha: float) -> np.ndarray:
    buckets = terms.shape[1]
    tau, gamma, intercept = parameters[:buckets], parameters[buckets : 2 * buckets], parameters[-1]
    effective = np.maximum(terms @ tau, EFFECTIVE_FLOOR)
    return intercept + effective ** (-alpha) + weights @ gamma


def residuals_and_jacobian(terms: np.ndarray, weights: np.ndarray, response: np.ndarray, alpha: float, ridge: float):
    """Residual and Jacobian callables of the bounded least-squares problem (data rows, then ridge rows)."""
    buckets = terms.shape[1]
    root_ridge = math.sqrt(ridge) if ridge > 0 else 0.0

    def residuals(parameters: np.ndarray) -> np.ndarray:
        fit = predict_rows(terms, weights, parameters, alpha) - response
        if root_ridge == 0.0:
            return fit
        return np.concatenate(
            [fit, root_ridge * (parameters[:buckets] - 1.0), root_ridge * parameters[buckets : 2 * buckets]]
        )

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        # d/dtau of effective^(-alpha) is -alpha effective^(-alpha - 1) terms; d/dgamma is the weights; d/dc is 1.
        effective = np.maximum(terms @ parameters[:buckets], EFFECTIVE_FLOOR)
        rows = np.hstack(
            [
                (-alpha * effective ** (-alpha - 1.0))[:, None] * terms,
                weights,
                np.ones((terms.shape[0], 1)),
            ]
        )
        if root_ridge == 0.0:
            return rows
        ridge_rows = np.zeros((2 * buckets, 2 * buckets + 1))
        ridge_rows[np.arange(2 * buckets), np.arange(2 * buckets)] = root_ridge
        return np.vstack([rows, ridge_rows])

    return residuals, jacobian


def fit_parameters(
    terms: np.ndarray, weights: np.ndarray, response: np.ndarray, alpha: float, ridge: float, start: np.ndarray | None
) -> tuple[np.ndarray, bool]:
    """Bounded least squares for (tau, gamma, intercept); the flag is the solver's own convergence status."""
    buckets = terms.shape[1]
    if start is None:
        spread = max(float(response.max() - response.min()), 1e-3)
        start = np.concatenate([np.full(buckets, 1.0), np.zeros(buckets), [float(response.min()) - spread]])
    residuals, jacobian = residuals_and_jacobian(terms, weights, response, alpha, ridge)

    lower = np.concatenate([np.zeros(2 * buckets), [-np.inf]])
    upper = np.full(2 * buckets + 1, np.inf)
    start = np.clip(start, lower, upper)
    # The jac-scaled trust-region solver can hit a non-converging SVD on wide (hundred-bucket) designs; fall back
    # to unit scaling and then to the dogbox method before giving up.
    fallback: tuple[float, np.ndarray] | None = None
    for options in ({"x_scale": "jac"}, {"x_scale": 1.0}, {"x_scale": 1.0, "method": "dogbox"}):
        try:
            result = least_squares(
                residuals, start, jac=jacobian, bounds=(lower, upper), max_nfev=POOLED_MAX_EVALUATIONS, **options
            )
        except np.linalg.LinAlgError:
            continue
        if not (np.isfinite(result.x).all() and np.isfinite(result.cost)):
            continue
        if result.success:
            return result.x, True
        if fallback is None or result.cost < fallback[0]:
            fallback = (float(result.cost), result.x)
    if fallback is None:
        raise ValueError("pooled effective-data fit did not converge")
    # Every solver exhausted its evaluation budget: keep the lowest-cost finite iterate and say so.
    return fallback[1], False


@dataclasses.dataclass(frozen=True)
class PooledEffectiveDataModel:
    """Concave pooled power law in effective data with linear share penalties; alpha, r1, ridge by inner CV."""

    model_id: str
    alphas: tuple[float, ...] = POOLED_ALPHAS
    repetition_scales: tuple[float, ...] = POOLED_REPETITION_SCALES
    ridge_grid: tuple[float, ...] = POOLED_RIDGE_GRID
    # The registry passes POOLED_REVISION explicitly: a non-default value is what enters the description hash,
    # because the harness hashes the models module, not this file. Bump POOLED_REVISION when the fit changes.
    revision: int = 0

    def fit(self, features: models.Features, response: np.ndarray, train: np.ndarray, inner, seed: int) -> models.Fitted:
        del seed
        best: tuple[float, float, float, float] | None = None
        for scale in self.repetition_scales:
            terms = effective_terms(features, scale)
            for alpha in self.alphas:
                for ridge in self.ridge_grid:
                    error, count = 0.0, 0
                    for inner_train, validation in inner:
                        parameters, _converged = fit_parameters(
                            terms[inner_train], features.weights[inner_train], response[inner_train], alpha, ridge, None
                        )
                        prediction = predict_rows(terms[validation], features.weights[validation], parameters, alpha)
                        if not np.isfinite(prediction).all():
                            error = float("inf")
                            break
                        error += float(np.sum((prediction - response[validation]) ** 2))
                        count += len(validation)
                    score = math.sqrt(error / max(count, 1)) if math.isfinite(error) else float("inf")
                    candidate = (score, scale, alpha, ridge)
                    if best is None or candidate < best:
                        best = candidate
        if best is None or not math.isfinite(best[0]):
            raise ValueError(f"{self.model_id}: no finite inner-CV candidate")
        score, scale, alpha, ridge = best
        terms = effective_terms(features, scale)
        parameters, converged = fit_parameters(
            terms[train], features.weights[train], response[train], alpha, ridge, None
        )
        buckets = features.buckets
        head = models.FittedHead(
            intercept=float(parameters[-1]),
            coefficients=parameters[:-1].copy(),
            floor=0.0,
            active=int(np.count_nonzero(parameters[:-1] > models.DSP_ACTIVE_TOL)),
        )
        return models.Fitted(
            shape={"alpha": alpha, "repetition_scale": scale},
            ridge=float(ridge),
            head=head,
            diagnostics={
                "inner_cv_rmse": score,
                "candidates": len(self.alphas) * len(self.repetition_scales) * len(self.ridge_grid),
                "converged": converged,
                "boundary_hits": (
                    int(alpha in (self.alphas[0], self.alphas[-1]))
                    + int(scale in (self.repetition_scales[0], self.repetition_scales[-1]))
                ),
                "effective_rank": 2 * buckets + 1,
                "columns": 2 * buckets + 1,
                "fitted_dof": head.active + 1 + 2,
                "nonlinear_dof": 2,
                "refine_evaluations": 0,
                "link": str(models.LinkKind.IDENTITY),
            },
        )

    def predict(self, fitted: models.Fitted, features: models.Features, rows: np.ndarray) -> np.ndarray:
        terms = effective_terms(features, float(fitted.shape["repetition_scale"]))
        parameters = np.concatenate([fitted.head.coefficients, [fitted.head.intercept]])
        return predict_rows(terms[rows], features.weights[rows], parameters, float(fitted.shape["alpha"]))

    def nonlinear_dof(self, features: models.Features) -> int:
        del features
        return 2

    def describe(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
