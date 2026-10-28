# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Olmix baseline, as an independent implementation.

Each task gets a positive log-linear law fitted with a Huber loss by scipy multistart (not upstream's torch fitter).
The exact KL-regularized proposer solves the same convex program as allenai/olmix's exact proposer; it needs the
olmix extra, which provides cvxpy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

HUBER_DELTA = 0.01
FIT_STARTS = 48
FIT_SEED = 0
MAX_LOG = 50.0
SOLVERS = ("CLARABEL", "ECOS", "SCS")


# ---------------------------------------------------------------------------------------------------------------
# Olmix baseline: positive log-linear law per task, exact KL-regularized proposer
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LogLinearLaw:
    log_c: float
    coefficients: np.ndarray
    huber_loss: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        logits = np.clip(np.atleast_2d(weights) @ self.coefficients, -MAX_LOG, MAX_LOG)
        return np.exp(np.clip(self.log_c, -MAX_LOG, MAX_LOG)) + np.exp(logits)


def huber_sum(residuals: np.ndarray, delta: float) -> float:
    magnitude = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (magnitude - 0.5 * delta)
    return float(np.where(magnitude <= delta, quadratic, linear).sum())


def fit_log_linear(
    weights: np.ndarray,
    targets: np.ndarray,
    delta: float = HUBER_DELTA,
    seed: int = FIT_SEED,
    starts: int = FIT_STARTS,
) -> LogLinearLaw:
    x = np.asarray(weights, dtype=float)
    y = np.asarray(targets, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("the log-linear law needs positive targets")
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        logits = np.clip(x @ params[1:], -MAX_LOG, MAX_LOG)
        return huber_sum(np.exp(params[0]) + np.exp(logits) - y, delta)

    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    initial: list[np.ndarray] = []
    for log_c in log_c_candidates:
        initial.append(np.concatenate([[log_c], np.zeros(x.shape[1])]))
        for _ in range(max(starts // len(log_c_candidates) - 1, 0)):
            initial.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))
    best_params, best_loss = None, float("inf")
    bounds = [(-MAX_LOG, MAX_LOG), *[(None, None)] * x.shape[1]]
    for start in initial:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss, best_params = float(result.fun), np.asarray(result.x, dtype=float)
    if best_params is None:
        raise RuntimeError("log-linear fit failed")
    return LogLinearLaw(float(best_params[0]), best_params[1:], best_loss)


def exact_proposal(
    laws: Sequence[LogLinearLaw], task_weights: np.ndarray, natural: np.ndarray, caps: np.ndarray, kl_reg: float
) -> tuple[np.ndarray, float, str]:
    """Olmix's exact proposer: minimize the weighted mean prediction plus kl_reg * KL(w || natural) on the
    capped simplex."""
    import cvxpy as cp  # noqa: PLC0415  (optional dependency: only for the Olmix baseline)

    weights = cp.Variable(len(natural))
    predicted = sum(
        float(task_weights[i]) * (float(np.exp(law.log_c)) + cp.exp(cp.sum(cp.multiply(law.coefficients, weights))))
        for i, law in enumerate(laws)
    )
    problem = cp.Problem(
        cp.Minimize(predicted + kl_reg * cp.sum(cp.rel_entr(weights, natural))),
        [weights >= 0, cp.sum(weights) == 1, weights <= caps],
    )
    for solver in SOLVERS:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            solution = np.clip(np.asarray(weights.value, float), 0.0, None)
            return solution / solution.sum(), float(predicted.value), f"{solver}:{problem.status}"
    raise RuntimeError("no solver converged")
