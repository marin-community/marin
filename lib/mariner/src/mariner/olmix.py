# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Olmix baseline, as an independent implementation.

Each task gets a positive log-linear law fitted with a Huber loss by scipy multistart (not upstream's torch fitter).
The exact KL-regularized proposer solves the same convex program as allenai/olmix's exact proposer; it needs the
olmix extra, which provides cvxpy. A proposal under an epoch cap is rounded to the sampler's 2,048-block grid without
letting any bucket exceed the cap, as the matched Olmix mixtures the paper trained were.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from mariner.model import Objective, Swarm
from mariner.optimize import MIXTURE_BLOCK_SIZE

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

HUBER_DELTA = 0.01
FIT_STARTS = 48
FIT_SEED = 0
MAX_LOG = 50.0
SOLVERS = ("CLARABEL", "ECOS", "SCS")
# Slack in blocks for floating rounding: a block maximum or a weight this close below a whole block reaches it.
CAP_ROUNDING_SLACK = 1e-9
WEIGHT_ROUNDING_SLACK = 1e-12


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


def fit_laws(swarm: Swarm, objective: Objective, delta: float = HUBER_DELTA) -> tuple[LogLinearLaw, ...]:
    """One law per task of ``objective``, in task order, fitted on every run of the swarm, the calibration run included.

    The laws are functions of the weights alone, so pool fractions do not enter.
    """
    return tuple(fit_log_linear(swarm.weights, swarm.outcomes[task].to_numpy(float), delta) for task in objective.tasks)


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


# ---------------------------------------------------------------------------------------------------------------
# Proposals under an epoch cap, rounded to the sampler grid without exceeding the cap
# ---------------------------------------------------------------------------------------------------------------


def quantize_under_cap(weights: np.ndarray, epochs_per_unit_weight: np.ndarray, cap: float) -> np.ndarray:
    """Round a mixture to the 2,048-block grid without letting any bucket exceed ``cap`` epochs.

    Every bucket takes the floor of its blocks, at most the cap's block maximum; the blocks left over go to the
    buckets in order of decreasing weight, each up to its maximum. Raises ``ValueError`` when the cap cannot fill
    the block or the weights sum to more than one.
    """
    maximum = np.floor(cap / epochs_per_unit_weight * MIXTURE_BLOCK_SIZE + CAP_ROUNDING_SLACK).astype(np.int64)
    maximum = np.minimum(maximum, MIXTURE_BLOCK_SIZE)
    if maximum.sum() < MIXTURE_BLOCK_SIZE:
        raise ValueError(f"an epoch cap of {cap:g} holds {maximum.sum()} of the {MIXTURE_BLOCK_SIZE} blocks")
    counts = np.minimum(np.floor(weights * MIXTURE_BLOCK_SIZE + WEIGHT_ROUNDING_SLACK).astype(np.int64), maximum)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    if remaining < 0:
        raise ValueError(f"the weights sum to {weights.sum():.12g}, more than one")
    for bucket in np.argsort(-weights):
        if remaining == 0:
            break
        added = min(int(maximum[bucket] - counts[bucket]), remaining)
        counts[bucket] += added
        remaining -= added
    return counts


@dataclasses.dataclass(frozen=True)
class OlmixProposal:
    """An Olmix mixture on the sampler grid, the exact proposer's solution it rounds, and the solver that found it."""

    counts: np.ndarray  # (buckets,) blocks of the 2,048-block grid
    continuous: np.ndarray  # (buckets,) the exact proposer's solution
    predicted: float  # the laws' objective at the continuous solution, without the KL penalty
    solver: str  # "<solver>:<status>", e.g. "CLARABEL:optimal"


def propose(
    laws: Sequence[LogLinearLaw], objective: Objective, epochs_per_unit_weight: np.ndarray, cap: float, kl: float
) -> OlmixProposal:
    """Olmix's exact proposal under an epoch cap, rounded to the sampler grid by ``quantize_under_cap``.

    ``laws`` holds one law per task of ``objective``, in task order. The KL penalty, weighted by ``kl``, pulls toward
    the proportional mixture (1 / epochs) / sum(1 / epochs), and each bucket's weight is capped at min(1, cap / epochs).
    Raises ``ValueError`` when the laws do not match the tasks or the cap cannot hold a complete mixture.
    """
    if len(laws) != len(objective.tasks):
        raise ValueError(f"{len(laws)} laws for the {len(objective.tasks)} tasks of {objective.name}")
    natural = 1.0 / epochs_per_unit_weight
    natural = natural / natural.sum()
    caps = np.minimum(1.0, cap / epochs_per_unit_weight)
    if caps.sum() < 1:
        raise ValueError(f"an epoch cap of {cap:g} cannot hold a complete mixture")
    continuous, predicted, solver = exact_proposal(laws, objective.weights, natural, caps, kl)
    return OlmixProposal(quantize_under_cap(continuous, epochs_per_unit_weight, cap), continuous, predicted, solver)
