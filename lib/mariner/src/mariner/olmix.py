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
import importlib.util
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from mariner.model import WEIGHT_SUM_TOLERANCE, Objective, Swarm
from mariner.optimize import (
    MIXTURE_BLOCK_SIZE,
    block_maxima,
    bounded_simplex_violation,
    bucket_upper_bounds,
    check_cap_and_kl,
    proportional_mixture,
)

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

HUBER_DELTA = 0.01
FIT_STARTS = 48
FIT_SEED = 0
# The values of log_c that the starts share (fit_log_linear): how many, the fraction of the smallest target at the
# lowest, and the least exp(log_c) at either end.
LOG_C_STARTS = 6
LOWEST_START_FRACTION = 0.25
LOG_C_START_FLOOR = 1e-3
MAX_LOG = 50.0
SOLVERS = ("CLARABEL", "SCS")
SOLVER_STATUSES = ("optimal", "optimal_inaccurate")
# How far a solver's solution may lie outside the simplex and the caps: the accuracy cvxpy asks of SCS (eps_abs and
# eps_rel 1e-5). The released solves lie 1.1e-10 outside at most, and SCS alone solves them 3.8e-8 outside. The weights
# returned are that solution clipped at zero and renormalized, so they lie on the simplex and only a cap can be
# exceeded, by at most this much.
SOLUTION_TOLERANCE = 1e-5
# Slack in blocks for floating rounding: a block maximum or a weight this close below a whole block reaches it.
CAP_ROUNDING_SLACK = 1e-9
WEIGHT_ROUNDING_SLACK = 1e-12
CVXPY = "cvxpy"
CVXPY_MISSING = "the Olmix baseline needs cvxpy, which marin-mariner's olmix extra installs"


def require_cvxpy() -> None:
    """Raise ``ImportError`` naming the olmix extra when cvxpy is not installed.

    It finds the module without importing it, so that a command can check before any compute.
    """
    if importlib.util.find_spec(CVXPY) is None:
        raise ImportError(CVXPY_MISSING)


# ---------------------------------------------------------------------------------------------------------------
# Olmix baseline: positive log-linear law per task, exact KL-regularized proposer
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LogLinearLaw:
    """A task's Olmix law, which predicts ``exp(log_c) + exp(w @ coefficients)`` at weights ``w``.

    ``huber_loss`` is the fit's Huber loss over the runs it was fitted on. ``predict`` clips both exponents at -50
    and 50.
    """

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
    """The log-linear law of ``targets`` at ``weights`` with the least Huber loss (``delta``), by multistart L-BFGS-B.

    The starts take six values of ``log_c``, evenly spaced from log(max(min(y) / 4, 1e-3)) to log(max(median(y),
    1e-3)), each with zero coefficients and with ``starts // 6 - 1`` draws of standard normal coefficients from a
    generator seeded with ``seed``. The fit with the least loss wins; a start whose optimizer reports failure counts
    only while no fit has been kept.

    Raises:
        ValueError: If a target is not positive.
        RuntimeError: If no start reaches a finite loss.
    """
    x = np.asarray(weights, dtype=float)
    y = np.asarray(targets, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("the log-linear law needs positive targets")
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        logits = np.clip(x @ params[1:], -MAX_LOG, MAX_LOG)
        return huber_sum(np.exp(params[0]) + np.exp(logits) - y, delta)

    log_c_candidates = np.linspace(
        np.log(max(np.min(y) * LOWEST_START_FRACTION, LOG_C_START_FLOOR)),
        np.log(max(np.median(y), LOG_C_START_FLOOR)),
        LOG_C_STARTS,
    )
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


def constraint_violation(solution: np.ndarray, caps: np.ndarray) -> float:
    """How far ``solution`` lies outside the simplex and the ``caps`` (``optimize.bounded_simplex_violation``).

    Inf when a weight is not finite.
    """
    if not np.isfinite(solution).all():
        return np.inf
    return bounded_simplex_violation(solution, caps)


class ExactSolution(NamedTuple):
    """The exact proposer's weights, the laws' weighted prediction at them, and the solver that found them."""

    weights: np.ndarray
    predicted: float  # without the KL penalty
    solver: str  # "<solver>:<status>", e.g. "CLARABEL:optimal"


def exact_proposal(
    laws: Sequence[LogLinearLaw], task_weights: np.ndarray, natural: np.ndarray, caps: np.ndarray, kl_reg: float
) -> ExactSolution:
    """Olmix's exact proposer: the weights that minimize the laws' weighted prediction plus kl_reg * KL(w || natural).

    The weights range over the simplex with each at most its cap. The solvers are tried in ``SOLVERS`` order, and the
    first solution that a solver reports optimal (or optimal_inaccurate), that lies within ``SOLUTION_TOLERANCE`` of
    the constraints (``constraint_violation``) and that, clipped at zero and renormalized, exceeds no cap by more than
    ``SOLUTION_TOLERANCE`` is returned in that form, with the laws' weighted prediction at it without the KL term and
    the solver and its status.

    Raises:
        ImportError: If cvxpy, which the olmix extra installs, does not import.
        RuntimeError: If no solver returns such a solution; the message gives each solver's status and outcome, or the
            error it raised.
    """
    try:
        import cvxpy as cp  # noqa: PLC0415  (optional dependency: the olmix extra)
    except ImportError as error:
        raise ImportError(CVXPY_MISSING) from error

    # The sums are products with constant vectors, which give the solvers the same problem: cp.sum takes its result's
    # shape from np.sum over an uninitialized array, which warns whenever that memory holds values whose sum is invalid
    # (+inf with -inf, or a signaling NaN) or overflows, so a proposal's warnings would depend on what ran before it in
    # the process. A quiet NaN sums silently.
    ones = np.ones(len(natural))
    weights = cp.Variable(len(natural))
    prediction = sum(
        float(task_weights[i]) * (float(np.exp(law.log_c)) + cp.exp(law.coefficients @ weights))
        for i, law in enumerate(laws)
    )
    problem = cp.Problem(
        cp.Minimize(prediction + kl_reg * (ones @ cp.rel_entr(weights, natural))),
        [weights >= 0, ones @ weights == 1, weights <= caps],
    )
    attempts = []
    for solver in SOLVERS:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.error.SolverError as error:
            attempts.append(f"{solver} failed: {error}")
            continue
        if problem.status not in SOLVER_STATUSES:
            attempts.append(f"{solver} {problem.status}")
            continue
        raw = np.asarray(weights.value, float)
        violation = constraint_violation(raw, caps)
        if violation > SOLUTION_TOLERANCE:
            attempts.append(f"{solver} {problem.status} at a point {violation:.3g} outside the constraints")
            continue
        solution = np.clip(raw, 0.0, None)
        solution = solution / solution.sum()
        excess = float(np.max(solution - caps))
        if excess > SOLUTION_TOLERANCE:
            attempts.append(
                f"{solver} {problem.status} at a point that exceeds a cap by {excess:.3g} once clipped and renormalized"
            )
            continue
        predicted = sum(float(task_weights[i]) * float(law.predict(solution)[0]) for i, law in enumerate(laws))
        return ExactSolution(solution, predicted, f"{solver}:{problem.status}")
    raise RuntimeError(
        f"no solver found a solution within {SOLUTION_TOLERANCE:g} of the simplex and the caps: {'; '.join(attempts)}"
    )


# ---------------------------------------------------------------------------------------------------------------
# Proposals under an epoch cap, rounded to the sampler grid without exceeding the cap
# ---------------------------------------------------------------------------------------------------------------


def quantize_under_cap(weights: np.ndarray, epochs_per_unit_weight: np.ndarray, cap: float) -> np.ndarray:
    """Round a mixture to the 2,048-block grid without letting any bucket exceed ``cap`` epochs.

    Every bucket takes the floor of its blocks, at most the cap's block maximum; the blocks left over go to the
    buckets in order of decreasing weight, ties to the lower index, each up to its maximum. With an infinite ``cap`` no
    bucket is capped. Raises ``ValueError`` when a weight is negative, the weights do not sum to one within 1e-6, or
    the cap cannot fill the block.
    """
    if not ((weights >= 0).all() and abs(weights.sum() - 1.0) <= WEIGHT_SUM_TOLERANCE):
        raise ValueError(
            f"the weights must be nonnegative and sum to one within {WEIGHT_SUM_TOLERANCE:g}:"
            f" their minimum is {weights.min():g} and their sum {weights.sum():.12g}"
        )
    maximum = block_maxima(bucket_upper_bounds(epochs_per_unit_weight, cap, CAP_ROUNDING_SLACK), CAP_ROUNDING_SLACK)
    if maximum.sum() < MIXTURE_BLOCK_SIZE:
        raise ValueError(f"an epoch cap of {cap:g} holds {maximum.sum()} of the {MIXTURE_BLOCK_SIZE} blocks")
    # Weights that sum to one within the tolerance floor to at most the whole block, so remaining is never negative.
    counts = np.minimum(np.floor(weights * MIXTURE_BLOCK_SIZE + WEIGHT_ROUNDING_SLACK).astype(np.int64), maximum)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    for bucket in np.argsort(-weights, kind="stable"):
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
    predicted: float  # the laws' weighted prediction at the continuous solution, without the KL penalty
    solver: str  # "<solver>:<status>", e.g. "CLARABEL:optimal"


def propose(
    laws: Sequence[LogLinearLaw], objective: Objective, epochs_per_unit_weight: np.ndarray, cap: float, kl: float
) -> OlmixProposal:
    """Olmix's exact proposal under an epoch cap, rounded to the sampler grid by ``quantize_under_cap``.

    ``laws`` holds one law per task of ``objective``, in task order. The KL penalty, weighted by ``kl``, pulls toward
    the proportional mixture (1 / epochs) / sum(1 / epochs), and each bucket's weight is capped at min(1, cap / epochs).
    Raises ``ValueError`` before any solve when the laws do not match the tasks, ``kl`` is negative or not finite, or
    the cap is not positive (NaN included) or cannot hold a complete mixture on the 2,048-block grid
    (``optimize.check_cap_and_kl``).
    """
    if len(laws) != len(objective.tasks):
        raise ValueError(f"{len(laws)} laws for the {len(objective.tasks)} tasks of {objective.name}")
    check_cap_and_kl(epochs_per_unit_weight, cap, kl, CAP_ROUNDING_SLACK)
    natural = proportional_mixture(epochs_per_unit_weight)
    caps = bucket_upper_bounds(epochs_per_unit_weight, cap, CAP_ROUNDING_SLACK)
    solution = exact_proposal(laws, objective.weights, natural, caps, kl)
    counts = quantize_under_cap(solution.weights, epochs_per_unit_weight, cap)
    return OlmixProposal(counts, solution.weights, solution.predicted, solution.solver)
