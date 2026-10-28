# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mixture optimization of a fitted objective.

The aggregate sum_t a_t f_t(w) is minimized by SLSQP over the simplex from several feasible starts (the proportional
mixture and four seeded swarm rows), polished, then rounded to the sampler grid of 2,048 blocks with an exchange
search. SLSQP finds a local minimizer; the multistart is the only guard against local optima. An epoch cap and a KL
penalty toward the proportional mixture are optional and not part of the procedure: the reported optima are
unconstrained.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from mariner.model import ObjectiveFit

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

START_SEED = 20_260_906
PANEL_STARTS = 4
OPTIMIZER_OPTIONS = {"maxiter": 500, "ftol": 1e-11, "eps": 1e-8}
POLISH_OPTIONS = {"maxiter": 2000, "ftol": 1e-10, "eps": 1e-6}
PROJECTION_OPTIONS = {"maxiter": 200, "ftol": 1e-13}
FEASIBILITY_TOLERANCE = 1e-7
ENDPOINT_FEASIBILITY_TOLERANCE = 10 * FEASIBILITY_TOLERANCE
MIXTURE_BLOCK_SIZE = 2_048
MAX_EXCHANGE_STEPS = 4 * MIXTURE_BLOCK_SIZE
REFINE_TOLERANCE = 1e-13
# Slack in blocks for floating rounding: a cap's block maximum this close below a whole block reaches it.
BLOCK_MAXIMUM_SLACK = 1e-12


# ---------------------------------------------------------------------------------------------------------------
# Mixture optimization: SLSQP from feasible starts, then rounding to the sampler grid
# ---------------------------------------------------------------------------------------------------------------


def project_start(weights: np.ndarray, upper: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the capped simplex by a convex quadratic solve."""
    if np.max(weights - upper) <= 1e-12 and weights.min() >= 0 and abs(weights.sum() - 1) <= 1e-12:
        return weights.copy()
    result = minimize(
        lambda candidate: 0.5 * float(np.square(candidate - weights).sum()),
        feasible,
        jac=lambda candidate: candidate - weights,
        bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
        constraints={"type": "eq", "fun": lambda c: c.sum() - 1, "jac": lambda c: np.ones_like(c)},
        method="SLSQP",
        options=PROJECTION_OPTIONS,
    )
    if not result.success or max(abs(result.x.sum() - 1), -result.x.min(), np.max(result.x - upper)) > 1e-10:
        raise RuntimeError(f"start projection failed: {result.message}")
    return result.x


def categorical_kl(weights: np.ndarray, natural: np.ndarray) -> float:
    return float(xlogy(weights, weights / natural).sum())


def optimize_start(
    predict: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    natural: np.ndarray,
    upper: np.ndarray,
    kl: float,
    options: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    def objective(weights: np.ndarray) -> float:
        return float(predict(weights[None])[0]) + kl * categorical_kl(weights, natural)

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options=options,
    )
    violation = float(max(abs(result.x.sum() - 1), -result.x.min(), np.max(result.x - upper)))
    weights = project_start(result.x, upper, natural)
    value = objective(weights)
    if not np.isfinite(value):
        raise RuntimeError("continuous optimizer produced a nonfinite objective")
    return weights, {
        "objective": value,
        "raw_feasibility_violation": violation,
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def proportional_mixture(epochs_per_unit_weight: np.ndarray) -> np.ndarray:
    """The mixture in proportion to the buckets' pools: (1 / epochs) / sum(1 / epochs)."""
    natural = 1.0 / epochs_per_unit_weight
    return natural / natural.sum()


def bucket_upper_bounds(epochs_per_unit_weight: np.ndarray, cap: float | None) -> np.ndarray:
    """Per-bucket weight bounds: the simplex alone (``cap`` None, the procedure) or an epoch cap, min(1, cap / epochs).

    Raises:
        ValueError: If the cap's bounds sum to less than one, so that no complete mixture satisfies them.
    """
    if cap is None:
        return np.ones(len(epochs_per_unit_weight))
    upper = np.minimum(1.0, cap / epochs_per_unit_weight)
    if upper.sum() < 1:
        raise ValueError(f"an epoch cap of {cap:g} cannot hold a complete mixture")
    return upper


def block_maxima(upper: np.ndarray, slack: float) -> np.ndarray:
    """The most blocks of the 2,048-block grid within the weight bounds ``upper``: floor(2,048 upper + ``slack``)."""
    return np.floor(upper * MIXTURE_BLOCK_SIZE + slack).astype(np.int64)


def check_cap_and_kl(epochs_per_unit_weight: np.ndarray, cap: float | None, kl: float, slack: float) -> None:
    """Raise ``ValueError`` unless the epoch cap and the KL weight admit a proposal; proposers call it before solving.

    ``kl`` must be finite and nonnegative. ``cap`` is None (no cap) or a number of epochs, inf included, whose weight
    bounds (``bucket_upper_bounds``) sum to at least one and whose block maxima, computed as the rounding computes
    them with its ``slack`` (``block_maxima``), hold the 2,048 blocks of a mixture.
    """
    if not (np.isfinite(kl) and kl >= 0.0):
        raise ValueError(f"the KL weight must be finite and nonnegative, got {kl:g}")
    if cap is None:
        return
    if np.isnan(cap):
        raise ValueError(f"the epoch cap must be a positive number of epochs or inf, got {cap:g}")
    blocks = int(block_maxima(bucket_upper_bounds(epochs_per_unit_weight, cap), slack).sum())
    if blocks < MIXTURE_BLOCK_SIZE:
        raise ValueError(f"an epoch cap of {cap:g} holds {blocks} of the {MIXTURE_BLOCK_SIZE} blocks")


def continuous_optimum(
    fit: ObjectiveFit, swarm_weights: np.ndarray, cap: float | None = None, kl: float = 0.0
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The lowest feasible SLSQP endpoint over the proportional start and four seeded swarm rows, polished."""
    natural = proportional_mixture(fit.epochs_per_unit_weight)
    upper = bucket_upper_bounds(fit.epochs_per_unit_weight, cap)
    indices = np.random.default_rng(START_SEED).choice(len(swarm_weights), PANEL_STARTS, replace=False)
    starts = [project_start(row, upper, natural) for row in (natural, *swarm_weights[indices])]
    records, solutions = [], []
    for index, start in enumerate(starts):
        weights, diagnostics = optimize_start(fit.predict, start, natural, upper, kl, OPTIMIZER_OPTIONS)
        records.append({"start": index, **diagnostics})
        solutions.append(weights)
    for index, start in enumerate(list(solutions)):
        weights, diagnostics = optimize_start(fit.predict, start, natural, upper, kl, POLISH_OPTIONS)
        if diagnostics["objective"] <= records[index]["objective"]:
            solutions[index] = weights
            records[index] = {"start": index, **diagnostics, "polished": True}
    feasible = [i for i, row in enumerate(records) if row["raw_feasibility_violation"] <= ENDPOINT_FEASIBILITY_TOLERANCE]
    if not feasible:
        raise RuntimeError("no restart ended feasibly")
    best = min(feasible, key=lambda i: records[i]["objective"])
    for row in records:
        row["selected"] = row["start"] == best
    return solutions[best], records


def constrained_counts(weights: np.ndarray, maximum_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(weights, dtype=float) * MIXTURE_BLOCK_SIZE
    counts = np.minimum(np.floor(target).astype(np.int64), maximum_counts)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    while remaining:
        available = np.flatnonzero(counts < maximum_counts)
        if len(available) == 0:
            raise ValueError("the epoch cap cannot support a complete mixture block")
        deficit = target[available] - counts[available]
        counts[int(available[int(np.argmax(deficit))])] += 1
        remaining -= 1
    return counts


def refine_counts(
    predict: Callable[[np.ndarray], np.ndarray], initial: np.ndarray, maximum: np.ndarray
) -> tuple[np.ndarray, int]:
    """Improving one-block exchanges on the sampler grid until none lowers the objective."""
    current = np.asarray(initial, dtype=np.int64).copy()
    current_value = float(predict((current / MIXTURE_BLOCK_SIZE)[None, :])[0])
    for step in range(MAX_EXCHANGE_STEPS):
        donors = np.flatnonzero(current > 0)
        receivers = np.flatnonzero(current < maximum)
        proposals, moves = [], []
        for donor in donors:
            for receiver in receivers:
                if donor == receiver:
                    continue
                proposal = current.copy()
                proposal[donor] -= 1
                proposal[receiver] += 1
                proposals.append(proposal)
                moves.append((donor, receiver))
        if not proposals:
            return current, step
        values = predict(np.asarray(proposals, dtype=float) / MIXTURE_BLOCK_SIZE)
        choice = int(np.argmin(values))
        if float(values[choice]) >= current_value - REFINE_TOLERANCE:
            return current, step
        donor, receiver = moves[choice]
        current[donor] -= 1
        current[receiver] += 1
        current_value = float(values[choice])
    raise RuntimeError(f"exchange refinement exceeded {MAX_EXCHANGE_STEPS} steps")


def runtime_policy(
    fit: ObjectiveFit, weights: np.ndarray, cap: float | None = None, kl: float = 0.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Round a continuous optimum to block counts (under the optional cap) and refine; returns counts and a summary.

    The exchange search minimizes the objective plus the KL penalty; the summary reports the fit's predictions,
    without the penalty, at the continuous optimum and at the counts' mixture.
    """
    maximum = block_maxima(bucket_upper_bounds(fit.epochs_per_unit_weight, cap), BLOCK_MAXIMUM_SLACK)
    natural = proportional_mixture(fit.epochs_per_unit_weight)

    def predict(rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(rows)
        value = fit.predict(rows)
        if kl:
            safe = np.maximum(rows, 1e-300)
            value = value + kl * np.sum(np.where(rows > 0, rows * np.log(safe / natural[None, :]), 0.0), axis=1)
        return value

    initial = constrained_counts(weights, maximum)
    counts, steps = refine_counts(predict, initial, maximum)
    runtime = counts / MIXTURE_BLOCK_SIZE
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE or counts.min() < 0 or np.any(counts > maximum):
        raise RuntimeError("invalid refined runtime allocation")
    return counts, {
        "continuous_prediction": float(fit.predict(weights)[0]),
        "runtime_prediction": float(fit.predict(runtime)[0]),
        "exchange_steps": float(steps),
        "max_materialized_epoch": float((runtime * fit.epochs_per_unit_weight).max()),
    }


@dataclasses.dataclass(frozen=True)
class Proposal:
    """A proposed mixture on the sampler grid, the continuous optimum it rounds, and the optimizer's diagnostics."""

    counts: np.ndarray  # (buckets,) blocks of the 2,048-block grid
    continuous: np.ndarray  # (buckets,) the continuous optimum
    summary: dict[str, float]  # fit's predictions before and after rounding, exchange steps, largest materialized epoch
    starts: list[dict[str, Any]]  # one record per SLSQP start

    @property
    def weights(self) -> np.ndarray:
        return self.counts / MIXTURE_BLOCK_SIZE


def propose(fit: ObjectiveFit, swarm_weights: np.ndarray, cap: float | None = None, kl: float = 0.0) -> Proposal:
    """The fitted objective's optimum (``continuous_optimum``) rounded to the sampler grid (``runtime_policy``).

    ``cap`` bounds every bucket's materialized epochs and ``kl`` weights a KL penalty toward the proportional
    mixture; both are optional and not part of the procedure. Raises ``ValueError`` before any solve when ``kl`` is
    negative or not finite, or the cap is NaN or cannot hold a complete mixture on the 2,048-block grid
    (``check_cap_and_kl``).
    """
    check_cap_and_kl(fit.epochs_per_unit_weight, cap, kl, BLOCK_MAXIMUM_SLACK)
    continuous, starts = continuous_optimum(fit, swarm_weights, cap, kl)
    counts, summary = runtime_policy(fit, continuous, cap, kl)
    return Proposal(counts, continuous, summary, starts)
