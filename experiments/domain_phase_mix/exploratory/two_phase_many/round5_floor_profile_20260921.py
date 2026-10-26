# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Training-local BPB floor profiling along unchanged native log-head optima.

Each gamma uses the native training-local floor, log-deficit squared loss and
raw coefficient ridge. The scalar floor is ranked by unpenalized training BPB
SSE. This is a nested estimator, not joint BPB least squares or a likelihood
profile. Every actual fitting subset must call this function independently.

The finite search includes every registered coarse point, both end intervals
and every sampled interior minimum bracket. It is not a global certificate.
"""

import dataclasses
import functools
import math
from typing import Any

import numpy as np
import round3_log_head_20260921 as log_head
from scipy.optimize import minimize_scalar

mariner = log_head.mariner
GAMMA_GRID = tuple(float(value) for value in np.linspace(1.0, 6.0, 21))
REFINEMENT_XATOL = 1e-6
REFINEMENT_MAXITER = 48


@dataclasses.dataclass(frozen=True)
class ProfileTrial:
    gamma: float
    floor: float
    training_sse: float
    deficit_clamp_count: int
    stage: str
    refinement_index: int | None
    cached: bool


@dataclasses.dataclass(frozen=True)
class Refinement:
    lower_gamma: float
    upper_gamma: float
    gamma: float
    training_sse: float
    success: bool
    status: int
    message: str
    evaluations: int


@dataclasses.dataclass(frozen=True)
class ProfileFit:
    head: log_head.Head
    gamma: float
    training_sse: float
    lower_floor: float
    upper_floor: float
    bound_hits: tuple[str, ...]
    training_sse_spread: float
    constant_response: bool
    deficit_clamp_count: int
    trace: tuple[ProfileTrial, ...]
    refinements: tuple[Refinement, ...]

    def to_json(self) -> dict[str, Any]:
        """Serialize floor-search metadata; the caller separately saves the head."""
        return {
            "gamma": self.gamma,
            "floor": self.head.floor,
            "training_sse": self.training_sse,
            "native_floor_envelope": [self.lower_floor, self.upper_floor],
            "gamma_bounds": [GAMMA_GRID[0], GAMMA_GRID[-1]],
            "bound_hits": list(self.bound_hits),
            "sampled_training_sse_spread": self.training_sse_spread,
            "constant_response": self.constant_response,
            "floor_identification": "not_identified_constant_response" if self.constant_response else "not_certified",
            "deficit_clamp_count": self.deficit_clamp_count,
            "conditional_solves": sum(not trial.cached for trial in self.trace),
            "trace": [dataclasses.asdict(trial) for trial in self.trace],
            "refinements": [dataclasses.asdict(refinement) for refinement in self.refinements],
            "search_claim": "Finite sampled-basin search; no global optimality or floor identification certificate",
        }


def fit_profile_head(matrix: np.ndarray, response: np.ndarray, ridge: float) -> ProfileFit:
    """Fit one native log head while estimating its floor from its own labels.

    Args:
        matrix: Signed benefit/harm columns; all fitted amplitudes are nonnegative.
        response: Positive finite BPB observations from this actual training subset.
        ridge: Native raw-coefficient ridge multiplier, held fixed during profiling.
    """
    matrix = np.asarray(matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    if matrix.ndim != 2 or response.ndim != 1 or not len(response) or matrix.shape[0] != len(response):
        raise ValueError("Features and positive BPB response must have the same nonzero row count")
    if not np.isfinite(matrix).all() or not np.isfinite(response).all() or np.any(response <= 0):
        raise ValueError("Features and BPB observations must be finite, with BPB positive")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Native raw ridge must be finite and nonnegative")
    empty_phase = np.empty((len(response), 0))
    aggregate_penalty, phase_penalty = np.ones(matrix.shape[1]), np.empty(0)
    cache: dict[float, tuple[log_head.Head, float, int]] = {}
    trace = []

    def evaluate(gamma: float, stage: str, refinement_index: int | None) -> float:
        gamma = float(gamma)
        if not GAMMA_GRID[0] <= gamma <= GAMMA_GRID[-1]:
            raise ValueError("Floor search left the registered native gamma envelope")
        cached = gamma in cache
        if not cached:
            head = log_head.fit_log_head(
                matrix,
                empty_phase,
                response,
                ridge,
                mariner.FloorSpec(float("nan"), 0.0, gamma),
                aggregate_penalty,
                phase_penalty,
            )
            prediction = head.predict(matrix, empty_phase)
            score = float(np.sum((prediction - response) ** 2))
            if not math.isfinite(score):
                raise FloatingPointError("A conditional head produced nonfinite training BPB error")
            clamps = int(np.count_nonzero(response - head.floor < mariner.DEFICIT_FLOOR))
            cache[gamma] = head, score, clamps
        head, score, clamps = cache[gamma]
        trace.append(ProfileTrial(gamma, head.floor, score, clamps, stage, refinement_index, cached))
        return score

    coarse = [evaluate(gamma, "coarse", None) for gamma in GAMMA_GRID]
    brackets = [(GAMMA_GRID[0], GAMMA_GRID[1])]
    for index in range(1, len(GAMMA_GRID) - 1):
        if coarse[index] <= coarse[index - 1] and coarse[index] <= coarse[index + 1]:
            brackets.append((GAMMA_GRID[index - 1], GAMMA_GRID[index + 1]))
    brackets.append((GAMMA_GRID[-2], GAMMA_GRID[-1]))
    refinements = []
    for index, (lower, upper) in enumerate(dict.fromkeys(brackets)):
        result = minimize_scalar(
            functools.partial(evaluate, stage="refinement", refinement_index=index),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": REFINEMENT_XATOL, "maxiter": REFINEMENT_MAXITER},
        )
        refinements.append(
            Refinement(
                lower,
                upper,
                float(result.x),
                float(result.fun),
                bool(result.success),
                int(result.status),
                str(result.message),
                int(result.nfev),
            )
        )
    gamma = min(cache, key=lambda value: (cache[value][1], value))
    head, score, clamps = cache[gamma]
    lower_floor = mariner.floor_value(response, mariner.FloorSpec(float("nan"), 0.0, GAMMA_GRID[-1]))
    upper_floor = mariner.floor_value(response, mariner.FloorSpec(float("nan"), 0.0, GAMMA_GRID[0]))
    bound_hits = tuple(
        name for name, bound in (("gamma_lower", GAMMA_GRID[0]), ("gamma_upper", GAMMA_GRID[-1])) if gamma == bound
    )
    return ProfileFit(
        head,
        gamma,
        score,
        lower_floor,
        upper_floor,
        bound_hits,
        max(trial.training_sse for trial in trace) - min(trial.training_sse for trial in trace),
        bool(np.all(response == response[0])),
        clamps,
        tuple(trace),
        tuple(refinements),
    )
