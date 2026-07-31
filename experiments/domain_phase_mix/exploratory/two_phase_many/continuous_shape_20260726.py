# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Continuous optimization of the nonlinear shape, scored on both dual objectives.

The Observatory shapes come from a coarse product grid: ``rate`` in {0.25, 1},
``power`` in {0.4, 0.7, 1}, ``late_multiplier`` in {0.5, 1, 2, 4},
``forgetting_rate`` in {0, 0.25, 1}, ``penalty_threshold`` in {0, 1, 2}. On the 300M
fit panel the selected ``hierarchical_phase_replay`` shape sits on a grid boundary in
three of its four active coordinates, so the grid is plausibly binding rather than
merely coarse. This module replaces the grid search with bounded Nelder-Mead over the
shape, keeping the inner nonnegative-least-squares head untouched, and asks the two
questions separately: does the finer shape fit better out of fold, and what does it do
to the optimum-quality arms.

Two details make the comparison honest.

Shape optimization runs inside the selection step, so it only ever sees the rows the
selection step is allowed to see. The censored arm therefore optimizes its shape on the
uncensored rows alone, exactly as the grid arm selects on them alone.

Grid and continuous arms share one scoring path. ``score_with`` takes a selector, and
``grid_selector`` reproduces ``dual_objective_harness_20260726.score_candidate``
numerically, which is asserted in ``audit_continuous_shape_20260726.py``. Any
difference in the reported numbers is then a difference in the shape, not in the
scoring code.

Bounds are wider than the grid on purpose. An optimum pinned to a grid edge is
uninformative about whether the edge or the coarseness is the constraint, so
``late_multiplier`` reaches 60 against a grid maximum of 4 and ``power`` reaches 3
against a grid maximum of 1.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import (  # noqa: E402
    CENSOR_FRACTIONS,
    Benchmark,
    fit_metrics,
    fit_on,
    out_of_fold_predictions,
)
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import Model, Panel  # noqa: E402

# Transform and bounds per shape coordinate. "log" keeps a positive scale multiplicative
# so a simplex step is a ratio rather than an increment; "linear" is used where zero is a
# meaningful value the grid contains (no forgetting, no overexposure allowance).
# Every upper bound is well outside the grid so a boundary solution is diagnostic.
SHAPE_SPEC: dict[str, tuple[str, float, float]] = {
    "rate": ("log", 0.02, 20.0),
    "power": ("log", 0.05, 3.0),
    "late_multiplier": ("log", 0.02, 60.0),
    "forgetting_rate": ("linear", 0.0, 12.0),
    "penalty_threshold": ("linear", 0.0, 6.0),
}

# Restart count for the multi-start Nelder-Mead. The objective is only piecewise smooth
# because the nonnegative head changes active set, so restarts matter more than
# tolerance.
N_RESTARTS = 5
NELDER_MEAD_OPTIONS = {"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000, "maxfev": 2000}
# Simplex edge in transformed units. Large enough to escape a flat active-set plateau.
INITIAL_SIMPLEX_STEP = 0.35


def criterion_score(metrics: dict[str, float], criterion: str) -> float:
    """Selection score, matching ``dual_objective_harness_20260726.select_by``."""
    return abs(metrics["low_tail_optimism"]) if criterion == "low_tail_honesty" else metrics[criterion]


def active_shape_keys(model: Model, panel: Panel, base: dict) -> tuple[str, ...]:
    """Shape coordinates that actually change the design matrix.

    ``hierarchical_phase_replay`` ignores ``rate`` and ``compact_retained_state``
    ignores ``penalty_threshold``, so both grids contain redundant axes. Optimizing a
    coordinate the design never reads would leave the simplex wandering on a plateau.
    """
    reference = model.build(panel, base).matrix
    active = []
    for key, (kind, low, high) in SHAPE_SPEC.items():
        if key not in base:
            continue
        probe = dict(base)
        probe[key] = float(np.sqrt(max(low, 1e-3) * high)) if kind == "log" else 0.5 * (low + high)
        if probe[key] == base[key]:
            probe[key] = float(base[key]) + 0.37
        if not np.allclose(model.build(panel, probe).matrix, reference):
            active.append(key)
    return tuple(active)


@dataclass(frozen=True)
class ShapeSpace:
    """Bijection between a shape dict and a bounded real vector over its active keys."""

    keys: tuple[str, ...]
    base: dict

    @property
    def bounds(self) -> list[tuple[float, float]]:
        out = []
        for key in self.keys:
            kind, low, high = SHAPE_SPEC[key]
            out.append((np.log(low), np.log(high)) if kind == "log" else (low, high))
        return out

    def to_vector(self, shape: dict) -> np.ndarray:
        values = []
        for key in self.keys:
            kind, low, high = SHAPE_SPEC[key]
            raw = float(np.clip(float(shape[key]), low, high))
            values.append(np.log(max(raw, low)) if kind == "log" else raw)
        return np.asarray(values, dtype=float)

    def to_shape(self, vector: np.ndarray) -> dict:
        shape = dict(self.base)
        for key, value in zip(self.keys, np.asarray(vector, dtype=float), strict=True):
            kind, low, high = SHAPE_SPEC[key]
            shape[key] = float(np.clip(np.exp(value) if kind == "log" else value, low, high))
        return shape


def _grid_scores(
    panel: Panel, model: Model, target: str, criterion: str, l2: float, rows: np.ndarray | None
) -> list[tuple[float, dict]]:
    observed = panel.targets[target] if rows is None else np.where(rows, panel.targets[target], np.nan)
    scored = []
    for shape in model.shapes():
        prediction = out_of_fold_predictions(panel, model, target, shape, l2, rows=rows)
        scored.append((criterion_score(fit_metrics(observed, prediction), criterion), shape))
    scored.sort(key=lambda item: item[0])
    return scored


def _distinct_starts(scored: list[tuple[float, dict]], keys: tuple[str, ...], count: int) -> list[dict]:
    """Best grid points that differ on the active coordinates, best first."""
    starts: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for _, shape in scored:
        signature = tuple(float(shape[key]) for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        starts.append(shape)
        if len(starts) == count:
            break
    return starts


def optimize_shape(
    panel: Panel,
    model: Model,
    target: str,
    criterion: str,
    l2: float,
    rows: np.ndarray | None = None,
    n_restarts: int = N_RESTARTS,
    grid_scored: list[tuple[float, dict]] | None = None,
) -> tuple[dict, float, dict[str, Any]]:
    """Minimize the out-of-fold selection criterion over a continuous shape.

    Starts from the best distinct grid points so the continuous arm can never lose to
    the grid arm by landing in a worse basin: the grid optimum is always one of the
    initial simplex centres, and its score is kept as a floor.
    """
    observed = panel.targets[target] if rows is None else np.where(rows, panel.targets[target], np.nan)
    scored = grid_scored if grid_scored is not None else _grid_scores(panel, model, target, criterion, l2, rows)
    space = ShapeSpace(active_shape_keys(model, panel, scored[0][1]), scored[0][1])
    if not space.keys:
        return scored[0][1], scored[0][0], {"evaluations": 0, "restarts": 0, "active_keys": ()}

    calls = 0

    def objective(vector: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        prediction = out_of_fold_predictions(panel, model, target, space.to_shape(vector), l2, rows=rows)
        score = criterion_score(fit_metrics(observed, prediction), criterion)
        return float(score) if np.isfinite(score) else 1e6

    bounds = space.bounds
    best_shape, best_score = scored[0][1], float(scored[0][0])
    for start in _distinct_starts(scored, space.keys, n_restarts):
        origin = space.to_vector(start)
        simplex = np.vstack([origin] + [origin + INITIAL_SIMPLEX_STEP * e for e in np.eye(len(origin))])
        simplex = np.clip(simplex, [b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(
            objective,
            origin,
            method="Nelder-Mead",
            bounds=bounds,
            options={**NELDER_MEAD_OPTIONS, "initial_simplex": simplex},
        )
        # Re-score the returned point through the same path used for reporting, because
        # the clipping inside to_shape can move a boundary solution.
        candidate = space.to_shape(result.x)
        predicted = out_of_fold_predictions(panel, model, target, candidate, l2, rows=rows)
        score = criterion_score(fit_metrics(observed, predicted), criterion)
        if np.isfinite(score) and score < best_score:
            best_shape, best_score = candidate, float(score)
    info = {
        "evaluations": calls,
        "restarts": len(_distinct_starts(scored, space.keys, n_restarts)),
        "active_keys": space.keys,
        "grid_score": float(scored[0][0]),
    }
    return best_shape, best_score, info


Selector = Callable[[Panel, Model, str, str, np.ndarray | None], tuple[dict, float, dict[str, Any]]]


def grid_selector(
    panel: Panel, model: Model, target: str, criterion: str, rows: np.ndarray | None = None
) -> tuple[dict, float, dict[str, Any]]:
    """Grid search over ``model.shapes()`` x ``model.l2_grid``, as the harness does."""
    best: tuple[float, dict, float] | None = None
    evaluated = 0
    for l2 in model.l2_grid:
        for score, shape in _grid_scores(panel, model, target, criterion, l2, rows):
            evaluated += 1
            if best is None or score < best[0]:
                best = (score, shape, l2)
    assert best is not None, "empty shape or ridge grid"
    return best[1], best[2], {"criterion": criterion, "evaluated": evaluated, "selected_score": best[0]}


def continuous_selector(
    panel: Panel, model: Model, target: str, criterion: str, rows: np.ndarray | None = None
) -> tuple[dict, float, dict[str, Any]]:
    """Continuous shape optimization at every ridge on the grid, best pair returned."""
    best: tuple[float, dict, float, dict] | None = None
    for l2 in model.l2_grid:
        scored = _grid_scores(panel, model, target, criterion, l2, rows)
        shape, score, info = optimize_shape(panel, model, target, criterion, l2, rows, grid_scored=scored)
        if best is None or score < best[0]:
            best = (score, shape, l2, info)
    assert best is not None, "empty ridge grid"
    score, shape, l2, info = best
    return shape, l2, {"criterion": criterion, "selected_score": score, **info}


def censored_metrics_with(
    panel: Panel, model: Model, target: str, criterion: str, fraction: float, selector: Selector
) -> dict[str, float]:
    """Harness ``censored_metrics`` with the selection step supplied by the caller."""
    observed = panel.targets[target]
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    censored = np.zeros(len(observed), dtype=bool)
    censored[ordering[:n_censored]] = True
    train = available & ~censored

    shape, l2, _ = selector(panel, model, target, criterion, train)
    prediction = fit_on(panel, model, target, shape, l2, rows=train).predict(panel)[censored]
    truth = observed[censored]
    residual = prediction - truth
    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    return {
        "n_censored": int(n_censored),
        "worst_censored_bpb": float(truth.max()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
        "shape": shape,
        "l2": l2,
    }


def phase_skill_with(
    benchmark: Benchmark, model: Model, target: str, criterion: str, selector: Selector
) -> dict[str, Any]:
    """Harness ``phase_skill_from`` with the selection step supplied by the caller."""
    shape, l2, _ = selector(benchmark.fit_60m, model, target, criterion, None)
    fitted = fit_on(benchmark.fit_60m, model, target, shape, l2)
    predicted_delta = fitted.predict(benchmark.paired_300m.two_phase_panel) - fitted.predict(
        benchmark.paired_300m.tied_panel
    )
    return {
        **phase_decision_skill(predicted_delta, benchmark.paired_300m.observed_delta[target]),
        "shape_60m": shape,
        "l2_60m": l2,
    }


def score_with(benchmark: Benchmark, model: Model, target: str, criterion: str, selector: Selector) -> dict[str, Any]:
    """``score_candidate`` with a pluggable selector, so arms share one scoring path."""
    shape, l2, selection = selector(benchmark.fit_300m, model, target, criterion, None)
    in_scale_oof = out_of_fold_predictions(benchmark.fit_300m, model, target, shape, l2)
    return {
        "model": model.name,
        "target": target,
        "criterion": criterion,
        "shape_300m": shape,
        "l2_300m": l2,
        "selection": selection,
        "fit": fit_metrics(benchmark.fit_300m.targets[target], in_scale_oof),
        "censored": {
            f"{fraction:.2f}": censored_metrics_with(benchmark.fit_300m, model, target, criterion, fraction, selector)
            for fraction in CENSOR_FRACTIONS
        },
        "phase": phase_skill_with(benchmark, model, target, criterion, selector),
    }


def state_model(name: str, build: Callable[[Panel, dict], Any], shapes: Callable[[], Iterable[dict]], l2_grid) -> Model:
    return Model(name, build, shapes, l2_grid=l2_grid)
