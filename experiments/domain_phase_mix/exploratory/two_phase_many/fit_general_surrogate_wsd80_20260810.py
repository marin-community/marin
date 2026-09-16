# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""GEN-002 on the WSD80 panel: bucket-general surrogate, no semantic assignment.

Runs the model defined in ``general_mixture_surrogate_20260809`` against the four frozen WSD80 gates.
Nothing here tells the surrogate what a bucket contains or which bucket the eval is about; it sees only
mixtures, per-bucket epoch geometry, and the topic grouping a production swarm already supplies.

Two poolings, by what determines each parameter. Readout exponents pool by TOPIC, because topic predicts
how much a family helps a given eval. Boundary scales pool by EXPOSURE STRATUM, because pool size
predicts how fast a bucket exhausts itself, and on the 39-bucket panel one topic spans a 359-fold range
of epochs per unit weight. Quality splits inside a topic share that topic's shape and differ only through
shrunk per-bucket amplitude departures, so the quality label groups but is never a feature.

Selection is nested and continuous, with multi-start over each boundary scale because the WSD80 objective
is multi-basin. Every printed line carries a configuration stamp so tallies from different configurations
cannot be merged silently -- that error was made once in this project and is worth making structurally
visible rather than trusting memory.

Usage: ``uv run python ... [seeds]``, default 0-10.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_gated_absorption_wsd80_20260807 as reference,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

RMSE_LIMIT = 0.007954
REGRET_LIMIT = 0.004842
DISTANCE_LIMIT = 0.05
GAIN_ERROR_LIMIT = harness.WSD_GAIN_ERROR_LIMIT
SURFACE_GRID = 401

PANEL = reference.PANEL
TARGETS = reference.TARGETS.values
INTERIOR = reference.INTERIOR
GEOMETRY = wsd.geometry()
SWARM = model.Panel(PANEL.weights, GEOMETRY.c0, GEOMETRY.c1, GEOMETRY.family_index)
N_FAMILIES = SWARM.n_families
N_STRATA = SWARM.n_exposure_strata()
BOUNDS = model.bounds(N_FAMILIES, N_STRATA)


def rows_panel(rows: np.ndarray) -> model.Panel:
    return model.Panel(SWARM.weights[rows], SWARM.epochs_early, SWARM.epochs_late, SWARM.family_index)


def grid_panel(weights: np.ndarray) -> model.Panel:
    return model.Panel(weights, SWARM.epochs_early, SWARM.epochs_late, SWARM.family_index)


def inner_error(vector: np.ndarray, panel: model.Panel, responses: np.ndarray, interior: np.ndarray, folds) -> float:
    shape, ridge = model.unpack(vector, N_FAMILIES, N_STRATA)
    free, constrained = model.design(panel, shape)
    if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
        return 1e6
    total = 0.0
    for train, test in folds:
        scale = np.maximum(responses[train].std(axis=0), 1e-9)
        for column in range(responses.shape[1]):
            b, a = model.fit_head(
                free[train], constrained[train], responses[train, column], ridge, model.pooled_width(panel)
            )
            residual = (free[test] @ b + constrained[test] @ a - responses[test, column])[interior[test]]
            if len(residual):
                total += float(residual @ residual) / (scale[column] ** 2)
    return total


def select(rows: np.ndarray, seed: int) -> np.ndarray:
    """Nested selection with multi-start over the boundary scales; no gate is ever consulted."""
    panel = rows_panel(rows)
    folds = harness.wsd80_folds("random", panel.weights, np.arange(len(rows)), reference.N_INNER_FOLDS, seed)
    args = (panel, TARGETS[rows], INTERIOR[rows], folds)
    best = differential_evolution(
        inner_error,
        BOUNDS,
        args=args,
        rng=np.random.default_rng(20260809),
        popsize=12,
        maxiter=70,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x
    score = inner_error(best, *args)
    for index in range(5 + N_FAMILIES, 5 + N_FAMILIES + N_STRATA):
        for start in np.linspace(BOUNDS[index][0], BOUNDS[index][1], 4):
            candidate = best.copy()
            candidate[index] = start
            refined = minimize(
                inner_error,
                candidate,
                args=args,
                bounds=list(BOUNDS),
                method="L-BFGS-B",
                options={"maxiter": 250, "eps": 1e-6},
            )
            if refined.fun < score:
                best, score = refined.x, float(refined.fun)
    return best


def evaluate(response: np.ndarray, seed: int) -> dict:
    outer = harness.wsd80_folds("random", SWARM.weights, np.arange(len(response)), reference.N_FOLDS, seed)
    predictions = np.empty_like(response)
    for train, test in outer:
        shape, ridge = model.unpack(select(train, seed), N_FAMILIES, N_STRATA)
        free, constrained = model.design(SWARM, shape)
        b, a = model.fit_head(free[train], constrained[train], response[train], ridge, model.pooled_width(SWARM))
        predictions[test] = free[test] @ b + constrained[test] @ a

    interior_rows = np.flatnonzero(INTERIOR)
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    ranked = interior_rows[np.argsort(predictions[interior_rows])]

    shape, ridge = model.unpack(select(np.arange(len(response)), seed), N_FAMILIES, N_STRATA)
    free, constrained = model.design(SWARM, shape)
    b, a = model.fit_head(free, constrained, response, ridge, model.pooled_width(SWARM))
    axis = np.linspace(0.0, 1.0, SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    fg, cg = model.design(grid_panel(wsd.grid_weights(flat_0, flat_1)), shape)
    surface = fg @ b + cg @ a
    tied_axis = np.linspace(0.0, 1.0, SURFACE_GRID * SURFACE_GRID // 4)
    ft, ct = model.design(grid_panel(wsd.grid_weights(tied_axis, tied_axis)), shape)
    best_cell = int(np.argmin(surface))

    return {
        "seed": seed,
        "rmse": float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2))),
        "regret_1": float(response[ranked[0]] - response[observed_best]),
        "regret_5": float(response[ranked[:5]].min() - response[observed_best]),
        "optimum": (float(flat_0[best_cell]), float(flat_1[best_cell])),
        "distance": float(
            np.hypot(
                flat_0[best_cell] - PANEL.phase_0[observed_best, 1], flat_1[best_cell] - PANEL.phase_1[observed_best, 1]
            )
        ),
        "gain_error": abs(float((ft @ b + ct @ a).min() - surface.min()) - harness.OBSERVED_WSD_GAIN),
    }


def main() -> None:
    primary = TARGETS[:, reference.TARGETS.names.index(harness.PRIMARY_TARGET)]
    seeds = [int(s) for s in sys.argv[1:]] or list(range(11))
    stamp = f"[cfg NF={N_FAMILIES} NS={N_STRATA}]"
    print(f"GEN-002 on WSD80, {len(PANEL.y)} rows, {int(INTERIOR.sum())} interior, {stamp}")
    print(
        f"gates: RMSE<={RMSE_LIMIT} Regret@1<={REGRET_LIMIT} distance<={DISTANCE_LIMIT} |gain err|<={GAIN_ERROR_LIMIT}"
    )
    passes = {"rmse": 0, "regret_1": 0, "distance": 0, "gain_error": 0}
    for seed in seeds:
        row = evaluate(primary, seed)
        checks = {
            "rmse": row["rmse"] <= RMSE_LIMIT,
            "regret_1": row["regret_1"] <= REGRET_LIMIT,
            "distance": row["distance"] <= DISTANCE_LIMIT,
            "gain_error": row["gain_error"] <= GAIN_ERROR_LIMIT,
        }
        for key, ok in checks.items():
            passes[key] += ok
        body = "  ".join(f"{k} {row[k]:.6f}{'P' if v else 'F'}" for k, v in checks.items())
        print(
            f"{stamp} seed {seed}: {body}  [{sum(checks.values())}/4]"
            f"  optimum ({row['optimum'][0]:.3f},{row['optimum'][1]:.3f})  Regret@5 {row['regret_5']:.6f}"
        )
    total = sum(passes.values())
    print(
        f"\n{stamp} {total}/{4 * len(seeds)} over {len(seeds)} seeds: "
        + "  ".join(f"{k} {v}/{len(seeds)}" for k, v in passes.items())
    )


if __name__ == "__main__":
    main()
