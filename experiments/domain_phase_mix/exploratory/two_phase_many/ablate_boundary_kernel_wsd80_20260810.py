# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Is the early-boundary kernel doing anything? Ablation, not inference from its scale (GEN-014).

A previous verdict in this round read a large fitted boundary scale as the model switching the term off.
That inference is wrong. For large `k`, ``exp(-E/k) = 1 - E/k + O(k^-2)``; the constant is absorbed by
the intercept and the solve normalises every residualised column to unit norm, cancelling the ``1/k``.
Measured on WSD80, the residualised code column correlates 0.9999600 with negative early exposure at
`k = 316.2` and 0.9999996 at `k = 3162.3`. A large scale does not disable the term, it LINEARISES it.

So the term's contribution has to be measured by removing it. This drops the whole boundary block from
the design -- both families at once -- and reruns the frozen WSD80 gates against the identical folds,
identical selection, and identical heads. Column slicing rather than a model edit, so the shared model
stays untouched while other reruns are in flight.

Reads the corrected rank-truncating solve, so it is not comparable to any pre-fix tally.

Usage: ``uv run python ... [seeds]``, default 0-3.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_general_surrogate_wsd80_20260810 as driver,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

NF = driver.N_FAMILIES
BOUNDARY_BLOCK = slice(2 * NF, 3 * NF)


def design_without_boundary(panel, shape):
    """The GEN-002 design with the boundary block removed, and its reduced pooled width."""
    free, constrained = model.design(panel, shape)
    keep = np.ones(constrained.shape[1], dtype=bool)
    keep[BOUNDARY_BLOCK] = False
    return free, constrained[:, keep], model.pooled_width(panel) - NF


def make_design(panel, shape, ablated: bool):
    if ablated:
        return design_without_boundary(panel, shape)
    free, constrained = model.design(panel, shape)
    return free, constrained, model.pooled_width(panel)


def inner_error(vector, panel, responses, interior, folds, ablated: bool) -> float:
    shape, ridge = model.unpack(vector, NF, driver.N_STRATA)
    free, constrained, pooled = make_design(panel, shape, ablated)
    if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
        return 1e6
    total = 0.0
    for train, test in folds:
        scale = np.maximum(responses[train].std(axis=0), 1e-9)
        for column in range(responses.shape[1]):
            b, a = model.fit_head(free[train], constrained[train], responses[train, column], ridge, pooled)
            residual = (free[test] @ b + constrained[test] @ a - responses[test, column])[interior[test]]
            if len(residual):
                total += float(residual @ residual) / (scale[column] ** 2)
    return total


def select(rows, seed: int, ablated: bool) -> np.ndarray:
    panel = driver.rows_panel(rows)
    folds = harness.wsd80_folds("random", panel.weights, np.arange(len(rows)), driver.reference.N_INNER_FOLDS, seed)
    return differential_evolution(
        inner_error,
        driver.BOUNDS,
        args=(panel, driver.TARGETS[rows], driver.INTERIOR[rows], folds, ablated),
        rng=np.random.default_rng(20260809),
        popsize=12,
        maxiter=70,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def evaluate(response: np.ndarray, seed: int, ablated: bool) -> dict:
    outer = harness.wsd80_folds("random", driver.SWARM.weights, np.arange(len(response)), driver.reference.N_FOLDS, seed)
    predictions = np.empty_like(response)
    for train, test in outer:
        shape, ridge = model.unpack(select(train, seed, ablated), NF, driver.N_STRATA)
        free, constrained, pooled = make_design(driver.SWARM, shape, ablated)
        b, a = model.fit_head(free[train], constrained[train], response[train], ridge, pooled)
        predictions[test] = free[test] @ b + constrained[test] @ a

    interior_rows = np.flatnonzero(driver.INTERIOR)
    observed_best = int(interior_rows[np.argmin(response[interior_rows])])
    ranked = interior_rows[np.argsort(predictions[interior_rows])]

    shape, ridge = model.unpack(select(np.arange(len(response)), seed, ablated), NF, driver.N_STRATA)
    free, constrained, pooled = make_design(driver.SWARM, shape, ablated)
    b, a = model.fit_head(free, constrained, response, ridge, pooled)
    axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    f0, f1 = g0.ravel(), g1.ravel()
    fg, cg, _ = make_design(driver.grid_panel(wsd.grid_weights(f0, f1)), shape, ablated)
    surface = fg @ b + cg @ a
    tied_axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID * driver.SURFACE_GRID // 4)
    ft, ct, _ = make_design(driver.grid_panel(wsd.grid_weights(tied_axis, tied_axis)), shape, ablated)
    cell = int(np.argmin(surface))
    amplitudes = a[BOUNDARY_BLOCK] if not ablated else np.array([])

    return {
        "rmse": float(np.sqrt(np.mean((predictions - response)[driver.INTERIOR] ** 2))),
        "regret_1": float(response[ranked[0]] - response[observed_best]),
        "distance": float(
            np.hypot(
                f0[cell] - driver.PANEL.phase_0[observed_best, 1], f1[cell] - driver.PANEL.phase_1[observed_best, 1]
            )
        ),
        "gain_error": abs(float((ft @ b + ct @ a).min() - surface.min()) - harness.OBSERVED_WSD_GAIN),
        "optimum": (float(f0[cell]), float(f1[cell])),
        "boundary_amplitudes": amplitudes,
        "boundary_scales": shape.boundary_scale,
    }


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3]
    primary = driver.TARGETS[:, driver.reference.TARGETS.names.index(harness.PRIMARY_TARGET)]
    print("GEN-014: does the early-boundary kernel earn its place? (corrected rank-truncating solve)")
    print(f"gates RMSE<={driver.RMSE_LIMIT} R@1<={driver.REGRET_LIMIT} dist<={driver.DISTANCE_LIMIT}")
    print(f"      |gain err|<={driver.GAIN_ERROR_LIMIT};  seeds {seeds}\n")

    tally = {"full": 0, "ablated": 0}
    for seed in seeds:
        for label, ablated in (("full", False), ("ablated", True)):
            row = evaluate(primary, seed, ablated)
            checks = {
                "rmse": row["rmse"] <= driver.RMSE_LIMIT,
                "regret_1": row["regret_1"] <= driver.REGRET_LIMIT,
                "distance": row["distance"] <= driver.DISTANCE_LIMIT,
                "gain_error": row["gain_error"] <= driver.GAIN_ERROR_LIMIT,
            }
            tally[label] += sum(checks.values())
            body = "  ".join(f"{k} {row[k]:.6f}{'P' if v else 'F'}" for k, v in checks.items())
            extra = ""
            if not ablated:
                amps = " ".join(f"{x:.3e}" for x in row["boundary_amplitudes"])
                scales = " ".join(f"{x:.1f}" for x in row["boundary_scales"])
                extra = f"  boundary amp [{amps}] scale [{scales}]"
            print(
                f" seed {seed} {label:8s}: {body}  [{sum(checks.values())}/4]"
                f"  opt ({row['optimum'][0]:.3f},{row['optimum'][1]:.3f}){extra}"
            )
        print()
    cells = 4 * len(seeds)
    print(f"totals over {len(seeds)} seeds: full {tally['full']}/{cells}  ablated {tally['ablated']}/{cells}")


if __name__ == "__main__":
    main()
