# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The current best WSD80 configuration, as a checked-in driver (GEN-039).

The exposed 43-of-44 score remains a historical result under its original frozen gates. This driver uses
the repaired nested-fold row alignment and reports the successor model-form diagnostics. Coordinate
distance is descriptive only: deployment quality is evaluated separately with fresh, same-seed paired
non-inferiority in ``evaluate_wsd80_selected_policy_noninferiority_20260815.py``.

The configuration is deliberately the COMMITTED design -- the power-law readout of
`general_mixture_surrogate_20260809`, unchanged -- with three things around it:

  range-penalised selection   the promoted change (GEN-034): the inner objective charges predictions
                              that leave the observed target range, scored over EVERY row's design
  single-target selection     theta selected against the primary evaluation alone (candidate A)
  widened exponent bound      now in the model itself, since it is measured rather than chosen

Note what this is NOT. `range_selected_surrogate_20260810` is a different candidate: it uses a SATURATING
readout, and at the widened bound it scores 15/20 over five seeds here, so the two must not be conflated.
The saturating readout's value was 300M-specific -- one component regression against three -- and it is
gate-for-gate identical to the power law on WSD80 once the range penalty is present.

Usage: ``uv run python ... [seeds]``, default 0-10.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
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
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    range_selected_surrogate_20260810 as ranged,
)

SWARM = driver.SWARM
INTERIOR = driver.INTERIOR
N_FAMILIES = SWARM.n_families
N_STRATA = SWARM.n_exposure_strata()
BOUNDS = model.bounds(N_FAMILIES, N_STRATA)


def inner_error(vector, panel, full, response, folds) -> float:
    shape, ridge = model.unpack(vector, N_FAMILIES, N_STRATA)
    free, constrained = model.design(panel, shape)
    free_all, constrained_all = model.design(full, shape)
    if not (
        np.isfinite(free).all()
        and np.isfinite(constrained).all()
        and np.isfinite(free_all).all()
        and np.isfinite(constrained_all).all()
    ):
        return 1e6
    pooled = model.pooled_width(panel)
    total = 0.0
    for train, test in folds:
        head, amplitudes = model.fit_head(free[train], constrained[train], response[train], ridge, pooled)
        residual = free[test] @ head + constrained[test] @ amplitudes - response[test]
        error = float(residual @ residual)
        excess = ranged.range_excess(free_all @ head + constrained_all @ amplitudes, response[train])
        total += error + ranged.RANGE_WEIGHT * excess**2 * (error + 1e-12)
    return total


def select(panel, full, response, folds, seed: int) -> np.ndarray:
    return differential_evolution(
        inner_error,
        BOUNDS,
        args=(panel, full, response, folds),
        rng=np.random.default_rng(20260812 + seed),
        popsize=10,
        maxiter=50,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def evaluate(response: np.ndarray, seed: int) -> dict:
    rows = np.arange(len(response))
    predictions = np.full_like(response, np.nan)
    for train, test in harness.wsd80_folds("random", SWARM.weights, rows, driver.reference.N_FOLDS, seed):
        sub = model.Panel(SWARM.weights[train], SWARM.epochs_early, SWARM.epochs_late, SWARM.family_index)
        inner = harness.wsd80_folds(
            "random",
            sub.weights,
            np.arange(len(train)),
            driver.reference.N_INNER_FOLDS,
            harness.WSD_INNER_SEED_BASE + seed,
        )
        shape, ridge = model.unpack(select(sub, SWARM, response[train], inner, seed), N_FAMILIES, N_STRATA)
        free, constrained = model.design(SWARM, shape)
        head, amplitudes = model.fit_head(
            free[train], constrained[train], response[train], ridge, model.pooled_width(SWARM)
        )
        predictions[test] = free[test] @ head + constrained[test] @ amplitudes
    if not np.isfinite(predictions).all():
        missing = np.flatnonzero(~np.isfinite(predictions)).tolist()
        raise ValueError(f"Outer folds did not produce finite predictions for rows {missing}")

    interior = np.flatnonzero(INTERIOR)
    best = int(interior[np.argmin(response[interior])])
    ranked = interior[np.argsort(predictions[interior])]

    inner_full = harness.wsd80_folds(
        "random", SWARM.weights, rows, driver.reference.N_INNER_FOLDS, harness.WSD_INNER_SEED_BASE + seed
    )
    shape, ridge = model.unpack(select(SWARM, SWARM, response, inner_full, seed), N_FAMILIES, N_STRATA)
    free, constrained = model.design(SWARM, shape)
    head, amplitudes = model.fit_head(free, constrained, response, ridge, model.pooled_width(SWARM))

    axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    grid_free, grid_constrained = model.design(driver.grid_panel(wsd.grid_weights(flat_0, flat_1)), shape)
    surface = grid_free @ head + grid_constrained @ amplitudes
    tied_free, tied_constrained = model.design(driver.grid_panel(wsd.grid_weights(axis, axis)), shape)
    cell = int(np.argmin(surface))

    return {
        "rmse": float(np.sqrt(np.mean((predictions - response)[INTERIOR] ** 2))),
        "regret_1": float(response[ranked[0]] - response[best]),
        "distance": float(
            np.hypot(flat_0[cell] - driver.PANEL.phase_0[best, 1], flat_1[cell] - driver.PANEL.phase_1[best, 1])
        ),
        "gain_error": abs(
            float((tied_free @ head + tied_constrained @ amplitudes).min() - surface.min()) - harness.OBSERVED_WSD_GAIN
        ),
        "optimum": (float(flat_0[cell]), float(flat_1[cell])),
    }


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or list(range(11))
    primary = driver.TARGETS[:, driver.reference.TARGETS.names.index(harness.PRIMARY_TARGET)]
    limits = {
        "rmse": driver.RMSE_LIMIT,
        "regret_1": driver.REGRET_LIMIT,
        "gain_error": driver.GAIN_ERROR_LIMIT,
    }
    print("GEN-039 frontier: committed power-law design, range-penalised single-target selection,")
    print("widened exponent bound. Successor diagnostics exclude coordinate distance;")
    print("the exposed 44-cell score remains historical. Distance is reported descriptively only.")
    print(f"Seeds {seeds}\n")
    tally = dict.fromkeys(limits, 0)
    optima = []
    for seed in seeds:
        row = evaluate(primary, seed)
        checks = {k: row[k] <= v for k, v in limits.items()}
        for key, ok in checks.items():
            tally[key] += ok
        optima.append(row["optimum"])
        body = "  ".join(f"{k} {row[k]:.6f}{'P' if v else 'F'}" for k, v in checks.items())
        print(
            f"  seed {seed:2d}: {body}  [{sum(checks.values())}/{len(checks)}]  "
            f"optimum ({row['optimum'][0]:.6f},{row['optimum'][1]:.6f})  "
            f"descriptive distance {row['distance']:.6f}",
            flush=True,
        )
    coordinates = np.array(optima)
    print(
        f"\n{sum(tally.values())}/{len(limits) * len(seeds)} successor diagnostic cells: "
        + "  ".join(f"{k} {v}/{len(seeds)}" for k, v in tally.items())
    )
    print(
        f"mean optimum ({coordinates[:, 0].mean():.4f}, {coordinates[:, 1].mean():.4f}) "
        "vs discovery-panel raw argmin (0.100, 0.500)"
    )


if __name__ == "__main__":
    main()
