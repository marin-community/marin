# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Does the bounded-readout candidate pay for itself on the panels that matter? (GEN-022)

GEN-021 fixed this project's worst measured failure -- 300M arc_challenge, once 6586 times worse than an
intercept -- with two changes that REMOVE structure rather than add it: a saturating readout,
``1 / (1 + (E / E0) ** gamma)``, which is bounded in [0,1] by construction, and deletion of the
per-bucket departures block, whose 39 columns let a held-out row activate more of the design than any
training row.

That was one component on one seed. This measures what those changes COST where the model is supposed to
work: 300M Uncheatable out-of-fold error, and WSD80's four frozen gates. Four arms, so the two changes
are separable rather than confounded:

  baseline    power-law readout, departures kept   -- the committed model
  saturating  saturating readout, departures kept
  no-dep      power-law readout, departures dropped
  candidate   saturating readout, departures dropped

Selection, folds and heads are identical across arms; only the design differs.

Usage: ``uv run python ... [300m|wsd80] [seeds]``, default 300m and seeds 0-2.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as packet,
)
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

ARMS = ("baseline", "saturating", "no-dep", "candidate")
N_FOLDS = 3
N_INNER_FOLDS = 3


def design(panel: model.Panel, shape: model.Shape, arm: str) -> tuple[np.ndarray, np.ndarray, int]:
    """The GEN-002 design with the readout form and the departures block selected by `arm`."""
    exponent = np.asarray(shape.readout_exponent, dtype=float)[panel.family_index]
    scale = np.asarray(shape.boundary_scale, dtype=float)[panel.exposure_stratum()]
    saturating = arm in ("saturating", "candidate")

    def readout(exposure: np.ndarray) -> np.ndarray:
        if saturating:
            powered = (np.maximum(exposure, 0.0) / shape.offset) ** exponent
            return 1.0 / (1.0 + powered)
        return (exposure + shape.offset) ** -exponent

    near = readout(panel.exposure(shape.near_horizon))
    late = readout(panel.exposure(1.0))
    boundary = np.exp(-panel.early_epochs() / scale)
    excess = np.maximum(panel.exposure(shape.damage_horizon) - 1.0, 0.0) / model.DAMAGE_KNEE
    powered = excess**shape.damage_exponent
    damage = powered / (1.0 + powered)

    free = np.column_stack([np.ones(len(panel.weights)), model.family_sums(panel.weights[:, 1, :], panel.family_index)])
    blocks = [
        model.family_sums(near, panel.family_index),
        model.family_sums(late, panel.family_index),
        model.family_sums(boundary, panel.family_index),
        model.family_sums(damage, panel.family_index),
    ]
    pooled = 4 * panel.n_families
    if arm in ("baseline", "saturating"):
        blocks.append(near)
    return free, np.column_stack(blocks), pooled


def bounds_for(panel: model.Panel, arm: str) -> tuple:
    """The saturating readout's `offset` is a SCALE in epochs, not a floor, so it needs a wider box."""
    base = list(model.bounds(panel.n_families, panel.n_exposure_strata()))
    if arm in ("saturating", "candidate"):
        base[2] = (-2.0, 1.5)
    return tuple(base)


def inner_error(vector, panel, response, folds, arm) -> float:
    shape, ridge = model.unpack(vector, panel.n_families, panel.n_exposure_strata())
    free, constrained, pooled = design(panel, shape, arm)
    if not (np.isfinite(free).all() and np.isfinite(constrained).all()):
        return 1e6
    total = 0.0
    for train, test in folds:
        head, amplitudes = model.fit_head(free[train], constrained[train], response[train], ridge, pooled)
        residual = free[test] @ head + constrained[test] @ amplitudes - response[test]
        total += float(residual @ residual)
    return total


def select(panel, response, folds, arm, seed: int) -> np.ndarray:
    return differential_evolution(
        inner_error,
        bounds_for(panel, arm),
        args=(panel, response, folds, arm),
        rng=np.random.default_rng(20260812 + seed),
        popsize=10,
        maxiter=50,
        tol=1e-11,
        polish=True,
        init="sobol",
    ).x


def evaluate_wsd80(seeds: list[int]) -> None:
    """The same four arms against WSD80's four frozen gates."""
    swarm = driver.SWARM
    response = driver.TARGETS[:, driver.reference.TARGETS.names.index(harness.PRIMARY_TARGET)]
    print("GEN-022: the same four arms against WSD80's frozen gates")
    print(f"gates RMSE<={driver.RMSE_LIMIT} R@1<={driver.REGRET_LIMIT} dist<={driver.DISTANCE_LIMIT}")
    print(f"      |gain err|<={driver.GAIN_ERROR_LIMIT}; seeds {seeds}")
    print("reference: the committed model scores 40/44 over 11 seeds\n")

    for arm in ARMS:
        passed = 0
        for seed in seeds:
            outer = harness.wsd80_folds(
                "random", swarm.weights, np.arange(len(response)), driver.reference.N_FOLDS, seed
            )
            predictions = np.empty_like(response)
            for train, test in outer:
                sub = model.Panel(swarm.weights[train], swarm.epochs_early, swarm.epochs_late, swarm.family_index)
                inner = harness.wsd80_folds(
                    "random", sub.weights, np.arange(len(train)), driver.reference.N_INNER_FOLDS, seed
                )
                vector = select(sub, response[train], inner, arm, seed)
                shape, ridge = model.unpack(vector, swarm.n_families, swarm.n_exposure_strata())
                free, constrained, pooled = design(swarm, shape, arm)
                head, amplitudes = model.fit_head(free[train], constrained[train], response[train], ridge, pooled)
                predictions[test] = free[test] @ head + constrained[test] @ amplitudes

            interior = np.flatnonzero(driver.INTERIOR)
            best = int(interior[np.argmin(response[interior])])
            ranked = interior[np.argsort(predictions[interior])]

            inner_full = harness.wsd80_folds(
                "random", swarm.weights, np.arange(len(response)), driver.reference.N_INNER_FOLDS, seed
            )
            shape, ridge = model.unpack(
                select(swarm, response, inner_full, arm, seed), swarm.n_families, swarm.n_exposure_strata()
            )
            free, constrained, pooled = design(swarm, shape, arm)
            head, amplitudes = model.fit_head(free, constrained, response, ridge, pooled)
            axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID)
            g0, g1 = np.meshgrid(axis, axis, indexing="ij")
            f0, f1 = g0.ravel(), g1.ravel()
            fg, cg, _ = design(driver.grid_panel(wsd.grid_weights(f0, f1)), shape, arm)
            surface = fg @ head + cg @ amplitudes
            tied_axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID * driver.SURFACE_GRID // 4)
            ft, ct, _ = design(driver.grid_panel(wsd.grid_weights(tied_axis, tied_axis)), shape, arm)
            cell = int(np.argmin(surface))

            checks = {
                "rmse": float(np.sqrt(np.mean((predictions - response)[driver.INTERIOR] ** 2))) <= driver.RMSE_LIMIT,
                "regret_1": float(response[ranked[0]] - response[best]) <= driver.REGRET_LIMIT,
                "distance": (
                    float(np.hypot(f0[cell] - driver.PANEL.phase_0[best, 1], f1[cell] - driver.PANEL.phase_1[best, 1]))
                    <= driver.DISTANCE_LIMIT
                ),
                "gain_error": (
                    abs(float((ft @ head + ct @ amplitudes).min() - surface.min()) - harness.OBSERVED_WSD_GAIN)
                    <= driver.GAIN_ERROR_LIMIT
                ),
            }
            passed += sum(checks.values())
            print(
                f"  {arm:11s} seed {seed}: [{sum(checks.values())}/4] optimum ({f0[cell]:.3f},{f1[cell]:.3f})",
                flush=True,
            )
        print(f"  {arm:11s} TOTAL {passed}/{4 * len(seeds)}\n", flush=True)


def main() -> None:
    argv = sys.argv[1:]
    panel_name = argv[0] if argv and argv[0] in ("300m", "wsd80") else "300m"
    seeds = [int(s) for s in argv[1:] if s.isdigit()] or [0, 1, 2]
    if panel_name == "wsd80":
        evaluate_wsd80(seeds)
        return
    data = packet.load_300m("uncheatable")
    panel = model.Panel(data.weights, data.c0, data.c1, data.family_index)
    y = data.y

    print("GEN-022: what the bounded-readout candidate costs on 300M Uncheatable")
    print(f"arms {ARMS}, {N_FOLDS} grouped outer folds, seeds {seeds}")
    print("reference: the committed model scores 0.005665 nested; intercept floor is 0.017782\n")

    for arm in ARMS:
        scores = []
        for seed in seeds:
            predictions = np.empty_like(y)
            for train, test in packet.grouped_folds(data.frame, seed, N_FOLDS):
                inner = packet.grouped_folds(data.frame.iloc[train].reset_index(drop=True), seed, N_INNER_FOLDS)
                sub = model.Panel(panel.weights[train], panel.epochs_early, panel.epochs_late, panel.family_index)
                vector = select(sub, y[train], inner, arm, seed)
                shape, ridge = model.unpack(vector, panel.n_families, panel.n_exposure_strata())
                free, constrained, pooled = design(panel, shape, arm)
                head, amplitudes = model.fit_head(free[train], constrained[train], y[train], ridge, pooled)
                predictions[test] = free[test] @ head + constrained[test] @ amplitudes
            scores.append(float(np.sqrt(np.mean((predictions - y) ** 2))))
        print(f"  {arm:11s} {' '.join(f'{v:.6f}' for v in scores)}   mean {np.mean(scores):.6f}", flush=True)


if __name__ == "__main__":
    main()
