# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Select theta on aggregate-matched contrast accuracy rather than level error (ATOM-006).

Every phase mechanism tried is expressible and capable of untied optima, and a criterion scoring
held-out LEVEL error declines all of them, because expressing phase structure does not reduce level
error. This changes only what theta is selected on. The head still fits levels, so no target is replaced
by a difference -- which is what separates this from the earlier matched-pair attempt that lost by
amplifying noise.

The contrast is c = y - A(aggregate), with A the tied response interpolated from the panel's tied rows.
Two policies sharing an aggregate differ only in ordering, so a criterion built on c cannot be satisfied
by getting the aggregate level right.

Three signatures were registered in advance: a non-zero predicted two-phase gain where level selection
gave exactly zero; improvement concentrated on the untied cells; and a COST in level accuracy, whose
absence would make the result suspect rather than good.

Usage: ``uv run python ... [--horizons-from N]``
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

CANDIDATES = ("two-horizon", "gated")


def tied_response(panel, response: np.ndarray, rows: np.ndarray):
    """A(aggregate) interpolated from the tied rows available in `rows`.

    The tied policies trace a one-dimensional curve in aggregate, so the control for any policy is
    evaluated rather than looked up, and every untied row gets an exact-aggregate counterpart.
    """
    tied = rows[panel.tied[rows]]
    order = np.argsort(panel.aggregate[tied])
    x = panel.aggregate[tied][order]
    y = response[tied][order]
    unique, index = np.unique(x, return_index=True)
    return unique, y[index]


def contrast(panel, response: np.ndarray, rows: np.ndarray, evaluate: np.ndarray) -> np.ndarray:
    x, y = tied_response(panel, response, rows)
    if len(x) < 4:
        return np.zeros(len(evaluate))
    return response[evaluate] - np.interp(panel.aggregate[evaluate], x, y)


def select(panel, response, folds, name: str, criterion: str, seed: int = 20260812) -> np.ndarray:
    box = stage1.bounds_for(name)

    def objective(theta):
        columns = stage1.design(panel, name, theta)
        if not np.isfinite(columns).all():
            return 1e6
        total = 0.0
        for train, test in folds:
            coefficients = stage1.solve(columns[train], response[train])
            predicted = columns[test] @ coefficients
            if criterion == "level":
                residual = predicted - response[test]
            else:
                # The model must reproduce the ordering effect using ITS OWN tied baseline. Subtracting a
                # common reference from both sides would cancel and leave the level criterion exactly --
                # a first attempt did that and reproduced the level numbers to the last decimal.
                x, y = tied_response(panel, response, train)
                if len(x) < 4:
                    return 1e6
                matched = stage1._grid_panel(panel, panel.aggregate[test], panel.aggregate[test])
                model_tied = stage1.design(matched, name, theta) @ coefficients
                observed_tied = np.interp(panel.aggregate[test], x, y)
                residual = (predicted - model_tied) - (response[test] - observed_tied)
            total += float(residual @ residual)
        return total

    return differential_evolution(
        objective,
        box,
        rng=np.random.default_rng(seed),
        popsize=10,
        maxiter=20,
        tol=1e-10,
        polish=True,
        init="sobol",
    ).x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons-from", type=int, default=2)
    args = parser.parse_args()

    frame = panel_module.load_full_pool()
    panels = panel_module.panels_by_horizon(frame)[args.horizons_from :]
    targets = panel_module.atomic_targets()

    records = []
    for panel in panels:
        folds = panel_module.spatial_folds(panel)
        for key in targets:
            y = panel.target(key)
            best = int(np.argmin(y))
            truth = (float(panel.phase_0[best]), float(panel.phase_1[best]))
            floor = np.empty_like(y)
            for train, test in folds:
                floor[test] = y[train].mean()
            base = float(np.sqrt(np.mean((floor - y) ** 2)))
            record = {"untied": not np.isclose(*truth)}
            for name in CANDIDATES:
                for criterion in ("level", "contrast"):
                    theta = select(panel, y, folds, name, criterion)
                    columns = stage1.design(panel, name, theta)
                    predictions = np.empty_like(y)
                    for train, test in folds:
                        predictions[test] = columns[test] @ stage1.solve(columns[train], y[train])
                    tag = f"{name}/{criterion}"
                    record[tag + "_fit"] = float(np.sqrt(np.mean((predictions - y) ** 2))) / base
                    coefficients = stage1.solve(columns, y)
                    axis = np.linspace(0.0, 1.0, stage1.GRID)
                    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
                    grid = stage1._grid_panel(panel, g0.ravel(), g1.ravel())
                    tied_grid = stage1._grid_panel(panel, axis, axis)
                    surface = stage1.design(grid, name, theta) @ coefficients
                    tied_surface = stage1.design(tied_grid, name, theta) @ coefficients
                    cell = int(np.argmin(surface))
                    record[tag + "_gain"] = float(tied_surface.min() - surface.min())
                    record[tag + "_dist"] = float(np.hypot(g0.ravel()[cell] - truth[0], g1.ravel()[cell] - truth[1]))
            records.append(record)
        print(f"  ...horizon {panel.horizon:.3f}B done", flush=True)

    untied = np.array([r["untied"] for r in records])
    print(f"\nATOM-006 CONTRAST CRITERION  n={len(records)}, untied={untied.sum()}, tied={(~untied).sum()}")
    for name in CANDIDATES:
        print(f"  {name}")
        for criterion in ("level", "contrast"):
            tag = f"{name}/{criterion}"
            gain = np.array([r[tag + "_gain"] for r in records])
            fit = np.array([r[tag + "_fit"] for r in records])
            distance = np.array([r[tag + "_dist"] for r in records])
            print(
                f"     {criterion:8s} median predicted 2p gain {np.median(gain):+.5f}  max {gain.max():+.5f}  "
                f"non-zero on {(np.abs(gain) > 1e-6).sum():2d}/{len(gain)}"
            )
            print(
                f"              level fit {np.median(fit):.4f}   distance: untied "
                f"{np.median(distance[untied]):.4f}  tied {np.median(distance[~untied]):.4f}"
            )


if __name__ == "__main__":
    main()
