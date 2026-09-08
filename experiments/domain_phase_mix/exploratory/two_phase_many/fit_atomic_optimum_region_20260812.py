# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Fit for the decision, and report the optimum as a region (ATOM-007).

Every fit so far minimises error over the whole mixture square, but the quantity wanted is the location
of one minimum, and no candidate places it accurately. Two changes, both about the decision:

  trust-region  refit under a Gaussian kernel in policy space centred on the model's OWN current
                recommendation, iterated. Weights never depend on observed BPB, which would be circular
                and would chase a one-seed selection-biased grid minimum.

  region        bootstrap rows, refit, and collect the argmin of each replicate. This is the uncertainty
                treatment the round has not delivered, and it also dissolves the reporting hazard from
                ATOM-004: a single-index surface has many equal minima, which a distribution represents
                honestly and an arbitrary grid cell does not.

The bootstrap resamples ROWS and therefore captures FIT uncertainty only. The calibration repeats are
unlaunched, so nothing here captures measurement noise in the BPB values themselves, and coverage is a
statement about the fit rather than about the experiment.

Usage: ``uv run python ... [--horizons-from N] [--replicates N]``
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_contrast_criterion_20260812 as contrast_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

CANDIDATE = "two-horizon"
BANDWIDTH = 0.25
ROUNDS = 3


def weighted_solve(columns: np.ndarray, response: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(weights)[:, None]
    return stage1.solve(columns * root, response * np.sqrt(weights))


def surface(panel, theta, coefficients):
    axis = np.linspace(0.0, 1.0, stage1.GRID)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    flat0, flat1 = g0.ravel(), g1.ravel()
    grid = stage1._grid_panel(panel, flat0, flat1)
    return flat0, flat1, stage1.design(grid, CANDIDATE, theta) @ coefficients


def trust_region_optimum(panel, response, theta, rows=None):
    """Refit under a kernel centred on the model's own recommendation, iterated."""
    rows = np.arange(len(response)) if rows is None else rows
    columns = stage1.design(panel, CANDIDATE, theta)
    weights = np.ones(len(rows))
    where = None
    for _ in range(ROUNDS):
        coefficients = weighted_solve(columns[rows], response[rows], weights)
        flat0, flat1, values = surface(panel, theta, coefficients)
        cell = int(np.argmin(values))
        where = (float(flat0[cell]), float(flat1[cell]))
        distance = np.hypot(panel.phase_0[rows] - where[0], panel.phase_1[rows] - where[1])
        weights = np.exp(-0.5 * (distance / BANDWIDTH) ** 2)
        weights = np.maximum(weights, 1e-3)
    return where


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons-from", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--reselect", action="store_true", help="re-select theta inside each bootstrap replicate")
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
            theta = contrast_module.select(panel, y, folds, CANDIDATE, "level")

            columns = stage1.design(panel, CANDIDATE, theta)
            flat0, flat1, values = surface(panel, theta, stage1.solve(columns, y))
            cell = int(np.argmin(values))
            plain = (float(flat0[cell]), float(flat1[cell]))
            local = trust_region_optimum(panel, y, theta)

            rng = np.random.default_rng(20260812)
            replicates = []
            for _ in range(args.replicates):
                rows = rng.integers(0, len(y), len(y))
                # Holding theta fixed excludes SELECTION uncertainty, which the first run showed to be the
                # dominant term: head-only regions covered the truth 24 percent of the time at a nominal
                # 90, and were NARROWEST on the untied cells where accuracy was worst.
                replicate_theta = theta
                if args.reselect:
                    sub = panel_module.spatial_folds(panel)
                    sub = [(np.intersect1d(a, rows), np.intersect1d(b, rows)) for a, b in sub]
                    sub = [(a, b) for a, b in sub if len(a) >= 8 and len(b) >= 4]
                    if sub:
                        replicate_theta = contrast_module.select(panel, y, sub, CANDIDATE, "level")
                    replicate_columns = stage1.design(panel, CANDIDATE, replicate_theta)
                else:
                    replicate_columns = columns
                coefficients = stage1.solve(replicate_columns[rows], y[rows])
                flat0b, flat1b, valuesb = surface(panel, replicate_theta, coefficients)
                spot = int(np.argmin(valuesb))
                replicates.append((flat0b[spot], flat1b[spot]))
            replicates = np.array(replicates)

            # Region = the smallest axis-aligned box covering 90 percent of replicate argmins.
            low = np.percentile(replicates, 5, axis=0)
            high = np.percentile(replicates, 95, axis=0)
            covered = bool(low[0] <= truth[0] <= high[0] and low[1] <= truth[1] <= high[1])
            records.append(
                {
                    "untied": not np.isclose(*truth),
                    "plain": float(np.hypot(plain[0] - truth[0], plain[1] - truth[1])),
                    "local": float(np.hypot(local[0] - truth[0], local[1] - truth[1])),
                    "covered": covered,
                    "width": float(np.hypot(high[0] - low[0], high[1] - low[1])),
                }
            )
        print(f"  ...horizon {panel.horizon:.3f}B done", flush=True)

    untied = np.array([r["untied"] for r in records])
    print(
        f"\nATOM-007  n={len(records)}, untied={untied.sum()}, tied={(~untied).sum()}, "
        f"{args.replicates} bootstrap replicates, bandwidth {BANDWIDTH}"
    )
    for label, mask in (("UNTIED", untied), ("TIED", ~untied), ("ALL", np.ones(len(records), bool))):
        plain = np.array([r["plain"] for r in records])[mask]
        local = np.array([r["local"] for r in records])[mask]
        covered = np.array([r["covered"] for r in records])[mask]
        width = np.array([r["width"] for r in records])[mask]
        print(
            f"  {label:6s} n={mask.sum():2d}   distance: plain {np.median(plain):.4f} -> "
            f"trust-region {np.median(local):.4f}   better on {(local < plain).sum()}/{mask.sum()}"
        )
        print(
            f"             90% region: covers the empirical optimum {covered.mean():.0%}   "
            f"median width {np.median(width):.4f}"
        )


if __name__ == "__main__":
    main()
