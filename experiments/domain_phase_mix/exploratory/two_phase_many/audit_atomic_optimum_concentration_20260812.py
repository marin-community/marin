# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Does a candidate's extra structure earn its keep WHERE IT WAS ADDED FOR? (ATOM-004/005)

The single-index result makes a specific prediction testable. A term added so that untied optima become
representable should improve the recommended optimum on the target-horizons whose empirical optimum is
UNTIED, and do little where the optimum is tied. If the advantage instead spreads evenly, or lands on the
tied cells, the term is buying generic flexibility and its mechanistic justification is wrong.

That test already refuted the second horizon: its distance improvement was twelvefold larger on tied
cells than untied ones. This runs the same test for the gated candidate, which encodes phase ORDER rather
than any weighting of accumulated exposure -- late StarCoder absorption is gated by how much complement
was seen early, so the term is a product of a phase-1 quantity with a function of phase-0 quantities and
cannot be written as a function of any single weighted sum.

A caution the earlier run established: counting how often a candidate PLACES an untied optimum is a
misleading diagnostic, because a single-index surface has many equal minima and the grid argmin breaks
the tie arbitrarily. Predicted GAIN and distance-to-empirical-optimum are the honest measures.

COMPARISON SETTINGS, not reported-number settings: the horizons and optimiser budget are reduced and
matched across arms, per the standing practice that design comparisons run cheap and matched while
promotion claims run full.

Usage: ``uv run python ... [--horizons-from N]``
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
    fit_atomic_stage1_20260811 as stage1,
)

CANDIDATES = ("two-bucket", "two-horizon", "gated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons-from", type=int, default=2, help="skip the shortest horizons")
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
                predictions = stage1.fit_predict(panel, name, y, folds)
                record[f"{name}_fit"] = float(np.sqrt(np.mean((predictions - y) ** 2))) / base
                where, gain, _ = stage1.surface_optimum(panel, name, y)
                record[f"{name}_dist"] = float(np.hypot(where[0] - truth[0], where[1] - truth[1]))
                record[f"{name}_gain"] = gain
            records.append(record)
        print(f"  ...horizon {panel.horizon:.3f}B done", flush=True)

    untied = np.array([r["untied"] for r in records])
    print(
        f"\nDOES THE EXTRA STRUCTURE CONCENTRATE ON UNTIED CELLS?  n={len(records)}, "
        f"untied={untied.sum()}, tied={(~untied).sum()}"
    )
    for label, mask in (("UNTIED empirical optimum", untied), ("TIED empirical optimum", ~untied)):
        print(f"  {label}  (n={mask.sum()})")
        reference = np.array([r["two-bucket_dist"] for r in records])[mask]
        for name in CANDIDATES:
            distance = np.array([r[f"{name}_dist"] for r in records])[mask]
            fit = np.array([r[f"{name}_fit"] for r in records])[mask]
            gain = np.array([r[f"{name}_gain"] for r in records])[mask]
            closer = (
                ""
                if name == "two-bucket"
                else f"   closer than two-bucket on {(distance < reference).sum()}/{mask.sum()}"
            )
            print(
                f"     {name:11s} distance {np.median(distance):.4f}   fit {np.median(fit):.4f}   "
                f"median predicted 2p gain {np.median(gain):+.5f}{closer}"
            )


if __name__ == "__main__":
    main()
