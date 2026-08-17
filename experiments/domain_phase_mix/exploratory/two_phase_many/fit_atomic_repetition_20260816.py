# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The with-repetition two-bucket case, one atomic objective at a time (ATOM-009).

Everything in ATOM-001 through ATOM-008 used the zero-replay panel, where the maximum StarCoder exposure
is 0.034 epochs and there is NO positive two-phase gain to predict -- best tied beats best untied by
0.0059 BPB at the longest horizon. A model that cannot express phase structure therefore scores well
there regardless, which is how a single-index defect survived several rounds.

The replay conditions are where the effect exists, and the gain grows with repetition: at 7.41B tokens
the untied advantage is -0.0059 at full support, +0.0030 at m025, and +0.0105 at m400. Those match the
independently reported full-pool and 4x-replay confirmation numbers to the digit.

So this panel can finally score the thing that matters: PREDICTED two-phase gain against an OBSERVED,
non-zero one. Candidates with and without a repetition-damage term are compared, since damage is inert
by construction on the zero-replay panel and only becomes testable here. m400's maximum exposure of
106.11 epochs is also the first to reach the 105-excess-epoch damage knee measured elsewhere.

Usage: ``uv run python ... [--supports m025,m400] [--horizon-index -1]``
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

CANDIDATES = ("two-bucket", "two-horizon", "two-bucket-damage", "two-horizon-damage")


def observed_gain(panel, response: np.ndarray) -> float:
    tied = panel.tied
    if not tied.any() or tied.all():
        return float("nan")
    return float(response[tied].min() - response[~tied].min())


def evaluate(panel, response, folds, name):
    theta = contrast_module.select(panel, response, folds, name, "level")
    columns = stage1.design(panel, name, theta)
    predictions = np.empty_like(response)
    for train, test in folds:
        predictions[test] = columns[test] @ stage1.solve(columns[train], response[train])
    coefficients = stage1.solve(columns, response)

    axis = np.linspace(0.0, 1.0, stage1.GRID)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    flat0, flat1 = g0.ravel(), g1.ravel()
    grid = stage1._grid_panel(panel, flat0, flat1)
    tied_grid = stage1._grid_panel(panel, axis, axis)
    surface = stage1.design(grid, name, theta) @ coefficients
    tied_surface = stage1.design(tied_grid, name, theta) @ coefficients
    cell = int(np.argmin(surface))
    where = (float(flat0[cell]), float(flat1[cell]))
    landed = int(np.argmin(np.hypot(panel.phase_0 - where[0], panel.phase_1 - where[1])))
    return {
        "fit": float(np.sqrt(np.mean((predictions - response) ** 2))),
        "gain": float(tied_surface.min() - surface.min()),
        "regret": float(response[landed] - response.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supports", default="m025,m400")
    parser.add_argument("--horizon-index", type=int, default=-1)
    args = parser.parse_args()

    targets = panel_module.atomic_targets()
    for support in args.supports.split(","):
        frame = panel_module.load_support(support)
        summary = panel_module.repetition_summary(frame)
        panel = panel_module.panels_by_horizon(frame)[args.horizon_index]
        folds = panel_module.spatial_folds(panel)
        print(
            f"=== support {support}: max {summary['max_epochs']:.2f} epochs, "
            f"{summary['rows_repeated']:.0%} of rows repeated, horizon {panel.horizon:.3f}B ===",
            flush=True,
        )

        rows = {name: {"fit": [], "gain_error": [], "regret": []} for name in CANDIDATES}
        truth = []
        for key in targets:
            y = panel.target(key)
            actual = observed_gain(panel, y)
            truth.append(actual)
            for name in CANDIDATES:
                result = evaluate(panel, y, folds, name)
                rows[name]["fit"].append(result["fit"])
                rows[name]["gain_error"].append(abs(result["gain"] - actual))
                rows[name]["regret"].append(result["regret"])
        truth = np.array(truth)
        print(
            f"  observed two-phase gain across {len(truth)} targets: "
            f"median {np.median(truth):+.5f}, positive on {(truth > 0).sum()}/{len(truth)}"
        )
        for name in CANDIDATES:
            fit = np.array(rows[name]["fit"])
            error = np.array(rows[name]["gain_error"])
            regret = np.array(rows[name]["regret"])
            print(
                f"     {name:20s} RMSE {np.median(fit):.5f}   |gain error| {np.median(error):.5f}   "
                f"selected-policy regret {np.median(regret):.5f}   within 0.002 on "
                f"{(regret <= 0.002).sum():2d}/{len(regret)}",
                flush=True,
            )
        print()


if __name__ == "__main__":
    main()
