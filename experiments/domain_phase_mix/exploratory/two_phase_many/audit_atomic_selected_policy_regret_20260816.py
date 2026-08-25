# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Re-score the atomic candidates on selected-policy BPB REGRET, not coordinate distance (ATOM-008).

The WSD80 deployment gate has dropped coordinate distance: two policies far apart in mixture space but
statistically indistinguishable in BPB belong to the same usable basin, so distance was measuring the
wrong thing. Every atomic conclusion recorded so far -- ATOM-004's finding that the second horizon's
benefit was an argmin tie-breaking artifact, and the placement framing in ATOM-005 and ATOM-006 -- was
scored in coordinates. Those conclusions have to be re-derived under the metric that now decides.

The replacement is what a deployment would actually experience: take the candidate's recommended policy,
find the nearest OBSERVED policy on the panel, and report its BPB shortfall against the panel's best
observed BPB. No new runs are needed and no interpolation is invented.

This deliberately keeps the panel's raw argmin as the reference, which is a one-seed selection-biased
minimum. That biases every candidate's regret upward by the same winner's-curse amount, so the RANKING
between candidates is meaningful while the absolute level is not, and the level must not be quoted as a
deployment regret.

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
    fit_atomic_contrast_criterion_20260812 as contrast_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

CANDIDATES = ("two-bucket", "two-horizon", "gated")


def nearest_observed(panel, where) -> int:
    return int(np.argmin(np.hypot(panel.phase_0 - where[0], panel.phase_1 - where[1])))


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
            spread = float(y.max() - y.min())
            record = {"untied": not np.isclose(*truth), "spread": spread}
            for name in CANDIDATES:
                theta = contrast_module.select(panel, y, folds, name, "level")
                columns = stage1.design(panel, name, theta)
                coefficients = stage1.solve(columns, y)
                axis = np.linspace(0.0, 1.0, stage1.GRID)
                g0, g1 = np.meshgrid(axis, axis, indexing="ij")
                grid = stage1._grid_panel(panel, g0.ravel(), g1.ravel())
                cell = int(np.argmin(stage1.design(grid, name, theta) @ coefficients))
                where = (float(g0.ravel()[cell]), float(g1.ravel()[cell]))
                landed = nearest_observed(panel, where)
                record[f"{name}_regret"] = float(y[landed] - y[best])
                record[f"{name}_dist"] = float(np.hypot(where[0] - truth[0], where[1] - truth[1]))
            records.append(record)
        print(f"  ...horizon {panel.horizon:.3f}B done", flush=True)

    untied = np.array([r["untied"] for r in records])
    spread = np.array([r["spread"] for r in records])
    print(f"\nATOM-008 SELECTED-POLICY BPB REGRET  n={len(records)}, untied={untied.sum()}")
    print(f"  panel BPB spread (max-min) median {np.median(spread):.4f}, so regret is on that scale")
    print("  reference is the raw one-seed argmin, so the LEVEL is winner's-curse biased; ranking is the point\n")
    for label, mask in (("UNTIED", untied), ("TIED", ~untied), ("ALL", np.ones(len(records), bool))):
        print(f"  {label:6s} n={mask.sum():2d}")
        for name in CANDIDATES:
            regret = np.array([r[f"{name}_regret"] for r in records])[mask]
            distance = np.array([r[f"{name}_dist"] for r in records])[mask]
            share = regret / np.maximum(spread[mask], 1e-12)
            print(
                f"     {name:11s} regret median {np.median(regret):.5f} BPB  "
                f"({np.median(share):.1%} of panel spread)   "
                f"within 0.002 BPB on {(regret <= 0.002).sum():2d}/{mask.sum()}   "
                f"[coordinate distance {np.median(distance):.4f}]"
            )


if __name__ == "__main__":
    main()
