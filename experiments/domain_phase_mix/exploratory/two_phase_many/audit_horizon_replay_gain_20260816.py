# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The observed two-phase gain across the full horizon-by-replay grid (ATOM-010).

Before fitting anything to the with-repetition case, characterise what a model would have to reproduce.
The confirmation panel established a horizon-by-replay INTERACTION rather than a simple "more repetition
gives more gain": the replay-induced ordering effect decays with horizon while the no-replay recency
effect is horizon-stable. ATOM-009 probed a single horizon and could not see that.

This measures observed gain over all 7 support conditions and all 4 horizons, for all 23 atomic targets,
with no model involved. Gain is best tied minus best untied at each cell, so positive means a two-phase
policy wins.

These are raw grid extrema on one seed each, so every value carries winner's curse and the ABSOLUTE
level is biased upward in magnitude. The pattern ACROSS cells is what this is for, and the independently
reported confirmation numbers give two anchor points to check it against: -0.0059 at full support and
+0.0105 at 4x replay, both at 7.41B.

Usage: ``uv run python ...``
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)

SUPPORTS = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
CODE_MARKERS = ("programing", "github", "arxiv_computer")


def main() -> None:
    targets = panel_module.atomic_targets()
    is_code = np.array([any(marker in key for marker in CODE_MARKERS) for key in targets])
    primary = targets[0]

    grids = {}
    epochs = {}
    for support in SUPPORTS:
        frame = panel_module.load_support(support)
        epochs[support] = panel_module.repetition_summary(frame)["max_epochs"]
        for panel in panel_module.panels_by_horizon(frame):
            gains = []
            for key in targets:
                y = panel.target(key)
                tied = panel.tied
                gains.append(float(y[tied].min() - y[~tied].min()))
            grids[(support, round(panel.horizon, 3))] = np.array(gains)

    horizons = sorted({h for _, h in grids})
    print("ATOM-010 OBSERVED two-phase gain (best tied minus best untied), positive means two-phase wins")
    print("raw one-seed grid extrema: winner's-curse biased in level, informative in pattern\n")

    for label, mask in (("PRIMARY code target", None), ("median over 23 atomic targets", slice(None))):
        print(f"  {label}")
        print("    support   max ep   " + "".join(f"{h:>11.3f}B" for h in horizons))
        for support in SUPPORTS:
            cells = []
            for h in horizons:
                g = grids[(support, h)]
                cells.append(g[0] if mask is None else float(np.median(g)))
            print(f"    {support:<8s} {epochs[support]:7.2f}   " + "".join(f"{c:>+12.5f}" for c in cells))
        print()

    print("  fraction of the 23 targets with a POSITIVE gain")
    print("    support   " + "".join(f"{h:>11.3f}B" for h in horizons))
    for support in SUPPORTS:
        cells = [float((grids[(support, h)] > 0).mean()) for h in horizons]
        print(f"    {support:<8s}  " + "".join(f"{c:>11.0%} " for c in cells))

    print("\n  anchors from the independent confirmation report, for validation:")
    print(f"    full support at 7.408B, primary target : {grids[('full', horizons[-1])][0]:+.5f}  (report -0.0059)")
    print(f"    m400 at 7.408B, primary target         : {grids[('m400', horizons[-1])][0]:+.5f}  (report +0.0105)")
    print(f"    primary target is {primary}")


if __name__ == "__main__":
    main()
