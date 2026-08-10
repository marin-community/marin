# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Is cross-scale transfer meaningful, or is the model simply insensitive to theta? (GEN-017)

Once the own-scale arm's selection leak was fixed, transferred theta scored as well as in-scale theta --
penalty 0.999 to 1.055, MSE recovery around 0.94 to 1.00. That reads as a strong transfer result, but it
admits a deflationary explanation: if theta barely matters, then ANY theta transfers and the finding is
vacuous.

This is the control that separates the two. It scores three arms on identical folds with identical heads:

  selected   theta selected in-fold at this scale       -- the ceiling
  random     theta drawn uniformly from the search box  -- the deflationary null
  floor      intercept only                             -- no model at all

If random theta lands near selected, the transfer result says nothing about theta and the whole
cross-scale claim collapses. If random theta lands near the floor, selection is doing real work and
transferred theta preserving it is a genuine finding.

Usage: ``uv run python ... [n_random] [seeds]``, default 24 draws and seeds 0-2.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    transfer_general_surrogate_scales_20260810 as transfer,
)

SCALES = ("60m", "300m")
TARGETS = ("uncheatable_bpb", "table9_macro_bpb")


def main() -> None:
    argv = sys.argv[1:]
    n_random = int(argv[0]) if argv else 24
    seeds = [int(s) for s in argv[1:]] or [0, 1, 2]

    print("GEN-017: does theta matter at all? random-theta control for the cross-scale transfer claim")
    print(f"{n_random} random draws from the search box, seeds {seeds}\n")

    for scale in SCALES:
        raw, _ = swarm.load_scale(scale)
        panel = transfer.to_model_panel(raw)
        for target in TARGETS:
            response = raw.targets[target]
            keep = np.flatnonzero(np.isfinite(response))
            here = transfer.subset(panel, keep)
            y = response[keep]
            groups = raw.group[keep]
            n_families, n_strata = here.n_families, here.n_exposure_strata()
            bounds = np.array(model.bounds(n_families, n_strata))

            for seed in seeds:
                folds = transfer.grouped_folds(groups, transfer.N_FOLDS, seed)
                floor = transfer.floor_rmse(y, folds)
                selected = transfer.nested_own_rmse(here, y, groups, folds, n_families, n_strata, seed)

                rng = np.random.default_rng(20260810 + seed)
                scores = []
                for _ in range(n_random):
                    vector = rng.uniform(bounds[:, 0], bounds[:, 1])
                    value = transfer.oof_rmse(here, y, vector, folds, n_families, n_strata)
                    if np.isfinite(value):
                        scores.append(value)
                scores = np.array(scores)
                # Share of explainable MSE that a random theta already recovers.
                headroom = floor**2 - selected**2
                recovered = (floor**2 - np.median(scores) ** 2) / headroom if headroom > 0 else float("nan")
                print(
                    f" {scale:5s} {target:16s} seed {seed}: selected {selected:.6f}  "
                    f"random median {np.median(scores):.6f}  best {scores.min():.6f}  "
                    f"floor {floor:.6f}   random recovers {recovered:.3f} of explainable MSE"
                )
        print()


if __name__ == "__main__":
    main()
