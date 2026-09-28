# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Ask the fitted surrogate whether two-phase mixing is worth anything at all (ATOM-033).

The held-out panels contain almost no two-phase advantage to find: at 300M every held-out row is
single-phase, and at 3e18 the best two-phase row beats the best single-phase row by 0.59 run noises on
Table 9 and 3.42 on Uncheatable. That measures the DATA. This measures the MODEL -- searching the fitted
surrogate's own predicted surface over the two-phase simplex and over the single-phase diagonal inside
it, and comparing the two optima.

The comparison is informative in both directions. A model that predicts a large two-phase gain where the
panel shows none is miscalibrated in a specific, locatable way. A model that predicts no gain agrees with
the panel, and then the programme's premise -- that a two-phase frontier exists at this scale -- is what
needs evidence, not the surrogate.

Search is by batched sampling rather than a quasi-Newton solve because `predict` vectorises over rows and
a numerical gradient over 78 dimensions does not.

The headline finding is that this comparison DOES NOT CONVERGE at any budget reachable here, so no number
it produces should be quoted as the model's predicted two-phase gain. The two sides are searched over
spaces of different dimension -- a 39-dimensional diagonal inside a 78-dimensional space -- and they
therefore converge at different rates. Raising the budget from 5k to 25k improves the single-phase optimum
by about 0.006 while the two-phase optimum moves less, so the apparent gain collapses, and on some cells
it changes sign. An early run of this script at a low budget appeared to show the model predicting gains
of fourteen to thirty-seven run noises; that was the search, not the model. The sweep over BUDGETS is
printed so the non-convergence is visible rather than inferred.

What survives search entirely, and is where the optimism claim should rest, is the held-out bias measured
by separation band: predictions run about 0.001 low on single-phase rows and about 0.02 low above TV 0.55,
on 1957 rows with no optimisation involved.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import evaluate_swarm39_selection_20260823 as selection  # noqa: E402
import fit_swarm39_split_damage_20260817 as split_damage  # noqa: E402
import general_mixture_surrogate_20260809 as gen  # noqa: E402
import numpy as np  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402

CONCENTRATIONS = (2000.0, 500.0, 100.0, 20.0, 5.0)
SEPARATION_CONCENTRATIONS = (5000.0, 1500.0, 400.0, 100.0, 25.0)
SEPARATION_CAPS = (0.05, 0.15, 0.35, 0.55, 1.0)
BUDGETS = (5_000, 12_500, 25_000)
BASE_SAMPLES = 40_000
TWO_PHASE_SAMPLE_MULTIPLE = 4
SEEDS = 3


def dirichlet_around(rng, centre: np.ndarray, count: int) -> np.ndarray:
    """Draws spread over several concentrations, so the search is not tied to one radius."""
    per = max(1, count // len(CONCENTRATIONS))
    blocks = [rng.dirichlet(np.maximum(centre * c, 1e-6), size=per) for c in CONCENTRATIONS]
    return np.concatenate(blocks, axis=0)


def search(fit, target: str, pooling: str, seed: int, samples: int) -> dict[str, float]:
    """Best predicted value the model reaches over two phases, and over the single-phase diagonal."""
    rng = np.random.default_rng(seed)
    index = selection.bucket_pooling(fit, pooling)
    ok = np.isfinite(fit.targets[target])
    fit_panel = gen.Panel(np.stack([fit.phase0[ok], fit.phase1[ok]], axis=1), fit.c0, fit.c1, index)
    response = fit.targets[target][ok]
    fitted = split_damage.fit_variant(fit_panel, response, "split", seed)

    def predicted(early: np.ndarray, late: np.ndarray) -> np.ndarray:
        panel = gen.Panel(np.stack([early, late], axis=1), fit.c0, fit.c1, index)
        return split_damage.predict(panel, fitted, "split")

    centres = [fit.proportional, *fit.phase0[ok][np.argsort(response)[:4]]]
    single = np.concatenate([dirichlet_around(rng, c, samples // len(centres)) for c in centres])

    # Phase 1 is drawn as a perturbation OF phase 0 rather than independently, so the pair's separation is
    # controlled by the concentration and the whole separation range gets covered. Independent draws land
    # almost entirely at large separation, which is the region this search has to be able to exclude.
    early_blocks, late_blocks = [], []
    for centre in centres:
        block = dirichlet_around(rng, centre, samples * TWO_PHASE_SAMPLE_MULTIPLE // len(centres))
        early_blocks.append(block)
        kappa = rng.choice(SEPARATION_CONCENTRATIONS, size=len(block))[:, None]
        late_blocks.append(np.stack([rng.dirichlet(np.maximum(row, 1e-9)) for row in block * kappa]))
    two_early = np.concatenate(early_blocks)
    two_late = np.concatenate(late_blocks)
    separations = 0.5 * np.abs(two_early - two_late).sum(axis=1)

    single_values = predicted(single, single)
    two_values = predicted(two_early, two_late)
    observed = predicted(fit.phase0[ok], fit.phase1[ok])
    result = {"single": float(single_values.min())}
    for cap in SEPARATION_CAPS:
        allowed = separations <= cap
        if not allowed.any():
            continue
        best = float(two_values[allowed].min())
        result[f"gain@{cap}"] = float(single_values.min() - best)
        result[f"sep@{cap}"] = float(separations[allowed][np.argmin(two_values[allowed])])
    result["escapes"] = float(gen.predictions_escape_range(np.array([two_values.min()]), observed))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="delphi_3e18")
    parser.add_argument("--samples", type=int, default=BASE_SAMPLES)
    args = parser.parse_args()

    for scale in args.scales.split(","):
        fit, held = swarm39.load_scale(scale)
        for target in selection.TARGETS:
            noise = selection.run_noise(held, target)
            print(f"\n=== {scale} / {target} ===  run noise {noise:.5f}")
            for pooling in selection.POOLINGS:
                print(f"  {pooling}:")
                for budget in BUDGETS:
                    results = [search(fit, target, pooling, seed, budget) for seed in range(SEEDS)]
                    single = np.median([r["single"] for r in results])
                    gains = np.array([r["gain@1.0"] for r in results if "gain@1.0" in r])
                    capped = np.array([r["gain@0.15"] for r in results if "gain@0.15" in r])
                    ratio = np.median(gains) / noise if np.isfinite(noise) and noise > 0 else float("nan")
                    print(
                        f"      budget {budget:>6}  single {single:.5f}  gain {np.median(gains):+.5f} "
                        f"[{gains.min():+.5f},{gains.max():+.5f}] = {ratio:6.2f} noises   "
                        f"capped at TV 0.15 {np.median(capped):+.5f}"
                    )


if __name__ == "__main__":
    main()
