# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The path-attributed damage split at 39 buckets, on a coordinate-disjoint heldout panel (ATOM-016).

Ports the one change that took the two-bucket with-repetition cell inside its deployment margin: splitting
the repetition term's ATTRIBUTION along the run's path, with the departure from the unsplit form shrunk as
a contribution rather than as a bare amplitude. Nothing else about the committed model changes.

Why this swarm and this panel. The 39-bucket exposure geometry is the same as the two-bucket ladder's --
`c1 = c0 (1 - alpha) / alpha`, so per-phase materialized epochs separate while the total is fixed by the
aggregate -- and 44% of (policy, bucket) cells exceed one epoch, so repetition is real. The 300M heldout
cannot be used: all 414 of its rows are tied, and on a tied policy the early share of materialized epochs
is exactly `alpha`, so both split columns collapse to functions of the total and the term is unidentifiable
there. The delphi 3e18 heldout can: 1532 of 1957 rows are untied and its phase total variation reaches
1.000 against 0.657 in the fit panel, so it is a genuine extrapolation rather than an interpolation.

Three variants that nest exactly:

  blended   the committed form, harm charged on exposure read at a FITTED horizon
  physical  the same, with the horizon pinned so the argument is the true total materialized epochs; this
            is the exact nested ablation, verified equal to `blended` at horizon `1 - alpha` to 1.5e-15
  split     that same argument with its attribution split across the phase boundary, the two increments
            summing back to `physical` to 5.6e-17

Scoring avoids the trap ATOM-013 documented. Predicted gain is never compared against a single-seed
argmin; the criteria are heldout level error, and the sign and rank of the observed tied-minus-untied
contrast within aggregate-matched cells. Those cells are the resampling unit -- there are only 20 of them
containing both a tied and an untied policy, two of which hold 395 rows each, so treating 19398 pairs as
independent would overstate the evidence by two orders of magnitude.

Usage: ``uv run python ... [--scale delphi_3e18] [--target uncheatable_bpb] [--seeds 0,1,2]``
"""

import argparse
import collections
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as gen,
)

VARIANTS = ("blended", "physical", "split")
DEPARTURE_BOUND = (-6.0, 2.0)  # log10 weight on the split's departure from the unsplit form


def bridge(panel) -> gen.Panel:
    """The surrogate sees mixtures, per-phase epoch rates and the topic grouping, and nothing else."""
    return gen.Panel(np.stack([panel.phase0, panel.phase1], axis=1), panel.c0, panel.c1, panel.family_index)


def departure_pairs(panel: gen.Panel, variant: str) -> tuple[tuple[int, int], ...]:
    """Index pairs of the family-summed damage increments, which sit after near, late and boundary."""
    if variant != "split":
        return ()
    families = panel.n_families
    return tuple((3 * families + index, 4 * families + index) for index in range(families))


def fit_variant(panel: gen.Panel, response: np.ndarray, variant: str, seed: int):
    box = list(gen.bounds(panel.n_families))
    if variant == "split":
        box.append(DEPARTURE_BOUND)
    folds = swarm39.mixture_blocked_splits(_as_swarm_panel(panel), n_splits=3, seed=seed)

    def head(vector, rows):
        weight = 10.0 ** vector[-1] if variant == "split" else 0.0
        shape, ridge = gen.unpack(vector[:-1] if variant == "split" else vector, panel.n_families)
        subset = gen.Panel(panel.weights[rows], panel.epochs_early, panel.epochs_late, panel.family_index)
        free, constrained = gen.design(subset, shape, variant)
        offsets, amplitudes = gen.fit_head(
            free,
            constrained,
            response[rows],
            ridge,
            gen.pooled_width(panel, variant),
            departure_pairs(panel, variant),
            weight,
        )
        return shape, offsets, amplitudes

    def objective(vector):
        total = 0.0
        for train, test in folds:
            shape, offsets, amplitudes = head(vector, train)
            subset = gen.Panel(panel.weights[test], panel.epochs_early, panel.epochs_late, panel.family_index)
            free, constrained = gen.design(subset, shape, variant)
            residual = free @ offsets + constrained @ amplitudes - response[test]
            if not np.isfinite(residual).all():
                return 1e6
            total += float(residual @ residual)
        return total

    vector = differential_evolution(
        objective,
        box,
        rng=np.random.default_rng(20260817 + seed),
        popsize=8,
        maxiter=15,
        tol=1e-10,
        polish=True,
        init="sobol",
    ).x
    return (vector, *head(vector, np.arange(len(response))))


def predict(panel: gen.Panel, fitted, variant: str) -> np.ndarray:
    _vector, shape, offsets, amplitudes = fitted
    free, constrained = gen.design(panel, shape, variant)
    return free @ offsets + constrained @ amplitudes


class _as_swarm_panel:
    """`mixture_blocked_splits` only needs the aggregate mixture, so expose that and nothing more."""

    def __init__(self, panel: gen.Panel):
        self.weights = panel.weights
        self.phase0, self.phase1 = panel.weights[:, 0, :], panel.weights[:, 1, :]
        self.alpha = 0.8
        self.row_id = np.arange(len(panel.weights))
        self.group = np.arange(len(panel.weights))

    @property
    def aggregate(self) -> np.ndarray:
        return self.alpha * self.phase0 + (1.0 - self.alpha) * self.phase1

    def __len__(self) -> int:
        return len(self.weights)


def matched_cells(panel, tolerance: int = 6) -> list[np.ndarray]:
    """Heldout rows grouped by exact aggregate mixture, keeping cells that hold both kinds of policy."""
    keys = collections.defaultdict(list)
    for index, row in enumerate(np.round(panel.aggregate, tolerance)):
        keys[tuple(row)].append(index)
    untied = panel.phase_tv > 1e-9
    return [np.array(rows) for rows in keys.values() if len(rows) > 1 and untied[rows].any() and (~untied[rows]).any()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="delphi_3e18")
    parser.add_argument("--target", default=swarm39.UNCHEATABLE)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    fit_panel, held_panel = swarm39.load_scale(args.scale)
    fit, held = bridge(fit_panel), bridge(held_panel)
    train_y, held_y = fit_panel.targets[args.target], held_panel.targets[args.target]
    cells = matched_cells(held_panel)
    untied = held_panel.phase_tv > 1e-9

    print(f"ATOM-016 damage split at 39 buckets: {args.scale}, target {args.target}")
    print(f"fit {len(fit_panel)} rows ({int((fit_panel.phase_tv > 1e-9).sum())} untied), ", end="")
    print(f"heldout {len(held_panel)} coordinate-disjoint rows ({int(untied.sum())} untied)")
    print(f"{len(cells)} aggregate-matched cells, sizes {sorted(len(c) for c in cells)}")
    print(f"heldout target spread {held_y.std():.5f} BPB\n")

    header = ("variant", "seed", "heldout RMSE", "cell rho", "cells>0", "sign", "regret", "vs tied")
    print(" ".join(f"{name:>{width}s}" for name, width in zip(header, (10, 4, 13, 9, 8, 7, 9, 9), strict=True)))
    for variant in VARIANTS:
        for seed in (int(value) for value in args.seeds.split(",")):
            fitted = fit_variant(fit, train_y, variant, seed)
            predicted = predict(held, fitted, variant)
            rmse = float(np.sqrt(np.mean((predicted - held_y) ** 2)))

            # Everything below is computed PER CELL and summarised across cells, because the cells are the
            # resampling unit: 790 of the 948 untied heldout rows sit in just two of the twenty cells, so
            # a statistic pooled over rows would be an assertion about those two aggregates.
            rhos, agreements, regrets, tied_regrets = [], [], [], []
            for rows in cells:
                base, alternatives = rows[~untied[rows]], rows[untied[rows]]
                observed = held_y[alternatives] - held_y[base].mean()
                expected = predicted[alternatives] - predicted[base].mean()
                if len(alternatives) >= 4:
                    rhos.append(spearmanr(expected, observed).statistic)
                agreements.append(float(np.mean(np.sign(expected) == np.sign(observed))))
                # The model recommends the alternative it ranks best, and pays what that one actually
                # costs against the best available in the cell. Staying tied is the do-nothing baseline.
                best = min(held_y[alternatives].min(), held_y[base].mean())
                choice = alternatives[int(np.argmin(expected))] if expected.min() < 0.0 else None
                regrets.append((held_y[choice] if choice is not None else held_y[base].mean()) - best)
                tied_regrets.append(held_y[base].mean() - best)
            print(
                f"{variant:10s} {seed:4d} {rmse:13.5f} {np.median(rhos):9.3f} "
                f"{int(np.sum(np.array(rhos) > 0)):3d}/{len(rhos):<4d} {np.mean(agreements):7.3f} "
                f"{np.mean(regrets):9.5f} {np.mean(tied_regrets):9.5f}"
            )


if __name__ == "__main__":
    main()
