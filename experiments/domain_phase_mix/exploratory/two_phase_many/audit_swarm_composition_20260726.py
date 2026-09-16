# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How to spend a fixed 280-row training budget: how many rows, and which ones.

There is no surplus pool of trained 300M two-phase policies to reselect from. The
280-row fit panel is the entire design, so a better swarm construction cannot be
demonstrated by subsetting a larger set. What the existing data can answer is the pair
of questions that would inform building one:

*Is the budget binding?* If a model fitted on 180 rows is nearly as good as one fitted
on 280, then row count is not the constraint and the interesting lever is which
policies get sampled. If the curve is still climbing at 280, more rows are the cheapest
improvement available.

*Which composition earns its place?* The panel is 241 qsplit-signal rows plus 39
domain-deletion rows. Domain deletion costs 14 percent of the budget and probes a very
different region: one bucket removed entirely rather than a smooth quality tilt. Holding
count fixed and varying composition prices that choice.

*Does a pre-specified selection rule beat random?* At a reduced count there is a surplus
to select from, so candidate rules can be compared honestly. Every rule here is
specified from policy coordinates alone and never consults an outcome, which is what
keeps this from being selection on the evaluation set. The rules are scored on the rows
they did not select plus the censored best rows.

Selection rules, all outcome-blind:
``random``            uniform, the control.
``space_filling``     greedy farthest-point in aggregate coordinates, maximizing coverage.
``leverage``          greedy maximization of design-matrix log-determinant, the D-optimal
                      idea applied to the surrogate's own feature space.
``epoch_stratified``  equal counts per quantile of maximum simulated epochs, so the
                      repetition regime is covered uniformly.
``contrast_balanced`` equal counts per quantile of phase total variation, so the
                      two-phase contrast axis is covered uniformly.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import aggregate_of, build_benchmark, fit_on, select_by  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "swarm_composition_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
CENSOR_FRACTION = 0.10
LEARNING_CURVE_SIZES = (80, 120, 160, 200, 240, 252)
LEARNING_CURVE_DRAWS = 12
SELECTION_SIZE = 160
SELECTION_DRAWS = 8
SEED = 20260726
# Fixed shape for the count and composition sweeps. Reselecting the shape inside every
# subset would confound "fewer rows hurt the fit" with "fewer rows changed the shape",
# and the shape is not what these questions are about.
FIXED_L2_GRID = (0.0,)


def hpr_model() -> Model:
    return Model("hpr", build_hierarchical_phase_replay, lambda: _state_shapes(True), l2_grid=FIXED_L2_GRID)


def censored_split(observed: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    available = np.isfinite(observed)
    n_censored = max(1, int(fraction * available.sum()))
    ordering = np.argsort(np.where(available, observed, np.inf))
    censored = np.zeros(len(observed), dtype=bool)
    censored[ordering[:n_censored]] = True
    return available & ~censored, censored


def score(
    panel: Panel, model: Model, target: str, shape: dict, l2: float, train: np.ndarray, evaluate: np.ndarray
) -> dict[str, float]:
    fitted = fit_on(panel, model, target, shape, l2, rows=train)
    prediction = fitted.predict(panel)[evaluate]
    truth = panel.targets[target][evaluate]
    finite = np.isfinite(prediction) & np.isfinite(truth)
    prediction, truth = prediction[finite], truth[finite]
    residual = prediction - truth
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return {
        "n_eval": int(finite.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
    }


def space_filling(coordinates: np.ndarray, pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point selection in aggregate coordinates."""
    chosen = [int(rng.choice(pool))]
    distance = np.abs(coordinates[pool] - coordinates[chosen[0]]).sum(axis=1) / 2.0
    while len(chosen) < size:
        nxt = int(pool[np.argmax(distance)])
        chosen.append(nxt)
        distance = np.minimum(distance, np.abs(coordinates[pool] - coordinates[nxt]).sum(axis=1) / 2.0)
    return np.asarray(chosen)


def leverage(design: np.ndarray, pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy log-determinant maximization on the surrogate's own design matrix."""
    chosen = [int(rng.choice(pool))]
    ridge = 1e-6 * np.eye(design.shape[1])
    gram = np.outer(design[chosen[0]], design[chosen[0]]) + ridge
    remaining = set(pool.tolist()) - set(chosen)
    while len(chosen) < size:
        inverse = np.linalg.inv(gram)
        candidates = np.fromiter(remaining, dtype=int)
        rows = design[candidates]
        gains = np.einsum("ij,jk,ik->i", rows, inverse, rows)
        nxt = int(candidates[np.argmax(gains)])
        chosen.append(nxt)
        remaining.discard(nxt)
        gram = gram + np.outer(design[nxt], design[nxt])
    return np.asarray(chosen)


def stratified(values: np.ndarray, pool: np.ndarray, size: int, rng: np.random.Generator, bins: int = 8) -> np.ndarray:
    """Equal counts per quantile bin of the supplied statistic."""
    edges = np.quantile(values[pool], np.linspace(0, 1, bins + 1))
    chosen: list[int] = []
    per_bin = max(1, size // bins)
    for low, high in itertools.pairwise(edges):
        members = pool[(values[pool] >= low) & (values[pool] <= high)]
        members = np.setdiff1d(members, np.asarray(chosen, dtype=int))
        if len(members) == 0:
            continue
        take = min(per_bin, len(members))
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())
    leftover = np.setdiff1d(pool, np.asarray(chosen, dtype=int))
    if len(chosen) < size and len(leftover):
        chosen.extend(rng.choice(leftover, size=min(size - len(chosen), len(leftover)), replace=False).tolist())
    return np.asarray(chosen[:size])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_300m
    model = hpr_model()
    rng = np.random.default_rng(SEED)
    coordinates = aggregate_of(panel)
    max_epochs = panel.epochs.max(axis=1)
    contrast = panel.phase_tv

    curve: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []

    for target in TARGETS:
        observed = panel.targets[target]
        train, censored = censored_split(observed, CENSOR_FRACTION)
        shape, l2, _ = select_by(panel, model, target, "rmse", rows=train)
        pool = np.flatnonzero(train)
        design = model.build(panel, shape).matrix

        # --- how many rows? ---
        for size in LEARNING_CURVE_SIZES:
            if size > len(pool):
                continue
            for draw in range(LEARNING_CURVE_DRAWS):
                picked = rng.choice(pool, size=size, replace=False)
                rows = np.zeros(len(observed), dtype=bool)
                rows[picked] = True
                held = train & ~rows
                curve.append(
                    {
                        "target": target,
                        "size": size,
                        "draw": draw,
                        **{f"held_{k}": v for k, v in score(panel, model, target, shape, l2, rows, held).items()},
                        **{f"cens_{k}": v for k, v in score(panel, model, target, shape, l2, rows, censored).items()},
                    }
                )

        # --- which composition, at fixed count? ---
        qsplit = pool[panel.series[pool] != "domain_deletion"]
        deletion = pool[panel.series[pool] == "domain_deletion"]
        budget = min(len(qsplit), 200)
        mixes = {
            "qsplit_only": (budget, 0),
            "qsplit_plus_all_deletions": (budget - len(deletion), len(deletion)),
            "qsplit_plus_half_deletions": (budget - len(deletion) // 2, len(deletion) // 2),
        }
        for label, (n_qsplit, n_deletion) in mixes.items():
            if n_qsplit <= 0 or n_deletion > len(deletion):
                continue
            for draw in range(LEARNING_CURVE_DRAWS):
                picked = np.concatenate(
                    [
                        rng.choice(qsplit, size=n_qsplit, replace=False),
                        rng.choice(deletion, size=n_deletion, replace=False) if n_deletion else np.empty(0, int),
                    ]
                )
                rows = np.zeros(len(observed), dtype=bool)
                rows[picked.astype(int)] = True
                composition.append(
                    {
                        "target": target,
                        "mix": label,
                        "draw": draw,
                        "n_total": int(rows.sum()),
                        "n_deletion": n_deletion,
                        **{f"cens_{k}": v for k, v in score(panel, model, target, shape, l2, rows, censored).items()},
                    }
                )

        # --- which rows, at a count that leaves a surplus? ---
        rules = {
            "random": lambda p, s, r: r.choice(p, size=s, replace=False),
            "space_filling": lambda p, s, r: space_filling(coordinates, p, s, r),
            "leverage": lambda p, s, r, _design=design: leverage(_design, p, s, r),
            "epoch_stratified": lambda p, s, r: stratified(max_epochs, p, s, r),
            "contrast_balanced": lambda p, s, r: stratified(contrast, p, s, r),
        }
        for label, rule in rules.items():
            for draw in range(SELECTION_DRAWS):
                picked = np.asarray(rule(pool, SELECTION_SIZE, np.random.default_rng(SEED + draw)), dtype=int)
                rows = np.zeros(len(observed), dtype=bool)
                rows[picked] = True
                held = train & ~rows
                selection.append(
                    {
                        "target": target,
                        "rule": label,
                        "draw": draw,
                        "n_selected": int(rows.sum()),
                        **{f"held_{k}": v for k, v in score(panel, model, target, shape, l2, rows, held).items()},
                        **{f"cens_{k}": v for k, v in score(panel, model, target, shape, l2, rows, censored).items()},
                    }
                )
        print(f"  finished {target}")

    curve_frame = pd.DataFrame(curve)
    composition_frame = pd.DataFrame(composition)
    selection_frame = pd.DataFrame(selection)
    curve_frame.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
    composition_frame.to_csv(OUTPUT_DIR / "composition.csv", index=False)
    selection_frame.to_csv(OUTPUT_DIR / "selection_rules.csv", index=False)

    print("\n=== is the 280-row budget binding? (mean over draws) ===")
    print(
        curve_frame.groupby(["target", "size"])[["held_rmse", "cens_rmse", "cens_bias", "cens_spearman"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:.5f}")
    )
    print("\n=== does domain deletion earn its 14 percent of the budget? ===")
    print(
        composition_frame.groupby(["target", "mix"])[["n_total", "cens_rmse", "cens_bias", "cens_spearman"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:.5f}")
    )
    print(f"\n=== outcome-blind selection rules at n={SELECTION_SIZE} (mean over draws) ===")
    print(
        selection_frame.groupby(["target", "rule"])[["held_rmse", "cens_rmse", "cens_bias", "cens_spearman"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:.5f}")
    )

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "censor_fraction": CENSOR_FRACTION,
                "learning_curve_sizes": list(LEARNING_CURVE_SIZES),
                "selection_size": SELECTION_SIZE,
                "seed": SEED,
                "note": "every selection rule reads policy coordinates only and never an outcome",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
