# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""One mechanism across the whole replay ladder, scored on fresh seeds (ATOM-014).

Every fit in ATOM-009 through ATOM-013 treated the seven replay conditions as seven separate problems,
125 noisy rows each. Measured against fresh seeds that produced predictions over-dispersed by about
tenfold: `two-horizon` ranks the 28 blocks well (Spearman 0.610) but a regression of fresh gain on its
prediction has slope 0.093, so it gets the ordering roughly right and the magnitude badly wrong.

The seven supports are not seven problems. They are the same experiment with the StarCoder pool size
changed, and the pool size enters the model as an INPUT it already reads: the epoch rate per unit phase-0
share runs 2.646, 5.292, 10.584, 21.169, 42.337, 84.674 across the finite-replay ladder, an exact
doubling, with the complement pool identical throughout. The zero-StarCoder cell is the same run shared by
all seven, and its BPB agrees across supports to 0.00000. So one theta and one set of amplitudes should
describe all of them, and requiring that is both a sevenfold increase in rows per parameter and a real
test of whether the mechanism is a mechanism.

Two modes:

  pooled  one fit per horizon over all 875 rows, used to predict every support at that horizon.
  loso    leave-one-support-out: fit on the other six, predict the held-out one. Nothing about the
          held-out pool size is seen except through the epoch inputs, which is exactly the extrapolation
          the 39-bucket setting will ask for.

Usage: ``uv run python ... [--mode pooled,loso] [--candidates a,b] [--workers N]``
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    evaluate_fresh_seed_gain_20260816 as fresh,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_contrast_criterion_20260812 as contrast_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

SUPPORTS = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")


@cache
def _panel(support: str, rung: int):
    return panel_module.panels_by_horizon(panel_module.load_support(support))[rung]


@cache
def _pooled(rung: int, supports: tuple[str, ...]):
    """These supports at one horizon stacked into a single panel."""
    combined = pd.concat([_panel(s, rung).frame for s in supports], ignore_index=True)
    return panel_module.AtomicPanel(combined, _panel(supports[0], rung).horizon)


def fit(rung: int, supports: tuple[str, ...], name: str, seed: int):
    """One theta and one amplitude vector for the whole ladder at this horizon."""
    panel = _pooled(rung, supports)
    response = panel.target(fresh.PRIMARY)
    folds = panel_module.spatial_folds(panel, seed=seed)
    theta = contrast_module.select(panel, response, folds, name, "level", seed=20260812 + seed)
    columns = stage1.design(panel, name, theta)
    return theta, stage1.solve(columns, response, stage1.shrink_for(name, theta))


def predict(rung: int, supports: tuple[str, ...], name: str, targets, seed: int = 0) -> list[dict]:
    """Predict each block's frozen policy pair from a single ladder-wide fit."""
    theta, coefficients = fit(rung, supports, name, seed)
    rows = []
    for support, tied, untied in targets:
        panel = _panel(support, rung)
        query = stage1._grid_panel(
            panel, np.array([tied[0], untied[0]], dtype=float), np.array([tied[1], untied[1]], dtype=float)
        )
        values = stage1.design(query, name, theta) @ coefficients
        rows.append(
            {"support": support, "rung": rung, "candidate": name, "predicted_gain": float(values[0] - values[1])}
        )
    return rows


def _work(item):
    return predict(*item)


def clustered_bound(regret: np.ndarray, cluster: np.ndarray, draws: int = 20000) -> float:
    """95th percentile of the mean regret under resampling whole supports.

    The 24 fresh blocks come from 240 separately trained runs, so the outcomes are independent. The
    model's errors are not: the four rungs of one support are fitted from discovery rows that share
    training runs. Resampling supports rather than blocks keeps that dependence intact and is the
    conservative reading of the deployment margin.
    """
    groups = [regret[cluster == name] for name in np.unique(cluster)]
    rng = np.random.default_rng(20260816)
    picks = rng.integers(0, len(groups), size=(draws, len(groups)))
    return float(np.percentile([np.concatenate([groups[i] for i in row]).mean() for row in picks], 95))


HEADINGS = ("candidate", "gain RMSE", "sign", "rho", "p", "regret", "U", "Uclust", "pass")
WIDTHS = (26, 10, 7, 7, 8, 9, 9, 8, 5)


def score(merged: pd.DataFrame, candidates: tuple[str, ...], label: str, floor: float, table=None) -> None:
    print(f"\n{label}")
    print(" ".join(f"{head:>{width}s}" for head, width in zip(HEADINGS, WIDTHS, strict=True)))
    print(f"{'-- measurement noise floor':26s} {floor:10.5f}")
    for name in candidates:
        subset = merged[merged["candidate"] == name].sort_values(["rung", "support"])
        predicted = subset["predicted_gain"].to_numpy()
        truth = subset["fresh_gain"].to_numpy()
        rho, pvalue = stats.spearmanr(predicted, truth)
        regret = np.where(predicted > 0.0, np.maximum(-truth, 0.0), np.maximum(truth, 0.0))
        bound = fresh.upper_bound(regret)
        clustered = clustered_bound(regret, subset["support"].to_numpy())
        print(
            f"{name:26s} {np.sqrt(np.mean((predicted - truth) ** 2)):10.5f} "
            f"{int((np.sign(predicted) == np.sign(truth)).sum()):4d}/{len(truth):<3d} {rho:7.3f} {pvalue:8.4f} "
            f"{regret.mean():9.5f} {bound:9.5f} {clustered:8.5f} "
            f"{'yes' if max(bound, clustered) <= fresh.MARGIN else 'no':>5s}"
        )
    if table is None:
        return
    truth = table["fresh_gain"].to_numpy()
    for name, decision in (
        ("always tied (do nothing)", np.zeros(len(truth), dtype=bool)),
        ("always untied", np.ones(len(truth), dtype=bool)),
        ("follow the discovery argmin", fresh.discovery_gain(table) > 0.0),
    ):
        regret = np.where(decision, np.maximum(-truth, 0.0), np.maximum(truth, 0.0))
        agree = int((np.sign(np.where(decision, 1.0, -1.0)) == np.sign(truth)).sum())
        bound = fresh.upper_bound(regret) if regret.any() else 0.0
        clustered = clustered_bound(regret, table["support"].to_numpy())
        print(
            f"{name:26s} {'':10s} {agree:4d}/{len(truth):<3d} {'':7s} {'':8s} {regret.mean():9.5f} "
            f"{bound:9.5f} {clustered:8.5f} {'yes' if max(bound, clustered) <= fresh.MARGIN else 'no':>5s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="pooled,loso")
    parser.add_argument("--candidates", default=",".join(fresh.CANDIDATES))
    parser.add_argument("--supports", default=",".join(SUPPORTS))
    parser.add_argument("--seed", type=int, default=0, help="fold and search seed; stability is measured, not assumed")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    candidates = tuple(args.candidates.split(","))
    supports = tuple(args.supports.split(","))

    table = fresh.blocks()
    table = table[table["support"].isin(supports)].reset_index(drop=True)
    ranks = {value: index for index, value in enumerate(sorted(table["rung"].unique()))}
    table["rung_index"] = table["rung"].map(ranks)
    floor = float(np.sqrt(np.mean(table["sem"] ** 2)))
    frames = panel_module.load_all_supports()

    for mode in args.modes.split(","):
        items = []
        for rung_index in sorted(table["rung_index"].unique()):
            group = table[table["rung_index"] == rung_index]
            targets = [(row.support, row.tied, row.untied) for row in group.itertuples()]
            if mode == "pooled":
                items.append((int(rung_index), supports, targets))
            else:
                items.extend(
                    (
                        int(rung_index),
                        tuple(name for name in supports if name != row.support),
                        [(row.support, row.tied, row.untied)],
                    )
                    for row in group.itertuples()
                )
        items = [(rung, pool, name, targets, args.seed) for rung, pool, targets in items for name in candidates]

        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=panel_module.seed_cache, initargs=(frames,)
        ) as pool:
            rows = [row for batch in pool.map(_work, items, chunksize=1) for row in batch]
        predictions = pd.DataFrame(rows).rename(columns={"rung": "rung_index"})
        merged = table.merge(predictions, on=["support", "rung_index"])
        held = (
            f"one theta per horizon over all {len(supports)} supports"
            if mode == "pooled"
            else f"fit on {len(supports) - 1} supports, predict the held-out one"
        )
        score(merged, candidates, f"MODE {mode}: {held}  [{', '.join(supports)}]", floor, table)


if __name__ == "__main__":
    main()
