# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Score candidates against FRESH-SEED two-phase gains rather than discovery argmins (ATOM-013).

Everything in ATOM-009 through ATOM-012 scored predicted gain against `y[best tied] - y[global min]` on a
single-seed panel. Both terms are extrema of a noisy surface, so that target carries a winner's curse:
across the sixteen discovery-positive blocks the discovery gain exceeds the fresh-seed gain by 0.001412
BPB on average, against gains of a few thousandths. The per-run seed noise is 0.00310 BPB (median over 56
five-seed cells), so a difference of two runs carries 0.00439 -- twice the 0.002 deployment margin. A
criterion built on one seed was measuring noise.

The confirmation panel replaces it. Twenty-eight preregistered blocks, one per horizon and replay
condition, each rerunning a frozen tied policy and a frozen untied policy on five fresh seeds. The gain is
the paired mean difference, with a standard error. Those runs were never used to fit anything, so a model
fitted on the discovery panel and asked for the same difference is making a genuine held-out prediction.

Two scores, both over the 28 blocks:

  gain error     RMSE of predicted minus fresh gain, with the noise floor implied by the block standard
                 errors reported alongside it, since no model can beat that.
  decision regret what a practitioner loses by following the model's tied-or-untied choice, measured in
                 fresh BPB against the oracle choice for that block. This is the quantity the 0.002
                 non-inferiority margin was written for, and 28 blocks give it real power where four
                 atomic targets did not.

Usage: ``uv run python ... [--candidates a,b,c] [--workers N]``
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
    fit_atomic_contrast_criterion_20260812 as contrast_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

CONFIRMATION = (
    panel_module.REFERENCE
    / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811"
    / "confirmation_observations.csv"
)
PRIMARY = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
CANDIDATES = (
    "two-bucket",
    "two-horizon",
    "two-bucket-damage",
    "two-horizon-damage",
    "two-bucket-split-damage",
    "two-horizon-split-damage",
)
MARGIN = 0.002  # BPB


def blocks() -> pd.DataFrame:
    """One row per preregistered block: the two frozen policies, the fresh paired gain, its noise."""
    frame = pd.read_csv(CONFIRMATION)
    rows = []
    for (horizon, support), group in frame.groupby(["rung", "support_id"]):
        tied = group[group["policy_role"] == "selected_tied"].sort_values("pair_seed")
        untied = group[group["policy_role"] == "selected_untied"].sort_values("pair_seed")
        paired = tied["observed_bpb"].to_numpy() - untied["observed_bpb"].to_numpy()
        rows.append(
            {
                "rung": horizon,
                "support": support,
                "tied": (float(tied["phase_0_starcoder"].iloc[0]), float(tied["phase_1_starcoder"].iloc[0])),
                "untied": (float(untied["phase_0_starcoder"].iloc[0]), float(untied["phase_1_starcoder"].iloc[0])),
                "fresh_gain": float(paired.mean()),
                "sem": float(paired.std(ddof=1) / np.sqrt(len(paired))),
                "seeds": len(paired),
            }
        )
    return pd.DataFrame(rows).sort_values(["rung", "support"]).reset_index(drop=True)


@cache
def _panels(support: str):
    return tuple(panel_module.panels_by_horizon(panel_module.load_support(support)))


def predict_gain(support: str, rung: int, tied, untied, name: str) -> dict:
    """Fit on the discovery panel for this block, then predict the two frozen policies' difference."""
    panels = _panels(support)
    panel = min(panels, key=lambda p: abs(p.horizon - rung_horizon(panels, rung)))
    response = panel.target(PRIMARY)
    folds = panel_module.spatial_folds(panel)

    theta = contrast_module.select(panel, response, folds, name, "level")
    columns = stage1.design(panel, name, theta)
    coefficients = stage1.solve(columns, response, stage1.shrink_for(name, theta))
    query = stage1._grid_panel(
        panel, np.array([tied[0], untied[0]], dtype=float), np.array([tied[1], untied[1]], dtype=float)
    )
    values = stage1.design(query, name, theta) @ coefficients
    return {"support": support, "rung": rung, "candidate": name, "predicted_gain": float(values[0] - values[1])}


def rung_horizon(panels, rung: int) -> float:
    """The panel horizons are the rung's materialized token counts, in the same ascending order."""
    return sorted(p.horizon for p in panels)[rung]


def _work(item):
    return predict_gain(*item)


def upper_bound(values: np.ndarray) -> float:
    return float(values.mean() + stats.t.ppf(0.95, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    candidates = tuple(args.candidates.split(","))

    table = blocks()
    ranks = {value: index for index, value in enumerate(sorted(table["rung"].unique()))}
    print(f"ATOM-013 fresh-seed scoring: {len(table)} blocks x {len(candidates)} candidates")
    print(f"fresh gain spans {table['fresh_gain'].min():+.5f} to {table['fresh_gain'].max():+.5f} BPB; ")
    print(f"block standard errors span {table['sem'].min():.5f} to {table['sem'].max():.5f}\n")

    frames = panel_module.load_all_supports()
    items = [
        (row.support, ranks[row.rung], row.tied, row.untied, name) for row in table.itertuples() for name in candidates
    ]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=panel_module.seed_cache, initargs=(frames,)) as pool:
        predictions = pd.DataFrame(list(pool.map(_work, items, chunksize=4)))
    predictions["rung"] = predictions["rung"].map({index: value for value, index in ranks.items()})
    merged = table.merge(predictions, on=["support", "rung"])

    # No model can predict the fresh gain better than the fresh gain is itself measured.
    floor = float(np.sqrt(np.mean(table["sem"] ** 2)))
    print(f"{'candidate':26s} {'gain RMSE':>10s} {'sign':>7s} {'regret':>9s} {'U':>9s} {'passes':>7s}")
    print(f"{'-- measurement noise floor':26s} {floor:10.5f}")
    for name in candidates:
        subset = merged[merged["candidate"] == name]
        error = subset["predicted_gain"] - subset["fresh_gain"]
        agree = int((np.sign(subset["predicted_gain"]) == np.sign(subset["fresh_gain"])).sum())
        # A practitioner deploys untied when the model predicts a positive gain, and loses the fresh gain
        # whenever that call was wrong in either direction.
        deploy_untied = subset["predicted_gain"].to_numpy() > 0.0
        fresh = subset["fresh_gain"].to_numpy()
        regret = np.where(deploy_untied, np.maximum(-fresh, 0.0), np.maximum(fresh, 0.0))
        bound = upper_bound(regret)
        print(
            f"{name:26s} {np.sqrt(np.mean(error**2)):10.5f} {agree:4d}/{len(subset):<3d} "
            f"{regret.mean():9.5f} {bound:9.5f} {'yes' if bound <= MARGIN else 'no':>7s}"
        )

    truth = table["fresh_gain"].to_numpy()
    for label, decision in (
        ("always tied (do nothing)", np.zeros(len(truth), dtype=bool)),
        ("always untied", np.ones(len(truth), dtype=bool)),
        ("follow the discovery argmin", discovery_gain(table) > 0.0),
        ("oracle", truth > 0.0),
    ):
        regret = np.where(decision, np.maximum(-truth, 0.0), np.maximum(truth, 0.0))
        agree = int((np.sign(np.where(decision, 1.0, -1.0)) == np.sign(truth)).sum())
        print(
            f"{label:26s} {'':10s} {agree:4d}/{len(truth):<3d} {regret.mean():9.5f} "
            f"{upper_bound(regret) if regret.any() else 0.0:9.5f}"
        )


def discovery_gain(table: pd.DataFrame) -> np.ndarray:
    """What the single-seed discovery panel said, which is what a practitioner without a model follows."""
    frame = pd.read_csv(CONFIRMATION).drop_duplicates(["rung", "support_id", "policy_role"])
    lookup = frame.set_index(["rung", "support_id", "policy_role"])["discovery_bpb"]
    return np.array(
        [
            lookup[(row.rung, row.support, "selected_tied")] - lookup[(row.rung, row.support, "selected_untied")]
            for row in table.itertuples()
        ]
    )


if __name__ == "__main__":
    main()
