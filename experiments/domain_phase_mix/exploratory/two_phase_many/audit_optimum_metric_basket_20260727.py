# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether the optimum-quality basket is four measurements or two wearing four names.

The corrected model ranking puts ``grp`` and ``crs_bounded`` ahead of the band ensemble on optimum
quality, and that conclusion is only as good as the basket it averages. The basket is regret@1,
fold-mean regret@1, lower-tail optimism and low-tail RMSE. Panel regret@1 is zero for the baseline in
three of four fit cells and ties between models in seven of nine heldout cells, so it may contribute
almost no ordering, leaving the two tail statistics to carry the result while appearing to be
corroborated by four.

Three things are measured. How correlated the four metrics are across cells, since two near-duplicate
metrics double-count. How much each metric actually discriminates, as the fraction of cells where it
separates the models at all. And whether the leaderboard survives dropping each metric in turn -- the
only test that says whether the conclusion depends on the basket's composition.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RANKS = SCRIPT_DIR / "reference_outputs" / "model_rank_clustered_bootstrap_20260727" / "model_cell_ranks.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "optimum_metric_basket_20260727"
OPTIMUM_METRICS = ("regretAt1", "foldMeanRegretAt1", "lowerTailOptimism", "lowTailRmse")
TOP_MODELS = ("grp", "crs_bounded", "hpr_band", "hierarchical_phase_bucket_replay")
TIE_TOLERANCE = 1e-12

if str(SCRIPT_DIR.parents[3]) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parents[3]))


def discrimination(frame: pd.DataFrame) -> pd.DataFrame:
    """Per metric: how often it separates models within a cell at all."""
    rows = []
    for metric, group in frame.groupby("metric"):
        cells, separating = 0, 0
        for _key, cell in group.groupby(["swarm", "target", "policy_class", "split"]):
            cells += 1
            if float(cell["value"].max() - cell["value"].min()) > TIE_TOLERANCE:
                separating += 1
        rows.append(
            {
                "metric": metric,
                "cells": cells,
                "separating_cells": separating,
                "separating_fraction": separating / max(cells, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("separating_fraction", ascending=False)


def metric_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlation between metrics over the model-by-cell normalized ranks they assign."""
    wide = frame.pivot_table(
        index=["swarm", "target", "policy_class", "split", "model"],
        columns="metric",
        values="normalized_rank",
    )
    available = [metric for metric in OPTIMUM_METRICS if metric in wide.columns]
    return wide[available].corr(method="spearman")


def leaderboard(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.Series:
    """Mean normalized rank per model over a chosen subset of the basket."""
    subset = frame[frame["metric"].isin(metrics)]
    return subset.groupby("model")["normalized_rank"].mean().sort_values()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=Path, default=DEFAULT_RANKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.ranks)
    optimum = frame[frame["metric"].isin(OPTIMUM_METRICS)]

    print("HOW OFTEN EACH METRIC SEPARATES THE MODELS INSIDE A CELL")
    separation = discrimination(optimum)
    print(separation.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\nRANK CORRELATION BETWEEN THE BASKET'S METRICS")
    correlation = metric_correlation(optimum)
    print(correlation.to_string(float_format=lambda value: f"{value:+.3f}"))
    pairs = [(left, right, float(correlation.loc[left, right])) for left, right in combinations(correlation.columns, 2)]
    redundant = [pair for pair in pairs if abs(pair[2]) > 0.9]
    print(f"\n  pairs correlating above 0.90: {len(redundant)} of {len(pairs)}")
    for left, right, value in sorted(pairs, key=lambda item: -abs(item[2])):
        print(f"    {left:<22} {right:<22} {value:+.3f}")

    print("\nLEADERBOARD UNDER THE FULL BASKET, AND WITH EACH METRIC DROPPED")
    full = leaderboard(optimum, OPTIMUM_METRICS)
    print(f"\n  full basket ({len(OPTIMUM_METRICS)} metrics)")
    for model in TOP_MODELS:
        if model in full.index:
            print(f"    {model:<34}{full[model]:.3f}   rank {list(full.index).index(model) + 1}")

    rows = []
    for dropped in OPTIMUM_METRICS:
        kept = tuple(metric for metric in OPTIMUM_METRICS if metric != dropped)
        board = leaderboard(optimum, kept)
        leader = board.index[0]
        print(f"\n  dropping {dropped}  ->  leader {leader}")
        for model in TOP_MODELS:
            if model in board.index:
                position = list(board.index).index(model) + 1
                print(f"    {model:<34}{board[model]:.3f}   rank {position}")
                rows.append({"dropped": dropped, "model": model, "mean_rank": board[model], "position": position})

    print("\nLEADERBOARD ON EACH METRIC ALONE")
    for metric in OPTIMUM_METRICS:
        board = leaderboard(optimum, (metric,))
        if board.empty:
            continue
        ordered = "  ".join(f"{model} {value:.3f}" for model, value in board.head(3).items())
        print(f"  {metric:<22} {ordered}")

    separation.to_csv(args.output_dir / "metric_discrimination.csv", index=False)
    correlation.to_csv(args.output_dir / "metric_correlation.csv")
    pd.DataFrame(rows).to_csv(args.output_dir / "leave_one_metric_out.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
