# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot rank differences between completed 300M/6B runs and the 60M swarm."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "qsplit240_300m_6b_completed_vs_60m.csv"
INPUT_SUMMARY_JSON = ROOT / "qsplit240_300m_6b_completed_vs_60m_summary.json"
OUTPUT_PNG = ROOT / "qsplit240_300m_6b_completed_vs_60m_rank_shift.png"

TOP_MOVER_COUNT = 8
ANNOTATED_RUNS = {
    "run_00018",
    "run_00125",
    "baseline_stratified",
    "baseline_unimax",
    "baseline_proportional",
}


def _display_label(run_name: str) -> str:
    if run_name.startswith("run_"):
        return run_name.replace("run_", "mix_", 1)
    return run_name


def _selected_movers(frame: pd.DataFrame) -> pd.DataFrame:
    upward = frame.nlargest(TOP_MOVER_COUNT, "rank_shift")
    downward = frame.nsmallest(TOP_MOVER_COUNT, "rank_shift")
    movers = (
        pd.concat([upward, downward], ignore_index=True)
        .drop_duplicates(subset=["run_name"])
        .sort_values("rank_shift")
        .reset_index(drop=True)
    )
    return movers


def _annotated_points(frame: pd.DataFrame) -> pd.DataFrame:
    top_300m = frame.nsmallest(5, "rank_300m_6b")
    top_60m = frame.nsmallest(5, "rank_60m_within_completed")
    annotated = (
        pd.concat(
            [
                frame[frame["run_name"].isin(ANNOTATED_RUNS)],
                top_300m,
                top_60m,
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["run_name"])
        .sort_values(["rank_300m_6b", "rank_60m_within_completed"])
        .reset_index(drop=True)
    )
    return annotated


def main() -> None:
    frame = pd.read_csv(INPUT_CSV)
    summary = json.loads(INPUT_SUMMARY_JSON.read_text())

    fig, (ax_scatter, ax_movers) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )

    cmap = plt.get_cmap("RdYlGn_r")
    bpb_norm = Normalize(vmin=frame["bpb_300m_6b"].min(), vmax=frame["bpb_300m_6b"].max())
    scatter = ax_scatter.scatter(
        frame["rank_60m_within_completed"],
        frame["rank_300m_6b"],
        c=frame["bpb_300m_6b"],
        cmap=cmap,
        norm=bpb_norm,
        s=44,
        alpha=0.9,
        edgecolors="none",
    )
    max_rank = int(frame[["rank_60m_within_completed", "rank_300m_6b"]].to_numpy().max())
    ax_scatter.plot([1, max_rank], [1, max_rank], linestyle="--", color="0.45", linewidth=1.2)
    ax_scatter.set_xlim(0, max_rank + 3)
    ax_scatter.set_ylim(max_rank + 3, 0)
    ax_scatter.set_xlabel("60M rank within completed set")
    ax_scatter.set_ylabel("300M/6B rank within completed set")
    ax_scatter.set_title("Completed-mixture rank comparison")
    ax_scatter.grid(alpha=0.18, linewidth=0.6)

    for _, row in _annotated_points(frame).iterrows():
        ax_scatter.annotate(
            _display_label(str(row["run_name"])),
            (row["rank_60m_within_completed"], row["rank_300m_6b"]),
            xytext=(4, -4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.92,
        )

    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.02)
    cbar.set_label("300M/6B BPB")

    movers = _selected_movers(frame)
    mover_colors = np.where(movers["rank_shift"] >= 0, "#1a9850", "#d73027")
    ax_movers.barh(movers["run_name"].map(lambda value: _display_label(str(value))), movers["rank_shift"], color=mover_colors, alpha=0.9)
    ax_movers.axvline(0, color="0.4", linewidth=1.0)
    ax_movers.set_xlabel("Rank shift = 60M rank - 300M/6B rank")
    ax_movers.set_title("Largest movers")
    ax_movers.grid(axis="x", alpha=0.18, linewidth=0.6)

    fig.suptitle(
        (
            "QSplit240 completed 300M/6B mixtures vs 60M swarm\n"
            f"n={summary['n_completed']} shared completed runs, "
            f"Spearman={summary['spearman_rho']:.3f}, Kendall={summary['kendall_tau']:.3f}"
        ),
        fontsize=13,
    )

    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
