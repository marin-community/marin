# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the archived, unsmoothed CPLM Plus ladder curves."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).with_name("ladder"))
    args = parser.parse_args()
    manifest = json.loads((args.evidence / "runs.json").read_text())
    with gzip.open(args.evidence / "curves.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    colors = {"belay": "#5F6368", "cplm": "#B36B16", "plus": "#1F77B4"}
    labels = {"belay": "Belay", "cplm": "CPLM", "plus": "CPLM Plus"}
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex="col")
    for col, width in enumerate((512, 768, 1024, 1280)):
        selected = [run for run in manifest["runs"] if run["width"] == width]
        axes[0, col].set_title(f"d{width}, B{selected[0]['batch_size']}" + (" (partial)" if width == 1280 else ""))
        for run in selected:
            history = [row for row in rows if row["run_id"] == run["id"]]
            for row_index, metric in enumerate(("train/loss", "eval/loss")):
                points = [row for row in history if row[metric]]
                if not points:
                    continue
                x = np.array([int(row["_step"]) + 1 for row in points]) * run["batch_size"] * 8192 / 1e9
                y = np.array([float(row[metric]) for row in points])
                axes[row_index, col].plot(
                    x,
                    y,
                    color=colors[run["variant"]],
                    linewidth=0.35 if row_index == 0 else 1.5,
                    alpha=0.45 if row_index == 0 else 1.0,
                    marker=None if row_index == 0 else ".",
                    markersize=3,
                    rasterized=row_index == 0,
                )
        axes[1, col].set_xlabel("Training tokens (billions)")
        for ax in axes[:, col]:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Training NLL (raw)")
    axes[1, 0].set_ylabel("Validation NLL")
    handles = [plt.Line2D([], [], color=colors[key], label=labels[key], linewidth=2) for key in colors]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "svg"):
        fig.savefig(args.evidence / f"loss-curves.{extension}", dpi=220, facecolor="white")


if __name__ == "__main__":
    main()
