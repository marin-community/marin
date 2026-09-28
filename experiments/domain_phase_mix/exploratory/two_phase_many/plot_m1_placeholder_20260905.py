# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-width rectangular placeholder for Figure M1 (the pipeline schematic)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "scaling_curves_placeholder_20260905"
STAGES = (
    "Corpus\npartition",
    "Swarm\ndesign",
    "Train and\nevaluate",
    "Reliability\nscreen",
    "Surrogate\nfit",
    "Constrained\noptimization",
    "Fresh\nvalidation",
)
PLOT_STYLE = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "text.usetex": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    plt.rcParams.update(PLOT_STYLE)
    figure, axis = plt.subplots(figsize=(7.0, 1.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(
        plt.Rectangle(
            (0.005, 0.08), 0.99, 0.84, facecolor="#f2f2f2", edgecolor="#9a9a9a", linestyle=(0, (4, 3)), linewidth=1.0
        )
    )
    count = len(STAGES)
    for index, stage in enumerate(STAGES):
        x = (index + 0.5) / count
        axis.add_patch(
            plt.Rectangle((x - 0.058, 0.30), 0.116, 0.40, facecolor="white", edgecolor="#555555", linewidth=0.8)
        )
        axis.text(x, 0.5, stage, ha="center", va="center", fontsize=6.5, color="#111111")
        if index < count - 1:
            axis.annotate(
                "",
                xy=(x + 0.084, 0.5),
                xytext=(x + 0.058, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            )
    axis.text(
        0.5,
        0.86,
        "Figure M1 placeholder: pipeline schematic (full page width)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#B00020",
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.16,
        "Cross-reference B1 and B4 for notation; final artwork to replace this box.",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#555555",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"m1_pipeline_placeholder.{extension}", dpi=300, bbox_inches="tight")
        if args.drive_dir is not None:
            shutil.copyfile(
                args.output_dir / f"m1_pipeline_placeholder.{extension}",
                args.drive_dir / f"m1_pipeline_placeholder.{extension}",
            )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
