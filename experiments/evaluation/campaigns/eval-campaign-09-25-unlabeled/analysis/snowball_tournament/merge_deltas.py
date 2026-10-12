#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot campaign merge scores minus their starting checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from pairwise import tracker_scores

MODEL_BASES = {
    "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12": (
        "open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10",
        "https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12/blob/271c4b3c6b38e8f645c88b8938a6ccf2b2195f2a/provenance/campaign3/CAMPAIGN3.md",
        "DARE-Linear-02",
        "dare-linear-02",
    ),
    "open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10": (
        "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02",
        "https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10/blob/153f5369f69a0c0a7203779099ec94e98c66748f/README.md",
        "DARE-TIES-02",
        "dare-ties-02",
    ),
    "open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09": (
        "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92",
        "https://huggingface.co/open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09/blob/462ad9af071055da39ef4bfea1d4ba2c9429a223/README.md",
        "Shared100",
        "shared100",
    ),
}
SCORE = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+)\)$")
COLORS = LinearSegmentedColormap.from_list("loss-tie-gain", ("#ad4040", "#f4f3ef", "#238550"))
COLUMNS = 7


def source_links(path: Path, models: set[str]) -> dict[tuple[str, str], str]:
    """Read the durable source link for each scored tracker cell in the comparison."""
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    links = {}
    for line in lines[3:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header) or cells[0] not in models:
            continue
        for benchmark, cell in zip(header[1:], cells[1:], strict=True):
            if match := SCORE.match(cell):
                links[(cells[0], benchmark)] = match.group(2)
    return links


def comparison_rows(tracker: Path) -> tuple[list[str], list[dict[str, object]]]:
    benchmarks, _, scores = tracker_scores(tracker)
    models = set(MODEL_BASES) | {base for base, _, _, _ in MODEL_BASES.values()}
    links = source_links(tracker, models)
    rows = []
    for merge, (base, card, short_name, _) in MODEL_BASES.items():
        if set(scores[merge]) != set(benchmarks) or set(scores[base]) != set(benchmarks):
            raise ValueError(f"merge and base need complete 26-benchmark rows: {merge}, {base}")
        for benchmark in benchmarks:
            merge_score = scores[merge][benchmark]
            base_score = scores[base][benchmark]
            rows.append(
                {
                    "merge": merge,
                    "merge_label": short_name,
                    "base": base,
                    "benchmark": benchmark,
                    "merge_score": merge_score,
                    "base_score": base_score,
                    "delta_pp": 100 * (merge_score - base_score),
                    "merge_result": links[(merge, benchmark)],
                    "base_result": links[(base, benchmark)],
                    "merge_card": card,
                }
            )
    return benchmarks, rows


def plot_model(rows: list[dict[str, object]], limit: float, title: str, output: Path) -> None:
    count = len(rows)
    nrows = math.ceil(count / COLUMNS)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    for index, row in enumerate(rows):
        column, grid_row = index % COLUMNS, index // COLUMNS
        delta = float(row["delta_pp"])
        ax.add_patch(
            Rectangle((column, grid_row), 1, 1, facecolor=COLORS(norm(delta)), edgecolor="#d6d8d2", linewidth=1)
        )
        ink = "white" if abs(delta) > limit * 0.62 else "#1f2824"
        benchmark = textwrap.fill(str(row["benchmark"]), width=15, break_on_hyphens=True)
        ax.text(column + 0.5, grid_row + 0.39, benchmark, ha="center", va="center", color=ink, fontsize=9)
        ax.text(column + 0.5, grid_row + 0.76, f"{delta:+.1f} pp", ha="center", va="center", color=ink, fontsize=11)
    ax.set_xlim(0, COLUMNS)
    ax.set_ylim(nrows, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, pad=18)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=COLORS), ax=ax, orientation="horizontal", fraction=0.04, pad=0.035
    )
    colorbar.set_label("Merge minus base (percentage points)")
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks, rows = comparison_rows(args.tracker)
    limit = max(abs(float(row["delta_pp"])) for row in rows)
    with (args.output_dir / "merge_deltas.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for merge, (base, _, short_name, stem) in MODEL_BASES.items():
        model_rows = [row for row in rows if row["merge"] == merge]
        plot_model(model_rows, limit, f"{short_name} - {base.rsplit('/', 1)[-1]}", args.output_dir / stem)
    metadata = {
        "tracker_sha256": hashlib.sha256(args.tracker.read_bytes()).hexdigest(),
        "score_precision": "Deltas use the three-decimal tracker point estimates, not unrounded trial aggregates.",
        "unit": "percentage points",
        "benchmark_count": len(benchmarks),
        "global_color_limit_pp": limit,
        "models": {
            merge: {"base": base, "model_card": card, "plot_stem": stem}
            for merge, (base, card, _, stem) in MODEL_BASES.items()
        },
        "excluded": {
            "open-athena/Grug-67B-A2B-BFCL-Recovery": "DPO continuation of Antidoom Step12, not a weight merge."
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"plotted {len(rows)} linked cells for {len(MODEL_BASES)} merges")


if __name__ == "__main__":
    main()
