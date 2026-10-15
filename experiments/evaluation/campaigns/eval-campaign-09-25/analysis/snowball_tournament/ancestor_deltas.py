#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare campaign scores with every published direct weight parent."""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from merge_deltas import COLORS, COLUMNS, source_links
from pairwise import tracker_scores

PREFIX = "open-athena/"
STEP92 = PREFIX + "Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"
STEP12 = PREFIX + "Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
STEP20 = PREFIX + "Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04"
RECOVERY = PREFIX + "Grug-67B-A2B-BFCL-Recovery"
TIES = PREFIX + "Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10"
GIST = "https://gist.githubusercontent.com/penfever/f93b39b8cc6dd12674e5e083f80c63d4/raw/cb839c12a1920f26439d59e92664bd34c498c9cb/README.md"


@dataclasses.dataclass(frozen=True)
class Parent:
    model: str
    label: str


@dataclasses.dataclass(frozen=True)
class Lineage:
    model: str
    label: str
    stem: str
    parents: tuple[Parent, ...]
    evidence: str


LINEAGES = (
    Lineage(
        PREFIX + "Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09",
        "Shared100",
        "shared100",
        (Parent(STEP92, "Step92"), Parent(STEP12, "Antidoom Step12"), Parent(STEP20, "Reference-KL Step20")),
        "https://huggingface.co/open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09/blob/462ad9af071055da39ef4bfea1d4ba2c9429a223/README.md",
    ),
    Lineage(
        RECOVERY,
        "BFCL-Recovery",
        "bfcl-recovery",
        (Parent(STEP12, "Antidoom Step12"),),
        "https://huggingface.co/open-athena/Grug-67B-A2B-BFCL-Recovery/blob/850a32627724d274004241dd10f6cb46fd3f6f83/README.md",
    ),
    Lineage(
        PREFIX + "Grug-67B-A2B-BFCL-Protocol-FTPO",
        "BFCL-Protocol-FTPO",
        "bfcl-protocol-ftpo",
        (Parent(RECOVERY, "BFCL-Recovery"),),
        "https://huggingface.co/open-athena/Grug-67B-A2B-BFCL-Protocol-FTPO/blob/d761ea0a94500c2aa24aa4cd71b5a149d962e5c8/README.md",
    ),
    Lineage(
        TIES,
        "Campaign 2 DARE-TIES-02",
        "dare-ties-02",
        (Parent(STEP12, "Antidoom Step12"), Parent(STEP92, "Step92"), Parent(STEP20, "Reference-KL Step20")),
        "https://huggingface.co/open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10/blob/153f5369f69a0c0a7203779099ec94e98c66748f/README.md",
    ),
    Lineage(
        PREFIX + "Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12",
        "Campaign 3 DARE-Linear-02",
        "dare-linear-02",
        (Parent(TIES, "DARE-TIES-02"), Parent("laion/snowball-67b-a2b-relay-sft-allkimi-step1203", "allkimi Step1203")),
        GIST,
    ),
)


@dataclasses.dataclass(frozen=True)
class Comparison:
    model: str
    parent: str
    benchmark: str
    model_score: float | None
    parent_score: float | None
    delta_pp: float | None
    model_result: str | None
    parent_result: str | None


def comparisons(tracker: Path, lineages: tuple[Lineage, ...]) -> list[Comparison]:
    benchmarks, _, scores = tracker_scores(tracker)
    models = {lineage.model for lineage in lineages} | {
        parent.model for lineage in lineages for parent in lineage.parents
    }
    missing = models - scores.keys()
    if missing:
        raise ValueError(f"models absent from tracker: {sorted(missing)}")
    links = source_links(tracker, models)
    rows = []
    for lineage in lineages:
        for parent in lineage.parents:
            for benchmark in benchmarks:
                model_score = scores[lineage.model].get(benchmark)
                parent_score = scores[parent.model].get(benchmark)
                delta = (
                    100 * (model_score - parent_score) if model_score is not None and parent_score is not None else None
                )
                rows.append(
                    Comparison(
                        lineage.model,
                        parent.model,
                        benchmark,
                        model_score,
                        parent_score,
                        delta,
                        links.get((lineage.model, benchmark)),
                        links.get((parent.model, benchmark)),
                    )
                )
    return rows


def draw_parent(ax, rows: list[Comparison], title: str, norm: TwoSlopeNorm) -> None:
    for index, row in enumerate(rows):
        x, y = index % COLUMNS, index // COLUMNS
        color = "#e5e7eb" if row.delta_pp is None else COLORS(norm(row.delta_pp))
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#c8ccc7", linewidth=1))
        ink = "white" if row.delta_pp is not None and abs(row.delta_pp) > norm.vmax * 0.62 else "#1f2824"
        ax.text(x + 0.5, y + 0.24, textwrap.fill(row.benchmark, 17), ha="center", va="center", color=ink, fontsize=9)
        model = "pending" if row.model_score is None else f"{100 * row.model_score:.1f}"
        parent = "pending" if row.parent_score is None else f"{100 * row.parent_score:.1f}"
        ax.text(x + 0.5, y + 0.59, f"{model} / {parent}", ha="center", va="center", color=ink, fontsize=10)
        delta = "pending" if row.delta_pp is None else f"{row.delta_pp:+.1f} pp"
        ax.text(x + 0.5, y + 0.82, delta, ha="center", va="center", color=ink, fontsize=11)
    ax.set_xlim(0, COLUMNS)
    ax.set_ylim(math.ceil(len(rows) / COLUMNS), 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, pad=12)


def plot_lineage(lineage: Lineage, rows: list[Comparison], limit: float, output: Path) -> None:
    fig, axes = plt.subplots(len(lineage.parents), 1, figsize=(14, 7 * len(lineage.parents)), squeeze=False)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig.suptitle(
        f"{lineage.label} — published-parent comparisons\nScores (%): model / parent; delta in percentage points",
        fontsize=15,
    )
    for ax, parent in zip(axes[:, 0], lineage.parents, strict=True):
        draw_parent(ax, [r for r in rows if r.parent == parent.model], f"Compared with {parent.label}", norm)
    fig.subplots_adjust(top=1 - 0.20 / len(lineage.parents), bottom=0.09 / len(lineage.parents), hspace=0.15)
    cax = fig.add_axes((0.25, 0.018 / len(lineage.parents), 0.5, 0.018 / len(lineage.parents)))
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=COLORS), cax=cax, orientation="horizontal")
    colorbar.set_label("Model minus parent (pp); red = lower, green = higher; grey = pending")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.output_dir / "TRACKER.snapshot.md"
    snapshot.write_bytes(args.tracker.read_bytes())
    rows = comparisons(snapshot, LINEAGES)
    limit = max(abs(row.delta_pp) for row in rows if row.delta_pp is not None) or 1
    with (args.output_dir / "ancestor_deltas.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=[field.name for field in dataclasses.fields(Comparison)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)
    for lineage in LINEAGES:
        plot_lineage(lineage, [row for row in rows if row.model == lineage.model], limit, args.output_dir / lineage.stem)
    metadata = {
        "tracker_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lineages": [dataclasses.asdict(lineage) for lineage in LINEAGES],
        "comparison_cells": len(rows),
        "scored_cells": sum(row.delta_pp is not None for row in rows),
        "color_limit_pp": limit,
        "precision": "Displayed campaign scores; three decimals before percentage-point differences.",
        "scope": "Published direct weight parents and initializers; no development scores or estimated missing scores.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(LINEAGES)} figures; {metadata['scored_cells']}/{len(rows)} comparisons scored")


if __name__ == "__main__":
    main()
