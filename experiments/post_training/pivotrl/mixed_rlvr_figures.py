# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Summary figures for the mixed-RLVR pivot arms, drawn from the paired-deltas CSV that ``mixed_rlvr_analysis plot`` writes.

    python -m experiments.post_training.pivotrl.mixed_rlvr_figures scatter DELTAS OUT_PREFIX
    python -m experiments.post_training.pivotrl.mixed_rlvr_figures grid DELTAS OUT_PREFIX

``scatter`` draws every checkpoint as its RL step number in its arm's colour, with the SWE-bench Verified change
against the SFT on the x-axis and one panel per reasoning benchmark on the y-axis (``--y-benchmarks``). Two marks with
the same colour and number are the two training seeds. Two marks of one arm that would print on top of each other are
offset vertically by a third of a point. ``grid`` draws every checkpoint against every benchmark: the cell text is the
change in points and the cell colour is that change in standard errors, so benchmarks of very different size colour
comparably. A benchmark's standard error is the median half-width of its 80% paired bootstrap intervals divided by
1.2816; each scatter panel shows it as a small cross. Both actions write OUT_PREFIX.png, .pdf and .svg.

``figures/`` holds the figures for the two-seed series of #10214, rendered from ``figures/mixed_rlvr_paired_deltas.json``:
the rows of the CSV that ``mixed_rlvr_analysis plot --shared-items`` wrote for the series a, b, c, a-seed43, b-seed43
and c-seed43 on 2026-10-22, saved as JSON because the repository ignores CSV files.
"""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ARMS = {
    "a": ("A  RLVR mix", "#8C8C8C"),
    "b": ("B  A mix, 20% selected pivots", "#D55E00"),
    "c": ("C  A mix, 20% random turns", "#0072B2"),
}
BENCHMARKS = {
    "swebench-verified": "SWE-bench Verified",
    "ot-tblite-recovery": "OT-TBLite",
    "gpqa-diamond": "GPQA-Diamond",
    "mmlu-pro": "MMLU-Pro",
    "math500": "Math500",
    "aime24": "AIME24",
}
X_BENCHMARK = "swebench-verified"
DEFAULT_Y_BENCHMARKS = "math500,mmlu-pro,gpqa-diamond"
STEPS = (4, 8, 12)
SEED_SUFFIXES = {"": "s42", "-seed43": "s43"}  # arm label suffix -> seed tag
Z80 = 1.2816  # normal quantile behind an 80% interval
NUDGE = 0.3  # points; vertical offset between two marks of one arm that would print on top of each other
GRID_CLIP = 4.0  # standard errors at which the grid colour saturates
TITLE = "Checkpoints of three RL arms from the Oct 5 SFT"
SUBTITLE = "mark = RL step (4, 8 or 12) · two training seeds per arm"
RC = {
    "font.family": "Helvetica Neue",
    "font.size": 9,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
}


@dataclass(frozen=True)
class Deltas:
    """Paired differences against the SFT in points, keyed (benchmark, arm label, step), and each benchmark's s.e."""

    points: dict[tuple[str, str, int], float]
    errors: dict[str, float]


@dataclass
class Mark:
    x: float
    y: float
    step: int
    arm: str  # base arm letter: a, b or c


def load_deltas(path: Path) -> Deltas:
    """Read the rows ``mixed_rlvr_analysis plot`` writes, from its CSV or from the same rows saved as a JSON list."""
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
    points = {(row["benchmark"], row["arm"], int(row["step"])): float(row["delta_points"]) for row in rows}
    half_widths: dict[str, list[float]] = {}
    for row in rows:
        half_widths.setdefault(row["benchmark"], []).append((float(row["ci80_high"]) - float(row["ci80_low"])) / 2)
    errors = {bench: statistics.median(widths) / Z80 for bench, widths in half_widths.items()}
    return Deltas(points, errors)


def arm_base(arm: str) -> str:
    return arm.split("-")[0]


def seed_tag(arm: str) -> str:
    for suffix, tag in SEED_SUFFIXES.items():
        if suffix and arm.endswith(suffix):
            return tag
    return SEED_SUFFIXES[""]


def scatter_marks(deltas: Deltas, y_benchmark: str) -> list[Mark]:
    """One mark per checkpoint scored on both axes, with coincident marks of one arm nudged apart."""
    marks = [
        Mark(deltas.points[(X_BENCHMARK, arm, step)], y, step, arm_base(arm))
        for (bench, arm, step), y in sorted(deltas.points.items())
        if bench == y_benchmark and (X_BENCHMARK, arm, step) in deltas.points
    ]
    for i, first in enumerate(marks):
        for second in marks[i + 1 :]:
            if first.arm == second.arm and abs(first.x - second.x) < 0.7 and abs(first.y - second.y) < 0.5:
                first.y += NUDGE
                second.y -= NUDGE
    return marks


def missing_note(deltas: Deltas, y_benchmark: str) -> str:
    """Name the checkpoints that have a SWE-bench score but no score on ``y_benchmark``."""
    missing = [
        f"{arm_base(arm).upper()}{step}" + ("" if seed_tag(arm) == SEED_SUFFIXES[""] else f" {seed_tag(arm)}")
        for bench, arm, step in sorted(deltas.points)
        if bench == X_BENCHMARK and (y_benchmark, arm, step) not in deltas.points
    ]
    return "not evaluated: " + ", ".join(missing) if missing else ""


def draw_scatter_panel(
    ax: Axes, marks: list[Mark], y_label: str, se_x: float, se_y: float, x_limits: tuple[float, float], note: str
) -> None:
    ax.axhline(0, color="#BBBBBB", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color="#BBBBBB", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for mark in marks:
        color = ARMS[mark.arm][1]
        ax.text(mark.x, mark.y, str(mark.step), color=color, fontsize=11, fontweight="bold", ha="center", va="center")
    ys = [mark.y for mark in marks]
    span = max(ys) - min(ys)
    ax.set_xlim(*x_limits)
    ax.set_ylim(min(ys) - max(1.6 * se_y + 0.6, 0.18 * span), max(ys) + max(0.12 * span, 1.0))
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylabel(y_label)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    cross_x = x_limits[0] + se_x + 0.4
    cross_y = ax.get_ylim()[0] + 1.3 * se_y + 0.2
    ax.errorbar(cross_x, cross_y, xerr=se_x, yerr=se_y, color="#888888", lw=0.8, capsize=2, capthick=0.8)
    ax.text(cross_x + se_x + 0.3, cross_y, "±1 s.e.", fontsize=7.5, color="#888888", ha="left", va="center")
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7.5, color="#888888", ha="left", va="top")


def save(fig: Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(f"{out_prefix}.{extension}", dpi=200 if extension == "png" else None)
    plt.close(fig)


def scatter(deltas: Deltas, out_prefix: Path, y_benchmarks: list[str]) -> None:
    plt.rcParams.update(RC)
    panels = len(y_benchmarks)
    fig, axes = plt.subplots(1, panels, figsize=(5.4 if panels == 1 else 3.6 * panels + 0.6, 4.3), squeeze=False)
    xs = [delta for (bench, _, _), delta in deltas.points.items() if bench == X_BENCHMARK]
    x_limits = (min(xs) - 2.5, max(xs) + 2.5)
    x_label = f"{BENCHMARKS[X_BENCHMARK]}, Δ accuracy vs SFT (pts)"
    for ax, y_benchmark in zip(axes[0], y_benchmarks, strict=True):
        y_label = f"{BENCHMARKS[y_benchmark]}, Δ accuracy vs SFT (pts)"
        errors = (deltas.errors[X_BENCHMARK], deltas.errors[y_benchmark])
        draw_scatter_panel(ax, scatter_marks(deltas, y_benchmark), y_label, *errors, x_limits, missing_note(deltas, y_benchmark))
        ax.set_xlabel(x_label)
    first, last = axes[0][0], axes[0][-1]
    first.set_title(TITLE, loc="left", fontsize=10, fontweight="bold", color="#111111", pad=16)
    first.text(0, 1.012, SUBTITLE, transform=first.transAxes, fontsize=8, color="#666666", va="bottom")
    for i, (label, color) in enumerate(ARMS.values()):
        y = 0.05 + 0.058 * (len(ARMS) - 1 - i)
        last.text(0.98, y, label, transform=last.transAxes, fontsize=8.5, color=color, ha="right", va="bottom", fontweight="bold")
    fig.tight_layout()
    save(fig, out_prefix)


def grid(deltas: Deltas, out_prefix: Path) -> None:
    rows: list[tuple[str, int]] = []  # (arm label, step)
    for base in ARMS:
        for step in STEPS:
            for suffix in SEED_SUFFIXES:
                arm = base + suffix
                if any((bench, arm, step) in deltas.points for bench in BENCHMARKS):
                    rows.append((arm, step))
    benches = list(BENCHMARKS)
    cells = np.full((len(rows), len(benches)), np.nan)
    for i, (arm, step) in enumerate(rows):
        for j, bench in enumerate(benches):
            if (bench, arm, step) in deltas.points:
                cells[i, j] = deltas.points[(bench, arm, step)] / deltas.errors[bench]

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 1.6))
    cmap = plt.get_cmap("PRGn").copy()
    cmap.set_bad("white")
    ax.imshow(np.clip(cells, -GRID_CLIP, GRID_CLIP), cmap=cmap, vmin=-GRID_CLIP, vmax=GRID_CLIP, aspect="auto")
    for i, (arm, step) in enumerate(rows):
        for j, bench in enumerate(benches):
            if (bench, arm, step) not in deltas.points:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#BBBBBB")
                continue
            strong = abs(cells[i, j]) > 0.65 * GRID_CLIP
            value = deltas.points[(bench, arm, step)]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="white" if strong else "#222222")
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([BENCHMARKS[bench].replace(" ", "\n") for bench in benches], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{arm_base(arm).upper()}  step {step:>2}  {seed_tag(arm)}" for arm, step in rows], fontsize=8.5, family="Menlo")
    for tick, (arm, _) in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_color(ARMS[arm_base(arm)][1])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(1, len(rows)):
        if arm_base(rows[i][0]) != arm_base(rows[i - 1][0]):
            ax.axhline(i - 0.5, color="white", lw=3)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    left = ax.get_position().x0
    fig.text(left, 0.985, "Δ accuracy vs SFT (pts), every checkpoint", fontsize=10, fontweight="bold", color="#111111", va="top")
    note = f"colour = Δ in standard errors, clipped at ±{GRID_CLIP:g} · — = not evaluated"
    fig.text(left, 0.962, note, fontsize=8, color="#666666", va="top")
    save(fig, out_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("scatter", "grid"))
    parser.add_argument("deltas", type=Path, help="the CSV from mixed_rlvr_analysis plot, or its rows as JSON")
    parser.add_argument("out_prefix", type=Path, help="output path without extension")
    parser.add_argument("--y-benchmarks", default=DEFAULT_Y_BENCHMARKS, help="scatter: comma-separated y-axis benchmarks")
    args = parser.parse_args()
    deltas = load_deltas(args.deltas)
    if args.action == "scatter":
        scatter(deltas, args.out_prefix, args.y_benchmarks.split(","))
    else:
        grid(deltas, args.out_prefix)


if __name__ == "__main__":
    main()
