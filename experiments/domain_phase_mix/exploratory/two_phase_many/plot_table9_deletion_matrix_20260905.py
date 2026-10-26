# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy", "pandas", "scipy"]
# ///

"""Bucket-deletion effect matrix for the Table-9 macro and tasks on the Qwen3 360M/1.6B panel.

For each of the 39 leave-one-bucket-out runs and each Table-9 task, the cell is the t statistic
of the deletion run against the 11 proportional reference runs (the panel baseline plus the
ten-run noise panel): t = (BPB_deletion - mean_ref) / (sd_ref * sqrt(1 + 1/n)), df = n - 1.
Positive t means deleting the bucket raised BPB (the bucket helps). Multi-subtask tasks are the
unweighted mean of their subtasks computed per run, so their reference SD keeps the correlation
between subtasks instead of assuming independence. Two-sided p-values are Holm-corrected across
the 39 deletions within each task.

Each Common Crawl topic is one row whose cells are split in two, the high-quality bucket on the left and the
low-quality bucket on the right, as in the paper's materialized-epochs figure; every other bucket has a full-width
cell. Two figures are written: the full matrix for the appendix and a version restricted to the buckets with at least
one Holm-significant task. Holm-significant cells carry their t value, rounded to an integer so it fits a half cell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    table9_snr_table_20260905 as snr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_olmix_vs_mariner_weights_compact_20260921 import (  # noqa: E402
    CC_TOPIC_LABELS,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_reliability_20260905"
BASELINE_RUN = "singleavg_fit_000_baseline_proportional"
DELETION_TOKEN = "pctrl_del_"
ALPHA = 0.05
T_CLIP = 6.0
FIGURE_WIDTH = 7.2
ROW_INCHES = 0.2013  # the row height of the earlier one-row-per-bucket layout
QUALITIES = ("high", "low")
MUTED = "#666666"
BOTTOM_MARGIN_INCHES = 1.4
TOP_MARGIN_INCHES = 0.55

PAPER = "#ffffff"
INK = "#111111"
GRID = "#b8b8b8"
FIGURE_SIZE = (11.6, 8.4)
STATIC_DPI = 300
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
OTHER_LABELS = {
    "dolma3_arxiv": "arXiv",
    "dolma3_finemath_3plus": "FineMath 3+",
    "dolma3_stack_edu": "Stack-Edu",
    "dolma3_wikipedia": "Wikipedia",
    "dolmino_common_crawl_hq": "Common Crawl HQ",
    "dolmino_olmocr_pdfs_hq": "olmOCR PDFs HQ",
    "dolmino_stack_edu_fim": "Stack-Edu FIM",
    "dolmino_stem_heavy_crawl": "STEM-heavy crawl",
    "dolmino_synth_code": "Synthetic code",
    "dolmino_synth_instruction": "Synthetic instruction",
    "dolmino_synth_math": "Synthetic math",
    "dolmino_synth_qa": "Synthetic QA",
    "dolmino_synth_thinking": "Synthetic thinking",
}
FAMILY_LABELS = {"cc": "Dolma 3 Common Crawl", "dolma3": "Dolma 3 other", "dolmino": "Dolmino"}
SHORT_FAMILY_LABELS = {"cc": "CC", "dolma3": "Dolma 3", "dolmino": "Dolmino"}
SHORT_LABEL_MAX_ROWS = 4
BRACE_DEPTH_INCHES = 0.10
BRACE_GAP_INCHES = 0.08
LABEL_GAP_INCHES = 0.06
GUTTER_EXTRA_INCHES = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def family(bucket: str) -> str:
    if bucket.startswith("dolma3_cc/"):
        return "cc"
    if bucket.startswith("dolma3_"):
        return "dolma3"
    return "dolmino"


def matrix_rows(buckets: list[str]) -> list[tuple[str, str, tuple[str, ...]]]:
    """(family, label, members) per row: a Common Crawl topic holds its high- and low-quality buckets."""
    rows: list[tuple[str, str, tuple[str, ...]]] = []
    for bucket in buckets:
        if family(bucket) != "cc":
            rows.append((family(bucket), OTHER_LABELS[bucket], (bucket,)))
            continue
        topic = bucket.removeprefix("dolma3_cc/").rpartition("_")[0]
        members = tuple(f"dolma3_cc/{topic}_{quality}" for quality in QUALITIES)
        if not rows or rows[-1][2] != members:
            rows.append(("cc", CC_TOPIC_LABELS[topic], members))
    return rows


def ordered_buckets(buckets: tuple[str, ...]) -> list[str]:
    total = sum(TOP_LEVEL_DOMAIN_TOKEN_COUNTS.values())
    natural = {bucket: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket] / total for bucket in buckets}
    cc = [bucket for bucket in buckets if family(bucket) == "cc"]
    topics: dict[str, float] = {}
    for bucket in cc:
        topic = bucket.removeprefix("dolma3_cc/").rpartition("_")[0]
        topics[topic] = topics.get(topic, 0.0) + natural[bucket]
    ordered = []
    for topic in sorted(topics, key=lambda name: -topics[name]):
        ordered.extend(f"dolma3_cc/{topic}_{quality}" for quality in ("high", "low"))
    for name in ("dolma3", "dolmino"):
        members = [bucket for bucket in buckets if family(bucket) == name]
        ordered.extend(sorted(members, key=lambda bucket: -natural[bucket]))
    if sorted(ordered) != sorted(buckets):
        raise ValueError("Bucket ordering lost or duplicated buckets")
    return ordered


def task_rows() -> list[tuple[str, str, list[str]]]:
    """Return (group, task, component keys) in Olmix Table 9 order, macro first."""
    rows = [("", snr.MACRO_LABEL, [])]
    for group_name, tasks in snr.TABLE9_LAYOUT:
        for task_name, subtasks in tasks:
            rows.append((group_name, task_name, [key for key, _ in subtasks]))
    return rows


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def deletion_cells(values: pd.DataFrame, noise: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    deletion_rows = {}
    for run in values.index:
        if DELETION_TOKEN in run:
            slug = run.split(DELETION_TOKEN, 1)[1]
            matches = [bucket for bucket in buckets if bucket.replace("/", "_") == slug]
            if len(matches) != 1:
                raise ValueError(f"Cannot map deletion run {run} to one bucket")
            deletion_rows[matches[0]] = run
    if len(deletion_rows) != len(buckets):
        raise ValueError(f"Expected {len(buckets)} deletion runs, found {len(deletion_rows)}")
    reference = pd.concat([noise, values.loc[[BASELINE_RUN], noise.columns]], axis=0)
    n_reference = len(reference)
    df = n_reference - 1
    records = []
    for group_name, task_name, keys in task_rows():
        if keys:
            reference_metric = reference[keys].mean(axis=1)
            deletion_metric = values[keys].mean(axis=1)
        else:
            reference_metric = reference.mean(axis=1)
            deletion_metric = values.mean(axis=1)
        mean_ref = float(reference_metric.mean())
        sd_ref = float(reference_metric.std(ddof=1))
        predictive_sd = sd_ref * math.sqrt(1.0 + 1.0 / n_reference)
        for bucket in buckets:
            value = float(deletion_metric[deletion_rows[bucket]])
            delta = value - mean_ref
            t_stat = delta / predictive_sd
            records.append(
                {
                    "group": group_name,
                    "task": task_name,
                    "bucket": bucket,
                    "bucket_family": family(bucket),
                    "deletion_bpb": value,
                    "reference_mean_bpb": mean_ref,
                    "reference_sd_bpb": sd_ref,
                    "reference_n": n_reference,
                    "delta_bpb": delta,
                    "t_statistic": t_stat,
                    "p_hurts": float(student_t.sf(t_stat, df=df)),
                    "p_two_sided": float(2.0 * student_t.sf(abs(t_stat), df=df)),
                }
            )
    cells = pd.DataFrame(records)
    cells["p_holm"] = cells.groupby("task", sort=False)["p_two_sided"].transform(lambda p: holm(p.to_numpy()))
    return cells


# The paper calls these runs single-bucket removals and the statistic T (Appendix C.1.1).
COLORBAR_LABEL = "$T$: removal minus proportional, in units of repeat std"


def row_label(task: str, keys: list[str]) -> str:
    if task == snr.MACRO_LABEL:
        return "OBE Easy mean (51)"
    return f"{task} ({len(keys)})" if len(keys) > 1 else task


def t_text(value: float) -> str:
    return f"{value:.0f}"


def brace_profile(n: int, shoulder: float) -> tuple[np.ndarray, np.ndarray]:
    """Curly-brace depth profile on [0, 1]: shoulders at both ends, cusp in the middle."""
    shoulder = min(max(shoulder, 0.02), 0.25)
    t = np.linspace(0.0, 1.0, n)
    profile = np.empty(n)
    for index, value in enumerate(t):
        u = value if value <= 0.5 else 1.0 - value
        if u < shoulder:
            w = u / shoulder
            profile[index] = 0.5 * w * w * (3 - 2 * w)
        elif u < 0.5 - shoulder:
            profile[index] = 0.5
        else:
            w = (u - (0.5 - shoulder)) / shoulder
            profile[index] = 0.5 + 0.5 * w * w * (3 - 2 * w)
    return t, profile


def draw_row_brace(axis: plt.Axes, x: float, depth: float, start: int, stop: int) -> float:
    """Brace in the left gutter (x, depth in axes fraction; rows in data units). Returns the tip x."""
    t, profile = brace_profile(241, 0.3 / max(stop - start + 0.8, 1.0))
    ys = (start - 0.4) + t * (stop - start + 0.8)
    xs = x - depth * profile
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    axis.plot(xs, ys, transform=transform, color=INK, linewidth=0.9, clip_on=False, solid_capstyle="round")
    return x - depth


def draw_column_brace(axis: plt.Axes, y: float, depth: float, start: int, stop: int) -> float:
    """Brace above a run of columns (data units; the image origin is at the top so up is negative)."""
    t, profile = brace_profile(241, 0.3 / max(stop - start + 0.8, 1.0))
    xs = (start - 0.4) + t * (stop - start + 0.8)
    ys = y - depth * profile
    axis.plot(xs, ys, color=INK, linewidth=0.9, clip_on=False, solid_capstyle="round")
    return y - depth


def build_figure(cells: pd.DataFrame, buckets: list[str]) -> plt.Figure:
    """Transposed matrix: buckets down the side with family braces, tasks across the bottom.

    Columns are two units wide: a Common Crawl row fills a task's two units with its high- and low-quality buckets,
    every other row fills both with the same value, so its cell reads as one.
    """
    tasks = task_rows()
    task_names = [task for _, task, _ in tasks]
    groups = [group for group, _, _ in tasks]
    rows = matrix_rows(buckets)
    families = [row_family for row_family, _, _ in rows]
    t_matrix = cells.pivot(index="bucket", columns="task", values="t_statistic")[task_names]
    p_matrix = cells.pivot(index="bucket", columns="task", values="p_holm")[task_names]
    values = np.empty((len(rows), 2 * len(task_names)))
    marks: list[tuple[float, int, float]] = []
    for row_index, (_family, _label, members) in enumerate(rows):
        for task_index, task in enumerate(task_names):
            for half in range(2):
                values[row_index, 2 * task_index + half] = t_matrix.loc[members[half % len(members)], task]
            for half, member in enumerate(members):
                if p_matrix.loc[member, task] < ALPHA:
                    x = 2 * task_index + (half if len(members) == 2 else 0.5)
                    marks.append((x, row_index, float(t_matrix.loc[member, task])))
    values = np.clip(values, -T_CLIP, T_CLIP)
    width = FIGURE_WIDTH
    height = TOP_MARGIN_INCHES + BOTTOM_MARGIN_INCHES + len(rows) * ROW_INCHES
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(width, height))
        figure.subplots_adjust(
            left=0.30, right=0.90, bottom=BOTTOM_MARGIN_INCHES / height, top=1.0 - TOP_MARGIN_INCHES / height
        )
        image = axis.imshow(values, cmap="RdBu_r", norm=Normalize(-T_CLIP, T_CLIP), aspect="auto")
        axis.set_autoscale_on(False)
        for x, row_index, value in marks:
            axis.text(
                x,
                row_index,
                t_text(value),
                ha="center",
                va="center",
                fontsize=5.2,  # a two-character label such as -5 fits a half cell
                fontweight="bold",
                color="white" if abs(value) > 4.0 else INK,
                zorder=5,
            )
        boundaries = {1} | {index for index in range(2, len(tasks)) if groups[index] != groups[index - 1]}
        for index in sorted(boundaries):
            axis.axvline(2 * index - 0.5, color=INK, linewidth=0.9, zorder=4)
        cc_rows = [index for index, name in enumerate(families) if name == "cc"]
        if cc_rows:
            # White rules between tasks pair the high- and low-quality halves of each Common Crawl cell.
            for index in range(1, len(tasks)):
                if index not in boundaries:
                    axis.vlines(
                        2 * index - 0.5, min(cc_rows) - 0.5, max(cc_rows) + 0.5, color=PAPER, linewidth=0.9, zorder=3
                    )
            for half, quality in enumerate(QUALITIES):
                axis.text(
                    half, -0.6, quality[0].upper(), ha="center", va="bottom", fontsize=5.6, color=MUTED, clip_on=False
                )
        for index in range(1, len(rows)):
            if families[index] != families[index - 1]:
                axis.axhline(index - 0.5, color=INK, linewidth=0.9, zorder=4)
        axis.set_xticks([2 * index + 0.5 for index in range(len(task_names))])
        axis.set_xticklabels([row_label(task, keys) for _, task, keys in tasks], rotation=90, fontsize=6.6, color=INK)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([label for _, label, _ in rows], fontsize=6.6, color=INK)
        axis.tick_params(axis="both", length=0, pad=3)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color(INK)
            spine.set_linewidth(0.8)
        # Measure the widest row label, then set the left margin and gutter positions from it.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        label_inches = max(label.get_window_extent(renderer).width for label in axis.get_yticklabels()) / figure.dpi
        pad_inches = 3.0 / 72.0
        left_inches = label_inches + pad_inches + GUTTER_EXTRA_INCHES + 0.15
        figure.subplots_adjust(left=left_inches / width)
        axes_inches = width * (0.90 - left_inches / width)
        brace_x = -(label_inches + pad_inches + BRACE_GAP_INCHES) / axes_inches
        depth = BRACE_DEPTH_INCHES / axes_inches
        starts: dict[str, int] = {}
        for index, name in enumerate(families):
            starts.setdefault(name, index)
        for name, start in starts.items():
            stop = max(index for index, other in enumerate(families) if other == name)
            tip = draw_row_brace(axis, brace_x, depth, start, stop)
            labels = SHORT_FAMILY_LABELS if stop - start + 1 <= SHORT_LABEL_MAX_ROWS else FAMILY_LABELS
            axis.text(
                tip - LABEL_GAP_INCHES / axes_inches,
                (start + stop) / 2,
                labels[name],
                transform=blended_transform_factory(axis.transAxes, axis.transData),
                rotation=90,
                ha="right",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color=INK,
                clip_on=False,
            )
        column_depth = BRACE_DEPTH_INCHES / ROW_INCHES
        group_starts: dict[str, int] = {}
        for index, name in enumerate(groups):
            if name:
                group_starts.setdefault(name, index)
        for name, start in group_starts.items():
            stop = max(index for index, other in enumerate(groups) if other == name)
            tip = draw_column_brace(axis, -0.55 - BRACE_GAP_INCHES / ROW_INCHES, column_depth, 2 * start, 2 * stop + 1)
            axis.text(
                start + stop + 0.5,
                tip - LABEL_GAP_INCHES / ROW_INCHES,
                name,
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=INK,
                clip_on=False,
            )
        axis.set_xlim(-0.5, 2 * len(task_names) - 0.5)
        axis.set_ylim(len(rows) - 0.5, -0.5)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03, orientation="vertical")
        colorbar.set_label(COLORBAR_LABEL, fontsize=7.0, color=INK)
        colorbar.set_ticks([-T_CLIP, -4, -2, 0, 2, 4, T_CLIP])
        colorbar.set_ticklabels([f"<= -{T_CLIP:.0f}", "-4", "-2", "0", "2", "4", f">= {T_CLIP:.0f}"])
        colorbar.ax.tick_params(labelsize=6.6, colors=INK, width=0.6, length=2.5)
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor(GRID)
        return figure


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = benchmark.load_panel(snr.PANEL)
    values, noise = snr.load_matrices()
    buckets = ordered_buckets(panel.buckets)
    cells = deletion_cells(values, noise, buckets)
    cells.to_csv(args.output_dir / "deletion_matrix_cells.csv", index=False)
    significant = cells.loc[cells["p_holm"] < ALPHA]
    detectable = [bucket for bucket in buckets if bucket in set(significant["bucket"])]
    for stem, subset in (
        ("deletion_matrix_full", buckets),
        ("deletion_matrix_main", detectable),
    ):
        figure = build_figure(cells, subset)
        figure.savefig(args.output_dir / f"{stem}.png", dpi=STATIC_DPI)
        figure.savefig(args.output_dir / f"{stem}.pdf")
        plt.close(figure)
    summary = {
        "reference_runs": int(cells["reference_n"].iloc[0]),
        "cells": len(cells),
        "holm_significant_cells": len(significant),
        "holm_significant_hurts": int((significant["t_statistic"] > 0).sum()),
        "holm_significant_helps": int((significant["t_statistic"] < 0).sum()),
        "detectable_buckets": detectable,
        "significant_per_task": significant.groupby("task").size().sort_values(ascending=False).to_dict(),
        "significant_per_bucket": significant.groupby("bucket").size().sort_values(ascending=False).to_dict(),
        "macro_row": (
            cells.loc[cells["task"].eq(snr.MACRO_LABEL)]
            .sort_values("t_statistic", ascending=False)[["bucket", "delta_bpb", "t_statistic", "p_holm"]]
            .head(8)
            .to_dict(orient="records")
        ),
    }
    (args.output_dir / "deletion_matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "macro_row"}, indent=1)[:2500])
    print(pd.DataFrame(summary["macro_row"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
