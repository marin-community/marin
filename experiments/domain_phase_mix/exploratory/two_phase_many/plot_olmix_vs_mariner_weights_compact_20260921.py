# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2.0", "pandas>=2.2"]
# ///
"""Compact bucket weights of matched Olmix (cap 4, KL 0) beside MARINER (no cap, no KL) at Qwen3 360M/1.6B.

Three top-aligned columns share one row height: the Common Crawl high-quality cells in the first column and the
low-quality cells in the second, both on their own weight scale, and the other Dolma 3 sources with Dolmino in the
third column on a wider scale. Bars give
mixture weights, labels the materialized epochs, ticks the proportional weight. The Olmix mixture is the Qwen-fitted
policy of ``delphi_matched_olmix_3e18_20260908`` (per-task log-linear laws on the frozen 280-run swarm, exact capped
proposer, KL 0); MARINER's is the frozen procedure's unconstrained proposal.

usage: uv run plot_olmix_vs_mariner_weights_compact_20260921.py [--target table9|uncheatable] [--drive-dir DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base_launch  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_wspu_worsened_vs_uncheatable_mixtures_20260905 as layout,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
OLMIX_TABLE = REFERENCE / "delphi_matched_olmix_3e18_20260908" / "candidate_weights.csv"
MARINER_TABLE = (
    REFERENCE
    / "delphi_corrected_screen_20260908"
    / "materialized_flat15_nocap"
    / "runtime_materialization"
    / "candidate_weights.csv"
)
OLMIX_UNCAPPED_TABLE = REFERENCE / "olmix_uncapped_20260921" / "candidate_weights.csv"
OUTPUT_DIR = REFERENCE / "olmix_vs_mariner_weights_compact_20260921"
TARGETS = {  # target: (Olmix KL-0 cap-4 candidate, MARINER candidate, objective label)
    "table9": ("olmixq_t9_kl0_cap04", "lwspu_t9_snc_cap08", "OlmoBaseEval Easy"),
    "uncheatable": ("olmixq_u_kl0_cap04", "lwspu_u_snc_cap06", "Uncheatable"),
}
# Olmix policy variants: table, candidate id per target, legend label
OLMIX_POLICIES = {
    "cap4": (
        OLMIX_TABLE,
        {"table9": "olmixq_t9_kl0_cap04", "uncheatable": "olmixq_u_kl0_cap04"},
        "Olmix optimum, cap 4, KL 0",
    ),
    "nocap": (
        OLMIX_UNCAPPED_TABLE,
        {"table9": "olmixq_t9_kl0_nocap", "uncheatable": "olmixq_u_kl0_nocap"},
        "Olmix optimum, no cap, no KL",
    ),
}
OLMIX_COLOR = "#CC79A7"
MARINER_COLOR = "#469C76"
SERIES = (
    ("olmix", OLMIX_COLOR, "Olmix optimum, cap 4, KL 0"),
    ("mariner", MARINER_COLOR, "MARINER optimum, no cap, no KL"),
)
BAR_HEIGHT = 0.38
OFFSETS = (0.21, -0.21)
EPOCH_TOLERANCE = 1e-6
CC_TOPIC_LABELS = {
    "finance_and_business": "Finance & business",
    "health": "Health",
    "entertainment": "Entertainment",
    "education_and_jobs": "Education & jobs",
    "science_math_and_technology": "Science, math & tech.",
    "literature": "Literature",
    "games": "Games",
    "food_and_dining": "Food & dining",
    "crime_and_law": "Crime & law",
    "electronics_and_hardware": "Electronics & hardware",
    "history_and_geography": "History & geography",
    "art_and_design": "Art & design",
    "industrial": "Industrial",
}
COLUMN_HEADERS = {"cc_high": "Common Crawl, high quality", "cc_low": "Common Crawl, low quality"}
# Figure geometry in inches: three columns with their own label gutters, one shared row height.
FIGURE_WIDTH = 7.4
ROW_INCH = 0.2
TITLE_ROWS = 1.1  # height of a group title, in row units
GROUP_PAD = 0.5  # half a row above the first and below the last bar pair
GROUP_GAP = 0.5  # between stacked groups of one column
TOP_INCH = 0.62  # title and legend row
BOTTOM_INCH = 0.4
COLUMN_LEFTS = (1.16, 3.48, 5.72)  # axis left edges; gutters fit the longest tick label of each column
COLUMN_WIDTHS = (1.2, 1.15, 1.58)
LABEL_FRACTION = 0.19  # share of each x range reserved for the epoch labels past the longest bar


def candidate_weights(table: Path, candidate: str) -> pd.DataFrame:
    frame = pd.read_csv(table)
    frame = frame[frame["candidate_id"].eq(candidate)].set_index("domain")
    if frame.empty:
        raise ValueError(f"{candidate}: not in {table}")
    return frame[["weight", "materialized_epochs"]].astype(float)


def bucket_label(domain: str) -> str:
    if domain.startswith("dolma3_cc/"):
        topic, _, _quality = domain.removeprefix("dolma3_cc/").rpartition("_")
        return CC_TOPIC_LABELS[topic]  # the column header names the quality stratum
    return layout.bucket_label(domain)


def load_mixtures(target: str, olmix_policy: str = "cap4") -> pd.DataFrame:
    _olmix_id, mariner_id, _label = TARGETS[target]
    olmix_table, olmix_ids, _legend = OLMIX_POLICIES[olmix_policy]
    tokens = base_launch.TOP_LEVEL_DOMAIN_TOKEN_COUNTS
    domains = list(tokens)
    pool = np.asarray([tokens[d] for d in domains], float)
    frame = pd.DataFrame({"proportional_weight": pool / pool.sum()}, index=domains)
    for name, table, candidate in (("olmix", olmix_table, olmix_ids[target]), ("mariner", MARINER_TABLE, mariner_id)):
        rows = candidate_weights(table, candidate).reindex(domains)
        if rows["weight"].isna().any():
            raise ValueError(f"{candidate}: buckets missing from {table}")
        weight = rows["weight"].to_numpy()
        epochs = base_launch.SIMULATED_EPOCH_TARGET_BUDGET * weight / pool
        if not np.allclose(epochs, rows["materialized_epochs"].to_numpy(), atol=EPOCH_TOLERANCE):
            raise ValueError(f"{candidate}: materialized epochs disagree with weight x budget / pool")
        frame[f"{name}_weight"] = weight
        frame[f"{name}_epochs"] = epochs
    frame["label"] = [bucket_label(domain) for domain in frame.index]
    return frame


def column_layout(frame: pd.DataFrame) -> list[list[tuple[str, list[str]]]]:
    """Three columns of titled groups: Common Crawl high-quality cells, low-quality cells (same topic order), then
    the other Dolma 3 sources above Dolmino. Each group is drawn in its own axes so the title sits above the plot."""
    columns = layout.column_rows(frame)
    cc = [key for kind, key in columns["left"] if kind == "bucket"]
    groups: dict[str, list[str]] = {}
    for kind, key in columns["right"]:
        if kind == "header":
            current = layout.FAMILY_HEADERS[key]
            groups[current] = []
        else:
            groups[current].append(key)
    return [
        [(COLUMN_HEADERS["cc_high"], [d for d in cc if d.endswith("_high")])],
        [(COLUMN_HEADERS["cc_low"], [d for d in cc if d.endswith("_low")])],
        list(groups.items()),
    ]


def draw_group(axis: plt.Axes, frame: pd.DataFrame, title: str, domains: list[str], x_max: float, x_axis: bool) -> None:
    ordered = frame.loc[domains]
    ys = -np.arange(len(domains), dtype=float)
    for (name, color, _label), offset in zip(SERIES, OFFSETS, strict=True):
        axis.barh(ys + offset, 100.0 * ordered[f"{name}_weight"].to_numpy(), height=BAR_HEIGHT, color=color, zorder=3)
    axis.vlines(
        100.0 * ordered["proportional_weight"].to_numpy(),
        ys + OFFSETS[-1] - BAR_HEIGHT / 2,
        ys + OFFSETS[0] + BAR_HEIGHT / 2,
        color=layout.INK,
        linewidth=0.9,
        zorder=4,
    )
    for y_row, (_, row) in zip(ys, ordered.iterrows(), strict=True):
        for (name, color, _label), offset in zip(SERIES, OFFSETS, strict=True):
            weight = float(row[f"{name}_weight"])
            # zero-weight labels sit a little further from the spine so their box leaves it unbroken
            axis.text(
                100.0 * weight + (0.012 if weight > 0.0 else 0.025) * x_max,
                y_row + offset,
                layout.epoch_text(weight, float(row[f"{name}_epochs"])),
                va="center",
                ha="left",
                fontsize=5.6,
                color=color if weight > 0.0 else "#8a8a8a",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.06", "facecolor": layout.PAPER, "edgecolor": "none", "alpha": 0.9},
            )
    axis.set_title(title, loc="left", fontsize=7.0, fontweight="bold", color=layout.INK, pad=3.5)
    axis.set_yticks(ys)
    axis.set_yticklabels(ordered["label"], fontsize=6.6, color=layout.INK)
    axis.set_ylim(ys.min() - GROUP_PAD, GROUP_PAD)
    axis.set_xlim(0.0, x_max)
    axis.xaxis.set_major_locator(MultipleLocator(10.0 if x_max > 15.0 else 5.0))
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=layout.GRID, linewidth=0.6, alpha=0.7)
    axis.tick_params(axis="y", colors=layout.INK, width=0, length=0, pad=3)
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    axis.spines["left"].set_color(layout.INK)
    axis.spines["left"].set_linewidth(0.7)
    if x_axis:
        axis.set_xlabel("Mixture weight (%)", fontsize=6.6, color=layout.INK, labelpad=3)
        axis.tick_params(axis="x", colors=layout.INK, labelsize=6.6, width=0.7, length=2.5)
        axis.spines["bottom"].set_color(layout.INK)
        axis.spines["bottom"].set_linewidth(0.7)
    else:  # an upper group shares the column's scale; only the lowest group carries the axis
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
        axis.spines["bottom"].set_visible(False)


def scale_limit(frame: pd.DataFrame, domains: list[str]) -> float:
    """Right x limit in percent: the longest bar plus room for its epoch label."""
    longest = 100.0 * max(frame.loc[domains, f"{name}_weight"].max() for name, _, _ in SERIES)
    return longest * (1.0 + LABEL_FRACTION) + 0.8


def build_figure(frame: pd.DataFrame, target: str, title: bool, olmix_policy: str = "cap4") -> plt.Figure:
    _olmix, _mariner, objective = TARGETS[target]
    legend_labels = {"olmix": OLMIX_POLICIES[olmix_policy][2], "mariner": SERIES[1][2]}
    columns = column_layout(frame)
    cc_domains = [d for column in columns[:2] for _title, group in column for d in group]
    other_domains = [d for _title, group in columns[2] for d in group]
    limits = (scale_limit(frame, cc_domains),) * 2 + (scale_limit(frame, other_domains),)
    # vertical extent of a column in row units: per group a title row, the bar rows and their padding
    spans = [
        sum(TITLE_ROWS + len(group) + 2 * GROUP_PAD for _title, group in column) + GROUP_GAP * (len(column) - 1)
        for column in columns
    ]
    height = TOP_INCH + ROW_INCH * max(spans) + BOTTOM_INCH
    with plt.rc_context(layout.PLOT_STYLE):
        figure = plt.figure(figsize=(FIGURE_WIDTH, height))
        for column, left, width, limit in zip(columns, COLUMN_LEFTS, COLUMN_WIDTHS, limits, strict=True):
            top = 1.0 - TOP_INCH / height
            for index, (group_title, group) in enumerate(column):
                top -= ROW_INCH * TITLE_ROWS / height
                axis_height = ROW_INCH * (len(group) + 2 * GROUP_PAD) / height
                axis = figure.add_axes((left / FIGURE_WIDTH, top - axis_height, width / FIGURE_WIDTH, axis_height))
                draw_group(axis, frame, group_title, group, limit, x_axis=index == len(column) - 1)
                top -= axis_height + ROW_INCH * GROUP_GAP / height
        handles = [Patch(facecolor=color, label=legend_labels[name]) for name, color, _label in SERIES]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="|",
                markersize=8,
                markeredgewidth=1.0,
                color=layout.INK,
                label=layout.TICK,
            )
        )
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 - (0.3 if title else 0.05) / height),
            ncol=3,
            frameon=False,
            fontsize=6.6,
            handlelength=1.4,
            columnspacing=1.2,
        )
        if title:
            figure.text(
                0.5,
                1.0 - 0.1 / height,
                f"{objective} optima at Qwen3 360M/1.6B",
                ha="center",
                va="top",
                fontsize=8.6,
                fontweight="bold",
                color=layout.INK,
            )
        return figure


def summary(frame: pd.DataFrame, olmix_policy: str = "cap4") -> pd.DataFrame:
    labels = {"olmix": OLMIX_POLICIES[olmix_policy][2], "mariner": SERIES[1][2]}
    rows = []
    for name, _color, _label in SERIES:
        weight = frame[f"{name}_weight"].to_numpy()
        epochs = frame[f"{name}_epochs"].to_numpy()
        rows.append(
            {
                "policy": labels[name],
                "active_buckets": int((weight > 1e-4).sum()),
                "max_epochs": float(epochs.max()),
                "largest_weight": float(weight.max()),
                "cc_weight": float(weight[[d.startswith("dolma3_cc/") for d in frame.index]].sum()),
                "tv_to_proportional": float(np.abs(weight - frame["proportional_weight"].to_numpy()).sum() / 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=tuple(TARGETS), default="table9")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-title", action="store_true", help="omit the in-figure title (the caption carries it)")
    parser.add_argument("--olmix", choices=sorted(OLMIX_POLICIES), default="cap4", help="which Olmix optimum to draw")
    parser.add_argument("--stem", default=None, help="output stem; default olmix_vs_mariner_weights_<target>")
    parser.add_argument("--drive-dir", type=Path, default=None, help="copy the figure here as a_<stem>")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_mixtures(args.target, args.olmix)
    stem = args.stem or (
        f"olmix_vs_mariner_weights_{args.target}"
        if args.olmix == "cap4"
        else f"olmix_{args.olmix}_vs_mariner_weights_{args.target}"
    )
    frame.to_csv(args.output_dir / f"{stem}.csv")
    table = summary(frame, args.olmix)
    table.to_csv(args.output_dir / f"{stem}_summary.csv", index=False)
    print(table.round(3).to_string(index=False))
    figure = build_figure(frame, args.target, title=not args.no_title, olmix_policy=args.olmix)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"{stem}.{extension}", dpi=layout.STATIC_DPI)
        if args.drive_dir is not None:
            shutil.copyfile(args.output_dir / f"{stem}.{extension}", args.drive_dir / f"a_{stem}.{extension}")
    plt.close(figure)
    print(f"wrote {args.output_dir / stem}.pdf")


if __name__ == "__main__":
    main()
