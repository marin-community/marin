# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy", "pandas"]
# ///

"""Compare the cap-6 WSPU optima for the full Uncheatable aggregate and for its three worsened components.

One horizontal grouped-bar chart: for every runtime bucket, one bar per mixture with the
bucket's materialized epochs written at the bar end, plus a tick at the proportional weight.
Rows are sorted by the weight the three-component optimum adds relative to the full optimum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
FULL_SWEEP = REFERENCE_OUTPUTS / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902" / "candidate_weights.csv"
WORSENED_SWEEP = REFERENCE_OUTPUTS / "delphi_one_phase_wspu_worsened_components_sweep_20260905" / "candidate_weights.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_one_phase_wspu_worsened_components_sweep_20260905"
FULL_ID = "wspu_uncheatable_cap06"
WORSENED_ID = "wspu_worsened_cap06"

PAPER = "#ffffff"
INK = "#111111"
GRID = "#b8b8b8"
FULL_COLOR = "#178A72"
WORSENED_COLOR = "#D95F32"
FIGURE_SIZE = (7.4, 5.9)
# Two columns: Common Crawl cells on the left, the other Dolma 3 sources and Dolmino on the right, top-aligned
# with the same row spacing; the legend sits in the space below the shorter right column.
LEFT_AXES = (0.255, 0.075, 0.255, 0.85)
RIGHT_AXES_LEFT = 0.705
RIGHT_AXES_WIDTH = 0.275
COLUMN_PAD = 0.7
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
BAR_HEIGHT = 0.36
BAR_OFFSET = 0.2
LABEL_PAD_PERCENT = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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
FAMILY_HEADERS = {
    "cc": "Dolma 3 Common Crawl",
    "dolma3": "Dolma 3 other sources",
    "dolmino": "Dolmino",
}
HEADER_GAP = 0.9
TICK = "Proportional weight"


def family(domain: str) -> str:
    if domain.startswith("dolma3_cc/"):
        return "cc"
    if domain.startswith("dolma3_"):
        return "dolma3"
    return "dolmino"


def bucket_label(domain: str) -> str:
    if domain.startswith("dolma3_cc/"):
        topic, _, quality = domain.removeprefix("dolma3_cc/").rpartition("_")
        return f"{topic.replace('_', ' ').capitalize()}, {quality}"
    return OTHER_LABELS[domain]


def ordered_rows(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (kind, key) rows top to bottom: family headers, then buckets by natural size."""
    return column_rows(frame)["left"] + column_rows(frame)["right"]


def column_rows(frame: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """Rows per column: Common Crawl cells (left); other Dolma 3 sources then Dolmino (right)."""
    left: list[tuple[str, str]] = []
    cc = frame.loc[[family(domain) == "cc" for domain in frame.index]].copy()
    cc["topic"] = [domain.removeprefix("dolma3_cc/").rpartition("_")[0] for domain in cc.index]
    topic_size = cc.groupby("topic")["proportional_weight"].sum().sort_values(ascending=False)
    left.append(("header", "cc"))
    for topic in topic_size.index:
        for quality in ("high", "low"):
            left.append(("bucket", f"dolma3_cc/{topic}_{quality}"))
    right: list[tuple[str, str]] = []
    for name in ("dolma3", "dolmino"):
        members = frame.loc[[family(domain) == name for domain in frame.index]]
        right.append(("header", name))
        for domain in members.sort_values("proportional_weight", ascending=False).index:
            right.append(("bucket", domain))
    return {"left": left, "right": right}


def load_mixtures() -> pd.DataFrame:
    """Per-bucket weights and materialized epochs of the two cap-6 optima, with the proportional weight."""
    full = pd.read_csv(FULL_SWEEP)
    full = full[full["candidate_id"].eq(FULL_ID)].set_index("domain")
    worsened = pd.read_csv(WORSENED_SWEEP)
    worsened = worsened[worsened["candidate_id"].eq(WORSENED_ID)].set_index("domain")
    if set(full.index) != set(worsened.index):
        raise ValueError("the two sweeps do not share the same buckets")
    worsened = worsened.loc[full.index]
    if not np.allclose(full["proportional_weight"], worsened["proportional_weight"]):
        raise ValueError("proportional weights differ between the sweeps")
    if not np.allclose(worsened["full_uncheatable_optimum_weight"], full["weight"]):
        raise ValueError("the worsened sweep's copy of the full optimum does not match the full sweep")
    frame = pd.DataFrame(
        {
            "full_weight": full["weight"],
            "full_epochs": full["materialized_epochs"],
            "worsened_weight": worsened["weight"],
            "worsened_epochs": worsened["materialized_epochs"],
            "proportional_weight": full["proportional_weight"],
        }
    )
    frame["label"] = [bucket_label(domain) for domain in frame.index]
    return frame


def epoch_text(weight: float, epochs: float) -> str:
    if weight <= 0.0:
        return "0"
    return f"{epochs:.2f}" if epochs < 0.1 else f"{epochs:.1f}"


def layout_rows(rows: list[tuple[str, str]]) -> tuple[dict[str, float], list[tuple[float, str]]]:
    positions: dict[str, float] = {}
    headers: list[tuple[float, str]] = []
    y = 0.0
    for kind, key in rows:
        if kind == "header":
            y -= HEADER_GAP if headers or positions else 0.0
            headers.append((y, FAMILY_HEADERS[key]))
            y -= 1.0
        else:
            positions[key] = y
            y -= 1.0
    return positions, headers


def draw_column(axis: plt.Axes, frame: pd.DataFrame, rows: list[tuple[str, str]], x_max: float) -> float:
    """Draw one column of grouped bars; return the vertical span in row units."""
    positions, headers = layout_rows(rows)
    ordered = frame.loc[list(positions)]
    ys = np.asarray([positions[domain] for domain in ordered.index])
    full_pct = 100.0 * ordered["full_weight"].to_numpy()
    worsened_pct = 100.0 * ordered["worsened_weight"].to_numpy()
    proportional_pct = 100.0 * ordered["proportional_weight"].to_numpy()
    axis.barh(ys + BAR_OFFSET, full_pct, height=BAR_HEIGHT, color=FULL_COLOR, edgecolor="none", zorder=3)
    axis.barh(ys - BAR_OFFSET, worsened_pct, height=BAR_HEIGHT, color=WORSENED_COLOR, edgecolor="none", zorder=3)
    axis.vlines(
        proportional_pct,
        ys - BAR_OFFSET - BAR_HEIGHT / 2,
        ys + BAR_OFFSET + BAR_HEIGHT / 2,
        color=INK,
        linewidth=1.1,
        zorder=2,
    )
    for y_row, (_, row) in zip(ys, ordered.iterrows(), strict=True):
        for offset, weight, epochs, color in (
            (BAR_OFFSET, row["full_weight"], row["full_epochs"], FULL_COLOR),
            (-BAR_OFFSET, row["worsened_weight"], row["worsened_epochs"], WORSENED_COLOR),
        ):
            axis.text(
                100.0 * weight + LABEL_PAD_PERCENT,
                y_row + offset,
                epoch_text(weight, epochs),
                va="center",
                ha="left",
                fontsize=6.4,
                color=color if weight > 0.0 else "#777777",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.9},
            )
    for y_header, text in headers:
        axis.text(
            -0.012,
            y_header,
            text,
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=INK,
        )
    axis.set_yticks(ys)
    axis.set_yticklabels(ordered["label"], fontsize=7.2, color=INK)
    axis.set_ylim(ys.min() - COLUMN_PAD, COLUMN_PAD)
    axis.set_xlim(0.0, x_max)
    axis.set_xlabel("Mixture weight (%)", fontsize=8.5, color=INK, labelpad=6)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID, linewidth=0.65, alpha=0.72)
    axis.tick_params(axis="x", colors=INK, labelsize=7.5, width=0.8, length=3)
    axis.tick_params(axis="y", colors=INK, width=0, length=0, pad=4)
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(INK)
        axis.spines[name].set_linewidth(0.8)
    return float(-ys.min() + 2 * COLUMN_PAD)


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    columns = column_rows(frame)
    x_max = 100.0 * max(frame["full_weight"].max(), frame["worsened_weight"].max()) + 3.5
    with plt.rc_context(PLOT_STYLE):
        figure = plt.figure(figsize=FIGURE_SIZE)
        left_axis = figure.add_axes(LEFT_AXES)
        left_span = draw_column(left_axis, frame, columns["left"], x_max)
        # Probe the right column's span with a throwaway axis so its height keeps the left column's row spacing.
        probe = figure.add_axes((0.0, 0.0, 0.01, 0.01))
        right_span = draw_column(probe, frame, columns["right"], x_max)
        probe.remove()
        left_x, left_bottom, _left_width, left_height = LEFT_AXES
        right_height = left_height * right_span / left_span
        right_axis = figure.add_axes(
            (RIGHT_AXES_LEFT, left_bottom + left_height - right_height, RIGHT_AXES_WIDTH, right_height)
        )
        draw_column(right_axis, frame, columns["right"], x_max)
        handles = [
            Patch(facecolor=FULL_COLOR, label="Full Uncheatable optimum (cap 6)"),
            Patch(facecolor=WORSENED_COLOR, label="News, fiction and Wikipedia optimum (cap 6)"),
            Line2D([0], [0], linestyle="none", marker="|", markersize=9, markeredgewidth=1.1, color=INK, label=TICK),
        ]
        figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(RIGHT_AXES_LEFT - 0.17, left_bottom + left_height - right_height - 0.09),
            frameon=True,
            framealpha=0.95,
            edgecolor=GRID,
            fontsize=7.4,
        )
        figure.text(
            0.5,
            0.985,
            "WSPU optima at epoch cap 6; bar labels give materialized epochs",
            ha="center",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            color=INK,
        )
        del left_x
        return figure


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_mixtures()
    frame.loc[[key for kind, key in ordered_rows(frame) if kind == "bucket"]].to_csv(
        args.output_dir / "mixture_comparison_cap06.csv"
    )
    figure = build_figure(frame)
    figure.savefig(args.output_dir / "mixture_comparison_cap06.png", dpi=STATIC_DPI)
    figure.savefig(args.output_dir / "mixture_comparison_cap06.pdf")
    plt.close(figure)
    print(f"Wrote {args.output_dir / 'mixture_comparison_cap06.png'}")


if __name__ == "__main__":
    main()
