# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Figure B5a-B at every repetition burden: fixed simulated-epoching ratio, scaling D and the StarCoder subset together.

One panel per burden level of the dense-replay atlas (0.125x to 4x the target repetition rate, left to right;
the 4x panel is Figure B5a-B). Every panel shows the same 210M proxy at four token horizons whose StarCoder
subsets grow with the horizon so that D_r / S_SC stays fixed; the star marks the observed minimum.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ATLAS_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_all_tied_curves_canonical_dsp_20260902"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_replay_burden_ladder_20260905"
FAMILY = "dense_horizon_replay"
TARGET_EPOCHS_AT_P1 = 26.457867  # D_tgt / P_SC = 5.73T / 216.57B
HORIZON_TOKENS = {"r0": 1.00e9, "r1": 1.92e9, "r2": 3.92e9, "r3": 7.41e9}
HORIZON_COLORS = {"r0": "#E377C2", "r1": "#7FCDBB", "r2": "#2A9D8F", "r3": "#0B5345"}
BURDENS = (
    (0.125, "0.125x burden"),
    (0.25, "0.25x burden"),
    (0.5, "0.5x burden"),
    (1.0, "1x burden"),
    (2.0, "2x burden"),
    (4.0, "4x burden"),
)
INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
DPI = 300


def load_curves() -> pd.DataFrame:
    reference = pd.read_csv(ATLAS_DIR / "curve_reference.csv")
    predictions = pd.read_csv(ATLAS_DIR / "predictions.csv")
    reference = reference[reference["family"].eq(FAMILY)].copy()
    reference["rung"] = reference["curve_id"].str.extract(r"dense_replay__(r\d)_")[0]
    frame = predictions.merge(reference[["curve_ref", "curve_label", "rung", "starcoder_epochs_at_p1"]], on="curve_ref")
    return frame


def format_tokens(tokens: float) -> str:
    if tokens >= 1e9:
        return f"{tokens / 1e9:.2f}B"
    return f"{tokens / 1e6:.1f}M"


def draw_panel(
    axis: plt.Axes, frame: pd.DataFrame, burden: float, label: str, letter: str, first: bool
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    part = frame[frame["curve_label"].eq(label)]
    epochs_at_p1 = float(part["starcoder_epochs_at_p1"].iloc[0])
    handles, texts = [], []
    for rung in ("r0", "r1", "r2", "r3"):
        curve = part[part["rung"].eq(rung)].sort_values("starcoder_weight")
        if curve.empty:
            continue
        p = curve["starcoder_weight"].to_numpy(float)
        bpb = curve["observed_bpb"].to_numpy(float)
        color = HORIZON_COLORS[rung]
        (line,) = axis.plot(
            p,
            bpb,
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=2.4,
            markerfacecolor=color,
            markeredgecolor=color,
            zorder=3,
        )
        best = int(np.argmin(bpb))
        axis.plot(
            [p[best]],
            [bpb[best]],
            marker="*",
            markersize=9,
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            zorder=5,
        )
        tokens = HORIZON_TOKENS[rung]
        subset = tokens / epochs_at_p1
        handles.append(line)
        texts.append(f"{format_tokens(tokens)}  {format_tokens(subset)}  {epochs_at_p1 * p[best]:.1f}")
        rows.append(
            {
                "burden": burden,
                "rung": rung,
                "tokens": tokens,
                "subset_tokens": subset,
                "epochs_at_p1": epochs_at_p1,
                "best_p": p[best],
                "best_bpb": bpb[best],
                "best_epochs": epochs_at_p1 * p[best],
            }
        )
    axis.set_title(
        f"{letter}. {burden:g}" + r"$\times$ target repetition", loc="left", fontsize=8, fontweight="bold", color=INK
    )
    axis.set_xlim(0, 1.0)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
    axis.tick_params(colors=INK, labelsize=7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
    axis.set_xlabel("StarCoder fraction p", color=INK, fontsize=7.5)
    if first:
        axis.set_ylabel("Programming Languages BPB", color=INK, fontsize=7.5)
    top = axis.secondary_xaxis("top", functions=(lambda x: x * epochs_at_p1, lambda e: e / epochs_at_p1))
    top.tick_params(colors=INK, labelsize=6.5, length=2)
    top.set_xlabel("StarCoder epochs", color=INK, fontsize=6.5)
    top.spines["top"].set_visible(False)
    legend = axis.legend(
        handles,
        texts,
        title="D      $S_{\\mathrm{SC}}$    $E^*_{\\mathrm{SC}}$",
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        fontsize=6,
        title_fontsize=6,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.25,
    )
    legend.get_title().set_ha("left")
    return rows


def build_figure(frame: pd.DataFrame, rows_of_panels: int = 1) -> tuple[plt.Figure, pd.DataFrame]:
    """One panel per degree of downsampling, in ``rows_of_panels`` rows (two rows fit a page width)."""
    columns = -(-len(BURDENS) // rows_of_panels)
    figure, axes = plt.subplots(rows_of_panels, columns, figsize=(2.35 * columns, 2.9 * rows_of_panels), squeeze=False)
    rows: list[dict[str, object]] = []
    for index, (burden, label) in enumerate(BURDENS):
        axis = axes[index // columns][index % columns]
        rows.extend(draw_panel(axis, frame, burden, label, "ABCDEFG"[index], index % columns == 0))
    figure.tight_layout(w_pad=1.0, h_pad=1.6)
    return figure, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--drive-stem", default="a_replay_burden_ladder", help="file stem of the Drive copy")
    parser.add_argument("--rows", type=int, default=1, help="rows of panels (2 for the appendix page width)")
    args = parser.parse_args()
    plt.rcParams.update(PLOT_STYLE)
    figure, table = build_figure(load_curves(), args.rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"replay_burden_ladder.{extension}", dpi=DPI, bbox_inches="tight")
        if args.drive_dir is not None:
            shutil.copyfile(
                args.output_dir / f"replay_burden_ladder.{extension}",
                args.drive_dir / f"{args.drive_stem}.{extension}",
            )
    table.to_csv(args.output_dir / "replay_burden_ladder_optima.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
