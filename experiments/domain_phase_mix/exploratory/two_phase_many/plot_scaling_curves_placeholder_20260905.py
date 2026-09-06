# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Placeholder scaling-curve figure: completed Delphi compute ladders plus the WSPU optima at 3e18 only.

Uses the 2026-07-11 W&B snapshot in ``delphi_scaling_progress_20260625`` for the ladders (3e18 to 1e21 FLOPs)
and the measured WSPU epoch-cap optima at 3e18. The WSPU points at the larger scales are not trained yet; the
figure says so.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT = SCRIPT_DIR / "reference_outputs" / "delphi_scaling_progress_20260625" / "delphi_scaling_completed_wandb.csv"
WSPU_MEASURED = (
    SCRIPT_DIR
    / "reference_outputs"
    / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"
    / "measured_results.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "scaling_curves_placeholder_20260905"
INK = "#111111"
GRID = "#b8b8b8"
PAPER = "white"
WSPU_COLOR = "#178A72"
OLMIX_COLOR = "#CC79A7"
DSP_COLOR = "#E24731"
PROPORTIONAL_COLOR = "#6C6F7D"
UNIMAX_COLOR = "#4C78A8"
SCALES = (3e18, 2e19, 3e20, 1e21)
PANELS = (
    (
        "uncheatable",
        "eval_uncheatable_eval_bpb",
        "Uncheatable BPB",
        (
            ("proportional", "Proportional", PROPORTIONAL_COLOR, "-"),
            ("unimax8", "UniMax-8", UNIMAX_COLOR, "-"),
            ("olmix_onephase_uncheatable_d001_kl005_cap4", "Olmix optimum (cap 4, KL 0.05)", OLMIX_COLOR, "-"),
            ("dsp_onephase_effexp_uncheatable_kl0p1", "DSP optimum (KL 0.1)", DSP_COLOR, "-"),
        ),
        "wspu_uncheatable_cap06",
        "uncheatable_bpb",
    ),
    (
        "table9",
        "olmo_base_easy_table9_51_component_macro_bpb",
        "OlmoBaseEval Easy mean BPB",
        (
            ("proportional", "Proportional", PROPORTIONAL_COLOR, "-"),
            ("unimax8", "UniMax-8", UNIMAX_COLOR, "-"),
            ("olmix_onephase_table9_d001_kl0p005_cap4", "Olmix optimum (cap 4, KL 0.005)", OLMIX_COLOR, "-"),
        ),
        "wspu_table9_cap06",
        "table9_macro_bpb",
    ),
)
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


def build_figure(snapshot: pd.DataFrame, wspu: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for axis, (target, column, y_label, series, wspu_id, wspu_column), letter in zip(axes, PANELS, "AB", strict=True):
        for mixture, label, color, style in series:
            frame = snapshot[snapshot["mixture"].eq(mixture) & snapshot["is_completed"]].sort_values("flops")
            frame = frame[frame[column].notna()]
            axis.plot(
                frame["flops"],
                frame[column],
                color=color,
                linestyle=style,
                linewidth=1.3,
                marker="o",
                markersize=3.2,
                markerfacecolor=color,
                markeredgecolor=PAPER,
                zorder=3,
                label=label,
            )
        measured = float(wspu[wspu["candidate_id"].eq(wspu_id)][wspu_column].iloc[0])
        axis.plot(
            [3e18],
            [measured],
            marker="*",
            markersize=11,
            color=WSPU_COLOR,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            linestyle="none",
            zorder=5,
            label="WSPU optimum (cap 6), measured",
        )
        axis.plot(
            [],
            [],
            marker="*",
            markersize=11,
            markerfacecolor=PAPER,
            markeredgecolor=WSPU_COLOR,
            linestyle="none",
            label="WSPU optimum, to be trained",
        )
        # Slots for the untrained scales, drawn clearly below every measured curve so they cannot read as data.
        completed = snapshot[snapshot["mixture"].isin([m for m, *_ in series]) & snapshot["is_completed"]]
        for scale in SCALES[1:]:
            floor = float(completed[completed["flops"].eq(scale)][column].min()) - 0.04
            axis.plot(
                [scale],
                [floor],
                marker="*",
                markersize=11,
                markerfacecolor=PAPER,
                markeredgecolor=WSPU_COLOR,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=5,
            )
        axis.set_xscale("log")
        axis.set_xticks(SCALES)
        axis.set_xticklabels(["3e18", "2e19", "3e20", "1e21"])
        axis.minorticks_off()
        axis.set_xlabel("Training compute (FLOPs)", color=INK, fontsize=7.5)
        axis.set_ylabel(y_label, color=INK, fontsize=7.5)
        axis.set_title(
            f"{letter}. {'Uncheatable' if target == 'uncheatable' else 'OlmoBaseEval Easy'}",
            loc="left",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )
        axis.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.6, zorder=0)
        axis.tick_params(colors=INK, labelsize=7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(INK)
        axis.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.6)
    figure.text(
        0.5,
        -0.04,
        "PLACEHOLDER: WSPU optima are measured at 3e18 only; their 2e19, 3e20 and 1e21 runs remain to be trained. "
        "Ladders from the 2026-07-11 snapshot.",
        ha="center",
        va="top",
        fontsize=6.8,
        color="#B00020",
        fontweight="bold",
    )
    figure.tight_layout(w_pad=1.5)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    snapshot = pd.read_csv(SNAPSHOT)
    wspu = pd.read_csv(WSPU_MEASURED)
    plt.rcParams.update(PLOT_STYLE)
    figure = build_figure(snapshot, wspu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"scaling_curves_placeholder.{extension}", dpi=DPI, bbox_inches="tight")
        if args.drive_dir is not None:
            shutil.copyfile(
                args.output_dir / f"scaling_curves_placeholder.{extension}",
                args.drive_dir / f"r1_scaling_curves_placeholder.{extension}",
            )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
