# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Compact paper plot for raw-optimum validation vs fit-set size."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from experiments.domain_phase_mix.exploratory.two_phase_many.convergence_plot_style import (
    BEST_OBSERVED_BPB_COLOR,
    GRP_COLOR,
    OLMIX_COLOR,
    PROPORTIONAL_BPB_COLOR,
    REGMIX_COLOR,
)

ROOT = Path(__file__).resolve().parents[1] / "two_phase_many"
PAPER_ROOT = Path(__file__).resolve().parent
IMG_DIR = PAPER_ROOT / "img"

GRP_CSV = ROOT / "two_phase_many_grp_power_family_penalty_raw_curve_points.csv"
REGMIX_CSV = ROOT / "two_phase_many_regmix_raw_curve_points.csv"
OLMIX_CSV = ROOT / "two_phase_many_olmix_loglinear_subset_curve_points.csv"
BASELINE_CSV = ROOT / "two_phase_many_all_60m_1p2b.csv"
OUTPUT_STEM = IMG_DIR / "f8_raw_optimum_convergence"
OUTPUT_CSV = IMG_DIR / "f8_raw_optimum_convergence_points.csv"

OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
Y_MIN = 1.025
Y_MAX = 1.106
CLIP_MARKER_Y = Y_MAX - 0.008
CLIP_LABEL_Y = Y_MAX - 0.003
METHOD_SPECS = (
    ("GRP", GRP_CSV, GRP_COLOR, "o"),
    ("RegMix", REGMIX_CSV, REGMIX_COLOR, "s"),
    ("Olmix", OLMIX_CSV, OLMIX_COLOR, "D"),
)
CLIPPED_X_OFFSETS = {
    "GRP": -2.2,
    "RegMix": 0.0,
    "Olmix": 2.2,
}
CLIPPED_LABEL_OFFSETS = {
    "GRP": (-1.0, 0.0),
    "RegMix": (0.0, 0.0),
    "Olmix": (1.0, 0.0),
}
TEXT_COLOR = "#232B32"
GRID_COLOR = "#E6E2DA"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#A8A29A",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _method_frame(method: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["subset_size", "actual_validated_bpb", "subset_best_observed_bpb"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = frame[required].copy()
    out["method"] = method
    out["is_clipped_high"] = out["actual_validated_bpb"] > Y_MAX
    out["plot_bpb"] = out["actual_validated_bpb"].clip(upper=Y_MAX)
    return out


def _baseline_bpb(run_name: str) -> float:
    frame = pd.read_csv(BASELINE_CSV, usecols=["run_name", OBJECTIVE_METRIC])
    matches = frame.loc[frame["run_name"] == run_name, OBJECTIVE_METRIC].dropna()
    if matches.empty:
        raise ValueError(f"Missing {run_name} in {BASELINE_CSV}")
    return float(matches.iloc[-1])


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    frames = [_method_frame(method, path) for method, path, _, _ in METHOD_SPECS]
    plot_frame = pd.concat(frames, ignore_index=True)
    plot_frame.to_csv(OUTPUT_CSV, index=False)
    subset_sizes = sorted(plot_frame["subset_size"].unique())
    best_observed = (
        plot_frame.loc[plot_frame["method"] == "GRP", ["subset_size", "subset_best_observed_bpb"]]
        .drop_duplicates()
        .sort_values("subset_size")
    )
    proportional_bpb = _baseline_bpb("baseline_proportional")

    fig, ax = plt.subplots(figsize=(8.2, 3.75), constrained_layout=True)
    for method, _, color, marker in METHOD_SPECS:
        method_frame = plot_frame.loc[plot_frame["method"] == method].sort_values("subset_size")
        # Plot actual y values and let the axes clip them. This keeps the slope
        # toward off-chart optima visible instead of flattening at the clip band.
        ax.plot(
            method_frame["subset_size"],
            method_frame["actual_validated_bpb"],
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=5.5,
            label=method,
            clip_on=True,
        )
        clipped = method_frame.loc[method_frame["is_clipped_high"]]
        if not clipped.empty:
            clipped_x = clipped["subset_size"].astype(float) + CLIPPED_X_OFFSETS[method]
            ax.scatter(
                clipped_x,
                np.full(len(clipped), CLIP_MARKER_Y),
                color=color,
                marker="^",
                s=60,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
            for _idx, row in enumerate(clipped.itertuples(index=False)):
                label_x_offset, label_y_offset = CLIPPED_LABEL_OFFSETS[method]
                ax.text(
                    float(row.subset_size) + CLIPPED_X_OFFSETS[method] + label_x_offset,
                    CLIP_LABEL_Y + label_y_offset,
                    f"{float(row.actual_validated_bpb):.2f}",
                    ha="center",
                    va="top",
                    fontsize=7.6,
                    color=color,
                    clip_on=True,
                )

    ax.plot(
        best_observed["subset_size"],
        best_observed["subset_best_observed_bpb"],
        color=BEST_OBSERVED_BPB_COLOR,
        marker="P",
        linewidth=1.6,
        linestyle=":",
        markersize=5.2,
        label="Best observed in subset",
    )
    ax.axhline(proportional_bpb, color=PROPORTIONAL_BPB_COLOR, linewidth=1.3, linestyle="--", label="Proportional")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(min(subset_sizes) - 6, max(subset_sizes) + 8)
    ax.set_xticks(subset_sizes)
    ax.set_xlabel("60M swarm runs used for fitting")
    ax.set_ylabel("Validated BPB")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=5,
        frameon=False,
        fontsize=8.7,
        handlelength=1.7,
        columnspacing=1.0,
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
