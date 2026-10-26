# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Appendix figure: why MARINER mis-orders the SciQ frontier of the held-out optima.

(A) MARINER's and Olmix's predicted SciQ loss against the measured value at the 157 held-out OlmoBaseEval Easy optima,
the 30 best measured circled on MARINER's predictions, axes cut at the 95th percentile, and the swarm's lowest
measured SciQ value marked. (B) The SciQ residual (measured minus
predicted) against each optimum's Stack-Edu FIM exposure, with the swarm's 95th-percentile exposure marked. (C) The same
residual for HellaSwag, a component of the same objective at the same SNR. Data from
`analyze_heldout_component_ordering_20260929.component_frame`.

Writes `sciq_head_diagnosis.pdf/.png` to `reference_outputs/heldout_table9_components_20260929/`.

usage: uv run --offline --no-sync python plot_sciq_head_diagnosis_20260929.py
"""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import analyze_heldout_component_ordering_20260929 as ordering
import matplotlib.pyplot as plt
import plot_analysis_floor_20260928 as floor
from matplotlib.lines import Line2D

OUTPUT = ordering.OUTPUT
EXPOSURE_BUCKET = "dolmino_stack_edu_fim"
MARINER_COLOR = floor.original.MARINER_COLOR
OLMIX_COLOR = "#C0397B"
BEST = 30
RESIDUAL_LIMITS = (-0.35, 0.15)


def style(axis, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=6.6, fontweight="bold", linespacing=1.1)
    axis.tick_params(labelsize=6)
    axis.grid(True, color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def draw_predictions(axis, frame: ordering.ComponentFrame) -> None:
    best = np.argsort(frame.measured, kind="stable")[:BEST]
    lo = min(frame.measured.min(), frame.mariner.min(), frame.olmix.min())
    hi = max(np.quantile(frame.mariner, 0.95), np.quantile(frame.measured, 0.95))
    pad = 0.05 * (hi - lo)
    axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#c9ccd1", linewidth=0.8, zorder=1)
    axis.axvline(frame.swarm_measured.min(), color=floor.CEILING, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    axis.scatter(frame.measured, frame.olmix, s=6, color=OLMIX_COLOR, alpha=0.7, linewidths=0, zorder=3)
    axis.scatter(frame.measured, frame.mariner, s=6, color=MARINER_COLOR, alpha=0.7, linewidths=0, zorder=4)
    axis.scatter(
        frame.measured[best], frame.mariner[best], s=16, facecolor="none", edgecolor="#222222", linewidths=0.5, zorder=5
    )
    axis.set_xlim(lo - pad, hi + pad)
    axis.set_ylim(lo - pad, hi + pad)
    axis.set_aspect("equal")
    axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    axis.set_xlabel("Measured SciQ loss (BPB)", fontsize=6.8)
    axis.set_ylabel("Predicted (BPB)", fontsize=6.8)
    axis.text(
        frame.swarm_measured.min() + 0.004,
        lo - pad + 0.02 * (hi - lo),
        "swarm\nminimum",
        fontsize=5.6,
        ha="left",
        va="bottom",
        color=floor.CEILING,
    )


def draw_residuals(axis, frame: ordering.ComponentFrame, label: str) -> None:
    x = frame.exposures[EXPOSURE_BUCKET].to_numpy()
    low, high = RESIDUAL_LIMITS
    axis.axhline(0.0, color="#c9ccd1", linewidth=0.8, zorder=1)
    axis.axvline(
        frame.swarm_exposures[EXPOSURE_BUCKET].quantile(0.95),
        color=floor.CEILING,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=2,
    )
    for prediction, color, zorder in ((frame.olmix, OLMIX_COLOR, 3), (frame.mariner, MARINER_COLOR, 4)):
        residual = frame.measured - prediction
        inside = (residual >= low) & (residual <= high)
        axis.scatter(x[inside], residual[inside], s=6, color=color, alpha=0.7, linewidths=0, zorder=zorder)
        clipped = ~inside
        if clipped.any():
            axis.scatter(
                x[clipped],
                np.clip(residual[clipped], low, high),
                s=9,
                marker="v",
                color=color,
                linewidths=0,
                zorder=zorder + 2,
            )
    axis.set_ylim(low - 0.02, high + 0.02)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 3, 10]))
    axis.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["1", "3", "10"]))
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_xlabel("Stack-Edu FIM epochs", fontsize=6.8)
    axis.set_ylabel(f"{label} residual (BPB)", fontsize=6.8)
    axis.text(
        frame.swarm_exposures[EXPOSURE_BUCKET].quantile(0.95) * 1.1,
        high,
        "swarm\n95th pct.",
        fontsize=5.6,
        ha="left",
        va="top",
        color=floor.CEILING,
    )


def main() -> None:
    sciq = ordering.component_frame("table9", "sciq")
    hellaswag = ordering.component_frame("table9", "hellaswag")
    plt.rcParams.update(floor.original.PLOT_STYLE)
    plt.rcParams.update({"text.usetex": False})
    figure = plt.figure(figsize=(5.5, 1.75), facecolor="white")
    axes = [figure.add_axes(rect) for rect in ((0.07, 0.2, 0.2, 0.66), (0.4, 0.2, 0.24, 0.66), (0.74, 0.2, 0.24, 0.66))]
    draw_predictions(axes[0], sciq)
    style(axes[0], "A. SciQ, held-out optima")
    draw_residuals(axes[1], sciq, "SciQ")
    style(axes[1], "B. SciQ residual by exposure")
    draw_residuals(axes[2], hellaswag, "HellaSwag")
    style(axes[2], "C. HellaSwag residual by exposure")
    handles = [
        Line2D([], [], color=MARINER_COLOR, marker="o", markersize=3, linewidth=0, label="MARINER"),
        Line2D([], [], color=OLMIX_COLOR, marker="o", markersize=3, linewidth=0, label="Olmix"),
        Line2D(
            [],
            [],
            color="#222222",
            marker="o",
            markersize=3.6,
            markerfacecolor="none",
            linewidth=0,
            label="30 best, MARINER",
        ),
    ]
    axes[0].legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        fontsize=5.2,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.25,
    )
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"sciq_head_diagnosis.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUTPUT / "sciq_head_diagnosis.pdf")
    rows = []
    for frame in (sciq, hellaswag):
        x = frame.exposures[EXPOSURE_BUCKET].to_numpy()
        above = x > frame.swarm_exposures[EXPOSURE_BUCKET].quantile(0.95)
        for label, prediction in (("MARINER", frame.mariner), ("Olmix", frame.olmix)):
            residual = frame.measured - prediction
            rows.append(
                f"{frame.component} {label}: residual mean above swarm p95 {residual[above].mean():+.3f} "
                f"(n={above.sum()}), below {residual[~above].mean():+.3f}"
            )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
