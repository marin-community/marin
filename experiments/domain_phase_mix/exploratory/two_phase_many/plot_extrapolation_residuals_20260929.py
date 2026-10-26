# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Figure 14 of the Analysis: MARINER's held-out residual against each optimum's distance from the swarm.

For arXiv physics (Uncheatable) and SciQ (OlmoBaseEval Easy), the measured minus predicted loss at every held-out
optimum is plotted against the optimum's total-variation distance to the nearest of the 280 swarm mixtures, with the
5 best-measured mixtures of each component circled and the best drawn as a star. Data from
`component_frame` in `analyze_heldout_component_ordering_20260929.py`.
Residuals below the axis are drawn as triangles at its edge.

Writes `extrapolation_residuals.pdf/.png` to `reference_outputs/heldout_table9_components_20260929/`.

usage: uv run --offline --no-sync python plot_extrapolation_residuals_20260929.py
"""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import analyze_heldout_component_ordering_20260929 as ordering
import matplotlib.pyplot as plt
import plot_analysis_floor_20260928 as floor
from matplotlib.lines import Line2D

OUTPUT = ordering.OUTPUT
COMPONENTS = (("uncheatable", "arxiv_physics", "arXiv physics"), ("table9", "sciq", "SciQ"))
COLORS = floor.TARGET_COLORS
BEST = 5
LIMITS = (-0.3, 0.12)


def nearest_swarm_distance(frame: ordering.ComponentFrame) -> np.ndarray:
    """Total-variation distance from each held-out optimum's mixture to the nearest swarm mixture."""
    _, swarm, _ = ordering.swarm_inputs()
    weights = frame.exposures.to_numpy() / np.asarray(swarm.inventory, float)[None, :]
    return 0.5 * np.abs(weights[:, None, :] - swarm.weights[None, :, :]).sum(axis=2).min(axis=1)


def main() -> None:
    plt.rcParams.update(floor.original.PLOT_STYLE)
    plt.rcParams.update({"text.usetex": False})
    figure = plt.figure(figsize=(2.3, 1.55), facecolor="white")
    axis = figure.add_axes((0.24, 0.24, 0.73, 0.58))
    low, high = LIMITS
    axis.axhline(0.0, color="#c9ccd1", linewidth=0.8, zorder=1)
    handles = []
    for target, component, label in COMPONENTS:
        frame = ordering.component_frame(target, component)
        distance = nearest_swarm_distance(frame)
        residual = frame.measured - frame.mariner
        order = np.argsort(frame.measured, kind="stable")
        best, top = order[1:BEST], order[0]  # the best itself is starred, the next four circled
        inside = residual >= low
        color = COLORS[target]
        axis.scatter(distance[inside], residual[inside], s=5, color=color, alpha=0.75, linewidths=0, zorder=3)
        axis.scatter(
            distance[~inside], np.full((~inside).sum(), low), s=9, marker="v", color=color, linewidths=0, zorder=4
        )
        axis.scatter(
            distance[best],
            np.clip(residual[best], low, high),
            s=16,
            facecolor="none",
            edgecolor="#222222",
            linewidths=0.5,
            zorder=5,
        )
        axis.scatter(
            distance[top],
            np.clip(residual[top], low, high),
            s=40,
            marker="*",
            color=color,
            edgecolor="#222222",
            linewidths=0.5,
            zorder=6,
        )
        handles.append(Line2D([], [], color=color, marker="o", markersize=3, linewidth=0, label=label))
        print(
            f"{label}: n {len(residual)}, residual below axis {int((~inside).sum())}, "
            f"best-{BEST} median distance {np.median(distance[best]):.2f}"
        )
    handles += [
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=4,
            markeredgecolor="#222222",
            markerfacecolor="none",
            linewidth=0,
            label=f"{BEST} best held-out mixtures",
        ),
        Line2D(
            [],
            [],
            color="#6b6f78",
            marker="*",
            markersize=6.5,
            markeredgecolor="#222222",
            linewidth=0,
            label="best held-out mixture",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.01),
        ncol=2,
        frameon=False,
        fontsize=5.2,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    axis.text(
        0.06, 0.97, "MARINER optimistic", transform=axis.transAxes, fontsize=5.6, color="#6b6f78", va="top", ha="left"
    )
    axis.text(
        0.06,
        0.08,
        "MARINER pessimistic",
        transform=axis.transAxes,
        fontsize=5.6,
        color="#6b6f78",
        va="bottom",
        ha="left",
    )
    axis.axhspan(0.0, high, color="#f4f5f7", zorder=0)
    axis.set_ylim(low - 0.015, high)
    axis.set_xlim(-0.01, 0.72)
    axis.set_xlabel("Distance to nearest swarm run (TV)", fontsize=6.4)
    axis.set_ylabel("Measured $-$ predicted (BPB)", fontsize=6.4)
    axis.tick_params(labelsize=6)
    axis.grid(True, color=floor.original.GRID, alpha=0.6, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"extrapolation_residuals.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUTPUT / "extrapolation_residuals.pdf")


if __name__ == "__main__":
    main()
