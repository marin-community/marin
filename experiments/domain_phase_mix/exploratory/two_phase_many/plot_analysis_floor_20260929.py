# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Figure 13 of the Analysis: ordering of the k best held-out optima on each objective (A, B) and on one component of
each objective at matched SNR (C, D), in one row.

Panels A and B repeat `plot_analysis_floor_20260928.py` (MARINER's pairwise sign accuracy among the k best-measured
held-out optima with its source-block bootstrap interval, the oracle under run noise and the neighbour kernel). Panels
C and D draw the same curves for arXiv physics (Uncheatable) and SciQ (OlmoBaseEval Easy) from
`analyze_heldout_component_ordering_20260929.py`'s table. Olmix's curves were previewed and dropped as too busy
(2026-09-29); its numbers stay in the appendix table.

Writes `analysis_floor_4panel.pdf/.png` to `reference_outputs/analysis_floor_figure_20260928/`.

usage: uv run --offline --no-sync python plot_analysis_floor_20260929.py
"""

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
import plot_analysis_floor_20260928 as floor
from matplotlib.lines import Line2D

OUTPUT = floor.OUTPUT
COMPONENTS = floor.REFERENCE / "heldout_table9_components_20260929" / "component_ordering.csv"
REGMIX = floor.REFERENCE / "delphi_top_band_ordering_20260928" / "regmix_band_ordering.csv"
REGMIX_COLOR = "#0072B2"
PANELS = (
    ("uncheatable", None, "A. Uncheatable"),
    ("table9", None, "B. OlmoBaseEval Easy"),
    ("uncheatable", "arxiv_physics", "C. arXiv physics"),
    ("table9", "sciq", "D. SciQ"),
)


def component_bands(components: pd.DataFrame, target: str, component: str) -> pd.DataFrame:
    """The component's ordering rows in the band table's schema (MARINER with its interval, and the kernel)."""
    rows = components[components.target.eq(target) & components.component.eq(component)].sort_values("band")
    mariner = rows.assign(method="MARINER")
    kernel = rows.assign(method=floor.KERNEL, sign_accuracy=rows.kernel_sign_accuracy)
    return pd.concat([mariner, kernel], ignore_index=True)


def regmix_rows(regmix: pd.DataFrame, target: str, component: str | None) -> pd.DataFrame:
    rows = regmix[regmix.target.eq(target)]
    rows = rows[rows.component.isna()] if component is None else rows[rows.component.eq(component)]
    return rows.sort_values("band")


def draw_regmix(axis, rows: pd.DataFrame) -> None:
    x = rows.band.to_numpy(float)
    axis.fill_between(
        x, rows.sign_accuracy_ci_low, rows.sign_accuracy_ci_high, color=REGMIX_COLOR, alpha=0.14, linewidth=0, zorder=2
    )
    axis.plot(x, rows.sign_accuracy, color=REGMIX_COLOR, linewidth=1.0, marker="s", markersize=2.2, zorder=3)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bands = pd.read_csv(floor.BANDS)
    components = pd.read_csv(COMPONENTS)
    regmix = pd.read_csv(REGMIX) if REGMIX.exists() else None
    plt.rcParams.update(floor.original.PLOT_STYLE)
    plt.rcParams.update({"text.usetex": False})
    figure = plt.figure(figsize=(5.5, 1.7), facecolor="white")
    axes = [figure.add_axes((x, 0.2, 0.17, 0.6)) for x in (0.075, 0.305, 0.535, 0.765)]
    for axis, (target, component, title) in zip(axes, PANELS, strict=True):
        table = bands if component is None else component_bands(components, target, component)
        floor.draw_bands(axis, table, target, "")
        if regmix is not None:
            draw_regmix(axis, regmix_rows(regmix, target, component))
        axis.set_title(f"{title}\nbest $k$ held-out", loc="left", fontsize=6.6, fontweight="bold", linespacing=1.1)
        axis.set_xlabel("best $k$ optima", fontsize=6.8)
        # Thin the crowded log-axis tick labels at this panel width.
        labels = [tick.get_text() for tick in axis.get_xticklabels()]
        axis.set_xticklabels([label if label not in ("20", "50") else "" for label in labels], fontsize=6.5)
        if component is not None:
            snr = float(table.snr.iloc[0])
            axis.text(0.96, 0.05, f"SNR {snr:.0f}", transform=axis.transAxes, fontsize=6.2, ha="right", va="bottom")
    axes[0].set_ylabel("Sign accuracy", fontsize=6.8)
    for axis in axes[1:]:
        axis.set_ylabel("")
        axis.set_yticklabels([])
    handles = [
        Line2D([], [], color=floor.original.MARINER_COLOR, marker="o", markersize=3, linewidth=1.4, label="MARINER"),
        Line2D([], [], color=REGMIX_COLOR, marker="s", markersize=2.2, linewidth=1.0, label="RegMix"),
        Line2D([], [], color=floor.MUTED, marker="o", markersize=2.2, linewidth=0.9, label="Neighbour kernel"),
        Line2D([], [], color=floor.CEILING, linewidth=1.0, linestyle=(0, (3, 2)), label="Oracle under noise"),
    ]
    axes[0].legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.25,
        fontsize=5.0,
        handlelength=1.3,
        handletextpad=0.4,
        borderaxespad=0.1,
        labelspacing=0.25,
        bbox_to_anchor=(1.0, 0.07),
    )
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"analysis_floor_4panel.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUTPUT / "analysis_floor_4panel.pdf")


if __name__ == "__main__":
    main()
