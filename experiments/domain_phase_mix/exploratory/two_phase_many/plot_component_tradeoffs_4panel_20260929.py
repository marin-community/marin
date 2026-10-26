# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Objective trade-offs at 3e18 FLOPs in four panels: full and subset optima of each objective, MARINER vs Olmix.

Every bar is a BPB change relative to proportional mixing (the eleven proportional Uncheatable runs; the ten proportional
OlmoBaseEval Easy repeats), averaged over the three seed-matched runs of each mixture. Top row: Uncheatable per
component,
for the mixtures optimized for the full objective (left) and for the broad-language subset AO3, BBC News and Wikipedia
(right). Bottom row: OlmoBaseEval Easy by component group, for the full-objective mixtures (left) and the mixtures
optimized for the 15 question-answering components (right). The optimized rows are shaded (the aggregate in the
full-objective panels, the subset in the subset panels), MARINER bars are filled and Olmix bars hatched in the row
colour. The x axis is a symmetric log scale (linear within 0.005 BPB of zero), so the small gains on the optimized
components and the large costs elsewhere are both legible.

Writes `r8_component_tradeoffs_4panel.pdf/.png` and `plotted_values.csv` to
`reference_outputs/component_tradeoffs_4panel_20260929/`.

usage: uv run --offline --no-sync python plot_component_tradeoffs_4panel_20260929.py [--olmix-style hatch|hollow|alpha]
           [--xscale symlog|linear] [--value-labels] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
import plot_component_tradeoffs_obe_20260929 as obe
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from uncheatable_objective import uncheatable_weights

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT = REFERENCE / "component_tradeoffs_4panel_20260929"
PROPORTIONAL_UNCHEATABLE = REFERENCE / "table9_reliability_20260905" / "proportional_uncheatable_components.csv"
UNCHEATABLE_RUNS = {
    "MARINER": (
        (REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908" / "measured_results.csv", "lwspu_u_snc_cap06"),
        (REFERENCE / "delphi_fairness_repeats_3e18_20260908" / "measured_results.csv", "lwspu_u_snc_cap06"),
    ),
    "Olmix": ((REFERENCE / "delphi_matched_olmix_3e18_20260908" / "measured_results.csv", "olmixq_u_kl0p05_cap04"),),
}
UNCHEATABLE_SUBSET_RUNS = {
    "MARINER": (
        (REFERENCE / "delphi_three_component_optimum_3e18_20260908" / "measured_results.csv", "lwspu_w3_snc_cap06"),
    ),
    "Olmix": ((REFERENCE / "delphi_olmix_subset_3e18_20260929" / "measured_results.csv", "olmixq_w3_kl0p05_cap04"),),
}
QA_SUBSET_RUNS = {
    "MARINER": (REFERENCE / "delphi_table9_qa_subset_3e18_20260929" / "measured_table9_components.csv", "qa15_t9_cap04"),
    "Olmix": (
        REFERENCE / "delphi_olmix_subset_3e18_20260929" / "measured_table9_components.csv",
        "olmixq_qa15_kl0p005_cap04",
    ),
}
UNCHEATABLE_COMPONENTS = (
    ("github_cpp", "GitHub C++"),
    ("github_python", "GitHub Python"),
    ("arxiv_physics", "arXiv physics"),
    ("arxiv_computer_science", "arXiv CS"),
    ("ao3_english", "AO3"),
    ("wikipedia_english", "Wikipedia"),
    ("bbc_news", "BBC News"),
)
UNCHEATABLE_SUBSET = ("ao3_english", "wikipedia_english", "bbc_news")
UNCHEATABLE_ALL = "all"
QA_SUBSET_GROUPS = ("QA",)
GROUP_LABELS = {
    "Code": "Code (19)",
    "Math": "Math (7)",
    "Basic Skills": "Basic Skills (6)",
    "MMLU": "MMLU (4)",
    "QA": "QA (15)",
    "All": "All (51)",
}
UNCHEATABLE_GROUPS = {
    "github_cpp": "Code",
    "github_python": "Code",
    "arxiv_physics": "Scientific",
    "arxiv_computer_science": "Scientific",
    "ao3_english": "Broad language",
    "wikipedia_english": "Broad language",
    "bbc_news": "Broad language",
    "all": "All",
}
UNCHEATABLE_COLORS = {**obe.UNCHEATABLE_COLORS, "All": obe.GROUP_COLORS["All"]}
ALPHA = {"MARINER": obe.MARINER_ALPHA, "Olmix": obe.OLMIX_ALPHA}
HIGHLIGHT = "#D6E6F5"
BAR = 0.36
# Symmetric log axis: linear within +-SYMLOG_THRESHOLD BPB, the same log scaling on both sides of zero.
SYMLOG_THRESHOLD = 0.005
SYMLOG_LINSCALE = 0.35
SYMLOG_MAJOR = (0.01, 0.1, 1.0)
SYMLOG_MINOR = (0.03, 0.3)


def uncheatable_runs(sources) -> pd.DataFrame:
    frames = []
    for path, candidate_id in sources:
        table = pd.read_csv(path)
        table = table[table.status.eq("measured") & table.candidate_id.eq(candidate_id)]
        columns = {f"uncheatable_{c}_bpb": c for c, _ in UNCHEATABLE_COMPONENTS}
        frames.append(table[list(columns)].rename(columns=columns).astype(float))
    runs = pd.concat(frames, ignore_index=True)
    if len(runs) != 3:
        raise ValueError(f"expected three runs, found {len(runs)} for {sources[0][1]}")
    runs[UNCHEATABLE_ALL] = uncheatable_aggregate(runs)
    return runs


def uncheatable_aggregate(table: pd.DataFrame) -> pd.Series:
    """The paper's Uncheatable objective: the fixed seven-component weighting."""
    weights = {key.split("/")[2]: weight for key, weight in uncheatable_weights().items()}
    return sum(weight * table[component] for component, weight in weights.items())


def uncheatable_baseline() -> pd.Series:
    table = pd.read_csv(PROPORTIONAL_UNCHEATABLE)
    if len(table) != 11:
        raise ValueError(f"expected eleven proportional runs, found {len(table)}")
    components = pd.DataFrame(
        {c: table[f"eval/uncheatable_eval/{c}/bpb"].astype(float) for c, _ in UNCHEATABLE_COMPONENTS}
    )
    components[UNCHEATABLE_ALL] = uncheatable_aggregate(components)
    return components.mean()


def table9_group_runs(sources) -> pd.DataFrame:
    return pd.DataFrame([obe.group_means(obe.components(path, cid)) for path, cid in sources])


def table9_baseline() -> pd.Series:
    proportional = pd.read_csv(obe.PROPORTIONAL_MATRIX, index_col=0)
    proportional.columns = [c.split("/")[2] for c in proportional.columns]
    return obe.group_means(proportional.mean(axis=0))


def signed_tick(value: float, _position: int) -> str:
    return "0" if abs(value) < 1e-10 else f"{value:+.2f}".replace("-", "\u2212")


def decade_tick(value: float, _position: int) -> str:
    """Shortest label for a symlog decade tick: 0.01, \u22120.1, 1; the sign is the side of the zero line."""
    return f"{value:g}".replace("-", "\u2212")


def style(axis) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#777777")
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.tick_params(axis="x", length=2, pad=2, width=0.6)
    axis.axvline(0, color="#555555", linewidth=0.7, zorder=1)
    axis.xaxis.set_major_formatter(FuncFormatter(signed_tick))
    axis.xaxis.grid(True, linewidth=0.45, color="#DDDDDD", zorder=0)
    axis.invert_yaxis()


def bar_style(mixture: str, colors: list[str], olmix_style: str) -> dict:
    """MARINER bars are filled; Olmix bars are lighter (alpha), outlined (hollow) or hatched in the row colour."""
    if mixture == "MARINER" or olmix_style == "alpha":
        return {"color": colors, "alpha": ALPHA[mixture]}
    if olmix_style == "hollow":
        return {"facecolor": "white", "edgecolor": colors, "linewidth": 0.7}
    if olmix_style == "hatch":
        return {"facecolor": "white", "edgecolor": colors, "linewidth": 0.4, "hatch": "////"}
    raise ValueError(olmix_style)


def draw_panel(
    axis,
    deltas: dict[str, pd.Series | None],
    labels: list[str],
    keys: list[str],
    colors: list[str],
    highlight: tuple[str, ...],
    rows: list[dict],
    panel: str,
    show_labels: bool,
    olmix_style: str,
    value_labels: bool,
) -> None:
    y = np.arange(len(keys), dtype=float)
    for position, key in zip(y, keys, strict=True):
        if key in highlight:
            axis.axhspan(position - 0.5, position + 0.5, color=HIGHLIGHT, zorder=0, linewidth=0)
    for offset, mixture in ((-BAR / 2, "MARINER"), (BAR / 2, "Olmix")):
        series = deltas.get(mixture)
        if series is None:
            continue
        values = series.loc[keys]
        bars = axis.barh(y + offset, values, height=BAR, zorder=2, **bar_style(mixture, colors, olmix_style))
        if mixture == "Olmix" and olmix_style != "alpha":
            # Strokes are centred on the outline, so an outlined bar reads half a line width too long on every side.
            # Doubling the width and clipping each bar to its own rectangle keeps the whole stroke inside the bar.
            for bar in bars:
                bar.set_linewidth(2 * bar.get_linewidth())
                bar.set_clip_path(bar)
        rows.extend({"panel": panel, "mixture": mixture, "item": k, "delta_bpb": float(values[k])} for k in keys)
    if value_labels:
        # One label per highlighted row on the empty side of the zero line: filled square for MARINER, hollow for
        # Olmix, matching the bars; the rows are too tight for one label per bar.
        for position, key in zip(y, keys, strict=True):
            if key not in highlight:
                continue
            parts = []
            for glyph, mixture in (("\u25a0", "MARINER"), ("\u25a1", "Olmix")):
                series = deltas.get(mixture)
                if series is not None:
                    parts.append(f"{glyph} {float(series[key]):+.3f}".replace("-", "\u2212"))
            negative = all(float(deltas[m][key]) < 0 for m in deltas if deltas[m] is not None)
            axis.annotate(
                "  ".join(parts),
                xy=(0.0, position),
                xytext=(3 if negative else -3, 0),
                textcoords="offset points",
                ha="left" if negative else "right",
                va="center",
                fontsize=4.8,
                color="#333333",
                zorder=3,
            )
    axis.set_yticks(y)
    if show_labels:
        axis.set_yticklabels(labels, fontsize=6.3)
    else:
        axis.set_yticklabels([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--olmix-style", choices=("hatch", "hollow", "alpha"), default="hatch")
    parser.add_argument("--xscale", choices=("linear", "symlog"), default="symlog")
    parser.add_argument("--value-labels", action="store_true", help="print the BPB changes beside the subset rows")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.usetex": False,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.5,
        }
    )
    u_base = uncheatable_baseline()
    u_all = {m: uncheatable_runs(s).mean() - u_base for m, s in UNCHEATABLE_RUNS.items()}
    u_sub = {m: uncheatable_runs(s).mean() - u_base for m, s in UNCHEATABLE_SUBSET_RUNS.items() if s[0][0].exists()}
    t_base = table9_baseline()
    t_all = {
        "MARINER": table9_group_runs(obe.MARINER_T9).mean() - t_base,
        "Olmix": table9_group_runs(obe.OLMIX_T9).mean() - t_base,
    }
    t_sub = {
        m: table9_group_runs(tuple((p, f"{cid}@t{s}") for s in (0, 1, 2))).mean() - t_base
        for m, (p, cid) in QA_SUBSET_RUNS.items()
        if p.exists()
    }

    fig = plt.figure(figsize=(5.5, 1.30), facecolor="white")
    rects = ((0.13, 0.24, 0.15, 0.62), (0.31, 0.24, 0.15, 0.62), (0.63, 0.24, 0.15, 0.62), (0.83, 0.24, 0.15, 0.62))
    axes = [fig.add_axes(rect) for rect in rects]
    titles = ("optimized for all", "optimized for broad language", "optimized for all", "optimized for QA")
    rows: list[dict] = []
    u_keys = [c for c, _ in UNCHEATABLE_COMPONENTS] + [UNCHEATABLE_ALL]
    u_labels = [label for _, label in UNCHEATABLE_COMPONENTS] + ["All"]
    u_colors = [UNCHEATABLE_COLORS[UNCHEATABLE_GROUPS[k]] for k in u_keys]
    g_keys = list(obe.GROUPS)
    g_labels = [GROUP_LABELS[g] for g in g_keys]
    g_colors = [obe.GROUP_COLORS[g] for g in g_keys]
    for axis in axes:
        style(axis)
    # The optimized rows are shaded: the aggregate in the full-objective panels, the subset in the subset panels; the
    # label column is shared within each pair.
    style_ = args.olmix_style
    # Values are printed only in the subset panels, where the highlighted bars are too short to read.
    draw_panel(axes[0], u_all, u_labels, u_keys, u_colors, (UNCHEATABLE_ALL,), rows, "A", True, style_, False)
    labels_ = args.value_labels
    draw_panel(axes[1], u_sub, u_labels, u_keys, u_colors, UNCHEATABLE_SUBSET, rows, "B", False, style_, labels_)
    draw_panel(axes[2], t_all, g_labels, g_keys, g_colors, ("All",), rows, "C", True, style_, False)
    draw_panel(axes[3], t_sub, g_labels, g_keys, g_colors, QA_SUBSET_GROUPS, rows, "D", False, style_, labels_)
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, loc="left", fontsize=6.0, pad=2.5)
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelsize=5.5)
        if args.xscale == "symlog":
            axis.set_xscale("symlog", linthresh=SYMLOG_THRESHOLD, linscale=SYMLOG_LINSCALE)
            extent = [patch.get_x() + patch.get_width() for patch in axis.patches if patch.get_height() == BAR]
            low = min(-3 * SYMLOG_THRESHOLD, min(extent) * 1.25)
            high = max(3 * SYMLOG_THRESHOLD, max(extent) * 1.25)
            axis.set_xlim(low, high)
            # Only the decades are labelled; the zero line is drawn and needs no label in these narrow panels.
            major = [s * v for v in SYMLOG_MAJOR for s in (-1, 1) if low <= s * v <= high]
            minor = [s * v for v in SYMLOG_MINOR for s in (-1, 1) if low <= s * v <= high]
            axis.xaxis.set_major_locator(FixedLocator(sorted(major)))
            axis.xaxis.set_minor_locator(FixedLocator(sorted(minor)))
            axis.xaxis.set_major_formatter(FuncFormatter(decade_tick))
            axis.tick_params(axis="x", which="minor", length=1.2, width=0.5)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2, steps=[1, 2, 5, 10]))
    for pair, name in (((0, 1), "Uncheatable"), ((2, 3), "OlmoBaseEval Easy")):
        left = rects[pair[0]][0]
        right = rects[pair[1]][0] + rects[pair[1]][2]
        fig.text((left + right) / 2, 0.975, name, ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    fig.text(
        0.42,
        0.005,
        "\u2190 better      BPB change vs. proportional mixing      worse \u2192",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    fig.legend(
        handles=[
            Patch(label="MARINER", **bar_style("MARINER", "#555555", args.olmix_style)),
            Patch(label="Olmix", **bar_style("Olmix", "#555555", args.olmix_style)),
        ],
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.995, -0.01),
        handlelength=1.0,
        columnspacing=1.2,
    )
    pd.DataFrame(rows).to_csv(output / "plotted_values.csv", index=False)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"r8_component_tradeoffs_4panel.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(output / "r8_component_tradeoffs_4panel.pdf")
    print(pd.DataFrame(rows).pivot(index=["panel", "item"], columns="mixture", values="delta_bpb").round(4).to_string())


if __name__ == "__main__":
    main()
