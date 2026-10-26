# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Objective trade-offs on OlmoBaseEval Easy at 3e18 FLOPs beside the Uncheatable trade-off (alternate 5.2 figure).

Panel A repeats the paper's Uncheatable panel (MARINER's Uncheatable mixture against the eleven proportional runs, per
component, from `component_tradeoffs_20260908/component_changes.csv`). Panel B shows MARINER's and Olmix's OlmoBaseEval
Easy mixtures against the ten proportional repeats by component group (three seed-matched runs each). Panel C shows the
question-answering optima (`delphi_table9_qa_subset_3e18_20260929`) against MARINER's full-objective runs at the same
trainer seeds, by group, with MARINER's predicted changes from `table9_qa_subset_optimum_20260929`; measured bars appear
once the collector has written the launch's component file.

Writes `r8_component_tradeoffs_obe.pdf/.png` and `plotted_values.csv` to
`reference_outputs/component_tradeoffs_obe_20260929/`.

usage: uv run --offline --no-sync python plot_component_tradeoffs_obe_20260929.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
OUTPUT = REFERENCE / "component_tradeoffs_obe_20260929"
UNCHEATABLE_CHANGES = REFERENCE / "component_tradeoffs_20260908" / "component_changes.csv"
PROPORTIONAL_MATRIX = REFERENCE / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
MARINER_T9 = (
    (
        REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908" / "measured_table9_components.csv",
        "lwspu_t9_snc_cap08",
    ),
    (REFERENCE / "delphi_fairness_repeats_3e18_20260908" / "measured_table9_components.csv", "lwspu_t9_snc_cap08@t1"),
    (REFERENCE / "delphi_fairness_repeats_3e18_20260908" / "measured_table9_components.csv", "lwspu_t9_snc_cap08@t2"),
)
OLMIX_T9 = tuple(
    (
        REFERENCE / "delphi_matched_olmix_3e18_20260908" / "measured_table9_components.csv",
        f"olmixq_t9_kl0p005_cap04@t{s}",
    )
    for s in (0, 1, 2)
)
QA_LAUNCH = REFERENCE / "delphi_table9_qa_subset_3e18_20260929" / "measured_table9_components.csv"
QA_PREDICTIONS = REFERENCE / "table9_qa_subset_optimum_20260929" / "component_predictions.csv"
QA_POLICIES = {
    "qa15_t9_cap04": ("table9_qa", "Optimized for QA"),
    "qa19_t9_cap05": ("table9_qa_mmlu", "Optimized for QA and MMLU"),
}
QA = (
    "arc_challenge",
    "arc_easy",
    "csqa",
    "hellaswag",
    "lambada",
    "medmcqa",
    "piqa",
    "sciq",
    "socialiqa",
    "winogrande",
    "coqa",
    "drop",
    "jeopardy",
    "naturalqs",
    "squad",
)
MMLU = ("mmlu_humanities", "mmlu_other", "mmlu_social_sciences", "mmlu_stem")
MATH = tuple(
    f"minerva_math_{s}"
    for s in (
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    )
)
BASIC = tuple(
    f"basic_skills_{s}"
    for s in ("arithmetic", "coding", "common_knowledge", "logical_reasoning", "pattern", "string_operations")
)
GROUPS = ("Code", "Math", "Basic Skills", "MMLU", "QA", "All")
GROUP_COLORS = {
    "Code": "#0072B2",
    "Math": "#CC79A7",
    "Basic Skills": "#E69F00",
    "MMLU": "#56B4E9",
    "QA": "#009E73",
    "All": "#555555",
}
UNCHEATABLE_COLORS = {"Code": "#0072B2", "Scientific": "#CC79A7", "Broad language": "#009E73"}
MARINER_ALPHA, OLMIX_ALPHA = 1.0, 0.45


def group_of(component: str) -> str:
    if component in QA:
        return "QA"
    if component in MMLU:
        return "MMLU"
    if component in MATH:
        return "Math"
    if component in BASIC:
        return "Basic Skills"
    return "Code"


def components(path: Path, candidate_id: str) -> pd.Series:
    table = pd.read_csv(path)
    rows = table[table.candidate_id.eq(candidate_id)].set_index("component")["bpb"].astype(float)
    if len(rows) != 51:
        raise ValueError(f"{candidate_id} in {path.parent.name}: {len(rows)} components")
    return rows


def group_means(values: pd.Series) -> pd.Series:
    groups = pd.Series({c: group_of(c) for c in values.index})
    out = values.groupby(groups).mean()
    out["All"] = values.mean()
    return out.loc[list(GROUPS)]


def panel_b_values() -> pd.DataFrame:
    proportional = pd.read_csv(PROPORTIONAL_MATRIX, index_col=0)
    proportional.columns = [c.split("/")[2] for c in proportional.columns]
    baseline = group_means(proportional.mean(axis=0))
    rows = []
    for label, sources in (("MARINER", MARINER_T9), ("Olmix", OLMIX_T9)):
        per_seed = pd.DataFrame([group_means(components(path, cid)) for path, cid in sources])
        delta = per_seed - baseline
        for group in GROUPS:
            rows.append(
                {
                    "panel": "B",
                    "mixture": label,
                    "group": group,
                    "delta_bpb": float(delta[group].mean()),
                    "delta_sd": float(delta[group].std(ddof=1)),
                    "runs": len(per_seed),
                }
            )
    return pd.DataFrame(rows)


def panel_c_values() -> pd.DataFrame:
    predictions = pd.read_csv(QA_PREDICTIONS).set_index("component")
    reference = pd.DataFrame([group_means(components(path, cid)) for path, cid in MARINER_T9])
    measured = pd.read_csv(QA_LAUNCH) if QA_LAUNCH.exists() else None
    rows = []
    for candidate_id, (column, label) in QA_POLICIES.items():
        predicted = group_means(predictions[column]) - group_means(predictions["full"])
        deltas = None
        if measured is not None:
            runs = []
            for seed in (0, 1, 2):
                cid = f"{candidate_id}@t{seed}"
                if cid in set(measured.candidate_id):
                    runs.append(
                        group_means(measured[measured.candidate_id.eq(cid)].set_index("component")["bpb"].astype(float))
                        - reference.iloc[seed]
                    )
            deltas = pd.DataFrame(runs) if runs else None
        for group in GROUPS:
            rows.append(
                {
                    "panel": "C",
                    "mixture": label,
                    "group": group,
                    "delta_bpb": float(deltas[group].mean()) if deltas is not None else np.nan,
                    "delta_se": float(deltas[group].std(ddof=1) / np.sqrt(len(deltas)))
                    if deltas is not None and len(deltas) > 1
                    else np.nan,
                    "runs": 0 if deltas is None else len(deltas),
                    "predicted_delta_bpb": float(predicted[group]),
                }
            )
    return pd.DataFrame(rows)


def signed_tick(value: float, _position: int) -> str:
    return "0" if abs(value) < 1e-10 else f"{value:+.2f}".replace("-", "\u2212")


def style_axis(axis) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#777777")
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.tick_params(axis="x", length=2, pad=2, width=0.6)
    axis.axvline(0, color="#555555", linewidth=0.7, zorder=1)
    axis.xaxis.set_major_formatter(FuncFormatter(signed_tick))
    axis.xaxis.grid(True, linewidth=0.45, color="#DDDDDD", zorder=0)
    axis.invert_yaxis()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
        }
    )
    uncheatable = pd.read_csv(UNCHEATABLE_CHANGES)
    b = panel_b_values()
    c = panel_c_values()
    pd.concat([b, c], ignore_index=True).to_csv(OUTPUT / "plotted_values.csv", index=False)

    fig = plt.figure(figsize=(5.5, 2.2), facecolor="white")
    axes = [fig.add_axes(rect) for rect in ((0.135, 0.2, 0.185, 0.58), (0.46, 0.2, 0.2, 0.58), (0.79, 0.2, 0.2, 0.58))]
    titles = ("A · Uncheatable mixture", "B · OlmoBaseEval Easy mixtures", "C · Optimized for QA")
    for axis, title, x in zip(axes, titles, (0.01, 0.345, 0.705), strict=True):
        fig.text(x, 0.94, title, fontsize=7.6, fontweight="bold")
        style_axis(axis)
    # A: per-component changes of MARINER's Uncheatable mixture against proportional.
    positions = [0, 1, 2.3, 3.3, 4.6, 5.6, 6.6]
    axes[0].barh(
        positions, uncheatable.delta_bpb, height=0.55, color=[UNCHEATABLE_COLORS[g] for g in uncheatable.group], zorder=2
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(uncheatable.label)
    axes[0].set_xlabel("BPB change vs. proportional")
    # B: group means, MARINER (solid) and Olmix (light), against the proportional repeats.
    y = np.arange(len(GROUPS)) * 1.0
    for offset, mixture, alpha in ((-0.19, "MARINER", MARINER_ALPHA), (0.19, "Olmix", OLMIX_ALPHA)):
        rows = b[b.mixture.eq(mixture)].set_index("group").loc[list(GROUPS)]
        axes[1].barh(
            y + offset, rows.delta_bpb, height=0.36, color=[GROUP_COLORS[g] for g in GROUPS], alpha=alpha, zorder=2
        )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{g} ({n})" for g, n in zip(GROUPS, (19, 7, 6, 4, 15, 51), strict=True)])
    axes[1].set_xlabel("BPB change vs. proportional")
    axes[1].legend(
        handles=[
            Patch(color="#555555", alpha=MARINER_ALPHA, label="MARINER"),
            Patch(color="#555555", alpha=OLMIX_ALPHA, label="Olmix"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=6,
        handlelength=1.0,
        borderaxespad=0.1,
        columnspacing=0.8,
    )
    # C: the QA optima against MARINER's full-objective runs; diamonds are MARINER's predicted changes.
    labels = list(QA_POLICIES.values())
    for offset, (_, label), alpha in zip((-0.19, 0.19), labels, (MARINER_ALPHA, OLMIX_ALPHA), strict=True):
        rows = c[c.mixture.eq(label)].set_index("group").loc[list(GROUPS)]
        if rows.runs.gt(0).any():
            axes[2].barh(
                y + offset,
                rows.delta_bpb,
                xerr=rows.delta_se,
                height=0.36,
                color=[GROUP_COLORS[g] for g in GROUPS],
                alpha=alpha,
                error_kw={"elinewidth": 0.6, "capsize": 1.2},
                zorder=2,
            )
        axes[2].plot(
            rows.predicted_delta_bpb,
            y + offset,
            "D",
            color="#333333",
            markersize=2.6,
            markerfacecolor="white" if alpha < 1 else "#333333",
            markeredgewidth=0.6,
            zorder=4,
        )
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([g for g in GROUPS])
    axes[2].set_xlabel("BPB change vs. MARINER")
    axes[2].legend(
        handles=[
            Patch(color="#555555", alpha=MARINER_ALPHA, label="QA"),
            Patch(color="#555555", alpha=OLMIX_ALPHA, label="QA and MMLU"),
            Line2D([], [], marker="D", color="#333333", linestyle="none", markersize=2.6, label="Predicted"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        frameon=False,
        fontsize=6,
        handlelength=1.0,
        borderaxespad=0.1,
        columnspacing=0.6,
        handletextpad=0.4,
    )
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"r8_component_tradeoffs_obe.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUTPUT / "r8_component_tradeoffs_obe.pdf")
    print(pd.concat([b, c]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
