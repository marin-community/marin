# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "matplotlib", "tabulate"]
# ///
"""Render the frozen offline coupling comparisons without fitting or selecting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path(__file__).parent / "reference_outputs" / "delphi_coupling_followup_20260906"
TARGET_LABELS = {"table9": "Table 9", "uncheatable": "Uncheatable"}
MODEL_LABELS = {
    "weibull_softplus_unscaled": "WSPU",
    "dsp_total_exposure": "Canonical DSP",
    "olmix_loglinear_taskwise": "Taskwise OLMix",
    "fixed_weibull_additive_exp": "Fixed basis, additive",
    "fixed_weibull_coupled_exp": "Fixed basis, coupled",
    "fixed_weibull_interaction_removed": "Fixed basis, interactions removed",
    "wspu_coupling_kappa_0": "WSPU reproduction (κ = 0)",
    "wspu_coupling_kappa_0p25": "WSPU sensitivity (κ = 0.25)",
    "wspu_coupling_kappa_0p5": "WSPU sensitivity (κ = 0.5)",
    "wspu_coupling_kappa_1": "WSPU + coupling (κ = 1)",
}
METRICS = ["regret_at_1", "best_of_5_regret", "best_of_10_regret", "selected_rank", "optimism", "rmse", "spearman"]
MODEL_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#56B4E9"]


def save_figure(figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=180, facecolor="white")
    figure.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def policy_disagreement(output: Path) -> None:
    data = pd.read_csv(output / "continuous_policies" / "matched_policy_comparisons.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    colors = mpl.colormaps["RdYlGn_r"]([0.05, 0.5, 0.95])
    for axis, target in zip(axes, TARGET_LABELS, strict=True):
        for kl, color, marker in zip((0, 0.005, 0.02), colors, ("o", "s", "^"), strict=True):
            rows = data[(data.target == target) & (data.kl_coefficient == kl)].sort_values("cap")
            axis.plot(
                rows.cap,
                rows.wspu_olmix_policy_tv * 100,
                color=color,
                marker=marker,
                markeredgecolor="#333333",
                markeredgewidth=0.7,
                linewidth=2.2,
                label=f"KL coefficient {kl:g}",
                path_effects=[path_effects.Stroke(linewidth=3.0, foreground="#777777"), path_effects.Normal()],
            )
        axis.set_title(TARGET_LABELS[target], loc="left", fontweight="bold")
        axis.set_xticks((4, 6, 8, 16))
        axis.set_xlabel("Cap in frozen exposure units")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Allocation mass moved between model optima (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    figure.suptitle("WSPU and OLMix propose different mixtures under identical constraints", fontsize=13, y=0.98)
    figure.text(
        0.5, 0.895, "Continuous surrogate optima; all 48 proposed policies are unmeasured", ha="center", color="#555555"
    )
    figure.subplots_adjust(left=0.085, right=0.98, top=0.80, bottom=0.26, wspace=0.16)
    save_figure(figure, output, "policy_disagreement")


def metric_tables(output: Path) -> pd.DataFrame:
    sources = [pd.read_csv(output / "coupling" / "metrics.csv")]
    incumbent = output / "incumbent_coupling" / "metrics.csv"
    if incumbent.exists():
        sources.append(pd.read_csv(incumbent))
    data = pd.concat(sources, ignore_index=True).drop_duplicates(
        ["target", "method", "population", "stratum", "policy", "repeat", "fold"]
    )
    selected = data[
        data.population.eq("external_development")
        & data.stratum.eq("optima")
        & data.policy.eq("point")
        & data.method.isin(MODEL_LABELS)
    ].copy()
    selected["model"] = selected.method.map(MODEL_LABELS)
    selected["target_label"] = selected.target.map(TARGET_LABELS)
    selected["rank"] = selected.apply(lambda row: f"{int(row.selected_rank)}/{int(row.rows)}", axis=1)
    selected.to_csv(output / "optima_comparison.csv", index=False)
    columns = ["target_label", "model", *METRICS]
    selected[columns].to_markdown(output / "optima_comparison.md", index=False, floatfmt=".6f")
    panel = data[data.population.eq("panel_oof") & data.method.isin(MODEL_LABELS)].copy()
    panel.to_csv(output / "panel_comparison.csv", index=False)
    return selected


def selection_and_calibration(data: pd.DataFrame, output: Path) -> None:
    names = [
        "weibull_softplus_unscaled",
        "olmix_loglinear_taskwise",
        "dsp_total_exposure",
        "fixed_weibull_additive_exp",
        "fixed_weibull_coupled_exp",
        "wspu_coupling_kappa_1",
    ]
    names = [name for name in names if name in set(data.method)]
    table = data[data.target.eq("table9")].set_index("method").loc[names]
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    y = np.arange(len(table))
    for axis, metric, title in zip(
        axes, ("regret_at_1", "optimism"), ("Selection error", "Selected-point optimism"), strict=True
    ):
        axis.barh(y, table[metric], color=MODEL_COLORS[: len(table)], height=0.64)
        axis.set_yticks(y, [MODEL_LABELS[name] for name in names] if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("BPB (lower regret is better)" if metric == "regret_at_1" else "Observed minus predicted BPB")
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.margins(x=0.28)
        for index, value in enumerate(table[metric]):
            axis.annotate(
                f"{value:.4f}",
                (value, index),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
            )
    figure.suptitle("Better calibration need not select a better Table-9 mixture", fontsize=13, y=0.98)
    figure.text(0.57, 0.88, "157 archived optima; retrospective development evidence", ha="center", color="#555555")
    figure.subplots_adjust(left=0.27, right=0.975, top=0.79, bottom=0.16, wspace=0.14)
    save_figure(figure, output, "selection_and_calibration")


def main() -> None:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    policy_disagreement(args.output_dir)
    selection_and_calibration(metric_tables(args.output_dir), args.output_dir)


if __name__ == "__main__":
    main()
