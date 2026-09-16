# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Complexity ladder: out-of-fold rank accuracy and retrospective-bank regret against nominal parameters per task.

Reads the 2026-09-08 comparator certify run (and any addendum directories given with --extra) and draws two rows of
panels: out-of-fold Spearman on the three swarms and bank regret at one on the Qwen3 3e18 bank, for Uncheatable and
OlmoBaseEval Easy, with models ordered by nominal response parameters per task at M = 39 buckets.

usage: uv run --offline --no-sync python plot_complexity_ladder_20260909.py [--extra DIR ...] [--drive-dir DIR]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN = SCRIPT_DIR / "reference_outputs" / "single_phase_observatory_comparators_20260908"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "complexity_ladder_20260909"
F = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
M = 39
# (model id, label, nominal parameters per task, family colour key)
LADDER = (
    ("linear_weight", "Linear in weights", M + 1, "baseline"),
    ("olmix_loglinear_taskwise", "Olmix", M + 1, "baseline"),
    ("olmix_loglinear_taskwise_log_epoch", "Olmix on log-epochs", M + 1, "baseline"),
    ("linear_log_epoch@kappa_floor_link_flat15_nocap", "Linear in log-epochs, floor link", M + 2, "comparator"),
    (f"{F}_benefit_only_kappa1", "Exponential benefit only", M + 3, "simplification"),
    (f"{F}_single_harm", "MARINER, one harm amplitude", M + 6, "simplification"),
    ("quadratic_log_epoch", "Quadratic in log-epochs, identity link", 2 * M + 1, "comparator"),
    (
        "quadratic_log_epoch@kappa_floor_link_flat15_nocap",
        "Quadratic in log-epochs, floor link",
        2 * M + 2,
        "comparator",
    ),
    (f"{F}_fixed_shape", "MARINER, one shape for all tasks", 2 * M + 2, "simplification"),
    (f"{F}_fixed_ridge", "MARINER, fixed ridge", 2 * M + 5, "simplification"),
    (f"{F}_kappa1_hinge", "MARINER, exponential benefit and hinge harm", 2 * M + 4, "simplification"),
    (f"{F}_kappa1", "MARINER, exponential benefit", 2 * M + 4, "simplification"),
    (f"{F}_hinge_harm", "MARINER, hinge harm", 2 * M + 5, "simplification"),
    (F, "MARINER", 2 * M + 5, "mariner"),
    ("spline_log_epoch@kappa_floor_link_flat15_nocap", "Natural cubic spline in log-epochs, floor link", 4 * M + 2, "comparator"),
    ("lightgbm_regmix", "LightGBM (RegMix)", 1000, "nonparametric"),
    ("hellinger_krr", "Hellinger kernel ridge", 1000, "nonparametric"),
    ("mlp_weights", "MLP", 1000, "nonparametric"),
)
COLORS = {
    "baseline": "#6C6F7D",
    "comparator": "#CC79A7",
    "simplification": "#4C78A8",
    "mariner": "#469C76",
    "nonparametric": "#E69F00",
}
PANELS = (
    ("delphi_3e18_39bucket", "Qwen3 360M/1.6B"),
    ("300m_39bucket", "Llama 200M/6B"),
    ("60m_39bucket", "Llama 160M/1.2B"),
)
TARGETS = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))


def load(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.concat([pd.read_csv(d / "aggregate_metrics.csv") for d in dirs if (d / "aggregate_metrics.csv").exists()])
    held = pd.concat(
        [
            pd.read_csv(d / "external_heldout_selection_metrics.csv")
            for d in dirs
            if (d / "external_heldout_selection_metrics.csv").exists()
        ]
    )
    agg = agg.drop_duplicates(subset=["model", "panel", "target"], keep="last")
    held = held[(held.stratum == "pooled") & (held.panel == "delphi_3e18_39bucket")].drop_duplicates(
        subset=["model", "target"], keep="last"
    )
    return agg, held


def build(agg: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, label, params, family in LADDER:
        row = {"model": model, "label": label, "params": params, "family": family}
        for panel, _ in PANELS:
            for target, _ in TARGETS:
                sel = agg[(agg.model == model) & (agg.panel == panel) & (agg.target == target)]
                row[f"spearman:{panel}:{target}"] = float(sel.spearman.iloc[0]) if len(sel) else np.nan
        for target, _ in TARGETS:
            sel = held[(held.model == model) & (held.target == target)]
            row[f"regret:{target}"] = float(sel.regret_at_1.iloc[0]) if len(sel) else np.nan
            row[f"optimism:{target}"] = float(sel.selection_optimism.iloc[0]) if len(sel) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame) -> plt.Figure:
    table = table[table.filter(like="spearman:").notna().any(axis=1)].reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    x = np.arange(len(table))
    markers = {"delphi_3e18_39bucket": "o", "300m_39bucket": "s", "60m_39bucket": "^"}
    for col, (target, tlabel) in enumerate(TARGETS):
        ax = axes[0, col]
        for panel, plabel in PANELS:
            y = table[f"spearman:{panel}:{target}"]
            ax.plot(x, y, linestyle="-", linewidth=0.8, color="#b8b8b8", zorder=1)
            ax.scatter(
                x,
                y,
                marker=markers[panel],
                s=34,
                c=[COLORS[f] for f in table.family],
                edgecolor="black",
                linewidth=0.4,
                label=plabel,
                zorder=3,
            )
        ref = float(table.loc[table.model == F, f"spearman:delphi_3e18_39bucket:{target}"].iloc[0])
        ax.axhline(ref, color=COLORS["mariner"], linewidth=0.8, linestyle="--")
        ax.set_title(f"{tlabel}: out-of-fold Spearman", fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax = axes[1, col]
        y = table[f"regret:{target}"]
        ax.bar(x, y, color=[COLORS[f] for f in table.family], edgecolor="black", linewidth=0.4)
        for xi, (r, o) in enumerate(zip(y, table[f"optimism:{target}"], strict=True)):
            if np.isfinite(o):
                ax.text(
                    xi,
                    (r if np.isfinite(r) else 0) + 0.002,
                    f"{o:+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="#444444",
                )
        ax.set_title(f"{tlabel}: retrospective-bank regret at one (labels: optimism, BPB)", fontsize=10)
        ax.set_ylim(0, min(0.2, max(0.03, np.nanmax(y) * 1.4)))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lbl}\n({p if p < 1000 else 'nonparametric'})" for lbl, p in zip(table.label, table.params, strict=True)],
            rotation=90,
            fontsize=7,
        )
        ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker=m, linestyle="none", color="gray", markeredgecolor="black") for m in ("o", "s", "^")
    ]
    axes[0, 0].legend(handles, [p for _, p in PANELS], fontsize=8, loc="lower right")
    fig.suptitle(
        "Complexity ladder: models ordered by nominal response parameters per task (M = 39); "
        "dashed line = MARINER at Qwen3 3e18",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    agg, held = load([MAIN, *args.extra])
    table = build(agg, held)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.output_dir / "complexity_ladder.csv", index=False)
    fig = draw(table)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"complexity_ladder.{suffix}", dpi=170)
        if args.drive_dir is not None:
            shutil.copyfile(
                args.output_dir / f"complexity_ladder.{suffix}", args.drive_dir / f"a_complexity_ladder.{suffix}"
            )
    print(args.output_dir / "complexity_ladder.png")


if __name__ == "__main__":
    main()
