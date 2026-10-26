# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Appendix learning curves beside the paper's Table 11: MARINER against its ablations, and against the baselines.

Both figures use Figure 3's protocol and metric on the Qwen3 360M/1.6B swarm: ten subset/fold draws at each swarm size,
five mixture-blocked folds per draw. The top row is per-fold selection regret as a fraction of a random pick's (as in
Figure 3: per draw, the fold-averaged regret of each model's pick divided by the fold-averaged regret of a uniformly
random pick; constant fits count as random). The bottom row is the out-of-fold correlation (Spearman by default,
Pearson with ``--correlation pearson``) over the pooled predictions of the five folds. Points are means over draws
with 95% t-intervals. Every model is fitted on the same subsets and folds; the records come from three studies that
share them:

- ``learning_curve_mariner_delphi_3e18_20260908/regmix_oof_20260924``: MARINER, Olmix, quadratic, spline, both RegMix;
- ``learning_curve_pbt_delphi_3e18_20260929``: the per-bucket-threshold ablation;
- ``learning_curve_extra_delphi_3e18_20261006``: the fully convex and additive ablations and kernel ridge.

Models whose study has not been summarized yet are left out and listed in the run's output.

usage: uv run --offline --no-sync python plot_learning_curve_appendix_pair_20261006.py [--correlation spearman|pearson]
           [--drive-dir <paper figures dir>]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_learning_curve_20260905 as plot,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCES = (
    OUTPUTS / "learning_curve_mariner_delphi_3e18_20260908" / "regmix_oof_20260924" / "metrics_long.csv",
    OUTPUTS / "learning_curve_pbt_delphi_3e18_20260929" / "metrics_long.csv",
    OUTPUTS / "learning_curve_extra_delphi_3e18_20261006" / "metrics_long.csv",
)
OUTPUT_DIR = OUTPUTS / "learning_curve_appendix_pair_20261006"

LABELS = {
    "mariner": "MARINER",
    "mariner_cvx": "fully convex objective",
    "mariner_add": "additive form",
    "mariner_pbt": "per-bucket threshold",
    "olmix": "Olmix",
    "quadratic": "Quadratic",
    "spline": "Natural cubic spline",
    "regmix": "RegMix (tuned)",
    "regmix_official": "RegMix (released)",
    "krr": "Hellinger kernel ridge",
}
COLORS = {
    "mariner": plot.MODEL_COLORS["mariner"],
    # Ablation hues, distinct within the ablation figure.
    "mariner_cvx": "#332288",
    "mariner_add": "#999933",
    "mariner_pbt": "#AA4499",
    "olmix": plot.MODEL_COLORS["olmix"],
    "quadratic": plot.MODEL_COLORS["quadratic"],
    # As in the main text: the spline blue (Figure 9), tuned RegMix burnt orange and released RegMix amber
    # (Figures 5 and 13).
    # With quadratic sky blue and kernel ridge wine, all seven pass the palette check (2026-10-13).
    "spline": "#0072B2",
    "regmix": "#B35806",
    "regmix_official": "#E69F00",
    "krr": "#882255",
}
# The spline keeps Figure 9's squares.
MARKERS = {"spline": "s"}
FIGURES = {
    "a_learning_curve_ablations": ("mariner", "mariner_cvx", "mariner_add", "mariner_pbt"),
    "a_learning_curve_baselines": ("mariner", "olmix", "regmix", "regmix_official", "quadratic", "spline", "krr"),
}
TARGETS = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))
RATIO_TICKS = (0.01, 0.03, 0.1, 0.3, 1.0)
CORRELATIONS = {"spearman": "Out-of-fold Spearman", "pearson": "Out-of-fold Pearson"}
X_TICKS = (21, 41, 81, 161, 280)


def load_metrics() -> pd.DataFrame:
    frames = []
    for path in SOURCES:
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    metrics = pd.concat(frames, ignore_index=True)
    oof = metrics[metrics.evaluation.eq("oof")]
    # MARINER appears in every study through the same linked records; keep one copy of each row.
    return oof.drop_duplicates(["target", "model", "k", "draw", "stratum", "fold"])


def interval(values: np.ndarray) -> tuple[float, float, float]:
    center = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return center, center - half, center + half


def summarize(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per target, model and swarm size: the regret ratio and the pooled out-of-fold Spearman."""
    folds = oof[oof.stratum.eq("fold")].assign(constant=lambda d: d.spearman.isna())
    tied = folds.groupby(["target", "model", "k"]).constant.all()
    means = oof[oof.stratum.eq("fold_mean")]
    pooled = oof[oof.stratum.eq("pooled")]
    rows = []
    for (target, model, k), group in means.groupby(["target", "model", "k"]):
        ratio = (group.regret_at_1 / group.random_regret_at_1).to_numpy(float)
        constant = bool(tied.loc[(target, model, k)])
        if constant:
            ratio = np.ones_like(ratio)  # constant fits pick arbitrarily: the random pick's expectation
        center, low, high = interval(ratio)
        cell = pooled[pooled.target.eq(target) & pooled.model.eq(model) & pooled.k.eq(k)]
        correlations = {}
        for name in CORRELATIONS:
            values = cell[name].dropna().to_numpy(float)
            correlations[name] = interval(values) if len(values) > 1 and not constant else (np.nan,) * 3
        rows.append(
            {
                "target": target,
                "model": model,
                "k": int(k),
                "mixtures": int(k) + 1,
                "draws": len(group),
                "ratio": center,
                "ratio_low": max(low, 1e-3),
                "ratio_high": high,
                **{
                    f"{name}{suffix}": value
                    for name, triple in correlations.items()
                    for suffix, value in zip(("", "_low", "_high"), triple, strict=True)
                },
                "constant": constant,
            }
        )
    return pd.DataFrame(rows)


def draw_series(axis, part: pd.DataFrame, model: str, column: str) -> None:
    color = COLORS[model]
    shown = part[~part.constant]
    axis.fill_between(
        shown.mixtures, shown[f"{column}_low"], shown[f"{column}_high"], color=color, alpha=0.14, linewidth=0, zorder=2
    )
    axis.plot(
        shown.mixtures,
        shown[column],
        color=color,
        linewidth=1.5 if model == "mariner" else 1.1,
        marker=MARKERS.get(model, "o"),
        markersize=2.4,
        markerfacecolor=plot.PAPER,
        markeredgecolor=color,
        zorder=5 if model == "mariner" else 4,
        label=LABELS[model],
    )
    constant = part[part.constant]
    if column == "ratio" and len(constant):
        # Constant fits sit at the random pick, as in Figure 3 (hollow diamonds joined by a dotted line).
        axis.plot(
            [*constant.mixtures, *shown.mixtures.iloc[:1]],
            [*constant.ratio, *shown.ratio.iloc[:1]],
            linestyle=":",
            color=color,
            linewidth=1.0,
            zorder=4,
        )
        axis.plot(
            constant.mixtures,
            constant.ratio,
            linestyle="none",
            marker="D",
            markersize=3.0,
            markerfacecolor=plot.PAPER,
            markeredgecolor=color,
            markeredgewidth=0.9,
            zorder=5,
        )


def build_figure(summary: pd.DataFrame, models: tuple[str, ...], correlation: str):
    figure, axes = plot.plt.subplots(2, 2, figsize=(5.5, 2.95), sharex=True)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.135, top=0.825, wspace=0.28, hspace=0.21)
    for column_index, (target, title) in enumerate(TARGETS):
        ratio_axis, rho_axis = axes[0, column_index], axes[1, column_index]
        ratio_axis.axhline(1.0, color=plot.RANDOM_COLOR, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1)
        ratio_axis.annotate(
            "Random pick",
            xy=(0.98, 1.0),
            xycoords=("axes fraction", "data"),
            xytext=(0, -3),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.0,
            color=plot.RANDOM_COLOR,
        )
        for model in models:
            part = summary[summary.target.eq(target) & summary.model.eq(model)].sort_values("k")
            if part.empty:
                continue
            draw_series(ratio_axis, part, model, "ratio")
            draw_series(rho_axis, part, model, correlation)
        ratio_axis.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
        ratio_axis.set_yscale("log")
        lows = summary[summary.model.isin(models) & summary.target.eq(target) & ~summary.constant].ratio_low
        ratio_axis.set_ylim(max(float(lows.min()), 0.008) / 1.3, 1.6)
        ratio_axis.set_yticks([t for t in RATIO_TICKS if ratio_axis.get_ylim()[0] <= t <= ratio_axis.get_ylim()[1]])
        ratio_axis.set_yticklabels([f"{t:g}" for t in ratio_axis.get_yticks()])
        for axis in (ratio_axis, rho_axis):
            axis.set_xscale("log")
            axis.minorticks_off()
            axis.set_xticks(X_TICKS)
            axis.set_xticklabels([str(t) for t in X_TICKS])
            axis.tick_params(labelsize=6.3, length=2.5, width=0.6)
            axis.grid(axis="y", color=plot.GRID, linewidth=0.5, alpha=0.72, linestyle=(0, (4, 2)))
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        rho_axis.set_ylim(top=1.0)
    axes[0, 0].set_ylabel("Regret / random pick", fontsize=6.8, labelpad=3)
    axes[1, 0].set_ylabel(CORRELATIONS[correlation], fontsize=6.8, labelpad=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
        ncol=4,
        frameon=False,
        fontsize=6.6,
        handlelength=1.6,
        columnspacing=1.2,
    )
    figure.text(0.54, 0.02, "Distinct mixtures (including anchor)", ha="center", fontsize=6.8)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correlation", choices=tuple(CORRELATIONS), default="spearman")
    parser.add_argument("--drive-dir", type=Path, help="copy the PDFs and PNGs into the paper's figures directory")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot.plt.rcParams.update(plot.PLOT_STYLE)
    summary = summarize(load_metrics())
    summary.to_csv(OUTPUT_DIR / "learning_curve_appendix_pair_summary.csv", index=False)
    for stem, models in FIGURES.items():
        missing = [model for model in models if summary.model.eq(model).sum() == 0]
        if missing:
            print(f"{stem}: not yet summarized, left out: {', '.join(missing)}")
        figure = build_figure(summary, models, args.correlation)
        name = stem if args.correlation == "spearman" else f"{stem}_{args.correlation}"
        for extension in ("png", "pdf"):
            path = OUTPUT_DIR / f"{name}.{extension}"
            figure.savefig(path, dpi=plot.DPI)
            if args.drive_dir:
                shutil.copyfile(path, args.drive_dir / path.name)
        plot.plt.close(figure)
        print(OUTPUT_DIR / f"{name}.pdf")


if __name__ == "__main__":
    main()
