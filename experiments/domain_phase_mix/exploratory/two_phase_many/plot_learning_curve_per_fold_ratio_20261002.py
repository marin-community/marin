# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Figure 3 of the one-phase paper (fig:r6): per-fold selection regret as a fraction of a random pick's.

Within each of the five mixture-blocked folds of a k-run subset, each model selects the run with its lowest
out-of-fold prediction; per draw, the fold-averaged regret of that pick is divided by the fold-averaged regret of a
uniformly random pick (`random_regret_at_1`). Points are means over the ten subset/fold draws with 95% t-intervals.
Released RegMix's fits are constant in every fold up to k = 60, so those sizes sit at the random pick (hollow
diamonds). Reads the out-of-fold records of `regmix_oof_20260924` (MARINER, Olmix, tuned and released RegMix, and the
natural cubic spline on the same subsets and folds); fits nothing.

usage: LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 uv run --offline --no-sync --with lightgbm==4.7.0 \
           python plot_learning_curve_per_fold_ratio_20261002.py [--drive-dir <paper figures dir>]
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
    learning_curve_mariner_fits_20260908 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_learning_curve_20260905 as plot,
)

STEM = "r6_learning_curve_per_fold_ratio"
SHOWN = ("mariner", "olmix", "regmix", "regmix_official", "spline")
LABELS = {
    "mariner": "MARINER",
    "olmix": "Olmix",
    "regmix": "RegMix (tuned)",
    "regmix_official": "RegMix (released)",
    "spline": "Cubic spline",
}
COLORS = {
    "mariner": plot.MODEL_COLORS["mariner"],
    "olmix": plot.MODEL_COLORS["olmix"],
    # Tuned RegMix in burnt orange and released RegMix in lighter amber, as in the paper's other figures (2026-10-13).
    "regmix": "#B35806",
    "regmix_official": "#E69F00",
    # The spline in Figure 7's blue, with its squares (2026-10-26).
    "spline": "#0072B2",
}
MARKERS = {"spline": "s"}
TITLES = {"uncheatable": "A · Uncheatable", "table9": "B · OlmoBaseEval Easy"}
Y_TICKS = (0.01, 0.03, 0.1, 0.3, 1.0)


def summary() -> pd.DataFrame:
    m = pd.read_csv(fits.OUTPUT_DIR / "regmix_oof_20260924" / "metrics_long.csv")
    oof = m[m.evaluation.eq("oof")]
    folds = oof[oof.stratum.eq("fold")].assign(constant=lambda d: d.spearman.isna())
    tied = folds.groupby(["target", "model", "k"]).constant.all()
    means = oof[oof.stratum.eq("fold_mean")]
    rows = []
    for (target, model, k), group in means.groupby(["target", "model", "k"]):
        ratio = (group.regret_at_1 / group.random_regret_at_1).to_numpy(float)
        if tied.loc[(target, model, k)]:
            ratio = np.ones_like(ratio)  # constant fits pick arbitrarily: the random pick's expectation
        n = len(ratio)
        # Arithmetic mean over draws: a draw whose picks are every fold's best has ratio 0, which a log mean cannot hold.
        half = stats.t.ppf(0.975, n - 1) * ratio.std(ddof=1) / np.sqrt(n)
        center = ratio.mean()
        rows.append(
            {
                "target": target,
                "model": model,
                "k": int(k),
                "mixtures": int(k) + 1,
                "center": center,
                "low": max(center - half, 1e-3),
                "high": center + half,
                "tied": bool(tied.loc[(target, model, k)]),
            }
        )
    return pd.DataFrame(rows)


HEIGHT = 1.8  # inches; the paper's Figure 5 was drawn at this height until 2026-10-11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-dir", type=Path, help="copy the PDF and PNG into the paper's figures directory")
    parser.add_argument("--height", type=float, default=HEIGHT, help="canvas height in inches; margins stay fixed")
    args = parser.parse_args()
    output = fits.OUTPUT_DIR / "per_fold_ratio_20261002"
    output.mkdir(parents=True, exist_ok=True)
    plot.plt.rcParams.update(plot.PLOT_STYLE)
    frame = summary()
    frame.to_csv(output / "per_fold_regret_ratio_summary.csv", index=False)
    height = args.height
    figure, axes = plot.plt.subplots(1, 2, figsize=(5.5, height))
    # Margins in inches (the 1.8 in layout's), so a taller canvas only grows the plotting areas.
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.45 / height, top=1 - 0.378 / height, wspace=0.32)
    lows = []
    for axis, target in zip(axes, ("uncheatable", "table9"), strict=True):
        axis.axhline(1.0, color=plot.RANDOM_COLOR, linestyle=(0, (4, 2)), linewidth=0.9, zorder=1)
        axis.annotate(
            "Random pick",
            xy=(0.98, 1.0),
            xycoords=("axes fraction", "data"),
            xytext=(0, -3),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.3,
            color=plot.RANDOM_COLOR,
        )
        for model in SHOWN:
            part = frame[frame.target.eq(target) & frame.model.eq(model)].sort_values("k")
            ranked, tied = part[~part.tied], part[part.tied]
            color = COLORS[model]
            axis.fill_between(ranked.mixtures, ranked.low, ranked.high, color=color, alpha=0.16, linewidth=0, zorder=2)
            axis.plot(
                ranked.mixtures,
                ranked.center,
                color=color,
                linewidth=1.3,
                marker=MARKERS.get(model, "o"),
                markersize=2.4 if model in MARKERS else 2.6,
                markerfacecolor=plot.PAPER,
                markeredgecolor=color,
                zorder=4,
                label=LABELS[model],
            )
            if len(tied):
                axis.plot(
                    [*tied.mixtures, *ranked.mixtures.iloc[:1]],
                    [*tied.center, *ranked.center.iloc[:1]],
                    linestyle=":",
                    color=color,
                    linewidth=1.0,
                    zorder=4,
                )
                axis.plot(
                    tied.mixtures,
                    tied.center,
                    linestyle="none",
                    marker="D",
                    markersize=3.2,
                    markerfacecolor=plot.PAPER,
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    zorder=5,
                )
            lows.append(ranked.low.min())
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.minorticks_off()
        axis.set_xticks([21, 41, 81, 161, 280])
        axis.set_xticklabels(["21", "41", "81", "161", "280"])
        axis.set_title(TITLES[target], loc="left", fontsize=8, fontweight="bold", pad=4)
        axis.tick_params(labelsize=6.5, length=2.5, width=0.6)
        axis.set_ylabel("Regret / random pick", fontsize=7, labelpad=3)
        axis.grid(axis="y", color=plot.GRID, linewidth=0.5, alpha=0.72, linestyle=(0, (4, 2)))
    for axis in axes:
        axis.set_ylim(max(min(lows), 0.008) / 1.3, 1.6)
        axis.set_yticks([t for t in Y_TICKS if axis.get_ylim()[0] <= t <= axis.get_ylim()[1]])
        axis.set_yticklabels([f"{t:g}" for t in axis.get_yticks()])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1 + 0.027 / height),
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        columnspacing=1.1,
    )
    figure.text(0.53, 0.081 / height, "Distinct mixtures (including anchor)", ha="center", fontsize=7)
    stem = STEM if height == HEIGHT else f"{STEM}_h{height:g}"
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        figure.savefig(path, dpi=plot.DPI)
        if args.drive_dir is not None:
            shutil.copyfile(path, args.drive_dir / path.name)
    print(output / f"{stem}.pdf")
    print(
        frame[frame.k.isin([20, 40, 80, 160, 279]) & frame.model.isin(["mariner", "regmix", "olmix", "spline"])]
        .pivot_table(index=["target", "k"], columns="model", values="center")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
