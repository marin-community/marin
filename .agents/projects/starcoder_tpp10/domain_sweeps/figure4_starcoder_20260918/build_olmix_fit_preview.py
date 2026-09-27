# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""Two-panel preview: Olmix's log-linear law fitted to each proxy sweep, against the measured target tracks.

Left: the law fitted to the proxy without simulated epoching (full parent pool, under one domain epoch).
Right: the law fitted to the simulated-epoching proxy. Each panel shows the two target tracks of Figure 4,
the proxy points the law was fitted to, the fitted curve, and the law's optimum over the weight range.
The law is exp(log c) + exp(w . coef) with w = (p, 1 - p), so it is monotone in p and its optimum is a
boundary. Run from the repository root:

    PYTHONPATH=. uv run --offline --no-sync python <this file>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(DIRECTORY.parent / "figure4_alternatives_20260913"))
import build_two_domain_preview as two  # noqa: E402
import plot_alternatives as base  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_starcoder_tpp10_mariner_20260911 as fitting  # noqa: E402
from experiments.domain_phase_mix.olmix_loglinear_fit import fit_olmix_loglinear_model  # noqa: E402

PANELS = (("unmatched", "{name} fit: proxy without simulated epoching"), ("matched", "{name} fit: simulated-epoching proxy"))
SURROGATE_NAMES = {"olmix": "Olmix", "mariner": "MARINER"}
FIT_COLOR_ALPHA = 0.55
OPTIMUM_MARKER = "X"


def fit_law(points):
    """Olmix's positive log-linear law on the eight grid points, weights (p, 1 - p)."""
    p = np.array([pt["percent"] / 100 for pt in points])
    y = np.array([pt["bpb"] for pt in points])
    fit = fit_olmix_loglinear_model(np.stack([p, 1 - p], axis=1), y)
    grid = np.linspace(0, 1, 401)
    curve = fit.predict(np.stack([grid, 1 - grid], axis=1))
    best = int(np.argmin(curve))
    return {"grid": grid, "curve": curve, "optimum_weight": float(grid[best]), "optimum_loss": float(curve[best]),
            "log_c": fit.log_c, "coefficients": fit.coefficients, "huber_loss": fit.huber_loss}


def fit_mariner(points, domain, arm, benchmark, design):
    """The frozen MARINER two-bucket fit of fit_starcoder_tpp10_mariner_20260911 on the proxy sweep; share = weight."""
    share = np.array([pt["percent"] / 100 for pt in points], dtype=float)
    y = np.array([pt["bpb"] for pt in points])
    metric = f"eval/{benchmark}/bpb"
    result = fitting.fit_curve(fitting.Curve(f"{domain}_{arm}_{benchmark}", arm, y, "descriptive_single_seed"), share, design, metric)
    grid = np.array(result["dense_share"]); curve = np.array(result["dense_prediction"])
    return {"grid": grid, "curve": curve, "optimum_weight": float(result["predicted_minimum_share"]), "optimum_loss": float(result["predicted_minimum_bpb"]),
            "rmse": result["fit_rmse_bpb"], "shape": result["shape"], "ridge": result["ridge"]}


def draw_panel(ax, curves, rows, arm, title, full, surrogate, design):
    """``full``: target-scale epochs per unit weight, the shared x scale of every track."""
    optima = {}
    for domain, benchmark, label in rows:
        color = base.COLORS[domain]
        # target track with its measured minimum
        tx, ty = two.measured(curves[domain, "target", benchmark])
        ax.plot(tx, ty, color=color, ls="--", marker="s", lw=1.3, ms=3.0, mfc="white", mew=0.7, zorder=2)
        i = int(np.argmin(ty))
        ax.scatter(tx[i], ty[i], marker="*", s=85, color=color, edgecolor="#222222", lw=0.55, zorder=4)
        # proxy points the law is fitted to, at the same weight-to-epoch scale
        points = curves[domain, arm, benchmark]
        px = np.array([pt["percent"] / 100 * full for pt in points])
        py = np.array([pt["bpb"] for pt in points])
        line, marker, _ = two.ARM_STYLES[arm]
        ax.plot(px, py, color=color, ls="none", marker=marker, ms=3.4, mfc=color if arm == "matched" else "white", mew=0.7, zorder=3)
        # the fitted law and its optimum
        law = fit_law(points) if surrogate == "olmix" else fit_mariner(points, domain, arm, benchmark, design)
        ax.plot(law["grid"] * full, law["curve"], color=color, lw=1.6, alpha=FIT_COLOR_ALPHA, zorder=2)
        xo, yo = law["optimum_weight"] * full, law["optimum_loss"]
        ax.plot([xo], [yo], marker=OPTIMUM_MARKER, ms=8, color=color, markeredgecolor="#222222", markeredgewidth=0.6, ls="none", zorder=6)
        ax.annotate(f"{SURROGATE_NAMES[surrogate]} optimum: p = {law['optimum_weight'] * 100:.0f}%, {xo:.1f} ep.", (xo, yo), xytext=(0, -12 if law["optimum_weight"] > 0.5 else 10), textcoords="offset points",
                    ha="right" if law["optimum_weight"] > 0.5 else "left", va="top" if law["optimum_weight"] > 0.5 else "bottom", fontsize=6.6, color=color,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=7)
        optima[domain] = {"arm": arm, "optimum_weight": law["optimum_weight"], "optimum_epochs": xo, **{k: v for k, v in law.items() if k not in ("grid", "curve")}}
    ax.set_xlim(-0.2, 16.4)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Materialized domain epochs (target scale)", fontsize=8)
    ax.set_title(title, fontsize=8.2, weight="bold", pad=6)
    return optima


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flan-eval", default="sciq_bpb", choices=two.FLAN_EVALS)
    parser.add_argument("--finemath-eval", default="gsm8k")
    parser.add_argument("--figsize", default="7.2,3.7")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--surrogate", choices=sorted(SURROGATE_NAMES), default="olmix")
    args = parser.parse_args()
    stem = args.stem or f"figure4_{args.surrogate}_fits_preview"
    name = SURROGATE_NAMES[args.surrogate]
    rows = [(two.FLAN, args.flan_eval, "Dolmino FLAN"), ("finemath_3plus", args.finemath_eval, "FineMath-3+")]
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.unicode_minus": False})
    curves = base.load_curves()
    curves.update(two.load_flan_curves())
    base.COLORS[two.FLAN] = two.FLAN_COLOR
    design = json.loads(base.DESIGN.read_text())
    two.load_unmatched_curves(curves, design, rows)
    full = curves[rows[0][0], "target", rows[0][1]][-1]["epochs"] / 100 * 100  # epochs at p = 100%
    figsize = tuple(float(v) for v in args.figsize.split(","))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    optima = {}
    for ax, (arm, title) in zip(axes, PANELS, strict=True):
        optima[arm] = draw_panel(ax, curves, rows, arm, title.format(name=name), full, args.surrogate, design)
    axes[0].set_ylabel("Evaluation loss (BPB)", fontsize=8)
    handles = [Line2D([], [], color=base.COLORS[d], lw=2, label=label) for d, _, label in rows]
    handles += [
        Line2D([], [], color="#444444", lw=1.3, marker="s", ms=3, mfc="white", ls="--", label="Target"),
        Line2D([], [], color="#444444", marker="^", ms=3.4, mfc="white", ls="none", label="Proxy without simulated epoching"),
        Line2D([], [], color="#444444", marker="o", ms=3.4, ls="none", label="Simulated-epoching proxy"),
        Line2D([], [], color="#444444", lw=1.6, alpha=FIT_COLOR_ALPHA, label=f"{name} fit" if args.surrogate == "mariner" else "Olmix log-linear fit"),
        Line2D([], [], color="#444444", marker=OPTIMUM_MARKER, ms=7, ls="none", markeredgecolor="#222222", label=f"{name} fitted optimum"),
        Line2D([], [], color="#555555", marker="*", ms=9, ls="none", label="Measured target minimum"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=6.9, handlelength=1.6, columnspacing=1.2)
    fig.text(0.5, 0.855, "  |  ".join(f"{label}: {two.EVAL_LABELS[benchmark]}" for _, benchmark, label in rows), ha="center", va="bottom", fontsize=6.9)
    fig.tight_layout(rect=(0, 0, 1, 0.845))
    for extension in ("pdf", "png"):
        fig.savefig(DIRECTORY / f"{stem}.{extension}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    (DIRECTORY / f"{stem}_fits.json").write_text(json.dumps(optima, indent=2) + "\n")
    for arm, per_domain in optima.items():
        for domain, o in per_domain.items():
            extra = f"coef = {tuple(round(c, 3) for c in o['coefficients'])}  huber = {o['huber_loss']:.5f}" if "coefficients" in o else f"rmse = {o['rmse']:.4f}  shape = {o['shape']}"
            print(f"{arm:<10} {domain:<15} optimum p = {o['optimum_weight'] * 100:5.1f}%  ({o['optimum_epochs']:.2f} target-scale ep.)  {extra}")


if __name__ == "__main__":
    main()
