# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the 535B-A23B / 18T-token scaling-ladder figures for issue #8435.

The scaling ladder (``launch_scaling_ladder.py``) trains one uniform hero EP recipe at five widths.
Four rungs finished (the d2048 rung crashed at ~81%); this script pulls their W&B histories and the
d6144 hero's compute budget, then writes five figures to ``scaling_ladder_figs/``:

* ``ladder_train_ce_fullres.png``       -- full-granularity training cross-entropy vs run progress.
* ``ladder_extrapolation_60_80.png``    -- per-rung 60-80% linear extrapolation vs the actual final.
* ``ladder_scaling_laws.png``           -- per-5% power-law fits ``L = 1.5 + A*C^-a`` (log-log).
* ``hero_prediction_by_percentile.png`` -- rung curves + the d6144 hero prediction at every 5%.
* ``scaling_law_comparison.png``        -- our 100% fit vs the May-Recipe and 67B-A2B scaling laws.

Run:  uv run python experiments/grug/moe_hero_ep/plot_scaling_ladder.py
Requires W&B read access to ``marin-community/marin_moe`` plus matplotlib.

Compute ``C`` excludes the lm_head. The per-rung constants below come from the ladder's own analytic
FLOPs calculator (``experiments/grug/moe_hero_ep/train.py::_compute_flops`` -> levanter
``lm_flops_per_token``) with the ``lm_head = 2*hidden*vocab`` term removed, times batch and steps:
``C_full = (flops/example_no_lmhead) * batch * steps``. active_params is ``_active_params(model)``.
"""

import json
import os

import matplotlib as mpl
import numpy as np
import wandb

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ENTITY_PROJECT = "marin-community/marin_moe"
FIG_DIR = os.path.join(os.path.dirname(__file__), "scaling_ladder_figs")
ISSUE_FOOTER = "github.com/marin-community/marin/issues/8435"

ASYMPTOTE = 1.5  # irreducible-loss anchor for the ladder power-law fits (paloma macro loss).
D2048_CORRECTION = 0.005  # bias correction applied to the d2048 crashed-rung extrapolation.

# W&B run name per rung (the d2048 rung crashed at ~81%).
RUNS = {
    "d768": "rav-ladder-d768-v2",
    "d1024": "rav-ladder-d1024",
    "d1536": "rav-ladder-d1536",
    "d2048": "rav-ladder-d2048-v3",
}
COLOR = {"d768": "#4C72B0", "d1024": "#DD8452", "d1536": "#55A868", "d2048": "#C44E52"}
# params/token notation for the legend (total-Aactive . tokens); no width in the labels.
TAG = {
    "d768": "1.6B-A61M · 48B tok",
    "d1024": "4.0B-A162M · 128B tok",
    "d1536": "11.5B-A481M · 381B tok",
    "d2048": "27.7B-A1.2B · 926B tok",
    "d6144": "535B-A23B · 18T tok",
}

# lm_head-excluded compute constants (see module docstring for the derivation).
FLOPS = {
    "d768": {"C_per_step": 2.320210052775936e15, "steps": 11420, "C_full": 2.649679880270119e19},
    "d1024": {"C_per_step": 1.129954355970048e16, "steps": 15276, "C_full": 1.7261182741798453e20},
    "d1536": {"C_per_step": 9.130921800656486e16, "steps": 15128, "C_full": 1.3813258500033133e21},
    "d2048": {"C_per_step": 3.857022193930076e17, "steps": 20072, "C_full": 7.741814947656449e21},
    "d6144": {"C_per_step": 6.690294608796058e18, "steps": 390251, "C_full": 2.6108941613772703e24},
}
DROPLESS_MACRO = "eval_dropless/paloma/macro_loss"
TRAIN_CE = "train/cross_entropy_loss"

# Prior paloma-macro-loss scaling laws for comparison, each (asymptote, A, alpha).
MAY_RECIPE_LAW = (1.6, 88.32, 0.0941)  # agent.md baseline (May Recipe, drop-1e18 fit)
RUN_67B_LAW = (1.4, 84.74, 0.0862)  # 67B-A2B 10T preregistered fit (issue #6044)


def _footer(fig):
    fig.text(0.99, 0.013, ISSUE_FOOTER, ha="right", va="bottom", fontsize=11.5, color="0.38", style="italic")


def fetch_wandb():
    """Return {rung: {nts, eval_frac->loss (snapped to 5%), ce_step[], ce[]}} from W&B."""
    api = wandb.Api()
    out = {}
    for rung, name in RUNS.items():
        run = api.runs(ENTITY_PROJECT, filters={"display_name": name})[0]
        nts = int(run.config["stop_after_steps"])
        hist = run.history(keys=["_step", DROPLESS_MACRO], samples=100000, pandas=True).dropna()
        grid = {}
        for step, loss in zip(hist["_step"], hist[DROPLESS_MACRO], strict=False):
            grid[round(round(step / nts * 20) / 20, 2)] = float(loss)  # snap eval to nearest 5%
        ce_step, ce = [], []
        for row in run.scan_history(keys=["_step", TRAIN_CE], page_size=10000):
            if row.get(TRAIN_CE) is not None:
                ce_step.append(row["_step"])
                ce.append(row[TRAIN_CE])
        order = np.argsort(ce_step)
        out[rung] = {
            "nts": nts,
            "grid": grid,
            "ce_step": np.array(ce_step, float)[order],
            "ce": np.array(ce, float)[order],
        }
    return out


def _d2048_pinned(grid):
    """Linear fit over the d2048 60-80% window, shifted so its 100% value is the corrected prediction."""
    window = [0.60, 0.65, 0.70, 0.75, 0.80]
    slope, intercept = np.polyfit(np.array(window), np.array([grid[w] for w in window]), 1)
    return slope, intercept


def plot_train_ce(data):
    gmin = min(data[r]["ce"].min() for r in RUNS)
    fig, ax = plt.subplots(figsize=(12, 7))
    for r in RUNS:
        d = data[r]
        ax.plot(d["ce_step"] / d["nts"] * 100, d["ce"], "-", color=COLOR[r], lw=0.55, alpha=0.85, label=TAG[r])
    hi = 3.0
    ax.axvline(80, color="0.2", ls="--", lw=1.6)  # datamix phase-2 boundary (uniform across rungs)
    ax.text(79.3, hi - 0.03, "datamix phase 2", rotation=90, va="top", ha="right", fontsize=10, color="0.2")
    ax.set_ylim(round(gmin - 0.05, 2), hi)
    ax.set_xlim(0, 100)
    ax.set_xlabel("run progress (%)")
    ax.set_ylabel("training cross-entropy loss")
    ax.set_title("535B-A23B 18T tok Scaling Ladder")
    ax.grid(alpha=0.25)
    ax.set_xticks(range(0, 101, 10))
    for line in ax.legend(fontsize=10, loc="upper right").get_lines():
        line.set_linewidth(2.5)
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "ladder_train_ce_fullres.png"), dpi=130)


def plot_extrapolation(data):
    window = [0.60, 0.65, 0.70, 0.75, 0.80]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, r in zip(axes.ravel(), RUNS, strict=False):
        grid, nts, c = data[r]["grid"], data[r]["nts"], COLOR[r]
        gs = sorted(grid)
        ax.plot([x * 100 for x in gs], [grid[k] for k in gs], "-", color="0.75", lw=1)
        ax.plot([x * 100 for x in gs], [grid[k] for k in gs], "o", color="0.6", ms=3)
        vw = np.array([grid[w] for w in window])
        ax.plot([w * 100 for w in window], vw, "o", color=c, ms=7, zorder=4, label="60-80% fit points (5)")
        slope, intercept = np.polyfit(np.array(window), vw, 1)
        xl = np.linspace(0.60, 1.0, 50)
        ax.plot(xl * 100, slope * xl + intercept, "--", color=c, lw=1.8, label="linear extrapolation")
        pred = slope + intercept
        ax.plot(100, pred, "o", mfc="none", mec=c, mew=2, ms=13, label=f"predicted 100%: {pred:.3f}")
        if 1.0 in grid:
            act = grid[1.0]
            ax.plot(100, act, "*", color="black", ms=15, label=f"actual 100%: {act:.3f}")
            ax.annotate(
                f"pred - actual = {pred - act:+.3f}", xy=(82, (pred + act) / 2), fontsize=9, color=c, va="center"
            )
        else:
            corr = pred + D2048_CORRECTION
            ax.plot(100, corr, "D", color=c, ms=9, label=f"corrected (+{D2048_CORRECTION:.3f}): {corr:.3f}")
            ax.axvspan(max(gs) * 100, 100, color="red", alpha=0.06)
            ax.text(
                0.5,
                0.06,
                "crashed ~80% — no actual 100%",
                transform=ax.transAxes,
                ha="center",
                color="#C44E52",
                fontsize=9,
            )
        ax.set_title(f"{r}  (stop_after_steps={nts})", fontsize=11)
        ax.set_xlabel("training progress (%)")
        ax.set_ylabel("dropless paloma macro-loss")
        ax.axvspan(60, 80, color=c, alpha=0.05)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Scaling-ladder rungs: 60-80% linear extrapolation of dropless paloma macro-loss vs actual final", fontsize=13
    )
    _footer(fig)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, "ladder_extrapolation_60_80.png"), dpi=130)


def _fit_and_hero(data):
    """Per-5% power-law fit across rungs; return (fits, hero) where hero[g]=(C_hero_g, predicted_loss)."""
    rungs = list(RUNS)
    d2048_max = max(data["d2048"]["grid"])
    slope, intercept = _d2048_pinned(data["d2048"]["grid"])

    def loss_at(r, g):
        if r == "d2048" and g > d2048_max:
            return slope * g + intercept + D2048_CORRECTION  # pinned extrapolation for the crashed rung
        return data[r]["grid"][round(g, 2)]

    def compute_at(r, g):
        return FLOPS[r]["C_per_step"] * FLOPS[r]["steps"] * g

    grid = [round(x, 2) for x in np.arange(0.05, 1.0001, 0.05)]
    cps_h, steps_h = FLOPS["d6144"]["C_per_step"], FLOPS["d6144"]["steps"]
    fits, hero = {}, {}
    for g in grid:
        cs = np.array([compute_at(r, g) for r in rungs])
        ls = np.array([loss_at(r, g) for r in rungs])
        s, i = np.polyfit(np.log(cs), np.log(ls - ASYMPTOTE), 1)  # log-log fit, fixed asymptote
        a, alpha = np.exp(i), -s
        fits[g] = (a, alpha, cs, ls)
        c_hero_g = cps_h * steps_h * g  # hero compute at this same training fraction
        hero[g] = (c_hero_g, ASYMPTOTE + a * c_hero_g ** (-alpha))
    return grid, fits, hero, slope, intercept, d2048_max, loss_at, compute_at


def plot_scaling_laws(data):
    grid, fits, hero, *_, compute_at = _fit_and_hero(data)
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(11, 7.5))
    xline = np.geomspace(compute_at("d768", 0.05) * 0.7, hero[1.0][0] * 1.3, 200)
    for g in grid:
        a, alpha, cs, ls = fits[g]
        c = cmap(g)
        ax.plot(
            xline,
            a * xline ** (-alpha),
            "-",
            color=c,
            lw=2.4 if g == 1.0 else 1.0,
            alpha=0.95 if g == 1.0 else 0.5,
            zorder=3 if g == 1.0 else 2,
        )
        ax.plot(cs, ls - ASYMPTOTE, "o", color=c, ms=4.5, zorder=4)
    hx = [hero[g][0] for g in grid]
    hy = [hero[g][1] - ASYMPTOTE for g in grid]
    ax.plot(
        hx[:-1], hy[:-1], "o", color="crimson", ms=7, mec="white", mew=0.7, zorder=6, label="hero prediction (per 5%)"
    )
    ax.plot(
        hx[-1],
        hy[-1],
        "*",
        color="crimson",
        ms=22,
        mec="white",
        mew=0.8,
        zorder=7,
        label=f"hero @100%: {hero[1.0][1]:.3f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training compute  C  (FLOPs, no lm_head)")
    ax.set_ylabel("dropless paloma macro-loss - 1.5  (excess loss)")
    ax.set_title("535B-A23B 18T tok Scaling Ladder — per-5% scaling-law fits (L = 1.5 + A*C$^{-\\alpha}$)")
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(5, 100))
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("training fraction (%)")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "ladder_scaling_laws.png"), dpi=130)


def plot_hero_by_percentile(data):
    grid, _fits, hero, slope, intercept, d2048_max, *_ = _fit_and_hero(data)
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for r in ["d768", "d1024", "d1536"]:
        gs = sorted(data[r]["grid"])
        ax.plot(
            [x * 100 for x in gs],
            [data[r]["grid"][k] for k in gs],
            "-o",
            color=COLOR[r],
            lw=1.8,
            ms=5,
            mfc=COLOR[r],
            mec=COLOR[r],
            label=TAG[r],
        )
    grid2048 = data["d2048"]["grid"]
    gm = [k for k in sorted(grid2048) if k <= d2048_max]
    ax.plot(
        [x * 100 for x in gm],
        [grid2048[k] for k in gm],
        "-o",
        color=COLOR["d2048"],
        lw=1.8,
        ms=5,
        mfc=COLOR["d2048"],
        mec=COLOR["d2048"],
        label=TAG["d2048"],
    )
    ext = [g for g in grid if g > d2048_max]
    xe = [d2048_max, *ext]
    ye = [grid2048[d2048_max], *(slope * g + intercept + D2048_CORRECTION for g in ext)]
    ax.plot(
        [x * 100 for x in xe],
        ye,
        "--o",
        color=COLOR["d2048"],
        lw=1.8,
        ms=7,
        mfc="white",
        mec=COLOR["d2048"],
        mew=1.6,
        label="extrapolated (pinned)",
    )
    hx = [g * 100 for g in grid]
    hy = [hero[g][1] for g in grid]
    ax.plot(hx, hy, "--", color="crimson", lw=2.6, zorder=5)
    ax.plot(
        hx,
        hy,
        ls="none",
        marker="*",
        ms=14,
        mfc="white",
        mec="crimson",
        mew=1.8,
        zorder=6,
        label=f"hero · {TAG['d6144']}",
    )
    ax.text(
        0.015,
        0.03,
        "solid + filled = measured      dashed + hollow = extrapolated / predicted",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        color="0.35",
    )
    ax.set_xlim(2, 103)
    ax.set_xlabel("training progress (percentile %)")
    ax.set_ylabel("dropless paloma macro-loss")
    ax.set_title("535B-A23B 18T tok Scaling Ladder")
    ax.grid(alpha=0.25)
    ax.set_xticks(range(0, 101, 10))
    ax.legend(fontsize=9, loc="upper right")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "hero_prediction_by_percentile.png"), dpi=130)
    return hero[1.0][1]


def plot_comparison(data):
    """Overlay our 100% ladder fit with the prior May-Recipe and 67B-A2B scaling laws."""
    _, fits, hero, *_ = _fit_and_hero(data)
    a, alpha, cs_ours, ls_ours = fits[1.0]  # our four rung anchors at 100% (d2048 is pinned-corrected)
    c_hero = hero[1.0][0]
    laws = [
        ("ours (535B-A23B ladder)", ASYMPTOTE, a, alpha, "#C44E52"),
        ("May Recipe baseline (agent.md)", *MAY_RECIPE_LAW, "#4C72B0"),
        ("67B-A2B 10T (preregistered)", *RUN_67B_LAW, "#55A868"),
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    xline = np.geomspace(1e17, 3e24, 300)
    for label, asymptote, aa, al, color in laws:
        ax.plot(
            xline,
            asymptote + aa * xline ** (-al),
            "-",
            color=color,
            lw=2.4,
            label=f"{label}:  L = {asymptote} + {aa:.3g}·C$^{{-{al:.4f}}}$",
        )
        ax.plot(c_hero, asymptote + aa * c_hero ** (-al), "o", color=color, ms=9, mec="white", mew=1, zorder=5)
    ax.plot(
        cs_ours,
        ls_ours,
        "o",
        color="#C44E52",
        ms=8,
        mec="white",
        mew=0.9,
        zorder=6,
        label="our fit points (4 rungs @100%)",
    )
    ax.axvline(c_hero, color="0.4", ls=":", lw=1.3)
    ax.annotate(
        f"hero compute\n{c_hero:.1e} FLOPs",
        xy=(c_hero, 2.05),
        xytext=(c_hero * 0.06, 2.4),
        fontsize=10,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2),
    )
    ax.set_xscale("log")
    ax.set_xlim(1e17, 3e24)
    ax.set_ylim(1.9, 4.0)
    ax.set_xlabel("training compute  C  (FLOPs)")
    ax.set_ylabel("paloma macro-loss")
    ax.set_title("Scaling-law comparison: 535B-A23B ladder vs prior fits")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10, loc="upper right")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "scaling_law_comparison.png"), dpi=130)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = fetch_wandb()
    plot_train_ce(data)
    plot_extrapolation(data)
    plot_scaling_laws(data)
    plot_comparison(data)
    hero_final = plot_hero_by_percentile(data)
    print(f"wrote 5 figures to {FIG_DIR}; predicted d6144 hero dropless paloma macro-loss @100% = {hero_final:.4f}")
    print(json.dumps({r: {"nts": data[r]["nts"], "evals": len(data[r]["grid"])} for r in RUNS}))


if __name__ == "__main__":
    main()
