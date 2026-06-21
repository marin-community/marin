# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001  # intentional unicode glyphs in figure labels

"""Endpoint-scaling overlay across ALL iso-token budgets vs the iso-FLOP K=0.20 ladder.

A generalization of the single-budget figure (``render_isotoken_png.py``): instead of
one iso-token budget (1B) vs iso-FLOP, this overlays every fixed-token budget we ran
(500M, 1B, 2B, 4B, 8B) as a blue gradient against the iso-FLOP K=0.20 ladder (red).

For each series the small ladder (3e18→3e20) is fit with the published Chinchilla
floor+power ``L(C) = E + A * (C/1e18)^-alpha``; 1e21/1e22 are held out and the fit's
extrapolation error there is the readout. Reads the CSVs emitted by
``delphi_isotoken_endpoint_scaling.py`` (rerun it first if budgets are missing).
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "sk_midtrain_analysis_fable"
HELD = (1e21, 1e22)  # held-out scales (the CSV `scale` column is numeric float64)
CUTOFF = 3e20


# iso-token budgets light→dark (more tokens = lower loss = darker); iso-FLOP in red.
SERIES = [
    ("tok500m", "iso-token 500M", "#9ecae1"),
    ("tok1b", "iso-token 1B", "#6baed6"),
    ("tok2b", "iso-token 2B", "#3182bd"),
    ("tok4b", "iso-token 4B", "#08519c"),
    ("tok8b", "iso-token 8B", "#08306b"),
    ("k0p20", "iso-FLOP K=0.20 (D grows 0.24B→32B)", "#E45756"),
]

isotoken = pd.read_csv(OUT / "isotoken_endpoints.csv")
isoflop = pd.read_csv(OUT / "isoflop_k020_endpoints.csv")
fits = pd.read_csv(OUT / "isotoken_scaling_fits.csv")
allpts = pd.concat([isotoken, isoflop], ignore_index=True)
fits["cf"] = fits["cutoff_scale"].astype(float)

xs = np.logspace(np.log10(3e18), 22, 200)
fig, ax = plt.subplots(figsize=(13, 8.2), dpi=160)
ax.set_xscale("log")
ax.set_yscale("log")

err_rows = []  # (label, color, err1e21, err1e22)
for series, label, color in SERIES:
    fit = fits[fits["series"].eq(series) & np.isclose(fits["cf"], CUTOFF)].iloc[0]
    E, A, al = fit["fp_floor"], fit["fp_amplitude"], fit["fp_alpha"]

    def predict(c, E=E, A=A, al=al):
        return E + A * (c / 1e18) ** (-al)

    frame = allpts[allpts["series"].eq(series)]
    train = frame[~frame["scale"].isin(HELD)]
    held = frame[frame["scale"].isin(HELD)]

    ax.plot(xs, predict(xs), ls=":", lw=2, color=color, zorder=2)
    ax.scatter(train["scale_flops"], train["value"], s=55, color=color, zorder=3)

    errs = {}  # keyed by held-out scale (float)
    for _, row in held.iterrows():
        c, act = row["scale_flops"], row["value"]
        pred = predict(c)
        errs[float(row["scale"])] = (pred / act - 1) * 100
        # held-out actual (triangle) and prediction (x), connected to show the gap
        ax.plot([c, c], [act, pred], color=color, lw=1.4, ls="-", alpha=0.55, zorder=3)
        ax.scatter([c], [act], s=150, marker="^", color=color, edgecolor="black", lw=0.7, zorder=5)
        ax.scatter([c], [pred], s=120, marker="x", color=color, lw=2.4, zorder=5)
    e21, e22 = errs.get(1e21), errs.get(1e22)
    err_rows.append((label, color, e21, e22))
    # legend proxy carrying the 1e22 error
    ax.plot([], [], color=color, lw=2.4, marker="o", ms=6, label=f"{label}   (1e22: {e22:+.1f}%)")

# marker-meaning legend proxies
ax.plot([], [], ls=":", lw=2, color="#667085", label="Chinchilla fit  E + A·(C/1e18)^(−α),  fit ≤ 3e20")
ax.scatter([], [], s=150, marker="^", color="#ffffff", edgecolor="black", label="held-out actual (1e21, 1e22)")
ax.scatter([], [], s=120, marker="x", color="#667085", lw=2.4, label="fit prediction at held-out scale")

ax.axvline(CUTOFF, color="#94a3b8", ls="--", lw=1)
ax.text(CUTOFF, ax.get_ylim()[1] * 0.98, " fit cutoff (3e20)", fontsize=9, color="#475569", va="top")

# call out the contrast: iso-FLOP's big miss vs the tight iso-token band at 1e22
ax.annotate(
    "iso-FLOP 1e22: actual sits FAR below its fit\n→ +18.6% “miss” (looks like acceleration)",
    (1e22, 0.5610),
    textcoords="offset points",
    xytext=(-220, -6),
    fontsize=9.5,
    color="#9b2226",
)
ax.annotate(
    "iso-token 1e22 (every budget):\nactual ~on the fit, −3 to −4%",
    (1e22, 0.80),
    textcoords="offset points",
    xytext=(-205, 28),
    fontsize=9.5,
    color="#08306b",
)

ax.set_xlim(1.8e18, 3.4e22)
ax.set_xlabel("base-model pretraining compute (FLOPs)", fontsize=12)
ax.set_ylabel("final math_val_loss", fontsize=12)
ax.set_title(
    "Delphi midtraining endpoint scaling — all iso-token budgets vs iso-FLOP K=0.20  (p33m67 / lr 0.5)\n"
    "Held-out 1e21/1e22 extrapolation error is flat & small (−2 to −4%) at EVERY fixed budget; "
    "iso-FLOP misses 1e22 by +18.6% (token-budget confound)",
    fontsize=11.5,
)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.grid(True, which="both", alpha=0.22)
fig.tight_layout()
out = OUT / "delphi_isotoken_all_budgets_vs_isoflop.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print(f"{'series':34} {'1e21 err%':>10} {'1e22 err%':>10}")
for label, _color, e21, e22 in err_rows:
    print(f"{label:34} {e21:+10.2f} {e22:+10.2f}")
