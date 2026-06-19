# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001  # intentional unicode glyphs in figure labels

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

# x = pretrain-FLOP scale C (the base keys ARE the C values). Fit: L(C)=E+A*(C/1e18)^(-alpha), fit through 3e20.
C = np.array([3e18, 9e18, 2e19, 3e19, 9e19, 2e20, 3e20, 1e21, 1e22])
SMALL = C[:7]  # fit-training ladder (3e18..3e20)
HELD = C[7:]  # 1e21, 1e22 (held out)

# iso-FLOP K=0.20 (D grows with N) — the ladder that produced the "1e22 miss"
isoflop_loss = np.array([1.43522, 1.27370, 1.19341, 1.13897, 1.02104, 0.95401, 0.90956, 0.79353, 0.56102])
isoflop_fit = dict(E=0.1595, A=1.4426, alpha=0.1138)  # fit through 3e20 (n=7), R2=0.99901

# iso-token, fixed D = 8B (the largest fixed-budget rung)
isotok_loss = np.array([1.06714, 1.02996, 0.96791, 0.94596, 0.90147, 0.86239, 0.83573, 0.80592, 0.72012])
isotok_fit = dict(E=0.0000, A=1.1400, alpha=0.0533)  # fit through 3e20 (n=7), R2=0.99038


def fit_curve(p, xs):
    return p["E"] + p["A"] * (xs / 1e18) ** (-p["alpha"])


def err(p, c, actual):  # (pred-actual)/actual * 100
    return (fit_curve(p, c) - actual) / actual * 100


xs_line = np.logspace(np.log10(3e18), np.log10(1e22), 200)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=150, sharey=True)

for ax, loss, p, title, hl_color, ladder_note in [
    (
        axL,
        isoflop_loss,
        isoflop_fit,
        "iso-FLOP ladder  (D grows with N: 0.24B → 32B)",
        "#E45756",
        "1e22 secretly trained on 32B tokens",
    ),
    (
        axR,
        isotok_loss,
        isotok_fit,
        "iso-token ladder  (fixed D = 8B for every base)",
        "#54A24B",
        "every base sees the SAME 8B tokens",
    ),
]:
    # fit line (through small ladder), extended across full range
    ax.plot(xs_line, fit_curve(p, xs_line), "--", color="#888", lw=1.8, zorder=1, label="scaling fit through 3e18–3e20")
    # small ladder (fit-training) points
    ax.scatter(SMALL, loss[:7], s=70, color="#4C78A8", zorder=3, label="fit ladder (3e18–3e20)")
    # held-out points
    ax.scatter(
        HELD, loss[7:], s=240, marker="*", color=hl_color, edgecolor="k", lw=0.6, zorder=4, label="held out (1e21, 1e22)"
    )
    # highlight the 1e22 gap (fit pred vs actual)
    c22, a22 = 1e22, loss[8]
    pr22 = fit_curve(p, c22)
    e22 = err(p, c22, a22)
    ax.plot([c22, c22], [a22, pr22], color=hl_color, lw=2.2, ls=":", zorder=2)
    ax.annotate(
        f"1e22: {e22:+.1f}%",
        (c22, (a22 * pr22) ** 0.5),
        textcoords="offset points",
        xytext=(-12, 0),
        ha="right",
        fontsize=12,
        fontweight="bold",
        color=hl_color,
    )
    e21 = err(p, 1e21, loss[7])
    ax.annotate(
        f"1e21: {e21:+.1f}%", (1e21, loss[7]), textcoords="offset points", xytext=(6, -16), fontsize=9.5, color=hl_color
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pretrain-FLOP scale  C", fontsize=12)
    ax.set_title(title, fontsize=12.5)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.annotate(
        ladder_note,
        (0.5, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
        color="#555",
    )

axL.set_ylabel("nemotron_cc_math 4plus  val loss", fontsize=12)
for ax in (axL, axR):
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_yticks([0.55, 0.7, 0.9, 1.1, 1.4])
fig.suptitle(
    "Why the “1e22 miss” is an iso-FLOP artifact, not a 1e22 effect\n"
    "Same scaling-law extrapolation (fit 3e18–3e20 → predict 1e21/1e22), two midtraining-token regimes",
    fontsize=13.5,
    fontweight="bold",
)
fig.text(
    0.5,
    0.005,
    "iso-FLOP over-predicts 1e22 loss by +18.6% (1e22 sits far BELOW its own fit) only because it also got "
    "vastly more tokens; hold tokens fixed and the same model lands ON the fit (−3%).",
    ha="center",
    fontsize=10,
    color="#333",
    bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#e0c200", lw=1),
)
fig.tight_layout(rect=[0, 0.04, 1, 0.93])
out = "sk_midtrain_analysis_fable/isoflop_miss_is_token_artifact.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print(
    f"iso-FLOP 1e22 err = {err(isoflop_fit, 1e22, isoflop_loss[8]):+.2f}% ; "
    f"iso-token-8B 1e22 err = {err(isotok_fit, 1e22, isotok_loss[8]):+.2f}%"
)
