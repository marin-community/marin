# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001  # intentional unicode glyphs in figure labels

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# Chinchilla floor+power fit through 3e20, held-out % error (pred/actual-1)*100
# Source: delphi_isotoken_endpoint_scaling.py print_summary (2026-06-18, 5-budget rerun w/ tok8b — FULL ladder)
labels = ["500M", "1B", "2B", "4B", "8B"]  # iso-token midtraining token budgets (each 2x the last)
err_1e21 = [-2.31, -2.17, -2.12, -2.28, -2.08]
err_1e22 = [-3.66, -3.78, -3.59, -3.73, -3.07]
isoflop_1e21, isoflop_1e22 = 2.92, 18.57  # iso-FLOP K=0.20 (D grows with N), not a fixed budget
x = list(range(len(labels)))  # evenly-spaced categorical positions (budgets double => uniform)

fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=150)

ax.axhline(
    isoflop_1e22, color="#E45756", ls="--", lw=1.7, label=f"iso-FLOP 1e22 (+{isoflop_1e22:.1f}%)  — D grows with N"
)
ax.axhline(isoflop_1e21, color="#E45756", ls=":", lw=1.4, alpha=0.85, label=f"iso-FLOP 1e21 (+{isoflop_1e21:.1f}%)")
ax.axhline(0, color="#888", lw=1.0)

ax.plot(x, err_1e22, "o-", color="#1f3c88", lw=2.4, ms=11, label="iso-token 1e22 (fixed D)")
ax.plot(x, err_1e21, "s-", color="#54A24B", lw=2.4, ms=10, label="iso-token 1e21 (fixed D)")
for xi, y in zip(x, err_1e22, strict=True):
    ax.annotate(
        f"{y:.2f}%",
        (xi, y),
        textcoords="offset points",
        xytext=(0, -16),
        ha="center",
        fontsize=10,
        color="#1f3c88",
        fontweight="bold",
    )
for xi, y in zip(x, err_1e21, strict=True):
    ax.annotate(
        f"{y:.2f}%", (xi, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10, color="#54A24B"
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.35, len(labels) - 0.65)
ax.set_ylim(-7, 21)
ax.set_xlabel("midtraining token budget  D  (iso-token; each step = 2×)", fontsize=12)
ax.set_ylabel("held-out extrapolation error  (pred − actual)/actual  [%]", fontsize=12)
ax.set_title(
    "Scaling-law extrapolation error vs midtraining tokens — full 500M→1B→2B→4B→8B\n"
    "(Chinchilla fit through 3e20, mix p33m67, lr0.5)",
    fontsize=12.5,
)
ax.legend(loc="center right", fontsize=9.5, framealpha=0.95)
ax.grid(True, axis="y", alpha=0.25)
ax.annotate(
    "iso-token error is flat & small (~−2 to −4%)\nacross the whole 16× token range (500M→8B)\n"
    '→ the +18.6% "1e22 miss" is an iso-FLOP artifact',
    (0.05, 8.5),
    fontsize=9.5,
    color="#333",
    bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#e0c200", lw=1),
)
fig.tight_layout()
out = "sk_midtrain_analysis_fable/error_vs_tokens.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
