"""PKO + Prope + LastLayer speedup (Issue 4976) vs canonical e64 baseline curve."""

import os

import matplotlib.pyplot as plt
import numpy as np

L_inf = 1.6
A = 95.18
alpha = 0.0941

baselines = [
    (2.19e17, 3.8104, 405630, "d512"),
    (1.70e18, 3.4339, 273532, "d768"),
    (9.00e18, 3.1605, 175162, "d1024"),
    (2.83e19, 3.0065, 128278, "d1280"),
]

# (compute, new_loss, new_tps, effective_speedup_from_table)
variant = [
    (2.19e17, 3.7722, 399792, 1.186),  # d512 lastpko
    (1.70e18, 3.3976, 271257, 1.226),  # d768 pko+prope (last layer naturally aligned)
    (9.00e18, 3.1356, 174856, 1.185),  # d1024 lastpko
    (2.83e19, 2.9805, 126554, 1.203),  # d1280 lastpko
]

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(10, 6.5))

x_lo, x_hi = 1.7e17, 3.7e19
c_grid = np.geomspace(x_lo, x_hi, 200)
loss_grid = L_inf + A * c_grid ** (-alpha)
ax.plot(c_grid, loss_grid, "-", color="#1f77b4", lw=2, label=f"e64 baseline: 1.6 + {A}·C$^{{-{alpha:.4f}}}$")

bx = [b[0] for b in baselines]
by = [b[1] for b in baselines]
ax.scatter(bx, by, s=70, color="#1f77b4", edgecolors="black", lw=0.5, zorder=4)

color = "#2ca02c"
for (compute, loss, var_tps, effective), (_, bl_loss, bl_tps, _) in zip(variant, baselines, strict=True):
    eq_tps = effective * bl_tps / var_tps
    c_proj = (A / (loss - L_inf)) ** (1 / alpha)
    ax.plot([compute, c_proj], [loss, loss], color=color, lw=2, alpha=0.85, zorder=4)
    ax.scatter([compute], [loss], s=110, color=color, edgecolors="black", lw=0.5, zorder=5)
    ax.annotate(
        f"{eq_tps:.2f}×\n({effective:.2f}×)",
        xy=(compute, loss),
        xytext=(0, -18),
        textcoords="offset points",
        fontsize=14,
        color=color,
        ha="center",
        va="top",
        fontweight="bold",
    )

ax.set_xscale("log")
ax.set_xlim(x_lo, x_hi)
ax.set_xlabel("compute (FLOPs)", fontsize=14)
ax.set_ylabel("paloma macro_loss", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=12)
ax.set_title("PKO + Prope + LastLayer (Issue 4976)", fontsize=15)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pko_lastlayer_speedup.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
