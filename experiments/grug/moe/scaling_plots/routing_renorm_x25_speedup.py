"""Routing Renorm X=2.5 speedup vs combined-feature baseline (Issue 5797), at d512/d768/d1024."""

import os

import matplotlib.pyplot as plt
import numpy as np

L_inf = 1.6
A_canon = 95.18
alpha = 0.0941

# Canonical e64 v16 baseline curve points (for reference).
canon_baselines = [
    (2.19e17, 3.8104, "d512"),
    (1.70e18, 3.4339, "d768"),
    (9.00e18, 3.1605, "d1024"),
    (2.83e19, 3.0065, "d1280"),
]

# 1pct-noclip baselines at d512/d768/d1024 (the actual baseline this comparison uses).
noclip_baselines = [
    (2.19e17, 3.6427, 328_207, "d512"),
    (1.70e18, 3.3040, 231_556, "d768"),
    (9.00e18, 3.0610, 153_848, "d1024"),
]

# Routing-renorm X=2.5 variant at the same scales, from the issue table.
# (compute, new_loss, var_tps, effective, eq_tps)
variant_x25 = [
    (2.19e17, 3.6380, 334_455, 1.044, 1.025),
    (1.70e18, 3.2959, 233_340, 1.060, 1.052),
    (9.00e18, 3.0562, 154_244, 1.038, 1.036),
]

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(10, 6.5))

x_lo, x_hi = 1.7e17, 3.7e19
c_grid = np.geomspace(x_lo, x_hi, 200)
loss_grid = L_inf + A_canon * c_grid ** (-alpha)
ax.plot(c_grid, loss_grid, "-", color="#9aa6b0", lw=1.5, alpha=0.7,
        label=f"e64 canonical: 1.6 + {A_canon}·C$^{{-{alpha:.4f}}}$")
ax.scatter([b[0] for b in canon_baselines], [b[1] for b in canon_baselines],
           s=55, color="#9aa6b0", edgecolors="black", lw=0.4, alpha=0.7, zorder=3)

# 1pct-noclip baseline points
ax.scatter([b[0] for b in noclip_baselines], [b[1] for b in noclip_baselines],
           s=90, color="#1f77b4", edgecolors="black", lw=0.5, zorder=4,
           label="combined feature baseline")

# Variant points + per-point speedup labels.
color = "#2ca02c"
for compute, loss, _var_tps, effective, eq_tps in variant_x25:
    # Horizontal segment length: (compute, loss) → (compute * eq_tps, loss).
    ax.plot([compute, compute * eq_tps], [loss, loss], color=color, lw=2, alpha=0.85, zorder=4)
    ax.scatter([compute], [loss], s=110, color=color, edgecolors="black", lw=0.5, zorder=5,
               label="X=2.5 variant" if compute == variant_x25[0][0] else None)
    ax.annotate(
        f"{eq_tps:.3f}×\n({effective:.3f}×)",
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
ax.set_title("Routing Renorm, vs Combined Feature Baseline (Issue 5797)", fontsize=15)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_renorm_x25_speedup.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
