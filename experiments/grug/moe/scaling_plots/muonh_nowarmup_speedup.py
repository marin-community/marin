"""MuonH no-warmup speedup (Issue 5619), equal-TPS via agent.md per-point A formula."""

import os

import matplotlib.pyplot as plt
import numpy as np

L_inf = 1.6
A = 95.18
alpha = 0.0941

# v16 baseline (loss, TPS) at each compute-optimal point.
baselines = [
    (2.19e17, 3.8104, 405_630, "d512"),
    (1.70e18, 3.4339, 273_532, "d768"),
    (9.00e18, 3.1605, 175_162, "d1024"),
    (2.83e19, 3.0065, 128_278, "d1280"),
]

# MuonH no-warmup variant (loss, TPS) from wandb.
variant = [
    (2.19e17, 3.7404, 411_770),  # d512
    (1.70e18, 3.3834, 280_575),  # d768
    (9.00e18, 3.1230, 179_490),  # d1024
    (2.83e19, 2.9706, 133_134),  # d1280
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

color = "#9467bd"
for (compute, loss, var_tps), (_, bl_loss, bl_tps, _) in zip(variant, baselines, strict=True):
    # agent.md formula: re-fit A through each baseline point, then find the
    # compute the baseline would need to match the variant loss.
    A_bl = (bl_loss - L_inf) * compute ** alpha
    c_needed = (A_bl / (loss - L_inf)) ** (1 / alpha)
    eq_tps = c_needed / compute
    effective = (c_needed / bl_tps) / (compute / var_tps)
    ax.plot([compute, c_needed], [loss, loss], color=color, lw=2, alpha=0.85, zorder=4)
    ax.scatter([compute], [loss], s=110, color=color, edgecolors="black", lw=0.5, zorder=5)
    ax.annotate(
        f"{eq_tps:.2f}×",
        xy=(compute, loss),
        xytext=(0, -16),
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
ax.set_title("MuonH No-Warmup Speedup (Issue 5619)", fontsize=15)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muonh_nowarmup_speedup.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
