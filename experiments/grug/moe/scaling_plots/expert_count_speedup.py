"""Expert Sparsity Speedup (Issue 5387): two subplots, e128 and e256 vs e64 baseline curve."""

import os

import matplotlib.pyplot as plt
import numpy as np

L_inf = 1.6
A = 95.18
alpha = 0.0941

baselines = [
    (2.19e17, 3.8104, "d512"),
    (1.70e18, 3.4339, "d768"),
    (9.00e18, 3.1605, "d1024"),
    (2.83e19, 3.0065, "d1280"),
]
# (dim, loss, tps, effective_speedup_from_table) from issue table
e64_tps = {512: 407378, 768: 274431, 1024: 176935, 1280: 128435}
e128 = [
    (512, 3.7910, 380640, 1.026),
    (768, 3.4059, 264680, 1.136),
    (1024, 3.1359, 170963, 1.144),
    (1280, 2.9784, 124474, 1.201),
]
e256 = [
    (512, 3.7801, 349528, 0.994),
    (768, 3.3851, 241107, 1.170),
    (1024, 3.1138, 160459, 1.252),
    # d1280 crashed
]
dim_to_budget = {512: 2.19e17, 768: 1.70e18, 1024: 9.00e18, 1280: 2.83e19}

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

x_lo, x_hi = 1.7e17, 3.7e19
c_grid = np.geomspace(x_lo, x_hi, 200)
loss_grid = L_inf + A * c_grid ** (-alpha)

def _draw(ax, points, title, color):
    # Baseline curve
    ax.plot(c_grid, loss_grid, "-", color="#1f77b4", lw=2, label=f"e64 baseline: 1.6 + {A}·C$^{{-{alpha:.4f}}}$")
    # Baseline data points (no per-dim labels — they're already visible from the new-point columns).
    bx = [b[0] for b in baselines]
    by = [b[1] for b in baselines]
    ax.scatter(bx, by, s=70, color="#1f77b4", edgecolors="black", lw=0.5, zorder=4)
    # New data points
    for dim, loss, var_tps, effective in points:
        c = dim_to_budget[dim]
        bl_loss = next(b[1] for b in baselines if b[2] == f"d{dim}")
        bl_tps = e64_tps[dim]
        # Equal-TPS (pure-quality) speedup: factor out the throughput difference.
        # effective = eq_tps * (var_tps / bl_tps), so eq_tps = effective * bl_tps / var_tps.
        eq_tps = effective * bl_tps / var_tps
        # Project the new-point loss onto the baseline curve: solve for C s.t.
        # loss = L_inf + A * C^-alpha  =>  C = (A / (loss - L_inf))^(1/alpha).
        c_proj = (A / (loss - L_inf)) ** (1 / alpha)
        ax.plot([c, c_proj], [loss, loss], color=color, lw=2, alpha=0.85, zorder=4)
        ax.scatter([c], [loss], s=110, color=color, edgecolors="black", lw=0.5, zorder=5)
        # Labels below: eq-TPS speedup (main), effective in parens.
        ax.annotate(
            f"{eq_tps:.2f}×\n({effective:.2f}×)",
            xy=(c, loss), xytext=(0, -18), textcoords="offset points",
            fontsize=14, color=color, ha="center", va="top", fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("compute (FLOPs)", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)

_draw(axes[0], e128, "128 experts  (4/128 = 3.1% sparsity)", "#ff7f0e")
_draw(axes[1], e256, "256 experts  (4/256 = 1.6% sparsity)", "#d62728")
axes[0].set_ylabel("paloma macro_loss", fontsize=14)

fig.suptitle("Expert Sparsity Speedup (Issue 5387)", fontsize=17)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_count_speedup.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
