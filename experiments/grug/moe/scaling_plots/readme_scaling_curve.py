"""README scaling curve: loss = 1.6 + 95.18 * C^-0.0941 through the 4 v16 baseline points."""

import os

import matplotlib.pyplot as plt
import numpy as np

L_inf = 1.6
A = 95.18
alpha = 0.0941

# v16 compute-optimal baseline points (README table)
points = [
    (2.19e17, 3.8104, "d512"),
    (1.70e18, 3.4339, "d768"),
    (9.00e18, 3.1605, "d1024"),
    (2.83e19, 3.0065, "d1280"),
]

fig, ax = plt.subplots(figsize=(9, 6))

# Fitted curve
c_grid = np.geomspace(min(p[0] for p in points) * 0.5, 1e23, 400)
loss_grid = L_inf + A * c_grid ** (-alpha)
ax.plot(c_grid, loss_grid, "-", color="#1f77b4", lw=2, label=f"loss = {L_inf} + {A} · C$^{{-{alpha:.4f}}}$")

# L_inf asymptote
ax.axhline(L_inf, color="gray", ls="--", lw=1, alpha=0.6)
ax.annotate(f"L∞ = {L_inf}", xy=(c_grid[-1], L_inf), xytext=(-90, 8), textcoords="offset points",
            color="gray", fontsize=9)

# Data points
xs = [p[0] for p in points]
ys = [p[1] for p in points]
labels = [p[2] for p in points]
ax.scatter(xs, ys, s=90, color="#d62728", zorder=5, edgecolors="black", lw=0.5, label="v16 compute-optimal baseline")
for x, y, lbl in zip(xs, ys, labels, strict=True):
    ax.annotate(f"{lbl}\n{y:.3f}", xy=(x, y), xytext=(8, 8), textcoords="offset points", fontsize=9)

# Highlight extrapolation budgets
for budget, label in [(1e21, "1e21"), (1e23, "1e23")]:
    projected = L_inf + A * budget ** (-alpha)
    ax.scatter([budget], [projected], marker="*", s=200, color="#2ca02c", zorder=5,
               edgecolors="black", lw=0.5)
    ax.annotate(f"{label} → {projected:.3f}", xy=(budget, projected), xytext=(8, -16),
                textcoords="offset points", fontsize=9, color="#2ca02c")

ax.set_xscale("log")
ax.set_xlim(1e17, 1.5e23)
ax.set_ylim(1.55, 4.0)
ax.set_xlabel("compute (FLOPs)")
ax.set_ylabel("paloma macro_loss")
ax.set_title("README scaling law (v16 isoflop fit): loss vs compute")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "readme_scaling_curve.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
