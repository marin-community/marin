# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot the completed proxy curves while the target sweep is unfinished."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#009E73"]
LABELS = ["Unmatched proxy", "Matched subset 1", "Matched subset 2", "Matched subset 3"]


def main():
    result = json.loads((ROOT / "pilot_proxy_results.json").read_text())
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
        for curve, color, label in zip(result["curves"], COLORS, LABELS, strict=True):
            x = np.array(curve["grid_percent"]) / 100
            y = np.array(curve["mean"])
            seeds = np.array(curve["trainer_values"])
            axes[0].errorbar(
                x, y, yerr=np.vstack((y - seeds.min(axis=0), seeds.max(axis=0) - y)),
                color=color, label=label, marker="o", markersize=4,
                linewidth=1.6, elinewidth=0.8, capsize=2,
            )
            mask = x >= 0.3
            axes[1].plot(x[mask], (y - y.min())[mask], color=color, marker="o", markersize=4, linewidth=1.6)
            selected = int(y.argmin())
            for axis, minimum in zip(axes, (y[selected], 0), strict=True):
                axis.plot(x[selected], minimum, marker="*", markersize=12, color=color,
                          markeredgecolor="white", markeredgewidth=0.5)
        for axis in axes:
            axis.set_xlabel("StarCoder token fraction, p")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.18)
        axes[0].set(xlim=(-0.02, 1.03), ylabel="Programming-languages BPB", title="Full proxy grid")
        axes[0].legend(loc="upper right", fontsize=9, frameon=False)
        axes[1].set(xlim=(0.28, 1.03), ylim=(-0.006, 0.16),
                    ylabel="BPB above own grid minimum", title="Curve shapes, p ≥ 0.3")
        axes[1].set_xticks([0.3, 0.5, 0.7, 0.9, 1.0])
        figure.suptitle("TPP10 pilot: completed proxy results", fontsize=13, fontweight="bold")
        figure.supxlabel(
            "Lines average two trainer seeds; bars span those seeds. Stars mark grid minima. Target sweep is still running.",
            fontsize=9,
        )
        figure.savefig(ROOT / "pilot_proxy_curves.png", dpi=180)
        figure.savefig(ROOT / "pilot_proxy_curves.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()
