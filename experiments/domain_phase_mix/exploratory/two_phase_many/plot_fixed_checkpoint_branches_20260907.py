# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8", "numpy==2.3.5", "pandas==2.2.2"]
# ///
"""Plot the frozen fixed-checkpoint continuation comparison without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

mpl.use("Agg")

OUTPUT = Path(__file__).resolve().parent / "reference_outputs/fixed_checkpoint_branch_wspu_20260907"
MODELS = tuple(f"BRW-{index:03d}" for index in range(6))
LABELS = (
    "Hellinger\nfixed anchor",
    "Hellinger\nfitted intercept",
    "Cumulative WSPU\nlog link",
    "Cumulative WSPU\nadditive BPB",
    "Continuation WSPU\nlog link",
    "Component WSPU\n7 weighted heads",
)
# Okabe-Ito categorical colors, with neutral grey for the historical baseline.
COLORS = ("#777777", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values_by_model(frame: pd.DataFrame, metric: str) -> np.ndarray:
    assert frame.model.is_unique
    return frame.set_index("model").reindex(MODELS)[metric].to_numpy(float)


def panel(
    ax: Axes,
    values: np.ndarray,
    title: str,
    xlabel: str,
    *,
    labels: bool,
    decimals: int,
    correlation: bool = False,
) -> None:
    positions = np.arange(len(MODELS))
    finite = np.isfinite(values)
    ax.barh(positions[finite], values[finite], color=np.asarray(COLORS)[finite], height=0.62, zorder=3)
    upper = 1.0 if correlation else float(np.nanmax(values)) * 1.25
    lower = min(0.0, float(np.nanmin(values)) * 1.2) if correlation else 0.0
    ax.set_xlim(lower, upper)
    ax.set_ylim(5.6, -0.6)
    ax.set_yticks(positions, LABELS if labels else [""] * len(MODELS))
    ax.tick_params(axis="y", length=0, pad=12, labelsize=10.3)
    ax.tick_params(axis="x", labelsize=9.4, color="#999999")
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=8)
    ax.set_title(title, loc="left", fontsize=12.3, fontweight="semibold", pad=16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)
    ax.grid(False, axis="y")
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AFAFAF")
    span = upper - lower
    for index, value in enumerate(values):
        if not np.isfinite(value):
            ax.text(
                0.035,
                index,
                "Not available: no component outcomes",
                transform=ax.get_yaxis_transform(),
                fontsize=9.2,
                color="#777777",
                va="center",
            )
            continue
        label = f"{value:.{decimals}f}"
        if value == 0.0:
            ax.plot(0, index, marker="|", color=COLORS[index], markersize=15, markeredgewidth=3, zorder=4)
        if correlation and value > 0.85:
            ax.text(value - 0.025 * span, index, label, color="white", ha="right", va="center", fontsize=10)
        else:
            ax.text(value + 0.02 * span, index, label, color="#252525", ha="left", va="center", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metrics_path = args.output / "comparison/metrics.csv"
    frame = pd.read_csv(metrics_path)
    primary = frame[frame.cohort.eq("proportional") & frame.test.eq("coverage")]
    cap10 = frame[frame.cohort.eq("cap10_local") & frame.test.eq("local80")]
    local = frame[frame.test.eq("local10")]
    assert set(primary.model) == set(MODELS)
    assert set(cap10.model) == set(MODELS[:-1])
    assert local.groupby("model").size().eq(5).all()
    local_mean = local.groupby("model", as_index=False).mean(numeric_only=True)
    series = (
        values_by_model(primary, "rmse"),
        values_by_model(cap10, "rmse"),
        values_by_model(local_mean, "regret1"),
        values_by_model(local_mean, "spearman"),
    )
    destination = args.output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.usetex": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    figure.subplots_adjust(left=0.195, right=0.965, bottom=0.17, top=0.865, hspace=0.48, wspace=0.22)
    figure.suptitle(
        "Fixed-checkpoint continuation prediction", x=0.195, y=0.975, ha="left", fontsize=17, weight="semibold"
    )
    figure.text(
        0.195, 0.927, "Frozen fits · held-out continuation actions · Uncheatable BPB", fontsize=11, color="#555555"
    )
    panel(axes[0, 0], series[0], "A  Proportional prefix: 40 coverage actions", "RMSE (BPB)  ↓", labels=True, decimals=4)
    panel(axes[0, 1], series[1], "B  Cap-10 prefix: 80 local actions", "RMSE (BPB)  ↓", labels=False, decimals=4)
    panel(
        axes[1, 0],
        series[2],
        "C  Five checkpoints: 10 local actions each",
        "Mean selected-action regret (BPB)  ↓",
        labels=True,
        decimals=6,
    )
    panel(
        axes[1, 1],
        series[3],
        "D  Five checkpoints: 10 local actions each",
        "Mean Spearman correlation  ↑",
        labels=False,
        decimals=3,
        correlation=True,
    )
    figure.text(
        0.195,
        0.047,
        "Local summaries weight the five checkpoints equally. Regret is measured against the best of the ten actions.\n"
        "Point estimates from adaptive development data; uncertainty is reported in the paired bootstrap tables.",
        fontsize=9.8,
        color="#555555",
        linespacing=1.6,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(destination / f"fixed_checkpoint_comparison.{suffix}", dpi=180, facecolor="white")
    plt.close(figure)
    plotted = pd.DataFrame(
        {
            "model": MODELS,
            "label": LABELS,
            "primary_rmse": series[0],
            "cap10_local_rmse": series[1],
            "local10_mean_regret1": series[2],
            "local10_mean_spearman": series[3],
        }
    )
    plotted.to_csv(destination / "plotted_values.csv", index=False)
    outputs = {
        path.name: sha256(path) for path in destination.glob("*") if path.is_file() and path.name != "manifest.json"
    }
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(Path(__file__)),
                "source_sha256": sha256(Path(__file__)),
                "inputs": {str(metrics_path): sha256(metrics_path)},
                "outputs": outputs,
            },
            indent=2,
        )
        + "\n"
    )
    print(plotted.to_string(index=False))


if __name__ == "__main__":
    main()
