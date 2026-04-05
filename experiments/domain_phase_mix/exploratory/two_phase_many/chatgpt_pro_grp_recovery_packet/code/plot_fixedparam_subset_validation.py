# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas"]
# ///
"""Plot the realized calibration gap from the first fixed-parameter subset sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")
matplotlib.rcParams["text.usetex"] = False


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "data" / "fixedparam_subset_validation_results.csv").sort_values("subset_size")
    best_observed = float(frame["observed_best_value"].iloc[0])

    fig, (ax_level, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(9.8, 7.0),
        dpi=180,
        sharex=True,
        constrained_layout=True,
    )
    cmap = plt.colormaps["RdYlGn_r"]

    observed = frame[frame["actual_bpb"].notna()].copy()
    ax_level.plot(
        frame["subset_size"],
        frame["predicted_optimum_value"],
        marker="o",
        linewidth=2.0,
        color=cmap(0.18),
        label="Predicted optimum BPB",
    )
    ax_level.plot(
        observed["subset_size"],
        observed["actual_bpb"],
        marker="s",
        linewidth=2.0,
        color=cmap(0.82),
        label="Realized BPB",
    )
    ax_level.axhline(
        best_observed, color="0.35", linestyle=":", linewidth=1.5, label=f"Best observed ({best_observed:.4f})"
    )
    ax_level.set_ylabel("BPB")
    ax_level.set_title("Fixed-parameter subset sweep: predicted vs realized BPB")
    ax_level.grid(True, alpha=0.25)
    ax_level.legend(loc="best", frameon=True)

    ax_gap.plot(
        observed["subset_size"],
        observed["prediction_error"],
        marker="D",
        linewidth=2.0,
        color=cmap(0.68),
        label="Actual - predicted",
    )
    ax_gap.axhline(0.0, color="0.35", linestyle=":", linewidth=1.2)
    ax_gap.set_xlabel("Observed runs used for fitting")
    ax_gap.set_ylabel("Calibration gap")
    ax_gap.grid(True, alpha=0.25)
    ax_gap.legend(loc="best", frameon=True)

    out_path = root / "reference_outputs" / "fixedparam_subset_validation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
