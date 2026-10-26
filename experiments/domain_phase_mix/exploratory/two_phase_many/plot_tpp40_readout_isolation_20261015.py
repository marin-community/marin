# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Illustrate frozen invented readouts; no fits or new policy evaluations."""

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent / "reference_outputs/tpp40_acquisition_retention_20261015/readout_isolation"
SOURCE = ROOT / "descriptive/head_rows_v1.csv"
OUTPUT = ROOT / "illustrations"
COLORS = {"hazard": "#0072B2", "additive": "#D55E00"}
TIMINGS = {"early_heavy": ("--", "o"), "late_heavy": ("-", "s")}


def main() -> None:
    mpl.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
    rows = pd.read_csv(SOURCE)
    selected = rows.loc[
        (rows.stratum == "main")
        & (rows.clock == "global_pool")
        & (rows.mu == 3)
        & (rows.aggregate_first == 0.5)
        & (rows.timing != "tied")
    ]
    assert len(selected) == 32
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), sharex=True, sharey=True)
    for row, inventory in enumerate(("balanced", "mixed")):
        for column, head in enumerate(("first_lighter", "first_heavier")):
            axis = axes[row, column]
            for timing, (style, marker) in TIMINGS.items():
                values = selected.loc[
                    (selected.inventory == inventory) & (selected["head"] == head) & (selected.timing == timing)
                ].sort_values("token_total")
                assert values.token_total.tolist() == [0.2, 2, 20, 200]
                for readout, color in COLORS.items():
                    axis.plot(
                        values.token_total,
                        values[f"{readout}_gain"],
                        linestyle=style,
                        marker=marker,
                        color=color,
                        linewidth=1.8,
                        markersize=5,
                    )
            axis.axhline(0, color="#555555", linewidth=0.9)
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=0.001)
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            pools = "Equal pool sizes" if inventory == "balanced" else "Pool sizes 1 : 100"
            amplitudes = "(1, 3)" if head == "first_lighter" else "(3, 1)"
            axis.set_title(f"{pools} · benefit amplitudes {amplitudes}", fontsize=12, pad=10)
            axis.set_xticks([0.2, 2, 20, 200], ["0.2", "2", "20", "200"])
            axis.tick_params(labelsize=10)
            if row == 1:
                axis.set_xlabel("Total token budget (invented units)", fontsize=11)
    handles = [
        Line2D([0], [0], color=COLORS["hazard"], lw=2, label="Survival readout"),
        Line2D([0], [0], color=COLORS["additive"], lw=2, label="Additive readout"),
        Line2D([0], [0], color="#444444", ls="--", marker="o", label="First bucket early"),
        Line2D([0], [0], color="#444444", ls="-", marker="s", label="First bucket late"),
    ]
    figure.suptitle("Removing attenuation amplifies both beneficial and harmful timing", fontsize=16, y=0.98)
    figure.supylabel("Gain versus matched tied policy (BPB)", x=0.015, y=0.5, fontsize=12)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False, fontsize=11)
    figure.text(
        0.5,
        0.025,
        "Invented fixed heads; aggregate shares 50:50; pool-mass clock; μ = 3. "
        "Positive gain is better.\nSigned log y-axis, linear within ±0.001 BPB. "
        "Lines connect four evaluated budgets; no fits or policy optimization.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.17, hspace=0.3, wspace=0.17)
    OUTPUT.mkdir(exist_ok=True)
    figure.savefig(OUTPUT / "readout_budget_v1.png", dpi=180)
    figure.savefig(OUTPUT / "readout_budget_v1.pdf")
    plt.close(figure)
    receipt = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": len(selected),
        "scope": "Declared subset for illustration only; full signed tables remain authoritative",
    }
    (OUTPUT / "receipt_v1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
