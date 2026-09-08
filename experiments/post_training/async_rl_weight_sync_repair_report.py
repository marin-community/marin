# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the two realized Snowball traces with all lag samples retained."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def render(captures: Path, output: Path) -> None:
    inputs = [
        "snowball-history.json",
        "snowball-finelog-raw.json",
        "snowball-repair-history.json",
        "snowball-repair-finelog-raw.json",
        "snowball-repair-finelog-audit.json",
    ]
    data = {name: json.loads((captures / name).read_text()) for name in inputs}
    audit = data["snowball-repair-finelog-audit.json"]
    assert audit["status"] == "PASS" and len(audit["driver_max_fold_crosschecks"]) == 40
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    rng = np.random.default_rng(0)
    summaries = {}
    for index, (name, prefix, color) in enumerate(
        [
            ("Original trace", "snowball", "#63738b"),
            ("Repairs", "snowball-repair", "#007a71"),
        ]
    ):
        history = data[prefix + "-history.json"]["rows"]
        syncs = [row["timing/sync_weights"] for row in history if "timing/sync_weights" in row]
        pauses = [row["timing/weight_pause"] for row in history if "timing/weight_pause" in row]
        lag = [row["value"] for row in data[prefix + "-finelog-raw.json"]["lag"]]
        assert len(syncs) == len(pauses) == 5 and min(pauses) >= 5.0
        assert lag and all(np.isfinite(value) and value >= 0 for value in lag)
        for axis, values in zip(axes, [syncs, lag], strict=True):
            axis.scatter(index + rng.uniform(-0.13, 0.13, len(values)), values, color=color, s=25, alpha=0.6)
        axes[0].plot([index - 0.2, index + 0.2], [np.median(syncs)] * 2, color=color, linewidth=3)
        summaries[name] = {
            "sync_seconds": syncs,
            "pause_seconds": pauses,
            "lag_samples": len(lag),
            "lag_max_seconds": max(lag),
        }
    axes[0].set(title="Whole weight sync: five updates per run", ylabel="Wall seconds")
    axes[1].set(title="Every event-loop lag sample", ylabel="Lag seconds")
    axes[1].axhline(2.0, color="#ad362e", linestyle="--", linewidth=1, label="Strict gate: below 2 s")
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.set_xticks([0, 1], ["Original trace", "Repairs"])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Snowball retains the five-second pause in both traces", fontsize=12)
    fig.text(
        0.5,
        -0.04,
        "Two realized runs; dots are observations, not independent seeds. Timers overlap and are not additive.",
        ha="center",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "snowball-repair.svg", bbox_inches="tight")
    fig.savefig(output / "snowball-repair.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    result = {
        "summaries": summaries,
        "input_sha256": {name: hashlib.sha256((captures / name).read_bytes()).hexdigest() for name in inputs},
    }
    (output / "plot-receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("SNOWBALL_REPAIR_PLOT_PASS syncs_per_arm=5 all_lag_samples_retained=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.captures, args.output)
