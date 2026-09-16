# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2.0", "pandas>=2.2"]
# ///
"""Swarm support per bucket against the proposed optima (paper Appendix, reviewer point on identifiability).

For each of the 39 buckets, the materialized epochs of every one of the 280 Qwen3 360M/1.6B swarm runs (one dot per run,
with exact zeros in a separate gutter) against the epochs of the two
headline optima (Uncheatable and OlmoBaseEval Easy; the frozen procedure's unconstrained proposals as the
runtime-quantized mixtures its validation runs trained) and the proportional mixture. A marker outside
its bucket's sampled range is an extrapolation.

usage: uv run plot_delphi_swarm_support_20260907.py [--drive-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
mpl.use("Agg")
REFERENCE = SCRIPT_DIR / "reference_outputs"
FROZEN = REFERENCE / "delphi_offline_selection_20260906"
OPTIMA = (
    REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908" / "runtime_materialization" / "candidate_weights.csv"
)
OUTPUT_DIR = REFERENCE / "delphi_swarm_support_20260907"
CANDIDATES = {
    "lwspu_u_snc_cap06": ("Uncheatable optimum", "#1b9e77", "D"),
    "lwspu_t9_snc_cap08": ("OlmoBaseEval Easy optimum", "#d95f02", "o"),
}
PROPORTIONAL_EPOCHS = 6_325_183_647_689 / 6_986_431_605_135
INK = "#222222"
CORE_COLOR = "#7f97ad"
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
}
POSITIVE_LIMITS = (0.00035, 320)


def bucket_label(name: str) -> str:
    if name.startswith("dolma3_cc/"):
        topic, quality = name.removeprefix("dolma3_cc/").rsplit("_", 1)
        return f"CC {topic.replace('_', ' ')} ({quality})"
    return name.replace("dolmino_", "Dolmino ").replace("dolma3_", "Dolma 3 ").replace("_", " ")


def support_table() -> pd.DataFrame:
    with np.load(FROZEN / "inputs" / "panel.npz", allow_pickle=False) as data:
        buckets = [str(b) for b in data["buckets"]]
        exposures = np.asarray(data["exposures"], dtype=float)
        inventory = np.asarray(data["inventory"], dtype=float)
    optima = pd.read_csv(OPTIMA)
    rows = []
    for j, bucket in enumerate(buckets):
        column = exposures[:, j]
        nonzero = column[column > 0]
        row = {
            "exposures": column.tolist(),
            "bucket": bucket,
            "label": bucket_label(bucket),
            "pool_tokens": 6_325_183_647_689 / inventory[j],
            "zero_share": float(np.mean(column <= 0)),
            "min_nonzero": float(nonzero.min()) if len(nonzero) else np.nan,
            "p05": float(np.percentile(nonzero, 5)) if len(nonzero) else np.nan,
            "p95": float(np.percentile(nonzero, 95)) if len(nonzero) else np.nan,
            "max": float(column.max()),
        }
        for candidate in CANDIDATES:
            weight = optima[(optima.candidate_id == candidate) & (optima.domain == bucket)].weight
            row[candidate] = float(weight.iloc[0]) * inventory[j] if len(weight) else 0.0
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("pool_tokens", ascending=False).reset_index(drop=True)
    for candidate in CANDIDATES:
        table[f"{candidate}_outside"] = (table[candidate] > table["max"]) | (
            (table[candidate] > 0) & (table[candidate] < table["min_nonzero"])
        )
    return table


def draw(table: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(PLOT_STYLE)
    figure, (zero_axis, axis) = plt.subplots(
        ncols=2, figsize=(5.8, 6.7), sharey=True, gridspec_kw={"width_ratios": [0.35, 3.1], "wspace": 0.04}
    )
    y = np.arange(len(table))[::-1]
    rng = np.random.default_rng(0)
    for yi, (_, row) in zip(y, table.iterrows(), strict=True):
        runs = np.asarray(row["exposures"], dtype=float)
        jitter = rng.uniform(-0.28, 0.28, len(runs))
        positive = runs > 0
        axis.scatter(runs[positive], yi + jitter[positive], s=2.5, color=CORE_COLOR, alpha=0.35, linewidth=0, zorder=1)
        zero_axis.scatter(
            np.full(np.sum(~positive), 0.5),
            yi + jitter[~positive],
            s=2.5,
            color=CORE_COLOR,
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
    axis.axvline(PROPORTIONAL_EPOCHS, color="#999999", linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
    axis.text(PROPORTIONAL_EPOCHS * 1.05, len(table) - 0.2, "proportional", fontsize=6, color="#666666", va="bottom")
    for candidate, (label, color, marker) in CANDIDATES.items():
        values = table[candidate].to_numpy(float)
        positive = values > 0
        outside = table[f"{candidate}_outside"].to_numpy(bool)
        zero_axis.scatter(
            np.full(np.sum(~positive), 0.5),
            y[~positive],
            marker=marker,
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.scatter(
            values[positive & ~outside],
            y[positive & ~outside],
            marker=marker,
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label=label,
        )
        if not outside.any():
            continue
        axis.scatter(
            values[outside],
            y[outside],
            marker=marker,
            s=40,
            facecolor="none",
            edgecolor=color,
            linewidth=1.3,
            zorder=5,
            label=f"{label}, outside the sampled range",
        )
    axis.set_xscale("log")
    axis.set_xlim(*POSITIVE_LIMITS)
    axis.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    axis.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    zero_axis.set_xlim(0, 1)
    zero_axis.set_xticks([0.5], ["0"])
    zero_axis.set_yticks(y)
    zero_axis.set_yticklabels(table["label"], fontsize=6)
    zero_axis.set_title("Zero", fontsize=7, pad=6)
    zero_axis.set_facecolor("#f5f7f9")
    axis.set_ylim(-0.8, len(table) - 0.2)
    figure.supxlabel("Materialized epochs (positive values on log scale)", y=0.092, fontsize=7.5)
    axis.tick_params(axis="y", left=False, labelleft=False)
    zero_axis.tick_params(axis="y", length=0)
    for target in (zero_axis, axis):
        for spine in ("top", "right", "left"):
            target.spines[spine].set_visible(False)
    axis.spines["left"].set_visible(True)
    axis.spines["left"].set_color("#c5cbd1")
    axis.grid(axis="x", color="#e5e5e5", linewidth=0.5, zorder=0)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    figure.subplots_adjust(left=0.36, right=0.99, top=0.975, bottom=0.15)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-dir", type=Path, default=None, help="copy the PDF and PNG there as a_swarm_support")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = support_table()
    table.drop(columns=["exposures"]).to_csv(OUTPUT_DIR / "swarm_support.csv", index=False)
    figure = draw(table)
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT_DIR / f"swarm_support.{extension}", dpi=200, bbox_inches="tight")
        if args.drive_dir is not None:
            figure.savefig(args.drive_dir / f"a_swarm_support.{extension}", dpi=200, bbox_inches="tight")
    for candidate in CANDIDATES:
        outside = table[table[f"{candidate}_outside"]]
        print(candidate, "outside the sampled range:", outside["bucket"].tolist())
    print(
        table[["label", "zero_share", "min_nonzero", "p05", "p95", "max", *CANDIDATES]].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
