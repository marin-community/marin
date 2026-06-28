# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot a clean 1e22 summary of old 4plus vs clean-seen endpoint scaling.

The earlier full overlay was too visually dense. This plot keeps the same visual
semantics, but collapses the readout to the two 1e22 quantities that matter:

1. heldout Chinchilla prediction error after fitting through 3e20
2. actual 1e22 validation loss as a function of midtraining token budget

Old 4plus is always dotted with hollow markers. Clean-seen is always solid with
filled markers.

Run:
    uv run --with scipy --with plotly --with pandas --with gcsfs --with matplotlib \
      python scripts/analysis/plot_isotoken_clean_seen_vs_old_endpoint_scaling.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from delphi_isotoken_clean_seen_unified_report import (
    DEFAULT_ISOTOKEN_INPUT,
    DEFAULT_K020_INPUT,
    SERIES,
    fit_all,
    load_points,
)
from delphi_isotoken_endpoint_scaling import DEFAULT_CUTOFF_SCALE, SCALE_ORDER

OUT_DIR = Path("sk_midtrain_analysis_fable")
DEFAULT_OUTPUT = OUT_DIR / "delphi_isotoken_clean_seen_vs_old_endpoint_scaling.png"
OLD_COLOR = "#64748b"
CLEAN_COLOR = "#1877F2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotoken-input", default=DEFAULT_ISOTOKEN_INPUT)
    parser.add_argument("--k020-input", default=DEFAULT_K020_INPUT)
    parser.add_argument("--fit-through-scale", choices=SCALE_ORDER[:-1], default=DEFAULT_CUTOFF_SCALE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def one_e22_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions[predictions["scale"].eq("1e22")].copy()
    return rows.pivot(index="series", columns="target_key", values="error_pct").reset_index()


def one_e22_actuals(points: pd.DataFrame) -> pd.DataFrame:
    rows = points[points["scale"].eq("1e22")].copy()
    rows["clean_minus_old"] = rows["clean_seen_loss"] - rows["old_4plus_loss"]
    return rows


def ordered_one_e22(points: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    actuals = one_e22_actuals(points)
    errors = one_e22_errors(predictions).rename(
        columns={"old_4plus": "old_4plus_error_pct", "clean_seen": "clean_seen_error_pct"}
    )
    merged = actuals.merge(errors, on="series", how="left")
    order = {series.key: index for index, series in enumerate(SERIES)}
    merged["series_order"] = merged["series"].map(order)
    return merged.sort_values("series_order").reset_index(drop=True)


def plot_summary(frame: pd.DataFrame, fit_through_scale: str, output: Path) -> None:
    x = np.arange(len(frame), dtype=float)
    labels = [f"{row.series_short_label}\n{row.nominal_midtrain_tokens_b:.0f}B" for row in frame.itertuples()]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), dpi=170)

    error_ax = axes[0]
    error_ax.axhline(0.0, color="#94a3b8", lw=1.0)
    error_ax.plot(
        x,
        frame["old_4plus_error_pct"],
        color=OLD_COLOR,
        ls=":",
        lw=2.2,
        marker="o",
        ms=8,
        markerfacecolor="white",
        markeredgecolor=OLD_COLOR,
        markeredgewidth=1.8,
        label="old 4plus",
    )
    error_ax.plot(
        x,
        frame["clean_seen_error_pct"],
        color=CLEAN_COLOR,
        ls="-",
        lw=2.2,
        marker="o",
        ms=8,
        markerfacecolor=CLEAN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.9,
        label="clean-seen",
    )
    fixed = frame[~frame["series"].eq("k0p20")]
    error_ax.text(
        0.03,
        0.86,
        (
            f"fit through {fit_through_scale}\n"
            "fixed-token 1e22 errors\n"
            f"old: {fixed['old_4plus_error_pct'].min():+.1f}% to {fixed['old_4plus_error_pct'].max():+.1f}%\n"
            f"clean: {fixed['clean_seen_error_pct'].min():+.1f}% to {fixed['clean_seen_error_pct'].max():+.1f}%"
        ),
        transform=error_ax.transAxes,
        color="#334155",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffffff", "edgecolor": "#d8dee8", "alpha": 0.92},
    )
    k020 = frame[frame["series"].eq("k0p20")].iloc[0]
    k020_x = x[-1]
    error_ax.text(
        k020_x - 0.03,
        float(k020["old_4plus_error_pct"]) + 0.8,
        f"{float(k020['old_4plus_error_pct']):+.1f}%",
        ha="right",
        va="bottom",
        fontsize=9,
        color=OLD_COLOR,
    )
    error_ax.text(
        k020_x - 0.03,
        float(k020["clean_seen_error_pct"]) - 0.8,
        f"{float(k020['clean_seen_error_pct']):+.1f}%",
        ha="right",
        va="top",
        fontsize=9,
        color=CLEAN_COLOR,
    )
    error_ax.set_title("1e22 heldout prediction error", fontsize=12)
    error_ax.set_ylabel("prediction error (%)")
    error_ax.set_xticks(x, labels)
    error_ax.set_ylim(
        min(-6.0, float(frame[["old_4plus_error_pct", "clean_seen_error_pct"]].min().min()) - 2.0),
        max(20.0, float(frame[["old_4plus_error_pct", "clean_seen_error_pct"]].max().max()) + 2.0),
    )
    error_ax.grid(True, axis="y", alpha=0.24)

    loss_ax = axes[1]
    loss_ax.plot(
        x,
        frame["old_4plus_loss"],
        color=OLD_COLOR,
        ls=":",
        lw=2.2,
        marker="o",
        ms=8,
        markerfacecolor="white",
        markeredgecolor=OLD_COLOR,
        markeredgewidth=1.8,
        label="old 4plus",
    )
    loss_ax.plot(
        x,
        frame["clean_seen_loss"],
        color=CLEAN_COLOR,
        ls="-",
        lw=2.2,
        marker="o",
        ms=8,
        markerfacecolor=CLEAN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.9,
        label="clean-seen",
    )
    for xpos, gap in zip(x, frame["clean_minus_old"].to_numpy(dtype=float), strict=True):
        loss_ax.text(xpos, frame["clean_seen_loss"].iloc[int(xpos)] + 0.012, f"+{gap:.3f}", ha="center", fontsize=8.5)
    loss_ax.set_title("1e22 actual validation loss", fontsize=12)
    loss_ax.set_ylabel("loss")
    loss_ax.set_xticks(x, labels)
    loss_ax.set_ylim(float(frame["old_4plus_loss"].min()) - 0.035, float(frame["clean_seen_loss"].max()) + 0.045)
    loss_ax.grid(True, axis="y", alpha=0.24)

    style_handles = [
        mlines.Line2D(
            [],
            [],
            color=OLD_COLOR,
            ls=":",
            lw=2.2,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=OLD_COLOR,
            markeredgewidth=1.8,
            label="old 4plus: dotted + hollow",
        ),
        mlines.Line2D(
            [],
            [],
            color=CLEAN_COLOR,
            ls="-",
            lw=2.2,
            marker="o",
            markerfacecolor=CLEAN_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="clean-seen: solid + filled",
        ),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Delphi 1e22 endpoint readout: old 4plus vs clean-seen validation",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    points = load_points(args.isotoken_input, args.k020_input, args.fit_through_scale)
    predictions, _ = fit_all(points, args.fit_through_scale)
    frame = ordered_one_e22(points, predictions)
    plot_summary(frame, args.fit_through_scale, args.output)
    print(f"wrote {args.output}")
    print(
        frame[
            [
                "series_label",
                "old_4plus_loss",
                "clean_seen_loss",
                "old_4plus_error_pct",
                "clean_seen_error_pct",
                "clean_minus_old",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
