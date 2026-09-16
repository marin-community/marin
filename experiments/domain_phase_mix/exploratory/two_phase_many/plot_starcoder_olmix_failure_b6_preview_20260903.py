# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "pandas"]
# ///

"""Build the paper B6 figure comparing non-monotone and OLMix curve fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments.domain_phase_mix.exploratory.two_phase_many.plot_starcoder_matched_scaling_b5_preview_20260903 import (
    D_SCALING_CURVES,
    DEFAULT_ATLAS_DIR,
    DEFAULT_DESIGN,
    DEFAULT_FIXED_TPP_DESIGN,
    DEFAULT_FIXED_TPP_DIR,
    INK,
    PLOT_STYLE,
    REFERENCE_OUTPUTS,
    STATIC_DPI,
    CurveMetadata,
    load_inputs,
    style_axis,
)

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_olmix_failure_b6_figure_20260903"
FIGURE_SIZE = (4.35, 3.85)
TITLE = "Monotone fits miss repetition-induced optima"
CURVE_REFS = D_SCALING_CURVES
COLORS = ("#e377c2", "#ff7f0e", "#d62728", "#8c564b")
Y_LIMIT = (0.78, 1.35)
Y_TICKS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_dense_fits(atlas_dir: Path) -> pd.DataFrame:
    dense = pd.read_csv(atlas_dir / "dense_curves.csv")
    selected = set(CURVE_REFS)
    dense = dense.loc[dense["curve_ref"].isin(selected)].copy()
    counts = dense.groupby("curve_ref").size()
    if set(counts.index) != selected or counts.nunique() != 1:
        raise ValueError(f"Dense fit grid is incomplete: {counts.to_dict()}")
    return dense.sort_values(["curve_number", "starcoder_weight"])


def add_fit_legend(axis) -> None:
    handles = (
        Line2D([], [], color=INK, linewidth=1.8, label="WSPU"),
        Line2D([], [], color=INK, linewidth=2.0, linestyle=(0, (4, 2)), label="Olmix"),
    )
    legend = axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#b8b8b8",
        fontsize=6.8,
        handlelength=2.35,
        columnspacing=1.25,
        handletextpad=0.5,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(0.5)


def add_comparison_curves(
    axis,
    *,
    curve_refs: tuple[str, ...],
    colors: tuple[str, ...],
    observations: pd.DataFrame,
    dense_fits: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
) -> None:
    for curve_ref, color in zip(curve_refs, colors, strict=True):
        observed = observations.loc[observations["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        dense = dense_fits.loc[dense_fits["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        axis.plot(
            dense["starcoder_weight"],
            dense["weibull_softplus_unscaled_full_fit_prediction_bpb"],
            color=color,
            linewidth=1.7,
            zorder=2,
        )
        axis.plot(
            dense["starcoder_weight"],
            dense["olmix_full_fit_prediction_bpb"],
            color=color,
            linewidth=1.45,
            linestyle=(0, (4, 2)),
            path_effects=[path_effects.Stroke(linewidth=3.3, foreground="black"), path_effects.Normal()],
            zorder=3,
        )
        axis.scatter(
            observed["starcoder_weight"],
            observed["observed_bpb"],
            facecolor="white",
            edgecolor=color,
            linewidth=0.85,
            s=15,
            zorder=5,
        )
        curve = metadata[curve_ref]
        axis.scatter(
            [curve.observed_optimum_weight],
            [curve.observed_optimum_bpb],
            color=color,
            edgecolor=INK,
            linewidth=0.65,
            marker="*",
            s=64,
            zorder=6,
        )


def build_figure(
    *,
    observations: pd.DataFrame,
    dense_fits: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
) -> Figure:
    with plt.rc_context(PLOT_STYLE):
        figure = plt.figure(figsize=FIGURE_SIZE)
        axis = figure.add_axes([0.145, 0.145, 0.83, 0.705])
        figure.text(0.55, 0.982, TITLE, ha="center", va="top", fontsize=10.5, fontweight="bold")
        add_comparison_curves(
            axis,
            curve_refs=CURVE_REFS,
            colors=COLORS,
            observations=observations,
            dense_fits=dense_fits,
            metadata=metadata,
        )
        style_axis(axis, metadata[CURVE_REFS[0]].epoch_scale)
        axis.set_ylim(*Y_LIMIT)
        axis.set_yticks(Y_TICKS)
        add_fit_legend(axis)
        return figure


def write_index(output_dir: Path) -> None:
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>B6 monotone-fit capacity figure</title>
<style>
body { margin:0; padding:34px; background:#f4f4f1; color:#111; font-family:Georgia,serif; }
main { max-width:760px; margin:auto; }
h1 { margin:0 0 10px; font-size:2rem; }
p { color:#333; font:1rem/1.5 "Helvetica Neue",sans-serif; }
.figure { margin-top:24px; padding:18px; border:1px solid #ccc; background:white; box-shadow:0 5px 18px #0000000d; }
.figure img { display:block; width:100%; height:auto; }
.caption, .links { margin-top:10px; font:.9rem/1.45 "Helvetica Neue",sans-serif; }
a { color:#174a75; }
@media (max-width:820px) { body { padding:18px; } }
</style>
</head>
<body><main>
<h1>B6 monotone-fit capacity figure</h1>
<p>The four curves hold model size fixed while increasing the token budget. They include both edge trends and
interior optima, rather than selecting only visually symmetric U-shaped examples.</p>
<div class="figure"><img src="figure.png" alt="Non-monotone and OLMix fits to four StarCoder response curves">
<div class="caption">Open circles are observations, stars are observed minima, solid lines are WSPU fits, and
dashed lines are OLMix fits. Each model is fit independently to all observations on each curve. This is an
in-sample capacity diagnostic, not a held-out comparison. WSPU has eight available response parameters in this
two-bucket setting; OLMix has three.</div>
<div class="links"><a href="figure.pdf">Open vector PDF</a></div>
</div>
</main></body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def write_data(output_dir: Path, observations: pd.DataFrame, dense_fits: pd.DataFrame) -> None:
    selected_observations = observations.loc[observations["curve_ref"].isin(CURVE_REFS)].copy()
    selected_observations.to_csv(output_dir / "observations.csv", index=False)
    dense_fits.to_csv(output_dir / "dense_fits.csv", index=False)


def main() -> None:
    args = parse_args()
    observations, metadata = load_inputs(args.atlas_dir, args.design, DEFAULT_FIXED_TPP_DIR, DEFAULT_FIXED_TPP_DESIGN)
    dense_fits = load_dense_fits(args.atlas_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(observations=observations, dense_fits=dense_fits, metadata=metadata)
    figure.savefig(args.output_dir / "figure.png", dpi=STATIC_DPI)
    figure.savefig(args.output_dir / "figure.pdf")
    plt.close(figure)
    write_index(args.output_dir)
    write_data(args.output_dir, observations, dense_fits)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
