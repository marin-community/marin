# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy", "pandas"]
# ///

"""Draw the standalone in-sample response-capacity figure from the saved MARINER-head fits.

Uses the four fits already computed for the motivation figure; it does not refit models. The saved parameters
are checked against the paper's floor-plus-exponential Weibull/softplus response before plotting. Their
per-curve calibration uses a pinned proportional run, leave-one-out shape/ridge/multiplier selection, and no
noise margin because these curves have no repeats. This is a response-capacity illustration, not validation
of the end-to-end swarm fitting procedure.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "reference_outputs"
FIT_DIR = REFERENCE_OUTPUTS / "motivation_composite_figure_20260908"
OBSERVATIONS = REFERENCE_OUTPUTS / "starcoder_all_tied_curves_canonical_dsp_20260902" / "predictions.csv"
OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_olmix_failure_b6_compact_20260908"
CURVE_REFS = ("C05", "C10", "C13", "C14")
COLORS = ("#0d0887", "#9c179e", "#ed7953", "#b87900")
INK = "#243549"
OLMIX_LINEWIDTH = 1.45
OLMIX_DASH_LENGTH = 4.35
OLMIX_END_BORDER = 0.55
OLMIX_DASH_GAP = 2.9
OLMIX_STYLE = {
    "linewidth": OLMIX_LINEWIDTH,
    "linestyle": (-OLMIX_END_BORDER, (OLMIX_DASH_LENGTH, OLMIX_DASH_GAP + 2 * OLMIX_END_BORDER)),
    "dash_capstyle": "butt",
    # Draw black end bars separately to avoid an antialiased dark fringe along the colored edges.
    "path_effects": [
        path_effects.Stroke(
            linewidth=OLMIX_LINEWIDTH,
            foreground="black",
            capstyle="butt",
            dashes={
                "dash_offset": 0,
                "dash_list": (OLMIX_END_BORDER, OLMIX_DASH_LENGTH, OLMIX_END_BORDER, OLMIX_DASH_GAP),
            },
        ),
        path_effects.Normal(),
    ],
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.formatter.use_mathtext": False,
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "lines.scale_dashes": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def verify_saved_response(fits: pd.DataFrame, parameters: list[dict], configurations: pd.DataFrame) -> float:
    """Verify that the plotted predictions use the paper's response equation, without an active numerical cap."""
    maximum_error = 0.0
    assert set(fits["curve_ref"]) == set(CURVE_REFS)
    assert {record["curve_ref"] for record in parameters} == set(CURVE_REFS)
    for record in parameters:
        curve_ref = record["curve_ref"]
        curve = fits.loc[fits["curve_ref"].eq(curve_ref)]
        configuration = configurations.loc[
            configurations["panel"].eq("C") & configurations["curve_ref"].eq(curve_ref)
        ].iloc[0]
        fraction = curve["starcoder_weight"].to_numpy(float)
        exposures = np.column_stack([1.0 - fraction, fraction * configuration["epochs_at_p1"]])
        shape = record["shape"]
        benefit = 1.0 - np.exp(-np.power(shape["rate"] * exposures, shape["power"]))
        harm = np.logaddexp(np.log1p(exposures) - shape["threshold"], 0.0) ** 2
        matrix = np.hstack([-benefit, harm])
        coefficients = np.asarray(record["coefficients"])
        assert (coefficients >= 0).all()
        linear_predictor = record["intercept"] + matrix @ coefficients
        assert np.max(np.abs(linear_predictor)) < 30.0
        prediction = record["floor"] + np.exp(linear_predictor)
        error = float(np.max(np.abs(prediction - curve["mariner_bpb"].to_numpy(float))))
        maximum_error = max(maximum_error, error)
        np.testing.assert_allclose(prediction, curve["mariner_bpb"], rtol=1e-12, atol=1e-12)
        olmix_prediction = np.exp(record["olmix_log_c"]) + np.exp(
            np.column_stack([1.0 - fraction, fraction]) @ np.asarray(record["olmix_coefficients"])
        )
        np.testing.assert_allclose(olmix_prediction, curve["olmix_bpb"], rtol=1e-12, atol=1e-12)
    return maximum_error


def build_figure(observations: pd.DataFrame, fits: pd.DataFrame) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(2.65, 2.05))
        figure.subplots_adjust(left=0.165, right=0.985, bottom=0.205, top=0.975)
        for curve_ref, color in zip(CURVE_REFS, COLORS, strict=True):
            observed = observations.loc[observations["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
            dense = fits.loc[fits["curve_ref"].eq(curve_ref)]
            assert len(observed) == 15
            assert np.isclose(observed["starcoder_weight"].min(), dense["starcoder_weight"].min())
            assert np.isclose(observed["starcoder_weight"].max(), dense["starcoder_weight"].max())
            axis.plot(dense["starcoder_weight"], dense["mariner_bpb"], color=color, linewidth=1.2, zorder=3)
            axis.plot(
                dense["starcoder_weight"],
                dense["olmix_bpb"],
                color=color,
                zorder=4,
                **OLMIX_STYLE,
            )
            axis.scatter(
                observed["starcoder_weight"],
                observed["observed_bpb"],
                facecolor="white",
                edgecolor=color,
                linewidth=0.55,
                s=8,
                zorder=5,
            )
            minimum = observed.loc[observed["observed_bpb"].idxmin()]
            axis.scatter(
                [minimum["starcoder_weight"]],
                [minimum["observed_bpb"]],
                color=color,
                edgecolor=INK,
                linewidth=0.45,
                marker="*",
                s=33,
                zorder=6,
            )
        axis.set_xlim(0.0, 0.92)
        axis.set_ylim(0.78, 1.32)
        axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        axis.set_yticks([0.8, 1.0, 1.2])
        axis.set_xlabel(r"StarCoder mixture fraction, $p$", fontsize=7, labelpad=2)
        axis.set_ylabel("Loss (BPB)", fontsize=7, labelpad=2)
        axis.tick_params(labelsize=6.5, length=2.5, width=0.6, pad=2)
        axis.grid(color="#D9E1E7", linewidth=0.5, linestyle="-")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(0.65)
            axis.spines[side].set_color(INK)
        axis.legend(
            handles=[
                Line2D([], [], color=INK, linewidth=1.2, label="MARINER"),
                Line2D([], [], color=INK, label="Olmix", **OLMIX_STYLE),
            ],
            loc="upper left",
            bbox_to_anchor=(0.035, 0.98),
            ncol=2,
            frameon=False,
            fontsize=7,
            handlelength=2.1,
            columnspacing=1.2,
            handletextpad=0.45,
            borderaxespad=0,
            numpoints=1,
        )
        return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=FIT_DIR)
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path)
    args = parser.parse_args()
    fits = pd.read_csv(args.fit_dir / "panel_c_fits.csv")
    parameters = json.loads((args.fit_dir / "panel_c_parameters.json").read_text())
    configurations = pd.read_csv(args.fit_dir / "configurations.csv")
    observations = pd.read_csv(args.observations)
    maximum_error = verify_saved_response(fits, parameters, configurations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(observations, fits)
    for extension in ("pdf", "png"):
        output = args.output_dir / f"figure.{extension}"
        with plt.rc_context(PLOT_STYLE):
            figure.savefig(output, dpi=300)
        if args.drive_dir is not None:
            shutil.copyfile(output, args.drive_dir / f"b6_olmix_capacity_failure.{extension}")
    plt.close(figure)
    print(f"Saved response-equation parity: max absolute error {maximum_error:.3g} BPB")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
