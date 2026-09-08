# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How the bounded log-deficit link fails the two-bucket StarCoder gate (round 2, 2026-09-02).

For each of the 45 tied curves: observed Programming-Languages BPB against the StarCoder fraction, the
in-sample fit of WSPU (identity link), of WSPU with the bounded log-deficit link, and of the benchmark's DSP
on a dense grid, and every model's out-of-fold prediction at the held-out points (hollow markers, from the
screen tier's component predictions). Panel titles carry the out-of-fold RMSE of the three models and are
red where the link is worse than DSP out of fold. A second figure summarizes the per-curve out-of-fold RMSE.

usage: uv run python plot_starcoder_link_gate_20260906.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

BENCHMARK_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_observatory_benchmark_20260902"
ATLAS_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_all_tied_curves_canonical_dsp_20260902"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_link_gate_plots_20260906"
MODELS = (
    ("weibull_softplus_unscaled", "WSPU (identity link)", "#178A72"),
    ("weibull_softplus_unscaled@log_deficit_bounded_link", "WSPU, bounded log-deficit link", "#D55E00"),
    ("dsp_total_exposure", "DSP (benchmark)", "#6C6F7D"),
)
FAMILY_COLORS = {
    "fixed_model_token_ladder": "#0072B2",
    "matched_nd": "#009E73",
    "dense_horizon_replay": "#CC79A7",
    "coupled_lr_onset": "#E69F00",
}
FAMILY_SHORT = {
    "fixed_model_token_ladder": "Token ladder",
    "matched_nd": "Matched ladder",
    "dense_horizon_replay": "Replay",
    "coupled_lr_onset": "Onset",
}
INK = "#111111"
GRID = "#b8b8b8"
DENSE_POINTS = 121
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "text.usetex": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}
DPI = 220


@dataclasses.dataclass(frozen=True)
class CurveFits:
    curve_id: str
    number: int
    ref: str
    family: str
    label: str
    p_observed: np.ndarray
    observed: np.ndarray
    p_dense: np.ndarray
    dense: dict[str, np.ndarray]
    oof: dict[str, np.ndarray]
    oof_rmse: dict[str, float]


def dense_features(features, starcoder_column: int, grid: np.ndarray):
    """The panel's features evaluated at every StarCoder fraction of ``grid``."""
    other = 1 - starcoder_column
    weights = np.zeros((len(grid), 2))
    weights[:, starcoder_column] = grid
    weights[:, other] = 1 - grid
    return dataclasses.replace(
        features,
        weights=weights,
        exposures=weights * features.inventory[None, :],
        label=features.label + "|dense",
    )


def oof_predictions() -> pd.DataFrame:
    frame = pd.read_csv(BENCHMARK_DIR / "screen" / "component_predictions.csv")
    frame = frame[
        frame["panel"].str.startswith(bench.STARCODER_PANEL_PREFIX) & frame["model"].isin([m for m, _, _ in MODELS])
    ]
    return frame[frame["repeat"].eq(frame["repeat"].min())]


def fit_curves(curve_ids: tuple[str, ...], reference: pd.DataFrame, oof: pd.DataFrame) -> list[CurveFits]:
    grid = np.linspace(0.0, 1.0, DENSE_POINTS)
    fits: list[CurveFits] = []
    for curve_id in curve_ids:
        panel = bench.load_panel(f"{bench.STARCODER_PANEL_PREFIX}{curve_id}")
        features = panel.features
        outcome = panel.groups[0].outcomes[:, 0].astype(float)
        train = np.arange(panel.rows)
        inner = bench.heldout_inner_folds(panel)
        starcoder_column = int(np.argmax(features.exposures.max(0)))
        p_observed = features.weights[:, starcoder_column]
        dense = dense_features(features, starcoder_column, grid)
        dense_predictions: dict[str, np.ndarray] = {}
        oof_values: dict[str, np.ndarray] = {}
        oof_rmse: dict[str, float] = {}
        panel_oof = oof[oof["panel"].eq(f"{bench.STARCODER_PANEL_PREFIX}{curve_id}")]
        for model_id, _label, _color in MODELS:
            entry = registry.ENTRY_BY_ID[model_id]
            transformed = registry.apply_transform(features, entry)
            model = entry.build(transformed)
            fit = model.fit(transformed, outcome, train, inner, 0)
            dense_predictions[model_id] = np.asarray(
                model.predict(fit, registry.apply_transform(dense, entry), np.arange(len(grid))), float
            )
            rows = panel_oof[panel_oof["model"].eq(model_id)].sort_values("row_index")
            values = np.full(panel.rows, np.nan)
            values[rows["row_index"].to_numpy(int)] = rows["prediction"].to_numpy(float)
            oof_values[model_id] = values
            oof_rmse[model_id] = float(np.sqrt(np.nanmean((values - outcome) ** 2)))
        meta = reference.loc[curve_id]
        fits.append(
            CurveFits(
                curve_id,
                int(meta["curve_number"]),
                str(meta["curve_ref"]),
                str(meta["family"]),
                f"{FAMILY_SHORT[str(meta['family'])]}: {meta['curve_label']}",
                p_observed,
                outcome,
                grid,
                dense_predictions,
                oof_values,
                oof_rmse,
            )
        )
        print(f"fitted {meta['curve_ref']} {curve_id}", flush=True)
    return sorted(fits, key=lambda item: item.number)


def draw_curve(axis: plt.Axes, curve: CurveFits) -> None:
    axis.scatter(curve.p_observed, curve.observed, s=9, color=INK, zorder=5, label="observed")
    for model_id, label, color in MODELS:
        axis.plot(
            curve.p_dense, curve.dense[model_id], color=color, linewidth=1.1, zorder=3, label=f"{label}, in-sample fit"
        )
        axis.scatter(
            curve.p_observed,
            curve.oof[model_id],
            s=14,
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
            zorder=4,
            label=f"{label}, out of fold",
        )
    low = float(np.nanmin(curve.observed))
    high = float(np.nanmax(curve.observed))
    span = max(high - low, 0.05)
    axis.set_ylim(low - 0.15 * span, high + 0.35 * span)
    axis.set_xlim(-0.02, 1.02)
    axis.grid(True, axis="y", color=GRID, alpha=0.6, linewidth=0.5)
    axis.tick_params(labelsize=6, colors=INK)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    link, dsp = MODELS[1][0], MODELS[2][0]
    worse = curve.oof_rmse[link] > curve.oof_rmse[dsp]
    rmse = "  ".join(f"{curve.oof_rmse[model_id]:.3f}" for model_id, _, _ in MODELS)
    axis.set_title(
        f"{curve.ref} {curve.label}\nOOF RMSE {rmse}",
        fontsize=6,
        loc="left",
        color="#B22222" if worse else INK,
        fontweight="bold" if worse else "normal",
    )


def build_grid(fits: list[CurveFits]) -> plt.Figure:
    columns = 5
    rows = -(-len(fits) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13.5, 2.35 * rows), squeeze=False)
    for index, curve in enumerate(fits):
        draw_curve(axes[index // columns][index % columns], curve)
    for index in range(len(fits), rows * columns):
        axes[index // columns][index % columns].axis("off")
    for axis in axes[-1]:
        axis.set_xlabel("StarCoder fraction p", fontsize=7)
    for row in axes:
        row[0].set_ylabel("Programming Languages BPB", fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.suptitle(
        "Two-bucket StarCoder gate: WSPU with and without the bounded log-deficit link against DSP "
        "(titles: out-of-fold RMSE in that order; red where the link is worse than DSP)",
        fontsize=10,
        y=0.997,
    )
    # Legend under the title, in two rows: lines are in-sample fits, hollow markers out-of-fold predictions.
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.988))
    figure.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.4, w_pad=1.0)
    return figure


def build_summary(fits: list[CurveFits]) -> plt.Figure:
    wspu, link, dsp = (model_id for model_id, _, _ in MODELS)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for curve in fits:
        color = FAMILY_COLORS[curve.family]
        left.scatter(
            curve.oof_rmse[wspu], curve.oof_rmse[link], s=22, color=color, edgecolor="white", linewidth=0.5, zorder=4
        )
    limits = (0.002, 0.7)
    left.plot(limits, limits, color=INK, linewidth=0.8, linestyle=(0, (4, 2)))
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(limits)
    left.set_ylim(limits)
    left.set_xlabel("WSPU out-of-fold RMSE (BPB)")
    left.set_ylabel("Bounded log-deficit link out-of-fold RMSE (BPB)")
    worse = sum(curve.oof_rmse[link] > curve.oof_rmse[wspu] for curve in fits)
    left.set_title(
        f"A. Link against WSPU: worse on {worse} of {len(fits)} curves", loc="left", fontsize=8, fontweight="bold"
    )
    for family, color in FAMILY_COLORS.items():
        left.scatter([], [], color=color, label=family.replace("_", " "))
    left.legend(frameon=False, fontsize=7, loc="upper left")
    ordered = sorted(fits, key=lambda curve: curve.oof_rmse[link] / curve.oof_rmse[dsp])
    ratios = [curve.oof_rmse[link] / curve.oof_rmse[dsp] for curve in ordered]
    colors = [FAMILY_COLORS[curve.family] for curve in ordered]
    right.bar(range(len(ordered)), ratios, color=colors, width=0.8)
    right.axhline(1.0, color=INK, linewidth=0.8, linestyle=(0, (4, 2)))
    right.set_yscale("log")
    right.set_xticks(range(len(ordered)))
    right.set_xticklabels([curve.ref for curve in ordered], rotation=90, fontsize=5.5)
    right.set_ylabel("Link RMSE / DSP RMSE, out of fold")
    losing = sum(ratio > 1 for ratio in ratios)
    right.set_title(
        f"B. Link against DSP: worse on {losing} of {len(fits)} curves", loc="left", fontsize=8, fontweight="bold"
    )
    for axis in (left, right):
        axis.grid(True, axis="y", color=GRID, alpha=0.6, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    plt.rcParams.update(PLOT_STYLE)
    reference = pd.read_csv(ATLAS_DIR / "curve_reference.csv").set_index("curve_id")
    fits = fit_curves(bench.tier_plan("screen").curves, reference, oof_predictions())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "curve": curve.number,
            "curve_ref": curve.ref,
            "curve_id": curve.curve_id,
            "family": curve.family,
            **{f"oof_rmse_{model_id}": curve.oof_rmse[model_id] for model_id, _, _ in MODELS},
        }
        for curve in fits
    )
    table.to_csv(args.output_dir / "link_gate_oof_rmse.csv", index=False)
    for name, builder in (("link_gate_curves", build_grid), ("link_gate_summary", build_summary)):
        figure = builder(fits)
        for extension in ("png", "pdf"):
            figure.savefig(args.output_dir / f"{name}.{extension}", dpi=DPI, bbox_inches="tight")
        plt.close(figure)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
