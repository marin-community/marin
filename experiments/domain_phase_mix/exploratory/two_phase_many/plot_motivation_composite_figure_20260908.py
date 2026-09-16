# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The paper's motivation figure: four controlled two-bucket experiments in one 2x2 panel.

(A) Fixed model size and token budget, five degrees of StarCoder downsampling (full pool, half the target's
repetition, the target's, twice and four times the target's): the observed optimum moves in mixture fraction
and stays approximately aligned in epochs. (B) Model size scaled at fixed token budget, (C) token budget scaled
at fixed model size, (D) both scaled
at fixed tokens per parameter, all at the target's repetition and along matched-compute rungs. All four panels
show observations. This script also saves per-curve fits for the separate response-capacity figure:
MARINER's task head (floored log-deficit with the Weibull benefit and softplus harm of the paper,
nonnegative amplitudes, shape and ridge by leave-one-out cross-validation, floor anchored at the curve's
proportional-mixture run) and Olmix's log-linear head (Huber fit, 48 multistarts), fitted to that curve's
observations only: an in-sample illustration of response capacity, not a validation.

Data: the StarCoder replay atlas (panel A and the fixed-D / fixed-N ladders) and the fixed-TPP diagonal, through
the loaders of the two earlier preview scripts. Writes figure.{png,pdf}, panel_c_fits.csv,
panel_c_parameters.json, configurations.csv and configurations_table.tex (the appendix table of every plotted
curve) to the output directory, and copies the figure to --drive-dir as b5_motivation_composite.{png,pdf}.

usage: uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.plot_motivation_composite_figure_20260908
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, nnls

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_starcoder_matched_scaling_b5_preview_20260903 as ladders,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_starcoder_replay_background_figure_20260902 as replay,
)

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "motivation_composite_figure_20260908"
DRIVE_STEM = "b5_motivation_composite"

# Panel A: the replay atlas curves at 1.00B tokens; multiplier = the subset's repetition relative to the target's.
DOWNSAMPLING_CURVES = (("C15", None), ("C18", 0.5), ("C19", 1.0), ("C20", 2.0), ("C21", 4.0))
DOWNSAMPLING_COLORS = ("#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2")
# Panels B to D: one colour per matched-compute rung, shared across the three ladders.
RUNG_COLORS = ("#0d0887", "#9c179e", "#ed7953", "#f0a800")
RUNG_LABELS = ("0.88e18", "1.7e18", "3.4e18", "6.5e18")
LADDER_PANELS = (
    ("B", r"Scale $N$; fixed $D$", ladders.N_SCALING_CURVES),
    ("C", r"Scale $D$; fixed $N$", ladders.D_SCALING_CURVES),
    ("D", r"Scale $N$ and $D$; fixed TPP", ladders.FIXED_TPP_CURVES),
)
NEMOTRON_POOL_TOKENS = 5_730_000_000_000  # Nemotron-CC pool
LADDER_EXPOSURE_REFERENCE_TOKENS = 5_730_000_000_000  # the ladders' target budget (Appendix, two-bucket ladders)
REPLAY_MODEL_PARAMETERS = 210_000_000  # every panel-A curve trains the 210M model
DENSE_POINTS = 400  # fits are drawn over each curve's observed range of p only

INK = "#111111"
GRID = "#b8b8b8"
FIGURE_SIZE = (5.5, 4.1)
A_Y_LIMIT = (0.62, 4.3)
LADDER_Y_LIMIT = (0.84, 1.43)
BOTTOM_Y_LIMIT = (0.78, 1.35)
LADDER_X_LIMIT = (0.0, 0.92)
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}
DPI = 300

# MARINER's per-task fitting rules (the constants of the frozen procedure), applied to one two-bucket curve.
SHAPES = tuple(
    {"rate": rate, "power": power, "threshold": threshold}
    for rate in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    for power in (0.3, 0.5, 0.7, 1.0)
    for threshold in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
)
RIDGE_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0)
GAMMA_PROVISIONAL = 2.5
GAMMA_BOUNDS = (1.0, 6.0)
GAMMA_FLAT_FRACTION = 0.1
GAMMA_FLAT_DEFAULT = 1.5
DEFICIT_FLOOR = 1e-9
LOG_CLIP = 30.0
HUBER_DELTA = 0.01
OLMIX_STARTS = 48


@dataclass(frozen=True)
class MarinerCurveFit:
    shape: dict[str, float]
    ridge: float
    gamma: float
    gamma_flat: bool
    floor: float
    intercept: float
    coefficients: np.ndarray


@dataclass(frozen=True)
class CurveFit:
    curve_ref: str
    shape: dict[str, float]
    ridge: float
    gamma: float
    gamma_flat: bool
    floor: float
    intercept: float
    coefficients: tuple[float, ...]
    olmix_log_c: float
    olmix_coefficients: tuple[float, ...]


def benefit(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return 1.0 - np.exp(-np.power(rate * np.maximum(exposure, 0.0), power))


def harm(exposure: np.ndarray, threshold: float) -> np.ndarray:
    return np.logaddexp(np.log1p(np.maximum(exposure, 0.0)) - threshold, 0.0) ** 2


def design_matrix(exposures: np.ndarray, shape: dict[str, float]) -> np.ndarray:
    return np.hstack([-benefit(exposures, shape["rate"], shape["power"]), harm(exposures, shape["threshold"])])


def nonnegative_solve(matrix: np.ndarray, target: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    width = matrix.shape[1]
    design_mean = matrix.mean(axis=0)
    target_mean = float(target.mean())
    rows = matrix - design_mean[None, :]
    rhs = target - target_mean
    if ridge > 0.0:
        rows = np.vstack([rows, np.diag(np.full(width, math.sqrt(ridge)))])
        rhs = np.concatenate([rhs, np.zeros(width)])
    coefficients, _ = nnls(rows, rhs, maxiter=50 * width)
    return target_mean - float(design_mean @ coefficients), coefficients


def floor_value(anchor: float, response: np.ndarray, gamma: float) -> float:
    """The paper's floor without the noise margin: the ladders have no repeated runs to estimate one."""
    return anchor - gamma * (anchor - float(np.min(response)))


def predict(matrix: np.ndarray, floor: float, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    return floor + np.exp(np.clip(intercept + matrix @ coefficients, -LOG_CLIP, LOG_CLIP))


def leave_one_out_rmse(matrix: np.ndarray, response: np.ndarray, anchor_index: int, ridge: float, gamma: float) -> float:
    """Leave-one-out error over the curve's runs; the anchor run stays in every training fold and is never scored."""
    errors = []
    for held in range(len(response)):
        if held == anchor_index:
            continue
        train = np.array([i for i in range(len(response)) if i != held])
        floor = floor_value(response[anchor_index], response[train], gamma)
        target = np.log(np.maximum(response[train] - floor, DEFICIT_FLOOR))
        intercept, coefficients = nonnegative_solve(matrix[train], target, ridge)
        prediction = predict(matrix[[held]], floor, intercept, coefficients)
        if not np.isfinite(prediction).all():
            return float("inf")
        errors.append(float(prediction[0] - response[held]))
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_mariner_curve(exposures: np.ndarray, response: np.ndarray, anchor_index: int) -> MarinerCurveFit:
    best = (float("inf"), None, None)
    for shape in SHAPES:
        matrix = design_matrix(exposures, shape)
        for ridge in RIDGE_GRID:
            score = leave_one_out_rmse(matrix, response, anchor_index, ridge, GAMMA_PROVISIONAL)
            if score < best[0]:
                best = (score, shape, ridge)
    _, shape, ridge = best
    assert shape is not None and ridge is not None
    matrix = design_matrix(exposures, shape)
    low, high = GAMMA_BOUNDS
    search = minimize_scalar(
        lambda log_gamma: leave_one_out_rmse(matrix, response, anchor_index, ridge, math.exp(log_gamma)),
        bounds=(math.log(low), math.log(high)),
        method="bounded",
        options={"maxiter": 24, "xatol": 0.02},
    )
    gamma = float(math.exp(search.x))
    flat = math.log(gamma) >= (1 - GAMMA_FLAT_FRACTION) * math.log(high)
    if flat:
        gamma = GAMMA_FLAT_DEFAULT
    floor = floor_value(response[anchor_index], response, gamma)
    intercept, coefficients = nonnegative_solve(matrix, np.log(np.maximum(response - floor, DEFICIT_FLOOR)), ridge)
    return MarinerCurveFit(dict(shape), float(ridge), gamma, flat, floor, intercept, coefficients)


def curve_exposures(fraction: np.ndarray, training_tokens: float, epoch_scale: float) -> np.ndarray:
    """Materialized epochs of (Nemotron-CC, StarCoder) at StarCoder fraction p under simulated epoching.

    Both buckets reproduce the target budget's repetition: Nemotron-CC at weight 1 - p sees
    (1 - p) D_tgt / P_NC epochs and StarCoder p D_tgt / P_SC = p * epoch_scale.
    """
    del training_tokens  # the proxy budget cancels under simulated epoching
    nemotron = (1.0 - fraction) * LADDER_EXPOSURE_REFERENCE_TOKENS / NEMOTRON_POOL_TOKENS
    return np.column_stack([nemotron, fraction * epoch_scale])


def fit_panel_c(points: pd.DataFrame, metadata: dict[str, ladders.CurveMetadata]) -> tuple[list[CurveFit], pd.DataFrame]:
    fits: list[CurveFit] = []
    rows = []
    for curve_ref in ladders.D_SCALING_CURVES:
        curve = metadata[curve_ref]
        group = points.loc[points["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        fraction = group["starcoder_weight"].to_numpy(float)
        response = group["observed_bpb"].to_numpy(float)
        anchor_index = int(np.argmin(fraction))  # the smallest StarCoder fraction is the proportional mixture
        exposures = curve_exposures(fraction, curve.training_tokens, curve.epoch_scale)
        mariner = fit_mariner_curve(exposures, response, anchor_index)
        olmix_fit = olmix.fit_olmix_loglinear_model(
            np.column_stack([1.0 - fraction, fraction]), response, delta=HUBER_DELTA, seed=0, n_starts=OLMIX_STARTS
        )
        fits.append(
            CurveFit(
                curve_ref=curve_ref,
                shape=mariner.shape,
                ridge=mariner.ridge,
                gamma=mariner.gamma,
                gamma_flat=mariner.gamma_flat,
                floor=mariner.floor,
                intercept=mariner.intercept,
                coefficients=tuple(float(c) for c in mariner.coefficients),
                olmix_log_c=float(olmix_fit.log_c),
                olmix_coefficients=tuple(float(c) for c in olmix_fit.coefficients),
            )
        )
        dense_grid = np.linspace(float(fraction.min()), float(fraction.max()), DENSE_POINTS)
        dense_exposures = curve_exposures(dense_grid, curve.training_tokens, curve.epoch_scale)
        dense_matrix = design_matrix(dense_exposures, mariner.shape)
        mariner_prediction = predict(dense_matrix, mariner.floor, mariner.intercept, mariner.coefficients)
        olmix_prediction = olmix_fit.predict(np.column_stack([1.0 - dense_grid, dense_grid]))
        rows.append(
            pd.DataFrame(
                {
                    "curve_ref": curve_ref,
                    "starcoder_weight": dense_grid,
                    "mariner_bpb": mariner_prediction,
                    "olmix_bpb": olmix_prediction,
                }
            )
        )
    return fits, pd.concat(rows, ignore_index=True)


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.55, alpha=0.72)
    axis.tick_params(axis="both", colors=INK, labelsize=6.5, width=0.7, length=2.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.7)
    axis.set_xlabel(r"StarCoder mixture fraction, $p$", fontsize=7, color=INK, labelpad=2)


def add_epoch_axis(axis: plt.Axes, epoch_scale: float) -> None:
    def to_epochs(fraction: np.ndarray) -> np.ndarray:
        return np.asarray(fraction, dtype=float) * epoch_scale

    def to_fraction(epochs: np.ndarray) -> np.ndarray:
        return np.asarray(epochs, dtype=float) / epoch_scale

    top = axis.secondary_xaxis("top", functions=(to_epochs, to_fraction))
    top.set_xticks([0, 5, 10, 15, 20])
    top.tick_params(colors=INK, labelsize=6.5, width=0.7, length=2.5, pad=1.5)
    top.spines["top"].set_visible(False)
    top.set_xlabel("Materialized StarCoder epochs", fontsize=6.8, color=INK, labelpad=2)


def draw_downsampling_panel(
    axis: plt.Axes, points: pd.DataFrame, metadata: dict[str, replay.CurveMetadata]
) -> list[Line2D]:
    handles = []
    for (curve_ref, multiplier), color in zip(DOWNSAMPLING_CURVES, DOWNSAMPLING_COLORS, strict=True):
        curve = metadata[curve_ref]
        if multiplier is None:
            assert np.isclose(curve.support_fraction, 1.0), curve_ref
            label = "full pool"
        else:
            assert curve.epoch_multiplier is not None and np.isclose(curve.epoch_multiplier, multiplier), curve_ref
            label = rf"{multiplier:g}$\times$ the target's repetition" + (" (matched)" if multiplier == 1.0 else "")
        group = points.loc[points["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        full_pool = multiplier is None
        axis.plot(
            group["starcoder_weight"],
            group["observed_bpb"],
            color=color,
            linewidth=1.6 if full_pool else 1.3,
            linestyle=(0, (4, 2)) if full_pool else "-",
            marker="o",
            markersize=2.8,
            markerfacecolor="white" if full_pool else color,
            markeredgecolor=color if full_pool else INK,
            markeredgewidth=0.7 if full_pool else 0.25,
            zorder=4 if full_pool else 3,
        )
        minimum = group.loc[group["observed_bpb"].idxmin()]
        axis.scatter(
            [minimum["starcoder_weight"]],
            [minimum["observed_bpb"]],
            color=color,
            edgecolor=INK,
            linewidth=0.6,
            marker="*",
            s=55,
            zorder=6,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=1.6 if full_pool else 1.3,
                linestyle=(0, (4, 2)) if full_pool else "-",
                marker="o",
                markersize=2.8,
                markerfacecolor="white" if full_pool else color,
                markeredgecolor=color if full_pool else INK,
                label=label,
            )
        )
    axis.set_xlim(-0.01, 1.01)
    axis.set_ylim(*A_Y_LIMIT)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 1.07),
        borderaxespad=0,
        fontsize=6,
        frameon=True,
        framealpha=1.0,
        edgecolor=GRID,
        handlelength=2.2,
        borderpad=0.35,
        labelspacing=0.2,
        columnspacing=1.0,
    )
    return handles


def draw_ladder_panel(
    axis: plt.Axes,
    points: pd.DataFrame,
    metadata: dict[str, ladders.CurveMetadata],
    curve_refs: tuple[str, ...],
) -> None:
    for curve_ref, color in zip(curve_refs, RUNG_COLORS, strict=True):
        group = points.loc[points["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        curve = metadata[curve_ref]
        axis.plot(
            group["starcoder_weight"],
            group["observed_bpb"],
            color=color,
            linewidth=1.3,
            marker="o",
            markersize=2.8,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.25,
            zorder=3,
        )
        axis.scatter(
            [curve.observed_optimum_weight],
            [curve.observed_optimum_bpb],
            color=color,
            edgecolor=INK,
            linewidth=0.6,
            marker="*",
            s=55,
            zorder=6,
        )
    axis.set_xlim(*LADDER_X_LIMIT)
    axis.set_ylim(*LADDER_Y_LIMIT)
    axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axis.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])


def build_figure(
    replay_points: pd.DataFrame,
    replay_metadata: dict[str, replay.CurveMetadata],
    ladder_points: pd.DataFrame,
    ladder_metadata: dict[str, ladders.CurveMetadata],
) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.9075, wspace=0.22, hspace=0.65)
        flat = axes.ravel()
        draw_downsampling_panel(flat[0], replay_points, replay_metadata)
        style_axis(flat[0])
        flat[0].set_title(r"A · Vary downsampling; fixed $N$, $D$", loc="left", fontsize=7.5, fontweight="bold", pad=20)
        epoch_scale = ladder_metadata[ladders.N_SCALING_CURVES[0]].epoch_scale
        for axis, (letter, title, curve_refs) in zip(flat[1:], LADDER_PANELS, strict=True):
            draw_ladder_panel(axis, ladder_points, ladder_metadata, curve_refs)
            if letter in ("C", "D"):
                axis.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
                axis.set_ylim(*BOTTOM_Y_LIMIT)
            style_axis(axis)
            add_epoch_axis(axis, epoch_scale)
            axis.set_title(f"{letter} · {title}", loc="left", fontsize=7.5, fontweight="bold", pad=20)
        flat[0].set_ylabel("Programming Languages BPB", fontsize=7, color=INK, labelpad=3)
        flat[2].set_ylabel("Programming Languages BPB", fontsize=7, color=INK, labelpad=3)
        rung_handles = [
            Line2D(
                [],
                [],
                color=color,
                linewidth=1.3,
                marker="o",
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.25,
                label=f"{label} FLOPs",
            )
            for color, label in zip(RUNG_COLORS, RUNG_LABELS, strict=True)
        ]
        rung_handles.append(
            Line2D(
                [],
                [],
                color=INK,
                marker="*",
                markersize=8,
                markerfacecolor="#d9d9d9",
                markeredgecolor=INK,
                markeredgewidth=0.6,
                linestyle="none",
                label="lowest measured grid point",
            )
        )
        figure.legend(
            handles=rung_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=5,
            fontsize=6.3,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.3,
            handletextpad=0.5,
            title="Panels B to D: matched-compute rungs, every curve at the target's repetition",
            title_fontsize=6.3,
        )
        return figure


def configurations(
    replay_metadata: dict[str, replay.CurveMetadata],
    replay_points: pd.DataFrame,
    ladder_metadata: dict[str, ladders.CurveMetadata],
    ladder_points: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for curve_ref, multiplier in DOWNSAMPLING_CURVES:
        curve = replay_metadata[curve_ref]
        group = replay_points.loc[replay_points["curve_ref"].eq(curve_ref)]
        minimum = group.loc[group["observed_bpb"].idxmin()]
        rows.append(
            {
                "panel": "A",
                "curve_ref": curve_ref,
                "total_parameters": REPLAY_MODEL_PARAMETERS,
                "training_tokens": curve.training_tokens,
                "starcoder_subset_tokens": curve.support_tokens,
                "downsampling": "full pool" if multiplier is None else f"{multiplier:g}x",
                "epochs_at_p1": curve.epochs_at_full_share,
                "optimum_fraction": float(minimum["starcoder_weight"]),
                "optimum_epochs": float(minimum["starcoder_weight"]) * curve.epochs_at_full_share,
                "optimum_bpb": float(minimum["observed_bpb"]),
            }
        )
    for letter, _title, curve_refs in LADDER_PANELS:
        for curve_ref in curve_refs:
            curve = ladder_metadata[curve_ref]
            rows.append(
                {
                    "panel": letter,
                    "curve_ref": curve_ref,
                    "total_parameters": curve.total_parameters,
                    "training_tokens": curve.training_tokens,
                    "starcoder_subset_tokens": curve.training_tokens / curve.epoch_scale,
                    "downsampling": "1x",
                    "epochs_at_p1": curve.epoch_scale,
                    "optimum_fraction": curve.observed_optimum_weight,
                    "optimum_epochs": curve.observed_optimum_weight * curve.epoch_scale,
                    "optimum_bpb": curve.observed_optimum_bpb,
                    "compute_flops": curve.compute_flops,
                }
            )
    return pd.DataFrame(rows)


def format_tokens(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    return f"{value / 1e6:.1f}M"


def configurations_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Panel & Downsampling & $N$ & $D$ & $S_{\mathrm{SC}}$ & $D/S_{\mathrm{SC}}$ & $p^*$ & $E^*_{\mathrm{SC}}$ \\",
        r"\midrule",
    ]
    previous = None
    for record in table.to_dict("records"):
        panel = str(record["panel"])
        if previous is not None and panel != previous:
            lines.append(r"\midrule")
        previous = panel
        downsampling = str(record["downsampling"]).replace("x", "$\\times$")
        epoch_precision = 4 if record["downsampling"] == "full pool" else 1
        lines.append(
            f"{panel} & {downsampling} & {float(record['total_parameters']) / 1e6:.0f}M & "
            f"{format_tokens(float(record['training_tokens']))} & "
            f"{format_tokens(float(record['starcoder_subset_tokens']))} & "
            f"{float(record['epochs_at_p1']):.{epoch_precision}f} & {float(record['optimum_fraction']):.2f} & "
            f"{float(record['optimum_epochs']):.{epoch_precision}f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    replay_points, replay_metadata = replay.load_inputs(replay.DEFAULT_ATLAS_DIR, replay.DEFAULT_DESIGN)
    ladder_points, ladder_metadata = ladders.load_inputs(
        ladders.DEFAULT_ATLAS_DIR,
        ladders.DEFAULT_DESIGN,
        ladders.DEFAULT_FIXED_TPP_DIR,
        ladders.DEFAULT_FIXED_TPP_DESIGN,
    )
    fits, dense = fit_panel_c(ladder_points, ladder_metadata)
    dense.to_csv(args.output_dir / "panel_c_fits.csv", index=False)
    (args.output_dir / "panel_c_parameters.json").write_text(
        json.dumps([fit.__dict__ for fit in fits], indent=1, default=float), encoding="utf-8"
    )
    table = configurations(replay_metadata, replay_points, ladder_metadata, ladder_points)
    table.to_csv(args.output_dir / "configurations.csv", index=False)
    (args.output_dir / "configurations_table.tex").write_text(configurations_table(table), encoding="utf-8")

    figure = build_figure(replay_points, replay_metadata, ladder_points, ladder_metadata)
    for extension in ("png", "pdf"):
        with plt.rc_context(PLOT_STYLE):
            figure.savefig(args.output_dir / f"figure.{extension}", dpi=DPI)
        if args.drive_dir is not None:
            shutil.copyfile(args.output_dir / f"figure.{extension}", args.drive_dir / f"{DRIVE_STEM}.{extension}")
    plt.close(figure)
    for fit in fits:
        print(
            fit.curve_ref,
            "shape",
            fit.shape,
            "ridge",
            fit.ridge,
            "gamma",
            round(fit.gamma, 3),
            "flat" if fit.gamma_flat else "",
            "floor",
            round(fit.floor, 4),
            "amplitudes",
            [round(c, 3) for c in fit.coefficients],
        )
    print(table.to_string(index=False))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
