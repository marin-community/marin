# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on proxy epochs: Figure 3 (``fig:b5``).

Figure 3 sweeps the StarCoder share of Nemotron-CC/StarCoder mixtures under simulated epoching along three paths of
matched compute: a larger model at fixed tokens, more tokens at a fixed model, and both at five tokens per parameter.
MARINER is fitted to each sweep (``mariner.figures.starcoder``), and the optimum's share and loss along each path follow
power laws in the scaled quantity. The dataset records training FLOPs for the fixed-TPP sweeps only
(``starcoder_diagonal``), so only their legend entries print compute.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mariner.figures.layout import NEUTRAL, FigureData, data_limits, figure_legend, panel_legend, panels, scale_colors
from mariner.figures.starcoder import (
    CURVE_POINTS,
    EPOCH_SCALE_TOLERANCE,
    FIXED_PARAMETERS_CURVES,
    FIXED_TOKENS_CURVES,
    SHARE_AXIS_LIMITS,
    StarcoderSweep,
    atlas_sweeps,
    diagonal_sweeps,
    fitted_sweep,
    parameters_label,
    tokens_label,
)


class Track(StrEnum):
    """The three paths of matched compute in Figure 3, by what each holds fixed."""

    FIXED_TOKENS = "fixed_tokens"
    FIXED_PARAMETERS = "fixed_parameters"
    FIXED_TPP = "fixed_tpp"


TRACK_TITLES = {
    Track.FIXED_TOKENS: "A. Fixed tokens, scaling parameters",
    Track.FIXED_PARAMETERS: "B. Fixed parameters, scaling tokens",
    Track.FIXED_TPP: "C. Fixed tokens per parameter, scaling both",
}
# The quantity each path scales, as a sweep attribute and as a symbol.
TRACK_SCALES = {
    Track.FIXED_TOKENS: ("total_parameters", "N"),
    Track.FIXED_PARAMETERS: ("train_tokens", "D"),
    Track.FIXED_TPP: ("total_parameters", "N"),
}
EXTRAPOLATION_FACTOR = 100.0  # the optimum paths continue (dotted) to this multiple of the largest scale


@dataclasses.dataclass(frozen=True)
class PathLaw:
    """The optimum's share and loss along a path as power laws of the scaled quantity q.

    share = share_coefficient (q / reference)^share_exponent and loss = loss_coefficient (q / reference)^loss_exponent,
    least-squares lines in log-log coordinates through the path's fitted minima.
    """

    reference: float
    share_coefficient: float
    share_exponent: float
    loss_coefficient: float
    loss_exponent: float

    def share(self, scale: np.ndarray) -> np.ndarray:
        return self.share_coefficient * (scale / self.reference) ** self.share_exponent

    def loss(self, scale: np.ndarray) -> np.ndarray:
        return self.loss_coefficient * (scale / self.reference) ** self.loss_exponent


def path_law(scales: np.ndarray, shares: np.ndarray, losses: np.ndarray) -> PathLaw:
    """The power laws through a path's minima, referred to its smallest scale."""
    reference = float(scales[0])
    log_scale = np.log(scales / reference)
    share_exponent, share_intercept = np.polyfit(log_scale, np.log(shares), 1)
    loss_exponent, loss_intercept = np.polyfit(log_scale, np.log(losses), 1)
    return PathLaw(
        reference,
        float(np.exp(share_intercept)),
        float(share_exponent),
        float(np.exp(loss_intercept)),
        float(loss_exponent),
    )


def tpp_label(tokens_per_parameter: float) -> str:
    return f"{tokens_per_parameter:.2f}" if tokens_per_parameter < 1 else f"{tokens_per_parameter:.1f}"


def sweep_label(sweep: StarcoderSweep, epochs: float) -> str:
    """A sweep's model size, tokens, tokens per parameter, compute where recorded, and fitted optimum's epochs."""
    label = (
        f"N {parameters_label(sweep.total_parameters)}, D {tokens_label(sweep.train_tokens)},"
        f" D/N {tpp_label(sweep.train_tokens / sweep.total_parameters)}"
    )
    if math.isfinite(sweep.compute_flops):
        label += f", {sweep.compute_flops:.2g} FLOPs"
    return f"{label}: E* {epochs:.1f}"


def law_label(law: PathLaw, symbol: str) -> str:
    """The share law as the paper prints it: referred to the smallest scale rounded to three digits."""
    reference = float(f"{law.reference:.3g}")
    coefficient = float(law.share(np.array([reference]))[0])
    unit = tokens_label(reference) if symbol == "D" else parameters_label(reference)
    return f"optimum fit: p* = {coefficient:.3f} ({symbol}/{unit})^{law.share_exponent:.3f}"


def shared_epochs_at_p1(path: Sequence[StarcoderSweep]) -> float:
    """The StarCoder epochs at share 1 of a path's sweeps, which the panel's epoch axis needs to be one value."""
    values = np.array([sweep.starcoder_epochs_at_p1 for sweep in path])
    if np.ptp(values) > EPOCH_SCALE_TOLERANCE * values.max():
        raise ValueError(f"the sweeps {', '.join(sweep.name for sweep in path)} differ in StarCoder's epochs per share")
    return float(values[0])


def matched_compute_paths(root: Path) -> dict[Track, list[StarcoderSweep]]:
    atlas = atlas_sweeps(root, (*FIXED_TOKENS_CURVES, *FIXED_PARAMETERS_CURVES))
    return {
        Track.FIXED_TOKENS: [atlas[ref] for ref in FIXED_TOKENS_CURVES],
        Track.FIXED_PARAMETERS: [atlas[ref] for ref in FIXED_PARAMETERS_CURVES],
        Track.FIXED_TPP: diagonal_sweeps(root),
    }


def draw_matched_compute_scaling(root: Path) -> FigureData:
    """Figure 3: the StarCoder sweeps of the three matched-compute paths, MARINER's fits and the optimum's path."""
    paths = matched_compute_paths(root)
    sweeps = {sweep.name: sweep for path in paths.values() for sweep in path}  # C05 lies on two paths
    fitted = {name: fitted_sweep(sweep) for name, sweep in sweeps.items()}
    figure, axes = panels(3, legend_height=1.0, sharey=True)
    observations, curves, minima, path_rows, laws = [], [], [], [], []
    for axis, (track, path) in zip(axes, paths.items(), strict=True):
        attribute, symbol = TRACK_SCALES[track]
        scales, optima = [], []
        handles = []
        for sweep, color in zip(path, scale_colors(len(path)), strict=True):
            curve = fitted[sweep.name]
            optimum = curve.optimum
            axis.plot(curve.shares, curve.predicted_bpb, color=color)
            axis.scatter(sweep.shares, sweep.bpb, s=12, color=color, edgecolor=NEUTRAL, linewidth=0.3, zorder=3)
            axis.scatter([optimum.share], [optimum.bpb], s=70, marker="*", color=color, edgecolor=NEUTRAL, zorder=5)
            label = sweep_label(sweep, curve.optimum_epochs)
            handles.append(Line2D([], [], color=color, marker="o", markersize=3, label=label))
            scales.append(getattr(sweep, attribute))
            optima.append(optimum)
            observations += [
                {"track": track, "curve": sweep.name, "share": share, "bpb": bpb}
                for share, bpb in zip(sweep.shares, sweep.bpb, strict=True)
            ]
            curves += [
                {"track": track, "curve": sweep.name, "share": share, "predicted_bpb": value}
                for share, value in zip(curve.shares, curve.predicted_bpb, strict=True)
            ]
            minima.append(
                {
                    "track": track,
                    "curve": sweep.name,
                    "total_parameters": sweep.total_parameters,
                    "train_tokens": sweep.train_tokens,
                    "tokens_per_parameter": sweep.train_tokens / sweep.total_parameters,
                    "compute_flops": sweep.compute_flops,
                    "share": optimum.share,
                    "epochs": curve.optimum_epochs,
                    "bpb": optimum.bpb,
                }
            )
        scale = np.asarray(scales)
        law = path_law(scale, np.array([o.share for o in optima]), np.array([o.bpb for o in optima]))
        measured = np.geomspace(scale[0], scale[-1], CURVE_POINTS)
        beyond = np.geomspace(scale[-1], EXTRAPOLATION_FACTOR * scale[-1], CURVE_POINTS)
        # The limits set below, from the sweeps' data, clip the extrapolated paths.
        axis.plot(law.share(measured), law.loss(measured), color=NEUTRAL, linestyle="--")
        axis.plot(law.share(beyond), law.loss(beyond), color=NEUTRAL, linestyle=":")
        handles.append(Line2D([], [], color=NEUTRAL, linestyle="--", label=law_label(law, symbol)))
        for values, extrapolated in ((measured, False), (beyond[1:], True)):
            path_rows += [
                {"track": track, "scale": q, "share": p, "bpb": loss, "extrapolated": extrapolated}
                for q, p, loss in zip(values, law.share(values), law.loss(values), strict=True)
            ]
        laws.append({"track": track, "coordinate": attribute, **dataclasses.asdict(law)})
        epochs_at_p1 = shared_epochs_at_p1(path)
        top = axis.secondary_xaxis("top", functions=(lambda p, s=epochs_at_p1: p * s, lambda e, s=epochs_at_p1: e / s))
        top.set_xlabel("Materialized StarCoder epochs")
        axis.set_title(TRACK_TITLES[track])
        axis.set_xlim(*SHARE_AXIS_LIMITS)
        axis.set_xlabel("StarCoder share of the mixture, p")
        panel_legend(axis, handles, fontsize=6.5)
    axes[0].set_ylabel("Programming-languages loss (BPB)")
    axes[0].set_ylim(*data_limits([row["bpb"] for row in observations] + [row["predicted_bpb"] for row in curves]))
    figure_legend(
        figure,
        [
            Line2D([], [], color=NEUTRAL, marker="o", linestyle="none", markersize=3, label="observed loss"),
            Line2D([], [], color=NEUTRAL, label="MARINER fit"),
            Line2D([], [], color=NEUTRAL, marker="*", linestyle="none", markersize=8, label="fitted minimum"),
            Line2D([], [], color=NEUTRAL, linestyle="--", label="optimum fit"),
            Line2D([], [], color=NEUTRAL, linestyle=":", label="extrapolation"),
        ],
        columns=5,
    )
    values = {
        "observations": pd.DataFrame(observations),
        "fits": pd.DataFrame(curves),
        "minima": pd.DataFrame(minima),
        "optimum_paths": pd.DataFrame(path_rows),
        "path_laws": pd.DataFrame(laws),
    }
    return FigureData(figure, values)
