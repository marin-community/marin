# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on proxy design: Figures 2 (``fig:epoch-matching``) and 3 (``fig:b5``).

Figure 2 trains proxies of a 301M-parameter target run at ten tokens per parameter on Nemotron-CC/StarCoder mixtures,
with and without simulated epoching, and the target at the share each proxy selects. Figure 3 sweeps the StarCoder
share under simulated epoching along three paths of matched compute: a larger model at fixed tokens, more tokens at a
fixed model, and both at five tokens per parameter. MARINER is fitted to each sweep (``mariner.figures.starcoder``),
and the optimum's share and loss along each path follow power laws in the scaled quantity.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from mariner.figures.layout import (
    NEUTRAL,
    FigureData,
    data_limits,
    figure_legend,
    flops_label,
    panel_legend,
    panels,
    rounded_label,
    scale_colors,
)
from mariner.figures.starcoder import (
    ARM_COLORS,
    CURVE_POINTS,
    EPOCH_SCALE_TOLERANCE,
    FIXED_PARAMETERS_CURVES,
    FIXED_TOKENS_CURVES,
    SHARE_AXIS_LIMITS,
    Arm,
    EpochMatching,
    StarcoderSweep,
    atlas_sweeps,
    diagonal_sweeps,
    draw_repetition,
    epoch_matching,
    fitted_sweep,
    parameters_label,
    tokens_label,
)

# ---------------------------------------------------------------------------------------------------------------
# Figure 2: proxies with and without simulated epoching, and the target at their selections
# ---------------------------------------------------------------------------------------------------------------

# The first shares in view: below them the proxies (share 0) and the target (shares 0 and 0.1) lie far off the loss
# scale of the rest.
PROXY_FIRST_SHARE = 0.1
TARGET_FIRST_SHARE = 0.3
EPOCH_TICK_SHARES = (0.6, 0.8, 1.0)  # the shares at which Figure 2's top axes print materialized epochs
EPOCH_ROW_PITCH = 10.0  # points between stacked rows of epochs above a panel


def epoch_rows(axis: Axes, matching: EpochMatching, arms: Sequence[Arm]) -> None:
    """Rows of each arm's materialized StarCoder epochs at ``EPOCH_TICK_SHARES`` above the panel, the first lowest.

    Each row is a top axis of its own; the rows above the first stack their labels higher and draw no ticks or spine.
    """
    for row, arm in enumerate(arms):
        top = axis.secondary_xaxis("top")
        top.set_xticks(EPOCH_TICK_SHARES, [rounded_label(matching.epochs_at(arm, share)) for share in EPOCH_TICK_SHARES])
        top.tick_params(axis="x", colors=ARM_COLORS[arm])
        if row:
            top.tick_params(axis="x", length=0, pad=float(mpl.rcParams["xtick.major.pad"]) + row * EPOCH_ROW_PITCH)
            for spine in top.spines.values():
                spine.set_visible(False)
    top.set_xlabel("Materialized StarCoder epochs")


def draw_epoch_matching(root: Path) -> FigureData:
    """Figure 2: proxy and target loss over the StarCoder share at ten tokens per parameter, the share that each proxy
    selects, and the target's loss at each selection."""
    matching = epoch_matching(root)
    figure, axes = panels(2, legend_height=1.1)
    proxy, target = axes
    values = draw_repetition(
        proxy, target, matching, matching.matched, "each subset", (PROXY_FIRST_SHARE, TARGET_FIRST_SHARE)
    )
    epoch_rows(proxy, matching, (Arm.UNMATCHED, Arm.MATCHED))
    epoch_rows(target, matching, (Arm.TARGET,))
    epochs = [
        {"arm": arm.value, "share": share, "epochs": matching.epochs_at(arm, share)}
        for arm in Arm
        for share in EPOCH_TICK_SHARES
    ]
    return FigureData(figure, {**values, "epochs": pd.DataFrame(epochs)})


# ---------------------------------------------------------------------------------------------------------------
# Figure 3: the optimum's epochs along three paths of matched compute
# ---------------------------------------------------------------------------------------------------------------


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


def sweep_label(minimum: pd.Series) -> str:
    """A sweep's model size, tokens, tokens per parameter, training FLOPs and fitted optimum's epochs, from its row of
    ``matched_compute_values``' minima."""
    parameters, tokens = minimum["total_parameters"], minimum["train_tokens"]
    label = f"N {parameters_label(parameters)}, D {tokens_label(tokens)}, D/N {rounded_label(tokens / parameters)}"
    return f"{label}, {flops_label(minimum['compute_flops'])} FLOPs: E* {minimum['epochs']:.1f}"


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


def matched_compute_values(root: Path) -> dict[str, pd.DataFrame]:
    """The sweeps of the three matched-compute paths, MARINER's fit to each and its minimum, and each path's optimum
    laws (``path_law``), measured and extrapolated, with the StarCoder epochs at share 1 that its epoch axis scales by.
    """
    paths = matched_compute_paths(root)
    sweeps = {sweep.name: sweep for path in paths.values() for sweep in path}  # C05 lies on two paths
    fitted = {name: fitted_sweep(sweep) for name, sweep in sweeps.items()}
    observations, curves, minima, path_rows, laws = [], [], [], [], []
    for track, path in paths.items():
        attribute = TRACK_SCALES[track][0]
        for sweep in path:
            curve = fitted[sweep.name]
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
                    "share": curve.optimum.share,
                    "epochs": curve.optimum_epochs,
                    "bpb": curve.optimum.bpb,
                }
            )
        scale = np.asarray([getattr(sweep, attribute) for sweep in path])
        optima = [fitted[sweep.name].optimum for sweep in path]
        law = path_law(scale, np.array([o.share for o in optima]), np.array([o.bpb for o in optima]))
        measured = np.geomspace(scale[0], scale[-1], CURVE_POINTS)
        beyond = np.geomspace(scale[-1], EXTRAPOLATION_FACTOR * scale[-1], CURVE_POINTS)
        for values, extrapolated in ((measured, False), (beyond[1:], True)):
            path_rows += [
                {"track": track, "scale": q, "share": p, "bpb": loss, "extrapolated": extrapolated}
                for q, p, loss in zip(values, law.share(values), law.loss(values), strict=True)
            ]
        laws.append(
            {
                "track": track,
                "coordinate": attribute,
                **dataclasses.asdict(law),
                "epochs_at_p1": shared_epochs_at_p1(path),
            }
        )
    return {
        "observations": pd.DataFrame(observations),
        "fits": pd.DataFrame(curves),
        "minima": pd.DataFrame(minima),
        "optimum_paths": pd.DataFrame(path_rows),
        "path_laws": pd.DataFrame(laws),
    }


def draw_matched_compute_scaling(root: Path) -> FigureData:
    """Figure 3: the StarCoder sweeps of the three matched-compute paths, MARINER's fits and the optimum's path."""
    values = matched_compute_values(root)
    observations = values["observations"].groupby(["track", "curve"], sort=False)
    fits = values["fits"].groupby(["track", "curve"], sort=False)
    optimum_paths = values["optimum_paths"].groupby("track", sort=False)
    laws = values["path_laws"].set_index("track")
    figure, axes = panels(3, legend_height=1.0, sharey=True)
    minima_by_track = values["minima"].groupby("track", sort=False)
    for axis, track in zip(axes, Track, strict=True):
        minima = minima_by_track.get_group(track)
        handles = []
        for (_, minimum), color in zip(minima.iterrows(), scale_colors(len(minima)), strict=True):
            curve = fits.get_group((track, minimum["curve"]))
            measured = observations.get_group((track, minimum["curve"]))
            axis.plot(curve["share"].to_numpy(), curve["predicted_bpb"].to_numpy(), color=color)
            axis.scatter(
                measured["share"].to_numpy(),
                measured["bpb"].to_numpy(),
                s=12,
                color=color,
                edgecolor=NEUTRAL,
                linewidth=0.3,
                zorder=3,
            )
            axis.scatter(
                [minimum["share"]], [minimum["bpb"]], s=70, marker="*", color=color, edgecolor=NEUTRAL, zorder=5
            )
            handles.append(Line2D([], [], color=color, marker="o", markersize=3, label=sweep_label(minimum)))
        # The extrapolated path continues from the measured path's last point. The limits set below, from the sweeps'
        # data, clip it.
        path = optimum_paths.get_group(track)
        within = path[~path["extrapolated"]]
        beyond = pd.concat([within.iloc[-1:], path[path["extrapolated"]]])
        axis.plot(within["share"].to_numpy(), within["bpb"].to_numpy(), color=NEUTRAL, linestyle="--")
        axis.plot(beyond["share"].to_numpy(), beyond["bpb"].to_numpy(), color=NEUTRAL, linestyle=":")
        law = PathLaw(**{field.name: float(laws.at[track, field.name]) for field in dataclasses.fields(PathLaw)})
        handles.append(Line2D([], [], color=NEUTRAL, linestyle="--", label=law_label(law, TRACK_SCALES[track][1])))
        epochs_at_p1 = float(laws.at[track, "epochs_at_p1"])
        top = axis.secondary_xaxis("top", functions=(lambda p, s=epochs_at_p1: p * s, lambda e, s=epochs_at_p1: e / s))
        top.set_xlabel("Materialized StarCoder epochs")
        axis.set_title(TRACK_TITLES[track])
        axis.set_xlim(*SHARE_AXIS_LIMITS)
        axis.set_xlabel("StarCoder share of the mixture, p")
        panel_legend(axis, handles, fontsize=6.5)
    axes[0].set_ylabel("Programming-languages loss (BPB)")
    axes[0].set_ylim(*data_limits([*values["observations"]["bpb"], *values["fits"]["predicted_bpb"]]))
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
    return FigureData(figure, values)
