# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on epochs and scale: Figure 3 (``fig:b5``) and the lower panels of Figure 1.

Figure 3 sweeps the StarCoder share of Nemotron-CC/StarCoder mixtures under simulated epoching along three paths of
matched compute: a larger model at fixed tokens, more tokens at a fixed model, and both at five tokens per parameter.
MARINER is fitted to each sweep as a two-bucket swarm (``fit_sweep``), and the optimum's share and loss along each path
follow power laws in the scaled quantity. The dataset records training FLOPs for the fixed-TPP sweeps only
(``starcoder_diagonal``), so only their legend entries print compute.

Figure 1 (``fig:pipeline``) is not registered: its proxy and target repetition panel needs the StarCoder epoch-matching
sweep, which the dataset does not hold yet. Its two lower panels are here: each mixture's Uncheatable loss across the
compute ladder with its fitted power law and MARINER's compute multiples (``scaling_values``), and downstream accuracy
at 1e21 FLOPs (``accuracy_values``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit, minimize_scalar

from mariner.data import OBSERVATION_ID, read_table
from mariner.figures.layout import (
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    FigureData,
    Method,
    data_limits,
    figure_legend,
    panel_legend,
    panels,
    scale_colors,
)
from mariner.fit import fit_objective
from mariner.model import Anchor, Objective, ObjectiveFit, Swarm

# ---------------------------------------------------------------------------------------------------------------
# The StarCoder sweeps and their MARINER fits (Figures 3 and 4)
# ---------------------------------------------------------------------------------------------------------------

FIXED_TOKENS_CURVES = ("C05", "C06", "C07", "C08")  # starcoder_curves refs: 1.00B tokens, 210M to 1.18B parameters
FIXED_PARAMETERS_CURVES = ("C05", "C10", "C13", "C14")  # 210M parameters, 1.00B to 7.41B tokens
# Under simulated epoching the sweeps' target budget is Nemotron-CC's pool, so a share of 1 is one Nemotron-CC epoch
# (the fitter description of starcoder_curves).
NEMOTRON_EPOCHS_PER_UNIT_WEIGHT = 1.0
SWEEP_BUCKETS = ("nemotron_cc", "starcoder")
SWEEP_TASK = "programming_languages_bpb"
SWEEP_OBJECTIVE = Objective(SWEEP_TASK, (SWEEP_TASK,), np.ones(1))
MINIMUM_GRID = 4_001  # shares that bracket a fit's local minima before a bounded search refines them
MINIMUM_XATOL = 1e-12
CURVE_POINTS = 400  # shares at which a fitted curve is drawn
EPOCH_SCALE_TOLERANCE = 1e-12  # relative: a diagonal cell's epochs over its shares, one value up to rounding


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
class StarcoderSweep:
    """One sweep of the StarCoder share at a fixed model and token budget, sorted by share.

    ``name`` is the atlas's ``curve_ref`` or the diagonal's ``cell_id``. ``compute_flops`` is NaN where the dataset
    records no compute.
    """

    name: str
    total_parameters: float
    train_tokens: float
    starcoder_epochs_at_p1: float  # StarCoder's materialized epochs at share 1
    compute_flops: float
    shares: np.ndarray
    bpb: np.ndarray


def sorted_sweep(
    name: str,
    total_parameters: float,
    train_tokens: float,
    starcoder_epochs_at_p1: float,
    compute_flops: float,
    shares: pd.Series,
    bpb: pd.Series,
) -> StarcoderSweep:
    order = np.argsort(shares.to_numpy(float))
    return StarcoderSweep(
        name,
        float(total_parameters),
        float(train_tokens),
        float(starcoder_epochs_at_p1),
        float(compute_flops),
        shares.to_numpy(float)[order],
        bpb.to_numpy(float)[order],
    )


def atlas_sweeps(root: Path, refs: Sequence[str]) -> dict[str, StarcoderSweep]:
    """The atlas's sweeps ``refs`` (``starcoder_curves``), with their runs' shares and BPB."""
    curves = read_table(root, "starcoder_curves").set_index("curve_ref")
    membership = read_table(root, "starcoder_curve_membership")
    observations = read_table(root, "starcoder_observations").set_index(OBSERVATION_ID)
    sweeps = {}
    for ref in dict.fromkeys(refs):
        curve = curves.loc[ref]
        runs = observations.loc[membership.loc[membership["curve_id"] == curve["curve_id"], OBSERVATION_ID]]
        sweeps[ref] = sorted_sweep(
            ref,
            curve["total_parameters"],
            curve["train_tokens"],
            curve["starcoder_epochs_at_p1"],
            math.nan,
            runs["starcoder_weight"],
            runs["observed_bpb"],
        )
    return sweeps


def diagonal_sweeps(root: Path) -> list[StarcoderSweep]:
    """The fixed-TPP sweeps (``starcoder_diagonal``), one per cell, from the smallest compute to the largest."""
    diagonal = read_table(root, "starcoder_diagonal").sort_values(["rung", "starcoder_weight"])
    sweeps = []
    for cell, rows in diagonal.groupby("cell_id", sort=False):
        scales = (rows["starcoder_epochs"] / rows["starcoder_weight"]).to_numpy(float)
        if np.ptp(scales) > EPOCH_SCALE_TOLERANCE * scales.max():
            raise ValueError(f"the runs of diagonal cell {cell} disagree on StarCoder's epochs per unit share")
        first = rows.iloc[0]
        sweeps.append(
            sorted_sweep(
                str(cell),
                first["total_parameters"],
                first["materialized_tokens"],
                float(np.median(scales)),
                first["compute_flops"],
                rows["starcoder_weight"],
                rows["starcoder_bpb"],
            )
        )
    return sweeps


def fit_sweep(sweep: StarcoderSweep) -> ObjectiveFit:
    """MARINER fitted to one sweep as the paper fitted Figures 3 and 4.

    The sweep is a two-bucket swarm, Nemotron-CC at share 1 - p and StarCoder at share p. Its smallest-share run, the
    proportional mixture, is the calibration run: it stays in every training fold, and its BPB anchors the floor
    without a noise margin, since the sweeps have no repeats. Leave-one-out cross-validation over the other runs
    selects the shape, the ridge weight and the floor multiplier.
    """
    rows = np.arange(len(sweep.shares))
    swarm = Swarm(
        runs=tuple(f"{sweep.name}/{row}" for row in rows),
        buckets=SWEEP_BUCKETS,
        weights=np.column_stack([1.0 - sweep.shares, sweep.shares]),
        epochs_per_unit_weight=np.array([NEMOTRON_EPOCHS_PER_UNIT_WEIGHT, sweep.starcoder_epochs_at_p1]),
        pool_fractions=np.ones((len(rows), len(SWEEP_BUCKETS))),
        outcomes=pd.DataFrame({SWEEP_TASK: sweep.bpb}),
        calibration=rows == 0,
    )
    folds = tuple((np.delete(rows, row), rows[row : row + 1]) for row in rows[1:])
    return fit_objective(swarm, SWEEP_OBJECTIVE, {SWEEP_TASK: Anchor(float(sweep.bpb[0]), 0.0)}, folds=folds)


def sweep_prediction(fit: ObjectiveFit, shares: np.ndarray) -> np.ndarray:
    return fit.predict(np.column_stack([1.0 - shares, shares]))


class Optimum(NamedTuple):
    share: float
    bpb: float


def fitted_minimum(fit: ObjectiveFit, low: float, high: float) -> Optimum:
    """The share in [``low``, ``high``] at which the fit predicts the least BPB, and that BPB.

    A grid of shares brackets each local minimum, which a bounded scalar search refines; the interval's ends compete
    too.
    """
    grid = np.linspace(low, high, MINIMUM_GRID)
    loss = sweep_prediction(fit, grid)
    candidates = [Optimum(grid[0], loss[0]), Optimum(grid[-1], loss[-1])]
    for index in np.flatnonzero((loss[1:-1] < loss[:-2]) & (loss[1:-1] <= loss[2:])) + 1:
        search = minimize_scalar(
            lambda share: float(sweep_prediction(fit, np.array([share]))[0]),
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
            options={"xatol": MINIMUM_XATOL},
        )
        candidates.append(Optimum(float(search.x), float(search.fun)))
    best = min(candidates, key=lambda optimum: optimum.bpb)
    return Optimum(float(best.share), float(best.bpb))


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


def parameters_label(count: float) -> str:
    return f"{count / 1e9:.2f}B" if count >= 1e9 else f"{count / 1e6:.0f}M"


def tokens_label(count: float) -> str:
    return f"{count / 1e9:.2f}B"


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
    sweeps = {sweep.name: sweep for path in paths.values() for sweep in path}
    fits = {name: fit_sweep(sweep) for name, sweep in sweeps.items()}
    figure, axes = panels(3, legend_height=1.0, sharey=True)
    observations, curves, minima, path_rows, laws = [], [], [], [], []
    for axis, (track, path) in zip(axes, paths.items(), strict=True):
        attribute, symbol = TRACK_SCALES[track]
        scales, optima = [], []
        handles = []
        for sweep, color in zip(path, scale_colors(len(path)), strict=True):
            fit = fits[sweep.name]
            optimum = fitted_minimum(fit, float(sweep.shares[0]), float(sweep.shares[-1]))
            epochs = optimum.share * sweep.starcoder_epochs_at_p1
            dense = np.linspace(sweep.shares[0], sweep.shares[-1], CURVE_POINTS)
            predicted = sweep_prediction(fit, dense)
            axis.plot(dense, predicted, color=color)
            axis.scatter(sweep.shares, sweep.bpb, s=12, color=color, edgecolor=NEUTRAL, linewidth=0.3, zorder=3)
            axis.scatter([optimum.share], [optimum.bpb], s=70, marker="*", color=color, edgecolor=NEUTRAL, zorder=5)
            handles.append(Line2D([], [], color=color, marker="o", markersize=3, label=sweep_label(sweep, epochs)))
            scales.append(getattr(sweep, attribute))
            optima.append(optimum)
            observations += [
                {"track": track, "curve": sweep.name, "share": share, "bpb": bpb}
                for share, bpb in zip(sweep.shares, sweep.bpb, strict=True)
            ]
            curves += [
                {"track": track, "curve": sweep.name, "share": share, "predicted_bpb": value}
                for share, value in zip(dense, predicted, strict=True)
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
                    "epochs": epochs,
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
        axis.set_xlim(0.0, 0.95)
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


# ---------------------------------------------------------------------------------------------------------------
# Figure 1's lower panels: loss across compute scales, and accuracy at 1e21 FLOPs
# ---------------------------------------------------------------------------------------------------------------

SCALING_OBJECTIVE = "uncheatable"  # the lower-left panel's evaluation
LADDER_METHODS = (Method.PROPORTIONAL, Method.UNIMAX, Method.OLMIX, Method.MARINER)
PROPOSED_METHODS = (Method.OLMIX, Method.MARINER)  # the ladder trains their proposals for each objective
MULTIPLE_BASELINES = (Method.OLMIX, Method.PROPORTIONAL)  # Figure 1 prints MARINER's compute multiples over these
COMPUTE_UNIT = 1e18
# A mixture's ladder law, L = E + A (C / 1e18)^-alpha, fitted to its mean loss at each rung by scipy's curve_fit from
# these starts and within these bounds, as the paper's compute-equivalence analysis fitted it.
LAW_START = (0.5, 0.5, 0.3)
LAW_BOUNDS = ((0.0, 0.0, 0.01), (2.0, 5.0, 2.0))
LAW_EVALUATIONS = 20_000
LAW_POINTS = 240
LAW_MARGIN = 1.12  # the extrapolated laws reach this factor beyond the farthest equivalent compute
# Figure 1's accuracy groups: the dataset's groups of the 51 OlmoBaseEval Easy tasks that each bar averages.
ACCURACY_GROUPS = (
    ("Basic", ("Basic Skills",)),
    ("Math", ("Minerva MATH",)),
    ("Code", ("Code", "Code: machine-translated MBPP")),
    ("MMLU", ("MMLU",)),
    ("QA", ("Question answering (lm-eval overlap)", "QA answer selection")),
)
ALL_TASKS = "All"


@dataclasses.dataclass(frozen=True)
class ScalingLaw:
    """A mixture's loss against training compute C in FLOPs: e + a (C / 1e18)^-alpha."""

    e: float
    a: float
    alpha: float

    def loss(self, compute: np.ndarray) -> np.ndarray:
        return self.e + self.a * (compute / COMPUTE_UNIT) ** -self.alpha

    def compute_at(self, loss: float) -> float:
        """The compute at which the law reaches ``loss``, which must lie above its floor ``e``."""
        if loss <= self.e:
            raise ValueError(f"the law never reaches {loss:.6f}: its floor is {self.e:.6f}")
        return COMPUTE_UNIT * ((loss - self.e) / self.a) ** (-1.0 / self.alpha)


def ladder_law(log_compute: np.ndarray, e: float, a: float, alpha: float) -> np.ndarray:
    """``ScalingLaw`` in log10 FLOPs, the coordinate the fit works in."""
    return e + a * 10.0 ** (-alpha * (log_compute - math.log10(COMPUTE_UNIT)))


def fit_scaling_law(compute: np.ndarray, loss: np.ndarray) -> ScalingLaw:
    fitted = curve_fit(ladder_law, np.log10(compute), loss, p0=LAW_START, bounds=LAW_BOUNDS, maxfev=LAW_EVALUATIONS)
    return ScalingLaw(*(float(value) for value in fitted[0]))


def ladder_means(root: Path, objective: str) -> pd.DataFrame:
    """Each ladder mixture's mean and SD (ddof 1) of ``objective`` at each compute scale.

    The mixtures are proportional, UniMax-8, and Olmix's and MARINER's proposals for ``objective``.
    """
    ladder = read_table(root, "ladder")
    method = ladder["method"]
    rows = ladder[
        method.isin([m.value for m in LADDER_METHODS])
        & (~method.isin([m.value for m in PROPOSED_METHODS]) | (ladder["proposal_objective"] == objective))
    ]
    grouped = rows.groupby(["method", "compute_flops"])[f"bpb__{objective}"]
    means = grouped.agg(mean_bpb="mean", sd_bpb="std", runs="size").reset_index()
    return means.sort_values(["method", "compute_flops"], ignore_index=True)


def scaling_values(root: Path) -> dict[str, pd.DataFrame]:
    """Figure 1's lower-left panel: the ladder's Uncheatable means, each mixture's law and MARINER's multiples.

    A multiple is the compute at which a baseline's law reaches MARINER's measured loss, over MARINER's compute; it
    is extrapolated where that compute lies beyond the ladder's largest scale.
    """
    points = ladder_means(root, SCALING_OBJECTIVE)
    laws = {
        method: fit_scaling_law(rows["compute_flops"].to_numpy(float), rows["mean_bpb"].to_numpy(float))
        for method, rows in points.groupby("method")
    }
    largest = float(points["compute_flops"].max())
    mariner = points[points["method"] == Method.MARINER]
    multiples = []
    for compute, loss in zip(mariner["compute_flops"], mariner["mean_bpb"], strict=True):
        for baseline in MULTIPLE_BASELINES:
            equivalent = laws[baseline].compute_at(loss)
            multiples.append(
                {
                    "compute_flops": compute,
                    "mariner_bpb": loss,
                    "baseline": baseline.value,
                    "equivalent_flops": equivalent,
                    "multiple": equivalent / compute,
                    "extrapolated": equivalent > largest,
                }
            )
    multiples_table = pd.DataFrame(multiples)
    reach = LAW_MARGIN * float(multiples_table["equivalent_flops"].max())
    smallest = float(points["compute_flops"].min())
    curve_rows = []
    for method, law in laws.items():
        for compute, extrapolated in (
            (np.geomspace(smallest, largest, LAW_POINTS), False),
            (np.geomspace(largest, reach, LAW_POINTS)[1:], True),
        ):
            curve_rows += [
                {"method": method, "compute_flops": c, "bpb": value, "extrapolated": extrapolated}
                for c, value in zip(compute, law.loss(compute), strict=True)
            ]
    return {
        "points": points,
        "laws": pd.DataFrame([{"method": method, **dataclasses.asdict(law)} for method, law in laws.items()]),
        "law_curves": pd.DataFrame(curve_rows),
        "multiples": multiples_table,
    }


def draw_scaling(axis: Axes, values: dict[str, pd.DataFrame]) -> None:
    """The lower-left panel of Figure 1 from ``scaling_values``."""
    points, curves = values["points"], values["law_curves"]
    for method in LADDER_METHODS:
        color = METHOD_COLORS[method]
        rows = points[points["method"] == method]
        axis.errorbar(
            rows["compute_flops"],
            rows["mean_bpb"],
            yerr=rows["sd_bpb"].fillna(0.0),
            fmt="o",
            color=color,
            capsize=2,
        )
        for extrapolated, linestyle in ((False, "-"), (True, ":")):
            curve = curves[(curves["method"] == method) & (curves["extrapolated"] == extrapolated)]
            axis.plot(curve["compute_flops"], curve["bpb"], color=color, linestyle=linestyle)
    for (compute, loss), rows in values["multiples"].groupby(["compute_flops", "mariner_bpb"], sort=False):
        start = compute
        for line, row in enumerate(rows.sort_values("equivalent_flops").itertuples()):
            color = METHOD_COLORS[Method(row.baseline)]
            axis.plot([start, row.equivalent_flops], [loss, loss], color=color, linewidth=0.9)
            start = row.equivalent_flops
            dagger = "†" if row.extrapolated else ""
            axis.annotate(
                f"{row.multiple:.1f}\N{MULTIPLICATION SIGN}{dagger}",
                xy=(compute, loss),
                xytext=(0, -8 - 8 * line),
                textcoords="offset points",
                ha="center",
                va="top",
                color=color,
                fontsize=6.5,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    # The loss spans less than a decade, so the log axis labels its in-between ticks too, in BPB.
    for ticks in (axis.yaxis.set_major_formatter, axis.yaxis.set_minor_formatter):
        ticks(FormatStrFormatter("%.1f"))
    axis.set_xlabel("Training compute (FLOPs)")
    axis.set_ylabel("Uncheatable Eval loss (BPB)")
    axis.set_title("Loss across compute scales († extrapolated)")


def accuracy_values(root: Path) -> dict[str, pd.DataFrame]:
    """Figure 1's lower-right panel: each mixture's mean OlmoBaseEval Easy accuracy (%) at 1e21 FLOPs by task group.

    The ``ladder_accuracy`` runs train each mixture at 1e21 FLOPs, Olmix's and MARINER's proposals for OlmoBaseEval
    Easy; ``All`` is the mean of the 51 tasks.
    """
    accuracy = read_table(root, "ladder_accuracy")
    groups = [*ACCURACY_GROUPS, (ALL_TASKS, tuple(accuracy["group"].unique()))]
    rows = []
    for name, members in groups:
        tasks = accuracy[accuracy["group"].isin(members)]
        for method in LADDER_METHODS:
            scores = tasks.loc[tasks["method"] == method, "accuracy"]
            rows.append({"group": name, "tasks": len(scores), "method": method.value, "accuracy": scores.mean()})
    return {"accuracy": pd.DataFrame(rows)}


def draw_accuracy(axis: Axes, values: dict[str, pd.DataFrame]) -> None:
    """The lower-right panel of Figure 1 from ``accuracy_values``."""
    accuracy = values["accuracy"]
    groups = list(dict.fromkeys(accuracy["group"]))
    width = 0.8 / len(LADDER_METHODS)
    for index, method in enumerate(LADDER_METHODS):
        rows = accuracy[accuracy["method"] == method].set_index("group").loc[groups]
        positions = np.arange(len(groups)) + (index - (len(LADDER_METHODS) - 1) / 2) * width
        axis.bar(positions, rows["accuracy"], width=width, color=METHOD_COLORS[method])
    tasks = accuracy.drop_duplicates("group").set_index("group").loc[groups, "tasks"]
    axis.set_xticks(np.arange(len(groups)), [f"{group}\n({count})" for group, count in tasks.items()])
    axis.set_xlabel("Task group (tasks)")
    axis.set_ylabel("Mean accuracy (%)")
    axis.set_title("OlmoBaseEval Easy accuracy, 1e21 FLOPs")


def draw_pipeline_lower_panels(root: Path) -> FigureData:
    """Figure 1's lower panels on their own; the figure needs its repetition panel before it is registered."""
    figure, axes = panels(2, legend_height=0.3)
    scaling, accuracy = scaling_values(root), accuracy_values(root)
    draw_scaling(axes[0], scaling)
    draw_accuracy(axes[1], accuracy)
    handles = [
        Line2D([], [], color=METHOD_COLORS[method], marker="o", label=METHOD_LABELS[method]) for method in LADDER_METHODS
    ]
    figure_legend(figure, handles, columns=len(LADDER_METHODS))
    return FigureData(figure, {**scaling, **accuracy})
