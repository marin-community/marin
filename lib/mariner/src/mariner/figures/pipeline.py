# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraw of the paper's Figure 1 (``fig:pipeline``): its proxy and target repetition, loss and accuracy panels.

The repetition panels draw the epoch-matching sweep of Figure 2 (``mariner.figures.starcoder``): the proxies' loss
with and without simulated epoching, the target's loss, and the excess loss that matching repetition avoids. The lower
panels draw each mixture's Uncheatable loss across the compute ladder with its fitted power law and MARINER's compute
multiples (``scaling_values``), and downstream accuracy at 1e21 FLOPs (``accuracy_values``). The paper's top-right
panel, which illustrates the surrogates' shapes on six of the matched proxy's points, is not redrawn.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

from mariner.data import read_table
from mariner.figures.layout import (
    METHOD_COLORS,
    METHOD_LABELS,
    FigureData,
    Method,
    figure_legend,
    panel_legend,
    panels,
)
from mariner.figures.starcoder import (
    ARM_COLORS,
    AVOIDED_COLOR,
    DIAMOND,
    Arm,
    draw_proxies,
    draw_target,
    epoch_matching,
    selection_values,
)

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
FIRST_SHARE = 0.1  # the repetition panels start at 10% StarCoder: the runs without it lie far off the scale


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


def draw_pipeline(root: Path) -> FigureData:
    """Figure 1: proxy and target loss over the StarCoder share with and without simulated epoching, loss across
    compute scales, and downstream accuracy at 1e21 FLOPs."""
    matching = epoch_matching(root)
    selections = selection_values(matching).set_index("arm")
    optimum = matching.target_optimum()
    figure, axes = panels(2, rows=2, legend_height=1.3)
    proxy, target, scaling_axis, accuracy_axis = axes
    # The figure draws the matched subsets' mean, where Figure 2 draws each subset.
    mean_track = matching.matched.mean().to_frame().T
    draw_proxies(proxy, matching, mean_track, FIRST_SHARE)
    draw_target(target, matching, FIRST_SHARE)
    with_color, without_color = ARM_COLORS[Arm.MATCHED], ARM_COLORS[Arm.UNMATCHED]
    panel_legend(
        proxy,
        [
            Line2D([], [], color=with_color, marker="o", markersize=3, label="proxy with simulated epoching (range)"),
            Line2D(
                [],
                [],
                color=without_color,
                marker="o",
                markerfacecolor="white",
                label="proxy without simulated epoching",
            ),
            Line2D(
                [],
                [],
                color=with_color,
                **DIAMOND,
                label=f"selected with: {100 * matching.selections[Arm.MATCHED]:.0f}%",
            ),
            Line2D(
                [],
                [],
                color=without_color,
                **DIAMOND,
                label=f"selected without: {100 * matching.selections[Arm.UNMATCHED]:.0f}%",
            ),
        ],
    )
    panel_legend(
        target,
        [
            Line2D([], [], color=ARM_COLORS[Arm.TARGET], marker="o", markerfacecolor="white", label="target scale"),
            Line2D(
                [],
                [],
                color=ARM_COLORS[Arm.TARGET],
                marker="*",
                linestyle="none",
                markersize=11,
                label=f"target optimum: {100 * optimum.share:.0f}%",
            ),
            Line2D([], [], color=with_color, **DIAMOND, label="at the selection with simulated epoching"),
            Line2D([], [], color=without_color, **DIAMOND, label="at the selection without"),
            Line2D([], [], color=AVOIDED_COLOR, label=f"excess loss avoided: {100 * matching.excess_loss_avoided:.0f}%"),
        ],
    )
    for axis, title, label in (
        (proxy, "Proxy scale", "Proxy loss (BPB)"),
        (target, "Target scale", "Target loss (BPB)"),
    ):
        axis.set_title(f"{title}: StarCoder repetition")
        axis.set_xlabel("StarCoder share of the mixture, p")
        axis.set_ylabel(label)
    scaling, accuracy = scaling_values(root), accuracy_values(root)
    draw_scaling(scaling_axis, scaling)
    draw_accuracy(accuracy_axis, accuracy)
    handles = [
        Line2D([], [], color=METHOD_COLORS[method], marker="o", label=METHOD_LABELS[method]) for method in LADDER_METHODS
    ]
    figure_legend(figure, handles, columns=len(LADDER_METHODS))
    shown = matching.shares >= FIRST_SHARE
    proxy_curves = pd.DataFrame(
        {
            "share": matching.shares[shown],
            Arm.MATCHED.value: mean_track.iloc[0].to_numpy(float)[shown],
            Arm.UNMATCHED.value: matching.unmatched.to_numpy(float)[shown],
        }
    )
    repetition = {
        "proxy_curves": proxy_curves,
        "target_curve": (
            matching.target[matching.target.index >= FIRST_SHARE].rename_axis("share").rename("bpb").reset_index()
        ),
        "selections": selections.reset_index(),
        "target_optimum": pd.DataFrame([optimum._asdict()]),
        "summary": pd.DataFrame({"excess_loss_avoided": [matching.excess_loss_avoided]}),
    }
    return FigureData(figure, {**repetition, **scaling, **accuracy})
