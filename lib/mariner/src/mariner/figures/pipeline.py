# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraw of the paper's Figure 1 (``fig:pipeline``): its repetition, surrogate, loss and accuracy panels.

The repetition panels draw the epoch-matching sweep of Figure 2 (``mariner.figures.starcoder``): the proxies' loss
with and without simulated epoching, the target's loss, and the excess loss that matching repetition avoids. The
surrogate panel fits MARINER, Olmix and LightGBM's boosted trees to six of the matched proxy's means
(``surrogate_values``). The lower panels draw each mixture's Uncheatable loss across the compute ladder with its
fitted power law and MARINER's compute multiples (``scaling_values``), and downstream accuracy at 1e21 FLOPs
(``accuracy_values``). Figure 9 (``mariner.figures.results``) draws the loss panel for both objectives, and Figure
8 draws the accuracy values against each task group's OlmoBaseEval Easy BPB.

LightGBM brings its own OpenMP runtime, and on macOS two OpenMP runtimes in one process can crash once both have
started thread pools: scikit-learn's k-means starts one. The trees run on one thread (``num_threads``), ``mariner
figures`` sets ``OMP_NUM_THREADS=1`` and ``OMP_THREAD_LIMIT=1`` before it imports this module, and it runs every fit
inside ``threadpool_limits(limits=1)``; a program that imports this module should do the same.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

from mariner.data import OUTCOME_PREFIX, read_table, require_columns
from mariner.figures.layout import (
    LOSS_LABELS,
    METHOD_COLORS,
    NEUTRAL,
    FigureData,
    figure_legend,
    mosaic,
    panel_legend,
)
from mariner.figures.starcoder import (
    DIAMOND,
    Arm,
    EpochMatching,
    draw_repetition,
    epoch_matching,
    fit_shares,
    realized_share_grid,
)
from mariner.methods import METHOD_LABELS, Method
from mariner.olmix import fit_log_linear
from mariner.tables import UNPROPOSED, ladder_track

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
    """Each ladder mixture's mean and SD (ddof 1) of ``objective`` at each compute scale, from
    ``mariner.tables.ladder_track``.

    The mixtures are proportional, UniMax-8, and Olmix's and MARINER's proposals for ``objective``.
    """
    ladder = read_table(root, "ladder")
    outcome = f"{OUTCOME_PREFIX}{objective}"
    require_columns(ladder, ["method", "proposal_objective", "compute_flops", outcome], "ladder")
    tracks = [
        ladder_track(ladder, method, objective if method in PROPOSED_METHODS else UNPROPOSED, outcome).assign(
            method=method.value
        )
        for method in LADDER_METHODS
    ]
    means = pd.concat(tracks, ignore_index=True)[["method", "compute_flops", "mean_bpb", "sd_bpb", "runs"]]
    return means.sort_values(["method", "compute_flops"], ignore_index=True)


def scaling_values(root: Path, objective: str) -> dict[str, pd.DataFrame]:
    """A loss-across-compute panel: the ladder's means of ``objective``, each mixture's law and MARINER's multiples.

    A multiple is the compute at which a baseline's law reaches MARINER's measured loss, over MARINER's compute; it
    is extrapolated where that compute lies beyond the ladder's largest scale.
    """
    points = ladder_means(root, objective)
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


def draw_scaling(axis: Axes, values: dict[str, pd.DataFrame], objective: str) -> None:
    """A panel of ``objective``'s loss across compute scales from ``scaling_values``, without a title."""
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
    axis.set_ylabel(LOSS_LABELS[objective])


def accuracy_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each mixture's mean OlmoBaseEval Easy accuracy (%) and BPB at 1e21 FLOPs by task group (Figures 1 and 8).

    The ``ladder_accuracy`` runs train each mixture at 1e21 FLOPs, Olmix's and MARINER's proposals for OlmoBaseEval
    Easy; ``All`` is the mean of the 51 tasks.
    """
    accuracy = read_table(root, "ladder_accuracy")
    require_columns(accuracy, ["group", "method", "accuracy", "bpb"], "ladder_accuracy")
    groups = [*ACCURACY_GROUPS, (ALL_TASKS, tuple(accuracy["group"].unique()))]
    rows = []
    for name, members in groups:
        tasks = accuracy[accuracy["group"].isin(members)]
        for method in LADDER_METHODS:
            scored = tasks[tasks["method"] == method]
            rows.append(
                {
                    "group": name,
                    "tasks": len(scored),
                    "method": method.value,
                    "accuracy": scored["accuracy"].mean(),
                    "bpb": scored["bpb"].mean(),
                }
            )
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


# ---------------------------------------------------------------------------------------------------------------
# The surrogate panel: MARINER, Olmix and boosted trees fitted to six means of the matched proxy's track
# ---------------------------------------------------------------------------------------------------------------

# The means of the proxy with simulated epoching that the paper's panel fits (its saved response selection
# A-matched-59366c24ded7073e): six shares through 90%, the 10% mean calibrating MARINER's floor. Each mean averages the
# three subsets' two trainer seeds, and every fit takes the StarCoder share that the runs drew.
SURROGATE_SHARES = (0.1, 0.3, 0.4, 0.5, 0.7, 0.9)
SURROGATE_POINTS = 400  # nominal shares from 10% to 100% at which the panel draws the fits
# The panel's LightGBM fit to six observations: its settings, one thread and LightGBM's deterministic mode.
TREE_PARAMETERS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "num_leaves": 2,
    "learning_rate": 0.1,
    "min_data_in_leaf": 1,
    "min_data_in_bin": 1,
    "max_bin": 255,
    "seed": 42,
    "num_threads": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
TREE_ROUNDS = 64
TREE_TIE_TOLERANCE = 1e-12  # BPB: the plateaus within this of the least form the trees' optimum
TREE_COLOR = "#B77B2B"
TREE_OPTIMUM_STYLE = {"color": TREE_COLOR, "alpha": 0.3, "linewidth": 5}  # the band over the trees' optimum
UNFITTED_COLOR = "#A5AEB6"


def fit_trees(shares: np.ndarray, bpb: np.ndarray) -> lgb.Booster:
    """LightGBM's boosted trees on the StarCoder share alone, with the panel's settings."""
    data = lgb.Dataset(shares[:, None], label=bpb, feature_name=["starcoder_share"])
    return lgb.train(TREE_PARAMETERS, data, num_boost_round=TREE_ROUNDS)


def tree_splits(node: dict) -> list[float]:
    if "leaf_value" in node:
        return []
    return [float(node["threshold"]), *tree_splits(node["left_child"]), *tree_splits(node["right_child"])]


def tree_plateaus(trees: lgb.Booster, low: float, high: float) -> pd.DataFrame:
    """The trees' prediction on [``low``, ``high``], one row per plateau between their split thresholds."""
    splits = {split for tree in trees.dump_model()["tree_info"] for split in tree_splits(tree["tree_structure"])}
    edges = np.array([low, *sorted(split for split in splits if low < split < high), high])
    values = trees.predict(((edges[:-1] + edges[1:]) / 2)[:, None], num_threads=1)
    rows: list[dict[str, float]] = []
    for left, right, value in zip(edges[:-1], edges[1:], values, strict=True):
        if rows and rows[-1]["bpb"] == value:
            rows[-1]["realized_right"] = float(right)
        else:
            rows.append({"realized_left": float(left), "realized_right": float(right), "bpb": float(value)})
    return pd.DataFrame(rows)


def curve_point(curves: pd.DataFrame, position: int, model: str) -> dict[str, str | float]:
    """A model's point at row ``position`` of the panel's curves: the nominal and realized share and its loss."""
    row = curves.iloc[position]
    return {
        "model": model,
        "share": float(row["share"]),
        "realized_share": float(row["realized_share"]),
        "bpb": float(row[f"{model}_bpb"]),
    }


def surrogate_values(matching: EpochMatching, realized_grid: pd.Series) -> dict[str, pd.DataFrame]:
    """The surrogate panel: the matched track's means from 10%, and MARINER's, Olmix's and the trees' fits to six.

    ``realized_grid`` is the share that the matched proxy draws at each of the design's grid weights
    (``realized_share_grid``). Each fit takes the grid's share at a fitted mean's weight, the share that its runs drew,
    and the panel draws the fits at nominal shares, each mapped to the share interpolated between the grid's weights.
    MARINER is fitted as Figures 3 and 4 fit it (``fit_shares``), with Nemotron-CC at the proxy's epochs of its finite
    pool, and its optimum is the least of its curve on the panel's grid. Olmix's law is monotone in the share, so it
    picks the end of the grid that its slope falls toward. The trees' optimum is every plateau at their least
    prediction.
    """
    track = matching.matched.mean()
    track = track[track.index >= SURROGATE_SHARES[0]]
    means = pd.DataFrame(
        {"share": track.index, "realized_share": realized_grid.loc[track.index].to_numpy(), "bpb": track.to_numpy()}
    )
    means["fitted"] = means["share"].isin(SURROGATE_SHARES)
    fitted = means[means["fitted"]]
    shares, bpb = fitted["realized_share"].to_numpy(float), fitted["bpb"].to_numpy(float)
    mariner = fit_shares(shares, bpb, matching.epochs_at_p1[Arm.MATCHED], matching.web_epochs_at_p0[Arm.MATCHED])
    olmix = fit_log_linear(np.column_stack([1.0 - shares, shares]), bpb)
    trees = fit_trees(shares, bpb)
    grid = np.linspace(SURROGATE_SHARES[0], 1.0, SURROGATE_POINTS)
    nominal, drawn = realized_grid.index.to_numpy(float), realized_grid.to_numpy(float)
    grid_realized = np.asarray(np.interp(grid, nominal, drawn))
    weights = np.column_stack([1.0 - grid_realized, grid_realized])
    curves = pd.DataFrame(
        {
            "share": grid,
            "realized_share": grid_realized,
            "mariner_bpb": mariner.predict(weights),
            "olmix_bpb": olmix.predict(weights),
            "trees_bpb": trees.predict(grid_realized[:, None], num_threads=1),
        }
    )
    plateaus = tree_plateaus(trees, grid_realized[0], grid_realized[-1])
    for side in ("left", "right"):
        plateaus[f"share_{side}"] = np.interp(plateaus[f"realized_{side}"], drawn, nominal)
    plateaus["optimum"] = plateaus["bpb"] <= plateaus["bpb"].min() + TREE_TIE_TOLERANCE
    lowest = int(np.argmin(curves["mariner_bpb"].to_numpy()))
    pick = len(curves) - 1 if olmix.coefficients[1] < olmix.coefficients[0] else 0
    optima = pd.DataFrame([curve_point(curves, lowest, "mariner"), curve_point(curves, pick, "olmix")])
    return {
        "surrogate_means": means,
        "surrogate_curves": curves,
        "surrogate_trees": plateaus,
        "surrogate_optima": optima,
    }


def draw_surrogates(axis: Axes, values: dict[str, pd.DataFrame]) -> None:
    """The surrogate panel from ``surrogate_values``."""
    means, curves, plateaus = values["surrogate_means"], values["surrogate_curves"], values["surrogate_trees"]
    optima = values["surrogate_optima"].set_index("model")
    mariner_color, olmix_color = METHOD_COLORS[Method.MARINER], METHOD_COLORS[Method.OLMIX]
    axis.plot(curves["share"], curves["mariner_bpb"], color=mariner_color, linewidth=1.6)
    axis.plot(curves["share"], curves["olmix_bpb"], color=olmix_color, linestyle="--")
    steps = plateaus[["share_left", "share_right"]].to_numpy().ravel()
    axis.plot(steps, np.repeat(plateaus["bpb"].to_numpy(), 2), color=TREE_COLOR, linewidth=1.0)
    for plateau in plateaus[plateaus["optimum"]].itertuples():
        axis.plot([plateau.share_left, plateau.share_right], [plateau.bpb] * 2, **TREE_OPTIMUM_STYLE)
    unfitted = means[~means["fitted"]]
    axis.scatter(unfitted["share"], unfitted["bpb"], s=18, color=UNFITTED_COLOR, edgecolor="white", zorder=4)
    fitted = means[means["fitted"]]
    axis.scatter(fitted["share"], fitted["bpb"], s=18, facecolor="white", edgecolor=NEUTRAL, zorder=5)
    mariner, olmix = optima.loc["mariner"], optima.loc["olmix"]
    axis.plot([mariner["share"]], [mariner["bpb"]], "*", color=mariner_color, markersize=11, markeredgecolor="white")
    axis.plot([olmix["share"]], [olmix["bpb"]], color=olmix_color, **DIAMOND)
    lowest = plateaus[plateaus["optimum"]]
    panel_legend(
        axis,
        [
            Line2D([], [], color=mariner_color, linewidth=1.6, label="MARINER fit"),
            Line2D(
                [],
                [],
                color=mariner_color,
                marker="*",
                linestyle="none",
                markersize=11,
                label=f"MARINER's interior optimum: p = {mariner['share']:.3f}",
            ),
            Line2D([], [], color=olmix_color, linestyle="--", label="Olmix fit (monotone)"),
            Line2D([], [], color=olmix_color, **DIAMOND, label=f"Olmix's pick: p = {olmix['share']:.2f}"),
            Line2D([], [], color=TREE_COLOR, label="boosted trees (piecewise constant)"),
            Line2D(
                [],
                [],
                **TREE_OPTIMUM_STYLE,
                label=(f"trees' optimum: p = {lowest['share_left'].min():.2f} to {lowest['share_right'].max():.2f}"),
            ),
            Line2D([], [], color=NEUTRAL, marker="o", markerfacecolor="white", linestyle="none", label="fitted means"),
            Line2D([], [], color=UNFITTED_COLOR, marker="o", linestyle="none", label="means not fitted"),
        ],
    )
    axis.set_title(f"Surrogates fitted to {int(means['fitted'].sum())} proxy means")
    axis.set_xlabel("StarCoder share of the mixture, p")
    axis.set_ylabel("Proxy loss (BPB)")


# ---------------------------------------------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------------------------------------------

PIPELINE_PANELS = (
    ("proxy", "proxy", "target", "target", "surrogates", "surrogates"),
    ("scaling", "scaling", "scaling", "accuracy", "accuracy", "accuracy"),
)


def draw_pipeline(root: Path) -> FigureData:
    """Figure 1: proxy and target loss over the StarCoder share with and without simulated epoching, the surrogates'
    fits to the proxy's means, loss across compute scales, and downstream accuracy at 1e21 FLOPs."""
    matching = epoch_matching(root)
    figure, axes = mosaic(PIPELINE_PANELS, columns=3, rows=2, legend_height=1.7)
    # Figure 1 draws the matched subsets' mean, where Figure 2 draws each subset.
    mean_track = matching.matched.mean().to_frame("mean of the subsets").T
    repetition = draw_repetition(
        axes["proxy"], axes["target"], matching, mean_track, "their mean", (FIRST_SHARE, FIRST_SHARE)
    )
    surrogates = surrogate_values(matching, realized_share_grid(root, matching, Arm.MATCHED))
    draw_surrogates(axes["surrogates"], surrogates)
    scaling, accuracy = scaling_values(root, SCALING_OBJECTIVE), accuracy_values(root)
    draw_scaling(axes["scaling"], scaling, SCALING_OBJECTIVE)
    axes["scaling"].set_title("Loss across compute scales († extrapolated)")
    draw_accuracy(axes["accuracy"], accuracy)
    handles = [
        Line2D([], [], color=METHOD_COLORS[method], marker="o", label=METHOD_LABELS[method]) for method in LADDER_METHODS
    ]
    figure_legend(figure, handles, columns=len(LADDER_METHODS))
    return FigureData(figure, {**repetition, **surrogates, **scaling, **accuracy})
