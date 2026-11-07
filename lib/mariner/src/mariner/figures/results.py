# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on the proposed mixtures at larger compute: Figures 8 to 11.

- ``fig:accuracy`` (Figure 8): each mixture's mean OlmoBaseEval Easy accuracy against its mean BPB at 1e21 FLOPs, by
  task group, from ``mariner.figures.pipeline.accuracy_values``; a line joins the mixtures in order of BPB.
- ``fig:r1`` (Figure 9): each mixture's loss across the compute ladder on both objectives, its fitted power law, and
  the compute at which Olmix's and proportional mixing's laws reach MARINER's loss, from
  ``mariner.figures.pipeline.scaling_values``.
- ``fig:off-objective`` (Figure 10): proportional mixing, and Olmix's and MARINER's mixtures on the objective each was
  proposed for and on the other, each with its fitted law.
- ``fig:component-tradeoffs`` (Figure 11): each component's change of BPB from proportional mixing at 3e18 FLOPs,
  for MARINER's and Olmix's mixtures proposed for each objective and for a subset of its components.

Figures 9 and 10 read the ``ladder`` table alone. Its 3e18 rung holds the main swarm's ten proportional repeats, the
June proportional run (``proportional_3e18-ebc4aa``, ``mariner.data.june_run``) and the validation launches' runs of
the other mixtures, so that the figures' 3e18 proportional point averages eleven runs; the dataset's index names the
swarm and the validation launches beside the ladder as those runs' sources.

Figure 11 measures each change from ``fits_reliability``'s proportional means, the means of the main swarm's ten
proportional repeats alone. The paper's Uncheatable panels averaged eleven proportional runs, those ten and the June
run: its bars differ from the redraw's by at most 2.4e-4 BPB, and none changes sign.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, MultipleLocator, NullLocator

from mariner.data import (
    OBJECTIVES,
    OUTCOME_PREFIX,
    RELIABILITY_SOURCES,
    RELIABILITY_TABLE,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    read_table,
    reliability_rows,
)
from mariner.figures.layout import (
    LOSS_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    OBJECTIVE_TITLES,
    FigureData,
    Method,
    figure_legend,
    mosaic,
    panels,
)
from mariner.figures.pipeline import (
    ACCURACY_GROUPS,
    ALL_TASKS,
    LADDER_METHODS,
    LAW_POINTS,
    accuracy_values,
    draw_scaling,
    fit_scaling_law,
    scaling_values,
)
from mariner.figures.surrogate import TRAINER_SEEDS
from mariner.tables import COMPUTE_SCALES, UNPROPOSED, compute_label

MARKER_SIZE = 7.0
JOIN_WIDTH = 0.8  # the lines that join a panel's points

# ---------------------------------------------------------------------------------------------------------------
# Figure 8: accuracy against BPB at 1e21 FLOPs by task group
# ---------------------------------------------------------------------------------------------------------------

# Every panel spans 15 points of accuracy and 0.2 BPB, as the paper's do (its caption: "Axes span 15 points and 0.2
# BPB"), so that the panels' slopes compare.
ACCURACY_SPAN = 15.0
BPB_SPAN = 0.2
ACCURACY_TICK = 5.0
BPB_TICK = 0.05
MIXTURE_MARKERS = {Method.PROPORTIONAL: "h", Method.UNIMAX: "s", Method.OLMIX: "D", Method.MARINER: "o"}
GROUP_COLUMNS = 3


def centered_limits(values: pd.Series, span: float) -> tuple[float, float]:
    """Limits ``span`` apart around the middle of ``values``; raises ``ValueError`` when the values spread wider."""
    low, high = float(values.min()), float(values.max())
    if high - low > span:
        raise ValueError(f"the values spread over {high - low:.3g}, wider than the panels' span of {span:g}")
    middle = (low + high) / 2
    return middle - span / 2, middle + span / 2


def mixture_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MIXTURE_MARKERS[method],
            markersize=MARKER_SIZE,
            color=METHOD_COLORS[method],
            markeredgecolor="white",
            label=METHOD_LABELS[method],
        )
        for method in LADDER_METHODS
    ]


def draw_accuracy_vs_bpb(root: Path) -> FigureData:
    """Figure 8: each mixture's mean accuracy against its mean BPB at 1e21 FLOPs, one panel per task group."""
    values = accuracy_values(root)
    accuracy = values["accuracy"]
    groups = [ALL_TASKS, *(name for name, _ in ACCURACY_GROUPS)]
    layout = [groups[start : start + GROUP_COLUMNS] for start in range(0, len(groups), GROUP_COLUMNS)]
    figure, axes = mosaic(layout, columns=GROUP_COLUMNS, rows=len(layout), legend_height=0.4)
    for group in groups:
        axis = axes[group]
        rows = accuracy[accuracy["group"] == group].set_index("method")
        joined = rows.sort_values("bpb")
        axis.plot(joined["bpb"], joined["accuracy"], color=NEUTRAL, linewidth=JOIN_WIDTH, zorder=1)
        for method in LADDER_METHODS:
            row = rows.loc[method.value]
            axis.plot(
                [row["bpb"]],
                [row["accuracy"]],
                linestyle="none",
                marker=MIXTURE_MARKERS[method],
                markersize=MARKER_SIZE,
                color=METHOD_COLORS[method],
                markeredgecolor="white",
                zorder=3,
            )
        axis.set_xlim(*centered_limits(rows["bpb"], BPB_SPAN))
        axis.set_ylim(*centered_limits(rows["accuracy"], ACCURACY_SPAN))
        axis.xaxis.set_major_locator(MultipleLocator(BPB_TICK))
        axis.yaxis.set_major_locator(MultipleLocator(ACCURACY_TICK))
        axis.set_title(f"{group} ({rows['tasks'].iloc[0]} tasks)")
        axis.set_xlabel("Mean BPB")
        axis.set_ylabel("Mean accuracy (%)")
    handles = [*mixture_handles(), Line2D([], [], color=NEUTRAL, linewidth=JOIN_WIDTH, label="mixtures in order of BPB")]
    figure_legend(figure, handles, columns=len(handles))
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 9: loss across compute scales on both objectives, with MARINER's compute multiples
# ---------------------------------------------------------------------------------------------------------------

SCALING_SERIES = ("points", "laws", "law_curves", "multiples")  # the series of scaling_values, per objective


def method_handles(methods: tuple[Method, ...]) -> list[Line2D]:
    return [Line2D([], [], color=METHOD_COLORS[method], marker="o", label=METHOD_LABELS[method]) for method in methods]


def draw_scaling_curves(root: Path) -> FigureData:
    """Figure 9: each mixture's loss across compute scales on each objective, with MARINER's compute multiples."""
    figure, axes = panels(len(OBJECTIVES), legend_height=0.6)
    drawn = {}
    for axis, objective in zip(axes, OBJECTIVES, strict=True):
        drawn[objective] = scaling_values(root, objective)
        draw_scaling(axis, drawn[objective], objective)
        axis.set_title(OBJECTIVE_TITLES[objective])
    handles = [
        *method_handles(LADDER_METHODS),
        Line2D([], [], color=NEUTRAL, linestyle=":", label="law extrapolated past the ladder"),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            linewidth=0.9,
            label="compute at which Olmix's, then proportional mixing's, law reaches MARINER's loss;\n"
            "labels: multiples of MARINER's compute († extrapolated)",
        ),
    ]
    figure_legend(figure, handles, columns=3)
    values = {
        series: pd.concat(
            [scaling[series].assign(objective=objective) for objective, scaling in drawn.items()], ignore_index=True
        )
        for series in SCALING_SERIES
    }
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 10: each method's mixtures on the objective they were proposed for and on the other
# ---------------------------------------------------------------------------------------------------------------

ON_OFF_METHODS = (Method.PROPORTIONAL, Method.OLMIX, Method.MARINER)
TRACK_STYLES = {True: "-", False: "--"}  # on-objective solid, off-objective dashed


def ladder_track(ladder: pd.DataFrame, method: Method, proposal: str, objective: str) -> pd.DataFrame:
    """The mean and SD (ddof 1) of ``objective`` over a ladder mixture's runs at each compute scale, and their count.

    The mixture is ``method``'s proposal for the objective ``proposal``, or ``none`` for a mixture no surrogate
    proposed. Raises ``ValueError`` unless it has a value at each of the ladder's four scales and nowhere else.
    """
    rows = ladder[(ladder["method"] == method.value) & (ladder["proposal_objective"] == proposal)]
    column = f"{OUTCOME_PREFIX}{objective}"
    if rows[column].isna().any():
        raise ValueError(f"the ladder lacks {objective} for {int(rows[column].isna().sum())} runs of {method}'s mixture")
    track = rows.groupby("compute_flops")[column].agg(mean_bpb="mean", sd_bpb="std", runs="size").reset_index()
    if track["compute_flops"].tolist() != list(COMPUTE_SCALES):
        raise ValueError(
            f"the ladder trains {method}'s {proposal} mixture at {track['compute_flops'].tolist()} FLOPs, not at"
            f" {list(COMPUTE_SCALES)}"
        )
    return track


def on_off_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each method's tracks on each objective, on and off the objective of its proposal, and their fitted laws."""
    ladder = read_table(root, "ladder")
    tracks, laws, curves = [], [], []
    for objective in OBJECTIVES:
        (other,) = (name for name in OBJECTIVES if name != objective)
        for method in ON_OFF_METHODS:
            proposals = [(UNPROPOSED, True)] if method == Method.PROPORTIONAL else [(objective, True), (other, False)]
            for proposal, on_objective in proposals:
                key = {
                    "objective": objective,
                    "method": method.value,
                    "proposal_objective": proposal,
                    "on_objective": on_objective,
                }
                track = ladder_track(ladder, method, proposal, objective)
                law = fit_scaling_law(track["compute_flops"].to_numpy(float), track["mean_bpb"].to_numpy(float))
                compute = np.geomspace(COMPUTE_SCALES[0], COMPUTE_SCALES[-1], LAW_POINTS)
                tracks.append(track.assign(**key))
                laws.append({**key, "e": law.e, "a": law.a, "alpha": law.alpha})
                curves.append(pd.DataFrame({**key, "compute_flops": compute, "bpb": law.loss(compute)}))
    return {
        "tracks": pd.concat(tracks, ignore_index=True),
        "laws": pd.DataFrame(laws),
        "law_curves": pd.concat(curves, ignore_index=True),
    }


def draw_on_off_objective(root: Path) -> FigureData:
    """Figure 10: each method's mixtures on and off the objective of their proposal, on each objective."""
    values = on_off_values(root)
    tracks, curves = values["tracks"], values["law_curves"]
    figure, axes = panels(len(OBJECTIVES), legend_height=0.6)
    for axis, objective in zip(axes, OBJECTIVES, strict=True):
        for method in ON_OFF_METHODS:
            color = METHOD_COLORS[method]
            for on_objective, linestyle in TRACK_STYLES.items():
                selected = (tracks["objective"] == objective) & (tracks["method"] == method.value)
                rows = tracks[selected & (tracks["on_objective"] == on_objective)]
                if rows.empty:
                    continue
                axis.errorbar(
                    rows["compute_flops"],
                    rows["mean_bpb"],
                    yerr=rows["sd_bpb"].fillna(0.0),
                    fmt="o",
                    color=color,
                    markerfacecolor=color if on_objective else "white",
                    capsize=2,
                )
                curve = curves[
                    (curves["objective"] == objective)
                    & (curves["method"] == method.value)
                    & (curves["on_objective"] == on_objective)
                ]
                axis.plot(curve["compute_flops"], curve["bpb"], color=color, linestyle=linestyle)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(COMPUTE_SCALES, [compute_label(compute) for compute in COMPUTE_SCALES])
        axis.xaxis.set_minor_locator(NullLocator())
        # The loss spans less than a decade, so the log axis labels its in-between ticks too, as Figure 9's does.
        for ticks in (axis.yaxis.set_major_formatter, axis.yaxis.set_minor_formatter):
            ticks(FormatStrFormatter("%.1f"))
        axis.set_xlabel("Training compute (FLOPs)")
        axis.set_ylabel(LOSS_LABELS[objective])
        axis.set_title(OBJECTIVE_TITLES[objective])
    handles = [
        *method_handles(ON_OFF_METHODS),
        Line2D([], [], color=NEUTRAL, marker="o", label="on-objective"),
        Line2D([], [], color=NEUTRAL, linestyle="--", marker="o", markerfacecolor="white", label="off-objective"),
    ]
    figure_legend(figure, handles, columns=len(handles))
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 11: each component's change from proportional mixing at 3e18 FLOPs
# ---------------------------------------------------------------------------------------------------------------

# Uncheatable's rows in the paper's order: code, science, then the three corpora of general English text that its
# subset objective, uncheatable_worsened, weights.
UNCHEATABLE_ROWS = (
    "uncheatable/github_cpp",
    "uncheatable/github_python",
    "uncheatable/arxiv_physics",
    "uncheatable/arxiv_computer_science",
    "uncheatable/ao3_english",
    "uncheatable/wikipedia_english",
    "uncheatable/bbc_news",
)
BROAD_LANGUAGE = ("uncheatable/ao3_english", "uncheatable/wikipedia_english", "uncheatable/bbc_news")
# OlmoBaseEval Easy's rows: fits_reliability's groups Code, Math and QA, with MMLU's and Basic Skills' tasks taken out
# of QA by their ids. obe_easy_qa15 weights the 15 that remain.
OBE_EASY_GROUPS = ("Code", "Math", "Basic Skills", "MMLU", "QA")
QA_SUBGROUPS = (("mmlu_", "MMLU"), ("basic_skills_", "Basic Skills"))
QA = "QA"
# The Olmix cap and KL sweep trained Olmix's deployed proposals once more, at trainer seed 0; the bars leave that run
# out, so that each averages one run at each trainer seed, as MARINER's do.
RERUN_LAUNCH = "olmix_cap_kl_sweep"
TRADEOFF_METHODS = (Method.MARINER, Method.OLMIX)
SYMLOG_THRESHOLD = 0.005  # BPB: the change axis is linear within this of zero and logarithmic beyond, as the paper's
SYMLOG_LINSCALE = 0.35
SYMLOG_DECADES = (0.01, 0.1, 1.0)
LIMIT_MARGIN = 1.25  # the change axis reaches this factor beyond the longest bars
BAR_HEIGHT = 0.36
OPTIMIZED_COLOR = "#D6E6F5"


class TradeoffPanel(NamedTuple):
    """A panel of Figure 11: the objective it measures, the objective its mixtures were proposed for, each method's
    mixture, and its optimized rows, which it shades."""

    objective: str
    proposal_objective: str
    mixtures: dict[Method, str]
    title: str
    optimized: tuple[str, ...]


TRADEOFF_PANELS = (
    TradeoffPanel(
        "uncheatable",
        "uncheatable",
        {Method.MARINER: TRAINED_MARINER["uncheatable"], Method.OLMIX: TRAINED_OLMIX["uncheatable"].mixture_id},
        "Uncheatable\noptimized for all",
        (ALL_TASKS,),
    ),
    TradeoffPanel(
        "uncheatable",
        "uncheatable_worsened",
        {Method.MARINER: "mix_d35652d26bfb4161", Method.OLMIX: "mix_de06cf1f70fca954"},
        "Uncheatable\noptimized for AO3, Wikipedia and BBC News",
        BROAD_LANGUAGE,
    ),
    TradeoffPanel(
        "obe_easy",
        "obe_easy",
        {Method.MARINER: TRAINED_MARINER["obe_easy"], Method.OLMIX: TRAINED_OLMIX["obe_easy"].mixture_id},
        "OlmoBaseEval Easy\noptimized for all",
        (ALL_TASKS,),
    ),
    TradeoffPanel(
        "obe_easy",
        "obe_easy_qa15",
        {Method.MARINER: "mix_520005cb5b644ee6", Method.OLMIX: "mix_e7254c0b10669665"},
        "OlmoBaseEval Easy\noptimized for QA",
        (QA,),
    ),
)


class TradeoffRow(NamedTuple):
    """A row of an objective's panels: an Uncheatable task, an OlmoBaseEval Easy group or ``All``, the aggregate."""

    item: str
    label: str
    tasks: tuple[str, ...]
    proportional_bpb: float


def obe_easy_group(task_id: str, group: str) -> str:
    for prefix, subgroup in QA_SUBGROUPS:
        if group == QA and task_id.startswith(prefix):
            return subgroup
    return group


def tradeoff_rows(reliability: pd.DataFrame, objective: str) -> list[TradeoffRow]:
    """``objective``'s rows with their proportional means from ``fits_reliability``, the aggregate's last.

    Raises ``ValueError`` when the table lacks the objective's rows, its aggregate's row or one of Uncheatable's tasks,
    or puts an OlmoBaseEval Easy task in a group that the figure does not draw.
    """
    rows = reliability_rows(reliability, RELIABILITY_SOURCES[objective])
    aggregate = rows[rows["task_id"].isna()]
    if len(aggregate) != 1:
        raise ValueError(f"{RELIABILITY_TABLE} has {len(aggregate)} aggregate rows for {objective}, not one")
    tasks = rows[rows["task_id"].notna()].set_index("task_id")
    if objective == "uncheatable":
        missing = sorted(set(UNCHEATABLE_ROWS) - set(tasks.index))
        if missing:
            raise ValueError(f"{RELIABILITY_TABLE} has no row of {', '.join(missing)}")
        items = [(task, str(tasks.loc[task, "task"]), (task,)) for task in UNCHEATABLE_ROWS]
    else:
        groups = pd.Series({task: obe_easy_group(task, row["group"]) for task, row in tasks.iterrows()})
        unknown = groups[~groups.isin(OBE_EASY_GROUPS)]
        if len(unknown):
            raise ValueError(
                f"{RELIABILITY_TABLE} puts OlmoBaseEval Easy tasks in groups the figure does not draw:"
                f" {', '.join(f'{task} ({group})' for task, group in unknown.items())}"
            )
        items = []
        for group in OBE_EASY_GROUPS:
            members = tuple(groups.index[groups == group])
            if not members:
                raise ValueError(f"{RELIABILITY_TABLE} has no OlmoBaseEval Easy task in the group {group}")
            items.append((group, f"{group} ({len(members)})", members))
    table = [
        TradeoffRow(item, label, members, float(tasks.loc[list(members), "proportional_mean"].mean()))
        for item, label, members in items
    ]
    every = tuple(tasks.index)
    proportional = float(aggregate["proportional_mean"].iloc[0])
    table.append(TradeoffRow(ALL_TASKS, f"{ALL_TASKS} ({len(every)})", every, proportional))
    return table


def seed_runs(validation: pd.DataFrame, mixture: str, proposal_objective: str) -> pd.DataFrame:
    """The runs of ``mixture`` at the three trainer seeds; raises ``ValueError`` unless they are those three and
    were proposed for ``proposal_objective``."""
    runs = validation[(validation["mixture_id"] == mixture) & (validation["launch"] != RERUN_LAUNCH)]
    if sorted(runs["trainer_seed"]) != list(TRAINER_SEEDS):
        raise ValueError(f"{mixture} has trainer seeds {sorted(runs['trainer_seed'])}, not {TRAINER_SEEDS}")
    if set(runs["proposal_objective"]) != {proposal_objective}:
        raise ValueError(
            f"{mixture} was proposed for {sorted(set(runs['proposal_objective']))}, not {proposal_objective}"
        )
    return runs


def tradeoff_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each panel's bars: a method's mean change from the proportional mean in each row, over its three runs.

    A task's row averages its tasks' BPB; ``All`` is the objective's recorded aggregate, ``bpb__<objective>``.
    """
    validation = read_table(root, "validation_3e18")
    reliability = read_table(root, RELIABILITY_TABLE)
    rows = {objective: tradeoff_rows(reliability, objective) for objective in OBJECTIVES}
    bars, runs_used = [], []
    for panel in TRADEOFF_PANELS:
        for method in TRADEOFF_METHODS:
            runs = seed_runs(validation, panel.mixtures[method], panel.proposal_objective)
            runs_used += [
                {"panel": panel.proposal_objective, "method": method.value, "observation_id": run, "trainer_seed": seed}
                for run, seed in zip(runs["observation_id"], runs["trainer_seed"], strict=True)
            ]
            for row in rows[panel.objective]:
                if row.item == ALL_TASKS:
                    measured = runs[f"{OUTCOME_PREFIX}{panel.objective}"]
                else:
                    measured = runs[[f"{OUTCOME_PREFIX}{task}" for task in row.tasks]].mean(axis=1, skipna=False)
                if measured.isna().any():
                    raise ValueError(f"a run of {panel.mixtures[method]} lacks a value of {row.item}")
                bars.append(
                    {
                        "panel": panel.proposal_objective,
                        "objective": panel.objective,
                        "proposal_objective": panel.proposal_objective,
                        "method": method.value,
                        "mixture_id": panel.mixtures[method],
                        "item": row.item,
                        "label": row.label,
                        "tasks": len(row.tasks),
                        "proportional_bpb": row.proportional_bpb,
                        "delta_bpb": float(measured.mean()) - row.proportional_bpb,
                        "optimized": row.item in panel.optimized,
                    }
                )
    return {"bars": pd.DataFrame(bars), "runs": pd.DataFrame(runs_used)}


def decade_label(value: float, _position: int) -> str:
    return f"{value:g}".replace("-", "\N{MINUS SIGN}")


def draw_tradeoff_panel(axis: Axes, bars: pd.DataFrame, title: str) -> None:
    """One panel of Figure 11: each row's bars, MARINER's filled and Olmix's outlined, its optimized rows shaded."""
    items = list(dict.fromkeys(bars["item"]))
    labels = bars.drop_duplicates("item").set_index("item").loc[items]
    positions = np.arange(len(items), dtype=float)
    for position, optimized in zip(positions, labels["optimized"], strict=True):
        if optimized:
            axis.axhspan(position - 0.5, position + 0.5, color=OPTIMIZED_COLOR, linewidth=0, zorder=0)
    for offset, method in zip((-BAR_HEIGHT / 2, BAR_HEIGHT / 2), TRADEOFF_METHODS, strict=True):
        change = bars[bars["method"] == method.value].set_index("item").loc[items, "delta_bpb"]
        color = METHOD_COLORS[method]
        style = {"color": color} if method == Method.MARINER else {"facecolor": "white", "edgecolor": color}
        axis.barh(positions + offset, change, height=BAR_HEIGHT, linewidth=1.0, zorder=2, **style)
    axis.axvline(0.0, color=NEUTRAL, linewidth=0.8, zorder=1)
    axis.set_xscale("symlog", linthresh=SYMLOG_THRESHOLD, linscale=SYMLOG_LINSCALE)
    low = min(-3 * SYMLOG_THRESHOLD, LIMIT_MARGIN * float(bars["delta_bpb"].min()))
    high = max(3 * SYMLOG_THRESHOLD, LIMIT_MARGIN * float(bars["delta_bpb"].max()))
    axis.set_xlim(low, high)
    ticks = sorted(sign * decade for decade in SYMLOG_DECADES for sign in (-1, 1) if low <= sign * decade <= high)
    axis.xaxis.set_major_locator(FixedLocator([*ticks, 0.0]))
    axis.xaxis.set_major_formatter(FuncFormatter(decade_label))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks(positions, labels["label"].tolist())
    axis.set_ylim(len(items) - 0.5, -0.5)
    axis.grid(False, axis="y")
    axis.set_xlabel("Change from proportional mixing (BPB, symlog)")
    axis.set_title(title)


def draw_component_tradeoffs(root: Path) -> FigureData:
    """Figure 11: each row's change from proportional mixing at 3e18 FLOPs, for the mixtures proposed for each
    objective and for a subset of its components."""
    values = tradeoff_values(root)
    bars = values["bars"]
    layout = [[panel.proposal_objective for panel in TRADEOFF_PANELS if panel.objective == o] for o in OBJECTIVES]
    figure, axes = mosaic(layout, columns=2, rows=len(layout), legend_height=0.4)
    for panel in TRADEOFF_PANELS:
        draw_tradeoff_panel(axes[panel.proposal_objective], bars[bars["panel"] == panel.proposal_objective], panel.title)
    for row in layout:
        for name in row[1:]:  # the rows of an objective's panels are its first panel's
            axes[name].tick_params(axis="y", labelleft=False)
    figure_legend(
        figure,
        [
            Patch(color=METHOD_COLORS[Method.MARINER], label=METHOD_LABELS[Method.MARINER]),
            Patch(facecolor="white", edgecolor=METHOD_COLORS[Method.OLMIX], label=METHOD_LABELS[Method.OLMIX]),
            Patch(color=OPTIMIZED_COLOR, label="rows the mixtures were optimized for"),
        ],
        columns=3,
    )
    return FigureData(figure, values)
