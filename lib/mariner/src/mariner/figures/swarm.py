# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on the main swarm's mixtures (Appendices C.2 and D.1): Figures 23
(``fig:a-deletion``) and 24 (``fig:a-support``).

- ``fig:a-deletion`` (Figure 23): the swarm's 39 single-bucket removals, each the proportional mixture without one
  bucket, against its eleven proportional runs (the ten repeats and its own proportional run). For the OlmoBaseEval Easy
  mean and each of its 23 tasks, whose components are averaged within each run, the removal statistic is
  T = (y - mean) / (sd sqrt(1 + 1/n)) over the n proportional runs. A removal without effect gives T a t distribution
  with n - 1 degrees of freedom, and the two-sided p-values are corrected with Holm's method across each task's 39
  removals. The tasks, their groups and the mean's signal-to-noise ratio come from ``fits_reliability``, the
  components from ``tasks``, and the buckets' names, sources and pools from ``partition``.
- ``fig:a-support`` (Figure 24): each bucket's materialized epochs in the swarm's 280 fit runs and in MARINER's
  proposals for both objectives, beside proportional sampling's. The swarm's zero epochs (each bucket's removal run
  among them) and the proposals' zero weights sit in a shaded column, as in the paper's figure; positive epochs are on
  a log axis. A proposal outside the swarm's range of positive epochs for a bucket is drawn hollow.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from mariner.data import (
    BUCKET_ID,
    CALIBRATION_RUN,
    COMMON_CRAWL,
    DELETION,
    DESIGN_KIND,
    MAIN_SWARM,
    MIXTURE_ID,
    PARTITION_TABLE,
    POOL_TOKENS,
    PROPOSAL_OBJECTIVE,
    RELIABILITY_COLLAPSED_SOURCE,
    RELIABILITY_SOURCES,
    RELIABILITY_TABLE,
    REPEAT,
    RUN_LABEL,
    RUN_TABLE,
    SOURCES,
    TASK_ID,
    VALIDATION_TABLE,
    ObjectiveName,
    bucket_names,
    bucket_sources,
    measured_outcomes,
    mixture_weights,
    objective_from_tasks,
    partition_from_table,
    proportional_epochs,
    read_table,
    release_runs,
    release_tasks,
    reliability_rows,
    require_columns,
    swarm_from_runs,
)
from mariner.figures.analysis import EPOCH_COLUMNS, mixture_epochs
from mariner.figures.layout import (
    METHOD_COLORS,
    NEUTRAL,
    OBJECTIVE_TITLES,
    SEPARATOR,
    FigureData,
    figure_legend,
    frameless,
    source_rows,
    text_color,
)
from mariner.methods import Method

BUCKET_COLUMNS = [BUCKET_ID, "paper_name", "pool", "quality_tier", POOL_TOKENS]


def partition_table(root: Path) -> pd.DataFrame:
    """The ``partition`` table with the columns these figures read: names, sources and pools."""
    table = read_table(root, PARTITION_TABLE)
    require_columns(table, BUCKET_COLUMNS, PARTITION_TABLE)
    return table


# ---------------------------------------------------------------------------------------------------------------
# Figure 23: the single-bucket removals' effects on each OlmoBaseEval Easy task
# ---------------------------------------------------------------------------------------------------------------

REMOVAL_OBJECTIVE = ObjectiveName.OBE_EASY
MEAN_TASK = "OlmoBaseEval Easy mean"
RELIABILITY_COLUMNS = [TASK_ID, "group", "task", "subtasks", "snr"]
HOLM_LEVEL = 0.05
T_LIMIT = 6.0  # the colors clip |T| here
QUALITY_SUFFIXES = {"high": " (H)", "low": " (L)"}  # how a Common Crawl topic bucket's name ends, by its quality
HALVES = ("high", "low")  # the qualities of a topic row's left and right half-cells
SIDES = ("left", "right")  # the heatmap rows' columns of the buckets in their left and right half-cells
REMOVAL_COLORMAP = "RdBu_r"  # red: the removal raises the loss
DARK_CELL = 0.3  # a cell whose color lies this far from the center of the scale takes white text
CELL_WIDTH = 0.3  # inches per task column
CELL_HEIGHT = 0.21  # inches per row
HEATMAP_MARGINS = (3.0, 2.6)  # inches beside and below the cells: row labels and source names; column labels and bar


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm's adjusted p-values: each p-value, in increasing order, times the number of hypotheses not yet rejected,
    made monotone and capped at one."""
    order = np.argsort(p_values)
    scaled = (len(p_values) - np.arange(len(p_values))) * p_values[order]
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def removal_statistic(removals: np.ndarray, proportional: np.ndarray) -> np.ndarray:
    """Each removal's T against the proportional runs' mean and standard deviation, as one more run."""
    scale = proportional.std(ddof=1) * math.sqrt(1.0 + 1.0 / len(proportional))
    return (removals - proportional.mean()) / scale


class RemovalTask(NamedTuple):
    """A column of Figure 23: a task or the OlmoBaseEval Easy mean, its group (none for the mean), and its components,
    whose losses it averages within each run."""

    name: str
    group: str | None
    components: list[str]


def removal_tasks(root: Path, components: tuple[str, ...]) -> tuple[list[RemovalTask], float]:
    """The heatmap's columns, the OlmoBaseEval Easy mean and its tasks in ``fits_reliability``'s order, and the mean's
    signal-to-noise ratio.

    Raises ``ValueError`` unless ``fits_reliability``'s components are ``components``, each task has as many as it
    records, and the table has one row of the mean.
    """
    reliability = read_table(root, RELIABILITY_TABLE)
    rows = reliability_rows(reliability, RELIABILITY_SOURCES[REMOVAL_OBJECTIVE])
    require_columns(rows, RELIABILITY_COLUMNS, RELIABILITY_TABLE)
    named = rows[rows[TASK_ID].notna()]
    if set(named[TASK_ID]) != set(components):
        raise ValueError(f"{RELIABILITY_TABLE}'s components are not the {REMOVAL_OBJECTIVE} tasks of the tasks table")
    task_of = named.set_index(TASK_ID)["task"]
    columns = [RemovalTask(MEAN_TASK, None, list(components))]
    tasks = reliability_rows(reliability, RELIABILITY_COLLAPSED_SOURCE)
    for task, group, count in tasks.loc[tasks["group"].notna(), ["task", "group", "subtasks"]].itertuples(index=False):
        members = [component for component in components if task_of[component] == task]
        if len(members) != count:
            raise ValueError(f"{RELIABILITY_TABLE} gives {task} {count} components, not {len(members)}")
        columns.append(RemovalTask(str(task), str(group), members))
    mean = rows.loc[rows[TASK_ID].isna(), "snr"]
    if len(mean) != 1:
        raise ValueError(f"{RELIABILITY_TABLE} has {len(mean)} rows of the {REMOVAL_OBJECTIVE} mean, not one")
    return columns, float(mean.iloc[0])


def removal_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The heatmap's rows by source, with the buckets of their left and right half-cells: each Common Crawl topic with
    its high-quality bucket on the left and its low-quality one on the right, by their joint pool, and every other
    bucket alone (in both halves), by its pool.

    Raises ``ValueError`` unless each topic has one bucket of each quality, named as such.
    """
    names, sources = bucket_names(table), bucket_sources(table)
    indexed = table.set_index(BUCKET_ID)
    rows = []
    for source in SOURCES:
        buckets = [bucket for bucket in table[BUCKET_ID] if sources[bucket] == source]
        if source != COMMON_CRAWL:
            rows += [{"source": source, "label": names[bucket], "left": bucket, "right": bucket} for bucket in buckets]
            continue
        topics: dict[str, dict[str, str]] = {}
        for bucket in buckets:
            quality = indexed.loc[bucket, "quality_tier"]
            suffix = QUALITY_SUFFIXES[quality]
            if not names[bucket].endswith(suffix):
                raise ValueError(f"the {quality}-quality bucket {bucket} is named {names[bucket]}")
            pair = topics.setdefault(names[bucket].removesuffix(suffix), {})
            if quality in pair:
                raise ValueError(f"two {quality}-quality buckets are named {names[bucket]}")
            pair[quality] = bucket
        if any(set(pair) != set(HALVES) for pair in topics.values()):
            raise ValueError("every Common Crawl topic needs one high- and one low-quality bucket")
        rows += [
            {"source": source, "label": topic, **dict(zip(SIDES, (pair[quality] for quality in HALVES), strict=True))}
            for topic, pair in topics.items()
        ]
    frame = pd.DataFrame(rows)
    pools = indexed[POOL_TOKENS]
    split = frame["left"] != frame["right"]
    frame["pool_tokens"] = pools[frame["left"]].to_numpy() + np.where(split, pools[frame["right"]].to_numpy(), 0)
    rank = frame["source"].map({source: index for index, source in enumerate(SOURCES)})
    order = np.lexsort((-frame["pool_tokens"].to_numpy(), rank.to_numpy()))
    return frame.iloc[order].reset_index(drop=True)


def deletion_values(root: Path) -> dict[str, pd.DataFrame]:
    """Every removal's T, p-value and Holm-adjusted p-value for the mean and each task, the heatmap's rows, and the
    mean's signal-to-noise ratio, the number of proportional runs and Holm's first-step threshold on |T|.

    Raises ``ValueError`` unless each removal run leaves out one bucket and every bucket is removed once.
    """
    runs = release_runs(root, MAIN_SWARM)
    require_columns(runs, [DESIGN_KIND, RUN_LABEL], RUN_TABLE)
    table = partition_table(root)
    partition = partition_from_table(table)
    components = objective_from_tasks(release_tasks(root), REMOVAL_OBJECTIVE).tasks
    columns, mean_snr = removal_tasks(root, components)
    removals = runs[runs[DESIGN_KIND] == DELETION]
    left_out = mixture_weights(removals, partition.bucket_ids, RUN_TABLE) == 0.0
    if not (left_out.sum(axis=1) == 1).all():
        raise ValueError(f"each {DELETION} run must leave out exactly one bucket")
    removed = [partition.bucket_ids[index] for index in left_out.argmax(axis=1)]
    if sorted(removed) != sorted(partition.bucket_ids):
        raise ValueError(f"the {DELETION} runs must remove every bucket once")
    proportional = runs[(runs[DESIGN_KIND] == REPEAT) | (runs[RUN_LABEL] == CALIBRATION_RUN)]
    removal_losses = measured_outcomes(removals, components)
    proportional_losses = measured_outcomes(proportional, components)
    names, sources = bucket_names(table), bucket_sources(table)
    degrees = len(proportional) - 1
    cells = []
    for task in columns:
        t = removal_statistic(
            removal_losses[task.components].mean(axis=1).to_numpy(),
            proportional_losses[task.components].mean(axis=1).to_numpy(),
        )
        p = 2.0 * stats.t.sf(np.abs(t), degrees)
        cells += [
            {
                "task": task.name,
                "group": task.group,
                "components": len(task.components),
                BUCKET_ID: bucket,
                "bucket": names[bucket],
                "source": sources[bucket],
                "t": value,
                "p": p_value,
                "holm_p": holm_p,
                "significant": holm_p < HOLM_LEVEL,
            }
            for bucket, value, p_value, holm_p in zip(removed, t, p, holm(p), strict=True)
        ]
    summary = {
        "mean_task": MEAN_TASK,
        "mean_snr": mean_snr,
        "proportional_runs": len(proportional),
        "holm_first_step": float(stats.t.ppf(1.0 - HOLM_LEVEL / (2 * len(removed)), degrees)),
    }
    return {"cells": pd.DataFrame(cells), "rows": removal_rows(table), "summary": pd.DataFrame([summary])}


def draw_deletion_matrix(root: Path) -> FigureData:
    """Figure 23: each removal's T for the OlmoBaseEval Easy mean and its 23 tasks, a Common Crawl topic's row split
    into its high- (left) and low-quality (right) buckets; numbers mark the removals that pass Holm's correction."""
    values = deletion_values(root)
    cells, rows, summary = values["cells"], values["rows"], values["summary"].iloc[0]
    tasks = list(dict.fromkeys(cells["task"]))
    columns = cells.drop_duplicates("task").set_index("task")
    statistic = cells.set_index(["task", BUCKET_ID])
    matrix = np.array(
        [[statistic.loc[(task, row[side]), "t"] for task in tasks for side in SIDES] for _, row in rows.iterrows()]
    )
    size = (len(tasks) * CELL_WIDTH + HEATMAP_MARGINS[0], len(rows) * CELL_HEIGHT + HEATMAP_MARGINS[1])
    figure = Figure(figsize=size, layout="constrained")
    axis = figure.subplots()
    norm = Normalize(-T_LIMIT, T_LIMIT)
    image = axis.imshow(np.clip(matrix, -T_LIMIT, T_LIMIT), cmap=REMOVAL_COLORMAP, norm=norm, aspect="auto")
    for index, row in rows.iterrows():
        split = row["left"] != row["right"]
        for column, task in enumerate(tasks):
            for offset, side in enumerate(SIDES if split else SIDES[:1]):
                cell = statistic.loc[(task, row[side])]
                if not cell["significant"]:
                    continue
                color = text_color(float(norm(np.clip(cell["t"], -T_LIMIT, T_LIMIT))), DARK_CELL)
                axis.text(
                    2 * column + (offset if split else 0.5),
                    index,
                    f"{cell['t']:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.0 if split else 6.0,
                    color=color,
                )
    labels = [task if count == 1 else f"{task} ({count})" for task, count in columns["components"].items()]
    axis.set_xticks(2 * np.arange(len(tasks)) + 0.5, labels, rotation=90)
    split_rows = rows["left"] != rows["right"]
    row_labels = [f"{label} (H | L)" if split else label for label, split in zip(rows["label"], split_rows, strict=True)]
    axis.set_yticks(range(len(rows)), row_labels)
    axis.tick_params(length=0, labelsize=6.5)
    frameless(axis)
    for column in range(1, len(tasks)):
        axis.axvline(2 * column - 0.5, color="white", linewidth=0.6)
    groups = columns["group"].fillna("").to_numpy()
    starts = [index for index in range(len(tasks)) if index == 0 or groups[index] != groups[index - 1]]
    for start in starts[1:]:
        axis.axvline(2 * start - 0.5, **SEPARATOR)
    for start, end in zip(starts, [*starts[1:], len(tasks)], strict=True):
        if not groups[start]:
            continue  # the mean's column belongs to no group
        axis.annotate(
            groups[start],
            xy=(start + end - 0.5, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    source_rows(axis, rows["source"])
    colorbar = figure.colorbar(image, ax=axis, location="bottom", shrink=0.5, aspect=40)
    colorbar.set_ticks([-T_LIMIT, -T_LIMIT / 2, 0.0, T_LIMIT / 2, T_LIMIT], labels=["≤ -6", "-3", "0", "3", "≥ 6"])
    colorbar.set_label("T: the removal's loss minus the proportional runs' mean, in their standard deviations")
    axis.set_title(
        f"Removals from the proportional mixture against its {int(summary['proportional_runs'])} runs. Numbers: T where"
        f" Holm's correction passes (first step: |T| > {summary['holm_first_step']:.2f}).\n{summary['mean_task']}:"
        f" {summary['mean_snr']:.1f} times its repeat standard deviation across the swarm.",
        fontsize=7.5,
        pad=18,
    )
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 24: the epochs the swarm sampled, against MARINER's proposals
# ---------------------------------------------------------------------------------------------------------------

PROPOSAL_COLUMNS = [MIXTURE_ID, "method", PROPOSAL_OBJECTIVE, "variant"]  # what mixture_epochs checks of a run
SUPPORT_MARKERS = {
    ObjectiveName.UNCHEATABLE: ("D", METHOD_COLORS[Method.MARINER]),
    ObjectiveName.OBE_EASY: ("o", "#D95F02"),
}
SWARM_DOTS = {"marker": ".", "markersize": 1.6, "color": "#7D93AB", "alpha": 0.35, "linestyle": "none"}
JITTER = 0.3  # a bucket's dots spread this far above and below its row
JITTER_SEED = 20_261_107
ZERO_SHADE = "#EEF0F3"
SUPPORT_WIDTH = 7.4  # inches
SUPPORT_ROW_HEIGHT = 0.16  # inches per bucket
SUPPORT_MARGIN = 1.7  # inches above and below the rows: titles, axis labels and legend
ZERO_COLUMN_RATIO = (1, 14)  # the zero column's width against the log axis'


def support_values(root: Path) -> dict[str, pd.DataFrame]:
    """The swarm's epochs of each bucket in each run (zero included), each bucket's sampled range of positive epochs,
    MARINER's proposals' epochs with whether each lies outside that range, and proportional sampling's epochs, the
    buckets by decreasing pool."""
    table = partition_table(root)
    partition = partition_from_table(table)
    swarm = swarm_from_runs(release_runs(root, MAIN_SWARM), partition, ())
    epochs = pd.DataFrame(swarm.exposures, index=pd.Index(swarm.runs, name="run"), columns=list(partition.bucket_ids))
    validation = read_table(root, VALIDATION_TABLE)
    require_columns(validation, PROPOSAL_COLUMNS, VALIDATION_TABLE)
    names = bucket_names(table)
    order = list(table.sort_values(POOL_TOKENS, ascending=False, kind="stable")[BUCKET_ID])
    positive = epochs.where(epochs > 0.0)
    ranges = pd.DataFrame(
        {
            BUCKET_ID: order,
            "bucket": names[order].to_numpy(),
            "pool_tokens": table.set_index(BUCKET_ID).loc[order, POOL_TOKENS].to_numpy(),
            "min_positive": positive.min()[order].to_numpy(),
            "max": epochs.max()[order].to_numpy(),
        }
    )
    proposals = []
    for column in EPOCH_COLUMNS:
        if column.method != Method.MARINER:
            continue
        proposed = mixture_epochs(validation, column, partition.bucket_ids, partition.epochs_per_unit_weight)
        by_bucket = dict(zip(partition.bucket_ids, proposed, strict=True))
        proposals += [
            {
                "objective": column.objective,
                BUCKET_ID: row[BUCKET_ID],
                "bucket": row["bucket"],
                "epochs": by_bucket[row[BUCKET_ID]],
                "outside": (
                    by_bucket[row[BUCKET_ID]] > row["max"] or 0.0 < by_bucket[row[BUCKET_ID]] < row["min_positive"]
                ),
            }
            for _, row in ranges.iterrows()
        ]
    dots = epochs[order].reset_index().melt(id_vars="run", var_name=BUCKET_ID, value_name="epochs")
    return {
        "swarm": dots,
        "ranges": ranges,
        "proposals": pd.DataFrame(proposals),
        "proportional": pd.DataFrame({"epochs": [proportional_epochs(partition)]}),
    }


def draw_swarm_support(root: Path) -> FigureData:
    """Figure 24: each bucket's epochs in the swarm's runs (faint dots) and in MARINER's proposals. Zero epochs, the
    swarm's and the proposals', sit in a shaded column; positive epochs are on a log axis. The buckets run by decreasing
    pool."""
    values = support_values(root)
    swarm, ranges, proposals = values["swarm"], values["ranges"], values["proposals"]
    proportional = float(values["proportional"]["epochs"].iloc[0])
    row_of = {bucket: index for index, bucket in enumerate(ranges[BUCKET_ID])}
    height = len(row_of) * SUPPORT_ROW_HEIGHT + SUPPORT_MARGIN
    figure = Figure(figsize=(SUPPORT_WIDTH, height), layout="constrained")
    zero_axis, axis = figure.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": ZERO_COLUMN_RATIO})
    jitter = np.random.default_rng(JITTER_SEED).uniform(-JITTER, JITTER, len(swarm))
    jittered = swarm[BUCKET_ID].map(row_of).to_numpy() + jitter
    sampled_zero = (swarm["epochs"] == 0.0).to_numpy()
    axis.plot(swarm["epochs"].to_numpy()[~sampled_zero], jittered[~sampled_zero], zorder=2, **SWARM_DOTS)
    zero_axis.plot(np.zeros(int(sampled_zero.sum())), jittered[sampled_zero], zorder=2, **SWARM_DOTS)
    axis.axvline(proportional, color=NEUTRAL, linestyle="--", linewidth=0.8, zorder=3)
    handles = [Line2D([], [], label=f"the swarm's {swarm['run'].nunique()} mixtures", **SWARM_DOTS | {"markersize": 5})]
    for objective, (marker, color) in SUPPORT_MARKERS.items():
        rows = proposals[proposals["objective"] == objective]
        y = rows[BUCKET_ID].map(row_of)
        zero = rows["epochs"] == 0.0
        for outside, face, markersize in ((False, color, 4.0), (True, "white", 6.5)):
            shown = ~zero & (rows["outside"] == outside)
            axis.plot(
                rows.loc[shown, "epochs"],
                y[shown],
                marker=marker,
                color=color,
                markerfacecolor=face,
                markersize=markersize,
                linestyle="none",
                zorder=5,
            )
        zero_axis.plot(np.zeros(int(zero.sum())), y[zero], marker=marker, color=color, markersize=4.0, linestyle="none")
        title = OBJECTIVE_TITLES[objective]
        handles.append(Line2D([], [], marker=marker, color=color, linestyle="none", label=f"MARINER's {title} proposal"))
    handles += [
        Line2D(
            [],
            [],
            marker="D",
            color=NEUTRAL,
            markerfacecolor="white",
            markersize=6.5,
            linestyle="none",
            label="hollow: outside the swarm's sampled positive epochs",
        ),
        Line2D([], [], color=NEUTRAL, linestyle="--", label=f"proportional sampling: {proportional:.2f} epochs"),
    ]
    axis.set_xscale("log")
    axis.set_xlabel("Materialized epochs in the target run (positive, log scale)")
    axis.tick_params(axis="y", length=0)
    zero_axis.set_facecolor(ZERO_SHADE)
    zero_axis.set_xlim(-1.0, 1.0)
    zero_axis.set_xticks([0.0], ["0"])
    zero_axis.set_title("zero", fontsize=7.5)
    zero_axis.set_yticks(range(len(row_of)), ranges["bucket"].tolist())
    zero_axis.tick_params(axis="y", length=0, labelsize=6.5)
    zero_axis.set_ylim(len(row_of) - 0.5, -0.5)  # the largest pool on top
    figure_legend(figure, handles, columns=2)
    return FigureData(figure, values)
