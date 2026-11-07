# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on two-phase schedules (Appendix E): Figures 25 (``fig:a-curricula-sweeps``), 26
(``fig:a-switch-smoothing``) and 27 (``fig:a-ramp-schedules``).

A two-phase schedule trains a 210M-parameter model on one Nemotron-CC/StarCoder mixture for the first 80% of training
and on another for the learning-rate decay. A gain is the programming-languages BPB of a selected single-phase mixture
minus that of a selected two-phase schedule, so a positive gain favors the schedule.

- ``fig:a-curricula-sweeps`` (Figure 25): the gains measured on five fresh paired seeds in each of the 28 blocks of
  token budget and StarCoder pool (``starcoder_curricula``), with their 95% paired-t intervals and whether they stay
  positive after Holm correction over the blocks: against the budget for four pools (A), and across all seven pools at
  the longest budget (B). Panel C draws the gains of the fitted loss surfaces along three matched-compute paths
  (``starcoder_curricula_matched_nd``) and, at the one cell that qualified, the gain measured on eight fresh pairs
  (``starcoder_curricula_matched_nd_confirmation``) with its 95% paired-t interval.
- ``fig:a-switch-smoothing`` (Figure 26): the change, around an abrupt switch, of the gap between two-phase runs and
  their paired single-phase runs, relative to the gap 64 steps before the switch (A); and the gain in final
  programming-languages BPB of linear ramps over the abrupt switch at the same seed (B), both from sub-tables of
  ``starcoder_transition``. The ramps' widths come from ``starcoder_ramp_designs``.
- ``fig:a-ramp-schedules`` (Figure 27): the StarCoder share of each 2,048-sequence block under the abrupt switch and
  each ramp, as the data loader allocated it (``starcoder_ramp_designs``' per-block counts), for the first data seed,
  as the paper's figure shows one seed; the released runs of the other seeds have the same blocks. Its values also
  hold each run's StarCoder total and the part of it drawn before the decay, at every seed.

The panels draw the intervals the dataset releases, except the confirmation's, which the release gives as its eight
pairs.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from mariner.data import read_table, require_columns
from mariner.figures.layout import (
    CONFIDENCE,
    NEUTRAL,
    FigureData,
    figure_legend,
    flops_label,
    mosaic,
    panel_legend,
    panels,
    scale_colors,
    t_interval,
)
from mariner.figures.repetition import FULL_POOL, pool_colors, pool_label, repetition_label
from mariner.figures.starcoder import tokens_label
from mariner.optimize import MIXTURE_BLOCK_SIZE

GAIN_SCALE = 1e3  # gains are drawn in 1e-3 BPB
GAIN_LABEL = "Two-phase gain (1e-3 BPB)"
ZERO_STYLE = {"color": NEUTRAL, "linewidth": 0.8, "zorder": 1}
DODGE = 0.04  # log-spaced offset between panel A's pools at one budget

# ---------------------------------------------------------------------------------------------------------------
# Figure 25: two-phase gains across token budgets, StarCoder repetition and matched compute
# ---------------------------------------------------------------------------------------------------------------

CURRICULA_TABLE = "starcoder_curricula"
MATCHED_TABLE = "starcoder_curricula_matched_nd"
CONFIRMATION_TABLE = "starcoder_curricula_matched_nd_confirmation"
BLOCK_COLUMNS = {
    "materialized_tokens": "train_tokens",
    "support_id": "support_id",
    "support_order": "support_order",
    "epoch_multiplier": "epoch_multiplier",
    "pair_count": "pairs",
    "mean_gain_bpb": "gain_bpb",
    "ci95_low": "low_bpb",
    "ci95_high": "high_bpb",
    "paired_t_holm_p": "holm_p",
    "holm_positive": "holm_positive",
}
BUDGET_PANEL_POOLS = (FULL_POOL, "m025", "m100", "m400")  # panel A's StarCoder pools, by starcoder_curricula's id
CELL_COLUMNS = {
    "cell_id": "cell_id",
    "track_memberships": "track_memberships",
    "materialized_tokens": "train_tokens",
    "compute_flops": "compute_flops",
    "policy_class_gap_bpb": "gain_bpb",
    "stage1_policy_class_gain_bpb": "stage1_gain_bpb",
    "confirmation_eligible": "confirmation_eligible",
}
CONFIRMATION_COLUMNS = ["cell_id", "pair_seed", "gain_tied_minus_untied_bpb"]


class ComputePath(StrEnum):
    """Panel C's matched-compute paths from the 210M-parameter model at 1.0B tokens, by ``track_memberships``."""

    TOKENS = "increase_d"
    TOKENS_AND_PARAMETERS = "increase_nd"
    PARAMETERS = "increase_n"


PATH_LABELS = {
    ComputePath.TOKENS: "tokens, at 210M parameters",
    ComputePath.TOKENS_AND_PARAMETERS: "tokens and parameters",
    ComputePath.PARAMETERS: "parameters, at 1.0B tokens",
}
PATH_COLORS = {
    ComputePath.TOKENS: "#2A9D8F",
    ComputePath.TOKENS_AND_PARAMETERS: "#3D5A80",
    ComputePath.PARAMETERS: "#9A8C98",
}
CONFIRMATION_STYLE = {"marker": "*", "markersize": 11, "color": "black", "markeredgecolor": "white", "zorder": 6}


def curricula_blocks(root: Path) -> pd.DataFrame:
    """The 28 blocks of token budget and StarCoder pool: each fresh-seed gain with its 95% paired-t interval, its
    Holm-adjusted p-value and whether it stays positive, by pool and then by budget."""
    table = read_table(root, CURRICULA_TABLE)
    require_columns(table, list(BLOCK_COLUMNS), CURRICULA_TABLE)
    blocks = table[list(BLOCK_COLUMNS)].rename(columns=BLOCK_COLUMNS)
    return blocks.sort_values(["support_order", "train_tokens"], ignore_index=True)


def path_cells(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The matched-compute cells, and each path's cells by compute (a cell can lie on several paths).

    Raises ``ValueError`` unless every cell's ``track_memberships`` is a list of known paths.
    """
    table = read_table(root, MATCHED_TABLE)
    require_columns(table, list(CELL_COLUMNS), MATCHED_TABLE)
    cells = table[list(CELL_COLUMNS)].rename(columns=CELL_COLUMNS)
    rows = []
    for cell in cells.itertuples():
        memberships = ast.literal_eval(str(cell.track_memberships))
        if not isinstance(memberships, list) or not set(memberships) <= set(ComputePath):
            raise ValueError(f"{MATCHED_TABLE} puts {cell.cell_id} on unknown paths: {cell.track_memberships}")
        rows += [
            {"path": path, "cell_id": cell.cell_id, "compute_flops": cell.compute_flops, "gain_bpb": cell.gain_bpb}
            for path in memberships
        ]
    paths = pd.DataFrame(rows).sort_values(["path", "compute_flops"], ignore_index=True)
    return cells.drop(columns="track_memberships"), paths


def confirmation_gain(root: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """The confirmed cell's measured gain: the mean over its fresh pairs with a 95% paired-t interval, and how many
    pairs favor the two-phase schedule. Raises ``ValueError`` unless the pairs come from one cell of ``cells``."""
    pairs = read_table(root, CONFIRMATION_TABLE)
    require_columns(pairs, CONFIRMATION_COLUMNS, CONFIRMATION_TABLE)
    confirmed = pairs["cell_id"].unique()
    if len(confirmed) != 1 or confirmed[0] not in set(cells["cell_id"]):
        raise ValueError(f"{CONFIRMATION_TABLE} holds pairs of {list(confirmed)}, not of one {MATCHED_TABLE} cell")
    gains = pairs["gain_tied_minus_untied_bpb"].to_numpy(float)
    mean, half = t_interval(gains)
    compute = cells.loc[cells["cell_id"] == confirmed[0], "compute_flops"].iloc[0]
    row = {
        "cell_id": confirmed[0],
        "compute_flops": compute,
        "pairs": len(gains),
        "positive_pairs": int((gains > 0).sum()),
        "gain_bpb": mean,
        "low_bpb": mean - half,
        "high_bpb": mean + half,
    }
    return pd.DataFrame([row])


def draw_gains(axis: Axes, x: np.ndarray, rows: pd.DataFrame, colors: Sequence[tuple[float, ...] | str]) -> None:
    """Gains at ``x`` with their intervals, a filled marker where the gain stays positive after Holm correction."""
    gain = rows["gain_bpb"].to_numpy(float) * GAIN_SCALE
    low, high = rows["low_bpb"].to_numpy(float) * GAIN_SCALE, rows["high_bpb"].to_numpy(float) * GAIN_SCALE
    for position, value, bottom, top, positive, color in zip(
        x, gain, low, high, rows["holm_positive"], colors, strict=True
    ):
        axis.plot([position, position], [bottom, top], color=color, linewidth=0.9, alpha=0.8, zorder=2)
        face = color if positive else "white"
        axis.plot([position], [value], "o", color=color, markerfacecolor=face, markersize=5, zorder=4)


def draw_curricula_sweeps(root: Path) -> FigureData:
    """Figure 25: two-phase gains against the token budget (A), StarCoder repetition at the longest budget (B), and
    training compute along matched-compute paths (C)."""
    blocks = curricula_blocks(root)
    cells, paths = path_cells(root)
    confirmation = confirmation_gain(root, cells)
    colors = pool_colors(list(blocks.sort_values("support_order")["support_id"]))
    pairs = blocks["pairs"].unique()
    interval = f"bars: {CONFIDENCE:.0%} paired-t over {', '.join(map(str, pairs))} fresh pairs"
    figure, axes = panels(3, legend_height=1.1)
    budget_axis, repetition_axis, compute_axis = axes
    budgets = np.sort(blocks["train_tokens"].unique())
    budget_rows = blocks[blocks["support_id"].isin(BUDGET_PANEL_POOLS)]
    handles = []
    for offset, (support, rows) in enumerate(budget_rows.groupby("support_id", sort=False)):
        color = colors[str(support)]
        # Pools side by side at each budget, so that their intervals do not overlap.
        x = rows["train_tokens"].to_numpy(float) * np.exp(DODGE * (offset - (len(BUDGET_PANEL_POOLS) - 1) / 2))
        budget_axis.plot(x, rows["gain_bpb"] * GAIN_SCALE, color=color, linewidth=1.2, zorder=3)
        draw_gains(budget_axis, x, rows, [color] * len(rows))
        handles.append(Line2D([], [], color=color, marker="o", label=pool_label(rows["epoch_multiplier"].iloc[0])))
    panel_legend(budget_axis, handles, ncols=2)
    budget_axis.set_xscale("log")
    budget_axis.set_xticks(budgets, [tokens_label(budget) for budget in budgets])
    budget_axis.minorticks_off()
    budget_axis.set_xlabel("Training tokens D (210M-parameter model)")
    budget_axis.set_title("A. Gain against the token budget")
    longest = blocks[blocks["train_tokens"] == budgets[-1]].sort_values("support_order")
    positions = np.arange(len(longest), dtype=float)
    repetition_axis.plot(positions, longest["gain_bpb"] * GAIN_SCALE, color=NEUTRAL, linestyle=":", linewidth=0.9)
    draw_gains(repetition_axis, positions, longest, [colors[support] for support in longest["support_id"]])
    labels = [repetition_label(multiplier).replace(" ", "\n") for multiplier in longest["epoch_multiplier"]]
    repetition_axis.set_xticks(positions, labels)
    repetition_axis.set_xlabel("StarCoder repetition, relative to the target run's")
    repetition_axis.set_title(f"B. Gain at {tokens_label(budgets[-1])} tokens")
    panel_legend(
        repetition_axis,
        [
            Line2D([], [], color=NEUTRAL, marker="o", linestyle="none", label="positive after Holm correction"),
            Line2D(
                [], [], color=NEUTRAL, marker="o", markerfacecolor="white", linestyle="none", label="not significant"
            ),
            Line2D([], [], linestyle="none", label=f"{interval}; Holm over the {len(blocks)} blocks"),
        ],
    )
    for path in ComputePath:
        rows = paths[paths["path"] == path]
        compute_axis.plot(
            rows["compute_flops"],
            rows["gain_bpb"] * GAIN_SCALE,
            color=PATH_COLORS[path],
            marker="o",
            markersize=3.5,
            zorder=3,
        )
    star = confirmation.iloc[0]
    gain = star["gain_bpb"] * GAIN_SCALE
    compute_axis.plot(
        [star["compute_flops"]] * 2, [star["low_bpb"] * GAIN_SCALE, star["high_bpb"] * GAIN_SCALE], color="black"
    )
    compute_axis.plot([star["compute_flops"]], [gain], linestyle="none", **CONFIRMATION_STYLE)
    rungs = np.sort(paths.loc[paths["path"] == ComputePath.TOKENS, "compute_flops"].unique())
    compute_axis.set_xscale("log")
    compute_axis.set_xticks(rungs, [flops_label(compute) for compute in rungs])
    compute_axis.minorticks_off()
    compute_axis.set_xlabel("Training compute (FLOPs)")
    compute_axis.set_title("C. Fitted gain along matched-compute paths")
    handles = [
        Line2D([], [], color=PATH_COLORS[path], marker="o", markersize=3.5, label=f"fitted, adding {PATH_LABELS[path]}")
        for path in ComputePath
    ]
    measured = f"measured on {int(star['pairs'])} fresh pairs, {CONFIDENCE:.0%} paired-t"
    handles.append(Line2D([], [], linestyle="none", label=measured, **CONFIRMATION_STYLE))
    panel_legend(compute_axis, handles)
    for axis in axes:
        axis.set_ylabel(GAIN_LABEL)
        axis.axhline(0.0, **ZERO_STYLE)
    values = {
        "blocks": blocks,
        "budget_gains": budget_rows.reset_index(drop=True),
        "repetition_gains": longest.reset_index(drop=True),
        "fitted_gains": paths,
        "cells": cells,
        "confirmation": confirmation,
    }
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figures 26 and 27: the switch between the phases, abrupt or ramped
# ---------------------------------------------------------------------------------------------------------------

TRANSITION_TABLE = "starcoder_transition"
SUB_TABLE = "table"  # starcoder_transition's column naming each row's sub-table
RAMP_TABLE = "starcoder_ramp_designs"
# Panel A's switch: from 0.02 to 0.82 StarCoder at the decay onset, at the target run's repetition, against the paired
# single-phase runs of the average mixture (0.18 StarCoder).
SWITCH_ARM = "p3_vs_p4_m100a"
SWITCH_LAGS = (-256, 1_500)  # the steps after the switch that panel A shows
SWITCH_COLUMNS = ["arm", "p0", "p1", "lr_state", "switch_step", "n"]
GAP_COLUMNS = ["arm", "metric", "step", "lag", "n", "double_diff", "ci", "baseline_step"]
GAIN_COLUMNS = ["arm", "n", "pl_gain_over_october_abrupt", "pl_gain_ci_low", "pl_gain_ci_high", "ramp_worse_in"]
METRIC_LABELS = {"PL": "programming-languages", "C4": "C4"}
METRIC_COLORS = {"PL": "#1F6FB2", "C4": "#D9622B"}
INTERVAL_STYLE = {"marker": "o", "markersize": 3.5, "capsize": 2, "elinewidth": 0.9, "zorder": 4}
RAMP_COLUMNS = [
    "arm",
    "ramp_width",
    "data_seed",
    "stage",
    "start_step",
    "total_steps",
    "boundary_step",
    "steps_per_block",
    "starcoder_block_sequences",
    "starcoder_stage_sequences",
]
RAMP_WINDOW = (0.685, 0.915)  # the fractions of training that Figure 27 shows, around the decay at 0.8


def transition_rows(transition: pd.DataFrame, name: str, columns: list[str]) -> pd.DataFrame:
    """The ``columns`` of ``starcoder_transition``'s sub-table ``name``, whose other sub-tables' columns are empty.

    Raises ``ValueError`` when no row belongs to ``name`` or the sub-table leaves a column of ``columns`` empty.
    """
    require_columns(transition, [SUB_TABLE], TRANSITION_TABLE)
    rows = transition[transition[SUB_TABLE] == name].dropna(axis=1, how="all")
    if rows.empty:
        raise ValueError(f"{TRANSITION_TABLE} has no sub-table {name}")
    require_columns(rows, columns, f"{TRANSITION_TABLE} {name}")
    return rows[columns].reset_index(drop=True)


def unique_row(rows: pd.DataFrame, what: str) -> pd.Series:
    """The one distinct row of ``rows``; raises ``ValueError`` naming ``what`` when they differ or there are none."""
    distinct = rows.drop_duplicates()
    if len(distinct) != 1:
        raise ValueError(f"{what} has {len(distinct)} distinct settings, not one")
    return distinct.iloc[0]


def switch_values(transition: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Panel A: the traced switch's settings, and the change of its gaps at each step after the switch.

    Raises ``ValueError`` unless the offline gaps share one pair count and one baseline step and lie at the steps
    after the arm's switch.
    """
    settings = transition_rows(transition, "eval_double_differences", SWITCH_COLUMNS)
    switch = unique_row(settings[settings["arm"] == SWITCH_ARM], f"the switch {SWITCH_ARM}")
    gaps = transition_rows(transition, "switch_window_offline_double_differences", GAP_COLUMNS)
    gaps = gaps[(gaps["arm"] == SWITCH_ARM) & gaps["lag"].between(*SWITCH_LAGS)]
    shared = unique_row(gaps[["n", "baseline_step"]], f"the offline gaps of {SWITCH_ARM}")
    if not ((gaps["step"] - gaps["lag"]) == switch["switch_step"]).all():
        raise ValueError(f"the offline gaps of {SWITCH_ARM} are not placed after its switch at {switch['switch_step']}")
    row = {
        "arm": SWITCH_ARM,
        "p0": switch["p0"],
        "p1": switch["p1"],
        "lr_state": switch["lr_state"],
        "pairs": int(shared["n"]),
        "switch_step": switch["switch_step"],
        "baseline_step": shared["baseline_step"],
    }
    renamed = gaps.rename(columns={"double_diff": "gap_change_bpb", "ci": "half_width_bpb"})
    columns = ["metric", "lag", "step", "gap_change_bpb", "half_width_bpb"]
    return {"switch": pd.DataFrame([row]), "gaps": renamed[columns].sort_values(["metric", "lag"], ignore_index=True)}


def ramp_designs(root: Path, columns: list[str]) -> pd.DataFrame:
    """The ``columns`` of ``starcoder_ramp_designs``, one row per stage of each switch run."""
    designs = read_table(root, RAMP_TABLE)
    require_columns(designs, columns, RAMP_TABLE)
    return designs[columns]


def ramp_gains(transition: pd.DataFrame, designs: pd.DataFrame) -> pd.DataFrame:
    """Panel B: each ramp's gain in final programming-languages BPB over the abrupt switch at the same seed, with its
    released 95% paired-t interval and the number of seeds in which the ramp was worse, by ramp width."""
    gains = transition_rows(transition, "ramp_gains_over_october_abrupt", GAIN_COLUMNS)
    widths = designs.drop_duplicates("arm").set_index("arm")["ramp_width"]
    unknown = sorted(set(gains["arm"]) - set(widths.index))
    if unknown:
        raise ValueError(f"{RAMP_TABLE} has no ramp width for {', '.join(unknown)}")
    table = gains.assign(width=gains["arm"].map(widths)).rename(
        columns={
            "n": "pairs",
            "pl_gain_over_october_abrupt": "gain_bpb",
            "pl_gain_ci_low": "low_bpb",
            "pl_gain_ci_high": "high_bpb",
            "ramp_worse_in": "worse_in",
        }
    )
    table = table.astype({"pairs": int, "worse_in": int})  # counts, which the sub-table's shared columns hold as floats
    return table[["arm", "width", "pairs", "gain_bpb", "low_bpb", "high_bpb", "worse_in"]].sort_values(
        "width", ignore_index=True
    )


def draw_switch_smoothing(root: Path) -> FigureData:
    """Figure 26: the change of the paired gaps around an abrupt switch (A), and the gains of linear ramps over the
    abrupt switch (B)."""
    transition = read_table(root, TRANSITION_TABLE)
    values = switch_values(transition)
    values["ramp_gains"] = ramp_gains(transition, ramp_designs(root, ["arm", "ramp_width"]))
    switch, ramps = values["switch"].iloc[0], values["ramp_gains"]
    figure, axes = panels(2, legend_height=0.7)
    gap_axis, ramp_axis = axes
    handles = []
    for metric in METRIC_LABELS:
        rows, color = values["gaps"][values["gaps"]["metric"] == metric], METRIC_COLORS[metric]
        gap_axis.errorbar(
            rows["lag"], rows["gap_change_bpb"], yerr=rows["half_width_bpb"], color=color, **INTERVAL_STYLE
        )
        handles.append(Line2D([], [], color=color, marker="o", markersize=3.5, label=f"{METRIC_LABELS[metric]} BPB"))
    gap_axis.axvline(0.0, color=NEUTRAL, linestyle=":", linewidth=0.8, zorder=1)
    note = (
        f"relative to the gap at step {switch['baseline_step']:,.0f};"
        f" bars: {CONFIDENCE:.0%} paired-t over {int(switch['pairs'])} seeds"
    )
    panel_legend(gap_axis, [*handles, Line2D([], [], linestyle="none", label=note)])
    gap_axis.set_xlabel("Steps after the switch")
    gap_axis.set_ylabel("Change in the paired gap (BPB)")
    shares = f"{switch['p0']:g} to {switch['p1']:g}"
    gap_axis.set_title(f"A. StarCoder {shares} at step {switch['switch_step']:,.0f} ({switch['lr_state']})")
    positions = np.arange(len(ramps) + 1, dtype=float)
    gain = ramps["gain_bpb"].to_numpy(float) * GAIN_SCALE
    error = np.vstack(
        [gain - ramps["low_bpb"].to_numpy(float) * GAIN_SCALE, ramps["high_bpb"].to_numpy(float) * GAIN_SCALE - gain]
    )
    ramp_axis.plot([positions[0]], [0.0], "o", color="black", markerfacecolor="white", markersize=5, zorder=4)
    ramp_axis.errorbar(positions[1:], gain, yerr=error, color="black", linestyle="none", **INTERVAL_STYLE)
    ramp_axis.set_xticks(positions, ["abrupt", *(f"{width:,.0f}" for width in ramps["width"])])
    ramp_axis.set_xlabel("Ramp width (steps)")
    ramp_axis.set_ylabel("Gain over the abrupt switch (1e-3 BPB)")
    ramp_axis.set_title("B. Linear ramps in place of the switch")
    seeds = ", ".join(map(str, ramps["pairs"].unique()))
    panel_legend(
        ramp_axis,
        [
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="none",
                label=f"mean gain; bars: {CONFIDENCE:.0%} paired-t over {seeds} paired seeds",
            ),
            Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="none", label="abrupt switch"),
        ],
    )
    for axis in axes:
        axis.axhline(0.0, **ZERO_STYLE)
    return FigureData(figure, values)


def ramp_values(designs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The first data seed's schedules, stage by stage with the StarCoder share of each block, every run's StarCoder
    total and the part of it before the decay, and the runs' shared length, decay start and block length.

    Raises ``ValueError`` unless the runs share one length, decay start and block length, and no stage spans the
    decay start.
    """
    shared = unique_row(designs[["total_steps", "boundary_step", "steps_per_block"]], RAMP_TABLE)
    total, decay = int(shared["total_steps"]), int(shared["boundary_step"])
    stages = designs.sort_values(["data_seed", "arm", "stage"], ignore_index=True)
    following = stages.groupby(["data_seed", "arm"])["start_step"].shift(-1)
    stages["end_step"] = following.fillna(total).astype(int)
    if ((stages["start_step"] < decay) & (stages["end_step"] > decay)).any():
        raise ValueError(f"a stage of {RAMP_TABLE} spans the decay start at step {decay}")
    stages["before_decay"] = stages["starcoder_stage_sequences"].where(stages["end_step"] <= decay, 0)
    totals = (
        stages.groupby(["data_seed", "arm"], sort=False)
        .agg(
            ramp_width=("ramp_width", "first"),
            starcoder_sequences=("starcoder_stage_sequences", "sum"),
            before_decay=("before_decay", "sum"),
        )
        .reset_index()
    )
    seed = int(stages["data_seed"].min())
    schedules = stages[stages["data_seed"] == seed].assign(
        share=lambda rows: rows["starcoder_block_sequences"] / MIXTURE_BLOCK_SIZE
    )
    columns = ["arm", "ramp_width", "stage", "start_step", "end_step", "starcoder_block_sequences", "share"]
    training = {
        "data_seed": seed,
        "total_steps": total,
        "decay_step": decay,
        "steps_per_block": int(shared["steps_per_block"]),
    }
    return {
        "schedules": schedules[columns].reset_index(drop=True),
        "totals": totals,
        "training": pd.DataFrame([training]),
    }


def ramp_label(width: float) -> str:
    return "abrupt switch" if math.isnan(width) else f"{width:,.0f}-step ramp"


def draw_ramp_schedules(root: Path) -> FigureData:
    """Figure 27: the StarCoder share of each 2,048-sequence block, as the data loader allocated it, under the abrupt
    switch and the linear ramps, around the start of the learning-rate decay."""
    values = ramp_values(ramp_designs(root, RAMP_COLUMNS))
    schedules, training = values["schedules"], values["training"].iloc[0]
    seed, total = int(training["data_seed"]), int(training["total_steps"])
    totals = values["totals"][values["totals"]["data_seed"] == seed].set_index("arm")
    figure, axes = mosaic([["ramps", "ramps"]], columns=2, rows=1, legend_height=1.2)
    axis = axes["ramps"]
    arms = schedules.drop_duplicates("arm").sort_values("ramp_width", na_position="first")
    colors = ["black", *scale_colors(len(arms) - 1)]
    handles = []
    for (arm, width), color in zip(arms[["arm", "ramp_width"]].itertuples(index=False), colors, strict=True):
        rows = schedules[schedules["arm"] == arm].sort_values("stage")
        steps = np.append(rows["start_step"].to_numpy(), total)
        shares = np.append(rows["share"].to_numpy(), rows["share"].iloc[-1])
        axis.plot(steps, shares, drawstyle="steps-post", color=color, linewidth=1.1)
        run = totals.loc[arm]
        before, everything = int(run["before_decay"]), int(run["starcoder_sequences"])
        label = f"{ramp_label(width)}: {before:,} of its {everything:,} StarCoder sequences before the decay"
        handles.append(Line2D([], [], color=color, label=label))
    axis.axvline(training["decay_step"], color=NEUTRAL, linestyle="--", linewidth=0.8, zorder=1)
    handles.append(
        Line2D([], [], color=NEUTRAL, linestyle="--", linewidth=0.8, label="start of the learning-rate decay")
    )
    axis.set_xlim(RAMP_WINDOW[0] * total, RAMP_WINDOW[1] * total)
    axis.set_ylim(0.0, None)
    top = axis.secondary_xaxis("top", functions=(lambda step: 100 * step / total, lambda percent: percent * total / 100))
    top.set_xlabel("Share of training (%)")
    axis.set_xlabel("Training step")
    axis.set_ylabel("StarCoder share of the block")
    block = f"{MIXTURE_BLOCK_SIZE:,}-sequence block ({int(training['steps_per_block'])} steps)"
    axis.set_title(f"StarCoder per {block} as the data loader allocated it, data seed {seed}")
    figure_legend(figure, handles, columns=1)
    return FigureData(figure, values)
