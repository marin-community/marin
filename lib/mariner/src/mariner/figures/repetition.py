# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's downsampling figures (Appendix B.3): Figures 21 (``fig:b5a``) and 22
(``fig:a-downsampling``).

The atlas's dense replay sweeps vary the StarCoder share p of a 210M-parameter model's Nemotron-CC/StarCoder mixture
at four token budgets D, with StarCoder drawn from its full pool or from one of six subsets. Each subset is sized to
repeat StarCoder 1/8 to 4 times as often as the target run does at the same share, its repetition relative to the
target's. The two-phase sweeps of ``starcoder_curricula`` train the same 28 pairs of budget and pool and name each
pool's repetition (``epoch_multiplier``), so a sweep takes its pool from the ``starcoder_curricula`` row with its
token budget and pool size. Figure 21 groups the 28 sweeps by budget and Figure 22 by repetition. Both star each
sweep's observed grid minimum, and Figure 22 gives StarCoder's epochs there.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from mariner.data import read_table, require_columns
from mariner.figures.epochs import shared_epochs_at_p1
from mariner.figures.layout import (
    NEUTRAL,
    FigureData,
    figure_legend,
    mosaic,
    panel_legend,
    panels,
    scale_colors,
)
from mariner.figures.starcoder import (
    CURRICULA_TABLE,
    CURVES_TABLE,
    Optimum,
    StarcoderSweep,
    atlas_sweeps,
    tokens_label,
)

REPLAY_FAMILY = "dense_horizon_replay"  # starcoder_curves' family of the 28 sweeps
FULL_POOL = "full"  # starcoder_curricula's support_id of the whole StarCoder pool, whose multiplier is NaN
# A sweep's pool is the support whose budget and pool size (SUPPORT_KEYS) equal the sweep's (CURVE_KEYS).
CURVE_KEYS = ["train_tokens", "starcoder_subset_tokens"]
SUPPORT_KEYS = ["materialized_tokens", "starcoder_realized_support_tokens"]
SUPPORT_COLUMNS = ["support_id", "support_order", "epoch_multiplier"]
FULL_POOL_COLOR = "#8A9099"
MINIMUM_STYLE = {"marker": "*", "markersize": 9, "markeredgecolor": NEUTRAL, "markeredgewidth": 0.6, "linestyle": "none"}
PANELS_PER_ROW = 3  # Figure 22's six panels sit in two rows


@dataclasses.dataclass(frozen=True)
class ReplaySweep:
    """A dense replay sweep, its StarCoder pool, and the pool's repetition relative to the target run's.

    ``support_order`` ranks the pools by repetition, the full pool first; ``epoch_multiplier`` is NaN for the full
    pool.
    """

    sweep: StarcoderSweep
    support_id: str
    support_order: int
    epoch_multiplier: float

    @property
    def full_pool(self) -> bool:
        return self.support_id == FULL_POOL

    @property
    def minimum(self) -> Optimum:
        """The swept share with the least measured loss, and that loss: the observed grid minimum."""
        best = int(np.argmin(self.sweep.bpb))
        return Optimum(float(self.sweep.shares[best]), float(self.sweep.bpb[best]))


def replay_sweeps(root: Path) -> list[ReplaySweep]:
    """The dense replay sweeps with their StarCoder pools, by token budget and then by repetition.

    Raises ``ValueError`` unless each sweep's token budget and pool size match exactly one ``starcoder_curricula``
    pool.
    """
    curves = read_table(root, CURVES_TABLE)
    require_columns(curves, ["curve_ref", "family", *CURVE_KEYS], CURVES_TABLE)
    supports = read_table(root, CURRICULA_TABLE)
    require_columns(supports, [*SUPPORT_KEYS, *SUPPORT_COLUMNS], CURRICULA_TABLE)
    replay = curves.loc[curves["family"] == REPLAY_FAMILY, ["curve_ref", *CURVE_KEYS]]
    pools = supports[[*SUPPORT_KEYS, *SUPPORT_COLUMNS]].drop_duplicates()
    matched = replay.merge(pools, how="left", left_on=CURVE_KEYS, right_on=SUPPORT_KEYS)
    matches = matched.groupby("curve_ref")["support_id"].count()
    unmatched = matches[matches != 1]
    if len(unmatched):
        raise ValueError(
            f"the {REPLAY_FAMILY} curves {', '.join(unmatched.index)} match"
            f" {', '.join(map(str, unmatched))} {CURRICULA_TABLE} pools by token budget and pool size, not one each"
        )
    matched = matched.sort_values(["train_tokens", "support_order"])
    atlas = atlas_sweeps(root, matched["curve_ref"])
    return [
        ReplaySweep(atlas[str(row.curve_ref)], str(row.support_id), int(row.support_order), float(row.epoch_multiplier))
        for row in matched.itertuples()
    ]


def repetition_label(multiplier: float) -> str:
    """A StarCoder pool's repetition relative to the target run's, such as 1/8x or 4x (with a multiplication sign), or
    ``full pool``."""
    if math.isnan(multiplier):
        return "full pool"
    return f"{Fraction(multiplier).limit_denominator()}\N{MULTIPLICATION SIGN}"


def pool_label(multiplier: float) -> str:
    """A StarCoder pool as a legend names it: the full pool, or a subset by its repetition."""
    return (
        repetition_label(multiplier) if math.isnan(multiplier) else f"{repetition_label(multiplier)} target repetition"
    )


def pool_colors(support_ids: Sequence[str]) -> dict[str, tuple[float, ...] | str]:
    """One color per StarCoder pool of ``support_ids``, which run from the least repeated pool to the most: gray for
    the full pool and a scale ladder for the subsets."""
    subsets = [support for support in dict.fromkeys(support_ids) if support != FULL_POOL]
    return {FULL_POOL: FULL_POOL_COLOR, **dict(zip(subsets, scale_colors(len(subsets)), strict=True))}


def replay_values(sweeps: Sequence[ReplaySweep]) -> dict[str, pd.DataFrame]:
    """The sweeps' runs and their grid minima, with StarCoder's epochs at each minimum."""
    curves, minima = [], []
    for replay in sweeps:
        sweep = replay.sweep
        keys = {
            "curve": sweep.name,
            "train_tokens": sweep.train_tokens,
            "support_id": replay.support_id,
            "support_order": replay.support_order,
            "epoch_multiplier": replay.epoch_multiplier,
        }
        curves += [{**keys, "share": share, "bpb": bpb} for share, bpb in zip(sweep.shares, sweep.bpb, strict=True)]
        minimum = replay.minimum
        minima.append(
            {
                **keys,
                "share": minimum.share,
                "bpb": minimum.bpb,
                "starcoder_epochs_at_p1": sweep.starcoder_epochs_at_p1,
                "epochs": minimum.share * sweep.starcoder_epochs_at_p1,
            }
        )
    return {"curves": pd.DataFrame(curves), "minima": pd.DataFrame(minima)}


def draw_sweep(axis: Axes, replay: ReplaySweep, color: tuple[float, ...] | str) -> None:
    """A sweep's measured loss over the StarCoder share, dashed with open markers for the full pool, which lies above
    the subsets' curves where they meet, and its grid minimum as a star."""
    sweep = replay.sweep
    style = {"linestyle": "--", "markerfacecolor": "white", "zorder": 3} if replay.full_pool else {}
    axis.plot(sweep.shares, sweep.bpb, color=color, marker="o", markersize=2.5, linewidth=1.0, **style)
    minimum = replay.minimum
    axis.plot([minimum.share], [minimum.bpb], color=color, zorder=5, **MINIMUM_STYLE)


def draw_downsampling_by_budget(root: Path) -> FigureData:
    """Figure 21: the replay sweeps of each token budget, one curve per StarCoder pool, with their grid minima."""
    sweeps = replay_sweeps(root)
    budgets = sorted({replay.sweep.train_tokens for replay in sweeps})
    pools = {replay.support_id: replay for replay in sorted(sweeps, key=lambda replay: replay.support_order)}
    colors = pool_colors(list(pools))
    figure, axes = panels(len(budgets), legend_height=0.6, sharey=True)
    for axis, budget in zip(axes, budgets, strict=True):
        for replay in sweeps:
            if replay.sweep.train_tokens == budget:
                draw_sweep(axis, replay, colors[replay.support_id])
        axis.set_title(f"D = {tokens_label(budget)} tokens")
        axis.set_xlabel("StarCoder share of the mixture, p")
    axes[0].set_ylabel("Programming-languages loss (BPB)")
    handles = [
        Line2D(
            [],
            [],
            color=colors[support],
            marker="o",
            markersize=2.5,
            linestyle="--" if replay.full_pool else "-",
            markerfacecolor="white" if replay.full_pool else colors[support],
            label=pool_label(replay.epoch_multiplier),
        )
        for support, replay in pools.items()
    ]
    handles.append(Line2D([], [], color=NEUTRAL, label="grid minimum", **MINIMUM_STYLE))
    figure_legend(figure, handles, columns=4)
    return FigureData(figure, replay_values(sweeps))


def draw_downsampling_by_repetition(root: Path) -> FigureData:
    """Figure 22: the replay sweeps of each StarCoder subset across token budgets, with their grid minima and the
    StarCoder epochs there; top axes give StarCoder's epochs at each share."""
    sweeps = [replay for replay in replay_sweeps(root) if not replay.full_pool]
    subsets = sorted({(replay.support_order, replay.support_id) for replay in sweeps})
    support_ids = [support for _, support in subsets]
    layout = [support_ids[start : start + PANELS_PER_ROW] for start in range(0, len(support_ids), PANELS_PER_ROW)]
    figure, axes = mosaic(layout, columns=PANELS_PER_ROW, rows=len(layout), legend_height=2.0)
    budgets = sorted({replay.sweep.train_tokens for replay in sweeps})
    colors = dict(zip(budgets, scale_colors(len(budgets)), strict=True))
    for support in support_ids:
        axis = axes[support]
        group = [replay for replay in sweeps if replay.support_id == support]
        epochs_at_p1 = shared_epochs_at_p1([replay.sweep for replay in group])
        handles = []
        for replay in group:
            color = colors[replay.sweep.train_tokens]
            draw_sweep(axis, replay, color)
            epochs = replay.minimum.share * epochs_at_p1
            label = f"D = {tokens_label(replay.sweep.train_tokens)}: minimum at {epochs:.1f} StarCoder epochs"
            handles.append(Line2D([], [], color=color, marker="o", markersize=2.5, label=label))
        top = axis.secondary_xaxis("top", functions=(lambda p, s=epochs_at_p1: p * s, lambda e, s=epochs_at_p1: e / s))
        top.set_xlabel("StarCoder epochs")
        axis.set_title(pool_label(group[0].epoch_multiplier))
        axis.set_xlabel("StarCoder share of the mixture, p")
        axis.set_ylabel("Programming-languages loss (BPB)")
        panel_legend(axis, handles, fontsize=6.5)
    figure_legend(
        figure,
        [
            Line2D([], [], color=NEUTRAL, marker="o", markersize=2.5, label="measured loss"),
            Line2D([], [], color=NEUTRAL, label="grid minimum", **MINIMUM_STYLE),
        ],
        columns=2,
    )
    return FigureData(figure, replay_values(sweeps))
