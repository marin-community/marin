# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The paper's main-text table, ``tab:scaling-main``: each mixture's loss across the compute ladder and its accuracy.

Each of the table's 44 numbers is one query of the dataset. A loss entry selects the ``ladder`` runs of one mixture
at one compute scale: the proportional or UniMax-8 mixture (proposal objective ``none``), or Olmix's or MARINER's
proposal for the column's objective. At 3e18 FLOPs the entry is the runs' mean and standard deviation (ddof 1), over
the eleven proportional runs or the three trainer seeds of the other mixtures; at the larger scales it is the one
run's value. The accuracy is the mean over the OlmoBaseEval Easy components of the mixture's 1e21 run
(``ladder_accuracy``), which trains Olmix's and MARINER's OlmoBaseEval Easy proposals.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from mariner.data import (
    LADDER_ACCURACY_TABLE,
    LADDER_TABLE,
    OBJECTIVES,
    OUTCOME_PREFIX,
    PROPOSAL_OBJECTIVE,
    ObjectiveName,
    read_table,
    require_columns,
)
from mariner.methods import METHOD_LABELS, Method

COMPUTE_SCALES = (3e18, 2e19, 3e20, 1e21)  # training FLOPs: the ladder's rungs, the table's loss columns
REPEATED_SCALE = 3e18  # the rung whose runs repeat, printed as mean and standard deviation
ACCURACY = "accuracy"
ACCURACY_SCALE = 1e21
UNPROPOSED = "none"  # the proposal objective of the mixtures that no surrogate proposed
LADDER_COLUMNS = ["method", PROPOSAL_OBJECTIVE, "compute_flops"]  # what selects a mixture's runs at one scale
BPB_DECIMALS = 3
ACCURACY_DECIMALS = 1
COLUMN_TITLES = {
    ObjectiveName.UNCHEATABLE: "Uncheatable BPB",
    ObjectiveName.OBE_EASY: "OlmoBaseEval Easy mean BPB",
    ACCURACY: "Accuracy",
}
MIXTURE_TITLE = "Mixture"
COLUMN_GAP = "   "
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"


class Statistic(StrEnum):
    """How a table entry aggregates the values it selects."""

    MEAN = "mean"
    STD = "std"  # ddof 1
    VALUE = "value"  # the one selected value


class Mixture(NamedTuple):
    """A row of the table, printed under its method's name: the method, and whether a surrogate proposed the mixture,
    so that each objective's columns read the proposal for that objective."""

    method: Method
    proposed: bool


MIXTURES = (
    Mixture(Method.PROPORTIONAL, proposed=False),
    Mixture(Method.UNIMAX, proposed=False),
    Mixture(Method.OLMIX, proposed=True),
    Mixture(Method.MARINER, proposed=True),
)


def ladder_track(ladder: pd.DataFrame, method: Method, proposal: str, outcome: str) -> pd.DataFrame:
    """The mean and standard deviation (ddof 1) of the column ``outcome`` over one mixture's ``ladder`` runs at each
    compute scale, from the smallest, with the number of runs.

    The mixture is ``method``'s proposal for the objective ``proposal``, or ``UNPROPOSED`` for a mixture that no
    surrogate proposed. Raises ``ValueError`` unless the mixture has runs at each of ``COMPUTE_SCALES`` and at no other
    scale, and every run has a value of ``outcome``.
    """
    runs = ladder[(ladder["method"] == method) & (ladder[PROPOSAL_OBJECTIVE] == proposal)]
    missing = runs[outcome].isna()
    if missing.any():
        raise ValueError(f"the ladder lacks {outcome} for {int(missing.sum())} runs of {method}'s {proposal} mixture")
    track = runs.groupby("compute_flops")[outcome].agg(mean_bpb="mean", sd_bpb="std", runs="size").reset_index()
    if track["compute_flops"].tolist() != list(COMPUTE_SCALES):
        raise ValueError(
            f"the ladder trains {method}'s {proposal} mixture at {track['compute_flops'].tolist()} FLOPs, not at"
            f" {list(COMPUTE_SCALES)}"
        )
    return track


def loss_entries(track: pd.DataFrame, query: str) -> dict[tuple[float, Statistic], float]:
    """A row's loss entries from one mixture's ``ladder_track``, by compute scale and statistic: the mean and standard
    deviation at ``REPEATED_SCALE`` and the one run's value at each larger scale.

    Raises ``ValueError`` naming ``query`` when the repeated scale has fewer than two runs or a larger one other than
    one run.
    """
    entries = {}
    for row in track.itertuples(index=False):
        compute, runs = float(row.compute_flops), int(row.runs)
        if compute == REPEATED_SCALE:
            if runs < 2:
                raise ValueError(f"{query} at {compute:g} FLOPs selects {runs} run, too few for a standard deviation")
            entries[(compute, Statistic.MEAN)] = float(row.mean_bpb)
            entries[(compute, Statistic.STD)] = float(row.sd_bpb)
        elif runs != 1:
            raise ValueError(f"{query} at {compute:g} FLOPs selects {runs} runs where it takes one run's value")
        else:
            entries[(compute, Statistic.VALUE)] = float(row.mean_bpb)
    return entries


def mean_accuracy(scores: pd.Series, query: str) -> float:
    """The mean of ``scores``; raises ``ValueError`` naming ``query`` when they are empty or one is missing."""
    if scores.empty:
        raise ValueError(f"{query} selects no run")
    if scores.isna().any():
        raise ValueError(f"{query} selects {int(scores.isna().sum())} missing values")
    return float(scores.mean())


def scaling_main_table(root: Path) -> pd.DataFrame:
    """``tab:scaling-main`` from the snapshot at ``root``: one row per mixture in the paper's order.

    The columns are the paper's nine, keyed (quantity, compute FLOPs, statistic): each objective's loss at the four
    compute scales, its 3e18 entry a mean with its standard deviation, and the mean accuracy (%) at 1e21 FLOPs. Raises
    ``ValueError`` naming the table and column when ``ladder`` or ``ladder_accuracy`` lacks a column the queries read,
    and naming the entry when a mixture lacks its runs or their values (``ladder_track``, ``loss_entries``).
    """
    ladder = read_table(root, LADDER_TABLE)
    accuracy = read_table(root, LADDER_ACCURACY_TABLE)
    outcomes = [f"{OUTCOME_PREFIX}{objective}" for objective in OBJECTIVES]
    require_columns(ladder, [*LADDER_COLUMNS, *outcomes], LADDER_TABLE)
    require_columns(accuracy, ["method", ACCURACY], LADDER_ACCURACY_TABLE)
    rows = {}
    for mixture in MIXTURES:
        name = METHOD_LABELS[mixture.method]
        cells: dict[tuple[str, float, Statistic], float] = {}
        for objective in OBJECTIVES:
            proposal = objective if mixture.proposed else UNPROPOSED
            track = ladder_track(ladder, mixture.method, proposal, f"{OUTCOME_PREFIX}{objective}")
            for (compute, statistic), value in loss_entries(track, f"{name}'s {objective}").items():
                cells[(str(objective), compute, statistic)] = value
        scores = accuracy.loc[accuracy["method"] == mixture.method, ACCURACY]
        cells[(ACCURACY, ACCURACY_SCALE, Statistic.MEAN)] = mean_accuracy(scores, f"{name}'s accuracy")
        rows[name] = cells
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["quantity", "compute_flops", "statistic"])
    return table


def compute_label(flops: float) -> str:
    """Training FLOPs as the table's columns name them: 3e18, 2e19, 3e20, 1e21."""
    return f"{flops:.0e}".replace("e+", "e")


def entry_text(row: pd.Series, quantity: str, compute: float) -> str:
    """One entry of a row of ``scaling_main_table`` at the paper's printed decimals."""
    if quantity == ACCURACY:
        return f"{row[(quantity, compute, Statistic.MEAN)]:.{ACCURACY_DECIMALS}f}%"
    if compute == REPEATED_SCALE:
        mean, std = row[(quantity, compute, Statistic.MEAN)], row[(quantity, compute, Statistic.STD)]
        return f"{mean:.{BPB_DECIMALS}f}{PLUS_MINUS}{std:.{BPB_DECIMALS}f}"
    return f"{row[(quantity, compute, Statistic.VALUE)]:.{BPB_DECIMALS}f}"


def format_scaling_main(table: pd.DataFrame) -> str:
    """``scaling_main_table`` as plain text: a title over each quantity's columns, a row of compute scales, and one
    line per mixture, at the paper's printed decimals (three for BPB, one for accuracy)."""
    columns = [(objective, compute) for objective in OBJECTIVES for compute in COMPUTE_SCALES]
    columns.append((ACCURACY, ACCURACY_SCALE))
    names = [MIXTURE_TITLE, "", *table.index]
    cells = [
        [compute_label(compute) for _, compute in columns],
        *([entry_text(row, quantity, compute) for quantity, compute in columns] for _, row in table.iterrows()),
    ]
    widths = [max(len(line[index]) for line in cells) for index in range(len(columns))]
    titles = []
    for quantity, title in COLUMN_TITLES.items():
        indices = [index for index, (column, _) in enumerate(columns) if column == quantity]
        span = sum(widths[index] for index in indices) + len(COLUMN_GAP) * (len(indices) - 1)
        # The title spans its quantity's columns; the last of them widens when the title is longer.
        widths[indices[-1]] += max(len(title) - span, 0)
        titles.append(title.ljust(max(len(title), span)))
    name_width = max(len(name) for name in names)
    lines = [MIXTURE_TITLE.ljust(name_width) + COLUMN_GAP + COLUMN_GAP.join(titles)]
    for name, line in zip(names[1:], cells, strict=True):
        entries = [text.rjust(width) for text, width in zip(line, widths, strict=True)]
        lines.append(name.ljust(name_width) + COLUMN_GAP + COLUMN_GAP.join(entries))
    return "\n".join(line.rstrip() for line in lines)
