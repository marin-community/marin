# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The paper's main-text table, ``tab:scaling-main``: each mixture's loss across the compute ladder and its accuracy.

Each of the table's 44 numbers is one query of the research code's number audit, a row labelled ``tab:scaling-main``
in its ``experiments/domain_phase_mix/release/paper_cells.csv``, so the printed table and the audited numbers share
one recipe; the README's "Research code" section gives the research code's location. A loss entry selects the
``ladder`` runs of one mixture at one compute scale: the proportional or UniMax-8 mixture (proposal objective
``none``), or Olmix's or MARINER's proposal for the column's objective. At 3e18 FLOPs the entry is the runs' mean and
standard deviation (ddof 1), over the eleven proportional runs or the three trainer seeds of the other mixtures; at
the larger scales it is the one run's value. The accuracy is the mean over the OlmoBaseEval Easy components of the
mixture's 1e21 run (``ladder_accuracy``), which trains Olmix's and MARINER's OlmoBaseEval Easy proposals.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from mariner.data import OBJECTIVES, OUTCOME_PREFIX, read_table

COMPUTE_SCALES = (3e18, 2e19, 3e20, 1e21)  # training FLOPs: the ladder's rungs, the table's loss columns
REPEATED_SCALE = 3e18  # the rung whose runs repeat, printed as mean and standard deviation
ACCURACY = "accuracy"
ACCURACY_SCALE = 1e21
UNPROPOSED = "none"  # the proposal objective of the mixtures that no surrogate proposed
BPB_DECIMALS = 3
ACCURACY_DECIMALS = 1
COLUMN_TITLES = {"uncheatable": "Uncheatable BPB", "obe_easy": "OlmoBaseEval Easy mean BPB", ACCURACY: "Accuracy"}
MIXTURE_TITLE = "Mixture"
COLUMN_GAP = "   "
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"


class Statistic(StrEnum):
    """How a table entry aggregates the values it selects, as the audit's queries name it."""

    MEAN = "mean"
    STD = "std"  # ddof 1
    VALUE = "value"  # the one selected value


class Mixture(NamedTuple):
    """A row of the table: its printed name, its ``method`` in the dataset, and whether a surrogate proposed it, so
    that each objective's columns read the proposal for that objective."""

    name: str
    method: str
    proposed: bool


MIXTURES = (
    Mixture("Proportional", "proportional", proposed=False),
    Mixture("UniMax-8", "unimax8", proposed=False),
    Mixture("Olmix", "olmix", proposed=True),
    Mixture("MARINER", "mariner", proposed=True),
)


def aggregate(values: pd.Series, statistic: Statistic, query: str) -> float:
    """``values`` aggregated by ``statistic``.

    Raises ``ValueError`` naming ``query`` when the selection is empty or holds a missing value, when it holds other
    than one value for ``VALUE``, or fewer than two for ``STD``.
    """
    if values.empty:
        raise ValueError(f"{query} selects no run")
    if values.isna().any():
        raise ValueError(f"{query} selects {int(values.isna().sum())} missing values")
    if statistic is Statistic.VALUE:
        if len(values) != 1:
            raise ValueError(f"{query} selects {len(values)} runs where it takes one run's value")
        return float(values.iloc[0])
    if statistic is Statistic.STD:
        if len(values) < 2:
            raise ValueError(f"{query} selects {len(values)} run, too few for a standard deviation")
        return float(values.std(ddof=1))
    return float(values.mean())


def scaling_main_table(root: Path) -> pd.DataFrame:
    """``tab:scaling-main`` from the snapshot at ``root``: one row per mixture in the paper's order.

    The columns are the paper's nine, keyed (quantity, compute FLOPs, statistic): each objective's loss at the four
    compute scales, its 3e18 entry a mean with its standard deviation, and the mean accuracy (%) at 1e21 FLOPs.
    """
    ladder = read_table(root, "ladder")
    accuracy = read_table(root, "ladder_accuracy")
    rows = {}
    for mixture in MIXTURES:
        cells: dict[tuple[str, float, Statistic], float] = {}
        for objective in OBJECTIVES:
            proposal = objective if mixture.proposed else UNPROPOSED
            runs = ladder[(ladder["method"] == mixture.method) & (ladder["proposal_objective"] == proposal)]
            for compute in COMPUTE_SCALES:
                values = runs.loc[runs["compute_flops"] == compute, f"{OUTCOME_PREFIX}{objective}"]
                statistics = (Statistic.MEAN, Statistic.STD) if compute == REPEATED_SCALE else (Statistic.VALUE,)
                for statistic in statistics:
                    query = f"{mixture.name}'s {objective} at {compute:g} FLOPs"
                    cells[(objective, compute, statistic)] = aggregate(values, statistic, query)
        scores = accuracy.loc[accuracy["method"] == mixture.method, ACCURACY]
        cells[(ACCURACY, ACCURACY_SCALE, Statistic.MEAN)] = aggregate(
            scores, Statistic.MEAN, f"{mixture.name}'s accuracy"
        )
        rows[mixture.name] = cells
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
    # Each title spans its quantity's columns; the last of them widens when the title is longer.
    for quantity, title in COLUMN_TITLES.items():
        indices = [index for index, (column, _) in enumerate(columns) if column == quantity]
        span = sum(widths[index] for index in indices) + len(COLUMN_GAP) * (len(indices) - 1)
        widths[indices[-1]] += max(len(title) - span, 0)
    titles = []
    for quantity, title in COLUMN_TITLES.items():
        indices = [index for index, (column, _) in enumerate(columns) if column == quantity]
        span = sum(widths[index] for index in indices) + len(COLUMN_GAP) * (len(indices) - 1)
        titles.append(title.ljust(span))
    name_width = max(len(name) for name in names)
    lines = [MIXTURE_TITLE.ljust(name_width) + COLUMN_GAP + COLUMN_GAP.join(titles)]
    for name, line in zip(names[1:], cells, strict=True):
        entries = [text.rjust(width) for text, width in zip(line, widths, strict=True)]
        lines.append(name.ljust(name_width) + COLUMN_GAP + COLUMN_GAP.join(entries))
    return "\n".join(line.rstrip() for line in lines)
