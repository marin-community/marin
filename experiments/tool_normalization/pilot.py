# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assign distinct diagnostic rows to six strata without counting overlaps twice."""

import random

import numpy as np
from scipy.optimize import linear_sum_assignment

STRATA = ("shell", "python", "files", "control_search", "terminal", "long_mixed")
INELIGIBLE_COST = 1_000_000


def balanced_pilot(rows: list[dict], per_stratum: int, seed: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (row["source"], row["shard"], row["row"]))
    random.Random(seed).shuffle(ordered)
    slots = [name for name in STRATA for _ in range(per_stratum)]
    costs = np.full((len(slots), len(ordered)), INELIGIBLE_COST, dtype=np.int64)
    for column, row in enumerate(ordered):
        eligible = set(row["families"])
        if row.get("tokens", 0) > 32768 or len(eligible) > 1:
            eligible.add("long_mixed")
        for index, stratum in enumerate(slots):
            if stratum in eligible:
                # Prefer already measured long rows over mixed rows of unknown length in this stratum.
                costs[index, column] = column + (
                    len(ordered) if stratum == "long_mixed" and row.get("tokens", 0) <= 32768 else 0
                )
    slot_indices, row_indices = linear_sum_assignment(costs)
    if len(slot_indices) != len(slots) or any(
        costs[i, j] >= INELIGIBLE_COST for i, j in zip(slot_indices, row_indices, strict=True)
    ):
        raise ValueError("Insufficient distinct diagnostic rows for all six strata")
    selected = [
        {**ordered[column], "pilot_stratum": slots[index]}
        for index, column in zip(slot_indices, row_indices, strict=True)
    ]
    covered = {tool for row in selected for tool in row["observed_tools"]}
    chosen = {(row["source"], row["shard"], row["row"]) for row in selected}
    for row in ordered:
        if (row["source"], row["shard"], row["row"]) not in chosen and set(row["observed_tools"]) - covered:
            selected.append({**row, "pilot_stratum": "rare_tool_supplement"})
            covered.update(row["observed_tools"])
    return selected
