# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assign grouped development and holdout partitions to recorded source samples."""

import random
from typing import Any

HOLDOUT_FRACTION = 0.3


def assign_partitions(rows: list[dict[str, Any]], randomizer: random.Random, sort_column: str) -> None:
    """Keep each group in one partition and sort development rows before holdout rows."""
    groups = sorted({row["sample_group"] for row in rows})
    randomizer.shuffle(groups)
    holdout = set(groups[: round(len(groups) * HOLDOUT_FRACTION)])
    for row in rows:
        row["sample_partition"] = "holdout" if row["sample_group"] in holdout else "development"
    rows.sort(key=lambda row: (row["sample_partition"] == "holdout", row[sort_column]))
