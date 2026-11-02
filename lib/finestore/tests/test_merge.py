# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The sorted-run merge against a dictionary of the newest version per key."""

import itertools
import random

import pyarrow as pa
import pytest
from finestore.layout import SystemColumns
from finestore.merge import merge_sorted_runs

PRIMARY_KEY = ("group", "item")
SCHEMA = pa.schema(
    [
        ("group", pa.string()),
        ("item", pa.int64()),
        ("value", pa.int64()),
        (SystemColumns.COMMIT, pa.int64()),
        (SystemColumns.SEQUENCE, pa.int64()),
        (SystemColumns.GENERATION, pa.int32()),
    ]
)


def _order(key: tuple) -> tuple:
    return tuple((value is None, value) for value in key)


def _random_runs(rng: random.Random) -> list[list[dict]]:
    """Sorted runs with null keys, repeated keys, and rows whose missing ``_seq`` counts as zero.

    Every row's version is distinct, so the newest row of each key is unambiguous.
    """
    runs = []
    row_id = 0
    for _ in range(rng.randint(1, 5)):
        rows = []
        for _ in range(rng.randint(0, 40)):
            row_id += 1
            legacy = rng.random() < 0.2
            rows.append(
                {
                    "group": rng.choice(["a", "b", None]),
                    "item": rng.choice([0, 1, 2, None]),
                    "value": row_id,
                    SystemColumns.COMMIT: row_id if legacy else rng.randint(0, 5),
                    SystemColumns.SEQUENCE: None if legacy else row_id,
                    SystemColumns.GENERATION: rng.randint(0, 2),
                }
            )
        rows.sort(key=lambda row: _order((row["group"], row["item"])))
        runs.append(rows)
    return runs


def _split(rows: list[dict], rng: random.Random) -> list[pa.Table]:
    """Cut a run into tables at random points, including inside a span of one key."""
    cuts = sorted(rng.choices(range(len(rows) + 1), k=rng.randint(0, 6)))
    bounds = [0, *cuts, len(rows)]
    return [pa.Table.from_pylist(rows[start:end], schema=SCHEMA) for start, end in itertools.pairwise(bounds)]


@pytest.mark.parametrize("seed", range(40))
def test_merge_yields_the_newest_row_per_key_once_in_key_order(seed):
    rng = random.Random(seed)
    runs = _random_runs(rng)
    newest: dict[tuple, tuple[tuple, int]] = {}
    for rows in runs:
        for row in rows:
            key = (row["group"], row["item"])
            version = (row[SystemColumns.COMMIT], row[SystemColumns.SEQUENCE] or 0, row[SystemColumns.GENERATION])
            if key not in newest or version > newest[key][0]:
                newest[key] = (version, row["value"])
    expected = [(key, value) for key, (_, value) in sorted(newest.items(), key=lambda item: _order(item[0]))]

    merged = merge_sorted_runs([_split(rows, rng) for rows in runs], PRIMARY_KEY)

    assert [((row["group"], row["item"]), row["value"]) for table in merged for row in table.to_pylist()] == expected
