# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact schema for downstream-scaling statistics."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any, Required, TypedDict, TypeGuard

import fsspec

STATISTICS_FILENAME = "statistics.jsonl.gz"


class StatisticRow(TypedDict, total=False):
    id: Required[str]
    values: Required[list[Any]]
    metadata: dict[str, Any]


def statistics_file(output_path: str) -> str:
    return os.path.join(output_path, STATISTICS_FILENAME)


def is_statistic_row(row: Any) -> TypeGuard[StatisticRow]:
    return isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("values"), list)


def read_statistic_rows(path: str) -> Iterator[StatisticRow]:
    seen_ids: set[str] = set()
    with fsspec.open(path, "rt", compression="gzip") as f:
        for line in f:
            row = json.loads(line)
            if not is_statistic_row(row):
                raise TypeError(f"Invalid StatisticRow: {path}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate StatisticRow id {row['id']!r}: {path}")
            seen_ids.add(row["id"])
            yield row
