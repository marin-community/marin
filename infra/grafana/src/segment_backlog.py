# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alert projection for per-table Finelog segment counts.

A table whose compaction falls behind its flushes accumulates small L0 segments.
Each L0 spans the whole key range, so every read opens all of them and queries
start missing their deadlines.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# Interim: ListNamespaces reports only the total across levels, and the largest
# healthy tables hold about 3,000 segments in L1-L3. Lower this when the alert
# can compare an L0 count.
SEGMENT_BACKLOG_THRESHOLD = 10_000
HEALTHY_SEGMENT_STATE = "healthy"
BACKLOGGED_SEGMENT_STATE = "backlogged"
UNAVAILABLE_SEGMENT_STATE = "unavailable"


@dataclass(frozen=True)
class TableStorageStats:
    """One table's storage statistics from ListNamespaces."""

    namespace: str
    segment_count: int
    row_count: int
    byte_size: int


def segment_backlog_alert_rows(cluster: str, tables: Sequence[TableStorageStats]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table in sorted(tables, key=lambda table: table.namespace):
        backlogged = table.segment_count >= SEGMENT_BACKLOG_THRESHOLD
        rows.append(
            {
                "cluster": cluster,
                "namespace": table.namespace,
                "state": BACKLOGGED_SEGMENT_STATE if backlogged else HEALTHY_SEGMENT_STATE,
                "value": 1 if backlogged else 0,
            }
        )
    return rows


def segment_backlog_unavailable_rows(cluster: str) -> list[dict[str, object]]:
    """Return the non-firing row for a hub that cannot list its tables."""
    return [{"cluster": cluster, "namespace": "", "state": UNAVAILABLE_SEGMENT_STATE, "value": 0}]
