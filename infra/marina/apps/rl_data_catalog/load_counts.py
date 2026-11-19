# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count large staged tables through bounded primary-key windows on resume."""

from typing import Any

from marina.client import AppletQuery

STAGED_COUNT_BATCH = 4096
LARGE_STAGE_TABLES = frozenset({"atlas_items", "atlas_packets"})


def staged_row_count(query: AppletQuery, table: str, load_id: str) -> int:
    """Return the exact staged total, including retained history, without a full scan statement."""
    parameters: dict[str, Any] = {"load_id": load_id, "batch": STAGED_COUNT_BATCH}
    if table not in LARGE_STAGE_TABLES:
        return query(f"SELECT COUNT(*) AS n FROM {table} WHERE load_id=:load_id", parameters)["rows"][0]["n"]
    total = 0
    cursor = None
    while True:
        after = "AND (pipeline_key,task_id) > (:pipeline_key,:task_id)" if cursor is not None else ""
        rows = query(
            "SELECT pipeline_key,task_id,COUNT(*) OVER () AS n FROM ("
            f"SELECT pipeline_key,task_id FROM {table} WHERE load_id=:load_id {after} "
            "ORDER BY pipeline_key,task_id LIMIT :batch) page "
            "ORDER BY pipeline_key DESC,task_id DESC LIMIT 1",
            parameters,
        )["rows"]
        if not rows:
            return total
        row = rows[0]
        count = int(row["n"])
        next_cursor = (row["pipeline_key"], row["task_id"])
        if not 0 < count <= STAGED_COUNT_BATCH or next_cursor == cursor:
            raise ValueError("Invalid staged count window")
        total += count
        if count < STAGED_COUNT_BATCH:
            return total
        cursor = next_cursor
        parameters.update(pipeline_key=cursor[0], task_id=cursor[1])
