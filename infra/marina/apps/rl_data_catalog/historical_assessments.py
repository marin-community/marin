# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep archived assessment evidence across catalog refreshes and source retirement."""

from collections.abc import Mapping
from types import MappingProxyType

from marina.client import AppletQuery

# Copy evidence before its assessment header; each query selects missing headers.
ASSESSMENT_TABLES = ("populations", "items", "packets", "assessments")
# Copy only a small page of missing task IDs at a time. The primary key supports
# these lookups without scanning a whole pipeline once per hash bucket.
RETAIN_COPY_BATCH_ROWS = MappingProxyType({"items": 32, "packets": 8})


def retain_prior_assessments(
    query: AppletQuery, columns: Mapping[str, str], prior_load_id: str, load_id: str
) -> dict[str, int]:
    """Preserve prior assessment units absent from the target's assessment headers.

    Existing task rows and complete target assessment headers are left unchanged.
    Copy headers last, after every missing item and packet has been staged.

    Returns:
        Newly inserted row counts keyed by assessment table name.
    """
    copied = {}
    for table in ASSESSMENT_TABLES:
        names = [column.strip().split()[0] for column in columns[table].split(",")]
        # The prior load's materialized cohort summaries carry its assessed pipeline
        # set in a small indexed table; probing the per-task tables directly scans
        # the whole retained item volume and exceeds the gateway statement timeout.
        missing = query(
            "SELECT pipeline_key FROM ("
            "SELECT DISTINCT pipeline_key FROM atlas_cohort_summaries WHERE load_id = :prior_load_id"
            ") candidates WHERE NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
            "WHERE current_assessment.load_id = :load_id "
            "AND current_assessment.pipeline_key = candidates.pipeline_key)",
            {"load_id": load_id, "prior_load_id": prior_load_id},
        )["rows"]
        total = 0
        for row in missing:
            parameters = {
                "load_id": load_id,
                "prior_load_id": prior_load_id,
                "pipeline_key": row["pipeline_key"],
            }
            if table in RETAIN_COPY_BATCH_ROWS:
                after_task_id = None
                while True:
                    cursor_clause = "AND previous.task_id > :after_task_id " if after_task_id is not None else ""
                    page = query(
                        f"SELECT previous.task_id FROM atlas_{table} previous "
                        "WHERE previous.load_id = :prior_load_id AND previous.pipeline_key = :pipeline_key "
                        + cursor_clause
                        + f"AND NOT EXISTS (SELECT 1 FROM atlas_{table} target "
                        "WHERE target.load_id = :load_id AND target.pipeline_key = previous.pipeline_key "
                        "AND target.task_id = previous.task_id) "
                        "ORDER BY previous.task_id LIMIT :batch_rows",
                        {**parameters, "after_task_id": after_task_id, "batch_rows": RETAIN_COPY_BATCH_ROWS[table]},
                    )["rows"]
                    if not page:
                        break
                    identities = {f"task_id_{index}": item["task_id"] for index, item in enumerate(page)}
                    placeholders = ", ".join(":" + name for name in identities)
                    result = query(
                        f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
                        f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
                        f"FROM atlas_{table} previous WHERE previous.load_id = :prior_load_id "
                        "AND previous.pipeline_key = :pipeline_key "
                        f"AND previous.task_id IN ({placeholders}) "
                        "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
                        "WHERE current_assessment.load_id = :load_id "
                        "AND current_assessment.pipeline_key = previous.pipeline_key) "
                        "ON CONFLICT DO NOTHING",
                        {**parameters, **identities},
                    )
                    total += result["row_count"]
                    after_task_id = page[-1]["task_id"]
            else:
                result = query(
                    f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
                    f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
                    f"FROM atlas_{table} previous WHERE previous.load_id = :prior_load_id "
                    "AND previous.pipeline_key = :pipeline_key "
                    "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
                    "WHERE current_assessment.load_id = :load_id "
                    "AND current_assessment.pipeline_key = previous.pipeline_key) "
                    "ON CONFLICT DO NOTHING",
                    parameters,
                )
                total += result["row_count"]
        copied[table] = total
    return copied
