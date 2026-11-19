# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize immutable catalog review aggregates before serving a load."""

import json
from collections.abc import Callable
from typing import Any

from .cohort_sql import COHORT_PROJECTION_SQL, cohort_counts_sql

COHORT_SUMMARY_VERSION = 1
COHORT_ITEM_BATCH = 4096
COUNTER_FIELDS = (
    "reviewed",
    "missing_evidence",
    "packets_with_omissions",
    "item_count",
    "good_count",
    "conclusive_count",
)
COUNT_MAP_FIELDS = ("verdict_counts", "defect_counts", "confidence_counts")
SummaryQuery = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


def materialize_cohort_summaries(query: SummaryQuery, load_id: str, *, pipeline_key: str | None = None) -> int:
    """Store the canonical view's results once per pipeline, within gateway limits."""
    parameters = {"load_id": load_id, "version": COHORT_SUMMARY_VERSION, "selected_pipeline_key": pipeline_key}
    pipelines = query(
        """
        SELECT DISTINCT p.pipeline_key FROM atlas_populations p
        WHERE p.load_id=:load_id
            AND (CAST(:selected_pipeline_key AS TEXT) IS NULL OR p.pipeline_key=:selected_pipeline_key)
            AND NOT EXISTS (
            SELECT 1 FROM atlas_cohort_summaries s
            WHERE s.load_id=p.load_id AND s.pipeline_key=p.pipeline_key
                AND s.cohort=p.cohort AND s.version=:version
        ) ORDER BY p.pipeline_key
        """,
        parameters,
    )
    for pipeline in pipelines:
        summaries = bounded_summaries(query, load_id, pipeline["pipeline_key"])
        query(
            """
                INSERT INTO atlas_cohort_summaries (load_id,pipeline_key,cohort,version,summary)
                SELECT :load_id,:pipeline_key,cohort,:version,summary
                FROM jsonb_to_recordset(CAST(:summaries AS JSONB)) AS r(cohort TEXT,summary JSONB)
                ON CONFLICT (load_id,pipeline_key,cohort) DO UPDATE
                    SET version=EXCLUDED.version,summary=EXCLUDED.summary
                    WHERE atlas_cohort_summaries.version<>EXCLUDED.version
                """,
            {
                **parameters,
                "pipeline_key": pipeline["pipeline_key"],
                "summaries": json.dumps(
                    [{"cohort": cohort, "summary": summary} for cohort, summary in sorted(summaries.items())],
                    allow_nan=False,
                ),
            },
        )
    return len(pipelines)


def bounded_summaries(query: SummaryQuery, load_id: str, pipeline_key: str) -> dict[str, dict]:
    """Return summary records by cohort using bounded task-ID windows and canonical grading."""
    totals: dict[str, dict] = {}
    lower = None
    parameters = {"load_id": load_id, "pipeline_key": pipeline_key, "batch": COHORT_ITEM_BATCH}
    while True:
        after = "AND task_id>:lower" if lower is not None else ""
        boundary = query(
            "SELECT MAX(task_id) AS upper FROM (SELECT task_id FROM atlas_items "
            f"WHERE load_id=:load_id AND pipeline_key=:pipeline_key {after} "
            "ORDER BY task_id LIMIT :batch) page",
            {**parameters, "lower": lower},
        )[0]["upper"]
        if boundary is None and totals:
            break
        window = "AND FALSE" if boundary is None else "AND i.task_id<=:upper"
        if lower is not None:
            window += " AND i.task_id>:lower"
        rows = query(
            cohort_counts_sql(
                population_filter="WHERE p.load_id=:load_id AND p.pipeline_key=:pipeline_key",
                item_filter=window,
            ),
            {**parameters, "lower": lower, "upper": boundary},
        )
        for row in rows:
            cohort = row["cohort"]
            if cohort not in totals:
                totals[cohort] = dict(row)
                continue
            total = totals[cohort]
            if any(total[key] != row[key] for key in ("population", "sample_count")):
                raise ValueError("Cohort population changed during summary aggregation")
            for field in COUNTER_FIELDS:
                total[field] += row[field]
            for field in COUNT_MAP_FIELDS:
                for key, value in row[field].items():
                    total[field][key] += value
        if boundary is None:
            break
        lower = boundary
    # Aggregate rows per cohort replace repeated scans of the full item table.
    rows = query(
        "WITH counts AS (SELECT * FROM jsonb_to_recordset(CAST(:counts AS JSONB)) AS r("
        "load_id TEXT,pipeline_key TEXT,cohort TEXT,population BIGINT,sample_count BIGINT,"
        "reviewed BIGINT,missing_evidence BIGINT,packets_with_omissions BIGINT,"
        "verdict_counts JSONB,defect_counts JSONB,confidence_counts JSONB,"
        "item_count BIGINT,good_count BIGINT,conclusive_count BIGINT)) " + COHORT_PROJECTION_SQL,
        {"counts": json.dumps(list(totals.values()), allow_nan=False)},
    )
    return {
        row["cohort"]: {key: value for key, value in row.items() if key not in ("load_id", "pipeline_key", "cohort")}
        for row in rows
    }
