# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize immutable catalog review aggregates before serving a load."""

import json
from collections.abc import Callable
from typing import Any

COHORT_SUMMARY_VERSION = 1
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
        summaries = {}
        cohorts = None
        projections = (
            "population,sample_count,reviewed",
            "missing_evidence,packets_with_omissions",
            "verdict_counts",
            "grade,good_fraction,result_reviewed,issue_count",
            "grade_reason,issue_rate,uncertain_count,unavailable_count",
            "defect_counts",
            "confidence_counts",
        )
        for fields in projections:
            rows = query(
                f"SELECT cohort,{fields} FROM atlas_cohort_summary "
                "WHERE load_id=:load_id AND pipeline_key=:pipeline_key ORDER BY cohort",
                {**parameters, "pipeline_key": pipeline["pipeline_key"]},
            )
            current = {row["cohort"] for row in rows}
            if not current or (cohorts is not None and current != cohorts):
                raise ValueError("Cohort populations changed during canonical summary projection")
            cohorts = current
            for row in rows:
                summaries.setdefault(row["cohort"], {}).update(
                    {name: value for name, value in row.items() if name != "cohort"}
                )
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
