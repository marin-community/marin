# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize immutable catalog review aggregates before serving a load."""

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
        query(
            """
                INSERT INTO atlas_cohort_summaries (load_id,pipeline_key,cohort,version,summary)
                SELECT load_id,pipeline_key,cohort,:version,
                    to_jsonb(v)-'load_id'-'pipeline_key'-'cohort'
                FROM atlas_cohort_summary v
                WHERE v.load_id=:load_id AND v.pipeline_key=:pipeline_key
                ON CONFLICT (load_id,pipeline_key,cohort) DO UPDATE
                    SET version=EXCLUDED.version,summary=EXCLUDED.summary
                    WHERE atlas_cohort_summaries.version<>EXCLUDED.version
                """,
            {**parameters, "pipeline_key": pipeline["pipeline_key"]},
        )
    return len(pipelines)
