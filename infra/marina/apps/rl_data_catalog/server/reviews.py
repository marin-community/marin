# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog-wide counts and definitions for typed content-review findings."""

import json
from collections import defaultdict
from itertools import batched
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .source_identity import canonical_source_id

QC_INVENTORY_BATCH = 16

DEFECT_DESCRIPTIONS = {
    "missing_context": "Information needed to solve or judge the task is absent from the supplied evidence.",
    "ambiguity": "The task or its grading requirements allow materially different interpretations.",
    "wrong_reference": "The recorded reference answer or expected result conflicts with the task.",
    "answer_leakage": "The solver-visible task reveals information that gives away the expected answer.",
    "malformed": "The task, data or grading contract is structurally broken or unusable.",
    "rubric_mismatch": "The grading rubric does not assess what the task asks the solver to deliver.",
}


def register_review_routes(api: FastAPI, engine: Engine) -> None:
    @api.get("/qc/inventory")
    def inventory() -> dict[str, Any]:
        with engine.connect() as connection:
            ids = (
                connection.execute(
                    text(
                        "SELECT s.id FROM atlas_sources s JOIN atlas_current c USING(load_id) "
                        "WHERE c.name='latest' ORDER BY s.id"
                    )
                )
                .scalars()
                .all()
            )
            panels = []
            for source_ids in batched(ids, QC_INVENTORY_BATCH):
                # Correlated counts use each pipeline's composite indexes instead of
                # scanning retained review history for every source batch.
                panels.extend(
                    connection.execute(
                        text(
                            """
                    WITH current_sources AS (
                        SELECT s.load_id,s.id FROM atlas_sources s JOIN atlas_current c USING(load_id)
                        WHERE c.name='latest'
                          AND s.id IN (SELECT value FROM jsonb_array_elements_text(CAST(:source_ids AS JSONB)))
                    )
                    SELECT s.id AS source_id,r.pipeline_key,a.scope,r.receipt,
                        COALESCE(p.task_count,0) AS task_count,raw.raw_count,
                        links.linked_count,links.missing_batches,chunks.missing_chunks
                    FROM current_sources s
                    LEFT JOIN atlas_review_current r ON r.load_id=s.load_id AND r.source_id=s.id
                    LEFT JOIN atlas_assessments a ON a.load_id=r.load_id AND a.pipeline_key=r.pipeline_key
                    LEFT JOIN LATERAL (
                        SELECT SUM(p.sample_count)::BIGINT AS task_count FROM atlas_populations p
                        WHERE p.load_id=r.load_id AND p.pipeline_key=r.pipeline_key
                    ) p ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT COUNT(*) AS raw_count FROM atlas_raw_reviews raw
                        WHERE raw.load_id=r.load_id AND raw.pipeline_key=r.pipeline_key
                    ) raw ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT COUNT(*) AS linked_count,
                            COUNT(*) FILTER (WHERE batches.batch_key IS NULL) AS missing_batches
                        FROM atlas_review_batch_tasks links
                        LEFT JOIN atlas_review_batches batches ON batches.load_id=links.load_id
                            AND batches.pipeline_key=links.pipeline_key AND batches.batch_key=links.batch_key
                        WHERE links.load_id=r.load_id AND links.pipeline_key=r.pipeline_key
                    ) links ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT COUNT(*) AS missing_chunks FROM atlas_review_batches batches
                        WHERE batches.load_id=r.load_id AND batches.pipeline_key=r.pipeline_key
                          AND batches.chunk_count<>(SELECT COUNT(*) FROM atlas_review_batch_chunks chunks
                              WHERE chunks.load_id=batches.load_id AND chunks.pipeline_key=batches.pipeline_key
                                AND chunks.batch_key=batches.batch_key)
                    ) chunks ON TRUE
                    ORDER BY s.id
                """
                        ),
                        {"source_ids": json.dumps(source_ids)},
                    )
                    .mappings()
                    .all()
                )
            rollouts = (
                connection.execute(
                    text(
                        """
                SELECT r.measurement_id,d.source_id,
                    CAST(r.metadata->>'attempt_count' AS BIGINT) AS expected_count,
                    (SELECT COUNT(*) FROM atlas_rollout_attempts a
                        WHERE a.measurement_id=r.measurement_id) AS attempt_count
                FROM atlas_rollout_sets r JOIN atlas_difficulty d USING(measurement_id)
                JOIN atlas_sources s ON s.id=d.source_id
                JOIN atlas_current selected ON selected.load_id=s.load_id AND selected.name='latest'
                ORDER BY r.measurement_id
            """
                    )
                )
                .mappings()
                .all()
            )
        return {"panels": [dict(row) for row in panels], "rollouts": [dict(row) for row in rollouts]}

    @api.get("/qc/defects")
    def defects() -> dict[str, Any]:
        counts: dict[str, dict[str, Any]] = {
            key: {"id": key, "description": description, "count": 0, "sources": []}
            for key, description in DEFECT_DESCRIPTIONS.items()
        }
        sources: dict[str, dict[str, Any]] = {}
        by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                SELECT s.id,s.declaration->>'display_name' AS name,c.summary->'defect_counts' AS counts
                FROM atlas_sources s JOIN atlas_current selected USING (load_id)
                LEFT JOIN atlas_review_current r ON r.load_id=s.load_id AND r.source_id=s.id
                JOIN atlas_cohort_summaries c ON c.load_id=s.load_id
                    AND c.pipeline_key=COALESCE(r.pipeline_key,s.assessment_key)
                WHERE selected.name='latest'
                ORDER BY s.id,c.cohort
                """
                )
            ).mappings()
            for row in rows:
                source_id = canonical_source_id(row["id"])
                sources[source_id] = {"id": source_id, "name": row["name"]}
                for defect, count in (row["counts"] or {}).items():
                    by_source[source_id][defect] += count
        for source_id, source_counts in by_source.items():
            for defect, count in source_counts.items():
                if count:
                    counts[defect]["count"] += count
                    counts[defect]["sources"].append({**sources[source_id], "count": count})
        return {"sources": len(sources), "defects": list(counts.values())}
