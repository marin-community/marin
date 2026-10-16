# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the explicitly loaded curation catalog from Postgres."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .curriculum import register_curriculum_routes
from .schema import SCHEMA_SQL


def migrate(connection: Connection) -> None:
    for statement in SCHEMA_SQL.split(";"):
        if statement.strip():
            connection.execute(text(statement))
    connection.execute(text("DROP TABLE IF EXISTS curation_catalog"))
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS catalog_reviews (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
            collection JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL
        )
    """
        )
    )
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS review_artifacts (
            review_id TEXT NOT NULL, path TEXT NOT NULL, content TEXT NOT NULL,
            sha256 TEXT NOT NULL, PRIMARY KEY (review_id, path)
        )
    """
        )
    )


def loaded_catalog(connection: Connection) -> dict[str, Any]:
    row = (
        connection.execute(
            text(
                """
        SELECT l.* FROM atlas_current c JOIN atlas_loads l USING (load_id)
        WHERE c.name = 'latest'
    """
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(503, "Load a curation catalog before opening Atlas")
    return dict(row)


def source_record(connection: Connection, source_id: str) -> dict[str, Any]:
    row = (
        connection.execute(
            text(
                """
        SELECT s.* FROM atlas_sources s JOIN atlas_current c USING (load_id)
        WHERE c.name = 'latest' AND s.id = :source_id
    """
            ),
            {"source_id": source_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(404, "Source not found")
    return dict(row)


def cohort_summaries(connection: Connection, load_id: str, pipeline_key: str) -> dict[str, Any]:
    return dict(
        connection.execute(
            text(
                """
            SELECT cohort, (to_jsonb(s) - 'load_id' - 'pipeline_key' - 'cohort') ||
                jsonb_build_object('examples', (
                    SELECT COALESCE(jsonb_agg(to_jsonb(example)), '[]'::jsonb) FROM (
                        SELECT task_id, verdict, explanation, evidence_ids FROM atlas_items i
                        WHERE i.load_id = s.load_id AND i.pipeline_key = s.pipeline_key
                            AND i.cohort = s.cohort AND i.verdict <> 'good'
                        ORDER BY task_id LIMIT 5
                    ) example
                )) AS summary
        FROM atlas_cohort_summary s WHERE load_id = :load_id AND pipeline_key = :pipeline_key
    """
            ),
            {"load_id": load_id, "pipeline_key": pipeline_key},
        )
        .tuples()
        .all()
    )


def source_coverage(connection: Connection) -> dict[str, dict[str, Any]]:
    """Read source sizes and leading subjects from the selected compact release."""
    rows = connection.execute(
        text(
            """
        SELECT s.source_id, (s.alignment->>'task_count')::bigint AS example_count,
            s.alignment->>'intended_use' AS intended_use,
            area.subject_id, area.node_id, area.name, area.instances
        FROM curriculum_sources s JOIN curriculum_current c USING (load_id)
        LEFT JOIN LATERAL (
            SELECT n.subject_id, n.node_id, n.name, statistics.instances
            FROM curriculum_statistics statistics JOIN curriculum_nodes n
                USING (load_id, subject_id, node_id)
            WHERE statistics.load_id=s.load_id AND statistics.source_id=s.source_id
                AND statistics.metric='placement' AND n.kind='subject' AND statistics.instances>0
            ORDER BY statistics.instances DESC, n.subject_id LIMIT 1
        ) area ON TRUE
        WHERE c.name='latest' AND s.alignment IS NOT NULL
        """
        )
    ).mappings()
    return {
        row["source_id"]: {
            "example_count": row["example_count"],
            "intended_use": row["intended_use"],
            "top_area": (
                {key: row[key] for key in ("subject_id", "node_id", "name", "instances")}
                if row["subject_id"] is not None
                else None
            ),
        }
        for row in rows
    }


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()
    register_curriculum_routes(api, engine)

    @api.get("/sources")
    def sources() -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            coverage = source_coverage(connection)
            rows = connection.execute(
                text(
                    """
                SELECT s.*, a.scope, a.model, summary.cohorts,
                    COALESCE((s.curation->>'raw_population_count')::bigint,
                        (s.declaration->>'task_count')::bigint) AS task_count
                FROM atlas_sources s
                LEFT JOIN atlas_assessments a
                    ON a.load_id = s.load_id AND a.pipeline_key = s.assessment_key
                LEFT JOIN (
                    SELECT load_id, pipeline_key,
                        jsonb_object_agg(cohort, to_jsonb(v) - 'load_id' - 'pipeline_key' - 'cohort') AS cohorts
                    FROM atlas_cohort_summary v WHERE v.load_id = :load_id GROUP BY load_id, pipeline_key
                ) summary ON summary.load_id = s.load_id AND summary.pipeline_key = s.assessment_key
                WHERE s.load_id = :load_id ORDER BY s.id
            """
                ),
                {"load_id": metadata["load_id"]},
            ).mappings()
            inventory = []
            for row in rows:
                inventory.append(
                    {
                        **row["declaration"],
                        "task_count": row["task_count"],
                        "curation": row["curation"],
                        "coverage": coverage.get(row["id"]),
                        "assessment": {
                            "status": "complete" if row["assessment_key"] else row["latest_attempt_status"],
                            "pipeline_key": row["assessment_key"],
                            "latest_attempt_key": row["latest_attempt_key"],
                            "latest_attempt_status": row["latest_attempt_status"],
                            "reason": row["reason"],
                            "scope": row["scope"],
                            "model": row["model"],
                            **(row["cohorts"] or {}),
                        },
                    }
                )
        return {
            "sources": inventory,
            "snapshot_id": metadata["snapshot_id"],
            "created_at": metadata["created_at"],
            "synced_at": metadata["loaded_at"],
        }

    @api.get("/assessment")
    def assessment(source_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            source = source_record(connection, source_id)
            if source["assessment_key"] is None:
                raise HTTPException(404, source["reason"] or "Assessment unavailable")
            parameters = {"load_id": source["load_id"], "pipeline_key": source["assessment_key"]}
            row = (
                connection.execute(
                    text(
                        """
                SELECT pipeline_key, source_id, scope, model, rubric, curation, curation_uri, traces
                FROM atlas_assessments WHERE load_id = :load_id AND pipeline_key = :pipeline_key
            """
                    ),
                    parameters,
                )
                .mappings()
                .one()
            )
            summaries = cohort_summaries(connection, **parameters)
            samples = [
                dict(item)
                for item in connection.execute(
                    text(
                        """
                SELECT task_id, cohort, verdict, status, exclusion_reason
                FROM atlas_items WHERE load_id = :load_id AND pipeline_key = :pipeline_key
                ORDER BY cohort, task_id
            """
                    ),
                    parameters,
                ).mappings()
            ]
        return {
            "source": source["declaration"]["display_name"],
            "report": {
                **row,
                "assessment": summaries,
                "evidence": {"samples": samples},
            },
        }

    @api.get("/evidence")
    def evidence(source_id: str, task_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                SELECT i.task_id, i.cohort, i.verdict, i.status, i.explanation, i.evidence_ids,
                    i.omissions, i.exclusion_reason, p.packet
                FROM atlas_sources s JOIN atlas_current c USING (load_id)
                JOIN atlas_items i ON i.load_id = s.load_id AND i.pipeline_key = s.assessment_key
                JOIN atlas_packets p ON p.load_id = i.load_id
                    AND p.pipeline_key = i.pipeline_key AND p.task_id = i.task_id
                WHERE c.name = 'latest' AND s.id = :source_id AND i.task_id = :task_id
            """
                    ),
                    {"source_id": source_id, "task_id": task_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise HTTPException(404, "Task is outside this assessment sample")
        return {
            "judgment": {
                key: row[key] for key in ("task_id", "cohort", "verdict", "status", "explanation", "evidence_ids")
            },
            "packet": json.loads(row["packet"]),
            "omissions": row["omissions"],
            "exclusion_reason": row["exclusion_reason"],
        }

    @api.get("/archive")
    def archive() -> dict[str, Any]:
        with engine.connect() as connection:
            rows = list(
                connection.execute(
                    text(
                        """
                SELECT id, source_id, updated_at, jsonb_array_length(collection->'reviews') AS review_count
                FROM catalog_reviews ORDER BY source_id, updated_at
            """
                    )
                ).mappings()
            )
        return {"reviews": [dict(row) for row in rows]}

    @api.get("/archive/reviews/{review_id}")
    def archived_review(review_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            record = (
                connection.execute(
                    text("SELECT id, source_id, collection, updated_at FROM catalog_reviews WHERE id = :id"),
                    {"id": review_id},
                )
                .mappings()
                .one_or_none()
            )
            files = list(
                connection.execute(
                    text("SELECT path, sha256 FROM review_artifacts WHERE review_id = :id ORDER BY path"),
                    {"id": review_id},
                ).mappings()
            )
        if record is None:
            raise HTTPException(404, "Archived review not found")
        return {**record, "artifacts": [dict(row) for row in files]}

    @api.get("/archive/reviews/{review_id}/artifacts/{path:path}")
    def archived_artifact(review_id: str, path: str) -> Response:
        with engine.connect() as connection:
            record = (
                connection.execute(
                    text("SELECT content, sha256 FROM review_artifacts WHERE review_id = :id AND path = :path"),
                    {"id": review_id, "path": path},
                )
                .mappings()
                .one_or_none()
            )
        if record is None:
            raise HTTPException(404, "Archived artifact not found")
        return Response(record["content"], media_type="text/plain", headers={"ETag": record["sha256"]})

    return api
