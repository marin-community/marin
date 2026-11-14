# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the explicitly loaded curation catalog from Postgres."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from finestore.rl.curation import DAYTONA_SAFE_SNAPSHOT_LIMIT
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection
from taskcompendium.source_fingerprints import DuplicateGroup, DuplicateReview

from .contamination import contamination
from .curriculum import register_curriculum_routes, source_coverage
from .daytona import daytona
from .difficulty import difficulty_records, difficulty_summary
from .filtration import filtration
from .quality import intended_use, issue_rate, quality_display, result
from .schema import SCHEMA_SQL
from .source_identity import canonical_source_id, original_source_id


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


def canonical_duplicate_group(row: dict[str, Any]) -> dict[str, Any]:
    return DuplicateGroup.model_validate(
        {
            **row,
            "canonical_source_id": canonical_source_id(row["canonical_source_id"]),
            "members": [{**member, "source_id": canonical_source_id(member["source_id"])} for member in row["members"]],
        }
    ).model_dump(mode="json")


def canonical_duplicate_review(row: dict[str, Any]) -> dict[str, Any]:
    similarity = row["similarity"]
    return DuplicateReview.model_validate(
        {
            **row,
            "similarity": {
                **similarity,
                "left": canonical_source_id(similarity["left"]),
                "right": canonical_source_id(similarity["right"]),
            },
        }
    ).model_dump(mode="json")


def canonical_source_record(row: dict[str, Any]) -> dict[str, Any]:
    """Project retained catalog metadata without rewriting its evidence or keys."""
    result = {**row, "id": canonical_source_id(row["id"])}
    result["declaration"] = {**row["declaration"], "id": canonical_source_id(row["declaration"]["id"])}
    if row.get("backend_capabilities") is not None:
        result["backend_capabilities"] = {
            **row["backend_capabilities"],
            "source_id": canonical_source_id(row["backend_capabilities"]["source_id"]),
        }
    if row.get("duplicate_group") is not None:
        result["duplicate_group"] = canonical_duplicate_group(row["duplicate_group"])
    return result


def source_record(connection: Connection, source_id: str) -> dict[str, Any]:
    row = (
        connection.execute(
            text(
                """
        SELECT s.* FROM atlas_sources s JOIN atlas_current c USING (load_id)
        WHERE c.name = 'latest' AND s.id IN (:source_id, :original_source_id)
    """
            ),
            {"source_id": canonical_source_id(source_id), "original_source_id": original_source_id(source_id)},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(404, "Source not found")
    return canonical_source_record(dict(row))


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


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()
    register_curriculum_routes(api, engine)

    @api.get("/sources")
    def sources() -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            coverage = source_coverage(connection)
            measurements = difficulty_records(connection)
            rows = connection.execute(
                text(
                    """
                SELECT s.*, a.scope, a.model, summary.cohorts, b.capabilities AS backend_capabilities,
                    (s.declaration->'within_source_dedup'->>'unique_count')::bigint AS task_count
                FROM atlas_sources s
                LEFT JOIN atlas_backend_sources b ON b.load_id = s.load_id AND b.source_id = s.id
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
            for retained_row in rows:
                row = canonical_source_record(dict(retained_row))
                display = quality_display(row["declaration"], row["curation"], row["assessment_key"])
                inventory.append(
                    {
                        **row["declaration"],
                        "task_count": row["task_count"],
                        "curation": row["curation"],
                        "quality": display.quality,
                        "quality_review_status": display.review_status,
                        "filtration": filtration(row["declaration"]),
                        "review_completeness": row["review_completeness"],
                        "result_provisional": (row["review_completeness"] or {}).get("provisional"),
                        "contamination_status": contamination(row["declaration"]),
                        "result": result(row["declaration"], (row["cohorts"] or {}).get("admitted")),
                        "issue_rate": issue_rate(row["declaration"], (row["cohorts"] or {}).get("admitted")),
                        "updated_at": metadata["loaded_at"],
                        "daytona": daytona(row["declaration"], DAYTONA_SAFE_SNAPSHOT_LIMIT),
                        "coverage": coverage.get(row["id"]),
                        "difficulty": difficulty_summary(row["declaration"], measurements.get(row["id"], [])),
                        "intended_use": intended_use(row["declaration"]),
                        "backend_capabilities": row["backend_capabilities"],
                        "source_fingerprint": (
                            {
                                key: value
                                for key, value in row["source_fingerprint"].items()
                                if key
                                in {
                                    "version",
                                    "seed",
                                    "task_count",
                                    "within_source_duplicates",
                                    "sample_sha256",
                                    "lineage",
                                }
                            }
                            if row["source_fingerprint"] is not None
                            else None
                        ),
                        "duplicate_group": row["duplicate_group"],
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
            "daytona_snapshot_limit": DAYTONA_SAFE_SNAPSHOT_LIMIT,
        }

    @api.get("/backends")
    def backends() -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            row = (
                connection.execute(
                    text("SELECT survey_id, survey FROM atlas_backend_surveys WHERE load_id = :load_id"),
                    {"load_id": metadata["load_id"]},
                )
                .mappings()
                .first()
            )
            if row is None:
                raise HTTPException(503, "Selected catalog has no backend survey")
            cells = (
                connection.execute(
                    text("SELECT cell FROM atlas_backend_cells WHERE load_id = :load_id ORDER BY backend, requirement"),
                    {"load_id": metadata["load_id"]},
                )
                .scalars()
                .all()
            )
            return {"survey_id": row["survey_id"], **row["survey"], "matrix": cells}

    @api.get("/source-fingerprint")
    def source_fingerprint(source_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            source = source_record(connection, source_id)
            return {"source_id": canonical_source_id(source_id), "fingerprint": source["source_fingerprint"]}

    @api.get("/source-duplicates")
    def source_duplicates() -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            parameters = {"load_id": metadata["load_id"]}
            survey = (
                connection.execute(
                    text("SELECT survey_id, metadata FROM atlas_identity_surveys WHERE load_id = :load_id"), parameters
                )
                .mappings()
                .first()
            )
            if survey is None:
                raise HTTPException(503, "Selected catalog has no source identity survey")
            reviews = (
                connection.execute(
                    text(
                        "SELECT review FROM atlas_duplicate_reviews WHERE load_id = :load_id "
                        "ORDER BY left_source, right_source"
                    ),
                    parameters,
                )
                .scalars()
                .all()
            )
            groups = (
                connection.execute(
                    text(
                        "SELECT DISTINCT duplicate_group FROM atlas_sources WHERE load_id = :load_id "
                        "AND duplicate_group IS NOT NULL"
                    ),
                    parameters,
                )
                .scalars()
                .all()
            )
            return {
                "survey_id": survey["survey_id"],
                **survey["metadata"],
                "reviews": [canonical_duplicate_review(row) for row in reviews],
                "groups": [canonical_duplicate_group(row) for row in groups],
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
                "source_id": canonical_source_id(row["source_id"]),
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
                WHERE c.name = 'latest' AND s.id IN (:source_id, :original_source_id) AND i.task_id = :task_id
            """
                    ),
                    {
                        "source_id": canonical_source_id(source_id),
                        "original_source_id": original_source_id(source_id),
                        "task_id": task_id,
                    },
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

    @api.get("/source")
    def source_evidence(source_id: str) -> dict[str, Any]:
        """Every quality-evidence pointer the catalog holds for one source."""
        with engine.connect() as connection:
            source = source_record(connection, source_id)
            assessment = None
            summaries: dict[str, Any] = {}
            if source["assessment_key"] is not None:
                parameters = {"load_id": source["load_id"], "pipeline_key": source["assessment_key"]}
                assessment = dict(
                    connection.execute(
                        text(
                            """
                    SELECT pipeline_key, scope, model, rubric, curation_uri, traces,
                        (SELECT COUNT(*) FROM atlas_items i WHERE i.load_id = a.load_id
                            AND i.pipeline_key = a.pipeline_key) AS sample_items
                    FROM atlas_assessments a WHERE load_id = :load_id AND pipeline_key = :pipeline_key
                """
                        ),
                        parameters,
                    )
                    .mappings()
                    .one()
                )
                summaries = cohort_summaries(connection, **parameters)
            reviews = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                SELECT id, updated_at, jsonb_array_length(collection->'reviews') AS review_count
                FROM catalog_reviews WHERE source_id IN (:source_id, :original_source_id) ORDER BY updated_at
            """
                    ),
                    {"source_id": canonical_source_id(source_id), "original_source_id": original_source_id(source_id)},
                ).mappings()
            ]
            capabilities = connection.execute(
                text(
                    """
                    SELECT capabilities FROM atlas_backend_sources WHERE load_id=:load_id
                        AND source_id IN (:source_id, :original_source_id)
                    """
                ),
                {
                    "load_id": source["load_id"],
                    "source_id": canonical_source_id(source_id),
                    "original_source_id": original_source_id(source_id),
                },
            ).scalar_one_or_none()
            load = loaded_catalog(connection)
            coverage = source_coverage(connection).get(canonical_source_id(source_id))
            difficulty = difficulty_summary(
                source["declaration"], difficulty_records(connection, source_id).get(canonical_source_id(source_id), [])
            )
        declaration = source["declaration"]
        review = summaries.get("admitted")
        display = quality_display(declaration, source["curation"], source["assessment_key"])
        return {
            "declaration": declaration,
            "coverage": coverage,
            "difficulty": difficulty,
            "intended_use": intended_use(declaration),
            "contamination_status": contamination(declaration),
            "filtration": filtration(declaration),
            "curation": source["curation"],
            "diagnostics": source["diagnostics"],
            "backend_capabilities": capabilities,
            "duplicate_group": source["duplicate_group"],
            "latest_attempt": {
                "key": source["latest_attempt_key"],
                "status": source["latest_attempt_status"],
                "reason": source["reason"],
            },
            "assessment": assessment,
            "cohorts": summaries,
            "archived_reviews": reviews,
            "quality": display.quality,
            "quality_review_status": display.review_status,
            "review_completeness": source["review_completeness"],
            "result_provisional": (source["review_completeness"] or {}).get("provisional"),
            "result": result(declaration, review),
            "issue_rate": issue_rate(declaration, review),
            "daytona": daytona(declaration, DAYTONA_SAFE_SNAPSHOT_LIMIT),
            "updated_at": load["loaded_at"],
            "catalog": {"root": load["catalog_root"], "snapshot_id": load["snapshot_id"]},
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
        return {"reviews": [{**row, "source_id": canonical_source_id(row["source_id"])} for row in rows]}

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
        return {
            "id": record["id"],
            "source_id": canonical_source_id(record["source_id"]),
            "collection": record["collection"],
            "updated_at": record["updated_at"],
            "artifacts": [dict(row) for row in files],
        }

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
