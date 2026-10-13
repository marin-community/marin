# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A catalog mirror with archived assessment and legacy evidence."""

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .catalog import CATALOG_ROOT, evidence_document, published_catalog, validate_catalog


def save_catalog(connection: Connection, document: dict[str, Any]) -> None:
    validate_catalog(document)
    connection.execute(
        text(
            """
            INSERT INTO curation_catalog (name, document, synced_at)
            VALUES ('latest', CAST(:document AS JSONB), NOW())
            ON CONFLICT (name) DO UPDATE SET document = EXCLUDED.document, synced_at = NOW()
        """
        ),
        {"document": json.dumps(document, ensure_ascii=False)},
    )


def initialize_catalog(connection: Connection) -> None:
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS curation_catalog (
            name TEXT PRIMARY KEY, document JSONB NOT NULL, synced_at TIMESTAMPTZ NOT NULL
        )
    """
        )
    )
    # Historical tables stay in place; their contents never enter current grades.
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


def migrate(connection: Connection) -> None:
    document = published_catalog(CATALOG_ROOT)
    initialize_catalog(connection)
    save_catalog(connection, document)


def saved_catalog(connection: Connection) -> dict[str, Any]:
    return connection.execute(text("SELECT document FROM curation_catalog WHERE name = 'latest'")).scalar_one()


def source_record(connection: Connection, source_id: str) -> tuple[dict[str, Any], str]:
    record = (
        connection.execute(
            text(
                """
            SELECT row.value AS source, c.document->>'catalog_root' AS catalog_root
            FROM curation_catalog c, jsonb_array_elements(c.document->'sources') AS row(value)
            WHERE c.name = 'latest' AND row.value->>'id' = :source_id
        """
            ),
            {"source_id": source_id},
        )
        .mappings()
        .one_or_none()
    )
    if record is None:
        raise HTTPException(404, "Source not found")
    return record["source"], record["catalog_root"]


def refresh_catalog(connection: Connection, root: str) -> dict[str, Any]:
    lock = connection.execute(
        text("SELECT pg_try_advisory_xact_lock(hashtext(current_schema() || '/curation-refresh'))")
    ).scalar_one()
    if not lock:
        return {"busy": True}
    previous = saved_catalog(connection)
    document = published_catalog(root, previous)
    changed = document["snapshot_id"] != previous["snapshot_id"] or document["revision"] != previous["revision"]
    if changed:
        save_catalog(connection, document)
    return {"busy": False, "changed": changed, "snapshot_id": document["snapshot_id"]}


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()
    with engine.connect() as connection:
        root = saved_catalog(connection)["catalog_root"]

    @api.get("/sources")
    def sources() -> dict[str, Any]:
        with engine.connect() as connection:
            document = saved_catalog(connection)
            synced_at = connection.execute(
                text("SELECT synced_at FROM curation_catalog WHERE name = 'latest'")
            ).scalar_one()
        rows = [
            {**row, "assessment": {name: value for name, value in row["assessment"].items() if name != "samples"}}
            for row in document["sources"]
        ]
        return {
            "sources": rows,
            "snapshot_id": document["snapshot_id"],
            "revision": document["revision"],
            "created_at": document["created_at"],
            "synced_at": synced_at,
        }

    @api.get("/assessment")
    def assessment(source_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            source, catalog_root = source_record(connection, source_id)
        data = source["assessment"]
        if data["status"] != "complete":
            raise HTTPException(404, data.get("reason", "Assessment unavailable"))
        return {"source": source["display_name"], "report": evidence_document(catalog_root, data["report_uri"])}

    @api.get("/evidence")
    def evidence(source_id: str, task_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            source, catalog_root = source_record(connection, source_id)
        sample = next(
            (sample for sample in source["assessment"].get("samples", []) if sample["task_id"] == task_id), None
        )
        if sample is None:
            raise HTTPException(404, "Task is outside this assessment sample")
        return evidence_document(catalog_root, sample["evidence_uri"])

    @api.post("/refresh")
    def refresh() -> dict[str, Any]:
        with engine.begin() as connection:
            return refresh_catalog(connection, root)

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
