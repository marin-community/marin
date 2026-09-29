# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent source inventory with atomic, independently recoverable upstream refreshes."""

import json
import logging
from dataclasses import asdict
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .catalog import (
    SKYRL,
    SKYRL_ORIGIN,
    TASKTROVE,
    TASKTROVE_CLASSIFICATION,
    TASKTROVE_ORIGIN,
    Snapshot,
    annotate_source,
    get_json,
    merge_gym_sources,
    skyrl_snapshot,
    tasktrove_snapshot,
)
from .hf_auth import HFCredentialError, HuggingFaceAuth, runtime_hf_token
from .verifier_policy import migrate_verifier_policy

logger = logging.getLogger(__name__)


def difficulty_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return measured solve counts for the catalog's visual comparison."""
    fields = (
        "size",
        "model",
        "model_revision",
        "provider",
        "solved",
        "verified",
        "unverified",
        "attempted",
        "solve_rate",
        "wilson_95",
    )
    models = [{key: model.get(key) for key in fields} for model in report["models"]]
    for followup in report.get("protocol_followups", []):
        if followup["state"] == "complete":
            models.append(
                {
                    **{key: followup.get(key) for key in fields},
                    "size": "hosted" if followup.get("kind") == "alternate_checkpoint" else "followup",
                }
            )
    return {"models": models, "estimated_at": report["estimated_at"], "sampling": report["sampling"]}


def source_with_review(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record["payload"])
    row.update(
        {
            key: record[key]
            for key in (
                "difficulty",
                "quality",
                "traces",
                "review_date",
                "review_id",
                "review_source_revision",
                "review_verifier_revision",
            )
        }
    )
    current_data = row.get("dataset_revision") or row.get("revision")
    row["review_stale"] = bool(
        row["review_id"]
        and (
            (row["review_source_revision"] is not None and row["review_source_revision"] != current_data)
            or (
                row["review_verifier_revision"] is not None
                and row["review_verifier_revision"] != row.get("verifier_revision")
            )
        )
    )
    row["review_applicability"] = (
        "stale" if row["review_stale"] else "unknown" if row["review_source_revision"] is None else "current"
    )
    if row["review_stale"]:
        row["quality"] = None
        row["difficulty"] = None
    row["verifier_issues"] = record["verifier_issues"]
    if row["verifier_issues"]:
        row["quality"] = "bad" if record["quality"] == "bad" else "some_issues"
        row["difficulty"] = None
    row["difficulty_summary"] = None
    if row["quality"] == "good" and row["difficulty"] and record.get("difficulty_report"):
        row["difficulty_summary"] = difficulty_summary(json.loads(record["difficulty_report"]))
    return row


def migrate(connection: Connection) -> None:
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS catalog_sources (
            id TEXT PRIMARY KEY, origin TEXT NOT NULL, payload JSONB NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            difficulty TEXT, quality TEXT, traces BIGINT
        )
    """
        )
    )

    for definition in (
        "review_date TIMESTAMPTZ",
        "review_id TEXT",
        "review_source_revision TEXT",
        "review_verifier_revision TEXT",
    ):
        connection.execute(text(f"ALTER TABLE catalog_sources ADD COLUMN IF NOT EXISTS {definition}"))
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
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS catalog_refreshes (
            origin TEXT PRIMARY KEY, revision TEXT, revised_at TEXT,
            checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            synced_at TIMESTAMPTZ, error TEXT, row_count INTEGER
        )
    """
        )
    )

    connection.execute(
        text("UPDATE catalog_sources SET payload = payload || CAST(:classification AS JSONB) WHERE origin = :origin"),
        {"classification": json.dumps(TASKTROVE_CLASSIFICATION), "origin": TASKTROVE_ORIGIN},
    )
    migrate_verifier_policy(connection)
    rows = [
        dict(payload)
        for payload in connection.execute(
            text("SELECT payload FROM catalog_sources WHERE active AND payload ? 'name'")
        ).scalars()
    ]
    rows = merge_gym_sources(rows)
    for row in rows:
        annotate_source(row)
    connection.execute(text("UPDATE catalog_sources SET active = FALSE WHERE payload->>'kind' = 'Environment'"))
    for row in rows:
        connection.execute(
            text("UPDATE catalog_sources SET payload = CAST(:payload AS JSONB), active = :active WHERE id = :id"),
            {
                "id": row["id"],
                "payload": json.dumps(row),
                "active": True,
            },
        )


def save_snapshot(connection: Connection, snapshot: Snapshot) -> None:
    connection.execute(
        text("UPDATE catalog_sources SET active = FALSE WHERE origin = :origin"), {"origin": snapshot.origin}
    )
    for row in snapshot.rows:
        connection.execute(
            text(
                """
            INSERT INTO catalog_sources (id, origin, payload, active)
            VALUES (:id, :origin, CAST(:payload AS JSONB), TRUE)
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, active = TRUE
        """
            ),
            {"id": row["id"], "origin": snapshot.origin, "payload": json.dumps(row)},
        )
    fields = asdict(snapshot)
    connection.execute(
        text(
            """
        INSERT INTO catalog_refreshes (origin, revision, revised_at, synced_at, row_count)
        VALUES (:origin, :revision, :revised_at, NOW(), :row_count)
        ON CONFLICT (origin) DO UPDATE SET revision = EXCLUDED.revision,
            revised_at = EXCLUDED.revised_at, checked_at = NOW(), synced_at = NOW(),
            row_count = EXCLUDED.row_count, error = NULL
    """
        ),
        {key: fields[key] for key in ("origin", "revision", "revised_at")} | {"row_count": len(snapshot.rows)},
    )


def refresh_catalog(connection: Connection, client: httpx.Client, force: bool) -> dict[str, Any]:
    lock = connection.execute(
        text("SELECT pg_try_advisory_xact_lock(hashtext(current_schema() || '/catalog-refresh'))")
    ).scalar_one()
    if not lock:
        return {"busy": True, "message": "Another visitor is refreshing the catalog. Your saved data remains available."}
    results = []
    for origin in (SKYRL_ORIGIN, TASKTROVE_ORIGIN):
        previous = connection.execute(
            text("SELECT revision FROM catalog_refreshes WHERE origin = :origin"), {"origin": origin}
        ).scalar_one_or_none()
        try:
            if origin == SKYRL_ORIGIN:
                head = get_json(client, f"https://api.github.com/repos/{SKYRL}/commits/main")
                revision = head["sha"]
                cached_rows = [
                    dict(row)
                    for row in connection.execute(
                        text("SELECT payload FROM catalog_sources WHERE origin = :origin AND active"), {"origin": origin}
                    ).scalars()
                ]
                snapshot = skyrl_snapshot(client, head, cached_rows, force=force)
            else:
                info = get_json(client, f"https://huggingface.co/api/datasets/{TASKTROVE}")
                revision = info["sha"]
                manifest = (
                    get_json(client, f"https://huggingface.co/datasets/{TASKTROVE}/raw/{revision}/manifest.json")
                    if force or revision != previous
                    else None
                )
                snapshot = tasktrove_snapshot(manifest, info) if manifest is not None else None
        except (
            httpx.HTTPError,
            HFCredentialError,
            ValueError,
            KeyError,
            TypeError,
            SyntaxError,
            StopIteration,
        ) as error:
            # Preserve the last successful upstream snapshot and expose refresh failures.
            logger.exception("Catalog refresh failed for %s", origin)
            message = str(error) or type(error).__name__
            connection.execute(
                text(
                    """
                INSERT INTO catalog_refreshes (origin, error) VALUES (:origin, :error)
                ON CONFLICT (origin) DO UPDATE SET checked_at = NOW(), error = EXCLUDED.error
            """
                ),
                {"origin": origin, "error": message},
            )
            results.append({"origin": origin, "error": message})
            continue
        if snapshot is not None:
            save_snapshot(connection, snapshot)
        else:
            connection.execute(
                text("UPDATE catalog_refreshes SET checked_at = NOW(), error = NULL WHERE origin = :origin"),
                {"origin": origin},
            )
        results.append({"origin": origin, "revision": revision, "changed": snapshot is not None})
    return {"busy": False, "results": results}


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/sources")
    def sources() -> dict[str, Any]:
        with engine.connect() as connection:
            rows = [
                source_with_review(dict(row))
                for row in connection.execute(
                    text(
                        """
                        SELECT s.*, a.content AS difficulty_report, COALESCE((
                            SELECT jsonb_agg(jsonb_build_object(
                                'issue_url', i.issue_url, 'review_id', i.review_id,
                                'status', i.status, 'created_at', i.created_at
                            ) ORDER BY i.created_at)
                            FROM catalog_verifier_issues i
                            WHERE i.source_id = s.id AND i.status = 'open'
                        ), '[]'::jsonb) AS verifier_issues
                        FROM catalog_sources s LEFT JOIN review_artifacts a
                        ON a.review_id = s.review_id AND a.path = 'difficulty.json'
                        WHERE s.active ORDER BY s.origin, s.id
                    """
                    )
                ).mappings()
            ]
            refreshes = [
                dict(row)
                for row in connection.execute(text("SELECT * FROM catalog_refreshes ORDER BY origin")).mappings()
            ]
        return {"sources": rows, "refreshes": refreshes}

    @api.get("/reviews/{review_id}")
    def review(review_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            record = (
                connection.execute(text("SELECT * FROM catalog_reviews WHERE id = :id"), {"id": review_id})
                .mappings()
                .one_or_none()
            )
            artifacts = [
                dict(row)
                for row in connection.execute(
                    text("SELECT path, sha256 FROM review_artifacts WHERE review_id = :id ORDER BY path"),
                    {"id": review_id},
                ).mappings()
            ]
            supplemental = []
            verifier_issues = []
            review_pool = []
            if record is not None:
                review_pool = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            "SELECT id, updated_at, jsonb_array_length(collection->'reviews') AS review_count "
                            "FROM catalog_reviews WHERE source_id = :source AND id <> :id "
                            "ORDER BY updated_at DESC, id"
                        ),
                        {"source": record["source_id"], "id": review_id},
                    ).mappings()
                ]
                verifier_issues = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            "SELECT issue_url, review_id, status FROM catalog_verifier_issues "
                            "WHERE source_id = :source AND status = 'open' ORDER BY issue_url"
                        ),
                        {"source": record["source_id"]},
                    ).mappings()
                ]
                supplemental = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            SELECT id, collection FROM catalog_reviews
                            WHERE source_id = :source AND id <> :id AND EXISTS (
                                SELECT 1 FROM jsonb_array_elements(collection->'reviews') r
                                WHERE r->'attributes'->>'review_pool_role' = 'verifier_defect'
                            ) ORDER BY updated_at, id
                        """
                        ),
                        {"source": record["source_id"], "id": review_id},
                    ).mappings()
                ]
        if record is None:
            raise HTTPException(404, "Review not found")
        result = {str(key): value for key, value in record.items()}
        result["artifacts"] = artifacts
        result["supplemental_reviews"] = supplemental
        result["verifier_issues"] = verifier_issues
        result["review_pool"] = review_pool
        return result

    @api.get("/reviews/{review_id}/artifacts/{path:path}")
    def artifact(review_id: str, path: str) -> Response:
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
            raise HTTPException(404, "Artifact not found")
        return Response(record["content"], media_type="text/plain", headers={"ETag": record["sha256"]})

    @api.post("/refresh")
    def refresh(request: Request, force: bool = False, hf_auth: Literal["auto", "runtime"] = "auto") -> dict[str, Any]:
        caller_token = request.headers.get("X-HuggingFace-Token")
        auth = HuggingFaceAuth(runtime_hf_token() if hf_auth == "runtime" else caller_token)
        with httpx.Client(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Marina-RL-Data-Atlas"},
            auth=auth,
        ) as client:
            with engine.begin() as connection:
                result = refresh_catalog(connection, client, force)
        credential_source = (
            "runtime_secret"
            if hf_auth == "runtime"
            else "caller" if caller_token else ("runtime_secret" if auth.token else "anonymous")
        )
        return {**result, "hf_authentication": credential_source}

    return api
