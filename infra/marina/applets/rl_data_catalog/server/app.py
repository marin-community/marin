# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated task-curation inventory with persistent reviews and revision checks."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from marina.applets import AppletServices
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .catalog import CATALOG_PATH, Snapshot, catalog_snapshots
from .verifier_policy import migrate_verifier_policy

DIFFICULTY_PROTOCOL = "atlas-difficulty-v3-65k16k-qwen-recommended-nonthinking"
JUDGE_VERIFIER_DIFFICULTY_PROTOCOL = "atlas-difficulty-v4-judge-verifier-nonthinking"
CHECKLIST_JUDGE_DIFFICULTY_PROTOCOL = "atlas-difficulty-v4-checklist-judge"
CHECKLIST_JUDGE_SOURCES = {
    "Task Trove:laion__nemotron-gym-safety-v3",
    "Task Trove:laion__stackexchange-overflow-sandboxes-verified-v2",
    "Task Trove:laion__stackexchange-unix-sandboxes-verified-v2",
}
DIFFICULTY_MODELS = {
    "small": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "large": "Qwen/Qwen3.5-122B-A10B",
    "hosted": "zai-org/GLM-5.3",
}
DIFFICULTY_LIMITS = {"context_window": 65536, "max_input_tokens": 49152, "max_output_tokens": 16384}
DIFFICULTY_GENERATION = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "repetition_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "max_tokens": 16384,
}


def difficulty_protocol_status(report: dict[str, Any]) -> tuple[str, str]:
    """Distinguish matched current measurements from retained historical runs."""
    protocol = report.get("protocol") or {}
    protocol_id = protocol.get("id")
    if protocol_id not in (
        DIFFICULTY_PROTOCOL,
        JUDGE_VERIFIER_DIFFICULTY_PROTOCOL,
        CHECKLIST_JUDGE_DIFFICULTY_PROTOCOL,
    ):
        return "historical", "Earlier model identities or generation budgets; retained as historical evidence."
    if protocol_id == JUDGE_VERIFIER_DIFFICULTY_PROTOCOL:
        judge = (report.get("verifier_configuration") or {}).get("tasktrove_judge") or {}
        if (
            judge.get("model") != "Qwen/Qwen3.5-9B"
            or judge.get("provider") != "Together"
            or judge.get("chat_template_kwargs") != {"enable_thinking": False}
            or not judge.get("relay_script_sha256")
        ):
            return "invalid", "The TaskTrove verifier judge lacks the recorded non-thinking Qwen3.5-9B configuration."
    if protocol_id == CHECKLIST_JUDGE_DIFFICULTY_PROTOCOL:
        judge = (report.get("verifier_configuration") or {}).get("tasktrove_judge") or {}
        if (
            report.get("atlas_id") not in CHECKLIST_JUDGE_SOURCES
            or judge.get("model") != "deepseek-ai/DeepSeek-V4-Pro-0813"
            or judge.get("provider") != "Together"
            or judge.get("base_url") != "https://api.together.xyz/v1"
            or judge.get("api_key_reference") != "${TOGETHER_API_KEY}"
            or judge.get("chat_template_kwargs") != {}
            or not judge.get("substitution_reason")
        ):
            return "invalid", "The TaskTrove checklist judge lacks its recorded Together configuration."
    if any(protocol.get(key) != value for key, value in DIFFICULTY_LIMITS.items()):
        return "invalid", "The recorded context or output limits do not match the current protocol."
    models = report["models"]
    if len(models) != len(DIFFICULTY_MODELS) or {model.get("size") for model in models} != DIFFICULTY_MODELS.keys():
        return "invalid", "The current comparison requires one Small, Large and Hosted model."
    for model in models:
        if model.get("model") != DIFFICULTY_MODELS[model["size"]]:
            return "invalid", "A model identity does not match its designated comparison role."
        parameters = model.get("generation_parameters") or {}
        expected_generation = DIFFICULTY_GENERATION
        if model["size"] == "large":
            expected_generation = {**DIFFICULTY_GENERATION, "top_p": 0.8, "presence_penalty": 1.5}
        if any(parameters.get(key) != value for key, value in expected_generation.items()):
            return "invalid", "A model's generation parameters do not match the current protocol."
        if (
            model["size"] == "large"
            and (parameters.get("chat_template_kwargs") or {}).get("enable_thinking") is not False
        ):
            return "invalid", "The Large model's recorded configuration does not disable thinking."
        if model["size"] == "hosted" and parameters.get("reasoning_effort") != "low":
            return "invalid", "The Hosted model's recorded configuration does not use Low reasoning effort."
    note = "Matched Small, Large and Hosted models at 65,536 context and 16,384 output tokens."
    if protocol_id == JUDGE_VERIFIER_DIFFICULTY_PROTOCOL:
        note += " The native TaskTrove verifier used Qwen3.5-9B with thinking disabled."
    if protocol_id == CHECKLIST_JUDGE_DIFFICULTY_PROTOCOL:
        note += " The native TaskTrove checklist verifier used DeepSeek-V4-Pro through Together."
    return "current", note


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
        "generation_parameters",
    )
    status, note = difficulty_protocol_status(report)
    models = [{key: model.get(key) for key in fields} for model in report["models"]]
    for model in models:
        model["measurement_status"] = status
    for followup in report.get("protocol_followups", []):
        if followup["state"] == "complete":
            models.append(
                {
                    **{key: followup.get(key) for key in fields},
                    "size": "hosted" if followup.get("kind") == "alternate_checkpoint" else "followup",
                    "measurement_status": "historical",
                    "followup_id": followup.get("id"),
                    "display_name": (
                        "Generation setting follow-up"
                        if followup.get("changed_parameter")
                        else "Native verifier recheck"
                    ),
                }
            )
    audited_ordering = any(
        artifact["path"].endswith("/ordering-audit.json") for artifact in report.get("protocol", {}).get("artifacts", [])
    )
    return {
        "models": models,
        "estimated_at": report["estimated_at"],
        "sampling": report["sampling"],
        "status": status,
        "status_note": note,
        "ordering_warning": report["limitations"][-1] if audited_ordering else None,
        "protocol": report.get("protocol"),
    }


def grading_binding_valid(record: dict[str, Any], row: dict[str, Any]) -> bool:
    """Check the archived proof against the current source and historical review."""
    grading_revision = row.get("grading_revision")
    grading_binding = record.get("grading_binding")
    proof = record.get("grading_proof")
    if not grading_revision or not grading_binding or not proof:
        return False
    content = proof["content"]
    digest = hashlib.sha256(content.encode()).hexdigest()
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = None  # Invalid evidence keeps the source stale; other sources remain available.
    claim = decoded if isinstance(decoded, dict) else {}
    expected = {
        "source_id": row["id"],
        "review_id": row["review_id"],
        "source_revision": row.get("dataset_revision") or row.get("revision"),
        "captured_verifier_revision": row["review_verifier_revision"],
        "grading_revision": grading_revision,
    }
    return (
        digest == proof["sha256"] == grading_binding["evidence_sha256"]
        and claim.get("schema_version") == 1
        and claim.get("equivalent") is True
        and all(grading_binding.get(key) == value and claim.get(key) == value for key, value in expected.items())
    )


def source_with_review(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record["payload"])
    row.update(
        {
            key: record[key] if record["review_id"] or record[key] is not None else row.get(key)
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
    grading_revision = row.get("grading_revision")
    binding_valid = grading_binding_valid(record, row)
    grading_enrolled = record.get("grading_enrolled", bool(grading_revision))
    row["grading_tracking"] = "source-specific" if grading_enrolled else "legacy"
    row["review_grading_revision"] = grading_revision if binding_valid else None
    verifier_changed = (
        not binding_valid
        if grading_enrolled
        else row["review_verifier_revision"] is not None
        and row["review_verifier_revision"] != row.get("verifier_revision")
    )
    row["review_stale"] = bool(
        (row["review_source_revision"] is not None and row["review_source_revision"] != current_data) or verifier_changed
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
        summary = row["difficulty_summary"]
        counts = "; ".join(
            f"{model['model'] or model.get('display_name') or 'Native verifier recheck'} "
            f"{model['solved']}/{model['verified']} solved"
            for model in summary["models"]
        )
        row["difficulty"] = f"{summary['status'].title()}: {counts}"
    return row


def migrate(connection: Connection, catalog_path: Path = CATALOG_PATH) -> None:
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
            CREATE TABLE IF NOT EXISTS catalog_grading_reviews (
                source_id TEXT NOT NULL, review_id TEXT NOT NULL,
                source_revision TEXT NOT NULL, captured_verifier_revision TEXT NOT NULL,
                grading_revision TEXT NOT NULL, evidence_path TEXT NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                PRIMARY KEY (source_id, review_id, source_revision, grading_revision)
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

    migrate_verifier_policy(connection)
    refresh_catalog(connection, catalog_path)


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
            ON CONFLICT (id) DO UPDATE SET origin = EXCLUDED.origin, payload = EXCLUDED.payload, active = TRUE
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


def refresh_catalog(connection: Connection, catalog_path: Path = CATALOG_PATH, force: bool = False) -> dict[str, Any]:
    """Reconcile a complete generated catalog without changing stored reviews."""
    snapshots = catalog_snapshots(catalog_path)
    lock = connection.execute(
        text("SELECT pg_try_advisory_xact_lock(hashtext(current_schema() || '/catalog-refresh'))")
    ).scalar_one()
    if not lock:
        return {"busy": True, "message": "Another visitor is refreshing the catalog. Your saved data remains available."}
    results = []
    for snapshot in snapshots:
        previous = connection.execute(
            text("SELECT revision FROM catalog_refreshes WHERE origin = :origin"), {"origin": snapshot.origin}
        ).scalar_one_or_none()
        changed = force or snapshot.revision != previous
        if changed:
            save_snapshot(connection, snapshot)
        else:
            connection.execute(
                text("UPDATE catalog_refreshes SET checked_at = NOW(), error = NULL WHERE origin = :origin"),
                {"origin": snapshot.origin},
            )
        results.append({"origin": snapshot.origin, "revision": snapshot.revision, "changed": changed})
    origins = {snapshot.origin for snapshot in snapshots}
    existing = connection.execute(text("SELECT DISTINCT origin FROM catalog_sources")).scalars()
    for origin in set(existing) - origins:
        connection.execute(text("UPDATE catalog_sources SET active = FALSE WHERE origin = :origin"), {"origin": origin})
        connection.execute(text("DELETE FROM catalog_refreshes WHERE origin = :origin"), {"origin": origin})
    return {"busy": False, "results": results}


def active_sources_with_review_state(connection: Connection) -> list[dict[str, Any]]:
    """Read active sources with their review and grading applicability evidence."""
    return [
        source_with_review(dict(row))
        for row in connection.execute(
            text(
                """
                SELECT s.*, a.content AS difficulty_report,
                EXISTS (SELECT 1 FROM catalog_grading_reviews h
                    WHERE h.source_id = s.id) AS grading_enrolled,
                to_jsonb(g) AS grading_binding,
                CASE WHEN p.path IS NOT NULL THEN jsonb_build_object(
                    'content', p.content, 'sha256', p.sha256
                ) END AS grading_proof, COALESCE((
                    SELECT jsonb_agg(jsonb_build_object(
                        'issue_url', i.issue_url, 'review_id', i.review_id,
                        'status', i.status, 'created_at', i.created_at
                    ) ORDER BY i.created_at)
                    FROM catalog_verifier_issues i
                    WHERE i.source_id = s.id AND i.status = 'open'
                ), '[]'::jsonb) AS verifier_issues
                FROM catalog_sources s LEFT JOIN review_artifacts a
                ON a.review_id = s.review_id AND a.path = 'difficulty.json'
                LEFT JOIN catalog_grading_reviews g ON g.source_id = s.id
                AND g.review_id = s.review_id
                AND g.source_revision = s.review_source_revision
                AND g.grading_revision = s.payload->>'grading_revision'
                LEFT JOIN review_artifacts p ON p.review_id = g.review_id
                AND p.path = g.evidence_path
                WHERE s.active ORDER BY s.origin, s.id
            """
            )
        ).mappings()
    ]


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/sources")
    def sources() -> dict[str, Any]:
        with engine.connect() as connection:
            rows = active_sources_with_review_state(connection)
            refreshes = [
                dict(row)
                for row in connection.execute(text("SELECT * FROM catalog_refreshes ORDER BY origin")).mappings()
            ]
        return {"sources": rows, "refreshes": refreshes}

    @api.get("/reviews/{review_id}/difficulty")
    def review_difficulty(review_id: str, path: str = "difficulty.json") -> dict[str, Any]:
        with engine.connect() as connection:
            content = connection.execute(
                text("SELECT content FROM review_artifacts WHERE review_id = :id AND path = :path"),
                {"id": review_id, "path": path},
            ).scalar_one_or_none()
        if content is None:
            raise HTTPException(404, "Difficulty report not found")
        report = json.loads(content)
        if not isinstance(report, dict) or "models" not in report:
            raise HTTPException(400, "Artifact is not a model comparison report")
        summary = difficulty_summary(report)
        if path != "difficulty.json":
            summary["status"] = "historical"
            summary["status_note"] = (
                "An archived report; original model identities, settings and evidence are preserved."
            )
            for model in summary["models"]:
                model["measurement_status"] = "historical"
        return summary

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
                                WHERE r->'attributes'->>'review_pool_role'
                                    IN ('verifier_defect', 'verifier_revision_attestation')
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
    def refresh(force: bool = False) -> dict[str, Any]:
        with engine.begin() as connection:
            return refresh_catalog(connection, force=force)

    return api
