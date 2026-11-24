# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the explicitly loaded curation catalog from Postgres."""

import json
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from finestore.rl.curation import DAYTONA_SAFE_SNAPSHOT_LIMIT, Confidence, Defect, ReferenceStatus, Verdict
from finestore.rl.status_taxonomy import taxonomy_tables
from marina.applets import AppletServices
from shellbox.requirements import Requirement
from sqlalchemy import text
from sqlalchemy.engine import Connection
from taskcompendium.source_fingerprints import DuplicateReview

from .audit import register_audit_routes
from .cohort_summaries import COHORT_SUMMARY_VERSION, materialize_cohort_summaries
from .consumer import consumer_projection
from .contamination import contamination
from .content_access import (
    backend_metadata,
    content_is_withheld,
    content_marker,
    current_source_declaration,
    fingerprint_metadata,
    withheld_declaration_projection,
)
from .contract_selection import contract_registry_key
from .count_evidence import SOURCE_COUNTS
from .curriculum import register_curriculum_routes, source_coverage
from .daytona import daytona
from .derived_contract import contract_projection, contract_records
from .difficulty import difficulty_records, difficulty_summary, table_difficulty
from .difficulty_coverage import difficulty_coverage
from .duplicate_folding import (
    SourceStanding,
    canonical_duplicate_group,
    canonical_duplicate_review,
    fold_groups,
    partial_overlap_notes,
)
from .evidence_content import unpack_content
from .filtration import filtration
from .identity_measurements import IDENTITY_REVIEWS_SQL, IDENTITY_SURVEY_SQL
from .inventory_state import CachedInventory, inventory_input_key
from .licensing import license_projection, license_registry_key
from .methods import variant_projection
from .quality import intended_use, issue_rate, quality_display, result, review_coverage
from .quality_evidence import full_review_records, quality_explanation
from .restrictions import (
    PRIVATE_TEXT_MARKER,
    declaration_restriction,
    redact_advisories,
    redact_mapping,
    restriction_projection,
    restriction_registry_key,
)
from .review_content import task_request_evidence
from .reviews import register_review_routes
from .rollouts import register_rollout_routes
from .schema import SCHEMA_SQL
from .sealed_difficulty import sealed_difficulty_records, sealed_difficulty_summary
from .source_areas import materialize_source_areas
from .source_counts import SourceCounts
from .source_identity import canonical_source_id, original_source_id
from .typed_status import require_typed_served_declaration

MAX_PAGE_ITEMS = 500
DEFAULT_PAGE_ITEMS = 100
COHORT_FILTERS = ("admitted", "excluded")
STATUS_FILTERS = ("reviewed", "unavailable")
DEFECT_FILTERS = tuple(defect.value for defect in Defect)
CONFIDENCE_FILTERS = (*tuple(confidence.value for confidence in Confidence), "unrecorded")
REFERENCE_FILTERS = (*tuple(reference.value for reference in ReferenceStatus), "unrecorded")
VERDICT_FILTERS = tuple(verdict.value for verdict in Verdict)


class SourceProjection(StrEnum):
    COMPLETE = "complete"
    TABLE = "table"


TABLE_SOURCE_FIELDS = frozenset(
    {
        "id",
        "display_name",
        "origin",
        "build_path",
        "status",
        "replacement_source_id",
        "notes",
        "family",
        "tags",
        "dataset_id",
        "dataset_revision",
        "current_release",
        "url",
        "pipeline",
        "grader_type",
        "grader_type_counts",
        "oracle_validation",
        "oracle_filtration",
        "within_source_dedup",
        "lexical_contamination",
        "grading_environment",
        "actor_census_input",
        "consumer_requirements",
        "consumer_receipt",
        "consumer_decision",
        "consumer_findings",
        "daytona",
        "curation",
        "task_count",
        "quality",
        "quality_review_status",
        "quality_explanation",
        "review_scope",
        "review_coverage",
        "filtration",
        "review_completeness",
        "result_provisional",
        "contamination_status",
        "result",
        "issue_rate",
        "updated_at",
        "coverage",
        "difficulty",
        "intended_use",
        "method_tags",
        "data_restriction",
        "backend_capabilities",
        "partial_overlaps",
        "duplicate_group",
        "assessment",
    }
)


def table_source_record(source: dict[str, Any]) -> dict[str, Any]:
    """Exclude stored-only evidence and retain source display values."""
    row = {
        key: table_difficulty(value) if key == "difficulty" else value
        for key, value in source.items()
        if key in TABLE_SOURCE_FIELDS or key.startswith("daytona_")
    }
    if row.get("backend_capabilities") is not None:
        row["backend_capabilities"] = {
            key: value
            for key, value in row["backend_capabilities"].items()
            if key in {"basis", "reason", "task_count", "backends"}
        }
    if row.get("assessment") is not None:
        row["assessment"] = {"status": row["assessment"]["status"]}
    if row.get("within_source_dedup") is not None:
        row["within_source_dedup"] = {
            key: value
            for key, value in row["within_source_dedup"].items()
            if key in {"raw_count", "unique_count", "within_source_duplicates", "drop_manifest_uri"}
        }
    return row


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

    def rows(sql: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        result = connection.execute(text(sql), parameters)
        return [dict(row) for row in result.mappings()] if result.returns_rows else []

    selected = connection.execute(text("SELECT load_id FROM atlas_current WHERE name='latest'")).scalar_one_or_none()
    if selected is not None:
        materialize_cohort_summaries(rows, selected)
    curriculum = connection.execute(
        text("SELECT load_id FROM curriculum_current WHERE name='latest'")
    ).scalar_one_or_none()
    if curriculum is not None:
        materialize_source_areas(rows, curriculum)
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


def duplicate_projection(
    connection: Connection, load_id: str, sources: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, str | float]]]]:
    """Recompute folds from the selected load's measured reviews and current display counts."""
    reviews = [
        DuplicateReview.model_validate(canonical_duplicate_review(row))
        for row in connection.execute(
            text(
                "SELECT review FROM ("
                + IDENTITY_REVIEWS_SQL
                + ") selected ORDER BY review->'similarity'->>'left',review->'similarity'->>'right'"
            ),
            {"load_id": load_id},
        ).scalars()
    ]
    standings = {
        canonical_source_id(row["id"]): SourceStanding(
            task_count=row["declaration"]["within_source_dedup"]["unique_count"],
            quality=quality_display(row["declaration"], row["curation"], row["assessment_key"]).quality,
        )
        for row in sources
    }
    reviews = [row for row in reviews if row.similarity.left in standings and row.similarity.right in standings]
    groups = fold_groups(reviews, standings)
    return (
        {member.source_id: group.model_dump(mode="json") for group in groups for member in group.members},
        partial_overlap_notes(reviews, groups),
    )


def loaded_duplicate_projection(
    connection: Connection, load_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, str | float]]]]:
    rows = connection.execute(
        text("SELECT id, declaration, curation, assessment_key FROM atlas_sources WHERE load_id = :load_id"),
        {"load_id": load_id},
    ).mappings()
    return duplicate_projection(connection, load_id, [dict(row) for row in rows])


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
    result.update(variant_projection(result["declaration"], row.get("method_variants")))
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
    result = canonical_source_record(dict(row))
    review_panel = (
        connection.execute(
            text("SELECT pipeline_key,receipt FROM atlas_review_current WHERE load_id=:load AND source_id=:source"),
            {"load": result["load_id"], "source": result["id"]},
        )
        .mappings()
        .one_or_none()
    )
    if review_panel is not None:
        result["assessment_key"] = review_panel["pipeline_key"]
        result["review_input"] = review_panel["receipt"]
    result["declaration"] = contract_projection(
        result["declaration"], contract_records(connection).get(result["id"], [])
    )
    return result


def redact_cohort_summaries(summaries: dict[str, Any], withheld: bool) -> dict[str, Any]:
    """Redact the reviewer explanations inside each cohort's example list for a private source."""
    if not withheld:
        return summaries
    return {
        cohort: {**summary, "examples": [redact_mapping(example) for example in summary.get("examples") or []]}
        for cohort, summary in summaries.items()
    }


def cohort_summaries(connection: Connection, load_id: str, pipeline_key: str) -> dict[str, Any]:
    return dict(
        connection.execute(
            text(
                """
            SELECT cohort, s.summary ||
                jsonb_build_object('examples', (
                    SELECT COALESCE(jsonb_agg(to_jsonb(example)), '[]'::jsonb) FROM (
                        SELECT task_id, verdict, explanation, evidence_ids FROM atlas_items i
                        WHERE i.load_id = s.load_id AND i.pipeline_key = s.pipeline_key
                            AND i.cohort = s.cohort AND i.verdict <> 'good'
                        ORDER BY task_id LIMIT 5
                    ) example
                )) AS summary
        FROM atlas_cohort_summaries s
        WHERE load_id = :load_id AND pipeline_key = :pipeline_key AND version = :version
    """
            ),
            {"load_id": load_id, "pipeline_key": pipeline_key, "version": COHORT_SUMMARY_VERSION},
        )
        .tuples()
        .all()
    )


def whole_population_counts(summaries: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the SQL cohort view's counts over every cohort of the assessment."""
    verdicts = {verdict: 0 for verdict in VERDICT_FILTERS}
    confidences = {confidence: 0 for confidence in CONFIDENCE_FILTERS}
    defects = {defect: 0 for defect in DEFECT_FILTERS}
    totals = {"item_count": 0, "population": 0, "reviewed": 0}
    for data in summaries.values():
        for key, counts in (
            ("verdict_counts", verdicts),
            ("confidence_counts", confidences),
            ("defect_counts", defects),
        ):
            for name, count in (data.get(key) or {}).items():
                counts[name] = counts.get(name, 0) + count
        # Every stored row carries a verdict, so the verdict counts sum to the row count.
        totals["item_count"] += sum((data.get("verdict_counts") or {}).values())
        totals["population"] += data.get("population") or 0
        totals["reviewed"] += data.get("reviewed") or 0
    return {**totals, "verdict_counts": verdicts, "confidence_counts": confidences, "defect_counts": defects}


def assessment_item_filters(
    cohort: str | None,
    verdict: str | None,
    confidence: str | None,
    reference: str | None,
    status: str | None,
    defect: str | None,
    search: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build the item WHERE clause from validated browse filters; unknown values fail."""
    clauses = []
    parameters: dict[str, Any] = {}
    if cohort is not None:
        if cohort not in COHORT_FILTERS:
            raise HTTPException(422, f"cohort must be one of {COHORT_FILTERS}")
        clauses.append("i.cohort = :cohort")
        parameters["cohort"] = cohort
    if verdict is not None:
        if verdict not in VERDICT_FILTERS:
            raise HTTPException(422, f"verdict must be one of {VERDICT_FILTERS}")
        clauses.append("i.verdict = :verdict")
        parameters["verdict"] = verdict
    if status is not None:
        if status not in STATUS_FILTERS:
            raise HTTPException(422, f"status must be one of {STATUS_FILTERS}")
        clauses.append("i.status = :status")
        parameters["status"] = status
    if confidence is not None:
        if confidence not in CONFIDENCE_FILTERS:
            raise HTTPException(422, f"confidence must be one of {CONFIDENCE_FILTERS}")
        if confidence == "unrecorded":
            clauses.append("i.confidence IS NULL")
        else:
            clauses.append("i.confidence = :confidence")
            parameters["confidence"] = confidence
    if reference is not None:
        if reference not in REFERENCE_FILTERS:
            raise HTTPException(422, f"reference must be one of {REFERENCE_FILTERS}")
        if reference == "unrecorded":
            clauses.append("i.reference_status IS NULL")
        else:
            clauses.append("i.reference_status = :reference")
            parameters["reference"] = reference
    if defect is not None:
        defects = [value.strip() for value in defect.split(",") if value.strip()]
        if not defects or any(value not in DEFECT_FILTERS for value in defects):
            raise HTTPException(422, f"defect must be a comma-separated list of {DEFECT_FILTERS}")
        clauses.append("i.defects ?| :defects")
        parameters["defects"] = defects
    if search is not None:
        if not search.strip():
            raise HTTPException(422, "search must not be empty")
        clauses.append("(i.task_id ILIKE :search OR i.explanation ILIKE :search)")
        parameters["search"] = f"%{search.strip()}%"
    return (f" AND {' AND '.join(clauses)}" if clauses else ""), parameters


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()
    counts = {item.source_id: item for item in SourceCounts.model_validate_json(SOURCE_COUNTS).sources}
    inventory_responses: dict[SourceProjection, CachedInventory] = {}
    register_curriculum_routes(api, engine)
    register_audit_routes(api, engine)
    register_review_routes(api, engine)
    register_rollout_routes(api, engine)

    def inventory_cache_key(connection: Connection, load_id: str) -> str:
        """Cache identity includes SQL row versions and the embedded projection policies."""
        return (
            f"{inventory_input_key(connection, load_id)}|{restriction_registry_key()}"
            f"|{license_registry_key()}|{contract_registry_key()}"
        )

    @api.get("/source-labels")
    def source_labels() -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            rows = connection.execute(
                text(
                    "SELECT id,declaration->>'display_name' AS display_name, "
                    "declaration->>'origin' AS origin,declaration->>'build_path' AS build_path "
                    "FROM atlas_sources WHERE load_id=:load_id ORDER BY id"
                ),
                {"load_id": metadata["load_id"]},
            ).mappings()
            labels = [{**row, "id": canonical_source_id(row["id"])} for row in rows]
        return {"sources": labels, "snapshot_id": metadata["snapshot_id"]}

    @api.get("/taxonomy")
    def taxonomy() -> dict[str, Any]:
        """Serve the canonical status, stage, reason and Use/Method/Format tables.

        Every page renders its status, stage, reason and column labels and
        explanations from these tables; the frontend defines no label of its
        own. The production backend and backend-blocker tables render the
        Backend Evidence page from the same canonical source.
        """
        return taxonomy_tables(requirement_values=[requirement.value for requirement in Requirement])

    @api.get("/source-statuses")
    def source_statuses() -> dict[str, Any]:
        """Serve status-page fields without loading review, backend or difficulty evidence."""
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            rows = connection.execute(
                text(
                    """
                    SELECT id,method_variants,jsonb_build_object(
                        'id',id,'display_name',declaration->'display_name',
                        'status',declaration->'status','stage',declaration->'stage',
                        'exclusion_claims',declaration->'exclusion_claims','notes',declaration->'notes',
                        'replacement_source_id',declaration->'replacement_source_id',
                        'dataset_id',declaration->'dataset_id','license_receipt',declaration->'license_receipt',
                        'within_source_dedup',declaration->'within_source_dedup',
                        'actor_census_input',declaration->'actor_census_input',
                        'intended_use',declaration->'intended_use',
                        'consumer_receipt',declaration->'consumer_receipt',
                        'consumer_decision',declaration->'consumer_decision',
                        'consumer_findings',declaration->'consumer_findings',
                        'consumer_requirements',declaration->'consumer_requirements'
                    ) || CASE WHEN declaration ? 'data_restriction'
                        THEN jsonb_build_object('data_restriction',declaration->'data_restriction')
                        ELSE '{}'::jsonb END AS declaration
                    FROM atlas_sources WHERE load_id=:load_id ORDER BY id
                    """
                ),
                {"load_id": metadata["load_id"]},
            ).mappings()
            sources = []
            for row in rows:
                record = canonical_source_record(dict(row))
                declaration = consumer_projection(license_projection(record["declaration"]))
                if content_is_withheld(declaration):
                    declaration = withheld_declaration_projection(declaration)
                else:
                    declaration = restriction_projection(declaration)
                method_tags = variant_projection(declaration, record.get("method_variants"))["method_tags"]
                sources.append(
                    require_typed_served_declaration(
                        {
                            key: declaration.get(key)
                            for key in (
                                "id",
                                "display_name",
                                "status",
                                "stage",
                                "replacement_source_id",
                                "exclusion_claims",
                                "notes",
                                "consumer_findings",
                                "intended_use",
                            )
                        }
                        | {
                            "data_restriction": declaration.get("data_restriction"),
                            "method_tags": method_tags,
                            "updated_at": metadata["loaded_at"],
                        }
                    )
                )
        return {"sources": sources, "snapshot_id": metadata["snapshot_id"], "synced_at": metadata["loaded_at"]}

    @api.get("/sources")
    def sources(projection: SourceProjection = SourceProjection.COMPLETE) -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            input_key = inventory_cache_key(connection, metadata["load_id"])
            cached = inventory_responses.get(projection)
            if cached is not None and cached.input_key == input_key:
                return cached.payload
            measurements = difficulty_records(connection)
            sealed_measurements = sealed_difficulty_records(connection, metadata["load_id"])
            contracts = contract_records(connection)
            full_reviews = full_review_records(connection)
            rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                SELECT s.id, s.declaration, s.assessment_key, s.latest_attempt_key,
                    s.latest_attempt_status, s.reason, s.curation, s.review_completeness, s.method_variants,
                    CASE WHEN s.source_fingerprint IS NULL THEN NULL ELSE jsonb_build_object(
                        'version', s.source_fingerprint->'version',
                        'seed', s.source_fingerprint->'seed',
                        'task_count', s.source_fingerprint->'task_count',
                        'within_source_duplicates', s.source_fingerprint->'within_source_duplicates',
                        'sample_sha256', s.source_fingerprint->'sample_sha256',
                        'lineage', s.source_fingerprint->'lineage'
                    ) END AS source_fingerprint,
                    a.scope, a.model, b.capabilities AS backend_capabilities,
                    (s.declaration->'within_source_dedup'->>'unique_count')::bigint AS task_count
                FROM atlas_sources s
                LEFT JOIN atlas_backend_sources b ON b.load_id = s.load_id AND b.source_id = s.id
                LEFT JOIN atlas_assessments a
                    ON a.load_id = s.load_id AND a.pipeline_key = s.assessment_key
                WHERE s.load_id = :load_id ORDER BY s.id
            """
                    ),
                    {"load_id": metadata["load_id"]},
                ).mappings()
            ]
            coverage = source_coverage(connection, {canonical_source_id(row["id"]): row["declaration"] for row in rows})
            cohorts: dict[str, Any] = {}
            for summary in connection.execute(
                text(
                    "SELECT pipeline_key,cohort,summary FROM atlas_cohort_summaries "
                    "WHERE load_id=:load_id AND version=:version"
                ),
                {"load_id": metadata["load_id"], "version": COHORT_SUMMARY_VERSION},
            ).mappings():
                cohorts.setdefault(summary["pipeline_key"], {})[summary["cohort"]] = summary["summary"]
            for retained_row in rows:
                retained_row["cohorts"] = cohorts.get(retained_row["assessment_key"])
            groups, overlaps = duplicate_projection(connection, metadata["load_id"], [dict(row) for row in rows])
            inventory = []
            for retained_row in rows:
                row = canonical_source_record(dict(retained_row))
                withheld = content_is_withheld(row["declaration"])
                restriction = declaration_restriction(row["declaration"])
                row["declaration"] = consumer_projection(
                    license_projection(contract_projection(row["declaration"], contracts.get(row["id"], [])))
                )
                served_declaration = (
                    withheld_declaration_projection(row["declaration"]) if withheld else row["declaration"]
                )
                served_declaration = require_typed_served_declaration(
                    {**served_declaration, "method_tags": row["method_tags"]}
                )
                display = quality_display(row["declaration"], row["curation"], row["assessment_key"])
                inventory.append(
                    {
                        **served_declaration,
                        "method_tags": row["method_tags"],
                        "method_variants": row["method_variants"],
                        "data_restriction": restriction.value,
                        "task_count": row["task_count"],
                        "curation": None if withheld else row["curation"],
                        "quality": display.quality,
                        "quality_review_status": display.review_status,
                        "quality_explanation": quality_explanation(
                            row["declaration"], display.quality, full_reviews.get(row["id"], [])
                        ),
                        "review_scope": display.quality.value,
                        "review_coverage": review_coverage(row["declaration"], row["review_completeness"]),
                        "filtration": filtration(row["declaration"], counts.get(row["id"])),
                        "review_completeness": row["review_completeness"],
                        "result_provisional": (row["review_completeness"] or {}).get("provisional"),
                        "contamination_status": contamination(row["declaration"]),
                        "result": result(row["declaration"], (row["cohorts"] or {}).get("admitted")),
                        "issue_rate": issue_rate(row["declaration"], (row["cohorts"] or {}).get("admitted")),
                        "updated_at": metadata["loaded_at"],
                        "daytona": daytona(row["declaration"], DAYTONA_SAFE_SNAPSHOT_LIMIT),
                        "coverage": coverage.get(row["id"]),
                        "difficulty": (
                            table_difficulty(
                                sealed_difficulty_summary(
                                    difficulty_summary(row["declaration"], measurements.get(row["id"], [])),
                                    row["declaration"],
                                    sealed_measurements.get(row["id"]),
                                )
                            )
                            if projection == SourceProjection.TABLE
                            else sealed_difficulty_summary(
                                difficulty_summary(row["declaration"], measurements.get(row["id"], [])),
                                row["declaration"],
                                sealed_measurements.get(row["id"]),
                            )
                        ),
                        "intended_use": intended_use(row["declaration"]),
                        "backend_capabilities": (
                            backend_metadata(row["backend_capabilities"]) if withheld else row["backend_capabilities"]
                        ),
                        "partial_overlaps": overlaps.get(row["id"], []),
                        "source_fingerprint": (
                            fingerprint_metadata(row["source_fingerprint"]) if withheld else row["source_fingerprint"]
                        ),
                        "duplicate_group": groups.get(row["id"]),
                        "assessment": {
                            "status": "complete" if row["assessment_key"] else row["latest_attempt_status"],
                            "pipeline_key": row["assessment_key"],
                            "latest_attempt_key": row["latest_attempt_key"],
                            "latest_attempt_status": row["latest_attempt_status"],
                            "reason": PRIVATE_TEXT_MARKER if withheld else row["reason"],
                            "scope": row["scope"],
                            "model": row["model"],
                            **redact_cohort_summaries(row["cohorts"] or {}, withheld),
                        },
                    }
                )
            final_key = inventory_cache_key(connection, metadata["load_id"])
        payload = {
            "sources": (
                [table_source_record(row) for row in inventory] if projection == SourceProjection.TABLE else inventory
            ),
            "snapshot_id": metadata["snapshot_id"],
            "created_at": metadata["created_at"],
            "synced_at": metadata["loaded_at"],
            "daytona_snapshot_limit": DAYTONA_SAFE_SNAPSHOT_LIMIT,
        }
        # A concurrent input change must not leave a cached mixture of versions.
        # Store at most one payload per projection; HTTP headers are never cached.
        if final_key == input_key:
            inventory_responses[projection] = CachedInventory(input_key, payload)
        return payload

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

    @api.get("/difficulty-coverage")
    def difficulty_coverage_view() -> dict[str, Any]:
        """The published Difficulty coverage for the selected load, computed at rebuild."""
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            row = (
                connection.execute(
                    text("SELECT coverage FROM atlas_difficulty_coverage WHERE load_id = :load_id"),
                    {"load_id": metadata["load_id"]},
                )
                .mappings()
                .first()
            )
            if row is not None:
                return {"load_id": metadata["load_id"], "snapshot_id": metadata["snapshot_id"], **row["coverage"]}
            # A load that predates the coverage step still serves a freshly computed document.
            declarations = {
                canonical_source_id(row["id"]): row["declaration"]
                for row in connection.execute(
                    text("SELECT id, declaration FROM atlas_sources WHERE load_id = :load_id"),
                    {"load_id": metadata["load_id"]},
                ).mappings()
            }
            document = difficulty_coverage(declarations, difficulty_records(connection))
            return {
                "load_id": metadata["load_id"],
                "snapshot_id": metadata["snapshot_id"],
                **document.model_dump(mode="json"),
            }

    @api.get("/source-fingerprint")
    def source_fingerprint(source_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            source = source_record(connection, source_id)
            return {
                "source_id": canonical_source_id(source_id),
                "fingerprint": (
                    fingerprint_metadata(source["source_fingerprint"])
                    if content_is_withheld(source["declaration"])
                    else source["source_fingerprint"]
                ),
            }

    @api.get("/source-duplicates")
    def source_duplicates() -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_catalog(connection)
            parameters = {"load_id": metadata["load_id"]}
            survey = connection.execute(text(IDENTITY_SURVEY_SQL), parameters).mappings().first()
            if survey is None:
                raise HTTPException(503, "Selected catalog has no source identity survey")
            reviews = (
                connection.execute(
                    text(
                        "SELECT review FROM ("
                        + IDENTITY_REVIEWS_SQL
                        + ") selected ORDER BY review->'similarity'->>'left',review->'similarity'->>'right'"
                    ),
                    parameters,
                )
                .scalars()
                .all()
            )
            projected, overlaps = loaded_duplicate_projection(connection, metadata["load_id"])
            groups = {group["group_id"]: group for group in projected.values()}
            return {
                "survey_id": survey["survey_id"],
                **survey["metadata"],
                "reviews": [canonical_duplicate_review(row) for row in reviews],
                "groups": list(groups.values()),
                "stored_groups": [
                    canonical_duplicate_group(row)
                    for row in connection.execute(
                        text(
                            "SELECT DISTINCT duplicate_group FROM atlas_sources WHERE load_id = :load_id "
                            "AND duplicate_group IS NOT NULL"
                        ),
                        parameters,
                    ).scalars()
                ],
                "partial_overlaps": overlaps,
            }

    @api.get("/assessment")
    def assessment(
        source_id: str,
        cohort: str | None = None,
        verdict: str | None = None,
        confidence: str | None = None,
        reference: str | None = None,
        status: str | None = None,
        defect: str | None = None,
        search: str | None = None,
        limit: int = DEFAULT_PAGE_ITEMS,
        offset: int = 0,
    ) -> dict[str, Any]:
        if limit < 1 or limit > MAX_PAGE_ITEMS:
            raise HTTPException(422, f"limit must be between 1 and {MAX_PAGE_ITEMS}")
        if offset < 0:
            raise HTTPException(422, "offset must not be negative")
        with engine.connect() as connection:
            source = source_record(connection, source_id)
            withheld = content_is_withheld(source["declaration"])
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
            summaries = redact_cohort_summaries(cohort_summaries(connection, **parameters), withheld)
            filter_sql, filter_parameters = assessment_item_filters(
                cohort, verdict, confidence, reference, status, defect, search
            )
            item_parameters = {**parameters, **filter_parameters, "limit": limit, "offset": offset}
            items = [
                dict(item)
                for item in connection.execute(
                    text(
                        f"""
                SELECT i.task_id, i.cohort, i.verdict, i.status, i.confidence, i.reference_status, i.defects,
                    i.explanation, i.provenance, i.replaced_task_sha256,
                    i.repair_advisories, i.exclusion_reason,r.reviewer
                FROM atlas_items i LEFT JOIN atlas_raw_reviews r
                    ON r.load_id=i.load_id AND r.pipeline_key=i.pipeline_key AND r.task_id=i.task_id
                WHERE i.load_id = :load_id AND i.pipeline_key = :pipeline_key{filter_sql}
                ORDER BY i.cohort, i.task_id
                LIMIT :limit OFFSET :offset
            """
                    ),
                    item_parameters,
                ).mappings()
            ]
            if withheld:
                items = [
                    {
                        **redact_mapping(item, fields=frozenset({"explanation", "provenance", "exclusion_reason"})),
                        "repair_advisories": redact_advisories(item["repair_advisories"]),
                    }
                    for item in items
                ]
            (filtered_total,) = connection.execute(
                text(
                    f"""
                SELECT COUNT(*) FROM atlas_items i
                WHERE i.load_id = :load_id AND i.pipeline_key = :pipeline_key{filter_sql}
            """
                ),
                item_parameters,
            ).one()
        declaration = source["declaration"]
        display = quality_display(source["declaration"], source["curation"], source["assessment_key"])
        finalization = declaration.get("quality_result_finalization")
        report = {**row, "source_id": canonical_source_id(row["source_id"])}
        if withheld:
            # Rollout traces are task attempts for this source and the rubric is
            # reviewer prompt text: a PRIVATE_ALWAYS source serves neither.
            report.pop("traces", None)
            report["rubric"] = PRIVATE_TEXT_MARKER
            report["curation"] = None
        return {
            "source": declaration["display_name"],
            "scope": display.quality.value,
            "review_completeness": source["review_completeness"],
            "review_input": source.get("review_input"),
            "result": result(declaration, summaries.get("admitted")),
            "issue_rate": issue_rate(declaration, summaries.get("admitted")),
            "waiver": declaration.get("quality_result_waiver"),
            "finalization": finalization,
            "report": {
                **report,
                "assessment": summaries,
                "counts": whole_population_counts(summaries),
                "evidence": {
                    "items": items,
                    "total": filtered_total,
                    "limit": limit,
                    "offset": offset,
                },
            },
        }

    @api.get("/evidence")
    def evidence(source_id: str, task_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                SELECT i.task_id, i.cohort, i.verdict, i.status, i.confidence, i.reference_status,
                    i.defects, i.provenance, i.replaced_task_sha256, i.repair_advisories,
                    i.explanation, i.evidence_ids,
                    i.omissions, i.exclusion_reason, p.packet, s.declaration AS source_declaration,
                    r.record AS raw_review,b.batch_key AS review_batch,
                    i.load_id AS review_load,i.pipeline_key AS review_key
                FROM atlas_sources s JOIN atlas_current c USING (load_id)
                LEFT JOIN atlas_review_current q ON q.load_id=s.load_id AND q.source_id=s.id
                JOIN atlas_items i ON i.load_id = s.load_id
                    AND i.pipeline_key = COALESCE(q.pipeline_key,s.assessment_key)
                LEFT JOIN atlas_packets p ON p.load_id = i.load_id
                    AND p.pipeline_key = i.pipeline_key AND p.task_id = i.task_id
                LEFT JOIN atlas_raw_reviews r ON r.load_id=i.load_id
                    AND r.pipeline_key=i.pipeline_key AND r.task_id=i.task_id
                LEFT JOIN atlas_review_batch_tasks b ON b.load_id=i.load_id
                    AND b.pipeline_key=i.pipeline_key AND b.task_id=i.task_id
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
        withheld = content_is_withheld(row["source_declaration"])
        packet = (
            content_marker(row["source_declaration"])
            if withheld
            else (json.loads(row["packet"]) if row["packet"] is not None else None)
        )
        judgment = {
            key: row[key]
            for key in (
                "task_id",
                "cohort",
                "verdict",
                "status",
                "confidence",
                "reference_status",
                "defects",
                "provenance",
                "replaced_task_sha256",
                "repair_advisories",
                "explanation",
                "evidence_ids",
            )
        }
        if withheld:
            judgment = {
                **redact_mapping(judgment, fields=frozenset({"explanation", "provenance"})),
                "repair_advisories": redact_advisories(judgment["repair_advisories"]),
            }
        raw_review = (
            content_marker(row["source_declaration"])
            if withheld
            else (unpack_content(row["raw_review"]) if row["raw_review"] is not None else None)
        )
        if not withheld and raw_review is not None and row["review_batch"] is not None:
            with engine.connect() as connection:
                shared = connection.execute(
                    text(
                        "SELECT record FROM atlas_review_batches WHERE load_id=:load "
                        "AND pipeline_key=:key AND batch_key=:batch"
                    ),
                    {"load": row["review_load"], "key": row["review_key"], "batch": row["review_batch"]},
                ).scalar_one_or_none()
            if shared is None:
                raise HTTPException(503, "Original reviewer batch evidence is unavailable")
            if "payload_chunks" in shared:
                payload = []
                with engine.connect() as connection:
                    for index in range(shared["payload_chunks"]):
                        part = connection.execute(
                            text(
                                "SELECT payload FROM atlas_review_batch_chunks WHERE load_id=:load "
                                "AND pipeline_key=:key AND batch_key=:batch AND chunk_index=:chunk"
                            ),
                            {
                                "load": row["review_load"],
                                "key": row["review_key"],
                                "batch": row["review_batch"],
                                "chunk": index,
                            },
                        ).scalar_one_or_none()
                        if part is None:
                            raise HTTPException(503, "Original reviewer batch chunk is unavailable")
                        payload.append(part)
                shared = {**shared, "payload": "".join(payload)}
            raw_review = {
                **raw_review,
                **task_request_evidence(unpack_content(shared)).get(task_id, {"input": [], "output": []}),
            }
        return {
            "judgment": judgment,
            "packet": packet,
            "raw_review": raw_review,
            "omissions": [] if withheld else row["omissions"],
            "exclusion_reason": PRIVATE_TEXT_MARKER if withheld and row["exclusion_reason"] else row["exclusion_reason"],
        }

    @api.get("/source")
    def source_evidence(source_id: str) -> dict[str, Any]:
        """Every quality-evidence pointer the catalog holds for one source."""
        with engine.connect() as connection:
            source = source_record(connection, source_id)
            withheld = content_is_withheld(source["declaration"])
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
                summaries = redact_cohort_summaries(cohort_summaries(connection, **parameters), withheld)
                if withheld:
                    assessment.pop("traces", None)
                    assessment["rubric"] = PRIVATE_TEXT_MARKER
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
            full_reviews = full_review_records(connection, canonical_source_id(source_id))
            groups, overlaps = loaded_duplicate_projection(connection, source["load_id"])
            load = loaded_catalog(connection)
            coverage = source_coverage(connection, {canonical_source_id(source_id): source["declaration"]}).get(
                canonical_source_id(source_id)
            )
            difficulty = sealed_difficulty_summary(
                difficulty_summary(
                    source["declaration"],
                    difficulty_records(connection, source_id).get(canonical_source_id(source_id), []),
                ),
                source["declaration"],
                sealed_difficulty_records(connection, source["load_id"]).get(canonical_source_id(source_id)),
            )
        explanation = quality_explanation(
            source["declaration"],
            quality_display(source["declaration"], source["curation"], source["assessment_key"]).quality,
            full_reviews.get(canonical_source_id(source_id), []),
        )
        declaration = consumer_projection(license_projection(source["declaration"]))
        declaration = restriction_projection(declaration)
        if withheld:
            declaration = withheld_declaration_projection(declaration)
        review = summaries.get("admitted")
        display = quality_display(source["declaration"], source["curation"], source["assessment_key"])
        return {
            "declaration": declaration,
            "coverage": coverage,
            "difficulty": difficulty,
            "intended_use": intended_use(declaration),
            "method_tags": source["method_tags"],
            "method_variants": source["method_variants"],
            "contamination_status": contamination(declaration),
            "filtration": filtration(source["declaration"], counts.get(declaration["id"])),
            "curation": None if withheld else source["curation"],
            "diagnostics": [] if withheld else source["diagnostics"],
            "backend_capabilities": backend_metadata(capabilities) if withheld else capabilities,
            "duplicate_group": groups.get(declaration["id"]),
            "partial_overlaps": overlaps.get(declaration["id"], []),
            "latest_attempt": {
                "key": source["latest_attempt_key"],
                "status": source["latest_attempt_status"],
                "reason": PRIVATE_TEXT_MARKER if withheld else source["reason"],
            },
            "assessment": assessment,
            "cohorts": summaries,
            "archived_reviews": reviews,
            "quality": display.quality,
            "quality_review_status": display.review_status,
            "quality_explanation": explanation,
            "review_scope": display.quality.value,
            "review_coverage": review_coverage(declaration, source["review_completeness"]),
            "review_completeness": source["review_completeness"],
            "result_provisional": (source["review_completeness"] or {}).get("provisional"),
            "result": result(declaration, review),
            "issue_rate": issue_rate(declaration, review),
            "daytona": daytona(declaration, DAYTONA_SAFE_SNAPSHOT_LIMIT),
            "daytona_snapshot_limit": DAYTONA_SAFE_SNAPSHOT_LIMIT,
            "updated_at": load["loaded_at"],
            "catalog": {"root": load["catalog_root"], "snapshot_id": load["snapshot_id"]},
        }

    @api.get("/archive")
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
            declaration = current_source_declaration(connection, record["source_id"])
        withheld = content_is_withheld(declaration)
        return {
            "id": record["id"],
            "source_id": canonical_source_id(record["source_id"]),
            "data_restriction": declaration_restriction(declaration).value,
            "collection": content_marker(declaration) if withheld else record["collection"],
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
            owner = connection.execute(
                text("SELECT source_id FROM catalog_reviews WHERE id = :id"), {"id": review_id}
            ).scalar_one_or_none()
            if record is None:
                raise HTTPException(404, "Archived artifact not found")
            declaration = current_source_declaration(connection, owner or "")
            if content_is_withheld(declaration):
                return Response(
                    content_marker(declaration)["marker"],
                    media_type="text/plain",
                    headers={
                        "ETag": record["sha256"],
                        "X-Data-Restriction": declaration_restriction(declaration).value,
                        "X-Content-Withheld": "true",
                    },
                )
        return Response(record["content"], media_type="text/plain", headers={"ETag": record["sha256"]})

    return api
