# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve precomputed curriculum counts and a bounded set of review examples."""

import base64
import binascii
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from finestore.rl.status_taxonomy import DataRestriction
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .catalog_text import display_catalog_value
from .content_access import content_is_withheld, content_marker
from .restrictions import (
    PRIVATE_TEXT_MARKER,
    declaration_restriction,
)
from .source_areas import SOURCE_AREA_VERSION
from .source_identity import canonical_source_id, original_source_id

TOTAL_METRICS = ("denominator", "assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped")
INVARIANT_SOURCE_BATCH = 16


@dataclass(frozen=True)
class CoverageQuery:
    intended_use: str
    source_id: str
    release: str


def coverage_query(
    intended_use: Literal["all", "train", "eval"] = "all",
    source_id: str = "",
    release: str = "",
) -> CoverageQuery:
    return CoverageQuery(intended_use, canonical_source_id(source_id), release)


RELEASE_SQL = """
        SELECT l.load_id,l.release_id,l.release_uri,l.loaded_at,
            jsonb_build_object(
                'created_at',l.manifest->'created_at',
                'population',l.manifest->'population',
                'sample_size',l.manifest->'sample_size',
                'curriculum_key',l.manifest->'curriculum_key',
                'coverage',l.manifest->'coverage',
                'nodes',l.manifest->'nodes',
                'lineage',l.manifest->'lineage',
                'confidence',l.manifest->'confidence',
                'role_projection',l.manifest->'role_projection'
            ) AS manifest
        FROM curriculum_loads l JOIN curriculum_current c
            ON l.load_id=c.load_id OR l.load_id=c.previous_load_id
        WHERE c.name='latest' AND ((:release='' AND l.load_id=c.load_id) OR l.release_id=:release)
        ORDER BY l.loaded_at DESC LIMIT 1
    """


def loaded_release(connection: Connection, release_id: str) -> dict[str, Any]:
    row = (
        connection.execute(
            text(RELEASE_SQL),
            {"release": release_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(503 if not release_id else 404, "No matching curriculum release has been loaded")
    return dict(row)


def source_coverage(connection: Connection, current: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read leading subjects only when the compact release matches the current input."""
    rows = connection.execute(
        text(
            """
        SELECT s.source_id, jsonb_build_object('dataset_id',s.declaration->'dataset_id',
            'dataset_revision',s.declaration->'dataset_revision','pipeline',s.declaration->'pipeline') AS declaration,
            (s.alignment->>'task_count')::bigint AS example_count,
            s.alignment->>'intended_use' AS intended_use,
            s.alignment->>'basis' AS basis,
            a.area
        FROM curriculum_sources s JOIN curriculum_current c USING (load_id)
        LEFT JOIN curriculum_source_areas a ON a.load_id=s.load_id AND a.source_id=s.source_id AND a.version=:version
        WHERE c.name='latest' AND s.alignment IS NOT NULL
        """
        ),
        {"version": SOURCE_AREA_VERSION},
    ).mappings()
    return {
        canonical_source_id(row["source_id"]): (
            {
                "example_count": row["example_count"],
                "intended_use": row["intended_use"],
                "basis": row["basis"],
                "top_area": row["area"],
            }
            if canonical_source_id(row["source_id"]) in current
            and all(
                row["declaration"].get(key) == current[canonical_source_id(row["source_id"])].get(key)
                for key in ("dataset_id", "dataset_revision", "pipeline")
            )
            else {
                # The alignment was computed at an older pin: the source's area is not yet
                # computed for the current declaration, not genuinely unknown.
                "unavailable_reason": (
                    "Not yet computed for this pin (last computed at " f"{row['declaration'].get('dataset_revision')})"
                ),
                "last_computed_dataset_revision": row["declaration"].get("dataset_revision"),
            }
        )
        for row in rows
    }


SOURCES_SQL = """
WITH selected_sources AS (
    SELECT s.*
    FROM curriculum_sources s WHERE s.load_id=:load_id
        AND (:source_id='' OR s.source_id IN (:source_id, :original_source_id))
        AND (:intended_use='all' OR s.alignment IS NULL OR s.alignment->>'intended_use'=:intended_use)
 )
"""

SAMPLE_SQL = (
    SOURCES_SQL
    + """
, selected_tasks AS (
    SELECT t.load_id,t.source_id,t.task_id,t.content_hash,t.description_status
    FROM curriculum_tasks t JOIN selected_sources s USING (load_id, source_id)
    WHERE (:node_id='' OR EXISTS (
        SELECT 1 FROM curriculum_matches area_match
        WHERE area_match.load_id=:load_id AND area_match.source_id=t.source_id
            AND area_match.task_id=t.task_id AND area_match.subject_id=:subject_id
            AND area_match.membership_similarity >= CAST(:subject_similarity AS DOUBLE PRECISION)))
), members AS (
    SELECT t.source_id, t.task_id, t.content_hash, m.subject_id, m.capability_id,
        m.capability_similarity
    FROM selected_tasks t JOIN curriculum_matches m ON m.load_id=:load_id
        AND m.source_id=t.source_id AND m.task_id=t.task_id
    WHERE m.membership_similarity >= CAST(:subject_similarity AS DOUBLE PRECISION)
), placements AS (
    SELECT source_id, task_id, content_hash, subject_id, 'subject:' || subject_id AS node_id FROM members
    UNION ALL
    -- The ancestor arm reads the match table, not the members CTE: a CTE carries no
    -- statistics, and its one-row estimate made the planner rescan it per ancestor row
    -- (R40, over 60 s for the largest subjects). Every match read binds the load id so the
    -- primary-key and area indexes serve one load, never every retained load.
    SELECT m.source_id, m.task_id, t.content_hash, m.subject_id, a.ancestor_id AS node_id
    FROM curriculum_matches m
    JOIN curriculum_ancestors a ON a.load_id=m.load_id
        AND a.subject_id=m.subject_id AND a.node_id=m.capability_id
    JOIN selected_tasks t ON t.source_id=m.source_id AND t.task_id=m.task_id
    WHERE m.load_id=:load_id
        AND m.membership_similarity >= CAST(:subject_similarity AS DOUBLE PRECISION)
        AND m.capability_similarity >= CAST(:capability_similarity AS DOUBLE PRECISION)
        AND a.ancestor_id <> 'subject:' || m.subject_id
), states AS (
    SELECT t.source_id, t.task_id, t.content_hash, CASE
        WHEN t.description_status='insufficient_context' THEN 'insufficient_context'
        WHEN t.description_status='skipped' THEN 'skipped'
        WHEN COUNT(m.task_id) FILTER (
            WHERE m.membership_similarity >= CAST(:subject_similarity AS DOUBLE PRECISION)
            AND m.capability_similarity >= CAST(:capability_similarity AS DOUBLE PRECISION))>0 THEN 'assigned'
        WHEN COUNT(m.task_id) FILTER (
            WHERE m.membership_similarity >= CAST(:subject_similarity AS DOUBLE PRECISION))>0 THEN 'subject_only'
        ELSE 'out_of_scope' END AS outcome
    FROM selected_tasks t LEFT JOIN curriculum_matches m ON m.load_id=:load_id
        AND m.source_id=t.source_id AND m.task_id=t.task_id
    GROUP BY t.source_id, t.task_id, t.content_hash, t.description_status
)

"""
)


def display_hierarchy(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinguish taxonomy wrappers from their subjects without changing archived IDs."""
    subjects = {node["subject_id"]: node["name"] for node in nodes if node["kind"] == "subject"}

    def normalized(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.casefold().replace("&", "and"))

    result = []
    for node in nodes:
        subject = subjects[node["subject_id"]]
        name = node["name"]
        if node["kind"] != "subject" and normalized(name) == normalized(subject):
            name = (
                f"Capability groups for {subject}"
                if node["kind"] == "group"
                else f"General {subject} (no specific capability)"
            )
        result.append({**node, "name": name})
    return result


def child_ids(metadata: dict[str, Any], subject_id: str, node_id: str) -> list[str]:
    nodes = metadata["manifest"]["nodes"]
    if not any(node["subject_id"] == subject_id and node["node_id"] == node_id for node in nodes):
        raise HTTPException(404, "Unknown curriculum area")
    return [node["node_id"] for node in nodes if node["subject_id"] == subject_id and node["parent_id"] == node_id]


AREA_SQL = (
    SOURCES_SQL
    + """
                SELECT s.source_id,s.declaration->>'display_name' AS name,
                    s.alignment->>'intended_use' AS intended_use,
                    CAST(SUM(CASE WHEN c.node_id=:node_id THEN c.instances
                        WHEN :scope='unlisted' THEN -c.instances ELSE 0 END) AS BIGINT) AS count
                FROM selected_sources s JOIN curriculum_statistics c USING(load_id,source_id)
                WHERE c.metric='placement' AND c.subject_id=:subject_id
                    AND (c.node_id=:node_id OR (:scope='unlisted' AND c.node_id=ANY(CAST(:children AS TEXT[]))))
                GROUP BY s.source_id,s.declaration->>'display_name',s.alignment->>'intended_use'
                HAVING SUM(CASE WHEN c.node_id=:node_id THEN c.instances
                    WHEN :scope='unlisted' THEN -c.instances ELSE 0 END)>0
                ORDER BY count DESC,s.source_id
                """
)


def area_parameters(
    metadata: dict[str, Any], filters: CoverageQuery, subject_id: str, node_id: str, scope: str
) -> dict[str, Any]:
    return {
        **asdict(filters),
        "original_source_id": original_source_id(filters.source_id),
        "load_id": metadata["load_id"],
        "subject_id": subject_id,
        "node_id": node_id,
        "children": child_ids(metadata, subject_id, node_id),
        "scope": scope,
    }


def area_response(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = [{**row, "source_id": canonical_source_id(row["source_id"])} for row in rows]
    return {"release_id": metadata["release_id"], "count": sum(row["count"] for row in sources), "sources": sources}


TASKS_SQL = (
    SAMPLE_SQL
    + """
                , eligible AS (
                SELECT t.load_id,t.source_id,t.task_id,t.content_hash,t.description_status,
                    CASE WHEN :scope='unlisted' THEN 'subject_only' ELSE s.outcome END AS outcome,
                    ROW_NUMBER() OVER (PARTITION BY t.source_id ORDER BY t.task_id) AS sample_rank
                FROM selected_tasks t JOIN states s USING (source_id,task_id,content_hash)
                WHERE (:node_id='' OR EXISTS (SELECT 1 FROM placements p
                    WHERE p.source_id=t.source_id AND p.task_id=t.task_id
                        AND p.subject_id=:subject_id AND p.node_id=:node_id))
                    AND (:outcome='all' OR s.outcome=:outcome)
                    AND (:scope='all' OR NOT EXISTS (SELECT 1 FROM placements p
                        WHERE p.source_id=t.source_id AND p.task_id=t.task_id
                            AND p.subject_id=:subject_id AND p.node_id=ANY(CAST(:children AS TEXT[]))))
                ), selected_page AS (
                SELECT * FROM eligible
                WHERE (sample_rank,source_id,task_id)>(:after_rank,:after_source,:after_task)
                ORDER BY sample_rank,source_id,task_id LIMIT :limit
                )
                SELECT p.source_id,p.task_id,p.content_hash,LEFT(d.detail->>'task_preview',600) AS task_preview,
                    jsonb_build_object('id',s.declaration->'id') ||
                        CASE WHEN s.declaration ? 'data_restriction'
                            THEN jsonb_build_object('data_restriction',s.declaration->'data_restriction')
                            ELSE '{}'::jsonb END AS source_declaration,
                    (SELECT jsonb_agg(jsonb_build_object(
                        'subject_id',m->>'subject_id','membership_similarity',m->'membership_similarity',
                        'candidates',jsonb_build_array(jsonb_build_object(
                            'capability_id',m->'candidates'->0->>'capability_id',
                            'similarity',m->'candidates'->0->'similarity'))))
                        FROM jsonb_array_elements(d.detail->'matches'->'subjects') m) AS matches,
                    p.description_status,p.outcome,p.sample_rank
                FROM selected_page p JOIN curriculum_tasks d USING (load_id,source_id,task_id)
                JOIN curriculum_sources s ON s.load_id=p.load_id AND s.source_id=p.source_id
                ORDER BY p.sample_rank,p.source_id,p.task_id
            """
)


def task_parameters(
    metadata: dict[str, Any],
    filters: CoverageQuery,
    subject_id: str,
    node_id: str,
    scope: str,
    outcome: str,
    cursor: str,
    limit: int,
    sample_pool: bool = False,
) -> dict[str, Any]:
    after_rank, after_source, after_task = 0, "", ""
    if cursor:
        try:
            after_rank, after_source, after_task = json.loads(base64.urlsafe_b64decode(cursor).decode())
        except (ValueError, TypeError, UnicodeError, binascii.Error) as error:
            raise HTTPException(400, "Invalid task cursor") from error
        if not isinstance(after_rank, int) or not isinstance(after_source, str) or not isinstance(after_task, str):
            raise HTTPException(400, "Invalid task cursor")
    children = child_ids(metadata, subject_id, node_id) if node_id else []
    if scope == "unlisted" and not node_id:
        raise HTTPException(400, "An unlisted selection requires a curriculum area")
    params = {
        **asdict(filters),
        "original_source_id": original_source_id(filters.source_id),
        "load_id": metadata["load_id"],
        "subject_id": subject_id,
        "node_id": node_id,
        "scope": scope,
        "children": children,
        "outcome": outcome,
        "after_rank": after_rank,
        "after_source": after_source,
        "after_task": after_task,
        "limit": limit + 1,
        "subject_similarity": 1 - metadata["manifest"]["coverage"]["subject_distance"],
        "capability_similarity": 1 - metadata["manifest"]["coverage"]["capability_distance"],
    }
    if sample_pool:
        confidence = metadata["manifest"].get("confidence")
        if not confidence:
            raise HTTPException(400, "This historical population has no confidence sample pool")
        params.update(
            subject_similarity=min(level["subject_min_similarity"] for level in confidence["levels"]),
            capability_similarity=min(level["capability_min_similarity"] for level in confidence["levels"]),
            node_id="subject:" + subject_id if subject_id else "",
            scope="all",
            outcome="all",
        )
    return params


def task_response(
    metadata: dict[str, Any], rows: list[dict[str, Any]], limit: int, restrictions: dict[str, DataRestriction]
) -> dict[str, Any]:
    next_cursor = ""
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = base64.urlsafe_b64encode(
            json.dumps([last["sample_rank"], last["source_id"], last["task_id"]]).encode()
        ).decode()
    tasks = []
    for row in rows[:limit]:
        source = canonical_source_id(row["source_id"])
        restriction = restrictions.get(source)
        withheld = (
            content_is_withheld(row["source_declaration"])
            if row.get("source_declaration") is not None
            else restriction is DataRestriction.PRIVATE_ALWAYS
        )
        served = {key: value for key, value in row.items() if key != "source_declaration"}
        if restriction is not None:
            served["data_restriction"] = restriction.value
        if withheld:
            served["task_preview"] = (
                content_marker(row["source_declaration"])["marker"]
                if row.get("source_declaration") is not None
                else PRIVATE_TEXT_MARKER
            )
        else:
            served = display_catalog_value(served)
        tasks.append({**served, "source_id": source})
    return {
        "release_id": metadata["release_id"],
        "tasks": tasks,
        "next_cursor": next_cursor,
    }


def current_content_policies(connection: Connection, source_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return current policies that withhold historical sample content."""
    ids = sorted({canonical_source_id(source_id) for source_id in source_ids})
    if not ids:
        return {}
    policies = {
        canonical_source_id(row["id"]): row["declaration"]
        for row in connection.execute(
            text(
                "SELECT s.id,jsonb_build_object('id',s.id,'dataset_id',s.declaration->'dataset_id',"
                "'license_receipt',s.declaration->'license_receipt',"
                "'exclusion_claims',s.declaration->'exclusion_claims') || "
                "CASE WHEN s.declaration ? 'data_restriction' "
                "THEN jsonb_build_object('data_restriction',s.declaration->'data_restriction') "
                "ELSE '{}'::jsonb END AS declaration "
                "FROM atlas_sources s JOIN atlas_current c USING(load_id) WHERE c.name='latest' "
                "AND s.id IN (SELECT value FROM jsonb_array_elements_text(CAST(:ids AS JSONB)))"
            ),
            {"ids": json.dumps(ids)},
        ).mappings()
    }
    return {source: declaration for source, declaration in policies.items() if content_is_withheld(declaration)}


def source_restrictions(rows: list[dict[str, Any]]) -> dict[str, DataRestriction]:
    """The effective tier of every source appearing in served curriculum rows.

    The declaration carries the tier, so the versioned override registry stays
    authoritative even when a release predates it or a stored row was tampered with.
    """
    resolved = {}
    for row in rows:
        declaration = row.get("source_declaration")
        if declaration is None:
            continue
        resolved[canonical_source_id(row["source_id"])] = declaration_restriction(declaration)
    return resolved


CONFIDENCE_SQL = """
SELECT metric,subject_id,node_id,instances FROM curriculum_confidence
WHERE load_id=:load_id AND scope_id=:scope_id
AND (:intended_use='all' OR scope_id LIKE '@%' OR EXISTS (
    SELECT 1 FROM curriculum_sources s WHERE s.load_id=:load_id AND s.source_id=:scope_id
    AND s.alignment->>'intended_use'=:intended_use))
ORDER BY metric,subject_id,node_id
"""

CONFIDENCE_AREA_SQL = (
    SOURCES_SQL
    + """
, vectors AS (
SELECT s.source_id,s.declaration->>'display_name' AS name,
    s.alignment->>'intended_use' AS intended_use,i,
    SUM(CASE WHEN c.node_id=:node_id THEN c.instances[i] ELSE -c.instances[i] END)::BIGINT AS count
FROM selected_sources s JOIN curriculum_confidence c
    ON c.load_id=s.load_id AND c.scope_id=s.source_id
CROSS JOIN generate_series(1,:levels) i
WHERE c.metric='placement' AND c.subject_id=:subject_id
    AND (c.node_id=:node_id OR (:scope='unlisted' AND c.node_id=ANY(CAST(:children AS TEXT[]))))
GROUP BY s.source_id,s.declaration,s.alignment,i
)
SELECT source_id,name,intended_use,array_agg(count ORDER BY i) AS instances
FROM vectors GROUP BY source_id,name,intended_use ORDER BY source_id
"""
)


def confidence_parameters(metadata: dict[str, Any], filters: CoverageQuery) -> dict[str, str]:
    return {
        "load_id": metadata["load_id"],
        "scope_id": filters.source_id or "@" + filters.intended_use,
        "intended_use": filters.intended_use,
    }


def confidence_response(metadata: dict[str, Any], rows: list[dict]) -> dict[str, Any] | None:
    confidence = metadata["manifest"].get("confidence")
    return {**confidence, "statistics": rows} if confidence else None


def connection_rows(connection: Connection, sql: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(text(sql), parameters).mappings()]


def coverage_document(
    query: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
    metadata: dict[str, Any],
    filters: CoverageQuery,
    subject_id: str,
) -> dict[str, Any]:
    params = {
        **asdict(filters),
        "original_source_id": original_source_id(filters.source_id),
        "load_id": metadata["load_id"],
        "subject_id": subject_id,
    }
    area_counts = query(
        SOURCES_SQL
        + """
        SELECT c.subject_id,c.node_id,CAST(SUM(c.instances) AS BIGINT) AS count
        FROM curriculum_statistics c JOIN selected_sources s USING (load_id,source_id)
        WHERE c.load_id=:load_id AND c.metric='placement' GROUP BY c.subject_id,c.node_id
    """,
        params,
    )
    totals = dict.fromkeys(TOTAL_METRICS, 0)
    for row in query(
        SOURCES_SQL
        + """
        SELECT c.metric,CAST(SUM(c.instances) AS BIGINT) AS count
        FROM curriculum_statistics c JOIN selected_sources s USING (load_id,source_id)
        WHERE c.load_id=:load_id AND c.subject_id='' AND c.node_id='' GROUP BY c.metric
    """,
        params,
    ):
        totals[row["metric"]] = row["count"]
    confidence = (
        confidence_response(metadata, query(CONFIDENCE_SQL, confidence_parameters(metadata, filters)))
        if metadata["manifest"].get("confidence")
        else None
    )
    scope_summary = None
    if subject_id:
        scope_summary = query(
            SOURCES_SQL
            + """
            SELECT CAST(COALESCE(SUM(c.instances) FILTER (WHERE c.metric='placement'),0) AS BIGINT) AS subject_members,
                CAST(COALESCE(SUM(c.instances) FILTER (WHERE c.metric='capability'),0) AS BIGINT) AS capability_matches
            FROM curriculum_statistics c JOIN selected_sources s USING (load_id,source_id)
            WHERE c.load_id=:load_id AND c.subject_id=:subject_id AND c.node_id='subject:' || :subject_id
                AND c.metric IN ('placement','capability')
        """,
            params,
        )[0]
    sources = query(
        (
            "SELECT source_id,declaration->>'display_name' AS name,family,alignment FROM "
            "curriculum_sources WHERE load_id=:load_id ORDER BY source_id"
        ),
        params,
    )
    sources = [
        {
            **row,
            "source_id": canonical_source_id(row["source_id"]),
            "alignment": (
                {**row["alignment"], "source_id": canonical_source_id(row["alignment"]["source_id"])}
                if row["alignment"] is not None
                else None
            ),
        }
        for row in sources
    ]
    manifest = metadata["manifest"]
    return {
        "release_id": metadata["release_id"],
        "created_at": manifest["created_at"],
        "population": manifest["population"],
        "sample_size": manifest["sample_size"],
        "curriculum_key": manifest["curriculum_key"],
        "filters": asdict(filters),
        "totals": totals,
        "confidence": confidence,
        "thresholds": {key: manifest["coverage"][key] for key in ("subject_distance", "capability_distance")},
        "examples": {
            key: manifest["coverage"][key] for key in ("sample_task_count", "source_sample_size", "area_sample_size")
        },
        "area_counts": area_counts,
        "sources": sources,
        "hierarchy": display_hierarchy(manifest["nodes"]),
        "lineage": manifest["lineage"],
        "scope_summary": scope_summary,
        "role_projection": manifest.get("role_projection"),
    }


def invariant_document(
    query: Callable[[str, dict[str, Any]], list[dict[str, Any]]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Expose count evidence once, without fetching task text or repeating it for each domain."""
    load_id = metadata["load_id"]
    sources = query(
        "SELECT source_id,alignment FROM curriculum_sources WHERE load_id=:load_id ORDER BY source_id",
        {"load_id": load_id},
    )
    vectors = {row["source_id"]: {"statistics": [], "confidence_statistics": []} for row in sources}
    # Keep the SQL gateway's response bounded when this projection is checked before publication.
    for offset in range(0, len(sources), INVARIANT_SOURCE_BATCH):
        params = {
            "load_id": load_id,
            "sources": json.dumps([row["source_id"] for row in sources[offset : offset + INVARIANT_SOURCE_BATCH]]),
        }
        for table, column, field in (
            ("curriculum_statistics", "source_id", "statistics"),
            ("curriculum_confidence", "scope_id", "confidence_statistics"),
        ):
            for row in query(
                f"SELECT {column} AS source_id,jsonb_agg(jsonb_build_array(metric,subject_id,node_id,instances) "
                f"ORDER BY metric,subject_id,node_id) AS statistics FROM {table} WHERE load_id=:load_id "
                f"AND {column} IN (SELECT jsonb_array_elements_text(CAST(:sources AS JSONB))) GROUP BY {column}",
                params,
            ):
                vectors[row["source_id"]][field] = row["statistics"]
    return {
        "release_id": metadata["release_id"],
        "taxonomy": metadata["manifest"]["nodes"],
        "views": {
            use: coverage_document(query, metadata, CoverageQuery(use, "", metadata["release_id"]), "")
            for use in ("all", "train", "eval")
        },
        "sources": [
            {
                "source_id": canonical_source_id(row["source_id"]),
                "alignment": row["alignment"],
                **vectors[row["source_id"]],
            }
            for row in sources
        ],
    }


def register_curriculum_routes(api: FastAPI, engine: Engine) -> None:
    @api.get("/curriculum/invariants")
    def invariants(release: str = "") -> dict[str, Any]:
        with engine.connect() as connection:
            return invariant_document(partial(connection_rows, connection), loaded_release(connection, release))

    @api.get("/curriculum/releases")
    def releases() -> dict[str, Any]:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT l.release_id, l.release_uri, l.manifest->>'created_at' AS created_at,
                        l.load_id=c.load_id AS selected
                    FROM curriculum_loads l JOIN curriculum_current c
                        ON l.load_id=c.load_id OR l.load_id=c.previous_load_id
                    WHERE c.name='latest' ORDER BY selected DESC
                    """
                )
            ).mappings()
            return {"releases": [dict(row) for row in rows]}

    @api.get("/curriculum")
    def coverage(filters: Annotated[CoverageQuery, Depends(coverage_query)], subject_id: str = "") -> dict[str, Any]:
        with engine.connect() as connection:
            return coverage_document(
                partial(connection_rows, connection), loaded_release(connection, filters.release), filters, subject_id
            )

    @api.get("/curriculum/area")
    def area(
        filters: Annotated[CoverageQuery, Depends(coverage_query)],
        subject_id: str,
        node_id: str,
        scope: Literal["all", "unlisted"] = "all",
    ) -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_release(connection, filters.release)
            params = area_parameters(metadata, filters, subject_id, node_id, scope)
            rows = [dict(row) for row in connection.execute(text(AREA_SQL), params).mappings()]
            result = area_response(metadata, rows)
            if metadata["manifest"].get("confidence"):
                params["levels"] = len(metadata["manifest"]["confidence"]["levels"])
                result["confidence_sources"] = [
                    dict(row) for row in connection.execute(text(CONFIDENCE_AREA_SQL), params).mappings()
                ]
        return result

    @api.get("/curriculum/tasks")
    def tasks(
        filters: Annotated[CoverageQuery, Depends(coverage_query)],
        subject_id: str = "",
        node_id: str = "",
        scope: Literal["all", "unlisted"] = "all",
        outcome: Literal["all", "assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped"] = "all",
        cursor: str = "",
        limit: int = Query(50, ge=1, le=1000),
        sample_pool: bool = False,
    ) -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_release(connection, filters.release)
            params = task_parameters(metadata, filters, subject_id, node_id, scope, outcome, cursor, limit, sample_pool)
            rows = [dict(row) for row in connection.execute(text(TASKS_SQL), params).mappings()]
            policies = current_content_policies(connection, [row["source_id"] for row in rows])
            rows = [
                {
                    **row,
                    "source_declaration": policies.get(canonical_source_id(row["source_id"]), row["source_declaration"]),
                }
                for row in rows
            ]
        return task_response(metadata, rows, limit, source_restrictions(rows))

    @api.get("/curriculum/task")
    def task(source_id: str, task_id: str, release: str = "") -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_release(connection, release)
            detail = (
                connection.execute(
                    text(
                        """
                SELECT t.detail, s.alignment->>'intended_use' AS intended_use,
                    s.declaration || jsonb_build_object('id',s.source_id) AS source_declaration
                FROM curriculum_tasks t JOIN curriculum_sources s USING(load_id, source_id)
                WHERE t.load_id=:load_id AND t.source_id IN (:source_id, :original_source_id) AND t.task_id=:task_id
            """
                    ),
                    {
                        "load_id": metadata["load_id"],
                        "source_id": canonical_source_id(source_id),
                        "original_source_id": original_source_id(source_id),
                        "task_id": task_id,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if detail is not None:
                policies = current_content_policies(connection, [source_id])
                detail = {
                    **detail,
                    "source_declaration": policies.get(canonical_source_id(source_id), detail["source_declaration"]),
                }
        if detail is None:
            raise HTTPException(404, "Task is outside the loaded alignment population")
        if content_is_withheld(detail["source_declaration"]):
            return {"release_id": metadata["release_id"], "task": content_marker(detail["source_declaration"])}
        return {
            "release_id": metadata["release_id"],
            "task": {
                **display_catalog_value(detail["detail"]),
                "source_id": canonical_source_id(detail["detail"]["source_id"]),
                "recorded_intended_use": detail["detail"]["intended_use"],
                "intended_use": detail["intended_use"],
            },
        }
