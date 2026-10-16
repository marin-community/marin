# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve precomputed curriculum counts and a bounded set of review examples."""

import base64
import binascii
import json
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

TOTAL_METRICS = ("denominator", "assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped")


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
    return CoverageQuery(intended_use, source_id, release)


def loaded_release(connection: Connection, release_id: str) -> dict[str, Any]:
    row = (
        connection.execute(
            text(
                """
        SELECT l.* FROM curriculum_loads l JOIN curriculum_current c
            ON l.load_id=c.load_id OR l.load_id=c.previous_load_id
        WHERE c.name='latest' AND ((:release='' AND l.load_id=c.load_id) OR l.release_id=:release)
        ORDER BY l.loaded_at DESC LIMIT 1
    """
            ),
            {"release": release_id},
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(503 if not release_id else 404, "No matching curriculum release has been loaded")
    return dict(row)


SOURCES_SQL = """
WITH selected_sources AS (
    SELECT s.*
    FROM curriculum_sources s WHERE s.load_id=:load_id
        AND (:source_id='' OR s.source_id=:source_id)
        AND (:intended_use='all' OR s.alignment IS NULL OR s.alignment->>'intended_use'=:intended_use)
 )
"""

SAMPLE_SQL = (
    SOURCES_SQL
    + """
, selected_tasks AS (
    SELECT t.load_id,t.source_id,t.task_id,t.content_hash,t.description_status
    FROM curriculum_tasks t JOIN selected_sources s USING (load_id, source_id)
), members AS (
    SELECT t.source_id, t.task_id, t.content_hash, m.subject_id, m.capability_id,
        m.capability_similarity
    FROM selected_tasks t JOIN curriculum_matches m USING (load_id, source_id, task_id)
    WHERE m.membership_similarity >= CAST(:subject_similarity AS DOUBLE PRECISION)
), placements AS (
    SELECT source_id, task_id, content_hash, subject_id, 'subject:' || subject_id AS node_id FROM members
    UNION ALL
    SELECT m.source_id, m.task_id, m.content_hash, m.subject_id, a.ancestor_id AS node_id
    FROM members m JOIN curriculum_ancestors a ON a.load_id=:load_id
        AND a.subject_id=m.subject_id AND a.node_id=m.capability_id
    WHERE m.capability_similarity >= CAST(:capability_similarity AS DOUBLE PRECISION)
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
    FROM selected_tasks t LEFT JOIN curriculum_matches m USING (load_id, source_id, task_id)
    GROUP BY t.source_id, t.task_id, t.content_hash, t.description_status
)
"""
)


def register_curriculum_routes(api: FastAPI, engine: Engine) -> None:
    @api.get("/curriculum")
    def coverage(
        filters: Annotated[CoverageQuery, Depends(coverage_query)],
        subject_id: str = "",
    ) -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_release(connection, filters.release)
            params = {
                **asdict(filters),
                "load_id": metadata["load_id"],
                "subject_id": subject_id,
            }
            area_counts = [
                dict(row)
                for row in connection.execute(
                    text(
                        SOURCES_SQL
                        + """
                SELECT c.subject_id,c.node_id,CAST(SUM(c.instances) AS BIGINT) AS count
                FROM curriculum_statistics c JOIN selected_sources s USING (load_id,source_id)
                WHERE c.load_id=:load_id AND c.metric='placement'
                GROUP BY c.subject_id,c.node_id
            """
                    ),
                    params,
                ).mappings()
            ]
            totals = dict.fromkeys(TOTAL_METRICS, 0)
            for row in connection.execute(
                text(
                    SOURCES_SQL
                    + """
                SELECT c.metric,CAST(SUM(c.instances) AS BIGINT) AS count
                FROM curriculum_statistics c JOIN selected_sources s USING (load_id,source_id)
                WHERE c.load_id=:load_id AND c.subject_id='' AND c.node_id=''
                GROUP BY c.metric
            """
                ),
                params,
            ).mappings():
                totals[row["metric"]] = row["count"]
            scope_summary = None
            if subject_id:
                scope_summary = dict(
                    connection.execute(
                        text(
                            SOURCES_SQL
                            + """
                    SELECT CAST(COALESCE(SUM(c.instances) FILTER (WHERE c.metric='placement'),0)
                            AS BIGINT) AS subject_members,
                        CAST(COALESCE(SUM(c.instances) FILTER (WHERE c.metric='capability'),0)
                            AS BIGINT) AS capability_matches
                    FROM curriculum_statistics c JOIN selected_sources s USING (load_id,source_id)
                    WHERE c.load_id=:load_id AND c.subject_id=:subject_id
                        AND c.node_id='subject:' || :subject_id AND c.metric IN ('placement','capability')
                """
                        ),
                        params,
                    )
                    .mappings()
                    .one()
                )
            sources = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                SELECT source_id,declaration->>'display_name' AS name,family,alignment
                FROM curriculum_sources WHERE load_id=:load_id ORDER BY source_id
            """
                    ),
                    params,
                ).mappings()
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
            "thresholds": {key: manifest["coverage"][key] for key in ("subject_distance", "capability_distance")},
            "examples": {
                key: manifest["coverage"][key] for key in ("sample_task_count", "source_sample_size", "area_sample_size")
            },
            "area_counts": area_counts,
            "sources": sources,
            "hierarchy": manifest["nodes"],
            "lineage": manifest["lineage"],
            "scope_summary": scope_summary,
        }

    @api.get("/curriculum/tasks")
    def tasks(
        filters: Annotated[CoverageQuery, Depends(coverage_query)],
        subject_id: str = "",
        node_id: str = "",
        outcome: Literal["all", "assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped"] = "all",
        cursor: str = "",
        limit: int = Query(50, ge=1, le=100),
    ) -> dict[str, Any]:
        after_rank, after_source, after_task = 0, "", ""
        if cursor:
            try:
                after_rank, after_source, after_task = json.loads(base64.urlsafe_b64decode(cursor).decode())
            except (ValueError, TypeError, UnicodeError, binascii.Error) as error:
                raise HTTPException(400, "Invalid task cursor") from error
            if not isinstance(after_rank, int) or not isinstance(after_source, str) or not isinstance(after_task, str):
                raise HTTPException(400, "Invalid task cursor")
        with engine.connect() as connection:
            metadata = loaded_release(connection, filters.release)
            params = {
                **asdict(filters),
                "load_id": metadata["load_id"],
                "subject_id": subject_id,
                "node_id": node_id,
                "outcome": outcome,
                "after_rank": after_rank,
                "after_source": after_source,
                "after_task": after_task,
                "limit": limit + 1,
                "subject_similarity": 1 - metadata["manifest"]["coverage"]["subject_distance"],
                "capability_similarity": 1 - metadata["manifest"]["coverage"]["capability_distance"],
            }
            rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        SAMPLE_SQL
                        + """
                , eligible AS (
                SELECT t.source_id,t.task_id,t.content_hash,d.detail->>'task_preview' AS task_preview,
                    t.description_status,s.outcome,
                    ROW_NUMBER() OVER (PARTITION BY t.source_id ORDER BY t.task_id) AS sample_rank
                FROM selected_tasks t JOIN states s USING (source_id,task_id,content_hash)
                JOIN curriculum_tasks d USING (load_id,source_id,task_id)
                WHERE (:node_id='' OR EXISTS (SELECT 1 FROM placements p
                    WHERE p.source_id=t.source_id AND p.task_id=t.task_id
                        AND p.subject_id=:subject_id AND p.node_id=:node_id))
                    AND (:outcome='all' OR s.outcome=:outcome)
                )
                SELECT * FROM eligible
                WHERE (sample_rank,source_id,task_id)>(:after_rank,:after_source,:after_task)
                ORDER BY sample_rank,source_id,task_id LIMIT :limit
            """
                    ),
                    params,
                ).mappings()
            ]
        next_cursor = ""
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = base64.urlsafe_b64encode(
                json.dumps([last["sample_rank"], last["source_id"], last["task_id"]]).encode()
            ).decode()
        return {"release_id": metadata["release_id"], "tasks": rows[:limit], "next_cursor": next_cursor}

    @api.get("/curriculum/task")
    def task(source_id: str, task_id: str, release: str = "") -> dict[str, Any]:
        with engine.connect() as connection:
            metadata = loaded_release(connection, release)
            detail = connection.execute(
                text(
                    """
                SELECT detail FROM curriculum_tasks WHERE load_id=:load_id AND source_id=:source_id AND task_id=:task_id
            """
                ),
                {"load_id": metadata["load_id"], "source_id": source_id, "task_id": task_id},
            ).scalar_one_or_none()
        if detail is None:
            raise HTTPException(404, "Task is outside the loaded alignment population")
        return {"release_id": metadata["release_id"], "task": detail}
