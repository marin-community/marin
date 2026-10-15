# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream a complete scored curriculum release into the checked-in Atlas app."""

import json
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import urlparse

import click
import pyarrow.parquet as pq
from finestore.rl.curriculum import AlignmentRelease
from marina.client import DEFAULT_MARINA_URL, marina_request
from rigging.filesystem.storage_path import StoragePath

from infra.marina.applets.rl_data_catalog.load import Query, batches

TABLE_COLUMNS = {
    "sources": "source_id TEXT, declaration JSONB, family TEXT, alignment JSONB",
    "nodes": "subject_id TEXT, node_id TEXT, parent_id TEXT, kind TEXT, name TEXT, description TEXT",
    "ancestors": "subject_id TEXT, node_id TEXT, ancestor_id TEXT",
    "tasks": "source_id TEXT, task_id TEXT, content_hash TEXT, description_status TEXT, description TEXT, detail JSONB",
}


def ancestor_rows(nodes: list[dict]) -> Iterator[dict]:
    by_id = {(row["subject_id"], row["node_id"]): row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Curriculum contains duplicate node IDs")
    for row in nodes:
        current = row
        visited = set()
        while True:
            identifier = current["node_id"]
            if identifier in visited:
                raise ValueError("Curriculum hierarchy contains a cycle")
            visited.add(identifier)
            yield {"subject_id": row["subject_id"], "node_id": row["node_id"], "ancestor_id": identifier}
            if current["parent_id"] is None:
                break
            current = by_id[(row["subject_id"], current["parent_id"])]


def release_tasks(paths: list[str]) -> Iterator[dict]:
    for path in paths:
        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield {
                        **{
                            key: row[key]
                            for key in ("source_id", "task_id", "content_hash", "description_status", "description")
                        },
                        "detail": row,
                    }


def load_curriculum(release_uri: str, query: Query) -> dict[str, Any]:
    """Stage bounded batches, validate lineage and counts, then select atomically."""
    started = time.monotonic()
    release = AlignmentRelease.model_validate_json(StoragePath(release_uri).read_text())
    current = query(
        "SELECT l.release_id, l.release_uri FROM curriculum_current c JOIN curriculum_loads l USING (load_id) "
        "WHERE c.name = 'latest'",
        {},
    )["rows"]
    if current and current[0] == {"release_id": release.release_id, "release_uri": release_uri}:
        return {"changed": False, "release_id": release.release_id}
    load_id = uuid.uuid4().hex
    requests = 1

    def execute(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        nonlocal requests
        requests += 1
        return query(sql, parameters)

    def insert(table: str, rows: Iterable[dict]) -> int:
        columns = TABLE_COLUMNS[table]
        names = ", ".join(column.strip().split()[0] for column in columns.split(","))
        count = 0
        for body in batches(rows):
            result = execute(
                f"INSERT INTO curriculum_{table} (load_id, {names}) SELECT :load_id, {names} "
                f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns})",
                {"load_id": load_id, "rows": body},
            )
            count += result["row_count"]
        return count

    for table in ("matches", "tasks", "ancestors", "nodes", "sources", "loads"):
        execute(
            f"DELETE FROM curriculum_{table} WHERE load_id NOT IN "
            "(SELECT load_id FROM curriculum_current WHERE load_id IS NOT NULL UNION "
            "SELECT previous_load_id FROM curriculum_current WHERE previous_load_id IS NOT NULL)",
            {},
        )
    execute(
        "INSERT INTO curriculum_loads (load_id, release_id, release_uri, manifest) "
        "VALUES (:load_id, :release_id, :release_uri, CAST(:manifest AS JSONB))",
        {
            "load_id": load_id,
            "release_id": release.release_id,
            "release_uri": release_uri,
            "manifest": release.model_dump_json(),
        },
    )
    selected = {row.source_id: row.model_dump(mode="json") for row in release.sources}
    if len(selected) != len(release.sources) or selected.keys() - {row.id for row in release.inventory}:
        raise ValueError("Release source inventory is inconsistent")
    nodes = [row.model_dump(mode="json") for row in release.nodes]
    counts = {
        "sources": insert(
            "sources",
            (
                {
                    "source_id": row.id,
                    "declaration": row.model_dump(mode="json"),
                    "family": row.family or row.origin,
                    "alignment": selected.get(row.id),
                }
                for row in release.inventory
            ),
        ),
        "nodes": insert("nodes", nodes),
        "ancestors": insert("ancestors", ancestor_rows(nodes)),
        "tasks": insert("tasks", release_tasks(release.paths)),
    }
    execute(
        """
        INSERT INTO curriculum_matches
            (load_id, source_id, task_id, subject_id, capability_id, facet, membership_similarity, capability_similarity)
        SELECT t.load_id, t.source_id, t.task_id, m->>'subject_id', m->>'capability_id', m->>'facet',
            (m->>'membership_similarity')::double precision, (m->'candidates'->0->>'similarity')::double precision
        FROM curriculum_tasks t CROSS JOIN LATERAL jsonb_array_elements(t.detail->'matches'->'subjects') m
        WHERE t.load_id = :load_id
    """,
        {"load_id": load_id},
    )
    invalid = execute(
        """
        SELECT t.source_id, t.task_id FROM curriculum_tasks t JOIN curriculum_sources s USING (load_id, source_id)
        WHERE t.load_id = :load_id AND (
            s.alignment IS NULL OR t.detail->>'source_key' IS DISTINCT FROM s.alignment->>'source_key'
            OR t.detail->>'population' IS DISTINCT FROM s.alignment->>'population'
            OR t.detail->>'intended_use' IS DISTINCT FROM s.alignment->>'intended_use'
            OR (t.description_status = 'ready') IS DISTINCT FROM EXISTS (
                SELECT 1 FROM curriculum_matches m WHERE m.load_id=t.load_id AND m.source_id=t.source_id
                    AND m.task_id=t.task_id
            )
        ) LIMIT 1
    """,
        {"load_id": load_id},
    )["rows"]
    unknown = execute(
        """
        SELECT m.task_id FROM curriculum_matches m
        LEFT JOIN curriculum_nodes s ON s.load_id=m.load_id AND s.subject_id=m.subject_id
            AND s.node_id='subject:' || m.subject_id AND s.kind='subject'
        LEFT JOIN curriculum_nodes c ON c.load_id=m.load_id AND c.subject_id=m.subject_id
            AND c.node_id=m.capability_id AND c.kind='capability'
        WHERE m.load_id=:load_id AND (s.node_id IS NULL OR c.node_id IS NULL) LIMIT 1
    """,
        {"load_id": load_id},
    )["rows"]
    if invalid or unknown:
        raise ValueError(
            f"Loaded task lineage, readiness or curriculum references are inconsistent: {invalid or unknown}"
        )
    actual = execute(
        """
        SELECT s.source_id, COUNT(t.task_id) AS tasks,
            COUNT(DISTINCT t.content_hash) AS unique_content,
            COUNT(t.task_id) FILTER (WHERE t.description_status='ready') AS ready,
            COUNT(t.task_id) FILTER (WHERE t.description_status='insufficient_context') AS insufficient
        FROM curriculum_sources s LEFT JOIN curriculum_tasks t USING (load_id, source_id)
        WHERE s.load_id=:load_id AND s.alignment IS NOT NULL GROUP BY s.source_id
    """,
        {"load_id": load_id},
    )["rows"]
    for row in actual:
        expected = selected[row["source_id"]]
        if (row["tasks"], row["unique_content"], row["ready"], row["insufficient"]) != (
            expected["task_count"],
            expected["unique_content_count"],
            expected["ready_count"],
            expected["insufficient_context_count"],
        ):
            raise ValueError(f"Loaded task counts differ from release: {row['source_id']}")
    totals = execute(
        "SELECT COUNT(*) AS tasks, COUNT(DISTINCT content_hash) AS unique_content, "
        "COUNT(DISTINCT description) FILTER (WHERE description_status='ready') AS embedding_texts "
        "FROM curriculum_tasks WHERE load_id=:load_id",
        {"load_id": load_id},
    )["rows"][0]
    if (totals["tasks"], totals["unique_content"], totals["embedding_texts"]) != (
        release.task_count,
        release.unique_content_count,
        release.unique_embedding_count,
    ):
        raise ValueError("Loaded corpus counts differ from release")
    execute(
        "UPDATE curriculum_current SET previous_load_id=load_id, load_id=:load_id WHERE name='latest'",
        {"load_id": load_id},
    )
    return {
        "changed": True,
        "release_id": release.release_id,
        "counts": counts,
        "sql_requests": requests,
        "seconds": time.monotonic() - started,
    }


@click.command(help=__doc__)
@click.option("--release", "release_uri", required=True, help="Immutable release.json; run in its storage region.")
@click.option("--applet-id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def main(release_uri: str, applet_id: str, service_url: str) -> None:
    details = marina_request(service_url, "GET", f"/api/marina/applets/{applet_id}")
    assert isinstance(details, dict)
    parsed = urlparse(details["authenticated_url"])
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else service_url.rstrip("/")

    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        result = marina_request(
            origin, "POST", f"/a/{applet_id}/query", json_body={"sql": sql, "parameters": parameters}
        )
        assert isinstance(result, dict)
        return result

    click.echo(json.dumps(load_curriculum(release_uri, query)))


if __name__ == "__main__":
    main()
