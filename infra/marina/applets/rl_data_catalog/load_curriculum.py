# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream a complete scored curriculum release into the checked-in Atlas app."""

import hashlib
import json
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

import click
import pyarrow.parquet as pq
from finestore.rl.curriculum import AlignmentRelease
from marina.client import DEFAULT_MARINA_URL
from rigging.filesystem.storage_path import StoragePath

from infra.marina.applets.rl_data_catalog.load import Query, applet_query, batches, insert_rows

DELETE_BATCH_SIZE = 10_000
TABLE_COLUMNS = {
    "sources": "source_id TEXT, declaration JSONB, family TEXT, alignment JSONB",
    "nodes": "subject_id TEXT, node_id TEXT, parent_id TEXT, kind TEXT, name TEXT, description TEXT",
    "ancestors": "subject_id TEXT, node_id TEXT, ancestor_id TEXT",
    "tasks": "source_id TEXT, task_id TEXT, content_hash TEXT, description_status TEXT, description TEXT, detail JSONB",
    "matches": (
        "source_id TEXT, task_id TEXT, subject_id TEXT, capability_id TEXT, facet TEXT, "
        "membership_similarity DOUBLE PRECISION, capability_similarity DOUBLE PRECISION"
    ),
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
        return insert_rows(execute, f"curriculum_{table}", TABLE_COLUMNS[table], load_id, rows)

    stale = execute(
        "SELECT load_id FROM curriculum_loads WHERE load_id NOT IN "
        "(SELECT load_id FROM curriculum_current WHERE load_id IS NOT NULL UNION "
        "SELECT previous_load_id FROM curriculum_current WHERE previous_load_id IS NOT NULL)",
        {},
    )["rows"]
    for row in stale:
        for table in ("matches", "tasks", "ancestors", "nodes", "sources", "loads"):
            while execute(
                f"DELETE FROM curriculum_{table} WHERE ctid IN "
                f"(SELECT ctid FROM curriculum_{table} WHERE load_id=:load_id LIMIT :limit)",
                {"load_id": row["load_id"], "limit": DELETE_BATCH_SIZE},
            )["row_count"]:
                pass
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
    }
    node_kinds = {(row["subject_id"], row["node_id"]): row["kind"] for row in nodes}
    source_counts: dict[str, Counter] = defaultdict(Counter)
    source_content: dict[str, set[bytes]] = defaultdict(set)
    content_hashes: set[bytes] = set()
    embedding_texts: set[bytes] = set()

    def checked_tasks() -> Iterator[dict]:
        for row in release_tasks(release.paths):
            detail = row["detail"]
            expected = selected.get(row["source_id"])
            subjects = (detail["matches"] or {}).get("subjects", [])
            if (
                expected is None
                or any(detail[key] != expected[key] for key in ("source_key", "population", "intended_use"))
                or (row["description_status"] == "ready") != bool(subjects)
                or row["description_status"] not in ("ready", "insufficient_context", "skipped")
            ):
                raise ValueError(f"Loaded task lineage or readiness is inconsistent: {row['task_id']}")
            for match in subjects:
                subject_id = match["subject_id"]
                if (
                    node_kinds.get((subject_id, "subject:" + subject_id)) != "subject"
                    or node_kinds.get((subject_id, match["capability_id"])) != "capability"
                ):
                    raise ValueError(f"Loaded task curriculum references are inconsistent: {row['task_id']}")
            source = row["source_id"]
            source_counts[source]["tasks"] += 1
            source_counts[source][row["description_status"]] += 1
            content = hashlib.sha256(row["content_hash"].encode()).digest()
            source_content[source].add(content)
            content_hashes.add(content)
            if row["description_status"] == "ready":
                embedding_texts.add(hashlib.sha256(row["description"].encode()).digest())
            yield row

    counts["tasks"] = 0
    counts["matches"] = 0
    for body in batches(checked_tasks()):
        rows = json.loads(body)
        counts["tasks"] += insert("tasks", rows)
        counts["matches"] += insert(
            "matches",
            (
                {
                    "source_id": row["source_id"],
                    "task_id": row["task_id"],
                    "subject_id": match["subject_id"],
                    "capability_id": match["capability_id"],
                    "facet": match["facet"],
                    "membership_similarity": match["membership_similarity"],
                    "capability_similarity": match["candidates"][0]["similarity"],
                }
                for row in rows
                for match in (row["detail"]["matches"] or {}).get("subjects", [])
            ),
        )
    for source, expected in selected.items():
        actual = source_counts[source]
        if (actual["tasks"], len(source_content[source]), actual["ready"], actual["insufficient_context"]) != (
            expected["task_count"],
            expected["unique_content_count"],
            expected["ready_count"],
            expected["insufficient_context_count"],
        ):
            raise ValueError(f"Loaded task counts differ from release: {source}")
    if (counts["tasks"], len(content_hashes), len(embedding_texts)) != (
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
    click.echo(json.dumps(load_curriculum(release_uri, applet_query(service_url, applet_id))))


if __name__ == "__main__":
    main()
