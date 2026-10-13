# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a published curation catalog into Atlas; run beside its object storage."""

import json
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlparse

import click
from finestore.rl.curation import CurationCatalog, validate_items
from marina.client import DEFAULT_MARINA_URL, marina_request

MAX_BATCH_BYTES = 1_000_000
PACKET_READ_BATCH = 512
TABLE_COLUMNS = {
    "sources": (
        "id TEXT, declaration JSONB, assessment_key TEXT, latest_attempt_key TEXT, "
        "latest_attempt_status TEXT, reason TEXT, curation JSONB, diagnostics JSONB"
    ),
    "assessments": (
        "pipeline_key TEXT, source_id TEXT, scope TEXT, model JSONB, rubric TEXT, "
        "curation JSONB, curation_uri TEXT, traces JSONB"
    ),
    "populations": "pipeline_key TEXT, cohort TEXT, population BIGINT, sample_count BIGINT",
    "items": (
        "pipeline_key TEXT, task_id TEXT, cohort TEXT, status TEXT, verdict TEXT, explanation TEXT, "
        "evidence_ids JSONB, packet_uri TEXT, has_task BOOLEAN, omissions JSONB, exclusion_reason TEXT"
    ),
    "packets": "pipeline_key TEXT, task_id TEXT, packet TEXT",
}
Query = Callable[[str, dict[str, Any]], dict[str, Any]]


def batches(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    """Bound SQL request bodies by encoded bytes, including packet escaping."""
    batch: list[str] = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, allow_nan=False, separators=(",", ":"))
        length = len(encoded.encode()) + 1
        if length + 2 > MAX_BATCH_BYTES:
            raise ValueError("One catalog row exceeds the SQL batch budget")
        if size + length > MAX_BATCH_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def load_catalog(catalog: CurationCatalog, query: Query) -> dict[str, Any]:
    """Stage, check and select one snapshot using a single operator."""
    started = time.monotonic()
    snapshot = catalog.current()
    current = query(
        """
        SELECT l.catalog_root, l.snapshot_id FROM atlas_current c
        JOIN atlas_loads l USING (load_id) WHERE c.name = 'latest'
    """,
        {},
    )["rows"]
    if current and current[0] == {"catalog_root": catalog.root, "snapshot_id": snapshot.snapshot_id}:
        return {"changed": False, "snapshot_id": snapshot.snapshot_id}
    load_id = uuid.uuid4().hex
    requests = 1

    def execute(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        nonlocal requests
        requests += 1
        return query(sql, parameters)

    def insert(table: str, rows: Iterable[dict[str, Any]]) -> int:
        columns = TABLE_COLUMNS[table]
        names = ", ".join(column.strip().split()[0] for column in columns.split(","))
        count = 0
        for body in batches(rows):
            result = execute(
                f"""
                INSERT INTO atlas_{table} (load_id, {names})
                SELECT :load_id, {names} FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns})
            """,
                {"load_id": load_id, "rows": body},
            )
            count += result["row_count"]
        return count

    # Discard abandoned and obsolete staging data before loading; retain the selected snapshot.
    for table in [*TABLE_COLUMNS, "loads"]:
        execute(
            f"DELETE FROM atlas_{table} WHERE load_id NOT IN "
            "(SELECT load_id FROM atlas_current WHERE load_id IS NOT NULL)",
            {},
        )
    execute(
        """
        INSERT INTO atlas_loads (load_id, catalog_root, snapshot_id, created_at)
        VALUES (:load_id, :catalog_root, :snapshot_id, :created_at)
    """,
        {
            "load_id": load_id,
            "catalog_root": catalog.root,
            "snapshot_id": snapshot.snapshot_id,
            "created_at": snapshot.created_at,
        },
    )
    sources = list(catalog.sources(snapshot.snapshot_id))
    if len(sources) != snapshot.source_count:
        raise ValueError("Catalog inventory does not cover its declared source count")
    metadata: dict[str, list[dict[str, Any]]] = {
        "sources": [row.model_dump(mode="json", exclude={"snapshot_id"}) for row in sources],
        "assessments": [],
        "populations": [],
        "items": [],
    }
    all_items = []
    for source in sources:
        key = source.assessment_key
        if key is None:
            continue
        assessment = catalog.assessment(key)
        if assessment.source_id != source.id:
            raise ValueError("Selected assessment belongs to another source")
        populations, items = catalog.populations(key), list(catalog.items(key))
        validate_items(populations, items)
        metadata["assessments"].append(assessment.model_dump(mode="json"))
        metadata["populations"].extend(row.model_dump(mode="json") for row in populations)
        metadata["items"].extend(row.model_dump(mode="json") for row in items)
        all_items.extend(items)
    counts = {table: insert(table, rows) for table, rows in metadata.items()}

    def packet_rows() -> Iterable[dict[str, Any]]:
        for offset in range(0, len(all_items), PACKET_READ_BATCH):
            for item, packet in catalog.packets(all_items[offset : offset + PACKET_READ_BATCH]):
                if ("task" in packet) != item.has_task:
                    raise ValueError(f"Packet evidence disagrees with catalog item: {item.task_id}")
                yield {
                    "pipeline_key": item.pipeline_key,
                    "task_id": item.task_id,
                    "packet": json.dumps(packet, allow_nan=False, separators=(",", ":")),
                }

    counts["packets"] = insert("packets", packet_rows())
    actual = execute(
        " UNION ALL ".join(
            f"SELECT '{table}' AS name, COUNT(*) AS count FROM atlas_{table} WHERE load_id = :load_id"
            for table in TABLE_COLUMNS
        ),
        {"load_id": load_id},
    )["rows"]
    if {row["name"]: row["count"] for row in actual} != counts:
        raise ValueError("SQL staging counts differ from the catalog")
    invalid = execute(
        """
        SELECT
            (SELECT COUNT(*) FROM atlas_cohort_summary WHERE load_id = :load_id
                AND reviewed <> sample_count) AS incomplete,
            (SELECT COUNT(*) FROM atlas_items i LEFT JOIN atlas_packets p
                USING (load_id, pipeline_key, task_id)
                WHERE i.load_id = :load_id AND p.task_id IS NULL) AS missing_packets,
            (SELECT COUNT(*) FROM atlas_items i LEFT JOIN atlas_populations p
                USING (load_id, pipeline_key, cohort)
                WHERE i.load_id = :load_id AND p.cohort IS NULL) AS missing_populations
    """,
        {"load_id": load_id},
    )["rows"][0]
    if any(invalid.values()):
        raise ValueError(f"SQL assessment coverage failed: {invalid}")
    packet_sizes = execute(
        """
        SELECT COALESCE(SUM(octet_length(packet)), 0) AS packet_bytes,
            MAX(octet_length(packet)) AS max_packet_bytes,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY octet_length(packet)) AS p95_packet_bytes
        FROM atlas_packets WHERE load_id = :load_id
    """,
        {"load_id": load_id},
    )["rows"][0]
    execute("UPDATE atlas_current SET load_id = :load_id WHERE name = 'latest'", {"load_id": load_id})
    return {
        "changed": True,
        "load_id": load_id,
        "snapshot_id": snapshot.snapshot_id,
        "counts": counts,
        **packet_sizes,
        "sql_requests": requests,
        "seconds": time.monotonic() - started,
    }


@click.command(help=__doc__)
@click.option("--catalog-root", required=True, help="Published FineStore catalog; run in its storage region.")
@click.option("--applet-id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def main(catalog_root: str, applet_id: str, service_url: str) -> None:
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

    click.echo(json.dumps(load_catalog(CurationCatalog(catalog_root), query)))


if __name__ == "__main__":
    main()
