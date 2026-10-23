# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a published curation catalog into Atlas."""

import json
import time
import uuid
from collections.abc import Iterable
from typing import Any

import click
from finestore.rl.curation import CurationCatalog, JudgmentStatus, validate_items
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3

MAX_BATCH_BYTES = 1_000_000
PACKET_READ_BATCH = 512
DEFAULT_CATALOG_ROOT = "s3://marin-us-east-02a/marin/rl-data/catalog/production"
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


def batches(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    """Bound SQL request bodies by encoded bytes, including packet escaping."""
    batch: list[str] = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # The SQL API carries the row array as a JSON string, adding another escape layer.
        length = len(json.dumps(encoded).encode()) + 1
        if length + 2 > MAX_BATCH_BYTES:
            raise ValueError("One catalog row exceeds the SQL batch budget")
        if size + length > MAX_BATCH_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def insert_rows(query: AppletQuery, table: str, columns: str, load_id: str, rows: Iterable[dict[str, Any]]) -> int:
    """Insert byte-bounded batches and return the stored row count."""
    names = ", ".join(column.strip().split()[0] for column in columns.split(","))
    count = 0
    for body in batches(rows):
        result = query(
            f"INSERT INTO {table} (load_id, {names}) SELECT :load_id, {names} "
            f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns})",
            {"load_id": load_id, "rows": body},
        )
        count += result["row_count"]
    return count


def load_catalog(catalog: CurationCatalog, query: AppletQuery) -> dict[str, Any]:
    """Stage and validate a snapshot, then select it if the serving pointer is unchanged."""
    started = time.monotonic()
    snapshot = catalog.current()
    current = query(
        """
        SELECT c.load_id, l.catalog_root, l.snapshot_id FROM atlas_current c
        JOIN atlas_loads l USING (load_id) WHERE c.name = 'latest'
    """,
        {},
    )["rows"]
    if current and (current[0]["catalog_root"], current[0]["snapshot_id"]) == (catalog.root, snapshot.snapshot_id):
        return {"changed": False, "snapshot_id": snapshot.snapshot_id}
    expected_load_id = current[0]["load_id"] if current else None
    obsolete_load_ids = json.dumps([row["load_id"] for row in query("SELECT load_id FROM atlas_loads", {})["rows"]])
    load_id = uuid.uuid4().hex
    requests = 2

    def execute(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        nonlocal requests
        requests += 1
        return query(sql, parameters)

    def insert(table: str, rows: Iterable[dict[str, Any]]) -> int:
        return insert_rows(execute, f"atlas_{table}", TABLE_COLUMNS[table], load_id, rows)

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
        if any(item.status != JudgmentStatus.REVIEWED for item in items):
            raise ValueError(f"Selected assessment has unavailable judgments: {key}")
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
                    "packet": json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                }

    counts["packets"] = insert("packets", packet_rows())
    packet_sizes = execute(
        """
        SELECT COALESCE(SUM(octet_length(packet)), 0) AS packet_bytes,
            MAX(octet_length(packet)) AS max_packet_bytes,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY octet_length(packet)) AS p95_packet_bytes
        FROM atlas_packets WHERE load_id = :load_id
    """,
        {"load_id": load_id},
    )["rows"][0]
    if CurationCatalog(catalog.root).current().snapshot_id != snapshot.snapshot_id:
        raise RuntimeError("Catalog selection changed during import; retry the load")
    selected = execute(
        "UPDATE atlas_current SET load_id = :load_id WHERE name = 'latest' "
        "AND load_id IS NOT DISTINCT FROM :expected_load_id",
        {"load_id": load_id, "expected_load_id": expected_load_id},
    )
    if selected["row_count"] != 1:
        raise RuntimeError("Atlas selection changed during import; retry the load")
    # Only collect loads observed before staging. A loader starting after our cutover may still succeed.
    for table in [*TABLE_COLUMNS, "loads"]:
        execute(
            f"DELETE FROM atlas_{table} WHERE load_id IN "
            "(SELECT jsonb_array_elements_text(CAST(:obsolete_load_ids AS JSONB))) "
            "AND load_id NOT IN (SELECT load_id FROM atlas_current WHERE load_id IS NOT NULL)",
            {"obsolete_load_ids": obsolete_load_ids},
        )
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
@click.option(
    "--catalog-root", default=DEFAULT_CATALOG_ROOT, show_default=True, help="Published FineStore catalog to import."
)
@click.option("--applet-id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def main(catalog_root: str, applet_id: str, service_url: str) -> None:
    configure_coreweave_s3()
    click.echo(json.dumps(load_catalog(CurationCatalog(catalog_root), applet_query(service_url, applet_id))))


if __name__ == "__main__":
    main()
