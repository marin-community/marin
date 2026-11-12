# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a published curation catalog into Atlas."""

import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
from finestore.rl.curation import MAX_UNAVAILABLE_JUDGMENT_SHARE, CurationCatalog, JudgmentStatus, validate_items
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3
from taskcompendium.backend_requirements import BackendSurvey, backend_schema_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.source_fingerprints import SourceIdentitySurvey

MAX_BATCH_BYTES = 1_000_000
PACKET_READ_BATCH = 512
DEFAULT_CATALOG_ROOT = "s3://marin-us-east-02a/marin/rl-data/catalog/production"
LOADER_VERSION = 2
"""Bump when loaded rows change shape; a load from an older loader is restaged even for the same snapshot."""
TABLE_COLUMNS = {
    "sources": (
        "id TEXT, declaration JSONB, assessment_key TEXT, latest_attempt_key TEXT, "
        "latest_attempt_status TEXT, reason TEXT, curation JSONB, diagnostics JSONB, updated_at TIMESTAMPTZ, "
        "source_fingerprint JSONB, duplicate_group JSONB"
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
    "backend_surveys": "survey_id TEXT, survey JSONB",
    "backend_sources": "source_id TEXT, capabilities JSONB",
    "backend_cells": "backend TEXT, requirement TEXT, cell JSONB",
    "identity_surveys": "survey_id TEXT, metadata JSONB",
    "duplicate_reviews": "left_source TEXT, right_source TEXT, review JSONB",
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


def load_catalog(
    catalog: CurationCatalog, query: AppletQuery, backend_map: BackendSurvey, identities: SourceIdentitySurvey
) -> dict[str, Any]:
    """Stage and validate a snapshot, then select it if the serving pointer is unchanged."""
    started = time.monotonic()
    snapshot = catalog.current()
    if (backend_map.catalog_root, backend_map.snapshot_id) != (catalog.root, snapshot.snapshot_id):
        raise ValueError("Backend map belongs to another catalog snapshot; survey the selected catalog")
    if backend_map.schema_sha256 != backend_schema_sha256():
        raise ValueError("Backend map uses another requirement schema; rerun the survey")
    if (identities.catalog_root, identities.snapshot_id) != (catalog.root, snapshot.snapshot_id):
        raise ValueError("Source identity map belongs to another catalog snapshot")
    identity_id = canonical_sha256(identities.model_dump(mode="json"))
    survey_id = canonical_sha256(backend_map.model_dump(mode="json"))
    current = query(
        """
        SELECT c.load_id, l.catalog_root, l.snapshot_id, l.loader_version,
            b.survey_id, i.survey_id AS identity_id FROM atlas_current c
        JOIN atlas_loads l USING (load_id)
        LEFT JOIN atlas_backend_surveys b USING (load_id)
        LEFT JOIN atlas_identity_surveys i USING (load_id) WHERE c.name = 'latest'
    """,
        {},
    )["rows"]
    if current and (
        current[0]["catalog_root"],
        current[0]["snapshot_id"],
        current[0]["loader_version"],
        current[0]["survey_id"],
        current[0]["identity_id"],
    ) == (
        catalog.root,
        snapshot.snapshot_id,
        LOADER_VERSION,
        survey_id,
        identity_id,
    ):
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
        INSERT INTO atlas_loads (load_id, catalog_root, snapshot_id, created_at, loader_version)
        VALUES (:load_id, :catalog_root, :snapshot_id, :created_at, :loader_version)
    """,
        {
            "load_id": load_id,
            "catalog_root": catalog.root,
            "snapshot_id": snapshot.snapshot_id,
            "created_at": snapshot.created_at,
            "loader_version": LOADER_VERSION,
        },
    )
    sources = list(catalog.sources(snapshot.snapshot_id))
    if len(sources) != snapshot.source_count:
        raise ValueError("Catalog inventory does not cover its declared source count")

    starts = catalog.source_starts()

    if {source.id for source in sources} != {source.source_id for source in backend_map.sources}:
        raise ValueError("Backend map does not cover the selected catalog inventory")

    if {source.id for source in sources} != set(identities.fingerprints):
        raise ValueError("Source identity map does not cover the selected catalog inventory")
    source_declarations = {source.id: source.declaration for source in sources}
    for source in backend_map.sources:
        evidence = source_declarations[source.source_id].grading_environment
        expected = evidence.model_dump(mode="json") if evidence is not None else None
        if source.grading_environment != expected:
            raise ValueError(f"Backend map grading evidence differs from the catalog: {source.source_id}")
    groups = {member.source_id: group.model_dump(mode="json") for group in identities.groups for member in group.members}
    metadata: dict[str, list[dict[str, Any]]] = {
        "sources": [
            {
                **row.model_dump(mode="json", exclude={"snapshot_id"}),
                "updated_at": starts[row.id].isoformat() if row.id in starts else None,
                "source_fingerprint": (
                    identities.fingerprints[row.id].model_dump(mode="json")
                    if identities.fingerprints[row.id] is not None
                    else None
                ),
                "duplicate_group": groups.get(row.id),
            }
            for row in sources
        ],
        "assessments": [],
        "populations": [],
        "items": [],
        "identity_surveys": [
            {
                "survey_id": identity_id,
                "metadata": identities.model_dump(mode="json", exclude={"fingerprints", "reviews", "groups"}),
            }
        ],
        "duplicate_reviews": [
            {
                "left_source": row.similarity.left,
                "right_source": row.similarity.right,
                "review": row.model_dump(mode="json"),
            }
            for row in identities.reviews
        ],
        "backend_surveys": [
            {"survey_id": survey_id, "survey": backend_map.model_dump(mode="json", exclude={"sources", "matrix"})}
        ],
        "backend_cells": [
            {"backend": cell.backend, "requirement": cell.requirement.value, "cell": cell.model_dump(mode="json")}
            for cell in backend_map.matrix
        ],
        "backend_sources": [
            {"source_id": source.source_id, "capabilities": source.model_dump(mode="json")}
            for source in backend_map.sources
        ],
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
        unavailable = sum(1 for item in items if item.status != JudgmentStatus.REVIEWED)
        if items and unavailable / len(items) > MAX_UNAVAILABLE_JUDGMENT_SHARE:
            raise ValueError(f"Selected assessment exceeds the unavailable-judgment threshold: {key}")
        metadata["assessments"].append(assessment.model_dump(mode="json"))
        metadata["populations"].extend(row.model_dump(mode="json") for row in populations)
        metadata["items"].extend(row.model_dump(mode="json") for row in items)
        all_items.extend(items)
    counts = {table: insert(table, rows) for table, rows in metadata.items()}
    if (
        counts["backend_sources"] != len(sources)
        or counts["backend_surveys"] != 1
        or counts["backend_cells"] != len(backend_map.matrix)
    ):
        raise ValueError("Backend survey import is incomplete")

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
@click.option("--app-id", "applet_id", required=True)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--backend-map", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--source-identities", type=click.Path(exists=True, path_type=Path), required=True)
def main(catalog_root: str, applet_id: str, service_url: str, backend_map: Path, source_identities: Path) -> None:
    configure_coreweave_s3()
    survey = BackendSurvey.model_validate_json(backend_map.read_text())
    identities = SourceIdentitySurvey.model_validate_json(source_identities.read_text())
    click.echo(
        json.dumps(load_catalog(CurationCatalog(catalog_root), applet_query(service_url, applet_id), survey, identities))
    )


if __name__ == "__main__":
    main()
