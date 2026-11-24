# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load immutable typed difficulty records without changing the selected source catalog."""

import json
from pathlib import Path
from typing import Any

import click
from finestore.reader import ReadView
from finestore.rl.difficulty import MEASUREMENTS, DifficultyMeasurement, DifficultyObservation
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3

from infra.marina.apps.rl_data_catalog.load import batches
from infra.marina.apps.rl_data_catalog.server.difficulty_coverage import DifficultyCoverage, difficulty_coverage
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

# The applet transport rejects single responses over 5 MiB; bulk reads page
# under that bound with these batch sizes.
DIFFICULTY_READ_BATCH = 50
DECLARATION_READ_BATCH = 100
MEASUREMENT_READBACK_BATCH = DIFFICULTY_READ_BATCH


def read_measurements(view: ReadView) -> list[DifficultyMeasurement]:
    """Read typed measurements without FineStore row metadata."""
    return [
        DifficultyMeasurement.model_validate(row)
        for row in view.iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields))
    ]


def load_measurements(records: list[DifficultyMeasurement], query: AppletQuery) -> dict[str, Any]:
    """Insert measurements in bounded batches; reject an existing ID with different evidence.

    Each batch inserts through a bounded JSONB recordset and the typed readback
    pages the inserted ids, so a full campaign costs a handful of requests
    instead of an insert plus a readback per record.
    """
    if len({record.measurement_id for record in records}) != len(records):
        raise ValueError("Duplicate difficulty measurement IDs")
    inserted = 0
    for body in batches(
        {
            "measurement_id": record.measurement_id,
            "source_id": canonical_source_id(record.source_id),
            "protocol_id": record.protocol_id,
            "created_at": record.created_at,
            "measurement": record.model_dump(mode="json"),
        }
        for record in records
    ):
        result = query(
            """
            INSERT INTO atlas_difficulty (measurement_id, source_id, protocol_id, created_at, measurement)
            SELECT r.measurement_id, r.source_id, r.protocol_id, r.created_at, r.measurement
            FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS r(
                measurement_id TEXT, source_id TEXT, protocol_id TEXT, created_at TIMESTAMPTZ, measurement JSONB)
            ON CONFLICT (measurement_id) DO NOTHING
            """,
            {"rows": body},
        )
        inserted += result["row_count"]
    stored = {}
    ids = sorted({record.measurement_id for record in records})
    for start in range(0, len(ids), MEASUREMENT_READBACK_BATCH):
        chunk = ids[start : start + MEASUREMENT_READBACK_BATCH]
        for row in query(
            "SELECT measurement_id, measurement FROM atlas_difficulty WHERE measurement_id IN "
            "(SELECT r.measurement_id FROM jsonb_to_recordset(CAST(:ids AS JSONB)) AS r(measurement_id TEXT))",
            {"ids": json.dumps([{"measurement_id": value} for value in chunk], separators=(",", ":"))},
        )["rows"]:
            stored[row["measurement_id"]] = row["measurement"]
    for record in records:
        measurement = stored.get(record.measurement_id)
        if measurement is None or DifficultyMeasurement.model_validate(measurement) != record:
            raise ValueError(f"Difficulty measurement ID already names different evidence: {record.measurement_id}")
    return {"inserted": inserted, "records": len(records)}


def difficulty_records_from_query(query: AppletQuery) -> dict[str, list[DifficultyObservation]]:
    """The stored difficulty observations, grouped by canonical source id.

    The read paginates by measurement id and projects the stored measurement
    with the server's summary shape (`#- '{sample,tasks}'`): the observation
    model summarizes the sample without its per-task rows, which stay
    immutable in storage.
    """
    records: dict[str, list[DifficultyObservation]] = {}
    offset = 0
    while True:
        rows = query(
            "SELECT measurement #- '{sample,tasks}' AS measurement FROM atlas_difficulty "
            "ORDER BY measurement_id LIMIT :limit OFFSET :offset",
            {"limit": DIFFICULTY_READ_BATCH, "offset": offset},
        )["rows"]
        for row in rows:
            record = DifficultyObservation.model_validate(row["measurement"])
            records.setdefault(canonical_source_id(record.source_id), []).append(record)
        if len(rows) < DIFFICULTY_READ_BATCH:
            return records
        offset += DIFFICULTY_READ_BATCH


def publish_coverage(query: AppletQuery, load_id: str, *, sealed_snapshot_id: str | None = None) -> dict[str, Any]:
    """Compute the Difficulty coverage over the loaded catalog and store it for this load.

    The pure classifier runs over the loaded declarations and the stored
    observations, so every rebuild republishes the coverage the Difficulty view
    and the campaign-queue CLI consume — never a hand-maintained list. Both
    bulk reads paginate under the applet transport's 5 MiB response bound.
    """
    declarations: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        rows = query(
            "SELECT id, declaration FROM atlas_sources WHERE load_id = :load_id ORDER BY id LIMIT :limit OFFSET :offset",
            {"load_id": load_id, "limit": DECLARATION_READ_BATCH, "offset": offset},
        )["rows"]
        for row in rows:
            declarations[canonical_source_id(row["id"])] = row["declaration"]
        if len(rows) < DECLARATION_READ_BATCH:
            break
        offset += DECLARATION_READ_BATCH
    observations = difficulty_records_from_query(query)
    sealed = {}
    if sealed_snapshot_id is not None:
        offset = 0
        while True:
            rows = query(
                "SELECT c.source_id,c.cell,s.snapshot_id,s.manifest_uri,s.manifest_sha256 "
                "FROM atlas_difficulty_seal_loads l JOIN atlas_difficulty_seals s USING(snapshot_id) "
                "JOIN atlas_difficulty_seal_cells c USING(snapshot_id) WHERE l.load_id=:load_id "
                "ORDER BY c.source_id,c.solver LIMIT :limit OFFSET :offset",
                {"load_id": load_id, "limit": DIFFICULTY_READ_BATCH, "offset": offset},
            )["rows"]
            for row in rows:
                record = sealed.setdefault(row["source_id"], {"cells": []})
                record.update(snapshot_id=row["snapshot_id"], uri=row["manifest_uri"], sha256=row["manifest_sha256"])
                record["cells"].append(row["cell"])
            if len(rows) < DIFFICULTY_READ_BATCH:
                break
            offset += DIFFICULTY_READ_BATCH
    document = difficulty_coverage(declarations, observations, sealed)
    payload = document.model_dump(mode="json", include={"sources", "counts", "generated_at"})
    payload["load_id"] = load_id
    stored = DifficultyCoverage.model_validate({**payload, "snapshot_id": None, "queue": []})
    query(
        """
        INSERT INTO atlas_difficulty_coverage (load_id, coverage)
        VALUES (:load_id, CAST(:coverage AS JSONB))
        ON CONFLICT (load_id) DO UPDATE SET coverage = EXCLUDED.coverage
        """,
        {"load_id": load_id, "coverage": json.dumps(payload, allow_nan=False)},
    )
    return {"load_id": load_id, "counts": stored.counts}


@click.command(help=__doc__)
@click.option("--catalog-root", required=True, help="FineStore archive containing difficulty_measurements.")
@click.option("--applet-id", required=True)
@click.option("--marina-url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--receipt", type=click.Path(path_type=Path), required=True)
def main(catalog_root: str, applet_id: str, marina_url: str, receipt: Path) -> None:
    configure_coreweave_s3()
    view = ReadView(catalog_root)
    records = read_measurements(view)
    query = applet_query(marina_url, applet_id)
    result = load_measurements(records, query)
    current = query("SELECT load_id FROM atlas_current WHERE name = 'latest'", {})["rows"]
    if current:
        result["coverage"] = publish_coverage(query, current[0]["load_id"])
    receipt.write_text(json.dumps({"catalog_root": catalog_root, **result}, indent=2) + "\n")
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
