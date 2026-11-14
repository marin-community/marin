# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load immutable typed difficulty records without changing the selected source catalog."""

import json
from pathlib import Path
from typing import Any

import click
from finestore.reader import ReadView
from finestore.rl.difficulty import MEASUREMENTS, DifficultyMeasurement
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3

from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


def load_measurements(records: list[DifficultyMeasurement], query: AppletQuery) -> dict[str, Any]:
    """Insert each measurement once; reject an existing ID with different evidence."""
    if len({record.measurement_id for record in records}) != len(records):
        raise ValueError("Duplicate difficulty measurement IDs")
    inserted = 0
    for record in records:
        payload = record.model_dump(mode="json")
        result = query(
            """
            INSERT INTO atlas_difficulty (measurement_id, source_id, protocol_id, created_at, measurement)
            VALUES (:id, :source, :protocol, :created, CAST(:measurement AS JSONB))
            ON CONFLICT (measurement_id) DO NOTHING
            """,
            {
                "id": record.measurement_id,
                "source": canonical_source_id(record.source_id),
                "protocol": record.protocol_id,
                "created": record.created_at,
                "measurement": json.dumps(payload, allow_nan=False),
            },
        )
        inserted += result["row_count"]
        stored = query(
            "SELECT measurement FROM atlas_difficulty WHERE measurement_id = :id", {"id": record.measurement_id}
        )["rows"]
        if len(stored) != 1 or stored[0]["measurement"] != payload:
            raise ValueError(f"Difficulty measurement ID already names different evidence: {record.measurement_id}")
    return {"inserted": inserted, "records": len(records)}


@click.command(help=__doc__)
@click.option("--catalog-root", required=True, help="FineStore archive containing difficulty_measurements.")
@click.option("--applet-id", required=True)
@click.option("--marina-url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--receipt", type=click.Path(path_type=Path), required=True)
def main(catalog_root: str, applet_id: str, marina_url: str, receipt: Path) -> None:
    configure_coreweave_s3()
    view = ReadView(catalog_root)
    records = [
        DifficultyMeasurement.model_validate(row)
        for row in view.iter_rows(MEASUREMENTS, columns=list(DifficultyMeasurement.model_fields))
    ]
    result = load_measurements(records, applet_query(marina_url, applet_id))
    receipt.write_text(json.dumps({"catalog_root": catalog_root, **result}, indent=2) + "\n")
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
