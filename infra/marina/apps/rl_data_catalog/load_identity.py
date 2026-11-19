# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a pinned overlap survey over an unchanged, input-matched source catalog."""

import json
from pathlib import Path
from typing import Any

import click
from marina.client import DEFAULT_MARINA_URL, AppletQuery, applet_query
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey
from infra.marina.apps.rl_data_catalog.load import batches, ensure_tables
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


def load_identity(reference: ManifestReference, query: AppletQuery) -> dict[str, Any]:
    """Bind reviews only to the selected immutable fingerprint inputs, preserving all old records."""
    survey = canonical_identity_survey(json.loads(manifest_bytes(reference)))
    current = query(
        "SELECT c.load_id,l.snapshot_id FROM atlas_current c JOIN atlas_loads l USING(load_id) WHERE c.name='latest'", {}
    )["rows"][0]
    load_id = current["load_id"]
    sources = query(
        "SELECT id,source_fingerprint->'inputs' AS inputs,"
        "source_fingerprint->'task_count' AS task_count,"
        "source_fingerprint->'sample_sha256' AS sample_sha256,source_fingerprint->'seed' AS seed,"
        "declaration->'within_source_dedup'->'unique_count' AS retained FROM atlas_sources "
        "WHERE load_id=:load_id ORDER BY id",
        {"load_id": load_id},
    )["rows"]
    active = set()
    empty_sources = []
    for row in sources:
        source_id = canonical_source_id(row["id"])
        fingerprint = survey.fingerprints.get(source_id)
        if row["task_count"] == 0 and row["retained"] == 0:
            empty_sources.append(source_id)
            continue
        if row["inputs"] is None:
            if fingerprint is not None:
                raise ValueError(f"Survey supplies an unbound fingerprint: {source_id}")
        elif fingerprint is None or any(
            getattr(fingerprint, key) != row[key] for key in ("inputs", "task_count", "sample_sha256", "seed")
        ):
            raise ValueError(f"Survey fingerprint differs from the frozen catalog: {source_id}")
        active.add(source_id)
    reviews = [
        review for review in survey.reviews if review.similarity.left in active and review.similarity.right in active
    ]
    measurement_id = reference.sha256
    metadata = {
        key: value
        for key, value in survey.model_dump(mode="json").items()
        if key not in ("reviews", "groups", "fingerprints")
    }
    metadata.update(
        reference=reference.model_dump(mode="json"),
        bound_snapshot_id=current["snapshot_id"],
        active_reviews=len(reviews),
        archived_reviews=len(survey.reviews),
        excluded_empty_sources=empty_sources,
    )
    ensure_tables(
        query,
        ("atlas_identity_measurements", "atlas_identity_measurement_reviews", "atlas_identity_measurement_current"),
    )
    query(
        (
            "INSERT INTO atlas_identity_measurements VALUES "
            "(:load_id,:measurement_id,CAST(:metadata AS JSONB)) ON CONFLICT DO NOTHING"
        ),
        {"load_id": load_id, "measurement_id": measurement_id, "metadata": json.dumps(metadata)},
    )
    stored = query(
        "SELECT metadata FROM atlas_identity_measurements WHERE load_id=:load_id AND measurement_id=:measurement_id",
        {"load_id": load_id, "measurement_id": measurement_id},
    )["rows"][0]["metadata"]
    if stored != metadata:
        raise ValueError("An overlap measurement ID already names different evidence")
    records = [
        {"left_source": row.similarity.left, "right_source": row.similarity.right, "review": row.model_dump(mode="json")}
        for row in reviews
    ]
    for body in batches(records):
        params = {"load_id": load_id, "measurement_id": measurement_id, "rows": body}
        query(
            """INSERT INTO atlas_identity_measurement_reviews
            SELECT :load_id,:measurement_id,left_source,right_source,review
            FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS r(left_source TEXT,right_source TEXT,review JSONB)
            ON CONFLICT DO NOTHING""",
            params,
        )
        conflicts = query(
            """SELECT COUNT(*) AS count
            FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS r(left_source TEXT,right_source TEXT,review JSONB)
            LEFT JOIN atlas_identity_measurement_reviews s
            ON s.load_id=:load_id AND s.measurement_id=:measurement_id
                AND s.left_source=r.left_source AND s.right_source=r.right_source
            WHERE s.review IS DISTINCT FROM r.review""",
            params,
        )["rows"][0]["count"]
        if conflicts:
            raise ValueError("Stored overlap reviews differ from the pinned survey")
    selected = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    if selected != [{"load_id": load_id}]:
        raise ValueError("Catalog selection changed while staging overlap evidence")
    query(
        (
            "INSERT INTO atlas_identity_measurement_current VALUES (:load_id,:measurement_id) ON "
            "CONFLICT (load_id) DO UPDATE SET measurement_id=EXCLUDED.measurement_id"
        ),
        {"load_id": load_id, "measurement_id": measurement_id},
    )
    return {
        "load_id": load_id,
        "measurement_id": measurement_id,
        "active_reviews": len(reviews),
        "archived_reviews": len(survey.reviews),
        "reference": reference.model_dump(mode="json"),
        "excluded_empty_sources": empty_sources,
    }


@click.command(help=__doc__)
@click.option("--survey", type=(str, str), metavar="URI SHA256", required=True)
@click.option("--app-id", required=True)
@click.option("--url", default=DEFAULT_MARINA_URL)
@click.option("--receipt", type=click.Path(path_type=Path), required=True)
def main(survey: tuple[str, str], app_id: str, url: str, receipt: Path) -> None:
    if receipt.exists():
        raise click.ClickException("Choose a new receipt path")
    configure_coreweave_s3()
    result = load_identity(ManifestReference(uri=survey[0], sha256=survey[1]), applet_query(url, app_id))
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    click.echo(json.dumps(result))


if __name__ == "__main__":
    main()
