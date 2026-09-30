from pathlib import Path

from capability_pipeline.catalog import (
    build_full_manifest,
    ingest_catalog,
    validate_pilot,
)


def test_full_catalog_manifest_preserves_every_source_record():
    source = Path(__file__).resolve().parents[1] / "catalog.json"
    ingestion = ingest_catalog(source)
    manifest = build_full_manifest(ingestion)
    validate_pilot(manifest)
    assert len(manifest["capabilities"]) == len(ingestion.records)
    assert {r["capability_id"] for r in manifest["capabilities"]} == {
        r.capability_id for r in ingestion.records
    }
    assert manifest["catalog_audit"]["status"] == "complete"
