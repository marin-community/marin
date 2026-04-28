# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
    write_ingestion_metadata_json,
)


def _manifest() -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key="GEM/totto",
        slice_key="structured_text/totto/validation",
        source_label="totto:validation",
        source_urls=("https://huggingface.co/datasets/GEM/totto",),
        source_license="CC BY-SA 3.0",
        source_format="huggingface_parquet_table_records",
        surface_form="wikipedia_table_tsv_plus_summary_sentence",
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only probe slice.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="high: direct eval contamination if reused for training",
            provenance_notes="Pinned HF revision.",
        ),
        staging=StagingMetadata(
            transform_name="stage_table_record_source",
            serializer_name="totto",
            split="validation",
            metadata={
                "output_filename": "staged.jsonl.gz",
                "provenance_fields": ["dataset", "split", "serializer", "index"],
            },
        ),
        epic_issue=5005,
        issue_numbers=(5059,),
        sample_caps=SampleCapConfig(max_bytes_per_source=30 * 1024 * 1024),
        compressed_size_bytes=123_456,
        rough_tokens_b=0.012,
        source_metadata={"hf_revision": "abc123"},
    )


def test_content_fingerprint_ignores_provenance_only_metadata():
    manifest = _manifest()
    updated = manifest.model_copy(
        update={
            "policy": manifest.policy.model_copy(
                update={"contamination_risk": "medium: still held out, but with different review outcome"}
            )
        }
    )

    assert manifest.fingerprint() == updated.fingerprint()
    assert manifest.provenance_fingerprint() != updated.provenance_fingerprint()


def test_content_fingerprint_changes_when_text_projection_changes():
    manifest = _manifest()
    updated = manifest.model_copy(
        update={"staging": manifest.staging.model_copy(update={"serializer_name": "wikitablequestions"})}
    )

    assert manifest.fingerprint() != updated.fingerprint()
    assert manifest.provenance_fingerprint() != updated.provenance_fingerprint()


def test_write_ingestion_metadata_json_includes_policy_and_runtime_fields(tmp_path):
    manifest = _manifest()
    materialized_output = MaterializedOutputMetadata(
        input_path="raw://totto",
        output_path=str(tmp_path),
        output_file=str(tmp_path / "staged.jsonl.gz"),
        record_count=17,
        bytes_written=4096,
        metadata={"source_file_count": 3},
    )
    metadata_path = write_ingestion_metadata_json(
        manifest=manifest,
        materialized_output=materialized_output,
    )

    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata_path == str(tmp_path / "metadata.json")
    assert set(payload) == {
        "schema_version",
        "manifest_fingerprint",
        "content_fingerprint",
        "source_manifest",
        "materialized_output",
    }
    assert payload["schema_version"] == 1
    assert payload["manifest_fingerprint"] == manifest.provenance_fingerprint()
    assert payload["content_fingerprint"] == manifest.fingerprint()
    assert payload["source_manifest"]["dataset_key"] == "GEM/totto"
    assert payload["source_manifest"]["policy"]["training_allowed"] is False
    assert payload["source_manifest"]["policy"]["eval_only"] is True
    assert payload["source_manifest"]["compressed_size_bytes"] == 123_456
    assert payload["source_manifest"]["rough_tokens_b"] == 0.012
    assert payload["source_manifest"]["staging"]["serializer_name"] == "totto"
    assert payload["source_manifest"]["staging"]["metadata"]["output_filename"] == "staged.jsonl.gz"
    assert payload["materialized_output"]["output_file"] == str(tmp_path / "staged.jsonl.gz")
    assert payload["materialized_output"]["record_count"] == 17
    assert payload["materialized_output"]["metadata"]["source_file_count"] == 3
