# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
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


def test_content_fingerprint_changes_when_text_projection_changes():
    manifest = _manifest()
    updated = manifest.model_copy(
        update={"staging": manifest.staging.model_copy(update={"serializer_name": "wikitablequestions"})}
    )

    assert manifest.fingerprint() != updated.fingerprint()
