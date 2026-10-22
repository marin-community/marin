# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.datakit.scripts.metadata_artifact_inventory import audit_provider_routes, sanitize_metadata


def test_secret_redaction_preserves_provider_provenance():
    revision = "a8fda3ce51dc49b70bd432346f59859ddc42fae34"
    metadata = {
        "dataset": "example/provider",
        "revision": revision,
        "provenance": {
            "command_line": ["iris", "-e", "HF_TOKEN", "private-hf-value", "API_KEY=private-api-value"],
        },
        "environment": {"AWS_SECRET_ACCESS_KEY": "private-aws-value"},
    }
    sanitized = sanitize_metadata(metadata)
    rendered = json.dumps(sanitized)
    assert "private-hf-value" not in rendered
    assert "private-api-value" not in rendered
    assert "private-aws-value" not in rendered
    assert sanitized["dataset"] == "example/provider"
    assert sanitized["revision"] == revision
    assert metadata["provenance"]["command_line"][3] == "private-hf-value"


def test_provider_audit_reads_original_legacy_provenance(tmp_path):
    copied = tmp_path / "copied" / "marin" / "raw" / "example"
    original_prefix = tmp_path / "original" / "marin"
    original = original_prefix / "raw" / "example"
    original.mkdir(parents=True)
    files = ["datasets/example/provider/data/train-1.parquet", "datasets/example/provider/data/train-2.parquet"]
    (original / "provenance.json").write_text(
        json.dumps({"dataset": "example/provider", "version": "pinned", "links": files})
    )
    inventory = {
        "records": {
            str(copied): {
                "status": "found",
                "record": {"config": {"attrs": {"hf_dataset_id": "example/provider", "revision": "pinned"}}},
            }
        },
        "sources": [{"marin_source": "example", "normalization_input_artifacts": [str(copied)]}],
    }
    report = audit_provider_routes(inventory, fetch_sidecars=True, original_prefix=str(original_prefix))
    route = report["sources"][0]["provider_routes"][0]
    assert route["original_selected_files"] == files
    assert route["metadata_lookup_path"] == str(original / "provenance.json")
    assert route["provider_kind"] == "hugging_face"
    assert route["split_order_status"] == "not_verified"
    assert report["sources_with_verified_split_order"] == 0
