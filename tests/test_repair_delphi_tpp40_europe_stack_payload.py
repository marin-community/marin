# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.transform.stack_edu.hydrate import stack_edu_record_id
from zephyr.readers import load_jsonl
from zephyr.writers import write_jsonl_file

from experiments.domain_phase_mix import repair_delphi_tpp40_europe_stack_payload as repair_module
from experiments.domain_phase_mix.repair_delphi_tpp40_europe_stack_payload import (
    apply_repair_bundles,
    extract_repair_bundle,
    load_repair_manifest,
)


def _metadata_row(blob_id: str) -> dict:
    return {
        "blob_id": blob_id,
        "repo_name": f"repo-{blob_id}",
        "path": f"/{blob_id}.py",
        "src_encoding": "utf-8",
        "detected_licenses": ["MIT"],
        "license_type": "permissive",
        "score": 0.9,
        "int_score": 4,
        "length_bytes": len(blob_id),
    }


def _document(language: str, row: dict, text: str) -> dict:
    return {
        "id": stack_edu_record_id(language, row),
        "text": text,
        "source": f"stack_edu/{language}",
        "metadata": {**row, "language": language},
    }


def _manifest_payload(root: Path, source_shard: Path, target_shard: Path, input_file: Path) -> dict:
    rows = [_metadata_row(blob_id) for blob_id in ("a", "b", "c", "d")]
    metric_path = root / "target" / ".metrics" / "hydrate-00000.jsonl"
    expected_metric = {
        "language": "Python",
        "input_file": str(input_file),
        "row_start": 0,
        "row_end": 4,
        "path": str(target_shard),
        "count": 3,
        "decoded_fallback": 0,
        "missing_blob": 1,
        "corrupt_blob": 0,
        "empty_blob": 0,
        "fetch_error": 0,
        "missing_blob_examples": ["d"],
        "corrupt_blob_examples": [],
        "empty_blob_examples": [],
        "fetch_error_examples": [],
    }
    return {
        "schema_version": 1,
        "repair_id": "test-repair",
        "source_payload": "test",
        "target_region": "local-target",
        "expected_add_records": 1,
        "expected_remove_records": 1,
        "expected_transfer_content_bytes": rows[1]["length_bytes"],
        "expected_transfer_content_bytes_by_source_region": {
            "local-source": rows[1]["length_bytes"],
        },
        "source_record_counts": {"local-source": 1},
        "tasks": [
            {
                "language": "Python",
                "source_region": "local-source",
                "source_shard": str(source_shard),
                "target_input_file": str(input_file),
                "target_shard": str(target_shard),
                "target_metrics_path": str(metric_path),
                "row_start": 0,
                "row_end": 4,
                "add_blob_ids": ["b"],
                "remove_blob_ids": ["d"],
                "baseline_missing_blob_ids": ["d"],
                "expected_metric": expected_metric,
            }
        ],
    }


def _write_manifest(path: Path, payload: dict) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_extract_and_apply_repair_preserve_metadata_order_and_rerun_idempotently(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_shard = source_root / "documents" / "source.jsonl.zst"
    target_shard = target_root / "documents" / "target.jsonl.zst"
    input_file = target_root / "raw" / "metadata.parquet"
    rows = [_metadata_row(blob_id) for blob_id in ("a", "b", "c", "d")]
    input_file.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), input_file)

    source_documents = [_document("Python", row, f"source-{row['blob_id']}") for row in rows]
    write_jsonl_file(source_documents, str(source_shard))
    target_documents = [source_documents[index] for index in (0, 2, 3)]
    write_jsonl_file(target_documents, str(target_shard))

    manifest_path = tmp_path / "manifest.json"
    manifest_sha256 = _write_manifest(
        manifest_path,
        _manifest_payload(tmp_path, source_shard, target_shard, input_file),
    )
    manifest = load_repair_manifest(str(manifest_path), manifest_sha256)

    region_prefixes = {
        "local-source": str(source_root),
        "local-target": str(target_root),
    }
    monkeypatch.setattr(repair_module, "marin_prefix_for_region", region_prefixes.__getitem__)

    bundle_path = source_root / "repair" / "bundle.jsonl.zst"
    extraction_result_path = source_root / "repair" / "result.jsonl"
    monkeypatch.setattr(repair_module, "region_from_metadata", lambda: "wrong-region")
    monkeypatch.setattr(repair_module, "cached_marin_region", lambda: "wrong-region")
    with pytest.raises(ValueError, match="VM in local-source"):
        extract_repair_bundle(
            manifest,
            manifest_sha256,
            "local-source",
            str(bundle_path),
            str(extraction_result_path),
        )
    monkeypatch.setattr(repair_module, "region_from_metadata", lambda: "local-source")
    monkeypatch.setattr(repair_module, "cached_marin_region", lambda: "local-source")
    outside_bundle = target_root / "wrong-bucket.jsonl.zst"
    with pytest.raises(ValueError, match="outside required storage prefix"):
        extract_repair_bundle(
            manifest,
            manifest_sha256,
            "local-source",
            str(outside_bundle),
            str(extraction_result_path),
        )
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="without its validated result marker"):
        extract_repair_bundle(
            manifest,
            manifest_sha256,
            "local-source",
            str(bundle_path),
            str(extraction_result_path),
        )
    bundle_path.unlink()
    first_extraction = extract_repair_bundle(
        manifest,
        manifest_sha256,
        "local-source",
        str(bundle_path),
        str(extraction_result_path),
    )
    second_extraction = extract_repair_bundle(
        manifest,
        manifest_sha256,
        "local-source",
        str(bundle_path),
        str(extraction_result_path),
    )
    assert first_extraction["record_count"] == 1
    assert second_extraction["skipped"] is True

    target_bundle = target_root / "repair" / "bundle.jsonl.zst"
    target_extraction_result = target_root / "repair" / "result.jsonl"
    target_bundle.parent.mkdir(parents=True)
    target_bundle.write_bytes(bundle_path.read_bytes())
    target_extraction_result.write_bytes(extraction_result_path.read_bytes())
    completion_path = target_root / "repair" / "completion.jsonl"
    monkeypatch.setattr(repair_module, "region_from_metadata", lambda: "wrong-region")
    monkeypatch.setattr(repair_module, "cached_marin_region", lambda: "wrong-region")
    with pytest.raises(ValueError, match="VM in local-target"):
        apply_repair_bundles(
            manifest,
            manifest_sha256,
            [str(target_bundle)],
            [str(target_extraction_result)],
            str(completion_path),
        )
    monkeypatch.setattr(repair_module, "region_from_metadata", lambda: "local-target")
    monkeypatch.setattr(repair_module, "cached_marin_region", lambda: "local-target")
    valid_bundle_bytes = target_bundle.read_bytes()
    target_bundle.write_bytes(valid_bundle_bytes + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        apply_repair_bundles(
            manifest,
            manifest_sha256,
            [str(target_bundle)],
            [str(target_extraction_result)],
            str(completion_path),
        )
    target_bundle.write_bytes(valid_bundle_bytes)
    first_apply = apply_repair_bundles(
        manifest,
        manifest_sha256,
        [str(target_bundle)],
        [str(target_extraction_result)],
        str(completion_path),
    )
    completion_path.unlink()
    second_apply = apply_repair_bundles(
        manifest,
        manifest_sha256,
        [str(target_bundle)],
        [str(target_extraction_result)],
        str(completion_path),
    )
    third_apply = apply_repair_bundles(
        manifest,
        manifest_sha256,
        [str(target_bundle)],
        [str(target_extraction_result)],
        str(completion_path),
    )

    repaired = list(load_jsonl(str(target_shard)))
    assert [record["metadata"]["blob_id"] for record in repaired] == ["a", "b", "c"]
    assert repaired[1]["text"] == "source-b"
    assert list(load_jsonl(manifest.tasks[0].target_metrics_path)) == [manifest.tasks[0].expected_metric]
    assert first_apply["repaired_tasks"] == 1
    assert first_apply["measured_delta"]["inserted_this_attempt"] == 1
    assert first_apply["measured_delta"]["removed_this_attempt"] == 1
    assert second_apply["measured_delta"]["already_present"] == 1
    assert second_apply["measured_delta"]["already_absent"] == 1
    assert second_apply["skipped"] is False
    assert third_apply["skipped"] is True
