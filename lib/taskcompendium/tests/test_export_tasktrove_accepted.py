# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from experiments.post_training.taskcompendium.audit_tasktrove_rights_metadata import audit_rights_metadata
from experiments.post_training.taskcompendium.export_tasktrove_accepted import export_accepted_records
from experiments.post_training.taskcompendium.ingest_tasktrove import PUBLIC_CANDIDATE_COHORTS
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.importers.tasktrove.models import IMPORTER_REVISION

RELEASE_URI = "s3://example/tasktrove/clean/test-release"
SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"
ARCHIVE_PATH = "synthetic-task.tar.gz"
OBJECT_PIN = (
    "2026.09.18.3#manifest-sha256=" + "a" * 64 + "#parquet-size=1234#parquet-etag=etag-value#parquet-version-id="
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_row(split: str, task_id: str) -> dict:
    return {
        "input_split": split,
        "input_file": f"{split}/part-00000.parquet",
        "input_row": 7,
        "source": SOURCE,
        "path": ARCHIVE_PATH,
        "mode": PUBLIC_CANDIDATE_COHORTS[SOURCE],
        "family": "qa-short-answer",
        "converter": "nemotron_mcqa",
        "template_id": "c814af4f124d",
        "tags": ["qa", "mcq"],
        "input_object_pin": OBJECT_PIN,
        "archive_sha256": "b" * 64,
        "disposition": "imported",
        "imported_id": task_id,
        "reason": None,
    }


def _candidate(task_id: str) -> dict:
    return {
        "record_version": 1,
        "id": task_id,
        "context": {},
        "environment_requirements": {},
        "tool_providers": {},
        "final_tools": {},
        "answer_type": "text",
        "source": {
            "dataset": RELEASE_URI,
            "revision": OBJECT_PIN,
            "row": f"{SOURCE}:{ARCHIVE_PATH}",
            "importer_revision": IMPORTER_REVISION,
        },
        "submission_instruction": "Answer the synthetic question.",
        "tags": ["qa", "mcq"],
        "source_category": "qa-short-answer",
    }


def _proof(task_id: str) -> dict:
    return {
        "candidate_id": task_id,
        "input_file": "tasks/part-00000.parquet",
        "input_row": 7,
        "input_object_pin": OBJECT_PIN,
        "source": SOURCE,
        "path": ARCHIVE_PATH,
        "archive_sha256": "b" * 64,
        "disposition": "imported",
    }


def _input_artifacts(tmp_path, *, pin=OBJECT_PIN, duplicate_candidate=False):
    task_id = "synthetic-task-id"
    tasks_row = _source_row("tasks", task_id)
    tasks_row["input_object_pin"] = pin
    sft_row = _source_row("sft", "private-sft-id")
    sft_row["input_file"] = "sft/part-00000.parquet"
    ledger_path = tmp_path / "ingestion-ledger.parquet"
    pq.write_table(pa.Table.from_pylist([tasks_row, sft_row]), ledger_path)
    catalog_path = tmp_path / "private-catalog.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "synthetic-task-id",
                    "source": SOURCE,
                    "path": ARCHIVE_PATH,
                    "tags": ["qa", "mcq"],
                    "source_metadata_json": "{}",
                },
                {
                    "id": "private-sft-id",
                    "source": SOURCE,
                    "path": ARCHIVE_PATH,
                    "tags": ["qa", "mcq"],
                    "source_metadata_json": "{}",
                },
            ]
        ),
        catalog_path,
    )

    candidate_path = tmp_path / "public-candidates.jsonl"
    candidates = [_candidate(task_id)] * (2 if duplicate_candidate else 1)
    if pin != OBJECT_PIN:
        candidates[0]["source"]["revision"] = pin
    candidate_path.write_text("".join(json.dumps(candidate) + "\n" for candidate in candidates))
    proof_path = tmp_path / "candidate-proof.jsonl"
    proof = _proof(task_id)
    proof["input_object_pin"] = pin
    proof_path.write_text(json.dumps(proof) + "\n")
    artifacts = {
        "ledger": {"sha256": _sha256(ledger_path)},
        "private_catalog": {"sha256": _sha256(catalog_path)},
        "public_candidates": {"sha256": _sha256(candidate_path)},
        "candidate_proof": {"sha256": _sha256(proof_path)},
    }
    manifest = {
        "status": "complete",
        "release_uri": RELEASE_URI,
        "release_revision": "2026.09.18.3",
        "source_manifest_sha256": "c" * 64,
        "manifest_uri": "s3://example/ingestion-manifest.json",
        "importer_revision": IMPORTER_REVISION,
        "artifacts": artifacts,
    }
    return tuple(StoragePath(str(path)) for path in (ledger_path, catalog_path, candidate_path, proof_path)), manifest


def test_export_holds_rows_until_packaging_clearance_and_excludes_sft(tmp_path):
    inputs, manifest = _input_artifacts(tmp_path)
    output = StoragePath(str(tmp_path / "accepted"))

    result = export_accepted_records(*inputs, output, ingestion_manifest=manifest, builder_revision="d" * 40)

    assert result["outputs"] == {}
    assert result["accepted_rows_by_source"] == {}
    assert result["rights_rejected_rows_by_source_reason"] == [
        {"source": SOURCE, "reason": "awaiting-exact-cohort-clearance", "rows": 1}
    ]
    terms = result["rights_terms_by_source_mode_family_converter"]
    assert {term["term_field"]: term["value"] for term in terms} == {
        "license": "<absent>",
        "source_license": "<absent>",
        "license_url": "<absent>",
        "attribution": "<absent>",
        "copyright": "<absent>",
        "author": "<absent>",
        "authors": "<absent>",
        "url": "<absent>",
        "source_url": "<absent>",
    }
    assert result["rights_term_examples_by_source_mode_family_converter"]


def test_rights_metadata_audit_verifies_proofs_without_clearing_rows(tmp_path, monkeypatch):
    inputs, manifest = _input_artifacts(tmp_path)
    ingestion = tmp_path / "ingestion"
    ingestion.mkdir()
    for source, name in zip(
        inputs,
        ("ingestion-ledger.parquet", "private-catalog.parquet", "public-candidates.jsonl", "candidate-proof.jsonl"),
        strict=True,
    ):
        target = ingestion / name
        with source.open("rb") as opened:
            target.write_bytes(opened.read())
    manifest_path = ingestion / "ingestion-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest["manifest_uri"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "experiments.post_training.taskcompendium.audit_tasktrove_rights_metadata.configure_coreweave_s3",
        lambda: None,
    )

    report = audit_rights_metadata(
        str(ingestion), str(tmp_path / "rights-audit"), tmp_path / "summary", builder_revision="d" * 40
    )

    summary = json.loads((tmp_path / "summary/rights-audit-summary.json").read_text())
    assert report["accepted_rows_by_source"] == {}
    assert summary["accepted_public_records"] == 0
    assert summary["candidate_sha256_verified"] is True
    assert summary["proof_sha256_verified"] is True
    assert summary["archive_payload_bytes_read"] == 0
    assert summary["private_task_specifications_loaded"] is False
    assert (tmp_path / "rights-audit/manifest.json").exists()


def test_export_writes_proof_wrapped_record_for_exact_cleared_cohort(tmp_path):
    inputs, manifest = _input_artifacts(tmp_path)
    output = StoragePath(str(tmp_path / "cleared"))
    clearance_key = (SOURCE, "mcq", "qa-short-answer", "nemotron_mcqa", "c814af4f124d", "tasks")

    result = export_accepted_records(
        *inputs,
        output,
        ingestion_manifest=manifest,
        builder_revision="d" * 40,
        clearance_audit_manifest_sha256="ab71556290d1ce54e596588b549de95ddd68366cd8fd3bb18b77ed9e99f0eed1",
        clearances={clearance_key: {"source_card_revision": "synthetic-card", "expected_rows": 1}},
    )

    output_record_path = output / "tasktrove_clean" / "train" / SOURCE / "qa-short-answer.jsonl"
    record = json.loads(output_record_path.read_bytes())
    assert set(record) == {"task", "source_proof"}
    assert record["task"]["id"] == "synthetic-task-id"
    assert record["task"]["tags"] == ["qa", "mcq"]
    assert record["source_proof"] == {
        "source_row": f"{SOURCE}:{ARCHIVE_PATH}",
        "input_file": "tasks/part-00000.parquet",
        "input_object_pin": OBJECT_PIN,
        "archive_path": ARCHIVE_PATH,
        "archive_sha256": "b" * 64,
    }
    assert result["accepted_rows_by_source_family_split"] == {f"{SOURCE}/qa-short-answer/tasks": 1}
    assert result["source_assets_by_source_family_split"] == {
        f"{SOURCE}/qa-short-answer/tasks": [{"path": "tasks/part-00000.parquet", "pin": OBJECT_PIN}]
    }
    assert result["original_split_to_public_split"] == {"tasks": "train", "sft": None}
    assert (output / "manifest.json").exists()


def test_export_rejects_unpinned_parquet_and_duplicate_candidates(tmp_path):
    bad_pin = "2026.09.18.3#manifest-sha256=" + "a" * 64 + "#parquet-size=1234#parquet-etag=#parquet-version-id="
    inputs, manifest = _input_artifacts(tmp_path, pin=bad_pin)
    with pytest.raises(ValueError, match="immutable object pin"):
        export_accepted_records(
            *inputs, StoragePath(str(tmp_path / "bad-pin")), ingestion_manifest=manifest, builder_revision="d" * 40
        )

    duplicate_directory = tmp_path / "duplicates"
    duplicate_directory.mkdir()
    duplicate_inputs, duplicate_manifest = _input_artifacts(duplicate_directory, duplicate_candidate=True)
    with pytest.raises(ValueError, match="audit failed"):
        export_accepted_records(
            *duplicate_inputs,
            StoragePath(str(tmp_path / "duplicate-candidate")),
            ingestion_manifest=duplicate_manifest,
            builder_revision="d" * 40,
        )


def test_export_flags_a_pinned_source_validation_split(tmp_path):
    inputs, manifest = _input_artifacts(tmp_path)
    manifest["upstream_tasktrove"] = {"validation": {"rows": 12}}

    with pytest.raises(ValueError, match="declares holdouts"):
        export_accepted_records(
            *inputs,
            StoragePath(str(tmp_path / "holdout")),
            ingestion_manifest=manifest,
            builder_revision="d" * 40,
        )


def test_export_rejects_candidate_tag_reordering_against_ledger_and_catalog(tmp_path):
    inputs, manifest = _input_artifacts(tmp_path)
    candidate_path = tmp_path / "public-candidates.jsonl"
    candidate = json.loads(candidate_path.read_text())
    candidate["tags"] = list(reversed(candidate["tags"]))
    candidate_path.write_text(json.dumps(candidate) + "\n")
    manifest["artifacts"]["public_candidates"]["sha256"] = _sha256(candidate_path)

    with pytest.raises(ValueError, match="ordered ledger tags"):
        export_accepted_records(
            *inputs,
            StoragePath(str(tmp_path / "tag-order")),
            ingestion_manifest=manifest,
            builder_revision="d" * 40,
        )
