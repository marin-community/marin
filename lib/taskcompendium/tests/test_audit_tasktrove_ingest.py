# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from experiments.post_training.taskcompendium.audit_tasktrove_ingest import audit_artifacts
from rigging.filesystem.storage_path import StoragePath


def _public_record(task_id: str) -> dict:
    return {
        "record_version": 1,
        "id": task_id,
        "context": {},
        "environment_requirements": {},
        "tool_providers": {},
        "final_tools": {},
        "answer_type": "text",
        "source": {},
        "submission_instruction": "Answer the synthetic question.",
        "tags": ["qa", "mcq"],
        "source_category": "mcq",
    }


def _artifacts(tmp_path, *, public_records=None, proof_records=None, accepted_rows=None):
    accepted_rows = accepted_rows or [
        {
            "input_split": "tasks",
            "input_file": "tasks/part-00000.parquet",
            "input_row": 4,
            "source": "laion__nemotron-gym-knowledge-mcqa-v2",
            "path": "synthetic.tar.gz",
            "mode": "mcq",
            "family": "qa-short-answer",
            "converter": "nemotron_mcqa",
            "tags": ["qa", "mcq"],
            "input_object_pin": "etag:synthetic",
            "archive_sha256": "a" * 64,
            "disposition": "imported",
            "imported_id": "task-1",
        },
    ]
    ledger_rows = [
        *accepted_rows,
        {
            "input_split": "sft",
            "input_file": "sft/part-00000.parquet",
            "input_row": 9,
            "source": "deferred",
            "path": "deferred.tar.gz",
            "mode": "judge",
            "family": "judge",
            "converter": "judge_rubric",
            "tags": ["judge"],
            "input_object_pin": "etag:synthetic",
            "archive_sha256": None,
            "disposition": "out-of-scope",
            "imported_id": None,
        },
    ]
    ledger_path = tmp_path / "ledger.parquet"
    pq.write_table(pa.Table.from_pylist(ledger_rows), ledger_path)
    public_path = tmp_path / "candidates.jsonl"
    public_path.write_text("".join(json.dumps(row) + "\n" for row in public_records or [_public_record("task-1")]))
    proof_path = tmp_path / "proof.jsonl"
    proof = proof_records or [
        {
            "candidate_id": "task-1",
            "input_file": "tasks/part-00000.parquet",
            "input_row": 4,
            "input_object_pin": "etag:synthetic",
            "source": "laion__nemotron-gym-knowledge-mcqa-v2",
            "path": "synthetic.tar.gz",
            "archive_sha256": "a" * 64,
            "disposition": "imported",
        }
    ]
    proof_path.write_text("".join(json.dumps(row) + "\n" for row in proof))
    return tuple(StoragePath(str(path)) for path in (ledger_path, public_path, proof_path))


def test_audit_counts_dispositions_and_validates_candidate_proof_join(tmp_path):
    result = audit_artifacts(*_artifacts(tmp_path))

    assert result["status"] == "passed"
    assert result["input_rows"] == 2
    assert result["disposition_counts"] == {"imported": 1, "out-of-scope": 1}
    assert result["unique_accepted_ids"] == 1
    assert (
        result["eligible_ledger_candidate_ids"] == result["public_candidate_ids"] == result["candidate_proof_ids"] == 1
    )
    assert result["counts_by_dimension_and_disposition"]["tag"]["qa"]["imported"] == 1
    assert result["counts_by_dimension_and_disposition"]["tag"]["judge"]["out-of-scope"] == 1


def test_audit_rejects_duplicate_candidates_and_invalid_allowlist(tmp_path):
    record = _public_record("task-1")
    leaked = {**record, "verifier": {"gold": "private"}}
    result = audit_artifacts(*_artifacts(tmp_path, public_records=[record, leaked]))

    assert result["status"] == "failed"
    assert result["validation_errors"]["candidate_allowlist_mismatch"] == 1
