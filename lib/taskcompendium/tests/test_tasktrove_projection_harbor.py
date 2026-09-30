# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from experiments.post_training.taskcompendium import trial_tasktrove_math_candidates, trial_tasktrove_projection
from experiments.post_training.taskcompendium.trial_tasktrove_projection import (
    _read_candidate_archive,
    _source_row_index,
)
from rigging.filesystem.storage_path import StoragePath


def _ledger_row(input_row: int, *, archive_sha256: str = "a" * 64) -> dict:
    return {
        "input_split": "tasks",
        "input_file": "s3://bucket/tasks/part-00000.parquet",
        "input_row": input_row,
        "source": "laion__nemotron-gym-knowledge-mcqa-v2",
        "path": "sample.tar.gz",
        "input_object_pin": "source-object-pin",
        "archive_sha256": archive_sha256,
        "disposition": "imported",
        "imported_id": "task-1",
    }


def _candidate() -> dict:
    return {
        "id": "task-1",
        "source": {"row": "laion__nemotron-gym-knowledge-mcqa-v2:sample.tar.gz"},
    }


def _proof() -> dict:
    return {
        "source_row": "laion__nemotron-gym-knowledge-mcqa-v2:sample.tar.gz",
        "input_file": "s3://bucket/tasks/part-00000.parquet",
        "input_object_pin": "source-object-pin",
        "archive_path": "sample.tar.gz",
        "archive_sha256": "a" * 64,
    }


def test_source_proof_resolves_private_row_index_without_public_row_field(tmp_path):
    ledger_path = tmp_path / "ledger.parquet"
    pq.write_table(pa.Table.from_pylist([_ledger_row(17), _ledger_row(18, archive_sha256="b" * 64)]), ledger_path)

    assert "input_row" not in _proof()
    assert _source_row_index(StoragePath(str(ledger_path)), _candidate(), _proof()) == 17


def test_source_proof_requires_one_exact_private_ledger_match(tmp_path):
    ledger_path = tmp_path / "ledger.parquet"
    pq.write_table(pa.Table.from_pylist([_ledger_row(17), _ledger_row(17)]), ledger_path)

    with pytest.raises(ValueError, match="expected exactly one"):
        _source_row_index(StoragePath(str(ledger_path)), _candidate(), _proof())


def test_reimport_uses_the_accepted_clean_release_identity(monkeypatch):
    candidate = {
        "source": {
            "dataset": "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3",
            "revision": "2026.09.18.3",
        }
    }
    captured = {}

    def record_arguments(*args):
        captured["args"] = args
        return object()

    monkeypatch.setattr(trial_tasktrove_projection, "read_archive", record_arguments)

    _read_candidate_archive(b"archive", candidate, "source", "path.tar.gz")

    assert captured["args"] == (
        b"archive",
        "source",
        "path.tar.gz",
        candidate["source"]["dataset"],
        candidate["source"]["revision"],
    )


def test_math_batch_samples_are_deterministic_and_unique():
    assert trial_tasktrove_math_candidates._sample_positions(1) == (0,)
    assert trial_tasktrove_math_candidates._sample_positions(4) == (0, 2, 3)
    with pytest.raises(ValueError, match="no candidate rows"):
        trial_tasktrove_math_candidates._sample_positions(0)


def test_math_batch_wrapper_converts_private_proof_to_public_proof_shape():
    source = "laion__nemotron-gym-math-openmathreasoning-v2"
    candidate = {
        "record_version": 1,
        "id": "tasktrove-synthetic",
        "context": {},
        "environment_requirements": {},
        "tool_providers": {},
        "final_tools": [],
        "answer_type": "text",
        "source": {"row": f"{source}:sample.tar.gz"},
        "submission_instruction": "Answer the synthetic question.",
        "tags": ["math", "synthetic"],
        "source_category": "math-answer",
    }
    proof = {
        "candidate_id": candidate["id"],
        "input_file": "s3://bucket/tasks/part-00000.parquet",
        "input_row": 3,
        "input_object_pin": "immutable-pin",
        "source": source,
        "path": "sample.tar.gz",
        "archive_sha256": "a" * 64,
        "disposition": "imported",
    }

    wrapper = trial_tasktrove_math_candidates._candidate_wrapper(candidate, proof)

    assert wrapper == {
        "task": candidate,
        "source_proof": {
            "source_row": f"{source}:sample.tar.gz",
            "input_file": proof["input_file"],
            "input_object_pin": proof["input_object_pin"],
            "archive_path": "sample.tar.gz",
            "archive_sha256": proof["archive_sha256"],
        },
    }


def test_math_batch_selects_matching_candidate_and_proof_rows(tmp_path):
    source = "laion__nemotron-gym-math-openmathreasoning-v2"
    candidate_rows = []
    proof_rows = []
    for index in range(3):
        candidate = {
            "record_version": 1,
            "id": f"tasktrove-synthetic-{index}",
            "context": {},
            "environment_requirements": {},
            "tool_providers": {},
            "final_tools": [],
            "answer_type": "text",
            "source": {"row": f"{source}:sample-{index}.tar.gz"},
            "submission_instruction": "Answer the synthetic question.",
            "tags": ["math", "synthetic"],
            "source_category": "math-answer",
        }
        proof = {
            "candidate_id": candidate["id"],
            "input_file": "s3://bucket/tasks/part-00000.parquet",
            "input_row": index,
            "input_object_pin": "immutable-pin",
            "source": source,
            "path": f"sample-{index}.tar.gz",
            "archive_sha256": f"{index:x}" * 64,
            "disposition": "imported",
        }
        candidate_rows.append(json.dumps(candidate))
        proof_rows.append(json.dumps(proof))
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text("\n".join(candidate_rows) + "\n")
    proof_path = tmp_path / "proof.jsonl"
    proof_path.write_text("\n".join(proof_rows) + "\n")

    wrappers = trial_tasktrove_math_candidates._selected_wrappers(
        StoragePath(str(candidate_path)), StoragePath(str(proof_path)), row_count=3
    )

    assert [wrapper["task"]["id"] for wrapper in wrappers] == [f"tasktrove-synthetic-{i}" for i in range(3)]
    assert wrappers[1]["source_proof"]["archive_path"] == "sample-1.tar.gz"
