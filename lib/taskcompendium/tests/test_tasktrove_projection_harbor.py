# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from experiments.post_training.taskcompendium.trial_tasktrove_projection import _source_row_index
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
