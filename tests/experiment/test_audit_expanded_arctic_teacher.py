# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from audit_expanded_arctic_teacher import aligned_columns_equal, selected_source_identity_table  # noqa: E402


def test_aligned_columns_equal_checks_all_rows_and_columns() -> None:
    left = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})
    same = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})
    changed_hash = pa.table({"raw_sha256": ["a", "c"], "train_rank": [0, 1]})
    changed_rank = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 2]})

    assert aligned_columns_equal(left, same, ("raw_sha256", "train_rank"))
    assert not aligned_columns_equal(left, changed_hash, ("raw_sha256", "train_rank"))
    assert not aligned_columns_equal(left, changed_rank, ("raw_sha256", "train_rank"))


def test_selected_source_identity_table_does_not_load_text(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "raw_sha256": ["a", "b", "c"],
                "split": ["eval", "train", "train"],
                "eval_rank": [0, None, None],
                "train_rank": [None, 0, 1],
                "in_30m": [False, True, False],
                "text": ["unused-a", "unused-b", "unused-c"],
            }
        ),
        path,
    )

    selected = selected_source_identity_table(str(path), "30m")

    assert selected.column_names == ["raw_sha256", "split", "eval_rank", "train_rank", "in_30m"]
    assert selected["raw_sha256"].to_pylist() == ["a", "b"]
