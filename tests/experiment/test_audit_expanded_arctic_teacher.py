# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from audit_expanded_arctic_teacher import aligned_columns_equal  # noqa: E402


def test_aligned_columns_equal_checks_all_rows_and_columns() -> None:
    left = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})
    same = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})
    changed_hash = pa.table({"raw_sha256": ["a", "c"], "train_rank": [0, 1]})
    changed_rank = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 2]})

    assert aligned_columns_equal(left, same, ("raw_sha256", "train_rank"))
    assert not aligned_columns_equal(left, changed_hash, ("raw_sha256", "train_rank"))
    assert not aligned_columns_equal(left, changed_rank, ("raw_sha256", "train_rank"))
