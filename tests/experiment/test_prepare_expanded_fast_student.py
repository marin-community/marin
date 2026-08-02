# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from prepare_expanded_fast_student import aligned_training_tables  # noqa: E402


def test_aligned_training_tables_accepts_exact_rows() -> None:
    left = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})
    right = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})

    aligned_training_tables(left, right)


def test_aligned_training_tables_rejects_changed_hash_or_rank() -> None:
    source = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 1]})
    changed_hash = pa.table({"raw_sha256": ["a", "c"], "train_rank": [0, 1]})
    changed_rank = pa.table({"raw_sha256": ["a", "b"], "train_rank": [0, 2]})

    with pytest.raises(ValueError, match="raw_sha256"):
        aligned_training_tables(source, changed_hash)
    with pytest.raises(ValueError, match="train_rank"):
        aligned_training_tables(source, changed_rank)
