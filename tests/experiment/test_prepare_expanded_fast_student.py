# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from prepare_expanded_fast_student import aligned_training_tables, prepare_source_names  # noqa: E402


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


def test_prepare_source_names_returns_complete_or_balanced_shard() -> None:
    manifest = {
        "sources": {
            "a": {"counts": {"train_30m": 8}},
            "b": {"counts": {"train_30m": 7}},
            "c": {"counts": {"train_30m": 6}},
            "d": {"counts": {"train_30m": 5}},
        }
    }

    assert prepare_source_names(manifest, "30m", None, None) == ["a", "b", "c", "d"]
    left = prepare_source_names(manifest, "30m", 0, 2)
    right = prepare_source_names(manifest, "30m", 1, 2)

    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(manifest["sources"])


def test_prepare_source_names_requires_both_shard_arguments() -> None:
    manifest = {"sources": {"a": {"counts": {"train_30m": 1}}}}

    with pytest.raises(ValueError, match="Both preparation shard arguments"):
        prepare_source_names(manifest, "30m", 0, None)
