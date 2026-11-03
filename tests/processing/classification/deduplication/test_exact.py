# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.processing.classification.deduplication.dedup_commons import DedupMode
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph

from tests.processing.classification.deduplication.conftest import load_dedup_parquet_outputs


def test_exact_paragraph_deduplication(fox_corpus):
    """Test exact deduplication using paragraph hashing (exact match)"""
    result = dedup_exact_paragraph(
        input_paths=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
    )
    assert result["success"]
    assert result["mode"] == DedupMode.EXACT_PARAGRAPH

    # Verify counters: 16 total paragraphs across 11 docs, 5 dups, 11 unique
    assert result["dedup/exact/paragraph/total"] == 16
    assert result["dedup/exact/paragraph/dups"] == 5
    assert result["dedup/exact/paragraph/unique"] == 11

    # Load outputs keyed by output file stem (mirrors input shard names)
    by_file = load_dedup_parquet_outputs(fox_corpus["output_dir"])

    # Output files should mirror the input shard structure
    assert "test_shard_0" in by_file
    assert "test_shard_1" in by_file

    # Flatten all records and index by doc id
    all_records = [r for records in by_file.values() for r in records]
    by_doc = {r["id"]: r for r in all_records}

    # Only docs with duplicate (non-canonical) paragraphs appear in output.
    # test_gray_dup_1 is canonical (first by id sort) so it should NOT appear.
    assert "test_gray_dup_1" not in by_doc

    # test_gray_dup_2 and test_gray_dup_3 are non-canonical, each has 2 duplicate paragraphs
    for dup_id in ["test_gray_dup_2", "test_gray_dup_3"]:
        assert dup_id in by_doc
        assert len(by_doc[dup_id]["dup_spans"]) == 2

    # test_gray_partial shares first paragraph with dup_1/2/3 (non-canonical)
    assert "test_gray_partial" in by_doc
    assert len(by_doc["test_gray_partial"]["dup_spans"]) == 1

    # Unique docs should not appear in output
    assert "test_unique_1" not in by_doc

    # Verify output file placement: dup_2/3/partial are from shard_0 input
    shard_0_ids = {r["id"] for r in by_file["test_shard_0"]}
    assert {"test_gray_dup_2", "test_gray_dup_3", "test_gray_partial"} <= shard_0_ids
