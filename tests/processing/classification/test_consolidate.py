# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    consolidate,
)
from marin.processing.classification.deduplication.dedup_commons import DedupMode
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from zephyr.readers import load_parquet


def test_dedupe_consolidate_integration(fox_corpus):
    """Integration test: dedupe generates attributes, consolidate filters based on them.

    This test verifies that:
    1. Dedupe outputs files with names matching input files (using rebase_file_path)
    2. Consolidate can find and use those attribute files
    3. The end-to-end workflow removes duplicate content
    """
    dedupe_output_dir = fox_corpus["output_dir"]
    consolidated_dir = os.path.join(fox_corpus["output_dir"], "consolidation")

    # Run deduplication to generate attributes using fixture's test data
    result = dedup_exact_paragraph(
        input_paths=fox_corpus["test_dir"],
        output_path=dedupe_output_dir,
        max_parallelism=4,
    )
    assert result["success"]
    assert result["mode"] == DedupMode.EXACT_PARAGRAPH

    # Verify dedupe output exists and has same structure as input
    dedupe_output_files = list(Path(dedupe_output_dir).glob("data/*.parquet"))
    assert len(dedupe_output_files) > 0

    # Now run consolidate using the dedupe attributes
    consolidate(
        input_path=fox_corpus["test_dir"],
        output_path=consolidated_dir,
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_SPANS,
                attribute_path=f"{dedupe_output_dir}/data",
                name="dup_spans",
                attribute_filetype="parquet",
                keep_if_missing=True,
            )
        ],
    )

    # Read consolidated output from all parquet shards
    consolidated_rows = []
    for pq_file in sorted(Path(consolidated_dir).glob("*.parquet")):
        consolidated_rows.extend(load_parquet(str(pq_file)))
    consolidated_by_id = {row["id"]: row for row in consolidated_rows}
    assert len(consolidated_by_id) > 0

    # Verify that duplicate spans have been removed
    test_docs = fox_corpus["test"]
    assert len(consolidated_by_id) == len(test_docs)

    # test_gray_dup_1 is canonical (first occurrence), should be unchanged
    original_dup_1 = next(d["text"] for d in test_docs if d["id"] == "test_gray_dup_1")
    assert consolidated_by_id["test_gray_dup_1"]["text"] == original_dup_1

    # test_gray_dup_2 and test_gray_dup_3 had all content marked as duplicate
    # text should be empty or nearly empty
    dup_2_text = consolidated_by_id["test_gray_dup_2"]["text"].strip()
    dup_3_text = consolidated_by_id["test_gray_dup_3"]["text"].strip()
    original_text = next(d["text"] for d in test_docs if d["id"] == "test_gray_dup_2")
    assert (
        len(dup_2_text) < len(original_text) / 2
    ), f"Expected test_gray_dup_2 to have most content removed, but got: {dup_2_text}"
    assert (
        len(dup_3_text) < len(original_text) / 2
    ), f"Expected test_gray_dup_3 to have most content removed, but got: {dup_3_text}"

    # Unique documents should be unchanged
    for doc_id in ["test_unique_1", "test_unique_2", "test_unique_3"]:
        original = next(d["text"] for d in test_docs if d["id"] == doc_id)
        assert consolidated_by_id[doc_id]["text"] == original
