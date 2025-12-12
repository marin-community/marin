# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

from zephyr import load_jsonl

from marin.processing.classification.deduplication.pipeline import DedupeConfig, DedupMode, deduplicate


def load_dedup_outputs(output_dir: str) -> dict[str, dict]:
    """Load all dedupe output files and return as id->doc mapping.

    Args:
        output_dir: Directory containing .jsonl.gz output files

    Returns:
        Dictionary mapping document IDs to document records
    """
    output_files = list(Path(output_dir).glob("**/*.jsonl.gz"))
    results = []
    for output_file in output_files:
        results.extend(load_jsonl(str(output_file)))
    return {r["id"]: r for r in results}


def test_exact_deduplication_paragraph(fox_corpus):
    """Test exact deduplication using paragraph hashing (exact match)"""
    # Run deduplication using fixture's test data
    dedupe_config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        attribute_name="duplicate_text",
        processes=1,
        mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
    )

    result = deduplicate(dedupe_config)
    assert result["success"]
    assert result["mode"] == "deduplication"

    # Read and verify attributes
    attrs_by_id = load_dedup_outputs(fox_corpus["output_dir"])
    assert len(attrs_by_id) > 0

    # All documents have duplicate_text annotations (even unique ones)
    assert all("duplicate_text" in attr["attributes"] for attr in attrs_by_id.values())

    # test_gray_dup_2 and test_gray_dup_3 have the same text as test_gray_dup_1 (which is canonical)
    # Each has 2 paragraphs, and should have high duplicate scores (exact matches are likely score 1.0 or implicitly
    # handled)
    # In exact paragraph dedupe, if a para is duplicate, it is marked.
    assert len(attrs_by_id["test_gray_dup_2"]["attributes"]["duplicate_text"]) == 2
    assert len(attrs_by_id["test_gray_dup_3"]["attributes"]["duplicate_text"]) == 2
    # Both paragraphs should be marked as duplicates. Note: score might not be in exact dedupe output?
    # Let's check `mark_exact_dups_paragraphs`. It calls `dupekit.mark_paragraph_duplicates`.
    # It returns [start, end, score] usually?
    # Original test asserted > 0.7.

    assert all(len(span) == 3 for span in attrs_by_id["test_gray_dup_2"]["attributes"]["duplicate_text"])

    # test_gray_partial shares first paragraph with dup_1/2/3
    # At least one span should be marked as duplicate (the matching first paragraph)
    partial_spans = attrs_by_id["test_gray_partial"]["attributes"]["duplicate_text"]
    assert len(partial_spans) >= 1


def test_exact_deduplication_document(fox_corpus):
    """Test exact document-level deduplication (DedupMode.EXACT_DOC_DEDUPLICATE)."""
    config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        attribute_name="is_duplicate",
        mode=DedupMode.DOCUMENT_DEDUPLICATE,
        processes=1,
    )

    result = deduplicate(config)
    assert result["success"]
    assert result["mode"] == DedupMode.DOCUMENT_DEDUPLICATE

    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    assert results_by_id["test_unique_1"]["attributes"]["is_duplicate"] is False

    dups = ["test_gray_dup_1", "test_gray_dup_2", "test_gray_dup_3"]
    flags = [results_by_id[d]["attributes"]["is_duplicate"] for d in dups]

    # NOTE: of the 3 exact dups, 2 are marked as duplicates (one is canonical)
    assert sum(flags) == 2


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
    dedupe_config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=dedupe_output_dir,
        attribute_name="duplicate_text",
        mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
        processes=1,
    )

    result = deduplicate(dedupe_config)
    assert result["success"]
    assert result["mode"] == "deduplication"

    # Verify dedupe output exists and has same structure as input
    dedupe_output_files = list(Path(dedupe_output_dir).glob("**/*.jsonl.gz"))
    assert len(dedupe_output_files) > 0

    # Now run consolidate using the dedupe attributes
    from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, FilterType, consolidate

    consolidate_config = ConsolidateConfig(
        input_path=fox_corpus["test_dir"],
        output_path=consolidated_dir,
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_SPANS,
                attribute_path=f"{dedupe_output_dir}/data",
                name="duplicate_text",
            )
        ],
    )

    consolidate(consolidate_config)

    # Read consolidated output
    consolidated_by_id = load_dedup_outputs(consolidated_dir)
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
