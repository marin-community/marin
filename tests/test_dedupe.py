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
import tempfile
from pathlib import Path

import pytest
from marin.utils import fsspec_exists
from zephyr import load_jsonl, write_jsonl_file

from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe


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


@pytest.fixture
def fox_corpus():
    """Realistic fox-themed corpus with natural duplication patterns.

    Returns:
        dict with 'train', 'test', 'train_dir', 'test_dir', and 'output_dir' keys
    """
    train = [
        {"id": "train_red_1", "text": "Red canids inhabit northern territories worldwide.", "source": "train"},
        {
            "id": "train_arctic_1",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath snow.",
            "source": "train",
        },
        {
            "id": "train_arctic_2",
            "text": (
                "The pale arctic canid transforms appearance seasonally.\n"
                "Its alabaster winter fur enables stealth in frozen landscapes."
            ),
            "source": "train",
        },
        {
            "id": "train_kit_1",
            "text": "Newborn kits emerge sightless and vulnerable.\nThey remain sheltered underground for many days.",
            "source": "train",
        },
        {"id": "train_diet_1", "text": "These carnivores consume various rodents and vegetation.", "source": "train"},
        # Duplicate of train_red_1
        {"id": "train_red_dup", "text": "Red canids inhabit northern territories worldwide.", "source": "train"},
        # Partial duplicate - shares second paragraph with train_kit_1
        {
            "id": "train_kit_partial",
            "text": (
                "Juvenile animals mature rapidly during springtime.\nThey remain sheltered underground for many days."
            ),
            "source": "train",
        },
    ]

    test = [
        # Exact duplicates of each other (for deduplication testing)
        {
            "id": "test_gray_dup_1",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
            "source": "test",
        },
        {
            "id": "test_gray_dup_2",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
            "source": "test",
        },
        {
            "id": "test_gray_dup_3",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "They frequently perch on elevated branches throughout daylight hours."
            ),
            "source": "test",
        },
        # Partial duplicate (shares first paragraph with dup_1/2/3)
        {
            "id": "test_gray_partial",
            "text": (
                "Gray climbing specialists ascend vegetation using retractable talons.\n"
                "Unlike crimson variants, they inhabit densely forested regions."
            ),
            "source": "test",
        },
        # Exact contamination - matches train_arctic_1
        {
            "id": "test_contaminated_1",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath snow.",
            "source": "test",
        },
        # Exact contamination - matches train_red_1
        {"id": "test_contaminated_2", "text": "Red canids inhabit northern territories worldwide.", "source": "test"},
        # High n-gram overlap with train_arctic_1 (minor word change)
        {
            "id": "test_high_overlap",
            "text": "Arctic predators have superior auditory capabilities for hunting beneath thick snow.",
            "source": "test",
        },
        # Partial paragraph match with train_arctic_2 (shares second paragraph)
        {
            "id": "test_para_match",
            "text": (
                "Polar mammals thrive in extreme frigid conditions.\n"
                "Its alabaster winter fur enables stealth in frozen landscapes."
            ),
            "source": "test",
        },
        # No overlap at all - completely different vocabulary
        {
            "id": "test_unique_1",
            "text": "Desert mammals possess oversized pinnae for thermal regulation.",
            "source": "test",
        },
        {"id": "test_unique_2", "text": "Rapid runners represent the most diminutive wild dogs.", "source": "test"},
        {
            "id": "test_unique_3",
            "text": "Isolated populations exist exclusively on Pacific archipelagos.",
            "source": "test",
        },
    ]

    with (
        tempfile.TemporaryDirectory() as train_dir,
        tempfile.TemporaryDirectory() as test_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        # Write train data across multiple shards
        for i, shard_docs in enumerate([train[:4], train[4:]]):
            train_file = os.path.join(train_dir, f"train_shard_{i}.jsonl.gz")
            write_jsonl_file(shard_docs, train_file)

        # Write test data across multiple shards
        for i, shard_docs in enumerate([test[:6], test[6:]]):
            test_file = os.path.join(test_dir, f"test_shard_{i}.jsonl.gz")
            write_jsonl_file(shard_docs, test_file)

        yield {
            "train": train,
            "test": test,
            "train_dir": train_dir,
            "test_dir": test_dir,
            "output_dir": output_dir,
        }


def test_decontamination(fox_corpus):
    """Test basic decontamination workflow"""
    # Run decontamination with multiple processes
    config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="contaminated",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        mode=DedupMode.DECONTAMINATE,
        processes=2,
    )

    result = dedupe(config)
    assert result["success"]
    assert result["mode"] == "decontamination"

    # Read output
    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])
    assert len(results_by_id) > 0

    # test_contaminated_1 is contaminated (exact match with train_arctic_1)
    assert len(results_by_id["test_contaminated_1"]["attributes"]["contaminated"]) == 1
    assert results_by_id["test_contaminated_1"]["attributes"]["contaminated"][0][2] == 1.0

    # test_contaminated_2 is contaminated (exact match with train_red_1 and train_red_dup)
    assert len(results_by_id["test_contaminated_2"]["attributes"]["contaminated"]) == 1
    assert results_by_id["test_contaminated_2"]["attributes"]["contaminated"][0][2] == 1.0

    # test_unique_1 is clean (no contamination entries)
    assert len(results_by_id["test_unique_1"]["attributes"]["contaminated"]) == 0


def test_ngram_decontamination(fox_corpus):
    """Test n-gram based decontamination"""
    # Run decontamination with n-grams
    config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="overlap",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        ngram=NGramConfig(ngram_length=3, stride=0, overlap_threshold=0.5),
        mode=DedupMode.DECONTAMINATE,
        processes=1,
    )

    result = dedupe(config)
    assert result["success"]
    assert result["mode"] == "decontamination"

    # Read output
    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    # test_high_overlap has high overlap (>50% of 3-grams match with train_arctic_1)
    assert len(results_by_id["test_high_overlap"]["attributes"]["overlap"]) == 1
    assert results_by_id["test_high_overlap"]["attributes"]["overlap"][0][2] > 0.5

    # test_unique_2 has low/no overlap (less than the high overlap case)
    assert len(results_by_id["test_unique_2"]["attributes"]["overlap"]) == 1
    high_overlap_score = results_by_id["test_high_overlap"]["attributes"]["overlap"][0][2]
    unique_overlap_score = results_by_id["test_unique_2"]["attributes"]["overlap"][0][2]
    assert unique_overlap_score < high_overlap_score


def test_train_test_overlap(fox_corpus):
    """Test train-test overlap with multiple n-gram sizes"""
    # Run train-test overlap with multiple n-gram sizes
    config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="overlap",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        ngram=NGramConfig(ngram_length=[3, 5], stride=0, overlap_threshold=0.0),  # Show all overlaps
        mode=DedupMode.TRAIN_TEST_OVERLAP,
        processes=1,
    )

    result = dedupe(config)
    assert result["success"]
    assert result["mode"] == "train_test_overlap"
    assert result["ngram_lengths_processed"] == [3, 5]

    # Check outputs for each n-gram size
    for ngram_len in [3, 5]:
        ngram_dir = os.path.join(fox_corpus["output_dir"], str(ngram_len))
        assert fsspec_exists(ngram_dir)

        results_by_id = load_dedup_outputs(ngram_dir)
        assert len(results_by_id) > 0

        # test_high_overlap should have some overlap with train
        assert len(results_by_id["test_high_overlap"]["attributes"][f"overlap_{ngram_len}"]) == 1
        high_score = results_by_id["test_high_overlap"]["attributes"][f"overlap_{ngram_len}"][0][2]
        assert high_score > 0.0

        # test_unique_1 should have much less overlap than test_high_overlap
        # (may have small overlap due to common words, but significantly less)
        unique_attrs = results_by_id["test_unique_1"]["attributes"][f"overlap_{ngram_len}"]
        if len(unique_attrs) > 0:
            unique_score = unique_attrs[0][2]
            assert unique_score < high_score, f"Expected unique ({unique_score}) < high overlap ({high_score})"


def test_multi_paragraph_decontamination(fox_corpus):
    """Test decontamination with multi-paragraph documents"""
    # Run decontamination (exact paragraph matching)
    config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="contaminated",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        mode=DedupMode.DECONTAMINATE,
        processes=1,
    )

    result = dedupe(config)
    assert result["success"]
    assert result["mode"] == "decontamination"

    # Read output
    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    # test_para_match: second paragraph is contaminated (matches train_arctic_2's second para)
    assert len(results_by_id["test_para_match"]["attributes"]["contaminated"]) == 1
    assert results_by_id["test_para_match"]["attributes"]["contaminated"][0][2] == 1.0  # Second paragraph (match!)

    # test_unique_1: single paragraph, all clean (no contamination entries)
    assert len(results_by_id["test_unique_1"]["attributes"]["contaminated"]) == 0


def test_exact_deduplication_paragraph(fox_corpus):
    """Test exact deduplication using n-gram matching"""
    # Run deduplication using fixture's test data
    dedupe_config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        attribute_name="duplicate_text",
        min_length=0,
        min_words=0,
        estimated_doc_count=100,
        false_positive_rate=0.001,
        ngram=NGramConfig(
            ngram_length=5,
            stride=0,
            overlap_threshold=0.7,
        ),
        mode=DedupMode.DEDUPLICATE,
    )

    result = dedupe(dedupe_config)
    assert result["success"]
    assert result["mode"] == "deduplication"

    # Read and verify attributes
    attrs_by_id = load_dedup_outputs(fox_corpus["output_dir"])
    assert len(attrs_by_id) > 0

    # All documents have duplicate_text annotations (even unique ones)
    assert all("duplicate_text" in attr["attributes"] for attr in attrs_by_id.values())

    # test_gray_dup_2 and test_gray_dup_3 have the same text as test_gray_dup_1 (which is canonical)
    # Each has 2 paragraphs, and should have high duplicate scores
    assert len(attrs_by_id["test_gray_dup_2"]["attributes"]["duplicate_text"]) == 2
    assert len(attrs_by_id["test_gray_dup_3"]["attributes"]["duplicate_text"]) == 2
    # Both paragraphs should be marked as duplicates with high scores
    assert all(span[2] > 0.7 for span in attrs_by_id["test_gray_dup_2"]["attributes"]["duplicate_text"])
    assert all(span[2] > 0.7 for span in attrs_by_id["test_gray_dup_3"]["attributes"]["duplicate_text"])

    # test_gray_partial shares first paragraph with dup_1/2/3
    # At least one span should be marked as duplicate (the matching first paragraph)
    partial_spans = attrs_by_id["test_gray_partial"]["attributes"]["duplicate_text"]
    assert len(partial_spans) >= 1
    assert any(span[2] > 0.5 for span in partial_spans)


def test_exact_deduplication_document(fox_corpus):
    """Test exact document-level deduplication (DedupMode.EXACT_DOC_DEDUPLICATE)."""
    config = DedupeConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        attribute_name="is_duplicate",
        mode=DedupMode.EXACT_DOC_DEDUPLICATE,
        processes=1,
    )

    result = dedupe(config)
    assert result["success"]
    assert result["mode"] == "exact_doc_deduplication"

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
        estimated_doc_count=10,
        false_positive_rate=0.01,
        mode=DedupMode.DEDUPLICATE,
        processes=1,
    )

    result = dedupe(dedupe_config)
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
