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
from marin.utils import fsspec_exists

from marin.processing.classification.decon import DeconConfig, DeconMode, NGramConfig, decontaminate


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


def test_decontamination(fox_corpus):
    """Test basic decontamination workflow"""
    # Run decontamination with multiple processes
    config = DeconConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="contaminated",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        mode=DeconMode.DECONTAMINATE,
        processes=2,
    )

    result = decontaminate(config)
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
    config = DeconConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="overlap",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        ngram=NGramConfig(ngram_length=3, stride=0, overlap_threshold=0.5),
        mode=DeconMode.DECONTAMINATE,
        processes=1,
    )

    result = decontaminate(config)
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
    config = DeconConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="overlap",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        ngram=NGramConfig(ngram_length=[3, 5], stride=0, overlap_threshold=0.0),  # Show all overlaps
        mode=DeconMode.TRAIN_TEST_OVERLAP,
        processes=1,
    )

    result = decontaminate(config)
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
    config = DeconConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="contaminated",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        mode=DeconMode.DECONTAMINATE,
        processes=1,
    )

    result = decontaminate(config)
    assert result["success"]
    assert result["mode"] == "decontamination"

    # Read output
    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    # test_para_match: second paragraph is contaminated (matches train_arctic_2's second para)
    assert len(results_by_id["test_para_match"]["attributes"]["contaminated"]) == 1
    assert results_by_id["test_para_match"]["attributes"]["contaminated"][0][2] == 1.0  # Second paragraph (match!)

    # test_unique_1: single paragraph, all clean (no contamination entries)
    assert len(results_by_id["test_unique_1"]["attributes"]["contaminated"]) == 0
