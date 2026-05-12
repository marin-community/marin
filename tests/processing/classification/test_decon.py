# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from marin.processing.classification.decon import DeconConfig, NGramConfig, decontaminate
from zephyr import load_jsonl


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
        processes=2,
    )

    result = decontaminate(config)
    assert result["success"]

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
    """N-gram decontamination flags high-overlap paragraphs and gates low-overlap ones via overlap_threshold."""
    config = DeconConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="overlap",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        ngram=NGramConfig(ngram_length=3, stride=0, overlap_threshold=0.5),
        processes=1,
    )

    result = decontaminate(config)
    assert result["success"]

    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    # test_high_overlap matches train_arctic_1 at >50% of 3-grams → recorded.
    high_spans = results_by_id["test_high_overlap"]["attributes"]["overlap"]
    assert len(high_spans) == 1
    assert high_spans[0][2] >= 0.5

    # test_unique_2 has near-zero overlap (different vocabulary) → gated out by threshold=0.5.
    assert results_by_id["test_unique_2"]["attributes"]["overlap"] == []


@pytest.mark.parametrize(
    "threshold, expect_high_flagged",
    [(0.0, True), (0.5, True), (0.85, False), (1.0, False)],
)
def test_overlap_threshold_gates_spans(fox_corpus, threshold, expect_high_flagged):
    """Regression: overlap_threshold gates which paragraphs get recorded as contaminated spans.

    Before #5519, NGramConfig.overlap_threshold was configured but ignored — any non-zero
    score was recorded. This test pins the gating semantics so the bug can't regress.
    """
    config = DeconConfig(
        input_path=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        decontaminate_source=fox_corpus["train_dir"],
        attribute_name="overlap",
        estimated_doc_count=20,
        false_positive_rate=0.01,
        ngram=NGramConfig(ngram_length=3, stride=0, overlap_threshold=threshold),
        processes=1,
    )
    decontaminate(config)
    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])
    spans = results_by_id["test_high_overlap"]["attributes"]["overlap"]
    if expect_high_flagged:
        assert spans, f"expected test_high_overlap flagged at threshold={threshold}"
        assert spans[0][2] >= threshold
    else:
        assert spans == [], f"expected test_high_overlap gated out at threshold={threshold}"


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
        processes=1,
    )

    result = decontaminate(config)
    assert result["success"]

    # Read output
    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    # test_para_match: second paragraph is contaminated (matches train_arctic_2's second para)
    assert len(results_by_id["test_para_match"]["attributes"]["contaminated"]) == 1
    assert results_by_id["test_para_match"]["attributes"]["contaminated"][0][2] == 1.0  # Second paragraph (match!)

    # test_unique_1: single paragraph, all clean (no contamination entries)
    assert len(results_by_id["test_unique_1"]["attributes"]["contaminated"]) == 0
