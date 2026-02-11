# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.processing.classification.deduplication.dedup_commons import DedupMode, DedupConfig, deduplicate
from tests.processing.classification.deduplication.conftest import load_dedup_outputs


def test_fuzzy_document_deduplication(fox_corpus):
    config = DedupConfig(
        input_paths=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        mode=DedupMode.FUZZY_DOCUMENT,
        processes=1,
    )

    result = deduplicate(config)
    assert result["success"]
    assert result["mode"] == DedupMode.FUZZY_DOCUMENT

    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    exact_dups = ["test_gray_dup_1", "test_gray_dup_2", "test_gray_dup_3"]
    fuzzy_dup_flags = [results_by_id[d]["attributes"].get(DedupMode.FUZZY_DOCUMENT, False) for d in exact_dups]

    # NOTE: 2 should be marked as fuzzy duplicates as well
    assert sum(fuzzy_dup_flags) == 2

    fuzzy_dups = ["test_contaminated_1", "test_contaminated_2"]

    fuzzy_dup_flags = [results_by_id[d]["attributes"].get(DedupMode.FUZZY_DOCUMENT, False) for d in fuzzy_dups]
    # NOTE: one is marked as fuzzy duplicate, one is canonical
    assert sum(fuzzy_dup_flags) == 1
