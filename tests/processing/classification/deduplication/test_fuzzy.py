# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.processing.classification.deduplication.dedup_commons import DedupMode
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from tests.processing.classification.deduplication.conftest import load_dedup_parquet_outputs


def test_fuzzy_document_deduplication(fox_corpus):
    result = dedup_fuzzy_document(
        input_paths=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        max_parallelism=4,
    )
    assert result["success"]
    assert result["mode"] == DedupMode.FUZZY_DOCUMENT

    # Verify counters: all docs processed, totals are consistent
    total = result["dedup/fuzzy/document/total"]
    dups = result["dedup/fuzzy/document/dups"]
    unique = result["dedup/fuzzy/document/unique"]
    assert total == 11
    assert dups + unique == total
    # At least 2 gray dups + 1 fuzzy dup (contaminated/high_overlap cluster)
    assert dups >= 3

    by_file = load_dedup_parquet_outputs(fox_corpus["output_dir"] + "/data")

    all_records = [r for records in by_file.values() for r in records]
    by_doc = {r["id"]: r for r in all_records}

    # Of the 3 exact-duplicate gray docs, 2 should be marked as fuzzy dups (1 is canonical)
    exact_dups = ["test_gray_dup_1", "test_gray_dup_2", "test_gray_dup_3"]
    assert sum(1 for d in exact_dups if d in by_doc) == 2

    # All output records should have dup_doc=True
    for record in all_records:
        assert record["attributes"]["dup_doc"] is True

    # Of the two contaminated (fuzzy-similar) docs, one is canonical and one is a dup
    fuzzy_dups = ["test_contaminated_1", "test_contaminated_2"]
    assert sum(1 for d in fuzzy_dups if d in by_doc) == 1

    # Unique docs should not appear in output
    assert "test_unique_1" not in by_doc
    assert "test_unique_2" not in by_doc
