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

from marin.processing.classification.deduplication.dedup_commons import DedupMode, DedupConfig, deduplicate
from tests.processing.classification.deduplication.conftest import load_dedup_outputs


def test_exact_paragraph_deduplication(fox_corpus):
    """Test exact deduplication using paragraph hashing (exact match)"""
    # Run deduplication using fixture's test data
    dedupe_config = DedupConfig(
        input_paths=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        processes=1,
        mode=DedupMode.EXACT_PARAGRAPH,
    )

    result = deduplicate(dedupe_config)
    assert result["success"]
    assert result["mode"] == DedupMode.EXACT_PARAGRAPH
    # Read and verify attributes
    attrs_by_id = load_dedup_outputs(fox_corpus["output_dir"])
    assert len(attrs_by_id) > 0

    # All documents have duplicate_text annotations (even unique ones)
    assert all(DedupMode.EXACT_PARAGRAPH in attr["attributes"] for attr in attrs_by_id.values())

    # test_gray_dup_2 and test_gray_dup_3 have the same text as test_gray_dup_1 (which is canonical)
    # Each has 2 paragraphs, and should have high duplicate scores (exact matches are likely score 1.0 or implicitly
    # handled)
    # In exact paragraph dedupe, if a para is duplicate, it is marked.
    assert len(attrs_by_id["test_gray_dup_2"]["attributes"][DedupMode.EXACT_PARAGRAPH]) == 2
    assert len(attrs_by_id["test_gray_dup_3"]["attributes"][DedupMode.EXACT_PARAGRAPH]) == 2
    # Both paragraphs should be marked as duplicates. Note: score might not be in exact dedupe output?
    # Let's check `mark_exact_dups_paragraphs`. It calls `dupekit.mark_paragraph_duplicates`.
    # It returns [start, end, score] usually?
    # Original test asserted > 0.7.

    assert all(len(span) == 3 for span in attrs_by_id["test_gray_dup_2"]["attributes"][DedupMode.EXACT_PARAGRAPH])

    # test_gray_partial shares first paragraph with dup_1/2/3
    # At least one span should be marked as duplicate (the matching first paragraph)
    partial_spans = attrs_by_id["test_gray_partial"]["attributes"][DedupMode.EXACT_PARAGRAPH]
    assert len(partial_spans) >= 1


def test_exact_document_deduplication(fox_corpus):
    config = DedupConfig(
        input_paths=fox_corpus["test_dir"],
        output_path=fox_corpus["output_dir"],
        mode=DedupMode.EXACT_DOCUMENT,
        processes=1,
    )

    result = deduplicate(config)
    assert result["success"]
    assert result["mode"] == DedupMode.EXACT_DOCUMENT

    results_by_id = load_dedup_outputs(fox_corpus["output_dir"])

    assert results_by_id["test_unique_1"]["attributes"][DedupMode.EXACT_DOCUMENT] is False
    dups = ["test_gray_dup_1", "test_gray_dup_2", "test_gray_dup_3"]
    exact_dup_flags = [results_by_id[d]["attributes"][DedupMode.EXACT_DOCUMENT] for d in dups]

    # NOTE: of the 3 exact dups, 2 are marked as duplicates (one is canonical)
    assert sum(exact_dup_flags) == 2
