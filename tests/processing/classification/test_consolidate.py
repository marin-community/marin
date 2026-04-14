# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import os
from pathlib import Path

from marin.processing.classification.deduplication.dedup_commons import DedupMode
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
import pytest
from ddsketch import DDSketch
from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    calculate_percentile_thresholds,
    consolidate,
)
from zephyr.readers import load_parquet


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_calculate_percentile_threshold(tmp_path):
    documents_dir = tmp_path / "documents"
    attributes_dir = tmp_path / "attributes"
    documents_dir.mkdir()
    attributes_dir.mkdir()

    attribute_rows = [
        [
            {"id": "doc-0", "attributes": {"quality": {"good": 0.1}}},
            {"id": "doc-1", "attributes": {"quality": {"good": 0.4}}},
        ],
        [
            {"id": "doc-2", "attributes": {"quality": {"good": 0.7}}},
            {"id": "doc-3", "attributes": {"quality": {"good": 0.9}}},
        ],
    ]

    for shard_index, rows in enumerate(attribute_rows):
        doc_path = documents_dir / f"part-{shard_index}.jsonl"
        doc_path.write_text("{}", encoding="utf-8")
        attr_path = attributes_dir / f"part-{shard_index}.jsonl"
        _write_jsonl(attr_path, rows)

    keep_fraction = 0.5
    filters = [
        FilterConfig(
            type=FilterType.CLASSIFY,
            attribute_path=str(attributes_dir),
            name="quality",
            label="good",
            keep_fraction=keep_fraction,
        )
    ]

    updated_filters = calculate_percentile_thresholds(input_path=str(documents_dir), filters=filters, filetype="jsonl")
    threshold = updated_filters[0].lower_threshold

    # Calculate expected threshold
    expected_sketch = DDSketch()
    for shard in attribute_rows:
        for row in shard:
            expected_sketch.add(row["attributes"]["quality"]["good"])
    expected_threshold = expected_sketch.get_quantile_value(1 - keep_fraction)

    assert threshold == pytest.approx(expected_threshold, rel=1e-6)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_consolidate_filters_and_writes_output(tmp_path):
    """Test that consolidate filters documents and writes output using zephyr."""
    input_root = tmp_path / "input"
    attributes_root = tmp_path / "attributes"
    output_root = tmp_path / "output"
    input_root.mkdir()
    attributes_root.mkdir()
    output_root.mkdir()

    input_rows = [
        {"id": "doc-0", "text": "first"},
        {"id": "doc-1", "text": "second"},
        {"id": "doc-2", "text": "third"},
    ]
    attribute_rows = [
        {"id": "doc-0", "attributes": {"quality": {"good": 0.1}}},
        {"id": "doc-1", "attributes": {"quality": {"good": 0.6}}},
        {"id": "doc-2", "attributes": {"quality": {"good": 0.8}}},
    ]

    input_file = input_root / "part-0000.jsonl.gz"
    attribute_file = attributes_root / "part-0000.jsonl.gz"
    _write_jsonl_gz(input_file, input_rows)
    _write_jsonl_gz(attribute_file, attribute_rows)

    consolidate(
        input_path=str(input_root),
        output_path=str(output_root),
        filters=[
            FilterConfig(
                type=FilterType.CLASSIFY,
                attribute_path=str(attributes_root),
                name="quality",
                label="good",
                lower_threshold=0.5,
            )
        ],
    )

    output_file = output_root / "part-0000.parquet"
    assert (
        output_file.exists()
    ), f"Expected consolidated output file to be written. Files in {output_root}: {list(output_root.iterdir())}"

    output_rows = load_parquet(str(output_file))

    kept_ids = {row["id"] for row in output_rows}
    assert kept_ids == {"doc-1", "doc-2"}, f"Expected to keep doc-1 and doc-2, but got {kept_ids}"


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
