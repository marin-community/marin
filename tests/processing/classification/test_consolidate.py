# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    consolidate,
)
from zephyr.readers import load_parquet

_SPAN_ATTR_SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("dup_spans", pa.list_(pa.list_(pa.int64())))])


def _line_spans(text: str) -> list[list[int]]:
    """Return ``[start, end)`` character spans for each newline-separated line of ``text``."""
    spans = []
    start = 0
    for line in text.split("\n"):
        spans.append([start, start + len(line)])
        start += len(line) + 1
    return spans


def test_consolidate_remove_spans(fox_corpus):
    """REMOVE_SPANS cuts the attributed spans out of matching documents and leaves the rest alone."""
    test_docs = {doc["id"]: doc["text"] for doc in fox_corpus["test"]}
    input_dir = os.path.join(fox_corpus["output_dir"], "input")
    attr_dir = os.path.join(fox_corpus["output_dir"], "spans")
    consolidated_dir = os.path.join(fox_corpus["output_dir"], "consolidation")

    # consolidate merge-joins each input shard with its attribute shard, so both
    # must be sorted by id. Every input shard needs an attribute file, empty
    # where nothing is removed.
    os.makedirs(input_dir)
    os.makedirs(attr_dir)
    sorted_docs = sorted(fox_corpus["test"], key=lambda doc: doc["id"])
    pq.write_table(pa.Table.from_pylist(sorted_docs[:6]), os.path.join(input_dir, "test_shard_0.parquet"))
    pq.write_table(pa.Table.from_pylist(sorted_docs[6:]), os.path.join(input_dir, "test_shard_1.parquet"))
    shard_0_rows = [
        {"id": "test_gray_dup_2", "dup_spans": _line_spans(test_docs["test_gray_dup_2"])},
        {"id": "test_gray_dup_3", "dup_spans": _line_spans(test_docs["test_gray_dup_3"])},
        {"id": "test_gray_partial", "dup_spans": _line_spans(test_docs["test_gray_partial"])[:1]},
    ]
    pq.write_table(
        pa.Table.from_pylist(shard_0_rows, schema=_SPAN_ATTR_SCHEMA), os.path.join(attr_dir, "test_shard_0.parquet")
    )
    pq.write_table(_SPAN_ATTR_SCHEMA.empty_table(), os.path.join(attr_dir, "test_shard_1.parquet"))

    consolidate(
        input_path=input_dir,
        output_path=consolidated_dir,
        filetype="parquet",
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_SPANS,
                attribute_path=attr_dir,
                name="dup_spans",
                attribute_filetype="parquet",
                keep_if_missing=True,
            )
        ],
    )

    consolidated_rows = []
    for pq_file in sorted(Path(consolidated_dir).glob("*.parquet")):
        consolidated_rows.extend(load_parquet(str(pq_file)))
    consolidated_by_id = {row["id"]: row["text"] for row in consolidated_rows}
    assert consolidated_by_id.keys() == test_docs.keys()

    # The first copy has no attribute row and stays unchanged.
    assert consolidated_by_id["test_gray_dup_1"] == test_docs["test_gray_dup_1"]

    # Later copies lose every attributed span; only the line separator is left.
    assert consolidated_by_id["test_gray_dup_2"] == "\n"
    assert consolidated_by_id["test_gray_dup_3"] == "\n"
    partial_lines = test_docs["test_gray_partial"].split("\n")
    assert consolidated_by_id["test_gray_partial"] == "\n" + partial_lines[1]

    # Documents without spans stay intact.
    for doc_id in ["test_unique_1", "test_unique_2", "test_unique_3"]:
        assert consolidated_by_id[doc_id] == test_docs[doc_id]
