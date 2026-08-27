# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for exploding the oracle sample into per-segment training labels.

``segment_rows`` decides which rows the scorer trains on, and both of its drops are silent in the
output metrics: a partially graded document must vanish entirely rather than train one segment on
a real label and another on a sentinel, and a short document's three duplicate windows must mark
only one row or every ungrouped split leaks near-duplicates across train/eval.
"""

import pyarrow as pa

from experiments.datakit.build_pdf_source.quality.build_labels import segment_rows
from experiments.datakit.build_pdf_source.quality.build_oracle_sample import (
    MIN_TOKENS_FOR_ALL_SEGMENTS,
    SCORE_COLUMNS,
    SEGMENT_TEXT_COLUMNS,
    SEGMENTS,
)


def _sample_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def _doc(doc_id: str, doc_tokens: int, scores: dict[str, int | None]) -> dict:
    row = {"id": doc_id, "source": "test", "doc_tokens": doc_tokens, "needs_ocr": False}
    for segment in SEGMENTS:
        row[SEGMENT_TEXT_COLUMNS[segment]] = f"{doc_id} {segment} text"
        row[SCORE_COLUMNS[segment]] = scores[segment]
    return row


def test_a_document_with_a_null_score_contributes_no_rows():
    """A partially graded document must never train one segment and sentinel another."""
    table = _sample_table(
        [
            _doc("graded", 5000, {"begin": 3, "middle": 2, "end": 1}),
            _doc("partial", 5000, {"begin": 3, "middle": None, "end": 1}),
            _doc("failed", 5000, {"begin": 3, "middle": -1, "end": 1}),
        ]
    )

    rows = segment_rows(table)

    assert rows.num_rows == len(SEGMENTS)
    assert set(rows["id"].to_pylist()) == {"graded"}


def test_a_short_document_marks_only_its_begin_row_for_training():
    """Below three windows the segments are literal duplicates; training on all three would leak."""
    table = _sample_table([_doc("short", MIN_TOKENS_FOR_ALL_SEGMENTS - 1, {"begin": 2, "middle": 2, "end": 2})])

    rows = segment_rows(table)

    by_segment = dict(zip(rows["segment"].to_pylist(), rows["use_for_training"].to_pylist(), strict=True))
    assert by_segment == {"begin": True, "middle": False, "end": False}
    # The unmarked rows still carry their text and score, so a scorer can be applied to them.
    assert all(rows["score_normalized"].to_pylist())


def test_a_long_document_trains_on_all_three_segments():
    table = _sample_table([_doc("long", MIN_TOKENS_FOR_ALL_SEGMENTS, {"begin": 0, "middle": 4, "end": 1})])

    rows = segment_rows(table)

    assert rows["use_for_training"].to_pylist() == [True] * len(SEGMENTS)
    # quality is the oracle level 1..5; score_normalized spans the observed 0..4 range.
    by_segment = dict(zip(rows["segment"].to_pylist(), rows["quality"].to_pylist(), strict=True))
    assert by_segment == {"begin": 1, "middle": 5, "end": 2}
