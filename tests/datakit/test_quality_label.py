# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quality-labeling the clean all-routes OCR corpus.

The step's contracts: the window cut is exactly the deployed ``score_bme`` convention (the
calibration was fit on scores produced by it), the 0-4 scale exists only through the calibration
knots, batched windows regroup to the documents they were cut from, the ``edu_max`` gate drops a
document only when no window reaches the threshold, and the output stays co-partitioned with the
id-sorted input. All of it runs against a deterministic fake scorer -- no model, no network.
"""

from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.build_pdf_source import quality_label as quality_module
from experiments.build_pdf_source.extract_ocr import _OUTPUT_SCHEMA
from experiments.build_pdf_source.quality_label import (
    QUALITY_SCHEMA,
    DocumentScores,
    calibrate,
    cut_windows,
    keep_document,
    score_shard,
    score_texts,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import CHUNK_CHARS, PooledScorer, score_bme

_IDENTITY_XK = np.array([0.0, 1.0])
_IDENTITY_YK = np.array([0.0, 1.0])


class _FakeScorer:
    """Deterministic stand-in for ``PooledScorer``: ``score(texts)`` returns a raw sigmoid score
    per window keyed on its first character, and records the window lists it was called with."""

    def __init__(self, by_first_char: dict[str, float] | None = None, default: float = 0.0) -> None:
        self._map = by_first_char or {}
        self._default = default
        self.calls: list[list[str]] = []

    def score(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([self._map.get(text[:1], self._default) for text in texts], dtype=np.float32)


def _as_scorer(fake: _FakeScorer) -> PooledScorer:
    return cast(PooledScorer, fake)


def _distinct_text(length: int) -> str:
    """Text whose every window is a distinct substring, so span mistakes are visible."""
    return "".join(str(i % 10) for i in range(length))


def _document(text: str) -> dict:
    """A stored clean-corpus record, shaped like the extraction schema the step reads."""
    page = text if text.endswith("\n") else text + "\n"
    return {
        "id": generate_id(page),
        "text": page,
        "source_id": "crawl-data/CC-MAIN-0001/warc/x.warc.gz:4096",
        "source": "common_crawl_focus_2026_22",
        "warc_filename": "crawl-data/CC-MAIN-0001/warc/x.warc.gz",
        "warc_record_offset": 4096,
        "content_digest": "sha1:ABCDEF",
        "url": "https://example.org/report.pdf",
        "num_pages": 1,
        "page_offsets": [len(page)],
        "extraction_status": "success",
        "extraction_error": None,
        "boilerplate_lines_removed": 0,
        "pages_ocred": 1,
        "pages_failed": 0,
        "pages_truncated": 0,
        "pages_unrendered": 0,
        "mean_render_dpi": 150.0,
        "pages_below_legibility_floor": 0,
        "completion_tokens": 900,
        "looped_pages": [],
        "loop_chars_dropped": 0,
    }


# ---------- cut_windows: parity with the deployed score_bme convention ----------


@pytest.mark.parametrize("length", [10, CHUNK_CHARS - 1, CHUNK_CHARS, CHUNK_CHARS + 1, 3 * CHUNK_CHARS])
def test_cut_windows_scores_exactly_the_windows_score_bme_scores(length):
    """The calibration was fit through this cut, so the windows must match byte for byte --
    including at the boundary: exactly CHUNK_CHARS is one window, one past it is three."""
    text = _distinct_text(length)
    recorder = _FakeScorer()
    score_bme(_as_scorer(recorder), [text])
    assert cut_windows(text) == recorder.calls[0]
    assert len(cut_windows(text)) == (1 if length <= CHUNK_CHARS else 3)


def test_cut_windows_spans_begin_middle_end_at_full_width():
    text = _distinct_text(2 * CHUNK_CHARS + 1)  # 4001 chars, middle at 2000
    begin, middle, end = cut_windows(text)
    assert begin == text[:CHUNK_CHARS]
    assert middle == text[2000 - CHUNK_CHARS // 2 : 2000 + CHUNK_CHARS // 2]
    assert end == text[-CHUNK_CHARS:]


# ---------- calibrate: raw sigmoid -> 0-4 through the knots ----------


def test_calibrate_maps_raw_scores_through_the_knots_onto_the_0_4_scale():
    xk = np.array([0.0, 0.5, 1.0])
    yk = np.array([0.0, 0.8, 1.0])
    raw = np.array([0.0, 0.25, 0.5, 1.0])
    # Hand-computed: 0.25 sits halfway up the first segment (0.4), 0.5 hits the middle knot (0.8).
    assert calibrate(raw, xk, yk).tolist() == pytest.approx([0.0, 1.6, 3.2, 4.0])


# ---------- score_texts: window regrouping and the short-doc path ----------


def test_short_document_carries_its_single_window_score_in_all_three_columns():
    fake = _FakeScorer({"S": 0.5})
    [scores] = score_texts(["S" * 100], _as_scorer(fake), _IDENTITY_XK, _IDENTITY_YK)
    assert scores.edu_begin == scores.edu_middle == scores.edu_end == pytest.approx(2.0)
    assert scores.edu_max == pytest.approx(2.0)


def test_batched_windows_regroup_to_the_documents_they_were_cut_from():
    """Short and long documents interleave, so the flat window stream misaligns with the document
    stream; each document must still get its own windows' scores in begin/middle/end order."""
    fake = _FakeScorer({"S": 0.5, "B": 0.1, "M": 0.2, "E": 0.3, "T": 0.9})
    long_doc = "B" * CHUNK_CHARS + "M" * CHUNK_CHARS + "E" * CHUNK_CHARS
    scores = score_texts(["S" * 50, long_doc, "T" * 50], _as_scorer(fake), _IDENTITY_XK, _IDENTITY_YK)

    assert scores[0].edu_begin == scores[0].edu_middle == scores[0].edu_end == scores[0].edu_max
    assert scores[0].edu_max == pytest.approx(2.0)
    assert scores[1].edu_begin == pytest.approx(0.4)
    assert scores[1].edu_middle == pytest.approx(0.8)
    assert scores[1].edu_end == pytest.approx(1.2)
    assert scores[1].edu_max == pytest.approx(1.2)
    assert scores[2].edu_begin == scores[2].edu_middle == scores[2].edu_end == scores[2].edu_max
    assert scores[2].edu_max == pytest.approx(3.6)


def test_keep_gate_is_inclusive_at_the_threshold():
    below = DocumentScores(edu_begin=0.2, edu_middle=0.99, edu_max=0.99, edu_end=0.5)
    at = DocumentScores(edu_begin=0.2, edu_middle=1.0, edu_max=1.0, edu_end=0.5)
    assert not keep_document(below)
    assert keep_document(at)


# ---------- score_shard: filtering, co-partitioning, and the output schema ----------


def _write_input_shard(tmp_path, rows: list[dict], basename: str) -> str:
    rows = sorted(rows, key=lambda row: row["id"])  # the clean corpus is id-sorted per shard
    input_file = tmp_path / "input" / basename
    input_file.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=_OUTPUT_SCHEMA), str(input_file))
    return str(input_file)


def _patch_loader(monkeypatch, fake: _FakeScorer) -> None:
    monkeypatch.setattr(quality_module, "_load_scorer", lambda model_dir: (fake, _IDENTITY_XK, _IDENTITY_YK))


def test_score_shard_drops_below_threshold_docs_and_preserves_basename_order_and_schema(tmp_path, monkeypatch):
    # Identity knots put the 0-4 score at raw * 4: 0.24 calibrates to 0.96 (just under the gate),
    # 0.26 to 1.04 (just over), 0.9 to 3.6.
    fake = _FakeScorer({"K": 0.26, "D": 0.24, "H": 0.9})
    _patch_loader(monkeypatch, fake)
    rows = [_document("K kept, barely"), _document("D dropped, barely"), _document("H kept, high quality")]
    input_file = _write_input_shard(tmp_path, rows, "part-00003-of-00023.parquet")

    result = score_shard(input_file, str(tmp_path / "out"), model_dir="unused")

    output_file = tmp_path / "out" / "part-00003-of-00023.parquet"
    assert output_file.exists()
    table = pq.read_table(str(output_file))
    assert table.schema.equals(QUALITY_SCHEMA)
    assert result["count"] == 2

    kept_ids = [row["id"] for row in sorted(rows, key=lambda row: row["id"]) if not row["text"].startswith("D")]
    assert table.column("id").to_pylist() == kept_ids  # input id order survives the filter

    edu_max = {row["text"][0]: row["edu_max"] for row in table.to_pylist()}
    assert edu_max["K"] == pytest.approx(1.04)
    assert edu_max["H"] == pytest.approx(3.6)
    for row in table.to_pylist():
        assert row["edu_begin"] == row["edu_middle"] == row["edu_end"] == row["edu_max"]  # short-doc path


def test_score_shard_with_nothing_kept_still_writes_the_empty_copartitioned_shard(tmp_path, monkeypatch):
    """Downstream steps join by basename, so a fully-junk input shard must still have a counterpart."""
    _patch_loader(monkeypatch, _FakeScorer(default=0.0))
    input_file = _write_input_shard(tmp_path, [_document("all junk")], "part-00000-of-00023.parquet")

    score_shard(input_file, str(tmp_path / "out"), model_dir="unused")

    table = pq.read_table(str(tmp_path / "out" / "part-00000-of-00023.parquet"))
    assert table.num_rows == 0
    assert table.schema.equals(QUALITY_SCHEMA)
