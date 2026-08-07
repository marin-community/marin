# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the PDF routing step.

The contracts worth pinning here are the ones whose breakage is silent: the driver must be able to
build the step without the ``datakit`` extra, every row must carry every declared column whatever
happened to the PDF, and the booster must refuse to score a feature vector it was not trained on.
A router that quietly scores the wrong columns produces a plausible probability for every document
and misroutes the whole corpus.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
import pytest

from experiments.datakit.build_pdf_source import classify
from experiments.datakit.build_pdf_source.quality import route_features
from experiments.datakit.build_pdf_source.quality.route_feature_names import FEATURE_NAMES


def pdf_bytes(*, pages: int = 1, text: str = "Alpha beta gamma delta epsilon zeta eta theta. " * 12) -> bytes:
    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page(width=612, height=792)
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), text, fontsize=11)
    return document.tobytes()


def source_row(payload: bytes, offset: int = 1) -> dict:
    return {
        "pdf": payload,
        "warc_filename": "CC-SUPPLEMENTAL-2026-22-00000.warc.gz",
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset:040d}",
        "url": f"https://example.org/{offset}.pdf",
    }


def test_a_document_that_cannot_be_parsed_still_produces_a_complete_row():
    """An unreadable PDF is data, not a pipeline failure, and lands on neither route."""
    rows = list(classify.classify_batch(pa.RecordBatch.from_pylist([source_row(b"not a pdf at all")]), "unused"))

    assert len(rows) == 1
    row = rows[0]
    assert row["needs_ocr"] is None
    assert row["docling_confidence"] is None
    assert row["classification_error"]
    # The declared schema is what an extraction step reads back; a partial row breaks that read.
    pa.Table.from_pylist(rows, schema=classify._OUTPUT_SCHEMA)


def test_an_encrypted_document_is_reported_rather_than_routed():
    document = pymupdf.open()
    document.new_page()
    encrypted = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")

    rows = list(classify.classify_batch(pa.RecordBatch.from_pylist([source_row(encrypted)]), "unused"))

    assert rows[0]["needs_ocr"] is None
    assert "encrypted" in rows[0]["classification_error"]


def test_the_sampling_seed_is_keyed_on_content_digest_where_present():
    """Identical PDFs must sample identical pages wherever in the crawl they appear."""
    seed = classify._document_seed("sha1:abc", "file.warc.gz", 42)
    assert seed == classify._document_seed("sha1:abc", "different.warc.gz", 99)


def test_documents_without_a_content_digest_still_get_a_stable_seed():
    first = classify._document_seed("", "file.warc.gz", 42)
    second = classify._document_seed("", "file.warc.gz", 42)
    other = classify._document_seed("", "file.warc.gz", 43)

    assert first == second
    assert first != other


def test_a_document_produces_exactly_the_features_the_booster_expects():
    """The invariant that silently misroutes the whole corpus if it breaks.

    :func:`classify.classify_batch` builds each model row as ``[vector[name] for name in
    FEATURE_NAMES]``, so a signal renamed or dropped in ``route_features`` without the names module
    following would either raise deep in a map task or, worse, line the columns up against the
    wrong trained features and return a confident probability for every document.

    Deliberately checked without XGBoost: on macOS, importing ``faiss`` (which the ``datakit`` extra
    pulls in and other tests in this directory load) and then calling into XGBoost's OpenMP runtime
    segfaults the process. Linux workers are unaffected, so the runtime guard in
    :func:`classify.load_booster` still covers the real failure, but a test that crashes its xdist
    worker is worse than no test.
    """
    with pymupdf.open(stream=pdf_bytes(pages=3), filetype="pdf") as document:
        signals = route_features.signals_for_routing(document, seed=1234)
    vector = signals.feature_vector()

    assert tuple(vector) == FEATURE_NAMES
    assert all(isinstance(value, float) for value in vector.values())
    assert np.isfinite(np.asarray([vector[name] for name in FEATURE_NAMES], dtype=np.float32)).all()


def test_routing_keys_separates_the_two_routes_and_drops_unreadable_documents(tmp_path):
    table = pa.Table.from_pylist(
        [
            {"warc_filename": "a.warc.gz", "warc_record_offset": 1, "needs_ocr": True},
            {"warc_filename": "a.warc.gz", "warc_record_offset": 2, "needs_ocr": False},
            {"warc_filename": "a.warc.gz", "warc_record_offset": 3, "needs_ocr": None},
        ],
        schema=pa.schema(
            [
                pa.field("warc_filename", pa.string()),
                pa.field("warc_record_offset", pa.int64()),
                pa.field("needs_ocr", pa.bool_()),
            ]
        ),
    )
    pq.write_table(table, tmp_path / "part-00000.parquet")

    assert classify.routing_keys(str(tmp_path), needs_ocr=True) == frozenset({("a.warc.gz", 1)})
    assert classify.routing_keys(str(tmp_path), needs_ocr=False) == frozenset({("a.warc.gz", 2)})


def test_routing_keys_refuses_an_empty_directory(tmp_path):
    """Silently returning no keys would route the whole corpus one way."""
    with pytest.raises(RuntimeError, match="No routing table"):
        classify.routing_keys(str(tmp_path), needs_ocr=True)


def test_the_shipped_threshold_sits_inside_the_probability_range():
    """A threshold outside (0, 1) would send every document one way regardless of the model."""
    assert 0.0 < classify.DOCLING_CONFIDENCE_THRESHOLD < 1.0
