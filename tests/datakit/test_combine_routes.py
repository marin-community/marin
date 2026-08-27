# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for choosing one extraction per document and joining the two routes into one corpus.

Under router v2 the routes overlap: pdf-inspector reads every fetched PDF and the VLM re-reads the
escalated subset, so an escalated document exists on both sides and the corpus must take exactly
one. That makes the union's job selection as well as concatenation, and both halves are asserted
here -- a document the router escalated must not arrive from the cheap route, and a record that does
arrive must come back with only ``needs_ocr`` and the other route's null columns added.
"""

import pyarrow as pa
import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.build_pdf_source.combine_routes import (
    _SOURCE_FILE_COLUMN,
    COMBINED_SCHEMA,
    _route_fields,
    tag_batch,
)
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.datakit.build_pdf_source.extract_inspector import INSPECTOR_FIELDS
from experiments.datakit.build_pdf_source.extract_ocr import OCR_FIELDS, OUTPUT_SCHEMA

_INSPECTOR_MAIN = (
    "s3://bucket/marin/data/datakit/extract/common_crawl_focus_2026_22_pdf_inspector_84cbb532/outputs/main/"
)
_OCR_MAIN = "s3://bucket/marin/data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_98f8b74a/outputs/main/"
_SHARD = "part-00000-of-01773.parquet"
_INSPECTOR_SHARD = _INSPECTOR_MAIN + _SHARD
_OCR_SHARD = _OCR_MAIN + _SHARD
_ROUTES = ((False, _INSPECTOR_MAIN), (True, _OCR_MAIN))

_WARC = "crawl-data/CC-MAIN-0001/warc/x.warc.gz"
_OCR_NAMES = frozenset(field.name for field in OCR_FIELDS)
_INSPECTOR_NAMES = frozenset(field.name for field in INSPECTOR_FIELDS)
_OCR_ONLY = tuple(sorted(_OCR_NAMES - _INSPECTOR_NAMES))
_INSPECTOR_ONLY = tuple(sorted(_INSPECTOR_NAMES - _OCR_NAMES))
# Both routes report what the render budget resolved the document to: one measured it while
# rendering, the other computed it from page geometry before deciding not to.
_SHARED_ROUTE_NAMES = ("mean_render_dpi", "pages_below_legibility_floor")

_PROSE = (
    "# Coastal erosion along the Holderness cliffs\n\n"
    "The Holderness coast retreats faster than any other shoreline in Europe, losing on average "
    "close to two metres of till each year to the North Sea.\n"
)


def _shared(text: str, offset: int) -> dict:
    """The part of a stored record both routes write identically."""
    return {
        "id": generate_id(text),
        "text": text,
        "source_id": f"{_WARC}:{offset}",
        "source": "common_crawl_focus_2026_22",
        "warc_filename": _WARC,
        "warc_record_offset": offset,
        "content_digest": "sha1:ABCDEF",
        "url": "https://example.org/report.pdf",
        "num_pages": 1,
        "page_offsets": [len(text)],
        "extraction_status": "success",
        "extraction_error": None,
        "boilerplate_lines_removed": 3,
    }


def _inspector_document(text: str = _PROSE, offset: int = 4096, **overrides) -> dict:
    """A stored pdf-inspector record: the shared record plus every column the router reads."""
    row = {
        **_shared(text, offset),
        **dict.fromkeys(field.name for field in INSPECTOR_FIELDS),
        "pdf_bytes": 91_000,
        "mean_render_dpi": 149.5,
        "pages_below_legibility_floor": 0,
        "inspector_pdf_type": "text_based",
        # Exact in float32, so a record can be compared against its own round trip.
        "inspector_confidence": 0.9375,
        "inspector_page_count": 1,
        "inspector_markdown_chars": len(text),
    }
    return {**row, **overrides}


def _ocr_document(text: str = _PROSE, offset: int = 4096, **overrides) -> dict:
    """A stored OCR record: the shared record plus the route's page accounting."""
    row = {
        **_shared(text, offset),
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
    return {**row, **overrides}


def _inspector_batch(shard: str, rows: list[dict]) -> pa.RecordBatch:
    """One pdf-inspector row group as the reader hands it over, source path column injected."""
    schema = pa.schema(
        [*PDF_DOCUMENT_FIELDS, *INSPECTOR_FIELDS, pa.field(_SOURCE_FILE_COLUMN, pa.string(), nullable=False)]
    )
    return pa.RecordBatch.from_pylist([{**row, _SOURCE_FILE_COLUMN: shard} for row in rows], schema=schema)


def _ocr_batch(shard: str, rows: list[dict]) -> pa.RecordBatch:
    """One OCR-route row group, carrying the route's own wider schema."""
    schema = pa.schema([*OUTPUT_SCHEMA, pa.field(_SOURCE_FILE_COLUMN, pa.string(), nullable=False)])
    return pa.RecordBatch.from_pylist([{**row, _SOURCE_FILE_COLUMN: shard} for row in rows], schema=schema)


def _keys(*offsets: int) -> frozenset[tuple[str, int]]:
    return frozenset((_WARC, offset) for offset in offsets)


# --- the combined schema ------------------------------------------------------------------------


def test_combined_schema_is_the_shared_record_plus_the_route_and_every_route_column_once():
    """The union may only add the router decision and make each route's own columns nullable."""
    assert COMBINED_SCHEMA.names[: len(PDF_DOCUMENT_FIELDS)] == [field.name for field in PDF_DOCUMENT_FIELDS]
    assert COMBINED_SCHEMA.names[len(PDF_DOCUMENT_FIELDS)] == "needs_ocr"
    assert not COMBINED_SCHEMA.field("needs_ocr").nullable
    assert COMBINED_SCHEMA.field("needs_ocr").type == pa.bool_()

    route_names = COMBINED_SCHEMA.names[len(PDF_DOCUMENT_FIELDS) + 1 :]
    assert set(route_names) == _OCR_NAMES | _INSPECTOR_NAMES
    assert len(route_names) == len(set(route_names)), "a column claimed by both routes is carried once"
    for name in route_names:
        assert COMBINED_SCHEMA.field(name).nullable, f"{name} must be nullable (null on the other route's rows)"


def test_a_column_both_routes_write_survives_as_one_column():
    """``mean_render_dpi`` means the same thing on both sides, so a consumer reads it uniformly."""
    for name in _SHARED_ROUTE_NAMES:
        assert name in _OCR_NAMES and name in _INSPECTOR_NAMES
        assert COMBINED_SCHEMA.field(name).type == OUTPUT_SCHEMA.field(name).type
        assert name not in _OCR_ONLY and name not in _INSPECTOR_ONLY


def test_routes_that_disagree_about_a_column_type_are_refused():
    """Carrying a shared column once is only sound if both routes mean the same type by it."""
    with pytest.raises(ValueError, match="mean_render_dpi"):
        _route_fields(OCR_FIELDS, (pa.field("mean_render_dpi", pa.string()),))


# --- selection: exactly one reading of each document ---------------------------------------------


def test_the_cheap_reading_of_an_escalated_document_is_dropped():
    """Both routes read it; keeping both would put two disagreeing copies in the corpus.

    Exact dedup keys on a hash of the text, so the VLM's reading and pdf-inspector's are two
    different documents and neither would remove the other.
    """
    escalated = _inspector_document(offset=4096)
    kept = _inspector_document("A document the router left alone.\n", offset=8192)

    records = list(tag_batch(_inspector_batch(_INSPECTOR_SHARD, [escalated, kept]), _ROUTES, _keys(8192)))

    assert [record["warc_record_offset"] for record in records] == [8192]


def test_the_ocr_route_is_not_filtered_against_the_kept_keys():
    """It only ever read the escalated subset, and those keys are by construction not in the set."""
    records = list(tag_batch(_ocr_batch(_OCR_SHARD, [_ocr_document(offset=4096)]), _ROUTES, _keys(8192)))

    assert [record["warc_record_offset"] for record in records] == [4096]


def test_a_document_with_no_extracted_text_never_reaches_the_corpus_from_the_cheap_route():
    """pdf-inspector stores a row for a document it read nothing from; the gate escalates every one.

    That row is what makes the no-text gate expressible, and this is where it stops being a
    document: it is absent from the kept keys, so the filter removes it.
    """
    empty = _inspector_document("", offset=4096, inspector_markdown_chars=0, extraction_status="empty")

    assert list(tag_batch(_inspector_batch(_INSPECTOR_SHARD, [empty]), _ROUTES, _keys(8192))) == []


# --- concatenation ------------------------------------------------------------------------------


def test_inspector_records_pass_through_with_the_route_and_null_ocr_columns_added():
    document = _inspector_document()
    records = list(tag_batch(_inspector_batch(_INSPECTOR_SHARD, [document]), _ROUTES, _keys(4096)))
    assert records == [{**document, **dict.fromkeys(_OCR_ONLY), "needs_ocr": False}]


def test_ocr_records_pass_through_with_the_route_and_null_inspector_columns_added():
    document = _ocr_document()
    records = list(tag_batch(_ocr_batch(_OCR_SHARD, [document]), _ROUTES, frozenset()))
    assert records == [{**document, **dict.fromkeys(_INSPECTOR_ONLY), "needs_ocr": True}]


def test_a_shard_under_neither_route_is_an_error_rather_than_a_guess():
    """The tag comes from the driver's own listing; a stray path means that listing is wrong."""
    with pytest.raises(ValueError, match="belongs to neither extraction route"):
        list(tag_batch(_inspector_batch("s3://bucket/elsewhere/" + _SHARD, [_inspector_document()]), _ROUTES, _keys()))


def test_an_empty_row_group_yields_nothing_rather_than_reaching_for_a_missing_path():
    assert list(tag_batch(_inspector_batch(_INSPECTOR_SHARD, []), _ROUTES, _keys())) == []


def test_tagged_records_from_both_routes_satisfy_the_combined_schema():
    inspector_rows = [_inspector_document(offset=1), _inspector_document("A second cheap reading.\n", offset=2)]
    ocr_rows = [_ocr_document("A transcribed document.\n", offset=3), _ocr_document("Another one.\n", offset=4)]
    records = [
        *tag_batch(_inspector_batch(_INSPECTOR_SHARD, inspector_rows), _ROUTES, _keys(1, 2)),
        *tag_batch(_ocr_batch(_OCR_SHARD, ocr_rows), _ROUTES, _keys(1, 2)),
    ]
    table = pa.Table.from_pylist(records, schema=COMBINED_SCHEMA)
    assert table.num_rows == 4
    assert table.column("needs_ocr").to_pylist() == [False, False, True, True]
