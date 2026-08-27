# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for joining the two extraction routes into one corpus.

The two routes extracted disjoint halves of the same sample, so the union has exactly one job:
carry both sides through unchanged while recording which side each document came from. The
schemas differ by the OCR diagnostics, so the union must also null-fill those columns on docling
rows. The cases here hold that boundary -- a record must come back with only ``needs_ocr`` (and,
for docling rows, the null diagnostics) added, the tag must follow the shard rather than be
guessed at, and the result must satisfy the schema the normalize step downstream is given.
"""

import pyarrow as pa
import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.build_pdf_source.combine_routes import (
    _SOURCE_FILE_COLUMN,
    COMBINED_SCHEMA,
    tag_batch,
)
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.datakit.build_pdf_source.extract_fleet import FLEET_FIELDS
from experiments.datakit.build_pdf_source.extract_ocr import OCR_FIELDS, OUTPUT_SCHEMA

_TEXT_MAIN = "s3://bucket/marin/data/datakit/extract/common_crawl_focus_2026_22_pdf_text_84cbb532/outputs/main/"
_OCR_MAIN = "s3://bucket/marin/data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_98f8b74a/outputs/main/"
_SHARD = "part-00000-of-01773.parquet"
_TEXT_SHARD = _TEXT_MAIN + _SHARD
_OCR_SHARD = _OCR_MAIN + _SHARD
_ROUTES = ((False, _TEXT_MAIN), (True, _OCR_MAIN))

_OCR_DIAGNOSTIC_NAMES = tuple(field.name for field in OCR_FIELDS)
_FLEET_COLUMN_NAMES = tuple(field.name for field in FLEET_FIELDS)

_PROSE = (
    "# Coastal erosion along the Holderness cliffs\n\n"
    "The Holderness coast retreats faster than any other shoreline in Europe, losing on average "
    "close to two metres of till each year to the North Sea.\n"
)


def _document(text: str = _PROSE, **overrides) -> dict:
    """A stored docling record, assembled the way the text-route conversion assembled it."""
    row = {
        "id": generate_id(text),
        "text": text,
        "source_id": "crawl-data/CC-MAIN-0001/warc/x.warc.gz:4096",
        "source": "common_crawl_focus_2026_22",
        "warc_filename": "crawl-data/CC-MAIN-0001/warc/x.warc.gz",
        "warc_record_offset": 4096,
        "content_digest": "sha1:ABCDEF",
        "url": "https://example.org/report.pdf",
        "num_pages": 1,
        "page_offsets": [len(text)],
        "extraction_status": "success",
        "extraction_error": None,
        "boilerplate_lines_removed": 3,
        "layout_backend": "torch_heron",
    }
    return {**row, **overrides}


def _ocr_document(text: str = _PROSE, **overrides) -> dict:
    """A stored OCR record: the shared record plus the route's diagnostic columns."""
    row = {
        **{key: value for key, value in _document(text).items() if key not in _FLEET_COLUMN_NAMES},
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


def _text_batch(shard: str, rows: list[dict]) -> pa.RecordBatch:
    """One text-route row group as the reader hands it over, with the source path column injected."""
    schema = pa.schema([*PDF_DOCUMENT_FIELDS, *FLEET_FIELDS, pa.field(_SOURCE_FILE_COLUMN, pa.string(), nullable=False)])
    return pa.RecordBatch.from_pylist([{**row, _SOURCE_FILE_COLUMN: shard} for row in rows], schema=schema)


def _ocr_batch(shard: str, rows: list[dict]) -> pa.RecordBatch:
    """One OCR-route row group, carrying the route's own wider schema."""
    schema = pa.schema([*OUTPUT_SCHEMA, pa.field(_SOURCE_FILE_COLUMN, pa.string(), nullable=False)])
    return pa.RecordBatch.from_pylist([{**row, _SOURCE_FILE_COLUMN: shard} for row in rows], schema=schema)


def test_combined_schema_is_the_shared_record_plus_route_and_nullable_route_columns():
    """The union may only add the router decision and make each route's own columns nullable."""
    assert COMBINED_SCHEMA.names == [
        *(field.name for field in PDF_DOCUMENT_FIELDS),
        "needs_ocr",
        *_OCR_DIAGNOSTIC_NAMES,
        *_FLEET_COLUMN_NAMES,
    ]
    assert COMBINED_SCHEMA.field("needs_ocr").type == pa.bool_()
    assert not COMBINED_SCHEMA.field("needs_ocr").nullable
    for name in _OCR_DIAGNOSTIC_NAMES:
        assert COMBINED_SCHEMA.field(name).nullable, f"{name} must be nullable (null on docling rows)"
        assert COMBINED_SCHEMA.field(name).type == OUTPUT_SCHEMA.field(name).type
    for name in _FLEET_COLUMN_NAMES:
        assert COMBINED_SCHEMA.field(name).nullable, f"{name} must be nullable (null on OCR rows)"


def test_docling_records_pass_through_with_the_route_and_null_diagnostics_added():
    document = _document()
    records = list(tag_batch(_text_batch(_TEXT_SHARD, [document]), _ROUTES))
    assert records == [{**document, **dict.fromkeys(_OCR_DIAGNOSTIC_NAMES), "needs_ocr": False}]


def test_ocr_records_pass_through_with_the_route_and_null_fleet_columns_added():
    document = _ocr_document()
    records = list(tag_batch(_ocr_batch(_OCR_SHARD, [document]), _ROUTES))
    assert records == [{**document, **dict.fromkeys(_FLEET_COLUMN_NAMES), "needs_ocr": True}]


def test_a_shard_under_neither_route_is_an_error_rather_than_a_guess():
    """The tag comes from the driver's own listing; a stray path means that listing is wrong."""
    with pytest.raises(ValueError, match="belongs to neither extraction route"):
        list(tag_batch(_text_batch("s3://bucket/elsewhere/" + _SHARD, [_document()]), _ROUTES))


def test_an_empty_row_group_yields_nothing_rather_than_reaching_for_a_missing_path():
    assert list(tag_batch(_text_batch(_TEXT_SHARD, []), _ROUTES)) == []


def test_tagged_records_from_both_routes_satisfy_the_combined_schema():
    text_rows = [_document(), _document("A second converted document.\n")]
    ocr_rows = [_ocr_document("A transcribed document.\n"), _ocr_document("Another transcription.\n")]
    records = [
        *tag_batch(_text_batch(_TEXT_SHARD, text_rows), _ROUTES),
        *tag_batch(_ocr_batch(_OCR_SHARD, ocr_rows), _ROUTES),
    ]
    table = pa.Table.from_pylist(records, schema=COMBINED_SCHEMA)
    assert table.num_rows == 4
    assert table.column("needs_ocr").to_pylist() == [False, False, True, True]
    # Both inputs' rows come through exactly once, docling's with null diagnostics.
    assert table.column("text").to_pylist() == [row["text"] for row in [*text_rows, *ocr_rows]]
    assert table.column("pages_ocred").to_pylist() == [None, None, 1, 1]
    assert table.column("layout_backend").to_pylist() == ["torch_heron", "torch_heron", None, None]
