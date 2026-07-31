# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the docling-based PDF text extractor.

The behaviour under test is the difference between docling's own output and this extractor's:
words rebuilt from the spans a PDF split them into, letter-spaced headings collapsed, tables read
from ruling lines, and the model fields the postprocessors depend on actually surviving.
"""

from io import BytesIO

import pytest

pytest.importorskip("docling")
pytest.importorskip("pymupdf")

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc.base import BoundingBox

from experiments.build_pdf_source.docling_extract.assemble import (
    collapse_letter_spacing,
    is_letter_spaced_heading,
    join_cluster_cells,
    replace_faulty_characters,
)
from experiments.build_pdf_source.docling_extract.backend import (
    PyMuPdfDocumentBackend,
    PyMuPdfPageBackend,
    blocks_to_cells,
)
from experiments.build_pdf_source.docling_extract.fields import patch_docling_models
from experiments.build_pdf_source.docling_extract.serializer import alphabetic_ratio

_PAGE_HEIGHT = 800.0
# A 10-point font advancing about 5 points per glyph, which is what the gap thresholds are relative to.
_LINE_HEIGHT = 12.0
_CHAR_WIDTH = 5.0


def _spans(*specs: tuple[str, float, float], flags: int = 0) -> list:
    """Build text cells from ``(text, left, top)`` triples, sized as a 10-point font would be.

    Mirrors what PyMuPDF's ``get_text("dict")`` hands the backend, so this exercises the real
    cell-construction path rather than a hand-built approximation of it.
    """
    patch_docling_models()
    lines: dict[float, list] = {}
    for text, left, top in specs:
        lines.setdefault(top, []).append(
            {
                "text": text,
                "bbox": (left, top, left + _CHAR_WIDTH * len(text), top + _LINE_HEIGHT),
                "flags": flags,
            }
        )
    blocks = [{"lines": [{"dir": (1, 0), "spans": spans} for spans in lines.values()]}]
    return blocks_to_cells(blocks, page_height=_PAGE_HEIGHT)


def test_a_word_split_across_spans_is_rejoined_without_a_space():
    """A font or colour change mid-word splits a span; docling's own assembler inserts a space."""
    cells = _spans(("dif", 72.0, 100.0), ("ficult", 87.0, 100.0))

    assert join_cluster_cells(cells).text == "difficult"


def test_a_real_word_gap_still_produces_a_space():
    cells = _spans(("hello", 72.0, 100.0), ("world", 110.0, 100.0))

    assert join_cluster_cells(cells).text == "hello world"


def test_a_hyphen_at_a_line_break_is_dropped_and_the_word_rejoined():
    cells = _spans(("con-", 72.0, 100.0), ("tinues", 72.0, 130.0))

    assert join_cluster_cells(cells).text == "continues"


def test_a_hyphen_before_a_non_alphanumeric_continuation_is_kept():
    """Only a hyphen that split a word is a line-break hyphen."""
    cells = _spans(("page 3 -", 72.0, 100.0), ("- see also", 72.0, 130.0))

    assert join_cluster_cells(cells).text.startswith("page 3 -")


def test_a_span_drawn_twice_in_the_same_place_is_emitted_once():
    """Some producers fake a bold weight by drawing the same text twice, slightly offset."""
    cells = _spans(("Heading", 72.0, 100.0), ("Heading", 72.4, 100.0))

    assert join_cluster_cells(cells).text == "Heading"


def test_a_superscript_is_separated_from_the_word_it_follows():
    """A footnote marker hard against a word must not become part of it."""
    body = _spans(("evidence", 72.0, 100.0))
    marker = _spans(("12", 112.0, 100.0), flags=1)

    assert join_cluster_cells(body + marker).text == "evidence 12"


def test_the_median_glyph_advance_and_last_line_are_recorded():
    """The paragraph and span mergers measure distances in these units, so they must be present."""
    assembled = join_cluster_cells(_spans(("first line", 72.0, 100.0), ("second line", 72.0, 130.0)))

    assert assembled.median_char_width == pytest.approx(_CHAR_WIDTH)
    assert assembled.last_line_bbox is not None
    # The document uses a bottom-left origin, so the last line's top is measured from the page foot.
    assert assembled.last_line_bbox.t == pytest.approx(_PAGE_HEIGHT - 130.0)


def test_an_empty_cluster_yields_no_geometry():
    assembled = join_cluster_cells([])

    assert assembled.text == ""
    assert assembled.median_char_width is None
    assert assembled.last_line_bbox is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A N N U A L  R E P O R T", True),
        ("H E L L O", True),
        ("A B C D", True),
        # Three letters is too short to distinguish tracking from spaced initials.
        ("A B C", False),
        ("Annual Report", False),
        ("A quick brown fox", False),
    ],
)
def test_letter_spaced_headings_are_told_apart_from_ordinary_text(text, expected):
    assert is_letter_spaced_heading(text) is expected


def test_letter_spacing_collapses_to_words():
    """Single spaces are tracking and go; runs of two or more are real word boundaries."""
    assert collapse_letter_spacing("A N N U A L  R E P O R T") == "ANNUAL REPORT"


def test_substituted_punctuation_is_mapped_back_to_ascii():
    assert replace_faulty_characters("“quoted” ‘word’ 1⁄2") == "\"quoted\" 'word' 1/2"  # noqa: RUF001


@pytest.mark.parametrize(
    ("text", "expected"),
    [("all letters", 10 / 11), ("", 0.0), ("12345", 0.0)],
)
def test_alphabetic_ratio_measures_prose_likeness(text, expected):
    assert alphabetic_ratio(text) == pytest.approx(expected)


def _one_page_pdf() -> bytes:
    """A minimal single-page PDF, built in memory so the test needs no fixture file."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "hello")
    data = document.tobytes()
    document.close()
    return data


def test_pages_stay_readable_after_the_document_backend_is_unloaded():
    """Docling's threaded pipeline can reach a page after the document backend is unloaded.

    Closing the PyMuPDF document there invalidates every page already handed out, and the next
    access dies inside PyMuPDF with a bare "page is None" assertion rather than anything
    diagnosable. Only the DOCLING table backend reaches a page that late, which is why the PyMuPDF
    table path never saw it.

    Dropping the last Python reference is not the same thing and would not catch this: a
    ``pymupdf.Page`` holds its document through ``.parent``, so only an explicit ``close()``
    invalidates it.
    """
    document = InputDocument(
        path_or_stream=BytesIO(_one_page_pdf()),
        format=InputFormat.PDF,
        backend=PyMuPdfDocumentBackend,
        filename="test.pdf",
    )
    page_backend = document._backend.load_page(0)

    document._backend.unload()

    assert page_backend.is_valid()
    assert page_backend.get_size().width > 0
    assert page_backend.get_segmented_page() is not None


def test_unloading_a_page_backend_marks_it_invalid():
    """`valid` gates every accessor, so unload must clear it or each guard becomes a null deref."""
    document = pymupdf.open(stream=_one_page_pdf(), filetype="pdf")
    page_backend = PyMuPdfPageBackend(document, "test-hash", 0)

    page_backend.unload()

    assert not page_backend.is_valid()
    assert page_backend.get_segmented_page() is None
    assert page_backend.get_text_in_rect(BoundingBox(l=0, t=0, r=10, b=10)) == ""
    assert list(page_backend.get_text_cells()) == []
