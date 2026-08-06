# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the cheap routing signals, against PDFs whose properties are known by construction.

Each test builds a document that has exactly one interesting property -- an invisible text layer, a
ruled grid, two columns read out of order -- and asserts the signal meant to detect it fires while
a clean control document leaves it alone. A signal that fires on everything is as useless to the
router as one that fires on nothing, so both directions are checked.
"""

import pymupdf
import pytest

from experiments.build_pdf_source.quality.route_features import document_signals, page_signals

BODY = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi. "


def signals_for(pdf: bytes) -> dict[str, float]:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return document_signals(doc, list(range(len(doc)))).feature_vector()


def plain_page(*, width: float = 612, height: float = 792) -> tuple[pymupdf.Document, pymupdf.Page]:
    doc = pymupdf.open()
    return doc, doc.new_page(width=width, height=height)


def test_clean_single_column_page_trips_no_difficulty_signal():
    """The control: a page the Docling route should handle must look easy on every axis."""
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), BODY * 8, fontsize=11)
    features = signals_for(doc.tobytes())

    assert features["mean_invisible_char_ratio"] == pytest.approx(0.0)
    assert features["mean_ruling_line_count"] == pytest.approx(0.0)
    assert features["mean_rule_grid_cells"] == pytest.approx(0.0)
    assert features["mean_column_count"] == pytest.approx(1.0)
    assert features["mean_stream_order_inversion_ratio"] == pytest.approx(0.0)
    assert features["mean_replacement_ratio"] == pytest.approx(0.0)
    assert features["mean_alphanum_ratio"] > 0.9


def test_invisible_text_layer_is_detected():
    """Text drawn in render mode 3 is an OCR layer, not something a reader sees."""
    doc, page = plain_page()
    page.insert_text(pymupdf.Point(72, 100), "visible heading of the document body", fontsize=12)
    page.insert_textbox(pymupdf.Rect(72, 150, 540, 700), BODY * 4, fontsize=11, render_mode=3)

    features = signals_for(doc.tobytes())

    assert features["mean_invisible_char_ratio"] > 0.8


def test_ruled_grid_is_detected_and_plain_prose_is_not():
    doc, page = plain_page()
    for index in range(6):
        page.draw_line(pymupdf.Point(60, 100 + index * 40), pymupdf.Point(550, 100 + index * 40))
    for index in range(4):
        page.draw_line(pymupdf.Point(60 + index * 165, 100), pymupdf.Point(60 + index * 165, 300))
    for row in range(5):
        for column in range(3):
            page.insert_text(pymupdf.Point(70 + column * 165, 125 + row * 40), f"cell {row}{column}", fontsize=9)

    features = signals_for(doc.tobytes())

    assert features["mean_ruling_line_count"] >= 8
    assert features["mean_rule_grid_cells"] >= 10
    assert 0.1 < features["mean_ruled_area_ratio"] < 1.0


def test_a_chart_axis_is_not_mistaken_for_a_table_grid():
    """Two crossing rules are an axis; a grid needs repeated rules in both directions."""
    doc, page = plain_page()
    page.draw_line(pymupdf.Point(80, 600), pymupdf.Point(540, 600))
    page.draw_line(pymupdf.Point(80, 150), pymupdf.Point(80, 600))

    features = signals_for(doc.tobytes())

    assert features["mean_rule_grid_cells"] == pytest.approx(0.0)


def test_two_columns_are_counted_as_two():
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 290, 700), BODY * 4, fontsize=10)
    page.insert_textbox(pymupdf.Rect(330, 50, 560, 700), BODY * 4, fontsize=10)

    assert signals_for(doc.tobytes())["mean_column_count"] == pytest.approx(2.0)


def test_stream_order_matching_reading_order_reports_no_inversions():
    """Two columns written left-then-right are in reading order and must not look risky."""
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 290, 400), BODY * 2, fontsize=10)
    page.insert_textbox(pymupdf.Rect(330, 50, 560, 400), BODY * 2, fontsize=10)

    assert signals_for(doc.tobytes())["mean_stream_order_inversion_ratio"] == pytest.approx(0.0)


def test_stream_order_against_reading_order_reports_inversions():
    """The same layout written right column first is exactly the risk the signal is for."""
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(330, 50, 560, 400), BODY * 2, fontsize=10)
    page.insert_textbox(pymupdf.Rect(50, 50, 290, 400), BODY * 2, fontsize=10)

    assert signals_for(doc.tobytes())["mean_stream_order_inversion_ratio"] > 0.0


def test_standard_encoded_fonts_are_not_flagged_as_unmappable():
    """Base-14 fonts carry no ToUnicode and decode perfectly; flagging them would condemn everything."""
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 400), BODY * 2, fontsize=11)

    features = signals_for(doc.tobytes())

    assert features["mean_fonts_without_tounicode"] == pytest.approx(1.0)
    assert features["mean_fonts_unmappable"] == pytest.approx(0.0)


def test_symbol_soup_scores_low_on_alphanumeric_ratio():
    """Marker's garble proxy: text that is mostly punctuation is text nobody can read.

    U+FFFD itself is deliberately not tested here. Producing it requires a genuinely broken font
    program -- MuPDF substitutes ``?`` for glyphs it cannot render and falls back to the standard
    encoding for names it does not know, so no PDF this test could build would carry one.
    """
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 400), "@#%&<>~^ " * 30, fontsize=11)

    assert signals_for(doc.tobytes())["mean_alphanum_ratio"] < 0.1


def test_dot_leaders_do_not_read_as_garble():
    """A table of contents is full of dot leaders and is perfectly extractable."""
    doc, page = plain_page()
    entries = "".join(f"Chapter {index} {'.' * 40} {index * 7}\n" for index in range(1, 12))
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 700), entries, fontsize=11)

    assert signals_for(doc.tobytes())["mean_alphanum_ratio"] > 0.5


def test_short_pages_do_not_report_confident_ratios():
    """A near-empty page has no evidence, and must not be scored as totally garbled."""
    doc, page = plain_page()
    page.insert_text(pymupdf.Point(72, 100), "ok", fontsize=11)

    features = signals_for(doc.tobytes())

    assert features["mean_char_count"] < 40
    assert features["mean_alphanum_ratio"] == pytest.approx(0.0)


def test_a_page_with_no_text_reports_no_text():
    doc, _ = plain_page()

    features = signals_for(doc.tobytes())

    assert features["mean_char_count"] == pytest.approx(0.0)
    assert features["mean_text_block_count"] == pytest.approx(0.0)


def test_document_signals_survive_an_unreadable_page():
    """One bad page must cost that page, not the document -- the rest still carry evidence."""
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 400), BODY * 3, fontsize=11)
    with pymupdf.open(stream=doc.tobytes(), filetype="pdf") as opened:
        result = document_signals(opened, [0, 99])

    assert result.pages_sampled == 1
    assert result.mean["char_count"] > 0


def test_cjk_and_latin_scripts_are_distinguished():
    doc, page = plain_page()
    page.insert_text(pymupdf.Point(72, 100), "日本語のテキストがここにあります" * 4, fontsize=12, fontname="china-s")

    features = signals_for(doc.tobytes())

    assert features["mean_cjk_ratio"] > 0.5
    assert features["mean_latin_ratio"] < 0.2


def test_page_signals_are_computed_per_page_not_per_document():
    """A two-page document with one hard page must show it in the max, not only in the mean."""
    doc = pymupdf.open()
    easy = doc.new_page(width=612, height=792)
    easy.insert_textbox(pymupdf.Rect(72, 72, 540, 700), BODY * 6, fontsize=11)
    hard = doc.new_page(width=612, height=792)
    hard.insert_textbox(pymupdf.Rect(72, 72, 540, 700), BODY * 6, fontsize=11, render_mode=3)

    features = signals_for(doc.tobytes())

    assert features["max_invisible_char_ratio"] > 0.9
    assert features["mean_invisible_char_ratio"] < features["max_invisible_char_ratio"]


def test_page_signals_reads_one_page_at_a_time():
    """The per-page entry point is the unit the aggregates are built from."""
    doc, page = plain_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 400), BODY * 2, fontsize=11)
    with pymupdf.open(stream=doc.tobytes(), filetype="pdf") as opened:
        result = page_signals(opened, opened.load_page(0))

    assert result.char_count > 100
    assert result.text_block_count >= 1
