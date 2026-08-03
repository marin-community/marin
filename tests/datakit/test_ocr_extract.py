# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OCR extraction route: page rendering and the sender's document assembly."""

import time

import pyarrow as pa
import pymupdf
import pytest

from experiments.build_pdf_source import extract, extract_ocr
from experiments.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.build_pdf_source.extract_ocr import OcrStatus, ocr_batch
from experiments.build_pdf_source.ocr_extract.client import OcrEndpoint, PageOcr, unwrap_markdown_fence
from experiments.build_pdf_source.ocr_extract.render import (
    MAX_PIXELS,
    VISUAL_TOKEN_PIXELS,
    RenderOptions,
    effective_dpi,
    iter_rendered_pages,
    open_pdf,
    target_dimensions,
)

# US Letter and ISO A0, in points. A0 is the case the visual-token budget handles badly on purpose:
# it costs the model the same as a Letter page and therefore gets far less resolution.
_LETTER = (612, 792)
_A0 = (2384, 3370)


def _word(index: int) -> str:
    """A digit-free, fixed-length token unique to ``index``.

    Page bodies have to differ by more than a digit. Boilerplate detection folds digits to zero, so
    "body 1" and "body 2" are the same line to it, and a fixture with numbered bodies would have
    every page's body detected as a running header and stripped.
    """
    return "abcdefghijklmnopqrstuvwxyz"[index % 26] * 5


def _pdf(page_count: int, size: tuple[float, float] = _LETTER) -> bytes:
    """A real PDF with ``page_count`` numbered pages, so rendering is exercised for real."""
    document = pymupdf.open()
    for index in range(page_count):
        page = document.new_page(width=size[0], height=size[1])
        page.insert_text((72, 72), f"Page {index} content")
    return document.tobytes()


def _batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("pdf", pa.binary()),
                pa.field("warc_filename", pa.string()),
                pa.field("warc_record_offset", pa.int64()),
                pa.field("content_digest", pa.string()),
                pa.field("url", pa.string()),
            ]
        ),
    )


def _row(offset: int, pdf: bytes) -> dict:
    return {
        "pdf": pdf,
        "warc_filename": "crawl.warc.gz",
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
    }


def _keys(*offsets: int) -> frozenset[tuple[str, int]]:
    return frozenset(("crawl.warc.gz", offset) for offset in offsets)


@pytest.fixture
def endpoint():
    return OcrEndpoint(base_url="http://unused/v1", model="test-model", max_visual_tokens=2048)


@pytest.fixture
def run_batch(monkeypatch, endpoint):
    """Run ``ocr_batch`` with a stand-in for the endpoint call.

    The substitute replaces the network, not the logic under test: rendering, the in-flight bound,
    page ordering, boilerplate removal and record assembly all run for real.
    """

    def run(rows, respond, *, keys=None, render_options=None, boilerplate=None):
        monkeypatch.setattr(extract_ocr, "ocr_page", respond)
        return list(
            ocr_batch(
                _batch(rows),
                keys=keys if keys is not None else _keys(*(row["warc_record_offset"] for row in rows)),
                endpoint=endpoint,
                render_options=render_options or RenderOptions(),
                boilerplate=boilerplate or extract_ocr.BOILERPLATE_OPTIONS,
            )
        )

    return run


def _page_text(_endpoint, _connections, page) -> PageOcr:
    return PageOcr(text=f"body {_word(page.page_index)}", completion_tokens=10)


# --- rendering -------------------------------------------------------------------------------


def test_a_letter_page_fills_the_visual_token_budget():
    options = RenderOptions(max_visual_tokens=2048)
    height, width = target_dimensions(*_LETTER, options)
    tokens = (height * width) / VISUAL_TOKEN_PIXELS
    # Alignment to the patch stride keeps it just under the budget rather than exactly at it.
    assert 1900 <= tokens <= 2048


def test_a_small_page_is_not_upscaled_past_the_dpi_cap():
    """A business-card page could fill the budget only by being blown up past any useful detail."""
    options = RenderOptions(max_visual_tokens=2048, max_render_dpi=300.0)
    height, width = target_dimensions(200, 200, options)
    assert effective_dpi(height * width, 200, 200) <= 300.5
    assert (height * width) / VISUAL_TOKEN_PIXELS < 2048


def test_a_large_format_page_lands_below_the_legibility_floor():
    """The budget holds cost constant, so paper size comes out of resolution. This is the cost."""
    options = RenderOptions(max_visual_tokens=2048)
    height, width = target_dimensions(*_A0, options)
    assert effective_dpi(height * width, *_A0) < options.legibility_floor_dpi


def test_a_budget_above_the_model_ceiling_is_refused():
    with pytest.raises(ValueError, match="upstream client ceiling"):
        RenderOptions(max_visual_tokens=MAX_PIXELS // VISUAL_TOKEN_PIXELS + 1)


def test_rendering_stops_at_the_page_budget():
    with open_pdf(_pdf(10)) as document:
        rendered = list(iter_rendered_pages(document, RenderOptions(max_pages=4)))
    assert [page.page_index for page in rendered] == [0, 1, 2, 3]


def test_rendered_pages_are_png_data_uris():
    with open_pdf(_pdf(1)) as document:
        page = next(iter(iter_rendered_pages(document, RenderOptions())))
    assert page.data_uri.startswith("data:image/png;base64,")
    assert page.dpi > 0


# --- document assembly -----------------------------------------------------------------------


def test_pages_are_assembled_in_reading_order_when_requests_finish_out_of_order(run_batch):
    """The whole point of the in-order queue: completion order must not reach the document."""

    def respond(_endpoint, _connections, page) -> PageOcr:
        # Earlier pages answer last.
        time.sleep(0.02 * (5 - page.page_index))
        return PageOcr(text=f"page {_word(page.page_index)}", completion_tokens=1)

    (record,) = run_batch([_row(0, _pdf(5))], respond)
    assert record["text"] == "".join(f"page {_word(index)}\n" for index in range(5))
    assert record["page_offsets"] == [11, 22, 33, 44, 55]


def test_documents_are_emitted_in_input_order(run_batch):
    def respond(_endpoint, _connections, page) -> PageOcr:
        time.sleep(0.01)
        return PageOcr(text=f"page {_word(page.page_index)}", completion_tokens=1)

    rows = [_row(offset, _pdf(2)) for offset in (10, 20, 30)]
    records = run_batch(rows, respond)
    assert [record["warc_record_offset"] for record in records] == [10, 20, 30]


def test_a_document_longer_than_the_in_flight_window_is_assembled_whole(run_batch, monkeypatch):
    """The bound exists to cap memory on long documents; it must not cost pages."""
    monkeypatch.setattr(extract_ocr, "_PAGES_IN_FLIGHT", 3)
    (record,) = run_batch([_row(0, _pdf(12))], _page_text)
    assert record["num_pages"] == 12
    assert record["pages_ocred"] == 12
    assert len(record["page_offsets"]) == 12
    assert record["extraction_status"] == OcrStatus.SUCCESS


def test_a_failed_page_becomes_an_empty_page_and_the_document_survives(run_batch):
    def respond(_endpoint, _connections, page) -> PageOcr:
        if page.page_index == 1:
            raise RuntimeError("upstream said no")
        return PageOcr(text=f"page {_word(page.page_index)}", completion_tokens=1)

    (record,) = run_batch([_row(0, _pdf(3))], respond)
    assert record["extraction_status"] == OcrStatus.PARTIAL
    assert record["pages_failed"] == 1
    assert record["pages_ocred"] == 2
    # The empty page keeps its slot, so an offset still marks where page 1 would have been.
    assert record["page_offsets"] == [11, 11, 22]
    assert "upstream said no" in record["extraction_error"]


def test_a_document_whose_every_page_fails_is_dropped(run_batch):
    def respond(_endpoint, _connections, _page) -> PageOcr:
        raise RuntimeError("endpoint down")

    assert run_batch([_row(0, _pdf(3))], respond) == []


def test_an_unreadable_pdf_is_dropped_without_blocking_later_documents(run_batch):
    rows = [_row(0, b"this is not a pdf"), _row(1, _pdf(2))]
    records = run_batch(rows, _page_text)
    assert [record["warc_record_offset"] for record in records] == [1]


def test_documents_routed_to_text_extraction_are_not_ocred(run_batch):
    rows = [_row(0, _pdf(1)), _row(1, _pdf(1))]
    records = run_batch(rows, _page_text, keys=_keys(1))
    assert [record["warc_record_offset"] for record in records] == [1]


def test_no_key_set_ocrs_every_route(run_batch):
    """``keys=None`` is the all-routes comparison run: no document is filtered."""
    rows = [_row(0, _pdf(1)), _row(1, _pdf(1))]
    records = run_batch(rows, _page_text, keys=None)
    assert [record["warc_record_offset"] for record in records] == [0, 1]


def test_truncated_documents_report_the_pages_they_lost(run_batch):
    (record,) = run_batch([_row(0, _pdf(9))], _page_text, render_options=RenderOptions(max_pages=4))
    assert record["num_pages"] == 4
    assert record["pages_unrendered"] == 5
    assert record["extraction_status"] == OcrStatus.PARTIAL
    assert "5 of 9 pages were not rendered" in record["extraction_error"]


def test_running_headers_are_stripped_before_the_id_is_computed(run_batch):
    def respond(_endpoint, _connections, page) -> PageOcr:
        return PageOcr(text=f"ACME Confidential\nbody {_word(page.page_index)}", completion_tokens=1)

    (record,) = run_batch([_row(0, _pdf(8))], respond)
    assert "ACME Confidential" not in record["text"]
    assert record["boilerplate_lines_removed"] == 8
    # Offsets are recomputed against the stored text, not the text that was OCR'd.
    assert record["page_offsets"][-1] == len(record["text"])


def test_large_format_pages_are_counted_against_the_legibility_floor(run_batch):
    (record,) = run_batch([_row(0, _pdf(3, size=_A0))], _page_text)
    assert record["pages_below_legibility_floor"] == 3
    assert record["mean_render_dpi"] < RenderOptions().legibility_floor_dpi


def test_completion_tokens_are_summed_over_the_document(run_batch):
    (record,) = run_batch([_row(0, _pdf(4))], _page_text)
    assert record["completion_tokens"] == 40


# --- the contract between the two routes -------------------------------------------------------


def test_both_extraction_routes_share_a_column_prefix():
    """The two routes are concatenated downstream, so the shared columns must line up exactly."""
    shared = [(field.name, field.type) for field in PDF_DOCUMENT_FIELDS]
    docling = [(field.name, field.type) for field in extract._OUTPUT_SCHEMA]
    ocr = [(field.name, field.type) for field in extract_ocr._OUTPUT_SCHEMA]
    assert docling == shared
    assert ocr[: len(shared)] == shared


def test_the_ocr_record_matches_its_declared_schema(run_batch):
    """A record that does not fit the schema fails only at write time, a whole shard later."""
    records = run_batch([_row(0, _pdf(2))], _page_text)
    assert pa.RecordBatch.from_pylist(records, schema=extract_ocr._OUTPUT_SCHEMA).num_rows == 1


# --- page fence unwrapping ---------------------------------------------------------------------


def test_a_page_wrapped_in_a_markdown_fence_is_unwrapped():
    """The model returns whole pages as ```markdown blocks; a page is not a code listing."""
    assert unwrap_markdown_fence("```markdown\n# Title\n\nbody\n```") == "# Title\n\nbody"
    assert unwrap_markdown_fence("```md\n# Title\n```") == "# Title"


@pytest.mark.parametrize(
    "text",
    [
        # A page really can be a code listing, and there the fence is content.
        "```\nint main(void);\n```",
        "```python\nx = 1\n```",
        # A fence inside the page is content whatever surrounds it.
        "# Title\n\n```c\nint x;\n```\n\nprose",
        # Never treat an unterminated fence as a wrapper.
        "```markdown\n# Title\n\nprose",
        "# Title\n\nprose",
    ],
)
def test_fences_that_are_not_page_wrappers_are_left_alone(text):
    assert unwrap_markdown_fence(text) == text


def test_a_truncated_page_loses_its_unterminated_opening_fence():
    """A page cut off at the token cap never emits a closing fence, so both-ends matching fails.

    Wrapping is universal (3,000 of 3,000 raw responses), so an opener on an unfinished page can
    only be a wrapper. Without this the leftover markers land exactly on the damaged pages.
    """
    cut_off = "```markdown\n# Title\n\nhalf a sen"
    assert unwrap_markdown_fence(cut_off, truncated=True) == "# Title\n\nhalf a sen"
    # Absent the truncation signal the same text must be left alone: an unterminated fence is
    # otherwise indistinguishable from a page whose content legitimately opens with one.
    assert unwrap_markdown_fence(cut_off) == cut_off


def test_truncation_does_not_license_stripping_a_non_wrapper_fence():
    assert unwrap_markdown_fence("```python\nx = 1", truncated=True) == "```python\nx = 1"
    assert unwrap_markdown_fence("# Title\n\nprose", truncated=True) == "# Title\n\nprose"
