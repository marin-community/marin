# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OCR extraction route: page rendering and the sender's document assembly.

Rendering runs in a child process, as in production, so every test through ``ocr_batch`` exercises
the real protocol against the real rasteriser. A child that dies or stalls is covered in
``test_ocr_render_isolation.py`` with a stub child.
"""

import base64
import io
import threading
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.build_pdf_source import extract_ocr
from experiments.datakit.build_pdf_source.classify import ROUTING_SCHEMA
from experiments.datakit.build_pdf_source.common import SOURCE_FILE_COLUMN
from experiments.datakit.build_pdf_source.extract_ocr import OcrStatus, ocr_batch
from experiments.datakit.build_pdf_source.ocr_extract import client
from experiments.datakit.build_pdf_source.ocr_extract.client import OcrEndpoint, PageOcr, unwrap_markdown_fence
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    DEFAULT_MAX_VISUAL_TOKENS,
    MAX_PIXELS,
    RAISED_MAX_VISUAL_TOKENS,
    VISUAL_TOKEN_PIXELS,
    RenderedPage,
    RenderOptions,
    effective_dpi,
    iter_rendered_pages,
    open_pdf,
    target_dimensions,
)
from experiments.datakit.build_pdf_source.ocr_extract.render_worker import RenderWorker

# The rasteriser ships in the ``pdf`` extra, which the workspace root does not install.
pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")

# US Letter and ISO A0, in points; A0 is the sheet the default budget renders below the legibility floor.
_LETTER = (612, 792)
_A0 = (2384, 3370)

# The fetched shard a batch comes from, and therefore the routing shard it reads its decisions from.
_SHARD = "part-00000-of-00001.parquet"
_WARC = "crawl.warc.gz"
# ``run_batch``'s default routing: every document in the batch escalated at the default budget.
_ALL_ESCALATED = object()


def _word(index: int) -> str:
    """A digit-free, fixed-length token unique to ``index``.

    Boilerplate detection folds digits to zero, so numbered bodies would be stripped as a running header.
    """
    return "abcdefghijklmnopqrstuvwxyz"[index % 26] * 5


def _pdf(page_count: int, size: tuple[float, float] = _LETTER) -> bytes:
    """A real PDF with ``page_count`` numbered pages, written out directly rather than through a library.

    Object numbering: 1 catalog, 2 page tree, 3 the shared font, then a page and a content stream
    per page. Offsets are collected while the body is written so the xref table is exact.
    """
    width, height = size
    bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{4 + 2 * index} 0 R" for index in range(page_count)), page_count
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index in range(page_count):
        stream = f"BT /F1 24 Tf 72 {height - 72:g} Td (Page {index} content) Tj ET"
        bodies.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} {height:g}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>"
        )
        bodies.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode("latin-1")
    return bytes(out)


def _batch(rows: list[dict]) -> pa.RecordBatch:
    """One fetched row group as the reader hands it over, with the shard's own path injected."""
    return pa.RecordBatch.from_pylist(
        [{**row, SOURCE_FILE_COLUMN: f"s3://bucket/fetch/{_SHARD}"} for row in rows],
        schema=pa.schema(
            [
                pa.field("pdf", pa.binary()),
                pa.field("warc_filename", pa.string()),
                pa.field("warc_record_offset", pa.int64()),
                pa.field("content_digest", pa.string()),
                pa.field("url", pa.string()),
                pa.field(SOURCE_FILE_COLUMN, pa.string()),
            ]
        ),
    )


def _row(offset: int, pdf: bytes) -> dict:
    return {
        "pdf": pdf,
        "warc_filename": _WARC,
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
    }


def _routing_row(offset: int, budget: int | None) -> dict:
    """One routing decision: escalated at ``budget``, or kept on the cheap route when ``None``."""
    return {
        "warc_filename": _WARC,
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
        "needs_ocr": budget is not None,
        "route_reason": "score",
        "escalation_score": None,
        "render_visual_tokens": DEFAULT_MAX_VISUAL_TOKENS if budget is None else budget,
        "inspector_markdown_chars": None,
        "mean_render_dpi": None,
        "num_pages": None,
    }


@pytest.fixture
def endpoint():
    return OcrEndpoint(base_url="http://unused/v1", model="test-model", max_visual_tokens=2048)


@pytest.fixture(scope="module")
def worker():
    """One rasteriser child for the module, as a sender process holds one for its whole life."""
    worker = RenderWorker(deadline=30.0)
    yield worker
    worker.stop()


@pytest.fixture
def run_batch(monkeypatch, endpoint, worker, tmp_path):
    """Run ``ocr_batch`` with a stand-in for the endpoint call; everything else runs for real.

    ``routing`` maps an offset to the render budget it is escalated at, or to ``None`` to keep it
    on the cheap route; an offset absent from the map is absent from the shard. The default
    escalates every document at the default budget, and ``None`` runs without a routing table.
    """
    runs = iter(range(1_000))

    def run(rows, respond, *, routing=_ALL_ESCALATED, render_options=None, boilerplate=None, loop=None):
        monkeypatch.setattr(extract_ocr, "ocr_page", respond)
        monkeypatch.setattr(extract_ocr, "render_worker", lambda deadline: worker)
        routing_dir = None
        if routing is not None:
            if routing is _ALL_ESCALATED:
                routing = {row["warc_record_offset"]: DEFAULT_MAX_VISUAL_TOKENS for row in rows}
            # A fresh directory per run: the lookup caches a shard by its path.
            directory = tmp_path / f"routing-{next(runs)}"
            directory.mkdir()
            decisions = [_routing_row(offset, budget) for offset, budget in routing.items()]
            pq.write_table(pa.Table.from_pylist(decisions, schema=ROUTING_SCHEMA), directory / _SHARD)
            routing_dir = str(directory)
        return list(
            ocr_batch(
                _batch(rows),
                routing_dir=routing_dir,
                endpoint=endpoint,
                render_options=render_options or RenderOptions(),
                raised_render_options=RenderOptions(max_visual_tokens=RAISED_MAX_VISUAL_TOKENS),
                boilerplate=boilerplate or extract_ocr.BOILERPLATE_OPTIONS,
                loop=loop or extract_ocr.LOOP_OPTIONS,
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
    """The budget holds cost constant, so paper size comes out of resolution."""
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


def _blue_page_pdf() -> bytes:
    """A single page flooded with pure blue, for the one bug a monochrome fixture cannot see."""
    width, height = _LETTER
    stream = f"0 0 1 rg 0 0 {width:g} {height:g} re f"
    bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} {height:g}] /Contents 4 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    return bytes(out)


def test_a_coloured_page_keeps_its_channels():
    """The render flag, not the wrapper's ``rev_byteorder``, reverses the BGR buffer; dropping it
    swaps red and blue, which black-on-white text cannot show."""
    from PIL import Image  # noqa: PLC0415

    with open_pdf(_blue_page_pdf()) as document:
        page = next(iter(iter_rendered_pages(document, RenderOptions())))
    image = Image.open(io.BytesIO(page.png))
    red, green, blue = (band.getextrema()[0] for band in image.convert("RGB").split())
    assert blue > 200, f"the blue channel should dominate a blue page, got r={red} g={green} b={blue}"
    assert red < 60, f"the red channel should be empty on a blue page, got r={red} g={green} b={blue}"


def test_rendered_pages_come_back_as_png_bytes():
    """Bytes, not the data URI: the page crosses a pipe before anything asks for base64."""
    with open_pdf(_pdf(1)) as document:
        page = next(iter(iter_rendered_pages(document, RenderOptions())))
    assert page.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert page.dpi > 0


# --- document assembly -----------------------------------------------------------------------


def _reverse_completion_responder(total: int, text):
    """A responder whose requests complete in strictly reverse submission order.

    Request *k* blocks until request *k+1* has returned, so the last submission resolves first.
    """
    returned = [threading.Event() for _ in range(total + 1)]
    returned[total].set()
    submissions = iter(range(total))
    lock = threading.Lock()

    def respond(_endpoint, _connections, page) -> PageOcr:
        with lock:
            index = next(submissions)
        assert returned[index + 1].wait(timeout=30), "the completion chain stalled"
        result = PageOcr(text=text(page), completion_tokens=1)
        returned[index].set()
        return result

    return respond


def test_pages_are_assembled_in_reading_order_when_requests_finish_out_of_order(run_batch):
    """Completion order must not reach the document."""
    respond = _reverse_completion_responder(5, lambda page: f"page {_word(page.page_index)}")

    (record,) = run_batch([_row(0, _pdf(5))], respond)
    assert record["text"] == "".join(f"page {_word(index)}\n" for index in range(5))
    assert record["page_offsets"] == [11, 22, 33, 44, 55]


def test_documents_are_emitted_in_input_order(run_batch):
    """The last document's pages complete first, and the output order must not follow suit."""
    respond = _reverse_completion_responder(6, lambda page: f"page {_word(page.page_index)}")

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
    records = run_batch(rows, _page_text, routing={0: None, 1: DEFAULT_MAX_VISUAL_TOKENS})
    assert [record["warc_record_offset"] for record in records] == [1]


def test_without_a_routing_table_every_document_is_ocred(run_batch):
    """``routing_dir=None`` is the all-routes comparison run: no document is filtered."""
    rows = [_row(0, _pdf(1)), _row(1, _pdf(1))]
    records = run_batch(rows, _page_text, routing=None)
    assert [record["warc_record_offset"] for record in records] == [0, 1]


def test_a_document_absent_from_its_routing_shard_is_an_error_rather_than_a_skip(run_batch):
    """The table is total over the extraction, so an unknown key is a broken join, not a kept document."""
    rows = [_row(0, _pdf(1)), _row(1, _pdf(1))]
    with pytest.raises(ValueError, match="no routing decision"):
        run_batch(rows, _page_text, routing={1: DEFAULT_MAX_VISUAL_TOKENS})


def test_a_render_budget_this_step_does_not_render_at_is_refused(run_batch):
    """A budget neither option set renders at means the routing table was built for other render
    options than this step's."""
    with pytest.raises(ValueError, match="render budget"):
        run_batch([_row(0, _pdf(1))], _page_text, routing={0: 4 * DEFAULT_MAX_VISUAL_TOKENS})


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


def _budget_recorder(seen: list[tuple[int, int]]):
    """A responder that records what each page was rendered at and what the request declared."""

    def respond(endpoint, _connections, page) -> PageOcr:
        seen.append((page.pixels, endpoint.max_visual_tokens))
        return PageOcr(text=f"body {_word(page.page_index)}", completion_tokens=1)

    return respond


def test_the_render_policy_lifts_a_flagged_document_over_the_legibility_floor(run_batch):
    """The same sheet, read at a resolution the model can use; nothing else about the document changes."""
    default_pages: list[tuple[int, int]] = []
    raised_pages: list[tuple[int, int]] = []

    (baseline,) = run_batch([_row(0, _pdf(1, size=_A0))], _budget_recorder(default_pages))
    (rescued,) = run_batch(
        [_row(0, _pdf(1, size=_A0))], _budget_recorder(raised_pages), routing={0: RAISED_MAX_VISUAL_TOKENS}
    )

    assert baseline["mean_render_dpi"] < 100 <= rescued["mean_render_dpi"]
    assert baseline["pages_below_legibility_floor"] == 1
    assert rescued["pages_below_legibility_floor"] == 0
    assert raised_pages[0][0] == pytest.approx(8 * default_pages[0][0], rel=0.02), "8x the budget, 8x the pixels"


def test_a_raised_render_declares_its_own_budget_to_the_endpoint(run_batch):
    """The request restates the budget as ``max_pixels``; the server resizes against that
    declaration, so a stale one would shrink the raised render straight back."""
    seen: list[tuple[int, int]] = []

    run_batch(
        [_row(0, _pdf(1, size=_A0)), _row(1, _pdf(1, size=_A0))],
        _budget_recorder(seen),
        routing={0: DEFAULT_MAX_VISUAL_TOKENS, 1: RAISED_MAX_VISUAL_TOKENS},
    )

    declared = [tokens for _pixels, tokens in seen]
    assert declared == [2048, RAISED_MAX_VISUAL_TOKENS]


def test_a_document_the_policy_did_not_flag_keeps_the_default_budget(run_batch):
    """The raised budget is applied only to flagged documents."""
    seen: list[tuple[int, int]] = []

    run_batch([_row(0, _pdf(2))], _budget_recorder(seen), routing={0: DEFAULT_MAX_VISUAL_TOKENS})

    assert {tokens for _pixels, tokens in seen} == {2048}


def test_completion_tokens_are_summed_over_the_document(run_batch):
    (record,) = run_batch([_row(0, _pdf(4))], _page_text)
    assert record["completion_tokens"] == 40


# --- the output schema -------------------------------------------------------------------------


def test_the_ocr_record_matches_its_declared_schema(run_batch):
    """A record that does not fit the schema fails only at write time, a whole shard later."""
    records = run_batch([_row(0, _pdf(2))], _page_text)
    assert pa.RecordBatch.from_pylist(records, schema=extract_ocr.OUTPUT_SCHEMA).num_rows == 1


# --- the request the page is sent in ------------------------------------------------------------


def test_a_rendered_page_reaches_the_endpoint_as_a_png_data_uri(monkeypatch, endpoint):
    """The page travels as PNG bytes and is base64'd once, at the boundary that wants it."""
    sent: dict = {}

    class Completions:
        def create(self, **request):
            sent.update(request)
            message = SimpleNamespace(content="transcribed")
            return SimpleNamespace(
                usage=SimpleNamespace(completion_tokens=7),
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
            )

    monkeypatch.setattr(
        client,
        "_client",
        lambda endpoint, connections: SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))

    result = client.ocr_page(endpoint, 4, RenderedPage(png=png, page_index=0, pixels=2088960, dpi=149.5))

    image = sent["messages"][0]["content"][0]
    assert image["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(png).decode()
    assert image["max_pixels"] == endpoint.max_visual_tokens * VISUAL_TOKEN_PIXELS
    assert result == PageOcr(text="transcribed", completion_tokens=7, truncated=False)


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
    """A page cut off at the token cap never emits a closing fence, so an opener on an unfinished page
    can only be a wrapper."""
    cut_off = "```markdown\n# Title\n\nhalf a sen"
    assert unwrap_markdown_fence(cut_off, truncated=True) == "# Title\n\nhalf a sen"
    # Without the truncation signal an unterminated fence is indistinguishable from content.
    assert unwrap_markdown_fence(cut_off) == cut_off


def test_truncation_does_not_license_stripping_a_non_wrapper_fence():
    assert unwrap_markdown_fence("```python\nx = 1", truncated=True) == "```python\nx = 1"
    assert unwrap_markdown_fence("# Title\n\nprose", truncated=True) == "# Title\n\nprose"
