# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the pdf-inspector extraction route and the signals the router reads off it.

The library itself is not exercised: it runs in a child process, and what this pipeline can get
wrong is the row it builds from the reply. The reply is constructed directly, in each shape the
worker can return.
"""

import polars as pl
import pyarrow as pa
import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.build_pdf_source import extract_inspector
from experiments.datakit.build_pdf_source import route_v2_features as contract
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.datakit.build_pdf_source.extract import BOILERPLATE_OPTIONS
from experiments.datakit.build_pdf_source.extract_inspector import (
    INSPECTOR_FIELDS,
    OP_EXTRACT,
    OP_GEOMETRY,
    OUTPUT_SCHEMA,
    SIGNAL_COLUMNS,
    InspectorStatus,
    document_geometry,
    extract_document,
    output_statistics,
)
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RAISED_MAX_VISUAL_TOKENS,
    RenderGeometry,
    RenderOptions,
    render_geometry,
)

# US Letter and ISO A0, in points; A0 is the sheet the default budget renders below the legibility floor.
_LETTER = (612.0, 792.0)
_A0 = (2384.0, 3370.0)

_ROW = {
    "pdf": b"%PDF-1.7 not really a pdf",
    "warc_filename": "crawl-data/CC-MAIN-0001/warc/x.warc.gz",
    "warc_record_offset": 4096,
    "content_digest": "sha1:ABCDEF",
    "url": "https://example.org/report.pdf",
}

_GEOMETRY = RenderGeometry(pages=3, mean_dpi=149.5, pages_below_floor=0)


def _reply(pages: list[str], **overrides) -> dict:
    """A worker reply for a document both library calls succeeded on."""
    return {
        "inspector_pdf_type": "text_based",
        "inspector_confidence": 0.94,
        "inspector_page_count": len(pages),
        "inspector_has_title": True,
        "inspector_ocr_reasons": "{}",
        "inspector_detect_pages_needing_ocr": 0,
        "inspector_extract_is_complex_layout": False,
        "inspector_extract_pages_needing_ocr": 0,
        "inspector_extract_pages_with_tables": 0,
        "inspector_extract_pages_with_columns": 0,
        "inspector_extracted_pages": len(pages),
        "pages": pages,
    } | overrides


def _document(pages: list[str], reply: dict | None = None, geometry: RenderGeometry | None = _GEOMETRY) -> dict:
    return extract_document(_ROW, reply if reply is not None else _reply(pages), geometry, BOILERPLATE_OPTIONS)


# --- the stored record ---------------------------------------------------------------------------


def test_the_record_matches_its_declared_schema():
    """A record that does not fit the schema fails only at write time, a whole shard later."""
    record = _document(["First page.", "Second page."])

    assert pa.RecordBatch.from_pylist([record], schema=OUTPUT_SCHEMA).num_rows == 1


def test_the_shared_columns_come_first_and_unchanged():
    """The two routes are concatenated downstream, so the shared prefix has to line up exactly."""
    assert [field.name for field in OUTPUT_SCHEMA][: len(PDF_DOCUMENT_FIELDS)] == [
        field.name for field in PDF_DOCUMENT_FIELDS
    ]


def test_running_headers_are_stripped_before_the_id_is_computed():
    """The id has to be a hash of the text a consumer reads, not of the text before cleanup."""
    pages = [f"ACME QUARTERLY REPORT\nbody {letter * 5}" for letter in "abcdefgh"]

    record = _document(pages)

    assert "ACME QUARTERLY REPORT" not in record["text"]
    assert record["boilerplate_lines_removed"] == len(pages)
    assert record["id"] == generate_id(record["text"])


def test_page_offsets_index_the_stored_text():
    """Offsets are recomputed after stripping, so a span can be traced back to its page."""
    record = _document(["alpha", "beta", "gamma"])

    assert record["text"] == "alpha\nbeta\ngamma\n"
    assert record["page_offsets"] == [6, 11, 17]
    assert record["page_offsets"][-1] == len(record["text"])


def test_the_document_carries_the_geometry_the_router_will_score_on():
    """``num_pages`` is the rasteriser's count, because that is what the training table used."""
    record = _document(["one page of text"])

    assert record["num_pages"] == _GEOMETRY.pages
    assert record["mean_render_dpi"] == pytest.approx(_GEOMETRY.mean_dpi)
    assert record["pages_below_legibility_floor"] == 0


# --- a row for every document, including the ones with nothing in them ----------------------------


def test_a_document_with_no_text_is_a_row_rather_than_a_dropped_document():
    """The no-text gate escalates it, and the gate needs the row to fire on."""
    record = _document([])

    assert record["extraction_status"] == str(InspectorStatus.EMPTY)
    assert record["text"] == ""
    assert record["inspector_markdown_chars"] == 0
    assert record["source_id"] == f"{_ROW['warc_filename']}:{_ROW['warc_record_offset']}"


def test_a_scan_that_extracts_only_whitespace_is_empty_rather_than_successful():
    """pdf-inspector returns a page per sheet for a scan; whitespace is not text."""
    record = _document(["   ", "\n\n"])

    assert record["extraction_status"] == str(InspectorStatus.EMPTY)
    assert record["inspector_markdown_chars"] == 5, "the gate reads what the library produced, not what survived"


def test_a_failed_extraction_leaves_the_signals_null_rather_than_zero():
    """Null is what tells the booster the feature is missing; zero would be a measurement."""
    record = _document([], reply={"extract_error": "PanicException: index out of bounds"})

    assert record["extraction_status"] == str(InspectorStatus.FAILED)
    assert record["inspector_markdown_chars"] is None
    assert record["inspector_output_alpha_ratio"] is None
    assert "PanicException" in record["extraction_error"]
    assert record["inspector_error"] == record["extraction_error"]


def test_a_deadline_failure_is_recorded_as_data_rather_than_raised():
    """A document that ran past the deadline is a row, not a lost shard."""
    record = _document([], reply={"extract_error": "no reply within 30s"})

    assert record["extraction_status"] == str(InspectorStatus.FAILED)
    assert "no reply within 30s" in record["extraction_error"]


def test_detect_failing_does_not_discard_the_extraction():
    """The two library calls are independent, so one refusing must not cost the other's signals."""
    reply = _reply(["real text"], detect_error="ValueError: encrypted")
    for name in ("inspector_pdf_type", "inspector_confidence", "inspector_page_count", "inspector_has_title"):
        reply.pop(name)

    record = _document(["real text"], reply=reply)

    assert record["extraction_status"] == str(InspectorStatus.SUCCESS)
    assert record["text"] == "real text\n"
    assert record["inspector_pdf_type"] is None
    assert record["inspector_extract_pages_with_tables"] == 0
    assert "detect: ValueError: encrypted" in record["extraction_error"]


def test_a_document_nothing_can_render_keeps_null_geometry():
    """Null, not zero: a mean DPI of 0.0 would read as "renders far below the floor" and route the
    document the other way."""
    record = _document(["text the library could still read"], geometry=None)

    assert record["mean_render_dpi"] is None
    assert record["pages_below_legibility_floor"] is None
    assert record["num_pages"] == 0
    assert record["inspector_markdown_chars"] > 0


# --- the signal projection the router reads ------------------------------------------------------


def test_every_router_feature_is_derivable_from_the_projected_columns():
    """A column missing from the projection is a silent null feature, not an error."""
    record = _document(["some prose to measure, with enough words in it to be real"])
    frame = pl.DataFrame([{name: record[name] for name in SIGNAL_COLUMNS}])

    derived = contract.with_derived(frame)

    assert set(contract.ROUTER_FEATURES) <= set(derived.columns)
    assert derived.select(contract.ROUTER_FEATURES).null_count().to_numpy().sum() == 0


def test_the_projection_never_pulls_the_corpus_text_through_the_router():
    """At full-crawl scale that projection is the difference between scalars and tens of GB."""
    assert "text" not in SIGNAL_COLUMNS
    assert set(SIGNAL_COLUMNS) >= {field.name for field in INSPECTOR_FIELDS}


# --- output statistics -----------------------------------------------------------------------------


def test_output_statistics_separate_clean_text_from_the_two_failures_they_exist_to_catch():
    """Garbling and repetition are measured on the produced text."""
    clean = output_statistics(["The quick brown fox jumps over the lazy dog near the river bank."], 1)
    garbled = output_statistics(["Th� q�ick br�wn f�x"], 1)
    looping = output_statistics(["Continue reading\n" * 40], 1)

    assert clean["inspector_output_replacement_ratio"] == 0.0
    assert garbled["inspector_output_replacement_ratio"] > clean["inspector_output_replacement_ratio"]
    assert clean["inspector_output_repeat_line_ratio"] == pytest.approx(0.0)
    assert looping["inspector_output_repeat_line_ratio"] > 0.9
    assert looping["inspector_output_max_line_repeats"] == pytest.approx(40.0)


def test_output_statistics_of_an_empty_extraction_are_defined_rather_than_missing():
    """Nulls here would make an empty extraction indistinguishable from a document the pass never reached."""
    empty = output_statistics([], 5)

    assert empty["inspector_output_empty_page_fraction"] == 1.0
    assert empty["inspector_output_chars_per_source_page"] == 0.0
    assert empty["inspector_output_alpha_ratio"] == 0.0
    assert set(empty) == {f"inspector_output_{name}" for name in extract_inspector.OUTPUT_STATISTIC_NAMES}


def test_expected_output_length_is_absolute_because_it_predicts_truncation():
    """Truncation is a completion-budget failure, so the predictor has to be a length not a ratio."""
    short = output_statistics(["brief"], 1)
    long_page = output_statistics(["word " * 4000], 1)

    assert long_page["inspector_output_chars_per_source_page"] > 100 * short["inspector_output_chars_per_source_page"]


def test_output_length_is_measured_per_source_page_not_per_extracted_page():
    """A forty-page scan yielding one page of text is the case the ratio has to catch."""
    per_source = output_statistics(["a" * 400], 40)
    per_extracted = output_statistics(["a" * 400], 1)

    assert per_source["inspector_output_chars_per_source_page"] == pytest.approx(10.0)
    assert per_extracted["inspector_output_chars_per_source_page"] == pytest.approx(400.0)


# --- the child-process isolation ------------------------------------------------------------------
#
# The child is a stub whose behaviour the test chooses; what runs for real is the framing, the
# deadline and the respawn on this side of the pipe.

_STUB_WORKER = '''
"""A stand-in for the extraction worker, speaking the same length-prefixed protocol.

``STUB_MODE`` names the operation the stub misbehaves on, so a test can hang or kill the child on
the rasteriser's round trip while the extractor's still answers.
"""
import json
import os
import signal
import sys
import time

mode, bad_op = os.environ["STUB_MODE"].split(":")
marker = os.environ["STUB_MARKER"]
stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
while True:
    header = stdin.readline()
    if not header:
        break
    request = json.loads(header)
    size = request["size"]
    payload = b""
    while len(payload) < size:
        chunk = stdin.read(size - len(payload))
        if not chunk:
            sys.exit(3)
        payload += chunk
    if request["op"] == bad_op:
        if mode == "hang":
            time.sleep(60)
        if mode == "die_once" and not os.path.exists(marker):
            open(marker, "w").close()
            os._exit(9)
        if mode == "segv_once" and not os.path.exists(marker):
            open(marker, "w").close()
            os.kill(os.getpid(), signal.SIGSEGV)
    if request["op"] == "geometry":
        reply = {"page_rectangles": [[612.0, 792.0]]}
    else:
        reply = {"pages": [payload.decode()], "inspector_extracted_pages": 1}
    stdout.write(json.dumps(reply).encode() + b"\\n")
    stdout.flush()
'''


@pytest.fixture
def stub_worker(monkeypatch, tmp_path):
    """Point :class:`InspectorWorker` at a stub child whose behaviour the test picks."""

    def build(mode: str = "echo:none", deadline: float = 5.0):
        (tmp_path / "inspector_stub_worker.py").write_text(_STUB_WORKER)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.setenv("STUB_MODE", mode)
        monkeypatch.setenv("STUB_MARKER", str(tmp_path / "died"))
        monkeypatch.setattr(extract_inspector, "MODULE_NAME", "inspector_stub_worker")
        return extract_inspector.InspectorWorker(deadline=deadline)

    return build


def test_a_document_round_trips_through_the_child_process(stub_worker):
    """The framing has to survive arbitrary PDF bytes, including a newline in the payload."""
    worker = stub_worker()
    try:
        assert worker.call(OP_EXTRACT, b"%PDF-1.7\nbody")["pages"] == ["%PDF-1.7\nbody"]
        assert worker.call(OP_EXTRACT, b"second")["pages"] == ["second"]
        assert worker.call(OP_GEOMETRY, b"second")["page_rectangles"] == [[612.0, 792.0]]
        assert worker.spawns == 1, "one child serves every document the process reads"
    finally:
        worker.stop()


def test_a_document_that_never_returns_is_bounded_from_outside_the_library(stub_worker):
    """The library has no deadline of its own; in process a hung document would hold the map task
    until its heartbeat expired."""
    worker = stub_worker(f"hang:{OP_EXTRACT}", deadline=0.5)
    try:
        reply = worker.call(OP_EXTRACT, b"a document that hangs")

        assert "no reply within" in reply["extract_error"]
        assert worker.spawns == 2, "the hung child is replaced rather than reused"
    finally:
        worker.stop()


def test_a_child_that_dies_costs_one_document_rather_than_the_shard(stub_worker):
    """A native abort is a signal no ``except`` can catch; in process it would fail the shard on
    every retry."""
    worker = stub_worker(f"die_once:{OP_GEOMETRY}")
    try:
        reply = worker.call(OP_GEOMETRY, b"the document that kills the rasteriser")
        assert "worker exited with 9" in reply["geometry_error"]

        assert worker.call(OP_EXTRACT, b"the next one")["pages"] == ["the next one"]
        assert worker.spawns == 2
    finally:
        worker.stop()


def test_a_child_killed_by_a_signal_is_named_as_a_death_and_not_as_a_deadline(stub_worker):
    """A signal death is named as a death, not filed under the deadline.

    The two are told apart by the child's EOF rather than ``poll()``, whose answer can still be
    "running" on the read that saw stdout close. The deadline is long so only the naming is under test.
    """
    worker = stub_worker(f"segv_once:{OP_EXTRACT}", deadline=30.0)
    try:
        reply = worker.call(OP_EXTRACT, b"a document that aborts the crate")

        assert "killed by SIGSEGV" in reply["extract_error"]
        assert "no reply within" not in reply["extract_error"]
        assert worker.call(OP_EXTRACT, b"the next one")["pages"] == ["the next one"]
    finally:
        worker.stop()


def test_a_rasteriser_that_dies_does_not_cost_the_extraction(stub_worker):
    """Each library gets its own round trip so one cannot take the other's result down."""
    worker = stub_worker(f"die_once:{OP_GEOMETRY}")
    try:
        extracted = worker.call(OP_EXTRACT, b"real text")
        geometry = document_geometry(worker.call(OP_GEOMETRY, b"real text"), RenderOptions())
    finally:
        worker.stop()

    record = extract_document(_ROW, extracted, geometry, BOILERPLATE_OPTIONS)
    assert record["extraction_status"] == str(InspectorStatus.SUCCESS)
    assert record["text"] == "real text\n"
    assert record["mean_render_dpi"] is None, "and the router keeps it, because nothing can render it"


def test_a_dead_child_becomes_a_failed_row_with_its_provenance_intact(stub_worker):
    """The row is what the router gates on, so it has to survive the death that produced it."""
    worker = stub_worker(f"die_once:{OP_EXTRACT}")
    try:
        record = extract_document(_ROW, worker.call(OP_EXTRACT, _ROW["pdf"]), _GEOMETRY, BOILERPLATE_OPTIONS)
    finally:
        worker.stop()

    assert record["extraction_status"] == str(InspectorStatus.FAILED)
    assert record["url"] == _ROW["url"]
    assert record["inspector_markdown_chars"] is None
    assert pa.RecordBatch.from_pylist([record], schema=OUTPUT_SCHEMA).num_rows == 1


def test_geometry_comes_back_as_the_router_reads_it(stub_worker):
    """The rectangles cross the pipe; the arithmetic that prices them stays in the parent."""
    worker = stub_worker()
    try:
        geometry = document_geometry(worker.call(OP_GEOMETRY, b"a letter page"), RenderOptions())
    finally:
        worker.stop()

    assert geometry.pages == 1
    assert geometry.mean_dpi == pytest.approx(149.47, abs=0.1)


# --- render geometry ------------------------------------------------------------------------------
#
# Arithmetic over page rectangles, free of any PDF library.


def test_geometry_is_the_arithmetic_the_render_would_have_applied():
    """A Letter page fills the default budget at ~149 DPI."""
    geometry = render_geometry([_LETTER, _LETTER], RenderOptions())

    assert geometry.pages == 2
    assert geometry.mean_dpi == pytest.approx(149.47, abs=0.1)
    assert geometry.pages_below_floor == 0


def test_a_large_format_sheet_lands_under_the_floor_and_the_raised_budget_lifts_it_over():
    """The render policy's trigger and its effect: an A0 sheet lands under the floor at the default
    budget and over it at the raised one."""
    default = render_geometry([_A0], RenderOptions())
    raised = render_geometry([_A0], RenderOptions(max_visual_tokens=RAISED_MAX_VISUAL_TOKENS))

    assert default.mean_dpi < RenderOptions().legibility_floor_dpi
    assert default.pages_below_floor == 1
    assert raised.mean_dpi >= RenderOptions().legibility_floor_dpi
    assert raised.pages_below_floor == 0


def test_a_degenerate_page_is_excluded_rather_than_counted_as_zero_dpi():
    """The renderer refuses sub-point pages; counting one as 0 DPI would drag a legible document
    under the floor."""
    geometry = render_geometry([_LETTER, (0.5, 0.5)], RenderOptions())

    assert geometry.pages == 1
    assert geometry.pages_below_floor == 0


def test_a_document_with_no_usable_page_reports_no_geometry():
    """Which is what :func:`document_geometry` turns into the "nothing can render this" signal."""
    assert render_geometry([], RenderOptions()).pages == 0
