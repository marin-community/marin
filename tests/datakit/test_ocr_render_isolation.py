# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What the sender does when the rasteriser's child dies, stalls, or refuses a document.

This is the machinery the isolation exists for, and none of it can be exercised against the real
rasteriser: PDFium recorded zero native aborts in 3,577,944 page renders, which is the whole reason
the residual risk is a bound rather than an observation. So the child here is a stub the test
steers, and what runs for real is everything on this side of the pipe -- the framing, the deadline,
the respawn, and the sender's decision to keep the pages a dying document had already produced.

The stub reads its instructions out of the document itself: the "PDF" a row carries is a JSON plan
naming how many pages to stream and where to misbehave. That lets one shard mix a document that
kills the child with documents that do not, which is the case that matters -- Zephyr restarts a
failed shard from row zero, so a document that costs the shard costs every document beside it too.

No PDF library is imported here, by either process. That is itself the claim under test: after the
render moved out, the sender task opens nothing.
"""

import json
import threading
from collections import Counter

import pyarrow as pa
import pytest

from experiments.datakit.build_pdf_source import extract_ocr
from experiments.datakit.build_pdf_source.ocr_extract import render_worker
from experiments.datakit.build_pdf_source.ocr_extract.client import OcrEndpoint, PageOcr
from experiments.datakit.build_pdf_source.ocr_extract.render import RAISED_MAX_VISUAL_TOKENS, RenderOptions
from experiments.datakit.build_pdf_source.ocr_extract.render_worker import (
    DEADLINE_EXCEEDED,
    PROTOCOL_ERROR,
    WORKER_DIED,
    RenderWorker,
)

# A page payload chosen to break a protocol that framed on anything but a byte count: it holds a
# newline, a NUL, and something that reads as a frame header of its own.
_PAGE = b'\x89PNG\r\n\x1a\n{"frame": "page", "size": 0}\n\x00\xff' * 8

_COUNTER_PREFIX = "focus_crawl_pdf_ocr"

_STUB_WORKER = '''
"""A stand-in for the rasteriser's child, speaking the same frames and misbehaving on request.

The document it is handed is a JSON plan: ``pages`` to stream, ``mode`` to end on, and ``at`` for
the page index the misbehaviour starts at.
"""
import json
import os
import signal
import sys
import time

stdin, stdout = sys.stdin.buffer, sys.stdout.buffer


def frame(header, payload=b""):
    stdout.write(json.dumps(header).encode() + b"\\n")
    stdout.write(payload)
    stdout.flush()


while True:
    line = stdin.readline()
    if not line:
        break
    request = json.loads(line)
    body = b""
    while len(body) < request["size"]:
        chunk = stdin.read(request["size"] - len(body))
        if not chunk:
            sys.exit(3)
        body += chunk
    plan = json.loads(body)
    if plan["mode"] == "refuse":
        frame({"frame": "end", "error_type": "PdfiumError", "error": "PdfiumError: not a document"})
        continue
    if plan["mode"] == "garbage":
        stdout.write(b"a native library wrote this to stdout\\n")
        stdout.flush()
        continue
    if plan["mode"] == "options":
        payload = json.dumps(request["options"], sort_keys=True).encode()
    else:
        payload = bytes.fromhex(plan["payload_hex"])
    frame({"frame": "start", "declared_pages": plan["pages"]})
    for index in range(plan["pages"]):
        if index == plan["at"]:
            if plan["mode"] == "die":
                os._exit(9)
            if plan["mode"] == "abort":
                os.kill(os.getpid(), signal.SIGSEGV)
            if plan["mode"] == "hang":
                time.sleep(60)
            if plan["mode"] == "fail":
                break
        header = {"frame": "page", "page_index": index, "pixels": 2088960, "dpi": 149.47, "size": len(payload)}
        frame(header, payload)
    if plan["mode"] == "fail":
        frame({"frame": "end", "error_type": "PdfiumError", "error": "PdfiumError: page tree damaged"})
    else:
        frame({"frame": "end", "error_type": None, "error": None})
'''


def _plan(pages: int = 3, mode: str = "echo", at: int | None = None, payload: bytes = _PAGE) -> bytes:
    """The bytes a row carries, which the stub reads as its instructions."""
    return json.dumps({"pages": pages, "mode": mode, "at": at, "payload_hex": payload.hex()}).encode()


def _row(offset: int, pdf: bytes) -> dict:
    return {
        "pdf": pdf,
        "warc_filename": "crawl.warc.gz",
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
    }


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


def _word(index: int) -> str:
    """A digit-free, fixed-length token unique to ``index``.

    Page bodies have to differ by more than a digit: boilerplate detection folds digits to zero, so
    numbered bodies would all be detected as a running header and stripped.
    """
    return "abcdefghijklmnopqrstuvwxyz"[index % 26] * 5


def _page_text(_endpoint, _connections, page) -> PageOcr:
    return PageOcr(text=f"body {_word(page.page_index)}", completion_tokens=10)


@pytest.fixture
def tallies(monkeypatch) -> Counter:
    """Every counter the batch emits, which on a run whose logs are gone is the only diagnosis."""
    recorded: Counter = Counter()

    class Recorder:
        def update_counter(self, name: str, value: int | float) -> None:
            recorded[name] += value

    monkeypatch.setattr(extract_ocr.counters, "pipeline", Recorder())
    return recorded


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """A :class:`RenderWorker` whose child is the stub rather than the rasteriser."""

    def build(deadline: float = 10.0) -> RenderWorker:
        (tmp_path / "render_stub_worker.py").write_text(_STUB_WORKER)
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.setattr(render_worker, "MODULE_NAME", "render_stub_worker")
        return RenderWorker(deadline=deadline)

    return build


@pytest.fixture
def run_batch(monkeypatch, tallies):
    """Run ``ocr_batch`` against a chosen worker, with a stand-in for the endpoint call."""

    def run(rows, worker, respond=_page_text, *, pages_in_flight: int | None = None):
        monkeypatch.setattr(extract_ocr, "ocr_page", respond)
        monkeypatch.setattr(extract_ocr, "render_worker", lambda deadline: worker)
        if pages_in_flight is not None:
            monkeypatch.setattr(extract_ocr, "_PAGES_IN_FLIGHT", pages_in_flight)
        return list(
            extract_ocr.ocr_batch(
                _batch(rows),
                keys=None,
                raised_keys=frozenset(),
                endpoint=OcrEndpoint(base_url="http://unused/v1", model="test-model", max_visual_tokens=2048),
                render_options=RenderOptions(),
                raised_render_options=RenderOptions(max_visual_tokens=RAISED_MAX_VISUAL_TOKENS),
                boilerplate=extract_ocr.BOILERPLATE_OPTIONS,
                loop=extract_ocr.LOOP_OPTIONS,
            )
        )

    return run


# --- the protocol ------------------------------------------------------------------------------


def test_a_page_crosses_the_pipe_byte_for_byte(worker):
    """PNG is binary, so the framing has to be a byte count. Anything that framed on content would
    be cut short by the newline in this payload and confused by the header inside it."""
    rendered = []
    child = worker()
    try:
        with child.render(_plan(pages=3), RenderOptions()) as stream:
            assert stream.declared_pages == 3
            rendered = list(stream)
    finally:
        child.stop()

    assert [page.page_index for page in rendered] == [0, 1, 2]
    assert [page.png for page in rendered] == [_PAGE] * 3
    assert child.spawns == 1, "one child serves every document the process reads"


def test_the_render_budget_travels_with_the_document(worker):
    """The router's render policy is a per-document choice, so the budget has to cross the pipe.

    A budget that stopped at the parent would leave every flagged document rendered at the default
    while the request still declared the raised one -- the policy silently doing nothing, which is
    exactly what it looks like when it works.
    """
    options = RenderOptions(max_visual_tokens=RAISED_MAX_VISUAL_TOKENS)
    child = worker()
    try:
        with child.render(_plan(pages=1, mode="options"), options) as stream:
            (page,) = list(stream)
    finally:
        child.stop()

    assert json.loads(page.png) == {
        "max_visual_tokens": RAISED_MAX_VISUAL_TOKENS,
        "max_render_dpi": options.max_render_dpi,
        "legibility_floor_dpi": options.legibility_floor_dpi,
        "max_pages": options.max_pages,
    }


# --- a child that dies ---------------------------------------------------------------------------


def test_a_child_that_dies_mid_document_keeps_the_pages_it_streamed(run_batch, worker, tallies):
    """The failure the isolation is bought for. In process this is a signal, not an exception: the
    map task is gone, Zephyr restarts its shard from row zero, and three attempts later the stage
    has failed with the shard's finished pages thrown away."""
    (record,) = run_batch([_row(0, _plan(pages=6, mode="die", at=3))], worker())

    assert record["pages_ocred"] == 3
    assert record["pages_unrendered"] == 3
    assert record["extraction_status"] == extract_ocr.OcrStatus.PARTIAL
    assert "3 of 6 pages were not rendered" in record["extraction_error"]
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/{WORKER_DIED}"] == 1


def test_a_child_the_kernel_kills_is_named_by_the_signal_that_killed_it(run_batch, worker, tallies):
    """The failure this whole module exists for, injected: a ``SIGSEGV`` mid-document.

    It has to be reported as a death rather than as a stall. The two are told apart by the
    descriptor, not by ``poll()``, whose answer can still be "running" on the read that saw the
    child's stdout close -- and a native abort filed under the deadline's counter sends the next
    reader hunting a hang that never happened.
    """
    (record,) = run_batch([_row(0, _plan(pages=5, mode="abort", at=2))], worker())

    assert record["pages_ocred"] == 2
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/{WORKER_DIED}"] == 1
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/{DEADLINE_EXCEEDED}"] == 0


def test_a_dead_child_is_replaced_and_the_documents_behind_it_still_render(run_batch, worker):
    """Zephyr restarts a shard from row zero, so a document that costs the shard costs every
    document beside it. Replacing the child is what keeps the cost to the one document."""
    child = worker()
    rows = [_row(0, _plan(pages=4, mode="die", at=1)), _row(1, _plan(pages=2)), _row(2, _plan(pages=3))]

    records = run_batch(rows, child)

    assert [record["warc_record_offset"] for record in records] == [0, 1, 2]
    assert [record["pages_ocred"] for record in records] == [1, 2, 3]
    assert child.spawns == 2, "the dead child is replaced once, not once per document"


def test_a_child_that_dies_before_its_first_page_costs_only_that_document(run_batch, worker, tallies):
    """Nothing was rendered, so there is no row -- but the shard keeps going, which is the point."""
    records = run_batch([_row(0, _plan(pages=5, mode="die", at=0)), _row(1, _plan(pages=2))], worker())

    assert [record["warc_record_offset"] for record in records] == [1]
    assert tallies[f"{_COUNTER_PREFIX}/render_failed"] == 1


# --- a child that stops answering ------------------------------------------------------------------


def test_a_child_that_stops_answering_is_bounded_by_the_page_deadline(run_batch, worker, tallies):
    """A hang is the other way a native library ends a task: in process the map task holds the
    document until its heartbeat expires and the retry lands on the same one."""
    child = worker(deadline=0.5)

    (record,) = run_batch([_row(0, _plan(pages=8, mode="hang", at=2))], child)

    assert record["pages_ocred"] == 2
    assert record["pages_unrendered"] == 6
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/{DEADLINE_EXCEEDED}"] == 1
    assert child.spawns == 2, "a child still inside the library cannot be handed the next document"


def test_a_stalled_child_does_not_cost_the_documents_behind_it(run_batch, worker):
    child = worker(deadline=0.5)
    rows = [_row(0, _plan(pages=6, mode="hang", at=1)), _row(1, _plan(pages=3))]

    records = run_batch(rows, child)

    assert [record["pages_ocred"] for record in records] == [1, 3]


# --- a child that stays healthy and says no ---------------------------------------------------------


def test_a_document_the_child_refuses_is_data_rather_than_a_failure(run_batch, worker, tallies):
    """A crawl PDF that no library will open is an ordinary outcome, and the child survives it, so
    there is nothing to replace and nothing to raise."""
    child = worker()

    records = run_batch([_row(0, _plan(mode="refuse")), _row(1, _plan(pages=2))], child)

    assert [record["warc_record_offset"] for record in records] == [1]
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/PdfiumError"] == 1
    assert child.spawns == 1, "the document failed, not the child"


def test_a_document_that_fails_part_way_keeps_its_pages_and_its_child(run_batch, worker, tallies):
    """The library raising half-way down a document is the in-process failure mode, unchanged: the
    pages before it are still worth keeping and the process is still fine."""
    child = worker()

    (record,) = run_batch([_row(0, _plan(pages=5, mode="fail", at=2))], child)

    assert record["pages_ocred"] == 2
    assert record["pages_unrendered"] == 3
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/PdfiumError"] == 1
    assert child.spawns == 1


def test_a_child_that_writes_something_that_is_not_a_frame_is_retired(run_batch, worker, tallies):
    """Anything printing to the child's stdout desynchronises every document after it, so the
    stream is not recoverable and the child goes."""
    child = worker(deadline=1.0)

    records = run_batch([_row(0, _plan(mode="garbage")), _row(1, _plan(pages=2))], child)

    assert [record["pages_ocred"] for record in records] == [2]
    assert tallies[f"{_COUNTER_PREFIX}/render_failed/{PROTOCOL_ERROR}"] == 1
    assert child.spawns == 2


# --- the in-flight bound ---------------------------------------------------------------------------


def test_the_in_flight_bound_caps_the_pages_a_task_holds(run_batch, worker, monkeypatch):
    """The bound is what keeps a long document from accumulating encoded pages -- over a megabyte
    each -- and it is also what stops the child running ahead of the fleet, since the child blocks
    writing a page nothing has read. Measured on the queue itself: submissions in, absorptions out.
    """
    depth = {"held": 0, "peak": 0}
    pool = extract_ocr._request_pool(4)

    class Probe:
        def submit(self, *arguments):
            depth["held"] += 1
            depth["peak"] = max(depth["peak"], depth["held"])
            return pool.submit(*arguments)

    absorb = extract_ocr._Document.absorb

    def counted(self, future, loop) -> None:
        depth["held"] -= 1
        absorb(self, future, loop)

    monkeypatch.setattr(extract_ocr, "_request_pool", lambda threads: Probe())
    monkeypatch.setattr(extract_ocr._Document, "absorb", counted)

    (record,) = run_batch([_row(0, _plan(pages=20))], worker(), pages_in_flight=4)

    assert depth["peak"] == 4, "the task never holds more rendered pages than the bound allows"
    assert record["pages_ocred"] == 20, "and bounding what it holds must not cost it pages"


def test_a_document_longer_than_the_in_flight_window_is_assembled_in_reading_order(run_batch, worker):
    """Completion order must not reach the document, however the requests resolve."""
    total = 12
    returned = [threading.Event() for _ in range(total + 1)]
    returned[total].set()
    submissions = iter(range(total))
    lock = threading.Lock()

    def respond(_endpoint, _connections, page) -> PageOcr:
        with lock:
            index = next(submissions)
        assert returned[index + 1].wait(timeout=30), "the completion chain stalled"
        result = PageOcr(text=f"page {_word(page.page_index)}", completion_tokens=1)
        returned[index].set()
        return result

    (record,) = run_batch([_row(0, _plan(pages=total))], worker(), respond)

    assert record["text"] == "".join(f"page {_word(index)}\n" for index in range(total))
