# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3: read every fetched PDF's text layer with pdf-inspector, and measure what came out.

This is the cheap route, and under router v2 it is also the pass the router reads. Both facts follow
from one number: pdf-inspector extracts a page for **2.1 CPU core-hours per million crawl pages**
against Docling's 278 -- 132x -- for corpus-wide quality parity, ~0.51 page-weighted in the blind
head-to-head (``pdf-inspector-evaluation.md``). At that price the pipeline stops routing *before*
extraction and starts routing *after* it: running the cheap extractor on 100% of documents and then
deciding costs less than running a 3.4 core-h/M PyMuPDF feature pass on 100% of documents to decide
first, and it gives the router signals measured on real output instead of predicted from font tables
(``pdf-router-v2.md``, "Free features against paid ones").

So this step has two outputs in one table. Every fetched PDF gets exactly one row:

* the shared document record -- text, provenance, page offsets, extraction status -- which is what
  :mod:`~experiments.datakit.build_pdf_source.combine_routes` unions into the corpus, and
* :data:`INSPECTOR_FIELDS`, the raw columns
  :mod:`~experiments.datakit.build_pdf_source.classify` turns into the router's 43 features.

They live in one table rather than two because Parquet is columnar: the router reads its ~30 narrow
columns without touching ``text``, and the union reads ``text`` without paying for the signals. A
document pdf-inspector produced nothing for is still a row -- with ``text`` empty -- because that is
the router's single most decisive input. Every one of the 2,054 labelled such documents was
escalated by the judge, an escalation rate of 1.000, so the gate keeps them out of the corpus and
they never reach the score (``pdf-router-v2.md``, "One gate is arithmetic").

**Nothing opens a PDF in the map task.** Both native libraries this step touches run in a child
process the task is willing to lose, for reasons that are measured rather than precautionary and
that differ between them.

pdf-inspector itself is bought isolation for its *deadline*, not for a crash: no panic and no worker
death in 100,000 crawl PDFs on either architecture, but ``extract_pages_markdown_bytes`` has no page
cap, byte cap or deadline of its own and 17 of those documents ran past 30 seconds -- 1.77x slower at
1.17.0 than at 1.14.1, with a p99 ratio of 10.5x. In-process one of those holds a Zephyr map task
until its heartbeat expires, the task retries onto the same document, and the shard is lost. The
residual crash risk rides along for free: three unbounded-depth recursions over nested Form XObjects
remain in the crate, and a stack overflow there is a ``SIGSEGV`` no ``except`` can catch.

The rasteriser is bought isolation for exactly the crash. Rendering every page of the 100,000
document sample through MuPDF found one abort -- a ``SIGSEGV`` on page 48 of a 58-page document,
deterministic across re-renders into fresh processes -- against zero in 3,577,944 PDFium page
renders. Zephyr gives a shard three attempts, restarts it from row zero and has no poison-pill
detection, so a deterministic abort exhausts the budget and fails the stage permanently; at ~1 in
100,000 documents that is ~10 blocking documents crawl-wide. This step reads page geometry from
every fetched document, so it would meet all of them.

Each library gets its own round trip and therefore its own deadline, so a document that hangs one
does not cost the other's result.

``pdf_inspector`` and PyMuPDF are both imported inside the child alone -- the latter through
:mod:`~experiments.datakit.build_pdf_source.ocr_extract.render`'s own deferred imports. Both live in
marin-core's ``pdf`` extra, which the Zephyr workers get through ``pip_dependency_groups`` and the
entrypoint job does not -- its ``uv sync`` carries no extras. Since
:mod:`~experiments.datakit.build_pdf_source.pipeline` imports this module to build its DAG, a
module-scope import of either would kill the driver before it submitted anything.
"""

import json
import logging
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from enum import StrEnum
from functools import cache, partial
from importlib.metadata import version
from itertools import accumulate

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData, generate_id, make_split_writer
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions, strip_boilerplate
from experiments.datakit.build_pdf_source.common import FOCUS_CRAWL, PdfSourceData
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS, source_id
from experiments.datakit.build_pdf_source.extract import BOILERPLATE_OPTIONS, SOURCE_COLUMNS, keep_all
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderGeometry,
    RenderOptions,
    open_pdf,
    page_rectangles,
    render_geometry,
)

logger = logging.getLogger(__name__)

# Pinned, and checked inside the worker before a single document is read. The library is a native
# extension whose parsing limits, timings and failure rate move release to release -- 1.17.0 is
# 1.77x slower than 1.14.1 at extraction -- and the router's booster was fit on columns this build
# produced, so a different build silently shifts the feature distribution the threshold was
# calibrated against.
LIBRARY_VERSION = "1.17.0"

MODULE_NAME = "experiments.datakit.build_pdf_source.extract_inspector"
WORKER_FLAG = "--worker"

# The two things the child is asked to do, each in its own round trip so that one hanging or dying
# does not cost the other's result. They are separate for a measured reason on each side: the
# library's own p99 latency ratio is 10.5x and it has no deadline, and the rasteriser has a
# deterministic SIGSEGV on ~1 crawl document in 100,000.
OP_EXTRACT = "extract"
OP_GEOMETRY = "geometry"

# The deadline the evaluation measured against: 17 of 100,000 documents exceeded it on
# ``extract_pages_markdown_bytes`` alone. ``detect_pdf_bytes`` runs inside the same deadline and
# costs 0.461 ms/page against extraction's 7.709, so the measured rate carries. Generous against a
# library that claims single-digit milliseconds per page: a document still running after this long
# is a hang for any practical purpose.
CALL_DEADLINE = 30.0
_READ_CHUNK = 1 << 16

RENDER_OPTIONS = RenderOptions()

# A token this long is not a word. Long runs come from tables serialized without separators and from
# CID-mapped subsets that decode into one unbroken string, and both are extraction damage.
LONG_TOKEN_CHARS = 20
REPLACEMENT_CHAR = "�"

_TOKEN = re.compile(r"\S+")
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")

_COUNTER_PREFIX = "focus_crawl_pdf_inspector"


class InspectorStatus(StrEnum):
    """What pdf-inspector did with a document.

    ``EMPTY`` and ``FAILED`` are held apart because they are different facts about the corpus even
    though the router treats them alike. ``EMPTY`` is the library succeeding on a scan: 12,127 of the
    12,396 no-text documents in the 100k sample. ``FAILED`` is the 269 it refused, crashed on or ran
    past the deadline. Collapsing them would hide a library regression inside the corpus's scan rate.
    """

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


# Every statistic measured on the markdown pdf-inspector actually produced. Router v1 could only
# predict extraction damage from a page's fonts and geometry; these observe it.
OUTPUT_STATISTIC_NAMES: tuple[str, ...] = (
    "replacement_ratio",
    "alpha_ratio",
    "digit_ratio",
    "space_ratio",
    "newline_ratio",
    "single_char_token_ratio",
    "mean_token_length",
    "long_token_ratio",
    "repeat_line_ratio",
    "max_line_repeats",
    "empty_page_fraction",
    "chars_per_source_page",
    "pipe_row_ratio",
    "heading_ratio",
)

# The route's own columns, appended to the shared document record. Everything the router reads is
# here, which is why almost all of it is nullable: a document the library refused has provenance and
# an error and nothing else, and a null is what tells XGBoost the feature is missing rather than
# zero. ``inspector_ocr_reasons`` is carried as a JSON reason-to-page-count histogram because the
# library reports reasons per page and a variable-length index list is not a feature.
INSPECTOR_FIELDS: tuple[pa.Field, ...] = (
    pa.field("pdf_bytes", pa.int64(), nullable=False),
    # Null, not zero, when the geometry pass found no readable page: the router keeps such a
    # document on this route because nothing can render it, and a zero would read as "renders badly".
    pa.field("mean_render_dpi", pa.float32(), nullable=True),
    pa.field("pages_below_legibility_floor", pa.int32(), nullable=True),
    pa.field("inspector_pdf_type", pa.string(), nullable=True),
    pa.field("inspector_confidence", pa.float32(), nullable=True),
    pa.field("inspector_page_count", pa.int32(), nullable=True),
    pa.field("inspector_has_title", pa.bool_(), nullable=True),
    pa.field("inspector_ocr_reasons", pa.string(), nullable=True),
    pa.field("inspector_detect_pages_needing_ocr", pa.int32(), nullable=True),
    pa.field("inspector_extract_is_complex_layout", pa.bool_(), nullable=True),
    pa.field("inspector_extract_pages_needing_ocr", pa.int32(), nullable=True),
    pa.field("inspector_extract_pages_with_tables", pa.int32(), nullable=True),
    pa.field("inspector_extract_pages_with_columns", pa.int32(), nullable=True),
    pa.field("inspector_extracted_pages", pa.int32(), nullable=True),
    # The no-text gate's own input, stored so a routing decision can be re-derived from the corpus.
    pa.field("inspector_markdown_chars", pa.int64(), nullable=True),
    *(pa.field(f"inspector_output_{name}", pa.float64(), nullable=True) for name in OUTPUT_STATISTIC_NAMES),
    pa.field("inspector_error", pa.string(), nullable=True),
)

OUTPUT_SCHEMA = pa.schema([*PDF_DOCUMENT_FIELDS, *INSPECTOR_FIELDS])

# What the router reads back out of this step: identity, plus every signal column. Named here rather
# than derived from INSPECTOR_FIELDS so the projection is a contract the classify step can be
# checked against, and so ``text`` can never accidentally be pulled through it at corpus scale.
SIGNAL_COLUMNS: list[str] = [
    "warc_filename",
    "warc_record_offset",
    "content_digest",
    "url",
    "num_pages",
    *(field.name for field in INSPECTOR_FIELDS),
]

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# The library is internally parallel -- lopdf is built against rayon -- so a task is costed at two
# CPUs rather than one, and its child process is where the work happens. Task disk is unused: shards
# stream from object storage.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="4g")
# Phase 2 merge-sorts extracted text. No PDFs, no child processes, one CPU.
_NORMALIZE_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=2 * _TASKS_PER_WORKER, ram="64g", disk="32g")
_MAX_WORKERS = 32
# Not Zephyr's 1 GB default: a shared-pool coordinator holds both executions' shard, retry and
# result state, and across this pipeline family the default is what gets OOM-killed (exit 137) one
# task short of the end of a stage, after the work is already on disk.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
# A task holding a pathological document can legitimately go a long time without finishing one, and
# CALL_DEADLINE bounds how long that can be.
_HEARTBEAT_TIMEOUT = 30 * 60


# ---------------------------------------------------------------------------
# Measuring the output, which is the signal router v1 structurally could not have
# ---------------------------------------------------------------------------


def output_statistics(pages: list[str], source_pages: int) -> dict[str, float]:
    """Every statistic the router reads off one document's page markdown.

    One pass over the concatenated text for the character ratios and one over the lines for the
    structural ones. Ratios are per character or per line rather than per document so a long report
    and a flyer are on the same scale; the two quantities deliberately *not* normalized that way are
    ``chars_per_source_page``, which is the truncation predictor and has to be an absolute length
    (a page with 12,000 characters will not fit the VLM's completion budget, and the router should
    know that before it spends the render), and ``max_line_repeats``, which is a count because one
    line repeated 400 times is a decode loop whatever the document's length.

    An empty extraction returns defined zeros rather than nulls. It is the strongest signal in the
    table and it has to be distinguishable from a document this pass never reached.
    """
    text = "\n".join(pages)
    characters = len(text)
    if characters == 0:
        return {
            f"inspector_output_{name}": 0.0
            for name in OUTPUT_STATISTIC_NAMES
            if name not in ("empty_page_fraction", "mean_token_length")
        } | {
            "inspector_output_empty_page_fraction": 1.0,
            "inspector_output_mean_token_length": 0.0,
        }

    alpha = digits = spaces = newlines = replacements = 0
    for character in text:
        alpha += character.isalpha()
        digits += character.isdigit()
        spaces += character == " "
        newlines += character == "\n"
        replacements += character == REPLACEMENT_CHAR

    tokens = _TOKEN.findall(text)
    token_count = max(len(tokens), 1)
    lines = [line.strip() for line in text.split("\n")]
    content_lines = [line for line in lines if line]
    line_count = max(len(content_lines), 1)
    repeats = Counter(content_lines)

    return {
        "inspector_output_replacement_ratio": replacements / characters,
        "inspector_output_alpha_ratio": alpha / characters,
        "inspector_output_digit_ratio": digits / characters,
        "inspector_output_space_ratio": spaces / characters,
        "inspector_output_newline_ratio": newlines / characters,
        "inspector_output_single_char_token_ratio": sum(len(token) == 1 for token in tokens) / token_count,
        "inspector_output_mean_token_length": sum(len(token) for token in tokens) / token_count,
        "inspector_output_long_token_ratio": sum(len(token) > LONG_TOKEN_CHARS for token in tokens) / token_count,
        # Lines beyond the first occurrence of each distinct line: 0.0 for a document that never
        # repeats itself, approaching 1.0 for one that says the same thing over and over.
        "inspector_output_repeat_line_ratio": (line_count - len(repeats)) / line_count,
        "inspector_output_max_line_repeats": float(max(repeats.values(), default=0)),
        "inspector_output_empty_page_fraction": sum(not page.strip() for page in pages) / max(len(pages), 1),
        "inspector_output_chars_per_source_page": characters / max(source_pages, 1),
        "inspector_output_pipe_row_ratio": sum(bool(_PIPE_ROW.match(line)) for line in content_lines) / line_count,
        "inspector_output_heading_ratio": sum(bool(_HEADING.match(line)) for line in content_lines) / line_count,
    }


# ---------------------------------------------------------------------------
# The library, in a process the task is willing to lose
# ---------------------------------------------------------------------------


def _detect_signals(result) -> dict:
    """The classification signals ``detect_pdf_bytes`` reports, flattened for a row.

    ``pages_needing_ocr`` and friends arrive as page-index lists; the routing question is how much of
    the document is affected, so they are carried as counts. ``has_encoding_issues``,
    ``is_complex_layout``, ``pages_with_tables`` and ``pages_with_columns`` are *not* carried from
    this call: the evaluation found all four constant across all 100,000 documents in both 1.14.1 and
    1.17.0, so detect populates none of its declared layout signals and the extraction's own read of
    them is the only one worth storing.
    """
    reasons: Counter[str] = Counter()
    for page in result.ocr_reasons_by_page:
        reasons.update(page.reasons)
    return {
        "inspector_pdf_type": result.pdf_type,
        "inspector_confidence": float(result.confidence),
        "inspector_page_count": result.page_count,
        "inspector_has_title": result.title is not None,
        "inspector_ocr_reasons": json.dumps(dict(sorted(reasons.items()))),
        "inspector_detect_pages_needing_ocr": len(result.pages_needing_ocr),
    }


def _extract_signals(result) -> dict:
    """The extraction's own read of the layout signals, alongside the page markdown itself."""
    return {
        "inspector_extract_is_complex_layout": result.is_complex,
        "inspector_extract_pages_needing_ocr": len(result.pages_needing_ocr),
        "inspector_extract_pages_with_tables": len(result.pages_with_tables),
        "inspector_extract_pages_with_columns": len(result.pages_with_columns),
        "inspector_extracted_pages": len(result.pages),
        "pages": [page.markdown for page in result.pages],
    }


def read_exactly(stream, size: int) -> bytes:
    """Read exactly ``size`` bytes, or raise. Pipe reads are short whenever the writer is."""
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(min(_READ_CHUNK, size - len(buffer)))
        if not chunk:
            raise EOFError(f"stream closed after {len(buffer)} of {size} bytes")
        buffer.extend(chunk)
    return bytes(buffer)


def _extract_reply(payload: bytes) -> dict:
    """Both pdf-inspector calls against one document, each allowed to fail on its own.

    ``detect_pdf_bytes`` refusing a document the extraction reads fine must not discard the
    extraction, and the reverse is worth the same care, so neither failure aborts the other.

    ``BaseException`` and not ``Exception``: PyO3 derives ``PanicException`` from the former, and a
    panic reported as a worker death would turn a catchable failure into a respawn.
    """
    import pdf_inspector  # noqa: PLC0415 - the whole point is to import it out of process

    reply: dict = {}
    for name, call, flatten in (
        ("detect", pdf_inspector.detect_pdf_bytes, _detect_signals),
        ("extract", pdf_inspector.extract_pages_markdown_bytes, _extract_signals),
    ):
        try:
            reply.update(flatten(call(payload)))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            reply[f"{name}_error"] = f"{type(error).__name__}: {error}"[:500]
    return reply


def _geometry_reply(payload: bytes) -> dict:
    """Every page's size in points, read by the rasteriser and measured by nobody yet.

    The rectangles cross the pipe rather than the DPIs, so the arithmetic that turns them into the
    router's features stays in the parent where it is testable without a PDF library.
    """
    try:
        with open_pdf(payload) as document:
            return {"page_rectangles": page_rectangles(document)}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:
        return {"geometry_error": f"{type(error).__name__}: {error}"[:500]}


def worker_main() -> None:
    """Serve length-prefixed PDFs from stdin until the driver closes it."""
    import faulthandler  # noqa: PLC0415 - only the disposable child needs a fault handler

    installed = version("pdf-inspector")
    if installed != LIBRARY_VERSION:
        raise RuntimeError(f"this step is pinned to pdf-inspector {LIBRARY_VERSION}; {installed} is installed")
    faulthandler.enable()

    handlers = {OP_EXTRACT: _extract_reply, OP_GEOMETRY: _geometry_reply}
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = stdin.readline()
        if not header:
            return
        request = json.loads(header)
        payload = read_exactly(stdin, request["size"])
        stdout.write(json.dumps(handlers[request["op"]](payload)).encode() + b"\n")
        stdout.flush()


class InspectorWorker:
    """A pdf-inspector subprocess, bounded by a deadline and replaced whenever it stops answering.

    Deliberately ``subprocess`` rather than ``multiprocessing``: an Iris callable entrypoint runs at
    module top level of ``__main__`` with no ``if __name__ == "__main__"`` guard, so both ``spawn``
    and ``forkserver`` would re-execute the job body in every child.

    It replaces itself rather than being replaced by its caller, so the caller can hold one handle
    for the life of the process (see :func:`inspector_worker`). Starting one costs an interpreter
    and the crate's import, which is why a per-document child is not an option against a 7.7
    ms/page call.
    """

    def __init__(self, deadline: float = CALL_DEADLINE) -> None:
        self._deadline = deadline
        self._process: subprocess.Popen | None = None
        self._selector = selectors.DefaultSelector()
        self.spawns = 0
        self.start()

    def start(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", MODULE_NAME, WORKER_FLAG],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr stays on the job log so a panic or abort message survives the process.
            env=os.environ.copy(),
        )
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self.spawns += 1

    def stop(self) -> None:
        if self._process is None:
            return
        self._selector.unregister(self._process.stdout)
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._process = None

    def _death(self) -> str:
        """How the worker died, named rather than numbered."""
        code = self._process.poll()
        if code is not None and code < 0:
            return f"worker killed by {signal.Signals(-code).name}"
        return f"worker exited with {code}"

    def _read_reply(self, deadline: float) -> str | None:
        """One newline-terminated reply, or ``None`` on timeout or on the worker's EOF."""
        buffer = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                return None
            chunk = os.read(self._process.stdout.fileno(), _READ_CHUNK)
            if not chunk:
                return None
            buffer.extend(chunk)
            if b"\n" in buffer:
                return buffer.split(b"\n", 1)[0].decode()

    def call(self, op: str, pdf: bytes) -> dict:
        """Run one operation on one document, replacing the child if it does not come back.

        Returns the worker's reply, or a dict carrying ``<op>_error`` when the child died or ran
        past its deadline -- which the caller records against the document rather than raising,
        because on a crawl corpus that is data.
        """
        deadline = time.monotonic() + self._deadline
        try:
            self._process.stdin.write(json.dumps({"op": op, "size": len(pdf)}).encode() + b"\n")
            self._process.stdin.write(pdf)
            self._process.stdin.flush()
            line = self._read_reply(deadline)
        except (BrokenPipeError, OSError) as error:
            line, failure = None, f"{type(error).__name__}: {error}"
        else:
            failure = None

        if line is not None:
            return json.loads(line)

        # Either the child is gone or it is still inside the library. Both mean this process is no
        # longer usable: a worker mid-document cannot be handed the next one.
        died = self._process.poll() is not None
        reason = failure or (self._death() if died else f"no reply within {self._deadline:.0f}s")
        outcome = "worker_died" if died else "deadline_exceeded"
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{outcome}/{op}", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/worker_respawned", 1)
        self.stop()
        self.start()
        return {f"{op}_error": reason}


# ---------------------------------------------------------------------------
# One document, one row
# ---------------------------------------------------------------------------


def document_geometry(reply: dict, options: RenderOptions) -> RenderGeometry | None:
    """What the render budget would resolve this document to, or ``None`` if nothing can render it.

    A PDF the rasteriser cannot open, or one with no page it will accept, is not an error here. It is
    the fact that decides the document's route: the VLM route renders through this same library, so
    escalating a document nothing can open buys nothing, and the router keeps it on whatever text
    pdf-inspector managed instead. ``None`` and not a zero geometry, because a mean DPI of 0.0 reads
    as "renders far below the legibility floor", which is a different document.
    """
    if "geometry_error" in reply:
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/geometry_failed", 1)
        logger.debug("Could not measure render geometry: %s", reply["geometry_error"])
        return None
    measured = render_geometry(reply.get("page_rectangles") or (), options)
    return measured if measured.pages else None


def _blank_signals() -> dict:
    """Every route column absent, so a row is fully formed whatever the library did."""
    return dict.fromkeys(field.name for field in INSPECTOR_FIELDS)


def _error_summary(reply: dict) -> str | None:
    failures = (("detect", "detect_error"), ("extract", "extract_error"))
    return "; ".join(f"{name}: {reply[key]}" for name, key in failures if key in reply) or None


def extract_document(row: dict, reply: dict, geometry: RenderGeometry | None, boilerplate: BoilerplateOptions) -> dict:
    """Assemble one stored row from the library's reply and the document's geometry.

    The row is always complete: identity and provenance are present whatever happened, and the
    signal columns are null where the library produced nothing. That is what lets the router treat
    "no text" as a decision rather than as a missing row, and it is why this function never raises.

    Page furniture is stripped before the text is hashed into ``id``, so the id is computed over the
    text a consumer actually reads. The stripping does not overlap the library's own: 1.17.0's
    ``extract_pages_markdown_mem`` passes ``strip_headers_footers=false``, and running headers still
    reach its output on 12.1% of multi-page documents (``pdf-inspector-evaluation.md``).
    """
    pages: list[str] = reply.get("pages") or []
    # Per *source* page, so the denominator is the document's page count rather than the number of
    # pages the extraction produced -- a scan yields one and has forty. The rasteriser's count is
    # preferred because it is what the training table used; the library's is the fallback for the
    # documents where the rasteriser had no answer, which the study would simply have dropped.
    source_pages = (geometry.pages if geometry else 0) or reply.get("inspector_page_count") or len(pages)
    signals = _blank_signals() | {
        name: value for name, value in reply.items() if name in {field.name for field in INSPECTOR_FIELDS}
    }
    if "extract_error" not in reply:
        signals |= output_statistics(pages, source_pages)
        signals["inspector_markdown_chars"] = sum(len(page) for page in pages)

    stripped = strip_boilerplate(pages, boilerplate)
    # Give every page a trailing newline so the last line of one cannot fuse with the first line of
    # the next. After stripping, not before: a blank line at the foot of every page is itself a
    # repeated edge pattern, so adding the newlines first would have the boilerplate pass take them
    # straight back off again.
    normalized = [page if not page or page.endswith("\n") else page + "\n" for page in stripped.pages]
    text = "".join(normalized)

    if "extract_error" in reply:
        status = InspectorStatus.FAILED
    elif text.strip():
        status = InspectorStatus.SUCCESS
    else:
        status = InspectorStatus.EMPTY

    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{status}", 1)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_pages", len(pages))
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_characters", len(text))
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/boilerplate_lines_removed", stripped.lines_removed)

    return {
        "id": generate_id(text),
        "text": text,
        "source_id": source_id(row["warc_filename"], row["warc_record_offset"]),
        "source": FOCUS_CRAWL,
        "warc_filename": row["warc_filename"],
        "warc_record_offset": row["warc_record_offset"],
        "content_digest": row["content_digest"],
        "url": row["url"],
        "num_pages": geometry.pages if geometry else 0,
        "page_offsets": list(accumulate(len(page) for page in normalized)),
        "extraction_status": str(status),
        "extraction_error": _error_summary(reply),
        "boilerplate_lines_removed": stripped.lines_removed,
        **signals,
        "pdf_bytes": len(row["pdf"]),
        "mean_render_dpi": geometry.mean_dpi if geometry else None,
        "pages_below_legibility_floor": geometry.pages_below_floor if geometry else None,
        "inspector_error": _error_summary(reply),
    }


@cache
def inspector_worker(deadline: float) -> InspectorWorker:
    """One pdf-inspector child per task process, shared by every batch the process runs.

    Never shut down: it lives as long as the process. Rebuilding it per Parquet row group would pay
    an interpreter start and the crate's import for every few hundred documents, against a call that
    costs 7.7 ms/page. It replaces itself when a document kills it, so a long-lived handle is safe,
    and a child orphaned by a dying parent reads EOF on its stdin pipe and exits on its own.
    """
    return InspectorWorker(deadline)


def extract_batch(
    batch: pa.RecordBatch,
    render_options: RenderOptions,
    boilerplate: BoilerplateOptions,
) -> Iterator[dict]:
    """Extract one Parquet row group. Every line that opens a PDF runs in the child, not here.

    Two round trips per document rather than one, so that the rasteriser and the extractor get their
    own deadline and their own chance to die: neither of them can then cost the other's result. The
    extra cost is a second pipe write of the document's bytes against ~140 ms of parsing.
    """
    worker = inspector_worker(CALL_DEADLINE)
    for row in batch.to_pylist():
        extracted = worker.call(OP_EXTRACT, row["pdf"])
        geometry = document_geometry(worker.call(OP_GEOMETRY, row["pdf"]), render_options)
        yield extract_document(row, extracted, geometry, boilerplate)


def extract_pdf_text(output_path: str, source_output_path: str) -> NormalizedData:
    """Extract every fetched PDF, in two phases on one warm worker pool.

    Phase 1 maps the fetch shards and writes **raw per-source-shard Parquet** -- no shuffle -- with
    ``skip_existing``, so it is the checkpoint: a retry re-extracts only the shards whose raw file
    never landed. Phase 2 runs the one legitimate shuffle, the normalized format's global sort by
    content-hash ``id``, on the same Zephyr worker pool. The sort key is a hash of the extracted
    text, so the repartition cannot begin until extraction ends and is pure CPU-and-storage work.

    This is the shape :mod:`~experiments.datakit.build_pdf_source.extract_ocr` runs for the same
    reason, minus the fleet: there are no GPUs to release here, only child processes.
    """
    source = read_artifact(source_output_path, PdfSourceData)
    shards = sorted(str(shard) for shard in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob())
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")
    raw_dir = prefix_join(output_path, "raw")
    logger.info(
        "Extracting %d shards from %s with pdf-inspector %s", len(shards), source.main_output_dir, LIBRARY_VERSION
    )

    tallies: dict[str, int | float] = {}
    with ZephyrContext(
        name="focus-crawl-pdf-inspector",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        coordinator_resources=_COORDINATOR_RESOURCES,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ) as pool:
        extraction = (
            Dataset.from_list(shards)
            .load_parquet(columns=SOURCE_COLUMNS, batch_mode=True)
            .flat_map(partial(extract_batch, render_options=RENDER_OPTIONS, boilerplate=BOILERPLATE_OPTIONS))
            .write_parquet(
                prefix_join(raw_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
                schema=OUTPUT_SCHEMA,
                skip_existing=True,
            )
        )
        outcome = pool.execute(extraction, map_task_resources=_MAP_TASK_RESOURCES)
        tallies.update(outcome.counters)

        normalize = (
            Dataset.from_files(prefix_join(raw_dir, "*.parquet"))
            .load_parquet()
            .group_by(
                key=lambda record: record["id"],
                reducer=keep_all,
                sort_by=lambda record: record["id"],
                num_output_shards=len(shards),
            )
            .map_shard(make_split_writer(output_path, output_schema=OUTPUT_SCHEMA))
        )
        outcome = pool.execute(normalize, map_task_resources=_NORMALIZE_TASK_RESOURCES)
        tallies.update(outcome.counters)

    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=tallies,
    )


def inspector_extract_step(source: StepSpec) -> StepSpec:
    """Build the pdf-inspector extraction step, which runs over every fetched PDF.

    It depends on the fetch step alone. That is the shape change router v2 makes: extraction no
    longer waits on a routing decision, the routing decision waits on extraction.
    """
    return StepSpec(
        name="data/datakit/extract/common_crawl_focus_2026_22_pdf_inspector",
        deps=[source],
        hash_attrs={
            "library_version": LIBRARY_VERSION,
            "call_deadline": CALL_DEADLINE,
            # The geometry columns the router scores on are a function of the render budget, so the
            # budget is part of this step's identity even though nothing here renders a pixel.
            "max_visual_tokens": RENDER_OPTIONS.max_visual_tokens,
            "max_render_dpi": RENDER_OPTIONS.max_render_dpi,
            "legibility_floor_dpi": RENDER_OPTIONS.legibility_floor_dpi,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            "schema_version": 1,
        },
        fn=remote(
            partial(extract_pdf_text, source_output_path=source.output_path),
            resources=_DRIVER_RESOURCES,
            # The map tasks import pdf_inspector in a child process and pymupdf through the render
            # module's deferred imports; both live in the ``pdf`` extra, not in ``datakit``.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        raise SystemExit(f"{MODULE_NAME} is a pipeline step; run it through pipeline.py, or with {WORKER_FLAG}")
