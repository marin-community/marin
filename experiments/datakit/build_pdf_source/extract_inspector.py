# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3: read every fetched PDF's text layer with pdf-inspector, and measure what came out.

This step has two outputs in one table. Every fetched PDF gets exactly one row:

* the shared document record -- text, provenance, page offsets, extraction status -- which is what
  :mod:`~experiments.datakit.build_pdf_source.combine_routes` unions into the corpus, and
* :data:`INSPECTOR_FIELDS`, the raw columns
  :mod:`~experiments.datakit.build_pdf_source.classify` turns into the router's 43 features.

**Nothing opens a PDF in the map task.** Both native libraries this step touches run in a child
process the task is willing to lose for crash and timeout isolation.

``pdf_inspector`` and the rasteriser are both imported inside the child alone -- the latter
through :mod:`~experiments.datakit.build_pdf_source.ocr_extract.render`'s own deferred imports. Both
live in marin-core's ``pdf`` extra, which the Zephyr workers get through ``pip_dependency_groups``
and the entrypoint job does not -- its ``uv sync`` carries no extras. Since
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
from marin.datakit.normalize import generate_id
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions, strip_boilerplate
from experiments.datakit.build_pdf_source.child_framing import READ_CHUNK, read_frame, write_frame
from experiments.datakit.build_pdf_source.common import (
    FOCUS_CRAWL,
    MAIN_OUTPUT_SUBDIR,
    SHARD_PATTERN,
    PdfDocumentsData,
    PdfSourceData,
)
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS, source_id
from experiments.datakit.build_pdf_source.extract import BOILERPLATE_OPTIONS, RENDER_OPTIONS, SOURCE_COLUMNS
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    RenderGeometry,
    RenderOptions,
    open_pdf,
    page_rectangles,
    render_geometry,
)
from experiments.datakit.build_pdf_source.ocr_extract.render_worker import DEADLINE_EXCEEDED, WORKER_DIED

logger = logging.getLogger(__name__)

# Pinned, and checked inside the worker before a single document is read: the router's booster was
# fit on columns this build produced, so a different build silently shifts the feature distribution
# the threshold was calibrated against.
LIBRARY_VERSION = "1.17.0"

MODULE_NAME = "experiments.datakit.build_pdf_source.extract_inspector"
WORKER_FLAG = "--worker"

# The two things the child is asked to do, each in its own round trip so that one hanging or dying
# does not cost the other's result.
OP_EXTRACT = "extract"
OP_GEOMETRY = "geometry"

CALL_DEADLINE = 30.0

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
    though the router treats them alike. Collapsing them would hide a library regression inside the
    corpus's scan rate.
    """

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


# Every statistic measured on the markdown pdf-inspector actually produced.
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

# The route's own columns, appended to the shared document record.
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
# CPUs rather than one, and its child process is where the work happens.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=2 * _TASKS_PER_WORKER, ram="64g", disk="32g")
_MAX_WORKERS = 32
# A task holding a pathological document can legitimately go a long time without finishing one, and
# CALL_DEADLINE bounds how long that can be.
_HEARTBEAT_TIMEOUT = 30 * 60


# ---------------------------------------------------------------------------
# Measuring the output
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
    """The classification signals ``detect_pdf_bytes`` reports, flattened for a row."""
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


def _extract_reply(payload: bytes) -> dict:
    """Both pdf-inspector calls against one document, each allowed to fail on its own.

    ``detect_pdf_bytes`` refusing a document the extraction reads fine must not discard the
    extraction, and the reverse is worth the same care.
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
        # BaseException: PyO3 derives PanicException from it, and a panic reported as a worker death
        # would turn a catchable failure into a respawn.
        except BaseException as error:
            reply[f"{name}_error"] = f"{type(error).__name__}: {error}"[:500]
    return reply


def _geometry_reply(payload: bytes) -> dict:
    """Every page's size in points."""
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
        frame = read_frame(stdin)
        if frame is None:
            return
        request, payload = frame
        write_frame(stdout, handlers[request["op"]](payload))


class InspectorWorker:
    """A pdf-inspector subprocess, bounded by a deadline and replaced whenever it stops answering.

    ``subprocess`` rather than ``multiprocessing``, because an Iris callable entrypoint runs at
    module top level of ``__main__`` and ``spawn`` would re-execute the job body in every child.
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
        # Per child: whether its stdout has reached EOF, which is what distinguishes a death from a
        # deadline in :meth:`call`.
        self._eof = False
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
        """How the worker died."""
        code = self._process.wait()
        if code < 0:
            return f"worker killed by {signal.Signals(-code).name}"
        return f"worker exited with {code}"

    def _read_reply(self, deadline: float) -> str | None:
        """One newline-terminated reply, or ``None`` on timeout or on the worker's EOF."""
        buffer = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                return None
            chunk = os.read(self._process.stdout.fileno(), READ_CHUNK)
            if not chunk:
                self._eof = True
                return None
            buffer.extend(chunk)
            if b"\n" in buffer:
                return buffer.split(b"\n", 1)[0].decode()

    def call(self, op: str, pdf: bytes) -> dict:
        """Run one operation on one document, replacing the child if it does not come back."""
        deadline = time.monotonic() + self._deadline
        try:
            write_frame(self._process.stdin, {"op": op, "size": len(pdf)}, pdf)
            line = self._read_reply(deadline)
        except (BrokenPipeError, OSError) as error:
            # A write that fails means the child is already gone; its stdout would read EOF too.
            self._eof = True
            line, failure = None, f"{type(error).__name__}: {error}"
        else:
            failure = None

        if line is not None:
            return json.loads(line)

        # Either the child is gone or it is still inside the library. Both mean this process is no
        # longer usable: a worker mid-document cannot be handed the next one.
        died = self._eof
        reason = failure or (self._death() if died else f"no reply within {self._deadline:.0f}s")
        outcome = WORKER_DIED if died else DEADLINE_EXCEEDED
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{outcome}/{op}", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/worker_respawned", 1)
        self.stop()
        self.start()
        return {f"{op}_error": reason}


# ---------------------------------------------------------------------------
# One document, one row
# ---------------------------------------------------------------------------


def document_geometry(reply: dict, options: RenderOptions) -> RenderGeometry | None:
    """What the render budget would resolve this document to, or ``None`` if nothing can render it."""
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
    text a consumer actually reads.
    """
    pages: list[str] = reply.get("pages") or []
    # Per *source* page, so the denominator is the document's page count rather than the number of
    # pages the extraction produced.
    source_pages = (geometry.pages if geometry else 0) or reply.get("inspector_page_count") or len(pages)
    signals = _blank_signals() | {
        name: value for name, value in reply.items() if name in {field.name for field in INSPECTOR_FIELDS}
    }
    if "extract_error" not in reply:
        signals |= output_statistics(pages, source_pages)
        signals["inspector_markdown_chars"] = sum(len(page) for page in pages)

    stripped = strip_boilerplate(pages, boilerplate)
    # Every page ends in a newline so two pages cannot fuse; added after stripping, since a trailing
    # blank line on every page would itself read as boilerplate.
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
    """One pdf-inspector child per task process, shared by every batch the process runs and never
    shut down."""
    return InspectorWorker(deadline)


def extract_batch(
    batch: pa.RecordBatch,
    render_options: RenderOptions,
    boilerplate: BoilerplateOptions,
) -> Iterator[dict]:
    """Extract one Parquet row group. Every line that opens a PDF runs in the child, not here.

    Two round trips per document rather than one, so that the rasteriser and the extractor get their
    own deadline and their own chance to die.
    """
    worker = inspector_worker(CALL_DEADLINE)
    for row in batch.to_pylist():
        extracted = worker.call(OP_EXTRACT, row["pdf"])
        geometry = document_geometry(worker.call(OP_GEOMETRY, row["pdf"]), render_options)
        yield extract_document(row, extracted, geometry, boilerplate)


def extract_pdf_text(output_path: str, source_output_path: str) -> PdfDocumentsData:
    """Extract every fetched PDF: one map over the fetched shards, one output shard per input shard.

    The output shard keeps its input's basename, which is what keeps the routing table this feeds
    co-partitioned with the fetch; ``skip_existing`` makes the output its own checkpoint.
    """
    source = read_artifact(source_output_path, PdfSourceData)
    shards = sorted(str(shard) for shard in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob())
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")
    output_dir = prefix_join(output_path, MAIN_OUTPUT_SUBDIR)
    logger.info(
        "Extracting %d shards from %s with pdf-inspector %s", len(shards), source.main_output_dir, LIBRARY_VERSION
    )

    pipeline = (
        Dataset.from_list(shards)
        .load_parquet(columns=SOURCE_COLUMNS, batch_mode=True)
        .flat_map(partial(extract_batch, render_options=RENDER_OPTIONS, boilerplate=BOILERPLATE_OPTIONS))
        .write_parquet(prefix_join(output_dir, SHARD_PATTERN), schema=OUTPUT_SCHEMA, skip_existing=True)
    )
    outcome = ZephyrContext(
        name="focus-crawl-pdf-inspector",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
    return PdfDocumentsData(main_output_dir=output_dir, counters=dict(outcome.counters))


def inspector_extract_step(source: StepSpec) -> StepSpec:
    """Build the pdf-inspector extraction step, which runs over every fetched PDF."""
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
            # 2: one unsorted shard per fetched shard, named after it. 1 was the normalized layout,
            # sorted by content hash into as many shards as there were fetched files.
            "schema_version": 2,
        },
        fn=remote(
            partial(extract_pdf_text, source_output_path=source.output_path),
            resources=_DRIVER_RESOURCES,
            # The map tasks import pdf_inspector in a child process and pypdfium2 through the
            # render module's deferred imports; both live in the ``pdf`` extra, not in ``datakit``.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        worker_main()
    else:
        raise SystemExit(f"{MODULE_NAME} is a pipeline step; run it through pipeline.py, or with {WORKER_FLAG}")
