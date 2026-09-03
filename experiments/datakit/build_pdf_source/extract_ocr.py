# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5: re-read the escalated PDFs with a vision model, from rendered pages.

A Zephyr map task reads its own Parquet shard, renders pages, posts them, and writes documents; the
model, the batching, and the queueing live behind the endpoint. The driver hands each task the
routing table's address and the endpoint address, and a task pulls its own PDF bytes from object
storage and reads the routing shard co-partitioned with them
(:func:`~experiments.datakit.build_pdf_source.classify.shard_routing`).

**Nothing opens a PDF in the map task.** The rasteriser runs in a child process the task is willing
to lose (:mod:`~experiments.datakit.build_pdf_source.ocr_extract.render_worker`), streaming pages
back one at a time. Each task overlaps rendering with waiting, and :func:`sender_fleet_size`
provisions enough tasks that their combined offered rate meets the engines' throughput.

The output is the same shape the pdf-inspector route produces -- one shard per fetched shard, over
the same shared columns -- so a consumer unions the two routes without knowing which extractor
produced a document. Running headers and footers are stripped by the same
:mod:`~experiments.datakit.build_pdf_source.boilerplate` pass, before the text is hashed into ``id``.
"""

import logging
import math
import os
import threading
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache, partial
from hashlib import sha256
from itertools import accumulate

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import generate_id
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.inference.iris import remote_inference
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions, strip_boilerplate
from experiments.datakit.build_pdf_source.classify import shard_routing
from experiments.datakit.build_pdf_source.common import (
    FOCUS_CRAWL,
    SHARD_PATTERN,
    SOURCE_FILE_COLUMN,
    PdfClassificationData,
    PdfDocumentsData,
    PdfSourceData,
)
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS, source_id
from experiments.datakit.build_pdf_source.extract import SOURCE_COLUMNS
from experiments.datakit.build_pdf_source.loop_repair import LoopOptions, repair_page
from experiments.datakit.build_pdf_source.ocr_extract import fleet
from experiments.datakit.build_pdf_source.ocr_extract.client import (
    DEFAULT_MAX_TOKENS,
    PROMPT_DOC2MD,
    OcrEndpoint,
    PageOcr,
    ocr_page,
)
from experiments.datakit.build_pdf_source.ocr_extract.fleet import MODEL, build_inference_config
from experiments.datakit.build_pdf_source.ocr_extract.render import RAISED_MAX_VISUAL_TOKENS, RenderOptions
from experiments.datakit.build_pdf_source.ocr_extract.render_worker import (
    PAGE_DEADLINE,
    RenderFailure,
    RenderWorker,
)

logger = logging.getLogger(__name__)

RENDER_OPTIONS = RenderOptions()
# The router's render policy for documents whose pages fall below the legibility floor at the
# default budget.
RAISED_RENDER_OPTIONS = RenderOptions(max_visual_tokens=RAISED_MAX_VISUAL_TOKENS)
BOILERPLATE_OPTIONS = BoilerplateOptions()
LOOP_OPTIONS = LoopOptions()

_COUNTER_PREFIX = "focus_crawl_pdf_ocr"


class OcrStatus(StrEnum):
    """Whether a document was OCR'd whole.

    ``PARTIAL`` covers every way a page can come back short: a render failure, the page budget, a
    failed request, the token cap, or a repaired repetition loop. The per-page counts in the record
    say which.
    """

    SUCCESS = "success"
    PARTIAL = "partial"


OCR_FIELDS: tuple[pa.Field, ...] = (
    pa.field("pages_ocred", pa.int32(), nullable=False),
    pa.field("pages_failed", pa.int32(), nullable=False),
    # Pages the model was cut off on at ``max_tokens``: the text is present but incomplete.
    pa.field("pages_truncated", pa.int32(), nullable=False),
    # Pages the PDF declares that never became a request: render failures plus page-budget
    # truncation.
    pa.field("pages_unrendered", pa.int32(), nullable=False),
    # The budget holds per-page cost constant and lets per-page resolution vary with paper size;
    # these two record what each document got.
    pa.field("mean_render_dpi", pa.float32(), nullable=False),
    pa.field("pages_below_legibility_floor", pa.int32(), nullable=False),
    pa.field("completion_tokens", pa.int32(), nullable=False),
    # Pages whose text was cut back because the model fell into a repetition loop. 1-based, in
    # reading order, as ``page_offsets`` indexes the text.
    pa.field("looped_pages", pa.list_(pa.int32()), nullable=False),
    # Characters the loop repair removed from ``text``.
    pa.field("loop_chars_dropped", pa.int32(), nullable=False),
)

OUTPUT_SCHEMA = pa.schema([*PDF_DOCUMENT_FIELDS, *OCR_FIELDS])

# A task delivers about one page per second: the rasteriser child, the request encode and every
# request thread share one cgroup-throttled CPU, so throughput scales by adding tasks, not threads.
# 32 threads also keeps the fleet's maximum offered in-flight under the proxy's pending cap.
_REQUEST_THREADS = 32
# How many rendered pages a task may hold. Twice the thread count keeps every thread fed while
# bounding encoded-page memory, and the child blocks writing a page nothing has read, so it runs at
# most one page ahead of this queue.
_PAGES_IN_FLIGHT = 2 * _REQUEST_THREADS

# The two rates that set the task:instance ratio, measured at the fleet's operating point.
_PAGES_PER_SECOND_PER_INSTANCE = 17.75
_PAGES_PER_SECOND_PER_TASK = 0.75

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# A task is almost entirely blocked on the endpoint, so it costs one CPU and multiplexes eight-deep
# per worker.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="40g", disk="32g")
# A task holding a long document can legitimately go a long time without finishing one.
_HEARTBEAT_TIMEOUT = 30 * 60


def _check_alive_bounded(session, timeout_seconds: float = 120.0) -> None:
    """``session.check_alive()``, bounded.

    The check is advisory (it distinguishes lost capacity from lost pages); an answer that cannot
    arrive within the bound is not worth holding GPUs for.
    """
    outcome: list[BaseException | None] = [None]

    def probe() -> None:
        try:
            session.check_alive()
        except BaseException as error:
            outcome[0] = error

    prober = threading.Thread(target=probe, name="check-alive", daemon=True)
    prober.start()
    prober.join(timeout_seconds)
    if prober.is_alive():
        logger.warning("check_alive did not return within %.0fs; proceeding to fleet teardown", timeout_seconds)
        return
    if outcome[0] is not None:
        raise outcome[0]


def sender_fleet_size(instances: int) -> tuple[int, int]:
    """How many map tasks and Zephyr workers keep a fleet of ``instances`` engines full.

    Returns ``(sender_tasks, max_workers)``: tasks enough that their offered rate meets the engines'
    throughput, with the fleet's in-flight budget as a floor.
    """
    rate_basis = math.ceil(instances * _PAGES_PER_SECOND_PER_INSTANCE / _PAGES_PER_SECOND_PER_TASK)
    inflight_basis = fleet.MAX_IN_FLIGHT * instances // _REQUEST_THREADS
    sender_tasks = max(1, rate_basis, inflight_basis)
    return sender_tasks, max(1, math.ceil(sender_tasks / _TASKS_PER_WORKER))


@cache
def _request_pool(threads: int) -> ThreadPoolExecutor:
    """One request pool per sender process, shared by every shard the process runs.

    Never shut down: rebuilding it per shard would drop the endpoint's keep-alive connections.
    """
    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ocr-page")


@cache
def render_worker(deadline: float) -> RenderWorker:
    """One rasteriser child per sender process, shared by every shard the process runs.

    Never shut down: it lives as long as the process and replaces itself when a document kills it.
    """
    return RenderWorker(deadline)


@dataclass
class _Document:
    """One document's pages as they come back from the endpoint.

    Pages are submitted in order and resolved in the same order, so appending to :attr:`pages` as
    futures complete keeps the document in reading order without tracking indices.
    """

    row: dict
    declared_pages: int
    submitted: int = 0
    closed: bool = False
    pages: list[str] = field(default_factory=list)
    dpis: list[float] = field(default_factory=list)
    failed: int = 0
    truncated: int = 0
    completion_tokens: int = 0
    looped_pages: list[int] = field(default_factory=list)
    loop_chars_dropped: int = 0
    first_error: str | None = None

    @property
    def complete(self) -> bool:
        """Every page submitted and every submitted page resolved."""
        return self.closed and len(self.pages) == self.submitted

    def absorb(self, future: "Future[PageOcr]", loop: LoopOptions) -> None:
        """Record one page's result, keeping a failed page as an empty page.

        The exception is not propagated: the request has already exhausted its retries, and one
        unreadable page is recorded on the row and counted rather than failing the shard.
        """
        try:
            page = future.result()
        except Exception as error:
            self.failed += 1
            self.pages.append("")
            self.first_error = self.first_error or f"{type(error).__name__}: {error}"
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_request_failed", 1)
            # Also by type, so the counters say how pages were lost.
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_request_failed/{type(error).__name__}", 1)
            return
        repair = repair_page(page.text, page.truncated, loop)
        self.pages.append(repair.text)
        self.completion_tokens += page.completion_tokens
        if page.truncated:
            # The page is real but incomplete.
            self.truncated += 1
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_truncated", 1)
        if repair.looped:
            self.looped_pages.append(len(self.pages))
            self.loop_chars_dropped += repair.dropped_chars
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_looped", 1)
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/loop_chars_dropped", repair.dropped_chars)
            if not repair.text:
                # The loop began before any transcription did, so the page carries nothing.
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_looped_emptied", 1)

    def record(self, boilerplate: BoilerplateOptions, floor_dpi: float) -> dict | None:
        """Build the output row, or ``None`` if the document has no text worth keeping."""
        if not any(page.strip() for page in self.pages):
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/empty_document_filtered", 1)
            return None

        stripped = strip_boilerplate(self.pages, boilerplate)
        # A trailing newline per page so the last line of one cannot fuse with the first of the next.
        # After stripping: a blank foot line on every page is itself a repeated edge pattern.
        pages = [page if not page or page.endswith("\n") else page + "\n" for page in stripped.pages]
        text = "".join(pages)
        if not text.strip():
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/empty_after_boilerplate", 1)
            return None

        unrendered = max(0, self.declared_pages - self.submitted)
        ocred = self.submitted - self.failed
        whole = self.failed == 0 and unrendered == 0 and self.truncated == 0 and not self.looped_pages
        status = OcrStatus.SUCCESS if whole else OcrStatus.PARTIAL
        below_floor = sum(1 for dpi in self.dpis if dpi < floor_dpi)

        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_pages", ocred)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_characters", len(text))
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/completion_tokens", self.completion_tokens)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/pages_unrendered", unrendered)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/pages_below_legibility_floor", below_floor)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/boilerplate_lines_removed", stripped.lines_removed)
        if status is OcrStatus.PARTIAL:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/partial_extraction", 1)

        return {
            "id": generate_id(text),
            "text": text,
            "source_id": source_id(self.row["warc_filename"], self.row["warc_record_offset"]),
            "source": FOCUS_CRAWL,
            "warc_filename": self.row["warc_filename"],
            "warc_record_offset": self.row["warc_record_offset"],
            "content_digest": self.row["content_digest"],
            "url": self.row["url"],
            "num_pages": self.submitted,
            "page_offsets": list(accumulate(len(page) for page in pages)),
            "extraction_status": str(status),
            "extraction_error": self._error_summary(unrendered),
            "boilerplate_lines_removed": stripped.lines_removed,
            "pages_ocred": ocred,
            "pages_failed": self.failed,
            "pages_truncated": self.truncated,
            "pages_unrendered": unrendered,
            "mean_render_dpi": sum(self.dpis) / len(self.dpis) if self.dpis else 0.0,
            "pages_below_legibility_floor": below_floor,
            "completion_tokens": self.completion_tokens,
            "looped_pages": list(self.looped_pages),
            "loop_chars_dropped": self.loop_chars_dropped,
        }

    def _error_summary(self, unrendered: int) -> str | None:
        parts = []
        if self.truncated:
            parts.append(f"{self.truncated} of {self.submitted} pages hit the token cap and were cut off")
        if self.looped_pages:
            parts.append(
                f"{len(self.looped_pages)} of {self.submitted} pages repeated themselves and were "
                f"cut back, dropping {self.loop_chars_dropped} characters"
            )
        if unrendered:
            parts.append(f"{unrendered} of {self.declared_pages} pages were not rendered")
        if self.failed:
            parts.append(f"{self.failed} of {self.submitted} page requests failed: {self.first_error}")
        return "; ".join(parts) or None


def _count_render_failure(reason: str) -> None:
    """One document lost to the rasteriser, counted in total and by reason."""
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/render_failed", 1)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/render_failed/{reason}", 1)


def ocr_batch(
    batch: pa.RecordBatch,
    routing_dir: str | None,
    endpoint: OcrEndpoint,
    render_options: RenderOptions,
    raised_render_options: RenderOptions,
    boilerplate: BoilerplateOptions,
    loop: LoopOptions,
) -> Iterator[dict]:
    """OCR the OCR-routed documents in one Parquet row group.

    Rendering and waiting are overlapped: a page is submitted the moment the rasteriser's child
    streams it back, and the next document starts rendering while the previous one is in flight.
    At :data:`_PAGES_IN_FLIGHT` outstanding pages the task waits for the oldest before submitting
    another. Documents are emitted in the order they were read.

    ``routing_dir`` is the routing table: the batch came from one fetched shard, and its decisions
    are the routing shard of the same name. A document the router kept is skipped, and a document
    the table does not know is an error. ``None`` OCRs every document in the shard. A document
    flagged by the router's render policy is rendered at the raised budget, and the endpoint is
    rebuilt for it because the request restates the budget as ``max_pixels``.
    """
    if not batch.num_rows:
        return
    routing = None
    if routing_dir is not None:
        routing = shard_routing(routing_dir, os.path.basename(batch.column(SOURCE_FILE_COLUMN)[0].as_py()))
    budgets = {render_options.max_visual_tokens: (render_options, endpoint)}
    budgets[raised_render_options.max_visual_tokens] = (
        raised_render_options,
        replace(endpoint, max_visual_tokens=raised_render_options.max_visual_tokens),
    )
    pool = _request_pool(_REQUEST_THREADS)
    worker = render_worker(PAGE_DEADLINE)
    inflight: deque[tuple[_Document, Future[PageOcr]]] = deque()
    documents: deque[_Document] = deque()

    def resolve_oldest() -> None:
        document, future = inflight.popleft()
        document.absorb(future, loop)

    def emit_ready() -> Iterator[dict]:
        while documents and documents[0].complete:
            record = documents.popleft().record(boilerplate, render_options.legibility_floor_dpi)
            if record is not None:
                yield record

    for row in batch.to_pylist():
        key = (row["warc_filename"], row["warc_record_offset"])
        budget = render_options.max_visual_tokens
        if routing is not None:
            decision = routing.get(key)
            if decision is None:
                raise ValueError(f"{row['url']} ({key[0]}:{key[1]}) has no routing decision under {routing_dir}")
            if not decision.needs_ocr:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/skipped_inspector_route", 1)
                continue
            budget = decision.render_visual_tokens
        if budget not in budgets:
            raise ValueError(
                f"{row['url']} is routed at a {budget}-token render budget, but this step renders at "
                f"{sorted(budgets)}; the routing table was not built for these render options"
            )
        options, page_endpoint = budgets[budget]
        if options is raised_render_options:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/raised_render_budget", 1)

        try:
            with worker.render(row["pdf"], options) as stream:
                document = _Document(row=row, declared_pages=stream.declared_pages)
                documents.append(document)
                try:
                    for page in stream:
                        inflight.append((document, pool.submit(ocr_page, page_endpoint, _REQUEST_THREADS, page)))
                        document.submitted += 1
                        document.dpis.append(page.dpi)
                        while len(inflight) >= _PAGES_IN_FLIGHT:
                            resolve_oldest()
                            yield from emit_ready()
                finally:
                    # Whatever went wrong mid-render, the document has to be closed or it would sit
                    # in the queue forever and take every document behind it with it.
                    document.closed = True
        except RenderFailure as error:
            # The child has already been replaced and the pages it did stream are kept; the failure
            # is data, counted under the name it carries.
            _count_render_failure(error.reason)
            logger.warning("Could not render %s (%s): %s", row["url"], error.reason, error)
        except Exception as error:
            _count_render_failure(type(error).__name__)
            logger.warning("Could not render %s: %s", row["url"], error)
        yield from emit_ready()

    while inflight:
        resolve_oldest()
        yield from emit_ready()
    yield from emit_ready()


def ocr_pdf_text(
    output_path: str,
    source_output_path: str,
    classification_output_path: str | None = None,
    *,
    ocr_route_only: bool = True,
    instances: int = fleet.INSTANCES,
    partition: tuple[int, int] = (0, 1),
) -> PdfDocumentsData:
    """Run the OCR route: one map over the fetched shards, holding the fleet only while it runs.

    The map writes one output shard per fetched shard inside the ``remote_inference`` context, so
    the fleet is released when the last page lands. The output is also the checkpoint: the map
    writes with ``skip_existing``, so a retry re-OCRs only the shards whose file never landed, and
    with every shard present the fleet is never started.

    ``partition`` is ``(index, count)`` over the sorted source shards, so several of these steps can
    run side by side, each with its own fleet, over disjoint slices of the corpus.
    """
    partition_index, partition_count = partition
    source = read_artifact(source_output_path, PdfSourceData)
    routing_dir = None
    if ocr_route_only:
        if classification_output_path is None:
            raise ValueError("ocr_route_only requires classification_output_path")
        routing_dir = read_artifact(classification_output_path, PdfClassificationData).main_output_dir

    shards = sorted(str(shard) for shard in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob())
    shards = shards[partition_index::partition_count]
    num_shards = len(shards)
    if not num_shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")

    sender_tasks, max_workers = sender_fleet_size(instances)
    logger.info(
        "OCR partition %d/%d: %d shards, %d instances, %d sender tasks on %d workers",
        partition_index,
        partition_count,
        num_shards,
        instances,
        sender_tasks,
        max_workers,
    )

    output_dir = prefix_join(output_path, "outputs/main")
    shards_present = len(StoragePath(prefix_join(output_dir, "*.parquet")).glob())
    if shards_present >= num_shards:
        logger.info("All %d output shards already written under %s; skipping the OCR phase", num_shards, output_dir)
        return PdfDocumentsData(main_output_dir=output_dir, counters={})

    with remote_inference(build_inference_config(instances=instances)) as session:
        endpoint = OcrEndpoint(
            base_url=session.model.endpoint.base_url,
            model=session.model.endpoint.model,
            max_visual_tokens=RENDER_OPTIONS.max_visual_tokens,
        )
        logger.info("OCR endpoint ready at %s (%s)", endpoint.base_url, session.backend_name)

        pipeline = (
            Dataset.from_list(shards)
            # The reader injects the shard's own path, which is how a task finds the routing shard
            # co-partitioned with it; a projection has to name the injected column to keep it.
            .load_parquet(
                columns=[*SOURCE_COLUMNS, SOURCE_FILE_COLUMN],
                batch_mode=True,
                include_file_paths=True,
                file_path_column=SOURCE_FILE_COLUMN,
            )
            .flat_map(
                partial(
                    ocr_batch,
                    routing_dir=routing_dir,
                    endpoint=endpoint,
                    render_options=RENDER_OPTIONS,
                    raised_render_options=RAISED_RENDER_OPTIONS,
                    boilerplate=BOILERPLATE_OPTIONS,
                    loop=LOOP_OPTIONS,
                )
            )
            .write_parquet(prefix_join(output_dir, SHARD_PATTERN), schema=OUTPUT_SCHEMA, skip_existing=True)
        )
        outcome = ZephyrContext(
            name=f"focus-crawl-pdf-ocr-{partition_index}",
            resources=_WORKER_RESOURCES,
            max_workers=max_workers,
            stage_runner_factory=SubprocessRunner,
            heartbeat_timeout=_HEARTBEAT_TIMEOUT,
        ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
        # A fleet that died partway through shows up as failed pages, not as an error. Raise only
        # when it cost something: an instance exiting after the last shard landed is not a failure.
        lost_pages = int(outcome.counters.get(f"{_COUNTER_PREFIX}/page_request_failed", 0))
        try:
            _check_alive_bounded(session)
        except Exception as error:
            if lost_pages:
                raise
            logger.warning("An inference job ended before the fleet was released: %s", error)
            logger.warning("No page request failed, so the extracted corpus is complete.")

    return PdfDocumentsData(main_output_dir=output_dir, counters=dict(outcome.counters))


def ocr_extract_step(source: StepSpec, classification: StepSpec) -> StepSpec:
    """Build the OCR extraction step for the router's OCR route."""
    return StepSpec(
        name="data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr",
        deps=[source, classification],
        hash_attrs={
            # Change the model or the prompt and the text changes.
            "model": MODEL,
            "prompt_digest": sha256(PROMPT_DOC2MD.encode("utf-8")).hexdigest()[:16],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_visual_tokens": RENDER_OPTIONS.max_visual_tokens,
            # The render policy changes what the flagged documents are read from.
            "raised_max_visual_tokens": RAISED_RENDER_OPTIONS.max_visual_tokens,
            "max_render_dpi": RENDER_OPTIONS.max_render_dpi,
            "max_pages": RENDER_OPTIONS.max_pages,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            # Loop repair rewrites the stored text. Its thresholds were calibrated against
            # ``max_tokens`` above: a runaway that no longer hits the cap is not marked truncated.
            "loop_min_page_chars": LOOP_OPTIONS.min_page_chars,
            "loop_min_loop_chars": LOOP_OPTIONS.min_loop_chars,
            "loop_min_loop_fraction": LOOP_OPTIONS.min_loop_fraction,
            "loop_max_trailing_chars": LOOP_OPTIONS.max_trailing_chars,
            "loop_min_counter_score": LOOP_OPTIONS.min_counter_score,
            "loop_min_salvage_prefix": LOOP_OPTIONS.min_salvage_prefix,
            "schema_version": 2,
        },
        fn=remote(
            partial(
                ocr_pdf_text,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            # The sender tasks rasterise with pypdfium2 and encode with pillow, both in the ``pdf`` extra.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )
