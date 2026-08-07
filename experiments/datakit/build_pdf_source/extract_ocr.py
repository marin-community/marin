# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6: OCR the PDFs the router could not read embedded text from.

This is the expensive route. In the 10% sample it is 101,332 of 315,776 classified documents and
2.0M of 5.6M pages, and every one of those pages goes through a vision model instead of a text
parser. At the measured 71 pages/s per GB200 node that is about 8 node-hours for the sample.

**The senders are thin, and that is the design.** A Zephyr map task reads its own Parquet shard,
renders pages, posts them, and writes documents; the model, the batching, and the queueing live
behind the endpoint. Nothing large moves through the driver -- it broadcasts the routing key set and
the endpoint address, both tiny, and every task pulls its own PDF bytes straight from object
storage. The one thing a sender cannot delegate is rendering: the endpoint speaks OpenAI chat
completions and takes images, so it cannot read Parquet or open a PDF. That cost is priced from
measurement -- a task delivers ~0.75 pages/s once rendering, encoding, and its request threads
share one cgroup-throttled CPU on the Grace-heavy worker fleet -- so a fleet is fed at about 24
one-CPU sender tasks per GPU (see the sizing block below for why the naive per-core render rate
overstates this ~17x).

What the senders *must* get right is keeping the fleet full. Each task overlaps rendering with
waiting rather than alternating between them, and :func:`sender_fleet_size` provisions enough
tasks that their combined offered rate meets the engines' throughput with the fleet's in-flight
budget as a floor.

The output is the same :class:`~marin.datakit.normalize.NormalizedData` shape the docling route
produces, over the same shared columns, so a consumer joins the two routes without knowing which
extractor produced a document. Running headers and footers are stripped by the same
:mod:`~experiments.datakit.build_pdf_source.boilerplate` pass, before the text is hashed into ``id``.
"""

import logging
import math
import threading
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, partial
from hashlib import sha256
from itertools import accumulate

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import (
    NormalizedData,
    generate_id,
    make_split_writer,
)
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.inference.iris import remote_inference
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions, strip_boilerplate
from experiments.datakit.build_pdf_source.classify import routing_keys
from experiments.datakit.build_pdf_source.common import FOCUS_CRAWL, PdfClassificationData, PdfSourceData
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS, source_id
from experiments.datakit.build_pdf_source.extract import keep_all
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
from experiments.datakit.build_pdf_source.ocr_extract.render import RenderOptions, iter_rendered_pages, open_pdf

logger = logging.getLogger(__name__)

RENDER_OPTIONS = RenderOptions()
BOILERPLATE_OPTIONS = BoilerplateOptions()
LOOP_OPTIONS = LoopOptions()

_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]

_COUNTER_PREFIX = "focus_crawl_pdf_ocr"


class OcrStatus(StrEnum):
    """Whether a document was OCR'd whole.

    ``PARTIAL`` covers every way a page can come back short: MuPDF declining to render it, the page
    budget truncating a very long document, the request failing after its retries, the model hitting
    its token cap part-way down the page, or the model falling into a repetition loop whose output
    was cut back to the transcription in front of it. The per-page counts in the record say which.

    The last two are the ones worth watching, because they are the only ones that yield text that
    looks complete. A page dropped for any other reason is empty and obvious; a page cut off at
    ``max_tokens`` is ordinary Markdown that simply stops, and nothing but ``pages_truncated`` says
    so; a repaired page reads as a clean short page and only ``looped_pages`` says otherwise.
    """

    SUCCESS = "success"
    PARTIAL = "partial"


OCR_FIELDS: tuple[pa.Field, ...] = (
    pa.field("pages_ocred", pa.int32(), nullable=False),
    pa.field("pages_failed", pa.int32(), nullable=False),
    # Pages the model was cut off on at ``max_tokens``. The text is present but incomplete, which
    # no other field would reveal -- a truncated page is an ordinary 200 with a shorter body.
    pa.field("pages_truncated", pa.int32(), nullable=False),
    # Pages the PDF declares that never became a request: render failures plus page-budget
    # truncation. Non-zero here means the document is incomplete, not merely imperfect.
    pa.field("pages_unrendered", pa.int32(), nullable=False),
    # The budget holds per-page *cost* constant, which means it lets per-page *resolution* vary with
    # paper size. These two carry that consequence into the corpus so it can be audited rather than
    # inferred: a document whose pages landed below the legibility floor was read at a resolution
    # the model is not reliable at.
    pa.field("mean_render_dpi", pa.float32(), nullable=False),
    pa.field("pages_below_legibility_floor", pa.int32(), nullable=False),
    pa.field("completion_tokens", pa.int32(), nullable=False),
    # Pages whose text was cut back because the model fell into a repetition loop. 1-based, in
    # reading order. Listed rather than counted: a consumer excluding repaired pages needs to know
    # which ones, and ``page_offsets`` already indexes the text that way.
    pa.field("looped_pages", pa.list_(pa.int32()), nullable=False),
    # Characters the repair removed. Non-zero means ``text`` is shorter than what the model
    # returned, which nothing else in the record would reveal.
    pa.field("loop_chars_dropped", pa.int32(), nullable=False),
)

OUTPUT_SCHEMA = pa.schema([*PDF_DOCUMENT_FIELDS, *OCR_FIELDS])

# Sender sizing, from the measured economics of a task rather than from either intuition that
# preceded them. A task delivers about one page per second -- not the 13.2 a lone unthrottled
# render thread manages -- because the render loop, the ~2 MB JSON/base64 encode per request, and
# every waking request thread all contend for one cgroup-throttled CPU. Threads make that worse,
# not better (128 threads measured 0.7 pages/s against 64's 1.5 on x86), so throughput scales by
# adding TASKS -- each its own process; MuPDF holds the GIL through a render, so threads cannot
# scale rendering at all. 32 threads is ample cover for the ~17 requests a task actually keeps in
# flight (its rate times the ~21s page latency), and it keeps the whole sender fleet's *maximum*
# offered in-flight under the proxy's pending cap -- a rate-sized fleet at 64 threads could
# collectively exceed it when queueing inflates latency, and past the cap the proxy sheds load as
# 429s that consume page-request retries.
_REQUEST_THREADS = 32
# How many rendered pages a task may hold. Twice the thread count keeps every thread fed while
# bounding encoded-page memory to roughly 90 MB per task.
_PAGES_IN_FLIGHT = 2 * _REQUEST_THREADS

# The two rates that set the task:instance ratio. Engine-side throughput at the operating point is
# the sweep's. Task-side is the third ceiling bench: 0.78 pages/s per task on a fleet ~70% Grace
# workers, half the x86-measured 1.49 -- ARM tasks degrade harder under thread contention than the
# serial render probe predicted. 0.75 plans for the ARM-heavy fleet the full-sample run will get.
_PAGES_PER_SECOND_PER_INSTANCE = 17.75
_PAGES_PER_SECOND_PER_TASK = 0.75

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# A task is almost entirely blocked on the endpoint, so it costs one CPU and multiplexes eight-deep
# per worker. The fleet's total CPU is several times the measured render requirement, which is the
# headroom for JSON encoding and the HTTP write of ~1.9 MB per request.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
# Phase 2's tasks read raw text shards and merge-sort them -- no rendered pages, no request
# threads. Same one-CPU shape so they pack the same worker pods the senders just vacated.
_NORMALIZE_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="40g", disk="32g")
# A task holding a long document can legitimately go a long time without finishing one.
_HEARTBEAT_TIMEOUT = 30 * 60


def _check_alive_bounded(session, timeout_seconds: float = 120.0) -> None:
    """``session.check_alive()``, bounded.

    ``check_alive`` makes one controller RPC per fleet job, and the broker ceiling bench's driver
    hung for over an hour after a completed run somewhere between this call and teardown -- with
    the whole idle fleet still billed. The check is advisory (it distinguishes lost capacity from
    lost pages); an answer that cannot arrive within the bound is not worth holding GPUs for.
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

    Returns ``(sender_tasks, max_workers)``. Sized by offered rate -- tasks enough that their
    measured :data:`_PAGES_PER_SECOND_PER_TASK` (0.75) each meets the engines'
    :data:`_PAGES_PER_SECOND_PER_INSTANCE` (17.75) -- with the in-flight budget as a floor for the
    regime where request latency, not task throughput, is what limits delivery.
    """
    rate_basis = math.ceil(instances * _PAGES_PER_SECOND_PER_INSTANCE / _PAGES_PER_SECOND_PER_TASK)
    inflight_basis = fleet.MAX_IN_FLIGHT * instances // _REQUEST_THREADS
    sender_tasks = max(1, rate_basis, inflight_basis)
    return sender_tasks, max(1, math.ceil(sender_tasks / _TASKS_PER_WORKER))


@cache
def _request_pool(threads: int) -> ThreadPoolExecutor:
    """One request pool per sender process, shared by every shard the process runs.

    Never shut down: it lives as long as the process, and rebuilding it per shard would drop the
    endpoint's keep-alive connections along with it.
    """
    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ocr-page")


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

        The exception is deliberately not propagated. The request has already exhausted its retries,
        and one unreadable page is data about the document rather than a reason to lose it or to
        fail the shard; it is recorded on the row and counted.

        A page the model looped on is repaired here rather than downstream, because this is the only
        place the per-page truncation flag exists -- the counts collapse into totals immediately --
        and because the boilerplate pass must see each page's real last line, not a loop's tail.
        """
        try:
            page = future.result()
        except Exception as error:
            self.failed += 1
            self.pages.append("")
            self.first_error = self.first_error or f"{type(error).__name__}: {error}"
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_request_failed", 1)
            # Also by type. A run that loses pages has to say *how* in its counters: a timeout, a
            # 5xx and a missing dependency are three different problems, and the counters are the
            # only diagnosis that survives a run whose logs cannot be retrieved.
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_request_failed/{type(error).__name__}", 1)
            return
        repair = repair_page(page.text, page.truncated, loop)
        self.pages.append(repair.text)
        self.completion_tokens += page.completion_tokens
        if page.truncated:
            # The page is real but incomplete. Counted rather than dropped: partial text from a
            # dense page is still worth more than nothing, but a consumer has to be able to tell.
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
        # Give every page a trailing newline so the last line of one cannot fuse with the first line
        # of the next. This has to happen *after* stripping, not before: a blank line at the foot of
        # every page is itself a repeated edge pattern, so adding the newlines first would have the
        # boilerplate pass take them straight back off again.
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


def ocr_batch(
    batch: pa.RecordBatch,
    keys: frozenset[tuple[str, int]] | None,
    endpoint: OcrEndpoint,
    render_options: RenderOptions,
    boilerplate: BoilerplateOptions,
    loop: LoopOptions,
) -> Iterator[dict]:
    """OCR the OCR-routed documents in one Parquet row group.

    Rendering and waiting are overlapped rather than alternated. A page is submitted the moment it
    is rendered, and the next document starts rendering while the previous one is still in flight,
    so the endpoint stays fed by a task that is mostly idle. The only thing that blocks is the
    in-flight bound: at :data:`_PAGES_IN_FLIGHT` outstanding pages the task waits for the oldest one
    before submitting another, which is what keeps encoded pages -- over a megabyte each -- from
    accumulating without limit on a long document.

    Documents are emitted in the order they were read, which they reach naturally: pages resolve in
    submission order, so a document becomes complete only once every document before it has.

    ``keys`` is the OCR route's key set, or ``None`` to OCR every document in the shard -- the
    all-routes comparison run, where the point is reading the same documents both extractors read.
    """
    pool = _request_pool(_REQUEST_THREADS)
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
        if keys is not None and (row["warc_filename"], row["warc_record_offset"]) not in keys:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/skipped_text_route", 1)
            continue

        try:
            with open_pdf(row["pdf"]) as pdf:
                document = _Document(row=row, declared_pages=len(pdf))
                documents.append(document)
                try:
                    for page in iter_rendered_pages(pdf, render_options):
                        inflight.append((document, pool.submit(ocr_page, endpoint, _REQUEST_THREADS, page)))
                        document.submitted += 1
                        document.dpis.append(page.dpi)
                        while len(inflight) >= _PAGES_IN_FLIGHT:
                            resolve_oldest()
                            yield from emit_ready()
                finally:
                    # Whatever went wrong mid-render, the document has to be closed or it would sit
                    # in the queue forever and take every document behind it with it.
                    document.closed = True
        except Exception as error:
            # MuPDF fails arbitrarily deep on adversarial input. A PDF we cannot open is data, not
            # a pipeline failure; pages already rendered from it are kept.
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/render_failed", 1)
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/render_failed/{type(error).__name__}", 1)
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
) -> NormalizedData:
    """Run the OCR route in two phases on one warm sender pool, holding GPUs only for the first.

    Phase 1 OCRs pages and writes **raw per-source-shard parquet** -- no shuffle -- inside the
    ``remote_inference`` context, so the fleet is released the moment the last page lands. Phase 2
    reads the raw shards back and runs the one legitimate shuffle (the normalized format's global
    sort by content-hash ``id``) on the same Zephyr worker pool, which #7145 keeps warm across
    ``execute()`` calls. Holding GPUs through the shuffle would buy nothing: the sort key is a
    hash computed after OCR, so the repartition is CPU-and-storage work by construction.

    The raw directory is also the checkpoint. Phase 1 writes with ``skip_existing``, so a retry
    of this step re-OCRs only the shards whose raw file never landed, and a phase-2 failure --
    shuffle-storage trouble burned a fleet once already -- costs no GPU time at all: with every
    raw shard present the fleet is never started.

    ``partition`` is ``(index, count)`` over the sorted source shards, which is how a run larger
    than one broker can carry scales out: several of these steps run side by side, each with its
    own fleet, broker, proxy, and sender fleet, over disjoint slices of the corpus. Sharding
    whole steps rather than endpoints keeps every per-fleet process at the size the ceiling
    benchmark validated -- one shared driver would park one thread per in-flight request across
    *all* fleets in a single process.
    """
    partition_index, partition_count = partition
    source = read_artifact(source_output_path, PdfSourceData)
    keys = None
    if ocr_route_only:
        if classification_output_path is None:
            raise ValueError("ocr_route_only requires classification_output_path")
        classification = read_artifact(classification_output_path, PdfClassificationData)
        keys = routing_keys(classification.main_output_dir, needs_ocr=True)

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

    raw_dir = prefix_join(output_path, "raw")
    tallies: dict[str, int | float] = {}

    with ZephyrContext(
        name=f"focus-crawl-pdf-ocr-{partition_index}",
        resources=_WORKER_RESOURCES,
        max_workers=max_workers,
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
        # Not the 1 GB default: a shared-pool coordinator holds both executions' shard, retry, and
        # result state, and at a full partition's scale (161 shards, ~380 tasks) the default was
        # OOM-killed (exit 137) at the end of the reduce -- after every output shard was written,
        # so the step failed with its work complete on disk.
        coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
    ) as pool:
        raw_shards_present = len(StoragePath(prefix_join(raw_dir, "*.parquet")).glob())
        if raw_shards_present < num_shards:
            with remote_inference(build_inference_config(instances=instances)) as session:
                endpoint = OcrEndpoint(
                    base_url=session.model.endpoint.base_url,
                    model=session.model.endpoint.model,
                    max_visual_tokens=RENDER_OPTIONS.max_visual_tokens,
                )
                logger.info("OCR endpoint ready at %s (%s)", endpoint.base_url, session.backend_name)

                ocr_pipeline = (
                    Dataset.from_list(shards)
                    .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
                    .flat_map(
                        partial(
                            ocr_batch,
                            keys=keys,
                            endpoint=endpoint,
                            render_options=RENDER_OPTIONS,
                            boilerplate=BOILERPLATE_OPTIONS,
                            loop=LOOP_OPTIONS,
                        )
                    )
                    .write_parquet(
                        prefix_join(raw_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
                        schema=OUTPUT_SCHEMA,
                        skip_existing=True,
                    )
                )
                outcome = pool.execute(ocr_pipeline, map_task_resources=_MAP_TASK_RESOURCES)
                tallies.update(outcome.counters)
                # A fleet that died partway through would show up as failed pages rather than as an
                # error, so the corpus would be quietly short. Surface that -- but only when it
                # actually cost something. An instance that exits after the last shard has landed is
                # not a reason to throw away a complete run, and treating it as one loses the whole
                # phase's output to a race on the way out.
                lost_pages = int(outcome.counters.get(f"{_COUNTER_PREFIX}/page_request_failed", 0))
                try:
                    _check_alive_bounded(session)
                except Exception as error:
                    if lost_pages:
                        raise
                    logger.warning("An inference job ended before the fleet was released: %s", error)
                    logger.warning("No page request failed, so the extracted corpus is complete.")
        else:
            logger.info("All %d raw shards already written under %s; skipping the OCR phase", num_shards, raw_dir)

        # The GPUs are gone; the same warm workers now run the shuffle.
        normalize_pipeline = (
            Dataset.from_files(prefix_join(raw_dir, "*.parquet"))
            .load_parquet()
            .group_by(
                key=lambda record: record["id"],
                reducer=keep_all,
                sort_by=lambda record: record["id"],
                num_output_shards=num_shards,
            )
            .map_shard(make_split_writer(output_path, output_schema=OUTPUT_SCHEMA))
        )
        outcome = pool.execute(normalize_pipeline, map_task_resources=_NORMALIZE_TASK_RESOURCES)
        tallies.update(outcome.counters)

    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=tallies,
    )


def ocr_extract_step(source: StepSpec, classification: StepSpec) -> StepSpec:
    """Build the OCR extraction step for the router's OCR route."""
    return StepSpec(
        name="data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr",
        deps=[source, classification],
        hash_attrs={
            # The model and the prompt are as much a part of this step's identity as the render
            # settings are: change either and the text changes.
            "model": MODEL,
            "prompt_digest": sha256(PROMPT_DOC2MD.encode("utf-8")).hexdigest()[:16],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_visual_tokens": RENDER_OPTIONS.max_visual_tokens,
            "max_render_dpi": RENDER_OPTIONS.max_render_dpi,
            "max_pages": RENDER_OPTIONS.max_pages,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            # Loop repair rewrites the stored text, so its thresholds re-key the step exactly as the
            # prompt does. The calibration behind them assumes ``max_tokens`` above: raising the cap
            # means a runaway no longer marks the page truncated, and the gate has to be re-derived.
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
            # The sender tasks render pages with pymupdf at runtime (lazy-imported inside
            # ocr_extract.render); it lives in the ``pdf`` extra.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )
