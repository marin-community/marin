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
completions and takes images, so it cannot read Parquet or open a PDF. That cost is priced -- 13.2
pages per core-second at this budget, so about 1.1 render cores per GPU -- and the sender fleet is
sized well above it.

What the senders *must* get right is keeping the fleet full. Throughput here is set by in-flight
count, not by how fast any one sender runs, so the fleet is sized to hold
:data:`~experiments.build_pdf_source.ocr_extract.fleet.CLIENT_CONCURRENCY` requests in flight across
all tasks, and each task overlaps rendering with waiting rather than alternating between them.

The output is the same :class:`~marin.datakit.normalize.NormalizedData` shape the docling route
produces, over the same shared columns, so a consumer joins the two routes without knowing which
extractor produced a document. Running headers and footers are stripped by the same
:mod:`~experiments.build_pdf_source.boilerplate` pass, before the text is hashed into ``id``.
"""

import logging
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
    MainOutput,
    NormalizedData,
    _make_split_writer,
    generate_id,
)
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.inference.iris import remote_inference
from rigging.filesystem import prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source.boilerplate import BoilerplateOptions, strip_boilerplate
from experiments.build_pdf_source.classify import routing_keys
from experiments.build_pdf_source.common import FOCUS_CRAWL, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS, source_id
from experiments.build_pdf_source.ocr_extract.client import (
    DEFAULT_MAX_TOKENS,
    PROMPT_DOC2MD,
    OcrEndpoint,
    PageOcr,
    ocr_page,
)
from experiments.build_pdf_source.ocr_extract.fleet import CLIENT_CONCURRENCY, MODEL, build_inference_config
from experiments.build_pdf_source.ocr_extract.render import RenderOptions, iter_rendered_pages, open_pdf

logger = logging.getLogger(__name__)

RENDER_OPTIONS = RenderOptions()
BOILERPLATE_OPTIONS = BoilerplateOptions()

_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]

_COUNTER_PREFIX = "focus_crawl_pdf_ocr"


class OcrStatus(StrEnum):
    """Whether a document was OCR'd whole.

    ``PARTIAL`` covers every way a page can come back short: MuPDF declining to render it, the page
    budget truncating a very long document, the request failing after its retries, or the model
    hitting its token cap part-way down the page. The per-page counts in the record say which.

    The last of those is the one worth watching, because it is the only one that yields text that
    looks complete. A page dropped for any other reason is empty and obvious; a page cut off at
    ``max_tokens`` is ordinary Markdown that simply stops, and nothing but ``pages_truncated`` says
    so.
    """

    SUCCESS = "success"
    PARTIAL = "partial"


_OCR_FIELDS: tuple[pa.Field, ...] = (
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
)

_OUTPUT_SCHEMA = pa.schema([*PDF_DOCUMENT_FIELDS, *_OCR_FIELDS])

# Sender sizing, derived from the fleet's operating point rather than guessed. Throughput is set by
# in-flight requests, so the product of these two is the number that matters and it is pinned to
# CLIENT_CONCURRENCY; splitting it 64 x 32 rather than, say, 256 x 8 keeps any one task's failure
# cheap and spreads the render work over more cores.
_REQUEST_THREADS = 64
_SENDER_TASKS = CLIENT_CONCURRENCY // _REQUEST_THREADS
# How many rendered pages a task may hold. Twice the thread count keeps every thread fed while
# bounding encoded-page memory to roughly 180 MB per task.
_PAGES_IN_FLIGHT = 2 * _REQUEST_THREADS

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# A task is almost entirely blocked on the endpoint, so it costs one CPU and multiplexes eight-deep
# per worker. The fleet's total CPU is several times the measured render requirement, which is the
# headroom for JSON encoding and the HTTP write of ~1.9 MB per request.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="3g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="32g", disk="32g")
_MAX_WORKERS = _SENDER_TASKS // _TASKS_PER_WORKER
# A task holding a long document can legitimately go a long time without finishing one.
_HEARTBEAT_TIMEOUT = 30 * 60


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
    first_error: str | None = None

    @property
    def complete(self) -> bool:
        """Every page submitted and every submitted page resolved."""
        return self.closed and len(self.pages) == self.submitted

    def absorb(self, future: "Future[PageOcr]") -> None:
        """Record one page's result, keeping a failed page as an empty page.

        The exception is deliberately not propagated. The request has already exhausted its retries,
        and one unreadable page is data about the document rather than a reason to lose it or to
        fail the shard; it is recorded on the row and counted.
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
        self.pages.append(page.text)
        self.completion_tokens += page.completion_tokens
        if page.truncated:
            # The page is real but incomplete. Counted rather than dropped: partial text from a
            # dense page is still worth more than nothing, but a consumer has to be able to tell.
            self.truncated += 1
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/page_truncated", 1)

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
        status = OcrStatus.SUCCESS if self.failed == 0 and unrendered == 0 and self.truncated == 0 else OcrStatus.PARTIAL
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
        }

    def _error_summary(self, unrendered: int) -> str | None:
        parts = []
        if self.truncated:
            parts.append(f"{self.truncated} of {self.submitted} pages hit the token cap and were cut off")
        if unrendered:
            parts.append(f"{unrendered} of {self.declared_pages} pages were not rendered")
        if self.failed:
            parts.append(f"{self.failed} of {self.submitted} page requests failed: {self.first_error}")
        return "; ".join(parts) or None


def ocr_batch(
    batch: pa.RecordBatch,
    keys: frozenset[tuple[str, int]],
    endpoint: OcrEndpoint,
    render_options: RenderOptions,
    boilerplate: BoilerplateOptions,
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
    """
    pool = _request_pool(_REQUEST_THREADS)
    inflight: deque[tuple[_Document, Future[PageOcr]]] = deque()
    documents: deque[_Document] = deque()

    def resolve_oldest() -> None:
        document, future = inflight.popleft()
        document.absorb(future)

    def emit_ready() -> Iterator[dict]:
        while documents and documents[0].complete:
            record = documents.popleft().record(boilerplate, render_options.legibility_floor_dpi)
            if record is not None:
                yield record

    for row in batch.to_pylist():
        if (row["warc_filename"], row["warc_record_offset"]) not in keys:
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


def _keep_all(_key: str, records: Iterator[dict]) -> Iterator[MainOutput]:
    """Emit every record to the main output.

    As on the text route, deduplication is #7620's decision to make across every source and against
    the eval sets, not this step's to make within one. The grouping still earns its cost: it sorts
    by ``id``, which is part of the normalized format and is what makes a later dedup pass a linear
    scan.
    """
    yield from (MainOutput(data=record) for record in records)


def ocr_pdf_text(output_path: str, source_output_path: str, classification_output_path: str) -> NormalizedData:
    """Run the OCR route against a fleet started for the duration of this step.

    The fleet is the expensive resource and it is held for exactly as long as there is work: the
    Zephyr run happens inside the ``remote_inference`` context, so the GPUs come up, get saturated
    by the sender fleet, and are released when the last shard lands.
    """
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=True)

    filesystem, path = url_to_fs(source.main_output_dir)
    num_shards = len(filesystem.glob(f"{path}/*.parquet"))
    if not num_shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")

    with remote_inference(build_inference_config()) as session:
        endpoint = OcrEndpoint(
            base_url=session.model.endpoint.base_url,
            model=session.model.endpoint.model,
            max_visual_tokens=RENDER_OPTIONS.max_visual_tokens,
        )
        logger.info("OCR endpoint ready at %s (%s)", endpoint.base_url, session.backend_name)

        pipeline = (
            Dataset.from_files(prefix_join(source.main_output_dir, "*.parquet"))
            .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
            .flat_map(
                partial(
                    ocr_batch,
                    keys=keys,
                    endpoint=endpoint,
                    render_options=RENDER_OPTIONS,
                    boilerplate=BOILERPLATE_OPTIONS,
                )
            )
            .group_by(
                key=lambda record: record["id"],
                reducer=_keep_all,
                sort_by=lambda record: record["id"],
                num_output_shards=num_shards,
            )
            .map_shard(_make_split_writer(output_path, output_schema=_OUTPUT_SCHEMA))
        )
        outcome = ZephyrContext(
            name="focus-crawl-pdf-ocr",
            resources=_WORKER_RESOURCES,
            max_workers=_MAX_WORKERS,
            stage_runner_factory=SubprocessRunner,
            map_task_resources=_MAP_TASK_RESOURCES,
            heartbeat_timeout=_HEARTBEAT_TIMEOUT,
        ).execute(pipeline)
        # A fleet that died partway through would show up as failed pages rather than as an error,
        # so the corpus would be quietly short. Surface that -- but only when it actually cost
        # something. An instance that exits after the last shard has landed is not a reason to throw
        # away a complete run, and treating it as one loses the whole step's output to a race on the
        # way out.
        lost_pages = int(outcome.counters.get(f"{_COUNTER_PREFIX}/page_request_failed", 0))
        try:
            session.check_alive()
        except Exception as error:
            if lost_pages:
                raise
            logger.warning("An inference job ended before the fleet was released: %s", error)
            logger.warning("No page request failed, so the extracted corpus is complete.")

    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
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
            "schema_version": 1,
        },
        fn=remote(
            partial(
                ocr_pdf_text,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
