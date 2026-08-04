# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5, brokered: extract the text route through a persistent CPU converter fleet.

This is the same extraction as :mod:`experiments.build_pdf_source.extract` -- same options, same
boilerplate pass, same record shape -- with the converter moved out of the map task and behind the
broker. Senders read fetch shards sequentially and post one whole document per request; the
broker's lease queue balances documents across the persistent converters, so a shard full of
3,000-page documents drains across the fleet instead of stalling the task that happened to read it.
The converter's build cost -- compiling the layout graph -- is paid once per converter process
instead of once per Zephyr shard.

The unit of work stays the whole document, not the page: cross-page paragraph merging and
boilerplate removal are document-scoped, so splitting a document across converters would change
the text that comes out.

**The fleet is elastic.** Each pool instance is an independent Iris job and the broker's queue is
pull-based, so the run starts converting as soon as the first converter is up, ramps as capacity
is allocated, and shrinks through preemptions without losing the run -- request more pods than the
cluster can currently grant and take whatever schedules. Sender sizing and the proxy queue are
derived from the requested converter count, so changing the pool constants rescales the senders.

Launch the production run via :mod:`~experiments.build_pdf_source.extract_fleet_run` -- this
module must stay importable rather than run as ``__main__``, or the step's functions pickle as
unresolvable ``__main__`` references on the driver.
"""

import json
import logging
import math
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import cache, partial

import httpx
import pyarrow as pa
from fray.types import ResourceConfig, create_environment
from marin.datakit.normalize import NormalizedData, generate_id, make_split_writer
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.inference.config import BrokerConfig, InferenceProxyConfig, InferenceWorkerConfig
from marin.inference.converter_pool import ConverterPoolConfig, ConverterPoolSession, remote_converter_pool
from rigging.filesystem import prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source.boilerplate import BoilerplateOptions, strip_document_boilerplate
from experiments.build_pdf_source.classify import routing_keys
from experiments.build_pdf_source.common import (
    FOCUS_CRAWL,
    LayoutModelData,
    PdfClassificationData,
    PdfSourceData,
)
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.docling_extract.service import (
    SOURCE_URL_HEADER,
    ConvertedDocument,
    build_arch_adaptive_handler,
    parse_converted,
)
from experiments.build_pdf_source.extract import (
    _OUTPUT_SCHEMA,
    _SOURCE_COLUMNS,
    BOILERPLATE_OPTIONS,
    PICTURE_ALPHA_RATIO,
    _keep_all,
)

logger = logging.getLogger(__name__)

MODEL_ID = "marin-docling-convert"

# The fleet's backends. TableFormer everywhere: it recovers the table content the ruling-line
# reader empties (matrix over 600 documents: 9.7M vs 6.9M table chars, 143 vs 1,286 empty tables).
# The layout backend is decided per converter process at startup and recorded per record in the
# ``layout_backend`` column, because placement decides where a converter runs and placement is not
# reproducible. Both arms are currently FP32 torch: INT8 was retired after a 100-document
# head-to-head against FP32 + TableFormer showed the quantized layout model fragments regions
# (~2x the block count on 579/600 documents), splicing multi-column reading order and severing
# table row/column bindings -- bigram token F1 0.851 with a quarter of documents under 0.80,
# invisible to unigram F1. The arch split and the provenance column stay: they are what makes a
# faster x86 backend safe to reintroduce if one earns it on order-sensitive metrics.
TABLE_BACKEND = TableBackend.DOCLING
X86_LAYOUT_BACKEND = LayoutBackend.TORCH_HERON
ARM_LAYOUT_BACKEND = LayoutBackend.TORCH_HERON

# The route-specific column, appended to the shared document fields the same way the OCR route
# appends its page accounting.
_FLEET_FIELDS: tuple[pa.Field, ...] = (pa.field("layout_backend", pa.string(), nullable=False),)
_FLEET_OUTPUT_SCHEMA = pa.schema([*_OUTPUT_SCHEMA, *_FLEET_FIELDS])

# The pool's operating point: 4 converters per pod won the shape comparison (compute tied with
# 256x1, scheduling 2x faster), so scale comes from more pods. 256 pods is a *request*, not a
# requirement: instances are independent jobs, so the run proceeds on whatever fraction schedules
# and the rest join as capacity frees up.
_POOL_INSTANCES = 256
_PROCESSES_PER_INSTANCE = 4
_CONVERTERS = _POOL_INSTANCES * _PROCESSES_PER_INSTANCE
# RAM follows extract.py's 7g-per-1cpu-task precedent: ~7g per converter process, plus headroom
# for the supervisor and the page tail.
_POOL_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="32g")

# The conversion budget is sized to keep long documents whole, not to keep the fleet snappy: a
# 300-page book at fp32 + TableFormer's ~4.3s/page needs ~22 minutes, and the documents the old
# 600s budget truncated held 1-2M characters each -- exactly the long-form content worth having.
# A whale occupies one converter core while 255 others keep draining, so the fleet cost of the
# tail is small (~3% of documents ran past 600s). The timeout exists to bound a genuinely wedged
# conversion, and it is a wall clock: a document that exceeds it truncates at a hardware-dependent
# page and is marked partial_success.
_DOCUMENT_TIMEOUT = 45 * 60.0
# The delivery chain satisfies BrokerConfig's 0 < worker < lease < proxy invariant, sitting above
# the document budget so a slow-but-finishing conversion is never killed in flight. The proxy
# budget is a full lease expiry plus a full document budget: a document leased by a pod that then
# got preempted is invisible until the lease expires at 3300, and the redelivered attempt deserves
# the whole 45 minutes, not whatever the first attempt left over -- otherwise every preempted whale
# times out at the proxy and is counted lost.
_WORKER_REQUEST_TIMEOUT = 3000.0
_LEASE_TIMEOUT = 3300.0
_PROXY_REQUEST_TIMEOUT = _LEASE_TIMEOUT + _DOCUMENT_TIMEOUT
# The first converter to answer /v1/models downloads the HF layout model before it can build.
_PROXY_READINESS_TIMEOUT = 1800.0

# Sender sizing, from the measured economics of a task rather than the in-flight ceiling -- the
# same lesson the OCR campaign's `sender_fleet_size` encodes. A task's delivered rate is not set
# by its 16 request threads but by whale parking: at the end of its shard window it waits on its
# slowest in-flight documents, and the 45-minute budget makes that wait most of its life. Measured
# on the live run: ~0.055 docs/s per task, an order below the thread-limited burst rate. The fleet
# absorbs ~_CONVERTERS / 72s (matrix mean, confirmed live), so tasks are provisioned to meet that
# at the measured rate with 30% headroom; the two-per-converter in-flight budget stays as a floor.
# Idle-parked tasks are cheap (one blocked CPU); an idle converter fleet is not.
_REQUEST_THREADS = 16
_DOCUMENTS_IN_FLIGHT = 2 * _REQUEST_THREADS
_MEAN_DOCUMENT_SECONDS = 72.0
_DOCS_PER_SECOND_PER_TASK = 0.055
_RATE_HEADROOM = 1.3
_SENDER_TASKS = max(
    math.ceil(2 * _CONVERTERS / _REQUEST_THREADS),
    math.ceil(_RATE_HEADROOM * _CONVERTERS / _MEAN_DOCUMENT_SECONDS / _DOCS_PER_SECOND_PER_TASK),
)
_TASKS_PER_WORKER = 8
_MAX_WORKERS = math.ceil(_SENDER_TASKS / _TASKS_PER_WORKER)
_MAX_PENDING_REQUESTS = 2 * _SENDER_TASKS * _REQUEST_THREADS

# What the senders offer beyond the converters queues as whole PDF payloads in the broker actor's
# memory: ~(offer - converters) documents at a low-MB mean.
_BROKER_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="32g", disk="20g", preemptible=False)

# A task is almost entirely blocked on the proxy, but it holds up to 2 x _REQUEST_THREADS whole
# PDFs in flight and tail PDFs run to tens of MB.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="6g", disk="4g")
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="48g", disk="32g")
# A sender task blocked on its oldest in-flight future -- a whale, possibly redelivered after a
# preemption -- can legitimately go quiet for the full proxy budget, so the heartbeat must outlast
# it however the chain above moves.
_HEARTBEAT_TIMEOUT = _PROXY_REQUEST_TIMEOUT + 300.0

# The driver runs the proxy, which parks one OS thread per in-flight request, each holding a whole
# PDF body until the response lands: ~2 x _CONVERTERS documents at a low-MB mean with tails in the
# tens of MB, plus the Zephyr coordinator and the routing key set. Sized to the OCR campaign's
# driver, which held a comparable in-flight payload.
_FLEET_DRIVER_RESOURCES = ResourceConfig(cpu=12, ram="96g", disk="32g")

_COUNTER_PREFIX = "focus_crawl_pdf_fleet"

# One FLEET-STATS JSON line per interval in the driver log: broker depth, converter registration,
# throughput, and pool-job states. The run watcher (_watch_fleet.py) tails these lines; zephyr
# separately pushes shard progress into the driver task's Iris status.
_STATS_INTERVAL = 60.0

# Characters that must survive percent-encoding for the header to remain a readable URL; everything
# outside latin-1 is encoded, which is what header transport requires.
_HEADER_SAFE = ":/?#[]@!$&'()*+,;=%"


def build_pool_config(handler_factory: Callable) -> ConverterPoolConfig:
    """The converter fleet: what each converter runs, and the broker between it and the senders."""
    return ConverterPoolConfig(
        handler_factory=handler_factory,
        model_id=MODEL_ID,
        instances=_POOL_INSTANCES,
        processes_per_instance=_PROCESSES_PER_INSTANCE,
        worker_resources=_POOL_WORKER_RESOURCES,
        # The same environment `remote(..., pip_dependency_groups=["datakit"])` builds for the
        # driver: uv extras carry the docling stack. `create_environment` maps `extras` onto the
        # workspace install, so the pool jobs resolve marin[datakit] exactly as the driver does.
        worker_environment=create_environment(extras=["datakit"]),
        broker=BrokerConfig(
            worker=InferenceWorkerConfig(
                # A converter is a single core running one document at a time.
                max_in_flight=1,
                request_timeout_seconds=_WORKER_REQUEST_TIMEOUT,
            ),
            broker_resources=_BROKER_RESOURCES,
            request_lease_timeout_seconds=_LEASE_TIMEOUT,
            proxy=InferenceProxyConfig(
                request_timeout_seconds=_PROXY_REQUEST_TIMEOUT,
                readiness_timeout_seconds=_PROXY_READINESS_TIMEOUT,
                max_pending_requests=_MAX_PENDING_REQUESTS,
            ),
        ),
    )


@cache
def _client(threads: int) -> httpx.Client:
    """One HTTP client per sender process, shared by every shard the process runs.

    The connection pool is sized to the thread count because httpx defaults to 100 connections,
    which would silently cap in-flight requests -- the lesson from the OCR campaign. The client
    timeout sits just above the proxy's, because the proxy is the real timeout authority: a request
    should come back as the proxy's 504 envelope, not as a client-side timeout racing it.
    """
    limits = httpx.Limits(max_connections=threads, max_keepalive_connections=threads)
    return httpx.Client(limits=limits, timeout=_PROXY_REQUEST_TIMEOUT + 60.0)


@cache
def _request_pool(threads: int) -> ThreadPoolExecutor:
    """One request pool per sender process. Never shut down: it lives as long as the process."""
    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix="convert-document")


def convert_document(base_url: str, threads: int, row: dict) -> ConvertedDocument:
    """Post one PDF to the pool and parse the conversion that comes back.

    Non-200 responses are broker or proxy envelopes -- 502 is a handler bug, 504 a timeout -- and
    are raised without retry: the document is treated as lost, which is what keeps a poison
    document from cycling through the fleet. Only a transport fault, which says nothing about the
    document, earns one retry.
    """
    url = f"{base_url.rstrip('/')}/convert"
    headers = {
        "content-type": "application/pdf",
        SOURCE_URL_HEADER: urllib.parse.quote(row["url"] or "document.pdf", safe=_HEADER_SAFE),
    }
    client = _client(threads)
    try:
        response = client.post(url, content=row["pdf"], headers=headers)
    except httpx.TransportError:
        response = client.post(url, content=row["pdf"], headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"convert returned {response.status_code}: {response.text[:200]}")
    return parse_converted(response.content)


def convert_shard(
    batches: Iterator[pa.RecordBatch],
    _shard_info: ShardInfo,
    *,
    keys: frozenset[tuple[str, int]],
    skipped_counter: str,
    base_url: str,
    boilerplate: BoilerplateOptions,
) -> Iterator[dict]:
    """Convert one fetch shard's routed documents through the pool.

    One submission window spans the whole shard, bounded at :data:`_DOCUMENTS_IN_FLIGHT`
    outstanding so a task never holds more than twice its thread count of whole PDFs in memory,
    and requests resolve in *completion* order. Both halves of that are load-bearing: the fetch
    shards carry ~47-row row groups, so a per-batch window drained to zero at every row-group
    boundary, and each drain parked the task behind that mini-batch's slowest document -- with the
    45-minute budget, most sender tasks sat at 0 items/s and the fleet at ~40% utilization.
    Paying the whale tail once per shard instead of once per row group roughly doubles a task's
    delivered rate; the rest of the gap is provisioned away by the rate-based sender sizing above.
    Downstream ``group_by`` sorts by id, so record order does not matter.
    """
    pool = _request_pool(_REQUEST_THREADS)
    inflight: dict[Future[ConvertedDocument], dict] = {}

    def resolve(future: Future[ConvertedDocument]) -> Iterator[dict]:
        row = inflight.pop(future)
        try:
            document = future.result()
        except Exception as error:
            # The request exhausted its one transport retry or came back as a broker envelope. The
            # per-type counter is the diagnosis that survives a run whose logs cannot be retrieved.
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/convert_request_failed", 1)
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/convert_request_failed/{type(error).__name__}", 1)
            logger.warning("Convert request failed for %s: %s", row["url"], error)
            return
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/fleet_convert_seconds", document.seconds)
        if document.status == "failure":
            # The converter answered, and the answer is that this document cannot be converted.
            # Same treatment as the in-task route: counted and dropped, not a pipeline failure.
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extraction_failed", 1)
            logger.warning("Could not extract %s: %s", row["url"], document.error)
            return
        record = _record(row, document, boilerplate)
        if record is not None:
            yield record

    def resolve_completed() -> Iterator[dict]:
        done, _ = wait(inflight, return_when=FIRST_COMPLETED)
        for future in done:
            yield from resolve(future)

    for batch in batches:
        for row in batch.to_pylist():
            if (row["warc_filename"], row["warc_record_offset"]) not in keys:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{skipped_counter}", 1)
                continue
            while len(inflight) >= _DOCUMENTS_IN_FLIGHT:
                yield from resolve_completed()
            inflight[pool.submit(convert_document, base_url, _REQUEST_THREADS, row)] = row

    while inflight:
        yield from resolve_completed()


def _record(row: dict, document: ConvertedDocument, boilerplate: BoilerplateOptions) -> dict | None:
    """Assemble one output record, byte-identical to what the in-task route produces."""
    stripped = strip_document_boilerplate(document.text, document.page_offsets, boilerplate)
    text = stripped.text
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted", 1)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_pages", document.num_pages)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/boilerplate_lines_removed", stripped.lines_removed)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/boilerplate_pages_stripped", stripped.pages_stripped)
    if document.error:
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/partial_extraction", 1)

    if not text.strip():
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/empty_text_filtered", 1)
        return None
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_characters", len(text))
    # Per-backend counts are the run-level view of how placement split the fleet.
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/extracted_by/{document.backend}", 1)

    return {
        "id": generate_id(text),
        "text": text,
        "source_id": f"{row['warc_filename']}:{row['warc_record_offset']}",
        "source": FOCUS_CRAWL,
        "warc_filename": row["warc_filename"],
        "warc_record_offset": row["warc_record_offset"],
        "content_digest": row["content_digest"],
        "url": row["url"],
        "num_pages": document.num_pages,
        "page_offsets": stripped.page_offsets,
        "extraction_status": document.status,
        "extraction_error": document.error,
        "boilerplate_lines_removed": stripped.lines_removed,
        "layout_backend": document.backend,
    }


def _log_fleet_stats(session: ConverterPoolSession, stop: threading.Event) -> None:
    """Emit a FLEET-STATS line every minute until the run releases the fleet.

    Job-state polling is one controller RPC per pool job, so it rides the same interval as the
    broker snapshot. This is diagnostics: a failed poll is logged and the run is left alone.
    """
    while not stop.wait(_STATS_INTERVAL):
        try:
            stats = session.broker.stats()
            job_states = Counter(str(job.status()) for job in session.jobs)
            logger.info(
                "FLEET-STATS %s",
                json.dumps(
                    {
                        "time": int(time.time()),
                        "pods_registered": stats.workers,
                        "converters": stats.workers * _PROCESSES_PER_INSTANCE,
                        "converting": stats.leased,
                        "queued": stats.queued,
                        "responses_ready": stats.responses_ready,
                        "completed_total": stats.completed_total,
                        "pool_jobs": dict(job_states),
                    }
                ),
            )
        except Exception:
            logger.warning("Fleet stats poll failed", exc_info=True)


def fleet_extract(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
    *,
    needs_ocr: bool,
) -> NormalizedData:
    """Run one routing side of the corpus against a converter fleet held for the duration.

    ``needs_ocr=False`` is the production text route. ``needs_ocr=True`` converts the router's
    OCR-route complement instead, so the union of the two runs is a docling conversion of every
    classified document.
    """
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    x86_options = ExtractionOptions(
        table_backend=TABLE_BACKEND,
        layout_backend=X86_LAYOUT_BACKEND,
        layout_model_path=layout_model.model_path,
        layout_label_map=layout_model.label_map,
        picture_alpha_ratio=PICTURE_ALPHA_RATIO,
        document_timeout=_DOCUMENT_TIMEOUT,
    )
    arm_options = ExtractionOptions(
        table_backend=TABLE_BACKEND,
        layout_backend=ARM_LAYOUT_BACKEND,
        picture_alpha_ratio=PICTURE_ALPHA_RATIO,
        document_timeout=_DOCUMENT_TIMEOUT,
    )
    handler_factory = partial(build_arch_adaptive_handler, x86_options, arm_options)
    keys = routing_keys(classification.main_output_dir, needs_ocr=needs_ocr)
    skipped_counter = "skipped_text_route" if needs_ocr else "skipped_ocr_route"

    filesystem, path = url_to_fs(source.main_output_dir)
    num_shards = len(filesystem.glob(f"{path}/*.parquet"))
    if not num_shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")

    with remote_converter_pool(build_pool_config(handler_factory)) as session:
        logger.info("Converter pool ready at %s (%d converters requested)", session.endpoint.base_url, _CONVERTERS)
        stop_stats = threading.Event()
        threading.Thread(target=_log_fleet_stats, args=(session, stop_stats), name="fleet-stats", daemon=True).start()
        pipeline = (
            Dataset.from_files(prefix_join(source.main_output_dir, "*.parquet"))
            .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
            .map_shard(
                partial(
                    convert_shard,
                    keys=keys,
                    skipped_counter=skipped_counter,
                    base_url=session.endpoint.base_url,
                    boilerplate=BOILERPLATE_OPTIONS,
                )
            )
            .group_by(
                key=lambda record: record["id"],
                reducer=_keep_all,
                sort_by=lambda record: record["id"],
                num_output_shards=num_shards,
            )
            .map_shard(make_split_writer(output_path, output_schema=_FLEET_OUTPUT_SCHEMA))
        )
        try:
            outcome = ZephyrContext(
                name="focus-crawl-pdf-extract-fleet",
                resources=_WORKER_RESOURCES,
                max_workers=_MAX_WORKERS,
                stage_runner_factory=SubprocessRunner,
                map_task_resources=_MAP_TASK_RESOURCES,
                heartbeat_timeout=_HEARTBEAT_TIMEOUT,
            ).execute(pipeline)
        finally:
            stop_stats.set()
        # A pool job that died partway through shows up as failed requests, not as an error, so the
        # corpus would be quietly short. Surface that -- but only when it actually cost something:
        # an instance exiting after the last shard landed is not a reason to lose a complete run.
        lost_documents = int(outcome.counters.get(f"{_COUNTER_PREFIX}/convert_request_failed", 0))
        try:
            session.check_alive()
        except Exception as error:
            if lost_documents:
                raise
            logger.warning("A converter pool job ended before the fleet was released: %s", error)
            logger.warning("No convert request failed, so the extracted corpus is complete.")

    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def fleet_extract_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    """The text extraction step, routed through the converter pool.

    Same name and deps as the in-task route's step; the ``transport`` hash attribute re-keys the
    output prefix so the two routes never share a directory.
    """
    return StepSpec(
        name="data/datakit/extract/common_crawl_focus_2026_22_pdf_text",
        deps=[source, classification, layout_model],
        hash_attrs={
            "table_backend": str(TABLE_BACKEND),
            # Adaptive: the backend is chosen per converter process from the arch it lands on, so
            # the attribute names the policy, not a single backend, and the per-document truth
            # lives in the layout_backend column.
            "layout_backend": f"{X86_LAYOUT_BACKEND}-on-x86/{ARM_LAYOUT_BACKEND}-on-arm",
            "picture_alpha_ratio": PICTURE_ALPHA_RATIO,
            "document_timeout": _DOCUMENT_TIMEOUT,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            "schema_version": 3,
            "transport": "converter-pool",
        },
        fn=remote(
            partial(
                fleet_extract,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
                layout_model_output_path=layout_model.output_path,
                needs_ocr=False,
            ),
            resources=_FLEET_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def fleet_backfill_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    """Docling conversion of the router's OCR route -- the complement of :func:`fleet_extract_step`.

    A one-off corpus, not a pipeline step: nothing downstream consumes it, and together with the
    ``pdf_text`` output it gives a docling conversion of the full classified sample -- the mirror
    of what ``extract_ocr_all`` produced for the OCR extractor. These are scanned-heavy documents
    by construction (the router sent them to OCR because embedded text was thin), so expect a
    large ``empty_text_filtered`` count; what survives is exactly what a text parser can still
    recover from them, which is the comparison the corpus exists to support.
    """
    return StepSpec(
        name="data/datakit/extract/common_crawl_focus_2026_22_pdf_docling_ocr_route",
        deps=[source, classification, layout_model],
        hash_attrs={
            "table_backend": str(TABLE_BACKEND),
            "layout_backend": f"{X86_LAYOUT_BACKEND}-on-x86/{ARM_LAYOUT_BACKEND}-on-arm",
            "picture_alpha_ratio": PICTURE_ALPHA_RATIO,
            "document_timeout": _DOCUMENT_TIMEOUT,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            "schema_version": 3,
            "transport": "converter-pool",
            # The route override is part of the corpus identity, so this can never collide with
            # the text route's cache.
            "routes": "ocr",
        },
        fn=remote(
            partial(
                fleet_extract,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
                layout_model_output_path=layout_model.output_path,
                needs_ocr=True,
            ),
            resources=_FLEET_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
