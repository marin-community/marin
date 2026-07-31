# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5, brokered: extract the text route through a persistent CPU converter fleet.

This is the same extraction as :mod:`experiments.build_pdf_source.extract` -- same options, same
boilerplate pass, same record shape -- with the converter moved out of the map task and behind the
broker. Senders read fetch shards sequentially and post one whole document per request; the
broker's lease queue balances documents across ~256 persistent converters, so a shard full of
3,000-page documents drains across the fleet instead of stalling the task that happened to read it.
The converter's build cost -- compiling the layout graph -- is paid once per converter process
instead of once per Zephyr shard.

The unit of work stays the whole document, not the page: cross-page paragraph merging and
boilerplate removal are document-scoped, so splitting a document across converters would change
the text that comes out.
"""

import logging
import urllib.parse
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache, partial

import httpx
import pyarrow as pa
from fray.types import ResourceConfig, create_environment
from marin.datakit.normalize import NormalizedData, generate_id, make_split_writer
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from marin.inference.config import BrokerConfig, InferenceProxyConfig, InferenceWorkerConfig
from marin.inference.converter_pool import ConverterPoolConfig, remote_converter_pool
from rigging.filesystem import prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
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
from experiments.build_pdf_source.docling_extract.service import (
    SOURCE_URL_HEADER,
    ConvertedDocument,
    build_handler,
    parse_converted,
)
from experiments.build_pdf_source.extract import (
    _DRIVER_RESOURCES,
    _OUTPUT_SCHEMA,
    _SOURCE_COLUMNS,
    BOILERPLATE_OPTIONS,
    DOCUMENT_TIMEOUT,
    LAYOUT_BACKEND,
    PICTURE_ALPHA_RATIO,
    TABLE_BACKEND,
    _keep_all,
)

logger = logging.getLogger(__name__)

MODEL_ID = "marin-docling-convert"

# The pool's operating point: 256 converter processes as 64 pods of 4. Docling's multicore scaling
# within a pod is suspected sublinear, so the alternative worth testing is 256 instances x 1
# process -- a two-constant edit here.
_POOL_INSTANCES = 64
_PROCESSES_PER_INSTANCE = 4
_CONVERTERS = _POOL_INSTANCES * _PROCESSES_PER_INSTANCE
# RAM follows extract.py's 7g-per-1cpu-task precedent: ~7g per converter process, plus headroom
# for the supervisor and the page tail.
_POOL_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="32g")

# Timeouts satisfy BrokerConfig's 0 < worker < lease < proxy invariant. Docling's own
# document_timeout=600 bounds a conversion, so the worker timeout is generous headroom over it; a
# segfaulted converter's lease is redelivered after 1020; the proxy at 1140 is the sender-visible
# bound on a document that keeps killing converters.
_WORKER_REQUEST_TIMEOUT = 900.0
_LEASE_TIMEOUT = 1020.0
_PROXY_REQUEST_TIMEOUT = 1140.0
# The first converter to answer /v1/models downloads the HF layout model before it can build.
_PROXY_READINESS_TIMEOUT = 1800.0
# Above what the senders offer (~512 in flight), so the proxy queues rather than rejects.
_MAX_PENDING_REQUESTS = 1024

# Sender sizing. Throughput is set by in-flight documents, so the target is two per converter:
# 2 x 256 = 512 = _SENDER_TASKS x _REQUEST_THREADS.
_REQUEST_THREADS = 16
_SENDER_TASKS = 32
_DOCUMENTS_IN_FLIGHT = 2 * _REQUEST_THREADS
_TASKS_PER_WORKER = 8
_MAX_WORKERS = _SENDER_TASKS // _TASKS_PER_WORKER

# A task is almost entirely blocked on the proxy, but it holds up to 2 x _REQUEST_THREADS whole
# PDFs in flight and tail PDFs run to tens of MB.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="6g", disk="4g")
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="48g", disk="32g")
# A task holding a long document can legitimately go a long time without finishing one.
_HEARTBEAT_TIMEOUT = 30 * 60

_COUNTER_PREFIX = "focus_crawl_pdf_fleet"

# Characters that must survive percent-encoding for the header to remain a readable URL; everything
# outside latin-1 is encoded, which is what header transport requires.
_HEADER_SAFE = ":/?#[]@!$&'()*+,;=%"


def build_pool_config(options: "ExtractionOptions") -> ConverterPoolConfig:  # noqa: F821
    """The converter fleet: what each converter runs, and the broker between it and the senders."""
    return ConverterPoolConfig(
        handler_factory=partial(build_handler, options),
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


def convert_batch(
    batch: pa.RecordBatch,
    keys: frozenset[tuple[str, int]],
    base_url: str,
    boilerplate: BoilerplateOptions,
) -> Iterator[dict]:
    """Convert the text-extractable documents in one Parquet row group through the pool.

    Submitting and waiting are overlapped: the next document is posted while earlier ones are still
    converting, bounded at :data:`_DOCUMENTS_IN_FLIGHT` outstanding so a task never holds more than
    twice its thread count of whole PDFs in memory. Records are yielded in input order.
    """
    pool = _request_pool(_REQUEST_THREADS)
    inflight: deque[tuple[dict, Future[ConvertedDocument]]] = deque()

    def resolve_oldest() -> Iterator[dict]:
        row, future = inflight.popleft()
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

    for row in batch.to_pylist():
        if (row["warc_filename"], row["warc_record_offset"]) not in keys:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/skipped_ocr_route", 1)
            continue
        while len(inflight) >= _DOCUMENTS_IN_FLIGHT:
            yield from resolve_oldest()
        inflight.append((row, pool.submit(convert_document, base_url, _REQUEST_THREADS, row)))

    while inflight:
        yield from resolve_oldest()


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
    }


def fleet_extract(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> NormalizedData:
    """Run the text route against a converter fleet held for exactly the duration of the run."""
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    options = ExtractionOptions(
        table_backend=TABLE_BACKEND,
        layout_backend=LAYOUT_BACKEND,
        layout_model_path=layout_model.model_path,
        layout_label_map=layout_model.label_map,
        picture_alpha_ratio=PICTURE_ALPHA_RATIO,
        document_timeout=DOCUMENT_TIMEOUT,
    )
    keys = routing_keys(classification.main_output_dir, needs_ocr=False)

    filesystem, path = url_to_fs(source.main_output_dir)
    num_shards = len(filesystem.glob(f"{path}/*.parquet"))
    if not num_shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")

    with remote_converter_pool(build_pool_config(options)) as session:
        logger.info("Converter pool ready at %s (%d converters)", session.endpoint.base_url, _CONVERTERS)
        pipeline = (
            Dataset.from_files(prefix_join(source.main_output_dir, "*.parquet"))
            .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
            .flat_map(
                partial(
                    convert_batch,
                    keys=keys,
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
            .map_shard(make_split_writer(output_path, output_schema=_OUTPUT_SCHEMA))
        )
        outcome = ZephyrContext(
            name="focus-crawl-pdf-extract-fleet",
            resources=_WORKER_RESOURCES,
            max_workers=_MAX_WORKERS,
            stage_runner_factory=SubprocessRunner,
            map_task_resources=_MAP_TASK_RESOURCES,
            heartbeat_timeout=_HEARTBEAT_TIMEOUT,
        ).execute(pipeline)
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
            "layout_backend": str(LAYOUT_BACKEND),
            "picture_alpha_ratio": PICTURE_ALPHA_RATIO,
            "document_timeout": DOCUMENT_TIMEOUT,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            "schema_version": 2,
            "transport": "converter-pool",
        },
        fn=remote(
            partial(
                fleet_extract,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
                layout_model_output_path=layout_model.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
