# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2: execute the fetch plan, writing raw PDF bytes with WARC provenance as Parquet.

One Zephyr shard per packed task, one output Parquet file per shard. A task issues its coalesced
range GETs in order, streams each into a temp file, and walks the WARC records inside it.

The output carries no ``id`` or ``text``: extraction (#7618) mints the text-derived id, and
``content_digest`` is the identity that survives an extractor swap in the meantime.
"""

import http.client
import logging
import re
import tempfile
from collections.abc import Iterator
from functools import partial
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.common import (
    COMMON_CRAWL_BASE_URL,
    DOWNLOAD_CHUNK_BYTES,
    FOCUS_CRAWL,
    PDF_MIME_TYPE,
    REQUEST_TIMEOUT,
    USER_AGENT,
    FetchTask,
    PdfFetchPlan,
    PdfSourceData,
    RangeFetch,
    session,
)

logger = logging.getLogger(__name__)

_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_MAX_DOWNLOAD_STALLS = 8

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("pdf", pa.binary(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_record_offset", pa.int64(), nullable=False),
        pa.field("warc_record_length", pa.int64(), nullable=False),
        pa.field("warc_date", pa.string(), nullable=False),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),
    ]
)

_DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
# Sized against cw-us-east-08a at plan time: four cd-gp-i64-erapids nodes (64 vCPU / 512 GB /
# 15.36 TB each), 256 vCPU in all. Seven cpu=8 workers per node -- 28, not the 32 that would pack
# the fleet exactly -- leaves the controller and system pods headroom.
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
# Fetching is I/O bound, so tasks are costed at one CPU and multiplex eight-deep per worker. Task
# disk holds one range's temp file at a time; the widest range in the whole crawl is 294 MiB.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="2g")
_MAX_WORKERS = 28
_HEARTBEAT_TIMEOUT = 15 * 60


def read_fetch_tasks(plan_path: str) -> list[FetchTask]:
    """Rebuild the packed tasks from the plan Parquet, which is written in task order."""
    with StoragePath(plan_path).open("rb") as stream:
        plan = pq.read_table(stream)

    grouped: dict[int, list[RangeFetch]] = {}
    for row in plan.to_pylist():
        grouped.setdefault(row["task_id"], []).append(
            RangeFetch(
                warc_filename=row["warc_filename"],
                start=row["range_start"],
                stop=row["range_end"],
                record_offsets=tuple(row["record_offsets"]),
            )
        )
    return [FetchTask(task_id=task_id, ranges=tuple(grouped[task_id])) for task_id in sorted(grouped)]


def _download_range(url: str, selected: RangeFetch, destination: BinaryIO) -> None:
    """Stream one byte range into ``destination``, resuming from wherever a broken transfer left off."""
    expected_bytes = selected.size
    stalls = 0
    while destination.tell() < expected_bytes:
        written = destination.tell()
        request_start = selected.start + written
        headers = {"Range": f"bytes={request_start}-{selected.stop - 1}", "user-agent": USER_AGENT}

        error: Exception | None = None
        try:
            with session().get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if response.status_code != http.client.PARTIAL_CONTENT or match is None:
                    raise RuntimeError(
                        f"WARC range {request_start}-{selected.stop - 1} did not return a valid partial response"
                    )
                response_start, response_stop, _ = (int(value) for value in match.groups())
                if response_start != request_start or response_stop != selected.stop - 1:
                    raise RuntimeError(
                        f"WARC range requested {request_start}-{selected.stop - 1}, "
                        f"received {response_start}-{response_stop}"
                    )
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if chunk:
                        destination.write(chunk)
        except (requests.exceptions.RequestException, http.client.IncompleteRead) as exc:
            error = exc

        if destination.tell() > written:
            stalls = 0
        else:
            stalls += 1
        if stalls > _MAX_DOWNLOAD_STALLS:
            raise RuntimeError(
                f"WARC range download stalled at {destination.tell()}/{expected_bytes} bytes after {stalls} attempts"
            ) from error
        if destination.tell() > expected_bytes:
            raise RuntimeError(f"WARC range download exceeded expected size {expected_bytes}")
        if error is not None:
            logger.warning(
                "WARC range download interrupted at %d/%d bytes; resuming: %s",
                destination.tell(),
                expected_bytes,
                error,
            )

    destination.seek(0)


def _declared_content_type(record) -> str:
    if record.http_headers is None:
        return ""
    return record.http_headers.get_header("Content-Type") or ""


def _is_pdf_record(record) -> bool:
    """Whether a WARC response record carries a PDF, by Tika's identification or the server's."""
    identified = record.rec_headers.get_header("WARC-Identified-Payload-Type") or ""
    declared = _declared_content_type(record)
    return PDF_MIME_TYPE in {value.partition(";")[0].strip().lower() for value in (identified, declared)}


def iter_planned_pdfs(stream: BinaryIO, selected: RangeFetch) -> Iterator[dict]:
    """Yield the planned PDF records from one downloaded range.

    A coalesced range spans the records between the PDFs it was built from, and those gaps hold
    PDFs the plan deliberately excluded (truncated payloads, non-200 responses). Selection is
    therefore by absolute record offset, not by MIME type -- the MIME check only avoids reading
    payloads we are certain to drop.
    """
    from warcio.archiveiterator import ArchiveIterator  # noqa: PLC0415

    wanted = set(selected.record_offsets)
    found: set[int] = set()

    records = ArchiveIterator(stream)
    for record in records:
        if record.rec_type != "response" or not _is_pdf_record(record):
            continue

        payload = record.content_stream().read()
        # Both offset and length become available only once the record has been read to its end.
        offset = selected.start + records.get_record_offset()
        if offset not in wanted:
            counters.pipeline.update_counter("focus_crawl_pdf/unplanned_pdf", 1)
            continue

        found.add(offset)
        counters.pipeline.update_counter("focus_crawl_pdf/documents", 1)
        counters.pipeline.update_counter("focus_crawl_pdf/pdf_bytes", len(payload))
        yield {
            "pdf": payload,
            "source_id": record.rec_headers.get_header("WARC-Record-ID") or "",
            "source": FOCUS_CRAWL,
            "url": record.rec_headers.get_header("WARC-Target-URI") or "",
            "warc_filename": selected.warc_filename,
            "warc_record_offset": offset,
            "warc_record_length": records.get_record_length(),
            "warc_date": record.rec_headers.get_header("WARC-Date") or "",
            "content_digest": record.rec_headers.get_header("WARC-Payload-Digest") or "",
            "content_type": _declared_content_type(record),
        }

    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError(
            f"{len(missing)} planned PDF records missing from {selected.warc_filename} "
            f"range {selected.start}-{selected.stop}: {missing[:8]}"
        )


def fetch_task_pdfs(task: FetchTask, base_url: str) -> Iterator[dict]:
    """Download one task's coalesced ranges and yield the planned PDF records from each."""
    counters.pipeline.update_counter("focus_crawl_pdf/range_requests", len(task.ranges))
    counters.pipeline.update_counter("focus_crawl_pdf/range_bytes", task.size)
    logger.info("Fetching task %d: %d ranges, %d bytes", task.task_id, len(task.ranges), task.size)
    for selected in task.ranges:
        url = f"{base_url.rstrip('/')}/{selected.warc_filename.lstrip('/')}"
        with tempfile.TemporaryFile(prefix="focus-crawl-pdf-", suffix=".warc.gz", dir=".") as warc_file:
            _download_range(url, selected, warc_file)
            yield from iter_planned_pdfs(warc_file, selected)


def fetch_planned_pdfs(output_path: str, plan_output_path: str) -> PdfSourceData:
    """Run the planned fetch and write raw PDF bytes to ``output_path``."""
    plan = read_artifact(plan_output_path, PdfFetchPlan)
    tasks = read_fetch_tasks(plan.plan_path)
    if len(tasks) != plan.num_tasks:
        raise ValueError(f"Plan declares {plan.num_tasks} tasks, {plan.plan_path} holds {len(tasks)}")
    logger.info("Fetching %d PDFs over %d ranges in %d tasks", plan.num_pdfs, plan.num_ranges, len(tasks))

    output_dir = prefix_join(output_path, "outputs/main")
    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(partial(fetch_task_pdfs, base_url=COMMON_CRAWL_BASE_URL))
        .write_parquet(
            prefix_join(output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_OUTPUT_SCHEMA,
            skip_existing=True,
        )
    )
    outcome = ZephyrContext(
        name="focus-crawl-pdf-fetch",
        resources=_WORKER_RESOURCES,
        max_workers=min(_MAX_WORKERS, len(tasks)),
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
    return PdfSourceData(main_output_dir=output_dir, counters=dict(outcome.counters))


def fetch_step(plan: StepSpec) -> StepSpec:
    """Build the focus-crawl PDF fetch step for a given plan step."""
    return StepSpec(
        name="data/datakit/raw/common_crawl_focus_2026_22_pdf",
        deps=[plan],
        hash_attrs={
            "base_url": COMMON_CRAWL_BASE_URL,
            "schema_version": 1,
        },
        fn=remote(
            partial(fetch_planned_pdfs, plan_output_path=plan.output_path),
            resources=_DRIVER_RESOURCES,
            # The fetch tasks walk WARC records with warcio at runtime; it lives in the ``pdf``
            # extra.
            pip_dependency_groups=["datakit", "pdf"],
        ),
    )
