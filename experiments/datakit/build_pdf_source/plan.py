# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1: turn the crawl-wide columnar index into a sampled, byte-budgeted PDF fetch plan.

Reads the ten cc-index part files, keeps the untruncated 200-response PDFs, coalesces each WARC's
records into range GETs, samples ranges, and packs them into fixed-byte tasks. The output is one
Parquet row per range, ordered by task, plus a :class:`~experiments.datakit.build_pdf_source.common.PdfFetchPlan`
artifact carrying the totals the fetch step commits to.

The whole index is read even for a small sample because the table is sorted by ``url_surtkey``:
a WARC's records are scattered across all ten parts, so there is no way to coalesce one WARC
without seeing all of them.
"""

import logging
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import prefix_join
from zephyr.writers import write_parquet_file

from experiments.datakit.build_pdf_source.common import (
    DOWNLOAD_CHUNK_BYTES,
    FETCH_SUCCESS_STATUS,
    FOCUS_CRAWL,
    FOCUS_INDEX_DIR,
    FOCUS_INDEX_JOB_UUID,
    FOCUS_INDEX_PART_COUNT,
    FOCUS_WARC_FILE_COUNT,
    PDF_MIME_TYPE,
    REQUEST_TIMEOUT,
    USER_AGENT,
    FetchTask,
    PdfFetchPlan,
    RangeFetch,
    session,
)

logger = logging.getLogger(__name__)

# Coalesce gap inherited from the HTML extraction run. PDF records are sparser than HTML ones, but
# still dense enough that this collapses a WARC's ~686 kept PDF records into ~74 range GETs, at the
# cost of also transferring the non-PDF records in the gaps -- 5.4% of the fetched bytes. Ranges
# run p50 5.4 MiB / p90 33 / p99 82, with a single 294 MiB worst case across the whole crawl.
#
# There is deliberately no cap on coalesced range size. Capping at 64 MiB was measured to add 2.5%
# more range GETs (348,265 vs 339,856) while changing total fetched bytes by 0.00 TiB and leaving
# the packed task distribution identical -- TASK_BYTES already bounds what a worker takes on, so a
# range cap only splits ranges that packing would have kept together anyway.
COALESCE_GAP_BYTES = 1 << 20

# One task per this many fetched bytes. Sets both the shard count and the output file size, and
# decouples them from WARC size and range size -- which is what makes a sampled run's per-task
# profile match a full run's. It is the only bound on per-task work: a range wider than this
# becomes its own task rather than being split.
TASK_BYTES = 256 << 20

# Fraction of coalesced ranges to fetch. At 1.0 the plan keeps all 339,856 ranges -- ~4.1 TiB,
# ~3.16M PDFs across all 4,573 WARCs. The pipeline was validated end to end at 0.1 (33,986
# ranges, 411 GiB, 316,297 PDFs, 1,773 tasks); sampling ranges rather than records keeps every
# PDF's inclusion probability equal, so a sampled run's per-task profile matches a full run's.
SAMPLE_FRACTION = 1.0
SAMPLE_SEED = 20260729

_INDEX_COLUMNS = [
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "content_mime_detected",
    "fetch_status",
    "content_truncated",
]
_INDEX_BATCH_ROWS = 1 << 18

_PLAN_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.int32(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("range_start", pa.int64(), nullable=False),
        pa.field("range_end", pa.int64(), nullable=False),
        pa.field("record_offsets", pa.list_(pa.int64()), nullable=False),
    ]
)
PLAN_FILENAME = "plan.parquet"

# Holds the ten index parts on local disk while scanning them, and ~1 GB of record arrays.
_PLAN_RESOURCES = ResourceConfig(cpu=8, ram="32g", disk="32g")


@dataclass(frozen=True)
class IndexScan:
    """Every PDF record the index offers, keyed by position in a sorted WARC name list."""

    warc_filenames: list[str]
    """All WARC files in the crawl, sorted. Indexed by ``warc_ids``."""
    warc_ids: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray


def index_part_urls() -> list[str]:
    """Return the ten cc-index part URLs, which share one Spark job UUID."""
    return [
        f"{FOCUS_INDEX_DIR}/part-{part:05d}-{FOCUS_INDEX_JOB_UUID}.c000.gz.parquet"
        for part in range(FOCUS_INDEX_PART_COUNT)
    ]


def _download_index_part(url: str, destination: Path) -> Path:
    with session().get(url, headers={"user-agent": USER_AGENT}, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                stream.write(chunk)
    logger.info("Downloaded index part %s (%d bytes)", destination.name, destination.stat().st_size)
    return destination


def _pdf_mask(batch: pa.RecordBatch) -> pa.Array:
    """Untruncated 200 responses that Tika identified as PDFs.

    Truncated records are dropped here rather than downstream: they are 1.0% of PDF records but
    765 GB of bytes, and a truncated PDF is unusable for extraction.
    """
    is_pdf = pc.equal(batch.column("content_mime_detected"), PDF_MIME_TYPE)
    is_success = pc.equal(batch.column("fetch_status"), FETCH_SUCCESS_STATUS)
    is_whole = pc.is_null(batch.column("content_truncated"))
    return pc.fill_null(pc.and_(pc.and_(is_pdf, is_success), is_whole), False)


def scan_index(part_paths: list[Path]) -> IndexScan:
    """Read the index parts and return every PDF record, plus the crawl's WARC file list."""
    all_warcs: set[str] = set()
    warc_id_by_name: dict[str, int] = {}
    id_chunks: list[np.ndarray] = []
    offset_chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []

    for path in part_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=_INDEX_COLUMNS, batch_size=_INDEX_BATCH_ROWS):
            all_warcs.update(pc.unique(batch.column("warc_filename")).to_pylist())
            selected = batch.filter(_pdf_mask(batch))
            if selected.num_rows == 0:
                continue
            names = selected.column("warc_filename").to_pylist()
            id_chunks.append(
                np.fromiter(
                    (warc_id_by_name.setdefault(name, len(warc_id_by_name)) for name in names),
                    dtype=np.int32,
                    count=selected.num_rows,
                )
            )
            offset_chunks.append(selected.column("warc_record_offset").to_numpy(zero_copy_only=False))
            length_chunks.append(selected.column("warc_record_length").to_numpy(zero_copy_only=False))
        logger.info("Scanned %s: %d PDF records so far", path.name, sum(len(chunk) for chunk in id_chunks))

    if len(all_warcs) != FOCUS_WARC_FILE_COUNT:
        raise ValueError(f"Expected {FOCUS_WARC_FILE_COUNT} WARC files for {FOCUS_CRAWL}, found {len(all_warcs)}")

    # Re-key from first-seen order to sorted order, so the plan does not depend on how the index
    # parts happened to interleave.
    warc_filenames = sorted(all_warcs)
    rank = {name: index for index, name in enumerate(warc_filenames)}
    remap = np.fromiter((rank[name] for name in warc_id_by_name), dtype=np.int32, count=len(warc_id_by_name))
    return IndexScan(
        warc_filenames=warc_filenames,
        warc_ids=remap[np.concatenate(id_chunks)],
        offsets=np.concatenate(offset_chunks).astype(np.int64),
        lengths=np.concatenate(length_chunks).astype(np.int64),
    )


def coalesce_ranges(scan: IndexScan, *, gap_bytes: int) -> list[RangeFetch]:
    """Merge each WARC's PDF records into range GETs, in (WARC, offset) order.

    A record is folded into the open range when it starts within ``gap_bytes`` of the range's end.
    Ranges never span WARCs and are never bounded from above -- see ``COALESCE_GAP_BYTES`` for why
    a size cap here is not worth its cost.
    """
    order = np.lexsort((scan.offsets, scan.warc_ids))
    warc_ids = scan.warc_ids[order]
    starts = scan.offsets[order]
    stops = starts + scan.lengths[order]

    overlapping = (warc_ids[1:] == warc_ids[:-1]) & (starts[1:] < stops[:-1])
    if overlapping.any():
        first = int(np.flatnonzero(overlapping)[0]) + 1
        raise ValueError(
            f"Overlapping index records in {scan.warc_filenames[warc_ids[first]]} at offset {starts[first]}"
        )

    runs: list[tuple[int, int, int, list[int]]] = []
    open_warc, open_start, open_stop = -1, -1, -1
    members: list[int] = []

    for warc_id, start, stop in zip(warc_ids.tolist(), starts.tolist(), stops.tolist(), strict=True):
        if members and warc_id == open_warc and start - open_stop <= gap_bytes:
            open_stop = stop
            members.append(start)
            continue
        if members:
            runs.append((open_warc, open_start, open_stop, members))
        open_warc, open_start, open_stop, members = warc_id, start, stop, [start]

    if members:
        runs.append((open_warc, open_start, open_stop, members))

    return [
        RangeFetch(warc_filename=scan.warc_filenames[warc_id], start=start, stop=stop, record_offsets=tuple(offsets))
        for warc_id, start, stop, offsets in runs
    ]


def sample_ranges(ranges: list[RangeFetch], *, fraction: float, seed: int) -> list[RangeFetch]:
    """Take ``fraction`` of the coalesced ranges uniformly at random, preserving plan order."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return ranges
    count = round(len(ranges) * fraction)
    chosen = np.random.default_rng(seed).choice(len(ranges), size=count, replace=False)
    return [ranges[index] for index in sorted(chosen.tolist())]


def pack_tasks(ranges: list[RangeFetch], *, task_bytes: int) -> list[FetchTask]:
    """Group consecutive ranges into tasks of at most ``task_bytes``, one range minimum.

    Packing exists for scheduling economics, not locality: at a 10% sample it turns 33,986 shards
    into 1,773, and so 33,986 output files averaging 12 MiB into 1,773 averaging 244 MiB. Whether
    a task's ranges share a WARC (median 3 WARCs per task, max 57) does not affect connection
    reuse -- every range GET in the job addresses the same host, and the pooled session in
    ``common.session`` is per worker process, so its connections are reused across tasks anyway.
    """
    tasks: list[FetchTask] = []
    current: list[RangeFetch] = []
    current_bytes = 0
    for selected in ranges:
        if current and current_bytes + selected.size > task_bytes:
            tasks.append(FetchTask(task_id=len(tasks), ranges=tuple(current)))
            current, current_bytes = [], 0
        current.append(selected)
        current_bytes += selected.size
    if current:
        tasks.append(FetchTask(task_id=len(tasks), ranges=tuple(current)))
    return tasks


def _plan_rows(tasks: list[FetchTask]) -> Iterator[dict]:
    for task in tasks:
        for selected in task.ranges:
            yield {
                "task_id": task.task_id,
                "warc_filename": selected.warc_filename,
                "range_start": selected.start,
                "range_end": selected.stop,
                "record_offsets": list(selected.record_offsets),
            }


def write_plan(tasks: list[FetchTask], plan_path: str) -> None:
    """Write the packed tasks as one Parquet row per range, in task order.

    This file is the whole contract between the two steps:
    :func:`~experiments.datakit.build_pdf_source.fetch.read_fetch_tasks` reads it back.
    """
    write_parquet_file(_plan_rows(tasks), output_path=plan_path, schema=_PLAN_SCHEMA)


def build_fetch_plan(output_path: str) -> PdfFetchPlan:
    """Read the crawl-wide index and write the sampled, packed PDF fetch plan."""
    with tempfile.TemporaryDirectory(prefix="focus-crawl-index-") as staging:
        urls = index_part_urls()
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            part_paths = list(
                pool.map(
                    _download_index_part,
                    urls,
                    [Path(staging) / f"part-{part:05d}.parquet" for part in range(len(urls))],
                )
            )
        scan = scan_index(part_paths)

    logger.info("Index holds %d PDF records across %d WARCs", len(scan.offsets), len(scan.warc_filenames))
    ranges = coalesce_ranges(scan, gap_bytes=COALESCE_GAP_BYTES)
    logger.info("Coalesced into %d ranges (%.2f TiB)", len(ranges), sum(r.size for r in ranges) / (1 << 40))

    selected_ranges = sample_ranges(ranges, fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED)
    tasks = pack_tasks(selected_ranges, task_bytes=TASK_BYTES)
    fetch_bytes = sum(selected.size for selected in selected_ranges)
    num_pdfs = sum(len(selected.record_offsets) for selected in selected_ranges)
    logger.info(
        "Sampled %d/%d ranges: %d PDFs, %.1f GiB to fetch, %d tasks",
        len(selected_ranges),
        len(ranges),
        num_pdfs,
        fetch_bytes / (1 << 30),
        len(tasks),
    )

    plan_path = prefix_join(output_path, PLAN_FILENAME)
    write_plan(tasks, plan_path)
    return PdfFetchPlan(
        plan_path=plan_path,
        num_warcs=len({selected.warc_filename for selected in selected_ranges}),
        num_ranges=len(selected_ranges),
        num_pdfs=num_pdfs,
        fetch_bytes=fetch_bytes,
        num_tasks=len(tasks),
    )


def plan_step() -> StepSpec:
    """Build the focus-crawl PDF fetch-plan step."""
    return StepSpec(
        name="data/datakit/plan/common_crawl_focus_2026_22_pdf",
        hash_attrs={
            "crawl": FOCUS_CRAWL,
            "index_dir": FOCUS_INDEX_DIR,
            "index_job_uuid": FOCUS_INDEX_JOB_UUID,
            "index_part_count": FOCUS_INDEX_PART_COUNT,
            "warc_file_count": FOCUS_WARC_FILE_COUNT,
            "mime_type": PDF_MIME_TYPE,
            "coalesce_gap_bytes": COALESCE_GAP_BYTES,
            "task_bytes": TASK_BYTES,
            "sample_fraction": SAMPLE_FRACTION,
            "sample_seed": SAMPLE_SEED,
            "schema_version": 1,
        },
        fn=remote(build_fetch_plan, resources=_PLAN_RESOURCES, pip_dependency_groups=["datakit"]),
    )
