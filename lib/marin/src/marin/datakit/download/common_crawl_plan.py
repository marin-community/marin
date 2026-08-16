# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover Common Crawl records and turn them into deterministic fetch tasks.

Discovery answers which records a caller selected. Planning answers how to transfer those records:
records are ordered by source, WARC, and offset; nearby records become one HTTP range; ranges are
optionally sampled per source; and consecutive ranges are packed into source-local byte-sized tasks.
The split keeps an expensive URL-index scan reusable when only transfer policy changes.
"""

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias
from urllib.parse import unquote, urlsplit

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join

from marin.datakit.download.common_crawl_warc import (
    COMMON_CRAWL_DATA_URL,
    MainIndexedRecord,
    SupplementalIndexedRecord,
    main_record_from_index_row,
    supplemental_record_from_index_row,
)

DEFAULT_COALESCE_GAP_BYTES = 1 << 20
DEFAULT_TASK_BYTES = 256 << 20
_MANIFEST_BATCH_ROWS = 1 << 16
DISCOVERY_FILENAME = "records.parquet"
PLAN_FILENAME = "plan.parquet"

JSONScalar: TypeAlias = str | int | float | bool | None
IndexedRecord: TypeAlias = MainIndexedRecord | SupplementalIndexedRecord


class CommonCrawlIndexKind(StrEnum):
    """Common Crawl index schema used to discover and verify records."""

    MAIN = "main"
    SUPPLEMENTAL = "supplemental"


@dataclass(frozen=True)
class CommonCrawlSource:
    """One Common Crawl snapshot and URL-index surface."""

    crawl_id: str
    index_kind: CommonCrawlIndexKind
    paths_manifest_url: str
    base_url: str = COMMON_CRAWL_DATA_URL
    subset: str = "warc"

    def __post_init__(self) -> None:
        for name in ("crawl_id", "paths_manifest_url", "base_url", "subset"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    @property
    def source_id(self) -> str:
        """Return a stable identifier derived from semantic source fields."""
        payload = json.dumps(
            {
                "base_url": self.base_url.rstrip("/"),
                "crawl_id": self.crawl_id,
                "index_kind": self.index_kind.value,
                "paths_manifest_url": self.paths_manifest_url,
                "subset": self.subset,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CommonCrawlSelection:
    """Caller-owned, JSON-compatible explanation for selecting one index row."""

    metadata: Mapping[str, JSONScalar]

    def __post_init__(self) -> None:
        json.dumps(dict(self.metadata), sort_keys=True)


class CommonCrawlRecordSelector(Protocol):
    """Serializable selection policy applied to normalized URL-index rows."""

    @property
    def identity(self) -> Mapping[str, object]: ...

    def select(self, row: Mapping[str, object]) -> CommonCrawlSelection | None: ...


@dataclass(frozen=True)
class CommonCrawlFilter:
    """Standard status, truncation, MIME, and URL-suffix record selector."""

    successful_responses: bool = True
    exclude_truncated: bool = True
    declared_mime_types: frozenset[str] = frozenset()
    detected_mime_types: frozenset[str] = frozenset()
    url_suffixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.declared_mime_types or self.detected_mime_types or self.url_suffixes):
            raise ValueError("at least one MIME type or URL suffix must be configured")
        if any(not suffix.startswith(".") for suffix in self.url_suffixes):
            raise ValueError("url_suffixes must start with '.'")

    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "successful_responses": self.successful_responses,
            "exclude_truncated": self.exclude_truncated,
            "declared_mime_types": sorted(self.declared_mime_types),
            "detected_mime_types": sorted(self.detected_mime_types),
            "url_suffixes": list(self.url_suffixes),
        }

    def select(self, row: Mapping[str, object]) -> CommonCrawlSelection | None:
        status = row.get("fetch_status")
        if self.successful_responses and (
            isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300
        ):
            return None
        if self.exclude_truncated and row.get("content_truncated") is not None:
            return None

        declared = _mime_type(row.get("content_mime_type")) in self.declared_mime_types
        detected = _mime_type(row.get("content_mime_detected")) in self.detected_mime_types
        url = row.get("url")
        suffix = isinstance(url, str) and any(
            unquote(urlsplit(url).path).lower().endswith(candidate.lower()) for candidate in self.url_suffixes
        )
        if not (declared or detected or suffix):
            return None
        return CommonCrawlSelection(
            metadata={"declared_mime": declared, "detected_mime": detected, "url_suffix": suffix}
        )


@dataclass(frozen=True)
class SelectedCommonCrawlRecord:
    """One normalized index record selected for a later fetch plan."""

    source: CommonCrawlSource
    indexed_record: IndexedRecord
    selection: CommonCrawlSelection


@dataclass(frozen=True)
class PlannedCommonCrawlRange:
    """One exact HTTP interval and its ordered expected WARC records."""

    source: CommonCrawlSource
    warc_filename: str
    start: int
    stop: int
    records: tuple[IndexedRecord, ...]

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("planned range must have non-negative start and stop > start")
        if not self.records:
            raise ValueError("planned range must contain records")

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CommonCrawlFetchTask:
    """A source-local scheduling envelope containing one or more planned ranges."""

    task_id: int
    source: CommonCrawlSource
    ranges: tuple[PlannedCommonCrawlRange, ...]

    @property
    def size(self) -> int:
        return sum(selected.size for selected in self.ranges)


class CommonCrawlSamplingMode(StrEnum):
    NONE = "none"
    PER_SOURCE_RANGE = "per_source_range"


@dataclass(frozen=True)
class CommonCrawlPlanOptions:
    """Semantic transfer-planning policy."""

    coalesce_gap_bytes: int = DEFAULT_COALESCE_GAP_BYTES
    task_bytes: int = DEFAULT_TASK_BYTES
    sampling_mode: CommonCrawlSamplingMode = CommonCrawlSamplingMode.NONE
    sample_fraction: float = 1.0
    sample_seed: int = 0

    def __post_init__(self) -> None:
        if self.coalesce_gap_bytes < 0:
            raise ValueError("coalesce_gap_bytes must be non-negative")
        if self.task_bytes <= 0:
            raise ValueError("task_bytes must be positive")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.sampling_mode is CommonCrawlSamplingMode.NONE and self.sample_fraction != 1.0:
            raise ValueError("sample_fraction must be 1.0 when sampling is disabled")


class CommonCrawlSourceSummary(BaseModel):
    crawl_id: str
    num_warcs: int
    num_records: int
    num_ranges: int
    fetch_bytes: int
    num_tasks: int


class CommonCrawlPlanSummary(BaseModel):
    manifest_path: str
    num_sources: int
    num_warcs: int
    num_records: int
    num_ranges: int
    fetch_bytes: int
    num_tasks: int
    sources: dict[str, CommonCrawlSourceSummary]


class CommonCrawlDiscoverySummary(BaseModel):
    manifest_path: str
    num_sources: int
    num_records: int


class CommonCrawlPlanError(ValueError):
    """Raised when selected records or a persisted plan are inconsistent."""


_MEMBER_TYPE = pa.struct(
    [
        pa.field("offset", pa.int64(), nullable=False),
        pa.field("length", pa.int64(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("warc_record_id", pa.string()),
    ]
)
DISCOVERY_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("crawl_id", pa.string(), nullable=False),
        pa.field("index_kind", pa.string(), nullable=False),
        pa.field("paths_manifest_url", pa.string(), nullable=False),
        pa.field("base_url", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("content_digest", pa.string(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("warc_record_offset", pa.int64(), nullable=False),
        pa.field("warc_record_length", pa.int64(), nullable=False),
        pa.field("warc_record_id", pa.string()),
        pa.field("selection_metadata", pa.string(), nullable=False),
    ]
)
PLAN_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.int64(), nullable=False),
        pa.field("range_index", pa.int32(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("crawl_id", pa.string(), nullable=False),
        pa.field("index_kind", pa.string(), nullable=False),
        pa.field("paths_manifest_url", pa.string(), nullable=False),
        pa.field("base_url", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("warc_filename", pa.string(), nullable=False),
        pa.field("range_start", pa.int64(), nullable=False),
        pa.field("range_end", pa.int64(), nullable=False),
        pa.field("records", pa.list_(_MEMBER_TYPE), nullable=False),
    ]
)


def discover_index_partition(
    index_partition: str,
    *,
    source: CommonCrawlSource,
    selector: CommonCrawlRecordSelector,
    batch_rows: int,
) -> Iterator[SelectedCommonCrawlRecord]:
    """Scan one index partition and yield selected records in source order."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    partition_url = f"{source.base_url.rstrip('/')}/{index_partition.lstrip('/')}"
    with fsspec.open(partition_url, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        columns = [
            "url",
            "fetch_status",
            "content_mime_type",
            "content_mime_detected",
            "content_digest",
            "content_truncated",
            "warc_filename",
            "warc_record_offset",
            "warc_record_length",
        ]
        if source.index_kind is CommonCrawlIndexKind.MAIN:
            columns.append("warc_record_id")
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Index partition is missing required columns: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_rows):
            for row in batch.to_pylist():
                selection = selector.select(row)
                if selection is None:
                    continue
                try:
                    indexed = (
                        main_record_from_index_row(row, crawl_id=source.crawl_id)
                        if source.index_kind is CommonCrawlIndexKind.MAIN
                        else supplemental_record_from_index_row(row, crawl_id=source.crawl_id)
                    )
                except ValueError:
                    continue
                yield SelectedCommonCrawlRecord(source=source, indexed_record=indexed, selection=selection)


def plan_common_crawl_records(
    records: Iterable[SelectedCommonCrawlRecord],
    options: CommonCrawlPlanOptions = CommonCrawlPlanOptions(),
) -> list[CommonCrawlFetchTask]:
    """Coalesce, optionally sample, and pack selected records into source-local tasks."""
    ordered = sorted(
        records,
        key=lambda selected: (
            selected.source.source_id,
            selected.indexed_record.record_range.warc_filename,
            selected.indexed_record.record_range.offset,
        ),
    )
    ranges = _coalesce_ranges(ordered, gap_bytes=options.coalesce_gap_bytes)
    ranges = _sample_ranges(ranges, options)
    return _pack_tasks(ranges, task_bytes=options.task_bytes)


def write_common_crawl_discovery(
    records: Iterable[SelectedCommonCrawlRecord], output_path: str
) -> CommonCrawlDiscoverySummary:
    """Stream normalized selected records to a reusable discovery manifest."""
    manifest_path = prefix_join(output_path, DISCOVERY_FILENAME)
    StoragePath(output_path).mkdirs()
    source_ids: set[str] = set()
    num_records = 0
    with StoragePath(manifest_path).open("wb") as stream:
        with pq.ParquetWriter(stream, DISCOVERY_SCHEMA) as writer:
            rows: list[dict[str, object]] = []
            for selected in records:
                source_ids.add(selected.source.source_id)
                num_records += 1
                rows.append(_selected_record(selected))
                if len(rows) == _MANIFEST_BATCH_ROWS:
                    writer.write_table(pa.Table.from_pylist(rows, schema=DISCOVERY_SCHEMA))
                    rows.clear()
            if rows or num_records == 0:
                writer.write_table(pa.Table.from_pylist(rows, schema=DISCOVERY_SCHEMA))
    return CommonCrawlDiscoverySummary(
        manifest_path=manifest_path,
        num_sources=len(source_ids),
        num_records=num_records,
    )


def read_common_crawl_discovery(manifest_path: str) -> Iterator[SelectedCommonCrawlRecord]:
    """Yield selected records from a persisted discovery manifest in manifest order."""
    with StoragePath(manifest_path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches():
            for row in batch.to_pylist():
                yield _selected_from_row(row)


def plan_common_crawl_manifest(
    discovery_path: str,
    output_path: str,
    options: CommonCrawlPlanOptions = CommonCrawlPlanOptions(),
) -> CommonCrawlPlanSummary:
    """Plan a persisted discovery manifest and write its range/task manifest."""
    tasks = plan_common_crawl_records(read_common_crawl_discovery(discovery_path), options)
    return write_common_crawl_plan(tasks, output_path)


def _coalesce_ranges(
    records: list[SelectedCommonCrawlRecord], *, gap_bytes: int
) -> list[PlannedCommonCrawlRange]:
    ranges: list[PlannedCommonCrawlRange] = []
    open_records: list[IndexedRecord] = []
    open_source: CommonCrawlSource | None = None
    open_warc = ""
    open_start = -1
    open_stop = -1

    def close_range() -> None:
        if open_source is not None:
            ranges.append(
                PlannedCommonCrawlRange(
                    source=open_source,
                    warc_filename=open_warc,
                    start=open_start,
                    stop=open_stop,
                    records=tuple(open_records),
                )
            )

    for selected in records:
        record_range = selected.indexed_record.record_range
        same_warc = selected.source.source_id == (open_source.source_id if open_source else None) and (
            record_range.warc_filename == open_warc
        )
        if same_warc and record_range.offset < open_stop:
            raise CommonCrawlPlanError(
                f"Overlapping records in {record_range.warc_filename} at offset {record_range.offset}"
            )
        if open_records and same_warc and record_range.offset - open_stop <= gap_bytes:
            open_stop = record_range.stop
            open_records.append(selected.indexed_record)
            continue
        close_range()
        open_source = selected.source
        open_warc = record_range.warc_filename
        open_start = record_range.offset
        open_stop = record_range.stop
        open_records = [selected.indexed_record]
    close_range()
    return ranges


def _sample_ranges(
    ranges: list[PlannedCommonCrawlRange], options: CommonCrawlPlanOptions
) -> list[PlannedCommonCrawlRange]:
    if options.sampling_mode is CommonCrawlSamplingMode.NONE or options.sample_fraction == 1.0:
        return ranges
    selected: list[PlannedCommonCrawlRange] = []
    start = 0
    while start < len(ranges):
        source_id = ranges[start].source.source_id
        stop = start + 1
        while stop < len(ranges) and ranges[stop].source.source_id == source_id:
            stop += 1
        source_ranges = ranges[start:stop]
        count = round(len(source_ranges) * options.sample_fraction)
        seed_bytes = hashlib.sha256(f"{options.sample_seed}:{source_id}".encode()).digest()[:8]
        chosen = np.random.default_rng(int.from_bytes(seed_bytes)).choice(len(source_ranges), size=count, replace=False)
        selected.extend(source_ranges[index] for index in sorted(chosen.tolist()))
        start = stop
    return selected


def _pack_tasks(ranges: list[PlannedCommonCrawlRange], *, task_bytes: int) -> list[CommonCrawlFetchTask]:
    tasks: list[CommonCrawlFetchTask] = []
    current: list[PlannedCommonCrawlRange] = []
    current_bytes = 0
    current_source: CommonCrawlSource | None = None
    for selected in ranges:
        source_changed = current_source is not None and current_source.source_id != selected.source.source_id
        if current and (source_changed or current_bytes + selected.size > task_bytes):
            assert current_source is not None
            tasks.append(CommonCrawlFetchTask(len(tasks), current_source, tuple(current)))
            current, current_bytes = [], 0
        current_source = selected.source
        current.append(selected)
        current_bytes += selected.size
    if current:
        assert current_source is not None
        tasks.append(CommonCrawlFetchTask(len(tasks), current_source, tuple(current)))
    return tasks


def write_common_crawl_plan(tasks: Iterable[CommonCrawlFetchTask], output_path: str) -> CommonCrawlPlanSummary:
    """Write one deterministic Parquet row per range and return aggregate totals."""
    task_list = list(tasks)
    rows = [row for task in task_list for row in _task_rows(task)]
    plan_path = prefix_join(output_path, PLAN_FILENAME)
    StoragePath(output_path).mkdirs()
    with StoragePath(plan_path).open("wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=PLAN_SCHEMA), stream)
    return _plan_summary(task_list, plan_path)


def read_common_crawl_tasks(plan_path: str) -> list[CommonCrawlFetchTask]:
    """Reconstruct tasks in manifest order, rejecting inconsistent rows."""
    with StoragePath(plan_path).open("rb") as stream:
        rows = pq.read_table(stream, schema=PLAN_SCHEMA).to_pylist()
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["task_id"]), []).append(row)
    if sorted(grouped) != list(range(len(grouped))):
        raise CommonCrawlPlanError("task IDs must be contiguous from zero")
    tasks = []
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        if [int(row["range_index"]) for row in task_rows] != list(range(len(task_rows))):
            raise CommonCrawlPlanError(f"range indexes for task {task_id} must be contiguous from zero")
        ranges = tuple(_range_from_row(row) for row in task_rows)
        source = ranges[0].source
        if any(selected.source != source for selected in ranges):
            raise CommonCrawlPlanError(f"task {task_id} spans sources")
        tasks.append(CommonCrawlFetchTask(task_id, source, ranges))
    return tasks


def _task_rows(task: CommonCrawlFetchTask) -> Iterator[dict[str, object]]:
    for range_index, selected in enumerate(task.ranges):
        if selected.source != task.source:
            raise CommonCrawlPlanError(f"task {task.task_id} contains a range from another source")
        yield {
            "task_id": task.task_id,
            "range_index": range_index,
            "source_id": task.source.source_id,
            "crawl_id": task.source.crawl_id,
            "index_kind": task.source.index_kind.value,
            "paths_manifest_url": task.source.paths_manifest_url,
            "base_url": task.source.base_url,
            "subset": task.source.subset,
            "warc_filename": selected.warc_filename,
            "range_start": selected.start,
            "range_end": selected.stop,
            "records": [_member_record(record) for record in selected.records],
        }


def _selected_record(selected: SelectedCommonCrawlRecord) -> dict[str, object]:
    source = selected.source
    indexed = selected.indexed_record
    location = indexed.record_range
    expectation = indexed.expectation
    return {
        "source_id": source.source_id,
        "crawl_id": source.crawl_id,
        "index_kind": source.index_kind.value,
        "paths_manifest_url": source.paths_manifest_url,
        "base_url": source.base_url,
        "subset": source.subset,
        "url": expectation.url,
        "content_digest": expectation.content_digest,
        "warc_filename": location.warc_filename,
        "warc_record_offset": location.offset,
        "warc_record_length": location.length,
        "warc_record_id": expectation.warc_record_id if isinstance(indexed, MainIndexedRecord) else None,
        "selection_metadata": json.dumps(dict(selected.selection.metadata), sort_keys=True, separators=(",", ":")),
    }


def _selected_from_row(row: Mapping[str, object]) -> SelectedCommonCrawlRecord:
    source = _source_from_row(row)
    indexed = _indexed_from_row(row, source)
    metadata = json.loads(str(row["selection_metadata"]))
    if not isinstance(metadata, dict):
        raise CommonCrawlPlanError("selection metadata must be a JSON object")
    return SelectedCommonCrawlRecord(source, indexed, CommonCrawlSelection(metadata))


def _member_record(record: IndexedRecord) -> dict[str, object]:
    expectation = record.expectation
    return {
        "offset": record.record_range.offset,
        "length": record.record_range.length,
        "url": expectation.url,
        "content_digest": expectation.content_digest,
        "warc_record_id": expectation.warc_record_id if isinstance(record, MainIndexedRecord) else None,
    }


def _range_from_row(row: Mapping[str, object]) -> PlannedCommonCrawlRange:
    source = _source_from_row(row)
    warc_filename = str(row["warc_filename"])
    members = row["records"]
    assert isinstance(members, list)
    indexed: list[IndexedRecord] = []
    for member in members:
        assert isinstance(member, dict)
        index_row = {
            "url": member["url"],
            "content_digest": member["content_digest"],
            "warc_filename": warc_filename,
            "warc_record_offset": member["offset"],
            "warc_record_length": member["length"],
            "warc_record_id": member["warc_record_id"],
        }
        indexed.append(
            main_record_from_index_row(index_row, crawl_id=source.crawl_id)
            if source.index_kind is CommonCrawlIndexKind.MAIN
            else supplemental_record_from_index_row(index_row, crawl_id=source.crawl_id)
        )
    return PlannedCommonCrawlRange(
        source=source,
        warc_filename=warc_filename,
        start=int(row["range_start"]),
        stop=int(row["range_end"]),
        records=tuple(indexed),
    )


def _source_from_row(row: Mapping[str, object]) -> CommonCrawlSource:
    source = CommonCrawlSource(
        crawl_id=str(row["crawl_id"]),
        index_kind=CommonCrawlIndexKind(str(row["index_kind"])),
        paths_manifest_url=str(row["paths_manifest_url"]),
        base_url=str(row["base_url"]),
        subset=str(row["subset"]),
    )
    if row["source_id"] != source.source_id:
        raise CommonCrawlPlanError("persisted source ID does not match source fields")
    return source


def _indexed_from_row(row: Mapping[str, object], source: CommonCrawlSource) -> IndexedRecord:
    index_row = {
        "url": row["url"],
        "content_digest": row["content_digest"],
        "warc_filename": row["warc_filename"],
        "warc_record_offset": row["warc_record_offset"],
        "warc_record_length": row["warc_record_length"],
        "warc_record_id": row.get("warc_record_id"),
    }
    return (
        main_record_from_index_row(index_row, crawl_id=source.crawl_id)
        if source.index_kind is CommonCrawlIndexKind.MAIN
        else supplemental_record_from_index_row(index_row, crawl_id=source.crawl_id)
    )


def _plan_summary(tasks: list[CommonCrawlFetchTask], plan_path: str) -> CommonCrawlPlanSummary:
    per_source: dict[str, CommonCrawlSourceSummary] = {}
    for source_id in dict.fromkeys(task.source.source_id for task in tasks):
        source_tasks = [task for task in tasks if task.source.source_id == source_id]
        ranges = [selected for task in source_tasks for selected in task.ranges]
        source = source_tasks[0].source
        per_source[source_id] = CommonCrawlSourceSummary(
            crawl_id=source.crawl_id,
            num_warcs=len({selected.warc_filename for selected in ranges}),
            num_records=sum(len(selected.records) for selected in ranges),
            num_ranges=len(ranges),
            fetch_bytes=sum(selected.size for selected in ranges),
            num_tasks=len(source_tasks),
        )
    return CommonCrawlPlanSummary(
        manifest_path=plan_path,
        num_sources=len(per_source),
        num_warcs=sum(summary.num_warcs for summary in per_source.values()),
        num_records=sum(summary.num_records for summary in per_source.values()),
        num_ranges=sum(summary.num_ranges for summary in per_source.values()),
        fetch_bytes=sum(summary.fetch_bytes for summary in per_source.values()),
        num_tasks=len(tasks),
        sources=per_source,
    )


def _mime_type(value: object) -> str:
    return value.partition(";")[0].strip().lower() if isinstance(value, str) else ""
