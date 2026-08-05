# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a run's per-sample eval output for the dashboard sample browser.

A run's durable output is a finestore archive rooted at its ``results_path``: a ``samples`` table
(one row per evaluated question, the shared :class:`EvalSample` contract) plus a ``blobs`` table that
holds raw agent trajectories a sample references by a ``finestore://`` URI. This module discovers the
tasks, pages the samples for one task, and resolves a trajectory reference, returning typed responses.

Runs written before the archive existed still carry one ``samples_<task>_<ts>.parquet`` per (sub)task
and a ``gs://`` trajectory URI; the reader falls back to that layout when a run has no archive, so a
run browses correctly whether or not it has been migrated.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Generic, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.reader import CompositeReader
from fsspec.core import url_to_fs
from marin.evaluation.samples import (
    ARCHIVE_SAMPLES_TABLE,
    SAMPLES_PREFIX,
    SAMPLES_SUFFIX,
    EvalSample,
    primary_metric,
    sample_from_archive_row,
)
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)

CACHE_TTL = 120.0

# A lazily-loaded trajectory is capped so one request cannot pull an unbounded object into memory.
# Trajectories run tens of KB to a few MB; 16 MiB leaves headroom while refusing a pathological file.
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024

# The finestore reference scheme a migrated sample uses for its trajectory.
_ARCHIVE_URI_PREFIX = "finestore://"


class SampleTask(BaseModel):
    """One discovered task. ``files`` is the number of parquet objects backing its samples: on the
    archive path this is the run's whole ``samples``-table shard count (the same for every task); on
    the legacy path it is the count of that task's per-(sub)task parquet files."""

    model_config = ConfigDict(frozen=True)

    task: str
    files: int


class SampleTasksResponse(BaseModel):
    """Discovery result for the tasks available under one run."""

    model_config = ConfigDict(frozen=True)

    available: bool
    error: str | None
    tasks: tuple[SampleTask, ...]


class SampleCounts(BaseModel):
    """Unpaginated per-task outcome counts. ``ungraded`` (no primary metric, ``correct is None``) is
    tracked apart from ``incorrect`` so a sample that was never scored is not reported as a wrong answer."""

    model_config = ConfigDict(frozen=True)

    all: int
    correct: int
    incorrect: int
    ungraded: int


class SamplesResponse(BaseModel):
    """One page of validated samples plus task-level paging metadata."""

    model_config = ConfigDict(frozen=True)

    available: bool
    error: str | None
    task: str
    primary_metric: str | None
    metric_columns: tuple[str, ...]
    total: int
    offset: int
    limit: int
    counts: SampleCounts
    rows: tuple[EvalSample, ...]


class ArtifactResponse(BaseModel):
    """One sample-referenced artifact (the trajectory) resolved to text for the browser.

    ``available`` is False -- with a human-readable ``reason`` -- for every non-happy path (the run
    has no results directory, the reference is missing/unreadable, or it exceeds the size cap),
    mirroring the logs endpoint's reachability degradation rather than raising. ``text`` is the decoded
    object body when available, and ``media_type`` reflects the URI extension so the client knows
    whether to parse JSON.
    """

    model_config = ConfigDict(frozen=True)

    available: bool
    reason: str | None
    uri: str
    media_type: str
    size: int | None
    truncated: bool
    text: str | None


T = TypeVar("T")


class _TtlCache(Generic[T]):
    """A string-keyed cache whose entries expire ``ttl`` seconds after they are inserted."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._entries: dict[str, tuple[T, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            for expired in [k for k, (_, expires_at) in self._entries.items() if expires_at <= now]:
                self._entries.pop(expired)
            entry = self._entries.get(key)
            return entry[0] if entry is not None else None

    def put(self, key: str, value: T) -> None:
        with self._lock:
            self._entries[key] = (value, time.monotonic() + self._ttl)


_table_cache: _TtlCache[pa.Table] = _TtlCache(CACHE_TTL)
_artifact_cache: _TtlCache[ArtifactResponse] = _TtlCache(CACHE_TTL)
_discovery_cache: _TtlCache[tuple[int, tuple[str, ...]]] = _TtlCache(CACHE_TTL)


# --------------------------------------------------------------------------------------------------
# finestore archive access
# --------------------------------------------------------------------------------------------------


def _archive_discovery(results_path: str) -> tuple[int, tuple[str, ...]]:
    """The run's ``samples`` shard count and distinct task names; ``(0, ())`` means no archive.

    A finestore archive is live-appended, so this is TTL-cached rather than memoized with
    ``functools.cache``: the short window bounds repeated per-poll listings while still surfacing
    shards a later flush adds.
    """
    cached = _discovery_cache.get(results_path)
    if cached is not None:
        return cached
    reader = CompositeReader(results_path)
    shard_count = len(reader.list_shards(ARCHIVE_SAMPLES_TABLE))
    tasks: tuple[str, ...] = ()
    if shard_count:
        table = reader.scan(ARCHIVE_SAMPLES_TABLE, columns=["task"])
        if table is not None and table.num_rows:
            tasks = tuple(sorted({value for value in table.column("task").to_pylist() if value is not None}))
    result = (shard_count, tasks)
    _discovery_cache.put(results_path, result)
    return result


def _archive_task_table(results_path: str, task: str) -> pa.Table:
    """The deduplicated ``samples`` rows for one task, cached for the paging window."""
    key = f"archive|{results_path}|{task}"
    cached = _table_cache.get(key)
    if cached is not None:
        return cached
    table = CompositeReader(results_path).scan(ARCHIVE_SAMPLES_TABLE, where=[("task", "==", task)])
    if table is None:
        table = pa.table({})
    _table_cache.put(key, table)
    return table


# --------------------------------------------------------------------------------------------------
# legacy per-(sub)task parquet fallback (runs written before the archive)
# --------------------------------------------------------------------------------------------------


def _legacy_task(filename: str) -> str:
    """samples_<task>_<timestamp>.parquet -> <task> (the timestamp has no underscore)."""
    stem = filename[len(SAMPLES_PREFIX) : -len(SAMPLES_SUFFIX)]
    return stem.rsplit("_", 1)[0]


def _legacy_discover(results_path: str):
    """Return (fs, {task: [parquet_path, ...]}) for legacy sample files under results_path."""
    fs, root = url_to_fs(results_path)
    by_task: dict[str, list[str]] = {}
    for path in fs.find(root):
        name = path.rsplit("/", 1)[-1]
        if name.startswith(SAMPLES_PREFIX) and name.endswith(SAMPLES_SUFFIX):
            by_task.setdefault(_legacy_task(name), []).append(path)
    return fs, by_task


def _legacy_load_table(fs, paths: list[str]) -> pa.Table:
    """Read and cache the concatenated parquet table for one legacy task's sample files."""
    key = f"legacy|{'|'.join(sorted(paths))}"
    cached = _table_cache.get(key)
    if cached is not None:
        return cached
    tables = []
    for path in sorted(paths):
        with fs.open(path, "rb") as handle:
            tables.append(pq.read_table(handle))
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    _table_cache.put(key, table)
    return table


# --------------------------------------------------------------------------------------------------
# public API (the server's contract)
# --------------------------------------------------------------------------------------------------


def list_sample_tasks(results_path: str | None) -> SampleTasksResponse:
    """Discover the tasks with exported samples under a run's results directory."""
    if not results_path:
        return SampleTasksResponse(available=False, error="run has no results_path", tasks=())
    try:
        shard_count, archive_tasks = _archive_discovery(results_path)
        if shard_count:
            tasks = tuple(SampleTask(task=task, files=shard_count) for task in archive_tasks)
        else:
            _fs, by_task = _legacy_discover(results_path)
            tasks = tuple(SampleTask(task=task, files=len(paths)) for task, paths in sorted(by_task.items()))
    except Exception as exc:
        logger.info("sample discovery failed for %s: %s", results_path, exc)
        return SampleTasksResponse(available=False, error=f"{type(exc).__name__}: {exc}"[:400], tasks=())
    return SampleTasksResponse(available=True, error=None, tasks=tasks)


def _empty_samples(*, available: bool, error: str, task: str, offset: int, limit: int) -> SamplesResponse:
    return SamplesResponse(
        available=available,
        error=error,
        task=task,
        primary_metric=None,
        metric_columns=(),
        total=0,
        offset=offset,
        limit=limit,
        counts=SampleCounts(all=0, correct=0, incorrect=0, ungraded=0),
        rows=(),
    )


def _task_table(results_path: str, task: str) -> pa.Table | None:
    """The samples table for one task from the archive, or the legacy layout, or ``None`` if absent."""
    shard_count, _tasks = _archive_discovery(results_path)
    if shard_count:
        table = _archive_task_table(results_path, task)
        return table if table.num_rows else None
    _fs, by_task = _legacy_discover(results_path)
    paths = by_task.get(task)
    return _legacy_load_table(_fs, paths) if paths else None


def fetch_samples(
    results_path: str | None,
    task: str,
    *,
    offset: int,
    limit: int,
    correct: str,
) -> SamplesResponse:
    """Return one typed, correctness-filtered page of samples for a task."""
    if not results_path:
        return _empty_samples(available=False, error="run has no results_path", task=task, offset=offset, limit=limit)
    try:
        table = _task_table(results_path, task)
        if table is None:
            return _empty_samples(
                available=True, error=f"no samples for task {task!r}", task=task, offset=offset, limit=limit
            )
    except Exception as exc:
        logger.info("sample fetch failed for %s/%s: %s", results_path, task, exc)
        return _empty_samples(
            available=False, error=f"{type(exc).__name__}: {exc}"[:400], task=task, offset=offset, limit=limit
        )

    # Counts and the correctness filter need only the light ``correct`` and ``metrics`` columns, so
    # compute them straight from the arrow columns and validate just the page's rows. Validating every
    # row (and materializing every fat column) on each page request is what made this scale with the
    # run instead of the page.
    # ``metrics`` is a finestore ``map<string,double>`` in the archive (a struct in the legacy layout);
    # materialize map columns as dicts so a row is ``{name: value}`` either way. Non-map columns ignore
    # the flag.
    columns = set(table.column_names)
    correct_values = table.column("correct").to_pylist() if "correct" in columns else [None] * table.num_rows
    metric_maps = (
        table.column("metrics").to_pylist(maps_as_pydicts="strict") if "metrics" in columns else [None] * table.num_rows
    )
    metric_columns = tuple(sorted({name for row in metric_maps if row for name in row}))
    picked = primary_metric(dict.fromkeys(metric_columns, 0.0))
    primary = picked[0] if picked is not None else None
    n_correct = sum(1 for value in correct_values if value is True)
    n_ungraded = sum(1 for value in correct_values if value is None)
    counts = SampleCounts(
        all=table.num_rows,
        correct=n_correct,
        incorrect=table.num_rows - n_correct - n_ungraded,
        ungraded=n_ungraded,
    )
    if correct == "correct":
        indices = [i for i, value in enumerate(correct_values) if value is True]
    elif correct == "incorrect":
        indices = [i for i, value in enumerate(correct_values) if value is False]
    elif correct == "ungraded":
        indices = [i for i, value in enumerate(correct_values) if value is None]
    else:
        indices = list(range(table.num_rows))
    page_indices = indices[offset : offset + limit]
    page_rows = table.take(pa.array(page_indices, type=pa.int64())).to_pylist(maps_as_pydicts="strict")
    page = tuple(sample_from_archive_row(row) for row in page_rows)
    return SamplesResponse(
        available=True,
        error=None,
        task=task,
        primary_metric=primary,
        metric_columns=metric_columns,
        total=len(indices),
        offset=offset,
        limit=limit,
        counts=counts,
        rows=page,
    )


def _media_type(uri: str) -> str:
    """The artifact's media type inferred from its extension; only JSON is special-cased today."""
    return "application/json" if uri.endswith(".json") else "text/plain"


def _unavailable_artifact(
    uri: str, reason: str, *, size: int | None = None, truncated: bool = False
) -> ArtifactResponse:
    return ArtifactResponse(
        available=False,
        reason=reason,
        uri=uri,
        media_type=_media_type(uri),
        size=size,
        truncated=truncated,
        text=None,
    )


def _artifact_within_results(results_path: str, uri: str) -> bool:
    """True when a legacy ``uri`` sits under ``results_path`` with no upward traversal."""
    try:
        relative = StoragePath(uri).relative_to(StoragePath(results_path))
    except ValueError:
        return False
    return ".." not in relative.split("/")


def _resolve_archive_artifact(results_path: str, uri: str, max_bytes: int) -> ArtifactResponse:
    """Resolve a ``finestore://`` trajectory reference to decoded text from the run's blobs table."""
    try:
        raw = CompositeReader(results_path).resolve(uri)
    except Exception as exc:
        logger.info("archive artifact resolve failed for %s: %s", uri, exc)
        return _unavailable_artifact(uri, f"{type(exc).__name__}: {exc}"[:400])
    if raw is None:
        return _unavailable_artifact(uri, "artifact is not present in the archive")
    if len(raw) > max_bytes:
        return _unavailable_artifact(
            uri, f"artifact is {len(raw)} bytes; exceeds the {max_bytes}-byte cap", size=len(raw), truncated=True
        )
    return ArtifactResponse(
        available=True,
        reason=None,
        uri=uri,
        media_type=_media_type(uri),
        size=len(raw),
        truncated=False,
        text=raw.decode("utf-8", errors="replace"),
    )


def fetch_artifact(results_path: str | None, uri: str, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> ArtifactResponse:
    """Resolve one sample-referenced artifact URI to decoded text, size-capped.

    A ``finestore://`` reference resolves from the run's archive; a legacy ``gs://``/``s3://`` URI is
    resolved in place, path-restricted to under ``results_path``. Anything missing, out of tree, or
    oversized returns ``available=False`` with a reason instead of raising.
    """
    if not results_path:
        return _unavailable_artifact(uri, "run has no results_path")

    # A finestore:// URI is archive-relative, so two runs can share one (e.g. blobs/trial-1/...);
    # key the cache by the run as well so one run's artifact never answers for another's.
    cache_key = f"{results_path}|{uri}"
    cached = _artifact_cache.get(cache_key)
    if cached is not None:
        return cached

    if uri.startswith(_ARCHIVE_URI_PREFIX):
        response = _resolve_archive_artifact(results_path, uri, max_bytes)
        if response.available:
            _artifact_cache.put(cache_key, response)
        return response

    if not _artifact_within_results(results_path, uri):
        logger.warning("artifact fetch rejected: %s is not under %s", uri, results_path)
        return _unavailable_artifact(uri, "artifact URI is outside the run results directory")

    path = StoragePath(uri)
    try:
        size = path.size()
        if size > max_bytes:
            return _unavailable_artifact(
                uri, f"artifact is {size} bytes; exceeds the {max_bytes}-byte cap", size=size, truncated=True
            )
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except Exception as exc:
        logger.info("artifact fetch failed for %s: %s", uri, exc)
        return _unavailable_artifact(uri, f"{type(exc).__name__}: {exc}"[:400])

    if len(raw) > max_bytes:
        return _unavailable_artifact(uri, f"artifact exceeds the {max_bytes}-byte cap", size=size, truncated=True)

    response = ArtifactResponse(
        available=True,
        reason=None,
        uri=uri,
        media_type=_media_type(uri),
        size=size,
        truncated=False,
        text=raw.decode("utf-8", errors="replace"),
    )
    _artifact_cache.put(cache_key, response)
    return response
